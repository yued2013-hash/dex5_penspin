#!/usr/bin/env python3
"""Guarded Dex5 S0.7 hardware acceptance and replay dry-run."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "Roll_21R", "Pitch_22R", "Pitch_23R", "Pitch_24R",
    "Roll_31R", "Pitch_32R", "Pitch_33R", "Pitch_34R",
    "Roll_41R", "Pitch_42R", "Pitch_43R", "Pitch_44R",
    "Roll_51R", "Pitch_52R", "Pitch_53R", "Pitch_54R",
    "Yaw_11R", "Roll_12R", "Pitch_13R", "Pitch_14R",
]

# Public URDF command channels grouped by the 16 physical actuators. The
# vendor anemone process remains responsible for nonlinear URDF-to-motor maps.
ACTUATOR_GROUPS = [
    (0,), (1,), (2, 3), (4,), (5,), (6, 7), (8,), (9,),
    (10, 11), (12,), (13,), (14, 15), (16,), (17,), (18,), (19,),
]


def replay_dry_run(package_dir: Path, count: int) -> dict:
    command_path = package_dir / "open_loop_commands_100hz.npz"
    safety_path = package_dir / "safety_limits.json"
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    with np.load(command_path, allow_pickle=False) as data:
        command = np.asarray(data["sdk_absolute_target_100hz"], dtype=np.float64)
        virtual = np.asarray(data["virtual_joint_target_100hz"], dtype=np.float64)
        logical = np.asarray(data["logical_actuator_target_100hz"], dtype=np.float64)
        offsets = np.asarray(data["trajectory_offsets"], dtype=np.int64)
        hz = float(data["command_hz"])

    count = min(count, len(offsets) - 1)
    lo = np.asarray(safety["absolute_joint_lower_rad"], dtype=np.float64)
    hi = np.asarray(safety["absolute_joint_upper_rad"], dtype=np.float64)
    rows = int(offsets[count])
    command = command[:rows]
    virtual = virtual[:rows]
    logical = logical[:rows]
    per_trajectory_delta = []
    for i in range(count):
        segment = command[offsets[i]:offsets[i + 1]]
        per_trajectory_delta.append(
            float(np.max(np.abs(np.diff(segment, axis=0)))) if len(segment) > 1 else 0.0
        )
    coupling_error = max(
        float(np.max(np.abs(virtual[:, a] - virtual[:, b])))
        for a, b in ((2, 3), (6, 7), (10, 11), (14, 15))
    )
    max_delta = max(per_trajectory_delta, default=0.0)
    result = {
        "status": "PASS",
        "trajectory_count": count,
        "command_rows": rows,
        "command_hz": hz,
        "sdk_shape": list(command.shape),
        "logical_shape": list(logical.shape),
        "finite": bool(np.isfinite(command).all()),
        "within_limits": bool(np.all((command >= lo) & (command <= hi))),
        "max_per_trajectory_delta_rad": max_delta,
        "max_per_trajectory_velocity_rad_s": max_delta * hz,
        "max_coupling_error_rad": coupling_error,
        "commands_published": 0,
    }
    if not all((result["finite"], result["within_limits"], coupling_error <= 1e-9)):
        result["status"] = "FAIL"
    return result


class HardwareAcceptance:
    def __init__(self, args: argparse.Namespace):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool, Float64MultiArray
        from unitree_dex5_1_control.msg import Dex51JointState

        self.rclpy = rclpy
        self.Bool = Bool
        self.Float64MultiArray = Float64MultiArray
        self.node = Node("dex5_s0_7_hardware_acceptance")
        self.position_pub = self.node.create_publisher(
            Float64MultiArray, "/dex5_1/set_position", 10
        )
        estop_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.estop_pub = self.node.create_publisher(
            Bool, "/dex5_1/emergency_stop", estop_qos
        )
        self.node.create_subscription(
            Dex51JointState, "/dex5_1/joint_state", self._on_state, 50
        )
        self.args = args
        self.q = None
        self.dq = None
        self.position_diff_velocity = np.zeros(20, dtype=np.float64)
        self.velocity_reference_q = None
        self.velocity_reference_time = 0.0
        self.feedback_monotonic = 0.0
        self.samples: list[tuple[float, np.ndarray, np.ndarray]] = []
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._signal_stop)
        signal.signal(signal.SIGTERM, self._signal_stop)

    def _signal_stop(self, _signum, _frame):
        self.stop_requested = True

    def _on_state(self, msg) -> None:
        q = np.asarray(msg.finger_positions, dtype=np.float64)
        dq = np.asarray(msg.finger_velocities, dtype=np.float64)
        if q.shape != (20,) or dq.shape != (20,) or not np.isfinite(q).all():
            return
        now = time.monotonic()
        if self.velocity_reference_q is None:
            self.velocity_reference_q = q.copy()
            self.velocity_reference_time = now
        elif now - self.velocity_reference_time >= 0.05:
            dt = now - self.velocity_reference_time
            self.position_diff_velocity = (q - self.velocity_reference_q) / dt
            self.velocity_reference_q = q.copy()
            self.velocity_reference_time = now
        self.q = q
        self.dq = dq
        self.feedback_monotonic = now
        self.samples.append((now, q.copy(), dq.copy()))

    def spin_for(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and not self.stop_requested:
            self.rclpy.spin_once(self.node, timeout_sec=0.005)

    def estop(self) -> None:
        msg = self.Bool()
        msg.data = True
        for _ in range(5):
            self.estop_pub.publish(msg)
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def clear_estop(self) -> None:
        msg = self.Bool()
        msg.data = False
        for _ in range(3):
            self.estop_pub.publish(msg)
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def publish_target(self, q: np.ndarray) -> None:
        if self.stop_requested:
            raise RuntimeError("stop requested")
        if self.feedback_monotonic and time.monotonic() - self.feedback_monotonic > 0.10:
            raise RuntimeError("feedback watchdog expired")
        if np.max(np.abs(self.position_diff_velocity)) > self.args.velocity_abort_rad_s:
            joint = int(np.argmax(np.abs(self.position_diff_velocity)))
            speed = float(abs(self.position_diff_velocity[joint]))
            raise RuntimeError(
                f"50ms position-difference velocity {speed:.4f} rad/s on joint {joint} "
                f"exceeded {self.args.velocity_abort_rad_s:.4f} rad/s"
            )
        limit_tolerance = 0.002
        if np.any(q < self.args.lower - limit_tolerance) or np.any(q > self.args.upper + limit_tolerance):
            raise RuntimeError("target outside URDF limits")
        msg = self.Float64MultiArray()
        msg.data = q.tolist()
        self.position_pub.publish(msg)

    def hold(self, target: np.ndarray, duration: float) -> None:
        deadline = time.monotonic() + duration
        period = 0.01
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.002)
            now = time.monotonic()
            if now >= next_tick:
                self.publish_target(target)
                next_tick += period

    def ramp(self, start: np.ndarray, target: np.ndarray) -> None:
        distance = float(np.max(np.abs(target - start)))
        steps = max(1, int(math.ceil(distance / (self.args.speed_rad_s * 0.01))))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            self.hold(start + alpha * (target - start), 0.01)

    def wait_for_feedback(self) -> None:
        deadline = time.monotonic() + 5.0
        while self.q is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.q is None:
            raise RuntimeError("no /dex5_1/joint_state feedback within 5 seconds")

    def test_group(self, motor_index: int, group: tuple[int, ...]) -> dict:
        baseline = self.q.copy()
        margins_hi = self.args.upper[list(group)] - baseline[list(group)]
        margins_lo = baseline[list(group)] - self.args.lower[list(group)]
        sign = 1.0 if float(np.min(margins_hi)) >= self.args.delta_rad + 0.01 else -1.0
        if sign < 0 and float(np.min(margins_lo)) < self.args.delta_rad + 0.01:
            raise RuntimeError(f"motor {motor_index}: insufficient limit margin")

        target = baseline.copy()
        target[list(group)] += sign * self.args.delta_rad
        self.samples.clear()
        command_time = time.monotonic()
        self.ramp(baseline, target)
        self.hold(target, self.args.response_window_s)
        response = list(self.samples)
        end = self.q.copy()
        self.ramp(end, baseline)
        self.hold(baseline, 0.20)

        times = np.asarray([x[0] for x in response])
        q_hist = np.asarray([x[1] for x in response])
        expected_delta = end[list(group)] - baseline[list(group)]
        other = [i for i in range(20) if i not in group]
        crosstalk = float(np.max(np.abs(end[other] - baseline[other])))
        dominant = int(np.argmax(np.abs(end - baseline)))
        primary = group[0]
        progress = sign * (q_hist[:, primary] - baseline[primary])
        threshold10 = 0.1 * self.args.delta_rad
        threshold90 = 0.9 * self.args.delta_rad
        latency_hits = np.flatnonzero(progress >= threshold10)
        rise_hits = np.flatnonzero(progress >= threshold90)
        latency = float(times[latency_hits[0]] - command_time) if latency_hits.size else None
        rise90 = float(times[rise_hits[0]] - command_time) if rise_hits.size else None
        peak = float(np.max(progress))
        overshoot = max(0.0, peak - self.args.delta_rad)
        steady_error = float(abs(self.args.delta_rad - progress[-1]))
        direction_pass = bool(sign * expected_delta[0] > self.args.direction_min_rad)
        if len(group) == 2:
            direction_pass = direction_pass and bool(sign * expected_delta[1] > 0.0)
        mapping_pass = dominant in group and crosstalk <= self.args.crosstalk_abort_rad
        result = {
            "motor_index": motor_index,
            "joint_indices": list(group),
            "joint_names": [JOINT_NAMES[i] for i in group],
            "command_delta_rad": sign * self.args.delta_rad,
            "measured_delta_rad": expected_delta.tolist(),
            "direction_pass": direction_pass,
            "dominant_feedback_joint": dominant,
            "crosstalk_rad": crosstalk,
            "mapping_pass": mapping_pass,
            "latency_s": latency,
            "rise_time_90_s": rise90,
            "overshoot_rad": overshoot,
            "steady_error_rad": steady_error,
        }
        if len(group) == 2:
            denom = expected_delta[0]
            result["pip_dip_coupling_ratio"] = (
                float(expected_delta[1] / denom) if abs(denom) > 1e-6 else None
            )
        return result

    def run(self) -> dict:
        self.estop()
        self.wait_for_feedback()
        initial = self.q.copy()
        self.clear_estop()
        results = []
        try:
            for motor_index in self.args.motor_indices:
                group = ACTUATOR_GROUPS[motor_index]
                results.append(self.test_group(motor_index, group))
        finally:
            self.estop()
        latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
        rises = [r["rise_time_90_s"] for r in results if r["rise_time_90_s"] is not None]
        return {
            "status": "PASS" if all(r["direction_pass"] and r["mapping_pass"] for r in results) else "FAIL",
            "initial_joint_position_rad": initial.tolist(),
            "motor_results": results,
            "direction_pass_count": sum(r["direction_pass"] for r in results),
            "mapping_pass_count": sum(r["mapping_pass"] for r in results),
            "latency_median_s": float(np.median(latencies)) if latencies else None,
            "latency_max_s": max(latencies) if latencies else None,
            "rise_time_90_median_s": float(np.median(rises)) if rises else None,
            "software_estop_sent": True,
        }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Send guarded commands to powered Dex5")
    parser.add_argument("--confirm-soft-pad", default="NO", choices=("NO", "YES"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("acceptance_results.json"))
    parser.add_argument("--replay-package", type=Path, default=root / "real_replay_package")
    parser.add_argument("--replay-count", type=int, default=10)
    parser.add_argument("--delta-rad", type=float, default=0.01)
    parser.add_argument("--speed-rad-s", type=float, default=0.10)
    parser.add_argument("--velocity-abort-rad-s", type=float, default=0.30)
    parser.add_argument("--crosstalk-abort-rad", type=float, default=0.006)
    parser.add_argument("--direction-min-rad", type=float, default=0.0015)
    parser.add_argument("--response-window-s", type=float, default=0.80)
    parser.add_argument(
        "--motor-indices",
        default="all",
        help="Comma-separated physical actuator indices (0-15), or 'all'",
    )
    args = parser.parse_args()
    safety = json.loads((args.replay_package / "safety_limits.json").read_text(encoding="utf-8"))
    args.lower = np.asarray(safety["absolute_joint_lower_rad"], dtype=np.float64)
    args.upper = np.asarray(safety["absolute_joint_upper_rad"], dtype=np.float64)
    if args.motor_indices == "all":
        args.motor_indices = list(range(len(ACTUATOR_GROUPS)))
    else:
        try:
            args.motor_indices = [int(x) for x in args.motor_indices.split(",")]
        except ValueError as exc:
            parser.error(f"invalid --motor-indices: {exc}")
        if not args.motor_indices or any(i < 0 or i >= len(ACTUATOR_GROUPS) for i in args.motor_indices):
            parser.error("--motor-indices values must be in 0..15")
    if args.delta_rad > 0.01 or args.speed_rad_s > 0.10:
        parser.error("initial hardware acceptance caps delta at 0.01 rad and speed at 0.10 rad/s")
    return args


def main() -> None:
    args = parse_args()
    result = {
        "schema": "dex5-s0.7-hardware-acceptance-v1",
        "timestamp_unix_s": time.time(),
        "replay_dry_run": replay_dry_run(args.replay_package, args.replay_count),
        "hardware": {"status": "NOT_RUN"},
    }
    if args.execute:
        if args.confirm_soft_pad != "YES":
            raise SystemExit("--execute requires --confirm-soft-pad YES")
        import rclpy
        rclpy.init()
        runner = HardwareAcceptance(args)
        try:
            result["hardware"] = runner.run()
        except Exception as exc:
            runner.estop()
            result["hardware"] = {"status": "ABORTED", "error": str(exc), "software_estop_sent": True}
        finally:
            runner.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
