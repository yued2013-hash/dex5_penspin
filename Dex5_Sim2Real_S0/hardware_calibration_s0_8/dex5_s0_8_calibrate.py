#!/usr/bin/env python3
"""Guarded Dex5 S0.8 actuator calibration.

This is a deterministic calibration sequence, not a policy or autonomous
trajectory. It identifies the public 20D URDF command interface grouped by
the 16 physical actuator channels exposed through the vendor adapter.
"""

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
ACTUATOR_GROUPS = [
    (0,), (1,), (2, 3), (4,), (5,), (6, 7), (8,), (9,),
    (10, 11), (12,), (13,), (14, 15), (16,), (17,), (18,), (19,),
]


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


class Calibrator:
    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool, Float64MultiArray
        from unitree_dex5_1_control.msg import Dex51JointState

        self.rclpy = rclpy
        self.Bool = Bool
        self.Float64MultiArray = Float64MultiArray
        self.node = Node("dex5_s0_8_real_calibration")
        self.position_pub = self.node.create_publisher(
            Float64MultiArray, "/dex5_1/set_position", 10
        )
        self.node.create_subscription(
            Dex51JointState, "/dex5_1/joint_state", self._on_state, 100
        )
        self.node.create_subscription(
            Float64MultiArray, "/dex5_1/applied_position", self._on_applied, 100
        )
        estop_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.estop_pub = self.node.create_publisher(
            Bool, "/dex5_1/emergency_stop", estop_qos
        )
        self.args = args
        self.q: np.ndarray | None = None
        self.dq: np.ndarray | None = None
        self.feedback_time = 0.0
        self.state_samples: list[tuple[float, np.ndarray]] = []
        self.applied_samples: list[tuple[float, np.ndarray]] = []
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, _signum, _frame) -> None:
        self.stop_requested = True

    def _on_state(self, msg) -> None:
        q = np.asarray(msg.finger_positions, dtype=np.float64)
        dq = np.asarray(msg.finger_velocities, dtype=np.float64)
        if q.shape != (20,) or dq.shape != (20,) or not np.isfinite(q).all():
            return
        now = time.monotonic()
        self.q = q
        self.dq = dq
        self.feedback_time = now
        self.state_samples.append((now, q.copy()))

    def _on_applied(self, msg) -> None:
        q = np.asarray(msg.data, dtype=np.float64)
        if q.shape == (20,) and np.isfinite(q).all():
            self.applied_samples.append((time.monotonic(), q.copy()))

    def spin(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and not self.stop_requested:
            self.rclpy.spin_once(self.node, timeout_sec=0.003)

    def estop(self, value: bool = True) -> None:
        msg = self.Bool()
        msg.data = value
        for _ in range(5 if value else 3):
            self.estop_pub.publish(msg)
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def wait_feedback(self) -> None:
        deadline = time.monotonic() + 5.0
        while self.q is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.q is None:
            raise RuntimeError("no /dex5_1/joint_state feedback within 5 seconds")

    def publish(self, target: np.ndarray) -> float:
        if self.stop_requested:
            raise RuntimeError("stop requested")
        if time.monotonic() - self.feedback_time > 0.10:
            raise RuntimeError("joint feedback watchdog expired")
        if np.any(target < self.args.lower - 0.002) or np.any(target > self.args.upper + 0.002):
            raise RuntimeError("target outside configured URDF limits")
        msg = self.Float64MultiArray()
        msg.data = target.tolist()
        stamp = time.monotonic()
        self.position_pub.publish(msg)
        return stamp

    def hold(self, target: np.ndarray, duration: float) -> None:
        deadline = time.monotonic() + duration
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.002)
            now = time.monotonic()
            if now >= next_tick:
                self.publish(target)
                next_tick += 0.01

    def ramp(self, start: np.ndarray, target: np.ndarray) -> None:
        distance = float(np.max(np.abs(target - start)))
        steps = max(1, int(math.ceil(distance / (self.args.speed_rad_s * 0.01))))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            self.hold(start + alpha * (target - start), 0.01)

    def tail_position(self, since: float, tail_s: float = 0.15) -> np.ndarray:
        rows = [(t, q) for t, q in self.state_samples if t >= since]
        if not rows:
            raise RuntimeError("no feedback samples in response window")
        end = rows[-1][0]
        tail = np.asarray([q for t, q in rows if t >= end - tail_s])
        return np.median(tail, axis=0)

    def choose_sign(self, baseline: np.ndarray, group: tuple[int, ...], delta: float) -> float:
        indices = list(group)
        plus = float(np.min(self.args.upper[indices] - baseline[indices]))
        minus = float(np.min(baseline[indices] - self.args.lower[indices]))
        if plus >= delta + 0.005:
            return 1.0
        if minus >= delta + 0.005:
            return -1.0
        raise RuntimeError(f"group {group} has insufficient limit margin for {delta:.4f} rad")

    def mapping_trial(self, actuator: int, group: tuple[int, ...]) -> dict:
        baseline = self.q.copy()
        delta = self.args.mapping_delta_rad
        indices = list(group)
        plus_ok = float(np.min(self.args.upper[indices] - baseline[indices])) >= delta + 0.003
        minus_ok = float(np.min(baseline[indices] - self.args.lower[indices])) >= delta + 0.003
        signs = [s for s, ok in ((1.0, plus_ok), (-1.0, minus_ok)) if ok]
        if not signs:
            raise RuntimeError(f"actuator {actuator}: no safe mapping direction")

        endpoints: dict[str, list[float]] = {}
        for sign in signs:
            target = baseline.copy()
            target[indices] += sign * delta
            started = time.monotonic()
            self.ramp(self.q.copy(), target)
            self.hold(target, self.args.mapping_hold_s)
            endpoints[str(int(sign))] = self.tail_position(started).tolist()
            self.ramp(self.q.copy(), baseline)
            self.hold(baseline, 0.25)

        if len(signs) == 2:
            response = (
                np.asarray(endpoints["1"], dtype=np.float64)
                - np.asarray(endpoints["-1"], dtype=np.float64)
            ) / 2.0
        else:
            sign = signs[0]
            response = sign * (np.asarray(endpoints[str(int(sign))]) - baseline)
        slope = response / delta
        primary = group[0]
        direction_sign = int(np.sign(slope[primary]))
        dominant = int(np.argmax(np.abs(slope)))
        outside = [i for i in range(20) if i not in group]
        crosstalk = float(np.max(np.abs(response[outside]))) if outside else 0.0
        result = {
            "actuator_index": actuator,
            "joint_indices": indices,
            "joint_names": [JOINT_NAMES[i] for i in indices],
            "tested_signs": signs,
            "command_delta_rad": delta,
            "response_vector_rad": response.tolist(),
            "local_gain_vector": slope.tolist(),
            "primary_local_gain": float(slope[primary]),
            "direction_sign": direction_sign,
            "direction_confirmed": bool(abs(response[primary]) >= self.args.direction_min_rad),
            "dominant_feedback_joint": dominant,
            "mapping_confirmed": bool(dominant in group),
            "crosstalk_rad": crosstalk,
            "endpoints": endpoints,
        }
        if len(group) == 2:
            result["pip_dip_coupling_ratio"] = (
                float(response[group[1]] / response[group[0]])
                if abs(response[group[0]]) > 1.0e-6 else None
            )
        return result

    def dynamic_trial(self, actuator: int, group: tuple[int, ...]) -> dict:
        baseline = self.q.copy()
        sign = self.choose_sign(baseline, group, self.args.dynamic_delta_rad)
        target = baseline.copy()
        target[list(group)] += sign * self.args.dynamic_delta_rad
        self.state_samples.clear()
        self.applied_samples.clear()
        command_time = self.publish(target)
        self.hold(target, self.args.dynamic_window_s)
        end_time = time.monotonic()
        samples = [(t, q) for t, q in self.state_samples if command_time <= t <= end_time]
        applied = [(t, q) for t, q in self.applied_samples if t >= command_time]
        if not samples:
            raise RuntimeError(f"actuator {actuator}: no dynamic feedback")
        times = np.asarray([t for t, _ in samples])
        hist = np.asarray([q for _, q in samples])
        primary = group[0]
        progress = sign * (hist[:, primary] - baseline[primary])
        achieved = max(float(np.max(progress)), 0.0)
        noise_floor = max(self.args.onset_min_rad, 0.1 * achieved)
        onset = np.flatnonzero(progress >= noise_floor)
        threshold63 = 0.632 * achieved
        threshold90 = 0.9 * achieved
        hit63 = np.flatnonzero(progress >= threshold63) if achieved > 0 else np.asarray([])
        hit90 = np.flatnonzero(progress >= threshold90) if achieved > 0 else np.asarray([])
        applied_latency = None
        for stamp, value in applied:
            if np.max(np.abs(value - target)) <= 1.0e-6:
                applied_latency = stamp - command_time
                break
        response_latency = float(times[onset[0]] - command_time) if onset.size else None
        t63 = float(times[hit63[0]] - command_time) if hit63.size else None
        rise90 = float(times[hit90[0]] - command_time) if hit90.size else None
        steady = np.median(hist[max(0, len(hist) - 10):], axis=0)
        self.ramp(self.q.copy(), baseline)
        self.hold(baseline, 0.25)
        return {
            "actuator_index": actuator,
            "joint_indices": list(group),
            "command_delta_rad": sign * self.args.dynamic_delta_rad,
            "controller_accept_latency_s": applied_latency,
            "physical_onset_latency_s": response_latency,
            "time_to_63_s": t63,
            "rise_time_90_s": rise90,
            "achieved_primary_delta_rad": achieved,
            "steady_response_vector_rad": (steady - baseline).tolist(),
        }

    def run(self) -> dict:
        self.estop(True)
        self.wait_feedback()
        initial = self.q.copy()
        self.estop(False)
        mappings: list[dict] = []
        dynamics: list[dict] = []
        try:
            for actuator in self.args.actuators:
                group = ACTUATOR_GROUPS[actuator]
                mappings.append(self.mapping_trial(actuator, group))
                if not self.args.mapping_only:
                    dynamics.append(self.dynamic_trial(actuator, group))
        finally:
            self.estop(True)
        accept = [x["controller_accept_latency_s"] for x in dynamics if x["controller_accept_latency_s"] is not None]
        onset = [x["physical_onset_latency_s"] for x in dynamics if x["physical_onset_latency_s"] is not None]
        t63 = [x["time_to_63_s"] for x in dynamics if x["time_to_63_s"] is not None]
        return {
            "status": "PASS" if all(x["direction_confirmed"] and x["mapping_confirmed"] for x in mappings) else "PARTIAL",
            "initial_joint_position_rad": initial.tolist(),
            "mapping": mappings,
            "dynamics": dynamics,
            "direction_confirmed_count": sum(x["direction_confirmed"] for x in mappings),
            "mapping_confirmed_count": sum(x["mapping_confirmed"] for x in mappings),
            "latency_distribution": {
                "controller_accept_s": {"count": len(accept), "p05": percentile(accept, 5), "p50": percentile(accept, 50), "p95": percentile(accept, 95), "max": max(accept) if accept else None},
                "physical_onset_s": {"count": len(onset), "p05": percentile(onset, 5), "p50": percentile(onset, 50), "p95": percentile(onset, 95), "max": max(onset) if onset else None},
                "time_to_63_s": {"count": len(t63), "p05": percentile(t63, 5), "p50": percentile(t63, 50), "p95": percentile(t63, 95), "max": max(t63) if t63 else None},
            },
            "software_estop_sent": True,
        }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-soft-pad", choices=("NO", "YES"), default="NO")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("raw_calibration.json"))
    parser.add_argument("--replay-package", type=Path, default=root / "real_replay_package")
    parser.add_argument("--mapping-delta-rad", type=float, default=0.02)
    parser.add_argument("--dynamic-delta-rad", type=float, default=0.008)
    parser.add_argument("--speed-rad-s", type=float, default=0.05)
    parser.add_argument("--mapping-hold-s", type=float, default=0.45)
    parser.add_argument("--dynamic-window-s", type=float, default=0.55)
    parser.add_argument("--direction-min-rad", type=float, default=0.0015)
    parser.add_argument("--onset-min-rad", type=float, default=0.0005)
    parser.add_argument(
        "--actuators",
        default="0-15",
        help="comma-separated actuator indices or the inclusive range 0-15",
    )
    parser.add_argument("--mapping-only", action="store_true")
    args = parser.parse_args()
    safety = json.loads((args.replay_package / "safety_limits.json").read_text(encoding="utf-8"))
    args.lower = np.asarray(safety["absolute_joint_lower_rad"], dtype=np.float64)
    args.upper = np.asarray(safety["absolute_joint_upper_rad"], dtype=np.float64)
    if "-" in args.actuators and "," not in args.actuators:
        first, last = (int(x) for x in args.actuators.split("-", 1))
        args.actuators = list(range(first, last + 1))
    else:
        args.actuators = [int(x) for x in args.actuators.split(",") if x.strip()]
    if not args.actuators or any(x < 0 or x >= len(ACTUATOR_GROUPS) for x in args.actuators):
        parser.error("actuator indices must be in 0..15")
    thumb_only = all(x >= 12 for x in args.actuators)
    mapping_cap = 0.05 if thumb_only and args.mapping_only else 0.02
    if args.mapping_delta_rad > mapping_cap or args.dynamic_delta_rad > 0.008 or args.speed_rad_s > 0.05:
        parser.error(
            f"S0.8 caps mapping delta at {mapping_cap:.2f} rad for this mode, "
            "dynamic step at 0.008 rad, and speed at 0.05 rad/s"
        )
    return args


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise SystemExit("hardware calibration requires --execute")
    if args.confirm_soft_pad != "YES":
        raise SystemExit("--execute requires --confirm-soft-pad YES")
    import rclpy
    rclpy.init()
    runner = Calibrator(args)
    result = {"schema": "dex5-s0.8-real-calibration-v1", "timestamp_unix_s": time.time()}
    try:
        result["hardware"] = runner.run()
    except Exception as exc:
        runner.estop(True)
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
