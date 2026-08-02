#!/usr/bin/env python3
"""Guarded open-loop replay for Dex5 S0.9.

The program never evaluates a policy. It publishes a precomputed, speed-safe
20D URDF-joint trajectory through the public Dex5 ROS interface. Hardware
execution is opt-in and interactive by design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import signal
import time
from pathlib import Path

import numpy as np


EXPECTED_REPLAY_SHA256 = "dfcc88f356fe63815b527bde4b38db0a948e3a7a8444fbcc2942989565725108"
DIAGNOSTIC_MANIFEST_SCHEMA = "dex5-s0.9-slow-diagnostic-v1"
TEMP_RE = re.compile(r"temperature\[\d+\]:\s*(-?\d+)\s*°?C", re.IGNORECASE)
INDEPENDENT_FEEDBACK_JOINTS = np.asarray(
    [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19], dtype=np.int64
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if values.size else None


def interpolate_branch_command(branch: dict, raw_feedback: float) -> float:
    raw = np.asarray(branch["raw_unwrapped_feedback_rad"], dtype=np.float64)
    command = np.asarray(branch["command_rad"], dtype=np.float64)
    candidates = raw_feedback + 2.0 * math.pi * np.arange(-2, 3, dtype=np.float64)
    unwrapped = float(candidates[np.argmin(np.abs(candidates - np.median(raw)))])
    order = np.argsort(raw)
    return float(np.interp(unwrapped, raw[order], command[order]))


def mapped_resume_target(
    replay_target: np.ndarray,
    current_feedback: np.ndarray,
    expected_feedback: np.ndarray,
    mapping: dict,
    offset_limit: float,
) -> tuple[np.ndarray, dict]:
    """Estimate a low-jump command for the current physical P3 posture."""
    resume = replay_target.copy()

    # Non-thumb MCP/abduction channels are locally direct. PIP/DIP pairs share
    # an actuator, so apply the measured PIP displacement to both virtual rows.
    for base in (0, 4, 8, 12):
        for joint in (base, base + 1):
            delta = float(current_feedback[joint] - expected_feedback[joint])
            resume[joint] += float(np.clip(delta, -offset_limit, offset_limit))
        pip_delta = float(current_feedback[base + 2] - expected_feedback[base + 2])
        pip_delta = float(np.clip(pip_delta, -offset_limit, offset_limit))
        resume[base + 2] += pip_delta
        resume[base + 3] += pip_delta

    thumb_offsets = {}
    channels = {
        int(row["dominant_feedback_joint"]): row
        for row in mapping["analysis"]["channels"]
    }
    for joint in range(16, 20):
        channel = channels[joint]
        branch_name = mapping["analysis"]["inverse_p3_estimate"][str(joint)][
            "preferred_motion_branch"
        ]
        branch = channel["branch_models"][branch_name]
        current_command = interpolate_branch_command(branch, current_feedback[joint])
        expected_command = interpolate_branch_command(branch, expected_feedback[joint])
        delta = float(np.clip(current_command - expected_command, -offset_limit, offset_limit))
        resume[joint] += delta
        thumb_offsets[str(joint)] = {
            "branch": branch_name,
            "raw_feedback_delta_rad": float(
                current_feedback[joint] - expected_feedback[joint]
            ),
            "resume_command_offset_rad": delta,
        }
    return resume, thumb_offsets


def load_package(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    replay_hash = sha256(args.commands)
    if replay_hash != args.expected_replay_sha256:
        raise RuntimeError(
            "replay hash does not match --expected-replay-sha256: "
            f"expected={args.expected_replay_sha256}, actual={replay_hash}"
        )
    if replay_hash != EXPECTED_REPLAY_SHA256:
        if not args.diagnostic_package:
            raise RuntimeError(
                "an alternate replay package requires --diagnostic-package"
            )
        if manifest.get("schema") != DIAGNOSTIC_MANIFEST_SCHEMA:
            raise RuntimeError(
                f"diagnostic manifest schema must be {DIAGNOSTIC_MANIFEST_SCHEMA}"
            )
    if manifest.get("output_sha256") != replay_hash:
        raise RuntimeError("mapped replay manifest hash does not match command file")
    with np.load(args.commands, allow_pickle=False) as data:
        command = np.asarray(data["sdk_absolute_target_100hz"], dtype=np.float64)
        offsets = np.asarray(data["trajectory_offsets"], dtype=np.int64)
        hz = float(data["command_hz"])
    if command.ndim != 2 or command.shape[1] != 20:
        raise RuntimeError(f"expected Nx20 SDK targets, got {command.shape}")
    if offsets[0] != 0 or offsets[-1] != len(command):
        raise RuntimeError("invalid trajectory offsets")
    if not np.isfinite(command).all() or not np.all(np.diff(offsets) > 0):
        raise RuntimeError("non-finite commands or invalid offsets")
    if not math.isclose(hz, 100.0, abs_tol=1.0e-6):
        raise RuntimeError(f"expected 100 Hz replay, got {hz}")
    lo = np.asarray(args.lower, dtype=np.float64)
    hi = np.asarray(args.upper, dtype=np.float64)
    if not np.all((command >= lo) & (command <= hi)):
        raise RuntimeError("replay contains a target outside the audited URDF limits")
    max_delta = 0.0
    for start, end in zip(offsets[:-1], offsets[1:]):
        segment = command[start:end]
        if len(segment) > 1:
            max_delta = max(max_delta, float(np.max(np.abs(np.diff(segment, axis=0)))))
    max_speed = max_delta * hz
    if max_delta > args.delta_limit_rad + 1.0e-9 or max_speed > args.speed_limit_rad_s + 1.0e-9:
        raise RuntimeError(
            f"replay safety gate failed: delta={max_delta:.6f}, speed={max_speed:.6f}"
        )
    return {
        "command": command,
        "offsets": offsets,
        "hz": hz,
        "sha256": replay_hash,
        "max_delta": max_delta,
        "max_speed": max_speed,
    }


class PoweredReplay:
    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from rcl_interfaces.msg import Log
        from rcl_interfaces.msg import ParameterType
        from rclpy.parameter_client import AsyncParameterClient
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool, Float64MultiArray
        from unitree_dex5_1_control.msg import (
            Dex51FingertipForces,
            Dex51JointState,
            Dex51TouchData,
        )

        self.rclpy = rclpy
        self.Bool = Bool
        self.Float64MultiArray = Float64MultiArray
        self.node = Node("dex5_s0_9_powered_replay")
        self.position_pub = self.node.create_publisher(Float64MultiArray, "/dex5_1/set_position", 10)
        self.hybrid_pub = self.node.create_publisher(Float64MultiArray, "/dex5_1/set_hybrid", 10)
        estop_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.estop_pub = self.node.create_publisher(Bool, "/dex5_1/emergency_stop", estop_qos)
        self.node.create_subscription(Dex51JointState, "/dex5_1/joint_state", self._on_state, 100)
        self.node.create_subscription(Float64MultiArray, "/dex5_1/applied_position", self._on_applied, 100)
        self.node.create_subscription(Dex51TouchData, "/dex5_1/touch_data", self._on_touch, 100)
        self.node.create_subscription(
            Dex51FingertipForces, "/dex5_1/fingertip_forces", self._on_force, 100
        )
        self.node.create_subscription(Log, "/rosout", self._on_log, 100)
        self.gain_client = AsyncParameterClient(self.node, "/dex5_1_controller")
        self.ParameterType = ParameterType
        self.args = args
        self.q: np.ndarray | None = None
        self.dq: np.ndarray | None = None
        self.position_diff_velocity = np.zeros(20, dtype=np.float64)
        self.position_diff_velocity_generation = 0
        self.last_checked_velocity_generation = -1
        self.sustained_feedback_speed_violations = 0
        self.velocity_reference_q: np.ndarray | None = None
        self.velocity_reference_time = 0.0
        self.feedback_time = 0.0
        self.latest_touch = np.empty(0, dtype=np.float64)
        self.latest_force = np.empty(0, dtype=np.float64)
        self.temperature_samples: list[dict] = []
        self.applied_samples: list[tuple[float, np.ndarray]] = []
        self.active_samples: list[dict] = []
        self.active_trajectory_id: int | None = None
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._signal_stop)
        signal.signal(signal.SIGTERM, self._signal_stop)

    def _signal_stop(self, _signum, _frame) -> None:
        self.stop_requested = True

    def _on_state(self, msg) -> None:
        now = time.monotonic()
        q = np.asarray(msg.finger_positions, dtype=np.float64)
        dq = np.asarray(msg.finger_velocities, dtype=np.float64)
        if q.shape != (20,) or dq.shape != (20,) or not np.isfinite(q).all():
            return
        if self.velocity_reference_q is None:
            self.velocity_reference_q = q.copy()
            self.velocity_reference_time = now
        elif now - self.velocity_reference_time >= 0.05:
            dt = now - self.velocity_reference_time
            self.position_diff_velocity = (q - self.velocity_reference_q) / dt
            self.position_diff_velocity_generation += 1
            self.velocity_reference_q = q.copy()
            self.velocity_reference_time = now
        self.q = q
        self.dq = dq
        self.feedback_time = now

    def _on_applied(self, msg) -> None:
        value = np.asarray(msg.data, dtype=np.float64)
        if value.shape == (20,) and np.isfinite(value).all():
            self.applied_samples.append((time.monotonic(), value.copy()))

    def _on_touch(self, msg) -> None:
        self.latest_touch = np.asarray(msg.touch_pressures, dtype=np.float64)

    def _on_force(self, msg) -> None:
        self.latest_force = np.asarray(msg.normal_forces_n, dtype=np.float64)

    def _on_log(self, msg) -> None:
        if "temperature" not in msg.msg.lower():
            return
        match = TEMP_RE.search(msg.msg)
        value = float(match.group(1)) if match else None
        self.temperature_samples.append({"time": time.time(), "value_c": value, "message": msg.msg})
        if value is not None and value >= self.args.thermal_abort_c:
            self.stop_requested = True

    def spin_once(self, timeout: float = 0.001) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout)

    def estop(self, value: bool) -> None:
        message = self.Bool()
        message.data = value
        for _ in range(5 if value else 3):
            self.estop_pub.publish(message)
            self.spin_once(0.02)

    def wait_feedback(self) -> None:
        deadline = time.monotonic() + 5.0
        while self.q is None and time.monotonic() < deadline:
            self.spin_once(0.05)
        if self.q is None:
            raise RuntimeError("no /dex5_1/joint_state feedback within 5 seconds")

    def validate_low_gain_controller(self) -> dict:
        names = [
            "hybrid_kp_scale",
            "hybrid_kd_scale",
            "hybrid_thumb_kp_scale",
            "hybrid_thumb_kd_scale",
        ]
        if not self.gain_client.wait_for_services(timeout_sec=2.0):
            raise RuntimeError("controller parameter service is unavailable")
        future = self.gain_client.get_parameters(names)
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            self.spin_once(0.02)
        if not future.done() or future.result() is None:
            raise RuntimeError("controller hybrid-gain query timed out")
        values = future.result().values
        if len(values) != len(names):
            raise RuntimeError("controller returned an incomplete hybrid-gain response")
        gains = {}
        for name, value in zip(names, values):
            if value.type != self.ParameterType.PARAMETER_DOUBLE:
                raise RuntimeError(f"controller parameter {name} is not a double")
            gains[name] = float(value.double_value)

        for name in ("hybrid_kp_scale", "hybrid_thumb_kp_scale"):
            if not (0.0 < gains[name] <= self.args.max_preload_kp_scale):
                raise RuntimeError(
                    f"controller {name}={gains[name]:.3f} exceeds low-gain cap "
                    f"{self.args.max_preload_kp_scale:.3f}"
                )
        for name in ("hybrid_kd_scale", "hybrid_thumb_kd_scale"):
            if not (0.0 < gains[name] <= self.args.max_preload_kd_scale):
                raise RuntimeError(
                    f"controller {name}={gains[name]:.3f} exceeds damping cap "
                    f"{self.args.max_preload_kd_scale:.3f}"
                )
        return gains

    def _validate_command(
        self,
        target: np.ndarray,
        previous: np.ndarray,
        dt: float,
        feedback_speed_abort_rad_s: float | None = None,
        feedback_speed_abort_windows: int | None = None,
    ) -> None:
        if self.stop_requested:
            raise RuntimeError("stop or thermal abort requested")
        if time.monotonic() - self.feedback_time > self.args.watchdog_s:
            raise RuntimeError("joint feedback watchdog expired")
        delta = float(np.max(np.abs(target - previous)))
        if delta > self.args.delta_limit_rad + 1.0e-9:
            raise RuntimeError(f"command delta {delta:.6f} rad exceeds cap")
        if delta / dt > self.args.speed_limit_rad_s + 1.0e-9:
            raise RuntimeError(f"command speed {delta / dt:.6f} rad/s exceeds cap")
        # Match the S0.7 hardware acceptance tolerance for measured zero-offset
        # feedback during the first staging samples. Replay package targets are
        # still required to be strictly inside the audited limits at load time.
        limit_tolerance = 0.002
        if np.any(target < self.args.lower - limit_tolerance) or np.any(
            target > self.args.upper + limit_tolerance
        ):
            raise RuntimeError("target outside audited URDF limits plus 0.002 rad tolerance")
        if self.position_diff_velocity_generation != self.last_checked_velocity_generation:
            self.last_checked_velocity_generation = self.position_diff_velocity_generation
            independent_speed = np.abs(
                self.position_diff_velocity[INDEPENDENT_FEEDBACK_JOINTS]
            )
            measured_speed = float(np.max(independent_speed))
            speed_limit = (
                self.args.feedback_speed_abort_rad_s
                if feedback_speed_abort_rad_s is None
                else feedback_speed_abort_rad_s
            )
            window_limit = (
                self.args.feedback_speed_abort_windows
                if feedback_speed_abort_windows is None
                else feedback_speed_abort_windows
            )
            if measured_speed > speed_limit:
                self.sustained_feedback_speed_violations += 1
            else:
                self.sustained_feedback_speed_violations = 0
            if (
                self.sustained_feedback_speed_violations
                >= window_limit
            ):
                joint = int(INDEPENDENT_FEEDBACK_JOINTS[np.argmax(independent_speed)])
                raise RuntimeError(
                    f"independent-joint 50ms velocity {measured_speed:.4f} rad/s on joint "
                    f"{joint} exceeded {speed_limit:.4f} rad/s for "
                    f"{self.sustained_feedback_speed_violations} consecutive windows"
                )

    def publish(
        self,
        target: np.ndarray,
        previous: np.ndarray,
        dt: float,
        feedback_speed_abort_rad_s: float | None = None,
        feedback_speed_abort_windows: int | None = None,
    ) -> None:
        self._validate_command(
            target,
            previous,
            dt,
            feedback_speed_abort_rad_s,
            feedback_speed_abort_windows,
        )
        message = self.Float64MultiArray()
        message.data = target.tolist()
        self.position_pub.publish(message)

    def publish_low_gain(
        self,
        target: np.ndarray,
        previous: np.ndarray,
        dt: float,
        feedback_speed_abort_rad_s: float | None = None,
        feedback_speed_abort_windows: int | None = None,
    ) -> None:
        """Publish a zero-torque hybrid target using the controller's reduced gains."""
        self._validate_command(
            target,
            previous,
            dt,
            feedback_speed_abort_rad_s,
            feedback_speed_abort_windows,
        )
        message = self.Float64MultiArray()
        message.data = np.concatenate((target, np.zeros(20, dtype=np.float64))).tolist()
        self.hybrid_pub.publish(message)

    def reset_feedback_speed_gate(self) -> None:
        self.sustained_feedback_speed_violations = 0
        self.last_checked_velocity_generation = self.position_diff_velocity_generation

    def hold_low_gain_until_stable(
        self,
        target: np.ndarray,
        duration_s: float,
        required_stable_windows: int,
    ) -> dict:
        deadline = time.monotonic() + duration_s
        stable_windows = 0
        observed_windows = 0
        peak_speed = 0.0
        last_generation = self.position_diff_velocity_generation
        while time.monotonic() < deadline:
            self.publish_low_gain(
                target,
                target,
                0.01,
                self.args.preload_feedback_speed_abort_rad_s,
                1,
            )
            self.spin_once(0.004)
            time.sleep(0.006)
            if self.position_diff_velocity_generation == last_generation:
                continue
            last_generation = self.position_diff_velocity_generation
            self._validate_command(
                target,
                target,
                0.01,
                self.args.preload_feedback_speed_abort_rad_s,
                1,
            )
            observed_windows += 1
            speed = float(
                np.max(np.abs(self.position_diff_velocity[INDEPENDENT_FEEDBACK_JOINTS]))
            )
            peak_speed = max(peak_speed, speed)
            if speed <= self.args.preload_settle_speed_rad_s:
                stable_windows += 1
            else:
                stable_windows = 0
        if stable_windows < required_stable_windows:
            raise RuntimeError(
                "low-gain preload did not settle: "
                f"stable_windows={stable_windows}, required={required_stable_windows}, "
                f"observed={observed_windows}, peak={peak_speed:.4f} rad/s"
            )
        return {
            "observed_windows": observed_windows,
            "stable_windows": stable_windows,
            "peak_speed_rad_s": peak_speed,
        }

    def arm_low_gain_start(
        self,
        target: np.ndarray,
        expected_feedback: np.ndarray,
        mapping: dict,
        maximum_feedback_error: float,
    ) -> dict:
        resume, thumb_offsets = mapped_resume_target(
            target,
            self.q.copy(),
            expected_feedback,
            mapping,
            self.args.resume_command_offset_limit_rad,
        )
        resume = np.clip(resume, self.args.lower, self.args.upper)
        self.reset_feedback_speed_gate()

        preload = self.hold_low_gain_until_stable(
            resume,
            self.args.preload_hold_s,
            self.args.preload_stable_windows,
        )

        previous = resume.copy()
        max_step = self.args.preload_alignment_speed_rad_s * 0.01
        steps = max(1, int(math.ceil(float(np.max(np.abs(target - resume))) / max_step)))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            command = resume + alpha * (target - resume)
            self.publish_low_gain(
                command,
                previous,
                0.01,
                self.args.preload_feedback_speed_abort_rad_s,
                1,
            )
            previous = command
            self.spin_once(0.004)
            time.sleep(0.006)

        aligned = self.hold_low_gain_until_stable(
            target,
            self.args.preload_settle_s,
            self.args.preload_stable_windows,
        )

        if self.args.replay_control_mode == "low_gain_hybrid":
            self.reset_feedback_speed_gate()
            low_gain_validation = self.hold_low_gain_until_stable(
                target,
                self.args.full_gain_validation_s,
                self.args.preload_stable_windows,
            )
            return {
                "status": "MAPPED_LOW_GAIN_PRELOAD_ACCEPTED",
                "maximum_feedback_error_rad": maximum_feedback_error,
                "resume_command_max_offset_rad": float(np.max(np.abs(resume - target))),
                "resume_command": resume.tolist(),
                "thumb_resume_offsets": thumb_offsets,
                "preload": preload,
                "aligned": aligned,
                "preload_alignment_speed_rad_s": self.args.preload_alignment_speed_rad_s,
                "replay_control_mode": self.args.replay_control_mode,
                "low_gain_replay_validation": low_gain_validation,
            }

        # Switch stiffness only after reduced-gain control has settled.
        self.reset_feedback_speed_gate()
        switch_deadline = time.monotonic() + self.args.full_gain_validation_s
        while time.monotonic() < switch_deadline:
            self.publish(
                target,
                target,
                0.01,
                self.args.feedback_speed_abort_rad_s,
                1,
            )
            self.spin_once(0.004)
            time.sleep(0.006)
        self._validate_command(
            target,
            target,
            0.01,
            self.args.feedback_speed_abort_rad_s,
            1,
        )

        return {
            "status": "MAPPED_LOW_GAIN_PRELOAD_ACCEPTED",
            "maximum_feedback_error_rad": maximum_feedback_error,
            "resume_command_max_offset_rad": float(np.max(np.abs(resume - target))),
            "resume_command": resume.tolist(),
            "thumb_resume_offsets": thumb_offsets,
            "preload": preload,
            "aligned": aligned,
            "preload_alignment_speed_rad_s": self.args.preload_alignment_speed_rad_s,
            "full_gain_validation_s": self.args.full_gain_validation_s,
        }

    def ramp_to_start(self, target: np.ndarray) -> None:
        initial_error = float(np.max(np.abs(target - self.q)))
        if initial_error > self.args.initial_error_limit_rad:
            raise RuntimeError(
                f"initial pose mismatch {initial_error:.4f} rad exceeds "
                f"{self.args.initial_error_limit_rad:.4f}; manually restore the approved P3 entry"
            )
        previous = self.q.copy()
        steps = max(1, int(math.ceil(initial_error / (self.args.alignment_speed_rad_s * 0.01))))
        last = previous
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            target_i = previous + alpha * (target - previous)
            self.publish(target_i, last, 0.01)
            last = target_i
            self.spin_once(0.004)
            time.sleep(0.006)

    def arm_mapped_start(
        self, target: np.ndarray, expected_feedback: np.ndarray, mapping: dict
    ) -> dict:
        """Arm an already staged mapped-P3 pose without mixing command/feedback coordinates."""
        error = np.abs(self.q - expected_feedback)
        maximum = float(np.max(error))
        if maximum > self.args.initial_error_limit_rad:
            joint = int(np.argmax(error))
            raise RuntimeError(
                f"mapped P3 feedback mismatch {maximum:.4f} rad on joint {joint} exceeds "
                f"{self.args.initial_error_limit_rad:.4f}; rerun the empty-hand mapped-P3 gate"
            )
        if self.args.startup_mode == "low_gain_preload":
            return self.arm_low_gain_start(
                target,
                expected_feedback,
                mapping,
                maximum,
            )

        if self.args.startup_mode == "legacy_direct":
            self.publish(target, target, 0.01)
            deadline = time.monotonic() + self.args.direct_start_hold_s
            while time.monotonic() < deadline:
                self.spin_once(0.005)
                self.publish(target, target, 0.01)
                time.sleep(0.005)
            return {
                "status": "MAPPED_DIRECT_START_ACCEPTED",
                "maximum_feedback_error_rad": maximum,
                "maximum_feedback_error_joint": int(np.argmax(error)),
                "feedback_speed_gate": {
                    "independent_joint_indices": INDEPENDENT_FEEDBACK_JOINTS.tolist(),
                    "consecutive_windows": self.args.feedback_speed_abort_windows,
                },
            }

        resume, thumb_offsets = mapped_resume_target(
            target,
            self.q.copy(),
            expected_feedback,
            mapping,
            self.args.resume_command_offset_limit_rad,
        )
        resume = np.clip(resume, self.args.lower, self.args.upper)

        # Re-enable against the measured loaded posture first. This prevents
        # the vendor PD loop from restoring the no-load P3 target in one step.
        previous = resume.copy()
        self.publish(resume, resume, 0.01)
        hold_deadline = time.monotonic() + self.args.resume_hold_s
        while time.monotonic() < hold_deadline:
            self.spin_once(0.005)
            self.publish(resume, resume, 0.01)
            time.sleep(0.005)

        max_step = self.args.alignment_speed_rad_s * 0.01
        steps = max(1, int(math.ceil(float(np.max(np.abs(target - resume))) / max_step)))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            command = resume + alpha * (target - resume)
            self.publish(command, previous, 0.01)
            previous = command
            self.spin_once(0.004)
            time.sleep(0.006)

        settle_deadline = time.monotonic() + self.args.resume_settle_s
        while time.monotonic() < settle_deadline:
            self.spin_once(0.005)
            self.publish(target, target, 0.01)
            time.sleep(0.005)
        return {
            "status": "MAPPED_SOFT_RESUME_ACCEPTED",
            "maximum_feedback_error_rad": maximum,
            "maximum_feedback_error_joint": int(np.argmax(error)),
            "resume_command_max_offset_rad": float(np.max(np.abs(resume - target))),
            "resume_command": resume.tolist(),
            "thumb_resume_offsets": thumb_offsets,
        }

    def stage_to_start(self, target: np.ndarray) -> dict:
        """Move an empty powered hand to an audited replay start pose."""
        start = self.q.copy()
        distance = float(np.max(np.abs(target - start)))
        start_mono = time.monotonic()
        previous = start
        command_step = self.args.alignment_speed_rad_s * 0.01
        feedback_lead = 0.003
        feedback_lead_max = 0.010
        feedback_lead_peak = feedback_lead
        timeout_s = distance / self.args.alignment_speed_rad_s * 5.0 + 10.0
        velocity_peak = 0.0
        best_error = distance
        last_progress = start_mono
        last_lead_adjustment = start_mono
        while True:
            now = time.monotonic()
            elapsed = now - start_mono
            if elapsed > timeout_s:
                raise RuntimeError("feedback-limited staging timed out")
            measured_speed = float(np.max(np.abs(self.position_diff_velocity)))
            velocity_peak = max(velocity_peak, measured_speed)
            current_error = float(np.max(np.abs(target - self.q)))
            if current_error < best_error - 0.0005:
                best_error = current_error
                last_progress = now
            if measured_speed > self.args.speed_limit_rad_s:
                feedback_lead = max(0.003, feedback_lead - 0.001)
                feedback_limited = self.q.copy()
                last_lead_adjustment = now
            else:
                if now - last_progress >= 1.0 and now - last_lead_adjustment >= 1.0:
                    feedback_lead = min(feedback_lead_max, feedback_lead + 0.001)
                    feedback_lead_peak = max(feedback_lead_peak, feedback_lead)
                    last_lead_adjustment = now
                feedback_limited = self.q + np.clip(
                    target - self.q, -feedback_lead, feedback_lead
                )
            target_i = previous + np.clip(
                feedback_limited - previous, -command_step, command_step
            )
            self.publish(target_i, previous, 0.01)
            previous = target_i
            self.spin_once(0.004)
            time.sleep(0.006)
            if float(np.max(np.abs(target - self.q))) <= 0.01 and float(
                np.max(np.abs(target - previous))
            ) <= command_step:
                break
        hold_deadline = time.monotonic() + 0.5
        while time.monotonic() < hold_deadline:
            velocity_peak = max(
                velocity_peak, float(np.max(np.abs(self.position_diff_velocity)))
            )
            target_i = previous + np.clip(target - previous, -command_step, command_step)
            self.publish(target_i, previous, 0.01)
            previous = target_i
            self.spin_once(0.004)
            time.sleep(0.006)
        final_error = float(np.max(np.abs(target - self.q)))
        return {
            "status": "COMPLETED",
            "duration_s": time.monotonic() - start_mono,
            "initial_max_error_rad": distance,
            "final_max_error_rad": final_error,
            "alignment_speed_cap_rad_s": self.args.alignment_speed_rad_s,
            "feedback_lead_initial_rad": 0.003,
            "feedback_lead_peak_rad": feedback_lead_peak,
            "feedback_lead_cap_rad": feedback_lead_max,
            "position_diff_velocity_max_rad_s": velocity_peak,
            "temperature_events": list(self.temperature_samples),
        }

    def replay_segment(self, trajectory_id: int, segment: np.ndarray) -> dict:
        self.active_samples = []
        self.active_trajectory_id = trajectory_id
        self.applied_samples = []
        self.temperature_samples = []
        start_wall = time.time()
        start_mono = time.monotonic()
        next_tick = start_mono
        previous = segment[0].copy()
        for row_index, target in enumerate(segment):
            while time.monotonic() < next_tick:
                self.spin_once(0.001)
                time.sleep(0.0005)
            self.spin_once(0.0)
            if self.args.replay_control_mode == "low_gain_hybrid":
                self.publish_low_gain(
                    target,
                    previous,
                    0.01,
                    self.args.feedback_speed_abort_rad_s,
                    self.args.feedback_speed_abort_windows,
                )
            else:
                self.publish(target, previous, 0.01)
            raw_command_feedback_difference = target - self.q
            self.active_samples.append(
                {
                    "t": time.monotonic() - start_mono,
                    "target": target.copy(),
                    "q": self.q.copy(),
                    "dq": self.dq.copy() if self.dq is not None else np.zeros(20),
                    "position_diff_velocity": self.position_diff_velocity.copy(),
                    "raw_command_feedback_difference": raw_command_feedback_difference.copy(),
                    "touch": self.latest_touch.copy(),
                    "force": self.latest_force.copy(),
                }
            )
            previous = target
            next_tick = start_mono + (row_index + 1) * 0.01
        for _ in range(20):
            self.spin_once(0.005)
            if self.args.replay_control_mode == "low_gain_hybrid":
                self.publish_low_gain(
                    segment[-1],
                    segment[-1],
                    0.01,
                    self.args.feedback_speed_abort_rad_s,
                    self.args.feedback_speed_abort_windows,
                )
            else:
                self.publish(segment[-1], segment[-1], 0.01)
            time.sleep(0.005)
        raw_difference = np.stack(
            [sample["raw_command_feedback_difference"] for sample in self.active_samples]
        )
        dq = np.stack([sample["dq"] for sample in self.active_samples])
        position_diff_velocity = np.stack(
            [sample["position_diff_velocity"] for sample in self.active_samples]
        )
        command_delta = np.diff(segment, axis=0)
        contact_rows = [sample["touch"] for sample in self.active_samples if sample["touch"].size]
        force_rows = [sample["force"] for sample in self.active_samples if sample["force"].size]
        return {
            "trajectory_id": trajectory_id,
            "status": "COMPLETED",
            "start_unix_s": start_wall,
            "command_rows": len(segment),
            "duration_s": time.monotonic() - start_mono,
            "command_max_delta_rad": float(np.max(np.abs(command_delta))) if command_delta.size else 0.0,
            "command_max_speed_rad_s": float(np.max(np.abs(command_delta))) * 100.0 if command_delta.size else 0.0,
            # Command and raw feedback coordinates are not identical on the
            # calibrated thumb/coupled joints. Keep this diagnostic explicit;
            # it is not a physical joint tracking-error acceptance metric.
            "raw_command_feedback_difference_mean_rad": float(
                np.mean(np.abs(raw_difference))
            ),
            "raw_command_feedback_difference_p95_rad": percentile(
                np.abs(raw_difference).ravel(), 95
            ),
            "raw_command_feedback_difference_max_rad": float(
                np.max(np.abs(raw_difference))
            ),
            "reported_velocity_max_rad_s": float(np.max(np.abs(dq))),
            "position_diff_velocity_max_rad_s": float(
                np.max(np.abs(position_diff_velocity))
            ),
            "applied_position_samples": len(self.applied_samples),
            "touch_sample_count": len(contact_rows),
            "touch_active_ratio": float(np.mean(np.asarray(contact_rows) > 0.1)) if contact_rows else None,
            "force_sample_count": len(force_rows),
            "normal_force_max_n": float(np.max(force_rows)) if force_rows else None,
            "temperature_events": list(self.temperature_samples),
            "temperature_max_c": max(
                (x["value_c"] for x in self.temperature_samples if x["value_c"] is not None),
                default=None,
            ),
        }

    def partial_replay_summary(self) -> dict | None:
        if not self.active_samples:
            return None
        velocity = np.stack(
            [sample["position_diff_velocity"] for sample in self.active_samples]
        )
        peak = np.unravel_index(int(np.argmax(np.abs(velocity))), velocity.shape)
        last = self.active_samples[-1]
        return {
            "trajectory_id": self.active_trajectory_id,
            "status": "ABORTED_PARTIAL",
            "command_rows_recorded": len(self.active_samples),
            "duration_s": float(last["t"]),
            "position_diff_velocity_max_rad_s": float(np.max(np.abs(velocity))),
            "position_diff_velocity_peak_row": int(peak[0]),
            "position_diff_velocity_peak_joint": int(peak[1]),
            "last_target": last["target"].tolist(),
            "last_feedback": last["q"].tolist(),
        }


def render_report(result: dict) -> str:
    rows = []
    for item in result["trajectories"]:
        rows.append(
            f"| {item['trajectory_id']} | {item['status']} | "
            f"{item.get('raw_command_feedback_difference_mean_rad', 0):.4f} | "
            f"{item.get('raw_command_feedback_difference_p95_rad', 0):.4f} | "
            f"{item.get('reported_velocity_max_rad_s', 0):.4f} | "
            f"{item.get('position_diff_velocity_max_rad_s', 0):.4f} | "
            f"{item.get('temperature_max_c')} | {item.get('dropped_object')} | "
            f"{item.get('pen_visibly_rotated')} |"
        )
    return f"""# Dex5 S0.9 Powered Replay Acceptance

## 结论

```text
状态: {result['status']}
执行轨迹: {len(result['trajectories'])}/{result['requested_trajectory_count']}
autonomous policy: NOT RUN
PPO/student/online learning: NOT RUN
```

## 固定安全门

- mapped S0.9 replay SHA-256: `{result['replay_sha256']}`
- thumb mapping SHA-256: `{result['mapping_sha256']}`
- accepted P3 static gate SHA-256: `{result['static_gate_sha256']}`
- command frequency: `100 Hz`
- command speed cap: `{result['safety']['speed_limit_rad_s']} rad/s`
- command delta cap: `{result['safety']['delta_limit_rad']} rad`
- startup mode: `{result['safety']['startup_mode']}`
- feedback watchdog: `{result['safety']['watchdog_s']} s`
- emergency stop: 每条轨迹结束及异常路径均发送
- thermal monitor: `/rosout` controller temperature event，`{result['safety']['thermal_abort_c']} C` abort

## 逐轨迹结果

| ID | 状态 | raw cmd-feedback mean | raw cmd-feedback p95 | raw SDK dq max | 50ms position-diff max | max temp C | dropped | rotated |
|---:|---|---:|---:|---:|---:|---:|---|---|
{chr(10).join(rows) if rows else '| - | NOT RUN | - | - | - | - | - | - | - |'}

## 边界

安全超速判据使用 S0.7 已验收的 50 ms 位置差分速度；原始 SDK `dq` 仅作诊断。
温度只有在 vendor controller 产生温度日志时可观测；`None` 不代表低温。
掉笔由现场人员逐轨迹人工标注。`raw cmd-feedback` 只记录命令坐标与原始
SDK 反馈的数值差；由于拇指映射和 PIP/DIP 耦合，它不解释为物理关节跟踪误差。
motor response 使用公共 20D joint feedback 和 `/dex5_1/applied_position` 记录，
不解释为电机轴测量。
"""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    mapped_dir = root / "powered_replay_s0_9"
    calibration = root / "hardware_calibration_s0_8_1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--loaded-hold-only", action="store_true")
    parser.add_argument("--arm-token", default="")
    parser.add_argument("--confirm-soft-pad", choices=("NO", "YES"), default="NO")
    parser.add_argument("--confirm-pen-placed", choices=("NO", "YES"), default="NO")
    parser.add_argument("--confirm-empty-hand", choices=("NO", "YES"), default="NO")
    parser.add_argument("--commands", type=Path, default=mapped_dir / "mapped_replay_100hz.npz")
    parser.add_argument("--manifest", type=Path, default=mapped_dir / "mapped_replay_manifest.json")
    parser.add_argument("--expected-replay-sha256", default=EXPECTED_REPLAY_SHA256)
    parser.add_argument(
        "--diagnostic-package",
        action="store_true",
        help="Allow a separately versioned diagnostic package after explicit hash verification.",
    )
    parser.add_argument(
        "--replay-control-mode",
        choices=("position", "low_gain_hybrid"),
        default="position",
    )
    parser.add_argument(
        "--static-gate",
        type=Path,
        default=calibration / "mapped_p3_static_gate_joint16_corrected.json",
    )
    parser.add_argument("--safety", type=Path, default=root / "real_replay_package/safety_limits.json")
    parser.add_argument("--output-json", type=Path, default=Path(__file__).with_name("powered_replay_results.json"))
    parser.add_argument(
        "--output-report",
        type=Path,
        default=mapped_dir / "Dex5_S0_9_Mapped_Replay_Preflight.md",
    )
    parser.add_argument("--trajectory-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--speed-limit-rad-s", type=float, default=0.08)
    parser.add_argument("--delta-limit-rad", type=float, default=0.01)
    parser.add_argument("--watchdog-s", type=float, default=0.10)
    parser.add_argument("--alignment-speed-rad-s", type=float, default=0.04)
    parser.add_argument("--initial-error-limit-rad", type=float, default=0.12)
    parser.add_argument("--feedback-speed-abort-rad-s", type=float, default=0.12)
    parser.add_argument("--feedback-speed-abort-windows", type=int, default=3)
    parser.add_argument("--direct-start-hold-s", type=float, default=0.20)
    parser.add_argument(
        "--startup-mode",
        choices=("low_gain_preload", "legacy_direct", "legacy_soft_resume"),
        default="low_gain_preload",
    )
    parser.add_argument(
        "--soft-resume",
        action="store_true",
        help="Deprecated alias for --startup-mode legacy_soft_resume.",
    )
    parser.add_argument("--resume-command-offset-limit-rad", type=float, default=0.06)
    parser.add_argument("--resume-hold-s", type=float, default=0.25)
    parser.add_argument("--resume-settle-s", type=float, default=0.20)
    parser.add_argument("--preload-hold-s", type=float, default=0.40)
    parser.add_argument("--preload-settle-s", type=float, default=0.40)
    parser.add_argument("--preload-stable-windows", type=int, default=3)
    parser.add_argument("--preload-alignment-speed-rad-s", type=float, default=0.01)
    parser.add_argument("--preload-feedback-speed-abort-rad-s", type=float, default=0.12)
    parser.add_argument("--preload-settle-speed-rad-s", type=float, default=0.04)
    parser.add_argument("--full-gain-validation-s", type=float, default=0.30)
    parser.add_argument("--max-preload-kp-scale", type=float, default=0.20)
    parser.add_argument("--max-preload-kd-scale", type=float, default=0.40)
    parser.add_argument("--thermal-abort-c", type=float, default=75.0)
    args = parser.parse_args()
    safety = json.loads(args.safety.read_text(encoding="utf-8"))
    args.lower = np.asarray(safety["absolute_joint_lower_rad"], dtype=np.float64)
    args.upper = np.asarray(safety["absolute_joint_upper_rad"], dtype=np.float64)
    if args.speed_limit_rad_s > 0.08 or args.delta_limit_rad > 0.01:
        parser.error("S0.9 caps speed at 0.08 rad/s and delta at 0.01 rad")
    if args.diagnostic_package and args.replay_control_mode != "low_gain_hybrid":
        parser.error("diagnostic replay requires --replay-control-mode low_gain_hybrid")
    if args.count < 1 or args.count > 20:
        parser.error("count must be in 1..20")
    if args.feedback_speed_abort_windows < 1:
        parser.error("feedback speed abort windows must be positive")
    if args.soft_resume:
        if args.startup_mode != "low_gain_preload":
            parser.error("--soft-resume cannot be combined with --startup-mode")
        args.startup_mode = "legacy_soft_resume"
    exclusive_modes = sum((args.stage_only, args.startup_only, args.loaded_hold_only))
    if exclusive_modes > 1:
        parser.error("stage-only, startup-only, and loaded-hold-only are mutually exclusive")
    if args.startup_only and args.startup_mode != "low_gain_preload":
        parser.error("startup-only acceptance requires low_gain_preload")
    if args.loaded_hold_only and args.startup_mode != "low_gain_preload":
        parser.error("loaded-hold-only acceptance requires low_gain_preload")
    if args.preload_stable_windows < 1:
        parser.error("preload stable windows must be positive")
    minimum_stability_duration = args.preload_stable_windows * 0.05 + 0.05
    if args.preload_hold_s < minimum_stability_duration:
        parser.error("preload hold duration is too short for the requested stable windows")
    if args.preload_settle_s < minimum_stability_duration:
        parser.error("preload settle duration is too short for the requested stable windows")
    if args.full_gain_validation_s < 0.10:
        parser.error("full-gain validation must cover at least two 50 ms windows")
    if not (0.0 < args.max_preload_kp_scale <= 0.30):
        parser.error("maximum preload kp scale must be in (0, 0.30]")
    if not (0.0 < args.max_preload_kd_scale <= 0.50):
        parser.error("maximum preload kd scale must be in (0, 0.50]")
    if not (0.0 < args.preload_alignment_speed_rad_s <= 0.02):
        parser.error("preload alignment speed must be in (0, 0.02] rad/s")
    if not (0.0 < args.preload_feedback_speed_abort_rad_s <= 0.12):
        parser.error("preload feedback speed abort must be in (0, 0.12] rad/s")
    if not (0.0 < args.preload_settle_speed_rad_s < args.preload_feedback_speed_abort_rad_s):
        parser.error("preload settle speed must be below its abort threshold")
    return args


def main() -> None:
    args = parse_args()
    package = load_package(args)
    static_gate = json.loads(args.static_gate.read_text(encoding="utf-8"))
    mapping_path = (
        Path(__file__).resolve().parents[1]
        / "hardware_calibration_s0_8_1"
        / "thumb_global_mapping_all.json"
    )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    manifest_mapping_sha = json.loads(args.manifest.read_text(encoding="utf-8"))[
        "mapping_sha256"
    ]
    if sha256(mapping_path) != manifest_mapping_sha:
        raise SystemExit("thumb mapping hash does not match mapped replay manifest")
    if static_gate.get("mapped_p3_gate", {}).get("status") != "PASS_THUMB_P3_STATIC_GATE":
        raise SystemExit("mapped P3 static gate is not accepted")
    expected_start_feedback = np.asarray(
        static_gate["mapped_p3_gate"]["settled_raw_feedback"], dtype=np.float64
    )
    total = len(package["offsets"]) - 1
    end = args.trajectory_start + args.count
    if args.trajectory_start < 0 or end > total:
        raise SystemExit(f"requested trajectory range exceeds available 0..{total - 1}")
    result = {
        "schema": "dex5-s0.9-mapped-powered-replay-v2",
        "timestamp_unix_s": time.time(),
        "status": "OFFLINE_PREFLIGHT_PASS",
        "replay_sha256": package["sha256"],
        "mapping_sha256": manifest_mapping_sha,
        "static_gate_sha256": sha256(args.static_gate),
        "requested_trajectory_count": args.count,
        "safety": {
            "speed_limit_rad_s": args.speed_limit_rad_s,
            "delta_limit_rad": args.delta_limit_rad,
            "watchdog_s": args.watchdog_s,
            "thermal_abort_c": args.thermal_abort_c,
            "startup_mode": args.startup_mode,
            "replay_control_mode": args.replay_control_mode,
            "diagnostic_package": args.diagnostic_package,
            "feedback_speed_abort_rad_s": args.feedback_speed_abort_rad_s,
            "feedback_speed_abort_windows": args.feedback_speed_abort_windows,
            "feedback_speed_monitored_joint_indices": INDEPENDENT_FEEDBACK_JOINTS.tolist(),
        },
        "offline_preflight": {
            "trajectory_count": total,
            "command_rows": len(package["command"]),
            "maximum_delta_rad": package["max_delta"],
            "maximum_speed_rad_s": package["max_speed"],
        },
        "trajectories": [],
        "staging": None,
    }
    if args.execute:
        if args.startup_only:
            if (
                args.confirm_soft_pad != "YES"
                or args.confirm_empty_hand != "YES"
                or args.arm_token != "DEX5-S0.9-STARTUP"
            ):
                raise SystemExit(
                    "startup-only mode requires --confirm-soft-pad YES "
                    "--confirm-empty-hand YES --arm-token DEX5-S0.9-STARTUP"
                )
        elif args.loaded_hold_only:
            if (
                args.confirm_soft_pad != "YES"
                or args.confirm_pen_placed != "YES"
                or args.arm_token != "DEX5-S0.9-LOADED-HOLD"
            ):
                raise SystemExit(
                    "loaded-hold-only mode requires --confirm-soft-pad YES "
                    "--confirm-pen-placed YES --arm-token DEX5-S0.9-LOADED-HOLD"
                )
        elif (
            args.confirm_soft_pad != "YES"
            or args.confirm_pen_placed != "YES"
            or args.arm_token != "DEX5-S0.9-ARM"
        ):
            raise SystemExit(
                "powered mode requires --confirm-soft-pad YES "
                "--confirm-pen-placed YES --arm-token DEX5-S0.9-ARM"
            )
        import rclpy

        rclpy.init()
        runner = PoweredReplay(args)
        try:
            runner.estop(True)
            runner.wait_feedback()
            if args.startup_mode == "low_gain_preload":
                result["controller_hybrid_gains"] = runner.validate_low_gain_controller()
            if args.stage_only:
                raise RuntimeError(
                    "mapped replay staging must use the accepted S0.8.1 mapped-P3 static tool"
                )
            else:
                for trajectory_id in range(args.trajectory_start, end):
                    start = int(package["offsets"][trajectory_id])
                    stop = int(package["offsets"][trajectory_id + 1])
                    segment = package["command"][start:stop]
                    if args.startup_only:
                        answer = input(
                            f"Startup gate for trajectory {trajectory_id}: EMPTY hand/soft pad/"
                            "physical E-stop ready; type STARTUP: "
                        ).strip()
                        if answer != "STARTUP":
                            raise RuntimeError("operator did not arm startup-only acceptance")
                    elif args.loaded_hold_only:
                        answer = input(
                            f"Loaded hold for trajectory {trajectory_id}: pen/soft pad/"
                            "physical E-stop ready; type HOLD: "
                        ).strip()
                        if answer != "HOLD":
                            raise RuntimeError("operator did not arm loaded-hold acceptance")
                    else:
                        answer = input(
                            f"Trajectory {trajectory_id}: pen/soft pad/physical E-stop ready; type ARM: "
                        ).strip()
                        if answer != "ARM":
                            raise RuntimeError("operator did not arm trajectory")
                    runner.estop(False)
                    try:
                        result["staging"] = runner.arm_mapped_start(
                            segment[0], expected_start_feedback, mapping
                        )
                        item = None if (args.startup_only or args.loaded_hold_only) else (
                            runner.replay_segment(trajectory_id, segment)
                        )
                    finally:
                        runner.estop(True)
                    if args.startup_only:
                        result["status"] = "STARTUP_GATE_PASS"
                        break
                    if args.loaded_hold_only:
                        result["status"] = "LOADED_HOLD_GATE_PASS"
                        break
                    dropped = input("Did the pen drop? type YES or NO: ").strip().upper()
                    if dropped not in {"YES", "NO"}:
                        dropped = "UNRECORDED"
                    item["dropped_object"] = dropped == "YES" if dropped != "UNRECORDED" else dropped
                    rotated = input(
                        "Did the pen visibly rotate? type YES or NO: "
                    ).strip().upper()
                    if rotated not in {"YES", "NO"}:
                        rotated = "UNRECORDED"
                    item["pen_visibly_rotated"] = (
                        rotated == "YES" if rotated != "UNRECORDED" else rotated
                    )
                    result["trajectories"].append(item)
                if not (args.startup_only or args.loaded_hold_only):
                    completed = len(result["trajectories"]) == args.count
                    task_pass = completed and all(
                        row.get("pen_visibly_rotated") is True
                        and row.get("dropped_object") is False
                        for row in result["trajectories"]
                    )
                    result["status"] = (
                        "TASK_PASS"
                        if task_pass
                        else "INTERFACE_PASS_TASK_FAIL" if completed else "PARTIAL"
                    )
        except Exception as exc:
            runner.estop(True)
            result["status"] = "ABORTED"
            result["abort_reason"] = str(exc)
            result["partial_replay"] = runner.partial_replay_summary()
        finally:
            runner.estop(True)
            runner.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
