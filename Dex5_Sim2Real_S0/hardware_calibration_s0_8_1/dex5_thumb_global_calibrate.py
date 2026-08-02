#!/usr/bin/env python3
"""Guarded global thumb mapping audit for Dex5 S0.8.1.

The script scans one thumb command channel at a time, records all four raw
thumb feedback channels, unwraps angular discontinuities, and builds a
piecewise inverse estimate for the P3 entry. It never sends the estimated P3
command. This is deterministic hardware calibration, not policy execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import numpy as np


THUMB = (16, 17, 18, 19)
JOINT_NAMES = (
    "Roll_21R", "Pitch_22R", "Pitch_23R", "Pitch_24R",
    "Roll_31R", "Pitch_32R", "Pitch_33R", "Pitch_34R",
    "Roll_41R", "Pitch_42R", "Pitch_43R", "Pitch_44R",
    "Roll_51R", "Pitch_52R", "Pitch_53R", "Pitch_54R",
    "Yaw_11R", "Roll_12R", "Pitch_13R", "Pitch_14R",
)
P3_TARGET = np.asarray(
    [
        -0.2114138901233673, 1.1113417148590088, 0.4013415575027466,
        0.4013415575027466, -0.004041674081236124, 1.4672520160675049,
        0.07857968658208847, 0.07857968658208847, 0.21655035018920898,
        1.3888839483261108, 0.7834743857383728, 0.7834743857383728,
        0.3464620113372803, 1.4994786977767944, 0.8696357011795044,
        0.8696357011795044, 0.48914584517478943, 0.6559728384017944,
        0.9171354174613953, 0.45034945011138916,
    ],
    dtype=np.float64,
)


def finite_list(value: np.ndarray) -> list[float]:
    return [float(x) for x in value]


def make_leg(start: float, stop: float, spacing: float) -> np.ndarray:
    distance = abs(stop - start)
    count = max(1, int(math.ceil(distance / spacing)))
    return np.linspace(start, stop, count + 1, dtype=np.float64)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) <= 1.0e-12 or np.ptp(y) <= 1.0e-12:
        return 0.0
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(xr, yr)[0, 1])


def unwrap_and_calibrate(command: np.ndarray, raw: np.ndarray) -> dict:
    """Resolve wrap, sign, and constant offset in public URDF feedback.

    The vendor topic is expected to represent the commanded URDF angle with a
    possible sign inversion and 2*pi branch offset. A scale different from one
    is reported as residual/model error rather than silently absorbed.
    """
    unwrapped = np.unwrap(raw)
    candidates = []
    for sign in (-1, 1):
        offset = float(np.median(command - sign * unwrapped))
        corrected = sign * unwrapped + offset
        residual = corrected - command
        candidates.append(
            {
                "sign": sign,
                "offset": offset,
                "corrected": corrected,
                "rmse": float(np.sqrt(np.mean(residual * residual))),
                "max_error": float(np.max(np.abs(residual))),
            }
        )
    best = min(candidates, key=lambda row: row["rmse"])
    return {"unwrapped": unwrapped, **best}


def collapse_piecewise(command: np.ndarray, feedback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median repeated forward/reverse samples into a monotonic lookup table."""
    rounded = np.round(command, 6)
    unique = np.unique(rounded)
    x = np.asarray([np.median(command[rounded == value]) for value in unique])
    y = np.asarray([np.median(feedback[rounded == value]) for value in unique])
    order = np.argsort(x)
    return x[order], y[order]


def hysteresis_at_repeated_commands(command: np.ndarray, feedback: np.ndarray) -> float:
    rounded = np.round(command, 6)
    spreads = [np.ptp(feedback[rounded == value]) for value in np.unique(rounded)]
    return float(max(spreads, default=0.0))


class ThumbScanner:
    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from rcl_interfaces.msg import Log
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool, Float64MultiArray
        from unitree_dex5_1_control.msg import Dex51JointState

        self.rclpy = rclpy
        self.Bool = Bool
        self.Float64MultiArray = Float64MultiArray
        self.node = Node("dex5_s0_8_1_thumb_global_calibration")
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
            Dex51JointState, "/dex5_1/joint_state", self._on_state, 100
        )
        self.node.create_subscription(Log, "/rosout", self._on_log, 100)
        self.args = args
        self.q: np.ndarray | None = None
        self.feedback_time = 0.0
        self.history: list[tuple[float, np.ndarray]] = []
        self.temperature_events: list[dict] = []
        self.stop_requested = False
        self.allow_non_thumb_motion = False
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, _signum, _frame) -> None:
        self.stop_requested = True

    def _on_state(self, msg) -> None:
        q = np.asarray(msg.finger_positions, dtype=np.float64)
        if q.shape != (20,) or not np.isfinite(q).all():
            return
        now = time.monotonic()
        self.q = q
        self.feedback_time = now
        self.history.append((now, q.copy()))

    def _on_log(self, msg) -> None:
        text = msg.msg.lower()
        if "temperature" not in text:
            return
        values = [int(token[:-2]) for token in msg.msg.split() if token.endswith("°C") and token[:-2].isdigit()]
        value = max(values) if values else None
        self.temperature_events.append(
            {"time": time.time(), "value_c": value, "message": msg.msg}
        )
        if value is not None and value >= self.args.thermal_abort_c:
            self.stop_requested = True

    def spin(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and not self.stop_requested:
            self.rclpy.spin_once(self.node, timeout_sec=0.004)

    def estop(self, value: bool) -> None:
        message = self.Bool()
        message.data = value
        for _ in range(5 if value else 3):
            self.estop_pub.publish(message)
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def wait_feedback(self) -> None:
        deadline = time.monotonic() + 5.0
        while self.q is None and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.q is None:
            raise RuntimeError("no joint feedback within 5 seconds")

    def check_fresh_baseline(self) -> None:
        thumb = np.asarray(THUMB, dtype=np.int64)
        violation = np.maximum(
            self.args.lower[thumb] - self.q[thumb],
            self.q[thumb] - self.args.upper[thumb],
        )
        if float(np.max(violation)) > 0.02:
            joint = int(thumb[int(np.argmax(violation))])
            raise RuntimeError(
                f"initial feedback joint {joint}={self.q[joint]:.4f} is outside the "
                "thumb URDF range; power-cycle hardware/anemone before scanning"
            )

    def publish(self, target: np.ndarray) -> None:
        if self.stop_requested:
            raise RuntimeError("stop or thermal abort requested")
        if time.monotonic() - self.feedback_time > 0.10:
            raise RuntimeError("joint feedback watchdog expired")
        thumb = np.asarray(THUMB, dtype=np.int64)
        if np.any(target[thumb] < self.args.lower[thumb] - 0.002) or np.any(
            target[thumb] > self.args.upper[thumb] + 0.002
        ):
            raise RuntimeError("thumb command outside audited URDF range")
        non_thumb = np.asarray([joint for joint in range(20) if joint not in THUMB])
        if not self.allow_non_thumb_motion and np.max(
            np.abs(target[non_thumb] - self.fixed_baseline[non_thumb])
        ) > 1.0e-9:
            raise RuntimeError("non-thumb command changed during thumb-only audit")
        message = self.Float64MultiArray()
        message.data = finite_list(target)
        self.position_pub.publish(message)

    def ramp(self, start: np.ndarray, target: np.ndarray) -> None:
        distance = float(np.max(np.abs(target - start)))
        steps = max(1, int(math.ceil(distance / (self.args.command_speed_rad_s * 0.01))))
        previous = start.copy()
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            command = start + alpha * (target - start)
            if float(np.max(np.abs(command - previous))) > self.args.delta_cap_rad + 1.0e-9:
                raise RuntimeError("generated command delta exceeds cap")
            self.publish(command)
            previous = command
            self.spin(0.01)

    def sample_point(self, command: np.ndarray, branch: str) -> dict:
        self.ramp(self.last_command, command)
        self.last_command = command.copy()
        started = time.monotonic()
        self.spin(self.args.settle_s)
        rows = [q for stamp, q in self.history if stamp >= started + 0.5 * self.args.settle_s]
        if not rows:
            raise RuntimeError("no feedback samples at scan point")
        feedback = np.median(np.stack(rows), axis=0)
        return {
            "branch": branch,
            "command": finite_list(command),
            "feedback": finite_list(feedback),
            "time_unix_s": time.time(),
        }

    def scan_joint(self, joint: int, baseline: np.ndarray) -> dict:
        points: list[dict] = []
        self.last_command = baseline.copy()
        upper = float(self.args.upper[joint])
        lower = float(self.args.lower[joint])
        base = float(baseline[joint])
        legs = [
            ("up", make_leg(base, upper, self.args.grid_spacing_rad)),
            ("up_return", make_leg(upper, base, self.args.grid_spacing_rad)[1:]),
            ("down", make_leg(base, lower, self.args.grid_spacing_rad)[1:]),
            ("down_return", make_leg(lower, base, self.args.grid_spacing_rad)[1:]),
        ]
        for branch, values in legs:
            for value in values:
                command = baseline.copy()
                command[joint] = value
                points.append(self.sample_point(command, branch))
        return {
            "command_joint": joint,
            "command_joint_name": JOINT_NAMES[joint],
            "baseline_command": finite_list(baseline),
            "points": points,
        }

    def run(self) -> dict:
        self.estop(True)
        self.wait_feedback()
        self.check_fresh_baseline()
        initial = self.q.copy()
        # Preserve the observed powered posture for all non-thumb channels.
        # The public SDK reports a repeatable -0.084 rad zero offset on the four
        # non-thumb proximal pitch joints; clipping those channels would issue an
        # unrelated hand motion during a thumb-only audit.
        baseline = self.q.copy()
        self.fixed_baseline = baseline.copy()
        scans = []
        for joint in self.args.joints:
            answer = input(
                f"EMPTY hand, thumb channel {joint} ({JOINT_NAMES[joint]}), type SCAN: "
            ).strip()
            if answer != "SCAN":
                raise RuntimeError(f"operator did not arm thumb channel {joint}")
            self.estop(False)
            try:
                scans.append(self.scan_joint(joint, baseline))
            finally:
                self.estop(True)
            self.spin(0.20)
        return {
            "status": "SCAN_COMPLETED",
            "initial_feedback": finite_list(initial),
            "baseline_command": finite_list(baseline),
            "scans": scans,
            "temperature_events": self.temperature_events,
            "software_estop_sent": True,
        }

    def stage_mapped_p3(self, mapping: dict) -> dict:
        self.estop(True)
        self.wait_feedback()
        self.check_fresh_baseline()
        baseline = self.q.copy()
        self.fixed_baseline = baseline.copy()
        target = P3_TARGET.copy()
        for joint in THUMB:
            estimate = mapping["analysis"]["inverse_p3_estimate"][str(joint)]
            value = estimate.get("preferred_estimated_sdk_command_rad")
            if value is None or not estimate.get("covered"):
                raise RuntimeError(f"thumb joint {joint} has no covered preferred P3 inverse")
            target[joint] = float(value)
        if self.args.thumb_command_override is not None:
            target[16:20] = self.args.thumb_command_override
        answer = input("EMPTY hand, mapped P3 static gate, type P3: ").strip()
        if answer != "P3":
            raise RuntimeError("operator did not arm mapped P3 static gate")
        self.allow_non_thumb_motion = True
        self.estop(False)
        try:
            self.last_command = baseline.copy()
            self.ramp(baseline, target)
            started = time.monotonic()
            self.spin(max(1.0, self.args.settle_s))
            rows = [q for stamp, q in self.history if stamp >= started + 0.5]
            if not rows:
                raise RuntimeError("no settled P3 feedback samples")
            feedback = np.median(np.stack(rows), axis=0)
        finally:
            self.estop(True)
        normalized_thumb = []
        thumb_error = []
        for joint in THUMB:
            channel = next(
                row
                for row in mapping["analysis"]["channels"]
                if int(row["dominant_feedback_joint"]) == joint
            )
            branch_name = mapping["analysis"]["inverse_p3_estimate"][str(joint)][
                "preferred_motion_branch"
            ]
            branch = channel["branch_models"][branch_name]
            branch_raw = np.asarray(branch["raw_unwrapped_feedback_rad"], dtype=np.float64)
            branch_urdf = np.asarray(branch["normalized_urdf_feedback_rad"], dtype=np.float64)
            sign = int(channel["direction_sign"])
            offset = float(channel["feedback_offset_rad"])
            candidates = np.asarray(
                [sign * (feedback[joint] + 2.0 * math.pi * k) + offset for k in range(-2, 3)]
            )
            corrected = float(candidates[np.argmin(np.abs(candidates - np.median(branch_raw)))])
            order = np.argsort(branch_raw)
            value = float(np.interp(corrected, branch_raw[order], branch_urdf[order]))
            normalized_thumb.append(value)
            thumb_error.append(value - float(P3_TARGET[joint]))
        max_thumb_error = float(np.max(np.abs(thumb_error)))
        return {
            "status": "PASS_THUMB_P3_STATIC_GATE" if max_thumb_error <= 0.12 else "FAIL_THUMB_P3_STATIC_GATE",
            "baseline_feedback": finite_list(baseline),
            "mapped_sdk_target": finite_list(target),
            "thumb_command_override_used": self.args.thumb_command_override is not None,
            "settled_raw_feedback": finite_list(feedback),
            "normalized_thumb_feedback": normalized_thumb,
            "desired_thumb_urdf": finite_list(P3_TARGET[16:20]),
            "thumb_error_rad": thumb_error,
            "max_thumb_error_rad": max_thumb_error,
            "non_thumb_raw_error_max_rad": float(np.max(np.abs(feedback[:16] - P3_TARGET[:16]))),
            "software_estop_sent": True,
        }


def analyze_scan(hardware: dict) -> dict:
    channels = []
    inverse_p3 = {}
    for scan in hardware.get("scans", []):
        joint = int(scan["command_joint"])
        points = scan["points"]
        command = np.asarray([row["command"][joint] for row in points], dtype=np.float64)
        raw = np.asarray([row["feedback"][16:20] for row in points], dtype=np.float64)
        unwrapped = np.unwrap(raw, axis=0)
        response_range = np.ptp(unwrapped, axis=0)
        dominant_local = int(np.argmax(response_range))
        dominant_joint = 16 + dominant_local
        calibrated = unwrap_and_calibrate(command, raw[:, dominant_local])
        corrected = calibrated["corrected"]
        corr = rank_correlation(command, corrected)
        sign = int(calibrated["sign"])
        table_command, table_feedback = collapse_piecewise(command, corrected)
        table_corr = rank_correlation(table_command, table_feedback)
        lower = float(np.min(table_command))
        upper = float(np.max(table_command))
        low_feedback = float(table_feedback[int(np.argmin(table_command))])
        high_feedback = float(table_feedback[int(np.argmax(table_command))])
        feedback_span = high_feedback - low_feedback
        if abs(feedback_span) <= 1.0e-6:
            normalized = np.full_like(corrected, lower)
            normalized_table = np.full_like(table_feedback, lower)
            feedback_to_urdf_scale = None
        else:
            feedback_to_urdf_scale = (upper - lower) / feedback_span
            normalized = lower + (corrected - low_feedback) * feedback_to_urdf_scale
            normalized_table = lower + (
                table_feedback - low_feedback
            ) * feedback_to_urdf_scale
        normalized_hysteresis = hysteresis_at_repeated_commands(command, normalized)
        model_prediction = np.interp(command, table_command, normalized_table)
        model_residual = normalized - model_prediction
        model_rmse = float(np.sqrt(np.mean(model_residual * model_residual)))
        model_max_error = float(np.max(np.abs(model_residual)))
        p3_value = float(P3_TARGET[dominant_joint])
        point_branches = np.asarray([row["branch"] for row in points])
        branch_models = {}
        for branch in ("up", "up_return", "down", "down_return"):
            mask = point_branches == branch
            branch_command = command[mask]
            branch_feedback = corrected[mask]
            branch_normalized = normalized[mask]
            branch_order = np.argsort(branch_normalized)
            branch_covered = bool(
                len(branch_command) >= 2
                and np.min(branch_normalized) <= p3_value <= np.max(branch_normalized)
            )
            branch_models[branch] = {
                "sample_count": int(len(branch_command)),
                "command_rad": finite_list(branch_command),
                "raw_unwrapped_feedback_rad": finite_list(branch_feedback),
                "normalized_urdf_feedback_rad": finite_list(branch_normalized),
                "rank_correlation": rank_correlation(branch_command, branch_normalized),
                "p3_covered": branch_covered,
                "p3_estimated_sdk_command_rad": (
                    float(
                        np.interp(
                            p3_value,
                            branch_normalized[branch_order],
                            branch_command[branch_order],
                        )
                    )
                    if branch_covered
                    else None
                ),
            }
        preferred_branch = "up" if p3_value >= float(command[0]) else "down"
        preferred_p3 = branch_models[preferred_branch]
        order = np.argsort(normalized_table)
        ys = normalized_table[order]
        xs = table_command[order]
        unique_y, unique_index = np.unique(ys, return_index=True)
        unique_x = xs[unique_index]
        covered = bool(unique_y[0] <= p3_value <= unique_y[-1])
        estimate = float(np.interp(p3_value, unique_y, unique_x)) if covered else None
        inverse_p3[str(dominant_joint)] = {
            "desired_feedback_rad": p3_value,
            "estimated_sdk_command_rad": estimate,
            "preferred_motion_branch": preferred_branch,
            "preferred_estimated_sdk_command_rad": preferred_p3[
                "p3_estimated_sdk_command_rad"
            ],
            "estimated_raw_feedback_rad": (
                float(np.interp(p3_value, normalized_table, table_feedback))
                if covered
                else None
            ),
            "covered": covered,
            "source_command_joint": joint,
        }
        channels.append(
            {
                "command_joint": joint,
                "command_joint_name": JOINT_NAMES[joint],
                "dominant_feedback_joint": dominant_joint,
                "dominant_feedback_name": JOINT_NAMES[dominant_joint],
                "direction_sign": sign,
                "feedback_offset_rad": float(calibrated["offset"]),
                "rank_correlation": corr,
                "piecewise_rank_correlation": table_corr,
                "sign_offset_rmse_rad": float(calibrated["rmse"]),
                "sign_offset_max_error_rad": float(calibrated["max_error"]),
                "raw_feedback_span_rad": float(feedback_span),
                "feedback_to_urdf_scale": feedback_to_urdf_scale,
                "piecewise_model_rmse_rad": model_rmse,
                "piecewise_model_max_error_rad": model_max_error,
                "max_hysteresis_rad": normalized_hysteresis,
                "response_range_rad": finite_list(response_range),
                "raw_feedback": raw.tolist(),
                "unwrapped_feedback": unwrapped.tolist(),
                "corrected_feedback_rad": finite_list(corrected),
                "normalized_urdf_feedback_rad": finite_list(normalized),
                "command_values_rad": finite_list(command),
                "piecewise_command_rad": finite_list(table_command),
                "piecewise_feedback_rad": finite_list(table_feedback),
                "piecewise_normalized_urdf_rad": finite_list(normalized_table),
                "branch_models": branch_models,
            }
        )
    mapped = [row["dominant_feedback_joint"] for row in channels]
    one_to_one = len(mapped) == len(set(mapped)) == len(channels)
    full_scan = len(channels) == 4
    channel_quality = all(
        abs(row["piecewise_rank_correlation"]) >= 0.95
        and abs(row["raw_feedback_span_rad"]) >= 0.10
        and row["piecewise_model_rmse_rad"] <= 0.12
        and row["max_hysteresis_rad"] <= 0.12
        for row in channels
    )
    segmented_quality = all(
        all(
            abs(model["rank_correlation"]) >= 0.95
            for model in row["branch_models"].values()
            if model["sample_count"] >= 3
            and np.ptp(np.asarray(model["command_rad"], dtype=np.float64)) >= 0.05
        )
        and row["branch_models"][
            "up"
            if P3_TARGET[row["dominant_feedback_joint"]]
            >= row["command_values_rad"][0]
            else "down"
        ]["p3_covered"]
        for row in channels
    )
    p3_covered = full_scan and len(inverse_p3) == 4 and all(
        x["covered"] for x in inverse_p3.values()
    )
    if not one_to_one:
        status = "FAIL_CHANNEL_ASSIGNMENT"
    elif not channel_quality and not segmented_quality:
        status = "FAIL_SIGN_OFFSET_OR_HYSTERESIS"
    elif segmented_quality and not channel_quality and not full_scan:
        status = "PASS_SEGMENTED_SINGLE_CHANNEL_REVIEW_REQUIRED"
    elif segmented_quality and not channel_quality and not p3_covered:
        status = "FAIL_P3_COVERAGE"
    elif segmented_quality and not channel_quality:
        status = "PASS_SEGMENTED_MAPPING_ONLY"
    elif not full_scan:
        status = "PASS_SINGLE_CHANNEL_REVIEW_REQUIRED"
    elif not p3_covered:
        status = "FAIL_P3_COVERAGE"
    else:
        status = "PASS_MAPPING_ONLY"
    return {
        "status": status,
        "one_to_one_channel_assignment": one_to_one,
        "channel_quality_pass": channel_quality,
        "segmented_channel_quality_pass": segmented_quality,
        "full_four_channel_scan": full_scan,
        "p3_feedback_covered": p3_covered,
        "channels": channels,
        "inverse_p3_estimate": inverse_p3,
        "note": (
            "Inverse P3 estimates are analysis output only and were not sent to hardware. "
            "A PASS_MAPPING_ONLY result still requires a separate static mapped-P3 gate."
        ),
    }


def render_report(result: dict) -> str:
    analysis = result.get("analysis", {})
    rows = []
    for item in analysis.get("channels", []):
        rows.append(
            f"| {item['command_joint']} | {item['dominant_feedback_joint']} | "
            f"{item['direction_sign']:+d} | {item['feedback_offset_rad']:+.4f} | "
            f"{item['piecewise_rank_correlation']:.4f} | "
            f"{item['piecewise_model_rmse_rad']:.4f} | {item['max_hysteresis_rad']:.4f} |"
        )
    return f"""# Dex5 S0.8.1 Thumb Global Mapping

## Status

```text
hardware: {result.get('hardware', {}).get('status', 'NOT_RUN')}
analysis: {analysis.get('status', 'NOT_RUN')}
P3 feedback covered: {analysis.get('p3_feedback_covered', False)}
PPO/student/teacher/replay: NOT RUN
```

| command joint | dominant feedback joint | sign | offset rad | rank correlation | RMSE rad | hysteresis rad |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows) if rows else '| - | - | - | - | - | - | - |'}

The scan records raw and unwrapped thumb feedback for every command point.
Estimated inverse P3 commands are not transmitted automatically.
"""


def render_p3_report(result: dict) -> str:
    gate = result.get("mapped_p3_gate", {})
    return f"""# Dex5 S0.8.1 Mapped P3 Static Gate

```text
status: {gate.get('status', 'NOT_RUN')}
max normalized thumb error: {gate.get('max_thumb_error_rad')}
non-thumb raw error max: {gate.get('non_thumb_raw_error_max_rad')}
trajectory replay: NOT RUN
```

The hand was empty. The four thumb targets used the preferred direction-aware
segments from the global mapping. Motors were disabled after measurement.
"""


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reanalyze-existing", type=Path)
    parser.add_argument("--merge-existing", type=Path, nargs="+")
    parser.add_argument("--stage-mapped-p3", type=Path)
    parser.add_argument(
        "--thumb-command-override",
        help="Four comma-separated SDK commands for a measured mapped-P3 correction.",
    )
    parser.add_argument("--confirm-empty-hand", choices=("NO", "YES"), default="NO")
    parser.add_argument("--arm-token", default="")
    parser.add_argument("--joints", default="16,17,18,19")
    parser.add_argument("--grid-spacing-rad", type=float, default=0.05)
    parser.add_argument("--command-speed-rad-s", type=float, default=0.08)
    parser.add_argument("--delta-cap-rad", type=float, default=0.01)
    parser.add_argument("--settle-s", type=float, default=0.15)
    parser.add_argument("--thermal-abort-c", type=float, default=75.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.joints = [int(x) for x in args.joints.split(",") if x.strip()]
    if args.thumb_command_override is not None:
        values = [float(x) for x in args.thumb_command_override.split(",")]
        if len(values) != 4 or not np.isfinite(values).all():
            parser.error("--thumb-command-override requires four finite values")
        args.thumb_command_override = np.asarray(values, dtype=np.float64)
    if not args.joints or any(x not in THUMB for x in args.joints):
        parser.error("--joints must contain only 16,17,18,19")
    if args.command_speed_rad_s > 0.08 or args.delta_cap_rad > 0.01:
        parser.error("command speed/delta exceed S0.9 acceptance caps")
    suffix = "all" if args.joints == list(THUMB) else "_".join(f"j{x}" for x in args.joints)
    if args.stage_mapped_p3 is not None and args.output is None:
        args.output = here / "mapped_p3_static_gate.json"
    elif args.output is None:
        args.output = here / f"thumb_global_mapping_{suffix}.json"
    if args.stage_mapped_p3 is not None and args.report is None:
        args.report = here / "Dex5_S0_8_1_Mapped_P3_Static_Gate.md"
    elif args.report is None:
        args.report = here / f"Dex5_S0_8_1_Thumb_Global_Mapping_{suffix}.md"
    safety = json.loads(
        (root / "real_replay_package" / "safety_limits.json").read_text(encoding="utf-8")
    )
    args.lower = np.asarray(safety["absolute_joint_lower_rad"], dtype=np.float64)
    args.upper = np.asarray(safety["absolute_joint_upper_rad"], dtype=np.float64)
    return args


def main() -> None:
    args = parse_args()
    if args.merge_existing:
        source_artifacts = []
        scans = []
        temperatures = []
        initial_feedback_by_scan = {}
        for path in args.merge_existing:
            payload = json.loads(path.read_text(encoding="utf-8"))
            hardware = payload.get("hardware", {})
            if hardware.get("status") != "SCAN_COMPLETED" or len(hardware.get("scans", [])) != 1:
                raise SystemExit(f"{path} is not a completed single-channel scan")
            scan = hardware["scans"][0]
            joint = int(scan["command_joint"])
            scans.append(scan)
            initial_feedback_by_scan[str(joint)] = hardware.get("initial_feedback")
            temperatures.extend(hardware.get("temperature_events", []))
            source_artifacts.append(
                {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "command_joint": joint,
                }
            )
        joints = sorted(int(scan["command_joint"]) for scan in scans)
        if joints != list(THUMB):
            raise SystemExit(f"merged scans must cover joints 16-19 exactly; got {joints}")
        hardware = {
            "status": "SCAN_COMPLETED",
            "initial_feedback_by_scan": initial_feedback_by_scan,
            "scans": sorted(scans, key=lambda row: int(row["command_joint"])),
            "temperature_events": temperatures,
            "software_estop_sent_after_each_scan": True,
        }
        result = {
            "schema": "dex5-s0.8.1-thumb-global-mapping-v2",
            "timestamp_unix_s": time.time(),
            "source_artifacts": source_artifacts,
            "hardware": hardware,
            "analysis": analyze_scan(hardware),
        }
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        args.report.write_text(render_report(result), encoding="utf-8")
        print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
        return
    if args.reanalyze_existing is not None:
        result = json.loads(args.reanalyze_existing.read_text(encoding="utf-8"))
        if result.get("hardware", {}).get("status") != "SCAN_COMPLETED":
            raise SystemExit("existing artifact does not contain a completed hardware scan")
        result["analysis"] = analyze_scan(result["hardware"])
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        args.report.write_text(render_report(result), encoding="utf-8")
        print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
        return
    if args.stage_mapped_p3 is not None:
        if not args.execute:
            raise SystemExit("mapped P3 static gate requires --execute")
        if args.confirm_empty_hand != "YES" or args.arm_token != "DEX5-S0.8.1-P3":
            raise SystemExit("mapped P3 gate requires empty-hand confirmation and P3 arm token")
        mapping = json.loads(args.stage_mapped_p3.read_text(encoding="utf-8"))
        if mapping.get("analysis", {}).get("status") not in {
            "PASS_SEGMENTED_MAPPING_ONLY", "PASS_MAPPING_ONLY"
        }:
            raise SystemExit("thumb mapping has not passed the four-channel gate")
        import rclpy

        result = {
            "schema": "dex5-s0.8.1-mapped-p3-static-gate-v1",
            "timestamp_unix_s": time.time(),
            "mapping_sha256": hashlib.sha256(args.stage_mapped_p3.read_bytes()).hexdigest(),
        }
        rclpy.init()
        runner = ThumbScanner(args)
        try:
            result["mapped_p3_gate"] = runner.stage_mapped_p3(mapping)
        except Exception as exc:
            runner.estop(True)
            result["mapped_p3_gate"] = {
                "status": "ABORTED", "error": str(exc), "software_estop_sent": True
            }
        finally:
            runner.estop(True)
            runner.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        args.report.write_text(render_p3_report(result), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if not args.execute:
        raise SystemExit("global thumb mapping requires --execute")
    if args.confirm_empty_hand != "YES" or args.arm_token != "DEX5-S0.8.1-THUMB":
        raise SystemExit("powered scan requires empty-hand confirmation and arm token")
    import rclpy

    result = {
        "schema": "dex5-s0.8.1-thumb-global-mapping-v1",
        "timestamp_unix_s": time.time(),
    }
    rclpy.init()
    runner = ThumbScanner(args)
    try:
        result["hardware"] = runner.run()
        result["analysis"] = analyze_scan(result["hardware"])
    except Exception as exc:
        runner.estop(True)
        result["hardware"] = {
            "status": "ABORTED",
            "error": str(exc),
            "software_estop_sent": True,
        }
        result["analysis"] = {"status": "NOT_RUN"}
    finally:
        runner.estop(True)
        runner.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
