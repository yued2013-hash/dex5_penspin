#!/usr/bin/env python3
"""Build an offline S0.9 replay package using the accepted thumb mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


THUMB = (16, 17, 18, 19)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inverse_branch(channel: dict, desired: np.ndarray, branch: str) -> np.ndarray:
    model = channel["branch_models"][branch]
    feedback = np.asarray(model["normalized_urdf_feedback_rad"], dtype=np.float64)
    command = np.asarray(model["command_rad"], dtype=np.float64)
    order = np.argsort(feedback)
    feedback = feedback[order]
    command = command[order]
    if float(np.min(desired)) < float(feedback[0]) or float(np.max(desired)) > float(feedback[-1]):
        raise RuntimeError(
            f"desired range [{np.min(desired):.6f}, {np.max(desired):.6f}] is outside "
            f"the {branch} branch [{feedback[0]:.6f}, {feedback[-1]:.6f}]"
        )
    return np.interp(desired, feedback, command)


def rate_limit_segment(
    sdk: np.ndarray, virtual: np.ndarray, max_delta: float
) -> tuple[np.ndarray, np.ndarray]:
    sdk_rows = [sdk[0].copy()]
    virtual_rows = [virtual[0].copy()]
    for index in range(1, len(sdk)):
        delta = float(np.max(np.abs(sdk[index] - sdk[index - 1])))
        count = max(1, int(math.ceil(delta / max_delta)))
        for step in range(1, count + 1):
            alpha = step / count
            sdk_rows.append(sdk[index - 1] + alpha * (sdk[index] - sdk[index - 1]))
            virtual_rows.append(
                virtual[index - 1] + alpha * (virtual[index] - virtual[index - 1])
            )
    return np.stack(sdk_rows), np.stack(virtual_rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "hardware_calibration_s0_8/open_loop_commands_100hz_speed_safe.npz",
    )
    parser.add_argument(
        "--mapping", type=Path, default=Path(__file__).with_name("thumb_global_mapping_all.json")
    )
    parser.add_argument(
        "--static-gate",
        type=Path,
        default=Path(__file__).with_name("mapped_p3_static_gate_joint16_corrected.json"),
    )
    parser.add_argument(
        "--safety", type=Path, default=root / "real_replay_package/safety_limits.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "powered_replay_s0_9/mapped_replay_100hz.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "powered_replay_s0_9/mapped_replay_manifest.json",
    )
    parser.add_argument("--max-delta-rad", type=float, default=0.0008)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    gate = json.loads(args.static_gate.read_text(encoding="utf-8"))
    safety = json.loads(args.safety.read_text(encoding="utf-8"))
    if mapping["analysis"]["status"] != "PASS_SEGMENTED_MAPPING_ONLY":
        raise RuntimeError("global thumb mapping is not accepted")
    if gate["mapped_p3_gate"]["status"] != "PASS_THUMB_P3_STATIC_GATE":
        raise RuntimeError("mapped P3 static gate is not accepted")
    with np.load(args.source, allow_pickle=False) as data:
        source_sdk = np.asarray(data["sdk_absolute_target_100hz"], dtype=np.float64)
        source_virtual = np.asarray(data["virtual_joint_target_100hz"], dtype=np.float64)
        offsets = np.asarray(data["trajectory_offsets"], dtype=np.int64)
        hz = float(data["command_hz"])
    if not math.isclose(hz, 100.0, abs_tol=1.0e-9):
        raise RuntimeError("source replay is not 100 Hz")

    channels = {
        int(row["dominant_feedback_joint"]): row for row in mapping["analysis"]["channels"]
    }
    preferred = np.asarray(
        [
            mapping["analysis"]["inverse_p3_estimate"][str(joint)][
                "preferred_estimated_sdk_command_rad"
            ]
            for joint in THUMB
        ],
        dtype=np.float64,
    )
    accepted_p3 = np.asarray(gate["mapped_p3_gate"]["mapped_sdk_target"], dtype=np.float64)[16:20]
    local_correction = accepted_p3 - preferred

    sdk_segments = []
    virtual_segments = []
    new_offsets = [0]
    branch_usage = {str(joint): "up" for joint in THUMB}
    for start, stop in zip(offsets[:-1], offsets[1:]):
        virtual = source_virtual[start:stop].copy()
        sdk = source_sdk[start:stop].copy()
        for local, joint in enumerate(THUMB):
            sdk[:, joint] = (
                inverse_branch(channels[joint], virtual[:, joint], "up")
                + local_correction[local]
            )
        sdk[0, 16:20] = accepted_p3
        sdk, virtual = rate_limit_segment(sdk, virtual, args.max_delta_rad)
        sdk_segments.append(sdk)
        virtual_segments.append(virtual)
        new_offsets.append(new_offsets[-1] + len(sdk))

    sdk = np.concatenate(sdk_segments)
    virtual = np.concatenate(virtual_segments)
    new_offsets = np.asarray(new_offsets, dtype=np.int64)
    lower = np.asarray(safety["absolute_joint_lower_rad"], dtype=np.float64)
    upper = np.asarray(safety["absolute_joint_upper_rad"], dtype=np.float64)
    limit_ok = bool(np.all((sdk >= lower - 0.002) & (sdk <= upper + 0.002)))
    per_segment_delta = []
    for start, stop in zip(new_offsets[:-1], new_offsets[1:]):
        segment = sdk[start:stop]
        per_segment_delta.append(
            float(np.max(np.abs(np.diff(segment, axis=0)))) if len(segment) > 1 else 0.0
        )
    max_delta = max(per_segment_delta, default=0.0)
    if not limit_ok or max_delta > args.max_delta_rad + 1.0e-9:
        raise RuntimeError("mapped replay failed offline limit or rate gate")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sdk_absolute_target_100hz=sdk.astype(np.float32),
        virtual_joint_target_100hz=virtual.astype(np.float32),
        trajectory_offsets=new_offsets,
        command_hz=np.asarray(100.0, dtype=np.float64),
        source_sha256=np.asarray(sha256(args.source)),
        mapping_sha256=np.asarray(sha256(args.mapping)),
        static_gate_sha256=np.asarray(sha256(args.static_gate)),
    )
    manifest = {
        "schema": "dex5-s0.9-mapped-replay-v1",
        "status": "OFFLINE_PREFLIGHT_PASS_NOT_EXECUTED",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "source_sha256": sha256(args.source),
        "mapping_sha256": sha256(args.mapping),
        "static_gate_sha256": sha256(args.static_gate),
        "trajectory_count": int(len(new_offsets) - 1),
        "command_rows": int(len(sdk)),
        "command_hz": 100.0,
        "maximum_delta_rad": max_delta,
        "maximum_speed_rad_s": max_delta * 100.0,
        "within_limits": limit_ok,
        "thumb_branch_usage": branch_usage,
        "thumb_local_command_correction_rad": local_correction.tolist(),
        "mapped_p3_thumb_command_rad": accepted_p3.tolist(),
        "boundary": (
            "The first powered replay uses the validated P3 approach branch. "
            "This package does not validate reverse-branch tracking under load."
        ),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
