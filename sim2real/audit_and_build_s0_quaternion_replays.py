#!/usr/bin/env python3
"""Audit S0 quaternion rotation and build speed-tier replays only after a strict gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


THUMB = (16, 17, 18, 19)
DEFAULT_SPEEDS = (0.12, 0.18, 0.24, 0.36)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, rw = left[0], right[0]
    lv, rv = left[1:], right[1:]
    return np.concatenate(
        ([lw * rw - np.dot(lv, rv)], lw * rv + rw * lv + np.cross(lv, rv))
    )


def quaternion_metrics(quaternions: np.ndarray) -> dict[str, float]:
    quat = quaternions.astype(np.float64)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    initial_conjugate = quat[0].copy()
    initial_conjugate[1:] *= -1.0
    relative_angle = []
    relative_world_z = []
    for row in quat:
        relative = quat_mul(row, initial_conjugate)
        relative /= np.linalg.norm(relative)
        if relative[0] < 0.0:
            relative *= -1.0
        angle = 2.0 * math.acos(float(np.clip(relative[0], -1.0, 1.0)))
        sine = math.sin(0.5 * angle)
        axis_z = float(relative[3] / sine) if abs(sine) > 1.0e-9 else 0.0
        relative_angle.append(angle)
        relative_world_z.append(angle * axis_z)
    increments = [
        2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0)))
        for a, b in zip(quat[:-1], quat[1:])
    ]
    degrees = 180.0 / math.pi
    return {
        "quaternion_max_geodesic_deg": float(max(relative_angle) * degrees),
        "quaternion_final_geodesic_deg": float(relative_angle[-1] * degrees),
        "quaternion_cumulative_step_deg": float(sum(increments) * degrees),
        "quaternion_max_positive_world_z_component_deg": float(
            max(relative_world_z) * degrees
        ),
        "quaternion_max_abs_world_z_component_deg": float(
            max(abs(value) for value in relative_world_z) * degrees
        ),
    }


def audit_dataset(dataset: Path, minimum_rotation_deg: float) -> tuple[list[dict], list[dict]]:
    with np.load(dataset, allow_pickle=False) as data:
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        pose = np.asarray(data["pen_pose"], dtype=np.float64)
        angular_velocity = np.asarray(data["pen_angular_velocity"], dtype=np.float64)
        contact = np.asarray(data["contact_state"], dtype=bool)
        splits = np.asarray(data["episode_split"])
        bank_ids = np.asarray(data["episode_bank_id"], dtype=np.int64)
        source_ids = np.asarray(data["episode_source_trajectory_id"], dtype=np.int64)

    rows = []
    for episode, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        episode_pose = pose[start:stop]
        episode_contact = contact[start:stop]
        metrics = quaternion_metrics(episode_pose[:, 3:7])
        integrated = np.cumsum(angular_velocity[start:stop, 2]) / 60.0
        position = episode_pose[:, :3]
        row = {
            "episode_id": episode,
            "split": str(splits[episode]),
            "bank_id": int(bank_ids[episode]),
            "source_trajectory_id": int(source_ids[episode]),
            "steps_60hz": int(stop - start),
            "duration_s": float((stop - start) / 60.0),
            **metrics,
            "integrated_world_z_peak_deg": float(
                np.max(integrated) * 180.0 / math.pi
            ),
            "pen_endpoint_drift_mm": float(
                np.linalg.norm(position[-1] - position[0]) * 1000.0
            ),
            "pen_height_loss_mm": float(
                max(0.0, position[0, 2] - np.min(position[:, 2])) * 1000.0
            ),
            "tim_contact_ratio": float(np.mean(np.all(episode_contact[:, :3], axis=1))),
            "forbidden_contact_ratio": float(
                np.mean(np.any(episode_contact[:, 3:], axis=1))
            ),
        }
        row["quaternion_gate_pass"] = bool(
            row["quaternion_max_geodesic_deg"] >= minimum_rotation_deg
            and row["quaternion_final_geodesic_deg"] >= minimum_rotation_deg
        )
        rows.append(row)
    ranked = sorted(
        rows,
        key=lambda row: (
            row["quaternion_gate_pass"],
            row["quaternion_final_geodesic_deg"],
            row["quaternion_max_geodesic_deg"],
        ),
        reverse=True,
    )
    return rows, ranked


def inverse_branch(channel: dict, desired: np.ndarray, branch: str = "up") -> np.ndarray:
    model = channel["branch_models"][branch]
    feedback = np.asarray(model["normalized_urdf_feedback_rad"], dtype=np.float64)
    command = np.asarray(model["command_rad"], dtype=np.float64)
    order = np.argsort(feedback)
    if float(np.min(desired)) < float(feedback[order][0]) or float(np.max(desired)) > float(
        feedback[order][-1]
    ):
        raise RuntimeError(f"thumb target is outside joint branch {branch}")
    return np.interp(desired, feedback[order], command[order])


def resample(rows: np.ndarray, source_hz: float, target_hz: float, duration_scale: float) -> np.ndarray:
    source_time = np.arange(len(rows), dtype=np.float64) / source_hz
    duration = source_time[-1] * duration_scale
    count = int(math.ceil(duration * target_hz)) + 1
    target_time = np.arange(count, dtype=np.float64) / target_hz
    source_query = np.minimum(target_time / duration_scale, source_time[-1])
    return np.stack(
        [np.interp(source_query, source_time, rows[:, joint]) for joint in range(rows.shape[1])],
        axis=-1,
    )


def mapped_command(
    virtual: np.ndarray, source_sdk: np.ndarray, mapping: dict, gate: dict
) -> tuple[np.ndarray, np.ndarray]:
    sdk = source_sdk.copy()
    channels = {
        int(row["dominant_feedback_joint"]): row
        for row in mapping["analysis"]["channels"]
    }
    preferred = np.asarray(
        [
            mapping["analysis"]["inverse_p3_estimate"][str(joint)][
                "preferred_estimated_sdk_command_rad"
            ]
            for joint in THUMB
        ]
    )
    accepted = np.asarray(gate["mapped_p3_gate"]["mapped_sdk_target"])[16:20]
    correction = accepted - preferred
    for local, joint in enumerate(THUMB):
        sdk[:, joint] = inverse_branch(channels[joint], virtual[:, joint]) + correction[local]
    sdk[0, 16:20] = accepted
    return sdk, correction


def build_tier(
    dataset: Path,
    episode_id: int,
    speed: float,
    mapping_path: Path,
    gate_path: Path,
    safety_path: Path,
    output_dir: Path,
    audit_metrics: dict,
) -> dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    with np.load(dataset, allow_pickle=False) as data:
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        start, stop = offsets[episode_id : episode_id + 2]
        source_sdk = np.asarray(data["sdk_absolute_target"][start:stop], dtype=np.float64)
        source_virtual = np.asarray(
            data["adapter_projected_target"][start:stop], dtype=np.float64
        )
    mapped_source, correction = mapped_command(
        source_virtual, source_sdk, mapping, gate
    )
    source_peak = float(np.max(np.abs(np.diff(mapped_source, axis=0))) * 60.0)
    duration_scale = max(1.0, source_peak / speed)
    for _ in range(4):
        sdk = resample(mapped_source, 60.0, 100.0, duration_scale)
        virtual = resample(source_virtual, 60.0, 100.0, duration_scale)
        peak = float(np.max(np.abs(np.diff(sdk, axis=0))) * 100.0)
        if peak <= speed + 1.0e-9:
            break
        duration_scale *= peak / speed * 1.000001
    lower = np.asarray(safety["absolute_joint_lower_rad"], dtype=np.float64)
    upper = np.asarray(safety["absolute_joint_upper_rad"], dtype=np.float64)
    if not np.all((sdk >= lower - 0.002) & (sdk <= upper + 0.002)):
        raise RuntimeError("generated tier exceeds audited URDF limits")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"episode_{episode_id}_mapped_{speed:.2f}rad_s.npz"
    np.savez_compressed(
        output,
        sdk_absolute_target_100hz=sdk.astype(np.float32),
        virtual_joint_target_100hz=virtual.astype(np.float32),
        trajectory_offsets=np.asarray([0, len(sdk)], dtype=np.int64),
        command_hz=np.asarray(100.0),
        source_episode_id=np.asarray([episode_id], dtype=np.int64),
        requested_speed_limit_rad_s=np.asarray(speed),
        quaternion_max_geodesic_deg=np.asarray(
            audit_metrics["quaternion_max_geodesic_deg"]
        ),
        quaternion_final_geodesic_deg=np.asarray(
            audit_metrics["quaternion_final_geodesic_deg"]
        ),
    )
    return {
        "speed_limit_rad_s": speed,
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "rows_100hz": int(len(sdk)),
        "duration_s": float(len(sdk) / 100.0),
        "maximum_command_speed_rad_s": peak,
        "duration_scale_vs_simulation": duration_scale,
        "thumb_local_correction_rad": correction.tolist(),
        "within_limits": True,
        "powered_replay_executed": False,
    }


def write_report(path: Path, summary: dict) -> None:
    top = summary["top_candidates"][:10]
    rows = "\n".join(
        f"| {row['episode_id']} | {row['split']} | {row['quaternion_max_geodesic_deg']:.2f} | "
        f"{row['quaternion_final_geodesic_deg']:.2f} | {row['integrated_world_z_peak_deg']:.2f} | "
        f"{row['quaternion_max_positive_world_z_component_deg']:.2f} |"
        for row in top
    )
    path.write_text(
        f"""# S0 Quaternion Replay Selection Audit

## 结论

```text
状态: {summary['status']}
轨迹总数: {summary['episode_count']}
四元数门槛: max和final均 >= {summary['minimum_rotation_deg']:.1f} deg
合格轨迹: {summary['eligible_count']}
真机速度包: {'GENERATED' if summary['generated_tiers'] else 'NOT GENERATED'}
```

## 排名前十

| episode | split | quaternion max deg | quaternion final deg | integrated omega-z deg | quaternion world-z component deg |
|---:|---|---:|---:|---:|---:|
{rows}

## 边界

`integrated omega-z` 不再单独构成成功证据。只有四元数实际姿态变化通过门槛，
才允许生成 0.12/0.18/0.24/0.36 rad/s 包。当前若为阻断状态，不得继续真机回放。
""",
        encoding="utf-8",
    )


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project / "Dex5_Sim2Real_S0/oracle_dataset/Dex5_S0_Oracle_Success_Rollouts_1200.npz",
    )
    parser.add_argument("--minimum-rotation-deg", type=float, default=30.0)
    parser.add_argument("--speeds", default=",".join(str(value) for value in DEFAULT_SPEEDS))
    parser.add_argument(
        "--mapping",
        type=Path,
        default=project / "Dex5_Sim2Real_S0/hardware_calibration_s0_8_1/thumb_global_mapping_all.json",
    )
    parser.add_argument(
        "--static-gate",
        type=Path,
        default=project / "Dex5_Sim2Real_S0/hardware_calibration_s0_8_1/mapped_p3_static_gate_joint16_corrected.json",
    )
    parser.add_argument(
        "--safety",
        type=Path,
        default=project / "Dex5_Sim2Real_S0/real_replay_package/safety_limits.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "Dex5_Sim2Real_S0/quaternion_replay_audit",
    )
    args = parser.parse_args()
    speeds = tuple(float(value) for value in args.speeds.split(","))
    if any(value <= 0.0 for value in speeds):
        parser.error("all speed tiers must be positive")
    rows, ranked = audit_dataset(args.dataset, args.minimum_rotation_deg)
    eligible = [row for row in ranked if row["quaternion_gate_pass"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "all_1200_quaternion_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tiers = []
    selected = eligible[0] if eligible else None
    if selected is not None:
        for speed in speeds:
            tiers.append(
                build_tier(
                    args.dataset,
                    selected["episode_id"],
                    speed,
                    args.mapping,
                    args.static_gate,
                    args.safety,
                    args.output_dir / "speed_tiers",
                    selected,
                )
            )
    summary = {
        "schema": "dex5-s0-quaternion-replay-audit-v1",
        "status": "PASS_TIERS_GENERATED" if selected else "BLOCKED_NO_QUALIFYING_TRAJECTORY",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "episode_count": len(rows),
        "minimum_rotation_deg": args.minimum_rotation_deg,
        "eligible_count": len(eligible),
        "counts": {
            "max_geodesic_at_least_30_deg": sum(
                row["quaternion_max_geodesic_deg"] >= 30.0 for row in rows
            ),
            "max_geodesic_at_least_45_deg": sum(
                row["quaternion_max_geodesic_deg"] >= 45.0 for row in rows
            ),
            "final_geodesic_at_least_30_deg": sum(
                row["quaternion_final_geodesic_deg"] >= 30.0 for row in rows
            ),
        },
        "selected_episode": selected,
        "top_candidates": ranked[:20],
        "requested_speed_tiers_rad_s": list(speeds),
        "generated_tiers": tiers,
        "metrics_csv": str(csv_path.resolve()),
        "powered_replay_authorized": False,
    }
    json_path = args.output_dir / "quaternion_replay_audit.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "Quaternion_Replay_Selection_Report.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
