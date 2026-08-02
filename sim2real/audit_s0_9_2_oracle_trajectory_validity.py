#!/usr/bin/env python3
"""Offline S0.9.2 audit using actual quaternion motion as task evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CONTROL_HZ = 60.0
TARGET_DIRECTION = 1.0
TIER_A_MIN_ROTATION_DEG = 30.0
TIER_B_MIN_ROTATION_DEG = 15.0
MIN_AXIS_ALIGNMENT = 0.70
HOLD_STEPS = 9
MAX_HEIGHT_LOSS_M = 0.005
MAX_TARGET_SPEED_RAD_S = 2.5
MAX_WRONG_AXIS_SPEED_RAD_S = 2.0
MONOTONIC_STEP_TOLERANCE_DEG = 0.25
ANGLE_COMPARISON_EPS_DEG = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[0] * right[0] - np.dot(left[1:], right[1:])
    vector = (
        left[0] * right[1:]
        + right[0] * left[1:]
        + np.cross(left[1:], right[1:])
    )
    return np.concatenate(([scalar], vector))


def quaternion_progress(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return geodesic angle and signed world-z rotvec component in degrees."""
    quat = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("zero-norm quaternion")
    quat /= norms
    initial_conjugate = quat[0].copy()
    initial_conjugate[1:] *= -1.0
    geodesic = np.empty(len(quat), dtype=np.float64)
    target_component = np.empty(len(quat), dtype=np.float64)
    for index, row in enumerate(quat):
        relative = quat_mul(row, initial_conjugate)
        relative /= np.linalg.norm(relative)
        if relative[0] < 0.0:
            relative *= -1.0
        angle = 2.0 * math.acos(float(np.clip(relative[0], -1.0, 1.0)))
        sine = math.sin(0.5 * angle)
        axis_z = float(relative[3] / sine) if abs(sine) > 1.0e-9 else 0.0
        geodesic[index] = math.degrees(angle)
        target_component[index] = math.degrees(angle * axis_z) * TARGET_DIRECTION
    return geodesic, target_component


def quaternion_derived_angular_velocity(quaternions: np.ndarray) -> np.ndarray:
    """Infer world-frame angular velocity between recorded 60 Hz poses."""
    quat = np.asarray(quaternions, dtype=np.float64).copy()
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    rows = []
    for first, second in zip(quat[:-1], quat[1:]):
        conjugate = first.copy()
        conjugate[1:] *= -1.0
        relative = quat_mul(second, conjugate)
        relative /= np.linalg.norm(relative)
        if relative[0] < 0.0:
            relative *= -1.0
        angle = 2.0 * math.acos(float(np.clip(relative[0], -1.0, 1.0)))
        sine = math.sin(0.5 * angle)
        axis = relative[1:] / sine if abs(sine) > 1.0e-9 else np.zeros(3)
        rows.append(axis * angle * CONTROL_HZ)
    return np.asarray(rows, dtype=np.float64)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) < 1.0e-9 or np.std(second) < 1.0e-9:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def episode_metrics(
    pose: np.ndarray,
    angular_velocity: np.ndarray,
    contact: np.ndarray,
) -> dict[str, float | int | bool | str]:
    geodesic, target_progress = quaternion_progress(pose[:, 3:7])
    final_geodesic = float(geodesic[-1])
    final_target = float(target_progress[-1])
    final_alignment = final_target / max(final_geodesic, 1.0e-9)
    max_target = float(np.max(target_progress))
    max_target_index = int(np.argmax(target_progress))
    max_geodesic = float(np.max(geodesic))

    target_deltas = np.diff(target_progress)
    nonnegative_ratio = float(
        np.mean(target_deltas >= -MONOTONIC_STEP_TOLERANCE_DEG)
    ) if target_deltas.size else 1.0
    total_variation = float(np.sum(np.abs(target_deltas)))
    monotonic_efficiency = max(0.0, final_target) / max(total_variation, 1.0e-9)
    running_peak = np.maximum.accumulate(target_progress)
    maximum_backtrack = float(np.max(running_peak - target_progress))

    integrated = np.cumsum(
        TARGET_DIRECTION * angular_velocity[:, 2]
    ) / CONTROL_HZ
    integrated_peak = float(math.degrees(np.max(integrated)))
    derived_angular_velocity = quaternion_derived_angular_velocity(pose[:, 3:7])
    reported_for_steps = angular_velocity[:-1]
    omega_error = reported_for_steps - derived_angular_velocity
    reported_path = float(
        np.sum(np.linalg.norm(reported_for_steps, axis=1)) / CONTROL_HZ
    )
    derived_path = float(
        np.sum(np.linalg.norm(derived_angular_velocity, axis=1)) / CONTROL_HZ
    )
    height_loss = pose[0, 2] - pose[:, 2]
    target_speed = np.abs(TARGET_DIRECTION * angular_velocity[:, 2])
    wrong_axis_speed = np.linalg.norm(angular_velocity[:, :2], axis=1)
    tim_all = np.all(contact[:, :3], axis=1)
    forbidden_clear = ~np.any(contact[:, 3:], axis=1)

    instantaneous_alignment = target_progress / np.maximum(geodesic, 1.0e-9)
    strict_hold = (
        (geodesic + ANGLE_COMPARISON_EPS_DEG >= TIER_A_MIN_ROTATION_DEG)
        & (target_progress >= 0.0)
        & (instantaneous_alignment >= MIN_AXIS_ALIGNMENT)
        & (height_loss <= MAX_HEIGHT_LOSS_M)
        & (target_speed <= MAX_TARGET_SPEED_RAD_S)
        & (wrong_axis_speed <= MAX_WRONG_AXIS_SPEED_RAD_S)
        & tim_all
        & forbidden_clear
    )
    max_hold_steps = longest_true_run(strict_hold)
    hold_pass = max_hold_steps >= HOLD_STEPS
    tier_a = bool(
        final_geodesic + ANGLE_COMPARISON_EPS_DEG >= TIER_A_MIN_ROTATION_DEG
        and final_target >= 0.0
        and final_alignment >= MIN_AXIS_ALIGNMENT
        and hold_pass
    )
    if tier_a:
        tier = "A_TRANSFER_CANDIDATE"
    elif (
        final_geodesic + ANGLE_COMPARISON_EPS_DEG >= TIER_B_MIN_ROTATION_DEG
        and final_geodesic + ANGLE_COMPARISON_EPS_DEG < TIER_A_MIN_ROTATION_DEG
    ):
        tier = "B_PARTIAL_15_TO_30_DEG"
    else:
        tier = "C_FALSE_ROTATION"

    return {
        "tier": tier,
        "tier_a_pass": tier_a,
        "quaternion_max_geodesic_deg": max_geodesic,
        "quaternion_final_geodesic_deg": final_geodesic,
        "target_axis_max_component_deg": max_target,
        "target_axis_final_component_deg": final_target,
        "target_axis_final_alignment": float(final_alignment),
        "target_axis_max_component_step": max_target_index,
        "integrated_target_axis_peak_deg": integrated_peak,
        "integral_minus_quaternion_target_deg": integrated_peak - max_target,
        "reported_vs_quaternion_omega_rmse_rad_s": float(
            np.sqrt(np.mean(np.square(omega_error)))
        ),
        "reported_vs_quaternion_omega_z_correlation": correlation(
            reported_for_steps[:, 2], derived_angular_velocity[:, 2]
        ),
        "reported_omega_path_rad": reported_path,
        "quaternion_derived_path_rad": derived_path,
        "quaternion_to_reported_path_ratio": derived_path / max(reported_path, 1.0e-9),
        "monotonic_nonnegative_step_ratio": nonnegative_ratio,
        "monotonic_efficiency": float(monotonic_efficiency),
        "maximum_target_axis_backtrack_deg": maximum_backtrack,
        "maximum_height_loss_mm": float(max(0.0, np.max(height_loss)) * 1000.0),
        "endpoint_position_drift_mm": float(
            np.linalg.norm(pose[-1, :3] - pose[0, :3]) * 1000.0
        ),
        "tim_contact_ratio": float(np.mean(tim_all)),
        "forbidden_contact_ratio": float(np.mean(~forbidden_clear)),
        "max_strict_hold_steps": max_hold_steps,
        "max_strict_hold_ms": float(max_hold_steps / CONTROL_HZ * 1000.0),
        "strict_hold_150ms_pass": hold_pass,
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def write_report(path: Path, result: dict) -> None:
    rows = "\n".join(
        "| {episode_id} | {tier} | {quaternion_final_geodesic_deg:.2f} | "
        "{target_axis_final_component_deg:.2f} | {target_axis_final_alignment:.3f} | "
        "{integrated_target_axis_peak_deg:.2f} | {monotonic_efficiency:.3f} | "
        "{max_strict_hold_ms:.1f} |".format(**row)
        for row in result["top_ranked"][:15]
    )
    counts = result["tier_counts"]
    statistics = result["summary_statistics"]
    path.write_text(
        f"""# Dex5 S0.9.2 Oracle Trajectory Validity Audit

## 结论

```text
状态: {result['status']}
总轨迹: {result['episode_count']}
Tier A: {counts['A_TRANSFER_CANDIDATE']}
Tier B: {counts['B_PARTIAL_15_TO_30_DEG']}
Tier C: {counts['C_FALSE_ROTATION']}
Student dataset: {result['student_dataset_status']}
真机回放: BLOCKED
```

旧 success flag 只证明角速度积分、接触及保持门通过，不再作为物体旋转成功证据。
本审计只读原始 NPZ，未修改 O0、dataset、reward、policy 或真机状态。

此外，相邻四元数反推的角速度与数据集记录角速度缺乏一致性。新 rollout 在进入
任务筛选前，必须先通过 pose/velocity 运动学一致性检查。

## 核心统计

| 指标 | mean | p95 | max |
|---|---:|---:|---:|
| quaternion final deg | {statistics['quaternion_final_geodesic_deg']['mean']:.2f} | {statistics['quaternion_final_geodesic_deg']['p95']:.2f} | {statistics['quaternion_final_geodesic_deg']['max']:.2f} |
| target-axis quaternion deg | {statistics['target_axis_final_component_deg']['mean']:.2f} | {statistics['target_axis_final_component_deg']['p95']:.2f} | {statistics['target_axis_final_component_deg']['max']:.2f} |
| integrated omega-z deg | {statistics['integrated_target_axis_peak_deg']['mean']:.2f} | {statistics['integrated_target_axis_peak_deg']['p95']:.2f} | {statistics['integrated_target_axis_peak_deg']['max']:.2f} |
| pose-derived/reported omega-z correlation | {statistics['reported_vs_quaternion_omega_z_correlation']['mean']:.3f} | {statistics['reported_vs_quaternion_omega_z_correlation']['p95']:.3f} | {statistics['reported_vs_quaternion_omega_z_correlation']['max']:.3f} |
| quaternion/reported angular path ratio | {statistics['quaternion_to_reported_path_ratio']['mean']:.3f} | {statistics['quaternion_to_reported_path_ratio']['p95']:.3f} | {statistics['quaternion_to_reported_path_ratio']['max']:.3f} |

## 分层合同

- Tier A：最终四元数姿态变化 >=30 deg，目标轴一致性 >=0.70，并满足连续 150 ms
  的高度、速度、TIM 接触和 ring/pinky clear 门。
- Tier B：最终四元数姿态变化为 15-30 deg，仅用于课程或诊断，不得进入 student 成功集。
- Tier C：其余旧 success，包括高角速度积分但实际姿态变化不足的 false positive。

## 排名前十五

| episode | tier | net quaternion deg | target-axis deg | axis alignment | integrated omega deg | monotonic efficiency | strict hold ms |
|---:|---|---:|---:|---:|---:|---:|---:|
{rows}

## 判定

Tier A 数量不足，因此 student 训练、四档速度包和后续 powered replay 全部保持阻断。
下一步应先修正仿真任务中的旋转状态与成功判据，再采集新的 oracle rollout；不能从
现有 Tier B/C 数据中放宽门槛凑成功样本。
""",
        encoding="utf-8",
    )


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project
        / "Dex5_Sim2Real_S0/oracle_dataset/Dex5_S0_Oracle_Success_Rollouts_1200.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "Dex5_Sim2Real_S0/oracle_validity_audit_s0_9_2",
    )
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        pose = np.asarray(data["pen_pose"], dtype=np.float64)
        angular_velocity = np.asarray(data["pen_angular_velocity"], dtype=np.float64)
        contact = np.asarray(data["contact_state"], dtype=bool)
        splits = np.asarray(data["episode_split"])
        bank_ids = np.asarray(data["episode_bank_id"], dtype=np.int64)
        state_ids = np.asarray(data["episode_state_id"], dtype=np.int64)
        source_ids = np.asarray(data["episode_source_trajectory_id"], dtype=np.int64)

    rows: list[dict] = []
    for episode_id, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        row = {
            "episode_id": episode_id,
            "split": str(splits[episode_id]),
            "bank_id": int(bank_ids[episode_id]),
            "state_id": int(state_ids[episode_id]),
            "source_trajectory_id": int(source_ids[episode_id]),
            "steps_60hz": int(stop - start),
            "duration_s": float((stop - start) / CONTROL_HZ),
        }
        row.update(
            episode_metrics(
                pose[start:stop], angular_velocity[start:stop], contact[start:stop]
            )
        )
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda row: (
            row["tier_a_pass"],
            row["quaternion_final_geodesic_deg"],
            row["target_axis_final_alignment"],
            row["max_strict_hold_steps"],
        ),
        reverse=True,
    )
    tier_names = (
        "A_TRANSFER_CANDIDATE",
        "B_PARTIAL_15_TO_30_DEG",
        "C_FALSE_ROTATION",
    )
    tier_counts = {
        tier: sum(row["tier"] == tier for row in rows) for tier in tier_names
    }
    candidates = [row for row in ranked if row["tier_a_pass"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "trajectory_validity_metrics_1200.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema": "dex5-s0.9.2-student-dataset-candidate-manifest-v1",
        "status": "PASS" if candidates else "BLOCKED_NO_TIER_A",
        "source_dataset": str(args.dataset.resolve()),
        "source_dataset_sha256": sha256(args.dataset),
        "candidate_count": len(candidates),
        "candidate_episode_ids": [row["episode_id"] for row in candidates],
        "candidates": candidates,
        "student_training_authorized": bool(candidates),
    }
    manifest_path = args.output_dir / "student_dataset_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    result = {
        "schema": "dex5-s0.9.2-oracle-trajectory-validity-audit-v1",
        "status": "PASS_TIER_A_AVAILABLE" if candidates else "BLOCKED_NO_TIER_A",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "episode_count": len(rows),
        "thresholds": {
            "tier_a_min_quaternion_rotation_deg": TIER_A_MIN_ROTATION_DEG,
            "tier_b_min_quaternion_rotation_deg": TIER_B_MIN_ROTATION_DEG,
            "minimum_target_axis_alignment": MIN_AXIS_ALIGNMENT,
            "strict_hold_steps": HOLD_STEPS,
            "strict_hold_ms": HOLD_STEPS / CONTROL_HZ * 1000.0,
            "maximum_height_loss_mm": MAX_HEIGHT_LOSS_M * 1000.0,
            "maximum_target_speed_rad_s": MAX_TARGET_SPEED_RAD_S,
            "maximum_wrong_axis_speed_rad_s": MAX_WRONG_AXIS_SPEED_RAD_S,
        },
        "tier_counts": tier_counts,
        "student_dataset_status": manifest["status"],
        "summary_statistics": {
            "quaternion_final_geodesic_deg": summarize(
                [row["quaternion_final_geodesic_deg"] for row in rows]
            ),
            "target_axis_final_component_deg": summarize(
                [row["target_axis_final_component_deg"] for row in rows]
            ),
            "integrated_target_axis_peak_deg": summarize(
                [row["integrated_target_axis_peak_deg"] for row in rows]
            ),
            "monotonic_efficiency": summarize(
                [row["monotonic_efficiency"] for row in rows]
            ),
            "reported_vs_quaternion_omega_rmse_rad_s": summarize(
                [row["reported_vs_quaternion_omega_rmse_rad_s"] for row in rows]
            ),
            "reported_vs_quaternion_omega_z_correlation": summarize(
                [row["reported_vs_quaternion_omega_z_correlation"] for row in rows]
            ),
            "quaternion_to_reported_path_ratio": summarize(
                [row["quaternion_to_reported_path_ratio"] for row in rows]
            ),
            "max_strict_hold_ms": summarize(
                [row["max_strict_hold_ms"] for row in rows]
            ),
        },
        "top_ranked": ranked[:30],
        "metrics_csv": str(metrics_path.resolve()),
        "candidate_manifest": str(manifest_path.resolve()),
        "student_training_authorized": bool(candidates),
        "powered_replay_authorized": False,
    }
    result_path = args.output_dir / "success_statistics.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "Dex5_S0_9_2_Oracle_Validity_Audit.md", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
