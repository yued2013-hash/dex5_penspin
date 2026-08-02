from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
SOURCES = (
    (0.0, "isaac_canonical_p3_to_p4_braked55_bundle.npz"),
    (7.0, "isaac_canonical_p3_entry7_to_p4_entry12_bundle.npz"),
    (12.0, "isaac_canonical_entry12_to_p4_approach20_bundle.npz"),
    (20.0, "isaac_canonical_entry20_to_p4_midway35_bundle.npz"),
    (35.0, "isaac_canonical_entry35_to_p4_approach45_bundle.npz"),
    (45.0, "isaac_canonical_entry45_to_p4_braked50_bundle.npz"),
)


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float32,
    )


def yaw_rotate(quat: np.ndarray, degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) / 2.0
    yaw = np.asarray((np.cos(half), 0.0, 0.0, np.sin(half)), dtype=np.float32)
    result = quat_mul(yaw, quat)
    return result / np.linalg.norm(result)


def build(
    name: str,
    included_stages: set[float],
    use_final_hand_target: bool = False,
    stage_repeats: dict[float, int] | None = None,
) -> None:
    rows: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "source_qpos",
            "source_qvel",
            "source_target_qpos",
            "source_policy_action",
            "target_qpos",
            "target_qvel",
            "phase_ids",
            "source_phase_offset_rad",
            "target_remaining_rad",
            "policy_target_pen_quat",
            "source_contact_force_n",
            "curriculum_stage_deg",
        )
    }
    canonical = np.load(REPORT / SOURCES[0][1])
    canonical_p3_quat = np.asarray(canonical["source_qpos"][0, 23:27], dtype=np.float32)
    final_quat = yaw_rotate(canonical_p3_quat, -50.0)
    with np.load(REPORT / "phase_capture50_successful_teacher50.npz") as final_teacher:
        final_lengths = np.asarray(final_teacher["lengths"], dtype=np.int64)
        final_hand_targets = np.stack(
            [
                final_teacher["qpos"][index, length - 1, :20]
                for index, length in enumerate(final_lengths)
            ]
        ).astype(np.float32)
    included: list[float] = []
    for stage_deg, filename in SOURCES:
        if stage_deg not in included_stages:
            continue
        with np.load(REPORT / filename) as data:
            count = min(50, len(data["source_qpos"]))
            indices = np.linspace(0, len(data["source_qpos"]) - 1, count, dtype=np.int64)
            source_qpos = np.asarray(data["source_qpos"][indices], dtype=np.float32)
            target_qpos = source_qpos.copy()
            if use_final_hand_target:
                target_qpos[:, :20] = final_hand_targets[:count]
            target_qpos[:, 23:27] = final_quat
            rows["source_qpos"].append(source_qpos)
            rows["source_qvel"].append(np.asarray(data["source_qvel"][indices], dtype=np.float32))
            rows["source_target_qpos"].append(
                np.asarray(data["source_target_qpos"][indices], dtype=np.float32)
            )
            rows["source_policy_action"].append(
                np.asarray(data["source_policy_action"][indices], dtype=np.float32)
            )
            rows["target_qpos"].append(target_qpos)
            rows["target_qvel"].append(np.zeros((count, 26), dtype=np.float32))
            rows["phase_ids"].append(np.full(count, 3, dtype=np.int64))
            source_phase = np.asarray(
                data.get("source_phase_offset_rad", np.full(len(data["source_qpos"]), np.deg2rad(stage_deg)))[
                    indices
                ],
                dtype=np.float32,
            )
            rows["source_phase_offset_rad"].append(source_phase)
            rows["target_remaining_rad"].append(np.deg2rad(50.0).astype(np.float32) - source_phase)
            rows["policy_target_pen_quat"].append(source_qpos[:, 23:27])
            rows["source_contact_force_n"].append(
                np.asarray(data.get("source_contact_force_n", np.zeros((len(source_qpos), 5)))[indices], dtype=np.float32)
                if "source_contact_force_n" in data
                else np.zeros((count, 5), dtype=np.float32)
            )
            rows["curriculum_stage_deg"].append(np.full(count, stage_deg, dtype=np.float32))
            repeat = (stage_repeats or {}).get(stage_deg, 1)
            if repeat > 1:
                for key in rows:
                    rows[key].extend([rows[key][-1].copy() for _ in range(repeat - 1)])
            included.append(stage_deg)
    payload = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
    count = len(payload["source_qpos"])
    payload["source_episode_step"] = np.zeros(count, dtype=np.int64)
    payload["source_teacher_actions"] = np.zeros((count, 0, 20), dtype=np.float32)
    output = REPORT / f"isaac_unified_p3_to_braked_p4_{name}_bundle.npz"
    np.savez_compressed(output, **payload)
    metadata = {
        "output": str(output.relative_to(ROOT)),
        "stages_deg": included,
        "rows_per_stage": 50,
        "stage_repeats": stage_repeats or {},
        "total_rows": count,
        "target_phase_deg": 50.0,
        "target_hand": "validated final P4 capture" if use_final_hand_target else "source capture",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


build("late", {35.0, 45.0})
build("mid", {20.0, 35.0, 45.0})
build("focus20", {20.0})
build("focus12", {12.0})
build("early12", {12.0, 20.0, 35.0, 45.0})
build("early7", {7.0, 12.0, 20.0, 35.0, 45.0})
build("all", {0.0, 7.0, 12.0, 20.0, 35.0, 45.0})
build("focus0_gait", {0.0}, use_final_hand_target=True)
build("focus12_gait", {12.0}, use_final_hand_target=True)
build("early12_gait", {12.0, 20.0, 35.0, 45.0}, use_final_hand_target=True)
build("early7_gait", {7.0, 12.0, 20.0, 35.0, 45.0}, use_final_hand_target=True)
build("all_gait", {0.0, 7.0, 12.0, 20.0, 35.0, 45.0}, use_final_hand_target=True)
build(
    "all_gait_focus0x3",
    {0.0, 7.0, 12.0, 20.0, 35.0, 45.0},
    use_final_hand_target=True,
    stage_repeats={0.0: 3},
)
