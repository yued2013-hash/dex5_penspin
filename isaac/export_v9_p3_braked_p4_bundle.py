from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
P3_BANK = REPORT / "v9_p3_physx_reset_bank_v2.npz"
P4_BANK = REPORT / "v9_p4_physx_reset_bank.npz"
HOLD_REPORT = REPORT / "v9_p3_hold100_action_search.json"
OUTPUT = REPORT / "v9_p3_to_braked_p4_physx_bundle.npz"
ENTRY7_OUTPUT = REPORT / "v9_p3_to_entry7_physx_bundle.npz"
P4_HAND_TARGET_FRACTION = 0.625


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    ).astype(np.float32)


def yaw_rotate(quats: np.ndarray, degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) / 2.0
    yaw = np.zeros_like(quats)
    yaw[:, 0] = np.cos(half)
    yaw[:, 3] = np.sin(half)
    result = quat_mul(yaw, quats)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def main() -> None:
    with np.load(P3_BANK) as source:
        source_qpos = np.asarray(source["source_qpos"], dtype=np.float32)
        source_qvel = np.asarray(source["source_qvel"], dtype=np.float32)
        source_target = np.asarray(source["source_target_qpos"], dtype=np.float32)
        source_candidate_ids = np.asarray(
            source.get("source_candidate_ids", np.arange(len(source_qpos))), dtype=np.int64
        )
    with np.load(P4_BANK) as target:
        p4_qpos = np.asarray(target["source_qpos"], dtype=np.float32)
        p4_candidate_ids = np.asarray(
            target.get("source_candidate_ids", np.arange(len(p4_qpos))), dtype=np.int64
        )

    count = len(source_qpos)
    target_indices = np.arange(count) % len(p4_qpos)
    target_qpos = source_qpos.copy()
    target_qpos[:, :20] = source_qpos[:, :20] + P4_HAND_TARGET_FRACTION * (
        p4_qpos[target_indices, :20] - source_qpos[:, :20]
    )
    target_qpos[:, 20:23] = source_qpos[:, 20:23]
    target_qpos[:, 23:27] = yaw_rotate(source_qpos[:, 23:27], 50.0)

    hold = json.loads(HOLD_REPORT.read_text(encoding="utf-8"))["best"]
    first = np.asarray(hold["action"], dtype=np.float32)
    second = np.asarray(hold["second_action"], dtype=np.float32)
    teacher = np.repeat(second[None, None, :], count, axis=0)
    teacher = np.repeat(teacher, 8, axis=1)
    teacher[:, :2] = first

    payload = {
        "source_qpos": source_qpos,
        "source_qvel": source_qvel,
        "source_target_qpos": source_target,
        "source_policy_action": np.repeat(second[None, :], count, axis=0),
        "source_teacher_actions": teacher,
        "target_qpos": target_qpos,
        "target_qvel": np.zeros((count, 26), dtype=np.float32),
        "phase_ids": np.full(count, 3, dtype=np.int64),
        "source_episode_step": np.zeros(count, dtype=np.int64),
        "source_phase_offset_rad": np.zeros(count, dtype=np.float32),
        "target_remaining_rad": np.full(count, np.deg2rad(50.0), dtype=np.float32),
        "policy_target_pen_quat": source_qpos[:, 23:27].copy(),
        "source_contact_force_n": np.zeros((count, 5), dtype=np.float32),
        "curriculum_stage_deg": np.zeros(count, dtype=np.float32),
        "source_candidate_ids": source_candidate_ids,
        "target_p4_candidate_ids": p4_candidate_ids[target_indices],
    }
    np.savez_compressed(OUTPUT, **payload)
    metadata = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "source": str(P3_BANK.relative_to(ROOT)),
        "p4_hand_target_source": str(P4_BANK.relative_to(ROOT)),
        "rows": count,
        "unique_source_candidate_ids": int(np.unique(source_candidate_ids).size),
        "unique_p4_hand_targets": int(np.unique(p4_candidate_ids[target_indices]).size),
        "teacher_steps": 8,
        "teacher_switch_step": 2,
        "dynamic_target_phase_deg": 50.0,
        "planar_direction": "counter_clockwise_positive",
        "target_yaw_delta_deg": 50.0,
        "v9_geometric_p3_to_p4_phase_deg": 78.1817,
        "p4_hand_target_fraction": P4_HAND_TARGET_FRACTION,
        "p4_hand_target_equivalent_phase_deg": 78.1817 * P4_HAND_TARGET_FRACTION,
        "semantic_note": (
            "v9 P4 supplies the gait hand target; the dynamic pen target remains the "
            "50-degree lower edge of the braked 50-60 degree success band."
        ),
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))

    entry7 = {key: value.copy() for key, value in payload.items()}
    entry7_fraction = 7.0 / 78.1817
    entry7["target_qpos"][:, :20] = source_qpos[:, :20] + entry7_fraction * (
        p4_qpos[target_indices, :20] - source_qpos[:, :20]
    )
    entry7["target_qpos"][:, 23:27] = yaw_rotate(source_qpos[:, 23:27], 7.0)
    entry7["target_remaining_rad"][:] = np.deg2rad(7.0)
    np.savez_compressed(ENTRY7_OUTPUT, **entry7)
    entry7_metadata = {
        **metadata,
        "output": str(ENTRY7_OUTPUT.relative_to(ROOT)),
        "dynamic_target_phase_deg": 7.0,
        "target_yaw_delta_deg": 7.0,
        "p4_hand_target_fraction": entry7_fraction,
        "p4_hand_target_equivalent_phase_deg": 7.0,
        "semantic_note": "v9 P3 to 7-degree directional entry curriculum before braked-P4 training.",
    }
    ENTRY7_OUTPUT.with_suffix(".json").write_text(
        json.dumps(entry7_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(entry7_metadata, indent=2))


if __name__ == "__main__":
    main()
