from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
SOURCE = REPORT / "v9_p3_entry7_strict12_teacher200.npz"
P4_BANK = REPORT / "v9_p4_physx_reset_bank.npz"
OUTPUT = REPORT / "v9_entry7_to_approach20_physx_bundle.npz"
TEACHER_REPORT = REPORT / "v9_entry7_approach20_action_local4096_seed792.json"
TEACHER_CANDIDATE_ID = 3052
V9_GEOMETRIC_PHASE_DEG = 78.1817
TARGET_PHASE_DEG = 20.0
RESET_HAND_BLEND = 0.25


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


def yaw_rotate(quats: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    half = np.deg2rad(degrees) / 2.0
    yaw = np.zeros_like(quats)
    yaw[:, 0] = np.cos(half)
    yaw[:, 3] = np.sin(half)
    result = quat_mul(yaw, quats)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def main() -> None:
    with np.load(SOURCE) as data:
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        rows = np.arange(len(lengths))
        last = lengths - 1
        source_qpos = np.asarray(data["qpos"][rows, last], dtype=np.float32)
        source_qvel = np.asarray(data["qvel"][rows, last], dtype=np.float32)
        source_action = np.asarray(data["action"][rows, last], dtype=np.float32)
        source_target = np.asarray(
            data["actuator_target_qpos"][rows, last], dtype=np.float32
        )
        source_force = np.asarray(data["contact_force_n"][rows, last], dtype=np.float32)
        source_phase = np.asarray(data["phase_deg"][rows, last], dtype=np.float32)
        initial_qpos = np.asarray(data["initial_qpos"], dtype=np.float32)
    with np.load(P4_BANK) as data:
        p4_qpos = np.asarray(data["source_qpos"], dtype=np.float32)
        p4_ids = np.asarray(data.get("source_candidate_ids", np.arange(len(p4_qpos))))

    count = len(source_qpos)
    teacher_report = json.loads(TEACHER_REPORT.read_text(encoding="utf-8"))
    teacher_row = next(
        row for row in teacher_report["top"] if row["candidate_id"] == TEACHER_CANDIDATE_ID
    )
    drive_action = np.asarray(teacher_row["drive_action"], dtype=np.float32)
    brake_action = np.asarray(teacher_row["brake_action"], dtype=np.float32)
    teacher_actions = np.zeros((count, 24, 20), dtype=np.float32)
    teacher_actions[:, 6:16] = drive_action
    teacher_actions[:, 16:24] = brake_action
    target_ids = np.arange(count) % len(p4_qpos)
    hand_fraction = TARGET_PHASE_DEG / V9_GEOMETRIC_PHASE_DEG
    target_qpos = source_qpos.copy()
    target_qpos[:, :20] = initial_qpos[:, :20] + hand_fraction * (
        p4_qpos[target_ids, :20] - initial_qpos[:, :20]
    )
    target_qpos[:, 20:23] = source_qpos[:, 20:23]
    remaining_deg = TARGET_PHASE_DEG - source_phase
    target_qpos[:, 23:27] = yaw_rotate(source_qpos[:, 23:27], remaining_deg)

    projected_qpos = source_qpos.copy()
    projected_qpos[:, :20] = initial_qpos[:, :20] + RESET_HAND_BLEND * (
        source_qpos[:, :20] - initial_qpos[:, :20]
    )
    payload = {
        "source_qpos": projected_qpos,
        "source_measured_qpos": source_qpos,
        # PhysX contact warm-start impulses are not serializable across resets.
        # Keep the measured terminal velocity for audit, but project the reset to
        # a static state before validating it as the next curriculum entrance.
        "source_qvel": np.zeros_like(source_qvel),
        "source_measured_qvel": source_qvel,
        "source_target_qpos": projected_qpos[:, :20].copy(),
        "source_measured_target_qpos": source_target,
        "source_policy_action": np.zeros_like(source_action),
        "source_measured_policy_action": source_action,
        "source_teacher_actions": teacher_actions,
        "target_qpos": target_qpos,
        "target_qvel": np.zeros((count, 26), dtype=np.float32),
        "phase_ids": np.full(count, 3, dtype=np.int64),
        "source_episode_step": np.zeros(count, dtype=np.int64),
        "source_phase_offset_rad": np.deg2rad(source_phase).astype(np.float32),
        "target_remaining_rad": np.deg2rad(remaining_deg).astype(np.float32),
        "policy_target_pen_quat": source_qpos[:, 23:27].copy(),
        "source_contact_force_n": source_force,
        "curriculum_stage_deg": source_phase.copy(),
        "target_p4_candidate_ids": p4_ids[target_ids],
    }
    np.savez_compressed(OUTPUT, **payload)
    summary = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "source": str(SOURCE.relative_to(ROOT)),
        "rows": count,
        "source_phase_deg": {
            "min": float(source_phase.min()),
            "mean": float(source_phase.mean()),
            "max": float(source_phase.max()),
        },
        "target_phase_deg": TARGET_PHASE_DEG,
        "remaining_phase_deg": {
            "min": float(remaining_deg.min()),
            "mean": float(remaining_deg.mean()),
            "max": float(remaining_deg.max()),
        },
        "hand_target_fraction": hand_fraction,
        "reset_hand_blend_from_initial_to_measured": RESET_HAND_BLEND,
        "teacher_report": str(TEACHER_REPORT.relative_to(ROOT)),
        "teacher_candidate_id": TEACHER_CANDIDATE_ID,
        "teacher_schedule": "6 settle + 10 drive + 8 brake control steps",
        "planar_direction": "counter_clockwise_positive",
        "semantic_note": (
            "Strict v9 entry7 success geometries are projected to zero-velocity "
            "PhysX resets; measured terminal velocities are retained separately."
        ),
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
