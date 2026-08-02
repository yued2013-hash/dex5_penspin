from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface"
SOURCE = REPORT / "earlier_pre_ring_index_recapture_teacher_50.npz"
OUTPUT = REPORT / "isaac_earlier_to_p3_entry_bundle.npz"
BRIDGE_DEG = 7.0


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def main() -> None:
    with np.load(SOURCE, allow_pickle=False) as data:
        source = {key: np.asarray(data[key]) for key in data.files}
    rows = np.arange(source["lengths"].shape[0])
    lengths = source["lengths"].astype(np.int64)
    tip_dist = source["tip_surface_dist_m"].astype(np.float32)
    contact_force = source["contact_force_n"].astype(np.float32)
    selected_rows = []
    selected_steps = []
    for row, length in enumerate(lengths):
        valid = np.arange(length)
        # The collector records after env.step(), so a terminal step may already
        # contain the auto-reset state. Select the actual recapture frame instead
        # of assuming the final recorded frame is the useful bridge entry.
        valid = valid[(tip_dist[row, valid, 1] <= 0.014) & (contact_force[row, valid, 0] > 0.03)]
        if valid.size == 0:
            continue
        score = (
            2.0 * np.clip(contact_force[row, valid, 2] / 0.10, 0.0, 1.0)
            + np.clip(contact_force[row, valid, 0] / 0.30, 0.0, 1.0)
            + np.clip(contact_force[row, valid, 1] / 0.30, 0.0, 1.0)
            - 1.5 * np.clip(tip_dist[row, valid, 1] / 0.014, 0.0, 2.0)
        )
        selected_rows.append(row)
        selected_steps.append(int(valid[int(np.argmax(score))]))
    rows = np.asarray(selected_rows, dtype=np.int64)
    selected_steps = np.asarray(selected_steps, dtype=np.int64)
    source_qpos = source["qpos"][rows, selected_steps].astype(np.float32)
    source_qvel = source["qvel"][rows, selected_steps].astype(np.float32)
    source_hand = source_qpos[:, :20].copy()
    source_pen_quat = source_qpos[:, 23:27].copy()

    # Use a small negative yaw increment as the dynamic P3-entry target. The
    # hand target stays at the measured recapture endpoint; middle engagement
    # must be learned through the action, not a reset pose teleport.
    half = np.full(len(rows), -0.5 * np.deg2rad(BRIDGE_DEG), dtype=np.float32)
    yaw_delta = np.stack(
        (np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)), axis=-1
    )
    target_quat = quat_multiply(yaw_delta, source_pen_quat)
    target_quat /= np.linalg.norm(target_quat, axis=-1, keepdims=True).clip(1.0e-6)
    target_qpos = source_qpos.copy()
    target_qpos[:, :20] = source_hand
    target_qpos[:, 23:27] = target_quat

    teacher = source["actions"][rows].astype(np.float32)
    bundle = {
        "source_qpos": source_qpos,
        "source_qvel": source_qvel,
        "source_target_qpos": source_hand,
        "source_policy_action": np.zeros((len(rows), 20), dtype=np.float32),
        "source_episode_step": np.zeros(len(rows), dtype=np.int64),
        "source_teacher_actions": teacher,
        "target_qpos": target_qpos,
        "target_qvel": source_qvel,
        "phase_ids": np.full(len(rows), 3, dtype=np.int64),
        "source_phase_offset_rad": np.zeros(len(rows), dtype=np.float32),
        "target_remaining_rad": np.full(len(rows), np.deg2rad(BRIDGE_DEG), dtype=np.float32),
        "policy_target_pen_quat": target_quat.astype(np.float32),
        "source_trajectory_lengths": lengths[rows],
        "source_selected_steps": selected_steps,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **bundle)
    metadata = {
        "stage": "earlier index/thumb recapture -> P3 entry",
        "source": str(SOURCE.relative_to(ROOT)),
        "count": int(len(rows)),
        "bridge_phase_deg": BRIDGE_DEG,
        "reset_hand_target": "measured final recapture qpos",
        "required_contacts": ["thumb", "index", "middle"],
        "teacher_steps_per_row": int(teacher.shape[1]),
        "source_trajectory_lengths": lengths[rows].tolist(),
        "source_selected_steps": selected_steps.tolist(),
        "selection": "index tip <=14mm and thumb contact, then favor middle/index contact",
        "note": "The endpoint is an action-conditioned recapture state, not a canonical static P3 pose.",
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
