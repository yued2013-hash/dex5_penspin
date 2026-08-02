from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface"


def nearest_rows(current_hand: np.ndarray, source_hand: np.ndarray) -> np.ndarray:
    result = np.empty(len(current_hand), dtype=np.int64)
    for start in range(0, len(current_hand), 64):
        chunk = current_hand[start : start + 64]
        distance = np.mean((chunk[:, None, :] - source_hand[None, :, :]) ** 2, axis=-1)
        result[start : start + len(chunk)] = np.argmin(distance, axis=1)
    return result


def yaw_rotate(quat: np.ndarray, degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) * 0.5
    yaw = np.zeros_like(quat)
    yaw[:, 0] = np.cos(half)
    yaw[:, 3] = np.sin(half)
    w1, x1, y1, z1 = yaw.T
    w2, x2, y2, z2 = quat.T
    result = np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )
    return result / np.linalg.norm(result, axis=-1, keepdims=True).clip(1.0e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=REPORT_DIR / "v9_current_approach49_continuous_teacher250.npz",
    )
    parser.add_argument(
        "--stage1-bundle",
        type=Path,
        default=REPORT_DIR / "v9_approach40_to_entry45_lowpen136_physx_bundle.npz",
    )
    parser.add_argument(
        "--approach-bundle",
        type=Path,
        default=REPORT_DIR / "v9_current_upstream45_to_approach49_bundle.npz",
    )
    parser.add_argument(
        "--final-bundle",
        type=Path,
        default=REPORT_DIR / "v9_entry45_to_braked50_relaxed50ms_lowpen197_physx_bundle.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "v9_current_approach49_prefill_to_braked50_bundle.npz",
    )
    args = parser.parse_args()

    with np.load(args.trajectories) as data:
        trajectories = {name: data[name] for name in data.files}
    with np.load(args.stage1_bundle) as data:
        stage1 = {name: data[name] for name in data.files}
    with np.load(args.approach_bundle) as data:
        approach = {name: data[name] for name in data.files}
    with np.load(args.final_bundle) as data:
        final = {name: data[name] for name in data.files}

    rows = []
    for trajectory_id, length in enumerate(trajectories["lengths"]):
        stages = trajectories["stage_id"][trajectory_id, :length]
        stage1_steps = np.flatnonzero(stages == 6)
        approach_steps = np.flatnonzero(stages == 7)
        final_steps = np.flatnonzero(stages == 8)
        if len(stage1_steps) == 0 or len(approach_steps) == 0 or len(final_steps) == 0:
            continue
        start = int(stage1_steps[0])
        approach_start = int(approach_steps[0])
        end = int(final_steps[0])
        if not (start < approach_start < end):
            continue
        rows.append((trajectory_id, start, approach_start, end))

    source_qpos = np.stack([trajectories["qpos"][i, start] for i, start, _, _ in rows])
    source_qvel = np.stack([trajectories["qvel"][i, start] for i, start, _, _ in rows])
    source_contact = np.stack(
        [trajectories["contact_force_n"][i, start] for i, start, _, _ in rows]
    )
    source_phase_deg = np.asarray(
        [trajectories["phase_deg"][i, start] for i, start, _, _ in rows], dtype=np.float32
    )
    approach_start_qpos = np.stack(
        [trajectories["qpos"][i, approach_start] for i, _, approach_start, _ in rows]
    )
    approach_end_qpos = np.stack([trajectories["qpos"][i, end] for i, _, _, end in rows])
    teacher_stage1_lengths = np.asarray(
        [approach_start - start for _, start, approach_start, _ in rows], dtype=np.int64
    )
    teacher_stage2_lengths = np.asarray(
        [end - approach_start for _, _, approach_start, end in rows], dtype=np.int64
    )
    teacher_lengths = teacher_stage1_lengths + teacher_stage2_lengths
    teacher_horizon = int(teacher_lengths.max())
    teacher_actions = np.zeros((len(rows), teacher_horizon, 20), dtype=np.float32)
    for row_id, (trajectory_id, start, _, end) in enumerate(rows):
        actions = trajectories["action"][trajectory_id, start:end]
        teacher_actions[row_id, : len(actions)] = actions
        teacher_actions[row_id, len(actions) :] = actions[-1]

    stage1_ids = nearest_rows(source_qpos[:, :20], stage1["source_qpos"][:, :20])
    approach_ids = nearest_rows(
        approach_start_qpos[:, :20], approach["source_qpos"][:, :20]
    )
    final_ids = nearest_rows(approach_end_qpos[:, :20], final["source_qpos"][:, :20])
    target_qpos = final["target_qpos"][final_ids].copy()
    target_qpos[:, 20:23] = approach_end_qpos[:, 20:23] + (
        final["target_qpos"][final_ids, 20:23] - final["source_qpos"][final_ids, 20:23]
    )
    target_qpos[:, 23:27] = yaw_rotate(trajectories["initial_qpos"][:, 23:27], 50.0)

    payload = {
        "source_qpos": source_qpos.astype(np.float32),
        "source_qvel": source_qvel.astype(np.float32),
        "source_target_qpos": stage1["source_target_qpos"][stage1_ids].astype(np.float32),
        "source_policy_action": stage1["source_policy_action"][stage1_ids].astype(np.float32),
        "source_contact_force_n": source_contact.astype(np.float32),
        "source_teacher_actions": teacher_actions,
        "teacher_action_lengths": teacher_lengths,
        "teacher_stage1_lengths": teacher_stage1_lengths,
        "teacher_stage1_target_hand_qpos": stage1["target_qpos"][stage1_ids, :20].astype(np.float32),
        "teacher_stage2_source_target_qpos": approach["source_target_qpos"][approach_ids].astype(np.float32),
        "teacher_stage2_target_hand_qpos": approach["target_qpos"][approach_ids, :20].astype(np.float32),
        "final_source_target_qpos": final["source_target_qpos"][final_ids].astype(np.float32),
        "target_qpos": target_qpos.astype(np.float32),
        "target_qvel": np.zeros((len(rows), 26), dtype=np.float32),
        "phase_ids": np.full(len(rows), 3, dtype=np.int64),
        "source_episode_step": np.zeros(len(rows), dtype=np.int64),
        "source_phase_offset_rad": np.deg2rad(source_phase_deg).astype(np.float32),
        "target_remaining_rad": np.deg2rad(50.0 - source_phase_deg).astype(np.float32),
        "policy_target_pen_quat": source_qpos[:, 23:27].astype(np.float32),
        "curriculum_stage_deg": source_phase_deg,
        "stage1_bundle_row": stage1_ids,
        "approach_bundle_row": approach_ids,
        "final_bundle_row": final_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"output={args.output}")
    print(f"rows={len(rows)}")
    print(
        "teacher_steps="
        f"{teacher_lengths.min()}/{teacher_lengths.mean():.2f}/{teacher_lengths.max()}"
    )
    print(
        "stage1_steps="
        f"{teacher_stage1_lengths.min()}/{teacher_stage1_lengths.mean():.2f}/"
        f"{teacher_stage1_lengths.max()}"
    )
    print(
        "source_phase_deg="
        f"{source_phase_deg.min():.3f}/{source_phase_deg.mean():.3f}/{source_phase_deg.max():.3f}"
    )


if __name__ == "__main__":
    main()
