from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_v9_approach49_prefill_bundle import nearest_rows, yaw_rotate


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact actuator-target replay between two cascade stages."
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument(
        "--failure-repeat",
        type=int,
        default=1,
        help="Repeat trajectories labelled as final failures to emphasize hard entries.",
    )
    parser.add_argument("--start-stage-id", type=int, default=5)
    parser.add_argument("--end-stage-id", type=int, default=8)
    parser.add_argument("--target-deg", type=float, default=50.0)
    parser.add_argument(
        "--final-bundle",
        type=Path,
        default=REPORT_DIR / "v9_entry45_to_braked50_relaxed50ms_lowpen197_physx_bundle.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.trajectories) as data:
        trajectories = {name: data[name] for name in data.files}
    if "processed_target" not in trajectories:
        raise KeyError("Trajectories do not contain processed_target")
    with np.load(args.final_bundle) as data:
        final = {name: data[name] for name in data.files}

    rows: list[tuple[int, int, int]] = []
    for trajectory_id, length in enumerate(trajectories["lengths"]):
        stages = trajectories["stage_id"][trajectory_id, :length]
        start_steps = np.flatnonzero(stages == args.start_stage_id)
        final_steps = np.flatnonzero(stages == args.end_stage_id)
        if len(start_steps) == 0 or len(final_steps) == 0:
            continue
        start = int(start_steps[0])
        end = int(final_steps[0])
        if start < end:
            rows.append((trajectory_id, start, end))

    if not rows:
        raise RuntimeError("No successful stage 5 -> stage 8 spans found")
    if args.failure_repeat < 1:
        parser.error("--failure-repeat must be >= 1")
    row_success = {}
    for trajectory_id, _, _ in rows:
        stages = trajectories["stage_id"][trajectory_id, : trajectories["lengths"][trajectory_id]]
        if args.end_stage_id + 1 < len(trajectories["stage_names"]):
            success = bool(np.any(stages == args.end_stage_id + 1))
        else:
            success = bool(trajectories.get("final_success", np.ones(len(trajectories["lengths"]), dtype=np.bool_))[trajectory_id])
        row_success[trajectory_id] = success
    if args.failure_repeat > 1:
        rows = [
            row
            for row in rows
            for _ in range(
                1 if row_success[row[0]] else args.failure_repeat
            )
        ]

    lengths = np.asarray([end - start for _, start, end in rows], dtype=np.int64)
    horizon = int(lengths.max())
    teacher_actions = np.zeros((len(rows), horizon, 20), dtype=np.float32)
    teacher_targets = np.zeros_like(teacher_actions)
    for row_id, (trajectory_id, start, end) in enumerate(rows):
        actions = trajectories["action"][trajectory_id, start:end]
        targets = trajectories["processed_target"][trajectory_id, start:end]
        teacher_actions[row_id, : len(actions)] = actions
        teacher_actions[row_id, len(actions) :] = actions[-1]
        teacher_targets[row_id, : len(targets)] = targets
        teacher_targets[row_id, len(targets) :] = targets[-1]

    source_qpos = np.stack([trajectories["qpos"][i, start] for i, start, _ in rows])
    source_qvel = np.stack([trajectories["qvel"][i, start] for i, start, _ in rows])
    source_contact = np.stack(
        [trajectories["contact_force_n"][i, start] for i, start, _ in rows]
    )
    source_phase_deg = np.asarray(
        [trajectories["phase_deg"][i, start] for i, start, _ in rows], dtype=np.float32
    )
    final_entry_qpos = np.stack([trajectories["qpos"][i, end] for i, _, end in rows])
    final_ids = nearest_rows(final_entry_qpos[:, :20], final["source_qpos"][:, :20])
    target_qpos = final["target_qpos"][final_ids].copy()
    target_qpos[:, 20:23] = final_entry_qpos[:, 20:23] + (
        final["target_qpos"][final_ids, 20:23]
        - final["source_qpos"][final_ids, 20:23]
    )
    initial_quat = np.stack(
        [trajectories["initial_qpos"][i, 23:27] for i, _, _ in rows]
    )
    target_qpos[:, 23:27] = yaw_rotate(initial_quat, args.target_deg)

    first_targets = teacher_targets[:, 0]
    last_targets = np.stack(
        [teacher_targets[row_id, length - 1] for row_id, length in enumerate(lengths)]
    )
    payload = {
        "source_qpos": source_qpos.astype(np.float32),
        "source_qvel": source_qvel.astype(np.float32),
        "source_target_qpos": first_targets.astype(np.float32),
        "source_policy_action": np.stack(
            [trajectories["action"][i, start] for i, start, _ in rows]
        ).astype(np.float32),
        "source_contact_force_n": source_contact.astype(np.float32),
        "source_teacher_actions": teacher_actions,
        "teacher_processed_targets": teacher_targets,
        "teacher_action_lengths": lengths,
        "teacher_stage1_lengths": lengths,
        "teacher_stage1_target_hand_qpos": last_targets.astype(np.float32),
        "teacher_stage2_source_target_qpos": first_targets.astype(np.float32),
        "teacher_stage2_target_hand_qpos": last_targets.astype(np.float32),
        "final_source_target_qpos": final["source_target_qpos"][final_ids].astype(np.float32),
        "target_qpos": target_qpos.astype(np.float32),
        "target_qvel": np.zeros((len(rows), 26), dtype=np.float32),
        "phase_ids": np.full(len(rows), 3, dtype=np.int64),
        "source_episode_step": np.zeros(len(rows), dtype=np.int64),
        "source_phase_offset_rad": np.deg2rad(source_phase_deg).astype(np.float32),
        "target_remaining_rad": np.deg2rad(
            args.target_deg - source_phase_deg
        ).astype(np.float32),
        "policy_target_pen_quat": source_qpos[:, 23:27].astype(np.float32),
        "curriculum_stage_deg": source_phase_deg,
        "final_bundle_row": final_ids,
    }
    if row_success:
        payload["source_task_success"] = np.asarray(
            [row_success[trajectory_id] for trajectory_id, _, _ in rows],
            dtype=np.bool_,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"output={args.output}")
    print(f"rows={len(rows)}")
    if "source_task_success" in payload:
        print(
            "source_success/failure="
            f"{int(payload['source_task_success'].sum())}/"
            f"{int((~payload['source_task_success']).sum())}"
        )
    print(f"replay_steps={lengths.min()}/{lengths.mean():.2f}/{lengths.max()}")
    print(
        "source_phase_deg="
        f"{source_phase_deg.min():.3f}/{source_phase_deg.mean():.3f}/{source_phase_deg.max():.3f}"
    )


if __name__ == "__main__":
    main()
