from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a paired phase-17 reset cache and teacher tail."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input) as data:
        successful = np.flatnonzero(np.asarray(data["success"], dtype=bool))
        if len(successful) == 0:
            raise ValueError(f"No strict success rows in {args.input}")

        rows = successful
        hand_qpos = np.asarray(data["entry_hand_qpos"][rows], dtype=np.float32)
        hand_qvel = np.asarray(data["entry_hand_qvel"][rows], dtype=np.float32)
        pen_state = np.asarray(data["entry_pen_root_state"][rows], dtype=np.float32)
        source_qpos = np.concatenate((hand_qpos, pen_state[:, :7]), axis=-1)
        source_qvel = np.concatenate((hand_qvel, pen_state[:, 7:13]), axis=-1)
        target_hand = np.asarray(
            data["entry_target_hand_qpos"][rows], dtype=np.float32
        )
        target_qpos = np.concatenate((target_hand, pen_state[:, :7]), axis=-1)

        bundle = {
            "source_qpos": source_qpos,
            "source_qvel": source_qvel,
            "source_target_qpos": np.asarray(
                data["entry_target_qpos"][rows], dtype=np.float32
            ),
            "source_policy_action": np.asarray(
                data["entry_policy_action"][rows], dtype=np.float32
            ),
            "source_target_hand_qpos": target_hand,
            "source_closure_origin_hand": np.asarray(
                data["entry_closure_origin_hand"][rows], dtype=np.float32
            ),
            "source_closure_brace_hand": np.asarray(
                data["entry_closure_brace_hand"][rows], dtype=np.float32
            ),
            "source_closure_origin_pen_pos": np.asarray(
                data["entry_closure_origin_pen_pos"][rows], dtype=np.float32
            ),
            "source_closure_origin_pen_quat": np.asarray(
                data["entry_closure_origin_pen_quat"][rows], dtype=np.float32
            ),
            "source_init_pen_pos": np.asarray(
                data["entry_init_pen_pos"][rows], dtype=np.float32
            ),
            "source_init_pen_axis": np.asarray(
                data["entry_init_pen_axis"][rows], dtype=np.float32
            ),
            "source_max_penetration_m": np.asarray(
                data["entry_max_penetration_m"][rows], dtype=np.float32
            ),
            "source_contact_force_n": np.asarray(
                data["entry_contact_force_n"][rows], dtype=np.float32
            ),
            "source_chain_phase_id": np.asarray(
                data["entry_chain_phase_id"][rows], dtype=np.int64
            ),
            "target_qpos": target_qpos,
            "target_qvel": source_qvel.copy(),
            "phase_ids": np.zeros(len(rows), dtype=np.int64),
            "source_teacher_actions": np.zeros(
                (len(rows), 0, 20), dtype=np.float32
            ),
            "source": np.asarray("strict phase16-to-17 paired entry"),
        }

        teacher_action = np.asarray(
            data["candidate_action_second"][rows], dtype=np.float32
        )
        teacher = {
            "candidate_action_first": teacher_action.copy(),
            "candidate_action_second": teacher_action.copy(),
            "first_duration": np.zeros(len(rows), dtype=np.int64),
            "second_duration": np.full(len(rows), 220, dtype=np.int64),
            "selection_kind": np.full(len(rows), "success"),
            "source_bank_id": np.arange(len(rows), dtype=np.int64),
            "best_brake_score": np.asarray(
                data["max_hold_steps"][rows], dtype=np.float32
            ),
            "source_search_row": rows.astype(np.int64),
        }

    args.bundle_output.parent.mkdir(parents=True, exist_ok=True)
    args.teacher_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.bundle_output, **bundle)
    np.savez_compressed(args.teacher_output, **teacher)
    print(
        f"Built {len(rows)} paired phase-17 rows: "
        f"{args.bundle_output} and {args.teacher_output}"
    )


if __name__ == "__main__":
    main()
