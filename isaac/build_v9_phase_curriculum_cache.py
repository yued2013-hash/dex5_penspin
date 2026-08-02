from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REQUIRED_KEYS = (
    "source_qpos",
    "source_qvel",
    "source_target_qpos",
    "source_policy_action",
    "source_target_hand_qpos",
    "source_closure_origin_hand",
    "source_closure_brace_hand",
    "source_closure_origin_pen_pos",
    "source_closure_origin_pen_quat",
    "source_init_pen_pos",
    "source_init_pen_axis",
    "source_max_penetration_m",
    "source_contact_force_n",
    "source_chain_phase_id",
    "target_qpos",
    "target_qvel",
    "phase_ids",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine a genuine no-reset phase entry with same-phase near "
            "success states for local Isaac Lab curriculum training."
        )
    )
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--min-hold", type=int, default=0)
    parser.add_argument(
        "--min-entry-hold",
        type=int,
        default=0,
        help="Also require best_hold_steps on entry rows.",
    )
    parser.add_argument(
        "--match-candidate-source-bank",
        action="store_true",
        help=(
            "Build a bundle from the original entry rows referenced by each "
            "candidate, preserving one-to-one teacher/action alignment."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.entry) as entry_data:
        entry_phase = np.asarray(entry_data["source_chain_phase_id"])
        entry_mask = entry_phase == args.phase
        if args.min_entry_hold > 0:
            entry_mask &= (
                np.asarray(entry_data["best_hold_steps"])
                >= args.min_entry_hold
            )
        entry = {
            key: np.asarray(entry_data[key])[entry_mask]
            for key in REQUIRED_KEYS
        }

    with np.load(args.candidates) as candidate_data:
        candidate_phase = np.asarray(
            candidate_data["source_chain_phase_id"]
        )
        candidate_mask = candidate_phase == args.phase
        if args.min_hold > 0:
            candidate_mask &= (
                np.asarray(candidate_data["best_hold_steps"])
                >= args.min_hold
            )
        candidates = {
            key: np.asarray(candidate_data[key])[candidate_mask]
            for key in REQUIRED_KEYS
        }
        candidate_holds = np.asarray(
            candidate_data["best_hold_steps"]
        )[candidate_mask]
        candidate_source_bank = np.asarray(
            candidate_data["source_bank_id"], dtype=np.int64
        )[candidate_mask]

    if args.match_candidate_source_bank:
        if len(candidate_source_bank) == 0:
            raise ValueError("No candidate rows selected for paired bundle.")
        payload = {
            key: entry[key][candidate_source_bank]
            for key in REQUIRED_KEYS
        }
        payload["curriculum_source_kind"] = np.full(
            len(candidate_source_bank), "同episode入口配对teacher"
        )
        payload["best_hold_steps"] = candidate_holds.astype(np.int64)
        payload["source"] = np.asarray(
            "候选动作对应的同episode原始入口配对训练包",
            dtype=np.str_,
        )
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **payload)
        print(f"{output}: {len(candidate_source_bank)} paired rows")
        return

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: np.concatenate((entry[key], candidates[key]), axis=0)
        for key in REQUIRED_KEYS
    }
    payload["curriculum_source_kind"] = np.concatenate(
        (
            np.full(len(entry["source_qpos"]), "真实无重置入口"),
            np.full(len(candidates["source_qpos"]), "近成功状态"),
        )
    )
    payload["best_hold_steps"] = np.concatenate(
        (
            np.zeros(len(entry["source_qpos"]), dtype=np.int64),
            candidate_holds.astype(np.int64),
        )
    )
    payload["source"] = np.asarray(
        "真实无重置入口 + 同阶段近成功状态 curriculum",
        dtype=np.str_,
    )
    np.savez_compressed(output, **payload)
    print(
        f"{output}: {len(entry['source_qpos'])} entry + "
        f"{len(candidates['source_qpos'])} near = "
        f"{len(payload['source_qpos'])} rows"
    )


if __name__ == "__main__":
    main()
