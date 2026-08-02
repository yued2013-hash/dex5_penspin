from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def pen_axis(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        axis=-1,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote collision-free thumb/index/middle capture states into "
            "a V9 sequential-closure brace bundle."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", type=int, default=4)
    args = parser.parse_args()

    with np.load(args.input) as data:
        qpos = np.asarray(data["source_qpos"], dtype=np.float32)
        qvel = np.asarray(data["source_qvel"], dtype=np.float32)
        hand = qpos[:, :20].copy()
        policy_target = np.asarray(
            data["source_target_qpos"], dtype=np.float32
        )
        policy_action = np.asarray(
            data["source_policy_action"], dtype=np.float32
        )
        contact_force = np.asarray(
            data["source_contact_force_n"], dtype=np.float32
        )
        teacher_actions = np.asarray(
            data["source_teacher_actions"], dtype=np.float32
        )

    count = len(qpos)
    pen_pos = qpos[:, 20:23].copy()
    pen_quat = qpos[:, 23:27].copy()
    payload = {
        "source_qpos": qpos,
        "source_qvel": qvel,
        "source_target_qpos": policy_target,
        "source_policy_action": policy_action,
        "source_target_hand_qpos": hand,
        "source_closure_origin_hand": hand,
        "source_closure_brace_hand": hand,
        "source_closure_origin_pen_pos": pen_pos,
        "source_closure_origin_pen_quat": pen_quat,
        "source_init_pen_pos": pen_pos,
        "source_init_pen_axis": pen_axis(pen_quat),
        "source_max_penetration_m": np.zeros(count, dtype=np.float32),
        "source_contact_force_n": contact_force,
        "source_chain_phase_id": np.full(
            count, args.phase, dtype=np.int64
        ),
        "target_qpos": qpos,
        "target_qvel": np.zeros_like(qvel),
        "phase_ids": np.zeros(count, dtype=np.int64),
        "source_teacher_actions": teacher_actions,
        "contact_manifest": np.asarray(
            "thumb/index/middle; ring and pinky clear", dtype=np.str_
        ),
        "source": np.asarray(
            "collision-free P4 thumb/index/middle capture promotion",
            dtype=np.str_,
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(f"{output}: {count} thumb/index/middle brace rows")


if __name__ == "__main__":
    main()
