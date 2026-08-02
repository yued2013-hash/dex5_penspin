from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a P5 capture bundle with collision-free retract targets."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_gain", type=float, default=0.0)
    parser.add_argument("--teacher_horizon", type=int, default=96)
    args = parser.parse_args()

    source = np.load(args.input)
    arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    target_qpos = arrays["target_qpos"]

    # Preserve the physically validated clear posture for middle/ring/pinky.
    # Thumb/index and pen targets remain the canonical P5 targets.
    target_qpos[:, 4:16] = arrays["source_qpos"][:, 4:16]
    arrays["target_qpos"] = target_qpos
    if args.teacher_gain > 0.0:
        count = target_qpos.shape[0]
        policy_target = arrays["source_target_qpos"].astype(np.float32).copy()
        target_hand = target_qpos[:, :20].astype(np.float32)
        mask = np.zeros((count, 20), dtype=np.float32)
        mask[:, :4] = 0.35
        mask[:, 16:20] = 0.35
        increment = 0.25 * 0.075 * mask
        teacher = np.zeros(
            (count, args.teacher_horizon, 20), dtype=np.float32
        )
        for step in range(args.teacher_horizon):
            action = args.teacher_gain * (target_hand - policy_target)
            action = np.divide(
                action,
                np.maximum(increment, 1.0e-6),
                out=np.zeros_like(action),
                where=mask > 0.0,
            )
            action = np.clip(action, -1.0, 1.0)
            teacher[:, step] = action
            policy_target += increment * action
        arrays["source_teacher_actions"] = teacher

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(
        f"Saved {target_qpos.shape[0]} collision-free capture resets to {output}; "
        f"teacher horizon={arrays['source_teacher_actions'].shape[1]}"
    )


if __name__ == "__main__":
    main()
