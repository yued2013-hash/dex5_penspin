from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced five-stage bundle.")
    parser.add_argument("--hold", required=True)
    parser.add_argument("--middle", required=True)
    parser.add_argument("--ring", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-stage", type=int, default=512)
    args = parser.parse_args()

    paths = (args.hold, args.middle, args.ring, args.joint, args.capture)
    keys = (
        "source_qpos",
        "source_qvel",
        "source_target_qpos",
        "source_policy_action",
        "target_qpos",
        "target_qvel",
        "source_contact_force_n",
        "source_phase_deg",
        "source_angular_speed_rad_s",
        "source_max_penetration_m",
        "source_entry_labels",
        "source_trajectory_ids",
        "original_source_bank_ids",
    )
    merged: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    stages: list[np.ndarray] = []
    rng = np.random.default_rng(1700)
    for stage, path in enumerate(paths):
        data = np.load(path)
        size = data["source_qpos"].shape[0]
        replace = size < args.per_stage
        ids = rng.choice(size, args.per_stage, replace=replace)
        for key in keys:
            if key in data:
                merged[key].append(np.asarray(data[key])[ids])
            elif key == "target_qvel":
                merged[key].append(
                    np.zeros((args.per_stage, 26), dtype=np.float32)
                )
            else:
                dtype = np.int64 if key.endswith(("labels", "ids")) else np.float32
                merged[key].append(np.zeros(args.per_stage, dtype=dtype))
        stages.append(np.full(args.per_stage, stage, dtype=np.int64))
    arrays = {key: np.concatenate(rows) for key, rows in merged.items()}
    arrays["phase_ids"] = np.concatenate(stages)
    arrays["source_teacher_actions"] = np.zeros(
        (len(arrays["phase_ids"]), 0, 20), dtype=np.float32
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"Saved {len(arrays['phase_ids'])} balanced resets to {output}")


if __name__ == "__main__":
    main()
