from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Merge verified P1-to-P2 long-hold teacher banks.")
parser.add_argument("--inputs", nargs="+", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--summary", required=True, type=Path)
parser.add_argument("--zero-baseline-summary", type=Path, default=None)
parser.add_argument(
    "--truncate-steps",
    type=int,
    default=0,
    help="Trim time-series arrays to the pre-reset teacher horizon.",
)
args = parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    inputs = [path.resolve() for path in args.inputs]
    loaded = [np.load(path, allow_pickle=False) for path in inputs]
    keys = set(loaded[0].files)
    if any(set(data.files) != keys for data in loaded[1:]):
        raise ValueError("Teacher banks do not have identical schemas")
    trajectory_keys = {
        key for key in keys if loaded[0][key].ndim > 0 and loaded[0][key].shape[0] == loaded[0]["action"].shape[0]
    }
    payload = {}
    for key in keys:
        if key in trajectory_keys:
            payload[key] = np.concatenate([data[key] for data in loaded], axis=0)
        else:
            reference = loaded[0][key]
            if any(not np.array_equal(reference, data[key]) for data in loaded[1:]):
                raise ValueError(f"Non-trajectory metadata differs for key: {key}")
            payload[key] = reference
    for data in loaded:
        data.close()

    original_steps = int(payload["action"].shape[1])
    truncate_steps = int(args.truncate_steps)
    if truncate_steps > 0:
        if truncate_steps > original_steps:
            raise ValueError("truncate_steps exceeds the teacher horizon")
        for key, value in tuple(payload.items()):
            if value.ndim >= 2 and value.shape[1] == original_steps:
                payload[key] = value[:, :truncate_steps].copy()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    action = payload["action"]
    latch = payload["hold_latched"].astype(np.bool_)
    ring_l2 = np.linalg.norm(action[:, :, 8:12], axis=-1)
    tim_indices = [17, 18, 19, 1, 2, 3, 5, 6, 7]
    tim_l2 = np.linalg.norm(action[:, :, tim_indices], axis=-1)
    baseline = None
    if args.zero_baseline_summary is not None:
        baseline = json.loads(args.zero_baseline_summary.resolve().read_text(encoding="utf-8"))
    summary = {
        "inputs": [str(path) for path in inputs],
        "input_sha256": [sha256(path) for path in inputs],
        "output": str(output),
        "output_sha256": sha256(output),
        "teacher_trajectories": int(action.shape[0]),
        "steps_per_trajectory": int(action.shape[1]),
        "original_steps_per_trajectory": original_steps,
        "truncated_before_auto_reset": truncate_steps > 0,
        "teacher_rows": int(action.shape[0] * action.shape[1]),
        "unique_source_bundle_rows": int(np.unique(payload["source_bundle_row"]).size),
        "min_consecutive_strict_steps": int(payload["max_consecutive_steps"].min()),
        "max_consecutive_strict_steps": int(payload["max_consecutive_steps"].max()),
        "post_latch_ring_action_l2_mean": float(ring_l2[latch].mean()),
        "post_latch_ring_action_l2_max": float(ring_l2[latch].max()),
        "post_latch_ring_action_abs_max": float(np.abs(action[:, :, 8:12][latch]).max()),
        "post_latch_tim_action_l2_mean": float(tim_l2[latch].mean()),
        "post_latch_tim_action_l2_max": float(tim_l2[latch].max()),
        "zero_increment_baseline_trajectories": (
            int(baseline["accepted_teacher_trajectories"]) if baseline is not None else None
        ),
        "zero_increment_baseline_rate": (
            float(baseline["final_acceptance_rate"]) if baseline is not None else None
        ),
        "normal_physx_search": True,
        "ppo_started": False,
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_LONG_HOLD_TEACHER=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
