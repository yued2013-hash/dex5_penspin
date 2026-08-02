from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Keep one coherent middle-clear teacher mode.")
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--parents", type=Path, required=True)
parser.add_argument("--cluster", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def main() -> None:
    with np.load(args.parents) as parent_data:
        parents = np.concatenate(
            (parent_data["pinch_actions"], parent_data["clear_actions"]), axis=1
        )
    with np.load(args.dataset) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    actions = np.concatenate(
        (payload["pinch_actions"], payload["clear_actions"]), axis=1
    )
    cluster_ids = np.mean(
        (actions[:, None, :] - parents[None, :, :]) ** 2, axis=2
    ).argmin(axis=1)
    keep = cluster_ids == args.cluster
    filtered = {
        key: value[keep] if value.ndim > 0 and len(value) == len(keep) else value
        for key, value in payload.items()
    }
    filtered["teacher_cluster_id"] = np.full(
        int(keep.sum()), args.cluster, dtype=np.int64
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **filtered)
    summary = {
        "dataset": str(args.dataset),
        "parents": str(args.parents),
        "cluster": args.cluster,
        "source_rows": len(keep),
        "kept_rows": int(keep.sum()),
        "unique_bank_ids": int(np.unique(payload["bank_ids"][keep]).size),
        "mean_hold_steps": float(payload["max_clear_hold"][keep].mean()),
        "max_hold_steps": int(payload["max_clear_hold"][keep].max()),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
