from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Build a distance-balanced local arrival bank.")
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--rows-per-bin", type=int, default=128)
parser.add_argument("--seed", type=int, default=2821)
args = parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


with np.load(args.input, allow_pickle=False) as data:
    payload = {key: np.asarray(data[key]) for key in data.files}

distance = payload["target_distance_mm"]
edges = np.asarray((5.0, 6.0, 7.0, 7.5, 8.0001), dtype=np.float32)
rng = np.random.default_rng(args.seed)
selected_parts = []
source_counts = {}
for low, high in zip(edges[:-1], edges[1:], strict=True):
    ids = np.flatnonzero((distance >= low) & (distance < high))
    if ids.size == 0:
        raise ValueError(f"No arrival states in distance bin [{low}, {high}) mm")
    chosen = rng.choice(ids, size=args.rows_per_bin, replace=ids.size < args.rows_per_bin)
    selected_parts.append(chosen)
    source_counts[f"{low:.1f}-{high:.1f}mm"] = int(ids.size)

selected = np.concatenate(selected_parts)
rng.shuffle(selected)
balanced = {
    key: value[selected] if value.ndim > 0 and value.shape[0] == distance.shape[0] else value
    for key, value in payload.items()
}
balanced["arrival_distance_bin"] = np.digitize(
    balanced["target_distance_mm"], edges[1:-1], right=False
).astype(np.int64)

args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output, **balanced)
bin_id_paths = {}
for bin_id in range(4):
    ids_path = args.output.with_name(f"{args.output.stem}_bin{bin_id}_ids.npy")
    np.save(ids_path, np.flatnonzero(balanced["arrival_distance_bin"] == bin_id))
    bin_id_paths[str(bin_id)] = str(ids_path.resolve())
summary = {
    "input": str(args.input.resolve()),
    "input_sha256": sha256(args.input),
    "output": str(args.output.resolve()),
    "output_sha256": sha256(args.output),
    "rows": int(selected.size),
    "rows_per_bin": int(args.rows_per_bin),
    "source_bin_counts": source_counts,
    "balanced_bin_counts": {
        str(bin_id): int(np.sum(balanced["arrival_distance_bin"] == bin_id))
        for bin_id in range(4)
    },
    "bin_id_paths": bin_id_paths,
    "distance_mm_min": float(balanced["target_distance_mm"].min()),
    "distance_mm_max": float(balanced["target_distance_mm"].max()),
    "note": "All rows originate from normal PhysX policy rollouts; balancing only resamples rows.",
}
args.output.with_suffix(".json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("P1_P2_LOCAL_ARRIVAL_BANK=" + json.dumps(summary, ensure_ascii=False))
