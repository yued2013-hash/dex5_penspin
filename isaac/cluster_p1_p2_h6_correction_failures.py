from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kmeans(features: np.ndarray, clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_labels = None
    best_centers = None
    best_cost = np.inf
    for _ in range(20):
        centers = features[rng.choice(len(features), size=clusters, replace=False)].copy()
        labels = np.zeros(len(features), dtype=np.int64)
        for _ in range(100):
            distance = ((features[:, None] - centers[None]) ** 2).sum(axis=-1)
            new_labels = distance.argmin(axis=1)
            new_centers = centers.copy()
            for cluster in range(clusters):
                members = features[new_labels == cluster]
                if len(members):
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    new_centers[cluster] = features[rng.integers(len(features))]
            if np.array_equal(labels, new_labels):
                centers = new_centers
                labels = new_labels
                break
            labels, centers = new_labels, new_centers
        cost = ((features - centers[labels]) ** 2).sum()
        if cost < best_cost:
            best_cost = float(cost)
            best_labels = labels.copy()
            best_centers = centers.copy()
    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster v14 H6 latch failures by correction error direction."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--representatives-per-cluster", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2821)
    args = parser.parse_args()

    source = args.input.resolve()
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    required = {
        "failed_episode_id",
        "target_condition",
        "target_condition_delta",
        "semantic_margin",
        "target_distance_mm",
        "angular_speed",
        "linear_speed",
        "latch_age",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"Missing clustering fields: {sorted(missing)}")

    episode_ids = np.unique(arrays["failed_episode_id"])
    representative_rows = []
    for episode_id in episode_ids:
        rows = np.flatnonzero(arrays["failed_episode_id"] == episode_id)
        target_age = np.clip(9, arrays["latch_age"][rows].min(), arrays["latch_age"][rows].max())
        row = rows[np.argmin(np.abs(arrays["latch_age"][rows] - target_age))]
        representative_rows.append(int(row))
    representative_rows = np.asarray(representative_rows, dtype=np.int64)
    raw_features = np.concatenate(
        (
            arrays["target_condition"][representative_rows],
            arrays["target_condition_delta"][representative_rows],
            arrays["semantic_margin"][representative_rows, None],
            arrays["target_distance_mm"][representative_rows, None],
            arrays["angular_speed"][representative_rows, None],
            arrays["linear_speed"][representative_rows, None],
        ),
        axis=-1,
    ).astype(np.float64)
    median = np.median(raw_features, axis=0)
    scale = np.median(np.abs(raw_features - median), axis=0)
    scale = np.maximum(scale, 1.0e-5)
    features = np.clip((raw_features - median) / scale, -8.0, 8.0)
    labels, centers = kmeans(features, int(args.clusters), int(args.seed))

    selected_rows = []
    selected_clusters = []
    cluster_counts = []
    selected_counts = []
    for cluster in range(int(args.clusters)):
        members = np.flatnonzero(labels == cluster)
        cluster_counts.append(int(len(members)))
        distances = ((features[members] - centers[cluster]) ** 2).sum(axis=-1)
        ordered = members[np.argsort(distances, kind="stable")]
        keep = ordered[: int(args.representatives_per_cluster)]
        selected_rows.extend(representative_rows[keep].tolist())
        selected_clusters.extend([cluster] * len(keep))
        selected_counts.append(int(len(keep)))
    selected_rows_np = np.asarray(selected_rows, dtype=np.int64)
    selected_clusters_np = np.asarray(selected_clusters, dtype=np.int64)
    payload = {
        key: value[selected_rows_np]
        for key, value in arrays.items()
        if value.ndim > 0 and value.shape[0] == len(arrays["source_qpos"])
    }
    payload["correction_cluster_id"] = selected_clusters_np
    payload["original_source_bundle_row"] = selected_rows_np
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "episode_count": int(len(episode_ids)),
        "clusters": int(args.clusters),
        "representatives_per_cluster": int(args.representatives_per_cluster),
        "cluster_episode_counts": cluster_counts,
        "selected_counts": selected_counts,
        "selected_rows": int(len(selected_rows_np)),
        "feature_median": median.tolist(),
        "feature_scale": scale.tolist(),
        "cluster_centers_standardized": centers.tolist(),
        "ppo_started": False,
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_H6_CORRECTION_CLUSTERS=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
