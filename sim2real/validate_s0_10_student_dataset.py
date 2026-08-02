from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_FIELDS = {
    "student_obs",
    "action",
    "next_obs",
    "success",
    "terminal",
    "split_id",
    "episode_id",
    "source_trajectory_id",
    "episode_offsets",
    "episode_source_trajectory_id",
    "episode_split",
    "episode_success",
    "observation_mean_train",
    "observation_std_train",
}
SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}
PRIVILEGED_TOKENS = (
    "oracle",
    "pen_pose",
    "pen_velocity",
    "angular_velocity",
    "mass",
    "friction",
    "center_of_mass",
    "wrench",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    metrics: dict[str, object] = {}

    with np.load(args.dataset, allow_pickle=False) as data:
        fields = set(data.files)
        obs = np.asarray(data["student_obs"])
        action = np.asarray(data["action"])
        next_obs = np.asarray(data["next_obs"])
        success = np.asarray(data["success"])
        terminal = np.asarray(data["terminal"])
        split = np.asarray(data["split_id"])[:, 0]
        episode_id = np.asarray(data["episode_id"])[:, 0]
        source_id = np.asarray(data["source_trajectory_id"])[:, 0]
        offsets = np.asarray(data["episode_offsets"])
        mean = np.asarray(data["observation_mean_train"])
        std = np.asarray(data["observation_std_train"])

        rows = obs.shape[0]
        episodes = len(offsets) - 1
        checks.update(
            {
                "exact_field_allowlist": fields == EXPECTED_FIELDS,
                "no_privileged_field_name": not any(
                    token in field.lower()
                    for field in fields
                    for token in PRIVILEGED_TOKENS
                ),
                "observation_shape": obs.shape == (rows, 340),
                "action_shape": action.shape == (rows, 20),
                "next_observation_shape": next_obs.shape == (rows, 340),
                "label_shapes": success.shape == terminal.shape == (rows, 1),
                "all_finite": bool(
                    np.isfinite(obs).all()
                    and np.isfinite(action).all()
                    and np.isfinite(next_obs).all()
                    and np.isfinite(mean).all()
                    and np.isfinite(std).all()
                ),
                "action_bounded": bool(np.max(np.abs(action)) <= 1.0 + 1.0e-7),
                "normalization_shape": mean.shape == std.shape == (340,),
                "normalization_std_positive": bool((std >= 1.0e-6).all()),
                "offsets_cover_rows": bool(
                    offsets[0] == 0
                    and offsets[-1] == rows
                    and np.all(np.diff(offsets) > 0)
                ),
                "terminal_count": int(terminal.sum()) == episodes,
                "success_count": int(success.sum()) == episodes,
                "success_only_at_terminal": bool(np.all(~success | terminal)),
                "all_splits_present": set(np.unique(split).tolist())
                == set(SPLIT_NAMES),
            }
        )

        terminal_layout_ok = True
        episode_layout_ok = True
        continuity_max_error = 0.0
        for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            expected_terminal = np.zeros(stop - start, dtype=bool)
            expected_terminal[-1] = True
            terminal_layout_ok &= np.array_equal(
                terminal[start:stop, 0], expected_terminal
            )
            episode_layout_ok &= bool(np.all(episode_id[start:stop] == index))
            if stop - start > 1:
                continuity_max_error = max(
                    continuity_max_error,
                    float(
                        np.max(
                            np.abs(
                                next_obs[start : stop - 1]
                                - obs[start + 1 : stop]
                            )
                        )
                    ),
                )
        checks["terminal_layout"] = terminal_layout_ok
        checks["episode_layout"] = episode_layout_ok
        checks["nonterminal_next_obs_continuity_exact"] = continuity_max_error == 0.0
        checks["next_obs_action_history_exact"] = bool(
            np.array_equal(next_obs[:, 40:60], action)
        )

        source_splits: dict[int, set[int]] = {}
        for source, split_value in zip(source_id, split, strict=True):
            source_splits.setdefault(int(source), set()).add(int(split_value))
        leakage = sum(len(values) != 1 for values in source_splits.values())
        checks["source_split_leakage_zero"] = leakage == 0

        train_obs = obs[split == 0]
        recomputed_mean = train_obs.mean(axis=0, dtype=np.float64).astype(np.float32)
        recomputed_std = train_obs.std(axis=0, dtype=np.float64).astype(np.float32)
        recomputed_std = np.maximum(recomputed_std, 1.0e-6)
        checks["train_normalization_mean_exact"] = bool(
            np.array_equal(recomputed_mean, mean)
        )
        checks["train_normalization_std_exact"] = bool(
            np.array_equal(recomputed_std, std)
        )

        metrics.update(
            {
                "rows": rows,
                "episodes": episodes,
                "terminal_rows": int(terminal.sum()),
                "success_rows": int(success.sum()),
                "maximum_abs_action": float(np.max(np.abs(action))),
                "next_obs_continuity_max_error": continuity_max_error,
                "source_split_leakage": leakage,
                "row_split_counts": {
                    name: int((split == split_id).sum())
                    for split_id, name in SPLIT_NAMES.items()
                },
                "unique_sources_by_split": {
                    name: len(set(source_id[split == split_id].astype(int).tolist()))
                    for split_id, name in SPLIT_NAMES.items()
                },
            }
        )

    checks["dataset_hash_matches_manifest"] = (
        _sha256(args.dataset) == manifest["student_dataset_sha256"]
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": "dex5-s0.10-student-dataset-validation-v1",
        "status": status,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "manifest": str(args.manifest.resolve()),
        "checks": checks,
        "metrics": metrics,
        "hardware_commands_issued": False,
        "optimizer_step_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
