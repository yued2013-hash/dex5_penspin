from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dex5_hardware_adapter import COUPLED_PAIRS, JOINT_TO_ACTUATOR


SPLIT_TO_ID = {"train": 0, "validation": 1, "test": 2}
OUTPUT_FIELDS = {
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
PRIVILEGED_FIELDS = {
    "oracle_observation",
    "pen_pose",
    "pen_pose_next",
    "pen_linear_velocity",
    "pen_angular_velocity",
    "pen_angular_velocity_physx_instantaneous",
    "contact_force_w",
    "contact_force_w_next",
    "mass",
    "friction",
    "center_of_mass",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--observation-schema",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    observation_schema = json.loads(args.observation_schema.read_text(encoding="utf-8"))
    assert observation_schema["total_dimension"] == 340
    assert observation_schema["history_frames"] == 4
    assert all(field["real_available"] for field in observation_schema["frame_fields"])

    checks: dict[str, bool] = {}
    with np.load(args.input, allow_pickle=False) as source:
        checks["source_dataset_authorized"] = bool(source["dataset_authorized"])
        student_obs = np.asarray(source["student_observation"], dtype=np.float32)
        next_obs = np.asarray(source["student_observation_next"], dtype=np.float32)
        action = np.asarray(source["virtual_action"], dtype=np.float32)
        offsets = np.asarray(source["episode_offsets"], dtype=np.int64)
        episode_sources = np.asarray(
            source["episode_source_trajectory_id"], dtype=np.int64
        )
        episode_splits = np.asarray(source["episode_split"])
        episode_success = np.asarray(
            source["episode_corrected_success"], dtype=bool
        )
        rows = int(offsets[-1])
        episodes = len(offsets) - 1

        terminal = np.zeros((rows, 1), dtype=bool)
        success = np.zeros((rows, 1), dtype=bool)
        split_id = np.empty((rows, 1), dtype=np.int8)
        episode_id = np.empty((rows, 1), dtype=np.int32)
        row_source = np.empty((rows, 1), dtype=np.int64)
        continuity_error = 0.0
        for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            terminal[stop - 1, 0] = True
            success[stop - 1, 0] = episode_success[index]
            split_id[start:stop, 0] = SPLIT_TO_ID[str(episode_splits[index])]
            episode_id[start:stop, 0] = index
            row_source[start:stop, 0] = episode_sources[index]
            if stop - start > 1:
                continuity_error = max(
                    continuity_error,
                    float(
                        np.max(
                            np.abs(
                                next_obs[start : stop - 1]
                                - student_obs[start + 1 : stop]
                            )
                        )
                    ),
                )

        source_to_split: dict[int, set[str]] = {}
        for source_id, split in zip(episode_sources, episode_splits, strict=True):
            source_to_split.setdefault(int(source_id), set()).add(str(split))
        leakage = sum(len(splits) != 1 for splits in source_to_split.values())

        projected = np.asarray(source["adapter_projected_target"], dtype=np.float32)
        sdk = np.asarray(source["sdk_absolute_target"], dtype=np.float32)
        logical = np.asarray(source["logical_actuator_target"], dtype=np.float32)
        coupling_error = 0.0
        for first, second in COUPLED_PAIRS:
            coupling_error = max(
                coupling_error,
                float(np.max(np.abs(projected[:, first] - projected[:, second]))),
                float(np.max(np.abs(sdk[:, first] - sdk[:, second]))),
            )
        reconstruction_error = float(
            np.max(np.abs(logical[:, JOINT_TO_ACTUATOR] - projected))
        )

        checks.update(
            {
                "student_obs_shape": student_obs.shape == (rows, 340),
                "next_obs_shape": next_obs.shape == (rows, 340),
                "action_shape": action.shape == (rows, 20),
                "all_finite": bool(
                    np.isfinite(student_obs).all()
                    and np.isfinite(next_obs).all()
                    and np.isfinite(action).all()
                ),
                "all_episode_success": bool(episode_success.all()),
                "terminal_count_matches_episodes": int(terminal.sum()) == episodes,
                "success_count_matches_episodes": int(success.sum()) == episodes,
                "next_obs_continuity_exact": continuity_error == 0.0,
                "next_obs_action_history_exact": bool(
                    np.array_equal(next_obs[:, 40:60], action)
                ),
                "action_bounded": bool(np.max(np.abs(action)) <= 1.0 + 1.0e-7),
                "adapter_coupling_exact": coupling_error <= 1.0e-7,
                "adapter_reconstruction_exact": reconstruction_error <= 1.0e-7,
                "source_split_leakage_zero": leakage == 0,
                "all_splits_present": set(map(str, episode_splits.tolist()))
                == set(SPLIT_TO_ID),
                "observation_schema_real_only": not (
                    set(observation_schema.get("forbidden_fields", []))
                    & {
                        field["name"]
                        for field in observation_schema["frame_fields"]
                    }
                ),
                "output_allowlist_excludes_privileged": not (
                    OUTPUT_FIELDS & PRIVILEGED_FIELDS
                ),
            }
        )

    if not all(checks.values()):
        raise RuntimeError(f"student dataset checks failed: {checks}")

    train_rows = split_id[:, 0] == SPLIT_TO_ID["train"]
    observation_mean = student_obs[train_rows].mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    observation_std = student_obs[train_rows].std(axis=0, dtype=np.float64).astype(
        np.float32
    )
    observation_std = np.maximum(observation_std, 1.0e-6).astype(np.float32)

    payload = {
        "student_obs": student_obs,
        "action": action,
        "next_obs": next_obs,
        "success": success,
        "terminal": terminal,
        "split_id": split_id,
        "episode_id": episode_id,
        "source_trajectory_id": row_source,
        "episode_offsets": offsets,
        "episode_source_trajectory_id": episode_sources,
        "episode_split": episode_splits,
        "episode_success": episode_success,
        "observation_mean_train": observation_mean,
        "observation_std_train": observation_std,
    }
    if set(payload) != OUTPUT_FIELDS:
        raise RuntimeError("student dataset output allowlist changed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    row_split_counts = {
        name: int((split_id[:, 0] == split_value).sum())
        for name, split_value in SPLIT_TO_ID.items()
    }
    episode_split_counts = {
        name: int((episode_splits == name).sum()) for name in SPLIT_TO_ID
    }
    unique_sources = {
        name: int(
            len(set(episode_sources[episode_splits == name].astype(int).tolist()))
        )
        for name in SPLIT_TO_ID
    }
    manifest = {
        "schema": "dex5-s0.10-student-dataset-v1",
        "status": "PASS_BC_SMOKE_READY",
        "source_rollout": str(args.input.resolve()),
        "source_rollout_sha256": _sha256(args.input),
        "student_dataset": str(args.output.resolve()),
        "student_dataset_sha256": _sha256(args.output),
        "observation_schema": str(args.observation_schema.resolve()),
        "observation_schema_sha256": _sha256(args.observation_schema),
        "rows": rows,
        "episodes": episodes,
        "episode_split_counts": episode_split_counts,
        "row_split_counts": row_split_counts,
        "unique_sources_by_split": unique_sources,
        "source_split_leakage": leakage,
        "fields": {
            "student_obs": [rows, 340],
            "action": [rows, 20],
            "next_obs": [rows, 340],
            "success": [rows, 1],
        },
        "action_contract": {
            "semantics": "bounded 20D virtual normalized delta action",
            "range": [-1.0, 1.0],
            "hardware_path": "20D virtual -> coupling projector -> 16D logical actuator -> 20D SDK absolute target",
        },
        "success_contract": "true only on the final transition of a quaternion-validated successful episode",
        "normalization": {
            "source": "train split rows only",
            "minimum_std": 1.0e-6,
        },
        "privileged_fields_excluded": sorted(PRIVILEGED_FIELDS),
        "checks": checks,
        "metrics": {
            "next_obs_continuity_max_error": continuity_error,
            "adapter_coupling_max_error_rad": coupling_error,
            "adapter_reconstruction_max_error_rad": reconstruction_error,
            "maximum_abs_action": float(np.max(np.abs(action))),
        },
        "training_status": "SMOKE_ONLY_NO_OPTIMIZER_STEP",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
