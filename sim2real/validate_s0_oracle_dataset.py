from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dex5_hardware_adapter import COUPLED_PAIRS, JOINT_TO_ACTUATOR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    required_shapes = {
        "oracle_observation": 152,
        "student_observation": 340,
        "virtual_action": 20,
        "policy_action_raw": 20,
        "adapter_projected_target": 20,
        "logical_actuator_target": 16,
        "sdk_absolute_target": 20,
        "joint_position": 20,
        "joint_velocity": 20,
        "pen_pose": 7,
        "pen_linear_velocity": 3,
        "pen_angular_velocity": 3,
        "contact_state": 5,
        "contact_force_w": 15,
        "contact_force_norm": 5,
    }
    checks: dict[str, bool] = {}
    metrics: dict[str, int | float] = {}
    with np.load(args.dataset, allow_pickle=False) as data:
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        lengths = np.asarray(data["episode_length"], dtype=np.int64)
        rows = int(offsets[-1])
        checks["at_least_1000_episodes"] = lengths.size >= 1000
        checks["offsets_match_lengths"] = bool(
            offsets.shape == (lengths.size + 1,)
            and offsets[0] == 0
            and np.array_equal(np.diff(offsets), lengths)
        )
        checks["positive_episode_lengths"] = bool(np.all(lengths > 0))
        for name, width in required_shapes.items():
            value = np.asarray(data[name])
            checks[f"shape_{name}"] = value.shape == (rows, width)
            checks[f"finite_{name}"] = bool(np.isfinite(value).all())

        transition_fields = {
            "student_observation_next": 340,
            "pen_pose_next": 7,
            "pen_linear_velocity_next": 3,
            "contact_state_next": 5,
            "contact_force_w_next": 15,
            "contact_force_norm_next": 5,
            "transition_same_episode": 1,
            "pen_angular_velocity_pose_delta": 3,
            "pen_angular_velocity_physx_instantaneous": 3,
            "pen_angular_velocity_physx_instantaneous_next": 3,
        }
        if any(name in data.files for name in transition_fields):
            for name, width in transition_fields.items():
                checks[f"present_{name}"] = name in data.files
                if name in data.files:
                    value = np.asarray(data[name])
                    checks[f"shape_{name}"] = value.shape == (rows, width)
                    checks[f"finite_{name}"] = bool(np.isfinite(value).all())
            checks["dataset_authorized"] = bool(
                "dataset_authorized" in data.files
                and np.asarray(data["dataset_authorized"]).item()
            )
            checks["all_transitions_have_terminal_safe_next_state"] = bool(
                np.asarray(data["transition_same_episode"], dtype=bool).all()
            )
            checks["canonical_velocity_is_pose_delta"] = bool(
                np.array_equal(
                    np.asarray(data["pen_angular_velocity"]),
                    np.asarray(data["pen_angular_velocity_pose_delta"]),
                )
            )

        projected = np.asarray(data["adapter_projected_target"])
        sdk_target = np.asarray(data["sdk_absolute_target"])
        logical = np.asarray(data["logical_actuator_target"])
        coupling_error = 0.0
        for first, second in COUPLED_PAIRS:
            coupling_error = max(
                coupling_error,
                float(np.max(np.abs(projected[:, first] - projected[:, second]))),
                float(np.max(np.abs(sdk_target[:, first] - sdk_target[:, second]))),
            )
        reconstructed = logical[:, JOINT_TO_ACTUATOR]
        reconstruction_error = float(np.max(np.abs(reconstructed - projected)))
        checks["coupling_exact"] = coupling_error <= 1.0e-7
        checks["logical_reconstruction_exact"] = reconstruction_error <= 1.0e-7

        source_ids = np.asarray(data["episode_source_trajectory_id"], dtype=np.int64)
        splits = np.asarray(data["episode_split"])
        source_to_splits: dict[int, set[str]] = {}
        for source_id, split in zip(source_ids.tolist(), splits.tolist(), strict=True):
            source_to_splits.setdefault(int(source_id), set()).add(str(split))
        leakage = sum(len(values) != 1 for values in source_to_splits.values())
        checks["source_split_leakage_zero"] = leakage == 0
        checks["executed_virtual_action_bounded"] = bool(
            np.max(np.abs(np.asarray(data["virtual_action"]))) <= 1.0 + 1.0e-7
        )
        checks["all_splits_present"] = set(splits.tolist()) == {
            "train",
            "validation",
            "test",
        }
        metrics.update(
            {
                "episodes": int(lengths.size),
                "transition_rows": rows,
                "unique_sources": len(source_to_splits),
                "source_split_leakage": leakage,
                "max_coupling_error_rad": coupling_error,
                "max_logical_reconstruction_error_rad": reconstruction_error,
                "max_abs_virtual_action": float(
                    np.max(np.abs(np.asarray(data["virtual_action"])))
                ),
            }
        )

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "checks": checks,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
