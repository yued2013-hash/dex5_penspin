from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ISAAC_DIR = Path(__file__).resolve().parents[1] / "isaac"
sys.path.insert(0, str(ISAAC_DIR))

from canonical_reset_schema import (  # noqa: E402
    SCHEMA_VERSION,
    concatenate_payloads,
    convert_legacy_source,
    grouped_split,
    subset_payload,
    validate_r0_arrays,
)


def _valid_payload(count: int = 4) -> dict[str, np.ndarray]:
    pose = np.zeros((count, 7), dtype=np.float32)
    pose[:, 3] = 1.0
    return {
        "metadata_state_id": np.arange(count, dtype=np.int64),
        "metadata_canonical_id": np.full(count, "C1", dtype="U2"),
        "metadata_source_trajectory_id": np.arange(count, dtype=np.int64),
        "metadata_source_uri": np.full(count, "source.npz", dtype="U10"),
        "metadata_source_sha256": np.full(count, "sourcehash", dtype="U10"),
        "metadata_derived_source_uri": np.full(count, "source.npz", dtype="U10"),
        "metadata_derived_source_sha256": np.full(count, "sourcehash", dtype="U10"),
        "metadata_asset_hash": np.full(count, "sha256:test", dtype="U16"),
        "metadata_environment_version": np.full(count, "env", dtype="U3"),
        "metadata_physics_version": np.full(count, "physics", dtype="U7"),
        "metadata_joint_order_version": np.full(count, "joint", dtype="U5"),
        "metadata_state_schema_version": np.full(count, SCHEMA_VERSION),
        "metadata_split": np.full(count, "train", dtype="U10"),
        "robot_qpos": np.zeros((count, 20), dtype=np.float32),
        "robot_qvel": np.zeros((count, 20), dtype=np.float32),
        "robot_q_target": np.zeros((count, 20), dtype=np.float32),
        "robot_q_target_available": np.ones(count, dtype=np.bool_),
        "robot_previous_action": np.zeros((count, 20), dtype=np.float32),
        "robot_previous_action_available": np.ones(count, dtype=np.bool_),
        "robot_action_history_buffer": np.zeros((count, 1, 20), dtype=np.float32),
        "robot_action_history_valid_length": np.ones(count, dtype=np.int64),
        "robot_control_delay_buffer": np.zeros((count, 1, 20), dtype=np.float32),
        "robot_control_delay_valid_length": np.zeros(count, dtype=np.int64),
        "robot_control_delay_enabled": np.zeros(count, dtype=np.bool_),
        "robot_continuation_q_target": np.zeros((count, 8, 20), dtype=np.float32),
        "robot_continuation_action": np.zeros((count, 8, 20), dtype=np.float32),
        "robot_continuation_valid_length": np.zeros(count, dtype=np.int64),
        "object_pose": pose,
        "object_linear_velocity": np.zeros((count, 3), dtype=np.float32),
        "object_angular_velocity": np.zeros((count, 3), dtype=np.float32),
        "contact_positions_frame": np.full(count, "world", dtype="U5"),
        "physics_mass": np.full(count, 0.0094, dtype=np.float32),
        "physics_static_friction": np.full(count, 0.6, dtype=np.float32),
        "physics_dynamic_friction": np.full(count, 0.45, dtype=np.float32),
        "physics_com_offset": np.zeros((count, 3), dtype=np.float32),
        "physics_dt": np.full(count, 1.0 / 120.0, dtype=np.float32),
        "physics_control_dt": np.full(count, 1.0 / 60.0, dtype=np.float32),
        "physics_contact_offset": np.full(count, 0.001, dtype=np.float32),
        "physics_rest_offset": np.zeros(count, dtype=np.float32),
        "labels_static_or_dynamic": np.full(count, "static", dtype="U7"),
    }


def test_r0_accepts_complete_finite_payload() -> None:
    payload = _valid_payload()
    passed, checks = validate_r0_arrays(payload)
    assert passed.all()
    assert all(check.all() for check in checks.values())


def test_v9_grasp_namespace_is_accepted(tmp_path: Path) -> None:
    source_path = tmp_path / "p5.npz"
    qpos = np.zeros((1, 27), dtype=np.float32)
    qpos[:, 23] = 1.0
    np.savez_compressed(
        source_path,
        source_qpos=qpos,
        source_qvel=np.zeros((1, 26), dtype=np.float32),
        source_target_qpos=np.zeros((1, 20), dtype=np.float32),
        source_policy_action=np.zeros((1, 20), dtype=np.float32),
    )
    payload = convert_legacy_source(
        source_path,
        "P5",
        "static",
        source_label="v9 grasp",
        asset_hash="sha256:asset",
        environment_version="env",
        physics_version="physics",
    )
    assert np.all(payload["metadata_canonical_id"] == "P5")


def test_r0_rejects_non_unit_quaternion_and_nan() -> None:
    payload = _valid_payload()
    payload["object_pose"][1, 3] = 0.5
    payload["robot_qvel"][2, 0] = np.nan
    passed, checks = validate_r0_arrays(payload)
    assert not passed[1]
    assert not checks["quaternion_valid"][1]
    assert not passed[2]
    assert not checks["finite"][2]


def test_grouped_split_has_no_trajectory_leakage() -> None:
    trajectory_ids = np.asarray((1, 1, 2, 2, 3, 4, 4), dtype=np.int64)
    split = grouped_split(trajectory_ids)
    for trajectory_id in np.unique(trajectory_ids):
        assert np.unique(split[trajectory_ids == trajectory_id]).size == 1


def test_subset_slices_only_row_arrays() -> None:
    payload = _valid_payload()
    payload["scalar"] = np.asarray("value")
    selected = subset_payload(payload, np.asarray((1, 3)))
    assert selected["robot_qpos"].shape == (2, 20)
    assert selected["scalar"].shape == ()


def test_derived_source_preserves_original_provenance_and_control_context(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "derived.npz"
    qpos = np.zeros((2, 27), dtype=np.float32)
    qpos[:, 23] = 1.0
    qvel = np.zeros((2, 26), dtype=np.float32)
    continuation = np.zeros((2, 8, 20), dtype=np.float32)
    np.savez_compressed(
        source_path,
        source_qpos=qpos,
        source_qvel=qvel,
        source_target_qpos=np.zeros((2, 20), dtype=np.float32),
        source_policy_action=np.zeros((2, 20), dtype=np.float32),
        source_trajectory_ids=np.asarray((10, 11), dtype=np.int64),
        source_original_uri=np.asarray("/original/rollout.npz"),
        source_original_sha256=np.asarray("a" * 64),
        robot_continuation_q_target=continuation,
        robot_continuation_action=continuation,
        robot_continuation_valid_length=np.full(2, 8, dtype=np.int64),
    )
    payload = convert_legacy_source(
        source_path,
        "C3",
        "dynamic",
        source_label="test",
        asset_hash="sha256:asset",
        environment_version="env",
        physics_version="physics",
    )
    payload["metadata_split"] = grouped_split(
        payload["metadata_source_trajectory_id"]
    )
    passed, _ = validate_r0_arrays(payload)

    assert passed.all()
    assert np.all(payload["metadata_source_uri"] == "/original/rollout.npz")
    assert np.all(payload["metadata_source_sha256"] == "a" * 64)
    assert np.all(payload["metadata_derived_source_uri"] == str(source_path))
    assert np.all(payload["robot_continuation_valid_length"] == 8)
    assert np.all(payload["contact_positions_frame"] == "world")


def test_concatenate_requires_exact_schema_and_preserves_rows() -> None:
    first = _valid_payload(2)
    second = _valid_payload(3)
    combined = concatenate_payloads([first, second])
    assert combined["metadata_state_id"].shape == (5,)
    second.pop("physics_mass")
    try:
        concatenate_payloads([first, second])
    except ValueError as exc:
        assert "same schema keys" in str(exc)
    else:
        raise AssertionError("schema mismatch was not rejected")
