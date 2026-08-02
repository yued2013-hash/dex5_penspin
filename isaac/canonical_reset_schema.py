from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "dex5-canonical-reset-v1"
EXPLORATION_CANONICAL_IDS = ("C1", "C2", "C3", "C4", "C5", "C6", "CR")
V9_GRASP_IDS = ("P1", "P2", "P3", "P4", "P5")
CANONICAL_IDS = EXPLORATION_CANONICAL_IDS + V9_GRASP_IDS
STATE_TYPES = ("static", "dynamic")
JOINT_ORDER = (
    "Roll_21R",
    "Pitch_22R",
    "Pitch_23R",
    "Pitch_24R",
    "Roll_31R",
    "Pitch_32R",
    "Pitch_33R",
    "Pitch_34R",
    "Roll_41R",
    "Pitch_42R",
    "Pitch_43R",
    "Pitch_44R",
    "Roll_51R",
    "Pitch_52R",
    "Pitch_53R",
    "Pitch_54R",
    "Yaw_11R",
    "Roll_12R",
    "Pitch_13R",
    "Pitch_14R",
)
FINGERTIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
DEFAULT_PEN_COM_OFFSET = np.asarray((0.00935, 0.0, 0.0), dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_text_id(value: str) -> np.int64:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return np.int64(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))


def joint_order_version() -> str:
    return "sha256:" + hashlib.sha256("\n".join(JOINT_ORDER).encode("ascii")).hexdigest()


def _row_array(value: Any, count: int, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.repeat(array.reshape(1), count, axis=0)
    if array.shape[0] != count:
        return np.repeat(array[None, ...], count, axis=0)
    return array


def _optional_rows(
    source: Mapping[str, np.ndarray],
    keys: tuple[str, ...],
    indices: np.ndarray,
    default: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    for key in keys:
        if key in source:
            value = np.asarray(source[key])
            if value.ndim > 0 and value.shape[0] >= int(indices.max(initial=-1)) + 1:
                rows = np.asarray(value[indices], dtype=default.dtype)
                return rows, np.ones(indices.size, dtype=np.bool_)
    return np.repeat(default[None, ...], indices.size, axis=0), np.zeros(
        indices.size, dtype=np.bool_
    )


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    v = np.broadcast_to(np.asarray(vector, dtype=np.float64), (q.shape[0], 3))
    xyz = q[:, 1:4]
    uv = np.cross(xyz, v)
    uuv = np.cross(xyz, uv)
    return (v + 2.0 * (q[:, :1] * uv + uuv)).astype(np.float32)


def _source_trajectory_ids(
    source: Mapping[str, np.ndarray],
    indices: np.ndarray,
    source_hash: str,
) -> tuple[np.ndarray, str]:
    for key in (
        "source_trajectory_ids",
        "source_trajectory_id",
        "trajectory_ids",
        "source_global_bank_ids",
        "source_candidate_ids",
        "source_row_ids",
        "original_source_bank_ids",
        "initial_bank_ids",
    ):
        if key in source:
            values = np.asarray(source[key])
            if values.ndim == 1 and values.shape[0] > int(indices.max(initial=-1)):
                raw = values[indices]
                result = np.asarray(
                    [stable_text_id(f"{source_hash}:{key}:{value}") for value in raw],
                    dtype=np.int64,
                )
                return result, key
    result = np.asarray(
        [stable_text_id(f"{source_hash}:row:{int(row)}") for row in indices],
        dtype=np.int64,
    )
    return result, "derived_row_id"


def convert_legacy_source(
    source_path: Path,
    canonical_id: str,
    state_type: str,
    *,
    source_label: str,
    asset_hash: str,
    environment_version: str,
    physics_version: str,
    indices: np.ndarray | None = None,
    phase_id: int | None = None,
    mass: float = 0.0094,
    static_friction: float = 0.60,
    dynamic_friction: float = 0.45,
    com_offset: np.ndarray = DEFAULT_PEN_COM_OFFSET,
    physics_dt: float = 1.0 / 120.0,
    control_dt: float = 1.0 / 60.0,
    solver_position_iterations: int = 12,
    solver_velocity_iterations: int = 2,
    contact_offset: float = 0.001,
    rest_offset: float = 0.0,
) -> dict[str, np.ndarray]:
    if canonical_id not in CANONICAL_IDS:
        raise ValueError(f"Unknown canonical_id: {canonical_id}")
    if state_type not in STATE_TYPES:
        raise ValueError(f"Unknown state_type: {state_type}")

    source_path = source_path.resolve()
    derived_source_hash = sha256_file(source_path)
    with np.load(source_path, allow_pickle=False) as loaded:
        source = {key: loaded[key] for key in loaded.files}

    source_uri = str(source_path)
    source_hash = derived_source_hash
    if "source_original_uri" in source:
        original_uri = np.asarray(source["source_original_uri"])
        source_uri = str(original_uri.reshape(-1)[0])
    if "source_original_sha256" in source:
        original_hash = np.asarray(source["source_original_sha256"])
        source_hash = str(original_hash.reshape(-1)[0])

    if "source_qpos" not in source or "source_qvel" not in source:
        raise ValueError(f"{source_path} is not a reset-state source")
    total = int(source["source_qpos"].shape[0])
    if indices is None:
        indices = np.arange(total, dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)
    if phase_id is not None:
        if "phase_ids" not in source:
            raise ValueError(f"phase_id filter requested but {source_path} has no phase_ids")
        indices = indices[np.asarray(source["phase_ids"])[indices] == phase_id]
    if indices.size == 0:
        raise ValueError(f"No rows selected from {source_path}")
    if np.any(indices < 0) or np.any(indices >= total):
        raise IndexError(f"Selected rows are outside [0, {total})")

    qpos = np.asarray(source["source_qpos"][indices], dtype=np.float32)
    qvel = np.asarray(source["source_qvel"][indices], dtype=np.float32)
    if qpos.shape[1:] != (27,) or qvel.shape[1:] != (26,):
        raise ValueError(f"Expected source_qpos (N,27) and source_qvel (N,26), got {qpos.shape}, {qvel.shape}")
    count = indices.size

    q_target, q_target_available = _optional_rows(
        source,
        ("source_target_qpos", "processed_target", "q_target"),
        indices,
        np.zeros(20, dtype=np.float32),
    )
    if not np.all(q_target_available):
        q_target = qpos[:, :20].copy()
    previous_action, previous_action_available = _optional_rows(
        source,
        ("source_policy_action", "previous_action", "action"),
        indices,
        np.zeros(20, dtype=np.float32),
    )
    if previous_action.ndim == 3:
        previous_action = previous_action[:, 0]

    contact_force, contact_force_available = _optional_rows(
        source,
        ("source_contact_force_n", "contact_force_thumb_index_middle_ring_pinky", "contact_force_n"),
        indices,
        np.full(5, np.nan, dtype=np.float32),
    )
    if contact_force.ndim == 3:
        contact_force = contact_force[:, 0]
    contact_state = np.isfinite(contact_force) & (contact_force > 0.03)
    contact_positions = np.full((count, 5, 3), np.nan, dtype=np.float32)
    source_penetration, source_penetration_available = _optional_rows(
        source,
        ("source_max_penetration_m", "max_penetration_m"),
        indices,
        np.asarray(np.nan, dtype=np.float32),
    )
    if not np.any(source_penetration_available) and "max_penetration_mm" in source:
        source_penetration = np.asarray(
            source["max_penetration_mm"][indices], dtype=np.float32
        ) / 1000.0
        source_penetration_available = np.ones(count, dtype=np.bool_)

    trajectory_ids, trajectory_id_source = _source_trajectory_ids(source, indices, source_hash)
    object_pose = qpos[:, 20:27].copy()
    object_com = object_pose[:, :3] + _quat_rotate_wxyz(
        object_pose[:, 3:7], np.asarray(com_offset, dtype=np.float32)
    )
    state_ids = np.asarray(
        [stable_text_id(f"{source_hash}:{int(row)}:{canonical_id}") for row in indices],
        dtype=np.int64,
    )

    payload: dict[str, np.ndarray] = {
        "metadata_state_id": state_ids,
        "metadata_canonical_id": np.full(count, canonical_id, dtype="U2"),
        "metadata_source_trajectory_id": trajectory_ids,
        "metadata_source_trajectory_id_origin": np.full(count, trajectory_id_source, dtype="U32"),
        "metadata_source_row": indices.astype(np.int64),
        "metadata_source_uri": np.full(count, source_uri, dtype=f"U{max(len(source_uri), 1)}"),
        "metadata_source_sha256": np.full(count, source_hash, dtype="U64"),
        "metadata_derived_source_uri": np.full(
            count, str(source_path), dtype=f"U{max(len(str(source_path)), 1)}"
        ),
        "metadata_derived_source_sha256": np.full(
            count, derived_source_hash, dtype="U64"
        ),
        "metadata_parent_state_id": np.full(count, -1, dtype=np.int64),
        "metadata_physics_variant_id": np.full(count, -1, dtype=np.int64),
        "metadata_asset_hash": np.full(count, asset_hash, dtype=f"U{max(len(asset_hash), 1)}"),
        "metadata_environment_version": np.full(
            count, environment_version, dtype=f"U{max(len(environment_version), 1)}"
        ),
        "metadata_physics_version": np.full(
            count, physics_version, dtype=f"U{max(len(physics_version), 1)}"
        ),
        "metadata_joint_order_version": np.full(count, joint_order_version(), dtype="U71"),
        "metadata_state_schema_version": np.full(count, SCHEMA_VERSION, dtype=f"U{len(SCHEMA_VERSION)}"),
        "robot_qpos": qpos[:, :20],
        "robot_qvel": qvel[:, :20],
        "robot_q_target": np.asarray(q_target, dtype=np.float32),
        "robot_q_target_available": q_target_available,
        "robot_previous_action": np.asarray(previous_action, dtype=np.float32),
        "robot_previous_action_available": previous_action_available,
        "robot_action_history_buffer": np.asarray(previous_action, dtype=np.float32)[:, None, :],
        "robot_action_history_valid_length": previous_action_available.astype(np.int64),
        "robot_control_delay_buffer": np.asarray(previous_action, dtype=np.float32)[:, None, :],
        "robot_control_delay_valid_length": np.zeros(count, dtype=np.int64),
        "robot_control_delay_enabled": np.zeros(count, dtype=np.bool_),
        "robot_q_target_semantics": np.full(
            count, "actual_actuator_target", dtype="U22"
        ),
        "object_pose": object_pose,
        "object_linear_velocity": qvel[:, 20:23],
        "object_angular_velocity": qvel[:, 23:26],
        "object_com": object_com,
        "contact_state": contact_state.astype(np.bool_),
        "contact_state_available": contact_force_available,
        "contact_fingertip_positions": contact_positions,
        "contact_positions_available": np.zeros(count, dtype=np.bool_),
        "contact_positions_frame": np.full(count, "world", dtype="U5"),
        "contact_force": np.asarray(contact_force, dtype=np.float32),
        "contact_force_available": contact_force_available,
        "physics_mass": np.full(count, mass, dtype=np.float32),
        "physics_static_friction": np.full(count, static_friction, dtype=np.float32),
        "physics_dynamic_friction": np.full(count, dynamic_friction, dtype=np.float32),
        "physics_com_offset": np.repeat(np.asarray(com_offset, dtype=np.float32)[None, :], count, axis=0),
        "physics_dt": np.full(count, physics_dt, dtype=np.float32),
        "physics_control_dt": np.full(count, control_dt, dtype=np.float32),
        "physics_solver_position_iterations": np.full(count, solver_position_iterations, dtype=np.int64),
        "physics_solver_velocity_iterations": np.full(count, solver_velocity_iterations, dtype=np.int64),
        "physics_contact_offset": np.full(count, contact_offset, dtype=np.float32),
        "physics_rest_offset": np.full(count, rest_offset, dtype=np.float32),
        "labels_static_or_dynamic": np.full(count, state_type, dtype=f"U{len(state_type)}"),
        "labels_source": np.full(count, source_label, dtype=f"U{max(len(source_label), 1)}"),
        "labels_source_max_penetration_m": np.asarray(
            source_penetration, dtype=np.float32
        ).reshape(count),
        "labels_source_penetration_available": source_penetration_available,
        "labels_validation_status": np.full(count, "candidate", dtype="U10"),
        "labels_r0_pass": np.zeros(count, dtype=np.bool_),
        "labels_r1_pass": np.zeros(count, dtype=np.bool_),
        "labels_r2_pass": np.zeros(count, dtype=np.bool_),
        "labels_r3a_pass": np.zeros(count, dtype=np.bool_),
        "labels_r3b_pass": np.zeros(count, dtype=np.bool_),
        "labels_rejection_code": np.full(count, "not_validated", dtype="U32"),
    }
    continuation_horizon = 8
    continuation_targets = np.repeat(
        np.asarray(q_target, dtype=np.float32)[:, None, :],
        continuation_horizon,
        axis=1,
    )
    continuation_actions = np.repeat(
        np.asarray(previous_action, dtype=np.float32)[:, None, :],
        continuation_horizon,
        axis=1,
    )
    continuation_length = np.zeros(count, dtype=np.int64)
    if "robot_continuation_q_target" in source:
        raw_targets = np.asarray(
            source["robot_continuation_q_target"][indices], dtype=np.float32
        )
        horizon = min(raw_targets.shape[1], continuation_horizon)
        continuation_targets[:, :horizon] = raw_targets[:, :horizon]
        if "robot_continuation_valid_length" in source:
            continuation_length = np.minimum(
                np.asarray(
                    source["robot_continuation_valid_length"][indices],
                    dtype=np.int64,
                ),
                horizon,
            )
        else:
            continuation_length.fill(horizon)
    if "robot_continuation_action" in source:
        raw_actions = np.asarray(
            source["robot_continuation_action"][indices], dtype=np.float32
        )
        horizon = min(raw_actions.shape[1], continuation_horizon)
        continuation_actions[:, :horizon] = raw_actions[:, :horizon]
    payload["robot_continuation_q_target"] = continuation_targets
    payload["robot_continuation_action"] = continuation_actions
    payload["robot_continuation_valid_length"] = continuation_length
    return payload


def row_count(payload: Mapping[str, np.ndarray]) -> int:
    if "metadata_state_id" not in payload:
        raise KeyError("metadata_state_id")
    return int(np.asarray(payload["metadata_state_id"]).shape[0])


def subset_payload(payload: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    count = row_count(payload)
    selected = np.asarray(indices, dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        result[key] = array[selected] if array.ndim > 0 and array.shape[0] == count else array
    return result


def concatenate_payloads(payloads: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not payloads:
        raise ValueError("No payloads to concatenate")
    keys = set(payloads[0])
    if any(set(payload) != keys for payload in payloads[1:]):
        raise ValueError("All canonical payloads must share the same schema keys")
    result: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        arrays = [np.asarray(payload[key]) for payload in payloads]
        result[key] = np.concatenate(arrays, axis=0)
    return result


def validate_r0_arrays(
    payload: Mapping[str, np.ndarray],
    *,
    max_source_penetration_m: float = 0.0005,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = row_count(payload)
    qpos = np.asarray(payload["robot_qpos"], dtype=np.float64)
    qvel = np.asarray(payload["robot_qvel"], dtype=np.float64)
    q_target = np.asarray(payload["robot_q_target"], dtype=np.float64)
    pose = np.asarray(payload["object_pose"], dtype=np.float64)
    linvel = np.asarray(payload["object_linear_velocity"], dtype=np.float64)
    angvel = np.asarray(payload["object_angular_velocity"], dtype=np.float64)
    finite = np.isfinite(qpos).all(1) & np.isfinite(qvel).all(1)
    finite &= np.isfinite(q_target).all(1) & np.isfinite(pose).all(1)
    finite &= np.isfinite(linvel).all(1) & np.isfinite(angvel).all(1)
    action_history = np.asarray(payload["robot_action_history_buffer"], dtype=np.float64)
    delay_buffer = np.asarray(payload["robot_control_delay_buffer"], dtype=np.float64)
    continuation_target = np.asarray(
        payload["robot_continuation_q_target"], dtype=np.float64
    )
    continuation_action = np.asarray(
        payload["robot_continuation_action"], dtype=np.float64
    )
    finite &= np.isfinite(action_history).all(axis=(1, 2))
    finite &= np.isfinite(delay_buffer).all(axis=(1, 2))
    finite &= np.isfinite(continuation_target).all(axis=(1, 2))
    finite &= np.isfinite(continuation_action).all(axis=(1, 2))
    shape_valid = np.full(
        count,
        qpos.shape == (count, 20)
        and qvel.shape == (count, 20)
        and q_target.shape == (count, 20)
        and pose.shape == (count, 7)
        and linvel.shape == (count, 3)
        and angvel.shape == (count, 3)
        and action_history.ndim == 3
        and action_history.shape[0] == count
        and action_history.shape[2] == 20
        and delay_buffer.ndim == 3
        and delay_buffer.shape[0] == count
        and delay_buffer.shape[2] == 20
        and continuation_target.shape == (count, 8, 20)
        and continuation_action.shape == (count, 8, 20),
        dtype=np.bool_,
    )
    quat_norm = np.linalg.norm(pose[:, 3:7], axis=1)
    quaternion_valid = np.isfinite(quat_norm) & (np.abs(quat_norm - 1.0) <= 1.0e-3)
    target_available = np.asarray(payload["robot_q_target_available"], dtype=np.bool_)
    previous_action_available = np.asarray(
        payload["robot_previous_action_available"], dtype=np.bool_
    )
    metadata_valid = np.ones(count, dtype=np.bool_)
    for key in (
        "metadata_canonical_id",
        "metadata_source_trajectory_id",
        "metadata_source_uri",
        "metadata_source_sha256",
        "metadata_derived_source_uri",
        "metadata_derived_source_sha256",
        "metadata_asset_hash",
        "metadata_environment_version",
        "metadata_physics_version",
        "metadata_joint_order_version",
        "metadata_state_schema_version",
        "metadata_split",
        "contact_positions_frame",
    ):
        values = np.asarray(payload[key])
        if values.dtype.kind in "US":
            metadata_valid &= np.char.str_len(values.astype("U")) > 0
        else:
            metadata_valid &= np.isfinite(values)
    metadata_valid &= np.asarray(payload["metadata_state_schema_version"]) == SCHEMA_VERSION
    metadata_valid &= np.isin(np.asarray(payload["metadata_canonical_id"]), CANONICAL_IDS)
    metadata_valid &= np.isin(np.asarray(payload["labels_static_or_dynamic"]), STATE_TYPES)
    metadata_valid &= np.isin(
        np.asarray(payload["metadata_split"]), ("train", "validation", "test")
    )
    metadata_valid &= np.asarray(payload["contact_positions_frame"]) == "world"

    continuation_length = np.asarray(
        payload["robot_continuation_valid_length"], dtype=np.int64
    )
    action_history_length = np.asarray(
        payload["robot_action_history_valid_length"], dtype=np.int64
    )
    delay_length = np.asarray(
        payload["robot_control_delay_valid_length"], dtype=np.int64
    )
    control_context_valid = (
        (continuation_length >= 0)
        & (continuation_length <= continuation_target.shape[1])
        & (action_history_length >= 0)
        & (action_history_length <= action_history.shape[1])
        & (delay_length >= 0)
        & (delay_length <= delay_buffer.shape[1])
    )

    physics_valid = np.ones(count, dtype=np.bool_)
    for key in (
        "physics_mass",
        "physics_static_friction",
        "physics_dynamic_friction",
        "physics_dt",
        "physics_control_dt",
        "physics_contact_offset",
        "physics_rest_offset",
    ):
        physics_valid &= np.isfinite(np.asarray(payload[key], dtype=np.float64))
    physics_valid &= np.asarray(payload["physics_mass"]) > 0.0
    physics_valid &= np.asarray(payload["physics_static_friction"]) >= 0.0
    physics_valid &= np.asarray(payload["physics_dynamic_friction"]) >= 0.0
    physics_valid &= np.isfinite(
        np.asarray(payload["physics_com_offset"], dtype=np.float64)
    ).all(axis=1)

    joint_valid = np.ones(count, dtype=np.bool_)
    if joint_lower is not None and joint_upper is not None:
        lower = np.asarray(joint_lower, dtype=np.float64).reshape(1, 20)
        upper = np.asarray(joint_upper, dtype=np.float64).reshape(1, 20)
        tolerance = 1.0e-5
        joint_valid &= (qpos >= lower - tolerance).all(1) & (qpos <= upper + tolerance).all(1)
        joint_valid &= (q_target >= lower - tolerance).all(1) & (q_target <= upper + tolerance).all(1)

    source_penetration = np.asarray(
        payload.get("labels_source_max_penetration_m", np.full(count, np.nan)),
        dtype=np.float64,
    )
    source_penetration_available = np.asarray(
        payload.get("labels_source_penetration_available", np.zeros(count)),
        dtype=np.bool_,
    )
    penetration_valid = ~source_penetration_available | (
        np.isfinite(source_penetration)
        & (source_penetration <= max_source_penetration_m)
    )
    passed = (
        finite
        & shape_valid
        & quaternion_valid
        & target_available
        & previous_action_available
        & metadata_valid
        & control_context_valid
        & physics_valid
        & joint_valid
        & penetration_valid
    )
    checks = {
        "finite": finite,
        "shape_valid": shape_valid,
        "quaternion_valid": quaternion_valid,
        "target_available": target_available,
        "previous_action_available": previous_action_available,
        "metadata_valid": metadata_valid,
        "control_context_valid": control_context_valid,
        "physics_valid": physics_valid,
        "joint_valid": joint_valid,
        "penetration_valid": penetration_valid,
    }
    return passed, checks


def grouped_split(
    source_trajectory_ids: np.ndarray,
    *,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
) -> np.ndarray:
    ids = np.asarray(source_trajectory_ids, dtype=np.int64)
    unique = np.unique(ids)
    assignment: dict[int, str] = {}
    for value in unique:
        bucket = int(stable_text_id(f"split:{int(value)}") % 10_000) / 10_000.0
        if bucket < train_fraction:
            assignment[int(value)] = "train"
        elif bucket < train_fraction + validation_fraction:
            assignment[int(value)] = "validation"
        else:
            assignment[int(value)] = "test"
    return np.asarray([assignment[int(value)] for value in ids], dtype="U10")


def save_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
