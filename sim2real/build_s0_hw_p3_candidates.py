from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim2real.dex5_hardware_adapter import (
    JOINT_LOWER,
    JOINT_UPPER,
    Dex5HardwareCompatibilityAdapter,
)
from dex5_penspin.isaac.canonical_reset_schema import (
    save_npz_atomic,
    sha256_file,
    stable_text_id,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0/"
    "canonical_reset_bank/P3_v9_physx_projected_R3_validated.npz"
)
DEFAULT_OUTPUT = ROOT / (
    "Dex5_Sim2Real_S0/reset_distribution/candidate/"
    "Dex5_S0_HW_P3_projected_candidates.npz"
)
S0_SCHEMA = "dex5-s0-hw-reset-v1"


def _project(values: np.ndarray) -> np.ndarray:
    shape = values.shape
    return Dex5HardwareCompatibilityAdapter.project_virtual_target(
        np.asarray(values).reshape(-1, 20)
    ).reshape(shape).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--q-noise", type=float, default=0.0005)
    parser.add_argument("--dq-noise", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    source_hash = sha256_file(source_path)
    with np.load(source_path, allow_pickle=False) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    if np.unique(source["metadata_canonical_id"]).tolist() != ["P3"]:
        raise ValueError("S0 candidate source must contain only P3")
    if args.variants < 1:
        raise ValueError("variants must be positive")

    source_count = source["robot_qpos"].shape[0]
    parent_rows = np.repeat(np.arange(source_count, dtype=np.int64), args.variants)
    variant_ids = np.tile(np.arange(args.variants, dtype=np.int64), source_count)
    payload: dict[str, np.ndarray] = {}
    for key, value in source.items():
        array = np.asarray(value)
        payload[key] = (
            array[parent_rows]
            if array.ndim > 0 and array.shape[0] == source_count
            else array.copy()
        )

    rng = np.random.default_rng(args.seed)
    q_noise = rng.normal(0.0, args.q_noise, size=payload["robot_qpos"].shape)
    dq_noise = rng.normal(0.0, args.dq_noise, size=payload["robot_qvel"].shape)
    exact = variant_ids == 0
    q_noise[exact] = 0.0
    dq_noise[exact] = 0.0
    payload["robot_qpos"] = np.clip(
        payload["robot_qpos"] + q_noise, JOINT_LOWER, JOINT_UPPER
    ).astype(np.float32)
    payload["robot_qvel"] = (payload["robot_qvel"] + dq_noise).astype(np.float32)

    original_target = np.asarray(payload["robot_q_target"], dtype=np.float32).copy()
    projected_target = _project(original_target)
    residual = projected_target - original_target
    payload["robot_q_target"] = projected_target
    payload["robot_continuation_q_target"] = _project(
        payload["robot_continuation_q_target"]
    )
    payload["robot_q_target_semantics"] = np.full(
        parent_rows.size, "s0_coupled_abs_target", dtype="U22"
    )

    original_state_ids = np.asarray(source["metadata_state_id"], dtype=np.int64)[
        parent_rows
    ]
    projected_state_ids = np.asarray(
        [
            stable_text_id(
                f"{S0_SCHEMA}:{source_hash}:{int(parent)}:{int(variant)}:{args.seed}"
            )
            for parent, variant in zip(original_state_ids, variant_ids, strict=True)
        ],
        dtype=np.int64,
    )
    payload["metadata_parent_state_id"] = original_state_ids
    payload["metadata_state_id"] = projected_state_ids
    payload["metadata_s0_original_state_id"] = original_state_ids
    payload["metadata_s0_projected_state_id"] = projected_state_ids
    payload["metadata_s0_candidate_variant_id"] = variant_ids
    payload["metadata_s0_schema_version"] = np.full(
        parent_rows.size, S0_SCHEMA, dtype=f"U{len(S0_SCHEMA)}"
    )
    payload["metadata_derived_source_uri"] = np.full(
        parent_rows.size, str(source_path), dtype=f"U{len(str(source_path))}"
    )
    payload["metadata_derived_source_sha256"] = np.full(
        parent_rows.size, source_hash, dtype="U64"
    )
    payload["s0_original_robot_q_target"] = original_target
    payload["s0_projection_residual"] = residual.astype(np.float32)
    payload["s0_projection_residual_rms_rad"] = np.sqrt(
        np.mean(np.square(residual), axis=1)
    ).astype(np.float32)
    payload["s0_projection_residual_max_rad"] = np.max(
        np.abs(residual), axis=1
    ).astype(np.float32)
    payload["s0_logical_actuator_target"] = (
        Dex5HardwareCompatibilityAdapter.virtual_to_logical_actuator(projected_target)
        .astype(np.float32)
    )
    payload["s0_adapter_config"] = np.full(
        parent_rows.size,
        "shared_pairs=2:3,6:7,10:11,14:15;distal_wins",
        dtype="U55",
    )
    payload["labels_source"] = np.full(
        parent_rows.size, "S0 projected P3 candidate", dtype="U25"
    )
    payload["labels_validation_status"] = np.full(
        parent_rows.size, "candidate", dtype="U10"
    )
    for key in (
        "labels_r0_pass",
        "labels_r1_pass",
        "labels_r2_pass",
        "labels_r3a_pass",
        "labels_r3b_pass",
    ):
        payload[key] = np.zeros(parent_rows.size, dtype=np.bool_)
    payload["labels_rejection_code"] = np.full(
        parent_rows.size, "not_validated", dtype="U32"
    )

    save_npz_atomic(output_path, payload)
    report = {
        "status": "candidate_only_not_physx_settled",
        "source": str(source_path),
        "source_sha256": source_hash,
        "o0_modified": False,
        "source_rows": source_count,
        "variants": args.variants,
        "candidate_rows": int(parent_rows.size),
        "unique_source_trajectories": int(
            np.unique(payload["metadata_source_trajectory_id"]).size
        ),
        "projection_residual_rms_mean_rad": float(
            payload["s0_projection_residual_rms_rad"].mean()
        ),
        "projection_residual_max_rad": float(
            payload["s0_projection_residual_max_rad"].max()
        ),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json_atomic(output_path.with_suffix(".json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
