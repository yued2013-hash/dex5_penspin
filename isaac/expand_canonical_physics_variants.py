from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from canonical_reset_schema import save_npz_atomic, sha256_file, stable_text_id, write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand R3-validated canonical states across physics context only."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--mass-scale", type=float, default=0.02)
    parser.add_argument("--friction-scale", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    input_path = _resolve(args.input).resolve()
    output_path = _resolve(args.output).resolve()
    with np.load(input_path, allow_pickle=False) as loaded:
        source = {key: loaded[key] for key in loaded.files}
    final_pass = (
        np.asarray(source["labels_r0_pass"], dtype=np.bool_)
        & np.asarray(source["labels_r1_pass"], dtype=np.bool_)
        & np.asarray(source["labels_r2_pass"], dtype=np.bool_)
        & np.asarray(source["labels_r3a_pass"], dtype=np.bool_)
        & np.asarray(source["labels_r3b_pass"], dtype=np.bool_)
    )
    parent_rows = np.flatnonzero(final_pass)
    if parent_rows.size == 0:
        raise ValueError("Input has no fully R0-R3 validated parent states")
    variants = int(args.variants)
    if variants < 1:
        raise ValueError("variants must be positive")
    repeated = np.repeat(parent_rows, variants)
    source_count = source["metadata_state_id"].shape[0]
    payload: dict[str, np.ndarray] = {}
    for key, value in source.items():
        array = np.asarray(value)
        payload[key] = array[repeated] if array.ndim > 0 and array.shape[0] == source_count else array

    rng = np.random.default_rng(args.seed)
    mass_multiplier = rng.uniform(
        1.0 - args.mass_scale, 1.0 + args.mass_scale, repeated.size
    ).astype(np.float32)
    static_multiplier = rng.uniform(
        1.0 - args.friction_scale, 1.0 + args.friction_scale, repeated.size
    ).astype(np.float32)
    dynamic_multiplier = rng.uniform(
        1.0 - args.friction_scale, 1.0 + args.friction_scale, repeated.size
    ).astype(np.float32)
    payload["physics_mass"] = payload["physics_mass"] * mass_multiplier
    payload["physics_static_friction"] = (
        payload["physics_static_friction"] * static_multiplier
    )
    payload["physics_dynamic_friction"] = np.minimum(
        payload["physics_dynamic_friction"] * dynamic_multiplier,
        payload["physics_static_friction"],
    )

    parent_state_ids = np.asarray(source["metadata_state_id"])[repeated]
    variant_ids = np.tile(np.arange(variants, dtype=np.int64), parent_rows.size)
    payload["metadata_parent_state_id"] = parent_state_ids
    payload["metadata_physics_variant_id"] = variant_ids
    if "metadata_derived_source_uri" not in payload:
        payload["metadata_derived_source_uri"] = payload["metadata_source_uri"].copy()
    if "metadata_derived_source_sha256" not in payload:
        payload["metadata_derived_source_sha256"] = payload[
            "metadata_source_sha256"
        ].copy()
    if "contact_positions_frame" not in payload:
        payload["contact_positions_frame"] = np.full(
            repeated.size, "world", dtype="U5"
        )
    if "robot_control_delay_buffer" not in payload:
        payload["robot_control_delay_buffer"] = payload[
            "robot_action_history_buffer"
        ].copy()
        payload["robot_control_delay_valid_length"] = np.zeros(
            repeated.size, dtype=np.int64
        )
        payload["robot_control_delay_enabled"] = np.zeros(
            repeated.size, dtype=np.bool_
        )
    payload["metadata_state_id"] = np.asarray(
        [
            stable_text_id(f"physics-variant:{int(parent)}:{int(variant)}:{args.seed}")
            for parent, variant in zip(parent_state_ids, variant_ids, strict=True)
        ],
        dtype=np.int64,
    )
    payload["labels_source"] = np.full(
        repeated.size, "R3 parent physics-context expansion", dtype="U35"
    )
    payload["labels_validation_status"] = np.full(
        repeated.size, "candidate", dtype="U10"
    )
    for key in (
        "labels_r0_pass",
        "labels_r1_pass",
        "labels_r2_pass",
        "labels_r3a_pass",
        "labels_r3b_pass",
    ):
        payload[key] = np.zeros(repeated.size, dtype=np.bool_)
    payload["labels_rejection_code"] = np.full(
        repeated.size, "not_validated", dtype="U32"
    )

    save_npz_atomic(output_path, payload)
    report = {
        "status": "candidate_only_not_physx_validated",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "validated_parent_states": int(parent_rows.size),
        "variants_per_parent": variants,
        "candidate_states": int(repeated.size),
        "unique_source_trajectories": int(
            np.unique(payload["metadata_source_trajectory_id"]).size
        ),
        "mass_scale": args.mass_scale,
        "friction_scale": args.friction_scale,
        "seed": args.seed,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json_atomic(output_path.with_suffix(".json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
