from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
OUTPUT = REPORT / "v9_p3_to_braked_p4_lineage_teacher200.npz"

STAGES = (
    ("P3_to_7", "v9_p3_entry7_strict12_teacher200.npz", "v9_p3_to_entry7_physx_bundle.npz"),
    ("7_to_20", "v9_entry7_approach20_strict12_teacher200.npz", "v9_entry7_to_approach20_physx_bundle.npz"),
    ("20_to_25", "v9_approach20_entry25_teacher200.npz", "v9_approach20_to_entry25_physx_bundle.npz"),
    ("25_to_30", "v9_entry25_mid30_teacher200.npz", "v9_entry25_to_mid30_physx_bundle.npz"),
    ("30_to_35", "v9_mid30_mid35_teacher200.npz", "v9_mid30_to_mid35_physx_bundle.npz"),
    ("35_to_40", "v9_mid35_approach40_strict05_teacher200.npz", "v9_mid35_to_approach40_physx_bundle.npz"),
    ("40_to_42_bootstrap", "v9_bridge42_bootstrap_teacher200.npz", "v9_approach40_to_bridge42_physx_bundle.npz"),
    ("42_strict", "v9_bridge42_strict_hold6_teacher200.npz", "v9_bridge42_strict_cache_bundle.npz"),
    ("42_to_45", "v9_bridge42_entry45_teacher200.npz", "v9_bridge42_to_entry45_physx_bundle.npz"),
    ("45_to_47", "v9_entry45_approach47_teacher200.npz", "v9_entry45_to_approach47_physx_bundle.npz"),
    ("47_to_braked_P4", "v9_entry45_braked50_teacher200.npz", "v9_entry45_to_braked50_physx_bundle.npz"),
)

TRACE_KEYS = (
    "qpos",
    "qvel",
    "action",
    "contact_force_n",
    "phase_deg",
    "angular_speed_rad_s",
    "linear_speed_m_s",
    "drift_m",
    "hand_rmse_rad",
    "max_penetration_m",
    "hold_count",
)


def nearest_rows(query: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(query[:, None, :] - bank[None, :, :], axis=-1)
    indices = np.argmin(distance, axis=1)
    return indices.astype(np.int64), distance[np.arange(len(query)), indices]


def main() -> None:
    trajectories = [np.load(REPORT / trajectory) for _, trajectory, _ in STAGES]
    bundles = [np.load(REPORT / bundle) for _, _, bundle in STAGES]
    chain_count = len(trajectories[-1]["lengths"])
    stage_indices = np.zeros((chain_count, len(STAGES)), dtype=np.int64)
    stage_indices[:, -1] = np.arange(chain_count)
    reset_match_error = np.zeros((chain_count, len(STAGES)), dtype=np.float32)

    for stage in range(len(STAGES) - 1, 0, -1):
        current_rows = stage_indices[:, stage]
        current_initial = trajectories[stage]["initial_qpos"][current_rows]
        parent_rows, error = nearest_rows(current_initial, bundles[stage]["source_qpos"])
        stage_indices[:, stage - 1] = parent_rows
        reset_match_error[:, stage] = error

    first_rows, first_error = nearest_rows(
        trajectories[0]["initial_qpos"][stage_indices[:, 0]], bundles[0]["source_qpos"]
    )
    reset_match_error[:, 0] = first_error

    stage_lengths = np.zeros_like(stage_indices)
    chain_lengths = np.zeros(chain_count, dtype=np.int64)
    assembled: dict[str, list[np.ndarray]] = {key: [] for key in TRACE_KEYS}
    boundary_qpos_error = np.zeros((chain_count, len(STAGES) - 1), dtype=np.float32)
    boundary_qvel_error = np.zeros_like(boundary_qpos_error)
    for chain in range(chain_count):
        pieces: dict[str, list[np.ndarray]] = {key: [] for key in TRACE_KEYS}
        for stage, trajectory in enumerate(trajectories):
            row = stage_indices[chain, stage]
            length = int(trajectory["lengths"][row])
            stage_lengths[chain, stage] = length
            for key in TRACE_KEYS:
                pieces[key].append(np.asarray(trajectory[key][row, :length]))
            if stage > 0:
                previous = trajectories[stage - 1]
                previous_row = stage_indices[chain, stage - 1]
                previous_length = int(previous["lengths"][previous_row])
                boundary_qpos_error[chain, stage - 1] = np.linalg.norm(
                    previous["qpos"][previous_row, previous_length - 1]
                    - trajectory["initial_qpos"][row]
                )
                boundary_qvel_error[chain, stage - 1] = np.linalg.norm(
                    previous["qvel"][previous_row, previous_length - 1]
                    - trajectory["initial_qvel"][row]
                )
        chain_lengths[chain] = sum(len(piece) for piece in pieces["qpos"])
        for key in TRACE_KEYS:
            assembled[key].append(np.concatenate(pieces[key], axis=0))

    max_length = int(chain_lengths.max())
    payload: dict[str, np.ndarray] = {
        "lengths": chain_lengths,
        "stage_names": np.asarray([stage[0] for stage in STAGES]),
        "stage_indices": stage_indices,
        "stage_lengths": stage_lengths,
        "initial_reset_bank_id": first_rows,
        "reset_match_qpos_error": reset_match_error,
        "boundary_qpos_error": boundary_qpos_error,
        "boundary_qvel_error": boundary_qvel_error,
        "initial_qpos": trajectories[0]["initial_qpos"][stage_indices[:, 0]],
        "initial_qvel": trajectories[0]["initial_qvel"][stage_indices[:, 0]],
        "control_dt_s": np.asarray(trajectories[-1]["control_dt_s"]),
    }
    for key, rows in assembled.items():
        tail = rows[0].shape[1:]
        array = np.zeros((chain_count, max_length, *tail), dtype=rows[0].dtype)
        for row, values in enumerate(rows):
            array[row, : len(values)] = values
        payload[key] = array
    np.savez_compressed(OUTPUT, **payload)

    report = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "trajectory_count": chain_count,
        "stage_count": len(STAGES),
        "stages": [stage[0] for stage in STAGES],
        "length_steps": {
            "min": int(chain_lengths.min()),
            "mean": float(chain_lengths.mean()),
            "max": int(chain_lengths.max()),
        },
        "lineage_reset_match_qpos_error_max": float(reset_match_error.max()),
        "stitched_boundary_qpos_error_max": float(boundary_qpos_error.max()),
        "stitched_boundary_qvel_error_max": float(boundary_qvel_error.max()),
        "terminal_phase_deg": {
            "min": float(min(row[length - 1] for row, length in zip(payload["phase_deg"], chain_lengths))),
            "max": float(max(row[length - 1] for row, length in zip(payload["phase_deg"], chain_lengths))),
        },
        "max_penetration_mm": float(
            max(np.max(row[:length]) for row, length in zip(payload["max_penetration_m"], chain_lengths))
            * 1000.0
        ),
        "warning": "This lineage artifact preserves parentage but still contains reset-boundary jumps; use continuous PhysX replay before calling it uninterrupted.",
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
