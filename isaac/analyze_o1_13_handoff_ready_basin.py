from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from canonical_reset_schema import JOINT_ORDER


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
FEATURES_PATH = REPORT_DIR / "O1_9_P3_initial_feature_snapshot.json"
O1_12_PATH = REPORT_DIR / "O1_12_Post_Handoff_Axis_Stabilization_Audit.json"
BANK_PATH = (
    ROOT
    / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0/"
    "canonical_reset_bank/P1_P5_v9_physx_projected_R3_validated.npz"
)
OUTPUT = REPORT_DIR / "O1_13_Handoff_Readiness_Candidates.json"

FEASIBLE_PROTOTYPE = 996
FAILURE_PROTOTYPES = (1245, 1171, 1203)
BASELINE_IDS = (771, 817, 857, 900, 948, 996, 1040, 1083, 1130, 1175, 1216, 1252)
SELECT_COUNT = 12


def _group_distance(
    standardized: np.ndarray,
    prototype: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    distances = []
    for group in np.unique(groups):
        mask = groups == group
        distances.append(
            np.sqrt(np.mean((standardized[:, mask] - prototype[mask]) ** 2, axis=1))
        )
    return np.mean(np.stack(distances, axis=1), axis=1)


def main() -> None:
    snapshot = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    o1_12 = json.loads(O1_12_PATH.read_text(encoding="utf-8"))
    bank_ids = np.asarray(snapshot["bank_ids"], dtype=np.int64)
    features = np.asarray(snapshot["features"], dtype=np.float64)
    feature_names = list(snapshot["feature_names"])
    feature_groups = list(snapshot["feature_groups"])
    parents = np.asarray(snapshot["parent_state_ids"], dtype=np.int64)
    sources = np.asarray(snapshot["source_trajectory_ids"], dtype=np.int64)
    if snapshot["bank_sha256"] != o1_12["bank_sha256"]:
        raise RuntimeError("O1.9 and O1.12 do not use the same frozen O0 bank")

    with np.load(BANK_PATH, allow_pickle=False) as bank:
        qpos = np.asarray(bank["robot_qpos"][bank_ids], dtype=np.float64)
    features = np.concatenate((features, qpos), axis=1)
    feature_names.extend(f"joint_{name}_rad" for name in JOINT_ORDER)
    feature_groups.extend("joint_configuration" for _ in JOINT_ORDER)
    feature_groups_array = np.asarray(feature_groups)

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    variable = std > 1.0e-8
    standardized = np.clip(
        (features[:, variable] - mean[variable]) / std[variable], -5.0, 5.0
    )
    active_names = np.asarray(feature_names)[variable]
    active_groups = feature_groups_array[variable]
    row_by_id = {int(bank_id): index for index, bank_id in enumerate(bank_ids)}
    prototype_ids = (FEASIBLE_PROTOTYPE, *FAILURE_PROTOTYPES)
    missing = [bank_id for bank_id in prototype_ids if bank_id not in row_by_id]
    if missing:
        raise RuntimeError(f"Prototype bank IDs absent from O1.9 features: {missing}")

    feasible_vector = standardized[row_by_id[FEASIBLE_PROTOTYPE]]
    feasible_distance = _group_distance(
        standardized, feasible_vector, active_groups
    )
    negative_distance_matrix = np.stack(
        [
            _group_distance(
                standardized, standardized[row_by_id[bank_id]], active_groups
            )
            for bank_id in FAILURE_PROTOTYPES
        ],
        axis=1,
    )
    nearest_failure_distance = negative_distance_matrix.min(axis=1)
    prototype_margin = nearest_failure_distance - feasible_distance
    margin_scale = max(float(np.std(prototype_margin)), 1.0e-6)
    readiness = 1.0 / (1.0 + np.exp(-prototype_margin / margin_scale))

    excluded_parents = {int(parents[row_by_id[value]]) for value in prototype_ids}
    excluded_sources = {int(sources[row_by_id[value]]) for value in prototype_ids}
    order = np.argsort(readiness)[::-1]
    selected = []
    selected_parents: set[int] = set()
    selected_sources: set[int] = set()
    for row in order:
        bank_id = int(bank_ids[row])
        parent = int(parents[row])
        source = int(sources[row])
        if bank_id in prototype_ids:
            continue
        if parent in excluded_parents or source in excluded_sources:
            continue
        if parent in selected_parents or source in selected_sources:
            continue
        selected.append(row)
        selected_parents.add(parent)
        selected_sources.add(source)
        if len(selected) == SELECT_COUNT:
            break
    if len(selected) != SELECT_COUNT:
        raise RuntimeError("Could not select 12 provenance-independent readiness rows")

    negative_rows = np.asarray([row_by_id[value] for value in FAILURE_PROTOTYPES])
    contrast = np.abs(
        standardized[row_by_id[FEASIBLE_PROTOTYPE]]
        - standardized[negative_rows].mean(axis=0)
    )
    top_features = np.argsort(contrast)[-20:][::-1]
    group_contrast = {}
    prototype_group_distance = {}
    top_features_by_group = {}
    for group in np.unique(active_groups):
        mask = active_groups == group
        group_contrast[str(group)] = float(np.sqrt(np.mean(contrast[mask] ** 2)))
        prototype_group_distance[str(group)] = {
            str(bank_id): float(
                np.sqrt(
                    np.mean(
                        (
                            standardized[row_by_id[bank_id], mask]
                            - feasible_vector[mask]
                        )
                        ** 2
                    )
                )
            )
            for bank_id in prototype_ids
        }
        group_indices = np.flatnonzero(mask)
        selected_group_features = group_indices[
            np.argsort(contrast[group_indices])[-min(5, group_indices.size) :][::-1]
        ]
        top_features_by_group[str(group)] = [
            {
                "name": str(active_names[index]),
                "absolute_contrast_z": float(contrast[index]),
                "prototype_values": {
                    str(bank_id): float(
                        features[row_by_id[bank_id], np.flatnonzero(variable)[index]]
                    )
                    for bank_id in prototype_ids
                },
            }
            for index in selected_group_features
        ]

    centered = standardized - standardized.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pca = centered @ vh[:2].T
    selected_ids = [int(bank_ids[row]) for row in selected]
    selected_set = set(selected_ids)
    baseline_set = set(BASELINE_IDS)
    labels = []
    for bank_id in bank_ids:
        value = int(bank_id)
        if value == FEASIBLE_PROTOTYPE:
            labels.append("feasible_prototype")
        elif value in FAILURE_PROTOTYPES:
            labels.append("failure_prototype")
        elif value in selected_set:
            labels.append("readiness_selected")
        elif value in baseline_set:
            labels.append("spread_baseline")
        else:
            labels.append("P3_pool")

    report = {
        "status": "PASS_ANALYSIS_PENDING_CEM_VALIDATION",
        "phase": "O1.13 Handoff-Ready Canonical Basin Analysis",
        "artifact_role": "read_only_score_and_selection_not_reset_dataset",
        "bank_sha256": snapshot["bank_sha256"],
        "P3_rows": int(bank_ids.size),
        "feature_schema": {
            "features": int(variable.sum()),
            "groups": sorted(set(active_groups.tolist())),
            "group_balancing": "equal mean of per-group RMS standardized distances",
            "joint_order": list(JOINT_ORDER),
        },
        "score_definition": {
            "name": "prototype_margin_readiness",
            "formula": (
                "sigmoid((distance_to_nearest_failure_prototype - "
                "distance_to_bank996) / population_margin_std)"
            ),
            "calibration": "not a probability; one feasible prototype is insufficient for a classifier",
            "feasible_prototype": FEASIBLE_PROTOTYPE,
            "failure_prototypes": {
                "axis_misaligned": [1245, 1171],
                "transition_not_reproduced": [1203],
            },
        },
        "prototype_comparison": {
            str(bank_id): {
                "readiness_score": float(readiness[row_by_id[bank_id]]),
                "distance_to_996": float(feasible_distance[row_by_id[bank_id]]),
                "distance_to_nearest_failure": float(
                    nearest_failure_distance[row_by_id[bank_id]]
                ),
                "O1_12_outcome": next(
                    row["primary_failure"]
                    for row in o1_12["post_handoff_search"]["states"]
                    if row["bank_id"] == bank_id
                ),
            }
            for bank_id in prototype_ids
        },
        "group_contrast_feasible_vs_failure_mean_RMS_z": group_contrast,
        "prototype_group_RMS_z_distance_to_996": prototype_group_distance,
        "top_features_by_group": top_features_by_group,
        "top_contrast_features": [
            {
                "name": str(active_names[index]),
                "group": str(active_groups[index]),
                "absolute_contrast_z": float(contrast[index]),
            }
            for index in top_features
        ],
        "readiness_selected": {
            "bank_ids": selected_ids,
            "scores": [float(readiness[row]) for row in selected],
            "parent_ids": [int(parents[row]) for row in selected],
            "source_trajectory_ids": [int(sources[row]) for row in selected],
            "prototype_parent_or_source_leakage": False,
            "within_selection_parent_or_source_leakage": False,
        },
        "spread_baseline": {
            "bank_ids": list(BASELINE_IDS),
            "definition": "the original O1.8 12-entry spread sample",
        },
        "basin_map": {
            "bank_ids": bank_ids.tolist(),
            "pca_coordinates": pca.tolist(),
            "readiness_scores": readiness.tolist(),
            "labels": labels,
        },
        "constraints": {
            "PPO_trained": False,
            "reward_modified": False,
            "teacher_generated": False,
            "O0_modified": False,
            "new_reset_dataset_created": False,
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selected_bank_ids": selected_ids,
                "selected_score_range": [
                    min(report["readiness_selected"]["scores"]),
                    max(report["readiness_selected"]["scores"]),
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
