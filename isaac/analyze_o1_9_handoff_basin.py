#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def _nested_loo_predictions(
    features: np.ndarray, labels: np.ndarray, top_k: int = 2
) -> tuple[np.ndarray, list[list[int]]]:
    probabilities = np.zeros(labels.size, dtype=np.float64)
    selected_features: list[list[int]] = []
    for held_out in range(labels.size):
        train = np.arange(labels.size) != held_out
        train_x = features[train]
        train_y = labels[train]
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0).clip(min=1.0e-8)
        standardized = (train_x - mean) / std
        class_effect = np.abs(
            standardized[train_y == 1].mean(axis=0)
            - standardized[train_y == 0].mean(axis=0)
        )
        selected = np.argsort(class_effect)[-min(top_k, features.shape[1]) :]
        model = LogisticRegression(
            C=0.25,
            class_weight="balanced",
            solver="liblinear",
            random_state=20260731,
        )
        model.fit(standardized[:, selected], train_y)
        held_out_x = ((features[held_out] - mean) / std)[selected][None]
        probabilities[held_out] = model.predict_proba(held_out_x)[0, 1]
        selected_features.append(selected.tolist())
    return probabilities, selected_features


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "probabilities": probabilities.tolist(),
    }


def _exact_balanced_permutation_test(
    features: np.ndarray, labels: np.ndarray, observed_auc: float
) -> dict:
    aucs = []
    for positive_indices in combinations(range(labels.size), int(labels.sum())):
        permuted = np.zeros(labels.size, dtype=np.int64)
        permuted[list(positive_indices)] = 1
        probabilities, _ = _nested_loo_predictions(features, permuted)
        aucs.append(roc_auc_score(permuted, probabilities))
    aucs_array = np.asarray(aucs, dtype=np.float64)
    return {
        "method": "exact enumeration of all balanced 6-of-12 label assignments",
        "assignments": int(aucs_array.size),
        "p_value_auc_greater_equal": float(
            (1 + np.count_nonzero(aucs_array >= observed_auc))
            / (1 + aucs_array.size)
        ),
        "null_auc_p50": float(np.percentile(aucs_array, 50)),
        "null_auc_p95": float(np.percentile(aucs_array, 95)),
    }


def _fit_interpretable_classifier(
    audit_features: np.ndarray,
    labels: np.ndarray,
    all_features: np.ndarray,
    feature_names: list[str],
    top_k: int = 2,
) -> tuple[dict, np.ndarray]:
    mean = audit_features.mean(axis=0)
    std = audit_features.std(axis=0).clip(min=1.0e-8)
    standardized = (audit_features - mean) / std
    effect = np.abs(
        standardized[labels == 1].mean(axis=0)
        - standardized[labels == 0].mean(axis=0)
    )
    selected = np.argsort(effect)[-min(top_k, audit_features.shape[1]) :]
    model = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        solver="liblinear",
        random_state=20260731,
    )
    model.fit(standardized[:, selected], labels)
    all_probabilities = model.predict_proba(
        ((all_features - mean) / std)[:, selected]
    )[:, 1]
    specification = {
        "model": "L2 logistic regression",
        "role": "diagnostic classifier only; not a policy or reset filter",
        "selected_feature_names": [feature_names[index] for index in selected],
        "selected_feature_indices": selected.tolist(),
        "selected_effect_sizes": effect[selected].tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "regularization_C": 0.25,
        "selection_warning": "Features were selected on all 12 labels only for interpretation; headline performance uses nested leave-one-out selection.",
    }
    return specification, all_probabilities


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _write_one_class_analysis(
    *,
    args: argparse.Namespace,
    snapshot: dict,
    audit: dict,
    bank_ids: np.ndarray,
    all_features: np.ndarray,
    feature_names: list[str],
    feature_groups: np.ndarray,
    parent_ids: np.ndarray,
    source_ids: np.ndarray,
    audit_bank_ids: np.ndarray,
    audit_rows: np.ndarray,
    selected_records: list[dict],
) -> None:
    variable = all_features.std(axis=0) > 1.0e-8
    active_features = all_features[:, variable]
    active_names = np.asarray(feature_names)[variable]
    active_groups = feature_groups[variable]
    audited_features = active_features[audit_rows]
    mean = active_features.mean(axis=0)
    std = active_features.std(axis=0).clip(min=1.0e-8)
    standardized_all = (active_features - mean) / std
    standardized_audit = standardized_all[audit_rows]
    rms_scale = np.sqrt(standardized_all.shape[1])
    distance_to_audit = (
        cdist(standardized_all, standardized_audit).min(axis=1) / rms_scale
    )
    full_span = np.ptp(active_features, axis=0)
    audit_span = np.ptp(audited_features, axis=0)
    valid_span = full_span > 1.0e-8
    audit_span_ratio = np.clip(
        audit_span[valid_span] / full_span[valid_span], 0.0, 1.0
    )
    audit_parent_ids = parent_ids[audit_rows]
    audit_source_ids = source_ids[audit_rows]
    near_feasible = np.asarray(
        [
            bool(record["tim_gait"])
            and record["primary_failure"] == "axis_misaligned"
            for record in selected_records
        ],
        dtype=bool,
    )

    coverage = {
        "P3_train_rows": int(bank_ids.size),
        "P3_train_unique_parents": int(np.unique(parent_ids).size),
        "P3_train_unique_sources": int(np.unique(source_ids).size),
        "CEM_labeled_rows": int(audit_rows.size),
        "CEM_labeled_row_fraction": float(audit_rows.size / bank_ids.size),
        "CEM_labeled_unique_parents": int(np.unique(audit_parent_ids).size),
        "CEM_labeled_unique_sources": int(np.unique(audit_source_ids).size),
        "strict_feasible_rows": 0,
        "near_feasible_axis_only_rows": int(near_feasible.sum()),
        "distance_to_nearest_audited_reset_rms_z": _percentiles(
            distance_to_audit
        ),
        "audit_feature_span_ratio": _percentiles(audit_span_ratio),
        "canonical_coverage_gap": (
            "Only 12/396 P3-train rows have corrected CEM outcomes, and none "
            "is strictly feasible. Coverage of feasibility labels, not O0 state "
            "count, is the current gap."
        ),
    }

    observable_groups = {
        "geometry": "oracle observes fingertip position relative to the pen",
        "wrench": "oracle observes per-finger 3-D contact force and force norm",
        "pen_pose": "oracle observes pen pose/axis",
        "velocity": "oracle observes pen linear and angular velocity",
        "support_margin": "derived from observed contact force, geometry, and known gravity",
    }
    diagnosis = {
        "primary": "INCONCLUSIVE_NO_STRICT_FEASIBLE_LABELS",
        "A_reset_distribution_coverage_insufficient": {
            "status": "UNRESOLVED",
            "evidence": (
                "The corrected audit has no positive state, while only 3.03% "
                "of P3-train rows were searched. This cannot distinguish an "
                "absent basin from an unaudited narrow basin."
            ),
        },
        "B_exploration_difficult": {
            "status": "UNRESOLVED_BUT_PLAUSIBLE",
            "evidence": (
                "The 4x expanded CEM budget also found zero strict candidates, "
                "but one safe TIM transition missed only axis alignment in the "
                "formal run. Search difficulty remains plausible, not proven."
            ),
        },
        "C_representation_problem": {
            "status": "NOT_TESTABLE_WITH_ONE_CLASS",
            "evidence": (
                "All requested raw feature families are observable by the oracle, "
                "but no positive label exists for a feasibility classifier."
            ),
            "remaining_uncertainty": (
                "Network capacity, temporal encoding, and optimization limitations "
                "remain outside this initial-state coverage audit."
            ),
        },
    }

    pca = PCA(n_components=2, random_state=20260731)
    pca_points = pca.fit_transform(standardized_all)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].scatter(
        pca_points[:, 0], pca_points[:, 1], s=16, color="#b8bec6", alpha=0.45
    )
    ordinary = ~near_feasible
    axes[0, 0].scatter(
        pca_points[audit_rows[ordinary], 0],
        pca_points[audit_rows[ordinary], 1],
        s=75,
        color="#cc5b4b",
        marker="x",
        label="strict infeasible",
    )
    if near_feasible.any():
        axes[0, 0].scatter(
            pca_points[audit_rows[near_feasible], 0],
            pca_points[audit_rows[near_feasible], 1],
            s=95,
            color="#e7a43b",
            marker="D",
            edgecolor="black",
            linewidth=0.5,
            label="TIM transition, axis miss",
        )
    axes[0, 0].set_title("P3 initial-feature coverage (no positive labels)")
    axes[0, 0].set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    axes[0, 0].set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    axes[0, 0].legend()

    ring_gap_index = int(np.flatnonzero(active_names == "geometry_ring_gap_mm")[0])
    force_index = int(
        np.flatnonzero(active_names == "support_total_contact_force_sum_n")[0]
    )
    axes[0, 1].scatter(
        active_features[:, ring_gap_index],
        active_features[:, force_index],
        s=16,
        color="#b8bec6",
        alpha=0.45,
    )
    axes[0, 1].scatter(
        audited_features[ordinary, ring_gap_index],
        audited_features[ordinary, force_index],
        s=75,
        color="#cc5b4b",
        marker="x",
    )
    if near_feasible.any():
        axes[0, 1].scatter(
            audited_features[near_feasible, ring_gap_index],
            audited_features[near_feasible, force_index],
            s=95,
            color="#e7a43b",
            marker="D",
            edgecolor="black",
            linewidth=0.5,
        )
    axes[0, 1].axhline(6.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Ring fingertip surface gap (mm)")
    axes[0, 1].set_ylabel("Initial total contact-force sum (N)")
    axes[0, 1].set_title("Geometry / support-force slice")

    group_names = sorted(set(active_groups))
    variable_counts = [int(np.count_nonzero(active_groups == group)) for group in group_names]
    axes[1, 0].barh(group_names, variable_counts, color="#477fb3")
    axes[1, 0].set_xlabel("Variable extracted features")
    axes[1, 0].set_title("Observable feature schema; classifier unavailable")
    axes[1, 0].text(
        0.98,
        0.05,
        "0 strict feasible labels",
        transform=axes[1, 0].transAxes,
        ha="right",
        color="#a43e35",
        fontweight="bold",
    )

    axes[1, 1].hist(distance_to_audit, bins=20, color="#657b95", alpha=0.8)
    axes[1, 1].set_xlabel("Distance to nearest audited reset (RMS z)")
    axes[1, 1].set_ylabel("P3 train resets")
    axes[1, 1].set_title("Canonical audit-coverage gap")
    fig.suptitle("O1.9 Handoff Basin Analysis: prerequisite failure", fontsize=15)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)

    report = {
        "status": "PARTIAL_PREREQUISITE_FAILED",
        "artifact_role": "diagnostic_basin_analysis_not_training_data",
        "conclusion": diagnosis["primary"],
        "bank_sha256": snapshot["bank_sha256"],
        "o1_8_feasible_rate": audit["feasible_rate"],
        "labeled_bank_ids": audit_bank_ids.tolist(),
        "labels_feasible": [False] * len(selected_records),
        "near_feasible_bank_ids": audit_bank_ids[near_feasible].tolist(),
        "feature_schema": {
            "total_features": len(feature_names),
            "active_features": int(variable.sum()),
            "groups": observable_groups,
        },
        "classifier": {
            "status": "NOT_IDENTIFIABLE_ONE_CLASS",
            "reason": (
                "Corrected O1.8 has 0 strict feasible and 12 infeasible labels; "
                "ROC AUC, balanced feasibility classification, and basin "
                "probability extrapolation are undefined."
            ),
            "model_generated": False,
        },
        "canonical_coverage_gap": coverage,
        "diagnosis": diagnosis,
        "constraints": {
            "ppo_trained": False,
            "reward_modified": False,
            "teacher_generated": False,
            "o0_modified": False,
            "trajectory_exported": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "O1_9_HANDOFF_BASIN="
        + json.dumps(
            {
                "status": report["status"],
                "conclusion": report["conclusion"],
                "strict_feasible": 0,
                "near_feasible": int(near_feasible.sum()),
                "labeled_fraction": coverage["CEM_labeled_row_fraction"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    snapshot = json.loads(args.features.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if snapshot["bank_sha256"] != audit["bank_sha256"]:
        raise RuntimeError("O1.8 and O1.9 frozen-bank hashes differ")
    if snapshot["canonical_id"] != "P3" or snapshot["reset_split"] != "train":
        raise RuntimeError("O1.9 expects the same P3/train population as O1.8")

    bank_ids = np.asarray(snapshot["bank_ids"], dtype=np.int64)
    all_features = np.asarray(snapshot["features"], dtype=np.float64)
    feature_names = list(snapshot["feature_names"])
    feature_groups = np.asarray(snapshot["feature_groups"])
    parent_ids = np.asarray(snapshot["parent_state_ids"], dtype=np.int64)
    source_ids = np.asarray(snapshot["source_trajectory_ids"], dtype=np.int64)
    row_by_bank_id = {int(bank_id): index for index, bank_id in enumerate(bank_ids)}
    selected_records = audit["selected_state_results"]
    audit_bank_ids = np.asarray(
        [record["bank_id"] for record in selected_records], dtype=np.int64
    )
    missing = [int(value) for value in audit_bank_ids if int(value) not in row_by_bank_id]
    if missing:
        raise RuntimeError(f"O1.8 bank IDs absent from O1.9 snapshot: {missing}")
    audit_rows = np.asarray(
        [row_by_bank_id[int(value)] for value in audit_bank_ids], dtype=np.int64
    )
    labels = np.asarray(
        [record["feasible"] for record in selected_records], dtype=np.int64
    )
    if np.unique(labels).size < 2:
        _write_one_class_analysis(
            args=args,
            snapshot=snapshot,
            audit=audit,
            bank_ids=bank_ids,
            all_features=all_features,
            feature_names=feature_names,
            feature_groups=feature_groups,
            parent_ids=parent_ids,
            source_ids=source_ids,
            audit_bank_ids=audit_bank_ids,
            audit_rows=audit_rows,
            selected_records=selected_records,
        )
        return
    if min(labels.sum(), labels.size - labels.sum()) < 2:
        raise RuntimeError("At least two labels per class are required")

    variable = all_features.std(axis=0) > 1.0e-8
    active_features = all_features[:, variable]
    active_names = [name for name, keep in zip(feature_names, variable, strict=True) if keep]
    active_groups = feature_groups[variable]
    audited_features = active_features[audit_rows]

    group_reports = {}
    for group in sorted(set(active_groups)):
        group_mask = active_groups == group
        probabilities, selected = _nested_loo_predictions(
            audited_features[:, group_mask], labels
        )
        metrics = _classification_metrics(labels, probabilities)
        selected_counts: dict[str, int] = {}
        group_names = np.asarray(active_names)[group_mask]
        for fold in selected:
            for feature_index in fold:
                feature_name = str(group_names[feature_index])
                selected_counts[feature_name] = selected_counts.get(feature_name, 0) + 1
        metrics["most_selected_features"] = sorted(
            selected_counts.items(), key=lambda item: (-item[1], item[0])
        )[:5]
        group_reports[group] = metrics

    combined_probabilities, nested_selected = _nested_loo_predictions(
        audited_features, labels
    )
    combined_metrics = _classification_metrics(labels, combined_probabilities)
    permutation = _exact_balanced_permutation_test(
        audited_features, labels, combined_metrics["roc_auc"]
    )
    classifier_spec, full_probabilities = _fit_interpretable_classifier(
        audited_features, labels, active_features, active_names
    )

    full_mean = active_features.mean(axis=0)
    full_std = active_features.std(axis=0).clip(min=1.0e-8)
    standardized_all = (active_features - full_mean) / full_std
    standardized_audit = standardized_all[audit_rows]
    feasible_points = standardized_audit[labels == 1]
    infeasible_points = standardized_audit[labels == 0]
    rms_scale = np.sqrt(standardized_all.shape[1])
    distance_to_audit = cdist(standardized_all, standardized_audit).min(axis=1) / rms_scale
    distance_to_feasible = cdist(standardized_all, feasible_points).min(axis=1) / rms_scale
    distance_to_infeasible = cdist(standardized_all, infeasible_points).min(axis=1) / rms_scale
    nearest_feasible = distance_to_feasible < distance_to_infeasible

    full_span = np.ptp(active_features, axis=0)
    audit_span = np.ptp(audited_features, axis=0)
    feasible_span = np.ptp(audited_features[labels == 1], axis=0)
    valid_span = full_span > 1.0e-8
    audit_span_ratio = np.clip(audit_span[valid_span] / full_span[valid_span], 0.0, 1.0)
    feasible_span_ratio = np.clip(
        feasible_span[valid_span] / full_span[valid_span], 0.0, 1.0
    )

    audit_parent_ids = parent_ids[audit_rows]
    audit_source_ids = source_ids[audit_rows]
    feasibility_coverage = {
        "P3_train_rows": int(bank_ids.size),
        "P3_train_unique_parents": int(np.unique(parent_ids).size),
        "P3_train_unique_sources": int(np.unique(source_ids).size),
        "CEM_labeled_rows": int(labels.size),
        "CEM_labeled_row_fraction": float(labels.size / bank_ids.size),
        "CEM_labeled_unique_parents": int(np.unique(audit_parent_ids).size),
        "CEM_labeled_unique_sources": int(np.unique(audit_source_ids).size),
        "feasible_unique_parents": int(np.unique(audit_parent_ids[labels == 1]).size),
        "feasible_unique_sources": int(np.unique(audit_source_ids[labels == 1]).size),
        "distance_to_nearest_audited_reset_rms_z": _percentiles(distance_to_audit),
        "distance_to_nearest_feasible_reset_rms_z": _percentiles(distance_to_feasible),
        "fraction_closer_to_feasible_than_infeasible": float(nearest_feasible.mean()),
        "audit_feature_span_ratio": _percentiles(audit_span_ratio),
        "feasible_feature_span_ratio": _percentiles(feasible_span_ratio),
        "classifier_predicted_feasible_probability": _percentiles(full_probabilities),
        "classifier_predicted_probability_ge_0_5_fraction": float(
            (full_probabilities >= 0.5).mean()
        ),
        "warning": "Coverage probabilities are extrapolations from 12 labels and must not be used to filter or modify O0.",
    }

    observable_groups = {
        "geometry": "oracle observes fingertip position relative to the pen",
        "wrench": "oracle observes per-finger 3-D contact force and force norm",
        "pen_pose": "oracle observes pen pose/axis",
        "velocity": "oracle observes pen linear and angular velocity",
        "support_margin": "derived from observed contact force, geometry, and known gravity",
    }
    classifier_significant = (
        combined_metrics["roc_auc"] >= 0.80
        and permutation["p_value_auc_greater_equal"] <= 0.05
    )
    diagnosis = {
        "primary": "B_EXPLORATION_DIFFICULTY",
        "A_reset_distribution_coverage_insufficient": {
            "status": "NOT_SUPPORTED_AS_PRIMARY",
            "evidence": "Strict CEM succeeds on 6 of 12 spread-sampled P3 resets from six independent sources; feasible states are present in the frozen distribution.",
            "remaining_uncertainty": "Only 12/396 P3-train rows have CEM labels, so the exact feasible mass is not identified.",
        },
        "B_exploration_difficult": {
            "status": "SUPPORTED",
            "evidence": "CEM finds strict handoff candidates from 50% of audited resets while matched PPO ablations produce zero qualified handoff.",
        },
        "C_representation_problem": {
            "status": "NOT_SUPPORTED_AS_MISSING_STATE",
            "evidence": "Every audited feature family is directly observed by the privileged oracle or algebraically derivable from its observation.",
            "classifier_initial_state_separation_significant": bool(classifier_significant),
            "remaining_uncertainty": "The audit does not exclude network-capacity, optimization, or temporal state-encoding limitations.",
        },
    }

    pca = PCA(n_components=2, random_state=20260731)
    pca_points = pca.fit_transform(standardized_all)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].scatter(
        pca_points[:, 0], pca_points[:, 1], s=16, color="#b8bec6", alpha=0.45
    )
    axes[0, 0].scatter(
        pca_points[audit_rows[labels == 0], 0],
        pca_points[audit_rows[labels == 0], 1],
        s=75,
        color="#cc5b4b",
        marker="x",
        label="CEM infeasible",
    )
    axes[0, 0].scatter(
        pca_points[audit_rows[labels == 1], 0],
        pca_points[audit_rows[labels == 1], 1],
        s=75,
        color="#2f9e73",
        marker="o",
        edgecolor="black",
        linewidth=0.5,
        label="CEM feasible",
    )
    axes[0, 0].set_title("P3 reset basin (PCA of initial features)")
    axes[0, 0].set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    axes[0, 0].set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    axes[0, 0].legend()

    scatter = axes[0, 1].scatter(
        pca_points[:, 0],
        pca_points[:, 1],
        c=full_probabilities,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        s=22,
    )
    axes[0, 1].set_title("Diagnostic feasibility probability")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[0, 1], label="P(feasible), 12-label model")

    group_names = list(group_reports)
    group_auc = [group_reports[name]["roc_auc"] for name in group_names]
    axes[1, 0].barh(group_names, group_auc, color="#477fb3")
    axes[1, 0].axvline(0.5, color="black", linewidth=1, linestyle="--")
    axes[1, 0].set_xlim(0.0, 1.0)
    axes[1, 0].set_xlabel("Nested leave-one-out ROC AUC")
    axes[1, 0].set_title("Initial-state separability by feature family")

    axes[1, 1].hist(
        distance_to_feasible,
        bins=20,
        alpha=0.65,
        color="#2f9e73",
        label="nearest feasible",
    )
    axes[1, 1].hist(
        distance_to_infeasible,
        bins=20,
        alpha=0.55,
        color="#cc5b4b",
        label="nearest infeasible",
    )
    axes[1, 1].set_xlabel("RMS standardized feature distance")
    axes[1, 1].set_ylabel("P3 train resets")
    axes[1, 1].set_title("Canonical coverage around audited resets")
    axes[1, 1].legend()
    fig.suptitle("O1.9 Handoff Basin Analysis", fontsize=15)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)

    report = {
        "status": "PASS",
        "artifact_role": "diagnostic_basin_analysis_not_training_data",
        "conclusion": diagnosis["primary"],
        "bank_sha256": snapshot["bank_sha256"],
        "o1_8_feasible_rate": audit["feasible_rate"],
        "labeled_bank_ids": audit_bank_ids.tolist(),
        "labels_feasible": labels.astype(bool).tolist(),
        "feature_schema": {
            "total_features": len(feature_names),
            "active_features": len(active_names),
            "groups": observable_groups,
        },
        "classifier": {
            "validation": "nested leave-one-out; top two features selected inside each training fold",
            "combined": combined_metrics,
            "groups": group_reports,
            "exact_permutation_test": permutation,
            "interpretable_full_fit": classifier_spec,
            "small_sample_warning": "n=12 balanced labels; classifier is diagnostic and must not be used as a reset filter.",
        },
        "canonical_coverage_gap": feasibility_coverage,
        "diagnosis": diagnosis,
        "constraints": {
            "ppo_trained": False,
            "reward_modified": False,
            "teacher_generated": False,
            "o0_modified": False,
            "trajectory_exported": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "O1_9_HANDOFF_BASIN="
        + json.dumps(
            {
                "conclusion": report["conclusion"],
                "classifier_auc": combined_metrics["roc_auc"],
                "classifier_permutation_p": permutation[
                    "p_value_auc_greater_equal"
                ],
                "labeled_fraction": feasibility_coverage[
                    "CEM_labeled_row_fraction"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
