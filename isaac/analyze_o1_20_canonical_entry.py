from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from canonical_reset_schema import JOINT_ORDER


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
SNAPSHOT_PATH = REPORT_DIR / "O1_9_P3_initial_feature_snapshot.json"
DYNAMIC_PATH = REPORT_DIR / "O1_14_Dynamic_Probe_Responses.npz"
O1_15_PATH = REPORT_DIR / "O1_15_Long_Horizon_Handoff_Basin_Audit.json"
BANK_PATH = (
    ROOT
    / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0/"
    "canonical_reset_bank/P1_P5_v9_physx_projected_R3_validated.npz"
)
OUTPUT_JSON = REPORT_DIR / "O1_20_Canonical_Entry_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_20_Canonical_Entry_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_20_P3_Skill_Entry_Basin_Map.png"

POSITIVE_BANK_IDS = (831, 894, 1155)
PERMUTATION_SAMPLES = 1000
RNG_SEED = 12020


def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = q75 - q25
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1.0e-8, scale, fallback)
    keep = scale > 1.0e-8
    return np.clip((values[:, keep] - median[keep]) / scale[keep], -8.0, 8.0), median, keep


def _dynamic_summary(path: Path) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    names: list[str] = []
    groups: list[str] = []
    columns: list[np.ndarray] = []
    with np.load(path, allow_pickle=False) as data:
        bank_ids = np.asarray(data["bank_ids"], dtype=np.int64)
        nonzero = np.any(np.abs(data["probe_offsets_rad"]) > 1.0e-9, axis=1)
        for horizon in ("h3", "h6"):
            definitions = (
                ("axis_best", "axis_response", np.max(data[f"{horizon}_mean_axis_alignment"][:, nonzero], axis=1)),
                ("axis_final_best", "axis_response", np.max(data[f"{horizon}_final_axis_alignment"][:, nonzero], axis=1)),
                ("wrong_axis_min", "axis_response", np.min(data[f"{horizon}_wrong_axis_speed_rad_s"][:, nonzero], axis=1)),
                ("new_support_gain_best", "support_response", np.max(data[f"{horizon}_new_support_gain_n"][:, nonzero], axis=1)),
                ("old_support_unload_best", "support_response", np.max(data[f"{horizon}_old_support_unload_n"][:, nonzero], axis=1)),
                ("transfer_margin_best", "support_response", np.max(data[f"{horizon}_transfer_pair_margin"][:, nonzero], axis=1)),
                ("min_force_best", "support_response", np.max(data[f"{horizon}_min_total_force_n"][:, nonzero], axis=1)),
                ("max_force_min", "support_response", np.min(data[f"{horizon}_max_total_force_n"][:, nonzero], axis=1)),
                ("height_floor_best_mm", "height_response", 1000.0 * np.max(data[f"{horizon}_min_height_delta_m"][:, nonzero], axis=1)),
                ("height_final_best_mm", "height_response", 1000.0 * np.max(data[f"{horizon}_final_height_delta_m"][:, nonzero], axis=1)),
                ("height_accel_best", "height_response", np.max(data[f"{horizon}_height_acceleration_m_s2"][:, nonzero], axis=1)),
                ("tim_response_count", "contact_response", np.sum(data[f"{horizon}_tim_response"][:, nonzero], axis=1)),
                ("short_admissible_count", "contact_response", np.sum(data[f"{horizon}_short_admissible"][:, nonzero], axis=1)),
                ("unsafe_count", "safety_response", np.sum(data[f"{horizon}_unsafe"][:, nonzero], axis=1)),
                ("release_finger_count", "contact_response", np.max(np.sum(data[f"{horizon}_release_seen"][:, nonzero], axis=2), axis=1)),
                ("recontact_finger_count", "contact_response", np.max(np.sum(data[f"{horizon}_recontact_seen"][:, nonzero], axis=2), axis=1)),
            )
            for suffix, group, values in definitions:
                names.append(f"{horizon}_{suffix}")
                groups.append(group)
                columns.append(np.asarray(values, dtype=np.float64))
    return np.stack(columns, axis=1), names, groups, bank_ids


def _group_distance(values: np.ndarray, center: np.ndarray, groups: np.ndarray) -> np.ndarray:
    distances = []
    for group in np.unique(groups):
        mask = groups == group
        distances.append(np.sqrt(np.mean((values[:, mask] - center[mask]) ** 2, axis=1)))
    return np.mean(np.stack(distances, axis=1), axis=1)


def _cross_validated_scores(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    parents: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    scores = np.empty(labels.size, dtype=np.float64)
    for test in range(labels.size):
        train = (parents != parents[test]) & (sources != sources[test])
        positive = train & labels
        negative = train & ~labels
        if positive.sum() == 0 or negative.sum() == 0:
            raise RuntimeError("Provenance holdout removed an entire class")
        positive_center = np.median(values[positive], axis=0)
        negative_center = np.median(values[negative], axis=0)
        row = values[test : test + 1]
        scores[test] = (
            _group_distance(row, negative_center, groups)[0]
            - _group_distance(row, positive_center, groups)[0]
        )
    return scores


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    return float(
        np.mean(
            (positive[:, None] > negative[None, :])
            + 0.5 * (positive[:, None] == negative[None, :])
        )
    )


def _permutation_p_value(
    values: np.ndarray,
    observed_labels: np.ndarray,
    groups: np.ndarray,
    parents: np.ndarray,
    sources: np.ndarray,
    observed_auc: float,
) -> tuple[float, int]:
    all_assignments = list(combinations(range(observed_labels.size), int(observed_labels.sum())))
    rng = np.random.default_rng(RNG_SEED)
    if len(all_assignments) > PERMUTATION_SAMPLES:
        selected = rng.choice(len(all_assignments), size=PERMUTATION_SAMPLES, replace=False)
        assignments = [all_assignments[index] for index in selected]
    else:
        assignments = all_assignments
    exceed = 0
    for assignment in assignments:
        labels = np.zeros(observed_labels.size, dtype=bool)
        labels[list(assignment)] = True
        score = _cross_validated_scores(values, labels, groups, parents, sources)
        exceed += _auc(score, labels) >= observed_auc - 1.0e-12
    return (exceed + 1.0) / (len(assignments) + 1.0), len(assignments)


def _cliffs_delta(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    return np.mean(
        (positive[:, None, :] > negative[None, :, :]).astype(np.int8)
        - (positive[:, None, :] < negative[None, :, :]).astype(np.int8),
        axis=(0, 1),
    )


def _evaluate_family(
    name: str,
    all_values: np.ndarray,
    feature_names: list[str],
    feature_groups: list[str],
    measured_rows: np.ndarray,
    labels: np.ndarray,
    parents: np.ndarray,
    sources: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    scaled, _, keep = _robust_scale(all_values)
    active_names = np.asarray(feature_names)[keep]
    active_groups = np.asarray(feature_groups)[keep]
    measured = scaled[measured_rows]
    scores = _cross_validated_scores(measured, labels, active_groups, parents, sources)
    auc = _auc(scores, labels)
    p_value, samples = _permutation_p_value(
        measured, labels, active_groups, parents, sources, auc
    )
    positive_percentiles = [
        float(np.mean(scores[~labels] < value)) for value in scores[labels]
    ]
    delta = _cliffs_delta(measured[labels], measured[~labels])
    top = np.argsort(np.abs(delta))[-15:][::-1]
    group_effect = {
        str(group): float(np.sqrt(np.mean(delta[active_groups == group] ** 2)))
        for group in np.unique(active_groups)
    }
    report = {
        "name": name,
        "features": int(keep.sum()),
        "groups": sorted(set(active_groups.tolist())),
        "provenance_holdout_auc": auc,
        "permutation_p_value": p_value,
        "permutation_samples": samples,
        "positive_score_percentiles_vs_failures": positive_percentiles,
        "all_positive_in_top_half": bool(min(positive_percentiles) >= 0.5),
        "group_effect_RMS_cliffs_delta": group_effect,
        "top_feature_contrasts": [
            {
                "feature": str(active_names[index]),
                "group": str(active_groups[index]),
                "cliffs_delta": float(delta[index]),
                "positive_median": float(np.median(all_values[measured_rows[labels]][:, np.flatnonzero(keep)[index]])),
                "failure_median": float(np.median(all_values[measured_rows[~labels]][:, np.flatnonzero(keep)[index]])),
            }
            for index in top
        ],
    }
    return report, scores, scaled, active_groups


def main() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    o1_15 = json.loads(O1_15_PATH.read_text(encoding="utf-8"))
    if snapshot["bank_sha256"] != o1_15["bank_sha256"]:
        raise RuntimeError("O1.9 and O1.15 do not use the same frozen O0 bank")

    bank_ids = np.asarray(snapshot["bank_ids"], dtype=np.int64)
    row_by_id = {int(bank_id): row for row, bank_id in enumerate(bank_ids)}
    static_values = np.asarray(snapshot["features"], dtype=np.float64)
    static_names = list(snapshot["feature_names"])
    static_groups = list(snapshot["feature_groups"])
    with np.load(BANK_PATH, allow_pickle=False) as bank:
        qpos = np.asarray(bank["robot_qpos"][bank_ids], dtype=np.float64)
        qvel = np.asarray(bank["robot_qvel"][bank_ids], dtype=np.float64)
        qtarget_error = np.asarray(
            bank["robot_q_target"][bank_ids] - bank["robot_qpos"][bank_ids],
            dtype=np.float64,
        )
    static_values = np.concatenate((static_values, qpos, qvel, qtarget_error), axis=1)
    static_names.extend(f"joint_q_{joint}_rad" for joint in JOINT_ORDER)
    static_names.extend(f"joint_dq_{joint}_rad_s" for joint in JOINT_ORDER)
    static_names.extend(f"control_target_error_{joint}_rad" for joint in JOINT_ORDER)
    static_groups.extend("joint_configuration" for _ in JOINT_ORDER)
    static_groups.extend("joint_velocity" for _ in JOINT_ORDER)
    static_groups.extend("control_context" for _ in JOINT_ORDER)

    dynamic_values, dynamic_names, dynamic_groups, dynamic_ids = _dynamic_summary(DYNAMIC_PATH)
    if not np.array_equal(bank_ids, dynamic_ids):
        raise RuntimeError("O1.9 and O1.14 P3 row order differs")

    ranked = o1_15["state_ranking"]
    measured_ids = np.asarray([row["bank_id"] for row in ranked], dtype=np.int64)
    measured_rows = np.asarray([row_by_id[int(value)] for value in measured_ids], dtype=np.int64)
    labels = np.asarray([row["bank_id"] in POSITIVE_BANK_IDS for row in ranked], dtype=bool)
    parents = np.asarray([row["parent_state_id"] for row in ranked], dtype=np.int64)
    sources = np.asarray([row["source_trajectory_id"] for row in ranked], dtype=np.int64)
    if set(measured_ids[labels]) != set(POSITIVE_BANK_IDS):
        raise RuntimeError("O1.15 strict positive labels changed")
    if (
        len(set(parents[labels].tolist())) != int(labels.sum())
        or len(set(sources[labels].tolist())) != int(labels.sum())
    ):
        raise RuntimeError("O1.15 strict positives are not provenance independent")

    combined_values = np.concatenate((static_values, dynamic_values), axis=1)
    combined_names = static_names + dynamic_names
    combined_groups = static_groups + dynamic_groups
    families = {}
    score_by_family = {}
    scaled_by_family = {}
    groups_by_family = {}
    for family in (
        ("static_state", static_values, static_names, static_groups),
        ("short_action_response", dynamic_values, dynamic_names, dynamic_groups),
        ("combined", combined_values, combined_names, combined_groups),
    ):
        result, scores, scaled, active_groups = _evaluate_family(
            *family, measured_rows, labels, parents, sources
        )
        families[family[0]] = result
        score_by_family[family[0]] = scores
        scaled_by_family[family[0]] = scaled
        groups_by_family[family[0]] = active_groups

    combined = families["combined"]
    promotion = bool(
        combined["provenance_holdout_auc"] >= 0.80
        and combined["permutation_p_value"] < 0.05
        and combined["all_positive_in_top_half"]
    )

    scaled = scaled_by_family["combined"]
    groups = groups_by_family["combined"]
    positive_center = np.median(scaled[measured_rows[labels]], axis=0)
    negative_center = np.median(scaled[measured_rows[~labels]], axis=0)
    readiness = _group_distance(scaled, negative_center, groups) - _group_distance(
        scaled, positive_center, groups
    )
    measured_set = set(measured_ids.tolist())
    candidate_order = np.argsort(readiness)[::-1]
    candidate_rows = []
    used_parents: set[int] = set(parents.tolist())
    used_sources: set[int] = set(sources.tolist())
    all_parents = np.asarray(snapshot["parent_state_ids"], dtype=np.int64)
    all_sources = np.asarray(snapshot["source_trajectory_ids"], dtype=np.int64)
    for row in candidate_order:
        bank_id = int(bank_ids[row])
        parent = int(all_parents[row])
        source = int(all_sources[row])
        if bank_id in measured_set or parent in used_parents or source in used_sources:
            continue
        candidate_rows.append(
            {"bank_id": bank_id, "score": float(readiness[row]), "parent_state_id": parent, "source_trajectory_id": source}
        )
        used_parents.add(parent)
        used_sources.add(source)
        if len(candidate_rows) == 20:
            break

    failure_counts = Counter(row["O1_15_failure"] for row in ranked if not row["feasible"])
    report = {
        "status": "PASS_DIAGNOSTIC_P3H_NOT_ESTABLISHED" if not promotion else "PASS_DIAGNOSTIC_P3H_CANDIDATE_SUPPORTED",
        "phase": "O1.20 Canonical Entry Audit",
        "artifact_role": "read_only_skill_entry_semantic_audit_not_classifier_reset_filter_or_teacher",
        "bank_sha256": snapshot["bank_sha256"],
        "evidence_reconciliation": {
            "O1_8_corrected_strict_feasible": 0,
            "O1_15_conditional_population": int(labels.size),
            "O1_15_strict_positive_count": int(labels.sum()),
            "strict_positive_bank_ids": list(POSITIVE_BANK_IDS),
            "positive_parent_count": len(set(parents[labels].tolist())),
            "positive_source_count": len(set(sources[labels].tolist())),
            "conditional_population_unique_parent_count": len(set(parents.tolist())),
            "conditional_population_unique_source_count": len(set(sources.tolist())),
            "validation_split": "leave the test row's complete parent and source groups out of both class prototypes",
            "failure_taxonomy": dict(failure_counts),
            "scope": "conditional on O1.14 short-admissible 37-state layer, not all P3",
        },
        "feature_families": families,
        "skill_entry_classifier": {
            "status": "SUPPORTED_FOR_FUTURE_CONFIRMATION" if promotion else "REJECTED_INSUFFICIENT_GENERALIZATION",
            "promotion_gate": {
                "combined_auc_at_least_0P80": combined["provenance_holdout_auc"] >= 0.80,
                "combined_permutation_p_below_0P05": combined["permutation_p_value"] < 0.05,
                "every_positive_above_failure_median": combined["all_positive_in_top_half"],
                "all_pass": promotion,
            },
            "small_sample_warning": "Only three independent strict positives exist; no deployable classifier or P3-H distribution is claimed.",
        },
        "quarantined_candidate_ranking": {
            "status": "analysis_only_do_not_use_as_reset_filter",
            "definition": "full-P3 combined prototype margin trained on all 3 positives and 34 failures",
            "rows": candidate_rows,
            "parent_source_overlap_with_O1_15": False,
        },
        "canonical_refinement_proposal": {
            "decision": "DO_NOT_MODIFY_P3_OR_CREATE_P3H" if not promotion else "RUN_INDEPENDENT_CONFIRMATORY_AUDIT_BEFORE_P3H",
            "reason": (
                "Static recoverability remains valid, but three sparse trajectory-level positives are insufficient to define a robust skill-entry manifold."
            ),
            "next_gate": (
                "Evaluate the quarantined, provenance-disjoint ranking with the frozen O1.15 search contract; require independent strict positives and clone robustness before defining P3-H."
            ),
        },
        "constraints": {
            "PPO_trained": False,
            "reward_modified": False,
            "O0_modified": False,
            "teacher_generated": False,
            "reset_bank_modified": False,
            "CEM_action_serialized": False,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    colors = {"feasible": "#2a9d8f", "contact_force_overload": "#e76f51", "axis_misaligned_after_transition": "#e9c46a", "no_TIM_transition": "#6c757d"}
    outcomes = ["feasible" if row["feasible"] else row["O1_15_failure"] for row in ranked]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5), constrained_layout=True)
    for axis, family in zip(axes[:2], ("static_state", "short_action_response"), strict=True):
        scores = score_by_family[family]
        for outcome in dict.fromkeys(outcomes):
            mask = np.asarray([value == outcome for value in outcomes])
            axis.scatter(np.flatnonzero(mask), scores[mask], s=48, color=colors[outcome], label=outcome, alpha=0.9)
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_title(f"{family}\nCV AUC={families[family]['provenance_holdout_auc']:.3f}, p={families[family]['permutation_p_value']:.3f}")
        axis.set_xlabel("O1.15 ranked state index")
        axis.set_ylabel("held-out readiness score")
    effects = families["combined"]["group_effect_RMS_cliffs_delta"]
    ordered = sorted(effects, key=effects.get)
    axes[2].barh(ordered, [effects[value] for value in ordered], color="#457b9d")
    axes[2].set_xlabel("RMS |Cliff's delta|")
    axes[2].set_title("Feasible vs failed group contrast")
    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("O1.20 P3 skill-entry audit: sparse strict positives, provenance-held-out")
    fig.savefig(OUTPUT_PNG, dpi=170)
    plt.close(fig)

    md = f"""# O1.20 Canonical Entry Audit 验收报告

## 结论

`{report['status']}`

O0 的 P3 状态仍然是合法、可重建的 canonical reset；但当前证据不足以把其改成 P3-H skill-entry distribution。O1.15 的 37 个条件候选中只有 3 个独立严格正样本，且均不是鲁棒 basin。

## 证据口径

- O1.8 本地修正记录：`0/12` strict feasible。
- O1.15 后续严格存在性：`3/37`，bank `{', '.join(map(str, POSITIVE_BANK_IDS))}`。
- 正样本来自 3 个独立 parent 和 3 个独立 source trajectory。
- 全部结论只适用于 O1.14 筛出的 37 个短时响应候选，不代表完整 P3 分布。

## 区分能力

| 特征族 | provenance holdout AUC | permutation p | 三个正样本均高于失败中位数 |
|---|---:|---:|---:|
| 静态状态 | {families['static_state']['provenance_holdout_auc']:.3f} | {families['static_state']['permutation_p_value']:.3f} | {families['static_state']['all_positive_in_top_half']} |
| 短动作响应 | {families['short_action_response']['provenance_holdout_auc']:.3f} | {families['short_action_response']['permutation_p_value']:.3f} | {families['short_action_response']['all_positive_in_top_half']} |
| 合并 | {families['combined']['provenance_holdout_auc']:.3f} | {families['combined']['permutation_p_value']:.3f} | {families['combined']['all_positive_in_top_half']} |

分类器门槛：`{report['skill_entry_classifier']['status']}`。候选排名仅用于下一次独立确证审计，不能用作 reset filter、teacher 或 PPO curriculum。

## Canonical 决策

`{report['canonical_refinement_proposal']['decision']}`

当前可支持的结论是：稳定 P3 不等于 handoff-ready P3，但尚未找到能在独立来源上泛化的 skill-entry 边界。下一步应使用冻结的 O1.15 搜索口径审计 quarantine ranking，而不是现在修改 O0/P3。

![O1.20 basin map]({OUTPUT_PNG.name})
"""
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT_JSON), "classifier": report["skill_entry_classifier"], "families": {key: {"auc": value["provenance_holdout_auc"], "p": value["permutation_p_value"]} for key, value in families.items()}}, indent=2))


if __name__ == "__main__":
    main()
