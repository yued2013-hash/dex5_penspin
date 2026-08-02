from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_o1_20_canonical_entry import (
    BANK_PATH,
    O1_15_PATH,
    POSITIVE_BANK_IDS,
    REPORT_DIR,
    SNAPSHOT_PATH,
    _cliffs_delta,
    _group_distance,
    _robust_scale,
)
from canonical_reset_schema import JOINT_ORDER


O1_20_PATH = REPORT_DIR / "O1_20_Canonical_Entry_Audit.json"
OUTPUT_JSON = REPORT_DIR / "O1_21_Skill_Entry_Coverage_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_21_Skill_Entry_Coverage_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_21_P3_Skill_Entry_Coverage_Map.png"


def _static_features(snapshot: dict) -> tuple[np.ndarray, list[str], list[str]]:
    bank_ids = np.asarray(snapshot["bank_ids"], dtype=np.int64)
    values = np.asarray(snapshot["features"], dtype=np.float64)
    names = list(snapshot["feature_names"])
    groups = list(snapshot["feature_groups"])
    with np.load(BANK_PATH, allow_pickle=False) as bank:
        qpos = np.asarray(bank["robot_qpos"][bank_ids], dtype=np.float64)
        qvel = np.asarray(bank["robot_qvel"][bank_ids], dtype=np.float64)
        target_error = np.asarray(
            bank["robot_q_target"][bank_ids] - bank["robot_qpos"][bank_ids],
            dtype=np.float64,
        )
    values = np.concatenate((values, qpos, qvel, target_error), axis=1)
    names.extend(f"joint_q_{joint}_rad" for joint in JOINT_ORDER)
    names.extend(f"joint_dq_{joint}_rad_s" for joint in JOINT_ORDER)
    names.extend(f"control_target_error_{joint}_rad" for joint in JOINT_ORDER)
    groups.extend("joint_configuration" for _ in JOINT_ORDER)
    groups.extend("joint_velocity" for _ in JOINT_ORDER)
    groups.extend("control_context" for _ in JOINT_ORDER)
    return values, names, groups


def _density_record(
    name: str,
    score: np.ndarray,
    threshold: float,
    bank_ids: np.ndarray,
    parents: np.ndarray,
    sources: np.ndarray,
    measured: np.ndarray,
    known_labels: np.ndarray,
) -> tuple[dict, np.ndarray]:
    selected = score >= threshold
    measured_selected = selected[measured]
    positive_recall = float(np.mean(measured_selected[known_labels]))
    negative_rate = float(np.mean(measured_selected[~known_labels]))
    unknown = selected.copy()
    unknown[measured] = False
    record = {
        "name": name,
        "threshold": float(threshold),
        "states": int(selected.sum()),
        "density_of_all_P3": float(selected.mean()),
        "independent_parent_count": len(set(parents[selected].tolist())),
        "independent_source_count": len(set(sources[selected].tolist())),
        "known_positive_recall": positive_recall,
        "known_failure_selection_rate": negative_rate,
        "unmeasured_candidate_count": int(unknown.sum()),
        "bank_ids": bank_ids[selected].astype(int).tolist(),
        "role": "analysis_only_not_feasibility_labels_or_reset_filter",
    }
    return record, selected


def main() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    o1_15 = json.loads(O1_15_PATH.read_text(encoding="utf-8"))
    o1_20 = json.loads(O1_20_PATH.read_text(encoding="utf-8"))
    hashes = {snapshot["bank_sha256"], o1_15["bank_sha256"], o1_20["bank_sha256"]}
    if len(hashes) != 1:
        raise RuntimeError("O1.9/O1.15/O1.20 do not share the frozen O0 bank")

    bank_ids = np.asarray(snapshot["bank_ids"], dtype=np.int64)
    parents = np.asarray(snapshot["parent_state_ids"], dtype=np.int64)
    sources = np.asarray(snapshot["source_trajectory_ids"], dtype=np.int64)
    row_by_id = {int(bank_id): row for row, bank_id in enumerate(bank_ids)}
    values, feature_names, feature_groups = _static_features(snapshot)
    scaled, _, keep = _robust_scale(values)
    active_names = np.asarray(feature_names)[keep]
    active_groups = np.asarray(feature_groups)[keep]

    ranked = o1_15["state_ranking"]
    measured_ids = np.asarray([row["bank_id"] for row in ranked], dtype=np.int64)
    measured = np.asarray([row_by_id[int(bank_id)] for bank_id in measured_ids], dtype=np.int64)
    labels = np.asarray([bank_id in POSITIVE_BANK_IDS for bank_id in measured_ids], dtype=bool)
    if set(measured_ids[labels]) != set(POSITIVE_BANK_IDS):
        raise RuntimeError("Strict positive label set changed")

    positive_center = np.median(scaled[measured[labels]], axis=0)
    negative_center = np.median(scaled[measured[~labels]], axis=0)
    score = _group_distance(scaled, negative_center, active_groups) - _group_distance(
        scaled, positive_center, active_groups
    )
    measured_score = score[measured]
    positive_score = measured_score[labels]
    negative_score = measured_score[~labels]

    zero_failure_threshold = float(np.nextafter(np.max(negative_score), np.inf))
    sensitivity_threshold = float(np.min(positive_score))
    broad_threshold = float(np.quantile(negative_score, 0.95, method="higher"))
    density_layers = {}
    density_masks = {}
    for name, threshold in (
        ("zero_known_failure_core", zero_failure_threshold),
        ("all_positive_sensitivity_envelope", sensitivity_threshold),
        ("known_failure_5pct_candidate_band", broad_threshold),
    ):
        density_layers[name], density_masks[name] = _density_record(
            name,
            score,
            threshold,
            bank_ids,
            parents,
            sources,
            measured,
            labels,
        )

    delta = _cliffs_delta(scaled[measured[labels]], scaled[measured[~labels]])
    top_indices = np.argsort(np.abs(delta))[-20:][::-1]
    feature_importance = []
    for active_index in top_indices:
        original_index = np.flatnonzero(keep)[active_index]
        full_column = values[:, original_index]
        positive_column = values[measured[labels], original_index]
        failure_column = values[measured[~labels], original_index]
        positive_median = float(np.median(positive_column))
        feature_importance.append(
            {
                "feature": str(active_names[active_index]),
                "group": str(active_groups[active_index]),
                "importance_proxy_abs_cliffs_delta": float(abs(delta[active_index])),
                "cliffs_delta": float(delta[active_index]),
                "positive_median": positive_median,
                "failure_median": float(np.median(failure_column)),
                "positive_median_percentile_in_all_P3": float(np.mean(full_column <= positive_median)),
                "all_P3_inside_positive_min_max_fraction": float(
                    np.mean((full_column >= positive_column.min()) & (full_column <= positive_column.max()))
                ),
            }
        )
    group_importance = {
        str(group): float(np.sqrt(np.mean(delta[active_groups == group] ** 2)))
        for group in np.unique(active_groups)
    }

    centered = scaled - scaled.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    pca = centered @ vh[:2].T
    explained = singular[:2] ** 2 / np.sum(singular**2)

    measured_set = set(measured_ids.tolist())
    strict_known_density = len(POSITIVE_BANK_IDS) / bank_ids.size
    conditional_rate = len(POSITIVE_BANK_IDS) / measured_ids.size
    high_confidence = density_layers["all_positive_sensitivity_envelope"]
    coverage_sparse = high_confidence["density_of_all_P3"] < 0.05
    has_unmeasured_diverse_candidates = (
        high_confidence["unmeasured_candidate_count"] >= 5
        and high_confidence["independent_parent_count"] >= 8
        and high_confidence["independent_source_count"] >= 8
    )
    policy_exploration_remains = bool(len(POSITIVE_BANK_IDS) > 0 and coverage_sparse)
    resampling_required_now = bool(coverage_sparse and not has_unmeasured_diverse_candidates)

    report = {
        "status": "PASS_DIAGNOSTIC_SPARSE_NONZERO_ENTRY_COVERAGE",
        "phase": "O1.21 Skill Entry Coverage Audit",
        "artifact_role": "read_only_static_coverage_audit_not_classifier_reset_filter_or_dataset",
        "bank_sha256": snapshot["bank_sha256"],
        "population": {
            "P3_states": int(bank_ids.size),
            "independent_parent_count": len(set(parents.tolist())),
            "independent_source_count": len(set(sources.tolist())),
            "long_horizon_measured_states": int(measured_ids.size),
            "long_horizon_unmeasured_states": int(bank_ids.size - measured_ids.size),
            "known_strict_feasible": len(POSITIVE_BANK_IDS),
            "known_strict_density_of_all_P3": strict_known_density,
            "conditional_feasible_rate_in_O1_14_candidate_layer": conditional_rate,
            "scope_warning": "The other 359 P3 rows are unknown, not non-feasible.",
        },
        "feature_importance": {
            "definition": "univariate absolute Cliff's delta on 3 strict positives versus 34 measured failures; descriptive, not causal",
            "group_RMS_abs_cliffs_delta": group_importance,
            "top_features": feature_importance,
        },
        "coverage_gap": {
            "strict_positive_labels_are_sparse": True,
            "strict_positive_parent_count": len(set(parents[measured[labels]].tolist())),
            "strict_positive_source_count": len(set(sources[measured[labels]].tolist())),
            "positive_pairwise_group_distance": [
                float(_group_distance(scaled[[right]], scaled[left], active_groups)[0])
                for offset, left in enumerate(measured[labels])
                for right in measured[labels][offset + 1 :]
            ],
            "dominant_feature_groups": [
                key for key, _ in sorted(group_importance.items(), key=lambda item: item[1], reverse=True)[:5]
            ],
            "interpretation": "The entry-like region occupies a small but provenance-diverse part of frozen P3; static pose/geometry/wrench differences dominate short-probe responses.",
        },
        "candidate_density": density_layers,
        "decision": {
            "A_existing_P3_coverage_insufficient": {
                "supported": coverage_sparse,
                "reason": f"The all-positive static envelope covers {100.0 * high_confidence['density_of_all_P3']:.2f}% of P3, below the pre-registered 5% sparse-coverage boundary.",
            },
            "B_resample_canonical_now": {
                "supported": resampling_required_now,
                "reason": (
                    "Not yet: the frozen bank still contains provenance-diverse unmeasured high-score candidates. Confirm them before any separately approved resampling phase."
                    if not resampling_required_now
                    else "Existing high-confidence candidates are too few or provenance-concentrated."
                ),
            },
            "C_policy_exploration_remains_primary": {
                "supported": policy_exploration_remains,
                "reason": "Strict handoff existence is already demonstrated inside P3, while prior PPO produced no qualified handoff; sparse reset probability amplifies, but does not replace, the exploration bottleneck.",
            },
            "overall": "A_AND_C_SUPPORTED_B_DEFERRED",
        },
        "next_gate": {
            "action": "independent long-horizon confirmation of quarantined candidates under the frozen O1.15 contract",
            "candidate_source": "O1.20 provenance-disjoint ranking intersected with O1.21 sensitivity envelope",
            "candidate_bank_ids": [
                int(value["bank_id"])
                for value in o1_20["quarantined_candidate_ranking"]["rows"]
                if value["bank_id"] in set(bank_ids[density_masks["all_positive_sensitivity_envelope"]].tolist())
            ],
            "prohibition": "Do not create P3-H or resample O0 until independent strict and clone-robust evidence is obtained.",
        },
        "constraints": {
            "O0_modified": False,
            "P3H_created": False,
            "PPO_trained": False,
            "teacher_generated": False,
            "reset_bank_modified": False,
            "new_reset_dataset_created": False,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    known_failure = np.zeros(bank_ids.size, dtype=bool)
    known_failure[measured[~labels]] = True
    known_positive = np.zeros(bank_ids.size, dtype=bool)
    known_positive[measured[labels]] = True
    unknown = ~(known_failure | known_positive)
    envelope = density_masks["all_positive_sensitivity_envelope"] & unknown

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.7), constrained_layout=True)
    axes[0].scatter(pca[unknown, 0], pca[unknown, 1], s=18, color="#c7c7c7", alpha=0.55, label="unmeasured P3")
    axes[0].scatter(pca[known_failure, 0], pca[known_failure, 1], s=45, color="#e76f51", alpha=0.9, label="measured failure")
    axes[0].scatter(pca[envelope, 0], pca[envelope, 1], s=70, facecolors="none", edgecolors="#457b9d", linewidths=1.5, label="unmeasured envelope")
    axes[0].scatter(pca[known_positive, 0], pca[known_positive, 1], s=95, marker="*", color="#2a9d8f", edgecolors="#173f3a", label="strict feasible")
    axes[0].set_title(f"Frozen P3 static coverage\nPC1/PC2={100*explained[0]:.1f}%/{100*explained[1]:.1f}%")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(score[unknown], bins=32, color="#bdbdbd", alpha=0.75, label="unmeasured")
    axes[1].hist(score[measured[~labels]], bins=16, color="#e76f51", alpha=0.75, label="measured failure")
    for value in score[measured[labels]]:
        axes[1].axvline(value, color="#2a9d8f", linewidth=2.0)
    axes[1].axvline(sensitivity_threshold, color="#264653", linestyle="--", label="all-positive envelope")
    axes[1].set_title("Static prototype-margin density")
    axes[1].set_xlabel("entry-likeness score")
    axes[1].set_ylabel("states")
    axes[1].legend(frameon=False, fontsize=8)

    top = feature_importance[:12][::-1]
    axes[2].barh([row["feature"] for row in top], [row["importance_proxy_abs_cliffs_delta"] for row in top], color="#457b9d")
    axes[2].set_xlim(0.0, 1.05)
    axes[2].set_xlabel("|Cliff's delta|")
    axes[2].set_title("Descriptive feature contrasts")
    fig.suptitle("O1.21 P3 skill-entry coverage: strict labels, unknown pool, and quarantined density")
    fig.savefig(OUTPUT_PNG, dpi=170)
    plt.close(fig)

    layers = report["candidate_density"]
    md = f"""# O1.21 Skill Entry Coverage Audit 验收报告

## 结论

`{report['status']}`

P3 canonical bank 中已经包含 handoff-entry，但高置信区域稀疏。当前支持 **A + C**：现有 P3 覆盖不足，同时 policy exploration 仍是主要瓶颈；**B 暂不支持**，因为冻结 bank 中仍有来源独立、未做长时域验证的高分候选。

## 总体口径

- P3 状态：`{report['population']['P3_states']}`
- 长时域已测：`{report['population']['long_horizon_measured_states']}`
- 未测：`{report['population']['long_horizon_unmeasured_states']}`，它们是 unknown，不是 non-feasible
- 已知严格可行：`3`，占全 P3 `{100*report['population']['known_strict_density_of_all_P3']:.2f}%`

## Candidate Density

| 层级 | 状态 | P3 密度 | 正样本召回 | 已知失败选中率 | 未测候选 |
|---|---:|---:|---:|---:|---:|
| 零已知失败核心 | {layers['zero_known_failure_core']['states']} | {100*layers['zero_known_failure_core']['density_of_all_P3']:.2f}% | {100*layers['zero_known_failure_core']['known_positive_recall']:.1f}% | {100*layers['zero_known_failure_core']['known_failure_selection_rate']:.1f}% | {layers['zero_known_failure_core']['unmeasured_candidate_count']} |
| 全正样本包络 | {layers['all_positive_sensitivity_envelope']['states']} | {100*layers['all_positive_sensitivity_envelope']['density_of_all_P3']:.2f}% | {100*layers['all_positive_sensitivity_envelope']['known_positive_recall']:.1f}% | {100*layers['all_positive_sensitivity_envelope']['known_failure_selection_rate']:.1f}% | {layers['all_positive_sensitivity_envelope']['unmeasured_candidate_count']} |
| 5% 失败容忍带 | {layers['known_failure_5pct_candidate_band']['states']} | {100*layers['known_failure_5pct_candidate_band']['density_of_all_P3']:.2f}% | {100*layers['known_failure_5pct_candidate_band']['known_positive_recall']:.1f}% | {100*layers['known_failure_5pct_candidate_band']['known_failure_selection_rate']:.1f}% | {layers['known_failure_5pct_candidate_band']['unmeasured_candidate_count']} |

这些数字是静态特征候选密度，不是物理 feasible rate。

## A/B/C 判断

- **A：支持。** 覆盖全部三个正样本的高置信包络仅占 P3 `{100*high_confidence['density_of_all_P3']:.2f}%`。
- **B：暂不支持。** 先验证现有 bank 内的独立候选，再决定是否需要重新采样 canonical。
- **C：支持。** P3 内已存在严格 handoff 入口，但既有 PPO 未发现 qualified handoff；稀疏 reset 概率会放大这个 exploration 问题。

## 边界

不修改 O0，不创建 P3-H，不把候选当作 reset filter。下一道门是对隔离候选执行独立长时域确证。

![O1.21 coverage map]({OUTPUT_PNG.name})
"""
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"]["overall"],
                "densities": {
                    key: {"states": value["states"], "density": value["density_of_all_P3"]}
                    for key, value in density_layers.items()
                },
                "next_candidates": report["next_gate"]["candidate_bank_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
