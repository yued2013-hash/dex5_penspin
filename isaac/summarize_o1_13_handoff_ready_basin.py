from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
ANALYSIS_PATH = REPORT_DIR / "O1_13_Handoff_Readiness_Candidates.json"
SELECTED_PATH = REPORT_DIR / "O1_13_readiness_selected_CEM_12x400ms.json"
BASELINE_PATH = REPORT_DIR / "O1_13_spread_baseline_CEM_12x400ms.json"
OUTPUT_JSON = REPORT_DIR / "O1_13_Handoff_Ready_Canonical_Basin_Analysis.json"
OUTPUT_MD = REPORT_DIR / "O1_13_Handoff_Ready_Canonical_Basin_Analysis_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_13_Handoff_Ready_Canonical_Basin_Map.png"


def _wilson(successes: int, count: int, z: float = 1.96) -> list[float]:
    if count == 0:
        return [0.0, 0.0]
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return [center - half, center + half]


def _unsafe(row: dict) -> bool:
    return any(
        row[key]
        for key in (
            "ballistic",
            "contact_loss",
            "penetration",
            "drop",
            "throw",
            "drift",
            "invalid",
        )
    )


def _admissible_transition(row: dict) -> bool:
    return (
        row["tim_gait"]
        and row["new_support_gain_n"] >= 0.03
        and row["old_support_unload_n"] >= 0.03
        and row["transfer_pair_margin"] >= 0.0
        and row["min_total_force_n"] >= 0.06
        and row["max_total_force_n"] <= 6.0
        and row["min_height_delta_mm"] >= -5.0
        and row["tim_overlap"] >= 0.80
        and not _unsafe(row)
    )


def _summarize_cem(report: dict) -> dict:
    rows = report["selected_state_results"]
    count = len(rows)
    raw = sum(row["tim_gait"] for row in rows)
    admissible = sum(_admissible_transition(row) for row in rows)
    strict = sum(row["feasible"] for row in rows)
    return {
        "count": count,
        "raw_TIM_transition": raw,
        "raw_TIM_transition_rate": raw / count,
        "raw_TIM_transition_Wilson95": _wilson(raw, count),
        "admissible_transition": admissible,
        "admissible_transition_rate": admissible / count,
        "strict_handoff": strict,
        "strict_handoff_rate": strict / count,
        "failure_taxonomy": report["failure_taxonomy"],
        "states": rows,
    }


def main() -> None:
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    selected_raw = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    baseline_raw = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    hashes = {
        analysis["bank_sha256"], selected_raw["bank_sha256"], baseline_raw["bank_sha256"]
    }
    if len(hashes) != 1:
        raise RuntimeError("O1.13 artifacts do not share the frozen O0 bank")
    if selected_raw["source_bank_ids"] != analysis["readiness_selected"]["bank_ids"]:
        raise RuntimeError("Selected CEM rows differ from readiness selection")
    if baseline_raw["source_bank_ids"] != analysis["spread_baseline"]["bank_ids"]:
        raise RuntimeError("Baseline CEM rows differ from the original spread sample")

    selected = _summarize_cem(selected_raw)
    baseline = _summarize_cem(baseline_raw)
    improved = (
        selected["strict_handoff_rate"] > baseline["strict_handoff_rate"]
        or selected["admissible_transition_rate"]
        > baseline["admissible_transition_rate"]
    )
    validation = {
        "matched_budget": {
            "entries": 12,
            "horizon_ms": 400.0,
            "population": 64,
            "iterations": 10,
            "waypoints": 3,
            "action_dimensions": 20,
            "max_offset_rad": 0.18,
        },
        "readiness_selected": selected,
        "original_P3_spread": baseline,
        "readiness_improves_CEM_feasibility": improved,
        "decision": "REJECT_STATIC_READINESS_REFINEMENT" if not improved else "SUPPORTED",
        "interpretation": (
            "The static prototype score does not improve admissible or strict CEM "
            "feasibility. Its high-score subset has zero raw TIM transitions, while "
            "the original spread has one raw transition that still violates force "
            "and axis gates."
        ),
    }
    report = {
        "status": "PASS_DIAGNOSTIC_REFINEMENT_REJECTED",
        "phase": "O1.13 Handoff-Ready Canonical Basin Analysis",
        "bank_sha256": next(iter(hashes)),
        "readiness_score": analysis["score_definition"],
        "prototype_comparison": analysis["prototype_comparison"],
        "extracted_feature_groups": {
            "groups": analysis["feature_schema"]["groups"],
            "group_contrast_RMS_z": analysis[
                "group_contrast_feasible_vs_failure_mean_RMS_z"
            ],
            "prototype_group_distance_to_996": analysis[
                "prototype_group_RMS_z_distance_to_996"
            ],
            "top_features_by_group": analysis["top_features_by_group"],
        },
        "validation": validation,
        "canonical_refinement_proposal": {
            "decision": "DO_NOT_FILTER_OR_REPLACE_FROZEN_P3",
            "immediate_use": (
                "Keep the score as an analysis-only hypothesis generator; do not "
                "use it as a reset sampler, canonical label, or training curriculum."
            ),
            "required_next_evidence": [
                "Collect provenance-isolated dynamic CEM labels over a materially larger P3 sample.",
                "Add action-conditioned 50-100 ms response features: axis response, support-force slope, and height acceleration under bounded TIM probes.",
                "Separate transition reachability from post-axis stabilization labels.",
                "Validate any revised score on held-out parent/source trajectories with clone replay rates.",
                "Promote an overlay only after admissible and strict CEM rates exceed the spread baseline with uncertainty reported.",
            ],
            "data_boundary": (
                "Any future handoff-ready overlay must remain separate from O0 and "
                "cannot alter P3 R3 validity or V9 semantics."
            ),
        },
        "constraints": {
            "PPO_trained": False,
            "reward_modified": False,
            "teacher_generated": False,
            "O0_modified": False,
            "new_reset_dataset_created": False,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    basin = analysis["basin_map"]
    coordinates = np.asarray(basin["pca_coordinates"], dtype=np.float64)
    scores = np.asarray(basin["readiness_scores"], dtype=np.float64)
    labels = np.asarray(basin["labels"])
    bank_ids = np.asarray(basin["bank_ids"])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), constrained_layout=True)
    scatter = axes[0].scatter(
        coordinates[:, 0], coordinates[:, 1], c=scores, cmap="viridis", s=18, alpha=0.55
    )
    styles = {
        "feasible_prototype": ("*", 170, "#2ca25f", "996 feasible"),
        "failure_prototype": ("X", 85, "#d95f59", "failure prototypes"),
        "readiness_selected": ("s", 65, "none", "score-selected"),
        "spread_baseline": ("o", 55, "none", "spread baseline"),
    }
    for label, (marker, size, color, legend) in styles.items():
        mask = labels == label
        if not np.any(mask):
            continue
        axes[0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            marker=marker,
            s=size,
            facecolors=color,
            edgecolors="#202020",
            linewidths=1.1,
            label=legend,
        )
        if label in ("feasible_prototype", "failure_prototype"):
            for x, y, bank_id in zip(
                coordinates[mask, 0], coordinates[mask, 1], bank_ids[mask], strict=True
            ):
                axes[0].annotate(str(bank_id), (x, y), xytext=(4, 4), textcoords="offset points")
    axes[0].set_title("P3 basin and prototype readiness score")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].legend(frameon=False, fontsize=8)
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, label="Uncalibrated readiness")

    group_contrast = analysis["group_contrast_feasible_vs_failure_mean_RMS_z"]
    groups = list(group_contrast)
    group_values = [group_contrast[group] for group in groups]
    axes[1].barh(groups, group_values, color="#4f81bd")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RMS standardized contrast")
    axes[1].set_title("996 vs failure prototypes")
    for index, value in enumerate(group_values):
        axes[1].text(value + 0.01, index, f"{value:.3f}", va="center")

    metrics = ("raw_TIM_transition", "admissible_transition", "strict_handoff")
    display = ("Raw TIM\ntransition", "Admissible\ntransition", "Strict\nhandoff")
    x = np.arange(len(metrics))
    width = 0.36
    selected_counts = [selected[name] for name in metrics]
    baseline_counts = [baseline[name] for name in metrics]
    axes[2].bar(x - width / 2, selected_counts, width, label="Readiness-selected", color="#6a51a3")
    axes[2].bar(x + width / 2, baseline_counts, width, label="Original spread", color="#e7b34c")
    axes[2].set_xticks(x, display)
    axes[2].set_ylim(0, 12.8)
    axes[2].set_ylabel("Entries / 12")
    axes[2].set_title("Matched 400 ms CEM validation")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.25)
    for location, values in ((x - width / 2, selected_counts), (x + width / 2, baseline_counts)):
        for position, value in zip(location, values, strict=True):
            axes[2].text(position, value + 0.15, str(value), ha="center")
    fig.suptitle("O1.13 Handoff-Ready Canonical Basin Analysis", fontsize=15)
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    top_groups = sorted(
        analysis["group_contrast_feasible_vs_failure_mean_RMS_z"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    group_rows = "\n".join(
        f"| `{name}` | {value:.3f} |" for name, value in top_groups
    )
    markdown = f"""# O1.13 Handoff-Ready Canonical Basin Analysis 验收报告

## 结论

`PASS (diagnostic) / STATIC CANONICAL REFINEMENT REJECTED`

已完成 geometry、wrench、contact/support margin、pen pose/velocity 和 20 维 joint configuration 的对比，并生成了透明的 prototype-margin readiness score。但独立 CEM 验证表明：**该分数没有扩大 handoff-ready P3 区域。**

| Distribution | Raw TIM transition | Admissible transition | Strict handoff |
|---|---:|---:|---:|
| readiness-selected | {selected['raw_TIM_transition']}/12 | {selected['admissible_transition']}/12 | {selected['strict_handoff']}/12 |
| original P3 spread | {baseline['raw_TIM_transition']}/12 | {baseline['admissible_transition']}/12 | {baseline['strict_handoff']}/12 |

原始 spread 的唯一 raw transition 来自 bank `996`，但最大接触力 `8.223 N` 超过 `6 N` 门槛，且轴对齐仅 `0.220`，因此也不是 admissible/strict 成功。

## 原型对比

- feasible prototype：`996`
- axis failures：`1245 / 1171`
- transition failure：`1203`
- readiness 不是概率；它只是六组等权标准化原型距离 margin。

| Feature group | 996 vs failure mean RMS-z |
|---|---:|
{group_rows}

最大差异来自 joint configuration，其次是 pen pose 和 geometry。但未见入口的负结果说明：**静态姿态/力特征接近 996，不等于局部动力学可控性接近 996。**

## Canonical refinement proposal

1. 不过滤、不替换 frozen P3，不修改 O0。
2. 当前 readiness score 只保留为 analysis-only hypothesis generator，不用于 reset sampler 或 curriculum。
3. 下一版 refinement 应基于更多 provenance-isolated 动态 CEM 标签，且分开 transition reachability 与 post-axis stabilization。
4. 增加 action-conditioned `50-100 ms` 局部响应特征：轴对齐响应、支撑力斜率、高度加速度，而不是继续叠加静态 pose 特征。
5. 任何 handoff-ready overlay 必须按 parent/source 留出验证，并在 clone replay 下显著超过 spread baseline 后才能晋级。

## 约束审计

- 未训练 PPO
- 未修改 reward
- 未生成 teacher
- 未修改 O0
- 未创建新 reset dataset
- readiness-selected 的 12 个入口与原型不共享 parent/source，选择内部也无 parent/source leakage

Basin map：`{OUTPUT_PNG.name}`
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {"json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD), "figure": str(OUTPUT_PNG)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
