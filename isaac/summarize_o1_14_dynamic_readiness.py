from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from o1_14_probe_design import least_squares_gain_norms, principal_coordinates


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
RAW_NPZ = REPORT_DIR / "O1_14_Dynamic_Probe_Responses.npz"
RAW_JSON = RAW_NPZ.with_suffix(".json")
OUTPUT_JSON = REPORT_DIR / "O1_14_Dynamic_Handoff_Readiness_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_14_Dynamic_Handoff_Readiness_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_14_Dynamic_Readiness_Map.png"

ANCHOR_OUTCOMES = {
    996: "O1.12 strict post success",
    1245: "axis failure",
    1171: "axis failure",
    1203: "transition replay failure",
}


def _quantile(values: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(values, q, axis=1)


def _build_state_features(data: dict[str, np.ndarray], horizons: list[int]) -> tuple[list[str], np.ndarray, dict]:
    names: list[str] = []
    columns: list[np.ndarray] = []
    summaries: dict[str, dict[str, np.ndarray]] = {}
    offsets = data["probe_offsets_rad"]
    for horizon in horizons:
        prefix = f"h{horizon}_"
        axis = data[prefix + "axis_rate_delta_rad_s"]
        max_axis = data[prefix + "max_positive_axis_delta_rad_s"]
        alignment = data[prefix + "mean_axis_alignment"]
        support_slope = data[prefix + "total_force_slope_n_s"]
        vertical_slope = data[prefix + "vertical_support_slope_n_s"]
        height_accel = data[prefix + "height_acceleration_m_s2"]
        min_height_mm = 1000.0 * data[prefix + "min_height_delta_m"]
        pair_margin = data[prefix + "transfer_pair_margin"]
        tim_response = data[prefix + "tim_response"].astype(np.float64)
        admissible = data[prefix + "short_admissible"].astype(np.float64)
        unsafe = data[prefix + "unsafe"].astype(np.float64)
        force_delta = data[prefix + "final_force_delta_n"]

        axis_gain = least_squares_gain_norms(offsets, axis)[..., 0]
        support_gain = least_squares_gain_norms(offsets, support_slope)[..., 0]
        vertical_gain = least_squares_gain_norms(offsets, vertical_slope)[..., 0]
        height_gain = least_squares_gain_norms(offsets, height_accel)[..., 0]
        finger_force_gain = least_squares_gain_norms(offsets, force_delta)
        metrics = {
            "axis_gain_norm": axis_gain,
            "axis_response_p90_rad_s": _quantile(max_axis, 0.90),
            "axis_alignment_p90": _quantile(alignment, 0.90),
            "support_gain_norm": support_gain,
            "vertical_support_gain_norm": vertical_gain,
            "height_acceleration_gain_norm": height_gain,
            "height_delta_p10_mm": _quantile(min_height_mm, 0.10),
            "transfer_pair_margin_p90": _quantile(pair_margin, 0.90),
            "TIM_response_rate": tim_response.mean(axis=1),
            "short_admissible_rate": admissible.mean(axis=1),
            "unsafe_rate": unsafe.mean(axis=1),
        }
        for finger_index, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            metrics[f"{finger}_force_gain_norm"] = finger_force_gain[:, finger_index]
        summaries[str(horizon)] = metrics
        for name, values in metrics.items():
            names.append(f"h{horizon}_{name}")
            columns.append(values)
    return names, np.stack(columns, axis=1), summaries


def _anchor_table(
    bank_ids: np.ndarray,
    summaries: dict[str, dict[str, np.ndarray]],
    horizons: list[int],
) -> dict[str, dict]:
    row_by_id = {int(value): index for index, value in enumerate(bank_ids)}
    table = {}
    for bank_id, outcome in ANCHOR_OUTCOMES.items():
        row = row_by_id[bank_id]
        table[str(bank_id)] = {
            "known_O1_12_outcome": outcome,
            "responses": {
                str(horizon): {
                    key: float(values[row])
                    for key, values in summaries[str(horizon)].items()
                }
                for horizon in horizons
            },
        }
    return table


def main() -> None:
    sidecar = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    with np.load(RAW_NPZ, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    bank_ids = data["bank_ids"]
    horizons = data["horizons_steps"].astype(int).tolist()
    names, features, summaries = _build_state_features(data, horizons)
    coordinates = principal_coordinates(features)

    longest = str(max(horizons))
    long_metrics = summaries[longest]
    long_admissible = data[f"h{max(horizons)}_short_admissible"].astype(np.int64)
    nonzero_probe_admissible = long_admissible[:, 1:].sum(axis=1)
    response_margin = np.minimum.reduce(
        (
            long_metrics["axis_alignment_p90"] / 0.5 - 1.0,
            long_metrics["transfer_pair_margin_p90"],
            (long_metrics["height_delta_p10_mm"] + 5.0) / 5.0,
            1.0 - long_metrics["unsafe_rate"],
        )
    )
    dynamic_score = 1.0 / (1.0 + np.exp(-np.clip(response_margin, -20.0, 20.0)))
    anchor_comparison = _anchor_table(bank_ids, summaries, horizons)
    row_by_id = {int(value): index for index, value in enumerate(bank_ids)}
    anchor_scores = {
        str(bank_id): float(dynamic_score[row_by_id[bank_id]])
        for bank_id in ANCHOR_OUTCOMES
    }
    feasible_rank = int(
        np.flatnonzero(
            bank_ids[np.argsort(dynamic_score)[::-1]] == 996
        )[0]
        + 1
    )
    top_ids = bank_ids[np.argsort(dynamic_score)[::-1][:20]].astype(int).tolist()
    separates_anchor = anchor_scores["996"] > max(
        anchor_scores[str(value)] for value in (1245, 1171, 1203)
    )
    anchor_admissible_counts = {
        str(bank_id): int(nonzero_probe_admissible[row_by_id[bank_id]])
        for bank_id in ANCHOR_OUTCOMES
    }
    gate_separates_anchor = anchor_admissible_counts["996"] > 0 and all(
        anchor_admissible_counts[str(value)] == 0 for value in (1245, 1171, 1203)
    )
    candidate_rows = np.flatnonzero(nonzero_probe_admissible > 0)
    candidate_ids = bank_ids[candidate_rows].astype(int).tolist()
    candidate_counts = nonzero_probe_admissible[candidate_rows].astype(int).tolist()

    report = {
        "status": "PASS_DIAGNOSTIC_DYNAMIC_MAP",
        "phase": "O1.14 Dynamic Handoff Readiness Audit",
        "artifact_role": "analysis_only_not_classifier_reset_filter_or_teacher",
        "bank_sha256": sidecar["bank_sha256_after"],
        "P3_rows": int(bank_ids.size),
        "probe_design": {
            "count": int(data["probe_offsets_rad"].shape[0]),
            "max_offset_rad": sidecar["max_offset_rad"],
            "type": sidecar["probe_design"],
            "horizons_steps": horizons,
            "horizons_ms": data["horizons_ms"].tolist(),
            "same_probe_set_for_every_state": True,
        },
        "dynamic_feature_schema": names,
        "dynamic_map": {
            "bank_ids": bank_ids.astype(int).tolist(),
            "parent_state_ids": data["parent_state_ids"].astype(int).tolist(),
            "source_trajectory_ids": data["source_trajectory_ids"].astype(int).tolist(),
            "PCA_coordinates": coordinates.tolist(),
            "response_margin": response_margin.tolist(),
            "dynamic_response_score": dynamic_score.tolist(),
            "score_definition": (
                "sigmoid(min(axis-alignment margin, transfer-pair p90 margin, "
                "height p10 margin, one-minus-unsafe-rate)); diagnostic envelope, "
                "not a calibrated feasibility probability"
            ),
        },
        "anchor_comparison": anchor_comparison,
        "anchor_dynamic_scores": anchor_scores,
        "anchor_nonzero_action_short_admissible_counts": anchor_admissible_counts,
        "bank996_dynamic_score_rank_of_P3": feasible_rank,
        "top20_dynamic_response_bank_ids": top_ids,
        "continuous_score_separates_known_anchors": separates_anchor,
        "short_admissible_gate_separates_known_anchors": gate_separates_anchor,
        "short_admissible_candidate_layer": {
            "states_with_at_least_one_nonzero_action_pass": int(candidate_rows.size),
            "fraction_of_P3": float(candidate_rows.size / bank_ids.size),
            "maximum_passes_per_state": int(nonzero_probe_admissible.max()),
            "bank_ids": candidate_ids,
            "pass_counts_out_of_nonzero_probes": candidate_counts,
            "bank996_tied_rank": int(
                1 + np.sum(nonzero_probe_admissible > nonzero_probe_admissible[row_by_id[996]])
            ),
            "promotion_status": "CANDIDATE_ONLY_REQUIRES_LONG_HORIZON_CEM_AND_CLONE_REPLAY",
        },
        "interpretation": (
            "The pre-specified joint short-window gate distinguishes bank 996 from "
            "the three known failure anchors, while the continuous response score "
            "does not. The gate also admits dozens of unlabeled P3 states, so it is "
            "a candidate generator rather than a feasibility classifier."
        ),
        "constraints": sidecar["constraints"],
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5), constrained_layout=True)
    scatter = axes[0].scatter(
        coordinates[:, 0], coordinates[:, 1], c=dynamic_score, cmap="viridis", s=22, alpha=0.65
    )
    anchor_colors = {996: "#2ca25f", 1245: "#d95f59", 1171: "#d95f59", 1203: "#e7a83e"}
    for bank_id, color in anchor_colors.items():
        row = row_by_id[bank_id]
        axes[0].scatter(
            coordinates[row, 0], coordinates[row, 1], marker="*", s=180,
            color=color, edgecolor="#202020", linewidth=1.0,
        )
        axes[0].annotate(str(bank_id), coordinates[row], xytext=(5, 5), textcoords="offset points")
    axes[0].scatter(
        coordinates[candidate_rows, 0], coordinates[candidate_rows, 1], marker="s",
        s=42, facecolors="none", edgecolors="#111111", linewidths=0.8,
        label="short-gate candidate",
    )
    axes[0].set_title("P3 dynamic-response map")
    axes[0].set_xlabel("Dynamic PCA 1")
    axes[0].set_ylabel("Dynamic PCA 2")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.colorbar(scatter, ax=axes[0], label="Diagnostic response score")

    anchor_order = (996, 1245, 1171, 1203)
    heat_metrics = (
        "axis_response_p90_rad_s", "axis_alignment_p90",
        "transfer_pair_margin_p90", "height_delta_p10_mm",
        "TIM_response_rate", "unsafe_rate",
    )
    heat = np.asarray(
        [[summaries[longest][metric][row_by_id[bank_id]] for metric in heat_metrics] for bank_id in anchor_order]
    )
    mean = np.nanmean(heat, axis=0)
    std = np.nanstd(heat, axis=0)
    heat_z = (heat - mean) / np.where(std > 1.0e-9, std, 1.0)
    image = axes[1].imshow(heat_z, cmap="coolwarm", vmin=-1.8, vmax=1.8, aspect="auto")
    axes[1].set_yticks(range(len(anchor_order)), [str(value) for value in anchor_order])
    heat_labels = ("Axis delta", "Axis align", "Transfer", "Height", "TIM response", "Unsafe")
    axes[1].set_xticks(range(len(heat_metrics)), heat_labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_title(f"Anchor contrast at {int(data['horizons_ms'][-1])} ms")
    fig.colorbar(image, ax=axes[1], label="Within-anchor z score")

    axes[2].scatter(
        long_metrics["axis_response_p90_rad_s"],
        long_metrics["height_delta_p10_mm"],
        c=long_metrics["transfer_pair_margin_p90"], cmap="plasma", s=22, alpha=0.6,
    )
    for bank_id, color in anchor_colors.items():
        row = row_by_id[bank_id]
        axes[2].scatter(
            long_metrics["axis_response_p90_rad_s"][row],
            long_metrics["height_delta_p10_mm"][row],
            marker="*", s=180, color=color, edgecolor="#202020",
        )
        axes[2].annotate(
            str(bank_id),
            (long_metrics["axis_response_p90_rad_s"][row], long_metrics["height_delta_p10_mm"][row]),
            xytext=(5, 5), textcoords="offset points",
        )
    axes[2].axhline(-5.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("90th percentile positive axis response (rad/s)")
    axes[2].set_ylabel("10th percentile minimum height delta (mm)")
    axes[2].set_title("Axis response and height")
    axes[2].grid(alpha=0.2)
    fig.suptitle("O1.14 Dynamic Handoff Readiness Audit", fontsize=15)
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    rows = []
    for bank_id in ANCHOR_OUTCOMES:
        row = row_by_id[bank_id]
        metrics = summaries[longest]
        rows.append(
            f"| `{bank_id}` | {ANCHOR_OUTCOMES[bank_id]} | "
            f"{metrics['axis_response_p90_rad_s'][row]:.3f} | "
            f"{metrics['axis_alignment_p90'][row]:.3f} | "
            f"{metrics['transfer_pair_margin_p90'][row]:.3f} | "
            f"{metrics['height_delta_p10_mm'][row]:.3f} | "
            f"{metrics['TIM_response_rate'][row]:.1%} | "
            f"{metrics['unsafe_rate'][row]:.1%} | {dynamic_score[row]:.3f} |"
        )
    decision = (
        "预先固定的短窗联合物理门在四个锚点上分离了 996 与三个失败锚点；"
        f"996 有 `{anchor_admissible_counts['996']}/{data['probe_offsets_rad'].shape[0] - 1}` 个非零 probe 通过，"
        "1245/1171/1203 均为 0。"
        if gate_separates_anchor
        else "短窗联合物理门未能分离已知成功与失败锚点。"
    )
    markdown = f"""# O1.14 Dynamic Handoff Readiness Audit 验收报告

## 结论

O1.14 已完成。对冻结 O0 中的 `{bank_ids.size}` 条 P3/train 状态施加了同一组
`{data['probe_offsets_rad'].shape[0]}` 个固定、对称、最大 `±{sidecar['max_offset_rad']:.2f} rad` 的 TIM 小动作 probe，
并在 `{data['horizons_ms'][0]:.0f}/{data['horizons_ms'][-1]:.0f} ms` 记录轴响应、承力响应、高度加速度和接触转换。

{decision}这是局部 action-conditioned 证据，不是 400 ms 严格 handoff 的可行性分类器。
全部 P3 中仍有 `{candidate_rows.size}/{bank_ids.size}` 个状态至少通过一个非零 probe，因此只能形成待后续 CEM 和 clone replay 验证的候选层。

## 四个代表状态

| Bank | O1.12 已知结局 | 轴响应 P90 | 轴对齐 P90 | 交接 margin P90 | 高度 P10 (mm) | TIM 响应 | 不安全 | 动态分数 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`996` 在全部 P3 中的连续动态响应分数排名为 `{feasible_rank}/{bank_ids.size}`，它没有靠连续分数单独脱颖而出。
按非零 short-admissible probe 数量排名时，996 并列第 `{report['short_admissible_candidate_layer']['bank996_tied_rank']}`。
两种分数都不是成功概率，也不能用来修改 P3 reset 分布。

## 边界

- 未训练 PPO，未修改 reward，未生成 teacher。
- O0 bank 前后 SHA-256 均为 `{sidecar['bank_sha256_after']}`。
- Probe 是预先固定的通用小扰动，不是 CEM 优化轨迹。
- 本轮只能用于提出后续独立 parent/source 的 CEM 候选，不能筛选、替换或重标 O0 P3。

## 产物

- 原始响应：`{RAW_NPZ}`
- 机器可读结论：`{OUTPUT_JSON}`
- 动态图：`{OUTPUT_PNG}`
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_JSON),
                "map": str(OUTPUT_PNG),
                "known_anchor_separation": separates_anchor,
                "short_gate_anchor_separation": gate_separates_anchor,
                "bank996_rank": feasible_rank,
                "candidate_states": int(candidate_rows.size),
                "anchor_scores": anchor_scores,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
