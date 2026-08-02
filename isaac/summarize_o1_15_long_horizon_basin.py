from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
O1_14_PATH = REPORT_DIR / "O1_14_Dynamic_Handoff_Readiness_Audit.json"
OUTPUT_JSON = REPORT_DIR / "O1_15_Long_Horizon_Handoff_Basin_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_15_Long_Horizon_Handoff_Basin_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_15_Long_Horizon_Handoff_Basin_Map.png"
ROBUST_CLONE_RATE = 0.50


def _wilson(successes: int, count: int, z: float = 1.96) -> list[float]:
    if count == 0:
        return [0.0, 0.0]
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return [center - half, center + half]


def _record_level(record: dict) -> int:
    robust = record["clone_replay_recovered_strict_rate"] >= ROBUST_CLONE_RATE
    if robust:
        return 4
    if record["clone_replay_recovered_strict_rate"] > 0.0:
        return 3
    if record["clone_replay_feasible_rate"] > 0.0:
        return 2
    if record["tim_gait"]:
        return 1
    return 0


def _best_record(records: list[dict]) -> dict:
    return max(
        records,
        key=lambda row: (
            _record_level(row),
            row["clone_replay_recovered_strict_rate"],
            row["clone_replay_feasible_rate"],
            row["handoff_margin"],
        ),
    )


def _failure(record: dict) -> str:
    level = _record_level(record)
    if level == 4:
        return "robust_recovered_handoff"
    if level == 3:
        return "recovered_but_low_clone_reproducibility"
    if level == 2:
        return "strict_but_height_not_recovered"
    if record["tim_gait"] and record["primary_failure"] == "axis_misaligned":
        return "axis_misaligned_after_transition"
    return record["primary_failure"]


def main() -> None:
    o1_14 = json.loads(O1_14_PATH.read_text(encoding="utf-8"))
    candidate = o1_14["short_admissible_candidate_layer"]
    candidate_ids = [int(value) for value in candidate["bank_ids"]]
    pass_counts = dict(zip(candidate_ids, candidate["pass_counts_out_of_nonzero_probes"], strict=True))
    batch_paths = sorted(REPORT_DIR.glob("O1_15_CEM_batch_*.json"))
    if not batch_paths:
        raise RuntimeError("No O1.15 CEM batch reports found")
    batches = [json.loads(path.read_text(encoding="utf-8")) for path in batch_paths]
    bank_hashes = {batch["bank_sha256"] for batch in batches}
    if bank_hashes != {o1_14["bank_sha256"]}:
        raise RuntimeError("O1.14 and O1.15 do not use the same frozen O0 bank")
    expected_config = {
        "population": 64,
        "iterations": 10,
        "waypoints": 3,
        "action_mode": "all",
        "max_offset_rad": 0.18,
        "horizons_steps": [18, 24, 30],
    }
    for batch in batches:
        for key, expected in expected_config.items():
            if batch[key] != expected:
                raise RuntimeError(f"Mismatched O1.15 batch {key}: {batch[key]}")
    measured_ids = [bank_id for batch in batches for bank_id in batch["source_bank_ids"]]
    if len(measured_ids) != len(set(measured_ids)):
        raise RuntimeError("Duplicate P3 state across O1.15 batches")
    if set(measured_ids) != set(candidate_ids):
        raise RuntimeError("O1.15 batches do not exactly cover the O1.14 candidate layer")

    records_by_state: dict[int, list[dict]] = {bank_id: [] for bank_id in candidate_ids}
    horizon_ms: dict[int, float] = {}
    for batch in batches:
        for horizon in batch["horizons"]:
            steps = int(horizon["horizon_steps"])
            horizon_ms[steps] = float(horizon["horizon_ms"])
            for row in horizon["states"]:
                item = dict(row)
                item["horizon_steps"] = steps
                item["horizon_ms"] = float(horizon["horizon_ms"])
                records_by_state[int(row["bank_id"])].append(item)
    if any(len(rows) != 3 for rows in records_by_state.values()):
        raise RuntimeError("Every candidate must have exactly three horizon records")

    map_data = o1_14["dynamic_map"]
    parent_by_id = dict(zip(map_data["bank_ids"], map_data["parent_state_ids"], strict=True))
    source_by_id = dict(zip(map_data["bank_ids"], map_data["source_trajectory_ids"], strict=True))
    state_results = []
    for bank_id in candidate_ids:
        rows = sorted(records_by_state[bank_id], key=lambda row: row["horizon_steps"])
        best = dict(_best_record(rows))
        best["O1_14_nonzero_probe_passes"] = int(pass_counts[bank_id])
        best["parent_state_id"] = int(parent_by_id[bank_id])
        best["source_trajectory_id"] = int(source_by_id[bank_id])
        best["basin_level"] = _record_level(best)
        best["O1_15_failure"] = _failure(best)
        best["horizon_results"] = rows
        state_results.append(best)
    state_results.sort(
        key=lambda row: (
            row["basin_level"],
            row["clone_replay_recovered_strict_rate"],
            row["clone_replay_feasible_rate"],
            row["handoff_margin"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(state_results, start=1):
        row["rank"] = rank

    count = len(state_results)
    existence = sum(
        any(row["clone_replay_feasible_rate"] > 0.0 for row in records_by_state[bank_id])
        for bank_id in candidate_ids
    )
    recovered = sum(
        any(
            row["clone_replay_recovered_strict_rate"] > 0.0
            for row in records_by_state[bank_id]
        )
        for bank_id in candidate_ids
    )
    robust = sum(row["basin_level"] == 4 for row in state_results)
    existence_rows = [row for row in state_results if row["basin_level"] >= 2]
    recovered_rows = [row for row in state_results if row["basin_level"] >= 3]
    robust_rows = [row for row in state_results if row["basin_level"] == 4]
    taxonomy = dict(Counter(row["O1_15_failure"] for row in state_results))
    horizon_summary = {}
    for steps in sorted(horizon_ms):
        rows = [record for records in records_by_state.values() for record in records if record["horizon_steps"] == steps]
        horizon_summary[str(steps)] = {
            "horizon_ms": horizon_ms[steps],
            "TIM_transition": sum(row["tim_gait"] for row in rows),
            "strict_handoff": sum(row["clone_replay_feasible_rate"] > 0.0 for row in rows),
            "strict_and_height_recovered": sum(row["clone_replay_recovered_strict_rate"] > 0.0 for row in rows),
            "robust_clone_replay": sum(row["clone_replay_recovered_strict_rate"] >= ROBUST_CLONE_RATE for row in rows),
        }

    report = {
        "status": "PASS_DIAGNOSTIC_LONG_HORIZON_BASIN_MEASURED",
        "phase": "O1.15 Long-Horizon Handoff Basin Audit",
        "artifact_role": "conditional_reachability_summary_not_teacher_reset_filter_or_training_data",
        "bank_sha256": next(iter(bank_hashes)),
        "input": {
            "source": str(O1_14_PATH),
            "candidate_states": count,
            "selection": "all O1.14 states with at least one nonzero 100 ms short-gate probe",
        },
        "search": {
            **expected_config,
            "horizons_ms": [horizon_ms[value] for value in expected_config["horizons_steps"]],
            "batch_count": len(batches),
            "action_plan_serialized": False,
        },
        "handoff_basin_size": {
            "denominator": count,
            "scope": "conditional on the O1.14 37-state candidate layer, not all frozen P3",
            "strict_existence": existence,
            "strict_existence_rate": existence / count,
            "strict_existence_Wilson95": _wilson(existence, count),
            "strict_and_height_recovered": recovered,
            "strict_and_height_recovered_rate": recovered / count,
            "strict_existence_independent_parent_count": len(
                {row["parent_state_id"] for row in existence_rows}
            ),
            "strict_existence_independent_source_count": len(
                {row["source_trajectory_id"] for row in existence_rows}
            ),
            "recovered_independent_parent_count": len(
                {row["parent_state_id"] for row in recovered_rows}
            ),
            "recovered_independent_source_count": len(
                {row["source_trajectory_id"] for row in recovered_rows}
            ),
            "robust_recovered_clone_rate_at_least_0P50": robust,
            "robust_recovered_rate": robust / count,
            "independent_parent_count": len({row["parent_state_id"] for row in robust_rows}),
            "independent_source_count": len({row["source_trajectory_id"] for row in robust_rows}),
        },
        "horizon_summary": horizon_summary,
        "failure_taxonomy": taxonomy,
        "state_ranking": state_results,
        "cross_phase_note": {
            "bank_996_rank": next(row["rank"] for row in state_results if row["bank_id"] == 996),
            "bank_996_outcome": next(row["O1_15_failure"] for row in state_results if row["bank_id"] == 996),
            "interpretation": (
                "Bank 996 does not reproduce strict feasibility under the O1.15 "
                "population-64 x 10 one-stage search. This does not withdraw O1.12, "
                "which used a population-128 x 20 two-stage handoff plus post-axis "
                "stabilization search; it demonstrates search-budget and parameterization sensitivity."
            ),
        },
        "interpretation": (
            "This is a practical long-horizon CEM estimate inside the O1.14 candidate "
            "layer. It does not estimate the full P3 measure and cannot modify O0. "
            "Existence, terminal height recovery, and clone reproducibility are reported "
            "separately to avoid promoting one lucky rollout as a basin."
        ),
        "constraints": {
            "PPO_trained": False,
            "reward_modified": False,
            "O0_modified": False,
            "teacher_generated": False,
            "trajectory_exported": False,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    ordered_ids = [row["bank_id"] for row in state_results]
    matrix = np.zeros((count, 3), dtype=np.int64)
    for row_index, bank_id in enumerate(ordered_ids):
        by_horizon = {row["horizon_steps"]: row for row in records_by_state[bank_id]}
        for column, steps in enumerate(sorted(horizon_ms)):
            matrix[row_index, column] = _record_level(by_horizon[steps])
    figure_height = max(9.0, 0.27 * count)
    fig, axes = plt.subplots(
        1, 3, figsize=(17.5, figure_height),
        gridspec_kw={"width_ratios": (1.0, 1.35, 0.9)}, constrained_layout=True,
    )
    axes[0].imshow(
        matrix,
        cmap=ListedColormap(["#d95f59", "#e7b34c", "#4f81bd", "#6a51a3", "#2ca25f"]),
        vmin=0, vmax=4, aspect="auto",
    )
    axes[0].set_xticks(range(3), [f"{horizon_ms[value]:.0f} ms" for value in sorted(horizon_ms)])
    axes[0].set_yticks(range(count), [str(value) for value in ordered_ids], fontsize=7)
    axes[0].set_xlabel("CEM horizon")
    axes[0].set_ylabel("P3 bank ID, ranked")
    axes[0].set_title("Long-horizon handoff levels")
    from matplotlib.patches import Patch
    axes[0].legend(
        handles=[
            Patch(color="#d95f59", label="0: no TIM transition"),
            Patch(color="#e7b34c", label="1: TIM transition only"),
            Patch(color="#4f81bd", label="2: strict, no height recovery"),
            Patch(color="#6a51a3", label="3: recovered, low clone rate"),
            Patch(color="#2ca25f", label="4: clone rate >=50%"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2,
        frameon=False, fontsize=8,
    )

    y = np.arange(count)
    clone_rates = [row["clone_replay_recovered_strict_rate"] for row in state_results]
    margins = [row["handoff_margin"] for row in state_results]
    axes[1].barh(y, clone_rates, color="#2ca25f", alpha=0.85, label="Recovered strict clone rate")
    axes[1].axvline(ROBUST_CLONE_RATE, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(y, [str(value) for value in ordered_ids], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Clone replay rate")
    axes[1].set_title("Clone reproducibility")
    margin_axis = axes[1].twiny()
    margin_axis.plot(margins, y, "o", color="#6a51a3", markersize=3, label="Handoff margin")
    margin_axis.set_xlabel("Handoff margin")

    labels = list(taxonomy)
    display_labels = {
        "recovered_but_low_clone_reproducibility": "Recovered, low clone rate",
        "contact_force_overload": "Contact-force overload",
        "axis_misaligned_after_transition": "Axis misaligned",
        "no_TIM_transition": "No TIM transition",
    }
    values = [taxonomy[label] for label in labels]
    axes[2].barh([display_labels.get(label, label) for label in labels], values, color="#4f81bd")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("States")
    axes[2].set_title("Best-horizon failure taxonomy")
    for index, value in enumerate(values):
        axes[2].text(value + 0.15, index, str(value), va="center", fontsize=8)
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    top_rows = "\n".join(
        f"| {row['rank']} | `{row['bank_id']}` | {row['horizon_ms']:.0f} | "
        f"{row['basin_level']} | {row['clone_replay_recovered_strict_rate']:.1%} | "
        f"{row['axis_alignment']:.3f} | {row['min_height_delta_mm']:.3f} | "
        f"{row['final_height_delta_mm']:.3f} | `{row['O1_15_failure']}` |"
        for row in state_results[:20]
    )
    taxonomy_rows = "\n".join(f"| `{name}` | {value} |" for name, value in taxonomy.items())
    basin = report["handoff_basin_size"]
    markdown = f"""# O1.15 Long-Horizon Handoff Basin Audit 验收报告

## 结论

O1.15 对 O1.14 的全部 `{count}` 条动态候选执行了匹配的 `300/400/500 ms` CEM。
这里的 basin size 是 **O1.14 候选层内的条件估计**，不是全部 P3 的体积或成功率。

- 至少一个 horizon 找到 strict handoff：`{basin['strict_existence']}/{count} = {basin['strict_existence_rate']:.2%}`
- strict 且末端高度恢复到 `-2 mm` 内：`{basin['strict_and_height_recovered']}/{count} = {basin['strict_and_height_recovered_rate']:.2%}`
- recovered-strict clone replay `>=50%`：`{basin['robust_recovered_clone_rate_at_least_0P50']}/{count} = {basin['robust_recovered_rate']:.2%}`
- 存在性成功来自 `{basin['strict_existence_independent_parent_count']}` 个 parent、`{basin['strict_existence_independent_source_count']}` 个 source trajectory。
- 鲁棒子集来自 `{basin['independent_parent_count']}` 个 parent、`{basin['independent_source_count']}` 个 source trajectory。

`996` 在本轮排名第 `{report['cross_phase_note']['bank_996_rank']}`，结局为 `{report['cross_phase_note']['bank_996_outcome']}`。
这不撤回 O1.12 的可达性证据：O1.12 使用 `128x20` 两阶段 handoff+post-axis 搜索，
本轮为 `64x10` 单阶段搜索，差异反映搜索预算和参数化敏感性。

## Horizon 对比

| Horizon | TIM transition | Strict | Strict + height recovery | Robust clone |
|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {item['horizon_ms']:.0f} ms | {item['TIM_transition']}/{count} | "
        f"{item['strict_handoff']}/{count} | {item['strict_and_height_recovered']}/{count} | "
        f"{item['robust_clone_replay']}/{count} |"
        for item in horizon_summary.values()
    ) + f"""

## 状态排名（前 20）

Level：`0=无转换，1=TIM转换，2=strict，3=strict+高度恢复，4=clone>=50%`。

| Rank | Bank | Horizon (ms) | Level | Clone recovered-strict | Axis | Min z (mm) | Final z (mm) | 结局 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
{top_rows}

## Failure taxonomy

| 类型 | 状态数 |
|---|---:|
{taxonomy_rows}

## 边界

- CEM 未序列化最优 action plan，也未生成 teacher。
- 未训练 PPO，未修改 reward 或 O0。
- 一次存在性 rollout 与 clone 可复现性分开报告，不把偶然成功直接当作 basin。

## 产物

- 机器可读结果：`{OUTPUT_JSON}`
- 可视化：`{OUTPUT_PNG}`
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_JSON),
                "map": str(OUTPUT_PNG),
                "strict_existence": existence,
                "recovered": recovered,
                "robust": robust,
                "taxonomy": taxonomy,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
