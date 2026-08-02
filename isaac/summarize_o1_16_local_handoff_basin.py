from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
RAW_PATH = REPORT_DIR / "O1_16_Local_Handoff_Basin_Raw.json"
OUTPUT_JSON = REPORT_DIR / "O1_16_Local_Handoff_Basin_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_16_Local_Handoff_Basin_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_16_Local_Handoff_Basin_Map.png"


def _scenario_map(state: dict) -> dict[str, dict]:
    return {row["name"]: row for row in state["scenarios"]}


def _classification(nominal_rate: float, joint_l1_rate: float) -> str:
    if nominal_rate <= 0.0:
        return "PLAN_RECOVERY_FAILED_BASIN_NOT_MEASURED"
    if joint_l1_rate >= 0.30:
        return "EXPANDABLE_LOCAL_BASIN"
    if joint_l1_rate >= 0.05:
        return "NARROW_LOCAL_BASIN"
    return "ISOLATED_TRAJECTORY_SOLUTION"


def _factor_widths(state: dict) -> dict[str, dict]:
    scenarios = _scenario_map(state)
    nominal = scenarios["nominal"]["recovered_strict_rate"]
    factors = sorted({row["factor"] for row in state["scenarios"] if row["factor"] not in ("nominal", "joint")})
    result = {}
    for factor in factors:
        rows = sorted(
            [row for row in state["scenarios"] if row["factor"] == factor],
            key=lambda row: row["level"],
        )
        absolute = [row for row in rows if row["recovered_strict_rate"] >= 0.10]
        relative = [
            row for row in rows
            if nominal > 0.0 and row["recovered_strict_rate"] >= 0.5 * nominal
        ]
        result[factor] = {
            "rates": [row["recovered_strict_rate"] for row in rows],
            "scales": [row["scales"][factor] for row in rows],
            "largest_scale_at_absolute_10pct": (
                absolute[-1]["scales"][factor] if absolute else None
            ),
            "largest_scale_at_half_nominal": (
                relative[-1]["scales"][factor] if relative else None
            ),
        }
    return result


def main() -> None:
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    if raw["bank_sha256_before"] != raw["bank_sha256_after"]:
        raise RuntimeError("O0 hash changed during O1.16")
    states = []
    for source in raw["states"]:
        scenarios = _scenario_map(source)
        nominal = scenarios["nominal"]["recovered_strict_rate"]
        joint_rates = [scenarios[f"joint_L{level}"]["recovered_strict_rate"] for level in range(1, 4)]
        states.append(
            {
                "bank_id": source["bank_id"],
                "horizon_ms": source["horizon_ms"],
                "nominal_recovered_strict_rate": nominal,
                "joint_L1_rate": joint_rates[0],
                "joint_L2_rate": joint_rates[1],
                "joint_L3_rate": joint_rates[2],
                "plan_recovered": nominal > 0.0,
                "classification": _classification(nominal, joint_rates[0]),
                "factor_widths": _factor_widths(source),
                "search_best_score": source["search"]["best_score"],
                "scenario_results": source["scenarios"],
            }
        )
    states.sort(
        key=lambda row: (
            row["joint_L2_rate"], row["joint_L1_rate"], row["nominal_recovered_strict_rate"]
        ),
        reverse=True,
    )
    for rank, row in enumerate(states, start=1):
        row["rank"] = rank

    recovered_ids = {row["bank_id"] for row in states if row["plan_recovered"]}
    joint_l2_taxonomy = Counter()
    for state in raw["states"]:
        if state["bank_id"] not in recovered_ids:
            continue
        row = _scenario_map(state)["joint_L2"]
        joint_l2_taxonomy.update(row["failure_taxonomy"])
    total_replays = sum(row["episodes"] for state in raw["states"] for row in state["scenarios"])
    classification_counts = Counter(row["classification"] for row in states)
    report = {
        "status": "PASS_DIAGNOSTIC_ONE_NARROW_OPEN_LOOP_BASIN",
        "phase": "O1.16 Local Handoff Basin Perturbation Audit",
        "artifact_role": "local_open_loop_plan_basin_lower_bound_not_policy_attractor_or_reset_dataset",
        "bank_sha256": raw["bank_sha256_after"],
        "search": raw["search"],
        "perturbation_design": raw["perturbation_design"],
        "total_replay_episodes": total_replays,
        "classification_gate": {
            "PLAN_RECOVERY_FAILED_BASIN_NOT_MEASURED": (
                "nominal recovered-strict rate = 0%; no basin inference"
            ),
            "EXPANDABLE_LOCAL_BASIN": "joint L1 recovered-strict rate >=30%",
            "NARROW_LOCAL_BASIN": "joint L1 recovered-strict rate in [5%, 30%)",
            "ISOLATED_TRAJECTORY_SOLUTION": "joint L1 recovered-strict rate <5%",
        },
        "classification_counts": dict(classification_counts),
        "state_ranking": states,
        "joint_L2_failure_taxonomy_plan_recovered_only": dict(joint_l2_taxonomy),
        "interpretation": (
            "The audit re-finds each CEM plan in memory and measures fixed-plan replay "
            "under reset-state perturbations. It is a lower bound on local open-loop "
            "robustness, not a closed-loop policy attractor estimate. A zero nominal "
            "rate means plan recovery failed in this independent search and leaves the "
            "basin unmeasured; it is not evidence of dynamics unreachability. Contact "
            "force is recomputed after contact rebuild rather than directly perturbed."
        ),
        "constraints": raw["constraints"],
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    raw_by_id = {state["bank_id"]: state for state in raw["states"]}
    scenario_names = [row["name"] for row in raw["states"][0]["scenarios"]]
    rates = np.asarray(
        [
            (
                [_scenario_map(raw_by_id[state["bank_id"]])[name]["recovered_strict_rate"] for name in scenario_names]
                if state["plan_recovered"]
                else [np.nan] * len(scenario_names)
            )
            for state in states
        ]
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), constrained_layout=True)
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("#d9d9d9")
    image = axes[0].imshow(
        rates,
        cmap=color_map,
        vmin=0.0,
        vmax=max(0.30, float(np.nanmax(rates))),
        aspect="auto",
    )
    axes[0].set_yticks(
        range(len(states)),
        [
            str(state["bank_id"]) if state["plan_recovered"] else f"{state['bank_id']} (not measured)"
            for state in states
        ],
    )
    axes[0].set_xticks(range(len(scenario_names)), scenario_names, rotation=70, ha="right", fontsize=7)
    axes[0].set_title("Recovered-strict replay rates")
    axes[0].set_ylabel("P3 seed, ranked")
    fig.colorbar(image, ax=axes[0], label="Success rate")

    colors = {894: "#2ca25f", 1155: "#6a51a3", 831: "#e7a83e"}
    factor_order = list(raw["perturbation_design"]["scales"])
    x = np.arange(1, 4)
    for state in states:
        if not state["plan_recovered"]:
            continue
        source = raw_by_id[state["bank_id"]]
        mapping = _scenario_map(source)
        mean_rates = [
            np.mean([mapping[f"{factor}_L{level}"]["recovered_strict_rate"] for factor in factor_order])
            for level in range(1, 4)
        ]
        axes[1].plot(x, mean_rates, marker="o", linewidth=2, color=colors[state["bank_id"]], label=str(state["bank_id"]))
        axes[1].plot(x, [mapping[f"joint_L{level}"]["recovered_strict_rate"] for level in range(1, 4)], marker="s", linestyle="--", color=colors[state["bank_id"]], alpha=0.8)
    axes[1].axhline(0.30, color="#333333", linestyle=":", label="Expandable gate")
    axes[1].axhline(0.05, color="#777777", linestyle=":", label="Narrow gate")
    axes[1].set_xticks(x, ["L1", "L2", "L3"])
    axes[1].set_ylim(0.0, max(0.35, float(np.nanmax(rates)) * 1.08))
    axes[1].set_ylabel("Recovered-strict rate")
    axes[1].set_title("One-factor mean (solid) vs joint (dashed)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].grid(alpha=0.2)

    labels = list(joint_l2_taxonomy)
    values = [joint_l2_taxonomy[label] for label in labels]
    display = [label.replace("_", " ") for label in labels]
    axes[2].barh(display, values, color="#4f81bd")
    axes[2].invert_yaxis()
    axes[2].set_title("Joint-L2 failure taxonomy (plan recovered)")
    axes[2].set_xlabel("Episodes across banks 1155 and 831")
    for index, value in enumerate(values):
        axes[2].text(value + 1, index, str(value), va="center", fontsize=8)
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    state_rows = "\n".join(
        f"| {row['rank']} | `{row['bank_id']}` | {row['horizon_ms']:.0f} | "
        f"{row['nominal_recovered_strict_rate']:.1%} | {row['joint_L1_rate']:.1%} | "
        f"{row['joint_L2_rate']:.1%} | {row['joint_L3_rate']:.1%} | `{row['classification']}` |"
        for row in states
    )
    width_sections = []
    for state in states:
        rows = "\n".join(
            f"| `{factor}` | {values['largest_scale_at_absolute_10pct']} | "
            f"{values['largest_scale_at_half_nominal']} | "
            f"{' / '.join(f'{rate:.1%}' for rate in values['rates'])} |"
            for factor, values in state["factor_widths"].items()
        )
        width_sections.append(
            f"### Bank {state['bank_id']}\n\n"
            "| Factor | Largest scale @10% | Largest scale @half nominal | L1/L2/L3 |\n"
            "|---|---:|---:|---|\n" + rows
        )
    taxonomy_rows = "\n".join(f"| `{name}` | {value} |" for name, value in joint_l2_taxonomy.items())
    markdown = f"""# O1.16 Local Handoff Basin Perturbation Audit 验收报告

## 结论

O1.16 已在同一 PhysX 进程内重新搜索 `894/1155/831` 的 CEM plan，然后对每个 plan 执行结构化 reset-state 扰动重放。
共评估 `{total_replays}` 个 episode；动作 plan 和扰动状态都未序列化。

| Rank | Bank | Horizon | Nominal | Joint L1 | Joint L2 | Joint L3 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---|
{state_rows}

判定口径：先要求 nominal plan 在本轮至少复现一次；否则标记为 `PLAN_RECOVERY_FAILED_BASIN_NOT_MEASURED`，不推断 basin。复现后，Joint-L1 `>=30%` 为可扩展局部 basin，`5%-30%` 为窄 basin，`<5%` 为近似孤立轨迹解。

本轮得到一个窄 open-loop basin（bank `1155`）、一个近似孤立轨迹解（bank `831`）；bank `894` 的独立 CEM 未恢复 nominal plan，因此其 O1.15 历史可行性仍保留，但本轮 basin 宽度未测得。

## 单因素宽度

{chr(10).join(width_sections)}

## Joint-L2 failure taxonomy（仅 nominal plan 已恢复）

| 类型 | Episode |
|---|---:|
{taxonomy_rows}

## 解释边界

- 这是固定 open-loop plan 的局部鲁棒性下界，不是 closed-loop policy attractor 的体积。
- nominal 为零表示本轮搜索没有恢复 plan，不表示该状态动力学不可达，也不撤销 O1.15 的条件存在性证据。
- `q` 和 `q_target` 同步移动，保留原始控制上下文；`target_context` 另行审计。
- `q/dq/pen state` 的尺度均为逐分量均匀扰动上界；Joint-L1 是多组因素同时扰动，不是单一向量范数半径。
- Contact force 没有被直接设置，而是扰动后经过两步 contact rebuild 重新求解。
- 未训练 PPO，未修改 reward/O0，未生成 teacher 或 reset dataset。

## 产物

- 机器可读结果：`{OUTPUT_JSON}`
- 可视化：`{OUTPUT_PNG}`
- 原始汇总：`{RAW_PATH}`
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_JSON),
                "map": str(OUTPUT_PNG),
                "ranking": [
                    {
                        "bank_id": row["bank_id"],
                        "joint_L1": row["joint_L1_rate"],
                        "classification": row["classification"],
                    }
                    for row in states
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
