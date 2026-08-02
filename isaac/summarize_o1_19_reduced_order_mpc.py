from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
RAW_PATH = REPORT_DIR / "O1_19_Reduced_Order_MPC_Raw.json"
OUTPUT_JSON = REPORT_DIR / "O1_19_Reduced_Order_MPC_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_19_Reduced_Order_MPC_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_19_Reduced_Order_MPC_Map.png"


def _scenario_map(mode: dict) -> dict[str, dict]:
    return {row["name"]: row for row in mode["scenarios"]}


def main() -> None:
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    if raw["bank_sha256_before"] != raw["bank_sha256_after"]:
        raise RuntimeError("O0 hash changed during O1.19")
    modes = raw["held_out_evaluation"]
    scenarios = ("nominal", "joint_L1", "joint_L2", "joint_L3")
    labels = ("A open", "B task-space", "C MPC")
    colors = ("#8c96c6", "#e7a83e", "#2ca25f")
    comparison = []
    for mode, label in zip(modes, labels, strict=True):
        mapping = _scenario_map(mode)
        comparison.append(
            {
                "mode": mode["mode"],
                "label": label,
                "scenarios": [
                    {
                        "scenario": name,
                        "basin_rate": mapping[name]["recovered_strict_rate"],
                        "handoff_rate": mapping[name]["strict_rate"],
                        "height_recovery_rate": mapping[name]["height_recovery_rate"],
                        "min_height_delta_mm_p10": mapping[name]["min_height_delta_mm_p10"],
                        "unsafe_rate": mapping[name]["unsafe_rate"],
                        "failure_taxonomy": mapping[name]["failure_taxonomy"],
                    }
                    for name in scenarios
                ],
            }
        )
    report = {
        "status": raw["status"],
        "phase": raw["phase"],
        "artifact_role": raw["artifact_role"],
        "bank_id": raw["bank_id"],
        "bank_sha256": raw["bank_sha256_after"],
        "mpc_contract": raw["mpc_contract"],
        "jacobian_identification": raw["jacobian_identification"],
        "selected_mpc_controller": raw["selected_mpc_controller"],
        "selected_mpc_passed_calibration_safety": raw["selected_mpc_passed_calibration_safety"],
        "selection_fallback_used_for_diagnostic_comparison": raw[
            "selection_fallback_used_for_diagnostic_comparison"
        ],
        "selected_mpc_config": next(
            row["controller"]
            for row in raw["calibration"]
            if row["mode"] == raw["selected_mpc_controller"]
        ),
        "perturbation_protocol": raw["perturbation_protocol"],
        "calibration": raw["calibration"],
        "held_out_comparison": comparison,
        "widening_gate": raw["widening_gate"],
        "widening_gate_pass": raw["widening_gate_pass"],
        "interpretation": (
            "A/B/C use matched held-out perturbations. C performs 50 or 100 ms "
            "receding-horizon random shooting over bounded index/middle/thumb joint "
            "residuals using the in-memory reduced-order task model. The result is local "
            "to bank 1155 and is not a policy, teacher, global attractor, or sim-to-real claim."
        ),
        "constraints": raw["constraints"],
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    x = np.arange(len(scenarios))
    width = 0.24
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    metrics = (
        ("basin_rate", "Recovered-strict basin", 0.30, False),
        ("handoff_rate", "Strict handoff", None, False),
        ("min_height_delta_mm_p10", "P10 downward height excursion", None, True),
        ("unsafe_rate", "Unsafe", None, False),
    )
    for axis, (metric, title, gate, negate) in zip(axes.flat, metrics, strict=True):
        plotted = []
        for index, (mode, label, color) in enumerate(zip(comparison, labels, colors, strict=True)):
            values = [row[metric] for row in mode["scenarios"]]
            if negate:
                values = [-value for value in values]
            plotted.extend(values)
            axis.bar(x + (index - 1) * width, values, width, label=label, color=color)
        if gate is not None:
            axis.axhline(gate, color="#333333", linestyle=":", label="L1 gate")
        axis.set_xticks(x, scenarios)
        axis.set_ylim(0.0, max(0.01, 1.15 * max(plotted)) if negate else 1.0)
        axis.set_title(title)
        axis.set_ylabel("mm" if negate else "Rate")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    result_rows = []
    for mode in comparison:
        for row in mode["scenarios"]:
            result_rows.append(
                f"| {mode['label']} | `{row['scenario']}` | {row['basin_rate']:.2%} | "
                f"{row['handoff_rate']:.2%} | {row['height_recovery_rate']:.2%} | "
                f"{row['min_height_delta_mm_p10']:.3f} | {row['unsafe_rate']:.2%} |"
            )
    calibration_rows = "\n".join(
        f"| `{row['mode']}` | {row['scenarios'][0]['recovered_strict_rate']:.2%} | "
        f"{row['scenarios'][1]['recovered_strict_rate']:.2%} | "
        f"{row['scenarios'][1]['unsafe_rate']:.2%} | {row['selection_score']:.4f} |"
        for row in raw["calibration"]
    )
    gate_rows = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in raw["widening_gate"].items()
    )
    mpc_taxonomy_rows = "\n".join(
        f"| `{row['scenario']}` | "
        + "; ".join(
            f"`{name}`={count}"
            for name, count in sorted(
                row["failure_taxonomy"].items(), key=lambda item: item[1], reverse=True
            )
        )
        + " |"
        for row in comparison[2]["scenarios"]
    )
    verdict = (
        "Reduced-order MPC 通过全部预注册 widening gates。"
        if raw["widening_gate_pass"]
        else "Reduced-order MPC 未通过全部预注册 widening gates。"
    )
    markdown = f"""# O1.19 Reduced-order Handoff MPC Feasibility Audit 验收报告

## 结论

{verdict}

用于正式诊断比较的 MPC：`{raw['selected_mpc_controller']}`。Calibration safety eligibility：`{'PASS' if raw['selected_mpc_passed_calibration_safety'] else 'FAIL'}`。

## Calibration

| Mode | Nominal basin | Joint-L1 basin | Joint-L1 unsafe | Score |
|---|---:|---:|---:|---:|
{calibration_rows}

## Held-out A/B/C

| Mode | Scenario | Basin | Handoff | Height recovery | Min height p10 (mm) | Unsafe |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(result_rows)}

## Widening gates

| Gate | 结果 |
|---|---|
{gate_rows}

## Failure taxonomy

| MPC scenario | Episode taxonomy |
|---|---|
{mpc_taxonomy_rows}

完整 A/B/C taxonomy 同时保存在机器可读 JSON 的 `held_out_comparison[*].scenarios[*].failure_taxonomy`。

## 方法边界

- MPC horizon 为 50/100 ms，每步从 64 个候选序列中选择第一步动作并重新规划。
- 只优化 index/middle/thumb 的 12 个关节，残差上限为 `0.025 rad`。
- 预测约束覆盖支撑力、高度、轴向、聚合接触和动作边界。
- MPC 序列、CEM plan、Jacobian 和参考轨迹均未保存。
- 未训练 PPO、未修改 reward/O0、未生成 teacher/reset dataset。

## 产物

- 机器可读报告：`{OUTPUT_JSON}`
- 可视化：`{OUTPUT_PNG}`
- 原始结果：`{RAW_PATH}`
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_JSON), "map": str(OUTPUT_PNG), "status": raw["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
