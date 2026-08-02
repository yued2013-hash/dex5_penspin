from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
RAW_PATH = REPORT_DIR / "O1_18_Task_Space_Basin_Raw.json"
OUTPUT_JSON = REPORT_DIR / "O1_18_Task_Space_Basin_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_18_Task_Space_Basin_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_18_Task_Space_Basin_Map.png"


def _scenario_map(mode: dict) -> dict[str, dict]:
    return {row["name"]: row for row in mode["scenarios"]}


def main() -> None:
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    if raw["bank_sha256_before"] != raw["bank_sha256_after"]:
        raise RuntimeError("O0 hash changed during O1.18")
    modes = raw["held_out_evaluation"]
    scenarios = ("nominal", "joint_L1", "joint_L2", "joint_L3")
    labels = ("A open", "B q-track", "C task-space")
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
        "task_feature_contract": raw["task_feature_contract"],
        "jacobian_identification": raw["jacobian_identification"],
        "selected_task_controller": raw["selected_task_controller"],
        "selected_task_controller_passed_calibration_safety": raw[
            "selected_task_controller_passed_calibration_safety"
        ],
        "selection_fallback_used_for_diagnostic_comparison": raw[
            "selection_fallback_used_for_diagnostic_comparison"
        ],
        "selected_task_controller_config": next(
            row["controller"]
            for row in raw["calibration"]
            if row["mode"] == raw["selected_task_controller"]
        ),
        "perturbation_protocol": raw["perturbation_protocol"],
        "calibration": raw["calibration"],
        "held_out_comparison": comparison,
        "widening_gate": raw["widening_gate"],
        "widening_gate_pass": raw["widening_gate_pass"],
        "interpretation": (
            "A/B/C use matched held-out perturbations. C uses a three-stage local "
            "task Jacobian identified in memory from antithetic probes and maps pen "
            "height, angular velocity, aggregate support wrench, and aggregate contact "
            "count error to bounded joint-target residuals. The result is local to bank "
            "1155 and does not establish a global attractor or sim-to-real robustness."
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
        plotted_values = []
        for index, (mode, label, color) in enumerate(zip(comparison, labels, colors, strict=True)):
            values = [row[metric] for row in mode["scenarios"]]
            if negate:
                values = [-value for value in values]
            plotted_values.extend(values)
            axis.bar(x + (index - 1) * width, values, width, label=label, color=color)
        if gate is not None:
            axis.axhline(gate, color="#333333", linestyle=":", label="L1 gate")
        axis.set_xticks(x, scenarios)
        axis.set_ylim(0.0, max(0.01, 1.15 * max(plotted_values)) if negate else 1.0)
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
                f"{row['min_height_delta_mm_p10']:.3f} | "
                f"{row['unsafe_rate']:.2%} |"
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
    verdict = (
        "Task-space residual feedback 通过全部预注册 widening gates。"
        if raw["widening_gate_pass"]
        else "Task-space residual feedback 未通过全部预注册 widening gates。"
    )
    markdown = f"""# O1.18 Task-Space Closed-Loop Basin Audit 验收报告

## 结论

{verdict}

用于诊断比较的 task-space 控制器：`{raw['selected_task_controller']}`。A/B/C 在 held-out 集合内使用完全匹配的扰动。

Calibration safety eligibility：`{'PASS' if raw['selected_task_controller_passed_calibration_safety'] else 'FAIL'}`。若为 FAIL，C 只是最优 fallback 的诊断结果，不是合格控制器。

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

## 方法边界

- Task input：笔高度、笔角速度、总支撑 force/torque、聚合接触数量。
- 输出为最大 `0.02-0.04 rad` 的小关节目标残差；没有指定某根手指必须接触。
- 三阶段 task Jacobian 通过内存中的正负关节探针辨识；Jacobian、CEM plan 和参考轨迹均未保存。
- 三阶段 Jacobian 的有效秩和条件数必须结合机器可读报告解释；病态局部映射会放大 task error 并触发 residual clipping。
- 未训练 PPO、未修改 reward/O0、未生成 teacher 或 reset dataset。
- 结论仅覆盖 bank `1155`、300 ms 和当前扰动尺度。

## 产物

- 机器可读报告：`{OUTPUT_JSON}`
- 可视化：`{OUTPUT_PNG}`
- 原始结果：`{RAW_PATH}`
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_JSON), "map": str(OUTPUT_PNG), "status": raw["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
