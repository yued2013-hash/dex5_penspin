from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
RAW_PATH = REPORT_DIR / "O1_17_Closed_Loop_Basin_Raw.json"
OUTPUT_JSON = REPORT_DIR / "O1_17_Closed_Loop_Basin_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_17_Closed_Loop_Basin_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_17_Closed_Loop_Basin_Map.png"


def _scenario_map(mode: dict) -> dict[str, dict]:
    return {row["name"]: row for row in mode["scenarios"]}


def main() -> None:
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    if raw["bank_sha256_before"] != raw["bank_sha256_after"]:
        raise RuntimeError("O0 hash changed during O1.17")
    baseline, closed_loop = raw["held_out_evaluation"]
    baseline_rows = _scenario_map(baseline)
    closed_rows = _scenario_map(closed_loop)
    scenario_names = ("nominal", "joint_L1", "joint_L2", "joint_L3")

    comparison = []
    for name in scenario_names:
        before = baseline_rows[name]
        after = closed_rows[name]
        comparison.append(
            {
                "scenario": name,
                "open_loop_rate": before["recovered_strict_rate"],
                "closed_loop_rate": after["recovered_strict_rate"],
                "absolute_gain": after["recovered_strict_rate"] - before["recovered_strict_rate"],
                "rate_ratio": (
                    after["recovered_strict_rate"] / before["recovered_strict_rate"]
                    if before["recovered_strict_rate"] > 0.0
                    else None
                ),
                "open_loop_unsafe_rate": before["unsafe_rate"],
                "closed_loop_unsafe_rate": after["unsafe_rate"],
                "open_loop_taxonomy": before["failure_taxonomy"],
                "closed_loop_taxonomy": after["failure_taxonomy"],
            }
        )

    report = {
        "status": raw["status"],
        "phase": raw["phase"],
        "artifact_role": raw["artifact_role"],
        "bank_id": raw["bank_id"],
        "bank_sha256": raw["bank_sha256_after"],
        "selected_controller": raw["selected_controller"],
        "selected_controller_config": next(
            row["controller"]
            for row in raw["calibration"]
            if row["mode"] == raw["selected_controller"]
        ),
        "calibration": raw["calibration"],
        "perturbation_protocol": raw["perturbation_protocol"],
        "held_out_comparison": comparison,
        "widening_gate": raw["widening_gate"],
        "widening_gate_pass": raw["widening_gate_pass"],
        "historical_o1_16_joint_L1_rate": raw["historical_o1_16_joint_L1_rate"],
        "interpretation": (
            "The selected deterministic high-level feedback controller was chosen on "
            "a calibration perturbation seed and evaluated on a disjoint held-out seed. "
            "It tracks an in-memory nominal q/dq trajectory and adapts gain from total "
            "support force and pen height, without prescribing a finger contact pattern. "
            "Passing all gates is evidence of widening under the tested perturbations, "
            "not proof of a global policy attractor or sim-to-real robustness."
        ),
        "constraints": raw["constraints"],
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    x = np.arange(len(scenario_names))
    open_rates = np.asarray([baseline_rows[name]["recovered_strict_rate"] for name in scenario_names])
    closed_rates = np.asarray([closed_rows[name]["recovered_strict_rate"] for name in scenario_names])
    open_unsafe = np.asarray([baseline_rows[name]["unsafe_rate"] for name in scenario_names])
    closed_unsafe = np.asarray([closed_rows[name]["unsafe_rate"] for name in scenario_names])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, open_rates, width, label="open loop", color="#8c96c6")
    axes[0].bar(x + width / 2, closed_rates, width, label="closed loop", color="#2ca25f")
    axes[0].axhline(0.30, color="#333333", linestyle=":", label="L1 widening gate")
    axes[0].axhline(0.10, color="#777777", linestyle=":", label="L2 gate")
    axes[0].set_xticks(x, scenario_names)
    axes[0].set_ylim(0.0, max(0.35, float(max(open_rates.max(), closed_rates.max())) * 1.15))
    axes[0].set_ylabel("Recovered-strict rate")
    axes[0].set_title("Held-out basin comparison")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x - width / 2, open_unsafe, width, label="open loop", color="#8c96c6")
    axes[1].bar(x + width / 2, closed_unsafe, width, label="closed loop", color="#de2d26")
    axes[1].set_xticks(x, scenario_names)
    axes[1].set_ylabel("Unsafe rate")
    axes[1].set_title("Safety comparison")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    gate_names = list(raw["widening_gate"])
    gate_values = [1 if raw["widening_gate"][name] else 0 for name in gate_names]
    gate_labels = [name.replace("joint_L1_", "L1 ").replace("joint_L2_", "L2 ").replace("_", " ") for name in gate_names]
    axes[2].scatter(
        gate_values,
        np.arange(len(gate_values)),
        marker="X",
        s=150,
        color=["#2ca25f" if value else "#de2d26" for value in gate_values],
    )
    axes[2].set_yticks(np.arange(len(gate_values)), gate_labels)
    axes[2].set_xlim(-0.08, 1.08)
    axes[2].set_xticks((0, 1), ("FAIL", "PASS"))
    axes[2].set_title("Pre-registered widening gates")
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    rows = "\n".join(
        f"| `{row['scenario']}` | {row['open_loop_rate']:.2%} | "
        f"{row['closed_loop_rate']:.2%} | {row['absolute_gain']:+.2%} | "
        f"{row['open_loop_unsafe_rate']:.2%} | {row['closed_loop_unsafe_rate']:.2%} |"
        for row in comparison
    )
    gates = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in raw["widening_gate"].items()
    )
    calibration_rows = "\n".join(
        f"| `{row['mode']}` | {row['scenarios'][0]['recovered_strict_rate']:.2%} | "
        f"{row['scenarios'][1]['recovered_strict_rate']:.2%} | "
        f"{row['scenarios'][0]['unsafe_rate']:.2%} | "
        f"{row['scenarios'][1]['unsafe_rate']:.2%} | {row['selection_score']:.4f} |"
        for row in raw["calibration"]
    )
    verdict = (
        "闭环反馈通过预注册 widening gate；在本次受限扰动范围内形成了更宽的局部吸引域。"
        if raw["widening_gate_pass"]
        else "闭环反馈未通过全部 widening gate；当前仍不能称为可靠技能吸引域。"
    )
    markdown = f"""# O1.17 闭环 Basin Widening Audit 验收报告

## 结论

{verdict}

选择的反馈器：`{raw['selected_controller']}`。反馈器在 calibration seed 上选择，以下正式结果来自不同的 held-out perturbation seed。

## Calibration

| Mode | Nominal success | Joint-L1 success | Nominal unsafe | Joint-L1 unsafe | Selection score |
|---|---:|---:|---:|---:|---:|
{calibration_rows}

| Scenario | Open loop | Closed loop | 增量 | Open unsafe | Closed unsafe |
|---|---:|---:|---:|---:|---:|
{rows}

## 预注册门槛

| Gate | 结果 |
|---|---|
{gates}

## 方法边界

- 反馈只跟踪内存中的 nominal `q/dq` 参考，并根据总支撑力和笔高度连续调节增益；没有规定手指接触顺序。
- calibration 与 held-out 使用不同扰动种子；open/closed 在各自集合内使用完全匹配的扰动。
- 这是 bank `1155`、300 ms、指定扰动尺度内的局部结论，不证明全局 attractor、完整 finger gait 或 sim2real 鲁棒性。
- 本轮只否定“单条 nominal `q/dq` 跟踪 + 支撑力/高度增益调节”这一反馈架构，不否定其他 object-state feedback、MPC 或学习型闭环控制。
- CEM plan 和 nominal reference 均未序列化；未训练 PPO、未修改 reward/O0、未生成 teacher/reset dataset。

## 产物

- 机器可读报告：`{OUTPUT_JSON}`
- 可视化：`{OUTPUT_PNG}`
- 原始结果：`{RAW_PATH}`
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_JSON), "map": str(OUTPUT_PNG), "status": raw["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
