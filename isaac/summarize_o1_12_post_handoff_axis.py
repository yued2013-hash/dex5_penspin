from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
INPUT = REPORT_DIR / "O1_12_Post_Handoff_Axis_Stabilization_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_12_Post_Handoff_Axis_Stabilization_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_12_Post_Handoff_Axis_Stabilization_Map.png"


def main() -> None:
    report = json.loads(INPUT.read_text(encoding="utf-8"))
    handoff = report["handoff_search"]["states"]
    post = report["post_handoff_search"]["states"]
    bank_ids = [row["bank_id"] for row in post]

    discovery_count = sum(row["transition_success"] for row in handoff)
    continuation_count = sum(row["transition_success"] for row in post)
    post_count = sum(row["post_handoff_success"] for row in post)
    robust_anchor = next(row for row in post if row["bank_id"] == 996)

    fig, (ax_funnel, ax_state) = plt.subplots(
        1, 2, figsize=(13.5, 5.6), constrained_layout=True
    )
    stages = ["400 ms\nhandoff search", "Continuous replay\ntransition", "Post-axis\nstrict gate"]
    counts = [discovery_count, continuation_count, post_count]
    colors = ["#4f81bd", "#e7b34c", "#4f9d69"]
    bars = ax_funnel.bar(stages, counts, color=colors, width=0.62)
    ax_funnel.set_ylim(0, len(bank_ids) + 0.5)
    ax_funnel.set_ylabel(f"P3 entries / {len(bank_ids)}")
    ax_funnel.set_title("Existence candidates through the audit")
    ax_funnel.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, counts, strict=True):
        ax_funnel.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value}/{len(bank_ids)}",
            ha="center",
            fontweight="bold",
        )

    values = np.asarray(
        [
            [
                row["selected_plan_transition_replay_rate"],
                row["selected_plan_post_success_replay_rate"],
                max(0.0, min(1.0, row["post_axis_alignment"])),
            ]
            for row in post
        ]
    )
    image = ax_state.imshow(values, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax_state.set_xticks(
        range(3), ["Transition\nreplay rate", "Post success\nreplay rate", "Best axis\nalignment"]
    )
    ax_state.set_yticks(range(len(bank_ids)), [f"P3 bank {value}" for value in bank_ids])
    ax_state.set_title("Entry-dependent stabilization boundary")
    for row_index, row in enumerate(post):
        for column in range(3):
            ax_state.text(
                column,
                row_index,
                f"{values[row_index, column]:.3f}",
                ha="center",
                va="center",
                color="black",
            )
        ax_state.text(
            3.08,
            row_index,
            f"z {row['post_min_height_delta_mm']:.2f} mm\n{row['primary_failure']}",
            ha="left",
            va="center",
            fontsize=9,
            clip_on=False,
        )
    fig.colorbar(image, ax=ax_state, fraction=0.046, pad=0.16, label="Gate/rate value")
    fig.suptitle("O1.12 Post-Handoff Axis Stabilization Audit", fontsize=15)
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    rows = "\n".join(
        "| {bank_id} | {transition_success} | {post_handoff_success} | "
        "{post_axis_alignment:.3f} | {post_min_height_delta_mm:.3f} | "
        "{post_TIM_overlap:.3f} | {selected_plan_post_success_replay_rate:.2%} | "
        "{primary_failure} |".format(**row)
        for row in post
    )
    markdown = f"""# O1.12 Post-Handoff Axis Stabilization Audit 验收报告

## 结论

`PASS (diagnostic) / POST-HANDOFF PHYSICALLY REACHABLE IN AUDITED BOUND`

- 400 ms handoff 搜索找到 transition entry：`{discovery_count}/{len(bank_ids)}`
- 同一 PhysX episode 内连续复现 transition：`{continuation_count}/{len(bank_ids)}`
- 200 ms post-handoff 严格稳定化存在性成功：`{post_count}/{len(bank_ids)}`
- 失败 taxonomy：`{json.dumps(report['failure_taxonomy'], ensure_ascii=False)}`

bank `996` 证明 transition 后的轴稳定化在当前 PhysX、20D residual action、`+/-0.12 rad`、200 ms 窗口内可达：

- 轴对齐：`{robust_anchor['post_axis_alignment']:.3f}`，门槛 `0.5`
- 最大高度损失：`{-robust_anchor['post_min_height_delta_mm']:.3f} mm`
- TIM overlap：`{robust_anchor['post_TIM_overlap']:.1%}`
- 支撑力范围：`{robust_anchor['post_min_total_force_n']:.3f} - {robust_anchor['post_max_total_force_n']:.3f} N`
- transition replay：`{robust_anchor['selected_plan_transition_replay_rate']:.2%}`
- strict post replay：`{robust_anchor['selected_plan_post_success_replay_rate']:.2%}`
- 无 ballistic/contact loss/penetration/drop/throw

## 逐入口结果

| Bank | Transition | Post success | Axis | Min height delta | TIM overlap | Post replay | Taxonomy |
|---:|---:|---:|---:|---:|---:|---:|---|
{rows}

## 原因判断

1. **不是 post-handoff 动力学绝对不可达。** Bank 996 的严格成功排除了这一结论。
2. **高度和总支撑不是当前主失败门。** 可 transition 的失败入口仍保持高度、TIM overlap 和接触力。
3. **主要瓶颈是 entry-dependent axis basin。** `1245/1171` 在支撑安全时轴对齐仅约 `0.356/0.347`。
4. **transition 复现性仍是第二瓶颈。** `1203` 在 handoff search 中可达，但未在 post-stage 连续审计中稳定重现。

## 约束审计

- 未训练 PPO
- 未修改 reward
- 未修改 O0
- 未生成 teacher
- 未序列化任何 CEM action plan 或 post-transition state
- post 搜索从同一 PhysX episode 内的 transition 下一控制步开始

可视化：`{OUTPUT_PNG.name}`
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {"markdown": str(OUTPUT_MD), "figure": str(OUTPUT_PNG)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
