from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = (
    ROOT
    / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
)
INPUTS = {
    "Baseline\n300 ms, all, +/-0.18": REPORT_DIR
    / "O1_11_baseline_all_300ms_3states.json",
    "TIM cooperation\n300 ms, +/-0.12": REPORT_DIR
    / "O1_11_TIM_cooperative_012rad_300ms_3states.json",
    "Horizon expansion\n400 ms, all, +/-0.18": REPORT_DIR
    / "O1_11_horizon_400ms_all_3states.json",
}
OUTPUT_JSON = REPORT_DIR / "O1_11_Handoff_Boundary_Expansion.json"
OUTPUT_MD = REPORT_DIR / "O1_11_Handoff_Boundary_Expansion_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_11_Handoff_Feasibility_Boundary_Map.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_transition(record: dict) -> bool:
    unsafe = any(
        record[key]
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
    return (
        record["tim_gait"]
        and record["min_height_delta_mm"] >= -5.0
        and not unsafe
    )


def main() -> None:
    reports = {label: json.loads(path.read_text()) for label, path in INPUTS.items()}
    bank_hashes = {report["bank_sha256"] for report in reports.values()}
    bank_ids = reports[next(iter(reports))]["source_bank_ids"]
    if len(bank_hashes) != 1:
        raise RuntimeError("O1.11 inputs do not use the same frozen O0 bank")
    if any(report["source_bank_ids"] != bank_ids for report in reports.values()):
        raise RuntimeError("O1.11 inputs do not use the same P3 bank IDs")

    conditions = []
    by_condition = {}
    for label, report in reports.items():
        records = {row["bank_id"]: row for row in report["selected_state_results"]}
        strict = sum(row["feasible"] for row in records.values())
        transitions = sum(row["tim_gait"] for row in records.values())
        safe_transitions = sum(_safe_transition(row) for row in records.values())
        item = {
            "label": label.replace("\n", " "),
            "source": str(INPUTS[label].relative_to(ROOT)),
            "source_sha256": _sha256(INPUTS[label]),
            "horizon_ms": report["horizons"][0]["horizon_ms"],
            "action_mode": report["action_mode"],
            "active_action_dimensions": report["active_action_dimensions"],
            "max_offset_rad": report["max_offset_rad"],
            "population": report["population"],
            "iterations": report["iterations"],
            "strict_feasible_count": strict,
            "TIM_transition_count": transitions,
            "safe_TIM_transition_count": safe_transitions,
            "failure_taxonomy": report["failure_taxonomy"],
            "states": [records[bank_id] for bank_id in bank_ids],
        }
        conditions.append(item)
        by_condition[label] = records

    horizon_records = by_condition["Horizon expansion\n400 ms, all, +/-0.18"]
    horizon_transition_ids = [
        bank_id for bank_id in bank_ids if horizon_records[bank_id]["tim_gait"]
    ]
    report = {
        "status": "PASS_DIAGNOSTIC_STRICT_SKILL_FAIL",
        "phase": "O1.11 Handoff Boundary Expansion",
        "artifact_role": "feasibility_boundary_summary_only_not_teacher_or_training_data",
        "bank_sha256": next(iter(bank_hashes)),
        "representative_P3_bank_ids": bank_ids,
        "matched_search_budget": {"population": 64, "iterations": 10, "waypoints": 3},
        "conditions": conditions,
        "strict_feasible_total_by_condition": {
            item["label"]: item["strict_feasible_count"] for item in conditions
        },
        "boundary_diagnosis": {
            "A_action_space": {
                "finding": "NOT_SUPPORTED_BY_TIM_ONLY_0P12",
                "evidence": (
                    "TIM-only +/-0.12 rad produced 0/3 TIM transitions and 0/3 "
                    "strict handoffs; the tested cooperation space is insufficient, "
                    "but this does not prove that the original 20-D action space is too small."
                ),
            },
            "B_time_horizon": {
                "finding": "CONTRIBUTING_BUT_NOT_SUFFICIENT",
                "evidence": (
                    f"400 ms produced a TIM transition on bank {horizon_transition_ids}, "
                    "but strict feasibility remained 0/3."
                ),
            },
            "C_P3_canonical_entry": {
                "finding": "ENTRY_DEPENDENCE_OBSERVED",
                "evidence": (
                    "Only bank 996 reached a safe TIM transition at 400 ms; banks 817 "
                    "and 948 remained no-transition entries under the matched budget."
                ),
            },
        },
        "closest_boundary": {
            "bank_id": 996,
            "condition": "Horizon expansion 400 ms, all, +/-0.18",
            "strict_feasible": horizon_records[996]["feasible"],
            "TIM_transition": horizon_records[996]["tim_gait"],
            "height_loss_mm": -horizon_records[996]["min_height_delta_mm"],
            "axis_alignment": horizon_records[996]["axis_alignment"],
            "required_axis_alignment": 0.5,
            "handoff_margin": horizon_records[996]["handoff_margin"],
            "primary_failure": horizon_records[996]["primary_failure"],
        },
        "conclusion": (
            "No strict handoff was found within the O1.11 boundary. Longer horizon "
            "changes reachability on the favorable P3 entry, while TIM-only +/-0.12 "
            "does not. The evidence points to a coupled horizon-entry boundary, with "
            "axis alignment remaining the nearest strict gate; it does not establish "
            "absolute dynamics unreachability."
        ),
        "trajectory_exported": False,
        "teacher_generated": False,
        "O0_modified": False,
        "PPO_trained": False,
        "reward_modified": False,
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    labels = list(INPUTS)
    categories = np.zeros((len(bank_ids), len(labels)), dtype=float)
    annotations = []
    for row_index, bank_id in enumerate(bank_ids):
        annotation_row = []
        for col_index, label in enumerate(labels):
            record = by_condition[label][bank_id]
            categories[row_index, col_index] = (
                2.0 if record["feasible"] else 1.0 if _safe_transition(record) else 0.0
            )
            annotation_row.append(
                f"{'T' if record['tim_gait'] else '-'}\n"
                f"z {record['min_height_delta_mm']:.2f} mm\n"
                f"axis {record['axis_alignment']:.2f}"
            )
        annotations.append(annotation_row)

    fig, (ax_map, ax_metrics) = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.8),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
        constrained_layout=True,
    )
    from matplotlib.colors import ListedColormap

    ax_map.imshow(
        categories,
        cmap=ListedColormap(["#d95f59", "#e7b34c", "#4f9d69"]),
        vmin=0,
        vmax=2,
        aspect="equal",
    )
    ax_map.set_xticks(
        range(len(labels)),
        [
            "Baseline\n300 ms / 20D\nbound 0.18",
            "TIM-only\n300 ms / 12D\nbound 0.12",
            "Long horizon\n400 ms / 20D\nbound 0.18",
        ],
    )
    ax_map.set_yticks(range(len(bank_ids)), [f"P3 bank {value}" for value in bank_ids])
    ax_map.set_title("Strict handoff feasibility boundary")
    for i in range(len(bank_ids)):
        for j in range(len(labels)):
            ax_map.text(j, i, annotations[i][j], ha="center", va="center", fontsize=9)
    from matplotlib.patches import Patch

    ax_map.legend(
        handles=[
            Patch(color="#d95f59", label="No safe TIM transition"),
            Patch(color="#e7b34c", label="Safe transition, strict gate missed"),
            Patch(color="#4f9d69", label="Strict feasible"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=1,
        frameon=False,
        fontsize=9,
    )

    x = np.arange(len(labels))
    strict = [item["strict_feasible_count"] for item in conditions]
    transitions = [item["TIM_transition_count"] for item in conditions]
    ax_metrics.bar(x - 0.18, transitions, width=0.36, color="#e7b34c", label="TIM transition")
    ax_metrics.bar(x + 0.18, strict, width=0.36, color="#4f9d69", label="Strict handoff")
    ax_metrics.set_xticks(x, ["Baseline", "TIM +/-0.12", "400 ms"])
    ax_metrics.set_ylim(0, 3.3)
    ax_metrics.set_ylabel("States / 3")
    ax_metrics.set_title("Boundary counts")
    ax_metrics.legend(frameon=False)
    ax_metrics.grid(axis="y", alpha=0.25)
    fig.suptitle("O1.11 Handoff Boundary Expansion", fontsize=15)
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    closest = report["closest_boundary"]
    markdown = f"""# O1.11 Handoff Boundary Expansion 验收报告

## 结论

`PASS (diagnostic) / STRICT HANDOFF FAIL`

三个条件的严格 handoff 均为 `0/3`。当前不能宣称 P3 handoff 已物理可达，也不能宣称绝对不可达。

| 条件 | Strict | TIM transition | 主要结果 |
|---|---:|---:|---|
| baseline: 300 ms, 20D, +/-0.18 rad | {conditions[0]['strict_feasible_count']}/3 | {conditions[0]['TIM_transition_count']}/3 | 三个入口均未换指 |
| TIM cooperative: 300 ms, 12D, +/-0.12 rad | {conditions[1]['strict_feasible_count']}/3 | {conditions[1]['TIM_transition_count']}/3 | 小幅 TIM 协同未扩张可行域 |
| horizon: 400 ms, 20D, +/-0.18 rad | {conditions[2]['strict_feasible_count']}/3 | {conditions[2]['TIM_transition_count']}/3 | bank 996 出现安全 transition，但轴对齐失败 |

## 边界判断

- **A. Action space 不足：未被支持。** TIM-only `+/-0.12 rad` 没有产生 transition，只能说这个小幅协同空间不足，不能证明原 20D action space 太小。
- **B. Time horizon 不足：有贡献，但不是唯一原因。** 400 ms 使 bank 996 出现 TIM release/recontact，但 strict 仍为 0。
- **C. P3 canonical 入口不足：存在明显 entry dependence。** 只有 bank 996 接近严格边界，817/948 在同预算下均未 transition。

## 最接近边界

- bank: `{closest['bank_id']}`
- 条件: 400 ms, all joints, `+/-0.18 rad`
- TIM transition: `{closest['TIM_transition']}`
- 最大高度损失: `{closest['height_loss_mm']:.3f} mm`
- 轴对齐: `{closest['axis_alignment']:.3f}`，门槛 `0.5`
- handoff margin: `{closest['handoff_margin']:.3f}`
- 最终失败类型: `{closest['primary_failure']}`

## 约束审计

- 未训练 PPO
- 未修改 reward
- 未生成或导出 teacher/action plan
- 未修改 O0 bank
- 三组均使用 bank `817/948/996`、population `64`、iterations `10`、3 waypoints

边界图：`{OUTPUT_PNG.name}`
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps({"json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD), "figure": str(OUTPUT_PNG)}))


if __name__ == "__main__":
    main()
