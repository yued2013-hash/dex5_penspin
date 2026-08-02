from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any

from m3_2_23_short_primitive_reward import (
    PrimitiveSpec,
    ShortPrimitiveWeights,
    readiness_class,
    summary_reward_terms,
)


ROOT = Path("dex5_penspin")
REPORT_ROOT = ROOT / "reports" / "manus_mujoco_presets"
OUTPUT_DIR = REPORT_ROOT / "short_primitive_reward_m3_2_23"


def p(path: str) -> str:
    return str(REPORT_ROOT / path)


def primitive_specs() -> list[PrimitiveSpec]:
    m322 = "lifted_b12c_capture_m3_2_22"
    m320 = "optimized_b12_m3_2_20"
    m319 = "active_b12_b23_m3_2_19"
    m318 = "p1_p2_micro_rebuild_m3_2_18"
    m316 = "rebuilt_p4_p5_transitions_m3_2_16"
    m317 = "cycle_closure_eval_m3_2_17"
    return [
        PrimitiveSpec(
            label="P1_to_B12b_handoff",
            start_seed=p(f"{m322}/P1_four_finger_clamp_micro_thumb_supported_seed.json"),
            target_seed=p(f"{m322}/B12b_middle_ring_handoff_seed.json"),
            duration_s=0.36,
            target_contacts=["thumb", "middle", "ring"],
            allowed_extra_contacts=["index"],
            forbidden_contacts=["pinky"],
            reset_mode="static_keypose",
            source_summary_csv=p(f"{m320}/transition_summary.csv"),
            source_summary_label="optimized_P1_B12a_B12b",
            priority="high",
            notes="B12b is useful as a dynamic handoff, but current scripted route still brings pinky into support.",
        ),
        PrimitiveSpec(
            label="P1_to_P2_entry",
            start_seed=p(f"{m322}/P1_four_finger_clamp_micro_thumb_supported_seed.json"),
            target_seed=p(f"{m318}/P2_ring_wrap_to_front_micro_ring_guarded_seed.json"),
            duration_s=0.36,
            target_contacts=["thumb", "middle", "ring"],
            allowed_extra_contacts=["index"],
            forbidden_contacts=["pinky"],
            reset_mode="static_keypose",
            source_summary_csv=p(f"{m319}/transition_summary.csv"),
            source_summary_label="active_P1_B12_P2",
            priority="high",
            notes="Early entry needs explicit pinky suppression and ring force limiting.",
        ),
        PrimitiveSpec(
            label="P2_to_B23_P3_capture",
            start_seed=p(f"{m318}/P2_ring_wrap_to_front_micro_ring_guarded_seed.json"),
            target_seed=p(f"{m322}/P3_index_wrap_to_front_reference_seed.json"),
            duration_s=0.36,
            target_contacts=["thumb", "index", "ring"],
            allowed_extra_contacts=["middle"],
            forbidden_contacts=["pinky"],
            reset_mode="static_keypose",
            source_summary_csv=p(f"{m319}/transition_summary.csv"),
            source_summary_label="active_P2_B23_P3",
            priority="critical",
            notes="This is the cleanest available B23/P3 reset-start teacher, but final P3 is still not strict.",
        ),
        PrimitiveSpec(
            label="B12cLift_to_B23_capture",
            start_seed=p(f"{m322}/B12c_lifted_support_seed.json"),
            target_seed=p(f"{m322}/B23_reference_index_ring_capture_seed.json"),
            duration_s=0.72,
            target_contacts=["thumb", "index", "ring"],
            allowed_extra_contacts=["middle"],
            forbidden_contacts=["pinky"],
            reset_mode="dynamic_state_cache",
            source_summary_csv=p(f"{m322}/transition_summary.csv"),
            source_summary_label="selected_lifted_P1_B12a_B12b_B12cLift_B23",
            priority="critical",
            notes="Main current blocker: survives and supports, but final target contact is middle/ring instead of thumb/index/ring.",
        ),
        PrimitiveSpec(
            label="B23_to_P3_capture",
            start_seed=p(f"{m322}/B23_reference_index_ring_capture_seed.json"),
            target_seed=p(f"{m322}/P3_index_wrap_to_front_reference_seed.json"),
            duration_s=0.90,
            target_contacts=["thumb", "index", "ring"],
            allowed_extra_contacts=["middle"],
            forbidden_contacts=["pinky"],
            reset_mode="dynamic_state_cache",
            source_summary_csv=p(f"{m322}/transition_summary.csv"),
            source_summary_label="selected_lifted_P1_B12a_B12b_B12cLift_B23_P3",
            priority="critical",
            notes="Should be trained from B23-arrival state cache, not from a clean static B23 alone.",
        ),
        PrimitiveSpec(
            label="P3_to_P4_thumb_entry",
            start_seed=p(f"{m322}/P3_index_wrap_to_front_reference_seed.json"),
            target_seed=p(f"{m316}/P4_rebuilt_thumb_supported_seed.json"),
            duration_s=0.56,
            target_contacts=["thumb", "index", "middle"],
            allowed_extra_contacts=[],
            forbidden_contacts=["pinky", "ring"],
            reset_mode="static_keypose",
            source_summary_csv=p(f"{m316}/m3_2_16_evaluation.csv"),
            source_summary_label="transition_P3_B34_P4",
            priority="medium",
            notes="Mostly trainable; reward should reduce middle-only gaps.",
        ),
        PrimitiveSpec(
            label="P4_to_P5_thumb_index_capture",
            start_seed=p(f"{m316}/P4_rebuilt_thumb_supported_seed.json"),
            target_seed=p(f"{m316}/P5_rebuilt_thumb_index_capture_seed.json"),
            duration_s=0.56,
            target_contacts=["thumb", "index"],
            allowed_extra_contacts=["middle"],
            forbidden_contacts=["pinky", "ring"],
            reset_mode="static_keypose",
            source_summary_csv=p(f"{m316}/m3_2_16_evaluation.csv"),
            source_summary_label="transition_P4_B45_P5",
            priority="low",
            notes="Best current teacher segment; use for policy warm-up and sanity check.",
        ),
        PrimitiveSpec(
            label="P5_to_P1_cycle_entry",
            start_seed=p(f"{m316}/P5_rebuilt_thumb_index_capture_seed.json"),
            target_seed=p(f"{m317}/P1_cycle_entry_from_P5_seed.json"),
            duration_s=0.28,
            target_contacts=["thumb", "index", "middle"],
            allowed_extra_contacts=["ring"],
            forbidden_contacts=["pinky"],
            reset_mode="static_keypose",
            source_summary_csv=p(f"{m317}/transition_summary.csv"),
            source_summary_label="transition_P5_to_P1_cycle_entry",
            priority="medium",
            notes="Good P1-like return; strict four-finger P1 ring can be a bonus, not the first terminal requirement.",
        ),
        PrimitiveSpec(
            label="P1_cycle_to_P2_entry",
            start_seed=p(f"{m317}/P1_cycle_entry_from_P5_seed.json"),
            target_seed=p(f"{m318}/P2_ring_wrap_to_front_micro_ring_guarded_seed.json"),
            duration_s=0.28,
            target_contacts=["thumb", "middle", "ring"],
            allowed_extra_contacts=["index"],
            forbidden_contacts=["pinky"],
            reset_mode="static_keypose",
            source_summary_csv=p(f"{m317}/transition_summary.csv"),
            source_summary_label="transition_P1cycle_to_P2",
            priority="critical",
            notes="Cycle-front blocker: current route collapses into pinky/ring.",
        ),
    ]


def parse_value(value: str) -> Any:
    if value in {"", "none", "None"}:
        return value
    if value in {"True", "False"}:
        return value == "True"
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return float(value)
        except ValueError:
            return value


def load_summary_row(path: str, label: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label") != label:
                continue
            return {key: parse_value(value) for key, value in row.items()}
    raise KeyError(f"Missing row {label!r} in {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row_line(row: dict[str, Any]) -> str:
    return (
        f"| {row['primitive']} | {row['readiness']} | {row['priority']} | {row['reset_mode']} | "
        f"{row['score_total']:.1f} | {row['survival_s']:.2f}/{row['duration_s']:.2f} | "
        f"{row['target_final']:.2f} | {row['support_final']:.2f} | "
        f"{row['thumb_ratio']:.2f} | {row['index_ratio']:.2f} | {row['ring_ratio']:.2f} | "
        f"{row['pinky_ratio']:.2f} | {row['max_penetration_mm']:.2f} | {row['final_contacts']} |"
    )


def classify_readiness(score: float, spec: PrimitiveSpec, row: dict[str, Any]) -> str:
    base = readiness_class(score)
    contact_ratio = row.get("contact_ratio") or {}
    pinky_ratio = float(contact_ratio.get("pinky", 0.0))
    target_final = float(row.get("target_final_window_ratio", row.get("desired_ratio", 0.0)))
    support_final = float(row.get("support_final_window_ratio", row.get("active_contact_ratio", 0.0)))
    terminated = bool(row.get("terminated", False))

    if "pinky" in spec.forbidden_contacts and pinky_ratio >= 0.45:
        return "needs_focused_reward"
    if spec.priority == "critical" and target_final < 0.20:
        if terminated or support_final < 0.25:
            return "needs_pose_or_reset_fix"
        return "needs_focused_reward"
    if terminated:
        return "needs_pose_or_reset_fix"
    return base


def reward_design_doc(score_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# M3.2.23 short primitive reward 设计",
        "",
        "本轮把目标从“完整转笔一次性训练”收窄为 `keypose-guided short primitive RL`。每条边单独训练，成功后再做 stitching 或 phase-conditioned full-cycle fine-tune。",
        "",
        "## Reward 结构",
        "",
        "单步 reward 分成四组：",
        "",
        "1. 生存与物理可信：alive、height guard、center drift、penetration、joint limit、action/action smoothness。",
        "2. 接触角色：support contact、desired contact、terminal contact、thumb support、index capture。",
        "3. 反作弊项：pinky penalty、ring overload penalty。ring 可以防滚，但 force share 超过阈值会扣分。",
        "4. 运动目标：rotation progress、target axis、target center。早期课程弱化 terminal pose，后期逐步加大 terminal contact/axis 权重。",
        "",
        "核心公式在 `dex5_penspin/sim/m3_2_23_short_primitive_reward.py`，其中 `step_reward_terms()` 是给 MuJoCo/Isaac 环境逐步调用的版本，`summary_reward_terms()` 是当前用 CSV 评估 scripted baseline 的离线版本。",
        "",
        "## 当前 scorecard",
        "",
        "| primitive | readiness | priority | reset | score | survival | target final | support final | thumb | index | ring | pinky | pen mm | final contacts |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines += [row_line(row) for row in score_rows]
    lines += [
        "",
        "## 训练顺序",
        "",
        "第一批 teacher/warm-up：`P4_to_P5_thumb_index_capture`、`P3_to_P4_thumb_entry`、`P5_to_P1_cycle_entry`。这些段已经有较好 scripted baseline，适合验证 RL 框架和 reward 是否工作。",
        "",
        "第二批重点攻关：`P2_to_B23_P3_capture`、`B12cLift_to_B23_capture`、`B23_to_P3_capture`。这里要显式提高 terminal `thumb/index/ring`，否则策略会继续收敛到 middle/ring 兜底。",
        "",
        "第三批闭环前段：`P1_to_P2_entry`、`P1_cycle_to_P2_entry`。主要 reward 压力是 pinky 退出、ring 不抢主承力、thumb 保持上托。",
        "",
        "## Isaac Gym / Isaac Lab 迁移要点",
        "",
        "- 每个 primitive 一个 task config，或一个 phase-conditioned task 加 `primitive_id` one-hot。",
        "- `static_keypose` 段可以从 seed 附近 domain randomization reset。",
        "- `dynamic_state_cache` 段必须从 MuJoCo/teacher rollout 的状态缓存 reset，不能直接从单独 JSON keypose reset。",
        "- contact flags/force share 在 Isaac 中用 fingertip force sensor 或 contact tensor 得到；没有可靠 contact tensor 前，不要移除 MuJoCo 显微镜。",
        "- reward curriculum：前 20% 训练 survival/support，中间加 rotation/axis，后 40% 加 terminal contact 和 pinky/ring overload 压力。",
        "",
        "## 下一步执行",
        "",
        "1. 先把 `P4_to_P5` 和 `P3_to_P4` 接到现有 TD3/MuJoCo wrapper，确认 learning loop 能超过 scripted 或至少复现 scripted。",
        "2. 导出 `B12cLift_to_B23` 的 dynamic state cache，作为 Isaac/MuJoCo 短段 reset 分布。",
        "3. 对 `B12cLift_to_B23` 单独训练，评价指标只看 final window 的 `thumb/index/ring`，不要被 survival-only 分数骗过去。",
    ]
    return "\n".join(lines) + "\n"


def isaac_sketch() -> str:
    return '''"""Vectorized reward sketch for Isaac Gym/Lab migration.

This is not imported by current MuJoCo scripts. It documents the tensor shape
expected when porting M3.2.23 reward terms to an Isaac task.
"""

import torch


def compute_short_primitive_reward(
    contact_flags,      # [N, 5] thumb,index,middle,ring,pinky in {0,1}
    force_share,        # [N, 5]
    obj_pos,            # [N, 3]
    init_pos,           # [N, 3]
    target_pos,         # [N, 3]
    obj_axis,           # [N, 3]
    target_axis,        # [N, 3]
    phase_delta,        # [N]
    actions,            # [N, A]
    prev_actions,       # [N, A]
    penetration,        # [N]
    progress_buf,       # [N]
    max_steps,          # int
    target_mask,        # [N, 5]
    allowed_extra_mask, # [N, 5]
    phase_direction,    # [N]
):
    tau = torch.clamp(progress_buf.float() / float(max_steps), 0.0, 1.0)
    terminal_gate = torch.clamp((tau - 0.55) / 0.45, 0.0, 1.0)
    terminal_gate = terminal_gate * terminal_gate * (3.0 - 2.0 * terminal_gate)

    desired_ratio = (contact_flags * target_mask).sum(-1) / torch.clamp(target_mask.sum(-1), min=1.0)
    all_target = ((contact_flags * target_mask) >= target_mask).all(-1).float()
    core_support = contact_flags[:, :4].sum(-1).clamp(max=2.0) / 2.0

    thumb = contact_flags[:, 0]
    index = contact_flags[:, 1]
    ring = contact_flags[:, 3]
    pinky = contact_flags[:, 4]
    ring_allowed = torch.clamp(target_mask[:, 3] + allowed_extra_mask[:, 3], 0.0, 1.0)
    ring_overload = torch.clamp(force_share[:, 3] - 0.35, min=0.0) + (1.0 - ring_allowed) * 0.35 * ring

    axis_cos = (obj_axis * target_axis).sum(-1) / (obj_axis.norm(dim=-1) * target_axis.norm(dim=-1) + 1e-6)
    axis_error = torch.acos(axis_cos.clamp(-1.0, 1.0))
    center_dist = (obj_pos - target_pos).norm(dim=-1)
    height_drop = torch.clamp(init_pos[:, 2] - obj_pos[:, 2], min=0.0)
    progress = torch.clamp(phase_direction * phase_delta, min=0.0, max=0.08) / 0.08

    reward = 0.10
    reward = reward + 0.85 * core_support
    reward = reward + 1.15 * desired_ratio
    reward = reward + 2.75 * terminal_gate * all_target
    reward = reward + 0.65 * thumb * torch.clamp(force_share[:, 0] / 0.30, max=1.0)
    reward = reward + 0.75 * terminal_gate * index
    reward = reward + 0.15 * ring_allowed * ring * (1.0 - torch.clamp(ring_overload / 0.45, max=1.0))
    reward = reward + 0.55 * progress
    reward = reward - 0.50 * torch.clamp(axis_error / 0.65, max=1.0)
    reward = reward - 0.45 * torch.clamp(center_dist / 0.030, max=1.0)
    reward = reward - 0.80 * torch.clamp(height_drop / 0.035, max=1.0)
    reward = reward - 1.25 * torch.clamp(ring_overload / 0.45, max=1.0)
    reward = reward - 0.90 * pinky
    reward = reward - 1.40 * torch.clamp((penetration - 0.006) / 0.006, min=0.0, max=1.0)
    reward = reward - 0.06 * actions.norm(dim=-1)
    reward = reward - 0.08 * (actions - prev_actions).norm(dim=-1)
    return reward
'''


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = ShortPrimitiveWeights()
    specs = primitive_specs()
    spec_data = {
        "stage": "M3.2.23 short primitive reward plan",
        "weights": weights.__dict__,
        "primitives": [spec.to_json() for spec in specs],
    }
    (OUTPUT_DIR / "primitive_reward_spec.json").write_text(
        json.dumps(spec_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "isaac_reward_vectorized_sketch.py").write_text(isaac_sketch(), encoding="utf-8")

    score_rows: list[dict[str, Any]] = []
    for spec in specs:
        source_row = load_summary_row(spec.source_summary_csv, spec.source_summary_label)
        terms = summary_reward_terms(source_row, spec, weights)
        contact_ratio = source_row.get("contact_ratio") or {}
        row = {
            "primitive": spec.label,
            "priority": spec.priority,
            "reset_mode": spec.reset_mode,
            "source_label": spec.source_summary_label,
            "duration_s": float(spec.duration_s),
            "survival_s": float(source_row.get("survival_s", 0.0)),
            "class": source_row.get("class", ""),
            "target_final": float(source_row.get("target_final_window_ratio", source_row.get("desired_ratio", 0.0))),
            "support_final": float(source_row.get("support_final_window_ratio", source_row.get("active_contact_ratio", 0.0))),
            "thumb_ratio": float(contact_ratio.get("thumb", 0.0)),
            "index_ratio": float(contact_ratio.get("index", 0.0)),
            "middle_ratio": float(contact_ratio.get("middle", 0.0)),
            "ring_ratio": float(contact_ratio.get("ring", 0.0)),
            "pinky_ratio": float(contact_ratio.get("pinky", 0.0)),
            "max_penetration_mm": float(source_row.get("max_penetration_mm", 0.0)),
            "final_contacts": ",".join(source_row.get("final_contacts", [])) if isinstance(source_row.get("final_contacts"), list) else source_row.get("final_contacts", ""),
            "readiness": classify_readiness(terms["total"], spec, source_row),
            **{f"score_{key}": value for key, value in terms.items()},
        }
        row["score_total"] = row.pop("score_total")
        score_rows.append(row)

    write_csv(OUTPUT_DIR / "primitive_reward_scorecard.csv", score_rows)
    (OUTPUT_DIR / "REWARD_DESIGN.zh.md").write_text(reward_design_doc(score_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "primitives": len(score_rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
