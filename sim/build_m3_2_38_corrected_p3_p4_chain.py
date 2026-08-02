from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

import build_m3_2_33_narrow_021_five_step_chain as m333
import build_m3_2_34_continuous_p1_p2_p3 as m334
import export_m3_2_23_dynamic_state_cache as m323
import optimize_m3_2_36_p45_mpc as mpc
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_35_early_p51_closure as early
import search_m3_2_36_semantic_p45b as semantic
import search_m3_2_37_corrected_tail as corrected
from evaluate_m3_2_15_short_primitives import FINGERS

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


FINGER_GROUPS = (
    np.arange(0, 4),
    np.arange(4, 8),
    np.arange(8, 12),
    np.arange(16, 20),
)

# This low-drift target starts at the real B23 exit. It preserves the
# index/middle/thumb drive and only prepositions ring as a light guard.
P34_TARGET = np.asarray(
    [
        0.0002896619100418926,
        0.6710735391085142,
        0.5615082414280266,
        0.5615082414280266,
        0.0516119974291017,
        0.9913615752728722,
        0.7508510630826386,
        0.7508510630826386,
        0.17635806346709818,
        1.0518475205086408,
        0.2571691422922081,
        0.2571691422922081,
        0.3839699923992157,
        1.4994279146194458,
        0.6666946411132812,
        0.6666946411132812,
        0.6802999973297119,
        0.6621581482413219,
        1.0495544119329634,
        0.19511850499402578,
    ],
    dtype=np.float64,
)
P34_DELAYS = (5, 3, 1, 6)
P34_RAMPS = (12, 2, 16, 7)
P34_GAINS = (
    0.8807311004494089,
    0.4248337070870809,
    0.35464278889184553,
    0.46661258219040835,
)
P34_LIMITS = (
    0.01249274277352724,
    0.03717376999007838,
    0.0061696118604016295,
    0.015958122206780438,
)
P34_STEPS = 16

RING_GUARD_TARGET = np.asarray(
    [
        0.19893001910288147,
        1.1933714293386044,
        0.5044441314335133,
        0.5044441314335133,
    ],
    dtype=np.float64,
)
RING_GUARD_DELAY = 4
RING_GUARD_RAMP = 10
RING_GUARD_GAIN = 0.8984783561268748
RING_GUARD_LIMIT = 0.026040742961002124

GUARD_TO_MIDDLE_ACTIONS = np.asarray(
    [
        [-0.04, -0.04, 0.0301570083, 0.0301570083, -0.0090786088, -0.0227839286, 0.04, 0.04, 0.0231304931, -0.0308741177, -0.04, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0278147715, 0.0035681542, -0.0214439710, -0.04],
        [-0.0361114092, -0.04, 0.0168202458, 0.0168202458, 0.0325935436, -0.0390792303, 0.04, 0.04, 0.04, -0.0161078884, 0.0226850773, 0.0226850773, 0.0, 0.0, 0.0, 0.0, 0.0254156149, -0.0057035300, 0.0077659866, -0.0202677536],
        [0.0027191436, -0.0366808981, 0.04, 0.04, 0.0144522429, 0.04, 0.04, 0.04, 0.0196013888, 0.0213436215, 0.04, 0.04, 0.0, 0.0, 0.0, 0.0, -0.0042661853, 0.0029440622, -0.0104638569, -0.0004446748],
    ],
    dtype=np.float64,
)

CONTACT_SWITCH_ACTION = np.asarray(
    [
        0.04,
        -0.04,
        0.04,
        0.04,
        -0.018499644463911006,
        -0.023672796015729105,
        0.038972085269330894,
        0.038972085269330894,
        -0.03659073756504638,
        0.029047423777671163,
        0.04,
        0.04,
        0.0,
        0.0,
        0.0,
        0.0,
        0.03259501462607819,
        -0.017117990932711408,
        0.035078655927521143,
        -0.04,
    ],
    dtype=np.float64,
)
CONTACT_SWITCH_STEPS = 3

P4_CAPTURE_ACTION = np.asarray(
    [
        0.000693454254167919,
        0.024537279490138696,
        0.00900304994249234,
        0.00900304994249234,
        0.03242609512920027,
        0.03648270839219788,
        0.02312897499375951,
        0.02312897499375951,
        0.019287804774774965,
        0.016365223186458326,
        -0.00044627526973015086,
        -0.00044627526973015086,
        0.0,
        0.0,
        0.0,
        0.0,
        0.024392534474215792,
        0.03331207579479579,
        -0.01721412458365122,
        -0.03675718525337542,
    ],
    dtype=np.float64,
)
P4_CAPTURE_STEPS = 5
RAW_P3_TO_P4_TARGET_DEG = 53.87


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def state_seed(
    template_path: Path,
    state: dict[str, Any],
    *,
    stage: str,
    phase: float,
    name: str,
    summary: str,
    desired_contacts: list[str],
    actuator_kp: float,
    actuator_kv: float,
    action_delta_limit: float,
    tip_proxy_radius: float,
    tip_proxy_friction: float,
) -> dict[str, Any]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    contacts = str(state["contacts"])
    return {
        "stage": stage,
        "phase": phase,
        "name": name,
        "summary": summary,
        "desired_contacts": desired_contacts,
        "placement_support": desired_contacts,
        "model_path": template["model_path"],
        "hand_qpos": np.asarray(state["hand_qpos"], dtype=np.float64).tolist(),
        "target_qpos": np.asarray(state["target_qpos"], dtype=np.float64).tolist(),
        "joint_order": template["joint_order"],
        "pen_pos": np.asarray(state["pen_pos"], dtype=np.float64).tolist(),
        "pen_quat": np.asarray(state["pen_quat"], dtype=np.float64).tolist(),
        "pen_qvel": np.asarray(state["pen_qvel"], dtype=np.float64).tolist(),
        "contact_fingers_initial": [] if contacts == "none" else contacts.split(","),
        "max_penetration": float(state["max_penetration_mm"]) / 1000.0,
        "actuator_kp": actuator_kp,
        "actuator_kv": actuator_kv,
        "action_delta_limit": action_delta_limit,
        "tip_proxy_radius": tip_proxy_radius,
        "tip_proxy_friction": tip_proxy_friction,
        "thumb_force_limit": template.get("thumb_force_limit", 0.0),
        "visual_only_markers": bool(template.get("visual_only_markers", True)),
        "disable_thumb_mesh_collision": bool(
            template.get("disable_thumb_mesh_collision", True)
        ),
        "disable_distal_mesh_collision": bool(
            template.get("disable_distal_mesh_collision", True)
        ),
        "validation_status": "dynamic_short_primitive_entry",
    }


def write_td3_handoff(
    output_dir: Path,
    root: Path,
    handoff_state: dict[str, Any],
    recapture_state: dict[str, Any],
    p4_state: dict[str, Any],
    handoff_row: dict[str, Any],
    recapture_row: dict[str, Any],
    p4_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, str]:
    template_dir = root / "photo_five_phase_hand_pen_tuned_m3_2_10"
    start_seed_path = output_dir / "B34_index_thumb_recapture_seed.json"
    handoff_seed_path = output_dir / "B34_middle_handoff_seed.json"
    target_seed_path = output_dir / "dynamic_P4_entry_seed.json"
    reward_plan_path = output_dir / "corrected_recapture_to_dynamic_P4_reward.json"
    start_seed = state_seed(
        template_dir / "P3_index_wrap_to_front_hand_pen_tuned_seed.json",
        recapture_state,
        stage="M3.2.38 corrected P3-P4 TD3 handoff",
        phase=3.5,
        name="B34_index_thumb_recapture",
        summary="食指与拇指完成短时重捕获，作为动态 P4 捕获段的缓存入口。",
        desired_contacts=["thumb", "index"],
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        action_delta_limit=args.action_delta_limit,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
    )
    handoff_seed = state_seed(
        template_dir / "P3_index_wrap_to_front_hand_pen_tuned_seed.json",
        handoff_state,
        stage="M3.2.38 corrected P3-P4 TD3 handoff",
        phase=3.4,
        name="B34_middle_handoff",
        summary="食指/中指持续推滚、拇指支撑避让的动态 P4 制动入口。",
        desired_contacts=["thumb", "index", "middle"],
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        action_delta_limit=args.action_delta_limit,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
    )
    target_seed = state_seed(
        template_dir / "P4_index_middle_to_thumb_regrasp_hand_pen_tuned_seed.json",
        p4_state,
        stage="M3.2.38 corrected P3-P4 TD3 handoff",
        phase=4.0,
        name="P4_flight_entry_candidate",
        summary="拇指/食指撞击形成的 P4 flight-entry candidate；后验测试未通过短时 P4 验收。",
        desired_contacts=["thumb", "index"],
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        action_delta_limit=args.action_delta_limit,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
    )
    save_json(start_seed_path, start_seed)
    save_json(handoff_seed_path, handoff_seed)
    save_json(target_seed_path, target_seed)
    phase_delta = np.deg2rad(
        float(p4_row["planar_rotation_deg"])
        - float(recapture_row["planar_rotation_deg"])
    )
    reward_plan = {
        "stage": "M3.2.38 corrected recapture to dynamic P4 TD3 handoff",
        "weights": {
            "alive": 0.10,
            "support_contact": 1.00,
            "desired_contact": 1.50,
            "terminal_contact": 3.50,
            "thumb_support": 0.95,
            "index_capture": 1.05,
            "rotation_progress": 0.85,
            "target_axis": 0.65,
            "target_center": 0.70,
            "height_guard": 1.20,
            "ring_guard_contact": 0.05,
            "ring_overload": 1.60,
            "pinky": 1.30,
            "penetration": 2.00,
            "action_l2": 0.04,
            "action_smoothness": 0.06,
            "joint_limit": 0.50,
            "failure": 6.00,
            "middle_proximity": 1.10,
            "terminal_linear_speed": 2.40,
            "terminal_angular_speed": 1.20,
        },
        "primitives": [
            {
                "label": "corrected_recapture_to_dynamic_P4",
                "start_seed": str(start_seed_path),
                "target_seed": str(target_seed_path),
                "duration_s": P4_CAPTURE_STEPS / 100.0,
                "target_contacts": ["thumb", "index"],
                "support_contacts": ["thumb", "index", "middle", "ring"],
                "allowed_extra_contacts": ["middle", "ring"],
                "forbidden_contacts": ["pinky"],
                "reset_mode": "dynamic_state_cache",
                "source_summary_csv": str(
                    output_dir / "corrected_P3_to_dynamic_P4_trace.csv"
                ),
                "source_summary_label": "B34 index-thumb recapture",
                "phase_direction": 1.0,
                "target_phase_delta": float(phase_delta),
                "priority": "critical",
                "notes": (
                    "Five-step dynamic capture from the corrected index/thumb recapture. "
                    "Thumb/index are terminal contacts; middle/ring may only assist transiently."
                ),
            }
            ,
            {
                "label": "corrected_handoff_to_braked_P4",
                "start_seed": str(handoff_seed_path),
                "target_seed": str(target_seed_path),
                "duration_s": (CONTACT_SWITCH_STEPS + P4_CAPTURE_STEPS) / 100.0,
                "target_contacts": ["thumb", "index"],
                "support_contacts": ["thumb", "index", "middle", "ring"],
                "allowed_extra_contacts": ["middle", "ring"],
                "forbidden_contacts": ["pinky"],
                "reset_mode": "dynamic_state_cache",
                "source_summary_csv": str(
                    output_dir / "corrected_P3_to_dynamic_P4_trace.csv"
                ),
                "source_summary_label": "B34 middle handoff",
                "phase_direction": 1.0,
                "target_phase_delta": float(
                    np.deg2rad(
                        float(p4_row["planar_rotation_deg"])
                        - float(handoff_row["planar_rotation_deg"])
                    )
                ),
                "priority": "critical",
                "notes": (
                    "Overlap curriculum from the middle handoff. Terminal thumb/index contact "
                    "must be braked, while middle remains close enough for the next recapture leg."
                ),
            },
        ],
    }
    save_json(reward_plan_path, reward_plan)
    return {
        "cache": str(output_dir / "B34_index_thumb_recapture.npz"),
        "brake_cache": str(output_dir / "B34_middle_handoff.npz"),
        "brake_start_seed": str(handoff_seed_path),
        "start_seed": str(start_seed_path),
        "target_seed": str(target_seed_path),
        "reward_plan": str(reward_plan_path),
        "teacher_actions": str(output_dir / "selected_actions.json"),
        "teacher_action_key": "p4_capture_action",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_true_p3(
    env: Any,
    args: argparse.Namespace,
    root: Path,
) -> dict[str, Any]:
    route = m323.build_route(root / "photo_five_phase_hand_pen_tuned_m3_2_10")
    initial = m334.apply_p1_safety_open(
        route,
        {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0},
    )[0]
    cache = np.load(
        root
        / "short_primitive_reward_m3_2_23/dynamic_state_cache/earlier_pre_ring_for_B12b_to_B23.npz"
    )
    cache_index = m333.m332.cache_index(cache, args.source_t)
    references = tail.load_references(root / "p2_bottom_cradle_b23_thumb_pass_m3_2_31")
    prelude_summary, prelude_trace = m334.prelude_rollout(
        env,
        route,
        initial,
        m334.source_controller_args(args),
        args,
        render=True,
    )
    b12_summary, b12_trace = m333.b12_rollout(
        env,
        cache,
        cache_index,
        m333.Perturbation("continuous_nominal"),
        m334.SELECTED_B12,
        args,
        duration_s=args.b12_duration,
        render=True,
        reset_state=False,
    )
    b23_summary, b23_trace = m333.b23_rollout(
        env,
        b12_trace["snapshots"][-1],
        references,
        early.PREPARED_P3,
        args,
        label="m3_2_38_true_P3",
        render=True,
        reset_state=False,
    )
    return {
        "prelude_summary": prelude_summary,
        "prelude_trace": prelude_trace,
        "b12_summary": b12_summary,
        "b12_trace": b12_trace,
        "b23_summary": b23_summary,
        "b23_trace": b23_trace,
    }


def row_label(row: dict[str, Any], stage: str) -> str:
    return (
        f"{stage} | {row['planar_rotation_deg']:+.1f} deg | "
        f"{row['contacts']} | T/I/M/R="
        f"{row['thumb_share']:.2f}/{row['index_share']:.2f}/"
        f"{row['middle_share']:.2f}/{row['ring_share']:.2f}"
    )


def no_force_gap(rows: list[dict[str, Any]]) -> int:
    best = 0
    current = 0
    for row in rows:
        supported = sum(float(row[f"{finger}_share"]) for finger in FINGERS) > 1e-6
        current = 0 if supported else current + 1
        best = max(best, current)
    return best


def save_trace_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    t_ms = np.asarray([row["t_from_p3_s"] * 1000.0 for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t_ms, [row["planar_rotation_deg"] for row in rows], label="phase")
    axes[0].axhline(
        RAW_P3_TO_P4_TARGET_DEG,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="raw P4 phase",
    )
    axes[0].set_ylabel("rotation (deg)")
    axes[0].legend()
    for key in ("dx_mm", "dy_mm", "dz_mm"):
        axes[1].plot(t_ms, [row[key] for row in rows], label=key[:2])
    axes[1].set_ylabel("pen delta (mm)")
    axes[1].legend(ncol=3)
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        axes[2].plot(t_ms, [row[f"{finger}_share"] for row in rows], label=finger)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel("normal-force share")
    axes[2].legend(ncol=5)
    axes[3].plot(t_ms, [row["pen_linear_speed"] for row in rows], label="linear m/s")
    axes[3].plot(t_ms, [row["pen_angular_speed"] for row in rows], label="angular rad/s")
    axes[3].plot(t_ms, [row["max_penetration_mm"] for row in rows], label="penetration mm")
    axes[3].set_ylabel("speed / penetration")
    axes[3].set_xlabel("time from corrected P3 (ms)")
    axes[3].legend(ncol=3)
    for axis in axes:
        for boundary in (160.0, 190.0, 220.0):
            axis.axvline(boundary, color="#6b7280", linestyle="--", linewidth=0.8)
    fig.suptitle("M3.2.38 corrected P3 -> P4 flight-entry candidate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_visual_summary(output_dir: Path, report: dict[str, Any]) -> None:
    p3 = report["key_states"]["P3_drive_exit_candidate"]
    guard = report["key_states"]["B34_ring_guard"]
    handoff = report["key_states"]["B34_middle_handoff"]
    recapture = report["key_states"]["B34_index_thumb_recapture"]
    p4 = report["key_states"]["P4_flight_entry_candidate"]
    status = report["acceptance"]
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M3.2.38 corrected P3 P4</title>
<style>
body {{ margin:0; background:#f3f4f6; color:#202124; font-family:Arial,"Noto Sans CJK SC",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 8px; font-size:24px; }} h2 {{ margin-top:24px; font-size:18px; }}
p {{ line-height:1.6; }} .status {{ border-top:1px solid #b8bec7; border-bottom:1px solid #b8bec7; padding:12px 0; }}
.warn {{ color:#8a3b12; font-weight:700; }}
table {{ width:100%; border-collapse:collapse; background:white; margin:16px 0; }}
th,td {{ padding:9px 10px; border:1px solid #d5d9df; text-align:left; font-size:14px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
figure {{ margin:0; background:white; border:1px solid #d5d9df; border-radius:6px; overflow:hidden; }}
img {{ display:block; width:100%; height:auto; }} figcaption {{ padding:10px 12px; font-size:14px; line-height:1.5; }}
@media(max-width:760px) {{ main {{ padding:14px; }} .grid {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>M3.2.38：P3 及后续诊断支路</h1>
<p class="status"><span class="warn">旧 M3.2.35 的 P3/P4/P5 标签已作废；当前 P3 也不再标作 canonical keyframe 已通过。</span> 回放保留同一 MuJoCo env、无 reset 的 P1→P2 物理前缀，并从接触语义正确但位姿尚未贴合 Manus 的 P3 drive-exit candidate 向后诊断。末端 thumb/index 撞击帧在 20–30ms 内失去接触，只能作为 RL teacher/reset。</p>
<table><tr><th>阶段</th><th>状态</th><th>物理语义与指标</th></tr>
<tr><td>P1→P2 连续前缀</td><td>{'PASS' if status['continuous_prefix'] else 'CHECK'}</td><td>同一 env、无边界 reset，可连到 B23/P3 drive-exit 诊断状态。</td></tr>
<tr><td>P3 接触语义</td><td>{'PASS' if status['p3_contact_semantics'] else 'CHECK'}</td><td>末端 {p3['contacts']}；thumb/index/middle 承力 {p3['thumb_share']:.3f}/{p3['index_share']:.3f}/{p3['middle_share']:.3f}，不再强制 ring recapture。</td></tr>
<tr><td>Canonical P3 位姿</td><td>REJECT</td><td>当前物理手姿相对 Manus P3 的 qpos RMS 偏差为 {report['p3_pose_audit']['hand_qpos_rms_to_manus_rad']:.3f}rad，需要从 B23 重建，不得直接标作 P3 keyframe。</td></tr>
<tr><td>P3→ring 轻防滚</td><td>{'PASS' if status['ring_guard'] else 'CHECK'}</td><td>偏转 {guard['planar_rotation_deg']:.1f}°；thumb/index/ring 承力 {guard['thumb_share']:.3f}/{guard['index_share']:.3f}/{guard['ring_share']:.3f}；ring 不是主承力点。</td></tr>
<tr><td>ring→middle 接棒</td><td>{'PASS' if status['middle_handoff'] else 'CHECK'}</td><td>偏转 {handoff['planar_rotation_deg']:.1f}°；末端 thumb/index/middle 承力 {handoff['thumb_share']:.3f}/{handoff['index_share']:.3f}/{handoff['middle_share']:.3f}。</td></tr>
<tr><td>接触切换→index/thumb recapture</td><td>{'PASS' if status['contact_switch'] else 'CHECK'}</td><td>偏转 {recapture['planar_rotation_deg']:.1f}°；末端 thumb/index 承力 {recapture['thumb_share']:.3f}/{recapture['index_share']:.3f}。</td></tr>
<tr><td>P4-flight-entry candidate</td><td>{'PASS' if status['p4_flight_entry_candidate'] else 'CHECK'}</td><td>目标 53.9°，实际 {p4['planar_rotation_deg']:.1f}°；撞击帧 thumb/index 承力 {p4['thumb_share']:.3f}/{p4['index_share']:.3f}；穿透 {p4['max_penetration_mm']:.2f}mm，但线速度仍有 {p4['pen_linear_speed']:.3f}m/s。</td></tr>
<tr><td>短时 P4 keyframe</td><td>REJECT</td><td>零残差外推首次无接触出现在 {report['post_p4_audit']['first_no_contact_ms']:.0f}ms；100ms 高度变化 {report['post_p4_audit']['final_dz_mm']:+.1f}mm。当前不满足几十到几百毫秒可用状态。</td></tr>
<tr><td>P4→P5→P1</td><td>OPEN</td><td>从该动态 P4 事后启动 P45 的直接搜索没有有效 thumb 回接；下一步必须做 P4/P45 重叠控制或 RL curriculum。</td></tr>
</table>
<div class="grid">
<figure><img src="corrected_P1_P2_P3_to_dynamic_P4.gif"><figcaption>同一 env 的连续主回放。P3 后不再进入旧 index/ring/thumb 假 recapture，回放在新的 P4 flight-entry candidate 停止。</figcaption></figure>
<figure><img src="corrected_P3_to_dynamic_P4_slow.gif"><figcaption>P3→flight-entry 慢放：index/middle/thumb 推滚，ring 短暂轻防滚，middle 接棒，随后短接触切换并由 thumb/index 撞击。</figcaption></figure>
<figure><img src="corrected_chain_keyframes.png"><figcaption>P1、P2、P3 drive-exit candidate、ring guard、index/thumb recapture 与 P4 flight-entry candidate 的诊断关键帧。</figcaption></figure>
<figure><img src="corrected_P3_to_dynamic_P4_trace.png"><figcaption>相位、位移、法向力分配、速度与穿透。虚线为 primitive 边界，点线为原始 P4 相位参考。</figcaption></figure>
<figure><img src="P4_flight_entry_zero_residual_audit.gif"><figcaption>关键后验检查：在 flight-entry 后不给额外 residual，笔迅速失去接触并下落。因此该帧只保留为 RL 入口，不计入完整链。</figcaption></figure>
</div>
<h2>当前边界</h2>
<p>当前通过的是 P3→flight-entry 的相位与撞击接触，不是 P4 keyframe。该段最长无承力窗口为 {report['tail_summary']['max_no_force_gap_ms']:.0f}ms；P45/制动控制必须从 0.19s middle-handoff 或更早开始预加载，不能在撞击帧之后才启动。</p>
</main></body></html>"""
    (output_dir / "visual_summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "corrected_p3_p4_chain_m3_2_38"
        ),
    )
    parser.add_argument("--seed", type=int, default=245)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parsed = parser.parse_args()
    args = corrected.make_args(parsed)
    args.render_every = parsed.render_every
    root = Path("dex5_penspin/reports/manus_mujoco_presets")
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = corrected.make_env(args, root, render=True)
    try:
        source = build_true_p3(env, args, root)
        p3_state = tail.state_snapshot(env)
        p3 = env.get_state()
        p3_center = p3.pen_center.copy()
        p3_axis = p3.pen_axis.copy()
        p3_row = semantic.metric_row(env, p3_center, p3_axis, 0)
        p3_frame = m316.overlay_frame(
            env.render(),
            row_label(p3_row, "P3 drive-exit candidate"),
        )

        base = env.target_qpos.astype(np.float64).copy()
        rows: list[dict[str, Any]] = []
        detailed_frames: list[np.ndarray] = []
        main_tail_frames: list[np.ndarray] = []
        key_frames: dict[str, np.ndarray] = {
            "03_P3_drive_exit_candidate": p3_frame
        }

        def step_and_record(action: np.ndarray, stage: str) -> dict[str, Any]:
            env.step(np.asarray(action, dtype=np.float32))
            step = len(rows) + 1
            row = semantic.metric_row(env, p3_center, p3_axis, step)
            row["stage"] = stage
            row["t_from_p3_s"] = step / env.control_hz
            rows.append(row)
            frame = m316.overlay_frame(env.render(), row_label(row, stage))
            detailed_frames.append(frame)
            if step % parsed.render_every == 0:
                main_tail_frames.append(frame)
            return row

        for step in range(P34_STEPS):
            action = np.zeros(20, dtype=np.float64)
            for group_index, indices in enumerate(FINGER_GROUPS):
                x = (
                    0.0
                    if step < P34_DELAYS[group_index]
                    else min(
                        1.0,
                        (step - P34_DELAYS[group_index] + 1)
                        / P34_RAMPS[group_index],
                    )
                )
                alpha = smoothstep(x)
                effective = base[indices] + alpha * (
                    P34_TARGET[indices] - base[indices]
                )
                action[indices] = np.clip(
                    P34_GAINS[group_index]
                    * (effective - env.target_qpos[indices].astype(np.float64)),
                    -P34_LIMITS[group_index],
                    P34_LIMITS[group_index],
                )
            ring_indices = FINGER_GROUPS[2]
            x = (
                0.0
                if step < RING_GUARD_DELAY
                else min(
                    1.0,
                    (step - RING_GUARD_DELAY + 1) / RING_GUARD_RAMP,
                )
            )
            alpha = smoothstep(x)
            effective_ring = base[ring_indices] + alpha * (
                RING_GUARD_TARGET - base[ring_indices]
            )
            action[ring_indices] = np.clip(
                RING_GUARD_GAIN
                * (
                    effective_ring
                    - env.target_qpos[ring_indices].astype(np.float64)
                ),
                -RING_GUARD_LIMIT,
                RING_GUARD_LIMIT,
            )
            guard_row = step_and_record(
                mpc.enforce_action_consistency(action),
                "B34 ring guard",
            )
        key_frames["04_B34_ring_guard"] = detailed_frames[-1]
        guard_state = tail.state_snapshot(env)

        for action in GUARD_TO_MIDDLE_ACTIONS:
            handoff_row = step_and_record(action, "B34 middle handoff")
        key_frames["05_B34_middle_handoff"] = detailed_frames[-1]
        handoff_state = tail.state_snapshot(env)

        for _ in range(CONTACT_SWITCH_STEPS):
            recapture_row = step_and_record(
                CONTACT_SWITCH_ACTION,
                "B34 index-thumb recapture",
            )
        key_frames["06_index_thumb_recapture"] = detailed_frames[-1]
        recapture_state = tail.state_snapshot(env)

        for _ in range(P4_CAPTURE_STEPS):
            p4_row = step_and_record(
                P4_CAPTURE_ACTION,
                "P4 flight-entry candidate",
            )
        if len(rows) % parsed.render_every:
            main_tail_frames.append(detailed_frames[-1])
        key_frames["07_P4_flight_entry_candidate"] = detailed_frames[-1]
        p4_state = tail.state_snapshot(env)

        p4_audit_rows: list[dict[str, Any]] = []
        p4_audit_frames = [detailed_frames[-1]]
        p4_center = env.get_state().pen_center.copy()
        p4_axis = env.get_state().pen_axis.copy()
        for audit_step in range(10):
            env.step(np.zeros(20, dtype=np.float32))
            audit_row = semantic.metric_row(
                env,
                p4_center,
                p4_axis,
                audit_step + 1,
            )
            audit_row["stage"] = "P4 flight-entry zero-residual audit"
            audit_row["t_from_p4_s"] = (audit_step + 1) / env.control_hz
            p4_audit_rows.append(audit_row)
            p4_audit_frames.append(
                m316.overlay_frame(
                    env.render(),
                    (
                        f"P4 audit {audit_step + 1}/10 | {audit_row['contacts']} | "
                        f"dz={audit_row['dz_mm']:+.1f}mm | "
                        f"v={audit_row['pen_linear_speed']:.3f}m/s"
                    ),
                )
            )

        prelude_frames = source["prelude_trace"]["frames"]
        b12_frames = source["b12_trace"]["frames"]
        b23_frames = source["b23_trace"]["frames"]
        full_frames = prelude_frames + b12_frames + b23_frames + main_tail_frames
        if full_frames:
            imageio.mimsave(
                output_dir / "corrected_P1_P2_P3_to_dynamic_P4.gif",
                full_frames,
                duration=parsed.render_every / env.control_hz,
            )
        imageio.mimsave(
            output_dir / "corrected_P3_to_dynamic_P4_slow.gif",
            [p3_frame] + detailed_frames,
            duration=0.08,
        )
        imageio.mimsave(
            output_dir / "P4_flight_entry_zero_residual_audit.gif",
            p4_audit_frames,
            duration=0.12,
        )

        if prelude_frames:
            key_frames["01_P1"] = prelude_frames[0]
        if b12_frames:
            key_frames["02_P2"] = b12_frames[-1]
        ordered_keys = sorted(key_frames)
        m316.save_montage(
            [key_frames[key] for key in ordered_keys],
            output_dir / "corrected_chain_keyframes.png",
            cols=3,
        )
        for key in ordered_keys:
            imageio.imwrite(output_dir / f"{key}.png", key_frames[key])

        write_csv(output_dir / "corrected_P3_to_dynamic_P4_trace.csv", rows)
        write_csv(output_dir / "P4_flight_entry_zero_residual_audit.csv", p4_audit_rows)
        save_trace_plot(output_dir / "corrected_P3_to_dynamic_P4_trace.png", rows)
        m334.save_final_cache(output_dir / "corrected_true_P3_entry.npz", p3_state)
        m334.save_final_cache(output_dir / "B34_ring_guard.npz", guard_state)
        m334.save_final_cache(
            output_dir / "B34_middle_handoff.npz",
            handoff_state,
        )
        m334.save_final_cache(
            output_dir / "B34_index_thumb_recapture.npz",
            recapture_state,
        )
        m334.save_final_cache(output_dir / "dynamic_P4_entry.npz", p4_state)
        td3_handoff = write_td3_handoff(
            output_dir,
            root,
            handoff_state,
            recapture_state,
            p4_state,
            handoff_row,
            recapture_row,
            p4_row,
            args,
        )

        p3_contacts = set(p3_row["contacts"].split(","))
        manuscript_p3 = json.loads(
            (
                root
                / "photo_five_phase_hand_pen_tuned_m3_2_10/"
                "P3_index_wrap_to_front_hand_pen_tuned_seed.json"
            ).read_text(encoding="utf-8")
        )
        p3_hand_qpos_rms = float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(p3_state["hand_qpos"], dtype=np.float64)
                        - np.asarray(manuscript_p3["hand_qpos"], dtype=np.float64)
                    )
                    ** 2
                )
            )
        )
        continuous_prefix = bool(source["b23_summary"]["passed"])
        p3_contact_semantics = bool(
            {"thumb", "index", "middle"}.issubset(p3_contacts)
            and p3_row["ring_share"] < 0.05
        )
        acceptance = {
            "continuous_prefix": continuous_prefix,
            "p3_contact_semantics": p3_contact_semantics,
            "p3_canonical_pose": False,
            "continuous_p1_p2_p3": False,
            "ring_guard": bool(
                guard_row["thumb_share"] > 0.45
                and 0.01 < guard_row["ring_share"] < 0.15
                and guard_row["max_penetration_mm"] < 1.5
            ),
            "middle_handoff": bool(
                handoff_row["thumb_share"] > 0.45
                and handoff_row["middle_share"] > 0.05
                and handoff_row["ring_share"] < 0.05
            ),
            "contact_switch": bool(
                recapture_row["thumb_share"] > 0.05
                and recapture_row["index_share"] > 0.05
                and recapture_row["dz_mm"] > -4.0
            ),
            "p4_flight_entry_candidate": bool(
                abs(p4_row["planar_rotation_deg"] - RAW_P3_TO_P4_TARGET_DEG) < 8.0
                and p4_row["thumb_share"] > 0.50
                and p4_row["index_share"] > 0.10
                and p4_row["dz_mm"] > -5.0
                and p4_row["max_penetration_mm"] < 2.0
                and p4_row["pen_linear_speed"] < 0.25
                and no_force_gap(rows) <= 3
            ),
            "dynamic_p4_entry": False,
            "p4_short_state": False,
            "p4_to_p5": False,
            "p5_to_p1": False,
            "complete_cycle": False,
        }
        key_states = {
            "P3_drive_exit_candidate": p3_row,
            "B34_ring_guard": guard_row,
            "B34_middle_handoff": handoff_row,
            "B34_index_thumb_recapture": recapture_row,
            "P4_flight_entry_candidate": p4_row,
        }
        report = {
            "stage": "M3.2.38 P3 and downstream diagnostic branch, no-reset prefix",
            "semantic_correction": {
                "retired_behavior": (
                    "M3.2.35/M3.2.36 forced B23 exit through an index/ring/thumb "
                    "recapture and therefore mislabeled the downstream P3/P4/P5 states."
                ),
                "P3": (
                    "The B23 exit now has the intended index/middle/thumb contact roles and does "
                    "not force ring recapture. Its hand pose is still too far from the Manus P3, "
                    "so it is a drive-exit candidate rather than an accepted canonical keyframe."
                ),
                "B34": (
                    "Index/middle drive around a yielding/supporting thumb. Ring briefly carries "
                    "a small guard load, then unloads as middle resumes the drive."
                ),
                "P4": (
                    "A thumb-main/index-secondary impact near the raw P4 phase. Post-entry "
                    "rollout disproves it as a short keyframe, so it is retained only as a "
                    "flight-entry candidate for an earlier brake curriculum."
                ),
            },
            "source": {
                "p1_p2_p3_controller": (
                    "M3.2.34 no-reset prefix with the M3.2.35 prepared P3 "
                    "drive-exit candidate"
                ),
                "b23_summary": source["b23_summary"],
                "raw_p3_to_p4_target_deg": RAW_P3_TO_P4_TARGET_DEG,
            },
            "key_states": key_states,
            "p3_pose_audit": {
                "hand_qpos_rms_to_manus_rad": p3_hand_qpos_rms,
                "acceptance_threshold_rad": 0.12,
                "passed": bool(p3_hand_qpos_rms < 0.12),
            },
            "tail_summary": {
                "steps": len(rows),
                "duration_ms": len(rows) * 1000.0 / env.control_hz,
                "max_no_force_gap_steps": no_force_gap(rows),
                "max_no_force_gap_ms": no_force_gap(rows) * 1000.0 / env.control_hz,
                "max_penetration_mm": float(
                    max(row["max_penetration_mm"] for row in rows)
                ),
                "min_dz_mm": float(min(row["dz_mm"] for row in rows)),
                "final_rotation_deg": float(p4_row["planar_rotation_deg"]),
            },
            "post_p4_audit": {
                "duration_ms": len(p4_audit_rows) * 1000.0 / env.control_hz,
                "first_no_contact_ms": float(
                    next(
                        (
                            row["t_from_p4_s"] * 1000.0
                            for row in p4_audit_rows
                            if row["contacts"] == "none"
                        ),
                        len(p4_audit_rows) * 1000.0 / env.control_hz,
                    )
                ),
                "final_contacts": p4_audit_rows[-1]["contacts"],
                "final_dz_mm": float(p4_audit_rows[-1]["dz_mm"]),
                "final_linear_speed": float(
                    p4_audit_rows[-1]["pen_linear_speed"]
                ),
            },
            "acceptance": acceptance,
            "downstream_search_boundary": {
                "direct_post_p4_p45_probe": (
                    "No accepted thumb recapture was found in the first 20,000 direct "
                    "five-step probes from this dynamic P4 neighborhood."
                ),
                "next_required_primitive": (
                    "Overlap the P4 capture command with an earlier P45 preload; do not start "
                    "P45 only after the P4 frame."
                ),
            },
            "td3_handoff": td3_handoff,
        }
        save_json(output_dir / "summary.json", report)
        save_json(
            output_dir / "selected_actions.json",
            {
                "p34_target": P34_TARGET.tolist(),
                "p34_delays": list(P34_DELAYS),
                "p34_ramps": list(P34_RAMPS),
                "p34_gains": list(P34_GAINS),
                "p34_limits": list(P34_LIMITS),
                "ring_guard_target": RING_GUARD_TARGET.tolist(),
                "guard_to_middle_actions": GUARD_TO_MIDDLE_ACTIONS.tolist(),
                "contact_switch_action": CONTACT_SWITCH_ACTION.tolist(),
                "contact_switch_steps": CONTACT_SWITCH_STEPS,
                "p4_capture_action": P4_CAPTURE_ACTION.tolist(),
                "p4_capture_steps": P4_CAPTURE_STEPS,
                "p4_overlap_teacher_actions": (
                    [CONTACT_SWITCH_ACTION.tolist()] * CONTACT_SWITCH_STEPS
                    + [P4_CAPTURE_ACTION.tolist()] * P4_CAPTURE_STEPS
                ),
            },
        )
        write_visual_summary(output_dir, report)
        assessment = f"""# M3.2.38 P3 and downstream diagnostic branch

- The old forced index/ring/thumb P3 recapture is retired.
- P3 drive-exit candidate contacts: `{p3_row['contacts']}`; force shares T/I/M/R `{p3_row['thumb_share']:.3f}/{p3_row['index_share']:.3f}/{p3_row['middle_share']:.3f}/{p3_row['ring_share']:.3f}`.
- P3 hand-qpos RMS to the Manus canonical pose: `{p3_hand_qpos_rms:.3f}` rad; canonical P3 acceptance is `False`.
- Ring guard endpoint: `{guard_row['planar_rotation_deg']:.2f}` deg; ring share `{guard_row['ring_share']:.3f}`; thumb share `{guard_row['thumb_share']:.3f}`.
- Middle handoff endpoint: `{handoff_row['planar_rotation_deg']:.2f}` deg; T/I/M `{handoff_row['thumb_share']:.3f}/{handoff_row['index_share']:.3f}/{handoff_row['middle_share']:.3f}`.
- P4 flight-entry candidate: `{p4_row['planar_rotation_deg']:.2f}` deg versus `{RAW_P3_TO_P4_TARGET_DEG:.2f}` deg; T/I `{p4_row['thumb_share']:.3f}/{p4_row['index_share']:.3f}`; dz `{p4_row['dz_mm']:+.2f}` mm; penetration `{p4_row['max_penetration_mm']:.2f}` mm; linear speed `{p4_row['pen_linear_speed']:.3f}` m/s.
- Maximum no-force interval in the corrected tail: `{report['tail_summary']['max_no_force_gap_ms']:.0f}` ms.
- Zero-residual post-entry audit: first no-contact at `{report['post_p4_audit']['first_no_contact_ms']:.0f}` ms; dz `{report['post_p4_audit']['final_dz_mm']:+.1f}` mm after `{report['post_p4_audit']['duration_ms']:.0f}` ms.

This validates a continuous diagnostic prefix and corrected contact roles, not the canonical P3 keyframe. Canonical P3, braked P4, P4->P5, and P5->P1 remain open and are deliberately absent from the claimed cycle.
"""
        (output_dir / "ASSESSMENT.zh.md").write_text(assessment, encoding="utf-8")
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "acceptance": acceptance,
                    "key_states": key_states,
                    "tail_summary": report["tail_summary"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
