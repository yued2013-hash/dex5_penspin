from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

import build_m3_2_33_narrow_021_five_step_chain as m333
import build_m3_2_34_continuous_p1_p2_p3 as m334
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_cycle_closure as cycle
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_35_early_p51_closure as early
import search_m3_2_35_p1_touch_extension as extension
from evaluate_m3_2_15_short_primitives import FINGERS

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


SELECTED_P1_EXTENSION = extension.TouchExtensionCandidate(
    delta=(
        -0.008188137567590107,
        -0.14027820998732493,
        0.07633576581044192,
        0.07633576581044192,
        -0.0016485616831568043,
        0.14440602786941897,
        0.14323749442649897,
        0.14323749442649897,
        -0.026573657941789074,
        -0.05861559302349922,
        0.09993540844742113,
        0.09993540844742113,
        0.0,
        0.0,
        0.0,
        0.0,
        0.04472984542703308,
        -0.02411684661728275,
        -0.07444703407275159,
        0.022258627158628176,
    ),
    ramp=1,
    gain=0.7906349072236202,
    action_limit=0.010849617863130072,
)
P34_STEPS = 16
P45_STEPS = extension.P1_TOUCH_STEP
P1_EXTENSION_STEPS = 2


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


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


def contact_ratio(rows: list[dict[str, Any]], finger: str) -> float:
    return float(
        sum(finger in row["contacts"].split(",") for row in rows)
        / max(1, len(rows))
    )


def force_share_mean(rows: list[dict[str, Any]], finger: str) -> float:
    return float(np.mean([row[f"{finger}_share"] for row in rows]))


def stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_contacts = (
        [] if rows[-1]["contacts"] == "none" else rows[-1]["contacts"].split(",")
    )
    return {
        "steps": len(rows),
        "survival_s": len(rows) / 100.0,
        "final_contacts": final_contacts,
        "contact_ratio": {
            finger: contact_ratio(rows, finger) for finger in FINGERS
        },
        "force_share_mean": {
            finger: force_share_mean(rows, finger) for finger in FINGERS
        },
        "max_penetration_mm": float(
            max(row["max_penetration_mm"] for row in rows)
        ),
        "min_dz_mm": float(min(row["dz_mm"] for row in rows)),
        "final_dz_mm": float(rows[-1]["dz_mm"]),
        "final_linear_speed": float(rows[-1]["pen_linear_speed"]),
        "final_angular_speed": float(rows[-1]["pen_angular_speed"]),
        "rotation_deg": float(rows[-1]["planar_rotation_deg"]),
        "tip_distance_final_mm": {
            finger: float(rows[-1][f"{finger}_tip_distance_mm"])
            for finger in FINGERS
        },
    }


def append_visual_row(
    env: Any,
    *,
    step: int,
    stage: str,
    initial_center: np.ndarray,
    initial_axis: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = early.rollout_row(
        env,
        step=step,
        initial_center=initial_center,
        initial_axis=initial_axis,
    )
    row["t"] = step / env.control_hz
    row["stage"] = stage
    state = m333.snapshot(
        env,
        row["contacts"],
        row["max_penetration_mm"],
        row["t"],
    )
    return row, state


def run_p34_visual(
    env: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = env.target_qpos.astype(np.float64).copy()
    target = np.clip(
        m316.enforce_shared_motor_consistency(base + tail.P34_DELTA),
        env.joint_low,
        env.joint_high,
    )
    ring_target = np.clip(
        np.asarray(extension.sync.SELECTED_RING.p34_ring_target),
        env.joint_low[8:12],
        env.joint_high[8:12],
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    key_frames: dict[str, np.ndarray] = {}
    for step in range(P34_STEPS):
        alpha = 0.0 if step < tail.P34_DELAY else min(
            1.0, (step - tail.P34_DELAY + 1) / tail.P34_RAMP
        )
        effective = base + alpha * (target - base)
        action = np.clip(
            tail.P34_GAIN * (effective - env.target_qpos.astype(np.float64)),
            -tail.P34_ACTION_LIMIT,
            tail.P34_ACTION_LIMIT,
        )
        ring_alpha = min(
            1.0, (step + 1) / extension.sync.SELECTED_RING.p34_ring_ramp
        )
        ring_effective = base[8:12] + ring_alpha * (
            ring_target - base[8:12]
        )
        action[8:12] = np.clip(
            extension.sync.SELECTED_RING.p34_ring_gain
            * (ring_effective - env.target_qpos[8:12].astype(np.float64)),
            -extension.sync.SELECTED_RING.p34_ring_action_limit,
            extension.sync.SELECTED_RING.p34_ring_action_limit,
        )
        env.step(action.astype(np.float32))
        row, state = append_visual_row(
            env,
            step=step + 1,
            stage="P3_to_P4_dynamic_entry",
            initial_center=initial.pen_center,
            initial_axis=initial.pen_axis,
        )
        rows.append(row)
        snapshots.append(state)
        label = (
            f"P3 -> P4 | +{row['t']:.2f}s | {row['contacts']} | "
            f"T/I/M={row['thumb_share']:.2f}/{row['index_share']:.2f}/"
            f"{row['middle_share']:.2f}"
        )
        if step % args.render_every == 0 or step == P34_STEPS - 1:
            frames.append(m316.overlay_frame(env.render(), label))
        if step == P34_STEPS - 1:
            key_frames["04_P4_dynamic_entry"] = m316.overlay_frame(
                env.render(), label
            )
    return {
        "rows": rows,
        "snapshots": snapshots,
        "frames": frames,
        "key_frames": key_frames,
    }


def run_p45_visual(
    env: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate = extension.P1_TOUCH_MIDDLE
    selected_ring = extension.sync.SELECTED_RING
    base = env.target_qpos.astype(np.float64).copy()
    target = np.clip(
        m316.enforce_shared_motor_consistency(base + cycle.P45_DELTA),
        env.joint_low,
        env.joint_high,
    )
    middle_target = np.clip(
        target[4:8] + np.asarray(candidate.middle_delta),
        env.joint_low[4:8],
        env.joint_high[4:8],
    )
    ring_target = np.clip(
        np.asarray(selected_ring.p45_ring_target),
        env.joint_low[8:12],
        env.joint_high[8:12],
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    key_frames: dict[str, np.ndarray] = {}
    for step in range(P45_STEPS):
        alpha = 0.0 if step < cycle.P45_DELAY else min(
            1.0, (step - cycle.P45_DELAY + 1) / cycle.P45_RAMP
        )
        effective = base + alpha * (target - base)
        action = np.clip(
            cycle.P45_GAIN * (effective - env.target_qpos.astype(np.float64)),
            -cycle.P45_ACTION_LIMIT,
            cycle.P45_ACTION_LIMIT,
        )
        middle_alpha = min(1.0, (step + 1) / max(1, candidate.middle_ramp))
        middle_effective = base[4:8] + middle_alpha * (
            middle_target - base[4:8]
        )
        action[4:8] = np.clip(
            candidate.middle_gain
            * (middle_effective - env.target_qpos[4:8].astype(np.float64)),
            -candidate.middle_action_limit,
            candidate.middle_action_limit,
        )
        ring_alpha = min(1.0, (step + 1) / selected_ring.p45_ring_ramp)
        ring_effective = base[8:12] + ring_alpha * (
            ring_target - base[8:12]
        )
        action[8:12] = np.clip(
            selected_ring.p45_ring_gain
            * (ring_effective - env.target_qpos[8:12].astype(np.float64)),
            -selected_ring.p45_ring_action_limit,
            selected_ring.p45_ring_action_limit,
        )
        env.step(action.astype(np.float32))
        stage = "P4_to_P5_drive" if step < P45_STEPS - 1 else "P5_to_P1_touch"
        row, state = append_visual_row(
            env,
            step=step + 1,
            stage=stage,
            initial_center=initial.pen_center,
            initial_axis=initial.pen_axis,
        )
        rows.append(row)
        snapshots.append(state)
        label = (
            f"P4 -> P5 -> P1 touch | +{row['t']:.2f}s | {row['contacts']} | "
            f"T/I/M/R={row['thumb_share']:.2f}/{row['index_share']:.2f}/"
            f"{row['middle_share']:.2f}/{row['ring_share']:.2f}"
        )
        if step % args.render_every == 0 or step == P45_STEPS - 1:
            frames.append(m316.overlay_frame(env.render(), label))
        if step == P45_STEPS - 2:
            key_frames["05_P5_drive_exit"] = m316.overlay_frame(
                env.render(), label
            )
        if step == P45_STEPS - 1:
            key_frames["06_P1_touch_entry"] = m316.overlay_frame(
                env.render(), label
            )
    return {
        "rows": rows,
        "snapshots": snapshots,
        "frames": frames,
        "key_frames": key_frames,
    }


def run_p1_extension_visual(
    env: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = env.target_qpos.astype(np.float64).copy()
    target = np.clip(
        m316.enforce_shared_motor_consistency(
            base + np.asarray(SELECTED_P1_EXTENSION.delta, dtype=np.float64)
        ),
        env.joint_low,
        env.joint_high,
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    key_frames: dict[str, np.ndarray] = {}
    for step in range(P1_EXTENSION_STEPS):
        alpha = min(1.0, (step + 1) / max(1, SELECTED_P1_EXTENSION.ramp))
        effective = base + alpha * (target - base)
        action = np.clip(
            SELECTED_P1_EXTENSION.gain
            * (effective - env.target_qpos.astype(np.float64)),
            -SELECTED_P1_EXTENSION.action_limit,
            SELECTED_P1_EXTENSION.action_limit,
        ).astype(np.float32)
        env.step(action)
        row, state = append_visual_row(
            env,
            step=step + 1,
            stage="P1_short_contact_extension",
            initial_center=initial.pen_center,
            initial_axis=initial.pen_axis,
        )
        rows.append(row)
        snapshots.append(state)
        label = (
            f"P1 short entry | +{row['t']:.2f}s | {row['contacts']} | "
            f"T/I/M/R={row['thumb_share']:.2f}/{row['index_share']:.2f}/"
            f"{row['middle_share']:.2f}/{row['ring_share']:.2f}"
        )
        frames.append(m316.overlay_frame(env.render(), label))
        if step == P1_EXTENSION_STEPS - 1:
            key_frames["07_P1_short_entry_30ms"] = m316.overlay_frame(
                env.render(), label
            )
    return {
        "rows": rows,
        "snapshots": snapshots,
        "frames": frames,
        "key_frames": key_frames,
    }


def save_trace_plot(
    path: Path,
    rows: list[dict[str, Any]],
    boundaries: list[tuple[float, str]],
) -> None:
    if plt is None or not rows:
        return
    t = np.asarray([row["t"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(t, [row["dx_mm"] for row in rows], label="dx")
    axes[0].plot(t, [row["dy_mm"] for row in rows], label="dy")
    axes[0].plot(t, [row["dz_mm"] for row in rows], label="dz")
    axes[0].set_ylabel("pen delta (mm)")
    axes[0].legend(ncol=3)
    for finger in FINGERS:
        axes[1].plot(t, [row[f"{finger}_share"] for row in rows], label=finger)
    axes[1].set_ylabel("force share")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(ncol=5)
    for finger in ("thumb", "index", "middle", "ring"):
        values = [row.get(f"{finger}_tip_distance_mm", np.nan) for row in rows]
        axes[2].plot(t, values, label=finger)
    axes[2].axhline(8.5, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("tip distance (mm)")
    axes[2].legend(ncol=4)
    axes[3].plot(t, [row["pen_linear_speed"] for row in rows], label="linear")
    axes[3].plot(t, [row["pen_angular_speed"] for row in rows], label="angular")
    axes[3].plot(
        t, [row["max_penetration_mm"] for row in rows], label="penetration mm"
    )
    axes[3].set_ylabel("speed / penetration")
    axes[3].legend(ncol=3)
    axes[4].plot(t, [row["planar_rotation_deg"] for row in rows], label="rotation")
    axes[4].set_ylabel("rotation (deg)")
    axes[4].set_xlabel("continuous chain time (s)")
    axes[4].legend()
    for axis in axes:
        for boundary, _ in boundaries:
            axis.axvline(boundary, color="black", linestyle="--", linewidth=0.7)
    for boundary, label in boundaries:
        axes[0].text(
            boundary,
            axes[0].get_ylim()[1],
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )
    fig.suptitle("M3.2.35 continuous P1-P2-P3-P4-P5-P1 entry (no reset)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_visual_summary(output_dir: Path, report: dict[str, Any]) -> None:
    chain = report["continuous_chain"]
    p1_entry = report["p5_to_p1_entry"]
    acceptance = report["acceptance"]
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M3.2.35 五步真实动作链</title>
<style>
body {{ margin:0; background:#f4f5f7; color:#202124; font-family:Arial,"Noto Sans SC",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 8px; font-size:24px; letter-spacing:0; }}
h2 {{ margin:26px 0 10px; font-size:18px; letter-spacing:0; }}
p {{ line-height:1.6; }}
.status {{ padding:12px 0; border-top:1px solid #c7cbd1; border-bottom:1px solid #c7cbd1; }}
.notice {{ border-left:4px solid #b56b00; padding:10px 12px; background:#fff8e8; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:18px; }}
figure {{ margin:0; overflow:hidden; background:white; border:1px solid #d6dae0; border-radius:6px; }}
img {{ display:block; width:100%; height:auto; }}
figcaption {{ padding:10px 12px; font-size:14px; line-height:1.5; }}
table {{ width:100%; margin-top:16px; border-collapse:collapse; background:white; }}
th,td {{ padding:8px 10px; border:1px solid #d6dae0; text-align:left; font-size:14px; }}
.pass {{ color:#176b3a; font-weight:700; }} .check {{ color:#9a5b00; font-weight:700; }}
@media(max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} main {{ padding:14px; }} }}
</style></head><body><main>
<h1>P1 → P2 → P3 → P4 → P5 → P1-entry</h1>
<p class="status"><span class="{'pass' if acceptance['continuous_five_step_entry_chain'] else 'check'}">{'ENTRY PASS' if acceptance['continuous_five_step_entry_chain'] else 'CHECK'}</span>：同一个 MuJoCo env 无重置连续运行 {chain['survival_s']:.2f}s，最大穿透 {chain['max_penetration_mm']:.2f}mm；末段获得连续 {p1_entry['consecutive_four_finger_steps']} 帧（{p1_entry['duration_s']*1000:.0f}ms）四指 P1-entry。</p>
<p class="notice"><strong>验收边界：</strong>这证明五个接触状态可以无 reset 连到一个短时 P1 回接入口，不等于已完成整圈笔的长时稳定闭环，也不等于已达到真实转笔的完整角位移。</p>
<table><tr><th>阶段</th><th>结果</th><th>关键指标</th></tr>
<tr><td>P1 → running</td><td>{'PASS' if acceptance['p1_to_running'] else 'CHECK'}</td><td>{report['p1_to_running']['survival_s']:.2f}s；末接触 {','.join(report['p1_to_running']['final_contacts'])}</td></tr>
<tr><td>running → P2</td><td>{'PASS' if acceptance['running_to_p2'] else 'CHECK'}</td><td>ring 转移到笔下方；穿透 {report['running_to_p2']['max_penetration_mm']:.2f}mm</td></tr>
<tr><td>P2 → P3</td><td>{'PASS' if acceptance['p2_to_p3'] else 'CHECK'}</td><td>偏转 {report['p2_to_p3']['planar_rotation_deg']:.2f}°；末接触 {','.join(report['p2_to_p3']['final_contacts'])}</td></tr>
<tr><td>P3 → P4</td><td>{'PASS' if acceptance['p3_to_p4'] else 'CHECK'}</td><td>thumb/index/middle 接触率 {report['p3_to_p4']['contact_ratio']['thumb']:.2f}/{report['p3_to_p4']['contact_ratio']['index']:.2f}/{report['p3_to_p4']['contact_ratio']['middle']:.2f}</td></tr>
<tr><td>P4 → P5</td><td>{'PASS' if acceptance['p4_to_p5'] else 'CHECK'}</td><td>末端笔速 {report['p4_to_p5']['final_linear_speed']:.3f}m/s；穿透 {report['p4_to_p5']['max_penetration_mm']:.2f}mm</td></tr>
<tr><td>P5 → P1-entry</td><td>{'PASS' if acceptance['p5_to_p1_entry'] else 'CHECK'}</td><td>{p1_entry['duration_s']*1000:.0f}ms 四指连续接触；thumb/ring 平均承力 {p1_entry['force_share_mean']['thumb']:.3f}/{p1_entry['force_share_mean']['ring']:.3f}</td></tr>
</table>
<div class="grid">
<figure><img src="continuous_five_step_chain.gif"><figcaption>同一 env 的连续回放：P1 教师前段、ring-under P2、index/middle 推滚 P3、P4/P5 动态尾段，以及 thumb 主承力的 P1 短时回接。</figcaption></figure>
<figure><img src="continuous_five_step_keyframes.png"><figcaption>关键帧依次展示 P1、P2、P3、P4、P5 和 P1-entry；ring 在末段仅做防滚接触。</figcaption></figure>
<figure><img src="continuous_five_step_trace.png"><figcaption>整链的笔位移、承力比例、指尖距离、速度/穿透和平面角位移。虚线是 primitive 边界。</figcaption></figure>
</div></main></body></html>"""
    (output_dir / "visual_summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "photo_five_phase_hand_pen_tuned_m3_2_10"
        ),
    )
    parser.add_argument(
        "--cache",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "short_primitive_reward_m3_2_23/dynamic_state_cache/"
            "earlier_pre_ring_for_B12b_to_B23.npz"
        ),
    )
    parser.add_argument(
        "--b23-reference-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p2_bottom_cradle_b23_thumb_pass_m3_2_31"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "continuous_five_step_chain_m3_2_35"
        ),
    )
    parser.add_argument("--source-t", type=float, default=0.21)
    parser.add_argument("--source-segment-duration", type=float, default=0.18)
    parser.add_argument("--b12-duration", type=float, default=0.58)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parser.add_argument("--p1-index-open", type=float, default=-0.012)
    parser.add_argument("--p1-middle-open", type=float, default=-0.004)
    parser.add_argument("--p1-ring-open", type=float, default=-0.020)
    parser.add_argument("--p1-thumb-open", type=float, default=0.0)
    args = parser.parse_args()
    args.action_delta_limit = 0.040
    args.thumb_force_limit = 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path("dex5_penspin/reports/manus_mujoco_presets")
    route = early.m323.build_route(Path(args.seed_dir))
    safety = {
        "index": args.p1_index_open,
        "middle": args.p1_middle_open,
        "ring": args.p1_ring_open,
        "thumb": args.p1_thumb_open,
    }
    initial_seed = m334.apply_p1_safety_open(route, safety)[0]
    references = tail.load_references(Path(args.b23_reference_dir))
    cache = np.load(Path(args.cache))
    cache_index = m333.m332.cache_index(cache, args.source_t)
    env = m316.make_env(initial_seed, m334.model_args(args), duration=2.4, render=True)
    controller_args = m334.source_controller_args(args)

    try:
        prelude_summary, prelude_trace = m334.prelude_rollout(
            env,
            route,
            initial_seed,
            controller_args,
            args,
            render=True,
        )
        step_after_prelude = int(env.step_count)
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
        step_after_b12 = int(env.step_count)
        b23_summary, b23_trace = m333.b23_rollout(
            env,
            b12_trace["snapshots"][-1],
            references,
            early.PREPARED_P3,
            args,
            label="continuous_prepared_P3",
            render=True,
            reset_state=False,
        )
        step_after_b23 = int(env.step_count)
        p34_trace = run_p34_visual(env, args)
        step_after_p34 = int(env.step_count)
        p45_trace = run_p45_visual(env, args)
        step_after_p45 = int(env.step_count)
        p1_trace = run_p1_extension_visual(env, args)
        step_after_p1_entry = int(env.step_count)
    finally:
        env.close()

    segments = [
        ("P1_to_running", prelude_trace["rows"], prelude_trace["snapshots"]),
        ("running_to_P2", b12_trace["rows"], b12_trace["snapshots"]),
        ("P2_to_P3", b23_trace["rows"], b23_trace["snapshots"]),
        ("P3_to_P4", p34_trace["rows"], p34_trace["snapshots"]),
        ("P4_to_P5", p45_trace["rows"], p45_trace["snapshots"]),
        ("P5_to_P1_entry", p1_trace["rows"], p1_trace["snapshots"]),
    ]
    combined_rows = m334.combine_rows(
        segments,
        np.asarray(prelude_trace["initial_center"]),
        np.asarray(prelude_trace["initial_axis"]),
        100.0,
    )
    write_csv(output_dir / "continuous_five_step_trace.csv", combined_rows)

    frames = (
        prelude_trace["frames"]
        + b12_trace["frames"]
        + b23_trace["frames"]
        + p34_trace["frames"]
        + p45_trace["frames"]
        + p1_trace["frames"]
    )
    if frames:
        imageio.mimsave(
            output_dir / "continuous_five_step_chain.gif",
            frames,
            duration=args.render_every / 100.0,
        )

    selected_frames: dict[str, np.ndarray] = {}
    if "01_P1_start" in prelude_trace["key_frames"]:
        selected_frames["01_P1_start"] = prelude_trace["key_frames"]["01_P1_start"]
    if "04_projected_P2_arrival" in b12_trace["key_frames"]:
        selected_frames["02_P2_ring_under"] = b12_trace["key_frames"][
            "04_projected_P2_arrival"
        ]
    if b23_trace["key_frames"]:
        selected_frames["03_P3_drive_exit"] = list(
            b23_trace["key_frames"].values()
        )[-1]
    selected_frames.update(p34_trace["key_frames"])
    selected_frames.update(p45_trace["key_frames"])
    selected_frames.update(p1_trace["key_frames"])
    for name, frame in selected_frames.items():
        imageio.imwrite(output_dir / f"{name}.png", frame)
    if selected_frames:
        m316.save_montage(
            list(selected_frames.values()),
            output_dir / "continuous_five_step_keyframes.png",
            cols=3,
        )

    lengths = [len(rows) for _, rows, _ in segments]
    cumulative = np.cumsum(lengths) / 100.0
    boundary_labels = ["P1", "P2", "P3", "P4", "P5"]
    boundaries = list(zip(cumulative[:-1], boundary_labels))
    save_trace_plot(
        output_dir / "continuous_five_step_trace.png",
        combined_rows,
        boundaries,
    )

    p34_summary = stage_summary(p34_trace["rows"])
    p45_summary = stage_summary(p45_trace["rows"])
    p1_window = [p45_trace["rows"][-1], *p1_trace["rows"]]
    p1_contact_sets = [set(row["contacts"].split(",")) for row in p1_window]
    four_flags = [
        {"thumb", "index", "middle", "ring"}.issubset(contacts)
        for contacts in p1_contact_sets
    ]
    consecutive_four = extension.sync.sustained.longest_run(four_flags)
    p1_entry_summary = {
        "window_steps": len(p1_window),
        "consecutive_four_finger_steps": consecutive_four,
        "duration_s": consecutive_four / 100.0,
        "contacts": [row["contacts"] for row in p1_window],
        "force_share_mean": {
            finger: force_share_mean(p1_window, finger) for finger in FINGERS
        },
        "max_penetration_mm": float(
            max(row["max_penetration_mm"] for row in p1_window)
        ),
        "min_dz_mm": float(min(row["dz_mm"] for row in p1_window)),
        "final_linear_speed": float(p1_window[-1]["pen_linear_speed"]),
        "final_contacts": sorted(p1_contact_sets[-1]),
    }

    expected_counts = [
        prelude_summary["expected_steps"],
        int(round(args.b12_duration * 100.0)),
        sum(steps for _, steps, _ in b23_trace["stages"]),
        P34_STEPS,
        P45_STEPS,
        P1_EXTENSION_STEPS,
    ]
    actual_cumulative = [
        step_after_prelude,
        step_after_b12,
        step_after_b23,
        step_after_p34,
        step_after_p45,
        step_after_p1_entry,
    ]
    expected_cumulative = np.cumsum(expected_counts).astype(int).tolist()
    continuous_step_count = actual_cumulative == expected_cumulative
    p1_to_running_pass = bool(
        prelude_summary["steps"] == prelude_summary["expected_steps"]
        and not prelude_summary["terminated"]
        and prelude_summary["max_penetration_mm"] <= 4.0
    )
    p3_to_p4_pass = bool(
        len(p34_trace["rows"]) == P34_STEPS
        and min(
            p34_summary["contact_ratio"]["thumb"],
            p34_summary["contact_ratio"]["index"],
            p34_summary["contact_ratio"]["middle"],
        )
        >= 0.85
        and p34_summary["max_penetration_mm"] <= 4.0
        and p34_summary["min_dz_mm"] >= -4.0
    )
    p4_to_p5_pass = bool(
        len(p45_trace["rows"]) == P45_STEPS
        and min(
            p45_summary["contact_ratio"]["thumb"],
            p45_summary["contact_ratio"]["index"],
        )
        >= 0.80
        and p45_summary["max_penetration_mm"] <= 4.0
        and p45_summary["min_dz_mm"] >= -4.0
        and p45_summary["final_linear_speed"] <= 0.10
    )
    p5_to_p1_pass = bool(
        consecutive_four >= 3
        and p1_entry_summary["force_share_mean"]["thumb"]
        >= p1_entry_summary["force_share_mean"]["ring"]
        and p1_entry_summary["force_share_mean"]["ring"] <= 0.20
        and p1_entry_summary["max_penetration_mm"] <= 4.0
        and p1_entry_summary["min_dz_mm"] >= -2.0
        and p1_entry_summary["final_linear_speed"] <= 0.10
    )
    acceptance = {
        "p1_to_running": p1_to_running_pass,
        "running_to_p2": bool(all(b12_summary["acceptance"].values())),
        "p2_to_p3": bool(b23_summary["passed"]),
        "p3_to_p4": p3_to_p4_pass,
        "p4_to_p5": p4_to_p5_pass,
        "p5_to_p1_entry": p5_to_p1_pass,
        "continuous_step_count": continuous_step_count,
    }
    acceptance["continuous_five_step_entry_chain"] = bool(
        all(acceptance.values())
        and max(row["max_penetration_mm"] for row in combined_rows) <= 4.0
        and m334.max_no_contact_gap(combined_rows) <= 2
    )
    acceptance["full_stable_closed_cycle"] = False

    chain_summary = {
        "steps": len(combined_rows),
        "survival_s": len(combined_rows) / 100.0,
        "boundary_reset_count": 0,
        "actual_step_count_boundaries": actual_cumulative,
        "expected_step_count_boundaries": expected_cumulative,
        "max_penetration_mm": float(
            max(row["max_penetration_mm"] for row in combined_rows)
        ),
        "min_height_delta_mm": float(min(row["dz_mm"] for row in combined_rows)),
        "max_no_contact_gap_steps": m334.max_no_contact_gap(combined_rows),
        "total_planar_rotation_deg": float(combined_rows[-1]["planar_rotation_deg"]),
        "final_contacts": p1_entry_summary["final_contacts"],
    }
    if p34_trace["snapshots"]:
        m334.save_final_cache(
            output_dir / "continuous_P4_entry.npz", p34_trace["snapshots"][-1]
        )
    if p45_trace["snapshots"]:
        m334.save_final_cache(
            output_dir / "continuous_P1_touch_source.npz",
            p45_trace["snapshots"][-1],
        )
    if p1_trace["snapshots"]:
        m334.save_final_cache(
            output_dir / "continuous_P1_short_entry.npz",
            p1_trace["snapshots"][-1],
        )

    report = {
        "stage": "M3.2.35 continuous five-step real-action contact-state chain",
        "controllers": {
            "b12": asdict(m334.SELECTED_B12),
            "b23_prepared": asdict(early.PREPARED_P3),
            "p34_p45_ring": asdict(extension.sync.SELECTED_RING),
            "p45_middle_sync": asdict(extension.P1_TOUCH_MIDDLE),
            "p1_extension": asdict(SELECTED_P1_EXTENSION),
        },
        "p1_to_running": prelude_summary,
        "running_to_p2": b12_summary,
        "p2_to_p3": b23_summary,
        "p3_to_p4": p34_summary,
        "p4_to_p5": p45_summary,
        "p5_to_p1_entry": p1_entry_summary,
        "continuous_chain": chain_summary,
        "acceptance": acceptance,
        "limits": (
            "The rollout reaches a 30 ms short P1 contact-state entry without any state reset. "
            "It does not establish a long-horizon closed cycle or a full physical pen revolution. "
            "The P1 entry should be used as a transition/reset state for curriculum training, not "
            "reported as a solved continuous pen-spin policy."
        ),
    }
    save_json(output_dir / "summary.json", report)
    write_visual_summary(output_dir, report)
    assessment = f"""# M3.2.35 五步真实动作链连续验收

- 单一 MuJoCo env，连续 `{chain_summary['survival_s']:.2f}s`，primitive 边界 reset 次数 `0`。
- P1->P2->P3 保留 M3.2.34 主链；B23 加入 ring 预布置，P3 末接触 `{','.join(b23_summary['final_contacts'])}`。
- P3->P4：thumb/index/middle 接触率 `{p34_summary['contact_ratio']['thumb']:.2f}/{p34_summary['contact_ratio']['index']:.2f}/{p34_summary['contact_ratio']['middle']:.2f}`。
- P5->P1-entry：连续 `{consecutive_four}` 帧 / `{consecutive_four * 10}ms` 四指接触；thumb/ring 平均承力 `{p1_entry_summary['force_share_mean']['thumb']:.3f}/{p1_entry_summary['force_share_mean']['ring']:.3f}`。
- 整链最大穿透 `{chain_summary['max_penetration_mm']:.2f}mm`，最低高度变化 `{chain_summary['min_height_delta_mm']:.2f}mm`，最长无接触间隔 `{chain_summary['max_no_contact_gap_steps']}` 帧。
- 短时入口验收：`{acceptance['continuous_five_step_entry_chain']}`；长时完整闭环：`False`。

当前结果适合作为 Isaac/MuJoCo curriculum 的五段短 primitive 与 reset/entry 状态。尚不能宣称完成整圈转笔；下一阶段需要训练 P1-entry 后的持续保持和角位移闭合。
"""
    (output_dir / "ASSESSMENT.zh.md").write_text(assessment, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "acceptance": acceptance,
                "continuous_chain": chain_summary,
                "p5_to_p1_entry": p1_entry_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
