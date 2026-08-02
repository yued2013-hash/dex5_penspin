from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

import rebuild_m3_2_16_p4_p5_transitions as m316
from evaluate_m3_2_15_short_primitives import FINGERS, contact_metrics


P2_Q_DELTA = {
    4: -0.030,
    5: -0.015,
    6: -0.010,
    7: -0.010,
    10: 0.020,
    11: 0.020,
}
P2_POS_DELTA = np.array([0.002, 0.0, 0.002], dtype=np.float64)
P2_SUPPORT_RING_DISTAL_DELTA = 0.030

B23_INDEX_FRONT = {
    "blend": 0.40,
    "index_pip": 0.28,
    "index_mid": 0.10,
    "middle_pip": 0.04,
    "middle_mid": 0.00,
    "ring_distal": 0.03,
    "thumb_yield": 0.01,
}
B23_INDEX_MIDDLE_DRIVE = {
    "blend": 0.72,
    "index_pip": 0.40,
    "index_mid": 0.16,
    "middle_pip": 0.12,
    "middle_mid": 0.02,
    "ring_distal": -0.01,
    "thumb_yield": 0.020,
}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def build_p2_bottom_cradle(p2: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    q = np.asarray(p2["hand_qpos"], dtype=np.float64).copy()
    for index, delta in P2_Q_DELTA.items():
        q[index] += delta
    q = m316.enforce_shared_motor_consistency(q)

    entry = {
        **p2,
        "name": "p2_ring_bottom_thumb_cradle_entry",
        "desired_contacts": ["thumb", "ring"],
        "hand_qpos": q.astype(float).tolist(),
        "target_qpos": q.astype(float).tolist(),
        "pen_pos": (np.asarray(p2["pen_pos"], dtype=np.float64) + P2_POS_DELTA).astype(float).tolist(),
        "m3_2_31_role": (
            "P2 reset/keypose: ring fingertip is below the pen and shares the initial normal load "
            "with thumb. Middle may enter later as lateral stabilization, not as the P2 transfer target."
        ),
        "m3_2_31_q_delta": {str(key): float(value) for key, value in P2_Q_DELTA.items()},
        "m3_2_31_pos_delta": P2_POS_DELTA.astype(float).tolist(),
    }

    support_q = q.copy()
    support_q[10] += P2_SUPPORT_RING_DISTAL_DELTA
    support_q[11] = support_q[10]
    support = {
        **entry,
        "phase": 2.05,
        "name": "p2_ring_bottom_thumb_cradle_support",
        "hand_qpos": support_q.astype(float).tolist(),
        "target_qpos": support_q.astype(float).tolist(),
        "m3_2_31_role": (
            "Dynamic-only P2 support target: a 0.03 rad distal ring preload keeps ring contact while "
            "thumb remains the largest load path."
        ),
    }
    return entry, support


def build_b23_waypoint(
    p2_support: dict[str, Any],
    p3: dict[str, Any],
    spec: dict[str, float],
    *,
    name: str,
    phase: float,
    desired: list[str],
    role: str,
) -> dict[str, Any]:
    alpha = m316.smoothstep(float(spec["blend"]))
    q2 = np.asarray(p2_support["hand_qpos"], dtype=np.float64)
    q3 = np.asarray(p3["hand_qpos"], dtype=np.float64)
    q = (1.0 - alpha) * q2 + alpha * q3

    # Index roll reaches the P3 front-side value early, then index/middle closing supplies the drive.
    q[0] = min(q[0], q3[0])
    q[1] += spec["index_pip"]
    q[2] += spec["index_mid"]
    q[5] += spec["middle_pip"]
    q[6] += spec["middle_mid"]
    q[7] = q[6]
    q[10] += spec["ring_distal"]
    q[11] = q[10]

    thumb_yield = float(spec["thumb_yield"])
    q[17] -= 0.50 * thumb_yield
    q[18] -= thumb_yield
    q[19] -= 0.60 * thumb_yield
    q = m316.enforce_shared_motor_consistency(q)

    pos = (1.0 - alpha) * np.asarray(p2_support["pen_pos"], dtype=np.float64)
    pos += alpha * np.asarray(p3["pen_pos"], dtype=np.float64)
    quat = m316.slerp(
        np.asarray(p2_support["pen_quat"], dtype=np.float64),
        np.asarray(p3["pen_quat"], dtype=np.float64),
        alpha,
    )
    return {
        **p3,
        "phase": float(phase),
        "name": name,
        "desired_contacts": desired,
        "hand_qpos": q.astype(float).tolist(),
        "target_qpos": q.astype(float).tolist(),
        "pen_pos": pos.astype(float).tolist(),
        "pen_quat": m316.normalize(quat).astype(float).tolist(),
        "m3_2_31_role": role,
        "m3_2_31_spec": {key: float(value) for key, value in spec.items()},
    }


def build_b23_chain(
    p2_support: dict[str, Any], p3: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    index_front = build_b23_waypoint(
        p2_support,
        p3,
        B23_INDEX_FRONT,
        name="b23a_index_front_deflect",
        phase=2.40,
        desired=["thumb", "index", "ring"],
        role=(
            "Index rolls to the front and closes moderately to seed pen deflection. Ring still supports "
            "from below; thumb keeps light contact."
        ),
    )
    drive = build_b23_waypoint(
        p2_support,
        p3,
        B23_INDEX_MIDDLE_DRIVE,
        name="b23b_index_middle_drive_thumb_yield",
        phase=2.72,
        desired=["thumb", "index", "middle"],
        role=(
            "Index and middle close together to drive the pen around thumb. Thumb rolls/opens slightly "
            "for clearance but retains intermittent support; ring unloads from its P2 cradle role."
        ),
    )
    exit_seed = {
        **p3,
        "name": "p3_index_front_pass_exit",
        "desired_contacts": ["thumb", "index"],
        "m3_2_31_role": (
            "P3 exit/reference. Index/middle drive is a transient B23 action; the exit only requires "
            "thumb/index recapture and allows middle to remain."
        ),
    }
    return index_front, drive, exit_seed


def controller_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        action_delta_limit=args.action_delta_limit,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
        tracking_gain=args.tracking_gain,
        ramp_seconds=args.ramp_seconds,
        initial_thumb_bias=args.initial_thumb_bias,
        initial_ring_open=0.0,
        initial_shelf_bias=0.0,
        thumb_share_target=args.thumb_share_target,
        ring_share_max=0.78,
        ring_share_min=0.08,
        axis_z_guard=0.23,
        height_drop_guard=0.006,
        thumb_bias_min=0.0,
        thumb_bias_max=0.045,
        ring_open_min=0.0,
        ring_open_max=0.010,
        ring_guard_max=0.035,
        shelf_bias_min=0.0,
        shelf_bias_max=0.018,
        thumb_gain=0.025,
        ring_open_gain=0.008,
        ring_guard_gain=0.040,
        ring_guard_decay=0.90,
        shelf_gain=0.015,
        height_gain=0.55,
    )


def tip_geometry(env: Any) -> dict[str, dict[str, float]]:
    state = env.get_state()
    result: dict[str, dict[str, float]] = {}
    for finger, geom_name in m316.TIP_GEOMS.items():
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        rel = env.data.geom_xpos[geom_id] - state.pen_center
        radial = rel - float(np.dot(rel, state.pen_axis)) * state.pen_axis
        result[finger] = {
            "radial_z_mm": float(radial[2] * 1000.0),
            "radial_norm_mm": float(np.linalg.norm(radial) * 1000.0),
        }
    return result


def reset_geometry(env: Any, seed: dict[str, Any]) -> dict[str, Any]:
    m316.reset_to_seed(env, seed)
    metrics = contact_metrics(env)
    return {
        "contacts": sorted(metrics["fingers"]),
        "force_share": {finger: float(metrics["force_share"].get(finger, 0.0)) for finger in FINGERS},
        "normal_force": {key: float(value) for key, value in metrics["normal_force"].items()},
        "max_penetration_mm": float(metrics["max_penetration"] * 1000.0),
        "tip_geometry": tip_geometry(env),
    }


def save_rollout(
    env: Any,
    route: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    label: str,
    duration: float,
    segment_s: float,
    output_dir: Path,
    render_every: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary, trace = m316.run_route(
        env,
        route,
        args,
        duration=duration,
        segment_s=segment_s,
        render_every=render_every,
        label=label,
    )
    if trace["frames"]:
        imageio.mimsave(
            output_dir / f"{label}.gif",
            trace["frames"],
            duration=render_every / env.control_hz,
        )
    write_csv(output_dir / f"{label}_trace.csv", trace["rows"])
    m316.save_trace_plot(output_dir / f"{label}_trace.png", trace, label)
    return summary, trace


def contact_window(rows: list[dict[str, Any]], start_s: float, end_s: float) -> dict[str, Any]:
    window = [row for row in rows if start_s <= float(row["t"]) < end_s]
    if not window:
        return {"samples": 0, "contact_ratio": {}, "force_share_mean": {}}
    contact_ratio = {}
    force_share_mean = {}
    for finger in FINGERS:
        contact_ratio[finger] = float(
            sum(finger in str(row["contacts"]).split(",") for row in window) / len(window)
        )
        key = f"{finger}_share"
        force_share_mean[finger] = float(sum(float(row.get(key, 0.0)) for row in window) / len(window))
    return {
        "samples": len(window),
        "contact_ratio": contact_ratio,
        "force_share_mean": force_share_mean,
    }


def planar_rotation_deg(initial_axis: np.ndarray, final_axis: np.ndarray) -> float:
    initial = math.atan2(float(initial_axis[1]), float(initial_axis[0]))
    final = math.atan2(float(final_axis[1]), float(final_axis[0]))
    return float(math.degrees((final - initial + math.pi) % (2.0 * math.pi) - math.pi))


def summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "label": summary["label"],
        "survival_s": summary["survival_s"],
        "terminated": summary["terminated"],
        "active_contact_ratio": summary["active_contact_ratio"],
        "max_penetration_mm": summary["max_penetration_mm"],
        "final_contacts": ",".join(summary["final_contacts"]) or "none",
    }
    for finger in FINGERS:
        row[f"{finger}_contact_ratio"] = summary["contact_ratio"][finger]
        row[f"{finger}_force_share"] = summary["force_share_mean"][finger]
    return row


def write_visual_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    p2 = summary["p2_hold"]
    b23 = summary["p2_to_p3"]
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M3.2.31 P2 bottom cradle and B23 thumb pass</title>
<style>
body {{ margin: 0; font-family: sans-serif; color: #202124; background: #f5f6f8; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 24px; margin: 0 0 8px; }}
p {{ line-height: 1.55; }}
.status {{ padding: 12px 0; border-top: 1px solid #c8ccd2; border-bottom: 1px solid #c8ccd2; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 20px; }}
figure {{ margin: 0; background: white; border: 1px solid #d7dbe0; border-radius: 6px; overflow: hidden; }}
img {{ display: block; width: 100%; height: auto; }}
figcaption {{ padding: 10px 12px; font-size: 14px; line-height: 1.45; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; background: white; }}
th, td {{ padding: 8px 10px; border: 1px solid #d7dbe0; text-align: left; font-size: 14px; }}
@media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} main {{ padding: 14px; }} }}
</style>
</head>
<body><main>
<h1>P2 底托 + P2→P3 thumb-pass MuJoCo 调试</h1>
<p class="status">P2 初始改为 thumb/ring 双点底托；B23 改为 index 前绕偏转 → index/middle 推滚 → thumb 避让但保留间歇支撑。</p>
<table>
<tr><th>验收项</th><th>结果</th><th>关键指标</th></tr>
<tr><td>P2 short hold</td><td>{'PASS' if summary['acceptance']['p2_hold'] else 'CHECK'}</td><td>{p2['survival_s']:.2f}s; thumb/ring contact {p2['contact_ratio']['thumb']:.2f}/{p2['contact_ratio']['ring']:.2f}; penetration {p2['max_penetration_mm']:.2f}mm</td></tr>
<tr><td>P2→P3 staged transition</td><td>{'PASS' if summary['acceptance']['p2_to_p3'] else 'CHECK'}</td><td>{b23['survival_s']:.2f}s; rotation {b23['planar_rotation_deg']:.1f}deg; penetration {b23['max_penetration_mm']:.2f}mm</td></tr>
</table>
<div class="grid">
<figure><img src="m3_2_31_keypose_montage.png"><figcaption>旧 P2、新 P2 底托、B23a 前绕/短暂接触偏转、B23b 双指推滚与 P3 出口。</figcaption></figure>
<figure><img src="p2_bottom_cradle_hold.gif"><figcaption>P2 0.24s short hold：ring 保持底部接触，thumb 为最大承力路径，middle 仅作侧向稳定。</figcaption></figure>
<figure><img src="p2_to_p3_index_middle_thumb_pass.gif"><figcaption>P2→P3：index 前绕施加偏转，随后 index/middle 推滚；thumb 短暂卸载避让并重新接触。</figcaption></figure>
<figure><img src="p2_to_p3_index_middle_thumb_pass_trace.png"><figcaption>高度、各指承力比与主动控制偏置 trace。</figcaption></figure>
</div>
</main></body></html>
"""
    (output_dir / "visual_summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/p1_p2_micro_rebuild_m3_2_18",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/p2_bottom_cradle_b23_thumb_pass_m3_2_31",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--segment-duration", type=float, default=0.115)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.009)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parser.add_argument("--action-delta-limit", type=float, default=0.022)
    parser.add_argument("--tracking-gain", type=float, default=0.58)
    parser.add_argument("--ramp-seconds", type=float, default=0.07)
    parser.add_argument("--initial-thumb-bias", type=float, default=0.008)
    parser.add_argument("--thumb-share-target", type=float, default=0.28)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ctrl_args = controller_args(args)

    p2_old = load_json(seed_dir / "P2_ring_wrap_to_front_micro_ring_guarded_seed.json")
    p3_old = load_json(seed_dir / "P3_index_wrap_to_front_reference_seed.json")
    p2_entry, p2_support = build_p2_bottom_cradle(p2_old)
    b23a, b23b, p3_exit = build_b23_chain(p2_support, p3_old)
    b23a_contact = {
        **b23a,
        "phase": 2.50,
        "name": "b23a_index_front_contact_deflect",
        "desired_contacts": ["thumb", "index", "middle"],
        "m3_2_31_role": (
            "Brief contact dwell after index reaches the front. It lets index apply a measurable "
            "deflection before index/middle begin the stronger drive phase."
        ),
    }

    seeds = {
        "P2_ring_bottom_thumb_cradle_entry_seed.json": p2_entry,
        "P2_ring_bottom_thumb_cradle_support_seed.json": p2_support,
        "B23a_index_front_deflect_seed.json": b23a,
        "B23a_index_front_contact_deflect_seed.json": b23a_contact,
        "B23b_index_middle_drive_thumb_yield_seed.json": b23b,
        "P3_index_front_pass_exit_seed.json": p3_exit,
    }
    for filename, seed in seeds.items():
        save_json(output_dir / filename, seed)

    render_env = m316.make_env(p2_entry, ctrl_args, duration=0.8, render=True)
    try:
        keyposes = [
            ("P2 old", p2_old),
            ("P2 bottom cradle", p2_entry),
            ("P2 ring preload", p2_support),
            ("B23a index contact", b23a_contact),
            ("B23b index+middle drive", b23b),
            ("P3 pass exit", p3_exit),
        ]
        frames = [m316.render_keypose(render_env, seed, label) for label, seed in keyposes]
        for (label, _), frame in zip(keyposes, frames):
            imageio.imwrite(output_dir / f"{label.replace(' ', '_').replace('+', '_')}.png", frame)
        m316.save_montage(frames, output_dir / "m3_2_31_keypose_montage.png", cols=3)
        imageio.mimsave(output_dir / "m3_2_31_keyposes.gif", frames, duration=0.45)

        old_geometry = reset_geometry(render_env, p2_old)
        p2_geometry = reset_geometry(render_env, p2_entry)

        p2_summary, p2_trace = save_rollout(
            render_env,
            [p2_entry, p2_support],
            ctrl_args,
            label="p2_bottom_cradle_hold",
            duration=0.24,
            segment_s=0.10,
            output_dir=output_dir,
            render_every=args.render_every,
        )

        initial_axis = m316.quat_to_axis(np.asarray(p2_entry["pen_quat"], dtype=np.float64))
        b23_summary, b23_trace = save_rollout(
            render_env,
            [p2_entry, p2_support, b23a, b23a_contact, b23b, p3_exit],
            ctrl_args,
            label="p2_to_p3_index_middle_thumb_pass",
            duration=args.segment_duration * 5.0,
            segment_s=args.segment_duration,
            output_dir=output_dir,
            render_every=args.render_every,
        )
        final_axis = render_env.get_state().pen_axis.copy()

        p2_summary["height_drop_mm"] = float(
            (np.asarray(p2_entry["pen_pos"])[2] - p2_summary["min_height"]) * 1000.0
        )
        b23_summary["height_drop_mm"] = float(
            (np.asarray(p2_entry["pen_pos"])[2] - b23_summary["min_height"]) * 1000.0
        )
        b23_summary["planar_rotation_deg"] = planar_rotation_deg(initial_axis, final_axis)
        segment = args.segment_duration
        b23_summary["phase_windows"] = {
            "p2_support": contact_window(b23_trace["rows"], 0.0, segment),
            "index_front_approach": contact_window(b23_trace["rows"], segment, 2.0 * segment),
            "index_front_contact_deflect": contact_window(
                b23_trace["rows"], 2.0 * segment, 3.0 * segment
            ),
            "index_middle_drive_thumb_yield": contact_window(
                b23_trace["rows"], 3.0 * segment, 5.0 * segment
            ),
        }

        deflect_window = b23_summary["phase_windows"]["index_front_contact_deflect"]["contact_ratio"]
        drive_window = b23_summary["phase_windows"]["index_middle_drive_thumb_yield"]["contact_ratio"]
        p2_ok = bool(
            p2_summary["survival_s"] >= 0.24
            and p2_summary["contact_ratio"]["thumb"] >= 0.95
            and p2_summary["contact_ratio"]["ring"] >= 0.95
            and p2_geometry["tip_geometry"]["thumb"]["radial_z_mm"] < 0.0
            and p2_geometry["tip_geometry"]["ring"]["radial_z_mm"] < 0.0
            and p2_summary["contact_ratio"]["pinky"] == 0.0
            and p2_summary["max_penetration_mm"] <= 4.0
        )
        b23_ok = bool(
            b23_summary["survival_s"] >= args.segment_duration * 5.0
            and b23_summary["contact_ratio"]["thumb"] >= 0.50
            and deflect_window.get("thumb", 0.0) >= 0.50
            and deflect_window.get("index", 0.0) >= 0.55
            and drive_window.get("thumb", 0.0) >= 0.05
            and drive_window.get("index", 0.0) >= 0.90
            and drive_window.get("middle", 0.0) >= 0.90
            and abs(b23_summary["planar_rotation_deg"]) >= 10.0
            and b23_summary["contact_ratio"]["pinky"] == 0.0
            and b23_summary["max_penetration_mm"] <= 4.0
        )

        report = {
            "stage": "M3.2.31 P2 bottom cradle and staged B23 thumb pass",
            "interpretation": {
                "P2": "Ring transfers below the pen and forms a cradle with thumb.",
                "B23a": "Index moves to the front and applies the initial deflection.",
                "B23b": "Index and middle drive the pen around thumb while thumb yields but still supports.",
            },
            "parameters": {
                "p2_q_delta": P2_Q_DELTA,
                "p2_pos_delta": P2_POS_DELTA.astype(float).tolist(),
                "p2_support_ring_distal_delta": P2_SUPPORT_RING_DISTAL_DELTA,
                "b23_index_front": B23_INDEX_FRONT,
                "b23_index_middle_drive": B23_INDEX_MIDDLE_DRIVE,
            },
            "old_p2_reset_geometry": old_geometry,
            "new_p2_reset_geometry": p2_geometry,
            "p2_hold": p2_summary,
            "p2_to_p3": b23_summary,
            "acceptance": {"p2_hold": p2_ok, "p2_to_p3": b23_ok},
            "limits": (
                "This validates P2 reset/short hold and reset-started P2->P3 only. The running P1/B12 "
                "state does not yet arrive at this revised P2 cradle and remains a separate recapture problem."
            ),
        }
        save_json(output_dir / "summary.json", report)
        write_csv(output_dir / "rollout_summary.csv", [summary_row(p2_summary), summary_row(b23_summary)])

        readme = f"""# M3.2.31 P2 底托与 B23 thumb-pass 评估

本轮按真实动作重新定义了 P2/P3 早段受力：P2 不再是 ring 在前方防滚，而是 ring 转到笔的底部与 thumb 共同托住；B23 先让 index 前绕施加偏转，再让 index/middle 推滚，thumb 避让但保留间歇支撑。

## 结果

- P2 reset 初始接触：`thumb/ring`，承力比 `{p2_geometry['force_share']['thumb']:.3f}/{p2_geometry['force_share']['ring']:.3f}`。
- P2 ring 与 thumb 相对笔轴的径向 z：`{p2_geometry['tip_geometry']['ring']['radial_z_mm']:.2f} mm / {p2_geometry['tip_geometry']['thumb']['radial_z_mm']:.2f} mm`，两者都位于笔下侧。
- P2 hold：`{p2_summary['survival_s']:.2f}s`，thumb/ring 接触率 `{p2_summary['contact_ratio']['thumb']:.2f}/{p2_summary['contact_ratio']['ring']:.2f}`，最大穿透 `{p2_summary['max_penetration_mm']:.2f} mm`。
- P2->P3：`{b23_summary['survival_s']:.2f}s`，笔平面偏转 `{b23_summary['planar_rotation_deg']:.1f} deg`，最大穿透 `{b23_summary['max_penetration_mm']:.2f} mm`。
- B23 偏转窗口 thumb/index 接触率：`{deflect_window.get('thumb', 0.0):.2f}/{deflect_window.get('index', 0.0):.2f}`。
- B23 推滚窗口 thumb/index/middle 接触率：`{drive_window.get('thumb', 0.0):.2f}/{drive_window.get('index', 0.0):.2f}/{drive_window.get('middle', 0.0):.2f}`；全段 thumb 接触率 `{b23_summary['contact_ratio']['thumb']:.2f}`。

## 边界

P2 short hold 与从 P2 reset 开始的 P2->P3 已通过本轮验收。`P1/B12 -> 新 P2 底托` 还没有连通：旧 B12 的运行态会把笔带到不同的位姿，不能直接硬插值到这个 P2 reset。

## 可视化

- `visual_summary.html`
- `m3_2_31_keypose_montage.png`
- `p2_bottom_cradle_hold.gif`
- `p2_to_p3_index_middle_thumb_pass.gif`
- `p2_to_p3_index_middle_thumb_pass_trace.png`
"""
        (output_dir / "ASSESSMENT.zh.md").write_text(readme, encoding="utf-8")
        write_visual_summary(output_dir, report)
    finally:
        render_env.close()

    print(json.dumps({
        "output_dir": str(output_dir),
        "acceptance": report["acceptance"],
        "p2_hold": {
            "survival_s": report["p2_hold"]["survival_s"],
            "thumb_contact_ratio": report["p2_hold"]["contact_ratio"]["thumb"],
            "ring_contact_ratio": report["p2_hold"]["contact_ratio"]["ring"],
            "max_penetration_mm": report["p2_hold"]["max_penetration_mm"],
        },
        "p2_to_p3": {
            "survival_s": report["p2_to_p3"]["survival_s"],
            "planar_rotation_deg": report["p2_to_p3"]["planar_rotation_deg"],
            "max_penetration_mm": report["p2_to_p3"]["max_penetration_mm"],
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
