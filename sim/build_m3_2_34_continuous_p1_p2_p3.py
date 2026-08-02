from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

import build_m3_2_33_narrow_021_five_step_chain as m333
import evaluate_m3_2_17_cycle_closure as m317
import export_m3_2_23_dynamic_state_cache as m323
import rebuild_m3_2_16_p4_p5_transitions as m316
from evaluate_m3_2_15_short_primitives import FINGERS, contact_metrics

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


SELECTED_B12 = m333.B12Candidate("bs7_m1.00_t1.40_r1", 7, 1.0, 1.40, 1)
SELECTED_B23 = m333.B23Candidate(
    "i1.80_m1.10_t0.20_d-0.040_s0.88",
    1.80,
    1.10,
    0.20,
    -0.040,
    -0.020,
    0.88,
)
DEFAULT_P1_SAFETY_OPEN = {
    "index": -0.012,
    "middle": -0.004,
    "ring": -0.020,
    "thumb": 0.0,
}


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


def source_controller_args(args: argparse.Namespace) -> argparse.Namespace:
    source = argparse.Namespace(
        seed=args.seed,
        action_delta_limit=0.026,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=0.0,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
        tracking_gain=0.62,
        ramp_seconds=0.10,
        initial_thumb_bias=0.035,
        initial_ring_open=0.0,
        initial_shelf_bias=0.018,
        thumb_share_target=0.32,
        ring_share_max=0.25,
        ring_share_min=0.0,
        axis_z_guard=0.17,
        height_drop_guard=0.006,
        thumb_bias_min=0.0,
        thumb_bias_max=0.13,
        ring_open_min=0.0,
        ring_open_max=0.05,
        ring_guard_max=0.10,
        shelf_bias_min=0.0,
        shelf_bias_max=0.05,
        thumb_gain=0.065,
        ring_open_gain=0.035,
        ring_guard_gain=0.08,
        ring_guard_decay=1.0,
        shelf_gain=0.045,
        height_gain=1.2,
    )
    return m317.make_controller_args(source)


def model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        action_delta_limit=0.040,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=0.0,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
    )


def apply_p1_safety_open(
    route: list[dict[str, Any]], safety_open: dict[str, float]
) -> list[dict[str, Any]]:
    safe_route = [dict(seed) for seed in route]
    p1 = dict(safe_route[0])
    q = np.asarray(p1["hand_qpos"], dtype=np.float64).copy()
    q[1] += safety_open["index"]
    q[2] += 0.70 * safety_open["index"]
    q[3] = q[2]
    q[5] += safety_open["middle"]
    q[6] += 0.70 * safety_open["middle"]
    q[7] = q[6]
    q[9] += 0.50 * safety_open["ring"]
    q[10] += safety_open["ring"]
    q[11] = q[10]
    q[16:20] += safety_open["thumb"] * np.asarray(
        [0.04, 0.18, 1.00, 0.80], dtype=np.float64
    )
    p1["name"] = "four_finger_clamp_micro_thumb_supported_safety_open"
    p1["hand_qpos"] = q.astype(float).tolist()
    p1["target_qpos"] = q.astype(float).tolist()
    p1["m3_2_34_safety_open"] = dict(safety_open)
    safe_route[0] = p1
    return safe_route


def overlay(env: Any, label: str) -> np.ndarray:
    return m316.overlay_frame(env.render(), label)


def prelude_rollout(
    env: Any,
    route: list[dict[str, Any]],
    initial_seed: dict[str, Any],
    controller_args: argparse.Namespace,
    args: argparse.Namespace,
    *,
    render: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env.reset(seed=args.seed)
    m316.reset_to_seed(env, initial_seed)
    initial_state = env.get_state()
    initial_metrics = contact_metrics(env)
    frames: list[np.ndarray] = []
    key_frames: dict[str, np.ndarray] = {}
    if render:
        key_frames["01_P1_start"] = overlay(
            env,
            f"P1 start | contacts={','.join(sorted(initial_metrics['fingers'])) or 'none'}",
        )
    ctrl = m316.ControllerState(
        thumb_bias=controller_args.initial_thumb_bias,
        ring_open=controller_args.initial_ring_open,
        ring_guard=0.0,
        shelf_bias=controller_args.initial_shelf_bias,
    )
    steps = int(round(args.source_t * env.control_hz)) + 1
    steps_per_segment = int(round(args.source_segment_duration * env.control_hz))
    z0 = float(initial_state.pen_center[2])
    dt = 1.0 / env.control_hz
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    terminated = False
    truncated = False

    for step in range(steps):
        state = env.get_state()
        metrics = contact_metrics(env)
        m316.update_controller(ctrl, metrics, state, z0, controller_args, dt)
        ramp = min(1.0, (step + 1) / max(1.0, controller_args.ramp_seconds * env.control_hz))
        base_q = m316.interpolate_route_q(route, step, steps_per_segment)
        target = m316.apply_controller(base_q, ctrl, env, ramp)
        action = np.clip(
            controller_args.tracking_gain * (target - env.target_qpos.astype(np.float64)),
            -controller_args.action_delta_limit,
            controller_args.action_delta_limit,
        ).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        state = env.get_state()
        metrics = contact_metrics(env)
        contacts = ",".join(sorted(metrics["fingers"])) or "none"
        penetration_mm = float(metrics["max_penetration"] * 1000.0)
        cache_t = float(step / env.control_hz)
        row: dict[str, Any] = {
            "step": step,
            "t": float((step + 1) / env.control_hz),
            "cache_t": cache_t,
            "stage": "P1_teacher_to_B12_running",
            "contacts": contacts,
            "pen_linear_speed": float(
                np.linalg.norm(env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 3])
            ),
            "pen_angular_speed": float(
                np.linalg.norm(env.data.qvel[env.pen_qvel_adr + 3 : env.pen_qvel_adr + 6])
            ),
            "max_penetration_mm": penetration_mm,
        }
        for finger in FINGERS:
            row[f"{finger}_share"] = float(metrics["force_share"].get(finger, 0.0))
        rows.append(row)
        snapshots.append(m333.snapshot(env, contacts, penetration_mm, row["t"]))

        label = (
            f"P1 teacher -> B12 running | cache t={cache_t:.2f}s | {contacts} | "
            f"thumb/middle={row['thumb_share']:.2f}/{row['middle_share']:.2f}"
        )
        if render:
            if step % args.render_every == 0 or step == steps - 1:
                frames.append(overlay(env, label))
            if step == steps - 1:
                key_frames["02_B12_running_021"] = overlay(env, label)
        if terminated or truncated:
            break

    summary = {
        "steps": len(rows),
        "expected_steps": steps,
        "survival_s": float(len(rows) / env.control_hz),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "initial_contacts": sorted(initial_metrics["fingers"]),
        "final_contacts": [] if not rows or rows[-1]["contacts"] == "none" else rows[-1]["contacts"].split(","),
        "max_penetration_mm": float(max(row["max_penetration_mm"] for row in rows)),
    }
    return summary, {
        "rows": rows,
        "snapshots": snapshots,
        "frames": frames,
        "key_frames": key_frames,
        "initial_center": initial_state.pen_center.copy(),
        "initial_axis": initial_state.pen_axis.copy(),
    }


def source_match(env: Any, cache: Any, cache_index: int) -> dict[str, Any]:
    metrics = contact_metrics(env)
    return {
        "qpos_max_abs_error": float(np.max(np.abs(env.data.qpos - cache["qpos"][cache_index]))),
        "qvel_max_abs_error": float(np.max(np.abs(env.data.qvel - cache["qvel"][cache_index]))),
        "target_qpos_max_abs_error": float(
            np.max(np.abs(env.target_qpos.astype(np.float64) - cache["target_qpos"][cache_index]))
        ),
        "rotation_phase_abs_error": float(
            abs(float(env.get_state().rotation_phase) - float(cache["rotation_phase"][cache_index]))
        ),
        "cache_contacts": str(cache["contacts"][cache_index]),
        "live_contacts": ",".join(sorted(metrics["fingers"])) or "none",
    }


def combine_rows(
    segments: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
    initial_center: np.ndarray,
    initial_axis: np.ndarray,
    control_hz: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    global_step = 0
    for segment_name, rows, snapshots in segments:
        for row, state in zip(rows, snapshots):
            center = np.asarray(state["pen_center"], dtype=np.float64)
            displacement = (center - initial_center) * 1000.0
            combined = dict(row)
            combined["segment"] = segment_name
            combined["segment_t"] = float(row["t"])
            combined["step"] = global_step
            combined["t"] = float((global_step + 1) / control_hz)
            combined["pen_x"] = float(center[0])
            combined["pen_y"] = float(center[1])
            combined["pen_z"] = float(center[2])
            combined["dx_mm"] = float(displacement[0])
            combined["dy_mm"] = float(displacement[1])
            combined["dz_mm"] = float(displacement[2])
            combined["planar_rotation_deg"] = m333.signed_planar_rotation_deg(
                initial_axis, np.asarray(state["pen_axis"], dtype=np.float64)
            )
            result.append(combined)
            global_step += 1
    return result


def max_no_contact_gap(rows: list[dict[str, Any]]) -> int:
    maximum = 0
    current = 0
    for row in rows:
        current = current + 1 if row["contacts"] == "none" else 0
        maximum = max(maximum, current)
    return maximum


def save_trace_plot(path: Path, rows: list[dict[str, Any]], boundaries: list[float]) -> None:
    if plt is None or not rows:
        return
    t = np.asarray([row["t"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
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
    axes[2].plot(t, [row["pen_linear_speed"] for row in rows], label="linear speed")
    axes[2].plot(t, [row["pen_angular_speed"] for row in rows], label="angular speed")
    axes[2].plot(t, [row["max_penetration_mm"] for row in rows], label="penetration mm")
    axes[2].set_ylabel("speed / penetration")
    axes[2].legend(ncol=3)
    axes[3].plot(t, [row["planar_rotation_deg"] for row in rows], label="planar rotation")
    axes[3].axhline(-4.0, color="gray", linestyle="--", linewidth=0.8)
    axes[3].axhline(-7.0, color="gray", linestyle=":", linewidth=0.8)
    axes[3].set_ylabel("rotation (deg)")
    axes[3].set_xlabel("continuous chain time (s)")
    axes[3].legend()
    for axis in axes:
        for boundary in boundaries:
            axis.axvline(boundary, color="black", linestyle="--", linewidth=0.8)
    fig.suptitle("M3.2.34 continuous P1 -> P2 -> P3-entry (no reset)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_final_cache(path: Path, state: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        t=np.asarray([state["t"]], dtype=np.float64),
        qpos=np.asarray([state["qpos"]], dtype=np.float64),
        qvel=np.asarray([state["qvel"]], dtype=np.float64),
        target_qpos=np.asarray([state["target_qpos"]], dtype=np.float64),
        hand_qpos=np.asarray([state["hand_qpos"]], dtype=np.float64),
        hand_qvel=np.asarray([state["hand_qvel"]], dtype=np.float64),
        pen_pos=np.asarray([state["pen_pos"]], dtype=np.float64),
        pen_quat=np.asarray([state["pen_quat"]], dtype=np.float64),
        pen_qvel=np.asarray([state["pen_qvel"]], dtype=np.float64),
        pen_center=np.asarray([state["pen_center"]], dtype=np.float64),
        pen_axis=np.asarray([state["pen_axis"]], dtype=np.float64),
        rotation_phase=np.asarray([state["rotation_phase"]], dtype=np.float64),
        contacts=np.asarray([state["contacts"]]),
        max_penetration_mm=np.asarray([state["max_penetration_mm"]], dtype=np.float64),
    )


def write_visual_summary(output_dir: Path, report: dict[str, Any]) -> None:
    prelude = report["p1_to_running"]
    b12 = report["running_to_p2"]
    b23 = report["p2_to_p3_entry"]
    chain = report["continuous_chain"]
    ring_under = b12["first_ring_under_s"]
    ring_under_text = "none" if ring_under is None else f"{ring_under:.2f}s"
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M3.2.34 continuous P1 P2 P3</title>
<style>
body {{ margin:0; background:#f4f5f7; color:#202124; font-family:sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 8px; font-size:24px; }} p {{ line-height:1.55; }}
.status {{ padding:12px 0; border-top:1px solid #c7cbd1; border-bottom:1px solid #c7cbd1; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:18px; }}
figure {{ margin:0; overflow:hidden; background:white; border:1px solid #d6dae0; border-radius:6px; }}
img {{ display:block; width:100%; height:auto; }} figcaption {{ padding:10px 12px; font-size:14px; line-height:1.45; }}
table {{ width:100%; margin-top:18px; border-collapse:collapse; background:white; }}
th,td {{ padding:8px 10px; border:1px solid #d6dae0; text-align:left; font-size:14px; }}
@media(max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} main {{ padding:14px; }} }}
</style></head><body><main>
<h1>P1 → P2 → P3-entry 连续 MuJoCo 链路</h1>
<p class="status"><strong>{'PASS' if report['acceptance']['continuous_p1_p2_p3_entry'] else 'CHECK'}</strong>：同一个 env 连续运行 {chain['survival_s']:.2f}s，边界 reset 次数为 0；P1 使用四指安全微开起点，整链最大穿透 {chain['max_penetration_mm']:.2f}mm。</p>
<table><tr><th>阶段</th><th>结果</th><th>指标</th></tr>
<tr><td>P1 → 0.21s running</td><td>{'PASS' if report['acceptance']['p1_safe_to_running'] else 'CHECK'}</td><td>{prelude['survival_s']:.2f}s；穿透 {prelude['max_penetration_mm']:.2f}mm；末接触 {','.join(prelude['final_contacts']) or 'none'}</td></tr>
<tr><td>running → projected P2</td><td>{'PASS' if all(b12['acceptance'].values()) else 'CHECK'}</td><td>ring-under {ring_under_text}；穿透 {b12['max_penetration_mm']:.2f}mm</td></tr>
<tr><td>P2 → B23/P3-entry</td><td>{'PASS' if b23['passed'] else 'CHECK'}</td><td>偏转 {b23['planar_rotation_deg']:.1f}°；穿透 {b23['max_penetration_mm']:.2f}mm；末接触 {','.join(b23['final_contacts'])}</td></tr>
</table>
<div class="grid">
<figure><img src="continuous_P1_to_P2_to_P3.gif"><figcaption>完整连续回放：P1 teacher 前段、thumb-brake/ring-under P2、index/middle/thumb B23/P3-entry。中间没有 reset 或 qvel 清零。</figcaption></figure>
<figure><img src="continuous_P1_P2_P3_keyframes.png"><figcaption>P1 起点、0.21s running-state、P2 下托再捕获和 P3-entry 的关键帧。</figcaption></figure>
<figure><img src="continuous_P1_to_P2_to_P3_trace.png"><figcaption>全链路位移、承力、速度/穿透和累计平面偏转；虚线是控制器切换边界。</figcaption></figure>
</div></main></body></html>"""
    (output_dir / "visual_summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--cache",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/"
            "dynamic_state_cache/earlier_pre_ring_for_B12b_to_B23.npz"
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
            "continuous_p1_p2_p3_m3_2_34"
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
    parser.add_argument(
        "--p1-index-open", type=float, default=DEFAULT_P1_SAFETY_OPEN["index"]
    )
    parser.add_argument(
        "--p1-middle-open", type=float, default=DEFAULT_P1_SAFETY_OPEN["middle"]
    )
    parser.add_argument(
        "--p1-ring-open", type=float, default=DEFAULT_P1_SAFETY_OPEN["ring"]
    )
    parser.add_argument(
        "--p1-thumb-open", type=float, default=DEFAULT_P1_SAFETY_OPEN["thumb"]
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(Path(args.cache))
    cache_index = m333.m332.cache_index(cache, args.source_t)
    source_route = m323.build_route(Path(args.seed_dir))
    safety_open = {
        "index": float(args.p1_index_open),
        "middle": float(args.p1_middle_open),
        "ring": float(args.p1_ring_open),
        "thumb": float(args.p1_thumb_open),
    }
    safety_route = apply_p1_safety_open(source_route, safety_open)
    initial_seed = safety_route[0]
    route = source_route
    controller_args = source_controller_args(args)
    env = m316.make_env(initial_seed, model_args(args), duration=1.8, render=True)

    reference_dir = Path(args.b23_reference_dir)
    reference_files = {
        "p2": "P2_ring_bottom_thumb_cradle_support_seed.json",
        "b23a": "B23a_index_front_deflect_seed.json",
        "b23a_contact": "B23a_index_front_contact_deflect_seed.json",
        "b23b": "B23b_index_middle_drive_thumb_yield_seed.json",
        "p3": "P3_index_front_pass_exit_seed.json",
    }
    references = {
        key: np.asarray(
            json.loads((reference_dir / filename).read_text(encoding="utf-8"))["hand_qpos"],
            dtype=np.float64,
        )
        for key, filename in reference_files.items()
    }

    try:
        prelude_summary, prelude_trace = prelude_rollout(
            env, route, initial_seed, controller_args, args, render=True
        )
        match = source_match(env, cache, cache_index)
        step_count_after_prelude = int(env.step_count)

        b12_summary, b12_trace = m333.b12_rollout(
            env,
            cache,
            cache_index,
            m333.Perturbation("continuous_nominal"),
            SELECTED_B12,
            args,
            duration_s=args.b12_duration,
            render=True,
            reset_state=False,
        )
        step_count_after_b12 = int(env.step_count)
        b12_arrival = b12_trace["snapshots"][-1]
        b23_summary, b23_trace = m333.b23_rollout(
            env,
            b12_arrival,
            references,
            SELECTED_B23,
            args,
            label="continuous_P2_arrival",
            render=True,
            reset_state=False,
        )
        step_count_after_b23 = int(env.step_count)
    finally:
        env.close()

    combined_rows = combine_rows(
        [
            ("P1_to_running", prelude_trace["rows"], prelude_trace["snapshots"]),
            ("running_to_P2", b12_trace["rows"], b12_trace["snapshots"]),
            ("P2_to_P3_entry", b23_trace["rows"], b23_trace["snapshots"]),
        ],
        np.asarray(prelude_trace["initial_center"], dtype=np.float64),
        np.asarray(prelude_trace["initial_axis"], dtype=np.float64),
        100.0,
    )
    frames = prelude_trace["frames"] + b12_trace["frames"] + b23_trace["frames"]
    if frames:
        imageio.mimsave(
            output_dir / "continuous_P1_to_P2_to_P3.gif",
            frames,
            duration=args.render_every / 100.0,
        )
    key_frames = {
        **prelude_trace["key_frames"],
        **{f"b12_{key}": value for key, value in b12_trace["key_frames"].items()},
        **{f"b23_{key}": value for key, value in b23_trace["key_frames"].items()},
    }
    for name, frame in key_frames.items():
        imageio.imwrite(output_dir / f"{name}.png", frame)
    if key_frames:
        m316.save_montage(
            list(key_frames.values()), output_dir / "continuous_P1_P2_P3_keyframes.png", cols=3
        )

    prelude_boundary = prelude_summary["steps"] / 100.0
    b12_boundary = prelude_boundary + len(b12_trace["rows"]) / 100.0
    write_csv(output_dir / "continuous_P1_to_P2_to_P3_trace.csv", combined_rows)
    save_trace_plot(
        output_dir / "continuous_P1_to_P2_to_P3_trace.png",
        combined_rows,
        [prelude_boundary, b12_boundary],
    )
    if b23_trace["snapshots"]:
        save_final_cache(
            output_dir / "continuous_P3_entry_arrival.npz", b23_trace["snapshots"][-1]
        )

    expected_prelude_steps = prelude_summary["expected_steps"]
    expected_b12_steps = int(round(args.b12_duration * 100.0))
    expected_b23_steps = sum(steps for _, steps, _ in b23_trace["stages"])
    p1_safe_to_running = bool(
        prelude_summary["steps"] == prelude_summary["expected_steps"]
        and not prelude_summary["terminated"]
        and prelude_summary["max_penetration_mm"] <= 4.0
        and "thumb" in prelude_summary["initial_contacts"]
        and len(prelude_summary["initial_contacts"]) >= 3
    )
    continuous_steps = bool(
        step_count_after_prelude == expected_prelude_steps
        and step_count_after_b12 == expected_prelude_steps + expected_b12_steps
        and step_count_after_b23
        == expected_prelude_steps + expected_b12_steps + expected_b23_steps
    )
    chain_summary = {
        "steps": len(combined_rows),
        "survival_s": float(len(combined_rows) / 100.0),
        "step_count_after_prelude": step_count_after_prelude,
        "step_count_after_b12": step_count_after_b12,
        "step_count_after_b23": step_count_after_b23,
        "boundary_reset_count": 0,
        "max_penetration_mm": float(max(row["max_penetration_mm"] for row in combined_rows)),
        "max_no_contact_gap_steps": max_no_contact_gap(combined_rows),
        "min_height_delta_mm": float(min(row["dz_mm"] for row in combined_rows)),
        "final_contacts": b23_summary["final_contacts"],
        "total_planar_rotation_deg": float(combined_rows[-1]["planar_rotation_deg"]),
    }
    acceptance = {
        "p1_safe_to_running": p1_safe_to_running,
        "continuous_step_count": continuous_steps,
        "running_to_p2": bool(all(b12_summary["acceptance"].values())),
        "p2_to_p3_entry": bool(b23_summary["passed"]),
    }
    acceptance["continuous_p1_p2_p3_entry"] = bool(
        all(acceptance.values())
        and chain_summary["max_penetration_mm"] <= 4.0
        and chain_summary["max_no_contact_gap_steps"] <= 2
    )
    report = {
        "stage": "M3.2.34 continuous P1 to P2 to P3-entry",
        "source_route": [f"P{seed['phase']}:{seed['name']}" for seed in source_route],
        "executed_initial_seed": f"P{initial_seed['phase']}:{initial_seed['name']}",
        "executed_teacher_route": [f"P{seed['phase']}:{seed['name']}" for seed in route],
        "source_cache": str(args.cache),
        "source_t": float(args.source_t),
        "p1_safety_open": dict(safety_open),
        "controllers": {
            "p1_teacher": "M3.2.23 source scripted teacher/controller",
            "b12": SELECTED_B12.__dict__,
            "b23": SELECTED_B23.__dict__,
        },
        "model_parameters": {
            "tip_proxy_radius_m": float(args.tip_proxy_radius),
            "tip_proxy_friction": float(args.tip_proxy_friction),
            "actuator_kp": float(args.actuator_kp),
            "actuator_kv": float(args.actuator_kv),
        },
        "source_match": match,
        "p1_to_running": prelude_summary,
        "running_to_p2": b12_summary,
        "p2_to_p3_entry": b23_summary,
        "continuous_chain": chain_summary,
        "acceptance": acceptance,
        "limits": (
            "This is a genuine no-reset P1->P2->B23/P3-entry rollout. P1 uses a small safety-open "
            "micro-adjustment, so the 0.21s state is reported against the old cache as a diagnostic "
            "rather than required to match exactly. The final state is the short P3-entry criterion "
            "(>=4 degrees), not the full >=7-degree P3 deflection."
        ),
    }
    save_json(output_dir / "summary.json", report)
    write_visual_summary(output_dir, report)
    assessment = f"""# M3.2.34 continuous P1 -> P2 -> P3-entry

- One MuJoCo environment, no state reset at the 0.21s or P2 boundary.
- P1 safety-open deltas: `{safety_open}`; source-cache diagnostic error at 0.21s: qpos `{match['qpos_max_abs_error']:.3e}`, qvel `{match['qvel_max_abs_error']:.3e}`.
- Continuous survival: `{chain_summary['survival_s']:.2f}s`; step counters `{step_count_after_prelude} -> {step_count_after_b12} -> {step_count_after_b23}`.
- B12 acceptance: `{b12_summary['acceptance']}`.
- B23/P3-entry acceptance: `{b23_summary['passed']}`; rotation `{b23_summary['planar_rotation_deg']:.2f}` deg; final contacts `{','.join(b23_summary['final_contacts'])}`.
- Whole-chain max penetration: `{chain_summary['max_penetration_mm']:.2f}` mm; max no-contact gap `{chain_summary['max_no_contact_gap_steps']}` steps.

The final state remains a short P3-entry. Full 7-degree deflection and a no-reset P3-entry->P4 relay are still open.
"""
    (output_dir / "ASSESSMENT.zh.md").write_text(assessment, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "acceptance": acceptance,
                "source_match": match,
                "continuous_chain": chain_summary,
                "b12": b12_summary["acceptance"],
                "b23_passed": b23_summary["passed"],
                "b23_rotation_deg": b23_summary["planar_rotation_deg"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
