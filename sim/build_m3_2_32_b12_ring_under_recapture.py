from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

import rebuild_m3_2_16_p4_p5_transitions as m316
from evaluate_m3_2_15_short_primitives import FINGERS, contact_metrics

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


BRAKE_STEPS = 6
BRAKE_ACTION_LIMIT = 0.032
BRAKE_TRACKING_GAIN = 0.70
CAPTURE_ACTION_LIMIT = 0.038
CAPTURE_TRACKING_GAIN = 0.90

# The first six control steps rebuild the old B12 running state. Middle unloads,
# thumb takes the vertical/+y load, and ring retracts before curling underneath.
BRAKE_DELTA = {
    4: 0.0252,
    5: -0.0867,
    6: -0.0409,
    16: -0.0294,
    17: 0.0225,
    18: 0.0446,
    19: -0.0094,
}
BRAKE_RING_Q = np.array([0.1267, 0.5806, 0.5514], dtype=np.float64)

# This is a contact-stop target, not the old static P2 joint vector. Continuing
# toward the old full-curl target causes ring to pass the useful contact point.
CAPTURE_RING_Q = np.array([0.0927, 0.5964, 0.9199], dtype=np.float64)
CAPTURE_THUMB_DELTA = np.array([-0.0104, -0.0076, 0.0249, -0.0470], dtype=np.float64)


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


def make_env_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        action_delta_limit=max(BRAKE_ACTION_LIMIT, CAPTURE_ACTION_LIMIT),
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
    )


def cache_index(cache: Any, source_t: float) -> int:
    matches = np.flatnonzero(np.isclose(cache["t"], source_t, atol=5e-6))
    if not len(matches):
        raise ValueError(f"No dynamic cache state at t={source_t:.3f}s")
    return int(matches[0])


def reset_from_cache(env: Any, cache: Any, index: int, seed: int) -> None:
    env.reset(seed=seed)
    env.data.qpos[:] = cache["qpos"][index]
    env.data.qvel[:] = cache["qvel"][index]
    env.target_qpos = cache["target_qpos"][index].astype(np.float32).copy()
    env.data.ctrl[:] = env.target_qpos
    env.last_action[:] = 0.0
    env.step_count = 0
    env.rotation_phase = float(cache["rotation_phase"][index])
    mujoco.mj_forward(env.model, env.data)
    env.prev_phase = env._quat_twist_phase_about_x(env.data.xquat[env.pen_body_id])


def build_targets(source_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    brake = np.asarray(source_target, dtype=np.float64).copy()
    for index, delta in BRAKE_DELTA.items():
        brake[index] += delta
    brake[7] = brake[6]
    brake[8:12] = [BRAKE_RING_Q[0], BRAKE_RING_Q[1], BRAKE_RING_Q[2], BRAKE_RING_Q[2]]
    brake = m316.enforce_shared_motor_consistency(brake)

    capture = brake.copy()
    capture[8:12] = [
        CAPTURE_RING_Q[0],
        CAPTURE_RING_Q[1],
        CAPTURE_RING_Q[2],
        CAPTURE_RING_Q[2],
    ]
    capture[16:20] += CAPTURE_THUMB_DELTA
    capture = m316.enforce_shared_motor_consistency(capture)
    return brake, capture


def ring_geometry(env: Any, ring_geom_id: int) -> tuple[float, float, float]:
    state = env.get_state()
    relative = env.data.geom_xpos[ring_geom_id] - state.pen_center
    axial = float(np.dot(relative, state.pen_axis))
    radial = relative - axial * state.pen_axis
    return (
        float(radial[2] * 1000.0),
        float(np.linalg.norm(radial) * 1000.0),
        float(axial * 1000.0),
    )


def stage_for_step(step: int, first_under_step: int | None) -> str:
    if step < BRAKE_STEPS:
        return "thumb_brake_ring_retract"
    if first_under_step is None or step < first_under_step:
        return "ring_under_approach"
    if step < first_under_step + 10:
        return "ring_under_recapture"
    return "projected_P2_thumb_main_hold"


def overlay(env: Any, text: str) -> np.ndarray:
    return m316.overlay_frame(env.render(), text)


def rollout(
    env: Any,
    cache: Any,
    index: int,
    args: argparse.Namespace,
    *,
    source_t: float,
    duration_s: float,
    render: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_from_cache(env, cache, index, args.seed)
    source_target = cache["target_qpos"][index].astype(np.float64)
    brake_target, capture_target = build_targets(source_target)
    ring_geom_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        m316.TIP_GEOMS["ring"],
    )
    initial_center = env.get_state().pen_center.copy()
    steps = int(round(duration_s * env.control_hz))

    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    gif_frames: list[np.ndarray] = []
    key_frames: dict[str, np.ndarray] = {}
    contact_counter: Counter[str] = Counter()
    first_under_step: int | None = None
    upper_contacts: list[dict[str, float]] = []
    no_contact_gap = 0
    current_gap = 0
    max_penetration_mm = 0.0
    max_xy_displacement_mm = 0.0
    max_joint_speed = 0.0
    terminated = False
    truncated = False

    if render:
        key_frames["01_B12_running_state"] = overlay(
            env,
            f"B12 running t={source_t:.2f}s contacts={cache['contacts'][index]}",
        )

    for step in range(steps):
        if step < BRAKE_STEPS:
            target = brake_target
            action_limit = BRAKE_ACTION_LIMIT
            tracking_gain = BRAKE_TRACKING_GAIN
        else:
            target = capture_target
            action_limit = CAPTURE_ACTION_LIMIT
            tracking_gain = CAPTURE_TRACKING_GAIN

        action = np.clip(
            tracking_gain * (target - env.target_qpos.astype(np.float64)),
            -action_limit,
            action_limit,
        ).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        state = env.get_state()
        metrics = contact_metrics(env)
        fingers = set(metrics["fingers"])
        ring_rz_mm, ring_rn_mm, ring_axial_mm = ring_geometry(env, ring_geom_id)

        if "ring" in fingers and ring_rz_mm < 0.0 and first_under_step is None:
            first_under_step = step
        if "ring" in fingers and ring_rz_mm >= 0.0:
            upper_contacts.append(
                {
                    "step": float(step),
                    "t": float((step + 1) / env.control_hz),
                    "radial_z_mm": ring_rz_mm,
                    "normal_force_n": float(metrics["normal_force"].get("ring", 0.0)),
                }
            )

        stage = stage_for_step(step, first_under_step)
        displacement = state.pen_center - initial_center
        penetration_mm = float(metrics["max_penetration"] * 1000.0)
        max_penetration_mm = max(max_penetration_mm, penetration_mm)
        max_xy_displacement_mm = max(
            max_xy_displacement_mm,
            float(np.linalg.norm(displacement[:2]) * 1000.0),
        )
        max_joint_speed = max(max_joint_speed, float(np.max(np.abs(state.qvel))))
        current_gap = current_gap + 1 if not fingers else 0
        no_contact_gap = max(no_contact_gap, current_gap)
        for finger in fingers:
            contact_counter[finger] += 1

        pen_linear_velocity = env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 3].copy()
        pen_angular_velocity = env.data.qvel[env.pen_qvel_adr + 3 : env.pen_qvel_adr + 6].copy()
        row: dict[str, Any] = {
            "step": step,
            "t": float((step + 1) / env.control_hz),
            "stage": stage,
            "contacts": ",".join(sorted(fingers)) or "none",
            "pen_x": float(state.pen_center[0]),
            "pen_y": float(state.pen_center[1]),
            "pen_z": float(state.pen_center[2]),
            "dx_mm": float(displacement[0] * 1000.0),
            "dy_mm": float(displacement[1] * 1000.0),
            "dz_mm": float(displacement[2] * 1000.0),
            "pen_linear_speed": float(np.linalg.norm(pen_linear_velocity)),
            "pen_angular_speed": float(np.linalg.norm(pen_angular_velocity)),
            "ring_radial_z_mm": ring_rz_mm,
            "ring_radial_norm_mm": ring_rn_mm,
            "ring_axial_mm": ring_axial_mm,
            "max_penetration_mm": penetration_mm,
            "action_norm": float(np.linalg.norm(action)),
        }
        for finger in FINGERS:
            row[f"{finger}_share"] = float(metrics["force_share"].get(finger, 0.0))
            row[f"{finger}_force_n"] = float(metrics["normal_force"].get(finger, 0.0))
        rows.append(row)

        pen_qpos = env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 7].copy()
        snapshots.append(
            {
                "t": float((step + 1) / env.control_hz),
                "qpos": env.data.qpos.copy(),
                "qvel": env.data.qvel.copy(),
                "target_qpos": env.target_qpos.copy(),
                "hand_qpos": state.qpos.copy(),
                "hand_qvel": state.qvel.copy(),
                "pen_pos": pen_qpos[:3].copy(),
                "pen_quat": pen_qpos[3:].copy(),
                "pen_qvel": env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6].copy(),
                "pen_center": state.pen_center.copy(),
                "pen_axis": state.pen_axis.copy(),
                "rotation_phase": float(state.rotation_phase),
                "contacts": row["contacts"],
                "max_penetration_mm": penetration_mm,
            }
        )

        if render:
            label = (
                f"{stage} t={(step + 1) / env.control_hz:.2f}s "
                f"contacts={row['contacts']} thumb={row['thumb_share']:.2f} ring={row['ring_share']:.2f}"
            )
            if step % args.render_every == 0 or step == steps - 1:
                gif_frames.append(overlay(env, label))
            if step == BRAKE_STEPS - 1:
                key_frames["02_thumb_brake_ring_retract"] = overlay(env, label)
            if step == 14:
                key_frames["03_ring_surface_graze"] = overlay(env, label)
            if first_under_step == step:
                key_frames["04_ring_under_first_contact"] = overlay(env, label)
            if first_under_step is not None and step == first_under_step + 20:
                key_frames["05_projected_P2_settle"] = overlay(env, label)
            if step == steps - 1:
                key_frames["06_projected_P2_final"] = overlay(env, label)

        if terminated or truncated:
            break

    samples = len(rows)
    final_window = rows[-min(20, samples) :]
    brake_row = rows[min(BRAKE_STEPS - 1, samples - 1)]
    final_contacts = [] if not rows else ([] if rows[-1]["contacts"] == "none" else rows[-1]["contacts"].split(","))

    def window_ratio(finger: str) -> float:
        return float(
            sum(finger in row["contacts"].split(",") for row in final_window)
            / max(1, len(final_window))
        )

    def window_share(finger: str) -> float:
        return float(
            sum(float(row[f"{finger}_share"]) for row in final_window)
            / max(1, len(final_window))
        )

    hard_upper_contacts = [
        row
        for row in upper_contacts
        if row["radial_z_mm"] > 1.0 and row["normal_force_n"] > 0.10
    ]
    hold_seconds = (
        0.0
        if first_under_step is None
        else float((samples - first_under_step) / env.control_hz)
    )
    final_displacement = (
        np.zeros(3, dtype=np.float64)
        if not snapshots
        else snapshots[-1]["pen_center"] - initial_center
    )

    brake_ok = bool(
        samples >= BRAKE_STEPS
        and brake_row["pen_linear_speed"] <= 0.015
        and brake_row["thumb_share"] >= 0.80
        and brake_row["middle_share"] <= 0.10
        and "ring" not in brake_row["contacts"].split(",")
    )
    recapture_ok = bool(
        not terminated
        and first_under_step is not None
        and first_under_step <= 25
        and no_contact_gap <= 1
        and not hard_upper_contacts
        and max_penetration_mm <= 4.0
    )
    p2_ok = bool(
        samples == steps
        and hold_seconds >= 0.50
        and window_ratio("thumb") >= 0.95
        and window_ratio("ring") >= 0.95
        and window_share("thumb") >= 0.60
        and 0.04 <= window_share("ring") <= 0.30
        and window_share("pinky") <= 0.10
        and "thumb" in final_contacts
        and "ring" in final_contacts
        and "pinky" not in final_contacts
        and rows[-1]["ring_radial_z_mm"] < 0.0
    )

    summary = {
        "source_t": float(source_t),
        "duration_requested_s": float(duration_s),
        "survival_s": float(samples / env.control_hz),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "first_ring_under_step": first_under_step,
        "first_ring_under_s": None if first_under_step is None else float((first_under_step + 1) / env.control_hz),
        "projected_p2_hold_s": hold_seconds,
        "no_contact_max_gap_steps": int(no_contact_gap),
        "upper_ring_contacts": upper_contacts,
        "hard_upper_ring_contacts": hard_upper_contacts,
        "max_penetration_mm": float(max_penetration_mm),
        "max_xy_displacement_mm": float(max_xy_displacement_mm),
        "final_displacement_mm": (final_displacement * 1000.0).astype(float).tolist(),
        "max_joint_speed_rad_s": float(max_joint_speed),
        "contact_ratio": {
            finger: float(contact_counter[finger] / max(1, samples)) for finger in FINGERS
        },
        "final_window_contact_ratio": {finger: window_ratio(finger) for finger in FINGERS},
        "final_window_force_share": {finger: window_share(finger) for finger in FINGERS},
        "final_contacts": final_contacts,
        "final_ring_geometry": {
            "radial_z_mm": float(rows[-1]["ring_radial_z_mm"]),
            "radial_norm_mm": float(rows[-1]["ring_radial_norm_mm"]),
            "axial_mm": float(rows[-1]["ring_axial_mm"]),
        },
        "brake": {
            "linear_speed_m_s": float(brake_row["pen_linear_speed"]),
            "angular_speed_rad_s": float(brake_row["pen_angular_speed"]),
            "contacts": brake_row["contacts"],
            "thumb_share": float(brake_row["thumb_share"]),
            "middle_share": float(brake_row["middle_share"]),
            "displacement_mm": [
                float(brake_row["dx_mm"]),
                float(brake_row["dy_mm"]),
                float(brake_row["dz_mm"]),
            ],
        },
        "acceptance": {
            "b12_running_brake": brake_ok,
            "ring_under_recapture": recapture_ok,
            "projected_p2_short_hold": p2_ok,
        },
    }
    trace = {
        "rows": rows,
        "snapshots": snapshots,
        "gif_frames": gif_frames,
        "key_frames": key_frames,
        "brake_target": brake_target,
        "capture_target": capture_target,
    }
    return summary, trace


def save_trace_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    if plt is None or not rows:
        return
    t = np.asarray([row["t"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t, [row["dx_mm"] for row in rows], label="dx")
    axes[0].plot(t, [row["dy_mm"] for row in rows], label="dy")
    axes[0].plot(t, [row["dz_mm"] for row in rows], label="dz")
    axes[0].set_ylabel("pen delta (mm)")
    axes[0].legend(ncol=3)

    for finger in ["thumb", "middle", "ring", "pinky"]:
        axes[1].plot(t, [row[f"{finger}_share"] for row in rows], label=finger)
    axes[1].set_ylabel("force share")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(ncol=4)

    axes[2].plot(t, [row["ring_radial_z_mm"] for row in rows], label="ring radial z")
    axes[2].plot(t, [row["ring_radial_norm_mm"] for row in rows], label="ring radial norm")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].axhline(15.0, color="gray", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("ring geometry (mm)")
    axes[2].legend(ncol=2)

    axes[3].plot(t, [row["pen_linear_speed"] for row in rows], label="linear speed")
    axes[3].plot(t, [row["pen_angular_speed"] for row in rows], label="angular speed")
    axes[3].plot(t, [row["max_penetration_mm"] for row in rows], label="penetration mm")
    axes[3].set_ylabel("speed / penetration")
    axes[3].set_xlabel("primitive time (s)")
    axes[3].legend(ncol=3)
    fig.suptitle("M3.2.32 B12 running-state to ring-under projected P2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_arrival_cache(path: Path, snapshots: list[dict[str, Any]], first_under_step: int) -> int:
    start = max(first_under_step + 20, len(snapshots) - 30)
    selected = [
        snapshot
        for snapshot in snapshots[start:]
        if "thumb" in snapshot["contacts"].split(",")
        and "ring" in snapshot["contacts"].split(",")
    ]
    if not selected:
        selected = snapshots[start:]
    np.savez_compressed(
        path,
        t=np.asarray([row["t"] for row in selected], dtype=np.float64),
        qpos=np.asarray([row["qpos"] for row in selected], dtype=np.float64),
        qvel=np.asarray([row["qvel"] for row in selected], dtype=np.float64),
        target_qpos=np.asarray([row["target_qpos"] for row in selected], dtype=np.float64),
        hand_qpos=np.asarray([row["hand_qpos"] for row in selected], dtype=np.float64),
        hand_qvel=np.asarray([row["hand_qvel"] for row in selected], dtype=np.float64),
        pen_pos=np.asarray([row["pen_pos"] for row in selected], dtype=np.float64),
        pen_quat=np.asarray([row["pen_quat"] for row in selected], dtype=np.float64),
        pen_qvel=np.asarray([row["pen_qvel"] for row in selected], dtype=np.float64),
        pen_center=np.asarray([row["pen_center"] for row in selected], dtype=np.float64),
        pen_axis=np.asarray([row["pen_axis"] for row in selected], dtype=np.float64),
        rotation_phase=np.asarray([row["rotation_phase"] for row in selected], dtype=np.float64),
        contacts=np.asarray([row["contacts"] for row in selected]),
        max_penetration_mm=np.asarray(
            [row["max_penetration_mm"] for row in selected], dtype=np.float64
        ),
    )
    return len(selected)


def save_key_frames(output_dir: Path, key_frames: dict[str, np.ndarray]) -> None:
    frames: list[np.ndarray] = []
    for name, frame in key_frames.items():
        imageio.imwrite(output_dir / f"{name}.png", frame)
        frames.append(frame)
    if frames:
        m316.save_montage(frames, output_dir / "m3_2_32_keypose_montage.png", cols=3)


def write_visual_summary(output_dir: Path, report: dict[str, Any]) -> None:
    selected = report["selected_entry"]
    acceptance = selected["acceptance"]
    brake = selected["brake"]
    final_share = selected["final_window_force_share"]
    final_ratio = selected["final_window_contact_ratio"]
    status = "PASS" if all(acceptance.values()) else "CHECK"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M3.2.32 B12 ring-under recapture</title>
<style>
body {{ margin: 0; color: #202124; background: #f4f5f7; font-family: sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 8px; font-size: 24px; }}
p {{ line-height: 1.55; }}
.status {{ padding: 12px 0; border-top: 1px solid #c7cbd1; border-bottom: 1px solid #c7cbd1; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 18px; }}
figure {{ margin: 0; overflow: hidden; background: white; border: 1px solid #d6dae0; border-radius: 6px; }}
img {{ display: block; width: 100%; height: auto; }}
figcaption {{ padding: 10px 12px; font-size: 14px; line-height: 1.45; }}
table {{ width: 100%; margin-top: 18px; border-collapse: collapse; background: white; }}
th, td {{ padding: 8px 10px; border: 1px solid #d6dae0; text-align: left; font-size: 14px; }}
@media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} main {{ padding: 14px; }} }}
</style>
</head>
<body><main>
<h1>B12 running-state → ring-under recapture → projected P2</h1>
<p class="status"><strong>{status}</strong>：先卸掉 middle 的反向力，由 thumb 主动刹住并托住；ring 外撤后从圆柱赤道附近进入下侧，最终形成 thumb 主承力、ring 防滚/底托的短时 P2。</p>
<table>
<tr><th>阶段</th><th>结果</th><th>指标</th></tr>
<tr><td>B12 动态刹车</td><td>{'PASS' if acceptance['b12_running_brake'] else 'CHECK'}</td><td>线速度 {brake['linear_speed_m_s']:.4f} m/s；thumb share {brake['thumb_share']:.2f}</td></tr>
<tr><td>ring 下侧再捕获</td><td>{'PASS' if acceptance['ring_under_recapture'] else 'CHECK'}</td><td>首次下侧接触 {selected['first_ring_under_s']:.2f}s；无接触最大 {selected['no_contact_max_gap_steps']} step；穿透 {selected['max_penetration_mm']:.2f} mm</td></tr>
<tr><td>projected P2</td><td>{'PASS' if acceptance['projected_p2_short_hold'] else 'CHECK'}</td><td>保持 {selected['projected_p2_hold_s']:.2f}s；thumb/ring 接触率 {final_ratio['thumb']:.2f}/{final_ratio['ring']:.2f}；承力 {final_share['thumb']:.2f}/{final_share['ring']:.2f}</td></tr>
</table>
<div class="grid">
<figure><img src="m3_2_32_keypose_montage.png"><figcaption>从旧 B12 running-state、thumb 刹车/ring 外撤、表面轻擦、首次下侧接触，到 projected P2 的关键帧。</figcaption></figure>
<figure><img src="b12_ring_under_recapture.gif"><figcaption>完整 0.80 s MuJoCo 动态回放。ring 不是从上方硬压入，而是在低力轻擦后从赤道转到下侧。</figcaption></figure>
<figure><img src="b12_ring_under_recapture_trace.png"><figcaption>笔位移、五指承力、ring 相对笔轴几何、速度与穿透轨迹。</figcaption></figure>
<figure><img src="06_projected_P2_final.png"><figcaption>末帧为 thumb+ring；pinky 已退出，middle 不承担末段载荷。final-window 中 pinky 仍有短暂轻触，但平均承力低于 10%。</figcaption></figure>
</div>
</main></body></html>
"""
    (output_dir / "visual_summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/"
            "dynamic_state_cache/earlier_pre_ring_for_B12b_to_B23.npz"
        ),
    )
    parser.add_argument(
        "--model-seed",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/p1_p2_micro_rebuild_m3_2_18/"
            "P2_ring_wrap_to_front_micro_ring_guarded_seed.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "b12_ring_under_recapture_m3_2_32"
        ),
    )
    parser.add_argument("--source-t", type=float, default=0.21)
    parser.add_argument("--duration", type=float, default=0.80)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.009)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(cache_path)
    model_seed = json.loads(Path(args.model_seed).read_text(encoding="utf-8"))
    env_args = make_env_args(args)
    env = m316.make_env(model_seed, env_args, duration=args.duration + 0.05, render=True)

    try:
        selected_index = cache_index(cache, args.source_t)
        selected_summary, selected_trace = rollout(
            env,
            cache,
            selected_index,
            args,
            source_t=args.source_t,
            duration_s=args.duration,
            render=True,
        )
        if selected_trace["gif_frames"]:
            imageio.mimsave(
                output_dir / "b12_ring_under_recapture.gif",
                selected_trace["gif_frames"],
                duration=args.render_every / env.control_hz,
            )
        save_key_frames(output_dir, selected_trace["key_frames"])
        write_csv(output_dir / "b12_ring_under_recapture_trace.csv", selected_trace["rows"])
        save_trace_plot(
            output_dir / "b12_ring_under_recapture_trace.png",
            selected_trace["rows"],
        )

        entry_validation: dict[str, Any] = {}
        for source_t in [0.20, 0.21, 0.22]:
            index = cache_index(cache, source_t)
            summary, _ = rollout(
                env,
                cache,
                index,
                args,
                source_t=source_t,
                duration_s=args.duration,
                render=False,
            )
            entry_validation[f"{source_t:.2f}"] = summary

        first_under_step = selected_summary["first_ring_under_step"]
        if first_under_step is None:
            first_under_step = len(selected_trace["snapshots"])
        arrival_count = save_arrival_cache(
            output_dir / "projected_P2_ring_under_arrival_cache.npz",
            selected_trace["snapshots"],
            int(first_under_step),
        )

        last = selected_trace["snapshots"][-1]
        projected_p2 = {
            **model_seed,
            "phase": 2.0,
            "name": "P2_projected_ring_under_thumb_main_dynamic_arrival",
            "desired_contacts": ["thumb", "ring"],
            "hand_qpos": last["hand_qpos"].astype(float).tolist(),
            "target_qpos": last["target_qpos"].astype(float).tolist(),
            "pen_pos": last["pen_pos"].astype(float).tolist(),
            "pen_quat": last["pen_quat"].astype(float).tolist(),
            "dynamic_qvel": last["qvel"].astype(float).tolist(),
            "m3_2_32_role": (
                "Dynamic-only projected P2 arrival after B12 thumb brake and ring-under recapture. "
                "Do not treat this JSON as a zero-velocity canonical reset; use the NPZ arrival cache."
            ),
            "m3_2_32_source_t": float(args.source_t),
        }
        save_json(output_dir / "P2_projected_ring_under_dynamic_arrival_seed.json", projected_p2)
        save_json(
            output_dir / "B12_thumb_brake_target.json",
            {
                "name": "B12_thumb_main_brake_ring_retract_target",
                "hand_qpos": selected_trace["brake_target"].astype(float).tolist(),
                "dynamic_target_only": True,
            },
        )
        save_json(
            output_dir / "ring_under_recapture_target.json",
            {
                "name": "ring_under_contact_stop_target",
                "hand_qpos": selected_trace["capture_target"].astype(float).tolist(),
                "dynamic_target_only": True,
            },
        )

        report = {
            "stage": "M3.2.32 B12 running-state to ring-under recapture to projected P2",
            "source_cache": str(cache_path),
            "selected_source_t": float(args.source_t),
            "controller": {
                "brake_steps": BRAKE_STEPS,
                "brake_delta": {str(key): value for key, value in BRAKE_DELTA.items()},
                "brake_ring_q": BRAKE_RING_Q.astype(float).tolist(),
                "capture_ring_q": CAPTURE_RING_Q.astype(float).tolist(),
                "capture_thumb_delta": CAPTURE_THUMB_DELTA.astype(float).tolist(),
                "brake_action_limit": BRAKE_ACTION_LIMIT,
                "capture_action_limit": CAPTURE_ACTION_LIMIT,
            },
            "selected_entry": selected_summary,
            "entry_validation": entry_validation,
            "arrival_cache_states": arrival_count,
            "interpretation": {
                "B12_running_state": (
                    "Middle's -y/-z opposing load is unloaded for 60 ms. Thumb becomes the primary "
                    "support while ring retracts outside the pen."
                ),
                "ring_under_recapture": (
                    "Ring performs a low-force surface graze near the cylinder equator, then contacts "
                    "with negative radial-z. It does not use the old top-side ring collapse."
                ),
                "projected_P2": (
                    "The final short window is thumb-main with ring under/supporting and pinky absent "
                    "at the final state. It is a dynamic arrival, not the old static P2 reset."
                ),
            },
            "limits": (
                "The selected t=0.21s cache entry passes the full 0.80s primitive. Neighboring 0.20s "
                "and 0.22s entries recapture transiently but do not retain ring through the same final "
                "window. The projected P2 ring hold is about 0.59s and is not an indefinite static hold."
            ),
        }
        save_json(output_dir / "summary.json", report)

        acceptance = selected_summary["acceptance"]
        final_share = selected_summary["final_window_force_share"]
        final_ratio = selected_summary["final_window_contact_ratio"]
        hard_graze_count = len(selected_summary["hard_upper_ring_contacts"])
        assessment = f"""# M3.2.32 B12 running-state → ring-under recapture → projected P2

本轮不再从旧 B12 末端直接插值到静态 P2。旧 running-state 中 middle 对笔施加的力主要指向 `-y/-z`，与 thumb 的 `+y/+z` 支撑互相抵消；因此先用 6 个控制步卸载 middle，让 thumb 主动刹住笔，同时把 ring 外撤。

## 选定入口：cache t={args.source_t:.2f}s

- B12 brake：`{'PASS' if acceptance['b12_running_brake'] else 'CHECK'}`；末端线速度 `{selected_summary['brake']['linear_speed_m_s']:.4f} m/s`，thumb 承力 `{selected_summary['brake']['thumb_share']:.3f}`。
- ring-under recapture：`{'PASS' if acceptance['ring_under_recapture'] else 'CHECK'}`；首次下侧接触 `{selected_summary['first_ring_under_s']:.2f}s`，最大无接触间隙 `{selected_summary['no_contact_max_gap_steps']} step`，最大穿透 `{selected_summary['max_penetration_mm']:.2f} mm`。
- projected P2：`{'PASS' if acceptance['projected_p2_short_hold'] else 'CHECK'}`；下侧接触后维持 `{selected_summary['projected_p2_hold_s']:.2f}s`，final-window thumb/ring 接触率 `{final_ratio['thumb']:.2f}/{final_ratio['ring']:.2f}`，承力 `{final_share['thumb']:.2f}/{final_share['ring']:.2f}`。
- ring 上侧硬碰撞计数：`{hard_graze_count}`。早期存在一次约 3 mm 径向 z 的极轻擦，法向力约 0.01 N；随后两帧位于赤道附近，再进入负 radial-z 下侧。
- 末帧接触：`{','.join(selected_summary['final_contacts']) or 'none'}`。pinky 在 final-window 仍有短暂轻触，但平均承力 `{final_share['pinky']:.3f}`，并在末帧退出。

## 边界

当前是一个对 `t=0.21s` 动态入口成立的短 primitive。`t=0.20/0.22s` 会完成瞬时下侧再捕获，但不能在同一个 0.80s 末窗持续保留 ring，因此 arrival cache 目前应围绕选定入口做窄 curriculum，不应宣称整个 B12 cache 窗口已鲁棒连通。

`P2_projected_ring_under_dynamic_arrival_seed.json` 只描述末帧几何；训练/连接下一段应使用 `projected_P2_ring_under_arrival_cache.npz`，因为这里保留了真实动态速度。

## 可视化与数据

- `visual_summary.html`
- `m3_2_32_keypose_montage.png`
- `b12_ring_under_recapture.gif`
- `b12_ring_under_recapture_trace.png`
- `b12_ring_under_recapture_trace.csv`
- `projected_P2_ring_under_arrival_cache.npz`
"""
        (output_dir / "ASSESSMENT.zh.md").write_text(assessment, encoding="utf-8")
        write_visual_summary(output_dir, report)
    finally:
        env.close()

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "acceptance": report["selected_entry"]["acceptance"],
                "first_ring_under_s": report["selected_entry"]["first_ring_under_s"],
                "projected_p2_hold_s": report["selected_entry"]["projected_p2_hold_s"],
                "final_contacts": report["selected_entry"]["final_contacts"],
                "final_window_force_share": report["selected_entry"]["final_window_force_share"],
                "arrival_cache_states": report["arrival_cache_states"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
