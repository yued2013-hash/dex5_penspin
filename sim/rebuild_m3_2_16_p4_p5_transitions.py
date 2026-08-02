from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

from build_m3_2_9_photo_five_phase_debug import PHASE_SPECS, normalize, quat_from_x_axis, save_montage, set_state, slerp, smoothstep
from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from evaluate_m3_2_15_short_primitives import FINGERS, contact_metrics, quat_to_axis


TIP_GEOMS = {
    "thumb": "Link_14R_tip_proxy",
    "index": "Link_24R_tip_proxy",
    "middle": "Link_34R_tip_proxy",
    "ring": "Link_44R_tip_proxy",
    "pinky": "Link_54R_tip_proxy",
}

PHASE_SPEC_BY_PHASE = {int(spec["phase"]): spec for spec in PHASE_SPECS}


@dataclass
class ControllerState:
    thumb_bias: float
    ring_open: float
    ring_guard: float
    shelf_bias: float


def load_chain(seed_dir: Path) -> dict[int, dict[str, Any]]:
    seeds: dict[int, dict[str, Any]] = {}
    for spec in PHASE_SPECS:
        phase = int(spec["phase"])
        path = seed_dir / f"P{phase}_{spec['name']}_hand_pen_tuned_seed.json"
        seed = json.loads(path.read_text(encoding="utf-8"))
        seed["_path"] = str(path)
        seeds[phase] = seed
    return seeds


def make_env(seed: dict[str, Any], args: argparse.Namespace, *, duration: float, render: bool) -> Dex5PenSpinMinimalEnv:
    env = Dex5PenSpinMinimalEnv(
        model_path=seed["model_path"],
        pen_init_pos=seed["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=duration,
        action_delta_limit=args.action_delta_limit,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        hard_hold_hand=False,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        render_mode="rgb_array" if render else None,
        seed=args.seed,
    )
    configure_tip_proxies(env, args)
    return env


def configure_tip_proxies(env: Dex5PenSpinMinimalEnv, args: argparse.Namespace) -> None:
    for geom_id in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "tip_proxy" not in geom_name:
            continue
        env.model.geom_size[geom_id, 0] = float(args.tip_proxy_radius)
        env.model.geom_rgba[geom_id, 3] = max(float(env.model.geom_rgba[geom_id, 3]), 0.65)
        env.model.geom_friction[geom_id] = np.array([args.tip_proxy_friction, 0.08, 0.01], dtype=np.float64)


def enforce_shared_motor_consistency(qpos: np.ndarray) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float64).copy()
    for pip, dip in [(2, 3), (6, 7), (10, 11), (14, 15)]:
        q[dip] = q[pip]
    return q


def reset_to_seed(env: Dex5PenSpinMinimalEnv, seed: dict[str, Any]) -> None:
    set_state(
        env,
        np.asarray(seed["hand_qpos"], dtype=np.float64),
        np.asarray(seed["pen_pos"], dtype=np.float64),
        np.asarray(seed["pen_quat"], dtype=np.float64),
    )


def axis_to_quat_like(base_quat: np.ndarray, dtheta_deg: float, dz: float) -> np.ndarray:
    axis = quat_to_axis(np.asarray(base_quat, dtype=np.float64))
    theta = float(np.arctan2(axis[1], axis[0]) + np.deg2rad(dtheta_deg))
    new_axis = np.array([np.cos(theta), np.sin(theta), float(axis[2] + dz)], dtype=np.float64)
    return quat_from_x_axis(new_axis)


def tip_surface_distances(env: Dex5PenSpinMinimalEnv) -> dict[str, float]:
    a = env.data.site_xpos[env.pen_end_a_id].copy()
    b = env.data.site_xpos[env.pen_end_b_id].copy()
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-12
    out: dict[str, float] = {}
    for finger, geom_name in TIP_GEOMS.items():
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            continue
        p = env.data.geom_xpos[geom_id].copy()
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        closest = a + t * ab
        out[finger] = float(np.linalg.norm(p - closest) - 0.006)
    return out


def mutate_seed(
    base: dict[str, Any],
    *,
    q_delta: np.ndarray,
    pos_delta: np.ndarray,
    dtheta_deg: float,
    axis_z_delta: float,
    label: str,
) -> dict[str, Any]:
    q = enforce_shared_motor_consistency(np.asarray(base["hand_qpos"], dtype=np.float64) + q_delta)
    pos = np.asarray(base["pen_pos"], dtype=np.float64) + pos_delta
    quat = axis_to_quat_like(np.asarray(base["pen_quat"], dtype=np.float64), dtheta_deg, axis_z_delta)
    seed = {
        **base,
        "hand_qpos": q.astype(float).tolist(),
        "target_qpos": q.astype(float).tolist(),
        "pen_pos": pos.astype(float).tolist(),
        "pen_quat": quat.astype(float).tolist(),
        "m3_2_16_label": label,
        "m3_2_16_q_delta": q_delta.astype(float).tolist(),
        "m3_2_16_pos_delta": pos_delta.astype(float).tolist(),
        "m3_2_16_dtheta_deg": float(dtheta_deg),
        "m3_2_16_axis_z_delta": float(axis_z_delta),
    }
    return seed


def random_q_delta(phase: int, rng: np.random.Generator) -> np.ndarray:
    q = np.zeros(20, dtype=np.float64)
    if phase == 4:
        ranges = {
            0: (-0.12, 0.16),
            1: (-0.05, 0.52),
            2: (-0.05, 0.48),
            4: (-0.12, 0.12),
            5: (-0.22, 0.36),
            6: (-0.08, 0.48),
            8: (-0.10, 0.10),
            9: (-0.62, 0.02),
            10: (-0.42, 0.06),
            16: (-0.08, 0.00),
            17: (-0.06, 0.26),
            18: (0.02, 0.55),
            19: (0.10, 0.95),
        }
    elif phase == 5:
        ranges = {
            0: (-0.08, 0.18),
            1: (-0.02, 0.62),
            2: (-0.02, 0.58),
            4: (-0.12, 0.12),
            5: (-0.24, 0.28),
            6: (-0.18, 0.28),
            8: (-0.12, 0.08),
            9: (-0.72, -0.08),
            10: (-0.32, 0.08),
            16: (-0.08, 0.00),
            17: (-0.06, 0.30),
            18: (0.08, 0.72),
            19: (0.18, 1.05),
        }
    else:
        ranges = {
            0: (-0.08, 0.12),
            1: (-0.03, 0.38),
            2: (-0.04, 0.36),
            4: (-0.10, 0.10),
            5: (-0.16, 0.30),
            6: (-0.08, 0.32),
            8: (-0.08, 0.08),
            9: (-0.46, 0.06),
            10: (-0.30, 0.08),
            16: (-0.06, 0.00),
            17: (-0.04, 0.22),
            18: (0.02, 0.48),
            19: (0.08, 0.78),
        }
    for idx, (lo, hi) in ranges.items():
        q[idx] = rng.uniform(lo, hi)
    return enforce_shared_motor_consistency(q)


def deterministic_deltas(phase: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    if phase == 4:
        for thumb_tip in [0.25, 0.45, 0.70]:
            for thumb_pitch in [0.12, 0.28, 0.42]:
                for index in [0.12, 0.30]:
                    q = np.zeros(20, dtype=np.float64)
                    q[1] = index
                    q[2] = 0.18
                    q[5] = -0.08
                    q[6] = 0.20
                    q[9] = -0.34
                    q[10] = -0.22
                    q[17] = 0.08
                    q[18] = thumb_pitch
                    q[19] = thumb_tip
                    out.append(enforce_shared_motor_consistency(q))
    elif phase == 5:
        for thumb_tip in [0.35, 0.60, 0.90]:
            for index in [0.18, 0.38, 0.56]:
                q = np.zeros(20, dtype=np.float64)
                q[1] = index
                q[2] = 0.24 + 0.35 * index
                q[5] = -0.10
                q[6] = -0.04
                q[9] = -0.46
                q[10] = -0.16
                q[17] = 0.10
                q[18] = 0.32
                q[19] = thumb_tip
                out.append(enforce_shared_motor_consistency(q))
    else:
        for thumb_tip in [0.25, 0.45, 0.65]:
            q = np.zeros(20, dtype=np.float64)
            q[1] = 0.16
            q[2] = 0.12
            q[5] = 0.08
            q[6] = 0.14
            q[9] = -0.25
            q[10] = -0.12
            q[17] = 0.08
            q[18] = 0.24
            q[19] = thumb_tip
            out.append(enforce_shared_motor_consistency(q))
    return out


def static_score(env: Dex5PenSpinMinimalEnv, seed: dict[str, Any], desired: set[str], allowed_extra: set[str]) -> tuple[float, dict[str, Any]]:
    metrics = contact_metrics(env)
    fingers = set(metrics["fingers"])
    distances = tip_surface_distances(env)
    missing = desired.difference(fingers)
    extra = fingers.difference(desired).difference(allowed_extra)
    hits = desired.intersection(fingers)
    score = 60.0 * len(hits) - 38.0 * len(missing) - 22.0 * len(extra)
    if desired.issubset(fingers):
        score += 85.0
    if "thumb" in fingers:
        score += 30.0
    if "ring" in fingers and "ring" not in desired:
        score -= 18.0
    if "pinky" in fingers:
        score -= 45.0
    for finger in desired:
        score -= 780.0 * max(0.0, distances.get(finger, 0.05))
    max_pen = float(metrics["max_penetration"])
    score -= 1250.0 * max(0.0, max_pen - 0.012)
    q_delta = np.asarray(seed.get("m3_2_16_q_delta", np.zeros(20)), dtype=np.float64)
    pos_delta = np.asarray(seed.get("m3_2_16_pos_delta", np.zeros(3)), dtype=np.float64)
    score -= 14.0 * float(np.linalg.norm(q_delta))
    score -= 85.0 * float(np.linalg.norm(pos_delta))
    score -= 0.05 * abs(float(seed.get("m3_2_16_dtheta_deg", 0.0)))
    return float(score), {
        "contacts": sorted(fingers),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "max_penetration_mm": float(max_pen * 1000.0),
        "tip_surface_distance": distances,
    }


def apply_controller(base_q: np.ndarray, ctrl: ControllerState, env: Dex5PenSpinMinimalEnv, ramp: float) -> np.ndarray:
    q = np.asarray(base_q, dtype=np.float64).copy()
    thumb = ramp * ctrl.thumb_bias
    ring_open = ramp * max(0.0, ctrl.ring_open - ctrl.ring_guard)
    guard = ramp * ctrl.ring_guard
    shelf = ramp * ctrl.shelf_bias

    q[1] += 0.35 * shelf
    q[2] += 0.55 * shelf
    q[5] += 0.20 * shelf
    q[6] += 0.45 * shelf

    q[9] -= ring_open
    q[10] -= 0.75 * ring_open
    q[9] += 0.35 * guard
    q[10] += 0.45 * guard

    q[16] += 0.04 * thumb
    q[17] += 0.18 * thumb
    q[18] += 1.00 * thumb
    q[19] += 0.80 * thumb

    q = np.clip(q, env.joint_low, env.joint_high)
    return enforce_shared_motor_consistency(q)


def update_controller(ctrl: ControllerState, metrics: dict[str, Any], state: Any, z0: float, args: argparse.Namespace, dt: float) -> None:
    fingers = set(metrics["fingers"])
    shares = metrics["force_share"]
    thumb_share = float(shares.get("thumb", 0.0))
    ring_share = float(shares.get("ring", 0.0))
    axis_abs_z = abs(float(state.pen_axis[2]))
    height_drop = max(0.0, z0 - float(state.pen_center[2]))
    support = {"thumb", "index", "middle", "ring"}.intersection(fingers)

    if "thumb" not in fingers or thumb_share < args.thumb_share_target:
        ctrl.thumb_bias += args.thumb_gain * max(args.thumb_share_target - thumb_share, 0.08) * dt
    elif thumb_share > args.thumb_share_target + 0.10:
        ctrl.thumb_bias -= 0.30 * args.thumb_gain * (thumb_share - args.thumb_share_target) * dt

    if ring_share > args.ring_share_max:
        ctrl.ring_open += args.ring_open_gain * (ring_share - args.ring_share_max) * dt
    elif ring_share < args.ring_share_min and axis_abs_z < args.axis_z_guard and len(support) >= 3:
        ctrl.ring_open -= 0.20 * args.ring_open_gain * (args.ring_share_min - ring_share) * dt

    if axis_abs_z > args.axis_z_guard or height_drop > args.height_drop_guard or len(support) < 2:
        ctrl.ring_guard += args.ring_guard_gain * dt
    else:
        ctrl.ring_guard -= args.ring_guard_decay * ctrl.ring_guard * dt

    if "index" not in fingers or "middle" not in fingers or len(support) < 2:
        ctrl.shelf_bias += args.shelf_gain * dt
    else:
        ctrl.shelf_bias -= 0.25 * args.shelf_gain * dt

    if height_drop > args.height_drop_guard:
        ctrl.thumb_bias += 0.18 * args.height_gain * (height_drop - args.height_drop_guard) * dt
        ctrl.shelf_bias += args.height_gain * (height_drop - args.height_drop_guard) * dt

    ctrl.thumb_bias = float(np.clip(ctrl.thumb_bias, args.thumb_bias_min, args.thumb_bias_max))
    ctrl.ring_open = float(np.clip(ctrl.ring_open, args.ring_open_min, args.ring_open_max))
    ctrl.ring_guard = float(np.clip(ctrl.ring_guard, 0.0, args.ring_guard_max))
    ctrl.shelf_bias = float(np.clip(ctrl.shelf_bias, args.shelf_bias_min, args.shelf_bias_max))


def blend_seed(a: dict[str, Any], b: dict[str, Any], t: float, *, phase: float, name: str, desired: list[str], extras: dict[str, float] | None = None) -> dict[str, Any]:
    alpha = smoothstep(t)
    qa = np.asarray(a["hand_qpos"], dtype=np.float64)
    qb = np.asarray(b["hand_qpos"], dtype=np.float64)
    q = (1.0 - alpha) * qa + alpha * qb
    pos = (1.0 - alpha) * np.asarray(a["pen_pos"], dtype=np.float64) + alpha * np.asarray(b["pen_pos"], dtype=np.float64)
    quat = slerp(np.asarray(a["pen_quat"], dtype=np.float64), np.asarray(b["pen_quat"], dtype=np.float64), alpha)
    e = extras or {}
    q[1] += e.get("index_pip", 0.0)
    q[2] += e.get("index_mid", 0.0)
    q[5] += e.get("middle_pip", 0.0)
    q[6] += e.get("middle_mid", 0.0)
    q[9] -= e.get("ring_open", 0.0)
    q[10] -= 0.7 * e.get("ring_open", 0.0)
    q[17] += e.get("thumb_roll", 0.0)
    q[18] += e.get("thumb_pitch", 0.0)
    q[19] += e.get("thumb_tip", 0.0)
    pos += np.array([e.get("pos_x", 0.0), e.get("pos_y", 0.0), e.get("pos_z", 0.0)], dtype=np.float64)
    out = {
        **b,
        "phase": phase,
        "name": name,
        "desired_contacts": desired,
        "hand_qpos": enforce_shared_motor_consistency(q).astype(float).tolist(),
        "target_qpos": enforce_shared_motor_consistency(q).astype(float).tolist(),
        "pen_pos": pos.astype(float).tolist(),
        "pen_quat": normalize(quat).astype(float).tolist(),
        "m3_2_16_label": name,
        "m3_2_16_bridge_from": f"P{a['phase']}->{b['phase']}",
        "m3_2_16_bridge_t": float(t),
        "m3_2_16_bridge_extras": dict(e),
    }
    return out


def desired_for_time(route: list[dict[str, Any]], step: int, steps_per_segment: int) -> set[str]:
    idx = min(len(route) - 1, max(0, step // max(1, steps_per_segment) + 1))
    return set(route[idx].get("desired_contacts", []))


def interpolate_route_q(route: list[dict[str, Any]], step: int, steps_per_segment: int) -> np.ndarray:
    if len(route) == 1:
        return np.asarray(route[0]["hand_qpos"], dtype=np.float64)
    seg = min(len(route) - 2, max(0, step // max(1, steps_per_segment)))
    local = (step - seg * steps_per_segment) / max(1, steps_per_segment - 1)
    alpha = smoothstep(local)
    q0 = np.asarray(route[seg]["hand_qpos"], dtype=np.float64)
    q1 = np.asarray(route[seg + 1]["hand_qpos"], dtype=np.float64)
    return (1.0 - alpha) * q0 + alpha * q1


def overlay_frame(frame: np.ndarray, text: str) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 38], fill=(255, 255, 255))
    draw.text((8, 9), text, fill=(0, 0, 0))
    return np.asarray(image)


def run_route(
    env: Dex5PenSpinMinimalEnv,
    route: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    duration: float,
    segment_s: float | None,
    render_every: int = 0,
    label: str = "route",
) -> tuple[dict[str, Any], dict[str, Any]]:
    env.reset(seed=args.seed)
    reset_to_seed(env, route[0])
    ctrl = ControllerState(
        thumb_bias=args.initial_thumb_bias,
        ring_open=args.initial_ring_open,
        ring_guard=0.0,
        shelf_bias=args.initial_shelf_bias,
    )
    steps = int(round(duration * env.control_hz))
    steps_per_segment = int(round((segment_s if segment_s is not None else duration) * env.control_hz))
    z0 = float(env.get_state().pen_center[2])
    dt = 1.0 / env.control_hz

    contact_counter: Counter[str] = Counter()
    exact_counter: Counter[str] = Counter()
    desired_frames = 0
    active_frames = 0
    max_penetration = 0.0
    max_actuator_force = 0.0
    min_height = z0
    share_sum = {finger: 0.0 for finger in FINGERS}
    bias_history = {"thumb": [], "ring_open": [], "ring_guard": [], "shelf": []}
    time_values: list[float] = []
    heights: list[float] = []
    thumb_shares: list[float] = []
    ring_shares: list[float] = []
    index_shares: list[float] = []
    middle_shares: list[float] = []
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False

    for step in range(steps):
        state = env.get_state()
        metrics = contact_metrics(env)
        update_controller(ctrl, metrics, state, z0, args, dt)
        ramp = min(1.0, (step + 1) / max(1.0, args.ramp_seconds * env.control_hz))
        base_q = interpolate_route_q(route, step, steps_per_segment)
        target = apply_controller(base_q, ctrl, env, ramp)
        action = np.clip(
            args.tracking_gain * (target - env.target_qpos.astype(np.float64)),
            -args.action_delta_limit,
            args.action_delta_limit,
        ).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)

        state = env.get_state()
        metrics = contact_metrics(env)
        fingers = set(metrics["fingers"])
        desired = desired_for_time(route, step, steps_per_segment)
        desired_frames += int(bool(desired) and desired.issubset(fingers))
        active_frames += int(len(fingers) >= 2)
        for finger in fingers:
            contact_counter[finger] += 1
        exact_counter[",".join(sorted(fingers)) or "none"] += 1
        max_penetration = max(max_penetration, float(metrics["max_penetration"]))
        min_height = min(min_height, float(state.pen_center[2]))
        for finger, share in metrics["force_share"].items():
            share_sum[finger] += float(share)
        if hasattr(env.data, "actuator_force") and len(env.data.actuator_force):
            max_actuator_force = max(max_actuator_force, float(np.max(np.abs(env.data.actuator_force))))

        t = step / env.control_hz
        time_values.append(float(t))
        heights.append(float(state.pen_center[2]))
        thumb_shares.append(float(metrics["force_share"].get("thumb", 0.0)))
        ring_shares.append(float(metrics["force_share"].get("ring", 0.0)))
        index_shares.append(float(metrics["force_share"].get("index", 0.0)))
        middle_shares.append(float(metrics["force_share"].get("middle", 0.0)))
        bias_history["thumb"].append(ctrl.thumb_bias)
        bias_history["ring_open"].append(ctrl.ring_open)
        bias_history["ring_guard"].append(ctrl.ring_guard)
        bias_history["shelf"].append(ctrl.shelf_bias)
        rows.append(
            {
                "t": float(t),
                "z": float(state.pen_center[2]),
                "contacts": ",".join(sorted(fingers)) or "none",
                "thumb_share": float(metrics["force_share"].get("thumb", 0.0)),
                "ring_share": float(metrics["force_share"].get("ring", 0.0)),
                "index_share": float(metrics["force_share"].get("index", 0.0)),
                "middle_share": float(metrics["force_share"].get("middle", 0.0)),
                "thumb_bias": float(ctrl.thumb_bias),
                "ring_open": float(ctrl.ring_open),
                "ring_guard": float(ctrl.ring_guard),
                "shelf_bias": float(ctrl.shelf_bias),
                "max_penetration_mm": float(metrics["max_penetration"] * 1000.0),
            }
        )
        if render_every > 0 and (step == 0 or step % render_every == 0):
            text = (
                f"{label} t={t:.2f}s contacts={','.join(sorted(fingers)) or 'none'} "
                f"thumb={metrics['force_share'].get('thumb', 0.0):.2f} ring={metrics['force_share'].get('ring', 0.0):.2f}"
            )
            frames.append(overlay_frame(env.render(), text))
        if terminated or truncated:
            break

    if render_every > 0:
        metrics = contact_metrics(env)
        frames.append(
            overlay_frame(
                env.render(),
                f"{label} final contacts={','.join(sorted(metrics['fingers'])) or 'none'}",
            )
        )
    steps_done = max(1, len(time_values))
    summary = {
        "label": label,
        "route": [f"P{x['phase']}:{x['name']}" for x in route],
        "duration_requested_s": float(duration),
        "survival_s": float(steps_done / env.control_hz),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "desired_ratio": float(desired_frames / steps_done),
        "active_contact_ratio": float(active_frames / steps_done),
        "contact_ratio": {finger: contact_counter[finger] / steps_done for finger in FINGERS},
        "force_share_mean": {finger: share_sum[finger] / steps_done for finger in FINGERS},
        "top_contacts": exact_counter.most_common(8),
        "min_height": float(min_height),
        "final_height": float(env.get_state().pen_center[2]),
        "max_penetration_mm": float(max_penetration * 1000.0),
        "max_actuator_force": float(max_actuator_force),
        "final_contacts": sorted(contact_metrics(env)["fingers"]),
        "controller_bias_final": {
            "thumb": float(ctrl.thumb_bias),
            "ring_open": float(ctrl.ring_open),
            "ring_guard": float(ctrl.ring_guard),
            "shelf": float(ctrl.shelf_bias),
        },
    }
    trace = {
        "rows": rows,
        "frames": frames,
        "time": time_values,
        "height": heights,
        "thumb_share": thumb_shares,
        "ring_share": ring_shares,
        "index_share": index_shares,
        "middle_share": middle_shares,
        "bias": bias_history,
    }
    return summary, trace


def active_score(summary: dict[str, Any], desired: set[str]) -> float:
    score = 120.0 * summary["survival_s"]
    score += 55.0 * summary["active_contact_ratio"]
    score += 45.0 * summary["desired_ratio"]
    score += 45.0 * summary["contact_ratio"].get("thumb", 0.0)
    score += 25.0 * summary["force_share_mean"].get("thumb", 0.0)
    score -= 42.0 * max(0.0, summary["force_share_mean"].get("ring", 0.0) - 0.18)
    score -= 30.0 * summary["contact_ratio"].get("pinky", 0.0)
    score -= 4.0 * max(0.0, summary["max_penetration_mm"] - 12.0)
    score -= 120.0 * int(summary["terminated"])
    if desired.issubset(set(summary["final_contacts"])):
        score += 35.0
    return float(score)


def generate_candidates(base: dict[str, Any], phase: int, rng: np.random.Generator, samples: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    pos_means = {
        4: np.array([-0.001, -0.014, 0.004], dtype=np.float64),
        5: np.array([-0.004, 0.000, 0.006], dtype=np.float64),
        34: np.array([0.000, -0.004, 0.004], dtype=np.float64),
        45: np.array([-0.002, -0.002, 0.004], dtype=np.float64),
    }
    pos_scales = {
        4: np.array([0.007, 0.014, 0.006], dtype=np.float64),
        5: np.array([0.008, 0.010, 0.007], dtype=np.float64),
        34: np.array([0.005, 0.008, 0.005], dtype=np.float64),
        45: np.array([0.006, 0.008, 0.005], dtype=np.float64),
    }
    for q_delta in deterministic_deltas(phase):
        for dx in [-0.006, 0.0, 0.006]:
            for dy in [-0.012, 0.0, 0.012]:
                for dz in [-0.004, 0.004, 0.010]:
                    for dtheta in [-8.0, 0.0, 8.0]:
                        candidates.append(
                            mutate_seed(
                                base,
                                q_delta=q_delta,
                                pos_delta=pos_means.get(phase, np.zeros(3)) + np.array([dx, dy, dz], dtype=np.float64),
                                dtheta_deg=dtheta,
                                axis_z_delta=0.0,
                                label=f"m3_2_16_candidate_p{phase}",
                            )
                        )
    for _ in range(samples):
        q_delta = random_q_delta(phase, rng)
        pos_delta = pos_means.get(phase, np.zeros(3)) + rng.normal(0.0, pos_scales.get(phase, np.array([0.006, 0.008, 0.005])))
        candidates.append(
            mutate_seed(
                base,
                q_delta=q_delta,
                pos_delta=pos_delta,
                dtheta_deg=float(rng.normal(0.0, 10.0)),
                axis_z_delta=float(rng.normal(0.0, 0.035)),
                label=f"m3_2_16_candidate_p{phase}",
            )
        )
    return candidates


def rebuild_seed(
    env: Dex5PenSpinMinimalEnv,
    base: dict[str, Any],
    *,
    phase: int,
    desired: set[str],
    allowed_extra: set[str],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = generate_candidates(base, phase, rng, args.static_samples)
    static_rows: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates:
        reset_to_seed(env, candidate)
        score, debug = static_score(env, candidate, desired, allowed_extra)
        static_rows.append((score, candidate, debug))
    static_rows.sort(key=lambda x: x[0], reverse=True)

    active_rows: list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for static_rank, (score, candidate, debug) in enumerate(static_rows[: args.active_top], start=1):
        summary, _ = run_route(
            env,
            [candidate],
            args,
            duration=args.hold_duration,
            segment_s=None,
            render_every=0,
            label=f"P{phase}_candidate",
        )
        a_score = active_score(summary, desired)
        active_rows.append((a_score, candidate, summary, {"static_rank": static_rank, "static_score": score, **debug}))
    active_rows.sort(key=lambda x: x[0], reverse=True)
    best_score, best_seed, best_summary, best_debug = active_rows[0]
    best_seed = {
        **best_seed,
        "m3_2_16_rebuilt": True,
        "m3_2_16_active_score": float(best_score),
        "m3_2_16_hold_summary": best_summary,
        "m3_2_16_static_debug": best_debug,
    }
    report = {
        "phase": phase,
        "name": best_seed["name"],
        "desired": sorted(desired),
        "allowed_extra": sorted(allowed_extra),
        "active_score": float(best_score),
        "hold_summary": best_summary,
        "static_debug": best_debug,
        "top_active": [
            {
                "rank": i + 1,
                "active_score": float(row[0]),
                "survival_s": float(row[2]["survival_s"]),
                "desired_ratio": float(row[2]["desired_ratio"]),
                "active_contact_ratio": float(row[2]["active_contact_ratio"]),
                "thumb_contact_ratio": float(row[2]["contact_ratio"].get("thumb", 0.0)),
                "thumb_share_mean": float(row[2]["force_share_mean"].get("thumb", 0.0)),
                "ring_share_mean": float(row[2]["force_share_mean"].get("ring", 0.0)),
                "max_penetration_mm": float(row[2]["max_penetration_mm"]),
                "top_contacts": row[2]["top_contacts"][:3],
                "static_rank": int(row[3]["static_rank"]),
            }
            for i, row in enumerate(active_rows[:8])
        ],
    }
    return best_seed, report


def selected_b34_bridge(p3: dict[str, Any], p4: dict[str, Any]) -> dict[str, Any]:
    bridge = blend_seed(
        p3,
        p4,
        0.70,
        phase=3.5,
        name="bridge_p3_to_p4_thumb_lift",
        desired=["thumb", "index", "middle"],
        extras={
            "ring_open": 0.12,
            "pos_z": 0.002,
        },
    )
    bridge["m3_2_16_selected_bridge"] = "70pct_p3_to_rebuilt_p4; moderate ring opening; no extra thumb over-close"
    return bridge


def selected_b45_bridge(p4: dict[str, Any], p5: dict[str, Any]) -> dict[str, Any]:
    bridge = blend_seed(
        p4,
        p5,
        0.85,
        phase=4.5,
        name="bridge_p4_to_p5_thumb_index_capture",
        desired=["thumb", "index"],
        extras={
            "ring_open": 0.14,
            "pos_z": 0.001,
        },
    )
    q = np.asarray(bridge["hand_qpos"], dtype=np.float64)
    q[16:20] = np.asarray(p5["hand_qpos"], dtype=np.float64)[16:20]
    bridge["hand_qpos"] = enforce_shared_motor_consistency(q).astype(float).tolist()
    bridge["target_qpos"] = bridge["hand_qpos"]
    bridge["pen_pos"] = list(p5["pen_pos"])
    bridge["pen_quat"] = list(p5["pen_quat"])
    bridge["desired_contacts"] = ["thumb", "index"]
    bridge["m3_2_16_selected_bridge"] = "85pct_p4_to_rebuilt_p5; reuse P5 thumb and pen capture geometry; middle remains optional support"
    return bridge


def bridge_report(env: Dex5PenSpinMinimalEnv, seed: dict[str, Any], args: argparse.Namespace, *, desired: set[str], allowed_extra: set[str]) -> dict[str, Any]:
    reset_to_seed(env, seed)
    score, static_debug = static_score(env, seed, desired, allowed_extra)
    hold_summary, _ = run_route(
        env,
        [seed],
        args,
        duration=args.hold_duration,
        segment_s=None,
        render_every=0,
        label=seed["name"],
    )
    return {
        "phase": seed["phase"],
        "name": seed["name"],
        "desired": sorted(desired),
        "allowed_extra": sorted(allowed_extra),
        "static_score": float(score),
        "hold_summary": hold_summary,
        "static_debug": static_debug,
        "selected_bridge": seed.get("m3_2_16_selected_bridge", ""),
    }


def render_keypose(env: Dex5PenSpinMinimalEnv, seed: dict[str, Any], label: str) -> np.ndarray:
    reset_to_seed(env, seed)
    metrics = contact_metrics(env)
    text = f"{label} contacts={','.join(sorted(metrics['fingers'])) or 'none'} pen={metrics['max_penetration']*1000.0:.1f}mm"
    return overlay_frame(env.render(), text)


def save_trace_plot(path: Path, traces: dict[str, Any], title: str) -> None:
    if plt is None:
        return
    time = np.asarray(traces["time"], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(time, traces["height"], label="pen z")
    axes[0].set_ylabel("z (m)")
    axes[0].legend(loc="best")
    axes[1].plot(time, traces["thumb_share"], label="thumb")
    axes[1].plot(time, traces["index_share"], label="index")
    axes[1].plot(time, traces["middle_share"], label="middle")
    axes[1].plot(time, traces["ring_share"], label="ring")
    axes[1].set_ylabel("force share")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="best", ncol=4)
    axes[2].plot(time, traces["bias"]["thumb"], label="thumb_bias")
    axes[2].plot(time, traces["bias"]["ring_open"], label="ring_open")
    axes[2].plot(time, traces["bias"]["ring_guard"], label="ring_guard")
    axes[2].plot(time, traces["bias"]["shelf"], label="shelf")
    axes[2].set_ylabel("controller")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="best", ncol=4)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/rebuilt_p4_p5_transitions_m3_2_16",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--static-samples", type=int, default=2600)
    parser.add_argument("--active-top", type=int, default=80)
    parser.add_argument("--hold-duration", type=float, default=0.24)
    parser.add_argument("--segment-duration", type=float, default=0.28)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.009)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parser.add_argument("--action-delta-limit", type=float, default=0.028)
    parser.add_argument("--tracking-gain", type=float, default=0.62)
    parser.add_argument("--ramp-seconds", type=float, default=0.12)
    parser.add_argument("--initial-thumb-bias", type=float, default=0.045)
    parser.add_argument("--initial-ring-open", type=float, default=0.045)
    parser.add_argument("--initial-shelf-bias", type=float, default=0.020)
    parser.add_argument("--thumb-share-target", type=float, default=0.36)
    parser.add_argument("--ring-share-max", type=float, default=0.18)
    parser.add_argument("--ring-share-min", type=float, default=0.02)
    parser.add_argument("--axis-z-guard", type=float, default=0.17)
    parser.add_argument("--height-drop-guard", type=float, default=0.006)
    parser.add_argument("--thumb-bias-min", type=float, default=0.0)
    parser.add_argument("--thumb-bias-max", type=float, default=0.16)
    parser.add_argument("--ring-open-min", type=float, default=0.012)
    parser.add_argument("--ring-open-max", type=float, default=0.095)
    parser.add_argument("--ring-guard-max", type=float, default=0.065)
    parser.add_argument("--shelf-bias-min", type=float, default=0.000)
    parser.add_argument("--shelf-bias-max", type=float, default=0.065)
    parser.add_argument("--thumb-gain", type=float, default=0.075)
    parser.add_argument("--ring-open-gain", type=float, default=0.055)
    parser.add_argument("--ring-guard-gain", type=float, default=0.090)
    parser.add_argument("--ring-guard-decay", type=float, default=1.15)
    parser.add_argument("--shelf-gain", type=float, default=0.055)
    parser.add_argument("--height-gain", type=float, default=1.40)
    parser.add_argument("--render-every", type=int, default=2)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    seeds = load_chain(seed_dir)

    env = make_env(seeds[3], args, duration=max(args.hold_duration, args.segment_duration * 4.0) + 0.5, render=False)
    try:
        p4, p4_report = rebuild_seed(
            env,
            seeds[4],
            phase=4,
            desired={"thumb", "index", "middle"},
            allowed_extra=set(),
            rng=rng,
            args=args,
        )
        p4["name"] = "index_middle_to_thumb_regrasp_rebuilt"
        p4["desired_contacts"] = ["thumb", "index", "middle"]

        p5, p5_report = rebuild_seed(
            env,
            seeds[5],
            phase=5,
            desired={"thumb", "index"},
            allowed_extra={"middle"},
            rng=rng,
            args=args,
        )
        p5["name"] = "thumb_index_capture_rebuilt"
        p5["desired_contacts"] = ["thumb", "index"]

        b34 = selected_b34_bridge(seeds[3], p4)
        b34_report = bridge_report(
            env,
            b34,
            args,
            desired={"thumb", "index", "middle"},
            allowed_extra={"ring"},
        )

        b45 = selected_b45_bridge(p4, p5)
        b45_report = bridge_report(
            env,
            b45,
            args,
            desired={"thumb", "index"},
            allowed_extra={"middle", "ring"},
        )
    finally:
        env.close()

    artifacts: dict[str, str] = {}
    seed_outputs = [
        ("P4_rebuilt_thumb_supported_seed.json", p4),
        ("P5_rebuilt_thumb_index_capture_seed.json", p5),
        ("B34_p3_to_p4_thumb_lift_bridge_seed.json", b34),
        ("B45_p4_to_p5_thumb_index_capture_bridge_seed.json", b45),
    ]
    for filename, seed in seed_outputs:
        path = output_dir / filename
        path.write_text(json.dumps(seed, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts[filename] = str(path)

    render_env = make_env(seeds[3], args, duration=2.0, render=True)
    try:
        keypose_frames = [
            render_keypose(render_env, seeds[3], "P3 reference"),
            render_keypose(render_env, b34, "B34 bridge"),
            render_keypose(render_env, p4, "P4 rebuilt"),
            render_keypose(render_env, b45, "B45 bridge"),
            render_keypose(render_env, p5, "P5 rebuilt"),
        ]
        for name, frame in zip(["P3_reference", "B34_bridge", "P4_rebuilt", "B45_bridge", "P5_rebuilt"], keypose_frames):
            imageio.imwrite(output_dir / f"{name}.png", frame)
        save_montage(keypose_frames, output_dir / "m3_2_16_keypose_montage.png", cols=5)

        eval_rows: list[dict[str, Any]] = []
        traces_by_label: dict[str, Any] = {}
        for label, route, duration, segment_s in [
            ("P4_active_hold", [p4], args.hold_duration, None),
            ("P5_active_hold", [p5], args.hold_duration, None),
            ("B34_active_hold", [b34], args.hold_duration, None),
            ("B45_active_hold", [b45], args.hold_duration, None),
            ("transition_P3_B34_P4", [seeds[3], b34, p4], args.segment_duration * 2.0, args.segment_duration),
            ("transition_P4_B45_P5", [p4, b45, p5], args.segment_duration * 2.0, args.segment_duration),
            ("chain_P3_to_P5_active_controller", [seeds[3], b34, p4, b45, p5], args.segment_duration * 4.0, args.segment_duration),
        ]:
            summary, trace = run_route(
                render_env,
                route,
                args,
                duration=duration,
                segment_s=segment_s,
                render_every=args.render_every,
                label=label,
            )
            eval_rows.append(summary)
            traces_by_label[label] = trace
            if trace["frames"]:
                gif_path = output_dir / f"{label}.gif"
                imageio.mimsave(gif_path, trace["frames"], duration=args.render_every / render_env.control_hz)
                artifacts[f"{label}.gif"] = str(gif_path)
            write_csv(output_dir / f"{label}_trace.csv", trace["rows"])
            save_trace_plot(output_dir / f"{label}_trace.png", trace, label)
        write_csv(output_dir / "m3_2_16_evaluation.csv", eval_rows)
    finally:
        render_env.close()

    manifest = {
        "stage": "M3.2.16 rebuilt P4/P5 with thumb active support and ring guard",
        "source_seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "controller": {
            "thumb_share_target": args.thumb_share_target,
            "ring_share_max": args.ring_share_max,
            "initial_thumb_bias": args.initial_thumb_bias,
            "initial_ring_open": args.initial_ring_open,
            "tip_proxy_radius": args.tip_proxy_radius,
            "tip_proxy_friction": args.tip_proxy_friction,
            "actuator_kp": args.actuator_kp,
            "actuator_kv": args.actuator_kv,
        },
        "reports": [p4_report, p5_report, b34_report, b45_report],
        "evaluation": eval_rows,
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.16 Rebuilt P4/P5 + Transition Bridges",
        "",
        "This pass rebuilds P4/P5 as short-lived transition entries, adds B34/B45 bridge states, and runs a small active controller.",
        "",
        "Controller intent:",
        "",
        "- thumb keeps a small upward/closing preload and targets the main support share;",
        "- ring opens by default and only adds guard contact when the pen rolls or support is lost;",
        "- index/middle act as push/catch shelf fingers rather than letting ring become the main load path.",
        "",
        "## Evaluation",
        "",
        "| label | survival | desired ratio | active contact | thumb share | ring share | max penetration mm | final contacts |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in eval_rows:
        readme.append(
            f"| {row['label']} | {row['survival_s']:.2f} | {row['desired_ratio']:.2f} | "
            f"{row['active_contact_ratio']:.2f} | {row['force_share_mean']['thumb']:.2f} | "
            f"{row['force_share_mean']['ring']:.2f} | {row['max_penetration_mm']:.2f} | "
            f"{','.join(row['final_contacts']) or 'none'} |"
        )
    readme += [
        "",
        "## Visuals",
        "",
        "- `m3_2_16_keypose_montage.png`",
        "- `P4_active_hold.gif`",
        "- `P5_active_hold.gif`",
        "- `B34_active_hold.gif`",
        "- `B45_active_hold.gif`",
        "- `transition_P3_B34_P4.gif`",
        "- `transition_P4_B45_P5.gif`",
        "- `chain_P3_to_P5_active_controller.gif`",
        "",
        "The full P3->P5 chain is still a diagnostic primitive, not a final spin policy.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "manifest": str(output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
