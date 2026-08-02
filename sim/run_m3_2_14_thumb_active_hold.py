from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import BODY_TO_FINGER
from view_m2_mujoco import DEFAULT_MODEL


DEFAULT_SEED = Path(
    "dex5_penspin/reports/manus_mujoco_presets/"
    "p1_ring_thumb_horizontal_support_m3_2_14k/"
    "P1_ring_thumb_horizontal_extra_force_seed.json"
)

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
LINK_PREFIX_TO_FINGER = {
    "Link_1": "thumb",
    "Link_2": "index",
    "Link_3": "middle",
    "Link_4": "ring",
    "Link_5": "pinky",
}


@dataclass
class ControllerState:
    thumb_bias: float
    ring_open: float
    index_middle_bias: float
    ring_guard_bias: float = 0.0


def load_seed(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def make_env(seed: dict[str, Any], args: argparse.Namespace) -> Dex5PenSpinMinimalEnv:
    use_proxy_collisions = bool(args.proxy_only_collisions or not args.mesh_collisions)
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path or seed.get("model_path", DEFAULT_MODEL),
        pen_init_pos=seed["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=args.duration,
        action_delta_limit=args.action_delta_limit,
        actuator_kp=args.actuator_kp if args.actuator_kp is not None else seed.get("actuator_kp", 1500.0),
        actuator_kv=args.actuator_kv if args.actuator_kv is not None else seed.get("actuator_kv", 75.0),
        thumb_force_limit=(
            args.thumb_force_limit
            if args.thumb_force_limit is not None
            else seed.get("thumb_force_limit", 0.0)
        ),
        hard_hold_hand=args.hard_hold_hand,
        visual_only_markers=bool(seed.get("visual_only_markers", True)),
        disable_thumb_mesh_collision=use_proxy_collisions,
        disable_distal_mesh_collision=use_proxy_collisions,
        seed=args.seed,
    )
    configure_collision_geometry(env, args, use_proxy_collisions=use_proxy_collisions)
    return env


def configure_collision_geometry(
    env: Dex5PenSpinMinimalEnv,
    args: argparse.Namespace,
    *,
    use_proxy_collisions: bool,
) -> None:
    if use_proxy_collisions:
        for geom_id in range(env.model.ngeom):
            geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body_name = mujoco.mj_id2name(
                env.model,
                mujoco.mjtObj.mjOBJ_BODY,
                env.model.geom_bodyid[geom_id],
            )
            if "tip_proxy" in geom_name:
                continue
            if not (body_name or "").startswith("Link_"):
                continue
            if env.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                env.model.geom_contype[geom_id] = 0
                env.model.geom_conaffinity[geom_id] = 0

    if args.tip_proxy_radius is not None:
        radius = float(args.tip_proxy_radius)
        for geom_id in range(env.model.ngeom):
            geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if "tip_proxy" not in geom_name:
                continue
            env.model.geom_size[geom_id, 0] = radius
            env.model.geom_rgba[geom_id, 3] = max(float(env.model.geom_rgba[geom_id, 3]), args.tip_proxy_alpha)
            env.model.geom_friction[geom_id] = np.array(
                [args.tip_proxy_friction, 0.08, 0.01],
                dtype=np.float64,
            )


def enforce_shared_motor_consistency(qpos: np.ndarray) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float64).copy()
    for pip, dip in [(2, 3), (6, 7), (10, 11), (14, 15)]:
        q[dip] = q[pip]
    return q


def set_state(env: Dex5PenSpinMinimalEnv, seed: dict[str, Any]) -> None:
    qpos = np.asarray(seed.get("target_qpos", seed["hand_qpos"]), dtype=np.float64)
    qpos = np.clip(qpos, env.joint_low, env.joint_high)
    qpos = enforce_shared_motor_consistency(qpos)
    env.target_qpos = qpos.astype(np.float32)
    env.data.qpos[env.qpos_adr] = env.target_qpos
    env.data.ctrl[:] = env.target_qpos

    quat = np.asarray(seed["pen_quat"], dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = np.asarray(seed["pen_pos"], dtype=np.float64)
    env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = quat
    if "pen_qvel" in seed:
        env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6] = np.asarray(seed["pen_qvel"], dtype=np.float64)
    else:
        env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6] = 0.0
    env.prev_phase = env._quat_twist_phase_about_x(quat)
    env.rotation_phase = 0.0
    mujoco.mj_forward(env.model, env.data)


def finger_for_body(body_name: str | None) -> str | None:
    if not body_name:
        return None
    if body_name in BODY_TO_FINGER:
        return BODY_TO_FINGER[body_name]
    for prefix, finger in LINK_PREFIX_TO_FINGER.items():
        if body_name.startswith(prefix):
            return finger
    if body_name.startswith("thumb_"):
        return "thumb"
    if body_name.startswith("index_"):
        return "index"
    if body_name.startswith("middle_"):
        return "middle"
    if body_name.startswith("ring_"):
        return "ring"
    if body_name.startswith("pinky_"):
        return "pinky"
    return None


def contact_snapshot(env: Dex5PenSpinMinimalEnv) -> tuple[set[str], Counter[str], float]:
    fingers: set[str] = set()
    geoms: Counter[str] = Counter()
    max_penetration = 0.0
    for contact_id in range(env.data.ncon):
        con = env.data.contact[contact_id]
        geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom{con.geom1}"
        geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom{con.geom2}"
        if "pen" not in geom1 and "pen" not in geom2:
            continue
        other_geom_id = con.geom2 if "pen" in geom1 else con.geom1
        other_geom = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id) or f"geom{other_geom_id}"
        geoms[other_geom] += 1
        body_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            env.model.geom_bodyid[other_geom_id],
        )
        finger = finger_for_body(body_name)
        if finger is not None:
            fingers.add(finger)
        max_penetration = max(max_penetration, float(max(0.0, -con.dist)))
    return fingers, geoms, max_penetration


def contact_force_shares(env: Dex5PenSpinMinimalEnv) -> tuple[dict[str, float], dict[str, float]]:
    forces: dict[str, float] = defaultdict(float)
    for contact_id in range(env.data.ncon):
        con = env.data.contact[contact_id]
        geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
        geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
        if "pen" not in geom1 and "pen" not in geom2:
            continue
        other_geom_id = con.geom2 if "pen" in geom1 else con.geom1
        body_id = env.model.geom_bodyid[other_geom_id]
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        finger = finger_for_body(body_name)
        if finger is None:
            continue
        contact_force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, contact_id, contact_force)
        forces[finger] += abs(float(contact_force[0]))

    total = float(sum(forces.values()))
    shares = {finger: 0.0 for finger in FINGER_ORDER}
    if total > 1e-9:
        for finger, force in forces.items():
            shares[finger] = float(force / total)
    return dict(forces), shares


def target_from_controller(
    base_qpos: np.ndarray,
    state: ControllerState,
    env: Dex5PenSpinMinimalEnv,
    ramp: float,
) -> np.ndarray:
    q = np.asarray(base_qpos, dtype=np.float64).copy()
    thumb = ramp * state.thumb_bias
    ring_open = ramp * (state.ring_open - state.ring_guard_bias)
    index_middle = ramp * state.index_middle_bias

    # Index/middle remain a broad shelf. They are intentionally lighter than
    # thumb so the thumb can become a real support point instead of a marker.
    for idx in (1, 2, 5, 6):
        q[idx] += index_middle

    # Ring opens away from the pen under high load. A small guard bias can close
    # it back when the pen begins to roll, but ring_open remains the dominant term.
    for idx in (9, 10):
        q[idx] -= ring_open

    # Thumb preload: close pitch/tip, plus a tiny roll/yaw trim that presses the
    # proxy fingertip under the pen rather than just into the side of the pen.
    q[16] += 0.12 * thumb
    q[17] += 0.18 * thumb
    q[18] += 1.00 * thumb
    q[19] += 0.72 * thumb

    q = np.clip(q, env.joint_low, env.joint_high)
    return enforce_shared_motor_consistency(q)


def update_controller(
    ctrl: ControllerState,
    shares: dict[str, float],
    fingers: set[str],
    *,
    axis_abs_z: float,
    height_drop: float,
    args: argparse.Namespace,
    dt: float,
) -> None:
    thumb_share = shares.get("thumb", 0.0)
    ring_share = shares.get("ring", 0.0)
    support_contacts = {"thumb", "index", "middle", "ring"}.intersection(fingers)

    thumb_error = args.thumb_share_target - thumb_share
    if thumb_error > 0.0 or "thumb" not in fingers:
        ctrl.thumb_bias += args.thumb_gain * max(thumb_error, 0.08) * dt
    elif thumb_share > args.thumb_share_target + 0.08:
        ctrl.thumb_bias -= 0.35 * args.thumb_gain * (thumb_share - args.thumb_share_target) * dt

    if ring_share > args.ring_share_max:
        ctrl.ring_open += args.ring_gain * (ring_share - args.ring_share_max) * dt
    elif ring_share < args.ring_share_min and axis_abs_z < args.axis_z_guard:
        ctrl.ring_open -= 0.35 * args.ring_gain * (args.ring_share_min - ring_share) * dt

    if axis_abs_z > args.axis_z_guard:
        ctrl.ring_guard_bias += args.ring_guard_gain * (axis_abs_z - args.axis_z_guard) * dt
    else:
        ctrl.ring_guard_bias -= args.ring_guard_decay * ctrl.ring_guard_bias * dt

    if len(support_contacts) <= 2:
        missing = 3 - len(support_contacts)
        ctrl.ring_open -= args.contact_recovery_ring_close * missing * dt
        ctrl.index_middle_bias += args.contact_recovery_shelf_close * missing * dt
        if support_contacts == {"thumb"}:
            ctrl.thumb_bias -= args.thumb_only_relief_gain * dt
    if "index" not in fingers or "middle" not in fingers:
        ctrl.index_middle_bias += args.shelf_contact_gain * dt
    if "ring" not in fingers and axis_abs_z > 0.5 * args.axis_z_guard:
        ctrl.ring_guard_bias += args.ring_guard_gain * 0.5 * dt

    if height_drop > args.height_drop_guard:
        drop_error = height_drop - args.height_drop_guard
        ctrl.index_middle_bias += args.height_gain * drop_error * dt
        ctrl.ring_guard_bias += 0.45 * args.height_gain * drop_error * dt
        if thumb_share < args.thumb_share_target:
            ctrl.thumb_bias += 0.25 * args.height_gain * drop_error * dt

    ctrl.thumb_bias = float(np.clip(ctrl.thumb_bias, args.thumb_bias_min, args.thumb_bias_max))
    ctrl.ring_open = float(np.clip(ctrl.ring_open, args.ring_open_min, args.ring_open_max))
    ctrl.index_middle_bias = float(np.clip(ctrl.index_middle_bias, args.index_middle_min, args.index_middle_max))
    ctrl.ring_guard_bias = float(np.clip(ctrl.ring_guard_bias, 0.0, args.ring_guard_max))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "min": float(arr.min())}


def rollout(env: Dex5PenSpinMinimalEnv, seed: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    env.reset(seed=args.seed)
    set_state(env, seed)
    base_q = np.asarray(seed.get("target_qpos", seed["hand_qpos"]), dtype=np.float64)
    ctrl = ControllerState(
        thumb_bias=args.initial_thumb_bias,
        ring_open=args.initial_ring_open,
        index_middle_bias=args.initial_index_middle_bias,
    )

    center0 = env.get_state().pen_center.copy()
    z0 = float(center0[2])
    dt = 1.0 / env.control_hz
    steps = int(round(args.duration * env.control_hz))

    contact_counter: Counter[str] = Counter()
    exact_counter: Counter[str] = Counter()
    max_penetration = 0.0
    share_history: dict[str, list[float]] = {finger: [] for finger in FINGER_ORDER}
    bias_history: dict[str, list[float]] = {"thumb": [], "ring_open": [], "ring_guard": [], "index_middle": []}
    heights: list[float] = []
    axis_abs_zs: list[float] = []
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}

    for step in range(steps):
        state = env.get_state()
        forces, shares = contact_force_shares(env)
        fingers, _, penetration = contact_snapshot(env)
        max_penetration = max(max_penetration, float(penetration))
        axis_abs_z = abs(float(state.pen_axis[2]))
        height_drop = max(0.0, z0 - float(state.pen_center[2]))
        update_controller(ctrl, shares, fingers, axis_abs_z=axis_abs_z, height_drop=height_drop, args=args, dt=dt)

        ramp = min(1.0, (step + 1) / max(1.0, args.ramp_seconds * env.control_hz))
        target = target_from_controller(base_q, ctrl, env, ramp)
        delta = args.tracking_gain * (target - env.target_qpos.astype(np.float64))
        action = np.clip(delta, -args.action_delta_limit, args.action_delta_limit).astype(np.float32)
        _, _, terminated, truncated, final_info = env.step(action)

        for finger in fingers:
            contact_counter[finger] += 1
        exact_counter[",".join(sorted(fingers)) or "none"] += 1
        for finger in FINGER_ORDER:
            share_history[finger].append(float(shares.get(finger, 0.0)))
        bias_history["thumb"].append(ctrl.thumb_bias)
        bias_history["ring_open"].append(ctrl.ring_open)
        bias_history["ring_guard"].append(ctrl.ring_guard_bias)
        bias_history["index_middle"].append(ctrl.index_middle_bias)
        heights.append(float(state.pen_center[2]))
        axis_abs_zs.append(axis_abs_z)

        if args.print_every > 0 and step % args.print_every == 0:
            print(
                f"step={step:04d} z={state.pen_center[2]:.4f} axis_abs_z={axis_abs_z:.3f} "
                f"contacts={sorted(fingers)} "
                f"share thumb={shares.get('thumb', 0.0):.3f} ring={shares.get('ring', 0.0):.3f} "
                f"idx={shares.get('index', 0.0):.3f} mid={shares.get('middle', 0.0):.3f} "
                f"penet={penetration * 1000.0:.2f}mm "
                f"bias thumb={ctrl.thumb_bias:.3f} ring_open={ctrl.ring_open:.3f} "
                f"guard={ctrl.ring_guard_bias:.3f}"
            )
        if terminated or truncated:
            break

    steps_done = max(1, len(heights))
    final_state = env.get_state()
    summary = {
        "seed": seed["_path"],
        "collision_mode": "mesh" if args.mesh_collisions and not args.proxy_only_collisions else "tip_proxy",
        "tip_proxy_radius": None if args.tip_proxy_radius is None else float(args.tip_proxy_radius),
        "duration_requested": float(args.duration),
        "duration_survived": float(steps_done / env.control_hz),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_z": float(final_state.pen_center[2]),
        "min_z": float(min(heights) if heights else final_state.pen_center[2]),
        "max_axis_abs_z": float(max(axis_abs_zs) if axis_abs_zs else abs(final_state.pen_axis[2])),
        "final_axis_abs_z": float(abs(final_state.pen_axis[2])),
        "max_penetration_mm": float(max_penetration * 1000.0),
        "final_contacts": sorted(contact_snapshot(env)[0]),
        "contact_ratio": {finger: contact_counter[finger] / steps_done for finger in FINGER_ORDER},
        "top_contact_sets": exact_counter.most_common(8),
        "share": {finger: summarize(share_history[finger]) for finger in FINGER_ORDER},
        "bias": {name: summarize(vals) for name, vals in bias_history.items()},
        "final_info_step_count": int(final_info.get("step_count", env.step_count)) if final_info else int(env.step_count),
    }
    return summary


def run_viewer(env: Dex5PenSpinMinimalEnv, seed: dict[str, Any], args: argparse.Namespace) -> None:
    env.reset(seed=args.seed)
    set_state(env, seed)
    base_q = np.asarray(seed.get("target_qpos", seed["hand_qpos"]), dtype=np.float64)
    ctrl = ControllerState(
        thumb_bias=args.initial_thumb_bias,
        ring_open=args.initial_ring_open,
        index_middle_bias=args.initial_index_middle_bias,
    )
    z0 = float(env.get_state().pen_center[2])
    dt = 1.0 / env.control_hz
    max_steps = int(round(args.duration * env.control_hz))

    print("MuJoCo viewer: active thumb hold starts immediately. Close the window to stop.")
    print(
        f"collision_mode={'mesh' if args.mesh_collisions and not args.proxy_only_collisions else 'tip_proxy'}, "
        f"target thumb_share={args.thumb_share_target:.2f}, "
        f"ring_share_max={args.ring_share_max:.2f}, kp={env.model.actuator_gainprm[0, 0]:.1f}, "
        f"kv={-env.model.actuator_biasprm[0, 2]:.1f}"
    )
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = np.array([-0.02, 0.06, 0.06])
        viewer.cam.distance = 0.32
        viewer.cam.azimuth = -120
        viewer.cam.elevation = -20

        step = 0
        while viewer.is_running() and step < max_steps:
            loop_t0 = time.time()
            state = env.get_state()
            forces, shares = contact_force_shares(env)
            fingers, _, penetration = contact_snapshot(env)
            axis_abs_z = abs(float(state.pen_axis[2]))
            height_drop = max(0.0, z0 - float(state.pen_center[2]))
            update_controller(ctrl, shares, fingers, axis_abs_z=axis_abs_z, height_drop=height_drop, args=args, dt=dt)

            ramp = min(1.0, (step + 1) / max(1.0, args.ramp_seconds * env.control_hz))
            target = target_from_controller(base_q, ctrl, env, ramp)
            delta = args.tracking_gain * (target - env.target_qpos.astype(np.float64))
            action = np.clip(delta, -args.action_delta_limit, args.action_delta_limit).astype(np.float32)
            _, _, terminated, truncated, info = env.step(action)

            if args.print_every > 0 and step % args.print_every == 0:
                print(
                    f"step={step:04d} z={info['pen_center'][2]:.4f} axis_abs_z={abs(info['pen_axis'][2]):.3f} "
                    f"contacts={sorted(fingers)} "
                    f"share thumb={shares.get('thumb', 0.0):.3f} ring={shares.get('ring', 0.0):.3f} "
                    f"idx={shares.get('index', 0.0):.3f} mid={shares.get('middle', 0.0):.3f} "
                    f"penet={penetration * 1000.0:.2f}mm "
                    f"bias thumb={ctrl.thumb_bias:.3f} ring_open={ctrl.ring_open:.3f} "
                    f"guard={ctrl.ring_guard_bias:.3f}"
                )
            viewer.sync()
            step += 1
            if terminated or truncated:
                print("physics stopped:", "terminated" if terminated else "truncated")
                break
            sleep_s = dt - (time.time() - loop_t0)
            if sleep_s > 0.0:
                time.sleep(sleep_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-preset", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-viewer", action="store_true", help="Run an offline rollout instead of launching MuJoCo.")
    parser.add_argument("--print-every", type=int, default=50)

    parser.add_argument("--actuator-kp", type=float, default=None)
    parser.add_argument("--actuator-kv", type=float, default=None)
    parser.add_argument("--thumb-force-limit", type=float, default=None)
    parser.add_argument("--hard-hold-hand", action="store_true")
    parser.add_argument("--mesh-collisions", action="store_true", help="Experimental: collide against imported STL meshes.")
    parser.add_argument("--proxy-only-collisions", action="store_true", help="Force fingertip proxy collision mode.")
    parser.add_argument("--tip-proxy-radius", type=float, default=0.009)
    parser.add_argument("--tip-proxy-alpha", type=float, default=0.65)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parser.add_argument("--action-delta-limit", type=float, default=0.025)
    parser.add_argument("--tracking-gain", type=float, default=0.45)
    parser.add_argument("--ramp-seconds", type=float, default=0.75)

    parser.add_argument("--thumb-share-target", type=float, default=0.34)
    parser.add_argument("--ring-share-max", type=float, default=0.30)
    parser.add_argument("--ring-share-min", type=float, default=0.04)
    parser.add_argument("--axis-z-guard", type=float, default=0.16)
    parser.add_argument("--height-drop-guard", type=float, default=0.006)

    parser.add_argument("--initial-thumb-bias", type=float, default=0.025)
    parser.add_argument("--initial-ring-open", type=float, default=0.030)
    parser.add_argument("--initial-index-middle-bias", type=float, default=0.035)
    parser.add_argument("--thumb-bias-min", type=float, default=0.000)
    parser.add_argument("--thumb-bias-max", type=float, default=0.110)
    parser.add_argument("--ring-open-min", type=float, default=0.005)
    parser.add_argument("--ring-open-max", type=float, default=0.080)
    parser.add_argument("--index-middle-min", type=float, default=0.005)
    parser.add_argument("--index-middle-max", type=float, default=0.045)
    parser.add_argument("--ring-guard-max", type=float, default=0.060)

    parser.add_argument("--thumb-gain", type=float, default=0.035)
    parser.add_argument("--ring-gain", type=float, default=0.012)
    parser.add_argument("--ring-guard-gain", type=float, default=0.40)
    parser.add_argument("--ring-guard-decay", type=float, default=1.20)
    parser.add_argument("--height-gain", type=float, default=1.15)
    parser.add_argument("--contact-recovery-ring-close", type=float, default=0.10)
    parser.add_argument("--contact-recovery-shelf-close", type=float, default=0.08)
    parser.add_argument("--shelf-contact-gain", type=float, default=0.018)
    parser.add_argument("--thumb-only-relief-gain", type=float, default=0.030)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = load_seed(args.seed_preset)
    env = make_env(seed, args)
    try:
        if args.no_viewer:
            summary = rollout(env, seed, args)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            run_viewer(env, seed, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
