from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from build_m3_2_9_photo_five_phase_debug import PHASE_SPECS, normalize, set_state, smoothstep
from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import BODY_TO_FINGER


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
LINK_PREFIX_TO_FINGER = {
    "Link_1": "thumb",
    "Link_2": "index",
    "Link_3": "middle",
    "Link_4": "ring",
    "Link_5": "pinky",
}


def load_keyframes(seed_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in PHASE_SPECS:
        path = seed_dir / f"P{spec['phase']}_{spec['name']}_hand_pen_tuned_seed.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        out.append(data)
    return out


def make_env(
    preset: dict[str, Any],
    *,
    duration: float,
    action_delta_limit: float,
    collision_mode: str,
    hard_hold_hand: bool,
) -> Dex5PenSpinMinimalEnv:
    if collision_mode == "seed":
        disable_thumb_mesh = bool(preset.get("disable_thumb_mesh_collision", True))
        disable_distal_mesh = bool(preset.get("disable_distal_mesh_collision", True))
    elif collision_mode == "proxy":
        disable_thumb_mesh = True
        disable_distal_mesh = True
    elif collision_mode == "mesh":
        disable_thumb_mesh = False
        disable_distal_mesh = False
    else:
        raise ValueError(f"Unknown collision mode: {collision_mode}")

    return Dex5PenSpinMinimalEnv(
        model_path=preset["model_path"],
        pen_init_pos=preset["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=duration,
        action_delta_limit=action_delta_limit,
        actuator_kp=preset.get("actuator_kp", 400.0),
        actuator_kv=preset.get("actuator_kv", 20.0),
        thumb_force_limit=preset.get("thumb_force_limit", 0.0),
        hard_hold_hand=hard_hold_hand,
        visual_only_markers=bool(preset.get("visual_only_markers", True)),
        disable_thumb_mesh_collision=disable_thumb_mesh,
        disable_distal_mesh_collision=disable_distal_mesh,
        seed=0,
    )


def reset_to_keyframe(env: Dex5PenSpinMinimalEnv, keyframe: dict[str, Any]) -> None:
    set_state(
        env,
        np.asarray(keyframe["hand_qpos"], dtype=np.float64),
        np.asarray(keyframe["pen_pos"], dtype=np.float64),
        np.asarray(keyframe["pen_quat"], dtype=np.float64),
    )


def finger_for_body(body_name: str | None) -> str | None:
    if not body_name:
        return None
    if body_name in BODY_TO_FINGER:
        return BODY_TO_FINGER[body_name]
    for prefix, finger in LINK_PREFIX_TO_FINGER.items():
        if body_name.startswith(prefix):
            return finger
    return None


def contact_metrics(env: Dex5PenSpinMinimalEnv) -> dict[str, Any]:
    fingers: set[str] = set()
    geoms: Counter[str] = Counter()
    normal_force: dict[str, float] = defaultdict(float)
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
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(env.model, env.data, contact_id, force)
            normal_force[finger] += abs(float(force[0]))
        max_penetration = max(max_penetration, float(max(0.0, -con.dist)))

    total_force = float(sum(normal_force.values()))
    shares = {finger: 0.0 for finger in FINGERS}
    if total_force > 1e-9:
        for finger, force in normal_force.items():
            shares[finger] = float(force / total_force)

    return {
        "fingers": fingers,
        "geoms": geoms,
        "max_penetration": max_penetration,
        "normal_force": dict(normal_force),
        "force_share": shares,
        "total_normal_force": total_force,
    }


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return q / (np.linalg.norm(q) + 1e-12)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * float(angle)
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def quat_to_axis(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(np.asarray(quat, dtype=np.float64))
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float64,
    )


def axis_angle_xy(axis: np.ndarray) -> float:
    return float(np.arctan2(axis[1], axis[0]))


def joint_limit_margin(env: Dex5PenSpinMinimalEnv, qpos: np.ndarray) -> tuple[float, str]:
    margin = np.minimum(qpos - env.joint_low, env.joint_high - qpos)
    idx = int(np.argmin(margin))
    return float(margin[idx]), DEX5_JOINT_ORDER[idx]


def desired_hit(fingers: set[str], desired: set[str]) -> bool:
    return bool(desired and desired.issubset(fingers))


def classify_row(row: dict[str, Any], *, duration: float, desired_contacts: set[str]) -> str:
    if row["terminated"] or row["survival_s"] + 1e-9 < duration:
        if row["survival_s"] >= 0.18 and row["max_penetration_mm"] <= 15.0:
            return "transient_only"
        return "reject"
    if row["max_penetration_mm"] > 15.0:
        return "reject_penetration"
    if row["min_height"] < 0.045:
        return "reject_low"
    if duration <= 0.25:
        return "short_usable"
    if row["desired_ratio"] >= 0.4 or not desired_contacts:
        return "usable"
    return "survives_contact_drift"


def run_hold(
    keyframe: dict[str, Any],
    *,
    duration: float,
    collision_mode: str,
    action_delta_limit: float,
    hard_hold_hand: bool,
    pen_pos_delta: np.ndarray | None = None,
    pen_quat_delta: np.ndarray | None = None,
) -> dict[str, Any]:
    env = make_env(
        keyframe,
        duration=duration,
        action_delta_limit=action_delta_limit,
        collision_mode=collision_mode,
        hard_hold_hand=hard_hold_hand,
    )
    env.reset(seed=0)
    reset_to_keyframe(env, keyframe)
    if pen_pos_delta is not None or pen_quat_delta is not None:
        pos = env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3].copy()
        quat = env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7].copy()
        if pen_pos_delta is not None:
            pos += np.asarray(pen_pos_delta, dtype=np.float64)
        if pen_quat_delta is not None:
            quat = quat_mul(np.asarray(pen_quat_delta, dtype=np.float64), quat)
        env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = pos
        env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = quat / np.linalg.norm(quat)
        env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)

    desired = set(keyframe.get("desired_contacts", []))
    initial_metrics = contact_metrics(env)
    state0 = env.get_state()
    center0 = state0.pen_center.copy()
    axis0 = state0.pen_axis.copy()
    margin, margin_joint = joint_limit_margin(env, np.asarray(keyframe["hand_qpos"], dtype=np.float64))

    exact_counter: Counter[str] = Counter()
    contact_counter: Counter[str] = Counter()
    share_sum = {finger: 0.0 for finger in FINGERS}
    desired_frames = 0
    active_frames = 0
    max_penetration = 0.0
    max_total_force = 0.0
    max_actuator_force = 0.0
    min_height = float(center0[2])
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}

    steps = int(round(duration * env.control_hz))
    action = np.zeros(20, dtype=np.float32)
    steps_done = 0
    for _ in range(steps):
        metrics = contact_metrics(env)
        fingers = metrics["fingers"]
        exact = ",".join(sorted(fingers)) or "none"
        exact_counter[exact] += 1
        for finger in fingers:
            contact_counter[finger] += 1
        desired_frames += int(desired_hit(fingers, desired))
        active_frames += int(len(fingers) >= 2)
        max_penetration = max(max_penetration, float(metrics["max_penetration"]))
        max_total_force = max(max_total_force, float(metrics["total_normal_force"]))
        for finger, share in metrics["force_share"].items():
            share_sum[finger] += float(share)
        if hasattr(env.data, "actuator_force") and len(env.data.actuator_force):
            max_actuator_force = max(max_actuator_force, float(np.max(np.abs(env.data.actuator_force))))

        _, _, terminated, truncated, final_info = env.step(action)
        state = env.get_state()
        min_height = min(min_height, float(state.pen_center[2]))
        steps_done += 1
        if terminated or truncated:
            break

    statef = env.get_state()
    steps_norm = max(1, steps_done)
    row = {
        "phase": int(keyframe["phase"]),
        "name": keyframe["name"],
        "duration_s": float(duration),
        "collision_mode": collision_mode,
        "survival_s": float(steps_done / env.control_hz),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "initial_contacts": ",".join(sorted(initial_metrics["fingers"])),
        "final_contacts": ",".join(sorted(contact_metrics(env)["fingers"])),
        "top_contacts": exact_counter.most_common(5),
        "desired_ratio": float(desired_frames / steps_norm),
        "active_contact_ratio": float(active_frames / steps_norm),
        "contact_ratio": {finger: contact_counter[finger] / steps_norm for finger in FINGERS},
        "force_share_mean": {finger: share_sum[finger] / steps_norm for finger in FINGERS},
        "max_penetration_mm": float(max_penetration * 1000.0),
        "max_total_normal_force": float(max_total_force),
        "max_actuator_force": float(max_actuator_force),
        "min_height": min_height,
        "xy_drift": float(np.linalg.norm(statef.pen_center[:2] - center0[:2])),
        "z_delta": float(statef.pen_center[2] - center0[2]),
        "axis_angle_delta_deg": float(np.rad2deg(np.arccos(np.clip(np.dot(axis0, statef.pen_axis), -1.0, 1.0)))),
        "joint_margin_min": margin,
        "joint_margin_joint": margin_joint,
        "step_count": int(final_info.get("step_count", env.step_count)) if final_info else int(env.step_count),
    }
    row["class"] = classify_row(row, duration=duration, desired_contacts=desired)
    env.close()
    return row


def perturbations(pos_mm: float, deg: float) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    p = float(pos_mm) / 1000.0
    a = np.deg2rad(float(deg))
    out: list[tuple[str, np.ndarray, np.ndarray | None]] = [
        ("dx+", np.array([p, 0.0, 0.0]), None),
        ("dx-", np.array([-p, 0.0, 0.0]), None),
        ("dy+", np.array([0.0, p, 0.0]), None),
        ("dy-", np.array([0.0, -p, 0.0]), None),
        ("dz+", np.array([0.0, 0.0, p]), None),
        ("dz-", np.array([0.0, 0.0, -p]), None),
        ("rx+", np.zeros(3), quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), a)),
        ("rx-", np.zeros(3), quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -a)),
        ("ry+", np.zeros(3), quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), a)),
        ("ry-", np.zeros(3), quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), -a)),
        ("rz+", np.zeros(3), quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), a)),
        ("rz-", np.zeros(3), quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -a)),
    ]
    return out


def evaluate_perturbations(
    keyframes: list[dict[str, Any]],
    *,
    duration: float,
    collision_mode: str,
    action_delta_limit: float,
    pos_mm: float,
    deg: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trials = perturbations(pos_mm=pos_mm, deg=deg)
    for keyframe in keyframes:
        ok = 0
        classes: Counter[str] = Counter()
        max_penetration = 0.0
        min_survival = duration
        for label, dp, dq in trials:
            row = run_hold(
                keyframe,
                duration=duration,
                collision_mode=collision_mode,
                action_delta_limit=action_delta_limit,
                hard_hold_hand=False,
                pen_pos_delta=dp,
                pen_quat_delta=dq,
            )
            row["perturbation"] = label
            ok += int(row["class"] in {"short_usable", "usable", "survives_contact_drift"})
            classes[row["class"]] += 1
            max_penetration = max(max_penetration, float(row["max_penetration_mm"]))
            min_survival = min(min_survival, float(row["survival_s"]))
        rows.append(
            {
                "phase": int(keyframe["phase"]),
                "name": keyframe["name"],
                "duration_s": float(duration),
                "trial_count": len(trials),
                "success_count": ok,
                "success_ratio": ok / len(trials),
                "class_counts": dict(classes),
                "min_survival_s": float(min_survival),
                "max_penetration_mm": float(max_penetration),
                "pos_mm": float(pos_mm),
                "deg": float(deg),
            }
        )
    return rows


def interpolate_qpos(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    alpha = smoothstep(float(np.clip(t, 0.0, 1.0)))
    return (1.0 - alpha) * q0 + alpha * q1


def evaluate_transition(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    segment_s: float,
    collision_mode: str,
    action_delta_limit: float,
    tracking_gain: float,
) -> dict[str, Any]:
    env = make_env(
        a,
        duration=segment_s,
        action_delta_limit=action_delta_limit,
        collision_mode=collision_mode,
        hard_hold_hand=False,
    )
    env.reset(seed=0)
    reset_to_keyframe(env, a)

    q0 = np.asarray(a["hand_qpos"], dtype=np.float64)
    q1 = np.asarray(b["hand_qpos"], dtype=np.float64)
    desired_b = set(b.get("desired_contacts", []))
    center0 = env.get_state().pen_center.copy()
    axis0 = env.get_state().pen_axis.copy()
    target_axis = quat_to_axis(np.asarray(b["pen_quat"], dtype=np.float64))
    target_delta = float(np.unwrap([axis_angle_xy(axis0), axis_angle_xy(target_axis)])[1] - axis_angle_xy(axis0))
    sign = float(np.sign(target_delta) or 1.0)

    exact_counter: Counter[str] = Counter()
    contact_counter: Counter[str] = Counter()
    final_window_hits = 0
    final_window_steps = 0
    max_penetration = 0.0
    min_height = float(center0[2])
    max_actuator_force = 0.0
    axis_progress: list[float] = []
    terminated = False
    truncated = False
    steps = int(round(segment_s * env.control_hz))
    steps_done = 0
    for step in range(steps):
        target = interpolate_qpos(q0, q1, step / max(1, steps - 1))
        action = np.clip(
            tracking_gain * (target - env.target_qpos.astype(np.float64)),
            -action_delta_limit,
            action_delta_limit,
        ).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        state = env.get_state()
        metrics = contact_metrics(env)
        fingers = metrics["fingers"]
        exact = ",".join(sorted(fingers)) or "none"
        exact_counter[exact] += 1
        for finger in fingers:
            contact_counter[finger] += 1
        max_penetration = max(max_penetration, float(metrics["max_penetration"]))
        min_height = min(min_height, float(state.pen_center[2]))
        progress = sign * (np.unwrap([axis_angle_xy(axis0), axis_angle_xy(state.pen_axis)])[1] - axis_angle_xy(axis0))
        axis_progress.append(float(progress))
        if step >= int(round(0.75 * steps)):
            final_window_steps += 1
            final_window_hits += int(desired_hit(fingers, desired_b))
        if hasattr(env.data, "actuator_force") and len(env.data.actuator_force):
            max_actuator_force = max(max_actuator_force, float(np.max(np.abs(env.data.actuator_force))))
        steps_done += 1
        if terminated or truncated:
            break

    statef = env.get_state()
    target_abs = abs(target_delta)
    final_progress = axis_progress[-1] if axis_progress else 0.0
    peak_progress = max(axis_progress) if axis_progress else 0.0
    row = {
        "transition": f"P{a['phase']}->P{b['phase']}",
        "from_name": a["name"],
        "to_name": b["name"],
        "segment_s": float(segment_s),
        "collision_mode": collision_mode,
        "survival_s": float(steps_done / env.control_hz),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "min_height": float(min_height),
        "xy_drift": float(np.linalg.norm(statef.pen_center[:2] - center0[:2])),
        "z_delta": float(statef.pen_center[2] - center0[2]),
        "target_axis_delta_rad": float(target_delta),
        "final_progress_rad": float(final_progress),
        "peak_progress_rad": float(peak_progress),
        "progress_ratio": float(final_progress / (target_abs + 1e-9)),
        "peak_progress_ratio": float(peak_progress / (target_abs + 1e-9)),
        "final_to_target_pen_dist": float(np.linalg.norm(statef.pen_center - np.asarray(b["pen_pos"], dtype=np.float64))),
        "final_to_target_axis_angle_deg": float(np.rad2deg(np.arccos(np.clip(np.dot(statef.pen_axis, target_axis), -1.0, 1.0)))),
        "target_contact_final_window_ratio": float(final_window_hits / max(1, final_window_steps)),
        "contact_ratio": {finger: contact_counter[finger] / max(1, steps_done) for finger in FINGERS},
        "top_contacts": exact_counter.most_common(6),
        "max_penetration_mm": float(max_penetration * 1000.0),
        "max_actuator_force": float(max_actuator_force),
    }
    if row["terminated"] or row["survival_s"] + 1e-9 < segment_s:
        row["class"] = "reject"
    elif row["max_penetration_mm"] > 15.0 or row["min_height"] < 0.045:
        row["class"] = "reject_physics"
    elif row["target_contact_final_window_ratio"] >= 0.35 and row["peak_progress_ratio"] > 0.05:
        row["class"] = "connectable"
    elif row["peak_progress_ratio"] > 0.05:
        row["class"] = "motion_but_contact_mismatch"
    else:
        row["class"] = "hold_like_no_progress"
    env.close()
    return row


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
        default="dex5_penspin/reports/manus_mujoco_presets/short_primitive_eval_m3_2_15",
    )
    parser.add_argument("--collision-mode", choices=["seed", "proxy", "mesh"], default="seed")
    parser.add_argument("--hold-durations", nargs="+", type=float, default=[0.2, 0.5, 1.0])
    parser.add_argument("--transition-durations", nargs="+", type=float, default=[0.25, 0.45, 0.70])
    parser.add_argument("--perturb-duration", type=float, default=0.2)
    parser.add_argument("--perturb-pos-mm", type=float, default=1.5)
    parser.add_argument("--perturb-deg", type=float, default=3.0)
    parser.add_argument("--action-delta-limit", type=float, default=0.026)
    parser.add_argument("--tracking-gain", type=float, default=0.75)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes = load_keyframes(seed_dir)

    hold_rows: list[dict[str, Any]] = []
    for keyframe in keyframes:
        for duration in args.hold_durations:
            hold_rows.append(
                run_hold(
                    keyframe,
                    duration=duration,
                    collision_mode=args.collision_mode,
                    action_delta_limit=args.action_delta_limit,
                    hard_hold_hand=False,
                )
            )

    perturb_rows = evaluate_perturbations(
        keyframes,
        duration=args.perturb_duration,
        collision_mode=args.collision_mode,
        action_delta_limit=args.action_delta_limit,
        pos_mm=args.perturb_pos_mm,
        deg=args.perturb_deg,
    )

    transition_rows: list[dict[str, Any]] = []
    for a, b in zip(keyframes[:-1], keyframes[1:]):
        for segment_s in args.transition_durations:
            transition_rows.append(
                evaluate_transition(
                    a,
                    b,
                    segment_s=segment_s,
                    collision_mode=args.collision_mode,
                    action_delta_limit=args.action_delta_limit,
                    tracking_gain=args.tracking_gain,
                )
            )

    write_csv(output_dir / "pose_short_hold.csv", hold_rows)
    write_csv(output_dir / "pose_perturbation.csv", perturb_rows)
    write_csv(output_dir / "transition_segments.csv", transition_rows)

    summary = {
        "stage": "M3.2.15 short primitive evaluation",
        "principle": (
            "Evaluate canonical grasps as short-lived contact states and transition entries, "
            "not as indefinitely stable static holds."
        ),
        "seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "collision_mode": args.collision_mode,
        "hold_rows": hold_rows,
        "perturbation_rows": perturb_rows,
        "transition_rows": transition_rows,
        "artifacts": {
            "pose_short_hold_csv": str(output_dir / "pose_short_hold.csv"),
            "pose_perturbation_csv": str(output_dir / "pose_perturbation.csv"),
            "transition_segments_csv": str(output_dir / "transition_segments.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# M3.2.15 Short Primitive Evaluation",
        "",
        "This evaluates P1-P5 as short-lived canonical states and transition entries, not as long static holds.",
        "",
        "## Pose Hold",
        "",
        "| phase | name | duration | class | survival | desired ratio | active contact | max penetration mm | top contacts |",
        "|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in hold_rows:
        lines.append(
            f"| {row['phase']} | {row['name']} | {row['duration_s']:.2f} | {row['class']} | "
            f"{row['survival_s']:.2f} | {row['desired_ratio']:.2f} | {row['active_contact_ratio']:.2f} | "
            f"{row['max_penetration_mm']:.2f} | {row['top_contacts'][:3]} |"
        )
    lines += [
        "",
        "## Perturbation",
        "",
        "| phase | name | duration | success ratio | class counts | max penetration mm |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for row in perturb_rows:
        lines.append(
            f"| {row['phase']} | {row['name']} | {row['duration_s']:.2f} | "
            f"{row['success_ratio']:.2f} | {row['class_counts']} | {row['max_penetration_mm']:.2f} |"
        )
    lines += [
        "",
        "## Transitions",
        "",
        "| transition | duration | class | survival | peak progress | target contact final ratio | min height | max penetration mm | top contacts |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in transition_rows:
        lines.append(
            f"| {row['transition']} | {row['segment_s']:.2f} | {row['class']} | "
            f"{row['survival_s']:.2f} | {row['peak_progress_ratio']:.2f} | "
            f"{row['target_contact_final_window_ratio']:.2f} | {row['min_height']:.3f} | "
            f"{row['max_penetration_mm']:.2f} | {row['top_contacts'][:3]} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "summary": str(output_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
