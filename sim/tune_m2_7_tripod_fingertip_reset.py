from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from teleop_m2_initial_state import command_for_preset
from view_m2_mujoco import DEFAULT_MODEL


TIP_GEOMS = {
    "thumb": "Link_14R_tip_proxy",
    "index": "Link_24R_tip_proxy",
    "middle": "Link_34R_tip_proxy",
    "ring": "Link_44R_tip_proxy",
    "pinky": "Link_54R_tip_proxy",
}

BODY_TO_FINGER = {
    "Link_14R": "thumb",
    "Link_24R": "index",
    "Link_34R": "middle",
    "Link_44R": "ring",
    "Link_54R": "pinky",
}

SUPPORT_FINGERS = ("index", "middle", "ring")
CONTACT_RADIUS = 0.0175
KINEMATIC_TARGET_RADIUS = 0.0155


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return v.copy()
    return v / norm


def quat_from_x_axis(axis: np.ndarray) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=np.float64))
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    dot = float(np.clip(np.dot(x_axis, axis), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rot_axis = normalize(np.cross(x_axis, axis))
    angle = math.acos(dot)
    return np.array(
        [
            math.cos(0.5 * angle),
            *(math.sin(0.5 * angle) * rot_axis),
        ],
        dtype=np.float64,
    )


def set_state(
    env: Dex5PenSpinMinimalEnv,
    hand_qpos: np.ndarray,
    pen_pos: np.ndarray,
    pen_quat: np.ndarray,
) -> None:
    qpos = np.clip(np.asarray(hand_qpos, dtype=np.float64), env.joint_low, env.joint_high)
    env.target_qpos = qpos.astype(np.float32)
    env.data.qpos[env.qpos_adr] = qpos
    env.data.ctrl[:] = qpos
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = np.asarray(pen_pos, dtype=np.float64)
    quat = np.asarray(pen_quat, dtype=np.float64)
    env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = quat / np.linalg.norm(quat)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def point_line_distances(points: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = normalize(axis)
    rel = points - center[None, :]
    proj = (rel @ axis)[:, None] * axis[None, :]
    return np.linalg.norm(rel - proj, axis=1)


def contact_snapshot(env: Dex5PenSpinMinimalEnv) -> tuple[set[str], Counter[str], float]:
    fingers: set[str] = set()
    geoms: Counter[str] = Counter()
    max_penetration = 0.0
    for i in range(env.data.ncon):
        con = env.data.contact[i]
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
        if body_name in BODY_TO_FINGER:
            fingers.add(BODY_TO_FINGER[body_name])
        max_penetration = max(max_penetration, float(max(0.0, -con.dist)))
    return fingers, geoms, max_penetration


def sample_tripod_qpos(
    rng: np.random.Generator,
    base_qpos: np.ndarray,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
) -> np.ndarray:
    q = base_qpos.copy()

    # Index / middle / ring get real freedom now. The goal is a fingertip shelf,
    # not preserving the exact Manus pose.
    q[0] = rng.uniform(-0.38, -0.02)
    q[1] = rng.uniform(0.45, 1.15)
    q[2] = rng.uniform(0.25, 0.95)
    q[3] = q[2] + rng.uniform(-0.10, 0.18)

    q[4] = rng.uniform(-0.28, 0.18)
    q[5] = rng.uniform(0.45, 1.20)
    q[6] = rng.uniform(0.25, 1.00)
    q[7] = q[6] + rng.uniform(-0.10, 0.18)

    q[8] = rng.uniform(-0.18, 0.38)
    q[9] = rng.uniform(0.55, 1.55)
    q[10] = rng.uniform(0.25, 1.20)
    q[11] = q[10] + rng.uniform(-0.10, 0.20)

    # Keep pinky present but less dominant.
    q[12] = rng.uniform(0.10, 0.38)
    q[13] = rng.uniform(0.85, 1.55)
    q[14] = rng.uniform(0.35, 0.95)
    q[15] = q[14] + rng.uniform(-0.10, 0.18)

    # Thumb stays light during tripod search. It should not create the main
    # V-slot before the index/middle/ring shelf is valid.
    q[16] = rng.uniform(0.30, 0.68)
    q[17] = rng.uniform(0.25, 0.75)
    q[18] = rng.uniform(0.20, 0.75)
    q[19] = rng.uniform(0.10, 0.50)

    return np.clip(q, joint_low, joint_high)


def fit_pen_from_tripod(
    support_points: np.ndarray,
    offset_z: float,
    offset_n: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    center_support = np.mean(support_points, axis=0)
    centered = support_points - center_support[None, :]
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    axis = normalize(vh[0])
    if axis[1] < 0.0:
        axis = -axis
    # The pen should be roughly horizontal during reset.
    axis[2] = float(np.clip(axis[2], -0.16, 0.16))
    axis = normalize(axis)

    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal = np.cross(axis, world_z)
    if np.linalg.norm(normal) < 1e-6:
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    normal = normalize(normal)
    pen_center = center_support + offset_z * world_z + offset_n * normal
    pen_quat = quat_from_x_axis(axis)

    support_dists = point_line_distances(support_points, pen_center, axis)
    span = float(np.max(centered @ axis) - np.min(centered @ axis))
    line_residual = float(np.mean(np.abs(support_dists - CONTACT_RADIUS)))
    z_std = float(np.std(support_points[:, 2]))
    return pen_center, pen_quat, span, line_residual, z_std


def build_kinematic_candidates(args: argparse.Namespace, base_qpos: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=1.0,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        seed=args.seed,
    )
    env.reset(seed=args.seed)
    tip_ids = {
        finger: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        for finger, geom_name in TIP_GEOMS.items()
    }

    rows: list[dict[str, Any]] = []
    offset_z_values = [0.010, 0.012, 0.014, 0.016, 0.018, 0.020]
    offset_n_values = [-0.006, -0.003, 0.0, 0.003, 0.006]

    for sample_i in range(args.samples):
        if sample_i == 0:
            qpos = base_qpos.copy()
        else:
            qpos = sample_tripod_qpos(rng, base_qpos, env.joint_low, env.joint_high)
        env.target_qpos = qpos.astype(np.float32)
        env.data.qpos[env.qpos_adr] = qpos
        env.data.ctrl[:] = qpos
        mujoco.mj_forward(env.model, env.data)

        tips = {finger: env.data.geom_xpos[gid].copy() for finger, gid in tip_ids.items()}
        support_points = np.stack([tips[finger] for finger in SUPPORT_FINGERS])
        support_z_std = float(np.std(support_points[:, 2]))
        if support_z_std > 0.025:
            continue

        support_xy = support_points[:, :2]
        if np.max(np.linalg.norm(support_xy[:, None, :] - support_xy[None, :, :], axis=-1)) < 0.025:
            continue

        for offset_z in offset_z_values:
            for offset_n in offset_n_values:
                fitted = fit_pen_from_tripod(support_points, offset_z=offset_z, offset_n=offset_n)
                if fitted is None:
                    continue
                pen_center, pen_quat, span, line_residual, z_std = fitted
                if not (0.045 <= pen_center[2] <= 0.105):
                    continue
                if span < 0.030 or span > 0.145:
                    continue

                pen_axis = quat_to_axis(pen_quat)
                support_dists = point_line_distances(support_points, pen_center, pen_axis)
                thumb_dist = float(point_line_distances(tips["thumb"][None, :], pen_center, pen_axis)[0])
                pinky_dist = float(point_line_distances(tips["pinky"][None, :], pen_center, pen_axis)[0])
                distance_error = float(np.mean(np.abs(support_dists - KINEMATIC_TARGET_RADIUS)))
                ring_error = float(abs(support_dists[2] - KINEMATIC_TARGET_RADIUS))
                balance = float(np.std(support_dists))
                # Kinematic pre-score: close three support tips, enough span,
                # thumb not too dominant, pinky not preferred.
                score = (
                    -260.0 * distance_error
                    -90.0 * ring_error
                    -90.0 * balance
                    -35.0 * line_residual
                    -20.0 * z_std
                    +4.0 * min(span, 0.08)
                    -12.0 * max(0.0, 0.022 - thumb_dist)
                    -8.0 * max(0.0, 0.020 - pinky_dist)
                )
                rows.append(
                    {
                        "sample_i": sample_i,
                        "qpos": qpos.tolist(),
                        "pen_pos": pen_center.tolist(),
                        "pen_quat": pen_quat.tolist(),
                        "span": span,
                        "support_z_std": support_z_std,
                        "line_residual": line_residual,
                        "distance_error": distance_error,
                        "balance": balance,
                        "thumb_axis_dist": thumb_dist,
                        "pinky_axis_dist": pinky_dist,
                        "kin_score": float(score),
                        "index_axis_dist": float(support_dists[0]),
                        "middle_axis_dist": float(support_dists[1]),
                        "ring_axis_dist": float(support_dists[2]),
                    }
                )

    env.close()
    rows.sort(key=lambda row: row["kin_score"], reverse=True)
    return rows


def quat_to_axis(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    # First column of the rotation matrix for MuJoCo wxyz quaternion.
    return normalize(
        np.array(
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + z * w),
                2.0 * (x * z - y * w),
            ],
            dtype=np.float64,
        )
    )


def evaluate_candidate(args: argparse.Namespace, candidate: dict[str, Any]) -> dict[str, Any]:
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=candidate["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=args.episode_seconds,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        seed=args.seed,
    )
    env.reset(seed=args.seed)
    set_state(
        env,
        np.asarray(candidate["qpos"], dtype=np.float64),
        np.asarray(candidate["pen_pos"], dtype=np.float64),
        np.asarray(candidate["pen_quat"], dtype=np.float64),
    )

    initial_fingers, initial_geoms, initial_penetration = contact_snapshot(env)
    center0 = env.get_state().pen_center.copy()
    axis0 = env.get_state().pen_axis.copy()
    centers = [center0]
    axes = [axis0]
    geom_counter: Counter[str] = Counter()
    finger_frames: Counter[str] = Counter()
    tripod_frames = 0
    two_or_more_support_frames = 0
    max_penetration = initial_penetration
    ncons: list[int] = []
    terminated = False

    for _ in range(env.max_episode_steps):
        _, _, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
        fingers, geoms, penetration = contact_snapshot(env)
        geom_counter.update(geoms)
        for finger in fingers:
            finger_frames[finger] += 1
        if set(SUPPORT_FINGERS).issubset(fingers):
            tripod_frames += 1
        if len(set(SUPPORT_FINGERS).intersection(fingers)) >= 2:
            two_or_more_support_frames += 1
        max_penetration = max(max_penetration, penetration)
        centers.append(info["pen_center"].copy())
        axes.append(info["pen_axis"].copy())
        ncons.append(int(env.data.ncon))
        if terminated or truncated:
            break

    centers_np = np.asarray(centers, dtype=np.float64)
    axes_np = np.asarray(axes, dtype=np.float64)
    drift = float(np.linalg.norm(centers_np[-1] - center0))
    xy_drift = float(np.linalg.norm(centers_np[-1, :2] - center0[:2]))
    z_delta = float(centers_np[-1, 2] - center0[2])
    axis_angle = float(
        math.acos(
            np.clip(
                np.dot(axes_np[0], axes_np[-1]) / (np.linalg.norm(axes_np[0]) * np.linalg.norm(axes_np[-1]) + 1e-9),
                -1.0,
                1.0,
            )
        )
    )
    support_min_frames = min(finger_frames[f] for f in SUPPORT_FINGERS)
    support_sum_frames = sum(finger_frames[f] for f in SUPPORT_FINGERS)

    score = (
        0.22 * tripod_frames
        + 0.08 * two_or_more_support_frames
        + 0.06 * support_min_frames
        + 0.015 * support_sum_frames
        + 0.006 * min(finger_frames["thumb"], 120)
        - 0.016 * finger_frames["pinky"]
        - 145.0 * drift
        - 75.0 * abs(z_delta)
        - 8.0 * axis_angle
        - 260.0 * max(0.0, max_penetration - 0.006)
    )
    if tripod_frames == 0:
        score -= 4.0
    if support_min_frames < 20:
        score -= 2.0
    if terminated:
        score -= 4.0

    row = {
        **candidate,
        "score": float(score),
        "initial_support_fingers": sorted(initial_fingers),
        "initial_tip_geoms": initial_geoms.most_common(10),
        "initial_penetration": float(initial_penetration),
        "tripod_frames": int(tripod_frames),
        "two_or_more_support_frames": int(two_or_more_support_frames),
        "index_frames": int(finger_frames["index"]),
        "middle_frames": int(finger_frames["middle"]),
        "ring_frames": int(finger_frames["ring"]),
        "thumb_frames": int(finger_frames["thumb"]),
        "pinky_frames": int(finger_frames["pinky"]),
        "support_min_frames": int(support_min_frames),
        "support_sum_frames": int(support_sum_frames),
        "drift": drift,
        "xy_drift": xy_drift,
        "z_delta": z_delta,
        "final_z": float(centers_np[-1, 2]),
        "axis_angle": axis_angle,
        "max_penetration": float(max_penetration),
        "mean_ncon": float(np.mean(ncons)) if ncons else 0.0,
        "terminated": bool(terminated),
        "top_geom_contacts": geom_counter.most_common(10),
    }
    if set(SUPPORT_FINGERS).issubset(initial_fingers):
        row["score"] += 10.0
        row["initial_tripod"] = True
    else:
        row["score"] -= 3.0
        row["initial_tripod"] = False
    env.close()
    return row


def render_preset(args: argparse.Namespace, preset: dict[str, Any], output_path: Path, steps: int) -> None:
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=preset["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=max(args.episode_seconds, steps / 100.0),
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        seed=args.seed,
        render_mode="rgb_array",
    )
    env.reset(seed=args.seed)
    set_state(
        env,
        np.asarray(preset["hand_qpos"], dtype=np.float64),
        np.asarray(preset["pen_pos"], dtype=np.float64),
        np.asarray(preset["pen_quat"], dtype=np.float64),
    )
    for _ in range(steps):
        env.step(np.zeros(20, dtype=np.float32))
    imageio.imwrite(output_path, env.render())
    env.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "score",
        "kin_score",
        "sample_i",
        "span",
        "support_z_std",
        "line_residual",
        "distance_error",
        "balance",
        "index_axis_dist",
        "middle_axis_dist",
        "ring_axis_dist",
        "thumb_axis_dist",
        "pinky_axis_dist",
        "tripod_frames",
        "two_or_more_support_frames",
        "index_frames",
        "middle_frames",
        "ring_frames",
        "thumb_frames",
        "pinky_frames",
        "support_min_frames",
        "support_sum_frames",
        "drift",
        "xy_drift",
        "z_delta",
        "final_z",
        "axis_angle",
        "max_penetration",
        "mean_ncon",
        "terminated",
        "initial_tripod",
        "initial_support_fingers",
        "initial_tip_geoms",
        "top_geom_contacts",
        "pen_pos",
        "pen_quat",
        "qpos",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows):
            out = {field: row.get(field) for field in fields}
            out["rank"] = rank
            for key in ["initial_support_fingers", "initial_tip_geoms", "top_geom_contacts", "pen_pos", "pen_quat", "qpos"]:
                out[key] = json.dumps(out[key], ensure_ascii=False)
            writer.writerow(out)


def save_best_preset(
    args: argparse.Namespace,
    base_data: dict[str, Any],
    best: dict[str, Any],
    output_dir: Path,
    preset_dir: Path,
) -> tuple[Path, Path, Path]:
    stem = "current_manus_reset_tripod_fingertip_support_m2_7"
    preset_path = preset_dir / f"{stem}.json"
    preset = {
        "model_path": str(args.model_path),
        "hand_qpos": [float(v) for v in best["qpos"]],
        "joint_order": base_data.get("joint_order", DEX5_JOINT_ORDER),
        "pen_pos": [float(v) for v in best["pen_pos"]],
        "pen_quat": [float(v) for v in best["pen_quat"]],
        "pen_qvel": [0.0] * 6,
        "actuator_kp": float(args.actuator_kp),
        "actuator_kv": float(args.actuator_kv),
        "thumb_force_limit": float(args.thumb_force_limit),
        "hard_hold_hand": False,
        "visual_only_markers": True,
        "disable_thumb_mesh_collision": True,
        "disable_distal_mesh_collision": True,
        "thumb_tip_proxy_pos": None,
        "thumb_tip_proxy_size": None,
        "selected": "M2.7 tripod fingertip support reset; index/middle/ring proxy-only support first, thumb light later",
        "tuning_summary": {key: value for key, value in best.items() if key != "qpos"},
        "command": "",
    }
    preset_path.write_text(json.dumps(preset, indent=2, ensure_ascii=False), encoding="utf-8")
    preset["command"] = command_for_preset(preset_path)
    preset_path.write_text(json.dumps(preset, indent=2, ensure_ascii=False), encoding="utf-8")

    initial_png = output_dir / f"{stem}_initial.png"
    final_png = output_dir / f"{stem}_final.png"
    render_preset(args, preset, initial_png, steps=0)
    render_preset(args, preset, final_png, steps=int(args.episode_seconds * 100))
    return preset_path, initial_png, final_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--base-preset", default="dex5_penspin/reports/manus_mujoco_presets/current_manus_reset.json")
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m2_7_tripod_fingertip_reset")
    parser.add_argument("--preset-dir", default="dex5_penspin/reports/manus_mujoco_presets")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--physical-top-k", type=int, default=160)
    parser.add_argument("--episode-seconds", type=float, default=2.0)
    parser.add_argument("--actuator-kp", type=float, default=400.0)
    parser.add_argument("--actuator-kv", type=float, default=20.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    preset_dir = Path(args.preset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preset_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_preset)
    base_data = json.loads(base_path.read_text(encoding="utf-8"))
    base_qpos = np.asarray(base_data["hand_qpos"], dtype=np.float64)

    print("M2.7 tripod fingertip reset tuning")
    print(f"model_path: {args.model_path}")
    print(f"base_preset: {base_path}")
    print(f"samples: {args.samples}, physical_top_k: {args.physical_top_k}")

    kin_rows = build_kinematic_candidates(args, base_qpos)
    if not kin_rows:
        raise RuntimeError("No kinematic candidates found. Broaden qpos or offset search ranges.")
    kin_path = output_dir / "kinematic_candidates.json"
    kin_path.write_text(json.dumps(kin_rows[: args.physical_top_k * 2], indent=2), encoding="utf-8")
    print(f"kinematic candidates: {len(kin_rows)}")
    print("top kinematic:")
    for row in kin_rows[:5]:
        print(
            f"  kin={row['kin_score']:.3f} span={row['span']:.4f} "
            f"dists=({row['index_axis_dist']:.4f},{row['middle_axis_dist']:.4f},{row['ring_axis_dist']:.4f}) "
            f"pen={tuple(round(v, 4) for v in row['pen_pos'])}"
        )

    physical_rows: list[dict[str, Any]] = []
    selected = kin_rows[: args.physical_top_k]
    for i, candidate in enumerate(selected, start=1):
        row = evaluate_candidate(args, candidate)
        physical_rows.append(row)
        if i % 20 == 0 or i == len(selected):
            best_so_far = max(physical_rows, key=lambda r: r["score"])
            print(
                f"eval {i:03d}/{len(selected)} "
                f"best score={best_so_far['score']:.2f} tripod={best_so_far['tripod_frames']} "
                f"min={best_so_far['support_min_frames']} drift={best_so_far['drift']:.4f}"
            )

    physical_rows.sort(key=lambda row: row["score"], reverse=True)
    csv_path = output_dir / "physical_candidates.csv"
    write_csv(csv_path, physical_rows)

    best = physical_rows[0]
    preset_path, initial_png, final_png = save_best_preset(args, base_data, best, output_dir, preset_dir)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preset_path": str(preset_path),
        "csv_path": str(csv_path),
        "kinematic_path": str(kin_path),
        "initial_png": str(initial_png),
        "final_png": str(final_png),
        "best": best,
        "top10": physical_rows[:10],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nBEST")
    for key in [
        "score",
        "tripod_frames",
        "two_or_more_support_frames",
        "support_min_frames",
        "index_frames",
        "middle_frames",
        "ring_frames",
        "thumb_frames",
        "pinky_frames",
        "drift",
        "xy_drift",
        "z_delta",
        "axis_angle",
        "max_penetration",
        "terminated",
    ]:
        print(f"{key}: {best[key]}")
    print(f"preset: {preset_path}")
    print(f"summary: {summary_path}")
    print(f"csv: {csv_path}")
    print(f"initial_png: {initial_png}")
    print(f"final_png: {final_png}")
    print("command:")
    print(json.loads(preset_path.read_text(encoding="utf-8"))["command"])


if __name__ == "__main__":
    main()
