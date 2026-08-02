from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from teleop_m2_initial_state import command_for_preset
from tune_m2_7_tripod_fingertip_reset import (
    SUPPORT_FINGERS,
    contact_snapshot,
    point_line_distances,
    set_state,
)


THUMB_TARGET_DISTANCE = 0.0185
THUMB_MIN_DISTANCE = 0.0130
THUMB_MAX_DISTANCE = 0.0270


def thumb_kinematic_candidates(args: argparse.Namespace, base: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=base["pen_pos"],
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

    base_qpos = np.asarray(base["hand_qpos"], dtype=np.float64)
    pen_pos = np.asarray(base["pen_pos"], dtype=np.float64)
    pen_quat = np.asarray(base["pen_quat"], dtype=np.float64)
    thumb_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "Link_14R_tip_proxy")

    rows: list[dict[str, Any]] = []
    current_thumb = base_qpos[16:20].copy()

    seeds = [
        current_thumb,
        np.array([0.68, 1.20, 1.12, 0.38], dtype=np.float64),
        np.array([0.68, 1.32, 1.10, 0.50], dtype=np.float64),
        np.array([0.55, 1.30, 1.20, 0.25], dtype=np.float64),
        np.array([0.55, 1.45, 1.10, 0.50], dtype=np.float64),
    ]

    def sample_thumb(sample_i: int) -> np.ndarray:
        if sample_i < len(seeds):
            return seeds[sample_i].copy()
        if rng.random() < 0.35:
            center = seeds[int(rng.integers(1, len(seeds)))]
            noise = rng.normal([0.0, 0.0, 0.0, 0.0], [0.08, 0.12, 0.12, 0.14])
            return center + noise
        return np.array(
            [
                rng.uniform(0.20, 0.6803),
                rng.uniform(0.70, 1.55),
                rng.uniform(0.65, 1.45),
                rng.uniform(0.00, 1.05),
            ],
            dtype=np.float64,
        )

    for sample_i in range(args.samples):
        qpos = base_qpos.copy()
        qpos[16:20] = sample_thumb(sample_i)
        qpos = np.clip(qpos, env.joint_low, env.joint_high)
        set_state(env, qpos, pen_pos, pen_quat)
        state = env.get_state()
        thumb_pos = env.data.geom_xpos[thumb_gid].copy()
        dist = float(point_line_distances(thumb_pos[None, :], state.pen_center, state.pen_axis)[0])
        axis_coord = float(np.dot(thumb_pos - state.pen_center, state.pen_axis))
        if dist < THUMB_MIN_DISTANCE or dist > THUMB_MAX_DISTANCE:
            continue
        if abs(axis_coord) > 0.085:
            continue

        delta = float(np.linalg.norm(qpos[16:20] - current_thumb))
        distance_error = abs(dist - THUMB_TARGET_DISTANCE)
        # Prefer a thumb that can lightly touch the pen without being deeply
        # inside the cylinder/proxy sum and without requiring an extreme pose.
        kin_score = (
            -260.0 * distance_error
            -0.55 * delta
            -10.0 * max(0.0, THUMB_TARGET_DISTANCE - dist)
            -2.0 * abs(axis_coord)
        )
        rows.append(
            {
                "sample_i": sample_i,
                "qpos": qpos.tolist(),
                "thumb_qpos": qpos[16:20].tolist(),
                "thumb_tip_pos": thumb_pos.tolist(),
                "thumb_axis_dist": dist,
                "thumb_axis_coord": axis_coord,
                "thumb_delta": delta,
                "kin_score": float(kin_score),
                "pen_pos": base["pen_pos"],
                "pen_quat": base["pen_quat"],
            }
        )

    env.close()
    rows.sort(key=lambda row: row["kin_score"], reverse=True)
    return rows


def evaluate_candidate(args: argparse.Namespace, base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=base["pen_pos"],
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
        np.asarray(base["pen_pos"], dtype=np.float64),
        np.asarray(base["pen_quat"], dtype=np.float64),
    )

    initial_fingers, initial_geoms, initial_penetration = contact_snapshot(env)
    center0 = env.get_state().pen_center.copy()
    axis0 = env.get_state().pen_axis.copy()
    centers = [center0]
    axes = [axis0]

    geom_counter: Counter[str] = Counter()
    finger_frames: Counter[str] = Counter()
    tripod_frames = 0
    quad_frames = 0
    support_two_or_more_frames = 0
    max_penetration = float(initial_penetration)
    ncons: list[int] = []
    terminated = False

    for _ in range(env.max_episode_steps):
        _, _, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
        fingers, geoms, penetration = contact_snapshot(env)
        geom_counter.update(geoms)
        for finger in fingers:
            finger_frames[finger] += 1
        support_hit = set(SUPPORT_FINGERS).intersection(fingers)
        if set(SUPPORT_FINGERS).issubset(fingers):
            tripod_frames += 1
        if set(SUPPORT_FINGERS).issubset(fingers) and "thumb" in fingers:
            quad_frames += 1
        if len(support_hit) >= 2:
            support_two_or_more_frames += 1
        max_penetration = max(max_penetration, float(penetration))
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
        np.arccos(
            np.clip(
                np.dot(axes_np[0], axes_np[-1]) / (np.linalg.norm(axes_np[0]) * np.linalg.norm(axes_np[-1]) + 1e-9),
                -1.0,
                1.0,
            )
        )
    )

    support_min_frames = min(finger_frames[finger] for finger in SUPPORT_FINGERS)
    thumb_frames = int(finger_frames["thumb"])
    thumb_target = 70.0
    thumb_light_score = 9.0 - 0.075 * abs(thumb_frames - thumb_target)
    if thumb_frames < args.thumb_min_frames:
        thumb_light_score -= 5.0 + 0.10 * (args.thumb_min_frames - thumb_frames)
    if thumb_frames > args.thumb_max_frames:
        thumb_light_score -= 6.0 + 0.08 * (thumb_frames - args.thumb_max_frames)

    score = (
        0.20 * tripod_frames
        + 0.05 * support_two_or_more_frames
        + 0.05 * support_min_frames
        + 0.025 * quad_frames
        + thumb_light_score
        - 0.018 * finger_frames["pinky"]
        - 145.0 * drift
        - 70.0 * abs(z_delta)
        - 9.0 * axis_angle
        - 260.0 * max(0.0, max_penetration - 0.006)
    )
    if tripod_frames < args.tripod_min_frames:
        score -= 12.0 + 0.10 * (args.tripod_min_frames - tripod_frames)
    if support_min_frames < args.tripod_min_frames:
        score -= 8.0
    if terminated:
        score -= 4.0

    row = {
        **candidate,
        "score": float(score),
        "initial_support_fingers": sorted(initial_fingers),
        "initial_tip_geoms": initial_geoms.most_common(10),
        "initial_penetration": float(initial_penetration),
        "tripod_frames": int(tripod_frames),
        "quad_frames": int(quad_frames),
        "support_two_or_more_frames": int(support_two_or_more_frames),
        "index_frames": int(finger_frames["index"]),
        "middle_frames": int(finger_frames["middle"]),
        "ring_frames": int(finger_frames["ring"]),
        "thumb_frames": thumb_frames,
        "pinky_frames": int(finger_frames["pinky"]),
        "support_min_frames": int(support_min_frames),
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
        "thumb_qpos",
        "thumb_axis_dist",
        "thumb_axis_coord",
        "thumb_delta",
        "tripod_frames",
        "quad_frames",
        "support_two_or_more_frames",
        "index_frames",
        "middle_frames",
        "ring_frames",
        "thumb_frames",
        "pinky_frames",
        "support_min_frames",
        "drift",
        "xy_drift",
        "z_delta",
        "final_z",
        "axis_angle",
        "max_penetration",
        "mean_ncon",
        "terminated",
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
            for key in [
                "thumb_qpos",
                "initial_support_fingers",
                "initial_tip_geoms",
                "top_geom_contacts",
                "pen_pos",
                "pen_quat",
                "qpos",
            ]:
                out[key] = json.dumps(out[key], ensure_ascii=False)
            writer.writerow(out)


def save_best_preset(
    args: argparse.Namespace,
    base: dict[str, Any],
    best: dict[str, Any],
    output_dir: Path,
    preset_dir: Path,
) -> tuple[Path, Path, Path]:
    stem = "current_manus_reset_thumb_light_clamp_m2_8"
    preset_path = preset_dir / f"{stem}.json"
    preset = {
        "model_path": str(args.model_path),
        "hand_qpos": [float(v) for v in best["qpos"]],
        "joint_order": base.get("joint_order", DEX5_JOINT_ORDER),
        "pen_pos": [float(v) for v in base["pen_pos"]],
        "pen_quat": [float(v) for v in base["pen_quat"]],
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
        "selected": "M2.8 thumb light clamp from M2.7 tripod support; thumb tip proxy only",
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
    parser.add_argument("--model-path", default="dex5_penspin/sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml")
    parser.add_argument("--base-preset", default="dex5_penspin/reports/manus_mujoco_presets/current_manus_reset_tripod_fingertip_support_m2_7.json")
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m2_8_thumb_light_clamp")
    parser.add_argument("--preset-dir", default="dex5_penspin/reports/manus_mujoco_presets")
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--physical-top-k", type=int, default=220)
    parser.add_argument("--episode-seconds", type=float, default=2.0)
    parser.add_argument("--actuator-kp", type=float, default=400.0)
    parser.add_argument("--actuator-kv", type=float, default=20.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--tripod-min-frames", type=int, default=160)
    parser.add_argument("--thumb-min-frames", type=int, default=20)
    parser.add_argument("--thumb-max-frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    preset_dir = Path(args.preset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preset_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_preset)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    print("M2.8 thumb light clamp tuning")
    print(f"base_preset: {base_path}")
    print(f"samples: {args.samples}, physical_top_k: {args.physical_top_k}")

    kin_rows = thumb_kinematic_candidates(args, base)
    if not kin_rows:
        raise RuntimeError("No thumb kinematic candidates found. Broaden thumb search ranges.")
    kin_path = output_dir / "thumb_kinematic_candidates.json"
    kin_path.write_text(json.dumps(kin_rows[: args.physical_top_k * 2], indent=2), encoding="utf-8")
    print(f"kinematic candidates: {len(kin_rows)}")
    print("top kinematic:")
    for row in kin_rows[:5]:
        print(
            f"  kin={row['kin_score']:.3f} thumb_dist={row['thumb_axis_dist']:.4f} "
            f"axis_s={row['thumb_axis_coord']:.4f} q={tuple(round(v, 3) for v in row['thumb_qpos'])}"
        )

    selected = kin_rows[: args.physical_top_k]
    physical_rows: list[dict[str, Any]] = []
    for i, candidate in enumerate(selected, start=1):
        row = evaluate_candidate(args, base, candidate)
        physical_rows.append(row)
        if i % 25 == 0 or i == len(selected):
            best_so_far = max(physical_rows, key=lambda r: r["score"])
            print(
                f"eval {i:03d}/{len(selected)} "
                f"best score={best_so_far['score']:.2f} tripod={best_so_far['tripod_frames']} "
                f"thumb={best_so_far['thumb_frames']} drift={best_so_far['drift']:.4f}"
            )

    physical_rows.sort(key=lambda row: row["score"], reverse=True)
    valid_rows = [
        row
        for row in physical_rows
        if row["tripod_frames"] >= args.tripod_min_frames
        and args.thumb_min_frames <= row["thumb_frames"] <= args.thumb_max_frames
        and row["xy_drift"] <= 0.006
        and row["max_penetration"] <= 0.006
        and not row["terminated"]
    ]
    best = valid_rows[0] if valid_rows else physical_rows[0]

    csv_path = output_dir / "physical_candidates.csv"
    write_csv(csv_path, physical_rows)
    preset_path, initial_png, final_png = save_best_preset(args, base, best, output_dir, preset_dir)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_preset": str(base_path),
        "preset_path": str(preset_path),
        "csv_path": str(csv_path),
        "kinematic_path": str(kin_path),
        "initial_png": str(initial_png),
        "final_png": str(final_png),
        "valid_count": len(valid_rows),
        "best": best,
        "top10": physical_rows[:10],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nBEST")
    print(f"valid_count: {len(valid_rows)}")
    for key in [
        "score",
        "tripod_frames",
        "quad_frames",
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
        "thumb_qpos",
        "thumb_axis_dist",
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
