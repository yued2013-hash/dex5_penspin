from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from dex5_penspin_env import Dex5PenSpinMinimalEnv
from tune_m2_7_tripod_fingertip_reset import SUPPORT_FINGERS, contact_snapshot, set_state
from view_m2_mujoco import action_for_policy


def load_preset(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reset_to_preset(env: Dex5PenSpinMinimalEnv, preset: dict[str, Any]) -> None:
    set_state(
        env,
        np.asarray(preset["hand_qpos"], dtype=np.float64),
        np.asarray(preset["pen_pos"], dtype=np.float64),
        np.asarray(preset["pen_quat"], dtype=np.float64),
    )
    # Avoid counting the initial pen quaternion as a spin phase jump.
    env.prev_phase = env._quat_twist_phase_about_x(env.data.xquat[env.pen_body_id])
    env.rotation_phase = 0.0


def make_env(args: argparse.Namespace, preset: dict[str, Any], duration: float, render_mode: str | None = None) -> Dex5PenSpinMinimalEnv:
    return Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=preset["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=duration,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        seed=args.seed,
        render_mode=render_mode,
    )


def evaluate_policy(args: argparse.Namespace, preset: dict[str, Any], config: dict[str, Any], duration: float, render: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    env = make_env(args, preset, duration=duration, render_mode="rgb_array" if render else None)
    env.reset(seed=args.seed)
    reset_to_preset(env, preset)

    initial_fingers, initial_geoms, initial_penetration = contact_snapshot(env)
    center0 = env.get_state().pen_center.copy()
    axis0 = env.get_state().pen_axis.copy()
    endpoint0 = env.get_state().pen_endpoints.copy()

    centers = [center0]
    axes = [axis0]
    phases = [0.0]
    endpoint_height_diffs = [float(abs(endpoint0[0, 2] - endpoint0[1, 2]))]
    ncons: list[int] = []
    geom_counter: Counter[str] = Counter()
    finger_frames: Counter[str] = Counter()
    tripod_frames = 0
    quad_frames = 0
    support_two_or_more_frames = 0
    max_penetration = float(initial_penetration)
    returns = 0.0
    terminated = False
    truncated = False
    frames: list[np.ndarray] = []
    if render:
        frames.append(env.render())

    step = 0
    max_steps = int(round(duration * env.control_hz))
    while step < max_steps:
        action = action_for_policy(
            env,
            config["policy"],
            step,
            config.get("action_scale", 0.0),
            phase_preset=config.get("phase_preset", "m2_3"),
            center_gain=config.get("center_gain", 0.0),
            target_center=preset["pen_pos"],
            cycle_steps=config.get("cycle_steps", 220),
            thumb_scale=config.get("thumb_scale", 0.25),
        )
        _, reward, terminated, truncated, info = env.step(action)
        returns += float(reward)
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
        phases.append(float(info["rotation_phase"]))
        endpoints = np.asarray(info["obj_ends"], dtype=np.float64)
        endpoint_height_diffs.append(float(abs(endpoints[0, 2] - endpoints[1, 2])))
        ncons.append(int(env.data.ncon))
        step += 1

        if render and step % args.render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break

    if render and (not frames or step % args.render_every != 0):
        frames.append(env.render())

    centers_np = np.asarray(centers, dtype=np.float64)
    axes_np = np.asarray(axes, dtype=np.float64)
    phases_np = np.asarray(phases, dtype=np.float64)
    endpoint_np = np.asarray(endpoint_height_diffs, dtype=np.float64)
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
    steps = max(1, step)
    support_min_frames = min(finger_frames[finger] for finger in SUPPORT_FINGERS)
    rotation_phase_abs = float(abs(phases_np[-1]))
    rotation_rounds = float(rotation_phase_abs / (2.0 * np.pi))
    survival_seconds = float(step / env.control_hz)
    tripod_ratio = float(tripod_frames / steps)
    thumb_ratio = float(finger_frames["thumb"] / steps)

    # M2.9 score: reward visible phase progress, but protect the reset quality.
    score = (
        18.0 * rotation_phase_abs
        + 4.0 * tripod_ratio
        + 1.5 * min(thumb_ratio, 0.7)
        + 0.25 * survival_seconds
        - 65.0 * xy_drift
        - 35.0 * abs(z_delta)
        - 7.0 * float(np.max(endpoint_np))
        - 3.0 * axis_angle
        - 200.0 * max(0.0, max_penetration - 0.006)
    )
    if tripod_ratio < args.min_tripod_ratio:
        score -= 8.0 + 18.0 * (args.min_tripod_ratio - tripod_ratio)
    if not (args.min_thumb_ratio <= thumb_ratio <= args.max_thumb_ratio):
        score -= 2.0 + 4.0 * min(abs(thumb_ratio - args.min_thumb_ratio), abs(thumb_ratio - args.max_thumb_ratio))
    if xy_drift > args.max_xy_drift:
        score -= 5.0 + 100.0 * (xy_drift - args.max_xy_drift)
    if terminated:
        score -= 8.0

    row = {
        **config,
        "score": float(score),
        "return": float(returns),
        "steps": int(step),
        "survival_seconds": survival_seconds,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "initial_support_fingers": sorted(initial_fingers),
        "initial_tip_geoms": initial_geoms.most_common(10),
        "tripod_frames": int(tripod_frames),
        "quad_frames": int(quad_frames),
        "support_two_or_more_frames": int(support_two_or_more_frames),
        "index_frames": int(finger_frames["index"]),
        "middle_frames": int(finger_frames["middle"]),
        "ring_frames": int(finger_frames["ring"]),
        "thumb_frames": int(finger_frames["thumb"]),
        "pinky_frames": int(finger_frames["pinky"]),
        "support_min_frames": int(support_min_frames),
        "tripod_ratio": tripod_ratio,
        "thumb_ratio": thumb_ratio,
        "drift": drift,
        "xy_drift": xy_drift,
        "z_delta": z_delta,
        "final_z": float(centers_np[-1, 2]),
        "axis_angle": axis_angle,
        "endpoint_height_diff_initial": float(endpoint_np[0]),
        "endpoint_height_diff_final": float(endpoint_np[-1]),
        "endpoint_height_diff_max": float(np.max(endpoint_np)),
        "endpoint_height_diff_mean": float(np.mean(endpoint_np)),
        "rotation_phase_abs": rotation_phase_abs,
        "rotation_rounds": rotation_rounds,
        "max_penetration": float(max_penetration),
        "mean_ncon": float(np.mean(ncons)) if ncons else 0.0,
        "top_geom_contacts": geom_counter.most_common(10),
    }

    traces = None
    if render:
        traces = {
            "frames": np.asarray(frames, dtype=np.uint8),
            "centers": centers_np,
            "phases": phases_np,
            "endpoint_height_diffs": endpoint_np,
        }
    env.close()
    return row, traces


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "score",
        "policy",
        "action_scale",
        "phase_preset",
        "cycle_steps",
        "thumb_scale",
        "center_gain",
        "survival_seconds",
        "terminated",
        "rotation_phase_abs",
        "rotation_rounds",
        "tripod_ratio",
        "thumb_ratio",
        "tripod_frames",
        "quad_frames",
        "index_frames",
        "middle_frames",
        "ring_frames",
        "thumb_frames",
        "pinky_frames",
        "xy_drift",
        "z_delta",
        "axis_angle",
        "endpoint_height_diff_initial",
        "endpoint_height_diff_final",
        "endpoint_height_diff_max",
        "max_penetration",
        "return",
        "top_geom_contacts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows):
            out = {field: row.get(field) for field in fields}
            out["rank"] = rank
            out["top_geom_contacts"] = json.dumps(out["top_geom_contacts"], ensure_ascii=False)
            writer.writerow(out)


def plot_trace(traces: dict[str, np.ndarray], path: Path, title: str) -> None:
    centers = traces["centers"]
    phases = traces["phases"]
    endpoint = traces["endpoint_height_diffs"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = np.arange(len(centers))
    axes[0].plot(t, centers[:, 0], label="x")
    axes[0].plot(t, centers[:, 1], label="y")
    axes[0].plot(t, centers[:, 2], label="z")
    axes[0].set_title(title)
    axes[0].set_ylabel("pen center")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(np.arange(len(phases)), phases, label="phase")
    axes[1].plot(np.arange(len(phases)), phases / (2.0 * np.pi), label="rounds")
    axes[1].set_ylabel("rotation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(np.arange(len(endpoint)), endpoint, label="endpoint dz")
    axes[2].set_ylabel("endpoint dz")
    axes[2].set_xlabel("step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def viewer_command(preset: dict[str, Any], row: dict[str, Any]) -> str:
    qpos = " ".join(f"{float(v):.6f}" for v in preset["hand_qpos"])
    pen_pos = " ".join(f"{float(v):.6f}" for v in preset["pen_pos"])
    pen_quat = " ".join(f"{float(v):.6f}" for v in preset["pen_quat"])
    return (
        "python dex5_penspin/sim/view_m2_mujoco.py "
        f"--policy {row['policy']} "
        f"--action-scale {float(row.get('action_scale', 0.0)):.6f} "
        f"--phase-preset {row.get('phase_preset', 'm2_3')} "
        f"--cycle-steps {int(row.get('cycle_steps', 220))} "
        f"--thumb-scale {float(row.get('thumb_scale', 0.25)):.6f} "
        f"--center-gain {float(row.get('center_gain', 0.0)):.6f} "
        "--duration 10 --realtime --ignore-termination --pause-on-end "
        f"--pen-init-pos {pen_pos} --pen-quat {pen_quat} "
        "--actuator-kp 400.000000 --actuator-kv 20.000000 --thumb-force-limit 0.000000 "
        "--visual-only-markers --disable-thumb-mesh-collision --disable-distal-mesh-collision "
        f"--custom-qpos {qpos}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="dex5_penspin/sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml")
    parser.add_argument("--preset", default="dex5_penspin/reports/manus_mujoco_presets/current_manus_reset_thumb_light_clamp_m2_8.json")
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m2_9_micro_gait")
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    parser.add_argument("--micro-seconds", type=float, default=6.0)
    parser.add_argument("--action-scales", nargs="+", type=float, default=[0.0001, 0.0002, 0.0003, 0.0005, 0.001])
    parser.add_argument("--phase-presets", nargs="+", default=["m2_3", "paired", "quadrature", "thumb_late"])
    parser.add_argument("--cycle-steps", nargs="+", type=int, default=[220, 320])
    parser.add_argument("--thumb-scales", nargs="+", type=float, default=[0.0, 0.15, 0.25])
    parser.add_argument("--center-gains", nargs="+", type=float, default=[0.0, 0.005, 0.01])
    parser.add_argument("--actuator-kp", type=float, default=400.0)
    parser.add_argument("--actuator-kv", type=float, default=20.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--min-tripod-ratio", type=float, default=0.70)
    parser.add_argument("--min-thumb-ratio", type=float, default=0.10)
    parser.add_argument("--max-thumb-ratio", type=float, default=0.70)
    parser.add_argument("--max-xy-drift", type=float, default=0.015)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preset = load_preset(args.preset)

    print("M2.9 hold + micro-gait evaluation")
    print(f"preset: {args.preset}")
    print(f"output_dir: {output_dir}")

    hold_config = {
        "policy": "hold",
        "action_scale": 0.0,
        "phase_preset": "m2_3",
        "cycle_steps": 220,
        "thumb_scale": 0.0,
        "center_gain": 0.0,
    }
    hold_row, hold_traces = evaluate_policy(args, preset, hold_config, duration=args.hold_seconds, render=True)
    hold_row["viewer_command"] = viewer_command(preset, hold_row)
    print(
        "hold:",
        f"survival={hold_row['survival_seconds']:.2f}s",
        f"phase={hold_row['rotation_phase_abs']:.4f}",
        f"tripod={hold_row['tripod_ratio']:.3f}",
        f"thumb={hold_row['thumb_ratio']:.3f}",
        f"xy={hold_row['xy_drift']:.4f}",
        f"terminated={hold_row['terminated']}",
    )
    hold_equiv_phase = hold_row["rotation_phase_abs"] * args.micro_seconds / max(hold_row["survival_seconds"], 1e-6)

    rows: list[dict[str, Any]] = []
    total = (
        len(args.action_scales)
        * len(args.phase_presets)
        * len(args.cycle_steps)
        * len(args.thumb_scales)
        * len(args.center_gains)
    )
    print(f"micro-gait candidates: {total}")
    i = 0
    for action_scale in args.action_scales:
        for phase_preset in args.phase_presets:
            for cycle_steps in args.cycle_steps:
                for thumb_scale in args.thumb_scales:
                    for center_gain in args.center_gains:
                        config = {
                            "policy": "micro_gait",
                            "action_scale": action_scale,
                            "phase_preset": phase_preset,
                            "cycle_steps": cycle_steps,
                            "thumb_scale": thumb_scale,
                            "center_gain": center_gain,
                        }
                        row, _ = evaluate_policy(args, preset, config, duration=args.micro_seconds, render=False)
                        row["phase_gain_over_hold"] = row["rotation_phase_abs"] - hold_equiv_phase
                        row["viewer_command"] = viewer_command(preset, row)
                        rows.append(row)
                        i += 1
                        if i % 20 == 0 or i == total:
                            best_so_far = max(rows, key=lambda r: r["score"])
                            print(
                                f"eval {i:03d}/{total} "
                                f"best score={best_so_far['score']:.2f} "
                                f"phase={best_so_far['rotation_phase_abs']:.4f} "
                                f"tripod={best_so_far['tripod_ratio']:.3f} "
                                f"xy={best_so_far['xy_drift']:.4f}"
                            )

    rows.sort(key=lambda row: row["score"], reverse=True)
    valid_rows = [
        row
        for row in rows
        if row["survival_seconds"] >= args.micro_seconds
        and row["tripod_ratio"] >= args.min_tripod_ratio
        and args.min_thumb_ratio <= row["thumb_ratio"] <= args.max_thumb_ratio
        and row["xy_drift"] <= args.max_xy_drift
        and row["rotation_phase_abs"] > hold_equiv_phase
        and not row["terminated"]
    ]
    best = valid_rows[0] if valid_rows else rows[0]
    best_row, best_traces = evaluate_policy(args, preset, best, duration=args.micro_seconds, render=True)
    best_row["phase_gain_over_hold"] = best_row["rotation_phase_abs"] - hold_equiv_phase
    best_row["viewer_command"] = viewer_command(preset, best_row)

    write_csv(output_dir / "micro_gait_candidates.csv", rows)
    hold_json = output_dir / "hold_10s_summary.json"
    hold_json.write_text(json.dumps(hold_row, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preset": args.preset,
        "hold": hold_row,
        "hold_equiv_phase_for_micro_duration": hold_equiv_phase,
        "valid_count": len(valid_rows),
        "best": best_row,
        "top10": rows[:10],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if hold_traces is not None:
        imageio.mimsave(output_dir / "hold_10s.gif", list(hold_traces["frames"]), fps=20)
        plot_trace(hold_traces, output_dir / "hold_10s_trace.png", "M2.9 hold 10s")
    if best_traces is not None:
        imageio.mimsave(output_dir / "best_micro_gait.gif", list(best_traces["frames"]), fps=20)
        plot_trace(best_traces, output_dir / "best_micro_gait_trace.png", "M2.9 best micro-gait")

    viewer_path = output_dir / "viewer_commands.sh"
    viewer_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -e",
                "",
                "# Hold baseline",
                hold_row["viewer_command"],
                "",
                "# Best micro-gait",
                best_row["viewer_command"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    viewer_path.chmod(0o755)

    print("\nBEST")
    print(f"valid_count: {len(valid_rows)}")
    for key in [
        "score",
        "policy",
        "action_scale",
        "phase_preset",
        "cycle_steps",
        "thumb_scale",
        "center_gain",
        "survival_seconds",
        "rotation_phase_abs",
        "phase_gain_over_hold",
        "rotation_rounds",
        "tripod_ratio",
        "thumb_ratio",
        "xy_drift",
        "endpoint_height_diff_max",
        "max_penetration",
        "terminated",
    ]:
        print(f"{key}: {best_row[key]}")
    print(f"summary: {summary_path}")
    print(f"hold_json: {hold_json}")
    print(f"csv: {output_dir / 'micro_gait_candidates.csv'}")
    print(f"viewer_commands: {viewer_path}")
    print("viewer command:")
    print(best_row["viewer_command"])


if __name__ == "__main__":
    main()
