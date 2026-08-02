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
import numpy as np

from dex5_penspin_env import Dex5PenSpinMinimalEnv
from run_m2_9_micro_gait import load_preset, make_env, reset_to_preset
from tune_m2_7_tripod_fingertip_reset import SUPPORT_FINGERS, contact_snapshot
from view_m2_mujoco import action_for_policy


def phase_monotonicity(phases: np.ndarray, direction: float) -> dict[str, float]:
    sign = float(np.sign(direction) or 1.0)
    directed = sign * (phases - phases[0])
    deltas = np.diff(directed)
    positive = float(np.sum(np.clip(deltas, 0.0, None)))
    negative = float(np.sum(np.clip(-deltas, 0.0, None)))
    total = positive + negative
    running_max = np.maximum.accumulate(directed)
    drawdown = running_max - directed
    final_progress = float(directed[-1])
    return {
        "final_progress": final_progress,
        "positive_progress": positive,
        "negative_progress": negative,
        "monotonic_ratio": float(positive / (total + 1e-9)),
        "max_drawdown": float(np.max(drawdown)),
        "peak_progress": float(np.max(directed)),
        "raw_phase_final": float(phases[-1]),
    }


def evaluate_directed_policy(
    args: argparse.Namespace,
    preset: dict[str, Any],
    config: dict[str, Any],
    render: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    env = make_env(args, preset, duration=args.duration, render_mode="rgb_array" if render else None)
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
    max_penetration = float(initial_penetration)
    frames: list[np.ndarray] = []
    terminated = False
    truncated = False
    returns = 0.0
    if render:
        frames.append(env.render())

    step = 0
    max_steps = int(round(args.duration * env.control_hz))
    while step < max_steps:
        action = action_for_policy(
            env,
            "directed_micro_gait",
            step,
            config["action_scale"],
            phase_preset=config["phase_preset"],
            center_gain=config["center_gain"],
            target_center=preset["pen_pos"],
            cycle_steps=config["cycle_steps"],
            thumb_scale=config["thumb_scale"],
            phase_direction=config["phase_direction"],
            return_scale=config["return_scale"],
            roll_scale=config["roll_scale"],
        )
        _, reward, terminated, truncated, info = env.step(action)
        returns += float(reward)

        fingers, geoms, penetration = contact_snapshot(env)
        geom_counter.update(geoms)
        for finger in fingers:
            finger_frames[finger] += 1
        if set(SUPPORT_FINGERS).issubset(fingers):
            tripod_frames += 1
        if set(SUPPORT_FINGERS).issubset(fingers) and "thumb" in fingers:
            quad_frames += 1
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
    phase_metrics = phase_monotonicity(phases_np, config["phase_direction"])

    steps = max(1, step)
    survival_seconds = float(step / env.control_hz)
    tripod_ratio = float(tripod_frames / steps)
    thumb_ratio = float(finger_frames["thumb"] / steps)
    support_min_frames = min(finger_frames[finger] for finger in SUPPORT_FINGERS)
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

    score = (
        22.0 * max(0.0, phase_metrics["final_progress"])
        + 7.0 * phase_metrics["monotonic_ratio"]
        + 4.0 * tripod_ratio
        + 1.0 * min(thumb_ratio, 0.7)
        + 0.20 * survival_seconds
        - 12.0 * phase_metrics["negative_progress"]
        - 9.0 * phase_metrics["max_drawdown"]
        - 85.0 * xy_drift
        - 40.0 * abs(z_delta)
        - 11.0 * float(np.max(endpoint_np))
        - 4.0 * axis_angle
        - 220.0 * max(0.0, max_penetration - 0.006)
    )
    if survival_seconds < args.duration:
        score -= 8.0
    if tripod_ratio < args.min_tripod_ratio:
        score -= 10.0 + 20.0 * (args.min_tripod_ratio - tripod_ratio)
    if phase_metrics["final_progress"] < args.min_final_progress:
        score -= 6.0 + 10.0 * (args.min_final_progress - phase_metrics["final_progress"])
    if phase_metrics["monotonic_ratio"] < args.min_monotonic_ratio:
        score -= 4.0 + 8.0 * (args.min_monotonic_ratio - phase_metrics["monotonic_ratio"])
    if phase_metrics["max_drawdown"] > args.max_drawdown:
        score -= 3.0 + 7.0 * (phase_metrics["max_drawdown"] - args.max_drawdown)
    if xy_drift > args.max_xy_drift:
        score -= 6.0 + 120.0 * (xy_drift - args.max_xy_drift)
    if float(np.max(endpoint_np)) > args.max_endpoint_dz:
        score -= 4.0 + 35.0 * (float(np.max(endpoint_np)) - args.max_endpoint_dz)
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
        "max_penetration": float(max_penetration),
        "mean_ncon": float(np.mean(ncons)) if ncons else 0.0,
        "top_geom_contacts": geom_counter.most_common(10),
        **phase_metrics,
    }

    traces = None
    if render:
        directed = float(np.sign(config["phase_direction"]) or 1.0) * (phases_np - phases_np[0])
        traces = {
            "frames": np.asarray(frames, dtype=np.uint8),
            "centers": centers_np,
            "phases": phases_np,
            "directed_phases": directed,
            "endpoint_height_diffs": endpoint_np,
        }
    env.close()
    return row, traces


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "score",
        "action_scale",
        "phase_preset",
        "cycle_steps",
        "thumb_scale",
        "center_gain",
        "phase_direction",
        "return_scale",
        "roll_scale",
        "survival_seconds",
        "terminated",
        "final_progress",
        "positive_progress",
        "negative_progress",
        "monotonic_ratio",
        "max_drawdown",
        "peak_progress",
        "raw_phase_final",
        "tripod_ratio",
        "thumb_ratio",
        "tripod_frames",
        "quad_frames",
        "xy_drift",
        "z_delta",
        "axis_angle",
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
    directed = traces["directed_phases"]
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
    axes[1].plot(np.arange(len(phases)), phases, label="raw phase")
    axes[1].plot(np.arange(len(directed)), directed, label="directed phase")
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
        "--policy directed_micro_gait "
        f"--action-scale {float(row['action_scale']):.6f} "
        f"--phase-preset {row['phase_preset']} "
        f"--cycle-steps {int(row['cycle_steps'])} "
        f"--thumb-scale {float(row['thumb_scale']):.6f} "
        f"--center-gain {float(row['center_gain']):.6f} "
        f"--phase-direction {float(row['phase_direction']):.1f} "
        f"--return-scale {float(row['return_scale']):.6f} "
        f"--roll-scale {float(row['roll_scale']):.6f} "
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
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m2_10_directed_micro_gait")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--action-scales", nargs="+", type=float, default=[0.00015, 0.0002, 0.00025, 0.0003])
    parser.add_argument("--phase-presets", nargs="+", default=["m2_3", "thumb_late", "paired"])
    parser.add_argument("--cycle-steps", nargs="+", type=int, default=[320, 420])
    parser.add_argument("--thumb-scales", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--center-gains", nargs="+", type=float, default=[0.005, 0.01])
    parser.add_argument("--phase-directions", nargs="+", type=float, default=[1.0, -1.0])
    parser.add_argument("--return-scales", nargs="+", type=float, default=[0.85, 1.0])
    parser.add_argument("--roll-scales", nargs="+", type=float, default=[0.25])
    parser.add_argument("--actuator-kp", type=float, default=400.0)
    parser.add_argument("--actuator-kv", type=float, default=20.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--min-tripod-ratio", type=float, default=0.80)
    parser.add_argument("--min-final-progress", type=float, default=0.50)
    parser.add_argument("--min-monotonic-ratio", type=float, default=0.65)
    parser.add_argument("--max-drawdown", type=float, default=0.30)
    parser.add_argument("--max-xy-drift", type=float, default=0.006)
    parser.add_argument("--max-endpoint-dz", type=float, default=0.012)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preset = load_preset(args.preset)

    print("M2.10 directed micro-gait evaluation")
    print(f"preset: {args.preset}")
    print(f"output_dir: {output_dir}")
    total = (
        len(args.action_scales)
        * len(args.phase_presets)
        * len(args.cycle_steps)
        * len(args.thumb_scales)
        * len(args.center_gains)
        * len(args.phase_directions)
        * len(args.return_scales)
        * len(args.roll_scales)
    )
    print(f"candidates: {total}")

    rows: list[dict[str, Any]] = []
    i = 0
    for action_scale in args.action_scales:
        for phase_preset in args.phase_presets:
            for cycle_steps in args.cycle_steps:
                for thumb_scale in args.thumb_scales:
                    for center_gain in args.center_gains:
                        for phase_direction in args.phase_directions:
                            for return_scale in args.return_scales:
                                for roll_scale in args.roll_scales:
                                    config = {
                                        "action_scale": action_scale,
                                        "phase_preset": phase_preset,
                                        "cycle_steps": cycle_steps,
                                        "thumb_scale": thumb_scale,
                                        "center_gain": center_gain,
                                        "phase_direction": phase_direction,
                                        "return_scale": return_scale,
                                        "roll_scale": roll_scale,
                                    }
                                    row, _ = evaluate_directed_policy(args, preset, config, render=False)
                                    row["viewer_command"] = viewer_command(preset, row)
                                    rows.append(row)
                                    i += 1
                                    if i % 24 == 0 or i == total:
                                        best_so_far = max(rows, key=lambda r: r["score"])
                                        print(
                                            f"eval {i:03d}/{total} "
                                            f"best score={best_so_far['score']:.2f} "
                                            f"progress={best_so_far['final_progress']:.3f} "
                                            f"mono={best_so_far['monotonic_ratio']:.3f} "
                                            f"tripod={best_so_far['tripod_ratio']:.3f}"
                                        )

    rows.sort(key=lambda row: row["score"], reverse=True)
    valid_rows = [
        row
        for row in rows
        if row["survival_seconds"] >= args.duration
        and row["tripod_ratio"] >= args.min_tripod_ratio
        and row["final_progress"] >= args.min_final_progress
        and row["monotonic_ratio"] >= args.min_monotonic_ratio
        and row["max_drawdown"] <= args.max_drawdown
        and row["xy_drift"] <= args.max_xy_drift
        and row["endpoint_height_diff_max"] <= args.max_endpoint_dz
        and not row["terminated"]
    ]
    best = valid_rows[0] if valid_rows else rows[0]
    best_row, best_traces = evaluate_directed_policy(args, preset, best, render=True)
    best_row["viewer_command"] = viewer_command(preset, best_row)

    csv_path = output_dir / "directed_micro_gait_candidates.csv"
    write_csv(csv_path, rows)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preset": args.preset,
        "valid_count": len(valid_rows),
        "best": best_row,
        "top10": rows[:10],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if best_traces is not None:
        imageio.mimsave(output_dir / "best_directed_micro_gait.gif", list(best_traces["frames"]), fps=20)
        plot_trace(best_traces, output_dir / "best_directed_micro_gait_trace.png", "M2.10 best directed micro-gait")

    viewer_path = output_dir / "viewer_commands.sh"
    viewer_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -e",
                "",
                "# Best directed micro-gait",
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
        "action_scale",
        "phase_preset",
        "cycle_steps",
        "thumb_scale",
        "center_gain",
        "phase_direction",
        "return_scale",
        "roll_scale",
        "survival_seconds",
        "final_progress",
        "positive_progress",
        "negative_progress",
        "monotonic_ratio",
        "max_drawdown",
        "tripod_ratio",
        "thumb_ratio",
        "xy_drift",
        "endpoint_height_diff_max",
        "max_penetration",
        "terminated",
    ]:
        print(f"{key}: {best_row[key]}")
    print(f"summary: {summary_path}")
    print(f"csv: {csv_path}")
    print(f"viewer_commands: {viewer_path}")
    print("viewer command:")
    print(best_row["viewer_command"])


if __name__ == "__main__":
    main()
