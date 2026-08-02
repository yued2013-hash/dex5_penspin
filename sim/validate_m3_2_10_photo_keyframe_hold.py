from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from build_m3_2_9_photo_five_phase_debug import PHASE_SPECS, set_state
from dex5_penspin_env import Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import contact_snapshot


def make_env(preset: dict[str, Any], *, duration: float, render: bool, hard_hold_hand: bool) -> Dex5PenSpinMinimalEnv:
    return Dex5PenSpinMinimalEnv(
        model_path=preset["model_path"],
        pen_init_pos=preset["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=duration,
        actuator_kp=preset.get("actuator_kp", 400.0),
        actuator_kv=preset.get("actuator_kv", 20.0),
        thumb_force_limit=preset.get("thumb_force_limit", 0.0),
        hard_hold_hand=hard_hold_hand,
        visual_only_markers=bool(preset.get("visual_only_markers", True)),
        disable_thumb_mesh_collision=bool(preset.get("disable_thumb_mesh_collision", True)),
        disable_distal_mesh_collision=bool(preset.get("disable_distal_mesh_collision", True)),
        render_mode="rgb_array" if render else None,
        seed=0,
    )


def reset_to_preset(env: Dex5PenSpinMinimalEnv, preset: dict[str, Any]) -> None:
    set_state(
        env,
        np.asarray(preset["hand_qpos"], dtype=np.float64),
        np.asarray(preset["pen_pos"], dtype=np.float64),
        np.asarray(preset["pen_quat"], dtype=np.float64),
    )


def validate_one(
    preset_path: Path,
    preset: dict[str, Any],
    *,
    duration: float,
    render_every: int,
    output_dir: Path,
    hard_hold_hand: bool,
) -> dict[str, Any]:
    env = make_env(preset, duration=duration, render=True, hard_hold_hand=hard_hold_hand)
    env.reset(seed=0)
    reset_to_preset(env, preset)

    center0 = env.get_state().pen_center.copy()
    axis0 = env.get_state().pen_axis.copy()
    initial_fingers, _, initial_penetration = contact_snapshot(env)

    frames = [env.render()]
    contact_counter: Counter[str] = Counter()
    exact_counter: Counter[str] = Counter()
    max_penetration = float(initial_penetration)
    min_height = float(center0[2])
    final_center = center0.copy()
    final_axis = axis0.copy()
    terminated = False
    truncated = False
    step = 0
    max_steps = int(round(duration * env.control_hz))
    action = np.zeros(20, dtype=np.float32)

    while step < max_steps:
        _, _, terminated, truncated, info = env.step(action)
        fingers, _, penetration = contact_snapshot(env)
        for finger in fingers:
            contact_counter[finger] += 1
        exact_counter[",".join(sorted(fingers)) or "none"] += 1
        max_penetration = max(max_penetration, float(penetration))
        final_center = np.asarray(info["pen_center"], dtype=np.float64).copy()
        final_axis = np.asarray(info["pen_axis"], dtype=np.float64).copy()
        min_height = min(min_height, float(final_center[2]))
        step += 1
        if step % render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break

    if step % render_every != 0:
        frames.append(env.render())

    survival_seconds = float(step / env.control_hz)
    drift = float(np.linalg.norm(final_center - center0))
    xy_drift = float(np.linalg.norm(final_center[:2] - center0[:2]))
    z_delta = float(final_center[2] - center0[2])
    axis_angle = float(np.rad2deg(np.arccos(np.clip(np.dot(axis0, final_axis), -1.0, 1.0))))
    desired = set(preset.get("desired_contacts", []))
    desired_frame_ratio = 0.0
    if desired and step > 0:
        desired_frame_ratio = sum(
            count
            for exact, count in exact_counter.items()
            if desired.issubset(set(exact.split(",")) if exact != "none" else set())
        ) / step

    gif_path = output_dir / f"P{preset['phase']}_{preset['name']}_zero_action_hold.gif"
    imageio.mimsave(gif_path, frames, duration=render_every / env.control_hz)
    env.close()

    return {
        "phase": int(preset["phase"]),
        "name": preset["name"],
        "preset_path": str(preset_path),
        "desired_contacts": ",".join(preset.get("desired_contacts", [])),
        "initial_contacts": ",".join(sorted(initial_fingers)),
        "survival_seconds": survival_seconds,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "drift": drift,
        "xy_drift": xy_drift,
        "z_delta": z_delta,
        "min_height": min_height,
        "axis_angle_deg": axis_angle,
        "max_penetration": max_penetration,
        "desired_frame_ratio": desired_frame_ratio,
        "contact_counter": dict(contact_counter),
        "exact_contact_counter": dict(exact_counter),
        "gif": str(gif_path),
        "hard_hold_hand": bool(hard_hold_hand),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hold_validation_m3_2_10",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--hard-hold-hand", action="store_true")
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in PHASE_SPECS:
        path = seed_dir / f"P{spec['phase']}_{spec['name']}_hand_pen_tuned_seed.json"
        preset = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            validate_one(
                path,
                preset,
                duration=args.duration,
                render_every=args.render_every,
                output_dir=output_dir,
                hard_hold_hand=args.hard_hold_hand,
            )
        )

    csv_path = output_dir / "zero_action_hold_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "stage": "M3.2.10 zero-action hold validation",
        "seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "duration": args.duration,
        "hard_hold_hand": bool(args.hard_hold_hand),
        "rows": rows,
        "csv": str(csv_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.10 Zero-Action Hold Validation",
        "",
        "| phase | name | survival_s | initial contacts | desired frame ratio | drift | axis angle deg |",
        "|---:|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        readme.append(
            f"| {row['phase']} | {row['name']} | {row['survival_seconds']:.2f} | "
            f"{row['initial_contacts'] or 'none'} | {row['desired_frame_ratio']:.2f} | "
            f"{row['drift']:.4f} | {row['axis_angle_deg']:.1f} |"
        )
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
