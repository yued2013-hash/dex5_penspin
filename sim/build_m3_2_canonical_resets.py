from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from m3_oracle_hold_env import Dex5PenSpinOracleHoldEnv, summarize_rollout
from run_m2_9_micro_gait import make_env, reset_to_preset
from view_m2_mujoco import action_for_policy


M2_7_PRESET = Path("dex5_penspin/reports/manus_mujoco_presets/current_manus_reset_tripod_fingertip_support_m2_7.json")
M2_8_PRESET = Path("dex5_penspin/reports/manus_mujoco_presets/current_manus_reset_thumb_light_clamp_m2_8.json")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_with_metadata(src: Path, dst: Path, *, reset_id: int, label: str, source: str) -> dict[str, Any]:
    data = load_json(src)
    data.update(
        {
            "reset_id": int(reset_id),
            "canonical_label": label,
            "canonical_source": source,
            "stage": "M3.2 canonical reset seed",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    write_json(dst, data)
    return data


def derive_directed_mid_reset(args: argparse.Namespace, src: Path, dst: Path) -> dict[str, Any]:
    preset = load_json(src)
    env = make_env(args, preset, duration=args.derive_seconds, render_mode=None)
    env.reset(seed=args.seed)
    reset_to_preset(env, preset)
    steps = int(round(args.derive_seconds * env.control_hz))
    for step in range(steps):
        action = action_for_policy(
            env,
            "directed_micro_gait",
            step,
            args.derive_action_scale,
            phase_preset=args.derive_phase_preset,
            center_gain=args.derive_center_gain,
            target_center=preset["pen_pos"],
            cycle_steps=args.derive_cycle_steps,
            thumb_scale=args.derive_thumb_scale,
            phase_direction=args.derive_phase_direction,
            return_scale=args.derive_return_scale,
            roll_scale=args.derive_roll_scale,
        )
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    state = env.get_state()
    data = {
        **preset,
        "reset_id": 2,
        "canonical_label": "reset_02_directed_midphase",
        "canonical_source": "M2.10 directed_micro_gait rollout mid-state",
        "stage": "M3.2 canonical reset seed",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hand_qpos": state.qpos.astype(float).tolist(),
        "pen_pos": state.pen_center.astype(float).tolist(),
        "pen_quat": state.pen_quat.astype(float).tolist(),
        "pen_qvel": np.zeros(6, dtype=float).tolist(),
        "derived_from": str(src),
        "derive_policy": {
            "policy": "directed_micro_gait",
            "seconds": args.derive_seconds,
            "action_scale": args.derive_action_scale,
            "phase_preset": args.derive_phase_preset,
            "cycle_steps": args.derive_cycle_steps,
            "thumb_scale": args.derive_thumb_scale,
            "center_gain": args.derive_center_gain,
            "phase_direction": args.derive_phase_direction,
            "return_scale": args.derive_return_scale,
            "roll_scale": args.derive_roll_scale,
        },
    }
    env.close()
    write_json(dst, data)
    return data


def derive_stable_offset_reset(src: Path, dst: Path) -> dict[str, Any]:
    """Create a neighboring stable canonical reset around M2.8.

    We first tried directly saving a M2.10 mid-gait dynamic state, but that
    state did not survive a zero-action hold check. For M3.2 we want the
    multi-reset curriculum to be hard enough to avoid overfitting, but still
    stable enough that directed-phase learning is debuggable. This small
    y/z/quaternion offset plays that role.
    """
    data = load_json(src)
    data["pen_pos"][1] -= 0.004
    data["pen_pos"][2] -= 0.001
    quat = np.asarray(data["pen_quat"], dtype=np.float64)
    quat[3] += 0.015
    quat = quat / np.linalg.norm(quat)
    data["pen_quat"] = quat.astype(float).tolist()
    data.update(
        {
            "reset_id": 2,
            "canonical_label": "reset_02_stable_neighbor",
            "canonical_source": "M2.8 thumb light clamp with small y/z/quaternion offset",
            "stage": "M3.2 canonical reset seed",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "derived_from": str(src),
            "offset_from_source": {
                "pen_pos_delta": [0.0, -0.004, -0.001],
                "pen_quat_delta_index_3": 0.015,
            },
        }
    )
    write_json(dst, data)
    return data


def validate_hold(preset_path: Path, args: argparse.Namespace, *, render: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = Dex5PenSpinOracleHoldEnv(
        preset_path=preset_path,
        episode_seconds=args.validate_seconds,
        action_scale=args.validate_action_scale,
        seed=args.seed,
        render_mode="rgb_array" if render else None,
    )
    obs, _ = env.reset(seed=args.seed)
    infos: list[dict[str, Any]] = []
    rewards: list[float] = []
    centers: list[np.ndarray] = []
    phases: list[float] = []
    endpoint_dz: list[float] = []
    frames: list[np.ndarray] = []
    if render:
        frames.append(env.render())
    total_reward = 0.0
    step = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
        total_reward += float(reward)
        infos.append(info)
        rewards.append(float(reward))
        centers.append(np.asarray(info["pen_center"], dtype=np.float64))
        phases.append(float(info["rotation_phase"]))
        endpoint_dz.append(float(info["endpoint_height_diff"]))
        step += 1
        if render and step % args.render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    summary = summarize_rollout(infos, total_reward, step, env.control_hz)
    summary["terminated"] = bool(infos[-1].get("failed", False)) if infos else False
    summary["preset"] = str(preset_path)
    traces = {
        "rewards": np.asarray(rewards, dtype=np.float32),
        "centers": np.asarray(centers, dtype=np.float64),
        "phases": np.asarray(phases, dtype=np.float64),
        "endpoint_height_diffs": np.asarray(endpoint_dz, dtype=np.float64),
    }
    if render:
        traces["frames"] = np.asarray(frames, dtype=np.uint8)
    env.close()
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dex5_penspin/reports/manus_mujoco_presets/canonical_resets")
    parser.add_argument("--report-dir", default="dex5_penspin/reports/m3_2_canonical_resets")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--validate-seconds", type=float, default=6.0)
    parser.add_argument("--validate-action-scale", type=float, default=5e-4)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument(
        "--dynamic-mid-reset",
        action="store_true",
        help="Use an experimental M2.10 mid-gait state for reset_02. Off by default because it is currently unstable.",
    )
    parser.add_argument("--derive-seconds", type=float, default=2.0)
    parser.add_argument("--derive-action-scale", type=float, default=0.0003)
    parser.add_argument("--derive-phase-preset", choices=["m2_3", "quadrature", "paired", "thumb_late"], default="paired")
    parser.add_argument("--derive-cycle-steps", type=int, default=320)
    parser.add_argument("--derive-thumb-scale", type=float, default=0.10)
    parser.add_argument("--derive-center-gain", type=float, default=0.005)
    parser.add_argument("--derive-phase-direction", type=float, default=-1.0)
    parser.add_argument("--derive-return-scale", type=float, default=1.0)
    parser.add_argument("--derive-roll-scale", type=float, default=0.25)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--actuator-kp", type=float, default=400.0)
    parser.add_argument("--actuator-kv", type=float, default=20.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    reset_paths = [
        output_dir / "reset_00_tripod_hold.json",
        output_dir / "reset_01_thumb_light_clamp.json",
        output_dir / "reset_02_stable_neighbor.json",
    ]
    copy_with_metadata(M2_7_PRESET, reset_paths[0], reset_id=0, label="reset_00_tripod_hold", source=str(M2_7_PRESET))
    copy_with_metadata(M2_8_PRESET, reset_paths[1], reset_id=1, label="reset_01_thumb_light_clamp", source=str(M2_8_PRESET))
    if args.dynamic_mid_reset:
        reset_paths[2] = output_dir / "reset_02_directed_midphase.json"
        derive_directed_mid_reset(args, M2_8_PRESET, reset_paths[2])
    else:
        derive_stable_offset_reset(M2_8_PRESET, reset_paths[2])

    rows: list[dict[str, Any]] = []
    for i, path in enumerate(reset_paths):
        row, traces = validate_hold(path, args, render=(i == 0))
        rows.append(row)
        if "frames" in traces:
            imageio.mimsave(report_dir / "reset_00_tripod_hold_zero_action.gif", list(traces["frames"]), fps=25)

    valid = [
        row
        for row in rows
        if row["survival_seconds"] >= args.validate_seconds
        and row["tripod_ratio"] >= 0.80
        and row["xy_drift_max"] <= 0.006
        and not row["terminated"]
    ]
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "M3.2 canonical reset library",
        "reset_paths": [str(path) for path in reset_paths],
        "validation": rows,
        "valid_count": len(valid),
        "acceptance": {
            "survival_seconds": f">= {args.validate_seconds}",
            "tripod_ratio": ">= 0.80",
            "xy_drift_max": "<= 0.006",
            "terminated": False,
        },
    }
    write_json(report_dir / "summary.json", summary)
    # Keep a manifest next to the reset json files so train scripts can consume
    # the accepted library without scanning unrelated presets.
    write_json(
        output_dir / "manifest.json",
        {
            "generated_at": summary["generated_at"],
            "accepted_resets": [row["preset"] for row in valid],
            "all_resets": [str(path) for path in reset_paths],
            "report": str(report_dir / "summary.json"),
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
