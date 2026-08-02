from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from m3_2_directed_phase_env import DEFAULT_MANIFEST, Dex5PenSpinDirectedPhaseEnv
from m3_oracle_hold_env import Dex5PenSpinOracleHoldEnv, summarize_rollout
from view_m2_mujoco import action_for_policy


FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
TARGET_COMBOS = {
    "index_middle_ring": {"index", "middle", "ring"},
    "middle_ring_thumb": {"middle", "ring", "thumb"},
    "index_ring_thumb": {"index", "ring", "thumb"},
    "index_middle_thumb": {"index", "middle", "thumb"},
    "regrasp_quad": {"index", "middle", "ring", "thumb"},
}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def contact_set_from_info(info: dict[str, Any]) -> set[str]:
    flags = info.get("contact_flags", {})
    return {finger for finger, hit in flags.items() if bool(hit)}


def exact_pattern_name(fingers: set[str]) -> str:
    if not fingers:
        return "none"
    return "_".join([finger for finger in FINGER_ORDER if finger in fingers])


def semantic_labels(fingers: set[str], previous_fingers: set[str] | None = None) -> list[str]:
    labels: list[str] = []
    if fingers == TARGET_COMBOS["index_middle_ring"]:
        labels.append("index_middle_ring")
    if fingers == TARGET_COMBOS["middle_ring_thumb"]:
        labels.append("middle_ring_thumb")
    if fingers == TARGET_COMBOS["index_ring_thumb"]:
        labels.append("index_ring_thumb")
    if fingers == TARGET_COMBOS["index_middle_thumb"]:
        labels.append("index_middle_thumb")
    if TARGET_COMBOS["regrasp_quad"].issubset(fingers):
        labels.append("regrasp_quad")
    if fingers == {"index", "ring"}:
        labels.append("index_ring_bridge")
    if previous_fingers is not None and fingers == TARGET_COMBOS["index_middle_ring"] and previous_fingers != fingers:
        labels.append("regrasp_to_tripod")
    return labels


def scripted_action(env: Dex5PenSpinDirectedPhaseEnv, step: int, args: argparse.Namespace) -> np.ndarray:
    delta = action_for_policy(
        env.env,
        "directed_micro_gait",
        step,
        args.scripted_action_scale,
        phase_preset=args.scripted_phase_preset,
        center_gain=args.scripted_center_gain,
        target_center=env.initial_center,
        cycle_steps=args.scripted_cycle_steps,
        thumb_scale=args.scripted_thumb_scale,
        phase_direction=args.phase_direction,
        return_scale=args.scripted_return_scale,
        roll_scale=args.scripted_roll_scale,
    )
    return np.clip(delta / max(args.action_scale, 1e-9), -1.0, 1.0).astype(np.float32)


def candidate_from_env(
    env: Dex5PenSpinDirectedPhaseEnv,
    *,
    label: str,
    exact_pattern: str,
    fingers: set[str],
    reset_index: int,
    step: int,
    info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    state = env.env.get_state()
    base = dict(env.preset)
    base.update(
        {
            "stage": "M3.2.5 canonical phase keyframe candidate",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keyframe_label": label,
            "exact_contact_pattern": exact_pattern,
            "support_fingers": sorted(fingers),
            "source_reset_index": int(reset_index),
            "source_reset_path": str(env.active_reset_path),
            "source_step": int(step),
            "source_policy": "M3.2 scripted directed baseline",
            "source_directed_phase": float(info.get("directed_phase", 0.0)),
            "source_phase_delta": float(info.get("phase_delta", 0.0)),
            "source_contact_geoms": info.get("contact_geoms", []),
            "source_metrics": {
                "xy_drift": float(info.get("xy_drift", 0.0)),
                "z_delta": float(info.get("z_delta", 0.0)),
                "endpoint_height_diff": float(info.get("endpoint_height_diff", 0.0)),
                "axis_angle": float(info.get("axis_angle", 0.0)),
                "max_penetration": float(info.get("max_penetration", 0.0)),
            },
            "hand_qpos": state.qpos.astype(float).tolist(),
            "pen_pos": state.pen_center.astype(float).tolist(),
            "pen_quat": state.pen_quat.astype(float).tolist(),
            "pen_qvel": np.zeros(6, dtype=float).tolist(),
        }
    )
    return base


def validate_keyframe(path: Path, args: argparse.Namespace, *, render: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = Dex5PenSpinOracleHoldEnv(
        preset_path=path,
        episode_seconds=args.validate_seconds,
        action_scale=args.validate_action_scale,
        seed=args.seed,
        render_mode="rgb_array" if render else None,
    )
    env.reset(seed=args.seed)
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
        _, reward, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
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
    summary["preset"] = str(path)
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


def keyframe_score(row: dict[str, Any]) -> float:
    return (
        100.0 * float(row["survival_seconds"])
        + 80.0 * float(row["tripod_ratio"])
        + 10.0 * float(row.get("source_directed_phase", 0.0))
        - 150.0 * float(row["xy_drift_max"])
        - 35.0 * float(row["endpoint_height_diff_max"])
        - 4.0 * float(row["phase_abs_max"])
        - (200.0 if row.get("terminated", False) else 0.0)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_validation_row(candidate_path: Path, candidate: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    row = {
        "candidate_path": str(candidate_path),
        "keyframe_label": candidate["keyframe_label"],
        "exact_contact_pattern": candidate["exact_contact_pattern"],
        "support_fingers": "/".join(candidate["support_fingers"]),
        "source_reset_index": candidate["source_reset_index"],
        "source_step": candidate["source_step"],
        "source_directed_phase": candidate["source_directed_phase"],
        "source_xy_drift": candidate["source_metrics"]["xy_drift"],
        "source_endpoint_height_diff": candidate["source_metrics"]["endpoint_height_diff"],
        "survival_seconds": validation["survival_seconds"],
        "tripod_ratio": validation["tripod_ratio"],
        "thumb_ratio": validation["thumb_ratio"],
        "index_ratio": validation["index_ratio"],
        "middle_ratio": validation["middle_ratio"],
        "ring_ratio": validation["ring_ratio"],
        "xy_drift_max": validation["xy_drift_max"],
        "endpoint_height_diff_max": validation["endpoint_height_diff_max"],
        "phase_abs_max": validation["phase_abs_max"],
        "terminated": validation["terminated"],
    }
    row["score"] = keyframe_score(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m3_2_5_phase_keyframes")
    parser.add_argument("--library-dir", default="dex5_penspin/reports/manus_mujoco_presets/canonical_phase_keyframes")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--action-scale", type=float, default=6e-4)
    parser.add_argument("--phase-direction", type=float, default=-1.0)
    parser.add_argument("--target-progress", type=float, default=0.20)
    parser.add_argument("--phase-reward-scale", type=float, default=45.0)
    parser.add_argument("--negative-phase-penalty", type=float, default=55.0)
    parser.add_argument("--hold-reward-scale", type=float, default=1.0)
    parser.add_argument("--scripted-action-scale", type=float, default=0.0003)
    parser.add_argument("--scripted-phase-preset", choices=["m2_3", "quadrature", "paired", "thumb_late"], default="paired")
    parser.add_argument("--scripted-cycle-steps", type=int, default=320)
    parser.add_argument("--scripted-thumb-scale", type=float, default=0.10)
    parser.add_argument("--scripted-center-gain", type=float, default=0.005)
    parser.add_argument("--scripted-return-scale", type=float, default=1.0)
    parser.add_argument("--scripted-roll-scale", type=float, default=0.25)
    parser.add_argument("--validate-seconds", type=float, default=3.0)
    parser.add_argument("--validate-action-scale", type=float, default=5e-4)
    parser.add_argument("--min-hold-seconds", type=float, default=2.0)
    parser.add_argument("--max-xy-drift", type=float, default=0.010)
    parser.add_argument("--max-endpoint-dz", type=float, default=0.020)
    parser.add_argument("--min-step-gap", type=int, default=35)
    parser.add_argument("--max-candidates-per-label", type=int, default=8)
    parser.add_argument("--render-every", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidates"
    library_dir = Path(args.library_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    library_dir.mkdir(parents=True, exist_ok=True)

    env = Dex5PenSpinDirectedPhaseEnv(
        manifest_path=args.manifest_path,
        episode_seconds=args.episode_seconds,
        control_hz=args.control_hz,
        action_scale=args.action_scale,
        seed=args.seed,
        render_mode=None,
        phase_direction=args.phase_direction,
        phase_reward_scale=args.phase_reward_scale,
        negative_phase_penalty=args.negative_phase_penalty,
        hold_reward_scale=args.hold_reward_scale,
        target_progress=args.target_progress,
    )

    candidates: list[tuple[Path, dict[str, Any]]] = []
    pattern_counter: Counter[str] = Counter()
    semantic_counter: Counter[str] = Counter()
    contains_counter: Counter[str] = Counter()
    last_saved_step: dict[tuple[int, str], int] = defaultdict(lambda: -10_000)
    previous_fingers: set[str] | None = None

    for reset_index in range(len(env.presets)):
        env.reset(seed=args.seed + reset_index, options={"reset_index": reset_index})
        previous_fingers = None
        for step in range(env.max_episode_steps):
            action = scripted_action(env, step, args)
            _, _, terminated, truncated, info = env.step(action)
            fingers = contact_set_from_info(info)
            exact = exact_pattern_name(fingers)
            pattern_counter[exact] += 1
            for target, combo in TARGET_COMBOS.items():
                if combo.issubset(fingers):
                    contains_counter[target] += 1
            labels = semantic_labels(fingers, previous_fingers)
            for label in labels:
                semantic_counter[label] += 1
                key = (reset_index, label)
                if (
                    step - last_saved_step[key] >= args.min_step_gap
                    and sum(1 for _, c in candidates if c["keyframe_label"] == label) < args.max_candidates_per_label
                ):
                    candidate = candidate_from_env(
                        env,
                        label=label,
                        exact_pattern=exact,
                        fingers=fingers,
                        reset_index=reset_index,
                        step=step,
                        info=info,
                        args=args,
                    )
                    path = candidate_dir / f"{label}_r{reset_index}_s{step:04d}.json"
                    write_json(path, candidate)
                    candidates.append((path, candidate))
                    last_saved_step[key] = step
            previous_fingers = fingers
            if terminated or truncated:
                break
    env.close()

    validation_rows: list[dict[str, Any]] = []
    accepted_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, (path, candidate) in enumerate(candidates):
        validation, traces = validate_keyframe(path, args, render=(idx == 0))
        row = flatten_validation_row(path, candidate, validation)
        validation_rows.append(row)
        is_valid = (
            row["survival_seconds"] >= args.min_hold_seconds
            and row["xy_drift_max"] <= args.max_xy_drift
            and row["endpoint_height_diff_max"] <= args.max_endpoint_dz
            and not row["terminated"]
        )
        if is_valid:
            accepted_by_label[row["keyframe_label"]].append(row)
        if idx == 0 and "frames" in traces:
            imageio.mimsave(output_dir / "first_candidate_hold_check.gif", list(traces["frames"]), fps=25)

    selected_rows: list[dict[str, Any]] = []
    selected_paths: list[str] = []
    for label, rows in accepted_by_label.items():
        rows = sorted(rows, key=lambda r: r["score"], reverse=True)
        best = rows[0]
        selected_rows.append(best)
        src = Path(best["candidate_path"])
        data = json.loads(src.read_text(encoding="utf-8"))
        data.update(
            {
                "stage": "M3.2.5 accepted canonical phase keyframe",
                "validation": {
                    key: best[key]
                    for key in [
                        "survival_seconds",
                        "tripod_ratio",
                        "thumb_ratio",
                        "index_ratio",
                        "middle_ratio",
                        "ring_ratio",
                        "xy_drift_max",
                        "endpoint_height_diff_max",
                        "phase_abs_max",
                        "terminated",
                        "score",
                    ]
                },
            }
        )
        dst = library_dir / f"{len(selected_rows) - 1:02d}_{label}.json"
        write_json(dst, data)
        selected_paths.append(str(dst))

    target_status = {}
    exact_observed_sets = {
        tuple(pattern.split("_")) if pattern != "none" else tuple()
        for pattern in pattern_counter
    }
    for target, combo in TARGET_COMBOS.items():
        exact_name = exact_pattern_name(combo)
        target_status[target] = {
            "exact_frames": int(pattern_counter.get(exact_name, 0)),
            "contains_frames": int(contains_counter.get(target, 0)),
            "selected": any(row["keyframe_label"] == target for row in selected_rows),
        }

    write_csv(output_dir / "candidate_validation.csv", validation_rows)
    write_csv(output_dir / "selected_keyframes.csv", selected_rows)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "M3.2.5 canonical phase keyframe library",
        "accepted_keyframes": selected_paths,
        "candidate_count": len(candidates),
        "selected_count": len(selected_paths),
        "report": str(output_dir / "summary.json"),
    }
    write_json(library_dir / "manifest.json", manifest)
    summary = {
        "generated_at": manifest["generated_at"],
        "stage": "M3.2.5 canonical phase keyframe mining",
        "source_manifest": str(args.manifest_path),
        "candidate_count": len(candidates),
        "accepted_label_count": len(selected_rows),
        "pattern_counter": pattern_counter.most_common(),
        "semantic_counter": semantic_counter.most_common(),
        "contains_counter": contains_counter.most_common(),
        "target_status": target_status,
        "selected_keyframes": selected_rows,
        "library_manifest": str(library_dir / "manifest.json"),
        "notes": [
            "Exact target combos may be missing if the current scripted gait never releases the corresponding support finger.",
            "Accepted keyframes are zero-action hold validated for at least min_hold_seconds.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
