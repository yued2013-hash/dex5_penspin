from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from evaluate_m3_2_15_short_primitives import contact_metrics
from m3_2_24_short_primitive_env import (
    Dex5PenSpinShortPrimitiveEnv,
    INDEX_NEAR_SURFACE_THRESHOLD_M,
    summarize_short_primitive_rollout,
)
from rebuild_m3_2_16_p4_p5_transitions import tip_surface_distances


def smoothstep(x: float) -> float:
    t = float(np.clip(x, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def enforce_shared(env: Dex5PenSpinShortPrimitiveEnv, q: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(q, dtype=np.float64), env.env.joint_low, env.env.joint_high)
    for pip, dip in [(2, 3), (6, 7), (10, 11), (14, 15)]:
        out[dip] = np.clip(min(out[pip], env.env.joint_high[dip]), env.env.joint_low[dip], env.env.joint_high[dip])
    return out


def scripted_approach_action(
    env: Dex5PenSpinShortPrimitiveEnv,
    step: int,
    args: argparse.Namespace,
) -> np.ndarray:
    local_tau = smoothstep((step + 1) / max(1.0, args.ramp_fraction * env.max_episode_steps))
    target = env.scripted_start_qpos.copy()
    target[0] += args.index_roll_bias * local_tau
    target[1] += args.index_pip_bias * local_tau
    target[2] += args.index_mid_bias * local_tau
    target[9] -= args.ring_open_bias * local_tau
    target[10] -= args.ring_mid_open_bias * local_tau
    target = enforce_shared(env, target)
    delta = args.tracking_gain * (target - env.env.target_qpos.astype(np.float64))
    return np.clip(delta / max(env.action_scale, 1e-9), -1.0, 1.0).astype(np.float32)


def capture_state(env: Dex5PenSpinShortPrimitiveEnv) -> dict[str, Any]:
    state = env.env.get_state()
    metrics = contact_metrics(env.env)
    distances = tip_surface_distances(env.env)
    return {
        "qpos": env.env.data.qpos.copy(),
        "qvel": env.env.data.qvel.copy(),
        "target_qpos": env.env.target_qpos.copy(),
        "rotation_phase": float(state.rotation_phase),
        "pen_pos": env.env.data.qpos[env.env.pen_qpos_adr : env.env.pen_qpos_adr + 3].copy(),
        "pen_quat": env.env.data.qpos[env.env.pen_qpos_adr + 3 : env.env.pen_qpos_adr + 7].copy(),
        "pen_qvel": env.env.data.qvel[env.env.pen_qvel_adr : env.env.pen_qvel_adr + 6].copy(),
        "contacts": ",".join(sorted(metrics["fingers"])) or "none",
        "contact_fingers": sorted(metrics["fingers"]),
        "contact_geoms": metrics["geoms"].most_common(10),
        "force_share": metrics["force_share"],
        "max_penetration": float(metrics["max_penetration"]),
        "distances": distances,
    }


def rollout_approach(
    env: Dex5PenSpinShortPrimitiveEnv,
    args: argparse.Namespace,
    *,
    render: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[np.ndarray]]:
    infos: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    total = 0.0
    if render:
        frames.append(env.render())
    for step in range(args.approach_steps):
        action = scripted_approach_action(env, step, args)
        _, reward, terminated, truncated, info = env.step(action)
        infos.append(info)
        total += float(reward)
        if render and step % args.render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    state = capture_state(env)
    summary = summarize_short_primitive_rollout(infos, total, len(infos), env.control_hz)
    return summary, state, frames


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row_score(row: dict[str, Any]) -> float:
    final_dist = float(row.get("index_tip_dist_final", 0.04))
    near = 1.0 - min(1.0, max(0.0, final_dist) / INDEX_NEAR_SURFACE_THRESHOLD_M)
    under_14 = float(final_dist <= INDEX_NEAR_SURFACE_THRESHOLD_M)
    overpenetration = min(1.0, max(0.0, -float(row.get("index_tip_dist_min", 0.0)) - 0.003) / 0.006)
    return float(
        720.0 * under_14
        + 260.0 * near
        + 120.0 * row.get("survival_seconds", 0.0)
        + 80.0 * row.get("thumb_ratio", 0.0)
        + 50.0 * row.get("middle_ratio", 0.0)
        - 85.0 * row.get("index_ratio", 0.0)
        - 65.0 * row.get("pinky_ratio", 0.0)
        - 35.0 * overpenetration
        - 10.0 * row.get("max_penetration_mm", 0.0)
        - 900.0 * row.get("target_center_dist_final", 0.0)
    )


def make_seed_record(
    base_seed: dict[str, Any],
    state: dict[str, Any],
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    seed = dict(base_seed)
    seed.update(
        {
            "stage": "M3.2.30 index-near-surface approach seed",
            "phase": float(seed.get("phase", 2.0)),
            "name": "index_near_surface",
            "summary": "Approach-only keypose: index fingertip is within 14mm of the pen surface; contact is deferred.",
            "desired_contacts": [],
            "placement_support": list(state["contact_fingers"]),
            "hand_qpos": np.asarray(state["target_qpos"], dtype=float).tolist(),
            "target_qpos": np.asarray(state["target_qpos"], dtype=float).tolist(),
            "pen_pos": np.asarray(state["pen_pos"], dtype=float).tolist(),
            "pen_quat": np.asarray(state["pen_quat"], dtype=float).tolist(),
            "pen_qvel": np.asarray(state["pen_qvel"], dtype=float).tolist(),
            "contact_fingers_initial": list(state["contact_fingers"]),
            "contact_geoms_top": [[str(name), int(count)] for name, count in state["contact_geoms"]],
            "max_penetration": float(state["max_penetration"]),
            "validation_status": "m3_2_30_index_near_surface_approach_smoke",
            "m3_2_30_role": "index_near_surface",
            "m3_2_30_source_primitive_label": args.source_primitive_label,
            "m3_2_30_source_cache_t": float(row["cache_t"]),
            "m3_2_30_source_cache_index": int(row["idx"]),
            "m3_2_30_index_tip_dist_final": float(row["index_tip_dist_final"]),
            "m3_2_30_projection": {
                "approach_steps": args.approach_steps,
                "index_roll_bias": args.index_roll_bias,
                "index_pip_bias": args.index_pip_bias,
                "index_mid_bias": args.index_mid_bias,
                "ring_open_bias": args.ring_open_bias,
                "ring_mid_open_bias": args.ring_mid_open_bias,
                "tracking_gain": args.tracking_gain,
                "ramp_fraction": args.ramp_fraction,
            },
            "m3_2_30_hold_summary": {key: row[key] for key in row if isinstance(row[key], (int, float, str, bool))},
        }
    )
    return seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m3_2_30_index_near_surface/near_016_019")
    parser.add_argument("--source-primitive-label", default="earlier_pre_ring_016_019_index_thumb_recapture")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--num-cache-states", type=int, default=4)
    parser.add_argument("--approach-steps", type=int, default=17)
    parser.add_argument("--episode-seconds", type=float, default=0.17)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--action-scale", type=float, default=0.026)
    parser.add_argument("--actuator-kp", type=float, default=900.0)
    parser.add_argument("--actuator-kv", type=float, default=45.0)
    parser.add_argument("--index-roll-bias", type=float, default=0.0)
    parser.add_argument("--index-pip-bias", type=float, default=0.40)
    parser.add_argument("--index-mid-bias", type=float, default=0.0)
    parser.add_argument("--ring-open-bias", type=float, default=0.05)
    parser.add_argument("--ring-mid-open-bias", type=float, default=0.03)
    parser.add_argument("--tracking-gain", type=float, default=0.68)
    parser.add_argument("--ramp-fraction", type=float, default=0.45)
    parser.add_argument("--render-index", type=int, default=0)
    parser.add_argument("--render-every", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cache_records: list[dict[str, Any]] = []
    render_frames: list[np.ndarray] = []
    best_seed: dict[str, Any] | None = None
    best_row: dict[str, Any] | None = None

    for idx in range(args.num_cache_states):
        env = Dex5PenSpinShortPrimitiveEnv(
            primitive_label=args.source_primitive_label,
            control_hz=args.control_hz,
            action_scale=args.action_scale,
            actuator_kp=args.actuator_kp,
            actuator_kv=args.actuator_kv,
            episode_seconds=args.episode_seconds,
            render_mode="rgb_array" if idx == args.render_index else None,
        )
        try:
            env.spec.support_contacts = ["thumb", "index", "middle", "ring", "pinky"]
            env.spec.allowed_extra_contacts = ["thumb", "index", "middle", "ring", "pinky"]
            env.spec.forbidden_contacts = []
            env.reset(seed=args.seed, options={"cache_index": idx})
            base_seed = dict(env.start_seed)
            summary, state, frames = rollout_approach(env, args, render=idx == args.render_index)
            if frames:
                render_frames = frames
            row = {
                "idx": idx,
                "cache_t": env.active_cache_t,
                "initial_contacts": state["contacts"],
                "initial_index_dist": state["distances"]["index"],
                "initial_thumb_dist": state["distances"]["thumb"],
                "initial_middle_dist": state["distances"]["middle"],
                "initial_ring_dist": state["distances"]["ring"],
                "initial_pinky_dist": state["distances"]["pinky"],
                "initial_thumb_share": state["force_share"].get("thumb", 0.0),
                "initial_index_share": state["force_share"].get("index", 0.0),
                "initial_middle_share": state["force_share"].get("middle", 0.0),
                "initial_ring_share": state["force_share"].get("ring", 0.0),
                "initial_pinky_share": state["force_share"].get("pinky", 0.0),
                "initial_max_penetration_mm": state["max_penetration"] * 1000.0,
            }
            row.update(summary)
            row["index_tip_final_under_14mm"] = bool(row["index_tip_dist_final"] <= INDEX_NEAR_SURFACE_THRESHOLD_M)
            row["near_surface_score"] = row_score(row)
            rows.append(row)
            cache_records.append(
                {
                    "qpos": state["qpos"],
                    "qvel": state["qvel"],
                    "target_qpos": state["target_qpos"],
                    "rotation_phase": state["rotation_phase"],
                    "cache_t": env.active_cache_t,
                    "contacts": state["contacts"],
                }
            )
            if best_row is None or row["near_surface_score"] > best_row["near_surface_score"]:
                best_row = dict(row)
                best_seed = make_seed_record(base_seed, state, row, args)
        finally:
            env.close()

    if not rows or best_seed is None or best_row is None:
        raise RuntimeError("No index-near-surface states were generated")

    np.savez_compressed(
        output_dir / "index_near_surface_cache.npz",
        qpos=np.asarray([record["qpos"] for record in cache_records], dtype=np.float64),
        qvel=np.asarray([record["qvel"] for record in cache_records], dtype=np.float64),
        target_qpos=np.asarray([record["target_qpos"] for record in cache_records], dtype=np.float64),
        rotation_phase=np.asarray([record["rotation_phase"] for record in cache_records], dtype=np.float64),
        cache_t=np.asarray([record["cache_t"] for record in cache_records], dtype=np.float64),
        contacts=np.asarray([record["contacts"] for record in cache_records]),
    )
    (output_dir / "index_near_surface_seed.json").write_text(
        json.dumps(best_seed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "index_near_surface_summary.csv", rows)
    if render_frames:
        imageio.mimsave(output_dir / "index_near_surface_hold_idx0.gif", render_frames, fps=25)

    aggregate = {
        "output_dir": str(output_dir),
        "num_states": len(rows),
        "source_primitive_label": args.source_primitive_label,
        "avg_survival": float(np.mean([row["survival_seconds"] for row in rows])),
        "avg_index_tip_dist_final": float(np.mean([row["index_tip_dist_final"] for row in rows])),
        "avg_index_tip_dist_final_mm": float(np.mean([row["index_tip_dist_final"] for row in rows]) * 1000.0),
        "under_14mm_count": int(sum(bool(row["index_tip_final_under_14mm"]) for row in rows)),
        "avg_index_ratio": float(np.mean([row["index_ratio"] for row in rows])),
        "avg_pinky_ratio": float(np.mean([row["pinky_ratio"] for row in rows])),
        "avg_max_penetration_mm": float(np.mean([row["max_penetration_mm"] for row in rows])),
        "initial_contacts": [row["initial_contacts"] for row in rows],
        "best_idx": int(best_row["idx"]),
        "best_cache_t": float(best_row["cache_t"]),
        "best_score": float(best_row["near_surface_score"]),
        "best_index_tip_dist_final_mm": float(best_row["index_tip_dist_final"] * 1000.0),
        "approach": {
            "approach_steps": args.approach_steps,
            "index_roll_bias": args.index_roll_bias,
            "index_pip_bias": args.index_pip_bias,
            "index_mid_bias": args.index_mid_bias,
            "ring_open_bias": args.ring_open_bias,
            "tracking_gain": args.tracking_gain,
            "ramp_fraction": args.ramp_fraction,
        },
        "artifacts": {
            "cache": str(output_dir / "index_near_surface_cache.npz"),
            "seed": str(output_dir / "index_near_surface_seed.json"),
            "gif": str(output_dir / "index_near_surface_hold_idx0.gif"),
            "csv": str(output_dir / "index_near_surface_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
