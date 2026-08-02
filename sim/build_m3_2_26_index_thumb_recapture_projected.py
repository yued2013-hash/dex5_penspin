from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from evaluate_m3_2_15_short_primitives import contact_metrics
from m3_2_24_short_primitive_env import Dex5PenSpinShortPrimitiveEnv, summarize_short_primitive_rollout
from rebuild_m3_2_16_p4_p5_transitions import tip_surface_distances


def enforce_shared(env: Dex5PenSpinShortPrimitiveEnv, q: np.ndarray) -> np.ndarray:
    q = np.clip(q, env.env.joint_low, env.env.joint_high)
    for pip, dip in [(2, 3), (6, 7), (10, 11), (14, 15)]:
        q[dip] = np.clip(min(q[pip], env.env.joint_high[dip]), env.env.joint_low[dip], env.env.joint_high[dip])
    return q


def project_recapture_state(env: Dex5PenSpinShortPrimitiveEnv, args: argparse.Namespace) -> dict[str, Any]:
    q = env.env.data.qpos[env.env.qpos_adr].copy()
    q[0] += args.index_roll_delta
    q[1] += args.index_pitch_delta
    q[2] += args.index_mid_delta
    q[9] -= args.ring_open_delta
    q[10] -= args.ring_mid_open_delta
    q = enforce_shared(env, q)

    env.env.data.qpos[env.env.qpos_adr] = q
    env.env.target_qpos = q.astype(np.float32)
    env.env.data.ctrl[:] = env.env.target_qpos
    env.env.data.qvel[env.env.qvel_adr] = 0.0
    env.env.last_action[:] = 0.0
    env.env.step_count = 0
    mujoco.mj_forward(env.env.model, env.env.data)

    state = env.env.get_state()
    env.initial_center = state.pen_center.copy()
    env.initial_axis = state.pen_axis.copy()
    env.initial_endpoint_dz = float(abs(state.pen_endpoints[0, 2] - state.pen_endpoints[1, 2]))
    env.initial_phase = float(state.rotation_phase)
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


def rollout_zero_hold(
    env: Dex5PenSpinShortPrimitiveEnv,
    *,
    steps: int,
    render: bool,
    render_every: int,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    infos: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    total = 0.0
    if render:
        frames.append(env.render())
    for step in range(steps):
        _, reward, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
        infos.append(info)
        total += float(reward)
        if render and step % render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    return summarize_short_primitive_rollout(infos, total, len(infos), env.control_hz), frames


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


def row_score(row: dict[str, Any]) -> float:
    return float(
        500.0 * row.get("target_final_window_ratio", 0.0)
        + 140.0 * row.get("index_ratio", 0.0)
        + 110.0 * row.get("thumb_ratio", 0.0)
        + 90.0 * row.get("middle_ratio", 0.0)
        + 80.0 * row.get("survival_seconds", 0.0)
        - 130.0 * row.get("pinky_ratio", 0.0)
        - 16.0 * row.get("max_penetration_mm", 0.0)
    )


def make_seed_record(
    env: Dex5PenSpinShortPrimitiveEnv,
    projected: dict[str, Any],
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    seed = dict(env.start_seed)
    seed.update(
        {
            "stage": "M3.2.26 projected index/thumb recapture seed",
            "phase": float(seed.get("phase", 2.0)),
            "name": "index_thumb_recapture_projected",
            "summary": "Projected short keypose: thumb/index/middle recapture from pre-ring dynamic cache.",
            "desired_contacts": ["thumb", "index", "middle"],
            "placement_support": ["thumb", "index", "middle"],
            "hand_qpos": np.asarray(projected["target_qpos"], dtype=float).tolist(),
            "target_qpos": np.asarray(projected["target_qpos"], dtype=float).tolist(),
            "pen_pos": np.asarray(projected["pen_pos"], dtype=float).tolist(),
            "pen_quat": np.asarray(projected["pen_quat"], dtype=float).tolist(),
            "pen_qvel": np.asarray(projected["pen_qvel"], dtype=float).tolist(),
            "contact_fingers_initial": list(projected["contact_fingers"]),
            "contact_geoms_top": [[str(name), int(count)] for name, count in projected["contact_geoms"]],
            "max_penetration": float(projected["max_penetration"]),
            "validation_status": "projected_recapture_hold_smoke",
            "m3_2_26_role": "index_thumb_recapture_projected",
            "m3_2_26_source_cache_t": float(row["cache_t"]),
            "m3_2_26_source_cache_index": int(row["idx"]),
            "m3_2_26_projection": {
                "index_roll_delta": args.index_roll_delta,
                "index_pitch_delta": args.index_pitch_delta,
                "index_mid_delta": args.index_mid_delta,
                "ring_open_delta": args.ring_open_delta,
                "ring_mid_open_delta": args.ring_mid_open_delta,
            },
            "m3_2_26_hold_summary": {key: row[key] for key in row if isinstance(row[key], (int, float, str, bool))},
        }
    )
    return seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m3_2_26_index_thumb_recapture_projected")
    parser.add_argument("--primitive-label", default="early_pre_ring_index_thumb_recapture")
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--num-cache-states", type=int, default=9)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--action-scale", type=float, default=0.012)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--index-roll-delta", type=float, default=0.0)
    parser.add_argument("--index-pitch-delta", type=float, default=0.40)
    parser.add_argument("--index-mid-delta", type=float, default=0.0)
    parser.add_argument("--ring-open-delta", type=float, default=0.05)
    parser.add_argument("--ring-mid-open-delta", type=float, default=0.03)
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
            primitive_label=args.primitive_label,
            control_hz=args.control_hz,
            action_scale=args.action_scale,
            actuator_kp=args.actuator_kp,
            actuator_kv=args.actuator_kv,
            render_mode="rgb_array" if idx == args.render_index else None,
        )
        try:
            env.reset(seed=args.seed, options={"cache_index": idx})
            projected = project_recapture_state(env, args)
            summary, frames = rollout_zero_hold(
                env,
                steps=args.hold_steps,
                render=idx == args.render_index,
                render_every=args.render_every,
            )
            if frames:
                render_frames = frames
            row = {
                "idx": idx,
                "cache_t": env.active_cache_t,
                "initial_contacts": projected["contacts"],
                "initial_index_dist": projected["distances"]["index"],
                "initial_thumb_dist": projected["distances"]["thumb"],
                "initial_middle_dist": projected["distances"]["middle"],
                "initial_ring_dist": projected["distances"]["ring"],
                "initial_thumb_share": projected["force_share"].get("thumb", 0.0),
                "initial_index_share": projected["force_share"].get("index", 0.0),
                "initial_middle_share": projected["force_share"].get("middle", 0.0),
                "initial_ring_share": projected["force_share"].get("ring", 0.0),
                "initial_pinky_share": projected["force_share"].get("pinky", 0.0),
            }
            row.update(summary)
            row["projected_score"] = row_score(row)
            rows.append(row)
            cache_records.append(
                {
                    "qpos": projected["qpos"],
                    "qvel": projected["qvel"],
                    "target_qpos": projected["target_qpos"],
                    "rotation_phase": projected["rotation_phase"],
                    "cache_t": env.active_cache_t,
                    "contacts": projected["contacts"],
                }
            )
            if best_row is None or row["projected_score"] > best_row["projected_score"]:
                best_row = dict(row)
                best_seed = make_seed_record(env, projected, row, args)
        finally:
            env.close()

    np.savez_compressed(
        output_dir / "index_thumb_recapture_projected_cache.npz",
        qpos=np.asarray([record["qpos"] for record in cache_records], dtype=np.float64),
        qvel=np.asarray([record["qvel"] for record in cache_records], dtype=np.float64),
        target_qpos=np.asarray([record["target_qpos"] for record in cache_records], dtype=np.float64),
        rotation_phase=np.asarray([record["rotation_phase"] for record in cache_records], dtype=np.float64),
        cache_t=np.asarray([record["cache_t"] for record in cache_records], dtype=np.float64),
        contacts=np.asarray([record["contacts"] for record in cache_records]),
    )
    write_csv(output_dir / "projected_recapture_summary.csv", rows)
    if best_seed is not None:
        (output_dir / "index_thumb_recapture_projected_seed.json").write_text(
            json.dumps(best_seed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if render_frames:
        imageio.mimsave(output_dir / "projected_recapture_hold_idx0.gif", render_frames, fps=25)

    aggregate = {
        "output_dir": str(output_dir),
        "num_states": len(rows),
        "avg_survival": float(np.mean([row["survival_seconds"] for row in rows])),
        "avg_target_final": float(np.mean([row["target_final_window_ratio"] for row in rows])),
        "avg_desired": float(np.mean([row["desired_ratio_mean"] for row in rows])),
        "avg_index_ratio": float(np.mean([row["index_ratio"] for row in rows])),
        "avg_pinky_ratio": float(np.mean([row["pinky_ratio"] for row in rows])),
        "initial_contacts": [row["initial_contacts"] for row in rows],
        "best_idx": int(best_row["idx"]) if best_row is not None else None,
        "best_cache_t": float(best_row["cache_t"]) if best_row is not None else None,
        "best_score": float(best_row["projected_score"]) if best_row is not None else None,
        "primitive_label": args.primitive_label,
        "projection": {
            "index_pitch_delta": args.index_pitch_delta,
            "index_mid_delta": args.index_mid_delta,
            "ring_open_delta": args.ring_open_delta,
            "ring_mid_open_delta": args.ring_mid_open_delta,
        },
        "artifacts": {
            "cache": str(output_dir / "index_thumb_recapture_projected_cache.npz"),
            "seed": str(output_dir / "index_thumb_recapture_projected_seed.json"),
            "gif": str(output_dir / "projected_recapture_hold_idx0.gif"),
            "csv": str(output_dir / "projected_recapture_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
