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
    out = np.clip(np.asarray(q, dtype=np.float64), env.env.joint_low, env.env.joint_high)
    for pip, dip in [(2, 3), (6, 7), (10, 11), (14, 15)]:
        out[dip] = np.clip(min(out[pip], env.env.joint_high[dip]), env.env.joint_low[dip], env.env.joint_high[dip])
    return out


def project_b23_entry_state(env: Dex5PenSpinShortPrimitiveEnv, args: argparse.Namespace) -> dict[str, Any]:
    q = env.env.data.qpos[env.env.qpos_adr].copy()

    # Ring-touch entry: close ring gently enough to register contact without making it
    # the main shelf. Middle is still allowed to carry the transition.
    q[8] += args.ring_roll_delta
    q[9] += args.ring_guard_delta
    q[10] += args.ring_mid_guard_delta
    q[17] += args.thumb_roll_delta
    q[18] += args.thumb_pitch_delta
    q[19] += args.thumb_tip_delta
    q[13] -= args.pinky_retreat_delta
    q[14] -= args.pinky_mid_retreat_delta
    q[5] -= args.middle_retreat_delta
    q[6] -= args.middle_mid_retreat_delta
    q = enforce_shared(env, q)

    env.env.data.qpos[env.env.qpos_adr] = q
    env.env.target_qpos = q.astype(np.float32)
    env.env.data.ctrl[:] = env.env.target_qpos
    env.env.data.qvel[env.env.qvel_adr] = 0.0
    env.env.last_action[:] = 0.0
    env.env.step_count = 0
    mujoco.mj_forward(env.env.model, env.env.data)

    for _ in range(args.settle_steps):
        env.step(np.zeros(20, dtype=np.float32))

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
        + 160.0 * row.get("ring_ratio", 0.0)
        + 120.0 * row.get("index_ratio", 0.0)
        + 100.0 * row.get("thumb_ratio", 0.0)
        + 80.0 * row.get("middle_ratio", 0.0)
        + 80.0 * row.get("survival_seconds", 0.0)
        - 170.0 * row.get("pinky_ratio", 0.0)
        - 18.0 * row.get("max_penetration_mm", 0.0)
        - 40.0 * max(0.0, row.get("initial_ring_share", 0.0) - 0.22)
    )


def make_seed_record(
    env: Dex5PenSpinShortPrimitiveEnv,
    projected: dict[str, Any],
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    seed = dict(env.target_seed)
    seed.update(
        {
            "stage": "M3.2.28 B23-entry ring-touch projected seed",
            "phase": float(seed.get("phase", 2.5)),
            "name": "B23_entry_ring_touch_projected",
            "summary": "Intermediate entry: thumb/index stay engaged while ring lightly touches; middle may remain support.",
            "desired_contacts": ["thumb", "index", "ring"],
            "placement_support": ["thumb", "index", "middle", "ring"],
            "hand_qpos": np.asarray(projected["target_qpos"], dtype=float).tolist(),
            "target_qpos": np.asarray(projected["target_qpos"], dtype=float).tolist(),
            "pen_pos": np.asarray(projected["pen_pos"], dtype=float).tolist(),
            "pen_quat": np.asarray(projected["pen_quat"], dtype=float).tolist(),
            "pen_qvel": np.asarray(projected["pen_qvel"], dtype=float).tolist(),
            "contact_fingers_initial": list(projected["contact_fingers"]),
            "contact_geoms_top": [[str(name), int(count)] for name, count in projected["contact_geoms"]],
            "max_penetration": float(projected["max_penetration"]),
            "validation_status": "m3_2_28_projected_entry_hold_smoke",
            "m3_2_28_role": "B23_entry_ring_touch",
            "m3_2_28_source_cache_t": float(row["cache_t"]),
            "m3_2_28_source_cache_index": int(row["idx"]),
            "m3_2_28_projection": {
                "ring_roll_delta": args.ring_roll_delta,
                "ring_guard_delta": args.ring_guard_delta,
                "ring_mid_guard_delta": args.ring_mid_guard_delta,
                "thumb_roll_delta": args.thumb_roll_delta,
                "thumb_pitch_delta": args.thumb_pitch_delta,
                "thumb_tip_delta": args.thumb_tip_delta,
                "pinky_retreat_delta": args.pinky_retreat_delta,
                "pinky_mid_retreat_delta": args.pinky_mid_retreat_delta,
                "middle_retreat_delta": args.middle_retreat_delta,
                "middle_mid_retreat_delta": args.middle_mid_retreat_delta,
                "settle_steps": args.settle_steps,
            },
            "m3_2_28_hold_summary": {key: row[key] for key in row if isinstance(row[key], (int, float, str, bool))},
        }
    )
    return seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m3_2_28_recapture_split_b23_entry/projected_to_b23_entry_ring_touch")
    parser.add_argument("--source-primitive-label", default="projected_index_thumb_recapture_to_B23")
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--num-cache-states", type=int, default=7)
    parser.add_argument("--hold-steps", type=int, default=24)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--action-scale", type=float, default=0.012)
    parser.add_argument("--actuator-kp", type=float, default=900.0)
    parser.add_argument("--actuator-kv", type=float, default=45.0)
    parser.add_argument("--ring-roll-delta", type=float, default=0.0)
    parser.add_argument("--ring-guard-delta", type=float, default=0.075)
    parser.add_argument("--ring-mid-guard-delta", type=float, default=0.040)
    parser.add_argument("--thumb-roll-delta", type=float, default=0.000)
    parser.add_argument("--thumb-pitch-delta", type=float, default=0.020)
    parser.add_argument("--thumb-tip-delta", type=float, default=0.015)
    parser.add_argument("--pinky-retreat-delta", type=float, default=0.035)
    parser.add_argument("--pinky-mid-retreat-delta", type=float, default=0.030)
    parser.add_argument("--middle-retreat-delta", type=float, default=0.000)
    parser.add_argument("--middle-mid-retreat-delta", type=float, default=0.000)
    parser.add_argument("--render-index", type=int, default=0)
    parser.add_argument("--render-every", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cache_records: list[dict[str, Any]] = []
    render_frames: list[np.ndarray] = []
    best_projected: dict[str, Any] | None = None
    best_seed_env: Dex5PenSpinShortPrimitiveEnv | None = None
    best_row: dict[str, Any] | None = None

    for idx in range(args.num_cache_states):
        env = Dex5PenSpinShortPrimitiveEnv(
            primitive_label=args.source_primitive_label,
            control_hz=args.control_hz,
            action_scale=args.action_scale,
            actuator_kp=args.actuator_kp,
            actuator_kv=args.actuator_kv,
            render_mode="rgb_array" if idx == args.render_index else None,
        )
        try:
            env.reset(seed=args.seed, options={"cache_index": idx})
            projected = project_b23_entry_state(env, args)
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
                "initial_max_penetration_mm": projected["max_penetration"] * 1000.0,
            }
            row.update(summary)
            row["entry_score"] = row_score(row)
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
            if best_row is None or row["entry_score"] > best_row["entry_score"]:
                best_row = dict(row)
                best_projected = dict(projected)
                if best_seed_env is not None:
                    best_seed_env.close()
                best_seed_env = env
                env = None  # keep it alive until the representative seed is written
        finally:
            if env is not None:
                env.close()

    if not rows or best_projected is None or best_row is None or best_seed_env is None:
        raise RuntimeError("No B23-entry ring-touch states were generated")

    np.savez_compressed(
        output_dir / "b23_entry_ring_touch_cache.npz",
        qpos=np.asarray([record["qpos"] for record in cache_records], dtype=np.float64),
        qvel=np.asarray([record["qvel"] for record in cache_records], dtype=np.float64),
        target_qpos=np.asarray([record["target_qpos"] for record in cache_records], dtype=np.float64),
        rotation_phase=np.asarray([record["rotation_phase"] for record in cache_records], dtype=np.float64),
        cache_t=np.asarray([record["cache_t"] for record in cache_records], dtype=np.float64),
        contacts=np.asarray([record["contacts"] for record in cache_records]),
    )
    seed_record = make_seed_record(best_seed_env, best_projected, best_row, args)
    (output_dir / "B23_entry_ring_touch_seed.json").write_text(
        json.dumps(seed_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    best_seed_env.close()

    write_csv(output_dir / "b23_entry_ring_touch_summary.csv", rows)
    if render_frames:
        imageio.mimsave(output_dir / "b23_entry_ring_touch_hold_idx0.gif", render_frames, fps=25)

    aggregate = {
        "output_dir": str(output_dir),
        "num_states": len(rows),
        "source_primitive_label": args.source_primitive_label,
        "avg_survival": float(np.mean([row["survival_seconds"] for row in rows])),
        "avg_target_final": float(np.mean([row["target_final_window_ratio"] for row in rows])),
        "avg_desired": float(np.mean([row["desired_ratio_mean"] for row in rows])),
        "avg_thumb_ratio": float(np.mean([row["thumb_ratio"] for row in rows])),
        "avg_index_ratio": float(np.mean([row["index_ratio"] for row in rows])),
        "avg_middle_ratio": float(np.mean([row["middle_ratio"] for row in rows])),
        "avg_ring_ratio": float(np.mean([row["ring_ratio"] for row in rows])),
        "avg_pinky_ratio": float(np.mean([row["pinky_ratio"] for row in rows])),
        "avg_max_penetration_mm": float(np.mean([row["max_penetration_mm"] for row in rows])),
        "initial_contacts": [row["initial_contacts"] for row in rows],
        "best_idx": int(best_row["idx"]),
        "best_cache_t": float(best_row["cache_t"]),
        "best_score": float(best_row["entry_score"]),
        "best_initial_contacts": str(best_row["initial_contacts"]),
        "projection": {
            "ring_guard_delta": args.ring_guard_delta,
            "ring_mid_guard_delta": args.ring_mid_guard_delta,
            "pinky_retreat_delta": args.pinky_retreat_delta,
            "pinky_mid_retreat_delta": args.pinky_mid_retreat_delta,
        },
        "artifacts": {
            "cache": str(output_dir / "b23_entry_ring_touch_cache.npz"),
            "seed": str(output_dir / "B23_entry_ring_touch_seed.json"),
            "gif": str(output_dir / "b23_entry_ring_touch_hold_idx0.gif"),
            "csv": str(output_dir / "b23_entry_ring_touch_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
