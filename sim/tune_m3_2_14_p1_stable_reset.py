from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from build_m3_2_9_photo_five_phase_debug import PHASE_SPECS, normalize, quat_from_x_axis, set_state
from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import contact_snapshot
from tune_m3_2_10_photo_phase_hand_pen_contacts import enforce_shared_motor_consistency, render_seed


P1_NAME = "four_finger_clamp"
DESIRED = {"thumb", "index", "middle", "ring"}


# Small local corrections around the current M3.2.10 P1 pose. These are kept
# deliberately conservative so the pose remains recognizably the Manus-captured
# four-finger clamp.
HAND_DELTA_RANGES = {
    0: (-0.04, 0.04),
    1: (-0.02, 0.08),
    2: (-0.02, 0.09),
    3: (-0.02, 0.09),
    5: (-0.02, 0.10),
    6: (-0.02, 0.12),
    7: (-0.02, 0.12),
    9: (-0.01, 0.13),
    10: (-0.01, 0.14),
    11: (-0.01, 0.14),
    16: (-0.06, 0.02),
    17: (-0.05, 0.08),
    18: (-0.03, 0.10),
    19: (-0.02, 0.16),
}


def make_env(preset: dict[str, Any], *, duration: float, render: bool = False) -> Dex5PenSpinMinimalEnv:
    return Dex5PenSpinMinimalEnv(
        model_path=preset["model_path"],
        pen_init_pos=preset["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=duration,
        actuator_kp=preset.get("actuator_kp", 400.0),
        actuator_kv=preset.get("actuator_kv", 20.0),
        thumb_force_limit=preset.get("thumb_force_limit", 0.0),
        hard_hold_hand=False,
        visual_only_markers=bool(preset.get("visual_only_markers", True)),
        disable_thumb_mesh_collision=bool(preset.get("disable_thumb_mesh_collision", True)),
        disable_distal_mesh_collision=bool(preset.get("disable_distal_mesh_collision", True)),
        render_mode="rgb_array" if render else None,
        seed=0,
    )


def pen_axis_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(quat)
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float64,
    )


def candidate_quat(base_quat: np.ndarray, ddeg: float, dz_axis: float) -> np.ndarray:
    base_axis = pen_axis_from_quat(base_quat)
    theta = float(np.arctan2(base_axis[1], base_axis[0]) + np.deg2rad(ddeg))
    axis = np.array([np.cos(theta), np.sin(theta), base_axis[2] + dz_axis], dtype=np.float64)
    return quat_from_x_axis(axis)


def sample_hand_delta(rng: np.random.Generator, mode: str) -> np.ndarray:
    delta = np.zeros(20, dtype=np.float64)
    if mode == "zero":
        return delta
    for idx, (lo, hi) in HAND_DELTA_RANGES.items():
        if mode == "mid":
            delta[idx] = 0.5 * (lo + hi)
        else:
            delta[idx] = rng.uniform(lo, hi)
    return enforce_shared_motor_consistency(delta)


def evaluate_candidate(
    env: Dex5PenSpinMinimalEnv,
    base_seed: dict[str, Any],
    *,
    q_delta: np.ndarray,
    pen_offset: np.ndarray,
    ddeg: float,
    dz_axis: float,
    duration: float,
    render: bool,
    render_every: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    base_q = np.asarray(base_seed["hand_qpos"], dtype=np.float64)
    base_pos = np.asarray(base_seed["pen_pos"], dtype=np.float64)
    base_quat = np.asarray(base_seed["pen_quat"], dtype=np.float64)
    q = np.clip(base_q + q_delta, env.joint_low, env.joint_high)
    q = enforce_shared_motor_consistency(q)
    pos = base_pos + pen_offset
    quat = candidate_quat(base_quat, ddeg, dz_axis)

    env.reset(seed=0)
    set_state(env, q, pos, quat)

    center0 = env.get_state().pen_center.copy()
    axis0 = env.get_state().pen_axis.copy()
    initial_fingers, _, initial_penetration = contact_snapshot(env)

    contact_counter: Counter[str] = Counter()
    exact_counter: Counter[str] = Counter()
    desired_hits_sum = 0
    exact_frames = 0
    near_frames = 0
    support_frames = 0
    active_frames = 0
    lost_frames = 0
    index_frames = 0
    middle_frames = 0
    ring_frames = 0
    thumb_frames = 0
    max_penetration = float(initial_penetration)
    min_height = float(center0[2])
    final_center = center0.copy()
    final_axis = axis0.copy()
    terminated = False
    truncated = False
    frames: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    if render:
        frames.append(env.render())

    action = np.zeros(20, dtype=np.float32)
    max_steps = int(round(duration * env.control_hz))
    step = 0
    while step < max_steps:
        _, _, terminated, truncated, info = env.step(action)
        fingers, _, penetration = contact_snapshot(env)
        hits = len(DESIRED.intersection(fingers))
        desired_hits_sum += hits
        exact_frames += int(DESIRED.issubset(fingers))
        near_frames += int(hits >= 3)
        support_frames += int(hits >= 2)
        active_frames += int(len(fingers) >= 2)
        lost_frames += int(len(fingers) == 0)
        index_frames += int("index" in fingers)
        middle_frames += int("middle" in fingers)
        ring_frames += int("ring" in fingers)
        thumb_frames += int("thumb" in fingers)
        for finger in fingers:
            contact_counter[finger] += 1
        exact_counter[",".join(sorted(fingers)) or "none"] += 1
        max_penetration = max(max_penetration, float(penetration))
        final_center = np.asarray(info["pen_center"], dtype=np.float64).copy()
        final_axis = np.asarray(info["pen_axis"], dtype=np.float64).copy()
        centers.append(final_center.copy())
        min_height = min(min_height, float(final_center[2]))
        step += 1
        if render and step % render_every == 0:
            frame = env.render()
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, image.width, 34], fill=(255, 255, 255))
            draw.text(
                (8, 8),
                f"t={step / env.control_hz:.2f}s contacts={','.join(sorted(fingers)) or 'none'}",
                fill=(0, 0, 0),
            )
            frames.append(np.asarray(image))
        if terminated or truncated:
            break

    if render and (not frames or step % render_every != 0):
        frames.append(env.render())

    steps = max(1, step)
    survival_seconds = float(step / env.control_hz)
    drift = float(np.linalg.norm(final_center - center0))
    xy_drift = float(np.linalg.norm(final_center[:2] - center0[:2]))
    z_delta = float(final_center[2] - center0[2])
    axis_angle = float(np.rad2deg(np.arccos(np.clip(np.dot(axis0, final_axis), -1.0, 1.0))))
    exact_ratio = exact_frames / steps
    near_ratio = near_frames / steps
    support_ratio = support_frames / steps
    active_ratio = active_frames / steps
    lost_ratio = lost_frames / steps
    index_ratio = index_frames / steps
    middle_ratio = middle_frames / steps
    ring_ratio = ring_frames / steps
    thumb_ratio = thumb_frames / steps
    desired_hit_mean = desired_hits_sum / (steps * len(DESIRED))
    three_finger_floor = min(index_ratio, middle_ratio, ring_ratio)
    four_finger_floor = min(index_ratio, middle_ratio, ring_ratio, thumb_ratio)
    q_norm = float(np.linalg.norm(q - base_q))
    q_max = float(np.max(np.abs(q - base_q)))
    pen_offset_norm = float(np.linalg.norm(pen_offset))
    survival_ratio = survival_seconds / max(duration, 1e-9)

    score = (
        1700.0 * survival_ratio
        + 620.0 * near_ratio
        + 560.0 * ring_ratio
        + 560.0 * middle_ratio
        + 250.0 * index_ratio
        + 520.0 * thumb_ratio
        + 820.0 * three_finger_floor
        + 1050.0 * four_finger_floor
        + 260.0 * support_ratio
        + 190.0 * active_ratio
        + 850.0 * exact_ratio
        + 220.0 * desired_hit_mean
        - 950.0 * lost_ratio
        - 1300.0 * xy_drift
        - 420.0 * abs(min(0.0, z_delta))
        - 600.0 * max(0.0, 0.055 - min_height)
        - 320.0 * max(0.0, max_penetration - 0.014)
        - 230.0 * q_norm
        - 90.0 * q_max
        - 90.0 * pen_offset_norm
        - 1.5 * abs(float(ddeg))
        - 120.0 * abs(float(dz_axis))
        - (480.0 if terminated else 0.0)
    )

    row = {
        "score": float(score),
        "survival_seconds": survival_seconds,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "initial_contacts": ",".join(sorted(initial_fingers)),
        "exact_ratio": float(exact_ratio),
        "near_ratio": float(near_ratio),
        "support_ratio": float(support_ratio),
        "active_ratio": float(active_ratio),
        "lost_ratio": float(lost_ratio),
        "index_ratio": float(index_ratio),
        "middle_ratio": float(middle_ratio),
        "ring_ratio": float(ring_ratio),
        "thumb_ratio": float(thumb_ratio),
        "three_finger_floor": float(three_finger_floor),
        "four_finger_floor": float(four_finger_floor),
        "desired_hit_mean": float(desired_hit_mean),
        "drift": drift,
        "xy_drift": xy_drift,
        "z_delta": z_delta,
        "min_height": min_height,
        "axis_angle_deg": axis_angle,
        "max_penetration": float(max_penetration),
        "q_delta_norm": q_norm,
        "q_delta_max": q_max,
        "pen_offset_norm": pen_offset_norm,
        "ddeg": float(ddeg),
        "axis_z_delta": float(dz_axis),
        "contact_counter": dict(contact_counter),
        "exact_contact_counter": dict(exact_counter),
        "q_delta": (q - base_q).astype(float).tolist(),
        "pen_offset": pen_offset.astype(float).tolist(),
        "hand_qpos": q.astype(float).tolist(),
        "pen_pos": pos.astype(float).tolist(),
        "pen_quat": quat.astype(float).tolist(),
    }
    traces = None
    if render:
        traces = {
            "frames": np.asarray(frames, dtype=np.uint8),
            "centers": np.asarray(centers, dtype=np.float64) if centers else np.zeros((1, 3), dtype=np.float64),
        }
    return row, traces


def make_candidates(rng: np.random.Generator, samples: int) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    candidates: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    # Deterministic small grid around the current hand pose and likely pen shelf.
    for mode in ["zero", "mid"]:
        q_delta = sample_hand_delta(rng, mode)
        for dx in [-0.006, 0.0, 0.006]:
            for dy in [-0.010, -0.004, 0.002, 0.008]:
                for dz in [-0.003, 0.003, 0.009, 0.014]:
                    for ddeg in [-8.0, -3.0, 3.0, 8.0]:
                        candidates.append((q_delta, np.array([dx, dy, dz], dtype=np.float64), ddeg, 0.0))
    for _ in range(samples):
        q_delta = sample_hand_delta(rng, "random")
        # Bias slightly upward and inward: previous P1 dropped mostly because
        # ring contact disappeared and the pen slid down/out.
        pen_offset = rng.normal(0.0, [0.0055, 0.0075, 0.0055]) + np.array([0.0, -0.003, 0.004])
        pen_offset = np.clip(pen_offset, [-0.014, -0.018, -0.008], [0.014, 0.014, 0.018])
        ddeg = float(np.clip(rng.normal(0.0, 7.0), -16.0, 16.0))
        dz_axis = float(np.clip(rng.normal(0.0, 0.035), -0.08, 0.08))
        candidates.append((q_delta, pen_offset, ddeg, dz_axis))
    return candidates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "score",
        "survival_seconds",
        "terminated",
        "exact_ratio",
        "near_ratio",
        "support_ratio",
        "active_ratio",
        "lost_ratio",
        "index_ratio",
        "middle_ratio",
        "ring_ratio",
        "thumb_ratio",
        "three_finger_floor",
        "four_finger_floor",
        "desired_hit_mean",
        "xy_drift",
        "z_delta",
        "min_height",
        "axis_angle_deg",
        "max_penetration",
        "q_delta_norm",
        "q_delta_max",
        "pen_offset_norm",
        "ddeg",
        "axis_z_delta",
        "initial_contacts",
        "contact_counter",
        "exact_contact_counter",
        "pen_offset",
        "q_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {"rank": rank, **row}
            writer.writerow({field: out.get(field, "") for field in fields})


def plot_trace(path: Path, traces: dict[str, np.ndarray], row: dict[str, Any]) -> None:
    centers = traces["centers"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(centers[:, 0], label="x")
    axes[0].plot(centers[:, 1], label="y")
    axes[0].plot(centers[:, 2], label="z")
    axes[0].set_ylabel("pen center")
    axes[0].legend()
    if len(centers) > 0:
        xy0 = centers[0, :2]
        axes[1].plot(np.linalg.norm(centers[:, :2] - xy0, axis=1))
    axes[1].set_ylabel("xy drift")
    axes[2].bar(
        ["exact", "near", "index", "middle", "ring", "thumb", "lost"],
        [
            row["exact_ratio"],
            row["near_ratio"],
            row["index_ratio"],
            row["middle_ratio"],
            row["ring_ratio"],
            row["thumb_ratio"],
            row["lost_ratio"],
        ],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("ratio")
    axes[2].set_title(f"survival={row['survival_seconds']:.2f}s score={row['score']:.1f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_stable_seed_dir(source_dir: Path, output_dir: Path, best_seed: dict[str, Any]) -> Path:
    stable_dir = output_dir / "stable_seed_dir"
    stable_dir.mkdir(parents=True, exist_ok=True)
    for spec in PHASE_SPECS:
        src = source_dir / f"P{spec['phase']}_{spec['name']}_hand_pen_tuned_seed.json"
        dst = stable_dir / src.name
        if spec["phase"] == 1:
            dst.write_text(json.dumps(best_seed, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            shutil.copy2(src, dst)
    return stable_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/p1_stable_reset_m3_2_14",
    )
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p1_path = seed_dir / f"P1_{P1_NAME}_hand_pen_tuned_seed.json"
    base_seed = json.loads(p1_path.read_text(encoding="utf-8"))

    rng = np.random.default_rng(args.seed)
    candidates = make_candidates(rng, args.samples)
    env = make_env(base_seed, duration=args.duration, render=False)
    rows: list[dict[str, Any]] = []
    for q_delta, pen_offset, ddeg, dz_axis in candidates:
        row, _ = evaluate_candidate(
            env,
            base_seed,
            q_delta=q_delta,
            pen_offset=pen_offset,
            ddeg=ddeg,
            dz_axis=dz_axis,
            duration=args.duration,
            render=False,
            render_every=args.render_every,
        )
        rows.append(row)
    env.close()
    rows.sort(key=lambda x: x["score"], reverse=True)

    best = rows[0]
    best_seed = {
        **base_seed,
        "stage": "M3.2.14 P1 stable reset tuned",
        "hand_qpos": best["hand_qpos"],
        "target_qpos": best["hand_qpos"],
        "pen_pos": best["pen_pos"],
        "pen_quat": best["pen_quat"],
        "contact_fingers_initial": sorted(best["initial_contacts"].split(",")) if best["initial_contacts"] else [],
        "p1_stable_tuning": {
            key: best[key]
            for key in [
                "score",
                "survival_seconds",
                "exact_ratio",
                "near_ratio",
                "support_ratio",
                "active_ratio",
                "lost_ratio",
                "index_ratio",
                "middle_ratio",
                "ring_ratio",
                "thumb_ratio",
                "three_finger_floor",
                "four_finger_floor",
                "xy_drift",
                "z_delta",
                "min_height",
                "max_penetration",
                "q_delta_norm",
                "q_delta_max",
                "pen_offset_norm",
                "ddeg",
                "axis_z_delta",
            ]
        },
        "p1_stable_q_delta_from_m3_2_10": best["q_delta"],
        "p1_stable_pen_offset_from_m3_2_10": best["pen_offset"],
    }

    # Render and re-evaluate the best candidate in a fresh render-enabled env.
    render_env = make_env(best_seed, duration=args.duration, render=True)
    best_render, traces = evaluate_candidate(
        render_env,
        base_seed,
        q_delta=np.asarray(best["q_delta"], dtype=np.float64),
        pen_offset=np.asarray(best["pen_offset"], dtype=np.float64),
        ddeg=float(best["ddeg"]),
        dz_axis=float(best["axis_z_delta"]),
        duration=args.duration,
        render=True,
        render_every=args.render_every,
    )
    render_env.close()
    if traces is not None:
        imageio.mimsave(output_dir / "best_p1_stable_reset_hold.gif", list(traces["frames"]), duration=args.render_every / 100.0)
        plot_trace(output_dir / "best_p1_stable_reset_trace.png", traces, best_render)

    stable_dir = write_stable_seed_dir(seed_dir, output_dir, best_seed)
    best_path = output_dir / "P1_four_finger_clamp_stable_seed.json"
    best_path.write_text(json.dumps(best_seed, indent=2, ensure_ascii=False), encoding="utf-8")
    image_env = make_env(best_seed, duration=args.duration, render=True)
    image_env.reset(seed=0)
    set_state(image_env, np.asarray(best_seed["hand_qpos"]), np.asarray(best_seed["pen_pos"]), np.asarray(best_seed["pen_quat"]))
    imageio.imwrite(output_dir / "best_p1_stable_reset_initial.png", render_seed(image_env, best_seed))
    image_env.close()

    write_csv(output_dir / "p1_stable_reset_sweep.csv", rows)
    summary = {
        "stage": "M3.2.14 P1 stable reset tuning",
        "source_seed": str(p1_path),
        "output_dir": str(output_dir),
        "samples": args.samples,
        "candidate_count": len(rows),
        "duration": args.duration,
        "best_seed": str(best_path),
        "stable_seed_dir": str(stable_dir),
        "best": {
            key: best_render.get(key, best.get(key))
            for key in [
                "score",
                "survival_seconds",
                "terminated",
                "exact_ratio",
                "near_ratio",
                "support_ratio",
                "active_ratio",
                "lost_ratio",
                "index_ratio",
                "middle_ratio",
                "ring_ratio",
                "thumb_ratio",
                "three_finger_floor",
                "four_finger_floor",
                "xy_drift",
                "z_delta",
                "min_height",
                "max_penetration",
                "q_delta_norm",
                "q_delta_max",
                "pen_offset_norm",
                "ddeg",
                "axis_z_delta",
                "initial_contacts",
                "contact_counter",
                "exact_contact_counter",
            ]
        },
        "artifacts": {
            "csv": str(output_dir / "p1_stable_reset_sweep.csv"),
            "gif": str(output_dir / "best_p1_stable_reset_hold.gif"),
            "trace": str(output_dir / "best_p1_stable_reset_trace.png"),
            "initial": str(output_dir / "best_p1_stable_reset_initial.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.14 P1 Stable Reset Tuning",
        "",
        "Goal: improve P1 normal-servo zero-action hold while keeping the Manus hand pose close to the M3.2.10 pose.",
        "",
        "Best result:",
        "",
        "```json",
        json.dumps(summary["best"], indent=2, ensure_ascii=False),
        "```",
        "",
        "Artifacts:",
        "",
        f"- Best seed: `{best_path}`",
        f"- Stable seed dir: `{stable_dir}`",
        f"- Hold GIF: `{output_dir / 'best_p1_stable_reset_hold.gif'}`",
        f"- Trace: `{output_dir / 'best_p1_stable_reset_trace.png'}`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
