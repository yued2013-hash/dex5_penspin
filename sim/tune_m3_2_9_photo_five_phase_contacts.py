from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

from build_m3_2_9_photo_five_phase_debug import PHASE_SPECS, normalize, quat_from_x_axis, save_montage, set_state
from dex5_penspin_env import Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import contact_snapshot


def make_env(model_path: str) -> Dex5PenSpinMinimalEnv:
    return Dex5PenSpinMinimalEnv(
        model_path=model_path,
        pen_init_pos=(0.0, 0.0, 0.1),
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=3.0,
        actuator_kp=400.0,
        actuator_kv=20.0,
        thumb_force_limit=0.0,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        render_mode="rgb_array",
        seed=0,
    )


def axis_to_phase_deg(quat: np.ndarray, base_theta: float) -> float:
    # MuJoCo pen x-axis from quaternion wxyz.
    w, x, y, z = normalize(np.asarray(quat, dtype=np.float64))
    axis = np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float64,
    )
    theta = np.arctan2(axis[1], axis[0])
    return float(np.rad2deg(np.arctan2(np.sin(theta - base_theta), np.cos(theta - base_theta))))


def score_contacts(
    fingers: set[str],
    desired: list[str],
    *,
    max_penetration: float,
    phase_error: float,
    center_delta: float,
) -> float:
    desired_set = set(desired)
    desired_hits = len(desired_set.intersection(fingers))
    undesired_hits = len(fingers.difference(desired_set))
    missing = len(desired_set.difference(fingers))
    score = 12.0 * desired_hits - 4.0 * undesired_hits - 7.0 * missing
    if desired_set.issubset(fingers):
        score += 12.0
    # A little penetration is expected for proxy contacts; excessive
    # penetration usually means the pen is jammed through fingers.
    score -= 400.0 * max(0.0, max_penetration - 0.010)
    score -= 0.05 * abs(phase_error)
    score -= 80.0 * center_delta
    return float(score)


def render_seed(env: Dex5PenSpinMinimalEnv, seed: dict[str, Any]) -> np.ndarray:
    set_state(env, np.asarray(seed["hand_qpos"]), np.asarray(seed["pen_pos"]), np.asarray(seed["pen_quat"]))
    frame = env.render()
    if Image is None or ImageDraw is None:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    fingers, _, _ = contact_snapshot(env)
    text = f"P{seed['phase']} {seed['name']} contacts:{','.join(sorted(fingers)) or 'none'}"
    draw.rectangle([0, 0, image.width, 36], fill=(255, 255, 255))
    draw.text((8, 8), text, fill=(0, 0, 0))
    return np.asarray(image)


def tune_seed(
    env: Dex5PenSpinMinimalEnv,
    seed: dict[str, Any],
    spec: dict[str, Any],
    *,
    base_theta: float,
    rng: np.random.Generator,
    samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hand = np.asarray(seed["hand_qpos"], dtype=np.float64)
    base_pos = np.asarray(seed["pen_pos"], dtype=np.float64)
    base_phase = float(spec["phase_deg"])
    desired = list(spec["desired_contacts"])
    best_seed = dict(seed)
    best_debug: dict[str, Any] = {}
    best_score = -1e18

    # Include deterministic local grid plus random samples.
    candidates: list[tuple[np.ndarray, float, float]] = []
    for dx in np.linspace(-0.010, 0.010, 5):
        for dy in np.linspace(-0.012, 0.012, 5):
            for dz in np.linspace(-0.008, 0.010, 5):
                for ddeg in [-18.0, -9.0, 0.0, 9.0, 18.0]:
                    candidates.append((np.array([dx, dy, dz], dtype=np.float64), ddeg, 0.0))
    for _ in range(samples):
        candidates.append(
            (
                rng.normal(0.0, [0.008, 0.010, 0.006]),
                float(rng.normal(0.0, 14.0)),
                float(rng.normal(0.0, 0.04)),
            )
        )

    for offset, ddeg, dz_axis in candidates:
        phase_deg = base_phase + ddeg
        theta = base_theta + np.deg2rad(phase_deg)
        axis = np.array([np.cos(theta), np.sin(theta), float(spec["axis_z"]) + dz_axis], dtype=np.float64)
        quat = quat_from_x_axis(axis)
        pos = base_pos + offset
        set_state(env, hand, pos, quat)
        fingers, geoms, max_penetration = contact_snapshot(env)
        center_delta = float(np.linalg.norm(offset))
        score = score_contacts(
            fingers,
            desired,
            max_penetration=float(max_penetration),
            phase_error=float(ddeg),
            center_delta=center_delta,
        )
        if score > best_score:
            best_score = score
            best_seed = {
                **seed,
                "pen_pos": pos.astype(float).tolist(),
                "pen_quat": quat.astype(float).tolist(),
                "phase_deg_tuned": float(phase_deg),
                "contact_fingers_initial": sorted(fingers),
                "contact_geoms_top": geoms.most_common(20),
                "max_penetration": float(max_penetration),
                "tuning_status": "auto_tuned_contact_seed",
                "tuning_score": float(score),
            }
            best_debug = {
                "score": float(score),
                "offset": offset.astype(float).tolist(),
                "ddeg": float(ddeg),
                "axis_z_delta": float(dz_axis),
                "contacts": sorted(fingers),
                "desired": desired,
                "max_penetration": float(max_penetration),
            }
    return best_seed, best_debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_debug_m3_2_9",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_tuned_contacts_m3_2_9",
    )
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds: list[dict[str, Any]] = []
    for spec in PHASE_SPECS:
        path = seed_dir / f"P{spec['phase']}_{spec['name']}_seed.json"
        seeds.append(json.loads(path.read_text(encoding="utf-8")))

    env = make_env(seeds[0]["model_path"])
    env.reset(seed=0)
    rng = np.random.default_rng(args.seed)
    base_theta = np.deg2rad(20.0)

    tuned: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    for seed, spec in zip(seeds, PHASE_SPECS):
        tuned_seed, debug = tune_seed(
            env,
            seed,
            spec,
            base_theta=base_theta,
            rng=rng,
            samples=args.samples,
        )
        tuned.append(tuned_seed)
        debug_rows.append({"phase": spec["phase"], "name": spec["name"], **debug})
        out = output_dir / f"P{spec['phase']}_{spec['name']}_tuned_seed.json"
        out.write_text(json.dumps(tuned_seed, indent=2, ensure_ascii=False), encoding="utf-8")
        frame = render_seed(env, tuned_seed)
        frames.append(frame)
        imageio.imwrite(output_dir / f"P{spec['phase']}_{spec['name']}_tuned_seed.png", frame)

    montage = output_dir / "photo_five_phase_tuned_seed_montage.png"
    save_montage(frames, montage, cols=5)

    summary = {
        "stage": "M3.2.9 photo five phase contact tuning",
        "source_seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "samples_per_phase": args.samples,
        "montage": str(montage),
        "debug": debug_rows,
        "seeds": [
            {
                "phase": x["phase"],
                "name": x["name"],
                "desired_contacts": x["desired_contacts"],
                "initial_contacts": x["contact_fingers_initial"],
                "score": x["tuning_score"],
                "max_penetration": x["max_penetration"],
                "pen_pos": x["pen_pos"],
                "pen_quat": x["pen_quat"],
            }
            for x in tuned
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = [
        "# M3.2.9 Tuned Photo Five-Phase Contact Seeds",
        "",
        "| phase | name | desired contacts | tuned contacts | score |",
        "|---:|---|---|---|---:|",
    ]
    for x in tuned:
        readme.append(
            f"| {x['phase']} | {x['name']} | {','.join(x['desired_contacts'])} | "
            f"{','.join(x['contact_fingers_initial']) or 'none'} | {x['tuning_score']:.2f} |"
        )
    readme.append("")
    readme.append("These are contact seeds only; run hold validation before promoting them to reset states.")
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    env.close()
    print(f"wrote tuned seeds: {len(tuned)}")
    print(f"montage: {montage}")
    for row in debug_rows:
        print(row)


if __name__ == "__main__":
    main()
