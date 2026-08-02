from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from build_m3_2_9_photo_five_phase_debug import (
    PHASE_SPECS,
    build_preview_frames,
    normalize,
    quat_from_x_axis,
    save_montage,
    set_state,
)
from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import contact_snapshot
from tune_m3_2_9_photo_five_phase_contacts import make_env, render_seed


TIP_GEOMS = {
    "thumb": "Link_14R_tip_proxy",
    "index": "Link_24R_tip_proxy",
    "middle": "Link_34R_tip_proxy",
    "ring": "Link_44R_tip_proxy",
}


# Only the phases that were missing a key contact are allowed to change hand qpos.
# The ranges are intentionally local: enough to close the missing fingertip, but
# not enough to erase the Manus-captured gesture identity.
HAND_DELTA_RANGES = {
    1: {
        0: (-0.12, 0.16),  # index support repair
        1: (-0.05, 0.35),
        2: (-0.04, 0.36),
        3: (-0.04, 0.36),
        5: (-0.03, 0.18),  # middle support repair
        6: (-0.02, 0.18),
        7: (-0.02, 0.18),
        9: (-0.03, 0.22),  # ring support repair
        10: (-0.02, 0.20),
        11: (-0.02, 0.20),
        16: (-0.18, 0.00),  # thumb yaw, current pose is already at upper limit
        17: (-0.08, 0.12),
        18: (-0.02, 0.24),
        19: (0.00, 0.55),
    },
    2: {
        16: (-0.24, 0.00),
        17: (-0.10, 0.15),
        18: (-0.02, 0.30),
        19: (0.08, 0.78),
    },
    4: {
        0: (-0.12, 0.16),  # index roll
        1: (0.00, 0.58),
        2: (-0.04, 0.46),
        3: (-0.04, 0.46),
        17: (-0.06, 0.12),
        18: (-0.02, 0.18),
        19: (0.00, 0.35),
    },
}


def enforce_shared_motor_consistency(qpos: np.ndarray) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float64).copy()
    for pip, dip in [(2, 3), (6, 7), (10, 11), (14, 15)]:
        q[dip] = q[pip]
    return q


def pen_tip_distances(env: Dex5PenSpinMinimalEnv) -> dict[str, float]:
    a = env.data.site_xpos[env.pen_end_a_id].copy()
    b = env.data.site_xpos[env.pen_end_b_id].copy()
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-12
    out: dict[str, float] = {}
    for finger, geom_name in TIP_GEOMS.items():
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid < 0:
            continue
        p = env.data.geom_xpos[gid].copy()
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        closest = a + t * ab
        # Use centerline distance as a smooth proxy. The MuJoCo capsule radius is
        # about 6 mm; smaller means the tip is closer to real contact.
        out[finger] = float(np.linalg.norm(p - closest) - 0.006)
    return out


def score_candidate(
    env: Dex5PenSpinMinimalEnv,
    fingers: set[str],
    desired: list[str],
    *,
    phase: int,
    max_penetration: float,
    q_delta: np.ndarray,
    pen_offset: np.ndarray,
    ddeg: float,
) -> float:
    desired_set = set(desired)
    undesired = fingers.difference(desired_set)
    missing = desired_set.difference(fingers)
    distances = pen_tip_distances(env)

    score = 45.0 * len(desired_set.intersection(fingers))
    score -= 18.0 * len(missing)
    score -= 8.0 * len(undesired)
    if desired_set.issubset(fingers):
        score += 70.0

    # Pull missing desired fingertips close even before contact is achieved.
    for finger in missing:
        score -= 520.0 * max(0.0, min(distances.get(finger, 0.05), 0.05))
    for finger in desired_set.intersection(fingers):
        score += 8.0 * max(0.0, 0.012 - max(0.0, distances.get(finger, 0.012)))

    score -= 700.0 * max(0.0, max_penetration - 0.012)
    q_weight = 70.0 if phase == 1 else 210.0
    pen_weight = 35.0 if phase == 1 else 80.0
    phase_weight = 0.03 if phase == 1 else 0.08
    score -= q_weight * float(np.linalg.norm(q_delta))
    score -= pen_weight * float(np.linalg.norm(pen_offset))
    score -= phase_weight * abs(float(ddeg))
    return float(score)


def random_hand_delta(
    phase: int,
    rng: np.random.Generator,
    *,
    mode: str,
) -> np.ndarray:
    q_delta = np.zeros(20, dtype=np.float64)
    ranges = HAND_DELTA_RANGES.get(phase, {})
    if not ranges:
        return q_delta
    for idx, (lo, hi) in ranges.items():
        if mode == "zero":
            value = 0.0
        elif mode == "mid":
            value = 0.5 * (lo + hi)
        else:
            value = rng.uniform(lo, hi)
        q_delta[idx] = value
    return enforce_shared_motor_consistency(q_delta)


def tune_one_phase(
    env: Dex5PenSpinMinimalEnv,
    seed: dict[str, Any],
    spec: dict[str, Any],
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = int(seed["phase"])
    desired = list(spec["desired_contacts"])
    base_q = np.asarray(seed["hand_qpos"], dtype=np.float64)
    base_pos = np.asarray(seed["pen_pos"], dtype=np.float64)
    base_quat = np.asarray(seed["pen_quat"], dtype=np.float64)

    # P3/P5 were already contact-correct; keep them fixed to protect the
    # trajectory shape while we repair P1/P2/P4.
    if phase not in HAND_DELTA_RANGES:
        set_state(env, base_q, base_pos, base_quat)
        fingers, geoms, max_penetration = contact_snapshot(env)
        fixed = {
            **seed,
            "contact_fingers_initial": sorted(fingers),
            "contact_geoms_top": geoms.most_common(20),
            "max_penetration": float(max_penetration),
            "hand_delta": [0.0] * 20,
            "tuning_status": "kept_from_previous_tuning",
            "tuning_score": 0.0,
        }
        return fixed, {
            "phase": phase,
            "name": seed["name"],
            "mode": "fixed",
            "contacts": sorted(fingers),
            "desired": desired,
            "max_penetration": float(max_penetration),
            "score": 0.0,
        }

    best_seed = dict(seed)
    best_debug: dict[str, Any] = {}
    best_score = -1e18

    candidates: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for mode in ["zero", "mid"]:
        q_delta = random_hand_delta(phase, rng, mode=mode)
        for dx in np.linspace(-0.007, 0.007, 3):
            for dy in np.linspace(-0.010, 0.010, 5):
                for dz in np.linspace(-0.006, 0.006, 3):
                    for ddeg in [-10.0, 0.0, 10.0]:
                        candidates.append((q_delta, np.array([dx, dy, dz], dtype=np.float64), ddeg, 0.0))

    for _ in range(samples):
        q_delta = random_hand_delta(phase, rng, mode="random")
        if phase == 1:
            pen_offset = rng.normal(0.0, [0.012, 0.018, 0.010]) + np.array([0.0, -0.010, -0.004])
            ddeg = float(rng.normal(0.0, 26.0))
            dz_axis = float(rng.normal(0.0, 0.075))
        else:
            pen_offset = rng.normal(0.0, [0.006, 0.009, 0.005])
            ddeg = float(rng.normal(0.0, 12.0))
            dz_axis = float(rng.normal(0.0, 0.035))
        candidates.append((q_delta, pen_offset, ddeg, dz_axis))

    base_axis = None
    w, x, y, z = normalize(base_quat)
    base_axis = np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float64,
    )
    base_theta = float(np.arctan2(base_axis[1], base_axis[0]))
    base_axis_z = float(base_axis[2])

    for q_delta, pen_offset, ddeg, dz_axis in candidates:
        q = np.clip(base_q + q_delta, env.joint_low, env.joint_high)
        q = enforce_shared_motor_consistency(q)
        theta = base_theta + np.deg2rad(ddeg)
        axis = np.array([np.cos(theta), np.sin(theta), base_axis_z + dz_axis], dtype=np.float64)
        quat = quat_from_x_axis(axis)
        pos = base_pos + pen_offset
        set_state(env, q, pos, quat)
        fingers, geoms, max_penetration = contact_snapshot(env)
        score = score_candidate(
            env,
            fingers,
            desired,
            phase=phase,
            max_penetration=float(max_penetration),
            q_delta=q - base_q,
            pen_offset=pen_offset,
            ddeg=ddeg,
        )
        if score > best_score:
            best_score = score
            best_seed = {
                **seed,
                "hand_qpos": q.astype(float).tolist(),
                "target_qpos": q.astype(float).tolist(),
                "pen_pos": pos.astype(float).tolist(),
                "pen_quat": quat.astype(float).tolist(),
                "contact_fingers_initial": sorted(fingers),
                "contact_geoms_top": geoms.most_common(20),
                "max_penetration": float(max_penetration),
                "hand_delta": (q - base_q).astype(float).tolist(),
                "pen_offset_from_previous_tuned": pen_offset.astype(float).tolist(),
                "ddeg_from_previous_tuned": float(ddeg),
                "axis_z_delta_from_previous_tuned": float(dz_axis),
                "tuning_status": "hand_pen_local_tuned",
                "tuning_score": float(score),
            }
            best_debug = {
                "phase": phase,
                "name": seed["name"],
                "mode": "hand_pen_local_tuned",
                "contacts": sorted(fingers),
                "desired": desired,
                "missing": sorted(set(desired).difference(fingers)),
                "score": float(score),
                "max_penetration": float(max_penetration),
                "hand_delta_nonzero": {
                    DEX5_JOINT_ORDER[i]: float(q[i] - base_q[i])
                    for i in range(20)
                    if abs(float(q[i] - base_q[i])) > 1e-5
                },
                "pen_offset": pen_offset.astype(float).tolist(),
                "ddeg": float(ddeg),
                "axis_z_delta": float(dz_axis),
                "tip_surface_distance": pen_tip_distances(env),
            }
    return best_seed, best_debug


def render_preview(
    env: Dex5PenSpinMinimalEnv,
    seeds: list[dict[str, Any]],
    output_dir: Path,
    transition_frames: int,
    hold_frames: int,
    fps: int,
) -> None:
    preview = build_preview_frames(seeds, transition_frames=transition_frames, hold_frames=hold_frames)
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for idx, frame_spec in enumerate(preview):
        set_state(env, frame_spec["hand_qpos"], frame_spec["pen_pos"], frame_spec["pen_quat"])
        frame = render_seed(
            env,
            {
                "phase": frame_spec["phase"],
                "name": frame_spec["name"],
                "hand_qpos": np.asarray(frame_spec["hand_qpos"]).tolist(),
                "pen_pos": np.asarray(frame_spec["pen_pos"]).tolist(),
                "pen_quat": np.asarray(frame_spec["pen_quat"]).tolist(),
            },
        )
        frames.append(frame)
        fingers, _, max_penetration = contact_snapshot(env)
        rows.append(
            {
                "frame": idx,
                "phase": frame_spec["phase"],
                "name": frame_spec["name"],
                "contacts": ",".join(sorted(fingers)),
                "max_penetration": float(max_penetration),
            }
        )
    imageio.mimsave(output_dir / "photo_five_phase_hand_pen_tuned_preview.gif", frames, duration=1.0 / fps)
    save_montage(
        frames[:: max(1, len(frames) // 10)],
        output_dir / "photo_five_phase_hand_pen_tuned_preview_montage.png",
        cols=5,
    )
    with (output_dir / "photo_five_phase_hand_pen_tuned_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_tuned_contacts_m3_2_9",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument("--samples", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--transition-frames", type=int, default=40)
    parser.add_argument("--hold-frames", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds: list[dict[str, Any]] = []
    for spec in PHASE_SPECS:
        path = seed_dir / f"P{spec['phase']}_{spec['name']}_tuned_seed.json"
        seeds.append(json.loads(path.read_text(encoding="utf-8")))

    env = make_env(seeds[0]["model_path"])
    env.reset(seed=0)
    rng = np.random.default_rng(args.seed)

    tuned: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    images = []
    for seed, spec in zip(seeds, PHASE_SPECS):
        tuned_seed, row = tune_one_phase(env, seed, spec, rng=rng, samples=args.samples)
        tuned.append(tuned_seed)
        debug.append(row)
        out = output_dir / f"P{tuned_seed['phase']}_{tuned_seed['name']}_hand_pen_tuned_seed.json"
        out.write_text(json.dumps(tuned_seed, indent=2, ensure_ascii=False), encoding="utf-8")
        frame = render_seed(env, tuned_seed)
        imageio.imwrite(output_dir / f"P{tuned_seed['phase']}_{tuned_seed['name']}_hand_pen_tuned_seed.png", frame)
        images.append(frame)

    montage = output_dir / "photo_five_phase_hand_pen_tuned_seed_montage.png"
    save_montage(images, montage, cols=5)
    render_preview(env, tuned, output_dir, args.transition_frames, args.hold_frames, args.fps)

    summary = {
        "stage": "M3.2.10 local hand+pen contact tuning",
        "source_seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "samples_per_repaired_phase": args.samples,
        "montage": str(montage),
        "debug": debug,
        "seeds": [
            {
                "phase": x["phase"],
                "name": x["name"],
                "desired_contacts": x["desired_contacts"],
                "initial_contacts": x["contact_fingers_initial"],
                "score": x.get("tuning_score", 0.0),
                "max_penetration": x["max_penetration"],
                "hand_delta": x.get("hand_delta", [0.0] * 20),
                "pen_pos": x["pen_pos"],
                "pen_quat": x["pen_quat"],
            }
            for x in tuned
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.10 Local Hand + Pen Contact Tuning",
        "",
        "This pass keeps the Manus-captured phase identity, repairs only P1/P2/P4, and leaves P3/P5 fixed.",
        "",
        "| phase | name | desired contacts | tuned contacts | score | max penetration |",
        "|---:|---|---|---|---:|---:|",
    ]
    for x in tuned:
        readme.append(
            f"| {x['phase']} | {x['name']} | {','.join(x['desired_contacts'])} | "
            f"{','.join(x['contact_fingers_initial']) or 'none'} | "
            f"{x.get('tuning_score', 0.0):.2f} | {x['max_penetration']:.4f} |"
        )
    readme.append("")
    readme.append("Next: visually inspect the montage/GIF. If a repaired phase changes the gesture too much, re-capture that single phase.")
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    env.close()

    print(f"wrote local hand+pen tuned seeds: {len(tuned)}")
    print(f"montage: {montage}")
    for row in debug:
        print(row)


if __name__ == "__main__":
    main()
