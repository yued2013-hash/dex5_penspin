from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from build_m3_2_9_photo_five_phase_debug import normalize, set_state, slerp
from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from rebuild_m3_2_16_p4_p5_transitions import configure_tip_proxies
from teleop_m3_2_6_canonical_keyframes import contact_snapshot


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/manus_mujoco_presets"
MANUS_P3 = REPORTS / "photo_five_phase_hand_pen_tuned_m3_2_10/P3_index_wrap_to_front_hand_pen_tuned_seed.json"
DYNAMIC_P3 = REPORTS / "corrected_p3_p4_chain_m3_2_38/corrected_true_P3_entry.npz"
DEFAULT_OUTPUT = REPORTS / "canonical_p3_manus_constrained_m3_2_40"
TRIAD = frozenset(("thumb", "index", "middle"))


def make_env(reference: dict[str, Any], args: argparse.Namespace, *, render: bool) -> Dex5PenSpinMinimalEnv:
    model_path = ROOT.parent / reference["model_path"]
    env = Dex5PenSpinMinimalEnv(
        model_path=model_path,
        pen_init_pos=reference["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=0.25,
        action_delta_limit=0.04,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=0.0,
        hard_hold_hand=False,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        render_mode="rgb_array" if render else None,
        seed=args.seed,
    )
    configure_tip_proxies(env, args)
    return env


def enforce_shared_motor_consistency(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    for pip, dip in ((2, 3), (6, 7), (10, 11), (14, 15)):
        qpos[dip] = qpos[pip]
    return qpos


def perturb_quat(quat: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return normalize(quat)
    axis = rotvec / angle
    delta = np.asarray((np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)), dtype=np.float64)
    w1, x1, y1, z1 = delta
    w2, x2, y2, z2 = quat
    return normalize(
        np.asarray(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        )
    )


def evaluate(
    env: Dex5PenSpinMinimalEnv,
    manus_qpos: np.ndarray,
    hand_qpos: np.ndarray,
    pen_pos: np.ndarray,
    pen_quat: np.ndarray,
    control_bias: np.ndarray,
    steps: int,
) -> dict[str, Any]:
    env.reset(seed=0)
    set_state(env, hand_qpos, pen_pos, pen_quat)
    control_target = enforce_shared_motor_consistency(
        np.clip(hand_qpos + control_bias, env.joint_low, env.joint_high)
    )
    env.target_qpos = control_target.astype(np.float32)
    env.data.ctrl[:] = control_target
    mujoco.mj_forward(env.model, env.data)
    center0 = env.get_state().pen_center.copy()
    contact_rows: list[list[str]] = []
    penetration_rows: list[float] = []
    triad_run = 0
    longest_triad_run = 0
    terminated = False
    max_hand_rms = 0.0
    for step in range(steps + 1):
        current_hand = env.data.qpos[env.qpos_adr]
        max_hand_rms = max(
            max_hand_rms,
            float(np.sqrt(np.mean((current_hand - manus_qpos) ** 2))),
        )
        contacts, _, penetration = contact_snapshot(env)
        contact_rows.append(sorted(contacts))
        penetration_rows.append(float(penetration))
        if TRIAD.issubset(contacts):
            triad_run += 1
            longest_triad_run = max(longest_triad_run, triad_run)
        else:
            triad_run = 0
        if step < steps:
            _, _, terminated, _, _ = env.step(np.zeros(20, dtype=np.float32))
            if terminated:
                break
    state = env.get_state()
    contact_counts = {
        finger: sum(finger in contacts for contacts in contact_rows)
        for finger in ("thumb", "index", "middle", "ring", "pinky")
    }
    samples = len(contact_rows)
    rms = float(np.sqrt(np.mean((hand_qpos - manus_qpos) ** 2)))
    drift = float(np.linalg.norm(state.pen_center - center0))
    max_penetration = max(penetration_rows) if penetration_rows else 0.0
    hold_steps = min(steps, longest_triad_run)
    score = (
        1000.0 * hold_steps
        + 50.0 * sum(contact_counts[finger] for finger in TRIAD)
        - 20.0 * contact_counts["ring"]
        - 3_000_000.0 * max(0.0, max_penetration - 0.004)
        - 30000.0 * max(0.0, drift - 0.015)
        - 2000.0 * max(0.0, max_hand_rms - 0.10)
        - 250.0 * max_hand_rms
        - (5000.0 if terminated else 0.0)
    )
    return {
        "score": float(score),
        "manus_rms_rad": rms,
        "max_hand_rms_rad": max_hand_rms,
        "control_bias": control_bias.tolist(),
        "samples": samples,
        "triad_longest_run_steps": int(longest_triad_run),
        "triad_hold_ms": float(1000.0 * min(steps, longest_triad_run) / env.control_hz),
        "triad_all_samples": bool(longest_triad_run >= steps),
        "contact_counts": contact_counts,
        "contact_rows": contact_rows,
        "max_penetration_mm": float(max_penetration * 1000.0),
        "pen_drift_mm": float(drift * 1000.0),
        "terminated": bool(terminated),
        "final_pen_pos": state.pen_center.tolist(),
        "final_pen_quat": state.pen_quat.tolist(),
    }


def accepted(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        row["max_hand_rms_rad"] < args.max_rms
        and row["triad_hold_ms"] >= args.hold_ms
        and row["max_penetration_mm"] < 4.0
        and row["pen_drift_mm"] < 15.0
        and not row["terminated"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a Manus-constrained thumb/index/middle P3 hold.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=240)
    parser.add_argument("--hold-ms", type=float, default=100.0)
    parser.add_argument("--max-rms", type=float, default=0.12)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    args = parser.parse_args()

    reference = json.loads(MANUS_P3.read_text(encoding="utf-8"))
    with np.load(DYNAMIC_P3) as data:
        dynamic_hand = np.asarray(data["hand_qpos"][0], dtype=np.float64)
        dynamic_pen_pos = np.asarray(data["pen_pos"][0], dtype=np.float64)
        dynamic_pen_quat = np.asarray(data["pen_quat"][0], dtype=np.float64)
    manus_hand = np.asarray(reference["hand_qpos"], dtype=np.float64)
    manus_pen_pos = np.asarray(reference["pen_pos"], dtype=np.float64)
    manus_pen_quat = np.asarray(reference["pen_quat"], dtype=np.float64)
    dynamic_rms = float(np.sqrt(np.mean((dynamic_hand - manus_hand) ** 2)))
    max_alpha = min(1.0, args.max_rms * 0.995 / dynamic_rms)
    steps = int(round(args.hold_ms / 1000.0 * 100.0))
    rng = np.random.default_rng(args.seed)
    env = make_env(reference, args, render=False)

    rows: list[dict[str, Any]] = []
    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []
    for alpha in np.linspace(0.0, max_alpha, 13):
        hand = enforce_shared_motor_consistency(manus_hand + alpha * (dynamic_hand - manus_hand))
        for beta in np.linspace(0.0, 1.0, 9):
            candidates.append(
                (
                    hand,
                    manus_pen_pos + beta * (dynamic_pen_pos - manus_pen_pos),
                    slerp(manus_pen_quat, dynamic_pen_quat, float(beta)),
                    np.zeros(20, dtype=np.float64),
                    f"grid_a{alpha:.3f}_b{beta:.3f}",
                )
            )

    active = np.asarray([0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 19], dtype=np.int64)
    for sample in range(args.samples):
        alpha = rng.uniform(0.15 * max_alpha, max_alpha)
        beta = np.clip(alpha / max_alpha + rng.normal(0.0, 0.22), 0.0, 1.0)
        hand = manus_hand + alpha * (dynamic_hand - manus_hand)
        hand[active] += rng.normal(0.0, 0.018, size=active.size)
        hand = enforce_shared_motor_consistency(np.clip(hand, env.joint_low, env.joint_high))
        delta = hand - manus_hand
        rms = float(np.sqrt(np.mean(delta**2)))
        if rms >= args.max_rms:
            hand = enforce_shared_motor_consistency(manus_hand + delta * (args.max_rms * 0.99 / rms))
        pen_pos = manus_pen_pos + beta * (dynamic_pen_pos - manus_pen_pos)
        pen_pos += rng.normal(0.0, 0.0015, size=3)
        pen_quat = perturb_quat(
            slerp(manus_pen_quat, dynamic_pen_quat, float(beta)),
            rng.normal(0.0, np.deg2rad(2.0), size=3),
        )
        candidates.append(
            (hand, pen_pos, pen_quat, np.zeros(20, dtype=np.float64), f"random_{sample:04d}")
        )

    best: tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None = None
    accepted_count = 0
    for hand, pen_pos, pen_quat, control_bias, label in candidates:
        row = evaluate(env, manus_hand, hand, pen_pos, pen_quat, control_bias, steps)
        row["candidate"] = label
        row["accepted"] = accepted(row, args)
        accepted_count += int(row["accepted"])
        rows.append(row)
        if best is None or row["score"] > best[0]["score"]:
            best = (
                row,
                hand.copy(),
                pen_pos.copy(),
                pen_quat.copy(),
                control_bias.copy(),
                label,
            )
    assert best is not None
    _, local_hand, local_pen_pos, local_pen_quat, _, _ = best

    # Refine the best passive candidate with millimetre-scale pose changes and
    # a small constant servo preload. The measured hand pose remains subject to
    # the Manus RMS gate; only the actuator target receives this bias.
    local_samples = max(2000, args.samples * 2)
    for sample in range(local_samples):
        hand = local_hand.copy()
        hand[active] += rng.normal(0.0, 0.008, size=active.size)
        hand = enforce_shared_motor_consistency(np.clip(hand, env.joint_low, env.joint_high))
        delta = hand - manus_hand
        rms = float(np.sqrt(np.mean(delta**2)))
        if rms >= args.max_rms:
            hand = enforce_shared_motor_consistency(manus_hand + delta * (args.max_rms * 0.97 / rms))
        pen_pos = local_pen_pos + rng.normal(0.0, 0.0012, size=3)
        pen_quat = perturb_quat(local_pen_quat, rng.normal(0.0, np.deg2rad(1.2), size=3))
        control_bias = np.zeros(20, dtype=np.float64)
        control_bias[active] = rng.normal(0.0, 0.012, size=active.size)
        control_bias[8:12] += np.asarray((0.008, -0.012, -0.012, -0.012))
        control_bias = np.clip(control_bias, -0.025, 0.025)
        label = f"local_{sample:05d}"
        row = evaluate(
            env,
            manus_hand,
            hand,
            pen_pos,
            pen_quat,
            control_bias,
            steps,
        )
        row["candidate"] = label
        row["accepted"] = accepted(row, args)
        accepted_count += int(row["accepted"])
        rows.append(row)
        if best is None or row["score"] > best[0]["score"]:
            best = (
                row,
                hand.copy(),
                pen_pos.copy(),
                pen_quat.copy(),
                control_bias.copy(),
                label,
            )
    env.close()
    best_row, best_hand, best_pen_pos, best_pen_quat, best_control_bias, best_label = best

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_record = dict(reference)
    seed_record.update(
        {
            "stage": "M3.2.40 Manus-constrained canonical P3 search",
            "name": "canonical_p3_thumb_index_middle",
            "summary": "Manus-near P3 candidate with thumb/index/middle short hold.",
            "desired_contacts": ["thumb", "index", "middle"],
            "placement_support": ["thumb", "index", "middle"],
            "hand_qpos": best_hand.tolist(),
            "target_qpos": enforce_shared_motor_consistency(
                best_hand + best_control_bias
            ).tolist(),
            "joint_order": DEX5_JOINT_ORDER,
            "pen_pos": best_pen_pos.tolist(),
            "pen_quat": best_pen_quat.tolist(),
            "pen_qvel": [0.0] * 6,
            "manus_reference": str(MANUS_P3.relative_to(ROOT.parent)),
            "search_candidate": best_label,
            "validation": best_row,
        }
    )
    (output_dir / "canonical_P3_candidate_seed.json").write_text(
        json.dumps(seed_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    render_env = make_env(reference, args, render=True)
    render_env.reset(seed=0)
    set_state(render_env, best_hand, best_pen_pos, best_pen_quat)
    render_target = enforce_shared_motor_consistency(
        np.clip(best_hand + best_control_bias, render_env.joint_low, render_env.joint_high)
    )
    render_env.target_qpos = render_target.astype(np.float32)
    render_env.data.ctrl[:] = render_target
    mujoco.mj_forward(render_env.model, render_env.data)
    frames = [render_env.render()]
    for _ in range(steps):
        render_env.step(np.zeros(20, dtype=np.float32))
        frames.append(render_env.render())
    imageio.imwrite(output_dir / "canonical_P3_candidate.png", frames[0])
    imageio.mimsave(output_dir / "canonical_P3_hold100.gif", frames, duration=1.0 / render_env.control_hz)
    render_env.close()

    top = sorted(rows, key=lambda row: row["score"], reverse=True)[:25]
    report = {
        "stage": "M3.2.40 canonical P3 Manus-constrained search",
        "search_samples": len(candidates),
        "random_samples": args.samples,
        "local_refinement_samples": local_samples,
        "dynamic_p3_rms_rad": dynamic_rms,
        "acceptance": {
            "manus_rms_rad": f"<{args.max_rms}",
            "required_contacts": sorted(TRIAD),
            "hold_ms": f">={args.hold_ms}",
            "max_penetration_mm": "<4.0",
            "pen_drift_mm": "<15.0",
        },
        "accepted_count": accepted_count,
        "passed": accepted(best_row, args),
        "best": best_row,
        "top_candidates": top,
        "artifacts": {
            "seed": str(output_dir / "canonical_P3_candidate_seed.json"),
            "png": str(output_dir / "canonical_P3_candidate.png"),
            "gif": str(output_dir / "canonical_P3_hold100.gif"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
