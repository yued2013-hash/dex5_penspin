from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import build_m3_2_33_narrow_021_five_step_chain as m333
import build_m3_2_34_continuous_p1_p2_p3 as m334
import export_m3_2_23_dynamic_state_cache as m323
import rebuild_m3_2_16_p4_p5_transitions as m316
from evaluate_m3_2_15_short_primitives import FINGERS, contact_metrics


SHORT_P3 = m333.B23Candidate(
    "short_P3_exit",
    1.80,
    1.10,
    0.20,
    -0.040,
    -0.020,
    0.88,
    recapture_steps=1,
)

# Shooting result selected from the live short-P3 state. This is deliberately a
# dynamic P4 entry: middle participates during the segment but need not remain
# in contact at its last frame.
P34_DELTA = np.asarray(
    [
        -0.008212940195268331,
        -0.08102027530906356,
        0.11379739217027007,
        0.11379739217027007,
        0.06347213966968251,
        -0.16780925669953844,
        0.13761007652784768,
        0.13761007652784768,
        -0.055727262529983855,
        -0.06838900466525935,
        0.0723211780252126,
        0.0723211780252126,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02476748466491313,
        0.010646412536595928,
        -0.10944659396522408,
        0.022623720113772033,
    ],
    dtype=np.float64,
)
P34_DELAY = 8
P34_RAMP = 4
P34_GAIN = 0.41945897
P34_ACTION_LIMIT = 0.00774926
P34_STEPS = 30


@dataclass(frozen=True)
class ShootingCandidate:
    delta: tuple[float, ...]
    delay: int
    ramp: int
    gain: float
    action_limit: float


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def make_args(parsed: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=parsed.seed,
        action_delta_limit=0.040,
        actuator_kp=parsed.actuator_kp,
        actuator_kv=parsed.actuator_kv,
        thumb_force_limit=0.0,
        tip_proxy_radius=parsed.tip_proxy_radius,
        tip_proxy_friction=parsed.tip_proxy_friction,
        source_t=0.21,
        source_segment_duration=0.18,
        b12_duration=0.58,
        render_every=2,
        p1_index_open=-0.012,
        p1_middle_open=-0.004,
        p1_ring_open=-0.020,
        p1_thumb_open=0.0,
    )


def load_references(reference_dir: Path) -> dict[str, np.ndarray]:
    files = {
        "p2": "P2_ring_bottom_thumb_cradle_support_seed.json",
        "b23a": "B23a_index_front_deflect_seed.json",
        "b23a_contact": "B23a_index_front_contact_deflect_seed.json",
        "b23b": "B23b_index_middle_drive_thumb_yield_seed.json",
        "p3": "P3_index_front_pass_exit_seed.json",
    }
    return {
        key: np.asarray(
            json.loads((reference_dir / filename).read_text(encoding="utf-8"))[
                "hand_qpos"
            ],
            dtype=np.float64,
        )
        for key, filename in files.items()
    }


def state_snapshot(env: Any, t: float = 0.0) -> dict[str, Any]:
    metrics = contact_metrics(env)
    contacts = ",".join(sorted(metrics["fingers"])) or "none"
    return m333.snapshot(
        env,
        contacts,
        float(metrics["max_penetration"] * 1000.0),
        t,
    )


def restore_snapshot(env: Any, state: dict[str, Any]) -> None:
    # Reset MuJoCo's contact solver warm-start state before restoring the live
    # snapshot. Reusing only qpos/qvel makes adjacent shooting candidates leak
    # solver history into one another.
    env.reset(seed=0)
    env.data.qpos[:] = state["qpos"]
    env.data.qvel[:] = state["qvel"]
    env.target_qpos = np.asarray(state["target_qpos"], dtype=np.float32).copy()
    env.data.ctrl[:] = env.target_qpos
    env.last_action[:] = 0.0
    env.step_count = 0
    env.rotation_phase = float(state["rotation_phase"])
    mujoco.mj_forward(env.model, env.data)
    env.prev_phase = env._quat_twist_phase_about_x(env.data.xquat[env.pen_body_id])


def run_target(
    env: Any,
    base_target: np.ndarray,
    delta: np.ndarray,
    *,
    steps: int,
    delay: int,
    ramp: int,
    gain: float,
    action_limit: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = np.clip(
        m316.enforce_shared_motor_consistency(base_target + delta),
        env.joint_low,
        env.joint_high,
    )
    initial = env.get_state()
    initial_center = initial.pen_center.copy()
    initial_axis = initial.pen_axis.copy()
    rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    for step in range(steps):
        alpha = 0.0 if step < delay else min(1.0, (step - delay + 1) / max(1, ramp))
        effective_target = base_target + alpha * (target - base_target)
        action = np.clip(
            gain * (effective_target - env.target_qpos.astype(np.float64)),
            -action_limit,
            action_limit,
        ).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        state = env.get_state()
        metrics = contact_metrics(env)
        tip_distances = m316.tip_surface_distances(env)
        fingers = set(metrics["fingers"])
        displacement = (state.pen_center - initial_center) * 1000.0
        row: dict[str, Any] = {
            "step": step + 1,
            "contacts": ",".join(sorted(fingers)) or "none",
            "dx_mm": float(displacement[0]),
            "dy_mm": float(displacement[1]),
            "dz_mm": float(displacement[2]),
            "max_penetration_mm": float(metrics["max_penetration"] * 1000.0),
            "pen_linear_speed": float(
                np.linalg.norm(
                    env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 3]
                )
            ),
            "pen_angular_speed": float(
                np.linalg.norm(
                    env.data.qvel[env.pen_qvel_adr + 3 : env.pen_qvel_adr + 6]
                )
            ),
            "planar_rotation_deg": m333.signed_planar_rotation_deg(
                initial_axis, state.pen_axis
            ),
        }
        for finger in FINGERS:
            row[f"{finger}_share"] = float(
                metrics["force_share"].get(finger, 0.0)
            )
            row[f"{finger}_tip_distance_mm"] = float(
                tip_distances.get(finger, 0.1) * 1000.0
            )
        rows.append(row)
        if terminated or truncated or row["dz_mm"] < -15.0:
            break
    return rows, {
        "terminated": terminated,
        "truncated": truncated,
        "target_qpos": target,
    }


def summarize(rows: list[dict[str, Any]], expected_steps: int) -> dict[str, Any]:
    def contact_ratio(finger: str) -> float:
        return float(
            sum(finger in row["contacts"].split(",") for row in rows)
            / max(1, len(rows))
        )

    final_contacts = []
    if rows and rows[-1]["contacts"] != "none":
        final_contacts = rows[-1]["contacts"].split(",")
    active_ratio = float(
        sum(len(set(row["contacts"].split(",")) - {"none"}) >= 2 for row in rows)
        / max(1, len(rows))
    )
    result = {
        "steps": len(rows),
        "survival_s": len(rows) / 100.0,
        "min_dz_mm": float(min(row["dz_mm"] for row in rows)),
        "final_dz_mm": float(rows[-1]["dz_mm"]),
        "max_penetration_mm": float(
            max(row["max_penetration_mm"] for row in rows)
        ),
        "final_linear_speed": float(rows[-1]["pen_linear_speed"]),
        "final_angular_speed": float(rows[-1]["pen_angular_speed"]),
        "rotation_deg": float(rows[-1]["planar_rotation_deg"]),
        "active_contact_ratio": active_ratio,
        "contact_ratio": {finger: contact_ratio(finger) for finger in FINGERS},
        "force_share_mean": {
            finger: float(
                np.mean([row[f"{finger}_share"] for row in rows])
            )
            for finger in FINGERS
        },
        "tip_distance_final_mm": {
            finger: float(rows[-1][f"{finger}_tip_distance_mm"])
            for finger in FINGERS
        },
        "tip_distance_min_mm": {
            finger: float(
                min(row[f"{finger}_tip_distance_mm"] for row in rows)
            )
            for finger in FINGERS
        },
        "final_contacts": final_contacts,
    }
    ratios = result["contact_ratio"]
    shares = result["force_share_mean"]
    support_score = (
        105.0 * ratios["thumb"]
        + 90.0 * ratios["index"]
        + 22.0 * ratios["middle"]
        + 45.0 * active_ratio
        + 35.0 * min(1.0, shares["thumb"] / 0.28)
    )
    final_bonus = 80.0 * float(
        {"thumb", "index"}.issubset(set(final_contacts))
    )
    motion_score = 2.0 * min(12.0, abs(result["rotation_deg"]))
    penalties = (
        35.0 * max(0, expected_steps - len(rows))
        + 14.0 * max(0.0, -result["min_dz_mm"] - 4.0)
        + 18.0 * max(0.0, result["max_penetration_mm"] - 4.0)
        + 50.0 * ratios["pinky"]
        + 18.0 * ratios["ring"]
        + 40.0 * max(0.0, shares["ring"] - 0.18)
        + 90.0 * max(0.0, result["final_linear_speed"] - 0.08)
    )
    result["score"] = float(
        4.0 * len(rows) + support_score + final_bonus + motion_score - penalties
    )
    result["accepted"] = bool(
        len(rows) == expected_steps
        and result["min_dz_mm"] >= -6.0
        and result["max_penetration_mm"] <= 4.0
        and ratios["thumb"] >= 0.80
        and ratios["index"] >= 0.65
        and active_ratio >= 0.80
        and shares["ring"] <= 0.20
        and {"thumb", "index"}.issubset(set(final_contacts))
    )
    return result


def build_dynamic_p4(
    env: Any, args: argparse.Namespace, root: Path, *, p34_steps: int
) -> dict[str, Any]:
    cache = np.load(
        root
        / "short_primitive_reward_m3_2_23/dynamic_state_cache/earlier_pre_ring_for_B12b_to_B23.npz"
    )
    cache_index = m333.m332.cache_index(cache, args.source_t)
    route = m323.build_route(root / "photo_five_phase_hand_pen_tuned_m3_2_10")
    safety = {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0}
    initial_seed = m334.apply_p1_safety_open(route, safety)[0]
    references = load_references(root / "p2_bottom_cradle_b23_thumb_pass_m3_2_31")
    m334.prelude_rollout(
        env,
        route,
        initial_seed,
        m334.source_controller_args(args),
        args,
        render=False,
    )
    _, b12_trace = m333.b12_rollout(
        env,
        cache,
        cache_index,
        m333.Perturbation("continuous_nominal"),
        m334.SELECTED_B12,
        args,
        duration_s=args.b12_duration,
        render=False,
        reset_state=False,
    )
    p3_summary, _ = m333.b23_rollout(
        env,
        b12_trace["snapshots"][-1],
        references,
        SHORT_P3,
        args,
        label="short_P3_exit",
        render=False,
        reset_state=False,
    )
    p3_state = state_snapshot(env)
    p3_base = env.target_qpos.astype(np.float64).copy()
    p34_rows, _ = run_target(
        env,
        p3_base,
        P34_DELTA,
        steps=p34_steps,
        delay=P34_DELAY,
        ramp=P34_RAMP,
        gain=P34_GAIN,
        action_limit=P34_ACTION_LIMIT,
    )
    return {
        "p3_summary": p3_summary,
        "p3_state": p3_state,
        "p34_summary": summarize(p34_rows, p34_steps),
        "p4_state": state_snapshot(env),
    }


def random_delta(rng: np.random.Generator, center: np.ndarray, scale: float) -> np.ndarray:
    std = np.zeros(20, dtype=np.float64)
    std[[0, 4, 8]] = [0.045, 0.055, 0.045]
    std[[1, 5, 9]] = [0.105, 0.135, 0.105]
    std[[2, 6, 10]] = [0.120, 0.145, 0.120]
    std[16:20] = [0.035, 0.085, 0.125, 0.130]
    delta = center + rng.normal(0.0, scale * std)
    delta[3] = delta[2]
    delta[7] = delta[6]
    delta[11] = delta[10]
    delta[12:16] = 0.0
    return m316.enforce_shared_motor_consistency(delta)


def candidate_from_rng(
    rng: np.random.Generator, center: np.ndarray, scale: float
) -> ShootingCandidate:
    return ShootingCandidate(
        delta=tuple(random_delta(rng, center, scale).astype(float)),
        delay=int(rng.integers(0, 5)),
        ramp=int(rng.integers(2, 8)),
        gain=float(rng.uniform(0.28, 0.78)),
        action_limit=float(rng.uniform(0.0025, 0.0105)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "dynamic_tail_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=32)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--p34-exit-step", type=int, default=16)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parsed = parser.parse_args()
    args = make_args(parsed)
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path("dex5_penspin/reports/manus_mujoco_presets")
    route = m323.build_route(root / "photo_five_phase_hand_pen_tuned_m3_2_10")
    initial_seed = m334.apply_p1_safety_open(
        route, {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0}
    )[0]
    env = m316.make_env(initial_seed, m334.model_args(args), duration=4.0, render=False)
    rng = np.random.default_rng(parsed.seed)
    try:
        source = build_dynamic_p4(
            env, args, root, p34_steps=parsed.p34_exit_step
        )
        p4_state = source["p4_state"]
        base_target = np.asarray(p4_state["target_qpos"], dtype=np.float64)
        leaderboard: list[dict[str, Any]] = []
        centers = [np.zeros(20, dtype=np.float64)]
        per_generation = max(1, parsed.samples // max(1, parsed.generations))
        for generation in range(parsed.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample_index in range(per_generation):
                center = centers[sample_index % len(centers)]
                candidate = candidate_from_rng(rng, center, scale)
                restore_snapshot(env, p4_state)
                rows, _ = run_target(
                    env,
                    base_target,
                    np.asarray(candidate.delta, dtype=np.float64),
                    steps=parsed.steps,
                    delay=candidate.delay,
                    ramp=candidate.ramp,
                    gain=candidate.gain,
                    action_limit=candidate.action_limit,
                )
                summary = summarize(rows, parsed.steps)
                generation_rows.append(
                    {
                        "generation": generation,
                        "sample": sample_index,
                        "candidate": asdict(candidate),
                        "summary": summary,
                    }
                )
            generation_rows.sort(
                key=lambda row: (
                    row["summary"]["accepted"],
                    row["summary"]["score"],
                ),
                reverse=True,
            )
            leaderboard.extend(generation_rows[: max(parsed.elite, 64)])
            centers = [
                np.asarray(row["candidate"]["delta"], dtype=np.float64)
                for row in generation_rows[: parsed.elite]
            ]
            best = generation_rows[0]
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "accepted": sum(
                            row["summary"]["accepted"] for row in generation_rows
                        ),
                        "best": best,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        leaderboard.sort(
            key=lambda row: (
                row["summary"]["accepted"],
                row["summary"]["score"],
            ),
            reverse=True,
        )
        report = {
            "stage": "M3.2.35 dynamic P4-entry to P5 shooting search",
            "source": {
                "p3_summary": source["p3_summary"],
                "p34_summary": source["p34_summary"],
                "p34_delta": P34_DELTA.astype(float).tolist(),
            },
            "search": {
                "samples_requested": parsed.samples,
                "generations": parsed.generations,
                "samples_per_generation": per_generation,
                "steps": parsed.steps,
                "p34_exit_step": parsed.p34_exit_step,
                "accepted_in_leaderboard": sum(
                    row["summary"]["accepted"] for row in leaderboard
                ),
            },
            "leaderboard": leaderboard[:100],
        }
        save_json(output_dir / "p45_shooting_search.json", report)
        m334.save_final_cache(output_dir / "dynamic_P4_entry.npz", p4_state)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "p34": source["p34_summary"],
                    "best": leaderboard[0],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
