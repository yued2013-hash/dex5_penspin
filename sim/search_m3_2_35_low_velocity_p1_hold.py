from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import build_m3_2_34_continuous_p1_p2_p3 as m334
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_cycle_closure as cycle
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_35_early_p51_closure as early
import search_m3_2_35_low_velocity_ring_entry as low_ring
import search_m3_2_35_p1_entry_hold as hold_eval


SELECTED_RING_TRAJECTORY = low_ring.RingTrajectoryCandidate(
    ring_target=(
        0.20239661083815624,
        1.256406545601292,
        0.1742256922928092,
        0.1742256922928092,
    ),
    ring_ramp=2,
    ring_gain=0.9426091228279806,
    ring_action_limit=0.01827385197708141,
)
P45_ENTRY_STEP = 2


@dataclass(frozen=True)
class LowVelocityHoldCandidate:
    measured_delta: tuple[float, ...]
    support_delta: tuple[float, ...]
    gain: float
    action_limit: float


def build_source(
    env: Any,
    route: list[dict[str, Any]],
    references: dict[str, np.ndarray],
    cache: Any,
    cache_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    front = early.build_early_p5_states(
        env, route, references, cache, cache_index, args
    )
    p3_state = front["p3_trace"]["snapshots"][-1]
    tail.restore_snapshot(env, p3_state)
    p3_base = env.target_qpos.astype(np.float64).copy()
    p34_rows, p34_snapshots = early.run_mixed_target(
        env,
        p3_base,
        tail.P34_DELTA,
        steps=early.P34_STEPS,
        delay=tail.P34_DELAY,
        ramp=tail.P34_RAMP,
        gain=tail.P34_GAIN,
        action_limit=tail.P34_ACTION_LIMIT,
        ring_target=np.asarray(SELECTED_RING_TRAJECTORY.ring_target),
        ring_ramp=SELECTED_RING_TRAJECTORY.ring_ramp,
        ring_gain=SELECTED_RING_TRAJECTORY.ring_gain,
        ring_action_limit=SELECTED_RING_TRAJECTORY.ring_action_limit,
    )
    p4_base = env.target_qpos.astype(np.float64).copy()
    p45_rows, p45_snapshots = early.run_mixed_target(
        env,
        p4_base,
        cycle.P45_DELTA,
        steps=P45_ENTRY_STEP,
        delay=cycle.P45_DELAY,
        ramp=cycle.P45_RAMP,
        gain=cycle.P45_GAIN,
        action_limit=cycle.P45_ACTION_LIMIT,
        ring_target=np.asarray(SELECTED_RING_TRAJECTORY.ring_target),
        ring_ramp=SELECTED_RING_TRAJECTORY.ring_ramp,
        ring_gain=SELECTED_RING_TRAJECTORY.ring_gain,
        ring_action_limit=SELECTED_RING_TRAJECTORY.ring_action_limit,
    )
    source_summary = low_ring.summarize_candidate(
        p34_rows, p45_rows, p45_snapshots
    )
    return p45_snapshots[-1], {
        "front": front,
        "p34_rows": p34_rows,
        "p34_snapshots": p34_snapshots,
        "p45_rows": p45_rows,
        "p45_snapshots": p45_snapshots,
        "source_summary": source_summary,
    }


def random_candidate(
    rng: np.random.Generator,
    measured_centers: list[np.ndarray],
    support_centers: list[np.ndarray],
    scale: float,
) -> LowVelocityHoldCandidate:
    measured = measured_centers[int(rng.integers(0, len(measured_centers)))].copy()
    support = support_centers[int(rng.integers(0, len(support_centers)))].copy()
    measured_std = np.zeros(20, dtype=np.float64)
    measured_std[[4, 5, 6]] = [0.020, 0.040, 0.050]
    measured_std[[8, 9, 10]] = [0.012, 0.025, 0.030]
    support_std = np.zeros(20, dtype=np.float64)
    support_std[[0, 1, 2]] = [0.008, 0.018, 0.024]
    support_std[16:20] = [0.006, 0.016, 0.025, 0.025]
    measured += rng.normal(0.0, scale * measured_std)
    support += rng.normal(0.0, scale * support_std)
    measured[3] = measured[2]
    measured[7] = measured[6]
    measured[11] = measured[10]
    measured[0:4] = 0.0
    measured[12:20] = 0.0
    support[3] = support[2]
    support[4:16] = 0.0
    return LowVelocityHoldCandidate(
        measured_delta=tuple(
            m316.enforce_shared_motor_consistency(measured).astype(float)
        ),
        support_delta=tuple(
            m316.enforce_shared_motor_consistency(support).astype(float)
        ),
        gain=float(rng.uniform(0.45, 0.95)),
        action_limit=float(rng.uniform(0.008, 0.035)),
    )


def run_hold(
    env: Any,
    source: dict[str, Any],
    candidate: LowVelocityHoldCandidate,
    *,
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tail.restore_snapshot(env, source)
    base_target = env.target_qpos.astype(np.float64).copy()
    measured = env.get_state().qpos.astype(np.float64).copy()
    desired = base_target + np.asarray(candidate.support_delta, dtype=np.float64)
    measured_delta = np.asarray(candidate.measured_delta, dtype=np.float64)
    desired[4:12] = measured[4:12] + measured_delta[4:12]
    target = np.clip(
        m316.enforce_shared_motor_consistency(desired),
        env.joint_low,
        env.joint_high,
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    for step in range(steps):
        action = np.clip(
            candidate.gain * (target - env.target_qpos.astype(np.float64)),
            -candidate.action_limit,
            candidate.action_limit,
        ).astype(np.float32)
        env.step(action)
        rows.append(
            early.rollout_row(
                env,
                step=step + 1,
                initial_center=initial.pen_center,
                initial_axis=initial.pen_axis,
            )
        )
    final = rows[-1]
    return rows, early.m333.snapshot(
        env,
        final["contacts"],
        final["max_penetration_mm"],
        steps / env.control_hz,
    )


def rank_summary(summary: dict[str, Any]) -> tuple[bool, float]:
    ratios = summary["contact_ratio"]
    score = float(
        summary["score"]
        + 600.0 * ratios["ring"]
        + 320.0 * ratios["middle"]
        + 600.0 * summary["four_finger_contact_ratio"]
        + 200.0 * summary["p1_support_ratio"]
    )
    summary["low_velocity_hold_score"] = score
    return bool(summary["accepted"]), score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "low_velocity_p1_hold_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--source-t", type=float, default=0.21)
    parser.add_argument("--source-segment-duration", type=float, default=0.18)
    parser.add_argument("--b12-duration", type=float, default=0.58)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    args = parser.parse_args()
    args.action_delta_limit = 0.040
    args.thumb_force_limit = 0.0
    args.p1_index_open = -0.012
    args.p1_middle_open = -0.004
    args.p1_ring_open = -0.020
    args.p1_thumb_open = 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env, route, references, cache, cache_index = early.make_env_and_inputs(args)
    rng = np.random.default_rng(args.seed)
    try:
        source_state, source = build_source(
            env, route, references, cache, cache_index, args
        )
        measured_bias = np.zeros(20, dtype=np.float64)
        measured_bias[5:8] = [0.006, 0.012, 0.012]
        measured_bias[9:12] = [0.006, 0.002, 0.002]
        support_bias = np.zeros(20, dtype=np.float64)
        support_bias[1:4] = [-0.003, 0.004, 0.004]
        support_bias[17:20] = [-0.005, -0.008, -0.008]
        measured_centers = [np.zeros(20, dtype=np.float64), measured_bias]
        support_centers = [np.zeros(20, dtype=np.float64), support_bias]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, args.samples // max(1, args.generations))
        for generation in range(args.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                candidate = random_candidate(
                    rng, measured_centers, support_centers, scale
                )
                rows, _ = run_hold(
                    env, source_state, candidate, steps=args.steps
                )
                summary = hold_eval.hold_summary(rows)
                rank_summary(summary)
                generation_rows.append(
                    {
                        "generation": generation,
                        "sample": sample,
                        "candidate": asdict(candidate),
                        "summary": summary,
                    }
                )
            generation_rows.sort(
                key=lambda row: (
                    row["summary"]["accepted"],
                    row["summary"]["low_velocity_hold_score"],
                ),
                reverse=True,
            )
            leaderboard.extend(generation_rows[: max(args.elite, 100)])
            measured_centers = [
                np.asarray(row["candidate"]["measured_delta"], dtype=np.float64)
                for row in generation_rows[: args.elite]
            ]
            support_centers = [
                np.asarray(row["candidate"]["support_delta"], dtype=np.float64)
                for row in generation_rows[: args.elite]
            ]
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "accepted": sum(
                            row["summary"]["accepted"] for row in generation_rows
                        ),
                        "best": generation_rows[0],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        leaderboard.sort(
            key=lambda row: (
                row["summary"]["accepted"],
                row["summary"]["low_velocity_hold_score"],
            ),
            reverse=True,
        )
        best = leaderboard[0]
        best_candidate = LowVelocityHoldCandidate(**best["candidate"])
        _, final_state = run_hold(
            env, source_state, best_candidate, steps=args.steps
        )
        m334.save_final_cache(output_dir / "low_velocity_P1_hold_arrival.npz", final_state)
        report = {
            "stage": "M3.2.35 low-velocity P1-entry hold",
            "ring_trajectory": asdict(SELECTED_RING_TRAJECTORY),
            "source_summary": source["source_summary"],
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "steps": args.steps,
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "low_velocity_p1_hold_search.json", report)
        print(
            json.dumps(
                {"output_dir": str(output_dir), "best": best},
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
