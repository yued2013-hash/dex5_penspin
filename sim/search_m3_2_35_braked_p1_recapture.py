from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import build_m3_2_34_continuous_p1_p2_p3 as m334
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_35_early_p51_closure as early
import search_m3_2_35_p1_entry_hold as hold_eval


@dataclass(frozen=True)
class BrakedRecaptureCandidate:
    source_step: int
    measured_delta: tuple[float, ...]
    support_delta: tuple[float, ...]
    gain: float
    action_limit: float


def random_candidate(
    rng: np.random.Generator,
    measured_centers: list[np.ndarray],
    support_centers: list[np.ndarray],
    scale: float,
) -> BrakedRecaptureCandidate:
    measured_center = measured_centers[int(rng.integers(0, len(measured_centers)))]
    support_center = support_centers[int(rng.integers(0, len(support_centers)))]
    measured_std = np.zeros(20, dtype=np.float64)
    measured_std[[4, 5, 6]] = [0.020, 0.045, 0.060]
    measured_std[[8, 9, 10]] = [0.008, 0.014, 0.012]
    measured_delta = measured_center + rng.normal(0.0, scale * measured_std)
    measured_delta[3] = measured_delta[2]
    measured_delta[7] = measured_delta[6]
    measured_delta[11] = measured_delta[10]
    measured_delta[0:4] = 0.0
    measured_delta[12:20] = 0.0

    support_std = np.zeros(20, dtype=np.float64)
    support_std[[0, 1, 2]] = [0.010, 0.020, 0.025]
    support_std[16:20] = [0.008, 0.020, 0.030, 0.030]
    support_delta = support_center + rng.normal(0.0, scale * support_std)
    support_delta[3] = support_delta[2]
    support_delta[4:16] = 0.0
    return BrakedRecaptureCandidate(
        source_step=5 if rng.random() < 0.85 else 4,
        measured_delta=tuple(
            m316.enforce_shared_motor_consistency(measured_delta).astype(float)
        ),
        support_delta=tuple(
            m316.enforce_shared_motor_consistency(support_delta).astype(float)
        ),
        gain=float(rng.uniform(0.65, 1.00)),
        action_limit=float(rng.uniform(0.025, 0.040)),
    )


def run_recapture(
    env: Any,
    source: dict[str, Any],
    candidate: BrakedRecaptureCandidate,
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
    distances = summary["tip_distance_min_mm"]
    braked_score = float(
        summary["score"]
        + 650.0 * ratios["ring"]
        + 260.0 * ratios["middle"]
        + 500.0 * summary["four_finger_contact_ratio"]
        + 45.0 * max(0.0, 12.0 - distances["ring"])
        + 25.0 * max(0.0, 12.0 - distances["middle"])
    )
    summary["braked_score"] = braked_score
    return bool(summary["accepted"]), braked_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "braked_p1_recapture_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=16000)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=91)
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
        source = early.build_early_p5_states(
            env, route, references, cache, cache_index, args
        )
        source_states = {
            step: source["p45_snapshots"][step - 1] for step in (4, 5)
        }

        measured_bias = np.zeros(20, dtype=np.float64)
        measured_bias[5:8] = [0.020, 0.035, 0.035]
        measured_bias[9:12] = [0.016, -0.003, -0.003]
        support_bias = np.zeros(20, dtype=np.float64)
        support_bias[1:4] = [-0.004, 0.006, 0.006]
        support_bias[17:20] = [-0.008, -0.012, -0.012]
        measured_centers = [measured_bias]
        support_centers = [np.zeros(20, dtype=np.float64), support_bias]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, args.samples // max(1, args.generations))
        for generation in range(args.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                if sample == 0:
                    candidate = BrakedRecaptureCandidate(
                        source_step=5,
                        measured_delta=tuple(measured_centers[0].astype(float)),
                        support_delta=tuple(support_centers[0].astype(float)),
                        gain=0.80,
                        action_limit=0.040,
                    )
                else:
                    candidate = random_candidate(
                        rng, measured_centers, support_centers, scale
                    )
                rollout, _ = run_recapture(
                    env,
                    source_states[candidate.source_step],
                    candidate,
                    steps=args.steps,
                )
                summary = hold_eval.hold_summary(rollout)
                rank_summary(summary)
                rows.append(
                    {
                        "generation": generation,
                        "sample": sample,
                        "candidate": asdict(candidate),
                        "summary": summary,
                    }
                )
            rows.sort(
                key=lambda row: (
                    row["summary"]["accepted"],
                    row["summary"]["braked_score"],
                ),
                reverse=True,
            )
            leaderboard.extend(rows[: max(args.elite, 100)])
            measured_centers = [
                np.asarray(row["candidate"]["measured_delta"], dtype=np.float64)
                for row in rows[: args.elite]
            ]
            support_centers = [
                np.asarray(row["candidate"]["support_delta"], dtype=np.float64)
                for row in rows[: args.elite]
            ]
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "accepted": sum(row["summary"]["accepted"] for row in rows),
                        "best": rows[0],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        leaderboard.sort(
            key=lambda row: (
                row["summary"]["accepted"],
                row["summary"]["braked_score"],
            ),
            reverse=True,
        )
        best = leaderboard[0]
        best_candidate = BrakedRecaptureCandidate(**best["candidate"])
        _, final_state = run_recapture(
            env,
            source_states[best_candidate.source_step],
            best_candidate,
            steps=args.steps,
        )
        m334.save_final_cache(output_dir / "braked_P1_entry_arrival.npz", final_state)
        report = {
            "stage": "M3.2.35 simultaneous ring brake and P1 recapture",
            "source": {
                "prepared_b23": asdict(early.PREPARED_P3),
                "p34_summary": tail.summarize(source["p34_rows"], early.P34_STEPS),
                "p45_rows": source["p45_rows"],
            },
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "steps": args.steps,
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "braked_p1_recapture_search.json", report)
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
