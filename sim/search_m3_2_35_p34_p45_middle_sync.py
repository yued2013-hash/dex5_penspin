from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_35_early_p51_closure as early
import search_m3_2_35_p45_middle_ring_sync as p45_sync


@dataclass(frozen=True)
class TwoStageMiddleCandidate:
    p34_middle_delta: tuple[float, float, float, float]
    p34_middle_ramp: int
    p34_middle_gain: float
    p34_middle_action_limit: float
    p45_middle_delta: tuple[float, float, float, float]
    p45_middle_ramp: int
    p45_middle_gain: float
    p45_middle_action_limit: float


def random_candidate(
    rng: np.random.Generator,
    p34_centers: list[np.ndarray],
    p45_centers: list[np.ndarray],
    scale: float,
) -> TwoStageMiddleCandidate:
    p34 = p34_centers[int(rng.integers(0, len(p34_centers)))] + rng.normal(
        0.0, scale * np.asarray([0.040, 0.110, 0.140, 0.140])
    )
    p45 = p45_centers[int(rng.integers(0, len(p45_centers)))] + rng.normal(
        0.0, scale * np.asarray([0.050, 0.140, 0.180, 0.180])
    )
    p34[3] = p34[2]
    p45[3] = p45[2]
    return TwoStageMiddleCandidate(
        p34_middle_delta=tuple(p34.astype(float)),
        p34_middle_ramp=int(rng.integers(2, 8)),
        p34_middle_gain=float(rng.uniform(0.35, 0.90)),
        p34_middle_action_limit=float(rng.uniform(0.003, 0.014)),
        p45_middle_delta=tuple(p45.astype(float)),
        p45_middle_ramp=int(rng.integers(1, 7)),
        p45_middle_gain=float(rng.uniform(0.35, 1.00)),
        p45_middle_action_limit=float(rng.uniform(0.003, 0.025)),
    )


def run_p34(
    env: Any,
    p3_state: dict[str, Any],
    candidate: TwoStageMiddleCandidate,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tail.restore_snapshot(env, p3_state)
    base = env.target_qpos.astype(np.float64).copy()
    target = np.clip(
        m316.enforce_shared_motor_consistency(base + tail.P34_DELTA),
        env.joint_low,
        env.joint_high,
    )
    middle_target = np.clip(
        target[4:8] + np.asarray(candidate.p34_middle_delta),
        env.joint_low[4:8],
        env.joint_high[4:8],
    )
    ring_target = np.clip(
        np.asarray(p45_sync.SELECTED_RING.p34_ring_target),
        env.joint_low[8:12],
        env.joint_high[8:12],
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for step in range(early.P34_STEPS):
        alpha = 0.0 if step < tail.P34_DELAY else min(
            1.0, (step - tail.P34_DELAY + 1) / tail.P34_RAMP
        )
        effective = base + alpha * (target - base)
        action = np.clip(
            tail.P34_GAIN * (effective - env.target_qpos.astype(np.float64)),
            -tail.P34_ACTION_LIMIT,
            tail.P34_ACTION_LIMIT,
        )
        middle_alpha = 0.0 if step < tail.P34_DELAY else min(
            1.0,
            (step - tail.P34_DELAY + 1) / max(1, candidate.p34_middle_ramp),
        )
        middle_effective = base[4:8] + middle_alpha * (
            middle_target - base[4:8]
        )
        action[4:8] = np.clip(
            candidate.p34_middle_gain
            * (middle_effective - env.target_qpos[4:8].astype(np.float64)),
            -candidate.p34_middle_action_limit,
            candidate.p34_middle_action_limit,
        )
        ring_alpha = min(
            1.0, (step + 1) / p45_sync.SELECTED_RING.p34_ring_ramp
        )
        ring_effective = base[8:12] + ring_alpha * (ring_target - base[8:12])
        action[8:12] = np.clip(
            p45_sync.SELECTED_RING.p34_ring_gain
            * (ring_effective - env.target_qpos[8:12].astype(np.float64)),
            -p45_sync.SELECTED_RING.p34_ring_action_limit,
            p45_sync.SELECTED_RING.p34_ring_action_limit,
        )
        env.step(action.astype(np.float32))
        row = early.rollout_row(
            env,
            step=step + 1,
            initial_center=initial.pen_center,
            initial_axis=initial.pen_axis,
        )
        rows.append(row)
        snapshots.append(
            early.m333.snapshot(
                env,
                row["contacts"],
                row["max_penetration_mm"],
                (step + 1) / env.control_hz,
            )
        )
    return rows, snapshots


def p34_ratio(rows: list[dict[str, Any]], finger: str) -> float:
    return float(
        sum(finger in row["contacts"].split(",") for row in rows)
        / max(1, len(rows))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p34_p45_middle_sync_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=48)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=141)
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
        front = early.build_early_p5_states(
            env, route, references, cache, cache_index, args
        )
        p3_state = front["p3_trace"]["snapshots"][-1]
        p34_centers = [
            np.zeros(4, dtype=np.float64),
            np.asarray([0.0, 0.04, 0.08, 0.08]),
            np.asarray([0.0, -0.06, 0.12, 0.12]),
        ]
        p45_centers = [
            np.asarray([-0.06463675752357557, -0.14887103012617847, 0.09067742236287765, 0.09067742236287765]),
            np.zeros(4, dtype=np.float64),
            np.asarray([0.0, -0.08, 0.16, 0.16]),
        ]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, args.samples // max(1, args.generations))
        for generation in range(args.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                candidate = random_candidate(
                    rng, p34_centers, p45_centers, scale
                )
                p34_rows, p34_snapshots = run_p34(env, p3_state, candidate)
                p45_candidate = p45_sync.MiddleSyncCandidate(
                    middle_delta=candidate.p45_middle_delta,
                    middle_ramp=candidate.p45_middle_ramp,
                    middle_gain=candidate.p45_middle_gain,
                    middle_action_limit=candidate.p45_middle_action_limit,
                )
                p45_rows, _ = p45_sync.run_p45(
                    env, p34_snapshots[-1], p45_candidate, steps=args.steps
                )
                summary = p45_sync.summarize(p45_rows)
                p34_support = min(
                    p34_ratio(p34_rows, "thumb"),
                    p34_ratio(p34_rows, "index"),
                    p34_ratio(p34_rows, "middle"),
                )
                p34_penetration = max(
                    row["max_penetration_mm"] for row in p34_rows
                )
                summary["p34_support_ratio"] = p34_support
                summary["p34_max_penetration_mm"] = p34_penetration
                summary["accepted"] = bool(
                    summary["accepted"]
                    and p34_support >= 0.85
                    and p34_penetration <= 4.0
                )
                summary["score"] += 180.0 * p34_support
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
                    row["summary"]["score"],
                ),
                reverse=True,
            )
            leaderboard.extend(generation_rows[: max(args.elite, 100)])
            p34_centers = [
                np.asarray(row["candidate"]["p34_middle_delta"])
                for row in generation_rows[: args.elite]
            ]
            p45_centers = [
                np.asarray(row["candidate"]["p45_middle_delta"])
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
                row["summary"]["score"],
            ),
            reverse=True,
        )
        report = {
            "stage": "M3.2.35 two-stage P34/P45 middle-ring synchronization",
            "ring_trajectory": asdict(p45_sync.SELECTED_RING),
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "p45_steps": args.steps,
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "p34_p45_middle_sync_search.json", report)
        print(
            json.dumps(
                {"output_dir": str(output_dir), "best": leaderboard[0]},
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
