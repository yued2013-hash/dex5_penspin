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
import search_m3_2_35_p1_entry_hold as hold_eval
import search_m3_2_35_sustained_ring_entry as sustained


SELECTED_TAIL = sustained.SustainedRingCandidate(
    p34_ring_target=(
        0.23398004297820532,
        1.430236486771153,
        0.13646868745067597,
        0.13646868745067597,
    ),
    p34_ring_ramp=6,
    p34_ring_gain=0.9387127178762176,
    p34_ring_action_limit=0.0165143425239351,
    p45_ring_target=(
        0.18664904250689254,
        1.221311702576487,
        0.46099750512088034,
        0.46099750512088034,
    ),
    p45_ring_ramp=7,
    p45_ring_gain=0.5822719141671392,
    p45_ring_action_limit=0.010221550226792565,
)


@dataclass(frozen=True)
class P1RejoinCandidate:
    source_step: int
    index_scale: float
    middle_scale: float
    ring_scale: float
    thumb_scale: float
    ramp: int
    gain: float
    action_limit: float


def build_tail_sources(
    env: Any,
    route: list[dict[str, Any]],
    references: dict[str, np.ndarray],
    cache: Any,
    cache_index: int,
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], np.ndarray]:
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
        ring_target=np.asarray(SELECTED_TAIL.p34_ring_target),
        ring_ramp=SELECTED_TAIL.p34_ring_ramp,
        ring_gain=SELECTED_TAIL.p34_ring_gain,
        ring_action_limit=SELECTED_TAIL.p34_ring_action_limit,
    )
    p4_base = env.target_qpos.astype(np.float64).copy()
    p45_rows, p45_snapshots = early.run_mixed_target(
        env,
        p4_base,
        cycle.P45_DELTA,
        steps=12,
        delay=cycle.P45_DELAY,
        ramp=cycle.P45_RAMP,
        gain=cycle.P45_GAIN,
        action_limit=cycle.P45_ACTION_LIMIT,
        ring_target=np.asarray(SELECTED_TAIL.p45_ring_target),
        ring_ramp=SELECTED_TAIL.p45_ring_ramp,
        ring_gain=SELECTED_TAIL.p45_ring_gain,
        ring_action_limit=SELECTED_TAIL.p45_ring_action_limit,
    )
    eligible_steps = [
        index + 1
        for index, row in enumerate(p45_rows)
        if {"thumb", "index", "ring"}.issubset(set(row["contacts"].split(",")))
        or (
            {"thumb", "index"}.issubset(set(row["contacts"].split(",")))
            and row["ring_tip_distance_mm"] <= 8.75
        )
    ]
    if not eligible_steps:
        eligible_steps = [1, 2, 3]
    sources = {step: p45_snapshots[step - 1] for step in eligible_steps}
    p1_target = np.asarray(
        m334.apply_p1_safety_open(
            route,
            {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0},
        )[0]["hand_qpos"],
        dtype=np.float64,
    )
    return sources, {
        "front": front,
        "p34_rows": p34_rows,
        "p34_snapshots": p34_snapshots,
        "p45_rows": p45_rows,
        "p45_snapshots": p45_snapshots,
        "eligible_steps": eligible_steps,
    }, p1_target


def random_candidate(
    rng: np.random.Generator,
    centers: list[np.ndarray],
    source_steps: list[int],
    scale: float,
) -> P1RejoinCandidate:
    center = centers[int(rng.integers(0, len(centers)))]
    scales = np.clip(
        center + rng.normal(0.0, scale * np.asarray([0.16, 0.20, 0.14, 0.18])),
        0.0,
        1.0,
    )
    return P1RejoinCandidate(
        source_step=int(source_steps[int(rng.integers(0, len(source_steps)))]),
        index_scale=float(scales[0]),
        middle_scale=float(scales[1]),
        ring_scale=float(scales[2]),
        thumb_scale=float(scales[3]),
        ramp=int(rng.integers(2, 11)),
        gain=float(rng.uniform(0.28, 0.82)),
        action_limit=float(rng.uniform(0.002, 0.014)),
    )


def run_rejoin(
    env: Any,
    source: dict[str, Any],
    p1_target: np.ndarray,
    candidate: P1RejoinCandidate,
    *,
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tail.restore_snapshot(env, source)
    base = env.target_qpos.astype(np.float64).copy()
    target = base.copy()
    scales = (
        candidate.index_scale,
        candidate.middle_scale,
        candidate.ring_scale,
        0.0,
        candidate.thumb_scale,
    )
    for finger_index, finger_scale in enumerate(scales):
        start = 4 * finger_index
        target[start : start + 4] = base[start : start + 4] + finger_scale * (
            p1_target[start : start + 4] - base[start : start + 4]
        )
    target = np.clip(
        m316.enforce_shared_motor_consistency(target),
        env.joint_low,
        env.joint_high,
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    for step in range(steps):
        alpha = min(1.0, (step + 1) / max(1, candidate.ramp))
        effective = base + alpha * (target - base)
        action = np.clip(
            candidate.gain * (effective - env.target_qpos.astype(np.float64)),
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
        + 650.0 * ratios["ring"]
        + 350.0 * ratios["middle"]
        + 650.0 * summary["four_finger_contact_ratio"]
        + 300.0 * summary["p1_support_ratio"]
        - 900.0 * (1.0 - min(ratios["thumb"], ratios["index"]))
    )
    summary["p1_rejoin_score"] = score
    return bool(summary["accepted"]), score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "canonical_p1_rejoin_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=64)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=151)
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
        sources, source_trace, p1_target = build_tail_sources(
            env, route, references, cache, cache_index, args
        )
        source_steps = sorted(sources)
        centers = [
            np.asarray([0.12, 0.20, 0.08, 0.25]),
            np.asarray([0.25, 0.35, 0.12, 0.40]),
            np.asarray([0.05, 0.10, 0.03, 0.15]),
        ]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, args.samples // max(1, args.generations))
        for generation in range(args.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                candidate = random_candidate(
                    rng, centers, source_steps, scale
                )
                rows, _ = run_rejoin(
                    env,
                    sources[candidate.source_step],
                    p1_target,
                    candidate,
                    steps=args.steps,
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
                    row["summary"]["p1_rejoin_score"],
                ),
                reverse=True,
            )
            leaderboard.extend(generation_rows[: max(args.elite, 100)])
            centers = [
                np.asarray(
                    [
                        row["candidate"]["index_scale"],
                        row["candidate"]["middle_scale"],
                        row["candidate"]["ring_scale"],
                        row["candidate"]["thumb_scale"],
                    ]
                )
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
                row["summary"]["p1_rejoin_score"],
            ),
            reverse=True,
        )
        best = leaderboard[0]
        best_candidate = P1RejoinCandidate(**best["candidate"])
        _, final_state = run_rejoin(
            env,
            sources[best_candidate.source_step],
            p1_target,
            best_candidate,
            steps=args.steps,
        )
        m334.save_final_cache(output_dir / "canonical_P1_rejoin_arrival.npz", final_state)
        report = {
            "stage": "M3.2.35 canonical P1 partial rejoin from P45",
            "tail_trajectory": asdict(SELECTED_TAIL),
            "source_steps": source_steps,
            "p45_rows": source_trace["p45_rows"],
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "steps": args.steps,
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "canonical_p1_rejoin_search.json", report)
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
