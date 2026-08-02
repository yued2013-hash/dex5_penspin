from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import build_m3_2_34_continuous_p1_p2_p3 as m334
import export_m3_2_23_dynamic_state_cache as m323
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_cycle_closure as cycle
import search_m3_2_35_dynamic_tail as tail


MIDDLE_APPROACH_DELTA = np.asarray(
    [
        0.05046375245198887,
        -0.14626016745684212,
        0.2590265729017307,
        0.2590265729017307,
        -0.06595594985370577,
        0.2484921407761461,
        0.09398336717951211,
        0.09398336717951211,
        -0.04513856553027458,
        -0.09107767566218136,
        -0.08813747149910597,
        -0.08813747149910597,
        0.0,
        0.0,
        0.0,
        0.0,
        0.006905851240833236,
        -0.04965968206873876,
        -0.1568682438075629,
        0.09780919397906873,
    ],
    dtype=np.float64,
)
MIDDLE_APPROACH_RAMP = 4
MIDDLE_APPROACH_GAIN = 0.7296462498260419
MIDDLE_APPROACH_ACTION_LIMIT = 0.01024795692102689
MIDDLE_APPROACH_STEPS = 10


def build_middle_approach_state(
    env: Any, p4_state: dict[str, Any], *, approach_steps: int
) -> dict[str, Any]:
    p5 = cycle.build_dynamic_p5(env, p4_state, p45_steps=20)
    base_target = env.target_qpos.astype(np.float64).copy()
    rows, _ = tail.run_target(
        env,
        base_target,
        MIDDLE_APPROACH_DELTA,
        steps=approach_steps,
        delay=0,
        ramp=MIDDLE_APPROACH_RAMP,
        gain=MIDDLE_APPROACH_GAIN,
        action_limit=MIDDLE_APPROACH_ACTION_LIMIT,
    )
    return {
        "p45_summary": p5["p45_summary"],
        "middle_approach_summary": tail.summarize(rows, approach_steps),
        "state": tail.state_snapshot(env),
    }


def contact_rank(summary: dict[str, Any]) -> tuple[bool, float]:
    final_distance = float(summary["tip_distance_final_mm"]["middle"])
    min_distance = float(summary["tip_distance_min_mm"]["middle"])
    final_contacts = set(summary["final_contacts"])
    middle_contact = "middle" in final_contacts
    thumb_index_final = {"thumb", "index"}.issubset(final_contacts)
    support_ratio = min(
        summary["contact_ratio"]["thumb"], summary["contact_ratio"]["index"]
    )
    passed = bool(
        summary["steps"] > 0
        and support_ratio >= 0.75
        and thumb_index_final
        and middle_contact
        and summary["min_dz_mm"] >= -5.0
        and summary["max_penetration_mm"] <= 4.0
    )
    score = float(
        summary["score"]
        + 24.0 * max(0.0, 18.0 - final_distance)
        + 8.0 * max(0.0, 12.0 - min_distance)
        + 280.0 * middle_contact
        + 80.0 * thumb_index_final
    )
    summary["middle_touch_passed"] = passed
    summary["middle_touch_score"] = score
    return passed, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p4-cache",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "dynamic_tail_search_m3_2_35_p34s16/dynamic_P4_entry.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p1_recapture_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=48)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--approach-exit-step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parsed = parser.parse_args()
    args = tail.make_args(parsed)
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path("dex5_penspin/reports/manus_mujoco_presets")
    route = m323.build_route(root / "photo_five_phase_hand_pen_tuned_m3_2_10")
    initial_seed = m334.apply_p1_safety_open(
        route,
        {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0},
    )[0]
    env = m316.make_env(initial_seed, m334.model_args(args), duration=4.0, render=False)
    rng = np.random.default_rng(parsed.seed)
    try:
        p4_state = cycle.load_cache_state(Path(parsed.p4_cache))
        source = build_middle_approach_state(
            env, p4_state, approach_steps=parsed.approach_exit_step
        )
        approach_state = source["state"]
        base_target = np.asarray(approach_state["target_qpos"], dtype=np.float64)
        centers = [np.zeros(20, dtype=np.float64)]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, parsed.samples // max(1, parsed.generations))
        for generation in range(parsed.generations):
            scale = 0.55 if generation == 0 else 0.32 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample_index in range(per_generation):
                center = centers[sample_index % len(centers)]
                candidate = tail.candidate_from_rng(rng, center, scale)
                candidate = tail.ShootingCandidate(
                    delta=candidate.delta,
                    delay=0,
                    ramp=min(candidate.ramp, 3),
                    gain=candidate.gain,
                    action_limit=candidate.action_limit,
                )
                tail.restore_snapshot(env, approach_state)
                rows, _ = tail.run_target(
                    env,
                    base_target,
                    np.asarray(candidate.delta, dtype=np.float64),
                    steps=parsed.steps,
                    delay=0,
                    ramp=candidate.ramp,
                    gain=candidate.gain,
                    action_limit=candidate.action_limit,
                )
                summary = tail.summarize(rows, parsed.steps)
                contact_rank(summary)
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
                    row["summary"]["middle_touch_passed"],
                    row["summary"]["middle_touch_score"],
                ),
                reverse=True,
            )
            leaderboard.extend(generation_rows[: max(parsed.elite, 64)])
            centers = [
                np.asarray(row["candidate"]["delta"], dtype=np.float64)
                for row in generation_rows[: parsed.elite]
            ]
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "middle_touch_passed": sum(
                            row["summary"]["middle_touch_passed"]
                            for row in generation_rows
                        ),
                        "best": generation_rows[0],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        leaderboard.sort(
            key=lambda row: (
                row["summary"]["middle_touch_passed"],
                row["summary"]["middle_touch_score"],
            ),
            reverse=True,
        )
        report = {
            "stage": "M3.2.35 middle touch after P5 index/thumb support",
            "source": {
                "p45_summary": source["p45_summary"],
                "middle_approach_summary": source["middle_approach_summary"],
                "middle_approach_delta": MIDDLE_APPROACH_DELTA.astype(float).tolist(),
            },
            "search": {
                "samples_requested": parsed.samples,
                "generations": parsed.generations,
                "steps": parsed.steps,
                "approach_exit_step": parsed.approach_exit_step,
            },
            "leaderboard": leaderboard[:100],
        }
        tail.save_json(output_dir / "middle_touch_search.json", report)
        m334.save_final_cache(output_dir / "middle_approach_entry.npz", approach_state)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "source": report["source"],
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
