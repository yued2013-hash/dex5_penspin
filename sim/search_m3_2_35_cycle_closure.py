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
import search_m3_2_35_dynamic_tail as tail


P45_DELTA = np.asarray(
    [
        0.019455987167825893,
        -0.14889681482680955,
        0.2526795756757788,
        0.2526795756757788,
        -0.08028108253447672,
        0.10549046871902794,
        -0.03164519599322385,
        -0.03164519599322385,
        0.012701676315290728,
        0.08344806867414233,
        -0.005986040658698144,
        -0.005986040658698144,
        0.0,
        0.0,
        0.0,
        0.0,
        0.019707898098491168,
        -0.08904887560654379,
        -0.15149985982263497,
        0.08898939197984812,
    ],
    dtype=np.float64,
)
P45_DELAY = 2
P45_RAMP = 3
P45_GAIN = 0.41657353633098015
P45_ACTION_LIMIT = 0.004045236225836265
P45_STEPS = 30


def load_cache_state(path: Path) -> dict[str, Any]:
    cache = np.load(path)
    state = {key: cache[key][0] for key in cache.files}
    state["contacts"] = str(state["contacts"])
    return state


def build_dynamic_p5(
    env: Any, p4_state: dict[str, Any], *, p45_steps: int
) -> dict[str, Any]:
    tail.restore_snapshot(env, p4_state)
    base_target = env.target_qpos.astype(np.float64).copy()
    rows, _ = tail.run_target(
        env,
        base_target,
        P45_DELTA,
        steps=p45_steps,
        delay=P45_DELAY,
        ramp=P45_RAMP,
        gain=P45_GAIN,
        action_limit=P45_ACTION_LIMIT,
    )
    return {
        "p45_summary": tail.summarize(rows, p45_steps),
        "p5_state": tail.state_snapshot(env),
    }


def middle_approach_rank(summary: dict[str, Any]) -> tuple[bool, float]:
    final_distance = float(summary["tip_distance_final_mm"]["middle"])
    min_distance = float(summary["tip_distance_min_mm"]["middle"])
    middle_contact = "middle" in summary["final_contacts"]
    base_support = bool(summary["accepted"])
    approach_passed = bool(base_support and (middle_contact or final_distance < 14.0))
    approach_score = float(
        summary["score"]
        + 12.0 * max(0.0, 30.0 - final_distance)
        + 4.0 * max(0.0, 20.0 - min_distance)
        + 160.0 * middle_contact
    )
    summary["middle_approach_passed"] = approach_passed
    summary["middle_approach_score"] = approach_score
    return approach_passed, approach_score


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
            "cycle_closure_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=32)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--p45-exit-step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=51)
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
        p4_state = load_cache_state(Path(parsed.p4_cache))
        source = build_dynamic_p5(
            env, p4_state, p45_steps=parsed.p45_exit_step
        )
        p5_state = source["p5_state"]
        base_target = np.asarray(p5_state["target_qpos"], dtype=np.float64)
        centers = [np.zeros(20, dtype=np.float64)]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, parsed.samples // max(1, parsed.generations))
        for generation in range(parsed.generations):
            scale = 0.85 if generation == 0 else 0.48 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample_index in range(per_generation):
                center = centers[sample_index % len(centers)]
                candidate = tail.candidate_from_rng(rng, center, scale)
                # P5 has only a two-step passive window, so delayed actions are
                # invalid even if they score well by chance.
                candidate = tail.ShootingCandidate(
                    delta=candidate.delta,
                    delay=0,
                    ramp=min(candidate.ramp, 4),
                    gain=candidate.gain,
                    action_limit=candidate.action_limit,
                )
                tail.restore_snapshot(env, p5_state)
                rows, _ = tail.run_target(
                    env,
                    base_target,
                    np.asarray(candidate.delta, dtype=np.float64),
                    steps=parsed.steps,
                    delay=candidate.delay,
                    ramp=candidate.ramp,
                    gain=candidate.gain,
                    action_limit=candidate.action_limit,
                )
                summary = tail.summarize(rows, parsed.steps)
                middle_approach_rank(summary)
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
                    row["summary"]["middle_approach_passed"],
                    row["summary"]["middle_approach_score"],
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
                        "accepted": sum(
                            row["summary"]["accepted"] for row in generation_rows
                        ),
                        "middle_approach_passed": sum(
                            row["summary"]["middle_approach_passed"]
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
                row["summary"]["middle_approach_passed"],
                row["summary"]["middle_approach_score"],
            ),
            reverse=True,
        )
        report = {
            "stage": "M3.2.35 P5 index/thumb velocity recapture search",
            "source": {
                "p4_cache": str(parsed.p4_cache),
                "p45_summary": source["p45_summary"],
                "p45_delta": P45_DELTA.astype(float).tolist(),
            },
            "search": {
                "samples_requested": parsed.samples,
                "generations": parsed.generations,
                "samples_per_generation": per_generation,
                "steps": parsed.steps,
                "p45_exit_step": parsed.p45_exit_step,
            },
            "leaderboard": leaderboard[:100],
        }
        tail.save_json(output_dir / "p51_index_thumb_recapture_search.json", report)
        m334.save_final_cache(output_dir / "dynamic_P5_entry.npz", p5_state)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "p45": source["p45_summary"],
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
