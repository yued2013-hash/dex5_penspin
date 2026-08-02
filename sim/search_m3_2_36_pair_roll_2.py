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
import search_m3_2_36_after_middle_touch as previous
import search_m3_2_36_semantic_p45b as semantic


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "semantic_p45_pair_roll_2_search_m3_2_36"
        ),
    )
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=48)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--roll1-exit-step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=454)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parsed = parser.parse_args()
    args = semantic.make_args(parsed)
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path("dex5_penspin/reports/manus_mujoco_presets")
    route = m323.build_route(root / "photo_five_phase_hand_pen_tuned_m3_2_10")
    initial = m334.apply_p1_safety_open(
        route,
        {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0},
    )[0]
    env = m316.make_env(initial, m334.model_args(args), duration=4.0, render=False)
    rng = np.random.default_rng(parsed.seed)
    try:
        middle_touch_state, source_summary = previous.build_source(env, args, root)
        roll_1_rows = previous.run_candidate(
            env,
            middle_touch_state,
            previous.PAIR_ROLL_1,
            parsed.roll1_exit_step,
        )
        roll_1_state = tail.state_snapshot(env)
        source_summary["pair_roll_1"] = previous.summarize(
            roll_1_rows, parsed.roll1_exit_step
        )
        centers = [
            np.zeros(20, dtype=np.float64),
            0.35 * np.asarray(previous.PAIR_ROLL_1.delta, dtype=np.float64),
        ]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, parsed.samples // max(1, parsed.generations))
        for generation in range(parsed.generations):
            scale = 0.85 if generation == 0 else 0.50 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                candidate = previous.sample_candidate(
                    rng, centers[sample % len(centers)], scale
                )
                previous.run_candidate(
                    env,
                    middle_touch_state,
                    previous.PAIR_ROLL_1,
                    parsed.roll1_exit_step,
                )
                continuous_roll_1_state = tail.state_snapshot(env)
                rows = previous.run_candidate(
                    env,
                    continuous_roll_1_state,
                    candidate,
                    parsed.steps,
                    reset_state=False,
                )
                summary = previous.summarize(rows, parsed.steps)
                generation_rows.append(
                    {
                        "generation": generation,
                        "sample": sample,
                        "candidate": asdict(candidate),
                        "summary": summary,
                    }
                )
            generation_rows.sort(
                key=lambda item: (item["summary"]["accepted"], item["summary"]["score"]),
                reverse=True,
            )
            leaderboard.extend(generation_rows[: max(120, parsed.elite)])
            centers = [
                np.asarray(item["candidate"]["delta"], dtype=np.float64)
                for item in generation_rows[: parsed.elite]
            ]
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "accepted": sum(
                            item["summary"]["accepted"] for item in generation_rows
                        ),
                        "best": generation_rows[0],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        leaderboard.sort(
            key=lambda item: (item["summary"]["accepted"], item["summary"]["score"]),
            reverse=True,
        )
        report = {
            "stage": "M3.2.36 semantic P45 thumb-index pair roll leg 2",
            "source": source_summary,
            "search": {
                "samples": parsed.samples,
                "generations": parsed.generations,
                "steps": parsed.steps,
                "roll1_exit_step": parsed.roll1_exit_step,
            },
            "leaderboard": leaderboard[:150],
        }
        save_json(output_dir / "pair_roll_2_search.json", report)
        m334.save_final_cache(output_dir / "pair_roll_1_source.npz", roll_1_state)
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
