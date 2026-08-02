from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import build_m3_2_34_continuous_p1_p2_p3 as m334
import export_m3_2_23_dynamic_state_cache as m323
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_36_semantic_p45b as semantic


@dataclass(frozen=True)
class MiddleCandidate:
    delta: tuple[float, float, float]
    delay: int
    ramp: int
    gain: float
    action_limit: float


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_candidate(
    env: Any,
    p4_state: dict[str, Any],
    candidate: MiddleCandidate,
    steps: int,
) -> list[dict[str, Any]]:
    tail.restore_snapshot(env, p4_state)
    base = np.asarray(p4_state["target_qpos"], dtype=np.float64)
    main_target = np.clip(
        m316.enforce_shared_motor_consistency(base + semantic.P45A_DELTA),
        env.joint_low,
        env.joint_high,
    )
    middle_delta = np.asarray(
        [candidate.delta[0], candidate.delta[1], candidate.delta[2], candidate.delta[2]],
        dtype=np.float64,
    )
    middle_target = np.clip(
        base[4:8] + middle_delta,
        env.joint_low[4:8],
        env.joint_high[4:8],
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    for step in range(steps):
        main_alpha = semantic.smoothstep(
            min(1.0, (step + 1) / semantic.P45A_RAMP)
        )
        action = semantic.action_to_target(
            env,
            base + main_alpha * (main_target - base),
            semantic.P45A_GAIN,
            semantic.P45A_LIMIT,
        )
        middle_alpha = 0.0 if step < candidate.delay else semantic.smoothstep(
            min(1.0, (step - candidate.delay + 1) / max(1, candidate.ramp))
        )
        middle_effective = base[4:8] + middle_alpha * (
            middle_target - base[4:8]
        )
        action[4:8] = np.clip(
            candidate.gain
            * (middle_effective - env.target_qpos[4:8].astype(np.float64)),
            -candidate.action_limit,
            candidate.action_limit,
        )
        env.step(action)
        row = semantic.metric_row(
            env, initial.pen_center, initial.pen_axis, step + 1
        )
        rows.append(row)
        if row["dz_mm"] < -15.0:
            break
    return rows


def longest_run(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def summarize(rows: list[dict[str, Any]], expected_steps: int) -> dict[str, Any]:
    contacts = [set(row["contacts"].split(",")) - {"none"} for row in rows]
    ratios = {
        finger: float(sum(finger in c for c in contacts) / max(1, len(rows)))
        for finger in ("thumb", "index", "middle", "ring", "pinky")
    }
    pair = [{"thumb", "index"}.issubset(c) for c in contacts]
    triad = [{"thumb", "index", "middle"}.issubset(c) for c in contacts]
    rotation = float(rows[-1]["planar_rotation_deg"])
    speed = float(rows[-1]["pen_linear_speed"])
    angular = float(rows[-1]["pen_angular_speed"])
    min_dz = float(min(row["dz_mm"] for row in rows))
    max_pen = float(max(row["max_penetration_mm"] for row in rows))
    final_pair = bool(pair[-1])
    pair_tail = sum(pair[-3:])
    score = (
        6.0 * len(rows)
        + 4.0 * min(90.0, max(0.0, rotation))
        + 130.0 * ratios["thumb"]
        + 115.0 * ratios["index"]
        + 260.0 * ratios["middle"]
        + 80.0 * float(final_pair)
        + 30.0 * pair_tail
        + 120.0 * longest_run(triad)
        - 24.0 * ratios["ring"]
        - 40.0 * ratios["pinky"]
        - 30.0 * max(0.0, -min_dz - 3.0)
        - 25.0 * max(0.0, max_pen - 4.0)
        - 240.0 * max(0.0, speed - 0.16)
        - 10.0 * max(0.0, angular - 15.0)
    )
    return {
        "steps": len(rows),
        "rotation_deg": rotation,
        "min_dz_mm": min_dz,
        "max_penetration_mm": max_pen,
        "final_linear_speed": speed,
        "final_angular_speed": angular,
        "contact_ratio": ratios,
        "pair_ratio": float(sum(pair) / max(1, len(pair))),
        "pair_tail_steps": int(pair_tail),
        "triad_longest_run": longest_run(triad),
        "final_contacts": sorted(contacts[-1]),
        "score": float(score),
        "accepted": bool(
            len(rows) == expected_steps
            and rotation >= 40.0
            and min_dz >= -6.0
            and max_pen <= 4.0
            and ratios["thumb"] >= 0.70
            and ratios["index"] >= 0.60
            and ratios["middle"] >= 0.15
            and final_pair
            and pair_tail >= 2
            and speed <= 0.20
        ),
    }


def sample_candidate(
    rng: np.random.Generator,
    center: MiddleCandidate | None,
    scale: float,
) -> MiddleCandidate:
    if center is None:
        delta = np.asarray(
            [
                rng.uniform(0.12, 0.46),
                rng.uniform(0.12, 0.62),
                rng.uniform(0.24, 0.82),
            ],
            dtype=np.float64,
        )
        return MiddleCandidate(
            delta=tuple(delta.astype(float)),
            delay=int(rng.integers(0, 4)),
            ramp=int(rng.integers(1, 5)),
            gain=float(rng.uniform(0.55, 1.20)),
            action_limit=float(rng.uniform(0.018, 0.050)),
        )
    delta = np.asarray(center.delta) + rng.normal(
        0.0, scale * np.asarray([0.08, 0.18, 0.24])
    )
    return MiddleCandidate(
        delta=tuple(delta.astype(float)),
        delay=int(np.clip(center.delay + rng.integers(-2, 3), 0, 10)),
        ramp=int(np.clip(center.ramp + rng.integers(-3, 4), 1, 16)),
        gain=float(np.clip(center.gain + rng.normal(0.0, scale * 0.18), 0.18, 1.1)),
        action_limit=float(
            np.clip(
                center.action_limit + rng.normal(0.0, scale * 0.008),
                0.002,
                0.040,
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "semantic_p45_middle_sync_search_m3_2_36"
        ),
    )
    parser.add_argument("--samples", type=int, default=18000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=48)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=245)
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
        source = semantic.build_source(
            env, args, root, p45a_exit_step=0
        )
        p4_state = source["p4_state"]
        centers: list[MiddleCandidate | None] = [None]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, parsed.samples // max(1, parsed.generations))
        for generation in range(parsed.generations):
            generation_rows: list[dict[str, Any]] = []
            scale = 0.65 / max(1, generation)
            for sample in range(per_generation):
                candidate = sample_candidate(
                    rng, centers[sample % len(centers)], scale
                )
                rows = run_candidate(env, p4_state, candidate, parsed.steps)
                summary = summarize(rows, parsed.steps)
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
            middle_rows = sorted(
                generation_rows,
                key=lambda item: (
                    item["summary"]["contact_ratio"]["middle"],
                    item["summary"]["triad_longest_run"],
                    item["summary"]["score"],
                ),
                reverse=True,
            )
            leaderboard.extend(generation_rows[: max(120, parsed.elite)])
            leaderboard.extend(middle_rows[: max(60, parsed.elite // 2)])
            half = max(1, parsed.elite // 2)
            center_rows = generation_rows[:half] + middle_rows[:half]
            centers = [
                MiddleCandidate(**item["candidate"])
                for item in center_rows
            ]
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "accepted": sum(
                            item["summary"]["accepted"] for item in generation_rows
                        ),
                        "best": generation_rows[0],
                        "best_middle": middle_rows[0],
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
            "stage": "M3.2.36 P4-to-P5 middle-synchronized transition search",
            "search": {
                "samples": parsed.samples,
                "generations": parsed.generations,
                "steps": parsed.steps,
            },
            "leaderboard": leaderboard[:150],
        }
        save_json(output_dir / "p45_middle_sync_search.json", report)
        m334.save_final_cache(output_dir / "semantic_P4_source.npz", p4_state)
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
