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
import search_m3_2_36_p45_middle_sync as middle
import search_m3_2_36_semantic_p45b as semantic


MIDDLE_TOUCH = middle.MiddleCandidate(
    delta=(0.16570108793867613, 0.45040004273113066, 0.5557782864577221),
    delay=0,
    ramp=3,
    gain=0.9042277107240579,
    action_limit=0.041577770611045076,
)
MIDDLE_TOUCH_STEPS = 3


@dataclass(frozen=True)
class Candidate:
    delta: tuple[float, ...]
    ramp: int
    gain: float
    action_limit: float


PAIR_ROLL_1 = Candidate(
    delta=(
        0.05304919938017521,
        0.004607062274302444,
        0.984362673325562,
        0.984362673325562,
        -0.2954830869903853,
        0.12474512542172045,
        -1.1069791114340295,
        -1.1069791114340295,
        0.02023444986935167,
        0.001450217324022854,
        -0.22378210984756655,
        -0.22378210984756655,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.010793769905809436,
        0.2901576907661766,
        -0.1572209271607528,
        0.17743064056826036,
    ),
    ramp=4,
    gain=1.0236671054806408,
    action_limit=0.008097031468128197,
)
PAIR_ROLL_1_STEPS = 8


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_source(
    env: Any, args: argparse.Namespace, root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = semantic.build_source(env, args, root, p45a_exit_step=0)
    rows = middle.run_candidate(
        env, source["p4_state"], MIDDLE_TOUCH, MIDDLE_TOUCH_STEPS
    )
    return tail.state_snapshot(env), {
        "_p4_base": np.asarray(
            source["p4_state"]["target_qpos"], dtype=np.float64
        ).tolist(),
        "p3": semantic.summarize(
            source["p3_rows"], semantic.P3_RECAPTURE_STEPS
        ),
        "p34": semantic.summarize(
            source["p34_rows"],
            semantic.P34_DRIVE_STEPS + semantic.P34_SETTLE_STEPS,
        ),
        "middle_touch": middle.summarize(rows, MIDDLE_TOUCH_STEPS),
    }


def run_candidate(
    env: Any,
    source_state: dict[str, Any],
    candidate: Candidate,
    steps: int,
    *,
    reset_state: bool = True,
) -> list[dict[str, Any]]:
    if reset_state:
        tail.restore_snapshot(env, source_state)
    base = np.asarray(source_state["target_qpos"], dtype=np.float64)
    target = np.clip(
        m316.enforce_shared_motor_consistency(
            base + np.asarray(candidate.delta, dtype=np.float64)
        ),
        env.joint_low,
        env.joint_high,
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    for step in range(steps):
        alpha = semantic.smoothstep(
            min(1.0, (step + 1) / max(1, candidate.ramp))
        )
        env.step(
            semantic.action_to_target(
                env,
                base + alpha * (target - base),
                candidate.gain,
                candidate.action_limit,
            )
        )
        row = semantic.metric_row(
            env, initial.pen_center, initial.pen_axis, step + 1
        )
        rows.append(row)
        if row["dz_mm"] < -12.0:
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
    rotation = float(rows[-1]["planar_rotation_deg"])
    speed = float(rows[-1]["pen_linear_speed"])
    angular = float(rows[-1]["pen_angular_speed"])
    min_dz = float(min(row["dz_mm"] for row in rows))
    max_pen = float(max(row["max_penetration_mm"] for row in rows))
    final_pair = bool(pair[-1])
    pair_tail = sum(pair[-3:])
    score = (
        8.0 * len(rows)
        + 5.0 * min(65.0, max(0.0, rotation))
        + 150.0 * ratios["thumb"]
        + 135.0 * ratios["index"]
        + 45.0 * min(0.5, ratios["middle"])
        + 180.0 * float(final_pair)
        + 55.0 * pair_tail
        + 35.0 * longest_run(pair)
        - 30.0 * ratios["ring"]
        - 60.0 * ratios["pinky"]
        - 40.0 * max(0.0, -min_dz - 3.0)
        - 30.0 * max(0.0, max_pen - 4.0)
        - 330.0 * max(0.0, speed - 0.16)
        - 14.0 * max(0.0, angular - 16.0)
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
        "pair_longest_run": longest_run(pair),
        "final_contacts": sorted(contacts[-1]),
        "score": float(score),
        "accepted": bool(
            len(rows) == expected_steps
            and rotation >= 20.0
            and min_dz >= -5.0
            and max_pen <= 4.0
            and ratios["thumb"] >= 0.70
            and ratios["index"] >= 0.70
            and final_pair
            and pair_tail >= 2
            and speed <= 0.20
            and angular <= 20.0
        ),
    }


def random_delta(
    rng: np.random.Generator,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    std = np.zeros(20, dtype=np.float64)
    std[0:4] = [0.045, 0.16, 0.24, 0.24]
    std[4:8] = [0.08, 0.24, 0.34, 0.34]
    std[8:12] = [0.025, 0.08, 0.14, 0.14]
    std[16:20] = [0.07, 0.16, 0.23, 0.20]
    delta = center + rng.normal(0.0, scale * std)
    delta[3] = delta[2]
    delta[7] = delta[6]
    delta[11] = delta[10]
    delta[12:16] = 0.0
    return m316.enforce_shared_motor_consistency(delta)


def sample_candidate(
    rng: np.random.Generator,
    center: np.ndarray,
    scale: float,
) -> Candidate:
    return Candidate(
        delta=tuple(random_delta(rng, center, scale).astype(float)),
        ramp=int(rng.integers(1, 8)),
        gain=float(rng.uniform(0.35, 1.05)),
        action_limit=float(rng.uniform(0.008, 0.040)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "semantic_p45_after_middle_touch_search_m3_2_36"
        ),
    )
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=48)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=453)
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
        source_state, source_summary = build_source(env, args, root)
        base = np.asarray(source_state["target_qpos"], dtype=np.float64)
        p4_base = np.asarray(source_summary.pop("_p4_base"), dtype=np.float64)
        continuation = m316.enforce_shared_motor_consistency(
            p4_base + semantic.P45A_DELTA - base
        )
        centers = [continuation]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, parsed.samples // max(1, parsed.generations))
        for generation in range(parsed.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                candidate = sample_candidate(
                    rng, centers[sample % len(centers)], scale
                )
                rows = run_candidate(env, source_state, candidate, parsed.steps)
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
            "stage": "M3.2.36 P45 after middle-touch push and pair recapture",
            "source": source_summary,
            "search": {
                "samples": parsed.samples,
                "generations": parsed.generations,
                "steps": parsed.steps,
            },
            "leaderboard": leaderboard[:150],
        }
        save_json(output_dir / "after_middle_touch_search.json", report)
        m334.save_final_cache(output_dir / "middle_touch_source.npz", source_state)
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
