from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import search_m3_2_35_cycle_closure as cycle
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_35_early_p51_closure as early


@dataclass(frozen=True)
class SustainedRingCandidate:
    p34_ring_target: tuple[float, float, float, float]
    p34_ring_ramp: int
    p34_ring_gain: float
    p34_ring_action_limit: float
    p45_ring_target: tuple[float, float, float, float]
    p45_ring_ramp: int
    p45_ring_gain: float
    p45_ring_action_limit: float


def random_candidate(
    rng: np.random.Generator,
    p34_centers: list[np.ndarray],
    p45_centers: list[np.ndarray],
    scale: float,
) -> SustainedRingCandidate:
    p34 = p34_centers[int(rng.integers(0, len(p34_centers)))] + rng.normal(
        0.0,
        scale * np.asarray([0.025, 0.070, 0.070, 0.070]),
    )
    p45 = p45_centers[int(rng.integers(0, len(p45_centers)))] + rng.normal(
        0.0,
        scale * np.asarray([0.070, 0.220, 0.220, 0.220]),
    )
    p34[3] = p34[2]
    p45[3] = p45[2]
    return SustainedRingCandidate(
        p34_ring_target=tuple(p34.astype(float)),
        p34_ring_ramp=int(rng.integers(2, 8)),
        p34_ring_gain=float(rng.uniform(0.65, 1.00)),
        p34_ring_action_limit=float(rng.uniform(0.006, 0.022)),
        p45_ring_target=tuple(p45.astype(float)),
        p45_ring_ramp=int(rng.integers(1, 8)),
        p45_ring_gain=float(rng.uniform(0.45, 1.00)),
        p45_ring_action_limit=float(rng.uniform(0.004, 0.030)),
    )


def longest_run(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def summarize(
    p34_rows: list[dict[str, Any]],
    p45_rows: list[dict[str, Any]],
    p45_snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    ring_contacts = ["ring" in row["contacts"].split(",") for row in p45_rows]
    ring_near = [row["ring_tip_distance_mm"] <= 8.75 for row in p45_rows]
    four_contacts = [
        {"thumb", "index", "middle", "ring"}.issubset(
            set(row["contacts"].split(","))
        )
        for row in p45_rows
    ]
    min_index = min(
        range(len(p45_rows)),
        key=lambda index: p45_rows[index]["ring_tip_distance_mm"],
    )
    min_row = p45_rows[min_index]
    min_state = p45_snapshots[min_index]

    def ratio(rows: list[dict[str, Any]], finger: str) -> float:
        return float(
            sum(finger in row["contacts"].split(",") for row in rows)
            / max(1, len(rows))
        )

    result = {
        "ring_contact_ratio": float(sum(ring_contacts) / len(ring_contacts)),
        "ring_near_ratio": float(sum(ring_near) / len(ring_near)),
        "four_finger_contact_ratio": float(sum(four_contacts) / len(four_contacts)),
        "ring_longest_contact_steps": longest_run(ring_contacts),
        "ring_longest_near_steps": longest_run(ring_near),
        "min_ring_step": min_index + 1,
        "min_ring_distance_mm": float(min_row["ring_tip_distance_mm"]),
        "middle_distance_at_min_mm": float(min_row["middle_tip_distance_mm"]),
        "ring_joint_speed_at_min_rad_s": float(
            np.linalg.norm(np.asarray(min_state["hand_qvel"])[8:12])
        ),
        "contacts_at_min": min_row["contacts"],
        "ring_share_at_min": float(min_row["ring_share"]),
        "thumb_share_at_min": float(min_row["thumb_share"]),
        "index_share_at_min": float(min_row["index_share"]),
        "pen_linear_speed_at_min": float(min_row["pen_linear_speed"]),
        "pen_angular_speed_at_min": float(min_row["pen_angular_speed"]),
        "dz_at_min_mm": float(min_row["dz_mm"]),
        "max_penetration_mm": float(
            max(row["max_penetration_mm"] for row in p34_rows + p45_rows)
        ),
        "p34_support_ratio": min(
            ratio(p34_rows, "thumb"),
            ratio(p34_rows, "index"),
            ratio(p34_rows, "middle"),
        ),
        "p45_thumb_ratio": ratio(p45_rows, "thumb"),
        "p45_index_ratio": ratio(p45_rows, "index"),
        "p45_middle_ratio": ratio(p45_rows, "middle"),
    }
    result["accepted"] = bool(
        result["ring_contact_ratio"] >= 0.25
        and result["ring_longest_contact_steps"] >= 2
        and result["ring_near_ratio"] >= 0.40
        and result["p34_support_ratio"] >= 0.85
        and result["p45_thumb_ratio"] >= 0.90
        and result["p45_index_ratio"] >= 0.90
        and result["max_penetration_mm"] <= 4.0
        and result["dz_at_min_mm"] >= -4.0
        and result["ring_share_at_min"] <= 0.20
    )
    result["score"] = float(
        1200.0 * result["accepted"]
        + 900.0 * result["ring_contact_ratio"]
        + 500.0 * result["ring_near_ratio"]
        + 650.0 * result["four_finger_contact_ratio"]
        + 60.0 * result["ring_longest_contact_steps"]
        + 35.0 * result["ring_longest_near_steps"]
        + 250.0 * min(result["p45_thumb_ratio"], result["p45_index_ratio"])
        + 120.0 * result["p34_support_ratio"]
        + 35.0 * max(0.0, 12.0 - result["middle_distance_at_min_mm"])
        - 45.0 * result["ring_joint_speed_at_min_rad_s"]
        - 300.0 * max(0.0, result["ring_share_at_min"] - 0.20)
        - 35.0 * max(0.0, result["max_penetration_mm"] - 1.5)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "sustained_ring_entry_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=48)
    parser.add_argument("--p45-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=121)
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
        p3_base = np.asarray(p3_state["target_qpos"], dtype=np.float64)
        seed_target = np.asarray(
            [0.20239661083815624, 1.256406545601292, 0.1742256922928092, 0.1742256922928092]
        )
        p34_centers = [seed_target.copy()]
        p45_centers = [
            seed_target.copy(),
            np.asarray([0.20, 1.15, 0.25, 0.25]),
            np.asarray([0.16, 1.35, 0.10, 0.10]),
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
                tail.restore_snapshot(env, p3_state)
                p34_rows, _ = early.run_mixed_target(
                    env,
                    p3_base,
                    tail.P34_DELTA,
                    steps=early.P34_STEPS,
                    delay=tail.P34_DELAY,
                    ramp=tail.P34_RAMP,
                    gain=tail.P34_GAIN,
                    action_limit=tail.P34_ACTION_LIMIT,
                    ring_target=np.asarray(candidate.p34_ring_target),
                    ring_ramp=candidate.p34_ring_ramp,
                    ring_gain=candidate.p34_ring_gain,
                    ring_action_limit=candidate.p34_ring_action_limit,
                )
                p4_base = env.target_qpos.astype(np.float64).copy()
                p45_rows, p45_snapshots = early.run_mixed_target(
                    env,
                    p4_base,
                    cycle.P45_DELTA,
                    steps=args.p45_steps,
                    delay=cycle.P45_DELAY,
                    ramp=cycle.P45_RAMP,
                    gain=cycle.P45_GAIN,
                    action_limit=cycle.P45_ACTION_LIMIT,
                    ring_target=np.asarray(candidate.p45_ring_target),
                    ring_ramp=candidate.p45_ring_ramp,
                    ring_gain=candidate.p45_ring_gain,
                    ring_action_limit=candidate.p45_ring_action_limit,
                )
                summary = summarize(p34_rows, p45_rows, p45_snapshots)
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
                np.asarray(row["candidate"]["p34_ring_target"])
                for row in generation_rows[: args.elite]
            ]
            p45_centers = [
                np.asarray(row["candidate"]["p45_ring_target"])
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
            "stage": "M3.2.35 sustained ring entry with split P34/P45 targets",
            "source": {"prepared_b23": asdict(early.PREPARED_P3)},
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "p45_steps": args.p45_steps,
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "sustained_ring_entry_search.json", report)
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
