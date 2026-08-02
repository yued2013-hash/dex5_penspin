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


@dataclass(frozen=True)
class RingTrajectoryCandidate:
    ring_target: tuple[float, float, float, float]
    ring_ramp: int
    ring_gain: float
    ring_action_limit: float


def random_candidate(
    rng: np.random.Generator,
    centers: list[np.ndarray],
    scale: float,
) -> RingTrajectoryCandidate:
    center = centers[int(rng.integers(0, len(centers)))]
    target = center + rng.normal(
        0.0,
        scale * np.asarray([0.045, 0.130, 0.130, 0.130], dtype=np.float64),
    )
    target[3] = target[2]
    return RingTrajectoryCandidate(
        ring_target=tuple(target.astype(float)),
        ring_ramp=int(rng.integers(2, 11)),
        ring_gain=float(rng.uniform(0.40, 1.00)),
        ring_action_limit=float(rng.uniform(0.004, 0.024)),
    )


def summarize_candidate(
    p34_rows: list[dict[str, Any]],
    p45_rows: list[dict[str, Any]],
    p45_snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    min_index = min(
        range(len(p45_rows)),
        key=lambda index: p45_rows[index]["ring_tip_distance_mm"],
    )
    row = p45_rows[min_index]
    state = p45_snapshots[min_index]
    ring_velocity = float(np.linalg.norm(np.asarray(state["hand_qvel"])[8:12]))

    def ratio(rows: list[dict[str, Any]], finger: str) -> float:
        return float(
            sum(finger in item["contacts"].split(",") for item in rows)
            / max(1, len(rows))
        )

    all_rows = p34_rows + p45_rows
    result = {
        "p45_min_ring_step": min_index + 1,
        "p45_min_ring_distance_mm": float(row["ring_tip_distance_mm"]),
        "middle_distance_at_ring_min_mm": float(row["middle_tip_distance_mm"]),
        "ring_joint_speed_at_min_rad_s": ring_velocity,
        "contacts_at_ring_min": row["contacts"],
        "ring_share_at_min": float(row["ring_share"]),
        "thumb_share_at_min": float(row["thumb_share"]),
        "index_share_at_min": float(row["index_share"]),
        "pen_linear_speed_at_min": float(row["pen_linear_speed"]),
        "pen_angular_speed_at_min": float(row["pen_angular_speed"]),
        "dz_at_min_mm": float(row["dz_mm"]),
        "max_penetration_mm": float(
            max(item["max_penetration_mm"] for item in all_rows)
        ),
        "p34_contact_ratio": {
            finger: ratio(p34_rows, finger) for finger in ("thumb", "index", "middle")
        },
        "p45_contact_ratio": {
            finger: ratio(p45_rows, finger) for finger in ("thumb", "index", "middle", "ring")
        },
    }
    support_ok = {"thumb", "index"}.issubset(set(row["contacts"].split(",")))
    result["accepted"] = bool(
        result["p45_min_ring_distance_mm"] <= 8.50
        and result["middle_distance_at_ring_min_mm"] <= 9.20
        and ring_velocity <= 0.65
        and support_ok
        and result["p34_contact_ratio"]["thumb"] >= 0.90
        and result["p34_contact_ratio"]["index"] >= 0.90
        and result["p34_contact_ratio"]["middle"] >= 0.80
        and result["max_penetration_mm"] <= 4.0
        and result["dz_at_min_mm"] >= -4.0
        and result["ring_share_at_min"] <= 0.20
    )
    result["score"] = float(
        1000.0 * result["accepted"]
        + 320.0 * support_ok
        + 55.0 * max(0.0, 13.0 - result["p45_min_ring_distance_mm"])
        + 25.0 * max(0.0, 12.0 - result["middle_distance_at_ring_min_mm"])
        + 100.0 * min(result["p34_contact_ratio"].values())
        + 100.0 * min(
            result["p45_contact_ratio"]["thumb"],
            result["p45_contact_ratio"]["index"],
        )
        - 130.0 * ring_velocity
        - 250.0 * max(0.0, result["ring_share_at_min"] - 0.20)
        - 30.0 * max(0.0, result["max_penetration_mm"] - 1.5)
        - 30.0 * max(0.0, -result["dz_at_min_mm"] - 2.0)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "low_velocity_ring_entry_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=48)
    parser.add_argument("--p45-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=101)
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
        p3_state = source["p3_trace"]["snapshots"][-1]
        p3_base = np.asarray(p3_state["target_qpos"], dtype=np.float64)
        centers = [
            early.RING_PREPOSITION_TARGET.copy(),
            np.asarray([0.20, 1.34, 0.13, 0.13], dtype=np.float64),
            np.asarray([0.18, 1.42, 0.20, 0.20], dtype=np.float64),
        ]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, args.samples // max(1, args.generations))
        for generation in range(args.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                candidate = random_candidate(rng, centers, scale)
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
                    ring_target=np.asarray(candidate.ring_target, dtype=np.float64),
                    ring_ramp=candidate.ring_ramp,
                    ring_gain=candidate.ring_gain,
                    ring_action_limit=candidate.ring_action_limit,
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
                    ring_target=np.asarray(candidate.ring_target, dtype=np.float64),
                    ring_ramp=candidate.ring_ramp,
                    ring_gain=candidate.ring_gain,
                    ring_action_limit=candidate.ring_action_limit,
                )
                summary = summarize_candidate(p34_rows, p45_rows, p45_snapshots)
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
            centers = [
                np.asarray(row["candidate"]["ring_target"], dtype=np.float64)
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
            "stage": "M3.2.35 low-velocity ring entry search across P34/P45",
            "source": {
                "prepared_b23": asdict(early.PREPARED_P3),
                "p3_summary": source["p3_summary"],
            },
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "p34_steps": early.P34_STEPS,
                "p45_steps": args.p45_steps,
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "low_velocity_ring_entry_search.json", report)
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
