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
import search_m3_2_35_p45_middle_ring_sync as sync


P1_TOUCH_MIDDLE = sync.MiddleSyncCandidate(
    middle_delta=(
        0.014528392272009354,
        -0.1286777744125365,
        0.054708834311048025,
        0.054708834311048025,
    ),
    middle_ramp=2,
    middle_gain=0.5645041110448126,
    middle_action_limit=0.012337402505614718,
)
P1_TOUCH_STEP = 5


@dataclass(frozen=True)
class TouchExtensionCandidate:
    delta: tuple[float, ...]
    ramp: int
    gain: float
    action_limit: float


def build_source(
    env: Any,
    route: list[dict[str, Any]],
    references: dict[str, np.ndarray],
    cache: Any,
    cache_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        ring_target=np.asarray(sync.SELECTED_RING.p34_ring_target),
        ring_ramp=sync.SELECTED_RING.p34_ring_ramp,
        ring_gain=sync.SELECTED_RING.p34_ring_gain,
        ring_action_limit=sync.SELECTED_RING.p34_ring_action_limit,
    )
    p45_rows, p45_snapshots = sync.run_p45(
        env,
        p34_snapshots[-1],
        P1_TOUCH_MIDDLE,
        steps=P1_TOUCH_STEP,
    )
    return p45_snapshots[-1], {
        "front": front,
        "p34_rows": p34_rows,
        "p34_snapshots": p34_snapshots,
        "p45_rows": p45_rows,
        "p45_snapshots": p45_snapshots,
    }


def random_candidate(
    rng: np.random.Generator,
    centers: list[np.ndarray],
    scale: float,
) -> TouchExtensionCandidate:
    center = centers[int(rng.integers(0, len(centers)))]
    std = np.zeros(20, dtype=np.float64)
    std[[0, 1, 2]] = [0.025, 0.060, 0.080]
    std[[4, 5, 6]] = [0.040, 0.100, 0.130]
    std[[8, 9, 10]] = [0.040, 0.100, 0.130]
    std[16:20] = [0.020, 0.050, 0.075, 0.070]
    delta = center + rng.normal(0.0, scale * std)
    delta[3] = delta[2]
    delta[7] = delta[6]
    delta[11] = delta[10]
    delta[12:16] = 0.0
    return TouchExtensionCandidate(
        delta=tuple(m316.enforce_shared_motor_consistency(delta).astype(float)),
        ramp=int(rng.integers(1, 4)),
        gain=float(rng.uniform(0.30, 0.90)),
        action_limit=float(rng.uniform(0.002, 0.018)),
    )


def run_extension(
    env: Any,
    source: dict[str, Any],
    candidate: TouchExtensionCandidate,
    *,
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tail.restore_snapshot(env, source)
    base = env.target_qpos.astype(np.float64).copy()
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contact_sets = [set(row["contacts"].split(",")) for row in rows]
    four = [
        {"thumb", "index", "middle", "ring"}.issubset(contacts)
        for contacts in contact_sets
    ]

    def ratio(finger: str) -> float:
        return float(sum(finger in contacts for contacts in contact_sets) / len(rows))

    shares = {
        finger: float(np.mean([row[f"{finger}_share"] for row in rows]))
        for finger in ("thumb", "index", "middle", "ring")
    }
    result = {
        "steps": len(rows),
        "four_finger_ratio": float(sum(four) / len(rows)),
        "four_finger_longest_steps": sync.sustained.longest_run(four),
        "contact_ratio": {
            finger: ratio(finger) for finger in ("thumb", "index", "middle", "ring")
        },
        "force_share_mean": shares,
        "final_contacts": sorted(contact_sets[-1]),
        "max_penetration_mm": float(
            max(row["max_penetration_mm"] for row in rows)
        ),
        "min_dz_mm": float(min(row["dz_mm"] for row in rows)),
        "final_linear_speed": float(rows[-1]["pen_linear_speed"]),
        "final_angular_speed": float(rows[-1]["pen_angular_speed"]),
        "tip_distance_final_mm": {
            finger: float(rows[-1][f"{finger}_tip_distance_mm"])
            for finger in ("thumb", "index", "middle", "ring")
        },
    }
    ratios = result["contact_ratio"]
    result["accepted"] = bool(
        result["four_finger_longest_steps"] >= 2
        and ratios["thumb"] >= 1.0
        and ratios["index"] >= 1.0
        and ratios["middle"] >= 2.0 / 3.0
        and ratios["ring"] >= 2.0 / 3.0
        and shares["thumb"] >= shares["ring"]
        and shares["ring"] <= 0.20
        and result["max_penetration_mm"] <= 4.0
        and result["min_dz_mm"] >= -2.0
        and result["final_linear_speed"] <= 0.10
    )
    result["score"] = float(
        1200.0 * result["accepted"]
        + 900.0 * result["four_finger_ratio"]
        + 150.0 * result["four_finger_longest_steps"]
        + 500.0 * min(ratios.values())
        + 250.0 * min(ratios["thumb"], ratios["index"])
        + 120.0 * (shares["thumb"] >= shares["ring"])
        - 500.0 * max(0.0, shares["ring"] - 0.20)
        - 800.0 * (1.0 - min(ratios["thumb"], ratios["index"]))
        - 500.0 * max(0.0, result["final_linear_speed"] - 0.10)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p1_touch_extension_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=161)
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
        source_state, source = build_source(
            env, route, references, cache, cache_index, args
        )
        centers = [np.zeros(20, dtype=np.float64)]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, args.samples // max(1, args.generations))
        for generation in range(args.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                candidate = random_candidate(rng, centers, scale)
                rows, _ = run_extension(
                    env, source_state, candidate, steps=args.steps
                )
                summary = summarize(rows)
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
                np.asarray(row["candidate"]["delta"])
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
        best = leaderboard[0]
        best_candidate = TouchExtensionCandidate(**best["candidate"])
        _, final_state = run_extension(
            env, source_state, best_candidate, steps=args.steps
        )
        m334.save_final_cache(output_dir / "P1_touch_extension_arrival.npz", final_state)
        report = {
            "stage": "M3.2.35 P1 touch extension after completed P45 drive",
            "ring_trajectory": asdict(sync.SELECTED_RING),
            "middle_sync": asdict(P1_TOUCH_MIDDLE),
            "source_p45_rows": source["p45_rows"],
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "steps": args.steps,
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "p1_touch_extension_search.json", report)
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
