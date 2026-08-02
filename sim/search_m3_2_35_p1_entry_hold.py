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
from evaluate_m3_2_15_short_primitives import FINGERS


@dataclass(frozen=True)
class HoldCandidate:
    entry_rank: int
    delta: tuple[float, ...]
    ramp: int
    gain: float
    action_limit: float


def load_entry_candidates(path: Path, count: int) -> list[early.ClosureCandidate]:
    report = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[early.ClosureCandidate] = []
    for row in report["leaderboard"]:
        if row["summary"]["four_finger_touch"]:
            candidates.append(early.ClosureCandidate(**row["candidate"]))
        if len(candidates) >= count:
            break
    if not candidates:
        raise RuntimeError("No four-finger entry candidate in the source report")
    return candidates


def random_hold_delta(
    rng: np.random.Generator, centers: list[np.ndarray], scale: float
) -> np.ndarray:
    center = centers[int(rng.integers(0, len(centers)))]
    std = np.zeros(20, dtype=np.float64)
    std[[0, 1, 2]] = [0.012, 0.025, 0.035]
    std[[4, 5, 6]] = [0.025, 0.060, 0.075]
    std[[8, 9, 10]] = [0.020, 0.055, 0.070]
    std[16:20] = [0.010, 0.025, 0.040, 0.035]
    delta = center + rng.normal(0.0, scale * std)
    delta[3] = delta[2]
    delta[7] = delta[6]
    delta[11] = delta[10]
    delta[12:16] = 0.0
    return m316.enforce_shared_motor_consistency(delta)


def run_hold(
    env: Any,
    source: dict[str, Any],
    entry: early.ClosureCandidate,
    hold: HoldCandidate,
    *,
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    early.run_candidate(env, source, entry)
    base_target = env.target_qpos.astype(np.float64).copy()
    measured_q = env.get_state().qpos.astype(np.float64).copy()
    delta = np.asarray(hold.delta, dtype=np.float64)
    # P45 deliberately sends ring toward the pen quickly. At first touch its
    # proximal joint still has substantial closing velocity and the actuator
    # target is ahead of the measured joint. Rebase middle/ring on measured q
    # so this stage brakes at contact instead of driving through the pen.
    desired = base_target + delta
    desired[4:12] = measured_q[4:12] + delta[4:12]
    target = np.clip(
        m316.enforce_shared_motor_consistency(desired),
        env.joint_low,
        env.joint_high,
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    for step in range(steps):
        alpha = min(1.0, (step + 1) / max(1, hold.ramp))
        effective = base_target + alpha * (target - base_target)
        action = np.clip(
            hold.gain * (effective - env.target_qpos.astype(np.float64)),
            -hold.action_limit,
            hold.action_limit,
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
    state = early.m333.snapshot(
        env,
        final["contacts"],
        final["max_penetration_mm"],
        steps / env.control_hz,
    )
    return rows, state


def hold_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def contacts(row: dict[str, Any]) -> set[str]:
        return set() if row["contacts"] == "none" else set(row["contacts"].split(","))

    def ratio(finger: str) -> float:
        return float(sum(finger in contacts(row) for row in rows) / max(1, len(rows)))

    ratios = {finger: ratio(finger) for finger in FINGERS}
    shares = {
        finger: float(np.mean([row[f"{finger}_share"] for row in rows]))
        for finger in FINGERS
    }
    final_contacts = contacts(rows[-1])
    four_ratio = float(
        sum(
            {"thumb", "index", "middle", "ring"}.issubset(contacts(row))
            for row in rows
        )
        / max(1, len(rows))
    )
    p1_support_ratio = float(
        sum(
            {"thumb", "index"}.issubset(contacts(row))
            and bool({"middle", "ring"} & contacts(row))
            for row in rows
        )
        / max(1, len(rows))
    )
    result = {
        "steps": len(rows),
        "survival_s": len(rows) / 100.0,
        "final_contacts": sorted(final_contacts),
        "contact_ratio": ratios,
        "four_finger_contact_ratio": four_ratio,
        "p1_support_ratio": p1_support_ratio,
        "force_share_mean": shares,
        "force_share_final": {
            finger: float(rows[-1][f"{finger}_share"]) for finger in FINGERS
        },
        "tip_distance_final_mm": {
            finger: float(rows[-1][f"{finger}_tip_distance_mm"])
            for finger in FINGERS
        },
        "tip_distance_min_mm": {
            finger: float(min(row[f"{finger}_tip_distance_mm"] for row in rows))
            for finger in FINGERS
        },
        "min_dz_mm": float(min(row["dz_mm"] for row in rows)),
        "final_dz_mm": float(rows[-1]["dz_mm"]),
        "max_penetration_mm": float(max(row["max_penetration_mm"] for row in rows)),
        "final_linear_speed": float(rows[-1]["pen_linear_speed"]),
        "final_angular_speed": float(rows[-1]["pen_angular_speed"]),
    }
    thumb_primary = bool(
        shares["thumb"] >= shares["ring"]
        and shares["thumb"] >= 0.30
        and rows[-1]["thumb_share"] >= rows[-1]["ring_share"]
    )
    result["thumb_primary"] = thumb_primary
    result["accepted"] = bool(
        len(rows) >= 8
        and ratios["thumb"] >= 0.90
        and ratios["index"] >= 0.90
        and ratios["middle"] >= 0.25
        and ratios["ring"] >= 0.25
        and p1_support_ratio >= 0.75
        and four_ratio >= 0.15
        and thumb_primary
        and shares["ring"] <= 0.20
        and result["min_dz_mm"] >= -3.0
        and result["max_penetration_mm"] <= 4.0
        and result["final_linear_speed"] <= 0.08
    )
    distance_score = sum(
        max(0.0, 12.0 - result["tip_distance_final_mm"][finger])
        for finger in ("middle", "ring")
    )
    result["score"] = float(
        900.0 * result["accepted"]
        + 280.0 * four_ratio
        + 220.0 * p1_support_ratio
        + 100.0 * min(ratios["thumb"], ratios["index"])
        + 100.0 * min(ratios["middle"], ratios["ring"])
        + 18.0 * distance_score
        + 100.0 * thumb_primary
        - 400.0 * max(0.0, shares["ring"] - 0.20)
        - 180.0 * max(0.0, shares["ring"] - shares["thumb"])
        - 35.0 * max(0.0, result["max_penetration_mm"] - 1.5)
        - 1000.0 * max(0.0, result["final_linear_speed"] - 0.08)
        - 35.0 * max(0.0, -result["min_dz_mm"] - 2.0)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry-report",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "early_p51_closure_search_m3_2_35/early_p51_closure_search.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p1_entry_hold_search_m3_2_35"
        ),
    )
    parser.add_argument("--entry-count", type=int, default=20)
    parser.add_argument("--samples", type=int, default=16000)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=64)
    parser.add_argument("--hold-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=81)
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
    entries = load_entry_candidates(Path(args.entry_report), args.entry_count)
    env, route, references, cache, cache_index = early.make_env_and_inputs(args)
    rng = np.random.default_rng(args.seed)
    try:
        source = early.build_early_p5_states(
            env, route, references, cache, cache_index, args
        )
        source_states = {
            step: source["p45_snapshots"][step - 1] for step in (4, 5, 6, 7)
        }
        bias = np.zeros(20, dtype=np.float64)
        bias[1:4] = [-0.005, 0.006, 0.006]
        bias[5:8] = [0.015, 0.030, 0.030]
        bias[9:12] = [0.015, 0.020, 0.020]
        bias[17:20] = [-0.010, -0.015, -0.015]
        centers = [np.zeros(20, dtype=np.float64), bias]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, args.samples // max(1, args.generations))
        for generation in range(args.generations):
            scale = 1.0 if generation == 0 else 0.60 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                entry_rank = int(rng.integers(0, len(entries)))
                delta = random_hold_delta(rng, centers, scale)
                hold = HoldCandidate(
                    entry_rank=entry_rank,
                    delta=tuple(delta.astype(float)),
                    ramp=int(rng.integers(1, 4)),
                    gain=float(rng.uniform(0.55, 1.00)),
                    action_limit=float(rng.uniform(0.012, 0.040)),
                )
                entry = entries[entry_rank]
                rows, _ = run_hold(
                    env,
                    source_states[entry.source_step],
                    entry,
                    hold,
                    steps=args.hold_steps,
                )
                summary = hold_summary(rows)
                generation_rows.append(
                    {
                        "generation": generation,
                        "sample": sample,
                        "entry": asdict(entry),
                        "hold": asdict(hold),
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
                np.asarray(row["hold"]["delta"], dtype=np.float64)
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
        best_entry = early.ClosureCandidate(**best["entry"])
        best_hold = HoldCandidate(**best["hold"])
        _, final_state = run_hold(
            env,
            source_states[best_entry.source_step],
            best_entry,
            best_hold,
            steps=args.hold_steps,
        )
        m334.save_final_cache(output_dir / "P1_entry_hold_arrival.npz", final_state)
        report = {
            "stage": "M3.2.35 P5 to short P1-entry hold search",
            "entry_report": str(args.entry_report),
            "hold_steps": args.hold_steps,
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "entry_count": len(entries),
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "p1_entry_hold_search.json", report)
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
