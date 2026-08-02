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
import search_m3_2_35_sustained_ring_entry as sustained


SELECTED_RING = sustained.SustainedRingCandidate(
    p34_ring_target=(
        0.17168992308036626,
        1.3295410157732976,
        0.03231377672875459,
        0.03231377672875459,
    ),
    p34_ring_ramp=5,
    p34_ring_gain=0.7512180376648592,
    p34_ring_action_limit=0.020797923341055617,
    p45_ring_target=(
        0.12779108443175463,
        1.3248576878617433,
        0.5548422843562991,
        0.5548422843562991,
    ),
    p45_ring_ramp=3,
    p45_ring_gain=0.9775400439027817,
    p45_ring_action_limit=0.021987519220876977,
)


@dataclass(frozen=True)
class MiddleSyncCandidate:
    middle_delta: tuple[float, float, float, float]
    middle_ramp: int
    middle_gain: float
    middle_action_limit: float


def random_candidate(
    rng: np.random.Generator,
    centers: list[np.ndarray],
    scale: float,
) -> MiddleSyncCandidate:
    center = centers[int(rng.integers(0, len(centers)))]
    delta = center + rng.normal(
        0.0, scale * np.asarray([0.055, 0.150, 0.180, 0.180])
    )
    delta[3] = delta[2]
    return MiddleSyncCandidate(
        middle_delta=tuple(delta.astype(float)),
        middle_ramp=int(rng.integers(1, 7)),
        middle_gain=float(rng.uniform(0.35, 1.00)),
        middle_action_limit=float(rng.uniform(0.003, 0.025)),
    )


def run_p45(
    env: Any,
    p4_state: dict[str, Any],
    candidate: MiddleSyncCandidate,
    *,
    steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tail.restore_snapshot(env, p4_state)
    base = env.target_qpos.astype(np.float64).copy()
    target = np.clip(
        m316.enforce_shared_motor_consistency(base + cycle.P45_DELTA),
        env.joint_low,
        env.joint_high,
    )
    middle_target = np.clip(
        target[4:8] + np.asarray(candidate.middle_delta),
        env.joint_low[4:8],
        env.joint_high[4:8],
    )
    ring_target = np.clip(
        np.asarray(SELECTED_RING.p45_ring_target),
        env.joint_low[8:12],
        env.joint_high[8:12],
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for step in range(steps):
        alpha = 0.0 if step < cycle.P45_DELAY else min(
            1.0, (step - cycle.P45_DELAY + 1) / cycle.P45_RAMP
        )
        effective = base + alpha * (target - base)
        action = np.clip(
            cycle.P45_GAIN * (effective - env.target_qpos.astype(np.float64)),
            -cycle.P45_ACTION_LIMIT,
            cycle.P45_ACTION_LIMIT,
        )
        middle_alpha = min(1.0, (step + 1) / max(1, candidate.middle_ramp))
        middle_effective = base[4:8] + middle_alpha * (middle_target - base[4:8])
        action[4:8] = np.clip(
            candidate.middle_gain
            * (middle_effective - env.target_qpos[4:8].astype(np.float64)),
            -candidate.middle_action_limit,
            candidate.middle_action_limit,
        )
        ring_alpha = min(1.0, (step + 1) / SELECTED_RING.p45_ring_ramp)
        ring_effective = base[8:12] + ring_alpha * (ring_target - base[8:12])
        action[8:12] = np.clip(
            SELECTED_RING.p45_ring_gain
            * (ring_effective - env.target_qpos[8:12].astype(np.float64)),
            -SELECTED_RING.p45_ring_action_limit,
            SELECTED_RING.p45_ring_action_limit,
        )
        env.step(action.astype(np.float32))
        row = early.rollout_row(
            env,
            step=step + 1,
            initial_center=initial.pen_center,
            initial_axis=initial.pen_axis,
        )
        rows.append(row)
        snapshots.append(
            early.m333.snapshot(
                env,
                row["contacts"],
                row["max_penetration_mm"],
                (step + 1) / env.control_hz,
            )
        )
    return rows, snapshots


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contact_sets = [set(row["contacts"].split(",")) for row in rows]

    def ratio(finger: str) -> float:
        return float(sum(finger in contacts for contacts in contact_sets) / len(rows))

    four = [
        {"thumb", "index", "middle", "ring"}.issubset(contacts)
        for contacts in contact_sets
    ]
    ring = ["ring" in contacts for contacts in contact_sets]
    middle = ["middle" in contacts for contacts in contact_sets]
    synced = [r and m for r, m in zip(ring, middle)]
    best_index = min(
        range(len(rows)),
        key=lambda index: rows[index]["ring_tip_distance_mm"]
        + rows[index]["middle_tip_distance_mm"],
    )
    best = rows[best_index]
    result = {
        "four_finger_contact_ratio": float(sum(four) / len(rows)),
        "middle_ring_sync_ratio": float(sum(synced) / len(rows)),
        "four_finger_longest_steps": sustained.longest_run(four),
        "middle_ring_sync_longest_steps": sustained.longest_run(synced),
        "ring_contact_ratio": ratio("ring"),
        "middle_contact_ratio": ratio("middle"),
        "thumb_contact_ratio": ratio("thumb"),
        "index_contact_ratio": ratio("index"),
        "best_sync_step": best_index + 1,
        "contacts_at_best": best["contacts"],
        "ring_distance_at_best_mm": float(best["ring_tip_distance_mm"]),
        "middle_distance_at_best_mm": float(best["middle_tip_distance_mm"]),
        "thumb_share_at_best": float(best["thumb_share"]),
        "index_share_at_best": float(best["index_share"]),
        "middle_share_at_best": float(best["middle_share"]),
        "ring_share_at_best": float(best["ring_share"]),
        "max_penetration_mm": float(
            max(row["max_penetration_mm"] for row in rows)
        ),
        "min_dz_mm": float(min(row["dz_mm"] for row in rows)),
        "final_linear_speed": float(rows[-1]["pen_linear_speed"]),
    }
    result["accepted"] = bool(
        result["four_finger_longest_steps"] >= 2
        and result["middle_ring_sync_longest_steps"] >= 2
        and result["ring_contact_ratio"] >= 0.16
        and result["middle_contact_ratio"] >= 0.25
        and result["thumb_contact_ratio"] >= 0.90
        and result["index_contact_ratio"] >= 0.90
        and result["ring_share_at_best"] <= 0.20
        and result["thumb_share_at_best"] >= result["ring_share_at_best"]
        and result["max_penetration_mm"] <= 4.0
        and result["min_dz_mm"] >= -4.0
    )
    result["score"] = float(
        1300.0 * result["accepted"]
        + 1000.0 * result["four_finger_contact_ratio"]
        + 700.0 * result["middle_ring_sync_ratio"]
        + 100.0 * result["four_finger_longest_steps"]
        + 75.0 * result["middle_ring_sync_longest_steps"]
        + 350.0 * result["ring_contact_ratio"]
        + 250.0 * result["middle_contact_ratio"]
        + 600.0 * min(
            result["thumb_contact_ratio"], result["index_contact_ratio"]
        )
        - 1200.0
        * (1.0 - min(result["thumb_contact_ratio"], result["index_contact_ratio"]))
        - 350.0 * max(0.0, result["ring_share_at_best"] - 0.20)
        - 35.0 * max(0.0, result["max_penetration_mm"] - 1.5)
        - 450.0 * max(0.0, -result["min_dz_mm"] - 2.0)
        - 1200.0 * max(0.0, result["final_linear_speed"] - 0.08)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p45_middle_ring_sync_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=48)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=131)
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
            ring_target=np.asarray(SELECTED_RING.p34_ring_target),
            ring_ramp=SELECTED_RING.p34_ring_ramp,
            ring_gain=SELECTED_RING.p34_ring_gain,
            ring_action_limit=SELECTED_RING.p34_ring_action_limit,
        )
        p4_state = p34_snapshots[-1]
        centers = [
            np.zeros(4, dtype=np.float64),
            np.asarray([0.0, 0.08, 0.12, 0.12]),
            np.asarray([0.0, -0.06, 0.16, 0.16]),
        ]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, args.samples // max(1, args.generations))
        for generation in range(args.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                if sample == 0:
                    candidate = MiddleSyncCandidate(
                        middle_delta=(0.0, 0.0, 0.0, 0.0),
                        middle_ramp=cycle.P45_RAMP,
                        middle_gain=cycle.P45_GAIN,
                        middle_action_limit=cycle.P45_ACTION_LIMIT,
                    )
                else:
                    candidate = random_candidate(rng, centers, scale)
                rows, _ = run_p45(env, p4_state, candidate, steps=args.steps)
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
                np.asarray(row["candidate"]["middle_delta"])
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
        best_candidate = MiddleSyncCandidate(**best["candidate"])
        _, best_snapshots = run_p45(
            env, p4_state, best_candidate, steps=args.steps
        )
        m334.save_final_cache(
            output_dir / "P1_entry_sync_arrival.npz", best_snapshots[-1]
        )
        report = {
            "stage": "M3.2.35 P45 middle/ring synchronization",
            "ring_trajectory": asdict(SELECTED_RING),
            "p34_summary": tail.summarize(p34_rows, early.P34_STEPS),
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "steps": args.steps,
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "p45_middle_ring_sync_search.json", report)
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
