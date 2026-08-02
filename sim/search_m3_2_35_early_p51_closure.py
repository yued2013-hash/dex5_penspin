from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import build_m3_2_33_narrow_021_five_step_chain as m333
import build_m3_2_34_continuous_p1_p2_p3 as m334
import export_m3_2_23_dynamic_state_cache as m323
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_cycle_closure as cycle
import search_m3_2_35_dynamic_tail as tail
from evaluate_m3_2_15_short_primitives import FINGERS, contact_metrics


PREPARED_P3 = m333.B23Candidate(
    "short_P3_exit_ring_prepared",
    1.80,
    1.10,
    0.20,
    -0.040,
    -0.020,
    0.88,
    recapture_steps=1,
    ring_prep_roll=0.10,
    ring_prep_proximal=0.45,
    ring_prep_distal=-0.66,
    ring_action_limit=0.022,
    ring_gain=0.70,
)

RING_PREPOSITION_TARGET = np.asarray([0.20, 1.57, 0.12, 0.12], dtype=np.float64)
P34_STEPS = 16
P34_RING_RAMP = 5
P34_RING_GAIN = 0.70
P34_RING_ACTION_LIMIT = 0.014
P45_MAX_STEPS = 7


@dataclass(frozen=True)
class ClosureCandidate:
    source_step: int
    delta: tuple[float, ...]
    steps: int
    ramp: int
    gain: float
    action_limit: float


def make_env_and_inputs(
    args: argparse.Namespace,
) -> tuple[Any, list[dict[str, Any]], dict[str, np.ndarray], Any, int]:
    root = Path("dex5_penspin/reports/manus_mujoco_presets")
    route = m323.build_route(root / "photo_five_phase_hand_pen_tuned_m3_2_10")
    initial_seed = m334.apply_p1_safety_open(
        route,
        {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0},
    )[0]
    references = tail.load_references(root / "p2_bottom_cradle_b23_thumb_pass_m3_2_31")
    cache = np.load(
        root
        / "short_primitive_reward_m3_2_23/dynamic_state_cache/earlier_pre_ring_for_B12b_to_B23.npz"
    )
    cache_index = m333.m332.cache_index(cache, args.source_t)
    env = m316.make_env(initial_seed, m334.model_args(args), duration=4.0, render=False)
    return env, route, references, cache, cache_index


def rollout_row(
    env: Any,
    *,
    step: int,
    initial_center: np.ndarray,
    initial_axis: np.ndarray,
) -> dict[str, Any]:
    state = env.get_state()
    metrics = contact_metrics(env)
    distances = m316.tip_surface_distances(env)
    displacement = (state.pen_center - initial_center) * 1000.0
    fingers = sorted(metrics["fingers"])
    row: dict[str, Any] = {
        "step": int(step),
        "contacts": ",".join(fingers) or "none",
        "dx_mm": float(displacement[0]),
        "dy_mm": float(displacement[1]),
        "dz_mm": float(displacement[2]),
        "max_penetration_mm": float(metrics["max_penetration"] * 1000.0),
        "pen_linear_speed": float(
            np.linalg.norm(env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 3])
        ),
        "pen_angular_speed": float(
            np.linalg.norm(env.data.qvel[env.pen_qvel_adr + 3 : env.pen_qvel_adr + 6])
        ),
        "planar_rotation_deg": m333.signed_planar_rotation_deg(
            initial_axis, state.pen_axis
        ),
    }
    for finger in FINGERS:
        row[f"{finger}_share"] = float(metrics["force_share"].get(finger, 0.0))
        row[f"{finger}_tip_distance_mm"] = float(
            distances.get(finger, 0.1) * 1000.0
        )
    return row


def run_mixed_target(
    env: Any,
    base_target: np.ndarray,
    delta: np.ndarray,
    *,
    steps: int,
    delay: int,
    ramp: int,
    gain: float,
    action_limit: float,
    ring_target: np.ndarray = RING_PREPOSITION_TARGET,
    ring_ramp: int = P34_RING_RAMP,
    ring_gain: float = P34_RING_GAIN,
    ring_action_limit: float = P34_RING_ACTION_LIMIT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = np.clip(
        m316.enforce_shared_motor_consistency(base_target + delta),
        env.joint_low,
        env.joint_high,
    )
    ring_target = np.clip(ring_target, env.joint_low[8:12], env.joint_high[8:12])
    initial_state = env.get_state()
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for step in range(steps):
        alpha = 0.0 if step < delay else min(1.0, (step - delay + 1) / max(1, ramp))
        effective_target = base_target + alpha * (target - base_target)
        action = np.clip(
            gain * (effective_target - env.target_qpos.astype(np.float64)),
            -action_limit,
            action_limit,
        )
        ring_alpha = min(1.0, (step + 1) / max(1, ring_ramp))
        ring_effective = base_target[8:12] + ring_alpha * (
            ring_target - base_target[8:12]
        )
        action[8:12] = np.clip(
            ring_gain
            * (ring_effective - env.target_qpos[8:12].astype(np.float64)),
            -ring_action_limit,
            ring_action_limit,
        )
        env.step(action.astype(np.float32))
        row = rollout_row(
            env,
            step=step + 1,
            initial_center=initial_state.pen_center,
            initial_axis=initial_state.pen_axis,
        )
        rows.append(row)
        snapshots.append(
            m333.snapshot(
                env,
                row["contacts"],
                row["max_penetration_mm"],
                (step + 1) / env.control_hz,
            )
        )
        if row["dz_mm"] < -12.0:
            break
    return rows, snapshots


def build_early_p5_states(
    env: Any,
    route: list[dict[str, Any]],
    references: dict[str, np.ndarray],
    cache: Any,
    cache_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    controller_args = m334.source_controller_args(args)
    initial_seed = m334.apply_p1_safety_open(
        route,
        {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0},
    )[0]
    p1_summary, p1_trace = m334.prelude_rollout(
        env, route, initial_seed, controller_args, args, render=False
    )
    b12_summary, b12_trace = m333.b12_rollout(
        env,
        cache,
        cache_index,
        m333.Perturbation("continuous_nominal"),
        m334.SELECTED_B12,
        args,
        duration_s=args.b12_duration,
        render=False,
        reset_state=False,
    )
    p3_summary, p3_trace = m333.b23_rollout(
        env,
        b12_trace["snapshots"][-1],
        references,
        PREPARED_P3,
        args,
        label="prepared_short_P3_exit",
        render=False,
        reset_state=False,
    )
    p3_base = env.target_qpos.astype(np.float64).copy()
    p34_rows, p34_snapshots = run_mixed_target(
        env,
        p3_base,
        tail.P34_DELTA,
        steps=P34_STEPS,
        delay=tail.P34_DELAY,
        ramp=tail.P34_RAMP,
        gain=tail.P34_GAIN,
        action_limit=tail.P34_ACTION_LIMIT,
    )
    p4_base = env.target_qpos.astype(np.float64).copy()
    p45_rows, p45_snapshots = run_mixed_target(
        env,
        p4_base,
        cycle.P45_DELTA,
        steps=P45_MAX_STEPS,
        delay=cycle.P45_DELAY,
        ramp=cycle.P45_RAMP,
        gain=cycle.P45_GAIN,
        action_limit=cycle.P45_ACTION_LIMIT,
    )
    return {
        "p1_summary": p1_summary,
        "p1_trace": p1_trace,
        "b12_summary": b12_summary,
        "b12_trace": b12_trace,
        "p3_summary": p3_summary,
        "p3_trace": p3_trace,
        "p34_rows": p34_rows,
        "p34_snapshots": p34_snapshots,
        "p45_rows": p45_rows,
        "p45_snapshots": p45_snapshots,
    }


def probe_sources(
    env: Any, source_states: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    groups = {
        "index_roll": (0,),
        "index_proximal": (1,),
        "index_distal": (2, 3),
        "middle_roll": (4,),
        "middle_proximal": (5,),
        "middle_distal": (6, 7),
        "ring_roll": (8,),
        "ring_proximal": (9,),
        "ring_distal": (10, 11),
        "thumb_roll": (16,),
        "thumb_yaw": (17,),
        "thumb_proximal": (18,),
        "thumb_distal": (19,),
    }
    result: list[dict[str, Any]] = []
    for source_step, source in source_states.items():
        for name, indices in groups.items():
            for sign in (-1.0, 1.0):
                tail.restore_snapshot(env, source)
                before = m316.tip_surface_distances(env)
                for _ in range(2):
                    action = np.zeros(20, dtype=np.float32)
                    action[list(indices)] = np.float32(sign * 0.003)
                    env.step(action)
                after = m316.tip_surface_distances(env)
                metrics = contact_metrics(env)
                result.append(
                    {
                        "source_step": source_step,
                        "group": name,
                        "action": sign * 0.003,
                        "middle_distance_change_mm": float(
                            (after["middle"] - before["middle"]) * 1000.0
                        ),
                        "ring_distance_change_mm": float(
                            (after["ring"] - before["ring"]) * 1000.0
                        ),
                        "contacts": ",".join(sorted(metrics["fingers"])) or "none",
                        "max_penetration_mm": float(
                            metrics["max_penetration"] * 1000.0
                        ),
                    }
                )
    return result


def random_candidate(
    rng: np.random.Generator,
    centers: list[np.ndarray],
    source_steps: tuple[int, ...],
    scale: float,
) -> ClosureCandidate:
    center = centers[int(rng.integers(0, len(centers)))]
    std = np.zeros(20, dtype=np.float64)
    std[[0, 1, 2]] = [0.020, 0.045, 0.060]
    std[[4, 5, 6]] = [0.040, 0.105, 0.125]
    std[[8, 9, 10]] = [0.040, 0.120, 0.140]
    std[16:20] = [0.020, 0.055, 0.080, 0.070]
    delta = center + rng.normal(0.0, scale * std)
    delta[3] = delta[2]
    delta[7] = delta[6]
    delta[11] = delta[10]
    delta[12:16] = 0.0
    return ClosureCandidate(
        source_step=int(source_steps[int(rng.integers(0, len(source_steps)))]),
        delta=tuple(m316.enforce_shared_motor_consistency(delta).astype(float)),
        steps=int(rng.integers(1, 5)),
        ramp=int(rng.integers(1, 4)),
        gain=float(rng.uniform(0.32, 0.82)),
        action_limit=float(rng.uniform(0.0015, 0.0090)),
    )


def run_candidate(
    env: Any, source: dict[str, Any], candidate: ClosureCandidate
) -> list[dict[str, Any]]:
    tail.restore_snapshot(env, source)
    base_target = env.target_qpos.astype(np.float64).copy()
    target = np.clip(
        m316.enforce_shared_motor_consistency(
            base_target + np.asarray(candidate.delta, dtype=np.float64)
        ),
        env.joint_low,
        env.joint_high,
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    for step in range(candidate.steps):
        alpha = min(1.0, (step + 1) / max(1, candidate.ramp))
        effective = base_target + alpha * (target - base_target)
        action = np.clip(
            candidate.gain * (effective - env.target_qpos.astype(np.float64)),
            -candidate.action_limit,
            candidate.action_limit,
        ).astype(np.float32)
        env.step(action)
        rows.append(
            rollout_row(
                env,
                step=step + 1,
                initial_center=initial.pen_center,
                initial_axis=initial.pen_axis,
            )
        )
    return rows


def closure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final = rows[-1]
    final_contacts = set() if final["contacts"] == "none" else set(final["contacts"].split(","))

    def ratio(finger: str) -> float:
        return float(
            sum(finger in row["contacts"].split(",") for row in rows)
            / max(1, len(rows))
        )

    def mean_share(finger: str) -> float:
        return float(np.mean([row[f"{finger}_share"] for row in rows]))

    ratios = {finger: ratio(finger) for finger in FINGERS}
    shares = {finger: mean_share(finger) for finger in FINGERS}
    thumb_primary = bool(
        shares["thumb"] >= shares["ring"]
        and final["thumb_share"] >= final["ring_share"]
    )
    four_touch = {"thumb", "index", "middle", "ring"}.issubset(final_contacts)
    support_ok = {"thumb", "index"}.issubset(final_contacts)
    result = {
        "steps": len(rows),
        "final_contacts": sorted(final_contacts),
        "contact_ratio": ratios,
        "force_share_mean": shares,
        "force_share_final": {
            finger: float(final[f"{finger}_share"]) for finger in FINGERS
        },
        "tip_distance_final_mm": {
            finger: float(final[f"{finger}_tip_distance_mm"]) for finger in FINGERS
        },
        "tip_distance_min_mm": {
            finger: float(min(row[f"{finger}_tip_distance_mm"] for row in rows))
            for finger in FINGERS
        },
        "min_dz_mm": float(min(row["dz_mm"] for row in rows)),
        "max_penetration_mm": float(max(row["max_penetration_mm"] for row in rows)),
        "final_linear_speed": float(final["pen_linear_speed"]),
        "final_angular_speed": float(final["pen_angular_speed"]),
        "thumb_primary": thumb_primary,
        "four_finger_touch": four_touch,
    }
    result["accepted"] = bool(
        four_touch
        and support_ok
        and thumb_primary
        and ratios["thumb"] >= 0.75
        and ratios["index"] >= 0.75
        and shares["ring"] <= 0.25
        and result["min_dz_mm"] >= -3.0
        and result["max_penetration_mm"] <= 4.0
        and result["final_linear_speed"] <= 0.08
    )
    distance_score = sum(
        max(0.0, 12.0 - result["tip_distance_final_mm"][finger])
        for finger in ("middle", "ring")
    )
    result["score"] = float(
        500.0 * four_touch
        + 180.0 * ("ring" in final_contacts)
        + 180.0 * ("middle" in final_contacts)
        + 120.0 * support_ok
        + 80.0 * thumb_primary
        + 18.0 * distance_score
        + 80.0 * min(ratios["thumb"], ratios["index"])
        - 300.0 * max(0.0, shares["ring"] - 0.25)
        - 100.0 * max(0.0, shares["ring"] - shares["thumb"])
        - 30.0 * max(0.0, result["max_penetration_mm"] - 1.5)
        - 900.0 * max(0.0, result["final_linear_speed"] - 0.08)
        - 30.0 * max(0.0, -result["min_dz_mm"] - 2.0)
    )
    return result


def save_source_cache(path: Path, states: dict[int, dict[str, Any]]) -> None:
    ordered = sorted(states.items())
    np.savez_compressed(
        path,
        source_step=np.asarray([step for step, _ in ordered], dtype=np.int64),
        qpos=np.asarray([state["qpos"] for _, state in ordered]),
        qvel=np.asarray([state["qvel"] for _, state in ordered]),
        target_qpos=np.asarray([state["target_qpos"] for _, state in ordered]),
        hand_qpos=np.asarray([state["hand_qpos"] for _, state in ordered]),
        hand_qvel=np.asarray([state["hand_qvel"] for _, state in ordered]),
        rotation_phase=np.asarray([state["rotation_phase"] for _, state in ordered]),
        contacts=np.asarray([state["contacts"] for _, state in ordered]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "early_p51_closure_search_m3_2_35"
        ),
    )
    parser.add_argument("--samples", type=int, default=24000)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite", type=int, default=64)
    parser.add_argument("--seed", type=int, default=75)
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
    env, route, references, cache, cache_index = make_env_and_inputs(args)
    rng = np.random.default_rng(args.seed)
    try:
        source = build_early_p5_states(
            env, route, references, cache, cache_index, args
        )
        source_states = {
            step: source["p45_snapshots"][step - 1] for step in (4, 5, 6, 7)
        }
        probes = probe_sources(env, source_states)
        tail.save_json(output_dir / "joint_sensitivity_probe.json", probes)
        save_source_cache(output_dir / "early_P5_source_states.npz", source_states)

        centers = [np.zeros(20, dtype=np.float64)]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, args.samples // max(1, args.generations))
        for generation in range(args.generations):
            scale = 1.0 if generation == 0 else 0.60 / generation
            rows: list[dict[str, Any]] = []
            for sample in range(per_generation):
                candidate = random_candidate(
                    rng, centers, (4, 5, 6, 7), scale
                )
                rollout = run_candidate(
                    env, source_states[candidate.source_step], candidate
                )
                summary = closure_summary(rollout)
                rows.append(
                    {
                        "generation": generation,
                        "sample": sample,
                        "candidate": asdict(candidate),
                        "summary": summary,
                    }
                )
            rows.sort(
                key=lambda row: (
                    row["summary"]["accepted"],
                    row["summary"]["score"],
                ),
                reverse=True,
            )
            leaderboard.extend(rows[: max(args.elite, 100)])
            centers = [
                np.asarray(row["candidate"]["delta"], dtype=np.float64)
                for row in rows[: args.elite]
            ]
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "accepted": sum(row["summary"]["accepted"] for row in rows),
                        "four_finger_touch": sum(
                            row["summary"]["four_finger_touch"] for row in rows
                        ),
                        "best": rows[0],
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
            "stage": "M3.2.35 early P5 to P1-entry micro-contact search",
            "source": {
                "prepared_b23": asdict(PREPARED_P3),
                "p3_summary": source["p3_summary"],
                "p34_summary": tail.summarize(source["p34_rows"], P34_STEPS),
                "p45_rows": source["p45_rows"],
                "ring_preposition_target": RING_PREPOSITION_TARGET.astype(float).tolist(),
            },
            "search": {
                "samples": args.samples,
                "generations": args.generations,
                "sources": [4, 5, 6, 7],
                "acceptance": (
                    "final thumb+index+middle+ring contact; thumb mean/final share >= ring; "
                    "thumb/index contact ratio >= 0.75; ring mean share <= 0.25; "
                    "penetration <= 4 mm; dz >= -3 mm; final linear speed <= 0.08 m/s"
                ),
            },
            "leaderboard": leaderboard[:200],
        }
        tail.save_json(output_dir / "early_p51_closure_search.json", report)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "p3": source["p3_summary"],
                    "p34": report["source"]["p34_summary"],
                    "p45": source["p45_rows"],
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
