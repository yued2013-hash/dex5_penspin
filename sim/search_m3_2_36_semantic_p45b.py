from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import build_m3_2_33_narrow_021_five_step_chain as m333
import build_m3_2_34_continuous_p1_p2_p3 as m334
import export_m3_2_23_dynamic_state_cache as m323
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_35_early_p51_closure as early
from evaluate_m3_2_15_short_primitives import FINGERS, contact_metrics


P3_RECAPTURE_DELTA = np.asarray(
    [
        -0.0072232991217649825,
        -0.03857426490525864,
        0.04793665615114322,
        0.04793665615114322,
        -0.010018039940736214,
        0.0088213913049177,
        -0.03991284827005934,
        -0.03991284827005934,
        0.13161270006847567,
        0.14405481047098367,
        0.11891956418096428,
        0.11891956418096428,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.020442434419138317,
        -0.05916183821571673,
        0.02712606793475819,
        -0.05576943835514825,
    ],
    dtype=np.float64,
)
P3_RECAPTURE_STEPS = 10
P3_RECAPTURE_GAIN = 0.6326340360412818
P3_RECAPTURE_LIMIT = 0.013176821924744092

P34_MAIN_DELTA = np.asarray(
    [
        -0.0038907497405305663,
        -0.1324034344868754,
        0.052717354909183306,
        0.052717354909183306,
        -0.053411309956650777,
        0.0280726356884321,
        0.2735455048769181,
        0.2735455048769181,
        -0.08904136413751446,
        -0.19214006188139188,
        -0.3371608489956057,
        -0.3371608489956057,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.02084079356756912,
        -0.06448524388126009,
        -0.128772415266178,
        -0.04583490446545441,
    ],
    dtype=np.float64,
)
P34_MAIN_GAIN = 0.7214390509469173
P34_MAIN_LIMIT = 0.030669602710080374
P34_MIDDLE_DELTA = np.asarray(
    [0.07582273884382942, 0.16214251194671062, 0.20539326544239045],
    dtype=np.float64,
)
P34_MIDDLE_GAIN = 0.97652577
P34_MIDDLE_LIMIT = 0.03032718
P34_RING_GUARD_TARGET = np.asarray(
    [
        0.20822807941389632,
        1.4647739223178164,
        0.6349806691194856,
        0.6349806691194856,
    ],
    dtype=np.float64,
)
P34_RING_GUARD_RAMP = 4
P34_RING_GUARD_GAIN = 0.4266735600515246
P34_RING_GUARD_LIMIT = 0.03815303038910332
P34_DRIVE_STEPS = 9
P34_SETTLE_STEPS = 0

P45A_DELTA = np.asarray(
    [
        -0.030988664503869612,
        -0.01660727489998437,
        0.35974298903163515,
        0.35974298903163515,
        -0.1390221529262343,
        0.0610673564126305,
        -0.5583443116871937,
        -0.5583443116871937,
        0.05470813122794871,
        -0.18668031403238716,
        -0.3373877296346958,
        -0.3373877296346958,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.09086199380691623,
        0.1344500399221955,
        0.06234413311525494,
        0.041456436574694944,
    ],
    dtype=np.float64,
)
P45A_STEPS = 12
P45A_RAMP = 8
P45A_GAIN = 0.7864581649476986
P45A_LIMIT = 0.014533487147796521


@dataclass(frozen=True)
class Candidate:
    delta: tuple[float, ...]
    delay: int
    ramp: int
    gain: float
    action_limit: float


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def make_args(parsed: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=parsed.seed,
        action_delta_limit=0.040,
        actuator_kp=parsed.actuator_kp,
        actuator_kv=parsed.actuator_kv,
        thumb_force_limit=0.0,
        tip_proxy_radius=parsed.tip_proxy_radius,
        tip_proxy_friction=parsed.tip_proxy_friction,
        source_t=0.21,
        source_segment_duration=0.18,
        b12_duration=0.58,
        render_every=2,
        p1_index_open=-0.012,
        p1_middle_open=-0.004,
        p1_ring_open=-0.020,
        p1_thumb_open=0.0,
    )


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def action_to_target(
    env: Any,
    effective_target: np.ndarray,
    gain: float,
    action_limit: float,
) -> np.ndarray:
    return np.clip(
        gain * (effective_target - env.target_qpos.astype(np.float64)),
        -action_limit,
        action_limit,
    ).astype(np.float32)


def run_p3_recapture(env: Any) -> list[dict[str, Any]]:
    base = env.target_qpos.astype(np.float64).copy()
    target = np.clip(
        m316.enforce_shared_motor_consistency(base + P3_RECAPTURE_DELTA),
        env.joint_low,
        env.joint_high,
    )
    rows: list[dict[str, Any]] = []
    initial = env.get_state()
    for step in range(P3_RECAPTURE_STEPS):
        alpha = smoothstep((step + 1) / P3_RECAPTURE_STEPS)
        env.step(
            action_to_target(
                env,
                base + alpha * (target - base),
                P3_RECAPTURE_GAIN,
                P3_RECAPTURE_LIMIT,
            )
        )
        rows.append(metric_row(env, initial.pen_center, initial.pen_axis, step + 1))
    return rows


def run_p34(env: Any) -> list[dict[str, Any]]:
    base = env.target_qpos.astype(np.float64).copy()
    target = np.clip(
        m316.enforce_shared_motor_consistency(base + P34_MAIN_DELTA),
        env.joint_low,
        env.joint_high,
    )
    middle_target = np.clip(
        base[4:8]
        + np.asarray(
            [
                P34_MIDDLE_DELTA[0],
                P34_MIDDLE_DELTA[1],
                P34_MIDDLE_DELTA[2],
                P34_MIDDLE_DELTA[2],
            ]
        ),
        env.joint_low[4:8],
        env.joint_high[4:8],
    )
    ring_target = np.clip(
        P34_RING_GUARD_TARGET,
        env.joint_low[8:12],
        env.joint_high[8:12],
    )
    rows: list[dict[str, Any]] = []
    initial = env.get_state()
    for step in range(P34_DRIVE_STEPS):
        alpha = min(1.0, (step + 1) / 2.0)
        action = action_to_target(
            env,
            base + alpha * (target - base),
            P34_MAIN_GAIN,
            P34_MAIN_LIMIT,
        )
        middle_alpha = min(1.0, (step + 1) / 2.0)
        middle_effective = base[4:8] + middle_alpha * (
            middle_target - base[4:8]
        )
        action[4:8] = np.clip(
            P34_MIDDLE_GAIN
            * (middle_effective - env.target_qpos[4:8].astype(np.float64)),
            -P34_MIDDLE_LIMIT,
            P34_MIDDLE_LIMIT,
        )
        ring_alpha = smoothstep(
            min(1.0, (step + 1) / P34_RING_GUARD_RAMP)
        )
        ring_effective = base[8:12] + ring_alpha * (
            ring_target - base[8:12]
        )
        action[8:12] = np.clip(
            P34_RING_GUARD_GAIN
            * (ring_effective - env.target_qpos[8:12].astype(np.float64)),
            -P34_RING_GUARD_LIMIT,
            P34_RING_GUARD_LIMIT,
        )
        env.step(action)
        rows.append(metric_row(env, initial.pen_center, initial.pen_axis, step + 1))
    for settle_step in range(P34_SETTLE_STEPS):
        env.step(np.zeros(20, dtype=np.float32))
        rows.append(
            metric_row(
                env,
                initial.pen_center,
                initial.pen_axis,
                P34_DRIVE_STEPS + settle_step + 1,
            )
        )
    return rows


def run_p45a(env: Any, steps: int = P45A_STEPS) -> list[dict[str, Any]]:
    base = env.target_qpos.astype(np.float64).copy()
    target = np.clip(
        m316.enforce_shared_motor_consistency(base + P45A_DELTA),
        env.joint_low,
        env.joint_high,
    )
    initial = env.get_state()
    rows: list[dict[str, Any]] = []
    for step in range(steps):
        alpha = smoothstep(min(1.0, (step + 1) / P45A_RAMP))
        env.step(
            action_to_target(
                env,
                base + alpha * (target - base),
                P45A_GAIN,
                P45A_LIMIT,
            )
        )
        rows.append(metric_row(env, initial.pen_center, initial.pen_axis, step + 1))
    return rows


def build_source(
    env: Any,
    args: argparse.Namespace,
    root: Path,
    *,
    p45a_exit_step: int = P45A_STEPS,
) -> dict[str, Any]:
    route = m323.build_route(root / "photo_five_phase_hand_pen_tuned_m3_2_10")
    initial = m334.apply_p1_safety_open(
        route,
        {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0},
    )[0]
    cache = np.load(
        root
        / "short_primitive_reward_m3_2_23/dynamic_state_cache/earlier_pre_ring_for_B12b_to_B23.npz"
    )
    cache_index = m333.m332.cache_index(cache, args.source_t)
    references = tail.load_references(root / "p2_bottom_cradle_b23_thumb_pass_m3_2_31")
    m334.prelude_rollout(
        env,
        route,
        initial,
        m334.source_controller_args(args),
        args,
        render=False,
    )
    _, b12_trace = m333.b12_rollout(
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
    m333.b23_rollout(
        env,
        b12_trace["snapshots"][-1],
        references,
        early.PREPARED_P3,
        args,
        label="continuous_prepared_P3",
        render=False,
        reset_state=False,
    )
    p3_rows = run_p3_recapture(env)
    p3_state = tail.state_snapshot(env)
    p34_rows = run_p34(env)
    p4_state = tail.state_snapshot(env)
    p45a_rows = run_p45a(env, p45a_exit_step)
    return {
        "p3_rows": p3_rows,
        "p3_state": p3_state,
        "p34_rows": p34_rows,
        "p4_state": p4_state,
        "p45a_rows": p45a_rows,
        "p45a_state": tail.state_snapshot(env),
    }


def metric_row(
    env: Any,
    initial_center: np.ndarray,
    initial_axis: np.ndarray,
    step: int,
) -> dict[str, Any]:
    state = env.get_state()
    metrics = contact_metrics(env)
    tip_distances = m316.tip_surface_distances(env)
    fingers = set(metrics["fingers"])
    displacement = (state.pen_center - initial_center) * 1000.0
    row: dict[str, Any] = {
        "step": step,
        "contacts": ",".join(sorted(fingers)) or "none",
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
            tip_distances.get(finger, 0.1) * 1000.0
        )
    return row


def run_candidate(
    env: Any,
    base: np.ndarray,
    candidate: Candidate,
    steps: int,
) -> list[dict[str, Any]]:
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
        alpha = 0.0 if step < candidate.delay else min(
            1.0, (step - candidate.delay + 1) / max(1, candidate.ramp)
        )
        env.step(
            action_to_target(
                env,
                base + alpha * (target - base),
                candidate.gain,
                candidate.action_limit,
            )
        )
        row = metric_row(env, initial.pen_center, initial.pen_axis, step + 1)
        rows.append(row)
        if row["dz_mm"] < -15.0:
            break
    return rows


def summarize(rows: list[dict[str, Any]], expected_steps: int) -> dict[str, Any]:
    contacts = [set(row["contacts"].split(",")) - {"none"} for row in rows]
    ratios = {
        finger: float(sum(finger in contact for contact in contacts) / max(1, len(rows)))
        for finger in FINGERS
    }
    pair_flags = [{"thumb", "index"}.issubset(contact) for contact in contacts]
    final_pair = bool(pair_flags[-1]) if pair_flags else False
    pair_tail = sum(pair_flags[-3:]) if pair_flags else 0
    rotation = float(rows[-1]["planar_rotation_deg"])
    min_dz = float(min(row["dz_mm"] for row in rows))
    max_pen = float(max(row["max_penetration_mm"] for row in rows))
    final_speed = float(rows[-1]["pen_linear_speed"])
    target_rotation_score = max(0.0, min(rotation, 85.0))
    score = (
        5.0 * len(rows)
        + 4.0 * target_rotation_score
        + 120.0 * ratios["thumb"]
        + 100.0 * ratios["index"]
        + 45.0 * sum(pair_flags) / max(1, len(pair_flags))
        + 100.0 * float(final_pair)
        + 35.0 * pair_tail
        - 20.0 * ratios["pinky"]
        - 8.0 * ratios["ring"]
        - 30.0 * max(0.0, -min_dz - 4.0)
        - 25.0 * max(0.0, max_pen - 4.0)
        - 160.0 * max(0.0, final_speed - 0.16)
        - 5.0 * max(0.0, -rotation)
    )
    return {
        "steps": len(rows),
        "rotation_deg": rotation,
        "min_dz_mm": min_dz,
        "max_penetration_mm": max_pen,
        "final_linear_speed": final_speed,
        "final_angular_speed": float(rows[-1]["pen_angular_speed"]),
        "contact_ratio": ratios,
        "pair_ratio": float(sum(pair_flags) / max(1, len(pair_flags))),
        "pair_tail_steps": int(pair_tail),
        "final_contacts": sorted(contacts[-1]) if contacts else [],
        "score": float(score),
        "accepted": bool(
            len(rows) == expected_steps
            and rotation >= 35.0
            and min_dz >= -6.0
            and max_pen <= 4.0
            and ratios["thumb"] >= 0.55
            and ratios["index"] >= 0.55
            and pair_tail >= 2
            and final_pair
        ),
    }


def random_delta(
    rng: np.random.Generator, center: np.ndarray, scale: float
) -> np.ndarray:
    std = np.zeros(20, dtype=np.float64)
    std[0:4] = [0.045, 0.18, 0.32, 0.32]
    std[4:8] = [0.055, 0.22, 0.36, 0.36]
    std[8:12] = [0.035, 0.14, 0.24, 0.24]
    std[16:20] = [0.075, 0.18, 0.24, 0.22]
    delta = center + rng.normal(0.0, scale * std)
    delta[3] = delta[2]
    delta[7] = delta[6]
    delta[11] = delta[10]
    delta[12:16] = 0.0
    return m316.enforce_shared_motor_consistency(delta)


def sample_candidate(
    rng: np.random.Generator, center: np.ndarray, scale: float
) -> Candidate:
    return Candidate(
        delta=tuple(random_delta(rng, center, scale).astype(float)),
        delay=int(rng.integers(0, 4)),
        ramp=int(rng.integers(1, 9)),
        gain=float(rng.uniform(0.35, 1.0)),
        action_limit=float(rng.uniform(0.006, 0.035)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "semantic_p45b_search_m3_2_36"
        ),
    )
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=36)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--p45a-exit-step", type=int, default=P45A_STEPS)
    parser.add_argument("--seed", type=int, default=236)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parsed = parser.parse_args()
    args = make_args(parsed)
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
        source = build_source(
            env, args, root, p45a_exit_step=parsed.p45a_exit_step
        )
        source_state = source["p45a_state"]
        base = np.asarray(source_state["target_qpos"], dtype=np.float64)
        centers = [np.zeros(20, dtype=np.float64)]
        leaderboard: list[dict[str, Any]] = []
        per_generation = max(1, parsed.samples // max(1, parsed.generations))
        for generation in range(parsed.generations):
            scale = 1.0 if generation == 0 else 0.55 / generation
            generation_rows: list[dict[str, Any]] = []
            for sample_index in range(per_generation):
                candidate = sample_candidate(
                    rng, centers[sample_index % len(centers)], scale
                )
                tail.restore_snapshot(env, source_state)
                rows = run_candidate(env, base, candidate, parsed.steps)
                summary = summarize(rows, parsed.steps)
                generation_rows.append(
                    {
                        "generation": generation,
                        "sample": sample_index,
                        "candidate": asdict(candidate),
                        "summary": summary,
                    }
                )
            generation_rows.sort(
                key=lambda item: (item["summary"]["accepted"], item["summary"]["score"]),
                reverse=True,
            )
            leaderboard.extend(generation_rows[: max(parsed.elite, 80)])
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
            "stage": "M3.2.36 semantic P45a endpoint to P5 recapture search",
            "source": {
                "p3": summarize(source["p3_rows"], P3_RECAPTURE_STEPS),
                "p34": summarize(
                    source["p34_rows"], P34_DRIVE_STEPS + P34_SETTLE_STEPS
                ),
                "p45a": summarize(source["p45a_rows"], parsed.p45a_exit_step),
            },
            "search": {
                "samples": parsed.samples,
                "generations": parsed.generations,
                "steps": parsed.steps,
                "p45a_exit_step": parsed.p45a_exit_step,
            },
            "leaderboard": leaderboard[:120],
        }
        save_json(output_dir / "semantic_p45b_search.json", report)
        m334.save_final_cache(output_dir / "semantic_P45a_source.npz", source_state)
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
