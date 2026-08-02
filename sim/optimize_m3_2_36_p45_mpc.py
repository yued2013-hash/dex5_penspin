from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import build_m3_2_34_continuous_p1_p2_p3 as m334
import export_m3_2_23_dynamic_state_cache as m323
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_36_after_middle_touch as roll
import search_m3_2_36_p45_middle_sync as middle
import search_m3_2_36_semantic_p45b as semantic
from evaluate_m3_2_15_short_primitives import contact_metrics


ACTIVE_JOINTS = np.asarray(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
    ],
    dtype=np.int32,
)


@dataclass
class FullState:
    data: mujoco.MjData
    target_qpos: np.ndarray
    last_action: np.ndarray
    step_count: int
    rotation_phase: float
    prev_phase: float


def copy_full_state(env: Any) -> FullState:
    data = mujoco.MjData(env.model)
    mujoco.mj_copyData(data, env.model, env.data)
    return FullState(
        data=data,
        target_qpos=env.target_qpos.copy(),
        last_action=env.last_action.copy(),
        step_count=int(env.step_count),
        rotation_phase=float(env.rotation_phase),
        prev_phase=float(env.prev_phase),
    )


def restore_full_state(env: Any, state: FullState) -> None:
    mujoco.mj_copyData(env.data, env.model, state.data)
    env.target_qpos = state.target_qpos.copy()
    env.last_action = state.last_action.copy()
    env.step_count = state.step_count
    env.rotation_phase = state.rotation_phase
    env.prev_phase = state.prev_phase


def enforce_action_consistency(action: np.ndarray) -> np.ndarray:
    result = np.asarray(action, dtype=np.float64).copy()
    result[3] = result[2]
    result[7] = result[6]
    result[11] = result[10]
    result[15] = result[14]
    return result


def rollout_score(
    env: Any,
    source: FullState,
    action: np.ndarray,
    *,
    horizon: int,
    global_center: np.ndarray,
    global_axis: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    restore_full_state(env, source)
    score = -8.0 * float(np.linalg.norm(action[ACTIVE_JOINTS]))
    last_row: dict[str, Any] = {}
    pair_steps = 0
    for lookahead in range(horizon):
        env.step(action.astype(np.float32) if lookahead == 0 else np.zeros(20, np.float32))
        row = semantic.metric_row(
            env, global_center, global_axis, lookahead + 1
        )
        last_row = row
        contacts = set(row["contacts"].split(",")) - {"none"}
        pair = {"thumb", "index"}.issubset(contacts)
        pair_steps += int(pair)
        weight = 1.0 + 0.35 * lookahead
        thumb_distance = row["thumb_tip_distance_mm"]
        index_distance = row["index_tip_distance_mm"]
        middle_distance = row["middle_tip_distance_mm"]
        ring_distance = row["ring_tip_distance_mm"]
        thumb_error = abs(thumb_distance - 7.40)
        index_error = abs(index_distance - 7.40)
        middle_bridge = "middle" in contacts and (
            "thumb" in contacts or "index" in contacts
        )
        ring_guard = "ring" in contacts and (
            "thumb" in contacts or "index" in contacts
        )
        score += weight * (
            1050.0 * float(pair)
            + 240.0 * float("thumb" in contacts)
            + 220.0 * float("index" in contacts)
            + 430.0 * float("middle" in contacts)
            + 520.0 * float(middle_bridge)
            + 210.0 * float("ring" in contacts)
            + 330.0 * float(ring_guard)
            + 6.0 * float(row["planar_rotation_deg"])
            - 55.0 * thumb_error
            - 50.0 * index_error
            - 130.0 * max(0.0, thumb_distance - 7.80)
            - 120.0 * max(0.0, index_distance - 7.80)
            - 75.0 * max(0.0, middle_distance - 8.40)
            - 35.0 * max(0.0, ring_distance - 12.0)
            - 800.0 * max(0.0, row["pen_linear_speed"] - 0.12)
            - 24.0 * max(0.0, row["pen_angular_speed"] - 18.0)
            - 45.0 * max(0.0, -row["dz_mm"] - 3.0)
            - 35.0 * max(0.0, row["max_penetration_mm"] - 3.0)
            - 80.0 * float("pinky" in contacts)
            - 650.0 * max(0.0, row["ring_share"] - 0.20)
        )
    score += 1100.0 * pair_steps
    if last_row:
        final_contacts = set(last_row["contacts"].split(",")) - {"none"}
        score += 2600.0 * float(
            {"thumb", "index"}.issubset(final_contacts)
        )
    restore_full_state(env, source)
    return float(score), last_row


def cem_action(
    env: Any,
    source: FullState,
    rng: np.random.Generator,
    mean_action: np.ndarray,
    *,
    samples: int,
    iterations: int,
    elite: int,
    horizon: int,
    global_center: np.ndarray,
    global_axis: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    active_mean = mean_action[ACTIVE_JOINTS].astype(np.float64).copy()
    active_std = np.full(len(ACTIVE_JOINTS), 0.010, dtype=np.float64)
    best_action = mean_action.astype(np.float64).copy()
    best_score = -np.inf
    best_prediction: dict[str, Any] = {}
    for iteration in range(iterations):
        candidates = rng.normal(
            active_mean,
            active_std,
            size=(samples, len(ACTIVE_JOINTS)),
        )
        candidates = np.clip(candidates, -0.040, 0.040)
        candidates[0] = active_mean
        candidates[1] = 0.0
        scored: list[tuple[float, np.ndarray, dict[str, Any]]] = []
        for active in candidates:
            action = np.zeros(20, dtype=np.float64)
            action[ACTIVE_JOINTS] = active
            action = enforce_action_consistency(action)
            score, prediction = rollout_score(
                env,
                source,
                action,
                horizon=horizon,
                global_center=global_center,
                global_axis=global_axis,
            )
            scored.append((score, action, prediction))
            if score > best_score:
                best_score = score
                best_action = action.copy()
                best_prediction = prediction
        scored.sort(key=lambda item: item[0], reverse=True)
        elite_actions = np.asarray(
            [item[1][ACTIVE_JOINTS] for item in scored[:elite]],
            dtype=np.float64,
        )
        active_mean = elite_actions.mean(axis=0)
        active_std = np.maximum(0.0012, elite_actions.std(axis=0) * 0.72)
    restore_full_state(env, source)
    return best_action.astype(np.float32), {
        "score": float(best_score),
        "prediction": best_prediction,
    }


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "semantic_p45_mpc_m3_2_36"
        ),
    )
    parser.add_argument("--prefix-steps", type=int, default=3)
    parser.add_argument("--touch-steps", type=int, default=roll.MIDDLE_TOUCH_STEPS)
    parser.add_argument("--mpc-steps", type=int, default=24)
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=456)
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
    env = m316.make_env(initial, m334.model_args(args), duration=5.0, render=False)
    rng = np.random.default_rng(parsed.seed)
    try:
        source = semantic.build_source(env, args, root, p45a_exit_step=0)
        p4_state = source["p4_state"]
        p4_center = np.asarray(p4_state["pen_center"], dtype=np.float64)
        p4_axis = np.asarray(p4_state["pen_axis"], dtype=np.float64)
        touch_rows = middle.run_candidate(
            env,
            p4_state,
            roll.MIDDLE_TOUCH,
            parsed.touch_steps,
        )
        touch_state = semantic.tail.state_snapshot(env)
        prefix_rows = roll.run_candidate(
            env,
            touch_state,
            roll.PAIR_ROLL_1,
            parsed.prefix_steps,
        )
        global_center = p4_center.copy()
        global_axis = p4_axis.copy()
        rows: list[dict[str, Any]] = []
        actions: list[list[float]] = []
        diagnostics: list[dict[str, Any]] = []
        previous_action = np.zeros(20, dtype=np.float32)
        for step in range(parsed.mpc_steps):
            source_full = copy_full_state(env)
            action, diagnostic = cem_action(
                env,
                source_full,
                rng,
                0.35 * previous_action,
                samples=parsed.samples,
                iterations=parsed.iterations,
                elite=parsed.elite,
                horizon=parsed.horizon,
                global_center=global_center,
                global_axis=global_axis,
            )
            restore_full_state(env, source_full)
            env.step(action)
            row = semantic.metric_row(
                env, global_center, global_axis, step + 1
            )
            rows.append(row)
            actions.append(action.astype(float).tolist())
            diagnostics.append(diagnostic)
            previous_action = action
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "contacts": row["contacts"],
                        "rotation_deg": row["planar_rotation_deg"],
                        "dz_mm": row["dz_mm"],
                        "linear_speed": row["pen_linear_speed"],
                        "angular_speed": row["pen_angular_speed"],
                        "thumb_distance_mm": row["thumb_tip_distance_mm"],
                        "index_distance_mm": row["index_tip_distance_mm"],
                        "score": diagnostic["score"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if row["dz_mm"] < -15.0:
                break
        summary = roll.summarize(rows, parsed.mpc_steps)
        prefix_summary = (
            roll.summarize(prefix_rows, parsed.prefix_steps)
            if prefix_rows
            else {"steps": 0, "status": "omitted"}
        )
        report = {
            "stage": "M3.2.36 P45 rolling-contact MPC",
            "config": vars(parsed),
            "source": {
                "p3": semantic.summarize(
                    source["p3_rows"], semantic.P3_RECAPTURE_STEPS
                ),
                "p34": semantic.summarize(
                    source["p34_rows"],
                    semantic.P34_DRIVE_STEPS + semantic.P34_SETTLE_STEPS,
                ),
                "middle_touch": (
                    middle.summarize(touch_rows, parsed.touch_steps)
                    if touch_rows
                    else {"steps": 0, "status": "omitted"}
                ),
                "prefix": prefix_summary,
            },
            "mpc": summary,
            "rows": rows,
            "actions": actions,
            "diagnostics": diagnostics,
        }
        save_json(output_dir / "p45_mpc_search.json", report)
        print(
            json.dumps(
                {"output_dir": str(output_dir), "summary": summary},
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
