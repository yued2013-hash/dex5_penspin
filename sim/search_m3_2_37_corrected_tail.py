from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import build_m3_2_33_narrow_021_five_step_chain as m333
import build_m3_2_34_continuous_p1_p2_p3 as m334
import export_m3_2_23_dynamic_state_cache as m323
import optimize_m3_2_36_p45_mpc as mpc
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_35_early_p51_closure as early
import search_m3_2_36_semantic_p45b as semantic
from evaluate_m3_2_15_short_primitives import contact_metrics


# Seven no-reset actions found from the real B23 exit. Every endpoint keeps
# index/middle/thumb contact while advancing the pen phase.
P34_TRIAD_PUSH_ACTIONS = np.asarray(
    [
        [-0.04, -0.02395787276752508, -0.005371611707439941, -0.005371611707439941, -0.02555000191813577, 0.0015722022592569312, 0.008679358998447505, 0.008679358998447505, 0.01850372945177579, 0.001355042583977817, 0.011816842594406216, 0.011816842594406216, 0.0, 0.0, 0.0, 0.0, 0.019185824047047148, -0.019825806233190704, -0.030533303975046423, -0.0339619872410029],
        [-0.02877669402526729, -0.035189892135257146, 0.008807097229622673, 0.008807097229622673, -0.003950180555311421, 0.0035524057876254616, 0.023523537693091172, 0.023523537693091172, 0.005329699460290462, -0.006135082564010602, -0.005460471642584637, -0.005460471642584637, 0.0, 0.0, 0.0, 0.0, -0.01314621937842116, -0.04, -0.04, -0.013476831930295632],
        [-0.04, -0.04, 0.010227984074976547, 0.010227984074976547, -0.005667337578771692, 0.028155830299291435, 0.02142792507714919, 0.02142792507714919, -0.026448473912471397, -0.012469174269465312, -0.04, -0.04, 0.0, 0.0, 0.0, 0.0, -0.004097499291959338, -0.005713211150259931, -0.022687163203308946, -0.0039164237364706575],
        [0.022584812514229637, -0.03300994088586232, -0.008003270558404104, -0.008003270558404104, 0.01894447816002413, 0.04, 0.007942630544360313, 0.007942630544360313, 0.021874913641851103, 0.011296798015250189, 0.006817798801852446, 0.006817798801852446, 0.0, 0.0, 0.0, 0.0, 0.006251254613245453, 0.009945463541098152, -0.016942483462569144, -0.021218479971922433],
        [0.018329625615151397, -0.026221679724589757, 0.003780485413032036, 0.003780485413032036, 0.019319063015989346, 0.03394615881760985, 0.007069659679328678, 0.007069659679328678, -0.03329716801665172, -0.023976541087326594, 0.0014986954883412999, 0.0014986954883412999, 0.0, 0.0, 0.0, 0.0, 0.023656341877937175, -0.023141038173815347, -0.032884798200069, 0.007579757288145039],
        [0.011746895554032974, -0.04, -0.02848594117546112, -0.02848594117546112, -0.025913772541330548, 0.019397340261875114, 0.035795623608114537, 0.035795623608114537, -0.024705185193336414, -0.004322334662449839, -0.018077706292045657, -0.018077706292045657, 0.0, 0.0, 0.0, 0.0, -0.002139177325731982, -0.01603237137731312, -0.012486310198843045, -0.01935737682321098],
        [0.01516300190174622, -0.04, 0.002077903307626992, 0.002077903307626992, -0.004532375750288128, 0.03787223442776321, 0.021232220873098742, 0.021232220873098742, 0.021230723700987136, -0.019268006629130828, -0.008905664004896249, -0.008905664004896249, 0.0, 0.0, 0.0, 0.0, 0.0037132681922350252, -0.0071592498187352405, -0.008478248424963815, -0.018043040109871258],
    ],
    dtype=np.float64,
)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def make_args(parsed: argparse.Namespace) -> argparse.Namespace:
    return semantic.make_args(parsed)


def make_env(args: argparse.Namespace, root: Path, *, render: bool = False) -> Any:
    route = m323.build_route(root / "photo_five_phase_hand_pen_tuned_m3_2_10")
    initial = m334.apply_p1_safety_open(
        route,
        {"index": -0.012, "middle": -0.004, "ring": -0.020, "thumb": 0.0},
    )[0]
    return m316.make_env(
        initial,
        m334.model_args(args),
        duration=6.0,
        render=render,
    )


def build_true_p3(env: Any, args: argparse.Namespace, root: Path) -> dict[str, Any]:
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
    prelude = m334.prelude_rollout(
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
    b23_summary, b23_trace = m333.b23_rollout(
        env,
        b12_trace["snapshots"][-1],
        references,
        early.PREPARED_P3,
        args,
        label="m3_2_37_true_P3",
        render=False,
        reset_state=False,
    )
    return {
        "prelude": prelude,
        "b12_trace": b12_trace,
        "b23_summary": b23_summary,
        "b23_trace": b23_trace,
        "p3_state": tail.state_snapshot(env),
    }


def force_row(env: Any) -> dict[str, Any]:
    metrics = contact_metrics(env)
    return {
        "normal_force": {
            key: float(value) for key, value in metrics["normal_force"].items()
        },
        "total_normal_force": float(metrics["total_normal_force"]),
    }


def run_fixed_actions(
    env: Any,
    actions: np.ndarray,
    initial_center: np.ndarray,
    initial_axis: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    forces: list[dict[str, Any]] = []
    for step, action in enumerate(actions, start=1):
        env.step(np.asarray(action, dtype=np.float32))
        rows.append(semantic.metric_row(env, initial_center, initial_axis, step))
        forces.append(force_row(env))
    return rows, forces


def pair_action_score(
    env: Any,
    source: mpc.FullState,
    action: np.ndarray,
    *,
    horizon: int,
    initial_center: np.ndarray,
    initial_axis: np.ndarray,
    base_rotation_deg: float,
) -> tuple[float, dict[str, Any]]:
    mpc.restore_full_state(env, source)
    score = -8.0 * float(np.linalg.norm(action[mpc.ACTIVE_JOINTS]))
    pair_count = 0
    last_row: dict[str, Any] = {}
    for lookahead in range(horizon):
        env.step(
            action.astype(np.float32)
            if lookahead == 0
            else np.zeros(20, dtype=np.float32)
        )
        row = semantic.metric_row(
            env,
            initial_center,
            initial_axis,
            lookahead + 1,
        )
        last_row = row
        contacts = set(row["contacts"].split(",")) - {"none"}
        pair = {"thumb", "index"}.issubset(contacts)
        pair_count += int(pair)
        increment = float(row["planar_rotation_deg"] - base_rotation_deg)
        displacement = float(
            np.linalg.norm([row["dx_mm"], row["dy_mm"], row["dz_mm"]])
        )
        weight = 1.0 + 0.35 * lookahead
        score += weight * (
            170.0 * max(-2.0, min(increment, 12.0))
            + 6500.0 * float(pair)
            + 900.0 * float("thumb" in contacts)
            + 850.0 * float("index" in contacts)
            + 180.0 * float("middle" in contacts)
            - 6500.0 * float(not pair)
            - 650.0 * max(0.0, row["ring_share"] - 0.18)
            - 300.0 * float("pinky" in contacts)
            - 90.0 * abs(row["thumb_tip_distance_mm"] - 7.5)
            - 85.0 * abs(row["index_tip_distance_mm"] - 7.5)
            - 180.0 * max(0.0, row["max_penetration_mm"] - 2.5)
            - 4000.0 * max(0.0, row["pen_linear_speed"] - 0.14)
            - 110.0 * max(0.0, row["pen_angular_speed"] - 22.0)
            - 35.0 * max(0.0, displacement - 15.0)
            - 280.0 * max(0.0, -row["dz_mm"] - 6.0)
        )
    score += 4000.0 * pair_count
    mpc.restore_full_state(env, source)
    return float(score), last_row


def choose_pair_action(
    env: Any,
    source: mpc.FullState,
    rng: np.random.Generator,
    mean_action: np.ndarray,
    *,
    samples: int,
    iterations: int,
    elite: int,
    horizon: int,
    initial_center: np.ndarray,
    initial_axis: np.ndarray,
    base_rotation_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    active_mean = mean_action[mpc.ACTIVE_JOINTS].astype(np.float64)
    active_std = np.full(len(mpc.ACTIVE_JOINTS), 0.011, dtype=np.float64)
    best_score = -np.inf
    best_action = np.zeros(20, dtype=np.float64)
    best_prediction: dict[str, Any] = {}
    for _ in range(iterations):
        candidates = rng.normal(
            active_mean,
            active_std,
            size=(samples, len(mpc.ACTIVE_JOINTS)),
        )
        candidates = np.clip(candidates, -0.040, 0.040)
        candidates[0] = active_mean
        candidates[1] = 0.0
        scored: list[tuple[float, np.ndarray, dict[str, Any]]] = []
        for active in candidates:
            action = np.zeros(20, dtype=np.float64)
            action[mpc.ACTIVE_JOINTS] = active
            action = mpc.enforce_action_consistency(action)
            score, prediction = pair_action_score(
                env,
                source,
                action,
                horizon=horizon,
                initial_center=initial_center,
                initial_axis=initial_axis,
                base_rotation_deg=base_rotation_deg,
            )
            scored.append((score, action, prediction))
            if score > best_score:
                best_score = score
                best_action = action.copy()
                best_prediction = prediction
        scored.sort(key=lambda item: item[0], reverse=True)
        elite_actions = np.asarray(
            [item[1][mpc.ACTIVE_JOINTS] for item in scored[:elite]],
            dtype=np.float64,
        )
        active_mean = elite_actions.mean(axis=0)
        active_std = np.maximum(0.0012, elite_actions.std(axis=0) * 0.72)
    mpc.restore_full_state(env, source)
    return best_action.astype(np.float32), {
        "score": float(best_score),
        "prediction": best_prediction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "corrected_p3_p4_pair_roll_m3_2_37"
        ),
    )
    parser.add_argument("--pair-steps", type=int, default=6)
    parser.add_argument("--samples", type=int, default=420)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--elite", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=245)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    parsed = parser.parse_args()
    args = make_args(parsed)
    root = Path("dex5_penspin/reports/manus_mujoco_presets")
    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args, root)
    rng = np.random.default_rng(parsed.seed)
    try:
        source = build_true_p3(env, args, root)
        p3 = env.get_state()
        p3_center = p3.pen_center.copy()
        p3_axis = p3.pen_axis.copy()
        p34_rows, p34_forces = run_fixed_actions(
            env,
            P34_TRIAD_PUSH_ACTIONS,
            p3_center,
            p3_axis,
        )
        p4_state = tail.state_snapshot(env)
        pair_rows: list[dict[str, Any]] = []
        pair_forces: list[dict[str, Any]] = []
        actions: list[list[float]] = []
        diagnostics: list[dict[str, Any]] = []
        previous_action = np.zeros(20, dtype=np.float32)
        for step in range(parsed.pair_steps):
            base_rotation = semantic.metric_row(
                env,
                p3_center,
                p3_axis,
                step,
            )["planar_rotation_deg"]
            full_state = mpc.copy_full_state(env)
            action, diagnostic = choose_pair_action(
                env,
                full_state,
                rng,
                0.25 * previous_action,
                samples=parsed.samples,
                iterations=parsed.iterations,
                elite=parsed.elite,
                horizon=parsed.horizon,
                initial_center=p3_center,
                initial_axis=p3_axis,
                base_rotation_deg=base_rotation,
            )
            mpc.restore_full_state(env, full_state)
            env.step(action)
            row = semantic.metric_row(
                env,
                p3_center,
                p3_axis,
                step + 1,
            )
            pair_rows.append(row)
            pair_forces.append(force_row(env))
            actions.append(action.astype(float).tolist())
            diagnostics.append(diagnostic)
            previous_action = action
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "rotation_deg": row["planar_rotation_deg"],
                        "contacts": row["contacts"],
                        "linear_speed": row["pen_linear_speed"],
                        "score": diagnostic["score"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        report = {
            "stage": "M3.2.37 corrected P3/P4 and thumb-index pair-roll search",
            "config": vars(parsed),
            "semantic_correction": {
                "p3": "B23 exit is P3; no forced index/ring/thumb recapture",
                "p34": "index/middle push while thumb supports; ring carries no load",
                "p45": "middle releases into thumb/index pair roll",
            },
            "b23": source["b23_summary"],
            "p34_rows": p34_rows,
            "p34_forces": p34_forces,
            "pair_rows": pair_rows,
            "pair_forces": pair_forces,
            "pair_actions": actions,
            "pair_diagnostics": diagnostics,
        }
        save_json(output_dir / "corrected_tail_search.json", report)
        m334.save_final_cache(output_dir / "corrected_P3_entry.npz", source["p3_state"])
        m334.save_final_cache(output_dir / "corrected_P4_entry.npz", p4_state)
        m334.save_final_cache(
            output_dir / "pair_roll_exit.npz",
            tail.state_snapshot(env),
        )
        print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
