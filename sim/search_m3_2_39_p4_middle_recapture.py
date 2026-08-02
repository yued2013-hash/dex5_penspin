from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

import build_m3_2_34_continuous_p1_p2_p3 as m334
import optimize_m3_2_36_p45_mpc as mpc
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_36_semantic_p45b as semantic
from evaluate_m3_2_15_short_primitives import quat_to_axis


SOURCE_DIR = Path(
    "dex5_penspin/reports/manus_mujoco_presets/"
    "corrected_p3_p4_chain_m3_2_38"
)
BASE_ACTION = np.asarray(
    [
        0.000693454254167919,
        0.024537279490138696,
        0.00900304994249234,
        0.00900304994249234,
        0.03242609512920027,
        0.03648270839219788,
        0.02312897499375951,
        0.02312897499375951,
        0.019287804774774965,
        0.016365223186458326,
        -0.00044627526973015086,
        -0.00044627526973015086,
        0.0,
        0.0,
        0.0,
        0.0,
        0.024392534474215792,
        0.03331207579479579,
        -0.01721412458365122,
        -0.03675718525337542,
    ],
    dtype=np.float64,
)
TARGET_PROGRESS_DEG = 56.04638800142325 - 33.26725178969861


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed=args.seed,
        action_delta_limit=args.action_delta_limit,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=0.0,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
    )


def load_source_state(path: Path) -> dict[str, Any]:
    cache = np.load(path, allow_pickle=True)
    return {key: cache[key][0] for key in cache.files}


def make_candidate(rng: np.random.Generator, sample: int) -> np.ndarray:
    if sample == 0:
        return BASE_ACTION.copy()
    action = BASE_ACTION.copy()
    action[0:4] += rng.normal(0.0, 0.012, size=4)
    mode = sample % 4
    if mode == 0:
        middle = rng.uniform(-0.040, 0.040, size=4)
    elif mode == 1:
        middle = BASE_ACTION[4:8] + rng.normal(0.0, 0.026, size=4)
    elif mode == 2:
        middle = -BASE_ACTION[4:8] + rng.normal(0.0, 0.018, size=4)
    else:
        middle = np.asarray(
            [
                rng.uniform(-0.040, 0.015),
                rng.uniform(-0.040, 0.040),
                rng.uniform(-0.040, 0.015),
                0.0,
            ],
            dtype=np.float64,
        )
    middle[3] = middle[2]
    action[4:8] = middle
    action[8:12] = np.clip(
        0.35 * BASE_ACTION[8:12] + rng.normal(0.0, 0.010, size=4),
        -0.020,
        0.020,
    )
    action[11] = action[10]
    action[12:16] = 0.0
    action[16:20] += rng.normal(0.0, 0.012, size=4)
    return mpc.enforce_action_consistency(
        np.clip(action, -0.040, 0.040)
    ).astype(np.float64)


def evaluate(
    env: Any,
    source_state: dict[str, Any],
    action: np.ndarray,
    target_center: np.ndarray,
    target_axis: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    tail.restore_snapshot(env, source_state)
    start = env.get_state()
    rows: list[dict[str, Any]] = []
    max_penetration = 0.0
    no_force_gap = 0
    max_no_force_gap = 0
    for step in range(horizon):
        env.step(action.astype(np.float32))
        row = semantic.metric_row(env, start.pen_center, start.pen_axis, step + 1)
        rows.append(row)
        max_penetration = max(max_penetration, float(row["max_penetration_mm"]))
        supported = sum(
            float(row[f"{finger}_share"])
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        ) > 1e-6
        no_force_gap = 0 if supported else no_force_gap + 1
        max_no_force_gap = max(max_no_force_gap, no_force_gap)
    final = rows[-1]
    final_state = env.get_state()
    contacts = set(str(final["contacts"]).split(","))
    target_center_dist = float(np.linalg.norm(final_state.pen_center - target_center))
    axis_cos = float(
        np.clip(
            np.dot(final_state.pen_axis, target_axis)
            / (np.linalg.norm(final_state.pen_axis) * np.linalg.norm(target_axis) + 1e-9),
            -1.0,
            1.0,
        )
    )
    target_axis_error = float(np.arccos(axis_cos))
    phase_error = abs(float(final["planar_rotation_deg"]) - TARGET_PROGRESS_DEG)
    desired = {"thumb", "index", "middle"}
    valid = bool(
        desired.issubset(contacts)
        and "pinky" not in contacts
        and final["thumb_share"] > 0.20
        and final["index_share"] > 0.03
        and final["middle_share"] > 0.03
        and final["ring_share"] < 0.20
        and phase_error < 9.0
        and target_center_dist < 0.012
        and target_axis_error < 0.22
        and final["dz_mm"] > -5.0
        and max_penetration < 2.0
        and final["pen_linear_speed"] < 0.35
        and max_no_force_gap <= 3
    )
    desired_count = len(desired.intersection(contacts))
    score = (
        1800.0 * desired_count
        + 2400.0 * float(desired.issubset(contacts))
        + 600.0 * min(1.0, float(final["thumb_share"]) / 0.35)
        + 450.0 * min(1.0, float(final["middle_share"]) / 0.20)
        - 45.0 * phase_error
        - 90000.0 * target_center_dist
        - 900.0 * target_axis_error
        - 240.0 * max(0.0, max_penetration - 1.0)
        - 500.0 * float("pinky" in contacts)
        - 350.0 * max(0.0, float(final["ring_share"]) - 0.15)
        - 240.0 * max_no_force_gap
        - 300.0 * max(0.0, -float(final["dz_mm"]) - 2.0)
        - 500.0 * max(0.0, float(final["pen_linear_speed"]) - 0.25)
    )
    return {
        "valid": valid,
        "score": float(score),
        "rows": rows,
        "target_center_dist_mm": target_center_dist * 1000.0,
        "target_axis_error_deg": float(np.degrees(target_axis_error)),
        "phase_error_deg": phase_error,
        "max_penetration_mm": max_penetration,
        "max_no_force_gap_steps": max_no_force_gap,
    }


def render_selected(
    env: Any,
    source_state: dict[str, Any],
    action: np.ndarray,
    output_dir: Path,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tail.restore_snapshot(env, source_state)
    start = env.get_state()
    source_frame = m316.overlay_frame(
        env.render(),
        "corrected P4-entry source | index/thumb | middle approach",
    )
    frames = [source_frame]
    rows: list[dict[str, Any]] = []
    for step in range(horizon):
        env.step(action.astype(np.float32))
        row = semantic.metric_row(env, start.pen_center, start.pen_axis, step + 1)
        rows.append(row)
        frames.append(
            m316.overlay_frame(
                env.render(),
                (
                    f"P4 middle recapture {step + 1}/{horizon} | "
                    f"{row['planar_rotation_deg']:+.1f} deg | {row['contacts']} | "
                    f"T/I/M={row['thumb_share']:.2f}/"
                    f"{row['index_share']:.2f}/{row['middle_share']:.2f}"
                ),
            )
        )
    imageio.mimsave(output_dir / "P4_middle_recapture.gif", frames, duration=0.10)
    m316.save_montage(
        [frames[0], frames[max(1, len(frames) // 2)], frames[-1]],
        output_dir / "P4_middle_recapture_keyframes.png",
        cols=3,
    )
    final_state = tail.state_snapshot(env)
    m334.save_final_cache(output_dir / "P4_middle_recapture_entry.npz", final_state)
    return rows[-1], final_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p4_middle_recapture_m3_2_39"
        ),
    )
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--action-delta-limit", type=float, default=0.040)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_state = load_source_state(SOURCE_DIR / "B34_index_thumb_recapture.npz")
    source_seed = json.loads(
        (SOURCE_DIR / "B34_index_thumb_recapture_seed.json").read_text(
            encoding="utf-8"
        )
    )
    target_seed = json.loads(
        (SOURCE_DIR / "dynamic_P4_entry_seed.json").read_text(encoding="utf-8")
    )
    target_center = np.asarray(target_seed["pen_pos"], dtype=np.float64)
    target_axis = quat_to_axis(np.asarray(target_seed["pen_quat"], dtype=np.float64))
    rng = np.random.default_rng(args.seed)
    env = m316.make_env(source_seed, model_args(args), duration=1.0, render=False)
    leaderboard: list[dict[str, Any]] = []
    valid_count = 0
    try:
        for sample in range(args.samples):
            action = make_candidate(rng, sample)
            result = evaluate(
                env,
                source_state,
                action,
                target_center,
                target_axis,
                args.horizon,
            )
            valid_count += int(result["valid"])
            entry = {
                "sample": sample,
                "action": action.tolist(),
                **result,
            }
            leaderboard.append(entry)
            leaderboard.sort(key=lambda item: (item["valid"], item["score"]), reverse=True)
            del leaderboard[20:]
            if (sample + 1) % 5000 == 0:
                best = leaderboard[0]
                print(
                    f"sample={sample + 1} valid={valid_count} "
                    f"best={best['score']:.1f} contacts={best['rows'][-1]['contacts']}",
                    flush=True,
                )
        selected = leaderboard[0]
    finally:
        env.close()

    render_env = m316.make_env(source_seed, model_args(args), duration=1.0, render=True)
    try:
        final_row, final_state = render_selected(
            render_env,
            source_state,
            np.asarray(selected["action"], dtype=np.float64),
            output_dir,
            args.horizon,
        )
    finally:
        render_env.close()
    report = {
        "stage": "M3.2.39 corrected P4 middle recapture shooting",
        "source": str(SOURCE_DIR / "B34_index_thumb_recapture.npz"),
        "target_progress_deg": TARGET_PROGRESS_DEG,
        "samples": args.samples,
        "horizon": args.horizon,
        "valid_count": valid_count,
        "selected_valid": bool(selected["valid"]),
        "selected_action": selected["action"],
        "selected_metrics": {
            key: selected[key]
            for key in (
                "score",
                "target_center_dist_mm",
                "target_axis_error_deg",
                "phase_error_deg",
                "max_penetration_mm",
                "max_no_force_gap_steps",
            )
        },
        "final_row": final_row,
        "leaderboard": leaderboard,
        "artifacts": {
            "gif": str(output_dir / "P4_middle_recapture.gif"),
            "keyframes": str(output_dir / "P4_middle_recapture_keyframes.png"),
            "cache": str(output_dir / "P4_middle_recapture_entry.npz"),
        },
    }
    save_json(output_dir / "search_summary.json", report)
    print(json.dumps({key: report[key] for key in ("valid_count", "selected_valid", "selected_metrics", "final_row")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
