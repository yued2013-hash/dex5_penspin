from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

import build_m3_2_34_continuous_p1_p2_p3 as m334
import build_m3_2_38_corrected_p3_p4_chain as m338
import optimize_m3_2_36_p45_mpc as mpc
import rebuild_m3_2_16_p4_p5_transitions as m316
import search_m3_2_35_dynamic_tail as tail
import search_m3_2_36_semantic_p45b as semantic
import search_m3_2_39_p4_middle_recapture as p439
from evaluate_m3_2_15_short_primitives import quat_to_axis


SOURCE_DIR = p439.SOURCE_DIR
TARGET_PROGRESS_DEG = 56.04638800142325 - 22.467700586492317
FIRST_STEPS = 3
SECOND_STEPS = 5


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def consistent(action: np.ndarray) -> np.ndarray:
    return mpc.enforce_action_consistency(
        np.clip(action, -0.040, 0.040)
    ).astype(np.float64)


def preset_candidate(sample: int) -> tuple[np.ndarray, np.ndarray] | None:
    first = m338.CONTACT_SWITCH_ACTION.copy()
    second = m338.P4_CAPTURE_ACTION.copy()
    if sample == 0:
        return first, second
    if sample == 1:
        return np.zeros(20), np.zeros(20)
    if sample == 2:
        second[:] = 0.0
        return first, second
    if sample == 3:
        first[:] = 0.0
        return first, second
    if sample == 4:
        first[4:8] = 0.0
        second[4:8] = 0.0
        return first, second
    if sample == 5:
        first[4:8] = [-0.020, -0.020, -0.020, -0.020]
        second[4:8] = [-0.020, -0.020, -0.020, -0.020]
        return first, second
    return None


def random_candidate(
    rng: np.random.Generator,
    sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    preset = preset_candidate(sample)
    if preset is not None:
        return consistent(preset[0]), consistent(preset[1])
    first = m338.CONTACT_SWITCH_ACTION.copy()
    second = m338.P4_CAPTURE_ACTION.copy()
    first[0:4] += rng.normal(0.0, 0.013, size=4)
    second[0:4] += rng.normal(0.0, 0.013, size=4)
    mode = sample % 5
    if mode == 0:
        first[4:8] = rng.uniform(-0.040, 0.040, size=4)
        second[4:8] = rng.uniform(-0.040, 0.040, size=4)
    elif mode == 1:
        first[4:8] *= rng.uniform(-1.2, 1.0)
        second[4:8] *= rng.uniform(-1.2, 1.0)
    elif mode == 2:
        first[4:8] = rng.normal(-0.012, 0.016, size=4)
        second[4:8] = rng.normal(-0.012, 0.016, size=4)
    elif mode == 3:
        first[4:8] = rng.normal(0.0, 0.012, size=4)
        second[4:8] = rng.normal(0.0, 0.012, size=4)
    else:
        first[4:8] += rng.normal(0.0, 0.022, size=4)
        second[4:8] += rng.normal(0.0, 0.022, size=4)
    first[7] = first[6]
    second[7] = second[6]
    first[8:12] = np.clip(rng.normal(0.0, 0.010, size=4), -0.018, 0.018)
    second[8:12] = np.clip(rng.normal(0.0, 0.010, size=4), -0.018, 0.018)
    first[11] = first[10]
    second[11] = second[10]
    first[12:16] = 0.0
    second[12:16] = 0.0
    first[16:20] += rng.normal(0.0, 0.012, size=4)
    second[16:20] += rng.normal(0.0, 0.012, size=4)
    return consistent(first), consistent(second)


def evaluate(
    env: Any,
    source_state: dict[str, Any],
    first: np.ndarray,
    second: np.ndarray,
    target_center: np.ndarray,
    target_axis: np.ndarray,
) -> dict[str, Any]:
    tail.restore_snapshot(env, source_state)
    start = env.get_state()
    rows: list[dict[str, Any]] = []
    max_penetration = 0.0
    max_no_force_gap = 0
    no_force_gap = 0
    actions = [first] * FIRST_STEPS + [second] * SECOND_STEPS
    for step, action in enumerate(actions, start=1):
        env.step(action.astype(np.float32))
        row = semantic.metric_row(env, start.pen_center, start.pen_axis, step)
        rows.append(row)
        max_penetration = max(max_penetration, float(row["max_penetration_mm"]))
        supported = sum(
            float(row[f"{finger}_share"])
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        ) > 1e-6
        no_force_gap = 0 if supported else no_force_gap + 1
        max_no_force_gap = max(max_no_force_gap, no_force_gap)
    final = rows[-1]
    state = env.get_state()
    contacts = set(str(final["contacts"]).split(","))
    center_dist = float(np.linalg.norm(state.pen_center - target_center))
    axis_cos = float(
        np.clip(
            np.dot(state.pen_axis, target_axis)
            / (np.linalg.norm(state.pen_axis) * np.linalg.norm(target_axis) + 1e-9),
            -1.0,
            1.0,
        )
    )
    axis_error = float(np.arccos(axis_cos))
    phase_error = abs(float(final["planar_rotation_deg"]) - TARGET_PROGRESS_DEG)
    desired = {"thumb", "index", "middle"}
    desired_count = len(desired.intersection(contacts))
    valid = bool(
        desired.issubset(contacts)
        and "pinky" not in contacts
        and final["thumb_share"] > 0.20
        and final["index_share"] > 0.03
        and final["middle_share"] > 0.03
        and final["ring_share"] < 0.20
        and phase_error < 9.0
        and center_dist < 0.012
        and axis_error < 0.22
        and final["dz_mm"] > -5.0
        and max_penetration < 2.0
        and final["pen_linear_speed"] < 0.35
        and max_no_force_gap <= 3
    )
    score = (
        1800.0 * desired_count
        + 2600.0 * float(desired.issubset(contacts))
        + 650.0 * min(1.0, float(final["thumb_share"]) / 0.35)
        + 500.0 * min(1.0, float(final["middle_share"]) / 0.20)
        - 45.0 * phase_error
        - 90000.0 * center_dist
        - 900.0 * axis_error
        - 260.0 * max(0.0, max_penetration - 1.0)
        - 600.0 * float("pinky" in contacts)
        - 350.0 * max(0.0, float(final["ring_share"]) - 0.15)
        - 240.0 * max_no_force_gap
        - 300.0 * max(0.0, -float(final["dz_mm"]) - 2.0)
        - 500.0 * max(0.0, float(final["pen_linear_speed"]) - 0.25)
    )
    return {
        "valid": valid,
        "score": float(score),
        "rows": rows,
        "target_center_dist_mm": center_dist * 1000.0,
        "target_axis_error_deg": float(np.degrees(axis_error)),
        "phase_error_deg": phase_error,
        "max_penetration_mm": max_penetration,
        "max_no_force_gap_steps": max_no_force_gap,
    }


def render_selected(
    env: Any,
    source_state: dict[str, Any],
    first: np.ndarray,
    second: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tail.restore_snapshot(env, source_state)
    start = env.get_state()
    frames = [
        m316.overlay_frame(
            env.render(),
            "B34 middle handoff | index/middle/thumb | overlap source",
        )
    ]
    rows: list[dict[str, Any]] = []
    actions = [first] * FIRST_STEPS + [second] * SECOND_STEPS
    for step, action in enumerate(actions, start=1):
        env.step(action.astype(np.float32))
        row = semantic.metric_row(env, start.pen_center, start.pen_axis, step)
        rows.append(row)
        frames.append(
            m316.overlay_frame(
                env.render(),
                (
                    f"P4 overlap {step}/8 | {row['planar_rotation_deg']:+.1f} deg | "
                    f"{row['contacts']} | T/I/M={row['thumb_share']:.2f}/"
                    f"{row['index_share']:.2f}/{row['middle_share']:.2f}"
                ),
            )
        )
    imageio.mimsave(output_dir / "P4_overlap_middle_recapture.gif", frames, duration=0.10)
    m316.save_montage(
        [frames[0], frames[FIRST_STEPS], frames[-1]],
        output_dir / "P4_overlap_middle_keyframes.png",
        cols=3,
    )
    final_state = tail.state_snapshot(env)
    m334.save_final_cache(output_dir / "P4_overlap_middle_entry.npz", final_state)
    return rows[-1], final_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p4_overlap_middle_m3_2_39"
        ),
    )
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=240)
    parser.add_argument("--action-delta-limit", type=float, default=0.040)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_state = p439.load_source_state(SOURCE_DIR / "B34_middle_handoff.npz")
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
    env = m316.make_env(source_seed, p439.model_args(args), duration=1.0, render=False)
    leaderboard: list[dict[str, Any]] = []
    valid_count = 0
    try:
        for sample in range(args.samples):
            first, second = random_candidate(rng, sample)
            result = evaluate(
                env,
                source_state,
                first,
                second,
                target_center,
                target_axis,
            )
            valid_count += int(result["valid"])
            entry = {
                "sample": sample,
                "first_action": first.tolist(),
                "second_action": second.tolist(),
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

    render_env = m316.make_env(source_seed, p439.model_args(args), duration=1.0, render=True)
    try:
        final_row, final_state = render_selected(
            render_env,
            source_state,
            np.asarray(selected["first_action"], dtype=np.float64),
            np.asarray(selected["second_action"], dtype=np.float64),
            output_dir,
        )
    finally:
        render_env.close()
    report = {
        "stage": "M3.2.39 overlapping P4 middle recapture shooting",
        "source": str(SOURCE_DIR / "B34_middle_handoff.npz"),
        "target_progress_deg": TARGET_PROGRESS_DEG,
        "samples": args.samples,
        "valid_count": valid_count,
        "selected_valid": bool(selected["valid"]),
        "selected_first_action": selected["first_action"],
        "selected_second_action": selected["second_action"],
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
            "gif": str(output_dir / "P4_overlap_middle_recapture.gif"),
            "keyframes": str(output_dir / "P4_overlap_middle_keyframes.png"),
            "cache": str(output_dir / "P4_overlap_middle_entry.npz"),
        },
    }
    save_json(output_dir / "search_summary.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "valid_count",
                    "selected_valid",
                    "selected_metrics",
                    "final_row",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
