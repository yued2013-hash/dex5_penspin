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


SOURCE_DIR = p439.SOURCE_DIR


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def candidate(rng: np.random.Generator, sample: int) -> np.ndarray:
    if sample == 0:
        return np.zeros(20, dtype=np.float64)
    if sample == 1:
        return mpc.enforce_action_consistency(-m338.P4_CAPTURE_ACTION)
    if sample == 2:
        return m338.P4_CAPTURE_ACTION.copy()
    action = np.zeros(20, dtype=np.float64)
    mode = sample % 4
    if mode == 0:
        action[0:4] = rng.uniform(-0.040, 0.040, size=4)
        action[16:20] = rng.uniform(-0.040, 0.040, size=4)
    elif mode == 1:
        action[0:4] = m338.P4_CAPTURE_ACTION[0:4] + rng.normal(0.0, 0.022, size=4)
        action[16:20] = m338.P4_CAPTURE_ACTION[16:20] + rng.normal(0.0, 0.022, size=4)
    elif mode == 2:
        action[0:4] = -m338.P4_CAPTURE_ACTION[0:4] + rng.normal(0.0, 0.018, size=4)
        action[16:20] = -m338.P4_CAPTURE_ACTION[16:20] + rng.normal(0.0, 0.018, size=4)
    else:
        action[0:4] = rng.normal(0.0, 0.018, size=4)
        action[16:20] = rng.normal(0.0, 0.018, size=4)
    action[4:8] = rng.normal(-0.004, 0.012, size=4)
    action[7] = action[6]
    action[8:12] = np.clip(rng.normal(0.0, 0.008, size=4), -0.014, 0.014)
    action[11] = action[10]
    action[12:16] = 0.0
    return mpc.enforce_action_consistency(
        np.clip(action, -0.040, 0.040)
    ).astype(np.float64)


def evaluate(
    env: Any,
    source_state: dict[str, Any],
    action: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    tail.restore_snapshot(env, source_state)
    start = env.get_state()
    rows: list[dict[str, Any]] = []
    max_penetration = 0.0
    support_steps = 0
    pair_steps = 0
    for step in range(horizon):
        env.step(action.astype(np.float32))
        row = semantic.metric_row(env, start.pen_center, start.pen_axis, step + 1)
        rows.append(row)
        contacts = set(str(row["contacts"]).split(","))
        support_steps += int(bool({"thumb", "index"}.intersection(contacts)))
        pair_steps += int({"thumb", "index"}.issubset(contacts))
        max_penetration = max(max_penetration, float(row["max_penetration_mm"]))
    final = rows[-1]
    contacts = set(str(final["contacts"]).split(","))
    pair = {"thumb", "index"}.issubset(contacts)
    valid = bool(
        pair
        and "pinky" not in contacts
        and final["thumb_share"] > 0.10
        and final["index_share"] > 0.05
        and final["dz_mm"] > -4.0
        and final["pen_linear_speed"] < 0.32
        and max_penetration < 2.0
        and support_steps == horizon
    )
    score = (
        1800.0 * support_steps
        + 2600.0 * pair_steps
        + 3600.0 * float(pair)
        + 700.0 * min(1.0, float(final["thumb_share"]) / 0.35)
        + 500.0 * min(1.0, float(final["index_share"]) / 0.25)
        - 1700.0 * float(final["pen_linear_speed"])
        - 18.0 * float(final["pen_angular_speed"])
        - 260.0 * max(0.0, -float(final["dz_mm"]) - 1.0)
        - 260.0 * max(0.0, max_penetration - 1.0)
        - 450.0 * float("pinky" in contacts)
        - 240.0 * max(0.0, float(final["ring_share"]) - 0.15)
        - 45.0 * max(0.0, float(final["middle_tip_distance_mm"]) - 14.0)
    )
    return {
        "valid": valid,
        "score": float(score),
        "rows": rows,
        "support_steps": support_steps,
        "pair_steps": pair_steps,
        "max_penetration_mm": max_penetration,
    }


def render_selected(
    env: Any,
    source_state: dict[str, Any],
    action: np.ndarray,
    horizon: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tail.restore_snapshot(env, source_state)
    start = env.get_state()
    frames = [m316.overlay_frame(env.render(), "P4 flight-entry | brake source")]
    rows: list[dict[str, Any]] = []
    for step in range(horizon):
        env.step(action.astype(np.float32))
        row = semantic.metric_row(env, start.pen_center, start.pen_axis, step + 1)
        rows.append(row)
        frames.append(
            m316.overlay_frame(
                env.render(),
                (
                    f"thumb/index brake {step + 1}/{horizon} | {row['contacts']} | "
                    f"v={row['pen_linear_speed']:.3f} | dz={row['dz_mm']:+.1f}mm"
                ),
            )
        )
    imageio.mimsave(output_dir / "P4_thumb_index_brake.gif", frames, duration=0.12)
    m316.save_montage(
        [frames[0], frames[max(1, len(frames) // 2)], frames[-1]],
        output_dir / "P4_thumb_index_brake_keyframes.png",
        cols=3,
    )
    state = tail.state_snapshot(env)
    m334.save_final_cache(output_dir / "P4_thumb_index_brake_entry.npz", state)
    return rows[-1], state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "dex5_penspin/reports/manus_mujoco_presets/"
            "p4_thumb_index_brake_m3_2_39"
        ),
    )
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=241)
    parser.add_argument("--action-delta-limit", type=float, default=0.040)
    parser.add_argument("--actuator-kp", type=float, default=650.0)
    parser.add_argument("--actuator-kv", type=float, default=32.0)
    parser.add_argument("--tip-proxy-radius", type=float, default=0.0085)
    parser.add_argument("--tip-proxy-friction", type=float, default=2.4)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_state = p439.load_source_state(SOURCE_DIR / "dynamic_P4_entry.npz")
    source_seed = json.loads(
        (SOURCE_DIR / "dynamic_P4_entry_seed.json").read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(args.seed)
    env = m316.make_env(source_seed, p439.model_args(args), duration=1.0, render=False)
    leaderboard: list[dict[str, Any]] = []
    valid_count = 0
    try:
        for sample in range(args.samples):
            action = candidate(rng, sample)
            result = evaluate(env, source_state, action, args.horizon)
            valid_count += int(result["valid"])
            entry = {"sample": sample, "action": action.tolist(), **result}
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
            np.asarray(selected["action"], dtype=np.float64),
            args.horizon,
            output_dir,
        )
    finally:
        render_env.close()
    report = {
        "stage": "M3.2.39 P4 flight-entry thumb/index brake",
        "source": str(SOURCE_DIR / "dynamic_P4_entry.npz"),
        "samples": args.samples,
        "horizon": args.horizon,
        "valid_count": valid_count,
        "selected_valid": bool(selected["valid"]),
        "selected_action": selected["action"],
        "selected_metrics": {
            "score": selected["score"],
            "support_steps": selected["support_steps"],
            "pair_steps": selected["pair_steps"],
            "max_penetration_mm": selected["max_penetration_mm"],
        },
        "final_row": final_row,
        "leaderboard": leaderboard,
        "artifacts": {
            "gif": str(output_dir / "P4_thumb_index_brake.gif"),
            "keyframes": str(output_dir / "P4_thumb_index_brake_keyframes.png"),
            "cache": str(output_dir / "P4_thumb_index_brake_entry.npz"),
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
