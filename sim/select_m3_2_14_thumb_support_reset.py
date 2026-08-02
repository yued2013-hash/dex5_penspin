from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from build_m3_2_9_photo_five_phase_debug import set_state
from tune_m3_2_10_photo_phase_hand_pen_contacts import render_seed
from tune_m3_2_14_p1_stable_reset import (
    candidate_quat,
    evaluate_candidate,
    make_env,
    plot_trace,
    write_stable_seed_dir,
)


def parse_list(value: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(value), dtype=np.float64)


def load_rank(csv_path: Path, rank: int) -> dict[str, str]:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rank < 1 or rank > len(rows):
        raise ValueError(f"rank must be in [1, {len(rows)}], got {rank}")
    return rows[rank - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/p1_stable_reset_middle_m3_2_14b/stable_seed_dir",
    )
    parser.add_argument(
        "--csv",
        default="dex5_penspin/reports/manus_mujoco_presets/p1_stable_reset_four_contact_m3_2_14c/p1_stable_reset_sweep.csv",
    )
    parser.add_argument("--rank", type=int, default=14)
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/p1_thumb_support_reset_m3_2_14d",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--validate-duration", type=float, default=6.0)
    parser.add_argument("--render-every", type=int, default=10)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_seed_path = seed_dir / "P1_four_finger_clamp_hand_pen_tuned_seed.json"
    base_seed = json.loads(source_seed_path.read_text(encoding="utf-8"))
    row = load_rank(csv_path, args.rank)
    q_delta = parse_list(row["q_delta"])
    pen_offset = parse_list(row["pen_offset"])
    ddeg = float(row["ddeg"])
    dz_axis = float(row["axis_z_delta"])

    env = make_env(base_seed, duration=args.duration, render=True)
    render_row, traces = evaluate_candidate(
        env,
        base_seed,
        q_delta=q_delta,
        pen_offset=pen_offset,
        ddeg=ddeg,
        dz_axis=dz_axis,
        duration=args.duration,
        render=True,
        render_every=args.render_every,
    )
    env.close()

    pos = np.asarray(base_seed["pen_pos"], dtype=np.float64) + pen_offset
    quat = candidate_quat(np.asarray(base_seed["pen_quat"], dtype=np.float64), ddeg, dz_axis)
    thumb_seed = {
        **base_seed,
        "stage": "M3.2.14d P1 thumb-support reset selected from sweep",
        "hand_qpos": render_row["hand_qpos"],
        "target_qpos": render_row["hand_qpos"],
        "pen_pos": pos.astype(float).tolist(),
        "pen_quat": quat.astype(float).tolist(),
        "contact_fingers_initial": sorted(render_row["initial_contacts"].split(",")) if render_row["initial_contacts"] else [],
        "thumb_support_selection": {
            "source_csv": str(csv_path),
            "source_rank": int(args.rank),
            "source_seed": str(source_seed_path),
            "interpretation": "thumb/index/middle support reset; ring support is intentionally not dominant",
            "q_delta": q_delta.astype(float).tolist(),
            "pen_offset": pen_offset.astype(float).tolist(),
            "ddeg": ddeg,
            "axis_z_delta": dz_axis,
            "metrics_2s": {
                key: render_row[key]
                for key in [
                    "survival_seconds",
                    "terminated",
                    "exact_ratio",
                    "near_ratio",
                    "index_ratio",
                    "middle_ratio",
                    "ring_ratio",
                    "thumb_ratio",
                    "lost_ratio",
                    "xy_drift",
                    "z_delta",
                    "q_delta_max",
                    "pen_offset_norm",
                    "contact_counter",
                    "exact_contact_counter",
                ]
            },
        },
    }

    # Revalidate for a longer zero-action hold without rendering.
    val_env = make_env(thumb_seed, duration=args.validate_duration, render=False)
    validation, _ = evaluate_candidate(
        val_env,
        base_seed,
        q_delta=q_delta,
        pen_offset=pen_offset,
        ddeg=ddeg,
        dz_axis=dz_axis,
        duration=args.validate_duration,
        render=False,
        render_every=args.render_every,
    )
    val_env.close()

    best_path = output_dir / "P1_thumb_support_reset_seed.json"
    best_path.write_text(json.dumps(thumb_seed, indent=2, ensure_ascii=False), encoding="utf-8")
    stable_dir = write_stable_seed_dir(seed_dir, output_dir, thumb_seed)

    if traces is not None:
        imageio.mimsave(output_dir / "thumb_support_hold.gif", list(traces["frames"]), duration=args.render_every / 100.0)
        plot_trace(output_dir / "thumb_support_trace.png", traces, render_row)

    image_env = make_env(thumb_seed, duration=args.duration, render=True)
    image_env.reset(seed=0)
    set_state(image_env, np.asarray(thumb_seed["hand_qpos"]), np.asarray(thumb_seed["pen_pos"]), np.asarray(thumb_seed["pen_quat"]))
    imageio.imwrite(output_dir / "thumb_support_initial.png", render_seed(image_env, thumb_seed))
    image_env.close()

    summary = {
        "stage": "M3.2.14d P1 thumb-support reset",
        "output_dir": str(output_dir),
        "best_seed": str(best_path),
        "stable_seed_dir": str(stable_dir),
        "source_rank": int(args.rank),
        "source_csv": str(csv_path),
        "metrics_2s": thumb_seed["thumb_support_selection"]["metrics_2s"],
        "metrics_validate": {
            key: validation[key]
            for key in [
                "survival_seconds",
                "terminated",
                "exact_ratio",
                "near_ratio",
                "index_ratio",
                "middle_ratio",
                "ring_ratio",
                "thumb_ratio",
                "lost_ratio",
                "xy_drift",
                "z_delta",
                "contact_counter",
                "exact_contact_counter",
            ]
        },
        "artifacts": {
            "initial": str(output_dir / "thumb_support_initial.png"),
            "gif": str(output_dir / "thumb_support_hold.gif"),
            "trace": str(output_dir / "thumb_support_trace.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.14d P1 Thumb-Support Reset",
        "",
        "This is an alternate P1 reset selected for visual/debugging work where thumb support should be explicit.",
        "",
        "Interpretation:",
        "",
        "```text",
        "thumb / index / middle dominate the support.",
        "ring is intentionally not the main support.",
        "This reset trades some tripod-shelf robustness for thumb-side support.",
        "```",
        "",
        "Summary:",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
