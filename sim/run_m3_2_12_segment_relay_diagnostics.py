from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from run_m3_2_11_keyframe_relay import evaluate_relay, load_keyframes, plot_trace


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "segment",
        "rank",
        "score",
        "segment_seconds",
        "start_hold_seconds",
        "end_hold_seconds",
        "tracking_gain",
        "action_delta_limit",
        "hard_hold_hand",
        "survival_seconds",
        "total_duration",
        "terminated",
        "progress_ratio",
        "peak_progress_ratio",
        "final_axis_progress",
        "peak_axis_progress",
        "target_progress",
        "monotonic_ratio",
        "desired_contact_ratio",
        "active_contact_ratio",
        "xy_drift",
        "z_delta",
        "min_height",
        "max_penetration",
        "contact_counter",
        "segment_top_contacts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_trace(output_dir: Path, segment_name: str, row: dict[str, Any], traces: dict[str, Any] | None) -> None:
    if traces is None:
        return
    imageio.mimsave(output_dir / f"{segment_name}_best.gif", list(traces["frames"]), fps=25)
    plot_trace(output_dir / f"{segment_name}_best_trace.png", traces, row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/segment_relay_diagnostics_m3_2_12",
    )
    parser.add_argument("--render-every", type=int, default=5)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes = load_keyframes(seed_dir)

    all_rows: list[dict[str, Any]] = []
    best_by_segment: list[dict[str, Any]] = []
    for idx in range(len(keyframes) - 1):
        segment_keyframes = [keyframes[idx], keyframes[idx + 1]]
        segment_name = f"P{keyframes[idx]['phase']}_to_P{keyframes[idx + 1]['phase']}"
        rows: list[dict[str, Any]] = []
        configs: list[dict[str, float | bool]] = []
        for segment_seconds in [0.10, 0.14, 0.18, 0.24, 0.32, 0.45]:
            for start_hold_seconds in [0.00, 0.06, 0.12]:
                for tracking_gain in [0.45, 0.75, 1.10, 1.60]:
                    for action_delta_limit in [0.010, 0.018, 0.026, 0.035]:
                        for hard_hold_hand in [False, True]:
                            configs.append(
                                {
                                    "segment_seconds": float(segment_seconds),
                                    "start_hold_seconds": float(start_hold_seconds),
                                    "end_hold_seconds": 0.06,
                                    "tracking_gain": float(tracking_gain),
                                    "action_delta_limit": float(action_delta_limit),
                                    "hard_hold_hand": bool(hard_hold_hand),
                                }
                            )
        for config in configs:
            row, _ = evaluate_relay(segment_keyframes, config, render=False, render_every=args.render_every)
            row["segment"] = segment_name
            rows.append(row)
        rows.sort(key=lambda x: x["score"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        best_config = rows[0]
        best_row, best_trace = evaluate_relay(segment_keyframes, best_config, render=True, render_every=args.render_every)
        best_row["segment"] = segment_name
        best_row["rank"] = 1
        best_row["score"] = best_config["score"]
        save_trace(output_dir, segment_name, best_row, best_trace)
        best_by_segment.append(best_row)
        all_rows.extend(rows)

    write_csv(output_dir / "segment_relay_all_sweeps.csv", all_rows)
    write_csv(output_dir / "segment_relay_best.csv", best_by_segment)
    summary = {
        "stage": "M3.2.12 segment-wise relay diagnostics",
        "seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "best_by_segment": best_by_segment,
        "artifacts": {
            "all_sweeps": str(output_dir / "segment_relay_all_sweeps.csv"),
            "best_csv": str(output_dir / "segment_relay_best.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.12 Segment Relay Diagnostics",
        "",
        "| segment | hard hold | survival | progress ratio | active contact | desired contact | terminated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_by_segment:
        readme.append(
            f"| {row['segment']} | {row['hard_hold_hand']} | {row['survival_seconds']:.2f} | "
            f"{row['progress_ratio']:.3f} | {row['active_contact_ratio']:.3f} | "
            f"{row['desired_contact_ratio']:.3f} | {row['terminated']} |"
        )
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
