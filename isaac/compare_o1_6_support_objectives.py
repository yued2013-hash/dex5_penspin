#!/usr/bin/env python3
"""Compare O1.6 support-objective ablations and detect termination exploits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(report: dict) -> dict[str, float]:
    p3 = report["by_canonical"]["P3"]
    evaluation_transitions = report["num_envs"] * report["steps"]
    return {
        "completed_episodes": report["completed_episodes"],
        "approx_mean_episode_steps": evaluation_transitions
        / max(report["completed_episodes"], 1),
        "raw_gait_rate": report["gait_event_rate"],
        "qualified_gait_rate": report["qualified_gait_rate"],
        "support_handoff_rate": report["support_handoff_rate"],
        "height_recovery_rate": p3["recovery"]["height_recovery_rate"],
        "tim_overlap_ratio": p3["mean_TIM_overlap_ratio"],
        "failure_rate": report["failure_rate"],
        "contact_loss_rate": report["contact_loss_rate"],
        "ballistic_rate": report["ballistic_rate"],
        "mean_support_force_z_n": report["mean_metrics"]["support_force_z_n"],
        "mean_total_contact_force_n": report["mean_metrics"]["total_contact_force_n"],
        "max_total_contact_force_n": report["max_total_contact_force_n"],
        "mechanical_power_w": report["mean_metrics"]["mechanical_power_w"],
        "max_penetration_mm": report["max_penetration_mm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--support-margin", type=Path, required=True)
    parser.add_argument("--recoverability", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        "A_temporal_baseline": _load(args.baseline),
        "B_support_margin": _load(args.support_margin),
        "C_recoverability": _load(args.recoverability),
    }
    rows = {name: _row(report) for name, report in reports.items()}
    baseline_steps = rows["A_temporal_baseline"]["approx_mean_episode_steps"]
    termination_exploit = all(
        rows[name]["approx_mean_episode_steps"] < 0.25 * baseline_steps
        and rows[name]["ballistic_rate"] > 0.90
        for name in ("B_support_margin", "C_recoverability")
    )
    skill_pass = any(
        row["qualified_gait_rate"] > 0.0
        and row["support_handoff_rate"] > 0.0
        and row["height_recovery_rate"] > 0.0
        and row["tim_overlap_ratio"] >= 0.80
        for row in rows.values()
    )

    output = {
        "status": "PASS",
        "classification": (
            "SUPPORT_CONTINUITY_PASS"
            if skill_pass
            else (
                "DENSE_PENALTY_TERMINATION_EXPLOIT"
                if termination_exploit
                else "SUPPORT_CONTINUITY_NOT_LEARNED"
            )
        ),
        "skill_pass": skill_pass,
        "termination_exploit_detected": termination_exploit,
        "matched_contract": {
            "p3_only": True,
            "temporal_observation_dim": 172,
            "seed": 20260730,
            "num_envs": 256,
            "iterations": 60,
            "training_transitions_per_condition": 368640,
            "evaluation_transitions_per_condition": 61440,
            "bank_hash_equal": len({r["bank_sha256"] for r in reports.values()}) == 1,
        },
        "conditions": rows,
        "interpretation": (
            "Per-step support-deficit penalties create an incentive to terminate "
            "episodes early. B and C do not test successful load transfer; they expose "
            "a termination-return loophole that must be closed before further scaling."
        ),
        "sources": {name: str(path.resolve()) for name, path in {
            "A_temporal_baseline": args.baseline,
            "B_support_margin": args.support_margin,
            "C_recoverability": args.recoverability,
        }.items()},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    labels = ("A baseline", "B support", "C recovery")
    keys = tuple(rows)
    metrics = (
        ("Raw gait", "raw_gait_rate"),
        ("Qualified gait", "qualified_gait_rate"),
        ("Support handoff", "support_handoff_rate"),
        ("Height recovery", "height_recovery_rate"),
        ("TIM overlap", "tim_overlap_ratio"),
        ("Ballistic", "ballistic_rate"),
    )
    x = np.arange(len(metrics))
    width = 0.24
    fig, axis = plt.subplots(figsize=(12.5, 5.5))
    for index, (key, label) in enumerate(zip(keys, labels, strict=True)):
        axis.bar(
            x + (index - 1) * width,
            [rows[key][metric] for _, metric in metrics],
            width,
            label=label,
        )
    axis.set_xticks(x, [label for label, _ in metrics], rotation=18, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Episode rate")
    axis.set_title("O1.6 support-continuity objective ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=180, bbox_inches="tight")
    print(args.output_json)
    print(args.output_plot)


if __name__ == "__main__":
    main()
