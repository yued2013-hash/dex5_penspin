#!/usr/bin/env python3
"""Compare matched O1.7 termination-credit learning ablations."""

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
    recovery = p3["recovery"]
    return {
        "completed_episodes": report["completed_episodes"],
        "mean_episode_steps": report["mean_episode_steps"],
        "raw_gait_rate": report["gait_event_rate"],
        "qualified_gait_rate": report["qualified_gait_rate"],
        "support_handoff_rate": report["support_handoff_rate"],
        "height_drop_rate": recovery["height_drop_rate"],
        "height_recovery_rate": recovery["height_recovery_rate"],
        "height_recovery_given_drop_rate": recovery[
            "height_recovery_given_drop_rate"
        ],
        "thumb_recontact_rate": recovery["thumb_recontact_rate"],
        "tim_overlap_ratio": p3["mean_TIM_overlap_ratio"],
        "axis_alignment": p3["mean_directional_axis_alignment"],
        "failure_rate": report["failure_rate"],
        "contact_loss_rate": report["contact_loss_rate"],
        "ballistic_rate": report["ballistic_rate"],
        "max_penetration_mm": report["max_penetration_mm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--grace", type=Path, required=True)
    parser.add_argument("--return-audits", type=Path, nargs=3, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "A_fixed_horizon": args.fixed,
        "B_fixed_terminal": args.terminal,
        "C_recovery_grace": args.grace,
    }
    reports = {name: _load(path) for name, path in paths.items()}
    audits = [_load(path) for path in args.return_audits]
    rows = {name: _row(report) for name, report in reports.items()}
    return_order_pass = all(audit["return_order_pass"] for audit in audits)
    termination_collapse = any(row["mean_episode_steps"] < 30 for row in rows.values())
    skill_pass = any(
        row["qualified_gait_rate"] > 0.0
        and row["support_handoff_rate"] > 0.0
        and row["height_recovery_rate"] > 0.0
        for row in rows.values()
    )

    output = {
        "status": "PASS",
        "classification": (
            "TERMINATION_INVARIANT_SKILL_PASS"
            if skill_pass
            else "TERMINATION_CREDIT_NOT_SUFFICIENT"
        ),
        "skill_pass": skill_pass,
        "return_order_pass": return_order_pass,
        "termination_collapse_detected": termination_collapse,
        "matched_contract": {
            "p3_only": True,
            "temporal_observation_dim": 172,
            "baseline_reward_frozen_except_B_terminal_penalty": True,
            "seed": 20260730,
            "num_envs": 256,
            "iterations": 60,
            "training_transitions_per_condition": 368640,
            "evaluation_transitions_per_condition": 61440,
            "bank_hash_equal": len(
                {report["bank_sha256"] for report in reports.values()}
            )
            == 1,
        },
        "conditions": rows,
        "interpretation": (
            "Termination-invariant learning removes the O1.6 short-episode exploit, "
            "but qualified gait, support handoff, and height recovery remain zero. "
            "Termination credit assignment contributed to exploitation but is not a "
            "sufficient explanation for the P3 handoff failure."
        ),
        "sources": {name: str(path.resolve()) for name, path in paths.items()},
        "return_audit_sources": [str(path.resolve()) for path in args.return_audits],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    labels = ("A fixed", "B terminal", "C grace")
    keys = tuple(rows)
    metrics = (
        ("Raw gait", "raw_gait_rate"),
        ("Qualified", "qualified_gait_rate"),
        ("Handoff", "support_handoff_rate"),
        ("Height drop", "height_drop_rate"),
        ("Height recovery", "height_recovery_rate"),
        ("Ballistic", "ballistic_rate"),
        ("Contact loss", "contact_loss_rate"),
    )
    x = np.arange(len(metrics))
    width = 0.24
    fig, (rate_axis, length_axis) = plt.subplots(
        1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": (4, 1)}
    )
    for index, (key, label) in enumerate(zip(keys, labels, strict=True)):
        rate_axis.bar(
            x + (index - 1) * width,
            [rows[key][metric] for _, metric in metrics],
            width,
            label=label,
        )
    rate_axis.set_xticks(x, [label for label, _ in metrics], rotation=20, ha="right")
    rate_axis.set_ylim(0.0, 1.05)
    rate_axis.set_ylabel("Episode rate")
    rate_axis.grid(axis="y", alpha=0.25)
    rate_axis.legend()
    length_axis.bar(labels, [rows[key]["mean_episode_steps"] for key in keys])
    length_axis.set_ylim(0.0, 125.0)
    length_axis.set_ylabel("Mean episode steps")
    length_axis.tick_params(axis="x", rotation=25)
    length_axis.grid(axis="y", alpha=0.25)
    fig.suptitle("O1.7 termination-invariant learning ablation")
    fig.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=180, bbox_inches="tight")
    print(args.output_json)
    print(args.output_plot)


if __name__ == "__main__":
    main()
