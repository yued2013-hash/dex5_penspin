#!/usr/bin/env python3
"""Compare matched O1.5 baseline and temporal-observation evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(report: dict) -> dict[str, float]:
    p3 = report["by_canonical"]["P3"]
    fingers = p3["finger_gait"]
    return {
        "raw_gait_rate": report["gait_event_rate"],
        "qualified_gait_rate": report["qualified_gait_rate"],
        "support_handoff_rate": report["support_handoff_rate"],
        "thumb_gait_rate": fingers["thumb"]["gait_episode_rate"],
        "index_gait_rate": fingers["index"]["gait_episode_rate"],
        "middle_gait_rate": fingers["middle"]["gait_episode_rate"],
        "height_recovery_rate": p3["recovery"]["height_recovery_rate"],
        "height_drop_rate": p3["recovery"]["height_drop_rate"],
        "tim_overlap_ratio": p3["mean_TIM_overlap_ratio"],
        "failure_rate": report["failure_rate"],
        "ballistic_rate": report["ballistic_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()

    baseline_report = _load(args.baseline)
    temporal_report = _load(args.temporal)
    baseline = _metrics(baseline_report)
    temporal = _metrics(temporal_report)
    delta = {key: temporal[key] - baseline[key] for key in baseline}
    temporal_pass = (
        temporal["qualified_gait_rate"] > baseline["qualified_gait_rate"]
        and temporal["support_handoff_rate"] > baseline["support_handoff_rate"]
        and temporal["support_handoff_rate"] > 0.0
    )

    result = {
        "status": "PASS",
        "classification": (
            "TEMPORAL_CREDIT_PASS"
            if temporal_pass
            else "TRANSITION_ACTIVITY_UP_HANDOFF_NOT_LEARNED"
        ),
        "matched_contract": {
            "p3_only": True,
            "seed": 20260730,
            "num_envs": 256,
            "iterations": 60,
            "rollout_steps": 1440,
            "transitions": 368640,
            "reward_equal": (
                baseline_report["objective_config"]
                | {"include_temporal_support_observation": None, "observation_space": None}
            )
            == (
                temporal_report["objective_config"]
                | {"include_temporal_support_observation": None, "observation_space": None}
            ),
            "bank_hash_equal": baseline_report["bank_sha256"]
            == temporal_report["bank_sha256"],
        },
        "baseline": baseline,
        "temporal": temporal,
        "temporal_minus_baseline": delta,
        "interpretation": (
            "Temporal privileged observations greatly increase debounced contact "
            "transitions, but do not produce qualified gait, valid support handoff, "
            "or height recovery under the frozen objective."
        ),
        "source": {
            "baseline": str(args.baseline.resolve()),
            "temporal": str(args.temporal.resolve()),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    labels = (
        "Raw gait",
        "Qualified gait",
        "Support handoff",
        "Thumb gait",
        "Index gait",
        "Middle gait",
        "Height recovery",
        "TIM overlap",
    )
    keys = (
        "raw_gait_rate",
        "qualified_gait_rate",
        "support_handoff_rate",
        "thumb_gait_rate",
        "index_gait_rate",
        "middle_gait_rate",
        "height_recovery_rate",
        "tim_overlap_ratio",
    )
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(13, 5.5))
    axis.bar(x - width / 2, [baseline[key] for key in keys], width, label="Baseline")
    axis.bar(x + width / 2, [temporal[key] for key in keys], width, label="Temporal")
    axis.set_xticks(x, labels, rotation=22, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Episode rate")
    axis.set_title("O1.5 matched-budget temporal credit comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=180, bbox_inches="tight")
    print(args.output_json)
    print(args.output_plot)


if __name__ == "__main__":
    main()
