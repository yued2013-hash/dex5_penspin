from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from s0_11_student_policy import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--closed-loop-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    evaluation = json.loads(args.closed_loop_eval.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    records = evaluation["episode_records"]
    taxonomy_total = sum(evaluation["failure_taxonomy"].values())
    success_count = sum(bool(row["success"]) for row in records)
    rotation_count = sum(
        float(row["peak_quaternion_rotation_deg"]) >= 30.0 for row in records
    )
    hold_count = sum(int(row["max_hold_steps"]) >= 9 for row in records)
    checks = {
        "checkpoint_schema": checkpoint.get("schema")
        == "dex5-s0.11-student-checkpoint-v1",
        "checkpoint_hash_matches_training_report": sha256(args.checkpoint)
        == training["checkpoint_sha256"],
        "checkpoint_dataset_hash_matches_training_report": checkpoint[
            "dataset_sha256"
        ]
        == training["dataset_sha256"],
        "non_privileged_checkpoint": checkpoint["privileged_observation_used"]
        is False,
        "observation_dimension_340": checkpoint["architecture"][
            "observation_dimension"
        ]
        == 340,
        "action_dimension_20": checkpoint["architecture"]["action_dimension"]
        == 20,
        "overfit_gate_pass": training["overfit_smoke"]["pass"] is True,
        "frozen_episode_counts": training["episode_counts"]
        == {"overfit": 100, "train": 891, "validation": 112, "test": 197},
        "evaluation_complete": evaluation["status"]
        == "PASS_EVALUATION_COMPLETE",
        "evaluation_197_episodes": len(records) == evaluation["episodes"] == 197,
        "taxonomy_covers_all_episodes": taxonomy_total == 197,
        "success_count_recomputed": success_count == evaluation["successes"],
        "rotation_rate_recomputed": abs(
            rotation_count / 197.0 - evaluation["quaternion_rotation_30deg_rate"]
        )
        < 1.0e-12,
        "hold_rate_recomputed": abs(
            hold_count / 197.0 - evaluation["hold_150ms_rate"]
        )
        < 1.0e-12,
        "policy_did_not_read_oracle": evaluation[
            "oracle_observation_read_by_policy"
        ]
        is False,
        "no_hardware_commands": evaluation["hardware_commands_issued"] is False,
        "penetration_below_0_5mm": evaluation["penetration"]["maximum_mm"]
        <= 0.5,
    }
    report = {
        "schema": "dex5-s0.11-output-validation-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "successes": success_count,
            "episodes": len(records),
            "rotation_30deg_count": rotation_count,
            "hold_150ms_count": hold_count,
            "taxonomy_total": taxonomy_total,
            "maximum_penetration_mm": evaluation["penetration"]["maximum_mm"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
