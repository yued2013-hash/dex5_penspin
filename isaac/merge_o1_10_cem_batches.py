#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _taxonomy(records: list[dict]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        key = record["primary_failure"]
        result[key] = result.get(key, 0) + 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    hashes = {report["bank_sha256"] for report in reports}
    if len(hashes) != 1:
        raise RuntimeError("Batch reports use different frozen-bank hashes")
    configuration_keys = (
        "population",
        "iterations",
        "elite_fraction",
        "waypoints",
        "action_mode",
        "active_action_dimensions",
        "initial_std_rad",
        "max_offset_rad",
        "horizons_steps",
        "control_dt_s",
    )
    for key in configuration_keys:
        values = [report[key] for report in reports]
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError(f"Batch configuration mismatch for {key}: {values}")

    source_ids = [bank_id for report in reports for bank_id in report["source_bank_ids"]]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Duplicate bank IDs across O1.10 batches")
    selected = [
        record for report in reports for record in report["selected_state_results"]
    ]
    selected.sort(key=lambda record: source_ids.index(record["bank_id"]))
    feasible = sum(record["feasible"] for record in selected)
    margins = np.asarray([record["handoff_margin"] for record in selected])

    merged_horizons = []
    for horizon_index in range(len(reports[0]["horizons"])):
        rows = [
            record
            for report in reports
            for record in report["horizons"][horizon_index]["states"]
        ]
        feasible_horizon = sum(record["feasible"] for record in rows)
        merged_horizons.append(
            {
                "horizon_steps": reports[0]["horizons"][horizon_index][
                    "horizon_steps"
                ],
                "horizon_ms": reports[0]["horizons"][horizon_index][
                    "horizon_ms"
                ],
                "feasible_states": feasible_horizon,
                "feasible_rate": feasible_horizon / len(rows),
                "failure_taxonomy": _taxonomy(rows),
                "best_handoff_margin": float(
                    max(record["handoff_margin"] for record in rows)
                ),
                "states": rows,
            }
        )

    first = reports[0]
    merged = {
        "status": "PASS",
        "artifact_role": "batched_reachability_summary_only_not_teacher_or_training_data",
        "condition": args.condition,
        "conclusion": (
            "STRICT_FEASIBLE_FOUND"
            if feasible
            else "NO_STRICT_FEASIBLE_WITHIN_SCREENING_BOUNDS"
        ),
        "bank_sha256": first["bank_sha256"],
        "batch_files": [str(path.resolve()) for path in args.inputs],
        "batching": "four P3 states per Isaac process; CEM budget is per state",
        "states": len(selected),
        "population": first["population"],
        "iterations": first["iterations"],
        "elite_fraction": first["elite_fraction"],
        "waypoints": first["waypoints"],
        "action_mode": first["action_mode"],
        "active_action_indices": first["active_action_indices"],
        "active_action_dimensions": first["active_action_dimensions"],
        "initial_std_rad": first["initial_std_rad"],
        "max_offset_rad": first["max_offset_rad"],
        "horizons_steps": first["horizons_steps"],
        "control_dt_s": first["control_dt_s"],
        "source_bank_ids": source_ids,
        "strict_feasible_count": feasible,
        "strict_feasible_rate": feasible / len(selected),
        "failure_taxonomy": _taxonomy(selected),
        "handoff_margin": {
            "best": float(margins.max()),
            "median": float(np.median(margins)),
            "minimum": float(margins.min()),
        },
        "selected_state_results": selected,
        "horizons": merged_horizons,
        "trajectory_exported": False,
        "teacher_generated": False,
        "ppo_trained": False,
        "reward_modified": False,
        "o0_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "O1_10_BATCH_MERGE="
        + json.dumps(
            {
                "condition": args.condition,
                "strict_feasible_count": feasible,
                "failure_taxonomy": merged["failure_taxonomy"],
            }
        )
    )


if __name__ == "__main__":
    main()
