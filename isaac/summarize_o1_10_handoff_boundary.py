#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expanded-search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    expanded = json.loads(args.expanded_search.read_text(encoding="utf-8"))
    if baseline["bank_sha256"] != expanded["bank_sha256"]:
        raise RuntimeError("O1.10 inputs use different O0 banks")

    selected = baseline["selected_state_results"]
    near = [
        record
        for record in selected
        if record["tim_gait"] and record["primary_failure"] == "axis_misaligned"
    ]
    boundary = {
        "temporal_horizon": {
            "measured": "100, 200, 300 ms",
            "strict_feasible_count": baseline["feasible_states"],
            "boundary": "no strict feasible state at or below 300 ms",
            "400_600_ms": "NOT_MEASURED_SIMULATOR_THROUGHPUT",
        },
        "action_space": {
            "measured": "20 joint residual targets, three waypoints",
            "strict_feasible_count": baseline["feasible_states"],
            "TIM_only_small_cooperation": "NOT_MEASURED_SIMULATOR_THROUGHPUT",
        },
        "action_limit": {
            "measured": "+/-0.18 rad",
            "strict_feasible_count": baseline["feasible_states"],
            "plus_minus_0_30_rad": "NOT_MEASURED_SIMULATOR_THROUGHPUT",
        },
        "canonical_entry": {
            "measured_entries": baseline["source_bank_ids"],
            "strict_feasible_count": baseline["feasible_states"],
            "near_feasible_entries": [record["bank_id"] for record in near],
            "boundary": (
                "bank 996 completes a safe TIM transition but misses the axis "
                "gate; the other 11 best candidates do not pass strict handoff"
            ),
        },
    }
    report = {
        "status": "PARTIAL_BLOCKED_BY_SIMULATOR_THROUGHPUT",
        "artifact_role": "feasibility_boundary_summary_not_teacher_or_training_data",
        "conclusion": "BOUNDARY_NOT_CLOSED",
        "bank_sha256": baseline["bank_sha256"],
        "strict_gate": baseline["feasibility_gate"],
        "current_CEM_baseline": {
            "states": baseline["states"],
            "population": baseline["population"],
            "iterations": baseline["iterations"],
            "horizons_steps": baseline["horizons_steps"],
            "strict_feasible_count": baseline["feasible_states"],
            "strict_feasible_rate": baseline["feasible_rate"],
            "failure_taxonomy": baseline["failure_taxonomy"],
            "handoff_margin": baseline["handoff_margin"],
        },
        "search_budget_sensitivity": {
            "population": expanded["population"],
            "iterations": expanded["iterations"],
            "strict_feasible_count": expanded["feasible_states"],
            "strict_feasible_rate": expanded["feasible_rate"],
            "failure_taxonomy": expanded["failure_taxonomy"],
            "handoff_margin": expanded["handoff_margin"],
        },
        "feasibility_boundary": boundary,
        "runtime_blocker": {
            "symptom": (
                "New O1.10 Isaac runs entered an abnormally slow PhysX path; "
                "even three targeted entries at 600 ms did not complete after "
                "more than one hour and were terminated without an artifact."
            ),
            "interpretation": (
                "An aborted or timed-out condition is NOT a zero-feasibility result."
            ),
            "partial_files_promoted": False,
        },
        "constraints": {
            "ppo_trained": False,
            "reward_modified": False,
            "teacher_generated": False,
            "o0_modified": False,
            "trajectory_exported": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    horizons = baseline["horizons"]
    axes[0, 0].bar(
        [f'{row["horizon_ms"]:.0f} ms' for row in horizons],
        [row["feasible_states"] for row in horizons],
        color="#477fb3",
    )
    axes[0, 0].set_ylim(0, 12)
    axes[0, 0].set_ylabel("Strict feasible states / 12")
    axes[0, 0].set_title("Measured temporal boundary")

    taxonomy = baseline["failure_taxonomy"]
    axes[0, 1].barh(list(taxonomy), list(taxonomy.values()), color="#cc5b4b")
    axes[0, 1].set_xlabel("P3 entries")
    axes[0, 1].set_title("Baseline failure taxonomy")

    ordered = sorted(selected, key=lambda record: record["handoff_margin"])
    axes[1, 0].barh(
        [str(record["bank_id"]) for record in ordered],
        [record["handoff_margin"] for record in ordered],
        color=["#e7a43b" if record in near else "#cc5b4b" for record in ordered],
    )
    axes[1, 0].axvline(0.0, color="black", linewidth=1)
    axes[1, 0].set_xlabel("Strict handoff margin")
    axes[1, 0].set_ylabel("O0 bank ID")
    axes[1, 0].set_title("Canonical-entry boundary")

    labels = ["<=300 ms", "TIM-only", "+/-0.30 rad", "400-600 ms"]
    status = [1, 0, 0, 0]
    colors = ["#cc5b4b", "#8d99a8", "#8d99a8", "#8d99a8"]
    axes[1, 1].barh(labels, status, color=colors)
    axes[1, 1].set_xlim(0, 1.15)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_title("Boundary evidence status")
    axes[1, 1].text(1.02, 0, "measured: 0/12", va="center")
    for index in range(1, 4):
        axes[1, 1].text(0.02, index, "NOT MEASURED", va="center", color="#333333")
    fig.suptitle("O1.10 Handoff Feasibility Boundary Audit", fontsize=15)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)
    print(
        "O1_10_BOUNDARY="
        + json.dumps(
            {
                "status": report["status"],
                "baseline_strict_feasible": baseline["feasible_states"],
                "boundary_closed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
