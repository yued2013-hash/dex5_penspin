#!/usr/bin/env python3
"""Plot the summary-only O1.8 handoff reachability audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    horizons = report["horizons"]
    selected = sorted(
        report["selected_state_results"], key=lambda row: row["handoff_margin"]
    )
    taxonomy = report["failure_taxonomy"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)

    labels = [f'{row["horizon_ms"]:.0f} ms' for row in horizons]
    rates = [100.0 * row["feasible_rate"] for row in horizons]
    axes[0].bar(labels, rates, color=["#3478b9", "#2f9e73", "#7868a6"])
    axes[0].set_ylim(0, 55)
    axes[0].set_ylabel("Feasible states (%)")
    axes[0].set_title("Feasibility by horizon")
    for index, rate in enumerate(rates):
        axes[0].text(index, rate + 1.2, f"{rate:.1f}%", ha="center")

    failure_labels = list(taxonomy)
    failure_counts = [taxonomy[label] for label in failure_labels]
    colors = ["#2f9e73" if label == "feasible" else "#ca5b4b" for label in failure_labels]
    axes[1].barh(failure_labels, failure_counts, color=colors)
    axes[1].set_xlabel("P3 reset states")
    axes[1].set_title("Best-horizon outcome")
    axes[1].set_xlim(0, max(failure_counts) + 1)

    state_labels = [str(row["bank_id"]) for row in selected]
    margins = [row["handoff_margin"] for row in selected]
    margin_colors = ["#2f9e73" if row["feasible"] else "#ca5b4b" for row in selected]
    axes[2].barh(state_labels, margins, color=margin_colors)
    axes[2].axvline(0.0, color="#202020", linewidth=1)
    axes[2].set_xlabel("Strict handoff margin")
    axes[2].set_ylabel("O0 bank ID")
    axes[2].set_title("Best candidate per reset")

    fig.suptitle("O1.8 P3 Handoff Reachability Audit", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
