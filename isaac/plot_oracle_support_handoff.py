#!/usr/bin/env python3
"""Plot matched O1.4 support-handoff diagnostics from fixed-horizon traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
COLORS = {"zero": "#2d6a4f", "O1.2": "#d97706", "O1.3": "#b42318"}


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--o1-2", dest="o1_2", type=Path, required=True)
    parser.add_argument("--o1-3", dest="o1_3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-dt", type=float, default=1.0 / 60.0)
    args = parser.parse_args()

    traces = {
        "zero": _load(args.zero),
        "O1.2": _load(args.o1_2),
        "O1.3": _load(args.o1_3),
    }
    steps = min(trace["height_delta_m"].shape[0] for trace in traces.values())
    time_s = np.arange(steps) * args.control_dt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for label, trace in traces.items():
        color = COLORS[label]
        height_mm = np.median(trace["height_delta_m"][:steps], axis=1) * 1000.0
        tim_force = np.median(trace["contact_force_norm_n"][:steps, :, :3].sum(axis=2), axis=1)
        vertical_force = np.median(trace["net_force_proxy_n"][:steps, :, 2], axis=1)
        lateral_force = np.median(
            np.linalg.norm(trace["net_force_proxy_n"][:steps, :, :2], axis=2), axis=1
        )
        tim_count = np.median(trace["debounced_contact"][:steps, :, :3].sum(axis=2), axis=1)

        axes[0, 0].plot(time_s, height_mm, label=label, color=color)
        axes[0, 1].plot(time_s, tim_force, label=label, color=color)
        axes[1, 0].plot(time_s, vertical_force, label=label, color=color)
        axes[1, 0].plot(time_s, lateral_force, color=color, linestyle="--", alpha=0.75)
        axes[1, 1].plot(time_s, tim_count, label=label, color=color)

    axes[0, 0].axhline(-5.0, color="#555", linestyle=":", linewidth=1)
    axes[0, 0].set_title("Pen height relative to reset")
    axes[0, 0].set_ylabel("Height (mm)")
    axes[0, 1].set_title("TIM force magnitude sum")
    axes[0, 1].set_ylabel("Force proxy (N)")
    axes[1, 0].set_title("Net force direction (solid: vertical, dashed: lateral)")
    axes[1, 0].set_ylabel("Force proxy (N)")
    axes[1, 1].set_title("Debounced TIM contacts")
    axes[1, 1].set_ylabel("Contact count")
    axes[1, 1].set_ylim(-0.1, 3.1)
    for axis in axes[1, :]:
        axis.set_xlabel("Time (s)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.22)
        axis.legend(loc="best")

    fig.suptitle("O1.4 P3 Support Handoff: matched 256-episode median traces")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
