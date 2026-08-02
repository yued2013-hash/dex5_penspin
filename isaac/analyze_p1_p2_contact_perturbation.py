from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Analyze physical contact loss after perturbation.")
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--distance-mm", type=float, default=2.45)
parser.add_argument("--contact-force-n", type=float, default=0.03)
args = parser.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def main() -> None:
    input_path = args.input.resolve()
    with np.load(input_path, allow_pickle=False) as data:
        accepted = np.asarray(data["accepted_trajectory"], dtype=np.bool_)
        eligible = np.asarray(data["accepted_step"], dtype=np.bool_)
        ring_force = np.asarray(
            data["contact_force_thumb_index_middle_ring_pinky"][:, :, 3],
            dtype=np.float32,
        )
        normal_force = np.asarray(
            data["ring_contact_normal_force_signed"], dtype=np.float32
        )
        tangential_force = np.asarray(
            data["ring_contact_tangential_force"], dtype=np.float32
        )
        slip_speed = np.asarray(
            data["ring_contact_tangential_slip_speed"], dtype=np.float32
        )
        normal_speed = np.asarray(
            data["ring_contact_normal_relative_speed"], dtype=np.float32
        )
        target_distance = np.linalg.norm(
            np.asarray(data["target_condition"], dtype=np.float32)[:, :, :3], axis=-1
        ) * 10.0

    max_consecutive = np.zeros(eligible.shape[0], dtype=np.int64)
    exits: Counter[str] = Counter()
    before_exit_rows: list[tuple[int, int]] = []
    exit_rows: list[tuple[int, int]] = []
    for row, mask in enumerate(eligible):
        running = 0
        best = 0
        for value in mask:
            running = running + 1 if value else 0
            best = max(best, running)
        max_consecutive[row] = best
        true_steps = np.flatnonzero(mask)
        if best != 1 or true_steps.size == 0:
            continue
        step = int(true_steps[0])
        if step + 1 >= mask.size:
            exits["end_of_rollout"] += 1
            continue
        next_step = step + 1
        contact_ok = ring_force[row, next_step] >= float(args.contact_force_n)
        distance_ok = target_distance[row, next_step] <= float(args.distance_mm)
        if not contact_ok and not distance_ok:
            reason = "contact_and_distance_exit"
        elif not contact_ok:
            reason = "contact_lost"
        elif not distance_ok:
            reason = "distance_exit"
        else:
            reason = "other_physical_gate"
        exits[reason] += 1
        before_exit_rows.append((row, step))
        exit_rows.append((row, next_step))

    def gather(values: np.ndarray, indices: list[tuple[int, int]]) -> np.ndarray:
        return np.asarray([values[row, step] for row, step in indices])

    metrics = {
        "ring_force_n": ring_force,
        "normal_force_signed_n": normal_force,
        "tangential_force_n": tangential_force,
        "tangential_slip_speed_m_s": slip_speed,
        "normal_relative_speed_m_s": normal_speed,
        "target_distance_mm": target_distance,
    }
    summary = {
        "input": str(input_path),
        "trajectories": int(eligible.shape[0]),
        "accepted_h2": int(accepted.sum()),
        "max_consecutive_counts": {
            str(value): int((max_consecutive == value).sum())
            for value in np.unique(max_consecutive)
        },
        "first_exit_reason_counts_for_one_step_frontier": dict(exits),
        "one_step_frontier_rows": len(before_exit_rows),
        "before_exit": {
            name: stats(gather(values, before_exit_rows)) for name, values in metrics.items()
        },
        "first_exit": {
            name: stats(gather(values, exit_rows)) for name, values in metrics.items()
        },
        "interpretation_note": (
            "Signed normal force uses the pen-axis radial direction. Tangential slip is the "
            "fingertip velocity relative to the nearest rigid pen-axis point."
        ),
        "ppo_started": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_CONTACT_PERTURBATION=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
