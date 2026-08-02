from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Merge and validate current-v9 continuous teacher trajectories.")
parser.add_argument("--inputs", nargs="+", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--report", required=True)
parser.add_argument("--max_trajectories", type=int, default=200)
args = parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


datasets = []
for input_path in args.inputs:
    with np.load(resolve(input_path)) as data:
        datasets.append({key: np.asarray(data[key]).copy() for key in data.files})

trajectory_keys = {
    key
    for key, value in datasets[0].items()
    if value.ndim >= 1 and value.shape[0] == len(datasets[0]["lengths"])
}
lengths = np.concatenate([data["lengths"] for data in datasets])
selected_count = min(args.max_trajectories, len(lengths))
lengths = lengths[:selected_count]
max_length = int(lengths.max())
payload: dict[str, np.ndarray] = {}

for key in trajectory_keys:
    arrays = [data[key] for data in datasets]
    if key == "lengths":
        payload[key] = lengths
        continue
    if arrays[0].ndim >= 2 and arrays[0].shape[1] >= int(datasets[0]["lengths"].max()):
        sample_shape = arrays[0].shape[2:]
        merged = np.zeros((selected_count, max_length, *sample_shape), dtype=arrays[0].dtype)
        output_row = 0
        for data, array in zip(datasets, arrays, strict=True):
            for input_row, length in enumerate(data["lengths"]):
                if output_row >= selected_count:
                    break
                merged[output_row, : int(length)] = array[input_row, : int(length)]
                output_row += 1
            if output_row >= selected_count:
                break
        payload[key] = merged
    else:
        payload[key] = np.concatenate(arrays, axis=0)[:selected_count]

for key in ("control_dt_s", "stage_names", "continuous_without_intermediate_reset"):
    payload[key] = datasets[0][key]

output = resolve(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output, **payload)

terminal_phase = []
terminal_omega = []
max_penetration = []
max_global_drift = []
terminal_hold_valid = []
for row, length in enumerate(lengths):
    length = int(length)
    hold_slice = slice(length - 18, length)
    terminal_phase.append(float(payload["phase_deg"][row, length - 1]))
    terminal_omega.append(float(payload["angular_speed_rad_s"][row, length - 1]))
    max_penetration.append(float(payload["max_penetration_m"][row, :length].max()))
    max_global_drift.append(float(payload["global_drift_m"][row, :length].max()))
    forces = payload["contact_force_n"][row, hold_slice, :3]
    hold_valid = (
        np.all((payload["phase_deg"][row, hold_slice] >= 50.0) & (payload["phase_deg"][row, hold_slice] <= 60.0))
        and np.all(payload["angular_speed_rad_s"][row, hold_slice] <= 12.0)
        and np.all(forces > 0.03)
        and np.all(payload["drift_m"][row, hold_slice] < 0.015)
        and np.all(payload["max_penetration_m"][row, hold_slice] <= 0.0005)
    )
    terminal_hold_valid.append(bool(hold_valid))

report = {
    "status": "current-v9 continuous teacher dataset validated",
    "inputs": [str(resolve(path)) for path in args.inputs],
    "available_trajectories": int(sum(len(data["lengths"]) for data in datasets)),
    "selected_trajectories": selected_count,
    "continuous_without_intermediate_reset": bool(payload["continuous_without_intermediate_reset"]),
    "trajectory_length_steps": {
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "max": int(lengths.max()),
    },
    "terminal_phase_deg": {
        "min": min(terminal_phase),
        "mean": float(np.mean(terminal_phase)),
        "max": max(terminal_phase),
    },
    "terminal_angular_speed_rad_s": {
        "min": min(terminal_omega),
        "mean": float(np.mean(terminal_omega)),
        "max": max(terminal_omega),
    },
    "max_penetration_mm": {
        "mean": float(np.mean(max_penetration) * 1000.0),
        "max": float(max(max_penetration) * 1000.0),
    },
    "max_global_drift_mm": {
        "mean": float(np.mean(max_global_drift) * 1000.0),
        "max": float(max(max_global_drift) * 1000.0),
    },
    "terminal_18_step_hold_valid": {
        "count": int(sum(terminal_hold_valid)),
        "rate": float(np.mean(terminal_hold_valid)),
    },
}
report_path = resolve(args.report)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
