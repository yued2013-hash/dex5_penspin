from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Validate exported strict braked-P4 trajectories.")
parser.add_argument("input")
parser.add_argument("--output", required=True)
parser.add_argument("--min_phase_deg", type=float, default=50.0)
parser.add_argument("--max_phase_deg", type=float, default=60.0)
parser.add_argument("--max_angular_speed", type=float, default=12.0)
parser.add_argument("--hold_steps", type=int, default=6)
parser.add_argument("--max_penetration_mm", type=float, default=4.0)
parser.add_argument("--min_thumb_force_n", type=float, default=0.03)
parser.add_argument("--max_hand_rmse", type=float, default=0.22)
args = parser.parse_args()

data = np.load(args.input)
lengths = data["lengths"]
control_dt_s = float(data.get("control_dt_s", np.asarray(1.0 / 60.0)))
valid_rows: list[bool] = []
terminal_phase: list[float] = []
terminal_speed: list[float] = []
terminal_hold_ms: list[float] = []
max_penetration_mm: list[float] = []
max_drift_mm: list[float] = []

for index, length_value in enumerate(lengths):
    length = int(length_value)
    start = length - args.hold_steps
    phase = data["phase_deg"][index, start:length]
    speed = data["angular_speed_rad_s"][index, start:length]
    contact_force = data["contact_force_n"][index, start:length, :3]
    contacts = contact_force > 0.03
    drift = data["drift_m"][index, start:length]
    hand_rmse = data["hand_rmse_rad"][index, start:length]
    penetration = data["max_penetration_m"][index, start:length]
    hold_count = data["hold_count"][index, start:length]
    valid = bool(
        length >= args.hold_steps
        and np.all(phase >= args.min_phase_deg)
        and np.all(phase <= args.max_phase_deg)
        and np.all(speed <= args.max_angular_speed)
        and np.all(contacts)
        and np.all(contact_force[:, 0] >= args.min_thumb_force_n)
        and np.all(hand_rmse < args.max_hand_rmse)
        and np.all(drift < 0.015)
        and np.all(penetration <= args.max_penetration_mm / 1000.0)
        and int(hold_count[-1]) >= args.hold_steps
    )
    valid_rows.append(valid)
    terminal_phase.append(float(phase[-1]))
    terminal_speed.append(float(speed[-1]))
    terminal_hold_ms.append(float(hold_count[-1]) * 1000.0 * control_dt_s)
    max_penetration_mm.append(float(np.max(penetration)) * 1000.0)
    max_drift_mm.append(float(np.max(drift)) * 1000.0)

result = {
    "input": str(Path(args.input).resolve()),
    "trajectory_count": int(len(lengths)),
    "valid_trajectory_count": int(sum(valid_rows)),
    "valid_rate": float(np.mean(valid_rows)),
    "control_dt_s": control_dt_s,
    "max_penetration_limit_mm": args.max_penetration_mm,
    "min_thumb_force_n": args.min_thumb_force_n,
    "max_hand_rmse_rad": args.max_hand_rmse,
    "length_steps": {"min": int(lengths.min()), "max": int(lengths.max())},
    "terminal_phase_deg": {
        "min": float(np.min(terminal_phase)),
        "mean": float(np.mean(terminal_phase)),
        "max": float(np.max(terminal_phase)),
    },
    "terminal_angular_speed_rad_s": {
        "min": float(np.min(terminal_speed)),
        "mean": float(np.mean(terminal_speed)),
        "max": float(np.max(terminal_speed)),
    },
    "terminal_hold_ms": {
        "min": float(np.min(terminal_hold_ms)),
        "mean": float(np.mean(terminal_hold_ms)),
        "max": float(np.max(terminal_hold_ms)),
    },
    "max_penetration_mm": float(np.max(max_penetration_mm)),
    "max_drift_mm": float(np.max(max_drift_mm)),
}

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
