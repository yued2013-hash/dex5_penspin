from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a v9 PhysX reset cache by physical safety margins."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_penetration_mm", type=float, required=True)
    parser.add_argument("--max_angular_speed", type=float, required=True)
    parser.add_argument("--max_linear_speed", type=float, default=float("inf"))
    parser.add_argument("--max_hand_speed", type=float, default=float("inf"))
    parser.add_argument("--max_force_n", type=float, required=True)
    args = parser.parse_args()

    with np.load(args.source) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}

    count = len(payload["source_qpos"])
    penetration_mm = payload["source_max_penetration_m"] * 1000.0
    source_qvel = payload["source_qvel"]
    angular_speed = payload.get(
        "source_angular_speed_rad_s",
        np.linalg.norm(source_qvel[:, 23:26], axis=1),
    )
    linear_speed = np.linalg.norm(source_qvel[:, 20:23], axis=1)
    hand_speed = np.linalg.norm(source_qvel[:, :20], axis=1)
    peak_force = payload["source_contact_force_n"].max(axis=1)
    selected = (
        (penetration_mm <= args.max_penetration_mm)
        & (angular_speed <= args.max_angular_speed)
        & (linear_speed <= args.max_linear_speed)
        & (hand_speed <= args.max_hand_speed)
        & (peak_force <= args.max_force_n)
    )
    if not np.any(selected):
        raise RuntimeError("No reset rows passed the physical safety margins")

    filtered = {
        key: value[selected]
        if value.ndim > 0 and len(value) == count
        else value
        for key, value in payload.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **filtered)
    report = {
        "source": args.source,
        "output": str(output),
        "source_rows": count,
        "selected_rows": int(selected.sum()),
        "selection_rate": float(selected.mean()),
        "margins": {
            "max_penetration_mm": args.max_penetration_mm,
            "max_angular_speed_rad_s": args.max_angular_speed,
            "max_linear_speed_m_s": args.max_linear_speed,
            "max_hand_speed_rad_s": args.max_hand_speed,
            "max_force_n": args.max_force_n,
        },
        "selected_maxima": {
            "penetration_mm": float(penetration_mm[selected].max()),
            "angular_speed_rad_s": float(angular_speed[selected].max()),
            "linear_speed_m_s": float(linear_speed[selected].max()),
            "hand_speed_rad_s": float(hand_speed[selected].max()),
            "force_n": float(peak_force[selected].max()),
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
