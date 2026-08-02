from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Build a reset-filtered v9 bundle with bank-specific teacher actions."
)
parser.add_argument("--source_bundle", type=Path, required=True)
parser.add_argument("--search_report", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max_teacher_speed", type=float, default=20.0)
parser.add_argument("--settle_steps", type=int, default=6)
parser.add_argument("--drive_steps", type=int, default=10)
parser.add_argument("--brake_steps", type=int, default=10)
args = parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    source_path = resolve(args.source_bundle)
    report_paths = [resolve(path) for path in args.search_report]
    output_path = resolve(args.output)

    best_by_bank: dict[int, dict] = {}
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report.get("accepted_candidates", []):
            bank_id = row.get("reset_bank_id")
            if bank_id is None or row["max_angular_speed_rad_s"] > args.max_teacher_speed:
                continue
            previous = best_by_bank.get(int(bank_id))
            rank = (
                row["max_angular_speed_rad_s"],
                -row["max_phase_hold_steps"],
                -row["score"],
            )
            if previous is None or rank < previous["rank"]:
                best_by_bank[int(bank_id)] = {
                    "row": row,
                    "rank": rank,
                    "report": report_path,
                }

    if not best_by_bank:
        raise ValueError("No accepted bank-specific teacher actions passed the speed gate")

    bank_ids = np.asarray(sorted(best_by_bank), dtype=np.int64)
    with np.load(source_path) as data:
        payload = {key: np.asarray(data[key])[bank_ids].copy() for key in data.files}

    horizon = args.settle_steps + args.drive_steps + args.brake_steps
    teacher = np.zeros((len(bank_ids), horizon, 20), dtype=np.float32)
    drive_start = args.settle_steps
    brake_start = drive_start + args.drive_steps
    selected_rows = []
    for output_id, bank_id in enumerate(bank_ids.tolist()):
        selected = best_by_bank[bank_id]
        row = selected["row"]
        teacher[output_id, drive_start:brake_start] = np.asarray(
            row["drive_action"], dtype=np.float32
        )
        teacher[output_id, brake_start:] = np.asarray(
            row["brake_action"], dtype=np.float32
        )
        selected_rows.append(
            {
                "source_bank_id": bank_id,
                "candidate_id": row["candidate_id"],
                "report": str(selected["report"].relative_to(ROOT)),
                "max_phase_deg": row["max_phase_deg"],
                "max_phase_hold_steps": row["max_phase_hold_steps"],
                "max_angular_speed_rad_s": row["max_angular_speed_rad_s"],
                "max_penetration_mm": row["max_penetration_mm"],
            }
        )

    payload["source_teacher_actions"] = teacher
    payload["paired_source_bank_ids"] = bank_ids
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    summary = {
        "output": str(output_path.relative_to(ROOT)),
        "source_bundle": str(source_path.relative_to(ROOT)),
        "rows": len(bank_ids),
        "max_teacher_speed_rad_s": args.max_teacher_speed,
        "schedule": {
            "settle_steps": args.settle_steps,
            "drive_steps": args.drive_steps,
            "brake_steps": args.brake_steps,
        },
        "source_reports": [str(path.relative_to(ROOT)) for path in report_paths],
        "selected": selected_rows,
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("output", "rows", "schedule")}, indent=2))


if __name__ == "__main__":
    main()
