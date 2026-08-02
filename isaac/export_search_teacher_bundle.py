from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Pair searched bridge actions with their exact reset rows.")
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--search", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max_rows", type=int, default=512)
parser.add_argument("--min_hold_steps", type=int, default=1)
args = parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    bundle_path = resolve(args.bundle)
    search_path = resolve(args.search)
    output_path = resolve(args.output)
    report = json.loads(search_path.read_text(encoding="utf-8"))
    accepted = [
        row for row in report["accepted"] if row["max_hold_steps"] >= args.min_hold_steps
    ][: args.max_rows]
    if not accepted:
        raise RuntimeError("Search report contains no accepted teacher actions")
    bank_ids = np.asarray([row["reset_bank_id"] for row in accepted], dtype=np.int64)
    switch_step = int(report["switch_step"])
    steps = int(report["steps"])
    drive = np.asarray([row["drive_action"] for row in accepted], dtype=np.float32)
    brake = np.asarray([row["brake_action"] for row in accepted], dtype=np.float32)
    teacher = np.empty((len(accepted), steps, 20), dtype=np.float32)
    teacher[:, :switch_step] = drive[:, None]
    teacher[:, switch_step:] = brake[:, None]

    payload: dict[str, np.ndarray] = {}
    with np.load(bundle_path) as data:
        base_rows = len(data["source_qpos"])
        for key in data.files:
            value = np.asarray(data[key])
            payload[key] = value[bank_ids] if value.ndim > 0 and value.shape[0] == base_rows else value
    payload["source_teacher_actions"] = teacher
    payload["search_candidate_ids"] = np.asarray(
        [row["candidate_id"] for row in accepted], dtype=np.int64
    )
    payload["search_scores"] = np.asarray([row["score"] for row in accepted], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    summary = {
        "output": str(output_path),
        "bundle": str(bundle_path),
        "search": str(search_path),
        "rows": len(accepted),
        "teacher_steps": steps,
        "switch_step": switch_step,
        "unique_reset_rows": int(len(np.unique(bank_ids))),
        "max_penetration_mm": float(max(row["max_penetration_mm"] for row in accepted)),
        "max_hold_steps": int(max(row["max_hold_steps"] for row in accepted)),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
