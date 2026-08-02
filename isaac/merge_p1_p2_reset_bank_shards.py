from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge P1/P2 reset-bank accepted shards.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    input_paths = [resolved(path) for path in args.inputs]
    output = resolved(args.output)
    report_path = resolved(args.report)

    arrays_by_key: dict[str, list[np.ndarray]] = {}
    shard_rows: list[dict[str, object]] = []
    required = {
        "source_qpos",
        "source_qvel",
        "source_target_qpos",
        "source_policy_action",
        "source_teacher_actions",
        "target_qpos",
        "target_qvel",
        "phase_ids",
    }

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"{path} is missing required arrays: {missing}")
            row_count = int(data["phase_ids"].shape[0])
            phases, counts = np.unique(data["phase_ids"], return_counts=True)
            shard_rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "rows": row_count,
                    "phase_counts": {str(int(phase)): int(count) for phase, count in zip(phases, counts)},
                }
            )
            keys = list(data.files)
            for key in keys:
                arrays_by_key.setdefault(key, []).append(np.asarray(data[key]))

    merged: dict[str, np.ndarray] = {}
    all_keys = sorted(arrays_by_key)
    total_rows = sum(row["rows"] for row in shard_rows)
    for key in all_keys:
        rows = arrays_by_key[key]
        if len(rows) != len(input_paths):
            # Optional provenance fields may not exist in older shards.
            continue
        try:
            merged[key] = np.concatenate(rows, axis=0)
        except ValueError as exc:
            raise ValueError(f"cannot concatenate key {key}: {[row.shape for row in rows]}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)

    phases, counts = np.unique(merged["phase_ids"], return_counts=True)
    report = {
        "output": str(output.relative_to(PROJECT_ROOT)),
        "total_rows": int(total_rows),
        "phase_counts": {str(int(phase)): int(count) for phase, count in zip(phases, counts)},
        "shards": shard_rows,
        "keys": sorted(merged),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_RESET_BANK_MERGE=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
