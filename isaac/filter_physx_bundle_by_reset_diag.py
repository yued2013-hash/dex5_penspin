from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Filter a PhysX reset bundle using reset diagnostics.")
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--diagnostic", type=Path, required=True, nargs="+")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max_penetration_mm", type=float, choices=(0.25, 0.5), default=0.25)
args = parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    source_path = resolve(args.source)
    diagnostic_paths = [resolve(path) for path in args.diagnostic]
    output_path = resolve(args.output)
    key = "penetration_over_0_25mm_bank_ids" if args.max_penetration_mm == 0.25 else "penetration_over_0_5mm_bank_ids"
    excluded: set[int] = set()
    for diagnostic_path in diagnostic_paths:
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        for trace in diagnostic["traces"]:
            excluded.update(int(value) for value in trace[key])

    with np.load(source_path) as data:
        row_count = len(data["source_qpos"])
        keep = np.asarray([row not in excluded for row in range(row_count)])
        payload = {
            name: np.asarray(data[name])[keep].copy()
            for name in data.files
        }
    payload["original_source_bank_ids"] = np.flatnonzero(keep).astype(np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    summary = {
        "output": str(output_path.relative_to(ROOT)),
        "source": str(source_path.relative_to(ROOT)),
        "diagnostics": [str(path.relative_to(ROOT)) for path in diagnostic_paths],
        "max_penetration_mm": args.max_penetration_mm,
        "source_rows": row_count,
        "kept_rows": int(keep.sum()),
        "excluded_rows": sorted(excluded),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
