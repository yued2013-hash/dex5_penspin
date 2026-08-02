from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Select audited rows from a reset-matched teacher bundle.")
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rows", type=int, nargs="+", required=True)
args = parser.parse_args()


def main() -> None:
    selected = np.asarray(args.rows, dtype=np.int64)
    payload: dict[str, np.ndarray] = {}
    with np.load(args.input) as data:
        row_count = len(data["source_qpos"])
        if selected.min() < 0 or selected.max() >= row_count:
            raise ValueError(f"Selected rows must be within [0, {row_count})")
        for key in data.files:
            value = np.asarray(data[key])
            payload[key] = value[selected] if value.ndim > 0 and value.shape[0] == row_count else value
    payload["audited_source_row"] = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "input_rows": row_count,
        "selected_rows": selected.tolist(),
        "output_rows": len(selected),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
