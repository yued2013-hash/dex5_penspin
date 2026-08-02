from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Build a reset bundle from teacher-compatible banks.")
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--teacher", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def main() -> None:
    with np.load(args.teacher) as teacher:
        bank_ids = np.unique(np.asarray(teacher["bank_ids"], dtype=np.int64))
    with np.load(args.source) as source:
        payload = {
            key: np.asarray(source[key])[bank_ids].copy()
            for key in source.files
            if np.asarray(source[key]).ndim > 0
            and len(np.asarray(source[key])) == len(source["source_qpos"])
        }
    payload["parent_bank_ids"] = bank_ids
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    summary = {
        "source": str(args.source),
        "teacher": str(args.teacher),
        "source_rows": int(len(np.load(args.source)["source_qpos"])),
        "reset_rows": int(len(bank_ids)),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
