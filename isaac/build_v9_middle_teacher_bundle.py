from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Attach a fixed middle-clear teacher to every P4 reset.")
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--teacher", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=72)
parser.add_argument("--pinch-steps", type=int, default=12)
args = parser.parse_args()


def main() -> None:
    with np.load(args.source) as source:
        payload = {key: np.asarray(source[key]).copy() for key in source.files}
        rows = len(source["source_qpos"])
    with np.load(args.teacher) as teacher:
        pinch = np.asarray(teacher["pinch_actions"][0], dtype=np.float32)
        clear = np.asarray(teacher["clear_actions"][0], dtype=np.float32)
    sequence = np.repeat(clear[None, :], args.steps, axis=0)
    sequence[: args.pinch_steps] = pinch
    payload["source_teacher_actions"] = np.repeat(
        sequence[None, :, :], rows, axis=0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    summary = {
        "source": str(args.source),
        "teacher": str(args.teacher),
        "rows": rows,
        "steps": args.steps,
        "pinch_steps": args.pinch_steps,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
