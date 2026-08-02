from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def pad_teacher_actions(array: np.ndarray, horizon: int) -> np.ndarray:
    if array.shape[1] == horizon:
        return array
    padded = np.empty((len(array), horizon, array.shape[2]), dtype=array.dtype)
    padded[:, : array.shape[1]] = array
    padded[:, array.shape[1] :] = array[:, -1:, :]
    return padded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix a repeated core prefill bank with wider real continuous entries."
    )
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--core-repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.core_repeat < 1:
        parser.error("--core-repeat must be at least 1")

    with np.load(args.core) as data:
        core = {name: data[name] for name in data.files}
    with np.load(args.wide) as data:
        wide = {name: data[name] for name in data.files}

    common = sorted(set(core) & set(wide))
    if "source_teacher_actions" not in common:
        raise KeyError("Both bundles must contain source_teacher_actions")
    horizon = max(
        core["source_teacher_actions"].shape[1],
        wide["source_teacher_actions"].shape[1],
    )

    payload: dict[str, np.ndarray] = {}
    for name in common:
        core_value = core[name]
        wide_value = wide[name]
        if name == "source_teacher_actions":
            core_value = pad_teacher_actions(core_value, horizon)
            wide_value = pad_teacher_actions(wide_value, horizon)
        payload[name] = np.concatenate(
            ([core_value] * args.core_repeat) + [wide_value], axis=0
        )

    core_rows = len(core["source_qpos"]) * args.core_repeat
    payload["curriculum_source"] = np.concatenate(
        (
            np.zeros(core_rows, dtype=np.int64),
            np.ones(len(wide["source_qpos"]), dtype=np.int64),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"output={args.output}")
    print(f"core_rows={core_rows}")
    print(f"wide_rows={len(wide['source_qpos'])}")
    print(f"total_rows={len(payload['source_qpos'])}")
    print(f"core_fraction={core_rows / len(payload['source_qpos']):.6f}")


if __name__ == "__main__":
    main()
