from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def normalized_features(data: dict[str, np.ndarray]) -> np.ndarray:
    qpos = data["source_qpos"]
    qvel = data["source_qvel"]
    # Weight the hand geometry most strongly, while retaining pen pose and
    # incoming velocity so the widening follows the physical entry manifold.
    return np.concatenate(
        (
            qpos[:, :20],
            2.0 * qpos[:, 20:23],
            qpos[:, 23:27],
            0.15 * qvel[:, :20],
            0.05 * qvel[:, 20:26],
        ),
        axis=-1,
    ).astype(np.float32)


def nearest_core_distance(pool: np.ndarray, core: np.ndarray) -> np.ndarray:
    scale = np.std(np.concatenate((core, pool), axis=0), axis=0)
    scale = np.maximum(scale, 1.0e-4)
    core_n = core / scale
    pool_n = pool / scale
    result = np.empty(len(pool), dtype=np.float32)
    for start in range(0, len(pool), 64):
        chunk = pool_n[start : start + 64]
        mse = np.mean((chunk[:, None, :] - core_n[None, :, :]) ** 2, axis=-1)
        result[start : start + len(chunk)] = np.min(mse, axis=1)
    return result


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def concatenate_field(
    name: str,
    core: dict[str, np.ndarray],
    pool: dict[str, np.ndarray],
    selected: np.ndarray,
) -> np.ndarray:
    first = core[name]
    second = pool[name][selected]
    if name == "source_teacher_actions" and first.shape[1] != second.shape[1]:
        horizon = max(first.shape[1], second.shape[1])

        def pad(actions: np.ndarray) -> np.ndarray:
            if actions.shape[1] == horizon:
                return actions
            output = np.empty((len(actions), horizon, actions.shape[2]), dtype=actions.dtype)
            output[:, : actions.shape[1]] = actions
            output[:, actions.shape[1] :] = actions[:, -1:, :]
            return output

        first = pad(first)
        second = pad(second)
    return np.concatenate((first, second), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Widen the v9 P4 prefill curriculum by physical entry distance."
    )
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--add", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    core = load(args.core)
    pool = load(args.pool)
    add = min(args.add, len(pool["source_qpos"]))
    distance = nearest_core_distance(
        normalized_features(pool), normalized_features(core)
    )
    selected = np.argsort(distance, kind="stable")[:add]

    ignored = {"original_bank_ids", "source_dataset_id", "source_bank_id", "core_distance"}
    common = sorted((set(core) & set(pool)) - ignored)
    payload = {
        name: concatenate_field(name, core, pool, selected) for name in common
    }
    payload["source_dataset_id"] = np.concatenate(
        (
            np.zeros(len(core["source_qpos"]), dtype=np.int64),
            np.ones(add, dtype=np.int64),
        )
    )
    payload["source_bank_id"] = np.concatenate(
        (
            np.asarray(core.get("original_bank_ids", np.arange(len(core["source_qpos"])))),
            np.asarray(pool.get("original_bank_ids", np.arange(len(pool["source_qpos"]))))[
                selected
            ],
        )
    ).astype(np.int64)
    payload["core_distance"] = np.concatenate(
        (np.zeros(len(core["source_qpos"]), dtype=np.float32), distance[selected])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"output={args.output}")
    print(f"core_rows={len(core['source_qpos'])}")
    print(f"pool_rows={len(pool['source_qpos'])}")
    print(f"added_rows={add}")
    print(f"total_rows={len(payload['source_qpos'])}")
    print(
        "selected_core_distance="
        f"{distance[selected].min():.6f}/{distance[selected].mean():.6f}/"
        f"{distance[selected].max():.6f}"
    )
    remaining = np.delete(distance, selected)
    if len(remaining):
        print(
            "remaining_core_distance="
            f"{remaining.min():.6f}/{remaining.mean():.6f}/{remaining.max():.6f}"
        )


if __name__ == "__main__":
    main()
