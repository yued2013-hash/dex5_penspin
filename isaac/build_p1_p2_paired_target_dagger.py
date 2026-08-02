from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-state terminal teachers into target-conditioned P1/P2 DAgger data."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--drop-phase", type=int, default=5)
    parser.add_argument("--paired-phase", type=int, default=6)
    parser.add_argument("--phase-horizon-steps", type=int, default=18)
    args = parser.parse_args()

    with np.load(args.base.resolve()) as data:
        base = {key: np.asarray(data[key]) for key in data.files}
    keep = np.asarray(base["phase_kind"], dtype=np.int64) != args.drop_phase
    payload = {key: value[keep] for key, value in base.items()}
    old_observation = np.asarray(payload["observation"], dtype=np.float32)
    if old_observation.shape[-1] != 188:
        raise ValueError(f"Expected 188-D base observation, got {old_observation.shape}")
    payload["observation"] = np.concatenate(
        (
            old_observation,
            np.zeros((*old_observation.shape[:-1], 4), dtype=np.float32),
        ),
        axis=-1,
    )

    paired_observation: list[np.ndarray] = []
    paired_action: list[np.ndarray] = []
    paired_source: list[int] = []
    seen_sources: set[int] = set()
    trajectory_lengths: list[int] = []
    denominator = float(max(args.phase_horizon_steps - 1, 1))
    teacher_hashes: dict[str, str] = {}
    for teacher_path in args.teachers:
        teacher = teacher_path.resolve()
        teacher_hashes[str(teacher)] = sha256(teacher)
        with np.load(teacher) as data:
            source_rows = np.asarray(data["source_bundle_row"], dtype=np.int64)
            observations = np.asarray(data["observation"], dtype=np.float32)
            actions = np.asarray(data["action"], dtype=np.float32)
            accepted = np.asarray(data["accepted_step"], dtype=np.bool_)
            conditions = np.asarray(data["target_condition"], dtype=np.float32)
        for index, source_row in enumerate(source_rows):
            source_id = int(source_row)
            if source_id in seen_sources:
                continue
            accepted_indices = np.flatnonzero(accepted[index])
            if len(accepted_indices) == 0:
                continue
            length = int(accepted_indices[0]) + 1
            phase = np.minimum(
                np.arange(length, dtype=np.float32) / denominator, 1.0
            )
            terminal = np.ones((length, 1), dtype=np.float32)
            conditioned_observation = np.concatenate(
                (
                    observations[index, :length],
                    terminal,
                    phase[:, None],
                    conditions[index, :length],
                ),
                axis=-1,
            )
            if conditioned_observation.shape[-1] != 192:
                raise ValueError(
                    f"Expected 192-D paired observation, got {conditioned_observation.shape}"
                )
            paired_observation.append(conditioned_observation)
            paired_action.append(actions[index, :length])
            paired_source.extend([source_id] * length)
            trajectory_lengths.append(length)
            seen_sources.add(source_id)

    observation_rows = np.concatenate(paired_observation, axis=0)[:, None, :]
    action_rows = np.concatenate(paired_action, axis=0)[:, None, :]
    paired_count = len(observation_rows)
    append = {
        "observation": observation_rows,
        "action": action_rows,
        "student_action": np.zeros_like(action_rows),
        "lengths": np.ones(paired_count, dtype=np.int64),
        "phase_kind": np.full(paired_count, args.paired_phase, dtype=np.int64),
        "disagreement": np.linalg.norm(action_rows[:, 0], axis=-1).astype(np.float32),
        "target_excess_mm": np.zeros(paired_count, dtype=np.float32),
        "under_pass": np.zeros(paired_count, dtype=np.bool_),
        "semantic_pass": np.zeros(paired_count, dtype=np.bool_),
        "terminal_source_bundle_row": np.asarray(paired_source, dtype=np.int64),
        "terminal_correction_phase": observation_rows[:, 0, 187],
    }
    cursor = 0
    for length in trajectory_lengths:
        append["under_pass"][cursor + length - 1] = True
        append["semantic_pass"][cursor + length - 1] = True
        cursor += length
    for key in payload:
        if key not in append:
            raise KeyError(f"Missing paired value for base key {key!r}")
        payload[key] = np.concatenate((payload[key], append[key]), axis=0)

    output = args.output.resolve()
    summary_path = args.summary.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    phase_values, phase_counts = np.unique(payload["phase_kind"], return_counts=True)
    summary = {
        "base": str(args.base.resolve()),
        "base_sha256": sha256(args.base.resolve()),
        "teachers": teacher_hashes,
        "output": str(output),
        "output_sha256": sha256(output),
        "rows": int(len(payload["observation"])),
        "observation_dim": int(payload["observation"].shape[-1]),
        "paired_trajectories": len(trajectory_lengths),
        "paired_transitions": paired_count,
        "paired_unique_sources": len(seen_sources),
        "paired_length_min": min(trajectory_lengths),
        "paired_length_max": max(trajectory_lengths),
        "phase_counts": {
            str(int(value)): int(count)
            for value, count in zip(phase_values, phase_counts, strict=True)
        },
        "dropped_shared_teacher_phase": args.drop_phase,
        "ppo_started": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_PAIRED_TARGET_DATA=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
