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
        description="Append a continuous terminal-correction phase to P1/P2 DAgger data."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--terminal-phase-kind", type=int, default=5)
    parser.add_argument("--phase-horizon-steps", type=int, default=18)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    summary_path = args.summary.resolve()
    with np.load(source) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}

    observation = np.asarray(payload["observation"], dtype=np.float32)
    phase_kind = np.asarray(payload["phase_kind"], dtype=np.int64)
    source_rows = np.asarray(payload["terminal_source_bundle_row"], dtype=np.int64)
    if observation.ndim != 3 or observation.shape[1] != 1:
        raise ValueError(f"Unexpected observation shape: {observation.shape}")
    if args.phase_horizon_steps < 2:
        raise ValueError("--phase-horizon-steps must be at least 2")

    terminal_indices = np.flatnonzero(phase_kind == args.terminal_phase_kind)
    if len(terminal_indices) == 0:
        raise ValueError("No terminal teacher rows found")
    terminal_sources = source_rows[terminal_indices]
    run_starts = np.r_[0, np.flatnonzero(terminal_sources[1:] != terminal_sources[:-1]) + 1]
    run_ends = np.r_[run_starts[1:], len(terminal_indices)]

    correction_phase = np.zeros(len(observation), dtype=np.float32)
    run_lengths: list[int] = []
    denominator = float(args.phase_horizon_steps - 1)
    for start, end in zip(run_starts, run_ends, strict=True):
        rows = terminal_indices[start:end]
        run_length = int(end - start)
        run_lengths.append(run_length)
        correction_phase[rows] = np.minimum(
            np.arange(run_length, dtype=np.float32) / denominator, 1.0
        )

    payload["observation"] = np.concatenate(
        (observation, correction_phase[:, None, None]), axis=-1
    ).astype(np.float32)
    payload["terminal_correction_phase"] = correction_phase
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)

    terminal_phase = correction_phase[terminal_indices]
    summary = {
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "rows": int(len(observation)),
        "old_observation_dim": int(observation.shape[-1]),
        "new_observation_dim": int(payload["observation"].shape[-1]),
        "terminal_phase_kind": args.terminal_phase_kind,
        "terminal_rows": int(len(terminal_indices)),
        "terminal_trajectories": int(len(run_lengths)),
        "trajectory_length_min": int(min(run_lengths)),
        "trajectory_length_max": int(max(run_lengths)),
        "terminal_phase_min": float(terminal_phase.min()),
        "terminal_phase_max": float(terminal_phase.max()),
        "phase_horizon_steps": args.phase_horizon_steps,
        "ppo_started": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_TERMINAL_PHASE_DATA=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
