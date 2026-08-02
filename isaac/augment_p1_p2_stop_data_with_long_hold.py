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
        description="Add strict H6 closed-loop correction rows to v14 stop-adapter data."
    )
    parser.add_argument("--base-data", required=True, type=Path)
    parser.add_argument("--long-hold-teacher", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    base_path = args.base_data.resolve()
    teacher_path = args.long_hold_teacher.resolve()
    with np.load(base_path, allow_pickle=False) as data:
        base = {key: np.asarray(data[key]).copy() for key in data.files}
    with np.load(teacher_path, allow_pickle=False) as data:
        observation = np.asarray(data["observation"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        target = np.asarray(data["target_condition"], dtype=np.float32)
        delta = np.asarray(data["target_condition_delta"], dtype=np.float32)
        margin = np.asarray(data["semantic_margin"], dtype=np.float32)
        hold = np.asarray(data["hold_latched"], dtype=np.bool_)
        accepted = np.asarray(data["accepted_step"], dtype=np.bool_)
        source = np.asarray(data["source_bundle_row"], dtype=np.int64)

    if observation.shape[-1] != 186:
        raise ValueError(f"Expected 186-D local observations, got {observation.shape}")
    if not accepted.all():
        raise ValueError("Strict H6 teacher contains a non-eligible retained step")
    trajectories, horizon = observation.shape[:2]
    terminal = np.ones((trajectories, horizon, 1), dtype=np.float32)
    phase = np.ones_like(terminal)
    conditioned = np.concatenate(
        (
            observation,
            terminal,
            phase,
            target,
            margin[..., None],
            delta,
            hold[..., None].astype(np.float32),
        ),
        axis=-1,
    )
    if conditioned.shape[-1] != 198:
        raise ValueError(f"Expected 198-D conditioned observations, got {conditioned.shape}")

    correction_obs = conditioned.reshape(-1, 1, 198)
    correction_action = action.reshape(-1, 1, 20)
    correction_rows = correction_obs.shape[0]
    source_offset = int(base["source_bundle_row"].max(initial=0)) + 1
    correction_source = np.repeat(source + source_offset, horizon)
    payload = {
        "distill_observation": base["distill_observation"],
        "teacher_observation": np.concatenate(
            (base["teacher_observation"], correction_obs), axis=0
        ),
        "teacher_action": np.concatenate(
            (base["teacher_action"], correction_action), axis=0
        ),
        "stop_target": np.concatenate(
            (base["stop_target"], np.zeros(correction_rows, dtype=np.float32))
        ),
        "action_loss_mask": np.concatenate(
            (base["action_loss_mask"], np.ones(correction_rows, dtype=np.bool_))
        ),
        "source_bundle_row": np.concatenate(
            (base["source_bundle_row"], correction_source)
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "base_data": str(base_path),
        "base_data_sha256": sha256(base_path),
        "long_hold_teacher": str(teacher_path),
        "long_hold_teacher_sha256": sha256(teacher_path),
        "output": str(output),
        "output_sha256": sha256(output),
        "base_teacher_rows": int(len(base["teacher_observation"])),
        "long_hold_trajectories": int(trajectories),
        "long_hold_rows": int(correction_rows),
        "combined_teacher_rows": int(len(payload["teacher_observation"])),
        "long_hold_stop_target": 0,
        "long_hold_action_loss_enabled": True,
        "long_hold_latched_rows": int(hold.sum()),
        "ppo_started": False,
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_V15_STOP_DATA=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
