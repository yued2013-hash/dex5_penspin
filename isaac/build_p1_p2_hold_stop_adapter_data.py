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
        description="Build drive/hold/failed-hold labels for a terminal stop adapter."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.base.resolve()) as data:
        base_obs = np.asarray(data["observation"], dtype=np.float32)
        base_phase = np.asarray(data["phase_kind"], dtype=np.int64)
    base_keep = base_phase != 6
    base_obs = base_obs[base_keep]
    if base_obs.shape[-1] != 192:
        raise ValueError(f"Expected 192-D base observations, got {base_obs.shape}")
    base_obs = np.concatenate(
        (base_obs, np.zeros((*base_obs.shape[:-1], 6), dtype=np.float32)), axis=-1
    )

    teacher_obs: list[np.ndarray] = []
    teacher_action: list[np.ndarray] = []
    stop_target: list[np.ndarray] = []
    action_mask: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    seen: set[int] = set()
    drive_rows = stable_hold_rows = failed_hold_rows = 0
    teacher_hashes: dict[str, str] = {}
    for path in args.teachers:
        resolved = path.resolve()
        teacher_hashes[str(resolved)] = sha256(resolved)
        with np.load(resolved) as data:
            ids = np.asarray(data["source_bundle_row"], dtype=np.int64)
            observation = np.asarray(data["observation"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
            target = np.asarray(data["target_condition"], dtype=np.float32)
            delta = np.asarray(data["target_condition_delta"], dtype=np.float32)
            margin = np.asarray(data["semantic_margin"], dtype=np.float32)
            hold = np.asarray(data["hold_latched"], dtype=np.bool_)
            accepted = np.asarray(data["accepted_step"], dtype=np.bool_)
        for row, source_id_value in enumerate(ids):
            source_id = int(source_id_value)
            if source_id in seen:
                continue
            first_success = np.flatnonzero(accepted[row])
            if len(first_success) == 0:
                continue
            first = int(first_success[0])
            selected = np.arange(observation.shape[1])
            selected = selected[(selected <= first) | hold[row]]
            phase = np.minimum(selected.astype(np.float32) / 17.0, 1.0)
            terminal = np.ones((len(selected), 1), dtype=np.float32)
            dynamics = np.concatenate(
                (
                    margin[row, selected, None],
                    delta[row, selected],
                    hold[row, selected, None].astype(np.float32),
                ),
                axis=-1,
            )
            conditioned = np.concatenate(
                (
                    observation[row, selected],
                    terminal,
                    phase[:, None],
                    target[row, selected],
                    dynamics,
                ),
                axis=-1,
            )
            stable = hold[row, selected] & accepted[row, selected]
            failed = hold[row, selected] & ~accepted[row, selected]
            drive = ~hold[row, selected]
            teacher_obs.append(conditioned)
            teacher_action.append(action[row, selected])
            stop_target.append(stable.astype(np.float32))
            action_mask.append((drive | stable).astype(np.bool_))
            source_ids.append(np.full(len(selected), source_id, dtype=np.int64))
            drive_rows += int(drive.sum())
            stable_hold_rows += int(stable.sum())
            failed_hold_rows += int(failed.sum())
            seen.add(source_id)

    paired_obs = np.concatenate(teacher_obs, axis=0)
    if paired_obs.shape[-1] != 198:
        raise ValueError(f"Expected 198-D teacher observations, got {paired_obs.shape}")
    payload = {
        "distill_observation": base_obs,
        "teacher_observation": paired_obs[:, None, :],
        "teacher_action": np.concatenate(teacher_action, axis=0)[:, None, :],
        "stop_target": np.concatenate(stop_target),
        "action_loss_mask": np.concatenate(action_mask),
        "source_bundle_row": np.concatenate(source_ids),
    }
    output = args.output.resolve()
    summary_path = args.summary.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "base": str(args.base.resolve()),
        "base_sha256": sha256(args.base.resolve()),
        "teachers": teacher_hashes,
        "output": str(output),
        "output_sha256": sha256(output),
        "observation_dim": 198,
        "unique_teacher_trajectories": len(seen),
        "teacher_rows": int(len(payload["teacher_observation"])),
        "distill_rows": int(len(base_obs)),
        "drive_rows": drive_rows,
        "stable_hold_rows": stable_hold_rows,
        "failed_hold_rows": failed_hold_rows,
        "ppo_started": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_HOLD_STOP_DATA=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
