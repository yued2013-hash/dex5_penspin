from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Build a thumb-support correction bundle from deterministic reset-bank failures."
)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--audit", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--failure_rate_max", type=float, default=0.10)
parser.add_argument("--teacher_rate_min", type=float, default=0.90)
args = parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    source_path = resolve(args.source)
    audit_path = resolve(args.audit)
    output_path = resolve(args.output)
    with np.load(source_path) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))["per_reset_bank"]
    count = len(payload["source_qpos"])
    rates = np.asarray(
        [float(audit.get(str(index), {}).get("success_rate", 0.0)) for index in range(count)]
    )
    failed_ids = np.flatnonzero(rates <= args.failure_rate_max)
    teacher_ids = np.flatnonzero(rates >= args.teacher_rate_min)
    if not len(failed_ids) or not len(teacher_ids):
        raise ValueError("Audit does not contain both deterministic failure and teacher states")

    source_qpos = payload["source_qpos"][failed_ids].astype(np.float32)
    teacher_qpos = payload["source_qpos"][teacher_ids].astype(np.float32)
    # Match nearby pen geometry first, then minimize the required hand correction.
    pen_delta = source_qpos[:, None, 20:23] - teacher_qpos[None, :, 20:23]
    quat_dot = np.abs(
        np.sum(source_qpos[:, None, 23:27] * teacher_qpos[None, :, 23:27], axis=-1)
    )
    hand_delta = source_qpos[:, None, :20] - teacher_qpos[None, :, :20]
    cost = (
        2500.0 * np.sum(pen_delta * pen_delta, axis=-1)
        + 20.0 * (1.0 - np.clip(quat_dot, 0.0, 1.0))
        + np.mean(hand_delta * hand_delta, axis=-1)
    )
    nearest_teacher_ids = teacher_ids[np.argmin(cost, axis=1)]

    target_qpos = source_qpos.copy()
    target_qpos[:, :20] = payload["source_qpos"][nearest_teacher_ids, :20]
    correction = {
        "source_qpos": source_qpos,
        "source_qvel": payload["source_qvel"][failed_ids].astype(np.float32),
        "source_target_qpos": payload["source_target_qpos"][failed_ids].astype(np.float32),
        "source_policy_action": payload["source_policy_action"][failed_ids].astype(np.float32),
        "source_teacher_actions": np.zeros((len(failed_ids), 0, 20), dtype=np.float32),
        "target_qpos": target_qpos,
        "target_qvel": np.zeros((len(failed_ids), 26), dtype=np.float32),
        "phase_ids": payload["phase_ids"][failed_ids].astype(np.int64),
        "source_episode_step": np.zeros(len(failed_ids), dtype=np.int64),
        "source_phase_offset_rad": payload["source_phase_offset_rad"][failed_ids].astype(
            np.float32
        ),
        "target_remaining_rad": np.zeros(len(failed_ids), dtype=np.float32),
        "policy_target_pen_quat": source_qpos[:, 23:27].copy(),
        "source_contact_force_n": payload["source_contact_force_n"][failed_ids].astype(
            np.float32
        ),
        "curriculum_stage_deg": payload["curriculum_stage_deg"][failed_ids].astype(
            np.float32
        ),
        "target_p4_candidate_ids": payload["target_p4_candidate_ids"][failed_ids].astype(
            np.int64
        ),
        "source_bank_ids": failed_ids.astype(np.int64),
        "matched_teacher_bank_ids": nearest_teacher_ids.astype(np.int64),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **correction)
    thumb_force = payload["source_contact_force_n"][:, 0]
    summary = {
        "output": str(output_path.relative_to(ROOT)),
        "source": str(source_path.relative_to(ROOT)),
        "audit": str(audit_path.relative_to(ROOT)),
        "rows": int(len(failed_ids)),
        "failure_bank_count": int(len(failed_ids)),
        "teacher_bank_count": int(len(teacher_ids)),
        "failure_thumb_force_mean_n": float(thumb_force[failed_ids].mean()),
        "teacher_thumb_force_mean_n": float(thumb_force[teacher_ids].mean()),
        "mean_hand_correction_rms_rad": float(
            np.sqrt(np.mean((target_qpos[:, :20] - source_qpos[:, :20]) ** 2, axis=1)).mean()
        ),
        "intent": "keep pen phase fixed while moving failed entry states onto the thumb-supported hand branch",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
