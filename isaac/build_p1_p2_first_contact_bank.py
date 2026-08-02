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


def classify_force_window(force: np.ndarray, epsilon: float) -> str:
    if force.size < 2:
        return "insufficient"
    delta = np.diff(force)
    if force[-1] >= force[0] + epsilon and delta.mean() > 0.0:
        return "increasing"
    if force[1] <= force[0] - epsilon or force[-1] <= 0.75 * force[0]:
        return "immediate_decay"
    if delta.size >= 2 and np.any(delta[:-1] * delta[1:] < 0.0):
        return "oscillating"
    return "flat_or_mixed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pre-contact reset states from complete physical-H1 trajectories."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--contact-force-n", type=float, default=0.03)
    parser.add_argument("--trend-window-steps", type=int, default=4)
    parser.add_argument("--trend-epsilon-n", type=float, default=0.003)
    args = parser.parse_args()

    rows: dict[str, list[np.ndarray | int | float | str]] = {
        "source_qpos": [],
        "source_qvel": [],
        "source_target_qpos": [],
        "source_policy_action": [],
        "target_qpos": [],
        "target_qvel": [],
        "source_global_bank_ids": [],
        "original_source_bundle_row": [],
        "first_contact_step": [],
        "pre_contact_step": [],
        "first_contact_force_n": [],
        "post_contact_force_window_n": [],
        "post_contact_force_delta_n": [],
        "contact_trend_class": [],
        "teacher_source": [],
        "source_teacher_actions": [],
        "phase_ids": [],
        "policy_success_episode": [],
        "arrival_bucket": [],
    }
    skipped_no_contact = 0
    source_summaries = []
    trend_window = max(int(args.trend_window_steps), 2)

    for input_path in args.input:
        input_path = input_path.resolve()
        bundle = np.load(input_path)
        required = {
            "state_qpos",
            "state_qvel",
            "state_target_qpos",
            "action",
            "contact_force_thumb_index_middle_ring_pinky",
        }
        missing = required.difference(bundle.files)
        if missing:
            raise KeyError(f"{input_path} is missing {sorted(missing)}")
        captured = 0
        for trajectory_id in range(bundle["state_qpos"].shape[0]):
            ring_force = bundle["contact_force_thumb_index_middle_ring_pinky"][
                trajectory_id, :, 3
            ]
            contact_steps = np.flatnonzero(ring_force >= float(args.contact_force_n))
            if contact_steps.size == 0:
                skipped_no_contact += 1
                continue
            contact_step = int(contact_steps[0])
            pre_step = max(contact_step - 1, 0)
            force_window = ring_force[
                contact_step : min(contact_step + trend_window, ring_force.size)
            ].astype(np.float32)
            padded_force = np.full(trend_window, np.nan, dtype=np.float32)
            padded_force[: force_window.size] = force_window
            padded_delta = np.full(trend_window - 1, np.nan, dtype=np.float32)
            if force_window.size >= 2:
                padded_delta[: force_window.size - 1] = np.diff(force_window)

            source_id = (
                int(bundle["source_bundle_row"][trajectory_id])
                if "source_bundle_row" in bundle.files
                else trajectory_id
            )
            qpos = bundle["state_qpos"][trajectory_id, pre_step].astype(np.float32)
            qvel = bundle["state_qvel"][trajectory_id, pre_step].astype(np.float32)
            rows["source_qpos"].append(qpos)
            rows["source_qvel"].append(qvel)
            rows["source_target_qpos"].append(
                bundle["state_target_qpos"][trajectory_id, pre_step].astype(np.float32)
            )
            rows["source_policy_action"].append(
                bundle["action"][trajectory_id, pre_step].astype(np.float32)
            )
            rows["target_qpos"].append(qpos.copy())
            rows["target_qvel"].append(qvel.copy())
            rows["source_global_bank_ids"].append(source_id)
            rows["original_source_bundle_row"].append(source_id)
            rows["first_contact_step"].append(contact_step)
            rows["pre_contact_step"].append(pre_step)
            rows["first_contact_force_n"].append(float(ring_force[contact_step]))
            rows["post_contact_force_window_n"].append(padded_force)
            rows["post_contact_force_delta_n"].append(padded_delta)
            rows["contact_trend_class"].append(
                classify_force_window(force_window, float(args.trend_epsilon_n))
            )
            rows["teacher_source"].append(str(input_path))
            rows["source_teacher_actions"].append(
                bundle["action"][trajectory_id, pre_step : pre_step + 1].astype(np.float32)
            )
            # This bank is consumed by the local-arrival search task. Phase 2
            # selects its endpoint-search branch without changing the saved state.
            rows["phase_ids"].append(2)
            rows["policy_success_episode"].append(True)
            rows["arrival_bucket"].append("FC0")
            captured += 1
        source_summaries.append(
            {"path": str(input_path), "sha256": sha256(input_path), "captured": captured}
        )

    if not rows["source_qpos"]:
        raise RuntimeError("No first-contact transitions were found")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    np.savez_compressed(output, **arrays)
    labels, counts = np.unique(arrays["contact_trend_class"], return_counts=True)
    summary = {
        "output": str(output),
        "output_sha256": sha256(output),
        "inputs": source_summaries,
        "rows": int(arrays["source_qpos"].shape[0]),
        "contact_force_threshold_n": float(args.contact_force_n),
        "trend_window_steps": trend_window,
        "trend_epsilon_n": float(args.trend_epsilon_n),
        "trend_counts": {str(label): int(count) for label, count in zip(labels, counts)},
        "skipped_no_contact": skipped_no_contact,
        "reset_state": "one normal PhysX step before first ring contact",
        "ppo_started": False,
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_FIRST_CONTACT_BANK=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
