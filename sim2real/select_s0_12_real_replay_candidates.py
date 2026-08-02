from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONTACT_THRESHOLD_N = 0.03
CONTROL_HZ = 60.0
TARGET_MIN_DEG = 30.0
TARGET_MAX_DEG = 40.0
MIN_AXIS_ALIGNMENT = 0.70
HOLD_STEPS = 9
MAX_HEIGHT_LOSS_MM = 5.0
MAX_MAPPED_TARGET_DELTA_RAD = 0.01
MIN_TIM_CONTACT_RATIO = 0.75


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[:, 0] * right[:, 0] - np.sum(
        left[:, 1:] * right[:, 1:], axis=1
    )
    vector = (
        left[:, :1] * right[:, 1:]
        + right[:, :1] * left[:, 1:]
        + np.cross(left[:, 1:], right[:, 1:])
    )
    return np.concatenate((scalar[:, None], vector), axis=1)


def _quaternion_progress(
    quaternion: np.ndarray, initial_quaternion: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conjugate = np.repeat(initial_quaternion[None, :], len(quaternion), axis=0)
    conjugate[:, 1:] *= -1.0
    relative = _quat_mul(quaternion, conjugate)
    relative /= np.linalg.norm(relative, axis=1, keepdims=True)
    relative = np.where(relative[:, :1] < 0.0, -relative, relative)
    vector_norm = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(
        vector_norm, np.clip(relative[:, 0], 1.0e-9, None)
    )
    axis_z = np.divide(
        relative[:, 3],
        vector_norm,
        out=np.zeros_like(vector_norm),
        where=vector_norm > 1.0e-9,
    )
    return angle * axis_z, angle, axis_z


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    with np.load(args.reset_bank, allow_pickle=False) as bank:
        reset_penetration = np.asarray(
            bank["labels_baseline_max_penetration_m"], dtype=np.float64
        )

    with np.load(args.rollout, allow_pickle=False) as data:
        if not bool(data["dataset_authorized"]):
            raise RuntimeError("rollout dataset is not authorized")
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        pre_pose = np.asarray(data["pen_pose"], dtype=np.float64)
        post_pose = np.asarray(data["pen_pose_next"], dtype=np.float64)
        pose_velocity = np.asarray(data["pen_angular_velocity"], dtype=np.float64)
        contacts = np.asarray(data["contact_state_next"], dtype=bool)
        force = np.asarray(data["contact_force_norm_next"], dtype=np.float64)
        mapped_target = np.asarray(data["adapter_projected_target"], dtype=np.float64)
        virtual_action = np.asarray(data["virtual_action"], dtype=np.float64)
        source_ids = np.asarray(data["episode_source_trajectory_id"], dtype=np.int64)
        bank_ids = np.asarray(data["episode_bank_id"], dtype=np.int64)
        state_ids = np.asarray(data["episode_state_id"], dtype=np.int64)
        splits = np.asarray(data["episode_split"])
        successes = np.asarray(data["episode_corrected_success"], dtype=bool)

    rows: list[dict[str, object]] = []
    for episode, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        target_rotation, _, alignment = _quaternion_progress(
            post_pose[start:stop, 3:7], pre_pose[start, 3:7]
        )
        target_deg = np.rad2deg(target_rotation)
        height_loss_mm = 1000.0 * (
            pre_pose[start, 2] - post_pose[start:stop, 2]
        )
        target_speed = np.abs(pose_velocity[start:stop, 2])
        wrong_axis_speed = np.linalg.norm(pose_velocity[start:stop, :2], axis=1)
        tim = contacts[start:stop, :3].all(axis=1)
        ring_pinky_clear = ~contacts[start:stop, 3:].any(axis=1)
        hold = (
            (target_deg >= TARGET_MIN_DEG)
            & (target_deg <= TARGET_MAX_DEG)
            & (alignment >= MIN_AXIS_ALIGNMENT)
            & (target_speed <= 2.5)
            & (wrong_axis_speed <= 2.0)
            & (height_loss_mm <= MAX_HEIGHT_LOSS_MM)
            & tim
            & ring_pinky_clear
        )
        hold_steps = _longest_true_run(hold)
        target_delta = np.diff(mapped_target[start:stop], axis=0)
        action_delta = np.diff(virtual_action[start:stop], axis=0)
        max_target_delta = (
            float(np.max(np.abs(target_delta))) if len(target_delta) else 0.0
        )
        rms_target_delta = (
            float(np.sqrt(np.mean(np.square(target_delta))))
            if len(target_delta)
            else 0.0
        )
        action_total_variation = (
            float(np.sum(np.linalg.norm(action_delta, axis=1)))
            if len(action_delta)
            else 0.0
        )
        hold_force = force[start:stop, :3][hold]
        minimum_hold_tim_force = (
            float(np.min(hold_force)) if len(hold_force) else 0.0
        )
        peak_index = int(np.argmax(target_rotation))
        final_rotation = float(target_deg[-1])
        rotation_center_margin = max(0.0, 1.0 - abs(final_rotation - 35.0) / 5.0)
        smoothness_margin = max(
            0.0, 1.0 - max_target_delta / MAX_MAPPED_TARGET_DELTA_RAD
        )
        force_margin = min(
            1.0,
            max(0.0, minimum_hold_tim_force - CONTACT_THRESHOLD_N) / 0.20,
        )
        tim_ratio = float(np.mean(tim))
        score = (
            3.0 * rotation_center_margin
            + float(alignment[peak_index])
            + min(1.0, hold_steps / 10.0)
            + tim_ratio
            + smoothness_margin
            + 0.5 * force_margin
            - 0.05 * max(0.0, float(np.max(height_loss_mm)))
        )
        eligible = bool(
            successes[episode]
            and float(np.max(target_deg)) >= TARGET_MIN_DEG
            and final_rotation >= TARGET_MIN_DEG
            and float(alignment[peak_index]) >= MIN_AXIS_ALIGNMENT
            and hold_steps >= HOLD_STEPS
            and float(np.max(height_loss_mm)) <= MAX_HEIGHT_LOSS_MM
            and max_target_delta <= MAX_MAPPED_TARGET_DELTA_RAD
            and tim_ratio >= MIN_TIM_CONTACT_RATIO
            and reset_penetration[bank_ids[episode]] <= 0.0005
        )
        rows.append(
            {
                "episode_index": episode,
                "row_start": int(start),
                "row_stop": int(stop),
                "length_steps": int(stop - start),
                "duration_s": float((stop - start) / CONTROL_HZ),
                "split": str(splits[episode]),
                "source_trajectory_id": int(source_ids[episode]),
                "bank_id": int(bank_ids[episode]),
                "state_id": int(state_ids[episode]),
                "peak_target_quaternion_rotation_deg": float(np.max(target_deg)),
                "final_target_quaternion_rotation_deg": final_rotation,
                "axis_alignment_at_peak": float(alignment[peak_index]),
                "strict_hold_steps": hold_steps,
                "strict_hold_ms": float(hold_steps / CONTROL_HZ * 1000.0),
                "tim_contact_ratio": tim_ratio,
                "minimum_hold_tim_force_n": minimum_hold_tim_force,
                "maximum_height_loss_mm": float(np.max(height_loss_mm)),
                "reset_baseline_penetration_mm": float(
                    1000.0 * reset_penetration[bank_ids[episode]]
                ),
                "maximum_mapped_target_delta_rad": max_target_delta,
                "maximum_native_mapped_speed_rad_s": max_target_delta * CONTROL_HZ,
                "rms_mapped_target_delta_rad": rms_target_delta,
                "virtual_action_total_variation": action_total_variation,
                "score": score,
                "eligible": eligible,
            }
        )

    eligible_rows = [row for row in rows if bool(row["eligible"])]
    best_by_source: dict[int, dict[str, object]] = {}
    for row in eligible_rows:
        source_id = int(row["source_trajectory_id"])
        if source_id not in best_by_source or float(row["score"]) > float(
            best_by_source[source_id]["score"]
        ):
            best_by_source[source_id] = row
    train_candidates = sorted(
        (row for row in best_by_source.values() if row["split"] == "train"),
        key=lambda row: (-float(row["score"]), int(row["episode_index"])),
    )
    selected = train_candidates[: args.count]
    if len(selected) != args.count:
        raise RuntimeError(
            f"only {len(selected)} source-isolated train candidates passed"
        )

    split_counts = {
        split: sum(row["split"] == split for row in eligible_rows)
        for split in ("train", "validation", "test")
    }
    unique_sources_by_split = {
        split: len(
            {
                int(row["source_trajectory_id"])
                for row in eligible_rows
                if row["split"] == split
            }
        )
        for split in split_counts
    }
    manifest = {
        "schema": "dex5-s0.12-real-replay-selection-v1",
        "status": "PASS_OFFLINE_SELECTION",
        "rollout": str(args.rollout.resolve()),
        "rollout_sha256": _sha256(args.rollout),
        "reset_bank": str(args.reset_bank.resolve()),
        "reset_bank_sha256": _sha256(args.reset_bank),
        "total_rollouts": len(rows),
        "eligible_rollouts": len(eligible_rows),
        "eligible_unique_sources": len(best_by_source),
        "eligible_by_split": split_counts,
        "eligible_unique_sources_by_split": unique_sources_by_split,
        "selection_policy": "highest-score eligible trajectory per train source",
        "selection_count": len(selected),
        "selection_unique_sources": len(
            {int(row["source_trajectory_id"]) for row in selected}
        ),
        "eligibility": {
            "corrected_success": True,
            "peak_and_final_target_quaternion_rotation_deg_min": 30.0,
            "axis_alignment_at_peak_min": MIN_AXIS_ALIGNMENT,
            "strict_hold_steps_min": HOLD_STEPS,
            "strict_hold_ms_min": HOLD_STEPS / CONTROL_HZ * 1000.0,
            "maximum_height_loss_mm": MAX_HEIGHT_LOSS_MM,
            "maximum_mapped_target_delta_rad": MAX_MAPPED_TARGET_DELTA_RAD,
            "minimum_tim_contact_ratio": MIN_TIM_CONTACT_RATIO,
            "maximum_reset_baseline_penetration_mm": 0.5,
        },
        "proposed_real_speed_tiers_rad_s": [0.12, 0.18, 0.24],
        "selected_trajectories": selected,
        "runtime_penetration_note": (
            "The rollout file has no per-step penetration channel. Selection uses "
            "successful task termination plus reset-bank baseline penetration; real "
            "replay still requires physical validity monitoring."
        ),
        "hardware_commands_issued": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    selected_peak = [
        float(row["peak_target_quaternion_rotation_deg"]) for row in selected
    ]
    selected_final = [
        float(row["final_target_quaternion_rotation_deg"]) for row in selected
    ]
    selected_tim = [float(row["tim_contact_ratio"]) for row in selected]
    selected_delta = [
        float(row["maximum_mapped_target_delta_rad"]) for row in selected
    ]
    args.report.write_text(
        "# Dex5 S0.12 Offline Replay Candidate Audit\n\n"
        "## Result\n\n"
        "```text\nOffline candidate quantity: PASS\n"
        "Offline candidate quality: PASS\n"
        "Powered real replay: NOT RUN\n"
        "Real pen-pose measurement: REQUIRED\n```\n\n"
        f"- Audited rollouts: `{len(rows)}`\n"
        f"- Eligible rollouts: `{len(eligible_rows)}`\n"
        f"- Eligible unique sources: `{len(best_by_source)}`\n"
        f"- Selected first batch: `{len(selected)}` trajectories from "
        f"`{len(selected)}` distinct train sources\n\n"
        "## Selected batch envelope\n\n"
        f"- Peak quaternion rotation: `{min(selected_peak):.2f}-"
        f"{max(selected_peak):.2f} deg`\n"
        f"- Final quaternion rotation: `{min(selected_final):.2f}-"
        f"{max(selected_final):.2f} deg`\n"
        f"- TIM contact ratio: `{min(selected_tim) * 100.0:.1f}-"
        f"{max(selected_tim) * 100.0:.1f}%`\n"
        f"- Maximum mapped target delta: `{min(selected_delta):.5f}-"
        f"{max(selected_delta):.5f} rad`\n"
        "- Strict hold: every selected trajectory is at least `150 ms`\n"
        "- Source leakage inside the selected batch: `0`\n\n"
        "The selected trajectories are sufficient for a first controlled replay "
        "batch. Start at `0.12 rad/s` and promote to `0.18` and `0.24 rad/s` only "
        "after the preceding tier passes. Actual pen orientation measurement remains "
        "mandatory; this audit does not authorize blind replay or student deployment.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "selected_trajectories"}, indent=2))


if __name__ == "__main__":
    main()
