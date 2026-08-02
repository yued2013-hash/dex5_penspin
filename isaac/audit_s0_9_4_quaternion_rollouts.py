from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _quat_relative_rotvec(quat: np.ndarray, initial: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat)
    initial = initial / np.linalg.norm(initial)
    scalar = quat[0] * initial[0] + np.dot(quat[1:], initial[1:])
    vector = (
        -quat[0] * initial[1:]
        + initial[0] * quat[1:]
        - np.cross(quat[1:], initial[1:])
    )
    norm = np.linalg.norm(vector)
    if norm < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(norm, abs(float(scalar)))
    sign = 1.0 if scalar >= 0.0 else -1.0
    return vector * (sign * angle / norm)


def _longest_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-deg", type=float, default=35.0)
    parser.add_argument("--tolerance-deg", type=float, default=5.0)
    parser.add_argument("--hold-steps", type=int, default=9)
    args = parser.parse_args()

    reports = []
    for path in args.inputs:
        data = np.load(path)
        poses = data["pen_pose"]
        poses_next = data["pen_pose_next"]
        offsets = data["episode_offsets"]
        pose_velocity = data[
            "pen_angular_velocity_pose_delta"
            if "pen_angular_velocity_pose_delta" in data
            else "pen_angular_velocity"
        ]
        contacts = data[
            "contact_state_next" if "contact_state_next" in data else "contact_state"
        ].astype(bool)
        corrected = data["episode_corrected_success"].reshape(-1).astype(bool)

        gate_seen = np.zeros(6, dtype=np.int64)
        forbidden_history_episodes = 0
        max_progress = []
        max_alignment = []
        longest_hold = []
        episode_rows = []
        lower = math.radians(args.target_deg - args.tolerance_deg)
        upper = math.radians(args.target_deg + args.tolerance_deg)

        for episode_id, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            initial_quat = poses[start, 3:7].astype(np.float64)
            rotvec = np.stack(
                [
                    _quat_relative_rotvec(q.astype(np.float64), initial_quat)
                    for q in poses_next[start:stop, 3:7]
                ]
            )
            progress = rotvec[:, 2]
            geodesic = np.linalg.norm(rotvec, axis=1)
            alignment = np.divide(
                np.abs(progress),
                geodesic,
                out=np.zeros_like(progress),
                where=geodesic > 1.0e-9,
            )
            velocity = pose_velocity[start:stop]
            target_speed = velocity[:, 2]
            wrong_speed = np.linalg.norm(velocity[:, :2], axis=1)
            height_loss = poses[start, 2] - poses_next[start:stop, 2]
            episode_contacts = contacts[start:stop]
            tim = episode_contacts[:, :3].all(axis=1)
            forbidden_now = episode_contacts[:, 3:].any(axis=1) | ~tim
            forbidden_transition = np.zeros_like(forbidden_now)
            consecutive = 0
            latched = False
            for index, forbidden in enumerate(forbidden_now):
                consecutive = consecutive + 1 if forbidden else 0
                latched = latched or consecutive >= 3
                forbidden_transition[index] = latched
            forbidden_history_episodes += int(forbidden_transition.any())

            gates = [
                (progress >= lower) & (progress <= upper) & (alignment >= 0.70),
            ]
            gates.append(gates[-1] & (np.abs(target_speed) <= 2.5))
            gates.append(gates[-1] & (wrong_speed <= 2.0))
            gates.append(gates[-1] & (height_loss <= 0.005))
            gates.append(gates[-1] & tim)
            gates.append(gates[-1] & ~episode_contacts[:, 3:].any(axis=1))
            for index, gate in enumerate(gates):
                gate_seen[index] += int(gate.any())

            hold = _longest_run(gates[-1])
            peak_index = int(np.argmax(progress))
            max_progress.append(float(progress[peak_index]))
            max_alignment.append(float(alignment[peak_index]))
            longest_hold.append(hold)
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "max_target_rotation_deg": math.degrees(float(progress[peak_index])),
                    "axis_alignment_at_max": float(alignment[peak_index]),
                    "longest_gate_hold_steps": hold,
                    "corrected_success": bool(corrected[episode_id]),
                }
            )

        max_progress_array = np.asarray(max_progress)
        longest_hold_array = np.asarray(longest_hold)
        best_indices = np.argsort(max_progress_array)[::-1][:10]
        reports.append(
            {
                "input": str(path.resolve()),
                "episodes": len(max_progress),
                "corrected_successes": int(corrected.sum()),
                "max_target_rotation_deg": math.degrees(float(max_progress_array.max())),
                "p95_target_rotation_deg": math.degrees(
                    float(np.percentile(max_progress_array, 95))
                ),
                "median_target_rotation_deg": math.degrees(
                    float(np.median(max_progress_array))
                ),
                "episodes_at_least_30_deg": int(
                    (max_progress_array >= math.radians(30.0)).sum()
                ),
                "episodes_with_150ms_gate_hold": int(
                    (longest_hold_array >= args.hold_steps).sum()
                ),
                "maximum_gate_hold_steps": int(longest_hold_array.max()),
                "p95_gate_hold_steps": float(np.percentile(longest_hold_array, 95)),
                "episodes_with_any_gate_hold": int((longest_hold_array > 0).sum()),
                "gate_seen_episode_counts": {
                    "angle_and_axis": int(gate_seen[0]),
                    "target_speed": int(gate_seen[1]),
                    "wrong_axis_speed": int(gate_seen[2]),
                    "height": int(gate_seen[3]),
                    "tim_contact": int(gate_seen[4]),
                    "no_ring_or_pinky": int(gate_seen[5]),
                },
                "episodes_with_prior_forbidden_transition": forbidden_history_episodes,
                "top_progress_episodes": [episode_rows[index] for index in best_indices],
            }
        )

    result = {
        "schema": "dex5-s0.9.4-quaternion-rollout-audit-v1",
        "success_contract": {
            "actual_target_rotation_deg": [
                args.target_deg - args.tolerance_deg,
                args.target_deg + args.tolerance_deg,
            ],
            "minimum_axis_alignment": 0.70,
            "maximum_target_speed_rad_s": 2.5,
            "maximum_wrong_axis_speed_rad_s": 2.0,
            "maximum_height_loss_m": 0.005,
            "required_contacts": ["thumb", "index", "middle"],
            "forbidden_contacts": ["ring", "pinky"],
            "hold_steps": args.hold_steps,
            "hold_duration_ms_at_60hz": args.hold_steps / 60.0 * 1000.0,
        },
        "evaluations": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
