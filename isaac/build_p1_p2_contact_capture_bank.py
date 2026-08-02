from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Build a dynamic P1-to-P2 bank at the end of a real physical contact capture."
)
parser.add_argument("--input", action="append", required=True, type=Path)
parser.add_argument("--source-bundle", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--summary", required=True, type=Path)
parser.add_argument("--required-consecutive-steps", type=int, default=2)
parser.add_argument(
    "--exclude-accepted-trajectories",
    action="store_true",
    help="Keep only partial frontiers from diagnostic bundles.",
)
args = parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_capture_step(mask: np.ndarray, required: int) -> int | None:
    consecutive = 0
    for step, value in enumerate(mask.tolist()):
        consecutive = consecutive + 1 if value else 0
        if consecutive >= required:
            return step
    return None


def main() -> None:
    source_bundle = args.source_bundle.resolve()
    with np.load(source_bundle, allow_pickle=False) as source:
        source_global_ids = np.asarray(source["source_global_bank_ids"], dtype=np.int64)
        original_rows = (
            np.asarray(source["original_source_bundle_row"], dtype=np.int64)
            if "original_source_bundle_row" in source
            else np.arange(source_global_ids.size, dtype=np.int64)
        )
        cluster_ids = (
            np.asarray(source["correction_cluster_id"], dtype=np.int64)
            if "correction_cluster_id" in source
            else np.full(source_global_ids.size, -1, dtype=np.int64)
        )

    rows: list[dict[str, np.ndarray]] = []
    input_summaries = []
    required = int(args.required_consecutive_steps)
    for input_path_arg in args.input:
        input_path = input_path_arg.resolve()
        with np.load(input_path, allow_pickle=False) as teacher:
            required_keys = {
                "source_bundle_row",
                "accepted_step",
                "action",
                "state_qpos",
                "state_qvel",
                "state_target_qpos",
                "contact_force_thumb_index_middle_ring_pinky",
                "ring_feature_m",
                "pen_linear_velocity",
                "pen_angular_velocity",
                "target_condition",
                "semantic_margin",
            }
            missing = sorted(required_keys - set(teacher.files))
            if missing:
                raise ValueError(f"{input_path} is missing state capture keys: {missing}")
            source_rows = np.asarray(teacher["source_bundle_row"], dtype=np.int64)
            accepted = np.asarray(teacher["accepted_step"], dtype=np.bool_)
            accepted_trajectory = (
                np.asarray(teacher["accepted_trajectory"], dtype=np.bool_)
                if "accepted_trajectory" in teacher
                else np.zeros(source_rows.size, dtype=np.bool_)
            )
            capture_count = 0
            for trajectory in range(source_rows.size):
                if args.exclude_accepted_trajectories and accepted_trajectory[trajectory]:
                    continue
                capture_step = first_capture_step(accepted[trajectory], required)
                if capture_step is None:
                    continue
                source_row = int(source_rows[trajectory])
                state_qpos = np.asarray(
                    teacher["state_qpos"][trajectory, capture_step], dtype=np.float32
                )
                state_qvel = np.asarray(
                    teacher["state_qvel"][trajectory, capture_step], dtype=np.float32
                )
                state_target = np.asarray(
                    teacher["state_target_qpos"][trajectory, capture_step],
                    dtype=np.float32,
                )
                action = np.asarray(
                    teacher["action"][trajectory, capture_step], dtype=np.float32
                )
                condition = np.asarray(
                    teacher["target_condition"][trajectory, capture_step],
                    dtype=np.float32,
                )
                contact_force = np.asarray(
                    teacher["contact_force_thumb_index_middle_ring_pinky"][
                        trajectory, capture_step
                    ],
                    dtype=np.float32,
                )
                rows.append(
                    {
                        "source_qpos": state_qpos,
                        "source_qvel": state_qvel,
                        "source_target_qpos": state_target,
                        "source_policy_action": action,
                        "target_qpos": state_qpos.copy(),
                        "target_qvel": state_qvel.copy(),
                        "source_global_bank_ids": np.asarray(
                            source_global_ids[source_row], dtype=np.int64
                        ),
                        "original_source_bundle_row": np.asarray(
                            original_rows[source_row], dtype=np.int64
                        ),
                        "correction_cluster_id": np.asarray(
                            cluster_ids[source_row], dtype=np.int64
                        ),
                        "capture_step": np.asarray(capture_step, dtype=np.int64),
                        "target_distance_mm": np.asarray(
                            np.linalg.norm(condition[:3]) * 10.0, dtype=np.float32
                        ),
                        "semantic_margin": np.asarray(
                            teacher["semantic_margin"][trajectory, capture_step],
                            dtype=np.float32,
                        ),
                        "target_condition": condition,
                        "ring_feature_m": np.asarray(
                            teacher["ring_feature_m"][trajectory, capture_step],
                            dtype=np.float32,
                        ),
                        "contact_force_thumb_index_middle_ring_pinky": contact_force,
                        "pen_linear_velocity": np.asarray(
                            teacher["pen_linear_velocity"][trajectory, capture_step],
                            dtype=np.float32,
                        ),
                        "pen_angular_velocity": np.asarray(
                            teacher["pen_angular_velocity"][trajectory, capture_step],
                            dtype=np.float32,
                        ),
                        "teacher_source": np.asarray(input_path.name),
                    }
                )
                capture_count += 1
        input_summaries.append(
            {
                "path": str(input_path),
                "sha256": sha256(input_path),
                "captured_rows": capture_count,
            }
        )

    if not rows:
        raise RuntimeError("No physical contact-capture rows were found")
    payload = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    row_count = len(rows)
    payload["source_teacher_actions"] = np.zeros(
        (row_count, 1, payload["source_policy_action"].shape[-1]), dtype=np.float32
    )
    payload["phase_ids"] = np.full(row_count, 2, dtype=np.int64)
    payload["policy_success_episode"] = np.zeros(row_count, dtype=np.bool_)
    payload["arrival_bucket"] = np.full(row_count, "PC2", dtype="<U3")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "output": str(output),
        "output_sha256": sha256(output),
        "source_bundle": str(source_bundle),
        "source_bundle_sha256": sha256(source_bundle),
        "inputs": input_summaries,
        "required_consecutive_steps": required,
        "excluded_accepted_trajectories": bool(args.exclude_accepted_trajectories),
        "rows": row_count,
        "unique_original_source_rows": int(
            np.unique(payload["original_source_bundle_row"]).size
        ),
        "cluster_counts": {
            str(cluster): int((payload["correction_cluster_id"] == cluster).sum())
            for cluster in np.unique(payload["correction_cluster_id"])
        },
        "ring_force_n_min_mean_max": [
            float(payload["contact_force_thumb_index_middle_ring_pinky"][:, 3].min()),
            float(payload["contact_force_thumb_index_middle_ring_pinky"][:, 3].mean()),
            float(payload["contact_force_thumb_index_middle_ring_pinky"][:, 3].max()),
        ],
        "target_distance_mm_min_mean_max": [
            float(payload["target_distance_mm"].min()),
            float(payload["target_distance_mm"].mean()),
            float(payload["target_distance_mm"].max()),
        ],
        "normal_physx_step_states": True,
        "ppo_started": False,
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_CONTACT_CAPTURE_BANK=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
