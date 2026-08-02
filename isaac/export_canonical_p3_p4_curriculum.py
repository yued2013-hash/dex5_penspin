from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports/m3_2_30_index_near_surface"
DEFAULT_SOURCE = REPORT_DIR / "isaac_canonical_p3_physx_projected_bundle.npz"
STAGES = (
    ("entry7", 7.0, "isaac_braked_p4_stage0_bundle.npz", 7.0 / 20.0),
    ("approach20", 20.0, "isaac_braked_p4_stage0_bundle.npz", 1.0),
    ("midway35", 35.0, "isaac_braked_p4_stage1_bundle.npz", 1.0),
    ("braked55", 55.0, "isaac_braked_p4_bundle.npz", 1.0),
)


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def yaw_rotate(quat: np.ndarray, degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) / 2.0
    yaw = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)
    rotated = quat_mul(np.broadcast_to(yaw, quat.shape), quat)
    return rotated / np.linalg.norm(rotated, axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebase P3->P4 curricula on the projected canonical P3.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()

    source_path = args.source.resolve()
    output_dir = args.output_dir.resolve()
    with np.load(source_path) as data:
        source_qpos_one = np.asarray(data["source_qpos"][0], dtype=np.float32)
        source_qvel_one = np.asarray(data["source_qvel"][0], dtype=np.float32)

    count = args.count
    source_qpos = np.repeat(source_qpos_one[None, :], count, axis=0)
    source_qvel = np.repeat(source_qvel_one[None, :], count, axis=0)
    source_hand = source_qpos[:, :20]
    source_pen_pos = source_qpos[:, 20:23]
    source_pen_quat = source_qpos[:, 23:27]
    summaries = []

    for label, phase_deg, legacy_name, local_delta_scale in STAGES:
        legacy_path = REPORT_DIR / legacy_name
        with np.load(legacy_path) as legacy:
            hand_delta = np.median(
                np.asarray(legacy["target_qpos"][:, :20], dtype=np.float32)
                - np.asarray(legacy["source_qpos"][:, :20], dtype=np.float32),
                axis=0,
            )
            pen_pos_delta = np.median(
                np.asarray(legacy["target_qpos"][:, 20:23], dtype=np.float32)
                - np.asarray(legacy["source_qpos"][:, 20:23], dtype=np.float32),
                axis=0,
            )

        target_hand = source_hand + local_delta_scale * hand_delta[None, :]
        target_pen_pos = source_pen_pos + local_delta_scale * pen_pos_delta[None, :]
        # The reward convention treats clockwise planar rotation as positive.
        target_pen_quat = yaw_rotate(source_pen_quat, -phase_deg).astype(np.float32)
        target_qpos = np.concatenate((target_hand, target_pen_pos, target_pen_quat), axis=1)
        output = output_dir / f"isaac_canonical_p3_to_p4_{label}_bundle.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            source_qpos=source_qpos,
            source_qvel=source_qvel,
            source_target_qpos=source_hand,
            source_policy_action=np.zeros((count, 20), dtype=np.float32),
            source_episode_step=np.zeros(count, dtype=np.int64),
            source_teacher_actions=np.zeros((count, 0, 20), dtype=np.float32),
            target_qpos=target_qpos.astype(np.float32),
            target_qvel=np.zeros((count, 26), dtype=np.float32),
            phase_ids=np.full(count, 3, dtype=np.int64),
            source_phase_offset_rad=np.zeros(count, dtype=np.float32),
            target_remaining_rad=np.full(count, np.deg2rad(phase_deg), dtype=np.float32),
            policy_target_pen_quat=source_pen_quat.astype(np.float32),
        )
        metadata = {
            "stage": f"canonical P3 to P4 {label}",
            "phase_target_deg": phase_deg,
            "source": str(source_path.relative_to(PROJECT_ROOT)),
            "legacy_local_delta_source": str(legacy_path.relative_to(PROJECT_ROOT)),
            "output": str(output.relative_to(PROJECT_ROOT)),
            "count": count,
            "legacy_local_delta_scale": local_delta_scale,
            "teacher_trajectories": 0,
            "note": (
                "The reset is the PhysX-projected canonical P3. Pen orientation is a relative "
                "clockwise target; the hand target reuses only the legacy local target delta."
            ),
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summaries.append(metadata)

    print(json.dumps({"stages": summaries}, indent=2))


if __name__ == "__main__":
    main()
