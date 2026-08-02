from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports/m3_2_30_index_near_surface"
DEFAULT_SOURCE = REPORT_DIR / "canonical_p3_p4_entry7_teacher50.npz"
DEFAULT_TARGET = REPORT_DIR / "isaac_canonical_p3_to_p4_approach20_bundle.npz"
DEFAULT_OUTPUT = REPORT_DIR / "isaac_canonical_p3_entry7_to_p4_approach20_bundle.npz"


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float32,
    )


def yaw_rotate(quat: np.ndarray, degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) / 2.0
    yaw = np.asarray((np.cos(half), 0.0, 0.0, np.sin(half)), dtype=np.float32)
    rotated = quat_mul(yaw, quat)
    return rotated / np.linalg.norm(rotated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 7-degree to 20-degree P4 curriculum bundle.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target_phase_deg", type=float, default=20.0)
    parser.add_argument("--preserve_source_hand", action="store_true")
    args = parser.parse_args()

    source_path = args.source.resolve()
    target_path = args.target.resolve()
    output_path = args.output.resolve()
    with np.load(source_path) as source, np.load(target_path) as target:
        source_qpos = np.asarray(source["qpos"], dtype=np.float32)
        source_qvel = np.asarray(source["qvel"], dtype=np.float32)
        count = source_qpos.shape[0]
        p3_qpos = np.asarray(target["source_qpos"][0], dtype=np.float32)
        approach20_qpos = np.asarray(target["target_qpos"][0], dtype=np.float32)
        fraction = args.target_phase_deg / 20.0
        target_qpos_one = p3_qpos.copy()
        target_qpos_one[:23] = p3_qpos[:23] + fraction * (approach20_qpos[:23] - p3_qpos[:23])
        target_qpos_one[23:27] = yaw_rotate(p3_qpos[23:27], -args.target_phase_deg)
        target_qpos = np.repeat(target_qpos_one[None, :], count, axis=0)
        if args.preserve_source_hand:
            target_qpos[:, :23] = source_qpos[:, :23]
        target_qvel = np.zeros((count, 26), dtype=np.float32)
        source_phase = np.asarray(source["directional_deflection_rad"], dtype=np.float32)
        payload = {
            "source_qpos": source_qpos,
            "source_qvel": source_qvel,
            "source_target_qpos": np.asarray(source["target_qpos"], dtype=np.float32),
            "source_policy_action": np.asarray(source["policy_action"], dtype=np.float32),
            "source_episode_step": np.zeros(count, dtype=np.int64),
            "source_teacher_actions": np.asarray(source["teacher_actions"], dtype=np.float32),
            "target_qpos": target_qpos,
            "target_qvel": target_qvel,
            "phase_ids": np.full(count, 3, dtype=np.int64),
            "source_phase_offset_rad": source_phase,
            "target_remaining_rad": np.float32(np.deg2rad(args.target_phase_deg)) - source_phase,
            "policy_target_pen_quat": source_qpos[:, 23:27],
            "source_contact_force_n": np.asarray(source["contact_force_n"], dtype=np.float32),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    metadata = {
        "stage": f"canonical P3 7-degree entry to P4 {args.target_phase_deg:g}-degree target",
        "source": str(source_path.relative_to(PROJECT_ROOT)),
        "target": str(target_path.relative_to(PROJECT_ROOT)),
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "states": int(count),
        "teacher_trajectories": int(count),
        "teacher_steps": int(payload["source_teacher_actions"].shape[1]),
        "source_phase_deg": {
            "min": float(np.rad2deg(source_phase).min()),
            "mean": float(np.rad2deg(source_phase).mean()),
            "max": float(np.rad2deg(source_phase).max()),
        },
        "target_phase_deg": args.target_phase_deg,
        "preserve_source_hand": args.preserve_source_hand,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
