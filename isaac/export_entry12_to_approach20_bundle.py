from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "reports/m3_2_30_index_near_surface"


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
    result = quat_mul(yaw, quat)
    return result / np.linalg.norm(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a physical 12-to-20 degree curriculum bundle.")
    parser.add_argument(
        "--source", type=Path, default=REPORT / "canonical_entry7_to_entry12_teacher50.npz"
    )
    parser.add_argument(
        "--target", type=Path, default=REPORT / "isaac_canonical_p3_to_p4_approach20_bundle.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=REPORT / "isaac_canonical_entry12_to_p4_approach20_bundle.npz"
    )
    parser.add_argument("--target-phase-deg", type=float, default=20.0)
    args = parser.parse_args()

    source_path = args.source.resolve()
    target_path = args.target.resolve()
    output_path = args.output.resolve()
    with np.load(source_path) as source, np.load(target_path) as target:
        source_qpos = np.asarray(source["capture_qpos"], dtype=np.float32)
        source_qvel = np.asarray(source["capture_qvel"], dtype=np.float32)
        count = len(source_qpos)
        target_qpos = np.repeat(
            np.asarray(target["target_qpos"][0], dtype=np.float32)[None, :], count, axis=0
        )
        # Preserve each measured capture hand pose and pen position. Only the pen
        # orientation advances to the canonical 20-degree target.
        target_qpos[:, :23] = source_qpos[:, :23]
        canonical_p3_quat = np.asarray(target["source_qpos"][0, 23:27], dtype=np.float32)
        target_qpos[:, 23:27] = yaw_rotate(canonical_p3_quat, -args.target_phase_deg)
        success_steps = np.asarray(source["success_step"], dtype=np.int64)
        teacher_actions = np.asarray(source["teacher_actions"], dtype=np.float32)
        source_policy_action = teacher_actions[np.arange(count), success_steps - 1]
        source_phase = np.asarray(source["capture_phase_rad"], dtype=np.float32)
        payload = {
            "source_qpos": source_qpos,
            "source_qvel": source_qvel,
            "source_target_qpos": np.asarray(source["capture_target_qpos"], dtype=np.float32),
            "source_policy_action": source_policy_action,
            "source_episode_step": np.zeros(count, dtype=np.int64),
            "source_teacher_actions": np.zeros((count, 0, 20), dtype=np.float32),
            "target_qpos": target_qpos,
            "target_qvel": np.zeros((count, 26), dtype=np.float32),
            "phase_ids": np.full(count, 3, dtype=np.int64),
            "source_phase_offset_rad": source_phase,
            "target_remaining_rad": np.float32(np.deg2rad(args.target_phase_deg)) - source_phase,
            "policy_target_pen_quat": source_qpos[:, 23:27],
            "source_contact_force_n": np.asarray(source["capture_contact_force_n"], dtype=np.float32),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    metadata = {
        "stage": f"physical capture to P4 phase {args.target_phase_deg:g}",
        "source": str(source_path.relative_to(PROJECT_ROOT)),
        "target": str(target_path.relative_to(PROJECT_ROOT)),
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "states": count,
        "source_phase_deg": {
            "min": float(np.rad2deg(source_phase).min()),
            "mean": float(np.rad2deg(source_phase).mean()),
            "max": float(np.rad2deg(source_phase).max()),
        },
        "target_phase_deg": args.target_phase_deg,
        "teacher_prefill_steps": 0,
        "preserved": ["measured hand pose", "measured pen position", "measured qvel"],
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
