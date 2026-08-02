from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
P4_BANK = REPORT / "v9_p4_physx_reset_bank.npz"
V9_GEOMETRIC_PHASE_DEG = 78.1817

parser = argparse.ArgumentParser(description="Export a v9 successful phase cache into the next PhysX task bundle.")
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--target_phase_deg", type=float, required=True)
parser.add_argument(
    "--hand_fraction_scale",
    type=float,
    default=1.0,
    help="Scale the geometric hand interpolation while preserving the pen phase target.",
)
parser.add_argument("--source_phase_min_deg", type=float, default=None)
parser.add_argument("--source_phase_max_deg", type=float, default=None)
parser.add_argument(
    "--reset_hand_blend",
    type=float,
    default=1.0,
    help="Blend reset hand qpos from trajectory initial (0) to measured terminal (1).",
)
parser.add_argument("--teacher_report", type=Path, default=None)
parser.add_argument("--teacher_candidate_id", type=int, default=None)
parser.add_argument("--teacher_settle_steps", type=int, default=6)
parser.add_argument("--teacher_drive_steps", type=int, default=10)
parser.add_argument("--teacher_brake_steps", type=int, default=10)
args = parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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
    ).astype(np.float32)


def yaw_rotate(quats: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    half = np.deg2rad(degrees) / 2.0
    yaw = np.zeros_like(quats)
    yaw[:, 0] = np.cos(half)
    yaw[:, 3] = np.sin(half)
    result = quat_mul(yaw, quats)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def main() -> None:
    source_path = resolve(args.source)
    output_path = resolve(args.output)
    with np.load(source_path) as data:
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        rows = np.arange(len(lengths))
        last = lengths - 1
        source_qpos = np.asarray(data["qpos"][rows, last], dtype=np.float32)
        source_qvel = np.asarray(data["qvel"][rows, last], dtype=np.float32)
        source_action = np.asarray(data["action"][rows, last], dtype=np.float32)
        source_target = np.asarray(
            data["actuator_target_qpos"][rows, last], dtype=np.float32
        )
        initial_qpos = np.asarray(data["initial_qpos"], dtype=np.float32)
        source_force = np.asarray(data["contact_force_n"][rows, last], dtype=np.float32)
        source_phase = np.asarray(data["phase_deg"][rows, last], dtype=np.float32)
    selected = np.ones(len(source_phase), dtype=bool)
    if args.source_phase_min_deg is not None:
        selected &= source_phase >= args.source_phase_min_deg
    if args.source_phase_max_deg is not None:
        selected &= source_phase <= args.source_phase_max_deg
    if not np.any(selected):
        raise ValueError("No source trajectories satisfy the requested phase bounds")
    source_qpos = source_qpos[selected]
    source_qvel = source_qvel[selected]
    source_action = source_action[selected]
    source_target = source_target[selected]
    initial_qpos = initial_qpos[selected]
    source_force = source_force[selected]
    source_phase = source_phase[selected]
    with np.load(P4_BANK) as data:
        p4_qpos = np.asarray(data["source_qpos"], dtype=np.float32)
        p4_ids = np.asarray(data.get("source_candidate_ids", np.arange(len(p4_qpos))))

    count = len(source_qpos)
    teacher_actions = np.zeros((count, 0, 20), dtype=np.float32)
    teacher_metadata = None
    if args.teacher_report is not None:
        if args.teacher_candidate_id is None:
            raise ValueError("--teacher_candidate_id is required with --teacher_report")
        teacher_path = resolve(args.teacher_report)
        teacher_report = json.loads(teacher_path.read_text(encoding="utf-8"))
        teacher_row = next(
            row
            for row in teacher_report["top"]
            if row["candidate_id"] == args.teacher_candidate_id
        )
        drive_action = np.asarray(teacher_row["drive_action"], dtype=np.float32)
        brake_action = np.asarray(teacher_row["brake_action"], dtype=np.float32)
        horizon = (
            args.teacher_settle_steps
            + args.teacher_drive_steps
            + args.teacher_brake_steps
        )
        teacher_actions = np.zeros((count, horizon, 20), dtype=np.float32)
        drive_start = args.teacher_settle_steps
        brake_start = drive_start + args.teacher_drive_steps
        teacher_actions[:, drive_start:brake_start] = drive_action
        teacher_actions[:, brake_start:] = brake_action
        teacher_metadata = {
            "report": str(teacher_path.relative_to(ROOT)),
            "candidate_id": args.teacher_candidate_id,
            "schedule": {
                "settle_steps": args.teacher_settle_steps,
                "drive_steps": args.teacher_drive_steps,
                "brake_steps": args.teacher_brake_steps,
            },
        }
    target_ids = np.arange(count) % len(p4_qpos)
    remaining_deg = args.target_phase_deg - source_phase
    remaining_to_geometric = np.maximum(V9_GEOMETRIC_PHASE_DEG - source_phase, 1.0)
    hand_fraction = (
        np.clip(remaining_deg / remaining_to_geometric, 0.0, 1.0)
        * args.hand_fraction_scale
    ).astype(np.float32)
    target_qpos = source_qpos.copy()
    target_qpos[:, :20] = source_qpos[:, :20] + hand_fraction[:, None] * (
        p4_qpos[target_ids, :20] - source_qpos[:, :20]
    )
    target_qpos[:, 20:23] = source_qpos[:, 20:23]
    target_qpos[:, 23:27] = yaw_rotate(source_qpos[:, 23:27], remaining_deg)

    projected_qpos = source_qpos.copy()
    projected_qpos[:, :20] = initial_qpos[:, :20] + args.reset_hand_blend * (
        source_qpos[:, :20] - initial_qpos[:, :20]
    )
    payload = {
        "source_qpos": projected_qpos,
        "source_measured_qpos": source_qpos,
        "source_qvel": np.zeros_like(source_qvel),
        "source_measured_qvel": source_qvel,
        "source_target_qpos": projected_qpos[:, :20].copy(),
        "source_measured_target_qpos": source_target,
        "source_policy_action": np.zeros_like(source_action),
        "source_measured_policy_action": source_action,
        "source_teacher_actions": teacher_actions,
        "target_qpos": target_qpos,
        "target_qvel": np.zeros((count, 26), dtype=np.float32),
        "phase_ids": np.full(count, 3, dtype=np.int64),
        "source_episode_step": np.zeros(count, dtype=np.int64),
        "source_phase_offset_rad": np.deg2rad(source_phase).astype(np.float32),
        "target_remaining_rad": np.deg2rad(remaining_deg).astype(np.float32),
        "policy_target_pen_quat": source_qpos[:, 23:27].copy(),
        "source_contact_force_n": source_force,
        "curriculum_stage_deg": source_phase.copy(),
        "target_p4_candidate_ids": p4_ids[target_ids],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    summary = {
        "output": str(output_path.relative_to(ROOT)),
        "source": str(source_path.relative_to(ROOT)),
        "rows": count,
        "source_phase_deg": {
            "min": float(source_phase.min()),
            "mean": float(source_phase.mean()),
            "max": float(source_phase.max()),
        },
        "target_phase_deg": args.target_phase_deg,
        "hand_fraction_scale": args.hand_fraction_scale,
        "reset_hand_blend_from_initial_to_measured": args.reset_hand_blend,
        "source_phase_filter_deg": {
            "min": args.source_phase_min_deg,
            "max": args.source_phase_max_deg,
        },
        "remaining_phase_deg": {
            "min": float(remaining_deg.min()),
            "mean": float(remaining_deg.mean()),
            "max": float(remaining_deg.max()),
        },
        "hand_target_fraction": {
            "min": float(hand_fraction.min()),
            "mean": float(hand_fraction.mean()),
            "max": float(hand_fraction.max()),
        },
        "planar_direction": "counter_clockwise_positive",
        "teacher": teacher_metadata,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
