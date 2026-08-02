from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/"
    "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def rotation_noise(rng: np.random.Generator, count: int, std_deg: float, clip_deg: float) -> np.ndarray:
    rotvec = np.clip(
        rng.normal(0.0, np.deg2rad(std_deg), size=(count, 3)),
        -np.deg2rad(clip_deg),
        np.deg2rad(clip_deg),
    )
    angle = np.linalg.norm(rotvec, axis=1, keepdims=True)
    axis = rotvec / np.maximum(angle, 1.0e-12)
    return np.concatenate((np.cos(0.5 * angle), axis * np.sin(0.5 * angle)), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build V9-near candidates for PhysX reset projection."
    )
    parser.add_argument("--phase", type=int, choices=(1, 2, 3, 4, 5), required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--center-bank", type=Path)
    parser.add_argument("--center-label", default="", help="Optional boolean label used to select rows from --center-bank.")
    parser.add_argument("--center-labels", nargs="*", default=None)
    parser.add_argument("--center-contact-ids", type=int, nargs="*", default=None)
    parser.add_argument("--only-emphasis-joints", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint-noise-std", type=float, default=0.020)
    parser.add_argument("--joint-noise-clip", type=float, default=0.055)
    parser.add_argument(
        "--emphasis-noise-mean",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional directed offset for the phase emphasis joints, in the same "
            "order as that phase's emphasis-joint tuple."
        ),
    )
    parser.add_argument("--pen-pos-std-mm", type=float, default=1.8)
    parser.add_argument("--pen-pos-clip-mm", type=float, default=4.0)
    parser.add_argument("--pen-rot-std-deg", type=float, default=1.5)
    parser.add_argument("--pen-rot-clip-deg", type=float, default=4.0)
    parser.add_argument("--max-hand-rms-rad", type=float, default=0.12)
    parser.add_argument(
        "--target-mode",
        choices=("projected_q", "preserve_offset"),
        default="projected_q",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    canonical_id = f"P{args.phase}"
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    phase = next(row for row in manifest["phases"] if int(row["phase"]) == args.phase)
    reference_q = np.asarray(phase["hand_qpos"], dtype=np.float32)

    audited_path = (
        args.center_bank.resolve()
        if args.center_bank is not None
        else root / "validation_reports" / f"{canonical_id}_audited.npz"
    )
    with np.load(audited_path, allow_pickle=False) as loaded:
        audited = {key: loaded[key] for key in loaded.files}
    requested_labels = [label for label in (args.center_labels or ()) if label]
    if args.center_label:
        requested_labels.append(args.center_label)
    if args.center_bank is not None and requested_labels:
        label_gate = np.ones(audited["robot_qpos"].shape[0], dtype=np.bool_)
        for label in requested_labels:
            if label not in audited:
                raise KeyError(f"Missing center label {label!r} in {audited_path}")
            label_gate &= np.asarray(audited[label], dtype=np.bool_)
        r0_rows = np.flatnonzero(label_gate)
    elif args.center_bank is not None:
        r0_rows = np.arange(audited["robot_qpos"].shape[0], dtype=np.int64)
    else:
        r0_rows = np.flatnonzero(audited["labels_r0_pass"])
    if args.center_contact_ids:
        contact_gate = np.asarray(audited["contact_state"], dtype=np.bool_)[
            :, args.center_contact_ids
        ].all(axis=1)
        r0_rows = r0_rows[contact_gate[r0_rows]]
    if r0_rows.size == 0:
        raise RuntimeError(f"{canonical_id} has no R0-valid centers")

    rng = np.random.default_rng(args.seed + args.phase)
    selected = rng.choice(r0_rows, size=args.count, replace=True)
    payload = {key: value[selected].copy() for key, value in audited.items()}
    payload["metadata_parent_state_id"] = payload["metadata_state_id"].copy()
    # Reserve one million IDs per seed so independently generated candidate
    # rounds cannot overlap when their row ranges are concatenated.
    state_base = np.int64(args.phase * 1_000_000_000_000_000 + args.seed * 1_000_000)
    payload["metadata_state_id"] = state_base + np.arange(args.count, dtype=np.int64)
    payload["labels_source"] = np.full(
        args.count, f"{canonical_id}_v9_physx_projection", dtype="U48"
    )

    centers = np.asarray(payload["robot_qpos"], dtype=np.float32)
    center_target_offset = (
        np.asarray(payload["robot_q_target"], dtype=np.float32) - centers
    )
    q_noise = np.clip(
        rng.normal(0.0, args.joint_noise_std, size=(args.count, 20)),
        -args.joint_noise_clip,
        args.joint_noise_clip,
    ).astype(np.float32)
    # Dex5 control order is index, middle, ring, pinky, thumb.
    emphasis = {
        1: (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19),
        2: (8, 9, 10, 11),
        3: (0, 1, 2, 3),
        4: (16, 17, 18, 19),
        5: (0, 1, 2, 3, 16, 17, 18, 19),
    }
    if args.only_emphasis_joints:
        keep = np.zeros(20, dtype=np.bool_)
        keep[list(emphasis[args.phase])] = True
        q_noise[:, ~keep] = 0.0
    q_noise[:, emphasis[args.phase]] *= 1.6
    if args.emphasis_noise_mean is not None:
        mean = np.asarray(args.emphasis_noise_mean, dtype=np.float32)
        expected = len(emphasis[args.phase])
        if mean.shape != (expected,):
            raise ValueError(
                f"P{args.phase} expects {expected} emphasis means, got {mean.size}"
            )
        q_noise[:, emphasis[args.phase]] += mean[None, :]
    projected_q = centers + q_noise
    rms = np.sqrt(np.mean((projected_q - reference_q[None, :]) ** 2, axis=1))
    over = rms > args.max_hand_rms_rad
    if np.any(over):
        scale = args.max_hand_rms_rad / np.maximum(rms[over], 1.0e-9)
        projected_q[over] = reference_q + (
            projected_q[over] - reference_q
        ) * scale[:, None]
    payload["robot_qpos"] = projected_q.astype(np.float32)
    payload["robot_qvel"] = np.zeros_like(payload["robot_qvel"])
    projected_target = projected_q.astype(np.float32)
    if args.target_mode == "preserve_offset":
        projected_target = projected_target + center_target_offset
    payload["robot_q_target"] = projected_target.astype(np.float32)
    payload["robot_previous_action"] = np.zeros_like(payload["robot_previous_action"])
    payload["robot_action_history_buffer"] = np.zeros_like(
        payload["robot_action_history_buffer"]
    )
    payload["robot_control_delay_buffer"] = np.zeros_like(
        payload["robot_control_delay_buffer"]
    )
    payload["robot_continuation_q_target"][:] = projected_target[:, None, :]
    payload["robot_continuation_action"] = np.zeros_like(
        payload["robot_continuation_action"]
    )

    position_offset = np.clip(
        rng.normal(0.0, args.pen_pos_std_mm / 1000.0, size=(args.count, 3)),
        -args.pen_pos_clip_mm / 1000.0,
        args.pen_pos_clip_mm / 1000.0,
    ).astype(np.float32)
    payload["object_pose"][:, :3] += position_offset
    payload["object_com"] += position_offset
    delta_quat = rotation_noise(
        rng, args.count, args.pen_rot_std_deg, args.pen_rot_clip_deg
    ).astype(np.float32)
    quat = quat_multiply(delta_quat, payload["object_pose"][:, 3:7])
    quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1.0e-9)
    payload["object_pose"][:, 3:7] = quat.astype(np.float32)
    payload["object_linear_velocity"] = np.zeros_like(payload["object_linear_velocity"])
    payload["object_angular_velocity"] = np.zeros_like(payload["object_angular_velocity"])

    for key in (
        "labels_r0_pass",
        "labels_r1_pass",
        "labels_r2_pass",
        "labels_r3a_pass",
        "labels_r3b_pass",
    ):
        payload[key] = np.zeros(args.count, dtype=np.bool_)
    payload["labels_validation_status"] = np.full(
        args.count, "candidate", dtype="U10"
    )
    payload["labels_rejection_code"] = np.full(args.count, "unvalidated", dtype="U64")
    payload["contact_state"] = np.zeros_like(payload["contact_state"])
    payload["contact_force"] = np.zeros_like(payload["contact_force"])
    payload["contact_state_available"] = np.zeros(args.count, dtype=np.bool_)
    payload["contact_force_available"] = np.zeros(args.count, dtype=np.bool_)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    final_rms = np.sqrt(np.mean((projected_q - reference_q[None, :]) ** 2, axis=1))
    report = {
        "canonical_id": canonical_id,
        "count": args.count,
        "r0_center_count": int(r0_rows.size),
        "center_bank": str(audited_path),
        "median_hand_rms_rad": float(np.median(final_rms)),
        "max_hand_rms_rad": float(np.max(final_rms)),
        "max_pen_position_offset_mm": float(
            np.linalg.norm(position_offset, axis=1).max() * 1000.0
        ),
        "emphasis_noise_mean": args.emphasis_noise_mean,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
