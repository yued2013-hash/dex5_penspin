from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "dex5_penspin/reports"
SOURCE = REPORTS / "m3_2_30_index_near_surface/isaac_physics_p4_approach_delayed3_contact_states.npz"
OUTPUT = REPORTS / "m3_2_30_index_near_surface/isaac_bridge22_delayed3_contact_bundle.npz"
HOLD_OUTPUT = REPORTS / "m3_2_30_index_near_surface/isaac_bridge22_delayed3_contact_hold_bundle.npz"
STATIC_OUTPUT = REPORTS / "m3_2_30_index_near_surface/isaac_bridge22_delayed3_contact_static_bundle.npz"
PROCESSED_OUTPUT = REPORTS / "m3_2_30_index_near_surface/isaac_bridge22_delayed3_contact_processed_bundle.npz"
TARGET_PHASE_DEG = 22.0


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def build_bundle(
    source: dict[str, np.ndarray], output: Path, hold_source_target: bool, static_actions: bool = False
) -> dict:
    qpos = source["qpos"].astype(np.float32)
    qvel = source["qvel"].astype(np.float32)
    phase_offset = source["directional_deflection_rad"].astype(np.float32)
    controller_target = source["controller_base_target"].astype(np.float32)
    source_target = qpos[:, :20].copy() if hold_source_target else controller_target
    # Do not teleport the hand to the full controller target at reset.  A small
    # fraction preserves the measured contact state while still giving the task
    # a meaningful hand-side target.
    target_hand = qpos[:, :20] + (0.0 if hold_source_target else 0.35) * (controller_target - qpos[:, :20])

    target_phase = np.deg2rad(TARGET_PHASE_DEG)
    remaining = np.clip(target_phase - phase_offset, np.deg2rad(0.5), np.deg2rad(12.0))
    half_yaw = -0.5 * remaining
    yaw_delta = np.stack(
        (np.cos(half_yaw), np.zeros_like(half_yaw), np.zeros_like(half_yaw), np.sin(half_yaw)),
        axis=-1,
    )
    target_quat = quat_multiply(yaw_delta, qpos[:, 23:27])
    target_quat /= np.linalg.norm(target_quat, axis=-1, keepdims=True).clip(1.0e-6)

    # The policy target is the same small bridge target.  The environment can
    # still apply the captured post-contact teacher actions before policy control.
    target_qpos = qpos.copy()
    target_qpos[:, :20] = target_hand
    target_qpos[:, 23:27] = target_quat

    source_teacher = source.get(
        "teacher_actions", np.zeros((qpos.shape[0], 0, 20), dtype=np.float32)
    ).astype(np.float32)
    source_policy_action = source.get("policy_action", np.zeros((len(qpos), 20))).astype(np.float32)
    if static_actions:
        source_policy_action = np.zeros_like(source_policy_action)
        source_teacher = np.zeros_like(source_teacher)
    bundle = {
        "source_qpos": qpos,
        "source_qvel": qvel,
        "source_target_qpos": source_target,
        "source_policy_action": source_policy_action,
        "source_episode_step": source.get("episode_step", np.zeros(len(qpos))).astype(np.int64),
        "source_teacher_actions": source_teacher,
        "target_qpos": target_qpos,
        "target_qvel": qvel.copy(),
        "phase_ids": np.full(len(qpos), 3, dtype=np.int64),
        "source_phase_offset_rad": phase_offset,
        "target_remaining_rad": remaining.astype(np.float32),
        "policy_target_pen_quat": target_quat.astype(np.float32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **bundle)
    metadata = {
        "stage": "P3 delayed contact -> 22 degree bridge",
        "source": str(SOURCE),
        "count": int(len(qpos)),
        "source_capture_delay_steps": int(np.asarray(source.get("capture_delay_steps", 0)).item()),
        "source_phase_deg": {
            "min": float(np.rad2deg(phase_offset).min()),
            "mean": float(np.rad2deg(phase_offset).mean()),
            "max": float(np.rad2deg(phase_offset).max()),
        },
        "target_phase_deg": TARGET_PHASE_DEG,
        "remaining_phase_deg": {
            "min": float(np.rad2deg(remaining).min()),
            "mean": float(np.rad2deg(remaining).mean()),
            "max": float(np.rad2deg(remaining).max()),
        },
        "hand_target_fraction_from_measured_to_controller": 0.0 if hold_source_target else 0.35,
        "reset_actuator_target": "measured capture qpos" if hold_source_target else "controller target",
        "static_actions": static_actions,
        "teacher_steps": int(source_teacher.shape[1]),
        "contact_requirement": "thumb/index/middle were rechecked after delayed capture",
        "purpose": "isolate whether a physically valid delayed contact reset can enter the 22-degree bridge",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    with np.load(SOURCE, allow_pickle=False) as data:
        source = {key: np.asarray(data[key]) for key in data.files}
    metadata = [
        build_bundle(source, OUTPUT, hold_source_target=False),
        build_bundle(source, HOLD_OUTPUT, hold_source_target=True),
        build_bundle(source, STATIC_OUTPUT, hold_source_target=True, static_actions=True),
    ]
    # Rebuild only the actuator-target field from the exact processed target
    # recorded at capture time.  This keeps the remaining bundle identical to
    # the static measured-qpos baseline for a controlled comparison.
    with np.load(STATIC_OUTPUT, allow_pickle=False) as data:
        processed_bundle = {key: np.asarray(data[key]) for key in data.files}
    processed_bundle["source_target_qpos"] = source["target_qpos"].astype(np.float32)
    np.savez_compressed(PROCESSED_OUTPUT, **processed_bundle)
    processed_meta = dict(metadata[-1])
    processed_meta["reset_actuator_target"] = "processed target recorded at capture"
    PROCESSED_OUTPUT.with_suffix(".json").write_text(
        json.dumps(processed_meta, indent=2) + "\n", encoding="utf-8"
    )
    metadata.append(processed_meta)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
