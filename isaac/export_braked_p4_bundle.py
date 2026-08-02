from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "dex5_penspin/reports"
SOURCE = REPORTS / "m3_2_30_index_near_surface/isaac_physics_p3_states.npz"
OUTPUT = REPORTS / "m3_2_30_index_near_surface/isaac_braked_p4_bundle.npz"
TARGET_PHASE_DEG = 55.0


def mujoco_state_to_dex5_order(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).copy()
    values[..., :20] = np.concatenate((values[..., 4:20], values[..., 0:4]), axis=-1)
    return values


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main() -> None:
    source = load_npz(SOURCE)
    if "directional_deflection_rad" not in source:
        raise RuntimeError("P3 state bank must include directional_deflection_rad")

    old_p3 = load_npz(
        REPORTS / "manus_mujoco_presets/corrected_p3_p4_chain_m3_2_38/corrected_true_P3_entry.npz"
    )
    old_brake = load_npz(
        REPORTS / "manus_mujoco_presets/corrected_p3_p4_chain_m3_2_38/B34_middle_handoff.npz"
    )
    old_p3_qpos = mujoco_state_to_dex5_order(old_p3["qpos"])[0]
    old_brake_qpos = mujoco_state_to_dex5_order(old_brake["qpos"])[0]
    brake_hand_delta = old_brake_qpos[:20] - old_p3_qpos[:20]

    approach_path = REPORTS / "m3_2_30_index_near_surface/isaac_physics_p4_approach_hold100_states.npz"
    approach_source = load_npz(approach_path) if approach_path.exists() else source
    midway_path = REPORTS / "m3_2_30_index_near_surface/isaac_physics_p4_midway_states.npz"
    midway_source = load_npz(midway_path) if midway_path.exists() else approach_source
    outputs = (
        (20.0, 20.0, 0.0, source, "P3 physical states", OUTPUT.with_name("isaac_braked_p4_stage0_bundle.npz")),
        (
            35.0,
            20.0,
            0.0,
            approach_source,
            "successful 20deg approach PhysX states",
            OUTPUT.with_name("isaac_braked_p4_stage1_bundle.npz"),
        ),
        (TARGET_PHASE_DEG, 35.0, 0.25, midway_source, "latest successful P4 curriculum states", OUTPUT),
    )
    all_metadata = []
    for target_phase_deg, policy_phase_deg, hand_teacher_scale, stage_source, source_label, output in outputs:
        source_qpos = np.asarray(stage_source["qpos"], dtype=np.float32)
        source_qvel = np.asarray(stage_source["qvel"], dtype=np.float32)
        phase_offset = np.asarray(stage_source["directional_deflection_rad"], dtype=np.float32)
        target_phase = np.deg2rad(target_phase_deg)
        remaining = np.clip(target_phase - phase_offset, np.deg2rad(3.0), np.deg2rad(50.0))

        target_qpos = source_qpos.copy()
        source_active_target = np.asarray(
            stage_source.get("controller_base_target", stage_source["target_qpos"]), dtype=np.float32
        )
        source_policy_action = np.asarray(
            stage_source.get("policy_action", np.zeros_like(source_active_target)), dtype=np.float32
        )
        source_episode_step = np.asarray(
            stage_source.get("episode_step", np.zeros(source_qpos.shape[0])), dtype=np.int64
        )
        source_teacher_actions = np.asarray(
            stage_source.get("teacher_actions", np.zeros((source_qpos.shape[0], 0, 20))),
            dtype=np.float32,
        )
        target_qpos[:, :20] = source_active_target + hand_teacher_scale * brake_hand_delta.astype(np.float32)
        half_yaw = -0.5 * remaining
        yaw_delta = np.stack(
            (np.cos(half_yaw), np.zeros_like(half_yaw), np.zeros_like(half_yaw), np.sin(half_yaw)),
            axis=-1,
        )
        target_qpos[:, 23:27] = quat_multiply(yaw_delta, source_qpos[:, 23:27])
        target_qpos[:, 23:27] /= np.linalg.norm(target_qpos[:, 23:27], axis=-1, keepdims=True)

        policy_remaining = np.clip(np.deg2rad(policy_phase_deg) - phase_offset, 0.0, np.deg2rad(50.0))
        policy_half_yaw = -0.5 * policy_remaining
        policy_yaw_delta = np.stack(
            (
                np.cos(policy_half_yaw),
                np.zeros_like(policy_half_yaw),
                np.zeros_like(policy_half_yaw),
                np.sin(policy_half_yaw),
            ),
            axis=-1,
        )
        policy_target_pen_quat = quat_multiply(policy_yaw_delta, source_qpos[:, 23:27])
        policy_target_pen_quat /= np.linalg.norm(policy_target_pen_quat, axis=-1, keepdims=True)

        target_qvel = source_qvel.copy()
        target_qvel[:, 20:26] = 0.0
        bundle = {
            "source_qpos": source_qpos,
            "source_qvel": source_qvel,
            "source_target_qpos": source_active_target,
            "source_policy_action": source_policy_action,
            "source_episode_step": source_episode_step,
            "source_teacher_actions": source_teacher_actions,
            "target_qpos": target_qpos,
            "target_qvel": target_qvel,
            "phase_ids": np.full(source_qpos.shape[0], 3, dtype=np.int64),
            "source_phase_offset_rad": phase_offset,
            "target_remaining_rad": remaining.astype(np.float32),
            "policy_target_pen_quat": policy_target_pen_quat.astype(np.float32),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **bundle)
        metadata = {
            "stage": "Isaac P3 to braked P4 curriculum",
            "count": int(source_qpos.shape[0]),
            "target_total_phase_deg": target_phase_deg,
            "policy_bootstrap_phase_deg": policy_phase_deg,
            "source_bank": source_label,
            "source_phase_deg": {
                "min": float(np.rad2deg(phase_offset).min()),
                "mean": float(np.rad2deg(phase_offset).mean()),
                "max": float(np.rad2deg(phase_offset).max()),
            },
            "remaining_phase_deg": {
                "min": float(np.rad2deg(remaining).min()),
                "mean": float(np.rad2deg(remaining).mean()),
                "max": float(np.rad2deg(remaining).max()),
            },
            "brake_semantic_teacher": "scaled M3.2.38 B34_middle_handoff hand delta",
            "hand_teacher_scale": hand_teacher_scale,
            "acceptance": "stage-specific phase band, angular-speed brake, T/I/M >=2/3",
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        all_metadata.append(metadata)
    print(json.dumps(all_metadata, indent=2))


if __name__ == "__main__":
    main()
