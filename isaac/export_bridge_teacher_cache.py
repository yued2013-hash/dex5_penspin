"""Export a bridge-specific BC cache from the validated physical P4 hold bank."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
SOURCE = REPORT / "isaac_physics_p4_approach_hold100_states.npz"
FULL_TARGET = REPORT / "isaac_braked_p4_stage1_bundle.npz"
OUTPUT = REPORT / "isaac_bridge22_teacher_cache.npz"
PRESERVE_OUTPUT = REPORT / "isaac_bridge22_thumb_preserve_teacher_cache.npz"
SUMMARY = REPORT / "isaac_bridge22_teacher_cache.json"


def quat_lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    b = b.copy()
    signs = np.sum(a * b, axis=-1, keepdims=True) < 0.0
    b[signs[..., 0]] *= -1.0
    q = (1.0 - alpha) * a + alpha * b
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1.0e-6)


def main() -> None:
    with np.load(SOURCE, allow_pickle=False) as data:
        source_qpos = data["qpos"].astype(np.float32)
        source_qvel = data["qvel"].astype(np.float32)
        source_target = data["target_qpos"].astype(np.float32)
        teacher_actions = data["teacher_actions"].astype(np.float32)
        policy_action = data["policy_action"].astype(np.float32)
        episode_step = data["episode_step"].astype(np.int64)
    with np.load(FULL_TARGET, allow_pickle=False) as full_data:
        full_target = full_data["target_qpos"].astype(np.float32)

    # The validated source is approximately the 20-degree hold.  The full stage1
    # target is approximately 35 degrees, so this cache encodes a small 22-degree
    # bridge target while preserving the measured dynamic state and teacher action.
    alpha = 0.15
    bridge_target = source_qpos.copy()
    bridge_target[:, :20] = (1.0 - alpha) * source_target + alpha * full_target[:, :20]
    bridge_target[:, 20:23] = (1.0 - alpha) * source_qpos[:, 20:23] + alpha * full_target[:, 20:23]
    bridge_target[:, 23:27] = quat_lerp(
        source_qpos[:, 23:27],
        full_target[:, 23:27],
        alpha,
    )
    thumb_preserve_target = bridge_target.copy()
    thumb_preserve_target[:, :20] = source_target
    np.savez_compressed(
        OUTPUT,
        source_qpos=source_qpos,
        source_qvel=source_qvel,
        source_target_qpos=source_target,
        bridge_target_qpos=bridge_target,
        target_qpos=bridge_target,
        target_qvel=source_qvel,
        phase_ids=np.full(source_qpos.shape[0], 3, dtype=np.int64),
        teacher_actions=teacher_actions,
        source_teacher_actions=teacher_actions,
        source_policy_action=policy_action,
        source_episode_step=episode_step,
        bridge_phase_deg=np.full(source_qpos.shape[0], 22.0, dtype=np.float32),
        teacher_steps=np.asarray(teacher_actions.shape[1], dtype=np.int64),
    )
    np.savez_compressed(
        PRESERVE_OUTPUT,
        source_qpos=source_qpos,
        source_qvel=source_qvel,
        source_target_qpos=source_target,
        bridge_target_qpos=thumb_preserve_target,
        target_qpos=thumb_preserve_target,
        target_qvel=source_qvel,
        phase_ids=np.full(source_qpos.shape[0], 3, dtype=np.int64),
        source_teacher_actions=teacher_actions,
        source_policy_action=policy_action,
        source_episode_step=episode_step,
        bridge_phase_deg=np.full(source_qpos.shape[0], 22.0, dtype=np.float32),
        teacher_steps=np.asarray(teacher_actions.shape[1], dtype=np.int64),
    )
    summary = {
        "stage": "Bridge22 teacher/BC warm-start cache",
        "source": str(SOURCE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "thumb_preserve_output": str(PRESERVE_OUTPUT.relative_to(ROOT)),
        "states": int(source_qpos.shape[0]),
        "teacher_trajectories": int(teacher_actions.shape[0]),
        "teacher_steps": int(teacher_actions.shape[1]),
        "target_phase_deg": 22.0,
        "source_phase_deg": 20.0,
        "status": "warm-start cache; not an independently successful Bridge22 rollout",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
