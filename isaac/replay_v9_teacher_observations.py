from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Rebuild policy observations for saved v9 cascade states.")
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--num_envs", type=int, default=1024)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab.utils.math import quat_conjugate, quat_mul  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
STAGES = (
    ("v9_p3_to_entry7_physx_bundle.npz", 7.0, False),
    ("v9_entry7_to_approach20_physx_bundle.npz", 20.0, False),
    ("v9_approach20_to_entry25_physx_bundle.npz", 25.0, False),
    ("v9_continuous_p3_p4_cascade1024_v16_new_brake_report_25_to_30_entry_bundle.npz", 30.0, False),
    ("v9_mid30_to_mid35_physx_bundle.npz", 35.0, False),
    ("v9_mid35_to_approach40_physx_bundle.npz", 40.0, False),
    ("v9_continuous_p3_p4_cascade512_report_40_to_42_entry_bundle.npz", 42.0, False),
    ("v9_cascade42_to_entry45_teacher_bundle.npz", 45.0, False),
    ("v9_continuous_p3_p4_cascade1024_v17_mid30_brake_report_thumb_support_correction_entry_bundle.npz", 44.0, True),
    ("v9_continuous_p3_p4_cascade512_v7_local_frames_report_supported_to_45_entry_bundle.npz", 45.0, False),
    ("v9_continuous_p3_p4_cascade1024_v18_thumb_lowforce_report_supported_to_435_entry_bundle.npz", 45.0, False),
    ("v9_continuous_p3_p4_cascade1024_v19_supported435_report_supported_to_45_entry_bundle.npz", 45.0, False),
    ("v9_continuous_p3_p4_cascade512_v12_report_45_to_47_entry_bundle.npz", 47.0, False),
    ("v9_continuous_p3_p4_cascade512_v13_report_47_capture_entry_bundle.npz", 48.0, False),
    ("v9_continuous_p3_p4_cascade1024_v20_supported45_report_braked_P4_entry_bundle.npz", 50.0, False),
)


def quat_axis(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), axis=-1)


def yaw_rotate(q: np.ndarray, degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) * 0.5
    yaw = np.zeros_like(q)
    yaw[..., 0] = np.cos(half)
    yaw[..., 3] = np.sin(half)
    w1, x1, y1, z1 = np.moveaxis(yaw, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q, -1, 0)
    out = np.stack(
        (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
         w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2), axis=-1
    )
    return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(1e-6)


def nearest_rows(hand: np.ndarray, source_hand: np.ndarray) -> np.ndarray:
    result = []
    for chunk in np.array_split(hand, max(1, (len(hand) + 255) // 256)):
        distance = np.mean((chunk[:, None] - source_hand[None]) ** 2, axis=-1)
        result.append(np.argmin(distance, axis=1))
    return np.concatenate(result)


def build_rows(path: Path, stride: int) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        lengths = data["lengths"]
        qpos_all, qvel_all = data["qpos"], data["qvel"]
        action_all, force_all, stage_all = data["action"], data["contact_force_n"], data["stage_id"]
        initial_qpos = data["initial_qpos"]
    bundles = []
    for name, _, _ in STAGES:
        with np.load(REPORT / name) as data:
            bundles.append((data["source_qpos"].copy(), data["target_qpos"].copy()))

    rows = {key: [] for key in (
        "qpos", "qvel", "action", "previous_action", "contact_force_n", "stage_id",
        "stage_step", "init_pen_pos", "init_pen_axis", "target_hand_qpos",
        "target_pen_pos", "target_pen_quat",
    )}
    for trajectory in range(len(lengths)):
        length = int(lengths[trajectory])
        stages = stage_all[trajectory, :length]
        starts = np.zeros(length, dtype=np.int64)
        start = 0
        for index in range(length):
            if index and stages[index] != stages[index - 1]:
                start = index
            starts[index] = start
        ids = np.arange(0, length, stride)
        qpos = qpos_all[trajectory, ids]
        stage_ids = stages[ids]
        stage_starts = starts[ids]
        target_hand = np.zeros((len(ids), 20), dtype=np.float32)
        target_pos = np.zeros((len(ids), 3), dtype=np.float32)
        target_quat = np.zeros((len(ids), 4), dtype=np.float32)
        for stage_id in np.unique(stage_ids):
            mask = stage_ids == stage_id
            source, target = bundles[int(stage_id)]
            nearest = nearest_rows(qpos[mask, :20], source[:, :20])
            target_hand[mask] = target[nearest, :20]
            target_pos[mask] = qpos[mask, 20:23] + target[nearest, 20:23] - source[nearest, 20:23]
            if STAGES[int(stage_id)][2]:
                target_quat[mask] = qpos[mask, 23:27]
            else:
                target_quat[mask] = yaw_rotate(
                    np.repeat(initial_qpos[trajectory, None, 23:27], int(mask.sum()), axis=0),
                    STAGES[int(stage_id)][1],
                )
        previous = np.zeros((len(ids), 20), dtype=np.float32)
        for j, (index, stage_start) in enumerate(zip(ids, stage_starts)):
            if index > stage_start:
                previous[j] = action_all[trajectory, index - 1]
        local_qpos = qpos_all[trajectory, stage_starts]
        rows["qpos"].append(qpos)
        rows["qvel"].append(qvel_all[trajectory, ids])
        rows["action"].append(action_all[trajectory, ids])
        rows["previous_action"].append(previous)
        rows["contact_force_n"].append(force_all[trajectory, ids])
        rows["stage_id"].append(stage_ids)
        rows["stage_step"].append(ids - stage_starts)
        rows["init_pen_pos"].append(local_qpos[:, 20:23])
        rows["init_pen_axis"].append(quat_axis(local_qpos[:, 23:27]))
        rows["target_hand_qpos"].append(target_hand)
        rows["target_pen_pos"].append(target_pos)
        rows["target_pen_quat"].append(target_quat)
    return {key: np.concatenate(values).astype(np.float32) for key, values in rows.items()}


@hydra_task_config("Dex5-V9P3-Entry7-Direct-v0", "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    source = Path(args.input)
    if not source.is_absolute():
        source = ROOT / source
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    rows = build_rows(source, args.stride)
    total = len(rows["qpos"])
    env_cfg.scene.num_envs = min(args.num_envs, total)
    env_cfg.episode_length_s = 64.0 * env_cfg.sim.dt * env_cfg.decimation
    env = gym.make("Dex5-V9P3-Entry7-Direct-v0", cfg=env_cfg)
    direct = env.unwrapped
    env.reset()
    observations = []
    for start in range(0, total, direct.num_envs):
        stop = min(start + direct.num_envs, total)
        count = stop - start
        env_ids = torch.arange(count, device=direct.device)
        sl = slice(start, stop)
        qpos = torch.as_tensor(rows["qpos"][sl], device=direct.device)
        qvel = torch.as_tensor(rows["qvel"][sl], device=direct.device)
        direct.hand.write_joint_state_to_sim(qpos[:, :20], qvel[:, :20], joint_ids=direct.dex5_joint_ids, env_ids=env_ids)
        pen_pose = qpos[:, 20:27].clone(); pen_pose[:, :3] += direct.scene.env_origins[env_ids]
        direct.pen.write_root_pose_to_sim(pen_pose, env_ids)
        direct.pen.write_root_velocity_to_sim(qvel[:, 20:26], env_ids)
        direct.target_hand_qpos[env_ids] = torch.as_tensor(rows["target_hand_qpos"][sl], device=direct.device)
        direct.target_pen_pos[env_ids] = torch.as_tensor(rows["target_pen_pos"][sl], device=direct.device)
        direct.target_pen_quat[env_ids] = torch.as_tensor(rows["target_pen_quat"][sl], device=direct.device)
        direct.init_pen_pos[env_ids] = torch.as_tensor(rows["init_pen_pos"][sl], device=direct.device)
        direct.init_pen_axis[env_ids] = torch.as_tensor(rows["init_pen_axis"][sl], device=direct.device)
        direct.actions[env_ids] = torch.as_tensor(rows["previous_action"][sl], device=direct.device)
        direct.phase_ids[env_ids] = 3
        direct.episode_length_buf[env_ids] = torch.as_tensor(rows["stage_step"][sl], device=direct.device, dtype=torch.long)
        direct.sim.forward()
        direct.scene.update(dt=0.0)
        direct._compute_intermediate_values()
        direct.contact_force_norm[env_ids] = torch.as_tensor(rows["contact_force_n"][sl], device=direct.device)
        tau = torch.clamp(direct.episode_length_buf[env_ids].float() / max(float(direct.max_episode_length - 1), 1.0), 0.0, 1.0)
        target_quat_error = quat_mul(direct.target_pen_quat[env_ids], quat_conjugate(direct.pen_quat[env_ids]))
        base = torch.cat((
            direct.hand_qpos[env_ids], 0.2 * direct.hand_qvel[env_ids], direct.pen_pos[env_ids],
            direct.pen_quat[env_ids], direct.pen_linvel[env_ids], 0.1 * direct.pen_angvel[env_ids],
            (direct.fingertip_pos_local[env_ids] - direct.pen_pos[env_ids, None]).reshape(count, -1),
            direct.tip_surface_dist[env_ids], 0.1 * direct.contact_force_norm[env_ids],
            direct.target_hand_qpos[env_ids] - direct.hand_qpos[env_ids],
            direct.target_pen_pos[env_ids] - direct.pen_pos[env_ids], target_quat_error,
            direct.actions[env_ids], tau[:, None],
        ), dim=-1)
        phase_one_hot = torch.nn.functional.one_hot(direct.phase_ids[env_ids], num_classes=4).float()
        initial_yaw = torch.atan2(direct.init_pen_axis[env_ids, 1], direct.init_pen_axis[env_ids, 0])
        current_yaw = torch.atan2(direct.pen_axis[env_ids, 1], direct.pen_axis[env_ids, 0])
        planar = torch.atan2(torch.sin(current_yaw - initial_yaw), torch.cos(current_yaw - initial_yaw))
        observations.append(torch.cat((base, phase_one_hot, planar[:, None]), dim=-1).cpu().numpy())
    payload = {"observation": np.concatenate(observations), "action": rows["action"], "stage_id": rows["stage_id"], "source": np.asarray(str(source)), "stride": np.asarray(args.stride)}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {"output": str(output.relative_to(ROOT)), "rows": total, "observation_dim": payload["observation"].shape[1], "stage_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(rows["stage_id"], return_counts=True))}}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
