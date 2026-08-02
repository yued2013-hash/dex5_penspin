from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Relax every reset row in PhysX and export the settled states.")
parser.add_argument("--task", required=True)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--settle_steps", type=int, default=3)
parser.add_argument(
    "--max_output_penetration_mm",
    type=float,
    default=float("inf"),
    help="Discard projected states above this final instantaneous penetration.",
)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    bundle_path = resolve(args.bundle)
    output_path = resolve(args.output)
    with np.load(bundle_path) as data:
        source_payload = {key: np.asarray(data[key]).copy() for key in data.files}
        row_count = len(data["source_qpos"])

    env_cfg.scene.num_envs = row_count
    env_cfg.bundle_path = str(bundle_path)
    env_cfg.seed = args.seed
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    env_cfg.teacher_prefill_steps = 0
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    env.reset()

    env_ids = torch.arange(row_count, device=direct.device, dtype=torch.long)
    bank_ids = env_ids.clone()
    source_qpos = direct.source_qpos_bank[bank_ids]
    source_qvel = direct.source_qvel_bank[bank_ids]
    direct.current_bank_ids[:] = bank_ids
    if hasattr(direct, "phase_id_bank"):
        direct.phase_ids[:] = direct.phase_id_bank[bank_ids]
    direct.source_targets[:] = direct.source_target_bank[bank_ids]
    direct.target_hand_qpos[:] = direct.target_hand_bank[bank_ids]
    direct.target_pen_pos[:] = direct.target_pen_pos_bank[bank_ids]
    direct.target_pen_quat[:] = direct.target_pen_quat_bank[bank_ids]
    direct.processed_targets[:] = direct.source_targets
    direct.actions.zero_()
    direct.previous_actions.zero_()
    direct.hand.write_joint_state_to_sim(
        source_qpos[:, :20],
        source_qvel[:, :20],
        joint_ids=direct.dex5_joint_ids,
        env_ids=env_ids,
    )
    direct.hand.set_joint_position_target(
        direct.source_targets,
        joint_ids=direct.dex5_joint_ids,
        env_ids=env_ids,
    )
    pen_pose = source_qpos[:, 20:27].clone()
    pen_pose[:, :3] += direct.scene.env_origins
    direct.pen.write_root_pose_to_sim(pen_pose, env_ids)
    direct.pen.write_root_velocity_to_sim(source_qvel[:, 20:26], env_ids)
    direct.init_pen_pos[:] = source_qpos[:, 20:23]
    direct.init_pen_axis[:] = quat_apply(source_qpos[:, 23:27], direct.pen_x_axis)
    direct.episode_max_penetration.zero_()
    direct.episode_max_penetration_by_finger.zero_()
    direct.episode_length_buf.zero_()

    zero_actions = torch.zeros((row_count, direct.cfg.action_space), device=direct.device)
    for _ in range(args.settle_steps):
        direct._pre_physics_step(zero_actions)
        for _ in range(direct.cfg.decimation):
            direct._apply_action()
            direct.scene.write_data_to_sim()
            direct.sim.step(render=False)
            direct.scene.update(dt=direct.physics_dt)
        direct.episode_length_buf += 1
        direct._compute_intermediate_values()

    relaxed_qpos = torch.cat((direct.hand_qpos, direct.pen_pos, direct.pen_quat), dim=-1)
    relaxed_qvel = torch.cat((direct.hand_qvel, direct.pen_linvel, direct.pen_angvel), dim=-1)
    if hasattr(direct.cfg, "target_total_phase_deg"):
        target_axis = quat_apply(
            direct.target_pen_quat, direct.pen_x_axis
        )
        current_yaw = torch.atan2(
            direct.pen_axis[:, 1], direct.pen_axis[:, 0]
        )
        target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
        direction = float(
            getattr(direct.cfg, "phase_direction_sign", -1.0)
        )
        remaining = direction * torch.atan2(
            torch.sin(target_yaw - current_yaw),
            torch.cos(target_yaw - current_yaw),
        )
        phase_deg = (
            direct.cfg.target_total_phase_deg
            - remaining * (180.0 / torch.pi)
        )
    else:
        remaining = torch.zeros(row_count, device=direct.device)
        phase_deg = torch.zeros(row_count, device=direct.device)

    payload = source_payload
    payload["source_qpos"] = relaxed_qpos.cpu().numpy().astype(np.float32)
    payload["source_measured_qpos"] = payload["source_qpos"].copy()
    payload["source_qvel"] = np.zeros_like(relaxed_qvel.cpu().numpy(), dtype=np.float32)
    payload["source_measured_qvel"] = relaxed_qvel.cpu().numpy().astype(np.float32)
    payload["source_target_qpos"] = direct.hand_qpos.cpu().numpy().astype(np.float32)
    payload["source_measured_target_qpos"] = direct.processed_targets.cpu().numpy().astype(np.float32)
    payload["source_policy_action"] = np.zeros((row_count, 20), dtype=np.float32)
    payload["source_measured_policy_action"] = direct.actions.cpu().numpy().astype(np.float32)
    payload["source_episode_step"] = np.zeros(row_count, dtype=np.int64)
    payload["source_phase_offset_rad"] = np.deg2rad(phase_deg.cpu().numpy()).astype(np.float32)
    payload["target_remaining_rad"] = remaining.cpu().numpy().astype(np.float32)
    payload["policy_target_pen_quat"] = direct.pen_quat.cpu().numpy().astype(np.float32)
    payload["source_contact_force_n"] = direct.contact_force_norm.cpu().numpy().astype(np.float32)
    payload["curriculum_stage_deg"] = phase_deg.cpu().numpy().astype(np.float32)
    payload["relaxed_current_penetration_m"] = direct.current_max_penetration.cpu().numpy().astype(np.float32)
    relaxed_penetration = payload["relaxed_current_penetration_m"]
    valid = (
        payload["relaxed_current_penetration_m"]
        <= args.max_output_penetration_mm / 1000.0
    )
    valid &= np.isfinite(payload["source_qpos"]).all(axis=1)
    payload = {
        key: (
            value[valid]
            if value.ndim > 0 and len(value) == row_count
            else value
        )
        for key, value in payload.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    summary = {
        "output": str(output_path.relative_to(ROOT)),
        "source": str(bundle_path.relative_to(ROOT)),
        "source_rows": row_count,
        "rows": int(valid.sum()),
        "max_output_penetration_mm": args.max_output_penetration_mm,
        "settle_steps": args.settle_steps,
        "settle_ms": args.settle_steps * direct.cfg.decimation * direct.cfg.sim.dt * 1000.0,
        "phase_deg": {
            "min": float(phase_deg.min().item()),
            "mean": float(phase_deg.mean().item()),
            "max": float(phase_deg.max().item()),
        },
        "current_penetration_mm": {
            "mean": float(relaxed_penetration[valid].mean() * 1000.0),
            "max": float(relaxed_penetration[valid].max() * 1000.0),
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
