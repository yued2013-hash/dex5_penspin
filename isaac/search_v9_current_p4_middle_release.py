from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Search a collision-free radial middle-finger release from the current "
        "P3-to-braked-P4 terminal distribution."
    )
)
parser.add_argument(
    "--task", default="Dex5-V9P4-P5-CurrentP4MiddleClear-Direct-v0"
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--hold-steps", type=int, default=12)
parser.add_argument("--seed", type=int, default=2410)
parser.add_argument("--servo-gain", type=float, default=0.20)
parser.add_argument("--parent", type=Path)
parser.add_argument("--jitter", type=float, default=0.10)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        (args.steps + 2) * env_cfg.decimation * env_cfg.sim.dt,
    )
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    env.reset()

    generator = torch.Generator(device=direct.device).manual_seed(args.seed)
    lower = direct.joint_lower[:, 4:8]
    upper = direct.joint_upper[:, 4:8]
    source_middle = direct.hand_qpos[:, 4:8].clone()
    if args.parent is None:
        local_delta = torch.tensor(
            [0.45, 0.85, 1.20, 1.20], device=direct.device
        )
        local = source_middle + (
            2.0
            * torch.rand(
                source_middle.shape,
                generator=generator,
                device=direct.device,
            )
            - 1.0
        ) * local_delta
        broad = lower + torch.rand(
            lower.shape, generator=generator, device=direct.device
        ) * (upper - lower)
        broad_rows = (
            torch.arange(args.num_envs, device=direct.device) % 4 == 0
        ).unsqueeze(-1)
        candidate_middle = torch.where(broad_rows, broad, local)
    else:
        parent_path = args.parent
        if not parent_path.is_absolute():
            parent_path = Path(__file__).resolve().parents[1] / parent_path
        with np.load(parent_path) as parent:
            parent_targets = torch.as_tensor(
                parent["candidate_middle_target"],
                dtype=torch.float32,
                device=direct.device,
            )
        parent_ids = torch.randint(
            len(parent_targets),
            (args.num_envs,),
            generator=generator,
            device=direct.device,
        )
        candidate_middle = parent_targets[parent_ids] + (
            args.jitter
            * (upper - lower)
            * torch.randn(
                source_middle.shape,
                generator=generator,
                device=direct.device,
            )
        )
    candidate_middle = torch.maximum(
        torch.minimum(candidate_middle, upper), lower
    )
    desired_targets = direct.policy_targets.clone()
    desired_targets[:, 4:8] = candidate_middle

    failed_ever = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    clear_run = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    max_clear_run = torch.zeros_like(clear_run)
    support_steps = torch.zeros(args.num_envs, device=direct.device)
    max_penetration = direct.episode_max_penetration.clone()
    max_force = direct.contact_force_norm.max(dim=-1).values.clone()
    max_pen_drift = torch.zeros(args.num_envs, device=direct.device)
    max_axis_drift = torch.zeros(args.num_envs, device=direct.device)
    max_angular_speed = torch.zeros(args.num_envs, device=direct.device)
    success_step = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    success_hand_qpos = torch.zeros_like(direct.hand_qpos)
    success_hand_qvel = torch.zeros_like(direct.hand_qvel)
    success_pen_pos = torch.zeros_like(direct.pen_pos)
    success_pen_quat = torch.zeros_like(direct.pen_quat)
    success_pen_linvel = torch.zeros_like(direct.pen_linvel)
    success_pen_angvel = torch.zeros_like(direct.pen_angvel)
    success_policy_targets = torch.zeros_like(direct.policy_targets)
    success_action = torch.zeros_like(direct.actions)
    success_contact_force = torch.zeros_like(direct.contact_force_norm)
    success_penetration = torch.zeros_like(max_penetration)
    observation_trace: list[torch.Tensor] = []
    action_trace: list[torch.Tensor] = []

    for step in range(args.steps):
        observation_trace.append(
            direct._get_observations()["policy"].detach().cpu()
        )
        mask = direct._stage_action_mask()
        increment = 0.25 * direct.cfg.action_scale * mask
        error = desired_targets - direct.policy_targets
        actions = args.servo_gain * error / increment.clamp_min(1.0e-6)
        actions = torch.where(mask > 0.0, actions, torch.zeros_like(actions))
        actions.clamp_(-1.0, 1.0)
        action_trace.append(actions.detach().cpu())

        direct._pre_physics_step(actions)
        for _ in range(direct.cfg.decimation):
            direct._apply_action()
            direct.scene.write_data_to_sim()
            direct.sim.step(render=False)
            direct.scene.update(dt=direct.physics_dt)
        direct.episode_length_buf += 1
        direct._compute_intermediate_values()
        failed, _ = direct._compute_dones()
        failed_ever |= failed

        contacts = direct.contact_force_norm > direct.cfg.p5_contact_force_n
        support = contacts[:, 0] & contacts[:, 1]
        middle_gap = (
            direct.tip_surface_dist[:, 2] - direct.fingertip_radii[:, 2]
        )
        middle_clear = (
            direct.contact_force_norm[:, 2] < direct.cfg.p5_clear_force_n
        ) | (middle_gap > direct.cfg.p5_clear_surface_distance_m)
        drift = torch.linalg.norm(
            direct.pen_pos - direct.init_pen_pos, dim=-1
        )
        axis_dot = torch.sum(
            direct.pen_axis * direct.init_pen_axis, dim=-1
        ).clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot.abs())
        angular_speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        safe = (
            support
            & middle_clear
            & (drift <= direct.cfg.p5_max_position_drift_m)
            & (
                axis_drift
                <= torch.deg2rad(
                    torch.full_like(
                        axis_drift, direct.cfg.p5_max_axis_drift_deg
                    )
                )
            )
            & (angular_speed <= direct.cfg.p5_max_angular_speed_rad_s)
            & (
                direct.episode_max_penetration
                <= direct.cfg.p5_max_penetration_m
            )
            & ~failed_ever
        )
        clear_run = torch.where(
            safe, clear_run + 1, torch.zeros_like(clear_run)
        )
        max_clear_run = torch.maximum(max_clear_run, clear_run)
        newly_successful = (
            (clear_run >= args.hold_steps) & (success_step < 0)
        )
        success_step[newly_successful] = step
        success_hand_qpos[newly_successful] = direct.hand_qpos[
            newly_successful
        ]
        success_hand_qvel[newly_successful] = direct.hand_qvel[
            newly_successful
        ]
        success_pen_pos[newly_successful] = direct.pen_pos[newly_successful]
        success_pen_quat[newly_successful] = direct.pen_quat[newly_successful]
        success_pen_linvel[newly_successful] = direct.pen_linvel[
            newly_successful
        ]
        success_pen_angvel[newly_successful] = direct.pen_angvel[
            newly_successful
        ]
        success_policy_targets[newly_successful] = direct.policy_targets[
            newly_successful
        ]
        success_action[newly_successful] = actions[newly_successful]
        success_contact_force[newly_successful] = direct.contact_force_norm[
            newly_successful
        ]
        success_penetration[newly_successful] = (
            direct.episode_max_penetration[newly_successful]
        )
        support_steps += support.float()
        max_penetration = torch.maximum(
            max_penetration, direct.episode_max_penetration
        )
        max_force = torch.maximum(
            max_force, direct.contact_force_norm.max(dim=-1).values
        )
        max_pen_drift = torch.maximum(max_pen_drift, drift)
        max_axis_drift = torch.maximum(max_axis_drift, axis_drift)
        max_angular_speed = torch.maximum(max_angular_speed, angular_speed)

    success = (
        (success_step >= 0)
        & ~failed_ever
        & (max_penetration <= direct.cfg.p5_max_penetration_m)
        & (max_pen_drift <= direct.cfg.p5_max_position_drift_m)
    )
    score = (
        max_clear_run.float()
        + 6.0 * support_steps / args.steps
        - 60.0 * failed_ever.float()
        - 20.0 * torch.relu(max_penetration * 1000.0 - 0.5)
        - 2.0 * max_pen_drift * 1000.0
    )
    success_ids = torch.nonzero(success, as_tuple=False).squeeze(-1)
    selected_ids = (
        success_ids
        if success_ids.numel() > 0
        else torch.topk(score, k=min(512, args.num_envs)).indices
    )
    selected_cpu = selected_ids.cpu()
    observations = torch.stack(observation_trace, dim=1)[selected_cpu]
    teacher_actions = torch.stack(action_trace, dim=1)[selected_cpu]
    source_qpos = torch.cat(
        (
            success_hand_qpos[selected_ids],
            success_pen_pos[selected_ids],
            success_pen_quat[selected_ids],
        ),
        dim=-1,
    )
    source_qvel = torch.cat(
        (
            success_hand_qvel[selected_ids],
            success_pen_linvel[selected_ids],
            success_pen_angvel[selected_ids],
        ),
        dim=-1,
    )

    output = args.output
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        env_ids=selected_cpu.numpy(),
        bank_ids=direct.current_bank_ids[selected_ids].cpu().numpy(),
        observation=observations.numpy(),
        action=teacher_actions.numpy(),
        stage_id=np.full(
            (selected_ids.numel(), args.steps), 1, dtype=np.int64
        ),
        lengths=(success_step[selected_ids] + 1).cpu().numpy(),
        candidate_middle_target=candidate_middle[selected_ids].cpu().numpy(),
        success_step=success_step[selected_ids].cpu().numpy(),
        max_clear_hold=max_clear_run[selected_ids].cpu().numpy(),
        max_penetration=max_penetration[selected_ids].cpu().numpy(),
        max_force=max_force[selected_ids].cpu().numpy(),
        max_pen_drift=max_pen_drift[selected_ids].cpu().numpy(),
        max_axis_drift=max_axis_drift[selected_ids].cpu().numpy(),
        max_angular_speed=max_angular_speed[selected_ids].cpu().numpy(),
        terminal_hand_qpos=direct.hand_qpos[selected_ids].cpu().numpy(),
        terminal_hand_qvel=direct.hand_qvel[selected_ids].cpu().numpy(),
        terminal_pen_pos=direct.pen_pos[selected_ids].cpu().numpy(),
        terminal_pen_quat=direct.pen_quat[selected_ids].cpu().numpy(),
        terminal_pen_linvel=direct.pen_linvel[selected_ids].cpu().numpy(),
        terminal_pen_angvel=direct.pen_angvel[selected_ids].cpu().numpy(),
        source_qpos=source_qpos.cpu().numpy(),
        source_qvel=source_qvel.cpu().numpy(),
        source_target_qpos=success_policy_targets[selected_ids].cpu().numpy(),
        source_policy_action=success_action[selected_ids].cpu().numpy(),
        target_qpos=source_qpos.cpu().numpy(),
        target_qvel=np.zeros(
            (selected_ids.numel(), 26), dtype=np.float32
        ),
        source_contact_force_n=success_contact_force[selected_ids]
        .cpu()
        .numpy(),
        source_max_penetration_m=success_penetration[selected_ids]
        .cpu()
        .numpy(),
        phase_ids=np.full(selected_ids.numel(), 1, dtype=np.int64),
        source_teacher_actions=teacher_actions.numpy(),
    )
    summary = {
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "hold_steps": args.hold_steps,
        "seed": args.seed,
        "servo_gain": args.servo_gain,
        "parent": str(args.parent) if args.parent else None,
        "jitter": args.jitter,
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "clear_100ms_count": int(
            (max_clear_run >= args.hold_steps).sum().item()
        ),
        "failure_count": int(failed_ever.sum().item()),
        "selected_count": int(selected_ids.numel()),
        "selected_max_penetration_mm": float(
            max_penetration[selected_ids].max().item() * 1000.0
        ),
        "selected_max_pen_drift_mm": float(
            max_pen_drift[selected_ids].max().item() * 1000.0
        ),
        "selected_max_axis_drift_deg": float(
            torch.rad2deg(max_axis_drift[selected_ids]).max().item()
        ),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
