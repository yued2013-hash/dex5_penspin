from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Search collision-free middle-finger approach teachers from P5."
)
parser.add_argument(
    "--task", default="Dex5-V9P5-P1-MiddleApproach-Direct-v0"
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=72)
parser.add_argument("--seed", type=int, default=2111)
parser.add_argument("--servo_gain", type=float, default=0.22)
parser.add_argument("--parent", type=Path)
parser.add_argument("--jitter", type=float, default=0.12)
parser.add_argument("--bundle", type=Path)
parser.add_argument(
    "--match-p4-collision-proxies",
    action="store_true",
)
parser.add_argument(
    "--goal", choices=("approach", "contact"), default="approach"
)
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
    if args.bundle is not None:
        bundle = args.bundle
        if not bundle.is_absolute():
            bundle = Path(__file__).resolve().parents[1] / bundle
        env_cfg.bundle_path = str(bundle)
    if args.match_p4_collision_proxies:
        env_cfg.collision_proxy_radius_scales = (
            0.99,
            1.07,
            1.07,
            0.84,
            0.80,
        )
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
    source_middle = direct.source_targets[:, 4:8]
    if args.parent is None and args.goal == "approach":
        broad = lower + torch.rand(
            lower.shape, generator=generator, device=direct.device
        ) * (upper - lower)
        local = source_middle + (
            2.0
            * torch.rand(
                source_middle.shape, generator=generator, device=direct.device
            )
            - 1.0
        ) * torch.tensor(
            [0.35, 0.75, 0.75, 0.60], device=direct.device
        )
        candidate_middle = torch.where(
            (
                torch.arange(args.num_envs, device=direct.device) % 3
                == 0
            ).unsqueeze(-1),
            broad,
            local,
        )
    elif args.parent is not None:
        parent_path = args.parent
        if not parent_path.is_absolute():
            parent_path = Path(__file__).resolve().parents[1] / parent_path
        with np.load(parent_path) as parent:
            parent_targets = torch.as_tensor(
                parent["candidate_middle_target"],
                device=direct.device,
                dtype=torch.float32,
            )
        parent_ids = torch.randint(
            len(parent_targets),
            (args.num_envs,),
            generator=generator,
            device=direct.device,
        )
        radius = args.jitter * (upper - lower)
        candidate_middle = parent_targets[parent_ids] + radius * torch.randn(
            source_middle.shape, generator=generator, device=direct.device
        )
    else:
        candidate_middle = source_middle + (
            2.0
            * torch.rand(
                source_middle.shape, generator=generator, device=direct.device
            )
            - 1.0
        ) * torch.tensor(
            [0.15, 0.30, 0.30, 0.25], device=direct.device
        )
    candidate_middle = torch.maximum(
        torch.minimum(candidate_middle, upper), lower
    )
    desired_targets = direct.source_targets.clone()
    desired_targets[:, 4:8] = candidate_middle

    failed_ever = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    near_run = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    max_near_run = torch.zeros_like(near_run)
    min_surface_dist = torch.full(
        (args.num_envs,), 1.0, device=direct.device
    )
    max_penetration = torch.zeros(args.num_envs, device=direct.device)
    max_force = torch.zeros(args.num_envs, device=direct.device)
    max_pen_drift = torch.zeros(args.num_envs, device=direct.device)
    max_axis_drift = torch.zeros(args.num_envs, device=direct.device)
    support_steps = torch.zeros(args.num_envs, device=direct.device)
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
        for _physics_step in range(direct.cfg.decimation):
            direct._apply_action()
            direct.scene.write_data_to_sim()
            direct.sim.step(render=False)
            direct.scene.update(dt=direct.physics_dt)
        direct.episode_length_buf += 1
        direct._compute_intermediate_values()
        failed, _ = direct._compute_dones()
        failed_ever |= failed

        distance = (
            direct.tip_surface_dist[:, 2]
            - direct.fingertip_radii[:, 2]
        )
        contacts = (
            direct.contact_force_norm > direct.cfg.p1_contact_force_n
        )
        support = contacts[:, 0] & contacts[:, 1]
        drift = torch.linalg.norm(
            direct.pen_pos - direct.init_pen_pos, dim=-1
        )
        axis_dot = torch.sum(
            direct.pen_axis * direct.init_pen_axis, dim=-1
        ).clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot.abs())
        angular_speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        approach_eligible = (
            (distance >= direct.cfg.p1_middle_approach_min_gap_m)
            & (distance <= direct.cfg.p1_middle_near_distance_m)
        )
        contact_eligible = (
            (distance <= direct.cfg.p1_middle_contact_distance_m)
            & contacts[:, 2]
        )
        goal_geometry = (
            approach_eligible
            if args.goal == "approach"
            else contact_eligible
        )
        collision_free_near = (
            goal_geometry
            & support
            & (drift <= direct.cfg.p5_max_position_drift_m)
            & (
                axis_drift
                <= torch.deg2rad(
                    torch.full_like(
                        axis_drift, direct.cfg.p5_max_axis_drift_deg
                    )
                )
            )
            & (
                angular_speed
                <= direct.cfg.p5_max_angular_speed_rad_s
            )
            & (
                direct.contact_force_norm.max(dim=-1).values
                <= direct.cfg.p1_max_force_n
            )
            & ~failed_ever
            & (
                direct.episode_max_penetration
                <= direct.cfg.p5_max_penetration_m
            )
        )
        near_run = torch.where(
            collision_free_near,
            near_run + 1,
            torch.zeros_like(near_run),
        )
        max_near_run = torch.maximum(max_near_run, near_run)
        newly_successful = (near_run >= 12) & (success_step < 0)
        success_step[newly_successful] = step
        success_hand_qpos[newly_successful] = direct.hand_qpos[
            newly_successful
        ]
        success_hand_qvel[newly_successful] = direct.hand_qvel[
            newly_successful
        ]
        success_pen_pos[newly_successful] = direct.pen_pos[
            newly_successful
        ]
        success_pen_quat[newly_successful] = direct.pen_quat[
            newly_successful
        ]
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
        success_penetration[newly_successful] = direct.episode_max_penetration[
            newly_successful
        ]
        min_surface_dist = torch.minimum(min_surface_dist, distance)
        max_penetration = torch.maximum(
            max_penetration, direct.episode_max_penetration
        )
        max_force = torch.maximum(
            max_force, direct.contact_force_norm.max(dim=-1).values
        )
        max_pen_drift = torch.maximum(max_pen_drift, drift)
        max_axis_drift = torch.maximum(max_axis_drift, axis_drift)
        support_steps += support.float()

    success = (
        (max_near_run >= 12)
        & ~failed_ever
        & (max_penetration <= direct.cfg.p5_max_penetration_m)
        & (max_force <= direct.cfg.p1_max_force_n)
    )
    score = (
        max_near_run.float()
        + 4.0 * support_steps / args.steps
        - 60.0 * failed_ever.float()
        - 12.0
        * torch.relu(
            (min_surface_dist - direct.cfg.p1_middle_near_distance_m)
            * 1000.0
        )
        - 20.0 * torch.relu(max_penetration * 1000.0 - 0.75)
    )
    success_ids = torch.nonzero(success, as_tuple=False).squeeze(-1)
    if success_ids.numel() > 0:
        selected_ids = success_ids
    else:
        selected_ids = torch.topk(
            score, k=min(512, args.num_envs)
        ).indices
    selected_cpu = selected_ids.cpu()
    observations = torch.stack(observation_trace, dim=1)[selected_cpu]
    teacher_actions = torch.stack(action_trace, dim=1)[selected_cpu]

    output = args.output
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)
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
    np.savez_compressed(
        output,
        env_ids=selected_ids.cpu().numpy(),
        bank_ids=direct.current_bank_ids[selected_ids].cpu().numpy(),
        observation=observations.numpy(),
        action=teacher_actions.numpy(),
        stage_id=np.full(
            (selected_ids.numel(), args.steps),
            1 if args.goal == "approach" else 2,
            dtype=np.int64,
        ),
        lengths=np.full(
            selected_ids.numel(), args.steps, dtype=np.int64
        ),
        candidate_middle_target=candidate_middle[selected_ids]
        .cpu()
        .numpy(),
        terminal_hand_qpos=direct.hand_qpos[selected_ids].cpu().numpy(),
        terminal_hand_qvel=direct.hand_qvel[selected_ids].cpu().numpy(),
        terminal_pen_pos=direct.pen_pos[selected_ids].cpu().numpy(),
        terminal_pen_quat=direct.pen_quat[selected_ids].cpu().numpy(),
        terminal_pen_linvel=direct.pen_linvel[selected_ids].cpu().numpy(),
        terminal_pen_angvel=direct.pen_angvel[selected_ids].cpu().numpy(),
        max_near_hold=max_near_run[selected_ids].cpu().numpy(),
        min_surface_dist=min_surface_dist[selected_ids].cpu().numpy(),
        max_penetration=max_penetration[selected_ids].cpu().numpy(),
        max_force=max_force[selected_ids].cpu().numpy(),
        success_step=success_step[selected_ids].cpu().numpy(),
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
        source_angular_speed_rad_s=torch.linalg.norm(
            success_pen_angvel[selected_ids], dim=-1
        )
        .cpu()
        .numpy(),
        source_max_penetration_m=success_penetration[selected_ids]
        .cpu()
        .numpy(),
        original_source_bank_ids=direct.current_bank_ids[selected_ids]
        .cpu()
        .numpy(),
        phase_ids=np.full(
            selected_ids.numel(),
            2 if args.goal == "approach" else 3,
            dtype=np.int64,
        ),
        source_teacher_actions=teacher_actions.numpy(),
    )
    summary = {
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "servo_gain": args.servo_gain,
        "goal": args.goal,
        "parent": str(args.parent) if args.parent else None,
        "jitter": args.jitter,
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "near_100ms_count": int((max_near_run >= 12).sum().item()),
        "failure_count": int(failed_ever.sum().item()),
        "selected_count": int(selected_ids.numel()),
        "selected_min_surface_dist_mm": float(
            min_surface_dist[selected_ids].min().item() * 1000.0
        ),
        "selected_max_penetration_mm": float(
            max_penetration[selected_ids].max().item() * 1000.0
        ),
        "selected_max_force_n": float(
            max_force[selected_ids].max().item()
        ),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
