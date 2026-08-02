from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Search short P4 middle-clear teachers.")
parser.add_argument("--task", default="Dex5-V9P4-P5-MiddleClearPinch-Direct-v0")
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=72)
parser.add_argument("--pinch_steps", type=int, default=18)
parser.add_argument("--seed", type=int, default=1301)
parser.add_argument("--parent", default=None)
parser.add_argument("--jitter", type=float, default=0.12)
parser.add_argument("--fixed_parent_index", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    env.reset()

    generator = torch.Generator(device=direct.device).manual_seed(args.seed)
    if args.parent:
        parent_path = Path(args.parent)
        if not parent_path.is_absolute():
            parent_path = Path(__file__).resolve().parents[1] / parent_path
        parent = np.load(parent_path)
        if args.fixed_parent_index is None:
            parent_ids = torch.randint(
                0,
                len(parent["pinch_actions"]),
                (args.num_envs,),
                generator=generator,
                device=direct.device,
            )
        else:
            parent_ids = torch.full(
                (args.num_envs,),
                args.fixed_parent_index,
                dtype=torch.long,
                device=direct.device,
            )
        parent_pinch = torch.as_tensor(
            parent["pinch_actions"], device=direct.device
        )[parent_ids]
        parent_clear = torch.as_tensor(
            parent["clear_actions"], device=direct.device
        )[parent_ids]
        pinch_actions = torch.clamp(
            parent_pinch
            + args.jitter
            * torch.randn(
                parent_pinch.shape, generator=generator, device=direct.device
            ),
            -1.0,
            1.0,
        )
        clear_actions = torch.clamp(
            parent_clear
            + args.jitter
            * torch.randn(
                parent_clear.shape, generator=generator, device=direct.device
            ),
            -1.0,
            1.0,
        )
    else:
        pinch_actions = torch.zeros((args.num_envs, 20), device=direct.device)
        pinch_actions[:, :4] = 0.65 * torch.randn(
            (args.num_envs, 4), generator=generator, device=direct.device
        )
        pinch_actions[:, 16:20] = 0.65 * torch.randn(
            (args.num_envs, 4), generator=generator, device=direct.device
        )
        pinch_actions.clamp_(-1.0, 1.0)

        clear_actions = pinch_actions.clone()
        middle_direction = torch.sign(
            direct.target_hand_qpos[:, 4:8] - direct.source_targets[:, 4:8]
        )
        middle_amplitude = 0.35 + 0.65 * torch.rand(
            (args.num_envs, 1), generator=generator, device=direct.device
        )
        middle_jitter = 0.20 * torch.randn(
            (args.num_envs, 4), generator=generator, device=direct.device
        )
        clear_actions[:, 4:8] = torch.clamp(
            middle_direction * middle_amplitude + middle_jitter, -1.0, 1.0
        )

    failed_ever = torch.zeros(args.num_envs, dtype=torch.bool, device=direct.device)
    clear_hold = torch.zeros(args.num_envs, dtype=torch.long, device=direct.device)
    max_clear_hold = torch.zeros_like(clear_hold)
    max_penetration = torch.zeros(args.num_envs, device=direct.device)
    min_thumb_force = torch.full(
        (args.num_envs,), float("inf"), device=direct.device
    )
    min_index_force = torch.full_like(min_thumb_force, float("inf"))
    observation_trace: list[torch.Tensor] = []
    action_trace: list[torch.Tensor] = []

    for step in range(args.steps):
        actions = pinch_actions if step < args.pinch_steps else clear_actions
        observation_trace.append(
            direct._get_observations()["policy"].detach().cpu()
        )
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
        max_penetration = torch.maximum(
            max_penetration, direct.episode_max_penetration
        )
        min_thumb_force = torch.minimum(
            min_thumb_force, direct.contact_force_norm[:, 0]
        )
        min_index_force = torch.minimum(
            min_index_force, direct.contact_force_norm[:, 1]
        )
        eligible = (
            (direct.contact_force_norm[:, 0] > direct.cfg.p5_contact_force_n)
            & (direct.contact_force_norm[:, 1] > direct.cfg.p5_contact_force_n)
            & (direct.contact_force_norm[:, 2] < direct.cfg.p5_clear_force_n)
            & ~failed_ever
        )
        clear_hold = torch.where(
            eligible, clear_hold + 1, torch.zeros_like(clear_hold)
        )
        max_clear_hold = torch.maximum(max_clear_hold, clear_hold)

    score = (
        max_clear_hold.float()
        - 100.0 * failed_ever.float()
        - 20.0 * torch.relu(max_penetration * 1000.0 - 0.5)
    )
    top_count = min(512, args.num_envs)
    top_ids = torch.topk(score, k=top_count).indices
    success = (
        (max_clear_hold >= 12)
        & ~failed_ever
        & (max_penetration < 0.001)
    )
    success_ids = torch.nonzero(success, as_tuple=False).squeeze(-1)
    selected_ids = success_ids if success_ids.numel() > 0 else top_ids
    selected_ids_cpu = selected_ids.cpu()
    observations = torch.stack(observation_trace, dim=1)[selected_ids_cpu]
    teacher_actions = torch.stack(action_trace, dim=1)[selected_ids_cpu]
    stage_ids = torch.full(
        (selected_ids.numel(), args.steps), 1, dtype=torch.int64
    )

    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        env_ids=selected_ids.cpu().numpy(),
        bank_ids=direct.current_bank_ids[selected_ids].cpu().numpy(),
        observation=observations.numpy(),
        action=teacher_actions.numpy(),
        stage_id=stage_ids.numpy(),
        lengths=np.full(selected_ids.numel(), args.steps, dtype=np.int64),
        pinch_actions=pinch_actions[selected_ids].cpu().numpy(),
        clear_actions=clear_actions[selected_ids].cpu().numpy(),
        terminal_hand_qpos=direct.hand_qpos[selected_ids].cpu().numpy(),
        terminal_hand_qvel=direct.hand_qvel[selected_ids].cpu().numpy(),
        terminal_pen_pos=direct.pen_pos[selected_ids].cpu().numpy(),
        terminal_pen_quat=direct.pen_quat[selected_ids].cpu().numpy(),
        terminal_pen_linvel=direct.pen_linvel[selected_ids].cpu().numpy(),
        terminal_pen_angvel=direct.pen_angvel[selected_ids].cpu().numpy(),
        max_clear_hold=max_clear_hold[selected_ids].cpu().numpy(),
        max_penetration=max_penetration[selected_ids].cpu().numpy(),
    )
    summary = {
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "pinch_steps": args.pinch_steps,
        "parent": args.parent,
        "jitter": args.jitter,
        "fixed_parent_index": args.fixed_parent_index,
        "success_count": int(success.sum().item()),
        "clear_100ms_count": int((max_clear_hold >= 12).sum().item()),
        "failure_count": int(failed_ever.sum().item()),
        "max_clear_hold_steps": int(max_clear_hold.max().item()),
        "selected_count": int(selected_ids.numel()),
        "selected_max_penetration_mm": float(
            max_penetration[selected_ids].max().item() * 1000.0
        ),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
