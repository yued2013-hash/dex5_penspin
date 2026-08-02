from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Search robust four-joint thumb recapture actions.")
parser.add_argument("--task", default="Dex5-P4Midway-Direct-v0")
parser.add_argument("--output", required=True)
parser.add_argument("--candidates", type=int, default=512)
parser.add_argument("--repeats", type=int, default=8)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--teacher_steps", type=int, default=12)
parser.add_argument("--search_middle", action="store_true")
parser.add_argument("--local_thumb_center", type=float, nargs=4)
parser.add_argument("--local_radius", type=float, default=0.30)
parser.add_argument(
    "--recapture_steps",
    type=int,
    default=-1,
    help="Apply the candidate impulse for this many steps, then let the teacher act.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    num_envs = args_cli.candidates * args_cli.repeats
    env_cfg.scene.num_envs = num_envs
    env_cfg.teacher_prefill_steps = min(args_cli.steps, args_cli.teacher_steps)
    env_cfg.teacher_preserve_thumb_actions = args_cli.recapture_steps < 0
    env_cfg.teacher_preserve_middle_actions = args_cli.search_middle and args_cli.recapture_steps < 0
    env_cfg.thumb_recapture_steps = 0
    env_cfg.policy_target_blend_steps = 0
    env_cfg.reset_joint_pos_noise = 0.001
    env_cfg.reset_pen_pos_noise = 0.0001
    env = gym.make(args_cli.task, cfg=env_cfg)
    direct_env = env.unwrapped
    env.reset()

    generator = torch.Generator(device=direct_env.device).manual_seed(20260714)
    candidates = 2.0 * torch.rand(
        (args_cli.candidates, 4), generator=generator, device=direct_env.device
    ) - 1.0
    corners = torch.tensor(
        [[a, b, c, d] for a in (-1.0, 1.0) for b in (-1.0, 1.0)
         for c in (-1.0, 1.0) for d in (-1.0, 1.0)],
        device=direct_env.device,
    )
    candidates[: len(corners)] = corners
    if args_cli.local_thumb_center is not None:
        center = torch.tensor(args_cli.local_thumb_center, device=direct_env.device)
        local_count = args_cli.candidates // 2
        noise = 2.0 * torch.rand(
            (local_count, 4), generator=generator, device=direct_env.device
        ) - 1.0
        candidates[-local_count:] = torch.clamp(
            center + args_cli.local_radius * noise, -1.0, 1.0
        )
        candidates[-1] = center
    actions = torch.zeros((num_envs, 20), device=direct_env.device)
    actions[:, -4:] = candidates.repeat_interleave(args_cli.repeats, dim=0)
    middle_candidates = None
    if args_cli.search_middle:
        middle_candidates = 2.0 * torch.rand(
            (args_cli.candidates, 4), generator=generator, device=direct_env.device
        ) - 1.0
        actions[:, 4:8] = middle_candidates.repeat_interleave(args_cli.repeats, dim=0)

    min_distance = torch.full((num_envs,), 1.0, device=direct_env.device)
    contact_steps = torch.zeros(num_envs, device=direct_env.device)
    tim_contact_steps = torch.zeros(num_envs, device=direct_env.device)
    support_contact_steps = torch.zeros(num_envs, device=direct_env.device)
    max_force = torch.zeros(num_envs, device=direct_env.device)
    current_tim_run = torch.zeros(num_envs, dtype=torch.long, device=direct_env.device)
    max_tim_run = torch.zeros(num_envs, dtype=torch.long, device=direct_env.device)
    for _ in range(args_cli.steps):
        step_index = _
        if args_cli.recapture_steps >= 0 and step_index >= args_cli.recapture_steps:
            step_actions = torch.zeros_like(actions)
        else:
            step_actions = actions
        env.step(step_actions)
        direct_env._compute_intermediate_values()
        min_distance = torch.minimum(min_distance, direct_env.tip_surface_dist[:, 0])
        thumb_force = direct_env.contact_force_norm[:, 0]
        contact_steps += (thumb_force > 0.03).float()
        contacts = direct_env.contact_force_norm[:, :3] > 0.03
        tim_contact = contacts.all(dim=-1)
        tim_contact_steps += tim_contact.float()
        current_tim_run = torch.where(
            tim_contact, current_tim_run + 1, torch.zeros_like(current_tim_run)
        )
        max_tim_run = torch.maximum(max_tim_run, current_tim_run)
        support_contact_steps += contacts[:, 1:].all(dim=-1).float()
        max_force = torch.maximum(max_force, thumb_force)

    shape = (args_cli.candidates, args_cli.repeats)
    mean_min_distance = min_distance.reshape(shape).mean(dim=1)
    contact_ratio = contact_steps.reshape(shape).mean(dim=1) / args_cli.steps
    tim_contact_ratio = tim_contact_steps.reshape(shape).mean(dim=1) / args_cli.steps
    support_contact_ratio = support_contact_steps.reshape(shape).mean(dim=1) / args_cli.steps
    mean_max_force = max_force.reshape(shape).mean(dim=1)
    hold3_ratio = (max_tim_run.reshape(shape) >= 3).float().mean(dim=1)
    hold6_ratio = (max_tim_run.reshape(shape) >= 6).float().mean(dim=1)
    mean_max_tim_run = max_tim_run.reshape(shape).float().mean(dim=1)
    score = 8.0 * hold6_ratio + 4.0 * hold3_ratio + 3.0 * tim_contact_ratio \
        + support_contact_ratio + contact_ratio \
        - 4.0 * torch.clamp(mean_min_distance, min=0.0)
    top = torch.topk(score, k=min(32, args_cli.candidates)).indices
    rows = []
    for index in top.tolist():
        rows.append(
            {
                "candidate": index,
                "thumb_action": candidates[index].cpu().tolist(),
                "middle_action": (
                    middle_candidates[index].cpu().tolist() if middle_candidates is not None else None
                ),
                "contact_ratio": float(contact_ratio[index].item()),
                "tim_contact_ratio": float(tim_contact_ratio[index].item()),
                "hold3_ratio": float(hold3_ratio[index].item()),
                "hold6_ratio": float(hold6_ratio[index].item()),
                "mean_max_tim_run": float(mean_max_tim_run[index].item()),
                "index_middle_contact_ratio": float(support_contact_ratio[index].item()),
                "mean_min_surface_dist_mm": float(mean_min_distance[index].item() * 1000.0),
                "mean_max_force_n": float(mean_max_force[index].item()),
            }
        )
    result = {
        "task": args_cli.task,
        "candidates": args_cli.candidates,
        "repeats": args_cli.repeats,
        "steps": args_cli.steps,
        "teacher_steps": args_cli.teacher_steps,
        "recapture_steps": args_cli.recapture_steps,
        "top": rows,
    }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
