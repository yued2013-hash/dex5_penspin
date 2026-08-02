from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Search constant index actions for P3->P4 contact retention.")
parser.add_argument("--task", default="Dex5-CanonicalP3-Entry7ToP4Entry12-Direct-v0")
parser.add_argument("--output", required=True)
parser.add_argument("--candidates", type=int, default=512)
parser.add_argument("--repeats", type=int, default=8)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--seed", type=int, default=20260724)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    num_envs = args.candidates * args.repeats
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    direct_env = env.unwrapped
    env.reset()

    generator = torch.Generator(device=direct_env.device).manual_seed(args.seed)
    candidates = 2.0 * torch.rand(
        (args.candidates, 4), generator=generator, device=direct_env.device
    ) - 1.0
    corners = torch.tensor(
        [[a, b, c, d] for a in (-1.0, 1.0) for b in (-1.0, 1.0)
         for c in (-1.0, 1.0) for d in (-1.0, 1.0)],
        device=direct_env.device,
    )
    candidates[: len(corners)] = corners
    candidates[len(corners)] = 0.0
    actions = torch.zeros((num_envs, 20), device=direct_env.device)
    actions[:, :4] = candidates.repeat_interleave(args.repeats, dim=0)

    index_contact_steps = torch.zeros(num_envs, device=direct_env.device)
    triad_contact_steps = torch.zeros(num_envs, device=direct_env.device)
    current_triad_run = torch.zeros(num_envs, dtype=torch.long, device=direct_env.device)
    max_triad_run = torch.zeros_like(current_triad_run)
    max_hold = torch.zeros_like(current_triad_run)
    max_phase = torch.full((num_envs,), -180.0, device=direct_env.device)
    ring_force_sum = torch.zeros(num_envs, device=direct_env.device)
    max_penetration = torch.zeros(num_envs, device=direct_env.device)
    terminated_ever = torch.zeros(num_envs, dtype=torch.bool, device=direct_env.device)

    for _ in range(args.steps):
        _, _, terminated, _, _ = env.step(actions)
        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > 0.03
        triad = contacts[:, :3].all(dim=-1)
        index_contact_steps += contacts[:, 1].float()
        triad_contact_steps += triad.float()
        current_triad_run = torch.where(
            triad, current_triad_run + 1, torch.zeros_like(current_triad_run)
        )
        max_triad_run = torch.maximum(max_triad_run, current_triad_run)
        max_hold = torch.maximum(max_hold, direct_env.p4_hold_count)
        current_yaw = torch.atan2(direct_env.pen_axis[:, 1], direct_env.pen_axis[:, 0])
        target_quat = direct_env.target_pen_quat
        target_yaw = torch.atan2(
            2.0 * (target_quat[:, 1] * target_quat[:, 2] + target_quat[:, 0] * target_quat[:, 3]),
            1.0 - 2.0 * (target_quat[:, 2] ** 2 + target_quat[:, 3] ** 2),
        )
        remaining_error = -torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        phase = direct_env.cfg.target_total_phase_deg - remaining_error * (180.0 / torch.pi)
        max_phase = torch.maximum(max_phase, phase)
        ring_force_sum += direct_env.contact_force_norm[:, 3]
        max_penetration = torch.maximum(max_penetration, direct_env.episode_max_penetration)
        terminated_ever |= terminated

    shape = (args.candidates, args.repeats)
    index_ratio = index_contact_steps.reshape(shape).mean(dim=1) / args.steps
    triad_ratio = triad_contact_steps.reshape(shape).mean(dim=1) / args.steps
    hold3_ratio = (max_hold.reshape(shape) >= 3).float().mean(dim=1)
    hold6_ratio = (max_triad_run.reshape(shape) >= 6).float().mean(dim=1)
    mean_phase = max_phase.reshape(shape).mean(dim=1)
    mean_ring_force = ring_force_sum.reshape(shape).mean(dim=1) / args.steps
    mean_penetration = max_penetration.reshape(shape).mean(dim=1) * 1000.0
    terminated_ratio = terminated_ever.reshape(shape).float().mean(dim=1)
    phase_score = torch.exp(-0.15 * torch.abs(mean_phase - direct_env.cfg.target_total_phase_deg))
    score = (
        12.0 * hold3_ratio
        + 6.0 * hold6_ratio
        + 5.0 * triad_ratio
        + 3.0 * index_ratio
        + 2.0 * phase_score
        - 2.0 * torch.clamp(mean_ring_force / 0.10, 0.0, 2.0)
        - 8.0 * terminated_ratio
        - 2.0 * torch.relu(mean_penetration - 4.0)
    )
    top = torch.topk(score, k=min(32, args.candidates)).indices
    rows = []
    for candidate_id in top.tolist():
        rows.append(
            {
                "candidate_id": candidate_id,
                "index_action": candidates[candidate_id].cpu().tolist(),
                "hold3_ratio": float(hold3_ratio[candidate_id].item()),
                "hold6_triad_ratio": float(hold6_ratio[candidate_id].item()),
                "triad_contact_ratio": float(triad_ratio[candidate_id].item()),
                "index_contact_ratio": float(index_ratio[candidate_id].item()),
                "mean_max_phase_deg": float(mean_phase[candidate_id].item()),
                "mean_ring_force_n": float(mean_ring_force[candidate_id].item()),
                "mean_max_penetration_mm": float(mean_penetration[candidate_id].item()),
                "terminated_ratio": float(terminated_ratio[candidate_id].item()),
                "score": float(score[candidate_id].item()),
            }
        )
    result = {
        "task": args.task,
        "candidates": args.candidates,
        "repeats": args.repeats,
        "steps": args.steps,
        "top": rows,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
