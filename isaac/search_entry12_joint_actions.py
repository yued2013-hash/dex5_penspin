from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Search index-retention plus middle/thumb drive actions for the 7-to-12 degree primitive."
)
parser.add_argument("--task", default="Dex5-CanonicalP3-Entry7ToP4Entry12-Direct-v0")
parser.add_argument("--output", required=True)
parser.add_argument("--candidates", type=int, default=1024)
parser.add_argument("--repeats", type=int, default=4)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--seed", type=int, default=20260725)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


INDEX_ACTIONS = (
    (-0.67732245, 0.20928061, 0.82401538, -0.91803694),
    (-0.04472941, 0.37338376, 0.99040508, 0.55671883),
    (1.0, 1.0, 1.0, 1.0),
    (0.84941292, 0.99376798, 0.90057278, 0.93595898),
)


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    num_envs = args.candidates * args.repeats
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    direct_env = env.unwrapped
    env.reset()

    generator = torch.Generator(device=direct_env.device).manual_seed(args.seed)
    drive_actions = 2.0 * torch.rand(
        (args.candidates, 8), generator=generator, device=direct_env.device
    ) - 1.0
    # Keep zero and saturated directional probes in the search population.
    drive_actions[0] = 0.0
    for row in range(1, min(17, args.candidates)):
        bits = row - 1
        drive_actions[row, :4] = torch.tensor(
            [1.0 if bits & (1 << axis) else -1.0 for axis in range(4)],
            device=direct_env.device,
        )

    index_bank = torch.tensor(INDEX_ACTIONS, device=direct_env.device)
    index_ids = torch.arange(args.candidates, device=direct_env.device) % len(INDEX_ACTIONS)
    candidate_actions = torch.zeros((args.candidates, 20), device=direct_env.device)
    candidate_actions[:, :4] = index_bank[index_ids]
    candidate_actions[:, 4:8] = drive_actions[:, :4]
    candidate_actions[:, 16:20] = drive_actions[:, 4:8]
    actions = candidate_actions.repeat_interleave(args.repeats, dim=0)

    index_contact_steps = torch.zeros(num_envs, device=direct_env.device)
    triad_contact_steps = torch.zeros(num_envs, device=direct_env.device)
    current_triad_run = torch.zeros(num_envs, dtype=torch.long, device=direct_env.device)
    max_triad_run = torch.zeros_like(current_triad_run)
    max_hold = torch.zeros_like(current_triad_run)
    max_phase = torch.full((num_envs,), -180.0, device=direct_env.device)
    ring_force_sum = torch.zeros(num_envs, device=direct_env.device)
    max_angular_speed = torch.zeros(num_envs, device=direct_env.device)
    max_penetration = torch.zeros(num_envs, device=direct_env.device)
    max_drift = torch.zeros(num_envs, device=direct_env.device)
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
        target_axis = direct_env.target_pen_quat
        target_yaw = torch.atan2(
            2.0 * (target_axis[:, 1] * target_axis[:, 2] + target_axis[:, 0] * target_axis[:, 3]),
            1.0 - 2.0 * (target_axis[:, 2] ** 2 + target_axis[:, 3] ** 2),
        )
        remaining_error = -torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        phase = direct_env.cfg.target_total_phase_deg - remaining_error * (180.0 / torch.pi)
        max_phase = torch.maximum(max_phase, phase)
        ring_force_sum += direct_env.contact_force_norm[:, 3]
        max_angular_speed = torch.maximum(
            max_angular_speed, torch.linalg.norm(direct_env.pen_angvel, dim=-1)
        )
        max_penetration = torch.maximum(max_penetration, direct_env.episode_max_penetration)
        max_drift = torch.maximum(
            max_drift, torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
        )
        terminated_ever |= terminated

    shape = (args.candidates, args.repeats)
    index_ratio = index_contact_steps.reshape(shape).mean(dim=1) / args.steps
    triad_ratio = triad_contact_steps.reshape(shape).mean(dim=1) / args.steps
    hold3_ratio = (max_hold.reshape(shape) >= 3).float().mean(dim=1)
    hold6_ratio = (max_triad_run.reshape(shape) >= 6).float().mean(dim=1)
    mean_phase = max_phase.reshape(shape).mean(dim=1)
    mean_ring_force = ring_force_sum.reshape(shape).mean(dim=1) / args.steps
    mean_max_speed = max_angular_speed.reshape(shape).mean(dim=1)
    mean_penetration = max_penetration.reshape(shape).mean(dim=1) * 1000.0
    mean_drift = max_drift.reshape(shape).mean(dim=1) * 1000.0
    terminated_ratio = terminated_ever.reshape(shape).float().mean(dim=1)
    phase_score = torch.exp(-0.25 * torch.abs(mean_phase - direct_env.cfg.target_total_phase_deg))
    score = (
        20.0 * hold3_ratio
        + 5.0 * hold6_ratio
        + 5.0 * triad_ratio
        + 3.0 * index_ratio
        + 5.0 * phase_score
        - 2.0 * torch.clamp(mean_ring_force / 0.10, 0.0, 2.0)
        - 8.0 * terminated_ratio
        - 2.0 * torch.relu(mean_penetration - 4.0)
        - 0.25 * torch.relu(mean_drift - 15.0)
    )
    top = torch.topk(score, k=min(48, args.candidates)).indices
    rows = []
    for candidate_id in top.tolist():
        rows.append(
            {
                "candidate_id": candidate_id,
                "index_action_id": int(index_ids[candidate_id].item()),
                "action": candidate_actions[candidate_id].cpu().tolist(),
                "hold3_ratio": float(hold3_ratio[candidate_id].item()),
                "hold6_triad_ratio": float(hold6_ratio[candidate_id].item()),
                "triad_contact_ratio": float(triad_ratio[candidate_id].item()),
                "index_contact_ratio": float(index_ratio[candidate_id].item()),
                "mean_max_phase_deg": float(mean_phase[candidate_id].item()),
                "mean_max_angular_speed_rad_s": float(mean_max_speed[candidate_id].item()),
                "mean_ring_force_n": float(mean_ring_force[candidate_id].item()),
                "mean_max_penetration_mm": float(mean_penetration[candidate_id].item()),
                "mean_max_drift_mm": float(mean_drift[candidate_id].item()),
                "terminated_ratio": float(terminated_ratio[candidate_id].item()),
                "score": float(score[candidate_id].item()),
            }
        )
    result = {
        "task": args.task,
        "candidates": args.candidates,
        "repeats": args.repeats,
        "steps": args.steps,
        "action_layout": "index[0:4], middle[4:8], ring/pinky zero, thumb[16:20]",
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
