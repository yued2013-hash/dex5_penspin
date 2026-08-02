from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Collect successful target-teacher P4-to-P5 trajectories for BC."
)
parser.add_argument("--task", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--trajectories", type=int, default=1024)
parser.add_argument("--target_gain", type=float, default=0.01)
parser.add_argument(
    "--action_mode", choices=("target", "zero"), default="target"
)
parser.add_argument("--max_steps", type=int, default=12000)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    obs, _ = env.reset()
    horizon = direct.max_episode_length
    obs_dim = obs["policy"].shape[-1]
    obs_buffer = torch.zeros(
        (args.num_envs, horizon, obs_dim), device=direct.device
    )
    action_buffer = torch.zeros(
        (args.num_envs, horizon, direct.cfg.action_space), device=direct.device
    )
    success_seen = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    lengths: list[int] = []
    steps = 0

    while len(observations) < args.trajectories and steps < args.max_steps:
        indices = direct.episode_length_buf.long().clamp_max(horizon - 1)
        if args.action_mode == "zero":
            teacher_action = torch.zeros(
                (args.num_envs, direct.cfg.action_space), device=direct.device
            )
        else:
            mask = direct._stage_action_mask()
            increment = 0.25 * direct.cfg.action_scale * mask
            target_error = direct.target_hand_qpos - direct.policy_targets
            teacher_action = (
                args.target_gain * target_error / increment.clamp_min(1.0e-6)
            )
            teacher_action = torch.where(
                mask > 0.0, teacher_action, torch.zeros_like(teacher_action)
            ).clamp(-1.0, 1.0)
        rows = torch.arange(args.num_envs, device=direct.device)
        obs_buffer[rows, indices] = obs["policy"]
        action_buffer[rows, indices] = teacher_action

        obs, _, terminated, truncated, _ = env.step(teacher_action)
        success_seen |= direct.episode_p5_success
        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        for env_id in done_ids.tolist():
            if success_seen[env_id]:
                length = int(indices[env_id].item()) + 1
                observations.append(
                    obs_buffer[env_id, :length].detach().cpu().numpy().copy()
                )
                actions.append(
                    action_buffer[env_id, :length].detach().cpu().numpy().copy()
                )
                lengths.append(length)
                if len(observations) >= args.trajectories:
                    break
        if done_ids.numel() > 0:
            obs_buffer[done_ids] = 0.0
            action_buffer[done_ids] = 0.0
            success_seen[done_ids] = False
        steps += 1

    if len(observations) < args.trajectories:
        raise RuntimeError(
            f"Collected {len(observations)}/{args.trajectories} trajectories "
            f"after {steps} steps."
        )
    max_length = max(lengths)
    obs_array = np.zeros(
        (len(observations), max_length, obs_dim), dtype=np.float32
    )
    action_array = np.zeros(
        (len(actions), max_length, direct.cfg.action_space), dtype=np.float32
    )
    for row, (obs_row, action_row) in enumerate(zip(observations, actions)):
        obs_array[row, : len(obs_row)] = obs_row
        action_array[row, : len(action_row)] = action_row
    length_array = np.asarray(lengths, dtype=np.int64)
    stage_array = np.full(
        (len(observations), max_length),
        int(direct.cfg.curriculum_stage),
        dtype=np.int64,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observation=obs_array,
        action=action_array,
        stage_id=stage_array,
        lengths=length_array,
    )
    print(
        f"Saved {len(observations)} successful trajectories, "
        f"{int(length_array.sum())} transitions, after {steps} steps to {output}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
