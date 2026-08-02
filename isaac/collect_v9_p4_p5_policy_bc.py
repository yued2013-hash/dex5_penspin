from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Collect successful P4-to-P5 policy trajectories for BC."
)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--trajectories", type=int, default=1024)
parser.add_argument("--max_steps", type=int, default=12000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg["seed"] = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped, agent_cfg)
    checkpoint = Path(args.checkpoint).resolve()
    runner.agent.load(str(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    obs, _ = wrapped.reset()
    states = wrapped.state()
    horizon = direct.max_episode_length
    obs_dim = obs.shape[-1]
    obs_buffer = torch.zeros(
        (args.num_envs, horizon, obs_dim), device=direct.device
    )
    action_buffer = torch.zeros(
        (args.num_envs, horizon, direct.cfg.action_space), device=direct.device
    )
    stage_buffer = torch.zeros(
        (args.num_envs, horizon), dtype=torch.long, device=direct.device
    )
    success_seen = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stages: list[np.ndarray] = []
    lengths: list[int] = []
    steps = 0

    while len(observations) < args.trajectories and steps < args.max_steps:
        indices = direct.episode_length_buf.long().clamp_max(horizon - 1)
        with torch.inference_mode():
            outputs = runner.agent.act(
                obs, states, timestep=steps, timesteps=args.max_steps
            )
            action = outputs[-1].get("mean_actions", outputs[0])
        rows = torch.arange(args.num_envs, device=direct.device)
        obs_buffer[rows, indices] = obs
        action_buffer[rows, indices] = action
        stage_buffer[rows, indices] = direct.p5_stage_ids

        obs, _, terminated, truncated, _ = wrapped.step(action)
        states = wrapped.state()
        success_seen |= direct.episode_p5_success
        done = (terminated | truncated).reshape(-1)
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        for env_id in done_ids.tolist():
            if bool(success_seen[env_id]):
                length = int(indices[env_id].item()) + 1
                observations.append(
                    obs_buffer[env_id, :length].detach().cpu().numpy().copy()
                )
                actions.append(
                    action_buffer[env_id, :length].detach().cpu().numpy().copy()
                )
                stages.append(
                    stage_buffer[env_id, :length].detach().cpu().numpy().copy()
                )
                lengths.append(length)
                if len(observations) >= args.trajectories:
                    break
        if done_ids.numel() > 0:
            obs_buffer[done_ids] = 0.0
            action_buffer[done_ids] = 0.0
            stage_buffer[done_ids] = 0
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
    stage_array = np.zeros(
        (len(stages), max_length), dtype=np.int64
    )
    for row, (obs_row, action_row, stage_row) in enumerate(
        zip(observations, actions, stages, strict=True)
    ):
        obs_array[row, : len(obs_row)] = obs_row
        action_array[row, : len(action_row)] = action_row
        stage_array[row, : len(stage_row)] = stage_row
    length_array = np.asarray(lengths, dtype=np.int64)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observation=obs_array,
        action=action_array,
        stage_id=stage_array,
        lengths=length_array,
        task=np.asarray(args.task),
        checkpoint=np.asarray(str(checkpoint)),
        seed=np.asarray(args.seed, dtype=np.int64),
    )
    print(
        f"Saved {len(observations)} successful policy trajectories, "
        f"{int(length_array.sum())} transitions, after {steps} steps to {output}"
    )
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
