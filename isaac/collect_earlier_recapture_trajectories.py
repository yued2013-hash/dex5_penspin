from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect successful earlier recapture teacher trajectories.")
parser.add_argument("--task", default="Dex5-EarlierPreRingIndexContact-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--trajectories", type=int, default=50)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: F401, E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg["seed"]
    env = gym.make(args_cli.task, cfg=env_cfg)
    direct_env = env.unwrapped
    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped_env, agent_cfg)
    runner.agent.load(str(Path(args_cli.checkpoint).resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    observations, _ = wrapped_env.reset()
    states = wrapped_env.state()
    device = direct_env.device
    current: list[list[dict[str, np.ndarray]]] = [[] for _ in range(args_cli.num_envs)]
    successes: list[list[dict[str, np.ndarray]]] = []
    episode_index = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    episode_thumb = torch.zeros_like(episode_index)
    episode_near14 = torch.zeros_like(episode_index)
    episode_max_drift = torch.zeros(args_cli.num_envs, device=device)

    for step in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(observations, states, timestep=step, timesteps=args_cli.steps)
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, terminated, truncated, _ = wrapped_env.step(actions)
            states = wrapped_env.state()
        done = (terminated | truncated).reshape(-1)
        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > 0.03
        near14 = direct_env.tip_surface_dist[:, 1] <= 0.014
        drift = torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
        episode_index |= contacts[:, 1]
        episode_thumb |= contacts[:, 0]
        episode_near14 |= near14
        episode_max_drift = torch.maximum(episode_max_drift, drift)
        for env_id in range(args_cli.num_envs):
            current[env_id].append(
                {
                    "qpos": torch.cat((direct_env.hand_qpos[env_id], direct_env.pen_pos[env_id], direct_env.pen_quat[env_id])).cpu().numpy(),
                    "qvel": torch.cat((direct_env.hand_qvel[env_id], direct_env.pen_linvel[env_id], direct_env.pen_angvel[env_id])).cpu().numpy(),
                    "action": actions[env_id].cpu().numpy(),
                    "contact_force_n": direct_env.contact_force_norm[env_id].cpu().numpy(),
                    "tip_surface_dist_m": direct_env.tip_surface_dist[env_id].cpu().numpy(),
                }
            )
        done_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
        for env_id in done_ids:
            success = bool(
                episode_index[env_id]
                and episode_thumb[env_id]
                and episode_near14[env_id]
                and episode_max_drift[env_id] < 0.015
                and direct_env.episode_max_penetration[env_id] < 0.004
            )
            if success:
                successes.append(current[env_id])
            current[env_id] = []
            episode_index[env_id] = False
            episode_thumb[env_id] = False
            episode_near14[env_id] = False
            episode_max_drift[env_id] = 0.0
        if len(successes) >= args_cli.trajectories:
            break

    if len(successes) < args_cli.trajectories:
        raise RuntimeError(f"Collected only {len(successes)} successful trajectories")
    successes = successes[: args_cli.trajectories]
    max_len = max(len(row) for row in successes)
    qpos = np.zeros((len(successes), max_len, 27), dtype=np.float32)
    qvel = np.zeros((len(successes), max_len, 26), dtype=np.float32)
    actions = np.zeros((len(successes), max_len, 20), dtype=np.float32)
    force = np.zeros((len(successes), max_len, 5), dtype=np.float32)
    tip_dist = np.zeros((len(successes), max_len, 5), dtype=np.float32)
    lengths = np.zeros(len(successes), dtype=np.int64)
    for i, row in enumerate(successes):
        lengths[i] = len(row)
        qpos[i, : lengths[i]] = np.stack([x["qpos"] for x in row])
        qvel[i, : lengths[i]] = np.stack([x["qvel"] for x in row])
        actions[i, : lengths[i]] = np.stack([x["action"] for x in row])
        force[i, : lengths[i]] = np.stack([x["contact_force_n"] for x in row])
        tip_dist[i, : lengths[i]] = np.stack([x["tip_surface_dist_m"] for x in row])
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, qpos=qpos, qvel=qvel, actions=actions, contact_force_n=force, tip_surface_dist_m=tip_dist, lengths=lengths)
    metadata = {
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "trajectories": len(successes),
        "max_steps": int(max_len),
        "lengths": lengths.tolist(),
        "success_definition": "index/thumb contact, index <=14mm, pen drift <15mm, max penetration <4mm",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
