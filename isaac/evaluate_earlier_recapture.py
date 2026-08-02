from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate the earlier pre-ring index/thumb recapture.")
parser.add_argument("--task", default="Dex5-EarlierPreRingIndexContact-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=2048)
parser.add_argument("--max_steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
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
    checkpoint = Path(args_cli.checkpoint).resolve()
    runner.agent.load(str(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    observations, _ = wrapped_env.reset()
    states = wrapped_env.state()
    device = direct_env.device
    episode_index_contact = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    episode_thumb_contact = torch.zeros_like(episode_index_contact)
    episode_near14 = torch.zeros_like(episode_index_contact)
    episode_max_drift = torch.zeros(args_cli.num_envs, device=device)
    completed = successes = 0
    near14_sum = index_contact_sum = thumb_contact_sum = drift_sum = 0.0
    valid_samples = 0

    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            outputs = runner.agent.act(
                observations, states, timestep=step, timesteps=args_cli.max_steps
            )
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, terminated, truncated, _ = wrapped_env.step(actions)
            states = wrapped_env.state()

        done = (terminated | truncated).reshape(-1)
        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > 0.03
        index_contact = contacts[:, 1]
        thumb_contact = contacts[:, 0]
        near14 = direct_env.tip_surface_dist[:, 1] <= 0.014
        drift = torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
        episode_index_contact |= index_contact
        episode_thumb_contact |= thumb_contact
        episode_near14 |= near14
        episode_max_drift = torch.maximum(episode_max_drift, drift)
        valid = ~done
        near14_sum += float(near14[valid].sum().item())
        index_contact_sum += float(index_contact[valid].sum().item())
        thumb_contact_sum += float(thumb_contact[valid].sum().item())
        drift_sum += float(drift[valid].sum().item())
        valid_samples += int(valid.sum().item())

        if done.any():
            completed_now = int(done.sum().item())
            completed += completed_now
            episode_success = episode_index_contact & episode_thumb_contact & episode_near14 & (
                episode_max_drift < 0.015
            )
            successes += int(episode_success[done].sum().item())
            episode_index_contact[done] = False
            episode_thumb_contact[done] = False
            episode_near14[done] = False
            episode_max_drift[done] = 0.0
        if completed >= args_cli.episodes:
            break

    denominator = max(valid_samples, 1)
    result = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "episodes": completed,
        "successes": successes,
        "success_rate": successes / max(completed, 1),
        "success_definition": "index and thumb contact at least once, index tip <=14mm at least once, pen drift <15mm",
        "near14_ratio": near14_sum / denominator,
        "index_contact_ratio": index_contact_sum / denominator,
        "thumb_contact_ratio": thumb_contact_sum / denominator,
        "mean_pen_drift_mm": drift_sum / denominator * 1000.0,
        "valid_step_samples": valid_samples,
    }
    output = Path(args_cli.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
