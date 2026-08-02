from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate P2-to-P3 policy with completed-episode metrics.")
parser.add_argument("--task", default="Dex5-P3Entry-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=4096)
parser.add_argument("--max_steps", type=int, default=1000)
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
    steps = 0
    contact_sum = torch.zeros(5, device=direct_env.device)
    valid_contact_steps = 0
    while int(direct_env.completed_p3_episodes.item()) < args_cli.episodes and steps < args_cli.max_steps:
        with torch.inference_mode():
            outputs = runner.agent.act(observations, states, timestep=steps, timesteps=args_cli.max_steps)
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, _, _, _ = wrapped_env.step(actions)
            states = wrapped_env.state()
            direct_env._compute_intermediate_values()
            contact_sum += (direct_env.contact_force_norm > 0.03).float().sum(dim=0)
            valid_contact_steps += direct_env.num_envs
        steps += 1

    completed = int(direct_env.completed_p3_episodes.item())
    successes = int(direct_env.completed_p3_successes.item())
    result = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "num_envs": args_cli.num_envs,
        "episodes": completed,
        "successes": successes,
        "success_rate": successes / max(completed, 1),
        "mean_episode_max_directional_deflection_deg": (
            float(direct_env.completed_p3_deflection_sum.item())
            / max(completed, 1)
            * 180.0
            / torch.pi
        ),
        "success_definition": (
            "directional planar deflection >=7deg; hand RMSE <0.20rad; pen drift <15mm; "
            "thumb/index/middle contact ratio >=2/3; proxy penetration <4mm; "
            "P3 simultaneous contact hold >=100ms"
        ),
        "simulation_steps": steps,
        "step_contact_ratio": (
            contact_sum / max(valid_contact_steps, 1)
        ).detach().cpu().tolist(),
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
