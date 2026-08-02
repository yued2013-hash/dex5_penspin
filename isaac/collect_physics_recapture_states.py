from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect successful physical recapture states.")
parser.add_argument("--task", default="Dex5-IndexRecapture-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--max_states", type=int, default=512)
parser.add_argument("--required_indices", type=int, nargs="+", default=(0, 1, 2))
parser.add_argument("--min_directional_deflection_deg", type=float, default=0.0)
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
    qpos_rows: list[np.ndarray] = []
    qvel_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    deflection_rows: list[np.ndarray] = []

    for step in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(observations, states, timestep=step, timesteps=args_cli.steps)
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, _, _, _ = wrapped_env.step(actions)
            states = wrapped_env.state()

        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > 0.03
        required_ratio = contacts[:, args_cli.required_indices].float().mean(dim=-1)
        pen_error = torch.linalg.norm(direct_env.pen_pos - direct_env.target_pen_pos, dim=-1)
        if args_cli.min_directional_deflection_deg > 0.0:
            planar_deflection = direct_env._signed_planar_deflection()
            target_direction = float(np.sign(direct_env.cfg.p3_planar_target_deg))
            directional_deflection = target_direction * planar_deflection
            deflection_ok = directional_deflection >= np.deg2rad(args_cli.min_directional_deflection_deg)
        else:
            deflection_ok = torch.ones_like(required_ratio, dtype=torch.bool)
        selected = (
            (required_ratio >= 2.0 / 3.0)
            & (direct_env.episode_max_penetration < 0.004)
            & (pen_error < 0.015)
            & (direct_env.episode_length_buf > 0)
            & deflection_ok
        )
        ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
        if ids.numel():
            qpos = torch.cat((direct_env.hand_qpos[ids], direct_env.pen_pos[ids], direct_env.pen_quat[ids]), dim=-1)
            qvel = torch.cat(
                (direct_env.hand_qvel[ids], direct_env.pen_linvel[ids], direct_env.pen_angvel[ids]), dim=-1
            )
            qpos_rows.append(qpos.cpu().numpy())
            qvel_rows.append(qvel.cpu().numpy())
            target_rows.append(direct_env.processed_targets[ids].cpu().numpy())
            if args_cli.min_directional_deflection_deg > 0.0:
                deflection_rows.append(directional_deflection[ids].cpu().numpy())
        if sum(row.shape[0] for row in qpos_rows) >= args_cli.max_states:
            break

    if not qpos_rows:
        raise RuntimeError("No physical recapture states passed the collection filters")
    qpos = np.concatenate(qpos_rows)[: args_cli.max_states]
    qvel = np.concatenate(qvel_rows)[: args_cli.max_states]
    target_qpos = np.concatenate(target_rows)[: args_cli.max_states]
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"qpos": qpos, "qvel": qvel, "target_qpos": target_qpos}
    if deflection_rows:
        payload["directional_deflection_rad"] = np.concatenate(deflection_rows)[: args_cli.max_states]
    np.savez_compressed(output, **payload)
    print(f"COLLECTED_RECAPTURE_STATES={qpos.shape[0]} OUTPUT={output}", flush=True)
    wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
