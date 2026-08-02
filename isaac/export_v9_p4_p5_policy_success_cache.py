from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Export successful P4-to-P5 curriculum states as a reset bundle."
)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint")
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--samples", type=int, default=512)
parser.add_argument("--max_steps", type=int, default=4000)
parser.add_argument(
    "--action_mode", choices=("policy", "target", "zero"), default="policy"
)
parser.add_argument("--target_gain", type=float, default=0.01)
parser.add_argument("--phase_id", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg["seed"]
    env = gym.make(args_cli.task, cfg=env_cfg)
    direct = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = None
    if args_cli.action_mode == "policy":
        if not args_cli.checkpoint:
            raise ValueError("--checkpoint is required for policy action mode")
        runner = Runner(wrapped, agent_cfg)
        runner.agent.load(str(Path(args_cli.checkpoint).resolve()))
        runner.agent.enable_training_mode(False, apply_to_models=True)

    fields: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "source_qpos",
            "source_qvel",
            "source_target_qpos",
            "source_policy_action",
            "target_qpos",
            "target_qvel",
            "source_contact_force_n",
            "source_angular_speed_rad_s",
            "source_max_penetration_m",
            "original_source_bank_ids",
        )
    }
    observations, _ = wrapped.reset()
    states = wrapped.state()
    steps = 0

    while len(fields["source_qpos"]) < args_cli.samples and steps < args_cli.max_steps:
        with torch.inference_mode():
            if args_cli.action_mode == "policy":
                assert runner is not None
                outputs = runner.agent.act(
                    observations, states, timestep=steps, timesteps=args_cli.max_steps
                )
                actions = outputs[-1].get("mean_actions", outputs[0])
            elif args_cli.action_mode == "target":
                mask = direct._stage_action_mask()
                increment = 0.25 * direct.cfg.action_scale * mask
                target_error = direct.target_hand_qpos - direct.policy_targets
                actions = (
                    args_cli.target_gain
                    * target_error
                    / increment.clamp_min(1.0e-6)
                )
                actions = torch.where(
                    mask > 0.0, actions, torch.zeros_like(actions)
                ).clamp(-1.0, 1.0)
            else:
                actions = torch.zeros(
                    (args_cli.num_envs, direct.cfg.action_space),
                    device=direct.device,
                )
            observations, _, _, _, _ = wrapped.step(actions)
            states = wrapped.state()
            direct._compute_intermediate_values()
            reached = direct.stage_hold_count == direct.cfg.p5_hold_steps
            ids = torch.nonzero(reached, as_tuple=False).squeeze(-1)
            if ids.numel() > 0:
                remaining = args_cli.samples - len(fields["source_qpos"])
                ids = ids[:remaining]
                source_qpos = torch.cat(
                    (direct.hand_qpos[ids], direct.pen_pos[ids], direct.pen_quat[ids]),
                    dim=-1,
                )
                source_qvel = torch.cat(
                    (
                        direct.hand_qvel[ids],
                        direct.pen_linvel[ids],
                        direct.pen_angvel[ids],
                    ),
                    dim=-1,
                )
                target_qpos = torch.cat(
                    (
                        direct.target_hand_qpos[ids],
                        direct.target_pen_pos[ids],
                        direct.target_pen_quat[ids],
                    ),
                    dim=-1,
                )
                batch = {
                    "source_qpos": source_qpos,
                    "source_qvel": source_qvel,
                    "source_target_qpos": direct.policy_targets[ids],
                    "source_policy_action": actions[ids],
                    "target_qpos": target_qpos,
                    "target_qvel": torch.zeros_like(source_qvel),
                    "source_contact_force_n": direct.contact_force_norm[ids],
                    "source_angular_speed_rad_s": torch.linalg.norm(
                        direct.pen_angvel[ids], dim=-1
                    ),
                    "source_max_penetration_m": direct.episode_max_penetration[ids],
                    "original_source_bank_ids": direct.current_bank_ids[ids],
                }
                for key, value in batch.items():
                    fields[key].extend(value.detach().cpu().numpy())
        steps += 1

    count = len(fields["source_qpos"])
    if count < args_cli.samples:
        raise RuntimeError(
            f"Only collected {count}/{args_cli.samples} successful states in "
            f"{steps} steps."
        )

    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()}
    arrays["original_source_bank_ids"] = arrays["original_source_bank_ids"].astype(
        np.int64
    )
    arrays["phase_ids"] = np.full(
        count, args_cli.phase_id, dtype=np.int64
    )
    arrays["source_phase_deg"] = np.zeros(count, dtype=np.float32)
    arrays["source_entry_labels"] = np.full(count, 1, dtype=np.int64)
    arrays["source_trajectory_ids"] = np.arange(count, dtype=np.int64)
    arrays["source_teacher_actions"] = np.zeros((count, 0, 20), dtype=np.float32)

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(
        f"Exported {count} successful states after {steps} steps to {output.resolve()}"
    )
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
