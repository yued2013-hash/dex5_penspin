from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate the canonical v9 strict P3 hold policy.")
parser.add_argument("--task", default="Dex5-V9P3-StrictHold-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--episodes", type=int, default=4096)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=703)
parser.add_argument("--zero_actions", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg["seed"] = args_cli.seed
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
    episode_success = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    recorded_success = torch.zeros_like(episode_success)
    episode_max_penetration = torch.zeros(args_cli.num_envs, device=device)
    episode_max_hold = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    completed = 0
    successes = 0
    valid_samples = 0
    contact_sum = torch.zeros(5, device=device)
    force_sum = torch.zeros(5, device=device)
    gate_sum = torch.zeros(7, device=device)
    success_snapshot_sum = torch.zeros(5, device=device)
    success_snapshot_max = torch.zeros(5, device=device)
    completed_penetration_sum = 0.0
    completed_penetration_max = 0.0

    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            if args_cli.zero_actions:
                actions = torch.zeros(
                    (args_cli.num_envs, direct_env.cfg.action_space), device=device
                )
            else:
                outputs = runner.agent.act(
                    observations, states, timestep=step, timesteps=args_cli.max_steps
                )
                actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, terminated, truncated, _ = wrapped_env.step(actions)
            states = wrapped_env.state()

        done = (terminated | truncated).reshape(-1)
        valid = ~done
        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > direct_env.cfg.strict_min_contact_force_n
        triad = contacts[:, :3].all(dim=-1)
        ring_pinky_clear = ~contacts[:, 3:].any(dim=-1)
        axis_dot = torch.sum(
            direct_env.pen_axis * direct_env.init_pen_axis, dim=-1
        ).clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot.abs())
        axis_drift_deg = axis_drift * (180.0 / torch.pi)
        angular_speed = torch.linalg.norm(direct_env.pen_angvel, dim=-1)
        position_drift_mm = (
            torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
            * 1000.0
        )
        peak_force = direct_env.contact_force_norm.max(dim=-1).values
        penetration_mm = direct_env.episode_max_penetration * 1000.0
        gates = torch.stack(
            (
                triad,
                ring_pinky_clear,
                axis_drift_deg <= direct_env.cfg.strict_axis_drift_deg,
                angular_speed <= direct_env.cfg.strict_max_angular_speed_rad_s,
                position_drift_mm <= direct_env.cfg.strict_max_position_drift_m * 1000.0,
                peak_force <= direct_env.cfg.strict_max_contact_force_n,
                penetration_mm <= direct_env.cfg.penetration_threshold * 1000.0,
            ),
            dim=-1,
        )
        success_now = direct_env.p3_hold_count >= direct_env.cfg.p3_hold_steps
        newly_successful = success_now & ~recorded_success
        snapshot = torch.stack(
            (axis_drift_deg, angular_speed, position_drift_mm, penetration_mm, peak_force),
            dim=-1,
        )
        if newly_successful.any():
            success_snapshot_sum += snapshot[newly_successful].sum(dim=0)
            success_snapshot_max = torch.maximum(
                success_snapshot_max, snapshot[newly_successful].max(dim=0).values
            )
        recorded_success |= success_now
        episode_success |= success_now
        episode_max_hold = torch.maximum(episode_max_hold, direct_env.p3_hold_count)
        episode_max_penetration = torch.maximum(
            episode_max_penetration, direct_env.episode_max_penetration
        )
        contact_sum += contacts[valid].float().sum(dim=0)
        force_sum += direct_env.contact_force_norm[valid].sum(dim=0)
        gate_sum += gates[valid].float().sum(dim=0)
        valid_samples += int(valid.sum().item())

        if done.any():
            count = int(done.sum().item())
            completed += count
            successes += int(episode_success[done].sum().item())
            done_penetration = episode_max_penetration[done] * 1000.0
            completed_penetration_sum += float(done_penetration.sum().item())
            completed_penetration_max = max(
                completed_penetration_max, float(done_penetration.max().item())
            )
            episode_success[done] = False
            recorded_success[done] = False
            episode_max_hold[done] = 0
            episode_max_penetration[done] = 0.0
        if completed >= args_cli.episodes:
            break

    finger_names = ("thumb", "index", "middle", "ring", "pinky")
    snapshot_names = (
        "axis_drift_deg",
        "angular_speed_rad_s",
        "position_drift_mm",
        "penetration_mm",
        "peak_force_n",
    )
    gate_names = (
        "triad_contact",
        "ring_pinky_clear",
        "axis_drift",
        "angular_speed",
        "position_drift",
        "peak_force",
        "penetration",
    )
    result = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "seed": args_cli.seed,
        "zero_actions": args_cli.zero_actions,
        "episodes": completed,
        "successes": successes,
        "success_rate": successes / max(completed, 1),
        "hold_requirement_ms": direct_env.cfg.p3_hold_steps
        * direct_env.cfg.decimation
        * direct_env.cfg.sim.dt
        * 1000.0,
        "contact_ratio": {
            name: float(contact_sum[i].item()) / max(valid_samples, 1)
            for i, name in enumerate(finger_names)
        },
        "mean_force_n": {
            name: float(force_sum[i].item()) / max(valid_samples, 1)
            for i, name in enumerate(finger_names)
        },
        "instantaneous_gate_ratio": {
            name: float(gate_sum[i].item()) / max(valid_samples, 1)
            for i, name in enumerate(gate_names)
        },
        "successful_hold_snapshot_mean": {
            name: float(success_snapshot_sum[i].item()) / max(successes, 1)
            for i, name in enumerate(snapshot_names)
        },
        "successful_hold_snapshot_max": {
            name: float(success_snapshot_max[i].item())
            for i, name in enumerate(snapshot_names)
        },
        "episode_max_penetration_mm": {
            "mean": completed_penetration_sum / max(completed, 1),
            "max": completed_penetration_max,
        },
    }
    output = Path(args_cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
