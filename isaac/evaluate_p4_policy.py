from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a P3-to-P4 curriculum policy.")
parser.add_argument("--task", default="Dex5-P4Approach-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=4096)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--zero_actions", action="store_true")
parser.add_argument("--action_multiplier", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--direct_reset_seed",
    type=int,
    default=None,
    help="Force a fresh DirectRLEnv reset after Runner construction for reset-path audits.",
)
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
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
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

    if args_cli.direct_reset_seed is None:
        observations, _ = wrapped_env.reset()
    else:
        reset_observations, _ = env.reset(seed=args_cli.direct_reset_seed)
        observations = reset_observations["policy"]
    states = wrapped_env.state()
    device = direct_env.device
    episode_max_phase = torch.zeros(args_cli.num_envs, device=device)
    episode_success = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
    episode_max_hold = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    episode_max_penetration = torch.zeros(args_cli.num_envs, device=device)
    episode_start_stage = torch.zeros(args_cli.num_envs, device=device)
    if hasattr(direct_env, "curriculum_stage_deg"):
        episode_start_stage.copy_(direct_env.curriculum_stage_deg)
    episode_start_bank_id = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    if hasattr(direct_env, "current_bank_ids"):
        episode_start_bank_id.copy_(direct_env.current_bank_ids)
    completed = 0
    successes = 0
    max_phase_sum = 0.0
    max_hold_sum = 0.0
    max_penetration_sum = 0.0
    success_max_phase_sum = 0.0
    success_max_hold_sum = 0.0
    success_max_penetration_sum = 0.0
    contact_sum = torch.zeros(5, device=device)
    force_sum = torch.zeros(5, device=device)
    angular_speed_sum = 0.0
    valid_samples = 0
    gate_hits = torch.zeros(8, device=device)
    stage_stats: dict[float, dict[str, float]] = {}
    bank_stats: dict[int, dict[str, float]] = {}
    step_trace: list[dict[str, float | int]] = []

    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            if args_cli.zero_actions:
                actions = torch.zeros(
                    (args_cli.num_envs, direct_env.cfg.action_space), device=device
                )
            else:
                outputs = runner.agent.act(observations, states, timestep=step, timesteps=args_cli.max_steps)
                actions = outputs[-1].get("mean_actions", outputs[0]) * args_cli.action_multiplier
            observations, _, terminated, truncated, _ = wrapped_env.step(actions)
            states = wrapped_env.state()

        done = (terminated | truncated).reshape(-1)
        valid = ~done
        direct_env._compute_intermediate_values()
        contacts = (direct_env.contact_force_norm > 0.03).float()
        initial_yaw = torch.atan2(direct_env.init_pen_axis[:, 1], direct_env.init_pen_axis[:, 0])
        current_yaw = torch.atan2(direct_env.pen_axis[:, 1], direct_env.pen_axis[:, 0])
        target_axis = direct_env.target_pen_quat
        target_axis = torch.stack(
            (
                1.0 - 2.0 * (target_axis[:, 2] ** 2 + target_axis[:, 3] ** 2),
                2.0 * (target_axis[:, 1] * target_axis[:, 2] + target_axis[:, 0] * target_axis[:, 3]),
                2.0 * (target_axis[:, 1] * target_axis[:, 3] - target_axis[:, 0] * target_axis[:, 2]),
            ),
            dim=-1,
        )
        target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
        direction = float(getattr(direct_env.cfg, "phase_direction_sign", -1.0))
        remaining_error = direction * torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        total_phase_deg = direct_env.cfg.target_total_phase_deg - remaining_error * (180.0 / torch.pi)
        angular_speed = torch.linalg.norm(direct_env.pen_angvel, dim=-1)
        hand_rmse = torch.sqrt(
            torch.mean((direct_env.hand_qpos - direct_env.target_hand_qpos) ** 2, dim=-1)
        )
        pen_drift = torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
        gate_values = torch.stack(
            (
                total_phase_deg >= direct_env.cfg.min_phase_deg,
                total_phase_deg <= direct_env.cfg.max_phase_deg,
                angular_speed <= direct_env.cfg.max_braked_angular_speed,
                contacts[:, :3].all(dim=-1)
                & (
                    direct_env.contact_force_norm[:, 0]
                    >= float(direct_env.cfg.min_success_thumb_force_n)
                ),
                hand_rmse < float(direct_env.cfg.max_success_hand_rmse),
                pen_drift < 0.015,
                direct_env.episode_max_penetration
                <= float(direct_env.cfg.max_success_penetration_m),
                direct_env.p4_hold_count >= direct_env.cfg.hold_steps,
            ),
            dim=-1,
        ).float()
        if len(step_trace) < 32 and torch.any(valid):
            step_trace.append(
                {
                    "step": step,
                    "valid_envs": int(valid.sum().item()),
                    "mean_action_norm": float(
                        torch.linalg.norm(actions[valid], dim=-1).mean().item()
                    ),
                    "mean_phase_deg": float(total_phase_deg[valid].mean().item()),
                    "max_phase_deg": float(total_phase_deg[valid].max().item()),
                    "mean_angular_speed_rad_s": float(angular_speed[valid].mean().item()),
                    "mean_max_penetration_mm": float(
                        direct_env.episode_max_penetration[valid].mean().item() * 1000.0
                    ),
                    "max_hold_steps": int(direct_env.p4_hold_count[valid].max().item()),
                    "mean_episode_step": float(
                        direct_env.episode_length_buf[valid].float().mean().item()
                    ),
                }
            )

        episode_max_phase[valid] = torch.maximum(episode_max_phase[valid], total_phase_deg[valid])
        episode_success[valid] |= direct_env.p4_hold_count[valid] >= direct_env.cfg.hold_steps
        episode_max_hold[valid] = torch.maximum(episode_max_hold[valid], direct_env.p4_hold_count[valid])
        episode_max_penetration[valid] = torch.maximum(
            episode_max_penetration[valid], direct_env.episode_max_penetration[valid]
        )
        contact_sum += contacts[valid].sum(dim=0)
        force_sum += direct_env.contact_force_norm[valid].sum(dim=0)
        angular_speed_sum += float(angular_speed[valid].sum().item())
        gate_hits += gate_values[valid].sum(dim=0)
        valid_samples += int(valid.sum().item())

        if done.any():
            completed_now = int(done.sum().item())
            completed += completed_now
            successes += int(episode_success[done].sum().item())
            max_phase_sum += float(episode_max_phase[done].sum().item())
            max_hold_sum += float(episode_max_hold[done].sum().item())
            max_penetration_sum += float(episode_max_penetration[done].sum().item())
            successful_done = done & episode_success
            success_max_phase_sum += float(episode_max_phase[successful_done].sum().item())
            success_max_hold_sum += float(episode_max_hold[successful_done].sum().item())
            success_max_penetration_sum += float(
                episode_max_penetration[successful_done].sum().item()
            )
            for stage in torch.unique(episode_start_stage[done]).cpu().tolist():
                stage_value = float(stage)
                stage_done = done & torch.isclose(
                    episode_start_stage, torch.tensor(stage_value, device=device), atol=0.01
                )
                stage_success = stage_done & episode_success
                entry = stage_stats.setdefault(
                    stage_value,
                    {"episodes": 0.0, "successes": 0.0, "max_phase_sum": 0.0, "max_hold_sum": 0.0},
                )
                entry["episodes"] += float(stage_done.sum().item())
                entry["successes"] += float(stage_success.sum().item())
                entry["max_phase_sum"] += float(episode_max_phase[stage_done].sum().item())
                entry["max_hold_sum"] += float(episode_max_hold[stage_done].sum().item())
            for bank_id in torch.unique(episode_start_bank_id[done]).cpu().tolist():
                bank_value = int(bank_id)
                bank_done = done & (episode_start_bank_id == bank_value)
                bank_success = bank_done & episode_success
                entry = bank_stats.setdefault(
                    bank_value,
                    {
                        "episodes": 0.0,
                        "successes": 0.0,
                        "max_phase_sum": 0.0,
                        "max_hold_sum": 0.0,
                        "max_penetration_sum": 0.0,
                        "max_penetration": 0.0,
                    },
                )
                entry["episodes"] += float(bank_done.sum().item())
                entry["successes"] += float(bank_success.sum().item())
                entry["max_phase_sum"] += float(episode_max_phase[bank_done].sum().item())
                entry["max_hold_sum"] += float(episode_max_hold[bank_done].sum().item())
                entry["max_penetration_sum"] += float(
                    episode_max_penetration[bank_done].sum().item()
                )
                entry["max_penetration"] = max(
                    entry["max_penetration"],
                    float(episode_max_penetration[bank_done].max().item()),
                )
            episode_max_phase[done] = 0.0
            episode_success[done] = False
            episode_max_hold[done] = 0
            episode_max_penetration[done] = 0.0
            if hasattr(direct_env, "curriculum_stage_deg"):
                episode_start_stage[done] = direct_env.curriculum_stage_deg[done]
            if hasattr(direct_env, "current_bank_ids"):
                episode_start_bank_id[done] = direct_env.current_bank_ids[done]
        if completed >= args_cli.episodes:
            break

    finger_names = ("thumb", "index", "middle", "ring", "pinky")
    contact_ratio = {name: float(contact_sum[i].item()) / max(valid_samples, 1) for i, name in enumerate(finger_names)}
    mean_force_n = {name: float(force_sum[i].item()) / max(valid_samples, 1) for i, name in enumerate(finger_names)}
    result = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "seed": agent_cfg["seed"],
        "zero_actions": args_cli.zero_actions,
        "action_multiplier": args_cli.action_multiplier,
        "episodes": completed,
        "successes": successes,
        "success_rate": successes / max(completed, 1),
        "mean_episode_max_phase_deg": max_phase_sum / max(completed, 1),
        "mean_episode_max_hold_ms": max_hold_sum
        / max(completed, 1)
        * (1000.0 * direct_env.cfg.decimation * direct_env.cfg.sim.dt),
        "successful_episode_mean_max_phase_deg": success_max_phase_sum / max(successes, 1),
        "successful_episode_mean_max_hold_ms": success_max_hold_sum
        / max(successes, 1)
        * (1000.0 * direct_env.cfg.decimation * direct_env.cfg.sim.dt),
        "mean_angular_speed_rad_s": angular_speed_sum / max(valid_samples, 1),
        "mean_episode_max_penetration_mm": max_penetration_sum
        / max(completed, 1)
        * 1000.0,
        "successful_episode_mean_max_penetration_mm": success_max_penetration_sum
        / max(successes, 1)
        * 1000.0,
        "contact_ratio": contact_ratio,
        "mean_force_n": mean_force_n,
        "instantaneous_gate_ratio": {
            name: float(gate_hits[i].item()) / max(valid_samples, 1)
            for i, name in enumerate(
                (
                    "phase_min",
                    "phase_max",
                    "angular_speed",
                    "triad_contact",
                    "hand_rmse",
                    "drift",
                    "penetration",
                    "full_hold_success",
                )
            )
        },
        "valid_step_samples": valid_samples,
        "step_trace": step_trace,
        "environment_max_episode_length": int(direct_env.max_episode_length),
    }
    if stage_stats:
        result["per_reset_stage"] = {
            f"{stage:g}": {
                "episodes": int(values["episodes"]),
                "successes": int(values["successes"]),
                "success_rate": values["successes"] / max(values["episodes"], 1.0),
                "mean_max_phase_deg": values["max_phase_sum"] / max(values["episodes"], 1.0),
                "mean_max_hold_ms": values["max_hold_sum"]
                / max(values["episodes"], 1.0)
                * (1000.0 * direct_env.cfg.decimation * direct_env.cfg.sim.dt),
            }
            for stage, values in sorted(stage_stats.items())
        }
    if bank_stats:
        result["per_reset_bank"] = {
            str(bank_id): {
                "episodes": int(values["episodes"]),
                "successes": int(values["successes"]),
                "success_rate": values["successes"] / max(values["episodes"], 1.0),
                "mean_max_phase_deg": values["max_phase_sum"] / max(values["episodes"], 1.0),
                "mean_max_hold_ms": values["max_hold_sum"]
                / max(values["episodes"], 1.0)
                * (1000.0 * direct_env.cfg.decimation * direct_env.cfg.sim.dt),
                "mean_max_penetration_mm": values["max_penetration_sum"]
                / max(values["episodes"], 1.0)
                * 1000.0,
                "max_penetration_mm": values["max_penetration"] * 1000.0,
            }
            for bank_id, values in sorted(bank_stats.items())
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
