from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Dex5-S0-ShortRotation-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--max-steps", type=int, default=4000)
parser.add_argument("--seed", type=int, default=20260801)
parser.add_argument("--reset-split", choices=("train", "validation", "test"), default="test")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.reset_split = args.reset_split
    agent_cfg["seed"] = args.seed
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    raw = gym.make(args.task, cfg=env_cfg)
    direct = raw.unwrapped
    env = SkrlVecEnvWrapper(raw, ml_framework="torch")
    runner = Runner(env, agent_cfg)
    checkpoint = Path(args.checkpoint).resolve()
    runner.agent.load(str(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    observations, _ = env.reset()
    states = env.state()
    max_penetration = 0.0
    max_coupling_error = 0.0
    steps = 0
    while int(direct.completed_s0_episodes.item()) < args.episodes and steps < args.max_steps:
        with torch.inference_mode():
            outputs = runner.agent.act(
                observations, states, timestep=steps, timesteps=args.max_steps
            )
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, _, _, _ = env.step(actions)
            states = env.state()
        max_penetration = max(
            max_penetration,
            float(direct.episode_max_penetration.max().item()),
        )
        for first, second in ((2, 3), (6, 7), (10, 11), (14, 15)):
            max_coupling_error = max(
                max_coupling_error,
                float(
                    torch.max(
                        torch.abs(
                            direct.processed_targets[:, first]
                            - direct.processed_targets[:, second]
                        )
                    ).item()
                ),
            )
        steps += 1
    completed = int(direct.completed_s0_episodes.item())
    successes = int(direct.completed_s0_successes.item())
    bank_episodes = direct.completed_s0_bank_episodes.cpu()
    bank_successes = direct.completed_s0_bank_successes.cpu()
    evaluated_bank_ids = torch.nonzero(bank_episodes > 0, as_tuple=False).flatten()
    report = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "reset_split": args.reset_split,
        "num_envs": args.num_envs,
        "requested_episodes": args.episodes,
        "completed_episodes": completed,
        "successes": successes,
        "success_rate": successes / max(completed, 1),
        "forbidden_transition_rate": int(direct.completed_s0_forbidden.item())
        / max(completed, 1),
        "failure_rate": int(direct.completed_s0_failures.item()) / max(completed, 1),
        "timeout_rate": int(direct.completed_s0_timeouts.item()) / max(completed, 1),
        "mean_peak_target_angle_deg": float(
            torch.rad2deg(direct.completed_s0_peak_angle_sum / max(completed, 1)).item()
        ),
        "max_peak_target_angle_deg": float(
            torch.rad2deg(direct.completed_s0_peak_angle_max).item()
        ),
        "max_hold_steps": int(direct.completed_s0_max_hold.item()),
        "gate_ever_rates": {
            name: int(direct.completed_s0_gate_counts[index].item())
            / max(completed, 1)
            for index, name in enumerate(
                (
                    "target_band",
                    "target_speed",
                    "wrong_axis",
                    "height",
                    "tim_all",
                    "ring_pinky_clear",
                    "no_forbidden_history",
                    "all_instantaneous",
                )
            )
        },
        "max_penetration_mm": 1000.0 * max_penetration,
        "max_target_coupling_error_rad": max_coupling_error,
        "finite": bool(torch.isfinite(observations).all().item()),
        "control_steps": steps,
        "per_bank": [
            {
                "bank_id": int(bank_id),
                "episodes": int(bank_episodes[bank_id]),
                "successes": int(bank_successes[bank_id]),
                "success_rate": float(
                    bank_successes[bank_id].float()
                    / bank_episodes[bank_id].float()
                ),
            }
            for bank_id in evaluated_bank_ids.tolist()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raw.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
