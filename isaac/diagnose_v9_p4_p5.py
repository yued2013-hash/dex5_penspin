from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Diagnose the v9 P4-to-P5 PhysX task.")
parser.add_argument("--task", default="Dex5-V9P4-P5-Direct-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=96)
parser.add_argument("--seed", type=int, default=1101)
parser.add_argument(
    "--action_mode", choices=("zero", "random", "target"), default="zero"
)
parser.add_argument("--target_gain", type=float, default=0.10)
parser.add_argument("--bundle")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float()
    return {
        "mean": float(value.mean().cpu()),
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
    }


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.bundle:
        cfg.bundle_path = str(Path(args.bundle).resolve())
    cfg.episode_length_s = max(
        cfg.episode_length_s, (args.steps + 2) * cfg.decimation * cfg.sim.dt
    )
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    obs, _ = env.reset()
    reward_rows = []
    terminated_count = 0
    truncated_count = 0
    nonfinite_steps = 0
    max_penetration_mm = 0.0
    contact_sum = torch.zeros(5, device=direct.device)
    success_ever = torch.zeros(args.num_envs, dtype=torch.bool, device=direct.device)
    latest_log: dict[str, float] = {}

    for _ in range(args.steps):
        if args.action_mode == "zero":
            actions = torch.zeros(
                (args.num_envs, direct.cfg.action_space), device=direct.device
            )
        elif args.action_mode == "random":
            actions = 0.15 * torch.randn(
                (args.num_envs, direct.cfg.action_space), device=direct.device
            )
        else:
            mask = direct._stage_action_mask()
            increment = 0.25 * direct.cfg.action_scale * mask
            target_error = direct.target_hand_qpos - direct.policy_targets
            actions = args.target_gain * target_error / increment.clamp_min(1.0e-6)
            actions = torch.where(mask > 0.0, actions, torch.zeros_like(actions))
            actions = actions.clamp(-1.0, 1.0)
        obs, reward, terminated, truncated, _ = env.step(actions)
        direct._compute_intermediate_values()
        reward_rows.append(reward.detach())
        terminated_count += int(terminated.sum().item())
        truncated_count += int(truncated.sum().item())
        finite = (
            torch.isfinite(obs["policy"]).all(dim=-1)
            & torch.isfinite(reward)
            & torch.isfinite(direct.pen_pos).all(dim=-1)
            & torch.isfinite(direct.hand_qpos).all(dim=-1)
        )
        nonfinite_steps += int((~finite).sum().item())
        contact_sum += (
            direct.contact_force_norm > direct.cfg.p5_contact_force_n
        ).float().sum(dim=0)
        success_ever |= direct.episode_p5_success
        max_penetration_mm = max(
            max_penetration_mm,
            float((direct.episode_max_penetration.max() * 1000.0).cpu()),
        )
        latest_log = {
            key: float(value.detach().cpu())
            for key, value in direct.extras.get("log", {}).items()
        }

    rewards = torch.stack(reward_rows)
    report = {
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "bundle": (
            str(Path(args.bundle).resolve())
            if args.bundle
            else str(cfg.bundle_path)
        ),
        "action_mode": args.action_mode,
        "observation_shape": list(obs["policy"].shape),
        "finite_observation": bool(torch.isfinite(obs["policy"]).all().cpu()),
        "nonfinite_env_steps": nonfinite_steps,
        "reward": tensor_stats(rewards),
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "done_reasons": {
            "drop": int(direct.p5_done_drop_count.item()),
            "drift": int(direct.p5_done_drift_count.item()),
            "penetration": int(direct.p5_done_penetration_count.item()),
            "invalid": int(direct.p5_done_invalid_count.item()),
        },
        "contact_step_ratio": {
            name: float((contact_sum[index] / (args.num_envs * args.steps)).cpu())
            for index, name in enumerate(
                ("thumb", "index", "middle", "ring", "pinky")
            )
        },
        "success_ever_ratio": float(success_ever.float().mean().cpu()),
        "completed_episodes": int(direct.completed_p5_episodes.item()),
        "completed_successes": int(direct.completed_p5_successes.item()),
        "max_penetration_mm_observed": max_penetration_mm,
        "max_terminal_penetration_mm_by_finger": {
            name: float(
                direct.p5_done_max_penetration_by_finger[index].item()
                * 1000.0
            )
            for index, name in enumerate(
                ("thumb", "index", "middle", "ring", "pinky")
            )
        },
        "final_metrics": {
            key: value
            for key, value in latest_log.items()
            if key.startswith("metric/")
        },
        "final_reward_terms": {
            key: value
            for key, value in latest_log.items()
            if key.startswith("reward/")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
