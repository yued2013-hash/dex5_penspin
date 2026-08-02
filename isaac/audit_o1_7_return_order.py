from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Audit O1.7 return ordering before PPO training."
)
parser.add_argument("--task", required=True)
parser.add_argument("--num-envs", type=int, default=96)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


CONTROLLERS = ("zero", "random", "intentional_release")


def main() -> None:
    if args.num_envs % len(CONTROLLERS):
        raise ValueError("--num-envs must be divisible by 3")
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 20260730
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()

    group_size = args.num_envs // len(CONTROLLERS)
    group_ids = torch.arange(args.num_envs, device=direct.device) // group_size
    running_return = torch.zeros(args.num_envs, device=direct.device)
    running_length = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    completed_returns: list[list[float]] = [[] for _ in CONTROLLERS]
    completed_lengths: list[list[int]] = [[] for _ in CONTROLLERS]
    terminated_counts = [0 for _ in CONTROLLERS]
    truncated_counts = [0 for _ in CONTROLLERS]
    generator = torch.Generator(device=direct.device).manual_seed(20260730)
    finite = True

    for _ in range(args.steps):
        actions = torch.zeros(
            (args.num_envs, direct.cfg.action_space), device=direct.device
        )
        random_mask = group_ids == 1
        actions[random_mask] = 0.15 * torch.randn(
            (int(random_mask.sum().item()), direct.cfg.action_space),
            generator=generator,
            device=direct.device,
        )
        actions[group_ids == 2] = -1.0

        _, reward, terminated, truncated, _ = env.step(actions)
        finite &= bool(torch.isfinite(reward).all())
        running_return += reward
        running_length += 1
        done = terminated | truncated
        for controller_index in range(len(CONTROLLERS)):
            mask = done & (group_ids == controller_index)
            if torch.any(mask):
                completed_returns[controller_index].extend(
                    running_return[mask].detach().cpu().tolist()
                )
                completed_lengths[controller_index].extend(
                    running_length[mask].detach().cpu().tolist()
                )
                terminated_counts[controller_index] += int(
                    (terminated & mask).sum().item()
                )
                truncated_counts[controller_index] += int(
                    (truncated & mask).sum().item()
                )
                running_return[mask] = 0.0
                running_length[mask] = 0

    controllers = {}
    for controller_index, name in enumerate(CONTROLLERS):
        returns = completed_returns[controller_index]
        lengths = completed_lengths[controller_index]
        controllers[name] = {
            "completed_episodes": len(returns),
            "mean_return": sum(returns) / max(len(returns), 1),
            "min_return": min(returns, default=0.0),
            "max_return": max(returns, default=0.0),
            "mean_episode_steps": sum(lengths) / max(len(lengths), 1),
            "terminated": terminated_counts[controller_index],
            "truncated": truncated_counts[controller_index],
        }

    zero_return = controllers["zero"]["mean_return"]
    intentional_return = controllers["intentional_release"]["mean_return"]
    report = {
        "status": "PASS" if finite else "FAIL",
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "finite": finite,
        "termination_mode": str(direct.cfg.termination_mode),
        "terminal_failure_penalty_scale": float(
            direct.cfg.terminal_failure_penalty_scale
        ),
        "failure_grace_steps": int(direct.cfg.failure_grace_steps),
        "return_order_pass": intentional_return < zero_return,
        "controllers": controllers,
        "bank_sha256": direct.cfg.expected_bundle_sha256,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("O1_7_RETURN_ORDER=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
