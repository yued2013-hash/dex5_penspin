from __future__ import annotations

import argparse
import json
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test the O1 privileged oracle environment.")
parser.add_argument("--task", default="Dex5-PenSpin-PrivilegedOracle-Direct-v0")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=180)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 20260730
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    finite = bool(torch.isfinite(policy_obs).all())
    reward_finite = True
    termination_count = 0
    timeout_count = 0
    max_penetration = 0.0
    max_linear_speed = 0.0
    max_target_displacement = 0.0
    generator = torch.Generator(device=direct.device).manual_seed(20260730)
    for _ in range(args.steps):
        actions = 0.15 * torch.randn(
            (args.num_envs, direct.cfg.action_space),
            generator=generator,
            device=direct.device,
        )
        obs, reward, terminated, truncated, _ = env.step(actions)
        finite &= bool(torch.isfinite(obs["policy"]).all())
        reward_finite &= bool(torch.isfinite(reward).all())
        termination_count += int(terminated.sum().item())
        timeout_count += int(truncated.sum().item())
        max_penetration = max(
            max_penetration,
            float(direct.episode_max_penetration.max().item() * 1000.0),
        )
        max_linear_speed = max(
            max_linear_speed,
            float(torch.linalg.norm(direct.pen_linvel, dim=-1).max().item()),
        )
        max_target_displacement = max(
            max_target_displacement,
            float(torch.abs(direct.processed_targets - direct.source_targets).max().item()),
        )

    counts = torch.bincount(
        direct.canonical_indices_env, minlength=5
    ).detach().cpu().tolist()
    report = {
        "status": "PASS" if finite and reward_finite else "FAIL",
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "observation_shape": list(policy_obs.shape),
        "action_shape": [args.num_envs, direct.cfg.action_space],
        "observation_finite": finite,
        "reward_finite": reward_finite,
        "termination_count": termination_count,
        "timeout_count": timeout_count,
        "max_penetration_mm": max_penetration,
        "max_linear_speed_m_s": max_linear_speed,
        "max_joint_target_displacement_rad": max_target_displacement,
        "sampled_reset_counts": dict(
            zip(
                ("P1", "P2", "P3", "P4", "P5"),
                direct.sampled_reset_counts.detach().cpu().tolist(),
                strict=True,
            )
        ),
        "final_reset_canonical_counts": dict(zip(("P1", "P2", "P3", "P4", "P5"), counts, strict=True)),
        "frozen_bank_sha256": direct.cfg.expected_bundle_sha256,
    }
    print("PRIVILEGED_ORACLE_SMOKE=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
