from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="P3 PhysX regression for the audit-only S0.2 command chain."
)
parser.add_argument("--task", required=True)
parser.add_argument("--mode", choices=("zero", "small"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=20260731)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from dex5_penspin.isaac.dex5_penspin_isaaclab.tasks.index_near_surface.hardware_compatible_action import (  # noqa: E402
    COUPLED_PAIRS,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _small_action(step: int, num_envs: int, device: str) -> torch.Tensor:
    joint_phase = torch.arange(20, device=device).float() * 0.37
    env_phase = torch.arange(num_envs, device=device).float().unsqueeze(-1) * 0.013
    phase = 2.0 * math.pi * step / 24.0
    return 0.08 * torch.sin(phase + joint_phase.unsqueeze(0) + env_phase)


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    p3_rows = direct.reset_rows_by_canonical[2]
    repeats = (args.num_envs + p3_rows.numel() - 1) // p3_rows.numel()
    direct.forced_bank_ids = p3_rows.repeat(repeats)[: args.num_envs].clone()
    torch.manual_seed(args.seed + 1)
    env.reset(seed=args.seed)
    direct._compute_intermediate_values()

    initial_pen_pos = direct.pen_pos.clone()
    previous_target = direct.processed_targets.clone()
    contact_samples = 0
    tim_all_sum = 0.0
    tim_two_sum = 0.0
    any_contact_sum = 0.0
    max_target_step = 0.0
    max_coupling_error = 0.0
    max_limit_violation = 0.0
    max_drift = 0.0
    max_height_loss = 0.0
    done_count = 0
    finite = True
    ever_drop = torch.zeros(args.num_envs, dtype=torch.bool, device=direct.device)
    ever_contact_loss = torch.zeros_like(ever_drop)
    ever_ballistic = torch.zeros_like(ever_drop)

    for step in range(args.steps):
        if args.mode == "zero":
            actions = torch.zeros((args.num_envs, 20), device=direct.device)
        else:
            actions = _small_action(step, args.num_envs, direct.device)
        _, _, terminated, truncated, _ = env.step(actions)
        direct._compute_intermediate_values()
        contacts = direct.contact_force_norm > direct.cfg.contact_force_threshold_n
        tim_count = contacts[:, :3].sum(dim=-1)
        tim_all_sum += float((tim_count == 3).float().mean().item())
        tim_two_sum += float((tim_count >= 2).float().mean().item())
        any_contact_sum += float(contacts.any(dim=-1).float().mean().item())
        contact_samples += 1

        target = direct.processed_targets
        max_target_step = max(
            max_target_step, float(torch.max(torch.abs(target - previous_target)).item())
        )
        previous_target = target.clone()
        for proximal, distal in COUPLED_PAIRS:
            max_coupling_error = max(
                max_coupling_error,
                float(torch.max(torch.abs(target[:, proximal] - target[:, distal])).item()),
            )
        lower_violation = torch.clamp(direct.joint_lower - target, min=0.0)
        upper_violation = torch.clamp(target - direct.joint_upper, min=0.0)
        max_limit_violation = max(
            max_limit_violation,
            float(torch.maximum(lower_violation, upper_violation).max().item()),
        )
        drift = torch.linalg.norm(direct.pen_pos - initial_pen_pos, dim=-1)
        height_loss = initial_pen_pos[:, 2] - direct.pen_pos[:, 2]
        max_drift = max(max_drift, float(drift.max().item()))
        max_height_loss = max(max_height_loss, float(height_loss.max().item()))
        done_count += int((terminated | truncated).sum().item())
        ever_drop |= direct.current_drop
        ever_contact_loss |= direct.current_contact_loss
        ever_ballistic |= direct.current_ballistic
        finite &= bool(torch.isfinite(direct.hand.data.joint_pos).all().item())
        finite &= bool(torch.isfinite(direct.pen_pos).all().item())
        finite &= bool(torch.isfinite(target).all().item())

    actuator = getattr(direct, "s0_actuator_model", None)
    randomization = None
    command_updates = None
    initial_projection = None
    if actuator is not None:
        randomization = {
            "delay_ticks_min": int(actuator.delay_ticks.min().item()),
            "delay_ticks_max": int(actuator.delay_ticks.max().item()),
            "tracking_gain_min": float(actuator.tracking_gain.min().item()),
            "tracking_gain_max": float(actuator.tracking_gain.max().item()),
            "velocity_limit_rad_s_min": float(actuator.velocity_limit_rad_s.min().item()),
            "velocity_limit_rad_s_max": float(actuator.velocity_limit_rad_s.max().item()),
        }
        command_updates = {
            "min": int(actuator.command_update_count.min().item()),
            "max": int(actuator.command_update_count.max().item()),
        }
        initial_projection = {
            "mean_rad": float(direct.s0_initial_projection_delta_rad.mean().item()),
            "max_rad": float(direct.s0_initial_projection_delta_rad.max().item()),
        }

    report = {
        "task": args.task,
        "mode": args.mode,
        "audit_only": True,
        "training": False,
        "num_envs": args.num_envs,
        "control_steps": args.steps,
        "forced_p3_bank_ids": [
            int(direct.forced_bank_ids.min().item()),
            int(direct.forced_bank_ids.max().item()),
        ],
        "physics_hz": 1.0 / float(direct.physics_dt),
        "policy_hz": 1.0 / float(direct.step_dt),
        "finite": finite,
        "done_count": done_count,
        "tim_all_contact_ratio": tim_all_sum / contact_samples,
        "tim_at_least_two_contact_ratio": tim_two_sum / contact_samples,
        "any_contact_ratio": any_contact_sum / contact_samples,
        "drop_ever_rate": float(ever_drop.float().mean().item()),
        "contact_loss_ever_rate": float(ever_contact_loss.float().mean().item()),
        "ballistic_ever_rate": float(ever_ballistic.float().mean().item()),
        "max_pen_drift_mm": 1000.0 * max_drift,
        "max_height_loss_mm": 1000.0 * max_height_loss,
        "max_penetration_mm": float(direct.episode_max_penetration.max().item()) * 1000.0,
        "max_target_step_rad": max_target_step,
        "max_target_coupling_error_rad": max_coupling_error,
        "max_target_limit_violation_rad": max_limit_violation,
        "provisional_randomization": randomization,
        "command_updates_per_env": command_updates,
        "initial_coupling_projection_delta": initial_projection,
        "o0_sha256": direct.cfg.expected_bundle_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("S0_2_P3_REGRESSION=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
