from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/sim2real_s0_1/isaac_wrapper_smoke.json"

parser = argparse.ArgumentParser(description="Smoke-test the S0.1 target wrapper without training.")
parser.add_argument("--task", default="Dex5-PenSpin-Oracle-P3FixedHorizon-Direct-v0")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=12)
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
    Dex5IsaacHardwareCompatibilityWrapper,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 20260731
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()
    wrapper = Dex5IsaacHardwareCompatibilityWrapper(
        direct.joint_lower,
        direct.joint_upper,
        max_policy_delta_rad=0.025,
        max_velocity_rad_s=0.5,
        policy_hz=60.0,
    )
    generator = torch.Generator(device=direct.device).manual_seed(20260731)
    max_coupling_error = 0.0
    max_limit_violation = 0.0
    all_finite = True
    max_target_step = 0.0
    prior = wrapper.project_targets(direct.processed_targets)
    direct.processed_targets = prior
    for _ in range(args.steps):
        actions = torch.empty(
            (args.num_envs, 20), device=direct.device
        ).uniform_(-1.0, 1.0, generator=generator)
        target = wrapper.apply_to_env(direct, actions, action_scale_rad=0.025)
        max_target_step = max(
            max_target_step, float(torch.max(torch.abs(target - prior)).item())
        )
        prior = target.clone()
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
        for _ in range(int(cfg.decimation)):
            direct.scene.write_data_to_sim()
            direct.sim.step(render=False)
            direct.scene.update(dt=direct.physics_dt)
        all_finite &= bool(torch.isfinite(direct.hand.data.joint_pos).all())
        all_finite &= bool(torch.isfinite(target).all())

    report = {
        "status": "PASS"
        if all_finite and max_coupling_error <= 1e-7 and max_limit_violation <= 1e-7
        else "FAIL",
        "mode": "NO_POLICY_NO_TRAINING_TARGET_APPLICATION",
        "task": args.task,
        "num_envs": args.num_envs,
        "control_steps": args.steps,
        "finite": all_finite,
        "max_coupling_error_rad": max_coupling_error,
        "max_limit_violation_rad": max_limit_violation,
        "max_target_step_rad": max_target_step,
        "expected_max_target_step_rad": 0.5 / 60.0,
        "frozen_o0_sha256": direct.cfg.expected_bundle_sha256,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("S0_1_ISAAC_WRAPPER_SMOKE=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
