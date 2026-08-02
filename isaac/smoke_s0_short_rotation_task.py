from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Dex5_Sim2Real_S0/oracle_task/task_contract_smoke.json"
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Dex5-S0-ShortRotation-Direct-v0")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from dex5_penspin.isaac.dex5_penspin_isaaclab.tasks.index_near_surface.s0_short_rotation_env import (  # noqa: E402
    S0_P3_BANK_SHA256,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 20260801
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    observations, _ = env.reset(seed=20260801)
    finite = bool(torch.isfinite(observations["policy"]).all().item())
    max_coupling_error = 0.0
    done_count = 0
    for step in range(args.steps):
        phase = torch.arange(20, device=direct.device).float().unsqueeze(0)
        actions = 0.03 * torch.sin(phase + 0.25 * step).repeat(args.num_envs, 1)
        observations, reward, terminated, truncated, _ = env.step(actions)
        finite &= bool(torch.isfinite(observations["policy"]).all().item())
        finite &= bool(torch.isfinite(reward).all().item())
        done_count += int((terminated | truncated).sum().item())
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
    student_obs = direct.get_student_observation()
    canonical_values = set(str(value) for value in direct.canonical_id_bank.tolist())
    checks = {
        "finite": finite,
        "oracle_observation_152d": tuple(observations["policy"].shape)
        == (args.num_envs, 152),
        "student_observation_340d": tuple(student_obs.shape)
        == (args.num_envs, 340),
        "reset_bank_only_p3": canonical_values == {"P3"},
        "target_coupling_exact": max_coupling_error <= 1.0e-7,
        "bank_hash_matches_config": direct.cfg.expected_bundle_sha256
        == S0_P3_BANK_SHA256,
        "no_teacher_interface": not hasattr(direct, "teacher_action_bank"),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "task": args.task,
        "training": False,
        "teacher": False,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "done_count_during_small_action_smoke": done_count,
        "max_target_coupling_error_rad": max_coupling_error,
        "oracle_observation_shape": list(observations["policy"].shape),
        "student_observation_shape": list(student_obs.shape),
        "reset_bank_sha256": direct.cfg.expected_bundle_sha256,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    env.close()
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
