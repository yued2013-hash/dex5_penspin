from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Replay every train C2 state-teacher label once.")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--seed", type=int, default=20260722)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(
        "Dex5-P1P2-C2StateTeacherResidual-Direct-v0",
        device=args.device,
        num_envs=args.batch_size,
    )
    cfg.seed = args.seed
    env = gym.make("Dex5-P1P2-C2StateTeacherResidual-Direct-v0", cfg=cfg)
    direct = env.unwrapped
    actual_success = torch.zeros(direct.bank_size, dtype=torch.bool)
    actual_drop = torch.zeros_like(actual_success)

    for start in range(0, direct.bank_size, args.batch_size):
        valid = min(args.batch_size, direct.bank_size - start)
        local_ids = torch.arange(start, start + valid, dtype=torch.long)
        if valid < args.batch_size:
            local_ids = torch.cat((local_ids, local_ids[:1].repeat(args.batch_size - valid)))
        direct.forced_bank_ids = local_ids.to(direct.device)
        env.reset()
        active = torch.ones(args.batch_size, dtype=torch.bool, device=direct.device)
        batch_success = torch.zeros_like(active)
        batch_drop = torch.zeros_like(active)
        for _ in range(args.steps):
            actions = torch.zeros((args.batch_size, cfg.action_space), device=direct.device)
            _, _, terminated, truncated, _ = env.step(actions)
            done = active & (terminated | truncated)
            batch_success[done] = direct.last_completed_p1_p2_success[done]
            batch_drop[done] = direct.last_completed_p1_p2_drop[done]
            active &= ~done
            if not active.any():
                break
        actual_success[start : start + valid] = batch_success[:valid].cpu()
        actual_drop[start : start + valid] = batch_drop[:valid].cpu()
        print(f"STATE_TEACHER_REPLAY={start + valid}/{direct.bank_size}", flush=True)

    expected_success = direct.teacher_route_success_bank.cpu()
    false_positive = torch.nonzero(expected_success & ~actual_success, as_tuple=False).squeeze(-1)
    false_negative = torch.nonzero(~expected_success & actual_success, as_tuple=False).squeeze(-1)
    global_ids = direct.global_bank_id_bank.cpu()
    result = {
        "task": "Dex5-P1P2-C2StateTeacherResidual-Direct-v0",
        "split": "train",
        "test_split_accessed": False,
        "bank_size": direct.bank_size,
        "expected_successes": int(expected_success.sum().item()),
        "actual_successes": int(actual_success.sum().item()),
        "actual_success_rate": float(actual_success.float().mean().item()),
        "actual_drops": int(actual_drop.sum().item()),
        "agreement": int((expected_success == actual_success).sum().item()),
        "agreement_rate": float((expected_success == actual_success).float().mean().item()),
        "expected_success_but_failed": int(false_positive.numel()),
        "expected_failure_but_succeeded": int(false_negative.numel()),
        "expected_success_but_failed_global_ids": global_ids[false_positive].tolist()[:100],
        "expected_failure_but_succeeded_global_ids": global_ids[false_negative].tolist()[:100],
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_STATE_TEACHER_REPLAY=" + json.dumps(result, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
