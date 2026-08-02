from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Search small support-finger residuals for P1-to-P2 teacher.")
parser.add_argument("--task", default="Dex5-P1P2-C1-Direct-v0")
parser.add_argument("--output", required=True)
parser.add_argument("--split", choices=("train", "validation"), default="validation")
parser.add_argument("--values", type=float, nargs="+", default=(-0.2, 0.0, 0.2, 0.4))
parser.add_argument(
    "--ring-fractions", type=float, nargs="+", default=(0.30, 0.35, 0.40, 0.45, 0.50)
)
parser.add_argument("--replicas", type=int, default=4)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=20260722)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main() -> None:
    candidates = list(itertools.product(args.ring_fractions, args.values, args.values, args.values))
    num_envs = len(candidates) * args.replicas
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = args.seed
    if args.split == "validation":
        cfg.split_ids_path = str(
            PROJECT_ROOT
            / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/P1_P2_validation_ids.npy"
        )
        cfg.split_name = "validation"
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()
    candidate_ids = torch.arange(len(candidates), device=direct.device).repeat_interleave(args.replicas)
    support = torch.tensor(candidates, dtype=torch.float32, device=direct.device)[candidate_ids]
    completed = torch.zeros(len(candidates), dtype=torch.long, device=direct.device)
    successes = torch.zeros_like(completed)
    drops = torch.zeros_like(completed)

    for _ in range(args.steps):
        actions = torch.zeros((num_envs, direct.cfg.action_space), device=direct.device)
        increment = 0.25 * direct.cfg.action_scale
        ring_source = direct.source_targets[:, 8:12]
        ring_final = direct.p2_target_ring_qpos.unsqueeze(0).expand_as(ring_source)
        fraction = support[:, 0].unsqueeze(-1)
        ring_target = ring_source + fraction * (ring_final - ring_source)
        actions[:, 8:12] = torch.clamp(
            (ring_target - direct.policy_targets[:, 8:12]) / increment, -1.0, 1.0
        )
        for column, joint_slice in enumerate((slice(1, 4), slice(5, 8), slice(17, 20)), start=1):
            actions[:, joint_slice] = support[:, column].unsqueeze(-1)
        _, _, terminated, truncated, _ = env.step(actions)
        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        if done_ids.numel():
            ids = candidate_ids[done_ids]
            completed.scatter_add_(0, ids, torch.ones_like(ids))
            successes.scatter_add_(
                0, ids, direct.last_completed_p1_p2_success[done_ids].long()
            )
            drops.scatter_add_(0, ids, direct.last_completed_p1_p2_drop[done_ids].long())

    rows = []
    for candidate_id, values in enumerate(candidates):
        count = int(completed[candidate_id].item())
        success = int(successes[candidate_id].item())
        drop = int(drops[candidate_id].item())
        rows.append(
            {
                "candidate_id": candidate_id,
                "ring_fraction": values[0],
                "thumb_flex_action": values[1],
                "index_flex_action": values[2],
                "middle_flex_action": values[3],
                "completed_episodes": count,
                "successes": success,
                "success_rate": success / max(count, 1),
                "drops": drop,
                "drop_rate": drop / max(count, 1),
            }
        )
    rows.sort(key=lambda row: (row["success_rate"], -row["drop_rate"], row["successes"]), reverse=True)
    result = {
        "task": args.task,
        "split": args.split,
        "values": args.values,
        "ring_fractions": args.ring_fractions,
        "replicas": args.replicas,
        "steps": args.steps,
        "candidates": len(candidates),
        "top": rows[:20],
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_SUPPORT_SEARCH=" + json.dumps(result, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
