from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Launch the P1-paired ring-under P2 manifold search."
)
parser.add_argument("--task", default="Dex5-P1P2-TIMAssist005120HzH1-Direct-v0")
parser.add_argument("--source-bundle", type=Path, required=True)
parser.add_argument("--source-ids", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bank-output", type=Path, required=True)
parser.add_argument("--population", type=int, default=128)
parser.add_argument("--replicas", type=int, default=4)
parser.add_argument("--iterations", type=int, default=16)
parser.add_argument("--elite-fraction", type=float, default=0.20)
parser.add_argument("--rollout-steps", type=int, default=180)
parser.add_argument("--hold-steps", type=int, default=12)
parser.add_argument("--seed", type=int, default=2821)
parser.add_argument("--fixed-clearance-offset", type=float, nargs=4, required=True)
parser.add_argument("--tim-contact-target-n", type=float, default=0.06)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    source_bundle = args.source_bundle.resolve()
    source_ids = args.source_ids.resolve()
    cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.population * args.replicas
    )
    cfg.seed = args.seed
    if cfg.split_name != "train" or int(cfg.source_phase_id) != 1:
        raise ValueError("Paired P2 manifold search is restricted to train P1 states")
    if float(cfg.support_action_mask_scale) > 0.100001:
        raise ValueError("TIM compensation must not exceed 0.005 rad")
    cfg.episode_length_s = max(
        float(cfg.episode_length_s), float(args.rollout_steps + 4) / 120.0
    )
    cfg.bundle_path = str(source_bundle)
    cfg.teacher_prior_state_map_path = None
    ids = np.asarray(np.load(source_ids, allow_pickle=False), dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("--source-ids must contain a non-empty 1-D ID array")
    cfg.split_ids_path = str(source_ids)
    env = gym.make(args.task, cfg=cfg)
    # Finish PhysX scene initialization before importing the heavier search core.
    env.reset()

    os.environ["DEX5_PAIRED_MANIFOLD_CORE_ONLY"] = "1"
    from dex5_penspin.isaac import search_p1_p2_paired_manifold_cem as core

    core.args = args
    core.run_search(env, cfg, source_bundle, source_ids)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
