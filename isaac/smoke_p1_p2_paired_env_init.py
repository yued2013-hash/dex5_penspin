from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Dex5-P1P2-TIMAssist005120HzH1-Direct-v0")
parser.add_argument("--source-bundle", type=Path, required=True)
parser.add_argument("--source-ids", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--rollout-steps", type=int, default=60)
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
    run_manifold = os.environ.get("DEX5_RUN_PAIRED_MANIFOLD") == "1"
    smoke_manifold = os.environ.get("DEX5_PAIRED_MANIFOLD_SMOKE") == "1"
    requested_steps = 60 if smoke_manifold or not run_manifold else 220
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 2821
    cfg.episode_length_s = max(
        float(cfg.episode_length_s), float(requested_steps + 4) / 120.0
    )
    cfg.bundle_path = str(args.source_bundle.resolve())
    cfg.teacher_prior_state_map_path = None
    ids = np.asarray(np.load(args.source_ids.resolve(), allow_pickle=False), dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("invalid source IDs")
    cfg.split_ids_path = str(args.source_ids.resolve())
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    print("P1_P2_PAIRED_ENV_INIT=ok", flush=True)
    if run_manifold:
        smoke = smoke_manifold
        args.population = max(2, args.num_envs // (1 if smoke else 4))
        args.replicas = 1 if smoke else 4
        args.iterations = 1 if smoke else 16
        args.elite_fraction = 0.20
        args.rollout_steps = 60 if smoke else 220
        args.hold_steps = 4 if smoke else 12
        args.seed = 2821
        args.fixed_clearance_offset = (
            0.09726466983556747,
            -0.1489972621202469,
            0.02951819635927677,
            -0.15583617985248566,
        )
        args.tim_contact_target_n = 0.06
        output_dir = args.source_bundle.resolve().parent
        suffix = "smoke_v3" if smoke else "seed2821_v3"
        args.initial_report = (
            None
            if smoke
            else output_dir / "P1_617配对P2流形_CEM_seed2821_v2.json"
        )
        args.output = output_dir / f"P1_617配对P2流形_CEM_{suffix}.json"
        args.bank_output = output_dir / f"P1_617配对P2流形_CEM_{suffix}.npz"
        os.environ["DEX5_PAIRED_MANIFOLD_CORE_ONLY"] = "1"
        from dex5_penspin.isaac import search_p1_p2_paired_manifold_cem as core

        core.args = args
        core.run_search(
            env, cfg, args.source_bundle.resolve(), args.source_ids.resolve()
        )
    else:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
