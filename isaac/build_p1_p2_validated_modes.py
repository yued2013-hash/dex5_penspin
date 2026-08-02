from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Cluster train-only validated P2 states into physical modes.")
parser.add_argument("--task", default="Dex5-P1P2-ReverseR0ValidatedHold-Direct-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--npz-output", type=Path)
parser.add_argument("--clusters", type=int, default=6)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=80)
parser.add_argument("--seed", type=int, default=2821)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def kmeans(values: np.ndarray, clusters: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = values[rng.choice(len(values), size=clusters, replace=False)].copy()
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(iterations):
        distances = np.sum((values[:, None] - centers[None, :]) ** 2, axis=-1)
        next_labels = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for cluster in range(clusters):
            members = values[next_labels == cluster]
            if len(members):
                next_centers[cluster] = np.median(members, axis=0)
        if np.array_equal(labels, next_labels) and np.allclose(centers, next_centers):
            labels = next_labels
            break
        labels, centers = next_labels, next_centers
    return labels


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if cfg.split_name != "train" or int(cfg.source_phase_id) != 2:
        raise ValueError("P2 mode construction requires the frozen train-only P2 task")
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()
    count = direct.bank_size
    features = np.empty((count, 3), dtype=np.float32)
    ring_qpos = direct.source_qpos_bank[:, 8:12].detach().cpu().numpy().copy()
    global_ids = direct.global_bank_id_bank.detach().cpu().numpy().copy()
    all_env_ids = torch.arange(args.num_envs, device=direct.device)
    for start in range(0, count, args.num_envs):
        local_ids = np.arange(start, min(start + args.num_envs, count), dtype=np.int64)
        padded = np.pad(local_ids, (0, args.num_envs - len(local_ids)), mode="edge")
        direct.forced_bank_ids = torch.as_tensor(padded, device=direct.device)
        direct._reset_idx(all_env_ids)
        direct._compute_intermediate_values()
        batch_features = direct._ring_axis_features().detach().cpu().numpy()
        features[local_ids] = batch_features[: len(local_ids)]
    del direct.forced_bank_ids

    q_mean = ring_qpos.mean(axis=0)
    q_scale = ring_qpos.std(axis=0).clip(min=1.0e-4)
    f_mean = features.mean(axis=0)
    f_scale = features.std(axis=0).clip(min=1.0e-4)
    clustering_features = np.concatenate(
        ((ring_qpos - q_mean) / q_scale, (features - f_mean) / f_scale), axis=-1
    )
    labels = kmeans(clustering_features, args.clusters, args.iterations, args.seed)
    modes = []
    for mode_id in range(args.clusters):
        members = np.flatnonzero(labels == mode_id)
        modes.append(
            {
                "mode": mode_id,
                "rows": int(len(members)),
                "ring_qpos_center": np.median(ring_qpos[members], axis=0).tolist(),
                "ring_feature_center_m": np.median(features[members], axis=0).tolist(),
                "global_bank_ids": global_ids[members].tolist(),
            }
        )
    modes.sort(key=lambda row: row["rows"], reverse=True)
    for mode_id, row in enumerate(modes):
        row["mode"] = mode_id
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task": args.task,
        "split": "train",
        "validated_p2_rows": count,
        "clusters": args.clusters,
        "seed": args.seed,
        "modes": modes,
        "validation_or_test_accessed": False,
    }
    if args.npz_output is not None:
        npz_output = (
            args.npz_output if args.npz_output.is_absolute()
            else PROJECT_ROOT / args.npz_output
        )
        npz_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_output,
            global_bank_ids=global_ids,
            ring_qpos=ring_qpos,
            ring_feature_m=features,
            cluster_labels=labels,
        )
        result["npz_output"] = str(npz_output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_VALIDATED_MODES=" + json.dumps(result, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
