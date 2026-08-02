from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
    "P1_P2_reset_bank_phase1最终validated自复验_validated.npz"
)
DEFAULT_TRAIN_IDS = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/P1_P2_train_ids.npy"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
    "P1_P2_train_ring_geometry.json"
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


parser = argparse.ArgumentParser(description="Measure P1/P2 ring geometry in the frozen PhysX bank.")
parser.add_argument("--task", default="Dex5-V9P1-P2-ResetBank-Direct-v0")
parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
parser.add_argument("--ids", type=Path, default=DEFAULT_TRAIN_IDS)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--num_envs", type=int, default=256)
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


def ring_axis_features(env, env_ids: torch.Tensor) -> torch.Tensor:
    axis = env.pen_axis[env_ids]
    world_up = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand_as(axis)
    up = world_up - torch.sum(world_up * axis, dim=-1, keepdim=True) * axis
    up = torch.nn.functional.normalize(up, dim=-1)
    lateral = torch.nn.functional.normalize(torch.cross(axis, up, dim=-1), dim=-1)
    ring_rel = env.fingertip_pos_local[env_ids, 3] - env.pen_pos[env_ids]
    axial = torch.sum(ring_rel * axis, dim=-1)
    lateral_offset = torch.sum(ring_rel * lateral, dim=-1)
    vertical_offset = torch.sum(ring_rel * up, dim=-1)
    return torch.stack((axial, lateral_offset, vertical_offset), dim=-1)


def phase_summary(features: np.ndarray) -> dict[str, object]:
    median = np.median(features, axis=0)
    center_distance = np.linalg.norm(features - median[None, :], axis=1)
    abs_error = np.abs(features - median[None, :])
    return {
        "rows": int(features.shape[0]),
        "median_m": median.tolist(),
        "mean_m": features.mean(axis=0).tolist(),
        "std_m": features.std(axis=0).tolist(),
        "absolute_error_p90_m": np.percentile(abs_error, 90.0, axis=0).tolist(),
        "absolute_error_p95_m": np.percentile(abs_error, 95.0, axis=0).tolist(),
        "center_distance_p90_m": float(np.percentile(center_distance, 90.0)),
        "center_distance_p95_m": float(np.percentile(center_distance, 95.0)),
    }


def main() -> None:
    bank = resolve(args.bank).resolve()
    ids_path = resolve(args.ids).resolve()
    output = resolve(args.output).resolve()
    with np.load(bank) as data:
        phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)
    selected_ids = np.asarray(np.load(ids_path, allow_pickle=False), dtype=np.int64)
    if np.any((selected_ids < 0) | (selected_ids >= phase_ids.size)):
        raise ValueError("Split contains bank IDs outside the frozen bank")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.bundle_path = str(bank)
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    env_cfg.teacher_prefill_steps = 0
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    env.reset()

    collected: dict[int, list[np.ndarray]] = {1: [], 2: []}
    for start in range(0, selected_ids.size, args.num_envs):
        chunk_np = selected_ids[start : start + args.num_envs]
        count = int(chunk_np.size)
        env_ids = torch.arange(count, device=direct.device)
        chunk = torch.as_tensor(chunk_np, dtype=torch.long, device=direct.device)
        direct.forced_bank_ids = chunk
        direct._reset_idx(env_ids)
        direct._compute_intermediate_values()
        features = ring_axis_features(direct, env_ids).cpu().numpy()
        chunk_phases = phase_ids[chunk_np]
        for phase in (1, 2):
            mask = chunk_phases == phase
            if np.any(mask):
                collected[phase].append(features[mask])

    p1 = np.concatenate(collected[1], axis=0)
    p2 = np.concatenate(collected[2], axis=0)
    p1_summary = phase_summary(p1)
    p2_summary = phase_summary(p2)
    p1_center = np.asarray(p1_summary["median_m"], dtype=np.float64)
    p2_center = np.asarray(p2_summary["median_m"], dtype=np.float64)
    transfer = p2_center - p1_center
    transfer_norm = float(np.linalg.norm(transfer))
    if transfer_norm < 1.0e-6:
        raise RuntimeError("P1/P2 ring geometry centers are indistinguishable")

    report = {
        "bank": str(bank.relative_to(PROJECT_ROOT)),
        "ids": str(ids_path.relative_to(PROJECT_ROOT)),
        "split_usage": "train only; validation/test IDs excluded",
        "feature_definition": [
            "ring fingertip axial offset along pen axis",
            "ring fingertip lateral offset around pen axis",
            "ring fingertip vertical offset; positive is world-up projected normal to pen axis",
        ],
        "P1": p1_summary,
        "P2": p2_summary,
        "transfer_vector_m": transfer.tolist(),
        "transfer_distance_m": transfer_norm,
        "recommended_gate": {
            "target_center_m": p2_center.tolist(),
            "target_distance_p90_m": p2_summary["center_distance_p90_m"],
            "target_distance_p95_m": p2_summary["center_distance_p95_m"],
            "curriculum_progress": (
                "dot(current_feature - P1_median, P2_median - P1_median) / "
                "||P2_median - P1_median||^2"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_RING_GEOMETRY=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
