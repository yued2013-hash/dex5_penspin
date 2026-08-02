from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Search and test an unloaded ring sweep from stable P1.")
parser.add_argument("--task", default="Dex5-P1P2-RingOnly120HzH1-Direct-v0")
parser.add_argument("--source-bundle", type=Path, required=True)
parser.add_argument("--source-ids", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--population", type=int, default=128)
parser.add_argument("--replicas", type=int, default=4)
parser.add_argument("--move-steps", type=int, default=24)
parser.add_argument("--away-hold-steps", type=int, default=12)
parser.add_argument("--return-steps", type=int, default=24)
parser.add_argument("--final-hold-steps", type=int, default=60)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def aggregate(value: torch.Tensor, op: str) -> torch.Tensor:
    shaped = value.reshape(args.population, args.replicas, *value.shape[1:])
    return shaped.mean(dim=1) if op == "mean" else shaped.amax(dim=1)


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    num_envs = args.population * args.replicas
    total_steps = args.move_steps + args.away_hold_steps + args.return_steps + args.final_hold_steps
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    if cfg.split_name != "train" or int(cfg.source_phase_id) != 1:
        raise ValueError("Ring isolation is restricted to train P1 states")
    cfg.bundle_path = str(args.source_bundle.resolve())
    cfg.split_ids_path = str(args.source_ids.resolve())
    cfg.teacher_prior_state_map_path = None
    cfg.episode_length_s = max(2.0, (total_steps + 4) / 120.0)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()

    rng = np.random.default_rng(args.seed)
    candidates = rng.uniform(-1.0, 1.0, size=(args.population, 4)).astype(np.float32)
    candidates[:, 0] *= 0.10
    candidates[:, 1:] *= 0.16
    candidates[0] = 0.0
    expanded = torch.as_tensor(np.repeat(candidates, args.replicas, axis=0), device=direct.device)
    all_env_ids = torch.arange(num_envs, device=direct.device)
    direct.forced_bank_ids = torch.zeros(num_envs, dtype=torch.long, device=direct.device)
    direct._reset_idx(all_env_ids)
    direct._compute_intermediate_values()
    ring_start = direct.policy_targets[:, 8:12].clone()
    initial_pen_pos = direct.pen_pos.clone()
    initial_pen_axis = direct.pen_axis.clone()
    max_position_drift = torch.zeros(num_envs, device=direct.device)
    max_axis_drift = torch.zeros(num_envs, device=direct.device)
    max_ring_force = torch.zeros(num_envs, device=direct.device)
    max_penetration = torch.zeros(num_envs, device=direct.device)
    contact_steps = torch.zeros((num_envs, 3), device=direct.device)
    max_ring_displacement = torch.zeros(num_envs, device=direct.device)
    drop = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
    increment = 0.25 * float(direct.cfg.action_scale)

    for step in range(total_steps):
        if step < args.move_steps:
            alpha = smoothstep(torch.full((num_envs,), (step + 1) / args.move_steps, device=direct.device))
        elif step < args.move_steps + args.away_hold_steps:
            alpha = torch.ones(num_envs, device=direct.device)
        elif step < args.move_steps + args.away_hold_steps + args.return_steps:
            return_step = step - args.move_steps - args.away_hold_steps + 1
            alpha = 1.0 - smoothstep(
                torch.full((num_envs,), return_step / args.return_steps, device=direct.device)
            )
        else:
            alpha = torch.zeros(num_envs, device=direct.device)
        desired = ring_start + alpha.unsqueeze(-1) * expanded
        actions = torch.zeros((num_envs, direct.cfg.action_space), device=direct.device)
        actions[:, 8:12] = torch.clamp(
            (desired - direct.policy_targets[:, 8:12]) / increment, -1.0, 1.0
        )
        env.step(actions)
        direct._compute_intermediate_values()
        contact_steps += (direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n).float()
        max_ring_force = torch.maximum(max_ring_force, direct.contact_force_norm[:, 3])
        max_penetration = torch.maximum(max_penetration, direct.current_max_penetration)
        position_drift = torch.linalg.norm(direct.pen_pos - initial_pen_pos, dim=-1)
        axis_dot = torch.sum(direct.pen_axis * initial_pen_axis, dim=-1).abs().clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot)
        max_position_drift = torch.maximum(max_position_drift, position_drift)
        max_axis_drift = torch.maximum(max_axis_drift, axis_drift)
        max_ring_displacement = torch.maximum(
            max_ring_displacement,
            torch.linalg.norm(direct.hand_qpos[:, 8:12] - ring_start, dim=-1),
        )
        drop |= (initial_pen_pos[:, 2] - direct.pen_pos[:, 2]) > direct.cfg.drop_threshold

    ratios = contact_steps / float(total_steps)
    mean_drift = aggregate(max_position_drift, "mean")
    mean_axis = aggregate(max_axis_drift, "mean")
    mean_ring_force = aggregate(max_ring_force, "mean")
    mean_ring_move = aggregate(max_ring_displacement, "mean")
    mean_ratios = aggregate(ratios, "mean")
    max_penetration_candidate = aggregate(max_penetration, "max")
    drop_ratio = aggregate(drop.float(), "mean")
    baseline_position = float(mean_drift[0])
    baseline_axis = float(mean_axis[0])
    extra_position = mean_drift - baseline_position
    extra_axis = mean_axis - baseline_axis
    safe = (
        (mean_ring_move >= 0.08)
        & (mean_ring_force <= 0.03)
        & (mean_ratios >= 0.95).all(dim=-1)
        & (extra_position <= 0.001)
        & (extra_axis <= np.deg2rad(1.0))
        & (max_penetration_candidate <= 0.0005)
        & (drop_ratio == 0.0)
    )
    score = (
        4.0 * mean_ring_move
        - 1000.0 * extra_position.clamp_min(0.0)
        - 10.0 * torch.rad2deg(extra_axis.clamp_min(0.0))
        - 20.0 * mean_ring_force
        + 5.0 * safe.float()
    )
    best = int(torch.argmax(score))
    report = {
        "task": args.task,
        "split": "train",
        "source_bundle": str(args.source_bundle.resolve()),
        "source_bundle_sha256": sha256(args.source_bundle.resolve()),
        "seed": args.seed,
        "population": args.population,
        "replicas": args.replicas,
        "baseline_position_drift_mm": baseline_position * 1000.0,
        "baseline_axis_drift_deg": float(np.rad2deg(baseline_axis)),
        "safe_candidates": int(safe.sum()),
        "best_safe": bool(safe[best]),
        "best_ring_target_offset_rad": candidates[best].tolist(),
        "best_ring_displacement_rad": float(mean_ring_move[best]),
        "best_ring_force_n": float(mean_ring_force[best]),
        "best_TIM_contact_ratios": mean_ratios[best].cpu().tolist(),
        "best_extra_position_drift_mm": float(extra_position[best] * 1000.0),
        "best_extra_axis_drift_deg": float(torch.rad2deg(extra_axis[best])),
        "best_max_penetration_mm": float(max_penetration_candidate[best] * 1000.0),
        "best_drop_ratio": float(drop_ratio[best]),
        "acceptance": "ring move >=0.08 rad; no ring load; TIM >=95%; added drift <=1mm/1deg",
        "validation_or_test_accessed": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_RING_UNLOADED=" + json.dumps(report, ensure_ascii=False), flush=True)
    del direct.forced_bank_ids
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
