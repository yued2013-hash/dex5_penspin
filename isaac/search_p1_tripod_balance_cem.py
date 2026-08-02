from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Search a static TIM tripod P1 with a free pen.")
parser.add_argument("--task", default="Dex5-P1P2-RingOnly120HzH1-Direct-v0")
parser.add_argument("--source-bundle", type=Path, required=True)
parser.add_argument("--source-ids", type=Path, required=True)
parser.add_argument("--source-row", type=int, required=True)
parser.add_argument("--output-bank", type=Path, required=True)
parser.add_argument("--output-ids", type=Path, required=True)
parser.add_argument("--output-report", type=Path, required=True)
parser.add_argument("--population", type=int, default=128)
parser.add_argument("--replicas", type=int, default=4)
parser.add_argument("--iterations", type=int, default=8)
parser.add_argument("--elite-fraction", type=float, default=0.15)
parser.add_argument("--settle-steps", type=int, default=24)
parser.add_argument("--hold-steps", type=int, default=96)
parser.add_argument("--seed", type=int, default=2821)
parser.add_argument("--initial-report", type=Path)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


PARAMETER_NAMES = [
    "pen_axis_m", "pen_local_y_m", "pen_local_z_m",
    "pen_rot_x_rad", "pen_rot_y_rad", "pen_rot_z_rad",
    *(f"thumb_target_j{i}_rad" for i in range(4)),
    *(f"index_target_j{i}_rad" for i in range(4)),
    *(f"middle_target_j{i}_rad" for i in range(4)),
]
LOW = np.asarray(
    [-0.025, -0.010, -0.010, *([-np.deg2rad(8.0)] * 3), *([-0.08] * 12)],
    dtype=np.float32,
)
HIGH = -LOW
INITIAL_STD = np.asarray(
    [0.008, 0.004, 0.004, *([np.deg2rad(2.5)] * 3), *([0.015] * 12)],
    dtype=np.float32,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotvec_quat(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(angle > 1.0e-8, torch.sin(half) / angle, 0.5 - angle.square() / 48.0)
    return torch.cat((torch.cos(half), rotvec * scale), dim=-1)


def aggregate(values: torch.Tensor, population: int, replicas: int, op: str) -> torch.Tensor:
    shaped = values.reshape(population, replicas, *values.shape[1:])
    if op == "mean":
        return shaped.mean(dim=1)
    if op == "max":
        return shaped.amax(dim=1)
    if op == "min":
        return shaped.amin(dim=1)
    raise ValueError(op)


def rollout(direct, env, parameters: torch.Tensor) -> dict[str, torch.Tensor]:
    num_envs = direct.num_envs
    all_env_ids = torch.arange(num_envs, device=direct.device)
    direct.forced_bank_ids = torch.full(
        (num_envs,), int(args.source_row), dtype=torch.long, device=direct.device
    )
    direct._reset_idx(all_env_ids)
    direct._compute_intermediate_values()

    local_offset = parameters[:, :3]
    world_offset = quat_apply(direct.pen_quat, local_offset)
    delta_quat = rotvec_quat(parameters[:, 3:6])
    pen_pose = torch.cat(
        (direct.pen.data.root_pos_w + world_offset, quat_mul(direct.pen_quat, delta_quat)), dim=-1
    )
    direct.pen.write_root_pose_to_sim(pen_pose)
    direct.pen.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=direct.device))

    target_offsets = torch.zeros_like(direct.support_anchor_targets)
    target_offsets[:, 16:20] = parameters[:, 6:10]
    target_offsets[:, 0:4] = parameters[:, 10:14]
    target_offsets[:, 4:8] = parameters[:, 14:18]
    direct.support_anchor_targets += target_offsets
    direct.support_anchor_targets = torch.maximum(
        torch.minimum(direct.support_anchor_targets, direct.joint_upper), direct.joint_lower
    )
    direct.policy_targets.copy_(direct.support_anchor_targets)
    direct.processed_targets.copy_(direct.policy_targets)
    direct.hand.set_joint_position_target(direct.processed_targets, joint_ids=direct.dex5_joint_ids)
    direct._compute_intermediate_values()

    hold_initial_pen_pos = direct.pen_pos.clone()
    hold_initial_pen_axis = direct.pen_axis.clone()
    settled_state: dict[str, torch.Tensor] = {}
    zero_actions = torch.zeros((num_envs, direct.cfg.action_space), device=direct.device)
    contact_steps = torch.zeros((num_envs, 3), device=direct.device)
    sampled_steps = torch.zeros(num_envs, device=direct.device)
    max_position_drift = torch.zeros(num_envs, device=direct.device)
    max_axis_drift = torch.zeros(num_envs, device=direct.device)
    max_penetration = torch.zeros(num_envs, device=direct.device)
    max_ring_force = torch.zeros(num_envs, device=direct.device)
    max_pinky_force = torch.zeros(num_envs, device=direct.device)
    drop = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
    invalid = torch.zeros_like(drop)
    terminal_linear_history = []
    terminal_angular_history = []
    terminal_position_history = []
    terminal_axis_history = []

    total_steps = int(args.settle_steps + args.hold_steps)
    for step in range(total_steps):
        _, _, _, _, _ = env.step(zero_actions)
        direct._compute_intermediate_values()
        if step == args.settle_steps - 1:
            hold_initial_pen_pos = direct.pen_pos.clone()
            hold_initial_pen_axis = direct.pen_axis.clone()
            settled_state = {
                "hand_qpos": direct.hand_qpos.clone(),
                "hand_qvel": direct.hand_qvel.clone(),
                "pen_pos": direct.pen_pos.clone(),
                "pen_quat": direct.pen_quat.clone(),
                "pen_linvel": direct.pen_linvel.clone(),
                "pen_angvel": direct.pen_angvel.clone(),
                "target": direct.support_anchor_targets.clone(),
            }
        if step >= args.settle_steps:
            contact_steps += (direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n).float()
            sampled_steps += 1.0
            position_drift = torch.linalg.norm(direct.pen_pos - hold_initial_pen_pos, dim=-1)
            axis_dot = torch.sum(direct.pen_axis * hold_initial_pen_axis, dim=-1).abs().clamp(-1.0, 1.0)
            axis_drift = torch.acos(axis_dot)
            max_position_drift = torch.maximum(max_position_drift, position_drift)
            max_axis_drift = torch.maximum(max_axis_drift, axis_drift)
            max_penetration = torch.maximum(max_penetration, direct.current_max_penetration)
            max_ring_force = torch.maximum(max_ring_force, direct.contact_force_norm[:, 3])
            max_pinky_force = torch.maximum(max_pinky_force, direct.contact_force_norm[:, 4])
            drop |= (hold_initial_pen_pos[:, 2] - direct.pen_pos[:, 2]) > direct.cfg.drop_threshold
            invalid |= ~torch.isfinite(direct.pen_pos).all(dim=-1) | ~torch.isfinite(direct.hand_qpos).all(dim=-1)
            terminal_linear_history.append(direct.pen_linvel.clone())
            angular_perp = direct.pen_angvel - (
                torch.sum(direct.pen_angvel * direct.pen_axis, dim=-1, keepdim=True)
                * direct.pen_axis
            )
            terminal_angular_history.append(angular_perp)
            terminal_position_history.append(direct.pen_pos.clone())
            terminal_axis_history.append(direct.pen_axis.clone())

    ratios = contact_steps / sampled_steps.clamp_min(1.0).unsqueeze(-1)
    terminal_window = min(12, len(terminal_linear_history))
    terminal_duration = max(1, terminal_window - 1) / 120.0
    terminal_linear = torch.linalg.norm(
        terminal_position_history[-1] - terminal_position_history[-terminal_window], dim=-1
    ) / terminal_duration
    terminal_axis_dot = torch.sum(
        terminal_axis_history[-1] * terminal_axis_history[-terminal_window], dim=-1
    ).abs().clamp(-1.0, 1.0)
    terminal_angular = torch.acos(terminal_axis_dot) / terminal_duration
    hard = drop | invalid | (max_penetration > 0.0005)
    score = (
        18.0 * ratios.amin(dim=-1)
        + 4.0 * ratios.mean(dim=-1)
        - 350.0 * max_position_drift
        - 0.20 * torch.rad2deg(max_axis_drift)
        - 80.0 * terminal_linear
        - 1.00 * terminal_angular
        - 30.0 * max_penetration
        - 1.5 * max_ring_force
        - 1.5 * max_pinky_force
        - 30.0 * hard.float()
    )
    strict = (
        ~hard
        & (ratios >= 0.95).all(dim=-1)
        & (max_position_drift <= 0.002)
        & (max_axis_drift <= np.deg2rad(2.0))
        & (terminal_linear <= 0.01)
        & (terminal_angular <= 0.5)
        & (max_ring_force <= 0.03)
        & (max_pinky_force <= 0.03)
    )
    return {
        "score": score,
        "strict": strict,
        "contact_ratios": ratios,
        "max_position_drift": max_position_drift,
        "max_axis_drift": max_axis_drift,
        "max_penetration": max_penetration,
        "terminal_linear": terminal_linear,
        "terminal_angular": terminal_angular,
        "max_ring_force": max_ring_force,
        "max_pinky_force": max_pinky_force,
        "drop": drop,
        **settled_state,
    }


def main() -> None:
    if args.population * args.replicas <= 0:
        raise ValueError("population and replicas must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.population * args.replicas
    )
    if cfg.split_name != "train" or int(cfg.source_phase_id) != 1:
        raise ValueError("Tripod search is restricted to train P1 states")
    cfg.bundle_path = str(args.source_bundle.resolve())
    cfg.split_ids_path = str(args.source_ids.resolve())
    cfg.teacher_prior_state_map_path = None
    cfg.episode_length_s = max(2.5, (args.settle_steps + args.hold_steps + 4) / 120.0)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()
    if not 0 <= args.source_row < direct.bank_size:
        raise ValueError(f"source-row {args.source_row} outside bank of {direct.bank_size}")

    population, replicas = args.population, args.replicas
    mean = np.zeros(len(PARAMETER_NAMES), dtype=np.float32)
    std = INITIAL_STD.copy()
    if args.initial_report is not None:
        initial_payload = json.loads(args.initial_report.resolve().read_text(encoding="utf-8"))
        initial_values = initial_payload["best_parameters"]
        mean = np.asarray([initial_values[name] for name in PARAMETER_NAMES], dtype=np.float32)
        mean = np.clip(mean, LOW, HIGH)
    rng = np.random.default_rng(args.seed)
    best_score = -np.inf
    best_parameters = mean.copy()
    best_metrics: dict[str, float | list[float] | bool] = {}
    best_state: dict[str, np.ndarray] = {}
    history = []
    elite_count = max(2, int(round(population * args.elite_fraction)))

    for iteration in range(args.iterations):
        candidates = np.clip(
            rng.normal(mean, std, size=(population, len(mean))).astype(np.float32), LOW, HIGH
        )
        if iteration == 0:
            candidates[0] = 0.0
        expanded = np.repeat(candidates, replicas, axis=0)
        result = rollout(direct, env, torch.as_tensor(expanded, device=direct.device))
        candidate_score = aggregate(result["score"], population, replicas, "mean")
        candidate_strict = aggregate(result["strict"].float(), population, replicas, "mean")
        order = torch.argsort(candidate_score, descending=True)
        elite_ids = order[:elite_count].cpu().numpy()
        elite = candidates[elite_ids]
        mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
        std = np.maximum(0.65 * std + 0.35 * elite.std(axis=0), 0.03 * (HIGH - LOW))

        top = int(order[0])
        top_score = float(candidate_score[top])
        if top_score > best_score:
            best_score = top_score
            best_parameters = candidates[top].copy()
            slot = top * replicas + int(torch.argmax(result["score"][top * replicas:(top + 1) * replicas]))
            best_metrics = {
                "strict_replica_ratio": float(candidate_strict[top]),
                "contact_ratios": result["contact_ratios"][slot].cpu().tolist(),
                "max_position_drift_mm": float(result["max_position_drift"][slot] * 1000.0),
                "max_axis_drift_deg": float(torch.rad2deg(result["max_axis_drift"][slot])),
                "max_penetration_mm": float(result["max_penetration"][slot] * 1000.0),
                "terminal_linear_speed_m_s": float(result["terminal_linear"][slot]),
                "terminal_angular_speed_rad_s": float(result["terminal_angular"][slot]),
                "max_ring_force_n": float(result["max_ring_force"][slot]),
                "max_pinky_force_n": float(result["max_pinky_force"][slot]),
                "drop": bool(result["drop"][slot]),
            }
            best_state = {
                "hand_qpos": result["hand_qpos"][slot].cpu().numpy(),
                "hand_qvel": result["hand_qvel"][slot].cpu().numpy(),
                "pen_pos": result["pen_pos"][slot].cpu().numpy(),
                "pen_quat": result["pen_quat"][slot].cpu().numpy(),
                "pen_linvel": result["pen_linvel"][slot].cpu().numpy(),
                "pen_angvel": result["pen_angvel"][slot].cpu().numpy(),
                "target": result["target"][slot].cpu().numpy(),
            }
        history.append({
            "iteration": iteration,
            "best_score": top_score,
            "strict_candidates": int((candidate_strict >= 0.90).sum()),
            "mean_score": float(candidate_score.mean()),
        })
        print("P1_TRIPOD_CEM_ITER=" + json.dumps(history[-1]), flush=True)

    source_bundle = args.source_bundle.resolve()
    with np.load(source_bundle, allow_pickle=False) as source:
        row_count = int(np.asarray(source["phase_ids"]).shape[0])
        payload = {
            key: (
                np.asarray(source[key])[[args.source_row]].copy()
                if np.asarray(source[key]).ndim > 0 and np.asarray(source[key]).shape[0] == row_count
                else np.asarray(source[key]).copy()
            )
            for key in source.files
        }
    qpos = payload["source_qpos"][0]
    qvel = payload["source_qvel"][0]
    qpos[:20] = best_state["hand_qpos"]
    qpos[20:23] = best_state["pen_pos"]
    qpos[23:27] = best_state["pen_quat"]
    qvel[:20] = best_state["hand_qvel"]
    qvel[20:23] = best_state["pen_linvel"]
    qvel[23:26] = best_state["pen_angvel"]
    payload["source_target_qpos"][0] = best_state["target"]
    payload["tripod_search_parameters"] = best_parameters[None]
    output_bank = args.output_bank.resolve()
    output_ids = args.output_ids.resolve()
    output_report = args.output_report.resolve()
    for path in (output_bank, output_ids, output_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_bank, **payload)
    np.save(output_ids, np.asarray([0], dtype=np.int64))
    report = {
        "task": args.task,
        "split": "train",
        "source_bundle": str(source_bundle),
        "source_bundle_sha256": sha256(source_bundle),
        "source_row": args.source_row,
        "source_global_bank_id": int(direct.global_bank_id_bank[args.source_row]),
        "pen_com_offset_x_m": float(cfg.pen_com_offset_x),
        "population": population,
        "replicas": replicas,
        "iterations": args.iterations,
        "settle_steps": args.settle_steps,
        "hold_steps": args.hold_steps,
        "best_score": best_score,
        "best_parameters": dict(zip(PARAMETER_NAMES, best_parameters.tolist(), strict=True)),
        "best_metrics": best_metrics,
        "history": history,
        "output_bank": str(output_bank),
        "output_bank_sha256": sha256(output_bank),
        "output_ids": str(output_ids),
        "output_ids_sha256": sha256(output_ids),
        "validation_or_test_accessed": False,
    }
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_TRIPOD_CEM=" + json.dumps(report, ensure_ascii=False), flush=True)
    del direct.forced_bank_ids
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
