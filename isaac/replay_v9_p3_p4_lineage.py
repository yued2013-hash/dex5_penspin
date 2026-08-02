from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
LINEAGE = REPORT / "v9_p3_to_braked_p4_lineage_teacher200.npz"
RESET_BUNDLE = REPORT / "v9_p3_to_braked_p4_continuous_replay_bundle.npz"


def build_reset_bundle() -> None:
    with np.load(LINEAGE) as data:
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        rows = np.arange(len(lengths))
        final_qpos = np.asarray(data["qpos"][rows, lengths - 1], dtype=np.float32)
        source_qpos = np.asarray(data["initial_qpos"], dtype=np.float32)
        source_qvel = np.asarray(data["initial_qvel"], dtype=np.float32)
    count = len(lengths)
    payload = {
        "source_qpos": source_qpos,
        "source_qvel": source_qvel,
        "source_target_qpos": source_qpos[:, :20].copy(),
        "source_policy_action": np.zeros((count, 20), dtype=np.float32),
        "source_teacher_actions": np.zeros((count, 0, 20), dtype=np.float32),
        "target_qpos": final_qpos,
        "target_qvel": np.zeros((count, 26), dtype=np.float32),
        "phase_ids": np.zeros(count, dtype=np.int64),
        "source_episode_step": np.zeros(count, dtype=np.int64),
        "source_phase_offset_rad": np.zeros(count, dtype=np.float32),
        "target_remaining_rad": np.deg2rad(np.full(count, 50.0, dtype=np.float32)),
        "policy_target_pen_quat": source_qpos[:, 23:27].copy(),
        "curriculum_stage_deg": np.zeros(count, dtype=np.float32),
    }
    np.savez_compressed(RESET_BUNDLE, **payload)


build_reset_bundle()
parser = argparse.ArgumentParser(description="Continuously replay lineage-traced v9 P3-to-P4 teachers.")
parser.add_argument("--task", default="Dex5-V9P3-BrakedP4-Direct-v0")
parser.add_argument("--output", type=Path, default=REPORT / "v9_p3_to_braked_p4_continuous_teacher200.npz")
parser.add_argument("--report", type=Path, default=REPORT / "v9_p3_to_braked_p4_continuous_replay.json")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_trajectories", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def total_phase_deg(env) -> torch.Tensor:
    target_axis = quat_apply(env.target_pen_quat, env.pen_x_axis)
    current_yaw = torch.atan2(env.pen_axis[:, 1], env.pen_axis[:, 0])
    target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
    remaining = float(env.cfg.phase_direction_sign) * torch.atan2(
        torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
    )
    return env.cfg.target_total_phase_deg - remaining * (180.0 / torch.pi)


def main() -> None:
    with np.load(LINEAGE) as data:
        desired_qpos = torch.as_tensor(np.asarray(data["qpos"]), device=args.device)
        desired_lengths = torch.as_tensor(np.asarray(data["lengths"]), device=args.device)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.bundle_path = str(RESET_BUNDLE)
    cfg.episode_length_s = 2.0
    cfg.reset_joint_pos_noise = 0.0
    cfg.reset_joint_vel_noise = 0.0
    cfg.reset_pen_pos_noise = 0.0
    cfg.reset_pen_vel_noise = 0.0
    cfg.teacher_prefill_steps = 0
    cfg.bridge_warmup_steps = 0
    cfg.policy_target_blend_steps = 0
    cfg.drop_threshold = 1.0
    cfg.drift_threshold = 1.0
    cfg.penetration_threshold = 0.020
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()
    bank_ids = direct.current_bank_ids.detach().clone()
    chain_lengths = desired_lengths[bank_ids]
    max_steps = int(chain_lengths.max().item())

    traces: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "qpos",
            "qvel",
            "action",
            "contact_force_n",
            "phase_deg",
            "angular_speed_rad_s",
            "drift_m",
            "hand_rmse_rad",
            "max_penetration_m",
            "hold_count",
            "tracking_rms_rad",
        )
    }
    hold = torch.zeros(args.num_envs, dtype=torch.long, device=direct.device)
    max_hold = torch.zeros_like(hold)
    terminated_ever = torch.zeros(args.num_envs, dtype=torch.bool, device=direct.device)
    for step in range(max_steps):
        desired_step = torch.minimum(
            torch.full_like(chain_lengths, step), chain_lengths - 1
        )
        desired_hand = desired_qpos[bank_ids, desired_step, :20]
        tau = torch.clamp(
            direct.episode_length_buf.float() / max(float(direct.max_episode_length - 1), 1.0),
            0.0,
            1.0,
        )
        tau = tau * tau * (3.0 - 2.0 * tau)
        baseline = torch.lerp(direct.source_targets, direct.target_hand_qpos, tau.unsqueeze(-1))
        actions = torch.clamp((desired_hand - baseline) / float(direct.cfg.action_scale), -1.0, 1.0)
        _, _, terminated, truncated, _ = env.step(actions)
        terminated_ever |= terminated | truncated
        direct._compute_intermediate_values()
        phase = total_phase_deg(direct)
        speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        drift = torch.linalg.norm(direct.pen_pos - direct.init_pen_pos, dim=-1)
        hand_rmse = torch.sqrt(torch.mean((direct.hand_qpos - direct.target_hand_qpos) ** 2, dim=-1))
        tracking = torch.sqrt(torch.mean((direct.hand_qpos - desired_hand) ** 2, dim=-1))
        contacts = direct.contact_force_norm > 0.03
        eligible = (
            (phase >= 50.0)
            & (phase <= 60.0)
            & (speed <= 12.0)
            & contacts[:, :3].all(dim=-1)
            & (direct.contact_force_norm[:, 0] >= 0.03)
            & (hand_rmse < 0.22)
            & (drift < 0.015)
            & (direct.episode_max_penetration <= 0.0005)
        )
        hold = torch.where(eligible, hold + 1, torch.zeros_like(hold))
        max_hold = torch.maximum(max_hold, hold)
        active = step < chain_lengths
        traces["qpos"].append(torch.cat((direct.hand_qpos, direct.pen_pos, direct.pen_quat), dim=-1))
        traces["qvel"].append(torch.cat((direct.hand_qvel, direct.pen_linvel, direct.pen_angvel), dim=-1))
        traces["action"].append(actions)
        traces["contact_force_n"].append(direct.contact_force_norm)
        traces["phase_deg"].append(phase)
        traces["angular_speed_rad_s"].append(speed)
        traces["drift_m"].append(drift)
        traces["hand_rmse_rad"].append(hand_rmse)
        traces["max_penetration_m"].append(direct.episode_max_penetration.clone())
        traces["hold_count"].append(hold.clone())
        traces["tracking_rms_rad"].append(tracking)
        if not active.any():
            break

    stacked = {key: torch.stack(value, dim=1) for key, value in traces.items()}
    success = (max_hold >= 18) & ~terminated_ever
    success_ids = torch.nonzero(success, as_tuple=False).squeeze(-1)[: args.max_trajectories]
    count = int(success_ids.numel())
    if count == 0:
        raise RuntimeError("Continuous replay produced no strict braked-P4 trajectories")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload: dict[str, np.ndarray] = {
        "lengths": chain_lengths[success_ids].cpu().numpy(),
        "source_lineage_row": bank_ids[success_ids].cpu().numpy(),
        "initial_qpos": direct.source_qpos_bank[bank_ids[success_ids]].cpu().numpy(),
        "initial_qvel": direct.source_qvel_bank[bank_ids[success_ids]].cpu().numpy(),
        "control_dt_s": np.asarray(cfg.decimation * cfg.sim.dt, dtype=np.float32),
    }
    for key, value in stacked.items():
        payload[key] = value[success_ids].cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    valid = torch.arange(stacked["qpos"].shape[1], device=direct.device).unsqueeze(0) < chain_lengths.unsqueeze(1)
    report = {
        "output": str(output.relative_to(ROOT)),
        "num_envs": args.num_envs,
        "strict_successes": int(success.sum().item()),
        "strict_success_rate": float(success.float().mean().item()),
        "saved_trajectories": count,
        "unique_lineage_rows_saved": int(torch.unique(bank_ids[success_ids]).numel()),
        "length_steps": {
            "min": int(chain_lengths[success_ids].min().item()),
            "max": int(chain_lengths[success_ids].max().item()),
        },
        "max_tracking_rms_rad": float(stacked["tracking_rms_rad"][valid].max().item()),
        "max_penetration_mm": float(stacked["max_penetration_m"][success_ids].max().item() * 1000.0),
        "terminal_phase_deg": {
            "min": float(stacked["phase_deg"][success_ids, chain_lengths[success_ids] - 1].min().item()),
            "max": float(stacked["phase_deg"][success_ids, chain_lengths[success_ids] - 1].max().item()),
        },
        "max_hold_ms": float(max_hold[success_ids].max().item() * cfg.decimation * cfg.sim.dt * 1000.0),
        "uninterrupted_physx": True,
    }
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
