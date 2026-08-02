from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Search a two-stage v9 bridge drive/brake teacher.")
parser.add_argument("--task", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--candidates", type=int, default=4096)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--switch_step", type=int, default=10)
parser.add_argument("--hold_steps", type=int, default=3)
parser.add_argument("--seed", type=int, default=20260716)
parser.add_argument("--center_radius", type=float, default=0.45)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def total_phase_deg(env) -> torch.Tensor:
    target_axis = quat_apply(env.target_pen_quat, env.pen_x_axis)
    current_yaw = torch.atan2(env.pen_axis[:, 1], env.pen_axis[:, 0])
    target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
    direction = float(env.cfg.phase_direction_sign)
    remaining = direction * torch.atan2(
        torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
    )
    return env.cfg.target_total_phase_deg - remaining * (180.0 / torch.pi)


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.candidates
    env_cfg.seed = args.seed
    env_cfg.teacher_prefill_steps = 0
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    env.reset()
    reset_bank_ids = direct.current_bank_ids.detach().clone()

    source = args.source if args.source.is_absolute() else ROOT / args.source
    with np.load(source) as data:
        lengths = data["lengths"].astype(np.int64)
        terminal = np.stack([data["action"][i, lengths[i] - 1] for i in range(len(lengths))])
    center = torch.as_tensor(terminal.mean(axis=0), device=direct.device)

    generator = torch.Generator(device=direct.device).manual_seed(args.seed)
    drive = torch.zeros((args.candidates, 20), device=direct.device)
    brake = torch.zeros_like(drive)
    active = torch.tensor([*range(8), *range(16, 20)], device=direct.device)
    random_count = args.candidates // 2
    drive[:random_count, active] = 2.0 * torch.rand(
        (random_count, len(active)), generator=generator, device=direct.device
    ) - 1.0
    brake[:random_count, active] = 2.0 * torch.rand(
        (random_count, len(active)), generator=generator, device=direct.device
    ) - 1.0
    noise_drive = 2.0 * torch.rand(
        (args.candidates - random_count, len(active)), generator=generator, device=direct.device
    ) - 1.0
    noise_brake = 2.0 * torch.rand(
        (args.candidates - random_count, len(active)), generator=generator, device=direct.device
    ) - 1.0
    drive[random_count:, active] = torch.clamp(
        center[active] + args.center_radius * noise_drive, -1.0, 1.0
    )
    brake[random_count:, active] = torch.clamp(
        0.35 * center[active] + args.center_radius * noise_brake, -1.0, 1.0
    )
    drive[0] = center
    brake[0] = center
    drive[1] = center
    brake[1].zero_()

    max_phase = torch.full((args.candidates,), -360.0, device=direct.device)
    min_speed_in_band = torch.full((args.candidates,), 1.0e6, device=direct.device)
    max_hold = torch.zeros(args.candidates, dtype=torch.long, device=direct.device)
    hold = torch.zeros_like(max_hold)
    max_penetration = torch.zeros(args.candidates, device=direct.device)
    max_force = torch.zeros(args.candidates, device=direct.device)
    max_drift = torch.zeros(args.candidates, device=direct.device)
    max_thumb_force = torch.zeros(args.candidates, device=direct.device)
    min_hand_rmse = torch.full((args.candidates,), 1.0e6, device=direct.device)
    phase_trace: list[torch.Tensor] = []

    for step in range(args.steps):
        env.step(drive if step < args.switch_step else brake)
        direct._compute_intermediate_values()
        phase = total_phase_deg(direct)
        speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        triad = (direct.contact_force_norm[:, :3] > 0.03).all(dim=-1)
        drift = torch.linalg.norm(direct.pen_pos - direct.init_pen_pos, dim=-1)
        hand_rmse = torch.sqrt(
            torch.mean((direct.hand_qpos - direct.target_hand_qpos) ** 2, dim=-1)
        )
        in_band = (phase >= direct.cfg.min_phase_deg) & (phase <= direct.cfg.max_phase_deg)
        eligible = (
            in_band
            & triad
            & (speed <= direct.cfg.max_braked_angular_speed)
            & (direct.contact_force_norm[:, 0] >= direct.cfg.min_success_thumb_force_n)
            & (hand_rmse < direct.cfg.max_success_hand_rmse)
        )
        hold = torch.where(eligible, hold + 1, torch.zeros_like(hold))
        max_hold = torch.maximum(max_hold, hold)
        max_phase = torch.maximum(max_phase, phase)
        min_speed_in_band = torch.where(in_band, torch.minimum(min_speed_in_band, speed), min_speed_in_band)
        max_penetration = torch.maximum(max_penetration, direct.episode_max_penetration)
        max_force = torch.maximum(max_force, direct.contact_force_norm[:, :3].max(dim=-1).values)
        max_drift = torch.maximum(max_drift, drift)
        max_thumb_force = torch.maximum(max_thumb_force, direct.contact_force_norm[:, 0])
        min_hand_rmse = torch.minimum(min_hand_rmse, hand_rmse)
        if step in (0, args.switch_step - 1, args.steps - 1):
            phase_trace.append(phase.detach().clone())

    accepted = (
        (max_hold >= args.hold_steps)
        & (max_penetration <= direct.cfg.max_success_penetration_m)
        & (max_drift <= 0.015)
    )
    target = 0.5 * (direct.cfg.min_phase_deg + direct.cfg.max_phase_deg)
    phase_score = 1.0 - torch.clamp(torch.abs(max_phase - target) / 5.0, 0.0, 2.0)
    speed_score = torch.clamp(1.0 - min_speed_in_band / 12.0, -1.0, 1.0)
    score = (
        8.0 * torch.clamp(max_hold.float() / max(args.hold_steps, 1), 0.0, 1.0)
        + 5.0 * phase_score
        + 2.0 * speed_score
        - 3.0 * torch.clamp(max_penetration / direct.cfg.max_success_penetration_m, 0.0, 2.0)
        - 0.5 * torch.clamp(max_force / 3.0, 0.0, 3.0)
    )
    top_ids = torch.topk(score, k=min(64, args.candidates)).indices
    def candidate_row(candidate: int) -> dict:
        return {
                "candidate_id": candidate,
                "reset_bank_id": int(reset_bank_ids[candidate]),
                "accepted": bool(accepted[candidate]),
                "drive_action": drive[candidate].cpu().tolist(),
                "brake_action": brake[candidate].cpu().tolist(),
                "max_phase_deg": float(max_phase[candidate]),
                "max_hold_steps": int(max_hold[candidate]),
                "min_speed_in_band_rad_s": float(min_speed_in_band[candidate]),
                "max_penetration_mm": float(max_penetration[candidate] * 1000.0),
                "max_triad_force_n": float(max_force[candidate]),
                "max_thumb_force_n": float(max_thumb_force[candidate]),
                "min_hand_rmse_rad": float(min_hand_rmse[candidate]),
                "max_drift_mm": float(max_drift[candidate] * 1000.0),
                "phase_trace_deg": [float(trace[candidate]) for trace in phase_trace],
                "score": float(score[candidate]),
            }

    rows = [candidate_row(candidate) for candidate in top_ids.tolist()]
    accepted_ids = torch.nonzero(accepted, as_tuple=False).squeeze(-1)
    accepted_order = accepted_ids[torch.argsort(score[accepted_ids], descending=True)]
    accepted_rows = [candidate_row(candidate) for candidate in accepted_order.tolist()]
    result = {
        "task": args.task,
        "source": str(source),
        "candidates": args.candidates,
        "steps": args.steps,
        "switch_step": args.switch_step,
        "hold_steps": args.hold_steps,
        "phase_band_deg": [direct.cfg.min_phase_deg, direct.cfg.max_phase_deg],
        "accepted_count": int(accepted.sum()),
        "accepted_ratio": float(accepted.float().mean()),
        "accepted": accepted_rows,
        "top": rows,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
