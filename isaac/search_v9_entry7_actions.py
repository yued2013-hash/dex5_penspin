from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Search v9 P3-to-entry7 drive and brake actions.")
parser.add_argument("--task", default="Dex5-V9P3-Entry7-Direct-v0")
parser.add_argument("--output", required=True)
parser.add_argument("--candidates", type=int, default=4096)
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--switch_step", type=int, default=8)
parser.add_argument("--hold_steps", type=int, default=3)
parser.add_argument("--settle_steps", type=int, default=0)
parser.add_argument("--seed", type=int, default=20260716)
parser.add_argument("--local_report", type=Path, default=None)
parser.add_argument("--local_candidate", type=int, default=None)
parser.add_argument("--local_radius", type=float, default=0.25)
parser.add_argument(
    "--broadcast_candidate",
    action="store_true",
    help="Apply the selected local candidate to every reset for a robustness test.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def phase_deg(env) -> torch.Tensor:
    target_axis = quat_apply(env.target_pen_quat, env.pen_x_axis)
    current_yaw = torch.atan2(env.pen_axis[:, 1], env.pen_axis[:, 0])
    target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
    direction = float(getattr(env.cfg, "phase_direction_sign", -1.0))
    remaining = direction * torch.atan2(
        torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
    )
    return env.cfg.target_total_phase_deg - remaining * (180.0 / torch.pi)


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.candidates
    env_cfg.teacher_prefill_steps = 0
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    env.reset()

    generator = torch.Generator(device=direct.device).manual_seed(args.seed)
    drive = torch.zeros((args.candidates, 20), device=direct.device)
    brake = torch.zeros_like(drive)
    # Search only the intended T/I/M gait. Ring and pinky remain passive.
    active = torch.tensor([*range(8), *range(16, 20)], device=direct.device)
    drive[:, active] = 2.0 * torch.rand(
        (args.candidates, active.numel()), generator=generator, device=direct.device
    ) - 1.0
    brake[:, active] = 2.0 * torch.rand(
        (args.candidates, active.numel()), generator=generator, device=direct.device
    ) - 1.0

    # Include the known P3 hold action and the legacy successful entry teacher mean.
    hold_first = torch.tensor(
        [-0.05875725, 0.47818470, 0.91512668, -0.01302046,
         -0.93084800, 0.21482742, -0.16046578, -0.05733317,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.19393611, 0.20401275, -0.87499338, -0.94773507],
        device=direct.device,
    )
    hold_second = torch.tensor(
        [-0.29280782, 0.64384955, 0.66902483, -0.14371921,
         -0.56687689, 0.13527596, 0.01397680, -0.25655627,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.54662585, 0.55805105, -0.94919795, -1.0],
        device=direct.device,
    )
    drive[0].zero_()
    brake[0].zero_()
    drive[1] = hold_first
    brake[1] = hold_second
    legacy_path = PROJECT_ROOT / "reports/m3_2_30_index_near_surface/canonical_p3_p4_entry7_teacher50.npz"
    if legacy_path.exists():
        import numpy as np

        with np.load(legacy_path) as data:
            legacy = torch.as_tensor(data["teacher_actions"], device=direct.device)
        count = min(len(legacy), args.candidates - 2)
        drive[2 : count + 2] = legacy[:count, 0]
        brake[2 : count + 2] = legacy[:count, -1]
    if args.local_report is not None and args.local_candidate is not None:
        report_path = args.local_report
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        center = next(
            row for row in report["top"] if row["candidate_id"] == args.local_candidate
        )
        center_drive = torch.tensor(center["drive_action"], device=direct.device)
        center_brake = torch.tensor(center["brake_action"], device=direct.device)
        if args.broadcast_candidate:
            drive[:] = center_drive
            brake[:] = center_brake
        else:
            local_count = args.candidates // 2
            noise_drive = 2.0 * torch.rand(
                (local_count, active.numel()), generator=generator, device=direct.device
            ) - 1.0
            noise_brake = 2.0 * torch.rand(
                (local_count, active.numel()), generator=generator, device=direct.device
            ) - 1.0
            drive[-local_count:, active] = torch.clamp(
                center_drive[active] + args.local_radius * noise_drive, -1.0, 1.0
            )
            brake[-local_count:, active] = torch.clamp(
                center_brake[active] + args.local_radius * noise_brake, -1.0, 1.0
            )
            drive[-1] = center_drive
            brake[-1] = center_brake

    max_phase = torch.full((args.candidates,), -360.0, device=direct.device)
    max_penetration = torch.zeros(args.candidates, device=direct.device)
    max_speed = torch.zeros(args.candidates, device=direct.device)
    max_drift = torch.zeros(args.candidates, device=direct.device)
    triad_run = torch.zeros(args.candidates, dtype=torch.long, device=direct.device)
    max_triad_run = torch.zeros_like(triad_run)
    phase_hold = torch.zeros_like(triad_run)
    max_phase_hold = torch.zeros_like(triad_run)
    failed_ever = torch.zeros(args.candidates, dtype=torch.bool, device=direct.device)
    phase_trace = []

    if args.settle_steps > 0:
        settle_action = torch.zeros_like(drive)
        for _ in range(args.settle_steps):
            _, _, terminated, _, _ = env.step(settle_action)
            failed_ever |= terminated

    for step in range(args.steps):
        action = drive if step < args.switch_step else brake
        _, _, terminated, _, _ = env.step(action)
        failed_ever |= terminated
        direct._compute_intermediate_values()
        phase = phase_deg(direct)
        contacts = direct.contact_force_norm[:, :3] > 0.03
        triad = contacts.all(dim=-1)
        speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        drift = torch.linalg.norm(direct.pen_pos - direct.init_pen_pos, dim=-1)
        eligible = (
            triad
            & (phase >= direct.cfg.min_phase_deg)
            & (phase <= direct.cfg.max_phase_deg)
            & (speed <= direct.cfg.max_braked_angular_speed)
        )
        triad_run = torch.where(triad, triad_run + 1, torch.zeros_like(triad_run))
        phase_hold = torch.where(eligible, phase_hold + 1, torch.zeros_like(phase_hold))
        max_triad_run = torch.maximum(max_triad_run, triad_run)
        max_phase_hold = torch.maximum(max_phase_hold, phase_hold)
        max_phase = torch.maximum(max_phase, phase)
        max_penetration = torch.maximum(max_penetration, direct.episode_max_penetration)
        max_speed = torch.maximum(max_speed, speed)
        max_drift = torch.maximum(max_drift, drift)
        if step in (0, args.switch_step - 1, args.steps - 1):
            phase_trace.append(phase.detach().clone())

    accepted = (
        (max_phase_hold >= args.hold_steps)
        & (max_penetration <= 0.0005)
        & (max_drift <= 0.015)
        & ~failed_ever
    )
    score = (
        5.0 * torch.clamp(max_phase / 7.0, -1.0, 1.5)
        + 2.0 * torch.clamp(max_triad_run.float() / 6.0, 0.0, 1.0)
        + 5.0 * torch.clamp(max_phase_hold.float() / args.hold_steps, 0.0, 1.0)
        - 2.0 * torch.clamp(max_penetration / 0.0005, 0.0, 4.0)
        - torch.clamp(max_drift / 0.015, 0.0, 2.0)
    )
    top_ids = torch.topk(score, k=min(64, args.candidates)).indices
    accepted_ids = torch.nonzero(accepted, as_tuple=False).squeeze(-1)
    rows = []
    bank_ids = getattr(direct, "current_bank_ids", None)
    for candidate in top_ids.tolist():
        rows.append(
            {
                "candidate_id": candidate,
                "reset_bank_id": (
                    int(bank_ids[candidate].item()) if bank_ids is not None else None
                ),
                "accepted": bool(accepted[candidate].item()),
                "drive_action": drive[candidate].cpu().tolist(),
                "brake_action": brake[candidate].cpu().tolist(),
                "switch_step": args.switch_step,
                "max_phase_deg": float(max_phase[candidate].item()),
                "max_phase_hold_steps": int(max_phase_hold[candidate].item()),
                "max_triad_run_steps": int(max_triad_run[candidate].item()),
                "max_angular_speed_rad_s": float(max_speed[candidate].item()),
                "max_drift_mm": float(max_drift[candidate].item() * 1000.0),
                "max_penetration_mm": float(max_penetration[candidate].item() * 1000.0),
                "phase_trace_deg": [float(trace[candidate].item()) for trace in phase_trace],
                "score": float(score[candidate].item()),
            }
        )
    accepted_rows = []
    for candidate in accepted_ids.tolist():
        accepted_rows.append(
            {
                "candidate_id": candidate,
                "reset_bank_id": (
                    int(bank_ids[candidate].item()) if bank_ids is not None else None
                ),
                "drive_action": drive[candidate].cpu().tolist(),
                "brake_action": brake[candidate].cpu().tolist(),
                "max_phase_deg": float(max_phase[candidate].item()),
                "max_phase_hold_steps": int(max_phase_hold[candidate].item()),
                "max_angular_speed_rad_s": float(max_speed[candidate].item()),
                "max_drift_mm": float(max_drift[candidate].item() * 1000.0),
                "max_penetration_mm": float(max_penetration[candidate].item() * 1000.0),
                "score": float(score[candidate].item()),
            }
        )
    result = {
        "task": args.task,
        "candidates": args.candidates,
        "steps": args.steps,
        "switch_step": args.switch_step,
        "hold_steps": args.hold_steps,
        "settle_steps": args.settle_steps,
        "accepted_count": int(accepted.sum().item()),
        "accepted_ratio": float(accepted.float().mean().item()),
        "broadcast_candidate": args.broadcast_candidate,
        "accepted_reset_bank_ids": (
            bank_ids[accepted].cpu().tolist() if bank_ids is not None else []
        ),
        "accepted_candidates": accepted_rows,
        "seed_diagnostics": [
            {
                "candidate_id": candidate,
                "max_phase_deg": float(max_phase[candidate].item()),
                "max_phase_hold_steps": int(max_phase_hold[candidate].item()),
                "max_triad_run_steps": int(max_triad_run[candidate].item()),
                "max_angular_speed_rad_s": float(max_speed[candidate].item()),
                "phase_trace_deg": [float(trace[candidate].item()) for trace in phase_trace],
            }
            for candidate in range(min(3, args.candidates))
        ],
        "top": rows,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
