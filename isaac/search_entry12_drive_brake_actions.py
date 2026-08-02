from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Search a phase-local drive-to-brake action sequence for 7-to-12 degrees.")
parser.add_argument("--task", default="Dex5-CanonicalP3-Entry7ToP4Entry12-Direct-v0")
parser.add_argument("--drive-report", required=True)
parser.add_argument("--drive-candidate-id", type=int, default=859)
parser.add_argument("--brake-report")
parser.add_argument("--brake-candidate-id", type=int)
parser.add_argument("--local-sigma", type=float, default=0.0)
parser.add_argument("--sample-ring", action="store_true")
parser.add_argument("--output", required=True)
parser.add_argument("--candidates", type=int, default=1024)
parser.add_argument("--repeats", type=int, default=4)
parser.add_argument("--steps", type=int, default=14)
parser.add_argument("--seed", type=int, default=20260727)
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


INDEX_ACTIONS = (
    (-0.67732245, 0.20928061, 0.82401538, -0.91803694),
    (-0.04472941, 0.37338376, 0.99040508, 0.55671883),
    (1.0, 1.0, 1.0, 1.0),
    (0.84941292, 0.99376798, 0.90057278, 0.93595898),
)
SWITCH_STEPS = (8, 9, 10)


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    drive_report = json.loads(resolve(args.drive_report).read_text(encoding="utf-8"))
    drive_row = next(
        row for row in drive_report["top"] if row["candidate_id"] == args.drive_candidate_id
    )
    num_envs = args.candidates * args.repeats
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    direct_env = env.unwrapped
    env.reset()

    generator = torch.Generator(device=direct_env.device).manual_seed(args.seed)
    drive_action = torch.tensor(drive_row["action"], device=direct_env.device)
    if args.brake_report:
        if args.brake_candidate_id is None:
            raise ValueError("--brake-candidate-id is required with --brake-report")
        brake_report = json.loads(resolve(args.brake_report).read_text(encoding="utf-8"))
        brake_row = next(
            row for row in brake_report["top"] if row["candidate_id"] == args.brake_candidate_id
        )
        base_brake = torch.tensor(brake_row["brake_action"], device=direct_env.device)
        brake_drive = torch.cat((base_brake[4:8], base_brake[16:20])).repeat(args.candidates, 1)
        brake_drive += args.local_sigma * torch.randn(
            (args.candidates, 8), generator=generator, device=direct_env.device
        )
        brake_drive.clamp_(-1.0, 1.0)
    else:
        drive_dims = 12 if args.sample_ring else 8
        brake_drive = 2.0 * torch.rand(
            (args.candidates, drive_dims), generator=generator, device=direct_env.device
        ) - 1.0
    index_bank = torch.tensor(INDEX_ACTIONS, device=direct_env.device)
    if args.brake_report:
        base_index_id = int(brake_row["index_action_id"])
        index_ids = torch.full(
            (args.candidates,), base_index_id, dtype=torch.long, device=direct_env.device
        )
    else:
        index_ids = torch.arange(args.candidates, device=direct_env.device) % len(INDEX_ACTIONS)
    switch_steps = torch.tensor(
        [SWITCH_STEPS[i % len(SWITCH_STEPS)] for i in range(args.candidates)],
        dtype=torch.long,
        device=direct_env.device,
    )
    brake_actions = torch.zeros((args.candidates, 20), device=direct_env.device)
    brake_actions[:, :4] = index_bank[index_ids]
    brake_actions[:, 4:8] = brake_drive[:, :4]
    if args.sample_ring:
        brake_actions[:, 8:12] = brake_drive[:, 4:8]
        brake_actions[:, 16:20] = brake_drive[:, 8:12]
    else:
        brake_actions[:, 16:20] = brake_drive[:, 4:8]
    if args.brake_report:
        brake_actions[0] = base_brake
        switch_steps[0] = int(brake_row["switch_step"])
    else:
        # Include exact retention-oriented actions found in the first joint search.
        brake_actions[0] = torch.tensor(
            [-0.04472941, 0.37338376, 0.99040508, 0.55671883,
             0.56203628, 0.95029902, 0.39107537, 0.07516563,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.30172670, 0.11498058, -0.10870564, -0.18868178],
            device=direct_env.device,
        )
        brake_actions[1] = torch.tensor(
            [-0.04472941, 0.37338376, 0.99040508, 0.55671883,
             0.01103508, 0.75289714, -0.80682611, -0.25721967,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.26889598, 0.08648860, 0.12938488, -0.98984939],
            device=direct_env.device,
        )

    repeated_brake = brake_actions.repeat_interleave(args.repeats, dim=0)
    repeated_switch = switch_steps.repeat_interleave(args.repeats, dim=0)
    index_contact_steps = torch.zeros(num_envs, device=direct_env.device)
    triad_contact_steps = torch.zeros(num_envs, device=direct_env.device)
    max_hold = torch.zeros(num_envs, dtype=torch.long, device=direct_env.device)
    max_phase = torch.full((num_envs,), -180.0, device=direct_env.device)
    ring_force_sum = torch.zeros(num_envs, device=direct_env.device)
    max_speed = torch.zeros(num_envs, device=direct_env.device)
    max_penetration = torch.zeros(num_envs, device=direct_env.device)
    max_drift = torch.zeros(num_envs, device=direct_env.device)
    terminated_ever = torch.zeros(num_envs, dtype=torch.bool, device=direct_env.device)

    for step in range(args.steps):
        use_brake = step >= repeated_switch
        actions = torch.where(use_brake.unsqueeze(-1), repeated_brake, drive_action.unsqueeze(0))
        _, _, terminated, _, _ = env.step(actions)
        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > 0.03
        triad = contacts[:, :3].all(dim=-1)
        index_contact_steps += contacts[:, 1].float()
        triad_contact_steps += triad.float()
        max_hold = torch.maximum(max_hold, direct_env.p4_hold_count)
        current_yaw = torch.atan2(direct_env.pen_axis[:, 1], direct_env.pen_axis[:, 0])
        target_axis = quat_apply(direct_env.target_pen_quat, direct_env.pen_x_axis)
        target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
        remaining_error = -torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        phase = direct_env.cfg.target_total_phase_deg - remaining_error * (180.0 / torch.pi)
        max_phase = torch.maximum(max_phase, phase)
        ring_force_sum += direct_env.contact_force_norm[:, 3]
        max_speed = torch.maximum(max_speed, torch.linalg.norm(direct_env.pen_angvel, dim=-1))
        max_penetration = torch.maximum(max_penetration, direct_env.episode_max_penetration)
        max_drift = torch.maximum(
            max_drift, torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
        )
        terminated_ever |= terminated

    shape = (args.candidates, args.repeats)
    hold3_ratio = (max_hold.reshape(shape) >= 3).float().mean(dim=1)
    index_ratio = index_contact_steps.reshape(shape).mean(dim=1) / args.steps
    triad_ratio = triad_contact_steps.reshape(shape).mean(dim=1) / args.steps
    mean_phase = max_phase.reshape(shape).mean(dim=1)
    mean_ring_force = ring_force_sum.reshape(shape).mean(dim=1) / args.steps
    mean_speed = max_speed.reshape(shape).mean(dim=1)
    mean_penetration = max_penetration.reshape(shape).mean(dim=1) * 1000.0
    mean_drift = max_drift.reshape(shape).mean(dim=1) * 1000.0
    terminated_ratio = terminated_ever.reshape(shape).float().mean(dim=1)
    phase_score = torch.exp(-0.25 * torch.abs(mean_phase - direct_env.cfg.target_total_phase_deg))
    score = (
        30.0 * hold3_ratio
        + 6.0 * triad_ratio
        + 4.0 * index_ratio
        + 4.0 * phase_score
        - 2.0 * torch.clamp(mean_ring_force / 0.10, 0.0, 2.0)
        - 8.0 * terminated_ratio
        - 2.0 * torch.relu(mean_penetration - 4.0)
        - 0.25 * torch.relu(mean_drift - 15.0)
    )
    top = torch.topk(score, k=min(48, args.candidates)).indices
    rows = []
    for candidate_id in top.tolist():
        rows.append(
            {
                "candidate_id": candidate_id,
                "index_action_id": int(index_ids[candidate_id].item()),
                "switch_step": int(switch_steps[candidate_id].item()),
                "drive_action": drive_row["action"],
                "brake_action": brake_actions[candidate_id].cpu().tolist(),
                "hold3_ratio": float(hold3_ratio[candidate_id].item()),
                "triad_contact_ratio": float(triad_ratio[candidate_id].item()),
                "index_contact_ratio": float(index_ratio[candidate_id].item()),
                "mean_max_phase_deg": float(mean_phase[candidate_id].item()),
                "mean_max_angular_speed_rad_s": float(mean_speed[candidate_id].item()),
                "mean_ring_force_n": float(mean_ring_force[candidate_id].item()),
                "mean_max_penetration_mm": float(mean_penetration[candidate_id].item()),
                "mean_max_drift_mm": float(mean_drift[candidate_id].item()),
                "terminated_ratio": float(terminated_ratio[candidate_id].item()),
                "score": float(score[candidate_id].item()),
            }
        )
    result = {
        "task": args.task,
        "drive_candidate_id": args.drive_candidate_id,
        "brake_seed_candidate_id": args.brake_candidate_id,
        "local_sigma": args.local_sigma,
        "sample_ring": args.sample_ring,
        "candidates": args.candidates,
        "repeats": args.repeats,
        "steps": args.steps,
        "top": rows,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
