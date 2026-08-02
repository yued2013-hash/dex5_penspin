from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Stepwise gate diagnosis for a 7-to-12 degree action candidate.")
parser.add_argument("--task", default="Dex5-CanonicalP3-Entry7ToP4Entry12-Direct-v0")
parser.add_argument("--search-report", required=True)
parser.add_argument("--candidate-id", type=int, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--seed", type=int, default=20260726)
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


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    report = json.loads(resolve(args.search_report).read_text(encoding="utf-8"))
    row = next(item for item in report["top"] if item["candidate_id"] == args.candidate_id)
    drive_action = torch.tensor(
        row.get("drive_action", row.get("action")), dtype=torch.float32
    )
    brake_action = torch.tensor(
        row.get("brake_action", row.get("action")), dtype=torch.float32
    )
    switch_step = int(row.get("switch_step", args.steps))

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    direct_env = env.unwrapped
    env.reset()
    drive_actions = drive_action.to(direct_env.device).repeat(args.num_envs, 1)
    brake_actions = brake_action.to(direct_env.device).repeat(args.num_envs, 1)

    trace = []
    for step in range(args.steps):
        actions = brake_actions if step >= switch_step else drive_actions
        _, _, terminated, _, _ = env.step(actions)
        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > 0.03
        triad = contacts[:, :3].all(dim=-1)
        current_yaw = torch.atan2(direct_env.pen_axis[:, 1], direct_env.pen_axis[:, 0])
        target_axis = quat_apply(direct_env.target_pen_quat, direct_env.pen_x_axis)
        target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
        remaining_error = -torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        phase = direct_env.cfg.target_total_phase_deg - remaining_error * (180.0 / torch.pi)
        angular_speed = torch.linalg.norm(direct_env.pen_angvel, dim=-1)
        hand_rmse = torch.sqrt(
            torch.mean((direct_env.hand_qpos - direct_env.target_hand_qpos) ** 2, dim=-1)
        )
        drift = torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
        phase_gate = (phase >= direct_env.cfg.min_phase_deg) & (phase <= direct_env.cfg.max_phase_deg)
        speed_gate = angular_speed <= direct_env.cfg.max_braked_angular_speed
        hand_gate = hand_rmse < 0.22
        drift_gate = drift < 0.015
        penetration_gate = direct_env.episode_max_penetration < 0.004
        all_gate = phase_gate & speed_gate & triad & hand_gate & drift_gate & penetration_gate
        trace.append(
            {
                "step": step + 1,
                "time_ms": (step + 1) * (1000.0 / 60.0),
                "phase_deg_mean": float(phase.mean().item()),
                "phase_deg_min": float(phase.min().item()),
                "phase_deg_max": float(phase.max().item()),
                "angular_speed_rad_s_mean": float(angular_speed.mean().item()),
                "thumb_contact_ratio": float(contacts[:, 0].float().mean().item()),
                "index_contact_ratio": float(contacts[:, 1].float().mean().item()),
                "middle_contact_ratio": float(contacts[:, 2].float().mean().item()),
                "ring_contact_ratio": float(contacts[:, 3].float().mean().item()),
                "triad_contact_ratio": float(triad.float().mean().item()),
                "hand_rmse_rad_mean": float(hand_rmse.mean().item()),
                "drift_mm_mean": float((drift.mean() * 1000.0).item()),
                "phase_gate_ratio": float(phase_gate.float().mean().item()),
                "speed_gate_ratio": float(speed_gate.float().mean().item()),
                "hand_gate_ratio": float(hand_gate.float().mean().item()),
                "all_gate_ratio": float(all_gate.float().mean().item()),
                "hold3_ratio": float((direct_env.p4_hold_count >= 3).float().mean().item()),
                "hold6_ratio": float((direct_env.p4_hold_count >= 6).float().mean().item()),
                "target_hold_ratio": float(
                    (direct_env.p4_hold_count >= direct_env.cfg.hold_steps).float().mean().item()
                ),
                "terminated_ratio": float(terminated.float().mean().item()),
            }
        )

    result = {
        "task": args.task,
        "candidate_id": args.candidate_id,
        "drive_action": drive_action.tolist(),
        "brake_action": brake_action.tolist(),
        "switch_step": switch_step,
        "num_envs": args.num_envs,
        "trace": trace,
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
