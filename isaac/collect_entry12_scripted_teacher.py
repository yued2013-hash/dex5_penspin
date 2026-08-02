from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Collect successful scripted 7-to-12 degree teacher trajectories.")
parser.add_argument("--task", default="Dex5-CanonicalP3-Entry7ToP4Entry12-Direct-v0")
parser.add_argument("--search-report", required=True)
parser.add_argument("--candidate-id", type=int, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=14)
parser.add_argument("--max-trajectories", type=int, default=50)
parser.add_argument("--min-hold-steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=20260731)
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


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def markov_qpos(env) -> torch.Tensor:
    return torch.cat((env.hand_qpos, env.pen_pos, env.pen_quat), dim=-1)


def markov_qvel(env) -> torch.Tensor:
    return torch.cat((env.hand_qvel, env.pen_linvel, env.pen_angvel), dim=-1)


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    report = json.loads(resolve(args.search_report).read_text(encoding="utf-8"))
    row = next(item for item in report["top"] if item["candidate_id"] == args.candidate_id)
    drive_action = torch.tensor(row.get("drive_action", row.get("action")), dtype=torch.float32)
    brake_action = torch.tensor(row.get("brake_action", row.get("action")), dtype=torch.float32)
    switch_step = int(row.get("switch_step", args.steps))

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    direct_env = env.unwrapped
    env.reset()
    direct_env._compute_intermediate_values()

    source_qpos = markov_qpos(direct_env).clone()
    source_qvel = markov_qvel(direct_env).clone()
    source_target = direct_env.source_targets.clone()
    source_contact_force = direct_env.contact_force_norm.clone()
    drive_actions = drive_action.to(direct_env.device).repeat(args.num_envs, 1)
    brake_actions = brake_action.to(direct_env.device).repeat(args.num_envs, 1)
    teacher_plan = torch.stack(
        [drive_action if step < switch_step else brake_action for step in range(args.steps)]
    ).to(direct_env.device)
    captured = torch.zeros(args.num_envs, dtype=torch.bool, device=direct_env.device)
    rows: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "source_qpos", "source_qvel", "source_target_qpos", "source_contact_force_n",
            "capture_qpos", "capture_qvel", "capture_target_qpos", "capture_contact_force_n",
            "teacher_actions", "success_step", "capture_phase_rad", "capture_angular_speed_rad_s",
            "capture_drift_m", "capture_max_penetration_m",
        )
    }

    for step in range(args.steps):
        actions = brake_actions if step >= switch_step else drive_actions
        env.step(actions)
        direct_env._compute_intermediate_values()
        required_hold = args.min_hold_steps or direct_env.cfg.hold_steps
        success = (direct_env.p4_hold_count >= required_hold) & (~captured)
        ids = torch.nonzero(success, as_tuple=False).squeeze(-1)
        if ids.numel():
            remaining = args.max_trajectories - len(rows["source_qpos"])
            ids = ids[:remaining]
            target_axis = quat_apply(direct_env.target_pen_quat[ids], direct_env.pen_x_axis[ids])
            current_yaw = torch.atan2(direct_env.pen_axis[ids, 1], direct_env.pen_axis[ids, 0])
            target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
            remaining_error = -torch.atan2(
                torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
            )
            phase = torch.deg2rad(
                torch.full_like(remaining_error, direct_env.cfg.target_total_phase_deg)
            ) - remaining_error
            capture_qpos = markov_qpos(direct_env)[ids]
            capture_qvel = markov_qvel(direct_env)[ids]
            drift = torch.linalg.norm(direct_env.pen_pos[ids] - direct_env.init_pen_pos[ids], dim=-1)
            speed = torch.linalg.norm(direct_env.pen_angvel[ids], dim=-1)
            for env_id, qpos, qvel, phase_value, speed_value, drift_value in zip(
                ids.tolist(), capture_qpos, capture_qvel, phase, speed, drift
            ):
                rows["source_qpos"].append(source_qpos[env_id].cpu().numpy())
                rows["source_qvel"].append(source_qvel[env_id].cpu().numpy())
                rows["source_target_qpos"].append(source_target[env_id].cpu().numpy())
                rows["source_contact_force_n"].append(source_contact_force[env_id].cpu().numpy())
                rows["capture_qpos"].append(qpos.cpu().numpy())
                rows["capture_qvel"].append(qvel.cpu().numpy())
                rows["capture_target_qpos"].append(direct_env.processed_targets[env_id].cpu().numpy())
                rows["capture_contact_force_n"].append(
                    direct_env.contact_force_norm[env_id].cpu().numpy()
                )
                rows["teacher_actions"].append(teacher_plan.cpu().numpy())
                rows["success_step"].append(np.asarray(step + 1, dtype=np.int64))
                rows["capture_phase_rad"].append(phase_value.cpu().numpy())
                rows["capture_angular_speed_rad_s"].append(speed_value.cpu().numpy())
                rows["capture_drift_m"].append(drift_value.cpu().numpy())
                rows["capture_max_penetration_m"].append(
                    direct_env.episode_max_penetration[env_id].cpu().numpy()
                )
            captured[ids] = True
        if len(rows["source_qpos"]) >= args.max_trajectories:
            break

    count = len(rows["source_qpos"])
    if count < args.max_trajectories:
        raise RuntimeError(f"Collected only {count}/{args.max_trajectories} successful trajectories")
    payload = {key: np.stack(values) for key, values in rows.items()}
    payload["switch_step"] = np.asarray(switch_step, dtype=np.int64)
    payload["candidate_id"] = np.asarray(args.candidate_id, dtype=np.int64)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "output": str(output),
        "trajectories": count,
        "switch_step": switch_step,
        "required_hold_steps": int(args.min_hold_steps or direct_env.cfg.hold_steps),
        "success_step_mean": float(payload["success_step"].mean()),
        "capture_phase_deg_mean": float(np.rad2deg(payload["capture_phase_rad"]).mean()),
        "capture_angular_speed_rad_s_mean": float(payload["capture_angular_speed_rad_s"].mean()),
        "capture_drift_mm_mean": float(payload["capture_drift_m"].mean() * 1000.0),
        "capture_max_penetration_mm_max": float(payload["capture_max_penetration_m"].max() * 1000.0),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
