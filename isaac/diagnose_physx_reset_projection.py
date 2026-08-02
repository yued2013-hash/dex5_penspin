from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Diagnose a PhysX reset without automatic episode resets.")
parser.add_argument("--task", required=True)
parser.add_argument("--bundle", default=None)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--control_steps", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.num_envs
    if args.bundle is not None:
        bundle = Path(args.bundle)
        if not bundle.is_absolute():
            bundle = Path(__file__).resolve().parents[1] / bundle
        env_cfg.bundle_path = str(bundle)
    env_cfg.seed = args.seed
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    env_cfg.teacher_prefill_steps = 0
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    env.reset()
    actions = torch.zeros((args.num_envs, direct.cfg.action_space), device=direct.device)
    traces: list[dict[str, float | int]] = []

    def snapshot(step: int) -> None:
        direct._compute_intermediate_values()
        failed, _ = direct._compute_dones()
        height_drop = direct.init_pen_pos[:, 2] - direct.pen_pos[:, 2]
        drift = torch.linalg.norm(direct.pen_pos - direct.init_pen_pos, dim=-1)
        angular_speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        contacts = direct.contact_force_norm > 0.03
        penetration_mm = direct.episode_max_penetration * 1000.0
        current_penetration_mm = direct.current_max_penetration * 1000.0
        bank_ids = getattr(direct, "current_bank_ids", None)

        def penetration_bank_ids(threshold_mm: float) -> list[int]:
            if bank_ids is None:
                return []
            return torch.unique(bank_ids[penetration_mm > threshold_mm]).cpu().tolist()

        traces.append(
            {
                "control_step": step,
                "failed_count": int(failed.sum().item()),
                "height_drop_mm_mean": float(height_drop.mean().item() * 1000.0),
                "height_drop_mm_max": float(height_drop.max().item() * 1000.0),
                "drift_mm_mean": float(drift.mean().item() * 1000.0),
                "drift_mm_max": float(drift.max().item() * 1000.0),
                "penetration_mm_mean": float(penetration_mm.mean().item()),
                "penetration_mm_max": float(penetration_mm.max().item()),
                "current_penetration_mm_mean": float(current_penetration_mm.mean().item()),
                "current_penetration_mm_max": float(current_penetration_mm.max().item()),
                "penetration_over_0_25mm_count": int((penetration_mm > 0.25).sum().item()),
                "penetration_over_0_5mm_count": int((penetration_mm > 0.5).sum().item()),
                "penetration_over_0_25mm_bank_ids": penetration_bank_ids(0.25),
                "penetration_over_0_5mm_bank_ids": penetration_bank_ids(0.5),
                "angular_speed_rad_s_mean": float(angular_speed.mean().item()),
                "angular_speed_rad_s_max": float(angular_speed.max().item()),
                "thumb_contact_ratio": float(contacts[:, 0].float().mean().item()),
                "index_contact_ratio": float(contacts[:, 1].float().mean().item()),
                "middle_contact_ratio": float(contacts[:, 2].float().mean().item()),
            }
        )

    snapshot(0)
    for step in range(1, args.control_steps + 1):
        direct._pre_physics_step(actions)
        for _ in range(direct.cfg.decimation):
            direct._apply_action()
            direct.scene.write_data_to_sim()
            direct.sim.step(render=False)
            direct.scene.update(dt=direct.physics_dt)
        direct.episode_length_buf += 1
        snapshot(step)

    result = {
        "task": args.task,
        "num_envs": args.num_envs,
        "control_dt_s": direct.cfg.decimation * direct.cfg.sim.dt,
        "thresholds": {
            "drop_mm": direct.cfg.drop_threshold * 1000.0,
            "drift_mm": direct.cfg.drift_threshold * 1000.0,
            "penetration_mm": direct.cfg.penetration_threshold * 1000.0,
        },
        "traces": traces,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
