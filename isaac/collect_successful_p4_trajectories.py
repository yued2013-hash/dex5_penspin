from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export complete successful braked-P4 trajectories.")
parser.add_argument("--task", default="Dex5-CanonicalEntry45-PhaseCapture50-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_trajectories", type=int, default=50)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: F401, E402
from isaaclab.utils.math import quat_apply
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


FIELDS = (
    "qpos",
    "qvel",
    "action",
    "actuator_target_qpos",
    "contact_force_n",
    "phase_deg",
    "angular_speed_rad_s",
    "linear_speed_m_s",
    "drift_m",
    "hand_rmse_rad",
    "max_penetration_m",
    "hold_count",
)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    direct_env = env.unwrapped
    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped_env, agent_cfg)
    checkpoint = Path(args_cli.checkpoint).resolve()
    runner.agent.load(str(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    observations, _ = wrapped_env.reset()
    states = wrapped_env.state()
    histories = [{name: [] for name in FIELDS} for _ in range(args_cli.num_envs)]
    initial_qpos = [None] * args_cli.num_envs
    initial_qvel = [None] * args_cli.num_envs
    captured = np.zeros(args_cli.num_envs, dtype=bool)
    trajectories: list[dict[str, np.ndarray]] = []

    def current_state(env_id: int) -> tuple[np.ndarray, np.ndarray]:
        qpos = torch.cat(
            (direct_env.hand_qpos[env_id], direct_env.pen_pos[env_id], direct_env.pen_quat[env_id])
        )
        qvel = torch.cat(
            (direct_env.hand_qvel[env_id], direct_env.pen_linvel[env_id], direct_env.pen_angvel[env_id])
        )
        return qpos.cpu().numpy(), qvel.cpu().numpy()

    direct_env._compute_intermediate_values()
    for env_id in range(args_cli.num_envs):
        initial_qpos[env_id], initial_qvel[env_id] = current_state(env_id)

    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            outputs = runner.agent.act(observations, states, timestep=step, timesteps=args_cli.max_steps)
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, terminated, truncated, _ = wrapped_env.step(actions)
            states = wrapped_env.state()

        done = (terminated | truncated).reshape(-1)
        direct_env._compute_intermediate_values()
        target_axis = quat_apply(direct_env.target_pen_quat, direct_env.pen_x_axis)
        current_yaw = torch.atan2(direct_env.pen_axis[:, 1], direct_env.pen_axis[:, 0])
        target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
        direction = float(getattr(direct_env.cfg, "phase_direction_sign", -1.0))
        remaining_error = direction * torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        phase_deg = direct_env.cfg.target_total_phase_deg - remaining_error * (180.0 / torch.pi)
        angular_speed = torch.linalg.norm(direct_env.pen_angvel, dim=-1)
        linear_speed = torch.linalg.norm(direct_env.pen_linvel, dim=-1)
        drift = torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
        hand_rmse = torch.sqrt(
            torch.mean((direct_env.hand_qpos - direct_env.target_hand_qpos) ** 2, dim=-1)
        )

        for env_id in range(args_cli.num_envs):
            if captured[env_id] or bool(done[env_id]):
                continue
            qpos, qvel = current_state(env_id)
            row = {
                "qpos": qpos,
                "qvel": qvel,
                "action": direct_env.actions[env_id].cpu().numpy(),
                "actuator_target_qpos": direct_env.processed_targets[env_id].cpu().numpy(),
                "contact_force_n": direct_env.contact_force_norm[env_id].cpu().numpy(),
                "phase_deg": np.asarray(float(phase_deg[env_id]), dtype=np.float32),
                "angular_speed_rad_s": np.asarray(float(angular_speed[env_id]), dtype=np.float32),
                "linear_speed_m_s": np.asarray(float(linear_speed[env_id]), dtype=np.float32),
                "drift_m": np.asarray(float(drift[env_id]), dtype=np.float32),
                "hand_rmse_rad": np.asarray(float(hand_rmse[env_id]), dtype=np.float32),
                "max_penetration_m": np.asarray(
                    float(direct_env.episode_max_penetration[env_id]), dtype=np.float32
                ),
                "hold_count": np.asarray(int(direct_env.p4_hold_count[env_id]), dtype=np.int64),
            }
            for name in FIELDS:
                histories[env_id][name].append(row[name])
            if int(direct_env.p4_hold_count[env_id]) >= direct_env.cfg.hold_steps:
                trajectory = {
                    name: np.stack(histories[env_id][name]) for name in FIELDS
                }
                trajectory["initial_qpos"] = np.asarray(initial_qpos[env_id])
                trajectory["initial_qvel"] = np.asarray(initial_qvel[env_id])
                trajectories.append(trajectory)
                captured[env_id] = True
                if len(trajectories) >= args_cli.max_trajectories:
                    break
        if len(trajectories) >= args_cli.max_trajectories:
            break

        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1).cpu().tolist()
        for env_id in done_ids:
            histories[env_id] = {name: [] for name in FIELDS}
            captured[env_id] = False
            direct_env._compute_intermediate_values()
            initial_qpos[env_id], initial_qvel[env_id] = current_state(env_id)

    if len(trajectories) < args_cli.max_trajectories:
        raise RuntimeError(
            f"Only collected {len(trajectories)} successful trajectories; "
            f"required {args_cli.max_trajectories}"
        )

    trajectories = trajectories[: args_cli.max_trajectories]
    lengths = np.asarray([len(row["phase_deg"]) for row in trajectories], dtype=np.int64)
    max_length = int(lengths.max())
    payload: dict[str, np.ndarray] = {
        "lengths": lengths,
        "initial_qpos": np.stack([row["initial_qpos"] for row in trajectories]),
        "initial_qvel": np.stack([row["initial_qvel"] for row in trajectories]),
        "task": np.asarray(args_cli.task),
        "checkpoint": np.asarray(str(checkpoint)),
        "seed": np.asarray(args_cli.seed, dtype=np.int64),
        "hold_steps": np.asarray(direct_env.cfg.hold_steps, dtype=np.int64),
        "control_dt_s": np.asarray(
            direct_env.cfg.decimation * direct_env.cfg.sim.dt, dtype=np.float32
        ),
    }
    for name in FIELDS:
        sample_shape = trajectories[0][name].shape[1:]
        dtype = trajectories[0][name].dtype
        padded = np.zeros((len(trajectories), max_length, *sample_shape), dtype=dtype)
        for index, row in enumerate(trajectories):
            padded[index, : lengths[index]] = row[name]
        payload[name] = padded

    output = Path(args_cli.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(
        f"COLLECTED_SUCCESSFUL_P4_TRAJECTORIES={len(trajectories)} "
        f"MIN_LENGTH={lengths.min()} MAX_LENGTH={lengths.max()} OUTPUT={output}"
    )
    wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
