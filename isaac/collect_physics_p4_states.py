from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect successful physical P4 curriculum states.")
parser.add_argument("--task", default="Dex5-P4Approach-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--max_states", type=int, default=512)
parser.add_argument("--max_linear_speed", type=float, default=0.03)
parser.add_argument("--max_angular_speed", type=float, default=12.0)
parser.add_argument("--min_hold_steps", type=int, default=None)
parser.add_argument("--teacher_steps", type=int, default=0)
parser.add_argument("--require_all_contacts", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--capture_delay_steps",
    type=int,
    default=0,
    help="Wait this many successful-contact steps before capturing the Markov state.",
)
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


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg["seed"]
    env = gym.make(args_cli.task, cfg=env_cfg)
    direct_env = env.unwrapped
    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped_env, agent_cfg)
    runner.agent.load(str(Path(args_cli.checkpoint).resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    observations, _ = wrapped_env.reset()
    states = wrapped_env.state()
    qpos_rows: list[np.ndarray] = []
    qvel_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    controller_base_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []
    contact_force_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    episode_step_rows: list[np.ndarray] = []
    teacher_action_rows: list[np.ndarray] = []
    pending: dict[int, dict[str, np.ndarray | list[np.ndarray]]] = {}
    delayed_capture: dict[int, int] = {}

    def append_record(record: dict[str, np.ndarray | list[np.ndarray]]) -> None:
        qpos_rows.append(np.asarray(record["qpos"])[None, :])
        qvel_rows.append(np.asarray(record["qvel"])[None, :])
        target_rows.append(np.asarray(record["target"])[None, :])
        controller_base_rows.append(np.asarray(record["controller_base"])[None, :])
        phase_rows.append(np.asarray(record["phase"])[None])
        contact_force_rows.append(np.asarray(record["contact_force"])[None, :])
        action_rows.append(np.asarray(record["action"])[None, :])
        episode_step_rows.append(np.asarray(record["episode_step"])[None])
        if args_cli.teacher_steps:
            teacher_action_rows.append(np.stack(record["teacher_actions"]))

    for step in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(observations, states, timestep=step, timesteps=args_cli.steps)
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, terminated, truncated, _ = wrapped_env.step(actions)
            states = wrapped_env.state()

        done = (terminated | truncated).reshape(-1)
        direct_env._compute_intermediate_values()
        target_axis = quat_apply(direct_env.target_pen_quat, direct_env.pen_x_axis)
        current_yaw = torch.atan2(direct_env.pen_axis[:, 1], direct_env.pen_axis[:, 0])
        target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
        remaining_error = -torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        total_phase = torch.deg2rad(
            torch.full_like(remaining_error, direct_env.cfg.target_total_phase_deg)
        ) - remaining_error
        contacts = direct_env.contact_force_norm > 0.03
        required_ratio = contacts[:, :3].float().mean(dim=-1)
        contact_ok = contacts[:, :3].all(dim=-1) if args_cli.require_all_contacts else required_ratio >= 2.0 / 3.0
        angular_speed = torch.linalg.norm(direct_env.pen_angvel, dim=-1)
        linear_speed = torch.linalg.norm(direct_env.pen_linvel, dim=-1)
        drift = torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
        if args_cli.teacher_steps:
            for env_id in list(pending):
                if bool(done[env_id]):
                    del pending[env_id]
                    continue
                record = pending[env_id]
                record["teacher_actions"].append(direct_env.actions[env_id].cpu().numpy())
                if len(record["teacher_actions"]) >= args_cli.teacher_steps:
                    append_record(record)
                    del pending[env_id]
        required_hold_steps = args_cli.min_hold_steps or direct_env.cfg.hold_steps
        selected = (
            (~done)
            & (direct_env.p4_hold_count >= required_hold_steps)
            & contact_ok
            & (linear_speed <= args_cli.max_linear_speed)
            & (angular_speed <= args_cli.max_angular_speed)
            & (drift < 0.015)
            & (direct_env.episode_max_penetration < 0.004)
        )
        if args_cli.capture_delay_steps > 0:
            newly_selected = torch.nonzero(selected, as_tuple=False).squeeze(-1)
            for env_id_tensor in newly_selected:
                delayed_capture.setdefault(int(env_id_tensor.item()), 0)
            ready_ids: list[int] = []
            for env_id in list(delayed_capture):
                if bool(done[env_id]):
                    delayed_capture.pop(env_id, None)
                elif delayed_capture[env_id] >= args_cli.capture_delay_steps:
                    capture_valid = (
                        bool(contacts[env_id, :3].all())
                        and float(linear_speed[env_id]) <= args_cli.max_linear_speed
                        and float(angular_speed[env_id]) <= args_cli.max_angular_speed
                        and float(drift[env_id]) < 0.015
                        and float(direct_env.episode_max_penetration[env_id]) < 0.004
                    )
                    if capture_valid:
                        ready_ids.append(env_id)
                    delayed_capture.pop(env_id, None)
                else:
                    delayed_capture[env_id] += 1
            ids = torch.as_tensor(ready_ids, device=direct_env.device, dtype=torch.long)
        else:
            ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
        if ids.numel():
            qpos = torch.cat(
                (direct_env.hand_qpos[ids], direct_env.pen_pos[ids], direct_env.pen_quat[ids]), dim=-1
            ).cpu().numpy()
            qvel = torch.cat(
                (direct_env.hand_qvel[ids], direct_env.pen_linvel[ids], direct_env.pen_angvel[ids]), dim=-1
            ).cpu().numpy()
            if args_cli.teacher_steps:
                for row, env_id_tensor in enumerate(ids):
                    env_id = int(env_id_tensor.item())
                    if env_id in pending:
                        continue
                    pending[env_id] = {
                        "qpos": qpos[row],
                        "qvel": qvel[row],
                        "target": direct_env.processed_targets[env_id].cpu().numpy(),
                        "controller_base": direct_env.source_targets[env_id].cpu().numpy(),
                        "phase": total_phase[env_id].cpu().numpy(),
                        "contact_force": direct_env.contact_force_norm[env_id].cpu().numpy(),
                        "action": direct_env.actions[env_id].cpu().numpy(),
                        "episode_step": direct_env.episode_length_buf[env_id].cpu().numpy(),
                        "teacher_actions": [],
                    }
            else:
                qpos_rows.append(qpos)
                qvel_rows.append(qvel)
                target_rows.append(direct_env.processed_targets[ids].cpu().numpy())
                controller_base_rows.append(direct_env.source_targets[ids].cpu().numpy())
                phase_rows.append(total_phase[ids].cpu().numpy())
                contact_force_rows.append(direct_env.contact_force_norm[ids].cpu().numpy())
                action_rows.append(direct_env.actions[ids].cpu().numpy())
                episode_step_rows.append(direct_env.episode_length_buf[ids].cpu().numpy())
        if sum(row.shape[0] for row in qpos_rows) >= args_cli.max_states:
            break

    if not qpos_rows:
        raise RuntimeError("No P4 curriculum states passed the collection filters")
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        qpos=np.concatenate(qpos_rows)[: args_cli.max_states],
        qvel=np.concatenate(qvel_rows)[: args_cli.max_states],
        target_qpos=np.concatenate(target_rows)[: args_cli.max_states],
        controller_base_target=np.concatenate(controller_base_rows)[: args_cli.max_states],
        directional_deflection_rad=np.concatenate(phase_rows)[: args_cli.max_states],
        contact_force_n=np.concatenate(contact_force_rows)[: args_cli.max_states],
        policy_action=np.concatenate(action_rows)[: args_cli.max_states],
        episode_step=np.concatenate(episode_step_rows)[: args_cli.max_states],
    )
    if args_cli.teacher_steps:
        payload["teacher_actions"] = np.stack(teacher_action_rows)[: args_cli.max_states]
    payload["capture_delay_steps"] = np.asarray(args_cli.capture_delay_steps, dtype=np.int64)
    np.savez_compressed(output, **payload)
    print(f"COLLECTED_P4_STATES={min(sum(row.shape[0] for row in qpos_rows), args_cli.max_states)} OUTPUT={output}")
    wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
