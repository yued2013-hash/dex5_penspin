from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Search same-episode phase-15 hold residuals after the first strict "
        "index/middle/thumb eligible frame."
    )
)
parser.add_argument(
    "--task",
    default=(
        "Dex5-V9P4-P5-P1-ContinuousClosureTakeover"
        "Phase14To15TeacherPrefill-Direct-v0"
    ),
)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--parent_npz", type=Path)
parser.add_argument("--parent_row", type=int, default=0)
parser.add_argument("--fixed_bank_id", type=int, default=-1)
parser.add_argument("--target_phase", type=int, default=15)
parser.add_argument(
    "--candidate_start_phase",
    type=int,
    default=-1,
    help=(
        "Begin candidate actions at this phase, while strict hold timing "
        "still starts at --target_phase. Defaults to the target phase."
    ),
)
parser.add_argument("--zero_before_entry", action="store_true")
parser.add_argument("--zero_reset_noise", action="store_true")
parser.add_argument("--zero_candidate_fraction", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=260)
parser.add_argument("--seed", type=int, default=2830)
parser.add_argument("--action_scale", type=float, default=0.35)
parser.add_argument("--action_authority", type=float, default=0.12)
parser.add_argument(
    "--post_entry_action_authority", type=float, default=0.12
)
parser.add_argument("--parent_jitter", type=float, default=0.15)
parser.add_argument("--second_jitter", type=float, default=0.20)
parser.add_argument("--parent_repeat_fraction", type=float, default=0.0)
parser.add_argument("--first_duration", type=int, default=2)
parser.add_argument("--search_first_duration", action="store_true")
parser.add_argument("--max_selected", type=int, default=512)
parser.add_argument(
    "--trace_output",
    type=Path,
    help="Optional single-environment PhysX state trace for visualization.",
)
parser.add_argument("--trace_env_id", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401


def make_candidates(
    count: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    actions = torch.zeros((count, 20), device=device)
    active = torch.as_tensor(
        [*range(8), *range(16, 20)], device=device
    )
    parent_second = None
    if args.parent_npz is not None:
        parent_data = np.load(args.parent_npz)
        parent = torch.as_tensor(
            parent_data[
                "candidate_action_first"
                if "candidate_action_first" in parent_data
                else "candidate_action"
            ][args.parent_row],
            dtype=torch.float32,
            device=device,
        )
        if "candidate_action_second" in parent_data:
            parent_second = torch.as_tensor(
                parent_data["candidate_action_second"][args.parent_row],
                dtype=torch.float32,
                device=device,
            )
        actions[:] = parent
        jitter = torch.randn(
            (count, len(active)), generator=generator, device=device
        )
        actions[:, active] += args.parent_jitter * jitter
        actions.clamp_(-1.0, 1.0)
        actions[0] = parent
    else:
        samples = torch.randn(
            (count, len(active)), generator=generator, device=device
        )
        samples /= torch.linalg.norm(
            samples, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        amplitude = torch.rand(
            (count, 1), generator=generator, device=device
        )
        actions[:, active] = args.action_scale * amplitude * samples
        actions[0].zero_()
        for row in range(min(2 * len(active), count - 1)):
            actions[row + 1].zero_()
            joint = int(active[(row // 2) % len(active)].item())
            actions[row + 1, joint] = (
                -args.action_scale
                if row % 2 == 0
                else args.action_scale
            )
    second = (
        parent_second.expand(count, -1).clone()
        if parent_second is not None
        else actions.clone()
    )
    second[:, active] += args.second_jitter * torch.randn(
        (count, len(active)), generator=generator, device=device
    )
    second.clamp_(-1.0, 1.0)
    second[0] = (
        parent_second
        if parent_second is not None
        else torch.zeros_like(second[0])
    )
    repeat_count = int(count * args.parent_repeat_fraction)
    if repeat_count > 0 and args.parent_npz is not None:
        actions[:repeat_count] = parent
        second[:repeat_count] = (
            parent_second if parent_second is not None else parent
        )
    zero_count = int(count * args.zero_candidate_fraction)
    if zero_count > 0:
        actions[:zero_count].zero_()
        second[:zero_count].zero_()
    return actions, second


def strict_eligible(direct) -> torch.Tensor:
    contacts = (
        direct.contact_force_norm > direct.cfg.p1_contact_force_n
    )
    target_axis = quat_apply(
        direct.target_pen_quat, direct.pen_x_axis
    )
    axis_dot = torch.sum(
        direct.pen_axis * target_axis, dim=-1
    ).clamp(-1.0, 1.0)
    axis_error = torch.acos(axis_dot.abs())
    triad_error = torch.cat(
        (
            direct.hand_qpos[:, 0:8]
            - direct.target_hand_qpos[:, 0:8],
            direct.hand_qpos[:, 16:20]
            - direct.target_hand_qpos[:, 16:20],
        ),
        dim=-1,
    )
    middle_rmse = torch.sqrt(
        torch.mean(
            (
                direct.hand_qpos[:, 4:8]
                - direct.target_hand_qpos[:, 4:8]
            ).square(),
            dim=-1,
        )
    )
    closure_level = (
        direct.chain_phase_ids - int(direct.cfg.training_start_phase)
    )
    unloading = closure_level > int(
        direct.cfg.thumb_wrap_stage_boundaries[-1]
    )
    thumb_index_error = torch.cat(
        (
            direct.hand_qpos[:, 0:4]
            - direct.target_hand_qpos[:, 0:4],
            direct.hand_qpos[:, 16:20]
            - direct.target_hand_qpos[:, 16:20],
        ),
        dim=-1,
    )
    triad_rmse = torch.sqrt(
        torch.mean(triad_error.square(), dim=-1)
    )
    thumb_index_rmse = torch.sqrt(
        torch.mean(thumb_index_error.square(), dim=-1)
    )
    transfer_rmse = torch.where(
        unloading, thumb_index_rmse, triad_rmse
    )
    required_contact = torch.where(
        unloading,
        contacts[:, :2].all(dim=-1),
        contacts[:, :3].all(dim=-1),
    )
    return (
        (direct.chain_phase_ids == args.target_phase)
        & required_contact
        & (
            torch.linalg.norm(
                direct.pen_pos - direct.target_pen_pos, dim=-1
            )
            <= direct.cfg.closure_pen_position_tolerance_m
        )
        & (
            axis_error
            <= torch.deg2rad(
                torch.full_like(
                    axis_error,
                    direct.cfg.closure_pen_axis_tolerance_deg,
                )
            )
        )
        & (
            torch.linalg.norm(direct.pen_angvel, dim=-1)
            <= direct.cfg.thumb_wrap_handoff_max_angular_speed_rad_s
        )
        & (
            torch.linalg.norm(direct.pen_linvel, dim=-1)
            <= direct.cfg.thumb_wrap_handoff_max_linear_speed_m_s
        )
        & (
            transfer_rmse
            <= direct.cfg.closure_stage_rmse_thresholds[-1]
        )
        & (
            middle_rmse
            <= direct.cfg.closure_final_middle_pose_rmse_rad
        )
        & (
            direct.episode_max_penetration
            <= direct.cfg.penetration_threshold
        )
        & (
            direct.contact_force_norm.max(dim=-1).values
            <= direct.cfg.p1_max_force_n
        )
        & ~contacts[:, 4]
    )


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    if args.checkpoint is None and not args.zero_before_entry:
        raise ValueError(
            "--checkpoint is required unless --zero_before_entry is used"
        )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.final_residual_scale = args.action_authority
    env_cfg.fixed_reset_bank_id = args.fixed_bank_id
    if args.zero_reset_noise:
        env_cfg.reset_joint_pos_noise = 0.0
        env_cfg.reset_joint_vel_noise = 0.0
        env_cfg.reset_pen_pos_noise = 0.0
        env_cfg.reset_pen_vel_noise = 0.0
        env_cfg.reset_pen_rot_noise_rad = 0.0
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        (args.steps + 2) * env_cfg.decimation * env_cfg.sim.dt,
    )
    agent_cfg["seed"] = args.seed
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    direct.external_final_residual_scale = torch.full(
        (args.num_envs, 1),
        -1.0,
        dtype=torch.float32,
        device=direct.device,
    )
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped, agent_cfg)
    if args.checkpoint is not None:
        runner.agent.load(str(args.checkpoint.resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    observations, _ = wrapped.reset()
    states = wrapped.state()
    hold_steps_required = int(direct.cfg.final_hold_steps)
    observation_trace = torch.zeros(
        (
            args.num_envs,
            hold_steps_required,
            observations.shape[-1],
        ),
        dtype=observations.dtype,
        device=direct.device,
    )
    action_trace = torch.zeros(
        (args.num_envs, hold_steps_required, 20),
        dtype=observations.dtype,
        device=direct.device,
    )
    armed_trace_capacity = max(hold_steps_required + 48, 64)
    armed_observation_trace = torch.zeros(
        (
            args.num_envs,
            armed_trace_capacity,
            observations.shape[-1],
        ),
        dtype=observations.dtype,
        device=direct.device,
    )
    armed_action_trace = torch.zeros(
        (args.num_envs, armed_trace_capacity, 20),
        dtype=observations.dtype,
        device=direct.device,
    )
    armed_phase_trace = torch.full(
        (args.num_envs, armed_trace_capacity),
        -1,
        dtype=torch.long,
        device=direct.device,
    )
    armed_trace_length = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    entry_hand_qpos = torch.zeros_like(direct.hand_qpos)
    entry_hand_qvel = torch.zeros_like(direct.hand_qvel)
    entry_pen_root_state = torch.zeros(
        (args.num_envs, 13),
        dtype=observations.dtype,
        device=direct.device,
    )
    entry_target_qpos = torch.zeros_like(direct.policy_targets)
    entry_policy_action = torch.zeros_like(direct.actions)
    entry_target_hand_qpos = torch.zeros_like(direct.target_hand_qpos)
    entry_closure_origin_hand = torch.zeros_like(
        direct.closure_origin_hand
    )
    entry_closure_brace_hand = torch.zeros_like(
        direct.closure_brace_hand
    )
    entry_closure_origin_pen_pos = torch.zeros_like(
        direct.closure_origin_pen_pos
    )
    entry_closure_origin_pen_quat = torch.zeros_like(
        direct.closure_origin_pen_quat
    )
    entry_init_pen_pos = torch.zeros_like(direct.init_pen_pos)
    entry_init_pen_axis = torch.zeros_like(direct.init_pen_axis)
    entry_max_penetration = torch.zeros_like(
        direct.episode_max_penetration
    )
    entry_contact_force = torch.zeros_like(direct.contact_force_norm)
    entry_chain_phase = torch.zeros_like(direct.chain_phase_ids)
    terminal_qpos = torch.zeros(
        (args.num_envs, 27),
        dtype=observations.dtype,
        device=direct.device,
    )
    terminal_qvel = torch.zeros(
        (args.num_envs, 26),
        dtype=observations.dtype,
        device=direct.device,
    )
    terminal_target_qpos = torch.zeros_like(direct.policy_targets)
    terminal_policy_action = torch.zeros_like(direct.actions)
    terminal_target_hand_qpos = torch.zeros_like(
        direct.target_hand_qpos
    )
    terminal_closure_origin_hand = torch.zeros_like(
        direct.closure_origin_hand
    )
    terminal_closure_brace_hand = torch.zeros_like(
        direct.closure_brace_hand
    )
    terminal_closure_origin_pen_pos = torch.zeros_like(
        direct.closure_origin_pen_pos
    )
    terminal_closure_origin_pen_quat = torch.zeros_like(
        direct.closure_origin_pen_quat
    )
    terminal_init_pen_pos = torch.zeros_like(direct.init_pen_pos)
    terminal_init_pen_axis = torch.zeros_like(direct.init_pen_axis)
    terminal_max_penetration = torch.zeros_like(
        direct.episode_max_penetration
    )
    terminal_contact_force = torch.zeros_like(
        direct.contact_force_norm
    )
    terminal_chain_phase = torch.zeros_like(direct.chain_phase_ids)
    candidates_first, candidates_second = make_candidates(
        args.num_envs, direct.device
    )
    if args.search_first_duration:
        duration_options = torch.as_tensor(
            (1, 2, 3, 4, 6), dtype=torch.long, device=direct.device
        )
        first_durations = duration_options[
            torch.arange(args.num_envs, device=direct.device)
            % len(duration_options)
        ]
    else:
        first_durations = torch.full(
            (args.num_envs,),
            args.first_duration,
            dtype=torch.long,
            device=direct.device,
        )
    source_bank_ids = direct.current_bank_ids.clone()
    started = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    armed = torch.zeros_like(started)
    arm_step = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    failed = torch.zeros_like(started)
    success = torch.zeros_like(started)
    start_step = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    success_step = torch.full_like(start_step, -1)
    max_hold = torch.zeros_like(start_step)
    hold_run = torch.zeros_like(start_step)
    first_loss_contact = torch.zeros_like(started)
    first_loss_axis = torch.zeros_like(started)
    first_loss_speed = torch.zeros_like(started)
    first_loss_recorded = torch.zeros_like(started)
    initial_success_serial = direct.latest_success_serial.clone()
    trace_rows: list[dict[str, np.ndarray | int | bool | float]] = []

    for step in range(args.steps):
        with torch.inference_mode():
            direct._compute_intermediate_values()
            candidate_start_phase = (
                args.target_phase
                if args.candidate_start_phase < 0
                else args.candidate_start_phase
            )
            newly_armed = (
                (direct.chain_phase_ids >= candidate_start_phase)
                & ~armed
                & ~failed
            )
            armed |= newly_armed
            arm_step[newly_armed] = step
            pre_eligible = strict_eligible(direct)
            newly_started = (
                pre_eligible & ~started & ~failed
            )
            started |= newly_started
            start_step[newly_started] = step
            hold_run[newly_started] = 1
            entry_hand_qpos[newly_started] = direct.hand_qpos[
                newly_started
            ]
            entry_hand_qvel[newly_started] = direct.hand_qvel[
                newly_started
            ]
            entry_pen_root_state[newly_started] = (
                torch.cat(
                    (
                        direct.pen_pos,
                        direct.pen_quat,
                        direct.pen_linvel,
                        direct.pen_angvel,
                    ),
                    dim=-1,
                )[newly_started]
            )
            for destination, source in (
                (entry_target_qpos, direct.policy_targets),
                (entry_policy_action, direct.actions),
                (entry_target_hand_qpos, direct.target_hand_qpos),
                (entry_closure_origin_hand, direct.closure_origin_hand),
                (entry_closure_brace_hand, direct.closure_brace_hand),
                (
                    entry_closure_origin_pen_pos,
                    direct.closure_origin_pen_pos,
                ),
                (
                    entry_closure_origin_pen_quat,
                    direct.closure_origin_pen_quat,
                ),
                (entry_init_pen_pos, direct.init_pen_pos),
                (entry_init_pen_axis, direct.init_pen_axis),
                (entry_max_penetration, direct.episode_max_penetration),
                (entry_contact_force, direct.contact_force_norm),
                (entry_chain_phase, direct.chain_phase_ids),
            ):
                destination[newly_started] = source[newly_started]
            max_hold = torch.maximum(max_hold, hold_run)
            direct.external_final_residual_scale[armed] = (
                args.post_entry_action_authority
            )
            outputs = runner.agent.act(
                observations,
                states,
                timestep=step,
                timesteps=args.steps,
            )
            policy_actions = outputs[-1].get(
                "mean_actions", outputs[0]
            )
            if args.zero_before_entry:
                policy_actions = torch.zeros_like(policy_actions)
            elapsed = (step - arm_step).clamp_min(0)
            candidates = torch.where(
                (elapsed < first_durations).unsqueeze(-1),
                candidates_first,
                candidates_second,
            )
            actions = torch.where(
                armed.unsqueeze(-1), candidates, policy_actions
            )
            armed_trace_mask = (
                armed
                & ~success
                & (armed_trace_length < armed_trace_capacity)
            )
            armed_trace_ids = torch.nonzero(
                armed_trace_mask, as_tuple=False
            ).squeeze(-1)
            if len(armed_trace_ids) > 0:
                armed_steps = armed_trace_length[armed_trace_ids]
                armed_observation_trace[
                    armed_trace_ids, armed_steps
                ] = observations[armed_trace_ids]
                armed_action_trace[
                    armed_trace_ids, armed_steps
                ] = actions[armed_trace_ids]
                armed_phase_trace[
                    armed_trace_ids, armed_steps
                ] = direct.chain_phase_ids[armed_trace_ids]
                armed_trace_length[armed_trace_ids] += 1
            trace_mask = (
                started
                & (hold_run > 0)
                & (hold_run <= hold_steps_required)
            )
            trace_ids = torch.nonzero(
                trace_mask, as_tuple=False
            ).squeeze(-1)
            if len(trace_ids) > 0:
                trace_steps = hold_run[trace_ids] - 1
                observation_trace[trace_ids, trace_steps] = observations[
                    trace_ids
                ]
                action_trace[trace_ids, trace_steps] = actions[trace_ids]
            observations, _, terminated, truncated, _ = wrapped.step(
                actions
            )
            states = wrapped.state()
            direct._compute_intermediate_values()
            done = (terminated | truncated).reshape(-1)
            post_eligible = strict_eligible(direct)
            if args.trace_output is not None:
                trace_id = min(max(args.trace_env_id, 0), args.num_envs - 1)
                middle_error = torch.sqrt(
                    torch.mean(
                        (
                            direct.hand_qpos[trace_id, 4:8]
                            - direct.target_hand_qpos[trace_id, 4:8]
                        ).square()
                    )
                )
                trace_rows.append(
                    {
                        "qpos": torch.cat(
                            (
                                direct.hand_qpos[trace_id],
                                direct.pen_pos[trace_id],
                                direct.pen_quat[trace_id],
                            )
                        ).cpu().numpy(),
                        "qvel": torch.cat(
                            (
                                direct.hand_qvel[trace_id],
                                direct.pen_linvel[trace_id],
                                direct.pen_angvel[trace_id],
                            )
                        ).cpu().numpy(),
                        "phase_id": int(
                            direct.chain_phase_ids[trace_id].item()
                        ),
                        "eligible": bool(post_eligible[trace_id].item()),
                        "contact_force_n": direct.contact_force_norm[
                            trace_id
                        ].cpu().numpy(),
                        "middle_rmse_rad": float(middle_error.item()),
                        "max_penetration_m": float(
                            direct.episode_max_penetration[trace_id].item()
                        ),
                    }
                )
            restarted_run = (
                started
                & post_eligible
                & (hold_run == 0)
                & ~done
            )
            if torch.any(restarted_run):
                entry_hand_qpos[restarted_run] = direct.hand_qpos[
                    restarted_run
                ]
                entry_hand_qvel[restarted_run] = direct.hand_qvel[
                    restarted_run
                ]
                entry_pen_root_state[restarted_run] = torch.cat(
                    (
                        direct.pen_pos,
                        direct.pen_quat,
                        direct.pen_linvel,
                        direct.pen_angvel,
                    ),
                    dim=-1,
                )[restarted_run]
                for destination, source in (
                    (entry_target_qpos, direct.policy_targets),
                    (entry_policy_action, direct.actions),
                    (entry_target_hand_qpos, direct.target_hand_qpos),
                    (
                        entry_closure_origin_hand,
                        direct.closure_origin_hand,
                    ),
                    (
                        entry_closure_brace_hand,
                        direct.closure_brace_hand,
                    ),
                    (
                        entry_closure_origin_pen_pos,
                        direct.closure_origin_pen_pos,
                    ),
                    (
                        entry_closure_origin_pen_quat,
                        direct.closure_origin_pen_quat,
                    ),
                    (entry_init_pen_pos, direct.init_pen_pos),
                    (entry_init_pen_axis, direct.init_pen_axis),
                    (
                        entry_max_penetration,
                        direct.episode_max_penetration,
                    ),
                    (entry_contact_force, direct.contact_force_norm),
                    (entry_chain_phase, direct.chain_phase_ids),
                ):
                    destination[restarted_run] = source[restarted_run]
            continued = started & post_eligible
            hold_run = torch.where(
                continued,
                hold_run + 1,
                torch.zeros_like(hold_run),
            )
            max_hold = torch.maximum(
                max_hold, hold_run
            )

            active_lost = (
                started
                & ~post_eligible
                & ~first_loss_recorded
                & ~done
            )
            if torch.any(active_lost):
                contacts = (
                    direct.contact_force_norm
                    > direct.cfg.p1_contact_force_n
                )
                closure_level = (
                    direct.chain_phase_ids
                    - int(direct.cfg.training_start_phase)
                )
                unloading = closure_level > int(
                    direct.cfg.thumb_wrap_stage_boundaries[-1]
                )
                required_contact = torch.where(
                    unloading,
                    contacts[:, :2].all(dim=-1),
                    contacts[:, :3].all(dim=-1),
                )
                target_axis = quat_apply(
                    direct.target_pen_quat, direct.pen_x_axis
                )
                axis_dot = torch.sum(
                    direct.pen_axis * target_axis, dim=-1
                ).clamp(-1.0, 1.0)
                axis_error = torch.acos(axis_dot.abs())
                speed = torch.linalg.norm(
                    direct.pen_angvel, dim=-1
                )
                first_loss_contact |= active_lost & ~required_contact
                first_loss_axis |= active_lost & (
                    axis_error
                    > torch.deg2rad(
                        torch.full_like(
                            axis_error,
                            direct.cfg.closure_pen_axis_tolerance_deg,
                        )
                    )
                )
                first_loss_speed |= active_lost & (
                    speed
                    > direct.cfg.thumb_wrap_handoff_max_angular_speed_rad_s
                )
                first_loss_recorded |= active_lost

            reached = (
                direct.latest_success_serial > initial_success_serial
            ) | (hold_run >= int(direct.cfg.final_hold_steps))
            newly_successful = reached & ~success
            success |= newly_successful
            success_step[newly_successful] = step
            if torch.any(newly_successful):
                terminal_qpos[newly_successful] = torch.cat(
                    (
                        direct.hand_qpos[newly_successful],
                        direct.pen_pos[newly_successful],
                        direct.pen_quat[newly_successful],
                    ),
                    dim=-1,
                )
                terminal_qvel[newly_successful] = torch.cat(
                    (
                        direct.hand_qvel[newly_successful],
                        direct.pen_linvel[newly_successful],
                        direct.pen_angvel[newly_successful],
                    ),
                    dim=-1,
                )
                for destination, source in (
                    (terminal_target_qpos, direct.policy_targets),
                    (terminal_policy_action, direct.actions),
                    (
                        terminal_target_hand_qpos,
                        direct.target_hand_qpos,
                    ),
                    (
                        terminal_closure_origin_hand,
                        direct.closure_origin_hand,
                    ),
                    (
                        terminal_closure_brace_hand,
                        direct.closure_brace_hand,
                    ),
                    (
                        terminal_closure_origin_pen_pos,
                        direct.closure_origin_pen_pos,
                    ),
                    (
                        terminal_closure_origin_pen_quat,
                        direct.closure_origin_pen_quat,
                    ),
                    (terminal_init_pen_pos, direct.init_pen_pos),
                    (terminal_init_pen_axis, direct.init_pen_axis),
                    (
                        terminal_max_penetration,
                        direct.episode_max_penetration,
                    ),
                    (
                        terminal_contact_force,
                        direct.contact_force_norm,
                    ),
                    (
                        terminal_chain_phase,
                        direct.chain_phase_ids,
                    ),
                ):
                    destination[newly_successful] = source[
                        newly_successful
                    ]
            failed |= done & ~success

    rank_score = (
        1000.0 * success.float()
        + max_hold.float()
        - 0.05 * (
            torch.linalg.norm(candidates_first, dim=-1)
            + torch.linalg.norm(candidates_second, dim=-1)
        )
    )
    eligible_ids = torch.nonzero(
        started, as_tuple=False
    ).squeeze(-1)
    order = torch.argsort(
        rank_score[eligible_ids], descending=True
    )
    selected = eligible_ids[order[: args.max_selected]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        env_ids=selected.cpu().numpy().astype(np.int64),
        candidate_action=candidates_first[
            selected
        ].cpu().numpy().astype(np.float32),
        candidate_action_first=candidates_first[
            selected
        ].cpu().numpy().astype(np.float32),
        candidate_action_second=candidates_second[
            selected
        ].cpu().numpy().astype(np.float32),
        observation_trace=observation_trace[
            selected
        ].cpu().numpy().astype(np.float32),
        action_trace=action_trace[
            selected
        ].cpu().numpy().astype(np.float32),
        armed_observation_trace=armed_observation_trace[
            selected
        ].cpu().numpy().astype(np.float32),
        armed_action_trace=armed_action_trace[
            selected
        ].cpu().numpy().astype(np.float32),
        armed_phase_trace=armed_phase_trace[
            selected
        ].cpu().numpy().astype(np.int64),
        armed_trace_length=armed_trace_length[
            selected
        ].cpu().numpy().astype(np.int64),
        entry_hand_qpos=entry_hand_qpos[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_hand_qvel=entry_hand_qvel[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_pen_root_state=entry_pen_root_state[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_target_qpos=entry_target_qpos[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_policy_action=entry_policy_action[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_target_hand_qpos=entry_target_hand_qpos[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_closure_origin_hand=entry_closure_origin_hand[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_closure_brace_hand=entry_closure_brace_hand[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_closure_origin_pen_pos=entry_closure_origin_pen_pos[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_closure_origin_pen_quat=entry_closure_origin_pen_quat[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_init_pen_pos=entry_init_pen_pos[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_init_pen_axis=entry_init_pen_axis[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_max_penetration_m=entry_max_penetration[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_contact_force_n=entry_contact_force[
            selected
        ].cpu().numpy().astype(np.float32),
        entry_chain_phase_id=entry_chain_phase[
            selected
        ].cpu().numpy().astype(np.int64),
        terminal_qpos=terminal_qpos[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_qvel=terminal_qvel[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_target_qpos=terminal_target_qpos[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_policy_action=terminal_policy_action[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_target_hand_qpos=terminal_target_hand_qpos[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_closure_origin_hand=terminal_closure_origin_hand[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_closure_brace_hand=terminal_closure_brace_hand[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_closure_origin_pen_pos=terminal_closure_origin_pen_pos[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_closure_origin_pen_quat=terminal_closure_origin_pen_quat[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_init_pen_pos=terminal_init_pen_pos[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_init_pen_axis=terminal_init_pen_axis[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_max_penetration_m=terminal_max_penetration[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_contact_force_n=terminal_contact_force[
            selected
        ].cpu().numpy().astype(np.float32),
        terminal_chain_phase_id=terminal_chain_phase[
            selected
        ].cpu().numpy().astype(np.int64),
        first_duration=first_durations[
            selected
        ].cpu().numpy().astype(np.int64),
        searched_first_duration=first_durations[
            selected
        ].cpu().numpy().astype(np.int64),
        second_duration=np.full(
            len(selected),
            max(args.steps - args.first_duration, 1),
            dtype=np.int64,
        ),
        selection_kind=np.where(
            success[selected].cpu().numpy(),
            "success",
            "near_hold",
        ),
        best_brake_score=max_hold[
            selected
        ].cpu().numpy().astype(np.float32),
        source_bank_id=source_bank_ids[
            selected
        ].cpu().numpy().astype(np.int64),
        max_hold_steps=max_hold[
            selected
        ].cpu().numpy().astype(np.int64),
        success=success[selected].cpu().numpy(),
        start_step=start_step[
            selected
        ].cpu().numpy().astype(np.int64),
        arm_step=arm_step[
            selected
        ].cpu().numpy().astype(np.int64),
        success_step=success_step[
            selected
        ].cpu().numpy().astype(np.int64),
        seed=np.asarray(args.seed, dtype=np.int64),
    )
    summary = {
        "task": args.task,
        "checkpoint": (
            str(args.checkpoint.resolve())
            if args.checkpoint is not None
            else None
        ),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "action_scale": args.action_scale,
        "action_authority": args.action_authority,
        "post_entry_action_authority": (
            args.post_entry_action_authority
        ),
        "parent_npz": (
            str(args.parent_npz.resolve())
            if args.parent_npz is not None
            else None
        ),
        "parent_row": args.parent_row,
        "parent_jitter": args.parent_jitter,
        "second_jitter": args.second_jitter,
        "parent_repeat_fraction": args.parent_repeat_fraction,
        "first_duration": args.first_duration,
        "search_first_duration": args.search_first_duration,
        "fixed_bank_id": args.fixed_bank_id,
        "target_phase": args.target_phase,
        "candidate_start_phase": (
            args.target_phase
            if args.candidate_start_phase < 0
            else args.candidate_start_phase
        ),
        "zero_before_entry": args.zero_before_entry,
        "zero_reset_noise": args.zero_reset_noise,
        "zero_candidate_fraction": args.zero_candidate_fraction,
        "started": int(started.sum().item()),
        "successes": int(success.sum().item()),
        "success_rate_all": float(success.float().mean().item()),
        "success_rate_started": float(
            success.sum().item() / max(started.sum().item(), 1)
        ),
        "best_hold_steps": int(max_hold.max().item()),
        "best_hold_ms": float(
            max_hold.max().item()
            * direct.cfg.decimation
            * direct.cfg.sim.dt
            * 1000.0
        ),
        "first_loss_counts": {
            "triad_contact": int(first_loss_contact.sum().item()),
            "axis": int(first_loss_axis.sum().item()),
            "angular_speed": int(first_loss_speed.sum().item()),
        },
        "selected": int(selected.numel()),
        "output": str(args.output.resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.trace_output is not None and trace_rows:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_output,
            qpos=np.stack([row["qpos"] for row in trace_rows]),
            qvel=np.stack([row["qvel"] for row in trace_rows]),
            phase_ids=np.asarray(
                [row["phase_id"] for row in trace_rows], dtype=np.int64
            ),
            eligible=np.asarray(
                [row["eligible"] for row in trace_rows], dtype=np.bool_
            ),
            contact_force_n=np.stack(
                [row["contact_force_n"] for row in trace_rows]
            ),
            middle_rmse_rad=np.asarray(
                [row["middle_rmse_rad"] for row in trace_rows],
                dtype=np.float32,
            ),
            max_penetration_m=np.asarray(
                [row["max_penetration_m"] for row in trace_rows],
                dtype=np.float32,
            ),
            control_dt_s=np.asarray(
                direct.cfg.decimation * direct.cfg.sim.dt,
                dtype=np.float64,
            ),
            trajectory_kind=np.asarray(
                "successful_episode"
                if bool(success[args.trace_env_id].item())
                else "best_incomplete_phase18"
            ),
            deepest_phase=np.asarray(
                max(row["phase_id"] for row in trace_rows), dtype=np.int64
            ),
            state_reset=np.asarray("phase18_entry_cache"),
        )
    print(
        "PHASE15_HOLD_SEARCH="
        + json.dumps(summary, ensure_ascii=False)
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
