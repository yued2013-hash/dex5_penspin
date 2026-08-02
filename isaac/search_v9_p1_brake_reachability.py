from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Search short index/middle residuals that recenter the pen axis at "
        "the phase-16 P1 brake stage."
    )
)
parser.add_argument(
    "--task",
    default=(
        "Dex5-V9P4-P5-P1-ContinuousClosureCycleAlignedDenseTorque"
        "BrakeLateTrain-Direct-v0"
    ),
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=260)
parser.add_argument("--trace_steps", type=int, default=72)
parser.add_argument("--seed", type=int, default=2530)
parser.add_argument("--max_selected", type=int, default=512)
parser.add_argument("--offset_scale", type=float, default=0.85)
parser.add_argument("--include-thumb", action="store_true")
parser.add_argument("--parent", type=Path)
parser.add_argument("--parent-jitter", type=float, default=0.12)
parser.add_argument("--parent-action-scale", type=float, default=1.0)
parser.add_argument("--first-duration-max", type=int, default=20)
parser.add_argument("--second-duration-max", type=int, default=24)
parser.add_argument("--target-servo-search", action="store_true")
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


def _make_candidate_offsets(
    num_envs: int,
    device: str,
    seed: int,
    scale: float,
    include_thumb: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    first = torch.zeros((num_envs, 20), device=device)
    second = torch.zeros_like(first)

    active_joints = [*range(8)]
    if include_thumb:
        active_joints.extend(range(16, 20))
    active = torch.as_tensor(active_joints, device=device)
    dimensions = len(active_joints)
    random_first = torch.randn(
        (num_envs, dimensions), generator=generator, device=device
    )
    random_second = torch.randn(
        (num_envs, dimensions), generator=generator, device=device
    )
    if include_thumb:
        random_first[:, 8:] *= 0.65
        random_second[:, 8:] *= 0.65
    random_first /= torch.linalg.norm(
        random_first, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    random_second /= torch.linalg.norm(
        random_second, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    amplitude_first = (
        0.15
        + 0.85
        * torch.rand(
            (num_envs, 1), generator=generator, device=device
        )
    )
    amplitude_second = (
        0.10
        + 0.90
        * torch.rand(
            (num_envs, 1), generator=generator, device=device
        )
    )
    first[:, active] = scale * amplitude_first * random_first
    second[:, active] = scale * amplitude_second * random_second

    # Reserve deterministic rows for zero action and signed joint bases.
    first[0] = 0.0
    second[0] = 0.0
    basis_rows = min(num_envs - 1, 2 * dimensions)
    for row in range(basis_rows):
        joint = active_joints[(row // 2) % dimensions]
        sign = -1.0 if row % 2 == 0 else 1.0
        first[row + 1] = 0.0
        second[row + 1] = 0.0
        first[row + 1, joint] = sign * scale

    first_duration = torch.randint(
        4,
        max(args.first_duration_max, 4) + 1,
        (num_envs,),
        generator=generator,
        device=device,
    )
    second_duration = torch.randint(
        4,
        max(args.second_duration_max, 4) + 1,
        (num_envs,),
        generator=generator,
        device=device,
    )
    return first, second, first_duration, second_duration


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.bundle_path = str(args.bundle.resolve())
    if args.include_thumb:
        env_cfg.index_middle_only_residual = False
        env_cfg.index_middle_thumb_only_residual = True
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
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped, agent_cfg)
    runner.agent.load(str(args.checkpoint.resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    observations, _ = wrapped.reset()
    states = wrapped.state()
    reset_hand_qpos = direct.hand_qpos.clone()
    reset_hand_qvel = direct.hand_qvel.clone()
    reset_pen_pos = direct.pen_pos.clone()
    reset_pen_quat = direct.pen_quat.clone()
    reset_pen_linvel = direct.pen_linvel.clone()
    reset_pen_angvel = direct.pen_angvel.clone()
    reset_policy_targets = direct.policy_targets.clone()
    reset_target_hand = direct.target_hand_qpos.clone()
    reset_target_pen_pos = direct.target_pen_pos.clone()
    reset_target_pen_quat = direct.target_pen_quat.clone()
    reset_closure_origin_hand = direct.closure_origin_hand.clone()
    reset_closure_origin_pen_pos = direct.closure_origin_pen_pos.clone()
    reset_closure_origin_pen_quat = direct.closure_origin_pen_quat.clone()
    reset_init_pen_pos = direct.init_pen_pos.clone()
    reset_init_pen_axis = direct.init_pen_axis.clone()
    reset_max_penetration = direct.episode_max_penetration.clone()
    reset_contact_force = direct.contact_force_norm.clone()
    reset_phase = direct.chain_phase_ids.clone()
    reset_bank_ids = direct.current_bank_ids.clone()
    (
        first_offset,
        second_offset,
        first_duration,
        second_duration,
    ) = _make_candidate_offsets(
        args.num_envs,
        direct.device,
        args.seed,
        args.offset_scale,
        args.include_thumb,
    )
    if args.target_servo_search:
        first_offset = first_offset.abs()
        second_offset = second_offset.abs()
    if args.parent is not None:
        with np.load(args.parent.resolve()) as parent:
            parent_first = torch.as_tensor(
                parent["candidate_offset_first"],
                dtype=torch.float32,
                device=direct.device,
            )
            parent_second = torch.as_tensor(
                parent["candidate_offset_second"],
                dtype=torch.float32,
                device=direct.device,
            )
            parent_first_duration = torch.as_tensor(
                parent["first_duration"],
                dtype=torch.long,
                device=direct.device,
            )
            parent_second_duration = torch.as_tensor(
                parent["second_duration"],
                dtype=torch.long,
                device=direct.device,
            )
        parent_ids = direct.current_bank_ids % len(parent_first)
        generator = torch.Generator(device=direct.device).manual_seed(
            args.seed + 1701
        )
        active_mask = torch.zeros((1, 20), device=direct.device)
        active_mask[:, :8] = 1.0
        if args.include_thumb:
            active_mask[:, 16:20] = 0.65
        first_offset = (
            args.parent_action_scale * parent_first[parent_ids]
            + args.parent_jitter
            * torch.randn(
                (args.num_envs, 20),
                generator=generator,
                device=direct.device,
            )
            * active_mask
        ).clamp(-1.5, 1.5)
        second_offset = (
            args.parent_action_scale * parent_second[parent_ids]
            + args.parent_jitter
            * torch.randn(
                (args.num_envs, 20),
                generator=generator,
                device=direct.device,
            )
            * active_mask
        ).clamp(-1.5, 1.5)
        duration_jitter = torch.randint(
            -3,
            4,
            (args.num_envs,),
            generator=generator,
            device=direct.device,
        )
        first_duration = (
            parent_first_duration[parent_ids] + duration_jitter
        ).clamp(2, max(args.first_duration_max, 4) + 8)
        second_duration = (
            parent_second_duration[parent_ids] - duration_jitter
        ).clamp(2, max(args.second_duration_max, 4) + 8)

    search_age = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    failed = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    reached_brake = torch.zeros_like(failed)
    reached_final = torch.zeros_like(failed)
    strict_hold_success = torch.zeros_like(failed)
    first_final_step = torch.full_like(search_age, -1)
    min_axis_error = torch.full(
        (args.num_envs,), torch.inf, device=direct.device
    )
    min_position_error = torch.full_like(min_axis_error, torch.inf)
    min_triad_rmse = torch.full_like(min_axis_error, torch.inf)
    min_contact_triad_rmse = torch.full_like(min_axis_error, torch.inf)
    hold_run = torch.zeros_like(search_age)
    max_hold_run = torch.zeros_like(search_age)
    max_penetration = torch.zeros_like(min_axis_error)
    max_force = torch.zeros_like(min_axis_error)
    best_score = torch.full_like(min_axis_error, -torch.inf)
    best_hand_qpos = torch.zeros_like(direct.hand_qpos)
    best_hand_qvel = torch.zeros_like(direct.hand_qvel)
    best_pen_pos = torch.zeros_like(direct.pen_pos)
    best_pen_quat = torch.zeros_like(direct.pen_quat)
    best_pen_linvel = torch.zeros_like(direct.pen_linvel)
    best_pen_angvel = torch.zeros_like(direct.pen_angvel)
    best_policy_targets = torch.zeros_like(direct.policy_targets)
    best_contact_force = torch.zeros_like(direct.contact_force_norm)
    best_action = torch.zeros_like(direct.actions)
    best_phase = torch.zeros_like(direct.chain_phase_ids)
    contact_best_hand_qpos = torch.zeros_like(direct.hand_qpos)
    contact_best_hand_qvel = torch.zeros_like(direct.hand_qvel)
    contact_best_pen_pos = torch.zeros_like(direct.pen_pos)
    contact_best_pen_quat = torch.zeros_like(direct.pen_quat)
    contact_best_pen_linvel = torch.zeros_like(direct.pen_linvel)
    contact_best_pen_angvel = torch.zeros_like(direct.pen_angvel)
    contact_best_policy_targets = torch.zeros_like(direct.policy_targets)

    observation_trace = torch.zeros(
        (
            args.num_envs,
            args.trace_steps,
            observations.shape[-1],
        ),
        dtype=observations.dtype,
        device=direct.device,
    )
    action_trace = torch.zeros(
        (args.num_envs, args.trace_steps, 20),
        dtype=observations.dtype,
        device=direct.device,
    )
    phase_trace = torch.zeros(
        (args.num_envs, args.trace_steps),
        dtype=torch.long,
        device=direct.device,
    )
    trace_length = torch.zeros_like(search_age)

    rows = torch.arange(args.num_envs, device=direct.device)
    for step in range(args.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(
                observations,
                states,
                timestep=step,
                timesteps=args.steps,
            )
            policy_actions = outputs[-1].get("mean_actions", outputs[0])
            phase_before = direct.chain_phase_ids.clone()
            starts = (phase_before >= 16) & (search_age < 0) & ~failed
            search_age = torch.where(
                starts, torch.zeros_like(search_age), search_age
            )
            active = search_age >= 0
            use_first = active & (search_age < first_duration)
            use_second = (
                active
                & ~use_first
                & (
                    search_age
                    < first_duration + second_duration
                )
            )
            actions = policy_actions.clone()
            first_delta = first_offset
            second_delta = second_offset
            if args.target_servo_search:
                mask = direct._stage_action_mask()
                increment = (
                    0.25 * direct.cfg.action_scale * mask
                ).clamp_min(1.0e-6)
                target_direction = (
                    direct.target_hand_qpos - direct.policy_targets
                ) / increment
                target_direction = torch.where(
                    mask > 0.0,
                    target_direction,
                    torch.zeros_like(target_direction),
                )
                first_delta = first_offset * target_direction
                second_delta = second_offset * target_direction
            actions = torch.where(
                use_first.unsqueeze(-1),
                (actions + first_delta).clamp(-1.0, 1.0),
                actions,
            )
            actions = torch.where(
                use_second.unsqueeze(-1),
                (actions + second_delta).clamp(-1.0, 1.0),
                actions,
            )

            record = active & (trace_length < args.trace_steps)
            if torch.any(record):
                record_rows = rows[record]
                record_steps = trace_length[record]
                observation_trace[record_rows, record_steps] = observations[
                    record
                ]
                action_trace[record_rows, record_steps] = actions[record]
                phase_trace[record_rows, record_steps] = phase_before[record]
                trace_length[record] += 1

            observations, _, terminated, truncated, _ = wrapped.step(actions)
            states = wrapped.state()
            direct._compute_intermediate_values()
            done = (terminated | truncated).reshape(-1)
            reset_success = getattr(
                direct,
                "last_reset_success",
                torch.zeros_like(done),
            )
            strict_hold_success |= done & reset_success
            failed |= done & ~reset_success & ~strict_hold_success
            reached_brake |= direct.chain_phase_ids >= 16
            newly_final = (
                (direct.chain_phase_ids >= 17)
                & ~reached_final
                & ~failed
            )
            first_final_step[newly_final] = step
            reached_final |= direct.chain_phase_ids >= 17

            target_axis = quat_apply(
                direct.target_pen_quat, direct.pen_x_axis
            )
            axis_dot = torch.sum(
                direct.pen_axis * target_axis, dim=-1
            ).clamp(-1.0, 1.0)
            axis_error = torch.acos(axis_dot.abs())
            position_error = torch.linalg.norm(
                direct.pen_pos - direct.target_pen_pos, dim=-1
            )
            contacts = (
                direct.contact_force_norm
                > direct.cfg.p1_contact_force_n
            )
            triad = contacts[:, :3].all(dim=-1)
            triad_error = torch.cat(
                (
                    direct.hand_qpos[:, 0:8]
                    - direct.target_hand_qpos[:, 0:8],
                    direct.hand_qpos[:, 16:20]
                    - direct.target_hand_qpos[:, 16:20],
                ),
                dim=-1,
            )
            triad_rmse = torch.sqrt(
                torch.mean(triad_error.square(), dim=-1)
            )
            ring_rmse = torch.sqrt(
                torch.mean(
                    (
                        direct.hand_qpos[:, 8:12]
                        - direct.target_hand_qpos[:, 8:12]
                    ).square(),
                    dim=-1,
                )
            )
            speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
            penetration = direct.episode_max_penetration
            force = direct.contact_force_norm.max(dim=-1).values
            valid = (
                (direct.chain_phase_ids >= 16)
                & triad
                & (
                    position_error
                    <= direct.cfg.closure_pen_position_tolerance_m
                )
                & (
                    penetration
                    <= direct.cfg.penetration_threshold
                )
                & (force <= direct.cfg.p1_max_force_n)
                & ~failed
            )
            strict_eligible = (
                valid
                & (
                    axis_error
                    <= torch.deg2rad(
                        torch.full_like(
                            axis_error,
                            direct.cfg.closure_pen_axis_tolerance_deg,
                        )
                    )
                )
                & (speed <= direct.cfg.p5_max_angular_speed_rad_s)
                & (
                    triad_rmse
                    <= direct.cfg.closure_stage_rmse_thresholds[-1]
                )
                & (ring_rmse <= direct.cfg.p1_ring_pose_rmse_rad)
            )
            hold_run = torch.where(
                strict_eligible,
                hold_run + 1,
                torch.zeros_like(hold_run),
            )
            max_hold_run = torch.maximum(max_hold_run, hold_run)
            final_active = direct.chain_phase_ids >= 17
            min_triad_rmse = torch.where(
                final_active,
                torch.minimum(min_triad_rmse, triad_rmse),
                min_triad_rmse,
            )
            contact_rmse_improved = (
                final_active
                & triad
                & (triad_rmse < min_contact_triad_rmse)
            )
            contact_best_hand_qpos[contact_rmse_improved] = (
                direct.hand_qpos[contact_rmse_improved]
            )
            contact_best_hand_qvel[contact_rmse_improved] = (
                direct.hand_qvel[contact_rmse_improved]
            )
            contact_best_pen_pos[contact_rmse_improved] = (
                direct.pen_pos[contact_rmse_improved]
            )
            contact_best_pen_quat[contact_rmse_improved] = (
                direct.pen_quat[contact_rmse_improved]
            )
            contact_best_pen_linvel[contact_rmse_improved] = (
                direct.pen_linvel[contact_rmse_improved]
            )
            contact_best_pen_angvel[contact_rmse_improved] = (
                direct.pen_angvel[contact_rmse_improved]
            )
            contact_best_policy_targets[contact_rmse_improved] = (
                direct.policy_targets[contact_rmse_improved]
            )
            min_contact_triad_rmse = torch.where(
                final_active & triad,
                torch.minimum(min_contact_triad_rmse, triad_rmse),
                min_contact_triad_rmse,
            )
            score = (
                -torch.rad2deg(axis_error)
                - 0.25 * position_error * 1000.0
                - 0.15 * speed
                - 45.0 * triad_rmse
                - 12.0 * ring_rmse
                + 8.0 * triad.float()
                + 2.0 * max_hold_run.float()
                + 20.0 * (direct.chain_phase_ids >= 17).float()
            )
            improved = valid & (score > best_score)
            best_score = torch.where(improved, score, best_score)
            best_hand_qpos[improved] = direct.hand_qpos[improved]
            best_hand_qvel[improved] = direct.hand_qvel[improved]
            best_pen_pos[improved] = direct.pen_pos[improved]
            best_pen_quat[improved] = direct.pen_quat[improved]
            best_pen_linvel[improved] = direct.pen_linvel[improved]
            best_pen_angvel[improved] = direct.pen_angvel[improved]
            best_policy_targets[improved] = direct.policy_targets[improved]
            best_contact_force[improved] = direct.contact_force_norm[improved]
            best_action[improved] = actions[improved]
            best_phase[improved] = direct.chain_phase_ids[improved]
            min_axis_error = torch.where(
                valid,
                torch.minimum(min_axis_error, axis_error),
                min_axis_error,
            )
            min_position_error = torch.where(
                valid,
                torch.minimum(min_position_error, position_error),
                min_position_error,
            )
            max_penetration = torch.maximum(max_penetration, penetration)
            max_force = torch.maximum(max_force, force)
            search_age = torch.where(
                active, search_age + 1, search_age
            )

    successful = strict_hold_success & ~failed & (
        max_penetration <= direct.cfg.penetration_threshold
    )
    successful_ids = torch.nonzero(
        successful, as_tuple=False
    ).squeeze(-1)
    finite_ids = torch.nonzero(
        torch.isfinite(best_score), as_tuple=False
    ).squeeze(-1)
    if successful_ids.numel() > 0:
        order = torch.argsort(best_score[successful_ids], descending=True)
        selected = successful_ids[order[: args.max_selected]]
    elif finite_ids.numel() > 0:
        order = torch.argsort(best_score[finite_ids], descending=True)
        selected = finite_ids[order[: args.max_selected]]
    else:
        selected = torch.arange(
            min(args.max_selected, args.num_envs), device=direct.device
        )

    output = args.output.resolve()
    summary_path = args.summary.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    selected_cpu = selected.cpu()
    np.savez_compressed(
        output,
        env_ids=selected_cpu.numpy(),
        original_source_bank_ids=reset_bank_ids[selected]
        .cpu()
        .numpy(),
        observation=observation_trace[selected].cpu().numpy(),
        action=action_trace[selected].cpu().numpy(),
        stage_id=phase_trace[selected].cpu().numpy(),
        lengths=trace_length[selected].cpu().numpy(),
        candidate_offset_first=first_offset[selected].cpu().numpy(),
        candidate_offset_second=second_offset[selected].cpu().numpy(),
        first_duration=first_duration[selected].cpu().numpy(),
        second_duration=second_duration[selected].cpu().numpy(),
        reached_final=reached_final[selected].cpu().numpy(),
        first_final_step=first_final_step[selected].cpu().numpy(),
        min_axis_error_rad=min_axis_error[selected].cpu().numpy(),
        min_position_error_m=min_position_error[selected].cpu().numpy(),
        max_penetration_m=max_penetration[selected].cpu().numpy(),
        max_force_n=max_force[selected].cpu().numpy(),
        min_triad_rmse_rad=min_triad_rmse[selected].cpu().numpy(),
        min_contact_triad_rmse_rad=min_contact_triad_rmse[selected]
        .cpu()
        .numpy(),
        max_strict_hold_steps=max_hold_run[selected].cpu().numpy(),
        source_qpos=torch.cat(
            (
                reset_hand_qpos[selected],
                reset_pen_pos[selected],
                reset_pen_quat[selected],
            ),
            dim=-1,
        )
        .cpu()
        .numpy(),
        source_qvel=torch.cat(
            (
                reset_hand_qvel[selected],
                reset_pen_linvel[selected],
                reset_pen_angvel[selected],
            ),
            dim=-1,
        )
        .cpu()
        .numpy(),
        source_target_qpos=reset_policy_targets[selected].cpu().numpy(),
        source_policy_action=action_trace[selected, 0].cpu().numpy(),
        source_target_hand_qpos=reset_target_hand[selected]
        .cpu()
        .numpy(),
        source_closure_origin_hand=reset_closure_origin_hand[selected]
        .cpu()
        .numpy(),
        source_closure_origin_pen_pos=reset_closure_origin_pen_pos[selected]
        .cpu()
        .numpy(),
        source_closure_origin_pen_quat=reset_closure_origin_pen_quat[selected]
        .cpu()
        .numpy(),
        source_init_pen_pos=reset_init_pen_pos[selected].cpu().numpy(),
        source_init_pen_axis=reset_init_pen_axis[selected].cpu().numpy(),
        source_contact_force_n=reset_contact_force[selected].cpu().numpy(),
        source_max_penetration_m=reset_max_penetration[selected]
        .cpu()
        .numpy(),
        source_chain_phase_id=reset_phase[selected].cpu().numpy(),
        target_qpos=torch.cat(
            (
                reset_target_hand[selected],
                reset_target_pen_pos[selected],
                reset_target_pen_quat[selected],
            ),
            dim=-1,
        )
        .cpu()
        .numpy(),
        target_qvel=np.zeros((selected.numel(), 26), dtype=np.float32),
        phase_ids=np.zeros(selected.numel(), dtype=np.int64),
        source_teacher_actions=action_trace[selected].cpu().numpy(),
        best_qpos=torch.cat(
            (
                best_hand_qpos[selected],
                best_pen_pos[selected],
                best_pen_quat[selected],
            ),
            dim=-1,
        )
        .cpu()
        .numpy(),
        best_qvel=torch.cat(
            (
                best_hand_qvel[selected],
                best_pen_linvel[selected],
                best_pen_angvel[selected],
            ),
            dim=-1,
        )
        .cpu()
        .numpy(),
        best_target_qpos=best_policy_targets[selected].cpu().numpy(),
        best_policy_action=best_action[selected].cpu().numpy(),
        best_chain_phase_id=best_phase[selected].cpu().numpy(),
        contact_best_qpos=torch.cat(
            (
                contact_best_hand_qpos[selected],
                contact_best_pen_pos[selected],
                contact_best_pen_quat[selected],
            ),
            dim=-1,
        )
        .cpu()
        .numpy(),
        contact_best_qvel=torch.cat(
            (
                contact_best_hand_qvel[selected],
                contact_best_pen_linvel[selected],
                contact_best_pen_angvel[selected],
            ),
            dim=-1,
        )
        .cpu()
        .numpy(),
        contact_best_target_qpos=contact_best_policy_targets[selected]
        .cpu()
        .numpy(),
    )
    finite_axis = min_axis_error[torch.isfinite(min_axis_error)]
    summary = {
        "task": args.task,
        "checkpoint": str(args.checkpoint.resolve()),
        "bundle": str(args.bundle.resolve()),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "include_thumb": args.include_thumb,
        "parent": (
            str(args.parent.resolve()) if args.parent is not None else None
        ),
        "target_servo_search": args.target_servo_search,
        "brake_reached": int(reached_brake.sum().item()),
        "final_phase_reached": int(reached_final.sum().item()),
        "final_phase_rate": float(reached_final.float().mean().item()),
        "strict_hold_successes": int(strict_hold_success.sum().item()),
        "strict_hold_success_rate": float(
            strict_hold_success.float().mean().item()
        ),
        "failed": int(failed.sum().item()),
        "penetration_failures": int(
            (max_penetration > direct.cfg.penetration_threshold).sum().item()
        ),
        "best_axis_error_deg": (
            float(torch.rad2deg(finite_axis.min()).item())
            if finite_axis.numel() > 0
            else None
        ),
        "axis_below_10deg": int(
            (
                torch.isfinite(min_axis_error)
                & (torch.rad2deg(min_axis_error) <= 10.0)
            )
            .sum()
            .item()
        ),
        "best_triad_rmse_rad": (
            float(min_triad_rmse[torch.isfinite(min_triad_rmse)].min().item())
            if torch.any(torch.isfinite(min_triad_rmse))
            else None
        ),
        "best_contact_triad_rmse_rad": (
            float(
                min_contact_triad_rmse[
                    torch.isfinite(min_contact_triad_rmse)
                ]
                .min()
                .item()
            )
            if torch.any(torch.isfinite(min_contact_triad_rmse))
            else None
        ),
        "best_strict_hold_steps": int(max_hold_run.max().item()),
        "selected_rows": int(selected.numel()),
        "output": str(output),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    wrapped.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
