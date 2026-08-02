from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Search thumb support/wrap transitions while index and middle retain "
        "the primary pen clamp."
    )
)
parser.add_argument(
    "--task",
    default=(
        "Dex5-V9P4-P5-P1-ContinuousClosureThumbUnderPhase7LateTrain-"
        "Direct-v0"
    ),
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=140)
parser.add_argument("--seed", type=int, default=2600)
parser.add_argument("--target_phase", type=int, default=8)
parser.add_argument("--final_success", action="store_true")
parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--index_action_scale", type=float, default=0.0)
parser.add_argument("--middle_action_scale", type=float, default=0.0)
parser.add_argument("--ring_action_scale", type=float, default=0.0)
parser.add_argument("--first_duration_max", type=int, default=32)
parser.add_argument("--second_duration_max", type=int, default=56)
parser.add_argument("--max_selected", type=int, default=512)
parser.add_argument("--parent", type=Path)
parser.add_argument("--parent_jitter", type=float, default=0.12)
parser.add_argument("--parent-cycle", action="store_true")
parser.add_argument("--parent-success-only", action="store_true")
parser.add_argument("--parent-duration-jitter", type=int, default=3)
parser.add_argument(
    "--contact-feedback",
    action="store_true",
    help=(
        "Switch from approach to support action using live index/middle/thumb "
        "contacts instead of fixed action durations."
    ),
)
parser.add_argument(
    "--hold-feedback-only",
    action="store_true",
    help=(
        "Preserve the original timed approach, then latch the support action "
        "after the first strict eligible hold step."
    ),
)
parser.add_argument(
    "--eligible-hold-action-scale",
    type=float,
    default=0.0,
    help=(
        "Residual action scale after the strict eligible hold begins. The "
        "environment target servo remains active."
    ),
)
parser.add_argument("--eligible-hold-action-scale-min", type=float)
parser.add_argument("--eligible-hold-action-scale-max", type=float)
parser.add_argument("--feedback-support-min-steps", type=int, default=6)
parser.add_argument("--feedback-loss-patience", type=int, default=4)
parser.add_argument(
    "--min-near-hold",
    type=int,
    default=0,
    help=(
        "Also export candidates whose best consecutive eligible hold reaches "
        "this many steps without advancing. Zero keeps success-only output."
    ),
)
parser.add_argument("--trajectory_output", type=Path)
parser.add_argument("--teacher-dataset-output", type=Path)
parser.add_argument("--bundle", type=Path)
parser.add_argument(
    "--disable-scripted-teacher",
    action="store_true",
    help="Search direct residual actions without the cached scripted teacher.",
)
parser.add_argument(
    "--disable-brace-teacher",
    action="store_true",
    help="Allow candidate index/middle actions instead of the frozen brace policy.",
)
parser.add_argument(
    "--trainee-policy-scale",
    type=float,
    help="Override the environment residual action scale during search.",
)
parser.add_argument(
    "--scripted-residual-scale",
    type=float,
    help="Override residual authority around an enabled scripted teacher.",
)
parser.add_argument(
    "--preserve-parent-pen-reference",
    action="store_true",
    help=(
        "Keep the original same-episode pen target/reference when exporting "
        "a later-phase cache instead of rebasing it at the success snapshot."
    ),
)
parser.add_argument(
    "--restore-cached-phase-hold",
    action="store_true",
    help=(
        "Restore best_hold_steps from a near-success curriculum cache. "
        "Use only for local continuation training; no-reset evaluation "
        "accumulates hold progress naturally."
    ),
)
parser.add_argument(
    "--continue-bundle-action",
    action="store_true",
    help=(
        "Initialize both candidate segments from the cached policy action "
        "so a local curriculum reset preserves command continuity."
    ),
)
parser.add_argument(
    "--bundle-action-jitter",
    type=float,
    default=0.0,
    help="Gaussian jitter applied when continuing the cached policy action.",
)
parser.add_argument(
    "--export-braking-near",
    action="store_true",
    help=(
        "Export physically valid low-axis-rate near states even when the "
        "formal phase hold has not completed."
    ),
)
parser.add_argument(
    "--max-braking-axis-error-deg", type=float, default=10.0
)
parser.add_argument(
    "--max-braking-position-error-mm", type=float, default=10.0
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def make_candidates(
    count: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    first = torch.zeros((count, 20), device=device)
    second = torch.zeros_like(first)
    first[:, 16:20] = args.action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    second[:, 16:20] = args.action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    first[:, 0:4] = args.index_action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    second[:, 0:4] = args.index_action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    first[:, 4:8] = args.middle_action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    second[:, 4:8] = args.middle_action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    first[:, 8:12] = args.ring_action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    second[:, 8:12] = args.ring_action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    first[0].zero_()
    second[0].zero_()
    for row in range(min(32, count - 1)):
        joint = 16 + (row // 8)
        magnitude = (0.25, 0.50, 0.75, 1.0)[(row // 2) % 4]
        sign = -1.0 if row % 2 == 0 else 1.0
        first[row + 1].zero_()
        second[row + 1].zero_()
        first[row + 1, joint] = sign * args.action_scale * magnitude
        second[row + 1, joint] = sign * args.action_scale * magnitude
    first_duration = torch.randint(
        6,
        args.first_duration_max + 1,
        (count,),
        generator=generator,
        device=device,
    )
    second_duration = torch.randint(
        8,
        args.second_duration_max + 1,
        (count,),
        generator=generator,
        device=device,
    )
    return first, second, first_duration, second_duration


def main() -> None:
    cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs
    )
    cfg.seed = args.seed
    if args.bundle is not None:
        cfg.bundle_path = str(resolve(args.bundle))
    if args.disable_scripted_teacher:
        cfg.scripted_thumb_teacher_path = ""
    if args.disable_brace_teacher:
        cfg.brace_teacher_checkpoint = ""
    if args.trainee_policy_scale is not None:
        cfg.trainee_policy_scale = float(args.trainee_policy_scale)
    if args.scripted_residual_scale is not None:
        cfg.scripted_thumb_teacher_residual_scale = float(
            args.scripted_residual_scale
        )
    if args.restore_cached_phase_hold:
        cfg.restore_cached_phase_hold = True
    cfg.episode_length_s = max(
        cfg.episode_length_s,
        (args.steps + 2) * cfg.decimation * cfg.sim.dt,
    )
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()

    first, second, first_duration, second_duration = make_candidates(
        args.num_envs, direct.device
    )
    hold_scale_generator = torch.Generator(
        device=direct.device
    ).manual_seed(args.seed + 7919)
    if (
        args.eligible_hold_action_scale_min is not None
        and args.eligible_hold_action_scale_max is not None
    ):
        eligible_hold_scale = torch.empty(
            args.num_envs, device=direct.device
        ).uniform_(
            float(args.eligible_hold_action_scale_min),
            float(args.eligible_hold_action_scale_max),
            generator=hold_scale_generator,
        )
    else:
        eligible_hold_scale = torch.full(
            (args.num_envs,),
            float(args.eligible_hold_action_scale),
            device=direct.device,
        )
    parent_action_id = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    if args.parent is not None:
        with np.load(resolve(args.parent)) as parent:
            parent_mask = np.ones(
                len(parent["candidate_action_first"]), dtype=bool
            )
            if args.parent_success_only and "selection_kind" in parent:
                parent_mask = parent["selection_kind"] == "success"
            parent_first = torch.as_tensor(
                parent["candidate_action_first"][parent_mask],
                dtype=torch.float32,
                device=direct.device,
            )
            parent_second = torch.as_tensor(
                parent["candidate_action_second"][parent_mask],
                dtype=torch.float32,
                device=direct.device,
            )
            parent_first_duration = torch.as_tensor(
                parent["first_duration"][parent_mask],
                dtype=torch.long,
                device=direct.device,
            )
            parent_second_duration = torch.as_tensor(
                parent["second_duration"][parent_mask],
                dtype=torch.long,
                device=direct.device,
            )
        if len(parent_first) == 0:
            raise ValueError("Parent selection contains no candidate actions.")
        generator = torch.Generator(device=direct.device).manual_seed(
            args.seed
        )
        ids = (
            torch.arange(args.num_envs, device=direct.device)
            % len(parent_first)
            if args.parent_cycle
            else torch.randint(
                len(parent_first),
                (args.num_envs,),
                generator=generator,
                device=direct.device,
            )
        )
        parent_action_id = ids.clone()
        first = parent_first[ids]
        second = parent_second[ids]
        first[:, 16:20] += args.parent_jitter * torch.randn(
            (args.num_envs, 4),
            generator=generator,
            device=direct.device,
        )
        second[:, 16:20] += args.parent_jitter * torch.randn(
            (args.num_envs, 4),
            generator=generator,
            device=direct.device,
        )
        first[:, 0:4] += (
            args.parent_jitter
            * args.index_action_scale
            * torch.randn(
                (args.num_envs, 4),
                generator=generator,
                device=direct.device,
            )
        )
        second[:, 0:4] += (
            args.parent_jitter
            * args.index_action_scale
            * torch.randn(
                (args.num_envs, 4),
                generator=generator,
                device=direct.device,
            )
        )
        first[:, 4:8] += (
            args.parent_jitter
            * args.middle_action_scale
            * torch.randn(
                (args.num_envs, 4),
                generator=generator,
                device=direct.device,
            )
        )
        second[:, 4:8] += (
            args.parent_jitter
            * args.middle_action_scale
            * torch.randn(
                (args.num_envs, 4),
                generator=generator,
                device=direct.device,
            )
        )
        first[:, 8:12] += (
            args.parent_jitter
            * args.ring_action_scale
            * torch.randn(
                (args.num_envs, 4),
                generator=generator,
                device=direct.device,
            )
        )
        second[:, 8:12] += (
            args.parent_jitter
            * args.ring_action_scale
            * torch.randn(
                (args.num_envs, 4),
                generator=generator,
                device=direct.device,
            )
        )
        first.clamp_(-1.0, 1.0)
        second.clamp_(-1.0, 1.0)
        duration_jitter_limit = max(args.parent_duration_jitter, 0)
        if duration_jitter_limit:
            duration_jitter = torch.randint(
                -duration_jitter_limit,
                duration_jitter_limit + 1,
                (args.num_envs, 2),
                generator=generator,
                device=direct.device,
            )
        else:
            duration_jitter = torch.zeros(
                (args.num_envs, 2),
                dtype=torch.long,
                device=direct.device,
            )
        first_duration = (
            parent_first_duration[ids] + duration_jitter[:, 0]
        ).clamp(6, args.first_duration_max)
        second_duration = (
            parent_second_duration[ids] + duration_jitter[:, 1]
        ).clamp(8, args.second_duration_max)

    bank_ids = direct.current_bank_ids.clone()
    if args.continue_bundle_action:
        if not hasattr(direct, "late_previous_action_bank"):
            raise ValueError(
                "--continue-bundle-action requires source_policy_action "
                "in the curriculum bundle."
            )
        continued = direct.late_previous_action_bank[bank_ids].clone()
        if args.bundle_action_jitter > 0.0:
            generator = torch.Generator(device=direct.device).manual_seed(
                args.seed + 1543
            )
            continued += float(args.bundle_action_jitter) * torch.randn(
                continued.shape,
                generator=generator,
                device=direct.device,
            )
        first = continued.clamp(-1.0, 1.0)
        second = first.clone()
    initial_success_serial = direct.latest_success_serial.clone()
    success = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    failed = torch.zeros_like(success)
    success_step = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    best_qpos = torch.zeros(
        (args.num_envs, 27), device=direct.device
    )
    best_qvel = torch.zeros(
        (args.num_envs, 26), device=direct.device
    )
    best_penetration = torch.full(
        (args.num_envs,), torch.inf, device=direct.device
    )
    best_force = torch.zeros(
        (args.num_envs, 5), device=direct.device
    )
    best_hold = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    best_brake_score = torch.full(
        (args.num_envs,), -torch.inf, device=direct.device
    )
    best_axis_error = torch.full(
        (args.num_envs,), torch.inf, device=direct.device
    )
    best_axis_rate = torch.full_like(best_axis_error, torch.inf)
    best_angular_speed = torch.full_like(best_axis_error, torch.inf)
    best_position_error = torch.full_like(best_axis_error, torch.inf)
    best_phase = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    best_brake_step = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    best_policy_target = torch.zeros(
        (args.num_envs, 20), device=direct.device
    )
    best_policy_action = torch.zeros_like(best_policy_target)
    best_target_hand = torch.zeros(
        (args.num_envs, 20), device=direct.device
    )
    best_origin_hand = torch.zeros_like(best_target_hand)
    best_brace_hand = torch.zeros_like(best_target_hand)
    best_parent_init_pos = torch.zeros(
        (args.num_envs, 3), device=direct.device
    )
    best_parent_init_axis = torch.zeros_like(best_parent_init_pos)
    best_closure_origin_pen_pos = torch.zeros_like(best_parent_init_pos)
    best_closure_origin_pen_quat = torch.zeros(
        (args.num_envs, 4), device=direct.device
    )
    trajectory_qpos: list[torch.Tensor] = []
    trajectory_qvel: list[torch.Tensor] = []
    trajectory_phase: list[torch.Tensor] = []
    trajectory_penetration: list[torch.Tensor] = []
    teacher_observations: list[torch.Tensor] = []
    teacher_actions: list[torch.Tensor] = []

    if args.trajectory_output is not None:
        trajectory_qpos.append(
            torch.cat(
                (direct.hand_qpos, direct.pen_pos, direct.pen_quat), dim=-1
            ).detach().cpu()
        )
        trajectory_qvel.append(
            torch.cat(
                (direct.hand_qvel, direct.pen_linvel, direct.pen_angvel),
                dim=-1,
            ).detach().cpu()
        )
        trajectory_phase.append(direct.chain_phase_ids.detach().cpu())
        trajectory_penetration.append(
            direct.episode_max_penetration.detach().cpu()
        )

    age = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    feedback_approach_steps = torch.zeros(
        (), dtype=torch.long, device=direct.device
    )
    feedback_support_steps = torch.zeros_like(feedback_approach_steps)
    feedback_hold_steps = torch.zeros_like(feedback_approach_steps)
    feedback_support_latched = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    feedback_hold_latched = torch.zeros_like(feedback_support_latched)
    feedback_support_age = torch.zeros_like(age)
    feedback_loss_age = torch.zeros_like(age)
    handoff_component_steps = torch.zeros(
        7, dtype=torch.long, device=direct.device
    )
    handoff_phase_steps = torch.zeros(
        (), dtype=torch.long, device=direct.device
    )
    for step in range(args.steps):
        teacher_observation = None
        if args.teacher_dataset_output is not None:
            teacher_observation = direct._get_observations()[
                "policy"
            ].detach().cpu()
        use_first = age < first_duration
        use_second = (
            (age >= first_duration)
            & (age < first_duration + second_duration)
        )
        timed_actions = torch.where(
            use_first.unsqueeze(-1),
            first,
            torch.where(
                use_second.unsqueeze(-1),
                second,
                torch.zeros_like(first),
            ),
        )
        if args.hold_feedback_only:
            live_hold = direct.chain_phase_hold > 0
            feedback_hold_latched |= live_hold
            actions = torch.where(
                feedback_hold_latched.unsqueeze(-1),
                eligible_hold_scale.unsqueeze(-1) * second,
                timed_actions,
            )
            feedback_approach_steps += (
                ~feedback_hold_latched
            ).sum()
            feedback_hold_steps += feedback_hold_latched.sum()
        elif args.contact_feedback:
            live_contacts = (
                direct.contact_force_norm > direct.cfg.p1_contact_force_n
            )
            live_index_middle = (
                live_contacts[:, 1] & live_contacts[:, 2]
            )
            live_thumb_below = (
                direct.fingertip_pos_local[:, 0, 2]
                < direct.pen_pos[:, 2]
            )
            live_support = (
                live_index_middle
                & live_contacts[:, 0]
                & live_thumb_below
            )
            live_hold = direct.chain_phase_hold > 0
            feedback_support_latched |= live_support
            feedback_hold_latched |= live_hold
            feedback_support_age = torch.where(
                feedback_support_latched,
                feedback_support_age + 1,
                torch.zeros_like(feedback_support_age),
            )
            feedback_loss_age = torch.where(
                feedback_support_latched & ~live_support,
                feedback_loss_age + 1,
                torch.zeros_like(feedback_loss_age),
            )
            release_support = (
                feedback_support_latched
                & ~feedback_hold_latched
                & (
                    feedback_support_age
                    >= int(args.feedback_support_min_steps)
                )
                & (
                    feedback_loss_age
                    >= int(args.feedback_loss_patience)
                )
            )
            feedback_support_latched &= ~release_support
            feedback_support_age = torch.where(
                release_support,
                torch.zeros_like(feedback_support_age),
                feedback_support_age,
            )
            feedback_loss_age = torch.where(
                release_support,
                torch.zeros_like(feedback_loss_age),
                feedback_loss_age,
            )
            actions = torch.where(
                feedback_support_latched.unsqueeze(-1), second, first
            )
            actions = torch.where(
                feedback_hold_latched.unsqueeze(-1),
                eligible_hold_scale.unsqueeze(-1) * second,
                actions,
            )
            feedback_approach_steps += (
                ~feedback_support_latched
            ).sum()
            feedback_support_steps += (
                feedback_support_latched & ~feedback_hold_latched
            ).sum()
            feedback_hold_steps += feedback_hold_latched.sum()
        else:
            actions = timed_actions
        if teacher_observation is not None:
            teacher_action = actions.clone()
            if bool(getattr(direct.cfg, "thumb_only_residual", False)):
                teacher_action[:, :16] = 0.0
            teacher_observations.append(teacher_observation)
            teacher_actions.append(teacher_action.detach().cpu())
        _, _, terminated, truncated, _ = env.step(actions)
        direct._compute_intermediate_values()
        source_phase = args.target_phase - 1
        source_active = direct.chain_phase_ids == source_phase
        live_contacts = (
            direct.contact_force_norm > direct.cfg.p1_contact_force_n
        )
        index_middle = live_contacts[:, 1] & live_contacts[:, 2]
        thumb_contact = live_contacts[:, 0]
        thumb_below = (
            direct.fingertip_pos_local[:, 0, 2]
            < direct.pen_pos[:, 2]
        )
        angular_limit = float(
            direct.cfg.thumb_wrap_handoff_max_angular_speed_rad_s
        )
        linear_limit = float(
            direct.cfg.thumb_wrap_handoff_max_linear_speed_m_s
        )
        angular_ready = (
            torch.ones_like(source_active)
            if angular_limit <= 0.0
            else torch.linalg.norm(direct.pen_angvel, dim=-1)
            <= angular_limit
        )
        linear_ready = (
            torch.ones_like(source_active)
            if linear_limit <= 0.0
            else torch.linalg.norm(direct.pen_linvel, dim=-1)
            <= linear_limit
        )
        pinky_clear = ~live_contacts[:, 4]
        all_handoff = (
            index_middle
            & thumb_contact
            & thumb_below
            & angular_ready
            & linear_ready
            & pinky_clear
        )
        handoff_components = torch.stack(
            (
                index_middle,
                thumb_contact,
                thumb_below,
                angular_ready,
                linear_ready,
                pinky_clear,
                all_handoff,
            ),
            dim=-1,
        )
        handoff_phase_steps += source_active.sum()
        handoff_component_steps += (
            source_active.unsqueeze(-1) & handoff_components
        ).sum(dim=0)
        target_axis = quat_apply(
            direct.target_pen_quat, direct.pen_x_axis
        )
        axis_dot = torch.sum(
            direct.pen_axis * target_axis, dim=-1
        ).clamp(-1.0, 1.0)
        axis_error = torch.acos(axis_dot.abs())
        axis_rate = torch.linalg.norm(
            torch.linalg.cross(
                direct.pen_angvel, direct.pen_axis, dim=-1
            ),
            dim=-1,
        )
        angular_speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        position_error = torch.linalg.norm(
            direct.pen_pos - direct.target_pen_pos, dim=-1
        )
        active_phase = source_active | (
            direct.chain_phase_ids == args.target_phase
        )
        brake_valid = (
            active_phase
            & index_middle
            & thumb_contact
            & pinky_clear
            & (
                axis_error
                <= torch.deg2rad(
                    torch.full_like(
                        axis_error, args.max_braking_axis_error_deg
                    )
                )
            )
            & (
                position_error
                <= args.max_braking_position_error_mm / 1000.0
            )
            & (
                direct.episode_max_penetration
                <= direct.cfg.penetration_threshold
            )
            & (
                direct.contact_force_norm.max(dim=-1).values
                <= direct.cfg.p1_max_force_n
            )
            & ~success
            & ~failed
        )
        brake_score = (
            -torch.rad2deg(axis_error)
            - 2.5 * axis_rate
            - 0.5 * angular_speed
            - 0.25 * position_error * 1000.0
            + 2.0
            * (direct.chain_phase_ids == args.target_phase).float()
        )
        brake_improved = brake_valid & (
            brake_score > best_brake_score
        )
        if args.export_braking_near and torch.any(brake_improved):
            best_brake_score = torch.where(
                brake_improved, brake_score, best_brake_score
            )
            best_axis_error[brake_improved] = axis_error[brake_improved]
            best_axis_rate[brake_improved] = axis_rate[brake_improved]
            best_angular_speed[brake_improved] = angular_speed[
                brake_improved
            ]
            best_position_error[brake_improved] = position_error[
                brake_improved
            ]
            best_phase[brake_improved] = direct.chain_phase_ids[
                brake_improved
            ]
            best_brake_step[brake_improved] = step
            best_qpos[brake_improved] = torch.cat(
                (direct.hand_qpos, direct.pen_pos, direct.pen_quat),
                dim=-1,
            )[brake_improved]
            best_qvel[brake_improved] = torch.cat(
                (
                    direct.hand_qvel,
                    direct.pen_linvel,
                    direct.pen_angvel,
                ),
                dim=-1,
            )[brake_improved]
            best_penetration[brake_improved] = (
                direct.episode_max_penetration[brake_improved]
            )
            best_force[brake_improved] = direct.contact_force_norm[
                brake_improved
            ]
            best_policy_target[brake_improved] = direct.policy_targets[
                brake_improved
            ]
            best_policy_action[brake_improved] = direct.actions[
                brake_improved
            ]
            best_target_hand[brake_improved] = direct.target_hand_qpos[
                brake_improved
            ]
            best_origin_hand[brake_improved] = direct.closure_origin_hand[
                brake_improved
            ]
            best_brace_hand[brake_improved] = direct.closure_brace_hand[
                brake_improved
            ]
            best_parent_init_pos[brake_improved] = direct.init_pen_pos[
                brake_improved
            ]
            best_parent_init_axis[brake_improved] = direct.init_pen_axis[
                brake_improved
            ]
            best_closure_origin_pen_pos[brake_improved] = (
                direct.closure_origin_pen_pos[brake_improved]
            )
            best_closure_origin_pen_quat[brake_improved] = (
                direct.closure_origin_pen_quat[brake_improved]
            )
        current_hold = torch.where(
            direct.chain_phase_ids == source_phase,
            direct.chain_phase_hold,
            torch.zeros_like(direct.chain_phase_hold),
        )
        improved = (
            (current_hold > best_hold)
            & ~success
            & ~failed
            & (not args.export_braking_near)
        )
        if torch.any(improved):
            best_hold[improved] = current_hold[improved]
            best_qpos[improved] = torch.cat(
                (direct.hand_qpos, direct.pen_pos, direct.pen_quat),
                dim=-1,
            )[improved]
            best_qvel[improved] = torch.cat(
                (
                    direct.hand_qvel,
                    direct.pen_linvel,
                    direct.pen_angvel,
                ),
                dim=-1,
            )[improved]
            best_penetration[improved] = (
                direct.episode_max_penetration[improved]
            )
            best_force[improved] = direct.contact_force_norm[improved]
            best_policy_target[improved] = direct.policy_targets[improved]
            best_policy_action[improved] = direct.actions[improved]
            best_target_hand[improved] = direct.target_hand_qpos[improved]
            best_origin_hand[improved] = direct.closure_origin_hand[improved]
            best_brace_hand[improved] = direct.closure_brace_hand[improved]
            best_parent_init_pos[improved] = direct.init_pen_pos[improved]
            best_parent_init_axis[improved] = direct.init_pen_axis[improved]
            best_closure_origin_pen_pos[improved] = (
                direct.closure_origin_pen_pos[improved]
            )
            best_closure_origin_pen_quat[improved] = (
                direct.closure_origin_pen_quat[improved]
            )
        if args.trajectory_output is not None:
            trajectory_qpos.append(
                torch.cat(
                    (direct.hand_qpos, direct.pen_pos, direct.pen_quat),
                    dim=-1,
                ).detach().cpu()
            )
            trajectory_qvel.append(
                torch.cat(
                    (
                        direct.hand_qvel,
                        direct.pen_linvel,
                        direct.pen_angvel,
                    ),
                    dim=-1,
                ).detach().cpu()
            )
            trajectory_phase.append(
                direct.chain_phase_ids.detach().cpu()
            )
            trajectory_penetration.append(
                direct.episode_max_penetration.detach().cpu()
            )
        reached = (
            direct.latest_success_serial > initial_success_serial
            if args.final_success
            else direct.chain_phase_ids >= args.target_phase
        )
        newly = reached & ~success & ~failed
        if torch.any(newly):
            success[newly] = True
            success_step[newly] = step
            if args.final_success:
                best_qpos[newly] = direct.latest_success_qpos[newly]
                best_qvel[newly] = direct.latest_success_qvel[newly]
                best_penetration[newly] = (
                    direct.latest_success_penetration[newly]
                )
                best_force[newly] = (
                    direct.latest_success_contact_force[newly]
                )
            else:
                best_qpos[newly] = torch.cat(
                    (direct.hand_qpos, direct.pen_pos, direct.pen_quat),
                    dim=-1,
                )[newly]
                best_qvel[newly] = torch.cat(
                    (
                        direct.hand_qvel,
                        direct.pen_linvel,
                        direct.pen_angvel,
                    ),
                    dim=-1,
                )[newly]
                best_penetration[newly] = (
                    direct.episode_max_penetration[newly]
                )
                best_force[newly] = direct.contact_force_norm[newly]
            best_target_hand[newly] = direct.target_hand_qpos[newly]
            best_policy_target[newly] = direct.policy_targets[newly]
            best_policy_action[newly] = direct.actions[newly]
            best_origin_hand[newly] = direct.closure_origin_hand[newly]
            best_brace_hand[newly] = direct.closure_brace_hand[newly]
            best_parent_init_pos[newly] = direct.init_pen_pos[newly]
            best_parent_init_axis[newly] = direct.init_pen_axis[newly]
            best_closure_origin_pen_pos[newly] = (
                direct.closure_origin_pen_pos[newly]
            )
            best_closure_origin_pen_quat[newly] = (
                direct.closure_origin_pen_quat[newly]
            )
            best_axis_error[newly] = axis_error[newly]
            best_axis_rate[newly] = axis_rate[newly]
            best_angular_speed[newly] = angular_speed[newly]
            best_position_error[newly] = position_error[newly]
            best_phase[newly] = direct.chain_phase_ids[newly]
        done = (terminated | truncated).reshape(-1)
        failed |= done & ~success
        age += 1

    near = (
        (
            (args.min_near_hold > 0)
            & (best_hold >= args.min_near_hold)
        )
        | (
            args.export_braking_near
            & torch.isfinite(best_brake_score)
        )
    )
    selected = (success | near) & (
        best_penetration <= direct.cfg.penetration_threshold
    )
    ids = torch.where(selected)[0]
    if ids.numel() > args.max_selected:
        successful_ids = ids[success[ids]]
        successful_ids = successful_ids[
            torch.argsort(success_step[successful_ids])
        ]
        near_ids = ids[~success[ids]]
        near_rank = (
            best_brake_score[near_ids]
            if args.export_braking_near
            else best_hold[near_ids].float()
        )
        near_ids = near_ids[torch.argsort(near_rank, descending=True)]
        ids = torch.cat((successful_ids, near_ids))[
            : args.max_selected
        ]

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    qpos = best_qpos[ids].cpu().numpy().astype(np.float32)
    qvel = best_qvel[ids].cpu().numpy().astype(np.float32)
    pen_pos = qpos[:, 20:23]
    pen_quat = qpos[:, 23:27]
    w, x, y, z = np.moveaxis(pen_quat, -1, 0)
    pen_axis = np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        axis=-1,
    ).astype(np.float32)
    if args.preserve_parent_pen_reference:
        exported_closure_origin_pen_pos = (
            best_closure_origin_pen_pos[ids].cpu().numpy().astype(np.float32)
        )
        exported_closure_origin_pen_quat = (
            best_closure_origin_pen_quat[ids].cpu().numpy().astype(np.float32)
        )
        exported_init_pen_pos = (
            best_parent_init_pos[ids].cpu().numpy().astype(np.float32)
        )
        exported_init_pen_axis = (
            best_parent_init_axis[ids].cpu().numpy().astype(np.float32)
        )
    else:
        exported_closure_origin_pen_pos = pen_pos
        exported_closure_origin_pen_quat = pen_quat
        exported_init_pen_pos = pen_pos
        exported_init_pen_axis = pen_axis
    np.savez_compressed(
        output,
        source_qpos=qpos,
        source_qvel=qvel,
        source_target_qpos=best_policy_target[
            ids
        ].cpu().numpy().astype(np.float32),
        source_policy_action=best_policy_action[
            ids
        ].cpu().numpy().astype(np.float32),
        source_target_hand_qpos=best_target_hand[
            ids
        ].cpu().numpy().astype(np.float32),
        source_closure_origin_hand=best_origin_hand[
            ids
        ].cpu().numpy().astype(np.float32),
        source_closure_brace_hand=best_brace_hand[
            ids
        ].cpu().numpy().astype(np.float32),
        source_closure_origin_pen_pos=exported_closure_origin_pen_pos,
        source_closure_origin_pen_quat=exported_closure_origin_pen_quat,
        source_init_pen_pos=exported_init_pen_pos,
        source_init_pen_axis=exported_init_pen_axis,
        source_parent_init_pen_pos=best_parent_init_pos[
            ids
        ].cpu().numpy().astype(np.float32),
        source_parent_init_pen_axis=best_parent_init_axis[
            ids
        ].cpu().numpy().astype(np.float32),
        source_max_penetration_m=best_penetration[
            ids
        ].cpu().numpy().astype(np.float32),
        source_contact_force_n=best_force[
            ids
        ].cpu().numpy().astype(np.float32),
        source_chain_phase_id=torch.where(
            success[ids],
            torch.full_like(ids, args.target_phase),
            best_phase[ids],
        ).cpu().numpy().astype(np.int64),
        selection_kind=np.where(
            success[ids].cpu().numpy(),
            "success",
            "braking_near" if args.export_braking_near else "near_hold",
        ).astype(np.str_),
        best_hold_steps=best_hold[ids].cpu().numpy().astype(np.int64),
        target_qpos=qpos,
        target_qvel=np.zeros_like(qvel),
        phase_ids=np.zeros(len(ids), dtype=np.int64),
        candidate_action_first=first[ids].cpu().numpy().astype(np.float32),
        candidate_action_second=second[ids].cpu().numpy().astype(np.float32),
        first_duration=first_duration[ids].cpu().numpy().astype(np.int64),
        second_duration=second_duration[ids].cpu().numpy().astype(np.int64),
        eligible_hold_action_scale=eligible_hold_scale[
            ids
        ].cpu().numpy().astype(np.float32),
        source_bank_id=bank_ids[ids].cpu().numpy().astype(np.int64),
        parent_action_id=parent_action_id[ids]
        .cpu().numpy().astype(np.int64),
        success_step=success_step[ids].cpu().numpy().astype(np.int64),
        best_brake_score=best_brake_score[
            ids
        ].cpu().numpy().astype(np.float32),
        best_axis_error_rad=best_axis_error[
            ids
        ].cpu().numpy().astype(np.float32),
        best_axis_rate_rad_s=best_axis_rate[
            ids
        ].cpu().numpy().astype(np.float32),
        best_angular_speed_rad_s=best_angular_speed[
            ids
        ].cpu().numpy().astype(np.float32),
        best_position_error_m=best_position_error[
            ids
        ].cpu().numpy().astype(np.float32),
        best_brake_step=best_brake_step[
            ids
        ].cpu().numpy().astype(np.int64),
        seed=np.asarray(args.seed, dtype=np.int64),
        source=np.asarray(
            "local index/middle/thumb transition action search",
            dtype=np.str_,
        ),
    )
    trajectory_output = None
    trajectory_export_ids = ids[
        success[ids]
        | (
            args.export_braking_near
            & torch.isfinite(best_brake_score[ids])
        )
    ]
    if (
        args.trajectory_output is not None
        and trajectory_export_ids.numel()
    ):
        successful_exports = trajectory_export_ids[
            success[trajectory_export_ids]
        ]
        if successful_exports.numel():
            representative = int(
                successful_exports[
                    torch.argmin(success_step[successful_exports])
                ].item()
            )
            end_step = int(success_step[representative].item()) + 2
            trajectory_kind = "successful_local_primitive"
        else:
            representative = int(
                trajectory_export_ids[
                    torch.argmax(best_brake_score[trajectory_export_ids])
                ].item()
            )
            end_step = int(best_brake_step[representative].item()) + 2
            trajectory_kind = "low_axis_rate_braking_near"
        trajectory_output = resolve(args.trajectory_output)
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_output,
            qpos=torch.stack(trajectory_qpos[:end_step], dim=0)[
                :, representative
            ].numpy().astype(np.float32),
            qvel=torch.stack(trajectory_qvel[:end_step], dim=0)[
                :, representative
            ].numpy().astype(np.float32),
            phase_ids=torch.stack(
                trajectory_phase[:end_step], dim=0
            )[:, representative].numpy().astype(np.int64),
            max_penetration_m=torch.stack(
                trajectory_penetration[:end_step], dim=0
            )[:, representative].numpy().astype(np.float32),
            control_dt_s=np.asarray(
                direct.cfg.decimation * direct.cfg.sim.dt,
                dtype=np.float32,
            ),
            trajectory_kind=np.asarray(
                trajectory_kind, dtype=np.str_
            ),
            deepest_phase=np.asarray(
                args.target_phase, dtype=np.int64
            ),
            state_reset=np.asarray(
                "local primitive reset at trajectory start", dtype=np.str_
            ),
        )

    teacher_dataset_output = None
    if (
        args.teacher_dataset_output is not None
        and trajectory_export_ids.numel()
    ):
        teacher_dataset_output = resolve(args.teacher_dataset_output)
        teacher_dataset_output.parent.mkdir(parents=True, exist_ok=True)
        teacher_ids = trajectory_export_ids
        observation = torch.stack(teacher_observations, dim=1)[
            teacher_ids.cpu()
        ].numpy().astype(np.float32)
        teacher_action = torch.stack(teacher_actions, dim=1)[
            teacher_ids.cpu()
        ].numpy().astype(np.float32)
        teacher_end_step = torch.where(
            success[teacher_ids],
            success_step[teacher_ids],
            best_brake_step[teacher_ids],
        )
        lengths = (teacher_end_step + 1).cpu().numpy().astype(np.int64)
        stage_id = np.full(
            observation.shape[:2],
            args.target_phase - 1,
            dtype=np.int64,
        )
        np.savez_compressed(
            teacher_dataset_output,
            observation=observation,
            action=teacher_action,
            stage_id=stage_id,
            lengths=lengths,
            source_candidate_id=teacher_ids.cpu().numpy().astype(
                np.int64
            ),
            selection_kind=np.where(
                success[teacher_ids].cpu().numpy(),
                "success",
                "braking_near",
            ).astype(np.str_),
            source=np.asarray(
                "index/middle/thumb braking teacher trajectories",
                dtype=np.str_,
            ),
        )

    summary = {
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "target_phase": args.target_phase,
        "final_success": args.final_success,
        "steps": args.steps,
        "action_scale": args.action_scale,
        "index_action_scale": args.index_action_scale,
        "middle_action_scale": args.middle_action_scale,
        "ring_action_scale": args.ring_action_scale,
        "first_duration_max": args.first_duration_max,
        "second_duration_max": args.second_duration_max,
        "parent": str(resolve(args.parent)) if args.parent else None,
        "parent_jitter": args.parent_jitter,
        "parent_cycle": args.parent_cycle,
        "parent_success_only": args.parent_success_only,
        "parent_duration_jitter": args.parent_duration_jitter,
        "contact_feedback": args.contact_feedback,
        "hold_feedback_only": args.hold_feedback_only,
        "eligible_hold_action_scale": args.eligible_hold_action_scale,
        "eligible_hold_action_scale_min": (
            args.eligible_hold_action_scale_min
        ),
        "eligible_hold_action_scale_max": (
            args.eligible_hold_action_scale_max
        ),
        "feedback_support_min_steps": args.feedback_support_min_steps,
        "feedback_loss_patience": args.feedback_loss_patience,
        "disable_scripted_teacher": args.disable_scripted_teacher,
        "disable_brace_teacher": args.disable_brace_teacher,
        "trainee_policy_scale": args.trainee_policy_scale,
        "scripted_residual_scale": args.scripted_residual_scale,
        "preserve_parent_pen_reference": args.preserve_parent_pen_reference,
        "restore_cached_phase_hold": args.restore_cached_phase_hold,
        "continue_bundle_action": args.continue_bundle_action,
        "bundle_action_jitter": args.bundle_action_jitter,
        "export_braking_near": args.export_braking_near,
        "max_braking_axis_error_deg": args.max_braking_axis_error_deg,
        "max_braking_position_error_mm": (
            args.max_braking_position_error_mm
        ),
        "successes": int(success.sum().item()),
        "near_candidates": int((near & ~success).sum().item()),
        "min_near_hold": args.min_near_hold,
        "best_candidate_hold": int(best_hold.max().item()),
        "feedback_action_steps": {
            "approach": int(feedback_approach_steps.item()),
            "support": int(feedback_support_steps.item()),
            "eligible_hold": int(feedback_hold_steps.item()),
        },
        "handoff_component_steps": {
            name: int(value)
            for name, value in zip(
                (
                    "index_middle_contact",
                    "thumb_contact",
                    "thumb_below",
                    "angular_speed_ready",
                    "linear_speed_ready",
                    "pinky_clear",
                    "all_contact_speed_clear",
                ),
                handoff_component_steps.detach().cpu().tolist(),
            )
        },
        "handoff_phase_steps": int(handoff_phase_steps.item()),
        "selected": int(ids.numel()),
        "success_rate": float(success.float().mean().item()),
        "failed": int(failed.sum().item()),
        "max_penetration_mm": (
            float(best_penetration[ids].max().item() * 1000.0)
            if ids.numel()
            else None
        ),
        "best_braking_near": (
            {
                "axis_error_deg": float(
                    torch.rad2deg(best_axis_error[ids]).min().item()
                ),
                "axis_rate_rad_s": float(
                    best_axis_rate[ids].min().item()
                ),
                "angular_speed_rad_s": float(
                    best_angular_speed[ids].min().item()
                ),
                "position_error_mm": float(
                    best_position_error[ids].min().item() * 1000.0
                ),
            }
            if ids.numel()
            else None
        ),
        "mean_success_step": (
            float(success_step[ids].float().mean().item())
            if ids.numel()
            else None
        ),
        "output": str(output),
        "trajectory_output": (
            str(trajectory_output)
            if trajectory_output is not None
            else None
        ),
        "teacher_dataset_output": (
            str(teacher_dataset_output)
            if teacher_dataset_output is not None
            else None
        ),
    }
    if hasattr(direct, "closure_stage_step_counts"):
        summary["closure_stage_diagnostics"] = {
            "steps": direct.closure_stage_step_counts.detach()
            .cpu().tolist(),
            "contact_steps": (
                direct.closure_stage_triad_step_counts.detach()
                .cpu().tolist()
            ),
            "contact_pose_steps": (
                direct.closure_stage_contact_pose_step_counts.detach()
                .cpu().tolist()
            ),
            "safe_steps": direct.closure_stage_safe_step_counts.detach()
            .cpu().tolist(),
            "safe_component_steps": (
                direct.closure_stage_safe_component_step_counts.detach()
                .cpu().tolist()
            ),
            "eligible_steps": (
                direct.closure_stage_eligible_step_counts.detach()
                .cpu().tolist()
            ),
            "best_hold_steps": (
                direct.closure_stage_best_hold.detach().cpu().tolist()
            ),
            "min_target_rmse_rad": (
                direct.closure_stage_min_target_rmse.detach()
                .cpu().tolist()
            ),
            "min_position_error_m": (
                direct.closure_stage_min_position_error.detach()
                .cpu().tolist()
            ),
            "min_axis_error_rad": (
                direct.closure_stage_min_axis_error.detach()
                .cpu().tolist()
            ),
        }
    summary_path = resolve(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PHASE7_THUMB_UNDER_SEARCH=" + json.dumps(summary, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
