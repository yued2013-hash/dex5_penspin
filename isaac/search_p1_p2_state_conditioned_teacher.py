from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Measure train-only per-reset oracle coverage for C2 ring paths."
)
parser.add_argument("--task", default="Dex5-P1P2-C2-Direct-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--npz-output", type=Path, required=True)
parser.add_argument("--joint0-fractions", type=float, nargs="+", required=True)
parser.add_argument("--joint1-fractions", type=float, nargs="+", required=True)
parser.add_argument("--joint2-fractions", type=float, nargs="+", required=True)
parser.add_argument("--joint3-fractions", type=float, nargs="+", required=True)
parser.add_argument("--support-actions", type=float, nargs=3, default=(0.4, 0.4, 0.0))
parser.add_argument(
    "--support-action-candidate",
    dest="support_action_candidates",
    type=float,
    nargs=3,
    action="append",
    default=None,
    help="Repeat to pair each ring path with an index/middle/thumb support candidate.",
)
parser.add_argument("--support-roll-actions", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument(
    "--support-roll-action-candidate",
    dest="support_roll_action_candidates",
    type=float,
    nargs=3,
    action="append",
    default=None,
    help="Repeat for index Roll_21R, middle Roll_31R, and thumb Yaw_11R.",
)
parser.add_argument("--brake-progress-margin", type=float, default=0.08)
parser.add_argument("--brake-distance-mm", type=float, default=11.0)
parser.add_argument("--batch-reset-ids", type=int, default=8)
parser.add_argument("--max-reset-ids", type=int, default=120)
parser.add_argument("--episode-steps", type=int, default=48)
parser.add_argument(
    "--episode-length-s",
    type=float,
    default=None,
    help="Optional diagnostic episode horizon override; does not change gate thresholds.",
)
parser.add_argument("--entry-map", type=Path, default=None)
parser.add_argument("--entry-progress", type=float, default=0.60)
parser.add_argument("--entry-distance-mm", type=float, default=10.0)
parser.add_argument("--entry-hold-steps", type=int, default=1)
parser.add_argument(
    "--second-stage-mode",
    choices=(
        "simultaneous",
        "roll_then_flex",
        "flex_then_roll",
        "clearance_roll_flex",
        "roll_proximal_then_distal",
        "roll_distal_then_proximal",
        "hold_entry",
    ),
    default="simultaneous",
    help="Sequence Roll_41R and Pitch_42R/43R/44R after entering the C2 region.",
)
parser.add_argument(
    "--clearance-flex-fractions",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    help="Pitch_42R/43R/44R P1-to-P2 fractions used while Roll_41R moves laterally.",
)
parser.add_argument(
    "--clearance-joint-tolerance-rad",
    type=float,
    default=None,
    help="Optional looser pitch error for leaving the clearance sub-stage.",
)
parser.add_argument(
    "--sequence-joint-tolerance-rad",
    type=float,
    default=0.06,
    help="Actual-joint error that advances a sequenced second stage.",
)
parser.add_argument(
    "--substage-joint-tolerance-rad",
    type=float,
    default=None,
    help="Optional tolerance for the proximal/distal sub-stage after roll.",
)
parser.add_argument(
    "--final-ring-action-scale",
    type=float,
    default=1.0,
    help="Action multiplier during the final closure sub-stage.",
)
parser.add_argument(
    "--final-ring-force-limit-n",
    type=float,
    default=None,
    help="Back off proximal ring flexion above this final-stage contact load.",
)
parser.add_argument("--final-ring-backoff-action", type=float, default=0.20)
parser.add_argument("--seed", type=int, default=20260722)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ring_candidates = list(
        itertools.product(
            args.joint0_fractions,
            args.joint1_fractions,
            args.joint2_fractions,
            args.joint3_fractions,
        )
    )
    support_candidates = args.support_action_candidates or [args.support_actions]
    support_roll_candidates = args.support_roll_action_candidates or [args.support_roll_actions]
    candidates = [
        (ring_candidate, support_candidate, support_roll_candidate)
        for ring_candidate in ring_candidates
        for support_candidate in support_candidates
        for support_roll_candidate in support_roll_candidates
    ]
    ring_candidate_tensor = torch.tensor(
        [candidate[0] for candidate in candidates], dtype=torch.float32, device=args.device
    )
    support_candidate_tensor = torch.tensor(
        [candidate[1] for candidate in candidates], dtype=torch.float32, device=args.device
    )
    support_roll_candidate_tensor = torch.tensor(
        [candidate[2] for candidate in candidates], dtype=torch.float32, device=args.device
    )
    num_candidates = len(candidates)
    batch_size = args.batch_reset_ids
    num_envs = num_candidates * batch_size

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = args.seed
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped

    num_reset_ids = min(args.max_reset_ids, direct.bank_size)
    # A fixed permutation makes the probe reproducible while covering the train bank broadly.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    selected_local_ids = torch.randperm(direct.bank_size, generator=generator)[:num_reset_ids]
    selected_global_ids = direct.global_bank_id_bank[selected_local_ids.to(direct.device)].cpu()

    success_matrix = torch.zeros((num_reset_ids, num_candidates), dtype=torch.bool)
    drop_matrix = torch.zeros_like(success_matrix)
    eligible_steps = torch.zeros((num_reset_ids, num_candidates), dtype=torch.int16)
    geometry_steps = torch.zeros_like(eligible_steps)
    geometry_triad_steps = torch.zeros_like(eligible_steps)
    geometry_axis_steps = torch.zeros_like(eligible_steps)
    geometry_velocity_steps = torch.zeros_like(eligible_steps)
    second_stage_entered = torch.zeros_like(success_matrix)
    sequence_completed = torch.zeros_like(success_matrix)
    min_sequence_error_rad = torch.full(
        (num_reset_ids, num_candidates), float("inf")
    )
    min_distance_mm = torch.full((num_reset_ids, num_candidates), float("inf"))
    min_axis_drift_at_geometry_deg = torch.full(
        (num_reset_ids, num_candidates), float("inf")
    )
    min_axis_drift_at_geometry_triad_deg = torch.full(
        (num_reset_ids, num_candidates), float("inf")
    )
    min_ring_force_at_geometry_n = torch.full(
        (num_reset_ids, num_candidates), float("inf")
    )
    max_ring_force_n = torch.zeros((num_reset_ids, num_candidates))
    max_axis_drift_deg = torch.zeros((num_reset_ids, num_candidates))
    max_axis_drift_before_final_deg = torch.zeros((num_reset_ids, num_candidates))
    triad_steps = torch.zeros_like(eligible_steps)
    max_position_drift_mm = torch.zeros((num_reset_ids, num_candidates))
    final_axis_cross = torch.zeros((num_reset_ids, num_candidates, 3))
    second_stage_entry_step = torch.full((num_reset_ids, num_candidates), -1, dtype=torch.int16)
    sequence_complete_step = torch.full_like(second_stage_entry_step, -1)
    first_axis_violation_step = torch.full_like(second_stage_entry_step, -1)
    first_geometry_step = torch.full_like(second_stage_entry_step, -1)
    max_penetration_mm = torch.zeros((num_reset_ids, num_candidates))

    target_progress = float(cfg.stage_target_progress[int(cfg.curriculum_stage)])
    progress_tolerance = float(cfg.stage_progress_tolerance[int(cfg.curriculum_stage)])
    target_radius = float(cfg.stage_target_distance_m[int(cfg.curriculum_stage)])
    increment = 0.25 * cfg.action_scale
    entry_routes = None
    if args.entry_map is not None:
        entry_map_path = resolve(args.entry_map)
        with np.load(entry_map_path) as entry_data:
            entry_global_ids = np.asarray(entry_data["selected_global_ids"], dtype=np.int64)
            entry_candidates = np.asarray(entry_data["candidates"], dtype=np.float32)
            entry_best_ids = np.asarray(entry_data["best_candidate_ids"], dtype=np.int64)
        route_by_global = {
            int(global_id): entry_candidates[entry_best_ids[row]]
            for row, global_id in enumerate(entry_global_ids)
        }
        entry_routes = torch.as_tensor(
            np.asarray(
                [route_by_global[int(global_id)] for global_id in direct.global_bank_id_bank.cpu()],
                dtype=np.float32,
            ),
            device=direct.device,
        )

    for batch_start in range(0, num_reset_ids, batch_size):
        valid_count = min(batch_size, num_reset_ids - batch_start)
        batch_ids = selected_local_ids[batch_start : batch_start + valid_count]
        if valid_count < batch_size:
            batch_ids = torch.cat((batch_ids, batch_ids[:1].repeat(batch_size - valid_count)))

        # Candidate-major layout: each path sees exactly the same reset IDs.
        forced_ids = batch_ids.repeat(num_candidates).to(direct.device)
        candidate_ids = torch.arange(num_candidates, device=direct.device).repeat_interleave(
            batch_size
        )
        reset_slots = torch.arange(batch_size, device=direct.device).repeat(num_candidates)
        direct.forced_bank_ids = forced_ids
        env.reset()

        fractions = ring_candidate_tensor[candidate_ids]
        support = support_candidate_tensor[candidate_ids]
        support_roll = support_roll_candidate_tensor[candidate_ids]
        first_stage_fractions = (
            entry_routes[forced_ids] if entry_routes is not None else fractions
        )
        second_stage = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
        clearance_complete = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
        roll_complete = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
        substage_complete = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
        sequence_phase_two = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
        entry_hold_count = torch.zeros(num_envs, dtype=torch.long, device=direct.device)
        brake_latched = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
        brake_targets = torch.zeros((num_envs, 4), device=direct.device)
        active = torch.ones(num_envs, dtype=torch.bool, device=direct.device)
        batch_success = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
        batch_drop = torch.zeros_like(batch_success)
        batch_eligible = torch.zeros(num_envs, dtype=torch.int16, device=direct.device)
        batch_geometry = torch.zeros_like(batch_eligible)
        batch_geometry_triad = torch.zeros_like(batch_eligible)
        batch_geometry_axis = torch.zeros_like(batch_eligible)
        batch_geometry_velocity = torch.zeros_like(batch_eligible)
        batch_min_distance = torch.full(
            (num_envs,), float("inf"), dtype=torch.float32, device=direct.device
        )
        batch_min_sequence_error = torch.full(
            (num_envs,), float("inf"), dtype=torch.float32, device=direct.device
        )
        batch_min_axis_geometry = torch.full(
            (num_envs,), float("inf"), dtype=torch.float32, device=direct.device
        )
        batch_min_axis_geometry_triad = torch.full(
            (num_envs,), float("inf"), dtype=torch.float32, device=direct.device
        )
        batch_min_ring_force_geometry = torch.full(
            (num_envs,), float("inf"), dtype=torch.float32, device=direct.device
        )
        batch_max_ring_force = torch.zeros(num_envs, device=direct.device)
        batch_max_axis_drift = torch.zeros(num_envs, device=direct.device)
        batch_max_axis_before_final = torch.zeros(num_envs, device=direct.device)
        batch_triad_steps = torch.zeros(num_envs, dtype=torch.int16, device=direct.device)
        batch_max_position_drift = torch.zeros(num_envs, device=direct.device)
        batch_final_axis_cross = torch.zeros((num_envs, 3), device=direct.device)
        batch_entry_step = torch.full(
            (num_envs,), -1, dtype=torch.int16, device=direct.device
        )
        batch_sequence_complete_step = torch.full_like(batch_entry_step, -1)
        batch_first_axis_violation_step = torch.full_like(batch_entry_step, -1)
        batch_first_geometry_step = torch.full_like(batch_entry_step, -1)
        batch_max_penetration = torch.zeros(num_envs, device=direct.device)

        for step_index in range(args.episode_steps):
            feature, progress, target_distance, _ = direct._ring_geometry()
            pre_step_axis_cross = torch.cross(
                direct.init_pen_axis, direct.pen_axis, dim=-1
            )
            batch_final_axis_cross = torch.where(
                active.unsqueeze(-1), pre_step_axis_cross, batch_final_axis_cross
            )
            if entry_routes is not None:
                transfer_vector = direct.p2_ring_center.unsqueeze(0) - direct.initial_ring_feature
                entry_center = direct.initial_ring_feature + args.entry_progress * transfer_vector
                entry_distance = torch.linalg.norm(feature - entry_center, dim=-1)
                entry_geometry = (
                    (torch.abs(progress - args.entry_progress) <= 0.12)
                    & (entry_distance <= args.entry_distance_mm / 1000.0)
                )
                entry_contact = (direct.primary_contact_grace > 0).all(dim=-1)
                entry_eligible = active & entry_geometry & entry_contact
                entry_hold_count = torch.where(
                    entry_eligible, entry_hold_count + 1, torch.zeros_like(entry_hold_count)
                )
                newly_entered = ~second_stage & (
                    entry_hold_count >= args.entry_hold_steps
                )
                batch_entry_step = torch.where(
                    newly_entered,
                    torch.full_like(batch_entry_step, step_index),
                    batch_entry_step,
                )
                second_stage |= newly_entered

            ring_source = direct.source_targets[:, 8:12]
            ring_final = direct.p2_target_ring_qpos.unsqueeze(0).expand_as(ring_source)
            candidate_ring_target = ring_source + fractions * (ring_final - ring_source)
            entry_ring_target = ring_source + first_stage_fractions * (ring_final - ring_source)
            candidate_ring_target = torch.maximum(
                torch.minimum(candidate_ring_target, direct.joint_upper[:, 8:12]),
                direct.joint_lower[:, 8:12],
            )
            entry_ring_target = torch.maximum(
                torch.minimum(entry_ring_target, direct.joint_upper[:, 8:12]),
                direct.joint_lower[:, 8:12],
            )
            if args.second_stage_mode == "roll_then_flex":
                first_sequence_target = entry_ring_target.clone()
                first_sequence_target[:, 0] = candidate_ring_target[:, 0]
                sequence_error = torch.abs(
                    direct.hand_qpos[:, 8] - first_sequence_target[:, 0]
                )
                sequence_phase_two |= second_stage & (
                    sequence_error <= args.sequence_joint_tolerance_rad
                )
                sequenced_ring_target = torch.where(
                    sequence_phase_two.unsqueeze(-1),
                    candidate_ring_target,
                    first_sequence_target,
                )
            elif args.second_stage_mode == "flex_then_roll":
                first_sequence_target = entry_ring_target.clone()
                first_sequence_target[:, 1:4] = candidate_ring_target[:, 1:4]
                sequence_error = torch.amax(
                    torch.abs(direct.hand_qpos[:, 9:12] - first_sequence_target[:, 1:4]),
                    dim=-1,
                )
                sequence_phase_two |= second_stage & (
                    sequence_error <= args.sequence_joint_tolerance_rad
                )
                sequenced_ring_target = torch.where(
                    sequence_phase_two.unsqueeze(-1),
                    candidate_ring_target,
                    first_sequence_target,
                )
            elif args.second_stage_mode == "clearance_roll_flex":
                clearance_fractions = torch.tensor(
                    args.clearance_flex_fractions,
                    dtype=torch.float32,
                    device=direct.device,
                ).unsqueeze(0)
                clearance_target = entry_ring_target.clone()
                clearance_target[:, 1:4] = ring_source[:, 1:4] + clearance_fractions * (
                    ring_final[:, 1:4] - ring_source[:, 1:4]
                )
                clearance_target = torch.maximum(
                    torch.minimum(clearance_target, direct.joint_upper[:, 8:12]),
                    direct.joint_lower[:, 8:12],
                )
                clearance_error = torch.amax(
                    torch.abs(direct.hand_qpos[:, 9:12] - clearance_target[:, 1:4]),
                    dim=-1,
                )
                clearance_complete |= second_stage & (
                    clearance_error
                    <= (
                        args.clearance_joint_tolerance_rad
                        if args.clearance_joint_tolerance_rad is not None
                        else args.sequence_joint_tolerance_rad
                    )
                )
                roll_target = clearance_target.clone()
                roll_target[:, 0] = candidate_ring_target[:, 0]
                roll_error = torch.abs(direct.hand_qpos[:, 8] - roll_target[:, 0])
                sequence_phase_two |= clearance_complete & (
                    roll_error <= args.sequence_joint_tolerance_rad
                )
                first_sequence_target = torch.where(
                    clearance_complete.unsqueeze(-1), roll_target, clearance_target
                )
                sequence_error = torch.where(
                    clearance_complete, roll_error, clearance_error
                )
                sequenced_ring_target = torch.where(
                    sequence_phase_two.unsqueeze(-1),
                    candidate_ring_target,
                    first_sequence_target,
                )
            elif args.second_stage_mode in (
                "roll_proximal_then_distal",
                "roll_distal_then_proximal",
            ):
                roll_target = entry_ring_target.clone()
                roll_target[:, 0] = candidate_ring_target[:, 0]
                roll_error = torch.abs(direct.hand_qpos[:, 8] - roll_target[:, 0])
                roll_complete |= second_stage & (
                    roll_error <= args.sequence_joint_tolerance_rad
                )
                substage_target = roll_target.clone()
                if args.second_stage_mode == "roll_proximal_then_distal":
                    substage_target[:, 1] = candidate_ring_target[:, 1]
                    substage_error = torch.abs(
                        direct.hand_qpos[:, 9] - substage_target[:, 1]
                    )
                else:
                    substage_target[:, 2:4] = candidate_ring_target[:, 2:4]
                    substage_error = torch.amax(
                        torch.abs(direct.hand_qpos[:, 10:12] - substage_target[:, 2:4]),
                        dim=-1,
                    )
                substage_complete |= roll_complete & (
                    substage_error
                    <= (
                        args.substage_joint_tolerance_rad
                        if args.substage_joint_tolerance_rad is not None
                        else args.sequence_joint_tolerance_rad
                    )
                )
                sequence_phase_two |= substage_complete
                first_sequence_target = torch.where(
                    roll_complete.unsqueeze(-1), substage_target, roll_target
                )
                sequence_error = torch.where(
                    roll_complete, substage_error, roll_error
                )
                sequenced_ring_target = torch.where(
                    sequence_phase_two.unsqueeze(-1),
                    candidate_ring_target,
                    first_sequence_target,
                )
            elif args.second_stage_mode == "hold_entry":
                sequence_phase_two |= second_stage
                sequence_error = torch.zeros(num_envs, device=direct.device)
                sequenced_ring_target = entry_ring_target
            else:
                sequence_phase_two |= second_stage
                sequenced_ring_target = candidate_ring_target

            newly_sequence_complete = sequence_phase_two & (
                batch_sequence_complete_step < 0
            )
            batch_sequence_complete_step = torch.where(
                newly_sequence_complete,
                torch.full_like(batch_sequence_complete_step, step_index),
                batch_sequence_complete_step,
            )

            batch_min_sequence_error = torch.where(
                active & second_stage & ~sequence_phase_two,
                torch.minimum(batch_min_sequence_error, sequence_error),
                batch_min_sequence_error,
            ) if args.second_stage_mode != "simultaneous" else batch_min_sequence_error

            should_brake = (
                sequence_phase_two
                & (progress >= target_progress - args.brake_progress_margin)
                & (target_distance <= args.brake_distance_mm / 1000.0)
            )
            newly_braked = active & should_brake & ~brake_latched
            brake_targets[newly_braked] = direct.hand_qpos[newly_braked, 8:12]
            brake_latched |= should_brake

            ring_target = torch.where(
                second_stage.unsqueeze(-1), sequenced_ring_target, entry_ring_target
            )
            ring_target = torch.where(brake_latched.unsqueeze(-1), brake_targets, ring_target)
            actions = torch.zeros((num_envs, cfg.action_space), device=direct.device)
            actions[:, 8:12] = torch.clamp(
                (ring_target - direct.policy_targets[:, 8:12]) / increment, -1.0, 1.0
            )
            actions[:, 8:12] *= torch.where(
                sequence_phase_two,
                torch.full_like(sequence_phase_two, args.final_ring_action_scale, dtype=torch.float32),
                torch.ones_like(sequence_phase_two, dtype=torch.float32),
            ).unsqueeze(-1)
            if args.final_ring_force_limit_n is not None:
                ring_overload = sequence_phase_two & (
                    direct.contact_force_norm[:, 3] > args.final_ring_force_limit_n
                )
                actions[ring_overload, 8:12] = 0.0
                actions[ring_overload, 9] = -args.final_ring_backoff_action
            for column, joint_slice in enumerate(
                (slice(1, 4), slice(5, 8), slice(17, 20))
            ):
                actions[:, joint_slice] = support[:, column].unsqueeze(-1)
            for column, joint_id in enumerate((0, 4, 16)):
                actions[:, joint_id] = support_roll[:, column]
            actions[~active] = 0.0

            _, _, terminated, truncated, _ = env.step(actions)
            direct._compute_intermediate_values()
            eligible, metrics = direct._instantaneous_gate()
            _, progress, target_distance, _ = direct._ring_geometry()
            geometry = (
                (torch.abs(progress - target_progress) <= progress_tolerance)
                & (target_distance <= target_radius)
            )
            newly_axis_violation = (
                batch_first_axis_violation_step < 0
            ) & (
                metrics["axis_drift"]
                > torch.deg2rad(
                    torch.full_like(
                        metrics["axis_drift"], direct.cfg.p2_max_axis_drift_deg
                    )
                )
            )
            batch_first_axis_violation_step = torch.where(
                newly_axis_violation,
                torch.full_like(batch_first_axis_violation_step, step_index),
                batch_first_axis_violation_step,
            )
            newly_geometry = (batch_first_geometry_step < 0) & geometry
            batch_first_geometry_step = torch.where(
                newly_geometry,
                torch.full_like(batch_first_geometry_step, step_index),
                batch_first_geometry_step,
            )
            batch_eligible += (active & eligible).to(torch.int16)
            batch_geometry += (active & geometry).to(torch.int16)
            batch_geometry_triad += (
                active & geometry & (metrics["triad"] > 0.5)
            ).to(torch.int16)
            batch_geometry_axis += (
                active
                & geometry
                & (
                    metrics["axis_drift"]
                    <= torch.deg2rad(
                        torch.full_like(
                            metrics["axis_drift"], direct.cfg.p2_max_axis_drift_deg
                        )
                    )
                )
            ).to(torch.int16)
            batch_geometry_velocity += (
                active
                & geometry
                & (metrics["angular_speed"] <= direct.cfg.p2_max_angular_speed_rad_s)
                & (metrics["linear_speed"] <= direct.cfg.p2_max_linear_speed_m_s)
            ).to(torch.int16)
            batch_triad_steps += (active & (metrics["triad"] > 0.5)).to(torch.int16)
            batch_max_position_drift = torch.where(
                active,
                torch.maximum(batch_max_position_drift, metrics["position_drift"] * 1000.0),
                batch_max_position_drift,
            )
            batch_min_distance = torch.where(
                active, torch.minimum(batch_min_distance, target_distance * 1000.0), batch_min_distance
            )
            axis_drift_deg = metrics["axis_drift"] * 180.0 / torch.pi
            batch_max_axis_drift = torch.where(
                active,
                torch.maximum(batch_max_axis_drift, axis_drift_deg),
                batch_max_axis_drift,
            )
            batch_max_axis_before_final = torch.where(
                active & ~sequence_phase_two,
                torch.maximum(batch_max_axis_before_final, axis_drift_deg),
                batch_max_axis_before_final,
            )
            batch_min_axis_geometry = torch.where(
                active & geometry,
                torch.minimum(batch_min_axis_geometry, axis_drift_deg),
                batch_min_axis_geometry,
            )
            batch_min_axis_geometry_triad = torch.where(
                active & geometry & (metrics["triad"] > 0.5),
                torch.minimum(batch_min_axis_geometry_triad, axis_drift_deg),
                batch_min_axis_geometry_triad,
            )
            ring_force = direct.contact_force_norm[:, 3]
            batch_min_ring_force_geometry = torch.where(
                active & geometry,
                torch.minimum(batch_min_ring_force_geometry, ring_force),
                batch_min_ring_force_geometry,
            )
            batch_max_ring_force = torch.where(
                active,
                torch.maximum(batch_max_ring_force, ring_force),
                batch_max_ring_force,
            )
            batch_max_penetration = torch.where(
                active,
                torch.maximum(batch_max_penetration, direct.episode_max_penetration * 1000.0),
                batch_max_penetration,
            )
            done = active & (terminated | truncated)
            batch_success[done] = direct.last_completed_p1_p2_success[done]
            batch_drop[done] = direct.last_completed_p1_p2_drop[done]
            active &= ~done
            if not active.any():
                break

        # Any still-active row did not complete the success hold within the probe horizon.
        flat_valid = reset_slots < valid_count
        for env_id in torch.nonzero(flat_valid, as_tuple=False).squeeze(-1).cpu().tolist():
            reset_index = batch_start + int(reset_slots[env_id].item())
            candidate_index = int(candidate_ids[env_id].item())
            success_matrix[reset_index, candidate_index] = batch_success[env_id].cpu()
            drop_matrix[reset_index, candidate_index] = batch_drop[env_id].cpu()
            eligible_steps[reset_index, candidate_index] = batch_eligible[env_id].cpu()
            geometry_steps[reset_index, candidate_index] = batch_geometry[env_id].cpu()
            geometry_triad_steps[reset_index, candidate_index] = batch_geometry_triad[env_id].cpu()
            geometry_axis_steps[reset_index, candidate_index] = batch_geometry_axis[env_id].cpu()
            geometry_velocity_steps[reset_index, candidate_index] = batch_geometry_velocity[
                env_id
            ].cpu()
            second_stage_entered[reset_index, candidate_index] = second_stage[env_id].cpu()
            sequence_completed[reset_index, candidate_index] = sequence_phase_two[env_id].cpu()
            min_sequence_error_rad[reset_index, candidate_index] = (
                batch_min_sequence_error[env_id].cpu()
            )
            min_distance_mm[reset_index, candidate_index] = batch_min_distance[env_id].cpu()
            min_axis_drift_at_geometry_deg[reset_index, candidate_index] = (
                batch_min_axis_geometry[env_id].cpu()
            )
            min_axis_drift_at_geometry_triad_deg[reset_index, candidate_index] = (
                batch_min_axis_geometry_triad[env_id].cpu()
            )
            min_ring_force_at_geometry_n[reset_index, candidate_index] = (
                batch_min_ring_force_geometry[env_id].cpu()
            )
            max_ring_force_n[reset_index, candidate_index] = batch_max_ring_force[env_id].cpu()
            max_axis_drift_deg[reset_index, candidate_index] = batch_max_axis_drift[env_id].cpu()
            max_axis_drift_before_final_deg[reset_index, candidate_index] = (
                batch_max_axis_before_final[env_id].cpu()
            )
            triad_steps[reset_index, candidate_index] = batch_triad_steps[env_id].cpu()
            max_position_drift_mm[reset_index, candidate_index] = (
                batch_max_position_drift[env_id].cpu()
            )
            final_axis_cross[reset_index, candidate_index] = (
                batch_final_axis_cross[env_id].cpu()
            )
            second_stage_entry_step[reset_index, candidate_index] = batch_entry_step[
                env_id
            ].cpu()
            sequence_complete_step[reset_index, candidate_index] = (
                batch_sequence_complete_step[env_id].cpu()
            )
            first_axis_violation_step[reset_index, candidate_index] = (
                batch_first_axis_violation_step[env_id].cpu()
            )
            first_geometry_step[reset_index, candidate_index] = batch_first_geometry_step[
                env_id
            ].cpu()
            max_penetration_mm[reset_index, candidate_index] = batch_max_penetration[env_id].cpu()

        print(
            f"STATE_CONDITIONED_PROGRESS={min(batch_start + valid_count, num_reset_ids)}/{num_reset_ids}",
            flush=True,
        )

    candidate_scores = (
        success_matrix.to(torch.int32) * 1_000_000
        + eligible_steps.to(torch.int32) * 10_000
        + geometry_steps.to(torch.int32) * 100
        - torch.nan_to_num(min_distance_mm, nan=1000.0, posinf=1000.0).to(torch.int32)
        - drop_matrix.to(torch.int32) * 100_000
    )
    best_candidate_ids = torch.argmax(candidate_scores, dim=1)
    oracle_success = success_matrix.any(dim=1)
    selected_successes = success_matrix.gather(1, best_candidate_ids.unsqueeze(1)).squeeze(1)
    candidate_success_counts = success_matrix.sum(dim=0)
    top_candidate_order = torch.argsort(candidate_success_counts, descending=True)

    output_npz = resolve(args.npz_output)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        selected_local_ids=selected_local_ids.numpy(),
        selected_global_ids=selected_global_ids.numpy(),
        candidates=np.asarray([candidate[0] for candidate in candidates], dtype=np.float32),
        support_candidates=np.asarray(
            [candidate[1] for candidate in candidates], dtype=np.float32
        ),
        support_roll_candidates=np.asarray(
            [candidate[2] for candidate in candidates], dtype=np.float32
        ),
        success_matrix=success_matrix.numpy(),
        drop_matrix=drop_matrix.numpy(),
        eligible_steps=eligible_steps.numpy(),
        geometry_steps=geometry_steps.numpy(),
        geometry_triad_steps=geometry_triad_steps.numpy(),
        geometry_axis_steps=geometry_axis_steps.numpy(),
        geometry_velocity_steps=geometry_velocity_steps.numpy(),
        second_stage_entered=second_stage_entered.numpy(),
        sequence_completed=sequence_completed.numpy(),
        min_sequence_error_rad=min_sequence_error_rad.numpy(),
        min_distance_mm=min_distance_mm.numpy(),
        min_axis_drift_at_geometry_deg=min_axis_drift_at_geometry_deg.numpy(),
        min_axis_drift_at_geometry_triad_deg=min_axis_drift_at_geometry_triad_deg.numpy(),
        min_ring_force_at_geometry_n=min_ring_force_at_geometry_n.numpy(),
        max_ring_force_n=max_ring_force_n.numpy(),
        max_axis_drift_deg=max_axis_drift_deg.numpy(),
        max_axis_drift_before_final_deg=max_axis_drift_before_final_deg.numpy(),
        triad_steps=triad_steps.numpy(),
        max_position_drift_mm=max_position_drift_mm.numpy(),
        final_axis_cross=final_axis_cross.numpy(),
        second_stage_entry_step=second_stage_entry_step.numpy(),
        sequence_complete_step=sequence_complete_step.numpy(),
        first_axis_violation_step=first_axis_violation_step.numpy(),
        first_geometry_step=first_geometry_step.numpy(),
        max_penetration_mm=max_penetration_mm.numpy(),
        best_candidate_ids=best_candidate_ids.numpy(),
    )

    result = {
        "task": args.task,
        "split": "train",
        "test_split_accessed": False,
        "seed": args.seed,
        "reset_ids_evaluated": num_reset_ids,
        "candidate_count": num_candidates,
        "num_envs": num_envs,
        "episode_steps": args.episode_steps,
        "episode_length_s": float(cfg.episode_length_s),
        "brake_progress_margin": args.brake_progress_margin,
        "brake_distance_mm": args.brake_distance_mm,
        "entry_map": str(resolve(args.entry_map)) if args.entry_map is not None else None,
        "entry_progress": args.entry_progress if args.entry_map is not None else None,
        "entry_distance_mm": args.entry_distance_mm if args.entry_map is not None else None,
        "entry_hold_steps": args.entry_hold_steps if args.entry_map is not None else None,
        "second_stage_mode": args.second_stage_mode,
        "sequence_joint_tolerance_rad": args.sequence_joint_tolerance_rad,
        "substage_joint_tolerance_rad": args.substage_joint_tolerance_rad,
        "final_ring_action_scale": args.final_ring_action_scale,
        "final_ring_force_limit_n": args.final_ring_force_limit_n,
        "final_ring_backoff_action": args.final_ring_backoff_action,
        "clearance_flex_fractions": (
            list(args.clearance_flex_fractions)
            if args.second_stage_mode == "clearance_roll_flex"
            else None
        ),
        "clearance_joint_tolerance_rad": (
            args.clearance_joint_tolerance_rad
            if args.second_stage_mode == "clearance_roll_flex"
            else None
        ),
        "second_stage_entered_ids": int(second_stage_entered.any(dim=1).sum().item()),
        "sequence_completed_ids": int(sequence_completed.any(dim=1).sum().item()),
        "geometry_ids": int((geometry_steps > 0).any(dim=1).sum().item()),
        "geometry_triad_ids": int((geometry_triad_steps > 0).any(dim=1).sum().item()),
        "geometry_axis_ids": int((geometry_axis_steps > 0).any(dim=1).sum().item()),
        "oracle_success_ids": int(oracle_success.sum().item()),
        "oracle_success_rate": float(oracle_success.float().mean().item()),
        "selected_success_ids": int(selected_successes.sum().item()),
        "drop_any_ids": int(drop_matrix.any(dim=1).sum().item()),
        "invalid_count": 0,
        "npz": str(output_npz),
        "top_candidates": [
            {
                "candidate_id": int(candidate_id),
                "ring_joint_fractions": list(candidates[int(candidate_id)][0]),
                "support_actions_index_middle_thumb": list(candidates[int(candidate_id)][1]),
                "support_lateral_index_roll_middle_roll_thumb_yaw": list(
                    candidates[int(candidate_id)][2]
                ),
                "successful_reset_ids": int(candidate_success_counts[candidate_id].item()),
                "success_rate": float(candidate_success_counts[candidate_id].item() / num_reset_ids),
            }
            for candidate_id in top_candidate_order[:20]
        ],
        "best_candidate_histogram": {
            str(int(candidate_id)): int((best_candidate_ids == candidate_id).sum().item())
            for candidate_id in torch.unique(best_candidate_ids)
        },
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_STATE_CONDITIONED_SEARCH=" + json.dumps(result, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
