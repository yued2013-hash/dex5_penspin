from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(
    description="Search short ring-led corrections from real terminal P1-to-P2 states."
)
parser.add_argument(
    "--task", default="Dex5-P1P2-P1617Route539LocalArrival120HzH1-Direct-v0"
)
parser.add_argument("--source-bundle", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--teacher-output", required=True, type=Path)
parser.add_argument("--population", type=int, default=64)
parser.add_argument("--states-per-candidate", type=int, default=8)
parser.add_argument(
    "--replicate-source-row",
    action="store_true",
    help="Evaluate every candidate on repeated, independently perturbed resets of one source row.",
)
parser.add_argument(
    "--robust-score-quantile",
    type=float,
    default=None,
    help="Rank non-paired candidates by this replica-score quantile instead of the mean.",
)
parser.add_argument("--iterations", type=int, default=8)
parser.add_argument("--elite-fraction", type=float, default=0.20)
parser.add_argument("--segment-steps", type=int, nargs=3, default=(4, 4, 10))
parser.add_argument("--distance-limit-mm", type=float, default=2.431)
parser.add_argument("--seed", type=int, default=2821)
parser.add_argument(
    "--paired-per-state",
    action="store_true",
    help="Optimize one independent CEM route for every selected source state.",
)
parser.add_argument(
    "--selection",
    choices=("random", "semantic-boundary", "ordered"),
    default="random",
    help="Select random states or prioritize near-semantic-boundary failures.",
)
parser.add_argument("--selection-offset", type=int, default=0)
parser.add_argument(
    "--unique-group-key",
    default="",
    help="Optional source-bundle key used to keep one ranked state per group.",
)
parser.add_argument("--post-success-hold-steps", type=int, default=0)
parser.add_argument(
    "--gate-start-step",
    type=int,
    default=0,
    help="Ignore all success gates before this zero-based rollout step.",
)
parser.add_argument(
    "--closed-loop-hold",
    action="store_true",
    help="Keep tracking the final waypoint after semantic entry instead of sending zero action.",
)
parser.add_argument("--required-consecutive-steps", type=int, default=1)
parser.add_argument(
    "--acceptance-gate",
    choices=("semantic", "physical_contact", "wrench_feasibility", "stable_capture"),
    default="semantic",
    help="Hard gate optimized by CEM; both gate outcomes are always reported.",
)
parser.add_argument("--offset-limit-rad", type=float, default=0.085)
parser.add_argument("--initial-std-rad", type=float, default=0.030)
parser.add_argument(
    "--initial-mean-offsets-rad",
    type=float,
    nargs=12,
    default=None,
    metavar=(
        "W1Q0", "W1Q1", "W1Q2", "W1Q3",
        "W2Q0", "W2Q1", "W2Q2", "W2Q3",
        "W3Q0", "W3Q1", "W3Q2", "W3Q3",
    ),
    help="Optional three-waypoint prior used only to initialize CEM.",
)
parser.add_argument(
    "--ring-action-limit",
    type=float,
    default=1.0,
    help="Absolute normalized-action limit for each ring joint during correction.",
)
parser.add_argument(
    "--tim-action-scale",
    type=float,
    default=1.0,
    help="Scale contact-deficit TIM recovery; use <=0.25 for tiny compensation.",
)
parser.add_argument(
    "--ring-target-force-n",
    type=float,
    default=0.08,
    help="Soft target for physical-contact CEM scoring; hard force limits remain unchanged.",
)
parser.add_argument("--wrench-normal-force-min-n", type=float, default=0.04)
parser.add_argument("--wrench-normal-force-max-n", type=float, default=0.05)
parser.add_argument("--wrench-tangential-force-max-n", type=float, default=0.002)
parser.add_argument("--wrench-slip-speed-max-m-s", type=float, default=0.005)
parser.add_argument(
    "--wrench-friction-coefficient",
    type=float,
    default=0.45,
    help="Coefficient used only to report and score Coulomb friction margin.",
)
parser.add_argument("--stable-capture-min-normal-force-n", type=float, default=0.03)
parser.add_argument("--stable-capture-max-force-delta-n", type=float, default=0.01)
parser.add_argument("--stable-capture-max-slip-m-s", type=float, default=0.04)
parser.add_argument("--stable-capture-max-linear-speed-m-s", type=float, default=0.03)
parser.add_argument("--stable-capture-max-angular-speed-rad-s", type=float, default=2.0)
parser.add_argument(
    "--stable-capture-max-ring-target-error-rad",
    type=float,
    default=0.25,
    help="Maximum L2 error between the ring position target and measured ring joints.",
)
parser.add_argument(
    "--stable-capture-max-ring-joint-error-rad",
    type=float,
    default=0.116,
    help="Per-joint preload limit; 0.116 rad corresponds to 0.93 Nm at stiffness 8.",
)
parser.add_argument("--stable-capture-window-steps", type=int, default=6)
parser.add_argument(
    "--ring-force-servo",
    action="store_true",
    help="Apply a bounded ring joint-target correction from normal force and separation speed.",
)
parser.add_argument("--ring-force-servo-kp-rad-per-n", type=float, default=0.0)
parser.add_argument("--ring-force-servo-kd-rad-s-per-m", type=float, default=0.0)
parser.add_argument("--ring-force-servo-max-offset-rad", type=float, default=0.006)
parser.add_argument("--ring-force-servo-start-step", type=int, default=2)
parser.add_argument("--ring-force-servo-filter-alpha", type=float, default=0.35)
parser.add_argument(
    "--ring-force-servo-direction",
    type=float,
    nargs=4,
    default=(-0.671, 0.139, -1.0, -0.313),
    metavar=("Q0", "Q1", "Q2", "Q3"),
)
parser.add_argument(
    "--ring-force-servo-direction-grid",
    type=float,
    nargs="+",
    default=None,
    help="Optional flattened N x 4 direction grid, assigned cyclically across environments.",
)
parser.add_argument(
    "--ring-slip-servo",
    action="store_true",
    help="Cancel signed tangential contact velocity through the ring fingertip Jacobian.",
)
parser.add_argument("--ring-slip-servo-gain", type=float, default=1.0)
parser.add_argument("--ring-slip-servo-horizon-s", type=float, default=0.02)
parser.add_argument("--ring-slip-servo-damping", type=float, default=0.01)
parser.add_argument("--ring-slip-servo-max-offset-rad", type=float, default=0.004)
parser.add_argument("--ring-slip-servo-start-step", type=int, default=2)
parser.add_argument(
    "--ring-slip-servo-preserve-normal",
    action="store_true",
    help="Project tangential correction into the contact-normal Jacobian nullspace.",
)
parser.add_argument(
    "--ring-actuator-stiffness",
    type=float,
    default=None,
    help="Split the ring actuator and override only its implicit position stiffness.",
)
parser.add_argument(
    "--ring-actuator-damping",
    type=float,
    default=None,
    help="Split the ring actuator and override only its implicit velocity damping.",
)
parser.add_argument(
    "--fixed-parameters-json",
    type=Path,
    default=None,
    help="Replay previously searched best_offsets_rad instead of running CEM.",
)
parser.add_argument(
    "--fixed-parameters-row",
    type=int,
    default=0,
    help="Row in best_offsets_rad used for a fixed-parameter replay.",
)
parser.add_argument(
    "--save-all-trajectories",
    action="store_true",
    help="Store selected diagnostic replays even when they miss the hard gate.",
)
parser.add_argument(
    "--save-all-population",
    action="store_true",
    help="Retain every population replica, useful for reset-perturbation audits.",
)
for noise_name in (
    "reset_joint_pos_noise",
    "reset_joint_vel_noise",
    "reset_pen_pos_noise",
    "reset_pen_vel_noise",
    "reset_pen_rot_noise_rad",
):
    parser.add_argument(f"--{noise_name.replace('_', '-')}", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from dex5_penspin.isaac.dex5_penspin_isaaclab.tasks.index_near_surface.index_near_surface_physics_env import (  # noqa: E402
    DEX5_JOINT_ORDER,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


LOW = np.asarray([-0.085] * 12, dtype=np.float32)
HIGH = np.asarray([0.085] * 12, dtype=np.float32)
RING_JOINT_NAMES = set(DEX5_JOINT_ORDER[8:12])
INITIAL_STD = np.asarray([0.030] * 12, dtype=np.float32)
TIM_GAINS = torch.tensor(
    [[0.0, 0.55, 0.90], [0.0, 0.55, 0.90], [0.0, 0.55, 0.90]],
    dtype=torch.float32,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def semantic_cost(direct, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
    axial, lateral, up = direct._ring_axis_features().unbind(dim=-1)
    gap = direct.tip_surface_dist[:, 3] - direct.fingertip_radii[0, 3]

    def excess(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        return torch.relu(lower - value) + torch.relu(value - upper)

    return (
        1200.0 * excess(axial, -0.060, -0.020)
        + 1600.0 * excess(lateral, -0.010, 0.026)
        + 1600.0 * excess(up, -0.026, -0.004)
        + 1600.0 * torch.relu(torch.abs(lateral) + 0.001 + up)
        + 1800.0 * excess(gap, -0.0006, 0.0030)
        + 8.0 * (metrics["triad"] < 0.5).float()
    )


def reduce_mean(values: torch.Tensor, candidate_ids: torch.Tensor, count: int) -> torch.Tensor:
    result = torch.zeros(count, device=values.device)
    samples = torch.zeros(count, device=values.device)
    result.scatter_add_(0, candidate_ids, values.float())
    samples.scatter_add_(0, candidate_ids, torch.ones_like(values, dtype=torch.float32))
    return result / samples.clamp_min(1.0)


def rollout(
    direct,
    env,
    parameters: torch.Tensor,
    candidate_ids: torch.Tensor,
    forced_ids: torch.Tensor,
    capture: bool = False,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    num_envs = direct.num_envs
    num_candidates = int(candidate_ids.max().item()) + 1
    env_ids = torch.arange(num_envs, device=direct.device)
    direct.forced_bank_ids = forced_ids
    direct._reset_idx(env_ids)
    direct._compute_intermediate_values()
    _, initial_metrics = direct._instantaneous_gate()
    initial_cost = semantic_cost(direct, initial_metrics)
    source_target = direct.policy_targets[:, 8:12].clone()
    offsets = parameters.reshape(num_envs, 3, 4)
    waypoints = source_target[:, None, :] + offsets
    segment_steps = tuple(int(value) for value in args.segment_steps)
    total_steps = sum(segment_steps) + max(int(args.post_success_hold_steps), 0)
    increment = 0.25 * float(direct.cfg.action_scale)
    best_score = torch.full((num_envs,), -float("inf"), device=direct.device)
    best_distance = torch.full((num_envs,), float("inf"), device=direct.device)
    best_cost = initial_cost.clone()
    best_under = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
    best_semantic = torch.zeros_like(best_under)
    best_feature_bounds = torch.zeros_like(best_under)
    best_gap_pass = torch.zeros_like(best_under)
    best_gap = torch.full((num_envs,), float("inf"), device=direct.device)
    accepted = torch.zeros_like(best_under)
    semantic_accepted = torch.zeros_like(best_under)
    physical_accepted = torch.zeros_like(best_under)
    wrench_accepted = torch.zeros_like(best_under)
    stable_capture_accepted = torch.zeros_like(best_under)
    consecutive = torch.zeros(num_envs, dtype=torch.long, device=direct.device)
    max_consecutive = torch.zeros_like(consecutive)
    semantic_consecutive = torch.zeros_like(consecutive)
    physical_consecutive = torch.zeros_like(consecutive)
    wrench_consecutive = torch.zeros_like(consecutive)
    stable_capture_consecutive = torch.zeros_like(consecutive)
    semantic_max_consecutive = torch.zeros_like(consecutive)
    physical_max_consecutive = torch.zeros_like(consecutive)
    wrench_max_consecutive = torch.zeros_like(consecutive)
    stable_capture_max_consecutive = torch.zeros_like(consecutive)
    ring_contact_seen = torch.zeros_like(best_under)
    distance_seen = torch.zeros_like(best_under)
    contact_and_distance_seen = torch.zeros_like(best_under)
    physical_valid_seen = torch.zeros_like(best_under)
    hard_failure = torch.zeros_like(best_under)
    max_penetration = torch.zeros(num_envs, device=direct.device)
    max_position_drift = torch.zeros(num_envs, device=direct.device)
    max_axis_drift = torch.zeros(num_envs, device=direct.device)
    observation_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    accepted_rows: list[torch.Tensor] = []
    target_condition_rows: list[torch.Tensor] = []
    condition_delta_rows: list[torch.Tensor] = []
    semantic_margin_rows: list[torch.Tensor] = []
    hold_latched_rows: list[torch.Tensor] = []
    state_qpos_rows: list[torch.Tensor] = []
    state_qvel_rows: list[torch.Tensor] = []
    state_target_rows: list[torch.Tensor] = []
    contact_force_rows: list[torch.Tensor] = []
    ring_feature_rows: list[torch.Tensor] = []
    pen_linear_velocity_rows: list[torch.Tensor] = []
    pen_angular_velocity_rows: list[torch.Tensor] = []
    contact_force_vector_rows: list[torch.Tensor] = []
    contact_normal_force_rows: list[torch.Tensor] = []
    contact_tangential_force_rows: list[torch.Tensor] = []
    contact_normal_speed_rows: list[torch.Tensor] = []
    contact_slip_speed_rows: list[torch.Tensor] = []
    friction_margin_rows: list[torch.Tensor] = []
    tim_force_rows: list[torch.Tensor] = []
    tim_torque_rows: list[torch.Tensor] = []
    ring_force_on_pen_rows: list[torch.Tensor] = []
    ring_torque_rows: list[torch.Tensor] = []
    tim_gravity_residual_rows: list[torch.Tensor] = []
    total_gravity_residual_rows: list[torch.Tensor] = []
    tim_moment_arm_rows: list[torch.Tensor] = []
    pen_com_position_rows: list[torch.Tensor] = []
    pen_com_velocity_rows: list[torch.Tensor] = []
    ring_force_servo_offset_rows: list[torch.Tensor] = []
    ring_slip_servo_offset_rows: list[torch.Tensor] = []
    ring_tangential_velocity_vector_rows: list[torch.Tensor] = []
    ring_tangential_velocity_component_rows: list[torch.Tensor] = []
    ring_contact_frame_rows: list[torch.Tensor] = []
    ring_target_error_rows: list[torch.Tensor] = []
    ring_joint_target_error_rows: list[torch.Tensor] = []
    hold_latched = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
    previous_condition, _, _ = direct._terminal_condition_metrics()
    previous_condition = previous_condition.detach().clone()
    filtered_ring_normal_force = direct.contact_normal_force_signed[:, 3].clamp_min(0.0)
    previous_ring_normal_force = filtered_ring_normal_force.clone()
    stable_force_history: list[torch.Tensor] = []
    if args.ring_force_servo_direction_grid is not None:
        if len(args.ring_force_servo_direction_grid) % 4 != 0:
            raise ValueError("ring force-servo direction grid must contain N x 4 values")
        servo_directions = torch.as_tensor(
            args.ring_force_servo_direction_grid,
            dtype=torch.float32,
            device=direct.device,
        ).reshape(-1, 4)
    else:
        servo_directions = torch.as_tensor(
            args.ring_force_servo_direction,
            dtype=torch.float32,
            device=direct.device,
        ).reshape(1, 4)
    servo_directions = servo_directions / servo_directions.abs().amax(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    servo_direction_ids = torch.arange(num_envs, device=direct.device) % len(
        servo_directions
    )
    servo_direction = servo_directions[servo_direction_ids]

    segment_start = source_target
    elapsed = 0
    for step in range(total_steps):
        if step >= segment_steps[0] + segment_steps[1]:
            segment = 2
            local_step = step - segment_steps[0] - segment_steps[1]
        elif step >= segment_steps[0]:
            segment = 1
            local_step = step - segment_steps[0]
        else:
            segment = 0
            local_step = step
        if segment > 0 and local_step == 0:
            segment_start = waypoints[:, segment - 1]
        tau = smoothstep(
            torch.full(
                (num_envs,),
                float(local_step + 1) / float(segment_steps[segment]),
                device=direct.device,
            )
        )
        desired = segment_start + tau.unsqueeze(-1) * (
            waypoints[:, segment] - segment_start
        )
        servo_offset = torch.zeros(num_envs, device=direct.device)
        if args.ring_force_servo and step >= int(args.ring_force_servo_start_step):
            alpha = float(args.ring_force_servo_filter_alpha)
            alpha = min(max(alpha, 0.0), 1.0)
            measured_force = direct.contact_normal_force_signed[:, 3].clamp_min(0.0)
            filtered_ring_normal_force = (
                alpha * measured_force + (1.0 - alpha) * filtered_ring_normal_force
            )
            force_error = float(args.ring_target_force_n) - filtered_ring_normal_force
            separation_speed = direct.contact_normal_relative_speed[:, 3]
            servo_offset = (
                float(args.ring_force_servo_kp_rad_per_n) * force_error
                + float(args.ring_force_servo_kd_rad_s_per_m) * separation_speed
            ).clamp(
                -float(args.ring_force_servo_max_offset_rad),
                float(args.ring_force_servo_max_offset_rad),
            )
            desired = desired + servo_offset.unsqueeze(-1) * servo_direction
        slip_servo_offset = torch.zeros((num_envs, 4), device=direct.device)
        if args.ring_slip_servo and step >= int(args.ring_slip_servo_start_step):
            ring_body_id = int(direct.fingertip_body_ids[3])
            ring_jacobian_body_id = ring_body_id - 1
            ring_joint_ids = direct.dex5_joint_ids[8:12]
            body_jacobian = direct.hand.root_physx_view.get_jacobians()[
                :, ring_jacobian_body_id, :, ring_joint_ids
            ]
            body_to_tip = (
                direct.fingertip_pos[:, 3]
                - direct.hand.data.body_pos_w[:, ring_body_id]
            )
            angular_columns = body_jacobian[:, 3:, :].transpose(1, 2)
            tip_offset = body_to_tip.unsqueeze(1).expand(-1, 4, -1)
            offset_velocity_jacobian = torch.linalg.cross(
                angular_columns, tip_offset, dim=-1
            ).transpose(1, 2)
            fingertip_jacobian = body_jacobian[:, :3, :] + offset_velocity_jacobian
            tangent_basis = torch.stack(
                (
                    direct.contact_tangent_axis_w[:, 3],
                    direct.contact_binormal_axis_w[:, 3],
                ),
                dim=1,
            )
            tangential_jacobian = torch.bmm(tangent_basis, fingertip_jacobian)
            tangential_velocity = direct.contact_tangential_velocity_components[:, 3]
            damping = float(args.ring_slip_servo_damping)
            nullspace = torch.eye(
                4, dtype=fingertip_jacobian.dtype, device=fingertip_jacobian.device
            ).unsqueeze(0).expand(num_envs, -1, -1)
            if args.ring_slip_servo_preserve_normal:
                normal_jacobian = torch.bmm(
                    direct.contact_radial_direction_w[:, 3].unsqueeze(1),
                    fingertip_jacobian,
                )
                normal_inverse = normal_jacobian.transpose(1, 2) / (
                    torch.bmm(normal_jacobian, normal_jacobian.transpose(1, 2))
                    + damping * damping
                )
                nullspace = nullspace - torch.bmm(normal_inverse, normal_jacobian)
                tangential_jacobian = torch.bmm(tangential_jacobian, nullspace)
            gram = torch.bmm(
                tangential_jacobian, tangential_jacobian.transpose(1, 2)
            )
            gram = gram + (damping * damping) * torch.eye(
                2, dtype=gram.dtype, device=gram.device
            ).unsqueeze(0)
            correction_velocity = torch.bmm(
                nullspace,
                torch.bmm(
                    tangential_jacobian.transpose(1, 2),
                    torch.linalg.solve(gram, -tangential_velocity.unsqueeze(-1)),
                ),
            ).squeeze(-1)
            slip_servo_offset = (
                float(args.ring_slip_servo_gain)
                * float(args.ring_slip_servo_horizon_s)
                * correction_velocity
            ).clamp(
                -float(args.ring_slip_servo_max_offset_rad),
                float(args.ring_slip_servo_max_offset_rad),
            )
            contact_active = (
                direct.contact_force_norm[:, 3] >= direct.cfg.p2_contact_force_n
            )
            slip_servo_offset = torch.where(
                contact_active.unsqueeze(-1),
                slip_servo_offset,
                torch.zeros_like(slip_servo_offset),
            )
            desired = desired + slip_servo_offset
        actions = torch.zeros((num_envs, direct.cfg.action_space), device=direct.device)
        actions[:, 8:12] = torch.clamp(
            (desired - direct.policy_targets[:, 8:12]) / increment,
            -float(args.ring_action_limit),
            float(args.ring_action_limit),
        )
        target_force = 0.06
        deficit = torch.clamp(
            (target_force - direct.contact_force_norm[:, :3]) / target_force, 0.0, 1.0
        )
        gains = TIM_GAINS.to(direct.device)
        for finger, joint_slice in enumerate((slice(17, 20), slice(1, 4), slice(5, 8))):
            actions[:, joint_slice] = (
                float(args.tim_action_scale)
                * deficit[:, finger].unsqueeze(-1)
                * gains[finger]
            )
        if not args.closed_loop_hold:
            actions[hold_latched] = 0.0
        if capture:
            observation_rows.append(direct._get_observations()["policy"].detach().clone())
            action_rows.append(actions.detach().clone())
            feature = direct._ring_axis_features()
            feature_error = (
                (direct.nearest_p2_target_center - feature) / 0.010
            ).clamp(-3.0, 3.0)
            ring_gap = direct.tip_surface_dist[:, 3] - direct.fingertip_radii[0, 3]
            gap_error = ((0.0012 - ring_gap) / 0.003).clamp(-2.0, 2.0)
            target_condition_rows.append(
                torch.cat((feature_error, gap_error.unsqueeze(-1)), dim=-1)
                .detach()
                .clone()
            )
            condition_delta_rows.append(
                (target_condition_rows[-1] - previous_condition).detach().clone()
            )
            _, semantic_margin, _ = direct._terminal_condition_metrics()
            semantic_margin_rows.append(semantic_margin.detach().clone())
            hold_latched_rows.append(hold_latched.detach().clone())
            previous_condition = target_condition_rows[-1]
        _, _, terminated, truncated, _ = env.step(actions)
        direct._compute_intermediate_values()
        _, metrics = direct._instantaneous_gate()
        distance = metrics["target_distance"]
        cost = semantic_cost(direct, metrics)
        under = metrics["manifold_under"] > 0.5
        semantic = metrics["manifold_semantic"] > 0.5
        feature_bounds = metrics["manifold_feature_bounds"] > 0.5
        gap_pass = metrics["manifold_gap_gate"] > 0.5
        ring_gap = direct.tip_surface_dist[:, 3] - direct.fingertip_radii[0, 3]
        physical_valid = (
            (metrics["triad"] > 0.5)
            & (metrics["position_drift"] <= 0.010)
            & (metrics["axis_drift"] <= torch.deg2rad(torch.full_like(distance, 8.0)))
            & (metrics["angular_speed"] <= direct.cfg.p2_max_angular_speed_rad_s)
            & (metrics["linear_speed"] <= direct.cfg.p2_max_linear_speed_m_s)
            & (direct.episode_max_penetration <= 0.0006)
            & (metrics["ring_force_share"] <= direct.cfg.p2_max_ring_force_share)
            & (direct.contact_force_norm[:, 3] <= direct.cfg.p2_max_ring_force_n)
            & (direct.contact_force_norm[:, 4] <= direct.cfg.p2_max_pinky_force_n)
            & (metrics["failed"] < 0.5)
            & torch.isfinite(direct.hand_qpos).all(dim=-1)
            & torch.isfinite(direct.pen_pos).all(dim=-1)
        )
        ring_contact = direct.contact_force_norm[:, 3] > direct.cfg.p2_contact_force_n
        ring_force = direct.contact_force_norm[:, 3]
        ring_normal_force = direct.contact_normal_force_signed[:, 3].clamp_min(0.0)
        ring_tangential_force = direct.contact_tangential_force_norm[:, 3]
        ring_slip_speed = direct.contact_tangential_slip_speed[:, 3]
        ring_normal_speed = direct.contact_normal_relative_speed[:, 3]
        friction_margin = (
            float(args.wrench_friction_coefficient) * ring_normal_force
            - ring_tangential_force
        )
        ring_joint_target_error = torch.abs(
            direct.policy_targets[:, 8:12] - direct.hand_qpos[:, 8:12]
        )
        ring_target_error = torch.linalg.vector_norm(ring_joint_target_error, dim=-1)
        force_delta = torch.abs(ring_normal_force - previous_ring_normal_force)
        stable_force_history.append(ring_normal_force)
        stable_window_steps = max(int(args.stable_capture_window_steps), 1)
        stable_force_window = torch.stack(
            stable_force_history[-stable_window_steps:], dim=1
        )
        stable_window_min_force = stable_force_window.amin(dim=1)
        stable_window_force_variance = stable_force_window.var(dim=1, unbiased=False)
        semantic_qualifies = (
            physical_valid
            & under
            & semantic
            & (distance <= float(args.distance_limit_mm) / 1000.0)
        )
        physical_qualifies = (
            physical_valid
            & ring_contact
            & (distance <= float(args.distance_limit_mm) / 1000.0)
        )
        wrench_qualifies = (
            physical_valid
            & ring_contact
            & (ring_normal_force >= float(args.wrench_normal_force_min_n))
            & (ring_normal_force <= float(args.wrench_normal_force_max_n))
            & (ring_tangential_force <= float(args.wrench_tangential_force_max_n))
            & (ring_slip_speed <= float(args.wrench_slip_speed_max_m_s))
            & (friction_margin > 0.0)
        )
        stable_capture_qualifies = (
            physical_valid
            & ring_contact
            & (
                ring_normal_force
                >= float(args.stable_capture_min_normal_force_n)
            )
            & (force_delta <= float(args.stable_capture_max_force_delta_n))
            & (ring_slip_speed <= float(args.stable_capture_max_slip_m_s))
            & (
                metrics["linear_speed"]
                <= float(args.stable_capture_max_linear_speed_m_s)
            )
            & (
                metrics["angular_speed"]
                <= float(args.stable_capture_max_angular_speed_rad_s)
            )
            & (
                ring_target_error
                <= float(args.stable_capture_max_ring_target_error_rad)
            )
            & (
                ring_joint_target_error.amax(dim=-1)
                <= float(args.stable_capture_max_ring_joint_error_rad)
            )
            & (friction_margin > 0.0)
        )
        gate_enabled = step >= int(args.gate_start_step)
        if not gate_enabled:
            semantic_qualifies = torch.zeros_like(semantic_qualifies)
            physical_qualifies = torch.zeros_like(physical_qualifies)
            wrench_qualifies = torch.zeros_like(wrench_qualifies)
            stable_capture_qualifies = torch.zeros_like(stable_capture_qualifies)
        ring_contact_seen |= ring_contact
        distance_seen |= distance <= float(args.distance_limit_mm) / 1000.0
        contact_and_distance_seen |= ring_contact & (
            distance <= float(args.distance_limit_mm) / 1000.0
        )
        physical_valid_seen |= physical_valid
        if args.acceptance_gate == "semantic":
            qualifies = semantic_qualifies
        elif args.acceptance_gate == "physical_contact":
            qualifies = physical_qualifies
        elif args.acceptance_gate == "wrench_feasibility":
            qualifies = wrench_qualifies
        else:
            qualifies = stable_capture_qualifies
        consecutive = torch.where(qualifies, consecutive + 1, torch.zeros_like(consecutive))
        max_consecutive = torch.maximum(max_consecutive, consecutive)
        semantic_consecutive = torch.where(
            semantic_qualifies,
            semantic_consecutive + 1,
            torch.zeros_like(semantic_consecutive),
        )
        physical_consecutive = torch.where(
            physical_qualifies,
            physical_consecutive + 1,
            torch.zeros_like(physical_consecutive),
        )
        semantic_max_consecutive = torch.maximum(
            semantic_max_consecutive, semantic_consecutive
        )
        physical_max_consecutive = torch.maximum(
            physical_max_consecutive, physical_consecutive
        )
        wrench_consecutive = torch.where(
            wrench_qualifies,
            wrench_consecutive + 1,
            torch.zeros_like(wrench_consecutive),
        )
        wrench_max_consecutive = torch.maximum(
            wrench_max_consecutive, wrench_consecutive
        )
        stable_capture_consecutive = torch.where(
            stable_capture_qualifies,
            stable_capture_consecutive + 1,
            torch.zeros_like(stable_capture_consecutive),
        )
        stable_capture_max_consecutive = torch.maximum(
            stable_capture_max_consecutive, stable_capture_consecutive
        )
        hold_quality = 240.0 * torch.clamp(
            consecutive.float() / max(int(args.required_consecutive_steps), 1),
            0.0,
            1.0,
        )
        if args.acceptance_gate == "stable_capture":
            hold_fraction = torch.clamp(
                stable_capture_consecutive.float()
                / max(int(args.required_consecutive_steps), 1),
                0.0,
                1.0,
            )
            hold_quality = (
                1200.0 * hold_fraction
                + 3000.0
                * (
                    stable_capture_consecutive
                    >= int(args.required_consecutive_steps)
                ).float()
            )
        if args.acceptance_gate == "semantic":
            gate_quality = (
                280.0 * semantic.float()
                + 140.0 * under.float()
                + 140.0 * feature_bounds.float()
                + 180.0 * gap_pass.float()
                + 80.0 * metrics["manifold_semantic_soft"]
                + 24.0 * (initial_cost - cost)
            )
        elif args.acceptance_gate == "physical_contact":
            # Physical audit deliberately excludes every semantic-box term.
            # Pen speed is the current slip proxy because the model does not
            # expose a reliable tangential contact-point velocity.
            gate_quality = (
                300.0 * ring_contact.float()
                + 180.0
                * torch.exp(
                    -torch.abs(ring_normal_force - float(args.ring_target_force_n))
                    / max(float(args.ring_target_force_n), 0.02)
                )
                + 120.0 * torch.exp(-ring_slip_speed / 0.04)
                + 60.0 * torch.exp(-torch.abs(ring_normal_speed) / 0.02)
                + 100.0 * metrics["triad"]
                + 40.0
                * torch.exp(-metrics["linear_speed"] / 0.08)
                + 40.0
                * torch.exp(-metrics["angular_speed"] / 3.0)
            )
        elif args.acceptance_gate == "wrench_feasibility":
            force_center = 0.5 * (
                float(args.wrench_normal_force_min_n)
                + float(args.wrench_normal_force_max_n)
            )
            force_half_width = max(
                0.5
                * (
                    float(args.wrench_normal_force_max_n)
                    - float(args.wrench_normal_force_min_n)
                ),
                0.0025,
            )
            gate_quality = (
                320.0 * ring_contact.float()
                + 220.0
                * torch.exp(-torch.abs(ring_normal_force - force_center) / force_half_width)
                + 180.0
                * torch.exp(
                    -ring_tangential_force
                    / max(float(args.wrench_tangential_force_max_n), 0.0005)
                )
                + 220.0
                * torch.exp(
                    -ring_slip_speed
                    / max(float(args.wrench_slip_speed_max_m_s), 0.002)
                )
                + 80.0 * torch.tanh(torch.clamp_min(friction_margin, 0.0) / 0.01)
                + 100.0 * metrics["triad"]
                + 50.0 * torch.exp(-metrics["linear_speed"] / 0.04)
                + 50.0 * torch.exp(-metrics["angular_speed"] / 2.0)
            )
        else:
            force_scale = max(float(args.stable_capture_min_normal_force_n), 0.01)
            variance_scale = max(
                float(args.stable_capture_max_force_delta_n) ** 2, 1.0e-6
            )
            gate_quality = (
                320.0 * ring_contact.float()
                + 260.0 * torch.clamp(stable_window_min_force / force_scale, 0.0, 1.5)
                + 180.0 * torch.exp(-stable_window_force_variance / variance_scale)
                + 140.0
                * torch.exp(
                    -force_delta
                    / max(float(args.stable_capture_max_force_delta_n), 0.002)
                )
                + 160.0
                * torch.exp(
                    -ring_slip_speed
                    / max(float(args.stable_capture_max_slip_m_s), 0.01)
                )
                + 100.0 * torch.exp(
                    -metrics["linear_speed"]
                    / max(float(args.stable_capture_max_linear_speed_m_s), 0.01)
                )
                + 100.0 * torch.exp(
                    -metrics["angular_speed"]
                    / max(float(args.stable_capture_max_angular_speed_rad_s), 0.5)
                )
                + 220.0 * torch.exp(
                    -ring_target_error
                    / max(float(args.stable_capture_max_ring_target_error_rad), 0.01)
                )
                + 220.0 * torch.exp(
                    -ring_joint_target_error.amax(dim=-1)
                    / max(float(args.stable_capture_max_ring_joint_error_rad), 0.01)
                )
                + 80.0 * torch.tanh(torch.clamp_min(friction_margin, 0.0) / 0.01)
                + 80.0 * metrics["triad"]
            )
        distance_penalty = (
            torch.zeros_like(distance)
            if args.acceptance_gate in ("wrench_feasibility", "stable_capture")
            else 35.0 * distance * 1000.0
        )
        quality = (
            1400.0 * qualifies.float()
            + hold_quality
            + gate_quality
            - distance_penalty
            - 500.0 * (~physical_valid).float()
        )
        improved = quality > best_score
        best_score = torch.where(improved, quality, best_score)
        best_distance = torch.where(improved, distance, best_distance)
        best_cost = torch.where(improved, cost, best_cost)
        best_under = torch.where(improved, under, best_under)
        best_semantic = torch.where(improved, semantic, best_semantic)
        best_feature_bounds = torch.where(improved, feature_bounds, best_feature_bounds)
        best_gap_pass = torch.where(improved, gap_pass, best_gap_pass)
        best_gap = torch.where(improved, ring_gap, best_gap)
        accepted |= max_consecutive >= int(args.required_consecutive_steps)
        semantic_accepted |= (
            semantic_max_consecutive >= int(args.required_consecutive_steps)
        )
        physical_accepted |= (
            physical_max_consecutive >= int(args.required_consecutive_steps)
        )
        wrench_accepted |= (
            wrench_max_consecutive >= int(args.required_consecutive_steps)
        )
        stable_capture_accepted |= (
            stable_capture_max_consecutive >= int(args.required_consecutive_steps)
        )
        hold_latched |= qualifies
        hard_failure |= ~physical_valid
        max_penetration = torch.maximum(max_penetration, direct.episode_max_penetration)
        max_position_drift = torch.maximum(max_position_drift, metrics["position_drift"])
        max_axis_drift = torch.maximum(max_axis_drift, metrics["axis_drift"])
        previous_ring_normal_force = ring_normal_force
        if capture:
            accepted_rows.append(qualifies.detach().clone())
            state_qpos_rows.append(
                torch.cat((direct.hand_qpos, direct.pen_pos, direct.pen_quat), dim=-1)
                .detach()
                .clone()
            )
            state_qvel_rows.append(
                torch.cat((direct.hand_qvel, direct.pen_linvel, direct.pen_angvel), dim=-1)
                .detach()
                .clone()
            )
            state_target_rows.append(direct.policy_targets.detach().clone())
            contact_force_rows.append(direct.contact_force_norm.detach().clone())
            ring_feature_rows.append(direct._ring_axis_features().detach().clone())
            pen_linear_velocity_rows.append(direct.pen_linvel.detach().clone())
            pen_angular_velocity_rows.append(direct.pen_angvel.detach().clone())
            contact_force_vector_rows.append(
                direct.contact_force_w[:, 3].detach().clone()
            )
            contact_normal_force_rows.append(
                direct.contact_normal_force_signed[:, 3].detach().clone()
            )
            contact_tangential_force_rows.append(
                direct.contact_tangential_force_norm[:, 3].detach().clone()
            )
            contact_normal_speed_rows.append(
                direct.contact_normal_relative_speed[:, 3].detach().clone()
            )
            contact_slip_speed_rows.append(
                direct.contact_tangential_slip_speed[:, 3].detach().clone()
            )
            friction_margin_rows.append(friction_margin.detach().clone())
            tim_force_rows.append(direct.tim_force_on_pen_w.detach().clone())
            tim_torque_rows.append(direct.tim_torque_on_pen_w.detach().clone())
            ring_force_on_pen_rows.append(direct.ring_force_on_pen_w.detach().clone())
            ring_torque_rows.append(direct.ring_torque_on_pen_w.detach().clone())
            tim_gravity_residual_rows.append(
                direct.tim_gravity_force_residual_w.detach().clone()
            )
            total_gravity_residual_rows.append(
                direct.total_gravity_force_residual_w.detach().clone()
            )
            tim_moment_arm_rows.append(
                direct.tim_equivalent_moment_arm.detach().clone()
            )
            pen_com_position_rows.append(direct.pen_com_pos_w.detach().clone())
            pen_com_velocity_rows.append(direct.pen_com_vel_w.detach().clone())
            ring_force_servo_offset_rows.append(servo_offset.detach().clone())
            ring_slip_servo_offset_rows.append(slip_servo_offset.detach().clone())
            ring_tangential_velocity_vector_rows.append(
                direct.contact_tangential_relative_velocity_w[:, 3].detach().clone()
            )
            ring_tangential_velocity_component_rows.append(
                direct.contact_tangential_velocity_components[:, 3].detach().clone()
            )
            ring_contact_frame_rows.append(
                torch.stack(
                    (
                        direct.contact_radial_direction_w[:, 3],
                        direct.contact_tangent_axis_w[:, 3],
                        direct.contact_binormal_axis_w[:, 3],
                    ),
                    dim=1,
                ).detach().clone()
            )
            ring_target_error_rows.append(ring_target_error.detach().clone())
            ring_joint_target_error_rows.append(
                ring_joint_target_error.detach().clone()
            )
        elapsed += 1
        if (terminated | truncated).all():
            break

    result: dict[str, torch.Tensor | list[torch.Tensor]] = {
        "score": reduce_mean(best_score, candidate_ids, num_candidates),
        "score_env": best_score,
        "acceptance_rate": reduce_mean(accepted, candidate_ids, num_candidates),
        "under_rate": reduce_mean(best_under, candidate_ids, num_candidates),
        "semantic_rate": reduce_mean(best_semantic, candidate_ids, num_candidates),
        "feature_bounds_rate": reduce_mean(
            best_feature_bounds, candidate_ids, num_candidates
        ),
        "gap_pass_rate": reduce_mean(best_gap_pass, candidate_ids, num_candidates),
        "ring_gap_mm": reduce_mean(best_gap * 1000.0, candidate_ids, num_candidates),
        "min_distance_mm": reduce_mean(best_distance * 1000.0, candidate_ids, num_candidates),
        "semantic_cost_improvement": reduce_mean(
            initial_cost - best_cost, candidate_ids, num_candidates
        ),
        "hard_failure_rate": reduce_mean(hard_failure, candidate_ids, num_candidates),
        "max_penetration_mm": reduce_mean(max_penetration * 1000.0, candidate_ids, num_candidates),
        "max_position_drift_mm": reduce_mean(
            max_position_drift * 1000.0, candidate_ids, num_candidates
        ),
        "max_axis_drift_deg": reduce_mean(
            torch.rad2deg(max_axis_drift), candidate_ids, num_candidates
        ),
        "accepted_env": accepted,
        "semantic_accepted_env": semantic_accepted,
        "physical_accepted_env": physical_accepted,
        "wrench_accepted_env": wrench_accepted,
        "stable_capture_accepted_env": stable_capture_accepted,
        "max_consecutive_env": max_consecutive,
        "semantic_max_consecutive_env": semantic_max_consecutive,
        "physical_max_consecutive_env": physical_max_consecutive,
        "wrench_max_consecutive_env": wrench_max_consecutive,
        "stable_capture_max_consecutive_env": stable_capture_max_consecutive,
        "ring_contact_seen_env": ring_contact_seen,
        "distance_seen_env": distance_seen,
        "contact_and_distance_seen_env": contact_and_distance_seen,
        "physical_valid_seen_env": physical_valid_seen,
        "initial_cost_env": initial_cost,
        "best_cost_env": best_cost,
    }
    if capture:
        result["observation_rows"] = observation_rows
        result["action_rows"] = action_rows
        result["accepted_rows"] = accepted_rows
        result["target_condition_rows"] = target_condition_rows
        result["condition_delta_rows"] = condition_delta_rows
        result["semantic_margin_rows"] = semantic_margin_rows
        result["hold_latched_rows"] = hold_latched_rows
        result["state_qpos_rows"] = state_qpos_rows
        result["state_qvel_rows"] = state_qvel_rows
        result["state_target_rows"] = state_target_rows
        result["contact_force_rows"] = contact_force_rows
        result["ring_feature_rows"] = ring_feature_rows
        result["pen_linear_velocity_rows"] = pen_linear_velocity_rows
        result["pen_angular_velocity_rows"] = pen_angular_velocity_rows
        result["contact_force_vector_rows"] = contact_force_vector_rows
        result["contact_normal_force_rows"] = contact_normal_force_rows
        result["contact_tangential_force_rows"] = contact_tangential_force_rows
        result["contact_normal_speed_rows"] = contact_normal_speed_rows
        result["contact_slip_speed_rows"] = contact_slip_speed_rows
        result["friction_margin_rows"] = friction_margin_rows
        result["tim_force_rows"] = tim_force_rows
        result["tim_torque_rows"] = tim_torque_rows
        result["ring_force_on_pen_rows"] = ring_force_on_pen_rows
        result["ring_torque_rows"] = ring_torque_rows
        result["tim_gravity_residual_rows"] = tim_gravity_residual_rows
        result["total_gravity_residual_rows"] = total_gravity_residual_rows
        result["tim_moment_arm_rows"] = tim_moment_arm_rows
        result["pen_com_position_rows"] = pen_com_position_rows
        result["pen_com_velocity_rows"] = pen_com_velocity_rows
        result["ring_force_servo_offset_rows"] = ring_force_servo_offset_rows
        result["ring_slip_servo_offset_rows"] = ring_slip_servo_offset_rows
        result["ring_tangential_velocity_vector_rows"] = (
            ring_tangential_velocity_vector_rows
        )
        result["ring_tangential_velocity_component_rows"] = (
            ring_tangential_velocity_component_rows
        )
        result["ring_contact_frame_rows"] = ring_contact_frame_rows
        result["ring_target_error_rows"] = ring_target_error_rows
        result["ring_joint_target_error_rows"] = ring_joint_target_error_rows
        result["ring_force_servo_direction_ids"] = servo_direction_ids
        result["ring_force_servo_directions"] = servo_directions
    return result


def metric_row(metrics: dict[str, torch.Tensor | list[torch.Tensor]], index: int) -> dict:
    return {
        key: float(value[index].item())
        for key, value in metrics.items()
        if isinstance(value, torch.Tensor) and value.ndim == 1 and value.numel() > index
        and key not in {
            "accepted_env",
            "semantic_accepted_env",
            "physical_accepted_env",
            "max_consecutive_env",
            "semantic_max_consecutive_env",
            "physical_max_consecutive_env",
            "ring_contact_seen_env",
            "distance_seen_env",
            "contact_and_distance_seen_env",
            "physical_valid_seen_env",
            "initial_cost_env",
            "best_cost_env",
            "score_env",
        }
    }


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source_bundle = args.source_bundle.resolve()
    with np.load(source_bundle, allow_pickle=False) as data:
        row_count = int(data["source_qpos"].shape[0])
        semantic_soft = (
            np.asarray(data["manifold_semantic_soft"], dtype=np.float32)
            if "manifold_semantic_soft" in data
            else np.zeros(row_count, dtype=np.float32)
        )
        target_distance_mm = (
            np.asarray(data["target_distance_mm"], dtype=np.float32)
            if "target_distance_mm" in data
            else np.zeros(row_count, dtype=np.float32)
        )
        selection_groups = (
            np.asarray(data[args.unique_group_key])
            if args.unique_group_key and args.unique_group_key in data
            else None
        )
    if row_count == 0:
        raise ValueError("Terminal source bundle is empty")
    population = int(args.population)
    states_per_candidate = (
        int(args.states_per_candidate)
        if args.replicate_source_row
        else min(int(args.states_per_candidate), row_count)
    )
    num_envs = population * states_per_candidate
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = args.seed
    if args.ring_actuator_stiffness is not None or args.ring_actuator_damping is not None:
        base_actuator = cfg.robot_cfg.actuators["fingers"]
        support_actuator = copy.deepcopy(base_actuator)
        ring_actuator = copy.deepcopy(base_actuator)
        support_actuator.joint_names_expr = [
            name for name in DEX5_JOINT_ORDER if name not in RING_JOINT_NAMES
        ]
        ring_actuator.joint_names_expr = list(DEX5_JOINT_ORDER[8:12])
        if args.ring_actuator_stiffness is not None:
            ring_actuator.stiffness = float(args.ring_actuator_stiffness)
        if args.ring_actuator_damping is not None:
            ring_actuator.damping = float(args.ring_actuator_damping)
        cfg.robot_cfg.actuators = {
            "support": support_actuator,
            "ring": ring_actuator,
        }
    cfg.bundle_path = str(source_bundle)
    cfg.split_ids_path = ""
    cfg.use_all_bundle_rows = True
    cfg.teacher_prior_state_map_path = ""
    for noise_name in (
        "reset_joint_pos_noise",
        "reset_joint_vel_noise",
        "reset_pen_pos_noise",
        "reset_pen_vel_noise",
        "reset_pen_rot_noise_rad",
    ):
        noise_value = getattr(args, noise_name)
        if noise_value is not None:
            setattr(cfg, noise_name, float(noise_value))
    rollout_steps = sum(args.segment_steps) + max(int(args.post_success_hold_steps), 0)
    cfg.episode_length_s = max(float(cfg.episode_length_s), rollout_steps / 120.0)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()
    if args.paired_per_state:
        candidate_ids = torch.arange(num_envs, device=direct.device)
    else:
        candidate_ids = torch.arange(population, device=direct.device).repeat_interleave(
            states_per_candidate
        )
    generator = torch.Generator(device=direct.device).manual_seed(args.seed)
    if args.selection == "ordered":
        start = int(args.selection_offset)
        selected_np = np.arange(start, start + states_per_candidate, dtype=np.int64)
        if selected_np[-1] >= row_count:
            raise ValueError("Ordered selection exceeds source bundle")
        selected = torch.as_tensor(selected_np, dtype=torch.long, device=direct.device)
    elif args.selection == "semantic-boundary":
        ranking_score = semantic_soft - 0.02 * target_distance_mm
        ranked = np.argsort(-ranking_score, kind="stable")
        if selection_groups is not None:
            seen_groups: set[int] = set()
            unique_ranked = []
            for row in ranked.tolist():
                group = int(selection_groups[row])
                if group in seen_groups:
                    continue
                seen_groups.add(group)
                unique_ranked.append(row)
            ranked = np.asarray(unique_ranked, dtype=np.int64)
        start = int(args.selection_offset)
        selected_np = ranked[start : start + states_per_candidate]
        if selected_np.size < states_per_candidate:
            raise ValueError("Semantic-boundary selection exceeds source bundle")
        selected = torch.as_tensor(selected_np, dtype=torch.long, device=direct.device)
    else:
        selected = torch.randperm(
            row_count, generator=torch.Generator().manual_seed(args.seed)
        )[:states_per_candidate].to(direct.device)
    if args.replicate_source_row:
        if args.paired_per_state:
            raise ValueError("--replicate-source-row is incompatible with --paired-per-state")
        if selected.numel() == 0:
            raise ValueError("No source row is available for replication")
        selected = selected[:1].repeat(states_per_candidate)
    forced_ids = (
        selected.repeat_interleave(population)
        if args.paired_per_state
        else selected.repeat(population)
    )
    offset_limit = float(args.offset_limit_rad)
    low = torch.full((12,), -offset_limit, device=direct.device)
    high = torch.full((12,), offset_limit, device=direct.device)
    if args.paired_per_state:
        initial_mean = torch.as_tensor(
            args.initial_mean_offsets_rad
            if args.initial_mean_offsets_rad is not None
            else [0.0] * 12,
            dtype=torch.float32,
            device=direct.device,
        )
        mean = initial_mean.unsqueeze(0).repeat(states_per_candidate, 1)
        std = torch.full_like(mean, float(args.initial_std_rad))
        best_score = torch.full(
            (states_per_candidate,), -float("inf"), device=direct.device
        )
        best_parameters = torch.zeros_like(mean)
    else:
        mean = torch.as_tensor(
            args.initial_mean_offsets_rad
            if args.initial_mean_offsets_rad is not None
            else [0.0] * 12,
            dtype=torch.float32,
            device=direct.device,
        )
        std = torch.full((12,), float(args.initial_std_rad), device=direct.device)
        best_score = -float("inf")
        best_parameters = None
    elite_count = max(2, int(round(population * float(args.elite_fraction))))
    history = []
    best_metrics = None
    fixed_parameters_path = (
        args.fixed_parameters_json.resolve()
        if args.fixed_parameters_json is not None
        else None
    )
    if fixed_parameters_path is not None:
        fixed_payload = json.loads(fixed_parameters_path.read_text(encoding="utf-8"))
        fixed_rows = np.asarray(fixed_payload["best_offsets_rad"], dtype=np.float32)
        fixed_row = int(args.fixed_parameters_row)
        if fixed_row < 0 or fixed_row >= fixed_rows.shape[0]:
            raise ValueError("Fixed-parameter row is outside best_offsets_rad")
        fixed = torch.as_tensor(
            fixed_rows[fixed_row].reshape(12), device=direct.device
        )
        if args.paired_per_state:
            best_parameters = fixed.unsqueeze(0).repeat(states_per_candidate, 1)
        else:
            best_parameters = fixed
        search_iterations = 0
    else:
        search_iterations = int(args.iterations)
    for iteration in range(search_iterations):
        if args.paired_per_state:
            values = mean[:, None, :] + std[:, None, :] * torch.randn(
                (states_per_candidate, population, 12),
                generator=generator,
                device=direct.device,
            )
            values = torch.maximum(torch.minimum(values, high), low)
            if iteration == 0:
                values[:, 0] = mean
            flat_values = values.reshape(-1, 12)
            metrics = rollout(direct, env, flat_values, candidate_ids, forced_ids)
            scores = metrics["score"].reshape(states_per_candidate, population)
            elite_ids = torch.topk(scores, k=elite_count, dim=1).indices
            elites = torch.gather(
                values, 1, elite_ids.unsqueeze(-1).expand(-1, -1, 12)
            )
            mean = 0.20 * mean + 0.80 * elites.mean(dim=1)
            std = torch.maximum(
                0.0025 * torch.ones_like(std),
                0.25 * std + 0.75 * elites.std(dim=1, unbiased=False),
            )
            iteration_scores, iteration_ids = scores.max(dim=1)
            iteration_parameters = values[
                torch.arange(states_per_candidate, device=direct.device), iteration_ids
            ]
            improved = iteration_scores > best_score
            best_score = torch.where(improved, iteration_scores, best_score)
            best_parameters = torch.where(
                improved.unsqueeze(-1), iteration_parameters, best_parameters
            )
            history.append(
                {
                    "iteration": iteration,
                    "score_mean": float(iteration_scores.mean().item()),
                    "score_max": float(iteration_scores.max().item()),
                }
            )
        else:
            values = torch.maximum(
                torch.minimum(
                    mean.unsqueeze(0)
                    + std.unsqueeze(0)
                    * torch.randn((population, 12), generator=generator, device=direct.device),
                    high,
                ),
                low,
            )
            if iteration == 0:
                values[0] = mean
            metrics = rollout(direct, env, values[candidate_ids], candidate_ids, forced_ids)
            candidate_scores = metrics["score"]
            if args.robust_score_quantile is not None:
                quantile = float(args.robust_score_quantile)
                if not 0.0 <= quantile <= 1.0:
                    raise ValueError("--robust-score-quantile must be in [0, 1]")
                replica_scores = metrics["score_env"].reshape(
                    population, states_per_candidate
                )
                candidate_scores = torch.quantile(replica_scores, quantile, dim=1)
            elite_ids = torch.topk(candidate_scores, k=elite_count).indices
            elites = values[elite_ids]
            mean = 0.20 * mean + 0.80 * elites.mean(dim=0)
            std = torch.maximum(
                0.0025 * torch.ones_like(std),
                0.25 * std + 0.75 * elites.std(dim=0, unbiased=False),
            )
            best_id = int(torch.argmax(candidate_scores).item())
            row = metric_row(metrics, best_id)
            row["selection_score"] = float(candidate_scores[best_id].item())
            row["iteration"] = iteration
            row["offsets_rad"] = values[best_id].reshape(3, 4).detach().cpu().tolist()
            history.append(row)
            if row["selection_score"] > best_score:
                best_score = row["selection_score"]
                best_parameters = values[best_id].detach().clone()
                best_metrics = row

    assert best_parameters is not None
    if args.paired_per_state:
        final_values = best_parameters[:, None, :].expand(-1, population, -1).reshape(-1, 12)
    else:
        assert best_metrics is not None
        final_values = best_parameters.unsqueeze(0).expand(num_envs, -1)
    final = rollout(direct, env, final_values, candidate_ids, forced_ids, capture=True)
    observation_steps = torch.stack(final["observation_rows"], dim=1)
    action_steps = torch.stack(final["action_rows"], dim=1)
    accepted_steps = torch.stack(final["accepted_rows"], dim=1)
    target_condition_steps = torch.stack(final["target_condition_rows"], dim=1)
    condition_delta_steps = torch.stack(final["condition_delta_rows"], dim=1)
    semantic_margin_steps = torch.stack(final["semantic_margin_rows"], dim=1)
    hold_latched_steps = torch.stack(final["hold_latched_rows"], dim=1)
    state_qpos_steps = torch.stack(final["state_qpos_rows"], dim=1)
    state_qvel_steps = torch.stack(final["state_qvel_rows"], dim=1)
    state_target_steps = torch.stack(final["state_target_rows"], dim=1)
    contact_force_steps = torch.stack(final["contact_force_rows"], dim=1)
    ring_feature_steps = torch.stack(final["ring_feature_rows"], dim=1)
    pen_linear_velocity_steps = torch.stack(
        final["pen_linear_velocity_rows"], dim=1
    )
    pen_angular_velocity_steps = torch.stack(
        final["pen_angular_velocity_rows"], dim=1
    )
    contact_force_vector_steps = torch.stack(
        final["contact_force_vector_rows"], dim=1
    )
    contact_normal_force_steps = torch.stack(
        final["contact_normal_force_rows"], dim=1
    )
    contact_tangential_force_steps = torch.stack(
        final["contact_tangential_force_rows"], dim=1
    )
    contact_normal_speed_steps = torch.stack(
        final["contact_normal_speed_rows"], dim=1
    )
    contact_slip_speed_steps = torch.stack(
        final["contact_slip_speed_rows"], dim=1
    )
    friction_margin_steps = torch.stack(final["friction_margin_rows"], dim=1)
    tim_force_steps = torch.stack(final["tim_force_rows"], dim=1)
    tim_torque_steps = torch.stack(final["tim_torque_rows"], dim=1)
    ring_force_on_pen_steps = torch.stack(final["ring_force_on_pen_rows"], dim=1)
    ring_torque_steps = torch.stack(final["ring_torque_rows"], dim=1)
    tim_gravity_residual_steps = torch.stack(
        final["tim_gravity_residual_rows"], dim=1
    )
    total_gravity_residual_steps = torch.stack(
        final["total_gravity_residual_rows"], dim=1
    )
    tim_moment_arm_steps = torch.stack(final["tim_moment_arm_rows"], dim=1)
    pen_com_position_steps = torch.stack(final["pen_com_position_rows"], dim=1)
    pen_com_velocity_steps = torch.stack(final["pen_com_velocity_rows"], dim=1)
    ring_force_servo_offset_steps = torch.stack(
        final["ring_force_servo_offset_rows"], dim=1
    )
    ring_slip_servo_offset_steps = torch.stack(
        final["ring_slip_servo_offset_rows"], dim=1
    )
    ring_tangential_velocity_vector_steps = torch.stack(
        final["ring_tangential_velocity_vector_rows"], dim=1
    )
    ring_tangential_velocity_component_steps = torch.stack(
        final["ring_tangential_velocity_component_rows"], dim=1
    )
    ring_contact_frame_steps = torch.stack(final["ring_contact_frame_rows"], dim=1)
    ring_target_error_steps = torch.stack(final["ring_target_error_rows"], dim=1)
    ring_joint_target_error_steps = torch.stack(
        final["ring_joint_target_error_rows"], dim=1
    )
    # All candidates are identical in this replay; retain one copy per source state.
    if args.save_all_population:
        keep = torch.arange(num_envs, device=direct.device)
        kept_source_rows = forced_ids
    else:
        keep = (
            torch.arange(states_per_candidate, device=direct.device) * population
            if args.paired_per_state
            else torch.arange(states_per_candidate, device=direct.device)
        )
        kept_source_rows = selected
    accepted_any = final["accepted_env"][keep]
    save_all = args.save_all_trajectories or args.save_all_population
    saved_mask = torch.ones_like(accepted_any) if save_all else accepted_any
    accepted_ids = keep[saved_mask]
    servo_direction_ids = final["ring_force_servo_direction_ids"][keep]
    servo_directions = final["ring_force_servo_directions"]
    teacher_output = args.teacher_output.resolve()
    teacher_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        teacher_output,
        source_bundle_row=kept_source_rows[saved_mask].cpu().numpy(),
        accepted_trajectory=accepted_any[saved_mask].cpu().numpy(),
        observation=observation_steps[accepted_ids].cpu().numpy(),
        action=action_steps[accepted_ids].cpu().numpy(),
        target_condition=target_condition_steps[accepted_ids].cpu().numpy(),
        target_condition_delta=condition_delta_steps[accepted_ids].cpu().numpy(),
        semantic_margin=semantic_margin_steps[accepted_ids].cpu().numpy(),
        hold_latched=hold_latched_steps[accepted_ids].cpu().numpy(),
        state_qpos=state_qpos_steps[accepted_ids].cpu().numpy(),
        state_qvel=state_qvel_steps[accepted_ids].cpu().numpy(),
        state_target_qpos=state_target_steps[accepted_ids].cpu().numpy(),
        contact_force_thumb_index_middle_ring_pinky=contact_force_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_feature_m=ring_feature_steps[accepted_ids].cpu().numpy(),
        pen_linear_velocity=pen_linear_velocity_steps[accepted_ids].cpu().numpy(),
        pen_angular_velocity=pen_angular_velocity_steps[accepted_ids].cpu().numpy(),
        ring_contact_force_vector_w=contact_force_vector_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_contact_normal_force_signed=contact_normal_force_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_contact_tangential_force=contact_tangential_force_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_contact_normal_relative_speed=contact_normal_speed_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_contact_tangential_slip_speed=contact_slip_speed_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_contact_friction_margin_n=friction_margin_steps[
            accepted_ids
        ].cpu().numpy(),
        tim_force_on_pen_w=tim_force_steps[accepted_ids].cpu().numpy(),
        tim_torque_on_pen_about_com_w=tim_torque_steps[accepted_ids].cpu().numpy(),
        ring_force_on_pen_w=ring_force_on_pen_steps[accepted_ids].cpu().numpy(),
        ring_torque_on_pen_about_com_w=ring_torque_steps[accepted_ids].cpu().numpy(),
        tim_gravity_force_residual_w=tim_gravity_residual_steps[
            accepted_ids
        ].cpu().numpy(),
        total_gravity_force_residual_w=total_gravity_residual_steps[
            accepted_ids
        ].cpu().numpy(),
        tim_equivalent_moment_arm_m=tim_moment_arm_steps[
            accepted_ids
        ].cpu().numpy(),
        pen_com_position_w=pen_com_position_steps[accepted_ids].cpu().numpy(),
        pen_com_velocity_w=pen_com_velocity_steps[accepted_ids].cpu().numpy(),
        ring_force_servo_offset_rad=ring_force_servo_offset_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_slip_servo_offset_rad=ring_slip_servo_offset_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_contact_tangential_velocity_w=ring_tangential_velocity_vector_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_contact_tangential_velocity_components=ring_tangential_velocity_component_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_contact_frame_n_t_b_w=ring_contact_frame_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_target_error_rad=ring_target_error_steps[accepted_ids].cpu().numpy(),
        ring_joint_target_error_rad=ring_joint_target_error_steps[
            accepted_ids
        ].cpu().numpy(),
        ring_force_servo_direction_id=servo_direction_ids[saved_mask].cpu().numpy(),
        ring_force_servo_directions=servo_directions.cpu().numpy(),
        max_consecutive_steps=final["max_consecutive_env"][accepted_ids].cpu().numpy(),
        semantic_max_consecutive_steps=final["semantic_max_consecutive_env"][
            accepted_ids
        ].cpu().numpy(),
        physical_max_consecutive_steps=final["physical_max_consecutive_env"][
            accepted_ids
        ].cpu().numpy(),
        wrench_max_consecutive_steps=final["wrench_max_consecutive_env"][
            accepted_ids
        ].cpu().numpy(),
        stable_capture_max_consecutive_steps=final[
            "stable_capture_max_consecutive_env"
        ][accepted_ids].cpu().numpy(),
        accepted_step=accepted_steps[accepted_ids].cpu().numpy(),
        correction_offsets_rad=final_values[accepted_ids]
        .reshape(-1, 3, 4)
        .cpu()
        .numpy(),
        segment_steps=np.asarray(args.segment_steps, dtype=np.int64),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    direction_grid_results = []
    kept_max_consecutive = final["physical_max_consecutive_env"][keep]
    for direction_id in range(int(servo_directions.shape[0])):
        direction_mask = servo_direction_ids == direction_id
        direction_count = int(direction_mask.sum().item())
        direction_steps = kept_max_consecutive[direction_mask]
        direction_grid_results.append(
            {
                "direction_id": direction_id,
                "direction": servo_directions[direction_id].cpu().tolist(),
                "replicas": direction_count,
                "mean_max_consecutive_steps": float(
                    direction_steps.float().mean().item()
                ),
                "consecutive_rate_by_steps": {
                    str(steps): float((direction_steps >= steps).float().mean().item())
                    for steps in (1, 2, 3, 6, 12)
                },
            }
        )
    summary = {
        "task": args.task,
        "source_bundle": str(source_bundle),
        "source_bundle_sha256": sha256(source_bundle),
        "source_rows": row_count,
        "searched_source_rows": selected.cpu().tolist(),
        "population": population,
        "states_per_candidate": states_per_candidate,
        "replicate_source_row": bool(args.replicate_source_row),
        "robust_score_quantile": args.robust_score_quantile,
        "iterations": int(args.iterations),
        "executed_search_iterations": search_iterations,
        "fixed_parameters_json": (
            str(fixed_parameters_path) if fixed_parameters_path is not None else None
        ),
        "fixed_parameters_row": (
            int(args.fixed_parameters_row) if fixed_parameters_path is not None else None
        ),
        "segment_steps": list(args.segment_steps),
        "distance_limit_mm": float(args.distance_limit_mm),
        "paired_per_state": bool(args.paired_per_state),
        "selection": args.selection,
        "selection_offset": int(args.selection_offset),
        "unique_group_key": args.unique_group_key,
        "post_success_hold_steps": int(args.post_success_hold_steps),
        "gate_start_step": int(args.gate_start_step),
        "closed_loop_hold": bool(args.closed_loop_hold),
        "required_consecutive_steps": int(args.required_consecutive_steps),
        "acceptance_gate": args.acceptance_gate,
        "offset_limit_rad": float(args.offset_limit_rad),
        "initial_std_rad": float(args.initial_std_rad),
        "initial_mean_offsets_rad": (
            [float(value) for value in args.initial_mean_offsets_rad]
            if args.initial_mean_offsets_rad is not None
            else [0.0] * 12
        ),
        "ring_action_limit": float(args.ring_action_limit),
        "tim_action_scale": float(args.tim_action_scale),
        "ring_target_force_n": float(args.ring_target_force_n),
        "wrench_feasibility_gate": {
            "normal_force_min_n": float(args.wrench_normal_force_min_n),
            "normal_force_max_n": float(args.wrench_normal_force_max_n),
            "tangential_force_max_n": float(args.wrench_tangential_force_max_n),
            "slip_speed_max_m_s": float(args.wrench_slip_speed_max_m_s),
            "friction_coefficient": float(args.wrench_friction_coefficient),
            "joint_trajectory_is_latent_search_variable": True,
            "no_force_or_slip_controller": not (
                args.ring_force_servo or args.ring_slip_servo
            ),
        },
        "stable_capture_gate": {
            "min_normal_force_n": float(args.stable_capture_min_normal_force_n),
            "max_force_delta_n": float(args.stable_capture_max_force_delta_n),
            "max_slip_m_s": float(args.stable_capture_max_slip_m_s),
            "max_linear_speed_m_s": float(args.stable_capture_max_linear_speed_m_s),
            "max_angular_speed_rad_s": float(
                args.stable_capture_max_angular_speed_rad_s
            ),
            "max_ring_target_error_rad": float(
                args.stable_capture_max_ring_target_error_rad
            ),
            "max_ring_joint_error_rad": float(
                args.stable_capture_max_ring_joint_error_rad
            ),
            "preload_is_allowed_but_actuator_saturation_is_not": True,
            "window_steps": int(args.stable_capture_window_steps),
            "objective": (
                "maximize minimum Fn; penalize Fn variance, slip, COM motion, "
                "pen angular speed, ring tracking error, and actuator saturation"
            ),
        },
        "ring_force_servo": {
            "enabled": bool(args.ring_force_servo),
            "kp_rad_per_n": float(args.ring_force_servo_kp_rad_per_n),
            "kd_rad_s_per_m": float(args.ring_force_servo_kd_rad_s_per_m),
            "max_offset_rad": float(args.ring_force_servo_max_offset_rad),
            "start_step": int(args.ring_force_servo_start_step),
            "filter_alpha": float(args.ring_force_servo_filter_alpha),
            "direction": [float(value) for value in args.ring_force_servo_direction],
            "direction_grid_results": direction_grid_results,
        },
        "ring_slip_servo": {
            "enabled": bool(args.ring_slip_servo),
            "gain": float(args.ring_slip_servo_gain),
            "horizon_s": float(args.ring_slip_servo_horizon_s),
            "damping": float(args.ring_slip_servo_damping),
            "max_offset_rad": float(args.ring_slip_servo_max_offset_rad),
            "start_step": int(args.ring_slip_servo_start_step),
            "preserve_normal": bool(args.ring_slip_servo_preserve_normal),
            "contact_frame": "normal, pen-axis tangent, circumferential binormal",
        },
        "ring_actuator": {
            "stiffness": (
                float(args.ring_actuator_stiffness)
                if args.ring_actuator_stiffness is not None
                else 8.0
            ),
            "damping": (
                float(args.ring_actuator_damping)
                if args.ring_actuator_damping is not None
                else 0.2
            ),
            "support_actuator_unchanged": True,
        },
        "best_offsets_rad": best_parameters.reshape(-1, 3, 4).cpu().tolist(),
        "best_metrics": best_metrics,
        "final_acceptance_rate": float(accepted_any.float().mean().item()),
        "final_semantic_acceptance_rate": float(
            final["semantic_accepted_env"][keep].float().mean().item()
        ),
        "final_physical_acceptance_rate": float(
            final["physical_accepted_env"][keep].float().mean().item()
        ),
        "final_semantic_consecutive_rate_by_steps": {
            str(step): float(
                (final["semantic_max_consecutive_env"][keep] >= step).float().mean().item()
            )
            for step in (1, 2, 6, 12)
        },
        "final_physical_consecutive_rate_by_steps": {
            str(step): float(
                (final["physical_max_consecutive_env"][keep] >= step).float().mean().item()
            )
            for step in (1, 2, 6, 12)
        },
        "final_wrench_acceptance_rate": float(
            final["wrench_accepted_env"][keep].float().mean().item()
        ),
        "final_wrench_consecutive_rate_by_steps": {
            str(step): float(
                (final["wrench_max_consecutive_env"][keep] >= step)
                .float()
                .mean()
                .item()
            )
            for step in (1, 2, 3, 6, 12)
        },
        "final_stable_capture_acceptance_rate": float(
            final["stable_capture_accepted_env"][keep].float().mean().item()
        ),
        "final_stable_capture_consecutive_rate_by_steps": {
            str(step): float(
                (final["stable_capture_max_consecutive_env"][keep] >= step)
                .float()
                .mean()
                .item()
            )
            for step in (1, 2, 3, 6, 12)
        },
        "final_ring_contact_seen_rate": float(
            final["ring_contact_seen_env"][keep].float().mean().item()
        ),
        "final_distance_seen_rate": float(
            final["distance_seen_env"][keep].float().mean().item()
        ),
        "final_contact_and_distance_seen_rate": float(
            final["contact_and_distance_seen_env"][keep].float().mean().item()
        ),
        "final_physical_valid_seen_rate": float(
            final["physical_valid_seen_env"][keep].float().mean().item()
        ),
        "accepted_teacher_trajectories": int(accepted_any.sum().item()),
        "saved_diagnostic_trajectories": int(saved_mask.sum().item()),
        "save_all_population": bool(args.save_all_population),
        "reset_noise": {
            name: getattr(args, name)
            for name in (
                "reset_joint_pos_noise",
                "reset_joint_vel_noise",
                "reset_pen_pos_noise",
                "reset_pen_vel_noise",
                "reset_pen_rot_noise_rad",
            )
        },
        "teacher_output": str(teacher_output),
        "history": history,
        "ppo_started": False,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_TERMINAL_CEM=" + json.dumps(summary, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
