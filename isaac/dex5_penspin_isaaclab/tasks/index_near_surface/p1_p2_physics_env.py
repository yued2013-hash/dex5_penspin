from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from .recapture_p3_physics_env import (
    Dex5RecaptureP3PhysicsEnv,
    Dex5V9P1P2ResetBankPhysicsEnvCfg,
)


ROOT = Path(__file__).resolve().parents[5]
RESET_DIR = ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface/p1_p2_reset_bank"
FROZEN_BANK = str(
    RESET_DIR / "P1_P2_reset_bank_phase1最终validated自复验_validated.npz"
)
TRAIN_IDS = str(RESET_DIR / "P1_P2_train_ids.npy")
VALIDATION_IDS = str(RESET_DIR / "P1_P2_validation_ids.npy")
TEST_IDS = str(RESET_DIR / "P1_P2_test_ids.npy")
RING_GEOMETRY = str(RESET_DIR / "P1_P2_train_ring_geometry.json")
REVERSE_DIR = RESET_DIR / "reverse_curriculum"
REVERSE_R0 = str(REVERSE_DIR / "P1_P2_reverse_R0_P2_hold_train.npz")
REVERSE_R1 = str(REVERSE_DIR / "P1_P2_reverse_R1_2_4mm_train.npz")
REVERSE_R2 = str(REVERSE_DIR / "P1_P2_reverse_R2_4_8mm_train.npz")
REVERSE_R3 = str(REVERSE_DIR / "P1_P2_reverse_R3_8_12mm_train.npz")
ARRIVAL_BANK_512 = str(REVERSE_DIR / "P1_P2_validated_R0_frontier_arrival_bank_512.npz")
FAILURE_FRONTIER_240 = str(REVERSE_DIR / "P2鲁棒核心61_H6失败前沿_A2_A4_240.npz")
P2_H6_STABLE_IDS = str(REVERSE_DIR / "P2_H6_100ms鲁棒核心61_train_ids.npy")
P2_E1_CONTROLLED_EXPANSION_IDS = str(
    REVERSE_DIR / "P2_E1核心61加策略前沿24_train_ids.npy"
)
TRIPOD_BALANCE_DIR = ROOT / (
    "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "p1_p2_primitive/tripod_balance"
)
P1_617_BANK = str(TRIPOD_BALANCE_DIR / "P1_617_TIM三指平衡_CEM_v5低速_train_bank.npz")
P1_617_IDS = str(TRIPOD_BALANCE_DIR / "P1_617_TIM三指平衡_CEM_v5低速_train_ids.npy")
P1_617_PAIRED_P2_MANIFOLD = str(
    TRIPOD_BALANCE_DIR / "P1_617配对P2流形_500ms三seed可靠子集.npz"
)
P1_617_ROUTE539_LOCAL_ARRIVAL = str(
    TRIPOD_BALANCE_DIR / "P1_617单路线539_真实动态arrival_5到8mm_均衡512.npz"
)
P1_617_ROUTE539_LOCAL_ARRIVAL_BIN0_IDS = str(
    TRIPOD_BALANCE_DIR / "P1_617单路线539_真实动态arrival_5到8mm_均衡512_bin0_ids.npy"
)
P1_617_ROUTE539_MIXED_BANK = str(
    TRIPOD_BALANCE_DIR / "P1_617单路线539_P1与arrival均衡混合bank_1024.npz"
)
STATE_TEACHER_MAP = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p1_p2_primitive/"
    "C2_train_状态条件teacher_oracle全量900.npz"
)


@configclass
class Dex5P1P2PrimitivePhysicsEnvCfg(Dex5V9P1P2ResetBankPhysicsEnvCfg):
    """P1-to-P2 support-transfer primitive with frozen split isolation."""

    bundle_path = FROZEN_BANK
    split_ids_path = TRAIN_IDS
    target_statistics_ids_path = TRAIN_IDS
    target_statistics_bundle_path = FROZEN_BANK
    ring_geometry_path = RING_GEOMETRY
    split_name = "train"
    source_phase_id = 1
    source_phase_ids: tuple[int, ...] | None = None
    use_all_bundle_rows = False
    observation_space = 164
    episode_length_s = 0.80
    action_scale = 0.10
    curriculum_stage = 4
    stage_target_progress = (0.0, 0.30, 0.60, 1.0, 1.0, 1.0)
    stage_progress_tolerance = (0.08, 0.12, 0.12, 0.15, 0.15, 0.15)
    stage_target_distance_m = (0.00325, 0.0120, 0.0100, 0.00245, 0.00245, 0.00245)
    stage_hold_steps = (6, 3, 3, 3, 6, 6)
    p2_contact_force_n = 0.03
    p2_contact_grace_steps = 2
    p2_max_position_drift_m = 0.010
    p2_max_axis_drift_deg = 8.0
    p2_max_angular_speed_rad_s = 12.0
    p2_max_linear_speed_m_s = 0.40
    p2_max_penetration_m = 0.00060
    p2_max_ring_force_share = 0.25
    p2_max_ring_force_n = 0.60
    p2_max_pinky_force_n = 0.03
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    reset_pen_rot_noise_rad = 0.0
    pen_static_friction_range = (0.60, 0.60)
    pen_dynamic_friction_range = (0.45, 0.45)
    teacher_prior_enabled = False
    teacher_prior_mode = "direct"
    teacher_prior_ring_fraction = 0.40
    teacher_prior_ring_joint_fractions: tuple[float, float, float, float] | None = None
    teacher_prior_state_map_path = ""
    teacher_prior_support_actions = (0.40, 0.40, 0.0)
    support_action_mask_scale = 0.25
    teacher_prior_geometric_brake = False
    teacher_prior_brake_progress_margin = 0.06
    teacher_prior_brake_distance_m = 0.0105
    teacher_residual_scale = 0.25
    teacher_c3_entry_progress = 0.60
    teacher_c3_entry_progress_tolerance = 0.12
    teacher_c3_entry_distance_m = 0.010
    teacher_c3_entry_hold_steps = 1
    teacher_c3_endpoint_fractions = (0.40, 1.30, 1.10, 0.80)
    teacher_c3_sequence_joint_tolerance_rad = 0.08
    teacher_c3_final_action_scale = 0.50
    teacher_phase_observation_enabled = False
    p2_geometry_soft_reward_scale = 0.0
    p2_ignore_progress_gate = False
    p2_ignore_progress_reward = False
    p2_triad_capture_soft_reward_scale = 0.0
    middle_contact_near_p2_reward_scale = 0.0
    p2_capture_quality_reward_scale = 24.0
    p2_manifold_semantic_reward_scale = 0.0
    p2_hold_reward_scale = 8.0
    p2_success_reward_scale = 40.0
    p2_pen_position_reward_scale = 2.0
    p2_pen_axis_reward_scale = 2.0
    p2_ring_overload_penalty_scale = 4.0
    p2_pinky_load_penalty_scale = 3.0
    p2_pen_velocity_penalty_scale = 1.2
    p2_penetration_penalty_scale = 10.0
    p2_action_l2_penalty_scale = 0.03
    p2_action_smoothness_penalty_scale = 0.08
    p2_joint_power_penalty_scale = 0.002
    p2_failure_penalty_scale = 10.0
    hold_current_joint_targets_on_reset = False
    arrival_observation_enabled = False
    arrival_hold_reward_mode = False
    hold_action_mask_enabled = False
    hold_support_action_scale = 0.35
    hold_transfer_action_scale = 0.08
    hold_post_latch_action_scale = 0.20
    target_p2_ring_qpos_override: tuple[float, float, float, float] | None = None
    target_p2_ring_center_override: tuple[float, float, float] | None = None
    p2_manifold_path = ""
    # Keep the historical semantic box as the default success definition. The
    # physical-contact mode is reserved for criterion audits and still applies
    # distance, contact, drift, velocity, load, penetration, and failure gates.
    p2_success_gate_mode = "semantic"
    manifold_condition_observation_enabled = False
    manifold_waypoint_phase_observation_enabled = False
    conditioned_reward_uses_desired_mode = False
    fixed_desired_manifold_mode = -1
    fixed_manifold_waypoint_durations: tuple[float, float, float] | None = None
    fixed_route_clearance_offset: tuple[float, float, float, float] | None = None
    fixed_route_waypoint2_offset: tuple[float, float, float, float] | None = None
    fixed_route_endpoint_offset: tuple[float, float, float, float] | None = None
    waypoint_target_error_observation_enabled = False
    local_arrival_final_waypoint_observation = False
    mixed_route_arrival_observation = False
    mixed_arrival_action_scale = 0.25
    terminal_correction_observation_enabled = False
    terminal_correction_distance_m = 0.00395
    terminal_correction_phase_observation_enabled = False
    terminal_correction_phase_horizon_steps = 18
    terminal_correction_target_observation_enabled = False
    terminal_correction_dynamics_observation_enabled = False
    terminate_on_p2_success = True
    penetration_threshold = 0.00060
    drop_threshold = 0.020
    drift_threshold = 0.020


@configclass
class Dex5P1P2C0PhysicsEnvCfg(Dex5P1P2PrimitivePhysicsEnvCfg):
    curriculum_stage = 0


@configclass
class Dex5P1P2C1PhysicsEnvCfg(Dex5P1P2PrimitivePhysicsEnvCfg):
    curriculum_stage = 1


@configclass
class Dex5P1P2C1TeacherResidualPhysicsEnvCfg(Dex5P1P2C1PhysicsEnvCfg):
    teacher_prior_enabled = True


@configclass
class Dex5P1P2C2PhysicsEnvCfg(Dex5P1P2PrimitivePhysicsEnvCfg):
    curriculum_stage = 2


@configclass
class Dex5P1P2C2TeacherResidualPhysicsEnvCfg(Dex5P1P2C2PhysicsEnvCfg):
    teacher_prior_enabled = True
    teacher_prior_ring_joint_fractions = (0.40, 0.30, 0.65, 1.45)
    teacher_prior_geometric_brake = True


@configclass
class Dex5P1P2C2StateTeacherResidualPhysicsEnvCfg(Dex5P1P2C2PhysicsEnvCfg):
    teacher_prior_enabled = True
    teacher_prior_state_map_path = STATE_TEACHER_MAP
    teacher_prior_geometric_brake = True
    teacher_prior_support_actions = (0.20, 0.20, 0.0)
    support_action_mask_scale = 0.50
    teacher_prior_brake_progress_margin = 0.08
    teacher_prior_brake_distance_m = 0.011
    teacher_phase_observation_enabled = True
    observation_space = 168


@configclass
class Dex5P1P2C3PhysicsEnvCfg(Dex5P1P2PrimitivePhysicsEnvCfg):
    curriculum_stage = 3
    episode_length_s = 1.20


@configclass
class Dex5P1P2C3TeacherResidualPhysicsEnvCfg(Dex5P1P2C3PhysicsEnvCfg):
    teacher_prior_enabled = True
    teacher_prior_mode = "c3_distal_then_proximal"
    teacher_prior_state_map_path = STATE_TEACHER_MAP
    teacher_prior_geometric_brake = True
    # Keep the same +0.01 rad scripted support bias while doubling the bounded
    # residual authority available for closed-loop pen-axis stabilization.
    teacher_prior_support_actions = (0.20, 0.20, 0.0)
    support_action_mask_scale = 0.50
    teacher_prior_brake_progress_margin = 0.10
    teacher_prior_brake_distance_m = 0.0032
    teacher_c3_sequence_joint_tolerance_rad = 0.20
    teacher_c3_final_action_scale = 1.0
    teacher_phase_observation_enabled = True
    observation_space = 168
    episode_length_s = 1.50


@configclass
class Dex5P1P2C3Near8mmTeacherResidualPhysicsEnvCfg(
    Dex5P1P2C3TeacherResidualPhysicsEnvCfg
):
    """Bridge curriculum: reach near-P2 geometry before the strict P2 gate."""

    p2_max_axis_drift_deg = 25.0
    stage_progress_tolerance = (0.08, 0.12, 0.12, 0.20, 0.15, 0.15)
    stage_target_distance_m = (0.00325, 0.0120, 0.0100, 0.0080, 0.00245, 0.00245)
    stage_hold_steps = (6, 3, 3, 1, 6, 6)
    teacher_prior_brake_progress_margin = 0.12
    teacher_prior_brake_distance_m = 0.0100
    teacher_c3_final_action_scale = 0.65
    p2_geometry_soft_reward_scale = 6.0
    p2_triad_capture_soft_reward_scale = 8.0
    middle_contact_near_p2_reward_scale = 8.0


@configclass
class Dex5P1P2C3Near5mmTeacherResidualPhysicsEnvCfg(
    Dex5P1P2C3Near8mmTeacherResidualPhysicsEnvCfg
):
    """Bridge curriculum: tighten near-P2 geometry before the strict P2 gate."""

    stage_target_distance_m = (0.00325, 0.0120, 0.0100, 0.0050, 0.00245, 0.00245)
    teacher_prior_brake_distance_m = 0.0070
    teacher_c3_final_action_scale = 0.70


@configclass
class Dex5P1P2C3GeometryTeacherResidualPhysicsEnvCfg(
    Dex5P1P2C3TeacherResidualPhysicsEnvCfg
):
    """Acquire exact P2 ring geometry before tightening pen-axis control."""

    p2_max_axis_drift_deg = 25.0
    stage_hold_steps = (6, 3, 3, 1, 6, 6)
    teacher_prior_brake_distance_m = 0.0045
    teacher_c3_final_action_scale = 0.75


@configclass
class Dex5P1P2C3GeometryDenseTeacherResidualPhysicsEnvCfg(
    Dex5P1P2C3GeometryTeacherResidualPhysicsEnvCfg
):
    """Dense final-geometry curriculum for the last few millimeters into P2."""

    teacher_prior_brake_progress_margin = 0.08
    teacher_prior_brake_distance_m = 0.0032
    teacher_c3_final_action_scale = 0.45
    p2_geometry_soft_reward_scale = 12.0
    p2_triad_capture_soft_reward_scale = 12.0
    middle_contact_near_p2_reward_scale = 4.0


@configclass
class Dex5P1P2RingOnlyC3TeacherResidualPhysicsEnvCfg(
    Dex5P1P2C3GeometryDenseTeacherResidualPhysicsEnvCfg
):
    """P1-to-P2 entry curriculum with only the ring finger actuated by policy/teacher."""

    # Thumb, index, and middle remain on their per-reset P1 position targets.
    # Pinky is already excluded by the base action mask.
    support_action_mask_scale = 0.0
    teacher_prior_support_actions = (0.0, 0.0, 0.0)
    teacher_residual_scale = 0.25
    p2_action_smoothness_penalty_scale = 0.16


@configclass
class Dex5P1P2RingOnly120HzC3TeacherResidualPhysicsEnvCfg(
    Dex5P1P2RingOnlyC3TeacherResidualPhysicsEnvCfg
):
    """Higher-bandwidth ring-only entry with the same approximate target rate."""

    decimation = 1
    action_scale = 0.05
    stage_hold_steps = (12, 6, 6, 2, 12, 12)


@configclass
class Dex5P1P2RingOnly120HzH1PhysicsEnvCfg(
    Dex5P1P2RingOnly120HzC3TeacherResidualPhysicsEnvCfg
):
    """Pure-policy ring-only entry and 16.7 ms hold curriculum."""

    teacher_prior_enabled = False
    arrival_hold_reward_mode = True
    stage_hold_steps = (12, 6, 6, 2, 12, 12)


@configclass
class Dex5P1P2RingOnly120HzH2PhysicsEnvCfg(Dex5P1P2RingOnly120HzH1PhysicsEnvCfg):
    """Pure-policy ring-only entry and 33.3 ms hold curriculum."""

    stage_hold_steps = (12, 6, 6, 4, 12, 12)


@configclass
class Dex5P1P2RingOnly120HzH3PhysicsEnvCfg(Dex5P1P2RingOnly120HzH1PhysicsEnvCfg):
    """Pure-policy ring-only entry and 50 ms hold curriculum."""

    stage_hold_steps = (12, 6, 6, 6, 12, 12)


@configclass
class Dex5P1P2RingOnly120HzH6PhysicsEnvCfg(Dex5P1P2RingOnly120HzH1PhysicsEnvCfg):
    """Pure-policy ring-only entry and strict 100 ms hold curriculum."""

    stage_hold_steps = (12, 6, 6, 12, 12, 12)


@configclass
class Dex5P1P2TinyTIM120HzH1PhysicsEnvCfg(Dex5P1P2RingOnly120HzH1PhysicsEnvCfg):
    """Ablation with at most 0.0025 rad bounded TIM contact recovery."""

    support_action_mask_scale = 0.05


@configclass
class Dex5P1P2TIMAssist005120HzH1PhysicsEnvCfg(Dex5P1P2RingOnly120HzH1PhysicsEnvCfg):
    """Ring-led entry with at most 0.005 rad TIM contact recovery."""

    support_action_mask_scale = 0.10


@configclass
class Dex5P1P2P1617PairedManifold120HzH6PhysicsEnvCfg(
    Dex5P1P2TIMAssist005120HzH1PhysicsEnvCfg
):
    """P1_617 entry targeting the nearest verified ring-under P2 mode."""

    bundle_path = P1_617_BANK
    split_ids_path = P1_617_IDS
    source_phase_id = 1
    split_name = "train"
    p2_manifold_path = P1_617_PAIRED_P2_MANIFOLD
    curriculum_stage = 3
    episode_length_s = 2.0
    stage_hold_steps = (12, 6, 6, 12, 12, 12)
    p2_ignore_progress_gate = True
    p2_ignore_progress_reward = True
    arrival_hold_reward_mode = True
    arrival_observation_enabled = True
    teacher_phase_observation_enabled = False
    observation_space = 172


@configclass
class Dex5P1P2P1617PairedManifoldConditioned120HzH6PhysicsEnvCfg(
    Dex5P1P2P1617PairedManifold120HzH6PhysicsEnvCfg
):
    """Teacher task conditioned on a desired manifold mode and waypoint phase."""

    manifold_condition_observation_enabled = True
    manifold_waypoint_phase_observation_enabled = True
    conditioned_reward_uses_desired_mode = True
    observation_space = 182


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH6PhysicsEnvCfg(
    Dex5P1P2P1617PairedManifoldConditioned120HzH6PhysicsEnvCfg
):
    """Single-route proof task using the most stable verified route into mode 5."""

    fixed_desired_manifold_mode = 5
    fixed_manifold_waypoint_durations = (
        20.61328125,
        47.94880676269531,
        46.393524169921875,
    )
    fixed_route_clearance_offset = (
        0.09726466983556747,
        -0.1489972621202469,
        0.02951819635927677,
        -0.15583617985248566,
    )
    fixed_route_waypoint2_offset = (
        -0.44111165404319763,
        -0.23773685097694397,
        0.754718542098999,
        0.9519084692001343,
    )
    fixed_route_endpoint_offset = (
        -0.435596227645874,
        -0.1498880237340927,
        0.5493462085723877,
        0.8844786286354065,
    )
    waypoint_target_error_observation_enabled = True
    observation_space = 186


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH1PhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH6PhysicsEnvCfg
):
    """Single-route arrival curriculum with a one-step-equivalent hold gate."""

    stage_hold_steps = (12, 6, 6, 2, 12, 12)


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalWindowPhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH1PhysicsEnvCfg
):
    """H1 task with extra phase-3 control time for terminal correction."""

    episode_length_s = 3.0
    terminal_correction_observation_enabled = True
    observation_space = 187


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalPhasePhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalWindowPhysicsEnvCfg
):
    """Terminal correction task with a continuous normalized correction phase."""

    terminal_correction_phase_observation_enabled = True
    observation_space = 188


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalTargetPhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalPhasePhysicsEnvCfg
):
    """Terminal correction task conditioned on phase, feature error, and gap error."""

    terminal_correction_target_observation_enabled = True
    observation_space = 192


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalDynamicsPhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalTargetPhysicsEnvCfg
):
    """Terminal task with semantic margin, target derivative, and hold latch."""

    terminal_correction_dynamics_observation_enabled = True
    observation_space = 198


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH2TerminalDynamicsPhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalDynamicsPhysicsEnvCfg
):
    """Terminal dynamics task requiring four consecutive 120 Hz hold steps."""

    stage_hold_steps = (12, 6, 6, 4, 12, 12)


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH3TerminalDynamicsPhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalDynamicsPhysicsEnvCfg
):
    """Terminal dynamics task requiring six consecutive 120 Hz hold steps."""

    stage_hold_steps = (12, 6, 6, 6, 12, 12)


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH6TerminalDynamicsPhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH1TerminalDynamicsPhysicsEnvCfg
):
    """Terminal dynamics task requiring twelve consecutive 120 Hz hold steps."""

    stage_hold_steps = (12, 6, 6, 12, 12, 12)


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH6TerminalDynamicsPhysicalAuditPhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH6TerminalDynamicsPhysicsEnvCfg
):
    """H6 audit where physical ring contact replaces the semantic box hard gate."""

    p2_success_gate_mode = "physical_contact"


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH2PhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH6PhysicsEnvCfg
):
    """Single-route arrival curriculum with a 33 ms hold gate."""

    stage_hold_steps = (12, 6, 6, 4, 12, 12)


@configclass
class Dex5P1P2P1617SingleRoute539Conditioned120HzH3PhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH6PhysicsEnvCfg
):
    """Single-route arrival curriculum with a 50 ms hold gate."""

    stage_hold_steps = (12, 6, 6, 6, 12, 12)


@configclass
class Dex5P1P2P1617Route539LocalArrival120HzH6PhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH6PhysicsEnvCfg
):
    """Local 5-8 mm braking and correction task from real dynamic arrivals."""

    bundle_path = P1_617_ROUTE539_LOCAL_ARRIVAL
    split_ids_path = ""
    use_all_bundle_rows = True
    source_phase_id = 2
    episode_length_s = 0.50
    action_scale = 0.05
    teacher_prior_state_map_path = ""
    local_arrival_final_waypoint_observation = True
    p2_geometry_soft_reward_scale = 100.0
    p2_triad_capture_soft_reward_scale = 24.0
    p2_capture_quality_reward_scale = 55.0
    p2_manifold_semantic_reward_scale = 50.0
    p2_hold_reward_scale = 48.0
    middle_contact_near_p2_reward_scale = 4.0
    p2_pen_velocity_penalty_scale = 4.0
    p2_action_smoothness_penalty_scale = 0.28
    stage_hold_steps = (12, 6, 6, 12, 12, 12)


@configclass
class Dex5P1P2P1617Route539LocalArrival120HzH6PhysicalAuditPhysicsEnvCfg(
    Dex5P1P2P1617Route539LocalArrival120HzH6PhysicsEnvCfg
):
    """Local H6 feasibility audit using a real ring-contact success gate."""

    p2_success_gate_mode = "physical_contact"


@configclass
class Dex5P1P2P1617Route539LocalArrival120HzH1PhysicsEnvCfg(
    Dex5P1P2P1617Route539LocalArrival120HzH6PhysicsEnvCfg
):
    stage_hold_steps = (12, 6, 6, 2, 12, 12)


@configclass
class Dex5P1P2P1617Route539LocalArrivalNear120HzH1PhysicsEnvCfg(
    Dex5P1P2P1617Route539LocalArrival120HzH1PhysicsEnvCfg
):
    """Closest 5-6 mm arrival curriculum with quarter-size route increments."""

    split_ids_path = P1_617_ROUTE539_LOCAL_ARRIVAL_BIN0_IDS
    use_all_bundle_rows = False
    action_scale = 0.025
    p2_manifold_semantic_reward_scale = 80.0


@configclass
class Dex5P1P2P1617Route539Mixed120HzH1PhysicsEnvCfg(
    Dex5P1P2P1617SingleRoute539Conditioned120HzH1PhysicsEnvCfg
):
    """One policy trained equally on full P1 route and real local arrivals."""

    bundle_path = P1_617_ROUTE539_MIXED_BANK
    split_ids_path = ""
    use_all_bundle_rows = True
    source_phase_ids = (1, 2)
    mixed_route_arrival_observation = True
    teacher_prior_state_map_path = ""
    p2_manifold_semantic_reward_scale = 60.0
    p2_geometry_soft_reward_scale = 100.0
    p2_triad_capture_soft_reward_scale = 24.0
    p2_capture_quality_reward_scale = 55.0
    p2_hold_reward_scale = 48.0
    p2_pen_velocity_penalty_scale = 4.0
    p2_action_smoothness_penalty_scale = 0.24


@configclass
class Dex5P1P2P1617Route539LocalArrival120HzH2PhysicsEnvCfg(
    Dex5P1P2P1617Route539LocalArrival120HzH6PhysicsEnvCfg
):
    stage_hold_steps = (12, 6, 6, 4, 12, 12)


@configclass
class Dex5P1P2P1617Route539LocalArrival120HzH3PhysicsEnvCfg(
    Dex5P1P2P1617Route539LocalArrival120HzH6PhysicsEnvCfg
):
    stage_hold_steps = (12, 6, 6, 6, 12, 12)


@configclass
class Dex5P1P2TIMAssist0075120HzH1PhysicsEnvCfg(Dex5P1P2RingOnly120HzH1PhysicsEnvCfg):
    """Ring-led entry with at most 0.0075 rad TIM contact recovery."""

    support_action_mask_scale = 0.15


@configclass
class Dex5P1P2TIMAssist010120HzH1PhysicsEnvCfg(Dex5P1P2RingOnly120HzH1PhysicsEnvCfg):
    """Ring-led entry with at most 0.010 rad TIM contact recovery."""

    support_action_mask_scale = 0.20


@configclass
class Dex5P1P2ReverseP2SourcePhysicsEnvCfg(Dex5P1P2PrimitivePhysicsEnvCfg):
    """Train-only P2-source environment for reverse-curriculum state rollout."""

    source_phase_id = 2
    curriculum_stage = 3
    episode_length_s = 0.80
    action_scale = 0.10
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    p2_max_axis_drift_deg = 25.0
    stage_hold_steps = (6, 3, 3, 6, 6, 6)


@configclass
class Dex5P1P2ReverseR0HoldPhysicsEnvCfg(Dex5P1P2C3GeometryDenseTeacherResidualPhysicsEnvCfg):
    """R0 reverse curriculum: start at validated P2 and hold for 100 ms."""

    bundle_path = REVERSE_R0
    use_all_bundle_rows = True
    source_phase_id = 2
    curriculum_stage = 3
    episode_length_s = 0.60
    p2_max_axis_drift_deg = 25.0
    stage_hold_steps = (6, 3, 3, 6, 6, 6)
    teacher_prior_enabled = False
    teacher_prior_state_map_path = ""
    teacher_phase_observation_enabled = False
    observation_space = 164
    p2_ignore_progress_gate = True
    p2_ignore_progress_reward = True
    p2_geometry_soft_reward_scale = 40.0
    p2_triad_capture_soft_reward_scale = 20.0
    middle_contact_near_p2_reward_scale = 4.0
    hold_current_joint_targets_on_reset = False


@configclass
class Dex5P1P2ReverseR0ValidatedHoldPhysicsEnvCfg(Dex5P1P2ReverseR0HoldPhysicsEnvCfg):
    """R0 reverse curriculum from the frozen validated P2 train split."""

    bundle_path = FROZEN_BANK
    use_all_bundle_rows = False
    split_ids_path = TRAIN_IDS
    split_name = "train"


@configclass
class Dex5P1P2ReverseR0ValidatedStabilizePhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldPhysicsEnvCfg
):
    """R0 hold curriculum tuned for the last 5-8 mm deceleration/stability bottleneck."""

    p2_capture_quality_reward_scale = 36.0
    p2_hold_reward_scale = 24.0
    p2_success_reward_scale = 60.0
    p2_pen_position_reward_scale = 3.0
    p2_pen_axis_reward_scale = 4.0
    p2_pen_velocity_penalty_scale = 2.8
    p2_action_smoothness_penalty_scale = 0.16
    p2_joint_power_penalty_scale = 0.004
    middle_contact_near_p2_reward_scale = 8.0


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldPhysicsEnvCfg
):
    """Static P2 hold baseline with reduced action authority for TIM support."""

    hold_action_mask_enabled = True
    hold_support_action_scale = 0.35
    hold_transfer_action_scale = 0.05
    p2_capture_quality_reward_scale = 30.0
    p2_hold_reward_scale = 28.0
    p2_success_reward_scale = 65.0
    p2_pen_position_reward_scale = 3.0
    p2_pen_axis_reward_scale = 4.0
    p2_pen_velocity_penalty_scale = 2.8
    p2_action_l2_penalty_scale = 0.05
    p2_action_smoothness_penalty_scale = 0.22
    p2_joint_power_penalty_scale = 0.004
    middle_contact_near_p2_reward_scale = 8.0


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestrictedH1PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg
):
    """Static validated P2 hold curriculum, 1 strict step."""

    stage_hold_steps = (6, 3, 3, 1, 1, 1)


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestrictedH1ContinuePhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedH1PhysicsEnvCfg
):
    """Reward first-step capture but keep stepping to learn post-arrival correction."""

    terminate_on_p2_success = False


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestrictedH2PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg
):
    """Static validated P2 hold curriculum, 2 strict steps."""

    stage_hold_steps = (6, 3, 3, 2, 2, 2)


@configclass
class Dex5P1P2ReverseR0ValidatedHoldFrozenRingH1ContinuePhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedH1ContinuePhysicsEnvCfg
):
    """Post-arrival bridge with ring frozen and small TIM corrections."""

    hold_support_action_scale = 0.20
    hold_transfer_action_scale = 0.0


@configclass
class Dex5P1P2ReverseR0ValidatedHoldFrozenRingH2PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedH2PhysicsEnvCfg
):
    """Two-step strict hold with ring frozen and small TIM corrections."""

    hold_support_action_scale = 0.20
    hold_transfer_action_scale = 0.0


@configclass
class Dex5P1P2ReverseR0ValidatedHoldCurrentTargetFrozenRingH2PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldFrozenRingH2PhysicsEnvCfg
):
    """Reset-target ablation: anchor the hold controller at reset qpos."""

    hold_current_joint_targets_on_reset = True


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestrictedH3PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg
):
    """Static validated P2 hold curriculum, 3 strict steps."""

    stage_hold_steps = (6, 3, 3, 3, 3, 3)


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestrictedH4PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg
):
    """Static validated P2 hold curriculum, 4 strict steps."""

    stage_hold_steps = (6, 3, 3, 4, 4, 4)


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestrictedH5PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg
):
    """Static validated P2 hold curriculum, 5 strict steps."""

    stage_hold_steps = (6, 3, 3, 5, 5, 5)


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestrictedH6PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg
):
    """Static validated P2 hold curriculum, 6 strict steps (100 ms)."""

    stage_hold_steps = (6, 3, 3, 6, 6, 6)


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestricted120HzPhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedPhysicsEnvCfg
):
    """Control-bandwidth ablation for the static P2 hold baseline."""

    decimation = 1


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedH1PhysicsEnvCfg
):
    """120 Hz static hold curriculum, 2 control steps (16.7 ms)."""

    decimation = 1
    stage_hold_steps = (12, 6, 6, 2, 2, 2)


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1ContinuePhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1PhysicsEnvCfg
):
    """120 Hz first-step bridge that continues after initial capture."""

    terminate_on_p2_success = False


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepPhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1PhysicsEnvCfg
):
    """120 Hz bridge curriculum, 3 control steps (25 ms)."""

    stage_hold_steps = (12, 6, 6, 3, 3, 3)


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatchedPhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepPhysicsEnvCfg
):
    """Three-step bridge with post-entry hold reward and smaller actions."""

    arrival_hold_reward_mode = True
    arrival_observation_enabled = False
    hold_post_latch_action_scale = 0.20


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatchedObsPhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatchedPhysicsEnvCfg
):
    """Latched three-step bridge with explicit arrival/hold observations."""

    arrival_observation_enabled = True
    observation_space = 172


@configclass
class Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH2PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestrictedH2PhysicsEnvCfg
):
    """120 Hz static hold curriculum, 4 control steps (33.3 ms)."""

    decimation = 1
    stage_hold_steps = (12, 6, 6, 4, 4, 4)


@configclass
class Dex5P1P2StableManifold120HzH6PhysicsEnvCfg(
    Dex5P1P2ReverseR0ValidatedHoldRestricted120HzH1Bridge3StepLatchedObsPhysicsEnvCfg
):
    """Strict 100 ms hold task reset only from the PhysX-verified P2 core."""

    split_ids_path = P2_H6_STABLE_IDS
    split_name = "train"
    stage_hold_steps = (12, 6, 6, 12, 12, 12)
    hold_post_latch_action_scale = 1.0


@configclass
class Dex5P1P2ControlledExpansionE1_120HzH6PhysicsEnvCfg(
    Dex5P1P2StableManifold120HzH6PhysicsEnvCfg
):
    """E1 curriculum: stable P2 core plus policy-screened H6 frontier states."""

    split_ids_path = P2_E1_CONTROLLED_EXPANSION_IDS


@configclass
class Dex5P1P2ArrivalHoldH1PhysicsEnvCfg(Dex5P1P2ReverseR0ValidatedHoldPhysicsEnvCfg):
    """Arrival-to-hold curriculum from dynamic near-P2 states, 1 strict step."""

    bundle_path = ARRIVAL_BANK_512
    split_ids_path = ""
    use_all_bundle_rows = True
    source_phase_id = 2
    observation_space = 172
    episode_length_s = 0.35
    stage_hold_steps = (6, 3, 3, 1, 1, 1)
    hold_current_joint_targets_on_reset = True
    arrival_observation_enabled = True
    arrival_hold_reward_mode = True
    p2_capture_quality_reward_scale = 18.0
    p2_hold_reward_scale = 20.0
    p2_success_reward_scale = 45.0
    p2_pen_velocity_penalty_scale = 2.0
    p2_action_smoothness_penalty_scale = 0.18
    p2_joint_power_penalty_scale = 0.004


@configclass
class Dex5P1P2ArrivalHoldH2PhysicsEnvCfg(Dex5P1P2ArrivalHoldH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 2, 2, 2)


@configclass
class Dex5P1P2ArrivalHoldH3PhysicsEnvCfg(Dex5P1P2ArrivalHoldH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 3, 3, 3)


@configclass
class Dex5P1P2ArrivalHoldH4PhysicsEnvCfg(Dex5P1P2ArrivalHoldH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 4, 4, 4)


@configclass
class Dex5P1P2ArrivalHoldH5PhysicsEnvCfg(Dex5P1P2ArrivalHoldH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 5, 5, 5)


@configclass
class Dex5P1P2ArrivalHoldH6PhysicsEnvCfg(Dex5P1P2ArrivalHoldH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 6, 6, 6)
    episode_length_s = 0.45


@configclass
class Dex5P1P2ArrivalHoldRestrictedH1PhysicsEnvCfg(Dex5P1P2ArrivalHoldH1PhysicsEnvCfg):
    """Hold-only ablation: small TIM corrections, frozen-ish ring transfer action."""

    hold_action_mask_enabled = True
    hold_support_action_scale = 0.35
    hold_transfer_action_scale = 0.05
    hold_post_latch_action_scale = 0.18
    p2_action_l2_penalty_scale = 0.05
    p2_action_smoothness_penalty_scale = 0.24


@configclass
class Dex5P1P2ArrivalHoldRestrictedH2PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 2, 2, 2)


@configclass
class Dex5P1P2ArrivalHoldRestrictedH3PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 3, 3, 3)


@configclass
class Dex5P1P2ArrivalHoldRestrictedH4PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 4, 4, 4)


@configclass
class Dex5P1P2ArrivalHoldRestrictedH5PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 5, 5, 5)


@configclass
class Dex5P1P2ArrivalHoldRestrictedH6PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH1PhysicsEnvCfg):
    stage_hold_steps = (6, 3, 3, 6, 6, 6)
    episode_length_s = 0.45


@configclass
class Dex5P1P2ArrivalHoldRestricted120HzH1PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH1PhysicsEnvCfg):
    """Control-bandwidth ablation: same physics dt, policy acts every 8.3 ms."""

    decimation = 1


@configclass
class Dex5P1P2ArrivalHoldRestricted120HzH2PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH2PhysicsEnvCfg):
    decimation = 1


@configclass
class Dex5P1P2ArrivalHoldRestricted120HzH3PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH3PhysicsEnvCfg):
    decimation = 1


@configclass
class Dex5P1P2ArrivalHoldRestricted120HzH4PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH4PhysicsEnvCfg):
    decimation = 1


@configclass
class Dex5P1P2ArrivalHoldRestricted120HzH5PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH5PhysicsEnvCfg):
    decimation = 1


@configclass
class Dex5P1P2ArrivalHoldRestricted120HzH6PhysicsEnvCfg(Dex5P1P2ArrivalHoldRestrictedH6PhysicsEnvCfg):
    decimation = 1


@configclass
class Dex5P1P2FailureFrontier240_120HzH6PhysicsEnvCfg(
    Dex5P1P2ArrivalHoldRestricted120HzH6PhysicsEnvCfg
):
    """Strict dynamic hold task over real A2-A4 failure-frontier states."""

    bundle_path = FAILURE_FRONTIER_240
    stage_hold_steps = (12, 6, 6, 12, 12, 12)
    episode_length_s = 0.35


@configclass
class Dex5P1P2ReverseR1HoldPhysicsEnvCfg(Dex5P1P2ReverseR0HoldPhysicsEnvCfg):
    """R1 reverse curriculum: ring starts 2-4 mm from P2 geometry."""

    bundle_path = REVERSE_R1
    p2_ignore_progress_gate = False
    p2_ignore_progress_reward = False
    p2_geometry_soft_reward_scale = 0.0
    p2_triad_capture_soft_reward_scale = 0.0
    middle_contact_near_p2_reward_scale = 0.0


@configclass
class Dex5P1P2ReverseR2HoldPhysicsEnvCfg(Dex5P1P2ReverseR0HoldPhysicsEnvCfg):
    """R2 reverse curriculum: ring starts 4-8 mm from P2 geometry."""

    bundle_path = REVERSE_R2
    p2_ignore_progress_gate = False
    p2_ignore_progress_reward = False
    p2_geometry_soft_reward_scale = 0.0
    p2_triad_capture_soft_reward_scale = 0.0
    middle_contact_near_p2_reward_scale = 0.0


@configclass
class Dex5P1P2ReverseR3HoldPhysicsEnvCfg(Dex5P1P2ReverseR0HoldPhysicsEnvCfg):
    """R3 reverse curriculum: ring starts 8-12 mm from P2 geometry."""

    bundle_path = REVERSE_R3
    p2_ignore_progress_gate = False
    p2_ignore_progress_reward = False
    p2_geometry_soft_reward_scale = 0.0
    p2_triad_capture_soft_reward_scale = 0.0
    middle_contact_near_p2_reward_scale = 0.0


@configclass
class Dex5P1P2C3CaptureTeacherResidualPhysicsEnvCfg(
    Dex5P1P2C3TeacherResidualPhysicsEnvCfg
):
    """Tighten pen-axis control while retaining the exact P2 geometry."""

    p2_max_axis_drift_deg = 12.0
    stage_hold_steps = (6, 3, 3, 1, 6, 6)


@configclass
class Dex5P1P2C4PhysicsEnvCfg(Dex5P1P2PrimitivePhysicsEnvCfg):
    curriculum_stage = 4
    episode_length_s = 1.20


@configclass
class Dex5P1P2C5EasyDRPhysicsEnvCfg(Dex5P1P2PrimitivePhysicsEnvCfg):
    curriculum_stage = 5
    episode_length_s = 1.20
    reset_joint_pos_noise = 0.0008
    reset_joint_vel_noise = 0.025
    reset_pen_pos_noise = 0.00010
    reset_pen_vel_noise = 0.015
    reset_pen_rot_noise_rad = 0.00261799
    pen_static_friction_range = (0.50, 0.70)
    pen_dynamic_friction_range = (0.35, 0.55)


@configclass
class Dex5P1P2C4ValidationPhysicsEnvCfg(Dex5P1P2C4PhysicsEnvCfg):
    split_ids_path = VALIDATION_IDS
    split_name = "validation"


@configclass
class Dex5P1P2C4TestPhysicsEnvCfg(Dex5P1P2C4PhysicsEnvCfg):
    split_ids_path = TEST_IDS
    split_name = "test"


@configclass
class Dex5P1P2C5EasyDRTestPhysicsEnvCfg(Dex5P1P2C5EasyDRPhysicsEnvCfg):
    split_ids_path = TEST_IDS
    split_name = "test"


class Dex5P1P2PrimitivePhysicsEnv(Dex5RecaptureP3PhysicsEnv):
    cfg: Dex5P1P2PrimitivePhysicsEnvCfg

    def _load_bundle(self) -> None:
        debug = os.environ.get("DEX5_DEBUG_ENV_INIT") == "1"
        if debug:
            print("[DEX5_DEBUG] P1P2 _load_bundle: super begin", flush=True)
        super()._load_bundle()
        if debug:
            print("[DEX5_DEBUG] P1P2 _load_bundle: super done", flush=True)
        with np.load(self.cfg.bundle_path) as data:
            if debug:
                print(f"[DEX5_DEBUG] P1P2 _load_bundle: opened {self.cfg.bundle_path}", flush=True)
            phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)
            if self.cfg.use_all_bundle_rows:
                if self.cfg.source_phase_ids is None:
                    source_ids = np.flatnonzero(phase_ids == int(self.cfg.source_phase_id))
                else:
                    source_ids = np.flatnonzero(
                        np.isin(phase_ids, np.asarray(self.cfg.source_phase_ids, dtype=np.int64))
                    )
            else:
                split_ids = np.asarray(
                    np.load(self.cfg.split_ids_path, allow_pickle=False), dtype=np.int64
                )
                source_ids = split_ids[phase_ids[split_ids] == int(self.cfg.source_phase_id)]
            if source_ids.size == 0:
                raise ValueError("P1/P2 source split is empty")
            if debug:
                print(f"[DEX5_DEBUG] P1P2 _load_bundle: source_ids={source_ids.size}", flush=True)

            source_qpos = np.asarray(data["source_qpos"][source_ids], dtype=np.float32)
            source_qvel = np.asarray(data["source_qvel"][source_ids], dtype=np.float32)
            source_target = np.asarray(data["source_target_qpos"][source_ids], dtype=np.float32)
            if debug:
                print("[DEX5_DEBUG] P1P2 _load_bundle: source arrays loaded", flush=True)
            if "source_policy_action" in data:
                source_policy_action = np.asarray(
                    data["source_policy_action"][source_ids], dtype=np.float32
                )
            else:
                source_policy_action = np.zeros((source_ids.size, 20), dtype=np.float32)
            if "source_teacher_actions" in data:
                teacher_actions = np.asarray(
                    data["source_teacher_actions"][source_ids], dtype=np.float32
                )
            else:
                teacher_actions = np.zeros((source_ids.size, 1, 20), dtype=np.float32)
            if "target_qpos" in data:
                target_qpos_from_data = np.asarray(data["target_qpos"][source_ids], dtype=np.float32)
            else:
                target_qpos_from_data = None
            if "source_global_bank_ids" in data:
                global_bank_ids = np.asarray(
                    data["source_global_bank_ids"][source_ids], dtype=np.int64
                )
            else:
                global_bank_ids = source_ids.astype(np.int64)
            if debug:
                print("[DEX5_DEBUG] P1P2 _load_bundle: optional arrays loaded", flush=True)

        with np.load(self.cfg.target_statistics_bundle_path) as target_data:
            if debug:
                print(
                    f"[DEX5_DEBUG] P1P2 _load_bundle: opened target stats {self.cfg.target_statistics_bundle_path}",
                    flush=True,
                )
            target_phase_ids = np.asarray(target_data["phase_ids"], dtype=np.int64)
            statistics_ids = np.asarray(
                np.load(self.cfg.target_statistics_ids_path, allow_pickle=False), dtype=np.int64
            )
            p2_statistics_ids = statistics_ids[target_phase_ids[statistics_ids] == 2]
            if p2_statistics_ids.size == 0:
                raise ValueError("P2 train target statistics are empty")
            p2_hand_median = np.median(
                np.asarray(target_data["source_qpos"][p2_statistics_ids, :20], dtype=np.float32),
                axis=0,
            ).astype(np.float32)
            p2_target_median = np.median(
                np.asarray(target_data["source_target_qpos"][p2_statistics_ids], dtype=np.float32),
                axis=0,
            ).astype(np.float32)
            train_split_ids = np.asarray(np.load(TRAIN_IDS, allow_pickle=False), dtype=np.int64)
            train_source_ids = train_split_ids[target_phase_ids[train_split_ids] == 1]
            train_source_qpos = np.asarray(
                target_data["source_qpos"][train_source_ids], dtype=np.float32
            )
            if debug:
                print("[DEX5_DEBUG] P1P2 _load_bundle: target statistics loaded", flush=True)

        if target_qpos_from_data is None:
            target_qpos = source_qpos.copy()
            target_qpos[:, 8:12] = p2_target_median[None, 8:12]
        else:
            target_qpos = target_qpos_from_data
        if debug:
            print("[DEX5_DEBUG] P1P2 _load_bundle: tensor copy begin", flush=True)
        self.source_qpos_bank = torch.as_tensor(source_qpos, device=self.device)
        self.source_qvel_bank = torch.as_tensor(source_qvel, device=self.device)
        self.source_target_bank = torch.as_tensor(source_target, device=self.device)
        self.source_policy_action_bank = torch.as_tensor(source_policy_action, device=self.device)
        self.teacher_action_bank = torch.as_tensor(teacher_actions, device=self.device)
        self.teacher_action_horizon = self.teacher_action_bank.shape[1]
        self.target_hand_bank = torch.as_tensor(target_qpos[:, :20], device=self.device)
        self.target_pen_pos_bank = torch.as_tensor(target_qpos[:, 20:23], device=self.device)
        self.target_pen_quat_bank = torch.as_tensor(target_qpos[:, 23:27], device=self.device)
        self.phase_id_bank = torch.zeros(source_ids.size, dtype=torch.long, device=self.device)
        self.source_phase_id_bank = torch.as_tensor(
            phase_ids[source_ids], dtype=torch.long, device=self.device
        )
        self.global_bank_id_bank = torch.as_tensor(global_bank_ids, dtype=torch.long, device=self.device)
        self.phase_ids.zero_()
        self.bank_size = source_ids.size
        if debug:
            print("[DEX5_DEBUG] P1P2 _load_bundle: tensor copy done", flush=True)

        geometry = json.loads(Path(self.cfg.ring_geometry_path).read_text(encoding="utf-8"))
        self.p1_ring_center = torch.tensor(
            geometry["P1"]["median_m"], dtype=torch.float32, device=self.device
        )
        p2_ring_center = (
            self.cfg.target_p2_ring_center_override
            if self.cfg.target_p2_ring_center_override is not None
            else geometry["P2"]["median_m"]
        )
        self.p2_ring_center = torch.tensor(
            p2_ring_center, dtype=torch.float32, device=self.device
        )
        self.ring_transfer_vector = self.p2_ring_center - self.p1_ring_center
        self.ring_transfer_norm_sq = torch.sum(self.ring_transfer_vector.square()).clamp_min(1.0e-9)
        p2_target_ring_qpos = (
            self.cfg.target_p2_ring_qpos_override
            if self.cfg.target_p2_ring_qpos_override is not None
            else p2_target_median[8:12]
        )
        self.p2_target_ring_qpos = torch.tensor(
            p2_target_ring_qpos, dtype=torch.float32, device=self.device
        )
        self.p2_manifold_enabled = bool(self.cfg.p2_manifold_path)
        if self.p2_manifold_enabled:
            with np.load(self.cfg.p2_manifold_path, allow_pickle=False) as manifold:
                labels = np.asarray(manifold["manifold_mode"], dtype=np.int64)
                features = np.asarray(manifold["ring_feature_m"], dtype=np.float32)
                targets = np.asarray(manifold["source_target_qpos"], dtype=np.float32)[:, 8:12]
                route_parameters = (
                    np.asarray(manifold["search_parameters"], dtype=np.float32)
                    if "search_parameters" in manifold
                    else None
                )
            modes = np.unique(labels)
            if modes.size < 2:
                raise ValueError("P2 manifold must contain at least two verified modes")
            feature_centers = np.stack(
                [np.median(features[labels == mode], axis=0) for mode in modes]
            ).astype(np.float32)
            qpos_centers = np.stack(
                [np.median(targets[labels == mode], axis=0) for mode in modes]
            ).astype(np.float32)
            self.p2_manifold_mode_ids = torch.as_tensor(
                modes, dtype=torch.long, device=self.device
            )
            self.p2_manifold_centers = torch.as_tensor(
                feature_centers, dtype=torch.float32, device=self.device
            )
            self.p2_manifold_ring_qpos = torch.as_tensor(
                qpos_centers, dtype=torch.float32, device=self.device
            )
            if route_parameters is not None:
                duration_centers = np.stack(
                    [
                        np.median(route_parameters[labels == mode, 8:11], axis=0)
                        for mode in modes
                    ]
                ).astype(np.float32)
            else:
                duration_centers = np.full((len(modes), 3), 40.0, dtype=np.float32)
            self.p2_manifold_waypoint_durations = torch.as_tensor(
                duration_centers, dtype=torch.float32, device=self.device
            )
            self.p2_ring_center = torch.median(self.p2_manifold_centers, dim=0).values
            self.p2_target_ring_qpos = torch.median(
                self.p2_manifold_ring_qpos, dim=0
            ).values
        else:
            self.p2_manifold_mode_ids = torch.zeros(1, dtype=torch.long, device=self.device)
            self.p2_manifold_centers = self.p2_ring_center.unsqueeze(0)
            self.p2_manifold_ring_qpos = self.p2_target_ring_qpos.unsqueeze(0)
            self.p2_manifold_waypoint_durations = torch.full(
                (1, 3), 40.0, dtype=torch.float32, device=self.device
            )
        self.desired_p2_mode = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.fixed_p2_waypoint_durations = (
            torch.tensor(
                self.cfg.fixed_manifold_waypoint_durations,
                dtype=torch.float32,
                device=self.device,
            )
            if self.cfg.fixed_manifold_waypoint_durations is not None
            else None
        )
        self.initial_ring_feature = torch.zeros((self.num_envs, 3), device=self.device)
        self.policy_targets = torch.zeros((self.num_envs, 20), device=self.device)
        self.support_anchor_targets = torch.zeros((self.num_envs, 20), device=self.device)
        self.p1_p2_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.initial_stage_target_distance = torch.zeros(self.num_envs, device=self.device)
        self.previous_stage_target_distance = torch.zeros(self.num_envs, device=self.device)
        self.episode_p1_p2_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completed_p1_p2_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_p1_p2_successes = torch.zeros((), dtype=torch.long, device=self.device)
        self.last_completed_p1_p2_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_completed_p1_p2_drop = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_completed_global_bank_ids = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.completed_drop_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_invalid_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_max_penetration = torch.zeros((), device=self.device)
        self.current_global_bank_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.teacher_brake_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.teacher_brake_targets = torch.zeros((self.num_envs, 4), device=self.device)
        self.teacher_c3_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.teacher_c3_entry_hold_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.teacher_ring_joint_fractions_bank = torch.empty(
            (source_ids.size, 4), dtype=torch.float32, device=self.device
        )
        self.teacher_route_success_bank = torch.zeros(
            source_ids.size, dtype=torch.bool, device=self.device
        )
        self.primary_contact_grace = torch.zeros(
            (self.num_envs, 3), dtype=torch.long, device=self.device
        )
        self.arrival_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminal_correction_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.previous_terminal_condition = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.terminal_condition_delta = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.terminal_semantic_hold_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.previous_ring_target_error = torch.zeros((self.num_envs, 3), device=self.device)
        if debug:
            print("[DEX5_DEBUG] P1P2 _load_bundle: state buffers done", flush=True)
        if self.cfg.teacher_prior_state_map_path:
            if debug:
                print("[DEX5_DEBUG] P1P2 _load_bundle: teacher map begin", flush=True)
            with np.load(self.cfg.teacher_prior_state_map_path) as route_data:
                selected_local_ids = np.asarray(route_data["selected_local_ids"], dtype=np.int64)
                selected_global_ids = np.asarray(route_data["selected_global_ids"], dtype=np.int64)
                candidates = np.asarray(route_data["candidates"], dtype=np.float32)
                best_candidate_ids = np.asarray(route_data["best_candidate_ids"], dtype=np.int64)
                success_matrix = np.asarray(route_data["success_matrix"], dtype=np.bool_)
            route_by_global = {
                int(global_id): (
                    candidates[best_candidate_ids[row]],
                    bool(success_matrix[row, best_candidate_ids[row]]),
                )
                for row, global_id in enumerate(selected_global_ids)
            }
            train_routes = np.empty((train_source_ids.size, 4), dtype=np.float32)
            train_route_success = np.zeros(train_source_ids.size, dtype=np.bool_)
            for local_id, global_id in enumerate(train_source_ids):
                if int(global_id) not in route_by_global:
                    raise ValueError(f"State teacher map misses train P1 bank ID {int(global_id)}")
                train_routes[local_id], train_route_success[local_id] = route_by_global[int(global_id)]

            if self.cfg.split_name == "train":
                route_rows = np.asarray(
                    [np.flatnonzero(train_source_ids == global_id)[0] for global_id in source_ids],
                    dtype=np.int64,
                )
            else:
                feature_mean = train_source_qpos.mean(axis=0, keepdims=True)
                feature_scale = train_source_qpos.std(axis=0, keepdims=True).clip(min=1.0e-4)
                train_features = (train_source_qpos - feature_mean) / feature_scale
                source_features = (source_qpos - feature_mean) / feature_scale
                distances = np.sum(
                    (source_features[:, None, :] - train_features[None, :, :]) ** 2,
                    axis=-1,
                )
                route_rows = np.argmin(distances, axis=1)
            self.teacher_ring_joint_fractions_bank.copy_(
                torch.as_tensor(train_routes[route_rows], device=self.device)
            )
            self.teacher_route_success_bank.copy_(
                torch.as_tensor(train_route_success[route_rows], device=self.device)
            )
            if debug:
                print("[DEX5_DEBUG] P1P2 _load_bundle: teacher map done", flush=True)
        elif self.cfg.teacher_prior_ring_joint_fractions is None:
            self.teacher_ring_joint_fractions_bank.fill_(float(self.cfg.teacher_prior_ring_fraction))
        else:
            self.teacher_ring_joint_fractions_bank.copy_(
                torch.tensor(
                    self.cfg.teacher_prior_ring_joint_fractions,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0).expand(source_ids.size, -1)
            )
        if debug:
            print("[DEX5_DEBUG] P1P2 _load_bundle: complete", flush=True)

    def _ring_axis_features(self) -> torch.Tensor:
        axis = self.pen_axis
        world_up = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand_as(axis)
        up = world_up - torch.sum(world_up * axis, dim=-1, keepdim=True) * axis
        up = torch.nn.functional.normalize(up, dim=-1)
        lateral = torch.nn.functional.normalize(torch.cross(axis, up, dim=-1), dim=-1)
        ring_rel = self.fingertip_pos_local[:, 3] - self.pen_pos
        return torch.stack(
            (
                torch.sum(ring_rel * axis, dim=-1),
                torch.sum(ring_rel * lateral, dim=-1),
                torch.sum(ring_rel * up, dim=-1),
            ),
            dim=-1,
        )

    def _ring_geometry(
        self, *, force_nearest: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self._ring_axis_features()
        stage = int(self.cfg.curriculum_stage)
        target_progress = float(self.cfg.stage_target_progress[stage])
        mode_targets = self.initial_ring_feature.unsqueeze(1) + target_progress * (
            self.p2_manifold_centers.unsqueeze(0) - self.initial_ring_feature.unsqueeze(1)
        )
        mode_distances = torch.linalg.norm(feature.unsqueeze(1) - mode_targets, dim=-1)
        nearest_mode = torch.argmin(mode_distances, dim=-1)
        rows = torch.arange(self.num_envs, device=self.device)
        self.nearest_p2_mode = nearest_mode
        selected_mode = nearest_mode
        if self.cfg.conditioned_reward_uses_desired_mode and not force_nearest:
            selected_mode = self.desired_p2_mode
        target_center = mode_targets[rows, selected_mode]
        target_distance = mode_distances[rows, selected_mode]
        self.nearest_p2_target_center = target_center
        transfer_vector = target_center - self.initial_ring_feature
        transfer_norm_sq = torch.sum(transfer_vector.square(), dim=-1).clamp_min(1.0e-9)
        delta = feature - self.initial_ring_feature
        progress = torch.sum(delta * transfer_vector, dim=-1) / transfer_norm_sq
        perpendicular = delta - progress.unsqueeze(-1) * transfer_vector
        perpendicular_distance = torch.linalg.norm(perpendicular, dim=-1)
        return feature, progress, target_distance, perpendicular_distance

    def _terminal_condition_metrics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature, _, target_distance, _ = self._ring_geometry(force_nearest=True)
        feature_error = (
            (self.nearest_p2_target_center - feature) / 0.010
        ).clamp(-3.0, 3.0)
        ring_gap = self.tip_surface_dist[:, 3] - self.fingertip_radii[0, 3]
        gap_error = ((0.0012 - ring_gap) / 0.003).clamp(-2.0, 2.0)
        axial, lateral, up = feature.unbind(dim=-1)
        margin_terms = torch.stack(
            (
                axial + 0.060,
                -0.020 - axial,
                lateral + 0.010,
                0.026 - lateral,
                up + 0.026,
                -0.004 - up,
                -up - torch.abs(lateral) - 0.001,
                ring_gap + 0.0006,
                0.0030 - ring_gap,
            ),
            dim=-1,
        )
        semantic_margin = (margin_terms.min(dim=-1).values / 0.003).clamp(-3.0, 3.0)
        condition = torch.cat((feature_error, gap_error.unsqueeze(-1)), dim=-1)
        return condition, semantic_margin, target_distance

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        feature, progress, target_distance, _ = self._ring_geometry()
        controller_error = self.policy_targets - self.hand_qpos
        stage = torch.full(
            (self.num_envs,), int(self.cfg.curriculum_stage), dtype=torch.long, device=self.device
        )
        stage_one_hot = torch.nn.functional.one_hot(stage, num_classes=6).float()
        rows = torch.arange(self.num_envs, device=self.device)
        target_error = feature - self.nearest_p2_target_center
        observations["policy"] = torch.cat(
            (
                observations["policy"],
                feature,
                progress.unsqueeze(-1),
                target_error,
                controller_error,
                stage_one_hot,
            ),
            dim=-1,
        )
        if self.cfg.arrival_observation_enabled:
            hold_steps = max(int(self.cfg.stage_hold_steps[int(self.cfg.curriculum_stage)]), 1)
            normalized_hold = torch.clamp(
                self.p1_p2_hold_count.float() / hold_steps, 0.0, 1.0
            )
            target_error_delta = target_error - self.previous_ring_target_error
            arrival_obs = torch.cat(
                (
                    self.arrival_latched.float().unsqueeze(-1),
                    normalized_hold.unsqueeze(-1),
                    target_error,
                    target_error_delta,
                ),
                dim=-1,
            )
            observations["policy"] = torch.cat((observations["policy"], arrival_obs), dim=-1)
        if self.cfg.teacher_phase_observation_enabled:
            teacher_phase = torch.nn.functional.one_hot(
                self.teacher_c3_phase.clamp(0, 3), num_classes=4
            ).float()
            observations["policy"] = torch.cat(
                (observations["policy"], teacher_phase), dim=-1
            )
        if self.cfg.manifold_condition_observation_enabled:
            desired_mode = torch.nn.functional.one_hot(
                self.desired_p2_mode.clamp(0, self.p2_manifold_centers.shape[0] - 1),
                num_classes=self.p2_manifold_centers.shape[0],
            ).float()
            observations["policy"] = torch.cat(
                (observations["policy"], desired_mode), dim=-1
            )
        if self.cfg.manifold_waypoint_phase_observation_enabled:
            durations = self.p2_manifold_waypoint_durations[self.desired_p2_mode]
            if self.fixed_p2_waypoint_durations is not None:
                durations = self.fixed_p2_waypoint_durations.unsqueeze(0).expand(
                    self.num_envs, -1
                )
            elapsed = self.episode_length_buf.float() + 1.0
            d1 = durations[:, 0]
            d12 = d1 + durations[:, 1]
            d123 = d12 + durations[:, 2]
            waypoint_phase = torch.where(
                elapsed <= d1,
                torch.zeros_like(self.desired_p2_mode),
                torch.where(
                    elapsed <= d12,
                    torch.ones_like(self.desired_p2_mode),
                    torch.where(
                        elapsed <= d123,
                        torch.full_like(self.desired_p2_mode, 2),
                        torch.full_like(self.desired_p2_mode, 3),
                    ),
                ),
            )
            if self.cfg.local_arrival_final_waypoint_observation:
                waypoint_phase.fill_(3)
            elif self.cfg.mixed_route_arrival_observation:
                arrival_rows = self.source_phase_id_bank[self.current_bank_ids] == 2
                waypoint_phase = torch.where(
                    arrival_rows, torch.full_like(waypoint_phase, 3), waypoint_phase
                )
            phase_one_hot = torch.nn.functional.one_hot(
                waypoint_phase, num_classes=4
            ).float()
            observations["policy"] = torch.cat(
                (observations["policy"], phase_one_hot), dim=-1
            )
        if self.cfg.waypoint_target_error_observation_enabled:
            source = self.source_targets[:, 8:12]
            clearance = source + torch.tensor(
                self.cfg.fixed_route_clearance_offset,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            waypoint2 = source + torch.tensor(
                self.cfg.fixed_route_waypoint2_offset,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            endpoint = source + torch.tensor(
                self.cfg.fixed_route_endpoint_offset,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            durations = self.fixed_p2_waypoint_durations
            elapsed = self.episode_length_buf.float() + 1.0

            def blend(value: torch.Tensor) -> torch.Tensor:
                value = value.clamp(0.0, 1.0)
                return value.square() * (3.0 - 2.0 * value)

            d1, d2, d3 = durations.unbind()
            first = source + blend(elapsed / d1).unsqueeze(-1) * (clearance - source)
            second = clearance + blend((elapsed - d1) / d2).unsqueeze(-1) * (
                waypoint2 - clearance
            )
            third = waypoint2 + blend((elapsed - d1 - d2) / d3).unsqueeze(-1) * (
                endpoint - waypoint2
            )
            if self.cfg.local_arrival_final_waypoint_observation:
                rows = torch.arange(self.num_envs, device=self.device)
                desired = self.p2_manifold_ring_qpos[
                    self.desired_p2_mode[rows]
                ]
            else:
                desired = torch.where(
                    (elapsed <= d1).unsqueeze(-1),
                    first,
                    torch.where((elapsed <= d1 + d2).unsqueeze(-1), second, third),
                )
                if self.cfg.mixed_route_arrival_observation:
                    rows = torch.arange(self.num_envs, device=self.device)
                    endpoint_desired = self.p2_manifold_ring_qpos[
                        self.desired_p2_mode[rows]
                    ]
                    arrival_rows = self.source_phase_id_bank[self.current_bank_ids] == 2
                    desired = torch.where(
                        arrival_rows.unsqueeze(-1), endpoint_desired, desired
                    )
            waypoint_error = desired - self.policy_targets[:, 8:12]
            observations["policy"] = torch.cat(
                (observations["policy"], waypoint_error), dim=-1
            )
        if self.cfg.terminal_correction_observation_enabled:
            _, _, terminal_distance, _ = self._ring_geometry(force_nearest=True)
            terminal_correction = (
                terminal_distance <= float(self.cfg.terminal_correction_distance_m)
            ).float().unsqueeze(-1)
            observations["policy"] = torch.cat(
                (observations["policy"], terminal_correction), dim=-1
            )
        if self.cfg.terminal_correction_phase_observation_enabled:
            denominator = max(
                int(self.cfg.terminal_correction_phase_horizon_steps) - 1, 1
            )
            correction_phase = (
                self.terminal_correction_step.float() / float(denominator)
            ).clamp(0.0, 1.0).unsqueeze(-1)
            observations["policy"] = torch.cat(
                (observations["policy"], correction_phase), dim=-1
            )
        if self.cfg.terminal_correction_target_observation_enabled:
            target_condition, _, terminal_distance = self._terminal_condition_metrics()
            terminal_active = (
                terminal_distance <= float(self.cfg.terminal_correction_distance_m)
            ).float().unsqueeze(-1)
            target_condition = target_condition * terminal_active
            observations["policy"] = torch.cat(
                (observations["policy"], target_condition), dim=-1
            )
        if self.cfg.terminal_correction_dynamics_observation_enabled:
            _, semantic_margin, terminal_distance = self._terminal_condition_metrics()
            self.terminal_semantic_hold_latched |= (
                (terminal_distance <= 0.002431) & (semantic_margin >= 0.0)
            )
            dynamics = torch.cat(
                (
                    semantic_margin.unsqueeze(-1),
                    self.terminal_condition_delta,
                    self.terminal_semantic_hold_latched.float().unsqueeze(-1),
                ),
                dim=-1,
            )
            observations["policy"] = torch.cat((observations["policy"], dynamics), dim=-1)
        return observations

    def _action_mask(self) -> torch.Tensor:
        mask = torch.zeros((self.num_envs, 20), device=self.device)
        support_scale = float(self.cfg.support_action_mask_scale)
        mask[:, 0:8] = support_scale
        mask[:, 16:20] = support_scale
        if int(self.cfg.curriculum_stage) > 0:
            mask[:, 8:12] = 1.0
        if self.cfg.hold_action_mask_enabled:
            support = float(self.cfg.hold_support_action_scale)
            transfer = float(self.cfg.hold_transfer_action_scale)
            mask.zero_()
            mask[:, 0:8] = support
            mask[:, 16:20] = support
            mask[:, 8:12] = transfer
            if self.cfg.arrival_hold_reward_mode:
                latched = self.arrival_latched.unsqueeze(-1)
                post_scale = float(self.cfg.hold_post_latch_action_scale)
                mask[:, 0:8] = torch.where(
                    latched, torch.full_like(mask[:, 0:8], support * post_scale), mask[:, 0:8]
                )
                mask[:, 16:20] = torch.where(
                    latched, torch.full_like(mask[:, 16:20], support * post_scale), mask[:, 16:20]
                )
                mask[:, 8:12] = torch.where(
                    latched, torch.full_like(mask[:, 8:12], transfer * post_scale), mask[:, 8:12]
                )
        if self.cfg.mixed_route_arrival_observation:
            arrival_rows = self.source_phase_id_bank[self.current_bank_ids] == 2
            mask = torch.where(
                arrival_rows.unsqueeze(-1),
                float(self.cfg.mixed_arrival_action_scale) * mask,
                mask,
            )
        return mask

    def _teacher_prior_actions(self) -> torch.Tensor:
        teacher = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        increment = 0.25 * self.cfg.action_scale
        ring_source = self.source_targets[:, 8:12]
        ring_final = self.p2_target_ring_qpos.unsqueeze(0).expand_as(ring_source)
        fractions = self.teacher_ring_joint_fractions_bank[self.current_bank_ids]
        entry_ring_target = ring_source + fractions * (ring_final - ring_source)
        entry_ring_target = torch.maximum(
            torch.minimum(entry_ring_target, self.joint_upper[:, 8:12]),
            self.joint_lower[:, 8:12],
        )
        ring_target = entry_ring_target
        final_action_scale = torch.ones(self.num_envs, device=self.device)

        if self.cfg.teacher_prior_mode == "c3_distal_then_proximal":
            feature, progress, _, _ = self._ring_geometry()
            transfer_vector = self.p2_ring_center.unsqueeze(0) - self.initial_ring_feature
            entry_center = (
                self.initial_ring_feature
                + float(self.cfg.teacher_c3_entry_progress) * transfer_vector
            )
            entry_distance = torch.linalg.norm(feature - entry_center, dim=-1)
            entry_geometry = (
                torch.abs(progress - float(self.cfg.teacher_c3_entry_progress))
                <= float(self.cfg.teacher_c3_entry_progress_tolerance)
            ) & (entry_distance <= float(self.cfg.teacher_c3_entry_distance_m))
            entry_contact = (self.primary_contact_grace > 0).all(dim=-1)
            entry_eligible = (self.teacher_c3_phase == 0) & entry_geometry & entry_contact
            self.teacher_c3_entry_hold_count = torch.where(
                entry_eligible,
                self.teacher_c3_entry_hold_count + 1,
                torch.zeros_like(self.teacher_c3_entry_hold_count),
            )
            enter_roll = (
                (self.teacher_c3_phase == 0)
                & (
                    self.teacher_c3_entry_hold_count
                    >= int(self.cfg.teacher_c3_entry_hold_steps)
                )
            )
            self.teacher_c3_phase[enter_roll] = 1

            endpoint_fractions = torch.tensor(
                self.cfg.teacher_c3_endpoint_fractions,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            endpoint_target = ring_source + endpoint_fractions * (ring_final - ring_source)
            endpoint_target = torch.maximum(
                torch.minimum(endpoint_target, self.joint_upper[:, 8:12]),
                self.joint_lower[:, 8:12],
            )
            roll_target = entry_ring_target.clone()
            roll_target[:, 0] = endpoint_target[:, 0]
            roll_error = torch.abs(self.hand_qpos[:, 8] - roll_target[:, 0])
            enter_distal = (
                (self.teacher_c3_phase == 1)
                & (roll_error <= float(self.cfg.teacher_c3_sequence_joint_tolerance_rad))
            )
            self.teacher_c3_phase[enter_distal] = 2

            distal_target = roll_target.clone()
            distal_target[:, 2:4] = endpoint_target[:, 2:4]
            distal_error = torch.amax(
                torch.abs(self.hand_qpos[:, 10:12] - distal_target[:, 2:4]), dim=-1
            )
            enter_proximal = (
                (self.teacher_c3_phase == 2)
                & (distal_error <= float(self.cfg.teacher_c3_sequence_joint_tolerance_rad))
            )
            self.teacher_c3_phase[enter_proximal] = 3

            ring_target = torch.where(
                (self.teacher_c3_phase == 0).unsqueeze(-1),
                entry_ring_target,
                torch.where(
                    (self.teacher_c3_phase == 1).unsqueeze(-1),
                    roll_target,
                    torch.where(
                        (self.teacher_c3_phase == 2).unsqueeze(-1),
                        distal_target,
                        endpoint_target,
                    ),
                ),
            )
            final_action_scale = torch.where(
                self.teacher_c3_phase >= 3,
                torch.full_like(final_action_scale, float(self.cfg.teacher_c3_final_action_scale)),
                final_action_scale,
            )
        if self.cfg.teacher_prior_geometric_brake:
            _, progress, target_distance, _ = self._ring_geometry()
            stage = int(self.cfg.curriculum_stage)
            target_progress = float(self.cfg.stage_target_progress[stage])
            should_brake = (
                (progress >= target_progress - self.cfg.teacher_prior_brake_progress_margin)
                & (target_distance <= self.cfg.teacher_prior_brake_distance_m)
            )
            if self.cfg.teacher_prior_mode == "c3_distal_then_proximal":
                should_brake &= self.teacher_c3_phase >= 3
            newly_braked = should_brake & ~self.teacher_brake_latched
            self.teacher_brake_targets[newly_braked] = self.hand_qpos[newly_braked, 8:12]
            self.teacher_brake_latched |= should_brake
            ring_target = torch.where(
                self.teacher_brake_latched.unsqueeze(-1), self.teacher_brake_targets, ring_target
            )
        teacher[:, 8:12] = torch.clamp(
            (ring_target - self.policy_targets[:, 8:12]) / increment,
            -1.0,
            1.0,
        ) * final_action_scale.unsqueeze(-1)
        for value, joint_slice in zip(
            self.cfg.teacher_prior_support_actions,
            (slice(1, 4), slice(5, 8), slice(17, 20)),
            strict=True,
        ):
            teacher[:, joint_slice] = float(value)
        return teacher

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.terminal_correction_phase_observation_enabled:
            _, _, terminal_distance, _ = self._ring_geometry(force_nearest=True)
            terminal_active = terminal_distance <= float(
                self.cfg.terminal_correction_distance_m
            )
            self.terminal_correction_step = torch.where(
                terminal_active,
                self.terminal_correction_step + 1,
                torch.zeros_like(self.terminal_correction_step),
            )
        if self.cfg.terminal_correction_dynamics_observation_enabled:
            condition, semantic_margin, terminal_distance = (
                self._terminal_condition_metrics()
            )
            self.terminal_condition_delta = condition - self.previous_terminal_condition
            self.previous_terminal_condition.copy_(condition)
            terminal_active = terminal_distance <= float(
                self.cfg.terminal_correction_distance_m
            )
            self.terminal_semantic_hold_latched |= (
                terminal_active
                & (terminal_distance <= 0.002431)
                & (semantic_margin >= 0.0)
            )
        self.previous_actions.copy_(self.actions)
        policy_actions = torch.clamp(actions, -1.0, 1.0)
        if self.cfg.teacher_prior_enabled:
            policy_actions = self._teacher_prior_actions() + (
                float(self.cfg.teacher_residual_scale) * policy_actions
            )
        self.actions = torch.clamp(policy_actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        mask = self._action_mask()
        # Ring uses incremental targets to traverse the large P1-to-P2 joint range.
        self.policy_targets[:, 8:12] += (
            0.25 * self.cfg.action_scale * self.actions[:, 8:12] * mask[:, 8:12]
        )
        # Primary support fingers use bounded residuals around their P1 targets so
        # a small stationary policy bias cannot accumulate until contact is lost.
        for joint_slice in (slice(0, 8), slice(16, 20)):
            self.policy_targets[:, joint_slice] = (
                self.support_anchor_targets[:, joint_slice]
                + self.cfg.action_scale
                * self.actions[:, joint_slice]
                * mask[:, joint_slice]
            )
        self.policy_targets = torch.maximum(
            torch.minimum(self.policy_targets, self.joint_upper), self.joint_lower
        )
        self.processed_targets = self.policy_targets
        self.hand.set_joint_position_target(
            self.processed_targets, joint_ids=self.dex5_joint_ids
        )

    def _instantaneous_gate(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature, progress, target_distance, perpendicular_distance = self._ring_geometry(
            force_nearest=True
        )
        triad = (self.primary_contact_grace > 0).all(dim=-1)
        position_drift = torch.linalg.norm(self.pen_pos - self.init_pen_pos, dim=-1)
        axis_dot = torch.sum(self.pen_axis * self.init_pen_axis, dim=-1).clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot.abs())
        angular_speed = torch.linalg.norm(self.pen_angvel, dim=-1)
        linear_speed = torch.linalg.norm(self.pen_linvel, dim=-1)
        force_sum = self.contact_force_norm.sum(dim=-1).clamp_min(1.0e-6)
        ring_force_share = self.contact_force_norm[:, 3] / force_sum
        stage = int(self.cfg.curriculum_stage)
        target_progress = float(self.cfg.stage_target_progress[stage])
        progress_error = torch.abs(progress - target_progress)
        failed, _ = super()._compute_dones()
        if self.cfg.p2_ignore_progress_gate:
            geometry_gate = target_distance <= self.cfg.stage_target_distance_m[stage]
        else:
            geometry_gate = (
                (progress_error <= self.cfg.stage_progress_tolerance[stage])
                & (target_distance <= self.cfg.stage_target_distance_m[stage])
            )
        if stage == 0:
            geometry_gate = torch.ones_like(geometry_gate)
        manifold_semantic = torch.ones_like(geometry_gate)
        manifold_feature_bounds = torch.ones_like(geometry_gate)
        manifold_under = torch.ones_like(geometry_gate)
        manifold_gap_gate = torch.ones_like(geometry_gate)
        ring_gap = torch.zeros_like(target_distance)
        manifold_semantic_soft = torch.ones_like(target_distance)
        if self.p2_manifold_enabled:
            axial, lateral, up = feature.unbind(dim=-1)
            ring_gap = self.tip_surface_dist[:, 3] - self.fingertip_radii[0, 3]
            manifold_feature_bounds = (
                (axial >= -0.060)
                & (axial <= -0.020)
                & (lateral >= -0.010)
                & (lateral <= 0.026)
                & (up >= -0.026)
                & (up <= -0.004)
            )
            manifold_under = (-up) >= torch.abs(lateral) + 0.001
            manifold_gap_gate = (ring_gap >= -0.0006) & (ring_gap <= 0.0030)
            manifold_semantic = (
                manifold_feature_bounds & manifold_under & manifold_gap_gate
            )
            bound_violation = (
                torch.relu(-0.060 - axial)
                + torch.relu(axial + 0.020)
                + torch.relu(-0.010 - lateral)
                + torch.relu(lateral - 0.026)
                + torch.relu(-0.026 - up)
                + torch.relu(up + 0.004)
            )
            bounds_soft = torch.exp(-bound_violation / 0.004)
            under_soft = torch.exp(
                -torch.relu(torch.abs(lateral) + 0.001 + up) / 0.003
            )
            gap_soft = torch.exp(-torch.abs(ring_gap - 0.0012) / 0.0018)
            manifold_semantic_soft = bounds_soft * under_soft * gap_soft
            if self.cfg.p2_success_gate_mode == "semantic":
                geometry_gate &= manifold_semantic
            elif self.cfg.p2_success_gate_mode == "physical_contact":
                geometry_gate &= self.contact_force_norm[:, 3] > self.cfg.p2_contact_force_n
            else:
                raise ValueError(
                    f"Unknown P2 success gate mode: {self.cfg.p2_success_gate_mode}"
                )
        eligible = (
            geometry_gate
            & triad
            & (position_drift <= self.cfg.p2_max_position_drift_m)
            & (
                axis_drift
                <= torch.deg2rad(torch.full_like(axis_drift, self.cfg.p2_max_axis_drift_deg))
            )
            & (angular_speed <= self.cfg.p2_max_angular_speed_rad_s)
            & (linear_speed <= self.cfg.p2_max_linear_speed_m_s)
            & (self.episode_max_penetration <= self.cfg.p2_max_penetration_m)
            & (ring_force_share <= self.cfg.p2_max_ring_force_share)
            & (self.contact_force_norm[:, 3] <= self.cfg.p2_max_ring_force_n)
            & (self.contact_force_norm[:, 4] <= self.cfg.p2_max_pinky_force_n)
            & ~failed
        )
        metrics = {
            "progress": progress,
            "progress_error": progress_error,
            "target_distance": target_distance,
            "perpendicular_distance": perpendicular_distance,
            "triad": triad.float(),
            "position_drift": position_drift,
            "axis_drift": axis_drift,
            "angular_speed": angular_speed,
            "linear_speed": linear_speed,
            "ring_force_share": ring_force_share,
            "failed": failed.float(),
            "manifold_semantic": manifold_semantic.float(),
            "manifold_feature_bounds": manifold_feature_bounds.float(),
            "manifold_under": manifold_under.float(),
            "manifold_gap_gate": manifold_gap_gate.float(),
            "ring_gap": ring_gap,
            "manifold_semantic_soft": manifold_semantic_soft,
            "ring_contact": (
                self.contact_force_norm[:, 3] > self.cfg.p2_contact_force_n
            ).float(),
            "nearest_manifold_mode": self.nearest_p2_mode.float(),
        }
        if self.cfg.conditioned_reward_uses_desired_mode:
            _, desired_progress, desired_distance, desired_perpendicular = (
                self._ring_geometry(force_nearest=False)
            )
            metrics["progress"] = desired_progress
            metrics["progress_error"] = torch.abs(desired_progress - target_progress)
            metrics["target_distance"] = desired_distance
            metrics["perpendicular_distance"] = desired_perpendicular
        return eligible, metrics

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        primary_contacts = self.contact_force_norm[:, :3] > self.cfg.p2_contact_force_n
        self.primary_contact_grace = torch.where(
            primary_contacts,
            torch.full_like(self.primary_contact_grace, self.cfg.p2_contact_grace_steps),
            torch.clamp(self.primary_contact_grace - 1, min=0),
        )
        eligible, metrics = self._instantaneous_gate()
        stage = int(self.cfg.curriculum_stage)
        hold_steps = int(self.cfg.stage_hold_steps[stage])
        self.p1_p2_hold_count = torch.where(
            eligible, self.p1_p2_hold_count + 1, torch.zeros_like(self.p1_p2_hold_count)
        )
        success = self.p1_p2_hold_count >= hold_steps
        self.episode_p1_p2_success |= success

        target_radius = float(self.cfg.stage_target_distance_m[stage])
        if self.cfg.arrival_hold_reward_mode:
            self.arrival_latched |= metrics["target_distance"] <= target_radius
        required_reduction = (
            self.initial_stage_target_distance - target_radius
        ).clamp_min(0.002)
        transfer_progress = torch.clamp(
            (self.initial_stage_target_distance - metrics["target_distance"])
            / required_reduction,
            -1.0,
            1.0,
        )
        distance_delta = torch.clamp(
            (self.previous_stage_target_distance - metrics["target_distance"]) / 0.002,
            -1.0,
            1.0,
        )
        target_excess = torch.relu(metrics["target_distance"] - target_radius)
        target_score = torch.exp(-target_excess / 0.004)
        primary_contacts = self.contact_force_norm[:, :3] > self.cfg.p2_contact_force_n
        triad_ratio = primary_contacts.float().mean(dim=-1)
        transfer_weight = torch.clamp(metrics["progress"], 0.0, 1.0)
        middle_contact = primary_contacts[:, 2].float()
        transfer_axis_score = torch.exp(-12.0 * metrics["axis_drift"])
        position_score = torch.exp(-120.0 * metrics["position_drift"])
        axis_score = torch.exp(-8.0 * metrics["axis_drift"])
        if self.cfg.p2_ignore_progress_reward:
            progress_score = torch.ones_like(target_score)
            target_score = torch.exp(
                -0.5 * (metrics["target_distance"] / max(target_radius, 1.0e-6)).square()
            )
            transfer_progress = target_score
            distance_delta = torch.clamp(
                (self.previous_stage_target_distance - metrics["target_distance"]) / 0.001,
                -1.0,
                1.0,
            )
            transfer_weight = torch.zeros_like(target_score)
        else:
            progress_score = torch.exp(-8.0 * metrics["progress_error"])
        near_target_weight = torch.clamp(
            1.0 - metrics["target_distance"] / 0.018, 0.0, 1.0
        )
        geometry_soft = target_score * progress_score
        triad_capture_soft = geometry_soft * triad_ratio * axis_score
        middle_capture_soft = near_target_weight * progress_score * middle_contact
        capture_quality = (
            target_score
            * progress_score
            * metrics["triad"]
            * axis_score
        )
        speed_penalty = torch.clamp(metrics["angular_speed"] / 12.0, 0.0, 2.0) + torch.clamp(
            metrics["linear_speed"] / 0.40, 0.0, 2.0
        )
        ring_overload = torch.relu(metrics["ring_force_share"] - 0.10) + torch.relu(
            self.contact_force_norm[:, 3] - 0.30
        )
        pinky_load = torch.clamp(self.contact_force_norm[:, 4] / 0.10, 0.0, 3.0)
        penetration_excess = torch.relu(
            (self.episode_max_penetration - 0.00025) / 0.00035
        )
        joint_power = torch.sum(
            torch.abs(self.hand.data.applied_torque[:, self.dex5_joint_ids] * self.hand_qvel), dim=-1
        )
        hold_progress = torch.clamp(self.p1_p2_hold_count.float() / max(hold_steps, 1), 0.0, 1.0)
        terms = {
            "alive": torch.full_like(transfer_progress, 0.05),
            "ring_transfer_progress": 6.0 * transfer_progress,
            "ring_distance_delta": 2.0 * distance_delta,
            "ring_target_region": 6.0 * target_score,
            "triad_contacts": 6.0 * triad_ratio + 6.0 * metrics["triad"],
            "middle_contact_during_transfer": 6.0
            * transfer_weight
            * middle_contact,
            "stable_axis_during_transfer": 8.0
            * transfer_weight
            * transfer_axis_score,
            "geometry_with_triad": 10.0 * target_score * metrics["triad"],
            "p2_geometry_soft": float(self.cfg.p2_geometry_soft_reward_scale)
            * geometry_soft,
            "p2_triad_capture_soft": float(self.cfg.p2_triad_capture_soft_reward_scale)
            * triad_capture_soft,
            "middle_contact_near_p2": float(self.cfg.middle_contact_near_p2_reward_scale)
            * middle_capture_soft,
            "p2_capture_quality": float(self.cfg.p2_capture_quality_reward_scale)
            * capture_quality,
            "p2_manifold_semantic": float(self.cfg.p2_manifold_semantic_reward_scale)
            * metrics["manifold_semantic_soft"],
            "p2_hold": float(self.cfg.p2_hold_reward_scale) * hold_progress,
            "success": float(self.cfg.p2_success_reward_scale) * success.float(),
            "pen_position_stability": float(self.cfg.p2_pen_position_reward_scale)
            * position_score,
            "pen_axis_stability": float(self.cfg.p2_pen_axis_reward_scale) * axis_score,
            "ring_overload": -float(self.cfg.p2_ring_overload_penalty_scale)
            * ring_overload,
            "pinky_load": -float(self.cfg.p2_pinky_load_penalty_scale) * pinky_load,
            "pen_velocity": -float(self.cfg.p2_pen_velocity_penalty_scale)
            * speed_penalty,
            "penetration": -float(self.cfg.p2_penetration_penalty_scale)
            * penetration_excess,
            "action_l2": -float(self.cfg.p2_action_l2_penalty_scale)
            * torch.linalg.norm(self.actions, dim=-1),
            "action_smoothness": -float(self.cfg.p2_action_smoothness_penalty_scale)
            * torch.linalg.norm(
                self.actions - self.previous_actions, dim=-1
            ),
            "joint_power": -float(self.cfg.p2_joint_power_penalty_scale) * joint_power,
            "failure": -float(self.cfg.p2_failure_penalty_scale) * metrics["failed"],
        }
        if self.cfg.arrival_hold_reward_mode:
            post_arrival = self.arrival_latched.float()
            pre_arrival = 1.0 - post_arrival
            terms["ring_transfer_progress"] = terms["ring_transfer_progress"] * (
                pre_arrival + 0.15 * post_arrival
            )
            terms["ring_distance_delta"] = terms["ring_distance_delta"] * pre_arrival
            terms["arrival_strict_step"] = 12.0 * post_arrival * eligible.float()
            terms["arrival_hold_growth"] = 14.0 * post_arrival * hold_progress
            terms["arrival_exit"] = -24.0 * post_arrival * (1.0 - eligible.float())
            terms["arrival_target_excess"] = -18.0 * post_arrival * torch.clamp(
                target_excess / max(target_radius, 1.0e-6), 0.0, 3.0
            )
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
        self.previous_stage_target_distance.copy_(metrics["target_distance"])
        if self.cfg.arrival_observation_enabled:
            feature, _, _, _ = self._ring_geometry()
            target_center = self.nearest_p2_target_center
            self.previous_ring_target_error.copy_(feature - target_center)
        self.extras["log"] = {f"reward/{name}": value.mean() for name, value in terms.items()}
        self.extras["log"].update(
            {
                "metric/success_ratio": success.float().mean(),
                "metric/ring_progress": metrics["progress"].mean(),
                "metric/ring_target_distance_mm": metrics["target_distance"].mean() * 1000.0,
                "metric/manifold_semantic_ratio": metrics["manifold_semantic"].mean(),
                "metric/nearest_manifold_mode": metrics["nearest_manifold_mode"].mean(),
                "metric/ring_perpendicular_distance_mm": metrics["perpendicular_distance"].mean() * 1000.0,
                "metric/triad_contact_ratio": metrics["triad"].mean(),
                "metric/middle_contact_ratio": middle_contact.mean(),
                "metric/teacher_c3_final_phase_ratio": (
                    (self.teacher_c3_phase >= 3).float().mean()
                    if self.cfg.teacher_phase_observation_enabled
                    else torch.zeros((), device=self.device)
                ),
                "metric/hold_ms": self.p1_p2_hold_count.float().mean()
                * self.cfg.decimation
                * self.cfg.sim.dt
                * 1000.0,
                "metric/pen_position_drift_mm": metrics["position_drift"].mean() * 1000.0,
                "metric/pen_axis_drift_deg": metrics["axis_drift"].mean() * 180.0 / torch.pi,
                "metric/ring_force_share": metrics["ring_force_share"].mean(),
                "metric/pinky_contact_ratio": (
                    self.contact_force_norm[:, 4] > self.cfg.p2_contact_force_n
                ).float().mean(),
                "metric/max_penetration_mm": self.episode_max_penetration.mean() * 1000.0,
            }
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Completion diagnostics are edge-triggered: reset them before this
        # step's terminated environments are recorded by _reset_idx.
        if hasattr(self, "last_completed_global_bank_ids"):
            self.last_completed_global_bank_ids.fill_(-1)
            self.last_completed_p1_p2_success.zero_()
            self.last_completed_p1_p2_drop.zero_()
        failed, time_out = super()._compute_dones()
        if self.cfg.terminate_on_p2_success:
            failed = failed | self.episode_p1_p2_success
        return failed, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids_tensor = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if hasattr(self, "episode_p1_p2_success"):
            completed = self.episode_length_buf[env_ids_tensor] > 0
            self.last_completed_global_bank_ids[env_ids_tensor] = torch.where(
                completed,
                self.current_global_bank_ids[env_ids_tensor],
                torch.full_like(self.current_global_bank_ids[env_ids_tensor], -1),
            )
            self.last_completed_p1_p2_success[env_ids_tensor] = (
                completed & self.episode_p1_p2_success[env_ids_tensor]
            )
            self.completed_p1_p2_episodes += completed.sum()
            self.completed_p1_p2_successes += (
                completed & self.episode_p1_p2_success[env_ids_tensor]
            ).sum()
            height_drop = self.init_pen_pos[env_ids_tensor, 2] - self.pen_pos[env_ids_tensor, 2]
            completed_drop = completed & (height_drop > self.cfg.drop_threshold)
            self.last_completed_p1_p2_drop[env_ids_tensor] = completed_drop
            self.completed_drop_count += completed_drop.sum()
            invalid = ~torch.isfinite(self.pen_pos[env_ids_tensor]).all(dim=-1) | ~torch.isfinite(
                self.hand_qpos[env_ids_tensor]
            ).all(dim=-1)
            self.completed_invalid_count += (completed & invalid).sum()
            completed_penetration = torch.where(
                completed,
                self.episode_max_penetration[env_ids_tensor],
                torch.zeros_like(self.episode_max_penetration[env_ids_tensor]),
            ).max()
            self.completed_max_penetration = torch.maximum(
                self.completed_max_penetration, completed_penetration
            )

        super()._reset_idx(env_ids_tensor)
        if self.cfg.hold_current_joint_targets_on_reset:
            self.support_anchor_targets[env_ids_tensor] = self.hand_qpos[
                env_ids_tensor
            ]
        else:
            self.support_anchor_targets[env_ids_tensor] = self.source_targets[
                env_ids_tensor
            ]
        self.policy_targets[env_ids_tensor] = self.support_anchor_targets[
            env_ids_tensor
        ]
        self.processed_targets[env_ids_tensor] = self.policy_targets[env_ids_tensor]
        self.hand.set_joint_position_target(
            self.processed_targets[env_ids_tensor],
            joint_ids=self.dex5_joint_ids,
            env_ids=env_ids_tensor,
        )
        self.current_global_bank_ids[env_ids_tensor] = self.global_bank_id_bank[
            self.current_bank_ids[env_ids_tensor]
        ]
        self.p1_p2_hold_count[env_ids_tensor] = 0
        self.episode_p1_p2_success[env_ids_tensor] = False
        self.arrival_latched[env_ids_tensor] = False
        self.terminal_correction_step[env_ids_tensor] = 0
        self.previous_terminal_condition[env_ids_tensor] = 0.0
        self.terminal_condition_delta[env_ids_tensor] = 0.0
        self.terminal_semantic_hold_latched[env_ids_tensor] = False
        self.teacher_brake_latched[env_ids_tensor] = False
        self.teacher_brake_targets[env_ids_tensor] = self.hand_qpos[env_ids_tensor, 8:12]
        self.teacher_c3_phase[env_ids_tensor] = 0
        self.teacher_c3_entry_hold_count[env_ids_tensor] = 0
        if self.cfg.manifold_condition_observation_enabled:
            if int(self.cfg.fixed_desired_manifold_mode) >= 0:
                self.desired_p2_mode[env_ids_tensor] = int(
                    self.cfg.fixed_desired_manifold_mode
                )
            else:
                self.desired_p2_mode[env_ids_tensor] = torch.randint(
                    0,
                    self.p2_manifold_centers.shape[0],
                    (len(env_ids_tensor),),
                    device=self.device,
                )

        rotation_noise = float(self.cfg.reset_pen_rot_noise_rad)
        if rotation_noise > 0.0:
            count = len(env_ids_tensor)
            rotvec = rotation_noise * torch.randn((count, 3), device=self.device)
            angle = torch.linalg.norm(rotvec, dim=-1).clamp_min(1.0e-9)
            axis = rotvec / angle.unsqueeze(-1)
            delta_quat = torch.cat(
                (
                    torch.cos(0.5 * angle).unsqueeze(-1),
                    axis * torch.sin(0.5 * angle).unsqueeze(-1),
                ),
                dim=-1,
            )
            pen_pose = self.pen.data.root_pose_w[env_ids_tensor].clone()
            pen_pose[:, 3:7] = quat_mul(delta_quat, pen_pose[:, 3:7])
            self.pen.write_root_pose_to_sim(pen_pose, env_ids_tensor)
            self.init_pen_axis[env_ids_tensor] = quat_apply(
                pen_pose[:, 3:7], self.pen_x_axis[env_ids_tensor]
            )

        static_low, static_high = self.cfg.pen_static_friction_range
        dynamic_low, dynamic_high = self.cfg.pen_dynamic_friction_range
        if static_low != static_high or dynamic_low != dynamic_high:
            materials = self.pen.root_physx_view.get_material_properties()
            material_env_ids = env_ids_tensor.cpu()
            count = len(env_ids_tensor)
            static = static_low + (static_high - static_low) * torch.rand(
                (count, 1), device=materials.device
            )
            dynamic = dynamic_low + (dynamic_high - dynamic_low) * torch.rand(
                (count, 1), device=materials.device
            )
            dynamic = torch.minimum(dynamic, static)
            materials[material_env_ids, :, 0] = static
            materials[material_env_ids, :, 1] = dynamic
            self.pen.root_physx_view.set_material_properties(materials, material_env_ids)
        self._compute_intermediate_values()
        reset_contacts = (
            self.contact_force_norm[env_ids_tensor, :3] > self.cfg.p2_contact_force_n
        )
        self.primary_contact_grace[env_ids_tensor] = torch.where(
            reset_contacts,
            torch.full_like(
                self.primary_contact_grace[env_ids_tensor], self.cfg.p2_contact_grace_steps
            ),
            torch.zeros_like(self.primary_contact_grace[env_ids_tensor]),
        )
        self.initial_ring_feature[env_ids_tensor] = self._ring_axis_features()[env_ids_tensor]
        _, _, target_distance, _ = self._ring_geometry()
        self.initial_stage_target_distance[env_ids_tensor] = target_distance[env_ids_tensor]
        self.previous_stage_target_distance[env_ids_tensor] = target_distance[env_ids_tensor]
        feature, _, _, _ = self._ring_geometry()
        target_center = self.nearest_p2_target_center
        self.previous_ring_target_error[env_ids_tensor] = (
            feature - target_center
        )[env_ids_tensor]
