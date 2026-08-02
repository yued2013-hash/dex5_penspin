from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from isaaclab.utils.math import (
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
)
from isaaclab.utils import configclass

from .p4_p5_physics_env import (
    Dex5V9P4P5CurrentP4MiddleClearPhysicsEnvCfg,
    Dex5V9P4P5PhysicsEnv,
    Dex5V9P4P5PhysicsEnvCfg,
)


ROOT = Path(__file__).resolve().parents[5]
P5_P1_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "p5_p1_curriculum/p5_p1_five_stage_bundle.npz"
)
P5_P1_MIDDLE_APPROACH_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "p5_p1_curriculum/middle_approach_margin_gated_ppo_success_cache_1200.npz"
)
P5_P1_MIDDLE_CONTACT_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "p5_p1_curriculum/middle_contact_strict_margin_success_cache.npz"
)
P5_P1_STRICT_ACTIVE_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "p5_p1_curriculum/p5_p1_strict_active_bundle.npz"
)
CONTINUOUS_P5_P1_ROOT = (
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/P5到P1连续入口训练包"
)
P5_P1_CONTINUOUS_APPROACH_BUNDLE = str(
    CONTINUOUS_P5_P1_ROOT / "P1接近_连续入口训练包.npz"
)
P5_P1_CONTINUOUS_CONTACT_BUNDLE = str(
    CONTINUOUS_P5_P1_ROOT / "P1接触_连续入口训练包.npz"
)
P5_P1_CONTINUOUS_TRIAD_BUNDLE = str(
    CONTINUOUS_P5_P1_ROOT / "P1三指支撑_连续入口训练包.npz"
)
P5_P1_CONTINUOUS_STRICT_BUNDLE = str(
    CONTINUOUS_P5_P1_ROOT / "P1严格保持_连续入口训练包.npz"
)
P5_P1_MIDDLE_CHECKPOINT = (
    "/home/linkerhand/IsaacLab/logs/skrl/"
    "dex5_recapture_p3_physics_direct/"
    "2026-07-18_19-31-45_ppo_torch/checkpoints/best_agent.pt"
)
P5_P1_CAPTURE_CHECKPOINT = (
    "/home/linkerhand/IsaacLab/logs/skrl/"
    "dex5_recapture_p3_physics_direct/"
    "2026-07-18_19-44-48_ppo_torch/checkpoints/best_agent.pt"
)
P5_P1_APPROACH_CHECKPOINT = (
    "/home/linkerhand/IsaacLab/logs/skrl/"
    "dex5_recapture_p3_physics_direct/"
    "2026-07-17_21-20-25_ppo_torch/checkpoints/best_agent.pt"
)
P5_P1_CONTACT_CHECKPOINT = (
    "/home/linkerhand/IsaacLab/logs/skrl/"
    "dex5_recapture_p3_physics_direct/"
    "2026-07-17_21-29-11_ppo_torch/checkpoints/best_agent.pt"
)
CYCLE_ALIGNED_PHASE9_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/连续闭合_phase9入口缓存_seed2489.npz"
)
THUMB_WRAP_PHASE4_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/食指无名指夹持冷投影严格_seed2575.npz"
)
THUMB_UNDER_PHASE6_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/拇指下托phase6局部入口_seed2585.npz"
)
THUMB_UNDER_PHASE7_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase7真实入口加近成功课程_180条.npz"
)
THUMB_UNDER_PHASE7_TRUE_ENTRY = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase7真实无重置下托接触入口_seed2652.npz"
)
THUMB_UNDER_PHASE7_TEACHER = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase7到phase8保持后锁定承力_seed2674.npz"
)
THUMB_UNDER_PHASE8_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/拇指下托phase8合并入口_127条.npz"
)
THUMB_UNDER_PHASE8_PAIRED_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase8真实入口近成功teacher配对_15条.npz"
)
THUMB_UNDER_PHASE8_NEAR_TEACHER = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase8真实入口到phase9_近成功反馈扩增_seed2690.npz"
)
THUMB_WRAP_PHASE9_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase8到phase9最后下托搜索_seed2605.npz"
)
THUMB_WRAP_PHASE10_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase9拇指绕侧搜索_seed2606.npz"
)
THUMB_WRAP_PHASE11_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase10拇指绕侧搜索_seed2607.npz"
)
THUMB_WRAP_PHASE12_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase11拇指绕侧搜索_seed2608.npz"
)
THUMB_TAKEOVER_PHASE13_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase12拇指绕侧完成扩增_seed2610.npz"
)
THUMB_TAKEOVER_PHASE14_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase13拇指食指接管扩增_seed2612.npz"
)
THUMB_TAKEOVER_PHASE14_STABILIZE_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/连续参考系_phase14近成功到phase15局部扩增_seed2742.npz"
)
THUMB_TAKEOVER_PHASE14_BRAKING_TEACHER = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/食指中指稳夹_phase14主动制动教师入口_seed2816.npz"
)
MIDDLE_UNLOAD_PHASE15_TRUE_ENTRY_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/真实连续_phase15入口_完整夹持上下文_agent640_seed2793.npz"
)
MIDDLE_WITHDRAWAL_PHASE16_CORRECTED_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase16食指拇指接管_中指撤回入口200条.npz"
)
MIDDLE_WITHDRAWAL_PHASE16A_STRICT_HOLD_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase17_P1入口_来自phase16a严格12帧_seed2877.npz"
)
MIDDLE_WITHDRAWAL_PHASE18_ENTRY_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase18拇指食指接管_中指继续撤回入口200条.npz"
)
DUAL_ENTRY_THUMB_INDEX_PINCH_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/拇指继续绕侧_中指卸载_双入口400条.npz"
)
PHASE17_STRICT_TEACHER_NEAR_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase17内部gate严格teacher扩增B_seed2906.npz"
)
PHASE17_STRICT_TEACHER_30_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase16到17_严格成功teacher轨迹30条_残差训练_seed2939.npz"
)
PHASE17_MEAN_TEACHER_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase16到17_全量均值teacher_残差训练_seed3020.npz"
)
PHASE17_STRICT_PAIRED_ENTRY_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase17严格teacher_配对入口cache_seed2904.npz"
)
PHASE17_STRICT_PAIRED_TAIL = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase17严格teacher_配对尾段_seed2904.npz"
)
RING_UNLOAD_PHASE15_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase14拇指食指接管完成搜索_seed2613.npz"
)
RING_UNLOAD_PHASE16_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase15无名指卸载搜索_seed2614.npz"
)
P1_FINAL_PHASE17_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "no_reset_chain/phase16回到P1入口搜索_seed2615.npz"
)


@configclass
class Dex5V9P5P1PhysicsEnvCfg(Dex5V9P4P5PhysicsEnvCfg):
    bundle_path = P5_P1_BUNDLE
    episode_length_s = 0.90
    # P5 uses a settled index pinch; the P4 drive-phase 1.07 index proxy
    # overestimates this fingertip and creates cold-reset penetration spikes.
    collision_proxy_radius_scales = (0.99, 1.04, 1.07, 0.84, 0.80)
    min_curriculum_phase = 0
    max_curriculum_phase = 0
    curriculum_stage = 0
    use_bundle_source_target = True
    p5_hold_steps = 12
    p1_middle_near_distance_m = 0.004
    p1_middle_approach_min_gap_m = 0.0005
    p1_middle_contact_distance_m = 0.001
    p1_pose_rmse_rad = 0.22
    p1_triad_pose_rmse_rad = 0.32
    p1_ring_pose_rmse_rad = 0.14
    closure_final_ring_pose_gate = True
    closure_final_middle_pose_gate = False
    closure_final_middle_pose_rmse_rad = 0.18
    closure_return_middle_instead_of_ring = False
    closure_keep_ring_fixed_during_unload = False
    closure_final_contact_preload_rad = 0.0
    closure_latch_live_target_on_reset = False
    cached_action_replay_steps = 0
    cached_action_replay_scale = 1.0
    p1_contact_force_n = 0.03
    p1_max_force_n = 6.0
    penetration_threshold = 0.00075
    p5_max_penetration_m = 0.00075
    terminate_on_success = True


@configclass
class Dex5V9P5HoldPhysicsEnvCfg(Dex5V9P5P1PhysicsEnvCfg):
    curriculum_stage = 0
    min_curriculum_phase = 0
    max_curriculum_phase = 0


@configclass
class Dex5V9P5P1MiddleApproachPhysicsEnvCfg(Dex5V9P5P1PhysicsEnvCfg):
    curriculum_stage = 1
    min_curriculum_phase = 1
    max_curriculum_phase = 1
    # Keep the collision-free approach deliberately slower than contact.
    action_scale = 0.050
    penetration_threshold = 0.00060
    p5_max_penetration_m = 0.00060


@configclass
class Dex5V9P5P1MiddleContactPhysicsEnvCfg(Dex5V9P5P1PhysicsEnvCfg):
    bundle_path = P5_P1_MIDDLE_APPROACH_CACHE
    curriculum_stage = 2
    min_curriculum_phase = 2
    max_curriculum_phase = 2


@configclass
class Dex5V9P5P1TriadPhysicsEnvCfg(Dex5V9P5P1PhysicsEnvCfg):
    bundle_path = P5_P1_MIDDLE_CONTACT_CACHE
    curriculum_stage = 3
    min_curriculum_phase = 3
    max_curriculum_phase = 3


@configclass
class Dex5V9P5P1StrictPhysicsEnvCfg(Dex5V9P5P1PhysicsEnvCfg):
    bundle_path = P5_P1_STRICT_ACTIVE_BUNDLE
    curriculum_stage = 4
    min_curriculum_phase = 4
    max_curriculum_phase = 4
    p1_pose_rmse_rad = 0.10


@configclass
class Dex5V9P5P1StrictHold200PhysicsEnvCfg(Dex5V9P5P1StrictPhysicsEnvCfg):
    episode_length_s = 1.20
    p5_hold_steps = 24


@configclass
class Dex5V9P5P1StrictHold200MediumDRPhysicsEnvCfg(
    Dex5V9P5P1StrictHold200PhysicsEnvCfg
):
    reset_joint_pos_noise = 0.0008
    reset_joint_vel_noise = 0.025
    reset_pen_pos_noise = 0.00010
    reset_pen_vel_noise = 0.015
    reset_pen_rot_noise_rad = 0.00261799
    pen_static_friction_range = (0.40, 0.80)
    pen_dynamic_friction_range = (0.25, 0.65)
    control_delay_steps_max = 1
    penetration_threshold = 0.00060
    p5_max_penetration_m = 0.00060


@configclass
class Dex5V9P5P1ContinuousApproachPhysicsEnvCfg(
    Dex5V9P5P1MiddleApproachPhysicsEnvCfg
):
    bundle_path = P5_P1_CONTINUOUS_APPROACH_BUNDLE
    collision_proxy_radius_scales = (0.99, 1.07, 1.07, 0.84, 0.80)
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9P5P1ContinuousContactPhysicsEnvCfg(
    Dex5V9P5P1MiddleContactPhysicsEnvCfg
):
    bundle_path = P5_P1_CONTINUOUS_CONTACT_BUNDLE
    collision_proxy_radius_scales = (0.99, 1.07, 1.07, 0.84, 0.80)
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9P5P1ContinuousTriadPhysicsEnvCfg(
    Dex5V9P5P1TriadPhysicsEnvCfg
):
    bundle_path = P5_P1_CONTINUOUS_TRIAD_BUNDLE
    collision_proxy_radius_scales = (0.99, 1.07, 1.07, 0.84, 0.80)
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9P5P1ContinuousStrictPhysicsEnvCfg(
    Dex5V9P5P1StrictPhysicsEnvCfg
):
    bundle_path = P5_P1_CONTINUOUS_STRICT_BUNDLE
    collision_proxy_radius_scales = (0.99, 1.07, 1.07, 0.84, 0.80)
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9P5P1ContinuousStrictHold200PhysicsEnvCfg(
    Dex5V9P5P1ContinuousStrictPhysicsEnvCfg
):
    episode_length_s = 1.20
    p5_hold_steps = 24


@configclass
class Dex5V9P5P1UnifiedPhysicsEnvCfg(Dex5V9P5P1PhysicsEnvCfg):
    curriculum_stage = -1
    min_curriculum_phase = 0
    max_curriculum_phase = 4


@configclass
class Dex5V9P4P5P1NoResetPhysicsEnvCfg(
    Dex5V9P4P5CurrentP4MiddleClearPhysicsEnvCfg
):
    """P4 middle release through projected P1 in one PhysX episode."""

    episode_length_s = 3.0
    penetration_threshold = 0.00075
    p5_max_penetration_m = 0.00050
    p1_middle_near_distance_m = 0.004
    p1_middle_approach_min_gap_m = 0.0005
    p1_middle_contact_distance_m = 0.001
    p1_triad_pose_rmse_rad = 0.32
    p1_pose_rmse_rad = 0.10
    p1_ring_pose_rmse_rad = 0.14
    p1_contact_force_n = 0.03
    p1_max_force_n = 6.0


@configclass
class Dex5V9P4P5P1ContinuousTriadTrainPhysicsEnvCfg(
    Dex5V9P4P5P1NoResetPhysicsEnvCfg
):
    """Train canonical-P1 triad closure after a scripted no-reset prefix."""

    middle_checkpoint = P5_P1_MIDDLE_CHECKPOINT
    capture_checkpoint = P5_P1_CAPTURE_CHECKPOINT
    approach_checkpoint = P5_P1_APPROACH_CHECKPOINT
    contact_checkpoint = P5_P1_CONTACT_CHECKPOINT
    chain_blend_steps = 6
    chain_stage_hold_steps = 12
    triad_hold_steps = 12
    triad_hold_loss_patience_steps = 0
    restore_cached_phase_hold = False
    cached_action_replay_steps = 0
    cached_action_replay_scale = 1.0
    training_start_phase = 4
    canonical_target_blend = 1.0
    target_servo_gain = 0.0
    trainee_policy_scale = 1.0
    scripted_thumb_teacher_path = ""
    scripted_thumb_teacher_action_scale = 1.0
    scripted_thumb_teacher_residual_scale = 0.0
    scripted_thumb_teacher_hold_latch = False
    scripted_thumb_teacher_pair_with_bundle = False
    scripted_thumb_teacher_accept_near = False
    scripted_thumb_teacher_transition_blend_steps = 0
    scripted_thumb_teacher_use_normal_residual_path = False
    ring_only_residual = False
    middle_only_residual = False
    index_middle_only_residual = False
    index_middle_thumb_only_residual = False
    index_ring_thumb_only_residual = False
    thumb_only_residual = False
    thumb_index_only_residual = False
    brace_teacher_checkpoint = ""
    brace_teacher_index = True
    residual_start_phase = 4
    residual_end_phase = 1000000
    final_residual_scale = -1.0
    residual_reward_only = False
    sequential_closure = False
    cycle_aligned_p1 = False
    thumb_wrap_transfer = False
    phase_conditioned_transfer_residual = False
    phase_conditioned_brace_start_level = 3
    phase_conditioned_prefix_residual_scale = 1.0
    # Inclusive last stage for brace, thumb-under, thumb-wrap, and takeover.
    thumb_wrap_stage_boundaries = (1, 4, 8, 10)
    thumb_wrap_thumb_servo_scale = 0.35
    thumb_wrap_ring_approach_servo_scale = 1.50
    thumb_wrap_brace_pen_position_tolerance_m = 0.015
    thumb_wrap_acquisition_pen_axis_tolerance_deg = 22.0
    thumb_wrap_brace_pen_axis_tolerance_deg = 22.0
    thumb_wrap_handoff_max_angular_speed_rad_s = -1.0
    thumb_wrap_handoff_max_linear_speed_m_s = -1.0
    thumb_wrap_require_pinky_clear = False
    thumb_wrap_handoff_speed_reward_scale = 0.0
    thumb_wrap_pinky_contact_penalty_scale = 0.0
    # From phase 12 onward, contact alone is not a completed wrap. Thumb and
    # index must oppose each other across the pen at a similar axial station.
    # Phase 11/12 remains a valid "continue wrapping" entry. Enforce the
    # strict opposite-side pinch only once takeover starts at phase 13.
    thumb_wrap_opposite_gate_start_level = 9
    thumb_wrap_opposite_cos_max = -0.50
    thumb_wrap_max_axial_separation_m = 0.020
    thumb_wrap_opposite_reward_scale = 24.0
    thumb_wrap_unload_pinch_reward_scale = 0.0
    thumb_wrap_unload_thumb_loss_penalty_scale = 0.0
    closure_final_ring_pose_gate = True
    closure_final_middle_pose_gate = False
    closure_final_middle_pose_rmse_rad = 0.18
    closure_return_middle_instead_of_ring = False
    closure_target_blends = (0.50, 0.75, 1.00)
    closure_stage_rmse_thresholds = (0.10, 0.10, 0.10)
    final_hold_steps = 24
    approach_action_rescale = 2.0 / 3.0
    approach_middle_multiplier = -0.5
    contact_action_rescale = 1.0
    contact_middle_multiplier = -0.5
    prefix_middle_servo_gain = 0.0
    closure_pen_pose_gate = False
    closure_pen_target_progress = (
        0.0,
        0.15,
        0.30,
        0.45,
        0.60,
        0.75,
        0.90,
        1.0,
    )
    closure_pen_position_target_progress = ()
    closure_pen_position_tolerance_m = 0.010
    closure_pen_axis_tolerance_deg = 10.0
    closure_pen_position_reward_scale = 8.0
    closure_pen_axis_reward_scale = 14.0
    closure_stage_progress_reward_scale = 0.0
    closure_stage_advance_reward_scale = 0.0
    closure_pen_signed_progress_reward_scale = 0.0
    closure_pen_progress_tracking_reward_scale = 0.0
    closure_cap_signed_progress_at_target = False
    closure_pen_position_error_penalty_scale = 0.0
    closure_pen_axis_error_penalty_scale = 0.0
    closure_pen_angular_speed_penalty_scale = 0.0
    closure_pen_axis_rate_penalty_scale = 0.0
    closure_pen_linear_speed_penalty_scale = 0.0
    closure_joint_pose_reward_scale = 0.0
    closure_joint_safe_reward_scale = 0.0
    closure_action_penalty_scale = 0.0
    closure_hold_progress_reward_scale = 8.0
    closure_success_reward_scale = 40.0
    closure_final_gate_min_reward_scale = 0.0
    closure_final_gate_geomean_reward_scale = 0.0


@configclass
class Dex5V9P4P5P1ContinuousContactTriadTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousTriadTrainPhysicsEnvCfg
):
    """Train contact preservation and ring closure as one no-reset tail."""

    training_start_phase = 3
    canonical_target_blend = 0.25
    p1_triad_pose_rmse_rad = 0.10
    target_servo_gain = 0.02
    trainee_policy_scale = 0.0


@configclass
class Dex5V9P4P5P1ContinuousContactTriadBlend50TrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadTrainPhysicsEnvCfg
):
    canonical_target_blend = 0.50


@configclass
class Dex5V9P4P5P1ContinuousContactTriadBlend50SlowPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadBlend50TrainPhysicsEnvCfg
):
    target_servo_gain = 0.012


@configclass
class Dex5V9P4P5P1ContinuousContactTriadBlend50GentlePhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadBlend50TrainPhysicsEnvCfg
):
    target_servo_gain = 0.008


@configclass
class Dex5V9P4P5P1ContinuousContactTriadBlend50FastPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadBlend50TrainPhysicsEnvCfg
):
    target_servo_gain = 0.03


@configclass
class Dex5V9P4P5P1ContinuousContactTriadBlend50ResidualPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadBlend50TrainPhysicsEnvCfg
):
    trainee_policy_scale = 0.25


@configclass
class Dex5V9P4P5P1ContinuousContactTriadBlend75TrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadTrainPhysicsEnvCfg
):
    canonical_target_blend = 0.75


@configclass
class Dex5V9P4P5P1ContinuousContactTriadBlend75GentlePhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadBlend75TrainPhysicsEnvCfg
):
    target_servo_gain = 0.008


@configclass
class Dex5V9P4P5P1ContinuousContactTriadCanonicalTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadTrainPhysicsEnvCfg
):
    canonical_target_blend = 1.0


@configclass
class Dex5V9P4P5P1ContinuousContactTriadCanonicalGentlePhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadCanonicalTrainPhysicsEnvCfg
):
    target_servo_gain = 0.008


@configclass
class Dex5V9P4P5P1ContinuousClosureCurriculumPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousContactTriadBlend50GentlePhysicsEnvCfg
):
    """50% -> 75% -> canonical P1 closure in one PhysX episode."""

    sequential_closure = True
    episode_length_s = 4.5


@configclass
class Dex5V9P4P5P1ContinuousClosureHold100PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCurriculumPhysicsEnvCfg
):
    final_hold_steps = 12


@configclass
class Dex5V9P4P5P1ContinuousClosureFinePhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCurriculumPhysicsEnvCfg
):
    """Gradually close from the live P5 contact state to canonical P1."""

    closure_target_blends = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
    closure_stage_rmse_thresholds = (0.12, 0.12, 0.14, 0.14, 0.12, 0.10)
    # Each transfer stage must remain valid for about 83 ms. This keeps the
    # thumb-under and wrap motion slow enough for the brace to carry the pen.
    triad_hold_steps = 10
    final_hold_steps = 12
    episode_length_s = 6.0


@configclass
class Dex5V9P4P5P1ContinuousClosureFineSlowPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureFinePhysicsEnvCfg
):
    target_servo_gain = 0.005


@configclass
class Dex5V9P4P5P1ContinuousClosureProbePhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureFinePhysicsEnvCfg
):
    """Probe late closure feasibility without weakening the final P1 gate."""

    closure_target_blends = (
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    )
    closure_stage_rmse_thresholds = (
        0.14,
        0.14,
        0.16,
        0.18,
        0.18,
        0.18,
        0.18,
        0.10,
    )
    triad_hold_steps = 1
    final_hold_steps = 12


@configclass
class Dex5V9P4P5P1ContinuousClosureProbeSlowPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureProbePhysicsEnvCfg
):
    target_servo_gain = 0.005


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureProbeSlowPhysicsEnvCfg
):
    """Keep the live thumb/index pinch while middle/ring return to P1."""

    cycle_aligned_p1 = True
    training_start_phase = 4


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachM075PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    approach_middle_multiplier = -0.75


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachM100PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    approach_middle_multiplier = -1.0


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedContactScale050PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    contact_action_rescale = 0.50


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedContactScale075PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    contact_action_rescale = 0.75


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedContact050LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedContactScale050PhysicsEnvCfg
):
    """Use the safer contact teacher and learn only the phase-9 closure tail."""

    trainee_policy_scale = 0.05
    ring_only_residual = True
    residual_start_phase = 9
    residual_reward_only = True


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedPenPoseTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedContactScale050PhysicsEnvCfg
):
    """Track the canonical P1 pen pose instead of only closing the fingers."""

    closure_pen_pose_gate = True
    trainee_policy_scale = 0.05
    residual_start_phase = 4
    residual_end_phase = 11
    residual_reward_only = True


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedDensePenPoseTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedPenPoseTrainPhysicsEnvCfg
):
    """Use small same-episode hand and pen-axis increments toward canonical P1."""

    closure_target_blends = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.975,
        1.00,
    )
    closure_stage_rmse_thresholds = (
        0.14,
        0.14,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.22,
        0.10,
    )
    closure_pen_target_progress = (
        0.0,
        0.075,
        0.15,
        0.225,
        0.30,
        0.375,
        0.45,
        0.525,
        0.60,
        0.70,
        0.80,
        0.90,
        1.0,
    )
    residual_end_phase = 16
    closure_pen_position_reward_scale = 6.0
    closure_pen_axis_reward_scale = 24.0
    closure_stage_progress_reward_scale = 12.0
    closure_stage_advance_reward_scale = 30.0
    episode_length_s = 8.0


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedDensePenPoseTrainPhysicsEnvCfg
):
    """Learn index/middle differential torque after the safe no-reset prefix."""

    trainee_policy_scale = 0.10
    index_middle_only_residual = True
    residual_start_phase = 7
    residual_end_phase = 16
    closure_pen_axis_reward_scale = 30.0
    closure_stage_advance_reward_scale = 50.0
    closure_pen_signed_progress_reward_scale = 24.0


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueLateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueTrainPhysicsEnvCfg
):
    """Train the dense pen-axis tail from genuine same-episode phase entries."""

    episode_length_s = 2.0
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    reset_pen_rot_noise_rad = 0.0


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueTrainPhysicsEnvCfg
):
    """Add a pen-axis brake/recenter stage before strict canonical P1."""

    closure_target_blends = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.975,
        0.99,
        1.00,
    )
    closure_stage_rmse_thresholds = (
        0.14,
        0.14,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.22,
        0.22,
        0.10,
    )
    closure_pen_target_progress = (
        0.0,
        0.075,
        0.15,
        0.225,
        0.30,
        0.375,
        0.45,
        0.525,
        0.60,
        0.70,
        0.80,
        0.90,
        0.97,
        1.0,
    )
    residual_end_phase = 17
    final_residual_scale = 0.08
    closure_cap_signed_progress_at_target = True
    closure_pen_progress_tracking_reward_scale = 40.0


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeLateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeTrainPhysicsEnvCfg
):
    """Train the brake/recenter tail from stable same-episode entries."""

    episode_length_s = 2.0
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    reset_pen_rot_noise_rad = 0.0


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeThumbTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeTrainPhysicsEnvCfg
):
    """Allow thumb recentering while index/middle brake the pen axis."""

    index_middle_only_residual = False
    index_middle_thumb_only_residual = True


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeThumbLateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeThumbTrainPhysicsEnvCfg
):
    """Train the thumb-assisted brake tail from genuine phase entries."""

    episode_length_s = 2.0
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    reset_pen_rot_noise_rad = 0.0


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbWrapTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedDenseTorqueBrakeThumbTrainPhysicsEnvCfg
):
    """Transfer ring/index support to a slow thumb-under wrap and pinch."""

    thumb_wrap_transfer = True
    cycle_aligned_p1 = True
    # 14 closure stages: brace(2), under(3), wrap(4), takeover(2), unload(3).
    thumb_wrap_stage_boundaries = (1, 4, 8, 10)
    ring_only_residual = False
    index_middle_only_residual = False
    index_middle_thumb_only_residual = False
    residual_start_phase = 4
    residual_end_phase = 17
    trainee_policy_scale = 0.08
    target_servo_gain = 0.015
    final_residual_scale = 0.05
    # About 83 ms per transfer stage at the 120 Hz control rate.
    triad_hold_steps = 10
    final_hold_steps = 12
    # Keep the pen still while index/ring establish the brace and thumb
    # reaches underneath. Pen return begins only during the wrap/takeover.
    closure_pen_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.75,
        0.90,
        1.0,
    )
    closure_stage_rmse_thresholds = (
        0.28,
        0.20,
        0.14,
        0.16,
        0.17,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.22,
        0.22,
        0.10,
    )


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbWrapNoResetTeacherPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbWrapTrainPhysicsEnvCfg
):
    """Run the full P4-prefix and contact transfer in one episode."""

    phase_conditioned_transfer_residual = True
    # Phases 4-5 acquire and stabilize the clamp from exemplars. From
    # phase 6 onward, preserve index/ring while thumb is introduced.
    phase_conditioned_brace_start_level = 2
    brace_teacher_checkpoint = (
        "/home/linkerhand/IsaacLab/logs/skrl/"
        "dex5_recapture_p3_physics_direct/"
        "2026-07-20_04-22-19_ppo_torch/checkpoints/agent_2560.pt"
    )
    trainee_policy_scale = 0.35
    chain_blend_steps = 6
    episode_length_s = 8.0
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    reset_pen_rot_noise_rad = 0.0


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbWrapLateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbWrapTrainPhysicsEnvCfg
):
    """Train the corrected contact-switch tail from same-episode entries."""

    bundle_path = THUMB_WRAP_PHASE4_CACHE
    trainee_policy_scale = 0.35
    episode_length_s = 2.5
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    reset_pen_rot_noise_rad = 0.0


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbUnderLateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbWrapLateTrainPhysicsEnvCfg
):
    """Train thumb-under support from genuine phase-6 policy entries."""

    bundle_path = THUMB_UNDER_PHASE6_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbWrapLateTrainPhysicsEnvCfg
):
    """Continue slow thumb-under closure from genuine phase-7 entries."""

    bundle_path = THUMB_UNDER_PHASE7_CACHE
    thumb_only_residual = True
    brace_teacher_checkpoint = (
        "/home/linkerhand/IsaacLab/logs/skrl/"
        "dex5_recapture_p3_physics_direct/"
        "2026-07-20_04-22-19_ppo_torch/checkpoints/agent_2560.pt"
    )


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7To8TrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg
):
    """Train only sustained phase-7 support and the phase-8 hold."""

    bundle_path = THUMB_UNDER_PHASE7_TRUE_ENTRY
    scripted_thumb_teacher_path = THUMB_UNDER_PHASE7_TEACHER
    scripted_thumb_teacher_action_scale = 0.35
    scripted_thumb_teacher_residual_scale = 0.10
    scripted_thumb_teacher_hold_latch = True
    trainee_policy_scale = 1.0
    final_residual_scale = 1.0
    closure_target_blends = (0.50, 0.55, 0.60, 0.65, 0.70)
    closure_stage_rmse_thresholds = (0.28, 0.20, 0.14, 0.16, 0.17)
    closure_pen_target_progress = (0.0, 0.0, 0.0, 0.0, 0.0)
    residual_end_phase = 8
    triad_hold_steps = 10
    final_hold_steps = 10
    episode_length_s = 1.5


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbUnderPhase8LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg
):
    """Finish thumb-under support before the wrap starts."""

    bundle_path = THUMB_UNDER_PHASE8_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbUnderPhase8To9TrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg
):
    """Train the final thumb-under hold from paired same-episode entries."""

    bundle_path = THUMB_UNDER_PHASE8_PAIRED_CACHE
    scripted_thumb_teacher_path = THUMB_UNDER_PHASE8_NEAR_TEACHER
    scripted_thumb_teacher_action_scale = 0.35
    scripted_thumb_teacher_residual_scale = 0.10
    scripted_thumb_teacher_hold_latch = True
    scripted_thumb_teacher_pair_with_bundle = True
    scripted_thumb_teacher_accept_near = False
    scripted_thumb_teacher_transition_blend_steps = 0
    thumb_only_residual = False
    index_middle_thumb_only_residual = True
    index_ring_thumb_only_residual = False
    # Phase 8 ends at the completed under-support posture. Phase 9 only
    # begins the wrap, so index/middle remain the load-bearing clamp.
    closure_target_blends = (0.50, 0.55, 0.60, 0.65, 0.70, 0.72)
    closure_stage_rmse_thresholds = (
        0.28,
        0.20,
        0.14,
        0.16,
        0.17,
        0.17,
    )
    closure_pen_target_progress = (0.0, 0.0, 0.0, 0.0, 0.0, 0.05)
    residual_end_phase = 9
    thumb_wrap_handoff_max_angular_speed_rad_s = 2.0
    thumb_wrap_handoff_max_linear_speed_m_s = 0.025
    thumb_wrap_require_pinky_clear = True
    thumb_wrap_handoff_speed_reward_scale = 14.0
    thumb_wrap_pinky_contact_penalty_scale = 18.0
    # Phase 8 is a dynamic handoff state, but its 50 ms hold only counts
    # while the pen is braked and the inactive pinky remains clear.
    triad_hold_steps = 6
    triad_hold_loss_patience_steps = 1
    final_hold_steps = 10
    episode_length_s = 1.5


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbWrapPhase9LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg
):
    """Wrap the supported thumb toward the opposite side."""

    bundle_path = THUMB_WRAP_PHASE9_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbWrapPhase10LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg
):
    """Continue the supported thumb wrap from phase 10."""

    bundle_path = THUMB_WRAP_PHASE10_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbWrapPhase11LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg
):
    """Continue the supported thumb wrap from phase 11."""

    bundle_path = THUMB_WRAP_PHASE11_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbWrapPhase12LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg
):
    """Finish the supported thumb wrap before takeover."""

    bundle_path = THUMB_WRAP_PHASE12_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase13LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbUnderPhase7LateTrainPhysicsEnvCfg
):
    """Form the thumb/index pinch while middle retains the prior clamp."""

    bundle_path = THUMB_TAKEOVER_PHASE13_CACHE
    thumb_only_residual = False
    thumb_index_only_residual = True
    brace_teacher_index = False


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase14LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase13LateTrainPhysicsEnvCfg
):
    """Complete thumb/index takeover while middle still clamps the pen."""

    bundle_path = THUMB_TAKEOVER_PHASE14_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase14StabilizePhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase14LateTrainPhysicsEnvCfg
):
    """Train a real phase-14 hold from continuous-reference near states."""

    bundle_path = THUMB_TAKEOVER_PHASE14_STABILIZE_CACHE
    restore_cached_phase_hold = False
    episode_length_s = 2.0
    closure_pen_position_reward_scale = 20.0
    closure_pen_axis_reward_scale = 20.0
    closure_pen_signed_progress_reward_scale = 0.0
    closure_pen_progress_tracking_reward_scale = 120.0
    closure_stage_progress_reward_scale = 0.0
    closure_stage_advance_reward_scale = 180.0
    closure_pen_position_error_penalty_scale = 1000.0
    closure_pen_axis_error_penalty_scale = 10.0
    closure_pen_angular_speed_penalty_scale = 6.0
    closure_pen_linear_speed_penalty_scale = 300.0
    closure_joint_pose_reward_scale = 1000.0
    closure_joint_safe_reward_scale = 500.0
    closure_action_penalty_scale = 2.0
    closure_hold_progress_reward_scale = 300.0
    closure_success_reward_scale = 1000.0


@configclass
class Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase14EntryCollectPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase14StabilizePhysicsEnvCfg
):
    """Collect diverse no-reset phase-15 entries after a short takeover hold."""

    # This is an entry-generation curriculum, not the formal phase-14 gate.
    # Index/middle/thumb contact and all safety/pose gates remain active.
    triad_hold_steps = 1
    final_hold_steps = 1
    # Match the downstream phase-15 interface: continue translating the pen
    # toward P1 without commanding another axis rotation at the handoff.
    closure_pen_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.58,
        0.90,
        1.0,
    )
    closure_pen_position_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.75,
        0.90,
        1.0,
    )


@configclass
class Dex5V9P4P5P1ContinuousClosureMiddleUnloadPhase15StabilizePhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbTakeoverPhase14StabilizePhysicsEnvCfg
):
    """Confirm thumb/index takeover while index/middle retain the pen."""

    thumb_index_only_residual = False
    index_middle_thumb_only_residual = True
    trainee_policy_scale = 0.08
    # Phase 15 remains a takeover-confirmation stage. Middle withdrawal
    # starts in phase 16 only after the thumb/index pinch has held.
    thumb_wrap_stage_boundaries = (1, 4, 8, 11)
    closure_final_ring_pose_gate = False
    closure_final_middle_pose_gate = True
    closure_final_middle_pose_rmse_rad = 0.26
    closure_return_middle_instead_of_ring = True
    closure_keep_ring_fixed_during_unload = True
    closure_latch_live_target_on_reset = True
    cached_action_replay_steps = 4
    cached_action_replay_scale = -0.5
    thumb_wrap_unload_pinch_reward_scale = 60.0
    thumb_wrap_unload_thumb_loss_penalty_scale = 40.0
    residual_end_phase = 15
    closure_target_blends = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.95,
    )
    closure_stage_rmse_thresholds = (
        0.28,
        0.20,
        0.14,
        0.16,
        0.17,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.22,
    )
    closure_pen_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.58,
    )
    closure_pen_position_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.75,
    )
    final_hold_steps = 12
    episode_length_s = 1.5


@configclass
class Dex5V9P4P5P1ContinuousClosureMiddleUnloadPhase15EntryTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureMiddleUnloadPhase15StabilizePhysicsEnvCfg
):
    """Recover and hold the thumb/index pinch from true phase-15 entries."""

    bundle_path = MIDDLE_UNLOAD_PHASE15_TRUE_ENTRY_CACHE
    restore_cached_phase_hold = False
    closure_pen_position_tolerance_m = 0.012
    closure_pen_axis_tolerance_deg = 16.0
    closure_stage_rmse_thresholds = (
        0.28,
        0.20,
        0.14,
        0.16,
        0.17,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.26,
    )
    closure_hold_progress_reward_scale = 400.0
    closure_success_reward_scale = 1200.0


@configclass
class Dex5V9P4P5P1ContinuousClosureMiddleUnloadPhase15EntryStrictPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureMiddleUnloadPhase15EntryTrainPhysicsEnvCfg
):
    """Strict 10 mm / 10 degree evaluation from true phase-15 entries."""

    closure_pen_position_tolerance_m = 0.010
    closure_pen_axis_tolerance_deg = 10.0
    closure_stage_rmse_thresholds = (
        0.28,
        0.20,
        0.14,
        0.16,
        0.17,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.22,
    )


@configclass
class Dex5V9P4P5P1ContinuousClosureMiddleUnloadPhase15EntryMidPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureMiddleUnloadPhase15EntryTrainPhysicsEnvCfg
):
    """Intermediate pose curriculum before the strict phase-15 gate."""

    closure_pen_position_tolerance_m = 0.011
    closure_pen_axis_tolerance_deg = 12.0
    closure_stage_rmse_thresholds = (
        0.28,
        0.20,
        0.14,
        0.16,
        0.17,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.24,
    )


@configclass
class Dex5V9P4P5P1ContinuousClosureTakeoverPhase14To15NoResetPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureMiddleUnloadPhase15EntryStrictPhysicsEnvCfg
):
    """Train phase 14 through takeover confirmation without a state reset."""

    bundle_path = THUMB_TAKEOVER_PHASE14_STABILIZE_CACHE
    closure_latch_live_target_on_reset = False
    cached_action_replay_steps = 0
    cached_action_replay_scale = 1.0
    # Phase 14 is an entry gate; phase 15 carries the formal 100 ms hold.
    triad_hold_steps = 1
    final_hold_steps = 12
    final_residual_scale = 0.02
    thumb_wrap_handoff_max_angular_speed_rad_s = 2.0
    thumb_wrap_handoff_max_linear_speed_m_s = 0.025
    thumb_wrap_handoff_speed_reward_scale = 80.0
    closure_pen_axis_tolerance_deg = 10.0
    closure_pen_angular_speed_penalty_scale = 30.0
    closure_pen_axis_rate_penalty_scale = 80.0
    episode_length_s = 2.0


@configclass
class Dex5V9P4P5P1ContinuousClosureTakeoverPhase14To15Margin75PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureTakeoverPhase14To15NoResetPhysicsEnvCfg
):
    """Use a 7.5 degree phase handoff curriculum before strict evaluation."""

    closure_pen_axis_tolerance_deg = 7.5


@configclass
class Dex5V9P4P5P1ContinuousClosureTakeoverPhase14To15TeacherPrefillPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureTakeoverPhase14To15NoResetPhysicsEnvCfg
):
    """Prefill the phase-14 axis brake while index/middle retain the pen."""

    scripted_thumb_teacher_path = THUMB_TAKEOVER_PHASE14_BRAKING_TEACHER
    scripted_thumb_teacher_action_scale = 0.12
    scripted_thumb_teacher_residual_scale = 0.12
    scripted_thumb_teacher_pair_with_bundle = True
    scripted_thumb_teacher_accept_near = False
    scripted_thumb_teacher_hold_latch = True
    closure_final_contact_preload_rad = 0.0


@configclass
class Dex5V9P4P5P1ContinuousClosureTakeoverPhase15BCAuthorityPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureTakeoverPhase14To15TeacherPrefillPhysicsEnvCfg
):
    """Give the learned phase-15 clamp the authority used by its BC teacher."""

    final_residual_scale = 0.12


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase16MiddleWithdrawalPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureTakeoverPhase15BCAuthorityPhysicsEnvCfg
):
    """Start middle withdrawal after a verified thumb/index takeover."""

    bundle_path = MIDDLE_WITHDRAWAL_PHASE16_CORRECTED_CACHE
    scripted_thumb_teacher_path = ""
    closure_latch_live_target_on_reset = False
    cached_action_replay_steps = 0
    residual_start_phase = 16
    residual_end_phase = 16
    index_middle_thumb_only_residual = False
    middle_only_residual = True
    trainee_policy_scale = 0.05
    final_residual_scale = 0.05
    closure_pen_position_reward_scale = 30.0
    closure_pen_axis_reward_scale = 40.0
    closure_pen_position_error_penalty_scale = 1800.0
    closure_pen_axis_error_penalty_scale = 80.0
    closure_pen_angular_speed_penalty_scale = 60.0
    closure_pen_axis_rate_penalty_scale = 140.0
    closure_pen_linear_speed_penalty_scale = 500.0
    closure_target_blends = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.95,
        0.965,
    )
    closure_stage_rmse_thresholds = (
        0.28,
        0.20,
        0.14,
        0.16,
        0.17,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.22,
        0.24,
    )
    closure_pen_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.58,
        0.58,
    )
    closure_pen_position_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.75,
        0.75,
    )
    final_hold_steps = 12
    episode_length_s = 1.5


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase16MiddleWithdrawalApproachPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase16MiddleWithdrawalPhysicsEnvCfg
):
    """Withdraw middle about five percent before the full phase-16 target."""

    closure_target_blends = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.95,
        0.9575,
    )
    thumb_wrap_handoff_max_angular_speed_rad_s = 3.0
    thumb_wrap_handoff_max_linear_speed_m_s = 0.035


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase16ATo17NoResetPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase16MiddleWithdrawalApproachPhysicsEnvCfg
):
    """Confirm phase16a, then withdraw middle about ten percent."""

    bundle_path = MIDDLE_WITHDRAWAL_PHASE16A_STRICT_HOLD_CACHE
    residual_start_phase = 16
    residual_end_phase = 17
    # The successful phase16->17 teacher trajectories were collected with
    # 0.18 residual authority in both phases. Keep deployment identical so BC
    # and PPO do not see their learned corrections silently scaled to 28%.
    trainee_policy_scale = 0.18
    final_residual_scale = 0.18
    # This reset cache is exported only after a strict 12-frame (100 ms)
    # phase-16 hold, so one recovery frame is enough before continuing. The
    # eventual upstream no-reset chain still performs the full hold itself.
    triad_hold_steps = 1
    # The inherited middle-only mask would hide the index/thumb corrections
    # needed to preserve the pinch while middle unloads.
    middle_only_residual = False
    index_middle_thumb_only_residual = True
    closure_target_blends = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.95,
        0.9575,
        0.960,
    )
    closure_stage_rmse_thresholds = (
        0.28,
        0.20,
        0.14,
        0.16,
        0.17,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.22,
        0.24,
        0.24,
    )
    closure_pen_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.58,
        0.58,
        0.58,
    )
    closure_pen_position_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.75,
        0.75,
        0.75,
    )


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase18MiddleWithdrawalPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase16ATo17NoResetPhysicsEnvCfg
):
    """Withdraw middle to 15 percent while thumb/index preserve the pinch."""

    bundle_path = MIDDLE_WITHDRAWAL_PHASE18_ENTRY_CACHE
    residual_start_phase = 18
    residual_end_phase = 18
    triad_hold_steps = 1
    # This is an unloading approach state, not the final P1 brake. Allow the
    # pinch to remain dynamic here; later closure stages tighten it again.
    thumb_wrap_handoff_max_angular_speed_rad_s = 4.5
    thumb_wrap_handoff_max_linear_speed_m_s = 0.04
    chain_blend_steps = 6
    closure_target_blends = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.925,
        0.95,
        0.95,
        0.9575,
        0.960,
        0.9625,
    )
    closure_stage_rmse_thresholds = (
        0.28,
        0.20,
        0.14,
        0.16,
        0.17,
        0.17,
        0.18,
        0.18,
        0.18,
        0.20,
        0.20,
        0.22,
        0.24,
        0.24,
        0.25,
    )
    closure_pen_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.58,
        0.58,
        0.58,
        0.58,
    )
    closure_pen_position_target_progress = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.10,
        0.18,
        0.28,
        0.42,
        0.58,
        0.75,
        0.75,
        0.75,
        0.75,
    )


@configclass
class Dex5V9P4P5P1DualEntryThumbIndexPinchPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase18MiddleWithdrawalPhysicsEnvCfg
):
    """Train one strict thumb/index pinch from either valid entry route."""

    bundle_path = DUAL_ENTRY_THUMB_INDEX_PINCH_CACHE
    residual_start_phase = 11
    residual_end_phase = 18
    middle_only_residual = False
    index_middle_thumb_only_residual = True
    trainee_policy_scale = 0.18
    final_residual_scale = 0.18
    triad_hold_steps = 1
    episode_length_s = 2.0


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17TeacherPrefillPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase16ATo17NoResetPhysicsEnvCfg
):
    """PPO prefill around strict 11-frame phase17 teacher trajectories."""

    bundle_path = MIDDLE_WITHDRAWAL_PHASE16A_STRICT_HOLD_CACHE
    scripted_thumb_teacher_path = PHASE17_STRICT_TEACHER_NEAR_CACHE
    scripted_thumb_teacher_pair_with_bundle = False
    scripted_thumb_teacher_action_scale = 0.18
    scripted_thumb_teacher_residual_scale = 0.03
    scripted_thumb_teacher_accept_near = True
    scripted_thumb_teacher_hold_latch = False
    scripted_thumb_teacher_transition_blend_steps = 0
    scripted_thumb_teacher_use_normal_residual_path = True
    # At phase16/17 the original search is past the brace-teacher interval;
    # preserve all index/middle dimensions from the strict candidate.
    brace_teacher_checkpoint = ""
    trainee_policy_scale = 0.18
    # Reproduce the strict teacher on the deterministic reset used by the
    # search before adding reset noise as a curriculum dimension.
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    reset_pen_rot_noise_rad = 0.0
    final_residual_scale = 0.18


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17StrictTeacherResidualPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase17TeacherPrefillPhysicsEnvCfg
):
    """Train a small closed-loop residual around strict 100 ms teachers."""

    scripted_thumb_teacher_path = PHASE17_STRICT_TEACHER_30_CACHE
    scripted_thumb_teacher_accept_near = False
    scripted_thumb_teacher_residual_scale = 0.03
    closure_hold_progress_reward_scale = 2000.0
    closure_success_reward_scale = 5000.0
    closure_final_gate_min_reward_scale = 800.0
    closure_final_gate_geomean_reward_scale = 400.0


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17TeacherResidualMargin12PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase17StrictTeacherResidualPhysicsEnvCfg
):
    """First hold curriculum before restoring the strict phase17 gates."""

    closure_pen_position_tolerance_m = 0.012
    closure_pen_axis_tolerance_deg = 12.0
    thumb_wrap_handoff_max_angular_speed_rad_s = 3.0
    thumb_wrap_handoff_max_linear_speed_m_s = 0.035


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17MeanTeacherMargin12PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase17TeacherResidualMargin12PhysicsEnvCfg
):
    """Train around one Markov teacher formed from all strict candidates."""

    scripted_thumb_teacher_path = PHASE17_MEAN_TEACHER_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17StandaloneMargin12PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase16ATo17NoResetPhysicsEnvCfg
):
    """Evaluate or train the distilled policy without an online teacher."""

    closure_pen_position_tolerance_m = 0.012
    closure_pen_axis_tolerance_deg = 12.0
    thumb_wrap_handoff_max_angular_speed_rad_s = 3.0
    thumb_wrap_handoff_max_linear_speed_m_s = 0.035
    closure_hold_progress_reward_scale = 2000.0
    closure_success_reward_scale = 5000.0
    closure_final_gate_min_reward_scale = 800.0
    closure_final_gate_geomean_reward_scale = 400.0


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17StandaloneMargin11PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase17StandaloneMargin12PhysicsEnvCfg
):
    """Tighten a distilled standalone policy before the strict final gate."""

    closure_pen_position_tolerance_m = 0.011
    closure_pen_axis_tolerance_deg = 11.0
    thumb_wrap_handoff_max_angular_speed_rad_s = 2.5
    thumb_wrap_handoff_max_linear_speed_m_s = 0.030


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17TeacherResidualMargin11PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase17StrictTeacherResidualPhysicsEnvCfg
):
    """Bridge the successful 12 mm curriculum to the strict final gates."""

    closure_pen_position_tolerance_m = 0.011
    closure_pen_axis_tolerance_deg = 11.0
    thumb_wrap_handoff_max_angular_speed_rad_s = 2.5
    thumb_wrap_handoff_max_linear_speed_m_s = 0.030


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17TeacherResidualMargin115PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase17StrictTeacherResidualPhysicsEnvCfg
):
    """Half-step curriculum between the successful 12 and 11 mm gates."""

    closure_pen_position_tolerance_m = 0.0115
    closure_pen_axis_tolerance_deg = 11.5
    thumb_wrap_handoff_max_angular_speed_rad_s = 2.75
    thumb_wrap_handoff_max_linear_speed_m_s = 0.0325


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17TeacherResidualMargin14PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase17StrictTeacherResidualPhysicsEnvCfg
):
    """Bootstrap continuous hold before the 12 mm curriculum."""

    closure_pen_position_tolerance_m = 0.014
    closure_pen_axis_tolerance_deg = 15.0
    thumb_wrap_handoff_max_angular_speed_rad_s = 4.0
    thumb_wrap_handoff_max_linear_speed_m_s = 0.050


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17TeacherResidualMargin13PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase17StrictTeacherResidualPhysicsEnvCfg
):
    """Bridge the successful 14 mm hold curriculum to 12 mm."""

    closure_pen_position_tolerance_m = 0.013
    closure_pen_axis_tolerance_deg = 13.5
    thumb_wrap_handoff_max_angular_speed_rad_s = 3.5
    thumb_wrap_handoff_max_linear_speed_m_s = 0.0425


@configclass
class Dex5V9P4P5P1ContinuousClosurePhase17PairedEntryPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosurePhase17TeacherPrefillPhysicsEnvCfg
):
    """Diagnostic reset at a strict gate; PhysX warm-start is not restored."""

    bundle_path = PHASE17_STRICT_PAIRED_ENTRY_CACHE
    scripted_thumb_teacher_path = PHASE17_STRICT_PAIRED_TAIL
    scripted_thumb_teacher_pair_with_bundle = True
    scripted_thumb_teacher_accept_near = False


@configclass
class Dex5V9P4P5P1ContinuousClosureRingUnloadPhase15LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbWrapLateTrainPhysicsEnvCfg
):
    """Legacy ring-unload curriculum retained for old reports."""

    bundle_path = RING_UNLOAD_PHASE15_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureRingUnloadPhase16LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbWrapLateTrainPhysicsEnvCfg
):
    """Return middle/ring to the strict P1 entry."""

    bundle_path = RING_UNLOAD_PHASE16_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureP1FinalPhase17LateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureThumbWrapLateTrainPhysicsEnvCfg
):
    """Hold the final thumb/index P1 pinch for at least 100 ms."""

    bundle_path = P1_FINAL_PHASE17_CACHE


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    """Learn a bounded residual around the no-reset cycle-aligned teacher."""

    trainee_policy_scale = 0.10
    ring_only_residual = True
    residual_start_phase = 9
    residual_reward_only = True


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedTrainScale005PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedTrainPhysicsEnvCfg
):
    trainee_policy_scale = 0.05


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedTrainScale015PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedTrainPhysicsEnvCfg
):
    trainee_policy_scale = 0.15


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedEarlyTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    """Learn ring corrections for the 50%-through-85% closure stages."""

    trainee_policy_scale = 0.05
    ring_only_residual = True
    residual_start_phase = 4
    residual_end_phase = 8
    residual_reward_only = True


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedContactTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    """Learn a bounded middle residual for same-episode P1 recapture."""

    trainee_policy_scale = 0.10
    middle_only_residual = True
    residual_start_phase = 2
    residual_end_phase = 3
    residual_reward_only = True


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    """Learn only the same-episode P5-to-P1 middle approach correction."""

    trainee_policy_scale = 0.05
    middle_only_residual = True
    residual_start_phase = 2
    residual_end_phase = 2
    residual_reward_only = True


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachScale002PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachTrainPhysicsEnvCfg
):
    trainee_policy_scale = 0.02


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachScale0035PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedApproachTrainPhysicsEnvCfg
):
    trainee_policy_scale = 0.035


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedServo002PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    prefix_middle_servo_gain = 0.002


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedServo005PhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedProbePhysicsEnvCfg
):
    prefix_middle_servo_gain = 0.005


@configclass
class Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnvCfg(
    Dex5V9P4P5P1ContinuousClosureCycleAlignedTrainPhysicsEnvCfg
):
    """Train the 90%-to-P1 tail from same-episode phase-entry states."""

    bundle_path = CYCLE_ALIGNED_PHASE9_CACHE
    episode_length_s = 1.50
    trainee_policy_scale = 0.10
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    reset_pen_rot_noise_rad = 0.0


class _FrozenP5Policy(nn.Module):
    def __init__(self, checkpoint_path: str, device: str) -> None:
        super().__init__()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state = checkpoint["policy"]
        self.net = nn.Sequential(
            nn.Linear(156, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 20),
        )
        for layer, key in (
            (0, "net_container.0"),
            (2, "net_container.2"),
            (4, "net_container.4"),
            (6, "policy_layer"),
        ):
            self.net[layer].weight.data.copy_(state[f"{key}.weight"])
            self.net[layer].bias.data.copy_(state[f"{key}.bias"])
        preprocessor = checkpoint["observation_preprocessor"]
        self.register_buffer(
            "mean", preprocessor["running_mean"].float()
        )
        self.register_buffer(
            "variance", preprocessor["running_variance"].float()
        )
        self.to(device)
        self.eval()
        self.requires_grad_(False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(
            (observation - self.mean)
            / (torch.sqrt(self.variance) + 1.0e-8),
            -5.0,
            5.0,
        )
        return self.net(normalized)


class Dex5V9P4P5P1NoResetPhysicsEnv(Dex5V9P4P5PhysicsEnv):
    cfg: Dex5V9P4P5P1NoResetPhysicsEnvCfg

    def _load_bundle(self) -> None:
        super()._load_bundle()
        self.chain_phase_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _stage_action_mask(self) -> torch.Tensor:
        phase = self.chain_phase_ids
        mask = torch.zeros((self.num_envs, 20), device=self.device)
        p4_middle = phase == 0
        p5_capture = phase == 1
        p1_approach = phase == 2
        p1_contact = phase == 3
        p1_triad_or_strict = phase >= 4

        p4_support = 0.35 * (p4_middle | p5_capture).float()
        p1_support = (
            0.30 * (p1_approach | p1_contact).float()
            + p1_triad_or_strict.float()
        )
        support = p4_support + p1_support
        mask[:, :4] = support.unsqueeze(-1)
        mask[:, 16:20] = support.unsqueeze(-1)
        mask[:, 4:8] = (
            p4_middle
            | p1_approach
            | p1_contact
            | p1_triad_or_strict
        ).float().unsqueeze(-1)
        # P1 closure requires ring to retreat from below the pen to its
        # canonical rear position. Keep it passive until the triad is formed.
        mask[:, 8:12] = p1_triad_or_strict.float().unsqueeze(-1)
        return mask

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "chain_phase_ids"):
            return
        if env_ids is None:
            env_ids_tensor = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        rows = torch.arange(self.num_envs, device=self.device)
        reset_mask = (
            rows.unsqueeze(-1) == env_ids_tensor.reshape(1, -1)
        ).any(dim=-1)
        self.chain_phase_ids = torch.where(
            reset_mask,
            torch.zeros_like(self.chain_phase_ids),
            self.chain_phase_ids,
        )


class Dex5V9P4P5P1ContinuousTriadTrainPhysicsEnv(
    Dex5V9P4P5P1NoResetPhysicsEnv
):
    cfg: Dex5V9P4P5P1ContinuousTriadTrainPhysicsEnvCfg

    def _load_bundle(self) -> None:
        super()._load_bundle()
        self.prefix_policies = tuple(
            _FrozenP5Policy(path, self.device)
            for path in (
                self.cfg.middle_checkpoint,
                self.cfg.capture_checkpoint,
                self.cfg.approach_checkpoint,
                self.cfg.contact_checkpoint,
            )
        )
        self.brace_teacher_policy = (
            _FrozenP5Policy(
                self.cfg.brace_teacher_checkpoint, self.device
            )
            if self.cfg.brace_teacher_checkpoint
            else None
        )
        manifest_path = (
            ROOT
            / "dex5_penspin/reports/m3_2_30_index_near_surface/"
            "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
        )
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        p1 = next(
            row
            for row in manifest["phases"]
            if int(row["phase"]) == 1
        )
        self.p1_hand = torch.as_tensor(
            p1["hand_qpos"], dtype=torch.float32, device=self.device
        )
        self.p1_pen_pos = torch.as_tensor(
            p1["pen_pos"], dtype=torch.float32, device=self.device
        )
        self.p1_pen_quat = torch.as_tensor(
            p1["pen_quat"], dtype=torch.float32, device=self.device
        )
        self.p1_pen_axis = quat_apply(
            self.p1_pen_quat,
            torch.tensor(
                [1.0, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        p4 = next(
            row
            for row in manifest["phases"]
            if int(row["phase"]) == 4
        )
        p5 = next(
            row
            for row in manifest["phases"]
            if int(row["phase"]) == 5
        )
        p5a = next(
            row
            for row in p5["transition_keyframes"]
            if row["name"] == "P5a_thumb_index_takeover_middle_ring_below"
        )
        self.p4_thumb_under_hand = torch.as_tensor(
            p4["hand_qpos"], dtype=torch.float32, device=self.device
        )
        self.p5a_transfer_hand = torch.as_tensor(
            p5a["hand_qpos"], dtype=torch.float32, device=self.device
        )
        capture_path = (
            ROOT
            / "dex5_penspin/reports/m3_2_30_index_near_surface/"
            "p4_p5_curriculum/"
            "capture_collision_free_cache_seed1503.npz"
        )
        with np.load(capture_path) as capture:
            p5_hand = np.asarray(
                capture["target_qpos"], dtype=np.float32
            )[:, :20].mean(axis=0)
        self.p5_hand = torch.as_tensor(
            p5_hand, dtype=torch.float32, device=self.device
        )
        self.chain_phase_hold = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.chain_phase_hold_loss = torch.zeros_like(
            self.chain_phase_hold
        )
        self.chain_blend_age = torch.full(
            (self.num_envs,),
            int(self.cfg.chain_blend_steps),
            dtype=torch.long,
            device=self.device,
        )
        self.chain_previous_action = torch.zeros(
            (self.num_envs, 20), device=self.device
        )
        self.closure_origin_hand = torch.zeros(
            (self.num_envs, 20), device=self.device
        )
        self.closure_brace_hand = torch.zeros(
            (self.num_envs, 20), device=self.device
        )
        self.closure_origin_pen_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.closure_origin_pen_quat = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.closure_origin_pen_quat[:, 0] = 1.0
        closure_stage_count = len(self.cfg.closure_target_blends)
        self.chain_phase_entries = torch.zeros(
            int(self.cfg.training_start_phase) + closure_stage_count,
            dtype=torch.long,
            device=self.device,
        )
        self.chain_drop_by_phase = torch.zeros_like(
            self.chain_phase_entries
        )
        self.chain_drift_by_phase = torch.zeros_like(
            self.chain_phase_entries
        )
        self.chain_penetration_by_phase = torch.zeros_like(
            self.chain_phase_entries
        )
        self.closure_stage_entries = torch.zeros(
            closure_stage_count, dtype=torch.long, device=self.device
        )
        self.closure_stage_step_counts = torch.zeros(
            closure_stage_count, dtype=torch.long, device=self.device
        )
        self.closure_stage_triad_step_counts = torch.zeros(
            closure_stage_count, dtype=torch.long, device=self.device
        )
        self.closure_stage_finger_contact_step_counts = torch.zeros(
            (closure_stage_count, 5),
            dtype=torch.long,
            device=self.device,
        )
        self.closure_stage_transfer_component_step_counts = torch.zeros(
            (closure_stage_count, 4),
            dtype=torch.long,
            device=self.device,
        )
        self.closure_stage_safe_step_counts = torch.zeros(
            closure_stage_count, dtype=torch.long, device=self.device
        )
        self.closure_stage_safe_component_step_counts = torch.zeros(
            (closure_stage_count, 5),
            dtype=torch.long,
            device=self.device,
        )
        self.closure_stage_pose_step_counts = torch.zeros(
            closure_stage_count, dtype=torch.long, device=self.device
        )
        self.closure_stage_contact_pose_step_counts = torch.zeros(
            closure_stage_count, dtype=torch.long, device=self.device
        )
        self.closure_stage_eligible_step_counts = torch.zeros(
            closure_stage_count, dtype=torch.long, device=self.device
        )
        self.closure_stage_min_target_rmse = torch.full(
            (closure_stage_count,), torch.inf, device=self.device
        )
        self.closure_stage_min_canonical_rmse = torch.full(
            (closure_stage_count,), torch.inf, device=self.device
        )
        self.closure_stage_min_position_error = torch.full(
            (closure_stage_count,), torch.inf, device=self.device
        )
        self.closure_stage_min_axis_error = torch.full(
            (closure_stage_count,), torch.inf, device=self.device
        )
        self.closure_stage_best_hold = torch.zeros(
            closure_stage_count, dtype=torch.long, device=self.device
        )
        self.episode_final_stage_entered = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_final_triad_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_final_eligible_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_final_min_canonical_rmse = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self.episode_final_max_hold = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.completed_final_stage_entries = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_final_triad_seen = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_final_eligible_seen = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.latest_success_serial = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.latest_success_qpos = torch.zeros(
            (self.num_envs, 27), device=self.device
        )
        self.latest_success_qvel = torch.zeros(
            (self.num_envs, 26), device=self.device
        )
        self.latest_success_penetration = torch.zeros(
            self.num_envs, device=self.device
        )
        self.latest_success_contact_force = torch.zeros(
            (self.num_envs, 5), device=self.device
        )
        self.completed_final_min_canonical_rmse_sum = torch.zeros(
            (), device=self.device
        )
        self.completed_final_min_canonical_rmse_best = torch.full(
            (), torch.inf, device=self.device
        )
        self.completed_final_max_hold_sum = torch.zeros(
            (), device=self.device
        )
        self.completed_final_max_hold_best = torch.zeros(
            (), dtype=torch.long, device=self.device
        )

    @staticmethod
    def _replace_rows(
        value: torch.Tensor,
        mask: torch.Tensor,
        replacement: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(mask.unsqueeze(-1), replacement, value)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            reset_ids = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            reset_ids = env_ids
        else:
            reset_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        diagnostics_ready = hasattr(
            self, "episode_final_stage_entered"
        )
        success_ready = hasattr(self, "episode_p5_success")
        if success_ready:
            if not hasattr(self, "last_reset_success"):
                self.last_reset_success = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            else:
                self.last_reset_success.zero_()
        if diagnostics_ready:
            diagnostic_ids = reset_ids
            entered = self.episode_final_stage_entered[diagnostic_ids]
            entered_ids = diagnostic_ids[entered]
            entered_count = entered.sum()
            if torch.any(entered):
                final_rmse = self.episode_final_min_canonical_rmse[
                    entered_ids
                ]
                final_hold = self.episode_final_max_hold[entered_ids]
                self.completed_final_stage_entries += entered_count
                self.completed_final_triad_seen += (
                    self.episode_final_triad_seen[entered_ids].sum()
                )
                self.completed_final_eligible_seen += (
                    self.episode_final_eligible_seen[entered_ids].sum()
                )
                self.completed_final_min_canonical_rmse_sum += (
                    final_rmse.sum()
                )
                self.completed_final_min_canonical_rmse_best = torch.minimum(
                    self.completed_final_min_canonical_rmse_best,
                    final_rmse.min(),
                )
                self.completed_final_max_hold_sum += final_hold.float().sum()
                self.completed_final_max_hold_best = torch.maximum(
                    self.completed_final_max_hold_best,
                    final_hold.max(),
                )
        if success_ready:
            self.last_reset_success[reset_ids] = (
                self.episode_p5_success[reset_ids]
            )
        super()._reset_idx(env_ids)
        if not hasattr(self, "chain_phase_hold"):
            return
        if env_ids is None:
            env_ids_tensor = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        rows = torch.arange(self.num_envs, device=self.device)
        reset = (
            rows.unsqueeze(-1) == env_ids_tensor.reshape(1, -1)
        ).any(dim=-1)
        self.chain_phase_hold = torch.where(
            reset,
            torch.zeros_like(self.chain_phase_hold),
            self.chain_phase_hold,
        )
        self.chain_blend_age = torch.where(
            reset,
            torch.full_like(
                self.chain_blend_age,
                int(self.cfg.chain_blend_steps),
            ),
            self.chain_blend_age,
        )
        self.chain_previous_action = torch.where(
            reset.unsqueeze(-1),
            torch.zeros_like(self.chain_previous_action),
            self.chain_previous_action,
        )
        self.closure_origin_hand = torch.where(
            reset.unsqueeze(-1),
            torch.zeros_like(self.closure_origin_hand),
            self.closure_origin_hand,
        )
        self.closure_brace_hand = torch.where(
            reset.unsqueeze(-1),
            torch.zeros_like(self.closure_brace_hand),
            self.closure_brace_hand,
        )
        self.closure_origin_pen_pos = torch.where(
            reset.unsqueeze(-1),
            torch.zeros_like(self.closure_origin_pen_pos),
            self.closure_origin_pen_pos,
        )
        identity_quat = torch.zeros_like(self.closure_origin_pen_quat)
        identity_quat[:, 0] = 1.0
        self.closure_origin_pen_quat = torch.where(
            reset.unsqueeze(-1),
            identity_quat,
            self.closure_origin_pen_quat,
        )
        if diagnostics_ready:
            self.episode_final_stage_entered = torch.where(
                reset,
                torch.zeros_like(self.episode_final_stage_entered),
                self.episode_final_stage_entered,
            )
            self.episode_final_triad_seen = torch.where(
                reset,
                torch.zeros_like(self.episode_final_triad_seen),
                self.episode_final_triad_seen,
            )
            self.episode_final_eligible_seen = torch.where(
                reset,
                torch.zeros_like(self.episode_final_eligible_seen),
                self.episode_final_eligible_seen,
            )
            self.episode_final_min_canonical_rmse = torch.where(
                reset,
                torch.full_like(
                    self.episode_final_min_canonical_rmse, torch.inf
                ),
                self.episode_final_min_canonical_rmse,
            )
            self.episode_final_max_hold = torch.where(
                reset,
                torch.zeros_like(self.episode_final_max_hold),
                self.episode_final_max_hold,
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        with torch.inference_mode():
            policy_actions = actions
            scripted_teacher = None
            if hasattr(self, "scripted_thumb_teacher_first"):
                if bool(
                    self.cfg.scripted_thumb_teacher_pair_with_bundle
                ):
                    teacher_ids = (
                        self.current_bank_ids
                        % self.scripted_thumb_teacher_first.shape[0]
                    )
                else:
                    teacher_ids = (
                        torch.arange(self.num_envs, device=self.device)
                        % self.scripted_thumb_teacher_first.shape[0]
                    )
                first = self.scripted_thumb_teacher_first[teacher_ids]
                second = self.scripted_thumb_teacher_second[teacher_ids]
                first_duration = self.scripted_thumb_teacher_first_duration[
                    teacher_ids
                ]
                second_duration = self.scripted_thumb_teacher_second_duration[
                    teacher_ids
                ]
                age = self.episode_length_buf
                if bool(self.cfg.scripted_thumb_teacher_hold_latch):
                    self.scripted_thumb_teacher_hold_latched |= (
                        self.chain_phase_hold > 0
                    )
                use_first = age < first_duration
                use_second = (
                    (age < first_duration + second_duration)
                    | self.scripted_thumb_teacher_hold_latched
                ) & ~use_first
                transition_steps = int(
                    self.cfg.scripted_thumb_teacher_transition_blend_steps
                )
                if transition_steps > 0:
                    second_age = (
                        age - first_duration + 1
                    ).clamp(min=0)
                    second_alpha = (
                        second_age.float() / float(transition_steps)
                    ).clamp(0.0, 1.0)
                    second = torch.lerp(
                        first,
                        second,
                        second_alpha.unsqueeze(-1),
                    )
                scripted_teacher = torch.where(
                    use_first.unsqueeze(-1),
                    first,
                    torch.where(
                        use_second.unsqueeze(-1),
                        second,
                        torch.zeros_like(first),
                    ),
                )
            observation = self._get_observations()["policy"]
            candidates = [
                policy(observation) for policy in self.prefix_policies
            ]
            brace_teacher = (
                self.brace_teacher_policy(observation)
                if self.brace_teacher_policy is not None
                else None
            )
            if scripted_teacher is not None and bool(
                self.cfg.scripted_thumb_teacher_use_normal_residual_path
            ):
                teacher_scale = float(
                    self.cfg.scripted_thumb_teacher_action_scale
                )
                if teacher_scale <= 0.0:
                    raise ValueError(
                        "scripted_thumb_teacher_action_scale must be "
                        "positive for normal residual-path replay"
                    )
                policy_actions = scripted_teacher + (
                    float(
                        self.cfg.scripted_thumb_teacher_residual_scale
                    )
                    / teacher_scale
                ) * policy_actions
                scripted_teacher = None
            if scripted_teacher is not None and brace_teacher is not None:
                scripted_teacher = scripted_teacher.clone()
                scripted_teacher[:, 0:8] = brace_teacher[:, 0:8]
            candidates[2] = (
                self.cfg.approach_action_rescale * candidates[2]
            )
            candidates[3] = (
                self.cfg.contact_action_rescale * candidates[3]
            )
            candidates[2][:, 4:8] *= (
                self.cfg.approach_middle_multiplier
            )
            candidates[3][:, 4:8] *= (
                self.cfg.contact_middle_multiplier
            )
            selected = actions
            for phase_id, candidate in enumerate(candidates):
                active = (
                    (self.chain_phase_ids == phase_id)
                    & (phase_id < self.cfg.training_start_phase)
                )
                selected = torch.where(
                    active.unsqueeze(-1), candidate, selected
                )
            prefix_servo_active = (
                (self.chain_phase_ids == 2)
                | (self.chain_phase_ids == 3)
            )
            if float(self.cfg.prefix_middle_servo_gain) != 0.0:
                mask = self._stage_action_mask()
                increment = (
                    0.25 * self.cfg.action_scale * mask
                ).clamp_min(1.0e-6)
                prefix_servo = (
                    float(self.cfg.prefix_middle_servo_gain)
                    * (
                        self.target_hand_qpos
                        - self.policy_targets
                    )
                    / increment
                )
                prefix_servo[:, :4] = 0.0
                prefix_servo[:, 8:20] = 0.0
                selected = torch.where(
                    prefix_servo_active.unsqueeze(-1),
                    (selected + prefix_servo).clamp(-1.0, 1.0),
                    selected,
                )
            training_active = (
                self.chain_phase_ids >= self.cfg.training_start_phase
                if bool(self.cfg.sequential_closure)
                else self.chain_phase_ids
                == self.cfg.training_start_phase
            )
            residual_active = (
                (
                    self.chain_phase_ids
                    >= int(self.cfg.residual_start_phase)
                )
                & (
                    self.chain_phase_ids
                    <= int(self.cfg.residual_end_phase)
                )
            )
            if float(self.cfg.target_servo_gain) > 0.0:
                mask = self._stage_action_mask()
                increment = (
                    0.25 * self.cfg.action_scale * mask
                ).clamp_min(1.0e-6)
                servo = (
                    float(self.cfg.target_servo_gain)
                    * (
                        self.target_hand_qpos
                        - self.policy_targets
                    )
                    / increment
                )
                servo = torch.where(
                    mask > 0.0, servo, torch.zeros_like(servo)
                ).clamp(-1.0, 1.0)
                if bool(self.cfg.thumb_wrap_transfer):
                    closure_level = (
                        self.chain_phase_ids
                        - int(self.cfg.training_start_phase)
                    )
                    thumb_scale = torch.where(
                        (closure_level >= 2).unsqueeze(-1),
                        torch.full(
                            (self.num_envs, 1),
                            float(self.cfg.thumb_wrap_thumb_servo_scale),
                            device=self.device,
                        ),
                        torch.zeros(
                            (self.num_envs, 1), device=self.device
                        ),
                    )
                    servo[:, 16:20] *= thumb_scale
                    middle_brace = (
                        (closure_level >= 0)
                        & (
                            closure_level
                            <= int(
                                self.cfg.thumb_wrap_stage_boundaries[0]
                            )
                        )
                    )
                    servo[:, 4:8] *= torch.where(
                        middle_brace.unsqueeze(-1),
                        torch.full(
                            (self.num_envs, 1),
                            float(
                                self.cfg.thumb_wrap_ring_approach_servo_scale
                            ),
                            device=self.device,
                        ),
                        torch.ones(
                            (self.num_envs, 1), device=self.device
                        ),
                    )
                if bool(self.cfg.ring_only_residual):
                    residual_mask = torch.cat(
                        (
                            torch.zeros(
                                (self.num_envs, 8), device=self.device
                            ),
                            torch.ones(
                                (self.num_envs, 4), device=self.device
                            ),
                            torch.zeros(
                                (self.num_envs, 8), device=self.device
                            ),
                        ),
                        dim=-1,
                    )
                elif bool(self.cfg.middle_only_residual):
                    residual_mask = torch.cat(
                        (
                            torch.zeros(
                                (self.num_envs, 4), device=self.device
                            ),
                            torch.ones(
                                (self.num_envs, 4), device=self.device
                            ),
                            torch.zeros(
                                (self.num_envs, 12), device=self.device
                            ),
                        ),
                        dim=-1,
                    )
                elif bool(self.cfg.index_middle_only_residual):
                    residual_mask = torch.cat(
                        (
                            torch.ones(
                                (self.num_envs, 8), device=self.device
                            ),
                            torch.zeros(
                                (self.num_envs, 12), device=self.device
                            ),
                        ),
                        dim=-1,
                    )
                elif bool(self.cfg.index_middle_thumb_only_residual):
                    residual_mask = torch.cat(
                        (
                            torch.ones(
                                (self.num_envs, 8), device=self.device
                            ),
                            torch.zeros(
                                (self.num_envs, 8), device=self.device
                            ),
                            torch.ones(
                                (self.num_envs, 4), device=self.device
                            ),
                        ),
                        dim=-1,
                    )
                elif bool(self.cfg.index_ring_thumb_only_residual):
                    residual_mask = torch.cat(
                        (
                            torch.ones(
                                (self.num_envs, 4), device=self.device
                            ),
                            torch.zeros(
                                (self.num_envs, 4), device=self.device
                            ),
                            torch.ones(
                                (self.num_envs, 4), device=self.device
                            ),
                            torch.zeros(
                                (self.num_envs, 4), device=self.device
                            ),
                            torch.ones(
                                (self.num_envs, 4), device=self.device
                            ),
                        ),
                        dim=-1,
                    )
                elif bool(self.cfg.thumb_only_residual):
                    residual_mask = torch.cat(
                        (
                            torch.zeros(
                                (self.num_envs, 16), device=self.device
                            ),
                            torch.ones(
                                (self.num_envs, 4), device=self.device
                            ),
                        ),
                        dim=-1,
                    )
                elif bool(self.cfg.thumb_index_only_residual):
                    residual_mask = torch.cat(
                        (
                            torch.ones(
                                (self.num_envs, 4), device=self.device
                            ),
                            torch.zeros(
                                (self.num_envs, 12), device=self.device
                            ),
                            torch.ones(
                                (self.num_envs, 4), device=self.device
                            ),
                        ),
                        dim=-1,
                    )
                else:
                    residual_mask = torch.ones_like(actions)
                if bool(
                    self.cfg.phase_conditioned_transfer_residual
                ):
                    closure_level = (
                        self.chain_phase_ids
                        - int(self.cfg.training_start_phase)
                    )
                    prefix_transfer = closure_level < int(
                        self.cfg.phase_conditioned_brace_start_level
                    )
                    thumb_transfer = (
                        closure_level
                        >= int(
                            self.cfg.phase_conditioned_brace_start_level
                        )
                    ) & (
                        closure_level <= int(
                            self.cfg.thumb_wrap_stage_boundaries[2]
                        )
                    )
                    takeover_transfer = (
                        closure_level
                        > int(self.cfg.thumb_wrap_stage_boundaries[2])
                    ) & (
                        closure_level
                        <= int(self.cfg.thumb_wrap_stage_boundaries[3])
                    )
                    late_transfer = closure_level > int(
                        self.cfg.thumb_wrap_stage_boundaries[3]
                    )
                    residual_mask = torch.where(
                        prefix_transfer.unsqueeze(-1),
                        torch.ones_like(actions),
                        torch.zeros_like(actions),
                    )
                    residual_mask[:, 16:20] = torch.maximum(
                        residual_mask[:, 16:20],
                        (
                            thumb_transfer
                            | takeover_transfer
                            | late_transfer
                        ).float().unsqueeze(-1),
                    )
                    residual_mask[:, 0:4] = torch.maximum(
                        residual_mask[:, 0:4],
                        (
                            takeover_transfer | late_transfer
                        ).float().unsqueeze(-1),
                    )
                    residual_mask[:, 4:12] = torch.maximum(
                        residual_mask[:, 4:12],
                        late_transfer.float().unsqueeze(-1),
                    )
                if bool(self.cfg.thumb_wrap_transfer):
                    closure_level = (
                        self.chain_phase_ids
                        - int(self.cfg.training_start_phase)
                    )
                    transfer_mask = torch.zeros_like(actions)
                    transfer_mask[:, 0:8] = 1.0
                    thumb_active = closure_level >= 2
                    transfer_mask[:, 16:20] = thumb_active.float().unsqueeze(
                        -1
                    )
                    ring_active = closure_level > int(
                        self.cfg.thumb_wrap_stage_boundaries[-1]
                    )
                    transfer_mask[:, 8:12] = ring_active.float().unsqueeze(
                        -1
                    )
                    residual_mask *= transfer_mask
                residual = (
                    policy_actions
                    * residual_active.unsqueeze(-1)
                    * residual_mask
                )
                if brace_teacher is not None and scripted_teacher is None:
                    if bool(
                        self.cfg.phase_conditioned_transfer_residual
                    ):
                        closure_level = (
                            self.chain_phase_ids
                            - int(self.cfg.training_start_phase)
                        )
                        brace_started = closure_level >= int(
                            self.cfg.phase_conditioned_brace_start_level
                        )
                        teacher_index = brace_started & (
                            closure_level <= int(
                                self.cfg.thumb_wrap_stage_boundaries[2]
                            )
                        )
                        teacher_middle = brace_started & (
                            closure_level <= int(
                                self.cfg.thumb_wrap_stage_boundaries[3]
                            )
                        )
                        residual[:, 0:4] = torch.where(
                            teacher_index.unsqueeze(-1),
                            brace_teacher[:, 0:4],
                            residual[:, 0:4],
                        )
                        residual[:, 4:8] = torch.where(
                            teacher_middle.unsqueeze(-1),
                            brace_teacher[:, 4:8],
                            residual[:, 4:8],
                        )
                    else:
                        if bool(self.cfg.brace_teacher_index):
                            residual[:, 0:4] = brace_teacher[:, 0:4]
                        residual[:, 4:8] = brace_teacher[:, 4:8]
                residual_scale = torch.full(
                    (self.num_envs, 1),
                    float(self.cfg.trainee_policy_scale),
                    dtype=actions.dtype,
                    device=self.device,
                )
                if bool(
                    self.cfg.phase_conditioned_transfer_residual
                ):
                    closure_level = (
                        self.chain_phase_ids
                        - int(self.cfg.training_start_phase)
                    )
                    residual_scale = torch.where(
                        (
                            closure_level
                            < int(
                                self.cfg.phase_conditioned_brace_start_level
                            )
                        ).unsqueeze(-1),
                        torch.full_like(
                            residual_scale,
                            float(
                                self.cfg.phase_conditioned_prefix_residual_scale
                            ),
                        ),
                        residual_scale,
                    )
                if float(self.cfg.final_residual_scale) >= 0.0:
                    final_phase = (
                        int(self.cfg.training_start_phase)
                        + len(self.cfg.closure_target_blends)
                        - 1
                    )
                    residual_scale = torch.where(
                        (self.chain_phase_ids == final_phase).unsqueeze(-1),
                        torch.full_like(
                            residual_scale,
                            float(self.cfg.final_residual_scale),
                        ),
                        residual_scale,
                    )
                if hasattr(self, "external_final_residual_scale"):
                    external_scale = self.external_final_residual_scale
                    if external_scale.ndim == 1:
                        external_scale = external_scale.unsqueeze(-1)
                    residual_scale = torch.where(
                        external_scale >= 0.0,
                        external_scale.to(
                            dtype=residual_scale.dtype,
                            device=residual_scale.device,
                        ),
                        residual_scale,
                    )
                if scripted_teacher is None:
                    trainee = servo + residual_scale * residual
                else:
                    scripted_residual_scale = torch.full_like(
                        residual_scale,
                        float(
                            self.cfg.scripted_thumb_teacher_residual_scale
                        ),
                    )
                    if float(self.cfg.final_residual_scale) >= 0.0:
                        scripted_residual_scale = torch.where(
                            (
                                self.chain_phase_ids == final_phase
                            ).unsqueeze(-1),
                            torch.full_like(
                                scripted_residual_scale,
                                float(self.cfg.final_residual_scale),
                            ),
                            scripted_residual_scale,
                        )
                    if hasattr(self, "external_final_residual_scale"):
                        external_scale = self.external_final_residual_scale
                        if external_scale.ndim == 1:
                            external_scale = external_scale.unsqueeze(-1)
                        scripted_residual_scale = torch.where(
                            external_scale >= 0.0,
                            external_scale.to(
                                dtype=scripted_residual_scale.dtype,
                                device=scripted_residual_scale.device,
                            ),
                            scripted_residual_scale,
                        )
                    trainee = (
                        servo
                        + float(
                            self.cfg.scripted_thumb_teacher_action_scale
                        )
                        * scripted_teacher
                        + scripted_residual_scale * residual
                    )
                trainee = trainee.clamp(-1.0, 1.0)
                selected = torch.where(
                    training_active.unsqueeze(-1),
                    trainee,
                    selected,
                )
                prefix_residual = residual_active & ~training_active
                selected = torch.where(
                    prefix_residual.unsqueeze(-1),
                    (
                        selected
                        + float(self.cfg.trainee_policy_scale) * residual
                    ).clamp(-1.0, 1.0),
                    selected,
                )
            blending = (
                self.chain_blend_age < self.cfg.chain_blend_steps
            )
            alpha = (
                (self.chain_blend_age + 1).float()
                / max(int(self.cfg.chain_blend_steps), 1)
            ).clamp(0.0, 1.0).unsqueeze(-1)
            selected = torch.where(
                blending.unsqueeze(-1),
                torch.lerp(
                    self.chain_previous_action, selected, alpha
                ),
                selected,
            ).clamp(-1.0, 1.0)
            replay_steps = int(self.cfg.cached_action_replay_steps)
            if replay_steps > 0:
                replay_cached_action = self.episode_length_buf < replay_steps
                selected = torch.where(
                    replay_cached_action.unsqueeze(-1),
                    float(self.cfg.cached_action_replay_scale)
                    * self.chain_previous_action,
                    selected,
                )
            self.chain_blend_age = torch.where(
                blending,
                self.chain_blend_age + 1,
                self.chain_blend_age,
            )
        super()._pre_physics_step(selected)

    def _stage_action_mask(self) -> torch.Tensor:
        mask = super()._stage_action_mask()
        train_tail = (
            self.chain_phase_ids >= self.cfg.training_start_phase
        )
        mask[:, 8:12] = torch.maximum(
            mask[:, 8:12],
            train_tail.float().unsqueeze(-1),
        )
        if bool(self.cfg.thumb_wrap_transfer):
            active = train_tail.float().unsqueeze(-1)
            mask[:, 0:12] = torch.maximum(mask[:, 0:12], active)
            mask[:, 16:20] = torch.maximum(mask[:, 16:20], active)
        return mask

    def _closure_target(self, blend: torch.Tensor | float) -> torch.Tensor:
        canonical = self.p1_hand.unsqueeze(0).expand_as(
            self.closure_origin_hand
        )
        if bool(self.cfg.thumb_wrap_transfer):
            if isinstance(blend, torch.Tensor):
                progress = ((blend - 0.45) / 0.55).clamp(0.0, 1.0)
            else:
                progress = torch.full(
                    (self.num_envs, 1),
                    max(0.0, min(1.0, (float(blend) - 0.45) / 0.55)),
                    dtype=canonical.dtype,
                    device=self.device,
                )

            # Align interpolation with brace 4-5, under 6-8, wrap 9-12,
            # takeover 13-14, and unload 15-17.
            brace_alpha = (progress / 0.18).clamp(0.0, 1.0)
            under_alpha = ((progress - 0.18) / 0.28).clamp(0.0, 1.0)
            wrap_alpha = ((progress - 0.46) / 0.36).clamp(0.0, 1.0)
            takeover_alpha = ((progress - 0.82) / 0.10).clamp(0.0, 1.0)
            unload_alpha = ((progress - 0.92) / 0.08).clamp(0.0, 1.0)

            target = self.closure_origin_hand.clone()
            scripted_brace = self.p5a_transfer_hand.unsqueeze(0).expand_as(
                target
            )
            latched_brace = torch.where(
                (
                    torch.linalg.norm(
                        self.closure_brace_hand, dim=-1, keepdim=True
                    )
                    > 1.0e-6
                ),
                self.closure_brace_hand,
                self.closure_origin_hand,
            )
            brace = torch.where(
                (progress >= 0.15),
                latched_brace,
                scripted_brace,
            )

            # Establish the temporary index/middle clamp before moving thumb.
            for finger_slice in (slice(0, 4), slice(4, 8)):
                target[:, finger_slice] = torch.lerp(
                    self.closure_origin_hand[:, finger_slice],
                    brace[:, finger_slice],
                    brace_alpha,
                )

            # Thumb first reaches the P4 under-support posture, then wraps to
            # the opposite P5 side. It only closes toward P1 after that.
            thumb_origin = brace[:, 16:20]
            thumb_under = self.p4_thumb_under_hand[16:20].unsqueeze(0)
            thumb_opposite = self.p5a_transfer_hand[16:20].unsqueeze(0)
            thumb = torch.lerp(thumb_origin, thumb_under, under_alpha)
            thumb = torch.lerp(thumb, thumb_opposite, wrap_alpha)
            thumb = torch.lerp(
                thumb,
                canonical[:, 16:20],
                takeover_alpha,
            )
            target[:, 16:20] = thumb

            # Index and middle maintain the brace while thumb travels. Middle
            # returns only after the thumb/index pinch takes over; ring stays
            # inactive at its live entry pose in the corrected chain.
            target[:, 0:4] = torch.lerp(
                target[:, 0:4],
                canonical[:, 0:4],
                takeover_alpha,
            )
            target[:, 4:8] = torch.lerp(
                brace[:, 4:8],
                canonical[:, 4:8],
                unload_alpha,
            )
            if not bool(self.cfg.closure_keep_ring_fixed_during_unload):
                target[:, 8:12] = torch.lerp(
                    target[:, 8:12],
                    canonical[:, 8:12],
                    unload_alpha,
                )
            preload = float(self.cfg.closure_final_contact_preload_rad)
            if preload > 0.0:
                if isinstance(blend, torch.Tensor):
                    preload_active = (blend >= 0.95).float()
                else:
                    preload_active = torch.full(
                        (self.num_envs, 1),
                        float(float(blend) >= 0.95),
                        dtype=target.dtype,
                        device=self.device,
                    )
                for flexion_slice in (
                    slice(1, 4),
                    slice(5, 8),
                    slice(17, 20),
                ):
                    target[:, flexion_slice] += (
                        preload * preload_active
                    )
            return target
        if not bool(self.cfg.cycle_aligned_p1):
            return torch.lerp(
                self.closure_origin_hand, canonical, blend
            )
        target = self.closure_origin_hand.clone()
        relative_return = self.p1_hand - self.p5_hand
        if isinstance(blend, torch.Tensor):
            finger_blend = blend
        else:
            finger_blend = torch.full(
                (self.num_envs, 1),
                float(blend),
                dtype=target.dtype,
                device=self.device,
            )
        if bool(self.cfg.closure_latch_live_target_on_reset):
            self.target_hand_qpos[env_ids_tensor] = self.policy_targets[
                env_ids_tensor
            ]
        elif bool(self.cfg.closure_return_middle_instead_of_ring):
            # Thumb/index have taken over; move middle toward P1 while
            # keeping the already inactive ring at its live entry pose.
            target[:, 4:8] = (
                self.closure_origin_hand[:, 4:8]
                + finger_blend * relative_return[4:8]
            )
        else:
            target[:, 8:12] = (
                self.closure_origin_hand[:, 8:12]
                + finger_blend * relative_return[8:12]
            )
        return target

    def _closure_pen_target(
        self,
        progress: torch.Tensor | float,
        position_progress: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(progress, torch.Tensor):
            alpha = progress
        else:
            alpha = torch.full(
                (self.num_envs, 1),
                float(progress),
                dtype=self.closure_origin_pen_pos.dtype,
                device=self.device,
            )
        if position_progress is None:
            position_alpha = alpha
        elif isinstance(position_progress, torch.Tensor):
            position_alpha = position_progress
        else:
            position_alpha = torch.full_like(
                alpha, float(position_progress)
            )
        target_pos = torch.lerp(
            self.closure_origin_pen_pos,
            self.p1_pen_pos.unsqueeze(0).expand_as(
                self.closure_origin_pen_pos
            ),
            position_alpha,
        )
        pen_x_axis = torch.zeros_like(self.closure_origin_pen_pos)
        pen_x_axis[:, 0] = 1.0
        origin_axis = quat_apply(
            self.closure_origin_pen_quat, pen_x_axis
        )
        p1_axis = self.p1_pen_axis.unsqueeze(0).expand_as(origin_axis)
        aligned_p1_axis = torch.where(
            (torch.sum(origin_axis * p1_axis, dim=-1) < 0.0).unsqueeze(-1),
            -p1_axis,
            p1_axis,
        )
        target_axis = torch.lerp(
            origin_axis, aligned_p1_axis, alpha
        )
        target_axis = target_axis / torch.linalg.norm(
            target_axis, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        rotation_axis = torch.linalg.cross(
            origin_axis, target_axis, dim=-1
        )
        rotation_axis_norm = torch.linalg.norm(
            rotation_axis, dim=-1, keepdim=True
        )
        fallback_axis = torch.zeros_like(rotation_axis)
        fallback_axis[:, 2] = 1.0
        rotation_axis = torch.where(
            rotation_axis_norm > 1.0e-7,
            rotation_axis / rotation_axis_norm.clamp_min(1.0e-8),
            fallback_axis,
        )
        rotation_angle = torch.acos(
            torch.sum(origin_axis * target_axis, dim=-1).clamp(-1.0, 1.0)
        )
        target_quat = quat_mul(
            quat_from_angle_axis(rotation_angle, rotation_axis),
            self.closure_origin_pen_quat,
        )
        return target_pos, target_quat

    def _handoff(self, transition: torch.Tensor) -> None:
        if not torch.any(transition):
            return
        phase = self.chain_phase_ids
        next_phase = torch.where(transition, phase + 1, phase)
        for phase_id in range(1, int(self.cfg.training_start_phase) + 1):
            self.chain_phase_entries[phase_id] += (
                transition & (next_phase == phase_id)
            ).sum()
        next_stage = torch.where(
            next_phase == 1,
            torch.full_like(next_phase, 4),
            torch.where(
                next_phase == 2,
                torch.ones_like(next_phase),
                torch.where(
                    next_phase == 3,
                    torch.full_like(next_phase, 2),
                    torch.full_like(next_phase, 3),
                ),
            ),
        )
        self.chain_phase_ids = next_phase
        self.p5_stage_ids = torch.where(
            transition, next_stage, self.p5_stage_ids
        )
        target = self.target_hand_qpos.clone()

        to_capture = transition & (phase == 0)
        capture_target = self.policy_targets.clone()
        capture_target[:, :4] = self.p5_hand[:4]
        capture_target[:, 16:20] = self.p5_hand[16:20]
        target = self._replace_rows(
            target, to_capture, capture_target
        )

        to_approach = transition & (phase == 1)
        approach_target = self.policy_targets.clone()
        approach_target[:, 4:8] = (
            0.45 * self.policy_targets[:, 4:8]
            + 0.55 * self.p1_hand[4:8]
        )
        target = self._replace_rows(
            target, to_approach, approach_target
        )

        to_contact = transition & (phase == 2)
        if int(self.cfg.training_start_phase) == 3:
            projected = self.hand_qpos.clone()
            self.closure_origin_hand = self._replace_rows(
                self.closure_origin_hand, to_contact, projected
            )
            contact_target = self._closure_target(
                float(self.cfg.canonical_target_blend)
            )
            self.policy_targets = self._replace_rows(
                self.policy_targets, to_contact, projected
            )
            if bool(self.cfg.sequential_closure):
                self.closure_stage_entries[0] += to_contact.sum()
        else:
            contact_target = self.policy_targets.clone()
            contact_target[:, 4:8] = self.p1_hand[4:8]
        target = self._replace_rows(
            target, to_contact, contact_target
        )

        to_triad = transition & (phase == 3)
        projected = self.hand_qpos.clone()
        if int(self.cfg.training_start_phase) == 4:
            self.closure_origin_hand = self._replace_rows(
                self.closure_origin_hand, to_triad, projected
            )
            self.closure_origin_pen_pos = self._replace_rows(
                self.closure_origin_pen_pos,
                to_triad,
                self.pen_pos,
            )
            self.closure_origin_pen_quat = self._replace_rows(
                self.closure_origin_pen_quat,
                to_triad,
                self.pen_quat,
            )
            if bool(self.cfg.closure_pen_pose_gate):
                position_progress = None
                if len(
                    self.cfg.closure_pen_position_target_progress
                ) > 0:
                    position_progress = float(
                        self.cfg
                        .closure_pen_position_target_progress[0]
                    )
                pen_target_pos, pen_target_quat = (
                    self._closure_pen_target(
                        float(self.cfg.closure_pen_target_progress[0]),
                        position_progress,
                    )
                )
                self.target_pen_pos = self._replace_rows(
                    self.target_pen_pos, to_triad, pen_target_pos
                )
                self.target_pen_quat = self._replace_rows(
                    self.target_pen_quat, to_triad, pen_target_quat
                )
            if bool(self.cfg.sequential_closure):
                self.closure_stage_entries[0] += to_triad.sum()
        self.policy_targets = self._replace_rows(
            self.policy_targets, to_triad, projected
        )
        triad_target = self._closure_target(
            float(self.cfg.canonical_target_blend)
        )
        target = self._replace_rows(target, to_triad, triad_target)
        self.target_hand_qpos = target
        self.stage_hold_count = torch.where(
            transition,
            torch.zeros_like(self.stage_hold_count),
            self.stage_hold_count,
        )
        self.p5_hold_count = torch.where(
            transition,
            torch.zeros_like(self.p5_hold_count),
            self.p5_hold_count,
        )
        self.chain_phase_hold = torch.where(
            transition,
            torch.zeros_like(self.chain_phase_hold),
            self.chain_phase_hold,
        )
        self.chain_phase_hold_loss = torch.where(
            transition,
            torch.zeros_like(self.chain_phase_hold_loss),
            self.chain_phase_hold_loss,
        )
        self.chain_previous_action = self._replace_rows(
            self.chain_previous_action, transition, self.actions
        )
        self.chain_blend_age = torch.where(
            transition,
            torch.zeros_like(self.chain_blend_age),
            self.chain_blend_age,
        )

    def _advance_closure(
        self, advance: torch.Tensor, phase: torch.Tensor
    ) -> None:
        if not torch.any(advance):
            return
        next_phase = torch.where(advance, phase + 1, phase)
        if bool(self.cfg.thumb_wrap_transfer):
            latch_brace = advance & (
                phase == int(self.cfg.training_start_phase)
            )
            self.closure_brace_hand = self._replace_rows(
                self.closure_brace_hand,
                latch_brace,
                self.hand_qpos,
            )
            latch_thumb_start = advance & (
                phase == int(self.cfg.training_start_phase) + 1
            )
            self.closure_origin_pen_pos = self._replace_rows(
                self.closure_origin_pen_pos,
                latch_thumb_start,
                self.pen_pos,
            )
            self.closure_origin_pen_quat = self._replace_rows(
                self.closure_origin_pen_quat,
                latch_thumb_start,
                self.pen_quat,
            )
        for phase_id in range(
            int(self.cfg.training_start_phase) + 1,
            self.chain_phase_entries.numel(),
        ):
            self.chain_phase_entries[phase_id] += (
                advance & (next_phase == phase_id)
            ).sum()
        self.chain_phase_ids = next_phase
        next_stage = (next_phase - 1).clamp(2, 4)
        self.p5_stage_ids = torch.where(
            advance, next_stage, self.p5_stage_ids
        )
        blend_values = torch.as_tensor(
            self.cfg.closure_target_blends,
            dtype=torch.float32,
            device=self.device,
        )
        closure_start_phase = int(self.cfg.training_start_phase)
        blend_id = (next_phase - closure_start_phase).clamp(
            0, len(self.cfg.closure_target_blends) - 1
        )
        for stage_id in range(1, len(self.cfg.closure_target_blends)):
            self.closure_stage_entries[stage_id] += (
                advance & (blend_id == stage_id)
            ).sum()
        blend = blend_values[blend_id].unsqueeze(-1)
        next_target = self._closure_target(blend)
        self.target_hand_qpos = self._replace_rows(
            self.target_hand_qpos, advance, next_target
        )
        if bool(self.cfg.closure_pen_pose_gate):
            progress_values = torch.as_tensor(
                self.cfg.closure_pen_target_progress,
                dtype=torch.float32,
                device=self.device,
            )
            progress = progress_values[blend_id].unsqueeze(-1)
            position_progress = None
            if len(
                self.cfg.closure_pen_position_target_progress
            ) > 0:
                position_values = torch.as_tensor(
                    self.cfg.closure_pen_position_target_progress,
                    dtype=torch.float32,
                    device=self.device,
                )
                position_progress = position_values[
                    blend_id
                ].unsqueeze(-1)
            pen_target_pos, pen_target_quat = self._closure_pen_target(
                progress, position_progress
            )
            self.target_pen_pos = self._replace_rows(
                self.target_pen_pos, advance, pen_target_pos
            )
            self.target_pen_quat = self._replace_rows(
                self.target_pen_quat, advance, pen_target_quat
            )
        projected = self.hand_qpos.clone()
        self.policy_targets = self._replace_rows(
            self.policy_targets, advance, projected
        )
        self.chain_phase_hold = torch.where(
            advance,
            torch.zeros_like(self.chain_phase_hold),
            self.chain_phase_hold,
        )
        self.chain_phase_hold_loss = torch.where(
            advance,
            torch.zeros_like(self.chain_phase_hold_loss),
            self.chain_phase_hold_loss,
        )
        self.chain_previous_action = self._replace_rows(
            self.chain_previous_action, advance, self.actions
        )
        self.chain_blend_age = torch.where(
            advance,
            torch.zeros_like(self.chain_blend_age),
            self.chain_blend_age,
        )

    def _get_rewards(self) -> torch.Tensor:
        latched_chain_success = self.episode_p5_success.clone()
        super()._get_rewards()
        self._compute_intermediate_values()
        contacts = self.contact_force_norm > self.cfg.p1_contact_force_n
        thumb_index = contacts[:, :2].all(dim=-1)
        index_middle = contacts[:, 1] & contacts[:, 2]
        index_ring = contacts[:, 1] & contacts[:, 3]
        ring_gap = (
            self.tip_surface_dist[:, 3]
            - self.fingertip_radii[:, 3]
        )
        ring_near = ring_gap <= 0.004
        middle_gap = (
            self.tip_surface_dist[:, 2]
            - self.fingertip_radii[:, 2]
        )
        middle_near = (
            (middle_gap >= self.cfg.p1_middle_approach_min_gap_m)
            & (middle_gap <= self.cfg.p1_middle_near_distance_m)
        )
        middle_contact = contacts[:, 2] & (
            middle_gap <= self.cfg.p1_middle_contact_distance_m
        )
        triad_contact = thumb_index & middle_contact
        thumb_below_pen = (
            self.fingertip_pos_local[:, 0, 2] < self.pen_pos[:, 2]
        )
        thumb_index_rel = (
            self.fingertip_pos_local[:, :2]
            - self.pen_pos.unsqueeze(1)
        )
        thumb_index_axial = torch.sum(
            thumb_index_rel * self.pen_axis.unsqueeze(1), dim=-1
        )
        thumb_index_radial = (
            thumb_index_rel
            - thumb_index_axial.unsqueeze(-1)
            * self.pen_axis.unsqueeze(1)
        )
        thumb_index_radial_norm = torch.linalg.norm(
            thumb_index_radial, dim=-1
        ).clamp_min(1.0e-6)
        thumb_index_opposite_cos = (
            torch.sum(
                thumb_index_radial[:, 0]
                * thumb_index_radial[:, 1],
                dim=-1,
            )
            / (
                thumb_index_radial_norm[:, 0]
                * thumb_index_radial_norm[:, 1]
            )
        ).clamp(-1.0, 1.0)
        thumb_index_axial_separation = torch.abs(
            thumb_index_axial[:, 0] - thumb_index_axial[:, 1]
        )
        thumb_index_opposite = (
            thumb_index_opposite_cos
            <= float(self.cfg.thumb_wrap_opposite_cos_max)
        ) & (
            thumb_index_axial_separation
            <= float(self.cfg.thumb_wrap_max_axial_separation_m)
        )
        drift = torch.linalg.norm(
            self.pen_pos - self.init_pen_pos, dim=-1
        )
        axis_dot = torch.sum(
            self.pen_axis * self.init_pen_axis, dim=-1
        ).clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot.abs())
        speed = torch.linalg.norm(self.pen_angvel, dim=-1)
        axis_rate = torch.linalg.norm(
            torch.linalg.cross(self.pen_angvel, self.pen_axis, dim=-1),
            dim=-1,
        )
        linear_speed = torch.linalg.norm(self.pen_linvel, dim=-1)
        peak_force = self.contact_force_norm.max(dim=-1).values
        phase = self.chain_phase_ids
        closure_active = phase >= int(self.cfg.training_start_phase)
        target_pen_axis = quat_apply(
            self.target_pen_quat, self.pen_x_axis
        )
        target_position_error = torch.linalg.norm(
            self.pen_pos - self.target_pen_pos, dim=-1
        )
        target_axis_dot = torch.sum(
            self.pen_axis * target_pen_axis, dim=-1
        ).clamp(-1.0, 1.0)
        target_axis_error = torch.acos(target_axis_dot.abs())
        position_safe_from_init = (
            drift <= self.cfg.p5_max_position_drift_m
        )
        axis_safe_from_init = axis_drift <= torch.deg2rad(
            torch.full_like(
                axis_drift, self.cfg.p5_max_axis_drift_deg
            )
        )
        use_closure_pen_target = (
            closure_active & bool(self.cfg.closure_pen_pose_gate)
        )
        position_safe = torch.where(
            use_closure_pen_target,
            target_position_error
            <= float(self.cfg.closure_pen_position_tolerance_m),
            position_safe_from_init,
        )
        axis_safe = torch.where(
            use_closure_pen_target,
            target_axis_error
            <= torch.deg2rad(
                torch.full_like(
                    target_axis_error,
                    float(self.cfg.closure_pen_axis_tolerance_deg),
                )
            ),
            axis_safe_from_init,
        )
        speed_safe = speed <= self.cfg.p5_max_angular_speed_rad_s
        penetration_safe = (
            self.episode_max_penetration
            <= self.cfg.penetration_threshold
        )
        force_safe = peak_force <= self.cfg.p1_max_force_n
        if bool(self.cfg.thumb_wrap_transfer):
            closure_level_for_safety = (
                phase - int(self.cfg.training_start_phase)
            )
            brace_safety = closure_active & (
                closure_level_for_safety
                <= int(self.cfg.thumb_wrap_stage_boundaries[0])
            )
            acquisition_safety = closure_active & (
                closure_level_for_safety == 0
            )
            position_safe = torch.where(
                brace_safety,
                target_position_error
                <= float(
                    self.cfg.thumb_wrap_brace_pen_position_tolerance_m
                ),
                position_safe,
            )
            axis_safe = torch.where(
                brace_safety,
                target_axis_error
                <= torch.deg2rad(
                    torch.where(
                        acquisition_safety,
                        torch.full_like(
                            target_axis_error,
                            float(
                                self.cfg
                                .thumb_wrap_acquisition_pen_axis_tolerance_deg
                            ),
                        ),
                        torch.full_like(
                            target_axis_error,
                            float(
                                self.cfg
                                .thumb_wrap_brace_pen_axis_tolerance_deg
                            ),
                        ),
                    )
                ),
                axis_safe,
            )
        safe = (
            position_safe
            & axis_safe
            & speed_safe
            & penetration_safe
            & force_safe
        )
        manual_phase = (
            (phase >= 2)
            & (phase < self.cfg.training_start_phase)
        )
        manual_eligible = (
            ((phase == 2) & safe & thumb_index & middle_near)
            | ((phase == 3) & safe & triad_contact)
        ) & manual_phase
        self.chain_phase_hold = torch.where(
            manual_phase,
            torch.where(
                manual_eligible,
                self.chain_phase_hold + 1,
                torch.zeros_like(self.chain_phase_hold),
            ),
            self.chain_phase_hold,
        )
        manual_phase_before_handoff = phase.clone()
        manual_hold_progress = torch.clamp(
            self.chain_phase_hold.float()
            / float(self.cfg.chain_stage_hold_steps),
            0.0,
            1.0,
        )
        transition = (
            ((phase <= 1) & (
                self.stage_hold_count
                >= self.cfg.chain_stage_hold_steps
            ))
            | (
                ((phase == 2) | (phase == 3))
                & (phase < self.cfg.training_start_phase)
                & (
                    self.chain_phase_hold
                    >= self.cfg.chain_stage_hold_steps
                )
            )
        )
        self._handoff(transition)

        phase = self.chain_phase_ids
        triad_error = torch.cat(
            (
                self.hand_qpos[:, 0:8]
                - self.target_hand_qpos[:, 0:8],
                self.hand_qpos[:, 16:20]
                - self.target_hand_qpos[:, 16:20],
            ),
            dim=-1,
        )
        triad_rmse = torch.sqrt(
            torch.mean(triad_error.square(), dim=-1)
        )
        index_ring_error = torch.cat(
            (
                self.hand_qpos[:, 0:4]
                - self.target_hand_qpos[:, 0:4],
                self.hand_qpos[:, 8:12]
                - self.target_hand_qpos[:, 8:12],
            ),
            dim=-1,
        )
        index_ring_rmse = torch.sqrt(
            torch.mean(index_ring_error.square(), dim=-1)
        )
        index_middle_error = (
            self.hand_qpos[:, 0:8]
            - self.target_hand_qpos[:, 0:8]
        )
        index_middle_rmse = torch.sqrt(
            torch.mean(index_middle_error.square(), dim=-1)
        )
        thumb_index_middle_error = torch.cat(
            (
                index_middle_error,
                self.hand_qpos[:, 16:20]
                - self.target_hand_qpos[:, 16:20],
            ),
            dim=-1,
        )
        thumb_index_middle_rmse = torch.sqrt(
            torch.mean(thumb_index_middle_error.square(), dim=-1)
        )
        thumb_index_ring_error = torch.cat(
            (
                index_ring_error,
                self.hand_qpos[:, 16:20]
                - self.target_hand_qpos[:, 16:20],
            ),
            dim=-1,
        )
        thumb_index_ring_rmse = torch.sqrt(
            torch.mean(thumb_index_ring_error.square(), dim=-1)
        )
        thumb_index_error = torch.cat(
            (
                self.hand_qpos[:, 0:4]
                - self.target_hand_qpos[:, 0:4],
                self.hand_qpos[:, 16:20]
                - self.target_hand_qpos[:, 16:20],
            ),
            dim=-1,
        )
        thumb_index_rmse = torch.sqrt(
            torch.mean(thumb_index_error.square(), dim=-1)
        )
        ring_rmse = torch.sqrt(
            torch.mean(
                (
                    self.hand_qpos[:, 8:12]
                    - self.target_hand_qpos[:, 8:12]
                ).square(),
                dim=-1,
            )
        )
        middle_rmse = torch.sqrt(
            torch.mean(
                (
                    self.hand_qpos[:, 4:8]
                    - self.target_hand_qpos[:, 4:8]
                ).square(),
                dim=-1,
            )
        )
        canonical_error = torch.cat(
            (
                self.hand_qpos[:, 0:12] - self.p1_hand[0:12],
                self.hand_qpos[:, 16:20] - self.p1_hand[16:20],
            ),
            dim=-1,
        )
        canonical_rmse = torch.sqrt(
            torch.mean(canonical_error.square(), dim=-1)
        )
        training_active = (
            phase >= self.cfg.training_start_phase
            if bool(self.cfg.sequential_closure)
            else phase == self.cfg.training_start_phase
        )
        closure_start_phase = int(self.cfg.training_start_phase)
        final_phase = (
            closure_start_phase
            + len(self.cfg.closure_target_blends)
            - 1
        )
        closure_level = (phase - closure_start_phase).clamp(
            0, len(self.cfg.closure_target_blends) - 1
        )
        transfer_contact = triad_contact
        transfer_rmse = triad_rmse
        transfer_stage = torch.zeros_like(closure_level)
        if bool(self.cfg.thumb_wrap_transfer):
            brace_end, under_end, wrap_end, takeover_end = (
                int(value)
                for value in self.cfg.thumb_wrap_stage_boundaries
            )
            brace_stage = closure_level <= brace_end
            under_stage = (
                (closure_level > brace_end)
                & (closure_level <= under_end)
            )
            wrap_stage = (
                (closure_level > under_end)
                & (closure_level <= wrap_end)
            )
            takeover_stage = (
                (closure_level > wrap_end)
                & (closure_level <= takeover_end)
            )
            unload_stage = closure_level > takeover_end
            opposite_gate_active = closure_level >= int(
                self.cfg.thumb_wrap_opposite_gate_start_level
            )
            transfer_stage = torch.where(
                under_stage,
                torch.ones_like(transfer_stage),
                transfer_stage,
            )
            transfer_stage = torch.where(
                wrap_stage,
                torch.full_like(transfer_stage, 2),
                transfer_stage,
            )
            transfer_stage = torch.where(
                takeover_stage,
                torch.full_like(transfer_stage, 3),
                transfer_stage,
            )
            transfer_stage = torch.where(
                unload_stage,
                torch.full_like(transfer_stage, 4),
                transfer_stage,
            )
            thumb_contact = contacts[:, 0]
            transfer_contact = torch.where(
                brace_stage,
                index_middle,
                torch.where(
                    under_stage,
                    index_middle & thumb_contact & thumb_below_pen,
                    torch.where(
                        wrap_stage,
                        index_middle
                        & thumb_contact
                        & (~opposite_gate_active | thumb_index_opposite),
                        torch.where(
                            takeover_stage,
                            thumb_index
                            & contacts[:, 2]
                            & thumb_index_opposite,
                            thumb_index & thumb_index_opposite,
                        ),
                    ),
                ),
            )
            transfer_rmse = torch.where(
                brace_stage,
                index_middle_rmse,
                torch.where(
                    under_stage | wrap_stage | takeover_stage,
                    thumb_index_middle_rmse,
                    thumb_index_rmse,
                ),
            )
        rmse_threshold = torch.full_like(
            transfer_rmse, float(self.cfg.p1_triad_pose_rmse_rad)
        )
        if bool(self.cfg.sequential_closure):
            threshold_values = torch.as_tensor(
                self.cfg.closure_stage_rmse_thresholds,
                dtype=torch.float32,
                device=self.device,
            )
            rmse_threshold = threshold_values[closure_level]
        triad_eligible = (
            training_active
            & safe
            & transfer_contact
            & (transfer_rmse <= rmse_threshold)
        )
        handoff_speed_ready = torch.ones_like(
            triad_eligible, dtype=torch.bool
        )
        if (
            bool(self.cfg.thumb_wrap_transfer)
            and float(
                self.cfg.thumb_wrap_handoff_max_angular_speed_rad_s
            )
            > 0.0
        ):
            handoff_speed_ready &= speed <= float(
                self.cfg.thumb_wrap_handoff_max_angular_speed_rad_s
            )
        if (
            bool(self.cfg.thumb_wrap_transfer)
            and float(self.cfg.thumb_wrap_handoff_max_linear_speed_m_s)
            > 0.0
        ):
            handoff_speed_ready &= linear_speed <= float(
                self.cfg.thumb_wrap_handoff_max_linear_speed_m_s
            )
        pinky_clear = ~contacts[:, 4]
        if bool(self.cfg.thumb_wrap_require_pinky_clear):
            handoff_speed_ready &= pinky_clear
        triad_eligible &= handoff_speed_ready
        if bool(self.cfg.sequential_closure):
            final_aux_pose = torch.ones_like(
                triad_eligible, dtype=torch.bool
            )
            if bool(self.cfg.closure_final_ring_pose_gate):
                final_aux_pose &= (
                    ring_rmse <= self.cfg.p1_ring_pose_rmse_rad
                )
            if bool(self.cfg.closure_final_middle_pose_gate):
                final_aux_pose &= (
                    middle_rmse
                    <= float(
                        self.cfg.closure_final_middle_pose_rmse_rad
                    )
                )
            triad_eligible &= (phase < final_phase) | final_aux_pose
        else:
            final_aux_pose = torch.ones_like(
                triad_eligible, dtype=torch.bool
            )
        # Expose exact reward gates for evaluator/training diagnostics. These
        # tensors are overwritten every reward step and never affect physics.
        self.debug_closure_gate_components = torch.stack(
            (
                training_active,
                safe,
                transfer_contact,
                transfer_rmse <= rmse_threshold,
                handoff_speed_ready,
                (phase < final_phase) | final_aux_pose,
            ),
            dim=-1,
        )
        self.debug_closure_eligible = triad_eligible
        contact_dropout_grace = (
            training_active
            & safe
            & handoff_speed_ready
            & (transfer_rmse <= rmse_threshold)
        )
        patience = int(self.cfg.triad_hold_loss_patience_steps)
        next_hold_loss = torch.where(
            triad_eligible,
            torch.zeros_like(self.chain_phase_hold_loss),
            torch.where(
                contact_dropout_grace,
                self.chain_phase_hold_loss + 1,
                torch.full_like(
                    self.chain_phase_hold_loss, patience + 1
                ),
            ),
        )
        self.chain_phase_hold_loss = torch.where(
            training_active,
            next_hold_loss,
            self.chain_phase_hold_loss,
        )
        keep_during_grace = (
            contact_dropout_grace
            & (self.chain_phase_hold_loss <= patience)
        )
        self.chain_phase_hold = torch.where(
            training_active,
            torch.where(
                triad_eligible,
                self.chain_phase_hold + 1,
                torch.where(
                    keep_during_grace,
                    self.chain_phase_hold,
                    torch.zeros_like(self.chain_phase_hold),
                ),
            ),
            self.chain_phase_hold,
        )
        required_hold = torch.full_like(
            self.chain_phase_hold, int(self.cfg.triad_hold_steps)
        )
        if bool(self.cfg.sequential_closure):
            for stage_id in range(len(self.cfg.closure_target_blends)):
                stage_active = training_active & (
                    closure_level == stage_id
                )
                if torch.any(stage_active):
                    self.closure_stage_step_counts[stage_id] += (
                        stage_active.sum()
                    )
                    self.closure_stage_triad_step_counts[stage_id] += (
                        stage_active & transfer_contact
                    ).sum()
                    self.closure_stage_finger_contact_step_counts[
                        stage_id
                    ] += (
                        stage_active.unsqueeze(-1) & contacts
                    ).sum(dim=0)
                    transfer_components = torch.stack(
                        (
                            index_middle,
                            contacts[:, 0],
                            thumb_below_pen,
                            thumb_index,
                        ),
                        dim=-1,
                    )
                    self.closure_stage_transfer_component_step_counts[
                        stage_id
                    ] += (
                        stage_active.unsqueeze(-1) & transfer_components
                    ).sum(dim=0)
                    self.closure_stage_safe_step_counts[stage_id] += (
                        stage_active & safe
                    ).sum()
                    safe_components = torch.stack(
                        (
                            position_safe,
                            axis_safe,
                            speed_safe,
                            penetration_safe,
                            force_safe,
                        ),
                        dim=-1,
                    )
                    self.closure_stage_safe_component_step_counts[
                        stage_id
                    ] += (
                        stage_active.unsqueeze(-1) & safe_components
                    ).sum(dim=0)
                    stage_pose = transfer_rmse <= rmse_threshold
                    self.closure_stage_pose_step_counts[stage_id] += (
                        stage_active & stage_pose
                    ).sum()
                    self.closure_stage_contact_pose_step_counts[
                        stage_id
                    ] += (
                        stage_active & transfer_contact & stage_pose
                    ).sum()
                    self.closure_stage_eligible_step_counts[stage_id] += (
                        stage_active & triad_eligible
                    ).sum()
                    self.closure_stage_min_target_rmse[stage_id] = (
                        torch.minimum(
                            self.closure_stage_min_target_rmse[stage_id],
                            transfer_rmse[stage_active].min(),
                        )
                    )
                    self.closure_stage_min_canonical_rmse[stage_id] = (
                        torch.minimum(
                            self.closure_stage_min_canonical_rmse[stage_id],
                            canonical_rmse[stage_active].min(),
                        )
                    )
                    self.closure_stage_min_position_error[stage_id] = (
                        torch.minimum(
                            self.closure_stage_min_position_error[stage_id],
                            target_position_error[stage_active].min(),
                        )
                    )
                    self.closure_stage_min_axis_error[stage_id] = (
                        torch.minimum(
                            self.closure_stage_min_axis_error[stage_id],
                            target_axis_error[stage_active].min(),
                        )
                    )
                    self.closure_stage_best_hold[stage_id] = torch.maximum(
                        self.closure_stage_best_hold[stage_id],
                        self.chain_phase_hold[stage_active].max(),
                    )
        if bool(self.cfg.sequential_closure):
            required_hold = torch.where(
                phase == final_phase,
                torch.full_like(
                    required_hold, int(self.cfg.final_hold_steps)
                ),
                required_hold,
            )
        stage_complete = (
            training_active
            & (self.chain_phase_hold >= required_hold)
        )
        advance = (
            stage_complete
            & bool(self.cfg.sequential_closure)
            & (phase < final_phase)
        )
        success = stage_complete & (
            (phase == final_phase)
            if bool(self.cfg.sequential_closure)
            else torch.ones_like(phase, dtype=torch.bool)
        )
        final_active = (
            bool(self.cfg.sequential_closure) & (phase == final_phase)
        )
        self.episode_final_stage_entered |= final_active
        self.episode_final_triad_seen |= final_active & transfer_contact
        self.episode_final_eligible_seen |= final_active & triad_eligible
        self.episode_final_min_canonical_rmse = torch.where(
            final_active,
            torch.minimum(
                self.episode_final_min_canonical_rmse, canonical_rmse
            ),
            self.episode_final_min_canonical_rmse,
        )
        self.episode_final_max_hold = torch.where(
            final_active,
            torch.maximum(
                self.episode_final_max_hold, self.chain_phase_hold
            ),
            self.episode_final_max_hold,
        )
        self._advance_closure(advance, phase)
        self.episode_p5_success = (
            latched_chain_success & training_active
        ) | success
        if bool(self.cfg.thumb_wrap_transfer):
            brace_quality = torch.clamp(
                self.contact_force_norm[:, (1, 2)] / 0.15, 0.0, 1.0
            ).mean(dim=-1)
            pinch_quality = torch.clamp(
                self.contact_force_norm[:, :2] / 0.15, 0.0, 1.0
            ).mean(dim=-1)
            contact_quality = torch.where(
                transfer_stage <= 2, brace_quality, pinch_quality
            )
        else:
            contact_quality = (
                torch.clamp(
                    self.contact_force_norm[:, :3] / 0.15, 0.0, 1.0
                ).mean(dim=-1)
            )
        hold_progress = torch.clamp(
            self.chain_phase_hold.float()
            / required_hold.float().clamp_min(1.0),
            0.0,
            1.0,
        )
        reward = (
            8.0 * contact_quality
            + 16.0 * (transfer_contact & training_active).float()
            + 10.0 * torch.exp(-4.0 * transfer_rmse)
            + 8.0 * torch.exp(-5.0 * middle_rmse)
            + float(self.cfg.closure_hold_progress_reward_scale)
            * hold_progress
            + float(self.cfg.closure_success_reward_scale)
            * success.float()
            + 6.0 * safe.float()
            - 12.0 * (training_active & ~transfer_contact).float()
            - 0.03 * torch.linalg.norm(self.actions, dim=-1)
            - 15.0 * (~safe).float()
        )
        if float(self.cfg.closure_final_gate_min_reward_scale) > 0.0:
            position_quality = torch.exp(
                -0.5
                * (
                    target_position_error
                    / max(float(self.cfg.closure_pen_position_tolerance_m), 1e-6)
                ).square()
            )
            axis_tolerance = torch.deg2rad(
                torch.full_like(
                    target_axis_error,
                    max(float(self.cfg.closure_pen_axis_tolerance_deg), 1e-3),
                )
            )
            axis_quality = torch.exp(
                -0.5 * (target_axis_error / axis_tolerance).square()
            )
            pose_quality = torch.exp(
                -0.5
                * (
                    transfer_rmse / rmse_threshold.clamp_min(1e-4)
                ).square()
            )
            angular_limit = max(
                float(self.cfg.thumb_wrap_handoff_max_angular_speed_rad_s),
                1e-3,
            )
            linear_limit = max(
                float(self.cfg.thumb_wrap_handoff_max_linear_speed_m_s),
                1e-4,
            )
            angular_quality = torch.exp(
                -0.5 * (speed / angular_limit).square()
            )
            linear_quality = torch.exp(
                -0.5 * (linear_speed / linear_limit).square()
            )
            final_gate_components = torch.stack(
                (
                    contact_quality,
                    position_quality,
                    axis_quality,
                    pose_quality,
                    angular_quality,
                    linear_quality,
                ),
                dim=-1,
            )
            final_gate_quality = final_gate_components.amin(dim=-1)
            reward = reward + (
                float(self.cfg.closure_final_gate_min_reward_scale)
                * final_active.float()
                * final_gate_quality
            )
            if (
                float(self.cfg.closure_final_gate_geomean_reward_scale)
                > 0.0
            ):
                final_gate_geomean = torch.exp(
                    torch.log(final_gate_components.clamp_min(1.0e-6))
                    .mean(dim=-1)
                )
                reward = reward + (
                    float(
                        self.cfg.closure_final_gate_geomean_reward_scale
                    )
                    * final_active.float()
                    * final_gate_geomean
                )
        if bool(self.cfg.thumb_wrap_transfer):
            before_takeover = training_active & (transfer_stage <= 3)
            brace_active = training_active & (transfer_stage == 0)
            under_active = training_active & (transfer_stage == 1)
            wrap_active = training_active & (transfer_stage == 2)
            takeover_active = training_active & (transfer_stage == 3)
            unload_active = training_active & (transfer_stage == 4)
            opposite_active = training_active & (
                closure_level
                >= int(self.cfg.thumb_wrap_opposite_gate_start_level)
            )
            opposite_quality = (
                ((1.0 - thumb_index_opposite_cos) * 0.5)
                * torch.exp(
                    -thumb_index_axial_separation
                    / max(
                        float(
                            self.cfg.thumb_wrap_max_axial_separation_m
                        ),
                        1.0e-6,
                    )
                )
            )
            middle_brace_active = training_active & (
                closure_level <= int(
                    self.cfg.thumb_wrap_stage_boundaries[0]
                )
            )
            middle_brace_score = torch.clamp(
                self.contact_force_norm[:, 2] / 0.15, 0.0, 1.0
            )
            reward = (
                reward
                + 18.0
                * middle_brace_active.float()
                * middle_brace_score
                + 12.0 * (brace_active & index_middle).float()
                + 10.0 * (under_active & thumb_below_pen).float()
                + 10.0
                * (
                    under_active
                    & index_middle
                    & contacts[:, 0]
                    & thumb_below_pen
                ).float()
                + 10.0
                * (
                    wrap_active
                    & index_middle
                    & contacts[:, 0]
                ).float()
                + 8.0
                * (
                    training_active
                    & (transfer_stage >= 1)
                    & contacts[:, 0]
                ).float()
                + 10.0
                * (
                    takeover_active
                    & thumb_index
                    & contacts[:, 2]
                ).float()
                + float(self.cfg.thumb_wrap_opposite_reward_scale)
                * opposite_active.float()
                * opposite_quality
                - 18.0 * (before_takeover & ~contacts[:, 2]).float()
                + float(
                    self.cfg.thumb_wrap_handoff_speed_reward_scale
                )
                * training_active.float()
                * torch.exp(-0.45 * speed)
                * torch.exp(-25.0 * linear_speed)
                - float(
                    self.cfg.thumb_wrap_pinky_contact_penalty_scale
                )
                * (training_active & ~pinky_clear).float()
                + float(
                    self.cfg.thumb_wrap_unload_pinch_reward_scale
                )
                * (unload_active & thumb_index).float()
                - float(
                    self.cfg.thumb_wrap_unload_thumb_loss_penalty_scale
                )
                * (unload_active & ~contacts[:, 0]).float()
            )
        joint_pose_score = torch.zeros_like(target_position_error)
        joint_safe = torch.zeros_like(position_safe)
        if bool(self.cfg.closure_pen_pose_gate):
            closure_progress = closure_level.float() / max(
                len(self.cfg.closure_target_blends) - 1, 1
            )
            origin_axis = quat_apply(
                self.closure_origin_pen_quat, self.pen_x_axis
            )
            aligned_pen_axis = torch.where(
                (
                    torch.sum(origin_axis * self.pen_axis, dim=-1) < 0.0
                ).unsqueeze(-1),
                -self.pen_axis,
                self.pen_axis,
            )
            aligned_p1_axis = torch.where(
                (
                    torch.sum(
                        origin_axis
                        * self.p1_pen_axis.unsqueeze(0),
                        dim=-1,
                    )
                    < 0.0
                ).unsqueeze(-1),
                -self.p1_pen_axis.unsqueeze(0),
                self.p1_pen_axis.unsqueeze(0),
            )
            path_normal = torch.linalg.cross(
                origin_axis, aligned_p1_axis, dim=-1
            )
            path_normal_norm = torch.linalg.norm(
                path_normal, dim=-1, keepdim=True
            )
            path_normal = path_normal / path_normal_norm.clamp_min(
                1.0e-8
            )
            signed_angle = torch.atan2(
                torch.sum(
                    torch.linalg.cross(
                        origin_axis, aligned_pen_axis, dim=-1
                    )
                    * path_normal,
                    dim=-1,
                ),
                torch.sum(
                    origin_axis * aligned_pen_axis, dim=-1
                ).clamp(-1.0, 1.0),
            )
            total_angle = torch.acos(
                torch.sum(
                    origin_axis * aligned_p1_axis, dim=-1
                ).clamp(-1.0, 1.0)
            )
            signed_pen_progress = (
                signed_angle / total_angle.clamp_min(1.0e-6)
            ).clamp(-0.25, 1.25)
            progress_values = torch.as_tensor(
                self.cfg.closure_pen_target_progress,
                dtype=torch.float32,
                device=self.device,
            )
            desired_pen_progress = progress_values[closure_level]
            rewarded_pen_progress = signed_pen_progress
            if bool(self.cfg.closure_cap_signed_progress_at_target):
                rewarded_pen_progress = torch.minimum(
                    rewarded_pen_progress, desired_pen_progress
                )
            progress_tracking_score = torch.exp(
                -8.0
                * torch.abs(signed_pen_progress - desired_pen_progress)
            )
            position_tolerance = float(
                self.cfg.closure_pen_position_tolerance_m
            )
            axis_tolerance = torch.deg2rad(
                torch.full_like(
                    target_axis_error,
                    float(self.cfg.closure_pen_axis_tolerance_deg),
                )
            )
            position_score = torch.exp(
                -0.5
                * (
                    target_position_error
                    / max(position_tolerance, 1.0e-6)
                ).square()
            )
            axis_score = torch.exp(
                -0.5
                * (
                    target_axis_error
                    / axis_tolerance.clamp_min(1.0e-6)
                ).square()
            )
            angular_speed_score = torch.exp(-0.25 * speed)
            linear_speed_score = torch.exp(-20.0 * linear_speed)
            joint_pose_score = (
                torch.minimum(position_score, axis_score)
                * angular_speed_score
                * linear_speed_score
                * transfer_contact.float()
            )
            joint_safe = (
                position_safe
                & axis_safe
                & speed_safe
                & transfer_contact
                & (transfer_rmse <= rmse_threshold)
            )
            reward = (
                reward
                + float(self.cfg.closure_pen_position_reward_scale)
                * torch.exp(-120.0 * target_position_error)
                + float(self.cfg.closure_pen_axis_reward_scale)
                * torch.exp(-5.0 * target_axis_error)
                + float(self.cfg.closure_stage_progress_reward_scale)
                * closure_progress
                + float(self.cfg.closure_stage_advance_reward_scale)
                * advance.float()
                + float(
                    self.cfg.closure_pen_signed_progress_reward_scale
                )
                * rewarded_pen_progress
                + float(
                    self.cfg.closure_pen_progress_tracking_reward_scale
                )
                * progress_tracking_score
                - float(
                    self.cfg.closure_pen_position_error_penalty_scale
                )
                * target_position_error
                - float(self.cfg.closure_pen_axis_error_penalty_scale)
                * target_axis_error
                - float(
                    self.cfg.closure_pen_angular_speed_penalty_scale
                )
                * speed
                - float(
                    self.cfg.closure_pen_axis_rate_penalty_scale
                )
                * axis_rate
                - float(
                    self.cfg.closure_pen_linear_speed_penalty_scale
                )
                * linear_speed
                + float(self.cfg.closure_joint_pose_reward_scale)
                * joint_pose_score
                + float(self.cfg.closure_joint_safe_reward_scale)
                * joint_safe.float()
                - float(self.cfg.closure_action_penalty_scale)
                * torch.linalg.norm(self.actions, dim=-1)
            )
        else:
            signed_pen_progress = torch.zeros_like(target_axis_error)
        reward = torch.where(
            training_active, reward, torch.zeros_like(reward)
        )
        if bool(self.cfg.residual_reward_only):
            learning_active = (
                (
                    phase >= int(self.cfg.residual_start_phase)
                )
                & (
                    phase <= int(self.cfg.residual_end_phase)
                )
            )
            manual_learning_active = (
                (
                    manual_phase_before_handoff
                    >= int(self.cfg.residual_start_phase)
                )
                & (
                    manual_phase_before_handoff
                    <= int(self.cfg.residual_end_phase)
                )
                & (
                    (manual_phase_before_handoff == 2)
                    | (manual_phase_before_handoff == 3)
                )
            )
            middle_gap_target = torch.where(
                manual_phase_before_handoff == 2,
                torch.full_like(middle_gap, 0.0020),
                torch.zeros_like(middle_gap),
            )
            middle_geometry_score = torch.exp(
                -500.0 * torch.abs(middle_gap - middle_gap_target)
            )
            middle_force_quality = torch.clamp(
                self.contact_force_norm[:, 2] / 0.15, 0.0, 1.0
            )
            pinch_quality = torch.clamp(
                self.contact_force_norm[:, :2] / 0.15, 0.0, 1.0
            ).mean(dim=-1)
            approach_phase = manual_phase_before_handoff == 2
            contact_phase = manual_phase_before_handoff == 3
            manual_reward = (
                10.0 * middle_geometry_score
                + 8.0 * pinch_quality
                + 12.0
                * contact_phase.float()
                * middle_force_quality
                + 14.0
                * contact_phase.float()
                * triad_contact.float()
                + 6.0 * safe.float()
                + 8.0 * manual_hold_progress
                + 30.0
                * (
                    transition
                    & (
                        approach_phase | contact_phase
                    )
                ).float()
                - 10.0
                * contact_phase.float()
                * (~triad_contact).float()
                - 0.03 * torch.linalg.norm(self.actions, dim=-1)
                - 15.0 * (~safe).float()
            )
            reward = torch.where(
                manual_learning_active,
                manual_reward,
                torch.where(
                    learning_active, reward, torch.zeros_like(reward)
                ),
            )
        active_count = training_active.sum().clamp_min(1)
        self.extras["log"] = {
            "metric/chain_phase_mean": phase.float().mean(),
            "metric/triad_rmse_rad": (
                transfer_rmse * training_active
            ).sum() / active_count,
            "metric/ring_rmse_rad": (
                ring_rmse * training_active
            ).sum() / active_count,
            "metric/middle_rmse_rad": (
                middle_rmse * training_active
            ).sum() / active_count,
            "metric/canonical_p1_rmse_rad": (
                canonical_rmse * training_active
            ).sum() / active_count,
            "metric/triad_contact_ratio": (
                transfer_contact & training_active
            ).sum().float() / active_count,
            "metric/triad_hold_ms": (
                self.chain_phase_hold.float().mean()
                * self.cfg.decimation
                * self.cfg.sim.dt
                * 1000.0
            ),
            "metric/training_active_ratio": training_active.float().mean(),
            "metric/success_ratio": success.float().mean(),
            "metric/closure_pen_position_error_mm": (
                target_position_error * training_active
            ).sum()
            * 1000.0
            / active_count,
            "metric/closure_pen_axis_error_deg": (
                target_axis_error * training_active
            ).sum()
            * (180.0 / torch.pi)
            / active_count,
            "metric/closure_pen_signed_progress": (
                signed_pen_progress * training_active
            ).sum()
            / active_count,
            "metric/closure_joint_pose_score": (
                joint_pose_score * training_active
            ).sum()
            / active_count,
            "metric/closure_joint_safe_ratio": (
                joint_safe & training_active
            ).sum().float()
            / active_count,
        }
        if bool(self.cfg.thumb_wrap_transfer):
            self.extras["log"].update(
                {
                    "metric/thumb_wrap_stage_mean": (
                        transfer_stage.float() * training_active
                    ).sum()
                    / active_count,
                    "metric/index_middle_brace_ratio": (
                        index_middle & training_active
                    ).sum().float()
                    / active_count,
                    "metric/middle_surface_gap_mm": (
                        middle_gap * training_active
                    ).sum()
                    * 1000.0
                    / active_count,
                    "metric/thumb_below_ratio": (
                        thumb_below_pen & training_active
                    ).sum().float()
                    / active_count,
                    "metric/thumb_index_pinch_ratio": (
                        thumb_index & training_active
                    ).sum().float()
                    / active_count,
                    "metric/thumb_index_opposite_ratio": (
                        thumb_index_opposite & training_active
                    ).sum().float()
                    / active_count,
                    "metric/thumb_index_opposite_cos": (
                        thumb_index_opposite_cos * training_active
                    ).sum()
                    / active_count,
                    "metric/thumb_index_axial_separation_mm": (
                        thumb_index_axial_separation * training_active
                    ).sum()
                    * 1000.0
                    / active_count,
                    "metric/handoff_angular_speed_rad_s": (
                        speed * training_active
                    ).sum()
                    / active_count,
                    "metric/handoff_axis_rate_rad_s": (
                        axis_rate * training_active
                    ).sum()
                    / active_count,
                    "metric/handoff_linear_speed_mm_s": (
                        linear_speed * training_active
                    ).sum()
                    * 1000.0
                    / active_count,
                    "metric/handoff_speed_ready_ratio": (
                        handoff_speed_ready & training_active
                    ).sum().float()
                    / active_count,
                    "metric/pinky_clear_ratio": (
                        pinky_clear & training_active
                    ).sum().float()
                    / active_count,
                }
            )
        approach_active = manual_phase_before_handoff == 2
        approach_count = approach_active.sum().clamp_min(1)
        self.extras["log"].update(
            {
                "metric/approach_active_ratio": approach_active.float().mean(),
                "metric/approach_near_ratio": (
                    approach_active & middle_near
                ).sum().float()
                / approach_count,
                "metric/approach_safe_ratio": (
                    approach_active & safe
                ).sum().float()
                / approach_count,
                "metric/approach_pinch_ratio": (
                    approach_active & thumb_index
                ).sum().float()
                / approach_count,
                "metric/approach_handoff_ratio": (
                    transition & approach_active
                ).sum().float()
                / approach_count,
                "metric/approach_middle_gap_mm": (
                    middle_gap * approach_active
                ).sum()
                * 1000.0
                / approach_count,
            }
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        failed, time_out = super()._get_dones()
        phase = self.chain_phase_ids.clamp(
            0, self.chain_drop_by_phase.numel() - 1
        )
        height_drop = self.init_pen_pos[:, 2] - self.pen_pos[:, 2]
        position_drift = torch.linalg.norm(
            self.pen_pos - self.init_pen_pos, dim=-1
        )
        drop = height_drop > self.cfg.drop_threshold
        drift = position_drift > self.cfg.drift_threshold
        penetration = (
            self.episode_max_penetration
            > self.cfg.penetration_threshold
        )
        for phase_id in range(self.chain_drop_by_phase.numel()):
            phase_active = phase == phase_id
            self.chain_drop_by_phase[phase_id] += (
                phase_active & drop
            ).sum()
            self.chain_drift_by_phase[phase_id] += (
                phase_active & drift
            ).sum()
            self.chain_penetration_by_phase[phase_id] += (
                phase_active & penetration
            ).sum()
        success = self.episode_p5_success
        if torch.any(success):
            self.latest_success_serial[success] += 1
            self.latest_success_qpos[success] = torch.cat(
                (self.hand_qpos, self.pen_pos, self.pen_quat), dim=-1
            )[success]
            self.latest_success_qvel[success] = torch.cat(
                (
                    self.hand_qvel,
                    self.pen_linvel,
                    self.pen_angvel,
                ),
                dim=-1,
            )[success]
            self.latest_success_penetration[success] = (
                self.episode_max_penetration[success]
            )
            self.latest_success_contact_force[success] = (
                self.contact_force_norm[success]
            )
        return failed | success, time_out & ~success


class Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnv(
    Dex5V9P4P5P1ContinuousTriadTrainPhysicsEnv
):
    cfg: Dex5V9P4P5P1ContinuousClosureCycleAlignedLateTrainPhysicsEnvCfg

    def _load_bundle(self) -> None:
        super()._load_bundle()
        with np.load(self.cfg.bundle_path) as data:
            self.late_target_hand_bank = torch.as_tensor(
                np.asarray(
                    data["source_target_hand_qpos"], dtype=np.float32
                ),
                device=self.device,
            )
            self.late_closure_origin_bank = torch.as_tensor(
                np.asarray(
                    data["source_closure_origin_hand"], dtype=np.float32
                ),
                device=self.device,
            )
            if "source_closure_brace_hand" in data:
                self.late_closure_brace_bank = torch.as_tensor(
                    np.asarray(
                        data["source_closure_brace_hand"],
                        dtype=np.float32,
                    ),
                    device=self.device,
                )
            if "source_closure_origin_pen_pos" in data:
                self.late_closure_origin_pen_pos_bank = torch.as_tensor(
                    np.asarray(
                        data["source_closure_origin_pen_pos"],
                        dtype=np.float32,
                    ),
                    device=self.device,
                )
                self.late_closure_origin_pen_quat_bank = torch.as_tensor(
                    np.asarray(
                        data["source_closure_origin_pen_quat"],
                        dtype=np.float32,
                    ),
                    device=self.device,
                )
            self.late_init_pen_pos_bank = torch.as_tensor(
                np.asarray(data["source_init_pen_pos"], dtype=np.float32),
                device=self.device,
            )
            self.late_init_pen_axis_bank = torch.as_tensor(
                np.asarray(data["source_init_pen_axis"], dtype=np.float32),
                device=self.device,
            )
            self.late_max_penetration_bank = torch.as_tensor(
                np.asarray(
                    data["source_max_penetration_m"], dtype=np.float32
                ),
                device=self.device,
            )
            self.late_chain_phase_bank = torch.as_tensor(
                np.asarray(
                    data["source_chain_phase_id"], dtype=np.int64
                ),
                dtype=torch.long,
                device=self.device,
            )
            if "best_hold_steps" in data:
                self.late_phase_hold_bank = torch.as_tensor(
                    np.asarray(data["best_hold_steps"], dtype=np.int64),
                    dtype=torch.long,
                    device=self.device,
                )
            if "source_policy_action" in data:
                self.late_previous_action_bank = torch.as_tensor(
                    np.asarray(
                        data["source_policy_action"], dtype=np.float32
                    ),
                    device=self.device,
                )
        teacher_path = str(self.cfg.scripted_thumb_teacher_path)
        if teacher_path:
            with np.load(teacher_path) as teacher:
                if "selection_kind" in teacher:
                    kinds = np.asarray(teacher["selection_kind"]).astype(str)
                    selected = kinds == "success"
                    if bool(
                        self.cfg.scripted_thumb_teacher_accept_near
                    ):
                        selected |= np.isin(
                            kinds, ("near_hold", "braking_near")
                        )
                else:
                    selected = np.ones(
                        len(teacher["candidate_action_first"]),
                        dtype=bool,
                    )
                if not np.any(selected):
                    raise ValueError(
                        "Scripted thumb teacher has no successful rows: "
                        f"{teacher_path}"
                    )
                first = np.asarray(
                    teacher["candidate_action_first"][selected],
                    dtype=np.float32,
                )
                second = np.asarray(
                    teacher["candidate_action_second"][selected],
                    dtype=np.float32,
                )
                first_duration = np.asarray(
                    teacher["first_duration"][selected], dtype=np.int64
                )
                second_duration = np.asarray(
                    teacher["second_duration"][selected], dtype=np.int64
                )
                if (
                    bool(self.cfg.scripted_thumb_teacher_pair_with_bundle)
                    and "source_bank_id" in teacher
                ):
                    bank_count = len(self.late_chain_phase_bank)
                    paired_first = np.zeros(
                        (bank_count, first.shape[1]), dtype=np.float32
                    )
                    paired_second = np.zeros_like(paired_first)
                    paired_first_duration = np.zeros(
                        bank_count, dtype=np.int64
                    )
                    paired_second_duration = np.zeros_like(
                        paired_first_duration
                    )
                    bank_ids = np.asarray(
                        teacher["source_bank_id"][selected], dtype=np.int64
                    )
                    score = (
                        np.asarray(
                            teacher["best_brake_score"][selected],
                            dtype=np.float32,
                        )
                        if "best_brake_score" in teacher
                        else np.zeros(len(bank_ids), dtype=np.float32)
                    )
                    for bank_id in np.unique(bank_ids):
                        rows = np.flatnonzero(bank_ids == bank_id)
                        best = rows[np.argmax(score[rows])]
                        paired_first[bank_id] = first[best]
                        paired_second[bank_id] = second[best]
                        paired_first_duration[bank_id] = first_duration[best]
                        paired_second_duration[bank_id] = second_duration[best]
                    first = paired_first
                    second = paired_second
                    first_duration = paired_first_duration
                    second_duration = paired_second_duration
                self.scripted_thumb_teacher_first = torch.as_tensor(
                    first, device=self.device
                )
                self.scripted_thumb_teacher_second = torch.as_tensor(
                    second, device=self.device
                )
                self.scripted_thumb_teacher_first_duration = torch.as_tensor(
                    first_duration, dtype=torch.long, device=self.device
                )
                self.scripted_thumb_teacher_second_duration = torch.as_tensor(
                    second_duration, dtype=torch.long, device=self.device
                )
            self.scripted_thumb_teacher_hold_latched = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "late_chain_phase_bank"):
            return
        if env_ids is None:
            env_ids_tensor = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        bank_ids = self.current_bank_ids[env_ids_tensor]
        phase = self.late_chain_phase_bank[bank_ids]
        self.chain_phase_ids[env_ids_tensor] = phase
        self.p5_stage_ids[env_ids_tensor] = (phase - 1).clamp(2, 4)
        self.target_hand_qpos[env_ids_tensor] = (
            self.late_target_hand_bank[bank_ids]
        )
        self.closure_origin_hand[env_ids_tensor] = (
            self.late_closure_origin_bank[bank_ids]
        )
        if hasattr(self, "late_closure_brace_bank"):
            self.closure_brace_hand[env_ids_tensor] = (
                self.late_closure_brace_bank[bank_ids]
            )
        if bool(self.cfg.closure_return_middle_instead_of_ring):
            level = (
                self.chain_phase_ids
                - int(self.cfg.training_start_phase)
            ).clamp(0, len(self.cfg.closure_target_blends) - 1)
            blend_values = torch.as_tensor(
                self.cfg.closure_target_blends,
                dtype=torch.float32,
                device=self.device,
            )
            target = self._closure_target(
                blend_values[level].unsqueeze(-1)
            )
            self.target_hand_qpos[env_ids_tensor] = target[
                env_ids_tensor
            ]
        if hasattr(self, "late_closure_origin_pen_pos_bank"):
            self.closure_origin_pen_pos[env_ids_tensor] = (
                self.late_closure_origin_pen_pos_bank[bank_ids]
            )
            self.closure_origin_pen_quat[env_ids_tensor] = (
                self.late_closure_origin_pen_quat_bank[bank_ids]
            )
        self.init_pen_pos[env_ids_tensor] = (
            self.late_init_pen_pos_bank[bank_ids]
        )
        self.init_pen_axis[env_ids_tensor] = (
            self.late_init_pen_axis_bank[bank_ids]
        )
        self.episode_max_penetration[env_ids_tensor] = (
            self.late_max_penetration_bank[bank_ids]
        )
        if bool(self.cfg.closure_pen_pose_gate):
            level = (
                self.chain_phase_ids
                - int(self.cfg.training_start_phase)
            ).clamp(0, len(self.cfg.closure_pen_target_progress) - 1)
            progress_values = torch.as_tensor(
                self.cfg.closure_pen_target_progress,
                dtype=torch.float32,
                device=self.device,
            )
            progress = progress_values[level].unsqueeze(-1)
            position_progress = None
            if len(
                self.cfg.closure_pen_position_target_progress
            ) > 0:
                position_values = torch.as_tensor(
                    self.cfg.closure_pen_position_target_progress,
                    dtype=torch.float32,
                    device=self.device,
                )
                position_progress = position_values[
                    level
                ].unsqueeze(-1)
            pen_target_pos, pen_target_quat = self._closure_pen_target(
                progress, position_progress
            )
            self.target_pen_pos[env_ids_tensor] = pen_target_pos[
                env_ids_tensor
            ]
            self.target_pen_quat[env_ids_tensor] = pen_target_quat[
                env_ids_tensor
            ]
        if (
            bool(self.cfg.restore_cached_phase_hold)
            and hasattr(self, "late_phase_hold_bank")
        ):
            self.chain_phase_hold[env_ids_tensor] = (
                self.late_phase_hold_bank[bank_ids]
            )
        else:
            self.chain_phase_hold[env_ids_tensor] = 0
        self.chain_phase_hold_loss[env_ids_tensor] = 0
        if hasattr(self, "late_previous_action_bank"):
            self.chain_previous_action[env_ids_tensor] = (
                self.late_previous_action_bank[bank_ids]
            )
        else:
            self.chain_previous_action[env_ids_tensor] = self.actions[
                env_ids_tensor
            ]
        self.chain_blend_age[env_ids_tensor] = 0
        if hasattr(self, "scripted_thumb_teacher_hold_latched"):
            self.scripted_thumb_teacher_hold_latched[env_ids_tensor] = False


class Dex5V9P5P1PhysicsEnv(Dex5V9P4P5PhysicsEnv):
    cfg: Dex5V9P5P1PhysicsEnvCfg

    def _stage_action_mask(self) -> torch.Tensor:
        stage = self.p5_stage_ids
        mask = torch.zeros((self.num_envs, 20), device=self.device)
        hold = stage == 0
        approach = stage == 1
        contact = stage == 2
        triad = stage == 3
        strict = stage == 4
        support_scale = (
            0.30 * (hold | approach | contact).float()
            + 1.00 * (triad | strict).float()
        )
        mask[:, :4] = support_scale.unsqueeze(-1)
        mask[:, 16:20] = support_scale.unsqueeze(-1)
        mask[:, 4:8] = (
            approach | contact | triad | strict
        ).float().unsqueeze(-1)
        mask[:, 8:12] = (triad | strict).float().unsqueeze(-1)
        return mask

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if bool(self.cfg.terminate_on_success):
            actions = torch.where(
                self.episode_p5_success.unsqueeze(-1),
                torch.zeros_like(actions),
                actions,
            )
        super()._pre_physics_step(actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        failed, time_out = super()._get_dones()
        if not bool(self.cfg.terminate_on_success):
            return failed, time_out
        success = self.episode_p5_success
        return failed | success, time_out & ~success

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        failed, _ = self._compute_dones()
        contacts = self.contact_force_norm > self.cfg.p1_contact_force_n
        thumb_index = contacts[:, :2].all(dim=-1)
        middle_surface_gap = (
            self.tip_surface_dist[:, 2] - self.fingertip_radii[:, 2]
        )
        middle_geometry = (
            middle_surface_gap
            <= self.cfg.p1_middle_contact_distance_m
        )
        middle_contact = contacts[:, 2] & middle_geometry
        triad_contact = thumb_index & middle_contact
        middle_near = (
            (
                middle_surface_gap
                >= self.cfg.p1_middle_approach_min_gap_m
            )
            & (
                middle_surface_gap
                <= self.cfg.p1_middle_near_distance_m
            )
        )
        triad_rmse = torch.sqrt(
            torch.mean(
                torch.cat(
                    (
                        self.hand_qpos[:, 0:8]
                        - self.target_hand_qpos[:, 0:8],
                        self.hand_qpos[:, 16:20]
                        - self.target_hand_qpos[:, 16:20],
                    ),
                    dim=-1,
                ).square(),
                dim=-1,
            )
        )
        ring_rmse = torch.sqrt(
            torch.mean(
                (
                    self.hand_qpos[:, 8:12]
                    - self.target_hand_qpos[:, 8:12]
                ).square(),
                dim=-1,
            )
        )
        middle_rmse = torch.sqrt(
            torch.mean(
                (
                    self.hand_qpos[:, 4:8]
                    - self.target_hand_qpos[:, 4:8]
                ).square(),
                dim=-1,
            )
        )
        position_drift = torch.linalg.norm(
            self.pen_pos - self.init_pen_pos, dim=-1
        )
        axis_dot = torch.sum(
            self.pen_axis * self.init_pen_axis, dim=-1
        ).clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot.abs())
        angular_speed = torch.linalg.norm(self.pen_angvel, dim=-1)
        peak_force = self.contact_force_norm.max(dim=-1).values
        safe_pen = (
            (position_drift <= self.cfg.p5_max_position_drift_m)
            & (
                axis_drift
                <= torch.deg2rad(
                    torch.full_like(
                        axis_drift, self.cfg.p5_max_axis_drift_deg
                    )
                )
            )
            & (angular_speed <= self.cfg.p5_max_angular_speed_rad_s)
            & (
                self.episode_max_penetration
                <= self.cfg.p5_max_penetration_m
            )
            & (peak_force <= self.cfg.p1_max_force_n)
            & ~failed
        )
        stage = self.p5_stage_ids
        stage_eligible = torch.where(
            stage == 0,
            safe_pen & thumb_index,
            torch.where(
                stage == 1,
                safe_pen & thumb_index & middle_near,
                torch.where(
                    stage == 2,
                    safe_pen & triad_contact,
                    torch.where(
                        stage == 3,
                        safe_pen
                        & triad_contact
                        & (triad_rmse <= self.cfg.p1_triad_pose_rmse_rad),
                        safe_pen
                        & triad_contact
                        & (triad_rmse <= self.cfg.p1_pose_rmse_rad)
                        & (ring_rmse <= self.cfg.p1_ring_pose_rmse_rad),
                    ),
                ),
            ),
        )
        self.stage_hold_count = torch.where(
            stage_eligible,
            self.stage_hold_count + 1,
            torch.zeros_like(self.stage_hold_count),
        )
        self.p5_hold_count = self.stage_hold_count
        success = self.stage_hold_count >= self.cfg.p5_hold_steps
        self.episode_p5_success |= success

        thumb_quality = torch.clamp(
            self.contact_force_norm[:, 0] / 0.15, 0.0, 1.0
        )
        index_quality = torch.clamp(
            self.contact_force_norm[:, 1] / 0.15, 0.0, 1.0
        )
        middle_quality = torch.clamp(
            self.contact_force_norm[:, 2] / 0.15, 0.0, 1.0
        )
        near_score = torch.exp(
            -220.0
            * torch.relu(
                middle_surface_gap
                - self.cfg.p1_middle_contact_distance_m
            )
        )
        middle_pose_score = torch.exp(-5.0 * middle_rmse)
        triad_pose_score = torch.exp(-5.0 * triad_rmse)
        position_score = torch.exp(-100.0 * position_drift)
        axis_score = torch.exp(-8.0 * axis_drift)
        speed_score = torch.exp(-0.18 * angular_speed)
        hold_progress = torch.clamp(
            self.stage_hold_count.float() / self.cfg.p5_hold_steps,
            0.0,
            1.0,
        )
        middle_stage = (stage >= 1).float()
        contact_stage = (stage >= 2).float()
        pose_stage = (stage >= 3).float()
        strict_stage = (stage == 4).float()
        strict_pose_margin = torch.sigmoid(
            (
                self.cfg.p1_pose_rmse_rad
                - triad_rmse
            )
            / 0.015
        )
        terms = {
            "alive": torch.full_like(position_drift, 0.05),
            "thumb_contact": 5.0 * thumb_quality,
            "index_contact": 5.0 * index_quality,
            "pinch": 7.0 * thumb_quality * index_quality,
            "middle_near": 12.0 * middle_stage * near_score,
            "middle_pose": 5.0 * middle_stage * middle_pose_score,
            "middle_contact": 12.0
            * contact_stage
            * middle_contact.float(),
            "triad_contact": 15.0
            * contact_stage
            * triad_contact.float(),
            "middle_load": 4.0
            * contact_stage
            * middle_quality
            * thumb_quality
            * index_quality,
            "triad_pose": 10.0 * pose_stage * triad_pose_score,
            "strict_pose_margin": 24.0
            * strict_stage
            * triad_contact.float()
            * strict_pose_margin,
            "strict_contact_loss": -12.0
            * strict_stage
            * (~triad_contact).float(),
            "pen_position": 4.0 * position_score,
            "pen_axis": 5.0 * axis_score,
            "angular_brake": 4.0 * speed_score,
            "hold": 10.0 * hold_progress,
            "terminal_success": 50.0 * success.float(),
            "penetration": -16.0
            * torch.relu(
                (self.episode_max_penetration - 0.00035) / 0.00025
            ),
            "force_excess": -3.0
            * torch.relu((peak_force - 3.0) / 3.0),
            "action_l2": -0.03
            * torch.linalg.norm(self.actions, dim=-1),
            "action_smoothness": -0.05
            * torch.linalg.norm(
                self.actions - self.previous_actions, dim=-1
            ),
            "failure": -12.0 * failed.float(),
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
        self.extras["log"] = {
            f"reward/{name}": value.mean() for name, value in terms.items()
        }
        self.extras["log"].update(
            {
                "metric/p5_success_ratio": success.float().mean(),
                "metric/p5_instantaneous_ratio": stage_eligible.float().mean(),
                "metric/p5_partial_ratio": triad_contact.float().mean(),
                "metric/stage_eligible_ratio": stage_eligible.float().mean(),
                "metric/stage_hold_ms": self.stage_hold_count.float().mean()
                * self.cfg.decimation
                * self.cfg.sim.dt
                * 1000.0,
                "metric/p5_hold_ms": self.stage_hold_count.float().mean()
                * self.cfg.decimation
                * self.cfg.sim.dt
                * 1000.0,
                "metric/middle_clear_ratio": middle_near.float().mean(),
                "metric/ring_clear_ratio": triad_contact.float().mean(),
                "metric/middle_ring_clear_ratio": triad_contact.float().mean(),
                "metric/middle_rmse_rad": middle_rmse.mean(),
                "metric/ring_rmse_rad": ring_rmse.mean(),
                "metric/retract_rmse_rad": ring_rmse.mean(),
                "metric/pinch_rmse_rad": torch.zeros_like(
                    triad_rmse.mean()
                ),
                "metric/pen_drift_mm": position_drift.mean() * 1000.0,
                "metric/axis_drift_deg": axis_drift.mean()
                * (180.0 / torch.pi),
                "metric/angular_speed_rad_s": angular_speed.mean(),
                "metric/p1_middle_near_ratio": middle_near.float().mean(),
                "metric/p1_middle_contact_ratio": middle_contact.float().mean(),
                "metric/p1_triad_contact_ratio": triad_contact.float().mean(),
                "metric/p1_triad_rmse_rad": triad_rmse.mean(),
                "metric/p1_middle_surface_gap_mm": middle_surface_gap.mean()
                * 1000.0,
            }
        )
        return reward
