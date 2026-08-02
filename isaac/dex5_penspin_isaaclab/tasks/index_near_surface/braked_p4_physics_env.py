from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from .recapture_p3_physics_env import Dex5P3EntryPhysicsEnvCfg, Dex5RecaptureP3PhysicsEnv


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_BRAKED_P4_BUNDLE = str(
    ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_braked_p4_bundle.npz"
)
STAGE0_BRAKED_P4_BUNDLE = str(
    ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_braked_p4_stage0_bundle.npz"
)
STAGE1_BRAKED_P4_BUNDLE = str(
    ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_braked_p4_stage1_bundle.npz"
)
BRIDGE22_TEACHER_CACHE = str(
    ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_bridge22_teacher_cache.npz"
)
BRIDGE22_THUMB_PRESERVE_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_bridge22_thumb_preserve_teacher_cache.npz"
)
BRIDGE22_DELAYED_CONTACT_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_bridge22_delayed3_contact_bundle.npz"
)
BRIDGE22_DELAYED_CONTACT_HOLD_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_bridge22_delayed3_contact_hold_bundle.npz"
)
BRIDGE22_DELAYED_CONTACT_STATIC_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_bridge22_delayed3_contact_static_bundle.npz"
)
BRIDGE22_DELAYED_CONTACT_PROCESSED_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_bridge22_delayed3_contact_processed_bundle.npz"
)
CANONICAL_P3_P4_APPROACH20_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_p3_to_p4_approach20_bundle.npz"
)
CANONICAL_P3_P4_ENTRY7_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_p3_to_p4_entry7_bundle.npz"
)
CANONICAL_P3_ENTRY7_P4_APPROACH20_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_p3_entry7_to_p4_approach20_bundle.npz"
)
CANONICAL_P3_ENTRY7_P4_ENTRY12_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_p3_entry7_to_p4_entry12_bundle.npz"
)
CANONICAL_ENTRY12_P4_APPROACH20_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_entry12_to_p4_approach20_bundle.npz"
)
CANONICAL_ENTRY20_P4_MIDWAY35_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_entry20_to_p4_midway35_bundle.npz"
)
CANONICAL_ENTRY35_P4_BRAKED55_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_entry35_to_p4_braked55_bundle.npz"
)
CANONICAL_ENTRY35_P4_APPROACH45_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_entry35_to_p4_approach45_bundle.npz"
)
CANONICAL_ENTRY45_P4_BRAKED55_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_entry45_to_p4_braked55_bundle.npz"
)
CANONICAL_ENTRY45_P4_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_entry45_to_p4_braked50_bundle.npz"
)
CANONICAL_P3_P4_MIDWAY35_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_p3_to_p4_midway35_bundle.npz"
)
CANONICAL_P3_P4_BRAKED55_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_canonical_p3_to_p4_braked55_bundle.npz"
)
UNIFIED_P3_P4_LATE_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_late_bundle.npz"
)
UNIFIED_P3_P4_MID_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_mid_bundle.npz"
)
UNIFIED_P3_P4_ALL_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_all_bundle.npz"
)
UNIFIED_P3_P4_FOCUS20_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_focus20_bundle.npz"
)
UNIFIED_P3_P4_FOCUS12_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_focus12_bundle.npz"
)
UNIFIED_P3_P4_EARLY12_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_early12_bundle.npz"
)
UNIFIED_P3_P4_EARLY7_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_early7_bundle.npz"
)
UNIFIED_P3_P4_FOCUS0_GAIT_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_focus0_gait_bundle.npz"
)
UNIFIED_P3_P4_FOCUS12_GAIT_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_focus12_gait_bundle.npz"
)
UNIFIED_P3_P4_EARLY12_GAIT_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_early12_gait_bundle.npz"
)
UNIFIED_P3_P4_EARLY7_GAIT_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_early7_gait_bundle.npz"
)
UNIFIED_P3_P4_ALL_GAIT_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_all_gait_bundle.npz"
)
UNIFIED_P3_P4_ALL_GAIT_FOCUS0X3_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "isaac_unified_p3_to_braked_p4_all_gait_focus0x3_bundle.npz"
)
V9_P3_BRAKED_P4_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_p3_strict_to_braked_p4_physx_bundle.npz"
)
V9_P3_ENTRY7_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_p3_strict_to_entry7_physx_bundle.npz"
)
V9_ENTRY7_APPROACH20_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry7_to_approach20_physx_bundle.npz"
)
V9_APPROACH20_MID35_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry25_to_mid35_physx_bundle.npz"
)
V9_APPROACH20_ENTRY25_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_approach20_to_entry25_physx_bundle.npz"
)
V9_FORMAL4096_APPROACH20_ENTRY25_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_hardentry_source_formal4096_20_to_25_entry_bundle.npz"
)
V9_LONG_HISTORY7_APPROACH20_HARD5X_ENTRY25_PREFILL_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_longhistory7_approach20_hard5x_entry25_prefill_bundle.npz"
)
V9_ENTRY25_MID30_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry25_to_mid30_paired109_physx_bundle.npz"
)
V9_MID30_MID35_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_mid30_to_mid35_physx_bundle.npz"
)
V9_MID35_APPROACH40_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_mid35_to_approach40_physx_bundle.npz"
)
V9_APPROACH40_ENTRY45_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_approach40_to_entry45_lowpen136_physx_bundle.npz"
)
V9_CONTINUOUS_MID35_APPROACH40_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_expanded114_best_continuous_formal4096_hardreset_35_to_40_entry_bundle.npz"
)
V9_CONTINUOUS_APPROACH40_ENTRY45_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_continuous_approach40_entry45_lowpen325_bundle.npz"
)
V9_APPROACH40_BRIDGE42_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_approach40_to_bridge42_physx_bundle.npz"
)
V9_APPROACH40_BRIDGE42_TEACHER_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_approach40_to_bridge42_teacher_bundle.npz"
)
V9_BRIDGE42_STRICT_CACHE_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_bridge42_strict_cache_bundle.npz"
)
V9_BRIDGE42_STRICT_HOLD6_TEACHER_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_bridge42_strict_hold6_teacher_bundle.npz"
)
V9_BRIDGE42_ENTRY45_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_bridge42_to_entry45_physx_bundle.npz"
)
V9_BRIDGE42_ENTRY45_TEACHER_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_bridge42_to_entry45_teacher_bundle.npz"
)
V9_ENTRY45_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry45_to_braked50_relaxed50ms_lowpen197_physx_bundle.npz"
)
V9_CURRENT_UPSTREAM45_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_current_upstream45_matched_final_control_bundle.npz"
)
V9_CURRENT_UPSTREAM45_APPROACH49_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_current_upstream45_to_approach49_bundle.npz"
)
V9_CURRENT_APPROACH49_ENTRY_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_current_approach49_entry_matched_final_control_bundle.npz"
)
V9_CURRENT_APPROACH49_PREFILL_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_approach49_prefill_braked50_expanded244_lowpen325_bundle.npz"
)
V9_FORMAL4096_APPROACH49_PREFILL_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_new40to45_formal4096_approach49_prefill_raw_bundle.npz"
)
V9_FORMAL4096_APPROACH49_PREFILL_LOWPEN325_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_new40to45_formal4096_approach49_prefill_lowpen325_bundle.npz"
)
V9_APPROACH49_PREFILL_CORE5X_FORMAL541_MIX_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_approach49_prefill_core5x_formal541_mix_bundle.npz"
)
V9_LONG_HISTORY35_APPROACH49_PREFILL_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_longhistory35_approach49_prefill_raw_bundle.npz"
)
V9_LONG_HISTORY35_APPROACH49_HARD3X_PREFILL_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_longhistory35_approach49_hard3x4953_prefill_bundle.npz"
)
V9_CURRENT8345_LONG_HISTORY35_APPROACH49_HARD5X_PREFILL_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_current8345_longhistory35_approach49_hard5x4453_bundle.npz"
)
V9_DUAL_ENTRY_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_dual_entry0_45_to_braked50_physx_bundle.npz"
)
V9_ALL_STAGE_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_all_stage_to_braked50_physx_bundle.npz"
)
V9_BACKWARD_SUFFIX_BUNDLES = {
    stage: str(
        ROOT
        / "dex5_penspin/reports/m3_2_30_index_near_surface/"
        f"v9_backward_suffix{stage}_to_braked50_physx_bundle.npz"
    )
    for stage in (40, 35, 30, 25, 20, 7, 0)
}
V9_BACKWARD_SUFFIX35_FOCUS_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_backward_suffix35_focusx3_to_braked50_physx_bundle.npz"
)
V9_ENTRY45_BRAKED50_TEACHER_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry45_to_braked50_teacher_bundle.npz"
)
V9_ENTRY45_APPROACH47_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry45_to_approach47_physx_bundle.npz"
)
V9_ENTRY45_APPROACH47_TEACHER_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry45_to_approach47_teacher_bundle.npz"
)
V9_ENTRY45_APPROACH47_HARD_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry45_to_approach47_hard_physx_bundle.npz"
)
V9_ENTRY45_HARD44_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry45_hard_to_capture44_physx_bundle.npz"
)
V9_ENTRY45_THUMB_SUPPORT_CORRECTION_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_entry45_thumb_support_correction_physx_bundle.npz"
)
V9_THUMB_SUPPORTED_ENTRY_APPROACH47_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_thumb_supported_entry_to_approach47_physx_bundle.npz"
)
V9_THUMB_SUPPORTED_ENTRY_CAPTURE45_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_thumb_supported_entry_to_capture45_physx_bundle.npz"
)
V9_THUMB_SUPPORTED_CAPTURE45_APPROACH47_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_thumb_supported_capture45_to_approach47_physx_bundle.npz"
)
V9_THUMB_SUPPORTED_APPROACH47_CAPTURE48_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_thumb_supported_approach47_to_capture48_physx_bundle.npz"
)
V9_THUMB_SUPPORTED_CAPTURE48_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_thumb_supported_capture48_to_braked50_physx_bundle.npz"
)
V9_CASCADE40_BRIDGE42_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_p4_cascade512_report_40_to_42_entry_bundle.npz"
)
V9_CASCADE42_ENTRY45_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_p4_cascade512_v2_report_42_to_45_entry_bundle.npz"
)
V9_CASCADE42_ENTRY45_TEACHER_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_cascade42_to_entry45_teacher_bundle.npz"
)
V9_CASCADE43_ENTRY45_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_p4_cascade512_v3_report_43_to_45_entry_bundle.npz"
)
V9_CASCADE_THUMB_SUPPORT_CORRECTION_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_p4_cascade1024_v17_mid30_brake_report_thumb_support_correction_entry_bundle.npz"
)
V9_CASCADE_SUPPORTED_TO45_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_p4_cascade1024_v19_supported435_report_supported_to_45_entry_bundle.npz"
)
V9_CASCADE45_APPROACH47_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_p4_cascade512_v12_report_45_to_47_entry_bundle.npz"
)
V9_CASCADE47_CAPTURE48_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_p4_cascade512_v13_report_47_capture_entry_bundle.npz"
)
V9_CASCADE_BRAKED_P4_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_p4_cascade1024_v20_supported45_report_braked_P4_entry_bundle.npz"
)
V9_MID35_BRAKED50_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_mid35_to_braked50_physx_bundle.npz"
)


@configclass
class Dex5BrakedP4PhysicsEnvCfg(Dex5P3EntryPhysicsEnvCfg):
    bundle_path = DEFAULT_BRAKED_P4_BUNDLE
    episode_length_s = 0.60
    action_scale = 0.105
    reset_joint_pos_noise = 0.002
    reset_joint_vel_noise = 0.010
    reset_pen_pos_noise = 0.0004
    reset_pen_vel_noise = 0.005
    target_total_phase_deg = 55.0
    min_phase_deg = 50.0
    max_phase_deg = 60.0
    max_braked_angular_speed = 12.0
    min_success_thumb_force_n = 0.03
    max_success_hand_rmse = 0.22
    max_success_penetration_m = 0.004
    max_success_pen_drift_m = 0.015
    penetration_soft_limit_m = 0.0015
    penetration_penalty_span_m = 0.003
    hold_steps = 6
    use_bundle_source_target = True
    thumb_support_bias = (0.0, 0.0, 0.0, 0.0)
    policy_target_blend_steps = 0
    teacher_prefill_steps = 0
    teacher_preserve_thumb_actions = False
    teacher_preserve_middle_actions = False
    thumb_recapture_steps = 0
    thumb_recapture_action = (0.0, 0.0, 0.0, 0.0)
    middle_recapture_steps = 0
    middle_recapture_action = (0.0, 0.0, 0.0, 0.0)
    use_policy_target_as_task_target = False
    bridge_warmup_steps = 0
    index_drive_reward_weight = 1.5
    index_contact_loss_weight = 0.0
    middle_contact_loss_weight = 0.0
    triad_contact_reward_weight = 0.0
    ring_load_penalty_weight = 0.0
    phase_conditioned_capture = False
    local_phase_start_deg = 9.0
    capture_trigger_deg = 48.0
    capture_transition_width_deg = 1.0
    local_phase_progress_weight = 7.0
    directional_progress_weight = 0.0
    # Signed planar direction from the reset pen axis to the task target.
    # Legacy P3->P4 assets rotate clockwise; v9 rotates counter-clockwise.
    phase_direction_sign = -1.0
    middle_near_surface_weight = 0.0
    overshoot_penalty_weight = 0.0
    capture_speed_penalty_weight = 0.0
    drive_reward_capture_suppression = 0.0
    force_excess_penalty_weight = 0.0
    force_excess_threshold_n = 1.0
    force_excess_penalty_span_n = 3.0
    force_excess_penalty_max = 1.0
    thumb_load_reward_weight = 0.0
    thumb_load_target_n = 0.30
    instantaneous_braked_reward_weight = 5.0
    hold_reward_weight = 0.8
    terminal_success_reward_weight = 12.0
    drift_penalty_weight = 4.0
    drift_penalty_scale_m = 0.020
    drift_penalty_max = 1.0
    drift_phase_gate_start_deg = -180.0
    drift_phase_gate_width_deg = 1.0
    drift_phase_gate_floor = 1.0
    pen_target_reward_weight = 0.0
    pen_target_reward_decay = 80.0


@configclass
class Dex5P4ApproachPhysicsEnvCfg(Dex5BrakedP4PhysicsEnvCfg):
    bundle_path = STAGE0_BRAKED_P4_BUNDLE
    episode_length_s = 0.40
    target_total_phase_deg = 20.0
    min_phase_deg = 17.0
    max_phase_deg = 23.0
    max_braked_angular_speed = 16.0
    hold_steps = 6


@configclass
class Dex5CanonicalP3P4ApproachPhysicsEnvCfg(Dex5P4ApproachPhysicsEnvCfg):
    """First P3->P4 curriculum stage, reset from the projected canonical P3."""

    bundle_path = CANONICAL_P3_P4_APPROACH20_BUNDLE
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5CanonicalP3P4EntryPhysicsEnvCfg(Dex5CanonicalP3P4ApproachPhysicsEnvCfg):
    """Short index-retention entry before the 20 degree approach stage."""

    bundle_path = CANONICAL_P3_P4_ENTRY7_BUNDLE
    episode_length_s = 0.24
    target_total_phase_deg = 7.0
    min_phase_deg = 5.0
    max_phase_deg = 10.0
    hold_steps = 3
    index_drive_reward_weight = 3.5
    index_contact_loss_weight = 4.0
    ring_load_penalty_weight = 1.0


@configclass
class Dex5CanonicalP3Entry7P4ApproachPhysicsEnvCfg(Dex5CanonicalP3P4ApproachPhysicsEnvCfg):
    """Continue from successful 7-degree captures to the 20-degree approach."""

    bundle_path = CANONICAL_P3_ENTRY7_P4_APPROACH20_BUNDLE
    episode_length_s = 0.36
    teacher_prefill_steps = 3
    index_drive_reward_weight = 3.5
    index_contact_loss_weight = 4.0
    ring_load_penalty_weight = 1.0


@configclass
class Dex5CanonicalP3Entry7P4Entry12PhysicsEnvCfg(Dex5CanonicalP3P4ApproachPhysicsEnvCfg):
    """Five-degree contact-retaining increment from the learned 7-degree entry."""

    bundle_path = CANONICAL_P3_ENTRY7_P4_ENTRY12_BUNDLE
    episode_length_s = 0.28
    target_total_phase_deg = 12.0
    min_phase_deg = 10.0
    max_phase_deg = 14.0
    hold_steps = 3
    teacher_prefill_steps = 3
    index_drive_reward_weight = 4.0
    index_contact_loss_weight = 8.0
    ring_load_penalty_weight = 3.0


@configclass
class Dex5CanonicalEntry12P4ApproachPhysicsEnvCfg(Dex5CanonicalP3P4ApproachPhysicsEnvCfg):
    """Continue from measured 12-degree captures to the 20-degree approach."""

    bundle_path = CANONICAL_ENTRY12_P4_APPROACH20_BUNDLE
    episode_length_s = 0.32
    target_total_phase_deg = 20.0
    min_phase_deg = 17.0
    max_phase_deg = 23.0
    hold_steps = 6
    teacher_prefill_steps = 0
    index_drive_reward_weight = 4.0
    index_contact_loss_weight = 8.0
    ring_load_penalty_weight = 3.0


@configclass
class Dex5CanonicalP3BrakedP4PhysicsEnvCfg(Dex5BrakedP4PhysicsEnvCfg):
    bundle_path = CANONICAL_P3_P4_BRAKED55_BUNDLE
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5P4MidwayPhysicsEnvCfg(Dex5BrakedP4PhysicsEnvCfg):
    bundle_path = STAGE1_BRAKED_P4_BUNDLE
    episode_length_s = 0.40
    action_scale = 0.105
    target_total_phase_deg = 35.0
    min_phase_deg = 32.0
    max_phase_deg = 38.0
    max_braked_angular_speed = 14.0
    hold_steps = 4
    thumb_support_bias = (0.0, 0.0, 0.0, 0.0)
    policy_target_blend_steps = 9
    teacher_prefill_steps = 12
    teacher_preserve_thumb_actions = True
    thumb_recapture_steps = 6
    thumb_recapture_action = (0.9004, -0.3055, -0.4795, 0.3989)
    middle_recapture_steps = 0
    reset_joint_pos_noise = 0.001
    reset_pen_pos_noise = 0.0001
    target_phase_fraction = 1.0


@configclass
class Dex5CanonicalP3P4MidwayPhysicsEnvCfg(Dex5P4MidwayPhysicsEnvCfg):
    bundle_path = CANONICAL_P3_P4_MIDWAY35_BUNDLE
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5CanonicalEntry20P4MidwayPhysicsEnvCfg(Dex5CanonicalP3P4MidwayPhysicsEnvCfg):
    """Continue from measured 20-degree captures to the 35-degree midway state."""

    bundle_path = CANONICAL_ENTRY20_P4_MIDWAY35_BUNDLE
    episode_length_s = 0.36
    target_total_phase_deg = 35.0
    min_phase_deg = 32.0
    max_phase_deg = 38.0
    max_braked_angular_speed = 14.0
    hold_steps = 4
    teacher_prefill_steps = 0
    policy_target_blend_steps = 0
    thumb_recapture_steps = 0
    index_drive_reward_weight = 4.0
    index_contact_loss_weight = 8.0
    ring_load_penalty_weight = 3.0


@configclass
class Dex5CanonicalEntry35BrakedP4PhysicsEnvCfg(Dex5CanonicalP3BrakedP4PhysicsEnvCfg):
    """Final 35-degree capture to the 50-60 degree braked P4 target."""

    bundle_path = CANONICAL_ENTRY35_P4_BRAKED55_BUNDLE
    episode_length_s = 0.42
    target_total_phase_deg = 55.0
    min_phase_deg = 50.0
    max_phase_deg = 60.0
    max_braked_angular_speed = 12.0
    hold_steps = 6
    teacher_prefill_steps = 0
    policy_target_blend_steps = 0
    thumb_recapture_steps = 0
    index_drive_reward_weight = 4.0
    index_contact_loss_weight = 8.0
    middle_contact_loss_weight = 12.0
    triad_contact_reward_weight = 8.0
    ring_load_penalty_weight = 3.0


@configclass
class Dex5CanonicalEntry35P4Approach45PhysicsEnvCfg(Dex5CanonicalEntry35BrakedP4PhysicsEnvCfg):
    """Intermediate brake-entry state before the final 50-60 degree target."""

    bundle_path = CANONICAL_ENTRY35_P4_APPROACH45_BUNDLE
    target_total_phase_deg = 45.0
    min_phase_deg = 42.0
    max_phase_deg = 48.0
    max_braked_angular_speed = 14.0
    hold_steps = 4


@configclass
class Dex5CanonicalEntry45BrakedP4PhysicsEnvCfg(Dex5CanonicalEntry35BrakedP4PhysicsEnvCfg):
    """Final local task reset from measured 43-45 degree brake-entry captures."""

    bundle_path = CANONICAL_ENTRY45_P4_BRAKED55_BUNDLE


@configclass
class Dex5CanonicalEntry45BrakedP4At50PhysicsEnvCfg(Dex5CanonicalEntry45BrakedP4PhysicsEnvCfg):
    """Lower edge of the valid 50-60 degree braked P4 target band."""

    bundle_path = CANONICAL_ENTRY45_P4_BRAKED50_BUNDLE
    target_total_phase_deg = 50.0
    min_phase_deg = 50.0
    max_phase_deg = 60.0
    middle_contact_loss_weight = 6.0
    triad_contact_reward_weight = 4.0


@configclass
class Dex5CanonicalEntry45PhaseCapture47PhysicsEnvCfg(
    Dex5CanonicalEntry45BrakedP4At50PhysicsEnvCfg
):
    """First drive-to-capture curriculum gate; final target orientation remains 50 degrees."""

    min_phase_deg = 47.0
    hold_steps = 3
    phase_conditioned_capture = True
    local_phase_start_deg = 43.0
    capture_trigger_deg = 47.0
    local_phase_progress_weight = 14.0
    directional_progress_weight = 5.0
    middle_near_surface_weight = 5.0
    middle_contact_loss_weight = 8.0
    triad_contact_reward_weight = 8.0


@configclass
class Dex5CanonicalEntry45PhaseCapture48PhysicsEnvCfg(
    Dex5CanonicalEntry45PhaseCapture47PhysicsEnvCfg
):
    """Second curriculum gate with a longer contact-retaining capture."""

    min_phase_deg = 48.0
    hold_steps = 4
    capture_trigger_deg = 48.0


@configclass
class Dex5CanonicalEntry45PhaseCapture49PhysicsEnvCfg(
    Dex5CanonicalEntry45PhaseCapture47PhysicsEnvCfg
):
    """Last relaxed curriculum gate before strict braked-P4 evaluation."""

    min_phase_deg = 49.0
    hold_steps = 5
    capture_trigger_deg = 48.0


@configclass
class Dex5CanonicalEntry45PhaseCapture50PhysicsEnvCfg(
    Dex5CanonicalEntry45PhaseCapture47PhysicsEnvCfg
):
    """Strict 50-degree, six-step braked-P4 task with phase-conditioned shaping."""

    min_phase_deg = 50.0
    hold_steps = 6
    capture_trigger_deg = 48.0


@configclass
class Dex5CanonicalEntry45PhaseCapture50NoisePhysicsEnvCfg(
    Dex5CanonicalEntry45PhaseCapture50PhysicsEnvCfg
):
    """Strict final task with small reset perturbations for robustness auditing."""

    reset_joint_pos_noise = 0.002
    reset_joint_vel_noise = 0.010
    reset_pen_pos_noise = 0.0004
    reset_pen_vel_noise = 0.005


@configclass
class Dex5UnifiedP3BrakedP4LatePhysicsEnvCfg(Dex5CanonicalEntry45PhaseCapture50NoisePhysicsEnvCfg):
    """Unified strict task reset from measured 35- and 45-degree captures."""

    bundle_path = UNIFIED_P3_P4_LATE_BUNDLE
    episode_length_s = 0.60


@configclass
class Dex5UnifiedP3BrakedP4MidPhysicsEnvCfg(Dex5UnifiedP3BrakedP4LatePhysicsEnvCfg):
    """Extend the unified strict task to 20-degree captures."""

    bundle_path = UNIFIED_P3_P4_MID_BUNDLE
    episode_length_s = 0.72


@configclass
class Dex5UnifiedP3BrakedP4AllPhysicsEnvCfg(Dex5UnifiedP3BrakedP4LatePhysicsEnvCfg):
    """Full canonical-P3 reset curriculum through strict braked P4."""

    bundle_path = UNIFIED_P3_P4_ALL_BUNDLE
    episode_length_s = 0.90


@configclass
class Dex5UnifiedP3BrakedP4Focus20PhysicsEnvCfg(Dex5UnifiedP3BrakedP4LatePhysicsEnvCfg):
    """Focused end-to-end task from the measured 20-degree capture distribution."""

    bundle_path = UNIFIED_P3_P4_FOCUS20_BUNDLE
    episode_length_s = 0.72


@configclass
class Dex5UnifiedP3BrakedP4Focus12PhysicsEnvCfg(Dex5UnifiedP3BrakedP4LatePhysicsEnvCfg):
    """Focused end-to-end task from the measured 12-degree capture distribution."""

    bundle_path = UNIFIED_P3_P4_FOCUS12_BUNDLE
    episode_length_s = 0.82


@configclass
class Dex5UnifiedP3BrakedP4Early12PhysicsEnvCfg(Dex5UnifiedP3BrakedP4LatePhysicsEnvCfg):
    """Unified curriculum from 12 degrees and all later capture distributions."""

    bundle_path = UNIFIED_P3_P4_EARLY12_BUNDLE
    episode_length_s = 0.82


@configclass
class Dex5UnifiedP3BrakedP4Early7PhysicsEnvCfg(Dex5UnifiedP3BrakedP4LatePhysicsEnvCfg):
    """Unified curriculum from 7 degrees and all later capture distributions."""

    bundle_path = UNIFIED_P3_P4_EARLY7_BUNDLE
    episode_length_s = 0.86


@configclass
class Dex5UnifiedP3BrakedP4Focus12GaitPhysicsEnvCfg(Dex5UnifiedP3BrakedP4Focus12PhysicsEnvCfg):
    """12-degree focus with controller-base interpolation to the validated P4 hand pose."""

    bundle_path = UNIFIED_P3_P4_FOCUS12_GAIT_BUNDLE


@configclass
class Dex5UnifiedP3BrakedP4Focus0GaitPhysicsEnvCfg(Dex5UnifiedP3BrakedP4AllPhysicsEnvCfg):
    """Canonical-P3 focus with interpolation to the validated P4 hand pose."""

    bundle_path = UNIFIED_P3_P4_FOCUS0_GAIT_BUNDLE


@configclass
class Dex5UnifiedP3BrakedP4Early12GaitPhysicsEnvCfg(Dex5UnifiedP3BrakedP4Early12PhysicsEnvCfg):
    """Joint 12/20/35/45 curriculum with a physically measured final hand target."""

    bundle_path = UNIFIED_P3_P4_EARLY12_GAIT_BUNDLE


@configclass
class Dex5UnifiedP3BrakedP4Early7GaitPhysicsEnvCfg(Dex5UnifiedP3BrakedP4Early7PhysicsEnvCfg):
    """Joint 7-degree-and-later gait-aware curriculum."""

    bundle_path = UNIFIED_P3_P4_EARLY7_GAIT_BUNDLE


@configclass
class Dex5UnifiedP3BrakedP4AllGaitPhysicsEnvCfg(Dex5UnifiedP3BrakedP4AllPhysicsEnvCfg):
    """Full canonical-P3 gait-aware reset curriculum."""

    bundle_path = UNIFIED_P3_P4_ALL_GAIT_BUNDLE


@configclass
class Dex5UnifiedP3BrakedP4AllGaitFocus0X3PhysicsEnvCfg(Dex5UnifiedP3BrakedP4AllPhysicsEnvCfg):
    """Full gait-aware curriculum with threefold canonical-P3 sampling."""

    bundle_path = UNIFIED_P3_P4_ALL_GAIT_FOCUS0X3_BUNDLE


@configclass
class Dex5V9P3BrakedP4PhysicsEnvCfg(Dex5UnifiedP3BrakedP4Focus0GaitPhysicsEnvCfg):
    """v9 P3 reset distribution with a short T/I/M hold prefill and 50-degree capture."""

    bundle_path = V9_P3_BRAKED_P4_BUNDLE
    observation_space = 134
    collision_proxy_radius_scales = (0.99, 1.07, 1.07, 0.84, 0.80)
    # The current v9 cache contains a static P3 hold probe, not a directional
    # P3->P4 teacher. Do not mask the policy with that probe during training.
    teacher_prefill_steps = 0
    # The v9 bundle advances in the positive task direction.  Diagnostics must
    # use this configured sign rather than the legacy clockwise convention.
    phase_direction_sign = 1.0
    decimation = 1
    # 18 * (1 / 120 s) = 150 ms.
    hold_steps = 18
    overshoot_penalty_weight = 15.0
    capture_speed_penalty_weight = 10.0
    drive_reward_capture_suppression = 1.0
    force_excess_penalty_weight = 4.0
    instantaneous_braked_reward_weight = 12.0
    hold_reward_weight = 3.0
    terminal_success_reward_weight = 40.0
    drift_penalty_weight = 10.0
    drift_penalty_scale_m = 0.015
    drift_penalty_max = 2.0
    # Preserve the P3->P4 drive before capture, then tighten translation near P4.
    drift_phase_gate_start_deg = 48.0
    drift_phase_gate_width_deg = 2.0
    drift_phase_gate_floor = 0.05
    pen_target_reward_weight = 6.0
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9P3BrakedP4NoisePhysicsEnvCfg(Dex5V9P3BrakedP4PhysicsEnvCfg):
    """Small-reset-noise curriculum stage for v9 P3 to braked P4."""

    reset_joint_pos_noise = 0.001
    reset_joint_vel_noise = 0.005
    reset_pen_pos_noise = 0.0001
    reset_pen_vel_noise = 0.002


@configclass
class Dex5V9DualEntryBraked50PhysicsEnvCfg(Dex5V9P3BrakedP4PhysicsEnvCfg):
    """One phase-conditioned policy trained from balanced v9 P3 and 45-degree resets."""

    bundle_path = V9_DUAL_ENTRY_BRAKED50_BUNDLE
    observation_space = 131


@configclass
class Dex5V9DualEntryLongBraked50PhysicsEnvCfg(Dex5V9DualEntryBraked50PhysicsEnvCfg):
    """Long-horizon audit matching the measured duration of the reset-free teacher cascade."""

    episode_length_s = 3.5


@configclass
class Dex5V9AllStageBraked50PhysicsEnvCfg(Dex5V9DualEntryBraked50PhysicsEnvCfg):
    """Current-v9 backward curriculum spanning P3, intermediate phases, and P4 entry."""

    bundle_path = V9_ALL_STAGE_BRAKED50_BUNDLE


@configclass
class Dex5V9BackwardSuffix40PhysicsEnvCfg(Dex5V9DualEntryBraked50PhysicsEnvCfg):
    """Backward curriculum with a common final P4 target, starting at 40 degrees."""

    bundle_path = V9_BACKWARD_SUFFIX_BUNDLES[40]


@configclass
class Dex5V9BackwardSuffix35PhysicsEnvCfg(Dex5V9BackwardSuffix40PhysicsEnvCfg):
    bundle_path = V9_BACKWARD_SUFFIX_BUNDLES[35]


@configclass
class Dex5V9BackwardSuffix35FocusPhysicsEnvCfg(Dex5V9BackwardSuffix35PhysicsEnvCfg):
    bundle_path = V9_BACKWARD_SUFFIX35_FOCUS_BUNDLE


@configclass
class Dex5V9BackwardSuffix30PhysicsEnvCfg(Dex5V9BackwardSuffix40PhysicsEnvCfg):
    bundle_path = V9_BACKWARD_SUFFIX_BUNDLES[30]


@configclass
class Dex5V9BackwardSuffix25PhysicsEnvCfg(Dex5V9BackwardSuffix40PhysicsEnvCfg):
    bundle_path = V9_BACKWARD_SUFFIX_BUNDLES[25]


@configclass
class Dex5V9BackwardSuffix20PhysicsEnvCfg(Dex5V9BackwardSuffix40PhysicsEnvCfg):
    bundle_path = V9_BACKWARD_SUFFIX_BUNDLES[20]


@configclass
class Dex5V9BackwardSuffix7PhysicsEnvCfg(Dex5V9BackwardSuffix40PhysicsEnvCfg):
    bundle_path = V9_BACKWARD_SUFFIX_BUNDLES[7]


@configclass
class Dex5V9BackwardSuffix0PhysicsEnvCfg(Dex5V9BackwardSuffix40PhysicsEnvCfg):
    bundle_path = V9_BACKWARD_SUFFIX_BUNDLES[0]


@configclass
class Dex5V9P3Entry7PhysicsEnvCfg(Dex5CanonicalP3P4EntryPhysicsEnvCfg):
    """Directional 0-to-7 degree curriculum from the v9 P3 reset distribution."""

    bundle_path = V9_P3_ENTRY7_BUNDLE
    collision_proxy_radius_scales = (0.99, 1.07, 1.07, 0.84, 0.80)
    teacher_prefill_steps = 2
    action_scale = 0.090
    phase_direction_sign = 1.0
    decimation = 2
    # Match canonical P3: 6 * (2 / 120 s) = 100 ms.
    hold_steps = 6
    max_braked_angular_speed = 12.0
    phase_conditioned_capture = True
    local_phase_start_deg = 0.0
    capture_trigger_deg = 5.0
    capture_transition_width_deg = 1.0
    local_phase_progress_weight = 12.0
    directional_progress_weight = 4.0
    middle_near_surface_weight = 3.0
    middle_contact_loss_weight = 4.0
    triad_contact_reward_weight = 4.0
    overshoot_penalty_weight = 15.0
    capture_speed_penalty_weight = 8.0
    drive_reward_capture_suppression = 1.0
    force_excess_penalty_weight = 4.0
    max_success_penetration_m = 0.0005
    penetration_soft_limit_m = 0.00025
    penetration_penalty_span_m = 0.0005


@configclass
class Dex5V9P3Entry7WidePhysicsEnvCfg(Dex5V9P3Entry7PhysicsEnvCfg):
    """Relaxed 5-15 degree braking band before strict 5-10 degree fine-tuning."""

    max_phase_deg = 15.0


@configclass
class Dex5V9Entry7Approach20PhysicsEnvCfg(Dex5V9P3Entry7PhysicsEnvCfg):
    """Continue from strict v9 entry captures to a braked 20-degree approach."""

    bundle_path = V9_ENTRY7_APPROACH20_BUNDLE
    episode_length_s = 0.40
    target_total_phase_deg = 20.0
    min_phase_deg = 17.0
    max_phase_deg = 23.0
    capture_trigger_deg = 17.0
    hold_steps = 6
    # Settle for six steps and apply one directional drive step. The policy then
    # owns braking and the full continuous-hold criterion.
    teacher_prefill_steps = 7
    index_contact_loss_weight = 8.0
    middle_contact_loss_weight = 6.0
    triad_contact_reward_weight = 8.0
    capture_speed_penalty_weight = 10.0
    instantaneous_braked_reward_weight = 12.0
    hold_reward_weight = 3.0
    terminal_success_reward_weight = 40.0


@configclass
class Dex5V9Approach20Mid35PhysicsEnvCfg(Dex5V9Entry7Approach20PhysicsEnvCfg):
    """Continue from strict 20-degree captures to a braked 35-degree midpoint."""

    bundle_path = V9_APPROACH20_MID35_BUNDLE
    episode_length_s = 0.50
    target_total_phase_deg = 35.0
    min_phase_deg = 32.0
    max_phase_deg = 38.0
    capture_trigger_deg = 32.0
    hold_steps = 6


@configclass
class Dex5V9Approach20Entry25PhysicsEnvCfg(Dex5V9Entry7Approach20PhysicsEnvCfg):
    """Five-degree capture increment between the 20- and 35-degree stages."""

    bundle_path = V9_APPROACH20_ENTRY25_BUNDLE
    episode_length_s = 0.40
    target_total_phase_deg = 25.0
    min_phase_deg = 23.0
    max_phase_deg = 28.0
    capture_trigger_deg = 23.0
    hold_steps = 6


@configclass
class Dex5V9Formal4096Approach20Entry25PhysicsEnvCfg(
    Dex5V9Approach20Entry25PhysicsEnvCfg
):
    """Train on current reset-free 20-degree entries from the formal cascade."""

    bundle_path = V9_FORMAL4096_APPROACH20_ENTRY25_BUNDLE
    teacher_prefill_steps = 0
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9LongHistory7Approach20Hard5xEntry25PhysicsEnvCfg(
    Dex5V9Approach20Entry25PhysicsEnvCfg
):
    """Replay real 7-to-20 contact history before the 20-to-25 policy."""

    bundle_path = V9_LONG_HISTORY7_APPROACH20_HARD5X_ENTRY25_PREFILL_BUNDLE
    episode_length_s = 1.2
    teacher_prefill_steps = 0
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9Entry25Mid30PhysicsEnvCfg(Dex5V9Entry7Approach20PhysicsEnvCfg):
    """Narrow v9 curriculum from the validated 25-degree cache to 30 degrees."""

    bundle_path = V9_ENTRY25_MID30_BUNDLE
    episode_length_s = 0.40
    target_total_phase_deg = 30.0
    min_phase_deg = 28.0
    max_phase_deg = 33.0
    capture_trigger_deg = 28.0
    hold_steps = 6
    # Six settle steps plus two paired drive steps place the policy near the
    # transition without satisfying the six-step braked hold on its behalf.
    teacher_prefill_steps = 8


@configclass
class Dex5V9Mid30Mid35PhysicsEnvCfg(Dex5V9Entry7Approach20PhysicsEnvCfg):
    """Narrow v9 curriculum from validated 30-degree captures to 35 degrees."""

    bundle_path = V9_MID30_MID35_BUNDLE
    episode_length_s = 0.40
    target_total_phase_deg = 35.0
    min_phase_deg = 33.0
    max_phase_deg = 38.0
    capture_trigger_deg = 33.0
    hold_steps = 6


@configclass
class Dex5V9Mid35Approach40PhysicsEnvCfg(Dex5V9Entry7Approach20PhysicsEnvCfg):
    """Narrow v9 curriculum from validated 35-degree captures to 40 degrees."""

    bundle_path = V9_MID35_APPROACH40_BUNDLE
    episode_length_s = 0.40
    target_total_phase_deg = 40.0
    min_phase_deg = 38.0
    max_phase_deg = 43.0
    capture_trigger_deg = 38.0
    hold_steps = 6


@configclass
class Dex5V9Approach40Entry45PhysicsEnvCfg(Dex5V9Entry7Approach20PhysicsEnvCfg):
    """Narrow v9 curriculum from validated 40-degree captures to 45 degrees."""

    bundle_path = V9_APPROACH40_ENTRY45_BUNDLE
    episode_length_s = 0.40
    target_total_phase_deg = 45.0
    min_phase_deg = 43.0
    max_phase_deg = 48.0
    capture_trigger_deg = 43.0
    hold_steps = 6


@configclass
class Dex5V9ContinuousMid35Approach40PhysicsEnvCfg(Dex5V9Mid35Approach40PhysicsEnvCfg):
    """Real reset-free 35-degree entries exported by the formal cascade."""

    bundle_path = V9_CONTINUOUS_MID35_APPROACH40_BUNDLE
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9ContinuousApproach40Entry45PhysicsEnvCfg(Dex5V9Approach40Entry45PhysicsEnvCfg):
    """Real reset-free 40-degree entries exported by the formal cascade."""

    bundle_path = V9_CONTINUOUS_APPROACH40_ENTRY45_BUNDLE
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9Approach40Bridge42PhysicsEnvCfg(Dex5V9P3Entry7PhysicsEnvCfg):
    """Two-degree bridge that avoids the high-force direct 40-to-45 transition."""

    bundle_path = V9_APPROACH40_BRIDGE42_BUNDLE
    episode_length_s = 0.35
    target_total_phase_deg = 42.0
    min_phase_deg = 41.0
    max_phase_deg = 44.0
    capture_trigger_deg = 41.0
    hold_steps = 6
    action_scale = 0.06
    local_phase_progress_weight = 10.0
    directional_progress_weight = 3.5
    capture_speed_penalty_weight = 12.0
    force_excess_penalty_weight = 10.0


@configclass
class Dex5V9Cascade40Bridge42PhysicsEnvCfg(Dex5V9Approach40Bridge42PhysicsEnvCfg):
    """Retrain 40-to-42 on states reached by the reset-free P3 cascade."""

    bundle_path = V9_CASCADE40_BRIDGE42_BUNDLE


@configclass
class Dex5V9Approach40Bridge42ShortPhysicsEnvCfg(Dex5V9Approach40Bridge42PhysicsEnvCfg):
    """First teach a 25-ms brake before requiring the strict 50-ms hold."""

    hold_steps = 3
    force_excess_penalty_span_n = 1.0
    force_excess_penalty_max = 4.0
    thumb_load_reward_weight = 6.0


@configclass
class Dex5V9Approach40Bridge42BootstrapPhysicsEnvCfg(Dex5V9Approach40Bridge42ShortPhysicsEnvCfg):
    """Bootstrap a braking signal before restoring the strict 41-degree gate."""

    min_phase_deg = 40.0
    capture_trigger_deg = 40.0


@configclass
class Dex5V9Approach40Bridge42TeacherPhysicsEnvCfg(Dex5V9Approach40Bridge42BootstrapPhysicsEnvCfg):
    """Replay reset-matched searched actions to materialize the 42-degree cache."""

    bundle_path = V9_APPROACH40_BRIDGE42_TEACHER_BUNDLE
    teacher_prefill_steps = 20


@configclass
class Dex5V9Bridge42StrictCachePhysicsEnvCfg(Dex5V9Approach40Bridge42ShortPhysicsEnvCfg):
    """Tighten bootstrap terminal states to the strict 41-degree, 25-ms gate."""

    bundle_path = V9_BRIDGE42_STRICT_CACHE_BUNDLE
    teacher_prefill_steps = 0


@configclass
class Dex5V9Bridge42StrictHold6TeacherPhysicsEnvCfg(Dex5V9Bridge42StrictCachePhysicsEnvCfg):
    """Replay searched actions that already satisfy the strict 50-ms hold."""

    bundle_path = V9_BRIDGE42_STRICT_HOLD6_TEACHER_BUNDLE
    hold_steps = 6
    teacher_prefill_steps = 16


@configclass
class Dex5V9Bridge42Entry45PhysicsEnvCfg(Dex5V9P3Entry7PhysicsEnvCfg):
    """Continue from validated 42-degree bridge states to 45 degrees."""

    bundle_path = V9_BRIDGE42_ENTRY45_BUNDLE
    episode_length_s = 0.35
    target_total_phase_deg = 45.0
    min_phase_deg = 43.0
    max_phase_deg = 48.0
    capture_trigger_deg = 43.0
    hold_steps = 6
    action_scale = 0.06
    local_phase_progress_weight = 8.0
    directional_progress_weight = 3.0
    capture_speed_penalty_weight = 16.0
    force_excess_penalty_weight = 12.0
    force_excess_penalty_span_n = 1.0
    force_excess_penalty_max = 4.0
    thumb_load_reward_weight = 6.0


@configclass
class Dex5V9Cascade42Entry45PhysicsEnvCfg(Dex5V9Bridge42Entry45PhysicsEnvCfg):
    """Retrain 42-to-45 on states reached by the reset-free P3 cascade."""

    bundle_path = V9_CASCADE42_ENTRY45_BUNDLE


@configclass
class Dex5V9Cascade42Entry45TeacherPhysicsEnvCfg(Dex5V9Cascade42Entry45PhysicsEnvCfg):
    """Replay reset-matched actions found on reset-free cascade entry states."""

    bundle_path = V9_CASCADE42_ENTRY45_TEACHER_BUNDLE
    teacher_prefill_steps = 24


@configclass
class Dex5V9Cascade42Entry43PhysicsEnvCfg(Dex5V9Bridge42Entry45PhysicsEnvCfg):
    """Short curriculum from reset-free 42-degree entries toward the 45-degree gate."""

    bundle_path = V9_CASCADE42_ENTRY45_BUNDLE
    episode_length_s = 0.30
    # Keep the global 45-degree phase coordinate; only shorten this stage's gate.
    target_total_phase_deg = 45.0
    min_phase_deg = 42.5
    max_phase_deg = 45.0
    capture_trigger_deg = 42.5
    hold_steps = 3
    action_scale = 0.045
    local_phase_progress_weight = 10.0
    directional_progress_weight = 3.5
    capture_speed_penalty_weight = 12.0


@configclass
class Dex5V9Cascade43Entry45PhysicsEnvCfg(Dex5V9Bridge42Entry45PhysicsEnvCfg):
    """Retrain the 45-degree gate on entries produced by the reset-free 42-to-43 stage."""

    bundle_path = V9_CASCADE43_ENTRY45_BUNDLE


@configclass
class Dex5V9Bridge42Entry45TeacherPhysicsEnvCfg(Dex5V9Bridge42Entry45PhysicsEnvCfg):
    """Replay reset-matched searched actions to materialize the strict 45-degree cache."""

    bundle_path = V9_BRIDGE42_ENTRY45_TEACHER_BUNDLE
    teacher_prefill_steps = 20


@configclass
class Dex5V9Entry45Braked50PhysicsEnvCfg(Dex5V9Entry7Approach20PhysicsEnvCfg):
    """Strict 150-ms v9 braked-P4 capture from validated 45-degree states."""

    bundle_path = V9_ENTRY45_BRAKED50_BUNDLE
    episode_length_s = 0.50
    target_total_phase_deg = 50.0
    min_phase_deg = 50.0
    max_phase_deg = 60.0
    capture_trigger_deg = 48.0
    hold_steps = 18
    action_scale = 0.06
    force_excess_penalty_weight = 12.0
    force_excess_penalty_span_n = 1.0
    force_excess_penalty_max = 4.0
    thumb_load_reward_weight = 6.0


@configclass
class Dex5V9CurrentUpstream45Braked50PhysicsEnvCfg(Dex5V9Entry45Braked50PhysicsEnvCfg):
    """Braked-P4 capture from reset-free upstream states reached by the current cascade."""

    bundle_path = V9_CURRENT_UPSTREAM45_BRAKED50_BUNDLE
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    # The cache stores the preceding 40-to-45 teacher. The final oracle must
    # own the action immediately after reset rather than replaying that stage.
    teacher_prefill_steps = 0
    capture_trigger_deg = 49.0
    local_phase_progress_weight = 20.0
    directional_progress_weight = 8.0


@configclass
class Dex5V9CurrentUpstream45Approach49PhysicsEnvCfg(Dex5V9Entry45Braked50PhysicsEnvCfg):
    """Bridge broad reset-free upstream entries into a compact 49-degree capture."""

    bundle_path = V9_CURRENT_UPSTREAM45_APPROACH49_BUNDLE
    episode_length_s = 0.50
    target_total_phase_deg = 49.0
    min_phase_deg = 48.0
    max_phase_deg = 52.0
    capture_trigger_deg = 47.5
    hold_steps = 6
    action_scale = 0.09
    teacher_prefill_steps = 0
    local_phase_progress_weight = 20.0
    directional_progress_weight = 8.0
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9CurrentApproach49EntryBraked50PhysicsEnvCfg(Dex5V9Entry45Braked50PhysicsEnvCfg):
    """Final braking from reset-free entries produced by the upstream 49-degree bridge."""

    bundle_path = V9_CURRENT_APPROACH49_ENTRY_BRAKED50_BUNDLE
    teacher_prefill_steps = 0
    # Keep reset-reconstruction spikes observable; the success gate remains 0.5 mm.
    penetration_threshold = 0.004
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9Approach49PrefillBraked50PhysicsEnvCfg(Dex5V9Entry45Braked50PhysicsEnvCfg):
    """Recreate Approach49 contact history before handing control to final braking."""

    bundle_path = V9_CURRENT_APPROACH49_PREFILL_BRAKED50_BUNDLE
    episode_length_s = 1.50
    teacher_prefill_steps = 49
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5V9Formal4096Approach49PrefillBraked50PhysicsEnvCfg(
    Dex5V9Approach49PrefillBraked50PhysicsEnvCfg
):
    """Audit the real Approach49 entries from the hard-reset 4096 cascade."""

    bundle_path = V9_FORMAL4096_APPROACH49_PREFILL_BRAKED50_BUNDLE


@configclass
class Dex5V9Formal4096Approach49PrefillLowpen325Braked50PhysicsEnvCfg(
    Dex5V9Approach49PrefillBraked50PhysicsEnvCfg
):
    """Train on physically rebuildable entries from the hard-reset 4096 cascade."""

    bundle_path = V9_FORMAL4096_APPROACH49_PREFILL_LOWPEN325_BUNDLE


@configclass
class Dex5V9Approach49PrefillCore5xFormal541MixBraked50PhysicsEnvCfg(
    Dex5V9Approach49PrefillBraked50PhysicsEnvCfg
):
    """Retain the stable core while adapting to wider real continuous entries."""

    bundle_path = V9_APPROACH49_PREFILL_CORE5X_FORMAL541_MIX_BUNDLE


@configclass
class Dex5V9LongHistory35Approach49PrefillBraked50PhysicsEnvCfg(
    Dex5V9Approach49PrefillBraked50PhysicsEnvCfg
):
    """Replay exact actuator targets from real 35-degree continuous entries."""

    bundle_path = V9_LONG_HISTORY35_APPROACH49_PREFILL_BUNDLE
    episode_length_s = 2.0
    teacher_prefill_steps = 0


@configclass
class Dex5V9LongHistory35Approach49Hard3xPrefillBraked50PhysicsEnvCfg(
    Dex5V9LongHistory35Approach49PrefillBraked50PhysicsEnvCfg
):
    """Emphasize real entries that reached Approach49 but failed final braking."""

    bundle_path = V9_LONG_HISTORY35_APPROACH49_HARD3X_PREFILL_BUNDLE
    capture_trigger_deg = 49.0
    local_phase_progress_weight = 20.0
    directional_progress_weight = 8.0


@configclass
class Dex5V9Current8345LongHistory35Approach49Hard5xBraked50PhysicsEnvCfg(
    Dex5V9LongHistory35Approach49Hard3xPrefillBraked50PhysicsEnvCfg
):
    """Adapt final braking to failures from the current 83.45% cascade."""

    bundle_path = V9_CURRENT8345_LONG_HISTORY35_APPROACH49_HARD5X_PREFILL_BUNDLE


@configclass
class Dex5V9Entry45Braked50TeacherPhysicsEnvCfg(Dex5V9Entry45Braked50PhysicsEnvCfg):
    """Replay searched actions satisfying the strict 150-ms braked-P4 gate."""

    bundle_path = V9_ENTRY45_BRAKED50_TEACHER_BUNDLE
    teacher_prefill_steps = 40


@configclass
class Dex5V9Entry45Approach47PhysicsEnvCfg(Dex5V9P3Entry7PhysicsEnvCfg):
    """Low-force bridge from validated 45-degree states toward final P4 braking."""

    bundle_path = V9_ENTRY45_APPROACH47_BUNDLE
    episode_length_s = 0.50
    target_total_phase_deg = 47.0
    min_phase_deg = 45.0
    max_phase_deg = 49.0
    capture_trigger_deg = 45.0
    hold_steps = 6
    action_scale = 0.06
    local_phase_progress_weight = 20.0
    directional_progress_weight = 8.0
    capture_speed_penalty_weight = 8.0
    force_excess_penalty_weight = 8.0
    force_excess_penalty_span_n = 1.0
    force_excess_penalty_max = 4.0
    thumb_load_reward_weight = 3.0


@configclass
class Dex5V9Entry45Approach47TeacherPhysicsEnvCfg(Dex5V9Entry45Approach47PhysicsEnvCfg):
    """Replay strict searched actions for the 47-degree approach cache."""

    bundle_path = V9_ENTRY45_APPROACH47_TEACHER_BUNDLE
    teacher_prefill_steps = 20


@configclass
class Dex5V9Entry45Approach47HardPhysicsEnvCfg(Dex5V9Entry45Approach47PhysicsEnvCfg):
    """Focused curriculum for the lower-phase tail of the validated 45-degree cache."""

    bundle_path = V9_ENTRY45_APPROACH47_HARD_BUNDLE


@configclass
class Dex5V9Entry45HardCapture44PhysicsEnvCfg(Dex5V9Entry45Approach47PhysicsEnvCfg):
    """Short capture for the lower-phase tail before rejoining approach47."""

    bundle_path = V9_ENTRY45_HARD44_BUNDLE
    target_total_phase_deg = 45.0
    min_phase_deg = 44.0
    max_phase_deg = 47.0
    capture_trigger_deg = 44.0


@configclass
class Dex5V9Entry45ThumbSupportCorrectionPhysicsEnvCfg(Dex5V9P3Entry7PhysicsEnvCfg):
    """Hold pen phase while moving failed P4 entries onto the thumb-supported branch."""

    bundle_path = V9_ENTRY45_THUMB_SUPPORT_CORRECTION_BUNDLE
    episode_length_s = 0.40
    target_total_phase_deg = 44.0
    min_phase_deg = 42.0
    max_phase_deg = 46.0
    capture_trigger_deg = 42.0
    hold_steps = 6
    action_scale = 0.05
    local_phase_progress_weight = 0.0
    directional_progress_weight = 0.0
    capture_speed_penalty_weight = 12.0
    force_excess_penalty_weight = 20.0
    force_excess_threshold_n = 0.8
    force_excess_penalty_span_n = 0.8
    force_excess_penalty_max = 8.0
    thumb_load_reward_weight = 10.0
    thumb_load_target_n = 0.22
    min_success_thumb_force_n = 0.12
    max_success_hand_rmse = 0.04
    triad_contact_reward_weight = 8.0
    index_contact_loss_weight = 5.0
    middle_contact_loss_weight = 5.0


@configclass
class Dex5V9CascadeThumbSupportCorrectionPhysicsEnvCfg(
    Dex5V9Entry45ThumbSupportCorrectionPhysicsEnvCfg
):
    """Retrain thumb support on states reached by the reset-free teacher stage."""

    bundle_path = V9_CASCADE_THUMB_SUPPORT_CORRECTION_BUNDLE


@configclass
class Dex5V9ThumbSupportedEntryApproach47PhysicsEnvCfg(Dex5V9Entry45Approach47PhysicsEnvCfg):
    """Resume phase progress from entries corrected onto the thumb-supported branch."""

    bundle_path = V9_THUMB_SUPPORTED_ENTRY_APPROACH47_BUNDLE
    min_success_thumb_force_n = 0.12
    max_success_hand_rmse = 0.08
    thumb_load_reward_weight = 8.0
    thumb_load_target_n = 0.22


@configclass
class Dex5V9ThumbSupportedEntryCapture45PhysicsEnvCfg(
    Dex5V9ThumbSupportedEntryApproach47PhysicsEnvCfg
):
    """Two-degree phase bridge that preserves the corrected thumb support."""

    bundle_path = V9_THUMB_SUPPORTED_ENTRY_CAPTURE45_BUNDLE
    target_total_phase_deg = 45.0
    min_phase_deg = 44.0
    max_phase_deg = 47.0
    capture_trigger_deg = 44.0
    local_phase_progress_weight = 16.0
    directional_progress_weight = 6.0


@configclass
class Dex5V9CascadeSupportedTo45PhysicsEnvCfg(
    Dex5V9ThumbSupportedEntryCapture45PhysicsEnvCfg
):
    """Retrain the 45-degree capture on reset-free thumb-supported entries."""

    bundle_path = V9_CASCADE_SUPPORTED_TO45_BUNDLE


@configclass
class Dex5V9CascadeSupportedApproach43PhysicsEnvCfg(
    Dex5V9CascadeSupportedTo45PhysicsEnvCfg
):
    """Short supported approach before restoring the strict 44-degree capture gate."""

    min_phase_deg = 43.0
    max_phase_deg = 46.0
    capture_trigger_deg = 43.0
    hold_steps = 3


@configclass
class Dex5V9CascadeSupportedApproach435PhysicsEnvCfg(
    Dex5V9CascadeSupportedTo45PhysicsEnvCfg
):
    """Half-degree bridge from the reset-free 43-degree supported entries."""

    min_phase_deg = 43.5
    max_phase_deg = 46.5
    capture_trigger_deg = 43.5
    hold_steps = 3


@configclass
class Dex5V9ThumbSupportedCapture45Approach47PhysicsEnvCfg(
    Dex5V9ThumbSupportedEntryApproach47PhysicsEnvCfg
):
    """Continue corrected 45-degree captures to the 47-degree P4 approach."""

    bundle_path = V9_THUMB_SUPPORTED_CAPTURE45_APPROACH47_BUNDLE
    local_phase_progress_weight = 18.0
    directional_progress_weight = 7.0


@configclass
class Dex5V9Cascade45Approach47PhysicsEnvCfg(
    Dex5V9ThumbSupportedCapture45Approach47PhysicsEnvCfg
):
    """Retrain 45-to-47 on entries reached by the reset-free v11 cascade."""

    bundle_path = V9_CASCADE45_APPROACH47_BUNDLE


@configclass
class Dex5V9ThumbSupportedApproach47Capture48PhysicsEnvCfg(
    Dex5V9ThumbSupportedEntryApproach47PhysicsEnvCfg
):
    """Reach an explicit 47-degree gate before final P4 braking."""

    bundle_path = V9_THUMB_SUPPORTED_APPROACH47_CAPTURE48_BUNDLE
    target_total_phase_deg = 48.0
    min_phase_deg = 47.0
    max_phase_deg = 50.0
    capture_trigger_deg = 47.0
    local_phase_progress_weight = 18.0
    directional_progress_weight = 7.0


@configclass
class Dex5V9Cascade47Capture48PhysicsEnvCfg(
    Dex5V9ThumbSupportedApproach47Capture48PhysicsEnvCfg
):
    """Retrain the 48-degree capture on reset-free 47-degree entries."""

    bundle_path = V9_CASCADE47_CAPTURE48_BUNDLE


@configclass
class Dex5V9ThumbSupportedCapture48Braked50PhysicsEnvCfg(
    Dex5V9ThumbSupportedEntryApproach47PhysicsEnvCfg
):
    """Final v9 P4 brake: 50-60 degrees with a strict 150-ms hold."""

    bundle_path = V9_THUMB_SUPPORTED_CAPTURE48_BRAKED50_BUNDLE
    episode_length_s = 0.60
    target_total_phase_deg = 50.0
    min_phase_deg = 50.0
    max_phase_deg = 60.0
    capture_trigger_deg = 49.0
    # This task inherits decimation=2 at 120 Hz: 9 * (2 / 120 s) = 150 ms.
    hold_steps = 9
    local_phase_progress_weight = 18.0
    directional_progress_weight = 7.0
    capture_speed_penalty_weight = 12.0
    thumb_load_reward_weight = 10.0


@configclass
class Dex5V9CascadeBrakedP4PhysicsEnvCfg(
    Dex5V9ThumbSupportedCapture48Braked50PhysicsEnvCfg
):
    """Retrain final braking on reset-free capture48 entries."""

    bundle_path = V9_CASCADE_BRAKED_P4_BUNDLE


@configclass
class Dex5V9Mid35Braked50PhysicsEnvCfg(Dex5V9P3Entry7PhysicsEnvCfg):
    """Final local curriculum from 35 degrees to strict braked P4 at 50-60 degrees."""

    bundle_path = V9_MID35_BRAKED50_BUNDLE
    episode_length_s = 0.75
    target_total_phase_deg = 50.0
    min_phase_deg = 50.0
    max_phase_deg = 60.0
    capture_trigger_deg = 47.0
    # 18 * 1/120 s = 150 ms.
    hold_steps = 18
    episode_length_s = 0.32
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0


@configclass
class Dex5P4RecaptureHoldPhysicsEnvCfg(Dex5P4MidwayPhysicsEnvCfg):
    episode_length_s = 0.30
    target_total_phase_deg = 20.0
    min_phase_deg = 17.0
    max_phase_deg = 28.0
    max_braked_angular_speed = 16.0
    hold_steps = 3
    policy_target_blend_steps = 0
    use_policy_target_as_task_target = True


@configclass
class Dex5P4Bridge25PhysicsEnvCfg(Dex5P4MidwayPhysicsEnvCfg):
    """Intermediate target used to bridge the 20-degree hold to the 35-degree stage."""

    target_total_phase_deg = 25.0
    min_phase_deg = 22.0
    max_phase_deg = 28.0
    target_phase_fraction = 0.50
    episode_length_s = 0.40
    policy_target_blend_steps = 0
    teacher_prefill_steps = 12
    teacher_preserve_thumb_actions = False
    thumb_recapture_steps = 0


@configclass
class Dex5P4Bridge30PhysicsEnvCfg(Dex5P4MidwayPhysicsEnvCfg):
    """Intermediate target used to bridge the 25-degree target to the 35-degree stage."""

    target_total_phase_deg = 30.0
    min_phase_deg = 27.0
    max_phase_deg = 33.0
    target_phase_fraction = 0.75
    episode_length_s = 0.40
    policy_target_blend_steps = 0
    teacher_prefill_steps = 0
    thumb_recapture_steps = 0


@configclass
class Dex5P4Bridge22PhysicsEnvCfg(Dex5P4MidwayPhysicsEnvCfg):
    """Small phase increment above the validated 20-degree hold stage."""

    target_total_phase_deg = 22.0
    min_phase_deg = 19.0
    max_phase_deg = 25.0
    target_phase_fraction = 0.15
    episode_length_s = 0.36
    policy_target_blend_steps = 0
    teacher_prefill_steps = 12
    teacher_preserve_thumb_actions = False
    teacher_preserve_middle_actions = False
    thumb_recapture_steps = 0


@configclass
class Dex5P4Bridge22CachePhysicsEnvCfg(Dex5P4Bridge22PhysicsEnvCfg):
    """Consume the materialized Bridge22 target without a second interpolation."""

    bundle_path = BRIDGE22_TEACHER_CACHE
    target_phase_fraction = 1.0


@configclass
class Dex5P4Bridge22ThumbPreservePhysicsEnvCfg(Dex5P4Bridge22CachePhysicsEnvCfg):
    bundle_path = BRIDGE22_THUMB_PRESERVE_CACHE
    teacher_prefill_steps = 0
    bridge_warmup_steps = 12


@configclass
class Dex5P4Bridge22DelayedContactPhysicsEnvCfg(Dex5P4Bridge22PhysicsEnvCfg):
    """Bridge from a delayed, revalidated T/I/M contact capture."""

    bundle_path = BRIDGE22_DELAYED_CONTACT_BUNDLE
    target_phase_fraction = 1.0
    episode_length_s = 0.30
    target_total_phase_deg = 22.0
    min_phase_deg = 19.0
    max_phase_deg = 25.0
    max_braked_angular_speed = 18.0
    hold_steps = 6
    teacher_prefill_steps = 12
    teacher_preserve_thumb_actions = False
    teacher_preserve_middle_actions = False
    reset_joint_pos_noise = 0.0005
    reset_joint_vel_noise = 0.005
    reset_pen_pos_noise = 0.0001
    reset_pen_vel_noise = 0.002
    bridge_warmup_steps = 0


@configclass
class Dex5P4Bridge22DelayedContactHoldPhysicsEnvCfg(Dex5P4Bridge22DelayedContactPhysicsEnvCfg):
    """Same delayed bank, with reset actuator targets held at measured capture qpos."""

    bundle_path = BRIDGE22_DELAYED_CONTACT_HOLD_BUNDLE


@configclass
class Dex5P4Bridge22DelayedContactStaticPhysicsEnvCfg(Dex5P4Bridge22DelayedContactPhysicsEnvCfg):
    """Measured capture with zero initial action, used only as a contact baseline."""

    bundle_path = BRIDGE22_DELAYED_CONTACT_STATIC_BUNDLE
    teacher_prefill_steps = 0


@configclass
class Dex5P4Bridge22DelayedContactProcessedPhysicsEnvCfg(Dex5P4Bridge22DelayedContactPhysicsEnvCfg):
    """Static baseline using the exact processed actuator target at capture."""

    bundle_path = BRIDGE22_DELAYED_CONTACT_PROCESSED_BUNDLE
    teacher_prefill_steps = 0


class Dex5BrakedP4PhysicsEnv(Dex5RecaptureP3PhysicsEnv):
    cfg: Dex5BrakedP4PhysicsEnvCfg

    def _load_bundle(self) -> None:
        super()._load_bundle()
        data = np.load(self.cfg.bundle_path)
        phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)
        selected = (phase_ids >= self.cfg.min_curriculum_phase) & (phase_ids <= self.cfg.max_curriculum_phase)
        policy_targets = np.asarray(data.get("policy_target_pen_quat", data["target_qpos"][:, 23:27]))
        self.policy_target_pen_quat_bank = torch.as_tensor(
            policy_targets[selected], dtype=torch.float32, device=self.device
        )
        self.policy_target_pen_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.policy_target_pen_quat[:, 0] = 1.0
        source_episode_steps = np.asarray(
            data.get("source_episode_step", np.zeros(len(phase_ids))), dtype=np.int64
        )
        self.policy_episode_step_bank = torch.as_tensor(
            source_episode_steps[selected], dtype=torch.long, device=self.device
        )
        self.policy_episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        teacher_actions = np.asarray(
            data.get("source_teacher_actions", np.zeros((len(phase_ids), 0, 20))), dtype=np.float32
        )
        self.teacher_action_bank = torch.as_tensor(teacher_actions[selected], device=self.device)
        self.teacher_actions = torch.zeros(
            (self.num_envs, teacher_actions.shape[1], 20), device=self.device
        )
        teacher_action_lengths = np.asarray(
            data.get(
                "teacher_action_lengths",
                np.full(len(phase_ids), teacher_actions.shape[1]),
            ),
            dtype=np.int64,
        )
        self.teacher_action_lengths_bank = torch.as_tensor(
            teacher_action_lengths[selected], dtype=torch.long, device=self.device
        )
        self.teacher_action_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.p4_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.bridge_source_pen_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.bridge_source_pen_quat[:, 0] = 1.0
        self.previous_total_phase_deg = torch.full(
            (self.num_envs,), float(self.cfg.local_phase_start_deg), device=self.device
        )
        source_phase_offsets = np.asarray(
            data.get(
                "source_phase_offset_rad",
                np.full(len(phase_ids), np.deg2rad(self.cfg.local_phase_start_deg)),
            ),
            dtype=np.float32,
        )
        self.source_phase_deg_bank = torch.as_tensor(
            np.rad2deg(source_phase_offsets[selected]), dtype=torch.float32, device=self.device
        )
        self.local_phase_start_deg_env = torch.full(
            (self.num_envs,), float(self.cfg.local_phase_start_deg), device=self.device
        )
        curriculum_stages = np.asarray(
            data.get("curriculum_stage_deg", np.rad2deg(source_phase_offsets)), dtype=np.float32
        )
        self.curriculum_stage_deg_bank = torch.as_tensor(
            curriculum_stages[selected], dtype=torch.float32, device=self.device
        )
        self.curriculum_stage_deg = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.teacher_prefill_steps > 0 and self.teacher_actions.shape[1] > 0:
            teacher_step = torch.clamp(
                self.episode_length_buf, max=self.teacher_actions.shape[1] - 1
            )
            env_ids = torch.arange(self.num_envs, device=self.device)
            teacher = self.teacher_actions[env_ids, teacher_step]
            if self.cfg.teacher_preserve_thumb_actions:
                teacher = teacher.clone()
                teacher[:, -4:] = actions[:, -4:]
            if self.cfg.teacher_preserve_middle_actions:
                teacher = teacher.clone()
                teacher[:, 4:8] = actions[:, 4:8]
            prefill_limit = torch.minimum(
                self.teacher_action_lengths,
                torch.full_like(
                    self.teacher_action_lengths,
                    min(self.cfg.teacher_prefill_steps, self.teacher_actions.shape[1]),
                ),
            )
            use_teacher = self.episode_length_buf < prefill_limit
            actions = torch.where(use_teacher.unsqueeze(-1), teacher, actions)
        if self.cfg.thumb_recapture_steps > 0:
            use_recapture = self.episode_length_buf < self.cfg.thumb_recapture_steps
            recapture = torch.tensor(self.cfg.thumb_recapture_action, device=self.device)
            actions = actions.clone()
            actions[:, -4:] = torch.where(
                use_recapture.unsqueeze(-1), recapture.unsqueeze(0), actions[:, -4:]
            )
        if self.cfg.middle_recapture_steps > 0:
            use_recapture = self.episode_length_buf < self.cfg.middle_recapture_steps
            recapture = torch.tensor(self.cfg.middle_recapture_action, device=self.device)
            actions = actions.clone()
            actions[:, 4:8] = torch.where(
                use_recapture.unsqueeze(-1), recapture.unsqueeze(0), actions[:, 4:8]
            )
        super()._pre_physics_step(actions)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        bridge_steps = int(getattr(self.cfg, "bridge_warmup_steps", 0))
        if bridge_steps > 0:
            alpha = torch.clamp(
                self.episode_length_buf.float() / float(bridge_steps), 0.0, 1.0
            ).unsqueeze(-1)
            saved_hand = self.target_hand_qpos
            saved_pen_pos = self.target_pen_pos
            saved_pen_quat = self.target_pen_quat
            self.target_hand_qpos = torch.lerp(self.source_targets, saved_hand, alpha)
            self.target_pen_pos = torch.lerp(self.init_pen_pos, saved_pen_pos, alpha)
            source_quat = self.bridge_source_pen_quat
            aligned = torch.where(
                (torch.sum(source_quat * saved_pen_quat, dim=-1, keepdim=True) < 0.0),
                -saved_pen_quat,
                saved_pen_quat,
            )
            blended = (1.0 - alpha) * source_quat + alpha * aligned
            self.target_pen_quat = blended / torch.linalg.norm(
                blended, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
        else:
            saved_hand = saved_pen_pos = saved_pen_quat = None
        if self.cfg.policy_target_blend_steps <= 0:
            observations = super()._get_observations()
            if bridge_steps > 0:
                self.target_hand_qpos = saved_hand
                self.target_pen_pos = saved_pen_pos
                self.target_pen_quat = saved_pen_quat
            return observations
        final_target = self.target_pen_quat
        alpha = torch.clamp(
            self.episode_length_buf.float() / float(self.cfg.policy_target_blend_steps), 0.0, 1.0
        ).unsqueeze(-1)
        blended = (1.0 - alpha) * self.policy_target_pen_quat + alpha * final_target
        blended = blended / torch.linalg.norm(blended, dim=-1, keepdim=True).clamp_min(1.0e-6)
        self.target_pen_quat = blended
        actual_episode_length = self.episode_length_buf
        self.episode_length_buf = torch.clamp(
            actual_episode_length + self.policy_episode_step,
            max=self.max_episode_length - 1,
        )
        observations = super()._get_observations()
        self.episode_length_buf = actual_episode_length
        self.target_pen_quat = final_target
        return observations

    def _apply_action(self) -> None:
        bridge_steps = int(getattr(self.cfg, "bridge_warmup_steps", 0))
        if bridge_steps <= 0:
            super()._apply_action()
        else:
            alpha = torch.clamp(
                self.episode_length_buf.float() / float(bridge_steps), 0.0, 1.0
            ).unsqueeze(-1)
            saved_hand = self.target_hand_qpos
            saved_pen_pos = self.target_pen_pos
            saved_pen_quat = self.target_pen_quat
            self.target_hand_qpos = torch.lerp(self.source_targets, saved_hand, alpha)
            self.target_pen_pos = torch.lerp(self.init_pen_pos, saved_pen_pos, alpha)
            aligned = torch.where(
                (torch.sum(self.bridge_source_pen_quat * saved_pen_quat, dim=-1, keepdim=True) < 0.0),
                -saved_pen_quat,
                saved_pen_quat,
            )
            blended = (1.0 - alpha) * self.bridge_source_pen_quat + alpha * aligned
            self.target_pen_quat = blended / torch.linalg.norm(
                blended, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            super()._apply_action()
            self.target_hand_qpos = saved_hand
            self.target_pen_pos = saved_pen_pos
            self.target_pen_quat = saved_pen_quat
        thumb_bias = torch.tensor(self.cfg.thumb_support_bias, device=self.device)
        if torch.any(thumb_bias != 0.0):
            self.processed_targets[:, -4:] += thumb_bias
            self.processed_targets = torch.maximum(
                torch.minimum(self.processed_targets, self.joint_upper), self.joint_lower
            )
            self.hand.set_joint_position_target(
                self.processed_targets, joint_ids=self.dex5_joint_ids
            )

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        failed, _ = self._compute_dones()

        initial_yaw = torch.atan2(self.init_pen_axis[:, 1], self.init_pen_axis[:, 0])
        current_yaw = torch.atan2(self.pen_axis[:, 1], self.pen_axis[:, 0])
        target_axis = quat_apply(self.target_pen_quat, self.pen_x_axis)
        target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
        direction = float(self.cfg.phase_direction_sign)
        relative_deflection = direction * torch.atan2(
            torch.sin(current_yaw - initial_yaw), torch.cos(current_yaw - initial_yaw)
        )
        remaining_error = direction * torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        target_phase = torch.deg2rad(torch.full_like(relative_deflection, self.cfg.target_total_phase_deg))
        total_phase = target_phase - remaining_error
        total_phase_deg = total_phase * (180.0 / torch.pi)

        contacts = (self.contact_force_norm > 0.03).float()
        required_contact_ratio = contacts[:, :3].mean(dim=-1)
        angular_speed = torch.linalg.norm(self.pen_angvel, dim=-1)
        hand_rmse = torch.sqrt(torch.mean((self.hand_qpos - self.target_hand_qpos) ** 2, dim=-1))
        pen_drift = torch.linalg.norm(self.pen_pos - self.init_pen_pos, dim=-1)
        pen_target_error = torch.linalg.norm(self.pen_pos - self.target_pen_pos, dim=-1)
        phase_in_band = (total_phase_deg >= self.cfg.min_phase_deg) & (total_phase_deg <= self.cfg.max_phase_deg)
        instantaneous_braked = (
            phase_in_band
            & (angular_speed <= self.cfg.max_braked_angular_speed)
            & (required_contact_ratio >= 1.0)
            & (self.contact_force_norm[:, 0] >= self.cfg.min_success_thumb_force_n)
            & (hand_rmse < self.cfg.max_success_hand_rmse)
            & (pen_drift < self.cfg.max_success_pen_drift_m)
            & (self.episode_max_penetration <= self.cfg.max_success_penetration_m)
        )
        self.p4_hold_count = torch.where(
            instantaneous_braked, self.p4_hold_count + 1, torch.zeros_like(self.p4_hold_count)
        )
        success = self.p4_hold_count >= self.cfg.hold_steps

        target_error = torch.abs(total_phase - target_phase)
        contact_quality = torch.clamp(self.contact_force_norm[:, :3] / 0.06, 0.0, 1.0)
        thumb_quality, index_quality, middle_quality = contact_quality.unbind(dim=-1)
        thumb_tip_error = torch.abs(self.tip_surface_dist[:, 0] - 0.0005)
        thumb_near_surface = torch.exp(-45.0 * thumb_tip_error)
        contact_gate = 0.10 + 0.20 * index_quality + 0.35 * thumb_quality + 0.35 * middle_quality
        if self.cfg.phase_conditioned_capture:
            local_span_deg = torch.clamp(
                self.cfg.target_total_phase_deg - self.local_phase_start_deg_env, min=1.0
            )
            progress = torch.clamp(
                (total_phase_deg - self.local_phase_start_deg_env) / local_span_deg,
                0.0,
                1.2,
            )
            phase_step_deg = total_phase_deg - self.previous_total_phase_deg
            directional_progress = torch.clamp(phase_step_deg / 0.75, -1.0, 1.0)
            transition_width = max(self.cfg.capture_transition_width_deg, 0.1)
            capture_gate = torch.sigmoid(
                (total_phase_deg - self.cfg.capture_trigger_deg) / transition_width
            )
            recapture_gate = torch.sigmoid(
                (total_phase_deg - (self.cfg.capture_trigger_deg - 3.0)) / transition_width
            )
            drive_contact_gate = 0.20 + 0.40 * thumb_quality + 0.40 * index_quality
        else:
            expected_increment = max(self.cfg.target_total_phase_deg - 9.0, 3.0)
            progress = torch.clamp(
                relative_deflection
                / torch.deg2rad(torch.full_like(relative_deflection, expected_increment)),
                0.0,
                1.2,
            )
            directional_progress = torch.zeros_like(progress)
            capture_gate = torch.ones_like(progress)
            recapture_gate = torch.ones_like(progress)
            drive_contact_gate = contact_gate
        middle_tip_error = torch.abs(self.tip_surface_dist[:, 2] - 0.0005)
        middle_near_surface = torch.exp(-45.0 * middle_tip_error)
        brake_gate = torch.clamp(
            (total_phase_deg - (self.cfg.min_phase_deg - 3.0)) / 3.0, 0.0, 1.0
        )
        drive_reward_gate = 1.0 - self.cfg.drive_reward_capture_suppression * capture_gate
        overshoot = torch.clamp((total_phase_deg - self.cfg.max_phase_deg) / 5.0, 0.0, 1.0)
        triad_peak_force = self.contact_force_norm[:, :3].max(dim=-1).values
        force_excess = torch.clamp(
            (triad_peak_force - self.cfg.force_excess_threshold_n)
            / self.cfg.force_excess_penalty_span_n,
            0.0,
            self.cfg.force_excess_penalty_max,
        )
        thumb_load = torch.clamp(
            self.contact_force_norm[:, 0] / self.cfg.thumb_load_target_n, 0.0, 1.0
        )
        drift_phase_gate = self.cfg.drift_phase_gate_floor + (
            1.0 - self.cfg.drift_phase_gate_floor
        ) * torch.sigmoid(
            (total_phase_deg - self.cfg.drift_phase_gate_start_deg)
            / max(self.cfg.drift_phase_gate_width_deg, 0.1)
        )
        terms = {
            "alive": torch.full_like(total_phase, 0.08),
            "phase_progress": self.cfg.local_phase_progress_weight
            * drive_contact_gate
            * drive_reward_gate
            * progress,
            "directional_progress": self.cfg.directional_progress_weight
            * drive_contact_gate
            * drive_reward_gate
            * directional_progress,
            "phase_target": 4.0 * contact_gate * torch.exp(-8.0 * target_error),
            "brake": 4.0 * brake_gate * torch.exp(-0.12 * angular_speed),
            "capture_speed": -self.cfg.capture_speed_penalty_weight
            * capture_gate
            * torch.clamp(angular_speed / self.cfg.max_braked_angular_speed, 0.0, 2.0),
            "phase_overshoot": -self.cfg.overshoot_penalty_weight * overshoot,
            "force_excess": -self.cfg.force_excess_penalty_weight * force_excess,
            "thumb_load": self.cfg.thumb_load_reward_weight * thumb_load,
            "thumb_support": 3.5 * thumb_quality,
            "thumb_near_surface": 5.0 * thumb_near_surface,
            "thumb_approach": -5.0 * torch.clamp(thumb_tip_error / 0.030, 0.0, 1.0),
            "index_drive": self.cfg.index_drive_reward_weight * index_quality,
            "index_contact_loss": -self.cfg.index_contact_loss_weight * (1.0 - index_quality),
            "middle_support": 3.5 * middle_quality,
            "middle_near_surface": self.cfg.middle_near_surface_weight
            * recapture_gate
            * middle_near_surface,
            "middle_contact_loss": -self.cfg.middle_contact_loss_weight
            * capture_gate
            * (1.0 - middle_quality),
            "triad_contact": self.cfg.triad_contact_reward_weight
            * (0.15 + 0.85 * capture_gate)
            * thumb_quality
            * index_quality
            * middle_quality,
            "thumb_middle_loss": -3.0 * ((1.0 - thumb_quality) + (1.0 - middle_quality)),
            "ring_load": -self.cfg.ring_load_penalty_weight
            * torch.clamp(self.contact_force_norm[:, 3] / 0.10, 0.0, 1.0),
            "hand_target": 1.5 * torch.exp(-6.0 * hand_rmse),
            "instantaneous_braked": self.cfg.instantaneous_braked_reward_weight
            * instantaneous_braked.float(),
            "hold": self.cfg.hold_reward_weight
            * torch.clamp(self.p4_hold_count.float() / self.cfg.hold_steps, 0.0, 1.0),
            "terminal_success": self.cfg.terminal_success_reward_weight * success.float(),
            "pen_target": self.cfg.pen_target_reward_weight
            * torch.exp(-self.cfg.pen_target_reward_decay * pen_target_error),
            "drift": -self.cfg.drift_penalty_weight
            * drift_phase_gate
            * torch.clamp(
                pen_drift / self.cfg.drift_penalty_scale_m,
                0.0,
                self.cfg.drift_penalty_max,
            ),
            "penetration": -4.0
            * torch.clamp(
                (self.episode_max_penetration - self.cfg.penetration_soft_limit_m)
                / self.cfg.penetration_penalty_span_m,
                0.0,
                1.0,
            ),
            "action_l2": -0.03 * torch.linalg.norm(self.actions, dim=-1),
            "action_smoothness": -0.05 * torch.linalg.norm(self.actions - self.previous_actions, dim=-1),
            "failure": -6.0 * failed.float(),
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
        self.previous_total_phase_deg.copy_(total_phase_deg.detach())
        self.extras["log"] = {f"reward/{name}": value.mean() for name, value in terms.items()}
        self.extras["log"].update(
            {
                "metric/total_phase_deg": total_phase_deg.mean(),
                "metric/relative_deflection_deg": relative_deflection.mean() * (180.0 / torch.pi),
                "metric/angular_speed_rad_s": angular_speed.mean(),
                "metric/required_contact_ratio": required_contact_ratio.mean(),
                "metric/instantaneous_braked_ratio": instantaneous_braked.float().mean(),
                "metric/braked_p4_success_ratio": success.float().mean(),
                "metric/hold_ms": self.p4_hold_count.float().mean()
                * (1000.0 * self.cfg.decimation * self.cfg.sim.dt),
                "metric/hand_rmse_rad": hand_rmse.mean(),
                "metric/pen_drift_mm": pen_drift.mean() * 1000.0,
                "metric/max_penetration_mm": self.episode_max_penetration.mean() * 1000.0,
                "metric/thumb_contact_ratio": contacts[:, 0].mean(),
                "metric/index_contact_ratio": contacts[:, 1].mean(),
                "metric/middle_contact_ratio": contacts[:, 2].mean(),
                "metric/ring_contact_ratio": contacts[:, 3].mean(),
                "metric/pinky_contact_ratio": contacts[:, 4].mean(),
                "metric/thumb_force_n": self.contact_force_norm[:, 0].mean(),
                "metric/thumb_tip_dist_mm": self.tip_surface_dist[:, 0].mean() * 1000.0,
                "metric/index_force_n": self.contact_force_norm[:, 1].mean(),
                "metric/middle_force_n": self.contact_force_norm[:, 2].mean(),
                "metric/triad_peak_force_n": triad_peak_force.mean(),
                "metric/capture_gate": capture_gate.mean(),
                "metric/phase_step_deg": phase_step_deg.mean()
                if self.cfg.phase_conditioned_capture
                else torch.zeros((), device=self.device),
            }
        )
        return reward

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        bridge_fraction = float(getattr(self.cfg, "target_phase_fraction", 1.0))
        if bridge_fraction < 1.0:
            # Keep the real PhysX reset, but expose a shorter target so the policy can
            # learn a continuous phase curriculum before the full 35-degree target.
            full_target_hand = self.target_hand_qpos[env_ids].clone()
            full_target_pen_pos = self.target_pen_pos[env_ids].clone()
            full_target_pen_quat = self.target_pen_quat[env_ids].clone()
            source_hand = self.source_targets[env_ids]
            source_pen_pos = self.init_pen_pos[env_ids]
            source_pen_quat = self.pen_quat[env_ids]
            aligned_target_quat = torch.where(
                (torch.sum(source_pen_quat * full_target_pen_quat, dim=-1, keepdim=True) < 0.0),
                -full_target_pen_quat,
                full_target_pen_quat,
            )
            blended_quat = (1.0 - bridge_fraction) * source_pen_quat + bridge_fraction * aligned_target_quat
            self.target_hand_qpos[env_ids] = torch.lerp(source_hand, full_target_hand, bridge_fraction)
            self.target_pen_pos[env_ids] = torch.lerp(source_pen_pos, full_target_pen_pos, bridge_fraction)
            self.target_pen_quat[env_ids] = blended_quat / torch.linalg.norm(
                blended_quat, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
        self.policy_target_pen_quat[env_ids] = self.policy_target_pen_quat_bank[
            self.current_bank_ids[env_ids]
        ]
        if self.cfg.use_policy_target_as_task_target:
            self.target_pen_quat[env_ids] = self.policy_target_pen_quat[env_ids]
        self.policy_episode_step[env_ids] = self.policy_episode_step_bank[
            self.current_bank_ids[env_ids]
        ]
        if self.teacher_actions.shape[1] > 0:
            self.teacher_actions[env_ids] = self.teacher_action_bank[self.current_bank_ids[env_ids]]
        self.teacher_action_lengths[env_ids] = self.teacher_action_lengths_bank[
            self.current_bank_ids[env_ids]
        ]
        self.bridge_source_pen_quat[env_ids] = self.source_qpos_bank[
            self.current_bank_ids[env_ids], 23:27
        ]
        self.p4_hold_count[env_ids] = 0
        self.local_phase_start_deg_env[env_ids] = self.source_phase_deg_bank[
            self.current_bank_ids[env_ids]
        ]
        self.curriculum_stage_deg[env_ids] = self.curriculum_stage_deg_bank[
            self.current_bank_ids[env_ids]
        ]
        self.previous_total_phase_deg[env_ids] = self.local_phase_start_deg_env[env_ids]


class Dex5P4ApproachPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5P4ApproachPhysicsEnvCfg


class Dex5P4MidwayPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5P4MidwayPhysicsEnvCfg


class Dex5CanonicalP3P4ApproachPhysicsEnv(Dex5P4ApproachPhysicsEnv):
    cfg: Dex5CanonicalP3P4ApproachPhysicsEnvCfg


class Dex5CanonicalP3P4EntryPhysicsEnv(Dex5P4ApproachPhysicsEnv):
    cfg: Dex5CanonicalP3P4EntryPhysicsEnvCfg


class Dex5CanonicalP3Entry7P4ApproachPhysicsEnv(Dex5P4ApproachPhysicsEnv):
    cfg: Dex5CanonicalP3Entry7P4ApproachPhysicsEnvCfg


class Dex5CanonicalP3Entry7P4Entry12PhysicsEnv(Dex5P4ApproachPhysicsEnv):
    cfg: Dex5CanonicalP3Entry7P4Entry12PhysicsEnvCfg


class Dex5CanonicalEntry12P4ApproachPhysicsEnv(Dex5P4ApproachPhysicsEnv):
    cfg: Dex5CanonicalEntry12P4ApproachPhysicsEnvCfg


class Dex5CanonicalEntry20P4MidwayPhysicsEnv(Dex5P4MidwayPhysicsEnv):
    cfg: Dex5CanonicalEntry20P4MidwayPhysicsEnvCfg


class Dex5CanonicalEntry35BrakedP4PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalEntry35BrakedP4PhysicsEnvCfg


class Dex5CanonicalEntry35P4Approach45PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalEntry35P4Approach45PhysicsEnvCfg


class Dex5CanonicalEntry45BrakedP4PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalEntry45BrakedP4PhysicsEnvCfg


class Dex5CanonicalEntry45BrakedP4At50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalEntry45BrakedP4At50PhysicsEnvCfg


class Dex5CanonicalEntry45PhaseCapture47PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalEntry45PhaseCapture47PhysicsEnvCfg


class Dex5CanonicalEntry45PhaseCapture48PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalEntry45PhaseCapture48PhysicsEnvCfg


class Dex5CanonicalEntry45PhaseCapture49PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalEntry45PhaseCapture49PhysicsEnvCfg


class Dex5CanonicalEntry45PhaseCapture50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalEntry45PhaseCapture50PhysicsEnvCfg


class Dex5CanonicalEntry45PhaseCapture50NoisePhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalEntry45PhaseCapture50NoisePhysicsEnvCfg


class Dex5UnifiedP3BrakedP4LatePhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4LatePhysicsEnvCfg


class Dex5UnifiedP3BrakedP4MidPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4MidPhysicsEnvCfg


class Dex5UnifiedP3BrakedP4AllPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4AllPhysicsEnvCfg


class Dex5UnifiedP3BrakedP4Focus20PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4Focus20PhysicsEnvCfg


class Dex5UnifiedP3BrakedP4Focus12PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4Focus12PhysicsEnvCfg


class Dex5UnifiedP3BrakedP4Early12PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4Early12PhysicsEnvCfg


class Dex5UnifiedP3BrakedP4Early7PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4Early7PhysicsEnvCfg


class Dex5UnifiedP3BrakedP4Focus12GaitPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4Focus12GaitPhysicsEnvCfg


class Dex5UnifiedP3BrakedP4Focus0GaitPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4Focus0GaitPhysicsEnvCfg


class Dex5UnifiedP3BrakedP4Early12GaitPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4Early12GaitPhysicsEnvCfg


class Dex5UnifiedP3BrakedP4Early7GaitPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4Early7GaitPhysicsEnvCfg


class Dex5UnifiedP3BrakedP4AllGaitPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4AllGaitPhysicsEnvCfg


class Dex5UnifiedP3BrakedP4AllGaitFocus0X3PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5UnifiedP3BrakedP4AllGaitFocus0X3PhysicsEnvCfg


class Dex5V9P3BrakedP4PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9P3BrakedP4PhysicsEnvCfg

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        observations["policy"] = torch.cat(
            (observations["policy"], self.init_pen_axis - self.pen_axis), dim=-1
        )
        return observations


class Dex5V9P3BrakedP4NoisePhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9P3BrakedP4NoisePhysicsEnvCfg

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        observations["policy"] = torch.cat(
            (observations["policy"], self.init_pen_axis - self.pen_axis), dim=-1
        )
        return observations


class Dex5V9DualEntryBraked50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9DualEntryBraked50PhysicsEnvCfg


class Dex5V9DualEntryLongBraked50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9DualEntryLongBraked50PhysicsEnvCfg


class Dex5V9AllStageBraked50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9AllStageBraked50PhysicsEnvCfg


class Dex5V9BackwardSuffix40PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9BackwardSuffix40PhysicsEnvCfg


class Dex5V9BackwardSuffix35PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9BackwardSuffix35PhysicsEnvCfg


class Dex5V9BackwardSuffix35FocusPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9BackwardSuffix35FocusPhysicsEnvCfg


class Dex5V9BackwardSuffix30PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9BackwardSuffix30PhysicsEnvCfg


class Dex5V9BackwardSuffix25PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9BackwardSuffix25PhysicsEnvCfg


class Dex5V9BackwardSuffix20PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9BackwardSuffix20PhysicsEnvCfg


class Dex5V9BackwardSuffix7PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9BackwardSuffix7PhysicsEnvCfg


class Dex5V9BackwardSuffix0PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9BackwardSuffix0PhysicsEnvCfg


class Dex5V9P3Entry7PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9P3Entry7PhysicsEnvCfg


class Dex5V9P3Entry7WidePhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9P3Entry7WidePhysicsEnvCfg


class Dex5V9Entry7Approach20PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Entry7Approach20PhysicsEnvCfg


class Dex5V9Approach20Mid35PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Approach20Mid35PhysicsEnvCfg


class Dex5V9Approach20Entry25PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Approach20Entry25PhysicsEnvCfg


class Dex5V9Entry25Mid30PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Entry25Mid30PhysicsEnvCfg


class Dex5V9Mid30Mid35PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Mid30Mid35PhysicsEnvCfg


class Dex5V9Mid35Approach40PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Mid35Approach40PhysicsEnvCfg


class Dex5V9Approach40Entry45PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Approach40Entry45PhysicsEnvCfg


class Dex5V9ContinuousMid35Approach40PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9ContinuousMid35Approach40PhysicsEnvCfg


class Dex5V9ContinuousApproach40Entry45PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9ContinuousApproach40Entry45PhysicsEnvCfg


class Dex5V9Approach40Bridge42PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Approach40Bridge42PhysicsEnvCfg


class Dex5V9Cascade40Bridge42PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Cascade40Bridge42PhysicsEnvCfg


class Dex5V9Approach40Bridge42ShortPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Approach40Bridge42ShortPhysicsEnvCfg


class Dex5V9Approach40Bridge42BootstrapPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Approach40Bridge42BootstrapPhysicsEnvCfg


class Dex5V9Approach40Bridge42TeacherPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Approach40Bridge42TeacherPhysicsEnvCfg


class Dex5V9Bridge42StrictCachePhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Bridge42StrictCachePhysicsEnvCfg


class Dex5V9Bridge42StrictHold6TeacherPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Bridge42StrictHold6TeacherPhysicsEnvCfg


class Dex5V9Bridge42Entry45PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Bridge42Entry45PhysicsEnvCfg


class Dex5V9Cascade42Entry45PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Cascade42Entry45PhysicsEnvCfg


class Dex5V9Cascade42Entry45TeacherPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Cascade42Entry45TeacherPhysicsEnvCfg


class Dex5V9Cascade42Entry43PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Cascade42Entry43PhysicsEnvCfg


class Dex5V9Cascade43Entry45PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Cascade43Entry45PhysicsEnvCfg


class Dex5V9Bridge42Entry45TeacherPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Bridge42Entry45TeacherPhysicsEnvCfg


class Dex5V9Entry45Braked50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Entry45Braked50PhysicsEnvCfg


class Dex5V9CurrentUpstream45Braked50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9CurrentUpstream45Braked50PhysicsEnvCfg


class Dex5V9CurrentUpstream45Approach49PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9CurrentUpstream45Approach49PhysicsEnvCfg


class Dex5V9CurrentApproach49EntryBraked50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9CurrentApproach49EntryBraked50PhysicsEnvCfg


class Dex5V9Approach49PrefillBraked50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Approach49PrefillBraked50PhysicsEnvCfg

    def _load_bundle(self) -> None:
        super()._load_bundle()
        with np.load(self.cfg.bundle_path) as data:
            phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)
            selected = (phase_ids >= self.cfg.min_curriculum_phase) & (
                phase_ids <= self.cfg.max_curriculum_phase
            )
            self.teacher_stage1_lengths_bank = torch.as_tensor(
                np.asarray(data["teacher_stage1_lengths"])[selected],
                dtype=torch.long,
                device=self.device,
            )
            self.teacher_stage1_target_hand_bank = torch.as_tensor(
                np.asarray(data["teacher_stage1_target_hand_qpos"])[selected],
                dtype=torch.float32,
                device=self.device,
            )
            self.teacher_stage2_source_target_bank = torch.as_tensor(
                np.asarray(data["teacher_stage2_source_target_qpos"])[selected],
                dtype=torch.float32,
                device=self.device,
            )
            self.teacher_stage2_target_hand_bank = torch.as_tensor(
                np.asarray(data["teacher_stage2_target_hand_qpos"])[selected],
                dtype=torch.float32,
                device=self.device,
            )
            self.final_source_target_bank = torch.as_tensor(
                np.asarray(data["final_source_target_qpos"])[selected],
                dtype=torch.float32,
                device=self.device,
            )
        self.teacher_stage1_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.teacher_stage1_target_hand = torch.zeros(
            (self.num_envs, 20), dtype=torch.float32, device=self.device
        )
        self.teacher_stage2_source_target = torch.zeros_like(self.teacher_stage1_target_hand)
        self.teacher_stage2_target_hand = torch.zeros_like(self.teacher_stage1_target_hand)
        self.final_source_target = torch.zeros_like(self.teacher_stage1_target_hand)
        self.prefill_transition_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        bank_ids = self.current_bank_ids[env_ids]
        self.teacher_stage1_lengths[env_ids] = self.teacher_stage1_lengths_bank[bank_ids]
        self.teacher_stage1_target_hand[env_ids] = self.teacher_stage1_target_hand_bank[
            bank_ids
        ]
        self.teacher_stage2_source_target[env_ids] = self.teacher_stage2_source_target_bank[
            bank_ids
        ]
        self.teacher_stage2_target_hand[env_ids] = self.teacher_stage2_target_hand_bank[
            bank_ids
        ]
        self.final_source_target[env_ids] = self.final_source_target_bank[bank_ids]
        self.prefill_transition_initialized[env_ids] = False

    def _initialize_final_transition(self) -> None:
        ready = (self.episode_length_buf >= self.teacher_action_lengths) & (
            ~self.prefill_transition_initialized
        )
        if not torch.any(ready):
            return
        self._compute_intermediate_values()
        env_ids = torch.nonzero(ready, as_tuple=False).squeeze(-1)
        target_axis = quat_apply(self.target_pen_quat[env_ids], self.pen_x_axis[env_ids])
        current_axis = self.pen_axis[env_ids]
        target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
        current_yaw = torch.atan2(current_axis[:, 1], current_axis[:, 0])
        remaining = torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        total_phase_deg = self.cfg.target_total_phase_deg - torch.rad2deg(remaining)
        self.init_pen_pos[env_ids] = self.pen_pos[env_ids]
        self.init_pen_axis[env_ids] = current_axis
        self.local_phase_start_deg_env[env_ids] = total_phase_deg
        self.previous_total_phase_deg[env_ids] = total_phase_deg
        self.p4_hold_count[env_ids] = 0
        self.prefill_transition_initialized[env_ids] = True

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._initialize_final_transition()
        super()._pre_physics_step(actions)

    def _apply_action(self) -> None:
        stage1 = self.episode_length_buf < self.teacher_stage1_lengths
        prefill = self.episode_length_buf < self.teacher_action_lengths
        stage1_tau = torch.clamp(self.episode_length_buf.float() / 29.0, 0.0, 1.0)
        stage1_tau = stage1_tau * stage1_tau * (3.0 - 2.0 * stage1_tau)
        stage1_target = torch.lerp(
            self.source_targets,
            self.teacher_stage1_target_hand,
            stage1_tau.unsqueeze(-1),
        )
        stage2_step = torch.clamp(
            self.episode_length_buf - self.teacher_stage1_lengths, min=0
        ).float()
        stage2_tau = torch.clamp(stage2_step / 29.0, 0.0, 1.0)
        stage2_tau = stage2_tau * stage2_tau * (3.0 - 2.0 * stage2_tau)
        stage2_target = torch.lerp(
            self.teacher_stage2_source_target,
            self.teacher_stage2_target_hand,
            stage2_tau.unsqueeze(-1),
        )

        final_step = torch.clamp(
            self.episode_length_buf - self.teacher_action_lengths, min=0
        ).float()
        final_tau = torch.clamp(final_step / 29.0, 0.0, 1.0)
        final_tau = final_tau * final_tau * (3.0 - 2.0 * final_tau)
        final_target = torch.lerp(
            self.final_source_target, self.target_hand_qpos, final_tau.unsqueeze(-1)
        )
        prefill_target = torch.where(stage1.unsqueeze(-1), stage1_target, stage2_target)
        actuator_target = torch.where(prefill.unsqueeze(-1), prefill_target, final_target)
        action_scale = torch.where(
            prefill,
            torch.full_like(final_tau, 0.09),
            torch.full_like(final_tau, self.cfg.action_scale),
        )
        self.processed_targets = actuator_target + action_scale.unsqueeze(-1) * self.actions
        self.processed_targets = torch.maximum(
            torch.minimum(self.processed_targets, self.joint_upper), self.joint_lower
        )
        self.hand.set_joint_position_target(
            self.processed_targets, joint_ids=self.dex5_joint_ids
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        actual_episode_length = self.episode_length_buf
        self.episode_length_buf = torch.clamp(
            actual_episode_length - self.teacher_action_lengths, min=0
        )
        observations = super()._get_observations()
        self.episode_length_buf = actual_episode_length
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        prefill = self.episode_length_buf < self.teacher_action_lengths
        self.p4_hold_count[prefill] = 0
        return torch.where(prefill, torch.zeros_like(reward), reward)


class Dex5V9LongHistory35Approach49PrefillBraked50PhysicsEnv(
    Dex5V9Approach49PrefillBraked50PhysicsEnv
):
    cfg: Dex5V9LongHistory35Approach49PrefillBraked50PhysicsEnvCfg

    def _load_bundle(self) -> None:
        super()._load_bundle()
        with np.load(self.cfg.bundle_path) as data:
            phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)
            selected = (phase_ids >= self.cfg.min_curriculum_phase) & (
                phase_ids <= self.cfg.max_curriculum_phase
            )
            self.teacher_processed_target_bank = torch.as_tensor(
                np.asarray(data["teacher_processed_targets"])[selected],
                dtype=torch.float32,
                device=self.device,
            )
        self.teacher_processed_target = torch.zeros(
            (self.num_envs, *self.teacher_processed_target_bank.shape[1:]),
            dtype=torch.float32,
            device=self.device,
        )

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        self.teacher_processed_target[env_ids] = self.teacher_processed_target_bank[
            self.current_bank_ids[env_ids]
        ]

    def _apply_action(self) -> None:
        super()._apply_action()
        prefill = self.episode_length_buf < self.teacher_action_lengths
        if not torch.any(prefill):
            return
        env_ids = torch.nonzero(prefill, as_tuple=False).squeeze(-1)
        replay_step = self.episode_length_buf[env_ids].clamp_max(
            self.teacher_processed_target.shape[1] - 1
        )
        self.processed_targets[env_ids] = self.teacher_processed_target[
            env_ids, replay_step
        ]
        self.hand.set_joint_position_target(
            self.processed_targets, joint_ids=self.dex5_joint_ids
        )


class Dex5V9Entry45Braked50TeacherPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Entry45Braked50TeacherPhysicsEnvCfg


class Dex5V9Entry45Approach47PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Entry45Approach47PhysicsEnvCfg


class Dex5V9Entry45Approach47TeacherPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Entry45Approach47TeacherPhysicsEnvCfg


class Dex5V9Entry45Approach47HardPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Entry45Approach47HardPhysicsEnvCfg


class Dex5V9Entry45HardCapture44PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Entry45HardCapture44PhysicsEnvCfg


class Dex5V9Entry45ThumbSupportCorrectionPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Entry45ThumbSupportCorrectionPhysicsEnvCfg


class Dex5V9CascadeThumbSupportCorrectionPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9CascadeThumbSupportCorrectionPhysicsEnvCfg


class Dex5V9ThumbSupportedEntryApproach47PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9ThumbSupportedEntryApproach47PhysicsEnvCfg


class Dex5V9ThumbSupportedEntryCapture45PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9ThumbSupportedEntryCapture45PhysicsEnvCfg


class Dex5V9CascadeSupportedTo45PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9CascadeSupportedTo45PhysicsEnvCfg


class Dex5V9CascadeSupportedApproach43PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9CascadeSupportedApproach43PhysicsEnvCfg


class Dex5V9CascadeSupportedApproach435PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9CascadeSupportedApproach435PhysicsEnvCfg


class Dex5V9Cascade45Approach47PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Cascade45Approach47PhysicsEnvCfg


class Dex5V9ThumbSupportedCapture45Approach47PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9ThumbSupportedCapture45Approach47PhysicsEnvCfg


class Dex5V9ThumbSupportedApproach47Capture48PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9ThumbSupportedApproach47Capture48PhysicsEnvCfg


class Dex5V9Cascade47Capture48PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Cascade47Capture48PhysicsEnvCfg


class Dex5V9ThumbSupportedCapture48Braked50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9ThumbSupportedCapture48Braked50PhysicsEnvCfg


class Dex5V9CascadeBrakedP4PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9CascadeBrakedP4PhysicsEnvCfg


class Dex5V9Mid35Braked50PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5V9Mid35Braked50PhysicsEnvCfg


class Dex5CanonicalP3P4MidwayPhysicsEnv(Dex5P4MidwayPhysicsEnv):
    cfg: Dex5CanonicalP3P4MidwayPhysicsEnvCfg


class Dex5CanonicalP3BrakedP4PhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5CanonicalP3BrakedP4PhysicsEnvCfg


class Dex5P4Bridge25PhysicsEnv(Dex5P4MidwayPhysicsEnv):
    cfg: Dex5P4Bridge25PhysicsEnvCfg


class Dex5P4Bridge30PhysicsEnv(Dex5P4MidwayPhysicsEnv):
    cfg: Dex5P4Bridge30PhysicsEnvCfg


class Dex5P4Bridge22PhysicsEnv(Dex5P4MidwayPhysicsEnv):
    cfg: Dex5P4Bridge22PhysicsEnvCfg


class Dex5P4Bridge22CachePhysicsEnv(Dex5P4Bridge22PhysicsEnv):
    cfg: Dex5P4Bridge22CachePhysicsEnvCfg


class Dex5P4Bridge22ThumbPreservePhysicsEnv(Dex5P4Bridge22CachePhysicsEnv):
    cfg: Dex5P4Bridge22ThumbPreservePhysicsEnvCfg


class Dex5P4Bridge22DelayedContactPhysicsEnv(Dex5P4Bridge22PhysicsEnv):
    cfg: Dex5P4Bridge22DelayedContactPhysicsEnvCfg


class Dex5P4Bridge22DelayedContactHoldPhysicsEnv(Dex5P4Bridge22DelayedContactPhysicsEnv):
    cfg: Dex5P4Bridge22DelayedContactHoldPhysicsEnvCfg


class Dex5P4Bridge22DelayedContactStaticPhysicsEnv(Dex5P4Bridge22DelayedContactPhysicsEnv):
    cfg: Dex5P4Bridge22DelayedContactStaticPhysicsEnvCfg


class Dex5P4Bridge22DelayedContactProcessedPhysicsEnv(Dex5P4Bridge22DelayedContactPhysicsEnv):
    cfg: Dex5P4Bridge22DelayedContactProcessedPhysicsEnvCfg


class Dex5P4RecaptureHoldPhysicsEnv(Dex5BrakedP4PhysicsEnv):
    cfg: Dex5P4RecaptureHoldPhysicsEnvCfg
