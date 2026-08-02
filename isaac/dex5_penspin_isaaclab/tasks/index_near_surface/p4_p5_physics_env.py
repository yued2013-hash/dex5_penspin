from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from .recapture_p3_physics_env import (
    Dex5RecaptureP3PhysicsEnv,
    Dex5RecaptureP3PhysicsEnvCfg,
)


ROOT = Path(__file__).resolve().parents[5]
V9_P4_P5_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_p4_to_p5_physx_bundle_lowpen05_v3.npz"
)
V9_MIDDLE_CLEAR_TEACHER_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "middle_clear_teacher_reset_bundle.npz"
)
V9_MIDDLE_CLEAR_FIXED_TEACHER_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "middle_clear_fixed_teacher_bundle.npz"
)
V9_MIDDLE_CLEAR_LEARNED_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "middle_clear_learned_cache_seed1310.npz"
)
V9_RING_CLEAR_LEARNED_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "ring_clear_learned_cache_seed1402.npz"
)
V9_JOINT_CLEAR_LEARNED_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "joint_clear_learned_cache_seed1503.npz"
)
V9_COLLISION_FREE_CAPTURE_CACHE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "capture_collision_free_cache_seed1503.npz"
)
V9_COLLISION_FREE_CAPTURE_TEACHER = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "capture_collision_free_teacher_seed1503.npz"
)
V9_P4_P5_UNIFIED_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "p4_p5_unified_five_stage_bundle.npz"
)
V9_P4_P5_UPSTREAM_CAPTURE_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "p3_p4_real_terminal_to_p5_capture_bundle.npz"
)
V9_P4_P5_UPSTREAM_MIDDLE_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "p3_p4_real_terminal_to_middle_clear_bundle.npz"
)
V9_CURRENT_P4_P5_MIDDLE_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "current8345_p4_terminal_to_p5_bundle.npz"
)
V9_CURRENT_P4_P5_CAPTURE_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/no_reset_chain/"
    "current_middle_success_to_p5_capture_collision_free.npz"
)
V9_UPSTREAM_MIDDLE_SUCCESS_CAPTURE_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "upstream_middle_success_to_p5_capture_bundle.npz"
)


@configclass
class Dex5V9P4P5PhysicsEnvCfg(Dex5RecaptureP3PhysicsEnvCfg):
    """Initial PhysX task: braked P4 to thumb/index P5 pinch."""

    bundle_path = V9_P4_P5_BUNDLE
    # Base observation + controller target error + five-stage one-hot.
    observation_space = 156
    episode_length_s = 0.60
    decimation = 1
    action_scale = 0.075
    min_curriculum_phase = 0
    max_curriculum_phase = 0
    use_bundle_source_target = True
    collision_proxy_radius_scales = (0.99, 1.07, 1.07, 0.84, 0.80)
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    reset_pen_rot_noise_rad = 0.0
    pen_static_friction_range = (0.6, 0.6)
    pen_dynamic_friction_range = (0.45, 0.45)
    control_delay_steps_max = 0
    penetration_threshold = 0.0005
    p5_contact_force_n = 0.03
    p5_clear_force_n = 0.03
    p5_clear_surface_distance_m = 0.002
    p5_max_penetration_m = 0.0005
    p5_max_position_drift_m = 0.015
    p5_max_axis_drift_deg = 15.0
    p5_max_angular_speed_rad_s = 12.0
    p5_max_retract_rmse_rad = 0.22
    p5_max_pinch_rmse_rad = 0.18
    stage_ring_max_rmse_rad = 0.30
    stage_joint_max_retract_rmse_rad = 0.34
    # 12 * 1/120 s = 100 ms. Full P5 may override this to 18 steps.
    p5_hold_steps = 12
    curriculum_stage = 4
    middle_clear_reward_weight = 8.0
    ring_clear_reward_weight = 8.0
    finger_unload_reward_weight = 8.0
    post_clear_pinch_reward_weight = 8.0
    middle_retract_reward_weight = 5.0
    ring_retract_reward_weight = 5.0
    middle_retract_error_weight = 0.0
    retract_error_reward_weight = 0.0
    pinch_error_reward_weight = 0.0


@configclass
class Dex5V9P4HoldPhysicsEnvCfg(Dex5V9P4P5PhysicsEnvCfg):
    curriculum_stage = 0
    reset_joint_pos_noise = 0.002
    reset_joint_vel_noise = 0.01
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.005


@configclass
class Dex5V9P4P5MiddleClearPhysicsEnvCfg(Dex5V9P4P5PhysicsEnvCfg):
    curriculum_stage = 1


@configclass
class Dex5V9P4P5MiddleClearPinchPhysicsEnvCfg(
    Dex5V9P4P5MiddleClearPhysicsEnvCfg
):
    post_clear_pinch_reward_weight = 24.0


@configclass
class Dex5V9P4P5MiddleClearTeacherPhysicsEnvCfg(
    Dex5V9P4P5MiddleClearPinchPhysicsEnvCfg
):
    bundle_path = V9_MIDDLE_CLEAR_TEACHER_BUNDLE


@configclass
class Dex5V9P4P5MiddleClearBlend75PhysicsEnvCfg(
    Dex5V9P4P5MiddleClearPinchPhysicsEnvCfg
):
    bundle_path = V9_MIDDLE_CLEAR_FIXED_TEACHER_BUNDLE
    teacher_prefill_steps = 72
    teacher_action_blend = 0.75


@configclass
class Dex5V9P4P5MiddleClearBlend50PhysicsEnvCfg(
    Dex5V9P4P5MiddleClearBlend75PhysicsEnvCfg
):
    teacher_action_blend = 0.50


@configclass
class Dex5V9P4P5MiddleClearBlend25PhysicsEnvCfg(
    Dex5V9P4P5MiddleClearBlend75PhysicsEnvCfg
):
    teacher_action_blend = 0.25


@configclass
class Dex5V9P4P5MiddleClearBlend0PhysicsEnvCfg(
    Dex5V9P4P5MiddleClearBlend75PhysicsEnvCfg
):
    teacher_action_blend = 0.0


@configclass
class Dex5V9P4P5RingClearPhysicsEnvCfg(Dex5V9P4P5PhysicsEnvCfg):
    curriculum_stage = 2
    bundle_path = V9_MIDDLE_CLEAR_LEARNED_CACHE
    ring_retract_reward_weight = 15.0


@configclass
class Dex5V9P4P5JointClearPhysicsEnvCfg(Dex5V9P4P5PhysicsEnvCfg):
    curriculum_stage = 3
    bundle_path = V9_RING_CLEAR_LEARNED_CACHE
    middle_retract_reward_weight = 12.0
    ring_retract_reward_weight = 12.0
    middle_retract_error_weight = 18.0


@configclass
class Dex5V9P4P5CapturePhysicsEnvCfg(Dex5V9P4P5PhysicsEnvCfg):
    curriculum_stage = 4
    bundle_path = V9_COLLISION_FREE_CAPTURE_CACHE
    retract_error_reward_weight = 20.0
    pinch_error_reward_weight = 12.0
    post_clear_pinch_reward_weight = 16.0


@configclass
class Dex5V9P4P5CaptureBlend75PhysicsEnvCfg(
    Dex5V9P4P5CapturePhysicsEnvCfg
):
    bundle_path = V9_COLLISION_FREE_CAPTURE_TEACHER
    teacher_prefill_steps = 96
    teacher_action_blend = 0.75


@configclass
class Dex5V9P4P5CaptureBlend50PhysicsEnvCfg(
    Dex5V9P4P5CaptureBlend75PhysicsEnvCfg
):
    teacher_action_blend = 0.50


@configclass
class Dex5V9P4P5CaptureBlend25PhysicsEnvCfg(
    Dex5V9P4P5CaptureBlend75PhysicsEnvCfg
):
    teacher_action_blend = 0.25


@configclass
class Dex5V9P4P5CaptureBlend0PhysicsEnvCfg(
    Dex5V9P4P5CaptureBlend75PhysicsEnvCfg
):
    teacher_action_blend = 0.0


@configclass
class Dex5V9P4P5UnifiedPhysicsEnvCfg(Dex5V9P4P5PhysicsEnvCfg):
    bundle_path = V9_P4_P5_UNIFIED_BUNDLE
    min_curriculum_phase = 0
    max_curriculum_phase = 4
    curriculum_stage = -1


@configclass
class Dex5V9P4P5UnifiedEasyDRPhysicsEnvCfg(
    Dex5V9P4P5UnifiedPhysicsEnvCfg
):
    reset_joint_pos_noise = 0.0008
    reset_joint_vel_noise = 0.010
    reset_pen_pos_noise = 0.00010
    reset_pen_vel_noise = 0.005
    reset_pen_rot_noise_rad = 0.00261799
    pen_static_friction_range = (0.50, 0.70)
    pen_dynamic_friction_range = (0.35, 0.55)


@configclass
class Dex5V9P4P5UnifiedMediumDRPhysicsEnvCfg(
    Dex5V9P4P5UnifiedEasyDRPhysicsEnvCfg
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
class Dex5V9P4P5UnifiedHardDRPhysicsEnvCfg(
    Dex5V9P4P5UnifiedMediumDRPhysicsEnvCfg
):
    reset_joint_pos_noise = 0.006
    reset_joint_vel_noise = 0.050
    reset_pen_pos_noise = 0.0008
    reset_pen_vel_noise = 0.030
    reset_pen_rot_noise_rad = 0.0349066
    pen_static_friction_range = (0.30, 0.90)
    pen_dynamic_friction_range = (0.20, 0.75)
    control_delay_steps_max = 2


@configclass
class Dex5V9P4P5UpstreamCapturePhysicsEnvCfg(
    Dex5V9P4P5CapturePhysicsEnvCfg
):
    bundle_path = V9_P4_P5_UPSTREAM_CAPTURE_BUNDLE


@configclass
class Dex5V9P4P5UpstreamMiddleClearPhysicsEnvCfg(
    Dex5V9P4P5MiddleClearPinchPhysicsEnvCfg
):
    bundle_path = V9_P4_P5_UPSTREAM_MIDDLE_BUNDLE
    episode_length_s = 0.90
    middle_retract_reward_weight = 12.0
    middle_retract_error_weight = 12.0


@configclass
class Dex5V9P4P5CurrentP4MiddleClearPhysicsEnvCfg(
    Dex5V9P4P5UpstreamMiddleClearPhysicsEnvCfg
):
    """Middle-clear curriculum reset from the current 83.45% P3-to-P4 cascade."""

    bundle_path = V9_CURRENT_P4_P5_MIDDLE_BUNDLE


@configclass
class Dex5V9P4P5CurrentP4CapturePhysicsEnvCfg(
    Dex5V9P4P5CapturePhysicsEnvCfg
):
    """P5 capture reset from successful current-P4 middle releases."""

    bundle_path = V9_CURRENT_P4_P5_CAPTURE_BUNDLE
    episode_length_s = 0.90


@configclass
class Dex5V9P4P5UpstreamCaptureEasyDRPhysicsEnvCfg(
    Dex5V9P4P5UpstreamCapturePhysicsEnvCfg
):
    reset_joint_pos_noise = 0.0008
    reset_joint_vel_noise = 0.010
    reset_pen_pos_noise = 0.00010
    reset_pen_vel_noise = 0.005
    reset_pen_rot_noise_rad = 0.00261799
    pen_static_friction_range = (0.50, 0.70)
    pen_dynamic_friction_range = (0.35, 0.55)


@configclass
class Dex5V9P4P5UpstreamLearnedCapturePhysicsEnvCfg(
    Dex5V9P4P5CapturePhysicsEnvCfg
):
    bundle_path = V9_UPSTREAM_MIDDLE_SUCCESS_CAPTURE_BUNDLE


@configclass
class Dex5V9P4P5UpstreamLearnedCaptureHold200PhysicsEnvCfg(
    Dex5V9P4P5UpstreamLearnedCapturePhysicsEnvCfg
):
    episode_length_s = 0.90
    p5_hold_steps = 24


@configclass
class Dex5V9P4P5UpstreamLearnedCaptureEasyDRPhysicsEnvCfg(
    Dex5V9P4P5UpstreamLearnedCapturePhysicsEnvCfg
):
    reset_joint_pos_noise = 0.0008
    reset_joint_vel_noise = 0.010
    reset_pen_pos_noise = 0.00010
    reset_pen_vel_noise = 0.005
    reset_pen_rot_noise_rad = 0.00261799
    pen_static_friction_range = (0.50, 0.70)
    pen_dynamic_friction_range = (0.35, 0.55)


@configclass
class Dex5V9P4P5UpstreamLearnedCaptureMediumDRPhysicsEnvCfg(
    Dex5V9P4P5UpstreamLearnedCaptureEasyDRPhysicsEnvCfg
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
class Dex5V9P4P5UpstreamLearnedCaptureEasyDRHold200PhysicsEnvCfg(
    Dex5V9P4P5UpstreamLearnedCaptureEasyDRPhysicsEnvCfg
):
    episode_length_s = 0.90
    p5_hold_steps = 24


@configclass
class Dex5V9P4P5UpstreamLearnedCaptureMediumDRHold200PhysicsEnvCfg(
    Dex5V9P4P5UpstreamLearnedCaptureMediumDRPhysicsEnvCfg
):
    episode_length_s = 0.90
    p5_hold_steps = 24


class Dex5V9P4P5PhysicsEnv(Dex5RecaptureP3PhysicsEnv):
    cfg: Dex5V9P4P5PhysicsEnvCfg

    def _load_bundle(self) -> None:
        super()._load_bundle()
        if int(self.cfg.curriculum_stage) < 0:
            self.p5_stage_id_bank = self.phase_id_bank.clone().clamp(0, 4)
        # Recapture observations still use a separate four-class phase one-hot.
        # P4/P5 curriculum stages live in p5_stage_ids and may validly equal 4.
        self.phase_id_bank.zero_()
        self.policy_targets = torch.zeros(
            (self.num_envs, 20), dtype=torch.float32, device=self.device
        )
        self.p5_stage_ids = torch.full(
            (self.num_envs,),
            int(self.cfg.curriculum_stage),
            dtype=torch.long,
            device=self.device,
        )
        self.stage_hold_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.p5_hold_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_p5_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.completed_p5_episodes = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_p5_successes = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_stage_episodes = torch.zeros(
            5, dtype=torch.long, device=self.device
        )
        self.completed_stage_successes = torch.zeros(
            5, dtype=torch.long, device=self.device
        )
        self.p5_done_drop_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.p5_done_drift_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.p5_done_penetration_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.p5_done_max_penetration = torch.zeros((), device=self.device)
        self.p5_done_max_penetration_by_finger = torch.zeros(
            5, device=self.device
        )
        self.p5_done_invalid_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        delay_horizon = int(self.cfg.control_delay_steps_max) + 1
        self.action_delay_buffer = torch.zeros(
            (self.num_envs, delay_horizon, self.cfg.action_space),
            dtype=torch.float32,
            device=self.device,
        )
        self.control_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        if self.action_delay_buffer.shape[1] <= 1:
            return
        self.action_delay_buffer = torch.roll(
            self.action_delay_buffer, shifts=1, dims=1
        )
        self.action_delay_buffer[:, 0] = self.actions
        rows = torch.arange(self.num_envs, device=self.device)
        self.actions = self.action_delay_buffer[rows, self.control_delay_steps]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        controller_error = self.policy_targets - self.hand_qpos
        stage_one_hot = torch.nn.functional.one_hot(
            self.p5_stage_ids, num_classes=5
        ).float()
        observations["policy"] = torch.cat(
            (observations["policy"], controller_error, stage_one_hot), dim=-1
        )
        return observations

    def _stage_action_mask(self) -> torch.Tensor:
        stage = self.p5_stage_ids
        mask = torch.zeros((self.num_envs, 20), device=self.device)
        hold = stage == 0
        middle = stage == 1
        ring = stage == 2
        joint = stage == 3
        capture = stage == 4
        pinch_scale = (
            0.35 * (hold | middle | capture).float()
        )
        mask[:, :4] = pinch_scale.unsqueeze(-1)
        mask[:, 16:20] = pinch_scale.unsqueeze(-1)
        mask[:, 4:8] = (
            middle | joint
        ).float().unsqueeze(-1)
        mask[:, 8:12] = (
            ring
        ).float().unsqueeze(-1)
        # Pinky is already clear in the joint cache and is not part of P5 capture.
        mask[:, 12:16] = 0.0
        return mask

    def _apply_action(self) -> None:
        self.policy_targets += (
            0.25 * self.cfg.action_scale * self.actions * self._stage_action_mask()
        )
        self.policy_targets = torch.maximum(
            torch.minimum(self.policy_targets, self.joint_upper), self.joint_lower
        )
        self.processed_targets = self.policy_targets
        self.hand.set_joint_position_target(
            self.processed_targets, joint_ids=self.dex5_joint_ids
        )

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        failed, _ = self._compute_dones()

        contacts = self.contact_force_norm > self.cfg.p5_contact_force_n
        thumb_index = contacts[:, :2].all(dim=-1)
        fingertip_surface_gap = (
            self.tip_surface_dist - self.fingertip_radii
        )
        middle_clear = (
            self.contact_force_norm[:, 2] < self.cfg.p5_clear_force_n
        ) | (
            fingertip_surface_gap[:, 2]
            > self.cfg.p5_clear_surface_distance_m
        )
        ring_clear = (
            self.contact_force_norm[:, 3] < self.cfg.p5_clear_force_n
        ) | (
            fingertip_surface_gap[:, 3]
            > self.cfg.p5_clear_surface_distance_m
        )
        middle_ring_clear = middle_clear & ring_clear
        pinky_clear = (
            self.contact_force_norm[:, 4] < self.cfg.p5_clear_force_n
        ) | (
            fingertip_surface_gap[:, 4]
            > self.cfg.p5_clear_surface_distance_m
        )
        pinch_rmse = torch.sqrt(
            torch.mean(
                torch.cat(
                    (
                        self.hand_qpos[:, :4] - self.target_hand_qpos[:, :4],
                        self.hand_qpos[:, 16:20] - self.target_hand_qpos[:, 16:20],
                    ),
                    dim=-1,
                ).square(),
                dim=-1,
            )
        )
        retract_rmse = torch.sqrt(
            torch.mean(
                (self.hand_qpos[:, 4:12] - self.target_hand_qpos[:, 4:12]).square(),
                dim=-1,
            )
        )
        middle_rmse = torch.sqrt(
            torch.mean(
                (self.hand_qpos[:, 4:8] - self.target_hand_qpos[:, 4:8]).square(),
                dim=-1,
            )
        )
        ring_rmse = torch.sqrt(
            torch.mean(
                (self.hand_qpos[:, 8:12] - self.target_hand_qpos[:, 8:12]).square(),
                dim=-1,
            )
        )
        pinky_rmse = torch.sqrt(
            torch.mean(
                (self.hand_qpos[:, 12:16] - self.target_hand_qpos[:, 12:16]).square(),
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
        tau = torch.clamp(
            self.episode_length_buf.float()
            / max(float(self.max_episode_length - 1), 1.0),
            0.0,
            1.0,
        )
        final_gate = torch.clamp((tau - 0.45) / 0.30, 0.0, 1.0)
        instantaneous_p5 = (
            thumb_index
            & middle_ring_clear
            & pinky_clear
            & (pinch_rmse <= self.cfg.p5_max_pinch_rmse_rad)
            & (retract_rmse <= self.cfg.p5_max_retract_rmse_rad)
            & (position_drift <= self.cfg.p5_max_position_drift_m)
            & (
                axis_drift
                <= torch.deg2rad(
                    torch.full_like(axis_drift, self.cfg.p5_max_axis_drift_deg)
                )
            )
            & (angular_speed <= self.cfg.p5_max_angular_speed_rad_s)
            & (self.episode_max_penetration <= self.cfg.p5_max_penetration_m)
            & ~failed
        )
        safe_hold = (
            thumb_index
            & (position_drift <= self.cfg.p5_max_position_drift_m)
            & (
                axis_drift
                <= torch.deg2rad(
                    torch.full_like(axis_drift, self.cfg.p5_max_axis_drift_deg)
                )
            )
            & (angular_speed <= self.cfg.p5_max_angular_speed_rad_s)
            & (self.episode_max_penetration <= self.cfg.p5_max_penetration_m)
            & ~failed
        )
        partial_p5 = (
            safe_hold
            & middle_ring_clear
            & (pinch_rmse <= self.cfg.p5_max_pinch_rmse_rad)
            & (retract_rmse <= 0.32)
        )
        stage_eligible = torch.where(
            self.p5_stage_ids == 0,
            safe_hold,
            torch.where(
                self.p5_stage_ids == 1,
                safe_hold & middle_clear,
                torch.where(
                    self.p5_stage_ids == 2,
                    safe_hold
                    & ring_clear
                    & (ring_rmse <= self.cfg.stage_ring_max_rmse_rad),
                    torch.where(
                        self.p5_stage_ids == 3,
                        safe_hold
                        & middle_ring_clear
                        & (
                            retract_rmse
                            <= self.cfg.stage_joint_max_retract_rmse_rad
                        ),
                        instantaneous_p5,
                    ),
                ),
            ),
        )
        self.stage_hold_count = torch.where(
            stage_eligible,
            self.stage_hold_count + 1,
            torch.zeros_like(self.stage_hold_count),
        )
        self.p5_hold_count = torch.where(
            instantaneous_p5,
            self.p5_hold_count + 1,
            torch.zeros_like(self.p5_hold_count),
        )
        success = self.stage_hold_count >= self.cfg.p5_hold_steps
        self.episode_p5_success |= success

        thumb_quality = torch.clamp(
            self.contact_force_norm[:, 0] / 0.10, 0.0, 1.0
        )
        index_quality = torch.clamp(
            self.contact_force_norm[:, 1] / 0.10, 0.0, 1.0
        )
        pinch_balance = torch.exp(
            -3.0
            * torch.abs(
                self.contact_force_norm[:, 0] - self.contact_force_norm[:, 1]
            )
        )
        thumb_load_quality = torch.clamp(
            self.contact_force_norm[:, 0] / 0.35, 0.0, 1.0
        )
        index_load_quality = torch.clamp(
            self.contact_force_norm[:, 1] / 0.35, 0.0, 1.0
        )
        pinch_load_quality = torch.minimum(
            thumb_load_quality, index_load_quality
        )
        retract_score = torch.exp(-5.0 * retract_rmse)
        middle_retract_score = torch.exp(-3.0 * middle_rmse)
        ring_retract_score = torch.exp(-3.0 * ring_rmse)
        middle_unload_score = torch.exp(-2.0 * self.contact_force_norm[:, 2])
        ring_unload_score = torch.exp(-2.0 * self.contact_force_norm[:, 3])
        pinch_pose_score = torch.exp(-6.0 * pinch_rmse)
        axis_score = torch.exp(-8.0 * axis_drift)
        position_score = torch.exp(-100.0 * position_drift)
        speed_score = torch.exp(-0.16 * angular_speed)
        middle_ring_load = torch.clamp(
            self.contact_force_norm[:, 2:4].sum(dim=-1) / 0.20, 0.0, 2.0
        )
        penetration_excess = torch.relu(
            (self.episode_max_penetration - 0.00025) / 0.00025
        )
        force_excess = torch.relu((peak_force - 2.0) / 3.0)
        hold_progress = torch.clamp(
            self.stage_hold_count.float() / self.cfg.p5_hold_steps, 0.0, 1.0
        )
        stage_middle = (
            (self.p5_stage_ids == 1)
            | (self.p5_stage_ids == 3)
            | (self.p5_stage_ids == 4)
        ).float()
        stage_ring = (
            (self.p5_stage_ids == 2)
            | (self.p5_stage_ids == 3)
            | (self.p5_stage_ids == 4)
        ).float()
        stage_capture = (self.p5_stage_ids == 4).float()
        post_clear_pinch = (
            middle_ring_clear.float() * thumb_quality * index_quality
        )
        pinch_gate = 0.20 + 0.80 * pinch_load_quality
        unsafe_middle_clear = (
            stage_middle * middle_clear.float() * (~thumb_index).float()
        )
        unsafe_ring_clear = (
            stage_ring * ring_clear.float() * (~thumb_index).float()
        )

        terms = {
            "alive": torch.full_like(tau, 0.05),
            "thumb_contact": 4.0 * thumb_quality,
            "index_contact": 4.0 * index_quality,
            "pinch_product": 5.0 * thumb_quality * index_quality,
            "pinch_takeover": 8.0 * pinch_load_quality,
            "pinch_balance": 1.0 * pinch_balance,
            "pinch_pose": 2.0 * stage_capture * pinch_pose_score,
            "pinch_error": -self.cfg.pinch_error_reward_weight
            * stage_capture
            * pinch_rmse,
            "middle_retract": self.cfg.middle_retract_reward_weight
            * stage_middle
            * middle_retract_score,
            "middle_retract_error": -self.cfg.middle_retract_error_weight
            * stage_middle
            * middle_rmse,
            "ring_retract": self.cfg.ring_retract_reward_weight
            * stage_ring
            * ring_retract_score,
            "middle_unload": self.cfg.finger_unload_reward_weight
            * stage_middle
            * middle_unload_score
            * pinch_gate,
            "ring_unload": self.cfg.finger_unload_reward_weight
            * stage_ring
            * ring_unload_score
            * pinch_gate,
            "middle_clear": self.cfg.middle_clear_reward_weight
            * stage_middle
            * middle_clear.float()
            * thumb_index.float(),
            "ring_clear": self.cfg.ring_clear_reward_weight
            * stage_ring
            * ring_clear.float()
            * thumb_index.float(),
            "post_clear_pinch": self.cfg.post_clear_pinch_reward_weight
            * post_clear_pinch,
            "unsafe_clear": -12.0 * (unsafe_middle_clear + unsafe_ring_clear),
            "middle_ring_retract": 2.0 * stage_capture * retract_score,
            "retract_error": -self.cfg.retract_error_reward_weight
            * stage_capture
            * retract_rmse,
            "middle_ring_load": -4.0 * final_gate * middle_ring_load,
            "pinky_clear": 0.5 * pinky_clear.float(),
            "pen_position": 3.0 * position_score,
            "pen_axis": 4.0 * axis_score,
            "angular_brake": 3.0 * speed_score,
            "partial_p5": 10.0 * stage_capture * partial_p5.float(),
            "instantaneous_p5": 12.0 * stage_capture * instantaneous_p5.float(),
            "hold": 8.0 * hold_progress,
            "terminal_success": 40.0 * success.float(),
            "penetration": -12.0 * penetration_excess,
            "force_excess": -1.5 * force_excess,
            "action_l2": -0.03 * torch.linalg.norm(self.actions, dim=-1),
            "action_smoothness": -0.05
            * torch.linalg.norm(self.actions - self.previous_actions, dim=-1),
            "failure": -10.0 * failed.float(),
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
        self.extras["log"] = {
            f"reward/{name}": value.mean() for name, value in terms.items()
        }
        self.extras["log"].update(
            {
                "metric/p5_success_ratio": success.float().mean(),
                "metric/p5_instantaneous_ratio": instantaneous_p5.float().mean(),
                "metric/p5_partial_ratio": partial_p5.float().mean(),
                "metric/p5_hold_ms": self.p5_hold_count.float().mean()
                * self.cfg.decimation
                * self.cfg.sim.dt
                * 1000.0,
                "metric/thumb_contact_ratio": contacts[:, 0].float().mean(),
                "metric/index_contact_ratio": contacts[:, 1].float().mean(),
                "metric/thumb_force_n": self.contact_force_norm[:, 0].mean(),
                "metric/index_force_n": self.contact_force_norm[:, 1].mean(),
                "metric/pinch_load_quality": pinch_load_quality.mean(),
                "metric/middle_contact_ratio": contacts[:, 2].float().mean(),
                "metric/ring_contact_ratio": contacts[:, 3].float().mean(),
                "metric/pinky_contact_ratio": contacts[:, 4].float().mean(),
                "metric/middle_ring_clear_ratio": middle_ring_clear.float().mean(),
                "metric/middle_clear_ratio": middle_clear.float().mean(),
                "metric/ring_clear_ratio": ring_clear.float().mean(),
                "metric/stage_eligible_ratio": stage_eligible.float().mean(),
                "metric/stage_hold_ms": self.stage_hold_count.float().mean()
                * self.cfg.decimation
                * self.cfg.sim.dt
                * 1000.0,
                "metric/middle_rmse_rad": middle_rmse.mean(),
                "metric/ring_rmse_rad": ring_rmse.mean(),
                "metric/middle_tip_surface_dist_mm": fingertip_surface_gap[
                    :, 2
                ].mean()
                * 1000.0,
                "metric/ring_tip_surface_dist_mm": fingertip_surface_gap[:, 3].mean()
                * 1000.0,
                "metric/middle_force_n": self.contact_force_norm[:, 2].mean(),
                "metric/ring_force_n": self.contact_force_norm[:, 3].mean(),
                "metric/pinch_rmse_rad": pinch_rmse.mean(),
                "metric/retract_rmse_rad": retract_rmse.mean(),
                "metric/pinky_rmse_rad": pinky_rmse.mean(),
                "metric/pen_drift_mm": position_drift.mean() * 1000.0,
                "metric/axis_drift_deg": axis_drift.mean()
                * (180.0 / torch.pi),
                "metric/angular_speed_rad_s": angular_speed.mean(),
                "metric/max_penetration_mm": self.episode_max_penetration.mean()
                * 1000.0,
                "metric/peak_force_n": peak_force.mean(),
            }
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        height_drop = self.init_pen_pos[:, 2] - self.pen_pos[:, 2]
        drift = torch.linalg.norm(self.pen_pos - self.init_pen_pos, dim=-1)
        invalid = ~torch.isfinite(self.pen_pos).all(dim=-1) | ~torch.isfinite(
            self.hand_qpos
        ).all(dim=-1)
        drop = height_drop > self.cfg.drop_threshold
        drift_failure = drift > self.cfg.drift_threshold
        penetration = self.episode_max_penetration > self.cfg.penetration_threshold
        failed = drop | drift_failure | penetration | invalid
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.p5_done_drop_count += drop.sum()
        self.p5_done_drift_count += drift_failure.sum()
        self.p5_done_penetration_count += penetration.sum()
        if torch.any(penetration):
            self.p5_done_max_penetration = torch.maximum(
                self.p5_done_max_penetration,
                self.episode_max_penetration[penetration].max(),
            )
            self.p5_done_max_penetration_by_finger = torch.maximum(
                self.p5_done_max_penetration_by_finger,
                self.episode_max_penetration_by_finger[penetration].max(dim=0).values,
            )
        self.p5_done_invalid_count += invalid.sum()
        return failed, time_out

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids_tensor = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        if hasattr(self, "episode_p5_success"):
            completed = self.episode_length_buf[env_ids_tensor] > 0
            self.completed_p5_episodes += completed.sum()
            self.completed_p5_successes += (
                completed & self.episode_p5_success[env_ids_tensor]
            ).sum()
            completed_stages = self.p5_stage_ids[env_ids_tensor]
            for stage_id in range(5):
                stage_completed = completed & (completed_stages == stage_id)
                self.completed_stage_episodes[stage_id] += stage_completed.sum()
                self.completed_stage_successes[stage_id] += (
                    stage_completed & self.episode_p5_success[env_ids_tensor]
                ).sum()
        super()._reset_idx(env_ids_tensor)
        rotation_noise = float(self.cfg.reset_pen_rot_noise_rad)
        if rotation_noise > 0.0:
            count = len(env_ids_tensor)
            rotvec = rotation_noise * torch.randn(
                (count, 3), device=self.device
            )
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
            self.pen.root_physx_view.set_material_properties(
                materials, material_env_ids
            )

        max_delay = int(self.cfg.control_delay_steps_max)
        if max_delay > 0:
            self.control_delay_steps[env_ids_tensor] = torch.randint(
                0, max_delay + 1, (len(env_ids_tensor),), device=self.device
            )
        else:
            self.control_delay_steps[env_ids_tensor] = 0
        initial_actions = self.actions[env_ids_tensor].unsqueeze(1)
        self.action_delay_buffer[env_ids_tensor] = initial_actions.expand(
            -1, self.action_delay_buffer.shape[1], -1
        )
        self.policy_targets[env_ids_tensor] = self.source_targets[env_ids_tensor]
        self.p5_hold_count[env_ids_tensor] = 0
        self.stage_hold_count[env_ids_tensor] = 0
        if int(self.cfg.curriculum_stage) < 0:
            self.p5_stage_ids[env_ids_tensor] = self.p5_stage_id_bank[
                self.current_bank_ids[env_ids_tensor]
            ]
        else:
            self.p5_stage_ids[env_ids_tensor] = int(self.cfg.curriculum_stage)
        self.episode_p5_success[env_ids_tensor] = False
