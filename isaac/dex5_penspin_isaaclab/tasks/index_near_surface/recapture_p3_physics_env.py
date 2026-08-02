from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.utils import configclass

from .index_near_surface_physics_env import (
    Dex5IndexNearSurfacePhysicsEnv,
    Dex5IndexNearSurfacePhysicsEnvCfg,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_STAGED_BUNDLE_PATH = str(
    ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_recapture_p3_bundle.npz"
)
EARLIER_PRE_RING_RECAPTURE_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_earlier_pre_ring_index_recapture_bundle.npz"
)
EARLIER_TO_P3_ENTRY_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_earlier_to_p3_entry_bundle.npz"
)
V9_P3_DYNAMIC_HOLD_TEACHER_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_p3_dynamic_hold_local_teacher_bundle.npz"
)
V9_P3_STRICT_HOLD_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/"
    "v9_p3_strict_best_fixed4096_seed761.npz"
)
V9_P1_P2_INITIAL_BUNDLE = str(
    ROOT
    / "dex5_penspin/reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
    "v9_p1_p2_initial_bundle.npz"
)


@configclass
class Dex5RecaptureP3PhysicsEnvCfg(Dex5IndexNearSurfacePhysicsEnvCfg):
    observation_space = 131
    episode_length_s = 0.12
    bundle_path = DEFAULT_STAGED_BUNDLE_PATH
    action_scale = 0.055
    reset_joint_pos_noise = 0.002
    reset_joint_vel_noise = 0.010
    reset_pen_pos_noise = 0.0004
    reset_pen_vel_noise = 0.005
    min_curriculum_phase = 0
    max_curriculum_phase = 3
    p3_planar_target_deg = -10.0
    p3_success_deflection_deg = 7.0
    use_bundle_source_target = False


@configclass
class Dex5IndexContactPhysicsEnvCfg(Dex5RecaptureP3PhysicsEnvCfg):
    episode_length_s = 0.06
    collision_proxy_radius_scales = (0.90, 1.00, 0.85, 0.85, 0.80)
    max_curriculum_phase = 0


@configclass
class Dex5EarlierPreRingIndexContactPhysicsEnvCfg(Dex5IndexContactPhysicsEnvCfg):
    """Real PhysX index/thumb recapture from the earlier 0.22-0.30s window."""

    bundle_path = EARLIER_PRE_RING_RECAPTURE_BUNDLE
    episode_length_s = 0.12
    action_scale = 0.075
    reset_joint_pos_noise = 0.001
    reset_joint_vel_noise = 0.005
    reset_pen_pos_noise = 0.0002
    reset_pen_vel_noise = 0.002


@configclass
class Dex5IndexRecapturePhysicsEnvCfg(Dex5RecaptureP3PhysicsEnvCfg):
    collision_proxy_radius_scales = (1.00, 1.00, 0.90, 0.85, 0.80)
    min_curriculum_phase = 1
    max_curriculum_phase = 1


@configclass
class Dex5RingTouchPhysicsEnvCfg(Dex5RecaptureP3PhysicsEnvCfg):
    episode_length_s = 0.10
    collision_proxy_radius_scales = (1.00, 1.00, 0.85, 1.00, 0.80)
    min_curriculum_phase = 2
    max_curriculum_phase = 2


@configclass
class Dex5P3EntryPhysicsEnvCfg(Dex5RecaptureP3PhysicsEnvCfg):
    episode_length_s = 0.20
    action_scale = 0.090
    # Small PhysX proxy calibration for the index/middle fingertips. The
    # diagnostic penetration gate remains at 4 mm, so this does not turn
    # geometric proximity into an unconstrained contact.
    collision_proxy_radius_scales = (1.00, 1.08, 1.08, 0.85, 0.80)
    min_curriculum_phase = 3
    max_curriculum_phase = 3


@configclass
class Dex5EarlierToP3EntryPhysicsEnvCfg(Dex5P3EntryPhysicsEnvCfg):
    """Dynamic middle recapture and 7-degree P3-entry bridge."""

    bundle_path = EARLIER_TO_P3_ENTRY_BUNDLE
    episode_length_s = 0.20
    action_scale = 0.080
    p3_planar_target_deg = -7.0
    p3_success_deflection_deg = 5.0
    # Curriculum level 0: establish a reproducible PhysX contact entry before
    # restoring pose/velocity randomization in later levels.
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    use_bundle_source_target = True
    p3_hold_steps = 0
    teacher_prefill_steps = 0
    teacher_action_blend = 1.0


@configclass
class Dex5P3HoldPhysicsEnvCfg(Dex5P3EntryPhysicsEnvCfg):
    """P3 entry plus an explicit 100 ms simultaneous T/I/M hold gate."""

    episode_length_s = 0.32
    p3_hold_steps = 6


@configclass
class Dex5V9P3DynamicHoldTeacherPhysicsEnvCfg(Dex5P3HoldPhysicsEnvCfg):
    """Reset-matched 100-ms dynamic canonical-P3 hold teacher."""

    bundle_path = V9_P3_DYNAMIC_HOLD_TEACHER_BUNDLE
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    use_bundle_source_target = True
    teacher_prefill_steps = 2


@configclass
class Dex5V9P3StrictHoldPhysicsEnvCfg(Dex5P3HoldPhysicsEnvCfg):
    """Canonical v9 P3 brake and 100-ms T/I/M hold task."""

    bundle_path = V9_P3_STRICT_HOLD_BUNDLE
    observation_space = 134
    episode_length_s = 0.24
    action_scale = 0.090
    collision_proxy_radius_scales = (0.99, 1.07, 1.07, 0.84, 0.80)
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    use_bundle_source_target = False
    teacher_prefill_steps = 6
    p3_hold_steps = 6
    penetration_threshold = 0.0005
    strict_axis_drift_deg = 8.0
    strict_max_angular_speed_rad_s = 12.0
    strict_max_position_drift_m = 0.015
    strict_min_contact_force_n = 0.03
    strict_max_contact_force_n = 5.0


@configclass
class Dex5EarlierToP3EntryTeacherPhysicsEnvCfg(Dex5EarlierToP3EntryPhysicsEnvCfg):
    """Short teacher-prefill variant for BC/RL handoff diagnostics."""

    teacher_prefill_steps = 3


@configclass
class Dex5V9P1P2ResetBankPhysicsEnvCfg(Dex5RecaptureP3PhysicsEnvCfg):
    """Natural-hold screening environment for current-semantics P1/P2 resets."""

    bundle_path = V9_P1_P2_INITIAL_BUNDLE
    observation_space = 131
    episode_length_s = 0.12
    action_scale = 0.0
    min_curriculum_phase = 1
    max_curriculum_phase = 2
    use_bundle_source_target = True
    teacher_prefill_steps = 0
    reset_joint_pos_noise = 0.003
    reset_joint_vel_noise = 0.015
    reset_pen_pos_noise = 0.0004
    reset_pen_vel_noise = 0.003
    penetration_threshold = 0.0005
    drop_threshold = 0.020
    drift_threshold = 0.025
    collision_proxy_radius_scales = (0.92, 0.92, 0.92, 0.90, 0.82)


class Dex5RecaptureP3PhysicsEnv(Dex5IndexNearSurfacePhysicsEnv):
    cfg: Dex5RecaptureP3PhysicsEnvCfg

    def _load_bundle(self) -> None:
        data = np.load(self.cfg.bundle_path)
        phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)
        selected = (phase_ids >= self.cfg.min_curriculum_phase) & (phase_ids <= self.cfg.max_curriculum_phase)
        source_qpos = np.asarray(data["source_qpos"], dtype=np.float32)[selected]
        target_qpos = np.asarray(data["target_qpos"], dtype=np.float32)[selected]

        self.source_qpos_bank = torch.as_tensor(source_qpos, device=self.device)
        self.source_qvel_bank = torch.as_tensor(
            np.asarray(data["source_qvel"], dtype=np.float32)[selected], device=self.device
        )
        if self.cfg.use_bundle_source_target:
            source_target = np.asarray(data["source_target_qpos"], dtype=np.float32)[selected]
        else:
            # MuJoCo cached actuator commands can differ materially from measured qpos.
            source_target = source_qpos[:, :20]
        self.source_target_bank = torch.as_tensor(source_target, device=self.device)
        if "source_policy_action" in data:
            source_policy_action = np.asarray(data["source_policy_action"], dtype=np.float32)[selected]
        else:
            source_policy_action = np.zeros_like(source_target)
        self.source_policy_action_bank = torch.as_tensor(source_policy_action, device=self.device)
        if "source_teacher_actions" in data:
            teacher_actions = np.asarray(
                data["source_teacher_actions"], dtype=np.float32
            )[selected]
        else:
            # source_qpos has already been phase-filtered. Building the fallback
            # at the full bundle size and applying `selected` again used to
            # produce a 512-vs-1040 boolean-index mismatch for legacy bundles.
            teacher_actions = np.zeros(
                (source_qpos.shape[0], 1, 20), dtype=np.float32
            )
        self.teacher_action_bank = torch.as_tensor(teacher_actions, device=self.device)
        self.teacher_action_horizon = self.teacher_action_bank.shape[1]
        self.target_hand_bank = torch.as_tensor(target_qpos[:, :20], device=self.device)
        self.target_pen_pos_bank = torch.as_tensor(target_qpos[:, 20:23], device=self.device)
        self.target_pen_quat_bank = torch.as_tensor(target_qpos[:, 23:27], device=self.device)
        self.phase_id_bank = torch.as_tensor(phase_ids[selected], dtype=torch.long, device=self.device)
        self.phase_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.bank_size = self.source_qpos_bank.shape[0]
        self.episode_dynamic_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_max_directional_deflection = torch.zeros(self.num_envs, device=self.device)
        self.completed_p3_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_p3_successes = torch.zeros((), dtype=torch.long, device=self.device)
        self.completed_p3_deflection_sum = torch.zeros((), device=self.device)
        self.p3_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.required_contact_masks = torch.tensor(
            (
                (0.0, 1.0, 0.0, 0.0, 0.0),  # first explicit index contact
                (1.0, 1.0, 1.0, 0.0, 0.0),  # thumb, index, middle full recapture
                (1.0, 1.0, 0.0, 1.0, 0.0),  # thumb, index, ring touch
                (1.0, 1.0, 1.0, 0.0, 0.0),  # thumb, index, middle P3 entry
            ),
            device=self.device,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        phase_one_hot = torch.nn.functional.one_hot(self.phase_ids, num_classes=4).float()
        planar_deflection = self._signed_planar_deflection()
        observations["policy"] = torch.cat(
            (observations["policy"], phase_one_hot, planar_deflection.unsqueeze(-1)), dim=-1
        )
        return observations

    def _signed_planar_deflection(self) -> torch.Tensor:
        initial_yaw = torch.atan2(self.init_pen_axis[:, 1], self.init_pen_axis[:, 0])
        current_yaw = torch.atan2(self.pen_axis[:, 1], self.pen_axis[:, 0])
        return torch.atan2(torch.sin(current_yaw - initial_yaw), torch.cos(current_yaw - initial_yaw))

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        failed, _ = self._compute_dones()

        hand_rmse = torch.sqrt(torch.mean((self.hand_qpos - self.target_hand_qpos) ** 2, dim=-1))
        pen_pos_error = torch.linalg.norm(self.pen_pos - self.target_pen_pos, dim=-1)
        quat_dot = torch.abs(torch.sum(self.pen_quat * self.target_pen_quat, dim=-1)).clamp(0.0, 1.0)
        pen_angle_error = 2.0 * torch.acos(quat_dot)
        planar_deflection = self._signed_planar_deflection()
        p3_target = torch.deg2rad(torch.full_like(planar_deflection, self.cfg.p3_planar_target_deg))
        p3_direction = torch.sign(p3_target)
        directional_deflection = p3_direction * planar_deflection
        p3_target_magnitude = torch.abs(p3_target)
        p3_planar_error = torch.abs(planar_deflection - p3_target)
        p3_success_deflection = torch.deg2rad(
            torch.full_like(planar_deflection, self.cfg.p3_success_deflection_deg)
        )

        contacts = (self.contact_force_norm > 0.03).float()
        required = self.required_contact_masks[self.phase_ids]
        required_contact_ratio = torch.sum(contacts * required, dim=-1) / torch.sum(required, dim=-1)
        pinky_contact = contacts[:, 4]
        force_excess = torch.clamp((self.contact_force_norm.max(dim=-1).values - 4.0) / 6.0, 0.0, 1.0)
        penetration_excess = torch.clamp((self.episode_max_penetration - 0.0015) / 0.003, 0.0, 1.0)
        drift = torch.linalg.norm(self.pen_pos - self.init_pen_pos, dim=-1)
        height_drop = torch.clamp(self.init_pen_pos[:, 2] - self.pen_pos[:, 2], min=0.0)
        tau = torch.clamp(
            self.episode_length_buf.float() / max(float(self.max_episode_length - 1), 1.0), 0.0, 1.0
        )
        terminal_gate = torch.clamp((tau - 0.55) / 0.35, 0.0, 1.0)
        pose_success = (
            (hand_rmse < 0.16)
            & (pen_pos_error < 0.015)
            & (pen_angle_error < 0.40)
            & (required_contact_ratio >= 2.0 / 3.0)
            & (self.episode_max_penetration < 0.004)
        )
        phase_three = (self.phase_ids == 3).float()
        p3_contact_gate = 0.20 + 0.80 * required_contact_ratio
        close_target = torch.full_like(self.tip_surface_dist[:, :3], 0.0115)
        close_gap = torch.relu(self.tip_surface_dist[:, :3] - close_target)
        fingertip_close_score = torch.exp(-120.0 * close_gap).mean(dim=-1)
        dynamic_p3_success = (
            (hand_rmse < 0.20)
            & (pen_pos_error < 0.015)
            & (directional_deflection >= p3_success_deflection)
            & (required_contact_ratio >= 2.0 / 3.0)
            & (self.episode_max_penetration < 0.004)
        )
        p3_hold_eligible = dynamic_p3_success & (self.phase_ids == 3)
        self.p3_hold_count = torch.where(
            p3_hold_eligible,
            self.p3_hold_count + 1,
            torch.zeros_like(self.p3_hold_count),
        )
        if getattr(self.cfg, "p3_hold_steps", 0) > 0:
            dynamic_p3_success = p3_hold_eligible & (
                self.p3_hold_count >= self.cfg.p3_hold_steps
            )
        self.episode_dynamic_success |= dynamic_p3_success
        self.episode_max_directional_deflection = torch.maximum(
            self.episode_max_directional_deflection, directional_deflection
        )
        success = torch.where(self.phase_ids == 3, dynamic_p3_success, pose_success)
        hand_weight = torch.where(self.phase_ids == 1, 0.35, 1.50 + phase_three)
        pen_position_weight = 1.80 + 2.20 * phase_three
        pen_orientation_weight = 1.20 + 2.30 * phase_three
        phase_three_progress = phase_three * (
            p3_contact_gate
            * (
                6.0 * torch.clamp(directional_deflection / p3_target_magnitude, 0.0, 1.25)
                + 1.0 * (directional_deflection >= torch.deg2rad(torch.full_like(planar_deflection, 3.0))).float()
                + 1.5 * (directional_deflection >= torch.deg2rad(torch.full_like(planar_deflection, 5.0))).float()
                + 2.5 * (directional_deflection >= p3_success_deflection).float()
                + 2.0 * torch.exp(-12.0 * p3_planar_error)
            )
        )

        terms = {
            "alive": torch.full_like(hand_rmse, 0.08),
            "hand_target": hand_weight * torch.exp(-7.0 * hand_rmse),
            "pen_position": pen_position_weight * torch.exp(-70.0 * pen_pos_error),
            "pen_orientation": (1.0 - phase_three) * pen_orientation_weight * torch.exp(-2.5 * pen_angle_error),
            "phase_three_progress": phase_three_progress,
            "required_contacts": (1.60 + 3.00 * phase_three) * required_contact_ratio,
            "p3_fingertip_surface_close": 2.80 * phase_three * fingertip_close_score,
            "p3_contact_loss": -2.50 * phase_three * (1.0 - required_contact_ratio),
            "terminal_success": (3.00 + 7.00 * phase_three) * terminal_gate * success.float(),
            "penetration": -3.50 * penetration_excess,
            "force_excess": -0.60 * force_excess,
            "pinky_guard": -0.35 * (self.phase_ids > 0).float() * pinky_contact,
            "motion_guard": -0.70 * torch.clamp((drift + height_drop) / 0.045, 0.0, 1.0),
            "action_l2": -0.04 * torch.linalg.norm(self.actions, dim=-1),
            "action_smoothness": -0.06 * torch.linalg.norm(self.actions - self.previous_actions, dim=-1),
            "failure": -5.00 * failed.float(),
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
        self.extras["log"] = {f"reward/{name}": value.mean() for name, value in terms.items()}
        self.extras["log"].update(
            {
                "metric/hand_rmse_rad": hand_rmse.mean(),
                "metric/pen_pos_error_mm": pen_pos_error.mean() * 1000.0,
                "metric/pen_angle_error_deg": pen_angle_error.mean() * (180.0 / torch.pi),
                "metric/planar_deflection_deg": planar_deflection.mean() * (180.0 / torch.pi),
                "metric/directional_deflection_deg": directional_deflection.mean() * (180.0 / torch.pi),
                "metric/p3_target_error_deg": p3_planar_error.mean() * (180.0 / torch.pi),
                "metric/deflection_7deg_ratio": (
                    directional_deflection >= p3_success_deflection
                ).float().mean(),
                "metric/pen_angular_velocity_z": self.pen_angvel[:, 2].mean(),
                "metric/required_contact_ratio": required_contact_ratio.mean(),
                "metric/success_ratio": success.float().mean(),
                "metric/max_penetration_mm": self.episode_max_penetration.mean() * 1000.0,
                "metric/phase_mean": self.phase_ids.float().mean(),
            }
        )
        for phase in range(4):
            mask = self.phase_ids == phase
            self.extras["log"][f"metric/phase_{phase}_success"] = torch.where(
                mask, success.float(), torch.zeros_like(success.float())
            ).sum() / mask.float().sum().clamp_min(1.0)
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids_tensor = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if hasattr(self, "episode_dynamic_success"):
            completed = (self.episode_length_buf[env_ids_tensor] > 0) & (self.phase_ids[env_ids_tensor] == 3)
            self.completed_p3_episodes += completed.sum()
            self.completed_p3_successes += (
                completed & self.episode_dynamic_success[env_ids_tensor]
            ).sum()
            self.completed_p3_deflection_sum += torch.where(
                completed,
                self.episode_max_directional_deflection[env_ids_tensor],
                torch.zeros_like(self.episode_max_directional_deflection[env_ids_tensor]),
            ).sum()

        super()._reset_idx(env_ids_tensor)
        self.actions[env_ids_tensor] = self.source_policy_action_bank[
            self.current_bank_ids[env_ids_tensor]
        ]
        self.previous_actions[env_ids_tensor] = self.actions[env_ids_tensor]
        self.episode_dynamic_success[env_ids_tensor] = False
        self.episode_max_directional_deflection[env_ids_tensor] = 0.0
        self.p3_hold_count[env_ids_tensor] = 0


class Dex5IndexRecapturePhysicsEnv(Dex5RecaptureP3PhysicsEnv):
    cfg: Dex5IndexRecapturePhysicsEnvCfg


class Dex5IndexContactPhysicsEnv(Dex5RecaptureP3PhysicsEnv):
    cfg: Dex5IndexContactPhysicsEnvCfg


class Dex5EarlierPreRingIndexContactPhysicsEnv(Dex5IndexContactPhysicsEnv):
    cfg: Dex5EarlierPreRingIndexContactPhysicsEnvCfg


class Dex5RingTouchPhysicsEnv(Dex5RecaptureP3PhysicsEnv):
    cfg: Dex5RingTouchPhysicsEnvCfg


class Dex5P3EntryPhysicsEnv(Dex5RecaptureP3PhysicsEnv):
    cfg: Dex5P3EntryPhysicsEnvCfg


class Dex5EarlierToP3EntryPhysicsEnv(Dex5P3EntryPhysicsEnv):
    cfg: Dex5EarlierToP3EntryPhysicsEnvCfg


class Dex5EarlierToP3EntryTeacherPhysicsEnv(Dex5EarlierToP3EntryPhysicsEnv):
    cfg: Dex5EarlierToP3EntryTeacherPhysicsEnvCfg


class Dex5P3HoldPhysicsEnv(Dex5P3EntryPhysicsEnv):
    cfg: Dex5P3HoldPhysicsEnvCfg


class Dex5V9P3DynamicHoldTeacherPhysicsEnv(Dex5P3EntryPhysicsEnv):
    cfg: Dex5V9P3DynamicHoldTeacherPhysicsEnvCfg


class Dex5V9P3StrictHoldPhysicsEnv(Dex5P3EntryPhysicsEnv):
    cfg: Dex5V9P3StrictHoldPhysicsEnvCfg

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        axis_error = self.init_pen_axis - self.pen_axis
        observations["policy"] = torch.cat(
            (observations["policy"], axis_error), dim=-1
        )
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        failed, _ = self._compute_dones()

        contacts = self.contact_force_norm > self.cfg.strict_min_contact_force_n
        triad = contacts[:, :3].all(dim=-1)
        ring_pinky_clear = ~contacts[:, 3:].any(dim=-1)
        axis_dot = torch.sum(self.pen_axis * self.init_pen_axis, dim=-1).clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot.abs())
        angular_speed = torch.linalg.norm(self.pen_angvel, dim=-1)
        position_drift = torch.linalg.norm(self.pen_pos - self.init_pen_pos, dim=-1)
        peak_force = self.contact_force_norm.max(dim=-1).values
        hand_rmse = torch.sqrt(
            torch.mean((self.hand_qpos - self.target_hand_qpos) ** 2, dim=-1)
        )

        axis_limit = torch.deg2rad(
            torch.full_like(axis_drift, self.cfg.strict_axis_drift_deg)
        )
        instantaneous_hold = (
            triad
            & ring_pinky_clear
            & (axis_drift <= axis_limit)
            & (angular_speed <= self.cfg.strict_max_angular_speed_rad_s)
            & (position_drift <= self.cfg.strict_max_position_drift_m)
            & (peak_force <= self.cfg.strict_max_contact_force_n)
            & (self.episode_max_penetration <= self.cfg.penetration_threshold)
            & ~failed
        )
        self.p3_hold_count = torch.where(
            instantaneous_hold,
            self.p3_hold_count + 1,
            torch.zeros_like(self.p3_hold_count),
        )
        success = self.p3_hold_count >= self.cfg.p3_hold_steps
        self.episode_dynamic_success |= success

        contact_ratio = contacts[:, :3].float().mean(dim=-1)
        index_contact = contacts[:, 1].float()
        penetration_excess = torch.relu(
            (self.episode_max_penetration - 0.00025) / 0.00025
        )
        force_excess = torch.relu((peak_force - 1.0) / 3.0)
        angle_score = torch.exp(-10.0 * axis_drift)
        speed_score = torch.exp(-0.18 * angular_speed)
        speed_violation = torch.relu(
            angular_speed / self.cfg.strict_max_angular_speed_rad_s - 1.0
        )
        position_score = torch.exp(-120.0 * position_drift)
        hold_progress = torch.clamp(
            self.p3_hold_count.float() / self.cfg.p3_hold_steps, 0.0, 1.0
        )

        terms = {
            "alive": torch.full_like(axis_drift, 0.05),
            "triad_contacts": 4.0 * contact_ratio + 4.0 * triad.float(),
            "index_contact": 4.0 * index_contact,
            "ring_pinky_clear": 0.5 * ring_pinky_clear.float(),
            "axis_hold": 40.0 * angle_score,
            "axis_violation": -40.0 * torch.relu(axis_drift / axis_limit - 1.0).square(),
            "angular_brake": 18.0 * speed_score,
            "angular_violation": -24.0 * speed_violation.square(),
            "position_hold": 2.0 * position_score,
            "hand_target": 1.2 * torch.exp(-7.0 * hand_rmse),
            "hold_progress": 10.0 * hold_progress,
            "success": 50.0 * success.float(),
            "penetration": -16.0 * penetration_excess,
            "force_excess": -2.0 * force_excess,
            "contact_loss": -5.0 * (1.0 - contact_ratio),
            "index_contact_loss": -6.0 * (1.0 - index_contact),
            "action_l2": -0.04 * torch.linalg.norm(self.actions, dim=-1),
            "action_smoothness": -0.06
            * torch.linalg.norm(self.actions - self.previous_actions, dim=-1),
            "failure": -8.0 * failed.float(),
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
        self.extras["log"] = {
            f"reward/{name}": value.mean() for name, value in terms.items()
        }
        self.extras["log"].update(
            {
                "metric/success_ratio": success.float().mean(),
                "metric/triad_contact_ratio": triad.float().mean(),
                "metric/axis_drift_deg": axis_drift.mean() * (180.0 / torch.pi),
                "metric/angular_speed_rad_s": angular_speed.mean(),
                "metric/position_drift_mm": position_drift.mean() * 1000.0,
                "metric/max_penetration_mm": self.episode_max_penetration.mean() * 1000.0,
                "metric/peak_force_n": peak_force.mean(),
                "metric/hold_ms": self.p3_hold_count.float().mean()
                * self.cfg.decimation
                * self.cfg.sim.dt
                * 1000.0,
            }
        )
        return reward
