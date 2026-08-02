"""Hardware-compatible P3 to 45-degree short-rotation oracle task."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul

from .s0_hardware_compatible_env import (
    Dex5S0P3HardwareCompatibleAuditEnv,
    Dex5S0P3HardwareCompatibleAuditEnvCfg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
S0_P3_BANK = PROJECT_ROOT / (
    "Dex5_Sim2Real_S0/hardware_compatible_p3_reset_bank/"
    "Dex5_S0_HW_P3_Reset_Bank.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


S0_P3_BANK_SHA256 = _sha256(S0_P3_BANK)


@configclass
class Dex5S0ShortRotationEnvCfg(Dex5S0P3HardwareCompatibleAuditEnvCfg):
    """Nominal S0 short-rotation task using only the validated S0 P3 bank."""

    bundle_path = str(S0_P3_BANK)
    expected_bundle_sha256 = S0_P3_BANK_SHA256
    reset_mixture = (0.0, 0.0, 1.0, 0.0, 0.0)
    reset_split = "train"
    episode_length_s = 0.80
    termination_mode = "early"
    action_scale = 0.025
    observation_space = 152
    include_gait_history_observation = False
    include_temporal_support_observation = False

    target_rotation_rad = 0.7853981633974483
    target_tolerance_rad = 0.17453292519943295
    hold_steps = 9
    hold_max_target_speed_rad_s = 2.5
    hold_max_wrong_axis_speed_rad_s = 2.0
    hold_max_height_loss_m = 0.005
    hold_min_quaternion_axis_alignment = 0.70
    forbidden_contact_debounce_steps = 3

    s0_policy_hz = 60.0
    s0_command_hz = 100.0
    s0_max_policy_delta_rad = 0.025
    s0_delay_ticks_range = (0, 2)
    s0_tracking_gain_range = (0.85, 1.00)
    s0_velocity_limit_rad_s_range = (0.35, 0.65)


@configclass
class Dex5S0ShortRotationMediumDREnvCfg(Dex5S0ShortRotationEnvCfg):
    """Provisional medium DR; hardware identification is still pending."""

    reset_joint_pos_noise = 0.0015
    reset_joint_vel_noise = 0.030
    reset_pen_pos_noise = 0.0004
    reset_pen_vel_noise = 0.015
    canonical_reset_pen_rot_noise_rad = 0.008726646259971648
    canonical_mass_scale_range = (0.90, 1.10)
    canonical_static_friction_scale_range = (0.85, 1.15)
    canonical_dynamic_friction_scale_range = (0.85, 1.15)
    s0_delay_ticks_range = (0, 3)
    s0_tracking_gain_range = (0.75, 1.00)
    s0_velocity_limit_rad_s_range = (0.30, 0.70)


@configclass
class Dex5S0ShortRotation15DegEnvCfg(Dex5S0ShortRotationEnvCfg):
    """First curriculum stage: learn coupled rotation, braking, and hold."""

    target_rotation_rad = 0.2617993877991494
    target_tolerance_rad = 0.08726646259971647


@configclass
class Dex5S0ShortRotation30DegEnvCfg(Dex5S0ShortRotationEnvCfg):
    """Second curriculum stage before the unchanged 45-degree task."""

    target_rotation_rad = 0.5235987755982988
    target_tolerance_rad = 0.1308996938995747


@configclass
class Dex5S0ShortRotation37DegEnvCfg(Dex5S0ShortRotationEnvCfg):
    """Bridge the observed 30-to-45-degree curriculum boundary."""

    target_rotation_rad = 0.6544984694978736
    target_tolerance_rad = 0.15271630954950383


@configclass
class Dex5S0ShortRotation35DegEnvCfg(Dex5S0ShortRotationEnvCfg):
    """Require a 30-degree minimum while preserving the final hold gate."""

    target_rotation_rad = 0.6108652381980153
    target_tolerance_rad = 0.08726646259971647


@configclass
class Dex5S0ShortRotation40DegEnvCfg(Dex5S0ShortRotationEnvCfg):
    """Require a 32.5-degree minimum before the final 35-degree gate."""

    target_rotation_rad = 0.6981317007977318
    target_tolerance_rad = 0.1308996938995747


@configclass
class Dex5S0ShortRotation35DegLongEnvCfg(Dex5S0ShortRotation35DegEnvCfg):
    """Long-horizon reachability audit under unchanged physical gates."""

    episode_length_s = 1.20


@configclass
class Dex5S0ShortRotationLongEnvCfg(Dex5S0ShortRotationEnvCfg):
    """Final 45-degree task with additional time for limited actuators."""

    episode_length_s = 1.20


@configclass
class Dex5S0ShortRotationLongNominalEnvCfg(Dex5S0ShortRotationLongEnvCfg):
    """Fixed provisional actuator midpoint for the nominal policy gate."""

    s0_delay_ticks_range = (1, 1)
    s0_tracking_gain_range = (0.925, 0.925)
    s0_velocity_limit_rad_s_range = (0.50, 0.50)


@configclass
class Dex5S0ShortRotation16NominalEnvCfg(Dex5S0ShortRotationLongNominalEnvCfg):
    """Audit whether strict success is limited only by actuator time."""

    episode_length_s = 1.60


@configclass
class Dex5S0ShortRotation16MediumDREnvCfg(Dex5S0ShortRotationMediumDREnvCfg):
    """Final 1.6-second task under the provisional medium DR envelope."""

    episode_length_s = 1.60


class Dex5S0ShortRotationEnv(Dex5S0P3HardwareCompatibleAuditEnv):
    """Short rotation with no handoff and no contact-pattern reward."""

    cfg: Dex5S0ShortRotationEnvCfg

    def __init__(self, cfg: Dex5S0ShortRotationEnvCfg, **kwargs) -> None:
        self.s0_hold_counter = None
        self.s0_success = None
        self.s0_forbidden_counter = None
        self.s0_forbidden_transition = None
        self.student_frame_history = None
        self.completed_s0_episodes = None
        self.completed_s0_successes = None
        self.completed_s0_forbidden = None
        self.completed_s0_timeouts = None
        self.completed_s0_failures = None
        self.completed_s0_peak_angle_sum = None
        self.completed_s0_peak_angle_max = None
        self.episode_s0_max_hold = None
        self.completed_s0_max_hold = None
        self.s0_gate_seen = None
        self.completed_s0_gate_counts = None
        self.completed_s0_bank_episodes = None
        self.completed_s0_bank_successes = None
        self.last_completed_s0_success = None
        self.last_completed_s0_bank_id = None
        self.last_completed_s0_pen_pose = None
        self.last_completed_s0_pen_linvel = None
        self.last_completed_s0_pen_angvel = None
        self.last_completed_s0_contact_state = None
        self.last_completed_s0_contact_force_w = None
        self.last_completed_s0_contact_force_norm = None
        self.last_completed_s0_physics_quat_samples = None
        self.last_completed_s0_physics_angvel_samples = None
        self.last_completed_s0_student_observation = None
        self.last_completed_s0_peak_angle = None
        self.last_completed_s0_actual_angle = None
        self.last_completed_s0_axis_alignment = None
        self.last_completed_s0_max_hold = None
        self.last_completed_s0_max_penetration = None
        self.last_completed_s0_gate_seen = None
        self.last_completed_s0_failure_seen = None
        self.last_completed_s0_timeout = None
        self.last_completed_s0_forbidden = None
        self.completed_s0_generation = None
        self.s0_initial_pen_quat = None
        self.s0_actual_target_rotation = None
        self.s0_actual_geodesic_rotation = None
        self.s0_actual_axis_alignment = None
        self.s0_integrated_target_rotation = None
        self.s0_previous_pen_quat = None
        self.s0_pose_delta_angvel = None
        self.s0_physics_quat_before = None
        self.s0_physics_angvel_before = None
        self.s0_physics_trace_index = 0
        super().__init__(cfg, **kwargs)
        self.s0_hold_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.s0_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.s0_forbidden_counter = torch.zeros_like(self.s0_hold_counter)
        self.s0_forbidden_transition = torch.zeros_like(self.s0_success)
        self.student_frame_history = torch.zeros(
            (self.num_envs, 4, 85), device=self.device
        )
        self.completed_s0_episodes = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.completed_s0_successes = torch.zeros_like(self.completed_s0_episodes)
        self.completed_s0_forbidden = torch.zeros_like(self.completed_s0_episodes)
        self.completed_s0_timeouts = torch.zeros_like(self.completed_s0_episodes)
        self.completed_s0_failures = torch.zeros_like(self.completed_s0_episodes)
        self.completed_s0_peak_angle_sum = torch.zeros((), device=self.device)
        self.completed_s0_peak_angle_max = torch.zeros((), device=self.device)
        self.episode_s0_max_hold = torch.zeros_like(self.s0_hold_counter)
        self.completed_s0_max_hold = torch.zeros_like(self.completed_s0_episodes)
        self.s0_gate_seen = torch.zeros(
            (self.num_envs, 8), dtype=torch.bool, device=self.device
        )
        self.completed_s0_gate_counts = torch.zeros(
            8, dtype=torch.long, device=self.device
        )
        bank_count = int(self.bank_size)
        self.completed_s0_bank_episodes = torch.zeros(
            bank_count, dtype=torch.long, device=self.device
        )
        self.completed_s0_bank_successes = torch.zeros_like(
            self.completed_s0_bank_episodes
        )
        self.last_completed_s0_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_completed_s0_bank_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.last_completed_s0_pen_pose = torch.zeros(
            (self.num_envs, 7), device=self.device
        )
        self.last_completed_s0_pen_linvel = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.last_completed_s0_pen_angvel = torch.zeros_like(
            self.last_completed_s0_pen_linvel
        )
        self.last_completed_s0_contact_state = torch.zeros(
            (self.num_envs, 5), dtype=torch.bool, device=self.device
        )
        self.last_completed_s0_contact_force_w = torch.zeros(
            (self.num_envs, 5, 3), device=self.device
        )
        self.last_completed_s0_contact_force_norm = torch.zeros(
            (self.num_envs, 5), device=self.device
        )
        physics_sample_count = int(self.cfg.decimation) + 1
        self.last_completed_s0_physics_quat_samples = torch.zeros(
            (self.num_envs, physics_sample_count, 4), device=self.device
        )
        self.last_completed_s0_physics_angvel_samples = torch.zeros(
            (self.num_envs, physics_sample_count, 3), device=self.device
        )
        self.last_completed_s0_student_observation = torch.zeros(
            (self.num_envs, 340), device=self.device
        )
        self.last_completed_s0_peak_angle = torch.zeros(
            self.num_envs, device=self.device
        )
        self.last_completed_s0_actual_angle = torch.zeros_like(
            self.last_completed_s0_peak_angle
        )
        self.last_completed_s0_axis_alignment = torch.zeros_like(
            self.last_completed_s0_peak_angle
        )
        self.last_completed_s0_max_hold = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.last_completed_s0_max_penetration = torch.zeros_like(
            self.last_completed_s0_peak_angle
        )
        self.last_completed_s0_gate_seen = torch.zeros(
            (self.num_envs, 8), dtype=torch.bool, device=self.device
        )
        self.last_completed_s0_failure_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_completed_s0_timeout = torch.zeros_like(
            self.last_completed_s0_failure_seen
        )
        self.last_completed_s0_forbidden = torch.zeros_like(
            self.last_completed_s0_failure_seen
        )
        self.completed_s0_generation = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.s0_initial_pen_quat = self.pen_quat.clone()
        self.s0_actual_target_rotation = torch.zeros(
            self.num_envs, device=self.device
        )
        self.s0_actual_geodesic_rotation = torch.zeros_like(
            self.s0_actual_target_rotation
        )
        self.s0_actual_axis_alignment = torch.zeros_like(
            self.s0_actual_target_rotation
        )
        self.s0_integrated_target_rotation = torch.zeros_like(
            self.s0_actual_target_rotation
        )
        self.s0_previous_pen_quat = self.pen_quat.clone()
        self.s0_pose_delta_angvel = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.s0_physics_quat_before = torch.zeros(
            (self.num_envs, int(self.cfg.decimation), 4), device=self.device
        )
        self.s0_physics_angvel_before = torch.zeros(
            (self.num_envs, int(self.cfg.decimation), 3), device=self.device
        )
        self._fill_student_history(self.hand._ALL_INDICES)

    def _actual_quaternion_rotation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return signed world-z rotvec, geodesic angle, and axis alignment."""
        relative = quat_mul(self.pen_quat, quat_conjugate(self.s0_initial_pen_quat))
        relative = torch.where(relative[:, :1] < 0.0, -relative, relative)
        vector_norm = torch.linalg.norm(relative[:, 1:], dim=-1)
        angle = 2.0 * torch.atan2(vector_norm, relative[:, 0].clamp_min(1.0e-9))
        axis_z = torch.where(
            vector_norm > 1.0e-9,
            relative[:, 3] / vector_norm.clamp_min(1.0e-9),
            torch.zeros_like(vector_norm),
        )
        target_component = self.cfg.target_rotation_direction * angle * axis_z
        alignment = target_component / angle.clamp_min(1.0e-9)
        return target_component, angle, alignment

    def _pose_delta_angular_velocity(self, dt: float) -> torch.Tensor:
        relative = quat_mul(self.pen_quat, quat_conjugate(self.s0_previous_pen_quat))
        relative = torch.where(relative[:, :1] < 0.0, -relative, relative)
        vector_norm = torch.linalg.norm(relative[:, 1:], dim=-1)
        angle = 2.0 * torch.atan2(vector_norm, relative[:, 0].clamp_min(1.0e-9))
        axis = torch.where(
            (vector_norm > 1.0e-9).unsqueeze(-1),
            relative[:, 1:] / vector_norm.clamp_min(1.0e-9).unsqueeze(-1),
            torch.zeros_like(relative[:, 1:]),
        )
        return axis * (angle / dt).unsqueeze(-1)

    def _student_frame(self) -> torch.Tensor:
        contacts = self.contact_force_norm.clamp(0.0, 20.0)
        return torch.cat(
            (
                self.hand_qpos,
                self.hand_qvel,
                self.actions,
                self.processed_targets,
                contacts,
            ),
            dim=-1,
        )

    def _fill_student_history(self, env_ids: torch.Tensor) -> None:
        if self.student_frame_history is None:
            return
        self._compute_intermediate_values()
        frame = self._student_frame()[env_ids]
        self.student_frame_history[env_ids] = frame.unsqueeze(1)

    def get_student_observation(self) -> torch.Tensor:
        """Return the deployable 4-frame, 340D observation contract."""

        assert self.student_frame_history is not None
        return self.student_frame_history.reshape(self.num_envs, -1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        if self.student_frame_history is not None:
            frame = self._student_frame()
            self.student_frame_history[:, 1:] = self.student_frame_history[:, :-1].clone()
            self.student_frame_history[:, 0] = frame
        return observations

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.s0_physics_trace_index = 0
        super()._pre_physics_step(actions)

    def _apply_action(self) -> None:
        if (
            self.s0_physics_quat_before is not None
            and self.s0_physics_trace_index < int(self.cfg.decimation)
        ):
            self._compute_intermediate_values()
            self.s0_physics_quat_before[:, self.s0_physics_trace_index] = self.pen_quat
            self.s0_physics_angvel_before[:, self.s0_physics_trace_index] = self.pen_angvel
            self.s0_physics_trace_index += 1
        super()._apply_action()

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            ids = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.completed_s0_episodes is not None:
            completed = self.episode_length_buf[ids] > 0
            completed_ids = ids[completed]
            completed_bank_ids = self.current_bank_ids[completed_ids]
            if completed_ids.numel() > 0:
                self.last_completed_s0_pen_pose[completed_ids] = torch.cat(
                    (self.pen_pos[completed_ids], self.pen_quat[completed_ids]), dim=-1
                )
                self.last_completed_s0_pen_linvel[completed_ids] = self.pen_linvel[
                    completed_ids
                ]
                self.last_completed_s0_pen_angvel[completed_ids] = self.pen_angvel[
                    completed_ids
                ]
                self.last_completed_s0_contact_state[completed_ids] = (
                    self.contact_force_norm[completed_ids]
                    > self.cfg.contact_force_threshold_n
                )
                self.last_completed_s0_contact_force_w[completed_ids] = (
                    self.contact_force_w[completed_ids]
                )
                self.last_completed_s0_contact_force_norm[completed_ids] = (
                    self.contact_force_norm[completed_ids]
                )
                self.last_completed_s0_physics_quat_samples[completed_ids] = torch.cat(
                    (
                        self.s0_physics_quat_before[completed_ids],
                        self.pen_quat[completed_ids].unsqueeze(1),
                    ),
                    dim=1,
                )
                self.last_completed_s0_physics_angvel_samples[completed_ids] = torch.cat(
                    (
                        self.s0_physics_angvel_before[completed_ids],
                        self.pen_angvel[completed_ids].unsqueeze(1),
                    ),
                    dim=1,
                )
                terminal_history = self.student_frame_history[completed_ids].clone()
                terminal_history[:, 1:] = terminal_history[:, :-1].clone()
                terminal_history[:, 0] = self._student_frame()[completed_ids]
                self.last_completed_s0_student_observation[completed_ids] = (
                    terminal_history.reshape(completed_ids.numel(), -1)
                )
                self.last_completed_s0_peak_angle[completed_ids] = (
                    self.episode_peak_target_rotation[completed_ids]
                )
                self.last_completed_s0_actual_angle[completed_ids] = (
                    self.s0_actual_target_rotation[completed_ids]
                )
                self.last_completed_s0_axis_alignment[completed_ids] = (
                    self.s0_actual_axis_alignment[completed_ids]
                )
                self.last_completed_s0_max_hold[completed_ids] = (
                    self.episode_s0_max_hold[completed_ids]
                )
                self.last_completed_s0_max_penetration[completed_ids] = (
                    self.episode_max_penetration[completed_ids]
                )
                self.last_completed_s0_gate_seen[completed_ids] = (
                    self.s0_gate_seen[completed_ids]
                )
                self.last_completed_s0_failure_seen[completed_ids] = (
                    self.episode_failure_seen[completed_ids]
                )
                self.last_completed_s0_timeout[completed_ids] = (
                    self.current_timeout[completed_ids]
                )
                self.last_completed_s0_forbidden[completed_ids] = (
                    self.s0_forbidden_transition[completed_ids]
                )
            self.last_completed_s0_success[ids] = completed & self.s0_success[ids]
            self.last_completed_s0_bank_id[ids] = torch.where(
                completed,
                self.current_bank_ids[ids],
                -1,
            )
            self.completed_s0_generation[completed_ids] += 1
            self.completed_s0_episodes += completed.sum()
            self.completed_s0_successes += (
                completed & self.s0_success[ids]
            ).sum()
            self.completed_s0_forbidden += (
                completed & self.s0_forbidden_transition[ids]
            ).sum()
            self.completed_s0_timeouts += (
                completed & self.current_timeout[ids]
            ).sum()
            self.completed_s0_failures += (
                completed & self.episode_failure_seen[ids]
            ).sum()
            self.completed_s0_peak_angle_sum += torch.where(
                completed, self.episode_peak_target_rotation[ids], 0.0
            ).sum()
            completed_peak = torch.where(
                completed, self.episode_peak_target_rotation[ids], 0.0
            ).max()
            self.completed_s0_peak_angle_max = torch.maximum(
                self.completed_s0_peak_angle_max, completed_peak
            )
            self.completed_s0_max_hold = torch.maximum(
                self.completed_s0_max_hold,
                torch.where(completed, self.episode_s0_max_hold[ids], 0).max(),
            )
            self.completed_s0_gate_counts += (
                self.s0_gate_seen[ids] & completed.unsqueeze(-1)
            ).sum(dim=0)
            self.completed_s0_bank_episodes.index_add_(
                0,
                completed_bank_ids,
                torch.ones_like(completed_bank_ids),
            )
            self.completed_s0_bank_successes.index_add_(
                0,
                completed_bank_ids,
                self.s0_success[completed_ids].long(),
            )
        super()._reset_idx(ids)
        for buffer, value in (
            (self.s0_hold_counter, 0),
            (self.s0_success, False),
            (self.s0_forbidden_counter, 0),
            (self.s0_forbidden_transition, False),
            (self.episode_s0_max_hold, 0),
        ):
            if buffer is not None:
                buffer[ids] = value
        self.s0_gate_seen[ids] = False
        self._compute_intermediate_values()
        if self.s0_initial_pen_quat is not None:
            self.s0_initial_pen_quat[ids] = self.pen_quat[ids]
            self.s0_actual_target_rotation[ids] = 0.0
            self.s0_actual_geodesic_rotation[ids] = 0.0
            self.s0_actual_axis_alignment[ids] = 0.0
            self.s0_integrated_target_rotation[ids] = 0.0
            self.s0_previous_pen_quat[ids] = self.pen_quat[ids]
            self.s0_pose_delta_angvel[ids] = 0.0
        self._fill_student_history(ids)

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        instantaneous_target_velocity = (
            self.cfg.target_rotation_direction * self.pen_angvel[:, 2]
        )
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.s0_integrated_target_rotation += instantaneous_target_velocity * dt
        prior_angle = self.s0_actual_target_rotation.clone()
        self.s0_pose_delta_angvel = self._pose_delta_angular_velocity(dt)
        (
            self.s0_actual_target_rotation,
            self.s0_actual_geodesic_rotation,
            self.s0_actual_axis_alignment,
        ) = self._actual_quaternion_rotation()
        self.s0_previous_pen_quat.copy_(self.pen_quat)
        self.episode_target_rotation.copy_(self.s0_actual_target_rotation)
        self.episode_peak_target_rotation = torch.maximum(
            self.episode_peak_target_rotation, self.episode_target_rotation
        )
        progress = self.s0_actual_target_rotation - prior_angle
        target_velocity = (
            self.cfg.target_rotation_direction * self.s0_pose_delta_angvel[:, 2]
        )
        angular_speed = torch.linalg.norm(self.s0_pose_delta_angvel, dim=-1)
        wrong_axis_speed = torch.linalg.norm(
            self.s0_pose_delta_angvel[:, :2], dim=-1
        )
        alignment = target_velocity / angular_speed.clamp_min(1.0e-6)
        linear_speed = torch.linalg.norm(self.pen_linvel, dim=-1)
        height_loss = self.init_pen_pos[:, 2] - self.pen_pos[:, 2]
        angle_error = torch.abs(
            self.episode_target_rotation - self.cfg.target_rotation_rad
        )
        in_target = angle_error <= self.cfg.target_tolerance_rad
        actual_in_target = in_target
        actual_axis_aligned = (
            self.s0_actual_axis_alignment
            >= self.cfg.hold_min_quaternion_axis_alignment
        )
        contacts = self.contact_force_norm > self.cfg.contact_force_threshold_n
        tim_all = contacts[:, :3].all(dim=-1)
        forbidden_now = contacts[:, 3:].any(dim=-1) | ~tim_all
        assert self.s0_forbidden_counter is not None
        assert self.s0_forbidden_transition is not None
        self.s0_forbidden_counter = torch.where(
            forbidden_now,
            self.s0_forbidden_counter + 1,
            torch.zeros_like(self.s0_forbidden_counter),
        )
        self.s0_forbidden_transition |= (
            self.s0_forbidden_counter >= self.cfg.forbidden_contact_debounce_steps
        )
        hold_condition = (
            actual_in_target
            & actual_axis_aligned
            & (torch.abs(target_velocity) <= self.cfg.hold_max_target_speed_rad_s)
            & (wrong_axis_speed <= self.cfg.hold_max_wrong_axis_speed_rad_s)
            & (height_loss <= self.cfg.hold_max_height_loss_m)
            & tim_all
            & ~contacts[:, 3:].any(dim=-1)
            & ~self.current_failed
        )
        gate_conditions = torch.stack(
            (
                actual_in_target
                & actual_axis_aligned,
                actual_in_target
                & actual_axis_aligned
                & (torch.abs(target_velocity) <= self.cfg.hold_max_target_speed_rad_s),
                actual_in_target
                & actual_axis_aligned
                & (torch.abs(target_velocity) <= self.cfg.hold_max_target_speed_rad_s)
                & (wrong_axis_speed <= self.cfg.hold_max_wrong_axis_speed_rad_s),
                actual_in_target
                & actual_axis_aligned
                & (torch.abs(target_velocity) <= self.cfg.hold_max_target_speed_rad_s)
                & (wrong_axis_speed <= self.cfg.hold_max_wrong_axis_speed_rad_s)
                & (height_loss <= self.cfg.hold_max_height_loss_m),
                actual_in_target
                & actual_axis_aligned
                & (torch.abs(target_velocity) <= self.cfg.hold_max_target_speed_rad_s)
                & (wrong_axis_speed <= self.cfg.hold_max_wrong_axis_speed_rad_s)
                & (height_loss <= self.cfg.hold_max_height_loss_m)
                & tim_all,
                actual_in_target
                & actual_axis_aligned
                & (torch.abs(target_velocity) <= self.cfg.hold_max_target_speed_rad_s)
                & (wrong_axis_speed <= self.cfg.hold_max_wrong_axis_speed_rad_s)
                & (height_loss <= self.cfg.hold_max_height_loss_m)
                & tim_all
                & ~contacts[:, 3:].any(dim=-1),
                actual_in_target
                & actual_axis_aligned
                & (torch.abs(target_velocity) <= self.cfg.hold_max_target_speed_rad_s)
                & (wrong_axis_speed <= self.cfg.hold_max_wrong_axis_speed_rad_s)
                & (height_loss <= self.cfg.hold_max_height_loss_m)
                & tim_all
                & ~contacts[:, 3:].any(dim=-1)
                & ~self.current_failed,
                hold_condition,
            ),
            dim=-1,
        )
        self.s0_gate_seen |= gate_conditions
        assert self.s0_hold_counter is not None
        assert self.s0_success is not None
        self.s0_hold_counter = torch.where(
            hold_condition,
            self.s0_hold_counter + 1,
            torch.zeros_like(self.s0_hold_counter),
        )
        self.episode_s0_max_hold = torch.maximum(
            self.episode_s0_max_hold, self.s0_hold_counter
        )
        newly_successful = self.s0_hold_counter >= self.cfg.hold_steps
        self.s0_success |= newly_successful

        before_target = self.episode_target_rotation < (
            self.cfg.target_rotation_rad - self.cfg.target_tolerance_rad
        )
        overshoot = torch.relu(
            self.episode_target_rotation
            - (self.cfg.target_rotation_rad + self.cfg.target_tolerance_rad)
        )
        penetration_excess = torch.relu(self.episode_max_penetration - 0.0005) / 0.0005
        terms = {
            "rotation_progress": 12.0 * torch.where(
                before_target, torch.clamp(progress, -0.2, 0.2), 0.0
            ),
            "axis_alignment": 0.20 * torch.relu(alignment).clamp_max(1.0),
            "target_error": -0.35 * angle_error,
            "braking": -0.12 * torch.where(
                in_target, torch.abs(target_velocity), 0.0
            ),
            "overshoot": -3.0 * overshoot,
            "linear_velocity": -0.50 * linear_speed,
            "wrong_axis": -0.12 * wrong_axis_speed,
            "joint_motion": -0.01 * torch.mean(self.hand_qvel**2, dim=-1),
            "action": -0.01 * torch.mean(self.actions**2, dim=-1),
            "action_smoothness": -0.02
            * torch.mean((self.actions - self.previous_actions) ** 2, dim=-1),
            "drop": -8.0 * self.current_drop.float(),
            "penetration": -2.0 * penetration_excess.clamp_max(3.0),
            "strict_hold_progress": 0.75
            * (
                self.s0_hold_counter.float()
                / float(self.cfg.hold_steps)
            ).clamp_max(1.0),
            "terminal_hold": 20.0 * newly_successful.float(),
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
        self.extras["log"] = {
            **{f"reward/{key}": value.mean() for key, value in terms.items()},
            "metric/s0_success_ratio": self.s0_success.float().mean(),
            "metric/s0_target_angle_deg": torch.rad2deg(
                self.episode_target_rotation
            ).mean(),
            "metric/s0_actual_target_angle_deg": torch.rad2deg(
                self.s0_actual_target_rotation
            ).mean(),
            "metric/s0_actual_geodesic_angle_deg": torch.rad2deg(
                self.s0_actual_geodesic_rotation
            ).mean(),
            "metric/s0_actual_axis_alignment": self.s0_actual_axis_alignment.mean(),
            "metric/s0_integrated_omega_target_angle_deg": torch.rad2deg(
                self.s0_integrated_target_rotation
            ).mean(),
            "metric/s0_pose_delta_target_speed_rad_s": target_velocity.mean(),
            "metric/s0_instantaneous_target_speed_rad_s": instantaneous_target_velocity.mean(),
            "metric/s0_hold_steps": self.s0_hold_counter.float().mean(),
            "metric/s0_forbidden_transition_ratio": self.s0_forbidden_transition.float().mean(),
            "metric/s0_axis_alignment": alignment.mean(),
        }
        return reward

    def _compute_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._compute_dones()
        if self.s0_success is not None:
            terminated |= self.s0_success
        return terminated, time_out
