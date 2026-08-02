"""Torch action projection used to mirror the Dex5 S0.1 hardware adapter.

This is an opt-in wrapper utility. It does not register a training task and
does not change any existing O0/O1 environment.
"""

from __future__ import annotations

import torch


COUPLED_PAIRS = ((2, 3), (6, 7), (10, 11), (14, 15))
JOINT_TO_ACTUATOR = (
    0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15
)


class Dex5IsaacHardwareCompatibilityWrapper:
    """Project batched Isaac targets onto the provisional 16-actuator manifold."""

    def __init__(
        self,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        *,
        max_policy_delta_rad: float = 0.025,
        max_velocity_rad_s: float = 0.5,
        policy_hz: float = 60.0,
    ) -> None:
        if joint_lower.shape[-1] != 20 or joint_upper.shape[-1] != 20:
            raise ValueError("Dex5 hardware wrapper requires 20D joint limits")
        if policy_hz <= 0.0 or max_policy_delta_rad < 0.0 or max_velocity_rad_s < 0.0:
            raise ValueError("wrapper rates and bounds must be non-negative, with positive policy_hz")
        self.joint_lower = joint_lower
        self.joint_upper = joint_upper
        self.policy_hz = float(policy_hz)
        self.max_policy_delta_rad = float(max_policy_delta_rad)
        self.max_velocity_rad_s = float(max_velocity_rad_s)

    def project_targets(
        self, targets: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if targets.shape[-1] != 20:
            raise ValueError(f"targets must end in 20 values, got {tuple(targets.shape)}")
        if not torch.isfinite(targets).all():
            raise ValueError("targets contain NaN or Inf")
        lower = self.joint_lower
        upper = self.joint_upper
        if env_ids is not None and lower.ndim > 1:
            ids = env_ids.to(device=lower.device, dtype=torch.long)
            lower = lower[ids]
            upper = upper[ids]
        if lower.ndim > 1 and lower.shape[0] != targets.shape[0]:
            raise ValueError(
                "batched joint limits require matching targets or explicit env_ids"
            )
        projected = torch.maximum(torch.minimum(targets, upper), lower).clone()
        for proximal, distal in COUPLED_PAIRS:
            pair_lower = torch.maximum(lower[..., proximal], lower[..., distal])
            pair_upper = torch.minimum(upper[..., proximal], upper[..., distal])
            value = torch.clamp(
                projected[..., distal], min=pair_lower, max=pair_upper
            )
            projected[..., proximal] = value
            projected[..., distal] = value
        return projected

    def integrate_normalized_action(
        self,
        current_targets: torch.Tensor,
        normalized_actions: torch.Tensor,
        action_scale_rad: float,
    ) -> torch.Tensor:
        if current_targets.shape != normalized_actions.shape or current_targets.shape[-1] != 20:
            raise ValueError("current_targets and normalized_actions must have matching (..., 20) shape")
        actions = torch.clamp(normalized_actions, -1.0, 1.0)
        allowed = min(
            float(action_scale_rad),
            self.max_policy_delta_rad,
            self.max_velocity_rad_s / self.policy_hz,
        )
        return self.project_targets(current_targets + allowed * actions)

    def virtual_to_logical_actuator(self, targets: torch.Tensor) -> torch.Tensor:
        """Compress for diagnostics; output is not a motor-shaft command."""

        projected = self.project_targets(targets)
        actuator = projected.new_empty(projected.shape[:-1] + (16,))
        for joint_index, actuator_index in enumerate(JOINT_TO_ACTUATOR):
            actuator[..., actuator_index] = projected[..., joint_index]
        return actuator

    def logical_actuator_to_virtual(self, actuator_targets: torch.Tensor) -> torch.Tensor:
        if actuator_targets.shape[-1] != 16:
            raise ValueError("actuator_targets must end in 16 values")
        indices = torch.as_tensor(
            JOINT_TO_ACTUATOR, dtype=torch.long, device=actuator_targets.device
        )
        return self.project_targets(actuator_targets.index_select(-1, indices))

    def apply_to_env(
        self,
        env: object,
        normalized_actions: torch.Tensor,
        action_scale_rad: float,
    ) -> torch.Tensor:
        """Apply projected targets through an existing Isaac articulation.

        The caller remains responsible for invoking this once per policy step.
        This opt-in method intentionally does not patch or register an existing
        task, so frozen O0/O1 behavior is unchanged.
        """

        for attribute in ("processed_targets", "hand", "dex5_joint_ids"):
            if not hasattr(env, attribute):
                raise TypeError(f"env is missing required attribute {attribute!r}")
        targets = self.integrate_normalized_action(
            env.processed_targets, normalized_actions, action_scale_rad
        )
        env.processed_targets = targets
        env.hand.set_joint_position_target(targets, joint_ids=env.dex5_joint_ids)
        return targets


class Dex5S0ActuatorModel:
    """Batched provisional 60 Hz policy to 100 Hz command-chain model.

    The model is deliberately kinematic at the command boundary. ``gain`` is
    a command-tracking fraction, not an identified motor PD gain, and delay is
    represented in 100 Hz command ticks. PhysX remains responsible for the
    articulation dynamics after the resulting absolute target is applied.
    """

    def __init__(
        self,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        *,
        num_envs: int,
        policy_hz: float = 60.0,
        command_hz: float = 100.0,
        max_policy_delta_rad: float = 0.025,
        max_delay_ticks: int = 2,
        joint_response_scale: torch.Tensor | None = None,
    ) -> None:
        if num_envs <= 0 or policy_hz <= 0.0 or command_hz <= 0.0:
            raise ValueError("environment count and clock rates must be positive")
        if max_delay_ticks < 0:
            raise ValueError("max_delay_ticks must be non-negative")
        self.projector = Dex5IsaacHardwareCompatibilityWrapper(
            joint_lower,
            joint_upper,
            max_policy_delta_rad=max_policy_delta_rad,
            max_velocity_rad_s=float("inf"),
            policy_hz=policy_hz,
        )
        self.num_envs = int(num_envs)
        self.policy_hz = float(policy_hz)
        self.command_hz = float(command_hz)
        self.max_policy_delta_rad = float(max_policy_delta_rad)
        self.max_delay_ticks = int(max_delay_ticks)
        template = torch.zeros(
            (self.num_envs, 20),
            dtype=joint_lower.dtype,
            device=joint_lower.device,
        )
        self.segment_start = template.clone()
        self.segment_target = template.clone()
        self.applied_target = template.clone()
        self.calibration_origin = template.clone()
        if joint_response_scale is None:
            self.joint_response_scale = template.new_ones((self.num_envs, 20))
        else:
            scale = joint_response_scale.to(device=template.device, dtype=template.dtype)
            if scale.shape == (20,):
                scale = scale.unsqueeze(0).expand(self.num_envs, -1)
            if scale.shape != (self.num_envs, 20) or torch.any(scale <= 0.0):
                raise ValueError("joint_response_scale must be positive with shape (20,) or (num_envs, 20)")
            self.joint_response_scale = scale.clone()
        self.delay_buffer = template.unsqueeze(1).repeat(
            1, self.max_delay_ticks + 1, 1
        )
        self.segment_start_time = template.new_zeros(self.num_envs)
        self.segment_end_time = template.new_zeros(self.num_envs)
        self.last_command_tick = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=joint_lower.device
        )
        self.delay_ticks = torch.zeros_like(self.last_command_tick)
        self.tracking_gain = template.new_ones(self.num_envs)
        self.velocity_limit_rad_s = template.new_full((self.num_envs,), 0.5)
        self.command_update_count = torch.zeros_like(self.last_command_tick)

    def _env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(
                self.num_envs, dtype=torch.long, device=self.segment_start.device
            )
        return env_ids.to(device=self.segment_start.device, dtype=torch.long)

    def reset(
        self,
        targets: torch.Tensor,
        *,
        time_s: float,
        env_ids: torch.Tensor | None = None,
        delay_ticks: torch.Tensor | None = None,
        tracking_gain: torch.Tensor | None = None,
        velocity_limit_rad_s: torch.Tensor | None = None,
    ) -> None:
        ids = self._env_ids(env_ids)
        if targets.shape != (ids.numel(), 20):
            raise ValueError("reset targets must have shape (len(env_ids), 20)")
        projected = self.projector.project_targets(targets, env_ids=ids)
        self.segment_start[ids] = projected
        self.segment_target[ids] = projected
        self.applied_target[ids] = projected
        self.calibration_origin[ids] = projected
        self.delay_buffer[ids] = projected.unsqueeze(1)
        self.segment_start_time[ids] = float(time_s)
        self.segment_end_time[ids] = float(time_s)
        tick = int(torch.floor(torch.tensor(float(time_s) * self.command_hz + 1.0e-7)).item())
        self.last_command_tick[ids] = tick
        self.command_update_count[ids] = 0
        if delay_ticks is not None:
            values = delay_ticks.to(device=ids.device, dtype=torch.long)
            if torch.any(values < 0) or torch.any(values > self.max_delay_ticks):
                raise ValueError("delay_ticks exceed configured delay buffer")
            self.delay_ticks[ids] = values
        if tracking_gain is not None:
            values = tracking_gain.to(device=projected.device, dtype=projected.dtype)
            if torch.any(values < 0.0) or torch.any(values > 1.0):
                raise ValueError("tracking_gain must be in [0, 1]")
            self.tracking_gain[ids] = values
        if velocity_limit_rad_s is not None:
            values = velocity_limit_rad_s.to(
                device=projected.device, dtype=projected.dtype
            )
            if torch.any(values <= 0.0):
                raise ValueError("velocity limits must be positive")
            self.velocity_limit_rad_s[ids] = values

    def _interpolated_target(self, time_s: float) -> torch.Tensor:
        duration = (self.segment_end_time - self.segment_start_time).clamp_min(1.0e-9)
        alpha = ((float(time_s) - self.segment_start_time) / duration).clamp(0.0, 1.0)
        return self.projector.project_targets(
            torch.lerp(self.segment_start, self.segment_target, alpha.unsqueeze(-1))
        )

    def submit_policy_action(
        self,
        normalized_actions: torch.Tensor,
        *,
        action_scale_rad: float,
        time_s: float,
    ) -> torch.Tensor:
        if normalized_actions.shape != (self.num_envs, 20):
            raise ValueError("policy actions must have shape (num_envs, 20)")
        current = self._interpolated_target(time_s)
        allowed = torch.minimum(
            self.velocity_limit_rad_s / self.policy_hz,
            current.new_full(
                (self.num_envs,),
                min(float(action_scale_rad), self.max_policy_delta_rad),
            ),
        )
        destination = self.projector.project_targets(
            current
            + allowed.unsqueeze(-1) * torch.clamp(normalized_actions, -1.0, 1.0)
        )
        self.segment_start.copy_(current)
        self.segment_target.copy_(destination)
        self.segment_start_time.fill_(float(time_s))
        self.segment_end_time.fill_(float(time_s) + 1.0 / self.policy_hz)
        return destination

    def physics_step(self, time_s: float) -> torch.Tensor:
        """Advance 100 Hz command events and return the held PhysX target."""

        command_tick = int(
            torch.floor(torch.tensor(float(time_s) * self.command_hz + 1.0e-7)).item()
        )
        first_tick = int(self.last_command_tick.min().item()) + 1
        if command_tick < first_tick:
            return self.applied_target
        rows = torch.arange(self.num_envs, device=self.applied_target.device)
        for tick in range(first_tick, command_tick + 1):
            active = self.last_command_tick < tick
            if not bool(active.any()):
                continue
            command = self._interpolated_target(tick / self.command_hz)
            self.delay_buffer[active, 1:] = self.delay_buffer[active, :-1].clone()
            self.delay_buffer[active, 0] = command[active]
            delayed = self.delay_buffer[rows, self.delay_ticks]
            calibrated = self.calibration_origin + self.joint_response_scale * (
                delayed - self.calibration_origin
            )
            lower = self.projector.joint_lower
            upper = self.projector.joint_upper
            calibrated = torch.maximum(torch.minimum(calibrated, upper), lower)
            gain = self.tracking_gain.unsqueeze(-1)
            # Do not re-project here: measured PIP/DIP feedback moves on a
            # roughly 0.29 response ratio even though the public command pair
            # is equal. The command manifold remains enforced before delay.
            updated = self.applied_target + gain * (calibrated - self.applied_target)
            updated = torch.maximum(torch.minimum(updated, upper), lower)
            self.applied_target[active] = updated[active]
            self.last_command_tick[active] = tick
            self.command_update_count[active] += 1
        return self.applied_target
