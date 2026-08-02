"""S0.2 audit-only Dex5 environment with a hardware-compatible command chain."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import torch

from isaaclab.utils import configclass

from .hardware_compatible_action import Dex5S0ActuatorModel
from .canonical_reset_physics_env import (
    Dex5CanonicalResetPhysicsEnv,
    Dex5CanonicalResetPhysicsEnvCfg,
)
from .privileged_oracle_physics_env import (
    Dex5PrivilegedOracleP3FixedHorizonCfg,
    Dex5PrivilegedOraclePhysicsEnv,
)


@configclass
class Dex5S0P3HardwareCompatibleAuditEnvCfg(
    Dex5PrivilegedOracleP3FixedHorizonCfg
):
    """P3-only fixed-horizon audit config; no trainer is registered."""

    s0_policy_hz = 60.0
    s0_command_hz = 100.0
    s0_max_policy_delta_rad = 0.025
    s0_delay_ticks_range = (0, 2)
    s0_tracking_gain_range = (0.85, 1.00)
    s0_velocity_limit_rad_s_range = (0.35, 0.65)
    s0_hardware_config_path: str = ""


class Dex5S0P3HardwareCompatibleAuditEnv(Dex5PrivilegedOraclePhysicsEnv):
    """Apply P3 audit actions through coupling and provisional actuator gaps."""

    cfg: Dex5S0P3HardwareCompatibleAuditEnvCfg

    def __init__(self, cfg: Dex5S0P3HardwareCompatibleAuditEnvCfg, **kwargs) -> None:
        self.s0_actuator_model: Dex5S0ActuatorModel | None = None
        super().__init__(cfg, **kwargs)
        measured_scale = None
        if self.cfg.s0_hardware_config_path:
            config = json.loads(
                Path(self.cfg.s0_hardware_config_path).read_text(encoding="utf-8")
            )
            measured_scale = torch.as_tensor(
                config["joint_response_scale_nominal"],
                device=self.device,
                dtype=self.joint_lower.dtype,
            )
        self.s0_actuator_model = Dex5S0ActuatorModel(
            self.joint_lower,
            self.joint_upper,
            num_envs=self.num_envs,
            policy_hz=self.cfg.s0_policy_hz,
            command_hz=self.cfg.s0_command_hz,
            max_policy_delta_rad=self.cfg.s0_max_policy_delta_rad,
            max_delay_ticks=int(self.cfg.s0_delay_ticks_range[1]),
            joint_response_scale=measured_scale,
        )
        self.s0_initial_projection_delta_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._reset_s0_model(self.hand._ALL_INDICES)

    def _simulation_time_s(self) -> float:
        return float(self._sim_step_counter) * float(self.physics_dt)

    def _reset_s0_model(self, env_ids: torch.Tensor) -> None:
        if self.s0_actuator_model is None:
            return
        count = env_ids.numel()
        delay_low, delay_high = self.cfg.s0_delay_ticks_range
        gain_low, gain_high = self.cfg.s0_tracking_gain_range
        velocity_low, velocity_high = self.cfg.s0_velocity_limit_rad_s_range
        delay = torch.randint(
            int(delay_low),
            int(delay_high) + 1,
            (count,),
            device=self.device,
        )
        gain = gain_low + (gain_high - gain_low) * torch.rand(
            count, device=self.device
        )
        velocity = velocity_low + (velocity_high - velocity_low) * torch.rand(
            count, device=self.device
        )
        original_targets = self.processed_targets[env_ids].clone()
        targets = self.s0_actuator_model.projector.project_targets(
            original_targets, env_ids=env_ids
        )
        self.s0_initial_projection_delta_rad[env_ids] = torch.max(
            torch.abs(targets - original_targets), dim=-1
        ).values
        self.processed_targets[env_ids] = targets
        self.s0_actuator_model.reset(
            targets,
            time_s=self._simulation_time_s(),
            env_ids=env_ids,
            delay_ticks=delay,
            tracking_gain=gain,
            velocity_limit_rad_s=velocity,
        )
        self.hand.set_joint_position_target(
            targets, joint_ids=self.dex5_joint_ids, env_ids=env_ids
        )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids_tensor = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        super()._reset_idx(env_ids_tensor)
        self._reset_s0_model(env_ids_tensor)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # The inherited method only records/clamps the action. Integration is
        # intentionally performed exactly once per 60 Hz policy step here.
        super()._pre_physics_step(actions)
        assert self.s0_actuator_model is not None
        self.s0_actuator_model.submit_policy_action(
            self.actions,
            action_scale_rad=self.cfg.action_scale,
            time_s=self._simulation_time_s(),
        )

    def _apply_action(self) -> None:
        # DirectRLEnv calls this on each 120 Hz physics substep. The actuator
        # model advances only when a 100 Hz command boundary is crossed.
        assert self.s0_actuator_model is not None
        targets = self.s0_actuator_model.physics_step(self._simulation_time_s())
        self.processed_targets.copy_(targets)
        self.hand.set_joint_position_target(
            self.processed_targets, joint_ids=self.dex5_joint_ids
        )


@configclass
class Dex5S0P3MeasuredHardwareCompatibleAuditEnvCfg(
    Dex5S0P3HardwareCompatibleAuditEnvCfg
):
    """S0.8 opt-in local actuator model; no trainer is registered."""

    s0_hardware_config_path = str(
        Path(__file__).resolve().parents[4]
        / "Dex5_Sim2Real_S0/hardware_calibration_s0_8/hardware_simulation_config.json"
    )
    s0_delay_ticks_range = (1, 4)
    s0_tracking_gain_range = (0.80, 1.00)
    s0_velocity_limit_rad_s_range = (0.04, 0.08)


class Dex5S0P3MeasuredHardwareCompatibleAuditEnv(
    Dex5S0P3HardwareCompatibleAuditEnv
):
    """P3 audit environment using the measured S0.8 local actuator model."""

    cfg: Dex5S0P3MeasuredHardwareCompatibleAuditEnvCfg


@configclass
class Dex5S0HardwareCompatibleResetEnvCfg(Dex5CanonicalResetPhysicsEnvCfg):
    """Bundle-configurable reset/settle task with enforced target coupling."""


class Dex5S0HardwareCompatibleResetEnv(Dex5CanonicalResetPhysicsEnv):
    """Restore candidate states while keeping every target on the S0 manifold."""

    cfg: Dex5S0HardwareCompatibleResetEnvCfg

    def __init__(self, cfg: Dex5S0HardwareCompatibleResetEnvCfg, **kwargs) -> None:
        self.s0_projector = None
        super().__init__(cfg, **kwargs)
        from .hardware_compatible_action import Dex5IsaacHardwareCompatibilityWrapper

        self.s0_projector = Dex5IsaacHardwareCompatibilityWrapper(
            self.joint_lower, self.joint_upper
        )
        all_ids = self.hand._ALL_INDICES
        projected = self.s0_projector.project_targets(self.source_targets[all_ids])
        self.source_targets[all_ids] = projected
        self.processed_targets[all_ids] = projected
        self.hand.set_joint_position_target(
            projected, joint_ids=self.dex5_joint_ids, env_ids=all_ids
        )

    def set_replay_control_step(self, control_step: int) -> None:
        super().set_replay_control_step(control_step)
        if self.s0_projector is not None:
            self.source_targets[:] = self.s0_projector.project_targets(
                self.source_targets
            )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            ids = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(ids)
        if self.s0_projector is not None:
            projected = self.s0_projector.project_targets(
                self.source_targets[ids], env_ids=ids
            )
            self.source_targets[ids] = projected
            self.processed_targets[ids] = projected
            self.hand.set_joint_position_target(
                projected, joint_ids=self.dex5_joint_ids, env_ids=ids
            )

    def _apply_action(self) -> None:
        if self.s0_projector is None:
            super()._apply_action()
            return
        projected = self.s0_projector.project_targets(self.source_targets)
        self.source_targets.copy_(projected)
        self.processed_targets.copy_(projected)
        self.hand.set_joint_position_target(
            self.processed_targets, joint_ids=self.dex5_joint_ids
        )
