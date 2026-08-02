from __future__ import annotations

from collections.abc import Sequence
import os

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul

from .index_near_surface_physics_env import (
    Dex5IndexNearSurfacePhysicsEnv,
    Dex5IndexNearSurfacePhysicsEnvCfg,
)


@configclass
class Dex5CanonicalResetPhysicsEnvCfg(Dex5IndexNearSurfacePhysicsEnvCfg):
    """Physics-only environment for O0 canonical reset validation."""

    episode_length_s = 0.25
    action_scale = 0.0
    reset_joint_pos_noise = 0.0
    reset_joint_vel_noise = 0.0
    reset_pen_pos_noise = 0.0
    reset_pen_vel_noise = 0.0
    penetration_threshold = 0.0005
    drop_threshold = 0.020
    drift_threshold = 0.025
    canonical_reset_pen_rot_noise_rad = 0.0
    canonical_mass_scale_range = (1.0, 1.0)
    canonical_static_friction_scale_range = (1.0, 1.0)
    canonical_dynamic_friction_scale_range = (1.0, 1.0)


class Dex5CanonicalResetPhysicsEnv(Dex5IndexNearSurfacePhysicsEnv):
    """Restores complete O0 state records and holds their actuator targets."""

    cfg: Dex5CanonicalResetPhysicsEnvCfg

    def _load_bundle(self) -> None:
        with np.load(self.cfg.bundle_path, allow_pickle=False) as data:
            self.source_qpos_bank = torch.as_tensor(
                np.concatenate(
                    (
                        np.asarray(data["robot_qpos"], dtype=np.float32),
                        np.asarray(data["object_pose"], dtype=np.float32),
                    ),
                    axis=1,
                ),
                device=self.device,
            )
            self.source_qvel_bank = torch.as_tensor(
                np.concatenate(
                    (
                        np.asarray(data["robot_qvel"], dtype=np.float32),
                        np.asarray(data["object_linear_velocity"], dtype=np.float32),
                        np.asarray(data["object_angular_velocity"], dtype=np.float32),
                    ),
                    axis=1,
                ),
                device=self.device,
            )
            q_target = np.asarray(data["robot_q_target"], dtype=np.float32)
            previous_action = np.asarray(data["robot_previous_action"], dtype=np.float32)
            self.source_target_bank = torch.as_tensor(q_target, device=self.device)
            self.source_previous_action_bank = torch.as_tensor(
                previous_action, device=self.device
            )
            self.continuation_target_bank = torch.as_tensor(
                np.asarray(data["robot_continuation_q_target"], dtype=np.float32),
                device=self.device,
            )
            self.continuation_length_bank = torch.as_tensor(
                np.asarray(data["robot_continuation_valid_length"], dtype=np.int64),
                device=self.device,
            )
            self.target_hand_bank = torch.as_tensor(q_target, device=self.device)
            self.target_pen_pos_bank = torch.as_tensor(
                np.asarray(data["object_pose"][:, :3], dtype=np.float32),
                device=self.device,
            )
            self.target_pen_quat_bank = torch.as_tensor(
                np.asarray(data["object_pose"][:, 3:7], dtype=np.float32),
                device=self.device,
            )
            self.canonical_id_bank = np.asarray(data["metadata_canonical_id"])
            self.state_type_bank = np.asarray(data["labels_static_or_dynamic"])
            self.state_id_bank = np.asarray(data["metadata_state_id"], dtype=np.int64)
            self.mass_bank = torch.as_tensor(
                np.asarray(data["physics_mass"], dtype=np.float32), device=self.device
            )
            self.static_friction_bank = torch.as_tensor(
                np.asarray(data["physics_static_friction"], dtype=np.float32),
                device=self.device,
            )
            self.dynamic_friction_bank = torch.as_tensor(
                np.asarray(data["physics_dynamic_friction"], dtype=np.float32),
                device=self.device,
            )
            self.com_offset_bank = torch.as_tensor(
                np.asarray(data["physics_com_offset"], dtype=np.float32),
                device=self.device,
            )
            self.bank_size = self.source_qpos_bank.shape[0]

    def set_replay_control_step(self, control_step: int) -> None:
        bank_ids = self.current_bank_ids
        lengths = self.continuation_length_bank[bank_ids]
        horizon = self.continuation_target_bank.shape[1]
        step = torch.full_like(lengths, int(control_step)).clamp_max(horizon - 1)
        rows = torch.arange(self.num_envs, device=self.device)
        continuation = self.continuation_target_bank[bank_ids, step]
        use_continuation = lengths > int(control_step)
        self.source_targets[:] = torch.where(
            use_continuation.unsqueeze(-1), continuation, self.source_targets
        )

    def _apply_action(self) -> None:
        # q_target is the actual actuator target captured with the state. The
        # validator must not add the saved residual action a second time.
        self.processed_targets.copy_(self.source_targets)
        self.hand.set_joint_position_target(
            self.processed_targets, joint_ids=self.dex5_joint_ids
        )

    def _set_row_physics(
        self, env_ids: torch.Tensor, bank_ids: torch.Tensor
    ) -> None:
        debug = os.environ.get("DEX5_DEBUG_ENV_INIT") == "1"
        if debug:
            print("[DEX5_DEBUG] canonical _set_row_physics begin", flush=True)
        count = len(env_ids)
        mass_low, mass_high = self.cfg.canonical_mass_scale_range
        static_low, static_high = self.cfg.canonical_static_friction_scale_range
        dynamic_low, dynamic_high = self.cfg.canonical_dynamic_friction_scale_range

        mass_scale = mass_low + (mass_high - mass_low) * torch.rand(
            count, device=self.device
        )
        masses = self.pen.root_physx_view.get_masses().clone()
        masses[env_ids.cpu(), 0] = (self.mass_bank[bank_ids] * mass_scale).to(
            masses.device
        )
        self.pen.root_physx_view.set_masses(masses, env_ids.cpu())
        if debug:
            print("[DEX5_DEBUG] canonical masses set", flush=True)

        static_scale = static_low + (static_high - static_low) * torch.rand(
            count, device=self.device
        )
        dynamic_scale = dynamic_low + (dynamic_high - dynamic_low) * torch.rand(
            count, device=self.device
        )
        static = self.static_friction_bank[bank_ids] * static_scale
        dynamic = torch.minimum(
            self.dynamic_friction_bank[bank_ids] * dynamic_scale, static
        )
        materials = self.pen.root_physx_view.get_material_properties().clone()
        materials[env_ids.cpu(), :, 0] = static.to(materials.device).unsqueeze(-1)
        materials[env_ids.cpu(), :, 1] = dynamic.to(materials.device).unsqueeze(-1)
        self.pen.root_physx_view.set_material_properties(materials, env_ids.cpu())
        if debug:
            print("[DEX5_DEBUG] canonical materials set", flush=True)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        debug = os.environ.get("DEX5_DEBUG_ENV_INIT") == "1"
        if debug:
            print("[DEX5_DEBUG] canonical _reset_idx begin", flush=True)
        if env_ids is None:
            env_ids_tensor = self.hand._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        super()._reset_idx(env_ids_tensor)
        if debug:
            print("[DEX5_DEBUG] canonical super reset done", flush=True)
        bank_ids = self.current_bank_ids[env_ids_tensor]
        self.actions[env_ids_tensor] = self.source_previous_action_bank[bank_ids]
        self.previous_actions[env_ids_tensor] = self.actions[env_ids_tensor]
        self.processed_targets[env_ids_tensor] = self.source_targets[env_ids_tensor]
        self.hand.set_joint_position_target(
            self.source_targets[env_ids_tensor],
            joint_ids=self.dex5_joint_ids,
            env_ids=env_ids_tensor,
        )
        self._set_row_physics(env_ids_tensor, bank_ids)
        if debug:
            print("[DEX5_DEBUG] canonical _reset_idx physics done", flush=True)

        rotation_noise = float(self.cfg.canonical_reset_pen_rot_noise_rad)
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
