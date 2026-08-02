from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from dex5_penspin.isaac.m3_2_30_first_leg_reward import compute_index_near_surface_reward


ROOT = Path(__file__).resolve().parents[5]
DEX5_USD_PATH = str(ROOT / "dex5_penspin/isaac/assets/dex5_hand.usd")
DEFAULT_BUNDLE_PATH = str(
    ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_first_leg_bundle.npz"
)

DEX5_JOINT_ORDER = (
    "Roll_21R",
    "Pitch_22R",
    "Pitch_23R",
    "Pitch_24R",
    "Roll_31R",
    "Pitch_32R",
    "Pitch_33R",
    "Pitch_34R",
    "Roll_41R",
    "Pitch_42R",
    "Pitch_43R",
    "Pitch_44R",
    "Roll_51R",
    "Pitch_52R",
    "Pitch_53R",
    "Pitch_54R",
    "Yaw_11R",
    "Roll_12R",
    "Pitch_13R",
    "Pitch_14R",
)

FINGERTIP_BODY_NAMES = ("Link_14R", "Link_24R", "Link_34R", "Link_44R", "Link_54R")
FINGERTIP_LOCAL_OFFSETS = (
    (0.026, 0.002, 0.0),
    (0.020, -0.004, 0.0),
    (0.020, -0.004, 0.0),
    (0.020, -0.004, 0.0),
    (0.018, -0.004, 0.0),
)
FINGERTIP_RADII = (0.012, 0.012, 0.012, 0.012, 0.011)


@configclass
class Dex5IndexNearSurfacePhysicsEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 0.04
    action_space = 20
    observation_space = 126
    state_space = 0

    # The measured COM is 61.4 mm from the rear end of a 141.5 mm pen,
    # i.e. 9.35 mm away from the geometric center toward local +X (rear).
    # This matches the measured MuJoCo model's inertial frame convention.
    pen_com_offset_x = 0.00935

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.05,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**18,
        ),
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEX5_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                linear_damping=0.0,
                angular_damping=0.01,
                max_linear_velocity=10.0,
                max_angular_velocity=80.0,
                max_depenetration_velocity=0.2,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.0,
                stabilization_threshold=0.0005,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.93,
                velocity_limit_sim=4.18,
                stiffness=8.0,
                damping=0.2,
                friction=0.01,
            )
        },
        soft_joint_pos_limit_factor=1.0,
    )

    pen_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pen",
        spawn=sim_utils.CylinderCfg(
            # Equivalent cylinder for the measured 141.5 mm, tapered representative pen.
            # MuJoCo uses the three measured section diameters explicitly.
            radius=0.005083333333333333,
            height=0.1415,
            axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=True,
                linear_damping=0.002,
                angular_damping=0.002,
                max_linear_velocity=5.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=0.2,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.0,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0094),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=0.6,
                dynamic_friction=0.45,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.20, 0.14)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.005, 0.0, 0.145)),
    )

    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Link_.*4R",
        update_period=0.0,
        history_length=2,
        track_air_time=False,
        max_contact_data_count_per_prim=8,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=0.32,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    bundle_path = DEFAULT_BUNDLE_PATH
    action_scale = 0.040
    reset_joint_pos_noise = 0.004
    reset_joint_vel_noise = 0.020
    fixed_reset_bank_id = -1
    reset_pen_pos_noise = 0.0008
    reset_pen_vel_noise = 0.010
    pen_radius = 0.005083333333333333
    pen_half_length = 0.07075
    drop_threshold = 0.035
    drift_threshold = 0.060
    penetration_threshold = 0.001
    use_fingertip_collision_proxies = True
    enable_contact_sensor = True
    collision_proxy_radius_scale = 0.85
    collision_proxy_radius_scales: tuple[float, float, float, float, float] | None = None


class Dex5IndexNearSurfacePhysicsEnv(DirectRLEnv):
    cfg: Dex5IndexNearSurfacePhysicsEnvCfg

    def __init__(self, cfg: Dex5IndexNearSurfacePhysicsEnvCfg, render_mode: str | None = None, **kwargs):
        if os.environ.get("DEX5_DEBUG_ENV_INIT") == "1":
            print("[DEX5_DEBUG] DirectRLEnv.__init__ begin", flush=True)
        super().__init__(cfg, render_mode, **kwargs)
        if os.environ.get("DEX5_DEBUG_ENV_INIT") == "1":
            print("[DEX5_DEBUG] DirectRLEnv.__init__ done", flush=True)

        missing_joints = [name for name in DEX5_JOINT_ORDER if name not in self.hand.joint_names]
        if missing_joints:
            raise RuntimeError(f"Dex5 USD is missing joints: {missing_joints}")
        missing_bodies = [name for name in FINGERTIP_BODY_NAMES if name not in self.hand.body_names]
        if missing_bodies:
            raise RuntimeError(f"Dex5 USD is missing fingertip bodies: {missing_bodies}")

        self.dex5_joint_ids = [self.hand.joint_names.index(name) for name in DEX5_JOINT_ORDER]
        self.fingertip_body_ids = [self.hand.body_names.index(name) for name in FINGERTIP_BODY_NAMES]
        if self.cfg.enable_contact_sensor:
            self.sensor_body_ids = [
                self.contact_sensor.body_names.index(name) for name in FINGERTIP_BODY_NAMES
            ]
        else:
            self.sensor_body_ids = []

        joint_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.joint_lower = joint_limits[:, self.dex5_joint_ids, 0]
        self.joint_upper = joint_limits[:, self.dex5_joint_ids, 1]

        pen_coms = self.pen.root_physx_view.get_coms().clone()
        pen_coms[..., 0] = float(self.cfg.pen_com_offset_x)
        pen_indices = torch.arange(self.num_envs, dtype=torch.int32, device="cpu")
        self.pen.root_physx_view.set_coms(pen_coms, pen_indices)

        self.fingertip_offsets = torch.tensor(
            FINGERTIP_LOCAL_OFFSETS, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        radius_scales = self.cfg.collision_proxy_radius_scales or (self.cfg.collision_proxy_radius_scale,) * 5
        self.collision_proxy_radius_scales = radius_scales
        self.fingertip_radii = torch.tensor(
            [radius * scale for radius, scale in zip(FINGERTIP_RADII, radius_scales, strict=True)],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self.pen_x_axis = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.processed_targets = torch.zeros_like(self.actions)
        self.current_bank_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.source_targets = torch.zeros_like(self.actions)
        self.target_hand_qpos = torch.zeros_like(self.actions)
        self.target_pen_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pen_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_pen_quat[:, 0] = 1.0
        self.init_pen_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_pen_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_pen_axis[:, 0] = 1.0
        self.initial_index_tip_dist = torch.full((self.num_envs,), 0.030, device=self.device)
        self.needs_initial_distance = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_max_penetration = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_penetration_by_finger = torch.zeros(
            (self.num_envs, len(FINGERTIP_BODY_NAMES)), device=self.device
        )
        self.current_max_penetration = torch.zeros(self.num_envs, device=self.device)
        self.current_penetration_by_finger = torch.zeros(
            (self.num_envs, len(FINGERTIP_BODY_NAMES)), device=self.device
        )

        if os.environ.get("DEX5_DEBUG_ENV_INIT") == "1":
            print("[DEX5_DEBUG] _load_bundle begin", flush=True)
        self._load_bundle()
        if os.environ.get("DEX5_DEBUG_ENV_INIT") == "1":
            print("[DEX5_DEBUG] _load_bundle done", flush=True)
        self._compute_intermediate_values()
        if os.environ.get("DEX5_DEBUG_ENV_INIT") == "1":
            print("[DEX5_DEBUG] initial _compute_intermediate_values done", flush=True)

    def _setup_scene(self) -> None:
        self.hand = Articulation(self.cfg.robot_cfg)
        if self.cfg.use_fingertip_collision_proxies:
            proxy_material = sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=0.6,
                dynamic_friction=0.45,
                restitution=0.0,
            )
            radius_scales = self.cfg.collision_proxy_radius_scales or (self.cfg.collision_proxy_radius_scale,) * 5
            for body_name, offset, radius, radius_scale in zip(
                FINGERTIP_BODY_NAMES,
                FINGERTIP_LOCAL_OFFSETS,
                FINGERTIP_RADII,
                radius_scales,
                strict=True,
            ):
                proxy_cfg = sim_utils.SphereCfg(
                    radius=radius_scale * radius,
                    visible=False,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                        contact_offset=0.001,
                        rest_offset=0.0,
                    ),
                    physics_material=proxy_material,
                )
                proxy_cfg.func(
                    f"/World/envs/env_0/Robot/{body_name}/TipCollisionProxy",
                    proxy_cfg,
                    translation=offset,
                )
        self.pen = RigidObject(self.cfg.pen_cfg)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["pen"] = self.pen
        if self.cfg.enable_contact_sensor:
            self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
            self.scene.sensors["fingertip_contacts"] = self.contact_sensor

        ground_cfg = sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.22, 0.24, 0.27),
            ),
        )
        ground_cfg.func(
            "/World/ground", ground_cfg, translation=(0.0, 0.0, -0.075)
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=1400.0, color=(0.80, 0.80, 0.80))
        light_cfg.func("/World/Light", light_cfg)

    def _load_bundle(self) -> None:
        data = np.load(self.cfg.bundle_path)
        source_qpos_rows: list[np.ndarray] = []
        source_qvel_rows: list[np.ndarray] = []
        source_target_rows: list[np.ndarray] = []
        target_hand_rows: list[np.ndarray] = []
        target_pen_pos_rows: list[np.ndarray] = []
        target_pen_quat_rows: list[np.ndarray] = []

        for window in ("016_019", "019_022"):
            source_qpos = np.asarray(data[f"{window}_source_qpos"], dtype=np.float32)
            count = source_qpos.shape[0]
            source_qpos_rows.append(source_qpos)
            source_qvel_rows.append(np.asarray(data[f"{window}_source_qvel"], dtype=np.float32))
            source_target_rows.append(np.asarray(data[f"{window}_source_target_qpos"], dtype=np.float32))
            target_hand_rows.append(
                np.repeat(np.asarray(data[f"{window}_target_hand_qpos"], dtype=np.float32)[None, :], count, axis=0)
            )
            target_pen_pos_rows.append(
                np.repeat(np.asarray(data[f"{window}_target_pen_pos"], dtype=np.float32)[None, :], count, axis=0)
            )
            target_pen_quat_rows.append(
                np.repeat(np.asarray(data[f"{window}_target_pen_quat"], dtype=np.float32)[None, :], count, axis=0)
            )

        self.source_qpos_bank = torch.as_tensor(np.concatenate(source_qpos_rows), device=self.device)
        self.source_qvel_bank = torch.as_tensor(np.concatenate(source_qvel_rows), device=self.device)
        self.source_target_bank = torch.as_tensor(np.concatenate(source_target_rows), device=self.device)
        self.target_hand_bank = torch.as_tensor(np.concatenate(target_hand_rows), device=self.device)
        self.target_pen_pos_bank = torch.as_tensor(np.concatenate(target_pen_pos_rows), device=self.device)
        self.target_pen_quat_bank = torch.as_tensor(np.concatenate(target_pen_quat_rows), device=self.device)
        self.bank_size = self.source_qpos_bank.shape[0]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions = torch.clamp(actions, -1.0, 1.0)
        prefill_steps = int(getattr(self.cfg, "teacher_prefill_steps", 0))
        if (
            prefill_steps > 0
            and hasattr(self, "teacher_action_bank")
            and self.teacher_action_horizon > 0
        ):
            step_ids = self.episode_length_buf.clamp_max(self.teacher_action_horizon - 1)
            teacher_actions = self.teacher_action_bank[self.current_bank_ids, step_ids]
            use_teacher = (self.episode_length_buf < prefill_steps).unsqueeze(-1)
            blend = float(getattr(self.cfg, "teacher_action_blend", 1.0))
            self.actions = torch.where(
                use_teacher,
                torch.lerp(self.actions, teacher_actions, blend),
                self.actions,
            )

    def _apply_action(self) -> None:
        policy_horizon = getattr(self, "policy_horizon_env", self.max_episode_length)
        if isinstance(policy_horizon, torch.Tensor):
            horizon_denominator = (policy_horizon.float() - 1.0).clamp_min(1.0)
        else:
            horizon_denominator = max(float(policy_horizon - 1), 1.0)
        tau = torch.clamp(
            self.episode_length_buf.float() / horizon_denominator, 0.0, 1.0
        )
        tau = tau * tau * (3.0 - 2.0 * tau)
        teacher_target = torch.lerp(self.source_targets, self.target_hand_qpos, tau.unsqueeze(-1))
        action_scale = getattr(self, "action_scale_env", self.cfg.action_scale)
        if isinstance(action_scale, torch.Tensor):
            action_scale = action_scale.unsqueeze(-1)
        self.processed_targets = teacher_target + action_scale * self.actions
        self.processed_targets = torch.maximum(torch.minimum(self.processed_targets, self.joint_upper), self.joint_lower)
        self.hand.set_joint_position_target(self.processed_targets, joint_ids=self.dex5_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._compute_intermediate_values()
        if torch.any(self.needs_initial_distance):
            env_ids = torch.nonzero(self.needs_initial_distance, as_tuple=False).squeeze(-1)
            self.initial_index_tip_dist[env_ids] = self.tip_surface_dist[env_ids, 1]
            self.needs_initial_distance[env_ids] = False
        policy_horizon = getattr(self, "policy_horizon_env", self.max_episode_length)
        if isinstance(policy_horizon, torch.Tensor):
            horizon_denominator = (policy_horizon.float() - 1.0).clamp_min(1.0)
        else:
            horizon_denominator = max(float(policy_horizon - 1), 1.0)
        tau = torch.clamp(
            self.episode_length_buf.float() / horizon_denominator, 0.0, 1.0
        )
        target_quat_error = quat_mul(self.target_pen_quat, quat_conjugate(self.pen_quat))
        obs = torch.cat(
            (
                self.hand_qpos,
                0.2 * self.hand_qvel,
                self.pen_pos,
                self.pen_quat,
                self.pen_linvel,
                0.1 * self.pen_angvel,
                (self.fingertip_pos_local - self.pen_pos.unsqueeze(1)).reshape(self.num_envs, -1),
                self.tip_surface_dist,
                0.1 * self.contact_force_norm,
                self.target_hand_qpos - self.hand_qpos,
                self.target_pen_pos - self.pen_pos,
                target_quat_error,
                self.actions,
                tau.unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        failed, _ = self._compute_dones()
        reward, terms = compute_index_near_surface_reward(
            index_tip_dist=self.tip_surface_dist[:, 1],
            initial_index_tip_dist=self.initial_index_tip_dist,
            obj_pos=self.pen_pos,
            init_obj_pos=self.init_pen_pos,
            target_obj_pos=self.target_pen_pos,
            max_penetration=self.episode_max_penetration,
            actions=self.actions,
            prev_actions=self.previous_actions,
            progress_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            failed=failed,
        )
        self.extras["log"] = {f"reward/{name}": value.mean() for name, value in terms.items()}
        self.extras["log"].update(
            {
                "metric/index_tip_dist_mm": self.tip_surface_dist[:, 1].mean() * 1000.0,
                "metric/index_near_14mm_ratio": (self.tip_surface_dist[:, 1] <= 0.014).float().mean(),
                "metric/index_contact_ratio": (self.contact_force_norm[:, 1] > 0.05).float().mean(),
                "metric/pen_height_drop_mm": (self.init_pen_pos[:, 2] - self.pen_pos[:, 2]).mean() * 1000.0,
                "metric/max_penetration_mm": self.episode_max_penetration.mean() * 1000.0,
            }
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        failed, time_out = self._compute_dones()
        return failed, time_out

    def _compute_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        height_drop = self.init_pen_pos[:, 2] - self.pen_pos[:, 2]
        drift = torch.linalg.norm(self.pen_pos - self.init_pen_pos, dim=-1)
        invalid = ~torch.isfinite(self.pen_pos).all(dim=-1) | ~torch.isfinite(self.hand_qpos).all(dim=-1)
        failed = (
            (height_drop > self.cfg.drop_threshold)
            | (drift > self.cfg.drift_threshold)
            | (self.episode_max_penetration > self.cfg.penetration_threshold)
            | invalid
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return failed, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self.hand.reset(env_ids)
        self.pen.reset(env_ids)
        super()._reset_idx(env_ids)

        count = len(env_ids)
        if int(self.cfg.fixed_reset_bank_id) >= 0:
            fixed_bank_id = int(self.cfg.fixed_reset_bank_id)
            if fixed_bank_id >= self.bank_size:
                raise ValueError(
                    f"fixed_reset_bank_id={fixed_bank_id} exceeds "
                    f"bank_size={self.bank_size}"
                )
            bank_ids = torch.full(
                (count,),
                fixed_bank_id,
                dtype=torch.long,
                device=self.device,
            )
        elif hasattr(self, "forced_bank_ids"):
            bank_ids = self.forced_bank_ids[:count].to(device=self.device, dtype=torch.long)
            if bank_ids.numel() != count:
                raise ValueError(
                    f"forced_bank_ids has {bank_ids.numel()} rows, but reset needs {count}"
                )
            if torch.any((bank_ids < 0) | (bank_ids >= self.bank_size)):
                raise ValueError(
                    f"forced_bank_ids must be within [0, {self.bank_size})"
                )
        else:
            bank_ids = torch.randint(
                0, self.bank_size, (count,), device=self.device
            )
        self.current_bank_ids[env_ids] = bank_ids
        if hasattr(self, "phase_id_bank"):
            self.phase_ids[env_ids] = self.phase_id_bank[bank_ids]
        source_qpos = self.source_qpos_bank[bank_ids]
        source_qvel = self.source_qvel_bank[bank_ids]

        hand_pos = source_qpos[:, :20].clone()
        hand_vel = source_qvel[:, :20].clone()
        hand_pos += self.cfg.reset_joint_pos_noise * torch.randn_like(hand_pos)
        hand_vel += self.cfg.reset_joint_vel_noise * torch.randn_like(hand_vel)
        hand_pos = torch.maximum(torch.minimum(hand_pos, self.joint_upper[env_ids]), self.joint_lower[env_ids])

        self.source_targets[env_ids] = self.source_target_bank[bank_ids]
        self.target_hand_qpos[env_ids] = self.target_hand_bank[bank_ids]
        self.target_pen_pos[env_ids] = self.target_pen_pos_bank[bank_ids]
        self.target_pen_quat[env_ids] = self.target_pen_quat_bank[bank_ids]
        self.processed_targets[env_ids] = self.source_targets[env_ids]
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

        self.hand.write_joint_state_to_sim(
            hand_pos,
            hand_vel,
            joint_ids=self.dex5_joint_ids,
            env_ids=env_ids,
        )
        self.hand.set_joint_position_target(
            self.source_targets[env_ids],
            joint_ids=self.dex5_joint_ids,
            env_ids=env_ids,
        )

        pen_pose = source_qpos[:, 20:27].clone()
        pen_pose[:, :3] += self.cfg.reset_pen_pos_noise * torch.randn_like(pen_pose[:, :3])
        pen_pose[:, :3] += self.scene.env_origins[env_ids]
        pen_velocity = source_qvel[:, 20:26].clone()
        pen_velocity += self.cfg.reset_pen_vel_noise * torch.randn_like(pen_velocity)
        self.pen.write_root_pose_to_sim(pen_pose, env_ids)
        self.pen.write_root_velocity_to_sim(pen_velocity, env_ids)

        self.init_pen_pos[env_ids] = pen_pose[:, :3] - self.scene.env_origins[env_ids]
        self.init_pen_axis[env_ids] = quat_apply(pen_pose[:, 3:7], self.pen_x_axis[env_ids])
        self.needs_initial_distance[env_ids] = True
        self.episode_max_penetration[env_ids] = 0.0
        self.episode_max_penetration_by_finger[env_ids] = 0.0

    def _compute_intermediate_values(self) -> None:
        self.hand_qpos = self.hand.data.joint_pos[:, self.dex5_joint_ids]
        self.hand_qvel = self.hand.data.joint_vel[:, self.dex5_joint_ids]
        self.pen_pos = self.pen.data.root_pos_w - self.scene.env_origins
        self.pen_quat = self.pen.data.root_quat_w
        self.pen_linvel = self.pen.data.root_lin_vel_w
        self.pen_angvel = self.pen.data.root_ang_vel_w

        body_pos = self.hand.data.body_pos_w[:, self.fingertip_body_ids]
        body_quat = self.hand.data.body_quat_w[:, self.fingertip_body_ids]
        offsets = self.fingertip_offsets.expand(self.num_envs, -1, -1)
        self.fingertip_pos = body_pos + quat_apply(body_quat, offsets)
        self.fingertip_pos_local = self.fingertip_pos - self.scene.env_origins.unsqueeze(1)

        self.pen_axis = quat_apply(self.pen_quat, self.pen_x_axis)
        end_a = self.pen.data.root_pos_w - self.cfg.pen_half_length * self.pen_axis
        end_b = self.pen.data.root_pos_w + self.cfg.pen_half_length * self.pen_axis
        segment = end_b - end_a
        segment_norm_sq = torch.sum(segment * segment, dim=-1, keepdim=True).clamp_min(1e-9)
        rel = self.fingertip_pos - end_a.unsqueeze(1)
        projection = torch.sum(rel * segment.unsqueeze(1), dim=-1) / segment_norm_sq
        projection = torch.clamp(projection, 0.0, 1.0)
        closest = end_a.unsqueeze(1) + projection.unsqueeze(-1) * segment.unsqueeze(1)
        self.tip_surface_dist = torch.linalg.norm(self.fingertip_pos - closest, dim=-1) - self.cfg.pen_radius

        radial = self.fingertip_pos - closest
        radial_norm = torch.linalg.norm(radial, dim=-1, keepdim=True).clamp_min(1.0e-9)
        self.contact_radial_direction_w = radial / radial_norm
        body_linvel = self.hand.data.body_lin_vel_w[:, self.fingertip_body_ids]
        body_angvel = self.hand.data.body_ang_vel_w[:, self.fingertip_body_ids]
        fingertip_offset_w = self.fingertip_pos - body_pos
        self.fingertip_velocity_w = body_linvel + torch.linalg.cross(
            body_angvel, fingertip_offset_w, dim=-1
        )
        pen_point_offset_w = closest - self.pen.data.root_pos_w.unsqueeze(1)
        pen_point_velocity_w = self.pen_linvel.unsqueeze(1) + torch.linalg.cross(
            self.pen_angvel.unsqueeze(1).expand_as(pen_point_offset_w),
            pen_point_offset_w,
            dim=-1,
        )
        self.tip_pen_relative_velocity_w = self.fingertip_velocity_w - pen_point_velocity_w
        self.contact_normal_relative_speed = torch.sum(
            self.tip_pen_relative_velocity_w * self.contact_radial_direction_w, dim=-1
        )
        self.contact_tangential_relative_velocity_w = self.tip_pen_relative_velocity_w - (
            self.contact_normal_relative_speed.unsqueeze(-1)
            * self.contact_radial_direction_w
        )
        self.contact_tangential_slip_speed = torch.linalg.norm(
            self.contact_tangential_relative_velocity_w, dim=-1
        )
        pen_axis_at_contact = self.pen_axis.unsqueeze(1).expand_as(
            self.contact_radial_direction_w
        )
        tangent_axis = pen_axis_at_contact - torch.sum(
            pen_axis_at_contact * self.contact_radial_direction_w,
            dim=-1,
            keepdim=True,
        ) * self.contact_radial_direction_w
        self.contact_tangent_axis_w = tangent_axis / torch.linalg.norm(
            tangent_axis, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)
        self.contact_binormal_axis_w = torch.linalg.cross(
            self.contact_radial_direction_w, self.contact_tangent_axis_w, dim=-1
        )
        self.contact_tangential_velocity_components = torch.stack(
            (
                torch.sum(
                    self.contact_tangential_relative_velocity_w
                    * self.contact_tangent_axis_w,
                    dim=-1,
                ),
                torch.sum(
                    self.contact_tangential_relative_velocity_w
                    * self.contact_binormal_axis_w,
                    dim=-1,
                ),
            ),
            dim=-1,
        )

        estimated_penetration = torch.clamp(self.fingertip_radii - self.tip_surface_dist, min=0.0)
        current_max_penetration = estimated_penetration.max(dim=-1).values
        self.current_penetration_by_finger = estimated_penetration
        self.current_max_penetration = current_max_penetration
        self.episode_max_penetration = torch.maximum(self.episode_max_penetration, current_max_penetration)
        self.episode_max_penetration_by_finger = torch.maximum(
            self.episode_max_penetration_by_finger, estimated_penetration
        )

        if self.cfg.enable_contact_sensor:
            net_forces = self.contact_sensor.data.net_forces_w[:, self.sensor_body_ids]
            self.contact_force_w = net_forces
            self.contact_force_norm = torch.linalg.norm(net_forces, dim=-1)
            self.contact_normal_force_signed = torch.sum(
                net_forces * self.contact_radial_direction_w, dim=-1
            )
            tangential_force = net_forces - (
                self.contact_normal_force_signed.unsqueeze(-1)
                * self.contact_radial_direction_w
            )
            self.contact_tangential_force_norm = torch.linalg.norm(
                tangential_force, dim=-1
            )
        else:
            self.contact_force_w = torch.zeros(
                (self.num_envs, len(FINGERTIP_BODY_NAMES), 3),
                dtype=self.hand_qpos.dtype,
                device=self.device,
            )
            self.contact_force_norm = torch.zeros(
                (self.num_envs, len(FINGERTIP_BODY_NAMES)),
                dtype=self.hand_qpos.dtype,
                device=self.device,
            )
            self.contact_normal_force_signed = torch.zeros_like(self.contact_force_norm)
            self.contact_tangential_force_norm = torch.zeros_like(self.contact_force_norm)

        # Approximate each pen contact at the radial cylinder surface nearest
        # the fingertip. Contact sensors report force on the finger, so negate
        # it to obtain the corresponding force applied to the pen.
        self.pen_com_pos_w = self.pen.data.root_com_pos_w
        self.pen_com_vel_w = self.pen.data.root_com_vel_w[:, :3]
        self.contact_point_on_pen_w = closest + (
            self.cfg.pen_radius * self.contact_radial_direction_w
        )
        self.contact_force_on_pen_w = -self.contact_force_w
        contact_arm = self.contact_point_on_pen_w - self.pen_com_pos_w.unsqueeze(1)
        self.contact_torque_on_pen_about_com_w = torch.linalg.cross(
            contact_arm, self.contact_force_on_pen_w, dim=-1
        )
        self.tim_force_on_pen_w = self.contact_force_on_pen_w[:, :3].sum(dim=1)
        self.tim_torque_on_pen_w = self.contact_torque_on_pen_about_com_w[:, :3].sum(dim=1)
        self.ring_force_on_pen_w = self.contact_force_on_pen_w[:, 3]
        self.ring_torque_on_pen_w = self.contact_torque_on_pen_about_com_w[:, 3]
        self.total_contact_force_on_pen_w = self.contact_force_on_pen_w.sum(dim=1)
        self.total_contact_torque_on_pen_w = self.contact_torque_on_pen_about_com_w.sum(dim=1)
        gravity = torch.tensor(
            (0.0, 0.0, -0.0094 * 9.81),
            dtype=self.hand_qpos.dtype,
            device=self.device,
        )
        self.gravity_force_on_pen_w = gravity.unsqueeze(0).expand(self.num_envs, -1)
        self.tim_gravity_force_residual_w = self.tim_force_on_pen_w + self.gravity_force_on_pen_w
        self.total_gravity_force_residual_w = (
            self.total_contact_force_on_pen_w + self.gravity_force_on_pen_w
        )
        tim_force_norm = torch.linalg.norm(self.tim_force_on_pen_w, dim=-1).clamp_min(1.0e-9)
        self.tim_equivalent_moment_arm = (
            torch.linalg.norm(self.tim_torque_on_pen_w, dim=-1) / tim_force_norm
        )
