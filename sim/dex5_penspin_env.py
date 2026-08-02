from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - lets ROS system Python run viewer tools.
    class _FallbackEnv:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
            return None

    class _FallbackBox:
        def __init__(self, low: Any, high: Any, shape: tuple[int, ...] | None = None, dtype: Any = np.float32) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _FallbackGym:
        Env = _FallbackEnv

    class _FallbackSpaces:
        Box = _FallbackBox

    gym = _FallbackGym()
    spaces = _FallbackSpaces()


DEX5_JOINT_ORDER = [
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
]


@dataclass(frozen=True)
class Dex5PenState:
    qpos: np.ndarray
    qvel: np.ndarray
    target_qpos: np.ndarray
    last_action: np.ndarray
    pen_center: np.ndarray
    pen_quat: np.ndarray
    pen_axis: np.ndarray
    pen_angvel: np.ndarray
    pen_endpoints: np.ndarray
    rotation_phase: float


class Dex5PenSpinMinimalEnv(gym.Env):
    """Minimal MuJoCo/Gymnasium environment for the Dex5-PenSpin M2 stage.

    This environment is intentionally a lightweight interface sandbox. It uses
    Dex5 joint order and limits, a free pen object, and delta target-qpos
    actions. It is not yet the final high-fidelity Dex5 simulator.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | Path | None = None,
        render_mode: str | None = None,
        episode_seconds: float = 6.0,
        control_hz: float = 100.0,
        action_delta_limit: float = 0.03,
        pen_init_pos: tuple[float, float, float] | np.ndarray | None = None,
        grasp_scale: float = 1.0,
        init_noise_scale: float = 0.015,
        actuator_kp: float | None = None,
        actuator_kv: float | None = None,
        thumb_force_limit: float | None = None,
        hard_hold_hand: bool = False,
        visual_only_markers: bool = False,
        disable_thumb_mesh_collision: bool = False,
        disable_distal_mesh_collision: bool = False,
        thumb_tip_proxy_pos: tuple[float, float, float] | np.ndarray | None = None,
        thumb_tip_proxy_size: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "assets" / "dex5_penspin_minimal.xml"
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        if actuator_kp is not None or actuator_kv is not None:
            self._set_position_actuator_gains(actuator_kp=actuator_kp, actuator_kv=actuator_kv)
        if thumb_force_limit is not None:
            self._set_thumb_force_limit(thumb_force_limit)
        self._apply_contact_overrides(
            visual_only_markers=visual_only_markers,
            disable_thumb_mesh_collision=disable_thumb_mesh_collision,
            disable_distal_mesh_collision=disable_distal_mesh_collision,
            thumb_tip_proxy_pos=thumb_tip_proxy_pos,
            thumb_tip_proxy_size=thumb_tip_proxy_size,
        )
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.renderer: mujoco.Renderer | None = None
        self.control_hz = float(control_hz)
        self.frame_skip = max(1, int(round(1.0 / (self.control_hz * self.model.opt.timestep))))
        self.max_episode_steps = int(round(float(episode_seconds) * self.control_hz))
        self.action_delta_limit = float(action_delta_limit)
        self.grasp_scale = float(grasp_scale)
        self.init_noise_scale = float(init_noise_scale)
        self.hard_hold_hand = bool(hard_hold_hand)
        if pen_init_pos is None:
            pen_init_pos = (-0.005, 0.0, 0.145)
        self.pen_init_pos = np.asarray(pen_init_pos, dtype=np.float64)
        self.np_random = np.random.default_rng(seed)

        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in DEX5_JOINT_ORDER],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            missing = [name for name, jid in zip(DEX5_JOINT_ORDER, self.joint_ids) if jid < 0]
            raise ValueError(f"Missing Dex5 joints in model: {missing}")

        self.qpos_adr = np.asarray([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_adr = np.asarray([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.joint_low = np.asarray([self.model.jnt_range[jid, 0] for jid in self.joint_ids], dtype=np.float32)
        self.joint_high = np.asarray([self.model.jnt_range[jid, 1] for jid in self.joint_ids], dtype=np.float32)

        self.pen_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pen")
        self.pen_free_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pen_free")
        self.pen_qpos_adr = int(self.model.jnt_qposadr[self.pen_free_jid])
        self.pen_qvel_adr = int(self.model.jnt_dofadr[self.pen_free_jid])
        self.pen_end_a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pen_end_a")
        self.pen_end_b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pen_end_b")

        self.target_qpos = np.zeros(20, dtype=np.float32)
        self.last_action = np.zeros(20, dtype=np.float32)
        self.prev_phase = 0.0
        self.rotation_phase = 0.0
        self.step_count = 0

        obs_dim = 20 + 20 + 20 + 20 + 3 + 4 + 3 + 3 + 6 + 1
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.action_delta_limit,
            high=self.action_delta_limit,
            shape=(20,),
            dtype=np.float32,
        )

        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _set_position_actuator_gains(self, actuator_kp: float | None, actuator_kv: float | None) -> None:
        """Override MuJoCo position-servo gains for contact-heavy reset debugging.

        The imported Dex5 model intentionally starts with soft position servos.
        During initial-state tuning this can make a manually posed hand relax as
        soon as physics starts. Raising kp/kv is a debugging tool for checking
        whether the pose itself is valid before we tune real actuator limits.
        """
        if actuator_kp is None:
            actuator_kp = float(self.model.actuator_gainprm[0, 0])
        if actuator_kv is None:
            actuator_kv = float(-self.model.actuator_biasprm[0, 2])
        actuator_kp = float(actuator_kp)
        actuator_kv = float(actuator_kv)
        self.model.actuator_gainprm[:, 0] = actuator_kp
        self.model.actuator_biasprm[:, 1] = -actuator_kp
        self.model.actuator_biasprm[:, 2] = -actuator_kv

    def _set_thumb_force_limit(self, force_limit: float) -> None:
        """Override the imported thumb joint actuator force range for debugging."""
        force_limit = float(force_limit)
        for joint_name in ["Yaw_11R", "Roll_12R", "Pitch_13R", "Pitch_14R"]:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            if force_limit <= 0.0:
                self.model.jnt_actfrclimited[joint_id] = False
            else:
                self.model.jnt_actfrclimited[joint_id] = True
                self.model.jnt_actfrcrange[joint_id] = np.array([-force_limit, force_limit], dtype=np.float64)

    def _apply_contact_overrides(
        self,
        *,
        visual_only_markers: bool,
        disable_thumb_mesh_collision: bool,
        disable_distal_mesh_collision: bool,
        thumb_tip_proxy_pos: tuple[float, float, float] | np.ndarray | None,
        thumb_tip_proxy_size: float | None,
    ) -> None:
        if visual_only_markers:
            for geom_name in ["pen_marker_a", "pen_marker_b"]:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if geom_id >= 0:
                    self.model.geom_contype[geom_id] = 0
                    self.model.geom_conaffinity[geom_id] = 0

        if disable_thumb_mesh_collision:
            for body_name in ["Link_11R", "Link_12R", "Link_13R", "Link_14R"]:
                self._disable_body_mesh_collision(body_name)

        if disable_distal_mesh_collision:
            for body_name in ["Link_14R", "Link_24R", "Link_34R", "Link_44R", "Link_54R"]:
                self._disable_body_mesh_collision(body_name)

        thumb_proxy_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "Link_14R_tip_proxy")
        if thumb_proxy_id >= 0:
            if thumb_tip_proxy_pos is not None:
                self.model.geom_pos[thumb_proxy_id] = np.asarray(thumb_tip_proxy_pos, dtype=np.float64)
            if thumb_tip_proxy_size is not None:
                self.model.geom_size[thumb_proxy_id, 0] = float(thumb_tip_proxy_size)
            if thumb_tip_proxy_pos is not None or thumb_tip_proxy_size is not None:
                self.model.geom_friction[thumb_proxy_id] = np.array([3.0, 0.12, 0.02], dtype=np.float64)

    def _disable_body_mesh_collision(self, body_name: str) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] != body_id:
                continue
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if "tip_proxy" in geom_name:
                continue
            if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        canonical = self._canonical_grasp_qpos()
        noise = self.np_random.normal(0.0, self.init_noise_scale, size=20)
        self.target_qpos = np.clip(canonical + noise, self.joint_low, self.joint_high).astype(np.float32)
        self.data.qpos[self.qpos_adr] = self.target_qpos
        self.data.ctrl[:] = self.target_qpos

        pen_xyz = self.pen_init_pos.copy()
        pen_xyz += self.np_random.normal(0.0, self.init_noise_scale * 0.13, size=3)
        self.data.qpos[self.pen_qpos_adr : self.pen_qpos_adr + 3] = pen_xyz
        self.data.qpos[self.pen_qpos_adr + 3 : self.pen_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0

        self.last_action[:] = 0.0
        self.prev_phase = 0.0
        self.rotation_phase = 0.0
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs, self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.action_delta_limit, self.action_delta_limit)

        self.target_qpos = np.clip(self.target_qpos + action, self.joint_low, self.joint_high)
        self._enforce_shared_motor_consistency()
        self.data.ctrl[:] = self.target_qpos
        for _ in range(self.frame_skip):
            if self.hard_hold_hand:
                self._hard_hold_hand_state()
            mujoco.mj_step(self.model, self.data)
        if self.hard_hold_hand:
            self._hard_hold_hand_state(forward=True)

        self.last_action = action
        self.step_count += 1
        self._update_rotation_phase()

        obs = self._get_obs()
        reward, reward_terms = self._reward_terms(action)
        terminated = bool(self._pen_dropped() or self._object_out_of_workspace())
        truncated = bool(self.step_count >= self.max_episode_steps)
        info = self._info()
        info["reward_terms"] = reward_terms
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_state(self) -> Dex5PenState:
        qpos = self.data.qpos[self.qpos_adr].copy()
        qvel = self.data.qvel[self.qvel_adr].copy()
        pen_center = self.data.xpos[self.pen_body_id].copy()
        pen_quat = self.data.xquat[self.pen_body_id].copy()
        pen_axis = self._pen_axis()
        pen_angvel = self.data.qvel[self.pen_qvel_adr + 3 : self.pen_qvel_adr + 6].copy()
        pen_endpoints = np.stack([self.data.site_xpos[self.pen_end_a_id], self.data.site_xpos[self.pen_end_b_id]]).copy()
        return Dex5PenState(
            qpos=qpos,
            qvel=qvel,
            target_qpos=self.target_qpos.copy(),
            last_action=self.last_action.copy(),
            pen_center=pen_center,
            pen_quat=pen_quat,
            pen_axis=pen_axis,
            pen_angvel=pen_angvel,
            pen_endpoints=pen_endpoints,
            rotation_phase=float(self.rotation_phase),
        )

    def _get_obs(self) -> np.ndarray:
        state = self.get_state()
        return np.concatenate(
            [
                state.qpos,
                state.qvel,
                state.target_qpos,
                state.last_action,
                state.pen_center,
                state.pen_quat,
                state.pen_axis,
                state.pen_angvel,
                state.pen_endpoints.reshape(-1),
                np.array([state.rotation_phase], dtype=np.float64),
            ]
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        state = self.get_state()
        return {
            "step_count": self.step_count,
            "qpos": state.qpos,
            "qvel": state.qvel,
            "target_qpos": state.target_qpos,
            "pen_center": state.pen_center,
            "pen_quat": state.pen_quat,
            "pen_axis": state.pen_axis,
            "pen_angvel": state.pen_angvel,
            "obj_ends": state.pen_endpoints,
            "rotation_phase": state.rotation_phase,
            "rotation_rounds": abs(state.rotation_phase) / (2.0 * np.pi),
        }

    def _canonical_grasp_qpos(self) -> np.ndarray:
        q = np.zeros(20, dtype=np.float32)
        for base in [0, 4, 8, 12]:
            q[base + 0] = 0.0
            q[base + 1] = 0.75
            q[base + 2] = 0.70
            q[base + 3] = 0.70
        q[16] = 0.10
        q[17] = 0.75
        q[18] = 0.65
        q[19] = 0.55
        flex_idx = np.asarray([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])
        q[flex_idx] = q[flex_idx] * self.grasp_scale
        return np.clip(q, self.joint_low, self.joint_high)

    def _enforce_shared_motor_consistency(self) -> None:
        for pip, dip in [(2, 3), (6, 7), (10, 11), (14, 15)]:
            v = min(self.target_qpos[pip], self.joint_high[dip])
            self.target_qpos[dip] = np.clip(v, self.joint_low[dip], self.joint_high[dip])

    def _hard_hold_hand_state(self, forward: bool = False) -> None:
        self.data.qpos[self.qpos_adr] = self.target_qpos
        self.data.qvel[self.qvel_adr] = 0.0
        self.data.ctrl[:] = self.target_qpos
        if forward:
            mujoco.mj_forward(self.model, self.data)

    def _pen_axis(self) -> np.ndarray:
        xmat = self.data.xmat[self.pen_body_id].reshape(3, 3)
        axis = xmat[:, 0].copy()
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        return axis / norm

    def _update_rotation_phase(self) -> None:
        quat = self.data.xquat[self.pen_body_id]
        phase = self._quat_twist_phase_about_x(quat)
        delta = np.arctan2(np.sin(phase - self.prev_phase), np.cos(phase - self.prev_phase))
        self.rotation_phase += float(delta)
        self.prev_phase = float(phase)

    @staticmethod
    def _quat_twist_phase_about_x(quat_wxyz: np.ndarray) -> float:
        w, x, _, _ = quat_wxyz
        return 2.0 * np.arctan2(x, w)

    def _reward_terms(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        state = self.get_state()
        spin_velocity = float(np.dot(state.pen_angvel, state.pen_axis))
        center_error = float(np.linalg.norm(state.pen_center - np.array([-0.005, 0.0, 0.145])))
        height = float(state.pen_center[2])
        action_norm = float(np.linalg.norm(action))
        alive = 1.0 if height > 0.055 else 0.0

        rotation_reward = 0.05 * spin_velocity
        center_penalty = -2.0 * center_error
        action_penalty = -0.02 * action_norm
        drop_penalty = -5.0 if self._pen_dropped() else 0.0
        alive_reward = 0.05 * alive
        reward = rotation_reward + center_penalty + action_penalty + drop_penalty + alive_reward
        return reward, {
            "rotation_reward": rotation_reward,
            "center_penalty": center_penalty,
            "action_penalty": action_penalty,
            "drop_penalty": drop_penalty,
            "alive_reward": alive_reward,
            "spin_velocity": spin_velocity,
            "center_error": center_error,
        }

    def _pen_dropped(self) -> bool:
        return bool(self.data.xpos[self.pen_body_id, 2] < 0.045)

    def _object_out_of_workspace(self) -> bool:
        center = self.data.xpos[self.pen_body_id]
        return bool(np.linalg.norm(center[:2]) > 0.25 or center[2] > 0.35)
