from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from dex5_penspin_env import Dex5PenSpinMinimalEnv
from run_m2_9_micro_gait import reset_to_preset
from tune_m2_7_tripod_fingertip_reset import SUPPORT_FINGERS, contact_snapshot, set_state


DEFAULT_PRESET = Path("dex5_penspin/reports/manus_mujoco_presets/current_manus_reset_thumb_light_clamp_m2_8.json")


def load_preset(path: str | Path = DEFAULT_PRESET) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


class Dex5PenSpinOracleHoldEnv(gym.Env):
    """Privileged-state RL wrapper for M3.1 hold-oracle training.

    The wrapped MuJoCo environment is still the M2 Dex5 pen environment. This
    class defines the RL contract we need for oracle training:

    obs:
      normalized hand state, privileged pen state, contact flags, last action.
    action:
      normalized joint target-qpos delta in [-1, 1], scaled by action_scale.
    reward:
      hold-oriented reward that protects tripod contact and pen pose stability.
    done:
      early failure if the pen leaves the workspace / support collapses badly.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        preset_path: str | Path = DEFAULT_PRESET,
        model_path: str | Path | None = None,
        episode_seconds: float = 6.0,
        control_hz: float = 100.0,
        action_scale: float = 8e-4,
        seed: int | None = None,
        render_mode: str | None = None,
        reset_hand_noise: float = 0.0,
        reset_pen_pos_noise: float = 0.0,
        reset_pen_quat_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.preset_path = Path(preset_path)
        self.preset = load_preset(self.preset_path)
        model = _as_path(model_path or self.preset.get("model_path"))
        self.action_scale = float(action_scale)
        self.reset_hand_noise = float(reset_hand_noise)
        self.reset_pen_pos_noise = float(reset_pen_pos_noise)
        self.reset_pen_quat_noise = float(reset_pen_quat_noise)
        self.np_random = np.random.default_rng(seed)

        self.env = Dex5PenSpinMinimalEnv(
            model_path=model,
            pen_init_pos=self.preset["pen_pos"],
            grasp_scale=1.0,
            init_noise_scale=0.0,
            episode_seconds=episode_seconds,
            control_hz=control_hz,
            actuator_kp=self.preset.get("actuator_kp", 400.0),
            actuator_kv=self.preset.get("actuator_kv", 20.0),
            thumb_force_limit=self.preset.get("thumb_force_limit", 0.0),
            visual_only_markers=bool(self.preset.get("visual_only_markers", True)),
            disable_thumb_mesh_collision=bool(self.preset.get("disable_thumb_mesh_collision", True)),
            disable_distal_mesh_collision=bool(self.preset.get("disable_distal_mesh_collision", True)),
            seed=seed,
            render_mode=render_mode,
        )
        self.control_hz = self.env.control_hz
        self.max_episode_steps = self.env.max_episode_steps

        self.target_center = np.asarray(self.preset["pen_pos"], dtype=np.float64)
        self.initial_center = self.target_center.copy()
        self.initial_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.initial_endpoint_dz = 0.0
        self.initial_phase = 0.0
        self._last_info: dict[str, Any] = {}

        # Observation layout:
        #   qpos_norm 20
        #   qvel_scaled 20
        #   target_qpos_norm 20
        #   pen_center_delta_scaled 3
        #   pen_axis 3
        #   pen_angvel_scaled 3
        #   endpoint_height features 2
        #   rotation_phase_scaled 1
        #   contact_flags thumb/index/middle/ring/pinky 5
        #   last_action_norm 20
        self.obs_dim = 97
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)

    @property
    def model(self) -> mujoco.MjModel:
        return self.env.model

    @property
    def data(self) -> mujoco.MjData:
        return self.env.data

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.env.reset(seed=seed)

        hand_qpos = np.asarray(self.preset["hand_qpos"], dtype=np.float64).copy()
        pen_pos = np.asarray(self.preset["pen_pos"], dtype=np.float64).copy()
        pen_quat = np.asarray(self.preset["pen_quat"], dtype=np.float64).copy()

        if self.reset_hand_noise > 0.0:
            hand_qpos += self.np_random.normal(0.0, self.reset_hand_noise, size=hand_qpos.shape)
        if self.reset_pen_pos_noise > 0.0:
            pen_pos += self.np_random.normal(0.0, self.reset_pen_pos_noise, size=pen_pos.shape)
        if self.reset_pen_quat_noise > 0.0:
            pen_quat[:3] += self.np_random.normal(0.0, self.reset_pen_quat_noise, size=3)
            pen_quat = pen_quat / np.linalg.norm(pen_quat)

        set_state(self.env, hand_qpos, pen_pos, pen_quat)
        reset_to_preset(self.env, {"hand_qpos": hand_qpos.tolist(), "pen_pos": pen_pos.tolist(), "pen_quat": pen_quat.tolist()})

        state = self.env.get_state()
        self.initial_center = state.pen_center.copy()
        self.initial_axis = state.pen_axis.copy()
        self.initial_endpoint_dz = float(abs(state.pen_endpoints[0, 2] - state.pen_endpoints[1, 2]))
        self.initial_phase = float(state.rotation_phase)
        obs = self._get_obs()
        info = self._make_info(action=np.zeros(20, dtype=np.float32), base_info=self.env._info())
        self._last_info = info
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_norm = np.asarray(action, dtype=np.float32)
        action_norm = np.clip(action_norm, -1.0, 1.0)
        scaled_action = action_norm * self.action_scale
        _, _, base_terminated, base_truncated, base_info = self.env.step(scaled_action)

        info = self._make_info(action=action_norm, base_info=base_info)
        reward = self._hold_reward(info)
        failure = self._failure(info)
        terminated = bool(base_terminated or failure)
        truncated = bool(base_truncated)
        if terminated:
            reward -= 5.0

        obs = self._get_obs(info)
        info["oracle_reward"] = float(reward)
        info["failed"] = bool(failure)
        self._last_info = info
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def _get_obs(self, info: dict[str, Any] | None = None) -> np.ndarray:
        state = self.env.get_state()
        if info is None:
            info = self._make_info(action=np.zeros(20, dtype=np.float32), base_info=self.env._info())

        q_mid = 0.5 * (self.env.joint_low + self.env.joint_high)
        q_half = 0.5 * (self.env.joint_high - self.env.joint_low) + 1e-6
        qpos_norm = (state.qpos - q_mid) / q_half
        target_norm = (state.target_qpos - q_mid) / q_half
        qvel_scaled = np.clip(state.qvel / 8.0, -5.0, 5.0)
        center_delta = (state.pen_center - self.initial_center) / 0.04
        pen_angvel = np.clip(state.pen_angvel / 15.0, -5.0, 5.0)
        endpoint_dz = float(abs(state.pen_endpoints[0, 2] - state.pen_endpoints[1, 2]))
        endpoint_features = np.array(
            [
                endpoint_dz / 0.02,
                (endpoint_dz - self.initial_endpoint_dz) / 0.02,
            ],
            dtype=np.float64,
        )
        phase_scaled = np.array([(state.rotation_phase - self.initial_phase) / np.pi], dtype=np.float64)
        contact_flags = np.array(
            [
                float(info["contact_flags"].get("thumb", 0.0)),
                float(info["contact_flags"].get("index", 0.0)),
                float(info["contact_flags"].get("middle", 0.0)),
                float(info["contact_flags"].get("ring", 0.0)),
                float(info["contact_flags"].get("pinky", 0.0)),
            ],
            dtype=np.float64,
        )
        last_action_norm = np.clip(state.last_action / max(self.action_scale, 1e-9), -5.0, 5.0)

        obs = np.concatenate(
            [
                qpos_norm,
                qvel_scaled,
                target_norm,
                center_delta,
                state.pen_axis,
                pen_angvel,
                endpoint_features,
                phase_scaled,
                contact_flags,
                last_action_norm,
            ]
        ).astype(np.float32)
        if obs.shape != (self.obs_dim,):
            raise RuntimeError(f"Unexpected obs shape {obs.shape}, expected {(self.obs_dim,)}")
        return obs

    def _make_info(self, *, action: np.ndarray, base_info: dict[str, Any]) -> dict[str, Any]:
        state = self.env.get_state()
        fingers, geoms, max_penetration = contact_snapshot(self.env)
        support_count = sum(1 for finger in SUPPORT_FINGERS if finger in fingers)
        tripod = int(support_count == len(SUPPORT_FINGERS))
        endpoint_dz = float(abs(state.pen_endpoints[0, 2] - state.pen_endpoints[1, 2]))
        center_delta = state.pen_center - self.initial_center
        xy_drift = float(np.linalg.norm(center_delta[:2]))
        z_delta = float(center_delta[2])
        axis_cos = float(
            np.clip(
                np.dot(state.pen_axis, self.initial_axis)
                / (np.linalg.norm(state.pen_axis) * np.linalg.norm(self.initial_axis) + 1e-9),
                -1.0,
                1.0,
            )
        )
        axis_angle = float(np.arccos(axis_cos))
        phase_abs = float(abs(state.rotation_phase - self.initial_phase))
        contact_flags = {finger: float(finger in fingers) for finger in ["thumb", "index", "middle", "ring", "pinky"]}

        info = {
            **base_info,
            "support_fingers": sorted(fingers),
            "support_count": int(support_count),
            "tripod": int(tripod),
            "contact_flags": contact_flags,
            "contact_geoms": geoms.most_common(10),
            "xy_drift": xy_drift,
            "z_delta": z_delta,
            "axis_angle": axis_angle,
            "endpoint_height_diff": endpoint_dz,
            "endpoint_height_diff_delta": float(endpoint_dz - self.initial_endpoint_dz),
            "phase_abs": phase_abs,
            "max_penetration": float(max_penetration),
            "action_norm": float(np.linalg.norm(action)),
            "action_max_abs": float(np.max(np.abs(action))) if action.size else 0.0,
            "ncon": int(self.env.data.ncon),
        }
        return info

    def _hold_reward(self, info: dict[str, Any]) -> float:
        support_count = float(info["support_count"])
        tripod = float(info["tripod"])
        thumb = float(info["contact_flags"].get("thumb", 0.0))
        xy_drift = float(info["xy_drift"])
        z_delta = float(info["z_delta"])
        axis_angle = float(info["axis_angle"])
        endpoint_delta = abs(float(info["endpoint_height_diff_delta"]))
        phase_abs = float(info["phase_abs"])
        action_norm = float(info["action_norm"])
        penetration = float(info["max_penetration"])

        reward = 0.40
        reward += 1.20 * tripod
        reward += 0.35 * (support_count / 3.0)
        reward += 0.10 * thumb
        reward -= 70.0 * xy_drift
        reward -= 45.0 * abs(z_delta)
        reward -= 3.0 * axis_angle
        reward -= 10.0 * endpoint_delta
        reward -= 0.20 * phase_abs
        reward -= 0.12 * action_norm
        reward -= 120.0 * max(0.0, penetration - 0.006)
        return float(reward)

    def _failure(self, info: dict[str, Any]) -> bool:
        if info["xy_drift"] > 0.035:
            return True
        if info["z_delta"] < -0.030:
            return True
        if info["endpoint_height_diff"] > 0.030:
            return True
        if info["support_count"] < 2 and self.env.step_count > int(0.5 * self.control_hz):
            return True
        return False


def summarize_rollout(infos: list[dict[str, Any]], returns: float, steps: int, control_hz: float) -> dict[str, Any]:
    if not infos:
        return {"return": float(returns), "steps": int(steps), "survival_seconds": 0.0}
    tripod_frames = sum(int(info.get("tripod", 0)) for info in infos)
    finger_frames: Counter[str] = Counter()
    geom_counter: Counter[str] = Counter()
    for info in infos:
        for finger, flag in info.get("contact_flags", {}).items():
            if flag:
                finger_frames[finger] += 1
        geom_counter.update(dict(info.get("contact_geoms", [])))
    xy = np.asarray([info["xy_drift"] for info in infos], dtype=np.float64)
    z = np.asarray([info["z_delta"] for info in infos], dtype=np.float64)
    axis = np.asarray([info["axis_angle"] for info in infos], dtype=np.float64)
    endpoint = np.asarray([info["endpoint_height_diff"] for info in infos], dtype=np.float64)
    phase = np.asarray([info["phase_abs"] for info in infos], dtype=np.float64)
    action = np.asarray([info["action_norm"] for info in infos], dtype=np.float64)
    steps = max(1, int(steps))
    return {
        "return": float(returns),
        "steps": int(steps),
        "survival_seconds": float(steps / control_hz),
        "tripod_frames": int(tripod_frames),
        "tripod_ratio": float(tripod_frames / steps),
        "thumb_ratio": float(finger_frames["thumb"] / steps),
        "index_ratio": float(finger_frames["index"] / steps),
        "middle_ratio": float(finger_frames["middle"] / steps),
        "ring_ratio": float(finger_frames["ring"] / steps),
        "xy_drift_final": float(xy[-1]),
        "xy_drift_max": float(np.max(xy)),
        "z_delta_final": float(z[-1]),
        "z_delta_min": float(np.min(z)),
        "axis_angle_final": float(axis[-1]),
        "axis_angle_max": float(np.max(axis)),
        "endpoint_height_diff_final": float(endpoint[-1]),
        "endpoint_height_diff_max": float(np.max(endpoint)),
        "phase_abs_final": float(phase[-1]),
        "phase_abs_max": float(np.max(phase)),
        "action_norm_mean": float(np.mean(action)),
        "top_geom_contacts": geom_counter.most_common(10),
    }
