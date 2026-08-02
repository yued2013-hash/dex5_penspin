from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from dex5_penspin.isaac.m3_2_30_first_leg_reward import compute_index_near_surface_reward


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_BUNDLE_PATH = str(ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface/isaac_first_leg_bundle.npz")


@configclass
class Dex5IndexNearSurfaceCacheEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 0.18
    action_space = 20
    observation_space = 76
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=0.45, replicate_physics=True, clone_in_fabric=True
    )

    bundle_path = DEFAULT_BUNDLE_PATH
    action_scale = 0.055
    target_surface_dist = 0.006
    initial_surface_dist = 0.030
    fail_surface_dist = 0.060
    object_drift_per_action = 0.002


class Dex5IndexNearSurfaceCacheEnv(DirectRLEnv):
    cfg: Dex5IndexNearSurfaceCacheEnvCfg

    def __init__(self, cfg: Dex5IndexNearSurfaceCacheEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._load_bundle()
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.applied_delta = torch.zeros_like(self.actions)
        self.base_obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)
        self.current_obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)
        self.target_delta = torch.zeros_like(self.actions)
        self.target_delta_norm = torch.ones(self.num_envs, device=self.device)
        self.index_tip_dist = torch.full((self.num_envs,), self.cfg.initial_surface_dist, device=self.device)
        self.initial_index_tip_dist = self.index_tip_dist.clone()
        self.init_obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_pos = torch.zeros_like(self.init_obj_pos)
        self.target_obj_pos = torch.zeros_like(self.init_obj_pos)
        self.max_penetration = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(4.0, 4.0)))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _load_bundle(self) -> None:
        data = np.load(self.cfg.bundle_path)
        obs_rows = []
        target_delta_rows = []
        target_obj_rows = []
        for window_index, window in enumerate(("016_019", "019_022")):
            source_qpos = np.asarray(data[f"{window}_source_qpos"], dtype=np.float32)
            source_qvel = np.asarray(data[f"{window}_source_qvel"], dtype=np.float32)
            source_target_qpos = np.asarray(data[f"{window}_source_target_qpos"], dtype=np.float32)
            target_hand_qpos = np.asarray(data[f"{window}_target_hand_qpos"], dtype=np.float32)
            target_pen_pos = np.asarray(data[f"{window}_target_pen_pos"], dtype=np.float32)
            source_t = np.asarray(data[f"{window}_source_t"], dtype=np.float32)
            window_onehot = np.zeros((source_qpos.shape[0], 2), dtype=np.float32)
            window_onehot[:, window_index] = 1.0
            tau = ((source_t - source_t.min()) / max(float(source_t.max() - source_t.min()), 1e-6))[:, None]

            obs = np.concatenate((source_qpos, source_qvel, source_target_qpos, tau, window_onehot), axis=-1)
            obs_rows.append(obs)
            target_delta_rows.append(target_hand_qpos[None, :] - source_qpos[:, : target_hand_qpos.shape[0]])
            target_obj_rows.append(np.repeat(target_pen_pos[None, :], source_qpos.shape[0], axis=0))

        self._obs_bank = torch.as_tensor(np.concatenate(obs_rows, axis=0), device=self.device)
        self._target_delta_bank = torch.as_tensor(np.concatenate(target_delta_rows, axis=0), device=self.device)
        self._target_obj_bank = torch.as_tensor(np.concatenate(target_obj_rows, axis=0), device=self.device)
        self._bank_size = self._obs_bank.shape[0]
        self._tau_index = 27 + 26 + 20

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        self.applied_delta += self.actions * (self.cfg.action_scale / float(self.cfg.decimation))
        self.applied_delta = torch.clamp(self.applied_delta, -1.25, 1.25)
        self._update_cache_state()

    def _get_observations(self) -> dict:
        tau = self.episode_length_buf.float() / max(float(self.max_episode_length), 1.0)
        self.current_obs[:, self._tau_index] = torch.clamp(tau, 0.0, 1.0)
        return {"policy": self.current_obs}

    def _get_rewards(self) -> torch.Tensor:
        reward, terms = compute_index_near_surface_reward(
            index_tip_dist=self.index_tip_dist,
            initial_index_tip_dist=self.initial_index_tip_dist,
            obj_pos=self.obj_pos,
            init_obj_pos=self.init_obj_pos,
            target_obj_pos=self.target_obj_pos,
            max_penetration=self.max_penetration,
            actions=self.actions,
            prev_actions=self.prev_actions,
            progress_buf=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            failed=self.reset_terminated,
        )
        self.extras["log"] = {f"reward/{name}": value.mean() for name, value in terms.items()}
        self.extras["log"]["metric/index_tip_dist_mm"] = self.index_tip_dist.mean() * 1000.0
        self.extras["log"]["metric/near_14mm_ratio"] = (self.index_tip_dist <= 0.014).float().mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        failed = self.index_tip_dist > self.cfg.fail_surface_dist
        return failed, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        bank_ids = torch.randint(0, self._bank_size, (len(env_ids),), device=self.device)
        self.base_obs[env_ids] = self._obs_bank[bank_ids]
        self.current_obs[env_ids] = self.base_obs[env_ids]
        self.target_delta[env_ids] = self._target_delta_bank[bank_ids]
        self.applied_delta[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.target_delta_norm[env_ids] = torch.linalg.norm(self.target_delta[env_ids], dim=-1).clamp_min(1e-4)

        self.init_obj_pos[env_ids] = self.current_obs[env_ids, 20:23]
        self.obj_pos[env_ids] = self.init_obj_pos[env_ids]
        self.target_obj_pos[env_ids] = self._target_obj_bank[bank_ids]
        self.initial_index_tip_dist[env_ids] = self.cfg.initial_surface_dist
        self.index_tip_dist[env_ids] = self.cfg.initial_surface_dist
        self.max_penetration[env_ids] = 0.0

    def _update_cache_state(self) -> None:
        residual = torch.linalg.norm(self.target_delta - self.applied_delta, dim=-1) / self.target_delta_norm
        residual = torch.clamp(residual, 0.0, 1.0)
        self.index_tip_dist = self.cfg.target_surface_dist + (
            self.cfg.initial_surface_dist - self.cfg.target_surface_dist
        ) * residual
        self.max_penetration = torch.clamp(self.cfg.target_surface_dist - self.index_tip_dist, min=0.0)

        self.current_obs[:, :20] = self.base_obs[:, :20] + self.applied_delta
        self.current_obs[:, 27 : 27 + 20] = self.actions
        drift = self.cfg.object_drift_per_action * torch.tanh(torch.linalg.norm(self.applied_delta, dim=-1, keepdim=True))
        self.obj_pos = self.init_obj_pos + torch.cat((drift, torch.zeros_like(drift), -0.5 * drift), dim=-1)
