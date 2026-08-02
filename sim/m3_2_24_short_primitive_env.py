from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from evaluate_m3_2_15_short_primitives import contact_metrics, joint_limit_margin, quat_to_axis
from m3_2_23_short_primitive_reward import (
    FINGERS,
    PrimitiveSpec,
    ShortPrimitiveWeights,
    smoothstep,
    step_reward_terms,
)
from m3_oracle_hold_env import Dex5PenSpinOracleHoldEnv, load_preset
from rebuild_m3_2_16_p4_p5_transitions import tip_surface_distances


DEFAULT_REWARD_PLAN = Path(
    "dex5_penspin/reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/primitive_reward_spec.json"
)
DEFAULT_CACHE_MANIFEST = Path(
    "dex5_penspin/reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/dynamic_state_cache/manifest.json"
)

CACHE_WINDOW_BY_PRIMITIVE = {
    "earlier_pre_ring_index_thumb_recapture": "earlier_pre_ring_for_B12b_to_B23",
    "earlier_pre_ring_016_019_index_thumb_recapture": "earlier_pre_ring_016_019_for_B12b_to_B23",
    "earlier_pre_ring_019_022_index_thumb_recapture": "earlier_pre_ring_019_022_for_B12b_to_B23",
    "earlier_pre_ring_016_019_to_index_near_surface": "earlier_pre_ring_016_019_for_B12b_to_B23",
    "earlier_pre_ring_019_022_to_index_near_surface": "earlier_pre_ring_019_022_for_B12b_to_B23",
    "earlier_pre_ring_016_019_to_projected_recapture": "earlier_pre_ring_016_019_for_B12b_to_B23",
    "earlier_pre_ring_019_022_to_projected_recapture": "earlier_pre_ring_019_022_for_B12b_to_B23",
    "index_near_surface_016_019_to_projected_recapture": (
        "dex5_penspin/reports/m3_2_30_index_near_surface/"
        "near_016_019/index_near_surface_cache.npz"
    ),
    "index_near_surface_019_022_to_projected_recapture": (
        "dex5_penspin/reports/m3_2_30_index_near_surface/"
        "near_019_022/index_near_surface_cache.npz"
    ),
    "index_near_surface_016_019_to_index_contact": (
        "dex5_penspin/reports/m3_2_30_index_near_surface/"
        "near_016_019/index_near_surface_cache.npz"
    ),
    "index_near_surface_019_022_to_index_contact": (
        "dex5_penspin/reports/m3_2_30_index_near_surface/"
        "near_019_022/index_near_surface_cache.npz"
    ),
    "early_pre_ring_index_thumb_recapture": "early_pre_ring_for_B12b_to_B23",
    "early_pre_ring_to_B23_capture": "early_pre_ring_for_B12b_to_B23",
    "pre_ring_collapse_to_B23_capture": "pre_ring_collapse_for_B12b_to_B23",
    "projected_index_thumb_recapture_to_B23_entry": (
        "dex5_penspin/reports/m3_2_27_earlier_pre_ring_chain/"
        "earlier_projected_index_thumb_recapture/index_thumb_recapture_projected_cache.npz"
    ),
    "projected_index_thumb_recapture_to_B23": (
        "dex5_penspin/reports/m3_2_27_earlier_pre_ring_chain/"
        "earlier_projected_index_thumb_recapture/index_thumb_recapture_projected_cache.npz"
    ),
    "B23_entry_to_B23_capture": (
        "dex5_penspin/reports/m3_2_28_recapture_split_b23_entry/"
        "projected_to_b23_entry_ring_touch/b23_entry_ring_touch_cache.npz"
    ),
    "B12cLift_to_B23_capture": "b12c_lift_arrival_for_B12cLift_to_B23",
    "B23_to_P3_capture": "b23_arrival_for_B23_to_P3",
}

PROJECTED_RECAPTURE_TARGET_SEED_BY_PRIMITIVE = {
    "earlier_pre_ring_016_019_to_projected_recapture": (
        "dex5_penspin/reports/m3_2_28_recapture_split_b23_entry/"
        "recapture_016_019_projected/index_thumb_recapture_projected_seed.json"
    ),
    "earlier_pre_ring_019_022_to_projected_recapture": (
        "dex5_penspin/reports/m3_2_28_recapture_split_b23_entry/"
        "recapture_019_022_projected/index_thumb_recapture_projected_seed.json"
    ),
}

INDEX_NEAR_SURFACE_TARGET_SEED_BY_PRIMITIVE = {
    "earlier_pre_ring_016_019_to_index_near_surface": (
        "dex5_penspin/reports/m3_2_30_index_near_surface/"
        "near_016_019/index_near_surface_seed.json"
    ),
    "earlier_pre_ring_019_022_to_index_near_surface": (
        "dex5_penspin/reports/m3_2_30_index_near_surface/"
        "near_019_022/index_near_surface_seed.json"
    ),
}

INDEX_NEAR_SURFACE_LABELS = set(INDEX_NEAR_SURFACE_TARGET_SEED_BY_PRIMITIVE)
INDEX_NEAR_SURFACE_THRESHOLD_M = 0.014

NEAR_SURFACE_TO_PROJECTED_TARGET_SEED_BY_PRIMITIVE = {
    "index_near_surface_016_019_to_projected_recapture": PROJECTED_RECAPTURE_TARGET_SEED_BY_PRIMITIVE[
        "earlier_pre_ring_016_019_to_projected_recapture"
    ],
    "index_near_surface_019_022_to_projected_recapture": PROJECTED_RECAPTURE_TARGET_SEED_BY_PRIMITIVE[
        "earlier_pre_ring_019_022_to_projected_recapture"
    ],
}

INDEX_CONTACT_TARGET_SEED_BY_PRIMITIVE = {
    "index_near_surface_016_019_to_index_contact": (
        "dex5_penspin/reports/m3_2_30_index_near_surface/"
        "contact_016_019/index_contact_seed.json"
    ),
    "index_near_surface_019_022_to_index_contact": (
        "dex5_penspin/reports/m3_2_30_index_near_surface/"
        "contact_019_022/index_contact_seed.json"
    ),
}

INDEX_RECAPTURE_LABELS = {
    "earlier_pre_ring_index_thumb_recapture",
    "earlier_pre_ring_016_019_index_thumb_recapture",
    "earlier_pre_ring_019_022_index_thumb_recapture",
    "earlier_pre_ring_016_019_to_index_near_surface",
    "earlier_pre_ring_019_022_to_index_near_surface",
    "earlier_pre_ring_016_019_to_projected_recapture",
    "earlier_pre_ring_019_022_to_projected_recapture",
    "index_near_surface_016_019_to_projected_recapture",
    "index_near_surface_019_022_to_projected_recapture",
    "index_near_surface_016_019_to_index_contact",
    "index_near_surface_019_022_to_index_contact",
    "early_pre_ring_index_thumb_recapture",
}


def _mask(names: list[str] | tuple[str, ...]) -> np.ndarray:
    selected = set(names)
    return np.asarray([float(finger in selected) for finger in FINGERS], dtype=np.float32)


def _load_reward_plan(path: str | Path) -> tuple[ShortPrimitiveWeights, list[PrimitiveSpec]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = ShortPrimitiveWeights(**data.get("weights", {}))
    specs = [PrimitiveSpec(**raw) for raw in data.get("primitives", [])]
    return weights, specs


def load_short_primitive_spec(path: str | Path, label: str) -> tuple[ShortPrimitiveWeights, PrimitiveSpec]:
    weights, specs = _load_reward_plan(path)
    by_label = {spec.label: spec for spec in specs}
    if label in {
        "earlier_pre_ring_index_thumb_recapture",
        "earlier_pre_ring_016_019_index_thumb_recapture",
        "earlier_pre_ring_019_022_index_thumb_recapture",
        "early_pre_ring_index_thumb_recapture",
    }:
        base = by_label["B12cLift_to_B23_capture"]
        window = CACHE_WINDOW_BY_PRIMITIVE[label]
        return weights, PrimitiveSpec(
            label=label,
            start_seed=base.start_seed,
            target_seed=base.start_seed,
            duration_s=0.28,
            target_contacts=["thumb", "index", "middle"],
            support_contacts=list(base.support_contacts),
            allowed_extra_contacts=["ring"],
            forbidden_contacts=["pinky"],
            reset_mode="dynamic_state_cache",
            source_summary_csv=base.source_summary_csv,
            source_summary_label=window,
            phase_direction=base.phase_direction,
            target_phase_delta=base.target_phase_delta,
            priority="critical",
            notes=(
                "Focused recapture curriculum from early pre-ring states. "
                "Terminal target is thumb/index/middle; ring is only an allowed guard, not a target."
            ),
        )
    if label in INDEX_NEAR_SURFACE_TARGET_SEED_BY_PRIMITIVE:
        base = by_label["B12cLift_to_B23_capture"]
        window = CACHE_WINDOW_BY_PRIMITIVE[label]
        return weights, PrimitiveSpec(
            label=label,
            start_seed=base.start_seed,
            target_seed=INDEX_NEAR_SURFACE_TARGET_SEED_BY_PRIMITIVE[label],
            duration_s=0.17,
            target_contacts=[],
            support_contacts=["thumb", "index", "middle", "ring", "pinky"],
            allowed_extra_contacts=["thumb", "index", "middle", "ring", "pinky"],
            forbidden_contacts=[],
            reset_mode="dynamic_state_cache",
            source_summary_csv=base.source_summary_csv,
            source_summary_label=window,
            phase_direction=base.phase_direction,
            target_phase_delta=base.target_phase_delta,
            priority="critical",
            notes=(
                "Approach-only first-leg curriculum. Success is index fingertip near-surface "
                "distance under 14mm; index contact is intentionally deferred to the next leg."
            ),
        )
    if label in PROJECTED_RECAPTURE_TARGET_SEED_BY_PRIMITIVE:
        base = by_label["B12cLift_to_B23_capture"]
        window = CACHE_WINDOW_BY_PRIMITIVE[label]
        return weights, PrimitiveSpec(
            label=label,
            start_seed=base.start_seed,
            target_seed=PROJECTED_RECAPTURE_TARGET_SEED_BY_PRIMITIVE[label],
            duration_s=0.22,
            target_contacts=["thumb", "index", "middle"],
            support_contacts=list(base.support_contacts),
            allowed_extra_contacts=["ring"],
            forbidden_contacts=["pinky"],
            reset_mode="dynamic_state_cache",
            source_summary_csv=base.source_summary_csv,
            source_summary_label=window,
            phase_direction=base.phase_direction,
            target_phase_delta=base.target_phase_delta,
            priority="critical",
            notes=(
                "Split curriculum leg from earlier pre-ring dynamic states to the projected "
                "thumb/index/middle recapture seed."
            ),
        )
    if label in NEAR_SURFACE_TO_PROJECTED_TARGET_SEED_BY_PRIMITIVE:
        base = by_label["B12cLift_to_B23_capture"]
        window = CACHE_WINDOW_BY_PRIMITIVE[label]
        return weights, PrimitiveSpec(
            label=label,
            start_seed=base.start_seed,
            target_seed=NEAR_SURFACE_TO_PROJECTED_TARGET_SEED_BY_PRIMITIVE[label],
            duration_s=0.18,
            target_contacts=["thumb", "index", "middle"],
            support_contacts=list(base.support_contacts),
            allowed_extra_contacts=["ring"],
            forbidden_contacts=["pinky"],
            reset_mode="dynamic_state_cache",
            source_summary_csv=base.source_summary_csv,
            source_summary_label=str(window),
            phase_direction=base.phase_direction,
            target_phase_delta=base.target_phase_delta,
            priority="critical",
            notes=(
                "Contact leg from index-near-surface states to projected thumb/index/middle recapture. "
                "This is the first leg's actual contact primitive after approach-only succeeds."
            ),
        )
    if label in INDEX_CONTACT_TARGET_SEED_BY_PRIMITIVE:
        base = by_label["B12cLift_to_B23_capture"]
        window = CACHE_WINDOW_BY_PRIMITIVE[label]
        return weights, PrimitiveSpec(
            label=label,
            start_seed=base.start_seed,
            target_seed=INDEX_CONTACT_TARGET_SEED_BY_PRIMITIVE[label],
            duration_s=0.16,
            target_contacts=["index"],
            support_contacts=["thumb", "index", "middle", "ring", "pinky"],
            allowed_extra_contacts=["thumb", "middle", "ring", "pinky"],
            forbidden_contacts=[],
            reset_mode="dynamic_state_cache",
            source_summary_csv=base.source_summary_csv,
            source_summary_label=str(window),
            phase_direction=base.phase_direction,
            target_phase_delta=base.target_phase_delta,
            priority="critical",
            notes=(
                "Small contact leg from index-near-surface states to first explicit index contact. "
                "It does not ask for the full projected recapture pose yet."
            ),
        )
    if label == "projected_index_thumb_recapture_to_B23_entry":
        base = by_label["B12cLift_to_B23_capture"]
        window = CACHE_WINDOW_BY_PRIMITIVE[label]
        return weights, PrimitiveSpec(
            label=label,
            start_seed=base.start_seed,
            target_seed=(
                "dex5_penspin/reports/m3_2_28_recapture_split_b23_entry/"
                "projected_to_b23_entry_ring_touch/B23_entry_ring_touch_seed.json"
            ),
            duration_s=0.24,
            target_contacts=["thumb", "index", "ring"],
            support_contacts=list(base.support_contacts),
            allowed_extra_contacts=["middle"],
            forbidden_contacts=["pinky"],
            reset_mode="dynamic_state_cache",
            source_summary_csv=base.source_summary_csv,
            source_summary_label=str(window),
            phase_direction=base.phase_direction,
            target_phase_delta=base.target_phase_delta,
            priority="critical",
            notes=(
                "Intermediate ring-touch entry from projected thumb/index recapture. "
                "It asks for ring contact without forcing the full B23 reference pose yet."
            ),
        )
    if label == "projected_index_thumb_recapture_to_B23":
        base = by_label["B12cLift_to_B23_capture"]
        window = CACHE_WINDOW_BY_PRIMITIVE[label]
        return weights, PrimitiveSpec(
            label=label,
            start_seed=base.start_seed,
            target_seed=base.target_seed,
            duration_s=0.32,
            target_contacts=["thumb", "index", "ring"],
            support_contacts=list(base.support_contacts),
            allowed_extra_contacts=["middle"],
            forbidden_contacts=["pinky"],
            reset_mode="dynamic_state_cache",
            source_summary_csv=base.source_summary_csv,
            source_summary_label=str(window),
            phase_direction=base.phase_direction,
            target_phase_delta=base.target_phase_delta,
            priority="critical",
            notes=(
                "Second curriculum leg from projected thumb/index/middle recapture states toward B23 "
                "thumb/index/ring capture."
            ),
        )
    if label == "B23_entry_to_B23_capture":
        base = by_label["B12cLift_to_B23_capture"]
        window = CACHE_WINDOW_BY_PRIMITIVE[label]
        return weights, PrimitiveSpec(
            label=label,
            start_seed=(
                "dex5_penspin/reports/m3_2_28_recapture_split_b23_entry/"
                "projected_to_b23_entry_ring_touch/B23_entry_ring_touch_seed.json"
            ),
            target_seed=base.target_seed,
            duration_s=0.28,
            target_contacts=list(base.target_contacts),
            support_contacts=list(base.support_contacts),
            allowed_extra_contacts=list(base.allowed_extra_contacts),
            forbidden_contacts=list(base.forbidden_contacts),
            reset_mode="dynamic_state_cache",
            source_summary_csv=base.source_summary_csv,
            source_summary_label=str(window),
            phase_direction=base.phase_direction,
            target_phase_delta=base.target_phase_delta,
            priority="critical",
            notes="Final curriculum leg from B23-entry/ring-touch states to full B23 capture.",
        )
    if label in {"early_pre_ring_to_B23_capture", "pre_ring_collapse_to_B23_capture"}:
        base = by_label["B12cLift_to_B23_capture"]
        window = CACHE_WINDOW_BY_PRIMITIVE[label]
        return weights, PrimitiveSpec(
            label=label,
            start_seed=base.start_seed,
            target_seed=base.target_seed,
            duration_s=0.36,
            target_contacts=list(base.target_contacts),
            support_contacts=list(base.support_contacts),
            allowed_extra_contacts=list(base.allowed_extra_contacts),
            forbidden_contacts=list(base.forbidden_contacts),
            reset_mode="dynamic_state_cache",
            source_summary_csv=base.source_summary_csv,
            source_summary_label=window,
            phase_direction=base.phase_direction,
            target_phase_delta=base.target_phase_delta,
            priority="critical",
            notes=(
                f"Curriculum reset from the {window} dynamic cache. "
                "The target is still B23 thumb/index/ring capture; ring should guard, not carry."
            ),
        )
    if label not in by_label:
        known = ", ".join(sorted(by_label))
        raise KeyError(f"Unknown primitive label {label!r}; known labels: {known}")
    return weights, by_label[label]


def load_cache_path(manifest_path: str | Path, window: str) -> Path:
    direct_path = Path(window)
    if direct_path.exists() or direct_path.suffix == ".npz":
        return direct_path
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    caches = data.get("caches", {})
    if window not in caches:
        known = ", ".join(sorted(caches))
        raise KeyError(f"Unknown dynamic cache window {window!r}; known windows: {known}")
    return Path(caches[window]["path"])


class Dex5PenSpinShortPrimitiveEnv(Dex5PenSpinOracleHoldEnv):
    """MuJoCo/Gym wrapper for one keypose-guided short primitive.

    This wrapper keeps the existing Dex5 delta-target action contract but
    replaces the hold reward with the M3.2.23 short-primitive reward. It can
    reset either from a static JSON keypose or from a dynamic MuJoCo state cache.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        primitive_label: str,
        reward_plan_path: str | Path = DEFAULT_REWARD_PLAN,
        cache_manifest_path: str | Path = DEFAULT_CACHE_MANIFEST,
        cache_window: str | None = None,
        model_path: str | Path | None = None,
        episode_seconds: float | None = None,
        reward_scale: float = 1.0,
        assist_mode: str = "none",
        assist_delta_limit: float = 0.006,
        actuator_kp: float | None = None,
        actuator_kv: float | None = None,
        action_delta_limit: float | None = None,
        tip_proxy_radius: float | None = None,
        tip_proxy_friction: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.weights, self.spec = load_short_primitive_spec(reward_plan_path, primitive_label)
        self.start_seed_path = Path(self.spec.start_seed)
        self.target_seed_path = Path(self.spec.target_seed)
        self.start_seed = load_preset(self.start_seed_path)
        self.target_seed = load_preset(self.target_seed_path)
        self.reward_scale = float(reward_scale)
        self.assist_mode = str(assist_mode)
        self.assist_delta_limit = float(assist_delta_limit)
        self.cache_window = cache_window or CACHE_WINDOW_BY_PRIMITIVE.get(self.spec.label, "")
        self.cache: dict[str, np.ndarray] | None = None
        self.active_cache_index = -1
        self.active_cache_t = 0.0
        if self.spec.reset_mode == "dynamic_state_cache":
            if not self.cache_window:
                raise ValueError(f"Primitive {self.spec.label} needs a dynamic cache window")
            cache_path = load_cache_path(cache_manifest_path, self.cache_window)
            self.cache = dict(np.load(cache_path, allow_pickle=True))
        self.goal_center = np.asarray(self.target_seed["pen_pos"], dtype=np.float64)
        self.goal_axis = quat_to_axis(np.asarray(self.target_seed["pen_quat"], dtype=np.float64))
        self.target_hand_qpos = np.asarray(self.target_seed["hand_qpos"], dtype=np.float64)
        self.target_contact_mask = _mask(self.spec.target_contacts)
        self.allowed_extra_mask = _mask(self.spec.allowed_extra_contacts)
        self.forbidden_contact_mask = _mask(self.spec.forbidden_contacts)
        self.prev_action_norm = np.zeros(20, dtype=np.float32)
        self.scripted_start_qpos = np.asarray(self.start_seed["hand_qpos"], dtype=np.float64)
        self.initial_tip_distances = {finger: 0.0 for finger in FINGERS}

        model = model_path or self.start_seed.get("model_path")
        super().__init__(
            preset_path=self.start_seed_path,
            model_path=model,
            episode_seconds=float(episode_seconds or self.spec.duration_s),
            seed=seed,
            **kwargs,
        )
        if actuator_kp is not None or actuator_kv is not None:
            self.env._set_position_actuator_gains(actuator_kp=actuator_kp, actuator_kv=actuator_kv)
        self.action_delta_limit = (
            float(action_delta_limit)
            if action_delta_limit is not None
            else float(self.start_seed.get("action_delta_limit", self.env.action_delta_limit))
        )
        self.env.action_delta_limit = self.action_delta_limit
        self.tip_proxy_radius = (
            float(tip_proxy_radius)
            if tip_proxy_radius is not None
            else self.start_seed.get("tip_proxy_radius")
        )
        self.tip_proxy_friction = (
            float(tip_proxy_friction)
            if tip_proxy_friction is not None
            else self.start_seed.get("tip_proxy_friction")
        )
        if self.tip_proxy_radius is not None or self.tip_proxy_friction is not None:
            self._configure_tip_proxies()
        self.base_obs_dim = 97
        self.primitive_obs_extra_dim = 28
        self.obs_dim = self.base_obs_dim + self.primitive_obs_extra_dim
        self.max_episode_steps = self.env.max_episode_steps
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def _configure_tip_proxies(self) -> None:
        for geom_id in range(self.env.model.ngeom):
            geom_name = (
                mujoco.mj_id2name(
                    self.env.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                )
                or ""
            )
            if "tip_proxy" not in geom_name:
                continue
            if self.tip_proxy_radius is not None:
                self.env.model.geom_size[geom_id, 0] = float(self.tip_proxy_radius)
            if self.tip_proxy_friction is not None:
                self.env.model.geom_friction[geom_id] = np.asarray(
                    [self.tip_proxy_friction, 0.08, 0.01],
                    dtype=np.float64,
                )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.spec.reset_mode == "dynamic_state_cache":
            return self._reset_dynamic_cache(seed=seed, options=options)

        obs, info = super().reset(seed=seed, options=options)
        self.prev_action_norm[:] = 0.0
        self.active_cache_index = -1
        self.active_cache_t = 0.0
        self.scripted_start_qpos = self.env.target_qpos.astype(np.float64).copy()
        self.initial_tip_distances = tip_surface_distances(self.env)
        info = self._decorate_primitive_info(info)
        obs = self._get_obs(info)
        self._last_info = info
        return obs, info

    def _reset_dynamic_cache(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.cache is None:
            raise RuntimeError("Dynamic reset requested without a loaded cache")
        options = options or {}
        n_states = int(self.cache["qpos"].shape[0])
        if "cache_index" in options:
            idx = int(options["cache_index"]) % n_states
        else:
            idx = int(self.np_random.integers(0, n_states))

        self.env.reset(seed=seed)
        self.env.data.qpos[:] = np.asarray(self.cache["qpos"][idx], dtype=np.float64)
        self.env.data.qvel[:] = np.asarray(self.cache["qvel"][idx], dtype=np.float64)
        self.env.target_qpos = np.clip(
            np.asarray(self.cache["target_qpos"][idx], dtype=np.float64),
            self.env.joint_low,
            self.env.joint_high,
        ).astype(np.float32)
        self.env.data.ctrl[:] = self.env.target_qpos
        self.env.last_action[:] = 0.0
        self.env.step_count = 0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.prev_phase = float(self.env._quat_twist_phase_about_x(self.env.data.xquat[self.env.pen_body_id]))
        self.env.rotation_phase = float(self.cache["rotation_phase"][idx]) if "rotation_phase" in self.cache else 0.0

        state = self.env.get_state()
        self.initial_center = state.pen_center.copy()
        self.initial_axis = state.pen_axis.copy()
        self.initial_endpoint_dz = float(abs(state.pen_endpoints[0, 2] - state.pen_endpoints[1, 2]))
        self.initial_phase = float(state.rotation_phase)
        self.prev_action_norm[:] = 0.0
        self.active_cache_index = idx
        if "t" in self.cache:
            self.active_cache_t = float(self.cache["t"][idx])
        elif "cache_t" in self.cache:
            self.active_cache_t = float(self.cache["cache_t"][idx])
        else:
            self.active_cache_t = 0.0
        self.scripted_start_qpos = self.env.target_qpos.astype(np.float64).copy()
        self.initial_tip_distances = tip_surface_distances(self.env)

        info = self._make_info(action=np.zeros(20, dtype=np.float32), base_info=self.env._info())
        obs = self._get_obs(info)
        self._last_info = info
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_norm = np.asarray(action, dtype=np.float32)
        action_norm = np.clip(action_norm, -1.0, 1.0)
        prev_action = self.prev_action_norm.copy()
        assist_delta = self._assist_delta()
        scaled_action = np.clip(
            action_norm * self.action_scale + assist_delta,
            -self.env.action_delta_limit,
            self.env.action_delta_limit,
        )
        _, _, base_terminated, base_truncated, base_info = self.env.step(scaled_action)

        info = self._make_info(action=action_norm, base_info=base_info)
        info["assist_mode"] = self.assist_mode
        info["assist_delta_norm"] = float(np.linalg.norm(assist_delta))
        info["assist_delta_max_abs"] = float(np.max(np.abs(assist_delta))) if assist_delta.size else 0.0
        failure = self._failure(info)
        terminated = bool(base_terminated or failure)
        truncated = bool(base_truncated)
        reward, terms = self._primitive_reward(info, action_norm, prev_action, failed=terminated)
        obs = self._get_obs(info)

        info["primitive_reward_terms"] = terms
        info["oracle_reward"] = float(reward)
        info["failed"] = bool(failure)
        self.prev_action_norm = action_norm
        self._last_info = info
        return obs, float(reward), terminated, truncated, info

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
            [float(info["contact_flags"].get(finger, 0.0)) for finger in FINGERS],
            dtype=np.float64,
        )
        tip_distances = info.get("tip_surface_distances")
        if tip_distances is None:
            tip_distances = tip_surface_distances(self.env)
        tip_distance_features = np.array(
            [np.clip(float(tip_distances.get(finger, 0.04)) / 0.035, -1.0, 2.0) for finger in FINGERS],
            dtype=np.float64,
        )
        last_action_norm = np.clip(state.last_action / max(self.action_scale, 1e-9), -5.0, 5.0)
        primitive_extra = np.concatenate(
            [
                (self.goal_center - state.pen_center) / 0.04,
                self.goal_axis,
                self.target_contact_mask,
                self.allowed_extra_mask,
                self.forbidden_contact_mask,
                tip_distance_features,
                np.array([self._tau(), float(np.sign(self.spec.phase_direction))], dtype=np.float32),
            ]
        )
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
                primitive_extra,
            ]
        ).astype(np.float32)
        if obs.shape != (self.obs_dim,):
            raise RuntimeError(f"Unexpected obs shape {obs.shape}, expected {(self.obs_dim,)}")
        return obs

    def _make_info(self, *, action: np.ndarray, base_info: dict[str, Any]) -> dict[str, Any]:
        info = super()._make_info(action=action, base_info=base_info)
        return self._decorate_primitive_info(info)

    def _decorate_primitive_info(self, info: dict[str, Any]) -> dict[str, Any]:
        metrics = contact_metrics(self.env)
        tip_distances = tip_surface_distances(self.env)
        fingers = set(metrics["fingers"])
        state = self.env.get_state()
        contact_flags = {finger: float(finger in fingers) for finger in FINGERS}
        desired = set(self.spec.target_contacts)
        support = set(self.spec.support_contacts)
        desired_ratio = 0.0 if not desired else len(desired.intersection(fingers)) / len(desired)
        support_ratio = min(1.0, len(support.intersection(fingers)) / 2.0)
        terminal_hit = float(bool(desired) and desired.issubset(fingers))
        index_tip_dist = float(tip_distances.get("index", 0.04))
        index_near_surface_hit = float(index_tip_dist <= INDEX_NEAR_SURFACE_THRESHOLD_M)
        if self.spec.label in INDEX_NEAR_SURFACE_LABELS:
            terminal_hit = index_near_surface_hit
        axis_cos = float(
            np.clip(
                np.dot(state.pen_axis, self.goal_axis)
                / (np.linalg.norm(state.pen_axis) * np.linalg.norm(self.goal_axis) + 1e-9),
                -1.0,
                1.0,
            )
        )
        info.update(
            {
                "primitive_label": self.spec.label,
                "reset_mode": self.spec.reset_mode,
                "cache_window": self.cache_window,
                "active_cache_index": int(self.active_cache_index),
                "active_cache_t": float(self.active_cache_t),
                "support_fingers": sorted(fingers),
                "support_count": int(len(support.intersection(fingers))),
                "contact_flags": contact_flags,
                "contact_geoms": metrics["geoms"].most_common(10),
                "normal_force": metrics["normal_force"],
                "force_share": metrics["force_share"],
                "total_normal_force": float(metrics["total_normal_force"]),
                "max_penetration": float(metrics["max_penetration"]),
                "tip_surface_distances": {finger: float(tip_distances.get(finger, 0.04)) for finger in FINGERS},
                "index_tip_dist": index_tip_dist,
                "thumb_tip_dist": float(tip_distances.get("thumb", 0.04)),
                "middle_tip_dist": float(tip_distances.get("middle", 0.04)),
                "ring_tip_dist": float(tip_distances.get("ring", 0.04)),
                "pinky_tip_dist": float(tip_distances.get("pinky", 0.04)),
                "initial_index_tip_dist": float(self.initial_tip_distances.get("index", 0.04)),
                "index_near_surface_hit": index_near_surface_hit,
                "index_near_surface_threshold": INDEX_NEAR_SURFACE_THRESHOLD_M,
                "desired_ratio": float(desired_ratio),
                "support_ratio": float(support_ratio),
                "terminal_hit": float(terminal_hit),
                "target_center_dist": float(np.linalg.norm(state.pen_center - self.goal_center)),
                "target_axis_error": float(np.arccos(axis_cos)),
                "pen_linear_speed": float(
                    np.linalg.norm(
                        self.env.data.qvel[
                            self.env.pen_qvel_adr : self.env.pen_qvel_adr + 3
                        ]
                    )
                ),
                "pen_angular_speed": float(np.linalg.norm(state.pen_angvel)),
                "tau": float(self._tau()),
                "contacts": ",".join(sorted(fingers)) or "none",
            }
        )
        return info

    def _primitive_reward(
        self,
        info: dict[str, Any],
        action: np.ndarray,
        prev_action: np.ndarray,
        *,
        failed: bool,
    ) -> tuple[float, dict[str, float]]:
        state = self.env.get_state()
        margin, joint_name = joint_limit_margin(self.env, state.qpos)
        terms = step_reward_terms(
            contacts=set(info["support_fingers"]),
            force_share=info["force_share"],
            pen_center=state.pen_center,
            initial_center=self.initial_center,
            target_center=self.goal_center,
            pen_axis=state.pen_axis,
            target_axis=self.goal_axis,
            z0=float(self.initial_center[2]),
            phase_delta=float(state.rotation_phase - self.initial_phase),
            action=action,
            prev_action=prev_action,
            max_penetration_m=float(info["max_penetration"]),
            joint_limit_margin=float(margin),
            middle_tip_distance=float(info.get("middle_tip_dist", 0.04)),
            pen_linear_speed=float(info.get("pen_linear_speed", 0.0)),
            pen_angular_speed=float(info.get("pen_angular_speed", 0.0)),
            tau=float(info["tau"]),
            spec=self.spec,
            weights=self.weights,
            failed=failed,
        )
        if self.spec.label in INDEX_NEAR_SURFACE_LABELS:
            terms["index_capture"] = 0.0
        if self.spec.label in INDEX_RECAPTURE_LABELS:
            terms.update(self._index_recapture_distance_terms(info))
            terms["total"] = float(
                sum(value for key, value in terms.items() if key not in {"total", "joint_limit_joint"})
            )
        terms["joint_limit_margin"] = float(margin)
        terms["joint_limit_joint"] = joint_name  # type: ignore[assignment]
        return float(self.reward_scale * terms["total"]), terms

    def _index_recapture_distance_terms(self, info: dict[str, Any]) -> dict[str, float]:
        dist = float(info.get("index_tip_dist", 0.04))
        initial_dist = max(float(info.get("initial_index_tip_dist", 0.04)), 0.004)
        positive_dist = max(0.0, dist)
        tau = float(info.get("tau", 0.0))
        index_contact = float(info.get("contact_flags", {}).get("index", 0.0))

        approach = 1.0 - min(1.0, positive_dist / 0.030)
        near = 1.0 - min(1.0, positive_dist / 0.016)
        progress = min(1.0, max(0.0, initial_dist - dist) / max(initial_dist - 0.003, 0.006))
        terminal_gate = smoothstep((tau - 0.45) / 0.40)
        penetration = -min(1.0, max(0.0, -dist - 0.003) / 0.006)
        center_guard = -min(1.0, max(0.0, float(info.get("target_center_dist", 0.0)) - 0.020) / 0.030)
        near_surface_hit = float(info.get("index_near_surface_hit", 0.0))

        terms = {
            "index_tip_approach": 1.70 * approach,
            "index_tip_progress": 0.95 * progress,
            "index_tip_near_terminal": 1.20 * terminal_gate * near,
            "index_contact_terminal": 1.20 * terminal_gate * index_contact,
            "index_tip_overpenetration": 0.55 * penetration,
            "recapture_center_guard": 0.70 * center_guard,
        }
        if self.spec.label in INDEX_NEAR_SURFACE_LABELS:
            terms["index_contact_terminal"] = -0.35 * terminal_gate * index_contact
            terms["index_near_surface_terminal"] = 1.80 * terminal_gate * near_surface_hit
        return terms

    def _failure(self, info: dict[str, Any]) -> bool:
        state = self.env.get_state()
        if float(state.pen_center[2]) < 0.045:
            return True
        if float(np.linalg.norm(state.pen_center[:2] - self.initial_center[:2])) > 0.060:
            return True
        if float(self.initial_center[2] - state.pen_center[2]) > 0.040:
            return True
        if float(info["target_center_dist"]) > 0.075:
            return True
        if float(info["max_penetration"]) > 0.022:
            return True
        if int(info["support_count"]) < 1 and self.env.step_count > int(0.15 * self.control_hz):
            return True
        return False

    def _tau(self) -> float:
        return float(np.clip(self.env.step_count / max(1, self.max_episode_steps), 0.0, 1.0))

    def _assist_delta(self) -> np.ndarray:
        delta = np.zeros(20, dtype=np.float64)
        if self.assist_mode not in {"pre_ring", "auto"}:
            return delta.astype(np.float32)
        if self.assist_mode == "auto" and "pre_ring" not in self.spec.label:
            return delta.astype(np.float32)

        metrics = contact_metrics(self.env)
        fingers = set(metrics["fingers"])
        shares = metrics["force_share"]
        tau = self._tau()
        ramp = 0.35 + 0.65 * min(1.0, tau / 0.35)

        # Thumb and index are the intended rescue contacts for B23.
        thumb_need = 1.0 if "thumb" not in fingers else max(0.25 - float(shares.get("thumb", 0.0)), 0.0) / 0.25
        index_need = 1.0 if "index" not in fingers else 0.25
        delta[17] += 0.20 * thumb_need
        delta[18] += 1.00 * thumb_need
        delta[19] += 0.85 * thumb_need
        delta[1] += 0.60 * index_need
        delta[2] += 0.45 * index_need

        # Ring remains a guard; open it when it carries too much normal force.
        ring_relief = max(0.0, float(shares.get("ring", 0.0)) - 0.32) / 0.40
        delta[9] -= 0.55 * ring_relief
        delta[10] -= 0.40 * ring_relief

        # Pinky should withdraw from the support stack.
        if "pinky" in fingers:
            delta[13] -= 1.00
            delta[14] -= 0.75

        delta *= self.assist_delta_limit * ramp
        return np.clip(delta, -self.assist_delta_limit, self.assist_delta_limit).astype(np.float32)


def summarize_short_primitive_rollout(
    infos: list[dict[str, Any]],
    returns: float,
    steps: int,
    control_hz: float,
) -> dict[str, Any]:
    if not infos:
        return {"return": float(returns), "steps": int(steps), "survival_seconds": 0.0}

    steps = max(1, int(steps))
    finger_frames: Counter[str] = Counter()
    force_sum = {finger: 0.0 for finger in FINGERS}
    for info in infos:
        for finger, flag in info.get("contact_flags", {}).items():
            if flag:
                finger_frames[finger] += 1
        for finger, share in info.get("force_share", {}).items():
            if finger in force_sum:
                force_sum[finger] += float(share)

    final_n = max(1, int(0.25 * steps))
    final_infos = infos[-final_n:]
    target_final = float(np.mean([float(info.get("terminal_hit", 0.0)) for info in final_infos]))
    desired_final = float(np.mean([float(info.get("desired_ratio", 0.0)) for info in final_infos]))
    support_final = float(np.mean([float(info.get("support_ratio", 0.0)) for info in final_infos]))
    desired = np.asarray([float(info.get("desired_ratio", 0.0)) for info in infos], dtype=np.float64)
    support = np.asarray([float(info.get("support_ratio", 0.0)) for info in infos], dtype=np.float64)
    center = np.asarray([float(info.get("target_center_dist", 0.0)) for info in infos], dtype=np.float64)
    axis = np.asarray([float(info.get("target_axis_error", 0.0)) for info in infos], dtype=np.float64)
    penetration = np.asarray([float(info.get("max_penetration", 0.0)) for info in infos], dtype=np.float64)
    action = np.asarray([float(info.get("action_norm", 0.0)) for info in infos], dtype=np.float64)
    index_tip_dist = np.asarray([float(info.get("index_tip_dist", 0.04)) for info in infos], dtype=np.float64)
    thumb_tip_dist = np.asarray([float(info.get("thumb_tip_dist", 0.04)) for info in infos], dtype=np.float64)
    middle_tip_dist = np.asarray([float(info.get("middle_tip_dist", 0.04)) for info in infos], dtype=np.float64)
    linear_speed = np.asarray([float(info.get("pen_linear_speed", 0.0)) for info in infos], dtype=np.float64)
    angular_speed = np.asarray([float(info.get("pen_angular_speed", 0.0)) for info in infos], dtype=np.float64)
    index_near_surface = np.asarray(
        [float(info.get("index_near_surface_hit", 0.0)) for info in infos],
        dtype=np.float64,
    )
    reward_terms = [info.get("primitive_reward_terms", {}) for info in infos]

    row: dict[str, Any] = {
        "return": float(returns),
        "steps": int(steps),
        "survival_seconds": float(steps / control_hz),
        "desired_ratio_mean": float(np.mean(desired)),
        "support_ratio_mean": float(np.mean(support)),
        "target_final_window_ratio": target_final,
        "desired_final_window_ratio": desired_final,
        "support_final_window_ratio": support_final,
        "target_center_dist_final": float(center[-1]),
        "target_center_dist_min": float(np.min(center)),
        "target_center_dist_max": float(np.max(center)),
        "target_axis_error_final": float(axis[-1]),
        "target_axis_error_min": float(np.min(axis)),
        "target_axis_error_max": float(np.max(axis)),
        "max_penetration_mm": float(np.max(penetration) * 1000.0),
        "index_tip_dist_mean": float(np.mean(index_tip_dist)),
        "index_tip_dist_min": float(np.min(index_tip_dist)),
        "index_tip_dist_final": float(index_tip_dist[-1]),
        "index_near_surface_ratio": float(np.mean(index_near_surface)),
        "index_near_surface_final": float(index_near_surface[-1]),
        "thumb_tip_dist_mean": float(np.mean(thumb_tip_dist)),
        "middle_tip_dist_mean": float(np.mean(middle_tip_dist)),
        "middle_tip_dist_final": float(middle_tip_dist[-1]),
        "pen_linear_speed_final": float(linear_speed[-1]),
        "pen_linear_speed_max": float(np.max(linear_speed)),
        "pen_angular_speed_final": float(angular_speed[-1]),
        "pen_angular_speed_max": float(np.max(angular_speed)),
        "initial_index_tip_dist": float(infos[0].get("initial_index_tip_dist", 0.04)),
        "action_norm_mean": float(np.mean(action)),
        "terminated": bool(infos[-1].get("failed", False)),
        "final_contacts": str(infos[-1].get("contacts", "none")),
        "active_cache_index": int(infos[0].get("active_cache_index", -1)),
        "active_cache_t": float(infos[0].get("active_cache_t", 0.0)),
    }
    for finger in FINGERS:
        row[f"{finger}_ratio"] = float(finger_frames[finger] / steps)
        row[f"{finger}_force_share_mean"] = float(force_sum[finger] / steps)
    for key in [
        "alive",
        "support_contact",
        "desired_contact",
        "terminal_contact",
        "thumb_support",
        "index_capture",
        "ring_guard_contact",
        "ring_overload",
        "pinky",
        "penetration",
        "index_tip_approach",
        "index_tip_progress",
        "index_tip_near_terminal",
        "index_contact_terminal",
        "index_near_surface_terminal",
        "index_tip_overpenetration",
        "recapture_center_guard",
        "middle_proximity",
        "terminal_linear_speed",
        "terminal_angular_speed",
        "failure",
        "total",
    ]:
        vals = [float(terms.get(key, 0.0)) for terms in reward_terms]
        row[f"reward_{key}_mean"] = float(np.mean(vals)) if vals else 0.0
    return row
