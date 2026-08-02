from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from m3_oracle_hold_env import Dex5PenSpinOracleHoldEnv, load_preset, summarize_rollout
from run_m2_9_micro_gait import reset_to_preset
from tune_m2_7_tripod_fingertip_reset import SUPPORT_FINGERS, set_state


DEFAULT_MANIFEST = Path("dex5_penspin/reports/manus_mujoco_presets/canonical_resets/manifest.json")


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> list[Path]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    paths = data.get("accepted_resets") or data.get("all_resets") or []
    if not paths:
        raise ValueError(f"No reset paths found in manifest: {path}")
    return [Path(p) for p in paths]


def phase_metrics_from_infos(infos: list[dict[str, Any]]) -> dict[str, float]:
    if not infos:
        return {
            "final_progress": 0.0,
            "positive_progress": 0.0,
            "negative_progress": 0.0,
            "monotonic_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    directed = np.asarray([float(info.get("directed_phase", 0.0)) for info in infos], dtype=np.float64)
    deltas = np.diff(np.concatenate([[0.0], directed]))
    positive = float(np.sum(np.clip(deltas, 0.0, None)))
    negative = float(np.sum(np.clip(-deltas, 0.0, None)))
    running_max = np.maximum.accumulate(directed)
    drawdown = running_max - directed
    return {
        "final_progress": float(directed[-1]),
        "positive_progress": positive,
        "negative_progress": negative,
        "monotonic_ratio": float(positive / (positive + negative + 1e-9)),
        "max_drawdown": float(np.max(drawdown)),
    }


class Dex5PenSpinDirectedPhaseEnv(Dex5PenSpinOracleHoldEnv):
    """M3.2 oracle environment with multi-reset sampling and directed phase reward."""

    def __init__(
        self,
        *,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        preset_paths: list[str | Path] | None = None,
        phase_direction: float = -1.0,
        phase_reward_scale: float = 35.0,
        negative_phase_penalty: float = 45.0,
        hold_reward_scale: float = 1.0,
        target_progress: float = 0.20,
        **kwargs: Any,
    ) -> None:
        self.preset_paths = [Path(p) for p in (preset_paths or load_manifest(manifest_path))]
        self.presets = [load_preset(path) for path in self.preset_paths]
        self.active_reset_index = 0
        self.active_reset_path = self.preset_paths[0]
        self.phase_direction = float(np.sign(phase_direction) or 1.0)
        self.phase_reward_scale = float(phase_reward_scale)
        self.negative_phase_penalty = float(negative_phase_penalty)
        self.hold_reward_scale = float(hold_reward_scale)
        self.target_progress = float(target_progress)
        self.prev_directed_phase = 0.0
        self.max_directed_phase = 0.0
        self.positive_progress = 0.0
        self.negative_progress = 0.0
        self.reset_counts: Counter[int] = Counter()

        kwargs = dict(kwargs)
        kwargs.pop("preset_path", None)
        super().__init__(preset_path=self.preset_paths[0], **kwargs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}
        if "reset_index" in options:
            reset_index = int(options["reset_index"]) % len(self.presets)
        else:
            reset_index = int(self.np_random.integers(0, len(self.presets)))
        self.active_reset_index = reset_index
        self.active_reset_path = self.preset_paths[reset_index]
        self.preset = self.presets[reset_index]
        self.reset_counts[reset_index] += 1

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
        self.prev_directed_phase = 0.0
        self.max_directed_phase = 0.0
        self.positive_progress = 0.0
        self.negative_progress = 0.0
        info = self._make_info(action=np.zeros(20, dtype=np.float32), base_info=self.env._info())
        obs = self._get_obs(info)
        self._last_info = info
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, hold_reward, terminated, truncated, info = super().step(action)
        directed_phase = self.phase_direction * (float(info["rotation_phase"]) - self.initial_phase)
        phase_delta = directed_phase - self.prev_directed_phase
        positive_delta = max(0.0, float(phase_delta))
        negative_delta = max(0.0, float(-phase_delta))
        self.positive_progress += positive_delta
        self.negative_progress += negative_delta
        self.max_directed_phase = max(self.max_directed_phase, directed_phase)
        drawdown = self.max_directed_phase - directed_phase
        self.prev_directed_phase = directed_phase

        phase_reward = self.phase_reward_scale * positive_delta - self.negative_phase_penalty * negative_delta
        # Once the tiny target is reached, stop encouraging more motion in M3.2.
        phase_reward -= 2.0 * max(0.0, directed_phase - self.target_progress)
        reward = self.hold_reward_scale * float(hold_reward) + phase_reward

        info["hold_reward"] = float(hold_reward)
        info["phase_reward"] = float(phase_reward)
        info["oracle_reward"] = float(reward)
        info["directed_phase"] = float(directed_phase)
        info["phase_delta"] = float(phase_delta)
        info["positive_progress"] = float(self.positive_progress)
        info["negative_progress"] = float(self.negative_progress)
        info["monotonic_ratio"] = float(self.positive_progress / (self.positive_progress + self.negative_progress + 1e-9))
        info["max_drawdown"] = float(drawdown)
        info["active_reset_index"] = int(self.active_reset_index)
        info["active_reset_path"] = str(self.active_reset_path)
        return obs, float(reward), terminated, truncated, info


def summarize_directed_rollout(infos: list[dict[str, Any]], returns: float, steps: int, control_hz: float) -> dict[str, Any]:
    summary = summarize_rollout(infos, returns, steps, control_hz)
    summary.update(phase_metrics_from_infos(infos))
    summary["reset_indices"] = sorted({int(info.get("active_reset_index", -1)) for info in infos})
    if infos:
        summary["active_reset_index"] = int(infos[0].get("active_reset_index", -1))
        summary["active_reset_path"] = str(infos[0].get("active_reset_path", ""))
    return summary
