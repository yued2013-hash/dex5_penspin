from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Export v14 H6 failures observed 6-12 control steps after semantic latch."
)
parser.add_argument(
    "--task",
    default="Dex5-P1P2-P1617SingleRoute539Conditioned120HzH6TerminalDynamics-Direct-v0",
)
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--terminal-adapter", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--summary", required=True, type=Path)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--episodes", type=int, default=2048)
parser.add_argument("--max-steps", type=int, default=1600)
parser.add_argument("--max-rows", type=int, default=2048)
parser.add_argument("--latch-step-min", type=int, default=6)
parser.add_argument("--latch-step-max", type=int, default=12)
parser.add_argument("--seed", type=int, default=2821)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from skrl.utils import set_seed as set_skrl_seed
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


class TerminalStopAdapter(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, residual_scale: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(12, hidden_size), nn.Tanh())
        self.residual_head = nn.Linear(hidden_size, output_size)
        self.stop_head = nn.Linear(hidden_size, 1)
        self.residual_scale = residual_scale

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(condition)
        terminal = condition[:, :1]
        residual = terminal * self.residual_scale * torch.tanh(self.residual_head(hidden))
        stop = terminal * torch.sigmoid(self.stop_head(hidden))
        return residual, stop


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(direct, action: torch.Tensor, env_id: int, latch_age: int) -> dict[str, np.ndarray]:
    direct._compute_intermediate_values()
    eligible, metrics = direct._instantaneous_gate()
    condition, semantic_margin, _ = direct._terminal_condition_metrics()
    qpos = torch.cat((direct.hand_qpos, direct.pen_pos, direct.pen_quat), dim=-1)
    qvel = torch.cat((direct.hand_qvel, direct.pen_linvel, direct.pen_angvel), dim=-1)
    feature = direct._ring_axis_features()
    index = torch.tensor([env_id], dtype=torch.long, device=direct.device)

    def one(value: torch.Tensor) -> np.ndarray:
        return value[index].detach().cpu().numpy()[0].copy()

    return {
        "source_qpos": one(qpos),
        "source_qvel": one(qvel),
        "source_target_qpos": one(direct.policy_targets),
        "source_policy_action": one(action),
        "target_qpos": one(qpos),
        "target_qvel": one(qvel),
        "source_global_bank_ids": one(direct.current_global_bank_ids),
        "latch_age": np.asarray(latch_age, dtype=np.int64),
        "hold_steps": one(direct.p1_p2_hold_count),
        "route_step": one(direct.episode_length_buf),
        "eligible": one(eligible),
        "target_distance_mm": one(metrics["target_distance"] * 1000.0),
        "progress": one(metrics["progress"]),
        "manifold_under": one(metrics["manifold_under"] > 0.5),
        "manifold_semantic": one(metrics["manifold_semantic"] > 0.5),
        "manifold_semantic_soft": one(metrics["manifold_semantic_soft"]),
        "semantic_margin": one(semantic_margin),
        "target_condition": one(condition),
        "target_condition_delta": one(direct.terminal_condition_delta),
        "ring_feature_m": one(feature),
        "contact_force_thumb_index_middle_ring_pinky": one(direct.contact_force_norm),
        "position_drift_mm": one(metrics["position_drift"] * 1000.0),
        "axis_drift_deg": one(torch.rad2deg(metrics["axis_drift"])),
        "angular_speed": one(metrics["angular_speed"]),
        "linear_speed": one(metrics["linear_speed"]),
        "max_penetration_mm": one(direct.episode_max_penetration * 1000.0),
    }


def _stack_rows(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    payload = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    actions = payload["source_policy_action"].astype(np.float32)
    payload["source_teacher_actions"] = actions[:, None, :]
    payload["phase_ids"] = np.full(len(rows), 2, dtype=np.int64)
    payload["policy_success_episode"] = np.zeros(len(rows), dtype=np.bool_)
    payload["arrival_bucket"] = np.full(len(rows), "H6F", dtype="<U3")
    return payload


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    checkpoint = args.checkpoint.resolve()
    adapter_path = args.terminal_adapter.resolve()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    agent_cfg["seed"] = int(args.seed)
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    set_skrl_seed(args.seed)
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped, agent_cfg)
    runner.agent.load(str(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    artifact = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if artifact.get("base_checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("v14 adapter and base checkpoint SHA256 do not match")
    if not artifact.get("has_stop_head", False):
        raise ValueError("The supplied terminal adapter has no stop head")
    action_indices = torch.as_tensor(
        artifact["learned_action_indices"], dtype=torch.long, device=direct.device
    )
    input_columns = torch.as_tensor(
        artifact["input_columns"], dtype=torch.long, device=direct.device
    )
    adapter = TerminalStopAdapter(
        int(artifact["hidden_size"]), len(action_indices), float(artifact["residual_scale"])
    ).to(direct.device)
    adapter.load_state_dict(artifact["state_dict"])
    adapter.eval()

    observations, _ = wrapped.reset()
    states = wrapped.state()
    latch_age = torch.zeros(direct.num_envs, dtype=torch.long, device=direct.device)
    pending: list[list[dict[str, np.ndarray]]] = [[] for _ in range(direct.num_envs)]
    accepted: list[dict[str, np.ndarray]] = []
    failed_episodes_with_latch = 0
    successful_latched_episodes = 0
    steps = 0
    while (
        int(direct.completed_p1_p2_episodes.item()) < int(args.episodes)
        and steps < int(args.max_steps)
        and len(accepted) < int(args.max_rows)
    ):
        with torch.inference_mode():
            outputs = runner.agent.act(
                observations, states, timestep=steps, timesteps=int(args.max_steps)
            )
            actions = outputs[-1].get("mean_actions", outputs[0]).clone()
            residual, stop = adapter(observations[:, input_columns])
            drive = actions[:, action_indices] + residual
            actions[:, action_indices] = (1.0 - stop) * drive

            latched = direct.terminal_semantic_hold_latched
            latch_age = torch.where(latched, latch_age + 1, torch.zeros_like(latch_age))
            candidate_ids = torch.nonzero(
                latched
                & (latch_age >= int(args.latch_step_min))
                & (latch_age <= int(args.latch_step_max))
                & ~direct.episode_p1_p2_success,
                as_tuple=False,
            ).squeeze(-1)
            for env_id in candidate_ids.detach().cpu().tolist():
                pending[env_id].append(
                    _snapshot(direct, actions, env_id, int(latch_age[env_id].item()))
                )

            observations, _, terminated, truncated, _ = wrapped.step(actions)
            states = wrapped.state()
            done_mask = (terminated | truncated).reshape(-1)
            done_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
            for env_id in done_ids.detach().cpu().tolist():
                if pending[env_id]:
                    if bool(direct.last_completed_p1_p2_success[env_id].item()):
                        successful_latched_episodes += 1
                    else:
                        failed_episodes_with_latch += 1
                        for row in pending[env_id]:
                            row["failed_episode_id"] = np.asarray(
                                failed_episodes_with_latch - 1, dtype=np.int64
                            )
                        accepted.extend(pending[env_id])
                pending[env_id].clear()
                latch_age[env_id] = 0
            steps += 1

    accepted = accepted[: int(args.max_rows)]
    if not accepted:
        raise RuntimeError("No failed H6 latch states were collected")
    payload = _stack_rows(accepted)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "terminal_adapter": str(adapter_path),
        "terminal_adapter_sha256": sha256(adapter_path),
        "output": str(output),
        "output_sha256": sha256(output),
        "completed_episodes": int(direct.completed_p1_p2_episodes.item()),
        "failed_episodes_with_latch": failed_episodes_with_latch,
        "successful_latched_episodes_discarded": successful_latched_episodes,
        "accepted_rows": len(accepted),
        "latch_age_counts": dict(sorted(Counter(payload["latch_age"].tolist()).items())),
        "semantic_rows": int(payload["manifold_semantic"].sum()),
        "under_rows": int(payload["manifold_under"].sum()),
        "eligible_rows": int(payload["eligible"].sum()),
        "max_penetration_mm": float(payload["max_penetration_mm"].max()),
        "drops": int(direct.completed_drop_count.item()),
        "invalid": int(direct.completed_invalid_count.item()),
        "normal_physx_rollout": True,
        "ppo_started": False,
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_V14_H6_LATCH_FAILURES=" + json.dumps(summary, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
