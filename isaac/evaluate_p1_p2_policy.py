from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESET_DIR = PROJECT_ROOT / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank"
SPLIT_PATHS = {
    "train": RESET_DIR / "P1_P2_train_ids.npy",
    "validation": RESET_DIR / "P1_P2_validation_ids.npy",
    "test": RESET_DIR / "P1_P2_test_ids.npy",
}

parser = argparse.ArgumentParser(description="Evaluate a P1-to-P2 policy on a frozen split.")
parser.add_argument("--task", default="Dex5-P1P2-C1-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--split", choices=tuple(SPLIT_PATHS), default="validation")
parser.add_argument(
    "--split_ids_path",
    type=Path,
    default=None,
    help="Optional explicit split IDs; preserves specialized curriculum subsets.",
)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--local_checkpoint", default="")
parser.add_argument("--local_switch_distance_mm", type=float, default=8.0)
parser.add_argument("--local_action_scale", type=float, default=0.25)
parser.add_argument(
    "--terminal_adapter",
    default="",
    help="Optional nonlinear terminal residual adapter composed with the base checkpoint.",
)
parser.add_argument("--terminal_adapter_scale", type=float, default=1.0)
parser.add_argument(
    "--post_stop_correction_scale",
    type=float,
    default=1.0,
    help="Independent scale for a post-stop correction head.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from skrl.utils.runner.torch import Runner
from skrl.utils import set_seed as set_skrl_seed

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


class TerminalResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, residual_scale: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size), nn.Tanh(), nn.Linear(hidden_size, output_size)
        )
        self.residual_scale = residual_scale

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return (
            condition[:, :1]
            * self.residual_scale
            * torch.tanh(self.net(condition))
        )


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
        residual = terminal * self.residual_scale * torch.tanh(
            self.residual_head(hidden)
        )
        stop = terminal * torch.sigmoid(self.stop_head(hidden))
        return residual, stop


class PostStopCorrectionAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        residual_scale: float,
        correction_hidden_size: int,
        correction_scale: float,
    ) -> None:
        super().__init__()
        self.stop_adapter = TerminalStopAdapter(hidden_size, output_size, residual_scale)
        self.correction_net = nn.Sequential(
            nn.Linear(12, correction_hidden_size),
            nn.Tanh(),
            nn.Linear(correction_hidden_size, output_size),
        )
        self.correction_scale = correction_scale

    def forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual, stop = self.stop_adapter(condition)
        correction_gate = condition[:, :1] * condition[:, 11:12]
        correction = (
            correction_gate
            * self.correction_scale
            * torch.tanh(self.correction_net(condition))
        )
        return residual, stop, correction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    if args_cli.split_ids_path is not None:
        split_path = args_cli.split_ids_path.resolve()
    elif env_cfg.use_all_bundle_rows:
        split_path = None
    elif env_cfg.split_ids_path:
        split_path = Path(env_cfg.split_ids_path).resolve()
    else:
        split_path = SPLIT_PATHS[args_cli.split]
    checkpoint = Path(args_cli.checkpoint).resolve()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if split_path is not None:
        env_cfg.split_ids_path = str(split_path)
    env_cfg.split_name = args_cli.split
    agent_cfg["seed"] = args_cli.seed

    with np.load(env_cfg.bundle_path) as bank:
        phase_ids = np.asarray(bank["phase_ids"], dtype=np.int64)
        source_global_ids = (
            np.asarray(bank["source_global_bank_ids"], dtype=np.int64)
            if "source_global_bank_ids" in bank
            else np.arange(phase_ids.size, dtype=np.int64)
        )
    if split_path is None:
        selected_rows = np.flatnonzero(phase_ids == int(env_cfg.source_phase_id))
    else:
        split_ids = np.asarray(np.load(split_path, allow_pickle=False), dtype=np.int64)
        selected_rows = split_ids[phase_ids[split_ids] == int(env_cfg.source_phase_id)]
    allowed_source_ids = set(source_global_ids[selected_rows].tolist())

    env = gym.make(args_cli.task, cfg=env_cfg)
    direct = env.unwrapped
    set_skrl_seed(args_cli.seed)
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped, agent_cfg)
    runner.agent.load(str(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    terminal_adapter_path = (
        Path(args_cli.terminal_adapter).resolve() if args_cli.terminal_adapter else None
    )
    terminal_adapter = None
    terminal_adapter_has_stop = False
    terminal_adapter_has_post_stop_correction = False
    terminal_adapter_indices = None
    terminal_adapter_columns = None
    if terminal_adapter_path is not None:
        artifact = torch.load(
            terminal_adapter_path, map_location="cpu", weights_only=False
        )
        expected_base_hash = artifact.get("base_checkpoint_sha256")
        if expected_base_hash and expected_base_hash != sha256(checkpoint):
            raise ValueError("Terminal adapter was trained against a different base checkpoint")
        terminal_adapter_indices = torch.as_tensor(
            artifact["learned_action_indices"], dtype=torch.long, device=direct.device
        )
        terminal_adapter_columns = torch.as_tensor(
            artifact["input_columns"], dtype=torch.long, device=direct.device
        )
        terminal_adapter_has_stop = bool(artifact.get("has_stop_head", False))
        terminal_adapter_has_post_stop_correction = bool(
            artifact.get("has_post_stop_correction", False)
        )
        if terminal_adapter_has_post_stop_correction:
            terminal_adapter = PostStopCorrectionAdapter(
                int(artifact["hidden_size"]),
                len(artifact["learned_action_indices"]),
                float(artifact["residual_scale"]),
                int(artifact["correction_hidden_size"]),
                float(artifact["correction_scale"]),
            ).to(direct.device)
        else:
            adapter_class = (
                TerminalStopAdapter if terminal_adapter_has_stop else TerminalResidualAdapter
            )
            terminal_adapter = adapter_class(
                int(artifact["hidden_size"]),
                len(artifact["learned_action_indices"]),
                float(artifact["residual_scale"]),
            ).to(direct.device)
        terminal_adapter.load_state_dict(artifact["state_dict"])
        terminal_adapter.eval()
    local_checkpoint = Path(args_cli.local_checkpoint).resolve() if args_cli.local_checkpoint else None
    local_runner = None
    if local_checkpoint is not None:
        local_runner = Runner(wrapped, agent_cfg)
        local_runner.agent.load(str(local_checkpoint))
        local_runner.agent.enable_training_mode(False, apply_to_models=True)

    observations, _ = wrapped.reset()
    states = wrapped.state()
    steps = 0
    seen_global_ids: set[int] = set()
    contact_sum = torch.zeros(5, device=direct.device)
    force_sum = torch.zeros(5, device=direct.device)
    eligible_sum = 0
    sample_steps = 0
    gate_pass_sum: dict[str, int] = {}
    target_distance_sum = 0.0
    target_distance_min = float("inf")
    progress_sum = 0.0
    max_penetration = 0.0
    max_position_drift = 0.0
    max_ring_progress = -float("inf")
    per_id_episodes: dict[int, int] = {}
    per_id_successes: dict[int, int] = {}
    per_id_drops: dict[int, int] = {}
    local_latched = torch.zeros(direct.num_envs, dtype=torch.bool, device=direct.device)
    local_action_steps = 0
    while int(direct.completed_p1_p2_episodes.item()) < args_cli.episodes and steps < args_cli.max_steps:
        with torch.inference_mode():
            outputs = runner.agent.act(
                observations, states, timestep=steps, timesteps=args_cli.max_steps
            )
            actions = outputs[-1].get("mean_actions", outputs[0])
            if terminal_adapter is not None:
                adapter_output = terminal_adapter(
                    observations[:, terminal_adapter_columns]
                )
                if terminal_adapter_has_post_stop_correction:
                    residual, stop, correction = adapter_output
                    drive = actions[:, terminal_adapter_indices] + (
                        float(args_cli.terminal_adapter_scale) * residual
                    )
                    actions[:, terminal_adapter_indices] = (
                        (1.0 - stop) * drive
                        + float(args_cli.post_stop_correction_scale) * correction
                    )
                elif terminal_adapter_has_stop:
                    residual, stop = adapter_output
                    drive = actions[:, terminal_adapter_indices] + (
                        float(args_cli.terminal_adapter_scale) * residual
                    )
                    actions[:, terminal_adapter_indices] = (1.0 - stop) * drive
                else:
                    residual = adapter_output
                    actions[:, terminal_adapter_indices] += (
                        float(args_cli.terminal_adapter_scale) * residual
                    )
            if local_runner is not None:
                _, _, switch_distance, _ = direct._ring_geometry()
                local_latched |= switch_distance <= args_cli.local_switch_distance_mm / 1000.0
                local_outputs = local_runner.agent.act(
                    observations, states, timestep=steps, timesteps=args_cli.max_steps
                )
                local_actions = local_outputs[-1].get("mean_actions", local_outputs[0])
                local_actions = float(args_cli.local_action_scale) * local_actions
                actions = torch.where(local_latched.unsqueeze(-1), local_actions, actions)
                local_action_steps += int(local_latched.sum().item())
            observations, _, _, _, _ = wrapped.step(actions)
            states = wrapped.state()
            completed_mask = direct.last_completed_global_bank_ids >= 0
            if completed_mask.any():
                completed_ids = direct.last_completed_global_bank_ids[completed_mask].cpu().tolist()
                completed_success = direct.last_completed_p1_p2_success[completed_mask].cpu().tolist()
                completed_drop = direct.last_completed_p1_p2_drop[completed_mask].cpu().tolist()
                for global_id, succeeded, dropped in zip(
                    completed_ids, completed_success, completed_drop, strict=True
                ):
                    global_id = int(global_id)
                    per_id_episodes[global_id] = per_id_episodes.get(global_id, 0) + 1
                    per_id_successes[global_id] = per_id_successes.get(global_id, 0) + int(succeeded)
                    per_id_drops[global_id] = per_id_drops.get(global_id, 0) + int(dropped)
                local_latched[completed_mask] = False
            direct._compute_intermediate_values()
            eligible, metrics = direct._instantaneous_gate()
            _, progress, target_distance, _ = direct._ring_geometry()
            contacts = direct.contact_force_norm > direct.cfg.p2_contact_force_n
            contact_sum += contacts.float().sum(dim=0)
            force_sum += direct.contact_force_norm.sum(dim=0)
            eligible_sum += int(eligible.sum().item())
            sample_steps += direct.num_envs
            stage = int(direct.cfg.curriculum_stage)
            target_progress = float(direct.cfg.stage_target_progress[stage])
            gate_checks = {
                "progress": torch.abs(progress - target_progress)
                <= direct.cfg.stage_progress_tolerance[stage],
                "target_distance": target_distance
                <= direct.cfg.stage_target_distance_m[stage],
                "geometry": (
                    (torch.abs(progress - target_progress)
                    <= direct.cfg.stage_progress_tolerance[stage])
                    & (target_distance <= direct.cfg.stage_target_distance_m[stage])
                ),
                "triad": metrics["triad"] > 0.5,
                "manifold_semantic": metrics["manifold_semantic"] > 0.5,
                "manifold_feature_bounds": metrics["manifold_feature_bounds"] > 0.5,
                "manifold_under": metrics["manifold_under"] > 0.5,
                "manifold_gap": metrics["manifold_gap_gate"] > 0.5,
                "ring_contact": metrics["ring_contact"] > 0.5,
                "position_drift": metrics["position_drift"]
                <= direct.cfg.p2_max_position_drift_m,
                "axis_drift": metrics["axis_drift"]
                <= torch.deg2rad(
                    torch.full_like(metrics["axis_drift"], direct.cfg.p2_max_axis_drift_deg)
                ),
                "angular_speed": metrics["angular_speed"]
                <= direct.cfg.p2_max_angular_speed_rad_s,
                "linear_speed": metrics["linear_speed"]
                <= direct.cfg.p2_max_linear_speed_m_s,
                "penetration": direct.episode_max_penetration
                <= direct.cfg.p2_max_penetration_m,
                "ring_force_share": metrics["ring_force_share"]
                <= direct.cfg.p2_max_ring_force_share,
                "ring_force": direct.contact_force_norm[:, 3]
                <= direct.cfg.p2_max_ring_force_n,
                "pinky_force": direct.contact_force_norm[:, 4]
                <= direct.cfg.p2_max_pinky_force_n,
                "not_failed": metrics["failed"] < 0.5,
            }
            for name, passed in gate_checks.items():
                gate_pass_sum[name] = gate_pass_sum.get(name, 0) + int(passed.sum().item())
            target_distance_sum += float(target_distance.sum().item())
            target_distance_min = min(target_distance_min, float(target_distance.min().item()))
            progress_sum += float(progress.sum().item())
            max_penetration = max(
                max_penetration, float(direct.episode_max_penetration.max().item())
            )
            max_position_drift = max(
                max_position_drift,
                float(torch.linalg.norm(direct.pen_pos - direct.init_pen_pos, dim=-1).max().item()),
            )
            max_ring_progress = max(max_ring_progress, float(progress.max().item()))
            seen_global_ids.update(direct.current_global_bank_ids.cpu().tolist())
        steps += 1

    leaked_ids = sorted(seen_global_ids - allowed_source_ids)
    completed = int(direct.completed_p1_p2_episodes.item())
    successes = int(direct.completed_p1_p2_successes.item())
    drops = int(direct.completed_drop_count.item())
    invalid = int(direct.completed_invalid_count.item())
    total_force = force_sum.sum().clamp_min(1.0e-9)
    result = {
        "task": args_cli.task,
        "split": args_cli.split,
        "seed": args_cli.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "terminal_adapter": (
            str(terminal_adapter_path) if terminal_adapter_path is not None else None
        ),
        "terminal_adapter_sha256": (
            sha256(terminal_adapter_path) if terminal_adapter_path is not None else None
        ),
        "terminal_adapter_scale": (
            float(args_cli.terminal_adapter_scale)
            if terminal_adapter_path is not None
            else None
        ),
        "post_stop_correction_scale": (
            float(args_cli.post_stop_correction_scale)
            if terminal_adapter_has_post_stop_correction
            else None
        ),
        "local_checkpoint": str(local_checkpoint) if local_checkpoint is not None else None,
        "local_checkpoint_sha256": sha256(local_checkpoint) if local_checkpoint is not None else None,
        "local_switch_distance_mm": args_cli.local_switch_distance_mm if local_checkpoint is not None else None,
        "local_action_scale": args_cli.local_action_scale if local_checkpoint is not None else None,
        "local_action_step_ratio": local_action_steps / max(sample_steps, 1),
        "frozen_bank": env_cfg.bundle_path,
        "frozen_bank_sha256": sha256(Path(env_cfg.bundle_path)),
        "split_ids": str(split_path) if split_path is not None else None,
        "split_ids_sha256": sha256(split_path) if split_path is not None else None,
        "allowed_source_ids": len(allowed_source_ids),
        "seen_p1_ids": len(seen_global_ids),
        "leaked_id_count": len(leaked_ids),
        "leaked_ids": leaked_ids[:20],
        "num_envs": args_cli.num_envs,
        "requested_episodes": args_cli.episodes,
        "completed_episodes": completed,
        "successes": successes,
        "success_rate": successes / max(completed, 1),
        "drops": drops,
        "drop_rate": drops / max(completed, 1),
        "invalid_episodes": invalid,
        "completed_max_penetration_mm": float(direct.completed_max_penetration.item() * 1000.0),
        "observed_max_penetration_mm": max_penetration * 1000.0,
        "observed_max_position_drift_mm": max_position_drift * 1000.0,
        "max_ring_progress": max_ring_progress,
        "eligible_step_ratio": eligible_sum / max(sample_steps, 1),
        "gate_pass_ratios": {
            name: count / max(sample_steps, 1) for name, count in gate_pass_sum.items()
        },
        "ring_target_distance_mean_mm": target_distance_sum / max(sample_steps, 1) * 1000.0,
        "ring_target_distance_min_mm": target_distance_min * 1000.0,
        "ring_progress_mean": progress_sum / max(sample_steps, 1),
        "step_contact_ratio_thumb_index_middle_ring_pinky": (
            contact_sum / max(sample_steps, 1)
        ).cpu().tolist(),
        "force_share_thumb_index_middle_ring_pinky": (force_sum / total_force).cpu().tolist(),
        "simulation_steps": steps,
        "per_source_id": {
            str(global_id): {
                "episodes": episode_count,
                "successes": per_id_successes.get(global_id, 0),
                "success_rate": per_id_successes.get(global_id, 0) / max(episode_count, 1),
                "drops": per_id_drops.get(global_id, 0),
            }
            for global_id, episode_count in sorted(per_id_episodes.items())
        },
        "evaluation_complete": completed >= args_cli.episodes,
        "success_definition": (
            f"{env_cfg.p2_success_gate_mode} manifold gate; curriculum-stage ring distance; "
            "thumb/index/middle simultaneous contact; pen drift and velocity gates; "
            "ring/pinky load gates; stage hold; penetration <=0.6mm; no drop or invalid values"
        ),
    }
    output = Path(args_cli.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_EVALUATION=" + json.dumps(result, ensure_ascii=False), flush=True)
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
