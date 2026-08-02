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

parser = argparse.ArgumentParser(
    description="Export train-only dynamic near-P2/P2 states from P1->P2 policy rollouts."
)
parser.add_argument("--task", default="Dex5-P1P2-ReverseR0ValidatedHoldSlow-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--split", choices=tuple(SPLIT_PATHS), default="train")
parser.add_argument(
    "--split_ids_path",
    type=Path,
    default=None,
    help="Optional explicit train IDs for a specialized curriculum subset.",
)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=2048)
parser.add_argument("--max_steps", type=int, default=1200)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--min_hold_steps", type=int, default=0)
parser.add_argument("--strict_hold_steps", type=int, default=6)
parser.add_argument("--near_distance_mm", type=float, default=5.0)
parser.add_argument("--min_distance_mm", type=float, default=0.0)
parser.add_argument("--strict_distance_mm", type=float, default=2.45)
parser.add_argument("--max_rows", type=int, default=2000)
parser.add_argument(
    "--first_band_entry",
    action="store_true",
    help="Keep the first state entering [min_distance_mm, near_distance_mm].",
)
parser.add_argument(
    "--action_noise_std",
    type=float,
    default=0.0,
    help="Gaussian policy-action noise applied through normal environment steps.",
)
parser.add_argument(
    "--require-semantic-failure",
    action="store_true",
    help="Only export states that still fail ring-under or manifold semantic gates.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from skrl.utils.runner.torch import Runner
from skrl.utils import set_seed as set_skrl_seed

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_state(num_envs: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "qpos": torch.zeros((num_envs, 27), device=device),
        "qvel": torch.zeros((num_envs, 26), device=device),
        "target": torch.zeros((num_envs, 20), device=device),
        "action": torch.zeros((num_envs, 20), device=device),
        "bank_id": torch.full((num_envs,), -1, dtype=torch.long, device=device),
        "score": torch.full((num_envs,), -float("inf"), device=device),
        "target_distance": torch.full((num_envs,), float("inf"), device=device),
        "progress": torch.zeros((num_envs,), device=device),
        "hold_steps": torch.zeros((num_envs,), dtype=torch.long, device=device),
        "penetration": torch.zeros((num_envs,), device=device),
        "position_drift": torch.zeros((num_envs,), device=device),
        "axis_drift": torch.zeros((num_envs,), device=device),
        "angular_speed": torch.zeros((num_envs,), device=device),
        "linear_speed": torch.zeros((num_envs,), device=device),
        "contact_force": torch.zeros((num_envs, 5), device=device),
        "eligible": torch.zeros((num_envs,), dtype=torch.bool, device=device),
        "first_target_entry_radial_speed": torch.full((num_envs,), float("nan"), device=device),
        "first_target_entry_pen_angular_speed": torch.full((num_envs,), float("nan"), device=device),
        "first_target_entry_TIM": torch.zeros((num_envs,), dtype=torch.bool, device=device),
        "first_target_entry_axis_drift": torch.full((num_envs,), float("nan"), device=device),
        "first_exit_reason": torch.full((num_envs,), -1, dtype=torch.long, device=device),
        "target_reentry_count": torch.zeros((num_envs,), dtype=torch.long, device=device),
        "longest_consecutive_target_steps": torch.zeros((num_envs,), dtype=torch.long, device=device),
        "route_step": torch.zeros((num_envs,), dtype=torch.long, device=device),
        "radial_speed": torch.full((num_envs,), float("nan"), device=device),
        "manifold_under": torch.zeros((num_envs,), dtype=torch.bool, device=device),
        "manifold_semantic": torch.zeros((num_envs,), dtype=torch.bool, device=device),
        "manifold_semantic_soft": torch.zeros((num_envs,), device=device),
        "ring_feature": torch.zeros((num_envs, 3), device=device),
    }


def _reset_best(best: dict[str, torch.Tensor], env_ids: torch.Tensor) -> None:
    best["qpos"][env_ids] = 0.0
    best["qvel"][env_ids] = 0.0
    best["target"][env_ids] = 0.0
    best["action"][env_ids] = 0.0
    best["bank_id"][env_ids] = -1
    best["score"][env_ids] = -float("inf")
    best["target_distance"][env_ids] = float("inf")
    best["progress"][env_ids] = 0.0
    best["hold_steps"][env_ids] = 0
    best["penetration"][env_ids] = 0.0
    best["position_drift"][env_ids] = 0.0
    best["axis_drift"][env_ids] = 0.0
    best["angular_speed"][env_ids] = 0.0
    best["linear_speed"][env_ids] = 0.0
    best["contact_force"][env_ids] = 0.0
    best["eligible"][env_ids] = False
    best["first_target_entry_radial_speed"][env_ids] = float("nan")
    best["first_target_entry_pen_angular_speed"][env_ids] = float("nan")
    best["first_target_entry_TIM"][env_ids] = False
    best["first_target_entry_axis_drift"][env_ids] = float("nan")
    best["first_exit_reason"][env_ids] = -1
    best["target_reentry_count"][env_ids] = 0
    best["longest_consecutive_target_steps"][env_ids] = 0
    best["route_step"][env_ids] = 0
    best["radial_speed"][env_ids] = float("nan")
    best["manifold_under"][env_ids] = False
    best["manifold_semantic"][env_ids] = False
    best["manifold_semantic_soft"][env_ids] = 0.0
    best["ring_feature"][env_ids] = 0.0


def _current_qpos_qvel(direct) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = torch.cat((direct.hand_qpos, direct.pen_pos, direct.pen_quat), dim=-1)
    qvel = torch.cat((direct.hand_qvel, direct.pen_linvel, direct.pen_angvel), dim=-1)
    return qpos, qvel


def _append_selected(
    rows: dict[str, list[np.ndarray]],
    best: dict[str, torch.Tensor],
    env_ids: torch.Tensor,
    success_flags: torch.Tensor,
) -> None:
    if env_ids.numel() == 0:
        return
    rows["source_qpos"].append(best["qpos"][env_ids].detach().cpu().numpy().copy())
    rows["source_qvel"].append(best["qvel"][env_ids].detach().cpu().numpy().copy())
    rows["source_target_qpos"].append(best["target"][env_ids].detach().cpu().numpy().copy())
    rows["source_policy_action"].append(best["action"][env_ids].detach().cpu().numpy().copy())
    rows["source_teacher_actions"].append(
        best["action"][env_ids].detach().cpu().numpy()[:, None, :].copy()
    )
    rows["target_qpos"].append(best["qpos"][env_ids].detach().cpu().numpy().copy())
    rows["target_qvel"].append(best["qvel"][env_ids].detach().cpu().numpy().copy())
    rows["phase_ids"].append(np.full(env_ids.numel(), 2, dtype=np.int64))
    rows["source_global_bank_ids"].append(best["bank_id"][env_ids].detach().cpu().numpy().copy())
    rows["policy_success_episode"].append(success_flags.detach().cpu().numpy().astype(np.bool_))
    for key in (
        "target_distance",
        "progress",
        "hold_steps",
        "penetration",
        "position_drift",
        "axis_drift",
        "angular_speed",
        "linear_speed",
        "eligible",
        "first_target_entry_radial_speed",
        "first_target_entry_pen_angular_speed",
        "first_target_entry_TIM",
        "first_target_entry_axis_drift",
        "first_exit_reason",
        "target_reentry_count",
        "longest_consecutive_target_steps",
        "route_step",
        "radial_speed",
        "manifold_under",
        "manifold_semantic",
        "manifold_semantic_soft",
    ):
        rows[key].append(best[key][env_ids].detach().cpu().numpy().copy())
    rows["ring_feature_m"].append(
        best["ring_feature"][env_ids].detach().cpu().numpy().copy()
    )
    rows["contact_force_thumb_index_middle_ring_pinky"].append(
        best["contact_force"][env_ids].detach().cpu().numpy().copy()
    )


def _bucket_from_hold(hold_steps: np.ndarray, success: np.ndarray, strict_hold_steps: int) -> np.ndarray:
    buckets = np.empty(hold_steps.shape[0], dtype="<U2")
    buckets[:] = "A0"
    buckets[hold_steps >= 1] = "A1"
    buckets[hold_steps >= 2] = "A2"
    buckets[hold_steps >= 3] = "A3"
    buckets[hold_steps >= 5] = "A4"
    buckets[(hold_steps >= strict_hold_steps) & success] = "A5"
    return buckets


def _exit_reasons(direct, metrics: dict[str, torch.Tensor], target_distance: torch.Tensor, strict_m: float) -> torch.Tensor:
    contacts = direct.primary_contact_grace > 0
    axis_limit = torch.deg2rad(torch.full_like(metrics["axis_drift"], direct.cfg.p2_max_axis_drift_deg))
    force_sum = direct.contact_force_norm.sum(dim=-1).clamp_min(1.0e-6)
    ring_force_share = direct.contact_force_norm[:, 3] / force_sum
    reasons = torch.full((direct.num_envs,), 8, dtype=torch.long, device=direct.device)
    reasons = torch.where(target_distance > strict_m, torch.zeros_like(reasons), reasons)
    reasons = torch.where(
        (reasons == 8) & ~contacts[:, 0], torch.full_like(reasons, 1), reasons
    )
    reasons = torch.where(
        (reasons == 8) & ~contacts[:, 1], torch.full_like(reasons, 2), reasons
    )
    reasons = torch.where(
        (reasons == 8) & ~contacts[:, 2], torch.full_like(reasons, 3), reasons
    )
    reasons = torch.where(
        (reasons == 8) & (metrics["axis_drift"] > axis_limit), torch.full_like(reasons, 4), reasons
    )
    reasons = torch.where(
        (reasons == 8) & (metrics["position_drift"] > direct.cfg.p2_max_position_drift_m),
        torch.full_like(reasons, 5),
        reasons,
    )
    speed_bad = (
        (metrics["angular_speed"] > direct.cfg.p2_max_angular_speed_rad_s)
        | (metrics["linear_speed"] > direct.cfg.p2_max_linear_speed_m_s)
    )
    reasons = torch.where((reasons == 8) & speed_bad, torch.full_like(reasons, 6), reasons)
    reasons = torch.where(
        (reasons == 8) & (direct.episode_max_penetration > direct.cfg.p2_max_penetration_m),
        torch.full_like(reasons, 7),
        reasons,
    )
    ring_or_pinky_bad = (
        (ring_force_share > direct.cfg.p2_max_ring_force_share)
        | (direct.contact_force_norm[:, 3] > direct.cfg.p2_max_ring_force_n)
        | (direct.contact_force_norm[:, 4] > direct.cfg.p2_max_pinky_force_n)
    )
    reasons = torch.where((reasons == 8) & ring_or_pinky_bad, torch.full_like(reasons, 9), reasons)
    reasons = torch.where((reasons == 8) & (metrics["failed"] > 0.0), torch.full_like(reasons, 10), reasons)
    return reasons


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    if args_cli.split != "train":
        raise ValueError("Dynamic state export is restricted to the train split")
    if args_cli.split_ids_path is not None:
        split_path = args_cli.split_ids_path.resolve()
    elif env_cfg.split_ids_path:
        split_path = Path(env_cfg.split_ids_path).resolve()
    else:
        split_path = SPLIT_PATHS[args_cli.split]
    checkpoint = Path(args_cli.checkpoint).resolve()
    output = Path(args_cli.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if not env_cfg.use_all_bundle_rows:
        env_cfg.split_ids_path = str(split_path)
    env_cfg.split_name = args_cli.split
    agent_cfg["seed"] = args_cli.seed

    bundle_path = Path(env_cfg.bundle_path).resolve()
    with np.load(bundle_path) as bank:
        phase_ids = np.asarray(bank["phase_ids"], dtype=np.int64)
        global_ids = (
            np.asarray(bank["source_global_bank_ids"], dtype=np.int64)
            if "source_global_bank_ids" in bank
            else np.arange(phase_ids.size, dtype=np.int64)
        )
    split_ids = np.asarray(np.load(split_path, allow_pickle=False), dtype=np.int64)
    in_bundle_split_ids = split_ids[(split_ids >= 0) & (split_ids < phase_ids.size)]
    if in_bundle_split_ids.size:
        allowed_source_ids = set(
            global_ids[
                in_bundle_split_ids[
                    phase_ids[in_bundle_split_ids] == int(env_cfg.source_phase_id)
                ]
            ].tolist()
        )
    else:
        # Rebuilt train-only canonical bundles may retain original global IDs while
        # containing only the selected rows locally.
        allowed_source_ids = set(
            global_ids[phase_ids == int(env_cfg.source_phase_id)].tolist()
        )

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

    observations, _ = wrapped.reset()
    states = wrapped.state()
    best = _empty_state(direct.num_envs, direct.device)
    episode_bank_ids = direct.current_global_bank_ids.clone()
    previous_target_distance = torch.full((direct.num_envs,), float("inf"), device=direct.device)
    previous_in_target = torch.zeros(direct.num_envs, dtype=torch.bool, device=direct.device)
    current_target_run = torch.zeros(direct.num_envs, dtype=torch.long, device=direct.device)
    arrival_latched = torch.zeros(direct.num_envs, dtype=torch.bool, device=direct.device)
    first_entry_recorded = torch.zeros(direct.num_envs, dtype=torch.bool, device=direct.device)
    first_entry_radial_speed = torch.full((direct.num_envs,), float("nan"), device=direct.device)
    first_entry_angular_speed = torch.full((direct.num_envs,), float("nan"), device=direct.device)
    first_entry_tim = torch.zeros(direct.num_envs, dtype=torch.bool, device=direct.device)
    first_entry_axis_drift = torch.full((direct.num_envs,), float("nan"), device=direct.device)
    first_exit_reason = torch.full((direct.num_envs,), -1, dtype=torch.long, device=direct.device)
    target_reentry_count = torch.zeros(direct.num_envs, dtype=torch.long, device=direct.device)
    longest_target_run = torch.zeros(direct.num_envs, dtype=torch.long, device=direct.device)
    rows: dict[str, list[np.ndarray]] = {
        "source_qpos": [],
        "source_qvel": [],
        "source_target_qpos": [],
        "source_policy_action": [],
        "source_teacher_actions": [],
        "target_qpos": [],
        "target_qvel": [],
        "phase_ids": [],
        "source_global_bank_ids": [],
        "policy_success_episode": [],
        "target_distance": [],
        "progress": [],
        "hold_steps": [],
        "penetration": [],
        "position_drift": [],
        "axis_drift": [],
        "angular_speed": [],
        "linear_speed": [],
        "eligible": [],
        "contact_force_thumb_index_middle_ring_pinky": [],
        "first_target_entry_radial_speed": [],
        "first_target_entry_pen_angular_speed": [],
        "first_target_entry_TIM": [],
        "first_target_entry_axis_drift": [],
        "first_exit_reason": [],
        "target_reentry_count": [],
        "longest_consecutive_target_steps": [],
        "route_step": [],
        "radial_speed": [],
        "manifold_under": [],
        "manifold_semantic": [],
        "manifold_semantic_soft": [],
        "ring_feature_m": [],
    }
    completed = 0
    accepted = 0
    steps = 0
    near_m = args_cli.near_distance_mm / 1000.0
    min_m = args_cli.min_distance_mm / 1000.0
    strict_m = args_cli.strict_distance_mm / 1000.0

    while completed < args_cli.episodes and steps < args_cli.max_steps and accepted < args_cli.max_rows:
        with torch.inference_mode():
            direct._compute_intermediate_values()
            eligible, metrics = direct._instantaneous_gate()
            _, progress, target_distance, _ = direct._ring_geometry()
            qpos, qvel = _current_qpos_qvel(direct)
            hold = direct.p1_p2_hold_count.clone()
            dt = float(direct.cfg.decimation * direct.cfg.sim.dt)
            in_target = target_distance <= strict_m
            target_entry = in_target & ~previous_in_target
            first_entry = target_entry & ~first_entry_recorded
            radial_speed = (previous_target_distance - target_distance) / max(dt, 1.0e-6)
            radial_speed = torch.where(
                torch.isfinite(previous_target_distance),
                radial_speed,
                torch.full_like(radial_speed, float("nan")),
            )
            first_entry_radial_speed = torch.where(
                first_entry, radial_speed, first_entry_radial_speed
            )
            first_entry_angular_speed = torch.where(
                first_entry, metrics["angular_speed"], first_entry_angular_speed
            )
            first_entry_tim = torch.where(first_entry, metrics["triad"] > 0.5, first_entry_tim)
            first_entry_axis_drift = torch.where(
                first_entry, metrics["axis_drift"], first_entry_axis_drift
            )
            first_entry_recorded |= first_entry
            target_reentry_count += (target_entry & arrival_latched).long()
            arrival_latched |= in_target
            current_target_run = torch.where(
                in_target, current_target_run + 1, torch.zeros_like(current_target_run)
            )
            longest_target_run = torch.maximum(longest_target_run, current_target_run)
            first_exit = previous_in_target & ~eligible & (first_exit_reason < 0)
            reasons = _exit_reasons(direct, metrics, target_distance, strict_m)
            first_exit_reason = torch.where(first_exit, reasons, first_exit_reason)
            previous_in_target = in_target
            previous_target_distance = target_distance
            near = (target_distance >= min_m) & (target_distance <= near_m)
            semantic_failure = (metrics["manifold_under"] < 0.5) | (
                metrics["manifold_semantic"] < 0.5
            )
            if args_cli.require_semantic_failure:
                near &= semantic_failure
            score = (
                1000.0 * hold.float()
                - 100.0 * target_distance
                - 10.0 * metrics["axis_drift"]
                - metrics["angular_speed"]
            )
            if args_cli.first_band_entry:
                improve = near & ~torch.isfinite(best["score"])
            else:
                improve = near & (score > best["score"])
            if improve.any():
                ids = torch.nonzero(improve, as_tuple=False).squeeze(-1)
                best["qpos"][ids] = qpos[ids]
                best["qvel"][ids] = qvel[ids]
                best["target"][ids] = direct.policy_targets[ids]
                best["action"][ids] = direct.actions[ids]
                best["bank_id"][ids] = episode_bank_ids[ids]
                best["score"][ids] = score[ids]
                best["target_distance"][ids] = target_distance[ids]
                best["progress"][ids] = progress[ids]
                best["hold_steps"][ids] = hold[ids]
                best["penetration"][ids] = direct.episode_max_penetration[ids]
                best["position_drift"][ids] = metrics["position_drift"][ids]
                best["axis_drift"][ids] = metrics["axis_drift"][ids]
                best["angular_speed"][ids] = metrics["angular_speed"][ids]
                best["linear_speed"][ids] = metrics["linear_speed"][ids]
                best["contact_force"][ids] = direct.contact_force_norm[ids]
                best["eligible"][ids] = eligible[ids]
                best["route_step"][ids] = direct.episode_length_buf[ids]
                best["radial_speed"][ids] = radial_speed[ids]
                best["manifold_under"][ids] = metrics["manifold_under"][ids] > 0.5
                best["manifold_semantic"][ids] = metrics["manifold_semantic"][ids] > 0.5
                best["manifold_semantic_soft"][ids] = metrics[
                    "manifold_semantic_soft"
                ][ids]
                best["ring_feature"][ids] = direct._ring_axis_features()[ids]
                best["first_target_entry_radial_speed"][ids] = first_entry_radial_speed[ids]
                best["first_target_entry_pen_angular_speed"][ids] = first_entry_angular_speed[ids]
                best["first_target_entry_TIM"][ids] = first_entry_tim[ids]
                best["first_target_entry_axis_drift"][ids] = first_entry_axis_drift[ids]
                best["first_exit_reason"][ids] = first_exit_reason[ids]
                best["target_reentry_count"][ids] = target_reentry_count[ids]

            outputs = runner.agent.act(observations, states, timestep=steps, timesteps=args_cli.max_steps)
            actions = outputs[-1].get("mean_actions", outputs[0])
            if args_cli.action_noise_std > 0.0:
                actions = torch.clamp(
                    actions + float(args_cli.action_noise_std) * torch.randn_like(actions),
                    -1.0,
                    1.0,
                )
            observations, _, terminated, truncated, _ = wrapped.step(actions)
            states = wrapped.state()
            done = (terminated | truncated).reshape(-1)
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            if done_ids.numel():
                success_flags = direct.last_completed_p1_p2_success[done_ids]
                strict = (
                    (best["target_distance"][done_ids] <= strict_m)
                    & (best["hold_steps"][done_ids] >= int(args_cli.strict_hold_steps))
                    & success_flags
                )
                near_hold = (
                    (best["target_distance"][done_ids] <= near_m)
                    & (best["hold_steps"][done_ids] >= int(args_cli.min_hold_steps))
                    & (best["penetration"][done_ids] <= direct.cfg.p2_max_penetration_m)
                )
                best["first_target_entry_radial_speed"][done_ids] = first_entry_radial_speed[done_ids]
                best["first_target_entry_pen_angular_speed"][done_ids] = first_entry_angular_speed[done_ids]
                best["first_target_entry_TIM"][done_ids] = first_entry_tim[done_ids]
                best["first_target_entry_axis_drift"][done_ids] = first_entry_axis_drift[done_ids]
                best["first_exit_reason"][done_ids] = first_exit_reason[done_ids]
                best["target_reentry_count"][done_ids] = target_reentry_count[done_ids]
                best["longest_consecutive_target_steps"][done_ids] = longest_target_run[done_ids]
                selected = done_ids[strict | near_hold]
                if selected.numel():
                    room = max(args_cli.max_rows - accepted, 0)
                    selected = selected[:room]
                    selected_success = direct.last_completed_p1_p2_success[selected]
                    _append_selected(rows, best, selected, selected_success)
                    accepted += int(selected.numel())
                completed += int(done_ids.numel())
                episode_bank_ids[done_ids] = direct.current_global_bank_ids[done_ids]
                _reset_best(best, done_ids)
                previous_target_distance[done_ids] = float("inf")
                previous_in_target[done_ids] = False
                current_target_run[done_ids] = 0
                arrival_latched[done_ids] = False
                first_entry_recorded[done_ids] = False
                first_entry_radial_speed[done_ids] = float("nan")
                first_entry_angular_speed[done_ids] = float("nan")
                first_entry_tim[done_ids] = False
                first_entry_axis_drift[done_ids] = float("nan")
                first_exit_reason[done_ids] = -1
                target_reentry_count[done_ids] = 0
                longest_target_run[done_ids] = 0
        steps += 1

    payload: dict[str, np.ndarray] = {}
    for key, chunks in rows.items():
        if chunks:
            payload[key] = np.concatenate(chunks, axis=0)
        else:
            payload[key] = np.empty((0,), dtype=np.float32)
    count = int(payload["phase_ids"].shape[0]) if payload["phase_ids"].ndim else 0
    if count:
        leaked_ids = sorted(set(payload["source_global_bank_ids"].astype(np.int64).tolist()) - allowed_source_ids)
        payload["target_distance_mm"] = payload.pop("target_distance").astype(np.float32) * 1000.0
        payload["max_penetration_mm"] = payload.pop("penetration").astype(np.float32) * 1000.0
        payload["position_drift_mm"] = payload.pop("position_drift").astype(np.float32) * 1000.0
        payload["axis_drift_deg"] = payload.pop("axis_drift").astype(np.float32) * (180.0 / np.pi)
        payload["first_target_entry_radial_speed_m_s"] = payload.pop(
            "first_target_entry_radial_speed"
        ).astype(np.float32)
        payload["first_target_entry_pen_angular_speed_rad_s"] = payload.pop(
            "first_target_entry_pen_angular_speed"
        ).astype(np.float32)
        payload["first_target_entry_axis_drift_deg"] = payload.pop(
            "first_target_entry_axis_drift"
        ).astype(np.float32) * (180.0 / np.pi)
        payload["radial_speed_m_s"] = payload.pop("radial_speed").astype(np.float32)
        payload["arrival_bucket"] = _bucket_from_hold(
            payload["hold_steps"].astype(np.int64),
            payload["policy_success_episode"].astype(bool),
            int(args_cli.strict_hold_steps),
        )
    else:
        leaked_ids = []

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    bucket_counts = {}
    if count:
        unique_buckets, unique_counts = np.unique(payload["arrival_bucket"], return_counts=True)
        bucket_counts = {
            str(bucket): int(bucket_count)
            for bucket, bucket_count in zip(unique_buckets, unique_counts, strict=False)
        }
    exit_reason_labels = {
        -1: "no_exit_recorded",
        0: "geometry_exit",
        1: "thumb_lost",
        2: "index_lost",
        3: "middle_lost",
        4: "axis_drift",
        5: "position_drift",
        6: "speed_gate",
        7: "penetration",
        8: "unknown_gate",
        9: "ring_or_pinky_load",
        10: "drop_or_env_failure",
    }
    exit_reason_counts = {}
    if count:
        reason_ids, reason_counts = np.unique(payload["first_exit_reason"].astype(np.int64), return_counts=True)
        exit_reason_counts = {
            exit_reason_labels.get(int(reason), f"reason_{int(reason)}"): int(reason_count)
            for reason, reason_count in zip(reason_ids, reason_counts, strict=False)
        }
    summary = {
        "task": args_cli.task,
        "split": args_cli.split,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "frozen_bank": str(env_cfg.bundle_path),
        "frozen_bank_sha256": sha256(Path(env_cfg.bundle_path)),
        "split_ids": str(split_path),
        "split_ids_sha256": sha256(split_path),
        "output": str(output),
        "output_sha256": sha256(output),
        "completed_episodes": completed,
        "accepted_rows": count,
        "policy_success_rows": int(payload.get("policy_success_episode", np.zeros(0, dtype=bool)).sum()),
        "unique_source_bank_ids": int(np.unique(payload["source_global_bank_ids"]).size) if count else 0,
        "arrival_bucket_counts": bucket_counts,
        "first_exit_reason_counts": exit_reason_counts,
        "leaked_id_count": len(leaked_ids),
        "leaked_ids": leaked_ids[:20],
        "min_hold_steps": args_cli.min_hold_steps,
        "strict_hold_steps": args_cli.strict_hold_steps,
        "near_distance_mm": args_cli.near_distance_mm,
        "min_distance_mm": args_cli.min_distance_mm,
        "first_band_entry": args_cli.first_band_entry,
        "action_noise_std": args_cli.action_noise_std,
        "require_semantic_failure": args_cli.require_semantic_failure,
        "strict_distance_mm": args_cli.strict_distance_mm,
        "simulation_steps": steps,
        "note": "Train-only dynamic near-P2/P2 states collected through normal PhysX policy rollouts.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_DYNAMIC_STATES=" + json.dumps(summary, ensure_ascii=False), flush=True)
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
