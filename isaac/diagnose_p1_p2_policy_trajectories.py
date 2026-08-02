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
    description="Export per-episode diagnostics for strict P1/P2 policy rollouts."
)
parser.add_argument("--task", default="Dex5-P1P2-ReverseR0ValidatedHoldSlow-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--split", choices=tuple(SPLIT_PATHS), default="train")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--max_steps", type=int, default=800)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--reset_joint_pos_noise", type=float, default=None)
parser.add_argument("--reset_joint_vel_noise", type=float, default=None)
parser.add_argument("--reset_pen_pos_noise", type=float, default=None)
parser.add_argument("--reset_pen_vel_noise", type=float, default=None)
parser.add_argument("--reset_pen_rot_noise_rad", type=float, default=None)
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


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    split_path = SPLIT_PATHS[args_cli.split]
    checkpoint = Path(args_cli.checkpoint).resolve()
    output = Path(args_cli.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.split_ids_path = str(split_path)
    env_cfg.split_name = args_cli.split
    for name in (
        "reset_joint_pos_noise",
        "reset_joint_vel_noise",
        "reset_pen_pos_noise",
        "reset_pen_vel_noise",
        "reset_pen_rot_noise_rad",
    ):
        value = getattr(args_cli, name)
        if value is not None:
            setattr(env_cfg, name, value)
    agent_cfg["seed"] = args_cli.seed

    with np.load(RESET_DIR / "P1_P2_reset_bank_phase1最终validated自复验_validated.npz") as bank:
        phase_ids = np.asarray(bank["phase_ids"], dtype=np.int64)
    split_ids = np.asarray(np.load(split_path, allow_pickle=False), dtype=np.int64)
    allowed_source_ids = set(
        split_ids[phase_ids[split_ids] == int(env_cfg.source_phase_id)].tolist()
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
    device = direct.device
    num_envs = direct.num_envs
    target_radius = float(env_cfg.stage_target_distance_m[int(env_cfg.curriculum_stage)])
    thresholds = torch.tensor((0.008, 0.005, target_radius), device=device)

    episode_bank_ids = direct.current_global_bank_ids.clone()
    min_target_distance = torch.full((num_envs,), float("inf"), device=device)
    max_progress = torch.full((num_envs,), -float("inf"), device=device)
    max_hold = torch.zeros((num_envs,), dtype=torch.long, device=device)
    max_penetration = torch.zeros((num_envs,), device=device)
    max_axis_drift = torch.zeros((num_envs,), device=device)
    max_position_drift = torch.zeros((num_envs,), device=device)
    min_angular_speed_in_5mm = torch.full((num_envs,), float("inf"), device=device)
    min_linear_speed_in_5mm = torch.full((num_envs,), float("inf"), device=device)
    near_counts = torch.zeros((num_envs, 3), dtype=torch.long, device=device)
    near_triad_counts = torch.zeros_like(near_counts)
    near_middle_counts = torch.zeros_like(near_counts)
    near_slow_counts = torch.zeros_like(near_counts)
    steps_alive = torch.zeros((num_envs,), dtype=torch.long, device=device)

    rows: list[dict[str, float | int | bool]] = []
    completed_total = 0
    simulation_steps = 0
    while completed_total < args_cli.episodes and simulation_steps < args_cli.max_steps:
        with torch.inference_mode():
            direct._compute_intermediate_values()
            _, metrics = direct._instantaneous_gate()
            _, progress, target_distance, _ = direct._ring_geometry()
            primary_contacts = direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n
            triad = primary_contacts.all(dim=-1)
            middle = primary_contacts[:, 2]
            slow = (
                (metrics["angular_speed"] <= direct.cfg.p2_max_angular_speed_rad_s)
                & (metrics["linear_speed"] <= direct.cfg.p2_max_linear_speed_m_s)
            )
            near = target_distance.unsqueeze(-1) <= thresholds.unsqueeze(0)
            min_target_distance = torch.minimum(min_target_distance, target_distance)
            max_progress = torch.maximum(max_progress, progress)
            max_hold = torch.maximum(max_hold, direct.p1_p2_hold_count)
            max_penetration = torch.maximum(max_penetration, direct.episode_max_penetration)
            max_axis_drift = torch.maximum(max_axis_drift, metrics["axis_drift"])
            max_position_drift = torch.maximum(max_position_drift, metrics["position_drift"])
            near_counts += near.long()
            near_triad_counts += (near & triad.unsqueeze(-1)).long()
            near_middle_counts += (near & middle.unsqueeze(-1)).long()
            near_slow_counts += (near & slow.unsqueeze(-1)).long()
            in_5mm = near[:, 1]
            min_angular_speed_in_5mm = torch.where(
                in_5mm,
                torch.minimum(min_angular_speed_in_5mm, metrics["angular_speed"]),
                min_angular_speed_in_5mm,
            )
            min_linear_speed_in_5mm = torch.where(
                in_5mm,
                torch.minimum(min_linear_speed_in_5mm, metrics["linear_speed"]),
                min_linear_speed_in_5mm,
            )
            steps_alive += 1

            outputs = runner.agent.act(
                observations, states, timestep=simulation_steps, timesteps=args_cli.max_steps
            )
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, terminated, truncated, _ = wrapped.step(actions)
            states = wrapped.state()
            done = (terminated | truncated).reshape(-1)
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)

            for env_id in done_ids.tolist():
                if completed_total >= args_cli.episodes:
                    break
                bank_id = int(episode_bank_ids[env_id].item())
                rows.append(
                    {
                        "episode": completed_total,
                        "global_bank_id": bank_id,
                        "success": bool(direct.last_completed_p1_p2_success[env_id].item()),
                        "drop": bool(direct.last_completed_p1_p2_drop[env_id].item()),
                        "steps_alive": int(steps_alive[env_id].item()),
                        "min_target_distance_mm": float(
                            min_target_distance[env_id].item() * 1000.0
                        ),
                        "max_progress": float(max_progress[env_id].item()),
                        "max_hold_steps": int(max_hold[env_id].item()),
                        "max_hold_ms": float(
                            max_hold[env_id].item()
                            * direct.cfg.decimation
                            * direct.cfg.sim.dt
                            * 1000.0
                        ),
                        "max_penetration_mm": float(max_penetration[env_id].item() * 1000.0),
                        "max_axis_drift_deg": float(
                            max_axis_drift[env_id].item() * 180.0 / torch.pi
                        ),
                        "max_position_drift_mm": float(
                            max_position_drift[env_id].item() * 1000.0
                        ),
                        "steps_within_8mm": int(near_counts[env_id, 0].item()),
                        "steps_within_5mm": int(near_counts[env_id, 1].item()),
                        "steps_within_target": int(near_counts[env_id, 2].item()),
                        "triad_steps_within_8mm": int(near_triad_counts[env_id, 0].item()),
                        "triad_steps_within_5mm": int(near_triad_counts[env_id, 1].item()),
                        "triad_steps_within_target": int(near_triad_counts[env_id, 2].item()),
                        "middle_steps_within_5mm": int(near_middle_counts[env_id, 1].item()),
                        "slow_steps_within_5mm": int(near_slow_counts[env_id, 1].item()),
                        "min_angular_speed_in_5mm_rad_s": float(
                            min_angular_speed_in_5mm[env_id].item()
                        ),
                        "min_linear_speed_in_5mm_m_s": float(
                            min_linear_speed_in_5mm[env_id].item()
                        ),
                    }
                )
                completed_total += 1

            if done_ids.numel() > 0:
                episode_bank_ids[done_ids] = direct.current_global_bank_ids[done_ids]
                min_target_distance[done_ids] = float("inf")
                max_progress[done_ids] = -float("inf")
                max_hold[done_ids] = 0
                max_penetration[done_ids] = 0.0
                max_axis_drift[done_ids] = 0.0
                max_position_drift[done_ids] = 0.0
                min_angular_speed_in_5mm[done_ids] = float("inf")
                min_linear_speed_in_5mm[done_ids] = float("inf")
                near_counts[done_ids] = 0
                near_triad_counts[done_ids] = 0
                near_middle_counts[done_ids] = 0
                near_slow_counts[done_ids] = 0
                steps_alive[done_ids] = 0
        simulation_steps += 1

    leaked_ids = sorted({int(row["global_bank_id"]) for row in rows} - allowed_source_ids)
    success_rows = [row for row in rows if bool(row["success"])]
    near_8mm_rows = [row for row in rows if float(row["min_target_distance_mm"]) <= 8.0]
    near_5mm_rows = [row for row in rows if float(row["min_target_distance_mm"]) <= 5.0]
    near_target_rows = [
        row for row in rows if float(row["min_target_distance_mm"]) <= target_radius * 1000.0
    ]
    min_distances = np.asarray([row["min_target_distance_mm"] for row in rows], dtype=np.float32)
    hold_ms = np.asarray([row["max_hold_ms"] for row in rows], dtype=np.float32)
    unique_success_ids = {int(row["global_bank_id"]) for row in success_rows}
    unique_near5_ids = {int(row["global_bank_id"]) for row in near_5mm_rows}
    summary = {
        "task": args_cli.task,
        "split": args_cli.split,
        "seed": args_cli.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "frozen_bank": env_cfg.bundle_path,
        "frozen_bank_sha256": sha256(Path(env_cfg.bundle_path)),
        "split_ids": str(split_path),
        "split_ids_sha256": sha256(split_path),
        "reset_noise": {
            "joint_pos": float(env_cfg.reset_joint_pos_noise),
            "joint_vel": float(env_cfg.reset_joint_vel_noise),
            "pen_pos": float(env_cfg.reset_pen_pos_noise),
            "pen_vel": float(env_cfg.reset_pen_vel_noise),
            "pen_rot_rad": float(env_cfg.reset_pen_rot_noise_rad),
        },
        "completed_episodes": len(rows),
        "simulation_steps": simulation_steps,
        "evaluation_complete": len(rows) >= args_cli.episodes,
        "leaked_id_count": len(leaked_ids),
        "leaked_ids": leaked_ids[:20],
        "successes": len(success_rows),
        "success_rate": len(success_rows) / max(len(rows), 1),
        "unique_success_bank_ids": len(unique_success_ids),
        "within_8mm_episodes": len(near_8mm_rows),
        "within_8mm_rate": len(near_8mm_rows) / max(len(rows), 1),
        "within_5mm_episodes": len(near_5mm_rows),
        "within_5mm_rate": len(near_5mm_rows) / max(len(rows), 1),
        "within_target_episodes": len(near_target_rows),
        "within_target_rate": len(near_target_rows) / max(len(rows), 1),
        "unique_near5_bank_ids": len(unique_near5_ids),
        "min_target_distance_mm_p10": percentile(min_distances, 10),
        "min_target_distance_mm_p50": percentile(min_distances, 50),
        "min_target_distance_mm_p90": percentile(min_distances, 90),
        "max_hold_ms_p50": percentile(hold_ms, 50),
        "max_hold_ms_p90": percentile(hold_ms, 90),
        "target_radius_mm": target_radius * 1000.0,
        "note": (
            "Per-episode rollout diagnostics. Near windows are based on minimum "
            "ring target distance reached during the episode; success still uses "
            "the environment's strict 100 ms gate."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rows=np.asarray(json.dumps(rows, ensure_ascii=False), dtype=np.str_),
        global_bank_id=np.asarray([row["global_bank_id"] for row in rows], dtype=np.int64),
        success=np.asarray([row["success"] for row in rows], dtype=np.bool_),
        drop=np.asarray([row["drop"] for row in rows], dtype=np.bool_),
        min_target_distance_mm=min_distances,
        max_progress=np.asarray([row["max_progress"] for row in rows], dtype=np.float32),
        max_hold_ms=hold_ms,
        steps_within_8mm=np.asarray([row["steps_within_8mm"] for row in rows], dtype=np.int64),
        steps_within_5mm=np.asarray([row["steps_within_5mm"] for row in rows], dtype=np.int64),
        steps_within_target=np.asarray(
            [row["steps_within_target"] for row in rows], dtype=np.int64
        ),
        triad_steps_within_5mm=np.asarray(
            [row["triad_steps_within_5mm"] for row in rows], dtype=np.int64
        ),
        middle_steps_within_5mm=np.asarray(
            [row["middle_steps_within_5mm"] for row in rows], dtype=np.int64
        ),
        slow_steps_within_5mm=np.asarray(
            [row["slow_steps_within_5mm"] for row in rows], dtype=np.int64
        ),
    )
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps({"summary": summary, "episodes": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("P1_P2_TRAJECTORY_DIAGNOSTIC=" + json.dumps(summary, ensure_ascii=False), flush=True)
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
