from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Collect strict successful P1-to-P2 scripted trajectories.")
parser.add_argument("--task", default="Dex5-P1P2-C1-Direct-v0")
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--trajectories", type=int, default=1000)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--seed", type=int, default=20260722)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def scripted_actions(env) -> torch.Tensor:
    actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    increment = 0.25 * env.cfg.action_scale
    ring_source = env.source_targets[:, 8:12]
    ring_final = env.p2_target_ring_qpos.unsqueeze(0).expand_as(ring_source)
    fraction = float(env.cfg.stage_target_progress[int(env.cfg.curriculum_stage)])
    ring_target = ring_source + fraction * (ring_final - ring_source)
    actions[:, 8:12] = torch.clamp(
        (ring_target - env.policy_targets[:, 8:12]) / increment,
        -1.0,
        1.0,
    )
    return actions


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if cfg.split_name != "train":
        raise ValueError("Scripted teacher generation is restricted to the train split")
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    obs, _ = env.reset()
    horizon = direct.max_episode_length
    obs_dim = obs["policy"].shape[-1]
    obs_buffer = torch.zeros((args.num_envs, horizon, obs_dim), device=direct.device)
    action_buffer = torch.zeros(
        (args.num_envs, horizon, direct.cfg.action_space), device=direct.device
    )
    episode_bank_ids = direct.current_global_bank_ids.clone()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    lengths: list[int] = []
    bank_ids: list[int] = []
    steps = 0

    with np.load(cfg.bundle_path) as bank:
        phase_ids = np.asarray(bank["phase_ids"], dtype=np.int64)
    train_ids = np.asarray(np.load(cfg.split_ids_path, allow_pickle=False), dtype=np.int64)
    allowed_p1_ids = set(train_ids[phase_ids[train_ids] == 1].tolist())

    while len(observations) < args.trajectories and steps < args.max_steps:
        indices = direct.episode_length_buf.long().clamp_max(horizon - 1)
        teacher_action = scripted_actions(direct)
        rows = torch.arange(args.num_envs, device=direct.device)
        obs_buffer[rows, indices] = obs["policy"]
        action_buffer[rows, indices] = teacher_action
        obs, _, terminated, truncated, _ = env.step(teacher_action)
        done = terminated | truncated
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        for env_id in done_ids.tolist():
            if bool(direct.last_completed_p1_p2_success[env_id]):
                length = int(indices[env_id].item()) + 1
                observations.append(obs_buffer[env_id, :length].cpu().numpy().copy())
                actions.append(action_buffer[env_id, :length].cpu().numpy().copy())
                lengths.append(length)
                bank_ids.append(int(episode_bank_ids[env_id].item()))
                if len(observations) >= args.trajectories:
                    break
        if done_ids.numel() > 0:
            obs_buffer[done_ids] = 0.0
            action_buffer[done_ids] = 0.0
            episode_bank_ids[done_ids] = direct.current_global_bank_ids[done_ids]
        steps += 1

    if len(observations) < args.trajectories:
        raise RuntimeError(
            f"Collected only {len(observations)}/{args.trajectories} strict successes "
            f"after {steps} simulation steps"
        )
    leaked = sorted(set(bank_ids) - allowed_p1_ids)
    if leaked:
        raise RuntimeError(f"Non-train or non-P1 IDs entered teacher data: {leaked[:20]}")

    max_length = max(lengths)
    obs_array = np.zeros((len(observations), max_length, obs_dim), dtype=np.float32)
    action_array = np.zeros(
        (len(actions), max_length, direct.cfg.action_space), dtype=np.float32
    )
    for row, (obs_row, action_row) in enumerate(zip(observations, actions, strict=True)):
        obs_array[row, : len(obs_row)] = obs_row
        action_array[row, : len(action_row)] = action_row
    length_array = np.asarray(lengths, dtype=np.int64)
    stage_array = np.full(
        (len(observations), max_length), int(direct.cfg.curriculum_stage), dtype=np.int64
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observation=obs_array,
        action=action_array,
        stage_id=stage_array,
        lengths=length_array,
        global_bank_ids=np.asarray(bank_ids, dtype=np.int64),
    )
    summary = {
        "task": args.task,
        "output": str(output),
        "trajectories": len(observations),
        "transitions": int(length_array.sum()),
        "length_min": int(length_array.min()),
        "length_mean": float(length_array.mean()),
        "length_max": int(length_array.max()),
        "unique_train_p1_ids": len(set(bank_ids)),
        "leaked_id_count": 0,
        "simulation_steps": steps,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_TEACHER=" + json.dumps(summary, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
