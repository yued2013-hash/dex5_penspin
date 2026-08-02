from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect phase-conditioned P1/P2 DAgger labels.")
parser.add_argument("--task", default="Dex5-P1P2-P1617Route539Mixed120HzH1-Direct-v0")
parser.add_argument("--full-teacher", required=True, type=Path)
parser.add_argument("--local-teacher", required=True, type=Path)
parser.add_argument("--student", type=Path, default=None)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--append-dataset", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=320)
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--max-rows", type=int, default=120000)
parser.add_argument("--disagreement-threshold", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=2821)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from skrl.utils import set_seed as set_skrl_seed
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


ALLOWED = (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_agent(wrapped, agent_cfg: dict, checkpoint: Path):
    runner = Runner(wrapped, copy.deepcopy(agent_cfg))
    runner.agent.load(str(checkpoint.resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    return runner.agent


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    direct = env.unwrapped
    set_skrl_seed(args_cli.seed)
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    full_agent = make_agent(wrapped, agent_cfg, args_cli.full_teacher)
    local_agent = make_agent(wrapped, agent_cfg, args_cli.local_teacher)
    student_agent = (
        make_agent(wrapped, agent_cfg, args_cli.student)
        if args_cli.student is not None
        else None
    )

    observations, _ = wrapped.reset()
    states = wrapped.state()
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    student_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []
    disagreement_rows: list[np.ndarray] = []
    collected = 0

    for step in range(args_cli.steps):
        with torch.inference_mode():
            full_out = full_agent.act(observations, states, timestep=step, timesteps=args_cli.steps)
            local_out = local_agent.act(observations, states, timestep=step, timesteps=args_cli.steps)
            full_actions = full_out[-1].get("mean_actions", full_out[0])
            local_actions = local_out[-1].get("mean_actions", local_out[0])
            source_phase = direct.source_phase_id_bank[direct.current_bank_ids]
            teacher_actions = torch.where(
                (source_phase == 2).unsqueeze(-1), local_actions, full_actions
            )
            filtered_teacher = torch.zeros_like(teacher_actions)
            filtered_teacher[:, ALLOWED] = teacher_actions[:, ALLOWED]

            if student_agent is None:
                rollout_actions = filtered_teacher
                student_actions = filtered_teacher
            else:
                student_out = student_agent.act(
                    observations, states, timestep=step, timesteps=args_cli.steps
                )
                student_actions = student_out[-1].get("mean_actions", student_out[0])
                rollout_actions = student_actions

            disagreement = torch.sqrt(
                torch.mean((filtered_teacher[:, ALLOWED] - student_actions[:, ALLOWED]).square(), dim=-1)
            )
            if step % max(args_cli.stride, 1) == 0:
                selected = disagreement >= float(args_cli.disagreement_threshold)
                ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
                room = max(args_cli.max_rows - collected, 0)
                ids = ids[:room]
                if ids.numel():
                    obs_rows.append(observations[ids].detach().cpu().numpy())
                    action_rows.append(filtered_teacher[ids].detach().cpu().numpy())
                    student_rows.append(student_actions[ids].detach().cpu().numpy())
                    phase_rows.append(source_phase[ids].detach().cpu().numpy())
                    disagreement_rows.append(disagreement[ids].detach().cpu().numpy())
                    collected += int(ids.numel())
            observations, _, _, _, _ = wrapped.step(rollout_actions)
            states = wrapped.state()
            if collected >= args_cli.max_rows:
                break

    observation = np.concatenate(obs_rows) if obs_rows else np.empty((0, 186), np.float32)
    action = np.concatenate(action_rows) if action_rows else np.empty((0, 20), np.float32)
    student_action = np.concatenate(student_rows) if student_rows else np.empty((0, 20), np.float32)
    phase_kind = np.concatenate(phase_rows).astype(np.int64) if phase_rows else np.empty(0, np.int64)
    disagreement = np.concatenate(disagreement_rows) if disagreement_rows else np.empty(0, np.float32)

    if args_cli.append_dataset is not None:
        with np.load(args_cli.append_dataset, allow_pickle=False) as previous:
            previous_obs = np.asarray(previous["observation"], dtype=np.float32).reshape(-1, observation.shape[-1])
            previous_action = np.asarray(previous["action"], dtype=np.float32).reshape(-1, 20)
            previous_student = np.asarray(previous["student_action"], dtype=np.float32).reshape(-1, 20)
            previous_phase = np.asarray(previous["phase_kind"], dtype=np.int64).reshape(-1)
            previous_disagreement = np.asarray(previous["disagreement"], dtype=np.float32).reshape(-1)
        observation = np.concatenate((previous_obs, observation))
        action = np.concatenate((previous_action, action))
        student_action = np.concatenate((previous_student, student_action))
        phase_kind = np.concatenate((previous_phase, phase_kind))
        disagreement = np.concatenate((previous_disagreement, disagreement))

    count = observation.shape[0]
    payload = {
        "observation": observation[:, None, :],
        "action": action[:, None, :],
        "student_action": student_action[:, None, :],
        "lengths": np.ones(count, dtype=np.int64),
        "phase_kind": phase_kind,
        "disagreement": disagreement,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args_cli.output, **payload)
    phase_ids, phase_counts = np.unique(phase_kind, return_counts=True)
    summary = {
        "task": args_cli.task,
        "full_teacher": str(args_cli.full_teacher.resolve()),
        "full_teacher_sha256": sha256(args_cli.full_teacher),
        "local_teacher": str(args_cli.local_teacher.resolve()),
        "local_teacher_sha256": sha256(args_cli.local_teacher),
        "student": str(args_cli.student.resolve()) if args_cli.student else None,
        "output": str(args_cli.output.resolve()),
        "output_sha256": sha256(args_cli.output),
        "rows": count,
        "new_rows": collected,
        "phase_counts": {str(k): int(v) for k, v in zip(phase_ids, phase_counts, strict=True)},
        "disagreement_mean": float(disagreement.mean()) if count else None,
        "disagreement_p90": float(np.percentile(disagreement, 90)) if count else None,
        "pinky_teacher_max_abs": float(np.max(np.abs(action[:, 12:16]))) if count else 0.0,
        "rollout": "student" if student_agent is not None else "teacher",
    }
    args_cli.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_PHASE_DAGGER=" + json.dumps(summary, ensure_ascii=False), flush=True)
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
