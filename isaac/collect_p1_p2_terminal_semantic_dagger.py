from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Collect terminal under/semantic failures from full P1-to-P2 rollouts."
)
parser.add_argument(
    "--task", default="Dex5-P1P2-P1617SingleRoute539Conditioned120HzH1-Direct-v0"
)
parser.add_argument("--local-teacher", required=True, type=Path)
parser.add_argument("--student", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--append-dataset", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--max-rows", type=int, default=60000)
parser.add_argument("--strict-distance-mm", type=float, default=2.45)
parser.add_argument("--excess-min-mm", type=float, default=0.1)
parser.add_argument("--excess-max-mm", type=float, default=1.5)
parser.add_argument("--disagreement-threshold", type=float, default=0.01)
parser.add_argument("--rollout-action-noise", type=float, default=0.015)
parser.add_argument("--terminal-phase", type=int, default=3)
parser.add_argument("--seed", type=int, default=617)
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
    local_teacher = make_agent(wrapped, agent_cfg, args_cli.local_teacher)
    student = make_agent(wrapped, agent_cfg, args_cli.student)

    observations, _ = wrapped.reset()
    states = wrapped.state()
    generator = torch.Generator(device=observations.device)
    generator.manual_seed(args_cli.seed)
    rows: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "observation",
            "action",
            "student_action",
            "disagreement",
            "target_excess_mm",
            "under_pass",
            "semantic_pass",
        )
    }
    collected = 0
    candidate_count = 0
    under_failure_count = 0
    semantic_failure_count = 0
    lower_m = (args_cli.strict_distance_mm + args_cli.excess_min_mm) / 1000.0
    upper_m = (args_cli.strict_distance_mm + args_cli.excess_max_mm) / 1000.0

    for step in range(args_cli.steps):
        with torch.inference_mode():
            teacher_out = local_teacher.act(
                observations, states, timestep=step, timesteps=args_cli.steps
            )
            student_out = student.act(
                observations, states, timestep=step, timesteps=args_cli.steps
            )
            teacher_actions = teacher_out[-1].get("mean_actions", teacher_out[0])
            student_actions = student_out[-1].get("mean_actions", student_out[0])
            filtered_teacher = torch.zeros_like(teacher_actions)
            filtered_teacher[:, ALLOWED] = teacher_actions[:, ALLOWED]
            disagreement = torch.sqrt(
                torch.mean(
                    (filtered_teacher[:, ALLOWED] - student_actions[:, ALLOWED]).square(),
                    dim=-1,
                )
            )

            direct._compute_intermediate_values()
            _, metrics = direct._instantaneous_gate()
            target_distance = metrics["target_distance"]
            under_pass = metrics["manifold_under"] > 0.5
            semantic_pass = metrics["manifold_semantic"] > 0.5
            in_terminal_band = (target_distance >= lower_m) & (target_distance <= upper_m)
            semantic_failure = (~under_pass) | (~semantic_pass)
            candidates = in_terminal_band & semantic_failure
            candidate_count += int(candidates.sum())
            under_failure_count += int((in_terminal_band & ~under_pass).sum())
            semantic_failure_count += int((in_terminal_band & ~semantic_pass).sum())

            if step % max(args_cli.stride, 1) == 0:
                selected = candidates & (
                    disagreement >= float(args_cli.disagreement_threshold)
                )
                ids = torch.nonzero(selected, as_tuple=False).squeeze(-1)
                ids = ids[: max(args_cli.max_rows - collected, 0)]
                if ids.numel():
                    rows["observation"].append(observations[ids].cpu().numpy())
                    rows["action"].append(filtered_teacher[ids].cpu().numpy())
                    rows["student_action"].append(student_actions[ids].cpu().numpy())
                    rows["disagreement"].append(disagreement[ids].cpu().numpy())
                    rows["target_excess_mm"].append(
                        (target_distance[ids] * 1000.0 - args_cli.strict_distance_mm)
                        .cpu()
                        .numpy()
                    )
                    rows["under_pass"].append(under_pass[ids].cpu().numpy())
                    rows["semantic_pass"].append(semantic_pass[ids].cpu().numpy())
                    collected += int(ids.numel())

            noise = torch.randn(
                student_actions.shape,
                generator=generator,
                device=student_actions.device,
                dtype=student_actions.dtype,
            )
            rollout_actions = torch.clamp(
                student_actions + float(args_cli.rollout_action_noise) * noise,
                -1.0,
                1.0,
            )
            observations, _, _, _, _ = wrapped.step(rollout_actions)
            states = wrapped.state()
            if collected >= args_cli.max_rows:
                break

    def merged(key: str, width: int | None = None, dtype=np.float32) -> np.ndarray:
        if rows[key]:
            return np.concatenate(rows[key]).astype(dtype, copy=False)
        shape = (0, width) if width is not None else (0,)
        return np.empty(shape, dtype=dtype)

    observation = merged("observation", observations.shape[-1])
    action = merged("action", 20)
    student_action = merged("student_action", 20)
    disagreement = merged("disagreement")
    target_excess_mm = merged("target_excess_mm")
    under_pass = merged("under_pass", dtype=np.bool_)
    semantic_pass = merged("semantic_pass", dtype=np.bool_)
    phase_kind = np.full(collected, args_cli.terminal_phase, dtype=np.int64)

    if args_cli.append_dataset is not None:
        with np.load(args_cli.append_dataset, allow_pickle=False) as previous:
            previous_obs = np.asarray(previous["observation"], dtype=np.float32).reshape(
                len(previous["observation"]), -1
            )
            previous_action = np.asarray(previous["action"], dtype=np.float32).reshape(-1, 20)
            previous_student = np.asarray(
                previous["student_action"], dtype=np.float32
            ).reshape(-1, 20)
            previous_phase = np.asarray(previous["phase_kind"], dtype=np.int64).reshape(-1)
            previous_disagreement = np.asarray(
                previous["disagreement"], dtype=np.float32
            ).reshape(-1)
        if previous_obs.shape[-1] > observation.shape[-1]:
            raise ValueError(
                f"Previous observations have {previous_obs.shape[-1]} columns, "
                f"current task has {observation.shape[-1]}"
            )
        if previous_obs.shape[-1] < observation.shape[-1]:
            padded = np.zeros(
                (len(previous_obs), observation.shape[-1]), dtype=np.float32
            )
            padded[:, : previous_obs.shape[-1]] = previous_obs
            previous_obs = padded
        previous_count = len(previous_obs)
        observation = np.concatenate((previous_obs, observation))
        action = np.concatenate((previous_action, action))
        student_action = np.concatenate((previous_student, student_action))
        phase_kind = np.concatenate((previous_phase, phase_kind))
        disagreement = np.concatenate((previous_disagreement, disagreement))
        target_excess_mm = np.concatenate(
            (np.full(previous_count, np.nan, np.float32), target_excess_mm)
        )
        under_pass = np.concatenate(
            (np.full(previous_count, True, np.bool_), under_pass)
        )
        semantic_pass = np.concatenate(
            (np.full(previous_count, True, np.bool_), semantic_pass)
        )

    count = len(observation)
    payload = {
        "observation": observation[:, None, :],
        "action": action[:, None, :],
        "student_action": student_action[:, None, :],
        "lengths": np.ones(count, dtype=np.int64),
        "phase_kind": phase_kind,
        "disagreement": disagreement,
        "target_excess_mm": target_excess_mm,
        "under_pass": under_pass,
        "semantic_pass": semantic_pass,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args_cli.output, **payload)
    phase_ids, phase_counts = np.unique(phase_kind, return_counts=True)
    summary = {
        "task": args_cli.task,
        "local_teacher": str(args_cli.local_teacher.resolve()),
        "local_teacher_sha256": sha256(args_cli.local_teacher),
        "student": str(args_cli.student.resolve()),
        "student_sha256": sha256(args_cli.student),
        "output": str(args_cli.output.resolve()),
        "output_sha256": sha256(args_cli.output),
        "rows": count,
        "new_rows": collected,
        "phase_counts": {
            str(key): int(value)
            for key, value in zip(phase_ids, phase_counts, strict=True)
        },
        "terminal_band_mm": [
            args_cli.strict_distance_mm + args_cli.excess_min_mm,
            args_cli.strict_distance_mm + args_cli.excess_max_mm,
        ],
        "candidate_observations": candidate_count,
        "under_failure_observations": under_failure_count,
        "semantic_failure_observations": semantic_failure_count,
        "new_disagreement_mean": (
            float(disagreement[-collected:].mean()) if collected else None
        ),
        "new_target_excess_mean_mm": (
            float(target_excess_mm[-collected:].mean()) if collected else None
        ),
        "pinky_teacher_max_abs": float(np.max(np.abs(action[:, 12:16]))) if count else 0.0,
        "rollout_action_noise": args_cli.rollout_action_noise,
    }
    args_cli.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_TERMINAL_DAGGER=" + json.dumps(summary, ensure_ascii=False), flush=True)
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
