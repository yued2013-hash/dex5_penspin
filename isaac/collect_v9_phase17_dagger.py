from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Collect phase16-to-17 teacher-assisted DAgger samples."
)
parser.add_argument(
    "--task",
    default=(
        "Dex5-V9P4-P5-P1-ContinuousClosure"
        "Phase17TeacherPrefill-Direct-v0"
    ),
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--teacher-npz",
    type=Path,
    help="Optional complete phase16-to-17 teacher trajectory bank.",
)
parser.add_argument(
    "--teacher-rollout-fraction",
    type=float,
    default=0.0,
    help="Fraction of expert action used during rollout; labels stay fully expert.",
)
parser.add_argument(
    "--composite-expert-checkpoint",
    type=Path,
    help=(
        "Residual PPO checkpoint used to label teacher + scaled residual "
        "actions in standalone dynamics. Requires --teacher-npz."
    ),
)
parser.add_argument(
    "--composite-residual-ratio",
    type=float,
    help=(
        "Raw-action residual ratio in the composite label. Defaults to "
        "scripted residual scale / scripted teacher action scale."
    ),
)
parser.add_argument(
    "--standalone_rollout",
    action="store_true",
    help="Disable teacher actuation while retaining teacher labels.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch import nn
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401


class CheckpointPolicy(nn.Module):
    """Deterministic skrl policy used as the residual expert."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state = checkpoint["policy"]
        self.net = nn.Sequential(
            nn.Linear(156, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 20),
        )
        for layer, key in (
            (0, "net_container.0"),
            (2, "net_container.2"),
            (4, "net_container.4"),
            (6, "policy_layer"),
        ):
            self.net[layer].weight.data.copy_(state[f"{key}.weight"])
            self.net[layer].bias.data.copy_(state[f"{key}.bias"])
        preprocessor = checkpoint["observation_preprocessor"]
        self.register_buffer(
            "mean", preprocessor["running_mean"].float()
        )
        self.register_buffer(
            "variance", preprocessor["running_variance"].float()
        )
        self.to(device)
        self.eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(
            (observation - self.mean)
            / (torch.sqrt(self.variance) + 1.0e-8),
            -5.0,
            5.0,
        )
        return self.net(normalized)


def scripted_action(direct) -> torch.Tensor:
    if bool(direct.cfg.scripted_thumb_teacher_pair_with_bundle):
        teacher_ids = (
            direct.current_bank_ids
            % direct.scripted_thumb_teacher_first.shape[0]
        )
    else:
        teacher_ids = (
            torch.arange(direct.num_envs, device=direct.device)
            % direct.scripted_thumb_teacher_first.shape[0]
        )
    first = direct.scripted_thumb_teacher_first[teacher_ids]
    second = direct.scripted_thumb_teacher_second[teacher_ids]
    first_duration = direct.scripted_thumb_teacher_first_duration[
        teacher_ids
    ]
    second_duration = direct.scripted_thumb_teacher_second_duration[
        teacher_ids
    ]
    age = direct.episode_length_buf
    use_first = age < first_duration
    use_second = (age < first_duration + second_duration) & ~use_first
    return torch.where(
        use_first.unsqueeze(-1),
        first,
        torch.where(
            use_second.unsqueeze(-1), second, torch.zeros_like(first)
        ),
    )


def trajectory_action(
    direct,
    action_bank: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    teacher_ids = (
        torch.arange(direct.num_envs, device=direct.device)
        % action_bank.shape[0]
    )
    age = direct.episode_length_buf.to(dtype=torch.long)
    final_index = (lengths[teacher_ids] - 1).clamp_min(0)
    step_ids = torch.minimum(age, final_index)
    return action_bank[teacher_ids, step_ids]


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    if not 0.0 <= args.teacher_rollout_fraction <= 1.0:
        raise ValueError("--teacher-rollout-fraction must be in [0, 1]")
    composite_expert = args.composite_expert_checkpoint is not None
    if composite_expert and args.teacher_npz is None:
        raise ValueError(
            "--composite-expert-checkpoint requires --teacher-npz"
        )
    teacher_action_scale = float(
        env_cfg.scripted_thumb_teacher_action_scale
    )
    teacher_residual_scale = float(
        env_cfg.scripted_thumb_teacher_residual_scale
    )
    composite_residual_ratio = (
        float(args.composite_residual_ratio)
        if args.composite_residual_ratio is not None
        else teacher_residual_scale / teacher_action_scale
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.standalone_rollout or composite_expert:
        env_cfg.scripted_thumb_teacher_use_normal_residual_path = False
        env_cfg.scripted_thumb_teacher_action_scale = 0.0
        env_cfg.scripted_thumb_teacher_residual_scale = 0.18
        env_cfg.trainee_policy_scale = 0.18
        env_cfg.final_residual_scale = 0.18
    agent_cfg["seed"] = args.seed
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped, agent_cfg)
    runner.agent.load(str(args.checkpoint.resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    residual_policy = (
        CheckpointPolicy(
            args.composite_expert_checkpoint.resolve(), direct.device
        )
        if composite_expert
        else None
    )

    observations, _ = wrapped.reset()
    states = wrapped.state()
    teacher_action_bank = None
    teacher_lengths = None
    if args.teacher_npz is not None:
        with np.load(args.teacher_npz) as data:
            teacher_action_bank = torch.as_tensor(
                np.asarray(data["action"]),
                dtype=torch.float32,
                device=direct.device,
            )
            teacher_lengths = torch.as_tensor(
                np.asarray(data["lengths"]),
                dtype=torch.long,
                device=direct.device,
            )
    observation_rows = []
    expert_rows = []
    student_rows = []
    phase_rows = []
    executed_rows = []
    hand_qpos_rows = []
    pen_root_state_rows = []
    final_phase = (
        int(direct.cfg.training_start_phase)
        + len(direct.cfg.closure_target_blends)
        - 1
    )
    rollout_best_hold_per_env = torch.zeros(
        direct.num_envs, dtype=torch.long, device=direct.device
    )
    rollout_success_latched = torch.zeros(
        direct.num_envs, dtype=torch.bool, device=direct.device
    )

    for step in range(args.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(
                observations,
                states,
                timestep=step,
                timesteps=args.steps,
            )
            student = outputs[-1].get("mean_actions", outputs[0])
            if teacher_action_bank is None:
                expert = scripted_action(direct)
            else:
                expert = trajectory_action(
                    direct, teacher_action_bank, teacher_lengths
                )
            residual_expert = None
            if residual_policy is not None:
                residual_expert = residual_policy(observations)
                # Keep this raw action unbounded. The embedded path composes
                # teacher and residual before the environment applies its
                # 0.18 action authority and final selected-action clamp.
                expert = (
                    expert
                    + composite_residual_ratio * residual_expert
                )
            record_step = step % max(args.stride, 1) == 0
            if record_step:
                observation_rows.append(observations.cpu().numpy())
                expert_rows.append(expert.cpu().numpy())
                student_rows.append(student.cpu().numpy())
                phase_rows.append(direct.chain_phase_ids.cpu().numpy())
            rollout_fraction = (
                args.teacher_rollout_fraction
                if composite_expert or not args.standalone_rollout
                else 0.0
            )
            rollout_action = torch.lerp(
                student, expert, rollout_fraction
            )
            observations, _, _, _, _ = wrapped.step(rollout_action)
            states = wrapped.state()
            current_final_hold = torch.where(
                direct.chain_phase_ids == final_phase,
                direct.chain_phase_hold,
                torch.zeros_like(direct.chain_phase_hold),
            )
            rollout_best_hold_per_env = torch.maximum(
                rollout_best_hold_per_env, current_final_hold
            )
            rollout_success_latched |= (
                rollout_best_hold_per_env
                >= int(direct.cfg.final_hold_steps)
            )
            if record_step:
                executed_rows.append(direct.actions.cpu().numpy())
                hand_qpos_rows.append(direct.hand_qpos.cpu().numpy())
                pen_root_state_rows.append(
                    torch.cat(
                        (
                            direct.pen_pos,
                            direct.pen_quat,
                            direct.pen_linvel,
                            direct.pen_angvel,
                        ),
                        dim=-1,
                    )
                    .cpu()
                    .numpy()
                )

    payload = {
        "observation": np.concatenate(observation_rows),
        "action": np.concatenate(expert_rows),
        "student_action": np.concatenate(student_rows),
        "stage_id": np.concatenate(phase_rows),
        "executed_action": np.concatenate(executed_rows),
        "post_step_hand_qpos": np.concatenate(hand_qpos_rows),
        "post_step_pen_root_state": np.concatenate(pen_root_state_rows),
        "source_student_checkpoint": np.asarray(str(args.checkpoint)),
        "teacher_assisted_rollout": np.asarray(
            args.teacher_rollout_fraction > 0.0
            and (composite_expert or not args.standalone_rollout)
        ),
        "standalone_rollout": np.asarray(
            args.standalone_rollout or composite_expert
        ),
        "teacher_rollout_fraction": np.asarray(
            args.teacher_rollout_fraction
            if composite_expert or not args.standalone_rollout
            else 0.0,
            dtype=np.float32,
        ),
        "teacher_source": np.asarray(
            str(args.teacher_npz) if args.teacher_npz else "embedded"
        ),
        "composite_expert_checkpoint": np.asarray(
            str(args.composite_expert_checkpoint)
            if args.composite_expert_checkpoint
            else ""
        ),
        "composite_residual_ratio": np.asarray(
            composite_residual_ratio, dtype=np.float32
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    disagreement = np.linalg.norm(
        payload["action"] - payload["student_action"], axis=-1
    )
    stages, counts = np.unique(payload["stage_id"], return_counts=True)
    final_active = direct.chain_phase_ids == final_phase
    current_final_hold = torch.where(
        final_active,
        direct.chain_phase_hold,
        torch.zeros_like(direct.chain_phase_hold),
    )
    current_final_hold_steps = int(current_final_hold.max())
    episode_final_hold_steps = int(direct.episode_final_max_hold.max())
    per_env_best_hold = torch.maximum(
        rollout_best_hold_per_env,
        torch.maximum(
            current_final_hold, direct.episode_final_max_hold
        ),
    )
    best_hold_steps = max(
        current_final_hold_steps, episode_final_hold_steps
    )
    success_mask = rollout_success_latched | direct.episode_p5_success
    success_envs = int(success_mask.sum())
    success_env_ids = torch.nonzero(
        success_mask, as_tuple=False
    ).squeeze(-1)
    top_count = min(16, direct.num_envs)
    top_hold, top_ids = torch.topk(per_env_best_hold, top_count)
    report = {
        "rows": int(len(disagreement)),
        "mean_action_l2_disagreement": float(disagreement.mean()),
        "p95_action_l2_disagreement": float(
            np.quantile(disagreement, 0.95)
        ),
        "stage_counts": {
            str(int(stage)): int(count)
            for stage, count in zip(stages, counts, strict=True)
        },
        "teacher_source": (
            str(args.teacher_npz) if args.teacher_npz else "embedded"
        ),
        "teacher_rollout_fraction": (
            args.teacher_rollout_fraction
            if composite_expert or not args.standalone_rollout
            else 0.0
        ),
        "composite_expert_checkpoint": (
            str(args.composite_expert_checkpoint)
            if args.composite_expert_checkpoint
            else None
        ),
        "composite_residual_ratio": composite_residual_ratio,
        "rollout_success_envs": success_envs,
        "rollout_success_env_ids": [
            int(value) for value in success_env_ids[:64].tolist()
        ],
        "rollout_best_hold_steps": best_hold_steps,
        "rollout_best_hold_ms": (
            1000.0 * best_hold_steps * float(direct.step_dt)
        ),
        "rollout_top_hold": [
            {
                "env_id": int(env_id),
                "teacher_id": int(
                    env_id
                    % (
                        teacher_action_bank.shape[0]
                        if teacher_action_bank is not None
                        else direct.scripted_thumb_teacher_first.shape[0]
                    )
                ),
                "hold_steps": int(hold),
            }
            for env_id, hold in zip(
                top_ids.tolist(), top_hold.tolist(), strict=True
            )
        ],
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("PHASE17_DAGGER=" + json.dumps(report, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
