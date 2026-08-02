from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Evaluate middle-clear to P5 capture without a physics reset."
)
parser.add_argument(
    "--task",
    default="Dex5-V9P4-P5-CurrentP4MiddleClear-Direct-v0",
)
parser.add_argument("--middle-checkpoint", required=True)
parser.add_argument("--capture-checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--max_steps", type=int, default=4000)
parser.add_argument("--seed", type=int, default=1900)
parser.add_argument("--blend-steps", type=int, default=6)
parser.add_argument("--capture-hold-steps", type=int, default=12)
parser.add_argument("--progress-interval", type=int, default=100)
parser.add_argument("--success-cache")
parser.add_argument("--trajectory-output")
parser.add_argument("--max-success-states", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


TASK = args.task


class Policy(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
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
        layer_pairs = (
            (0, "net_container.0"),
            (2, "net_container.2"),
            (4, "net_container.4"),
            (6, "policy_layer"),
        )
        for layer, key in layer_pairs:
            self.net[layer].weight.data.copy_(state[f"{key}.weight"])
            self.net[layer].bias.data.copy_(state[f"{key}.bias"])
        preprocessor = checkpoint["observation_preprocessor"]
        self.register_buffer("mean", preprocessor["running_mean"].float())
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


def main() -> None:
    cfg = parse_env_cfg(
        TASK, device=args.device, num_envs=args.num_envs
    )
    cfg.seed = args.seed
    cfg.episode_length_s = 1.50
    print("cascade: creating environment", flush=True)
    env = gym.make(TASK, cfg=cfg)
    direct = env.unwrapped
    print("cascade: loading middle policy", flush=True)
    middle_policy = Policy(args.middle_checkpoint, direct.device)
    print("cascade: loading capture policy", flush=True)
    capture_policy = Policy(args.capture_checkpoint, direct.device)
    with np.load(
        "reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
        "capture_collision_free_cache_seed1503.npz"
    ) as template:
        p5_hand_np = np.asarray(
            template["target_qpos"], dtype=np.float32
        )[:, :20].mean(axis=0)
    p5_hand = torch.as_tensor(p5_hand_np, device=direct.device)

    print("cascade: resetting environment", flush=True)
    obs, _ = env.reset()
    print("cascade: rollout started", flush=True)
    capture_phase = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    middle_seen = torch.zeros_like(capture_phase)
    cascade_success = torch.zeros_like(capture_phase)
    blend_age = torch.full(
        (args.num_envs,),
        args.blend_steps,
        dtype=torch.long,
        device=direct.device,
    )
    previous_action = torch.zeros(
        (args.num_envs, 20), device=direct.device
    )
    completed = 0
    successful = 0
    middle_completed = 0
    dropped = 0
    max_penetration_mm = 0.0
    steps = 0
    success_rows: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "source_qpos",
            "source_qvel",
            "source_target_qpos",
            "source_policy_action",
            "source_contact_force_n",
            "source_angular_speed_rad_s",
            "source_max_penetration_m",
        )
    }
    trace_saved = False
    trace_horizon = direct.max_episode_length + 1
    if args.trajectory_output:
        trace_qpos = torch.zeros(
            (args.num_envs, trace_horizon, 27),
            dtype=torch.float32,
            device=direct.device,
        )
        trace_qvel = torch.zeros(
            (args.num_envs, trace_horizon, 26),
            dtype=torch.float32,
            device=direct.device,
        )
        trace_action = torch.zeros(
            (args.num_envs, trace_horizon, 20),
            dtype=torch.float32,
            device=direct.device,
        )
        trace_phase = torch.zeros(
            (args.num_envs, trace_horizon),
            dtype=torch.int64,
            device=direct.device,
        )
        trace_penetration = torch.zeros(
            (args.num_envs, trace_horizon),
            dtype=torch.float32,
            device=direct.device,
        )
        saved_trace: dict[str, np.ndarray] | None = None

    while completed < args.episodes and steps < args.max_steps:
        if args.progress_interval == 1:
            print(f"step={steps}: policy", flush=True)
        with torch.inference_mode():
            middle_actions = middle_policy(obs["policy"])
            capture_actions = capture_policy(obs["policy"])
            actions = torch.where(
                capture_phase.unsqueeze(-1),
                capture_actions,
                middle_actions,
            ).clamp(-1.0, 1.0)
            blending = capture_phase & (blend_age < args.blend_steps)
            if torch.any(blending):
                alpha = (
                    (blend_age[blending] + 1).float()
                    / max(args.blend_steps, 1)
                ).unsqueeze(-1)
                actions[blending] = torch.lerp(
                    previous_action[blending],
                    capture_actions[blending],
                    alpha,
                )
                blend_age[blending] += 1
            if args.progress_interval == 1:
                print(f"step={steps}: physics", flush=True)
            obs, _, terminated, truncated, _ = env.step(actions)
        if args.progress_interval == 1:
            print(f"step={steps}: metrics", flush=True)

        entered = (~capture_phase) & (
            direct.stage_hold_count >= direct.cfg.p5_hold_steps
        )
        if torch.any(entered):
            if args.progress_interval == 1:
                print(f"step={steps}: handoff begin", flush=True)
            middle_seen |= entered
            if args.progress_interval == 1:
                print(f"step={steps}: handoff middle_seen", flush=True)
            middle_completed += int(entered.sum().item())
            if args.progress_interval == 1:
                print(f"step={steps}: handoff count", flush=True)
            direct.p5_stage_ids = torch.where(
                entered,
                torch.full_like(direct.p5_stage_ids, 4),
                direct.p5_stage_ids,
            )
            if args.progress_interval == 1:
                print(f"step={steps}: handoff stage", flush=True)
            direct.stage_hold_count = torch.where(
                entered,
                torch.zeros_like(direct.stage_hold_count),
                direct.stage_hold_count,
            )
            direct.p5_hold_count = torch.where(
                entered,
                torch.zeros_like(direct.p5_hold_count),
                direct.p5_hold_count,
            )
            direct.episode_p5_success = (
                direct.episode_p5_success & ~entered
            )
            if args.progress_interval == 1:
                print(f"step={steps}: handoff counters", flush=True)
            capture_target = direct.policy_targets.clone()
            capture_target[:, :4] = p5_hand[:4]
            capture_target[:, 16:20] = p5_hand[16:20]
            direct.target_hand_qpos = torch.where(
                entered.unsqueeze(-1),
                capture_target,
                direct.target_hand_qpos,
            )
            if args.progress_interval == 1:
                print(f"step={steps}: handoff target base", flush=True)
            if args.progress_interval == 1:
                print(f"step={steps}: handoff target pinch", flush=True)
            previous_action = torch.where(
                entered.unsqueeze(-1), actions, previous_action
            )
            blend_age = torch.where(
                entered, torch.zeros_like(blend_age), blend_age
            )
            capture_phase |= entered
            if args.progress_interval == 1:
                print(f"step={steps}: handoff observations", flush=True)
            obs = direct._get_observations()
            if args.progress_interval == 1:
                print(f"step={steps}: handoff end", flush=True)

        success_now = capture_phase & (
            direct.stage_hold_count >= args.capture_hold_steps
        )
        newly_successful = success_now & ~cascade_success
        cascade_success |= success_now
        if args.progress_interval == 1:
            print(f"step={steps}: intermediate values", flush=True)
        direct._compute_intermediate_values()
        if args.trajectory_output and not trace_saved:
            rows = torch.arange(args.num_envs, device=direct.device)
            trace_step = direct.episode_length_buf.clamp(
                min=0, max=trace_horizon - 1
            )
            trace_qpos[rows, trace_step] = torch.cat(
                (
                    direct.hand_qpos,
                    direct.pen_pos,
                    direct.pen_quat,
                ),
                dim=-1,
            )
            trace_qvel[rows, trace_step] = torch.cat(
                (
                    direct.hand_qvel,
                    direct.pen_linvel,
                    direct.pen_angvel,
                ),
                dim=-1,
            )
            trace_action[rows, trace_step] = actions
            trace_phase[rows, trace_step] = torch.where(
                capture_phase,
                torch.full_like(direct.p5_stage_ids, 2),
                torch.ones_like(direct.p5_stage_ids),
            )
            trace_penetration[rows, trace_step] = (
                direct.episode_max_penetration
            )
        if torch.any(newly_successful):
            success_ids = torch.nonzero(
                newly_successful, as_tuple=False
            ).squeeze(-1)
            remaining_states = (
                args.max_success_states
                - sum(len(chunk) for chunk in success_rows["source_qpos"])
            )
            if args.success_cache and remaining_states > 0:
                success_ids = success_ids[:remaining_states]
                qpos = torch.cat(
                    (
                        direct.hand_qpos[success_ids],
                        direct.pen_pos[success_ids],
                        direct.pen_quat[success_ids],
                    ),
                    dim=-1,
                )
                qvel = torch.cat(
                    (
                        direct.hand_qvel[success_ids],
                        direct.pen_linvel[success_ids],
                        direct.pen_angvel[success_ids],
                    ),
                    dim=-1,
                )
                tensors = {
                    "source_qpos": qpos,
                    "source_qvel": qvel,
                    "source_target_qpos": direct.policy_targets[success_ids],
                    "source_policy_action": actions[success_ids],
                    "source_contact_force_n": direct.contact_force_norm[
                        success_ids
                    ],
                    "source_angular_speed_rad_s": torch.linalg.norm(
                        direct.pen_angvel[success_ids], dim=-1
                    ),
                    "source_max_penetration_m": (
                        direct.episode_max_penetration[success_ids]
                    ),
                }
                for key, value in tensors.items():
                    success_rows[key].append(value.detach().cpu().numpy())
            if args.trajectory_output and not trace_saved:
                env_id = int(success_ids[0].item())
                length = int(
                    direct.episode_length_buf[env_id].item()
                ) + 1
                saved_trace = {
                    "qpos": trace_qpos[env_id, :length]
                    .detach()
                    .cpu()
                    .numpy(),
                    "qvel": trace_qvel[env_id, :length]
                    .detach()
                    .cpu()
                    .numpy(),
                    "action": trace_action[env_id, :length]
                    .detach()
                    .cpu()
                    .numpy(),
                    "phase_ids": trace_phase[env_id, :length]
                    .detach()
                    .cpu()
                    .numpy(),
                    "max_penetration_m": trace_penetration[
                        env_id, :length
                    ]
                    .detach()
                    .cpu()
                    .numpy(),
                    "control_dt_s": np.asarray(
                        direct.cfg.decimation * direct.cfg.sim.dt,
                        dtype=np.float32,
                    ),
                }
                trace_saved = True
        max_penetration_mm = max(
            max_penetration_mm,
            float(direct.episode_max_penetration.max().item() * 1000.0),
        )
        done = (terminated | truncated).reshape(-1)
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            remaining = args.episodes - completed
            done_ids = done_ids[:remaining]
            successful += int(cascade_success[done_ids].sum().item())
            completed += len(done_ids)
            dropped = int(direct.p5_done_drop_count.item())
            capture_phase[done_ids] = False
            blend_age[done_ids] = args.blend_steps
            previous_action[done_ids] = 0.0
            middle_seen[done_ids] = False
            cascade_success[done_ids] = False
        steps += 1
        if args.progress_interval > 0 and steps % args.progress_interval == 0:
            print(
                f"step={steps} completed={completed} "
                f"middle_entries={middle_completed} successful={successful}",
                flush=True,
            )

    result = {
        "task": TASK,
        "middle_checkpoint": str(Path(args.middle_checkpoint).resolve()),
        "capture_checkpoint": str(Path(args.capture_checkpoint).resolve()),
        "seed": args.seed,
        "episodes": completed,
        "middle_entries": middle_completed,
        "cascade_successes": successful,
        "cascade_success_rate": successful / max(completed, 1),
        "simulation_steps": steps,
        "drop_counter": dropped,
        "max_penetration_mm_observed": max_penetration_mm,
        "handoff": "same PhysX episode; no hand or pen reset",
        "capture_hold_ms": (
            args.capture_hold_steps
            * direct.cfg.decimation
            * direct.cfg.sim.dt
            * 1000.0
        ),
        "blend_steps": args.blend_steps,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if args.success_cache:
        cache_path = Path(args.success_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in success_rows.items()
            if chunks
        }
        if cache_payload:
            count = len(cache_payload["source_qpos"])
            cache_payload["source_teacher_actions"] = np.zeros(
                (count, 0, 20), dtype=np.float32
            )
            cache_payload["original_source_bank_ids"] = np.arange(
                count, dtype=np.int64
            )
            np.savez_compressed(cache_path, **cache_payload)
            result["success_cache"] = str(cache_path)
            result["success_cache_rows"] = count
    if args.trajectory_output and saved_trace is not None:
        trajectory_path = Path(args.trajectory_output)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trajectory_path, **saved_trace)
        result["trajectory_output"] = str(trajectory_path)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
