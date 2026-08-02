from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Evaluate P4-to-P5-to-projected-P1 in one PhysX episode."
)
parser.add_argument(
    "--task", default="Dex5-V9P4-P5-P1-NoReset-Direct-v0"
)
parser.add_argument("--middle-checkpoint", required=True)
parser.add_argument("--capture-checkpoint", required=True)
parser.add_argument("--approach-checkpoint", required=True)
parser.add_argument("--contact-checkpoint", required=True)
parser.add_argument("--strict-checkpoint")
parser.add_argument("--strict-from-triad", action="store_true")
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--max_steps", type=int, default=4000)
parser.add_argument("--seed", type=int, default=2500)
parser.add_argument("--blend-steps", type=int, default=6)
parser.add_argument("--stage-hold-steps", type=int, default=12)
parser.add_argument("--p1-hold-steps", type=int, default=24)
parser.add_argument(
    "--approach-mode",
    choices=("policy", "target"),
    default="policy",
)
parser.add_argument("--approach-target-gain", type=float, default=0.02)
parser.add_argument(
    "--approach-middle-multiplier", type=float, default=1.0
)
parser.add_argument(
    "--approach-action-rescale",
    type=float,
    default=(2.0 / 3.0),
)
parser.add_argument(
    "--contact-middle-multiplier", type=float, default=1.0
)
parser.add_argument(
    "--teacher-prefix",
    help="Write successful continuous approach/contact BC datasets.",
)
parser.add_argument(
    "--transition-cache-prefix",
    help="Write same-episode state caches at every phase handoff.",
)
parser.add_argument(
    "--max-transition-states", type=int, default=4096
)
parser.add_argument(
    "--trajectory-output",
    help="Write the first complete successful no-reset trajectory.",
)
parser.add_argument("--progress-interval", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch import nn

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


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


def replace_rows(
    value: torch.Tensor, mask: torch.Tensor, replacement: torch.Tensor
) -> torch.Tensor:
    return torch.where(mask.unsqueeze(-1), replacement, value)


def main() -> None:
    cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs
    )
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    policies = [
        Policy(path, direct.device)
        for path in (
            args.middle_checkpoint,
            args.capture_checkpoint,
            args.approach_checkpoint,
            args.contact_checkpoint,
        )
    ]
    strict_policy = (
        Policy(args.strict_checkpoint, direct.device)
        if args.strict_checkpoint
        else None
    )

    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "reports/m3_2_30_index_near_surface/"
        "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    p1_row = next(
        row for row in manifest["phases"] if int(row["phase"]) == 1
    )
    p1_hand = torch.as_tensor(
        p1_row["hand_qpos"],
        dtype=torch.float32,
        device=direct.device,
    )
    with np.load(
        Path(__file__).resolve().parents[1]
        / "reports/m3_2_30_index_near_surface/"
        "p4_p5_curriculum/capture_collision_free_cache_seed1503.npz"
    ) as p5_template:
        p5_hand = torch.as_tensor(
            np.asarray(
                p5_template["target_qpos"], dtype=np.float32
            )[:, :20].mean(axis=0),
            device=direct.device,
        )
    with torch.inference_mode():
        obs, _ = env.reset()

    phase = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    phase_hold = torch.zeros_like(phase)
    chain_success = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    episode_counted = torch.zeros_like(chain_success)
    blend_age = torch.full_like(phase, args.blend_steps)
    previous_action = torch.zeros(
        (args.num_envs, 20), device=direct.device
    )
    completed = 0
    successful = 0
    transitions = torch.zeros(
        5, dtype=torch.long, device=direct.device
    )
    successful_max_penetration_mm = 0.0
    max_penetration_mm = 0.0
    approach_gap_min_mm = float("inf")
    approach_gap_max_mm = float("-inf")
    approach_near_env_steps = 0
    approach_env_steps = 0
    strict_env_steps = 0
    strict_safe_env_steps = 0
    strict_triad_env_steps = 0
    strict_pose_env_steps = 0
    max_strict_hold_steps = 0
    phase_age = torch.zeros_like(phase)
    teacher_rows: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        2: [],
        3: [],
    }
    transition_rows: dict[int, dict[str, list[np.ndarray]]] = {
        phase_id: {
            key: []
            for key in (
                "source_qpos",
                "source_qvel",
                "source_observation",
                "source_policy_action",
                "source_target_qpos",
                "source_contact_force_n",
                "source_tip_surface_dist_m",
                "source_max_penetration_m",
            )
        }
        for phase_id in range(5)
    }
    if args.teacher_prefix:
        teacher_horizon = direct.max_episode_length + 1
        teacher_obs_buffer = torch.zeros(
            (
                args.num_envs,
                teacher_horizon,
                obs["policy"].shape[-1],
            ),
            dtype=torch.float32,
            device=direct.device,
        )
        teacher_action_buffer = torch.zeros(
            (args.num_envs, teacher_horizon, 20),
            dtype=torch.float32,
            device=direct.device,
        )
    trace_saved = False
    saved_trace: dict[str, np.ndarray] | None = None
    chain_age = torch.zeros_like(phase)
    if args.trajectory_output:
        trace_horizon = direct.max_episode_length + 1
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
    steps = 0

    while completed < args.episodes and steps < args.max_steps:
        if args.progress_interval == 1:
            print(f"step={steps}: policy phase={phase.tolist()}", flush=True)
        with torch.inference_mode():
            policy_obs = obs["policy"]
            candidates = [policy(policy_obs) for policy in policies]
            candidates[2] = args.approach_action_rescale * candidates[2]
            candidates[2][:, 4:8] *= args.approach_middle_multiplier
            candidates[3][:, 4:8] *= args.contact_middle_multiplier
            if args.approach_mode == "target":
                mask = direct._stage_action_mask()
                increment = (
                    0.25 * direct.cfg.action_scale * mask
                ).clamp_min(1.0e-6)
                target_action = (
                    args.approach_target_gain
                    * (
                        direct.target_hand_qpos
                        - direct.policy_targets
                    )
                    / increment
                )
                candidates[2] = torch.where(
                    mask > 0.0,
                    target_action,
                    torch.zeros_like(target_action),
                ).clamp(-1.0, 1.0)
            actions = torch.zeros(
                (args.num_envs, 20), device=direct.device
            )
            for phase_id, candidate in enumerate(candidates):
                active = phase == phase_id
                actions = torch.where(
                    active.unsqueeze(-1), candidate, actions
                )
            if strict_policy is not None:
                strict_actions = strict_policy(policy_obs)
                strict_active = (
                    phase >= 4
                    if args.strict_from_triad
                    else phase == 5
                )
                actions = torch.where(
                    strict_active.unsqueeze(-1),
                    strict_actions,
                    actions,
                )
            blending = blend_age < args.blend_steps
            alpha = (
                (blend_age + 1).float()
                / max(args.blend_steps, 1)
            ).clamp(0.0, 1.0).unsqueeze(-1)
            actions = torch.where(
                blending.unsqueeze(-1),
                torch.lerp(previous_action, actions, alpha),
                actions,
            ).clamp(-1.0, 1.0)
            if args.teacher_prefix:
                recordable = (phase == 2) | (phase == 3)
                rows = torch.arange(
                    args.num_envs, device=direct.device
                )
                record_step = phase_age.clamp_max(
                    teacher_horizon - 1
                )
                current_obs = teacher_obs_buffer[rows, record_step]
                current_action = teacher_action_buffer[
                    rows, record_step
                ]
                teacher_obs_buffer[rows, record_step] = torch.where(
                    recordable.unsqueeze(-1),
                    policy_obs,
                    current_obs,
                )
                teacher_action_buffer[rows, record_step] = torch.where(
                    recordable.unsqueeze(-1),
                    actions,
                    current_action,
                )
            blend_age = torch.where(
                blending, blend_age + 1, blend_age
            )
            if args.progress_interval == 1:
                print(f"step={steps}: physics", flush=True)
            obs, _, terminated, truncated, _ = env.step(actions)

        if args.progress_interval == 1:
            print(f"step={steps}: gates", flush=True)
        direct._compute_intermediate_values()
        if args.trajectory_output and not trace_saved:
            rows = torch.arange(args.num_envs, device=direct.device)
            trace_step = chain_age.clamp_max(trace_horizon - 1)
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
            trace_phase[rows, trace_step] = phase
            trace_penetration[rows, trace_step] = (
                direct.episode_max_penetration
            )
            chain_age = (chain_age + 1).clamp_max(trace_horizon)
        contacts = direct.contact_force_norm > 0.03
        thumb_index = contacts[:, :2].all(dim=-1)
        middle_gap = (
            direct.tip_surface_dist[:, 2]
            - direct.fingertip_radii[:, 2]
        )
        middle_near = (middle_gap >= 0.0005) & (middle_gap <= 0.004)
        middle_contact = contacts[:, 2] & (middle_gap <= 0.001)
        triad_contact = thumb_index & middle_contact
        drift = torch.linalg.norm(
            direct.pen_pos - direct.init_pen_pos, dim=-1
        )
        axis_dot = torch.sum(
            direct.pen_axis * direct.init_pen_axis, dim=-1
        ).clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot.abs())
        speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        peak_force = direct.contact_force_norm.max(dim=-1).values
        safe_pen = (
            (drift <= 0.015)
            & (axis_drift <= torch.deg2rad(
                torch.full_like(axis_drift, 15.0)
            ))
            & (speed <= 12.0)
            & (direct.episode_max_penetration <= 0.00075)
            & (peak_force <= 6.0)
            & ~terminated
        )
        approach_active = phase == 2
        if torch.any(approach_active):
            active_gap = middle_gap[approach_active]
            approach_gap_min_mm = min(
                approach_gap_min_mm,
                float(active_gap.min().item() * 1000.0),
            )
            approach_gap_max_mm = max(
                approach_gap_max_mm,
                float(active_gap.max().item() * 1000.0),
            )
            approach_near_env_steps += int(
                (middle_near & approach_active).sum().item()
            )
            approach_env_steps += int(approach_active.sum().item())
        triad_rmse = torch.sqrt(
            torch.mean(
                (
                    direct.hand_qpos[:, :12]
                    - direct.target_hand_qpos[:, :12]
                ).square(),
                dim=-1,
            )
        )
        strict_active = phase == 5
        strict_env_steps += int(strict_active.sum().item())
        strict_safe_env_steps += int(
            (strict_active & safe_pen).sum().item()
        )
        strict_triad_env_steps += int(
            (strict_active & triad_contact).sum().item()
        )
        strict_pose_env_steps += int(
            (strict_active & (triad_rmse <= 0.10)).sum().item()
        )
        eligible = torch.zeros(
            args.num_envs, dtype=torch.bool, device=direct.device
        )
        eligible |= (phase == 0) & (
            direct.stage_hold_count >= args.stage_hold_steps
        )
        eligible |= (phase == 1) & (
            direct.stage_hold_count >= args.stage_hold_steps
        )
        eligible |= (
            (phase == 2) & safe_pen & thumb_index & middle_near
        )
        eligible |= (
            (phase == 3) & safe_pen & triad_contact
        )
        eligible |= (
            (phase == 4)
            & safe_pen
            & triad_contact
            & (triad_rmse <= 0.32)
        )
        eligible |= (
            (phase == 5)
            & safe_pen
            & triad_contact
            & (triad_rmse <= 0.10)
        )
        manual_phase = phase >= 2
        phase_hold = torch.where(
            manual_phase & eligible,
            phase_hold + 1,
            torch.where(
                manual_phase,
                torch.zeros_like(phase_hold),
                phase_hold,
            ),
        )
        phase_age += 1
        if torch.any(strict_active):
            max_strict_hold_steps = max(
                max_strict_hold_steps,
                int(phase_hold[strict_active].max().item()),
            )

        transition = torch.zeros_like(phase, dtype=torch.bool)
        transition |= (phase <= 1) & eligible
        transition |= (
            (phase >= 2)
            & (phase < 5)
            & (phase_hold >= args.stage_hold_steps)
        )
        for source_phase in range(5):
            source_mask = transition & (phase == source_phase)
            transitions[source_phase] += source_mask.sum()
            if (
                args.transition_cache_prefix
                and torch.any(source_mask)
            ):
                rows = transition_rows[source_phase]
                stored = sum(
                    len(chunk) for chunk in rows["source_qpos"]
                )
                remaining = args.max_transition_states - stored
                source_ids = torch.nonzero(
                    source_mask, as_tuple=False
                ).squeeze(-1)[:remaining]
                if source_ids.numel() > 0:
                    tensors = {
                        "source_qpos": torch.cat(
                            (
                                direct.hand_qpos[source_ids],
                                direct.pen_pos[source_ids],
                                direct.pen_quat[source_ids],
                            ),
                            dim=-1,
                        ),
                        "source_qvel": torch.cat(
                            (
                                direct.hand_qvel[source_ids],
                                direct.pen_linvel[source_ids],
                                direct.pen_angvel[source_ids],
                            ),
                            dim=-1,
                        ),
                        "source_observation": policy_obs[source_ids],
                        "source_policy_action": actions[source_ids],
                        "source_target_qpos": (
                            direct.target_hand_qpos[source_ids]
                        ),
                        "source_contact_force_n": (
                            direct.contact_force_norm[source_ids]
                        ),
                        "source_tip_surface_dist_m": (
                            direct.tip_surface_dist[source_ids]
                        ),
                        "source_max_penetration_m": (
                            direct.episode_max_penetration[source_ids]
                        ),
                    }
                    for key, value in tensors.items():
                        rows[key].append(
                            value.detach().cpu().numpy().copy()
                        )
            if args.teacher_prefix and source_phase in teacher_rows:
                source_ids = torch.nonzero(
                    source_mask, as_tuple=False
                ).squeeze(-1)
                for env_id in source_ids.tolist():
                    length = int(phase_age[env_id].item())
                    teacher_rows[source_phase].append(
                        (
                            teacher_obs_buffer[
                                env_id, :length
                            ].detach().cpu().numpy().copy(),
                            teacher_action_buffer[
                                env_id, :length
                            ].detach().cpu().numpy().copy(),
                        )
                    )

        if torch.any(transition):
            if args.progress_interval == 1:
                print(f"step={steps}: handoff begin", flush=True)
            next_phase = torch.where(transition, phase + 1, phase)
            next_stage = torch.where(
                next_phase == 0,
                torch.ones_like(next_phase),
                torch.where(
                    next_phase == 1,
                    torch.full_like(next_phase, 4),
                    torch.where(
                        next_phase == 2,
                        torch.ones_like(next_phase),
                        torch.where(
                            next_phase == 3,
                            torch.full_like(next_phase, 2),
                            torch.where(
                            next_phase == 4,
                                torch.full_like(
                                    next_phase,
                                    4 if args.strict_from_triad else 3,
                                ),
                                torch.full_like(next_phase, 4),
                            ),
                        ),
                    ),
                ),
            )
            direct.chain_phase_ids = next_phase
            direct.p5_stage_ids = next_stage
            new_target = direct.target_hand_qpos.clone()

            to_capture = transition & (phase == 0)
            capture_target = direct.policy_targets.clone()
            capture_target[:, :4] = p5_hand[:4]
            capture_target[:, 16:20] = p5_hand[16:20]
            new_target = replace_rows(
                new_target, to_capture, capture_target
            )

            to_approach = transition & (phase == 1)
            approach_target = direct.policy_targets.clone()
            approach_target[:, 4:8] = (
                0.45 * direct.policy_targets[:, 4:8]
                + 0.55 * p1_hand[4:8]
            )
            new_target = replace_rows(
                new_target, to_approach, approach_target
            )

            to_contact = transition & (phase == 2)
            contact_target = direct.policy_targets.clone()
            contact_target[:, 4:8] = p1_hand[4:8]
            new_target = replace_rows(
                new_target, to_contact, contact_target
            )

            to_hold = transition & (phase >= 3)
            projected_target = direct.hand_qpos.clone()
            direct.policy_targets = replace_rows(
                direct.policy_targets, to_hold, projected_target
            )
            canonical_p1_target = p1_hand.unsqueeze(0).expand_as(
                new_target
            )
            new_target = replace_rows(
                new_target, to_hold, canonical_p1_target
            )
            direct.target_hand_qpos = new_target
            direct.stage_hold_count = torch.where(
                transition,
                torch.zeros_like(direct.stage_hold_count),
                direct.stage_hold_count,
            )
            direct.p5_hold_count = torch.where(
                transition,
                torch.zeros_like(direct.p5_hold_count),
                direct.p5_hold_count,
            )
            phase_hold = torch.where(
                transition, torch.zeros_like(phase_hold), phase_hold
            )
            phase_age = torch.where(
                transition, torch.zeros_like(phase_age), phase_age
            )
            previous_action = replace_rows(
                previous_action, transition, actions
            )
            blend_age = torch.where(
                transition, torch.zeros_like(blend_age), blend_age
            )
            phase = next_phase
            obs = direct._get_observations()
            if args.progress_interval == 1:
                print(f"step={steps}: handoff end", flush=True)

        p1_success_now = (
            (phase == 5) & (phase_hold >= args.p1_hold_steps)
        )
        newly_successful = p1_success_now & ~chain_success
        if torch.any(newly_successful):
            successful_max_penetration_mm = max(
                successful_max_penetration_mm,
                float(
                    direct.episode_max_penetration[
                        newly_successful
                    ].max().item()
                    * 1000.0
                ),
            )
            countable = newly_successful & ~episode_counted
            remaining = args.episodes - completed
            countable_ids = torch.nonzero(
                countable, as_tuple=False
            ).squeeze(-1)[:remaining]
            if countable_ids.numel() > 0:
                newly_counted = torch.zeros_like(episode_counted)
                newly_counted = (
                    torch.arange(
                        args.num_envs, device=direct.device
                    ).unsqueeze(-1)
                    == countable_ids.reshape(1, -1)
                ).any(dim=-1)
                successful += int(countable_ids.numel())
                completed += int(countable_ids.numel())
                episode_counted |= newly_counted
            if args.trajectory_output and not trace_saved:
                env_id = int(
                    torch.nonzero(
                        newly_successful, as_tuple=False
                    )[0].item()
                )
                length = int(chain_age[env_id].item())
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
        chain_success |= p1_success_now
        max_penetration_mm = max(
            max_penetration_mm,
            float(direct.episode_max_penetration.max().item() * 1000.0),
        )

        done = (terminated | truncated).reshape(-1)
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            remaining = args.episodes - completed
            uncounted_done = done & ~episode_counted
            uncounted_ids = torch.nonzero(
                uncounted_done, as_tuple=False
            ).squeeze(-1)[:remaining]
            completed += int(uncounted_ids.numel())
            phase = torch.where(done, torch.zeros_like(phase), phase)
            phase_hold = torch.where(
                done, torch.zeros_like(phase_hold), phase_hold
            )
            phase_age = torch.where(
                done, torch.zeros_like(phase_age), phase_age
            )
            chain_age = torch.where(
                done, torch.zeros_like(chain_age), chain_age
            )
            episode_counted = episode_counted & ~done
            chain_success = chain_success & ~done
            blend_age = torch.where(
                done,
                torch.full_like(blend_age, args.blend_steps),
                blend_age,
            )
            previous_action = torch.where(
                done.unsqueeze(-1),
                torch.zeros_like(previous_action),
                previous_action,
            )
        steps += 1
        if (
            args.progress_interval > 0
            and steps % args.progress_interval == 0
        ):
            print(
                f"step={steps} episodes={completed} success={successful} "
                f"phase_counts={torch.bincount(phase, minlength=6).tolist()} "
                f"approach_gap_mm=[{approach_gap_min_mm:.3f},"
                f"{approach_gap_max_mm:.3f}]",
                flush=True,
            )

    local_product = 0.853986552 * 0.98046875 * 0.985479187 * 0.9921875
    success_rate = successful / max(completed, 1)
    result = {
        "task": args.task,
        "seed": args.seed,
        "episodes": completed,
        "successes": successful,
        "success_rate": success_rate,
        "simulation_steps": steps,
        "transition_counts": transitions.cpu().tolist(),
        "p1_hold_ms": (
            args.p1_hold_steps
            * direct.cfg.decimation
            * direct.cfg.sim.dt
            * 1000.0
        ),
        "drop_counter": int(direct.p5_done_drop_count.item()),
        "max_penetration_mm_observed": max_penetration_mm,
        "max_penetration_mm_at_success": successful_max_penetration_mm,
        "approach_gap_mm_observed": {
            "min": approach_gap_min_mm,
            "max": approach_gap_max_mm,
            "near_env_step_ratio": (
                approach_near_env_steps / max(approach_env_steps, 1)
            ),
        },
        "strict_hold_diagnostics": {
            "max_hold_steps": max_strict_hold_steps,
            "max_hold_ms": (
                max_strict_hold_steps
                * direct.cfg.decimation
                * direct.cfg.sim.dt
                * 1000.0
            ),
            "safe_pen_ratio": (
                strict_safe_env_steps / max(strict_env_steps, 1)
            ),
            "triad_contact_ratio": (
                strict_triad_env_steps / max(strict_env_steps, 1)
            ),
            "pose_ratio": (
                strict_pose_env_steps / max(strict_env_steps, 1)
            ),
        },
        "teacher_trajectories": {
            "approach": len(teacher_rows[2]),
            "contact": len(teacher_rows[3]),
        },
        "transition_cache_rows": {
            str(source_phase): sum(
                len(chunk)
                for chunk in rows["source_qpos"]
            )
            for source_phase, rows in transition_rows.items()
        },
        "handoff": "same PhysX episode; no hand or pen state reset",
        "approach_mode": args.approach_mode,
        "approach_target_gain": args.approach_target_gain,
        "approach_middle_multiplier": args.approach_middle_multiplier,
        "approach_action_rescale": args.approach_action_rescale,
        "contact_middle_multiplier": args.contact_middle_multiplier,
        "strict_from_triad": args.strict_from_triad,
        "local_primitive_product_reference": local_product,
        "linkage_efficiency": success_rate / local_product,
        "checkpoints": {
            "middle": str(Path(args.middle_checkpoint).resolve()),
            "capture": str(Path(args.capture_checkpoint).resolve()),
            "approach": str(Path(args.approach_checkpoint).resolve()),
            "contact": str(Path(args.contact_checkpoint).resolve()),
            "strict": (
                str(Path(args.strict_checkpoint).resolve())
                if args.strict_checkpoint
                else None
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.teacher_prefix:
        prefix = Path(args.teacher_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for source_phase, label in ((2, "approach"), (3, "contact")):
            rows = teacher_rows[source_phase]
            if not rows:
                continue
            lengths = np.asarray(
                [len(obs_row) for obs_row, _ in rows],
                dtype=np.int64,
            )
            horizon = int(lengths.max())
            observation = np.zeros(
                (
                    len(rows),
                    horizon,
                    rows[0][0].shape[-1],
                ),
                dtype=np.float32,
            )
            action = np.zeros(
                (len(rows), horizon, 20), dtype=np.float32
            )
            for row_id, (obs_row, action_row) in enumerate(rows):
                observation[row_id, : len(obs_row)] = obs_row
                action[row_id, : len(action_row)] = action_row
            np.savez_compressed(
                prefix.with_name(f"{prefix.name}_{label}.npz"),
                observation=observation,
                action=action,
                stage_id=np.full(
                    (len(rows), horizon),
                    1 if source_phase == 2 else 2,
                    dtype=np.int64,
                ),
                lengths=lengths,
                seed=np.asarray(args.seed, dtype=np.int64),
                source=np.asarray(
                    "same-episode P4->P5->P1 continuous teacher"
                ),
            )
    if args.transition_cache_prefix:
        prefix = Path(args.transition_cache_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        labels = {
            0: "P4中指退让到P5捕获",
            1: "P5捕获到P1接近",
            2: "P1接近到接触",
            3: "P1接触到三指支撑",
            4: "P1三指支撑到严格保持",
        }
        cache_outputs: dict[str, str] = {}
        for source_phase, rows in transition_rows.items():
            if not rows["source_qpos"]:
                continue
            payload = {
                key: np.concatenate(chunks, axis=0)
                for key, chunks in rows.items()
                if chunks
            }
            count = len(payload["source_qpos"])
            payload["source_phase_id"] = np.full(
                count, source_phase, dtype=np.int64
            )
            payload["target_phase_id"] = np.full(
                count, source_phase + 1, dtype=np.int64
            )
            payload["seed"] = np.asarray(args.seed, dtype=np.int64)
            payload["source"] = np.asarray(
                "same PhysX episode; no hand or pen state reset"
            )
            cache_path = prefix.with_name(
                f"{prefix.name}_{labels[source_phase]}.npz"
            )
            np.savez_compressed(cache_path, **payload)
            cache_outputs[str(source_phase)] = str(cache_path)
        result["transition_cache_outputs"] = cache_outputs
    if args.trajectory_output and saved_trace is not None:
        trajectory_path = Path(args.trajectory_output)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trajectory_path, **saved_trace)
        result["trajectory_output"] = str(trajectory_path)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
