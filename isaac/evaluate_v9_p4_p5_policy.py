from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a V9 P4-to-P5 policy.")
parser.add_argument("--task", default="Dex5-V9P4-P5-Direct-v0")
parser.add_argument("--checkpoint")
parser.add_argument(
    "--zero-actions",
    action="store_true",
    help="Evaluate an environment-embedded scripted teacher without a policy.",
)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--trajectory-output",
    help="Write the first successful same-episode state trajectory.",
)
parser.add_argument(
    "--stage-cache-output",
    help="Write first-entry states for a selected continuous-chain phase.",
)
parser.add_argument("--stage-cache-phase", type=int, default=9)
parser.add_argument(
    "--stage-cache-min-hold",
    type=int,
    default=0,
    help="Require this many consecutive strict handoff frames before export.",
)
parser.add_argument("--max-stage-cache-rows", type=int, default=4096)
parser.add_argument(
    "--bundle",
    help="Optional reset bundle override for upstream-distribution evaluation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from skrl.utils.runner.torch import Runner

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    if not args_cli.zero_actions and not args_cli.checkpoint:
        raise ValueError("--checkpoint is required unless --zero-actions is used")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.bundle:
        env_cfg.bundle_path = str(Path(args_cli.bundle).resolve())
    agent_cfg["seed"] = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    direct = env.unwrapped
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(wrapped, agent_cfg)
    checkpoint = (
        Path(args_cli.checkpoint).resolve()
        if args_cli.checkpoint
        else None
    )
    if checkpoint is not None:
        runner.agent.load(str(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    observations, _ = wrapped.reset()
    states = wrapped.state()
    contact_sum = torch.zeros(5, device=direct.device)
    reward_sum = 0.0
    reward_count = 0
    nonfinite_env_steps = 0
    max_penetration_mm = 0.0
    max_peak_force_n = 0.0
    metric_keys = (
        "metric/middle_clear_ratio",
        "metric/ring_clear_ratio",
        "metric/middle_ring_clear_ratio",
        "metric/p5_partial_ratio",
        "metric/p5_instantaneous_ratio",
        "metric/stage_eligible_ratio",
        "metric/stage_hold_ms",
        "metric/p5_hold_ms",
        "metric/middle_rmse_rad",
        "metric/ring_rmse_rad",
        "metric/retract_rmse_rad",
        "metric/pinch_rmse_rad",
        "metric/pen_drift_mm",
        "metric/axis_drift_deg",
        "metric/angular_speed_rad_s",
        "metric/thumb_index_opposite_ratio",
        "metric/thumb_index_opposite_cos",
        "metric/thumb_index_axial_separation_mm",
    )
    metric_sums = {key: 0.0 for key in metric_keys}
    metric_samples = 0
    stage_cache_chunks = []
    stage_cache_recorded = torch.zeros(
        direct.num_envs, dtype=torch.bool, device=direct.device
    )
    trace_saved = False
    saved_trace = None
    best_diagnostic_trace = None
    best_diagnostic_score = (-1, -1)
    if args_cli.trajectory_output:
        trace_horizon = direct.max_episode_length + 1
        trace_qpos = torch.zeros(
            (direct.num_envs, trace_horizon, 27),
            dtype=torch.float32,
            device=direct.device,
        )
        trace_qvel = torch.zeros(
            (direct.num_envs, trace_horizon, 26),
            dtype=torch.float32,
            device=direct.device,
        )
        trace_action = torch.zeros(
            (direct.num_envs, trace_horizon, 20),
            dtype=torch.float32,
            device=direct.device,
        )
        trace_phase = torch.zeros(
            (direct.num_envs, trace_horizon),
            dtype=torch.int64,
            device=direct.device,
        )
        trace_penetration = torch.zeros(
            (direct.num_envs, trace_horizon),
            dtype=torch.float32,
            device=direct.device,
        )
        trace_age = torch.zeros(
            direct.num_envs, dtype=torch.long, device=direct.device
        )
    handoff_component_names = (
        "triad_contact",
        "position",
        "axis",
        "angular_speed",
        "linear_speed",
        "target_pose",
        "middle_pose",
        "penetration",
        "peak_force",
        "pinky_clear",
    )
    handoff_component_steps = torch.zeros(
        len(handoff_component_names),
        dtype=torch.long,
        device=direct.device,
    )
    post_eligible_failure_counts = torch.zeros_like(
        handoff_component_steps
    )
    previous_handoff_eligible = torch.zeros(
        direct.num_envs, dtype=torch.bool, device=direct.device
    )
    handoff_eligible_seen = torch.zeros_like(previous_handoff_eligible)
    handoff_hold_run = torch.zeros(
        direct.num_envs, dtype=torch.long, device=direct.device
    )
    handoff_best_hold_run = torch.zeros_like(handoff_hold_run)
    handoff_phase_steps = torch.zeros(
        (), dtype=torch.long, device=direct.device
    )
    handoff_axis_rate_sum = torch.zeros(
        (), dtype=torch.float32, device=direct.device
    )
    handoff_axis_rate_samples = torch.zeros(
        (), dtype=torch.long, device=direct.device
    )
    internal_gate_names = (
        "training_active",
        "safe",
        "transfer_contact",
        "transfer_pose",
        "handoff_speed",
        "final_aux_pose",
    )
    external_internal_mismatch_steps = torch.zeros(
        len(internal_gate_names), dtype=torch.long, device=direct.device
    )
    external_eligible_internal_false = torch.zeros(
        (), dtype=torch.long, device=direct.device
    )
    steps = 0

    while (
        int(direct.completed_p5_episodes.item()) < args_cli.episodes
        and steps < args_cli.max_steps
    ):
        with torch.inference_mode():
            if args_cli.zero_actions:
                actions = torch.zeros(
                    (direct.num_envs, 20), device=direct.device
                )
            else:
                outputs = runner.agent.act(
                    observations,
                    states,
                    timestep=steps,
                    timesteps=args_cli.max_steps,
                )
                actions = outputs[-1].get("mean_actions", outputs[0])
            if args_cli.trajectory_output and not trace_saved:
                rows = torch.arange(direct.num_envs, device=direct.device)
                trace_step = trace_age.clamp_max(trace_horizon - 1)
                trace_qpos[rows, trace_step] = torch.cat(
                    (direct.hand_qpos, direct.pen_pos, direct.pen_quat),
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
                trace_phase[rows, trace_step] = direct.chain_phase_ids
                trace_penetration[rows, trace_step] = (
                    direct.episode_max_penetration
                )
                trace_age = (trace_age + 1).clamp_max(trace_horizon)
            observations, rewards, terminated, truncated, _ = wrapped.step(
                actions
            )
            states = wrapped.state()
            done = (terminated | truncated).reshape(-1)
            direct._compute_intermediate_values()
            if bool(getattr(direct.cfg, "sequential_closure", False)):
                final_phase = (
                    int(direct.cfg.training_start_phase)
                    + len(direct.cfg.closure_target_blends)
                    - 1
                )
                final_active = direct.chain_phase_ids == final_phase
                contacts = (
                    direct.contact_force_norm
                    > direct.cfg.p1_contact_force_n
                )
                final_level = final_phase - int(
                    direct.cfg.training_start_phase
                )
                final_unloading = final_level > int(
                    direct.cfg.thumb_wrap_stage_boundaries[-1]
                )
                triad_contact = (
                    contacts[:, :2].all(dim=-1)
                    if final_unloading
                    else contacts[:, :3].all(dim=-1)
                )
                position_error = torch.linalg.norm(
                    direct.pen_pos - direct.target_pen_pos, dim=-1
                )
                target_axis = quat_apply(
                    direct.target_pen_quat, direct.pen_x_axis
                )
                axis_dot = torch.sum(
                    direct.pen_axis * target_axis, dim=-1
                ).clamp(-1.0, 1.0)
                axis_error = torch.acos(axis_dot.abs())
                angular_speed = torch.linalg.norm(
                    direct.pen_angvel, dim=-1
                )
                linear_speed = torch.linalg.norm(
                    direct.pen_linvel, dim=-1
                )
                triad_error = torch.cat(
                    (
                        direct.hand_qpos[:, 0:8]
                        - direct.target_hand_qpos[:, 0:8],
                        direct.hand_qpos[:, 16:20]
                        - direct.target_hand_qpos[:, 16:20],
                    ),
                    dim=-1,
                )
                thumb_index_error = torch.cat(
                    (
                        direct.hand_qpos[:, 0:4]
                        - direct.target_hand_qpos[:, 0:4],
                        direct.hand_qpos[:, 16:20]
                        - direct.target_hand_qpos[:, 16:20],
                    ),
                    dim=-1,
                )
                transfer_error = (
                    thumb_index_error if final_unloading else triad_error
                )
                triad_rmse = torch.sqrt(
                    torch.mean(transfer_error.square(), dim=-1)
                )
                middle_rmse = torch.sqrt(
                    torch.mean(
                        (
                            direct.hand_qpos[:, 4:8]
                            - direct.target_hand_qpos[:, 4:8]
                        ).square(),
                        dim=-1,
                    )
                )
                component_ok = torch.stack(
                    (
                        triad_contact,
                        position_error
                        <= direct.cfg.closure_pen_position_tolerance_m,
                        axis_error
                        <= torch.deg2rad(
                            torch.full_like(
                                axis_error,
                                direct.cfg.closure_pen_axis_tolerance_deg,
                            )
                        ),
                        angular_speed
                        <= direct.cfg.thumb_wrap_handoff_max_angular_speed_rad_s,
                        linear_speed
                        <= direct.cfg.thumb_wrap_handoff_max_linear_speed_m_s,
                        triad_rmse
                        <= direct.cfg.closure_stage_rmse_thresholds[-1],
                        middle_rmse
                        <= direct.cfg.closure_final_middle_pose_rmse_rad,
                        direct.episode_max_penetration
                        <= direct.cfg.penetration_threshold,
                        direct.contact_force_norm.max(dim=-1).values
                        <= direct.cfg.p1_max_force_n,
                        ~contacts[:, 4],
                    ),
                    dim=-1,
                )
                handoff_eligible = final_active & component_ok.all(dim=-1)
                if hasattr(direct, "debug_closure_gate_components"):
                    internal_components = (
                        direct.debug_closure_gate_components
                    )
                    internal_eligible = direct.debug_closure_eligible
                    external_only = handoff_eligible & ~internal_eligible
                    external_eligible_internal_false += external_only.sum()
                    external_internal_mismatch_steps += (
                        external_only.unsqueeze(-1)
                        & ~internal_components
                    ).sum(dim=0)
                handoff_phase_steps += final_active.sum()
                handoff_component_steps += (
                    final_active.unsqueeze(-1) & component_ok
                ).sum(dim=0)
                post_eligible = previous_handoff_eligible & final_active
                post_eligible_failure_counts += (
                    post_eligible.unsqueeze(-1) & ~component_ok
                ).sum(dim=0)
                handoff_eligible_seen |= handoff_eligible
                handoff_hold_run = torch.where(
                    handoff_eligible,
                    handoff_hold_run + 1,
                    torch.zeros_like(handoff_hold_run),
                )
                handoff_best_hold_run = torch.maximum(
                    handoff_best_hold_run, handoff_hold_run
                )
                axis_rate = torch.linalg.norm(
                    torch.linalg.cross(
                        direct.pen_angvel, direct.pen_axis, dim=-1
                    ),
                    dim=-1,
                )
                handoff_axis_rate_sum += axis_rate[final_active].sum()
                handoff_axis_rate_samples += final_active.sum()
                previous_handoff_eligible = handoff_eligible & ~done
            if args_cli.trajectory_output and not trace_saved and torch.any(done):
                for env_id_tensor in torch.nonzero(done, as_tuple=False):
                    env_id = int(env_id_tensor.item())
                    length = int(trace_age[env_id].item())
                    if length <= 0:
                        continue
                    deepest_phase = int(
                        trace_phase[env_id, :length].max().item()
                    )
                    score = (deepest_phase, length)
                    if score <= best_diagnostic_score:
                        continue
                    best_diagnostic_score = score
                    best_diagnostic_trace = {
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
                        "deepest_phase": np.asarray(
                            deepest_phase, dtype=np.int64
                        ),
                    }
            if (
                args_cli.stage_cache_output
                and sum(len(chunk["source_qpos"]) for chunk in stage_cache_chunks)
                < args_cli.max_stage_cache_rows
            ):
                direct._compute_intermediate_values()
                cache_mask = (
                    (direct.chain_phase_ids == args_cli.stage_cache_phase)
                    & ~stage_cache_recorded
                    & ~done
                )
                if args_cli.stage_cache_min_hold > 0:
                    cache_mask &= (
                        handoff_hold_run
                        >= args_cli.stage_cache_min_hold
                    )
                cache_ids = torch.nonzero(
                    cache_mask, as_tuple=False
                ).squeeze(-1)
                remaining = args_cli.max_stage_cache_rows - sum(
                    len(chunk["source_qpos"])
                    for chunk in stage_cache_chunks
                )
                cache_ids = cache_ids[:remaining]
                if cache_ids.numel() > 0:
                    source_qpos = torch.cat(
                        (
                            direct.hand_qpos[cache_ids],
                            direct.pen_pos[cache_ids],
                            direct.pen_quat[cache_ids],
                        ),
                        dim=-1,
                    )
                    source_qvel = torch.cat(
                        (
                            direct.hand_qvel[cache_ids],
                            direct.pen_linvel[cache_ids],
                            direct.pen_angvel[cache_ids],
                        ),
                        dim=-1,
                    )
                    stage_cache_chunks.append(
                        {
                            "source_qpos": source_qpos.detach()
                            .cpu()
                            .numpy(),
                            "source_qvel": source_qvel.detach()
                            .cpu()
                            .numpy(),
                            "source_target_qpos": direct.policy_targets[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            "source_policy_action": direct.actions[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            "source_target_hand_qpos": direct.target_hand_qpos[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            "source_closure_origin_hand": direct.closure_origin_hand[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            "source_closure_brace_hand": direct.closure_brace_hand[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            "source_closure_origin_pen_pos": (
                                direct.closure_origin_pen_pos[cache_ids]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            "source_closure_origin_pen_quat": (
                                direct.closure_origin_pen_quat[cache_ids]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            "source_init_pen_pos": direct.init_pen_pos[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            "source_init_pen_axis": direct.init_pen_axis[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            "source_max_penetration_m": direct.episode_max_penetration[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            "source_contact_force_n": direct.contact_force_norm[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            "source_chain_phase_id": direct.chain_phase_ids[
                                cache_ids
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                        }
                    )
                    stage_cache_recorded[cache_ids] = True
            if (
                args_cli.trajectory_output
                and not trace_saved
                and hasattr(direct, "last_reset_success")
                and torch.any(direct.last_reset_success)
            ):
                env_id = int(
                    torch.nonzero(
                        direct.last_reset_success, as_tuple=False
                    )[0].item()
                )
                length = int(trace_age[env_id].item())
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
                    "seed": np.asarray(args_cli.seed, dtype=np.int64),
                    "checkpoint": np.asarray(
                        str(checkpoint) if checkpoint else "scripted_teacher"
                    ),
                    "handoff": np.asarray(
                        "same PhysX episode; no hand or pen state reset"
                    ),
                }
                trace_saved = True
            if args_cli.trajectory_output and not trace_saved:
                trace_age = torch.where(
                    done, torch.zeros_like(trace_age), trace_age
                )
            stage_cache_recorded = torch.where(
                done,
                torch.zeros_like(stage_cache_recorded),
                stage_cache_recorded,
            )
            direct._compute_intermediate_values()
            finite = (
                torch.isfinite(observations).all(dim=-1)
                & torch.isfinite(rewards)
                & torch.isfinite(direct.pen_pos).all(dim=-1)
                & torch.isfinite(direct.hand_qpos).all(dim=-1)
            )
            nonfinite_env_steps += int((~finite).sum().item())
            contact_sum += (
                direct.contact_force_norm > direct.cfg.p5_contact_force_n
            ).float().sum(dim=0)
            reward_sum += float(rewards.sum().item())
            reward_count += rewards.numel()
            max_penetration_mm = max(
                max_penetration_mm,
                float(direct.episode_max_penetration.max().item() * 1000.0),
            )
            max_peak_force_n = max(
                max_peak_force_n, float(direct.contact_force_norm.max().item())
            )
            log = direct.extras.get("log", {})
            if all(key in log for key in metric_keys):
                for key in metric_keys:
                    metric_sums[key] += float(log[key].item())
                metric_samples += 1
        steps += 1

    completed = int(direct.completed_p5_episodes.item())
    successes = int(direct.completed_p5_successes.item())
    closure_diagnostics = {}
    if hasattr(direct, "closure_stage_entries"):
        final_entries = int(
            direct.completed_final_stage_entries.item()
        )
        stage_steps = direct.closure_stage_step_counts.detach().cpu()
        stage_triad_steps = (
            direct.closure_stage_triad_step_counts.detach().cpu()
        )
        stage_finger_contact_steps = (
            direct.closure_stage_finger_contact_step_counts.detach().cpu()
        )
        stage_safe_steps = (
            direct.closure_stage_safe_step_counts.detach().cpu()
        )
        stage_safe_component_steps = (
            direct.closure_stage_safe_component_step_counts.detach().cpu()
        )
        stage_pose_steps = (
            direct.closure_stage_pose_step_counts.detach().cpu()
        )
        stage_contact_pose_steps = (
            direct.closure_stage_contact_pose_step_counts.detach().cpu()
        )
        stage_eligible_steps = (
            direct.closure_stage_eligible_step_counts.detach().cpu()
        )
        stage_target_min = (
            direct.closure_stage_min_target_rmse.detach().cpu()
        )
        stage_canonical_min = (
            direct.closure_stage_min_canonical_rmse.detach().cpu()
        )
        control_dt_ms = (
            direct.cfg.decimation * direct.cfg.sim.dt * 1000.0
        )
        closure_diagnostics = {
            "target_blends": list(direct.cfg.closure_target_blends),
            "stage_rmse_thresholds_rad": list(
                direct.cfg.closure_stage_rmse_thresholds
            ),
            "stage_entries": direct.closure_stage_entries.detach()
            .cpu()
            .tolist(),
            "stage_step_counts": stage_steps.tolist(),
            "stage_triad_contact_step_rates": (
                stage_triad_steps.float()
                / stage_steps.clamp_min(1).float()
            ).tolist(),
            "stage_finger_contact_step_rates": (
                stage_finger_contact_steps.float()
                / stage_steps.clamp_min(1).unsqueeze(-1).float()
            ).tolist(),
            "stage_safe_step_rates": (
                stage_safe_steps.float()
                / stage_steps.clamp_min(1).float()
            ).tolist(),
            "stage_safe_component_names": [
                "position",
                "axis",
                "angular_speed",
                "penetration",
                "peak_force",
            ],
            "stage_safe_component_step_rates": (
                stage_safe_component_steps.float()
                / stage_steps.clamp_min(1).unsqueeze(-1).float()
            ).tolist(),
            "stage_pose_step_rates": (
                stage_pose_steps.float()
                / stage_steps.clamp_min(1).float()
            ).tolist(),
            "stage_contact_pose_step_rates": (
                stage_contact_pose_steps.float()
                / stage_steps.clamp_min(1).float()
            ).tolist(),
            "stage_eligible_step_rates": (
                stage_eligible_steps.float()
                / stage_steps.clamp_min(1).float()
            ).tolist(),
            "stage_min_target_rmse_rad": [
                float(value) if torch.isfinite(value) else None
                for value in stage_target_min
            ],
            "stage_min_canonical_p1_rmse_rad": [
                float(value) if torch.isfinite(value) else None
                for value in stage_canonical_min
            ],
            "stage_min_pen_position_error_mm": [
                float(value * 1000.0) if torch.isfinite(value) else None
                for value in direct.closure_stage_min_position_error.detach().cpu()
            ],
            "stage_min_pen_axis_error_deg": [
                float(value * (180.0 / np.pi))
                if torch.isfinite(value)
                else None
                for value in direct.closure_stage_min_axis_error.detach().cpu()
            ],
            "stage_best_hold_ms": (
                direct.closure_stage_best_hold.detach().cpu().float()
                * control_dt_ms
            ).tolist(),
            "completed_final_stage_entries": final_entries,
            "final_triad_seen_rate": (
                int(direct.completed_final_triad_seen.item())
                / max(final_entries, 1)
            ),
            "final_eligible_seen_rate": (
                int(direct.completed_final_eligible_seen.item())
                / max(final_entries, 1)
            ),
            "mean_min_canonical_p1_rmse_rad": (
                float(
                    direct.completed_final_min_canonical_rmse_sum.item()
                )
                / max(final_entries, 1)
            ),
            "best_canonical_p1_rmse_rad": (
                float(
                    direct.completed_final_min_canonical_rmse_best.item()
                )
                if final_entries
                else None
            ),
            "mean_max_final_hold_ms": (
                float(direct.completed_final_max_hold_sum.item())
                * control_dt_ms
                / max(final_entries, 1)
            ),
            "best_final_hold_ms": (
                int(direct.completed_final_max_hold_best.item())
                * control_dt_ms
            ),
        }
        if hasattr(direct, "chain_phase_entries"):
            closure_diagnostics["chain_phase_entries"] = (
                direct.chain_phase_entries.detach().cpu().tolist()
            )
        for name in (
            "chain_drop_by_phase",
            "chain_drift_by_phase",
            "chain_penetration_by_phase",
        ):
            if hasattr(direct, name):
                closure_diagnostics[name] = (
                    getattr(direct, name).detach().cpu().tolist()
                )
    result = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "bundle": (
            str(Path(args_cli.bundle).resolve())
            if args_cli.bundle
            else str(env_cfg.bundle_path)
        ),
        "simulation_steps": steps,
        "episodes": completed,
        "successes": successes,
        "success_rate": successes / max(completed, 1),
        "stage_episodes": direct.completed_stage_episodes.detach().cpu().tolist(),
        "stage_successes": direct.completed_stage_successes.detach()
        .cpu()
        .tolist(),
        "stage_success_rates": (
            direct.completed_stage_successes.float()
            / direct.completed_stage_episodes.clamp_min(1).float()
        )
        .detach()
        .cpu()
        .tolist(),
        "mean_step_reward": reward_sum / max(reward_count, 1),
        "nonfinite_env_steps": nonfinite_env_steps,
        "step_contact_ratio": (
            contact_sum / max(steps * direct.num_envs, 1)
        ).cpu().tolist(),
        "done_reasons": {
            "drop": int(direct.p5_done_drop_count.item()),
            "drift": int(direct.p5_done_drift_count.item()),
            "penetration": int(direct.p5_done_penetration_count.item()),
            "invalid": int(direct.p5_done_invalid_count.item()),
        },
        "max_penetration_mm_observed": max_penetration_mm,
        "max_terminal_penetration_mm": float(
            direct.p5_done_max_penetration.item() * 1000.0
        ),
        "max_peak_force_n_observed": max_peak_force_n,
        "mean_metrics": {
            key.removeprefix("metric/"): value / max(metric_samples, 1)
            for key, value in metric_sums.items()
        },
        "closure_diagnostics": closure_diagnostics,
        "success_definition": (
            "stage-specific eligibility; hold >= "
            f"{direct.cfg.p5_hold_steps * direct.cfg.decimation * direct.cfg.sim.dt * 1000.0:.0f}ms"
        ),
    }
    if int(handoff_phase_steps.item()) > 0:
        phase_steps = int(handoff_phase_steps.item())
        strict_success_mask = (
            handoff_best_hold_run >= int(direct.cfg.final_hold_steps)
        )
        strict_success_ids = torch.nonzero(
            strict_success_mask, as_tuple=False
        ).squeeze(-1)
        top_count = min(16, direct.num_envs)
        top_hold, top_ids = torch.topk(
            handoff_best_hold_run, top_count
        )
        teacher_count = (
            int(direct.scripted_thumb_teacher_first.shape[0])
            if hasattr(direct, "scripted_thumb_teacher_first")
            else 0
        )
        result["handoff_hold_diagnostics"] = {
            "phase_steps": phase_steps,
            "eligible_envs": int(handoff_eligible_seen.sum().item()),
            "strict_success_envs": int(strict_success_mask.sum().item()),
            "strict_success_env_ids": [
                int(value) for value in strict_success_ids[:64].tolist()
            ],
            "strict_success_teacher_ids": [
                int(value) % teacher_count
                for value in strict_success_ids[:64].tolist()
            ]
            if teacher_count
            else [],
            "top_hold_envs": [
                {
                    "env_id": int(env_id),
                    "teacher_id": (
                        int(env_id) % teacher_count
                        if teacher_count
                        else None
                    ),
                    "hold_steps": int(hold),
                }
                for env_id, hold in zip(
                    top_ids.tolist(), top_hold.tolist(), strict=True
                )
            ],
            "strict_success_rate": float(
                (
                    handoff_best_hold_run
                    >= int(direct.cfg.final_hold_steps)
                ).float().mean().item()
            ),
            "best_hold_steps": int(handoff_best_hold_run.max().item()),
            "best_hold_ms": float(
                handoff_best_hold_run.max().item()
                * direct.cfg.decimation
                * direct.cfg.sim.dt
                * 1000.0
            ),
            "mean_axis_rate_rad_s": float(
                handoff_axis_rate_sum.item()
                / max(int(handoff_axis_rate_samples.item()), 1)
            ),
            "component_step_rates": {
                name: float(
                    handoff_component_steps[index].item() / phase_steps
                )
                for index, name in enumerate(handoff_component_names)
            },
            "post_eligible_failure_counts": {
                name: int(post_eligible_failure_counts[index].item())
                for index, name in enumerate(handoff_component_names)
            },
            "external_eligible_internal_false_steps": int(
                external_eligible_internal_false.item()
            ),
            "internal_gate_failures_on_external_eligible": {
                name: int(
                    external_internal_mismatch_steps[index].item()
                )
                for index, name in enumerate(internal_gate_names)
            },
        }
    trajectory_kind = None
    if (
        args_cli.trajectory_output
        and saved_trace is None
        and best_diagnostic_trace is not None
    ):
        saved_trace = best_diagnostic_trace
        trajectory_kind = "deepest_incomplete_episode"
    elif saved_trace is not None:
        trajectory_kind = "successful_episode"
    if args_cli.trajectory_output and saved_trace is not None:
        saved_trace["control_dt_s"] = np.asarray(
            direct.cfg.decimation * direct.cfg.sim.dt,
            dtype=np.float32,
        )
        saved_trace["seed"] = np.asarray(args_cli.seed, dtype=np.int64)
        saved_trace["checkpoint"] = np.asarray(str(checkpoint))
        saved_trace["handoff"] = np.asarray(
            "same PhysX episode; no hand or pen state reset"
        )
        saved_trace["trajectory_kind"] = np.asarray(trajectory_kind)
        trajectory_output = Path(args_cli.trajectory_output)
        if not trajectory_output.is_absolute():
            trajectory_output = (
                Path(__file__).resolve().parents[1] / trajectory_output
            )
        trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trajectory_output, **saved_trace)
        result["trajectory_output"] = str(trajectory_output)
        result["trajectory_kind"] = trajectory_kind
    if args_cli.stage_cache_output and stage_cache_chunks:
        stage_cache_output = Path(args_cli.stage_cache_output)
        if not stage_cache_output.is_absolute():
            stage_cache_output = (
                Path(__file__).resolve().parents[1] / stage_cache_output
            )
        stage_cache_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: np.concatenate(
                [chunk[key] for chunk in stage_cache_chunks], axis=0
            )
            for key in stage_cache_chunks[0]
        }
        count = len(payload["source_qpos"])
        payload["target_qpos"] = payload["source_qpos"].copy()
        payload["target_qvel"] = payload["source_qvel"].copy()
        payload["phase_ids"] = np.zeros(count, dtype=np.int64)
        payload["source_teacher_actions"] = np.zeros(
            (count, 0, 20), dtype=np.float32
        )
        payload["seed"] = np.asarray(args_cli.seed, dtype=np.int64)
        payload["source"] = np.asarray(
            "same-episode continuous-chain phase entry"
        )
        np.savez_compressed(stage_cache_output, **payload)
        result["stage_cache_output"] = str(stage_cache_output)
        result["stage_cache_rows"] = count
        result["stage_cache_min_hold"] = (
            args_cli.stage_cache_min_hold
        )
    output = Path(args_cli.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
