from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Smoke-test the P1-to-P2 primitive environment.")
parser.add_argument("--task", default="Dex5-P1P2-C4-Direct-v0")
parser.add_argument("--mode", choices=("zero", "random", "scripted"), default="zero")
parser.add_argument("--split", choices=("train", "validation"), default="train")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bundle-path", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument(
    "--support-actions",
    type=float,
    nargs=3,
    metavar=("THUMB", "INDEX", "MIDDLE"),
    default=(0.0, 0.0, 0.0),
)
parser.add_argument(
    "--teacher-support-actions",
    type=float,
    nargs=3,
    metavar=("THUMB", "INDEX", "MIDDLE"),
    default=None,
    help="Override the internal teacher support actions without adding policy residuals.",
)
parser.add_argument("--ring-fraction", type=float, default=None)
parser.add_argument("--ring-joint-fractions", type=float, nargs=4, default=None)
parser.add_argument("--teacher-c3-sequence-tolerance-rad", type=float, default=None)
parser.add_argument("--teacher-c3-final-action-scale", type=float, default=None)
parser.add_argument("--teacher-brake-distance-mm", type=float, default=None)
parser.add_argument("--fixed-reset-bank-id", type=int, default=-1)
parser.add_argument("--geometric-brake", action="store_true")
parser.add_argument("--brake-progress-margin", type=float, default=0.08)
parser.add_argument("--brake-distance-mm", type=float, default=11.0)
parser.add_argument("--seed", type=int, default=20260722)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def scripted_actions(env) -> torch.Tensor:
    actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    increment = 0.25 * env.cfg.action_scale
    ring_source = env.source_targets[:, 8:12]
    ring_final = env.p2_target_ring_qpos.unsqueeze(0).expand_as(ring_source)
    stage = int(env.cfg.curriculum_stage)
    if args.ring_joint_fractions is not None:
        target_fraction = torch.tensor(
            args.ring_joint_fractions, dtype=torch.float32, device=env.device
        ).unsqueeze(0)
    else:
        target_fraction = (
            float(args.ring_fraction)
            if args.ring_fraction is not None
            else float(env.cfg.stage_target_progress[stage])
        )
    ring_target = ring_source + target_fraction * (ring_final - ring_source)
    if args.geometric_brake:
        if not hasattr(env, "smoke_brake_latched"):
            env.smoke_brake_latched = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
            env.smoke_brake_targets = torch.zeros((env.num_envs, 4), device=env.device)
        reset_now = env.episode_length_buf == 0
        env.smoke_brake_latched[reset_now] = False
        _, progress, target_distance, _ = env._ring_geometry()
        should_brake = (
            (progress >= float(env.cfg.stage_target_progress[stage]) - args.brake_progress_margin)
            & (target_distance <= args.brake_distance_mm / 1000.0)
        )
        newly_braked = should_brake & ~env.smoke_brake_latched
        env.smoke_brake_targets[newly_braked] = env.hand_qpos[newly_braked, 8:12]
        env.smoke_brake_latched |= should_brake
        ring_target = torch.where(
            env.smoke_brake_latched.unsqueeze(-1), env.smoke_brake_targets, ring_target
        )
    ring_error = ring_target - env.policy_targets[:, 8:12]
    actions[:, 8:12] = torch.clamp(ring_error / increment, -1.0, 1.0)
    for value, joint_slice in zip(
        args.support_actions,
        (slice(1, 4), slice(5, 8), slice(17, 20)),
        strict=True,
    ):
        actions[:, joint_slice] = value
    return actions


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.fixed_reset_bank_id = args.fixed_reset_bank_id
    if args.bundle_path is not None:
        env_cfg.bundle_path = args.bundle_path
        env_cfg.use_all_bundle_rows = True
    if args.teacher_support_actions is not None:
        env_cfg.teacher_prior_support_actions = tuple(args.teacher_support_actions)
    if args.teacher_c3_sequence_tolerance_rad is not None:
        env_cfg.teacher_c3_sequence_joint_tolerance_rad = (
            args.teacher_c3_sequence_tolerance_rad
        )
    if args.teacher_c3_final_action_scale is not None:
        env_cfg.teacher_c3_final_action_scale = args.teacher_c3_final_action_scale
    if args.teacher_brake_distance_mm is not None:
        env_cfg.teacher_prior_brake_distance_m = args.teacher_brake_distance_mm / 1000.0
    if args.split == "validation":
        env_cfg.split_ids_path = str(
            PROJECT_ROOT
            / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/P1_P2_validation_ids.npy"
        )
        env_cfg.split_name = "validation"
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    observations, _ = env.reset()
    observation = observations["policy"]
    expected_observation_shape = (args.num_envs, env_cfg.observation_space)
    if tuple(observation.shape) != expected_observation_shape:
        raise RuntimeError(
            f"Observation shape {tuple(observation.shape)} != {expected_observation_shape}"
        )

    with np.load(
        PROJECT_ROOT
        / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
        "P1_P2_reset_bank_phase1最终validated自复验_validated.npz"
    ) as frozen_bank:
        frozen_phase_ids = np.asarray(frozen_bank["phase_ids"], dtype=np.int64)
    if getattr(env_cfg, "use_all_bundle_rows", False):
        allowed_source_ids = None
    else:
        allowed_split_ids = np.asarray(np.load(env_cfg.split_ids_path, allow_pickle=False), dtype=np.int64)
        allowed_source_ids = set(
            allowed_split_ids[
                frozen_phase_ids[allowed_split_ids] == int(env_cfg.source_phase_id)
            ].tolist()
        )

    seen_global_ids: set[int] = set()
    invalid_observation_count = 0
    invalid_reward_count = 0
    terminated_count = 0
    timeout_count = 0
    max_penetration_m = 0.0
    max_position_drift_m = 0.0
    max_progress = -float("inf")
    final_progress_mean = 0.0
    final_target_distance_mean_m = 0.0
    max_teacher_phase_seen = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    teacher_phase_step_exposure = torch.zeros(4, dtype=torch.long, device=direct.device)
    gate_peak_ratios: dict[str, float] = {}
    traces: list[dict[str, float | int]] = []
    for step in range(args.steps):
        if args.mode == "scripted":
            actions = scripted_actions(direct)
        elif args.mode == "random":
            actions = 2.0 * torch.rand(
                (args.num_envs, direct.cfg.action_space), device=direct.device
            ) - 1.0
        else:
            actions = torch.zeros(
                (args.num_envs, direct.cfg.action_space), device=direct.device
            )
        observations, rewards, terminated, truncated, _ = env.step(actions)
        direct._compute_intermediate_values()
        _, progress, target_distance, _ = direct._ring_geometry()
        eligible, gate_metrics = direct._instantaneous_gate()
        if hasattr(direct, "teacher_c3_phase"):
            teacher_phase = direct.teacher_c3_phase
            max_teacher_phase_seen = torch.maximum(max_teacher_phase_seen, teacher_phase)
            teacher_phase_step_exposure += torch.bincount(
                teacher_phase.clamp(max=3), minlength=4
            )
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
            "triad": gate_metrics["triad"] > 0.5,
            "position_drift": gate_metrics["position_drift"]
            <= direct.cfg.p2_max_position_drift_m,
            "axis_drift": gate_metrics["axis_drift"]
            <= torch.deg2rad(
                torch.full_like(
                    gate_metrics["axis_drift"], direct.cfg.p2_max_axis_drift_deg
                )
            ),
            "angular_speed": gate_metrics["angular_speed"]
            <= direct.cfg.p2_max_angular_speed_rad_s,
            "linear_speed": gate_metrics["linear_speed"]
            <= direct.cfg.p2_max_linear_speed_m_s,
            "penetration": direct.episode_max_penetration
            <= direct.cfg.p2_max_penetration_m,
            "ring_force_share": gate_metrics["ring_force_share"]
            <= direct.cfg.p2_max_ring_force_share,
            "ring_force": direct.contact_force_norm[:, 3]
            <= direct.cfg.p2_max_ring_force_n,
            "pinky_force": direct.contact_force_norm[:, 4]
            <= direct.cfg.p2_max_pinky_force_n,
            "not_failed": gate_metrics["failed"] < 0.5,
        }
        gate_checks["geometry_triad"] = (
            gate_checks["geometry"] & gate_checks["triad"]
        )
        gate_checks["geometry_axis"] = (
            gate_checks["geometry"] & gate_checks["axis_drift"]
        )
        gate_checks["geometry_triad_axis"] = (
            gate_checks["geometry"]
            & gate_checks["triad"]
            & gate_checks["axis_drift"]
        )
        gate_checks["eligible"] = eligible
        for name, passed in gate_checks.items():
            ratio = float(passed.float().mean().item())
            gate_peak_ratios[name] = max(gate_peak_ratios.get(name, 0.0), ratio)
        seen_global_ids.update(direct.current_global_bank_ids.cpu().tolist())
        invalid_observation_count += int(
            (~torch.isfinite(observations["policy"])).any(dim=-1).sum().item()
        )
        invalid_reward_count += int((~torch.isfinite(rewards)).sum().item())
        terminated_count += int(terminated.sum().item())
        timeout_count += int(truncated.sum().item())
        max_penetration_m = max(
            max_penetration_m, float(direct.episode_max_penetration.max().item())
        )
        max_position_drift_m = max(
            max_position_drift_m,
            float(torch.linalg.norm(direct.pen_pos - direct.init_pen_pos, dim=-1).max().item()),
        )
        max_progress = max(max_progress, float(progress.max().item()))
        final_progress_mean = float(progress.mean().item())
        final_target_distance_mean_m = float(target_distance.mean().item())
        traces.append(
            {
                "step": step + 1,
                "eligible_ratio": float(eligible.float().mean().item()),
                "hold_steps_mean": float(direct.p1_p2_hold_count.float().mean().item()),
                "ring_progress_mean": float(progress.mean().item()),
                "ring_target_distance_mm_mean": float(target_distance.mean().item() * 1000.0),
                "triad_ratio": float(gate_metrics["triad"].mean().item()),
                "position_drift_mm_mean": float(gate_metrics["position_drift"].mean().item() * 1000.0),
                "axis_drift_deg_mean": float(
                    gate_metrics["axis_drift"].mean().item() * 180.0 / torch.pi
                ),
                "angular_speed_rad_s_mean": float(gate_metrics["angular_speed"].mean().item()),
                "linear_speed_m_s_mean": float(gate_metrics["linear_speed"].mean().item()),
                "ring_force_share_mean": float(gate_metrics["ring_force_share"].mean().item()),
                "failed_ratio": float(gate_metrics["failed"].mean().item()),
                "teacher_c3_phase_mean": float(
                    direct.teacher_c3_phase.float().mean().item()
                ) if hasattr(direct, "teacher_c3_phase") else 0.0,
                "teacher_c3_final_phase_ratio": float(
                    (direct.teacher_c3_phase >= 3).float().mean().item()
                ) if hasattr(direct, "teacher_c3_phase") else 0.0,
            }
        )

    leaked_ids = [] if allowed_source_ids is None else sorted(seen_global_ids - allowed_source_ids)
    result = {
        "task": args.task,
        "mode": args.mode,
        "split_name": env_cfg.split_name,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "support_actions_thumb_index_middle": list(args.support_actions),
        "teacher_support_actions_thumb_index_middle": (
            list(env_cfg.teacher_prior_support_actions)
            if env_cfg.teacher_prior_enabled
            else None
        ),
        "teacher_c3_sequence_tolerance_rad": float(
            env_cfg.teacher_c3_sequence_joint_tolerance_rad
        ),
        "ring_fraction": args.ring_fraction,
        "ring_joint_fractions": args.ring_joint_fractions,
        "fixed_reset_bank_id": args.fixed_reset_bank_id,
        "geometric_brake": args.geometric_brake,
        "control_dt_s": env_cfg.decimation * env_cfg.sim.dt,
        "observation_shape": list(observation.shape),
        "seen_global_bank_ids": len(seen_global_ids),
        "test_or_wrong_phase_leak_count": len(leaked_ids),
        "leaked_ids": leaked_ids[:20],
        "invalid_observation_count": invalid_observation_count,
        "invalid_reward_count": invalid_reward_count,
        "terminated_count": terminated_count,
        "timeout_count": timeout_count,
        "completed_episodes": int(direct.completed_p1_p2_episodes.item()),
        "completed_successes": int(direct.completed_p1_p2_successes.item()),
        "completed_success_rate": float(
            direct.completed_p1_p2_successes.item()
            / max(int(direct.completed_p1_p2_episodes.item()), 1)
        ),
        "completed_drop_count": int(direct.completed_drop_count.item()),
        "completed_invalid_count": int(direct.completed_invalid_count.item()),
        "completed_max_penetration_mm": float(
            direct.completed_max_penetration.item() * 1000.0
        ),
        "observed_max_penetration_mm": max_penetration_m * 1000.0,
        "observed_max_position_drift_mm": max_position_drift_m * 1000.0,
        "max_ring_progress": max_progress,
        "final_ring_progress_mean": final_progress_mean,
        "final_ring_target_distance_mean_mm": final_target_distance_mean_m * 1000.0,
        "teacher_c3_phase_counts": (
            torch.bincount(direct.teacher_c3_phase, minlength=4).cpu().tolist()
            if hasattr(direct, "teacher_c3_phase")
            else None
        ),
        "teacher_c3_max_phase_seen_counts": (
            torch.bincount(max_teacher_phase_seen, minlength=4).cpu().tolist()
            if hasattr(direct, "teacher_c3_phase")
            else None
        ),
        "teacher_c3_phase_step_exposure": (
            teacher_phase_step_exposure.cpu().tolist()
            if hasattr(direct, "teacher_c3_phase")
            else None
        ),
        "gate_peak_ratios": gate_peak_ratios,
        "traces": traces,
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_SMOKE=" + json.dumps(result, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
