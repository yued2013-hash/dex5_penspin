from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate an O1 privileged-oracle checkpoint.")
parser.add_argument("--task", default="Dex5-PenSpin-PrivilegedOracle-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--output", required=True)
parser.add_argument(
    "--terminate-on-ballistic",
    action="store_true",
    default=None,
    help="End an episode as soon as the debounced ballistic condition is detected.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import skrl  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


CANONICAL_IDS = ("P1", "P2", "P3", "P4", "P5")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FAILURE_TYPES = (
    "contact_loss",
    "ballistic",
    "penetration",
    "throw",
    "drop",
    "drift",
    "invalid",
)


def _rate(count: int, total: int) -> float:
    return float(count) / max(total, 1)


def _write_markdown(output: Path, report: dict) -> None:
    lines = [
        "# O1.1 Deterministic Stability and Gait Report",
        "",
        f"**Classification: `{report['classification']}`**",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Environments / steps: `{report['num_envs']} / {report['steps']}`",
        f"- Ballistic termination guard: `{report['terminate_on_ballistic']}`",
        f"- Completed episodes: `{report['completed_episodes']}`",
        f"- Raw gait / qualified gait: `{report['gait_event_rate']:.2%} / {report['qualified_gait_rate']:.2%}`",
        f"- Support handoff success: `{report['support_handoff_rate']:.2%}`",
        f"- Raw 180 deg / qualified 180 deg: `{report['over_180deg_rate']:.2%} / {report['qualified_over_180deg_rate']:.2%}`",
        f"- Failure / contact loss / ballistic: `{report['failure_rate']:.2%} / {report['contact_loss_rate']:.2%} / {report['ballistic_rate']:.2%}`",
        f"- Mean ballistic exposure: `{report['mean_ballistic_steps']:.2f}` control steps per completed episode",
        "",
        "Qualified means: directional axis alignment >= 0.70, no ballistic, no contact-loss termination, and no failure.",
        "",
        "## By Canonical",
        "",
        "| Canonical | Episodes | Raw gait | Qualified gait | Support handoff | Raw 180 | Qualified 180 | Axis align | Failure | Contact loss | Ballistic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for canonical_id, row in report["by_canonical"].items():
        lines.append(
            f"| {canonical_id} | {row['episodes']} | {row['gait_rate']:.2%} | "
            f"{row['qualified_gait_rate']:.2%} | {row['support_handoff_rate']:.2%} | "
            f"{row['over_180deg_rate']:.2%} | "
            f"{row['qualified_over_180deg_rate']:.2%} | "
            f"{row['mean_directional_axis_alignment']:.3f} | "
            f"{row['failure_rate']:.2%} | {row['failure_types']['contact_loss']['rate']:.2%} | "
            f"{row['failure_types']['ballistic']['rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Thumb And Height Recovery",
            "",
            "A thumb attempt is any post-release raw thumb contact that has not yet passed the three-step re-contact debounce. Height recovery requires a prior >=5 mm drop followed by return to within 2 mm of reset height.",
            "",
            "| Canonical | Thumb release | Attempt | Re-contact | Mean break step | Mean attempt step | Height drop | Height recovery | Min height delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for canonical_id, row in report["by_canonical"].items():
        recovery = row["recovery"]
        lines.append(
            f"| {canonical_id} | {recovery['thumb_release_rate']:.2%} | "
            f"{recovery['thumb_attempt_rate']:.2%} | "
            f"{recovery['thumb_recontact_rate']:.2%} | "
            f"{recovery['mean_first_thumb_break_step']:.1f} | "
            f"{recovery['mean_first_thumb_attempt_step']:.1f} | "
            f"{recovery['height_drop_rate']:.2%} | "
            f"{recovery['height_recovery_rate']:.2%} | "
            f"{recovery['mean_min_height_delta_mm']:.2f} mm |"
        )
    lines.extend(
        [
            "",
            "## Primary Failure Type",
            "",
            "| Canonical | contact_loss | ballistic | penetration | throw | drop | drift | invalid |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for canonical_id, row in report["by_canonical"].items():
        primary = row["primary_failure_types"]
        lines.append(
            f"| {canonical_id} | "
            + " | ".join(str(primary[name]["count"]) for name in FAILURE_TYPES)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Debounced Finger Events",
            "",
            "Each break and re-contact must remain stable for at least 3 control steps. Re-contact is counted as gait only after a prior break in the same episode.",
            "",
            "| Canonical | Finger | Break episodes | Re-contact episodes | Gait episodes | Break events | Re-contact events |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for canonical_id, row in report["by_canonical"].items():
        for finger, finger_row in row["finger_gait"].items():
            lines.append(
                f"| {canonical_id} | {finger} | {finger_row['break_episodes']} | "
                f"{finger_row['recontact_episodes']} | {finger_row['gait_episodes']} | "
                f"{finger_row['break_events']} | {finger_row['recontact_events']} |"
            )
    lines.extend(
        [
            "",
            "This is deterministic evaluation of the existing checkpoint. No PPO training, teacher generation, or O0 mutation is part of this report.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = 20260730
    if args.terminate_on_ballistic is not None:
        env_cfg.terminate_on_ballistic = args.terminate_on_ballistic
    experiment_cfg["seed"] = 20260730
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    experiment_cfg["trainer"]["close_environment_at_exit"] = False

    raw_env = gym.make(args.task, cfg=env_cfg)
    direct = raw_env.unwrapped
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, experiment_cfg)
    runner.agent.load(str(Path(args.checkpoint).resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    obs, _ = env.reset()
    states = env.state()
    finite = True
    sums = {
        "target_angular_velocity_rad_s": 0.0,
        "wrong_axis_angular_speed_rad_s": 0.0,
        "linear_speed_m_s": 0.0,
        "contact_count": 0.0,
        "mean_abs_action": 0.0,
        "support_force_z_n": 0.0,
        "total_contact_force_n": 0.0,
        "mechanical_power_w": 0.0,
    }
    max_penetration_mm = 0.0
    max_total_contact_force_n = 0.0
    sample_count = 0
    for _ in range(args.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, reward, _, _, _ = env.step(actions)
            states = env.state()
        finite &= bool(torch.isfinite(obs).all() and torch.isfinite(reward).all())
        target_w = direct.cfg.target_rotation_direction * direct.pen_angvel[:, 2]
        sums["target_angular_velocity_rad_s"] += float(target_w.sum().item())
        sums["wrong_axis_angular_speed_rad_s"] += float(
            torch.linalg.norm(direct.pen_angvel[:, :2], dim=-1).sum().item()
        )
        sums["linear_speed_m_s"] += float(
            torch.linalg.norm(direct.pen_linvel, dim=-1).sum().item()
        )
        sums["contact_count"] += float(
            (direct.contact_force_norm > direct.cfg.contact_force_threshold_n)
            .float()
            .sum()
            .item()
        )
        sums["mean_abs_action"] += float(
            torch.abs(actions).mean(dim=-1).sum().item()
        )
        support_force_z = -direct.contact_force_w.sum(dim=1)[:, 2]
        total_contact_force = direct.contact_force_norm.sum(dim=-1)
        applied_torque = direct.hand.data.applied_torque[:, direct.dex5_joint_ids]
        mechanical_power = torch.mean(
            torch.abs(applied_torque * direct.hand_qvel), dim=-1
        )
        sums["support_force_z_n"] += float(support_force_z.sum().item())
        sums["total_contact_force_n"] += float(total_contact_force.sum().item())
        sums["mechanical_power_w"] += float(mechanical_power.sum().item())
        max_total_contact_force_n = max(
            max_total_contact_force_n, float(total_contact_force.max().item())
        )
        max_penetration_mm = max(
            max_penetration_mm,
            float(direct.episode_max_penetration.max().item() * 1000.0),
        )
        sample_count += args.num_envs

    completed = int(direct.completed_episodes.item())
    mean_episode_steps = float(direct.completed_episode_length_sum.item()) / max(
        completed, 1
    )
    completed_by = direct.completed_by_canonical.detach().cpu().tolist()
    episode_length_sum_by = (
        direct.completed_episode_length_sum_by_canonical.detach().cpu().tolist()
    )
    gait_by = direct.completed_gait_by_canonical.detach().cpu().tolist()
    qualified_gait_by = (
        direct.completed_qualified_gait_by_canonical.detach().cpu().tolist()
    )
    over_180_by = direct.completed_over_180deg_by_canonical.detach().cpu().tolist()
    qualified_180_by = (
        direct.completed_qualified_over_180deg_by_canonical.detach().cpu().tolist()
    )
    failures_by = direct.completed_failure_by_canonical.detach().cpu().tolist()
    timeouts_by = direct.completed_timeout_by_canonical.detach().cpu().tolist()
    axis_aligned_by = (
        direct.completed_axis_aligned_by_canonical.detach().cpu().tolist()
    )
    axis_sum_by = (
        direct.completed_axis_alignment_sum_by_canonical.detach().cpu().tolist()
    )
    absolute_axis_sum_by = (
        direct.completed_absolute_axis_alignment_sum_by_canonical.detach().cpu().tolist()
    )
    no_contact_sum_by = (
        direct.completed_max_no_contact_steps_sum_by_canonical.detach().cpu().tolist()
    )
    ballistic_steps_sum_by = (
        direct.completed_ballistic_steps_sum_by_canonical.detach().cpu().tolist()
    )
    thumb_release_by = (
        direct.completed_thumb_release_by_canonical.detach().cpu().tolist()
    )
    thumb_attempt_by = (
        direct.completed_thumb_attempt_by_canonical.detach().cpu().tolist()
    )
    thumb_recontact_by = (
        direct.completed_thumb_recontact_by_canonical.detach().cpu().tolist()
    )
    height_drop_by = direct.completed_height_drop_by_canonical.detach().cpu().tolist()
    height_recovery_by = (
        direct.completed_height_recovery_by_canonical.detach().cpu().tolist()
    )
    support_handoff_by = (
        direct.completed_support_handoff_by_canonical.detach().cpu().tolist()
    )
    tim_gait_by = direct.completed_tim_gait_by_canonical.detach().cpu().tolist()
    tim_overlap_ratio_sum_by = (
        direct.completed_tim_overlap_ratio_sum_by_canonical.detach().cpu().tolist()
    )
    max_zero_tim_run_sum_by = (
        direct.completed_max_zero_tim_run_sum_by_canonical.detach().cpu().tolist()
    )
    min_height_delta_sum_by = (
        direct.completed_min_height_delta_sum_by_canonical.detach().cpu().tolist()
    )
    thumb_break_step_sum_by = (
        direct.completed_thumb_break_step_sum_by_canonical.detach().cpu().tolist()
    )
    thumb_attempt_step_sum_by = (
        direct.completed_thumb_attempt_step_sum_by_canonical.detach().cpu().tolist()
    )
    thumb_recontact_step_sum_by = (
        direct.completed_thumb_recontact_step_sum_by_canonical.detach().cpu().tolist()
    )
    release_episode = (
        direct.completed_release_episode_by_canonical_finger.detach().cpu().tolist()
    )
    recontact_episode = (
        direct.completed_recontact_episode_by_canonical_finger.detach().cpu().tolist()
    )
    gait_episode = (
        direct.completed_gait_episode_by_canonical_finger.detach().cpu().tolist()
    )
    release_events = (
        direct.completed_release_events_by_canonical_finger.detach().cpu().tolist()
    )
    recontact_events = (
        direct.completed_recontact_events_by_canonical_finger.detach().cpu().tolist()
    )
    failure_type_counts = (
        direct.completed_failure_type_by_canonical.detach().cpu().tolist()
    )
    primary_failure_counts = (
        direct.completed_primary_failure_by_canonical.detach().cpu().tolist()
    )
    by_canonical = {}
    for canonical_index, canonical_id in enumerate(CANONICAL_IDS):
        episodes = int(completed_by[canonical_index])
        failure_types = {
            name: {
                "count": int(failure_type_counts[canonical_index][failure_index]),
                "rate": _rate(
                    int(failure_type_counts[canonical_index][failure_index]), episodes
                ),
            }
            for failure_index, name in enumerate(FAILURE_TYPES)
        }
        primary_failure_types = {
            name: {
                "count": int(primary_failure_counts[canonical_index][failure_index]),
                "rate": _rate(
                    int(primary_failure_counts[canonical_index][failure_index]), episodes
                ),
            }
            for failure_index, name in enumerate(FAILURE_TYPES)
        }
        finger_gait = {
            finger: {
                "break_episodes": int(release_episode[canonical_index][finger_index]),
                "break_episode_rate": _rate(
                    int(release_episode[canonical_index][finger_index]), episodes
                ),
                "recontact_episodes": int(
                    recontact_episode[canonical_index][finger_index]
                ),
                "recontact_episode_rate": _rate(
                    int(recontact_episode[canonical_index][finger_index]), episodes
                ),
                "gait_episodes": int(gait_episode[canonical_index][finger_index]),
                "gait_episode_rate": _rate(
                    int(gait_episode[canonical_index][finger_index]), episodes
                ),
                "break_events": int(release_events[canonical_index][finger_index]),
                "recontact_events": int(
                    recontact_events[canonical_index][finger_index]
                ),
            }
            for finger_index, finger in enumerate(FINGER_NAMES)
        }
        recovery = {
            "thumb_release_episodes": int(thumb_release_by[canonical_index]),
            "thumb_release_rate": _rate(
                int(thumb_release_by[canonical_index]), episodes
            ),
            "thumb_attempt_episodes": int(thumb_attempt_by[canonical_index]),
            "thumb_attempt_rate": _rate(
                int(thumb_attempt_by[canonical_index]), episodes
            ),
            "thumb_recontact_episodes": int(thumb_recontact_by[canonical_index]),
            "thumb_recontact_rate": _rate(
                int(thumb_recontact_by[canonical_index]), episodes
            ),
            "mean_first_thumb_break_step": float(
                thumb_break_step_sum_by[canonical_index]
            )
            / max(int(thumb_release_by[canonical_index]), 1),
            "mean_first_thumb_attempt_step": float(
                thumb_attempt_step_sum_by[canonical_index]
            )
            / max(int(thumb_attempt_by[canonical_index]), 1),
            "mean_first_thumb_recontact_step": float(
                thumb_recontact_step_sum_by[canonical_index]
            )
            / max(int(thumb_recontact_by[canonical_index]), 1),
            "height_drop_episodes": int(height_drop_by[canonical_index]),
            "height_drop_rate": _rate(int(height_drop_by[canonical_index]), episodes),
            "height_recovery_episodes": int(height_recovery_by[canonical_index]),
            "height_recovery_rate": _rate(
                int(height_recovery_by[canonical_index]), episodes
            ),
            "height_recovery_given_drop_rate": _rate(
                int(height_recovery_by[canonical_index]),
                int(height_drop_by[canonical_index]),
            ),
            "mean_min_height_delta_mm": 1000.0
            * float(min_height_delta_sum_by[canonical_index])
            / max(episodes, 1),
        }
        by_canonical[canonical_id] = {
            "episodes": episodes,
            "mean_episode_steps": float(episode_length_sum_by[canonical_index])
            / max(episodes, 1),
            "gait_episodes": int(gait_by[canonical_index]),
            "gait_rate": _rate(int(gait_by[canonical_index]), episodes),
            "qualified_gait_episodes": int(qualified_gait_by[canonical_index]),
            "qualified_gait_rate": _rate(
                int(qualified_gait_by[canonical_index]), episodes
            ),
            "support_handoff_episodes": int(
                support_handoff_by[canonical_index]
            ),
            "support_handoff_rate": _rate(
                int(support_handoff_by[canonical_index]), episodes
            ),
            "tim_gait_episodes": int(tim_gait_by[canonical_index]),
            "tim_gait_rate": _rate(int(tim_gait_by[canonical_index]), episodes),
            "mean_TIM_overlap_ratio": float(
                tim_overlap_ratio_sum_by[canonical_index]
            )
            / max(episodes, 1),
            "mean_max_zero_TIM_run_steps": float(
                max_zero_tim_run_sum_by[canonical_index]
            )
            / max(episodes, 1),
            "over_180deg_episodes": int(over_180_by[canonical_index]),
            "over_180deg_rate": _rate(int(over_180_by[canonical_index]), episodes),
            "qualified_over_180deg_episodes": int(
                qualified_180_by[canonical_index]
            ),
            "qualified_over_180deg_rate": _rate(
                int(qualified_180_by[canonical_index]), episodes
            ),
            "axis_aligned_episodes": int(axis_aligned_by[canonical_index]),
            "axis_aligned_rate": _rate(
                int(axis_aligned_by[canonical_index]), episodes
            ),
            "mean_directional_axis_alignment": float(
                axis_sum_by[canonical_index]
            )
            / max(episodes, 1),
            "mean_absolute_axis_alignment": float(
                absolute_axis_sum_by[canonical_index]
            )
            / max(episodes, 1),
            "mean_max_no_contact_steps": float(no_contact_sum_by[canonical_index])
            / max(episodes, 1),
            "mean_ballistic_steps": float(ballistic_steps_sum_by[canonical_index])
            / max(episodes, 1),
            "failures": int(failures_by[canonical_index]),
            "failure_rate": _rate(int(failures_by[canonical_index]), episodes),
            "timeouts": int(timeouts_by[canonical_index]),
            "timeout_rate": _rate(int(timeouts_by[canonical_index]), episodes),
            "failure_types": failure_types,
            "primary_failure_types": primary_failure_types,
            "finger_gait": finger_gait,
            "recovery": recovery,
        }
    mean_metrics = {
        key: value / max(sample_count, 1) for key, value in sums.items()
    }
    qualified_gait = int(direct.completed_qualified_gait.item())
    qualified_180 = int(direct.completed_qualified_over_180deg.item())
    gait_rate = _rate(int(direct.completed_gait_events.item()), completed)
    qualified_gait_rate = _rate(qualified_gait, completed)
    over_180_rate = _rate(int(direct.completed_over_180deg.item()), completed)
    qualified_180_rate = _rate(qualified_180, completed)
    failure_rate = _rate(int(direct.completed_failures.item()), completed)
    stability_pass = (
        failure_rate <= 0.20
        and mean_metrics["wrong_axis_angular_speed_rad_s"]
        <= mean_metrics["target_angular_velocity_rad_s"]
        and qualified_gait_rate >= 0.10
    )
    report = {
        "status": "PASS" if finite else "FAIL",
        "classification": (
            "STABILITY_SMOKE_PASS" if finite and stability_pass else "REACHABILITY_ONLY"
        ),
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "terminate_on_ballistic": bool(direct.cfg.terminate_on_ballistic),
        "finite": finite,
        "completed_episodes": completed,
        "mean_episode_steps": mean_episode_steps,
        "gait_event_episodes": int(direct.completed_gait_events.item()),
        "gait_event_rate": gait_rate,
        "qualified_gait_episodes": qualified_gait,
        "qualified_gait_rate": qualified_gait_rate,
        "support_handoff_episodes": int(direct.completed_support_handoff.item()),
        "support_handoff_rate": _rate(
            int(direct.completed_support_handoff.item()), completed
        ),
        "tim_gait_event_episodes": int(direct.completed_tim_gait.item()),
        "tim_gait_event_rate": _rate(
            int(direct.completed_tim_gait.item()), completed
        ),
        "over_180deg_episodes": int(direct.completed_over_180deg.item()),
        "over_180deg_rate": over_180_rate,
        "qualified_over_180deg_episodes": qualified_180,
        "qualified_over_180deg_rate": qualified_180_rate,
        "failure_episodes": int(direct.completed_failures.item()),
        "throw_episodes": int(direct.completed_throws.item()),
        "contact_loss_episodes": int(direct.completed_contact_losses.item()),
        "penetration_failure_episodes": int(direct.completed_penetrations.item()),
        "ballistic_episodes": int(direct.completed_ballistic.item()),
        "ballistic_rate": _rate(int(direct.completed_ballistic.item()), completed),
        "mean_ballistic_steps": float(sum(ballistic_steps_sum_by))
        / max(completed, 1),
        "failure_rate": failure_rate,
        "contact_loss_rate": _rate(
            int(direct.completed_contact_losses.item()), completed
        ),
        "mean_metrics": mean_metrics,
        "recovery_totals": {
            "thumb_release_episodes": int(sum(thumb_release_by)),
            "thumb_attempt_episodes": int(sum(thumb_attempt_by)),
            "thumb_recontact_episodes": int(sum(thumb_recontact_by)),
            "height_drop_episodes": int(sum(height_drop_by)),
            "height_recovery_episodes": int(sum(height_recovery_by)),
        },
        "by_canonical": by_canonical,
        "failure_type_totals": {
            name: int(
                direct.completed_failure_type_by_canonical[:, failure_index]
                .sum()
                .item()
            )
            for failure_index, name in enumerate(FAILURE_TYPES)
        },
        "primary_failure_type_totals": {
            name: int(
                direct.completed_primary_failure_by_canonical[:, failure_index]
                .sum()
                .item()
            )
            for failure_index, name in enumerate(FAILURE_TYPES)
        },
        "max_penetration_mm": max_penetration_mm,
        "max_total_contact_force_n": max_total_contact_force_n,
        "sampled_reset_counts": dict(
            zip(
                ("P1", "P2", "P3", "P4", "P5"),
                direct.sampled_reset_counts.detach().cpu().tolist(),
                strict=True,
            )
        ),
        "curriculum_sampling": {
            "configured_candidate_bank_ids": [
                int(value)
                for value in getattr(
                    direct.cfg, "curriculum_candidate_bank_ids", ()
                )
            ],
            "sampled_candidate_count": int(
                direct.sampled_curriculum_candidate_count.item()
            ),
            "sampled_non_candidate_count": int(
                direct.sampled_curriculum_non_candidate_count.item()
            ),
            "final_candidate_probability": float(
                direct._curriculum_candidate_probability()
            ),
        },
        "bank_sha256": direct.cfg.expected_bundle_sha256,
        "reset_mixture": [float(value) for value in direct.cfg.reset_mixture],
        "objective_config": {
            "rotation_reward_scale": float(direct.cfg.rotation_reward_scale),
            "axis_aligned_rotation_reward_scale": float(
                direct.cfg.axis_aligned_rotation_reward_scale
            ),
            "axis_alignment_quality_reward_scale": float(
                direct.cfg.axis_alignment_quality_reward_scale
            ),
            "horizontal_reward_scale": float(direct.cfg.horizontal_reward_scale),
            "linear_speed_penalty_scale": float(
                direct.cfg.linear_speed_penalty_scale
            ),
            "wrong_axis_penalty_scale": float(direct.cfg.wrong_axis_penalty_scale),
            "ballistic_penalty_scale": float(direct.cfg.ballistic_penalty_scale),
            "high_angular_speed_penalty_scale": float(
                direct.cfg.high_angular_speed_penalty_scale
            ),
            "height_drop_penalty_scale": float(
                direct.cfg.height_drop_penalty_scale
            ),
            "contact_gate_positive_rotation": bool(
                direct.cfg.contact_gate_positive_rotation
            ),
            "include_temporal_support_observation": bool(
                direct.cfg.include_temporal_support_observation
            ),
            "observation_space": int(direct.cfg.observation_space),
            "termination_mode": str(direct.cfg.termination_mode),
            "terminal_failure_penalty_scale": float(
                direct.cfg.terminal_failure_penalty_scale
            ),
            "failure_grace_steps": int(direct.cfg.failure_grace_steps),
            "support_target_force_n": float(direct.cfg.support_target_force_n),
            "support_margin_penalty_scale": float(
                direct.cfg.support_margin_penalty_scale
            ),
            "support_drop_penalty_scale": float(
                direct.cfg.support_drop_penalty_scale
            ),
            "support_force_excess_penalty_scale": float(
                direct.cfg.support_force_excess_penalty_scale
            ),
            "recovery_height_deficit_penalty_scale": float(
                direct.cfg.recovery_height_deficit_penalty_scale
            ),
            "recovery_downward_speed_penalty_scale": float(
                direct.cfg.recovery_downward_speed_penalty_scale
            ),
            "recovery_progress_reward_scale": float(
                direct.cfg.recovery_progress_reward_scale
            ),
        },
        "gait_rule": "per-finger release and later re-contact, each stable for >=3 control steps",
        "support_handoff_rule": (
            "debounced thumb/index/middle gait; TIM >=2 overlap for >=80% of episode; no more than "
            "2 consecutive zero-TIM steps; no ballistic/failure; and no >=5 mm "
            "height loss unless height later recovers to within 2 mm"
        ),
        "ballistic_rule": "no contact for >=2 control steps and linear speed >0.20 m/s or angular speed >12 rad/s",
        "axis_alignment_rule": "mean directional cosine between angular velocity and target +Z axis while angular speed >=0.50 rad/s",
        "qualified_rule": "axis alignment >=0.70 with no ballistic, contact loss, or failure",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output, report)
    print("PRIVILEGED_ORACLE_EVAL=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
