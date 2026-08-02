from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="CEM audit of short-horizon P3 support-handoff reachability."
)
parser.add_argument(
    "--task", default="Dex5-PenSpin-Oracle-P3FixedHorizon-Direct-v0"
)
parser.add_argument("--states", type=int, default=12)
parser.add_argument("--bank-ids", type=int, nargs="+")
parser.add_argument("--population", type=int, default=64)
parser.add_argument("--iterations", type=int, default=10)
parser.add_argument("--elite-fraction", type=float, default=0.20)
parser.add_argument("--horizons", type=int, nargs="+", default=(6, 12, 18))
parser.add_argument("--waypoints", type=int, default=3)
parser.add_argument("--initial-std-rad", type=float, default=0.06)
parser.add_argument("--max-offset-rad", type=float, default=0.18)
parser.add_argument(
    "--action-mode", choices=("all", "tim_only"), default="all"
)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TRANSFER_FORCE_N = 0.03
MIN_TOTAL_FORCE_N = 0.06
MAX_TOTAL_FORCE_N = 6.0
MAX_HEIGHT_LOSS_M = 0.005
MAX_FINAL_HEIGHT_LOSS_M = 0.002
MIN_AXIS_ALIGNMENT = 0.50
MIN_TIM_OVERLAP = 0.80
CONTACT_REBUILD_STEPS = 2
TIM_ACTION_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19)


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(1, indices.unsqueeze(-1)).squeeze(-1)


def _rollout(
    env,
    direct,
    plans: torch.Tensor,
    forced_bank_ids: torch.Tensor,
    horizon: int,
) -> dict[str, torch.Tensor]:
    direct.forced_bank_ids = forced_bank_ids
    env.reset()
    hold_action = torch.zeros(
        (direct.num_envs, direct.cfg.action_space), device=direct.device
    )
    for _ in range(CONTACT_REBUILD_STEPS):
        env.step(hold_action)
    # Contact sensors are not refreshed on the reset-return frame. Start the
    # handoff clock only after the controlled R1 contact-rebuild steps.
    direct.episode_release_seen.zero_()
    direct.episode_recontact_seen.zero_()
    direct.episode_release_count.zero_()
    direct.episode_recontact_count.zero_()
    direct.episode_tim_overlap_steps.zero_()
    initial_pen_pos = direct.pen_pos.clone()
    initial_force = direct.contact_force_norm.clone()
    dominant = torch.argmax(initial_force[:, :3], dim=-1)
    dominant_initial_force = _gather(initial_force[:, :3], dominant)
    non_dominant_mask = torch.ones_like(initial_force, dtype=torch.bool)
    non_dominant_mask.scatter_(1, dominant.unsqueeze(-1), False)

    min_height_delta = torch.zeros(direct.num_envs, device=direct.device)
    min_total_force = torch.full(
        (direct.num_envs,), float("inf"), device=direct.device
    )
    max_total_force = torch.zeros(direct.num_envs, device=direct.device)
    max_new_support_gain = torch.zeros(direct.num_envs, device=direct.device)
    max_old_support_unload = torch.zeros(direct.num_envs, device=direct.device)
    best_transfer_pair_margin = torch.full(
        (direct.num_envs,), -float("inf"), device=direct.device
    )
    axis_sum = torch.zeros(direct.num_envs, device=direct.device)
    axis_steps = torch.zeros(
        direct.num_envs, dtype=torch.long, device=direct.device
    )
    unexpected_done = torch.zeros(
        direct.num_envs, dtype=torch.bool, device=direct.device
    )

    for step in range(horizon):
        waypoint = min(args.waypoints - 1, step * args.waypoints // horizon)
        desired_target = direct.source_targets + plans[:, waypoint]
        actions = torch.clamp(
            (desired_target - direct.processed_targets) / direct.cfg.action_scale,
            -1.0,
            1.0,
        )
        _, _, terminated, truncated, _ = env.step(actions)
        unexpected_done |= terminated | truncated

        force = direct.contact_force_norm
        total_force = force.sum(dim=-1)
        min_total_force = torch.minimum(min_total_force, total_force)
        max_total_force = torch.maximum(max_total_force, total_force)
        new_gain = torch.where(
            non_dominant_mask,
            force - initial_force,
            torch.full_like(force, -1.0e6),
        ).max(dim=-1).values
        max_new_support_gain = torch.maximum(max_new_support_gain, new_gain)
        old_force = _gather(force[:, :3], dominant)
        max_old_support_unload = torch.maximum(
            max_old_support_unload, dominant_initial_force - old_force
        )
        pair_margin = torch.minimum(
            new_gain / TRANSFER_FORCE_N - 1.0,
            (dominant_initial_force - old_force) / TRANSFER_FORCE_N - 1.0,
        )
        best_transfer_pair_margin = torch.maximum(
            best_transfer_pair_margin, pair_margin
        )

        height_delta = direct.pen_pos[:, 2] - initial_pen_pos[:, 2]
        min_height_delta = torch.minimum(min_height_delta, height_delta)
        angular_speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        moving = angular_speed >= direct.cfg.axis_alignment_min_angular_speed_rad_s
        alignment = (
            direct.cfg.target_rotation_direction
            * direct.pen_angvel[:, 2]
            / angular_speed.clamp_min(1.0e-6)
        )
        axis_sum += torch.where(moving, alignment, 0.0)
        axis_steps += moving.long()

    gait_by_finger = direct.episode_release_seen & direct.episode_recontact_seen
    tim_gait = _gather(gait_by_finger[:, :3], dominant)
    tim_overlap = direct.episode_tim_overlap_steps.float() / float(horizon)
    axis_alignment = axis_sum / axis_steps.clamp_min(1).float()
    height_margin = (MAX_HEIGHT_LOSS_M + min_height_delta) / MAX_HEIGHT_LOSS_M
    transfer_margin = max_new_support_gain / TRANSFER_FORCE_N - 1.0
    unload_margin = max_old_support_unload / TRANSFER_FORCE_N - 1.0
    support_margin = min_total_force / MIN_TOTAL_FORCE_N - 1.0
    force_ceiling_margin = (MAX_TOTAL_FORCE_N - max_total_force) / MAX_TOTAL_FORCE_N
    axis_margin = (axis_alignment - MIN_AXIS_ALIGNMENT) / MIN_AXIS_ALIGNMENT
    overlap_margin = (tim_overlap - MIN_TIM_OVERLAP) / (1.0 - MIN_TIM_OVERLAP)
    transition_margin = torch.where(
        tim_gait,
        torch.ones_like(tim_overlap),
        -torch.ones_like(tim_overlap),
    )
    handoff_margin = torch.stack(
        (
            transition_margin,
            transfer_margin,
            unload_margin,
            support_margin,
            force_ceiling_margin,
            best_transfer_pair_margin,
            height_margin,
            axis_margin,
            overlap_margin,
        ),
        dim=-1,
    ).min(dim=-1).values

    failure_types = direct.episode_failure_type_seen
    unsafe = (
        unexpected_done
        | direct.episode_ballistic_seen
        | failure_types[:, 0]
        | failure_types[:, 2]
        | failure_types[:, 3]
        | failure_types[:, 4]
        | failure_types[:, 5]
        | failure_types[:, 6]
    )
    feasible = (
        tim_gait
        & (max_new_support_gain >= TRANSFER_FORCE_N)
        & (max_old_support_unload >= TRANSFER_FORCE_N)
        & (best_transfer_pair_margin >= 0.0)
        & (min_total_force >= MIN_TOTAL_FORCE_N)
        & (max_total_force <= MAX_TOTAL_FORCE_N)
        & (min_height_delta >= -MAX_HEIGHT_LOSS_M)
        & (axis_steps >= 2)
        & (axis_alignment >= MIN_AXIS_ALIGNMENT)
        & (tim_overlap >= MIN_TIM_OVERLAP)
        & ~unsafe
    )
    final_height_delta = direct.pen_pos[:, 2] - initial_pen_pos[:, 2]
    height_recovered = final_height_delta >= -MAX_FINAL_HEIGHT_LOSS_M
    score_components = torch.stack(
        (
            transfer_margin.clamp(-2.0, 3.0),
            unload_margin.clamp(-2.0, 3.0),
            support_margin.clamp(-2.0, 3.0),
            force_ceiling_margin.clamp(-3.0, 1.0),
            best_transfer_pair_margin.clamp(-2.0, 3.0),
            height_margin.clamp(-3.0, 1.0),
            axis_margin.clamp(-3.0, 1.0),
            overlap_margin.clamp(-3.0, 1.0),
        ),
        dim=-1,
    )
    score = score_components.sum(dim=-1) + 3.0 * tim_gait.float()
    score += 10.0 * feasible.float() - 15.0 * unsafe.float()
    return {
        "score": score,
        "feasible": feasible,
        "handoff_margin": handoff_margin,
        "initial_dominant_TIM_index": dominant,
        "initial_dominant_TIM_force_n": dominant_initial_force,
        "tim_gait": tim_gait,
        "tim_overlap": tim_overlap,
        "new_support_gain_n": max_new_support_gain,
        "old_support_unload_n": max_old_support_unload,
        "min_total_force_n": min_total_force,
        "max_total_force_n": max_total_force,
        "transfer_pair_margin": best_transfer_pair_margin,
        "min_height_delta_m": min_height_delta,
        "final_height_delta_m": final_height_delta,
        "height_recovered": height_recovered,
        "axis_alignment": axis_alignment,
        "axis_steps": axis_steps,
        "ballistic": direct.episode_ballistic_seen.clone(),
        "contact_loss": failure_types[:, 0].clone(),
        "penetration": failure_types[:, 2].clone(),
        "drop": failure_types[:, 4].clone(),
        "throw": failure_types[:, 3].clone(),
        "drift": failure_types[:, 5].clone(),
        "invalid": failure_types[:, 6].clone() | unexpected_done,
    }


def _select_rows(direct) -> torch.Tensor:
    rows = direct.reset_rows_by_canonical[2]
    if args.bank_ids is not None:
        if len(args.bank_ids) != args.states:
            raise ValueError("--bank-ids count must equal --states")
        selected = torch.as_tensor(
            args.bank_ids, dtype=torch.long, device=direct.device
        )
        valid = torch.isin(selected, rows)
        if not bool(valid.all().item()):
            invalid = selected[~valid].tolist()
            raise ValueError(f"--bank-ids are not P3/{direct.cfg.reset_split}: {invalid}")
        return selected
    if args.states > rows.numel():
        raise ValueError(f"Requested {args.states} P3 rows, bank has {rows.numel()}")
    positions = torch.linspace(
        0, rows.numel() - 1, args.states, device=direct.device
    ).round().long()
    return rows[positions]


def _primary_failure(metrics: dict[str, float | bool]) -> str:
    if metrics["feasible"]:
        return "feasible"
    if metrics["invalid"]:
        return "invalid_or_unexpected_reset"
    if metrics["penetration"]:
        return "penetration"
    if metrics["throw"]:
        return "throw"
    if metrics["drift"]:
        return "drift"
    if metrics["ballistic"]:
        return "ballistic"
    if metrics["contact_loss"]:
        return "contact_loss"
    if metrics["min_height_delta_mm"] < -5.0:
        return "height_loss"
    if not metrics["tim_gait"]:
        return "no_TIM_transition"
    if metrics["new_support_gain_n"] < TRANSFER_FORCE_N:
        return "no_new_support_gain"
    if metrics["old_support_unload_n"] < TRANSFER_FORCE_N:
        return "no_old_support_unload"
    if metrics["transfer_pair_margin"] < 0.0:
        return "non_overlapping_load_transfer"
    if metrics["min_total_force_n"] < MIN_TOTAL_FORCE_N:
        return "support_force_collapse"
    if metrics["max_total_force_n"] > MAX_TOTAL_FORCE_N:
        return "contact_force_overload"
    if metrics["axis_steps"] < 2 or metrics["axis_alignment"] < MIN_AXIS_ALIGNMENT:
        return "axis_misaligned"
    if metrics["tim_overlap"] < MIN_TIM_OVERLAP:
        return "insufficient_TIM_overlap"
    return "joint_gate_miss"


def _to_record(metrics: dict[str, torch.Tensor], index: int) -> dict:
    record = {
        "score": float(metrics["score"][index].item()),
        "feasible": bool(metrics["feasible"][index].item()),
        "handoff_margin": float(metrics["handoff_margin"][index].item()),
        "initial_dominant_TIM_index": int(
            metrics["initial_dominant_TIM_index"][index].item()
        ),
        "initial_dominant_TIM_force_n": float(
            metrics["initial_dominant_TIM_force_n"][index].item()
        ),
        "tim_gait": bool(metrics["tim_gait"][index].item()),
        "tim_overlap": float(metrics["tim_overlap"][index].item()),
        "new_support_gain_n": float(metrics["new_support_gain_n"][index].item()),
        "old_support_unload_n": float(metrics["old_support_unload_n"][index].item()),
        "min_total_force_n": float(metrics["min_total_force_n"][index].item()),
        "max_total_force_n": float(metrics["max_total_force_n"][index].item()),
        "transfer_pair_margin": float(
            metrics["transfer_pair_margin"][index].item()
        ),
        "min_height_delta_mm": 1000.0
        * float(metrics["min_height_delta_m"][index].item()),
        "final_height_delta_mm": 1000.0
        * float(metrics["final_height_delta_m"][index].item()),
        "height_recovered": bool(metrics["height_recovered"][index].item()),
        "axis_alignment": float(metrics["axis_alignment"][index].item()),
        "axis_steps": int(metrics["axis_steps"][index].item()),
        "ballistic": bool(metrics["ballistic"][index].item()),
        "contact_loss": bool(metrics["contact_loss"][index].item()),
        "penetration": bool(metrics["penetration"][index].item()),
        "drop": bool(metrics["drop"][index].item()),
        "throw": bool(metrics["throw"][index].item()),
        "drift": bool(metrics["drift"][index].item()),
        "invalid": bool(metrics["invalid"][index].item()),
    }
    record["primary_failure"] = _primary_failure(record)
    return record


def main() -> None:
    audit_started_at = time.perf_counter()
    traceback_timeout_s = int(os.environ.get("DEX5_TRACEBACK_TIMEOUT_S", "0"))
    if traceback_timeout_s > 0:
        faulthandler.dump_traceback_later(traceback_timeout_s, repeat=True)
    if not args.horizons or min(args.horizons) < 1:
        raise ValueError("--horizons must contain positive control-step counts")
    num_envs = args.states * args.population
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = 20260730
    control_dt = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = (max(args.horizons) + 5) * control_dt
    print("O1_CEM_STAGE=creating_environment", flush=True)
    env = gym.make(args.task, cfg=cfg)
    print("O1_CEM_STAGE=environment_ready", flush=True)
    direct = env.unwrapped
    source_rows = _select_rows(direct)
    print("O1_CEM_STAGE=source_rows_selected", flush=True)
    forced_bank_ids = source_rows.repeat_interleave(args.population)
    action_mask = torch.ones(
        (direct.cfg.action_space,), dtype=torch.bool, device=direct.device
    )
    if args.action_mode == "tim_only":
        action_mask.zero_()
        action_mask[list(TIM_ACTION_INDICES)] = True
    generator = torch.Generator(device=direct.device).manual_seed(20260730)
    elite_count = min(
        args.population, max(2, round(args.population * args.elite_fraction))
    )
    horizon_reports = []
    all_best_by_horizon: list[list[dict]] = []

    for horizon in args.horizons:
        horizon_started_at = time.perf_counter()
        mean = torch.zeros(
            (args.states, args.waypoints, direct.cfg.action_space),
            device=direct.device,
        )
        std = torch.full_like(mean, args.initial_std_rad)
        best_scores = torch.full(
            (args.states,), -float("inf"), device=direct.device
        )
        best_plans = torch.zeros_like(mean)
        history = []
        for iteration in range(args.iterations):
            iteration_started_at = time.perf_counter()
            noise = torch.randn(
                (
                    args.states,
                    args.population,
                    args.waypoints,
                    direct.cfg.action_space,
                ),
                generator=generator,
                device=direct.device,
            )
            population = torch.clamp(
                mean.unsqueeze(1) + std.unsqueeze(1) * noise,
                -args.max_offset_rad,
                args.max_offset_rad,
            )
            if args.action_mode == "tim_only":
                population = population * action_mask
            metrics = _rollout(
                env,
                direct,
                population.flatten(0, 1),
                forced_bank_ids,
                horizon,
            )
            scores = metrics["score"].reshape(args.states, args.population)
            elite_ids = torch.topk(scores, elite_count, dim=1).indices
            gather_shape = elite_ids[:, :, None, None].expand(
                -1, -1, args.waypoints, direct.cfg.action_space
            )
            elites = torch.gather(population, 1, gather_shape)
            mean = 0.25 * mean + 0.75 * elites.mean(dim=1)
            std = torch.maximum(
                torch.full_like(std, 0.005),
                0.30 * std + 0.70 * elites.std(dim=1, unbiased=False),
            )
            iteration_best_scores, iteration_best_ids = scores.max(dim=1)
            improve = iteration_best_scores > best_scores
            best_scores = torch.where(improve, iteration_best_scores, best_scores)
            selected = population[
                torch.arange(args.states, device=direct.device), iteration_best_ids
            ]
            best_plans = torch.where(
                improve[:, None, None], selected, best_plans
            )
            history.append(
                {
                    "iteration": iteration + 1,
                    "best_score": float(best_scores.max().item()),
                    "population_feasible_rate": float(
                        metrics["feasible"].float().mean().item()
                    ),
                    "wall_time_s": time.perf_counter() - iteration_started_at,
                }
            )
            print(
                "O1_CEM_PROGRESS="
                + json.dumps(
                    {
                        "horizon_steps": horizon,
                        "iteration": iteration + 1,
                        "iterations": args.iterations,
                        "best_score": history[-1]["best_score"],
                        "iteration_wall_time_s": history[-1]["wall_time_s"],
                    }
                ),
                flush=True,
            )

        replay_metrics = _rollout(
            env,
            direct,
            best_plans.repeat_interleave(args.population, dim=0),
            forced_bank_ids,
            horizon,
        )
        state_records = []
        for state_index in range(args.states):
            clone_start = state_index * args.population
            clone_end = clone_start + args.population
            clone_recovered_strict = (
                replay_metrics["feasible"][clone_start:clone_end]
                & replay_metrics["height_recovered"][clone_start:clone_end]
            )
            clone_feasible = replay_metrics["feasible"][clone_start:clone_end]
            if bool(clone_recovered_strict.any().item()):
                representative_offset = int(
                    torch.nonzero(clone_recovered_strict, as_tuple=False)[0, 0].item()
                )
                representative_selection = "first_recovered_strict_clone"
            elif bool(clone_feasible.any().item()):
                representative_offset = int(
                    torch.nonzero(clone_feasible, as_tuple=False)[0, 0].item()
                )
                representative_selection = "first_strict_clone"
            else:
                representative_offset = int(
                    replay_metrics["score"][clone_start:clone_end].argmax().item()
                )
                representative_selection = "highest_score_clone"
            representative_index = clone_start + representative_offset
            record = _to_record(
                replay_metrics, representative_index
            )
            record["state_index"] = state_index
            record["bank_id"] = int(source_rows[state_index].item())
            record["representative_clone_index"] = representative_offset
            record["representative_selection"] = representative_selection
            record["clone_replay_feasible_rate"] = float(
                replay_metrics["feasible"][clone_start:clone_end].float().mean().item()
            )
            record["clone_replay_height_recovery_rate"] = float(
                replay_metrics["height_recovered"][clone_start:clone_end]
                .float()
                .mean()
                .item()
            )
            record["clone_replay_recovered_strict_rate"] = float(
                (
                    replay_metrics["feasible"][clone_start:clone_end]
                    & replay_metrics["height_recovered"][clone_start:clone_end]
                )
                .float()
                .mean()
                .item()
            )
            state_records.append(record)
        feasible_count = sum(record["feasible"] for record in state_records)
        horizon_reports.append(
            {
                "horizon_steps": horizon,
                "horizon_ms": 1000.0 * horizon * control_dt,
                "feasible_states": feasible_count,
                "feasible_rate": feasible_count / max(args.states, 1),
                "best_handoff_margin": max(
                    record["handoff_margin"] for record in state_records
                ),
                "median_handoff_margin": float(
                    torch.tensor(
                        [record["handoff_margin"] for record in state_records]
                    ).median().item()
                ),
                "wall_time_s": time.perf_counter() - horizon_started_at,
                "history": history,
                "states": state_records,
            }
        )
        all_best_by_horizon.append(state_records)

    selected_records = []
    for state_index in range(args.states):
        candidates = [records[state_index] for records in all_best_by_horizon]
        selected = max(candidates, key=lambda record: record["handoff_margin"])
        selected = dict(selected)
        selected["selected_horizon_steps"] = args.horizons[
            candidates.index(max(candidates, key=lambda record: record["handoff_margin"]))
        ]
        selected_records.append(selected)

    feasible_states = sum(
        any(records[state_index]["feasible"] for records in all_best_by_horizon)
        for state_index in range(args.states)
    )
    taxonomy: dict[str, int] = {}
    for record in selected_records:
        taxonomy[record["primary_failure"]] = (
            taxonomy.get(record["primary_failure"], 0) + 1
        )
    feasible_rate = feasible_states / max(args.states, 1)
    conclusion = (
        "B_POLICY_LEARNING_BOTTLENECK"
        if feasible_states > 0
        else "A_UNREACHABLE_WITHIN_AUDITED_BOUNDS"
    )
    report = {
        "status": "PASS",
        "artifact_role": "reachability_summary_only_not_teacher_or_training_data",
        "conclusion": conclusion,
        "task": args.task,
        "seed": 20260730,
        "states": args.states,
        "population": args.population,
        "iterations": args.iterations,
        "elite_fraction": args.elite_fraction,
        "waypoints": args.waypoints,
        "action_mode": args.action_mode,
        "active_action_indices": torch.nonzero(
            action_mask, as_tuple=False
        ).squeeze(-1).tolist(),
        "active_action_dimensions": int(action_mask.sum().item()),
        "initial_std_rad": args.initial_std_rad,
        "max_offset_rad": args.max_offset_rad,
        "horizons_steps": args.horizons,
        "control_dt_s": control_dt,
        "num_envs": num_envs,
        "wall_time_s": time.perf_counter() - audit_started_at,
        "bank_sha256": direct.cfg.expected_bundle_sha256,
        "source_bank_ids": [int(value) for value in source_rows.tolist()],
        "feasible_states": feasible_states,
        "feasible_rate": feasible_rate,
        "failure_taxonomy": taxonomy,
        "handoff_margin": {
            "best": max(record["handoff_margin"] for record in selected_records),
            "median": float(
                torch.tensor(
                    [record["handoff_margin"] for record in selected_records]
                ).median().item()
            ),
            "minimum": min(
                record["handoff_margin"] for record in selected_records
            ),
        },
        "feasibility_gate": {
            "contact_rebuild_steps_before_handoff": CONTACT_REBUILD_STEPS,
            "TIM_release_recontact_debounce_steps": 3,
            "new_support_gain_n": TRANSFER_FORCE_N,
            "old_support_unload_n": TRANSFER_FORCE_N,
            "minimum_total_contact_force_n": MIN_TOTAL_FORCE_N,
            "maximum_total_contact_force_n": MAX_TOTAL_FORCE_N,
            "simultaneous_load_transfer_required": True,
            "releasing_finger": "initial strongest TIM finger, selected per reset",
            "maximum_height_loss_mm": 1000.0 * MAX_HEIGHT_LOSS_M,
            "height_recovery_diagnostic_final_loss_mm": 1000.0
            * MAX_FINAL_HEIGHT_LOSS_M,
            "minimum_axis_alignment": MIN_AXIS_ALIGNMENT,
            "minimum_axis_moving_steps": 2,
            "minimum_TIM_overlap": MIN_TIM_OVERLAP,
            "no_ballistic_contact_loss_penetration_drop_or_invalid": True,
        },
        "selected_state_results": selected_records,
        "horizons": horizon_reports,
        "limitations": (
            "A zero feasible rate establishes practical non-reachability only within "
            "the audited reset sample, residual target bounds, waypoint parameterization, "
            "CEM budget, and 100-300 ms horizons; it is not a mathematical impossibility proof."
        ),
        "trajectory_exported": False,
        "teacher_generated": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("O1_8_HANDOFF_REACHABILITY=" + json.dumps({
        "conclusion": conclusion,
        "feasible_rate": feasible_rate,
        "failure_taxonomy": taxonomy,
        "handoff_margin": report["handoff_margin"],
    }, ensure_ascii=False), flush=True)
    del direct.forced_bank_ids
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
