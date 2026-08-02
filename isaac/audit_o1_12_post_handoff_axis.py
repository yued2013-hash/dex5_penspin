from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Two-stage in-memory CEM audit of post-handoff axis stabilization."
)
parser.add_argument(
    "--task", default="Dex5-PenSpin-Oracle-P3FixedHorizon-Direct-v0"
)
parser.add_argument("--bank-ids", type=int, nargs="+", default=(996, 1245, 1203, 1171))
parser.add_argument("--handoff-population", type=int, default=128)
parser.add_argument("--handoff-iterations", type=int, default=20)
parser.add_argument("--handoff-horizon", type=int, default=24)
parser.add_argument("--handoff-max-offset-rad", type=float, default=0.18)
parser.add_argument("--post-population", type=int, default=128)
parser.add_argument("--post-iterations", type=int, default=20)
parser.add_argument("--post-horizon", type=int, default=12)
parser.add_argument("--post-max-offset-rad", type=float, default=0.12)
parser.add_argument("--waypoints", type=int, default=3)
parser.add_argument("--elite-fraction", type=float, default=0.20)
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


CONTACT_REBUILD_STEPS = 2
TRANSFER_FORCE_N = 0.03
MIN_TOTAL_FORCE_N = 0.06
MAX_TOTAL_FORCE_N = 6.0
MAX_HEIGHT_LOSS_M = 0.005
MIN_AXIS_ALIGNMENT = 0.50
MIN_TIM_OVERLAP = 0.80
SEED = 20260731


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(1, indices.unsqueeze(-1)).squeeze(-1)


def _actions_for_target(direct, desired_target: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        (desired_target - direct.processed_targets) / direct.cfg.action_scale,
        -1.0,
        1.0,
    )


def _clear_handoff_events(direct) -> None:
    direct.episode_release_seen.zero_()
    direct.episode_recontact_seen.zero_()
    direct.episode_release_count.zero_()
    direct.episode_recontact_count.zero_()
    direct.episode_tim_overlap_steps.zero_()


def _reset_and_rebuild(env, direct, forced_bank_ids: torch.Tensor) -> tuple:
    direct.forced_bank_ids = forced_bank_ids
    env.reset()
    hold_action = torch.zeros(
        (direct.num_envs, direct.cfg.action_space), device=direct.device
    )
    for _ in range(CONTACT_REBUILD_STEPS):
        env.step(hold_action)
    _clear_handoff_events(direct)
    initial_pen_pos = direct.pen_pos.clone()
    initial_force = direct.contact_force_norm.clone()
    dominant = torch.argmax(initial_force[:, :3], dim=-1)
    dominant_initial_force = _gather(initial_force[:, :3], dominant)
    non_dominant_mask = torch.ones_like(initial_force, dtype=torch.bool)
    non_dominant_mask.scatter_(1, dominant.unsqueeze(-1), False)
    return (
        initial_pen_pos,
        initial_force,
        dominant,
        dominant_initial_force,
        non_dominant_mask,
    )


def _unsafe_flags(direct) -> torch.Tensor:
    failures = direct.episode_failure_type_seen
    return (
        direct.episode_ballistic_seen
        | failures[:, 0]
        | failures[:, 2]
        | failures[:, 3]
        | failures[:, 4]
        | failures[:, 5]
        | failures[:, 6]
    )


def _handoff_rollout(
    env,
    direct,
    plans: torch.Tensor,
    forced_bank_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    horizon = args.handoff_horizon
    (
        initial_pen_pos,
        initial_force,
        dominant,
        dominant_initial_force,
        non_dominant_mask,
    ) = _reset_and_rebuild(env, direct, forced_bank_ids)
    min_height_delta = torch.zeros(direct.num_envs, device=direct.device)
    min_total_force = torch.full(
        (direct.num_envs,), float("inf"), device=direct.device
    )
    max_total_force = torch.zeros(direct.num_envs, device=direct.device)
    max_new_support_gain = torch.zeros(direct.num_envs, device=direct.device)
    max_old_support_unload = torch.zeros(direct.num_envs, device=direct.device)
    best_pair_margin = torch.full(
        (direct.num_envs,), -float("inf"), device=direct.device
    )
    axis_sum = torch.zeros(direct.num_envs, device=direct.device)
    axis_steps = torch.zeros(
        direct.num_envs, dtype=torch.long, device=direct.device
    )
    first_transition_step = torch.full_like(axis_steps, -1)

    for step in range(horizon):
        waypoint = min(args.waypoints - 1, step * args.waypoints // horizon)
        desired_target = direct.source_targets + plans[:, waypoint]
        env.step(_actions_for_target(direct, desired_target))
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
        old_unload = dominant_initial_force - old_force
        max_old_support_unload = torch.maximum(max_old_support_unload, old_unload)
        best_pair_margin = torch.maximum(
            best_pair_margin,
            torch.minimum(
                new_gain / TRANSFER_FORCE_N - 1.0,
                old_unload / TRANSFER_FORCE_N - 1.0,
            ),
        )
        min_height_delta = torch.minimum(
            min_height_delta, direct.pen_pos[:, 2] - initial_pen_pos[:, 2]
        )
        speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        moving = speed >= direct.cfg.axis_alignment_min_angular_speed_rad_s
        alignment = (
            direct.cfg.target_rotation_direction
            * direct.pen_angvel[:, 2]
            / speed.clamp_min(1.0e-6)
        )
        axis_sum += torch.where(moving, alignment, 0.0)
        axis_steps += moving.long()
        gait = _gather(
            (direct.episode_release_seen & direct.episode_recontact_seen)[:, :3],
            dominant,
        )
        new_transition = gait & (first_transition_step < 0)
        first_transition_step = torch.where(
            new_transition,
            torch.full_like(first_transition_step, step + 1),
            first_transition_step,
        )

    tim_gait = first_transition_step >= 0
    tim_overlap = direct.episode_tim_overlap_steps.float() / float(horizon)
    axis_alignment = axis_sum / axis_steps.clamp_min(1).float()
    unsafe = _unsafe_flags(direct)
    transition_success = (
        tim_gait
        & (max_new_support_gain >= TRANSFER_FORCE_N)
        & (max_old_support_unload >= TRANSFER_FORCE_N)
        & (best_pair_margin >= 0.0)
        & (min_total_force >= MIN_TOTAL_FORCE_N)
        & (max_total_force <= MAX_TOTAL_FORCE_N)
        & (min_height_delta >= -MAX_HEIGHT_LOSS_M)
        & (tim_overlap >= MIN_TIM_OVERLAP)
        & ~unsafe
    )
    transfer_margin = max_new_support_gain / TRANSFER_FORCE_N - 1.0
    unload_margin = max_old_support_unload / TRANSFER_FORCE_N - 1.0
    support_margin = min_total_force / MIN_TOTAL_FORCE_N - 1.0
    force_ceiling_margin = (MAX_TOTAL_FORCE_N - max_total_force) / MAX_TOTAL_FORCE_N
    height_margin = (MAX_HEIGHT_LOSS_M + min_height_delta) / MAX_HEIGHT_LOSS_M
    axis_margin = (axis_alignment - MIN_AXIS_ALIGNMENT) / MIN_AXIS_ALIGNMENT
    overlap_margin = (tim_overlap - MIN_TIM_OVERLAP) / (1.0 - MIN_TIM_OVERLAP)
    score = (
        transfer_margin.clamp(-2.0, 3.0)
        + unload_margin.clamp(-2.0, 3.0)
        + support_margin.clamp(-2.0, 3.0)
        + force_ceiling_margin.clamp(-3.0, 1.0)
        + best_pair_margin.clamp(-2.0, 3.0)
        + height_margin.clamp(-3.0, 1.0)
        + axis_margin.clamp(-3.0, 1.0)
        + overlap_margin.clamp(-3.0, 1.0)
        + 3.0 * tim_gait.float()
        + 10.0 * transition_success.float()
        - 15.0 * unsafe.float()
    )
    return {
        "score": score,
        "transition_success": transition_success,
        "tim_gait": tim_gait,
        "transition_step": first_transition_step,
        "axis_alignment": axis_alignment,
        "axis_steps": axis_steps,
        "min_height_delta_m": min_height_delta,
        "min_total_force_n": min_total_force,
        "max_total_force_n": max_total_force,
        "tim_overlap": tim_overlap,
        "new_support_gain_n": max_new_support_gain,
        "old_support_unload_n": max_old_support_unload,
        "transfer_pair_margin": best_pair_margin,
        "unsafe": unsafe,
    }


def _post_rollout(
    env,
    direct,
    handoff_plans: torch.Tensor,
    post_plans: torch.Tensor,
    forced_bank_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    handoff_horizon = args.handoff_horizon
    post_horizon = args.post_horizon
    initial_pen_pos, _, dominant, _, _ = _reset_and_rebuild(
        env, direct, forced_bank_ids
    )
    transitioned = torch.zeros(
        direct.num_envs, dtype=torch.bool, device=direct.device
    )
    transition_step = torch.full(
        (direct.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    post_steps = torch.zeros_like(transition_step)
    post_base_target = direct.source_targets.clone()
    transition_height_delta = torch.zeros(direct.num_envs, device=direct.device)
    transition_axis_alignment = torch.zeros_like(transition_height_delta)
    transition_total_force = torch.zeros_like(transition_height_delta)
    transition_TIM_overlap = torch.zeros_like(transitioned)
    unsafe_at_transition = torch.zeros_like(transitioned)
    post_min_height_delta = torch.zeros_like(transition_height_delta)
    post_min_total_force = torch.full_like(transition_height_delta, float("inf"))
    post_max_total_force = torch.zeros_like(transition_height_delta)
    post_overlap_steps = torch.zeros_like(transition_step)
    post_axis_sum = torch.zeros_like(transition_height_delta)
    post_axis_steps = torch.zeros_like(transition_step)
    post_unsafe_seen = torch.zeros_like(transitioned)
    post_ballistic_seen = torch.zeros_like(transitioned)
    post_failure_seen = torch.zeros(
        (direct.num_envs, direct.episode_failure_type_seen.shape[1]),
        dtype=torch.bool,
        device=direct.device,
    )

    total_steps = handoff_horizon + post_horizon
    for step in range(total_steps):
        active_post = transitioned & (post_steps < post_horizon)
        finished_post = transitioned & (post_steps >= post_horizon)
        pre_step = min(step, handoff_horizon - 1)
        pre_waypoint = min(
            args.waypoints - 1,
            pre_step * args.waypoints // handoff_horizon,
        )
        desired_target = direct.source_targets + handoff_plans[:, pre_waypoint]
        if torch.any(active_post):
            post_waypoint = torch.clamp(
                post_steps * args.waypoints // post_horizon,
                max=args.waypoints - 1,
            )
            selected_offsets = post_plans[
                torch.arange(direct.num_envs, device=direct.device), post_waypoint
            ]
            post_target = post_base_target + selected_offsets
            desired_target = torch.where(
                active_post.unsqueeze(-1), post_target, desired_target
            )
        if torch.any(finished_post):
            final_target = post_base_target + post_plans[:, -1]
            desired_target = torch.where(
                finished_post.unsqueeze(-1), final_target, desired_target
            )
        env.step(_actions_for_target(direct, desired_target))

        gait = _gather(
            (direct.episode_release_seen & direct.episode_recontact_seen)[:, :3],
            dominant,
        )
        new_transition = gait & ~transitioned & (step < handoff_horizon)
        if torch.any(new_transition):
            transitioned |= new_transition
            transition_step = torch.where(
                new_transition,
                torch.full_like(transition_step, step + 1),
                transition_step,
            )
            post_base_target = torch.where(
                new_transition.unsqueeze(-1),
                direct.processed_targets,
                post_base_target,
            )
            transition_height_delta = torch.where(
                new_transition,
                direct.pen_pos[:, 2] - initial_pen_pos[:, 2],
                transition_height_delta,
            )
            speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
            instant_axis = (
                direct.cfg.target_rotation_direction
                * direct.pen_angvel[:, 2]
                / speed.clamp_min(1.0e-6)
            )
            transition_axis_alignment = torch.where(
                new_transition, instant_axis, transition_axis_alignment
            )
            total_force = direct.contact_force_norm.sum(dim=-1)
            contacts = (
                direct.contact_force_norm[:, :3]
                > direct.cfg.contact_force_threshold_n
            )
            transition_total_force = torch.where(
                new_transition, total_force, transition_total_force
            )
            transition_TIM_overlap = torch.where(
                new_transition,
                contacts.sum(dim=-1) >= 2,
                transition_TIM_overlap,
            )
            unsafe_at_transition = torch.where(
                new_transition, _unsafe_flags(direct), unsafe_at_transition
            )

        measure_post = active_post
        if torch.any(measure_post):
            height_delta = direct.pen_pos[:, 2] - initial_pen_pos[:, 2]
            post_min_height_delta = torch.where(
                measure_post,
                torch.minimum(post_min_height_delta, height_delta),
                post_min_height_delta,
            )
            total_force = direct.contact_force_norm.sum(dim=-1)
            post_min_total_force = torch.where(
                measure_post,
                torch.minimum(post_min_total_force, total_force),
                post_min_total_force,
            )
            post_max_total_force = torch.where(
                measure_post,
                torch.maximum(post_max_total_force, total_force),
                post_max_total_force,
            )
            contacts = (
                direct.contact_force_norm[:, :3]
                > direct.cfg.contact_force_threshold_n
            )
            overlap = contacts.sum(dim=-1) >= 2
            post_overlap_steps += (measure_post & overlap).long()
            speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
            moving = (
                speed >= direct.cfg.axis_alignment_min_angular_speed_rad_s
            ) & measure_post
            alignment = (
                direct.cfg.target_rotation_direction
                * direct.pen_angvel[:, 2]
                / speed.clamp_min(1.0e-6)
            )
            post_axis_sum += torch.where(moving, alignment, 0.0)
            post_axis_steps += moving.long()
            post_unsafe_seen |= measure_post & _unsafe_flags(direct)
            post_ballistic_seen |= measure_post & direct.episode_ballistic_seen
            post_failure_seen |= (
                measure_post.unsqueeze(-1) & direct.episode_failure_type_seen
            )
            post_steps += measure_post.long()

    post_axis_alignment = post_axis_sum / post_axis_steps.clamp_min(1).float()
    post_overlap = post_overlap_steps.float() / float(post_horizon)
    unsafe = post_unsafe_seen | unsafe_at_transition
    transition_success = (
        transitioned
        & ~unsafe_at_transition
        & (transition_height_delta >= -MAX_HEIGHT_LOSS_M)
        & (transition_total_force >= MIN_TOTAL_FORCE_N)
        & (transition_total_force <= MAX_TOTAL_FORCE_N)
        & transition_TIM_overlap
    )
    post_success = (
        transition_success
        & (post_steps >= post_horizon)
        & (post_axis_steps >= 2)
        & (post_axis_alignment >= MIN_AXIS_ALIGNMENT)
        & (post_min_height_delta >= -MAX_HEIGHT_LOSS_M)
        & (post_min_total_force >= MIN_TOTAL_FORCE_N)
        & (post_max_total_force <= MAX_TOTAL_FORCE_N)
        & (post_overlap >= MIN_TIM_OVERLAP)
        & ~unsafe
    )
    axis_margin = (post_axis_alignment - MIN_AXIS_ALIGNMENT) / MIN_AXIS_ALIGNMENT
    height_margin = (
        MAX_HEIGHT_LOSS_M + post_min_height_delta
    ) / MAX_HEIGHT_LOSS_M
    support_margin = post_min_total_force / MIN_TOTAL_FORCE_N - 1.0
    force_ceiling_margin = (
        MAX_TOTAL_FORCE_N - post_max_total_force
    ) / MAX_TOTAL_FORCE_N
    overlap_margin = (post_overlap - MIN_TIM_OVERLAP) / (1.0 - MIN_TIM_OVERLAP)
    score = (
        5.0 * transitioned.float()
        + 4.0 * transition_success.float()
        + 3.0 * axis_margin.clamp(-3.0, 2.0)
        + 2.0 * height_margin.clamp(-3.0, 1.0)
        + support_margin.clamp(-2.0, 2.0)
        + force_ceiling_margin.clamp(-3.0, 1.0)
        + overlap_margin.clamp(-3.0, 1.0)
        + 12.0 * post_success.float()
        - 15.0 * unsafe.float()
    )
    return {
        "score": score,
        "transition_success": transition_success,
        "transition_step": transition_step,
        "transition_height_delta_m": transition_height_delta,
        "transition_axis_alignment": transition_axis_alignment,
        "transition_total_force_n": transition_total_force,
        "transition_TIM_overlap": transition_TIM_overlap,
        "post_success": post_success,
        "post_axis_alignment": post_axis_alignment,
        "post_axis_steps": post_axis_steps,
        "post_min_height_delta_m": post_min_height_delta,
        "post_min_total_force_n": torch.where(
            transitioned, post_min_total_force, torch.zeros_like(post_min_total_force)
        ),
        "post_max_total_force_n": post_max_total_force,
        "post_TIM_overlap": post_overlap,
        "unsafe": unsafe,
        "ballistic": post_ballistic_seen,
        "failure_types": post_failure_seen,
    }


def _cem(
    rollout,
    env,
    direct,
    bank_ids: torch.Tensor,
    population: int,
    iterations: int,
    initial_std: float,
    max_offset: float,
    handoff_plans: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[dict],
]:
    states = bank_ids.numel()
    forced = bank_ids.repeat_interleave(population)
    mean = torch.zeros(
        (states, args.waypoints, direct.cfg.action_space), device=direct.device
    )
    std = torch.full_like(mean, initial_std)
    best_scores = torch.full((states,), -float("inf"), device=direct.device)
    best_plans = torch.zeros_like(mean)
    generator = torch.Generator(device=direct.device).manual_seed(SEED)
    elite_count = min(
        population, max(2, round(population * args.elite_fraction))
    )
    history = []
    for iteration in range(iterations):
        started = time.perf_counter()
        noise = torch.randn(
            (states, population, args.waypoints, direct.cfg.action_space),
            generator=generator,
            device=direct.device,
        )
        candidates = torch.clamp(
            mean.unsqueeze(1) + std.unsqueeze(1) * noise,
            -max_offset,
            max_offset,
        )
        if handoff_plans is None:
            metrics = rollout(
                env, direct, candidates.flatten(0, 1), forced
            )
        else:
            repeated_handoff = handoff_plans.repeat_interleave(population, dim=0)
            metrics = rollout(
                env,
                direct,
                repeated_handoff,
                candidates.flatten(0, 1),
                forced,
            )
        scores = metrics["score"].reshape(states, population)
        elite_ids = torch.topk(scores, elite_count, dim=1).indices
        gather_shape = elite_ids[:, :, None, None].expand(
            -1, -1, args.waypoints, direct.cfg.action_space
        )
        elites = torch.gather(candidates, 1, gather_shape)
        mean = 0.25 * mean + 0.75 * elites.mean(dim=1)
        std = torch.maximum(
            torch.full_like(std, 0.005),
            0.30 * std + 0.70 * elites.std(dim=1, unbiased=False),
        )
        iteration_scores, iteration_ids = scores.max(dim=1)
        improve = iteration_scores > best_scores
        best_scores = torch.where(improve, iteration_scores, best_scores)
        selected = candidates[
            torch.arange(states, device=direct.device), iteration_ids
        ]
        best_plans = torch.where(improve[:, None, None], selected, best_plans)
        history.append(
            {
                "iteration": iteration + 1,
                "best_score": float(best_scores.max().item()),
                "wall_time_s": time.perf_counter() - started,
            }
        )
        print(
            "O1_12_CEM_PROGRESS="
            + json.dumps(
                {
                    "stage": "handoff" if handoff_plans is None else "post",
                    **history[-1],
                }
            ),
            flush=True,
        )

    repeated_best = best_plans.repeat_interleave(population, dim=0)
    if handoff_plans is None:
        replay = rollout(env, direct, repeated_best, forced)
    else:
        replay = rollout(
            env,
            direct,
            handoff_plans.repeat_interleave(population, dim=0),
            repeated_best,
            forced,
        )
    replay_scores = replay["score"].reshape(states, population)
    replay_best_ids = replay_scores.argmax(dim=1)
    replay_best_indices = (
        torch.arange(states, device=direct.device) * population + replay_best_ids
    )
    replay_best = {key: value[replay_best_indices] for key, value in replay.items()}
    return best_plans, replay_best, replay, history


def _handoff_record(metrics: dict[str, torch.Tensor], index: int, bank_id: int) -> dict:
    return {
        "bank_id": bank_id,
        "transition_success": bool(metrics["transition_success"][index].item()),
        "TIM_transition": bool(metrics["tim_gait"][index].item()),
        "transition_step": int(metrics["transition_step"][index].item()),
        "axis_alignment": float(metrics["axis_alignment"][index].item()),
        "axis_steps": int(metrics["axis_steps"][index].item()),
        "min_height_delta_mm": 1000.0
        * float(metrics["min_height_delta_m"][index].item()),
        "min_total_force_n": float(metrics["min_total_force_n"][index].item()),
        "max_total_force_n": float(metrics["max_total_force_n"][index].item()),
        "TIM_overlap": float(metrics["tim_overlap"][index].item()),
        "unsafe": bool(metrics["unsafe"][index].item()),
    }


def _post_failure(record: dict) -> str:
    if record["post_handoff_success"]:
        return "success"
    if not record["transition_success"]:
        return "handoff_transition_not_reproduced"
    if record["unsafe"]:
        if record["penetration"]:
            return "penetration"
        if record["throw"]:
            return "throw"
        if record["drop"]:
            return "drop"
        if record["contact_loss"]:
            return "contact_loss"
        if record["ballistic"]:
            return "ballistic"
        return "unsafe_or_invalid"
    if record["post_min_height_delta_mm"] < -5.0:
        return "height_loss"
    if record["post_min_total_force_n"] < MIN_TOTAL_FORCE_N:
        return "support_force_collapse"
    if record["post_max_total_force_n"] > MAX_TOTAL_FORCE_N:
        return "contact_force_overload"
    if record["post_TIM_overlap"] < MIN_TIM_OVERLAP:
        return "support_overlap_loss"
    if record["post_axis_steps"] < 2:
        return "insufficient_axis_motion"
    if record["post_axis_alignment"] < MIN_AXIS_ALIGNMENT:
        return "axis_misaligned"
    return "joint_gate_miss"


def _post_record(metrics: dict[str, torch.Tensor], index: int, bank_id: int) -> dict:
    failures = metrics["failure_types"][index]
    record = {
        "bank_id": bank_id,
        "transition_success": bool(metrics["transition_success"][index].item()),
        "transition_step": int(metrics["transition_step"][index].item()),
        "transition_height_delta_mm": 1000.0
        * float(metrics["transition_height_delta_m"][index].item()),
        "transition_axis_alignment": float(
            metrics["transition_axis_alignment"][index].item()
        ),
        "transition_total_force_n": float(
            metrics["transition_total_force_n"][index].item()
        ),
        "transition_TIM_overlap": bool(
            metrics["transition_TIM_overlap"][index].item()
        ),
        "post_handoff_success": bool(metrics["post_success"][index].item()),
        "post_axis_alignment": float(metrics["post_axis_alignment"][index].item()),
        "post_axis_steps": int(metrics["post_axis_steps"][index].item()),
        "post_min_height_delta_mm": 1000.0
        * float(metrics["post_min_height_delta_m"][index].item()),
        "post_min_total_force_n": float(
            metrics["post_min_total_force_n"][index].item()
        ),
        "post_max_total_force_n": float(
            metrics["post_max_total_force_n"][index].item()
        ),
        "post_TIM_overlap": float(metrics["post_TIM_overlap"][index].item()),
        "unsafe": bool(metrics["unsafe"][index].item()),
        "ballistic": bool(metrics["ballistic"][index].item()),
        "contact_loss": bool(failures[0].item()),
        "penetration": bool(failures[2].item()),
        "throw": bool(failures[3].item()),
        "drop": bool(failures[4].item()),
    }
    record["primary_failure"] = _post_failure(record)
    return record


def main() -> None:
    started = time.perf_counter()
    bank_ids = torch.as_tensor(args.bank_ids, dtype=torch.long, device=args.device)
    states = bank_ids.numel()
    if args.handoff_population != args.post_population:
        raise ValueError("Matched in-process audit requires equal stage populations")
    num_envs = states * args.handoff_population
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = SEED
    control_dt = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = (
        args.handoff_horizon + args.post_horizon + 10
    ) * control_dt
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    valid_rows = direct.reset_rows_by_canonical[2]
    if not bool(torch.isin(bank_ids, valid_rows).all().item()):
        raise ValueError("Every O1.12 bank ID must be a frozen P3/train reset")

    handoff_plans, handoff_metrics, handoff_replay, handoff_history = _cem(
        _handoff_rollout,
        env,
        direct,
        bank_ids,
        args.handoff_population,
        args.handoff_iterations,
        initial_std=0.06,
        max_offset=args.handoff_max_offset_rad,
    )
    _, post_metrics, post_replay, post_history = _cem(
        _post_rollout,
        env,
        direct,
        bank_ids,
        args.post_population,
        args.post_iterations,
        initial_std=0.04,
        max_offset=args.post_max_offset_rad,
        handoff_plans=handoff_plans,
    )
    handoff_records = [
        _handoff_record(handoff_metrics, index, bank_id)
        for index, bank_id in enumerate(args.bank_ids)
    ]
    post_records = [
        _post_record(post_metrics, index, bank_id)
        for index, bank_id in enumerate(args.bank_ids)
    ]
    for index, record in enumerate(handoff_records):
        start = index * args.handoff_population
        end = start + args.handoff_population
        record["selected_plan_transition_replay_rate"] = float(
            handoff_replay["transition_success"][start:end].float().mean().item()
        )
    for index, record in enumerate(post_records):
        start = index * args.post_population
        end = start + args.post_population
        record["selected_plan_transition_replay_rate"] = float(
            post_replay["transition_success"][start:end].float().mean().item()
        )
        record["selected_plan_post_success_replay_rate"] = float(
            post_replay["post_success"][start:end].float().mean().item()
        )
    taxonomy: dict[str, int] = {}
    for record in post_records:
        failure = record["primary_failure"]
        taxonomy[failure] = taxonomy.get(failure, 0) + 1
    transition_count = sum(record["transition_success"] for record in post_records)
    post_success_count = sum(record["post_handoff_success"] for record in post_records)
    report = {
        "status": (
            "PASS_DIAGNOSTIC_POST_HANDOFF_FOUND"
            if post_success_count
            else "PASS_DIAGNOSTIC_POST_HANDOFF_NOT_FOUND"
        ),
        "phase": "O1.12 Post-Handoff Axis Stabilization Audit",
        "artifact_role": "summary_only_no_state_or_action_plan_serialization",
        "task": args.task,
        "seed": SEED,
        "bank_sha256": direct.cfg.expected_bundle_sha256,
        "bank_ids": args.bank_ids,
        "bank_selection": {
            "anchor": 996,
            "nearby": [1245, 1203, 1171],
            "method": (
                "nearest independent-parent/source P3 states to bank 996 by RMS "
                "standardized O1.9 variable initial features"
            ),
            "RMS_z_distance": {"1245": 0.2768, "1203": 0.2845, "1171": 0.3031},
        },
        "control_dt_s": control_dt,
        "handoff_search": {
            "horizon_steps": args.handoff_horizon,
            "horizon_ms": 1000.0 * args.handoff_horizon * control_dt,
            "population": args.handoff_population,
            "iterations": args.handoff_iterations,
            "waypoints": args.waypoints,
            "max_offset_rad": args.handoff_max_offset_rad,
            "transition_success_count": sum(
                record["transition_success"] for record in handoff_records
            ),
            "states": handoff_records,
            "history": handoff_history,
        },
        "post_handoff_search": {
            "continuation": (
                "same PhysX episode; post action starts one control step after "
                "dominant TIM release/recontact detection"
            ),
            "horizon_steps": args.post_horizon,
            "horizon_ms": 1000.0 * args.post_horizon * control_dt,
            "population": args.post_population,
            "iterations": args.post_iterations,
            "waypoints": args.waypoints,
            "max_offset_rad": args.post_max_offset_rad,
            "transition_success_count": transition_count,
            "post_handoff_success_count": post_success_count,
            "conditional_success_rate": (
                post_success_count / transition_count if transition_count else 0.0
            ),
            "states": post_records,
            "history": post_history,
        },
        "strict_post_gate": {
            "minimum_axis_alignment": MIN_AXIS_ALIGNMENT,
            "minimum_axis_moving_steps": 2,
            "maximum_height_loss_mm_from_P3": 1000.0 * MAX_HEIGHT_LOSS_M,
            "minimum_total_contact_force_n": MIN_TOTAL_FORCE_N,
            "maximum_total_contact_force_n": MAX_TOTAL_FORCE_N,
            "minimum_TIM_overlap": MIN_TIM_OVERLAP,
            "no_ballistic_contact_loss_penetration_drop_throw_drift_or_invalid": True,
        },
        "failure_taxonomy": taxonomy,
        "wall_time_s": time.perf_counter() - started,
        "PPO_trained": False,
        "reward_modified": False,
        "teacher_generated": False,
        "O0_modified": False,
        "state_or_action_plan_exported": False,
        "limitations": (
            "A zero success count is bounded by four P3 entries, the stated CEM "
            "budgets, residual target limits, and 200 ms post window; it is not an "
            "absolute impossibility proof. Post states and plans exist only in memory."
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "O1_12_POST_HANDOFF_AXIS="
        + json.dumps(
            {
                "transition_success": transition_count,
                "post_handoff_success": post_success_count,
                "failure_taxonomy": taxonomy,
            }
        ),
        flush=True,
    )
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
