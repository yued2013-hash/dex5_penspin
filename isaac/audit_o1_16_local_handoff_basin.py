from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="In-memory CEM plan recovery and local P3 handoff-basin perturbation audit."
)
parser.add_argument("--task", default="Dex5-PenSpin-Oracle-P3FixedHorizon-Direct-v0")
parser.add_argument("--bank-ids", type=int, nargs="+", default=(894, 1155, 831))
parser.add_argument("--horizons", type=int, nargs="+", default=(24, 18, 30))
parser.add_argument("--population", type=int, default=128)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--waypoints", type=int, default=3)
parser.add_argument("--max-offset-rad", type=float, default=0.18)
parser.add_argument("--elite-fraction", type=float, default=0.20)
parser.add_argument("--closed-loop-audit", action="store_true")
parser.add_argument("--task-space-audit", action="store_true")
parser.add_argument("--mpc-audit", action="store_true")
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


CONTACT_REBUILD_STEPS = 2
TRANSFER_FORCE_N = 0.03
MIN_TOTAL_FORCE_N = 0.06
MAX_TOTAL_FORCE_N = 6.0
MAX_HEIGHT_LOSS_M = 0.005
MAX_FINAL_HEIGHT_LOSS_M = 0.002
MIN_AXIS_ALIGNMENT = 0.50
MIN_TIM_OVERLAP = 0.80
SEED = 20260731

SCALES = {
    "joint_position_rad": (0.005, 0.010, 0.020),
    "joint_velocity_rad_s": (0.020, 0.050, 0.100),
    "pen_position_m": (0.00025, 0.00050, 0.00100),
    "pen_rotation_rad": tuple(math.radians(value) for value in (0.5, 1.0, 2.0)),
    "pen_linear_velocity_m_s": (0.005, 0.010, 0.020),
    "pen_angular_velocity_rad_s": (0.10, 0.25, 0.50),
    "target_context_rad": (0.0025, 0.0050, 0.0100),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uniform(
    shape: tuple[int, ...], scale: float, generator: torch.Generator, device: str
) -> torch.Tensor:
    if scale == 0.0:
        return torch.zeros(shape, device=device)
    return scale * (2.0 * torch.rand(shape, generator=generator, device=device) - 1.0)


def _rotation_noise(
    count: int, scale: float, generator: torch.Generator, device: str
) -> torch.Tensor:
    if scale == 0.0:
        result = torch.zeros((count, 4), device=device)
        result[:, 0] = 1.0
        return result
    direction = torch.randn((count, 3), generator=generator, device=device)
    direction /= torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-9)
    angle = scale * torch.rand((count,), generator=generator, device=device)
    return torch.cat(
        (torch.cos(0.5 * angle).unsqueeze(-1), direction * torch.sin(0.5 * angle).unsqueeze(-1)),
        dim=-1,
    )


def _clear_events(direct) -> None:
    direct.episode_release_seen.zero_()
    direct.episode_recontact_seen.zero_()
    direct.episode_release_count.zero_()
    direct.episode_recontact_count.zero_()
    direct.episode_tim_overlap_steps.zero_()
    direct.episode_ballistic_seen.zero_()
    direct.episode_failure_type_seen.zero_()


def _apply_perturbation(
    direct,
    forced_bank_ids: torch.Tensor,
    scales: dict[str, float],
    generator: torch.Generator,
) -> None:
    if not any(scales.values()):
        return
    count = forced_bank_ids.numel()
    env_ids = torch.arange(count, dtype=torch.long, device=direct.device)
    source_qpos = direct.source_qpos_bank[forced_bank_ids]
    source_qvel = direct.source_qvel_bank[forced_bank_ids]

    q_noise = _uniform(
        (count, 20), scales.get("joint_position_rad", 0.0), generator, direct.device
    )
    hand_pos = torch.maximum(
        torch.minimum(source_qpos[:, :20] + q_noise, direct.joint_upper),
        direct.joint_lower,
    )
    hand_vel = source_qvel[:, :20] + _uniform(
        (count, 20), scales.get("joint_velocity_rad_s", 0.0), generator, direct.device
    )
    # Move q and target together so the saved q_target-q relation is preserved.
    target = direct.source_target_bank[forced_bank_ids] + q_noise
    target += _uniform(
        (count, 20), scales.get("target_context_rad", 0.0), generator, direct.device
    )
    target = torch.maximum(torch.minimum(target, direct.joint_upper), direct.joint_lower)
    direct.hand.write_joint_state_to_sim(
        hand_pos, hand_vel, joint_ids=direct.dex5_joint_ids, env_ids=env_ids
    )
    direct.source_targets.copy_(target)
    direct.processed_targets.copy_(target)
    direct.hand.set_joint_position_target(
        target, joint_ids=direct.dex5_joint_ids, env_ids=env_ids
    )

    pen_pose = source_qpos[:, 20:27].clone()
    pen_pose[:, :3] += _uniform(
        (count, 3), scales.get("pen_position_m", 0.0), generator, direct.device
    )
    delta_quat = _rotation_noise(
        count, scales.get("pen_rotation_rad", 0.0), generator, direct.device
    )
    pen_pose[:, 3:7] = quat_mul(delta_quat, pen_pose[:, 3:7])
    pen_pose[:, :3] += direct.scene.env_origins[env_ids]
    pen_velocity = source_qvel[:, 20:26].clone()
    pen_velocity[:, :3] += _uniform(
        (count, 3), scales.get("pen_linear_velocity_m_s", 0.0), generator, direct.device
    )
    pen_velocity[:, 3:6] += _uniform(
        (count, 3), scales.get("pen_angular_velocity_rad_s", 0.0), generator, direct.device
    )
    direct.pen.write_root_pose_to_sim(pen_pose, env_ids)
    direct.pen.write_root_velocity_to_sim(pen_velocity, env_ids)
    direct.init_pen_pos.copy_(pen_pose[:, :3] - direct.scene.env_origins[env_ids])
    direct.init_pen_axis.copy_(quat_apply(pen_pose[:, 3:7], direct.pen_x_axis[env_ids]))
    direct.sim.forward()
    direct.scene.update(dt=0.0)
    direct._compute_intermediate_values()


def _actions_for_target(direct, target: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        (target - direct.processed_targets) / direct.cfg.action_scale, -1.0, 1.0
    )


def _task_features(direct) -> torch.Tensor:
    contacts = (
        direct.contact_force_norm > direct.cfg.contact_force_threshold_n
    ).float()
    return torch.cat(
        (
            (direct.pen_pos[:, 2] / 0.001).unsqueeze(-1),
            direct.pen_angvel / 2.0,
            direct.total_contact_force_on_pen_w / 0.10,
            direct.total_contact_torque_on_pen_w / 0.001,
            (contacts.sum(dim=-1) / 3.0).unsqueeze(-1),
        ),
        dim=-1,
    )


HANDOFF_JOINT_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19)


def _mpc_residual(
    direct,
    reference: dict[str, torch.Tensor],
    jacobians: list[torch.Tensor],
    feedback: dict,
    step: int,
    generator: torch.Generator,
) -> torch.Tensor:
    count = direct.num_envs
    candidates = feedback["candidate_count"]
    horizon = feedback["mpc_horizon_steps"]
    active = len(HANDOFF_JOINT_IDS)
    sequence = torch.zeros(
        (count, candidates, horizon, direct.cfg.action_space), device=direct.device
    )
    half = (candidates - 1) // 2
    random_active = feedback["max_correction_rad"] * (
        2.0
        * torch.rand(
            (count, half, horizon, active), generator=generator, device=direct.device
        )
        - 1.0
    )
    active_ids = torch.as_tensor(HANDOFF_JOINT_IDS, device=direct.device)
    sequence[:, 1 : 1 + half, :, active_ids] = random_active
    sequence[:, 1 + half : 1 + 2 * half, :, active_ids] = -random_active

    current = _task_features(direct)
    error = current.unsqueeze(1) - reference["task"][step].view(1, 1, -1)
    cost = torch.zeros((count, candidates), device=direct.device)
    for offset in range(horizon):
        future_step = min(reference["task"].shape[0] - 1, step + offset + 1)
        waypoint = min(args.waypoints - 1, future_step * args.waypoints // 18)
        residual = sequence[:, :, offset]
        error = error + residual @ jacobians[waypoint].T
        predicted = reference["task"][future_step].unsqueeze(0).unsqueeze(0) + error
        height_error = error[:, :, 0]
        angular_error = error[:, :, 1:4]
        wrench_error = error[:, :, 4:10]
        contact_error = error[:, :, 10]
        cost += (
            2.0 * height_error.square()
            + 1.5 * angular_error.square().sum(dim=-1)
            + 0.30 * wrench_error.square().sum(dim=-1)
            + 0.20 * contact_error.square()
            + 0.02
            * (residual[:, :, active_ids] / feedback["max_correction_rad"]).square().mean(dim=-1)
        )
        predicted_angvel = 2.0 * predicted[:, :, 1:4]
        speed = torch.linalg.norm(predicted_angvel, dim=-1)
        alignment = (
            direct.cfg.target_rotation_direction * predicted_angvel[:, :, 2]
            / speed.clamp_min(1.0e-6)
        )
        force_norm = torch.linalg.norm(0.10 * predicted[:, :, 4:7], dim=-1)
        contact_count = 3.0 * predicted[:, :, 10]
        violation = (
            torch.relu(-2.0 - height_error).square()
            + torch.relu(0.5 - alignment).square() * (speed >= 0.5).float()
            + torch.relu(0.04 - force_norm).square() / (0.04**2)
            + torch.relu(force_norm - 6.0).square()
            + torch.relu(1.0 - contact_count).square()
        )
        cost += 1000.0 * violation
    best = torch.argmin(cost, dim=1)
    return sequence[torch.arange(count, device=direct.device), best, 0]


def _rollout(
    env,
    direct,
    plan: torch.Tensor,
    forced_bank_ids: torch.Tensor,
    horizon: int,
    perturbation: dict[str, float] | None = None,
    perturb_seed: int = SEED,
    reference: dict[str, torch.Tensor] | None = None,
    feedback: dict[str, float] | None = None,
    task_feedback_matrices: list[torch.Tensor] | None = None,
    record_reference: bool = False,
) -> dict[str, torch.Tensor]:
    direct.forced_bank_ids = forced_bank_ids
    env.reset()
    generator = torch.Generator(device=direct.device).manual_seed(perturb_seed)
    _apply_perturbation(direct, forced_bank_ids, perturbation or {}, generator)
    hold = torch.zeros((direct.num_envs, direct.cfg.action_space), device=direct.device)
    for _ in range(CONTACT_REBUILD_STEPS):
        env.step(hold)
    _clear_events(direct)

    initial_pen_pos = direct.pen_pos.clone()
    initial_force = direct.contact_force_norm.clone()
    dominant = torch.argmax(initial_force[:, :3], dim=-1)
    dominant_initial = initial_force[:, :3].gather(1, dominant.unsqueeze(-1)).squeeze(-1)
    non_dominant = torch.ones_like(initial_force, dtype=torch.bool)
    non_dominant.scatter_(1, dominant.unsqueeze(-1), False)
    min_height = torch.zeros(direct.num_envs, device=direct.device)
    min_force = torch.full((direct.num_envs,), float("inf"), device=direct.device)
    max_force = torch.zeros(direct.num_envs, device=direct.device)
    new_gain_max = torch.zeros(direct.num_envs, device=direct.device)
    old_unload_max = torch.zeros(direct.num_envs, device=direct.device)
    pair_margin_max = torch.full((direct.num_envs,), -float("inf"), device=direct.device)
    axis_sum = torch.zeros(direct.num_envs, device=direct.device)
    axis_steps = torch.zeros(direct.num_envs, dtype=torch.long, device=direct.device)
    unexpected_done = torch.zeros(direct.num_envs, dtype=torch.bool, device=direct.device)
    reference_history = {"q": [], "dq": [], "height": [], "force": [], "task": []}
    feedback_generator = torch.Generator(device=direct.device).manual_seed(
        perturb_seed + 99173
    )

    for step in range(horizon):
        hand_q = direct.hand.data.joint_pos[:, direct.dex5_joint_ids]
        hand_dq = direct.hand.data.joint_vel[:, direct.dex5_joint_ids]
        if record_reference:
            reference_history["q"].append(hand_q.clone())
            reference_history["dq"].append(hand_dq.clone())
            reference_history["height"].append(direct.pen_pos[:, 2].clone())
            reference_history["force"].append(direct.contact_force_norm.sum(dim=-1).clone())
            reference_history["task"].append(_task_features(direct).clone())
        waypoint = min(args.waypoints - 1, step * args.waypoints // horizon)
        desired = direct.source_targets + plan[:, waypoint]
        if feedback is not None and feedback.get("type") == "mpc":
            if reference is None or task_feedback_matrices is None:
                raise ValueError("MPC rollout requires reference and task Jacobians")
            desired = desired + _mpc_residual(
                direct,
                reference,
                task_feedback_matrices,
                feedback,
                step,
                feedback_generator,
            )
        elif feedback is not None and feedback.get("type", "joint_tracking") == "task_space":
            if reference is None or task_feedback_matrices is None:
                raise ValueError("task-space feedback requires reference and DLS matrices")
            task_error = reference["task"][step].unsqueeze(0) - _task_features(direct)
            residual = feedback["task_gain"] * (
                task_error @ task_feedback_matrices[waypoint].T
            )
            contact_count = (
                direct.contact_force_norm > direct.cfg.contact_force_threshold_n
            ).float().sum(dim=-1)
            contact_gate = torch.clamp(contact_count / 2.0, 0.25, 1.0).unsqueeze(-1)
            residual *= contact_gate
            residual = torch.clamp(
                residual, -feedback["max_correction_rad"], feedback["max_correction_rad"]
            )
            desired = desired + residual
        elif feedback is not None:
            if reference is None:
                raise ValueError("feedback rollout requires an in-memory reference")
            q_error = reference["q"][step].unsqueeze(0) - hand_q
            dq_error = reference["dq"][step].unsqueeze(0) - hand_dq
            gain = torch.full(
                (direct.num_envs, 1), feedback["q_gain"], device=direct.device
            )
            if feedback["adaptive_gain"] > 0.0:
                height_deficit = torch.clamp(
                    (reference["height"][step] - direct.pen_pos[:, 2]) / 0.001,
                    0.0,
                    1.0,
                )
                force_deficit = torch.clamp(
                    (
                        reference["force"][step]
                        - direct.contact_force_norm.sum(dim=-1)
                    )
                    / 0.10,
                    0.0,
                    1.0,
                )
                gain += feedback["adaptive_gain"] * torch.maximum(
                    height_deficit, force_deficit
                ).unsqueeze(-1)
            correction = gain * q_error + feedback["dq_gain"] * dq_error
            correction = torch.clamp(
                correction, -feedback["max_correction_rad"], feedback["max_correction_rad"]
            )
            desired = desired + correction
        _, _, terminated, truncated, _ = env.step(_actions_for_target(direct, desired))
        unexpected_done |= terminated | truncated
        force = direct.contact_force_norm
        total_force = force.sum(dim=-1)
        min_force = torch.minimum(min_force, total_force)
        max_force = torch.maximum(max_force, total_force)
        new_gain = torch.where(
            non_dominant, force - initial_force, torch.full_like(force, -1.0e6)
        ).max(dim=-1).values
        old_force = force[:, :3].gather(1, dominant.unsqueeze(-1)).squeeze(-1)
        old_unload = dominant_initial - old_force
        new_gain_max = torch.maximum(new_gain_max, new_gain)
        old_unload_max = torch.maximum(old_unload_max, old_unload)
        pair_margin_max = torch.maximum(
            pair_margin_max,
            torch.minimum(
                new_gain / TRANSFER_FORCE_N - 1.0,
                old_unload / TRANSFER_FORCE_N - 1.0,
            ),
        )
        min_height = torch.minimum(min_height, direct.pen_pos[:, 2] - initial_pen_pos[:, 2])
        speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        moving = speed >= direct.cfg.axis_alignment_min_angular_speed_rad_s
        alignment = (
            direct.cfg.target_rotation_direction * direct.pen_angvel[:, 2]
            / speed.clamp_min(1.0e-6)
        )
        axis_sum += torch.where(moving, alignment, 0.0)
        axis_steps += moving.long()

    gait = direct.episode_release_seen & direct.episode_recontact_seen
    tim_gait = gait[:, :3].gather(1, dominant.unsqueeze(-1)).squeeze(-1)
    tim_overlap = direct.episode_tim_overlap_steps.float() / float(horizon)
    axis_alignment = axis_sum / axis_steps.clamp_min(1).float()
    final_height = direct.pen_pos[:, 2] - initial_pen_pos[:, 2]
    height_recovered = final_height >= -MAX_FINAL_HEIGHT_LOSS_M
    failures = direct.episode_failure_type_seen
    unsafe = (
        unexpected_done
        | direct.episode_ballistic_seen
        | failures[:, 0]
        | failures[:, 2]
        | failures[:, 3]
        | failures[:, 4]
        | failures[:, 5]
        | failures[:, 6]
    )
    feasible = (
        tim_gait
        & (new_gain_max >= TRANSFER_FORCE_N)
        & (old_unload_max >= TRANSFER_FORCE_N)
        & (pair_margin_max >= 0.0)
        & (min_force >= MIN_TOTAL_FORCE_N)
        & (max_force <= MAX_TOTAL_FORCE_N)
        & (min_height >= -MAX_HEIGHT_LOSS_M)
        & (axis_steps >= 2)
        & (axis_alignment >= MIN_AXIS_ALIGNMENT)
        & (tim_overlap >= MIN_TIM_OVERLAP)
        & ~unsafe
    )
    recovered_strict = feasible & height_recovered
    margins = torch.stack(
        (
            torch.where(tim_gait, torch.ones_like(min_force), -torch.ones_like(min_force)),
            new_gain_max / TRANSFER_FORCE_N - 1.0,
            old_unload_max / TRANSFER_FORCE_N - 1.0,
            pair_margin_max,
            min_force / MIN_TOTAL_FORCE_N - 1.0,
            (MAX_TOTAL_FORCE_N - max_force) / MAX_TOTAL_FORCE_N,
            (MAX_HEIGHT_LOSS_M + min_height) / MAX_HEIGHT_LOSS_M,
            (axis_alignment - MIN_AXIS_ALIGNMENT) / MIN_AXIS_ALIGNMENT,
            (tim_overlap - MIN_TIM_OVERLAP) / (1.0 - MIN_TIM_OVERLAP),
        ),
        dim=-1,
    )
    handoff_margin = margins.min(dim=-1).values
    score = margins.clamp(-3.0, 3.0).sum(dim=-1)
    score += 10.0 * feasible.float() + 5.0 * recovered_strict.float() - 15.0 * unsafe.float()
    result = {
        "score": score,
        "feasible": feasible,
        "recovered_strict": recovered_strict,
        "handoff_margin": handoff_margin,
        "tim_gait": tim_gait,
        "axis_alignment": axis_alignment,
        "min_height_delta_mm": 1000.0 * min_height,
        "final_height_delta_mm": 1000.0 * final_height,
        "height_recovered": height_recovered,
        "min_total_force_n": min_force,
        "max_total_force_n": max_force,
        "tim_overlap": tim_overlap,
        "unsafe": unsafe,
        "ballistic": direct.episode_ballistic_seen.clone(),
        "contact_loss": failures[:, 0].clone(),
        "penetration": failures[:, 2].clone(),
        "throw": failures[:, 3].clone(),
        "drop": failures[:, 4].clone(),
        "drift": failures[:, 5].clone(),
        "invalid": failures[:, 6].clone() | unexpected_done,
    }
    if record_reference:
        result["reference_history"] = {
            key: torch.stack(values, dim=1) for key, values in reference_history.items()
        }
    return result


def _cem_plan(env, direct, bank_id: int, horizon: int) -> tuple[torch.Tensor, dict]:
    forced = torch.full(
        (args.population,), bank_id, dtype=torch.long, device=direct.device
    )
    mean = torch.zeros((args.waypoints, direct.cfg.action_space), device=direct.device)
    std = torch.full_like(mean, 0.06)
    best_score = -float("inf")
    best_plan = torch.zeros_like(mean)
    elite_count = max(2, round(args.population * args.elite_fraction))
    generator = torch.Generator(device=direct.device).manual_seed(SEED + bank_id)
    history = []
    for iteration in range(args.iterations):
        population = torch.clamp(
            mean.unsqueeze(0)
            + std.unsqueeze(0)
            * torch.randn(
                (args.population, args.waypoints, direct.cfg.action_space),
                generator=generator,
                device=direct.device,
            ),
            -args.max_offset_rad,
            args.max_offset_rad,
        )
        metrics = _rollout(env, direct, population, forced, horizon)
        elite_ids = torch.topk(metrics["score"], elite_count).indices
        elites = population[elite_ids]
        mean = 0.25 * mean + 0.75 * elites.mean(dim=0)
        std = torch.maximum(
            torch.full_like(std, 0.005),
            0.30 * std + 0.70 * elites.std(dim=0, unbiased=False),
        )
        iteration_score, iteration_id = metrics["score"].max(dim=0)
        if float(iteration_score.item()) > best_score:
            best_score = float(iteration_score.item())
            best_plan = population[int(iteration_id.item())].clone()
        history.append(
            {
                "iteration": iteration + 1,
                "best_score": best_score,
                "population_recovered_strict_rate": float(
                    metrics["recovered_strict"].float().mean().item()
                ),
            }
        )
    return best_plan, {"best_score": best_score, "history": history}


def _scenario_definitions() -> list[dict]:
    scenarios = [{"name": "nominal", "factor": "nominal", "level": 0, "scales": {}}]
    for factor, values in SCALES.items():
        for level, value in enumerate(values, start=1):
            scenarios.append(
                {
                    "name": f"{factor}_L{level}",
                    "factor": factor,
                    "level": level,
                    "scales": {factor: value},
                }
            )
    factors = tuple(SCALES)
    for level in range(1, 4):
        scenarios.append(
            {
                "name": f"joint_L{level}",
                "factor": "joint",
                "level": level,
                "scales": {factor: SCALES[factor][level - 1] for factor in factors},
            }
        )
    return scenarios


def _taxonomy(metrics: dict[str, torch.Tensor]) -> dict[str, int]:
    labels = []
    count = metrics["feasible"].numel()
    for index in range(count):
        if bool(metrics["recovered_strict"][index].item()):
            label = "recovered_strict"
        elif bool(metrics["feasible"][index].item()):
            label = "strict_no_height_recovery"
        elif bool(metrics["invalid"][index].item()):
            label = "invalid"
        elif bool(metrics["penetration"][index].item()):
            label = "penetration"
        elif bool(metrics["throw"][index].item()):
            label = "throw"
        elif bool(metrics["drop"][index].item()):
            label = "drop"
        elif bool(metrics["contact_loss"][index].item()):
            label = "contact_loss"
        elif bool(metrics["ballistic"][index].item()):
            label = "ballistic"
        elif not bool(metrics["tim_gait"][index].item()):
            label = "no_TIM_transition"
        elif float(metrics["max_total_force_n"][index].item()) > MAX_TOTAL_FORCE_N:
            label = "contact_force_overload"
        elif float(metrics["axis_alignment"][index].item()) < MIN_AXIS_ALIGNMENT:
            label = "axis_misaligned"
        elif not bool(metrics["height_recovered"][index].item()):
            label = "height_not_recovered"
        else:
            label = "joint_gate_miss"
        labels.append(label)
    return dict(Counter(labels))


def _summary(metrics: dict[str, torch.Tensor]) -> dict:
    return {
        "episodes": int(metrics["feasible"].numel()),
        "TIM_transition_rate": float(metrics["tim_gait"].float().mean().item()),
        "strict_rate": float(metrics["feasible"].float().mean().item()),
        "recovered_strict_rate": float(metrics["recovered_strict"].float().mean().item()),
        "height_recovery_rate": float(metrics["height_recovered"].float().mean().item()),
        "unsafe_rate": float(metrics["unsafe"].float().mean().item()),
        "axis_alignment_median": float(metrics["axis_alignment"].median().item()),
        "min_height_delta_mm_p10": float(torch.quantile(metrics["min_height_delta_mm"], 0.10).item()),
        "final_height_delta_mm_p10": float(torch.quantile(metrics["final_height_delta_mm"], 0.10).item()),
        "handoff_margin_p90": float(torch.quantile(metrics["handoff_margin"], 0.90).item()),
        "failure_taxonomy": _taxonomy(metrics),
    }


O1_17_CONTROLLERS = {
    "q_track_weak": {
        "q_gain": 0.40,
        "dq_gain": 0.010,
        "adaptive_gain": 0.0,
        "max_correction_rad": 0.020,
    },
    "q_track_medium": {
        "q_gain": 0.80,
        "dq_gain": 0.020,
        "adaptive_gain": 0.0,
        "max_correction_rad": 0.030,
    },
    "support_adaptive": {
        "q_gain": 0.50,
        "dq_gain": 0.015,
        "adaptive_gain": 0.80,
        "max_correction_rad": 0.040,
    },
}

O1_18_TASK_CONTROLLERS = {
    "task_dls_soft": {
        "type": "task_space",
        "task_gain": 0.25,
        "damping": 0.20,
        "max_correction_rad": 0.020,
    },
    "task_dls_medium": {
        "type": "task_space",
        "task_gain": 0.50,
        "damping": 0.15,
        "max_correction_rad": 0.030,
    },
    "task_dls_strong": {
        "type": "task_space",
        "task_gain": 0.75,
        "damping": 0.10,
        "max_correction_rad": 0.040,
    },
}

O1_19_MPC_CONTROLLERS = {
    "mpc_50ms": {
        "type": "mpc",
        "mpc_horizon_steps": 3,
        "candidate_count": 64,
        "max_correction_rad": 0.025,
    },
    "mpc_100ms": {
        "type": "mpc",
        "mpc_horizon_steps": 6,
        "candidate_count": 64,
        "max_correction_rad": 0.025,
    },
}


def _o1_17_scenarios() -> list[dict]:
    factors = tuple(SCALES)
    result = [{"name": "nominal", "scales": {}}]
    for level in range(1, 4):
        result.append(
            {
                "name": f"joint_L{level}",
                "scales": {factor: SCALES[factor][level - 1] for factor in factors},
            }
        )
    return result


def _extract_reference(metrics: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    successful = torch.nonzero(metrics["recovered_strict"], as_tuple=False).flatten()
    if successful.numel() > 0:
        selected = int(successful[torch.argmax(metrics["score"][successful])].item())
    else:
        selected = int(torch.argmax(metrics["score"]).item())
    history = metrics["reference_history"]
    reference = {
        "q": history["q"][selected],
        "dq": history["dq"][selected],
        "height": history["height"][selected],
        "force": history["force"][selected],
        "task": history["task"][selected],
    }
    return reference, selected


def _run_o1_17() -> None:
    if args.bank_ids != [1155] and tuple(args.bank_ids) != (1155,):
        raise ValueError("O1.17 is frozen to --bank-ids 1155")
    if args.horizons != [18] and tuple(args.horizons) != (18,):
        raise ValueError("O1.17 is frozen to an 18-step / 300 ms horizon")
    started = time.perf_counter()
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.population)
    cfg.seed = SEED
    control_dt = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = (18 + CONTACT_REBUILD_STEPS + 8) * control_dt
    bank_path = Path(cfg.bundle_path).resolve()
    bank_hash_before = _sha256(bank_path)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    if bank_hash_before != direct.cfg.expected_bundle_sha256:
        raise RuntimeError("Frozen O0 bank hash mismatch")
    if not bool(torch.isin(torch.tensor([1155], device=direct.device), direct.reset_rows_by_canonical[2]).all().item()):
        raise ValueError("O1.17 bank 1155 must remain in frozen P3/train")

    plan, search = _cem_plan(env, direct, 1155, 18)
    forced = torch.full((args.population,), 1155, dtype=torch.long, device=direct.device)
    plan_batch = plan.unsqueeze(0).repeat(args.population, 1, 1)
    reference_metrics = _rollout(
        env, direct, plan_batch, forced, 18, perturb_seed=SEED + 170000, record_reference=True
    )
    reference, reference_clone = _extract_reference(reference_metrics)

    calibration = []
    calibration_scenarios = (
        {"name": "nominal", "scales": {}},
        _o1_17_scenarios()[1],
    )
    modes: list[tuple[str, dict[str, float] | None]] = [("open_loop", None)] + list(
        O1_17_CONTROLLERS.items()
    )
    for mode_index, (name, controller) in enumerate(modes):
        rows = []
        for scenario_index, scenario in enumerate(calibration_scenarios):
            metrics = _rollout(
                env,
                direct,
                plan_batch,
                forced,
                18,
                perturbation=scenario["scales"],
                perturb_seed=SEED + 200000 + scenario_index,
                reference=reference,
                feedback=controller,
            )
            rows.append({"name": scenario["name"], **_summary(metrics)})
        calibration.append(
            {
                "mode": name,
                "controller": controller,
                "scenarios": rows,
                "selection_score": (
                    rows[1]["recovered_strict_rate"]
                    - 0.5 * rows[1]["unsafe_rate"]
                    + 0.1 * rows[0]["recovered_strict_rate"]
                ),
            }
        )
    eligible = [
        row
        for row in calibration
        if row["mode"] != "open_loop"
        and row["scenarios"][0]["unsafe_rate"]
        <= calibration[0]["scenarios"][0]["unsafe_rate"] + 0.02
    ]
    selected = max(eligible or calibration[1:], key=lambda row: row["selection_score"])

    evaluation = []
    for mode_name, controller in (("open_loop", None), (selected["mode"], selected["controller"])):
        rows = []
        for scenario_index, scenario in enumerate(_o1_17_scenarios()):
            metrics = _rollout(
                env,
                direct,
                plan_batch,
                forced,
                18,
                perturbation=scenario["scales"],
                perturb_seed=SEED + 300000 + scenario_index,
                reference=reference,
                feedback=controller,
            )
            rows.append({"name": scenario["name"], **_summary(metrics)})
        evaluation.append({"mode": mode_name, "controller": controller, "scenarios": rows})

    def by_name(mode: dict, scenario: str) -> dict:
        return next(row for row in mode["scenarios"] if row["name"] == scenario)

    baseline, closed_loop = evaluation
    b_l1 = by_name(baseline, "joint_L1")
    c_l1 = by_name(closed_loop, "joint_L1")
    c_l2 = by_name(closed_loop, "joint_L2")
    widening_gate = {
        "joint_L1_at_least_30pct": c_l1["recovered_strict_rate"] >= 0.30,
        "joint_L1_at_least_2x_matched_baseline": c_l1["recovered_strict_rate"]
        >= 2.0 * b_l1["recovered_strict_rate"],
        "joint_L2_at_least_10pct": c_l2["recovered_strict_rate"] >= 0.10,
        "joint_L1_unsafe_not_worse_by_more_than_2pp": c_l1["unsafe_rate"]
        <= b_l1["unsafe_rate"] + 0.02,
    }
    bank_hash_after = _sha256(bank_path)
    if bank_hash_after != bank_hash_before:
        raise RuntimeError("Frozen O0 bank changed during O1.17")
    report = {
        "status": (
            "PASS_CLOSED_LOOP_BASIN_WIDENED"
            if all(widening_gate.values())
            else "PASS_DIAGNOSTIC_TESTED_FEEDBACK_FAILED_TO_WIDEN"
        ),
        "phase": "O1.17 Closed-Loop Basin Widening Audit",
        "artifact_role": "matched_feedback_ablation_not_policy_teacher_or_reset_dataset",
        "task": args.task,
        "bank_id": 1155,
        "horizon_steps": 18,
        "horizon_ms": 1000.0 * 18 * control_dt,
        "bank_sha256_before": bank_hash_before,
        "bank_sha256_after": bank_hash_after,
        "search": {**search, "action_plan_serialized": False},
        "reference": {
            "selected_clone": reference_clone,
            "selected_clone_recovered_strict": bool(reference_metrics["recovered_strict"][reference_clone].item()),
            "serialized": False,
        },
        "controller_candidates_preregistered": O1_17_CONTROLLERS,
        "perturbation_protocol": {
            "calibration_seed_base": SEED + 200000,
            "held_out_seed_base": SEED + 300000,
            "same_perturbations_matched_across_modes": True,
            "calibration_and_held_out_seeds_disjoint": True,
            "controller_selection_nominal_unsafe_tolerance_pp": 2.0,
        },
        "calibration": calibration,
        "selected_controller": selected["mode"],
        "held_out_evaluation": evaluation,
        "widening_gate": widening_gate,
        "widening_gate_pass": all(widening_gate.values()),
        "historical_o1_16_joint_L1_rate": 0.1015625,
        "elapsed_seconds": time.perf_counter() - started,
        "constraints": {
            "PPO_trained": False,
            "reward_modified": False,
            "O0_modified": False,
            "teacher_generated": False,
            "action_plan_serialized": False,
            "reference_trajectory_serialized": False,
            "reset_dataset_generated": False,
            "contact_pattern_controller_used": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "O1_17_RAW="
        + json.dumps(
            {
                "output": str(output),
                "selected_controller": selected["mode"],
                "widening_gate_pass": report["widening_gate_pass"],
            }
        ),
        flush=True,
    )
    del direct.forced_bank_ids
    env.close()


def _identify_task_jacobians(env, direct, plan: torch.Tensor) -> tuple[list[torch.Tensor], list[dict]]:
    probe_rad = 0.010
    forced = torch.full((direct.num_envs,), 1155, dtype=torch.long, device=direct.device)
    matrices = []
    diagnostics = []
    for waypoint, start_step in enumerate((0, 6, 12)):
        direct.forced_bank_ids = forced
        env.reset()
        hold = torch.zeros((direct.num_envs, direct.cfg.action_space), device=direct.device)
        for _ in range(CONTACT_REBUILD_STEPS):
            env.step(hold)
        for step in range(start_step):
            stage = min(args.waypoints - 1, step * args.waypoints // 18)
            desired = direct.source_targets + plan[stage].unsqueeze(0)
            env.step(_actions_for_target(direct, desired))
        residual = torch.zeros((direct.num_envs, direct.cfg.action_space), device=direct.device)
        for joint in range(direct.cfg.action_space):
            residual[1 + 2 * joint, joint] = probe_rad
            residual[2 + 2 * joint, joint] = -probe_rad
        desired = direct.source_targets + plan[waypoint].unsqueeze(0) + residual
        env.step(_actions_for_target(direct, desired))
        features = _task_features(direct)
        jacobian = torch.stack(
            [
                (features[1 + 2 * joint] - features[2 + 2 * joint])
                / (2.0 * probe_rad)
                for joint in range(direct.cfg.action_space)
            ],
            dim=-1,
        )
        singular = torch.linalg.svdvals(jacobian)
        matrices.append(jacobian)
        diagnostics.append(
            {
                "waypoint": waypoint,
                "start_step": start_step,
                "probe_rad": probe_rad,
                "rank_at_1e_4": int((singular > 1.0e-4).sum().item()),
                "largest_singular_value": float(singular.max().item()),
                "smallest_nonzero_singular_value": float(
                    singular[singular > 1.0e-4].min().item()
                ) if bool((singular > 1.0e-4).any().item()) else 0.0,
                "effective_condition_number": float(
                    singular.max().item() / singular[singular > 1.0e-4].min().item()
                ) if bool((singular > 1.0e-4).any().item()) else None,
            }
        )
    return matrices, diagnostics


def _dls_feedback_matrices(
    jacobians: list[torch.Tensor], damping: float
) -> list[torch.Tensor]:
    result = []
    for jacobian in jacobians:
        identity = torch.eye(jacobian.shape[0], device=jacobian.device)
        result.append(
            jacobian.T
            @ torch.linalg.solve(
                jacobian @ jacobian.T + (damping**2) * identity,
                identity,
            )
        )
    return result


def _run_o1_18() -> None:
    if tuple(args.bank_ids) != (1155,) or tuple(args.horizons) != (18,):
        raise ValueError("O1.18 is frozen to bank 1155 and 18 steps / 300 ms")
    started = time.perf_counter()
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.population)
    cfg.seed = SEED
    control_dt = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = (18 + CONTACT_REBUILD_STEPS + 8) * control_dt
    bank_path = Path(cfg.bundle_path).resolve()
    bank_hash_before = _sha256(bank_path)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    if bank_hash_before != direct.cfg.expected_bundle_sha256:
        raise RuntimeError("Frozen O0 bank hash mismatch")
    if not bool(torch.isin(torch.tensor([1155], device=direct.device), direct.reset_rows_by_canonical[2]).all().item()):
        raise ValueError("O1.18 bank 1155 must remain in frozen P3/train")

    plan, search = _cem_plan(env, direct, 1155, 18)
    forced = torch.full((args.population,), 1155, dtype=torch.long, device=direct.device)
    plan_batch = plan.unsqueeze(0).repeat(args.population, 1, 1)
    reference_metrics = _rollout(
        env, direct, plan_batch, forced, 18, perturb_seed=SEED + 180000, record_reference=True
    )
    reference, reference_clone = _extract_reference(reference_metrics)
    jacobians, jacobian_diagnostics = _identify_task_jacobians(env, direct, plan)

    task_matrices = {
        name: _dls_feedback_matrices(jacobians, config["damping"])
        for name, config in O1_18_TASK_CONTROLLERS.items()
    }
    modes: list[tuple[str, dict | None, list[torch.Tensor] | None]] = [
        ("A_open_loop", None, None),
        ("B_o1_17_joint_tracking", O1_17_CONTROLLERS["support_adaptive"], None),
    ] + [
        (f"C_{name}", config, task_matrices[name])
        for name, config in O1_18_TASK_CONTROLLERS.items()
    ]
    calibration = []
    calibration_scenarios = ({"name": "nominal", "scales": {}}, _o1_17_scenarios()[1])
    for name, controller, matrices in modes:
        rows = []
        for scenario_index, scenario in enumerate(calibration_scenarios):
            metrics = _rollout(
                env,
                direct,
                plan_batch,
                forced,
                18,
                perturbation=scenario["scales"],
                perturb_seed=SEED + 400000 + scenario_index,
                reference=reference,
                feedback=controller,
                task_feedback_matrices=matrices,
            )
            rows.append({"name": scenario["name"], **_summary(metrics)})
        calibration.append(
            {
                "mode": name,
                "controller": controller,
                "scenarios": rows,
                "selection_score": (
                    rows[1]["recovered_strict_rate"]
                    - 0.5 * rows[1]["unsafe_rate"]
                    + 0.1 * rows[0]["recovered_strict_rate"]
                ),
            }
        )
    open_calibration = calibration[0]
    eligible_task = [
        row
        for row in calibration[2:]
        if row["scenarios"][0]["unsafe_rate"]
        <= open_calibration["scenarios"][0]["unsafe_rate"] + 0.02
    ]
    selected = max(eligible_task or calibration[2:], key=lambda row: row["selection_score"])
    selected_name = selected["mode"].removeprefix("C_")

    formal_modes = (
        ("A_open_loop", None, None),
        ("B_o1_17_joint_tracking", O1_17_CONTROLLERS["support_adaptive"], None),
        (selected["mode"], O1_18_TASK_CONTROLLERS[selected_name], task_matrices[selected_name]),
    )
    evaluation = []
    for name, controller, matrices in formal_modes:
        rows = []
        for scenario_index, scenario in enumerate(_o1_17_scenarios()):
            metrics = _rollout(
                env,
                direct,
                plan_batch,
                forced,
                18,
                perturbation=scenario["scales"],
                perturb_seed=SEED + 500000 + scenario_index,
                reference=reference,
                feedback=controller,
                task_feedback_matrices=matrices,
            )
            rows.append({"name": scenario["name"], **_summary(metrics)})
        evaluation.append({"mode": name, "controller": controller, "scenarios": rows})

    def by_name(mode: dict, scenario: str) -> dict:
        return next(row for row in mode["scenarios"] if row["name"] == scenario)

    open_mode, joint_mode, task_mode = evaluation
    a_l1 = by_name(open_mode, "joint_L1")
    b_l1 = by_name(joint_mode, "joint_L1")
    c_l1 = by_name(task_mode, "joint_L1")
    c_l2 = by_name(task_mode, "joint_L2")
    widening_gate = {
        "task_joint_L1_at_least_30pct": c_l1["recovered_strict_rate"] >= 0.30,
        "task_joint_L1_at_least_2x_open_loop": c_l1["recovered_strict_rate"]
        >= 2.0 * a_l1["recovered_strict_rate"],
        "task_joint_L1_better_than_o1_17_tracking": c_l1["recovered_strict_rate"]
        > b_l1["recovered_strict_rate"],
        "task_joint_L2_at_least_10pct": c_l2["recovered_strict_rate"] >= 0.10,
        "task_joint_L1_unsafe_not_worse_than_open_by_more_than_2pp": c_l1["unsafe_rate"]
        <= a_l1["unsafe_rate"] + 0.02,
    }
    bank_hash_after = _sha256(bank_path)
    if bank_hash_after != bank_hash_before:
        raise RuntimeError("Frozen O0 bank changed during O1.18")
    report = {
        "status": (
            "PASS_TASK_SPACE_BASIN_WIDENED"
            if all(widening_gate.values())
            else (
                "PASS_DIAGNOSTIC_TASK_CONTROLLER_FAILED_CALIBRATION_SAFETY"
                if not eligible_task
                else "PASS_DIAGNOSTIC_TASK_SPACE_FEEDBACK_NOT_YET_ATTRACTOR"
            )
        ),
        "phase": "O1.18 Task-Space Closed-Loop Basin Audit",
        "artifact_role": "matched_task_feedback_ablation_not_policy_teacher_or_reset_dataset",
        "task": args.task,
        "bank_id": 1155,
        "horizon_steps": 18,
        "horizon_ms": 1000.0 * 18 * control_dt,
        "bank_sha256_before": bank_hash_before,
        "bank_sha256_after": bank_hash_after,
        "search": {**search, "action_plan_serialized": False},
        "reference": {
            "selected_clone": reference_clone,
            "selected_clone_recovered_strict": bool(reference_metrics["recovered_strict"][reference_clone].item()),
            "serialized": False,
        },
        "task_feature_contract": {
            "dimension": 11,
            "features": [
                "pen_height/0.001m",
                "pen_angular_velocity_xyz/2rad_s",
                "total_support_force_xyz/0.10N",
                "total_support_torque_xyz/0.001Nm",
                "aggregate_contact_count/3",
            ],
            "per_finger_contact_identity_targeted": False,
        },
        "jacobian_identification": {
            "method": "three_stage_antithetic_one_step_joint_target_probes",
            "serialized": False,
            "diagnostics": jacobian_diagnostics,
        },
        "controller_candidates_preregistered": O1_18_TASK_CONTROLLERS,
        "perturbation_protocol": {
            "calibration_seed_base": SEED + 400000,
            "held_out_seed_base": SEED + 500000,
            "same_perturbations_matched_across_modes": True,
            "calibration_and_held_out_seeds_disjoint": True,
            "controller_selection_nominal_unsafe_tolerance_pp": 2.0,
        },
        "calibration": calibration,
        "selected_task_controller": selected["mode"],
        "selected_task_controller_passed_calibration_safety": bool(eligible_task),
        "selection_fallback_used_for_diagnostic_comparison": not bool(eligible_task),
        "held_out_evaluation": evaluation,
        "widening_gate": widening_gate,
        "widening_gate_pass": all(widening_gate.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "constraints": {
            "PPO_trained": False,
            "reward_modified": False,
            "O0_modified": False,
            "teacher_generated": False,
            "CEM_trajectory_serialized": False,
            "reference_trajectory_serialized": False,
            "jacobian_serialized": False,
            "reset_dataset_generated": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "O1_18_RAW="
        + json.dumps(
            {
                "output": str(output),
                "selected_task_controller": selected["mode"],
                "widening_gate_pass": report["widening_gate_pass"],
            }
        ),
        flush=True,
    )
    del direct.forced_bank_ids
    env.close()


def _run_o1_19() -> None:
    if tuple(args.bank_ids) != (1155,) or tuple(args.horizons) != (18,):
        raise ValueError("O1.19 is frozen to bank 1155 and 18 steps / 300 ms")
    started = time.perf_counter()
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.population)
    cfg.seed = SEED
    control_dt = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = (18 + CONTACT_REBUILD_STEPS + 8) * control_dt
    bank_path = Path(cfg.bundle_path).resolve()
    bank_hash_before = _sha256(bank_path)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    if bank_hash_before != direct.cfg.expected_bundle_sha256:
        raise RuntimeError("Frozen O0 bank hash mismatch")
    if not bool(torch.isin(torch.tensor([1155], device=direct.device), direct.reset_rows_by_canonical[2]).all().item()):
        raise ValueError("O1.19 bank 1155 must remain in frozen P3/train")

    plan, search = _cem_plan(env, direct, 1155, 18)
    forced = torch.full((args.population,), 1155, dtype=torch.long, device=direct.device)
    plan_batch = plan.unsqueeze(0).repeat(args.population, 1, 1)
    reference_metrics = _rollout(
        env, direct, plan_batch, forced, 18, perturb_seed=SEED + 190000, record_reference=True
    )
    reference, reference_clone = _extract_reference(reference_metrics)
    jacobians, jacobian_diagnostics = _identify_task_jacobians(env, direct, plan)
    o1_18_config = O1_18_TASK_CONTROLLERS["task_dls_soft"]
    o1_18_matrices = _dls_feedback_matrices(jacobians, o1_18_config["damping"])

    modes: list[tuple[str, dict | None, list[torch.Tensor] | None]] = [
        ("A_open_loop", None, None),
        ("B_o1_18_task_dls_soft", o1_18_config, o1_18_matrices),
    ] + [
        (f"C_{name}", config, jacobians)
        for name, config in O1_19_MPC_CONTROLLERS.items()
    ]
    calibration = []
    calibration_scenarios = ({"name": "nominal", "scales": {}}, _o1_17_scenarios()[1])
    for name, controller, matrices in modes:
        rows = []
        for scenario_index, scenario in enumerate(calibration_scenarios):
            metrics = _rollout(
                env,
                direct,
                plan_batch,
                forced,
                18,
                perturbation=scenario["scales"],
                perturb_seed=SEED + 600000 + scenario_index,
                reference=reference,
                feedback=controller,
                task_feedback_matrices=matrices,
            )
            rows.append({"name": scenario["name"], **_summary(metrics)})
        calibration.append(
            {
                "mode": name,
                "controller": controller,
                "scenarios": rows,
                "selection_score": (
                    rows[1]["recovered_strict_rate"]
                    - 0.5 * rows[1]["unsafe_rate"]
                    + 0.1 * rows[0]["recovered_strict_rate"]
                ),
            }
        )
    open_calibration = calibration[0]
    eligible_mpc = [
        row
        for row in calibration[2:]
        if row["scenarios"][0]["unsafe_rate"]
        <= open_calibration["scenarios"][0]["unsafe_rate"] + 0.02
    ]
    selected = max(eligible_mpc or calibration[2:], key=lambda row: row["selection_score"])
    selected_name = selected["mode"].removeprefix("C_")

    formal_modes = (
        ("A_open_loop", None, None),
        ("B_o1_18_task_dls_soft", o1_18_config, o1_18_matrices),
        (selected["mode"], O1_19_MPC_CONTROLLERS[selected_name], jacobians),
    )
    evaluation = []
    for name, controller, matrices in formal_modes:
        rows = []
        for scenario_index, scenario in enumerate(_o1_17_scenarios()):
            metrics = _rollout(
                env,
                direct,
                plan_batch,
                forced,
                18,
                perturbation=scenario["scales"],
                perturb_seed=SEED + 700000 + scenario_index,
                reference=reference,
                feedback=controller,
                task_feedback_matrices=matrices,
            )
            rows.append({"name": scenario["name"], **_summary(metrics)})
        evaluation.append({"mode": name, "controller": controller, "scenarios": rows})

    def by_name(mode: dict, scenario: str) -> dict:
        return next(row for row in mode["scenarios"] if row["name"] == scenario)

    open_mode, task_mode, mpc_mode = evaluation
    a_l1 = by_name(open_mode, "joint_L1")
    b_l1 = by_name(task_mode, "joint_L1")
    c_l1 = by_name(mpc_mode, "joint_L1")
    c_l2 = by_name(mpc_mode, "joint_L2")
    widening_gate = {
        "mpc_joint_L1_at_least_30pct": c_l1["recovered_strict_rate"] >= 0.30,
        "mpc_joint_L1_at_least_2x_open_loop": c_l1["recovered_strict_rate"]
        >= 2.0 * a_l1["recovered_strict_rate"],
        "mpc_joint_L1_better_than_o1_18_task_feedback": c_l1["recovered_strict_rate"]
        > b_l1["recovered_strict_rate"],
        "mpc_joint_L2_at_least_10pct": c_l2["recovered_strict_rate"] >= 0.10,
        "mpc_joint_L1_unsafe_not_worse_than_open_by_more_than_2pp": c_l1["unsafe_rate"]
        <= a_l1["unsafe_rate"] + 0.02,
    }
    bank_hash_after = _sha256(bank_path)
    if bank_hash_after != bank_hash_before:
        raise RuntimeError("Frozen O0 bank changed during O1.19")
    report = {
        "status": (
            "PASS_REDUCED_ORDER_MPC_BASIN_WIDENED"
            if all(widening_gate.values())
            else (
                "PASS_DIAGNOSTIC_MPC_FAILED_CALIBRATION_SAFETY"
                if not eligible_mpc
                else "PASS_DIAGNOSTIC_REDUCED_ORDER_MPC_NOT_YET_ATTRACTOR"
            )
        ),
        "phase": "O1.19 Reduced-order Handoff MPC Feasibility Audit",
        "artifact_role": "matched_receding_horizon_ablation_not_policy_teacher_or_reset_dataset",
        "task": args.task,
        "bank_id": 1155,
        "horizon_steps": 18,
        "horizon_ms": 1000.0 * 18 * control_dt,
        "bank_sha256_before": bank_hash_before,
        "bank_sha256_after": bank_hash_after,
        "search": {**search, "action_plan_serialized": False},
        "reference": {
            "selected_clone": reference_clone,
            "selected_clone_recovered_strict": bool(reference_metrics["recovered_strict"][reference_clone].item()),
            "serialized": False,
        },
        "mpc_contract": {
            "method": "receding_horizon_random_shooting_on_local_task_error_model",
            "active_joint_ids": list(HANDOFF_JOINT_IDS),
            "active_joint_roles": ["index", "middle", "thumb"],
            "task_dimension": 11,
            "constraints": ["support_force", "height", "axis_alignment", "aggregate_contact", "residual_bound"],
            "optimized_sequence_serialized": False,
        },
        "jacobian_identification": {
            "method": "three_stage_antithetic_one_step_joint_target_probes",
            "serialized": False,
            "diagnostics": jacobian_diagnostics,
        },
        "controller_candidates_preregistered": O1_19_MPC_CONTROLLERS,
        "perturbation_protocol": {
            "calibration_seed_base": SEED + 600000,
            "held_out_seed_base": SEED + 700000,
            "same_perturbations_matched_across_modes": True,
            "calibration_and_held_out_seeds_disjoint": True,
            "controller_selection_nominal_unsafe_tolerance_pp": 2.0,
        },
        "calibration": calibration,
        "selected_mpc_controller": selected["mode"],
        "selected_mpc_passed_calibration_safety": bool(eligible_mpc),
        "selection_fallback_used_for_diagnostic_comparison": not bool(eligible_mpc),
        "held_out_evaluation": evaluation,
        "widening_gate": widening_gate,
        "widening_gate_pass": all(widening_gate.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "constraints": {
            "PPO_trained": False,
            "reward_modified": False,
            "O0_modified": False,
            "teacher_generated": False,
            "CEM_trajectory_serialized": False,
            "MPC_sequence_serialized": False,
            "reference_trajectory_serialized": False,
            "reset_dataset_generated": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "O1_19_RAW="
        + json.dumps(
            {
                "output": str(output),
                "selected_mpc_controller": selected["mode"],
                "widening_gate_pass": report["widening_gate_pass"],
            }
        ),
        flush=True,
    )
    del direct.forced_bank_ids
    env.close()


def main() -> None:
    if args.mpc_audit:
        _run_o1_19()
        return
    if args.task_space_audit:
        _run_o1_18()
        return
    if args.closed_loop_audit:
        _run_o1_17()
        return
    if len(args.bank_ids) != len(args.horizons):
        raise ValueError("--bank-ids and --horizons must have equal length")
    started = time.perf_counter()
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.population)
    cfg.seed = SEED
    control_dt = float(cfg.sim.dt * cfg.decimation)
    cfg.episode_length_s = (max(args.horizons) + CONTACT_REBUILD_STEPS + 8) * control_dt
    bank_path = Path(cfg.bundle_path).resolve()
    bank_hash_before = _sha256(bank_path)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    if bank_hash_before != direct.cfg.expected_bundle_sha256:
        raise RuntimeError("Frozen O0 bank hash mismatch")
    valid_p3 = direct.reset_rows_by_canonical[2]
    requested = torch.as_tensor(args.bank_ids, device=direct.device)
    if not bool(torch.isin(requested, valid_p3).all().item()):
        raise ValueError("Every O1.16 bank ID must be frozen P3/train")

    scenarios = _scenario_definitions()
    state_reports = []
    for state_index, (bank_id, horizon) in enumerate(zip(args.bank_ids, args.horizons, strict=True)):
        plan, search = _cem_plan(env, direct, bank_id, horizon)
        forced = torch.full(
            (args.population,), bank_id, dtype=torch.long, device=direct.device
        )
        plan_batch = plan.unsqueeze(0).repeat(args.population, 1, 1)
        scenario_reports = []
        for scenario_index, scenario in enumerate(scenarios):
            metrics = _rollout(
                env,
                direct,
                plan_batch,
                forced,
                horizon,
                perturbation=scenario["scales"],
                perturb_seed=SEED + 100000 * state_index + scenario_index,
            )
            scenario_reports.append({**scenario, **_summary(metrics)})
        state_reports.append(
            {
                "bank_id": bank_id,
                "horizon_steps": horizon,
                "horizon_ms": 1000.0 * horizon * control_dt,
                "search": search,
                "scenarios": scenario_reports,
            }
        )
        print(
            "O1_16_STATE="
            + json.dumps(
                {
                    "bank_id": bank_id,
                    "nominal": scenario_reports[0]["recovered_strict_rate"],
                    "joint_L1": next(
                        row["recovered_strict_rate"]
                        for row in scenario_reports
                        if row["name"] == "joint_L1"
                    ),
                }
            ),
            flush=True,
        )

    bank_hash_after = _sha256(bank_path)
    if bank_hash_after != bank_hash_before:
        raise RuntimeError("Frozen O0 bank changed during O1.16")
    report = {
        "status": "PASS_RAW_LOCAL_BASIN_AUDIT",
        "phase": "O1.16 Local Handoff Basin Perturbation Audit",
        "artifact_role": "aggregate_replay_metrics_no_action_plan_or_reset_state_serialization",
        "task": args.task,
        "bank_sha256_before": bank_hash_before,
        "bank_sha256_after": bank_hash_after,
        "search": {
            "population": args.population,
            "iterations": args.iterations,
            "waypoints": args.waypoints,
            "max_offset_rad": args.max_offset_rad,
            "action_plan_serialized": False,
        },
        "perturbation_design": {
            "clone_count_per_scenario": args.population,
            "scales": {key: list(value) for key, value in SCALES.items()},
            "joint_position_and_target_shifted_together": True,
            "contact_force_directly_perturbed": False,
            "contact_rebuild_steps": CONTACT_REBUILD_STEPS,
        },
        "states": state_reports,
        "elapsed_seconds": time.perf_counter() - started,
        "constraints": {
            "PPO_trained": False,
            "reward_modified": False,
            "O0_modified": False,
            "teacher_generated": False,
            "action_plan_serialized": False,
            "reset_dataset_generated": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("O1_16_RAW=" + json.dumps({"output": str(output), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    del direct.forced_bank_ids
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
