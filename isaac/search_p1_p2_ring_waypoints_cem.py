from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Search a three-segment ring-only P1-to-P2 collision-avoiding path."
)
parser.add_argument("--task", default="Dex5-P1P2-RingOnly120HzH1-Direct-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--population", type=int, default=64)
parser.add_argument("--replicas", type=int, default=4)
parser.add_argument("--iterations", type=int, default=6)
parser.add_argument("--elite-fraction", type=float, default=0.20)
parser.add_argument(
    "--initial-std-fraction",
    type=float,
    default=0.10,
    help="Search-range fraction used as CEM std when warm-starting a plan.",
)
parser.add_argument("--rollout-steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=2821)
parser.add_argument(
    "--fixed-clearance-offset",
    type=float,
    nargs=4,
    help="Fixed ring target offset from P1 for the first collision-free waypoint.",
)
parser.add_argument("--plan-input", type=Path)
parser.add_argument(
    "--initial-plan",
    type=Path,
    help="Warm-start the CEM mean from a complete parameter plan while retaining search.",
)
parser.add_argument("--p2-modes", type=Path)
parser.add_argument("--p2-mode", type=int, default=-1)
parser.add_argument(
    "--source-ids",
    type=Path,
    help="Frozen train-only global P1 source IDs used as the reset subset.",
)
parser.add_argument(
    "--source-bundle",
    type=Path,
    help="Optional rebuilt P1 bundle; target statistics remain frozen to the original train bank.",
)
parser.add_argument(
    "--tiny-tim-compensation",
    action="store_true",
    help="Deprecated alias for --tim-assist.",
)
parser.add_argument(
    "--tim-assist",
    action="store_true",
    help="Apply bounded proportional TIM closing when contact force is below target.",
)
parser.add_argument("--tim-contact-target-n", type=float, default=0.06)
parser.add_argument(
    "--search-tim-gains",
    action="store_true",
    help="Add nine signed bounded TIM joint gains to the CEM parameter vector.",
)
parser.add_argument(
    "--search-middle-anchor",
    action="store_true",
    help="Search a bounded +/-0.01 rad offset for the three middle P1 flexion targets.",
)
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


PARAMETER_NAMES = [
    *(f"retreat_fraction_j{joint}" for joint in range(4)),
    *(f"wrap_fraction_j{joint}" for joint in range(4)),
    "retreat_steps",
    "wrap_steps",
    "settle_steps",
    "settle_gain",
]
TIM_GAIN_NAMES = [
    *(f"thumb_contact_gain_j{joint}" for joint in range(3)),
    *(f"index_contact_gain_j{joint}" for joint in range(3)),
    *(f"middle_contact_gain_j{joint}" for joint in range(3)),
]
LOW = np.asarray([-0.55] * 4 + [-0.10] * 4 + [8.0, 12.0, 16.0, 0.15], dtype=np.float32)
HIGH = np.asarray([0.35] * 4 + [1.35] * 4 + [30.0, 44.0, 56.0, 1.00], dtype=np.float32)
INITIAL_MEAN = np.asarray(
    [-0.18, -0.12, -0.20, -0.15, 0.35, 0.70, 0.82, 0.78, 16.0, 26.0, 34.0, 0.45],
    dtype=np.float32,
)
INITIAL_STD = 0.24 * (HIGH - LOW)
TIM_GAIN_LOW = np.full(9, -1.0, dtype=np.float32)
TIM_GAIN_HIGH = np.full(9, 1.0, dtype=np.float32)
TIM_GAIN_INITIAL = np.asarray(
    [0.0, 0.65, 1.0, 0.0, 0.65, 1.0, 0.0, 0.65, 1.0], dtype=np.float32
)
MIDDLE_ANCHOR_NAMES = [f"middle_anchor_offset_j{joint}" for joint in range(3)]
MIDDLE_ANCHOR_LOW = np.full(3, -0.01, dtype=np.float32)
MIDDLE_ANCHOR_HIGH = np.full(3, 0.01, dtype=np.float32)
MIDDLE_ANCHOR_INITIAL = np.zeros(3, dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def load_plan(
    path: Path,
    parameter_names: list[str],
    low: np.ndarray,
    high: np.ndarray,
    defaults: np.ndarray | None = None,
) -> np.ndarray:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    values = payload.get("best_parameters", payload.get("parameters"))
    if isinstance(values, dict):
        if defaults is None:
            values = [values[name] for name in parameter_names]
        else:
            values = [
                values.get(name, float(defaults[index]))
                for index, name in enumerate(parameter_names)
            ]
    plan = np.asarray(values, dtype=np.float32)
    if plan.shape != low.shape:
        raise ValueError(f"Expected {low.size} plan values, got {plan.shape}")
    return np.clip(plan, low, high)


def target_for_step(
    step: int,
    parameters: torch.Tensor,
    ring_source: torch.Tensor,
    ring_final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = ring_final - ring_source
    if args.fixed_clearance_offset is None:
        waypoint1 = ring_source + parameters[:, 0:4] * delta
    else:
        clearance = torch.as_tensor(
            args.fixed_clearance_offset, dtype=ring_source.dtype, device=ring_source.device
        )
        waypoint1 = ring_source + clearance.unsqueeze(0)
    waypoint2 = ring_source + parameters[:, 4:8] * delta
    duration1 = parameters[:, 8].round().clamp_min(1.0)
    duration2 = parameters[:, 9].round().clamp_min(1.0)
    duration3 = parameters[:, 10].round().clamp_min(1.0)
    time = torch.full_like(duration1, float(step + 1))
    tau1 = smoothstep(time / duration1)
    tau2 = smoothstep((time - duration1) / duration2)
    tau3 = smoothstep((time - duration1 - duration2) / duration3)
    first = ring_source + tau1.unsqueeze(-1) * (waypoint1 - ring_source)
    second = waypoint1 + tau2.unsqueeze(-1) * (waypoint2 - waypoint1)
    third = waypoint2 + tau3.unsqueeze(-1) * (ring_final - waypoint2)
    desired = torch.where(
        (time <= duration1).unsqueeze(-1),
        first,
        torch.where((time <= duration1 + duration2).unsqueeze(-1), second, third),
    )
    settling = time > duration1 + duration2
    gain = torch.where(settling, parameters[:, 11], torch.ones_like(time))
    return desired, gain


def run_rollout(direct, env, candidate_parameters: torch.Tensor, candidate_ids: torch.Tensor) -> dict:
    num_candidates = int(candidate_ids.max().item()) + 1
    num_envs = direct.num_envs
    direct._reset_idx(torch.arange(num_envs, device=direct.device))
    if args.search_middle_anchor:
        offset_start = 12 + (9 if args.search_tim_gains else 0)
        middle_offset = candidate_parameters[:, offset_start : offset_start + 3]
        direct.support_anchor_targets[:, 5:8] += middle_offset
        direct.policy_targets[:, 5:8] = direct.support_anchor_targets[:, 5:8]
    ring_source = direct.source_targets[:, 8:12].clone()
    ring_start_qpos = direct.hand_qpos[:, 8:12].clone()
    ring_final = direct.p2_target_ring_qpos.unsqueeze(0).expand_as(ring_source)
    increment = 0.25 * float(direct.cfg.action_scale)
    active = torch.ones(num_envs, dtype=torch.bool, device=direct.device)
    loss_streak = torch.zeros(num_envs, dtype=torch.long, device=direct.device)
    max_loss_streak = torch.zeros_like(loss_streak)
    hard_failure = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
    drop = torch.zeros_like(hard_failure)
    success = torch.zeros_like(hard_failure)
    geometry_hit = torch.zeros_like(hard_failure)
    min_distance = torch.full((num_envs,), float("inf"), device=direct.device)
    triad_steps = torch.zeros(num_envs, device=direct.device)
    primary_contact_steps = torch.zeros((num_envs, 3), device=direct.device)
    sampled_steps = torch.zeros(num_envs, device=direct.device)
    max_penetration = torch.zeros(num_envs, device=direct.device)
    max_position_drift = torch.zeros(num_envs, device=direct.device)
    max_axis_drift = torch.zeros(num_envs, device=direct.device)
    min_ring_joint_error = torch.full((num_envs,), float("inf"), device=direct.device)
    max_ring_joint_travel = torch.zeros(num_envs, device=direct.device)

    for step in range(args.rollout_steps):
        desired, gain = target_for_step(step, candidate_parameters, ring_source, ring_final)
        actions = torch.zeros((num_envs, direct.cfg.action_space), device=direct.device)
        actions[:, 8:12] = torch.clamp(
            gain.unsqueeze(-1) * (desired - direct.policy_targets[:, 8:12]) / increment,
            -1.0,
            1.0,
        )
        if args.tim_assist or args.tiny_tim_compensation:
            target_force = max(float(args.tim_contact_target_n), 1.0e-6)
            force_deficit = torch.clamp(
                (target_force - direct.contact_force_norm[:, :3]) / target_force,
                0.0,
                1.0,
            )
            # Contact order is thumb/index/middle; policy joint blocks are
            # index/middle/ring/pinky/thumb.
            tim_gains = (
                candidate_parameters[:, 12:21].reshape(-1, 3, 3)
                if candidate_parameters.shape[1] >= 21
                else torch.ones((num_envs, 3, 3), device=direct.device)
            )
            for finger, joint_slice in enumerate(
                (slice(17, 20), slice(1, 4), slice(5, 8))
            ):
                actions[:, joint_slice] = (
                    force_deficit[:, finger].unsqueeze(-1) * tim_gains[:, finger]
                )
        _, _, terminated, truncated, _ = env.step(actions)
        direct._compute_intermediate_values()
        _, progress, target_distance, _ = direct._ring_geometry()
        _, metrics = direct._instantaneous_gate()
        triad = (direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n).all(dim=-1)
        loss_streak = torch.where(triad, torch.zeros_like(loss_streak), loss_streak + 1)
        max_loss_streak = torch.maximum(max_loss_streak, loss_streak)
        axis_drift = metrics["axis_drift"]
        position_drift = metrics["position_drift"]
        penetration = direct.episode_max_penetration
        ring_joint_error = torch.linalg.norm(direct.hand_qpos[:, 8:12] - ring_final, dim=-1)
        ring_joint_travel = torch.linalg.norm(
            direct.hand_qpos[:, 8:12] - ring_start_qpos, dim=-1
        )
        finite = torch.isfinite(direct.hand_qpos).all(dim=-1) & torch.isfinite(direct.pen_pos).all(dim=-1)
        height_drop = direct.init_pen_pos[:, 2] - direct.pen_pos[:, 2]
        current_drop = height_drop > direct.cfg.drop_threshold
        gate_violation = (
            (loss_streak > 1)
            | (position_drift > 0.010)
            | (axis_drift > torch.deg2rad(torch.full_like(axis_drift, 8.0)))
            | (penetration > 0.00060)
            | current_drop
            | ~finite
        )
        catastrophic = current_drop | ~finite | (penetration > 0.0020)
        valid_sample = active
        min_distance = torch.where(valid_sample, torch.minimum(min_distance, target_distance), min_distance)
        triad_steps += (triad & valid_sample).float()
        primary_contact_steps += (
            (direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n)
            & valid_sample.unsqueeze(-1)
        ).float()
        sampled_steps += valid_sample.float()
        max_penetration = torch.where(valid_sample, torch.maximum(max_penetration, penetration), max_penetration)
        max_position_drift = torch.where(
            valid_sample, torch.maximum(max_position_drift, position_drift), max_position_drift
        )
        max_axis_drift = torch.where(valid_sample, torch.maximum(max_axis_drift, axis_drift), max_axis_drift)
        min_ring_joint_error = torch.where(
            valid_sample,
            torch.minimum(min_ring_joint_error, ring_joint_error),
            min_ring_joint_error,
        )
        max_ring_joint_travel = torch.where(
            valid_sample,
            torch.maximum(max_ring_joint_travel, ring_joint_travel),
            max_ring_joint_travel,
        )
        stage = int(direct.cfg.curriculum_stage)
        geometry = (
            (torch.abs(progress - float(direct.cfg.stage_target_progress[stage]))
             <= float(direct.cfg.stage_progress_tolerance[stage]))
            & (target_distance <= float(direct.cfg.stage_target_distance_m[stage]))
        )
        geometry_hit |= geometry & triad & valid_sample
        hard_failure |= gate_violation & valid_sample
        drop |= current_drop & valid_sample
        done = terminated | truncated
        success |= direct.last_completed_p1_p2_success & done & valid_sample
        # Keep rejected candidates alive long enough to expose approach-distance
        # shaping to CEM. Gate violations remain latched and cannot be accepted.
        active &= ~done & ~catastrophic
        if not active.any():
            break

    triad_ratio = triad_steps / sampled_steps.clamp_min(1.0)
    min_distance_mm = min_distance.clamp_max(1.0) * 1000.0
    score = (
        700.0 * success.float()
        + 160.0 * geometry_hit.float()
        + 80.0 * triad_ratio
        - 7.0 * min_distance_mm.clamp_max(60.0)
        - 120.0 * min_ring_joint_error
        - 500.0 * hard_failure.float()
        - 300.0 * drop.float()
        - 20.0 * max_loss_streak.float()
    )

    def reduce_mean(values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(num_candidates, device=direct.device)
        count = torch.zeros(num_candidates, device=direct.device)
        out.scatter_add_(0, candidate_ids, values.float())
        count.scatter_add_(0, candidate_ids, torch.ones_like(values, dtype=torch.float32))
        return out / count.clamp_min(1.0)

    result = {
        "score": reduce_mean(score),
        "success_rate": reduce_mean(success),
        "geometry_rate": reduce_mean(geometry_hit),
        "hard_failure_rate": reduce_mean(hard_failure),
        "drop_rate": reduce_mean(drop),
        "triad_ratio": reduce_mean(triad_ratio),
        "min_distance_mm": reduce_mean(min_distance_mm),
        "max_loss_streak": reduce_mean(max_loss_streak),
        "max_penetration_mm": reduce_mean(max_penetration * 1000.0),
        "max_position_drift_mm": reduce_mean(max_position_drift * 1000.0),
        "max_axis_drift_deg": reduce_mean(torch.rad2deg(max_axis_drift)),
        "min_ring_joint_error_rad": reduce_mean(min_ring_joint_error),
        "max_ring_joint_travel_rad": reduce_mean(max_ring_joint_travel),
    }
    for finger, name in enumerate(("thumb", "index", "middle")):
        result[f"{name}_contact_ratio"] = reduce_mean(
            primary_contact_steps[:, finger] / sampled_steps.clamp_min(1.0)
        )
    return result


def metrics_row(metrics: dict, candidate: int) -> dict:
    return {key: float(value[candidate].item()) for key, value in metrics.items()}


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    parameter_names = list(PARAMETER_NAMES)
    low_np = LOW
    high_np = HIGH
    initial_mean_np = INITIAL_MEAN
    initial_std_np = INITIAL_STD
    if args.search_tim_gains:
        if not (args.tim_assist or args.tiny_tim_compensation):
            raise ValueError("--search-tim-gains requires --tim-assist")
        parameter_names += TIM_GAIN_NAMES
        low_np = np.concatenate((LOW, TIM_GAIN_LOW))
        high_np = np.concatenate((HIGH, TIM_GAIN_HIGH))
        initial_mean_np = np.concatenate((INITIAL_MEAN, TIM_GAIN_INITIAL))
        initial_std_np = np.concatenate((INITIAL_STD, np.full(9, 0.45, dtype=np.float32)))
    if args.search_middle_anchor:
        parameter_names += MIDDLE_ANCHOR_NAMES
        low_np = np.concatenate((low_np, MIDDLE_ANCHOR_LOW))
        high_np = np.concatenate((high_np, MIDDLE_ANCHOR_HIGH))
        initial_mean_np = np.concatenate((initial_mean_np, MIDDLE_ANCHOR_INITIAL))
        initial_std_np = np.concatenate(
            (initial_std_np, np.full(3, 0.0045, dtype=np.float32))
        )
    if args.initial_plan:
        initial_mean_np = load_plan(
            args.initial_plan,
            parameter_names,
            low_np,
            high_np,
            defaults=initial_mean_np,
        )
        initial_std_np = float(args.initial_std_fraction) * (high_np - low_np)
    if args.plan_input:
        population = 1
        replicas = args.replicas
    else:
        population = args.population
        replicas = args.replicas
    num_envs = population * replicas
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = args.seed
    cfg.episode_length_s = max(
        float(cfg.episode_length_s), float(args.rollout_steps + 4) / 120.0
    )
    if cfg.split_name != "train":
        raise ValueError("Waypoint search is restricted to the train split")
    source_ids_path = None
    source_bundle_path = None
    if args.source_bundle:
        source_bundle_path = args.source_bundle.resolve()
        cfg.bundle_path = str(source_bundle_path)
        # Rebuilt source rows already contain their physical state and support
        # targets; original-bank scripted route lookups use a different ID space.
        cfg.teacher_prior_state_map_path = None
    if args.source_ids:
        source_ids_path = args.source_ids.resolve()
        source_ids = np.asarray(np.load(source_ids_path, allow_pickle=False), dtype=np.int64)
        if source_ids.ndim != 1 or source_ids.size == 0:
            raise ValueError("--source-ids must contain a non-empty 1-D ID array")
        cfg.split_ids_path = str(source_ids_path)
    selected_mode = None
    if args.p2_modes:
        mode_payload = json.loads(args.p2_modes.resolve().read_text(encoding="utf-8"))
        if args.p2_mode < 0 or args.p2_mode >= len(mode_payload["modes"]):
            raise ValueError("--p2-mode must select an available mode")
        selected_mode = mode_payload["modes"][args.p2_mode]
        cfg.target_p2_ring_qpos_override = tuple(selected_mode["ring_qpos_center"])
        cfg.target_p2_ring_center_override = tuple(selected_mode["ring_feature_center_m"])
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()
    candidate_ids = torch.arange(population, device=direct.device).repeat_interleave(replicas)
    low = torch.tensor(low_np, device=direct.device)
    high = torch.tensor(high_np, device=direct.device)
    history = []

    if args.plan_input:
        best = load_plan(args.plan_input, parameter_names, low_np, high_np)
        population_values = torch.tensor(best, device=direct.device).unsqueeze(0)
        parameters = population_values[candidate_ids]
        metrics = run_rollout(direct, env, parameters, candidate_ids)
        best_metrics = metrics_row(metrics, 0)
    else:
        mean = torch.tensor(initial_mean_np, device=direct.device)
        std = torch.tensor(initial_std_np, device=direct.device)
        generator = torch.Generator(device=direct.device)
        generator.manual_seed(args.seed)
        best_score = -float("inf")
        best = initial_mean_np.copy()
        best_metrics = {}
        elite_count = max(2, int(round(population * args.elite_fraction)))
        for iteration in range(args.iterations):
            noise = torch.randn(
                (population, low_np.size), generator=generator, device=direct.device
            )
            population_values = torch.maximum(
                torch.minimum(mean.unsqueeze(0) + std.unsqueeze(0) * noise, high), low
            )
            if iteration == 0:
                population_values[0] = mean
            parameters = population_values[candidate_ids]
            metrics = run_rollout(direct, env, parameters, candidate_ids)
            elite_ids = torch.topk(metrics["score"], k=elite_count).indices
            elites = population_values[elite_ids]
            mean = 0.25 * mean + 0.75 * elites.mean(dim=0)
            std = torch.maximum(
                0.25 * (high - low) / 20.0,
                0.30 * std + 0.70 * elites.std(dim=0, unbiased=False),
            )
            iteration_best = int(torch.argmax(metrics["score"]).item())
            row = metrics_row(metrics, iteration_best)
            row["iteration"] = iteration
            row["parameters"] = {
                name: float(population_values[iteration_best, index].item())
                for index, name in enumerate(parameter_names)
            }
            history.append(row)
            if row["score"] > best_score:
                best_score = row["score"]
                best = population_values[iteration_best].detach().cpu().numpy()
                best_metrics = {key: value for key, value in row.items() if key != "parameters"}

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task": args.task,
        "split": "train",
        "source_ids": str(source_ids_path) if source_ids_path else None,
        "source_ids_sha256": sha256(source_ids_path) if source_ids_path else None,
        "source_id_count": int(source_ids.size) if source_ids_path else None,
        "source_bundle": str(source_bundle_path) if source_bundle_path else None,
        "source_bundle_sha256": sha256(source_bundle_path) if source_bundle_path else None,
        "seed": args.seed,
        "population": population,
        "replicas": replicas,
        "iterations": 0 if args.plan_input else args.iterations,
        "initial_plan": str(args.initial_plan.resolve()) if args.initial_plan else None,
        "initial_plan_sha256": sha256(args.initial_plan.resolve()) if args.initial_plan else None,
        "rollout_steps": args.rollout_steps,
        "initial_std_fraction": float(args.initial_std_fraction),
        "fixed_clearance_offset_rad": args.fixed_clearance_offset,
        "search_continues_after_gate_violation": True,
        "ring_only_action_indices": [8, 9, 10, 11],
        "ring_is_primary_mover": True,
        "tim_assist": bool(args.tim_assist or args.tiny_tim_compensation),
        "tim_assist_contact_to_action_indices": {
            "thumb": [17, 18, 19],
            "index": [1, 2, 3],
            "middle": [5, 6, 7],
        },
        "tim_contact_target_n": float(args.tim_contact_target_n),
        "search_tim_gains": bool(args.search_tim_gains),
        "search_middle_anchor": bool(args.search_middle_anchor),
        "maximum_middle_anchor_offset_rad": 0.01 if args.search_middle_anchor else 0.0,
        "pinky_action_expected_zero": True,
        "maximum_tim_target_offset_rad": (
            float(cfg.action_scale * cfg.support_action_mask_scale)
            if (args.tim_assist or args.tiny_tim_compensation) else 0.0
        ),
        "p2_mode": args.p2_mode,
        "selected_mode": selected_mode,
        "best_parameters": {
            name: float(best[index]) for index, name in enumerate(parameter_names)
        },
        "best_metrics": best_metrics,
        "history": history,
        "acceptance": {
            "success_rate_min": 0.20,
            "drop_rate_max": 0.05,
            "accepted": bool(
                best_metrics.get("success_rate", 0.0) >= 0.20
                and best_metrics.get("drop_rate", 1.0) < 0.05
            ),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(output)
    result["output_sha256"] = sha256(output)
    print("P1_P2_RING_WAYPOINT_CEM=" + json.dumps(result, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
