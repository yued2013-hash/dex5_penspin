from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ONLY = os.environ.get("DEX5_PAIRED_MANIFOLD_CORE_ONLY") == "1"
if not CORE_ONLY:
    parser = argparse.ArgumentParser(
        description="Discover a dynamically reachable ring-under P2 manifold from a stable P1."
    )
    parser.add_argument("--task", default="Dex5-P1P2-TIMAssist005120HzH1-Direct-v0")
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument("--rollout-steps", type=int, default=180)
    parser.add_argument("--hold-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2821)
    parser.add_argument(
        "--fixed-clearance-offset",
        type=float,
        nargs=4,
        required=True,
        help="Verified unloaded ring target offset used as the first waypoint.",
    )
    parser.add_argument("--tim-contact-target-n", type=float, default=0.06)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_args]
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
else:
    args = None
    simulation_app = None

import gymnasium as gym
import numpy as np
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


PARAMETER_NAMES = [
    *(f"waypoint2_offset_j{joint}" for joint in range(4)),
    *(f"endpoint_offset_j{joint}" for joint in range(4)),
    "clearance_steps",
    "wrap_steps",
    "settle_steps",
    "settle_gain",
    *(f"thumb_contact_gain_j{joint}" for joint in range(3)),
    *(f"index_contact_gain_j{joint}" for joint in range(3)),
    *(f"middle_contact_gain_j{joint}" for joint in range(3)),
]
LOW = np.asarray(
    [-0.55, -0.35, -0.15, -0.15]
    + [-0.55, -0.15, 0.18, 0.18]
    + [12.0, 20.0, 20.0, 0.12]
    + [-1.0] * 9,
    dtype=np.float32,
)
HIGH = np.asarray(
    [0.20, 0.55, 1.00, 1.00]
    + [0.10, 0.55, 1.05, 1.05]
    + [32.0, 56.0, 48.0, 0.80]
    + [1.0] * 9,
    dtype=np.float32,
)
INITIAL_MEAN = np.asarray(
    [-0.15, 0.25, 0.42, 0.38]
    + [-0.35, 0.28, 0.62, 0.65]
    + [24.0, 42.0, 48.0, 0.28]
    + [0.0, 0.65, 1.0, 0.0, 0.55, 0.85, 0.0, 0.55, 0.90],
    dtype=np.float32,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def interval_excess(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.relu(lower - value) + torch.relu(value - upper)


def reduce_mean(values: torch.Tensor, candidate_ids: torch.Tensor, count: int) -> torch.Tensor:
    result = torch.zeros(count, device=values.device)
    samples = torch.zeros(count, device=values.device)
    result.scatter_add_(0, candidate_ids, values.float())
    samples.scatter_add_(0, candidate_ids, torch.ones_like(values, dtype=torch.float32))
    return result / samples.clamp_min(1.0)


def run_rollout(direct, env, parameters: torch.Tensor, candidate_ids: torch.Tensor) -> dict:
    num_envs = direct.num_envs
    num_candidates = int(candidate_ids.max().item()) + 1
    env_ids = torch.arange(num_envs, device=direct.device)
    direct.forced_bank_ids = torch.zeros(num_envs, dtype=torch.long, device=direct.device)
    direct._reset_idx(env_ids)
    direct._compute_intermediate_values()

    source = direct.policy_targets[:, 8:12].clone()
    clearance = source + torch.as_tensor(
        args.fixed_clearance_offset, dtype=source.dtype, device=source.device
    ).unsqueeze(0)
    waypoint2 = source + parameters[:, 0:4]
    endpoint = source + parameters[:, 4:8]
    waypoint2 = torch.maximum(torch.minimum(waypoint2, direct.joint_upper[:, 8:12]), direct.joint_lower[:, 8:12])
    endpoint = torch.maximum(torch.minimum(endpoint, direct.joint_upper[:, 8:12]), direct.joint_lower[:, 8:12])
    durations = parameters[:, 8:11].round().clamp_min(1.0)
    settle_gain = parameters[:, 11]
    tim_gains = parameters[:, 12:21].reshape(-1, 3, 3)
    increment = 0.25 * float(direct.cfg.action_scale)

    initial_pen_pos = direct.pen_pos.clone()
    initial_pen_axis = direct.pen_axis.clone()
    pose_history = [initial_pen_pos.clone() for _ in range(13)]
    axis_history = [initial_pen_axis.clone() for _ in range(13)]
    hold_count = torch.zeros(num_envs, dtype=torch.long, device=direct.device)
    max_hold = torch.zeros_like(hold_count)
    geometry_count = torch.zeros_like(hold_count)
    max_geometry_hold = torch.zeros_like(hold_count)
    stable_geometry_count = torch.zeros_like(hold_count)
    max_stable_geometry_hold = torch.zeros_like(hold_count)
    accepted = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
    active = torch.ones_like(accepted)
    drop = torch.zeros_like(accepted)
    max_penetration = torch.zeros(num_envs, device=direct.device)
    max_position_drift = torch.zeros(num_envs, device=direct.device)
    max_axis_drift = torch.zeros(num_envs, device=direct.device)
    best_semantic_cost = torch.full((num_envs,), float("inf"), device=direct.device)
    best_ring_gap = torch.full_like(best_semantic_cost, float("inf"))
    best_feature = torch.zeros((num_envs, 3), device=direct.device)
    captured = {
        "hand_qpos": torch.zeros((num_envs, 20), device=direct.device),
        "hand_qvel": torch.zeros((num_envs, 20), device=direct.device),
        "pen_pose": torch.zeros((num_envs, 7), device=direct.device),
        "pen_velocity": torch.zeros((num_envs, 6), device=direct.device),
        "policy_target": torch.zeros((num_envs, 20), device=direct.device),
        "ring_feature": torch.zeros((num_envs, 3), device=direct.device),
        "contact_force": torch.zeros((num_envs, 5), device=direct.device),
    }

    for step in range(args.rollout_steps):
        time = torch.full((num_envs,), float(step + 1), device=direct.device)
        d1, d2, d3 = durations.unbind(dim=-1)
        first = source + smoothstep(time / d1).unsqueeze(-1) * (clearance - source)
        second = clearance + smoothstep((time - d1) / d2).unsqueeze(-1) * (waypoint2 - clearance)
        third = waypoint2 + smoothstep((time - d1 - d2) / d3).unsqueeze(-1) * (endpoint - waypoint2)
        desired = torch.where(
            (time <= d1).unsqueeze(-1),
            first,
            torch.where((time <= d1 + d2).unsqueeze(-1), second, third),
        )
        final_phase = time > d1 + d2
        gain = torch.where(final_phase, settle_gain, torch.ones_like(settle_gain))
        actions = torch.zeros((num_envs, direct.cfg.action_space), device=direct.device)
        actions[:, 8:12] = torch.clamp(
            gain.unsqueeze(-1) * (desired - direct.policy_targets[:, 8:12]) / increment,
            -1.0,
            1.0,
        )
        target_force = max(float(args.tim_contact_target_n), 1.0e-6)
        force_deficit = torch.clamp(
            (target_force - direct.contact_force_norm[:, :3]) / target_force, 0.0, 1.0
        )
        for finger, joint_slice in enumerate((slice(17, 20), slice(1, 4), slice(5, 8))):
            actions[:, joint_slice] = force_deficit[:, finger].unsqueeze(-1) * tim_gains[:, finger]
        _, _, terminated, truncated, _ = env.step(actions)
        direct._compute_intermediate_values()

        pose_history.append(direct.pen_pos.clone())
        axis_history.append(direct.pen_axis.clone())
        old_pos = pose_history.pop(0)
        old_axis = axis_history.pop(0)
        position_speed = torch.linalg.norm(direct.pen_pos - old_pos, dim=-1) / 0.1
        axis_delta = torch.acos(torch.sum(direct.pen_axis * old_axis, dim=-1).abs().clamp(-1.0, 1.0))
        axis_speed = axis_delta / 0.1
        feature = direct._ring_axis_features()
        axial, lateral, up = feature.unbind(dim=-1)
        ring_gap = direct.tip_surface_dist[:, 3] - direct.fingertip_radii[0, 3]
        triad = (direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n).all(dim=-1)
        position_drift = torch.linalg.norm(direct.pen_pos - initial_pen_pos, dim=-1)
        axis_drift = torch.acos(
            torch.sum(direct.pen_axis * initial_pen_axis, dim=-1).abs().clamp(-1.0, 1.0)
        )
        penetration = direct.episode_max_penetration
        height_drop = initial_pen_pos[:, 2] - direct.pen_pos[:, 2]
        current_drop = height_drop > direct.cfg.drop_threshold
        finite = torch.isfinite(direct.hand_qpos).all(dim=-1) & torch.isfinite(direct.pen_pos).all(dim=-1)
        ring_under = (
            (axial >= -0.060)
            & (axial <= -0.020)
            & (lateral >= -0.010)
            & (lateral <= 0.026)
            & (up >= -0.026)
            & (up <= -0.004)
            & ((-up) >= torch.abs(lateral) + 0.001)
        )
        near_surface = (ring_gap >= -0.0006) & (ring_gap <= 0.0030)
        stable = (
            triad
            & (position_drift <= 0.005)
            & (axis_drift <= torch.deg2rad(torch.full_like(axis_drift, 4.0)))
            & (position_speed <= 0.030)
            & (axis_speed <= 1.5)
            & (penetration <= 0.0006)
            & (direct.contact_force_norm[:, 3] <= 0.30)
            & (direct.contact_force_norm[:, 4] <= 0.03)
            & ~current_drop
            & finite
        )
        eligible = final_phase & ring_under & near_surface & stable
        geometry_eligible = final_phase & ring_under & near_surface
        stable_geometry = geometry_eligible & triad & (position_drift <= 0.005) & (
            axis_drift <= torch.deg2rad(torch.full_like(axis_drift, 4.0))
        ) & (penetration <= 0.0006) & ~current_drop & finite
        geometry_count = torch.where(
            geometry_eligible, geometry_count + 1, torch.zeros_like(geometry_count)
        )
        stable_geometry_count = torch.where(
            stable_geometry,
            stable_geometry_count + 1,
            torch.zeros_like(stable_geometry_count),
        )
        max_geometry_hold = torch.maximum(max_geometry_hold, geometry_count)
        max_stable_geometry_hold = torch.maximum(
            max_stable_geometry_hold, stable_geometry_count
        )
        hold_count = torch.where(eligible, hold_count + 1, torch.zeros_like(hold_count))
        max_hold = torch.maximum(max_hold, hold_count)

        semantic_cost = (
            1200.0 * interval_excess(axial, -0.060, -0.020)
            + 1600.0 * interval_excess(lateral, -0.010, 0.026)
            + 1600.0 * interval_excess(up, -0.026, -0.004)
            + 1600.0 * torch.relu(torch.abs(lateral) + 0.001 + up)
            + 1800.0 * interval_excess(ring_gap, -0.0006, 0.0030)
            + 8.0 * (~triad).float()
            + 600.0 * torch.relu(position_drift - 0.005)
            + 8.0 * torch.relu(torch.rad2deg(axis_drift) - 4.0)
        )
        improved = semantic_cost < best_semantic_cost
        best_semantic_cost = torch.where(improved, semantic_cost, best_semantic_cost)
        best_ring_gap = torch.where(improved, ring_gap, best_ring_gap)
        best_feature = torch.where(improved.unsqueeze(-1), feature, best_feature)
        just_accepted = (hold_count >= args.hold_steps) & ~accepted
        mask = just_accepted.unsqueeze(-1)
        pen_pose = torch.cat((direct.pen_pos, direct.pen_quat), dim=-1)
        pen_velocity = torch.cat((direct.pen_linvel, direct.pen_angvel), dim=-1)
        current = {
            "hand_qpos": direct.hand_qpos,
            "hand_qvel": direct.hand_qvel,
            "pen_pose": pen_pose,
            "pen_velocity": pen_velocity,
            "policy_target": direct.policy_targets,
            "ring_feature": feature,
            "contact_force": direct.contact_force_norm,
        }
        for key, value in current.items():
            captured[key] = torch.where(mask, value, captured[key])
        accepted |= just_accepted
        max_penetration = torch.maximum(max_penetration, penetration)
        max_position_drift = torch.maximum(max_position_drift, position_drift)
        max_axis_drift = torch.maximum(max_axis_drift, axis_drift)
        drop |= current_drop
        active &= ~(terminated | truncated) & finite

    score_env = (
        1200.0 * accepted.float()
        + 15.0 * max_hold.float().clamp_max(float(args.hold_steps))
        + 4.0 * max_geometry_hold.float().clamp_max(float(args.hold_steps))
        + 8.0 * max_stable_geometry_hold.float().clamp_max(float(args.hold_steps))
        - 12.0 * best_semantic_cost
        - 250.0 * drop.float()
    )
    result = {
        "score": reduce_mean(score_env, candidate_ids, num_candidates),
        "acceptance_rate": reduce_mean(accepted, candidate_ids, num_candidates),
        "max_hold_steps": reduce_mean(max_hold, candidate_ids, num_candidates),
        "max_geometry_hold_steps": reduce_mean(
            max_geometry_hold, candidate_ids, num_candidates
        ),
        "max_stable_geometry_hold_steps": reduce_mean(
            max_stable_geometry_hold, candidate_ids, num_candidates
        ),
        "best_semantic_cost": reduce_mean(best_semantic_cost, candidate_ids, num_candidates),
        "best_ring_gap_mm": reduce_mean(best_ring_gap * 1000.0, candidate_ids, num_candidates),
        "max_penetration_mm": reduce_mean(max_penetration * 1000.0, candidate_ids, num_candidates),
        "max_position_drift_mm": reduce_mean(max_position_drift * 1000.0, candidate_ids, num_candidates),
        "max_axis_drift_deg": reduce_mean(torch.rad2deg(max_axis_drift), candidate_ids, num_candidates),
        "drop_rate": reduce_mean(drop, candidate_ids, num_candidates),
        "best_feature": torch.stack(
            [reduce_mean(best_feature[:, i], candidate_ids, num_candidates) for i in range(3)], dim=-1
        ),
        "accepted_env": accepted,
        "captured": captured,
        "endpoint": endpoint,
    }
    return result


def cluster_archive(parameters: np.ndarray, features: np.ndarray, max_modes: int = 6) -> np.ndarray:
    if len(parameters) == 0:
        return np.empty(0, dtype=np.int64)
    values = np.concatenate((parameters[:, 4:8], features), axis=-1)
    scale = values.std(axis=0).clip(min=1.0e-4)
    values = (values - values.mean(axis=0)) / scale
    modes = min(max_modes, len(values))
    centers = values[np.linspace(0, len(values) - 1, modes, dtype=np.int64)].copy()
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(40):
        next_labels = np.argmin(((values[:, None] - centers[None]) ** 2).sum(axis=-1), axis=1)
        next_centers = centers.copy()
        for mode in range(modes):
            members = values[next_labels == mode]
            if len(members):
                next_centers[mode] = np.median(members, axis=0)
        if np.array_equal(labels, next_labels) and np.allclose(centers, next_centers):
            break
        labels, centers = next_labels, next_centers
    return labels


def run_search(env, cfg, source_bundle: Path, source_ids: Path) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    direct = env.unwrapped
    env.reset()
    candidate_ids = torch.arange(args.population, device=direct.device).repeat_interleave(args.replicas)
    low = torch.as_tensor(LOW, device=direct.device)
    high = torch.as_tensor(HIGH, device=direct.device)
    initial_mean = INITIAL_MEAN.copy()
    initial_report = getattr(args, "initial_report", None)
    if initial_report is not None and Path(initial_report).exists():
        payload = json.loads(Path(initial_report).read_text(encoding="utf-8"))
        values = payload.get("best_parameters", {})
        initial_mean = np.asarray(
            [values.get(name, INITIAL_MEAN[index]) for index, name in enumerate(PARAMETER_NAMES)],
            dtype=np.float32,
        )
    initial_mean = np.clip(initial_mean, LOW, HIGH)
    mean = torch.as_tensor(initial_mean, device=direct.device)
    std = 0.28 * (high - low)
    generator = torch.Generator(device=direct.device)
    generator.manual_seed(args.seed)
    elite_count = max(2, int(round(args.population * args.elite_fraction)))
    history: list[dict] = []
    archive: dict[str, list[np.ndarray]] = {
        "parameters": [], "hand_qpos": [], "hand_qvel": [], "pen_pose": [],
        "pen_velocity": [], "policy_target": [], "ring_feature": [], "contact_force": [],
    }
    best_score = -float("inf")
    best_parameters = INITIAL_MEAN.copy()
    best_metrics: dict[str, float | list[float]] = {}

    for iteration in range(args.iterations):
        noise = torch.randn((args.population, LOW.size), generator=generator, device=direct.device)
        population = torch.maximum(torch.minimum(mean.unsqueeze(0) + std.unsqueeze(0) * noise, high), low)
        if iteration == 0:
            population[0] = mean
        expanded = population[candidate_ids]
        metrics = run_rollout(direct, env, expanded, candidate_ids)
        elite_ids = torch.topk(metrics["score"], k=elite_count).indices
        elites = population[elite_ids]
        mean = 0.25 * mean + 0.75 * elites.mean(dim=0)
        std = torch.maximum(0.0125 * (high - low), 0.30 * std + 0.70 * elites.std(dim=0, unbiased=False))
        winner = int(torch.argmax(metrics["score"]).item())
        row = {
            "iteration": iteration,
            "score": float(metrics["score"][winner]),
            "acceptance_rate": float(metrics["acceptance_rate"][winner]),
            "max_hold_steps": float(metrics["max_hold_steps"][winner]),
            "max_geometry_hold_steps": float(metrics["max_geometry_hold_steps"][winner]),
            "max_stable_geometry_hold_steps": float(
                metrics["max_stable_geometry_hold_steps"][winner]
            ),
            "best_semantic_cost": float(metrics["best_semantic_cost"][winner]),
            "best_ring_gap_mm": float(metrics["best_ring_gap_mm"][winner]),
            "max_penetration_mm": float(metrics["max_penetration_mm"][winner]),
            "max_position_drift_mm": float(metrics["max_position_drift_mm"][winner]),
            "max_axis_drift_deg": float(metrics["max_axis_drift_deg"][winner]),
            "drop_rate": float(metrics["drop_rate"][winner]),
            "best_ring_feature_m": metrics["best_feature"][winner].detach().cpu().tolist(),
        }
        history.append(row)
        if row["score"] > best_score:
            best_score = row["score"]
            best_parameters = population[winner].detach().cpu().numpy()
            best_metrics = row.copy()

        accepted_env = metrics["accepted_env"]
        if accepted_env.any():
            accepted_ids = torch.nonzero(accepted_env, as_tuple=False).squeeze(-1)
            # One deterministic replica per accepted candidate is enough for the archive.
            _, unique_positions = np.unique(
                candidate_ids[accepted_ids].detach().cpu().numpy(), return_index=True
            )
            accepted_ids = accepted_ids[torch.as_tensor(unique_positions, device=direct.device)]
            for key in archive:
                if key == "parameters":
                    archive[key].append(expanded[accepted_ids].detach().cpu().numpy())
                else:
                    archive[key].append(metrics["captured"][key][accepted_ids].detach().cpu().numpy())

    arrays = {
        key: np.concatenate(values, axis=0) if values else np.empty((0, LOW.size if key == "parameters" else 0))
        for key, values in archive.items()
    }
    if len(arrays["parameters"]):
        rounded = np.round(np.concatenate((arrays["parameters"][:, 4:8], arrays["ring_feature"]), axis=-1), 4)
        _, unique = np.unique(rounded, axis=0, return_index=True)
        unique = np.sort(unique)
        arrays = {key: value[unique] for key, value in arrays.items()}
    labels = cluster_archive(arrays["parameters"], arrays["ring_feature"])
    bank_output = args.bank_output.resolve()
    bank_output.parent.mkdir(parents=True, exist_ok=True)
    if len(arrays["parameters"]):
        source_qpos = np.concatenate((arrays["hand_qpos"], arrays["pen_pose"]), axis=-1)
        source_qvel = np.concatenate((arrays["hand_qvel"], arrays["pen_velocity"]), axis=-1)
    else:
        source_qpos = np.empty((0, 27), dtype=np.float32)
        source_qvel = np.empty((0, 26), dtype=np.float32)
    np.savez_compressed(
        bank_output,
        source_qpos=source_qpos,
        source_qvel=source_qvel,
        source_target_qpos=arrays["policy_target"],
        source_policy_action=np.zeros((len(labels), 20), dtype=np.float32),
        source_teacher_actions=np.zeros((len(labels), 1, 20), dtype=np.float32),
        target_qpos=source_qpos.copy(),
        target_qvel=np.zeros((len(labels), 26), dtype=np.float32),
        ring_feature_m=arrays["ring_feature"],
        contact_force_n=arrays["contact_force"],
        search_parameters=arrays["parameters"],
        manifold_mode=labels,
        phase_ids=np.full(len(labels), 2, dtype=np.int64),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "task": args.task,
        "split": "train",
        "source_bundle": str(source_bundle),
        "source_bundle_sha256": sha256(source_bundle),
        "source_ids": str(source_ids),
        "source_ids_sha256": sha256(source_ids),
        "seed": args.seed,
        "population": args.population,
        "replicas": args.replicas,
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps,
        "hold_steps": args.hold_steps,
        "fixed_clearance_offset_rad": args.fixed_clearance_offset,
        "maximum_tim_target_offset_rad": float(cfg.action_scale * cfg.support_action_mask_scale),
        "best_parameters": {name: float(best_parameters[i]) for i, name in enumerate(PARAMETER_NAMES)},
        "best_metrics": best_metrics,
        "accepted_unique_states": int(len(labels)),
        "manifold_mode_counts": {
            str(mode): int(np.sum(labels == mode)) for mode in np.unique(labels)
        },
        "semantic_gate": {
            "ring_axial_m": [-0.060, -0.020],
            "ring_lateral_m": [-0.010, 0.026],
            "ring_under_m": [-0.026, -0.004],
            "ring_under_dominance": "-up >= abs(lateral) + 0.001 m",
            "ring_proxy_surface_gap_m": [-0.0006, 0.0030],
            "TIM_contact_force_n_min": float(cfg.p2_contact_force_n),
            "ring_force_n_max": 0.30,
            "pinky_force_n_max": 0.03,
            "pen_position_drift_m_max": 0.005,
            "pen_axis_drift_deg_max": 4.0,
            "finite_difference_linear_speed_m_s_max": 0.030,
            "finite_difference_axis_speed_rad_s_max": 1.5,
            "penetration_m_max": 0.0006,
        },
        "bank_output": str(bank_output),
        "bank_output_sha256": sha256(bank_output),
        "validation_or_test_accessed": False,
        "history": history,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_PAIRED_P2_MANIFOLD=" + json.dumps(report, ensure_ascii=False), flush=True)
    del direct.forced_bank_ids
    env.close()


def main() -> None:
    source_bundle = args.source_bundle.resolve()
    source_ids = args.source_ids.resolve()
    num_envs = args.population * args.replicas
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = args.seed
    if cfg.split_name != "train" or int(cfg.source_phase_id) != 1:
        raise ValueError("Paired P2 manifold search is restricted to train P1 states")
    if float(cfg.support_action_mask_scale) > 0.100001:
        raise ValueError("TIM compensation must not exceed 0.005 rad")
    cfg.bundle_path = str(source_bundle)
    cfg.teacher_prior_state_map_path = None
    selected_source_ids = np.asarray(np.load(source_ids, allow_pickle=False), dtype=np.int64)
    if selected_source_ids.ndim != 1 or selected_source_ids.size == 0:
        raise ValueError("--source-ids must contain a non-empty 1-D ID array")
    cfg.split_ids_path = str(source_ids)
    cfg.episode_length_s = max(
        float(cfg.episode_length_s), float(args.rollout_steps + 4) / 120.0
    )
    env = gym.make(args.task, cfg=cfg)
    run_search(env, cfg, source_bundle, source_ids)


if __name__ == "__main__" and not CORE_ONLY:
    try:
        main()
    finally:
        simulation_app.close()
