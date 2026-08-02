from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Search train-only per-joint ring targets for the P1-to-P2 curriculum."
)
parser.add_argument("--task", default="Dex5-P1P2-C2-Direct-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--joint-fractions",
    type=float,
    nargs="+",
    default=(0.45, 0.70, 0.95, 1.20),
    help="Fractions of each P1-to-P2 ring-joint delta to combine independently.",
)
for joint_id in range(4):
    parser.add_argument(
        f"--joint{joint_id}-fractions",
        type=float,
        nargs="+",
        default=None,
        help=f"Optional search values for ring joint {joint_id}; overrides --joint-fractions.",
    )
parser.add_argument(
    "--support-actions",
    type=float,
    nargs=3,
    metavar=("THUMB", "INDEX", "MIDDLE"),
    default=(0.40, 0.40, 0.0),
)
parser.add_argument("--thumb-actions", type=float, nargs="+", default=None)
parser.add_argument("--index-actions", type=float, nargs="+", default=None)
parser.add_argument("--middle-actions", type=float, nargs="+", default=None)
parser.add_argument("--replicas", type=int, default=2)
parser.add_argument("--steps", type=int, default=72)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--top-k", type=int, default=30)
parser.add_argument("--geometric-brake", action="store_true")
parser.add_argument("--brake-progress-margin", type=float, default=0.10)
parser.add_argument("--brake-distance-mm", type=float, default=12.0)
parser.add_argument("--brake-progress-margins", type=float, nargs="+", default=None)
parser.add_argument("--brake-distances-mm", type=float, nargs="+", default=None)
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


def main() -> None:
    # Tuning is deliberately restricted to the training split. A locked candidate
    # can later receive one validation run; the test split is never available here.
    candidate_axes = [
        getattr(args, f"joint{joint_id}_fractions") or args.joint_fractions
        for joint_id in range(4)
    ]
    ring_candidates = list(itertools.product(*candidate_axes))
    support_axes = [
        args.thumb_actions or (args.support_actions[0],),
        args.index_actions or (args.support_actions[1],),
        args.middle_actions or (args.support_actions[2],),
    ]
    support_candidates = list(itertools.product(*support_axes))
    brake_candidates = list(
        itertools.product(
            args.brake_progress_margins or (args.brake_progress_margin,),
            args.brake_distances_mm or (args.brake_distance_mm,),
        )
    )
    candidates = list(itertools.product(ring_candidates, support_candidates, brake_candidates))
    num_candidates = len(candidates)
    num_envs = num_candidates * args.replicas
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()

    candidate_ids = torch.arange(num_candidates, device=direct.device).repeat_interleave(
        args.replicas
    )
    fractions = torch.tensor(
        [candidate[0] for candidate in candidates], dtype=torch.float32, device=direct.device
    )[candidate_ids]
    support_actions = torch.tensor(
        [candidate[1] for candidate in candidates], dtype=torch.float32, device=direct.device
    )[candidate_ids]
    brake_parameters = torch.tensor(
        [candidate[2] for candidate in candidates], dtype=torch.float32, device=direct.device
    )[candidate_ids]

    inf = torch.full((num_candidates,), float("inf"), device=direct.device)
    min_target_distance = inf.clone()
    min_target_perpendicular_distance = inf.clone()
    best_progress_error = inf.clone()
    best_geometry_step = torch.zeros(num_candidates, dtype=torch.long, device=direct.device)
    geometry_hits = torch.zeros(num_candidates, dtype=torch.long, device=direct.device)
    eligible_hits = torch.zeros_like(geometry_hits)
    triad_hits = torch.zeros_like(geometry_hits)
    completed = torch.zeros_like(geometry_hits)
    successes = torch.zeros_like(geometry_hits)
    drops = torch.zeros_like(geometry_hits)
    invalid = torch.zeros_like(geometry_hits)
    max_penetration = torch.zeros(num_candidates, device=direct.device)
    max_position_drift = torch.zeros(num_candidates, device=direct.device)
    max_axis_drift = torch.zeros(num_candidates, device=direct.device)
    brake_latched = torch.zeros(num_envs, dtype=torch.bool, device=direct.device)
    brake_targets = torch.zeros((num_envs, 4), device=direct.device)

    target_progress = float(cfg.stage_target_progress[int(cfg.curriculum_stage)])
    progress_tolerance = float(cfg.stage_progress_tolerance[int(cfg.curriculum_stage)])
    target_radius = float(cfg.stage_target_distance_m[int(cfg.curriculum_stage)])
    increment = 0.25 * cfg.action_scale

    for step in range(args.steps):
        actions = torch.zeros((num_envs, cfg.action_space), device=direct.device)
        ring_source = direct.source_targets[:, 8:12]
        ring_final = direct.p2_target_ring_qpos.unsqueeze(0).expand_as(ring_source)
        ring_target = ring_source + fractions * (ring_final - ring_source)
        if args.geometric_brake:
            _, current_progress, current_distance, _ = direct._ring_geometry()
            should_brake = (
                (current_progress >= target_progress - brake_parameters[:, 0])
                & (current_distance <= brake_parameters[:, 1] / 1000.0)
            )
            newly_braked = should_brake & ~brake_latched
            brake_targets[newly_braked] = direct.hand_qpos[newly_braked, 8:12]
            brake_latched |= should_brake
            ring_target = torch.where(brake_latched.unsqueeze(-1), brake_targets, ring_target)
        actions[:, 8:12] = torch.clamp(
            (ring_target - direct.policy_targets[:, 8:12]) / increment, -1.0, 1.0
        )
        for column, joint_slice in enumerate(
            (slice(1, 4), slice(5, 8), slice(17, 20)),
        ):
            actions[:, joint_slice] = support_actions[:, column].unsqueeze(-1)

        _, rewards, terminated, truncated, _ = env.step(actions)
        direct._compute_intermediate_values()
        _, progress, target_distance, perpendicular_distance = direct._ring_geometry()
        eligible, metrics = direct._instantaneous_gate()
        progress_error = torch.abs(progress - target_progress)
        geometry = (progress_error <= progress_tolerance) & (target_distance <= target_radius)
        triad = metrics["triad"] > 0.5

        # Reduce replicas and successive reset episodes into candidate-level evidence.
        min_target_distance.scatter_reduce_(
            0, candidate_ids, target_distance, reduce="amin", include_self=True
        )
        near_target_progress = progress_error <= progress_tolerance
        target_perpendicular = torch.where(
            near_target_progress, perpendicular_distance, torch.full_like(perpendicular_distance, float("inf"))
        )
        min_target_perpendicular_distance.scatter_reduce_(
            0, candidate_ids, target_perpendicular, reduce="amin", include_self=True
        )
        best_progress_error.scatter_reduce_(
            0, candidate_ids, progress_error, reduce="amin", include_self=True
        )
        geometry_hits.scatter_add_(0, candidate_ids, geometry.long())
        eligible_hits.scatter_add_(0, candidate_ids, eligible.long())
        triad_hits.scatter_add_(0, candidate_ids, triad.long())
        max_penetration.scatter_reduce_(
            0,
            candidate_ids,
            direct.episode_max_penetration,
            reduce="amax",
            include_self=True,
        )
        max_position_drift.scatter_reduce_(
            0,
            candidate_ids,
            metrics["position_drift"],
            reduce="amax",
            include_self=True,
        )
        max_axis_drift.scatter_reduce_(
            0, candidate_ids, metrics["axis_drift"], reduce="amax", include_self=True
        )
        improved = target_distance <= min_target_distance[candidate_ids] + 1.0e-9
        improved_ids = candidate_ids[improved]
        if improved_ids.numel():
            best_geometry_step.scatter_reduce_(
                0,
                improved_ids,
                torch.full_like(improved_ids, step + 1),
                reduce="amax",
                include_self=True,
            )

        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        if done_ids.numel():
            ids = candidate_ids[done_ids]
            completed.scatter_add_(0, ids, torch.ones_like(ids))
            successes.scatter_add_(0, ids, direct.last_completed_p1_p2_success[done_ids].long())
            drops.scatter_add_(0, ids, direct.last_completed_p1_p2_drop[done_ids].long())
            bad = (~torch.isfinite(rewards[done_ids])) | (
                ~torch.isfinite(direct.pen_pos[done_ids]).all(dim=-1)
            )
            invalid.scatter_add_(0, ids, bad.long())
            brake_latched[done_ids] = False

    rows = []
    samples_per_candidate = args.replicas * args.steps
    for candidate_id, (joint_fractions, candidate_support, candidate_brake) in enumerate(candidates):
        count = int(completed[candidate_id].item())
        success = int(successes[candidate_id].item())
        row = {
            "candidate_id": candidate_id,
            "ring_joint_fractions": list(joint_fractions),
            "support_actions_thumb_index_middle": list(candidate_support),
            "brake_progress_margin": candidate_brake[0],
            "brake_distance_mm": candidate_brake[1],
            "completed_episodes": count,
            "successes": success,
            "success_rate": success / max(count, 1),
            "drops": int(drops[candidate_id].item()),
            "invalid": int(invalid[candidate_id].item()),
            "geometry_hit_ratio": float(geometry_hits[candidate_id].item() / samples_per_candidate),
            "eligible_hit_ratio": float(eligible_hits[candidate_id].item() / samples_per_candidate),
            "triad_hit_ratio": float(triad_hits[candidate_id].item() / samples_per_candidate),
            "min_target_distance_mm": float(min_target_distance[candidate_id].item() * 1000.0),
            "min_perpendicular_distance_near_target_progress_mm": float(
                min_target_perpendicular_distance[candidate_id].item() * 1000.0
            ),
            "min_progress_error": float(best_progress_error[candidate_id].item()),
            "best_geometry_step": int(best_geometry_step[candidate_id].item()),
            "max_penetration_mm": float(max_penetration[candidate_id].item() * 1000.0),
            "max_position_drift_mm": float(max_position_drift[candidate_id].item() * 1000.0),
            "max_axis_drift_deg": float(max_axis_drift[candidate_id].item() * 180.0 / torch.pi),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["success_rate"],
            row["eligible_hit_ratio"],
            row["geometry_hit_ratio"],
            -row["min_target_distance_mm"],
            row["triad_hit_ratio"],
            -row["drops"],
        ),
        reverse=True,
    )
    result = {
        "task": args.task,
        "split": "train",
        "test_split_accessed": False,
        "joint_fraction_values": [list(values) for values in candidate_axes],
        "support_action_values": [list(values) for values in support_axes],
        "brake_parameter_values": {
            "progress_margins": list(args.brake_progress_margins or (args.brake_progress_margin,)),
            "distances_mm": list(args.brake_distances_mm or (args.brake_distance_mm,)),
        },
        "replicas": args.replicas,
        "steps": args.steps,
        "num_candidates": num_candidates,
        "num_envs": num_envs,
        "target_progress": target_progress,
        "progress_tolerance": progress_tolerance,
        "target_radius_mm": target_radius * 1000.0,
        "geometric_brake": args.geometric_brake,
        "brake_progress_margin": args.brake_progress_margin,
        "brake_distance_mm": args.brake_distance_mm,
        "top": rows[: args.top_k],
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_RING_JOINT_SEARCH=" + json.dumps(result, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
