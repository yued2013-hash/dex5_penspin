from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Search a physical index/ring brace after the no-reset P4 prefix."
)
parser.add_argument(
    "--task",
    default="Dex5-V9P4-P5-P1-ContinuousClosureThumbWrapTrain-Direct-v0",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--seed", type=int, default=2570)
parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--residual_scale", type=float, default=0.50)
parser.add_argument("--index_scale", type=float, default=0.18)
parser.add_argument("--first_duration_max", type=int, default=32)
parser.add_argument("--second_duration_max", type=int, default=40)
parser.add_argument("--contact_hold_steps", type=int, default=3)
parser.add_argument("--max_selected", type=int, default=512)
parser.add_argument("--parent", type=Path)
parser.add_argument("--parent_jitter", type=float, default=0.12)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def make_candidates(
    count: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    first = torch.zeros((count, 20), device=device)
    second = torch.zeros_like(first)
    first[:, 8:12] = args.action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    second[:, 8:12] = args.action_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    first[:, 0:4] = args.index_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    second[:, 0:4] = args.index_scale * torch.empty(
        (count, 4), device=device
    ).uniform_(-1.0, 1.0, generator=generator)
    first[0].zero_()
    second[0].zero_()
    for row in range(min(16, count - 1)):
        joint = 8 + (row // 4)
        magnitude = (0.25, 0.50, 0.75, 1.0)[row % 4]
        first[row + 1].zero_()
        second[row + 1].zero_()
        first[row + 1, joint] = (
            -args.action_scale * magnitude
            if row % 2 == 0
            else args.action_scale * magnitude
        )
    first_duration = torch.randint(
        8,
        args.first_duration_max + 1,
        (count,),
        generator=generator,
        device=device,
    )
    second_duration = torch.randint(
        8,
        args.second_duration_max + 1,
        (count,),
        generator=generator,
        device=device,
    )
    return first, second, first_duration, second_duration


def main() -> None:
    cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs
    )
    cfg.seed = args.seed
    cfg.trainee_policy_scale = args.residual_scale
    cfg.episode_length_s = max(
        cfg.episode_length_s,
        (args.steps + 2) * cfg.decimation * cfg.sim.dt,
    )
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    observation, _ = env.reset()
    del observation

    first, second, first_duration, second_duration = make_candidates(
        args.num_envs, direct.device
    )
    if args.parent is not None:
        with np.load(resolve(args.parent)) as parent:
            parent_first = torch.as_tensor(
                parent["candidate_action_first"],
                dtype=torch.float32,
                device=direct.device,
            )
            parent_second = torch.as_tensor(
                parent["candidate_action_second"],
                dtype=torch.float32,
                device=direct.device,
            )
            parent_first_duration = torch.as_tensor(
                parent["first_duration"],
                dtype=torch.long,
                device=direct.device,
            )
            parent_second_duration = torch.as_tensor(
                parent["second_duration"],
                dtype=torch.long,
                device=direct.device,
            )
        parent_ids = torch.arange(
            args.num_envs, device=direct.device
        ) % len(parent_first)
        generator = torch.Generator(device=direct.device).manual_seed(
            args.seed + 911
        )
        jitter = torch.zeros_like(first)
        jitter[:, 8:12] = args.parent_jitter * torch.randn(
            (args.num_envs, 4),
            generator=generator,
            device=direct.device,
        )
        jitter[:, 0:4] = (
            args.index_scale
            * args.parent_jitter
            * torch.randn(
                (args.num_envs, 4),
                generator=generator,
                device=direct.device,
            )
        )
        first = (parent_first[parent_ids] + jitter).clamp(-1.0, 1.0)
        second = (
            parent_second[parent_ids] - 0.7 * jitter
        ).clamp(-1.0, 1.0)
        duration_jitter = torch.randint(
            -12,
            13,
            (args.num_envs,),
            generator=generator,
            device=direct.device,
        )
        first_duration = (
            parent_first_duration[parent_ids] + duration_jitter
        ).clamp(8, args.first_duration_max + 24)
        second_duration = (
            parent_second_duration[parent_ids] - duration_jitter
        ).clamp(8, args.second_duration_max + 24)
    search_age = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    started = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    failed = torch.zeros_like(started)
    contact_run = torch.zeros_like(search_age)
    max_contact_run = torch.zeros_like(search_age)
    min_ring_gap = torch.full(
        (args.num_envs,), torch.inf, device=direct.device
    )
    max_penetration = torch.zeros_like(min_ring_gap)
    max_force = torch.zeros_like(min_ring_gap)
    best_score = torch.full_like(min_ring_gap, -torch.inf)
    best_qpos = torch.zeros((args.num_envs, 27), device=direct.device)
    best_qvel = torch.zeros((args.num_envs, 26), device=direct.device)
    best_policy_targets = torch.zeros(
        (args.num_envs, 20), device=direct.device
    )
    best_target_hand = torch.zeros_like(best_policy_targets)
    best_closure_origin = torch.zeros_like(best_policy_targets)
    best_closure_pen_pos = torch.zeros(
        (args.num_envs, 3), device=direct.device
    )
    best_closure_pen_quat = torch.zeros(
        (args.num_envs, 4), device=direct.device
    )
    best_init_pen_pos = torch.zeros_like(best_closure_pen_pos)
    best_init_pen_axis = torch.zeros_like(best_closure_pen_pos)
    best_contact_force = torch.zeros(
        (args.num_envs, 5), device=direct.device
    )
    best_penetration = torch.full_like(min_ring_gap, torch.inf)
    best_action = torch.zeros_like(best_policy_targets)
    start_qpos = torch.zeros_like(best_qpos)
    start_qvel = torch.zeros_like(best_qvel)
    start_bank_id = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=direct.device
    )
    best_phase = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    success = torch.zeros_like(started)

    for _ in range(args.steps):
        phase_before = direct.chain_phase_ids.clone()
        new_start = (phase_before >= 4) & ~started & ~failed
        if torch.any(new_start):
            start_qpos[new_start] = torch.cat(
                (direct.hand_qpos, direct.pen_pos, direct.pen_quat),
                dim=-1,
            )[new_start]
            start_qvel[new_start] = torch.cat(
                (
                    direct.hand_qvel,
                    direct.pen_linvel,
                    direct.pen_angvel,
                ),
                dim=-1,
            )[new_start]
            start_bank_id[new_start] = direct.current_bank_ids[
                new_start
            ]
        started |= new_start
        search_age = torch.where(
            new_start, torch.zeros_like(search_age), search_age
        )
        active = started & ~failed & ~success
        first_active = active & (search_age < first_duration)
        second_active = (
            active
            & ~first_active
            & (search_age < first_duration + second_duration)
        )
        actions = torch.zeros(
            (args.num_envs, 20), device=direct.device
        )
        actions = torch.where(
            first_active.unsqueeze(-1), first, actions
        )
        actions = torch.where(
            second_active.unsqueeze(-1), second, actions
        )
        _, _, terminated, truncated, _ = env.step(actions)
        direct._compute_intermediate_values()

        contacts = (
            direct.contact_force_norm > direct.cfg.p1_contact_force_n
        )
        index_ring = contacts[:, 1] & contacts[:, 3]
        ring_gap = (
            direct.tip_surface_dist[:, 3]
            - direct.fingertip_radii[:, 3]
        )
        drift = torch.linalg.norm(
            direct.pen_pos - direct.init_pen_pos, dim=-1
        )
        target_position_error = torch.linalg.norm(
            direct.pen_pos - direct.target_pen_pos, dim=-1
        )
        target_axis = quat_apply(
            direct.target_pen_quat, direct.pen_x_axis
        )
        target_axis_dot = torch.sum(
            direct.pen_axis * target_axis, dim=-1
        ).clamp(-1.0, 1.0)
        target_axis_error = torch.acos(target_axis_dot.abs())
        angular_speed = torch.linalg.norm(
            direct.pen_angvel, dim=-1
        )
        index_ring_error = torch.cat(
            (
                direct.hand_qpos[:, 0:4]
                - direct.target_hand_qpos[:, 0:4],
                direct.hand_qpos[:, 8:12]
                - direct.target_hand_qpos[:, 8:12],
            ),
            dim=-1,
        )
        index_ring_rmse = torch.sqrt(
            torch.mean(index_ring_error.square(), dim=-1)
        )
        force = direct.contact_force_norm.max(dim=-1).values
        penetration = direct.episode_max_penetration
        safe = (
            (penetration <= direct.cfg.penetration_threshold)
            & (force <= direct.cfg.p1_max_force_n)
            & (
                target_position_error
                <= direct.cfg.thumb_wrap_brace_pen_position_tolerance_m
            )
            & (
                target_axis_error
                <= torch.deg2rad(
                    torch.full_like(
                        target_axis_error,
                        direct.cfg
                        .thumb_wrap_acquisition_pen_axis_tolerance_deg,
                    )
                )
            )
            & (
                angular_speed
                <= direct.cfg.p5_max_angular_speed_rad_s
            )
        )
        valid = active & safe & contacts[:, 1]
        min_ring_gap = torch.where(
            valid, torch.minimum(min_ring_gap, ring_gap), min_ring_gap
        )
        max_penetration = torch.maximum(max_penetration, penetration)
        max_force = torch.maximum(max_force, force)
        contact_run = torch.where(
            valid
            & index_ring
            & (
                index_ring_rmse
                <= direct.cfg.closure_stage_rmse_thresholds[0]
            ),
            contact_run + 1,
            torch.zeros_like(contact_run),
        )
        max_contact_run = torch.maximum(max_contact_run, contact_run)

        score = (
            -1000.0 * torch.relu(ring_gap)
            + 8.0 * index_ring.float()
            + 2.0 * contacts[:, 1].float()
            + 4.0 * torch.exp(-4.0 * index_ring_rmse)
            - 120.0 * drift
            - 40.0 * target_axis_error
            - 80.0 * target_position_error
            - 0.25 * angular_speed
            - 20.0 * penetration
        )
        eligible = contact_run > 0
        improved = eligible & (score > best_score)
        if torch.any(improved):
            best_score[improved] = score[improved]
            best_qpos[improved] = torch.cat(
                (direct.hand_qpos, direct.pen_pos, direct.pen_quat),
                dim=-1,
            )[improved]
            best_qvel[improved] = torch.cat(
                (
                    direct.hand_qvel,
                    direct.pen_linvel,
                    direct.pen_angvel,
                ),
                dim=-1,
            )[improved]
            best_policy_targets[improved] = direct.policy_targets[
                improved
            ]
            best_target_hand[improved] = direct.target_hand_qpos[improved]
            best_closure_origin[improved] = direct.closure_origin_hand[
                improved
            ]
            best_closure_pen_pos[improved] = (
                direct.closure_origin_pen_pos[improved]
            )
            best_closure_pen_quat[improved] = (
                direct.closure_origin_pen_quat[improved]
            )
            best_init_pen_pos[improved] = direct.init_pen_pos[improved]
            best_init_pen_axis[improved] = direct.init_pen_axis[improved]
            best_contact_force[improved] = direct.contact_force_norm[
                improved
            ]
            best_penetration[improved] = penetration[improved]
            best_action[improved] = actions[improved]
            best_phase[improved] = direct.chain_phase_ids[improved]

        success |= (
            (contact_run >= args.contact_hold_steps)
            & (ring_gap <= 0.001)
        )
        done = (terminated | truncated).reshape(-1)
        failed |= done & started & ~success
        search_age = torch.where(
            active, search_age + 1, search_age
        )

    selected = (
        success
        & torch.isfinite(best_score)
        & (best_penetration <= direct.cfg.penetration_threshold)
    )
    selected_ids = torch.where(selected)[0]
    if selected_ids.numel() > args.max_selected:
        order = torch.argsort(best_score[selected_ids], descending=True)
        selected_ids = selected_ids[order[: args.max_selected]]
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ids = selected_ids
    qpos = best_qpos[ids].cpu().numpy().astype(np.float32)
    qvel = best_qvel[ids].cpu().numpy().astype(np.float32)
    np.savez_compressed(
        output,
        source_qpos=qpos,
        source_qvel=qvel,
        action_start_qpos=start_qpos[ids].cpu().numpy().astype(
            np.float32
        ),
        action_start_qvel=start_qvel[ids].cpu().numpy().astype(
            np.float32
        ),
        action_start_bank_id=start_bank_id[ids]
        .cpu().numpy().astype(np.int64),
        source_target_qpos=qpos[:, :20],
        source_policy_action=best_action[ids].cpu().numpy().astype(
            np.float32
        ),
        source_target_hand_qpos=best_target_hand[
            ids
        ].cpu().numpy().astype(np.float32),
        source_closure_origin_hand=best_closure_origin[
            ids
        ].cpu().numpy().astype(np.float32),
        source_closure_brace_hand=qpos[:, :20],
        source_closure_origin_pen_pos=best_closure_pen_pos[
            ids
        ].cpu().numpy().astype(np.float32),
        source_closure_origin_pen_quat=best_closure_pen_quat[
            ids
        ].cpu().numpy().astype(np.float32),
        source_init_pen_pos=best_init_pen_pos[
            ids
        ].cpu().numpy().astype(np.float32),
        source_init_pen_axis=best_init_pen_axis[
            ids
        ].cpu().numpy().astype(np.float32),
        source_max_penetration_m=best_penetration[
            ids
        ].cpu().numpy().astype(np.float32),
        source_contact_force_n=best_contact_force[
            ids
        ].cpu().numpy().astype(np.float32),
        source_chain_phase_id=best_phase[
            ids
        ].cpu().numpy().astype(np.int64),
        target_qpos=qpos,
        target_qvel=np.zeros_like(qvel),
        phase_ids=np.zeros(len(ids), dtype=np.int64),
        candidate_action_first=first[ids].cpu().numpy().astype(np.float32),
        candidate_action_second=second[ids].cpu().numpy().astype(
            np.float32
        ),
        first_duration=first_duration[ids].cpu().numpy().astype(np.int64),
        second_duration=second_duration[ids].cpu().numpy().astype(np.int64),
        min_ring_gap_m=min_ring_gap[ids].cpu().numpy().astype(np.float32),
        max_contact_run=max_contact_run[ids].cpu().numpy().astype(np.int64),
        seed=np.asarray(args.seed, dtype=np.int64),
        source=np.asarray(
            "same-episode P4 index/ring brace action search", dtype=np.str_
        ),
    )
    finite_gap = min_ring_gap[torch.isfinite(min_ring_gap)]
    summary = {
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "phase4_started": int(started.sum().item()),
        "ring_contact_successes": int(success.sum().item()),
        "selected": int(len(ids)),
        "success_rate_from_phase4": (
            float(success.sum().item() / max(started.sum().item(), 1))
        ),
        "min_ring_gap_mm": (
            float(finite_gap.min().item() * 1000.0)
            if finite_gap.numel()
            else None
        ),
        "max_contact_hold_steps": int(max_contact_run.max().item()),
        "selected_max_penetration_mm": (
            float(best_penetration[ids].max().item() * 1000.0)
            if ids.numel()
            else None
        ),
        "output": str(output),
    }
    summary_path = resolve(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
