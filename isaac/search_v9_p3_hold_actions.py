from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    PROJECT_ROOT / "reports/m3_2_30_index_near_surface/v9_p3_physx_reset_bank.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports/m3_2_30_index_near_surface/v9_p3_hold_action_search.json"
)


parser = argparse.ArgumentParser(
    description="Search a low-force T/I/M residual action that holds v9 P3 for 100 ms."
)
parser.add_argument("--task", default="Dex5-P3Hold-Direct-v0")
parser.add_argument("--bundle_path", type=Path, default=DEFAULT_BUNDLE)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--accepted_bundle", type=Path, default=None)
parser.add_argument("--candidates", type=int, default=4096)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument(
    "--required_hold_steps",
    type=int,
    default=6,
    help="Required consecutive T/I/M contact steps within the evaluation window.",
)
parser.add_argument("--seed", type=int, default=20260716)
parser.add_argument("--max_penetration_mm", type=float, default=0.5)
parser.add_argument("--max_drift_mm", type=float, default=15.0)
parser.add_argument("--max_force_n", type=float, default=4.0)
parser.add_argument("--max_planar_drift_deg", type=float, default=8.0)
parser.add_argument("--max_axis_drift_deg", type=float, default=180.0)
parser.add_argument("--max_angular_speed_rad_s", type=float, default=12.0)
parser.add_argument("--reset_joint_pos_noise", type=float, default=0.0)
parser.add_argument("--reset_joint_vel_noise", type=float, default=0.0)
parser.add_argument("--reset_pen_pos_noise", type=float, default=0.0)
parser.add_argument("--reset_pen_vel_noise", type=float, default=0.0)
parser.add_argument("--teacher_prefill_steps", type=int, default=0)
parser.add_argument("--collision_proxy_scales", type=float, nargs=5, default=(0.99, 1.07, 1.07, 0.84, 0.80))
parser.add_argument(
    "--local_center",
    type=float,
    nargs=12,
    default=None,
    help="Optional index[0:4], middle[4:8], thumb[8:12] action center.",
)
parser.add_argument("--local_radius", type=float, default=0.25)
parser.add_argument(
    "--switch_step",
    type=int,
    default=-1,
    help="Use a second local action after this step; negative keeps one constant action.",
)
parser.add_argument(
    "--second_local_radius",
    type=float,
    default=0.35,
    help="Radius of the second action around local_center when switch_step is enabled.",
)
parser.add_argument(
    "--second_local_center",
    type=float,
    nargs=12,
    default=None,
    help="Optional independent center for the second T/I/M action.",
)
parser.add_argument("--fixed_first_action", type=float, nargs=12, default=None)
parser.add_argument("--fixed_second_action", type=float, nargs=12, default=None)
parser.add_argument("--third_switch_step", type=int, default=-1)
parser.add_argument("--third_local_center", type=float, nargs=12, default=None)
parser.add_argument("--third_local_radius", type=float, default=0.25)
parser.add_argument("--fixed_third_action", type=float, nargs=12, default=None)
parser.add_argument(
    "--replicas_per_action",
    type=int,
    default=1,
    help="Repeat every sampled action sequence across multiple PhysX environments.",
)
parser.add_argument("--snapshot_bundle", type=Path, default=None)
parser.add_argument("--snapshot_step", type=int, default=-1)
parser.add_argument(
    "--snapshot_only_accepted",
    action="store_true",
    help="Only export running states whose complete rollout passes every acceptance gate.",
)
parser.add_argument(
    "--use_bundle_source_target",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override whether reset restores the actuator target stored in the bundle.",
)
parser.add_argument("--solver_velocity_iterations", type=int, default=None)
parser.add_argument("--max_depenetration_velocity", type=float, default=None)
parser.add_argument("--pen_contact_offset", type=float, default=None)
parser.add_argument("--env_spacing", type=float, default=None)
parser.add_argument(
    "--enable_external_forces_every_iteration", action="store_true"
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


def resolved(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.candidates)
    env_cfg.seed = args.seed
    env_cfg.bundle_path = str(resolved(args.bundle_path))
    env_cfg.reset_joint_pos_noise = args.reset_joint_pos_noise
    env_cfg.reset_joint_vel_noise = args.reset_joint_vel_noise
    env_cfg.reset_pen_pos_noise = args.reset_pen_pos_noise
    env_cfg.reset_pen_vel_noise = args.reset_pen_vel_noise
    if hasattr(env_cfg, "teacher_prefill_steps"):
        env_cfg.teacher_prefill_steps = args.teacher_prefill_steps
    if args.use_bundle_source_target is not None:
        env_cfg.use_bundle_source_target = args.use_bundle_source_target
    if args.solver_velocity_iterations is not None:
        env_cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = (
            args.solver_velocity_iterations
        )
        env_cfg.pen_cfg.spawn.rigid_props.solver_velocity_iteration_count = (
            args.solver_velocity_iterations
        )
    if args.max_depenetration_velocity is not None:
        env_cfg.robot_cfg.spawn.rigid_props.max_depenetration_velocity = (
            args.max_depenetration_velocity
        )
        env_cfg.pen_cfg.spawn.rigid_props.max_depenetration_velocity = (
            args.max_depenetration_velocity
        )
    if args.pen_contact_offset is not None:
        env_cfg.pen_cfg.spawn.collision_props.contact_offset = args.pen_contact_offset
    if args.env_spacing is not None:
        env_cfg.scene.env_spacing = args.env_spacing
    if args.enable_external_forces_every_iteration:
        env_cfg.sim.physx.enable_external_forces_every_iteration = True
    env_cfg.collision_proxy_radius_scales = tuple(args.collision_proxy_scales)
    env_cfg.penetration_threshold = 0.020
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        (args.steps + 2) * env_cfg.decimation * env_cfg.sim.dt,
    )
    env = gym.make(args.task, cfg=env_cfg)
    direct_env = env.unwrapped
    env.reset()

    generator = torch.Generator(device=direct_env.device).manual_seed(args.seed)
    tim_actions = 2.0 * torch.rand(
        (args.candidates, 12), generator=generator, device=direct_env.device
    ) - 1.0
    tim_actions[0] = 0.0

    # Include interpretable whole-finger probes before the random population.
    probe = 1
    for finger in range(3):
        for value in (-1.0, -0.5, 0.5, 1.0):
            if probe >= args.candidates:
                break
            tim_actions[probe] = 0.0
            tim_actions[probe, 4 * finger : 4 * (finger + 1)] = value
            probe += 1
    if args.local_center is not None:
        center = torch.tensor(args.local_center, device=direct_env.device)
        local_count = args.candidates // 2
        local_noise = 2.0 * torch.rand(
            (local_count, 12), generator=generator, device=direct_env.device
        ) - 1.0
        tim_actions[-local_count:] = torch.clamp(
            center + args.local_radius * local_noise, -1.0, 1.0
        )
        tim_actions[-1] = center
    if args.fixed_first_action is not None:
        tim_actions[:] = torch.tensor(args.fixed_first_action, device=direct_env.device)

    actions = torch.zeros((args.candidates, 20), device=direct_env.device)
    actions[:, 0:8] = tim_actions[:, 0:8]
    actions[:, 16:20] = tim_actions[:, 8:12]
    second_actions = actions.clone()
    if args.switch_step >= 0:
        if args.second_local_center is not None:
            second_center = torch.tensor(args.second_local_center, device=direct_env.device)
        else:
            second_center = tim_actions
        second_tim_actions = torch.clamp(
            second_center
            + args.second_local_radius
            * (2.0 * torch.rand(
                tim_actions.shape, generator=generator, device=direct_env.device
            ) - 1.0),
            -1.0,
            1.0,
        )
        second_actions[:, 0:8] = second_tim_actions[:, 0:8]
        second_actions[:, 16:20] = second_tim_actions[:, 8:12]
        second_actions[0] = actions[0]
    if args.fixed_second_action is not None:
        fixed_second = torch.tensor(args.fixed_second_action, device=direct_env.device)
        second_actions[:, 0:8] = fixed_second[:8]
        second_actions[:, 16:20] = fixed_second[8:12]
    third_actions = second_actions.clone()
    if args.third_switch_step >= 0:
        if args.third_local_center is not None:
            third_center = torch.tensor(args.third_local_center, device=direct_env.device)
        else:
            third_center = second_tim_actions
        third_tim_actions = torch.clamp(
            third_center
            + args.third_local_radius
            * (2.0 * torch.rand(
                tim_actions.shape, generator=generator, device=direct_env.device
            ) - 1.0),
            -1.0,
            1.0,
        )
        third_actions[:, 0:8] = third_tim_actions[:, 0:8]
        third_actions[:, 16:20] = third_tim_actions[:, 8:12]
    if args.fixed_third_action is not None:
        fixed_third = torch.tensor(args.fixed_third_action, device=direct_env.device)
        third_actions[:, 0:8] = fixed_third[:8]
        third_actions[:, 16:20] = fixed_third[8:12]

    if args.replicas_per_action < 1 or args.candidates % args.replicas_per_action:
        raise ValueError("candidates must be divisible by replicas_per_action")
    if args.replicas_per_action > 1:
        group_count = args.candidates // args.replicas_per_action
        for action_tensor in (actions, second_actions, third_actions):
            grouped = action_tensor.view(group_count, args.replicas_per_action, 20)
            grouped[:] = grouped[:, :1]

    contact_steps = torch.zeros((args.candidates, 5), device=direct_env.device)
    contact_trace = torch.zeros(
        (args.candidates, args.steps, 5), dtype=torch.bool, device=direct_env.device
    )
    axis_drift_trace = torch.zeros(
        (args.candidates, args.steps), device=direct_env.device
    )
    angular_speed_trace = torch.zeros_like(axis_drift_trace)
    actuator_target_trace = torch.zeros(
        (args.candidates, args.steps, 20), device=direct_env.device
    )
    current_triad_run = torch.zeros(args.candidates, dtype=torch.long, device=direct_env.device)
    max_triad_run = torch.zeros_like(current_triad_run)
    current_strict_run = torch.zeros_like(current_triad_run)
    max_strict_run = torch.zeros_like(current_triad_run)
    max_penetration = torch.zeros(args.candidates, device=direct_env.device)
    max_drift = torch.zeros(args.candidates, device=direct_env.device)
    max_force = torch.zeros(args.candidates, device=direct_env.device)
    force_sum = torch.zeros((args.candidates, 5), device=direct_env.device)
    terminated_ever = torch.zeros(args.candidates, dtype=torch.bool, device=direct_env.device)
    max_planar_drift = torch.zeros(args.candidates, device=direct_env.device)
    max_axis_drift = torch.zeros(args.candidates, device=direct_env.device)
    max_angular_speed = torch.zeros(args.candidates, device=direct_env.device)
    initial_yaw = torch.atan2(direct_env.init_pen_axis[:, 1], direct_env.init_pen_axis[:, 0])
    snapshot_qpos = None
    snapshot_qvel = None
    snapshot_actions = None
    snapshot_targets = None

    for step in range(args.steps):
        if args.third_switch_step >= 0 and step >= args.third_switch_step:
            step_actions = third_actions
        elif args.switch_step >= 0 and step >= args.switch_step:
            step_actions = second_actions
        else:
            step_actions = actions
        _, _, terminated, _, _ = env.step(step_actions)
        actuator_target_trace[:, step] = direct_env.processed_targets
        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > 0.03
        contact_trace[:, step] = contacts
        triad = contacts[:, :3].all(dim=-1)
        contact_steps += contacts.float()
        current_triad_run = torch.where(
            triad, current_triad_run + 1, torch.zeros_like(current_triad_run)
        )
        max_triad_run = torch.maximum(max_triad_run, current_triad_run)
        max_penetration = torch.maximum(max_penetration, direct_env.episode_max_penetration)
        max_drift = torch.maximum(
            max_drift,
            torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1),
        )
        max_force = torch.maximum(
            max_force, direct_env.contact_force_norm.max(dim=-1).values
        )
        current_yaw = torch.atan2(direct_env.pen_axis[:, 1], direct_env.pen_axis[:, 0])
        planar_drift = torch.abs(
            torch.atan2(torch.sin(current_yaw - initial_yaw), torch.cos(current_yaw - initial_yaw))
        )
        max_planar_drift = torch.maximum(max_planar_drift, planar_drift)
        axis_dot = torch.sum(
            direct_env.pen_axis * direct_env.init_pen_axis, dim=-1
        ).clamp(-1.0, 1.0)
        max_axis_drift = torch.maximum(max_axis_drift, torch.acos(axis_dot.abs()))
        axis_drift_trace[:, step] = torch.acos(axis_dot.abs())
        current_axis_drift = axis_drift_trace[:, step]
        current_angular_speed = torch.linalg.norm(direct_env.pen_angvel, dim=-1)
        angular_speed_trace[:, step] = current_angular_speed
        max_angular_speed = torch.maximum(
            max_angular_speed, current_angular_speed
        )
        strict_instantaneous = (
            triad
            & ~contacts[:, 3:].any(dim=-1)
            & (current_axis_drift < torch.deg2rad(torch.tensor(args.max_axis_drift_deg, device=direct_env.device)))
            & (current_angular_speed < args.max_angular_speed_rad_s)
            & (torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1) < args.max_drift_mm / 1000.0)
            & (direct_env.contact_force_norm.max(dim=-1).values < args.max_force_n)
            & (direct_env.episode_max_penetration < args.max_penetration_mm / 1000.0)
        )
        current_strict_run = torch.where(
            strict_instantaneous,
            current_strict_run + 1,
            torch.zeros_like(current_strict_run),
        )
        max_strict_run = torch.maximum(max_strict_run, current_strict_run)
        force_sum += direct_env.contact_force_norm
        terminated_ever |= terminated
        if args.snapshot_bundle is not None and step == args.snapshot_step:
            snapshot_qpos = torch.cat(
                (direct_env.hand_qpos, direct_env.pen_pos, direct_env.pen_quat), dim=-1
            ).clone()
            snapshot_qvel = torch.cat(
                (direct_env.hand_qvel, direct_env.pen_linvel, direct_env.pen_angvel), dim=-1
            ).clone()
            snapshot_actions = step_actions.clone()
            # Preserve the actuator target that produced this running state.
            # Reconstructing it as qpos + action_scale * action closes the
            # fingers a second time after reset and creates a large contact impulse.
            snapshot_targets = direct_env.processed_targets.clone()

    contact_ratio = contact_steps / args.steps
    ring_pinky_force_share = force_sum[:, 3:].sum(dim=-1) / force_sum.sum(dim=-1).clamp_min(1.0e-6)
    action_energy = torch.mean(actions * actions, dim=-1)
    accepted = (
        (max_strict_run >= args.required_hold_steps)
        & (max_penetration < args.max_penetration_mm / 1000.0)
        & (max_drift < args.max_drift_mm / 1000.0)
        & (max_force < args.max_force_n)
        & (max_planar_drift < torch.deg2rad(torch.tensor(args.max_planar_drift_deg, device=direct_env.device)))
        & (contact_steps[:, 3:].sum(dim=-1) == 0.0)
        & ~terminated_ever
    )
    gate_masks = {
        "triad_hold": max_triad_run >= args.required_hold_steps,
        "strict_hold": max_strict_run >= args.required_hold_steps,
        "penetration": max_penetration < args.max_penetration_mm / 1000.0,
        "position_drift": max_drift < args.max_drift_mm / 1000.0,
        "force": max_force < args.max_force_n,
        "planar_drift": max_planar_drift
        < torch.deg2rad(torch.tensor(args.max_planar_drift_deg, device=direct_env.device)),
        "axis_drift": max_axis_drift
        < torch.deg2rad(torch.tensor(args.max_axis_drift_deg, device=direct_env.device)),
        "angular_speed": max_angular_speed < args.max_angular_speed_rad_s,
        "ring_pinky_clear": contact_steps[:, 3:].sum(dim=-1) == 0.0,
        "not_terminated": ~terminated_ever,
    }
    score = (
        12.0 * torch.clamp(max_triad_run.float() / args.required_hold_steps, max=1.0)
        + 5.0 * contact_ratio[:, :3].mean(dim=-1)
        - 1000.0 * torch.relu(max_penetration - args.max_penetration_mm / 1000.0)
        - 30.0 * max_drift
        - 4.0 * ring_pinky_force_share
        - 0.25 * max_force
        - 0.08 * max_planar_drift * (180.0 / torch.pi)
        - 0.08 * max_axis_drift * (180.0 / torch.pi)
        - 0.03 * max_angular_speed
        - 0.20 * action_energy
        - 8.0 * terminated_ever.float()
    )
    group_count = args.candidates // args.replicas_per_action
    grouped_accepted = accepted.view(group_count, args.replicas_per_action)
    grouped_score = score.view(group_count, args.replicas_per_action)
    group_accept_ratio = grouped_accepted.float().mean(dim=1)
    group_mean_score = grouped_score.mean(dim=1)
    robust_order = torch.argsort(
        group_accept_ratio * 1000.0 + group_mean_score, descending=True
    )
    top_ids = torch.topk(score, k=min(64, args.candidates)).indices

    def row(candidate_id: int) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "reset_bank_id": int(direct_env.current_bank_ids[candidate_id].item()),
            "accepted": bool(accepted[candidate_id].item()),
            "action": actions[candidate_id].cpu().tolist(),
            "second_action": (
                second_actions[candidate_id].cpu().tolist() if args.switch_step >= 0 else None
            ),
            "third_action": (
                third_actions[candidate_id].cpu().tolist()
                if args.third_switch_step >= 0
                else None
            ),
            "contact_trace": [
                {
                    finger: bool(contact_trace[candidate_id, step, finger_id].item())
                    for finger_id, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"))
                }
                for step in range(args.steps)
            ],
            "axis_drift_trace_deg": (
                axis_drift_trace[candidate_id].cpu() * (180.0 / torch.pi)
            ).tolist(),
            "angular_speed_trace_rad_s": angular_speed_trace[candidate_id].cpu().tolist(),
            "triad_consecutive_steps": int(max_triad_run[candidate_id].item()),
            "strict_consecutive_steps": int(max_strict_run[candidate_id].item()),
            "contact_ratio": {
                finger: float(contact_ratio[candidate_id, finger_id].item())
                for finger_id, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"))
            },
            "max_penetration_mm": float(max_penetration[candidate_id].item() * 1000.0),
            "max_drift_mm": float(max_drift[candidate_id].item() * 1000.0),
            "max_force_n": float(max_force[candidate_id].item()),
            "max_planar_drift_deg": float(max_planar_drift[candidate_id].item() * (180.0 / torch.pi)),
            "max_axis_drift_deg": float(max_axis_drift[candidate_id].item() * (180.0 / torch.pi)),
            "max_angular_speed_rad_s": float(max_angular_speed[candidate_id].item()),
            "ring_pinky_force_share": float(ring_pinky_force_share[candidate_id].item()),
            "action_rms": float(torch.sqrt(action_energy[candidate_id]).item()),
            "terminated": bool(terminated_ever[candidate_id].item()),
            "score": float(score[candidate_id].item()),
        }

    accepted_ids = torch.nonzero(accepted, as_tuple=False).squeeze(-1)
    best_id = None
    if accepted_ids.numel() > 0:
        best_id = int(accepted_ids[torch.argmax(score[accepted_ids])].item())
    result = {
        "task": args.task,
        "bundle_path": str(resolved(args.bundle_path).relative_to(PROJECT_ROOT)),
        "candidate_count": args.candidates,
        "steps": args.steps,
        "switch_step": args.switch_step,
        "third_switch_step": args.third_switch_step,
        "elapsed_ms": args.steps * env_cfg.decimation * env_cfg.sim.dt * 1000.0,
        "action_layout": "index[0:4], middle[4:8], ring[8:12]=0, pinky[12:16]=0, thumb[16:20]",
        "collision_proxy_radius_scales": list(args.collision_proxy_scales),
        "reset_noise": {
            "joint_pos": args.reset_joint_pos_noise,
            "joint_vel": args.reset_joint_vel_noise,
            "pen_pos": args.reset_pen_pos_noise,
            "pen_vel": args.reset_pen_vel_noise,
        },
        "acceptance": {
            "triad_consecutive_steps": args.required_hold_steps,
            "max_penetration_mm": args.max_penetration_mm,
            "max_drift_mm": args.max_drift_mm,
            "max_force_n": args.max_force_n,
            "max_planar_drift_deg": args.max_planar_drift_deg,
            "max_axis_drift_deg": args.max_axis_drift_deg,
            "max_angular_speed_rad_s": args.max_angular_speed_rad_s,
            "ring_pinky_contact_steps": 0,
        },
        "accepted_count": int(accepted.sum().item()),
        "accepted_ratio": float(accepted.float().mean().item()),
        "replicas_per_action": args.replicas_per_action,
        "action_group_count": group_count,
        "robust_top": [
            {
                "action_group_id": int(group_id),
                "accepted_count": int(grouped_accepted[group_id].sum().item()),
                "accepted_ratio": float(group_accept_ratio[group_id].item()),
                "mean_score": float(group_mean_score[group_id].item()),
                "representative": row(int(group_id) * args.replicas_per_action),
            }
            for group_id in robust_order[: min(12, group_count)].tolist()
        ],
        "gate_pass_ratio": {
            name: float(mask.float().mean().item()) for name, mask in gate_masks.items()
        },
        "per_reset_bank": [
            {
                "reset_bank_id": int(bank_id),
                "count": int(mask.sum().item()),
                "accepted_count": int((accepted & mask).sum().item()),
                "accepted_ratio": float(accepted[mask].float().mean().item()),
            }
            for bank_id in torch.unique(direct_env.current_bank_ids).tolist()
            for mask in [direct_env.current_bank_ids == bank_id]
        ],
        "best": row(best_id) if best_id is not None else None,
        "top": [row(candidate_id) for candidate_id in top_ids.tolist()],
    }
    output = resolved(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.snapshot_bundle is not None:
        if (
            snapshot_qpos is None
            or snapshot_qvel is None
            or snapshot_actions is None
            or snapshot_targets is None
        ):
            raise ValueError("snapshot_step was not reached")
        snapshot_output = resolved(args.snapshot_bundle)
        snapshot_output.parent.mkdir(parents=True, exist_ok=True)
        snapshot_ids = (
            torch.nonzero(accepted, as_tuple=False).squeeze(-1)
            if args.snapshot_only_accepted
            else torch.arange(args.candidates, device=direct_env.device)
        )
        snapshot_rows = int(snapshot_ids.numel())
        zero_actions = torch.zeros_like(snapshot_actions[snapshot_ids])
        future_start = min(args.snapshot_step + 1, args.steps - 1)
        future_targets = actuator_target_trace[snapshot_ids, future_start:]
        replay_base_target = future_targets[:, 0]
        action_scale = float(env_cfg.action_scale)
        replay_actions = torch.clamp(
            (future_targets - replay_base_target[:, None, :]) / action_scale,
            -1.0,
            1.0,
        )
        if replay_actions.shape[1] < args.steps:
            replay_actions = torch.cat(
                (
                    replay_actions,
                    replay_actions[:, -1:].repeat(
                        1, args.steps - replay_actions.shape[1], 1
                    ),
                ),
                dim=1,
            )
        snapshot_teacher = replay_actions.cpu().numpy().astype(np.float32)
        snapshot_target_qpos = snapshot_qpos[snapshot_ids].clone()
        snapshot_target_qpos[:, :20] = replay_base_target
        np.savez_compressed(
            snapshot_output,
            source_qpos=snapshot_qpos[snapshot_ids].cpu().numpy().astype(np.float32),
            source_qvel=snapshot_qvel[snapshot_ids].cpu().numpy().astype(np.float32),
            source_target_qpos=replay_base_target.cpu().numpy().astype(np.float32),
            source_policy_action=zero_actions.cpu().numpy().astype(np.float32),
            source_teacher_actions=snapshot_teacher,
            target_qpos=snapshot_target_qpos.cpu().numpy().astype(np.float32),
            target_qvel=snapshot_qvel[snapshot_ids].cpu().numpy().astype(np.float32),
            phase_ids=np.full(snapshot_rows, 3, dtype=np.int64),
            source_candidate_ids=snapshot_ids.cpu().numpy().astype(np.int64),
        )
        result["snapshot_bundle"] = str(snapshot_output.relative_to(PROJECT_ROOT))
        result["snapshot_step"] = args.snapshot_step
        result["snapshot_count"] = snapshot_rows
        result["snapshot_only_accepted"] = args.snapshot_only_accepted
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.accepted_bundle is not None and accepted_ids.numel() > 0:
        bank_ids = direct_env.current_bank_ids[accepted_ids].cpu().numpy().astype(np.int64)
        first = actions[accepted_ids].cpu().numpy().astype(np.float32)
        second = second_actions[accepted_ids].cpu().numpy().astype(np.float32)
        third = third_actions[accepted_ids].cpu().numpy().astype(np.float32)
        teacher = np.repeat(first[:, None, :], args.steps, axis=1)
        if args.switch_step >= 0:
            teacher[:, args.switch_step :, :] = second[:, None, :]
        if args.third_switch_step >= 0:
            teacher[:, args.third_switch_step :, :] = third[:, None, :]
        payload: dict[str, np.ndarray] = {}
        source_bundle = resolved(args.bundle_path)
        with np.load(source_bundle) as data:
            source_rows = len(data["source_qpos"])
            for key in data.files:
                value = np.asarray(data[key])
                payload[key] = (
                    value[bank_ids]
                    if value.ndim > 0 and value.shape[0] == source_rows
                    else value
                )
        payload["source_policy_action"] = first
        payload["source_teacher_actions"] = teacher
        payload["search_reset_bank_id"] = bank_ids
        payload["search_candidate_id"] = accepted_ids.cpu().numpy().astype(np.int64)
        payload["search_max_penetration_m"] = max_penetration[accepted_ids].cpu().numpy()
        payload["search_max_drift_m"] = max_drift[accepted_ids].cpu().numpy()
        payload["search_max_planar_drift_rad"] = max_planar_drift[accepted_ids].cpu().numpy()
        payload["search_max_axis_drift_rad"] = max_axis_drift[accepted_ids].cpu().numpy()
        payload["search_max_angular_speed_rad_s"] = max_angular_speed[accepted_ids].cpu().numpy()
        accepted_output = resolved(args.accepted_bundle)
        accepted_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(accepted_output, **payload)
        result["accepted_bundle"] = str(accepted_output.relative_to(PROJECT_ROOT))
        result["accepted_bundle_rows"] = int(len(bank_ids))
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("DEX5_V9_P3_HOLD_ACTION_SEARCH=" + json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
