from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if os.environ.get("DEX5_DUMP_STACK_AFTER"):
    faulthandler.dump_traceback_later(
        float(os.environ["DEX5_DUMP_STACK_AFTER"]), repeat=False
    )


parser = argparse.ArgumentParser(description="Smoke-test the physical Dex5 + pen DirectRLEnv.")
parser.add_argument("--task", default="Dex5-IndexNearSurface-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--proxy_radius_scale", type=float, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--bundle_path", type=Path, default=None)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--export_phase", type=int, default=None)
parser.add_argument("--export_cache", type=Path, default=None)
parser.add_argument("--reset_joint_pos_noise", type=float, default=None)
parser.add_argument("--reset_joint_vel_noise", type=float, default=None)
parser.add_argument("--reset_pen_pos_noise", type=float, default=None)
parser.add_argument("--reset_pen_vel_noise", type=float, default=None)
parser.add_argument("--max_success_pen_drift_mm", type=float, default=None)
parser.add_argument("--hold_steps", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch import nn
from pxr import Usd, UsdPhysics

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def stats(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float()
    return {
        "mean": float(value.mean().cpu()),
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
    }


def p4_gate_masks(direct_env) -> dict[str, torch.Tensor]:
    initial_yaw = torch.atan2(
        direct_env.init_pen_axis[:, 1], direct_env.init_pen_axis[:, 0]
    )
    current_yaw = torch.atan2(direct_env.pen_axis[:, 1], direct_env.pen_axis[:, 0])
    target_quat = direct_env.target_pen_quat
    target_axis = torch.stack(
        (
            1.0 - 2.0 * (target_quat[:, 2] ** 2 + target_quat[:, 3] ** 2),
            2.0
            * (target_quat[:, 1] * target_quat[:, 2] + target_quat[:, 0] * target_quat[:, 3]),
            2.0
            * (target_quat[:, 1] * target_quat[:, 3] - target_quat[:, 0] * target_quat[:, 2]),
        ),
        dim=-1,
    )
    target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
    direction = float(direct_env.cfg.phase_direction_sign)
    remaining_error = direction * torch.atan2(
        torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
    )
    target_phase = torch.deg2rad(
        torch.full_like(remaining_error, direct_env.cfg.target_total_phase_deg)
    )
    total_phase_deg = (target_phase - remaining_error) * (180.0 / torch.pi)
    contacts = direct_env.contact_force_norm > 0.03
    hand_rmse = torch.sqrt(
        torch.mean((direct_env.hand_qpos - direct_env.target_hand_qpos) ** 2, dim=-1)
    )
    pen_drift = torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
    return {
        "phase_band": (total_phase_deg >= direct_env.cfg.min_phase_deg)
        & (total_phase_deg <= direct_env.cfg.max_phase_deg),
        "angular_speed": torch.linalg.norm(direct_env.pen_angvel, dim=-1)
        <= direct_env.cfg.max_braked_angular_speed,
        "triad_contact": contacts[:, :3].all(dim=-1),
        "thumb_force": direct_env.contact_force_norm[:, 0]
        >= direct_env.cfg.min_success_thumb_force_n,
        "hand_rmse": hand_rmse < direct_env.cfg.max_success_hand_rmse,
        "pen_drift": pen_drift < direct_env.cfg.max_success_pen_drift_m,
        "penetration": direct_env.episode_max_penetration
        <= direct_env.cfg.max_success_penetration_m,
    }


class Policy(nn.Module):
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        super().__init__()
        data = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        state = data["policy"]
        input_size = int(state["net_container.0.weight"].shape[1])
        output_size = int(state["policy_layer.weight"].shape[0])
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_size),
        )
        for layer, key in (
            (0, "net_container.0"),
            (2, "net_container.2"),
            (4, "net_container.4"),
            (6, "policy_layer"),
        ):
            self.net[layer].weight.data.copy_(state[f"{key}.weight"])
            self.net[layer].bias.data.copy_(state[f"{key}.bias"])
        preprocessor = data["observation_preprocessor"]
        self.register_buffer("mean", preprocessor["running_mean"].float())
        self.register_buffer(
            "variance", preprocessor["running_variance"].float()
        )
        self.to(device).eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(
            (observation - self.mean)
            / (torch.sqrt(self.variance) + 1.0e-8),
            -5.0,
            5.0,
        )
        return self.net(normalized)


def main() -> None:
    task = args.task
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    if args.bundle_path is not None:
        env_cfg.bundle_path = str(project_path(args.bundle_path))
    if args.proxy_radius_scale is not None:
        env_cfg.collision_proxy_radius_scale = args.proxy_radius_scale
    for name in (
        "reset_joint_pos_noise",
        "reset_joint_vel_noise",
        "reset_pen_pos_noise",
        "reset_pen_vel_noise",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(env_cfg, name, value)
    if args.max_success_pen_drift_mm is not None:
        env_cfg.max_success_pen_drift_m = args.max_success_pen_drift_mm / 1000.0
    if args.hold_steps is not None:
        env_cfg.hold_steps = args.hold_steps
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s, (args.steps + 2) * env_cfg.decimation * env_cfg.sim.dt
    )
    env = gym.make(task, cfg=env_cfg)
    direct_env = env.unwrapped
    observation, _ = env.reset()
    policy = (
        Policy(project_path(args.checkpoint), direct_env.device)
        if args.checkpoint is not None
        else None
    )
    direct_env._compute_intermediate_values()
    stage = direct_env.scene.stage
    robot_collision_prims = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith("/World/envs/env_0/Robot")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    enabled_robot_collision_prims = [
        prim
        for prim in robot_collision_prims
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]
    prototype_prims = [prim for prototype in stage.GetPrototypes() for prim in Usd.PrimRange(prototype)]
    prototype_collision_prims = [prim for prim in prototype_prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    collision_named_prims = [
        {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "schemas": list(prim.GetAppliedSchemas()),
        }
        for prim in stage.Traverse()
        if "collisions" in str(prim.GetPath())
    ][:5]
    initial_tip_dist = direct_env.tip_surface_dist.detach().clone()
    initial_pen_pos = direct_env.pen_pos.detach().clone()
    reset_alignment = {}
    if hasattr(direct_env, "current_bank_ids") and hasattr(direct_env, "source_qpos_bank"):
        bank_ids = direct_env.current_bank_ids
        source_qpos = direct_env.source_qpos_bank[bank_ids]
        runtime_qpos = torch.cat(
            (direct_env.hand_qpos, direct_env.pen_pos, direct_env.pen_quat), dim=-1
        )
        reset_alignment = {
            "bank_id_min": int(bank_ids.min().cpu()),
            "bank_id_max": int(bank_ids.max().cpu()),
            "qpos_max_abs_error": float(
                torch.max(torch.abs(runtime_qpos - source_qpos)).cpu()
            ),
        }

    terminated_total = torch.zeros(direct_env.num_envs, dtype=torch.bool, device=direct_env.device)
    truncated_total = torch.zeros_like(terminated_total)
    contact_step_sum = torch.zeros(5, dtype=torch.float32, device=direct_env.device)
    triad_contact_all_steps = torch.ones_like(terminated_total)
    triad_contact_steps = torch.zeros(
        direct_env.num_envs, dtype=torch.int32, device=direct_env.device
    )
    triad_contact_run = torch.zeros_like(triad_contact_steps)
    triad_contact_max_run = torch.zeros_like(triad_contact_steps)
    reward = torch.zeros(direct_env.num_envs, device=direct_env.device)
    chain_phase_max = (
        direct_env.chain_phase_ids.clone()
        if hasattr(direct_env, "chain_phase_ids")
        else None
    )
    p4_hold_max = (
        direct_env.p4_hold_count.clone()
        if hasattr(direct_env, "p4_hold_count")
        else None
    )
    p4_gate_step_sum = {}
    p4_gate_ever = {}
    p4_gate_all_except_ever = {}
    phase_capture = {}
    phase_captured = torch.zeros_like(terminated_total)
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = (
                policy(observation["policy"])
                if policy is not None
                else torch.zeros(
                    (
                        direct_env.num_envs,
                        direct_env.cfg.action_space,
                    ),
                    device=direct_env.device,
                )
            )
        observation, reward, terminated, truncated, _ = env.step(actions)
        terminated_total |= terminated
        truncated_total |= truncated
        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > 0.03
        triad_contact = contacts[:, :3].all(dim=-1)
        contact_step_sum += contacts.float().sum(dim=0)
        triad_contact_all_steps &= triad_contact
        triad_contact_steps += triad_contact.int()
        triad_contact_run = torch.where(
            triad_contact, triad_contact_run + 1, torch.zeros_like(triad_contact_run)
        )
        triad_contact_max_run = torch.maximum(triad_contact_max_run, triad_contact_run)
        if chain_phase_max is not None:
            chain_phase_max = torch.maximum(
                chain_phase_max, direct_env.chain_phase_ids
            )
        if p4_hold_max is not None:
            p4_hold_max = torch.maximum(
                p4_hold_max, direct_env.p4_hold_count
            )
            gates = p4_gate_masks(direct_env)
            if not p4_gate_step_sum:
                p4_gate_step_sum = {
                    name: torch.zeros((), device=direct_env.device) for name in gates
                }
                p4_gate_ever = {
                    name: torch.zeros_like(terminated_total) for name in gates
                }
                p4_gate_all_except_ever = {
                    name: torch.zeros_like(terminated_total) for name in gates
                }
            for name, gate in gates.items():
                p4_gate_step_sum[name] += gate.float().sum()
                p4_gate_ever[name] |= gate
                others = [value for other, value in gates.items() if other != name]
                p4_gate_all_except_ever[name] |= torch.stack(others).all(dim=0)
        if (
            args.export_phase is not None
            and hasattr(direct_env, "chain_phase_ids")
        ):
            capture = (
                (direct_env.chain_phase_ids == args.export_phase)
                & ~phase_captured
                & ~terminated
                & ~truncated
            )
            if torch.any(capture):
                phase_captured |= capture
                snapshot = {
                    "source_qpos": torch.cat(
                        (
                            direct_env.hand_qpos,
                            direct_env.pen_pos,
                            direct_env.pen_quat,
                        ),
                        dim=-1,
                    )[capture].detach().cpu().numpy(),
                    "source_qvel": torch.cat(
                        (
                            direct_env.hand_qvel,
                            direct_env.pen_linvel,
                            direct_env.pen_angvel,
                        ),
                        dim=-1,
                    )[capture].detach().cpu().numpy(),
                    "source_policy_action": direct_env.actions[
                        capture
                    ].detach().cpu().numpy(),
                    "source_policy_targets": direct_env.policy_targets[
                        capture
                    ].detach().cpu().numpy(),
                    "source_target_hand_qpos": direct_env.target_hand_qpos[
                        capture
                    ].detach().cpu().numpy(),
                    "source_closure_origin_hand": direct_env.closure_origin_hand[
                        capture
                    ].detach().cpu().numpy(),
                    "source_closure_brace_hand": direct_env.closure_brace_hand[
                        capture
                    ].detach().cpu().numpy(),
                    "source_parent_init_pen_pos": direct_env.init_pen_pos[
                        capture
                    ].detach().cpu().numpy(),
                    "source_parent_init_pen_axis": direct_env.init_pen_axis[
                        capture
                    ].detach().cpu().numpy(),
                    "source_max_penetration_m": direct_env.episode_max_penetration[
                        capture
                    ].detach().cpu().numpy(),
                    "source_contact_force_n": direct_env.contact_force_norm[
                        capture
                    ].detach().cpu().numpy(),
                }
                for name, rows in snapshot.items():
                    phase_capture.setdefault(name, []).append(rows)

    direct_env._compute_intermediate_values()
    p4_metrics = {}
    if hasattr(direct_env.cfg, "target_total_phase_deg"):
        initial_yaw = torch.atan2(direct_env.init_pen_axis[:, 1], direct_env.init_pen_axis[:, 0])
        current_yaw = torch.atan2(direct_env.pen_axis[:, 1], direct_env.pen_axis[:, 0])
        target_axis = direct_env.target_pen_quat
        target_axis = torch.stack(
            (
                1.0 - 2.0 * (target_axis[:, 2] ** 2 + target_axis[:, 3] ** 2),
                2.0 * (target_axis[:, 1] * target_axis[:, 2] + target_axis[:, 0] * target_axis[:, 3]),
                2.0 * (target_axis[:, 1] * target_axis[:, 3] - target_axis[:, 0] * target_axis[:, 2]),
            ),
            dim=-1,
        )
        target_yaw = torch.atan2(target_axis[:, 1], target_axis[:, 0])
        direction = float(direct_env.cfg.phase_direction_sign)
        relative_deflection = direction * torch.atan2(
            torch.sin(current_yaw - initial_yaw), torch.cos(current_yaw - initial_yaw)
        )
        remaining_error = direction * torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        target_phase = torch.deg2rad(
            torch.full_like(relative_deflection, direct_env.cfg.target_total_phase_deg)
        )
        total_phase = target_phase - remaining_error
        p4_metrics = {
            "relative_deflection_deg": stats(relative_deflection * (180.0 / torch.pi)),
            "total_phase_deg": stats(total_phase * (180.0 / torch.pi)),
            "angular_speed_rad_s": stats(torch.linalg.norm(direct_env.pen_angvel, dim=-1)),
            "final_hold_steps": stats(direct_env.p4_hold_count.float()),
        }
        if p4_hold_max is not None:
            required_hold_steps = int(direct_env.cfg.hold_steps)
            p4_metrics.update(
                {
                    "max_hold_steps": stats(p4_hold_max.float()),
                    "required_hold_steps": required_hold_steps,
                    "required_hold_ms": (
                        required_hold_steps
                        * env_cfg.decimation
                        * env_cfg.sim.dt
                        * 1000.0
                    ),
                    "required_max_pen_drift_mm": (
                        direct_env.cfg.max_success_pen_drift_m * 1000.0
                    ),
                    "success_ever_ratio": float(
                        (p4_hold_max >= required_hold_steps)
                        .float()
                        .mean()
                        .cpu()
                    ),
                }
            )
            if p4_gate_step_sum:
                final_gates = p4_gate_masks(direct_env)
                p4_metrics["gate_diagnostics"] = {
                    name: {
                        "step_ratio": float(
                            (
                                p4_gate_step_sum[name]
                                / (args.steps * direct_env.num_envs)
                            ).cpu()
                        ),
                        "ever_ratio": float(
                            p4_gate_ever[name].float().mean().cpu()
                        ),
                        "all_except_ever_ratio": float(
                            p4_gate_all_except_ever[name].float().mean().cpu()
                        ),
                        "final_ratio": float(
                            final_gates[name].float().mean().cpu()
                        ),
                    }
                    for name in p4_gate_step_sum
                }
    report = {
        "task": task,
        "num_envs": direct_env.num_envs,
        "steps": args.steps,
        "observation_shape": list(observation["policy"].shape),
        "reward": stats(reward),
        "initial_pen_pos_m": initial_pen_pos.mean(dim=0).cpu().tolist(),
        "initial_tip_surface_dist_mm": {
            name: stats(initial_tip_dist[:, index] * 1000.0)
            for index, name in enumerate(("thumb", "index", "middle", "ring", "pinky"))
        },
        "tip_surface_dist_mm": {
            name: stats(direct_env.tip_surface_dist[:, index] * 1000.0)
            for index, name in enumerate(("thumb", "index", "middle", "ring", "pinky"))
        },
        "contact_force_n": {
            name: stats(direct_env.contact_force_norm[:, index])
            for index, name in enumerate(("thumb", "index", "middle", "ring", "pinky"))
        },
        "contact_threshold_n": 0.03,
        "step_contact_ratio": {
            name: float((contact_step_sum[index] / (direct_env.num_envs * args.steps)).cpu())
            for index, name in enumerate(("thumb", "index", "middle", "ring", "pinky"))
        },
        "triad_contact": {
            "required_fingers": ["thumb", "index", "middle"],
            "all_steps_ratio": float(triad_contact_all_steps.float().mean().cpu()),
            "mean_contact_steps": float(triad_contact_steps.float().mean().cpu()),
            "mean_contact_ms": float(
                triad_contact_steps.float().mean().cpu()
                * env_cfg.decimation
                * env_cfg.sim.dt
                * 1000.0
            ),
            "mean_max_consecutive_steps": float(triad_contact_max_run.float().mean().cpu()),
            "hold_100ms_ratio": float(
                (
                    triad_contact_max_run.float()
                    * env_cfg.decimation
                    * env_cfg.sim.dt
                    >= 0.100 - 1.0e-6
                )
                .float()
                .mean()
                .cpu()
            ),
        },
        "index_tip_dist_mm": stats(direct_env.tip_surface_dist[:, 1] * 1000.0),
        "index_contact_force_n": stats(direct_env.contact_force_norm[:, 1]),
        "pen_height_drop_mm": stats(
            (direct_env.init_pen_pos[:, 2] - direct_env.pen_pos[:, 2]) * 1000.0
        ),
        "pen_drift_mm": stats(
            torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1) * 1000.0
        ),
        "max_penetration_mm": stats(direct_env.episode_max_penetration * 1000.0),
        "terminated_ever_ratio": float(terminated_total.float().mean().cpu()),
        "truncated_ever_ratio": float(truncated_total.float().mean().cpu()),
        "finite_observation": bool(torch.isfinite(observation["policy"]).all().cpu()),
        "robot_collision_prim_count": len(robot_collision_prims),
        "enabled_robot_collision_prim_count": len(enabled_robot_collision_prims),
        "prototype_collision_prim_count": len(prototype_collision_prims),
        "collision_named_prims": collision_named_prims,
        "reset_alignment": reset_alignment,
        "p4_metrics": p4_metrics,
    }
    if chain_phase_max is not None:
        report["chain_metrics"] = {
            "phase_max": stats(chain_phase_max.float()),
            "final_phase": int(
                direct_env.cfg.training_start_phase
                + len(direct_env.cfg.closure_target_blends)
                - 1
            ),
            "stage_entries": (
                direct_env.closure_stage_entries.detach().cpu().tolist()
            ),
            "stage_contact_steps": (
                direct_env.closure_stage_triad_step_counts.detach()
                .cpu()
                .tolist()
            ),
            "stage_finger_contact_steps": (
                direct_env.closure_stage_finger_contact_step_counts.detach()
                .cpu()
                .tolist()
            ),
            "stage_transfer_component_steps": (
                direct_env.closure_stage_transfer_component_step_counts.detach()
                .cpu()
                .tolist()
            ),
            "stage_safe_steps": (
                direct_env.closure_stage_safe_step_counts.detach()
                .cpu()
                .tolist()
            ),
            "stage_safe_component_steps": (
                direct_env.closure_stage_safe_component_step_counts.detach()
                .cpu()
                .tolist()
            ),
            "stage_pose_steps": (
                direct_env.closure_stage_pose_step_counts.detach()
                .cpu()
                .tolist()
            ),
            "stage_contact_pose_steps": (
                direct_env.closure_stage_contact_pose_step_counts.detach()
                .cpu()
                .tolist()
            ),
            "stage_eligible_steps": (
                direct_env.closure_stage_eligible_step_counts.detach()
                .cpu()
                .tolist()
            ),
            "stage_min_target_rmse_rad": (
                direct_env.closure_stage_min_target_rmse.detach()
                .cpu()
                .tolist()
            ),
            "stage_min_position_error_mm": (
                direct_env.closure_stage_min_position_error.detach()
                .cpu()
                .mul(1000.0)
                .tolist()
            ),
            "stage_min_axis_error_deg": (
                direct_env.closure_stage_min_axis_error.detach()
                .cpu()
                .mul(180.0 / torch.pi)
                .tolist()
            ),
            "stage_best_hold_steps": (
                direct_env.closure_stage_best_hold.detach().cpu().tolist()
            ),
        }
    if args.export_phase is not None:
        report["phase_cache_capture_count"] = int(
            phase_captured.sum().cpu()
        )
    if args.export_cache is not None:
        if args.export_phase is None:
            raise ValueError("--export_cache requires --export_phase")
        if not phase_capture:
            raise RuntimeError(
                f"No environments reached phase {args.export_phase}"
            )
        phase_capture = {
            name: np.concatenate(rows, axis=0).astype(np.float32)
            for name, rows in phase_capture.items()
        }
        qpos = phase_capture["source_qpos"]
        qvel = phase_capture["source_qvel"]
        pen_pos = qpos[:, 20:23]
        pen_quat = qpos[:, 23:27]
        # Rotate the local x-axis with the captured quaternion.
        w, x, y, z = np.moveaxis(pen_quat, -1, 0)
        pen_axis = np.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                2.0 * (x * z - w * y),
            ),
            axis=-1,
        ).astype(np.float32)
        cache_path = project_path(args.export_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            **phase_capture,
            source_target_qpos=qpos[:, :20],
            source_closure_origin_pen_pos=pen_pos,
            source_closure_origin_pen_quat=pen_quat,
            source_init_pen_pos=pen_pos,
            source_init_pen_axis=pen_axis,
            source_chain_phase_id=np.full(
                len(qpos), args.export_phase, dtype=np.int64
            ),
            target_qpos=qpos,
            target_qvel=np.zeros_like(qvel),
            # The base reset loader uses phase_ids only as its legacy
            # curriculum-bank selector (0-4). The continuous chain phase is
            # stored independently in source_chain_phase_id.
            phase_ids=np.zeros(len(qpos), dtype=np.int64),
            seed=np.asarray(args.seed, dtype=np.int64),
            source=np.asarray(
                "policy phase-entry local curriculum cache",
                dtype=np.str_,
            ),
        )
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text + "\n", encoding="utf-8")
    print("DEX5_PHYSICS_SMOKE=" + report_text, flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
