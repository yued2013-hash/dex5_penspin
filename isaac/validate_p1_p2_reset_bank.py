from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


parser = argparse.ArgumentParser(description="Exhaustively validate P1/P2 reset-bank rows.")
parser.add_argument("--task", default="Dex5-V9P1-P2-ResetBank-Direct-v0")
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--filtered_bundle", type=Path, default=None)
parser.add_argument("--progress", action="store_true")
parser.add_argument(
    "--paired-ring-under",
    action="store_true",
    help="Validate the P1-paired ring-under manifold without using the legacy P2 center.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    bundle_path = resolved(args.bundle)
    output = resolved(args.output)
    with np.load(bundle_path) as data:
        source_qpos_np = np.asarray(data["source_qpos"], dtype=np.float32)
        source_qvel_np = np.asarray(data["source_qvel"], dtype=np.float32)
        source_target_np = np.asarray(data["source_target_qpos"], dtype=np.float32)
        target_qpos_np = np.asarray(
            data.get("target_qpos", source_qpos_np.copy()), dtype=np.float32
        )
        source_policy_action_np = np.asarray(
            data.get("source_policy_action", np.zeros((source_qpos_np.shape[0], 20), dtype=np.float32)),
            dtype=np.float32,
        )
        source_teacher_actions_np = np.asarray(
            data.get("source_teacher_actions", np.zeros((source_qpos_np.shape[0], 1, 20), dtype=np.float32)),
            dtype=np.float32,
        )
        target_qvel_np = np.asarray(
            data.get("target_qvel", np.zeros((source_qpos_np.shape[0], 26), dtype=np.float32)),
            dtype=np.float32,
        )
        phase_ids_np = np.asarray(data["phase_ids"], dtype=np.int64)

    if args.progress:
        print(f"[validate] loaded {source_qpos_np.shape[0]} rows from {bundle_path}", flush=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.bundle_path = str(bundle_path)
    if args.paired_ring_under and hasattr(env_cfg, "source_phase_id"):
        env_cfg.source_phase_id = 2
        env_cfg.use_all_bundle_rows = True
        env_cfg.teacher_prior_state_map_path = None
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    env_cfg.teacher_prefill_steps = 0
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s, (args.steps + 2) * env_cfg.decimation * env_cfg.sim.dt
    )
    if args.progress:
        print(f"[validate] creating env task={args.task} num_envs={args.num_envs}", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    direct = env.unwrapped
    env.reset()
    if args.progress:
        print("[validate] env ready", flush=True)

    phase_ids = torch.as_tensor(phase_ids_np, dtype=torch.long, device=direct.device)

    total_rows = int(source_qpos_np.shape[0])
    device = direct.device
    valid = torch.zeros(total_rows, dtype=torch.bool, device=device)
    invalid = torch.zeros(total_rows, dtype=torch.bool, device=device)
    drop = torch.zeros(total_rows, dtype=torch.bool, device=device)
    terminated_ever = torch.zeros(total_rows, dtype=torch.bool, device=device)
    max_penetration = torch.zeros(total_rows, device=device)
    initial_penetration = torch.zeros(total_rows, device=device)
    max_drift = torch.zeros(total_rows, device=device)
    max_ang_speed = torch.zeros(total_rows, device=device)
    max_axis_drift = torch.zeros(total_rows, device=device)
    max_ring_force = torch.zeros(total_rows, device=device)
    max_pinky_force = torch.zeros(total_rows, device=device)
    semantic_steps = torch.zeros(total_rows, device=device)
    contact_steps = torch.zeros((total_rows, 5), device=device)
    force_sum = torch.zeros((total_rows, 5), device=device)

    actions = torch.zeros((args.num_envs, direct.cfg.action_space), device=device)

    for start in range(0, total_rows, args.num_envs):
        end = min(start + args.num_envs, total_rows)
        count = end - start
        env_ids = torch.arange(count, device=device)
        rows = torch.arange(start, end, device=device)
        trace_first = args.progress and start == 0
        if trace_first:
            print(f"[validate] first chunk begin count={count}", flush=True)

        if trace_first:
            print("[validate] forced env reset", flush=True)
        direct.forced_bank_ids = rows
        direct._reset_idx(env_ids)
        direct.needs_initial_distance[env_ids] = False
        if trace_first:
            print("[validate] compute initial intermediates", flush=True)
        direct._compute_intermediate_values()
        initial_penetration[rows] = direct.current_max_penetration[env_ids]
        direct.episode_max_penetration[env_ids] = direct.current_max_penetration[env_ids]
        direct.episode_max_penetration_by_finger[env_ids] = direct.current_penetration_by_finger[env_ids]

        local_contact_steps = torch.zeros((count, 5), device=device)
        local_force_sum = torch.zeros((count, 5), device=device)
        local_terminated = torch.zeros(count, dtype=torch.bool, device=device)
        local_drop = torch.zeros(count, dtype=torch.bool, device=device)
        local_invalid = torch.zeros(count, dtype=torch.bool, device=device)
        local_max_drift = torch.zeros(count, device=device)
        local_max_ang = torch.zeros(count, device=device)
        local_max_axis_drift = torch.zeros(count, device=device)
        local_max_ring_force = torch.zeros(count, device=device)
        local_max_pinky_force = torch.zeros(count, device=device)
        local_semantic_steps = torch.zeros(count, device=device)
        initial_axis = direct.pen_axis[env_ids].clone()

        for _ in range(args.steps):
            direct._pre_physics_step(actions)
            for _ in range(direct.cfg.decimation):
                direct._apply_action()
                direct.scene.write_data_to_sim()
                direct.sim.step(render=False)
                direct.scene.update(dt=direct.physics_dt)
            direct.episode_length_buf[env_ids] += 1
            direct._compute_intermediate_values()
            failed, _ = direct._compute_dones()
            local_terminated |= failed[env_ids]
            height_drop = direct.init_pen_pos[env_ids, 2] - direct.pen_pos[env_ids, 2]
            local_drop |= height_drop > direct.cfg.drop_threshold
            row_invalid = (
                ~torch.isfinite(direct.pen_pos[env_ids]).all(dim=-1)
                | ~torch.isfinite(direct.hand_qpos[env_ids]).all(dim=-1)
            )
            local_invalid |= row_invalid
            local_contact_steps += (direct.contact_force_norm[env_ids] > 0.03).float()
            local_force_sum += direct.contact_force_norm[env_ids]
            local_max_drift = torch.maximum(
                local_max_drift,
                torch.linalg.norm(direct.pen_pos[env_ids] - direct.init_pen_pos[env_ids], dim=-1),
            )
            local_max_ang = torch.maximum(
                local_max_ang, torch.linalg.norm(direct.pen_angvel[env_ids], dim=-1)
            )
            axis_dot = torch.sum(
                direct.pen_axis[env_ids] * initial_axis, dim=-1
            ).abs().clamp(-1.0, 1.0)
            local_max_axis_drift = torch.maximum(local_max_axis_drift, torch.acos(axis_dot))
            local_max_ring_force = torch.maximum(
                local_max_ring_force, direct.contact_force_norm[env_ids, 3]
            )
            local_max_pinky_force = torch.maximum(
                local_max_pinky_force, direct.contact_force_norm[env_ids, 4]
            )
            if args.paired_ring_under:
                feature = direct._ring_axis_features()[env_ids]
                axial, lateral, up = feature.unbind(dim=-1)
                ring_gap = (
                    direct.tip_surface_dist[env_ids, 3]
                    - direct.fingertip_radii[0, 3]
                )
                semantic = (
                    (axial >= -0.060)
                    & (axial <= -0.020)
                    & (lateral >= -0.010)
                    & (lateral <= 0.026)
                    & (up >= -0.026)
                    & (up <= -0.004)
                    & ((-up) >= torch.abs(lateral) + 0.001)
                    & (ring_gap >= -0.0006)
                    & (ring_gap <= 0.0030)
                )
                local_semantic_steps += semantic.float()

        contacts_ok = (local_contact_steps[:, :3] >= args.steps).all(dim=-1)
        penetration_ok = direct.episode_max_penetration[env_ids] <= direct.cfg.penetration_threshold
        drift_ok = local_max_drift <= direct.cfg.drift_threshold
        speed_ok = local_max_ang <= 12.0
        axis_ok = local_max_axis_drift <= torch.deg2rad(
            torch.full_like(local_max_axis_drift, 4.0)
        )
        load_ok = (local_max_ring_force <= 0.30) & (local_max_pinky_force <= 0.03)
        semantic_ok = (
            local_semantic_steps >= args.steps if args.paired_ring_under else torch.ones_like(contacts_ok)
        )
        valid_rows = (
            contacts_ok
            & penetration_ok
            & drift_ok
            & speed_ok
            & axis_ok
            & load_ok
            & semantic_ok
            & ~local_drop
            & ~local_invalid
        )

        valid[rows] = valid_rows
        invalid[rows] = local_invalid
        drop[rows] = local_drop
        terminated_ever[rows] = local_terminated
        max_penetration[rows] = direct.episode_max_penetration[env_ids]
        max_drift[rows] = local_max_drift
        max_ang_speed[rows] = local_max_ang
        max_axis_drift[rows] = local_max_axis_drift
        max_ring_force[rows] = local_max_ring_force
        max_pinky_force[rows] = local_max_pinky_force
        semantic_steps[rows] = local_semantic_steps
        contact_steps[rows] = local_contact_steps
        force_sum[rows] = local_force_sum
        if args.progress:
            print(
                f"[validate] rows {start}:{end} valid={int(valid_rows.sum().item())}/{count} "
                f"max_pen={float(direct.episode_max_penetration[env_ids].max().item() * 1000.0):.3f}mm",
                flush=True,
            )

    phase_report: dict[str, dict[str, float | int]] = {}
    for phase in sorted(np.unique(phase_ids_np).tolist()):
        mask = phase_ids == int(phase)
        count = int(mask.sum().item())
        phase_force_sum = force_sum[mask].sum(dim=0)
        phase_total_force = phase_force_sum.sum().clamp_min(1.0e-6)
        phase_report[str(int(phase))] = {
            "rows": count,
            "valid_rows": int((valid & mask).sum().item()),
            "hold_rate": float((valid & mask).sum().item() / max(count, 1)),
            "drop_rate": float((drop & mask).sum().item() / max(count, 1)),
            "invalid_rows": int((invalid & mask).sum().item()),
            "max_initial_penetration_mm": float(initial_penetration[mask].max().item() * 1000.0),
            "max_penetration_mm": float(max_penetration[mask].max().item() * 1000.0),
            "max_drift_mm": float(max_drift[mask].max().item() * 1000.0),
            "max_angular_speed_rad_s": float(max_ang_speed[mask].max().item()),
            "max_axis_drift_deg": float(torch.rad2deg(max_axis_drift[mask]).max().item()),
            "max_ring_force_n": float(max_ring_force[mask].max().item()),
            "max_pinky_force_n": float(max_pinky_force[mask].max().item()),
            "all_step_ring_under_rate": float(
                (semantic_steps[mask] >= args.steps).float().mean().item()
            ) if args.paired_ring_under else None,
            "all_step_thumb_contact_rate": float((contact_steps[mask, 0] >= args.steps).float().mean().item()),
            "all_step_index_contact_rate": float((contact_steps[mask, 1] >= args.steps).float().mean().item()),
            "all_step_middle_contact_rate": float((contact_steps[mask, 2] >= args.steps).float().mean().item()),
            "ring_force_share": float((phase_force_sum[3] / phase_total_force).item()),
            "pinky_force_share": float((phase_force_sum[4] / phase_total_force).item()),
        }

    report = {
        "bundle": str(bundle_path.relative_to(PROJECT_ROOT)),
        "rows": total_rows,
        "steps": args.steps,
        "elapsed_ms": args.steps * env_cfg.decimation * env_cfg.sim.dt * 1000.0,
        "thresholds": {
            "penetration_mm": env_cfg.penetration_threshold * 1000.0,
            "drift_mm": env_cfg.drift_threshold * 1000.0,
            "drop_mm": env_cfg.drop_threshold * 1000.0,
            "angular_speed_rad_s": 12.0,
            "axis_drift_deg": 4.0,
            "ring_force_n": 0.30,
            "pinky_force_n": 0.03,
        },
        "paired_ring_under_gate": bool(args.paired_ring_under),
        "total_valid_rows": int(valid.sum().item()),
        "total_hold_rate": float(valid.float().mean().item()),
        "total_drop_rate": float(drop.float().mean().item()),
        "total_invalid_rows": int(invalid.sum().item()),
        "max_initial_penetration_mm": float(initial_penetration.max().item() * 1000.0),
        "max_penetration_mm": float(max_penetration.max().item() * 1000.0),
        "phase_report": phase_report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_RESET_BANK_VALIDATE=" + json.dumps(report, ensure_ascii=False), flush=True)
    if args.filtered_bundle is not None:
        filtered_bundle = resolved(args.filtered_bundle)
        valid_np = valid.cpu().numpy()
        filtered_bundle.parent.mkdir(parents=True, exist_ok=True)
        with np.load(bundle_path, allow_pickle=False) as source:
            payload = {
                key: (
                    np.asarray(source[key])[valid_np]
                    if np.asarray(source[key]).ndim > 0
                    and np.asarray(source[key]).shape[0] == total_rows
                    else np.asarray(source[key])
                )
                for key in source.files
            }
        payload.setdefault("target_qpos", target_qpos_np[valid_np])
        payload.setdefault("target_qvel", target_qvel_np[valid_np])
        payload.setdefault("source_policy_action", source_policy_action_np[valid_np])
        payload.setdefault("source_teacher_actions", source_teacher_actions_np[valid_np])
        selected_local_rows = np.nonzero(valid_np)[0].astype(np.int64)
        if "source_row_ids" in source.files:
            payload["source_row_ids"] = np.asarray(source["source_row_ids"])[valid_np]
        else:
            payload["source_row_ids"] = selected_local_rows
        payload["parent_bundle_row_ids"] = selected_local_rows
        np.savez_compressed(filtered_bundle, **payload)
        if args.progress:
            print(f"[validate] wrote filtered bundle: {filtered_bundle}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
