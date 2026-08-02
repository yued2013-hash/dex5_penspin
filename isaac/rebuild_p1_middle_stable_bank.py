from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Rebuild train P1 resets through PhysX middle-target settling.")
parser.add_argument("--task", default="Dex5-P1P2-RingOnly120HzH1-Direct-v0")
parser.add_argument("--source-bundle", type=Path)
parser.add_argument("--source-ids", type=Path)
parser.add_argument("--output-bank", type=Path, required=True)
parser.add_argument("--output-ids", type=Path, required=True)
parser.add_argument("--output-diagnostics", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--settle-steps", type=int, default=18)
parser.add_argument("--hold-steps", type=int, default=30)
parser.add_argument("--middle-contact-ratio-min", type=float, default=0.95)
parser.add_argument(
    "--pen-axis-offsets-mm",
    default="0",
    help="Comma-separated reset offsets along the pen local X axis.",
)
parser.add_argument("--terminal-linear-speed-max", type=float, default=0.02)
parser.add_argument("--terminal-angular-speed-max", type=float, default=1.0)
parser.add_argument("--save-state", choices=("initial", "final"), default="final")
parser.add_argument("--seed", type=int, default=2821)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_physics_gates(
    direct,
    initial_pen_pos: torch.Tensor,
    initial_pen_axis: torch.Tensor,
    failed: torch.Tensor,
    drop: torch.Tensor,
    invalid: torch.Tensor,
    max_position_drift: torch.Tensor,
    max_axis_drift: torch.Tensor,
    max_penetration: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> None:
    position_drift = torch.linalg.norm(direct.pen_pos - initial_pen_pos, dim=-1)
    axis_dot = torch.sum(direct.pen_axis * initial_pen_axis, dim=-1).abs().clamp(-1.0, 1.0)
    axis_drift = torch.acos(axis_dot)
    penetration = direct.episode_max_penetration
    finite = torch.isfinite(direct.hand_qpos).all(dim=-1) & torch.isfinite(direct.pen_pos).all(dim=-1)
    current_drop = (initial_pen_pos[:, 2] - direct.pen_pos[:, 2]) > direct.cfg.drop_threshold
    current_invalid = ~finite
    failed |= (
        (position_drift > 0.010)
        | (axis_drift > torch.deg2rad(torch.full_like(axis_drift, 8.0)))
        | (penetration > 0.00060)
        | current_drop
        | current_invalid
        | terminated
        | truncated
    )
    drop |= current_drop
    invalid |= current_invalid
    torch.maximum(max_position_drift, position_drift, out=max_position_drift)
    torch.maximum(max_axis_drift, axis_drift, out=max_axis_drift)
    torch.maximum(max_penetration, penetration, out=max_penetration)


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if cfg.split_name != "train" or int(cfg.source_phase_id) != 1:
        raise ValueError("P1 rebuilding is restricted to the frozen train P1 split")
    cfg.seed = args.seed
    source_bundle = args.source_bundle.resolve() if args.source_bundle else None
    source_ids_path = args.source_ids.resolve() if args.source_ids else None
    if source_bundle is not None:
        cfg.bundle_path = str(source_bundle)
        cfg.teacher_prior_state_map_path = None
    if source_ids_path is not None:
        cfg.split_ids_path = str(source_ids_path)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()
    bank_size = direct.bank_size
    original_global_ids = direct.global_bank_id_bank.detach().cpu().numpy().copy()
    levels = np.asarray([-0.008, -0.004, 0.0, 0.004, 0.008], dtype=np.float32)
    offsets = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1).reshape(-1, 3)
    pen_axis_offsets = np.asarray(
        [float(value.strip()) / 1000.0 for value in args.pen_axis_offsets_mm.split(",")],
        dtype=np.float32,
    )
    candidate_middle_ids, candidate_pen_ids = np.meshgrid(
        np.arange(len(offsets), dtype=np.int64),
        np.arange(len(pen_axis_offsets), dtype=np.int64),
        indexing="ij",
    )
    candidate_middle_ids = candidate_middle_ids.reshape(-1)
    candidate_pen_ids = candidate_pen_ids.reshape(-1)
    candidates_per_source = len(candidate_middle_ids)
    source_rows = np.repeat(np.arange(bank_size, dtype=np.int64), candidates_per_source)
    candidate_rows = np.tile(np.arange(candidates_per_source, dtype=np.int64), bank_size)
    total = len(source_rows)
    best_score = np.full(bank_size, -np.inf, dtype=np.float32)
    accepted = np.zeros(bank_size, dtype=np.bool_)
    best_offset = np.zeros((bank_size, 3), dtype=np.float32)
    best_pen_axis_offset = np.zeros(bank_size, dtype=np.float32)
    best_middle_ratio = np.zeros(bank_size, dtype=np.float32)
    best_triad_ratio = np.zeros(bank_size, dtype=np.float32)
    best_position_drift = np.zeros(bank_size, dtype=np.float32)
    best_axis_drift = np.zeros(bank_size, dtype=np.float32)
    best_penetration = np.zeros(bank_size, dtype=np.float32)
    rebuilt_qpos = np.zeros((bank_size, 27), dtype=np.float32)
    rebuilt_qvel = np.zeros((bank_size, 26), dtype=np.float32)
    rebuilt_target = np.zeros((bank_size, 20), dtype=np.float32)
    all_env_ids = torch.arange(args.num_envs, device=direct.device)
    zero_actions = torch.zeros((args.num_envs, direct.cfg.action_space), device=direct.device)

    for start in range(0, total, args.num_envs):
        count = min(args.num_envs, total - start)
        batch_sources = source_rows[start : start + count]
        batch_candidates = candidate_rows[start : start + count]
        batch_offsets = candidate_middle_ids[batch_candidates]
        batch_pen_offsets = candidate_pen_ids[batch_candidates]
        padded_sources = np.pad(batch_sources, (0, args.num_envs - count), mode="edge")
        padded_offsets = np.pad(batch_offsets, (0, args.num_envs - count), mode="edge")
        direct.forced_bank_ids = torch.as_tensor(padded_sources, device=direct.device)
        direct._reset_idx(all_env_ids)
        direct._compute_intermediate_values()
        offset_tensor = torch.as_tensor(offsets[padded_offsets], device=direct.device)
        direct.support_anchor_targets[:, 5:8] += offset_tensor
        direct.policy_targets[:, 5:8] = direct.support_anchor_targets[:, 5:8]
        pen_axis_offset_tensor = torch.as_tensor(
            pen_axis_offsets[batch_pen_offsets], device=direct.device
        )
        if count < args.num_envs:
            pen_axis_offset_tensor = torch.cat(
                (
                    pen_axis_offset_tensor,
                    pen_axis_offset_tensor[-1:].repeat(args.num_envs - count),
                )
            )
        shifted_pen_pose = torch.cat((direct.pen.data.root_pos_w, direct.pen_quat), dim=-1)
        shifted_pen_pose[:, :3] += direct.pen_axis * pen_axis_offset_tensor.unsqueeze(-1)
        direct.pen.write_root_pose_to_sim(shifted_pen_pose)
        direct.pen.write_root_velocity_to_sim(
            torch.cat((direct.pen_linvel, direct.pen_angvel), dim=-1)
        )
        direct._compute_intermediate_values()
        initial_hand_qpos = direct.hand_qpos.clone()
        initial_hand_qvel = direct.hand_qvel.clone()
        initial_pen_pos = direct.pen_pos.clone()
        initial_pen_quat = direct.pen_quat.clone()
        initial_pen_linvel = direct.pen_linvel.clone()
        initial_pen_angvel = direct.pen_angvel.clone()
        initial_pen_axis = direct.pen_axis.clone()
        failed = torch.zeros(args.num_envs, dtype=torch.bool, device=direct.device)
        drop = torch.zeros_like(failed)
        invalid = torch.zeros_like(failed)
        max_position_drift = torch.zeros(args.num_envs, device=direct.device)
        max_axis_drift = torch.zeros(args.num_envs, device=direct.device)
        max_penetration = torch.zeros(args.num_envs, device=direct.device)
        for _ in range(args.settle_steps):
            _, _, terminated, truncated, _ = env.step(zero_actions)
            direct._compute_intermediate_values()
            update_physics_gates(
                direct, initial_pen_pos, initial_pen_axis, failed, drop, invalid,
                max_position_drift, max_axis_drift, max_penetration, terminated, truncated,
            )
        loss_streak = torch.zeros(args.num_envs, dtype=torch.long, device=direct.device)
        contact_steps = torch.zeros((args.num_envs, 3), device=direct.device)
        for _ in range(args.hold_steps):
            _, _, terminated, truncated, _ = env.step(zero_actions)
            direct._compute_intermediate_values()
            contacts = direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n
            triad = contacts.all(dim=-1)
            contact_steps += contacts.float()
            loss_streak = torch.where(triad, torch.zeros_like(loss_streak), loss_streak + 1)
            failed |= loss_streak > 1
            update_physics_gates(
                direct, initial_pen_pos, initial_pen_axis, failed, drop, invalid,
                max_position_drift, max_axis_drift, max_penetration, terminated, truncated,
            )
        ratios = contact_steps / float(args.hold_steps)
        terminal_contacts = direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n
        terminal_linear_speed = torch.linalg.norm(direct.pen_linvel, dim=-1)
        terminal_angular_speed = torch.linalg.norm(direct.pen_angvel, dim=-1)
        candidate_ok = (
            ~failed
            & (ratios[:, 2] >= float(args.middle_contact_ratio_min))
            & (ratios[:, 0] >= 0.90)
            & (ratios[:, 1] >= 0.90)
            & terminal_contacts.all(dim=-1)
            & (terminal_linear_speed <= float(args.terminal_linear_speed_max))
            & (terminal_angular_speed <= float(args.terminal_angular_speed_max))
        )
        terminal_contact_quality = torch.clamp(
            direct.contact_force_norm[:, :3] / max(float(direct.cfg.p2_contact_force_n), 1.0e-6),
            0.0,
            3.0,
        ).mean(dim=-1)
        score = (
            4.0 * ratios[:, 2]
            + 1.5 * ratios[:, 0]
            + 1.5 * ratios[:, 1]
            + 0.5 * terminal_contact_quality
            - 30.0 * max_position_drift
            - 0.5 * torch.rad2deg(max_axis_drift) / 8.0
            - 100.0 * max_penetration
            - 2.0 * terminal_linear_speed
            - 0.05 * terminal_angular_speed
        )
        if args.save_state == "initial":
            hand_qpos = initial_hand_qpos[:count].detach().cpu().numpy()
            hand_qvel = initial_hand_qvel[:count].detach().cpu().numpy()
            pen_pos = initial_pen_pos[:count].detach().cpu().numpy()
            pen_quat = initial_pen_quat[:count].detach().cpu().numpy()
            pen_linvel = initial_pen_linvel[:count].detach().cpu().numpy()
            pen_angvel = initial_pen_angvel[:count].detach().cpu().numpy()
        else:
            hand_qpos = direct.hand_qpos[:count].detach().cpu().numpy()
            hand_qvel = direct.hand_qvel[:count].detach().cpu().numpy()
            pen_pos = direct.pen_pos[:count].detach().cpu().numpy()
            pen_quat = direct.pen_quat[:count].detach().cpu().numpy()
            pen_linvel = direct.pen_linvel[:count].detach().cpu().numpy()
            pen_angvel = direct.pen_angvel[:count].detach().cpu().numpy()
        target = direct.support_anchor_targets[:count].detach().cpu().numpy()
        ok_np = candidate_ok[:count].cpu().numpy()
        score_np = score[:count].cpu().numpy()
        ratios_np = ratios[:count].cpu().numpy()
        for row in range(count):
            source = int(batch_sources[row])
            if not ok_np[row] or score_np[row] <= best_score[source]:
                continue
            accepted[source] = True
            best_score[source] = score_np[row]
            best_offset[source] = offsets[batch_offsets[row]]
            best_pen_axis_offset[source] = pen_axis_offsets[batch_pen_offsets[row]]
            best_middle_ratio[source] = ratios_np[row, 2]
            best_triad_ratio[source] = min(ratios_np[row])
            best_position_drift[source] = float(max_position_drift[row] * 1000.0)
            best_axis_drift[source] = float(torch.rad2deg(max_axis_drift[row]))
            best_penetration[source] = float(max_penetration[row] * 1000.0)
            rebuilt_qpos[source, :20] = hand_qpos[row]
            rebuilt_qpos[source, 20:23] = pen_pos[row]
            rebuilt_qpos[source, 23:27] = pen_quat[row]
            rebuilt_qvel[source, :20] = hand_qvel[row]
            rebuilt_qvel[source, 20:23] = pen_linvel[row]
            rebuilt_qvel[source, 23:26] = pen_angvel[row]
            rebuilt_target[source] = target[row]
    del direct.forced_bank_ids

    selected = np.flatnonzero(accepted)
    with np.load(cfg.bundle_path, allow_pickle=False) as original:
        phase_ids = np.asarray(original["phase_ids"], dtype=np.int64)
        train_ids = np.asarray(np.load(cfg.split_ids_path, allow_pickle=False), dtype=np.int64)
        original_p1_rows = train_ids[phase_ids[train_ids] == 1]
        source_policy_action = np.asarray(original["source_policy_action"][original_p1_rows[selected]])
        source_teacher_actions = np.asarray(original["source_teacher_actions"][original_p1_rows[selected]])
        target_qpos = np.asarray(original["target_qpos"][original_p1_rows[selected]])
        target_qvel = np.asarray(original["target_qvel"][original_p1_rows[selected]])
        source_row_ids = np.asarray(original["source_row_ids"][original_p1_rows[selected]])
    output_bank = args.output_bank if args.output_bank.is_absolute() else PROJECT_ROOT / args.output_bank
    output_ids = args.output_ids if args.output_ids.is_absolute() else PROJECT_ROOT / args.output_ids
    output_diagnostics = (
        args.output_diagnostics if args.output_diagnostics.is_absolute()
        else PROJECT_ROOT / args.output_diagnostics
    )
    for path in (output_bank, output_ids, output_diagnostics):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_bank,
        source_qpos=rebuilt_qpos[selected],
        source_qvel=rebuilt_qvel[selected],
        source_target_qpos=rebuilt_target[selected],
        source_policy_action=source_policy_action,
        source_teacher_actions=source_teacher_actions,
        target_qpos=target_qpos,
        target_qvel=target_qvel,
        phase_ids=np.ones(len(selected), dtype=np.int64),
        source_row_ids=source_row_ids,
        source_global_bank_ids=original_global_ids[selected],
        middle_anchor_offsets=best_offset[selected],
        pen_axis_offsets_m=best_pen_axis_offset[selected],
    )
    local_ids = np.arange(len(selected), dtype=np.int64)
    np.save(output_ids, local_ids)
    np.savez_compressed(
        output_diagnostics,
        original_global_bank_ids=original_global_ids,
        accepted=accepted,
        best_score=best_score,
        best_middle_anchor_offset=best_offset,
        best_pen_axis_offset_m=best_pen_axis_offset,
        middle_contact_ratio=best_middle_ratio,
        triad_contact_ratio=best_triad_ratio,
        max_position_drift_mm=best_position_drift,
        max_axis_drift_deg=best_axis_drift,
        max_penetration_mm=best_penetration,
    )
    report = {
        "task": args.task,
        "split": "train",
        "input_rows": int(bank_size),
        "middle_offset_candidates_per_row": int(len(offsets)),
        "pen_axis_offsets_mm": (pen_axis_offsets * 1000.0).tolist(),
        "total_candidates_per_row": int(candidates_per_source),
        "pen_com_offset_x_m": float(cfg.pen_com_offset_x),
        "settle_steps": args.settle_steps,
        "hold_steps": args.hold_steps,
        "save_state": args.save_state,
        "accepted_rows": int(len(selected)),
        "accepted_rate": float(len(selected) / bank_size),
        "middle_contact_ratio_min": float(args.middle_contact_ratio_min),
        "terminal_linear_speed_max": float(args.terminal_linear_speed_max),
        "terminal_angular_speed_max": float(args.terminal_angular_speed_max),
        "middle_contact_ratio_mean": float(best_middle_ratio[selected].mean()) if len(selected) else None,
        "triad_contact_ratio_mean": float(best_triad_ratio[selected].mean()) if len(selected) else None,
        "source_bundle": str(source_bundle) if source_bundle else str(cfg.bundle_path),
        "source_bundle_sha256": sha256(Path(cfg.bundle_path)),
        "source_ids": str(source_ids_path) if source_ids_path else str(cfg.split_ids_path),
        "output_bank": str(output_bank),
        "output_bank_sha256": sha256(output_bank),
        "output_ids": str(output_ids),
        "output_ids_sha256": sha256(output_ids),
        "output_diagnostics": str(output_diagnostics),
        "output_diagnostics_sha256": sha256(output_diagnostics),
        "validation_or_test_accessed": False,
    }
    output_diagnostics.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_MIDDLE_REBUILT_BANK=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
