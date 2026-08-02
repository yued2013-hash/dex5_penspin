from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Screen train-only P1 resets for stable TIM hold.")
parser.add_argument("--task", default="Dex5-P1P2-RingOnly120HzH1-Direct-v0")
parser.add_argument("--source-bundle", type=Path)
parser.add_argument("--source-ids", type=Path)
parser.add_argument("--output-ids", type=Path, required=True)
parser.add_argument("--output-diagnostics", type=Path, required=True)
parser.add_argument("--output-bank", type=Path)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--replicas-per-state", type=int, default=1)
parser.add_argument("--replica-pass-ratio-min", type=float, default=1.0)
parser.add_argument("--hold-steps", type=int, default=30)
parser.add_argument("--thumb-contact-ratio-min", type=float, default=0.0)
parser.add_argument("--index-contact-ratio-min", type=float, default=0.0)
parser.add_argument("--middle-contact-ratio-min", type=float, default=0.0)
parser.add_argument("--max-position-drift-mm", type=float, default=10.0)
parser.add_argument("--max-axis-drift-deg", type=float, default=8.0)
parser.add_argument("--max-penetration-mm", type=float, default=0.6)
parser.add_argument("--terminal-linear-speed-max", type=float, default=float("inf"))
parser.add_argument("--terminal-angular-speed-max", type=float, default=float("inf"))
parser.add_argument("--terminal-window-steps", type=int, default=12)
parser.add_argument("--max-ring-force-n", type=float, default=float("inf"))
parser.add_argument("--max-pinky-force-n", type=float, default=float("inf"))
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


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if cfg.split_name != "train" or int(cfg.source_phase_id) != 1:
        raise ValueError("P1 screening requires a frozen train-only P1 task")
    cfg.seed = args.seed
    cfg.episode_length_s = max(
        float(cfg.episode_length_s), float(args.hold_steps + 4) / 120.0
    )
    source_bundle = args.source_bundle.resolve() if args.source_bundle else None
    source_ids = args.source_ids.resolve() if args.source_ids else None
    if source_bundle is not None:
        cfg.bundle_path = str(source_bundle)
        # Rebuilt bundle rows are local projections, so the original global-ID
        # scripted route map is not applicable during a zero-action hold test.
        cfg.teacher_prior_state_map_path = None
    if source_ids is not None:
        cfg.split_ids_path = str(source_ids)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    env.reset()
    bank_size = direct.bank_size
    global_ids = direct.global_bank_id_bank.detach().cpu().numpy().copy()
    local_ids_all = np.arange(bank_size, dtype=np.int64)
    fields = {
        "stable": np.zeros(bank_size, dtype=np.bool_),
        "hard_failure": np.zeros(bank_size, dtype=np.bool_),
        "drop": np.zeros(bank_size, dtype=np.bool_),
        "invalid": np.zeros(bank_size, dtype=np.bool_),
        "max_loss_streak": np.zeros(bank_size, dtype=np.int64),
        "thumb_contact_ratio": np.zeros(bank_size, dtype=np.float32),
        "index_contact_ratio": np.zeros(bank_size, dtype=np.float32),
        "middle_contact_ratio": np.zeros(bank_size, dtype=np.float32),
        "max_position_drift_mm": np.zeros(bank_size, dtype=np.float32),
        "max_axis_drift_deg": np.zeros(bank_size, dtype=np.float32),
        "max_penetration_mm": np.zeros(bank_size, dtype=np.float32),
        "replica_pass_ratio": np.zeros(bank_size, dtype=np.float32),
        "terminal_linear_speed_m_s": np.zeros(bank_size, dtype=np.float32),
        "terminal_angular_speed_rad_s": np.zeros(bank_size, dtype=np.float32),
        "max_ring_force_n": np.zeros(bank_size, dtype=np.float32),
        "max_pinky_force_n": np.zeros(bank_size, dtype=np.float32),
    }
    all_env_ids = torch.arange(args.num_envs, device=direct.device)
    actions = torch.zeros((args.num_envs, direct.cfg.action_space), device=direct.device)
    replicas = max(1, int(args.replicas_per_state))
    states_per_batch = max(1, args.num_envs // replicas)
    for start in range(0, bank_size, states_per_batch):
        local_ids = np.arange(start, min(start + states_per_batch, bank_size), dtype=np.int64)
        # Interleave replicas so every state is exercised across widely spaced
        # PhysX env slots instead of one contiguous slot block.
        expanded_ids = np.tile(local_ids, replicas)
        padded = np.pad(expanded_ids, (0, args.num_envs - len(expanded_ids)), mode="edge")
        direct.forced_bank_ids = torch.as_tensor(padded, device=direct.device)
        direct._reset_idx(all_env_ids)
        direct._compute_intermediate_values()
        initial_pen_pos = direct.pen_pos.clone()
        initial_pen_axis = direct.pen_axis.clone()
        loss_streak = torch.zeros(args.num_envs, dtype=torch.long, device=direct.device)
        max_loss_streak = torch.zeros_like(loss_streak)
        contact_steps = torch.zeros((args.num_envs, 3), device=direct.device)
        hard_failure = torch.zeros(args.num_envs, dtype=torch.bool, device=direct.device)
        drop = torch.zeros_like(hard_failure)
        invalid = torch.zeros_like(hard_failure)
        max_position_drift = torch.zeros(args.num_envs, device=direct.device)
        max_axis_drift = torch.zeros(args.num_envs, device=direct.device)
        max_penetration = torch.zeros(args.num_envs, device=direct.device)
        max_ring_force = torch.zeros(args.num_envs, device=direct.device)
        max_pinky_force = torch.zeros(args.num_envs, device=direct.device)
        linear_speed_history = []
        angular_speed_history = []
        terminal_position_history = []
        terminal_axis_history = []
        for _ in range(args.hold_steps):
            _, _, terminated, truncated, _ = env.step(actions)
            direct._compute_intermediate_values()
            contacts = direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n
            triad = contacts.all(dim=-1)
            contact_steps += contacts.float()
            loss_streak = torch.where(triad, torch.zeros_like(loss_streak), loss_streak + 1)
            max_loss_streak = torch.maximum(max_loss_streak, loss_streak)
            position_drift = torch.linalg.norm(direct.pen_pos - initial_pen_pos, dim=-1)
            axis_dot = torch.sum(direct.pen_axis * initial_pen_axis, dim=-1).abs().clamp(-1.0, 1.0)
            axis_drift = torch.acos(axis_dot)
            penetration = direct.episode_max_penetration
            max_ring_force = torch.maximum(max_ring_force, direct.contact_force_norm[:, 3])
            max_pinky_force = torch.maximum(max_pinky_force, direct.contact_force_norm[:, 4])
            linear_speed_history.append(direct.pen_linvel.clone())
            angular_perp = direct.pen_angvel - (
                torch.sum(direct.pen_angvel * direct.pen_axis, dim=-1, keepdim=True)
                * direct.pen_axis
            )
            angular_speed_history.append(angular_perp)
            terminal_position_history.append(direct.pen_pos.clone())
            terminal_axis_history.append(direct.pen_axis.clone())
            finite = torch.isfinite(direct.hand_qpos).all(dim=-1) & torch.isfinite(direct.pen_pos).all(dim=-1)
            height_drop = initial_pen_pos[:, 2] - direct.pen_pos[:, 2]
            current_drop = height_drop > direct.cfg.drop_threshold
            current_invalid = ~finite
            hard_failure |= (
                (loss_streak > 1)
                | (position_drift > float(args.max_position_drift_mm) / 1000.0)
                | (axis_drift > torch.deg2rad(torch.full_like(axis_drift, float(args.max_axis_drift_deg))))
                | (penetration > float(args.max_penetration_mm) / 1000.0)
                | (max_ring_force > float(args.max_ring_force_n))
                | (max_pinky_force > float(args.max_pinky_force_n))
                | current_drop
                | current_invalid
                | terminated
                | truncated
            )
            drop |= current_drop
            invalid |= current_invalid
            max_position_drift = torch.maximum(max_position_drift, position_drift)
            max_axis_drift = torch.maximum(max_axis_drift, axis_drift)
            max_penetration = torch.maximum(max_penetration, penetration)
        count = len(local_ids)
        expanded_count = count * replicas
        rows = slice(start, start + count)
        hard_np = hard_failure[:expanded_count].cpu().numpy().reshape(replicas, count).T
        drop_np = drop[:expanded_count].cpu().numpy().reshape(replicas, count).T
        invalid_np = invalid[:expanded_count].cpu().numpy().reshape(replicas, count).T
        loss_np = max_loss_streak[:expanded_count].cpu().numpy().reshape(replicas, count).T
        ratios = (
            contact_steps[:expanded_count] / float(args.hold_steps)
        ).cpu().numpy().reshape(replicas, count, 3).transpose(1, 0, 2)
        replica_pass = (
            ~hard_np
            & (ratios[:, :, 0] >= args.thumb_contact_ratio_min)
            & (ratios[:, :, 1] >= args.index_contact_ratio_min)
            & (ratios[:, :, 2] >= args.middle_contact_ratio_min)
        )
        terminal_window = max(1, min(int(args.terminal_window_steps), len(linear_speed_history)))
        terminal_duration = max(1, terminal_window - 1) / 120.0
        terminal_linear_speed = torch.linalg.norm(
            terminal_position_history[-1] - terminal_position_history[-terminal_window], dim=-1
        ) / terminal_duration
        terminal_axis_dot = torch.sum(
            terminal_axis_history[-1] * terminal_axis_history[-terminal_window], dim=-1
        ).abs().clamp(-1.0, 1.0)
        terminal_angular_speed = torch.acos(terminal_axis_dot) / terminal_duration
        terminal_linear_np = terminal_linear_speed[:expanded_count].cpu().numpy().reshape(replicas, count).T
        terminal_angular_np = terminal_angular_speed[:expanded_count].cpu().numpy().reshape(replicas, count).T
        replica_pass &= (
            (terminal_linear_np <= float(args.terminal_linear_speed_max))
            & (terminal_angular_np <= float(args.terminal_angular_speed_max))
        )
        pass_ratio = replica_pass.mean(axis=1)
        fields["stable"][rows] = pass_ratio >= args.replica_pass_ratio_min
        fields["hard_failure"][rows] = hard_np.any(axis=1)
        fields["drop"][rows] = drop_np.any(axis=1)
        fields["invalid"][rows] = invalid_np.any(axis=1)
        fields["max_loss_streak"][rows] = loss_np.max(axis=1)
        fields["replica_pass_ratio"][rows] = pass_ratio
        mean_ratios = ratios.mean(axis=1)
        fields["thumb_contact_ratio"][rows] = mean_ratios[:, 0]
        fields["index_contact_ratio"][rows] = mean_ratios[:, 1]
        fields["middle_contact_ratio"][rows] = mean_ratios[:, 2]
        position_np = (max_position_drift[:expanded_count] * 1000.0).cpu().numpy().reshape(replicas, count).T
        axis_np = torch.rad2deg(max_axis_drift[:expanded_count]).cpu().numpy().reshape(replicas, count).T
        penetration_np = (max_penetration[:expanded_count] * 1000.0).cpu().numpy().reshape(replicas, count).T
        fields["max_position_drift_mm"][rows] = position_np.max(axis=1)
        fields["max_axis_drift_deg"][rows] = axis_np.max(axis=1)
        fields["max_penetration_mm"][rows] = penetration_np.max(axis=1)
        fields["terminal_linear_speed_m_s"][rows] = terminal_linear_np.max(axis=1)
        fields["terminal_angular_speed_rad_s"][rows] = terminal_angular_np.max(axis=1)
        fields["max_ring_force_n"][rows] = max_ring_force[:expanded_count].cpu().numpy().reshape(replicas, count).T.max(axis=1)
        fields["max_pinky_force_n"][rows] = max_pinky_force[:expanded_count].cpu().numpy().reshape(replicas, count).T.max(axis=1)
    del direct.forced_bank_ids
    # A split file indexes rows in its bundle. Preserve legacy global IDs only
    # when screening the original bundle; rebuilt bundles need local row IDs.
    stable_ids = (
        local_ids_all[fields["stable"]]
        if source_bundle is not None
        else global_ids[fields["stable"]]
    )
    output_ids = args.output_ids if args.output_ids.is_absolute() else PROJECT_ROOT / args.output_ids
    output_diagnostics = (
        args.output_diagnostics
        if args.output_diagnostics.is_absolute()
        else PROJECT_ROOT / args.output_diagnostics
    )
    output_ids.parent.mkdir(parents=True, exist_ok=True)
    output_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    saved_ids = (
        np.arange(len(stable_ids), dtype=np.int64)
        if args.output_bank is not None
        else stable_ids.astype(np.int64)
    )
    np.save(output_ids, saved_ids)
    np.savez_compressed(output_diagnostics, global_bank_ids=global_ids, **fields)
    output_bank = None
    if args.output_bank is not None:
        if source_bundle is None:
            raise ValueError("--output-bank requires --source-bundle so IDs are bundle-local")
        output_bank = (
            args.output_bank if args.output_bank.is_absolute()
            else PROJECT_ROOT / args.output_bank
        )
        output_bank.parent.mkdir(parents=True, exist_ok=True)
        stable_local_ids = local_ids_all[fields["stable"]]
        with np.load(source_bundle, allow_pickle=False) as source:
            row_count = int(np.asarray(source["phase_ids"]).shape[0])
            payload = {
                key: (
                    np.asarray(source[key])[stable_local_ids]
                    if np.asarray(source[key]).ndim > 0
                    and np.asarray(source[key]).shape[0] == row_count
                    else np.asarray(source[key])
                )
                for key in source.files
            }
        np.savez_compressed(output_bank, **payload)
    report = {
        "task": args.task,
        "split": "train",
        "source_phase_id": 1,
        "hold_steps": args.hold_steps,
        "control_frequency_hz": 120.0,
        "replicas_per_state": replicas,
        "replica_pass_ratio_min": args.replica_pass_ratio_min,
        "thumb_contact_ratio_min": args.thumb_contact_ratio_min,
        "index_contact_ratio_min": args.index_contact_ratio_min,
        "middle_contact_ratio_min": args.middle_contact_ratio_min,
        "max_position_drift_mm": args.max_position_drift_mm,
        "max_axis_drift_deg": args.max_axis_drift_deg,
        "max_penetration_mm": args.max_penetration_mm,
        "terminal_linear_speed_max": args.terminal_linear_speed_max,
        "terminal_angular_speed_max": args.terminal_angular_speed_max,
        "terminal_window_steps": args.terminal_window_steps,
        "max_ring_force_n": args.max_ring_force_n,
        "max_pinky_force_n": args.max_pinky_force_n,
        "pen_com_offset_x_m": float(cfg.pen_com_offset_x),
        "screened_rows": int(bank_size),
        "stable_rows": int(len(stable_ids)),
        "stable_rate": float(len(stable_ids) / bank_size),
        "drop_rows": int(fields["drop"].sum()),
        "invalid_rows": int(fields["invalid"].sum()),
        "stable_middle_contact_ratio_mean": (
            float(fields["middle_contact_ratio"][fields["stable"]].mean())
            if len(stable_ids) else None
        ),
        "source_bundle": str(source_bundle) if source_bundle else str(cfg.bundle_path),
        "source_bundle_sha256": sha256(Path(cfg.bundle_path)),
        "source_ids": str(source_ids) if source_ids else str(cfg.split_ids_path),
        "stable_global_bank_ids": global_ids[fields["stable"]].astype(np.int64).tolist(),
        "output_id_space": (
            "output_bank_local" if args.output_bank is not None
            else ("bundle_local" if source_bundle is not None else "bundle_global")
        ),
        "output_ids": str(output_ids),
        "output_ids_sha256": sha256(output_ids),
        "output_diagnostics": str(output_diagnostics),
        "output_diagnostics_sha256": sha256(output_diagnostics),
        "output_bank": str(output_bank) if output_bank else None,
        "output_bank_sha256": sha256(output_bank) if output_bank else None,
        "validation_or_test_accessed": False,
    }
    report_path = output_diagnostics.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_STABLE_SCREEN=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
