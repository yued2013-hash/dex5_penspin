from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESET_DIR = PROJECT_ROOT / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank"
FROZEN_BANK = RESET_DIR / "P1_P2_reset_bank_phase1最终validated自复验_validated.npz"
TRAIN_IDS = RESET_DIR / "P1_P2_train_ids.npy"
RING_GEOMETRY = RESET_DIR / "P1_P2_train_ring_geometry.json"


parser = argparse.ArgumentParser(
    description="Collect train-only P1-P2 reverse-curriculum states from physical P2 rollouts."
)
parser.add_argument("--task", default="Dex5-P1P2-ReverseP2Source-Direct-v0")
parser.add_argument("--output-dir", type=Path, default=RESET_DIR / "reverse_curriculum")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--per-level", type=int, default=128)
parser.add_argument("--max-steps", type=int, default=4000)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--levels", nargs="+", default=("R0", "R1", "R2", "R3"))
parser.add_argument("--progress-every", type=int, default=100)
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


LEVEL_BINS_MM = {
    "R0": (0.0, 2.0),
    "R1": (2.0, 4.0),
    "R2": (4.0, 8.0),
    "R3": (8.0, 12.0),
}
LEVEL_IDS = {"R0": 0, "R1": 1, "R2": 2, "R3": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ring_axis_feature(env) -> torch.Tensor:
    axis = env.pen_axis
    world_up = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand_as(axis)
    up = world_up - torch.sum(world_up * axis, dim=-1, keepdim=True) * axis
    up = torch.nn.functional.normalize(up, dim=-1)
    lateral = torch.nn.functional.normalize(torch.cross(axis, up, dim=-1), dim=-1)
    ring_rel = env.fingertip_pos_local[:, 3] - env.pen_pos
    return torch.stack(
        (
            torch.sum(ring_rel * axis, dim=-1),
            torch.sum(ring_rel * lateral, dim=-1),
            torch.sum(ring_rel * up, dim=-1),
        ),
        dim=-1,
    )


def append_rows(rows: dict[str, list[np.ndarray]], values: dict[str, torch.Tensor], ids: torch.Tensor) -> None:
    for key, tensor in values.items():
        rows[key].append(tensor[ids].detach().cpu().numpy().copy())


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    requested_levels = tuple(args.levels)
    unknown = sorted(set(requested_levels) - set(LEVEL_BINS_MM))
    if unknown:
        raise ValueError(f"Unknown reverse levels: {unknown}")

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(FROZEN_BANK) as bank:
        phase_ids = np.asarray(bank["phase_ids"], dtype=np.int64)
        source_qpos_all = np.asarray(bank["source_qpos"], dtype=np.float32)
        source_qvel_all = np.asarray(bank["source_qvel"], dtype=np.float32)
        source_target_all = np.asarray(bank["source_target_qpos"], dtype=np.float32)
        target_qpos_all = np.asarray(bank["target_qpos"], dtype=np.float32)
        train_ids = np.asarray(np.load(TRAIN_IDS, allow_pickle=False), dtype=np.int64)
        train_p1_ids = train_ids[phase_ids[train_ids] == 1]
        train_p2_ids = train_ids[phase_ids[train_ids] == 2]
        p1_target_median = np.median(source_target_all[train_p1_ids], axis=0).astype(np.float32)
        p2_target_median = np.median(source_target_all[train_p2_ids], axis=0).astype(np.float32)

    if train_p2_ids.size == 0:
        raise RuntimeError("No train P2 ids available for reverse curriculum generation")

    geometry = json.loads(RING_GEOMETRY.read_text(encoding="utf-8"))
    p2_ring_center_np = np.asarray(geometry["P2"]["median_m"], dtype=np.float32)

    # R0 is the original, frozen train P2 distribution. Keep it as a separate copy
    # so later training can mix reverse levels without reading validation/test rows.
    manifest: dict[str, object] = {
        "source": "P1_P2 reverse curriculum from train-only validated P2",
        "frozen_bank": str(FROZEN_BANK),
        "frozen_bank_sha256": sha256(FROZEN_BANK),
        "train_ids": str(TRAIN_IDS),
        "train_ids_sha256": sha256(TRAIN_IDS),
        "ring_geometry": str(RING_GEOMETRY),
        "levels": {},
        "seed": args.seed,
        "num_envs": args.num_envs,
        "per_level_requested": args.per_level,
        "leak_policy": "Only phase=2 IDs from P1_P2_train_ids.npy are used.",
    }
    if "R0" in requested_levels:
        count = min(args.per_level, train_p2_ids.size)
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(train_p2_ids, size=count, replace=train_p2_ids.size < count)
        r0_payload = {
            "source_qpos": source_qpos_all[selected].astype(np.float32),
            "source_qvel": source_qvel_all[selected].astype(np.float32),
            "target_qpos": target_qpos_all[selected].astype(np.float32),
            "source_target_qpos": source_target_all[selected].astype(np.float32),
            "source_policy_action": np.zeros((count, 20), dtype=np.float32),
            "source_teacher_actions": np.zeros((count, 1, 20), dtype=np.float32),
            "phase_ids": np.full(count, 2, dtype=np.int64),
            "reverse_level_ids": np.full(count, LEVEL_IDS["R0"], dtype=np.int64),
            "source_global_bank_ids": selected.astype(np.int64),
            "ring_target_distance_mm": np.zeros(count, dtype=np.float32),
            "max_penetration_mm": np.zeros(count, dtype=np.float32),
            "contact_force_thumb_index_middle_ring_pinky": np.zeros((count, 5), dtype=np.float32),
        }
        r0_path = output_dir / "P1_P2_reverse_R0_P2_hold_train.npz"
        np.savez_compressed(r0_path, **r0_payload)
        manifest["levels"]["R0"] = {
            "path": str(r0_path),
            "count": count,
            "bin_mm": LEVEL_BINS_MM["R0"],
            "source_global_ids_unique": int(np.unique(selected).size),
            "sha256": sha256(r0_path),
        }
        print(
            f"[reverse] wrote R0 count={count} path={r0_path}",
            flush=True,
        )

    rollout_levels = [level for level in requested_levels if level != "R0"]
    if rollout_levels:
        print(
            f"[reverse] starting rollout levels={rollout_levels} "
            f"num_envs={args.num_envs} per_level={args.per_level} max_steps={args.max_steps}",
            flush=True,
        )
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.action_scale = 0.10
        cfg.episode_length_s = 0.80
        cfg.reset_joint_pos_noise = 0.0
        cfg.reset_joint_vel_noise = 0.0
        cfg.reset_pen_pos_noise = 0.0
        cfg.reset_pen_vel_noise = 0.0
        cfg.penetration_threshold = 0.00060
        env = gym.make(args.task, cfg=cfg)
        direct = env.unwrapped

        train_p2_local = torch.arange(direct.bank_size, dtype=torch.long, device=direct.device)

        p2_ring_center = torch.tensor(p2_ring_center_np, dtype=torch.float32, device=direct.device)
        p1_target = torch.tensor(p1_target_median, dtype=torch.float32, device=direct.device)
        p2_target = torch.tensor(p2_target_median, dtype=torch.float32, device=direct.device)
        away_ring_target = p2_target[8:12] + 0.45 * (p1_target[8:12] - p2_target[8:12])
        away_ring_target = torch.maximum(
            torch.minimum(away_ring_target, direct.joint_upper[0, 8:12]),
            direct.joint_lower[0, 8:12],
        )
        support_target = p2_target.clone()
        support_target[0:8] = p2_target[0:8]
        support_target[16:20] = p2_target[16:20]

        rows = {
            "source_qpos": [],
            "source_qvel": [],
            "target_qpos": [],
            "source_target_qpos": [],
            "source_policy_action": [],
            "source_teacher_actions": [],
            "phase_ids": [],
            "reverse_level_ids": [],
            "source_global_bank_ids": [],
            "ring_target_distance_mm": [],
            "ring_progress_from_p2_to_p1": [],
            "max_penetration_mm": [],
            "contact_force_thumb_index_middle_ring_pinky": [],
        }
        accepted = {level: 0 for level in rollout_levels}
        seen_global_ids: set[int] = set()
        env.reset()
        reset_perm = torch.randint(
            0, train_p2_local.numel(), (args.num_envs,), device=direct.device
        )
        direct.forced_bank_ids = train_p2_local[reset_perm]
        env.reset()

        step = 0
        while step < args.max_steps and any(accepted[level] < args.per_level for level in rollout_levels):
            increment = 0.25 * float(cfg.action_scale)
            actions = torch.zeros((args.num_envs, direct.cfg.action_space), device=direct.device)
            ring_error = away_ring_target.unsqueeze(0) - direct.processed_targets[:, 8:12]
            actions[:, 8:12] = torch.clamp(ring_error / increment, -1.0, 1.0)
            support_error = support_target.unsqueeze(0) - direct.hand_qpos
            for joint_slice in (slice(0, 8), slice(16, 20)):
                actions[:, joint_slice] = torch.clamp(
                    support_error[:, joint_slice] / float(cfg.action_scale),
                    -0.35,
                    0.35,
                )
            _, _, terminated, truncated, _ = env.step(actions)
            direct._compute_intermediate_values()

            feature = ring_axis_feature(direct)
            distance_mm = torch.linalg.norm(feature - p2_ring_center.unsqueeze(0), dim=-1) * 1000.0
            p2_to_p1 = torch.tensor(
                np.asarray(geometry["P1"]["median_m"], dtype=np.float32) - p2_ring_center_np,
                dtype=torch.float32,
                device=direct.device,
            )
            progress_from_p2 = torch.sum(
                (feature - p2_ring_center.unsqueeze(0)) * p2_to_p1.unsqueeze(0),
                dim=-1,
            ) / torch.sum(p2_to_p1.square()).clamp_min(1.0e-9)
            height_drop = direct.init_pen_pos[:, 2] - direct.pen_pos[:, 2]
            finite = torch.isfinite(direct.pen_pos).all(dim=-1) & torch.isfinite(direct.hand_qpos).all(dim=-1)
            contact = direct.contact_force_norm
            primary = (contact[:, :3] > 0.03).all(dim=-1)
            force_sum = contact.sum(dim=-1).clamp_min(1.0e-6)
            ring_share = contact[:, 3] / force_sum
            valid = (
                finite
                & primary
                & (height_drop <= 0.020)
                & (direct.episode_max_penetration <= 0.00060)
                & (contact[:, 3] <= 0.60)
                & (ring_share <= 0.25)
                & (contact[:, 4] <= 0.03)
            )
            pen_pose_local = direct.pen.data.root_pose_w.clone()
            pen_pose_local[:, :3] -= direct.scene.env_origins
            qpos = torch.cat((direct.hand_qpos, pen_pose_local), dim=-1)
            qvel = torch.cat((direct.hand_qvel, direct.pen_linvel, direct.pen_angvel), dim=-1)
            target_qpos = torch.cat(
                (
                    p2_target[:20].unsqueeze(0).expand(args.num_envs, -1),
                    direct.target_pen_pos,
                    direct.target_pen_quat,
                ),
                dim=-1,
            )
            values = {
                "source_qpos": qpos,
                "source_qvel": qvel,
                "target_qpos": target_qpos,
                "source_target_qpos": direct.processed_targets,
                "source_policy_action": actions,
                "source_teacher_actions": actions.unsqueeze(1),
                "phase_ids": torch.full((args.num_envs,), 2, dtype=torch.long, device=direct.device),
                "source_global_bank_ids": direct.global_bank_id_bank[direct.current_bank_ids],
                "ring_target_distance_mm": distance_mm,
                "ring_progress_from_p2_to_p1": progress_from_p2,
                "max_penetration_mm": direct.episode_max_penetration * 1000.0,
                "contact_force_thumb_index_middle_ring_pinky": contact,
            }
            values["source_global_bank_ids"] = torch.as_tensor(
                values["source_global_bank_ids"], dtype=torch.long, device=direct.device
            )

            used = torch.zeros(args.num_envs, dtype=torch.bool, device=direct.device)
            for level in rollout_levels:
                if accepted[level] >= args.per_level:
                    continue
                lo, hi = LEVEL_BINS_MM[level]
                candidates = torch.nonzero(
                    valid & ~used & (distance_mm >= lo) & (distance_mm < hi),
                    as_tuple=False,
                ).squeeze(-1)
                if candidates.numel() == 0:
                    continue
                take = candidates[: args.per_level - accepted[level]]
                values["reverse_level_ids"] = torch.full(
                    (args.num_envs,), LEVEL_IDS[level], dtype=torch.long, device=direct.device
                )
                append_rows(rows, values, take)
                accepted[level] += int(take.numel())
                used[take] = True
                seen_global_ids.update(values["source_global_bank_ids"][take].detach().cpu().tolist())

            done = terminated | truncated | (distance_mm > 14.0)
            if torch.any(done):
                reset_perm = torch.randint(
                    0, train_p2_local.numel(), (args.num_envs,), device=direct.device
                )
                direct.forced_bank_ids = train_p2_local[reset_perm]
            step += 1
            if args.progress_every > 0 and step % args.progress_every == 0:
                print(
                    "[reverse] step="
                    f"{step} accepted="
                    + ",".join(f"{level}:{accepted[level]}" for level in rollout_levels),
                    flush=True,
                )

        env.close()
        print(
            "[reverse] rollout finished step="
            f"{step} accepted="
            + ",".join(f"{level}:{accepted[level]}" for level in rollout_levels),
            flush=True,
        )

        if rows["source_qpos"]:
            payload = {
                key: np.concatenate(parts, axis=0)
                for key, parts in rows.items()
            }
        else:
            payload = {
                "source_qpos": np.zeros((0, 27), dtype=np.float32),
                "source_qvel": np.zeros((0, 26), dtype=np.float32),
                "target_qpos": np.zeros((0, 27), dtype=np.float32),
                "source_target_qpos": np.zeros((0, 20), dtype=np.float32),
                "source_policy_action": np.zeros((0, 20), dtype=np.float32),
                "source_teacher_actions": np.zeros((0, 1, 20), dtype=np.float32),
                "phase_ids": np.zeros((0,), dtype=np.int64),
                "reverse_level_ids": np.zeros((0,), dtype=np.int64),
                "source_global_bank_ids": np.zeros((0,), dtype=np.int64),
                "ring_target_distance_mm": np.zeros((0,), dtype=np.float32),
                "ring_progress_from_p2_to_p1": np.zeros((0,), dtype=np.float32),
                "max_penetration_mm": np.zeros((0,), dtype=np.float32),
                "contact_force_thumb_index_middle_ring_pinky": np.zeros((0, 5), dtype=np.float32),
            }

        for level in rollout_levels:
            mask = payload["reverse_level_ids"] == LEVEL_IDS[level]
            level_payload = {key: value[mask] for key, value in payload.items()}
            level_path = output_dir / f"P1_P2_reverse_{level}_{LEVEL_BINS_MM[level][0]:.0f}_{LEVEL_BINS_MM[level][1]:.0f}mm_train.npz"
            np.savez_compressed(level_path, **level_payload)
            distances = level_payload["ring_target_distance_mm"]
            manifest["levels"][level] = {
                "path": str(level_path),
                "count": int(mask.sum()),
                "bin_mm": LEVEL_BINS_MM[level],
                "distance_mm_min": float(distances.min()) if distances.size else None,
                "distance_mm_mean": float(distances.mean()) if distances.size else None,
                "distance_mm_max": float(distances.max()) if distances.size else None,
                "source_global_ids_unique": int(np.unique(level_payload["source_global_bank_ids"]).size)
                if distances.size
                else 0,
                "sha256": sha256(level_path),
            }
        manifest["rollout_steps"] = step
        manifest["rollout_seen_train_p2_global_ids"] = len(seen_global_ids)

    manifest_path = output_dir / "P1_P2_reverse_curriculum_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("P1_P2_REVERSE_CURRICULUM=" + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
