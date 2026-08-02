from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = (
    PROJECT_ROOT
    / "reports/manus_mujoco_presets/canonical_p3_manus_constrained_m3_2_40/"
    "canonical_P3_candidate_seed.json"
)
DEFAULT_CANDIDATES = (
    PROJECT_ROOT / "reports/m3_2_30_index_near_surface/canonical_p3_physx_projection_candidates.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports/m3_2_30_index_near_surface/canonical_p3_physx_projection_search.json"
)
DEFAULT_BUNDLE = (
    PROJECT_ROOT / "reports/m3_2_30_index_near_surface/isaac_canonical_p3_physx_projected_bundle.npz"
)


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_candidates(
    seed_path: Path,
    output_path: Path,
    count: int,
    seed: int,
    center_bundle_path: Path | None = None,
    center_candidate_ids: list[int] | None = None,
    manifest_path: Path | None = None,
    manifest_phase: int = 3,
    override_center_finger_ids: list[int] | None = None,
    hand_noise_std: float = 0.012,
    hand_noise_clip: float = 0.030,
    pen_pos_noise_std: float = 0.0006,
    pen_pos_noise_clip: float = 0.0015,
    source_target_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = next(row for row in manifest["phases"] if int(row["phase"]) == manifest_phase)
        manus_qpos = np.asarray(record["hand_qpos"], dtype=np.float32)
    else:
        record = json.loads(seed_path.read_text(encoding="utf-8"))
        manus_path = PROJECT_ROOT.parent / record["manus_reference"]
        manus_record = json.loads(manus_path.read_text(encoding="utf-8"))
        manus_qpos = np.asarray(manus_record["hand_qpos"], dtype=np.float32)
    hand_qpos = np.asarray(record["hand_qpos"], dtype=np.float32)
    pen_pose = np.concatenate(
        (
            np.asarray(record["pen_pos"], dtype=np.float32),
            np.asarray(record["pen_quat"], dtype=np.float32),
        )
    )

    rng = np.random.default_rng(seed)
    if center_bundle_path is not None:
        with np.load(center_bundle_path) as center_data:
            all_center_qpos = np.asarray(center_data["source_qpos"], dtype=np.float32)
            if center_candidate_ids:
                center_qpos = all_center_qpos[center_candidate_ids]
            else:
                center_qpos = all_center_qpos
        selected_centers = center_qpos[np.arange(count) % len(center_qpos)]
        hand_noise = np.clip(
            rng.normal(0.0, hand_noise_std, size=(count, 20)),
            -hand_noise_clip,
            hand_noise_clip,
        ).astype(np.float32)
        # Continue opening ring/pinky for half the local samples, while keeping
        # unbiased probes around centers that already removed those contacts.
        opening = rng.uniform(0.0, hand_noise_clip, size=(count, 6)).astype(np.float32)
        opening[np.arange(count) % 4 == 0] *= -1.0
        hand_noise[:, [9, 10, 11, 13, 14, 15]] -= opening
        hand_candidates = selected_centers[:, :20] + hand_noise
        for finger_id in override_center_finger_ids or ():
            start = int(finger_id) * 4
            hand_candidates[:, start : start + 4] = (
                hand_qpos[None, start : start + 4] + hand_noise[:, start : start + 4]
            )
        pen_candidates = selected_centers[:, 20:27].copy()
        pen_candidates[:, :3] += np.clip(
            rng.normal(0.0, pen_pos_noise_std, size=(count, 3)),
            -pen_pos_noise_clip,
            pen_pos_noise_clip,
        ).astype(np.float32)
        for index, center in enumerate(center_qpos):
            if index >= count:
                break
            hand_candidates[index] = center[:20]
            for finger_id in override_center_finger_ids or ():
                start = int(finger_id) * 4
                hand_candidates[index, start : start + 4] = hand_qpos[start : start + 4]
            pen_candidates[index] = center[20:27]
    else:
        hand_noise = np.clip(
            rng.normal(0.0, hand_noise_std, size=(count, 20)),
            -hand_noise_clip,
            hand_noise_clip,
        ).astype(np.float32)
        # Ring and pinky should leave the P3 load path. Bias their flexion joints open,
        # while retaining symmetric samples so the search can correct a wrong sign.
        opening = rng.uniform(0.0, hand_noise_clip, size=(count, 6)).astype(np.float32)
        opening[: count // 5] *= -1.0
        hand_noise[:, [9, 10, 11, 13, 14, 15]] -= opening
        hand_candidates = hand_qpos[None, :] + hand_noise

        pen_offset = np.clip(
            rng.normal(0.0, pen_pos_noise_std, size=(count, 3)),
            -pen_pos_noise_clip,
            pen_pos_noise_clip,
        ).astype(np.float32)
        pen_candidates = np.repeat(pen_pose[None, :], count, axis=0)
        pen_candidates[:, :3] += pen_offset

        hand_candidates[0] = hand_qpos
        pen_candidates[0] = pen_pose
        # Structured translation probes make every Cartesian sign available even if
        # the random tail happens to be unhelpful.
        probe = 1
        for axis in range(3):
            for magnitude in (-0.004, -0.002, 0.002, 0.004):
                if probe >= count:
                    break
                hand_candidates[probe] = hand_qpos
                pen_candidates[probe] = pen_pose
                pen_candidates[probe, axis] += magnitude
                probe += 1

    initial_rms = np.sqrt(np.mean((hand_candidates - manus_qpos[None, :]) ** 2, axis=1))
    source_qpos = np.concatenate((hand_candidates, pen_candidates), axis=1).astype(np.float32)
    zeros_qvel = np.zeros((count, 26), dtype=np.float32)
    source_target_qpos = hand_candidates.astype(np.float32)
    target_qpos = source_qpos.copy()
    if source_target_offset is not None:
        source_target_qpos = (source_target_qpos + source_target_offset[None, :]).astype(np.float32)
        target_qpos[:, :20] = source_target_qpos
    bundle = {
        "source_qpos": source_qpos,
        "source_qvel": zeros_qvel,
        "source_target_qpos": source_target_qpos,
        "source_policy_action": np.zeros((count, 20), dtype=np.float32),
        "source_teacher_actions": np.zeros((count, 1, 20), dtype=np.float32),
        "target_qpos": target_qpos,
        "target_qvel": zeros_qvel.copy(),
        "phase_ids": np.full(count, manifest_phase, dtype=np.int64),
        "candidate_initial_manus_rms_rad": initial_rms.astype(np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **bundle)
    initial_pen_offset = np.linalg.norm(source_qpos[:, 20:23] - pen_pose[None, :3], axis=1)
    return manus_qpos, initial_rms, initial_pen_offset.astype(np.float32)


parser = argparse.ArgumentParser(description="Project canonical P3 into a non-penetrating PhysX reset.")
parser.add_argument("--task", default="Dex5-P3Hold-Direct-v0")
parser.add_argument("--seed_path", type=Path, default=DEFAULT_SEED)
parser.add_argument("--manifest_path", type=Path, default=None)
parser.add_argument("--manifest_phase", type=int, default=3)
parser.add_argument("--candidate_bundle", type=Path, default=DEFAULT_CANDIDATES)
parser.add_argument("--center_bundle", type=Path, default=None)
parser.add_argument("--center_candidate_ids", type=int, nargs="+", default=None)
parser.add_argument("--override_center_finger_ids", type=int, nargs="*", default=None)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--accepted_bundle", type=Path, default=DEFAULT_BUNDLE)
parser.add_argument("--candidates", type=int, default=1024)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--seed", type=int, default=20260714)
parser.add_argument("--max_penetration_mm", type=float, default=0.5)
parser.add_argument("--max_drift_mm", type=float, default=15.0)
parser.add_argument("--max_hand_rms_rad", type=float, default=0.20)
parser.add_argument("--max_initial_pen_offset_mm", type=float, default=2.0)
parser.add_argument("--max_axis_drift_deg", type=float, default=180.0)
parser.add_argument("--max_angular_speed_rad_s", type=float, default=100.0)
parser.add_argument("--hand_noise_std", type=float, default=0.012)
parser.add_argument("--hand_noise_clip", type=float, default=0.030)
parser.add_argument("--pen_pos_noise_std", type=float, default=0.0006)
parser.add_argument("--pen_pos_noise_clip", type=float, default=0.0015)
parser.add_argument("--source_target_offset", type=float, nargs=20, default=None)
parser.add_argument("--min_triad_contact_ratio", type=float, default=1.0)
parser.add_argument("--required_contact_ids", type=int, nargs="+", default=(0, 1, 2))
parser.add_argument("--min_contact_ratio_by_id", type=float, nargs="*", default=None)
parser.add_argument("--min_proxy_contact_ratio_by_id", type=float, nargs="*", default=None)
parser.add_argument("--nonprimary_contact_ids", type=int, nargs="*", default=(3, 4))
parser.add_argument("--max_nonprimary_force_share", type=float, default=0.15)
parser.add_argument("--max_nonprimary_contact_steps", type=float, default=3.0)
parser.add_argument("--max_force_share_by_id", type=float, nargs="*", default=None)
parser.add_argument("--collision_proxy_scales", type=float, nargs=5, default=None)
parser.add_argument("--fixed_first_action", type=float, nargs=20, default=None)
parser.add_argument("--fixed_second_action", type=float, nargs=20, default=None)
parser.add_argument("--switch_step", type=int, default=2)
parser.add_argument("--teacher_prefill_steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

seed_path = project_path(args.seed_path)
manifest_path = project_path(args.manifest_path) if args.manifest_path is not None else None
candidate_bundle = project_path(args.candidate_bundle)
center_bundle = project_path(args.center_bundle) if args.center_bundle is not None else None
output_path = project_path(args.output)
accepted_bundle_path = project_path(args.accepted_bundle)
source_target_offset_np = (
    np.asarray(args.source_target_offset, dtype=np.float32)
    if args.source_target_offset is not None
    else None
)
if args.min_contact_ratio_by_id is not None and len(args.min_contact_ratio_by_id) % 2 != 0:
    raise ValueError("--min_contact_ratio_by_id expects id ratio pairs")
if args.min_proxy_contact_ratio_by_id is not None and len(args.min_proxy_contact_ratio_by_id) % 2 != 0:
    raise ValueError("--min_proxy_contact_ratio_by_id expects id ratio pairs")
if args.max_force_share_by_id is not None and len(args.max_force_share_by_id) % 2 != 0:
    raise ValueError("--max_force_share_by_id expects id ratio pairs")
min_contact_ratio_by_id = {
    int(args.min_contact_ratio_by_id[index]): float(args.min_contact_ratio_by_id[index + 1])
    for index in range(0, len(args.min_contact_ratio_by_id or ()), 2)
}
min_proxy_contact_ratio_by_id = {
    int(args.min_proxy_contact_ratio_by_id[index]): float(args.min_proxy_contact_ratio_by_id[index + 1])
    for index in range(0, len(args.min_proxy_contact_ratio_by_id or ()), 2)
}
max_force_share_by_id = {
    int(args.max_force_share_by_id[index]): float(args.max_force_share_by_id[index + 1])
    for index in range(0, len(args.max_force_share_by_id or ()), 2)
}
manus_qpos_np, candidate_initial_rms_np, candidate_initial_pen_offset_np = build_candidates(
    seed_path,
    candidate_bundle,
    args.candidates,
    args.seed,
    center_bundle,
    args.center_candidate_ids,
    manifest_path,
    args.manifest_phase,
    args.override_center_finger_ids,
    args.hand_noise_std,
    args.hand_noise_clip,
    args.pen_pos_noise_std,
    args.pen_pos_noise_clip,
    source_target_offset_np,
)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.bundle_path = str(candidate_bundle)
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.reset_pen_pos_noise = 0.0
    env_cfg.reset_pen_vel_noise = 0.0
    if hasattr(env_cfg, "teacher_prefill_steps"):
        env_cfg.teacher_prefill_steps = args.teacher_prefill_steps
    if args.collision_proxy_scales is not None:
        env_cfg.collision_proxy_radius_scales = tuple(args.collision_proxy_scales)
    # Keep invalid candidates alive for measurement. Acceptance uses the
    # explicit strict gate below.
    env_cfg.penetration_threshold = 0.020
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s, (args.steps + 2) * env_cfg.decimation * env_cfg.sim.dt
    )
    env = gym.make(args.task, cfg=env_cfg)
    direct_env = env.unwrapped
    env.reset()

    manus_qpos = torch.as_tensor(manus_qpos_np, device=direct_env.device)
    candidate_initial_pen_offset = torch.as_tensor(
        candidate_initial_pen_offset_np, device=direct_env.device
    )
    required_all_steps = torch.ones(args.num_envs, dtype=torch.bool, device=direct_env.device)
    contact_steps = torch.zeros((args.num_envs, 5), device=direct_env.device)
    proxy_contact_steps = torch.zeros((args.num_envs, 5), device=direct_env.device)
    undesired_contact_steps = torch.zeros(args.num_envs, device=direct_env.device)
    fingertip_force_sum = torch.zeros((args.num_envs, 5), device=direct_env.device)
    max_drift = torch.zeros(args.num_envs, device=direct_env.device)
    max_hand_rms = torch.zeros(args.num_envs, device=direct_env.device)
    max_force = torch.zeros(args.num_envs, device=direct_env.device)
    max_axis_drift = torch.zeros(args.num_envs, device=direct_env.device)
    max_angular_speed = torch.zeros(args.num_envs, device=direct_env.device)
    max_penetration_by_finger = torch.zeros((args.num_envs, 5), device=direct_env.device)
    terminated_ever = torch.zeros(args.num_envs, dtype=torch.bool, device=direct_env.device)

    actions = torch.zeros((args.num_envs, direct_env.cfg.action_space), device=direct_env.device)
    second_actions = actions.clone()
    if args.fixed_first_action is not None:
        actions[:] = torch.tensor(args.fixed_first_action, device=direct_env.device)
    if args.fixed_second_action is not None:
        second_actions[:] = torch.tensor(args.fixed_second_action, device=direct_env.device)
    else:
        second_actions.copy_(actions)
    for step in range(args.steps):
        step_actions = second_actions if step >= args.switch_step else actions
        _, _, terminated, _, _ = env.step(step_actions)
        direct_env._compute_intermediate_values()
        contacts = direct_env.contact_force_norm > 0.03
        required_all_steps &= contacts[:, args.required_contact_ids].all(dim=-1)
        contact_steps += contacts.float()
        if args.nonprimary_contact_ids:
            undesired_contact_steps += contacts[:, args.nonprimary_contact_ids].any(dim=-1).float()
        fingertip_force_sum += direct_env.contact_force_norm
        max_drift = torch.maximum(
            max_drift, torch.linalg.norm(direct_env.pen_pos - direct_env.init_pen_pos, dim=-1)
        )
        max_hand_rms = torch.maximum(
            max_hand_rms,
            torch.sqrt(torch.mean((direct_env.hand_qpos - manus_qpos.unsqueeze(0)) ** 2, dim=-1)),
        )
        max_force = torch.maximum(max_force, direct_env.contact_force_norm.max(dim=-1).values)
        axis_dot = torch.sum(direct_env.pen_axis * direct_env.init_pen_axis, dim=-1).clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot.abs())
        max_axis_drift = torch.maximum(max_axis_drift, axis_drift)
        max_angular_speed = torch.maximum(
            max_angular_speed, torch.linalg.norm(direct_env.pen_angvel, dim=-1)
        )
        max_penetration_by_finger = torch.maximum(
            max_penetration_by_finger,
            direct_env.episode_max_penetration_by_finger,
        )
        proxy_contact_steps += (direct_env.current_penetration_by_finger > 0.0).float()
        terminated_ever |= terminated

    penetration = direct_env.episode_max_penetration
    contact_ratio = contact_steps / args.steps
    proxy_contact_ratio = proxy_contact_steps / args.steps
    required_contact_gate = (
        contact_ratio[:, args.required_contact_ids] >= args.min_triad_contact_ratio
    ).all(dim=-1)
    undesired_ratio = undesired_contact_steps / args.steps
    if args.nonprimary_contact_ids:
        nonprimary_force = fingertip_force_sum[:, args.nonprimary_contact_ids].sum(dim=-1)
    else:
        nonprimary_force = torch.zeros(args.num_envs, device=direct_env.device)
    nonprimary_force_share = nonprimary_force / fingertip_force_sum.sum(dim=-1).clamp_min(1.0e-6)
    force_share_by_finger = fingertip_force_sum / fingertip_force_sum.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    explicit_contact_gate = torch.ones(args.num_envs, dtype=torch.bool, device=direct_env.device)
    for finger_id, min_ratio in min_contact_ratio_by_id.items():
        explicit_contact_gate &= contact_ratio[:, finger_id] >= min_ratio
    explicit_proxy_gate = torch.ones(args.num_envs, dtype=torch.bool, device=direct_env.device)
    for finger_id, min_ratio in min_proxy_contact_ratio_by_id.items():
        explicit_proxy_gate &= proxy_contact_ratio[:, finger_id] >= min_ratio
    explicit_force_gate = torch.ones(args.num_envs, dtype=torch.bool, device=direct_env.device)
    for finger_id, max_share in max_force_share_by_id.items():
        explicit_force_gate &= force_share_by_finger[:, finger_id] <= max_share
    bank_ids = direct_env.current_bank_ids
    initial_pen_offset = candidate_initial_pen_offset[bank_ids]
    accepted = (
        required_contact_gate
        & (penetration < args.max_penetration_mm / 1000.0)
        & (max_drift < args.max_drift_mm / 1000.0)
        & (max_hand_rms < args.max_hand_rms_rad)
        & (initial_pen_offset < args.max_initial_pen_offset_mm / 1000.0)
        & (max_axis_drift < torch.deg2rad(torch.tensor(args.max_axis_drift_deg, device=direct_env.device)))
        & (max_angular_speed < args.max_angular_speed_rad_s)
        & (max_force < 4.0)
        & (undesired_contact_steps <= args.max_nonprimary_contact_steps)
        & (nonprimary_force_share <= args.max_nonprimary_force_share)
        & explicit_contact_gate
        & explicit_proxy_gate
        & explicit_force_gate
        & ~terminated_ever
    )
    score = (
        20.0 * contact_ratio[:, args.required_contact_ids].mean(dim=-1)
        - 1200.0 * torch.relu(penetration - args.max_penetration_mm / 1000.0)
        - 40.0 * max_drift
        - 4.0 * undesired_ratio
        - 8.0 * nonprimary_force_share
        - 3.0 * max_hand_rms
        - 40.0 * initial_pen_offset
        - 0.1 * max_force
        - 0.08 * max_axis_drift * (180.0 / torch.pi)
        - 0.03 * max_angular_speed
    )
    top_ids = torch.topk(score, k=min(32, args.num_envs)).indices

    rows = []
    for env_id in top_ids.tolist():
        candidate_id = int(bank_ids[env_id].item())
        rows.append(
            {
                "env_id": env_id,
                "candidate_id": candidate_id,
                "accepted": bool(accepted[env_id].item()),
                "required_all_steps": bool(required_all_steps[env_id].item()),
                "contact_ratio": {
                    finger: float(contact_ratio[env_id, finger_id].item())
                    for finger_id, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"))
                },
                "proxy_contact_ratio": {
                    finger: float(proxy_contact_ratio[env_id, finger_id].item())
                    for finger_id, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"))
                },
                "max_penetration_mm": float(penetration[env_id].item() * 1000.0),
                "max_penetration_by_finger_mm": {
                    finger: float(max_penetration_by_finger[env_id, finger_id].item() * 1000.0)
                    for finger_id, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"))
                },
                "pen_drift_mm": float(max_drift[env_id].item() * 1000.0),
                "max_hand_manus_rms_rad": float(max_hand_rms[env_id].item()),
                "initial_hand_manus_rms_rad": float(candidate_initial_rms_np[candidate_id]),
                "initial_pen_offset_from_manifest_mm": float(
                    candidate_initial_pen_offset_np[candidate_id] * 1000.0
                ),
                "ring_or_pinky_contact_ratio": float(undesired_ratio[env_id].item()),
                "nonprimary_force_share": float(nonprimary_force_share[env_id].item()),
                "force_share": {
                    finger: float(force_share_by_finger[env_id, finger_id].item())
                    for finger_id, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"))
                },
                "max_fingertip_force_n": float(max_force[env_id].item()),
                "max_axis_drift_deg": float(max_axis_drift[env_id].item() * 180.0 / torch.pi),
                "max_angular_speed_rad_s": float(max_angular_speed[env_id].item()),
                "terminated": bool(terminated_ever[env_id].item()),
                "score": float(score[env_id].item()),
            }
        )

    accepted_ids = torch.nonzero(accepted, as_tuple=False).squeeze(-1)
    best_row = None
    if accepted_ids.numel() > 0:
        best_env = accepted_ids[torch.argmax(score[accepted_ids])]
        accepted_candidate_ids = bank_ids[accepted_ids].cpu().numpy().astype(np.int64)
        accepted_env_ids_np = accepted_ids.cpu().numpy().astype(np.int64)
        with np.load(candidate_bundle) as data:
            source_qpos = np.asarray(data["source_qpos"][accepted_candidate_ids], dtype=np.float32)
            source_qvel = np.asarray(data["source_qvel"][accepted_candidate_ids], dtype=np.float32)
            source_target_qpos = np.asarray(
                data["source_target_qpos"][accepted_candidate_ids], dtype=np.float32
            )
            target_qpos = np.asarray(data["target_qpos"][accepted_candidate_ids], dtype=np.float32)
            target_qvel = np.asarray(data["target_qvel"][accepted_candidate_ids], dtype=np.float32)
        count = source_qpos.shape[0]
        teacher_actions = np.zeros((count, max(args.steps, 1), 20), dtype=np.float32)
        if args.fixed_first_action is not None:
            teacher_actions[:, : args.switch_step] = np.asarray(
                args.fixed_first_action, dtype=np.float32
            )
        if args.fixed_second_action is not None:
            teacher_actions[:, args.switch_step :] = np.asarray(
                args.fixed_second_action, dtype=np.float32
            )
        elif args.fixed_first_action is not None:
            teacher_actions[:, args.switch_step :] = np.asarray(
                args.fixed_first_action, dtype=np.float32
            )
        accepted_bundle = {
            "source_qpos": source_qpos,
            "source_qvel": source_qvel,
            "source_target_qpos": source_target_qpos,
            "source_policy_action": np.zeros((count, 20), dtype=np.float32),
            "source_teacher_actions": teacher_actions,
            "target_qpos": target_qpos,
            "target_qvel": target_qvel,
            "phase_ids": np.full(count, args.manifest_phase, dtype=np.int64),
            "source_candidate_ids": accepted_candidate_ids,
            "accepted_env_ids": accepted_env_ids_np,
        }
        accepted_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(accepted_bundle_path, **accepted_bundle)
        best_row = next(row for row in rows if row["env_id"] == int(best_env.item()))
    elif accepted_bundle_path.exists():
        accepted_bundle_path.unlink()

    result = {
        "task": args.task,
        "candidate_count": args.candidates,
        "num_envs": args.num_envs,
        "unique_candidates_evaluated": int(torch.unique(bank_ids).numel()),
        "steps": args.steps,
        "elapsed_ms": args.steps * env_cfg.decimation * env_cfg.sim.dt * 1000.0,
        "acceptance": {
            "required_contact_ids": args.required_contact_ids,
            "required_all_steps": True,
            "min_required_contact_ratio": args.min_triad_contact_ratio,
            "max_penetration_mm": args.max_penetration_mm,
            "max_pen_drift_mm": args.max_drift_mm,
            "max_hand_reference_rms_rad": args.max_hand_rms_rad,
            "max_initial_pen_offset_from_manifest_mm": args.max_initial_pen_offset_mm,
            "max_axis_drift_deg": args.max_axis_drift_deg,
            "max_angular_speed_rad_s": args.max_angular_speed_rad_s,
            "max_fingertip_force_n": 4.0,
            "nonprimary_contact_ids": args.nonprimary_contact_ids,
            "max_nonprimary_contact_steps": args.max_nonprimary_contact_steps,
            "max_nonprimary_force_share": args.max_nonprimary_force_share,
            "min_contact_ratio_by_id": min_contact_ratio_by_id,
            "min_proxy_contact_ratio_by_id": min_proxy_contact_ratio_by_id,
            "max_force_share_by_id": max_force_share_by_id,
        },
        "accepted_env_count": int(accepted.sum().item()),
        "accepted_unique_candidate_count": int(torch.unique(bank_ids[accepted]).numel()),
        "accepted_ratio": float(accepted.float().mean().item()),
        "best": best_row,
        "top": rows,
        "candidate_bundle": str(candidate_bundle.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)) if manifest_path else None,
        "manifest_phase": args.manifest_phase if manifest_path else None,
        "collision_proxy_radius_scales": (
            list(args.collision_proxy_scales)
            if args.collision_proxy_scales is not None
            else list(env_cfg.collision_proxy_radius_scales or ())
        ),
        "fixed_first_action": args.fixed_first_action,
        "fixed_second_action": args.fixed_second_action,
        "source_target_offset": args.source_target_offset,
        "override_center_finger_ids": args.override_center_finger_ids,
        "switch_step": args.switch_step,
        "accepted_bundle": (
            str(accepted_bundle_path.relative_to(PROJECT_ROOT)) if best_row is not None else None
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("DEX5_CANONICAL_P3_PHYSX_PROJECTION=" + json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
