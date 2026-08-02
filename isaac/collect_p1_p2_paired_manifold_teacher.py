from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / (
    "reports/m3_2_30_index_near_surface/p1_p2_primitive/tripod_balance"
)
parser = argparse.ArgumentParser(
    description="Collect P1_617 waypoint teachers for the paired six-mode P2 manifold."
)
parser.add_argument(
    "--task", default="Dex5-P1P2-P1617PairedManifold120HzH6-Direct-v0"
)
parser.add_argument(
    "--manifold-bank",
    type=Path,
    default=DEFAULT_DIR / "P1_617配对P2流形_500ms三seed可靠子集.npz",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trajectories", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--max_steps", type=int, default=12000)
parser.add_argument("--seed", type=int, default=2821)
parser.add_argument(
    "--fixed-clearance-offset",
    type=float,
    nargs=4,
    default=(
        0.09726466983556747,
        -0.1489972621202469,
        0.02951819635927677,
        -0.15583617985248566,
    ),
)
parser.add_argument("--tim-contact-target-n", type=float, default=0.06)
parser.add_argument(
    "--source-archive-row",
    type=int,
    default=None,
    help="Restrict collection to one verified paired-manifold route.",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def teacher_actions(
    direct, parameters: torch.Tensor, clearance_offset: torch.Tensor
) -> torch.Tensor:
    time = direct.episode_length_buf.float() + 1.0
    source = direct.source_targets[:, 8:12]
    clearance = source + clearance_offset.unsqueeze(0)
    waypoint2 = source + parameters[:, 0:4]
    endpoint = source + parameters[:, 4:8]
    durations = parameters[:, 8:11].round().clamp_min(1.0)
    d1, d2, d3 = durations.unbind(dim=-1)
    first = source + smoothstep(time / d1).unsqueeze(-1) * (clearance - source)
    second = clearance + smoothstep((time - d1) / d2).unsqueeze(-1) * (
        waypoint2 - clearance
    )
    third = waypoint2 + smoothstep((time - d1 - d2) / d3).unsqueeze(-1) * (
        endpoint - waypoint2
    )
    desired = torch.where(
        (time <= d1).unsqueeze(-1),
        first,
        torch.where((time <= d1 + d2).unsqueeze(-1), second, third),
    )
    final_phase = time > d1 + d2
    gain = torch.where(final_phase, parameters[:, 11], torch.ones_like(time))
    actions = torch.zeros(
        (direct.num_envs, direct.cfg.action_space), device=direct.device
    )
    increment = 0.25 * float(direct.cfg.action_scale)
    actions[:, 8:12] = torch.clamp(
        gain.unsqueeze(-1) * (desired - direct.policy_targets[:, 8:12]) / increment,
        -1.0,
        1.0,
    )
    target_force = max(float(args.tim_contact_target_n), 1.0e-6)
    force_deficit = torch.clamp(
        (target_force - direct.contact_force_norm[:, :3]) / target_force, 0.0, 1.0
    )
    tim_gains = parameters[:, 12:21].reshape(-1, 3, 3)
    for finger, joint_slice in enumerate(
        (slice(17, 20), slice(1, 4), slice(5, 8))
    ):
        actions[:, joint_slice] = force_deficit[:, finger].unsqueeze(-1) * tim_gains[
            :, finger
        ]
    return actions


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifold_path = args.manifold_bank.resolve()
    with np.load(manifold_path, allow_pickle=False) as manifold:
        search_parameters = np.asarray(manifold["search_parameters"], dtype=np.float32)
        manifold_modes = np.asarray(manifold["manifold_mode"], dtype=np.int64)
        archive_rows = np.asarray(manifold["source_row_ids"], dtype=np.int64)
        manifold_features = np.asarray(manifold["ring_feature_m"], dtype=np.float32)
    all_manifold_modes = manifold_modes.copy()
    all_manifold_features = manifold_features.copy()
    if args.source_archive_row is not None:
        selected = archive_rows == int(args.source_archive_row)
        if np.count_nonzero(selected) != 1:
            raise ValueError(
                f"Expected exactly one route for source row {args.source_archive_row}, "
                f"found {np.count_nonzero(selected)}"
            )
        search_parameters = search_parameters[selected]
        manifold_modes = manifold_modes[selected]
        archive_rows = archive_rows[selected]
    manifold_centers = np.stack(
        [
            np.median(all_manifold_features[all_manifold_modes == mode], axis=0)
            for mode in range(6)
        ]
    ).astype(np.float32)
    if search_parameters.ndim != 2 or search_parameters.shape[1] != 21:
        raise ValueError("Expected 21-parameter paired-manifold routes")
    if set(np.unique(all_manifold_modes).tolist()) != set(range(6)):
        raise ValueError("Teacher source must retain all six P2 manifold modes")

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if cfg.split_name != "train" or not cfg.p2_manifold_path:
        raise ValueError("Teacher generation requires the train-only paired-manifold task")
    cfg.seed = args.seed
    cfg.episode_length_s = max(float(cfg.episode_length_s), 2.0)
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    obs, _ = env.reset()
    row_by_env = np.arange(args.num_envs, dtype=np.int64) % len(search_parameters)
    parameter_by_env = torch.as_tensor(
        search_parameters[row_by_env], dtype=torch.float32, device=direct.device
    )
    mode_by_env = manifold_modes[row_by_env]
    archive_by_env = archive_rows[row_by_env]
    clearance_offset = torch.as_tensor(
        args.fixed_clearance_offset, dtype=torch.float32, device=direct.device
    )

    horizon = direct.max_episode_length
    dims = {
        "observation": obs["policy"].shape[-1],
        "action": direct.cfg.action_space,
        "hand_qpos": 20,
        "hand_qvel": 20,
        "pen_pose": 7,
        "pen_velocity": 6,
        "policy_target": 20,
        "contact_force": 5,
        "ring_feature": 3,
    }
    buffers = {
        key: torch.zeros((args.num_envs, horizon, dim), device=direct.device)
        for key, dim in dims.items()
    }
    accepted: dict[str, list[np.ndarray]] = {key: [] for key in dims}
    lengths: list[int] = []
    accepted_modes: list[int] = []
    accepted_archive_rows: list[int] = []
    loss_streak = torch.zeros(
        args.num_envs, dtype=torch.long, device=direct.device
    )
    hard_failure = torch.zeros(
        args.num_envs, dtype=torch.bool, device=direct.device
    )
    steps = 0
    while len(lengths) < args.trajectories and steps < args.max_steps:
        indices = direct.episode_length_buf.long().clamp_max(horizon - 1)
        rows = torch.arange(args.num_envs, device=direct.device)
        action = teacher_actions(direct, parameter_by_env, clearance_offset)
        if torch.count_nonzero(action[:, 12:16]):
            raise RuntimeError("Pinky action must remain exactly zero")
        allowed = torch.zeros_like(action, dtype=torch.bool)
        allowed[:, 8:12] = True
        allowed[:, 1:4] = True
        allowed[:, 5:8] = True
        allowed[:, 17:20] = True
        if torch.count_nonzero(torch.where(allowed, torch.zeros_like(action), action)):
            raise RuntimeError("Teacher emitted an action outside ring/TIM permissions")

        buffers["observation"][rows, indices] = obs["policy"]
        buffers["action"][rows, indices] = action
        buffers["hand_qpos"][rows, indices] = direct.hand_qpos
        buffers["hand_qvel"][rows, indices] = direct.hand_qvel
        buffers["pen_pose"][rows, indices, :3] = direct.pen_pos
        buffers["pen_pose"][rows, indices, 3:] = direct.pen.data.root_quat_w
        buffers["pen_velocity"][rows, indices, :3] = direct.pen_linvel
        buffers["pen_velocity"][rows, indices, 3:] = direct.pen_angvel
        buffers["policy_target"][rows, indices] = direct.policy_targets
        buffers["contact_force"][rows, indices] = direct.contact_force_norm
        buffers["ring_feature"][rows, indices] = direct._ring_axis_features()

        obs, _, terminated, truncated, _ = env.step(action)
        direct._compute_intermediate_values()
        triad = (
            direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n
        ).all(dim=-1)
        loss_streak = torch.where(
            triad, torch.zeros_like(loss_streak), loss_streak + 1
        )
        position_drift = torch.linalg.norm(
            direct.pen_pos - direct.init_pen_pos, dim=-1
        )
        axis_dot = torch.sum(
            direct.pen_axis * direct.init_pen_axis, dim=-1
        ).abs().clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot)
        hard_failure |= (
            (loss_streak > 1)
            | (position_drift > 0.005)
            | (axis_drift > torch.deg2rad(torch.full_like(axis_drift, 4.0)))
            | (direct.episode_max_penetration > 0.00060)
            | (direct.contact_force_norm[:, 3] > 0.30)
            | (direct.contact_force_norm[:, 4] > 0.03)
        )
        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        for env_id in done_ids.tolist():
            if bool(direct.last_completed_p1_p2_success[env_id]) and not bool(
                hard_failure[env_id]
            ):
                length = int(indices[env_id].item()) + 1
                for key, buffer in buffers.items():
                    accepted[key].append(
                        buffer[env_id, :length].cpu().numpy().copy()
                    )
                lengths.append(length)
                accepted_modes.append(int(mode_by_env[env_id]))
                accepted_archive_rows.append(int(archive_by_env[env_id]))
                if len(lengths) >= args.trajectories:
                    break
        if done_ids.numel():
            for buffer in buffers.values():
                buffer[done_ids] = 0.0
            loss_streak[done_ids] = 0
            hard_failure[done_ids] = False
        steps += 1

    if len(lengths) < args.trajectories:
        raise RuntimeError(
            f"Collected only {len(lengths)}/{args.trajectories} successful teachers "
            f"after {steps} PhysX steps"
        )
    max_length = max(lengths)
    arrays: dict[str, np.ndarray] = {}
    for key, rows_data in accepted.items():
        array = np.zeros((len(rows_data), max_length, dims[key]), dtype=np.float32)
        for row, values in enumerate(rows_data):
            array[row, : len(values)] = values
        arrays[key] = array
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    final_features = np.stack(
        [arrays["ring_feature"][row, length - 1] for row, length in enumerate(lengths)]
    )
    final_mode_distances = np.linalg.norm(
        final_features[:, None, :] - manifold_centers[None, :, :], axis=-1
    )
    actual_modes = np.argmin(final_mode_distances, axis=-1).astype(np.int64)
    final_distances_mm = (
        final_mode_distances[np.arange(len(actual_modes)), actual_modes] * 1000.0
    ).astype(np.float32)
    planned_modes = np.asarray(accepted_modes, dtype=np.int64)
    np.savez_compressed(
        output,
        **arrays,
        lengths=np.asarray(lengths, dtype=np.int64),
        manifold_mode=actual_modes,
        planned_manifold_mode=planned_modes,
        source_archive_row=np.asarray(accepted_archive_rows, dtype=np.int64),
        final_mode_distance_mm=final_distances_mm,
        fixed_clearance_offset=np.asarray(args.fixed_clearance_offset, dtype=np.float32),
    )
    mode_counts = {
        str(mode): int(np.sum(actual_modes == mode)) for mode in range(6)
    }
    planned_mode_counts = {
        str(mode): int(np.sum(planned_modes == mode)) for mode in range(6)
    }
    report = {
        "task": args.task,
        "output": str(output),
        "output_sha256": sha256(output),
        "manifold_bank": str(manifold_path),
        "manifold_bank_sha256": sha256(manifold_path),
        "trajectories": len(lengths),
        "transitions": int(sum(lengths)),
        "length_min": min(lengths),
        "length_mean": float(np.mean(lengths)),
        "length_max": max(lengths),
        "mode_counts": mode_counts,
        "planned_mode_counts": planned_mode_counts,
        "planned_actual_mode_match_rate": float(np.mean(planned_modes == actual_modes)),
        "simulation_steps": steps,
        "ring_is_primary_mover": True,
        "maximum_tim_target_offset_rad": float(
            direct.cfg.action_scale * direct.cfg.support_action_mask_scale
        ),
        "pinky_action_max_abs": 0.0,
        "success_gate": (
            "nearest verified P2 manifold mode + ring-under/near-surface semantics "
            "+ TIM/stability/load gates + 100 ms hold"
        ),
        "validation_or_test_accessed": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_PAIRED_MANIFOLD_TEACHER=" + json.dumps(report, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
