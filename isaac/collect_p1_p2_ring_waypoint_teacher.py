from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMETER_NAMES = [
    *(f"retreat_fraction_j{joint}" for joint in range(4)),
    *(f"wrap_fraction_j{joint}" for joint in range(4)),
    "retreat_steps", "wrap_steps", "settle_steps", "settle_gain",
]
TIM_GAIN_NAMES = [
    *(f"thumb_contact_gain_j{joint}" for joint in range(3)),
    *(f"index_contact_gain_j{joint}" for joint in range(3)),
    *(f"middle_contact_gain_j{joint}" for joint in range(3)),
]
MIDDLE_ANCHOR_NAMES = [f"middle_anchor_offset_j{joint}" for joint in range(3)]

parser = argparse.ArgumentParser(description="Collect strict successful ring-only waypoint teachers.")
parser.add_argument("--task", default="Dex5-P1P2-RingOnly120HzH1-Direct-v0")
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trajectories", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_steps", type=int, default=30000)
parser.add_argument("--seed", type=int, default=2821)
parser.add_argument("--p2-modes", type=Path)
parser.add_argument("--p2-mode", type=int, default=-1)
parser.add_argument("--tim-assist", action="store_true")
parser.add_argument("--tim-contact-target-n", type=float, default=0.06)
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


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def load_parameters(path: Path) -> np.ndarray:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    values = payload.get("best_parameters", payload.get("parameters"))
    if isinstance(values, dict):
        names = list(PARAMETER_NAMES)
        if TIM_GAIN_NAMES[0] in values:
            names += TIM_GAIN_NAMES
        if MIDDLE_ANCHOR_NAMES[0] in values:
            names += MIDDLE_ANCHOR_NAMES
        values = [values[name] for name in names]
    result = np.asarray(values, dtype=np.float32)
    valid_sizes = {
        len(PARAMETER_NAMES),
        len(PARAMETER_NAMES) + len(TIM_GAIN_NAMES),
        len(PARAMETER_NAMES) + len(MIDDLE_ANCHOR_NAMES),
        len(PARAMETER_NAMES) + len(TIM_GAIN_NAMES) + len(MIDDLE_ANCHOR_NAMES),
    }
    if result.size not in valid_sizes:
        raise ValueError(f"Invalid waypoint parameter shape {result.shape}")
    return result


def waypoint_actions(direct, parameters: torch.Tensor) -> torch.Tensor:
    time = direct.episode_length_buf.float() + 1.0
    source = direct.source_targets[:, 8:12]
    final = direct.p2_target_ring_qpos.unsqueeze(0).expand_as(source)
    delta = final - source
    waypoint1 = source + parameters[None, 0:4] * delta
    waypoint2 = source + parameters[None, 4:8] * delta
    d1, d2, d3 = (parameters[index].round().clamp_min(1.0) for index in (8, 9, 10))
    tau1 = smoothstep(time / d1)
    tau2 = smoothstep((time - d1) / d2)
    tau3 = smoothstep((time - d1 - d2) / d3)
    first = source + tau1[:, None] * (waypoint1 - source)
    second = waypoint1 + tau2[:, None] * (waypoint2 - waypoint1)
    third = waypoint2 + tau3[:, None] * (final - waypoint2)
    desired = torch.where(
        (time <= d1)[:, None], first,
        torch.where((time <= d1 + d2)[:, None], second, third),
    )
    gain = torch.where(time > d1 + d2, parameters[11], torch.ones_like(time))
    actions = torch.zeros((direct.num_envs, direct.cfg.action_space), device=direct.device)
    increment = 0.25 * float(direct.cfg.action_scale)
    actions[:, 8:12] = torch.clamp(
        gain[:, None] * (desired - direct.policy_targets[:, 8:12]) / increment,
        -1.0, 1.0,
    )
    if args.tim_assist:
        target_force = max(float(args.tim_contact_target_n), 1.0e-6)
        force_deficit = torch.clamp(
            (target_force - direct.contact_force_norm[:, :3]) / target_force,
            0.0,
            1.0,
        )
        # Contact order is thumb/index/middle; policy joint blocks are
        # index/middle/ring/pinky/thumb.
        tim_gains = (
            parameters[12:21].reshape(3, 3)
            if parameters.numel() >= 21
            else torch.ones((3, 3), device=direct.device)
        )
        for finger, joint_slice in enumerate(
            (slice(17, 20), slice(1, 4), slice(5, 8))
        ):
            actions[:, joint_slice] = (
                force_deficit[:, finger].unsqueeze(-1) * tim_gains[finger]
            )
    return actions


def main() -> None:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if cfg.split_name != "train":
        raise ValueError("Teacher collection is restricted to train split")
    selected_mode = None
    if args.p2_modes:
        payload = json.loads(args.p2_modes.resolve().read_text(encoding="utf-8"))
        selected_mode = payload["modes"][args.p2_mode]
        cfg.target_p2_ring_qpos_override = tuple(selected_mode["ring_qpos_center"])
        cfg.target_p2_ring_center_override = tuple(selected_mode["ring_feature_center_m"])
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    direct = env.unwrapped
    obs, _ = env.reset()
    parameters = torch.as_tensor(load_parameters(args.plan), device=direct.device)
    has_tim_gains = parameters.numel() in (21, 24)
    has_middle_anchor = parameters.numel() in (15, 24)
    middle_anchor_offset = (
        parameters[(21 if has_tim_gains else 12) :][:3]
        if has_middle_anchor
        else None
    )
    if middle_anchor_offset is not None:
        direct.support_anchor_targets[:, 5:8] += middle_anchor_offset
        direct.policy_targets[:, 5:8] = direct.support_anchor_targets[:, 5:8]
    horizon = direct.max_episode_length
    dims = {
        "observation": obs["policy"].shape[-1], "action": direct.cfg.action_space,
        "hand_qpos": 20, "hand_qvel": 20, "pen_pose": 7,
        "pen_velocity": 6, "policy_target": 20, "contact_force": 5,
    }
    buffers = {
        key: torch.zeros((args.num_envs, horizon, dim), device=direct.device)
        for key, dim in dims.items()
    }
    episode_bank_ids = direct.current_global_bank_ids.clone()
    loss_streak = torch.zeros(args.num_envs, dtype=torch.long, device=direct.device)
    hard_failure = torch.zeros(args.num_envs, dtype=torch.bool, device=direct.device)
    accepted: dict[str, list[np.ndarray]] = {key: [] for key in buffers}
    lengths: list[int] = []
    bank_ids: list[int] = []
    steps = 0
    while len(lengths) < args.trajectories and steps < args.max_steps:
        row = direct.episode_length_buf.long().clamp_max(horizon - 1)
        ids = torch.arange(args.num_envs, device=direct.device)
        action = waypoint_actions(direct, parameters)
        allowed = torch.zeros_like(action, dtype=torch.bool)
        allowed[:, 8:12] = True
        if args.tim_assist:
            allowed[:, 1:4] = True
            allowed[:, 5:8] = True
            allowed[:, 17:20] = True
        if torch.count_nonzero(torch.where(allowed, torch.zeros_like(action), action)):
            raise RuntimeError("Teacher emitted an action outside ring/TIM permissions")
        if torch.count_nonzero(action[:, 12:16]):
            raise RuntimeError("Teacher emitted a pinky action")
        buffers["observation"][ids, row] = obs["policy"]
        buffers["action"][ids, row] = action
        buffers["hand_qpos"][ids, row] = direct.hand_qpos
        buffers["hand_qvel"][ids, row] = direct.hand_qvel
        buffers["pen_pose"][ids, row, :3] = direct.pen_pos
        buffers["pen_pose"][ids, row, 3:] = direct.pen.data.root_quat_w
        buffers["pen_velocity"][ids, row, :3] = direct.pen_linvel
        buffers["pen_velocity"][ids, row, 3:] = direct.pen_angvel
        buffers["policy_target"][ids, row] = direct.policy_targets
        buffers["contact_force"][ids, row] = direct.contact_force_norm
        obs, _, terminated, truncated, _ = env.step(action)
        direct._compute_intermediate_values()
        triad = (direct.contact_force_norm[:, :3] > direct.cfg.p2_contact_force_n).all(dim=-1)
        loss_streak = torch.where(triad, torch.zeros_like(loss_streak), loss_streak + 1)
        position_drift = torch.linalg.norm(direct.pen_pos - direct.init_pen_pos, dim=-1)
        axis_dot = torch.sum(direct.pen_axis * direct.init_pen_axis, dim=-1).abs().clamp(-1.0, 1.0)
        axis_drift = torch.acos(axis_dot)
        hard_failure |= (
            (loss_streak > 1) | (position_drift > 0.010)
            | (axis_drift > torch.deg2rad(torch.full_like(axis_drift, 8.0)))
            | (direct.episode_max_penetration > 0.00060)
        )
        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
        for env_id in done_ids.tolist():
            if bool(direct.last_completed_p1_p2_success[env_id]) and not bool(hard_failure[env_id]):
                length = int(row[env_id].item()) + 1
                for key, buffer in buffers.items():
                    accepted[key].append(buffer[env_id, :length].cpu().numpy().copy())
                lengths.append(length)
                bank_ids.append(int(episode_bank_ids[env_id].item()))
                if len(lengths) >= args.trajectories:
                    break
        if done_ids.numel():
            for buffer in buffers.values():
                buffer[done_ids] = 0.0
            loss_streak[done_ids] = 0
            hard_failure[done_ids] = False
            episode_bank_ids[done_ids] = direct.current_global_bank_ids[done_ids]
            if middle_anchor_offset is not None:
                direct.support_anchor_targets[done_ids, 5:8] += middle_anchor_offset
                direct.policy_targets[done_ids, 5:8] = direct.support_anchor_targets[
                    done_ids, 5:8
                ]
        steps += 1
    if len(lengths) < args.trajectories:
        raise RuntimeError(
            f"Collected only {len(lengths)}/{args.trajectories} strict trajectories after {steps} steps"
        )
    max_length = max(lengths)
    arrays = {}
    for key, rows in accepted.items():
        array = np.zeros((len(rows), max_length, dims[key]), dtype=np.float32)
        for index, values in enumerate(rows):
            array[index, : len(values)] = values
        arrays[key] = array
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, **arrays, lengths=np.asarray(lengths, dtype=np.int64),
        stage_id=np.full((len(lengths), max_length), 3, dtype=np.int64),
        global_bank_ids=np.asarray(bank_ids, dtype=np.int64),
        waypoint_parameters=parameters.cpu().numpy(),
    )
    report = {
        "task": args.task, "output": str(output), "trajectories": len(lengths),
        "transitions": int(sum(lengths)), "unique_train_p1_ids": len(set(bank_ids)),
        "length_min": min(lengths), "length_mean": float(np.mean(lengths)),
        "length_max": max(lengths), "simulation_steps": steps,
        "p2_mode": args.p2_mode, "selected_mode": selected_mode,
        "ring_is_primary_mover": True,
        "tim_assist": args.tim_assist,
        "tim_contact_target_n": float(args.tim_contact_target_n),
        "pinky_action_max_abs": 0.0,
        "maximum_tim_target_offset_rad": (
            float(direct.cfg.action_scale * direct.cfg.support_action_mask_scale)
            if args.tim_assist else 0.0
        ),
        "middle_anchor_offset_rad": (
            middle_anchor_offset.cpu().tolist() if middle_anchor_offset is not None else None
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_RING_TEACHER=" + json.dumps(report, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
