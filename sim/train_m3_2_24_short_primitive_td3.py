from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from m3_2_23_short_primitive_reward import FINGERS
from m3_2_24_short_primitive_env import (
    DEFAULT_CACHE_MANIFEST,
    DEFAULT_REWARD_PLAN,
    CACHE_WINDOW_BY_PRIMITIVE,
    Dex5PenSpinShortPrimitiveEnv,
    INDEX_NEAR_SURFACE_LABELS,
    INDEX_NEAR_SURFACE_THRESHOLD_M,
    INDEX_RECAPTURE_LABELS,
    load_short_primitive_spec,
    summarize_short_primitive_rollout,
)
from train_m3_1_hold_oracle import Actor, Critic, ReplayBuffer, save_checkpoint, select_action, soft_update


PolicyFn = Callable[[np.ndarray, Dex5PenSpinShortPrimitiveEnv, int, argparse.Namespace], np.ndarray]


def make_env(
    args: argparse.Namespace,
    *,
    render: bool = False,
    seed: int | None = None,
) -> Dex5PenSpinShortPrimitiveEnv:
    return Dex5PenSpinShortPrimitiveEnv(
        primitive_label=args.primitive_label,
        reward_plan_path=args.reward_plan_path,
        cache_manifest_path=args.cache_manifest_path,
        cache_window=args.cache_window,
        model_path=args.model_path,
        episode_seconds=args.episode_seconds,
        control_hz=args.control_hz,
        action_scale=args.action_scale,
        reward_scale=args.reward_scale,
        assist_mode=args.assist_mode,
        assist_delta_limit=args.assist_delta_limit,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        action_delta_limit=args.action_delta_limit,
        tip_proxy_radius=args.tip_proxy_radius,
        tip_proxy_friction=args.tip_proxy_friction,
        reset_hand_noise=args.reset_hand_noise,
        reset_pen_pos_noise=args.reset_pen_pos_noise,
        reset_pen_quat_noise=args.reset_pen_quat_noise,
        render_mode="rgb_array" if render else None,
        seed=seed,
    )


def zero_action(_: np.ndarray, env: Dex5PenSpinShortPrimitiveEnv, step: int, args: argparse.Namespace) -> np.ndarray:
    return np.zeros(env.action_space.shape[0], dtype=np.float32)


def random_action(_: np.ndarray, env: Dex5PenSpinShortPrimitiveEnv, step: int, args: argparse.Namespace) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=env.action_space.shape[0]).astype(np.float32)


def smoothstep(x: float) -> float:
    t = float(np.clip(x, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def enforce_shared_motor_consistency(env: Dex5PenSpinShortPrimitiveEnv, q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    for pip, dip in [(2, 3), (6, 7), (10, 11), (14, 15)]:
        out[dip] = np.clip(min(out[pip], env.env.joint_high[dip]), env.env.joint_low[dip], env.env.joint_high[dip])
    return np.clip(out, env.env.joint_low, env.env.joint_high)


def scripted_delta_sequence(
    args: argparse.Namespace,
    action_dim: int,
) -> np.ndarray | None:
    if not args.scripted_action_file:
        return None
    cached = getattr(args, "_scripted_delta_sequence", None)
    if cached is not None:
        return cached
    value: Any = json.loads(
        Path(args.scripted_action_file).read_text(encoding="utf-8")
    )
    for key in args.scripted_action_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(
                f"Missing scripted action key {args.scripted_action_key!r} "
                f"in {args.scripted_action_file}"
            )
        value = value[key]
    sequence = np.asarray(value, dtype=np.float64)
    if sequence.ndim == 1:
        sequence = sequence[None, :]
    if sequence.ndim != 2 or sequence.shape[1] != action_dim:
        raise ValueError(
            f"Scripted delta sequence has shape {sequence.shape}, "
            f"expected (steps, {action_dim})"
        )
    setattr(args, "_scripted_delta_sequence", sequence)
    return sequence


def scripted_target_action(
    _: np.ndarray,
    env: Dex5PenSpinShortPrimitiveEnv,
    step: int,
    args: argparse.Namespace,
) -> np.ndarray:
    scripted_deltas = scripted_delta_sequence(args, env.action_space.shape[0])
    if scripted_deltas is not None:
        delta = scripted_deltas[min(step, scripted_deltas.shape[0] - 1)]
        return np.clip(
            delta / max(args.action_scale, 1e-9),
            -1.0,
            1.0,
        ).astype(np.float32)

    tau = smoothstep((step + 1) / max(1, env.max_episode_steps))
    start = env.scripted_start_qpos
    if env.spec.label in {
        "earlier_pre_ring_index_thumb_recapture",
        "earlier_pre_ring_016_019_index_thumb_recapture",
        "earlier_pre_ring_019_022_index_thumb_recapture",
        "earlier_pre_ring_016_019_to_projected_recapture",
        "earlier_pre_ring_019_022_to_projected_recapture",
        "early_pre_ring_index_thumb_recapture",
    } | INDEX_RECAPTURE_LABELS:
        local_tau = smoothstep((step + 1) / max(1.0, args.scripted_recapture_ramp_fraction * env.max_episode_steps))
        target = start.copy()
        index_roll_bias = args.scripted_index_roll_bias
        index_pip_bias = args.scripted_index_pip_bias
        if env.spec.label == "earlier_pre_ring_019_022_to_index_near_surface":
            index_roll_bias = args.scripted_index_late_roll_bias
            index_pip_bias = args.scripted_index_late_pip_bias
        target[0] += index_roll_bias * local_tau
        target[1] += index_pip_bias * local_tau
        target[2] += args.scripted_index_mid_bias * local_tau
        target[17] += args.scripted_recapture_thumb_roll_bias * local_tau
        target[18] += args.scripted_recapture_thumb_pitch_bias * local_tau
        target[19] += args.scripted_recapture_thumb_tip_bias * local_tau
        target[9] -= args.scripted_recapture_ring_open_bias * local_tau
        target[10] -= 0.60 * args.scripted_recapture_ring_open_bias * local_tau
        target = enforce_shared_motor_consistency(env, target)
        delta = args.scripted_tracking_gain * (target - env.env.target_qpos.astype(np.float64))
        return np.clip(delta / max(args.action_scale, 1e-9), -1.0, 1.0).astype(np.float32)

    target = (1.0 - tau) * start + tau * env.target_hand_qpos

    # Keep the hand doing the thing we actually want RL to discover:
    # thumb gives light upward/closing support, ring guards, pinky does not become a crutch.
    if "thumb" in env.spec.target_contacts:
        target[17] += args.scripted_thumb_roll_bias * tau
        target[18] += args.scripted_thumb_pitch_bias * tau
        target[19] += args.scripted_thumb_tip_bias * tau
    if "ring" in env.spec.target_contacts:
        target[9] -= args.scripted_ring_open_bias * tau
        target[10] -= 0.70 * args.scripted_ring_open_bias * tau
        target[9] += args.scripted_ring_guard_bias * tau
        target[10] += 0.45 * args.scripted_ring_guard_bias * tau
    if "pinky" in env.spec.forbidden_contacts:
        target[13] -= args.scripted_pinky_retreat_bias * tau
        target[14] -= 0.75 * args.scripted_pinky_retreat_bias * tau

    target = enforce_shared_motor_consistency(env, target)
    delta = args.scripted_tracking_gain * (target - env.env.target_qpos.astype(np.float64))
    return np.clip(delta / max(args.action_scale, 1e-9), -1.0, 1.0).astype(np.float32)


def actor_policy(actor: Actor, device: torch.device, noise_std: float = 0.0) -> PolicyFn:
    def _policy(obs: np.ndarray, env: Dex5PenSpinShortPrimitiveEnv, step: int, args: argparse.Namespace) -> np.ndarray:
        policy_action = select_action(actor, obs, device, noise_std=noise_std)
        return compose_environment_action(policy_action, obs, env, step, args)

    return _policy


def compose_environment_action(
    policy_action: np.ndarray,
    obs: np.ndarray,
    env: Dex5PenSpinShortPrimitiveEnv,
    step: int,
    args: argparse.Namespace,
) -> np.ndarray:
    residual_scale = float(args.scripted_residual_scale)
    if residual_scale <= 0.0:
        return np.asarray(policy_action, dtype=np.float32)
    teacher = scripted_target_action(obs, env, step, args)
    return np.clip(
        teacher + residual_scale * np.asarray(policy_action, dtype=np.float32),
        -1.0,
        1.0,
    ).astype(np.float32)


def load_actor_checkpoint(path: Path, obs_dim: int, action_dim: int, hidden_dim: int, device: torch.device) -> Actor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(obs_dim, action_dim, hidden=hidden_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def rollout(
    env: Dex5PenSpinShortPrimitiveEnv,
    policy_fn: PolicyFn,
    args: argparse.Namespace,
    *,
    cache_index: int | None = None,
    render: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reset_options = {"cache_index": cache_index} if cache_index is not None else None
    obs, _ = env.reset(options=reset_options)
    infos: list[dict[str, Any]] = []
    observations = [obs.copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    centers: list[np.ndarray] = []
    target_center_dist: list[float] = []
    target_axis_error: list[float] = []
    index_tip_dist: list[float] = []
    desired_ratio: list[float] = []
    support_ratio: list[float] = []
    contacts: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    total_reward = 0.0
    step = 0
    if render:
        frames.append(env.render())

    while True:
        action = policy_fn(obs, env, step, args)
        next_obs, reward, terminated, truncated, info = env.step(action)
        infos.append(info)
        observations.append(next_obs.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        centers.append(np.asarray(info["pen_center"], dtype=np.float64))
        target_center_dist.append(float(info["target_center_dist"]))
        target_axis_error.append(float(info["target_axis_error"]))
        index_tip_dist.append(float(info.get("index_tip_dist", 0.04)))
        desired_ratio.append(float(info["desired_ratio"]))
        support_ratio.append(float(info["support_ratio"]))
        contacts.append(np.asarray([info["contact_flags"].get(finger, 0.0) for finger in FINGERS], dtype=np.float32))
        total_reward += float(reward)
        step += 1
        obs = next_obs
        if render and step % args.render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break

    summary = summarize_short_primitive_rollout(infos, total_reward, step, env.control_hz)
    summary["primitive_label"] = env.spec.label
    traces: dict[str, np.ndarray] = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "centers": np.asarray(centers, dtype=np.float64),
        "target_center_dist": np.asarray(target_center_dist, dtype=np.float64),
        "target_axis_error": np.asarray(target_axis_error, dtype=np.float64),
        "index_tip_dist": np.asarray(index_tip_dist, dtype=np.float64),
        "desired_ratio": np.asarray(desired_ratio, dtype=np.float64),
        "support_ratio": np.asarray(support_ratio, dtype=np.float64),
        "contact_flags": np.asarray(contacts, dtype=np.float32),
    }
    if render:
        traces["frames"] = np.asarray(frames, dtype=np.uint8)
    return summary, traces


def primitive_score(row: dict[str, Any], duration_s: float) -> float:
    label = str(row.get("primitive_label", ""))
    is_near_surface = label in INDEX_NEAR_SURFACE_LABELS
    survival_fraction = float(row["survival_seconds"]) / max(float(duration_s), 1e-6)
    ring_overload = max(0.0, float(row.get("ring_force_share_mean", 0.0)) - 0.35)
    index_tip_final = float(row.get("index_tip_dist_final", 0.04))
    index_tip_mean = float(row.get("index_tip_dist_mean", 0.04))
    index_near_threshold = INDEX_NEAR_SURFACE_THRESHOLD_M if is_near_surface else 0.018
    index_near_final = 1.0 - min(1.0, max(0.0, index_tip_final) / index_near_threshold)
    index_near_mean = 1.0 - min(1.0, max(0.0, index_tip_mean) / 0.030)
    index_overpenetration = min(1.0, max(0.0, -float(row.get("index_tip_dist_min", 0.0)) - 0.003) / 0.006)
    index_contact_weight = 0.0 if is_near_surface else 75.0
    index_contact_term = float(row.get("index_ratio", 0.0))
    brake_score = 0.0
    if "braked_P4" in label:
        linear_final = float(row.get("pen_linear_speed_final", 1.0))
        angular_final = float(row.get("pen_angular_speed_final", 60.0))
        middle_final = float(row.get("middle_tip_dist_final", 0.04))
        brake_score = (
            260.0 * (1.0 - min(1.0, max(0.0, middle_final) / 0.025))
            - 900.0 * min(1.0, max(0.0, linear_final - 0.12) / 0.35)
            - 260.0 * min(1.0, max(0.0, angular_final - 12.0) / 30.0)
        )
    if is_near_surface:
        under_14 = float(row.get("index_near_surface_final", 0.0))
        miss_14 = max(0.0, index_tip_final - INDEX_NEAR_SURFACE_THRESHOLD_M) / 0.025
        return (
            360.0 * survival_fraction
            + 760.0 * under_14
            + 520.0 * float(row.get("target_final_window_ratio", 0.0))
            + 280.0 * index_near_final
            + 120.0 * index_near_mean
            + 65.0 * float(row.get("thumb_ratio", 0.0))
            + 35.0 * float(row.get("middle_ratio", 0.0))
            - 620.0 * min(1.0, miss_14)
            - 120.0 * float(row.get("pinky_ratio", 0.0))
            - 65.0 * float(row.get("index_ratio", 0.0))
            - 45.0 * index_overpenetration
            - 14.0 * float(row.get("max_penetration_mm", 0.0))
            - 900.0 * float(row.get("target_center_dist_final", 0.0))
            - 18.0 * float(row.get("action_norm_mean", 0.0))
            + float(row.get("return", 0.0))
            - (160.0 if row.get("terminated", False) else 0.0)
        )
    return (
        800.0 * survival_fraction
        + 180.0 * float(row.get("desired_ratio_mean", 0.0))
        + 260.0 * float(row.get("target_final_window_ratio", 0.0))
        + 130.0 * float(row.get("support_final_window_ratio", 0.0))
        + 85.0 * float(row.get("thumb_ratio", 0.0))
        + index_contact_weight * index_contact_term
        + 155.0 * index_near_final
        + 90.0 * index_near_mean
        - 150.0 * float(row.get("pinky_ratio", 0.0))
        - 120.0 * min(1.0, ring_overload / 0.45)
        - 45.0 * index_overpenetration
        - 16.0 * float(row.get("max_penetration_mm", 0.0))
        - 1200.0 * float(row.get("target_center_dist_final", 0.0))
        - 22.0 * float(row.get("action_norm_mean", 0.0))
        + float(row.get("return", 0.0))
        + brake_score
        - (260.0 if row.get("terminated", False) else 0.0)
    )


def aggregate_rows(rows: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    keys = [
        "score",
        "return",
        "survival_seconds",
        "desired_ratio_mean",
        "support_ratio_mean",
        "target_final_window_ratio",
        "desired_final_window_ratio",
        "support_final_window_ratio",
        "thumb_ratio",
        "index_ratio",
        "middle_ratio",
        "ring_ratio",
        "pinky_ratio",
        "thumb_force_share_mean",
        "index_force_share_mean",
        "middle_force_share_mean",
        "ring_force_share_mean",
        "target_center_dist_final",
        "target_axis_error_final",
        "max_penetration_mm",
        "index_tip_dist_mean",
        "index_tip_dist_min",
        "index_tip_dist_final",
        "index_near_surface_ratio",
        "index_near_surface_final",
        "thumb_tip_dist_mean",
        "middle_tip_dist_mean",
        "action_norm_mean",
    ]
    for row in rows:
        row["score"] = primitive_score(row, duration_s)
    out = {f"{key}_mean": float(np.mean([float(row.get(key, 0.0)) for row in rows])) for key in keys}
    out.update({f"{key}_min": float(np.min([float(row.get(key, 0.0)) for row in rows])) for key in keys})
    out["terminated_count"] = int(sum(bool(row.get("terminated", False)) for row in rows))
    out["episodes"] = int(len(rows))
    return out


def evaluate_policy(
    args: argparse.Namespace,
    policy_fn: PolicyFn,
    *,
    render_first: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray] | None]:
    env = make_env(args, render=render_first, seed=args.seed + 1000)
    rows: list[dict[str, Any]] = []
    first_trace: dict[str, np.ndarray] | None = None
    try:
        cache_indices: list[int | None]
        if args.eval_all_cache_states and env.cache is not None:
            cache_indices = list(range(int(env.cache["qpos"].shape[0])))
        else:
            cache_indices = [None] * int(args.eval_episodes)
        for i, cache_index in enumerate(cache_indices):
            row, traces = rollout(env, policy_fn, args, cache_index=cache_index, render=render_first and i == 0)
            row["episode"] = int(i)
            rows.append(row)
            if render_first and i == 0:
                first_trace = traces
    finally:
        env.close()
    duration = planned_duration(args)
    return aggregate_rows(rows, duration), rows, first_trace


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_trace(path: Path, traces: dict[str, np.ndarray], title: str) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    centers = traces["centers"]
    if centers.size:
        axes[0].plot(centers[:, 0], label="x")
        axes[0].plot(centers[:, 1], label="y")
        axes[0].plot(centers[:, 2], label="z")
    axes[0].set_ylabel("pen center")
    axes[0].legend(loc="best")
    axes[1].plot(traces["target_center_dist"], label="center dist")
    axes[1].plot(traces["target_axis_error"], label="axis error")
    if "index_tip_dist" in traces:
        axes[1].plot(traces["index_tip_dist"], label="index tip dist")
    axes[1].set_ylabel("target")
    axes[1].legend(loc="best")
    axes[2].plot(traces["desired_ratio"], label="desired")
    axes[2].plot(traces["support_ratio"], label="support")
    axes[2].set_ylabel("contact ratio")
    axes[2].legend(loc="best")
    contacts = traces["contact_flags"]
    for idx, finger in enumerate(FINGERS):
        axes[3].plot(contacts[:, idx] + 0.05 * idx, label=finger)
    axes[3].set_ylabel("contacts")
    axes[3].legend(loc="best", ncol=5)
    axes[4].plot(traces["rewards"])
    axes[4].set_ylabel("reward")
    axes[4].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_trace_artifacts(output_dir: Path, name: str, traces: dict[str, np.ndarray] | None) -> None:
    if traces is None:
        return
    np.savez_compressed(
        output_dir / f"{name}_rollout.npz",
        observations=traces["observations"],
        actions=traces["actions"],
        rewards=traces["rewards"],
        centers=traces["centers"],
        target_center_dist=traces["target_center_dist"],
        target_axis_error=traces["target_axis_error"],
        index_tip_dist=traces.get("index_tip_dist", np.asarray([], dtype=np.float64)),
        desired_ratio=traces["desired_ratio"],
        support_ratio=traces["support_ratio"],
        contact_flags=traces["contact_flags"],
    )
    plot_trace(output_dir / f"{name}_trace.png", traces, f"M3.2.24 {name}")
    if "frames" in traces:
        imageio.mimsave(output_dir / f"{name}_rollout.gif", list(traces["frames"]), fps=25)


def prefill_replay(
    env: Dex5PenSpinShortPrimitiveEnv,
    replay: ReplayBuffer,
    args: argparse.Namespace,
    policy_fn: PolicyFn,
    episodes: int,
) -> int:
    added = 0
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        step = 0
        while True:
            if args.scripted_residual_scale > 0.0 and policy_fn is scripted_target_action:
                replay_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                env_action = scripted_target_action(obs, env, step, args)
            else:
                replay_action = policy_fn(obs, env, step, args)
                env_action = compose_environment_action(
                    replay_action,
                    obs,
                    env,
                    step,
                    args,
                )
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = bool(terminated or truncated)
            replay.add(obs, replay_action, reward, next_obs, done)
            added += 1
            obs = next_obs
            step += 1
            if done:
                break
    return added


def collect_bc_dataset(
    env: Dex5PenSpinShortPrimitiveEnv,
    args: argparse.Namespace,
    episodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    cache_count = int(env.cache["qpos"].shape[0]) if env.cache is not None else 0
    for episode in range(int(episodes)):
        options = {"cache_index": episode % cache_count} if cache_count else None
        obs, _ = env.reset(options=options)
        step = 0
        while True:
            action = scripted_target_action(obs, env, step, args)
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            if terminated or truncated:
                break
    if not observations:
        return (
            np.zeros((0, int(env.observation_space.shape[0])), dtype=np.float32),
            np.zeros((0, int(env.action_space.shape[0])), dtype=np.float32),
        )
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def behavior_clone_actor(
    actor: Actor,
    observations: np.ndarray,
    actions: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if observations.shape[0] == 0:
        return {"samples": 0, "epochs": 0, "final_loss": 0.0}
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.bc_lr)
    n = int(obs_t.shape[0])
    final_loss = 0.0
    actor.train()
    for _ in range(int(args.bc_epochs)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, int(args.bc_batch_size)):
            idx = perm[start : start + int(args.bc_batch_size)]
            pred = actor(obs_t[idx])
            loss = F.mse_loss(pred, act_t[idx])
            loss = loss + args.bc_action_reg * pred.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
            opt.step()
            final_loss = float(loss.detach().cpu())
    actor.eval()
    with torch.no_grad():
        mse = F.mse_loss(actor(obs_t), act_t).detach().cpu()
    return {
        "samples": int(n),
        "epochs": int(args.bc_epochs),
        "batch_size": int(args.bc_batch_size),
        "final_loss": float(final_loss),
        "dataset_mse": float(mse),
    }


def planned_duration(args: argparse.Namespace) -> float:
    _, spec = load_short_primitive_spec(args.reward_plan_path, args.primitive_label)
    return float(args.episode_seconds or spec.duration_s)


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) / args.primitive_label
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args, seed=args.seed)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    duration_s = float(env.max_episode_steps / env.control_hz)

    actor = Actor(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    actor_target = Actor(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    if args.scripted_residual_scale > 0.0 and not args.init_actor_checkpoint:
        for network in (actor, actor_target):
            last = network.net[-2]
            if isinstance(last, torch.nn.Linear):
                torch.nn.init.zeros_(last.weight)
                torch.nn.init.zeros_(last.bias)
    if args.init_actor_checkpoint and Path(args.init_actor_checkpoint).exists():
        ckpt = torch.load(
            args.init_actor_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if int(ckpt.get("obs_dim", -1)) == obs_dim and int(ckpt.get("action_dim", -1)) == action_dim:
            init_actor = load_actor_checkpoint(
                Path(args.init_actor_checkpoint),
                obs_dim,
                action_dim,
                args.hidden_dim,
                device,
            )
            actor.load_state_dict(init_actor.state_dict())
    actor_target.load_state_dict(actor.state_dict())
    critic = Critic(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    critic_target = Critic(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(obs_dim, action_dim, args.replay_size, device)

    zero_eval, zero_rows, zero_trace = evaluate_policy(args, zero_action, render_first=True)
    scripted_eval, scripted_rows, scripted_trace = evaluate_policy(args, scripted_target_action, render_first=True)
    initial_eval, initial_rows, _ = evaluate_policy(args, actor_policy(actor, device, 0.0), render_first=False)
    save_trace_artifacts(output_dir, "zero_action_baseline", zero_trace)
    save_trace_artifacts(output_dir, "scripted_target_baseline", scripted_trace)
    write_csv(output_dir / "zero_action_per_episode.csv", zero_rows)
    write_csv(output_dir / "scripted_target_per_episode.csv", scripted_rows)
    write_csv(output_dir / "initial_actor_per_episode.csv", initial_rows)

    bc_stats: dict[str, Any] = {"samples": 0, "epochs": 0, "final_loss": 0.0}
    bc_eval: dict[str, Any] | None = None
    if args.bc_epochs > 0 and args.bc_episodes > 0:
        bc_obs, bc_actions = collect_bc_dataset(env, args, args.bc_episodes)
        bc_stats = behavior_clone_actor(actor, bc_obs, bc_actions, args, device)
        actor_target.load_state_dict(actor.state_dict())
        actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        bc_eval, bc_rows, bc_trace = evaluate_policy(args, actor_policy(actor, device, 0.0), render_first=True)
        save_trace_artifacts(output_dir, "bc_actor_warmstart", bc_trace)
        write_csv(output_dir / "bc_actor_per_episode.csv", bc_rows)

    best_score = float((bc_eval or initial_eval)["score_mean"])
    best_eval = {**(bc_eval or initial_eval), "label": "bc_actor" if bc_eval is not None else "initial_actor", "step": 0}
    save_checkpoint(
        output_dir / "td3_short_primitive_best.pt",
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        args=args,
        step=0,
        best_score=best_score,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    prefill_scripted = prefill_replay(env, replay, args, scripted_target_action, args.scripted_prefill_episodes)
    prefill_random = prefill_replay(env, replay, args, random_action, args.random_prefill_episodes)
    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_infos: list[dict[str, Any]] = []
    episode_scores: deque[float] = deque(maxlen=20)
    eval_rows: list[dict[str, Any]] = []
    start_time = time.time()

    print(
        f"[M3.2.24] {args.primitive_label} "
        f"zero={zero_eval['score_mean']:.2f} scripted={scripted_eval['score_mean']:.2f} "
        f"initial_actor={initial_eval['score_mean']:.2f} "
        f"bc={bc_stats.get('dataset_mse', 0.0):.5f}/{bc_stats.get('samples', 0)} "
        f"prefill={prefill_scripted + prefill_random}",
        flush=True,
    )

    for step in range(1, args.total_steps + 1):
        policy_action = select_action(actor, obs, device, noise_std=args.exploration_noise)
        env_action = compose_environment_action(
            policy_action,
            obs,
            env,
            len(ep_infos),
            args,
        )
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done = bool(terminated or truncated)
        replay.add(obs, policy_action, reward, next_obs, done)
        ep_return += float(reward)
        ep_infos.append(info)
        obs = next_obs

        if replay.size >= args.batch_size and step >= args.update_after:
            for _ in range(args.updates_per_step):
                b_obs, b_action, b_reward, b_next_obs, b_done = replay.sample(args.batch_size)
                with torch.no_grad():
                    noise_t = torch.randn_like(b_action) * args.target_noise
                    noise_t = torch.clamp(noise_t, -args.target_noise_clip, args.target_noise_clip)
                    next_action = torch.clamp(actor_target(b_next_obs) + noise_t, -1.0, 1.0)
                    target_q1, target_q2 = critic_target(b_next_obs, next_action)
                    target_q = torch.minimum(target_q1, target_q2)
                    target = b_reward + (1.0 - b_done) * args.gamma * target_q

                q1, q2 = critic(b_obs, b_action)
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
                critic_opt.step()

                if step % args.policy_delay == 0:
                    pred_action = actor(b_obs)
                    actor_loss = -critic.q1_value(b_obs, pred_action).mean()
                    actor_loss = actor_loss + args.actor_action_reg * pred_action.pow(2).mean()
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
                    actor_opt.step()
                    soft_update(actor_target, actor, args.tau)
                    soft_update(critic_target, critic, args.tau)

        if done:
            ep_summary = summarize_short_primitive_rollout(ep_infos, ep_return, len(ep_infos), env.control_hz)
            episode_scores.append(primitive_score(ep_summary, duration_s))
            obs, _ = env.reset()
            ep_return = 0.0
            ep_infos = []

        if step % args.eval_interval == 0 or step == args.total_steps:
            eval_row, per_episode, _ = evaluate_policy(args, actor_policy(actor, device, 0.0), render_first=False)
            eval_row["step"] = int(step)
            eval_row["elapsed_seconds"] = float(time.time() - start_time)
            eval_row["replay_size"] = int(replay.size)
            eval_row["train_score20_mean"] = float(np.mean(episode_scores)) if episode_scores else 0.0
            eval_rows.append(eval_row)
            write_csv(output_dir / f"eval_per_episode_step_{step}.csv", per_episode)
            print(
                f"[M3.2.24] {args.primitive_label} step={step:5d} "
                f"score={eval_row['score_mean']:.2f} "
                f"survival={eval_row['survival_seconds_mean']:.2f}s "
                f"target_final={eval_row['target_final_window_ratio_mean']:.2f} "
                f"desired={eval_row['desired_ratio_mean_mean']:.2f} "
                f"pinky={eval_row['pinky_ratio_mean']:.2f}",
                flush=True,
            )
            if float(eval_row["score_mean"]) > best_score:
                best_score = float(eval_row["score_mean"])
                best_eval = {**eval_row, "label": "td3_actor"}
                save_checkpoint(
                    output_dir / "td3_short_primitive_best.pt",
                    actor=actor,
                    critic=critic,
                    actor_target=actor_target,
                    critic_target=critic_target,
                    actor_opt=actor_opt,
                    critic_opt=critic_opt,
                    args=args,
                    step=step,
                    best_score=best_score,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                )

    save_checkpoint(
        output_dir / "td3_short_primitive_final.pt",
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        args=args,
        step=args.total_steps,
        best_score=best_score,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    env.close()

    best_actor = load_actor_checkpoint(
        output_dir / "td3_short_primitive_best.pt",
        obs_dim,
        action_dim,
        args.hidden_dim,
        device,
    )
    best_eval_final, best_rows, best_trace = evaluate_policy(args, actor_policy(best_actor, device, 0.0), render_first=True)
    save_trace_artifacts(output_dir, "best_td3_actor", best_trace)
    write_csv(output_dir / "training_log.csv", eval_rows)
    write_csv(output_dir / "best_td3_actor_per_episode.csv", best_rows)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "M3.2.24 short primitive TD3 smoke",
        "algorithm": "TD3-lite on MuJoCo short primitive wrapper",
        "primitive_label": args.primitive_label,
        "cache_window": args.cache_window or CACHE_WINDOW_BY_PRIMITIVE.get(args.primitive_label, ""),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "duration_s": duration_s,
        "action_scale": args.action_scale,
        "assist_mode": args.assist_mode,
        "assist_delta_limit": args.assist_delta_limit,
        "actuator_kp": args.actuator_kp,
        "actuator_kv": args.actuator_kv,
        "action_delta_limit": args.action_delta_limit,
        "tip_proxy_radius": args.tip_proxy_radius,
        "tip_proxy_friction": args.tip_proxy_friction,
        "total_steps": args.total_steps,
        "scripted_prefill_steps": prefill_scripted,
        "scripted_action_file": args.scripted_action_file,
        "scripted_action_key": args.scripted_action_key,
        "scripted_residual_scale": args.scripted_residual_scale,
        "random_prefill_steps": prefill_random,
        "zero_action": zero_eval,
        "scripted_target": scripted_eval,
        "initial_actor": initial_eval,
        "bc_stats": bc_stats,
        "bc_actor": bc_eval,
        "best_eval": best_eval,
        "best_td3_actor": best_eval_final,
        "artifacts": {
            "best_checkpoint": str(output_dir / "td3_short_primitive_best.pt"),
            "final_checkpoint": str(output_dir / "td3_short_primitive_final.pt"),
            "training_log": str(output_dir / "training_log.csv"),
            "scripted_trace": str(output_dir / "scripted_target_baseline_trace.png"),
            "scripted_gif": str(output_dir / "scripted_target_baseline_rollout.gif"),
            "best_trace": str(output_dir / "best_td3_actor_trace.png"),
            "best_gif": str(output_dir / "best_td3_actor_rollout.gif"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["best_td3_actor"], indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive-label", default="P4_to_P5_thumb_index_capture")
    parser.add_argument("--reward-plan-path", default=str(DEFAULT_REWARD_PLAN))
    parser.add_argument("--cache-manifest-path", default=str(DEFAULT_CACHE_MANIFEST))
    parser.add_argument("--cache-window", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m3_2_24_short_primitive_td3")
    parser.add_argument("--init-actor-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--action-scale", type=float, default=0.012)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--assist-mode", choices=["none", "auto", "pre_ring"], default="none")
    parser.add_argument("--assist-delta-limit", type=float, default=0.006)
    parser.add_argument("--actuator-kp", type=float, default=None)
    parser.add_argument("--actuator-kv", type=float, default=None)
    parser.add_argument("--action-delta-limit", type=float, default=None)
    parser.add_argument("--tip-proxy-radius", type=float, default=None)
    parser.add_argument("--tip-proxy-friction", type=float, default=None)
    parser.add_argument("--reset-hand-noise", type=float, default=0.0)
    parser.add_argument("--reset-pen-pos-noise", type=float, default=0.0)
    parser.add_argument("--reset-pen-quat-noise", type=float, default=0.0)
    parser.add_argument("--total-steps", type=int, default=300)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-after", type=int, default=64)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-all-cache-states", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.10)
    parser.add_argument("--target-noise", type=float, default=0.04)
    parser.add_argument("--target-noise-clip", type=float, default=0.10)
    parser.add_argument("--actor-action-reg", type=float, default=0.6)
    parser.add_argument("--scripted-prefill-episodes", type=int, default=2)
    parser.add_argument("--scripted-action-file", default="")
    parser.add_argument("--scripted-action-key", default="")
    parser.add_argument("--scripted-residual-scale", type=float, default=0.0)
    parser.add_argument("--random-prefill-episodes", type=int, default=1)
    parser.add_argument("--bc-episodes", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--bc-lr", type=float, default=8e-4)
    parser.add_argument("--bc-action-reg", type=float, default=0.02)
    parser.add_argument("--scripted-tracking-gain", type=float, default=0.68)
    parser.add_argument("--scripted-thumb-roll-bias", type=float, default=0.015)
    parser.add_argument("--scripted-thumb-pitch-bias", type=float, default=0.040)
    parser.add_argument("--scripted-thumb-tip-bias", type=float, default=0.030)
    parser.add_argument("--scripted-ring-open-bias", type=float, default=0.025)
    parser.add_argument("--scripted-ring-guard-bias", type=float, default=0.015)
    parser.add_argument("--scripted-pinky-retreat-bias", type=float, default=0.080)
    parser.add_argument("--scripted-index-roll-bias", type=float, default=0.00)
    parser.add_argument("--scripted-index-pip-bias", type=float, default=0.40)
    parser.add_argument("--scripted-index-mid-bias", type=float, default=0.00)
    parser.add_argument("--scripted-index-late-roll-bias", type=float, default=0.05)
    parser.add_argument("--scripted-index-late-pip-bias", type=float, default=0.70)
    parser.add_argument("--scripted-recapture-thumb-roll-bias", type=float, default=0.000)
    parser.add_argument("--scripted-recapture-thumb-pitch-bias", type=float, default=0.000)
    parser.add_argument("--scripted-recapture-thumb-tip-bias", type=float, default=0.000)
    parser.add_argument("--scripted-recapture-ring-open-bias", type=float, default=0.050)
    parser.add_argument("--scripted-recapture-ramp-fraction", type=float, default=0.45)
    parser.add_argument("--render-every", type=int, default=3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
