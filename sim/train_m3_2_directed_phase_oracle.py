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

from m3_2_directed_phase_env import DEFAULT_MANIFEST, Dex5PenSpinDirectedPhaseEnv, summarize_directed_rollout
from train_m3_1_hold_oracle import Actor, Critic, ReplayBuffer, load_actor, save_checkpoint, select_action, soft_update
from view_m2_mujoco import action_for_policy


def make_env(args: argparse.Namespace, *, render: bool = False, seed: int | None = None) -> Dex5PenSpinDirectedPhaseEnv:
    return Dex5PenSpinDirectedPhaseEnv(
        manifest_path=args.manifest_path,
        model_path=args.model_path,
        episode_seconds=args.episode_seconds,
        control_hz=args.control_hz,
        action_scale=args.action_scale,
        seed=seed,
        render_mode="rgb_array" if render else None,
        reset_hand_noise=args.reset_hand_noise,
        reset_pen_pos_noise=args.reset_pen_pos_noise,
        reset_pen_quat_noise=args.reset_pen_quat_noise,
        phase_direction=args.phase_direction,
        phase_reward_scale=args.phase_reward_scale,
        negative_phase_penalty=args.negative_phase_penalty,
        hold_reward_scale=args.hold_reward_scale,
        target_progress=args.target_progress,
    )


def zero_action(_: np.ndarray, env: Dex5PenSpinDirectedPhaseEnv, step: int, args: argparse.Namespace) -> np.ndarray:
    return np.zeros(20, dtype=np.float32)


def random_action(_: np.ndarray, env: Dex5PenSpinDirectedPhaseEnv, step: int, args: argparse.Namespace) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=20).astype(np.float32)


def scripted_directed_action(_: np.ndarray, env: Dex5PenSpinDirectedPhaseEnv, step: int, args: argparse.Namespace) -> np.ndarray:
    # action_for_policy returns a real joint delta. Convert it back to the
    # normalized action expected by the M3 wrapper.
    delta = action_for_policy(
        env.env,
        "directed_micro_gait",
        step,
        args.scripted_action_scale,
        phase_preset=args.scripted_phase_preset,
        center_gain=args.scripted_center_gain,
        target_center=env.initial_center,
        cycle_steps=args.scripted_cycle_steps,
        thumb_scale=args.scripted_thumb_scale,
        phase_direction=args.phase_direction,
        return_scale=args.scripted_return_scale,
        roll_scale=args.scripted_roll_scale,
    )
    return np.clip(delta / max(args.action_scale, 1e-9), -1.0, 1.0).astype(np.float32)


def rollout(
    env: Dex5PenSpinDirectedPhaseEnv,
    policy_fn: Callable[[np.ndarray, Dex5PenSpinDirectedPhaseEnv, int, argparse.Namespace], np.ndarray],
    args: argparse.Namespace,
    *,
    reset_index: int | None = None,
    render: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    obs, _ = env.reset(options={"reset_index": reset_index} if reset_index is not None else None)
    infos: list[dict[str, Any]] = []
    observations = [obs.copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    centers: list[np.ndarray] = []
    phases: list[float] = []
    directed: list[float] = []
    endpoint_dz: list[float] = []
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
        phases.append(float(info["rotation_phase"]))
        directed.append(float(info["directed_phase"]))
        endpoint_dz.append(float(info["endpoint_height_diff"]))
        total_reward += float(reward)
        step += 1
        obs = next_obs
        if render and step % args.render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    summary = summarize_directed_rollout(infos, total_reward, step, env.control_hz)
    summary["terminated"] = bool(infos[-1].get("failed", False)) if infos else False
    traces = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "centers": np.asarray(centers, dtype=np.float64),
        "phases": np.asarray(phases, dtype=np.float64),
        "directed_phases": np.asarray(directed, dtype=np.float64),
        "endpoint_height_diffs": np.asarray(endpoint_dz, dtype=np.float64),
    }
    if render:
        traces["frames"] = np.asarray(frames, dtype=np.uint8)
    return summary, traces


def actor_policy(actor: Actor, device: torch.device, noise_std: float = 0.0):
    def _policy(obs: np.ndarray, env: Dex5PenSpinDirectedPhaseEnv, step: int, args: argparse.Namespace) -> np.ndarray:
        return select_action(actor, obs, device, noise_std=noise_std)

    return _policy


def directed_score(row: dict[str, Any], episode_seconds: float) -> float:
    survival_fraction = float(row["survival_seconds"]) / max(float(episode_seconds), 1e-6)
    return (
        1000.0 * survival_fraction
        + 800.0 * max(0.0, float(row["final_progress"]))
        + 120.0 * float(row["tripod_ratio"])
        + 80.0 * float(row["monotonic_ratio"])
        - 450.0 * float(row["negative_progress"])
        - 250.0 * float(row["max_drawdown"])
        - 220.0 * float(row["xy_drift_max"])
        - 50.0 * float(row["endpoint_height_diff_max"])
        - 60.0 * float(row["action_norm_mean"])
        - (300.0 if row.get("terminated", False) else 0.0)
    )


def evaluate_policy(
    args: argparse.Namespace,
    policy_fn,
    *,
    render_best: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray] | None]:
    env = make_env(args, render=render_best, seed=args.seed + 100)
    rows: list[dict[str, Any]] = []
    best_trace = None
    try:
        for reset_index in range(len(env.presets)):
            row, traces = rollout(env, policy_fn, args, reset_index=reset_index, render=render_best and reset_index == 0)
            row["score"] = directed_score(row, args.episode_seconds)
            rows.append(row)
            if render_best and reset_index == 0:
                best_trace = traces
    finally:
        env.close()

    keys = [
        "score",
        "return",
        "survival_seconds",
        "tripod_ratio",
        "thumb_ratio",
        "final_progress",
        "positive_progress",
        "negative_progress",
        "monotonic_ratio",
        "max_drawdown",
        "xy_drift_max",
        "endpoint_height_diff_max",
        "action_norm_mean",
    ]
    aggregate = {f"{k}_mean": float(np.mean([row[k] for row in rows])) for k in keys}
    aggregate.update({f"{k}_min": float(np.min([row[k] for row in rows])) for k in keys})
    aggregate["terminated_count"] = int(sum(bool(row.get("terminated", False)) for row in rows))
    aggregate["episodes"] = int(len(rows))
    return aggregate, rows, best_trace


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_trace(path: Path, traces: dict[str, np.ndarray], title: str) -> None:
    centers = traces["centers"]
    directed = traces["directed_phases"]
    endpoint = traces["endpoint_height_diffs"]
    rewards = traces["rewards"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(centers[:, 0], label="x")
    axes[0].plot(centers[:, 1], label="y")
    axes[0].plot(centers[:, 2], label="z")
    axes[0].set_ylabel("pen center")
    axes[0].legend()
    axes[1].plot(directed)
    axes[1].set_ylabel("directed phase")
    axes[2].plot(endpoint)
    axes[2].set_ylabel("endpoint dz")
    axes[3].plot(rewards)
    axes[3].set_ylabel("reward")
    axes[3].set_xlabel("step")
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
        phases=traces["phases"],
        directed_phases=traces["directed_phases"],
        endpoint_height_diffs=traces["endpoint_height_diffs"],
    )
    plot_trace(output_dir / f"{name}_trace.png", traces, f"M3.2 {name}")
    if "frames" in traces:
        imageio.mimsave(output_dir / f"{name}_rollout.gif", list(traces["frames"]), fps=25)


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args, seed=args.seed)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    actor = Actor(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    actor_target = Actor(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    if args.init_actor_checkpoint and Path(args.init_actor_checkpoint).exists():
        init_actor = load_actor(Path(args.init_actor_checkpoint), obs_dim, action_dim, device)
        actor.load_state_dict(init_actor.state_dict())
        actor_target.load_state_dict(init_actor.state_dict())
    else:
        actor_target.load_state_dict(actor.state_dict())
    critic = Critic(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    critic_target = Critic(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(obs_dim, action_dim, args.replay_size, device)

    eval_rows: list[dict[str, Any]] = []
    episode_scores: deque[float] = deque(maxlen=20)
    best_score = -float("inf")
    best_eval: dict[str, Any] | None = None

    # Baselines before learning.
    zero_eval, zero_rows, zero_trace = evaluate_policy(args, zero_action, render_best=True)
    scripted_eval, scripted_rows, scripted_trace = evaluate_policy(args, scripted_directed_action, render_best=True)
    best_score = float(max(zero_eval["score_mean"], scripted_eval["score_mean"]))
    best_label = "zero_action" if zero_eval["score_mean"] >= scripted_eval["score_mean"] else "scripted_directed"
    best_eval = {**(zero_eval if best_label == "zero_action" else scripted_eval), "label": best_label, "step": 0}
    save_trace_artifacts(output_dir, "zero_action_baseline", zero_trace)
    save_trace_artifacts(output_dir, "scripted_directed_baseline", scripted_trace)

    save_checkpoint(
        output_dir / "td3_directed_phase_best.pt",
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

    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_infos: list[dict[str, Any]] = []
    start_time = time.time()
    print(
        f"[M3.2] baseline zero_score={zero_eval['score_mean']:.2f} "
        f"scripted_score={scripted_eval['score_mean']:.2f} best={best_label}",
        flush=True,
    )

    for step in range(1, args.total_steps + 1):
        action = select_action(actor, obs, device, noise_std=args.exploration_noise)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        replay.add(obs, action, reward, next_obs, done)
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
            ep_summary = summarize_directed_rollout(ep_infos, ep_return, len(ep_infos), env.control_hz)
            episode_scores.append(directed_score(ep_summary, args.episode_seconds))
            obs, _ = env.reset()
            ep_return = 0.0
            ep_infos = []

        if step % args.eval_interval == 0 or step == args.total_steps:
            eval_row, per_reset, _ = evaluate_policy(args, actor_policy(actor, device, noise_std=0.0), render_best=False)
            eval_row["step"] = int(step)
            eval_row["elapsed_seconds"] = float(time.time() - start_time)
            eval_row["replay_size"] = int(replay.size)
            eval_row["train_score20_mean"] = float(np.mean(episode_scores)) if episode_scores else 0.0
            eval_rows.append(eval_row)
            print(
                f"[M3.2] step={step:6d} score={eval_row['score_mean']:.2f} "
                f"survival={eval_row['survival_seconds_mean']:.2f}s "
                f"progress={eval_row['final_progress_mean']:.3f} "
                f"tripod={eval_row['tripod_ratio_mean']:.3f}",
                flush=True,
            )
            if eval_row["score_mean"] > best_score:
                best_score = float(eval_row["score_mean"])
                best_eval = {**eval_row, "label": "td3_actor"}
                save_checkpoint(
                    output_dir / "td3_directed_phase_best.pt",
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
            write_csv(output_dir / f"eval_per_reset_step_{step}.csv", per_reset)

    save_checkpoint(
        output_dir / "td3_directed_phase_final.pt",
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

    best_actor = load_actor(output_dir / "td3_directed_phase_best.pt", obs_dim, action_dim, device)
    best_actor_eval, best_actor_rows, best_actor_trace = evaluate_policy(args, actor_policy(best_actor, device, 0.0), render_best=True)
    save_trace_artifacts(output_dir, "best_td3_actor", best_actor_trace)
    write_csv(output_dir / "training_log.csv", eval_rows)
    write_csv(output_dir / "zero_action_per_reset.csv", zero_rows)
    write_csv(output_dir / "scripted_directed_per_reset.csv", scripted_rows)
    write_csv(output_dir / "best_td3_actor_per_reset.csv", best_actor_rows)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "M3.2 directed phase oracle",
        "algorithm": "TD3-lite with multi-reset directed-phase reward",
        "manifest_path": str(args.manifest_path),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "action_scale": args.action_scale,
        "phase_direction": args.phase_direction,
        "target_progress": args.target_progress,
        "total_steps": args.total_steps,
        "zero_action": zero_eval,
        "scripted_directed": scripted_eval,
        "best_eval": best_eval,
        "best_td3_actor": best_actor_eval,
        "artifacts": {
            "best_checkpoint": str(output_dir / "td3_directed_phase_best.pt"),
            "final_checkpoint": str(output_dir / "td3_directed_phase_final.pt"),
            "training_log": str(output_dir / "training_log.csv"),
            "scripted_trace": str(output_dir / "scripted_directed_baseline_trace.png"),
            "best_td3_trace": str(output_dir / "best_td3_actor_trace.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m3_2_directed_phase")
    parser.add_argument("--init-actor-checkpoint", default="dex5_penspin/reports/m3_1_hold_oracle/td3_hold_oracle_best.pt")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--action-scale", type=float, default=6e-4)
    parser.add_argument("--reset-hand-noise", type=float, default=0.0)
    parser.add_argument("--reset-pen-pos-noise", type=float, default=0.0)
    parser.add_argument("--reset-pen-quat-noise", type=float, default=0.0)
    parser.add_argument("--phase-direction", type=float, default=-1.0)
    parser.add_argument("--target-progress", type=float, default=0.20)
    parser.add_argument("--phase-reward-scale", type=float, default=45.0)
    parser.add_argument("--negative-phase-penalty", type=float, default=55.0)
    parser.add_argument("--hold-reward-scale", type=float, default=1.0)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--update-after", type=int, default=500)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.06)
    parser.add_argument("--target-noise", type=float, default=0.03)
    parser.add_argument("--target-noise-clip", type=float, default=0.08)
    parser.add_argument("--actor-action-reg", type=float, default=0.8)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--scripted-action-scale", type=float, default=0.0003)
    parser.add_argument("--scripted-phase-preset", choices=["m2_3", "quadrature", "paired", "thumb_late"], default="paired")
    parser.add_argument("--scripted-cycle-steps", type=int, default=320)
    parser.add_argument("--scripted-thumb-scale", type=float, default=0.10)
    parser.add_argument("--scripted-center-gain", type=float, default=0.005)
    parser.add_argument("--scripted-return-scale", type=float, default=1.0)
    parser.add_argument("--scripted-roll-scale", type=float, default=0.25)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
