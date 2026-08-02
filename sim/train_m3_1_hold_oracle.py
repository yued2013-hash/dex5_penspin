from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from m3_oracle_hold_env import DEFAULT_PRESET, Dex5PenSpinOracleHoldEnv, summarize_rollout


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )
        last = self.net[-2]
        if isinstance(last, nn.Linear):
            nn.init.uniform_(last.weight, -1e-3, 1e-3)
            nn.init.zeros_(last.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device) -> None:
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr, 0] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.action[idx], device=self.device),
            torch.as_tensor(self.reward[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.done[idx], device=self.device),
        )


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def make_env(args: argparse.Namespace, *, render: bool = False, seed: int | None = None) -> Dex5PenSpinOracleHoldEnv:
    return Dex5PenSpinOracleHoldEnv(
        preset_path=args.preset_path,
        model_path=args.model_path,
        episode_seconds=args.episode_seconds,
        control_hz=args.control_hz,
        action_scale=args.action_scale,
        seed=seed,
        render_mode="rgb_array" if render else None,
        reset_hand_noise=args.reset_hand_noise,
        reset_pen_pos_noise=args.reset_pen_pos_noise,
        reset_pen_quat_noise=args.reset_pen_quat_noise,
    )


def select_action(actor: Actor, obs: np.ndarray, device: torch.device, noise_std: float = 0.0) -> np.ndarray:
    actor.eval()
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = actor(obs_t).cpu().numpy()[0]
    actor.train()
    if noise_std > 0.0:
        action = action + np.random.normal(0.0, noise_std, size=action.shape)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def zero_action(obs: np.ndarray, action_dim: int = 20) -> np.ndarray:
    return np.zeros(action_dim, dtype=np.float32)


def random_action(obs: np.ndarray, action_dim: int = 20) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)


def rollout(
    env: Dex5PenSpinOracleHoldEnv,
    policy_fn,
    *,
    render: bool = False,
    render_every: int = 4,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    obs, _ = env.reset()
    infos: list[dict[str, Any]] = []
    observations = [obs.copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    centers: list[np.ndarray] = []
    phases: list[float] = []
    endpoint_dz: list[float] = []
    frames: list[np.ndarray] = []
    total_reward = 0.0
    step = 0
    if render:
        frames.append(env.render())
    while True:
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        infos.append(info)
        observations.append(next_obs.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        centers.append(np.asarray(info["pen_center"], dtype=np.float64))
        phases.append(float(info["rotation_phase"]))
        endpoint_dz.append(float(info["endpoint_height_diff"]))
        total_reward += float(reward)
        step += 1
        obs = next_obs
        if render and step % render_every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    summary = summarize_rollout(infos, total_reward, step, env.control_hz)
    summary["terminated"] = bool(infos[-1].get("failed", False)) if infos else False
    traces = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "centers": np.asarray(centers, dtype=np.float64),
        "phases": np.asarray(phases, dtype=np.float64),
        "endpoint_height_diffs": np.asarray(endpoint_dz, dtype=np.float64),
    }
    if render:
        traces["frames"] = np.asarray(frames, dtype=np.uint8)
    return summary, traces


def evaluate_actor(
    args: argparse.Namespace,
    actor: Actor,
    device: torch.device,
    episodes: int,
    seed_offset: int,
) -> dict[str, Any]:
    rows = []
    env = make_env(args, seed=args.seed + seed_offset)
    try:
        for _ in range(episodes):
            row, _ = rollout(env, lambda obs: select_action(actor, obs, device, noise_std=0.0))
            rows.append(row)
    finally:
        env.close()
    keys = [
        "return",
        "survival_seconds",
        "tripod_ratio",
        "thumb_ratio",
        "xy_drift_max",
        "endpoint_height_diff_max",
        "phase_abs_max",
        "action_norm_mean",
    ]
    summary = {f"{k}_mean": float(np.mean([row[k] for row in rows])) for k in keys}
    summary.update({f"{k}_std": float(np.std([row[k] for row in rows])) for k in keys})
    summary["terminated_mean"] = float(np.mean([float(row.get("terminated", False)) for row in rows]))
    summary["episodes"] = int(episodes)
    survival_fraction = summary["survival_seconds_mean"] / max(float(args.episode_seconds), 1e-6)
    # Hold score is survival-first. Without this, a policy can briefly create
    # good contacts and then fail, which is exactly the wrong behavior for M3.1.
    summary["hold_score"] = (
        1200.0 * survival_fraction
        + summary["return_mean"]
        + 120.0 * summary["tripod_ratio_mean"]
        - 250.0 * summary["terminated_mean"]
        - 120.0 * summary["xy_drift_max_mean"]
        - 40.0 * summary["endpoint_height_diff_max_mean"]
        - 4.0 * summary["action_norm_mean_mean"]
    )
    return summary


def save_checkpoint(
    path: Path,
    *,
    actor: Actor,
    critic: Critic,
    actor_target: Actor,
    critic_target: Critic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    best_score: float,
    obs_dim: int,
    action_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_target": actor_target.state_dict(),
            "critic_target": critic_target.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "step": int(step),
            "best_score": float(best_score),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "config": vars(args),
        },
        path,
    )


def load_actor(path: Path, obs_dim: int, action_dim: int, device: torch.device) -> Actor:
    ckpt = torch.load(path, map_location=device)
    actor = Actor(obs_dim, action_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rollout(path: Path, traces: dict[str, np.ndarray], title: str) -> None:
    centers = traces["centers"]
    phases = traces["phases"]
    endpoint = traces["endpoint_height_diffs"]
    rewards = traces["rewards"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(centers[:, 0], label="x")
    axes[0].plot(centers[:, 1], label="y")
    axes[0].plot(centers[:, 2], label="z")
    axes[0].set_ylabel("pen center")
    axes[0].legend()
    axes[1].plot(phases)
    axes[1].set_ylabel("rotation phase")
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


def save_rollout_artifacts(
    args: argparse.Namespace,
    output_dir: Path,
    name: str,
    policy_fn,
    *,
    render: bool = True,
) -> dict[str, Any]:
    env = make_env(args, render=render, seed=args.seed + 9000)
    try:
        summary, traces = rollout(env, policy_fn, render=render, render_every=args.render_every)
    finally:
        env.close()
    np.savez_compressed(
        output_dir / f"{name}_rollout.npz",
        observations=traces["observations"],
        actions=traces["actions"],
        rewards=traces["rewards"],
        centers=traces["centers"],
        phases=traces["phases"],
        endpoint_height_diffs=traces["endpoint_height_diffs"],
    )
    plot_rollout(output_dir / f"{name}_trace.png", traces, f"M3.1 {name} rollout")
    if render and "frames" in traces:
        imageio.mimsave(output_dir / f"{name}_rollout.gif", list(traces["frames"]), fps=25)
    return summary


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
    critic = Critic(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    critic_target = Critic(obs_dim, action_dim, hidden=args.hidden_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(obs_dim, action_dim, args.replay_size, device)

    log_rows: list[dict[str, Any]] = []
    episode_returns: deque[float] = deque(maxlen=20)
    episode_tripod: deque[float] = deque(maxlen=20)
    best_score = -float("inf")
    best_eval: dict[str, Any] | None = None

    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_infos: list[dict[str, Any]] = []
    start_time = time.time()

    initial_eval = evaluate_actor(args, actor, device, args.eval_episodes, seed_offset=0)
    initial_eval["step"] = 0
    initial_eval["elapsed_seconds"] = 0.0
    initial_eval["replay_size"] = 0
    initial_eval["train_return20_mean"] = 0.0
    initial_eval["train_tripod20_mean"] = 0.0
    log_rows.append(initial_eval)
    best_score = float(initial_eval["hold_score"])
    best_eval = initial_eval
    save_checkpoint(
        output_dir / "td3_hold_oracle_best.pt",
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
    print(
        f"[M3.1] step={0:6d} "
        f"eval_return={initial_eval['return_mean']:.2f} "
        f"survival={initial_eval['survival_seconds_mean']:.2f}s "
        f"tripod={initial_eval['tripod_ratio_mean']:.3f} "
        f"score={initial_eval['hold_score']:.2f}",
        flush=True,
    )

    for step in range(1, args.total_steps + 1):
        noise = args.exploration_noise
        action = select_action(actor, obs, device, noise_std=noise)
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
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
                critic_opt.step()

                if step % args.policy_delay == 0:
                    pred_action = actor(b_obs)
                    actor_loss = -critic.q1_value(b_obs, pred_action).mean()
                    actor_loss = actor_loss + args.actor_action_reg * pred_action.pow(2).mean()
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
                    actor_opt.step()
                    soft_update(actor_target, actor, args.tau)
                    soft_update(critic_target, critic, args.tau)

        if done:
            ep_summary = summarize_rollout(ep_infos, ep_return, len(ep_infos), env.control_hz)
            episode_returns.append(ep_summary["return"])
            episode_tripod.append(ep_summary["tripod_ratio"])
            obs, _ = env.reset()
            ep_return = 0.0
            ep_infos = []

        if step % args.eval_interval == 0 or step == args.total_steps:
            eval_row = evaluate_actor(args, actor, device, args.eval_episodes, seed_offset=step)
            eval_row["step"] = int(step)
            eval_row["elapsed_seconds"] = float(time.time() - start_time)
            eval_row["replay_size"] = int(replay.size)
            eval_row["train_return20_mean"] = float(np.mean(episode_returns)) if episode_returns else 0.0
            eval_row["train_tripod20_mean"] = float(np.mean(episode_tripod)) if episode_tripod else 0.0
            log_rows.append(eval_row)
            print(
                f"[M3.1] step={step:6d} "
                f"eval_return={eval_row['return_mean']:.2f} "
                f"survival={eval_row['survival_seconds_mean']:.2f}s "
                f"tripod={eval_row['tripod_ratio_mean']:.3f} "
                f"xy_max={eval_row['xy_drift_max_mean']:.4f} "
                f"score={eval_row['hold_score']:.2f}",
                flush=True,
            )
            if eval_row["hold_score"] > best_score:
                best_score = float(eval_row["hold_score"])
                best_eval = eval_row
                save_checkpoint(
                    output_dir / "td3_hold_oracle_best.pt",
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
        output_dir / "td3_hold_oracle_final.pt",
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
    write_training_log(output_dir / "training_log.csv", log_rows)
    env.close()

    best_actor = load_actor(output_dir / "td3_hold_oracle_best.pt", obs_dim, action_dim, device)
    zero_summary = save_rollout_artifacts(args, output_dir, "zero_action_baseline", lambda obs: zero_action(obs, action_dim))
    random_summary = save_rollout_artifacts(args, output_dir, "random_action_baseline", lambda obs: random_action(obs, action_dim), render=False)
    best_summary = save_rollout_artifacts(
        args,
        output_dir,
        "best_hold_oracle",
        lambda obs: select_action(best_actor, obs, device, noise_std=0.0),
    )

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "M3.1 hold oracle",
        "algorithm": "TD3-lite",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "action_scale": args.action_scale,
        "preset_path": str(args.preset_path),
        "total_steps": args.total_steps,
        "best_eval": best_eval,
        "zero_action_baseline": zero_summary,
        "random_action_baseline": random_summary,
        "best_hold_oracle": best_summary,
        "artifacts": {
            "best_checkpoint": str(output_dir / "td3_hold_oracle_best.pt"),
            "final_checkpoint": str(output_dir / "td3_hold_oracle_final.pt"),
            "training_log": str(output_dir / "training_log.csv"),
            "best_rollout": str(output_dir / "best_hold_oracle_rollout.npz"),
            "best_trace": str(output_dir / "best_hold_oracle_trace.png"),
            "best_gif": str(output_dir / "best_hold_oracle_rollout.gif"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    viewer_script = output_dir / "play_best_hold_oracle.sh"
    viewer_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n\n"
        "python dex5_penspin/sim/play_m3_1_hold_oracle.py "
        f"--checkpoint {output_dir / 'td3_hold_oracle_best.pt'} "
        f"--preset-path {args.preset_path} "
        f"--episode-seconds {args.episode_seconds} "
        f"--action-scale {args.action_scale} "
        "--realtime --pause-on-end\n",
        encoding="utf-8",
    )
    viewer_script.chmod(0o755)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset-path", default=str(DEFAULT_PRESET))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m3_1_hold_oracle")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--action-scale", type=float, default=8e-4)
    parser.add_argument("--reset-hand-noise", type=float, default=0.0)
    parser.add_argument("--reset-pen-pos-noise", type=float, default=0.0)
    parser.add_argument("--reset-pen-quat-noise", type=float, default=0.0)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--update-after", type=int, default=500)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.15)
    parser.add_argument("--target-noise", type=float, default=0.08)
    parser.add_argument("--target-noise-clip", type=float, default=0.20)
    parser.add_argument("--actor-action-reg", type=float, default=0.25)
    parser.add_argument("--render-every", type=int, default=4)
    args = parser.parse_args()

    summary = train(args)
    print(json.dumps(summary["best_hold_oracle"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
