from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dex5_penspin_env import Dex5PenSpinMinimalEnv


def random_policy(env: Dex5PenSpinMinimalEnv, scale: float) -> np.ndarray:
    return env.np_random.uniform(-scale, scale, size=env.action_space.shape).astype(np.float32)


def hold_policy(env: Dex5PenSpinMinimalEnv) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def rollout(env: Dex5PenSpinMinimalEnv, policy: str, action_scale: float) -> dict:
    obs, info = env.reset()
    observations = [obs]
    rewards = []
    qpos = [info["qpos"]]
    target_qpos = [info["target_qpos"]]
    pen_center = [info["pen_center"]]
    obj_ends = [info["obj_ends"]]
    rotation_phase = [info["rotation_phase"]]
    actions = []
    reward_terms = []

    terminated = False
    truncated = False
    while not (terminated or truncated):
        if policy == "hold":
            action = hold_policy(env)
        elif policy == "random":
            action = random_policy(env, action_scale)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        actions.append(action)
        qpos.append(info["qpos"])
        target_qpos.append(info["target_qpos"])
        pen_center.append(info["pen_center"])
        obj_ends.append(info["obj_ends"])
        rotation_phase.append(info["rotation_phase"])
        reward_terms.append(info["reward_terms"])

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "qpos": np.asarray(qpos, dtype=np.float32),
        "target_qpos": np.asarray(target_qpos, dtype=np.float32),
        "pen_center": np.asarray(pen_center, dtype=np.float32),
        "obj_ends": np.asarray(obj_ends, dtype=np.float32),
        "rotation_phase": np.asarray(rotation_phase, dtype=np.float32),
        "reward_terms": reward_terms,
        "return": float(np.sum(rewards)),
        "steps": int(len(rewards)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def plot_returns(returns: list[float], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(returns)), returns, marker="o", linewidth=1.6)
    ax.axhline(float(np.mean(returns)), color="tab:red", linestyle="--", label=f"mean={np.mean(returns):.2f}")
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pen_trajectory(trajectory: dict, output_path: Path, title: str) -> None:
    center = trajectory["pen_center"]
    phase = trajectory["rotation_phase"]
    t = np.arange(len(center))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, center[:, 0], label="x")
    axes[0].plot(t, center[:, 1], label="y")
    axes[0].plot(t, center[:, 2], label="z")
    axes[0].set_ylabel("pen center (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(title)

    axes[1].plot(t, phase, label="rotation_phase")
    axes[1].plot(t, phase / (2.0 * np.pi), label="rotation_rounds")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("phase / rounds")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_render(env: Dex5PenSpinMinimalEnv, output_path: Path) -> None:
    frame = env.render()
    plt.imsave(output_path, frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--policy", choices=["random", "hold"], default="random")
    parser.add_argument("--action-scale", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="dex5_penspin/reports/m2_minimal_sim")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--pen-init-pos", nargs=3, type=float, default=None)
    parser.add_argument("--grasp-scale", type=float, default=1.0)
    parser.add_argument("--init-noise-scale", type=float, default=0.015)
    parser.add_argument("--render-frame", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        episode_seconds=args.episode_seconds,
        pen_init_pos=args.pen_init_pos,
        grasp_scale=args.grasp_scale,
        init_noise_scale=args.init_noise_scale,
        seed=args.seed,
    )
    print("Dex5-PenSpin minimal sim")
    print(f"model_path: {env.model_path}")
    print(f"nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")
    print(f"obs_dim={env.observation_space.shape[0]}, action_dim={env.action_space.shape[0]}")
    print(f"control_hz={env.control_hz}, frame_skip={env.frame_skip}, max_episode_steps={env.max_episode_steps}")

    returns = []
    lengths = []
    first_trajectory = None
    for episode in range(args.episodes):
        trajectory = rollout(env, args.policy, args.action_scale)
        returns.append(trajectory["return"])
        lengths.append(trajectory["steps"])
        if first_trajectory is None:
            first_trajectory = trajectory
        print(
            f"episode {episode:02d}: return={trajectory['return']:.3f}, "
            f"steps={trajectory['steps']}, terminated={trajectory['terminated']}"
        )

    assert first_trajectory is not None
    np.savez_compressed(
        output_dir / f"{args.policy}_rollout_episode0.npz",
        observations=first_trajectory["observations"],
        rewards=first_trajectory["rewards"],
        actions=first_trajectory["actions"],
        qpos=first_trajectory["qpos"],
        target_qpos=first_trajectory["target_qpos"],
        pen_center=first_trajectory["pen_center"],
        obj_ends=first_trajectory["obj_ends"],
        rotation_phase=first_trajectory["rotation_phase"],
    )
    plot_returns(returns, output_dir / f"{args.policy}_returns.png", f"Dex5-PenSpin Minimal {args.policy} baseline")
    plot_pen_trajectory(
        first_trajectory,
        output_dir / f"{args.policy}_pen_trajectory.png",
        f"Dex5-PenSpin Minimal {args.policy} episode 0",
    )
    if args.render_frame:
        env.reset(seed=args.seed)
        save_render(env, output_dir / "initial_render.png")

    summary = {
        "policy": args.policy,
        "episodes": args.episodes,
        "episode_seconds": args.episode_seconds,
        "action_scale": args.action_scale,
        "pen_init_pos": args.pen_init_pos,
        "grasp_scale": args.grasp_scale,
        "init_noise_scale": args.init_noise_scale,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "length_mean": float(np.mean(lengths)),
        "model_path": str(env.model_path),
        "nq": int(env.model.nq),
        "nv": int(env.model.nv),
        "nu": int(env.model.nu),
        "obs_dim": int(env.observation_space.shape[0]),
        "action_dim": int(env.action_space.shape[0]),
    }
    with (output_dir / f"{args.policy}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    env.close()
    print(f"saved results to: {output_dir}")


if __name__ == "__main__":
    main()
