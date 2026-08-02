from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
import torch

from m3_oracle_hold_env import DEFAULT_PRESET, Dex5PenSpinOracleHoldEnv
from train_m3_1_hold_oracle import load_actor, select_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preset-path", default=str(DEFAULT_PRESET))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--action-scale", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--pause-on-end", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Dex5PenSpinOracleHoldEnv(
        preset_path=args.preset_path,
        model_path=args.model_path,
        episode_seconds=args.episode_seconds,
        control_hz=args.control_hz,
        action_scale=args.action_scale,
        seed=args.seed,
        render_mode=None,
    )
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    actor = load_actor(Path(args.checkpoint), obs_dim, action_dim, device)

    obs, _ = env.reset(seed=args.seed)
    dt = 1.0 / env.control_hz
    step = 0
    total_reward = 0.0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and step < env.max_episode_steps:
            t0 = time.time()
            action = select_action(actor, obs, device, noise_std=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step += 1
            viewer.sync()
            if terminated or truncated:
                break
            if args.realtime:
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        print(
            "playback finished: "
            f"steps={step} return={total_reward:.3f} "
            f"tripod={info.get('tripod')} xy={info.get('xy_drift', np.nan):.4f}"
        )
        if args.pause_on_end:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)
    env.close()


if __name__ == "__main__":
    main()
