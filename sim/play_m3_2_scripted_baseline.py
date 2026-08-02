from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from m3_2_directed_phase_env import DEFAULT_MANIFEST, Dex5PenSpinDirectedPhaseEnv
from view_m2_mujoco import action_for_policy


def scripted_action(env: Dex5PenSpinDirectedPhaseEnv, step: int, args: argparse.Namespace) -> np.ndarray:
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


def run_episode(
    env: Dex5PenSpinDirectedPhaseEnv,
    viewer,
    reset_index: int,
    args: argparse.Namespace,
) -> dict:
    obs, info = env.reset(seed=args.seed + reset_index, options={"reset_index": reset_index})
    viewer.sync()
    dt = 1.0 / env.control_hz
    total_reward = 0.0
    last_info = info
    for step in range(env.max_episode_steps):
        t0 = time.time()
        if args.policy == "zero":
            action = np.zeros(20, dtype=np.float32)
        elif args.policy == "random":
            action = np.random.uniform(-1.0, 1.0, size=20).astype(np.float32)
        else:
            action = scripted_action(env, step, args)
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        viewer.sync()
        if terminated or truncated:
            break
        if args.realtime:
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    print(
        f"reset_{reset_index:02d} finished: "
        f"steps={env.env.step_count} return={total_reward:.2f} "
        f"directed_phase={last_info.get('directed_phase', 0.0):.3f} "
        f"tripod={last_info.get('tripod')} xy={last_info.get('xy_drift', 0.0):.4f}",
        flush=True,
    )
    return last_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--policy", choices=["scripted", "zero", "random"], default="scripted")
    parser.add_argument("--reset-index", type=int, default=0)
    parser.add_argument("--cycle-resets", action="store_true")
    parser.add_argument("--episodes-per-reset", type=int, default=1)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--action-scale", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--phase-direction", type=float, default=-1.0)
    parser.add_argument("--target-progress", type=float, default=0.20)
    parser.add_argument("--phase-reward-scale", type=float, default=45.0)
    parser.add_argument("--negative-phase-penalty", type=float, default=55.0)
    parser.add_argument("--hold-reward-scale", type=float, default=1.0)
    parser.add_argument("--scripted-action-scale", type=float, default=0.0003)
    parser.add_argument("--scripted-phase-preset", choices=["m2_3", "quadrature", "paired", "thumb_late"], default="paired")
    parser.add_argument("--scripted-cycle-steps", type=int, default=320)
    parser.add_argument("--scripted-thumb-scale", type=float, default=0.10)
    parser.add_argument("--scripted-center-gain", type=float, default=0.005)
    parser.add_argument("--scripted-return-scale", type=float, default=1.0)
    parser.add_argument("--scripted-roll-scale", type=float, default=0.25)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--pause-between", type=float, default=0.8)
    parser.add_argument("--pause-on-end", action="store_true")
    args = parser.parse_args()

    env = Dex5PenSpinDirectedPhaseEnv(
        manifest_path=args.manifest_path,
        model_path=args.model_path,
        episode_seconds=args.episode_seconds,
        control_hz=args.control_hz,
        action_scale=args.action_scale,
        seed=args.seed,
        render_mode=None,
        phase_direction=args.phase_direction,
        phase_reward_scale=args.phase_reward_scale,
        negative_phase_penalty=args.negative_phase_penalty,
        hold_reward_scale=args.hold_reward_scale,
        target_progress=args.target_progress,
    )
    reset_indices = list(range(len(env.presets))) if args.cycle_resets else [args.reset_index % len(env.presets)]
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(args.episodes_per_reset):
            for reset_index in reset_indices:
                if not viewer.is_running():
                    break
                run_episode(env, viewer, reset_index, args)
                if args.pause_between > 0.0:
                    time.sleep(args.pause_between)
        if args.pause_on_end:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)
    env.close()


if __name__ == "__main__":
    main()
