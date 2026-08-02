from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from dex5_penspin_env import Dex5PenSpinMinimalEnv


DEFAULT_MODEL = "dex5_penspin/sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml"
DEFAULT_PEN_INIT_POS = (-0.025, 0.065, 0.120)


def phase_offsets_for_preset(phase_preset: str) -> dict[str, float]:
    if phase_preset == "m2_3":
        return {
            "index": 0.0,
            "middle": 2.0 * np.pi / 3.0,
            "ring": 4.0 * np.pi / 3.0,
            "thumb": np.pi,
        }
    if phase_preset == "quadrature":
        return {
            "index": 0.0,
            "middle": np.pi / 2.0,
            "ring": np.pi,
            "thumb": 3.0 * np.pi / 2.0,
        }
    if phase_preset == "paired":
        return {
            "index": 0.0,
            "middle": 0.0,
            "ring": np.pi,
            "thumb": np.pi,
        }
    if phase_preset == "thumb_late":
        return {
            "index": 0.0,
            "middle": 2.0 * np.pi / 3.0,
            "ring": 4.0 * np.pi / 3.0,
            "thumb": 3.0 * np.pi / 2.0,
        }
    raise ValueError(f"Unknown phase_preset: {phase_preset}")


def apply_center_stabilizer(
    env: Dex5PenSpinMinimalEnv,
    action: np.ndarray,
    center_gain: float,
    target_center: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    if center_gain <= 0:
        return action
    center = env.get_state().pen_center
    error = center[:2] - np.asarray(target_center, dtype=np.float64)[:2]
    corr_x = float(np.clip(center_gain * error[0], -0.002, 0.002))
    corr_y = float(np.clip(center_gain * error[1], -0.002, 0.002))

    stabilized = action.copy()
    # Experimental low-level heuristic: shift opposing flexion groups against
    # xy drift. Keep corrections tiny so this remains a stabilizer, not a policy.
    stabilized[[1, 2, 5, 6]] += corr_x
    stabilized[[9, 10, 13, 14]] -= corr_x
    stabilized[[17, 18]] -= corr_y
    stabilized[[1, 5, 9]] += 0.5 * corr_y
    return np.clip(stabilized, -env.action_delta_limit, env.action_delta_limit).astype(np.float32)


def action_for_policy(
    env: Dex5PenSpinMinimalEnv,
    policy: str,
    step: int,
    action_scale: float,
    phase_preset: str = "m2_3",
    center_gain: float = 0.0,
    target_center: tuple[float, float, float] | np.ndarray = DEFAULT_PEN_INIT_POS,
    cycle_steps: int = 220,
    thumb_scale: float = 0.25,
    phase_direction: float = 1.0,
    return_scale: float = 1.0,
    roll_scale: float = 0.25,
) -> np.ndarray:
    if policy == "hold":
        return np.zeros(20, dtype=np.float32)
    if policy == "random":
        return env.np_random.uniform(-action_scale, action_scale, size=20).astype(np.float32)
    if policy == "sine_grasp":
        action = np.zeros(20, dtype=np.float32)
        phase = 2.0 * np.pi * step / 160.0
        flex_indices = np.asarray([1, 2, 5, 6, 9, 10, 13, 14, 17, 18])
        action[flex_indices] = action_scale * np.sin(phase)
        # Bias thumb and index/middle slightly out of phase for an observable gait.
        action[[1, 2, 5, 6]] += 0.5 * action_scale * np.sin(phase + np.pi / 2.0)
        action[[17, 18]] += 0.5 * action_scale * np.sin(phase + np.pi)
        return np.clip(action, -env.action_delta_limit, env.action_delta_limit).astype(np.float32)
    if policy == "active_grasp":
        action = np.zeros(20, dtype=np.float32)
        ramp = min(1.0, step / 180.0)
        flex_indices = np.asarray([1, 2, 5, 6, 9, 10, 13, 14, 17, 18])
        action[flex_indices] = action_scale * ramp
        # Keep distal joints slightly quieter to reduce pen ejection.
        action[[3, 7, 11, 15, 19]] = 0.35 * action_scale * ramp
        return np.clip(action, -env.action_delta_limit, env.action_delta_limit).astype(np.float32)
    if policy == "finger_gait":
        action = np.zeros(20, dtype=np.float32)
        phase = 2.0 * np.pi * step / float(cycle_steps)
        offsets = phase_offsets_for_preset(phase_preset)
        groups = {
            "index": [1, 2, 3],
            "middle": [5, 6, 7],
            "ring": [9, 10, 11],
            "thumb": [17, 18, 19],
        }
        for name, indices in groups.items():
            wave = np.sin(phase + offsets[name])
            action[indices] = action_scale * wave
        action = np.clip(action, -env.action_delta_limit, env.action_delta_limit).astype(np.float32)
        return apply_center_stabilizer(env, action, center_gain, target_center)
    if policy == "micro_gait":
        action = np.zeros(20, dtype=np.float32)
        phase = 2.0 * np.pi * step / float(cycle_steps)
        offsets = phase_offsets_for_preset(phase_preset)
        support_groups = {
            "index": ([1, 2, 3], np.array([1.0, 0.75, 0.35], dtype=np.float32)),
            "middle": ([5, 6, 7], np.array([1.0, 0.75, 0.35], dtype=np.float32)),
            "ring": ([9, 10, 11], np.array([1.0, 0.75, 0.35], dtype=np.float32)),
        }
        for name, (indices, weights) in support_groups.items():
            wave = np.sin(phase + offsets[name])
            action[indices] = action_scale * wave * weights
        # Tiny lateral fingertip shifts make contact changes visible without
        # turning this into a large grasp-open command.
        action[[0, 4, 8]] = 0.25 * action_scale * np.array(
            [
                np.sin(phase + offsets["index"] + np.pi / 2.0),
                np.sin(phase + offsets["middle"] + np.pi / 2.0),
                np.sin(phase + offsets["ring"] + np.pi / 2.0),
            ],
            dtype=np.float32,
        )
        thumb_wave = np.sin(0.5 * phase + offsets["thumb"])
        action[[17, 18, 19]] = thumb_scale * action_scale * thumb_wave * np.array([0.6, 0.8, 0.35], dtype=np.float32)
        action = np.clip(action, -env.action_delta_limit, env.action_delta_limit).astype(np.float32)
        return apply_center_stabilizer(env, action, center_gain, target_center)
    if policy == "directed_micro_gait":
        action = np.zeros(20, dtype=np.float32)
        phase = float(np.sign(phase_direction) or 1.0) * 2.0 * np.pi * step / float(cycle_steps)
        offsets = phase_offsets_for_preset(phase_preset)
        support_groups = {
            "index": ([1, 2, 3], 0, np.array([1.0, 0.70, 0.30], dtype=np.float32)),
            "middle": ([5, 6, 7], 4, np.array([1.0, 0.70, 0.30], dtype=np.float32)),
            "ring": ([9, 10, 11], 8, np.array([1.0, 0.70, 0.30], dtype=np.float32)),
        }
        return_scale = float(np.clip(return_scale, 0.0, 1.2))
        for name, (flex_indices, roll_index, weights) in support_groups.items():
            wave = np.sin(phase + offsets[name])
            close_wave = max(0.0, float(wave))
            open_wave = max(0.0, float(-wave))
            ratchet_wave = close_wave - return_scale * open_wave
            action[flex_indices] = action_scale * ratchet_wave * weights
            action[roll_index] = (
                float(np.sign(phase_direction) or 1.0)
                * roll_scale
                * action_scale
                * np.cos(phase + offsets[name])
            )
        thumb_wave = np.sin(0.5 * phase + offsets["thumb"])
        action[[17, 18, 19]] = thumb_scale * action_scale * thumb_wave * np.array([0.5, 0.7, 0.25], dtype=np.float32)
        action = np.clip(action, -env.action_delta_limit, env.action_delta_limit).astype(np.float32)
        return apply_center_stabilizer(env, action, center_gain, target_center)
    raise ValueError(f"Unknown policy: {policy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--policy",
        choices=["hold", "random", "sine_grasp", "active_grasp", "finger_gait", "micro_gait", "directed_micro_gait"],
        default="hold",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--action-scale", type=float, default=0.004)
    parser.add_argument("--phase-preset", choices=["m2_3", "quadrature", "paired", "thumb_late"], default="m2_3")
    parser.add_argument("--center-gain", type=float, default=0.0)
    parser.add_argument("--cycle-steps", type=int, default=220)
    parser.add_argument("--thumb-scale", type=float, default=0.25)
    parser.add_argument("--phase-direction", type=float, default=1.0)
    parser.add_argument("--return-scale", type=float, default=1.0)
    parser.add_argument("--roll-scale", type=float, default=0.25)
    parser.add_argument("--pen-init-pos", nargs=3, type=float, default=DEFAULT_PEN_INIT_POS)
    parser.add_argument(
        "--pen-quat",
        nargs=4,
        type=float,
        default=None,
        help="Optional pen free-joint quaternion in MuJoCo order: qw qx qy qz.",
    )
    parser.add_argument(
        "--pen-qvel",
        nargs=6,
        type=float,
        default=None,
        help="Optional pen free-joint velocity: vx vy vz wx wy wz.",
    )
    parser.add_argument("--grasp-scale", type=float, default=1.4)
    parser.add_argument("--init-noise-scale", type=float, default=0.0)
    parser.add_argument(
        "--actuator-kp",
        type=float,
        default=None,
        help="Optional position-servo kp override. Useful when fingers relax during contact debugging.",
    )
    parser.add_argument(
        "--actuator-kv",
        type=float,
        default=None,
        help="Optional position-servo kv override. Useful with --actuator-kp for stronger pose holding.",
    )
    parser.add_argument(
        "--thumb-force-limit",
        type=float,
        default=None,
        help="Optional thumb joint actuator force limit override. Use 0 to disable the imported thumb limit.",
    )
    parser.add_argument(
        "--hard-hold-hand",
        action="store_true",
        help="Force hand qpos/qvel to the target every simulation step. Diagnostic only.",
    )
    parser.add_argument("--visual-only-markers", action="store_true")
    parser.add_argument("--disable-thumb-mesh-collision", action="store_true")
    parser.add_argument("--disable-distal-mesh-collision", action="store_true")
    parser.add_argument("--thumb-tip-proxy-pos", nargs=3, type=float, default=None)
    parser.add_argument("--thumb-tip-proxy-size", type=float, default=None)
    parser.add_argument(
        "--custom-qpos",
        nargs=20,
        type=float,
        default=None,
        help="Optional 20-joint Dex5 initial qpos. Used for tuned reset poses.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--ignore-termination",
        action="store_true",
        help="Keep stepping until duration even if the pen is dropped/out of workspace. Useful for visual debugging.",
    )
    parser.add_argument(
        "--pause-on-end",
        action="store_true",
        help="Keep the viewer window open after the episode ends. Close the window manually.",
    )
    args = parser.parse_args()

    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=args.pen_init_pos,
        grasp_scale=args.grasp_scale,
        init_noise_scale=args.init_noise_scale,
        episode_seconds=args.duration,
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        hard_hold_hand=args.hard_hold_hand,
        visual_only_markers=args.visual_only_markers,
        disable_thumb_mesh_collision=args.disable_thumb_mesh_collision,
        disable_distal_mesh_collision=args.disable_distal_mesh_collision,
        thumb_tip_proxy_pos=args.thumb_tip_proxy_pos,
        thumb_tip_proxy_size=args.thumb_tip_proxy_size,
        seed=args.seed,
    )
    env.reset(seed=args.seed)
    if args.custom_qpos is not None:
        qpos = np.asarray(args.custom_qpos, dtype=np.float32)
        qpos = np.clip(qpos, env.joint_low, env.joint_high)
        env.target_qpos = qpos
        env.data.qpos[env.qpos_adr] = qpos
        env.data.ctrl[:] = qpos
        mujoco.mj_forward(env.model, env.data)
    if args.pen_quat is not None:
        quat = np.asarray(args.pen_quat, dtype=np.float64)
        quat = quat / np.linalg.norm(quat)
        env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = quat
        mujoco.mj_forward(env.model, env.data)
    if args.pen_qvel is not None:
        env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6] = np.asarray(args.pen_qvel, dtype=np.float64)
        mujoco.mj_forward(env.model, env.data)

    print("M2/M2.2 MuJoCo viewer")
    print(f"model_path: {env.model_path}")
    print(f"policy: {args.policy}")
    print(f"pen_init_pos: {args.pen_init_pos}")
    if args.pen_quat is not None:
        print(f"pen_quat: {args.pen_quat}")
    if args.pen_qvel is not None:
        print(f"pen_qvel: {args.pen_qvel}")
    print(f"grasp_scale: {args.grasp_scale}")
    if args.actuator_kp is not None or args.actuator_kv is not None:
        print(f"actuator gains: kp={env.model.actuator_gainprm[0, 0]:.3f}, kv={-env.model.actuator_biasprm[0, 2]:.3f}")
    if args.thumb_force_limit is not None:
        print(f"thumb_force_limit: {args.thumb_force_limit}")
    if args.hard_hold_hand:
        print("hard_hold_hand: enabled")
    if args.visual_only_markers:
        print("visual_only_markers: enabled")
    if args.disable_thumb_mesh_collision:
        print("disable_thumb_mesh_collision: enabled")
    if args.disable_distal_mesh_collision:
        print("disable_distal_mesh_collision: enabled")
    if args.thumb_tip_proxy_pos is not None or args.thumb_tip_proxy_size is not None:
        print(f"thumb tip proxy override: pos={args.thumb_tip_proxy_pos}, size={args.thumb_tip_proxy_size}")
    if args.custom_qpos is not None:
        print("custom_qpos: enabled")
    print("Close the MuJoCo viewer window to stop.")

    dt = 1.0 / env.control_hz
    max_steps = int(args.duration * env.control_hz)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = np.array([-0.02, 0.03, 0.05])
        viewer.cam.distance = 0.33
        viewer.cam.azimuth = -120
        viewer.cam.elevation = -22

        step = 0
        wall_t0 = time.time()
        while viewer.is_running() and step < max_steps:
            loop_t0 = time.time()
            action = action_for_policy(
                env,
                args.policy,
                step,
                args.action_scale,
                phase_preset=args.phase_preset,
                center_gain=args.center_gain,
                target_center=args.pen_init_pos,
                cycle_steps=args.cycle_steps,
                thumb_scale=args.thumb_scale,
                phase_direction=args.phase_direction,
                return_scale=args.return_scale,
                roll_scale=args.roll_scale,
            )
            _, _, terminated, truncated, info = env.step(action)
            viewer.sync()

            if step % 50 == 0:
                print(
                    f"step={step:04d} "
                    f"z={info['pen_center'][2]:.3f} "
                    f"rounds={info['rotation_rounds']:.3f} "
                    f"ncon={env.data.ncon} "
                    f"terminated={terminated}"
                )
            if (terminated or truncated) and not args.ignore_termination:
                print("episode ended:", "terminated" if terminated else "truncated")
                break
            if (terminated or truncated) and args.ignore_termination and step % 50 == 0:
                print("episode would end:", "terminated" if terminated else "truncated", "but ignore-termination is enabled")

            step += 1
            if args.realtime:
                sleep_s = dt - (time.time() - loop_t0)
                if sleep_s > 0:
                        time.sleep(sleep_s)

        print(f"viewer finished after {step} steps, wall_time={time.time() - wall_t0:.2f}s")
        if args.pause_on_end and viewer.is_running():
            print("pause-on-end enabled: close the MuJoCo viewer window manually.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)
    env.close()


if __name__ == "__main__":
    main()
