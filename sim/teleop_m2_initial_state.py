from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np

from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import contact_snapshot
from view_m2_mujoco import DEFAULT_MODEL


DEFAULT_QPOS = np.asarray(
    [
        0.00,
        0.65,
        0.40,
        0.10,
        0.00,
        0.65,
        0.40,
        0.10,
        0.00,
        0.65,
        0.40,
        0.10,
        0.00,
        0.65,
        0.40,
        0.10,
        0.10,
        0.45,
        0.30,
        0.12,
    ],
    dtype=np.float32,
)

FINGER_GROUPS = {
    "index": [0, 1, 2, 3],
    "middle": [4, 5, 6, 7],
    "ring": [8, 9, 10, 11],
    "pinky": [12, 13, 14, 15],
    "thumb": [16, 17, 18, 19],
}

GROUP_KEYS = {
    ord("1"): "index",
    ord("2"): "middle",
    ord("3"): "ring",
    ord("4"): "pinky",
    ord("5"): "thumb",
}

JOINT_SLOT_KEYS = {
    ord("Q"): 0,
    ord("W"): 1,
    ord("E"): 2,
    ord("R"): 3,
}


def set_hand_qpos(env: Dex5PenSpinMinimalEnv, qpos: np.ndarray) -> None:
    qpos = np.clip(qpos, env.joint_low, env.joint_high).astype(np.float32)
    env.target_qpos = qpos.copy()
    env.data.qpos[env.qpos_adr] = qpos
    env.data.ctrl[:] = qpos
    mujoco.mj_forward(env.model, env.data)


def set_pen_state(
    env: Dex5PenSpinMinimalEnv,
    pen_pos: np.ndarray,
    pen_quat: np.ndarray,
    pen_qvel: np.ndarray | None = None,
) -> None:
    quat = np.asarray(pen_quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = np.asarray(pen_pos, dtype=np.float64)
    env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = quat
    if pen_qvel is not None:
        env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6] = np.asarray(pen_qvel, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)


def command_for_preset(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    qpos = " ".join(f"{v:.6f}" for v in data["hand_qpos"])
    pen_pos = " ".join(f"{v:.6f}" for v in data["pen_pos"])
    pen_quat = " ".join(f"{v:.6f}" for v in data["pen_quat"])
    gain_args = ""
    if data.get("actuator_kp") is not None:
        gain_args += f" --actuator-kp {float(data['actuator_kp']):.6f}"
    if data.get("actuator_kv") is not None:
        gain_args += f" --actuator-kv {float(data['actuator_kv']):.6f}"
    if data.get("thumb_force_limit") is not None:
        gain_args += f" --thumb-force-limit {float(data['thumb_force_limit']):.6f}"
    if data.get("hard_hold_hand"):
        gain_args += " --hard-hold-hand"
    if data.get("visual_only_markers"):
        gain_args += " --visual-only-markers"
    if data.get("disable_thumb_mesh_collision"):
        gain_args += " --disable-thumb-mesh-collision"
    if data.get("disable_distal_mesh_collision"):
        gain_args += " --disable-distal-mesh-collision"
    if data.get("thumb_tip_proxy_pos") is not None:
        pos = " ".join(f"{float(v):.6f}" for v in data["thumb_tip_proxy_pos"])
        gain_args += f" --thumb-tip-proxy-pos {pos}"
    if data.get("thumb_tip_proxy_size") is not None:
        gain_args += f" --thumb-tip-proxy-size {float(data['thumb_tip_proxy_size']):.6f}"
    return (
        "python dex5_penspin/sim/view_m2_mujoco.py "
        "--policy hold --duration 20 --realtime --ignore-termination --pause-on-end "
        f"--pen-init-pos {pen_pos} --pen-quat {pen_quat}{gain_args} --custom-qpos {qpos}"
    )


def print_help() -> None:
    print(
        """
Dex5-PenSpin manual initial-state teleop

Keys:
  1/2/3/4/5  select finger: index / middle / ring / pinky / thumb
  Q/W/E/R    select joint slot inside selected finger
             normal fingers: Roll / MCP / PIP / DIP
             thumb: Yaw / Roll / Pitch / Tip
  A/D        decrease / increase selected joint
  Z/C        decrease / increase all joints in selected finger by small step
  X          reset hand to the default beginner pose
  SPACE      toggle physics. Paused = pose hand. Running = test pen falling/contact.
  P          reset pen to the configured initial position and quaternion
  S          save current qpos + pen state + screenshot
  H          print this help

Recommended workflow:
  1. Keep physics paused.
  2. Use 1-5 and Q/W/E/R + A/D to pose fingers.
  3. Press P to put the pen back above the hand.
  4. Press SPACE and watch where the pen contacts.
  5. Press SPACE again to pause, adjust, repeat.
  6. Press S when the pose looks worth testing.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="dex5_penspin/reports/manual_initial_state_presets")
    parser.add_argument("--name", default="manual")
    parser.add_argument("--pen-init-pos", nargs=3, type=float, default=(-0.025, 0.075, 0.115))
    parser.add_argument("--pen-quat", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--custom-qpos", nargs=20, type=float, default=None)
    parser.add_argument(
        "--actuator-kp",
        type=float,
        default=None,
        help="Optional position-servo kp override. Increase when fingers relax after physics starts.",
    )
    parser.add_argument(
        "--actuator-kv",
        type=float,
        default=None,
        help="Optional position-servo kv override. Increase with kp to damp oscillation.",
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
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-autosave", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=args.pen_init_pos,
        grasp_scale=1.0,
        init_noise_scale=0.0,
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
    hand_qpos = np.asarray(args.custom_qpos, dtype=np.float32) if args.custom_qpos is not None else DEFAULT_QPOS.copy()
    set_hand_qpos(env, hand_qpos)
    set_pen_state(env, np.asarray(args.pen_init_pos), np.asarray(args.pen_quat), np.zeros(6))

    state = {
        "selected_group": "index",
        "selected_slot": 1,
        "running": False,
        "save_count": 0,
    }

    def selected_joint_index() -> int:
        return FINGER_GROUPS[state["selected_group"]][state["selected_slot"]]

    def describe_selection() -> str:
        idx = selected_joint_index()
        return (
            f"group={state['selected_group']} slot={state['selected_slot']} "
            f"joint[{idx}]={DEX5_JOINT_ORDER[idx]} value={env.target_qpos[idx]:.3f}"
        )

    def write_preset(stem: str, save_screenshot: bool, announce: bool) -> Path:
        preset_path = output_dir / f"{stem}.json"
        screenshot_path = output_dir / f"{stem}.png"
        data = {
            "model_path": str(args.model_path),
            "hand_qpos": env.data.qpos[env.qpos_adr].astype(float).tolist(),
            "joint_order": DEX5_JOINT_ORDER,
            "pen_pos": env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3].astype(float).tolist(),
            "pen_quat": env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7].astype(float).tolist(),
            "pen_qvel": env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6].astype(float).tolist(),
            "actuator_kp": float(env.model.actuator_gainprm[0, 0]) if args.actuator_kp is not None else None,
            "actuator_kv": float(-env.model.actuator_biasprm[0, 2]) if args.actuator_kv is not None else None,
            "thumb_force_limit": float(args.thumb_force_limit) if args.thumb_force_limit is not None else None,
            "hard_hold_hand": bool(args.hard_hold_hand),
            "visual_only_markers": bool(args.visual_only_markers),
            "disable_thumb_mesh_collision": bool(args.disable_thumb_mesh_collision),
            "disable_distal_mesh_collision": bool(args.disable_distal_mesh_collision),
            "thumb_tip_proxy_pos": list(args.thumb_tip_proxy_pos) if args.thumb_tip_proxy_pos is not None else None,
            "thumb_tip_proxy_size": float(args.thumb_tip_proxy_size) if args.thumb_tip_proxy_size is not None else None,
            "selected": describe_selection(),
            "command": "",
        }
        preset_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        data["command"] = command_for_preset(preset_path)
        preset_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        if save_screenshot:
            imageio.imwrite(screenshot_path, env.render())
        if announce:
            print("\nSaved preset:")
            print(f"  json: {preset_path}")
            if save_screenshot:
                print(f"  screenshot: {screenshot_path}")
            print("Replay command:")
            print(data["command"])
            print()
        return preset_path

    def autosave_latest() -> None:
        if not args.no_autosave:
            write_preset("autosave_latest", save_screenshot=False, announce=False)

    def save_preset() -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{args.name}_{ts}_{state['save_count']:02d}"
        state["save_count"] += 1
        write_preset(stem, save_screenshot=True, announce=True)

    def apply_delta(indices: list[int], delta: float) -> None:
        q = env.target_qpos.copy()
        q[indices] += delta
        set_hand_qpos(env, q)
        autosave_latest()
        print(describe_selection())

    def key_callback(key: int) -> None:
        if key in GROUP_KEYS:
            state["selected_group"] = GROUP_KEYS[key]
            state["selected_slot"] = min(state["selected_slot"], len(FINGER_GROUPS[state["selected_group"]]) - 1)
            print(describe_selection())
            return
        if key in JOINT_SLOT_KEYS:
            state["selected_slot"] = JOINT_SLOT_KEYS[key]
            print(describe_selection())
            return
        if key == ord("A"):
            apply_delta([selected_joint_index()], -args.step_size)
            return
        if key == ord("D"):
            apply_delta([selected_joint_index()], args.step_size)
            return
        if key == ord("Z"):
            apply_delta(FINGER_GROUPS[state["selected_group"]], -0.5 * args.step_size)
            return
        if key == ord("C"):
            apply_delta(FINGER_GROUPS[state["selected_group"]], 0.5 * args.step_size)
            return
        if key == ord("X"):
            set_hand_qpos(env, DEFAULT_QPOS.copy())
            autosave_latest()
            print("reset hand:", describe_selection())
            return
        if key == ord("P"):
            env.data.qvel[:] = 0.0
            set_pen_state(env, np.asarray(args.pen_init_pos), np.asarray(args.pen_quat), np.zeros(6))
            autosave_latest()
            print("reset pen")
            return
        if key == ord("S"):
            save_preset()
            return
        if key == ord("H"):
            print_help()
            print(describe_selection())
            return
        if key == ord(" "):
            state["running"] = not state["running"]
            print("physics:", "running" if state["running"] else "paused")
            return

    print_help()
    print("Initial selection:", describe_selection())
    if args.actuator_kp is not None or args.actuator_kv is not None:
        print(f"Actuator gains: kp={env.model.actuator_gainprm[0, 0]:.3f}, kv={-env.model.actuator_biasprm[0, 2]:.3f}")
    if args.thumb_force_limit is not None:
        print(f"Thumb force limit: {args.thumb_force_limit}")
    if args.hard_hold_hand:
        print("Hard hand hold: enabled")
    if args.visual_only_markers:
        print("Visual-only markers: enabled")
    if args.disable_thumb_mesh_collision:
        print("Disable thumb mesh collision: enabled")
    if args.disable_distal_mesh_collision:
        print("Disable distal mesh collision: enabled")
    if args.thumb_tip_proxy_pos is not None or args.thumb_tip_proxy_size is not None:
        print(f"Thumb tip proxy override: pos={args.thumb_tip_proxy_pos}, size={args.thumb_tip_proxy_size}")
    print("Close the MuJoCo viewer window to stop.")

    dt = 1.0 / env.control_hz
    max_steps = int(args.duration * env.control_hz)
    step = 0
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = np.array([-0.02, 0.04, 0.06])
        viewer.cam.distance = 0.32
        viewer.cam.azimuth = -120
        viewer.cam.elevation = -20

        while viewer.is_running() and step < max_steps:
            loop_t0 = time.time()
            if state["running"]:
                _, _, terminated, truncated, info = env.step(np.zeros(20, dtype=np.float32))
                if step % 50 == 0:
                    fingers, _, _ = contact_snapshot(env)
                    print(
                        f"step={step:04d} z={info['pen_center'][2]:.3f} "
                        f"rounds={info['rotation_rounds']:.3f} ncon={env.data.ncon} "
                        f"contacts={sorted(fingers)} "
                        f"terminated={terminated}"
                    )
                if terminated or truncated:
                    state["running"] = False
                    print("physics auto-paused:", "terminated" if terminated else "truncated")
            else:
                mujoco.mj_forward(env.model, env.data)
            viewer.sync()
            step += 1
            sleep_s = dt - (time.time() - loop_t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    if not args.no_autosave:
        ts = time.strftime("%Y%m%d_%H%M%S")
        write_preset(f"{args.name}_exit_autosave_{ts}", save_screenshot=True, announce=True)
    env.close()


if __name__ == "__main__":
    main()
