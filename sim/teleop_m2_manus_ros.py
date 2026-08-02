from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional screenshot dependency
    imageio = None

from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from view_m2_mujoco import DEFAULT_MODEL


def import_ros() -> tuple[Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float64MultiArray
        from unitree_dex5_1_control.msg import Dex51Command
    except Exception as exc:  # pragma: no cover - depends on sourced ROS env
        raise RuntimeError(
            "ROS2 Python packages are not available in this shell. Run:\n"
            '  source "/home/linkerhand/ofdemo_ros2_ws/install/setup.bash"\n'
            "then start this script again."
        ) from exc
    return rclpy, Node, (Dex51Command, Float64MultiArray)


class LatestTarget:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.qpos: np.ndarray | None = None
        self.source = ""
        self.stamp = 0.0
        self.count = 0

    def set(self, qpos: np.ndarray, source: str) -> None:
        with self.lock:
            self.qpos = np.asarray(qpos, dtype=np.float32).copy()
            self.source = source
            self.stamp = time.time()
            self.count += 1

    def get(self) -> tuple[np.ndarray | None, str, float, int]:
        with self.lock:
            if self.qpos is None:
                return None, self.source, float("inf"), self.count
            return self.qpos.copy(), self.source, time.time() - self.stamp, self.count


def make_ros_node(
    node_base: Any,
    target: LatestTarget,
    command_topic: str,
    float_topic: str,
    subscribe_float: bool,
) -> Any:
    Dex51Command, Float64MultiArray = import_ros()[2]

    class ManusTargetSubscriber(node_base):
        def __init__(self) -> None:
            super().__init__("dex5_penspin_manus_target_subscriber")
            self.create_subscription(Dex51Command, command_topic, self.on_command, 10)
            if subscribe_float:
                self.create_subscription(Float64MultiArray, float_topic, self.on_float, 10)
            self.get_logger().info(f"subscribed Dex51Command: {command_topic}")
            if subscribe_float:
                self.get_logger().info(f"subscribed Float64MultiArray: {float_topic}")

        def on_command(self, msg: Any) -> None:
            values = list(msg.positions)
            if len(values) < 20:
                self.get_logger().warn(f"ignore Dex51Command with {len(values)} positions")
                return
            target.set(np.asarray(values[:20], dtype=np.float32), command_topic)

        def on_float(self, msg: Any) -> None:
            values = list(msg.data)
            if len(values) < 20:
                self.get_logger().warn(f"ignore Float64MultiArray with {len(values)} values")
                return
            target.set(np.asarray(values[:20], dtype=np.float32), float_topic)

    return ManusTargetSubscriber()


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
    env.prev_phase = 0.0
    env.rotation_phase = 0.0
    mujoco.mj_forward(env.model, env.data)


def apply_hand_target(
    env: Dex5PenSpinMinimalEnv,
    qpos: np.ndarray,
    smoothing_alpha: float,
    hand_mode: str,
) -> None:
    target = np.clip(np.asarray(qpos, dtype=np.float32), env.joint_low, env.joint_high)
    alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
    env.target_qpos = ((1.0 - alpha) * env.target_qpos + alpha * target).astype(np.float32)
    env._enforce_shared_motor_consistency()
    env.data.ctrl[:] = env.target_qpos
    if hand_mode == "kinematic":
        env.data.qpos[env.qpos_adr] = env.target_qpos
        env.data.qvel[env.qvel_adr] = 0.0
        mujoco.mj_forward(env.model, env.data)


def command_for_current(env: Dex5PenSpinMinimalEnv) -> str:
    qpos = " ".join(f"{v:.6f}" for v in env.data.qpos[env.qpos_adr])
    pen_pos = " ".join(f"{v:.6f}" for v in env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3])
    pen_quat = " ".join(f"{v:.6f}" for v in env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7])
    return (
        "python dex5_penspin/sim/view_m2_mujoco.py "
        "--policy hold --duration 20 --realtime --ignore-termination --pause-on-end "
        f"--pen-init-pos {pen_pos} --pen-quat {pen_quat} --custom-qpos {qpos}"
    )


def save_preset(
    env: Dex5PenSpinMinimalEnv,
    output_dir: Path,
    name: str,
    source: str,
    target_age: float,
    target_count: int,
    save_count: int,
    announce: bool = True,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{name}_{ts}_{save_count:02d}"
    json_path = output_dir / f"{stem}.json"
    png_path = output_dir / f"{stem}.png"
    data = {
        "source": "manus_ros_teleop",
        "ros_source_topic": source,
        "target_age_sec": target_age,
        "target_count": target_count,
        "model_path": str(env.model_path),
        "hand_qpos": env.data.qpos[env.qpos_adr].astype(float).tolist(),
        "target_qpos": env.target_qpos.astype(float).tolist(),
        "joint_order": DEX5_JOINT_ORDER,
        "pen_pos": env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3].astype(float).tolist(),
        "pen_quat": env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7].astype(float).tolist(),
        "pen_qvel": env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6].astype(float).tolist(),
        "command": command_for_current(env),
    }
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    wrote_png = False
    if imageio is not None:
        imageio.imwrite(png_path, env.render())
        wrote_png = True
    if announce:
        print("\nSaved Manus teleop preset:")
        print(f"  json: {json_path}")
        if wrote_png:
            print(f"  screenshot: {png_path}")
        else:
            print("  screenshot: skipped because imageio is not installed for this Python")
        print("Replay command:")
        print(data["command"])
        print()
    return save_count + 1


def print_help() -> None:
    print(
        """
Dex5-PenSpin Manus -> MuJoCo teleop

Keys:
  SPACE  pause/run pen physics. Manus hand target continues to update.
  P      reset pen to initial position/quaternion.
  S      save current hand qpos + pen state + screenshot.
  H      print this help.

Workflow:
  1. Start Manus publisher and retarget node.
  2. Start this viewer.
  3. Keep physics paused and pose with the glove.
  4. Press P, then SPACE to let the pen fall/contact.
  5. Press SPACE to pause, adjust hand, repeat.
  6. Press S when the pose is worth testing.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--command-topic", default="/_dex5_mujoco/command")
    parser.add_argument("--float-topic", default="/dex5_1/applied_position")
    parser.add_argument("--subscribe-float", action="store_true")
    parser.add_argument("--output-dir", default="dex5_penspin/reports/manus_mujoco_presets")
    parser.add_argument("--name", default="manus")
    parser.add_argument("--pen-init-pos", nargs=3, type=float, default=(-0.025, 0.075, 0.115))
    parser.add_argument("--pen-quat", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand-mode", choices=["kinematic", "actuator"], default="kinematic")
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--physics-running", action="store_true")
    parser.add_argument("--target-timeout-sec", type=float, default=1.0)
    args = parser.parse_args()

    rclpy, Node, _ = import_ros()
    target = LatestTarget()
    rclpy.init()
    node = make_ros_node(Node, target, args.command_topic, args.float_topic, args.subscribe_float)

    env = Dex5PenSpinMinimalEnv(
        model_path=args.model_path,
        pen_init_pos=args.pen_init_pos,
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=args.duration,
        seed=args.seed,
    )
    env.reset(seed=args.seed)
    set_pen_state(env, np.asarray(args.pen_init_pos), np.asarray(args.pen_quat), np.zeros(6))

    state = {"running": bool(args.physics_running), "save_count": 0}
    output_dir = Path(args.output_dir)

    def key_callback(key: int) -> None:
        if key == ord(" "):
            state["running"] = not state["running"]
            print("physics:", "running" if state["running"] else "paused")
            return
        if key == ord("P"):
            env.data.qvel[:] = 0.0
            set_pen_state(env, np.asarray(args.pen_init_pos), np.asarray(args.pen_quat), np.zeros(6))
            print("reset pen")
            return
        if key == ord("S"):
            _, source, age, count = target.get()
            state["save_count"] = save_preset(
                env, output_dir, args.name, source, age, count, state["save_count"], announce=True
            )
            return
        if key == ord("H"):
            print_help()
            return

    print_help()
    print(f"command_topic: {args.command_topic}")
    if args.subscribe_float:
        print(f"float_topic: {args.float_topic}")
    print(f"hand_mode: {args.hand_mode}")
    print("Close the MuJoCo viewer window to stop.")

    dt = 1.0 / env.control_hz
    max_steps = int(args.duration * env.control_hz)
    step = 0
    last_status_t = 0.0

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            viewer.cam.lookat[:] = np.array([-0.02, 0.04, 0.06])
            viewer.cam.distance = 0.32
            viewer.cam.azimuth = -120
            viewer.cam.elevation = -20

            while rclpy.ok() and viewer.is_running() and step < max_steps:
                loop_t0 = time.time()
                rclpy.spin_once(node, timeout_sec=0.0)
                qpos, source, age, count = target.get()
                if qpos is not None and age <= args.target_timeout_sec:
                    apply_hand_target(env, qpos, args.smoothing_alpha, args.hand_mode)

                if state["running"]:
                    if args.hand_mode == "actuator":
                        env.data.ctrl[:] = env.target_qpos
                    for _ in range(env.frame_skip):
                        mujoco.mj_step(env.model, env.data)
                    env.step_count += 1
                    env._update_rotation_phase()
                    if env._pen_dropped() or env._object_out_of_workspace():
                        state["running"] = False
                        print("physics auto-paused: pen dropped/out of workspace")
                else:
                    mujoco.mj_forward(env.model, env.data)

                now = time.time()
                if now - last_status_t > 1.0:
                    last_status_t = now
                    center = env.get_state().pen_center
                    if qpos is None:
                        print("waiting for Manus/Dex5 target messages...")
                    else:
                        stale = " STALE" if age > args.target_timeout_sec else ""
                        print(
                            f"targets={count} age={age:.3f}s{stale} source={source} "
                            f"physics={'run' if state['running'] else 'pause'} "
                            f"pen_z={center[2]:.3f} ncon={env.data.ncon}"
                        )

                viewer.sync()
                step += 1
                sleep_s = dt - (time.time() - loop_t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        qpos, source, age, count = target.get()
        if qpos is not None:
            save_preset(env, output_dir, f"{args.name}_exit_autosave", source, age, count, 0, announce=True)
        env.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
