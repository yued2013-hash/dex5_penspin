from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
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


KEYFRAMES = [
    {
        "id": "K0",
        "label": "tripod_hold",
        "note": "Index/middle/ring fingertip tripod supports the pen before thumb clamp.",
    },
    {
        "id": "K1",
        "label": "thumb_light_clamp",
        "note": "Thumb lightly touches the pen while tripod support still dominates.",
    },
    {
        "id": "K2",
        "label": "index_release",
        "note": "Index begins to release while middle/ring/thumb keep the pen stable.",
    },
    {
        "id": "K3",
        "label": "middle_push",
        "note": "Middle/ring drive the phase forward without ejecting the pen.",
    },
    {
        "id": "K4",
        "label": "thumb_index_regrasp",
        "note": "Thumb/index re-establish contact for the next cycle.",
    },
    {
        "id": "K5",
        "label": "cycle_closure",
        "note": "Pose closes the cycle and can transition back toward K0.",
    },
]

BODY_TO_FINGER = {
    "Link_14R": "thumb",
    "Link_24R": "index",
    "Link_34R": "middle",
    "Link_44R": "ring",
    "Link_54R": "pinky",
}


def contact_snapshot(env: Dex5PenSpinMinimalEnv) -> tuple[set[str], Counter[str], float]:
    fingers: set[str] = set()
    geoms: Counter[str] = Counter()
    max_penetration = 0.0
    for i in range(env.data.ncon):
        con = env.data.contact[i]
        geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom{con.geom1}"
        geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom{con.geom2}"
        if "pen" not in geom1 and "pen" not in geom2:
            continue
        other_geom_id = con.geom2 if "pen" in geom1 else con.geom1
        other_geom = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id) or f"geom{other_geom_id}"
        geoms[other_geom] += 1
        body_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            env.model.geom_bodyid[other_geom_id],
        )
        if body_name in BODY_TO_FINGER:
            fingers.add(BODY_TO_FINGER[body_name])
        max_penetration = max(max_penetration, float(max(0.0, -con.dist)))
    return fingers, geoms, max_penetration


def import_ros() -> tuple[Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float64MultiArray
        from unitree_dex5_1_control.msg import Dex51Command
    except Exception as exc:  # pragma: no cover - depends on sourced ROS env
        raise RuntimeError(
            "ROS2 Python packages are not available in this shell. Run:\n"
            '  source "${DEX5_ROS_WS}/install/setup.bash"\n'
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
            super().__init__("dex5_penspin_m3_2_6_keyframe_subscriber")
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
    env.prev_phase = env._quat_twist_phase_about_x(quat)
    env.rotation_phase = 0.0
    mujoco.mj_forward(env.model, env.data)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * float(angle)
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def nudge_pen(
    env: Dex5PenSpinMinimalEnv,
    *,
    dpos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drot_axis: tuple[float, float, float] | None = None,
    drot_angle: float = 0.0,
) -> None:
    pen_pos = env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3].copy()
    pen_quat = env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7].copy()
    pen_pos += np.asarray(dpos, dtype=np.float64)
    if drot_axis is not None and abs(drot_angle) > 0.0:
        delta = quat_from_axis_angle(np.asarray(drot_axis, dtype=np.float64), drot_angle)
        pen_quat = quat_mul(delta, pen_quat)
    set_pen_state(env, pen_pos, pen_quat, np.zeros(6))


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


def keyframe_stem(index: int) -> str:
    key = KEYFRAMES[index]
    return f"{key['id']}_{key['label']}"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stage": "M3.2.6", "latest": {}, "saves": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"stage": "M3.2.6", "latest": {}, "saves": []}


def write_manifest(output_dir: Path, record: dict[str, Any]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    latest = manifest.setdefault("latest", {})
    latest[record["keyframe_id"]] = record
    saves = manifest.setdefault("saves", [])
    saves.append(record)
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def save_keyframe(
    env: Dex5PenSpinMinimalEnv,
    output_dir: Path,
    index: int,
    source: str,
    target_age: float,
    target_count: int,
    save_count: int,
    args: argparse.Namespace,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    key = KEYFRAMES[index]
    stem = keyframe_stem(index)
    ts = time.strftime("%Y%m%d_%H%M%S")
    versioned_stem = f"{stem}_{ts}_{save_count:02d}"
    json_path = output_dir / f"{versioned_stem}.json"
    latest_json_path = output_dir / f"{stem}_latest.json"
    png_path = output_dir / f"{versioned_stem}.png"
    latest_png_path = output_dir / f"{stem}_latest.png"
    fingers, geoms, max_penetration = contact_snapshot(env)

    data = {
        "stage": "M3.2.6 teleop canonical keyframe capture",
        "source": "manus_ros_teleop_canonical_keyframe_capture",
        "keyframe_index": int(index),
        "keyframe_id": key["id"],
        "keyframe_label": key["label"],
        "keyframe_note": key["note"],
        "ros_source_topic": source,
        "target_age_sec": float(target_age),
        "target_count": int(target_count),
        "model_path": str(env.model_path),
        "hand_qpos": env.data.qpos[env.qpos_adr].astype(float).tolist(),
        "target_qpos": env.target_qpos.astype(float).tolist(),
        "joint_order": DEX5_JOINT_ORDER,
        "pen_pos": env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3].astype(float).tolist(),
        "pen_quat": env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7].astype(float).tolist(),
        "pen_qvel": env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 6].astype(float).tolist(),
        "contact_fingers": sorted(fingers),
        "contact_geoms_top": geoms.most_common(20),
        "max_penetration": float(max_penetration),
        "actuator_kp": args.actuator_kp,
        "actuator_kv": args.actuator_kv,
        "thumb_force_limit": args.thumb_force_limit,
        "visual_only_markers": bool(args.visual_only_markers),
        "disable_thumb_mesh_collision": bool(args.disable_thumb_mesh_collision),
        "disable_distal_mesh_collision": bool(args.disable_distal_mesh_collision),
        "thumb_tip_proxy_pos": args.thumb_tip_proxy_pos,
        "thumb_tip_proxy_size": args.thumb_tip_proxy_size,
        "capture_args": vars(args),
        "command": command_for_current(env),
    }

    text = json.dumps(data, indent=2, ensure_ascii=False)
    json_path.write_text(text, encoding="utf-8")
    latest_json_path.write_text(text, encoding="utf-8")

    wrote_png = False
    if imageio is not None:
        frame = env.render()
        imageio.imwrite(png_path, frame)
        imageio.imwrite(latest_png_path, frame)
        wrote_png = True

    record = {
        "keyframe_id": key["id"],
        "keyframe_label": key["label"],
        "json": str(json_path),
        "latest_json": str(latest_json_path),
        "png": str(png_path) if wrote_png else None,
        "latest_png": str(latest_png_path) if wrote_png else None,
        "contact_fingers": sorted(fingers),
        "target_count": int(target_count),
        "target_age_sec": float(target_age),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_manifest(output_dir, record)

    print("\nSaved canonical keyframe:")
    print(f"  selected: {key['id']} {key['label']}")
    print(f"  json: {json_path}")
    print(f"  latest: {latest_json_path}")
    if wrote_png:
        print(f"  screenshot: {png_path}")
    else:
        print("  screenshot: skipped because imageio is not installed")
    print(f"  contacts: {sorted(fingers)}")
    print("Replay command:")
    print(data["command"])
    print()
    return save_count + 1


def validate_saved_keyframe(path: Path, seconds: float, seed: int) -> Path | None:
    try:
        from m3_oracle_hold_env import Dex5PenSpinOracleHoldEnv, summarize_rollout
    except Exception as exc:
        print(f"validation skipped: cannot import hold env ({exc})")
        return None

    env = Dex5PenSpinOracleHoldEnv(preset_path=path, episode_seconds=seconds, seed=seed)
    obs, info = env.reset(seed=seed)
    del obs
    infos: list[dict[str, Any]] = [info]
    returns = 0.0
    steps = 0
    terminated = False
    truncated = False
    try:
        while not (terminated or truncated):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            infos.append(info)
            returns += float(reward)
            steps += 1
        summary = summarize_rollout(infos, returns, steps, env.control_hz)
        summary["terminated"] = bool(terminated)
        summary["truncated"] = bool(truncated)
        summary["preset"] = str(path)
        summary["validation_seconds_requested"] = float(seconds)
        out_path = path.with_suffix(".validation.json")
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nValidation finished:")
        print(f"  preset: {path}")
        print(f"  result: {out_path}")
        print(
            "  survival={survival_seconds:.2f}s tripod_ratio={tripod_ratio:.3f} "
            "xy_drift_max={xy_drift_max:.4f} terminated={terminated}".format(**summary)
        )
        print()
        return out_path
    finally:
        env.close()


def print_help(selected_index: int = 0) -> None:
    key = KEYFRAMES[selected_index]
    print(
        f"""
Dex5-PenSpin M3.2.6 Manus -> MuJoCo canonical keyframe capture

Current keyframe:
  {key['id']} {key['label']} - {key['note']}

Keys:
  0-5      select canonical keyframe K0-K5.
  N / ]    select next keyframe.
  B / [    select previous keyframe.
  S        save current hand qpos + pen state as selected keyframe.
  V        validate latest saved selected keyframe with zero-action hold.
  P        reset pen to initial position/quaternion.
  SPACE    pause/run pen physics. Manus hand target continues to update.
  H        print this help.

Pen placement keys:
  J / L    move pen x - / x +
  K / I    move pen y - / y +
  U / O    move pen z - / z +
  F / R    roll pen - / roll +
  G / T    pitch pen - / pitch +
  Z / C    yaw pen - / yaw +

Suggested capture order:
  K0 tripod support -> K1 thumb light clamp -> K2 index release ->
  K3 middle/ring push -> K4 thumb/index regrasp -> K5 cycle closure.
"""
    )


def latest_keyframe_path(output_dir: Path, index: int) -> Path:
    return output_dir / f"{keyframe_stem(index)}_latest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--command-topic", default="/_dex5_mujoco/command")
    parser.add_argument("--float-topic", default="/dex5_1/applied_position")
    parser.add_argument("--subscribe-float", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/teleop_canonical_keyframes",
    )
    parser.add_argument("--pen-init-pos", nargs=3, type=float, default=(-0.004399, 0.094368, 0.066911))
    parser.add_argument("--pen-quat", nargs=4, type=float, default=(0.882100, 0.0, -0.021750, 0.470560))
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hand-mode", choices=["kinematic", "actuator"], default="kinematic")
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--physics-running", action="store_true")
    parser.add_argument("--target-timeout-sec", type=float, default=1.0)
    parser.add_argument("--actuator-kp", type=float, default=400.0)
    parser.add_argument("--actuator-kv", type=float, default=20.0)
    parser.add_argument("--thumb-force-limit", type=float, default=0.0)
    parser.add_argument("--visual-only-markers", action="store_true", default=True)
    parser.add_argument("--disable-thumb-mesh-collision", action="store_true", default=True)
    parser.add_argument("--disable-distal-mesh-collision", action="store_true", default=True)
    parser.add_argument("--thumb-tip-proxy-pos", nargs=3, type=float, default=None)
    parser.add_argument("--thumb-tip-proxy-size", type=float, default=None)
    parser.add_argument("--validate-seconds", type=float, default=3.0)
    parser.add_argument("--pen-nudge-pos", type=float, default=0.002)
    parser.add_argument("--pen-nudge-angle-deg", type=float, default=3.0)
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
        actuator_kp=args.actuator_kp,
        actuator_kv=args.actuator_kv,
        thumb_force_limit=args.thumb_force_limit,
        visual_only_markers=args.visual_only_markers,
        disable_thumb_mesh_collision=args.disable_thumb_mesh_collision,
        disable_distal_mesh_collision=args.disable_distal_mesh_collision,
        thumb_tip_proxy_pos=args.thumb_tip_proxy_pos,
        thumb_tip_proxy_size=args.thumb_tip_proxy_size,
        seed=args.seed,
    )
    env.reset(seed=args.seed)
    set_pen_state(env, np.asarray(args.pen_init_pos), np.asarray(args.pen_quat), np.zeros(6))

    state: dict[str, Any] = {
        "running": bool(args.physics_running),
        "save_count": 0,
        "selected_index": 0,
    }
    output_dir = Path(args.output_dir)

    def select_keyframe(index: int) -> None:
        state["selected_index"] = int(index) % len(KEYFRAMES)
        key = KEYFRAMES[state["selected_index"]]
        print(f"selected {key['id']} {key['label']}: {key['note']}")

    def key_callback(key: int) -> None:
        if ord("0") <= key <= ord("5"):
            select_keyframe(key - ord("0"))
            return
        if key in (ord("N"), ord("]")):
            select_keyframe(state["selected_index"] + 1)
            return
        if key in (ord("B"), ord("[")):
            select_keyframe(state["selected_index"] - 1)
            return
        if key == ord(" "):
            state["running"] = not state["running"]
            print("physics:", "running" if state["running"] else "paused")
            return
        if key == ord("P"):
            env.data.qvel[:] = 0.0
            set_pen_state(env, np.asarray(args.pen_init_pos), np.asarray(args.pen_quat), np.zeros(6))
            print("reset pen")
            return
        pos_step = float(args.pen_nudge_pos)
        angle_step = float(np.deg2rad(args.pen_nudge_angle_deg))
        pen_nudge_map: dict[int, tuple[tuple[float, float, float], tuple[float, float, float] | None, float, str]] = {
            ord("J"): ((-pos_step, 0.0, 0.0), None, 0.0, "pen x -"),
            ord("L"): ((pos_step, 0.0, 0.0), None, 0.0, "pen x +"),
            ord("K"): ((0.0, -pos_step, 0.0), None, 0.0, "pen y -"),
            ord("I"): ((0.0, pos_step, 0.0), None, 0.0, "pen y +"),
            ord("U"): ((0.0, 0.0, -pos_step), None, 0.0, "pen z -"),
            ord("O"): ((0.0, 0.0, pos_step), None, 0.0, "pen z +"),
            ord("F"): ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -angle_step, "pen roll -"),
            ord("R"): ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), angle_step, "pen roll +"),
            ord("G"): ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -angle_step, "pen pitch -"),
            ord("T"): ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), angle_step, "pen pitch +"),
            ord("Z"): ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -angle_step, "pen yaw -"),
            ord("C"): ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), angle_step, "pen yaw +"),
        }
        if key in pen_nudge_map:
            dpos, axis, angle, label = pen_nudge_map[key]
            nudge_pen(env, dpos=dpos, drot_axis=axis, drot_angle=angle)
            fingers, _, _ = contact_snapshot(env)
            center = env.get_state().pen_center
            print(f"{label}: pen=({center[0]:+.4f},{center[1]:+.4f},{center[2]:+.4f}) contacts={sorted(fingers)}")
            return
        if key == ord("S"):
            _, source, age, count = target.get()
            state["save_count"] = save_keyframe(
                env,
                output_dir,
                state["selected_index"],
                source,
                age,
                count,
                state["save_count"],
                args,
            )
            return
        if key == ord("V"):
            path = latest_keyframe_path(output_dir, state["selected_index"])
            if not path.exists():
                print(f"nothing to validate yet: {path}")
                return
            validate_saved_keyframe(path, args.validate_seconds, args.seed)
            return
        if key == ord("H"):
            print_help(state["selected_index"])
            return

    print_help(state["selected_index"])
    print(f"output_dir: {output_dir}")
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
            viewer.cam.lookat[:] = np.array([-0.02, 0.055, 0.065])
            viewer.cam.distance = 0.28
            viewer.cam.azimuth = -120
            viewer.cam.elevation = -18

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
                    selected = KEYFRAMES[state["selected_index"]]
                    if qpos is None:
                        print("waiting for Manus/Dex5 target messages...")
                    else:
                        fingers, _, _ = contact_snapshot(env)
                        stale = " STALE" if age > args.target_timeout_sec else ""
                        print(
                            f"{selected['id']} {selected['label']} "
                            f"targets={count} age={age:.3f}s{stale} source={source} "
                            f"physics={'run' if state['running'] else 'pause'} "
                            f"pen=({center[0]:+.3f},{center[1]:+.3f},{center[2]:+.3f}) "
                            f"contacts={sorted(fingers)} ncon={env.data.ncon}"
                        )

                viewer.sync()
                step += 1
                sleep_s = dt - (time.time() - loop_t0)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        env.close()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
