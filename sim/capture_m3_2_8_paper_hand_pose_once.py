from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from view_m2_mujoco import DEFAULT_MODEL


POSE_LABELS = {
    1: "paper_pose_1_initial_diagonal",
    2: "paper_pose_2_existing_reset_reference",
    3: "paper_pose_3_vertical_pass",
    4: "paper_pose_4_opposite_diagonal",
    5: "paper_pose_5_horizontal_regrasp",
    6: "paper_pose_6_cycle_closure",
}


def import_ros() -> tuple[Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float64MultiArray
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ROS2 Python packages are unavailable. Use /usr/bin/python3 and source ofdemo setup."
        ) from exc
    return rclpy, Node, Float64MultiArray


def capture_float_array(topic: str, samples: int, timeout_sec: float) -> tuple[np.ndarray, int, float]:
    rclpy, Node, Float64MultiArray = import_ros()
    values: list[np.ndarray] = []
    stamps: list[float] = []
    rclpy.init()

    class CaptureNode(Node):
        def __init__(self) -> None:
            super().__init__("dex5_penspin_m3_2_8_pose_capture_once")
            self.create_subscription(Float64MultiArray, topic, self.on_msg, 10)

        def on_msg(self, msg: Any) -> None:
            data = np.asarray(list(msg.data), dtype=np.float64)
            if data.size >= 20:
                values.append(data[:20].copy())
                stamps.append(time.time())

    node = CaptureNode()
    start = time.time()
    try:
        while len(values) < samples and time.time() - start < timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    if not values:
        raise RuntimeError(f"No valid 20D samples received from {topic} within {timeout_sec}s")
    arr = np.stack(values, axis=0)
    return np.mean(arr, axis=0), len(values), time.time() - stamps[-1]


def render_hand_pose(record: dict[str, Any], image_path: Path) -> bool:
    if Image is None or ImageDraw is None:
        return False
    env = Dex5PenSpinMinimalEnv(
        model_path=record["model_path"],
        pen_init_pos=(0.35, 0.35, 0.35),
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=1.0,
        actuator_kp=400.0,
        actuator_kv=20.0,
        thumb_force_limit=0.0,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        render_mode="rgb_array",
        seed=0,
    )
    try:
        env.reset(seed=0)
        qpos = np.asarray(record["hand_qpos"], dtype=np.float64)
        env.target_qpos = qpos.astype(np.float32)
        env.data.qpos[env.qpos_adr] = np.clip(qpos, env.joint_low, env.joint_high)
        env.data.ctrl[:] = env.data.qpos[env.qpos_adr]
        env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = np.array([0.35, 0.35, 0.35])
        env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        image = Image.fromarray(env.render())
        draw = ImageDraw.Draw(image)
        title = f"Pose {record['paper_pose_index']} {record['paper_pose_label']} hand-only"
        draw.rectangle([0, 0, image.width, 36], fill=(255, 255, 255))
        draw.text((8, 8), title, fill=(0, 0, 0))
        image.save(image_path)
        return True
    finally:
        env.close()


def update_manifest(output_dir: Path, record: dict[str, Any], json_path: Path, png_path: Path | None) -> None:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "stage": "M3.2.8 paper canonical hand pose capture",
            "latest": {},
            "captures": [],
            "note": "Pose 2 is expected to come from the earlier reset reference; current manual capture focuses on 1/3/4/5/6.",
        }
    pose_key = str(record["paper_pose_index"])
    entry = {
        "pose": record["paper_pose_index"],
        "label": record["paper_pose_label"],
        "json": str(json_path),
        "png": str(png_path) if png_path else None,
        "captured_at": record["captured_at"],
        "samples": record["samples"],
    }
    manifest.setdefault("latest", {})[pose_key] = entry
    manifest.setdefault("captures", []).append(entry)
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-index", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--topic", default="/dex5_1/applied_position")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=2.0)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/paper_teleop_hand_poses_m3_2_8",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qpos, sample_count, target_age_sec = capture_float_array(args.topic, args.samples, args.timeout_sec)
    label = POSE_LABELS[args.pose_index]
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"P{args.pose_index}_{label}_{ts}"
    json_path = output_dir / f"{stem}.json"
    latest_path = output_dir / f"P{args.pose_index}_{label}_latest.json"
    png_path = output_dir / f"{stem}.png"
    latest_png_path = output_dir / f"P{args.pose_index}_{label}_latest.png"

    record = {
        "stage": "M3.2.8 paper canonical hand pose capture",
        "source": "chat_triggered_ros_applied_position_capture",
        "paper_pose_index": args.pose_index,
        "paper_pose_label": label,
        "topic": args.topic,
        "samples": sample_count,
        "target_age_sec": target_age_sec,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "hand_qpos": qpos.astype(float).tolist(),
        "target_qpos": qpos.astype(float).tolist(),
        "joint_order": DEX5_JOINT_ORDER,
        "pen_pose_status": "not_recorded_yet",
        "pen_pose_note": "Pen pose will be added later using the paper schematic and contact tuning.",
        "qpos_stats": {
            "mean": float(np.mean(qpos)),
            "std": float(np.std(qpos)),
            "min": float(np.min(qpos)),
            "max": float(np.max(qpos)),
        },
    }
    text = json.dumps(record, indent=2, ensure_ascii=False)
    json_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    rendered = render_hand_pose(record, png_path)
    if rendered:
        latest_png_path.write_bytes(png_path.read_bytes())
    update_manifest(output_dir, record, json_path, png_path if rendered else None)

    print(f"saved pose {args.pose_index}: {label}")
    print(f"json: {json_path}")
    print(f"latest: {latest_path}")
    if rendered:
        print(f"png: {png_path}")
    print(f"samples: {sample_count}, target_age_sec: {target_age_sec:.4f}")


if __name__ == "__main__":
    main()
