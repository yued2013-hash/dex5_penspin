#!/usr/bin/env python3
"""Render an Isaac O1.23 failure trajectory with the Dex5 MuJoCo mesh asset."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY = ROOT / (
    "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/"
    "O1_23_典型P3到P4失败轨迹/failure_trajectory_states.npz"
)
DEFAULT_OUTPUT = ROOT / (
    "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/"
    "O1_23_典型P3到P4失败轨迹_MuJoCo"
)
MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml"
JOINTS = (
    "Roll_21R", "Pitch_22R", "Pitch_23R", "Pitch_24R",
    "Roll_31R", "Pitch_32R", "Pitch_33R", "Pitch_34R",
    "Roll_41R", "Pitch_42R", "Pitch_43R", "Pitch_44R",
    "Roll_51R", "Pitch_52R", "Pitch_53R", "Pitch_54R",
    "Yaw_11R", "Roll_12R", "Pitch_13R", "Pitch_14R",
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def _overlay(
    frame: np.ndarray,
    row: int,
    height_mm: float,
    angle_deg: float,
    contacts: np.ndarray,
    forces: np.ndarray,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 705, 112), fill=(18, 25, 33))
    draw.text((24, 22), "O1.23 failure | MuJoCo mesh state playback", fill=(255, 255, 255))
    draw.text(
        (24, 46),
        f"t={row / 60.0:.2f}s  height={height_mm:+.1f} mm  integrated omega_z={angle_deg:+.1f} deg",
        fill=(205, 225, 242),
    )
    active = "/".join(name for name, value in zip(FINGERS, contacts) if value) or "none"
    draw.text((24, 70), f"saved PhysX contacts: {active}", fill=(185, 235, 190))
    draw.text(
        (24, 92),
        f"saved T/I/M force={forces[0]:.2f}/{forces[1]:.2f}/{forces[2]:.2f} N",
        fill=(238, 190, 178),
    )
    return np.asarray(image)


def _write_index(output: Path, metrics: dict[str, object], trajectory: Path) -> None:
    output.joinpath("index.html").write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>O1.23 MuJoCo 失败轨迹回放</title>
<style>body{{margin:0;background:#eef1f4;color:#17202a;font:15px/1.6 system-ui,sans-serif}}header,main{{max-width:1080px;margin:auto;padding:24px}}header{{max-width:none;background:#fff;padding-left:max(24px,calc((100vw - 1032px)/2))}}h1{{margin:0 0 8px;font-size:28px;letter-spacing:0}}video,img{{display:block;width:100%;margin:18px 0;background:#111}}.warn{{background:#fff;border-left:4px solid #d99018;padding:10px 14px}}code{{font-family:ui-monospace,monospace}}</style>
</head><body><header><h1>O1.23 P4 绕行→P5 捕获失败：MuJoCo 网格回放</h1>
<p>用 Dex5+pen MuJoCo 网格显示冻结的 Isaac/PhysX policy rollout。</p></header><main>
<div class="warn"><strong>口径：</strong>这是逐帧运动学回放，用于看清手指和笔；不是在 MuJoCo 中重新执行 policy，也不能证明两引擎动力学一致。</div>
<video controls muted loop playsinline preload="metadata" poster="failure_mujoco_poster.png"><source src="failure_mujoco.mp4" type="video/mp4"></video>
<p>最大高度下降 <strong>{float(metrics['max_height_loss_mm']):.2f} mm</strong>，最终累计目标轴转角 <strong>{float(metrics['final_angle_deg']):.1f}°</strong>。</p>
<img src="../O1_23_典型P3到P4失败轨迹/failure_metrics.png" alt="PhysX trajectory metrics">
<p>source: <code>{html.escape(str(trajectory))}</code></p></main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    args = _args()
    trajectory = args.trajectory.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(trajectory, allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    sim_data = mujoco.MjData(model)
    joint_addresses = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINTS
    ]
    pen_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pen_free")
    pen_address = int(model.jnt_qposadr[pen_joint])
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "tip_proxy" in name:
            model.geom_group[geom_id] = 3
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.035, 0.045)
    camera.distance = 0.29
    camera.azimuth = -120.0
    camera.elevation = -18.0
    raw_video = output / "failure_mujoco_raw.mp4"
    video = output / "failure_mujoco.mp4"
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {raw_video}")
    angle = np.cumsum(data["angular_velocity"][:, 2]) / 60.0 * 180.0 / np.pi
    height = 1000.0 * (data["pen_pose"][:, 2] - data["pen_pose"][0, 2])
    first = None
    for row in range(0, data["joint_position"].shape[0], args.stride):
        sim_data.qpos[joint_addresses] = data["joint_position"][row]
        sim_data.qpos[pen_address : pen_address + 7] = data["pen_pose"][row]
        mujoco.mj_forward(model, sim_data)
        renderer.update_scene(sim_data, camera=camera, scene_option=option)
        frame = _overlay(
            renderer.render(), row, float(height[row]), float(angle[row]),
            data["contact"][row], data["force"][row],
        )
        if first is None:
            first = frame.copy()
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    renderer.close()
    if first is None:
        raise RuntimeError("no MuJoCo frames rendered")
    Image.fromarray(first).save(output / "failure_mujoco_poster.png")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
        ],
        check=True,
    )
    raw_video.unlink()
    metrics = {
        "status": "PASS",
        "source_simulator": "Isaac Lab PhysX policy rollout",
        "visualization_backend": "MuJoCo Dex5 mesh kinematic state playback",
        "dynamics_re_evaluated_in_mujoco": False,
        "max_height_loss_mm": float(max(0.0, -np.min(height))),
        "final_angle_deg": float(angle[-1]),
        "trajectory": str(trajectory),
        "artifact_role": "visualization_only_not_teacher_not_reset",
    }
    output.joinpath("manifest.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_index(output, metrics, trajectory)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
