#!/usr/bin/env python3
"""Render frozen Isaac S0 rollout states with the Dex5 MuJoCo mesh asset."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Dex5_Sim2Real_S0/oracle_dataset/Dex5_S0_Oracle_Success_Rollouts_1200.npz"
MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml"
OUTPUT = ROOT / "Dex5_Sim2Real_S0/oracle_trajectory_visualization"
ISAAC_JOINTS = (
    "Roll_21R", "Pitch_22R", "Pitch_23R", "Pitch_24R",
    "Roll_31R", "Pitch_32R", "Pitch_33R", "Pitch_34R",
    "Roll_41R", "Pitch_42R", "Pitch_43R", "Pitch_44R",
    "Roll_51R", "Pitch_52R", "Pitch_53R", "Pitch_54R",
    "Yaw_11R", "Roll_12R", "Pitch_13R", "Pitch_14R",
)
CONTACT_NAMES = ("thumb", "index", "middle", "ring", "pinky")
CONTACT_COLORS = ("#d4503f", "#2f7ed8", "#31a354", "#d99018", "#8c6bb1")


def select(data: dict[str, np.ndarray]) -> list[tuple[int, str]]:
    offsets, pose, contact = data["episode_offsets"], data["pen_pose"], data["contact_state"].astype(np.int8)
    rows = []
    for episode, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        q0 = pose[start, 3:7] / np.linalg.norm(pose[start, 3:7])
        q1 = pose[end - 1, 3:7] / np.linalg.norm(pose[end - 1, 3:7])
        angle = 2.0 * np.arccos(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
        transitions = int(np.abs(np.diff(contact[start:end], axis=0)).sum())
        rows.append((episode, angle, transitions))
    ordered = sorted(rows, key=lambda item: item[1])
    return [
        (ordered[len(ordered) // 2][0], "典型成功轨迹"),
        (max(rows, key=lambda item: item[2])[0], "接触变化最多"),
        (max(rows, key=lambda item: item[1])[0], "姿态变化最大"),
    ]


def accumulated_angle(omega: np.ndarray) -> np.ndarray:
    return np.cumsum(omega[:, 2]) / 60.0 * 180.0 / np.pi


def chart(path: Path, episode: dict[str, np.ndarray], label: str) -> dict[str, float]:
    time_s = np.arange(len(episode["joint_position"])) / 60.0
    angle = accumulated_angle(episode["pen_angular_velocity"])
    height = (episode["pen_pose"][:, 2] - episode["pen_pose"][0, 2]) * 1000.0
    wrong = np.linalg.norm(episode["pen_angular_velocity"][:, :2], axis=1)
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 7), dpi=140, sharex=True)
    axes[0].plot(time_s, angle, color="#1769aa", linewidth=2.2)
    axes[0].axhspan(35, 55, color="#31a354", alpha=.15)
    axes[0].set_ylabel("target angle (deg)")
    axes[1].plot(time_s, height, color="#c43c35", label="height delta (mm)")
    axes[1].plot(time_s, episode["pen_angular_velocity"][:, 2], color="#1769aa", label="omega z")
    axes[1].plot(time_s, wrong, color="#d99018", label="wrong-axis speed")
    axes[1].legend(ncol=3, frameon=False)
    axes[1].set_ylabel("mm / rad s-1")
    for name, color, values in zip(CONTACT_NAMES, CONTACT_COLORS, episode["contact_state"].T):
        axes[2].step(time_s, values.astype(float), where="post", color=color, label=name)
    axes[2].set_ylim(-.1, 1.15)
    axes[2].set_ylabel("saved contact")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(ncol=5, frameon=False)
    figure.suptitle(label)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return {
        "duration_s": float(len(time_s) / 60.0),
        "final_accumulated_angle_deg": float(angle[-1]),
        "peak_accumulated_angle_deg": float(np.max(angle)),
        "max_height_loss_mm": float(max(0, -np.min(height))),
        "contact_transition_count": int(np.abs(np.diff(episode["contact_state"].astype(np.int8), axis=0)).sum()),
    }


def overlay(frame: np.ndarray, episode: int, label: str, row: int, angle: float, contacts: np.ndarray) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 555, 84), fill=(18, 25, 33))
    english_label = {
        "典型成功轨迹": "typical success",
        "接触变化最多": "most contact changes",
        "姿态变化最大": "largest pose change",
    }.get(label, label)
    draw.text((22, 20), f"S0 oracle | episode {episode} | {english_label}", fill=(255, 255, 255))
    draw.text((22, 42), f"t={row / 60:.2f}s  accumulated omega_z={angle:.1f} deg", fill=(205, 225, 242))
    active = "/".join(name for name, state in zip(CONTACT_NAMES, contacts) if state) or "none"
    draw.text((22, 64), f"saved contacts: {active}", fill=(185, 235, 190))
    return np.asarray(image)


def write_index(records: list[dict[str, object]]) -> None:
    sections = []
    for record in records:
        metrics = record["metrics"]
        sections.append(f"""<section><div class="head"><div><h2>{html.escape(str(record['label']))}</h2><p>episode <code>{record['episode']}</code> · {metrics['duration_s']:.2f}s · peak {metrics['peak_accumulated_angle_deg']:.1f}° · contacts {metrics['contact_transition_count']}</p></div><b>SIM ROLLOUT</b></div><video controls muted loop playsinline preload="metadata" poster="{record['poster']}"><source src="{record['video']}" type="video/mp4"></video><img src="{record['chart']}" alt="trajectory curves"></section>""")
    (OUTPUT / "index.html").write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dex5 S0 Oracle 训练轨迹</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f7;color:#17202a;font:15px/1.55 system-ui,sans-serif}}header{{background:#fff;border-bottom:1px solid #d7dde2;padding:32px max(22px,calc((100vw - 1020px)/2))}}h1{{font-size:29px;margin:0 0 8px;letter-spacing:0}}header p{{margin:4px 0;color:#5b6670}}main{{max-width:1020px;margin:auto;padding:22px}}section{{padding:20px 0 34px;border-bottom:1px solid #d7dde2}}.head{{display:flex;justify-content:space-between;align-items:end;margin-bottom:12px;gap:16px}}h2{{margin:0;font-size:21px;letter-spacing:0}}.head p{{margin:3px 0;color:#5b6670}}.head b{{font-size:11px;background:#1769aa;color:white;padding:6px 8px;border-radius:4px}}video,img{{display:block;width:100%;background:#111}}video{{aspect-ratio:4/3;object-fit:contain}}img{{margin-top:12px;border:1px solid #d7dde2}}code{{font-family:ui-monospace,monospace}}@media(max-width:600px){{.head{{align-items:start;flex-direction:column}}}}</style></head><body><header><h1>Dex5 S0 Oracle 训练轨迹</h1><p>冻结的 Isaac rollout 状态，使用同一 Dex5+pen MuJoCo 网格资产离屏回放。</p><p>任务范围：P3 → 45°±10° 短旋转 → 制动保持。不是完整转笔、handoff 或真机执行。</p></header><main>{''.join(sections)}</main></body></html>""", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATASET, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    sim_data = mujoco.MjData(model)
    joint_addresses = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in ISAAC_JOINTS]
    pen_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pen_free")
    pen_address = int(model.jnt_qposadr[pen_joint])
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "tip_proxy" in name:
            model.geom_group[geom_id] = 3
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.035, 0.045)
    camera.distance = 0.27
    camera.azimuth = -120.0
    camera.elevation = -20.0
    records = []
    offsets = data["episode_offsets"]
    for episode_id, label in select(data):
        start, end = int(offsets[episode_id]), int(offsets[episode_id + 1])
        episode_data = {key: value[start:end] for key, value in data.items() if value.shape[:1] == data["joint_position"].shape[:1]}
        metric_path = OUTPUT / f"episode_{episode_id:04d}_metrics.png"
        metrics = chart(metric_path, episode_data, f"S0 oracle trajectory | episode {episode_id}")
        angle = accumulated_angle(episode_data["pen_angular_velocity"])
        video = OUTPUT / f"episode_{episode_id:04d}.mp4"
        poster = OUTPUT / f"episode_{episode_id:04d}.png"
        frame_dir = OUTPUT / f"episode_{episode_id:04d}_frames"
        frame_dir.mkdir(exist_ok=True)
        first = None
        for frame_index, row in enumerate(range(0, end - start, 2)):
            sim_data.qpos[joint_addresses] = episode_data["joint_position"][row]
            sim_data.qpos[pen_address:pen_address + 7] = episode_data["pen_pose"][row]
            mujoco.mj_forward(model, sim_data)
            renderer.update_scene(sim_data, camera=camera, scene_option=option)
            frame = overlay(renderer.render(), episode_id, label, row, float(angle[row]), episode_data["contact_state"][row])
            if first is None:
                first = frame.copy()
            Image.fromarray(frame).save(frame_dir / f"frame_{frame_index:04d}.png")
        Image.fromarray(first).save(poster)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(frame_dir / "frame_%04d.png"), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
        shutil.rmtree(frame_dir)
        records.append({"episode": episode_id, "label": label, "video": video.name, "poster": poster.name, "chart": metric_path.name, "metrics": metrics})
    renderer.close()
    write_index(records)
    manifest = {
        "status": "PASS",
        "source_dataset": str(DATASET),
        "source_simulator": "Isaac Lab PhysX rollout",
        "visualization_backend": "MuJoCo Dex5 mesh state playback (RTX unavailable)",
        "scope": "P3 constrained short rotation; no handoff; no powered real replay",
        "episodes": records,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
