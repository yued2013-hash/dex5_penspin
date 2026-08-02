#!/usr/bin/env python3
"""Render S0.12 selected PhysX states with the Dex5 MuJoCo mesh model."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2_mesh_plus_proxy.xml"
ISAAC_JOINTS = (
    "Roll_21R", "Pitch_22R", "Pitch_23R", "Pitch_24R",
    "Roll_31R", "Pitch_32R", "Pitch_33R", "Pitch_34R",
    "Roll_41R", "Pitch_42R", "Pitch_43R", "Pitch_44R",
    "Roll_51R", "Pitch_52R", "Pitch_53R", "Pitch_54R",
    "Yaw_11R", "Roll_12R", "Pitch_13R", "Pitch_14R",
)
CONTACT_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[:, 0] * right[:, 0] - np.sum(left[:, 1:] * right[:, 1:], axis=1)
    vector = (
        left[:, :1] * right[:, 1:]
        + right[:, :1] * left[:, 1:]
        + np.cross(left[:, 1:], right[:, 1:])
    )
    return np.concatenate((scalar[:, None], vector), axis=1)


def target_rotation_deg(quaternion: np.ndarray) -> np.ndarray:
    initial = quaternion[0] / np.linalg.norm(quaternion[0])
    conjugate = np.repeat(initial[None, :], len(quaternion), axis=0)
    conjugate[:, 1:] *= -1.0
    relative = quat_mul(quaternion, conjugate)
    relative /= np.linalg.norm(relative, axis=1, keepdims=True)
    relative = np.where(relative[:, :1] < 0.0, -relative, relative)
    vector_norm = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, 0], 1.0e-9, None))
    axis_z = np.divide(
        relative[:, 3], vector_norm, out=np.zeros_like(vector_norm), where=vector_norm > 1.0e-9
    )
    return np.rad2deg(angle * axis_z)


def overlay(
    frame: np.ndarray,
    rank: int,
    episode: int,
    step: int,
    rotation_deg: float,
    contacts: np.ndarray,
    strict_hold: bool,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 625, 105), fill=(18, 25, 33))
    draw.text((25, 22), f"Dex5 S0.12  |  rank {rank:02d}  |  episode {episode}", fill="white")
    draw.text(
        (25, 48),
        f"t={step / 60.0:.2f}s   actual quaternion rotation={rotation_deg:.1f} deg",
        fill=(205, 225, 242),
    )
    active = "/".join(name for name, state in zip(CONTACT_NAMES, contacts, strict=True) if state) or "none"
    phase = "STRICT HOLD" if strict_hold else "ROTATION"
    draw.text((25, 74), f"contacts={active}   phase={phase}", fill=(170, 235, 185))
    return np.asarray(image)


def write_index(output: Path, records: list[dict[str, object]], rollout: Path) -> None:
    sections = []
    for record in records:
        sections.append(f"""
<section><div class="head"><div><span>#{record['rank']:02d}</span><h2>Episode {record['episode']}</h2></div>
<dl><div><dt>实际四元数旋转</dt><dd>{record['final_rotation_deg']:.2f}°</dd></div><div><dt>时长</dt><dd>{record['duration_s']:.2f}s</dd></div><div><dt>严格保持</dt><dd>{record['strict_hold_ms']:.0f}ms</dd></div></dl></div>
<video controls muted loop playsinline preload="metadata" poster="{html.escape(str(record['poster']))}"><source src="{html.escape(str(record['video']))}" type="video/mp4"></video></section>""")
    output.joinpath("index.html").write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dex5 S0.12 转笔动作回放</title><style>:root{{--ink:#17202a;--muted:#59656f;--line:#d7dde2;--paper:#f4f6f7;--white:#fff;--blue:#1769aa;--green:#2f855a}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif;letter-spacing:0}}header{{background:#fff;padding:34px max(22px,calc((100vw - 1020px)/2));border-bottom:1px solid var(--line)}}h1{{margin:0 0 8px;font-size:30px;letter-spacing:0}}header p{{margin:5px 0;max-width:900px;color:var(--muted)}}main{{max-width:1020px;margin:auto;padding:24px 22px 60px}}section{{padding:22px 0 34px;border-bottom:1px solid var(--line)}}.head{{display:flex;justify-content:space-between;align-items:end;gap:22px;margin-bottom:13px}}.head>div{{display:flex;align-items:center;gap:10px}}.head span{{display:grid;place-items:center;background:var(--blue);color:#fff;font-weight:800;width:42px;height:30px;border-radius:4px}}h2{{margin:0;font-size:21px;letter-spacing:0}}dl{{display:grid;grid-template-columns:repeat(3,minmax(120px,1fr));gap:18px;margin:0}}dt{{font-size:11px;color:var(--muted)}}dd{{margin:1px 0 0;font-weight:700}}video{{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#111}}.note{{border-left:4px solid var(--green);background:#eef8f2;padding:12px 15px;margin-bottom:8px}}code{{font:12px ui-monospace,monospace}}@media(max-width:700px){{.head{{align-items:flex-start;flex-direction:column}}dl{{width:100%;grid-template-columns:1fr 1fr}}}}</style></head><body><header><h1>Dex5 S0.12 转笔动作回放</h1><p>排名前 3 条 oracle replay 候选的 Dex5 手+笔网格逐帧回放。关节状态和笔位姿来自同一步 PhysX rollout，旋转量由四元数差分计算，不是角速度积分动画。</p><p>当前任务是 P3 起始抓持 → 约 35° 短旋转 → 150ms 以上制动保持，不是完整一圈 PenSpin。</p></header><main><div class="note">视频保持原始 60 Hz 仿真时间尺度：每 2 帧取样，以 30 fps 编码。</div>{''.join(sections)}<p>rollout: <code>{html.escape(str(rollout.resolve()))}</code></p></main></body></html>""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", default="1,2,3")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("status") != "PASS_OFFLINE_SELECTION":
        raise RuntimeError("selection manifest is not accepted")
    if sha256(args.rollout) != selection["rollout_sha256"]:
        raise RuntimeError("rollout hash mismatch")
    selected = selection["selected_trajectories"]
    ranks = [int(value) for value in args.ranks.split(",")]
    with np.load(args.rollout, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
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
    records: list[dict[str, object]] = []

    for rank in ranks:
        record = selected[rank - 1]
        episode = int(record["episode_index"])
        start, stop = int(record["row_start"]), int(record["row_stop"])
        rotation = target_rotation_deg(data["pen_pose_next"][start:stop, 3:7])
        hold_start = stop - start - int(record["strict_hold_steps"])
        frame_dir = output / f"rank_{rank:02d}_frames"
        frame_dir.mkdir(exist_ok=True)
        video = output / f"rank_{rank:02d}_episode_{episode:04d}.mp4"
        poster = output / f"rank_{rank:02d}_episode_{episode:04d}.png"
        first_frame: np.ndarray | None = None
        frame_count = 0
        for local_step in range(0, stop - start, args.stride):
            row = start + local_step
            sim_data.qpos[joint_addresses] = data["joint_position"][row]
            sim_data.qpos[pen_address:pen_address + 7] = data["pen_pose"][row]
            mujoco.mj_forward(model, sim_data)
            renderer.update_scene(sim_data, camera=camera, scene_option=option)
            frame = overlay(
                renderer.render(), rank, episode, local_step, float(rotation[local_step]),
                data["contact_state_next"][row], local_step >= hold_start,
            )
            if first_frame is None:
                first_frame = frame.copy()
            Image.fromarray(frame).save(frame_dir / f"frame_{frame_count:04d}.png")
            frame_count += 1
        if first_frame is None:
            raise RuntimeError(f"episode {episode} produced no frames")
        Image.fromarray(first_frame).save(poster)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", str(frame_dir / "frame_%04d.png"), "-c:v", "libx264", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
        ], check=True)
        shutil.rmtree(frame_dir)
        records.append({
            "rank": rank,
            "episode": episode,
            "video": video.name,
            "poster": poster.name,
            "duration_s": (stop - start) / 60.0,
            "final_rotation_deg": float(rotation[-1]),
            "strict_hold_ms": float(record["strict_hold_ms"]),
            "source_trajectory_id": int(record["source_trajectory_id"]),
        })
    renderer.close()
    write_index(output, records, args.rollout)
    manifest = {
        "status": "PASS",
        "source_rollout": str(args.rollout.resolve()),
        "source_rollout_sha256": sha256(args.rollout),
        "selection_manifest": str(args.selection.resolve()),
        "visualization_backend": "MuJoCo mesh state playback; source dynamics are PhysX",
        "rotation_metric": "signed world-z component of quaternion delta",
        "records": records,
    }
    output.joinpath("manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
