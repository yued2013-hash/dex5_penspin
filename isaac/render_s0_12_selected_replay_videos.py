from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

parser = argparse.ArgumentParser()
parser.add_argument("--rollout", type=Path, required=True)
parser.add_argument("--selection", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--ranks", default="1,2,3")
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


TASK = "Dex5-S0.2-P3HardwareCompat-Audit-Direct-v0"
CONTROL_HZ = 60.0
CONTACT_NAMES = ("T", "I", "M", "R", "P")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[:, 0] * right[:, 0] - np.sum(
        left[:, 1:] * right[:, 1:], axis=1
    )
    vector = (
        left[:, :1] * right[:, 1:]
        + right[:, :1] * left[:, 1:]
        + np.cross(left[:, 1:], right[:, 1:])
    )
    return np.concatenate((scalar[:, None], vector), axis=1)


def _quaternion_rotation_deg(
    quaternion: np.ndarray, initial_quaternion: np.ndarray
) -> np.ndarray:
    conjugate = np.repeat(initial_quaternion[None, :], len(quaternion), axis=0)
    conjugate[:, 1:] *= -1.0
    relative = _quat_mul(quaternion, conjugate)
    relative /= np.linalg.norm(relative, axis=1, keepdims=True)
    relative = np.where(relative[:, :1] < 0.0, -relative, relative)
    vector_norm = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(
        vector_norm, np.clip(relative[:, 0], 1.0e-9, None)
    )
    axis_z = np.divide(
        relative[:, 3],
        vector_norm,
        out=np.zeros_like(vector_norm),
        where=vector_norm > 1.0e-9,
    )
    return np.rad2deg(angle * axis_z)


def _render_nonblank(direct, attempts: int = 16) -> np.ndarray:
    image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    for _ in range(attempts):
        image = np.asarray(direct.render(recompute=True), dtype=np.uint8)
        if image.size and float(image.std()) > 2.0:
            return image[:, :, :3]
    raise RuntimeError(
        f"Isaac render remained blank: shape={image.shape}, std={float(image.std()):.3f}"
    )


def _overlay(
    frame: np.ndarray,
    *,
    rank: int,
    episode: int,
    step: int,
    rotation_deg: float,
    contacts: np.ndarray,
    strict_hold: bool,
) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (16, 16), (660, 116), (18, 25, 33), thickness=-1)
    cv2.putText(
        result,
        f"Dex5 S0.12 | rank {rank:02d} | episode {episode}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        f"t={step / 60.0:.2f}s   actual quaternion rotation={rotation_deg:.1f} deg",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (205, 225, 242),
        1,
        cv2.LINE_AA,
    )
    active = "/".join(
        name for name, value in zip(CONTACT_NAMES, contacts, strict=True) if value
    ) or "none"
    hold_text = "STRICT HOLD" if strict_hold else "motion"
    color = (120, 240, 150) if strict_hold else (210, 215, 220)
    cv2.putText(
        result,
        f"contacts={active}   state={hold_text}",
        (30, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        color,
        1,
        cv2.LINE_AA,
    )
    return result


def _write_index(output: Path, records: list[dict], rollout: Path) -> None:
    sections = []
    for record in records:
        sections.append(
            f"""
<section>
  <div class="head">
    <div><span>#{record['rank']:02d}</span><h2>Episode {record['episode']}</h2></div>
    <dl><div><dt>实际旋转</dt><dd>{record['final_rotation_deg']:.2f}°</dd></div><div><dt>时长</dt><dd>{record['duration_s']:.2f}s</dd></div><div><dt>Source</dt><dd><code>{record['source_trajectory_id']}</code></dd></div></dl>
  </div>
  <video controls muted loop playsinline preload="metadata" poster="{html.escape(record['poster'])}"><source src="{html.escape(record['video'])}" type="video/mp4"></video>
</section>"""
        )
    output.joinpath("index.html").write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dex5 S0.12 转笔动作回放</title>
<style>:root{{--ink:#17202a;--muted:#59656f;--line:#d7dde2;--paper:#f4f6f7;--white:#fff;--blue:#1769aa;--green:#2f855a}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif;letter-spacing:0}}header{{background:var(--white);padding:34px max(22px,calc((100vw - 1050px)/2));border-bottom:1px solid var(--line)}}h1{{margin:0 0 8px;font-size:30px;letter-spacing:0}}header p{{margin:5px 0;max-width:860px;color:var(--muted)}}main{{max-width:1050px;margin:auto;padding:24px 22px 60px}}section{{padding:22px 0 34px;border-bottom:1px solid var(--line)}}.head{{display:flex;justify-content:space-between;align-items:end;gap:22px;margin-bottom:13px}}.head>div{{display:flex;align-items:center;gap:10px}}.head span{{display:grid;place-items:center;background:var(--blue);color:#fff;font-weight:800;width:42px;height:30px;border-radius:4px}}h2{{margin:0;font-size:21px;letter-spacing:0}}dl{{display:grid;grid-template-columns:repeat(3,minmax(105px,1fr));gap:18px;margin:0}}dt{{font-size:11px;color:var(--muted)}}dd{{margin:1px 0 0;font-weight:700}}video{{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#111}}code{{font:12px ui-monospace,monospace}}.note{{border-left:4px solid var(--green);background:#eef8f2;padding:12px 15px;margin:0 0 10px}}@media(max-width:700px){{h1{{font-size:25px}}.head{{align-items:flex-start;flex-direction:column}}dl{{width:100%;grid-template-columns:1fr 1fr}}}}</style></head>
<body><header><h1>Dex5 S0.12 转笔动作回放</h1><p>排名前 3 条 oracle replay 候选的 Isaac 真实网格逐帧状态回放。蓝色笔的姿态来自同步保存的 PhysX 四元数，不是角速度积分动画。</p><p>这是 P3 起始抓持到约 35° 短旋转与制动保持，不是完整一圈 PenSpin。</p></header><main><div class="note">视频按 60 Hz 记录、每两帧取一帧并以 30 fps 编码，因此保持原始仿真时间尺度。</div>{''.join(sections)}<p>rollout: <code>{html.escape(str(rollout.resolve()))}</code></p></main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("status") != "PASS_OFFLINE_SELECTION":
        raise RuntimeError("selection manifest is not accepted")
    if _sha256(args.rollout) != selection["rollout_sha256"]:
        raise RuntimeError("rollout hash mismatch")
    ranks = [int(value) for value in args.ranks.split(",")]
    selected = selection["selected_trajectories"]
    if any(rank < 1 or rank > len(selected) for rank in ranks):
        raise ValueError("rank outside selected trajectory list")

    with np.load(args.rollout, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.viewer.eye = (0.29, -0.31, 0.27)
    cfg.viewer.lookat = (0.0, 0.0, 0.115)
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.resolution = (args.width, args.height)
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
    direct = env.unwrapped
    env.reset()
    env_id = torch.tensor([0], device=direct.device)
    records: list[dict] = []

    for rank in ranks:
        row = selected[rank - 1]
        episode_id = int(row["episode_index"])
        start = int(row["row_start"])
        stop = int(row["row_stop"])
        q = data["joint_position"][start:stop]
        dq = data["joint_velocity"][start:stop]
        pen_pose = data["pen_pose"][start:stop]
        contacts = data["contact_state_next"][start:stop]
        rotation = _quaternion_rotation_deg(
            data["pen_pose_next"][start:stop, 3:7], pen_pose[0, 3:7]
        )
        hold_start = stop - start - int(row["strict_hold_steps"])
        raw_video = output / f"rank_{rank:02d}_episode_{episode_id:04d}_raw.mp4"
        final_video = output / f"rank_{rank:02d}_episode_{episode_id:04d}.mp4"
        poster = output / f"rank_{rank:02d}_episode_{episode_id:04d}.png"
        writer = cv2.VideoWriter(
            str(raw_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create {raw_video}")
        first_frame = None
        for local_step in range(0, stop - start, args.stride):
            joint_q = torch.as_tensor(q[local_step : local_step + 1], device=direct.device)
            joint_dq = torch.as_tensor(dq[local_step : local_step + 1], device=direct.device)
            pose = torch.as_tensor(
                pen_pose[local_step : local_step + 1], device=direct.device
            ).clone()
            pose[:, :3] += direct.scene.env_origins[env_id]
            direct.hand.write_joint_state_to_sim(
                joint_q, joint_dq, joint_ids=direct.dex5_joint_ids, env_ids=env_id
            )
            direct.hand.set_joint_position_target(
                joint_q, joint_ids=direct.dex5_joint_ids, env_ids=env_id
            )
            direct.pen.write_root_pose_to_sim(pose, env_id)
            direct.pen.write_root_velocity_to_sim(
                torch.zeros((1, 6), device=direct.device), env_id
            )
            direct.sim.forward()
            direct.scene.update(dt=0.0)
            frame = _overlay(
                _render_nonblank(direct),
                rank=rank,
                episode=episode_id,
                step=local_step,
                rotation_deg=float(rotation[local_step]),
                contacts=contacts[local_step],
                strict_hold=local_step >= hold_start,
            )
            if first_frame is None:
                first_frame = frame.copy()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        if first_frame is None:
            raise RuntimeError("trajectory generated no frames")
        Image.fromarray(first_frame).save(poster)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_video),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(final_video),
            ],
            check=True,
        )
        raw_video.unlink()
        records.append(
            {
                "rank": rank,
                "episode": episode_id,
                "source_trajectory_id": int(row["source_trajectory_id"]),
                "duration_s": (stop - start) / CONTROL_HZ,
                "final_rotation_deg": float(rotation[-1]),
                "video": final_video.name,
                "poster": poster.name,
                "video_sha256": _sha256(final_video),
            }
        )

    _write_index(output, records, args.rollout)
    manifest = {
        "schema": "dex5-s0.12-selected-motion-replay-v1",
        "status": "PASS",
        "rollout": str(args.rollout.resolve()),
        "rollout_sha256": _sha256(args.rollout),
        "selection": str(args.selection.resolve()),
        "selection_sha256": _sha256(args.selection),
        "task_scope": "P3 to approximately 35 degree short rotation and strict hold",
        "playback": "recorded joint and pen-pose state playback; no policy execution",
        "records": records,
        "hardware_commands_issued": False,
    }
    (output / "motion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
