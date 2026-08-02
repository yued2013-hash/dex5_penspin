from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "Dex5_Sim2Real_S0/oracle_dataset/Dex5_S0_Oracle_Success_Rollouts_1200.npz"
DEFAULT_OUTPUT = ROOT / "Dex5_Sim2Real_S0/oracle_trajectory_visualization"

parser = argparse.ArgumentParser(description="Render successful S0 oracle trajectories with Isaac RTX meshes.")
parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--episodes", default="auto", help="comma-separated episode IDs or auto")
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
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


TASK = "Dex5-S0.2-P3HardwareCompat-Audit-Direct-v0"
CONTACT_NAMES = ("thumb", "index", "middle", "ring", "pinky")
CONTACT_COLORS = ("#d4503f", "#2f7ed8", "#31a354", "#d99018", "#8c6bb1")


def select_episodes(data: dict[str, np.ndarray]) -> list[tuple[int, str]]:
    if args.episodes != "auto":
        return [(int(value), f"episode {value}") for value in args.episodes.split(",")]
    offsets = data["episode_offsets"]
    contact = data["contact_state"].astype(np.int8)
    pose = data["pen_pose"]
    rows = []
    for episode, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        q0 = pose[start, 3:7] / np.linalg.norm(pose[start, 3:7])
        q1 = pose[end - 1, 3:7] / np.linalg.norm(pose[end - 1, 3:7])
        endpoint_angle = 2.0 * np.arccos(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
        transitions = int(np.abs(np.diff(contact[start:end], axis=0)).sum())
        rows.append((episode, endpoint_angle, transitions))
    ordered = sorted(rows, key=lambda row: row[1])
    typical = ordered[len(ordered) // 2][0]
    contact_active = max(rows, key=lambda row: row[2])[0]
    largest_pose_change = max(rows, key=lambda row: row[1])[0]
    return [
        (typical, "典型成功轨迹"),
        (contact_active, "接触变化最多"),
        (largest_pose_change, "姿态变化最大"),
    ]


def render_nonblank(direct, attempts: int = 12) -> np.ndarray:
    image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    for _ in range(attempts):
        image = np.asarray(direct.render(recompute=True), dtype=np.uint8)
        if image.size and float(image.std()) > 2.0:
            return image[:, :, :3]
    raise RuntimeError(f"RTX render remained blank: shape={image.shape}, std={float(image.std()):.3f}")


def accumulated_angle_deg(angular_velocity: np.ndarray) -> np.ndarray:
    # Dataset snapshots are recorded once per 60 Hz policy step. S0 defines
    # the target rotation as the integral of world-frame pen omega_z.
    return np.cumsum(angular_velocity[:, 2]) / 60.0 * 180.0 / np.pi


def save_chart(path: Path, episode: dict[str, np.ndarray], title: str) -> dict[str, float]:
    time_s = np.arange(len(episode["joint_position"])) / 60.0
    angle = accumulated_angle_deg(episode["pen_angular_velocity"])
    height_mm = (episode["pen_pose"][:, 2] - episode["pen_pose"][0, 2]) * 1000.0
    target_speed = episode["pen_angular_velocity"][:, 2]
    wrong_speed = np.linalg.norm(episode["pen_angular_velocity"][:, :2], axis=1)
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), dpi=140, sharex=True)
    axes[0].plot(time_s, angle, color="#1769aa", linewidth=2.2, label="累计目标轴角度")
    axes[0].axhspan(35.0, 55.0, color="#31a354", alpha=0.14, label="35-55° success band")
    axes[0].set_ylabel("角度 (deg)")
    axes[0].legend(loc="best", frameon=False)
    axes[1].plot(time_s, height_mm, color="#c43c35", linewidth=2.0, label="笔高度变化")
    axes[1].plot(time_s, target_speed, color="#1769aa", linewidth=1.5, label="目标轴角速度")
    axes[1].plot(time_s, wrong_speed, color="#d99018", linewidth=1.5, label="非目标轴角速度")
    axes[1].set_ylabel("mm / rad s$^{-1}$")
    axes[1].legend(loc="best", ncol=3, frameon=False)
    for finger, color, values in zip(CONTACT_NAMES, CONTACT_COLORS, episode["contact_state"].T):
        axes[2].step(time_s, values.astype(float), where="post", label=finger, color=color, linewidth=1.6)
    axes[2].set_ylim(-0.1, 1.15)
    axes[2].set_ylabel("接触")
    axes[2].set_xlabel("时间 (s)")
    axes[2].legend(loc="upper right", ncol=5, frameon=False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return {
        "duration_s": len(time_s) / 60.0,
        "final_accumulated_angle_deg": float(angle[-1]),
        "peak_accumulated_angle_deg": float(np.max(angle)),
        "max_height_loss_mm": float(max(0.0, -np.min(height_mm))),
        "contact_transition_count": int(np.abs(np.diff(episode["contact_state"].astype(np.int8), axis=0)).sum()),
    }


def overlay(frame: np.ndarray, episode_id: int, label: str, step: int, angle: float, contacts: np.ndarray) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (14, 14), (640, 108), (18, 25, 33), thickness=-1)
    cv2.putText(result, f"S0 oracle | episode {episode_id} | {label}", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"t={step / 60.0:.2f}s   accumulated omega_z={angle:.1f} deg", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (205, 225, 242), 1, cv2.LINE_AA)
    active = "/".join(name for name, value in zip(CONTACT_NAMES, contacts) if value) or "none"
    cv2.putText(result, f"saved contacts: {active}", (28, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (185, 235, 190), 1, cv2.LINE_AA)
    return result


def make_index(output: Path, records: list[dict[str, object]], dataset: Path) -> None:
    sections = []
    for record in records:
        metrics = record["metrics"]
        sections.append(f"""
<section>
  <div class="heading"><div><h2>{html.escape(str(record['label']))}</h2><p>episode <code>{record['episode_id']}</code> · {metrics['duration_s']:.2f}s · peak {metrics['peak_accumulated_angle_deg']:.1f}°</p></div><span>SIMULATION ROLLOUT</span></div>
  <video controls muted loop playsinline preload="metadata" poster="{record['poster']}"><source src="{record['video']}" type="video/mp4"></video>
  <img class="chart" src="{record['chart']}" alt="trajectory metrics">
</section>""")
    output.joinpath("index.html").write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dex5 S0 Oracle 训练轨迹</title>
<style>
:root{{--ink:#17202a;--muted:#5b6670;--line:#d7dde2;--paper:#f4f6f7;--accent:#1769aa}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}header{{padding:34px max(24px,calc((100vw - 1120px)/2));background:#fff;border-bottom:1px solid var(--line)}}h1{{margin:0 0 8px;font-size:30px;letter-spacing:0}}header p{{margin:4px 0;color:var(--muted)}}main{{max-width:1120px;margin:0 auto;padding:26px 24px 60px}}section{{padding:24px 0 34px;border-bottom:1px solid var(--line)}}.heading{{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:14px}}h2{{margin:0;font-size:21px;letter-spacing:0}}.heading p{{margin:3px 0 0;color:var(--muted)}}.heading span{{font:600 11px/1 system-ui;color:#fff;background:var(--accent);padding:7px 9px;border-radius:4px}}video{{display:block;width:100%;aspect-ratio:4/3;background:#111;object-fit:contain}}.chart{{display:block;width:100%;margin-top:14px;background:#fff;border:1px solid var(--line)}}code{{font-family:ui-monospace,monospace}}@media(max-width:620px){{.heading{{align-items:start;flex-direction:column}}}}
</style></head><body><header><h1>Dex5 S0 Oracle 训练轨迹</h1><p>Isaac RTX 真实网格回放，来源为冻结的 1200 条成功 simulation rollout。</p><p>任务是 P3 → 45°±10° 短旋转 → 制动保持；不是完整转笔、handoff 或真机回放。</p><p>dataset: <code>{html.escape(str(dataset))}</code></p></header><main>{''.join(sections)}</main></body></html>""", encoding="utf-8")


def main() -> None:
    dataset = args.dataset.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    selections = select_episodes(data)

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
    records: list[dict[str, object]] = []

    for episode_id, label in selections:
        start = int(data["episode_offsets"][episode_id])
        end = int(data["episode_offsets"][episode_id + 1])
        episode = {key: value[start:end] for key, value in data.items() if value.shape[:1] == data["joint_position"].shape[:1]}
        chart = output / f"episode_{episode_id:04d}_metrics.png"
        metrics = save_chart(chart, episode, f"{label} | episode {episode_id}")
        angle = accumulated_angle_deg(episode["pen_angular_velocity"])
        raw_video = output / f"episode_{episode_id:04d}_raw.mp4"
        final_video = output / f"episode_{episode_id:04d}.mp4"
        poster = output / f"episode_{episode_id:04d}.png"
        writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to create {raw_video}")
        first_frame = None
        for row in range(0, end - start, args.stride):
            q = torch.as_tensor(episode["joint_position"][row:row + 1], device=direct.device)
            dq = torch.as_tensor(episode["joint_velocity"][row:row + 1], device=direct.device)
            pen_pose = torch.as_tensor(episode["pen_pose"][row:row + 1], device=direct.device).clone()
            pen_pose[:, :3] += direct.scene.env_origins[env_id]
            pen_velocity = torch.as_tensor(
                np.concatenate((episode["pen_linear_velocity"][row:row + 1], episode["pen_angular_velocity"][row:row + 1]), axis=1),
                device=direct.device,
            )
            direct.hand.write_joint_state_to_sim(q, dq, joint_ids=direct.dex5_joint_ids, env_ids=env_id)
            direct.hand.set_joint_position_target(q, joint_ids=direct.dex5_joint_ids, env_ids=env_id)
            direct.pen.write_root_pose_to_sim(pen_pose, env_id)
            direct.pen.write_root_velocity_to_sim(pen_velocity, env_id)
            direct.sim.forward()
            direct.scene.update(dt=0.0)
            frame = overlay(render_nonblank(direct), episode_id, label, row, float(angle[row]), episode["contact_state"][row])
            if first_frame is None:
                first_frame = frame.copy()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        if first_frame is None:
            raise RuntimeError(f"episode {episode_id} produced no frame")
        Image.fromarray(first_frame).save(poster)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_video),
        ], check=True)
        raw_video.unlink()
        records.append({
            "episode_id": episode_id,
            "label": label,
            "video": final_video.name,
            "poster": poster.name,
            "chart": chart.name,
            "metrics": metrics,
        })

    make_index(output, records, dataset)
    manifest = {
        "status": "PASS",
        "dataset": str(dataset),
        "task_scope": "P3 constrained short rotation; no handoff and no real replay",
        "render_backend": "Isaac RTX mesh state playback",
        "episodes": records,
    }
    output.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("S0_TRAJECTORY_GALLERY=" + json.dumps(manifest, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
