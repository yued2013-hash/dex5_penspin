from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(os.environ.get("DEX5_ARTIFACT_ROOT", ROOT / "artifacts"))
DEFAULT_CHECKPOINT = (
    ARTIFACT_ROOT / "o1_23/checkpoints/o1_22_A_uniform_agent_1440.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/"
    "O1_23_典型P3到P4失败轨迹"
)

parser = argparse.ArgumentParser(description="Render one representative O1.23 failure rollout.")
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--bank-id", type=int, default=996)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--render-backend", choices=("rtx", "software"), default="software")
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
from PIL import Image, ImageDraw  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


TASK = "Dex5-PenSpin-Oracle-P3Temporal-Direct-v0"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
COLORS = ("#d4503f", "#2f7ed8", "#31a354", "#d99018", "#8c6bb1")


def _render_nonblank(direct, attempts: int = 16) -> np.ndarray:
    image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    for _ in range(attempts):
        image = np.asarray(direct.render(recompute=True), dtype=np.uint8)
        if image.size and float(image.std()) > 2.0:
            return image[:, :, :3]
    raise RuntimeError(
        f"Isaac RTX render remained blank: shape={image.shape}, std={float(image.std()):.3f}"
    )


def _quat_rotate_x(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return np.asarray(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y))
    )


def _render_software(
    body_names: list[str],
    body_positions: np.ndarray,
    pen_pose: np.ndarray,
    contacts: np.ndarray,
) -> np.ndarray:
    positions = {name: body_positions[index] for index, name in enumerate(body_names)}
    palm_names = [name for name in body_names if "palm" in name.lower() or "base" in name.lower()]
    palm = positions[palm_names[0]] if palm_names else np.mean(body_positions[:2], axis=0)
    chains = {}
    for finger_index, finger in enumerate(FINGERS, start=1):
        names = [f"Link_{finger_index}{joint}R" for joint in range(1, 5)]
        chain = [palm, *(positions[name] for name in names if name in positions)]
        if len(chain) >= 2:
            chains[finger] = np.asarray(chain)
    pen_center = np.asarray(pen_pose[:3], dtype=np.float64)
    pen_axis = _quat_rotate_x(pen_pose[3:7])
    pen_ends = np.stack((pen_center - 0.07075 * pen_axis, pen_center + 0.07075 * pen_axis))
    eye = np.asarray((0.29, -0.31, 0.27), dtype=np.float64)
    target = np.asarray((0.0, 0.0, 0.115), dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    visible = np.concatenate((*chains.values(), pen_ends), axis=0)
    relative = visible - target
    visible_x, visible_y = relative @ right, relative @ up
    center_x = 0.5 * (float(visible_x.min()) + float(visible_x.max()))
    center_y = 0.5 * (float(visible_y.min()) + float(visible_y.max()))
    scale = min(
        0.76 * args.width / max(float(np.ptp(visible_x)), 0.06),
        0.62 * args.height / max(float(np.ptp(visible_y)), 0.06),
    )

    def project(points: np.ndarray) -> list[tuple[int, int]]:
        local = np.asarray(points) - target
        x = args.width * 0.5 + scale * ((local @ right) - center_x)
        y = args.height * 0.58 - scale * ((local @ up) - center_y)
        return [(int(px), int(py)) for px, py in zip(x, y, strict=True)]

    canvas = Image.new("RGB", (args.width, args.height), (235, 239, 242))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, int(args.height * 0.78), args.width, args.height), fill=(211, 217, 221))
    for index, (finger, color_hex) in enumerate(zip(FINGERS, COLORS, strict=True)):
        if finger not in chains:
            continue
        color = tuple(int(color_hex.lstrip("#")[offset : offset + 2], 16) for offset in (0, 2, 4))
        pixels = project(chains[finger])
        draw.line(pixels, fill=color, width=14, joint="curve")
        for point in pixels:
            draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=(250, 250, 250), outline=color, width=3)
        if bool(contacts[index]):
            point = pixels[-1]
            draw.ellipse((point[0] - 13, point[1] - 13, point[0] + 13, point[1] + 13), outline=(20, 150, 80), width=5)
    draw.line(project(pen_ends), fill=(225, 45, 45), width=13)
    center = project(pen_center[None])[0]
    draw.ellipse((center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5), fill=(20, 20, 20))
    return np.asarray(canvas)


def _overlay(
    frame: np.ndarray,
    step: int,
    height_mm: float,
    angle_deg: float,
    contacts: np.ndarray,
    forces: np.ndarray,
) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (14, 14), (710, 132), (18, 25, 33), thickness=-1)
    cv2.putText(
        result,
        f"O1.23 failure | P3 bank {args.bank_id} | t={step / 60.0:.2f}s",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        f"height={height_mm:+.1f} mm   integrated omega_z={angle_deg:+.1f} deg",
        (28, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (205, 225, 242),
        1,
        cv2.LINE_AA,
    )
    active = "/".join(name[0].upper() for name, value in zip(FINGERS, contacts) if value) or "none"
    cv2.putText(
        result,
        f"debounced contact={active}   T/I/M force={forces[0]:.2f}/{forces[1]:.2f}/{forces[2]:.2f} N",
        (28, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (185, 235, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "Failure: contact switching occurs, but load and height are not recovered",
        (28, 124),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (238, 190, 178),
        1,
        cv2.LINE_AA,
    )
    return result


def _save_chart(output: Path, trace: dict[str, np.ndarray]) -> None:
    time = np.arange(trace["height_m"].size) / 60.0
    height = 1000.0 * (trace["height_m"] - trace["height_m"][0])
    angle = np.cumsum(trace["angular_velocity"][:, 2]) / 60.0 * 180.0 / np.pi
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), dpi=150, sharex=True)
    axes[0].plot(time, height, color="#c43c35", linewidth=2.2)
    axes[0].axhline(-5.0, color="#777", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Height change (mm)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(time, angle, color="#1769aa", linewidth=2.0, label="integrated omega_z")
    axes[1].plot(
        time,
        np.linalg.norm(trace["angular_velocity"][:, :2], axis=1),
        color="#d99018",
        linewidth=1.4,
        label="wrong-axis speed",
    )
    axes[1].set_ylabel("deg / rad s^-1")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    for finger, color, values in zip(FINGERS, COLORS, trace["force"].T):
        axes[2].plot(time, values, color=color, linewidth=1.5, label=finger)
    axes[2].set_ylabel("Contact force (N)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(ncol=5, frameon=False)
    axes[2].grid(alpha=0.25)
    figure.suptitle(f"O1.23 P4 rotation to P5 capture failure | bank {args.bank_id}")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _write_index(output: Path, metrics: dict[str, object]) -> None:
    output.joinpath("index.html").write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>O1.23 P4 绕行→P5 捕获失败轨迹</title><style>
body{{margin:0;background:#eef1f4;color:#17202a;font:15px/1.6 system-ui,sans-serif}}
header,main{{max-width:1080px;margin:auto;padding:24px}}header{{background:#fff;max-width:none;padding-left:max(24px,calc((100vw - 1032px)/2))}}
h1{{margin:0 0 8px;font-size:28px;letter-spacing:0}}p{{margin:7px 0}}video,img{{display:block;width:100%;background:#111;margin:18px 0}}
.note{{border-left:4px solid #c43c35;padding:10px 14px;background:#fff}}code{{font-family:ui-monospace,monospace}}
</style></head><body><header><h1>O1.23 P4 绕行→P5 捕获失败轨迹</h1>
<p>固定 P3 bank <code>{args.bank_id}</code>，使用 O1.22-A uniform checkpoint 确定性运行。</p></header><main>
<div class="note"><strong>观察重点：</strong>食指/中指完成预期的 P4 绕行；笔端进入拇指与食指之间的 P5 捕获窗口后，策略没有切换为拇食指捏持，而是继续推进，并出现疑似穿透后掉笔。</div>
<video controls muted loop playsinline preload="metadata" poster="failure_poster.png"><source src="failure_trajectory.mp4" type="video/mp4"></video>
<p>最大高度下降：<strong>{float(metrics['max_height_loss_mm']):.2f} mm</strong>；累计目标轴转角：<strong>{float(metrics['final_angle_deg']):.1f}°</strong>；
最大连续无接触：<strong>{int(metrics['max_no_contact_steps'])} steps</strong>。</p>
<img src="failure_metrics.png" alt="接触力、高度和旋转曲线">
<p>这是 simulation policy failure visualization，不是成功轨迹、teacher 或真实机器人动作。</p>
</main></body></html>""",
        encoding="utf-8",
    )


@hydra_task_config(TASK, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.seed = 20260730
    env_cfg.fixed_reset_bank_id = args.bank_id
    env_cfg.episode_length_s = (args.steps + 10) / 60.0
    env_cfg.penetration_threshold = 1.0
    env_cfg.drop_threshold = 10.0
    env_cfg.drift_threshold = 10.0
    env_cfg.no_contact_termination_steps = args.steps + 10
    env_cfg.throw_linear_speed_m_s = 1000.0
    env_cfg.throw_angular_speed_rad_s = 1000.0
    env_cfg.terminate_on_ballistic = False
    render_mode = None
    if args.render_backend == "rtx":
        env_cfg.viewer.eye = (0.29, -0.31, 0.27)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.115)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.resolution = (args.width, args.height)
        render_mode = "rgb_array"
    experiment_cfg["seed"] = 20260730
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    experiment_cfg["trainer"]["close_environment_at_exit"] = False

    raw_env = gym.make(TASK, cfg=env_cfg, render_mode=render_mode)
    direct = raw_env.unwrapped
    body_names = list(direct.hand.body_names)
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, experiment_cfg)
    runner.agent.load(str(args.checkpoint.resolve()))
    runner.agent.enable_training_mode(False, apply_to_models=True)
    obs, _ = env.reset()
    states = env.state()
    initial_height = float(direct.pen_pos[0, 2].item())
    raw_video = output / "failure_trajectory_raw.mp4"
    final_video = output / "failure_trajectory.mp4"
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {raw_video}")
    trace = {
        "height_m": [],
        "angular_velocity": [],
        "force": [],
        "contact": [],
        "joint_position": [],
        "joint_velocity": [],
        "pen_pose": [],
        "pen_linear_velocity": [],
    }
    angle_deg = 0.0
    first_frame = None
    no_contact_run = max_no_contact_run = 0
    for step in range(args.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
            states = env.state()
        if bool(torch.any(terminated | truncated)):
            raise RuntimeError(f"unexpected reset at step {step}")
        height = float(direct.pen_pos[0, 2].item())
        angular = direct.pen_angvel[0].detach().cpu().numpy().copy()
        force = direct.contact_force_norm[0].detach().cpu().numpy().copy()
        contact = direct.debounced_contacts[0].detach().cpu().numpy().copy()
        angle_deg += float(angular[2]) / 60.0 * 180.0 / np.pi
        no_contact_run = no_contact_run + 1 if not bool(np.any(contact)) else 0
        max_no_contact_run = max(max_no_contact_run, no_contact_run)
        trace["height_m"].append(height)
        trace["angular_velocity"].append(angular)
        trace["force"].append(force)
        trace["contact"].append(contact)
        trace["joint_position"].append(
            direct.hand_qpos[0].detach().cpu().numpy().copy()
        )
        trace["joint_velocity"].append(
            direct.hand_qvel[0].detach().cpu().numpy().copy()
        )
        trace["pen_pose"].append(
            direct.pen.data.root_pose_w[0].detach().cpu().numpy().copy()
        )
        trace["pen_linear_velocity"].append(
            direct.pen_linvel[0].detach().cpu().numpy().copy()
        )
        body_positions = direct.hand.data.body_pos_w[0].detach().cpu().numpy().copy()
        pen_pose = direct.pen.data.root_pose_w[0].detach().cpu().numpy().copy()
        base_frame = (
            _render_nonblank(direct)
            if args.render_backend == "rtx"
            else _render_software(body_names, body_positions, pen_pose, contact)
        )
        frame = _overlay(
            base_frame,
            step,
            1000.0 * (height - initial_height),
            angle_deg,
            contact,
            force,
        )
        if first_frame is None:
            first_frame = frame.copy()
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    if first_frame is None:
        raise RuntimeError("rollout produced no frames")
    Image.fromarray(first_frame).save(output / "failure_poster.png")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_video),
        ],
        check=True,
    )
    raw_video.unlink()
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    np.savez_compressed(
        output / "failure_trajectory_states.npz",
        artifact_role=np.asarray("visualization_only_not_teacher_not_reset"),
        source_simulator=np.asarray("Isaac Lab PhysX policy rollout"),
        bank_id=np.asarray(args.bank_id),
        control_hz=np.asarray(60),
        **arrays,
    )
    _save_chart(output / "failure_metrics.png", arrays)
    metrics = {
        "bank_id": args.bank_id,
        "steps": args.steps,
        "max_height_loss_mm": float(1000.0 * (initial_height - np.min(arrays["height_m"]))),
        "final_angle_deg": angle_deg,
        "max_no_contact_steps": max_no_contact_run,
        "final_contact": dict(zip(FINGERS, arrays["contact"][-1].tolist(), strict=True)),
        "artifact_role": "visualization_only_not_teacher_not_reset",
        "render_backend": args.render_backend,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    (output / "manifest.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_index(output, metrics)
    print("O1_23_FAILURE_VISUAL=" + json.dumps(metrics, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
