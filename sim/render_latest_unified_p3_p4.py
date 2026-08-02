from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from dex5_penspin.sim.dex5_penspin_env import Dex5PenSpinMinimalEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the latest Isaac P3-to-P4 rollout in MuJoCo.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trajectory-index", type=int, default=0)
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(trajectory_path) as data:
        index = args.trajectory_index
        length = int(data["lengths"][index])
        qpos = np.asarray(data["qpos"][index, :length], dtype=np.float64)
        phase = np.asarray(data["phase_deg"][index, :length], dtype=np.float64)
        speed = np.asarray(data["angular_speed_rad_s"][index, :length], dtype=np.float64)
        contacts = np.asarray(data["contact_force_n"][index, :length], dtype=np.float64)
        hold = np.asarray(data["hold_count"][index, :length], dtype=np.int64)
        checkpoint = str(data["checkpoint"])

    model_path = (
        Path(__file__).resolve().parent
        / "assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
    )
    env = Dex5PenSpinMinimalEnv(model_path=model_path, render_mode="rgb_array")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.135)
    camera.distance = 0.24
    camera.azimuth = 142.0
    camera.elevation = -22.0
    frames: list[np.ndarray] = []
    for row in qpos:
        env.data.qpos[env.qpos_adr] = row[:20]
        env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 7] = row[20:27]
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.renderer.update_scene(env.data, camera=camera)
        frames.append(env.renderer.render())
    env.close()

    gif_path = output_dir / "latest_unified_p3_to_braked_p4.gif"
    imageio.mimsave(gif_path, frames, duration=1.0 / 30.0, loop=0)

    key_targets = np.asarray((0.0, 7.0, 12.0, 20.0, 35.0, 45.0, 50.0, 57.0))
    key_indices = sorted({int(np.argmin(np.abs(phase - target))) for target in key_targets})
    columns = 4
    rows = int(np.ceil(len(key_indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.6 * rows), facecolor="#101214")
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, frame_index in zip(axes, key_indices):
        axis.imshow(frames[frame_index])
        axis.set_title(
            f"step {frame_index:02d}  phase {phase[frame_index]:.1f} deg\n"
            f"speed {speed[frame_index]:.1f} rad/s  hold {hold[frame_index]}",
            color="#eef2f4",
            fontsize=10,
        )
        axis.axis("off")
    for axis in axes[len(key_indices) :]:
        axis.axis("off")
    figure.tight_layout()
    montage_path = output_dir / "latest_unified_p3_to_braked_p4_keyframes.png"
    figure.savefig(montage_path, dpi=150, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    time_s = np.arange(length) / 60.0
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor="#101214")
    for axis in axes:
        axis.set_facecolor("#191d20")
        axis.tick_params(colors="#a8b0b6")
        axis.grid(color="#30363d", alpha=0.8)
        for spine in axis.spines.values():
            spine.set_color("#30363d")
    axes[0].plot(time_s, phase, color="#63c5da", linewidth=2)
    axes[0].axhspan(50.0, 60.0, color="#7bd88f", alpha=0.16)
    axes[0].set_ylabel("phase (deg)", color="#eef2f4")
    axes[1].plot(time_s, speed, color="#ffd166", linewidth=2)
    axes[1].axhline(12.0, color="#ff7b72", linestyle="--", linewidth=1)
    axes[1].set_ylabel("angular speed", color="#eef2f4")
    names = ("thumb", "index", "middle", "ring", "pinky")
    colors = ("#63c5da", "#7bd88f", "#ffd166", "#f78c6c", "#c792ea")
    for finger, values, color in zip(names, contacts.T, colors):
        axes[2].plot(time_s, values, label=finger, color=color, linewidth=1.5)
    axes[2].set_ylabel("contact force (N)", color="#eef2f4")
    axes[2].set_xlabel("time (s)", color="#eef2f4")
    axes[2].legend(ncol=5, frameon=False, labelcolor="#eef2f4")
    figure.tight_layout()
    trace_path = output_dir / "latest_unified_p3_to_braked_p4_trace.png"
    figure.savefig(trace_path, dpi=160, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    summary = {
        "source": str(trajectory_path),
        "checkpoint": checkpoint,
        "trajectory_index": args.trajectory_index,
        "steps": length,
        "duration_s": length / 60.0,
        "phase_start_deg": float(phase[0]),
        "phase_final_deg": float(phase[-1]),
        "max_angular_speed_rad_s": float(speed.max()),
        "final_angular_speed_rad_s": float(speed[-1]),
        "final_hold_count": int(hold[-1]),
        "contact_ratio": {
            name: float(np.mean(values > 0.05)) for name, values in zip(names, contacts.T)
        },
        "artifacts": {
            "gif": gif_path.name,
            "keyframes": montage_path.name,
            "trace": trace_path.name,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
