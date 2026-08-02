from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from dex5_penspin.sim.dex5_penspin_env import Dex5PenSpinMinimalEnv


def interpolate_sequence(states: list[dict], samples: int = 18) -> list[dict]:
    frames = []
    for segment in range(len(states) - 1):
        first = states[segment]
        second = states[segment + 1]
        for alpha in np.linspace(0.0, 1.0, samples, endpoint=segment == len(states) - 2):
            hand = (1.0 - alpha) * np.asarray(first["hand_qpos"]) + alpha * np.asarray(second["hand_qpos"])
            pos = (1.0 - alpha) * np.asarray(first["pen_pos"]) + alpha * np.asarray(second["pen_pos"])
            quat = (1.0 - alpha) * np.asarray(first["pen_quat"]) + alpha * np.asarray(second["pen_quat"])
            quat /= np.linalg.norm(quat)
            frames.append({"hand_qpos": hand, "pen_pos": pos, "pen_quat": quat})
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = {int(row["phase"]): row for row in manifest["phases"]}

    p1 = phases[1]
    p2_frames = [
        ("P1 三指控笔", p1),
        ("P2a 无名指清障", phases[2]["transition_keyframes"][0]),
        ("P2b 无名指到达笔下", phases[2]["transition_keyframes"][1]),
    ]
    p3_frames = [
        ("P2 拇指/中指/无名指支撑", phases[2]),
        ("P3a 食指向外清障", phases[3]["transition_keyframes"][0]),
        ("P3c 类 Hold-Oracle 双指主动夹持", phases[3]["transition_keyframes"][-1]),
    ]
    p5_frames = [
        ("P5a 拇指/食指捏持", phases[5]["transition_keyframes"][0]),
        ("P5b 中指/无名指向后回收", phases[5]["transition_keyframes"][1]),
        ("P1 参考姿态", p1),
    ]
    sequences = [p2_frames, p3_frames, p5_frames]

    model_path = Path(__file__).resolve().parent / "assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
    env = Dex5PenSpinMinimalEnv(model_path=model_path, render_mode="rgb_array")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.075, 0.065)
    camera.distance = 0.225
    camera.azimuth = 205.0
    camera.elevation = -20.0

    rendered: list[list[np.ndarray]] = []
    for sequence in sequences:
        rows = []
        for _, state in sequence:
            env.data.qpos[env.qpos_adr] = np.asarray(state["hand_qpos"], dtype=np.float64)
            env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = np.asarray(
                state["pen_pos"], dtype=np.float64
            )
            env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = np.asarray(
                state["pen_quat"], dtype=np.float64
            )
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            env.renderer.update_scene(env.data, camera=camera)
            rows.append(env.renderer.render())
        rendered.append(rows)
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor="#0d1012")
    for row_index, (sequence, images) in enumerate(zip(sequences, rendered)):
        for column, ((label, _), image) in enumerate(zip(sequence, images)):
            axis = axes[row_index, column]
            axis.imshow(image)
            axis.set_title(label, color="#eef2f4", fontsize=12)
            axis.axis("off")
    figure.tight_layout()
    montage = output_dir / "canonical_p2_p3_p5_v4_transition_montage.png"
    figure.savefig(montage, dpi=160, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    animation_states = [
        ("p1_to_p2_v3.gif", [state for _, state in p2_frames]),
        ("p2_to_p3_v4.gif", [phases[2], *phases[3]["transition_keyframes"]]),
        ("p5_return_v3.gif", [state for _, state in p5_frames[:2]]),
    ]
    for filename, states in animation_states:
        frames = []
        for state in interpolate_sequence(states):
            env.data.qpos[env.qpos_adr] = state["hand_qpos"]
            env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = state["pen_pos"]
            env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = state["pen_quat"]
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            env.renderer.update_scene(env.data, camera=camera)
            frames.append(env.renderer.render())
        imageio.mimsave(output_dir / filename, frames, duration=1 / 18, loop=0)
    env.close()
    print(montage)


if __name__ == "__main__":
    main()
