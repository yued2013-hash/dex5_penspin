from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import font_manager

from dex5_penspin.sim.dex5_penspin_env import Dex5PenSpinMinimalEnv


font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


def render_frames(qpos: np.ndarray, model_path: Path) -> list[np.ndarray]:
    env = Dex5PenSpinMinimalEnv(
        model_path=model_path, render_mode="rgb_array"
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.135)
    camera.distance = 0.24
    camera.azimuth = 142.0
    camera.elevation = -22.0
    frames: list[np.ndarray] = []
    for row in qpos:
        env.data.qpos[env.qpos_adr] = row[:20]
        env.data.qpos[
            env.pen_qpos_adr : env.pen_qpos_adr + 7
        ] = row[20:27]
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.renderer.update_scene(env.data, camera=camera)
        frames.append(env.renderer.render())
    env.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="渲染 P4 到 P5 同 episode 无 reset 动态轨迹。"
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    with np.load(args.trajectory) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        qvel = np.asarray(data["qvel"], dtype=np.float64)
        phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)
        penetration = np.asarray(
            data["max_penetration_m"], dtype=np.float64
        )
        control_dt = float(data["control_dt_s"])

    valid = (phase_ids > 0) & (
        np.linalg.norm(qpos[:, 20:23], axis=1) > 1e-6
    )
    qpos = qpos[valid]
    qvel = qvel[valid]
    phase_ids = phase_ids[valid]
    penetration = penetration[valid]

    model_path = Path(__file__).resolve().parent / (
        "assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
    )
    frames = render_frames(qpos, model_path)
    gif_path = output_dir / "P4到P5无重置代表轨迹.gif"
    imageio.mimsave(
        gif_path, frames, duration=1.0 / 30.0, loop=0
    )

    indices = np.linspace(0, len(frames) - 1, 8).round().astype(int)
    figure, axes = plt.subplots(
        2, 4, figsize=(14, 7.2), facecolor="#0d1012"
    )
    for axis, index in zip(axes.reshape(-1), indices, strict=True):
        axis.imshow(frames[index])
        phase_name = (
            "P4 中指退让"
            if phase_ids[index] == 1
            else "P5 拇食指捕获"
        )
        axis.set_title(
            f"{index * control_dt * 1000:.0f} ms · {phase_name}",
            color="#eef2f4",
            fontsize=10,
        )
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(
        output_dir / "P4到P5无重置关键帧.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    time_ms = np.arange(len(qpos)) * control_dt * 1000.0
    speed = np.linalg.norm(qvel[:, 23:26], axis=1)
    figure, axes = plt.subplots(
        2, 1, figsize=(10.5, 6.2), sharex=True, facecolor="#0d1012"
    )
    for axis in axes:
        axis.set_facecolor("#171b1e")
        axis.tick_params(colors="#a8b0b6")
        axis.grid(color="#30363d", alpha=0.75)
        for spine in axis.spines.values():
            spine.set_color("#30363d")
    axes[0].plot(time_ms, speed, color="#e7b84b", linewidth=2)
    axes[0].axhline(12.0, color="#e78062", linestyle="--")
    axes[0].set_ylabel("角速度 (rad/s)", color="#eef2f4")
    axes[1].plot(
        time_ms, penetration * 1000.0, color="#e78062", linewidth=2
    )
    axes[1].axhline(0.5, color="#eef2f4", linestyle="--")
    axes[1].set_ylabel("最大穿透 (mm)", color="#eef2f4")
    axes[1].set_xlabel("时间 (ms)", color="#eef2f4")
    figure.tight_layout()
    figure.savefig(
        output_dir / "P4到P5无重置物理曲线.png",
        dpi=160,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    summary = {
        "轨迹帧数": len(qpos),
        "真实时长_ms": len(qpos) * control_dt * 1000.0,
        "可视化播放": "30 FPS 慢放",
        "最大轨迹穿透_mm": float(penetration.max() * 1000.0),
        "正式成功率": report["cascade_success_rate"],
        "正式样本": report["episodes"],
        "正式成功数": report["cascade_successes"],
        "状态重置": "无",
    }
    (output_dir / "动态轨迹摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Dex5 P4 到 P5 无重置动态视图</title>
<style>
body{{margin:0;background:#0d1012;color:#eef2f4;font-family:"Noto Sans CJK SC",sans-serif}}
main{{max-width:1180px;margin:auto;padding:30px}} h1{{font-size:30px;margin:0 0 8px}}
p{{color:#aeb7bd}} .grid{{display:grid;grid-template-columns:1.25fr .75fr;gap:18px}}
figure,section{{margin:0 0 18px;background:#15191c;border:1px solid #343b40;padding:16px;border-radius:6px}}
img{{display:block;width:100%;height:auto}} table{{width:100%;border-collapse:collapse}}
td{{padding:11px;border-bottom:1px solid #30363d}} td:last-child{{text-align:right;font-weight:700}}
.pass{{color:#65c879}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Dex5：P4 → P5 同 episode 动态视图</h1>
<p>真实 PhysX 状态轨迹投影到 Dex5 网格；手和笔均未在 primitive 切换处重置。</p>
<div class="grid"><figure><img src="{html.escape(gif_path.name)}"><figcaption>轨迹按 30 FPS 慢放，真实控制周期为 {control_dt * 1000:.2f} ms。</figcaption></figure>
<section><table>
<tr><td>正式成功率</td><td class="pass">{report["cascade_success_rate"] * 100:.2f}%</td></tr>
<tr><td>样本</td><td>{report["cascade_successes"]}/{report["episodes"]}</td></tr>
<tr><td>代表轨迹时长</td><td>{len(qpos) * control_dt * 1000:.0f} ms</td></tr>
<tr><td>代表轨迹最大穿透</td><td>{penetration.max() * 1000:.3f} mm</td></tr>
<tr><td>掉笔计数</td><td>{report["drop_counter"]}</td></tr>
<tr><td>状态 reset</td><td class="pass">无</td></tr>
</table></section></div>
<figure><img src="P4到P5无重置关键帧.png"></figure>
<figure><img src="P4到P5无重置物理曲线.png"></figure>
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
