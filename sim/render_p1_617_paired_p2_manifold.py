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


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
COLORS = ("#49b6cc", "#65c879", "#e7b84b", "#e78062", "#a97bc7", "#d8e1e6")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


def camera() -> mujoco.MjvCamera:
    result = mujoco.MjvCamera()
    result.type = mujoco.mjtCamera.mjCAMERA_FREE
    result.lookat[:] = (0.0, 0.072, 0.067)
    result.distance = 0.235
    result.azimuth = 142.0
    result.elevation = -23.0
    return result


def render_trajectory(qpos: np.ndarray) -> list[np.ndarray]:
    env = Dex5PenSpinMinimalEnv(model_path=MODEL, render_mode="rgb_array")
    view = camera()
    frames: list[np.ndarray] = []
    for row in qpos:
        mujoco.mj_resetData(env.model, env.data)
        env.data.qpos[env.qpos_adr] = row[:20]
        env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 7] = row[20:27]
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.renderer.update_scene(env.data, camera=view)
        frames.append(env.renderer.render())
    env.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 P1_617 到六模式 P2 teacher 中文可视化。")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.teacher_report.read_text(encoding="utf-8"))
    with np.load(args.teacher, allow_pickle=False) as data:
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        modes = np.asarray(data["manifold_mode"], dtype=np.int64)
        hand = np.asarray(data["hand_qpos"], dtype=np.float64)
        pen = np.asarray(data["pen_pose"], dtype=np.float64)
        feature = np.asarray(data["ring_feature"], dtype=np.float64) * 1000.0
        force = np.asarray(data["contact_force"], dtype=np.float64)

    representatives: dict[int, int] = {}
    final_frames: list[np.ndarray] = []
    selected_summary: dict[str, dict[str, object]] = {}
    for mode in range(6):
        rows = np.flatnonzero(modes == mode)
        local = int(np.argmin(np.abs(lengths[rows] - np.median(lengths[rows]))))
        trajectory = int(rows[local])
        representatives[mode] = trajectory
        length = int(lengths[trajectory])
        qpos = np.concatenate(
            (hand[trajectory, :length], pen[trajectory, :length]), axis=-1
        )
        sample_ids = np.unique(np.linspace(0, length - 1, min(length, 84), dtype=np.int64))
        frames = render_trajectory(qpos[sample_ids])
        imageio.mimsave(
            output / f"P2模式{mode}_waypoint路径.gif",
            frames,
            duration=1.0 / 36.0,
            loop=0,
        )
        final_frames.append(frames[-1])
        selected_summary[str(mode)] = {
            "teacher_row": trajectory,
            "length_steps": length,
            "duration_ms": length / 120.0 * 1000.0,
            "final_ring_feature_mm": feature[trajectory, length - 1].round(3).tolist(),
            "final_contact_force_n": force[trajectory, length - 1].round(4).tolist(),
        }

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), facecolor="#0d1012")
    for mode, axis in enumerate(axes.reshape(-1)):
        axis.imshow(final_frames[mode])
        axis.set_title(
            f"P2 模式 {mode}  |  ring-under 终点\n"
            f"teacher {representatives[mode]} · {selected_summary[str(mode)]['duration_ms']:.0f} ms",
            color="#eef2f4",
            fontsize=10,
        )
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(
        output / "六模式P2终点总览.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), facecolor="#0d1012")
    for axis in axes:
        axis.set_facecolor("#171b1e")
        axis.grid(color="#30363d", alpha=0.65)
        axis.tick_params(colors="#a8b0b6")
        for spine in axis.spines.values():
            spine.set_color("#30363d")
    for mode, trajectory in representatives.items():
        length = int(lengths[trajectory])
        time_ms = np.arange(length) / 120.0 * 1000.0
        axes[0].plot(time_ms, feature[trajectory, :length, 2], color=COLORS[mode], label=f"模式 {mode}")
        axes[1].plot(time_ms, force[trajectory, :length, 3], color=COLORS[mode], label=f"模式 {mode}")
    axes[0].axhspan(-26.0, -4.0, color="#65c879", alpha=0.10)
    axes[0].set_ylabel("ring 相对笔轴上下位置 (mm)", color="#eef2f4")
    axes[1].set_ylabel("无名指接触力 (N)", color="#eef2f4")
    axes[1].set_xlabel("时间 (ms)", color="#eef2f4")
    axes[0].legend(ncol=6, frameon=False, labelcolor="#eef2f4")
    figure.tight_layout()
    figure.savefig(output / "六模式waypoint物理轨迹.png", dpi=160, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    (output / "代表轨迹摘要.json").write_text(
        json.dumps(selected_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mode_links = "".join(
        f'<section><h2>P2 模式 {mode}</h2><img src="P2模式{mode}_waypoint路径.gif" '
        f'alt="P2 模式 {mode} waypoint 动画"></section>'
        for mode in range(6)
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P1_617 到六模式 P2 waypoint teacher</title>
<style>
:root{{--bg:#0d1012;--surface:#171b1e;--line:#30363d;--text:#eef2f4;--muted:#a8b0b6;--green:#65c879}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,"Noto Sans CJK SC",sans-serif}}
main{{width:min(1180px,calc(100vw - 32px));margin:auto;padding:26px 0 48px}}header{{border-bottom:1px solid var(--line);padding-bottom:18px}}
h1{{font-size:25px;margin:0 0 8px}}h2{{font-size:18px;margin:0 0 12px}}p{{color:var(--muted);line-height:1.65}}strong{{color:var(--green)}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--line)}}.metric{{padding:16px 12px;border-right:1px solid var(--line)}}
section{{padding:24px 0;border-bottom:1px solid var(--line)}}img{{display:block;width:100%;background:#000;border:1px solid var(--line)}}
@media(max-width:760px){{.metrics{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><header><h1>P1_617 到六模式 P2 waypoint teacher</h1>
<p>MuJoCo realmesh 用于几何可视化；轨迹数据、接触和成功验收来自 120 Hz PhysX 正常 step。</p></header>
<div class="metrics"><div class="metric"><strong>{report['trajectories']}</strong><br>teacher 轨迹</div>
<div class="metric"><strong>{report['transitions']}</strong><br>有效 transition</div>
<div class="metric"><strong>100 ms</strong><br>最近模式连续保持</div>
<div class="metric"><strong>0.005 rad</strong><br>TIM 最大补偿</div></div>
<section><h2>六模式终点</h2><img src="六模式P2终点总览.png"></section>
<section><h2>waypoint 物理轨迹</h2><img src="六模式waypoint物理轨迹.png"></section>{mode_links}
</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(output), "representatives": representatives}, ensure_ascii=False))


if __name__ == "__main__":
    main()
