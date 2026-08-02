from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import font_manager

from dex5_penspin.sim.dex5_penspin_env import Dex5PenSpinMinimalEnv


FINGER_NAMES = ("拇指", "食指", "中指", "无名指", "小指")
FINGER_COLORS = ("#49b6cc", "#65c879", "#e7b84b", "#e78062", "#a97bc7")
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


def terminal_hold_count(
    phase: np.ndarray,
    speed: np.ndarray,
    contact: np.ndarray,
    penetration: np.ndarray,
) -> np.ndarray:
    gate = (
        (phase >= 50.0)
        & (phase <= 60.0)
        & (speed <= 12.0)
        & np.all(contact[:, :3] > 0.05, axis=1)
        & (penetration <= 0.0005)
    )
    hold = np.zeros(len(gate), dtype=np.int64)
    for index in range(len(gate)):
        hold[index] = hold[index - 1] + 1 if gate[index] else 0
    return hold


def render_frames(qpos: np.ndarray, model_path: Path) -> list[np.ndarray]:
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
    return frames


def save_keyframes(
    frames: list[np.ndarray],
    phase: np.ndarray,
    speed: np.ndarray,
    hold: np.ndarray,
    output: Path,
) -> None:
    targets = np.asarray((0.0, 7.0, 20.0, 30.0, 40.0, 45.0, 50.0, 52.0))
    indices = sorted({int(np.argmin(np.abs(phase - target))) for target in targets})
    figure, axes = plt.subplots(2, 4, figsize=(14, 7.2), facecolor="#0d1012")
    for axis, frame_index in zip(axes.reshape(-1), indices):
        axis.imshow(frames[frame_index])
        axis.set_title(
            f"第 {frame_index} 步 · 相位 {phase[frame_index]:.1f}°\n"
            f"角速度 {speed[frame_index]:.1f} rad/s · 保持 {hold[frame_index]} 步",
            color="#eef2f4",
            fontsize=10,
        )
        axis.axis("off")
    for axis in axes.reshape(-1)[len(indices) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def save_trace(
    time_s: np.ndarray,
    phase: np.ndarray,
    speed: np.ndarray,
    contacts: np.ndarray,
    penetration: np.ndarray,
    output: Path,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, facecolor="#0d1012")
    for axis in axes:
        axis.set_facecolor("#171b1e")
        axis.tick_params(colors="#a8b0b6")
        axis.grid(color="#30363d", alpha=0.75)
        for spine in axis.spines.values():
            spine.set_color("#30363d")
    axes[0].plot(time_s, phase, color="#49b6cc", linewidth=2)
    axes[0].axhspan(50.0, 60.0, color="#65c879", alpha=0.16)
    axes[0].set_ylabel("笔相位 (°)", color="#eef2f4")
    axes[1].plot(time_s, speed, color="#e7b84b", linewidth=2)
    axes[1].axhline(12.0, color="#e78062", linestyle="--", linewidth=1)
    axes[1].set_ylabel("角速度 (rad/s)", color="#eef2f4")
    for name, values, color in zip(FINGER_NAMES, contacts.T, FINGER_COLORS):
        axes[2].plot(time_s, values, label=name, color=color, linewidth=1.4)
    axes[2].set_ylabel("接触力 (N)", color="#eef2f4")
    axes[2].legend(ncol=5, frameon=False, labelcolor="#eef2f4")
    axes[3].plot(time_s, penetration * 1000.0, color="#e78062", linewidth=2)
    axes[3].axhline(0.5, color="#eef2f4", linestyle="--", linewidth=1)
    axes[3].set_ylabel("最大穿透 (mm)", color="#eef2f4")
    axes[3].set_xlabel("时间 (s)", color="#eef2f4")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 V9 P3 与 P3 到 P4 中文训练可视化。")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--formal-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, default=1415)
    parser.add_argument("--p3-reference-gif", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formal = json.loads(args.formal_report.read_text(encoding="utf-8"))

    with np.load(args.trajectory) as data:
        index = args.trajectory_index
        length = int(data["lengths"][index])
        control_dt = float(data["control_dt_s"])
        entry_label = int(data["entry_labels"][index])
        qpos = np.asarray(data["qpos"][index, :length], dtype=np.float64)
        phase = np.asarray(data["phase_deg"][index, :length], dtype=np.float64)
        speed = np.asarray(data["angular_speed_rad_s"][index, :length], dtype=np.float64)
        contacts = np.asarray(data["contact_force_n"][index, :length], dtype=np.float64)
        penetration = np.asarray(data["max_penetration_m"][index, :length], dtype=np.float64)
        stage_id = np.asarray(data["stage_id"][index, :length], dtype=np.int64)
        stage_names = [str(value) for value in data["stage_names"]]

    hold = terminal_hold_count(phase, speed, contacts, penetration)
    model_path = Path(__file__).resolve().parent / (
        "assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
    )
    frames = render_frames(qpos, model_path)
    imageio.mimsave(
        output_dir / "P3到制动P4代表成功轨迹.gif",
        frames,
        duration=control_dt,
        loop=0,
    )
    save_keyframes(
        frames,
        phase,
        speed,
        hold,
        output_dir / "P3到制动P4关键帧.png",
    )
    save_trace(
        np.arange(length) * control_dt,
        phase,
        speed,
        contacts,
        penetration,
        output_dir / "P3到制动P4物理曲线.png",
    )
    shutil.copy2(args.p3_reference_gif, output_dir / "P3主动保持参考_100ms.gif")

    entrance = formal["entrance_stats"]
    stage_stats = formal["stage_stats"]
    stage_rows = []
    for stage_name in stage_names:
        stats = stage_stats[stage_name]
        stage_rows.append(
            "<tr>"
            f"<td>{html.escape(stage_name)}</td>"
            f"<td>{stats['entries']}</td>"
            f"<td>{stats['successes']}</td>"
            f"<td>{stats['conditional_success_rate'] * 100.0:.2f}%</td>"
            "</tr>"
        )
    used_stages = [stage_names[index] for index in np.unique(stage_id)]
    tail = slice(max(0, length - 18), length)
    selected = {
        "轨迹编号": index,
        "入口": entry_label,
        "控制步数": length,
        "时长_s": length * control_dt,
        "终态相位_deg": float(phase[-1]),
        "终态角速度_rad_s": float(speed[-1]),
        "末段三指接触比": float(np.mean(contacts[tail, :3] > 0.05)),
        "最大穿透_mm": float(np.max(penetration) * 1000.0),
        "末段连续保持步": int(hold[-1]),
        "经历阶段": used_stages,
    }
    (output_dir / "代表轨迹摘要.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dex5 V9：P3 与 P3 到制动 P4 训练结果</title>
<style>
:root{{--bg:#0d1012;--surface:#171b1e;--line:#30363d;--text:#eef2f4;--muted:#a8b0b6;--cyan:#49b6cc;--green:#65c879;--yellow:#e7b84b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,"Noto Sans CJK SC",sans-serif}}
main{{width:min(1260px,calc(100vw - 32px));margin:auto;padding:24px 0 48px}}
header{{display:flex;justify-content:space-between;gap:20px;align-items:end;padding-bottom:18px;border-bottom:1px solid var(--line)}}
h1{{font-size:25px;margin:0}} h2{{font-size:18px;margin:0 0 14px}} p{{color:var(--muted);line-height:1.65}}
.pass{{color:var(--green);font-weight:750}} .metrics{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));border-bottom:1px solid var(--line)}}
.metric{{padding:17px 13px;border-right:1px solid var(--line)}} .metric:last-child{{border:0}}
.metric span{{display:block;color:var(--muted);font-size:12px}} .metric strong{{display:block;margin-top:5px;font-size:19px;font-variant-numeric:tabular-nums}}
section{{padding:22px 0;border-bottom:1px solid var(--line)}} .grid{{display:grid;grid-template-columns:1.15fr .85fr;gap:16px}}
figure{{margin:0;min-width:0}} img{{display:block;width:100%;border:1px solid var(--line);border-radius:6px;background:#050607}}
figcaption{{color:var(--muted);font-size:13px;line-height:1.55;margin-top:8px}}
table{{width:100%;border-collapse:collapse;font-size:14px}} th,td{{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--muted)}} td:last-child{{color:var(--green);font-variant-numeric:tabular-nums}}
.note{{padding:13px 15px;background:var(--surface);border-left:3px solid var(--yellow);color:var(--muted);line-height:1.65}}
@media(max-width:880px){{header{{align-items:start;flex-direction:column}}.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div><h1>Dex5 V9：P3 与 P3 → 制动 P4</h1><p>最新 PhysX 正式评估与代表成功轨迹 · 2026-07-17</p></div><div class="pass">阶段目标通过</div></header>
<div class="metrics">
<div class="metric"><span>P3 strict hold</span><strong>83.03%</strong></div>
<div class="metric"><span>连续分段专家整链</span><strong>86.11%</strong></div>
<div class="metric"><span>同策略 0° 入口</span><strong>{entrance['0']['success_rate']*100:.2f}%</strong></div>
<div class="metric"><span>同策略 45° 入口</span><strong>{entrance['45']['success_rate']*100:.2f}%</strong></div>
<div class="metric"><span>代表轨迹终态</span><strong>{phase[-1]:.2f}°</strong></div>
<div class="metric"><span>代表轨迹终态角速度</span><strong>{speed[-1]:.2f} rad/s</strong></div>
</div>
<section><h2>P3 短时保持</h2><div class="grid">
<figure><img src="P3主动保持参考_100ms.gif"><figcaption>最新版 canonical P3 的 100 ms 主动保持姿态参考。P3 PhysX strict-hold 正式成功率为 83.03%，平均最大穿透约 0.239 mm。</figcaption></figure>
<div><p>P3 的验收目标不是无限静态持笔，而是形成可进入下一段的短时接触状态：拇指、食指、中指保持接触，笔姿态和角速度不立即失控，同时限制无名指、小指误承力与碰撞穿透。</p>
<div class="note">左侧 P3 动画是 canonical 状态的真实网格姿态检查；83.03% 成功率来自独立 Isaac Lab / PhysX 正式评估，两者用途不同。</div></div>
</div></section>
<section><h2>P3 → 制动 P4 代表成功轨迹</h2><div class="grid">
<figure><img src="P3到制动P4代表成功轨迹.gif"><figcaption>正式 0° P3 入口成功样本 #{index}。PhysX 轨迹投影到同一 Dex5 真实网格渲染，运动状态未做手工插值。</figcaption></figure>
<figure><img src="P3到制动P4物理曲线.png"><figcaption>相位、角速度、五指接触力和最大穿透。绿色区域为 50–60° 制动捕获窗口，虚线分别为 12 rad/s 与 0.5 mm 门槛。</figcaption></figure>
</div></section>
<section><h2>关键帧</h2><figure><img src="P3到制动P4关键帧.png"><figcaption>从 P3 起步，经分段推滚、接触切换与减速，进入 thumb/index/middle 三指制动保持。</figcaption></figure></section>
<section><h2>正式分段结果</h2><table><thead><tr><th>阶段</th><th>进入数</th><th>成功数</th><th>条件成功率</th></tr></thead>
<tbody>{''.join(stage_rows)}</tbody></table></section>
<section><h2>可视化边界</h2><p class="note">训练、状态轨迹、接触力、穿透和成功判定均来自 PhysX。由于当前 RTX 5060 上 Isaac Sim 5.1 的 Hydra RTX 离屏渲染会在环境创建前崩溃，动画画面使用 MuJoCo 只做真实网格离屏渲染；它不参与策略执行或指标计算，也不表示 MuJoCo 与 PhysX 已动力学对齐。</p></section>
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
