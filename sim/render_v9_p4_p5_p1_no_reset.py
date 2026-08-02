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


def phase_name(phase_id: int) -> str:
    if phase_id == 0:
        return "P4 中指退让"
    if phase_id == 1:
        return "P5 拇食指捕获"
    if phase_id == 2:
        return "P1 接近"
    if phase_id == 3:
        return "P1 三指接触"
    names = {
        4: "食指/中指夹持获取",
        5: "食指/中指夹持稳定",
        6: "拇指下托接近",
        7: "拇指下托接触",
        8: "拇指下托稳定",
        9: "拇指绕侧 1",
        10: "拇指绕侧 2",
        11: "拇指绕侧 3",
        12: "拇指绕至另一侧",
        13: "拇指/食指接管",
        14: "拇指/食指夹持稳定",
        15: "拇食指接管后中指开始卸载",
        16: "中指回撤",
        17: "P1 夹持保持",
    }
    if phase_id in names:
        return names[phase_id]
    return f"阶段 {phase_id}"


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
        description="渲染 P4 到 P5 到 P1 同 episode 无重置轨迹。"
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
        trajectory_kind = (
            str(data["trajectory_kind"])
            if "trajectory_kind" in data
            else "successful_episode"
        )
        deepest_phase = (
            int(data["deepest_phase"])
            if "deepest_phase" in data
            else int(phase_ids.max())
        )
        state_reset = (
            str(data["state_reset"])
            if "state_reset" in data
            else "unknown"
        )

    valid = np.linalg.norm(qpos[:, 20:23], axis=1) > 1.0e-6
    qpos = qpos[valid]
    qvel = qvel[valid]
    phase_ids = phase_ids[valid]
    penetration = penetration[valid]
    reset_points = np.flatnonzero(phase_ids[1:] < phase_ids[:-1]) + 1
    segments = np.split(
        np.arange(len(phase_ids)),
        reset_points,
    )
    selected = max(
        (segment for segment in segments if len(segment)),
        key=lambda segment: (
            int(phase_ids[segment].max()),
            len(segment),
        ),
    )
    qpos = qpos[selected]
    qvel = qvel[selected]
    phase_ids = phase_ids[selected]
    penetration = penetration[selected]
    deepest_phase = int(phase_ids.max())

    model_path = Path(__file__).resolve().parent / (
        "assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
    )
    render_indices = np.unique(
        np.r_[
            np.linspace(
                0, len(qpos) - 1, min(len(qpos), 360)
            ).round().astype(int),
            np.flatnonzero(
                np.r_[True, phase_ids[1:] != phase_ids[:-1]]
            ),
        ]
    )
    frames = render_frames(qpos[render_indices], model_path)
    gif_path = output_dir / "P4到P5到P1无重置代表轨迹.gif"
    imageio.mimsave(gif_path, frames, duration=1.0 / 30.0, loop=0)

    transitions = np.flatnonzero(np.r_[True, phase_ids[1:] != phase_ids[:-1]])
    anchors = np.unique(
        np.r_[
            0,
            transitions,
            np.linspace(0, len(frames) - 1, 12).round().astype(int),
            len(frames) - 1,
        ]
    )
    indices = np.unique(
        np.linspace(0, len(anchors) - 1, 12).round().astype(int)
    )
    indices = anchors[indices]
    key_frames = render_frames(qpos[indices], model_path)
    figure, axes = plt.subplots(
        3, 4, figsize=(14, 10.2), facecolor="#0d1012"
    )
    flat_axes = axes.reshape(-1)
    for axis, index, frame in zip(flat_axes, indices, key_frames):
        axis.imshow(frame)
        axis.set_title(
            f"{index * control_dt * 1000:.0f} ms · "
            f"{phase_name(int(phase_ids[index]))}",
            color="#eef2f4",
            fontsize=9,
        )
        axis.axis("off")
    for axis in flat_axes[len(indices) :]:
        axis.axis("off")
    figure.tight_layout()
    keyframe_path = output_dir / "P4到P5到P1无重置关键帧.png"
    figure.savefig(
        keyframe_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    time_ms = np.arange(len(qpos)) * control_dt * 1000.0
    angular_speed = np.linalg.norm(qvel[:, 23:26], axis=1)
    joint_step = np.r_[
        0.0, np.max(np.abs(np.diff(qpos[:, :20], axis=0)), axis=1)
    ]
    pen_step_mm = np.r_[
        0.0,
        np.linalg.norm(np.diff(qpos[:, 20:23], axis=0), axis=1)
        * 1000.0,
    ]
    figure, axes = plt.subplots(
        3, 1, figsize=(11, 8.2), sharex=True, facecolor="#0d1012"
    )
    for axis in axes:
        axis.set_facecolor("#171b1e")
        axis.tick_params(colors="#a8b0b6")
        axis.grid(color="#30363d", alpha=0.75)
        for spine in axis.spines.values():
            spine.set_color("#30363d")
    axes[0].plot(time_ms, angular_speed, color="#e7b84b", linewidth=1.8)
    axes[0].set_ylabel("笔角速度 (rad/s)", color="#eef2f4")
    axes[1].plot(
        time_ms, penetration * 1000.0, color="#e78062", linewidth=1.8
    )
    axes[1].axhline(0.5, color="#eef2f4", linestyle="--")
    axes[1].set_ylabel("累计最大穿透 (mm)", color="#eef2f4")
    axes[2].plot(
        time_ms, joint_step, label="最大关节步进 (rad)", linewidth=1.6
    )
    axes[2].plot(
        time_ms,
        pen_step_mm,
        label="笔质心步进 (mm)",
        linewidth=1.6,
    )
    axes[2].set_ylabel("相邻控制步变化", color="#eef2f4")
    axes[2].set_xlabel("时间 (ms)", color="#eef2f4")
    axes[2].legend()
    for transition in transitions[1:]:
        for axis in axes:
            axis.axvline(
                transition * control_dt * 1000.0,
                color="#65c879",
                alpha=0.35,
                linewidth=0.8,
            )
    figure.tight_layout()
    curve_path = output_dir / "P4到P5到P1无重置物理曲线.png"
    figure.savefig(
        curve_path,
        dpi=160,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    duration_ms = len(qpos) * control_dt * 1000.0
    max_penetration_mm = float(penetration.max() * 1000.0)
    evaluation_samples = report.get(
        "episodes",
        report.get("completed_episodes", report.get("num_envs", 0)),
    )
    summary = {
        "轨迹帧数": len(qpos),
        "真实时长_ms": duration_ms,
        "控制周期_ms": control_dt * 1000.0,
        "可视化播放": "30 FPS 慢放",
        "代表轨迹最大穿透_mm": max_penetration_mm,
        "最大相邻关节步进_rad": float(joint_step.max()),
        "最大相邻笔质心步进_mm": float(pen_step_mm.max()),
        "评估成功率": report["success_rate"],
        "评估样本": evaluation_samples,
        "评估成功数": report["successes"],
        "状态重置": state_reset,
        "轨迹性质": trajectory_kind,
        "最深阶段": phase_name(deepest_phase),
    }
    (output_dir / "动态轨迹摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    is_success = trajectory_kind == "successful_episode"
    status_class = "pass" if is_success else "warn"
    status_text = (
        "已进入严格 P1，并通过保持判据"
        if is_success
        else f"未完成；最深到达 {phase_name(deepest_phase)}"
    )
    local_primitive = state_reset != "none"
    trajectory_title = (
        "拇指下托局部动态视图"
        if local_primitive
        else "P4 → P5 → P1 同 episode 笔动态视图"
    )
    reset_text = (
        "仅轨迹起点使用 phase7 真实缓存"
        if local_primitive
        else "无"
    )
    evaluation_label = (
        "局部复测成功率" if local_primitive else "独立评估成功率"
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Dex5 P4 到 P5 到 P1 无重置动态视图</title>
<style>
body{{margin:0;background:#0d1012;color:#eef2f4;font-family:"Noto Sans CJK SC",sans-serif}}
main{{max-width:1180px;margin:auto;padding:30px}} h1{{font-size:30px;margin:0 0 8px}}
p{{color:#aeb7bd}} .grid{{display:grid;grid-template-columns:1.25fr .75fr;gap:18px}}
figure,section{{margin:0 0 18px;background:#15191c;border:1px solid #343b40;padding:16px;border-radius:6px}}
img{{display:block;width:100%;height:auto}} table{{width:100%;border-collapse:collapse}}
td{{padding:11px;border-bottom:1px solid #30363d}} td:last-child{{text-align:right;font-weight:700}}
.pass{{color:#65c879}} .warn{{color:#e7b84b}} .note{{font-size:14px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Dex5：{trajectory_title}</h1>
<p>真实 PhysX 状态轨迹投影到 Dex5 网格；局部轨迹不会冒充完整链路结果。</p>
<div class="grid"><figure><img src="{html.escape(gif_path.name)}"><figcaption>按 30 FPS 慢放；真实控制周期为 {control_dt * 1000:.2f} ms。</figcaption></figure>
<section><table>
<tr><td>{evaluation_label}</td><td class="warn">{report["success_rate"] * 100:.2f}%</td></tr>
<tr><td>成功 / 评估环境</td><td>{report["successes"]}/{evaluation_samples}</td></tr>
<tr><td>代表轨迹时长</td><td>{duration_ms:.0f} ms</td></tr>
<tr><td>轨迹状态</td><td class="{status_class}">{status_text}</td></tr>
<tr><td>代表轨迹最大穿透</td><td>{max_penetration_mm:.3f} mm</td></tr>
<tr><td>状态 reset</td><td class="warn">{reset_text}</td></tr>
</table><p class="note">动画展示当前策略真实产生的笔运动。未通过最终 P1 判定的轨迹仅用于定位停止点，不能视为完整链路成功。</p></section></div>
<figure><img src="{html.escape(keyframe_path.name)}"></figure>
<figure><img src="{html.escape(curve_path.name)}"></figure>
</main></body></html>"""
    page_path = output_dir / "index.html"
    page_path.write_text(page, encoding="utf-8")
    print(page_path)


if __name__ == "__main__":
    main()
