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


def render_frames(
    qpos: np.ndarray,
    model_path: Path,
    camera_mode: str = "dual",
) -> list[np.ndarray]:
    env = Dex5PenSpinMinimalEnv(
        model_path=model_path, render_mode="rgb_array"
    )
    cameras = []
    azimuths = (142.0,) if camera_mode == "single" else (142.0, -38.0)
    for azimuth in azimuths:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.135)
        camera.distance = 0.24
        camera.azimuth = azimuth
        camera.elevation = -22.0
        cameras.append(camera)
    frames: list[np.ndarray] = []
    for row in qpos:
        env.data.qpos[env.qpos_adr] = row[:20]
        env.data.qpos[
            env.pen_qpos_adr : env.pen_qpos_adr + 7
        ] = row[20:27]
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        views = []
        for camera in cameras:
            env.renderer.update_scene(env.data, camera=camera)
            views.append(env.renderer.render())
        frames.append(np.concatenate(views, axis=1))
    env.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="渲染拇指下托、绕侧、接管和无名指卸载局部轨迹。"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--segment",
        action="append",
        nargs=2,
        metavar=("中文名称", "轨迹文件"),
        required=True,
    )
    parser.add_argument(
        "--camera-mode",
        choices=("single", "dual"),
        default="dual",
    )
    parser.add_argument("--single-column", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(__file__).resolve().parent / (
        "assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
    )
    summaries: list[dict[str, object]] = []
    cards: list[str] = []
    key_rows: list[tuple[str, list[np.ndarray]]] = []

    for order, (name, source) in enumerate(args.segment, start=1):
        source_path = Path(source).resolve()
        with np.load(source_path) as data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
            phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)
            penetration = np.asarray(
                data["max_penetration_m"], dtype=np.float64
            )
            control_dt = float(data["control_dt_s"])
            trajectory_kind = str(
                data.get(
                    "trajectory_kind",
                    np.asarray("local_primitive"),
                )
            )
        indices = np.unique(
            np.linspace(
                0, len(qpos) - 1, min(len(qpos), 180)
            ).round().astype(int)
        )
        frames = render_frames(qpos[indices], model_path, args.camera_mode)
        filename = f"{order}_{name}.gif"
        imageio.mimsave(
            output_dir / filename,
            frames,
            duration=1.0 / 30.0,
            loop=0,
        )
        key_indices = np.linspace(
            0, len(qpos) - 1, 4
        ).round().astype(int)
        key_rows.append(
            (
                name,
                render_frames(
                    qpos[key_indices], model_path, args.camera_mode
                ),
            )
        )
        duration_ms = (len(qpos) - 1) * control_dt * 1000.0
        max_penetration_mm = float(penetration.max() * 1000.0)
        summaries.append(
            {
                "局部动作": name,
                "真实时长_ms": duration_ms,
                "起始阶段": int(phase_ids[0]),
                "结束阶段": int(phase_ids[-1]),
                "最大穿透_mm": max_penetration_mm,
                "轨迹性质": trajectory_kind,
                "轨迹文件": str(source_path),
            }
        )
        cards.append(
            f"""<section><h2>{html.escape(name)}</h2>
<img src="{html.escape(filename)}">
<table>
<tr><td>真实时长</td><td>{duration_ms:.0f} ms</td></tr>
<tr><td>阶段</td><td>{int(phase_ids[0])} → {int(phase_ids[-1])}</td></tr>
<tr><td>最大穿透</td><td>{max_penetration_mm:.3f} mm</td></tr>
<tr><td>轨迹性质</td><td>{html.escape(trajectory_kind)}</td></tr>
<tr><td>状态入口</td><td>该局部片段起点</td></tr>
</table></section>"""
        )

    figure, axes = plt.subplots(
        len(key_rows),
        4,
        figsize=(14, 3.2 * len(key_rows)),
        facecolor="#0d1012",
        squeeze=False,
    )
    for row, (name, frames) in enumerate(key_rows):
        for column, frame in enumerate(frames):
            axis = axes[row, column]
            axis.imshow(frame)
            axis.set_title(
                f"{name} · {column + 1}/4",
                color="#eef2f4",
                fontsize=10,
            )
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(
        output_dir / "拇指接管四段关键帧.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    (output_dir / "局部动作摘要.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Dex5 拇指接管局部动态视图</title>
<style>
body{{margin:0;background:#0d1012;color:#eef2f4;font-family:"Noto Sans CJK SC",sans-serif}}
main{{max-width:1180px;margin:auto;padding:30px}} h1{{font-size:30px;margin:0 0 8px}}
p{{color:#aeb7bd;line-height:1.7}} .grid{{display:grid;grid-template-columns:{'1fr' if args.single_column else '1fr 1fr'};gap:18px}}
section,figure{{margin:0;background:#15191c;border:1px solid #343b40;padding:16px;border-radius:6px}}
h2{{font-size:19px;margin:0 0 12px}} img{{display:block;width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;margin-top:10px}} td{{padding:8px;border-bottom:1px solid #30363d}}
td:last-child{{text-align:right;font-weight:700}} .warn{{color:#e7b84b}}
figure{{margin-top:18px}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Dex5：拇指下托到拇食指接管</h1>
<p>轨迹来自 PhysX 动力学回合，画面投影到 Dex5 网格。
<span class="warn">“deepest_incomplete_episode” 表示已进入最深阶段，
但尚未满足 100 ms 正式保持标准。</span></p>
<div class="grid">{''.join(cards)}</div>
<figure><img src="拇指接管四段关键帧.png"></figure>
</main></body></html>"""
    page_path = output_dir / "index.html"
    page_path.write_text(page, encoding="utf-8")
    print(page_path)


if __name__ == "__main__":
    main()
