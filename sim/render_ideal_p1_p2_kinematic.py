from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dex5_penspin.sim.dex5_penspin_env import Dex5PenSpinMinimalEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = (
    ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
    "P1_P2_reset_bank_phase1最终validated自复验_validated.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/P1到P2_理想运动学示意"
)
MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def representative(rows: np.ndarray) -> np.ndarray:
    center = np.median(rows, axis=0)
    weights = np.ones(rows.shape[1], dtype=np.float64)
    weights[20:23] = 8.0
    return rows[np.argmin(np.linalg.norm((rows - center) * weights[None, :], axis=1))].copy()


def make_camera(azimuth: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.075, 0.065)
    camera.distance = 0.235
    camera.azimuth = azimuth
    camera.elevation = -23.0
    return camera


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def render_view(
    env: Dex5PenSpinMinimalEnv, camera: mujoco.MjvCamera, qpos: np.ndarray
) -> np.ndarray:
    env.data.qpos[env.qpos_adr] = qpos[:20]
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 7] = qpos[20:27]
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = qpos[:20]
    mujoco.mj_forward(env.model, env.data)
    env.renderer.update_scene(env.data, camera=camera)
    return env.renderer.render().copy()


def compose_frame(left: np.ndarray, right: np.ndarray, title: str, progress: float) -> np.ndarray:
    view_w, view_h = 520, 390
    canvas = Image.new("RGB", (1080, 510), "#101417")
    draw = ImageDraw.Draw(canvas)
    left_image = Image.fromarray(left).resize((view_w, view_h), Image.Resampling.LANCZOS)
    right_image = Image.fromarray(right).resize((view_w, view_h), Image.Resampling.LANCZOS)
    canvas.paste(left_image, (16, 70))
    canvas.paste(right_image, (544, 70))
    draw.text((20, 16), title, fill="#f4f7f8", font=font(28))
    draw.text((24, 78), "主视角", fill="#e9eff2", font=font(18))
    draw.text((552, 78), "反向视角", fill="#e9eff2", font=font(18))
    draw.rectangle((20, 476, 1060, 484), fill="#303a40")
    draw.rectangle((20, 476, 20 + int(1040 * progress), 484), fill="#e0b849")
    draw.text(
        (20, 488),
        "运动学理想示意：笔姿态锁定，不代表物理可达性或训练验收",
        fill="#aeb9bf",
        font=font(14),
    )
    return np.asarray(canvas)


def build_trajectory(p1: np.ndarray, p2_ring: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    frames: list[np.ndarray] = []
    titles: list[str] = []

    def append(state: np.ndarray, title: str, count: int) -> None:
        for _ in range(count):
            frames.append(state.copy())
            titles.append(title)

    append(p1, "P1：拇指、食指、中指稳定控笔", 14)

    ring_start = p1[8:12].copy()
    # The waypoint opens the ring before curling it under the pen. This is a
    # visual motion-design cue, not a dynamically integrated hand state.
    clearance = ring_start + np.array([-0.16, -0.10, -0.12, -0.08])
    for i in range(18):
        t = smoothstep((i + 1) / 18.0)
        state = p1.copy()
        state[8:12] = (1.0 - t) * ring_start + t * clearance
        frames.append(state)
        titles.append("无名指先张开，与笔留出绕行间隙")

    for i in range(34):
        t = smoothstep((i + 1) / 34.0)
        state = p1.copy()
        state[8:12] = (1.0 - t) * clearance + t * p2_ring
        frames.append(state)
        titles.append("无名指从后侧缓慢绕到笔下")

    p2 = p1.copy()
    p2[8:12] = p2_ring
    append(p2, "P2：前三指仍控笔，无名指在笔下辅助托持", 28)
    return frames, titles


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an ideal, non-physical P1-to-P2 motion.")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.bank.resolve()) as data:
        source_qpos = np.asarray(data["source_qpos"], dtype=np.float64)
        phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)

    p1 = representative(source_qpos[phase_ids == 1])
    p2_ring = np.median(source_qpos[phase_ids == 2, 8:12], axis=0)
    states, titles = build_trajectory(p1, p2_ring)

    env = Dex5PenSpinMinimalEnv(model_path=MODEL, render_mode="rgb_array")
    cameras = (make_camera(142.0), make_camera(215.0))
    output_frames: list[np.ndarray] = []
    for index, (state, title) in enumerate(zip(states, titles, strict=True)):
        left = render_view(env, cameras[0], state)
        right = render_view(env, cameras[1], state)
        output_frames.append(compose_frame(left, right, title, index / (len(states) - 1)))
    env.close()

    gif_path = output_dir / "P1到P2_理想运动学双视角.gif"
    imageio.mimsave(gif_path, output_frames, duration=1.0 / 24.0, loop=0)

    key_indices = [0, 22, 47, len(output_frames) - 1]
    sheet = Image.new("RGB", (1080, 2040), "#101417")
    for row, frame_index in enumerate(key_indices):
        sheet.paste(Image.fromarray(output_frames[frame_index]), (0, row * 510))
    sheet_path = output_dir / "P1到P2_理想关键帧总览.png"
    sheet.save(sheet_path)

    summary = {
        "性质": "运动学理想示意，非物理仿真结果",
        "动作语义": [
            "拇指、食指、中指全程锁定并保持控笔",
            "笔的位置和姿态全程锁定",
            "无名指先张开，再从后侧绕到笔下",
            "P2 末态中无名指只提供辅助托持",
            "小拇指全程不参与",
        ],
        "来源": str(args.bank.resolve()),
        "P1手部关节": p1[:20].tolist(),
        "P2无名指关节": p2_ring.tolist(),
        "帧数": len(output_frames),
        "帧率": 24,
        "产物": {"动图": gif_path.name, "关键帧": sheet_path.name, "页面": "index.html"},
    }
    (output_dir / "说明.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>P1 到 P2 理想运动学示意</title><style>
body{{margin:0;background:#101417;color:#eef3f5;font:16px system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:28px}} h1{{font-size:30px;margin:0 0 10px}}
p,li{{color:#c7d0d5;line-height:1.65}} .notice{{border-left:4px solid #e0b849;padding:10px 14px;background:#181e22}}
img{{display:block;width:100%;background:#000;border:1px solid #354047;margin:18px 0}}
</style></head><body><main><h1>P1 → P2 理想动作</h1>
<p class="notice">这是不考虑接触力、惯性、执行器限制和穿透的运动学目标示意，不能用作 PhysX/MuJoCo 验收结果。</p>
<img src="{html.escape(gif_path.name)}" alt="P1到P2理想动图">
<p>动作定义：笔姿态不动，拇指、食指和中指一直保持 P1 控笔；无名指先留出间隙，再绕至笔下形成辅助托持；小拇指不承力。</p>
<h2>关键帧总览</h2><img src="{html.escape(sheet_path.name)}" alt="P1到P2关键帧">
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
