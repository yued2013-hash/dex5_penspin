from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from dex5_penspin.sim.dex5_penspin_env import Dex5PenSpinMinimalEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
    "P1_P2_reset_bank_phase1最终validated自复验_validated.npz"
)
DEFAULT_REPORT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
    "P1_P2_reset_bank_phase1最终validated自复验.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
    "P1_P2_ResetBank_可视化"
)
MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2.xml"


PHASE_TEXT = {
    1: {
        "title": "P1：拇指 / 食指 / 中指控笔",
        "desc": "三指主控笔，无名指和小拇指不作为主承力。",
        "contacts": "thumb / index / middle",
    },
    2: {
        "title": "P2：三指控笔 + ring 下移",
        "desc": "thumb/index/middle 继续控笔；ring 位姿转移到笔下，但不取代前三指主承力。",
        "contacts": "thumb/index/middle；ring pose-level",
    },
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def make_camera(azimuth: float, distance: float = 0.235) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.075, 0.065)
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = -23.0
    return camera


def nearest_to_mean(rows: np.ndarray) -> int:
    center = rows.mean(axis=0)
    weights = np.ones(rows.shape[1], dtype=np.float64)
    weights[20:23] = 8.0
    score = np.linalg.norm((rows - center) * weights[None, :], axis=1)
    return int(np.argmin(score))


def set_row(env: Dex5PenSpinMinimalEnv, row: np.ndarray, target: np.ndarray | None = None) -> None:
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env.qpos_adr] = row[:20]
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 7] = row[20:27]
    env.data.qvel[:] = 0.0
    if target is None:
        target = row[:20]
    env.target_qpos = target.astype(np.float32)
    env.data.ctrl[:] = env.target_qpos
    mujoco.mj_forward(env.model, env.data)


def render_view(env: Dex5PenSpinMinimalEnv, camera: mujoco.MjvCamera) -> np.ndarray:
    env.renderer.update_scene(env.data, camera=camera)
    return env.renderer.render()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render final P1/P2 PhysX reset bank examples.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bundle_path = resolve(args.bundle)
    report_path = resolve(args.report)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(bundle_path) as data:
        source_qpos = np.asarray(data["source_qpos"], dtype=np.float64)
        source_target = np.asarray(data["source_target_qpos"], dtype=np.float64)
        phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    env = Dex5PenSpinMinimalEnv(model_path=MODEL, render_mode="rgb_array")
    cameras = {
        "主视角": make_camera(142.0),
        "反向视角": make_camera(215.0),
        "侧视角": make_camera(178.0, distance=0.205),
    }

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    selected: dict[int, dict[str, object]] = {}
    frames_by_phase: dict[int, dict[str, np.ndarray]] = {}

    for phase in (1, 2):
        phase_indices = np.flatnonzero(phase_ids == phase)
        phase_rows = source_qpos[phase_indices]
        local_index = nearest_to_mean(phase_rows)
        row_index = int(phase_indices[local_index])
        selected[phase] = {
            "row_index": row_index,
            "phase_local_index": int(local_index),
            "phase_rows": int(phase_indices.size),
        }
        set_row(env, source_qpos[row_index], source_target[row_index])
        frames_by_phase[phase] = {}
        for view_name, camera in cameras.items():
            frame = render_view(env, camera)
            filename = f"P{phase}_{view_name}.png"
            imageio.imwrite(output_dir / filename, frame)
            frames_by_phase[phase][view_name] = frame

    env.close()

    figure, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor="#101316")
    for row_id, phase in enumerate((1, 2)):
        for col_id, view_name in enumerate(cameras):
            axis = axes[row_id, col_id]
            axis.imshow(frames_by_phase[phase][view_name])
            axis.set_title(
                f"{PHASE_TEXT[phase]['title']} | {view_name}\n"
                f"接触语义：{PHASE_TEXT[phase]['contacts']}",
                color="#eef2f4",
                fontsize=9,
            )
            axis.axis("off")
    figure.tight_layout()
    montage_path = output_dir / "P1_P2_reset_bank代表状态_montage.png"
    figure.savefig(montage_path, dpi=150, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    cycle_frames: list[np.ndarray] = []
    for _ in range(3):
        cycle_frames.extend([frames_by_phase[1]["主视角"]] * 18)
        cycle_frames.extend([frames_by_phase[2]["主视角"]] * 18)
    cycle_path = output_dir / "P1_P2_reset_bank代表状态切换.gif"
    imageio.mimsave(cycle_path, cycle_frames, duration=1.0 / 24.0, loop=0)

    summary = {
        "bundle": str(bundle_path),
        "report": str(report_path),
        "output_dir": str(output_dir),
        "selected_rows": selected,
        "validation": {
            "total_valid_rows": report["total_valid_rows"],
            "total_hold_rate": report["total_hold_rate"],
            "total_drop_rate": report["total_drop_rate"],
            "total_invalid_rows": report["total_invalid_rows"],
            "max_initial_penetration_mm": report["max_initial_penetration_mm"],
            "phase_report": report["phase_report"],
        },
        "artifacts": {
            "index": "index.html",
            "montage": montage_path.name,
            "cycle": cycle_path.name,
        },
    }
    (output_dir / "可视化摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    phase_rows = report["phase_report"]
    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>P1/P2 Reset Bank 可视化</title>
<style>
body{{margin:0;background:#101316;color:#eef2f4;font:16px system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px}}
h1{{font-size:28px;margin:0 0 8px}} h2{{margin-top:28px}}
p{{color:#c8d0d5;line-height:1.55}} code{{color:#dce6ea}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
figure{{margin:0}} img{{width:100%;display:block;background:#000;border:1px solid #3a444a}}
figcaption{{color:#c8d0d5;margin-top:8px;font-size:14px}}
table{{width:100%;border-collapse:collapse;margin-top:12px}} th,td{{border:1px solid #3a444a;padding:8px;text-align:left}} th{{background:#1a2025}}
.wide{{width:100%;display:block}}
@media(max-width:820px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>P1/P2 Reset Bank 可视化</h1>
<p>来源：最终 PhysX validated bank。图像用 MuJoCo realmesh 渲染，用于检查几何关系和动作语义；验收指标仍以 PhysX 全量 100 ms 复验报告为准。</p>
<table>
<thead><tr><th>相位</th><th>有效 reset</th><th>100 ms 保持率</th><th>最大初始穿透</th><th>掉笔率</th><th>非法数值</th></tr></thead>
<tbody>
<tr><td>P1</td><td>{phase_rows['1']['valid_rows']}</td><td>{phase_rows['1']['hold_rate'] * 100:.2f}%</td><td>{phase_rows['1']['max_initial_penetration_mm']:.3f} mm</td><td>{phase_rows['1']['drop_rate'] * 100:.2f}%</td><td>{phase_rows['1']['invalid_rows']}</td></tr>
<tr><td>P2</td><td>{phase_rows['2']['valid_rows']}</td><td>{phase_rows['2']['hold_rate'] * 100:.2f}%</td><td>{phase_rows['2']['max_initial_penetration_mm']:.3f} mm</td><td>{phase_rows['2']['drop_rate'] * 100:.2f}%</td><td>{phase_rows['2']['invalid_rows']}</td></tr>
</tbody></table>
<section><h2>总览</h2><img class="wide" src="{html.escape(montage_path.name)}"></section>
<section><h2>P1/P2 切换</h2><img class="wide" src="{html.escape(cycle_path.name)}"></section>
<section><h2>P1 代表状态</h2><p>{PHASE_TEXT[1]['desc']}</p><div class="grid">
<figure><img src="P1_主视角.png"><figcaption>主视角</figcaption></figure>
<figure><img src="P1_反向视角.png"><figcaption>反向视角</figcaption></figure>
<figure><img src="P1_侧视角.png"><figcaption>侧视角</figcaption></figure>
</div></section>
<section><h2>P2 代表状态</h2><p>{PHASE_TEXT[2]['desc']}</p><div class="grid">
<figure><img src="P2_主视角.png"><figcaption>主视角</figcaption></figure>
<figure><img src="P2_反向视角.png"><figcaption>反向视角</figcaption></figure>
<figure><img src="P2_侧视角.png"><figcaption>侧视角</figcaption></figure>
</div></section>
</main></body></html>
"""
    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
