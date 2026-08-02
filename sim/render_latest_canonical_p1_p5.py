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


PHASE_NAMES_ZH = {
    1: "拇指/食指/中指控笔",
    2: "无名指移至笔下",
    3: "食指绕前，三指局部控制",
    4: "食指/中指接管并绕拇指",
    5: "拇指/食指捏持回收",
}
PHASE_SUMMARIES_ZH = {
    1: "拇指、食指和中指控制笔，无名指与小拇指不承担主要支撑。",
    2: "保持 P1 的拇指/食指/中指接触和笔姿，无名指移至笔下辅助支撑。",
    3: "食指完成前绕，形成拇指/食指/中指局部控制入口；无名指与小拇指不作为主承力接触。",
    4: "中指接管、无名指卸载，食指和中指夹持并驱动笔绕过拇指，拇指继续作轻支撑与转轴。",
    5: "拇指和食指捏住笔，中指与无名指向 P1 后侧回收。",
}
FINGER_ZH = {"thumb": "拇指", "index": "食指", "middle": "中指", "ring": "无名指", "pinky": "小拇指"}


def contacts_zh(contacts: list[str]) -> str:
    return " / ".join(FINGER_ZH.get(finger, finger) for finger in contacts)


def make_camera(azimuth: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.075, 0.065)
    camera.distance = 0.235
    camera.azimuth = azimuth
    camera.elevation = -23.0
    return camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Render current canonical P1-P5 with real Dex5 meshes.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    validation = None
    if args.validation:
        validation = json.loads(Path(args.validation).read_text(encoding="utf-8"))

    model_path = (
        Path(__file__).resolve().parent
        / "assets/dex5_urdf_import/dex5_penspin_real_m2.xml"
    )
    env = Dex5PenSpinMinimalEnv(model_path=model_path, render_mode="rgb_array")
    cameras = (make_camera(142.0), make_camera(215.0))
    rendered: list[dict[str, object]] = []

    for phase in manifest["phases"]:
        hand_qpos = np.asarray(phase["hand_qpos"], dtype=np.float64)
        pen_qpos = np.concatenate(
            (
                np.asarray(phase["pen_pos"], dtype=np.float64),
                np.asarray(phase["pen_quat"], dtype=np.float64),
            )
        )
        env.data.qpos[env.qpos_adr] = hand_qpos
        env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 7] = pen_qpos
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        views = []
        for camera in cameras:
            env.renderer.update_scene(env.data, camera=camera)
            views.append(env.renderer.render())
        phase_id = int(phase["phase"])
        imageio.imwrite(output_dir / f"P{phase_id}_realmesh.png", views[0])
        imageio.imwrite(output_dir / f"P{phase_id}_realmesh_side.png", views[1])
        rendered.append({"phase": phase, "views": views})

    active_rollouts: dict[int, str] = {}
    for phase_id in (3, 4):
        phase = next(row for row in manifest["phases"] if int(row["phase"]) == phase_id)
        mujoco.mj_resetData(env.model, env.data)
        hand_qpos = np.asarray(phase["hand_qpos"], dtype=np.float64)
        env.data.qpos[env.qpos_adr] = hand_qpos
        env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = phase["pen_pos"]
        env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = phase["pen_quat"]
        env.data.qvel[:] = 0.0
        env._set_position_actuator_gains(
            actuator_kp=float(phase.get("actuator_kp", 2.0)),
            actuator_kv=float(phase.get("actuator_kv", 0.2)),
        )
        target = np.asarray(phase.get("control_target_qpos", hand_qpos), dtype=np.float64)
        env.target_qpos = target.astype(np.float32)
        env.data.ctrl[:] = target
        mujoco.mj_forward(env.model, env.data)
        frames = []
        for _ in range(11):
            env.renderer.update_scene(env.data, camera=cameras[0])
            frames.extend([env.renderer.render()] * 3)
            env.step(np.zeros(20, dtype=np.float32))
        filename = f"P{phase_id}_active_100ms.gif"
        imageio.mimsave(output_dir / filename, frames, duration=1.0 / 30.0, loop=0)
        active_rollouts[phase_id] = filename
    env.close()

    figure, axes = plt.subplots(5, 2, figsize=(11, 17), facecolor="#0d1012")
    for row, item in enumerate(rendered):
        phase = item["phase"]
        phase_id = int(phase["phase"])
        contacts = contacts_zh(phase["contact_set"])
        for column, frame in enumerate(item["views"]):
            axis = axes[row, column]
            axis.imshow(frame)
            view_name = "主视角" if column == 0 else "反向视角"
            axis.set_title(
                f"P{phase_id} {PHASE_NAMES_ZH[phase_id]} | {view_name}\n预期接触：{contacts}",
                color="#eef2f4",
                fontsize=10,
            )
            axis.axis("off")
    figure.tight_layout()
    montage_path = output_dir / "canonical_p1_p5_realmesh_montage.png"
    figure.savefig(montage_path, dpi=150, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    cycle_frames: list[np.ndarray] = []
    for item in rendered:
        cycle_frames.extend([item["views"][0]] * 24)
    cycle_path = output_dir / "canonical_p1_p5_realmesh_cycle.gif"
    imageio.mimsave(cycle_path, cycle_frames, duration=1.0 / 24.0, loop=0)

    p2_p4_frames: list[np.ndarray] = []
    for phase_id in (2, 3, 4):
        item = next(row for row in rendered if int(row["phase"]["phase"]) == phase_id)
        p2_p4_frames.extend([item["views"][0]] * 30)
    p2_p4_path = output_dir / "P2到P4接触接力候选.gif"
    imageio.mimsave(p2_p4_path, p2_p4_frames, duration=1.0 / 30.0, loop=0)

    summary = {
        "manifest": str(manifest_path),
        "model": str(model_path),
        "status": (
            "P1-P5 关键状态与已选局部过渡全部通过"
            if validation and validation.get("all_accepted")
            else "已生成关键姿态可视化"
        ),
        "phases": [
            {
                "phase": int(item["phase"]["phase"]),
                "name": item["phase"]["name"],
                "name_zh": PHASE_NAMES_ZH[int(item["phase"]["phase"])],
                "expected_contacts": item["phase"]["contact_set"],
                "expected_contacts_zh": contacts_zh(item["phase"]["contact_set"]),
                "placement_support": item["phase"]["placement_support"],
                "source_validation_status": item["phase"]["source_validation_status"],
                "image": f"P{item['phase']['phase']}_realmesh.png",
                "side_image": f"P{item['phase']['phase']}_realmesh_side.png",
            }
            for item in rendered
        ],
        "artifacts": {
            "cycle": cycle_path.name,
            "p2_to_p4_contact_relay": p2_p4_path.name,
            "montage": montage_path.name,
        },
    }
    (output_dir / "canonical_p1_p5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "可视化摘要.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    phase_metrics = {int(row["phase"]): row for row in (validation or {}).get("phases", [])}
    sections = []
    for item in rendered:
        phase = item["phase"]
        phase_id = int(phase["phase"])
        metrics = phase_metrics.get(phase_id, {})
        rollout_html = (
            f'<h3>100 ms 主动控制回放</h3><img class="wide" src="{active_rollouts[phase_id]}">'
            if phase_id in active_rollouts
            else ""
        )
        sections.append(
            f"""
            <section>
              <h2>P{phase_id} {PHASE_NAMES_ZH[phase_id]}</h2>
              <p>{PHASE_SUMMARIES_ZH[phase_id]}</p>
              <p class="metric">接触：{contacts_zh(phase['contact_set'])} |
              代理几何最大穿透：{metrics.get('max_penetration_mm', float('nan')):.3f} mm |
              真实网格最大穿透：{metrics.get('visual_mesh_penetration_mm', float('nan')):.3f} mm |
              100 ms 笔位移：{metrics.get('pen_drift_mm', float('nan')):.3f} mm</p>
              <div class="views"><img src="P{phase_id}_realmesh.png"><img src="P{phase_id}_realmesh_side.png"></div>
              {rollout_html}
            </section>
            """
        )
    transition_block = ""
    if (output_dir / "canonical_p2_p3_p5_v4_transition_montage.png").exists():
        transition_block = """
        <section><h2>P2 / P3 / P5 局部过渡</h2>
          <p>P2 在主抓持保持时将无名指移至笔下；P3 先让食指向外清障，再对齐至类 hold-oracle 食指/中指主动夹持；P5 由拇指/食指捏持，中指/无名指向 P1 后侧回收。</p>
          <img class="wide" src="canonical_p2_p3_p5_v4_transition_montage.png">
          <div class="views"><img src="p1_to_p2_v3.gif"><img src="p2_to_p3_v4.gif"><img src="p5_return_v3.gif"></div>
        </section>
        """
    p2_p4_block = f"""
        <section><h2>P2→P3→P4 接触接力修复</h2>
          <p>该动图依次展示三个经过碰撞投影的关键状态：P2 在拇指/食指/中指保持时将无名指移至笔下；P3 食指完成前绕，形成拇指/食指/中指局部控制入口；P4 由食指/中指驱动笔绕拇指，拇指转到笔端下方承托。它用于确认接触语义和相对笔姿，不代表连续动力学过渡已经通过。</p>
          <img class="wide" src="{p2_p4_path.name}">
        </section>
        """
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Dex5 P1-P5 关键状态可视化</title>
    <style>body{{margin:0;background:#0d1012;color:#eef2f4;font:16px sans-serif}}main{{max-width:1180px;margin:auto;padding:28px}}h1{{font-size:28px}}h2{{margin-top:28px}}p{{color:#c8d0d5}}.metric{{font-family:monospace}}section{{border-top:1px solid #354047;padding-top:12px}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img{{display:block;width:100%;background:#000}}.wide{{width:100%}}@media(max-width:760px){{.views{{grid-template-columns:1fr}}}}</style></head>
    <body><main><h1>Dex5 + 实测笔 P1-P5 关键状态</h1>
    <p><strong>验收状态：</strong>{html.escape(summary['status'])}。本报告验证短时关键状态与已选局部过渡，不代表已手工打通完整转笔闭环。</p>
    <img class="wide" src="canonical_p1_p5_realmesh_montage.png">{p2_p4_block}{''.join(sections)}{transition_block}</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
