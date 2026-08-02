from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import contact_snapshot


TIP_GEOMS = {
    "thumb": "Link_14R_tip_proxy",
    "index": "Link_24R_tip_proxy",
    "middle": "Link_34R_tip_proxy",
    "ring": "Link_44R_tip_proxy",
    "pinky": "Link_54R_tip_proxy",
}


PHASE_SPECS = [
    {
        "phase": 1,
        "name": "four_finger_clamp",
        "summary": "四个手指夹笔，作为半圈动作的稳定起点。",
        "desired_contacts": ["thumb", "index", "middle", "ring"],
        "placement_support": ["thumb", "index", "middle", "ring"],
        "phase_deg": 0.0,
        "axis_z": 0.06,
        "center_offset": [0.004, -0.002, 0.010],
    },
    {
        "phase": 2,
        "name": "ring_wrap_to_front",
        "summary": "无名指从笔下方绕到笔前方夹住笔。",
        "desired_contacts": ["thumb", "middle", "ring"],
        "placement_support": ["thumb", "middle", "ring"],
        "phase_deg": 32.0,
        "axis_z": 0.04,
        "center_offset": [0.002, 0.000, 0.009],
    },
    {
        "phase": 3,
        "name": "index_wrap_to_front",
        "summary": "食指从侧边绕到笔前方夹住笔。",
        "desired_contacts": ["thumb", "index", "ring"],
        "placement_support": ["thumb", "index", "ring"],
        "phase_deg": 72.0,
        "axis_z": 0.02,
        "center_offset": [0.000, 0.002, 0.008],
    },
    {
        "phase": 4,
        "name": "index_middle_to_thumb_regrasp",
        "summary": "食指中指夹住笔旋转到大拇指上，并绕过大拇指形成食指/中指/大拇指夹持。",
        "desired_contacts": ["thumb", "index", "middle"],
        "placement_support": ["thumb", "index", "middle"],
        "phase_deg": 125.0,
        "axis_z": 0.02,
        "center_offset": [-0.002, 0.001, 0.009],
    },
    {
        "phase": 5,
        "name": "thumb_index_around_middle_ring",
        "summary": "大拇指和食指夹住笔，使笔绕过中指和无名指。",
        "desired_contacts": ["thumb", "index"],
        "placement_support": ["thumb", "index", "middle"],
        "phase_deg": 180.0,
        "axis_z": -0.04,
        "center_offset": [-0.004, -0.002, 0.010],
    },
]


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def quat_from_x_axis(axis: np.ndarray) -> np.ndarray:
    axis = normalize(axis)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    dot = float(np.clip(np.dot(x_axis, axis), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rot_axis = normalize(np.cross(x_axis, axis))
    angle = float(np.arccos(dot))
    return np.array([np.cos(0.5 * angle), *(np.sin(0.5 * angle) * rot_axis)], dtype=np.float64)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize(q0)
    q1 = normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize(q0 + t * (q1 - q0))
    theta0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    theta = theta0 * t
    return normalize(
        (np.sin(theta0 - theta) / np.sin(theta0)) * q0
        + (np.sin(theta) / np.sin(theta0)) * q1
    )


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def latest_pose_path(source_dir: Path, pose_idx: int) -> Path:
    matches = sorted(source_dir.glob(f"P{pose_idx}_*_latest.json"))
    if not matches:
        raise FileNotFoundError(f"missing latest hand pose for P{pose_idx} in {source_dir}")
    return matches[-1]


def load_hand_records(source_dir: Path) -> list[dict[str, Any]]:
    records = []
    for idx in range(1, 6):
        path = latest_pose_path(source_dir, idx)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        records.append(data)
    return records


def make_env(model_path: str) -> Dex5PenSpinMinimalEnv:
    return Dex5PenSpinMinimalEnv(
        model_path=model_path,
        pen_init_pos=(0.0, 0.0, 0.1),
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=3.0,
        actuator_kp=400.0,
        actuator_kv=20.0,
        thumb_force_limit=0.0,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        render_mode="rgb_array",
        seed=0,
    )


def set_state(env: Dex5PenSpinMinimalEnv, hand_qpos: np.ndarray, pen_pos: np.ndarray, pen_quat: np.ndarray) -> None:
    qpos = np.clip(np.asarray(hand_qpos, dtype=np.float64), env.joint_low, env.joint_high)
    env.target_qpos = qpos.astype(np.float32)
    env.data.qpos[env.qpos_adr] = qpos
    env.data.ctrl[:] = qpos
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = np.asarray(pen_pos, dtype=np.float64)
    env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = normalize(pen_quat)
    env.data.qvel[:] = 0.0
    env.prev_phase = env._quat_twist_phase_about_x(env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7])
    env.rotation_phase = 0.0
    mujoco.mj_forward(env.model, env.data)


def tip_positions(env: Dex5PenSpinMinimalEnv) -> dict[str, np.ndarray]:
    out = {}
    for finger, geom_name in TIP_GEOMS.items():
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            out[finger] = env.data.geom_xpos[geom_id].copy()
    return out


def build_phase_seed(
    env: Dex5PenSpinMinimalEnv,
    hand_record: dict[str, Any],
    spec: dict[str, Any],
    *,
    base_theta: float,
) -> dict[str, Any]:
    hand_qpos = np.asarray(hand_record["hand_qpos"], dtype=np.float64)
    set_state(env, hand_qpos, np.array([0.0, 0.0, 0.1]), np.array([1.0, 0.0, 0.0, 0.0]))
    tips = tip_positions(env)
    support_pts = np.stack([tips[name] for name in spec["placement_support"] if name in tips], axis=0)
    center = support_pts.mean(axis=0) + np.asarray(spec["center_offset"], dtype=np.float64)
    theta = base_theta + np.deg2rad(float(spec["phase_deg"]))
    axis = np.array([np.cos(theta), np.sin(theta), float(spec["axis_z"])], dtype=np.float64)
    quat = quat_from_x_axis(axis)
    set_state(env, hand_qpos, center, quat)
    fingers, geoms, max_penetration = contact_snapshot(env)
    return {
        "stage": "M3.2.9 photo five phase seed",
        "phase": spec["phase"],
        "name": spec["name"],
        "summary": spec["summary"],
        "desired_contacts": spec["desired_contacts"],
        "placement_support": spec["placement_support"],
        "phase_deg": spec["phase_deg"],
        "model_path": hand_record["model_path"],
        "hand_pose_source": hand_record["_path"],
        "hand_qpos": hand_qpos.astype(float).tolist(),
        "target_qpos": hand_qpos.astype(float).tolist(),
        "joint_order": DEX5_JOINT_ORDER,
        "pen_pos": center.astype(float).tolist(),
        "pen_quat": quat.astype(float).tolist(),
        "pen_qvel": [0.0] * 6,
        "contact_fingers_initial": sorted(fingers),
        "contact_geoms_top": geoms.most_common(20),
        "max_penetration": float(max_penetration),
        "actuator_kp": 400.0,
        "actuator_kv": 20.0,
        "thumb_force_limit": 0.0,
        "visual_only_markers": True,
        "disable_thumb_mesh_collision": True,
        "disable_distal_mesh_collision": True,
        "validation_status": "not_hold_validated",
    }


def render_frame(env: Dex5PenSpinMinimalEnv, label: str, frame_idx: int | None = None, total: int | None = None) -> np.ndarray:
    frame = env.render()
    if Image is None or ImageDraw is None:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    fingers, _, _ = contact_snapshot(env)
    prefix = label
    if frame_idx is not None and total is not None:
        prefix = f"{label} {frame_idx + 1}/{total}"
    text = f"{prefix} contacts:{','.join(sorted(fingers)) or 'none'}"
    draw.rectangle([0, 0, image.width, 36], fill=(255, 255, 255))
    draw.text((8, 8), text, fill=(0, 0, 0))
    return np.asarray(image)


def save_montage(frames: list[np.ndarray], output_path: Path, cols: int = 5) -> None:
    if Image is None or not frames:
        return
    images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    w, h = images[0].size
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h), (245, 245, 245))
    for idx, image in enumerate(images):
        canvas.paste(image, ((idx % cols) * w, (idx // cols) * h))
    canvas.save(output_path)


def build_preview_frames(seeds: list[dict[str, Any]], transition_frames: int, hold_frames: int) -> list[dict[str, Any]]:
    frames = []
    for i in range(len(seeds) - 1):
        a = seeds[i]
        b = seeds[i + 1]
        for _ in range(hold_frames):
            frames.append({"phase": a["phase"], "name": a["name"], "hand_qpos": np.asarray(a["hand_qpos"]), "pen_pos": np.asarray(a["pen_pos"]), "pen_quat": np.asarray(a["pen_quat"])})
        for j in range(transition_frames):
            t = smoothstep((j + 1) / transition_frames)
            frames.append(
                {
                    "phase": f"{a['phase']}->{b['phase']}",
                    "name": f"{a['name']} to {b['name']}",
                    "hand_qpos": (1.0 - t) * np.asarray(a["hand_qpos"]) + t * np.asarray(b["hand_qpos"]),
                    "pen_pos": (1.0 - t) * np.asarray(a["pen_pos"]) + t * np.asarray(b["pen_pos"]),
                    "pen_quat": slerp(np.asarray(a["pen_quat"]), np.asarray(b["pen_quat"]), t),
                }
            )
    for _ in range(hold_frames):
        last = seeds[-1]
        frames.append({"phase": last["phase"], "name": last["name"], "hand_qpos": np.asarray(last["hand_qpos"]), "pen_pos": np.asarray(last["pen_pos"]), "pen_quat": np.asarray(last["pen_quat"])})
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/paper_teleop_hand_poses_m3_2_8",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/photo_five_phase_debug_m3_2_9",
    )
    parser.add_argument("--transition-frames", type=int, default=40)
    parser.add_argument("--hold-frames", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hand_records = load_hand_records(source_dir)
    model_path = hand_records[0]["model_path"]
    env = make_env(model_path)
    env.reset(seed=0)

    # Use the first captured hand pose as the visual orientation reference.
    base_theta = np.deg2rad(20.0)
    seeds: list[dict[str, Any]] = []
    seed_frames: list[np.ndarray] = []
    for hand_record, spec in zip(hand_records, PHASE_SPECS):
        seed = build_phase_seed(env, hand_record, spec, base_theta=base_theta)
        seeds.append(seed)
        seed_path = output_dir / f"P{seed['phase']}_{seed['name']}_seed.json"
        seed_path.write_text(json.dumps(seed, indent=2, ensure_ascii=False), encoding="utf-8")
        set_state(env, np.asarray(seed["hand_qpos"]), np.asarray(seed["pen_pos"]), np.asarray(seed["pen_quat"]))
        frame = render_frame(env, f"P{seed['phase']} {seed['name']}")
        seed_frames.append(frame)
        imageio.imwrite(output_dir / f"P{seed['phase']}_{seed['name']}_seed.png", frame)

    seed_montage = output_dir / "photo_five_phase_seed_montage.png"
    save_montage(seed_frames, seed_montage, cols=5)

    preview_specs = build_preview_frames(seeds, args.transition_frames, args.hold_frames)
    preview_frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(preview_specs):
        set_state(env, spec["hand_qpos"], spec["pen_pos"], spec["pen_quat"])
        state = env.get_state()
        fingers, geoms, max_penetration = contact_snapshot(env)
        preview_frames.append(render_frame(env, str(spec["phase"]), idx, len(preview_specs)))
        rows.append(
            {
                "frame": idx,
                "phase": spec["phase"],
                "name": spec["name"],
                "pen_x": float(state.pen_center[0]),
                "pen_y": float(state.pen_center[1]),
                "pen_z": float(state.pen_center[2]),
                "axis_x": float(state.pen_axis[0]),
                "axis_y": float(state.pen_axis[1]),
                "axis_z": float(state.pen_axis[2]),
                "contacts": ",".join(sorted(fingers)),
                "max_penetration": float(max_penetration),
                "top_contacts": geoms.most_common(8),
            }
        )

    gif_path = output_dir / "photo_five_phase_preview.gif"
    imageio.mimsave(gif_path, preview_frames, fps=args.fps)
    preview_montage = output_dir / "photo_five_phase_preview_montage.png"
    save_montage(
        [preview_frames[i] for i in np.linspace(0, len(preview_frames) - 1, 12, dtype=int)],
        preview_montage,
        cols=4,
    )

    csv_path = output_dir / "photo_five_phase_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "stage": "M3.2.9 photo-guided five-phase debug",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "mode": "kinematic photo-guided seed placement; not physical rolling",
        "seed_montage": str(seed_montage),
        "preview_gif": str(gif_path),
        "preview_montage": str(preview_montage),
        "trajectory_csv": str(csv_path),
        "phase_specs": PHASE_SPECS,
        "seeds": [
            {
                "phase": seed["phase"],
                "name": seed["name"],
                "desired_contacts": seed["desired_contacts"],
                "initial_contacts": seed["contact_fingers_initial"],
                "pen_pos": seed["pen_pos"],
                "pen_quat": seed["pen_quat"],
                "max_penetration": seed["max_penetration"],
            }
            for seed in seeds
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# M3.2.9 Photo-Guided Five-Phase Debug",
        "",
        "Five phases follow the user's real teleoperation/photo sequence.",
        "",
        "| phase | name | intended motion | desired contacts | initial contacts |",
        "|---:|---|---|---|---|",
    ]
    for seed in seeds:
        lines.append(
            f"| {seed['phase']} | {seed['name']} | {seed['summary']} | "
            f"{','.join(seed['desired_contacts'])} | {','.join(seed['contact_fingers_initial']) or 'none'} |"
        )
    lines += [
        "",
        "Current output is for visual/debug alignment only. Contact and hold stability still need tuning.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    env.close()

    print(f"wrote seeds: {len(seeds)}")
    print(f"seed montage: {seed_montage}")
    print(f"preview gif: {gif_path}")
    print(f"preview montage: {preview_montage}")
    print(f"trajectory: {csv_path}")


if __name__ == "__main__":
    main()
