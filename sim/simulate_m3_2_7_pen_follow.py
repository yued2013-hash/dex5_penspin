from __future__ import annotations

import argparse
import csv
import json
import time
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

from dex5_penspin_env import Dex5PenSpinMinimalEnv
from teleop_m3_2_6_canonical_keyframes import contact_snapshot


KEY_ORDER = ["K0", "K1", "K2", "K3", "K4", "K5"]


def load_seed_library(seed_dir: Path) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in seed_dir.glob("K*_paper_seed.json"):
        if path.name.endswith("latest.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        by_key[data["keyframe_id"]] = data
    missing = [key for key in KEY_ORDER if key not in by_key]
    if missing:
        raise RuntimeError(f"Missing paper seed keyframes: {missing}")
    return [by_key[key] for key in KEY_ORDER]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quat(q0 + t * (q1 - q0))
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta0 = np.sin(theta0)
    theta = theta0 * t
    s0 = np.sin(theta0 - theta) / sin_theta0
    s1 = np.sin(theta) / sin_theta0
    return normalize_quat(s0 * q0 + s1 * q1)


def quat_from_x_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    dot = float(np.clip(np.dot(x_axis, axis), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rot_axis = np.cross(x_axis, axis)
    rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-12)
    angle = float(np.arccos(dot))
    return np.array(
        [
            np.cos(0.5 * angle),
            *(np.sin(0.5 * angle) * rot_axis),
        ],
        dtype=np.float64,
    )


def pen_axis_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    w, x, y, z = q
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float64,
    )


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def make_env(model_path: str) -> Dex5PenSpinMinimalEnv:
    return Dex5PenSpinMinimalEnv(
        model_path=model_path,
        pen_init_pos=(0.0, 0.0, 0.1),
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=60.0,
        actuator_kp=400.0,
        actuator_kv=20.0,
        thumb_force_limit=0.0,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        render_mode="rgb_array",
        seed=0,
    )


def set_kinematic_state(
    env: Dex5PenSpinMinimalEnv,
    hand_qpos: np.ndarray,
    pen_pos: np.ndarray,
    pen_quat: np.ndarray,
) -> None:
    hand_qpos = np.clip(np.asarray(hand_qpos, dtype=np.float64), env.joint_low, env.joint_high)
    pen_quat = normalize_quat(pen_quat)
    env.target_qpos = hand_qpos.astype(np.float32)
    env.data.qpos[env.qpos_adr] = hand_qpos
    env.data.ctrl[:] = hand_qpos
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = np.asarray(pen_pos, dtype=np.float64)
    env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = pen_quat
    env.data.qvel[:] = 0.0
    env.prev_phase = env._quat_twist_phase_about_x(pen_quat)
    env.rotation_phase = 0.0
    mujoco.mj_forward(env.model, env.data)


def render_frame(env: Dex5PenSpinMinimalEnv, label: str, frame_idx: int, total: int) -> np.ndarray:
    frame = env.render()
    if Image is None or ImageDraw is None:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    fingers, _, _ = contact_snapshot(env)
    text = f"{label}  frame {frame_idx + 1}/{total}  contacts:{','.join(sorted(fingers)) or 'none'}"
    draw.rectangle([0, 0, image.width, 34], fill=(255, 255, 255))
    draw.text((8, 8), text, fill=(0, 0, 0))
    return np.asarray(image)


def build_frames(
    seeds: list[dict[str, Any]],
    transition_frames: int,
    hold_frames: int,
    *,
    closed_cycle: bool = True,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    n = len(seeds)
    segment_count = n if closed_cycle else n - 1
    for i in range(segment_count):
        seed_a = seeds[i]
        seed_b = seeds[(i + 1) % n]
        for h in range(hold_frames):
            frames.append(
                {
                    "phase": seed_a["keyframe_id"],
                    "label": seed_a["keyframe_label"],
                    "alpha": 0.0,
                    "hand_qpos": np.asarray(seed_a["hand_qpos"], dtype=np.float64),
                    "pen_pos": np.asarray(seed_a["pen_pos"], dtype=np.float64),
                    "pen_quat": np.asarray(seed_a["pen_quat"], dtype=np.float64),
                }
            )
        for j in range(transition_frames):
            t = smoothstep((j + 1) / transition_frames)
            frames.append(
                {
                    "phase": f"{seed_a['keyframe_id']}->{seed_b['keyframe_id']}",
                    "label": f"{seed_a['keyframe_label']} to {seed_b['keyframe_label']}",
                    "alpha": float(t),
                    "hand_qpos": (1.0 - t) * np.asarray(seed_a["hand_qpos"], dtype=np.float64)
                    + t * np.asarray(seed_b["hand_qpos"], dtype=np.float64),
                    "pen_pos": (1.0 - t) * np.asarray(seed_a["pen_pos"], dtype=np.float64)
                    + t * np.asarray(seed_b["pen_pos"], dtype=np.float64),
                    "pen_quat": slerp(
                        np.asarray(seed_a["pen_quat"], dtype=np.float64),
                        np.asarray(seed_b["pen_quat"], dtype=np.float64),
                        t,
                    ),
                }
            )
    return frames


def build_paper_schematic_180_frames(
    seeds: list[dict[str, Any]],
    transition_frames: int,
    hold_frames: int,
    *,
    orbit_scale: float,
) -> list[dict[str, Any]]:
    """Build an open-chain K0->K5 preview shaped like the paper's six-image schematic.

    The paper figure is not a metric pose source, so these anchors encode the
    visual intent: two shallow diagonal starts, a vertical-ish middle phase,
    opposite diagonal, near-horizontal, and a 180-degree closure.
    """
    if len(seeds) != 6:
        raise RuntimeError("paper schematic preview expects exactly six K0-K5 seeds")

    seed_positions = np.asarray([seed["pen_pos"] for seed in seeds], dtype=np.float64)
    center0 = seed_positions.mean(axis=0)
    k0_axis = pen_axis_from_quat_wxyz(np.asarray(seeds[0]["pen_quat"], dtype=np.float64))
    theta0 = float(np.arctan2(k0_axis[1], k0_axis[0]))

    # Relative phase anchors chosen to match the paper panel layout:
    # K0/K1 are similar, K2 becomes vertical, K3 opposite diagonal,
    # K4 almost horizontal after the flip, K5 closes the half-turn.
    rel_deg = np.array([0.0, 12.0, 65.0, 115.0, 155.0, 180.0], dtype=np.float64)
    z_axis = np.array([0.06, 0.05, 0.02, 0.05, 0.00, -0.04], dtype=np.float64)

    center_offsets = orbit_scale * np.array(
        [
            [0.70, 0.20, 0.35],
            [0.55, 0.35, 0.20],
            [0.10, 0.45, -0.15],
            [-0.35, 0.25, -0.10],
            [-0.65, 0.05, 0.15],
            [-0.70, -0.15, 0.30],
        ],
        dtype=np.float64,
    )

    frames: list[dict[str, Any]] = []

    def state_at_key(key_idx: int) -> dict[str, Any]:
        theta = theta0 + np.deg2rad(rel_deg[key_idx])
        axis = np.array([np.cos(theta), np.sin(theta), z_axis[key_idx]], dtype=np.float64)
        return {
            "phase": f"{seeds[key_idx]['keyframe_id']} paper {rel_deg[key_idx]:.1f}deg",
            "label": seeds[key_idx]["keyframe_label"],
            "alpha": float(key_idx / 5.0),
            "hand_qpos": np.asarray(seeds[key_idx]["hand_qpos"], dtype=np.float64),
            "pen_pos": center0 + center_offsets[key_idx],
            "pen_quat": quat_from_x_axis(axis),
            "paper_rel_deg": float(rel_deg[key_idx]),
        }

    key_states = [state_at_key(i) for i in range(6)]
    for i in range(5):
        a = key_states[i]
        b = key_states[i + 1]
        for _ in range(hold_frames):
            frames.append({**a})
        for j in range(transition_frames):
            t = smoothstep((j + 1) / transition_frames)
            frames.append(
                {
                    "phase": f"{seeds[i]['keyframe_id']}->{seeds[i + 1]['keyframe_id']} paper "
                    f"{(1.0 - t) * a['paper_rel_deg'] + t * b['paper_rel_deg']:.1f}deg",
                    "label": f"{seeds[i]['keyframe_label']} to {seeds[i + 1]['keyframe_label']}",
                    "alpha": (1.0 - t) * float(a["alpha"]) + t * float(b["alpha"]),
                    "hand_qpos": (1.0 - t) * np.asarray(a["hand_qpos"]) + t * np.asarray(b["hand_qpos"]),
                    "pen_pos": (1.0 - t) * np.asarray(a["pen_pos"]) + t * np.asarray(b["pen_pos"]),
                    "pen_quat": slerp(np.asarray(a["pen_quat"]), np.asarray(b["pen_quat"]), t),
                    "paper_rel_deg": (1.0 - t) * a["paper_rel_deg"] + t * b["paper_rel_deg"],
                }
            )
    for _ in range(hold_frames):
        frames.append({**key_states[-1]})
    return frames


def apply_monotonic_spin_path(
    frames: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    *,
    direction: float,
    spin_degrees: float,
    orbit_radius: float,
    orbit_y_scale: float,
    z_amp: float,
    axis_z_amp: float,
) -> None:
    """Override pen pose so the pen long axis turns one full direction-consistent cycle.

    This is a kinematic target preview, not a contact result. It keeps the hand
    interpolation from K0-K5, but makes the pen axis phase monotonic.
    """
    if not frames:
        return

    seed_positions = np.asarray([seed["pen_pos"] for seed in seeds], dtype=np.float64)
    center0 = seed_positions.mean(axis=0)
    k0_axis = pen_axis_from_quat_wxyz(np.asarray(seeds[0]["pen_quat"], dtype=np.float64))
    theta0 = float(np.arctan2(k0_axis[1], k0_axis[0]))
    total = len(frames)
    for idx, frame in enumerate(frames):
        phase = idx / max(1, total - 1)
        theta = theta0 + float(direction) * np.deg2rad(spin_degrees) * phase
        # Keep the center near the paper-seed cloud, but make the path visibly
        # cycle in one direction instead of bouncing between keyframes.
        frame["pen_pos"] = center0 + np.array(
            [
                orbit_radius * np.cos(theta),
                orbit_radius * orbit_y_scale * np.sin(theta),
                z_amp * np.sin(2.0 * theta),
            ],
            dtype=np.float64,
        )
        axis = np.array(
            [
                np.cos(theta),
                np.sin(theta),
                axis_z_amp * np.sin(theta + np.pi / 4.0),
            ],
            dtype=np.float64,
        )
        frame["pen_quat"] = quat_from_x_axis(axis)
        frame["phase"] = f"spin {np.rad2deg(theta - theta0):.1f}deg"
        frame["label"] = "monotonic one-direction pen spin preview"
        frame["alpha"] = float(phase)


def save_montage(frames: list[np.ndarray], output_path: Path, sample_count: int = 12) -> None:
    if Image is None or not frames:
        return
    idxs = np.linspace(0, len(frames) - 1, min(sample_count, len(frames)), dtype=int)
    images = [Image.fromarray(frames[i]).convert("RGB") for i in idxs]
    w, h = images[0].size
    cols = 4
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h), (245, 245, 245))
    for idx, image in enumerate(images):
        canvas.paste(image, ((idx % cols) * w, (idx // cols) * h))
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/paper_image_canonical_seeds_m3_2_7",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/paper_image_pen_follow_m3_2_7",
    )
    parser.add_argument("--transition-frames", type=int, default=36)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument(
        "--pen-motion-mode",
        choices=["seed_interp", "monotonic_spin", "paper_schematic_180"],
        default="seed_interp",
    )
    parser.add_argument("--spin-direction", choices=["ccw", "cw"], default="ccw")
    parser.add_argument("--spin-degrees", type=float, default=360.0)
    parser.add_argument("--orbit-radius", type=float, default=0.010)
    parser.add_argument("--paper-orbit-scale", type=float, default=0.010)
    parser.add_argument("--orbit-y-scale", type=float, default=0.75)
    parser.add_argument("--orbit-z-amp", type=float, default=0.004)
    parser.add_argument("--axis-z-amp", type=float, default=0.08)
    parser.add_argument("--closed-cycle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = load_seed_library(seed_dir)
    model_path = seeds[0]["model_path"]
    env = make_env(model_path)
    env.reset(seed=0)

    if args.pen_motion_mode == "paper_schematic_180":
        frames_spec = build_paper_schematic_180_frames(
            seeds,
            args.transition_frames,
            args.hold_frames,
            orbit_scale=args.paper_orbit_scale,
        ) * int(args.cycles)
    else:
        frames_spec = build_frames(
            seeds,
            args.transition_frames,
            args.hold_frames,
            closed_cycle=args.closed_cycle,
        ) * int(args.cycles)
    if args.pen_motion_mode == "monotonic_spin":
        apply_monotonic_spin_path(
            frames_spec,
            seeds,
            direction=1.0 if args.spin_direction == "ccw" else -1.0,
            spin_degrees=args.spin_degrees,
            orbit_radius=args.orbit_radius,
            orbit_y_scale=args.orbit_y_scale,
            z_amp=args.orbit_z_amp,
            axis_z_amp=args.axis_z_amp,
        )
    rendered: list[np.ndarray] = []
    trajectory_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(frames_spec):
        set_kinematic_state(env, spec["hand_qpos"], spec["pen_pos"], spec["pen_quat"])
        state = env.get_state()
        fingers, geoms, max_penetration = contact_snapshot(env)
        rendered.append(render_frame(env, str(spec["phase"]), idx, len(frames_spec)))
        trajectory_rows.append(
            {
                "frame": idx,
                "phase": spec["phase"],
                "label": spec["label"],
                "alpha": spec["alpha"],
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

    gif_path = output_dir / "paper_image_pen_follow.gif"
    imageio.mimsave(gif_path, rendered, fps=args.fps)
    montage_path = output_dir / "paper_image_pen_follow_montage.png"
    save_montage(rendered, montage_path)

    csv_path = output_dir / "pen_follow_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "phase",
                "label",
                "alpha",
                "pen_x",
                "pen_y",
                "pen_z",
                "axis_x",
                "axis_y",
                "axis_z",
                "contacts",
                "max_penetration",
                "top_contacts",
            ],
        )
        writer.writeheader()
        writer.writerows(trajectory_rows)

    summary = {
        "stage": "M3.2.7 kinematic pen follow preview",
        "seed_dir": str(seed_dir),
        "output_dir": str(output_dir),
        "frames": len(frames_spec),
        "fps": args.fps,
        "cycles": args.cycles,
        "mode": "kinematic interpolation; not a physical contact simulation",
        "pen_motion_mode": args.pen_motion_mode,
        "spin_direction": args.spin_direction if args.pen_motion_mode == "monotonic_spin" else None,
        "spin_degrees": args.spin_degrees if args.pen_motion_mode == "monotonic_spin" else None,
        "closed_cycle": args.closed_cycle,
        "gif": str(gif_path),
        "montage": str(montage_path),
        "trajectory_csv": str(csv_path),
        "keyframes": [
            {
                "key": seed["keyframe_id"],
                "label": seed["keyframe_label"],
                "pen_pos": seed["pen_pos"],
                "pen_quat": seed["pen_quat"],
                "contacts_initial": seed.get("contact_fingers_initial"),
            }
            for seed in seeds
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# M3.2.7 Paper-Image Pen Follow Preview",
        "",
        "This is a kinematic interpolation preview across K0-K5. It visualizes hand-pose and pen-pose continuity.",
        "",
        "It is not a physical rolling/contact simulation. Use it to inspect whether the phase sequence and pen path look plausible before contact tuning.",
        "",
        f"- GIF: `{gif_path.name}`",
        f"- Montage: `{montage_path.name}`",
        f"- Trajectory CSV: `{csv_path.name}`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    env.close()

    print(f"wrote GIF: {gif_path}")
    print(f"wrote montage: {montage_path}")
    print(f"wrote trajectory: {csv_path}")
    print(f"frames: {len(frames_spec)} fps: {args.fps}")


if __name__ == "__main__":
    main()
