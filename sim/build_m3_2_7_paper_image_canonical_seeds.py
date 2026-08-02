from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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

KEY_SPECS = [
    {
        "id": "K0",
        "label": "tripod_hold",
        "support": ["index", "middle", "ring"],
        "axis": [0.90, 0.42, 0.06],
        "note": "paper figure seed 1: pen diagonal, tripod support.",
    },
    {
        "id": "K1",
        "label": "thumb_light_clamp",
        "support": ["index", "middle", "ring", "thumb"],
        "axis": [0.86, 0.48, 0.04],
        "note": "paper figure seed 2: diagonal pen with light thumb contact.",
    },
    {
        "id": "K2",
        "label": "index_release",
        "support": ["middle", "ring", "thumb"],
        "axis": [0.05, 0.99, 0.02],
        "note": "paper figure seed 3: near vertical/front-back pen, index releases.",
    },
    {
        "id": "K3",
        "label": "middle_push",
        "support": ["middle", "ring", "thumb"],
        "axis": [0.82, -0.55, 0.08],
        "note": "paper figure seed 4: opposite diagonal during push.",
    },
    {
        "id": "K4",
        "label": "thumb_index_regrasp",
        "support": ["thumb", "index", "middle"],
        "axis": [1.0, 0.0, 0.0],
        "note": "paper figure seed 5: horizontal regrasp.",
    },
    {
        "id": "K5",
        "label": "cycle_closure",
        "support": ["index", "middle", "ring", "thumb"],
        "axis": [0.88, -0.46, 0.05],
        "note": "paper figure seed 6: diagonal closure back toward K0.",
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
    return np.array(
        [
            np.cos(0.5 * angle),
            *(np.sin(0.5 * angle) * rot_axis),
        ],
        dtype=np.float64,
    )


def load_records(source_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(source_dir.glob("K*_hand_*.json")):
        if path.name.endswith("latest.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        records.append(data)
    return records


def choose_hand_poses(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_key.setdefault(record["keyframe_id"], []).append(record)

    chosen: dict[str, dict[str, Any]] = {}
    if by_key.get("K0"):
        # The more closed K0 candidate is a better paper-style seed for a pen support shelf.
        chosen["K0"] = max(by_key["K0"], key=lambda r: r["qpos_stats"]["mean"])
    for key in ["K2", "K3", "K4"]:
        if by_key.get(key):
            chosen[key] = by_key[key][-1]

    if "K0" not in chosen:
        raise RuntimeError("K0 hand pose is required to synthesize paper-like seeds.")
    if "K2" not in chosen or "K3" not in chosen or "K4" not in chosen:
        raise RuntimeError("K2/K3/K4 hand poses are required. Capture or provide hand-only poses first.")

    k0 = np.asarray(chosen["K0"]["hand_qpos"], dtype=np.float64)
    k2 = np.asarray(chosen["K2"]["hand_qpos"], dtype=np.float64)
    k4 = np.asarray(chosen["K4"]["hand_qpos"], dtype=np.float64)

    k1 = 0.55 * k0 + 0.45 * k2
    k1[[16, 17, 18]] += np.array([0.0, 0.10, 0.08])
    k1[19] = max(k1[19], 0.12)

    k5 = 0.60 * k4 + 0.40 * k0
    k5[[0, 1, 4, 5, 8, 9, 17, 18]] += np.array([-0.03, 0.08, 0.02, 0.05, -0.02, 0.05, 0.04, 0.04])

    template = chosen["K0"].copy()
    chosen["K1"] = {
        **template,
        "keyframe_id": "K1",
        "keyframe_label": "thumb_light_clamp",
        "hand_qpos": k1.tolist(),
        "target_qpos": k1.tolist(),
        "source_file": "interpolated_from_K0_K2",
        "_path": "interpolated_from_K0_K2",
        "qpos_stats": stats(k1),
    }
    chosen["K5"] = {
        **template,
        "keyframe_id": "K5",
        "keyframe_label": "cycle_closure",
        "hand_qpos": k5.tolist(),
        "target_qpos": k5.tolist(),
        "source_file": "interpolated_from_K4_K0",
        "_path": "interpolated_from_K4_K0",
        "qpos_stats": stats(k5),
    }
    return chosen


def stats(q: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(q)),
        "std": float(np.std(q)),
        "min": float(np.min(q)),
        "max": float(np.max(q)),
    }


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


def set_hand(env: Dex5PenSpinMinimalEnv, qpos: np.ndarray) -> None:
    qpos = np.clip(qpos, env.joint_low, env.joint_high)
    env.target_qpos = qpos.astype(np.float32)
    env.data.qpos[env.qpos_adr] = qpos
    env.data.ctrl[:] = qpos
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def tip_positions(env: Dex5PenSpinMinimalEnv) -> dict[str, np.ndarray]:
    out = {}
    for finger, geom_name in TIP_GEOMS.items():
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            out[finger] = env.data.geom_xpos[geom_id].copy()
    return out


def place_pen_from_support(
    env: Dex5PenSpinMinimalEnv,
    support: list[str],
    axis: np.ndarray,
    lift: float,
) -> tuple[np.ndarray, np.ndarray]:
    tips = tip_positions(env)
    pts = np.stack([tips[name] for name in support if name in tips], axis=0)
    center = pts.mean(axis=0)
    center += np.array([0.0, 0.0, lift], dtype=np.float64)
    quat = quat_from_x_axis(axis)
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = center
    env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = quat
    env.data.qvel[:] = 0.0
    env.prev_phase = env._quat_twist_phase_about_x(quat)
    env.rotation_phase = 0.0
    mujoco.mj_forward(env.model, env.data)
    return center, quat


def render_montage(image_paths: list[Path], output_path: Path) -> None:
    if Image is None or not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    w, h = images[0].size
    cols = 3
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h), (245, 245, 245))
    for idx, image in enumerate(images):
        canvas.paste(image, ((idx % cols) * w, (idx // cols) * h))
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/canonical_hand_poses_m3_2_6",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/paper_image_canonical_seeds_m3_2_7",
    )
    parser.add_argument("--pen-lift", type=float, default=0.010)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(source_dir)
    chosen = choose_hand_poses(records)
    model_path = next(record.get("model_path") for record in records if record.get("model_path"))

    written: list[str] = []
    image_paths: list[Path] = []
    summary: list[dict[str, Any]] = []
    for idx, spec in enumerate(KEY_SPECS):
        key = spec["id"]
        hand = chosen[key]
        qpos = np.asarray(hand["hand_qpos"], dtype=np.float64)
        env = make_env(model_path)
        env.reset(seed=0)
        set_hand(env, qpos)
        pen_pos, pen_quat = place_pen_from_support(
            env,
            support=spec["support"],
            axis=np.asarray(spec["axis"], dtype=np.float64),
            lift=float(args.pen_lift),
        )
        fingers, geoms, max_penetration = contact_snapshot(env)

        record = {
            "stage": "M3.2.7 paper image inspired canonical seed",
            "keyframe_index": idx,
            "keyframe_id": key,
            "keyframe_label": spec["label"],
            "keyframe_note": spec["note"],
            "hand_pose_source": hand.get("source_file") or hand.get("_path"),
            "image_inference_status": "approximate_visual_seed_not_exact_reconstruction",
            "pen_pose_status": "image_inferred_seed_not_validated",
            "support_fingers_desired": spec["support"],
            "pen_axis_desired": spec["axis"],
            "model_path": model_path,
            "hand_qpos": qpos.tolist(),
            "target_qpos": qpos.tolist(),
            "joint_order": DEX5_JOINT_ORDER,
            "pen_pos": pen_pos.astype(float).tolist(),
            "pen_quat": pen_quat.astype(float).tolist(),
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
        }
        out = output_dir / f"{key}_{spec['label']}_paper_seed.json"
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / f"{key}_{spec['label']}_paper_seed_latest.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(str(out))

        if Image is not None and ImageDraw is not None:
            image = Image.fromarray(env.render())
            draw = ImageDraw.Draw(image)
            title = f"{key} {spec['label']} contacts:{','.join(sorted(fingers)) or 'none'}"
            draw.rectangle([0, 0, image.width, 34], fill=(255, 255, 255))
            draw.text((8, 8), title, fill=(0, 0, 0))
            image_path = output_dir / f"{key}_{spec['label']}_paper_seed.png"
            image.save(image_path)
            image_paths.append(image_path)

        summary.append(
            {
                "key": key,
                "label": spec["label"],
                "support_desired": spec["support"],
                "contacts_initial": sorted(fingers),
                "pen_pos": [round(float(x), 6) for x in pen_pos],
                "pen_quat": [round(float(x), 6) for x in pen_quat],
                "hand_pose_source": hand.get("source_file") or hand.get("_path"),
            }
        )
        env.close()

    montage = output_dir / "paper_image_canonical_seeds_montage.png"
    render_montage(image_paths, montage)

    lines = [
        "# M3.2.7 Paper-Image Canonical Seeds",
        "",
        "These are approximate visual seeds inferred from the paper figure, not exact recovered poses.",
        "",
        "| key | label | desired support | initial contacts | source |",
        "|---|---|---|---|---|",
    ]
    for item in summary:
        lines.append(
            f"| {item['key']} | {item['label']} | {','.join(item['support_desired'])} | "
            f"{','.join(item['contacts_initial']) or 'none'} | {Path(str(item['hand_pose_source'])).name} |"
        )
    lines += [
        "",
        "Use these seeds only as starting points for manual pen-pose/contact tuning.",
        "Promote a seed to the reset library only after zero-action hold validation succeeds.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(written)} paper-image seeds to {output_dir}")
    print(f"montage: {montage}")


if __name__ == "__main__":
    main()
