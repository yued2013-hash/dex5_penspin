from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional visual report dependency
    Image = None
    ImageDraw = None

from dex5_penspin_env import DEX5_JOINT_ORDER, Dex5PenSpinMinimalEnv


KEYFRAME_ORDER = ["K0", "K1", "K2", "K3", "K4", "K5"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in source_dir.glob("K[0-9]_*.json"):
        name = path.name
        if name.endswith(("latest.json", "validation.json", "quick_validation.json")):
            continue
        files.append(path)
    return sorted(files)


def sanitize(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\\-]+", "_", text.strip())
    return text.strip("_") or "pose"


def make_hand_pose_record(source_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    qpos = np.asarray(data["hand_qpos"], dtype=float)
    target = np.asarray(data.get("target_qpos", data["hand_qpos"]), dtype=float)
    return {
        "stage": "M3.2.6 hand-only canonical pose library",
        "source_file": str(source_path),
        "source_capture_stage": data.get("stage"),
        "source": data.get("source", "unknown"),
        "keyframe_index": data.get("keyframe_index"),
        "keyframe_id": data.get("keyframe_id"),
        "keyframe_label": data.get("keyframe_label"),
        "keyframe_note": data.get("keyframe_note"),
        "hand_qpos": qpos.tolist(),
        "target_qpos": target.tolist(),
        "joint_order": data.get("joint_order", DEX5_JOINT_ORDER),
        "model_path": data.get("model_path"),
        "ros_source_topic": data.get("ros_source_topic"),
        "target_age_sec": data.get("target_age_sec"),
        "target_count": data.get("target_count"),
        "pen_pose_status": "not_authoritative",
        "pen_pose_note": (
            "This library intentionally preserves only the hand phase pose. "
            "Pen position/quaternion must be added later per keyframe."
        ),
        "source_pen_pos_for_reference_only": data.get("pen_pos"),
        "source_pen_quat_for_reference_only": data.get("pen_quat"),
        "qpos_stats": {
            "mean": float(np.mean(qpos)),
            "std": float(np.std(qpos)),
            "min": float(np.min(qpos)),
            "max": float(np.max(qpos)),
        },
    }


def write_pose(output_dir: Path, source_path: Path, record: dict[str, Any]) -> Path:
    key_id = record.get("keyframe_id") or "Kx"
    label = sanitize(record.get("keyframe_label") or "pose")
    source_stem = source_path.stem
    out_path = output_dir / f"{key_id}_{label}_hand_{source_stem}.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_path = output_dir / f"{key_id}_{label}_hand_latest.json"
    latest_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def render_hand_pose(record: dict[str, Any], output_path: Path) -> bool:
    if Image is None or ImageDraw is None:
        return False
    model_path = record.get("model_path")
    if not model_path:
        return False
    env = Dex5PenSpinMinimalEnv(
        model_path=model_path,
        pen_init_pos=(0.35, 0.35, 0.35),
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=1.0,
        actuator_kp=400.0,
        actuator_kv=20.0,
        thumb_force_limit=0.0,
        visual_only_markers=True,
        disable_thumb_mesh_collision=True,
        disable_distal_mesh_collision=True,
        render_mode="rgb_array",
        seed=0,
    )
    try:
        env.reset(seed=0)
        qpos = np.asarray(record["hand_qpos"], dtype=np.float64)
        env.target_qpos = qpos.astype(np.float32)
        env.data.qpos[env.qpos_adr] = qpos
        env.data.ctrl[:] = qpos
        env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 3] = np.array([0.35, 0.35, 0.35])
        env.data.qpos[env.pen_qpos_adr + 3 : env.pen_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        image = Image.fromarray(env.render())
        draw = ImageDraw.Draw(image)
        title = f"{record.get('keyframe_id')} {record.get('keyframe_label')} hand-only"
        draw.rectangle([0, 0, image.width, 34], fill=(255, 255, 255))
        draw.text((8, 8), title, fill=(0, 0, 0))
        image.save(output_path)
        return True
    finally:
        env.close()


def write_report(output_dir: Path, records: list[dict[str, Any]], written: list[Path]) -> None:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_key.setdefault(str(record.get("keyframe_id")), []).append(record)

    lines = ["# M3.2.6 Hand-Only Canonical Pose Library", ""]
    lines.append("This library preserves hand phase poses only. Pen pose and contact state must be added later.")
    lines.append("")
    lines.append("| key | label | candidates | qpos mean/std | source |")
    lines.append("|---|---|---:|---:|---|")
    for key in KEYFRAME_ORDER:
        entries = by_key.get(key, [])
        if not entries:
            lines.append(f"| {key} | missing | 0 | - | - |")
            continue
        latest = entries[-1]
        stats = latest["qpos_stats"]
        lines.append(
            f"| {key} | {latest.get('keyframe_label')} | {len(entries)} | "
            f"{stats['mean']:.3f}/{stats['std']:.3f} | {Path(latest['source_file']).name} |"
        )

    lines += [
        "",
        "Next step:",
        "1. Complete missing K1 and K5 hand poses if needed.",
        "2. Load each hand pose, manually add pen pose, then validate contact.",
        "3. Only promote poses to the reset library after zero-action hold succeeds.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "stage": "M3.2.6 hand-only canonical pose library",
        "records": records,
        "written_files": [str(p) for p in written],
        "missing_keyframes": [key for key in KEYFRAME_ORDER if key not in by_key],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/teleop_canonical_keyframes",
    )
    parser.add_argument(
        "--output-dir",
        default="dex5_penspin/reports/manus_mujoco_presets/canonical_hand_poses_m3_2_6",
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    written: list[Path] = []
    image_paths: list[Path] = []
    for path in source_files(source_dir):
        data = load_json(path)
        record = make_hand_pose_record(path, data)
        out_path = write_pose(output_dir, path, record)
        records.append(record)
        written.append(out_path)
        if args.render:
            png_path = out_path.with_suffix(".png")
            if render_hand_pose(record, png_path):
                image_paths.append(png_path)

    write_report(output_dir, records, written)

    print(f"source files: {len(records)}")
    print(f"output_dir: {output_dir}")
    print(f"written hand poses: {len(written)}")
    if image_paths:
        print(f"rendered images: {len(image_paths)}")
    print(f"missing keyframes: {[k for k in KEYFRAME_ORDER if k not in {r.get('keyframe_id') for r in records}]}")


if __name__ == "__main__":
    main()
