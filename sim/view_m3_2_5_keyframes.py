from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from dex5_penspin_env import Dex5PenSpinMinimalEnv
from tune_m2_7_tripod_fingertip_reset import contact_snapshot, set_state


DEFAULT_MANIFEST = Path("dex5_penspin/reports/manus_mujoco_presets/canonical_phase_keyframes_merged/manifest.json")


def load_keyframes(manifest_path: str | Path) -> list[tuple[Path, dict]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    keyframes = []
    for path in manifest["accepted_keyframes"]:
        p = Path(path)
        keyframes.append((p, json.loads(p.read_text(encoding="utf-8"))))
    return keyframes


def make_env(preset: dict, render_mode: str | None = None) -> Dex5PenSpinMinimalEnv:
    return Dex5PenSpinMinimalEnv(
        model_path=preset.get("model_path"),
        pen_init_pos=preset["pen_pos"],
        grasp_scale=1.0,
        init_noise_scale=0.0,
        episode_seconds=3.0,
        actuator_kp=preset.get("actuator_kp", 400.0),
        actuator_kv=preset.get("actuator_kv", 20.0),
        thumb_force_limit=preset.get("thumb_force_limit", 0.0),
        visual_only_markers=bool(preset.get("visual_only_markers", True)),
        disable_thumb_mesh_collision=bool(preset.get("disable_thumb_mesh_collision", True)),
        disable_distal_mesh_collision=bool(preset.get("disable_distal_mesh_collision", True)),
        render_mode=render_mode,
    )


def set_keyframe(env: Dex5PenSpinMinimalEnv, preset: dict) -> None:
    set_state(
        env,
        np.asarray(preset["hand_qpos"], dtype=np.float64),
        np.asarray(preset["pen_pos"], dtype=np.float64),
        np.asarray(preset["pen_quat"], dtype=np.float64),
    )
    env.prev_phase = env._quat_twist_phase_about_x(env.data.xquat[env.pen_body_id])
    env.rotation_phase = 0.0


def render_keyframes(keyframes: list[tuple[Path, dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    images = []
    for idx, (path, preset) in enumerate(keyframes):
        env = make_env(preset, render_mode="rgb_array")
        env.reset(seed=idx)
        set_keyframe(env, preset)
        image = env.render()
        fingers, geoms, penetration = contact_snapshot(env)
        label = preset.get("keyframe_label", path.stem)
        exact = preset.get("exact_contact_pattern", "")
        role = preset.get("merged_library_role", "")
        images.append((image, label, exact, sorted(fingers), role))
        rows.append(
            {
                "index": idx,
                "path": str(path),
                "label": label,
                "exact_contact_pattern": exact,
                "visible_support_fingers": sorted(fingers),
                "role": role,
                "max_penetration": float(penetration),
            }
        )
        env.close()

    n = len(images)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.5 * cols, 4.3 * rows_n))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for idx, (image, label, exact, fingers, role) in enumerate(images):
        ax = axes[idx]
        ax.imshow(image)
        ax.axis("off")
        title = f"{idx:02d} {label}\n{exact}\ncontact: {'/'.join(fingers) or 'none'}"
        if "transient" in role:
            title += "\ntransient"
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    montage_path = output_dir / "m3_2_5_keyframes_montage.png"
    fig.savefig(montage_path, dpi=170)
    plt.close(fig)
    (output_dir / "m3_2_5_keyframes_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return montage_path


def live_view(keyframes: list[tuple[Path, dict]], args: argparse.Namespace) -> None:
    if not keyframes:
        return
    _, first = keyframes[0]
    env = make_env(first, render_mode=None)
    env.reset(seed=args.seed)
    set_keyframe(env, first)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(args.cycles):
            for idx, (path, preset) in enumerate(keyframes):
                if not viewer.is_running():
                    break
                set_keyframe(env, preset)
                fingers, _, _ = contact_snapshot(env)
                print(
                    f"show keyframe {idx:02d}: "
                    f"{preset.get('keyframe_label', path.stem)} "
                    f"exact={preset.get('exact_contact_pattern', '')} "
                    f"contact={'/'.join(sorted(fingers)) or 'none'}",
                    flush=True,
                )
                start = time.time()
                while viewer.is_running() and time.time() - start < args.pause:
                    viewer.sync()
                    time.sleep(0.03)
        if args.pause_on_end:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default="dex5_penspin/reports/m3_2_5_keyframe_visualization")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--pause", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--pause-on-end", action="store_true")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    keyframes = load_keyframes(args.manifest_path)
    montage_path = render_keyframes(keyframes, Path(args.output_dir))
    print(f"saved montage: {montage_path}")
    if args.viewer:
        live_view(keyframes, args)


if __name__ == "__main__":
    main()
