from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = PROJECT_ROOT / (
    "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0/"
    "canonical_reset_bank/C1_C6_validated.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0/"
    "visual_gallery"
)


parser = argparse.ArgumentParser(description="Render the O0 canonical visual audit gallery.")
parser.add_argument("--bank", default=str(DEFAULT_BANK))
parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
parser.add_argument("--samples-per-canonical", type=int, default=50)
parser.add_argument("--video-frames", type=int, default=6)
parser.add_argument("--fps", type=int, default=12)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--seed", type=int, default=20260730)
parser.add_argument(
    "--canonical",
    choices=("C1", "C2", "C3", "C4", "C5", "C6", "P1", "P2", "P3", "P4", "P5"),
)
parser.add_argument("--render-backend", choices=("rtx", "software"), default="software")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from dex5_penspin.isaac.canonical_reset_schema import (  # noqa: E402
    FINGERTIP_NAMES,
    sha256_file,
    write_json_atomic,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


TASK = "Dex5-CanonicalReset-Physics-Direct-v0"
FINGER_COLORS = {
    "thumb": "#d4503f",
    "index": "#2f7ed8",
    "middle": "#31a354",
    "ring": "#d99018",
    "pinky": "#8c6bb1",
}
FINGER_NUMBERS = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _select_rows(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args_cli.seed)
    selected: dict[str, np.ndarray] = {}
    canonical_ids = (
        (args_cli.canonical,)
        if args_cli.canonical
        else tuple(sorted(str(value) for value in np.unique(data["metadata_canonical_id"])))
    )
    for canonical_id in canonical_ids:
        candidates = np.flatnonzero(data["metadata_canonical_id"] == canonical_id)
        count = min(args_cli.samples_per_canonical, candidates.size)
        if count < args_cli.samples_per_canonical:
            raise ValueError(
                f"{canonical_id} only has {candidates.size} states; requested "
                f"{args_cli.samples_per_canonical}"
            )
        selected[canonical_id] = np.sort(rng.choice(candidates, count, replace=False))
    return selected


def _render_nonblank(direct, attempts: int = 12) -> np.ndarray:
    image = np.zeros((args_cli.height, args_cli.width, 3), dtype=np.uint8)
    for _ in range(attempts):
        image = np.asarray(direct.render(recompute=True), dtype=np.uint8)
        if image.size and float(image.std()) > 2.0:
            return image[:, :, :3]
    raise RuntimeError(
        f"Isaac renderer stayed blank after {attempts} attempts; "
        f"last shape={image.shape}, std={float(image.std()):.3f}"
    )


def _quat_rotate_x(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return np.asarray(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        dtype=np.float64,
    )


def _finger_chains(body_names: list[str], body_positions: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    positions = {name: body_positions[index] for index, name in enumerate(body_names)}
    palm_candidates = [
        name for name in body_names if "palm" in name.lower() or "base" in name.lower()
    ]
    palm = positions[palm_candidates[0]] if palm_candidates else np.mean(body_positions[:2], axis=0)
    chains: dict[str, np.ndarray] = {}
    for finger_name in FINGERTIP_NAMES:
        number = FINGER_NUMBERS[finger_name]
        names = [f"Link_{number}{joint}R" for joint in range(1, 5)]
        chain = [palm, *(positions[name] for name in names if name in positions)]
        if len(chain) >= 2:
            chains[finger_name] = np.asarray(chain)
    return palm, chains


def _render_software(
    body_names: list[str],
    body_positions: np.ndarray,
    pen_pose: np.ndarray,
    contact_state: np.ndarray,
    canonical_id: str,
    state_id: int,
    control_step: int,
) -> np.ndarray:
    _, chains = _finger_chains(body_names, body_positions)
    pen_center = np.asarray(pen_pose[:3], dtype=np.float64)
    pen_axis = _quat_rotate_x(pen_pose[3:7])
    pen_ends = np.stack((pen_center - 0.07075 * pen_axis, pen_center + 0.07075 * pen_axis))
    eye = np.asarray((0.29, -0.31, 0.27), dtype=np.float64)
    target = np.asarray((0.0, 0.0, 0.115), dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    visible_points = np.concatenate((*chains.values(), pen_ends), axis=0)
    visible_relative = visible_points - target
    visible_x = visible_relative @ right
    visible_y = visible_relative @ up
    center_x = 0.5 * (float(visible_x.min()) + float(visible_x.max()))
    center_y = 0.5 * (float(visible_y.min()) + float(visible_y.max()))
    span_x = max(float(np.ptp(visible_x)), 0.06)
    span_y = max(float(np.ptp(visible_y)), 0.06)
    scale = min(0.78 * args_cli.width / span_x, 0.68 * args_cli.height / span_y)

    def project(points: np.ndarray) -> list[tuple[int, int]]:
        relative = np.asarray(points, dtype=np.float64) - target
        x = args_cli.width * 0.5 + scale * ((relative @ right) - center_x)
        y = args_cli.height * 0.56 - scale * ((relative @ up) - center_y)
        return [(int(px), int(py)) for px, py in zip(x, y, strict=True)]

    canvas = Image.new("RGB", (args_cli.width, args_cli.height), (235, 239, 242))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, int(args_cli.height * 0.76), args_cli.width, args_cli.height), fill=(211, 217, 221))
    for finger_index, finger_name in enumerate(FINGERTIP_NAMES):
        if finger_name not in chains:
            continue
        pixels = project(chains[finger_name])
        color_hex = FINGER_COLORS[finger_name].lstrip("#")
        color = tuple(int(color_hex[index : index + 2], 16) for index in (0, 2, 4))
        draw.line(pixels, fill=color, width=14, joint="curve")
        for point in pixels:
            draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=(250, 250, 250), outline=color, width=3)
        if bool(contact_state[finger_index]):
            point = pixels[-1]
            draw.ellipse((point[0] - 12, point[1] - 12, point[0] + 12, point[1] + 12), outline=(20, 150, 80), width=4)
    pen_pixels = project(pen_ends)
    draw.line(pen_pixels, fill=(225, 45, 45), width=13)
    center_pixel = project(pen_center[None, :])[0]
    draw.ellipse((center_pixel[0] - 5, center_pixel[1] - 5, center_pixel[0] + 5, center_pixel[1] + 5), fill=(20, 20, 20))
    draw.rectangle((14, 14, 440, 76), fill=(24, 32, 42))
    draw.text((26, 25), f"{canonical_id}  state {state_id}", fill=(255, 255, 255))
    draw.text((26, 48), f"Isaac PhysX body state | software projection | step {control_step}", fill=(190, 208, 224))
    return np.asarray(canvas, dtype=np.uint8)


def _save_skeleton(
    path: Path,
    body_names: list[str],
    body_positions: np.ndarray,
    pen_pose: np.ndarray,
    contact_state: np.ndarray,
    canonical_id: str,
    state_id: int,
) -> None:
    positions = {name: body_positions[index] for index, name in enumerate(body_names)}
    figure = plt.figure(figsize=(7.2, 7.2), dpi=120)
    axis = figure.add_subplot(111, projection="3d")
    _, chains = _finger_chains(body_names, body_positions)
    for finger_index, finger_name in enumerate(FINGERTIP_NAMES):
        if finger_name not in chains:
            continue
        chain_array = chains[finger_name]
        color = FINGER_COLORS[finger_name]
        width = 3.5 if bool(contact_state[finger_index]) else 2.0
        axis.plot(
            chain_array[:, 0],
            chain_array[:, 1],
            chain_array[:, 2],
            color=color,
            linewidth=width,
            marker="o",
            markersize=4,
            label=f"{finger_name}{' contact' if contact_state[finger_index] else ''}",
        )

    pen_center = np.asarray(pen_pose[:3], dtype=np.float64)
    pen_axis = _quat_rotate_x(pen_pose[3:7])
    pen_ends = np.stack((pen_center - 0.07075 * pen_axis, pen_center + 0.07075 * pen_axis))
    axis.plot(
        pen_ends[:, 0], pen_ends[:, 1], pen_ends[:, 2], color="#e53935", linewidth=6, label="pen"
    )
    axis.scatter(*pen_center, color="#111111", s=24)

    points = np.concatenate((body_positions, pen_ends), axis=0)
    center = np.mean(points, axis=0)
    radius = max(float(np.ptp(points, axis=0).max()) * 0.58, 0.09)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=24, azim=-52)
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_zlabel("world Z (m)")
    axis.set_title(f"{canonical_id} | state {state_id} | Isaac body skeleton")
    axis.legend(loc="upper left", fontsize=7)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _write_gallery_index(directory: Path, canonical_id: str, records: list[dict[str, object]]) -> None:
    cards = []
    for record in records:
        ordinal = int(record["ordinal"])
        cards.append(
            f"""
            <article>
              <h2>{ordinal:04d} <small>state {record['state_id']}</small></h2>
              <div class="media">
                <img src="{ordinal:04d}.png" loading="lazy" alt="Isaac screenshot">
                <img src="{ordinal:04d}_skeleton.png" loading="lazy" alt="Isaac body skeleton">
              </div>
              <video controls preload="metadata" src="{ordinal:04d}.mp4"></video>
              <p>split={html.escape(str(record['split']))} | source={record['source_trajectory_id']} | "
            f"penetration={float(record['max_penetration_mm']):.3f} mm | "
            f"R3a={float(record['r3a_ratio']):.0%} | "
            f"R3b={float(record['r3b_r2_ratio']):.0%}</p>
            </article>
            """
        )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{canonical_id} visual audit</title>
<style>
body{{margin:0;background:#eef1f4;color:#18202a;font:14px/1.45 system-ui,sans-serif}}header{{padding:24px 3vw;background:#18202a;color:white}}main{{padding:20px 3vw;display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}}article{{background:white;border:1px solid #ccd3da;border-radius:6px;padding:14px}}h2{{margin:0 0 10px;font-size:18px}}small{{font-weight:400;color:#687481}}.media{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}img,video{{display:block;width:100%;background:#101418}}video{{margin-top:8px;aspect-ratio:4/3}}p{{overflow-wrap:anywhere;color:#45515e}}
</style></head><body><header><h1>{canonical_id} Canonical Visual Audit</h1><p>{len(records)} deterministic random samples from the final validated bank</p></header><main>{''.join(cards)}</main></body></html>"""
    (directory / "index.html").write_text(page, encoding="utf-8")


def _write_root_index(output_dir: Path, manifest: dict[str, object]) -> None:
    links = "".join(
        f'<li><a href="{canonical_id}_gallery/index.html">{canonical_id}</a> '
        f'({len(records)} samples)</li>'
        for canonical_id, records in manifest["canonical_galleries"].items()
    )
    render_label = (
        "Isaac RTX mesh screenshot"
        if manifest["rtx_mesh_rendered"]
        else "Isaac PhysX body-state software projection"
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>O0 Visual Audit</title>
<style>body{{max-width:820px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui,sans-serif;color:#17212b}}a{{color:#1769aa}}code{{background:#edf1f4;padding:2px 5px}}</style></head>
<body><h1>O0 Canonical Reset Visual Audit</h1><p>{render_label}, approximately 100 ms continuation video, and Isaac-body skeleton for each sampled state.</p><ul>{links}</ul><p>seed=<code>{manifest['seed']}</code>, bank SHA-256=<code>{manifest['bank_sha256']}</code></p></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    bank_path = _resolve(args_cli.bank)
    output_dir = _resolve(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(bank_path, allow_pickle=False) as loaded:
        bank = {key: loaded[key] for key in loaded.files}
    selections = _select_rows(bank)

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.bundle_path = str(bank_path)
    env_cfg.fixed_reset_bank_id = 0
    render_mode = None
    if args_cli.render_backend == "rtx":
        env_cfg.viewer.eye = (0.29, -0.31, 0.27)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.115)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.resolution = (args_cli.width, args_cli.height)
        render_mode = "rgb_array"
    env = gym.make(TASK, cfg=env_cfg, render_mode=render_mode)
    direct = env.unwrapped
    action = torch.zeros((1, 20), device=direct.device)

    body_names = list(direct.hand.body_names)
    manifest: dict[str, object] = {
        "status": "rendering",
        "bank": str(bank_path),
        "bank_sha256": sha256_file(bank_path),
        "seed": args_cli.seed,
        "samples_per_canonical": args_cli.samples_per_canonical,
        "video_frames": args_cli.video_frames,
        "width": args_cli.width,
        "height": args_cli.height,
        "physics_dt": float(direct.cfg.sim.dt),
        "control_dt": float(direct.cfg.sim.dt * direct.cfg.decimation),
        "body_names": body_names,
        "render_backend": args_cli.render_backend,
        "rtx_mesh_rendered": args_cli.render_backend == "rtx",
        "canonical_galleries": {},
    }

    for canonical_id, rows in selections.items():
        gallery_dir = output_dir / f"{canonical_id}_gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        for ordinal, bank_row in enumerate(rows, start=1):
            direct.cfg.fixed_reset_bank_id = int(bank_row)
            env.reset()
            frames: list[np.ndarray] = []
            body_positions = None
            pen_pose = None
            first_contact_state = None
            terminated_early = False
            for control_step in range(args_cli.video_frames):
                direct.set_replay_control_step(control_step)
                _, _, terminated, truncated, _ = env.step(action)
                current_body_positions = direct.hand.data.body_pos_w[0].detach().cpu().numpy().copy()
                current_pen_pose = direct.pen.data.root_pose_w[0].detach().cpu().numpy().copy()
                current_contact_state = (
                    direct.contact_force_norm[0].detach().cpu().numpy() > 0.03
                )
                if args_cli.render_backend == "rtx":
                    frame = _render_nonblank(direct)
                else:
                    frame = _render_software(
                        body_names,
                        current_body_positions,
                        current_pen_pose,
                        current_contact_state,
                        canonical_id,
                        int(bank["metadata_state_id"][bank_row]),
                        control_step,
                    )
                frames.append(frame)
                if control_step == 0:
                    body_positions = current_body_positions
                    pen_pose = current_pen_pose
                    first_contact_state = current_contact_state
                if bool(terminated[0] or truncated[0]):
                    terminated_early = True
                    break

            if body_positions is None or pen_pose is None or first_contact_state is None:
                raise RuntimeError(f"No Isaac state captured for bank row {bank_row}")
            image_path = gallery_dir / f"{ordinal:04d}.png"
            skeleton_path = gallery_dir / f"{ordinal:04d}_skeleton.png"
            video_path = gallery_dir / f"{ordinal:04d}.mp4"
            Image.fromarray(frames[0]).save(image_path)
            _save_skeleton(
                skeleton_path,
                body_names,
                body_positions,
                pen_pose,
                first_contact_state,
                canonical_id,
                int(bank["metadata_state_id"][bank_row]),
            )
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args_cli.fps),
                (args_cli.width, args_cli.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {video_path}")
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

            record = {
                "ordinal": ordinal,
                "bank_row": int(bank_row),
                "canonical_id": canonical_id,
                "state_id": int(bank["metadata_state_id"][bank_row]),
                "source_trajectory_id": int(bank["metadata_source_trajectory_id"][bank_row]),
                "split": str(bank["metadata_split"][bank_row]),
                "state_type": str(bank["labels_static_or_dynamic"][bank_row]),
                "max_penetration_mm": float(
                    bank["labels_baseline_max_penetration_m"][bank_row] * 1000.0
                ),
                "r3a_ratio": float(bank["labels_r3a_success_ratio"][bank_row]),
                "r3b_r1_ratio": float(bank["labels_r3b_r1_success_ratio"][bank_row]),
                "r3b_r2_ratio": float(bank["labels_r3b_r2_success_ratio"][bank_row]),
                "contact_state": bank["contact_state"][bank_row].astype(bool).tolist(),
                "reinstantiated_contact_state": first_contact_state.astype(bool).tolist(),
                "screenshot": str(image_path.relative_to(output_dir)),
                "skeleton": str(skeleton_path.relative_to(output_dir)),
                "video": str(video_path.relative_to(output_dir)),
                "video_frame_count": len(frames),
                "terminated_early": terminated_early,
                "screenshot_pixel_std": float(frames[0].std()),
            }
            records.append(record)
            print(
                "CANONICAL_GALLERY_PROGRESS="
                + json.dumps(
                    {"canonical": canonical_id, "completed": ordinal, "total": len(rows)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        manifest["canonical_galleries"][canonical_id] = records
        _write_gallery_index(gallery_dir, canonical_id, records)

    all_records = [
        record
        for records in manifest["canonical_galleries"].values()
        for record in records
    ]
    manifest["status"] = "pass" if all(
        not record["terminated_early"] and record["screenshot_pixel_std"] > 2.0
        for record in all_records
    ) else "fail"
    manifest["total_samples"] = len(all_records)
    manifest["early_termination_count"] = sum(
        bool(record["terminated_early"]) for record in all_records
    )
    manifest["blank_screenshot_count"] = sum(
        float(record["screenshot_pixel_std"]) <= 2.0 for record in all_records
    )
    write_json_atomic(output_dir / "visual_gallery_manifest.json", manifest)
    _write_root_index(output_dir, manifest)
    print("CANONICAL_GALLERY_REPORT=" + json.dumps(manifest, ensure_ascii=False), flush=True)
    env.close()
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
