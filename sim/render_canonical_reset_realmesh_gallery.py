from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import mujoco
import numpy as np

from dex5_penspin.sim.dex5_penspin_env import Dex5PenSpinMinimalEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = (
    ROOT
    / "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0/"
    "canonical_reset_bank/C1_C6_validated.npz"
)
DEFAULT_SOURCE_MANIFEST = (
    ROOT
    / "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0/"
    "visual_gallery/visual_gallery_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0/"
    "visual_gallery_realmesh"
)
MODEL = ROOT / "sim/assets/dex5_urdf_import/dex5_penspin_real_m2.xml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_camera(azimuth: float, distance: float = 0.235) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.075, 0.065)
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = -23.0
    return camera


def set_bank_row(env: Dex5PenSpinMinimalEnv, bank: dict[str, np.ndarray], row: int) -> None:
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env.qpos_adr] = bank["robot_qpos"][row]
    env.data.qpos[env.pen_qpos_adr : env.pen_qpos_adr + 7] = bank["object_pose"][row]
    env.data.qvel[env.qvel_adr] = bank["robot_qvel"][row]
    env.data.qvel[env.pen_qvel_adr : env.pen_qvel_adr + 3] = bank[
        "object_linear_velocity"
    ][row]
    env.data.qvel[env.pen_qvel_adr + 3 : env.pen_qvel_adr + 6] = bank[
        "object_angular_velocity"
    ][row]
    env.target_qpos = bank["robot_q_target"][row].astype(np.float32).copy()
    env.data.ctrl[:] = env.target_qpos
    mujoco.mj_forward(env.model, env.data)


def render(env: Dex5PenSpinMinimalEnv, camera: mujoco.MjvCamera) -> np.ndarray:
    env.renderer.update_scene(env.data, camera=camera)
    return env.renderer.render().copy()


def side_by_side(primary: np.ndarray, side: np.ndarray) -> np.ndarray:
    divider = np.full((primary.shape[0], 6, 3), 238, dtype=np.uint8)
    return np.concatenate((primary, divider, side), axis=1)


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def advance_control_step(
    env: Dex5PenSpinMinimalEnv,
    bank: dict[str, np.ndarray],
    row: int,
    control_step: int,
) -> None:
    length = int(bank["robot_continuation_valid_length"][row])
    if control_step < length:
        env.target_qpos = bank["robot_continuation_q_target"][row, control_step].astype(
            np.float32
        )
    env.data.ctrl[:] = env.target_qpos
    for _ in range(env.frame_skip):
        mujoco.mj_step(env.model, env.data)


def write_canonical_index(
    directory: Path, canonical_id: str, records: list[dict[str, object]]
) -> None:
    cards = []
    for record in records:
        ordinal = int(record["ordinal"])
        cards.append(
            f"""<article><h2>{ordinal:04d} <small>state {record['state_id']}</small></h2>
            <img src="{ordinal:04d}.png" loading="lazy" alt="Dex5 and pen real mesh views">
            <video controls preload="metadata" src="{ordinal:04d}.mp4"></video>
            <p>bank row={record['bank_row']} | split={html.escape(str(record['split']))} |
            penetration gate={float(record['max_penetration_mm']):.3f} mm</p></article>"""
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{canonical_id} realmesh</title>
<style>body{{margin:0;background:#eef1f4;color:#18202a;font:14px/1.45 system-ui,sans-serif}}header{{padding:24px 3vw;background:#18202a;color:white}}main{{padding:20px 3vw;display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:18px}}article{{background:white;border:1px solid #ccd3da;border-radius:6px;padding:14px}}h2{{margin:0 0 10px;font-size:18px}}small{{font-weight:400;color:#687481}}img,video{{display:block;width:100%;background:#101418}}video{{margin-top:8px;aspect-ratio:8/3}}p{{color:#45515e}}</style>
</head><body><header><h1>{canonical_id} 真实网格检查</h1><p>主视角 + 侧视角；位姿来自 O0 PhysX validated bank，由 MuJoCo realmesh 重现。</p></header><main>{''.join(cards)}</main></body></html>"""
    (directory / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render O0 bank rows with real Dex5 and pen meshes.")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-per-canonical", type=int)
    parser.add_argument("--video-frames", type=int, default=6)
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()

    bank_path = args.bank.resolve()
    source_manifest_path = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    bank_hash = sha256_file(bank_path)
    if source_manifest["bank_sha256"] != bank_hash:
        raise RuntimeError("Source gallery and reset bank hashes do not match")
    with np.load(bank_path, allow_pickle=False) as loaded:
        bank = {key: loaded[key] for key in loaded.files}

    env = Dex5PenSpinMinimalEnv(model_path=MODEL, render_mode="rgb_array", control_hz=120.0)
    primary_camera = make_camera(142.0)
    side_camera = make_camera(215.0)
    output_manifest: dict[str, object] = {
        "status": "rendering",
        "evidence_level": "MuJoCo realmesh geometry and sim2sim preview",
        "physx_validation_source": str(bank_path),
        "bank_sha256": bank_hash,
        "source_manifest": str(source_manifest_path),
        "model": str(MODEL),
        "canonical_galleries": {},
    }
    for canonical_id, source_records in source_manifest["canonical_galleries"].items():
        selected = source_records
        if args.limit_per_canonical is not None:
            selected = selected[: args.limit_per_canonical]
        gallery_dir = output_dir / f"{canonical_id}_gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for ordinal, source_record in enumerate(selected, start=1):
            row = int(source_record["bank_row"])
            if str(bank["metadata_canonical_id"][row]) != canonical_id:
                raise RuntimeError(f"Canonical mismatch at bank row {row}")
            set_bank_row(env, bank, row)
            primary = render(env, primary_camera)
            side = render(env, side_camera)
            initial = side_by_side(primary, side)
            image_path = gallery_dir / f"{ordinal:04d}.png"
            imageio.imwrite(image_path, initial)

            video_path = gallery_dir / f"{ordinal:04d}.mp4"
            frames = [initial]
            for control_step in range(max(0, args.video_frames - 1)):
                advance_control_step(env, bank, row, control_step)
                frame = side_by_side(render(env, primary_camera), render(env, side_camera))
                frames.append(frame)
            write_mp4(video_path, frames, args.fps)
            records.append(
                {
                    "ordinal": ordinal,
                    "bank_row": row,
                    "state_id": int(bank["metadata_state_id"][row]),
                    "split": str(bank["metadata_split"][row]),
                    "max_penetration_mm": float(
                        bank["labels_baseline_max_penetration_m"][row] * 1000.0
                    ),
                    "image": str(image_path.relative_to(output_dir)),
                    "video": str(video_path.relative_to(output_dir)),
                    "image_pixel_std": float(initial.std()),
                }
            )
            print(f"REALMESH_PROGRESS={canonical_id}:{ordinal}/{len(selected)}", flush=True)
        output_manifest["canonical_galleries"][canonical_id] = records
        write_canonical_index(gallery_dir, canonical_id, records)
    env.close()

    all_records = [
        record
        for records in output_manifest["canonical_galleries"].values()
        for record in records
    ]
    output_manifest["status"] = "pass" if all(
        float(record["image_pixel_std"]) > 2.0 for record in all_records
    ) else "fail"
    output_manifest["total_samples"] = len(all_records)
    (output_dir / "realmesh_gallery_manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    links = "".join(
        f'<li><a href="{canonical_id}_gallery/index.html">{canonical_id}</a> ({len(records)} 条)</li>'
        for canonical_id, records in output_manifest["canonical_galleries"].items()
    )
    root_page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>O0 realmesh gallery</title><style>body{{max-width:860px;margin:40px auto;padding:0 20px;font:16px/1.65 system-ui,sans-serif;color:#17212b}}a{{color:#1769aa}}.warn{{padding:14px;background:#fff3cd;border-left:4px solid #d29c18}}</style></head><body><h1>O0 Dex5 + 笔 真实网格检查</h1><p>与 PhysX 诊断图库使用完全相同的 bank row。静态图严格重现数据银的手关节和笔位姿。</p><p class="warn">视频是 MuJoCo sim2sim 短时预览，用于视觉排查，不替代 PhysX R0-R3 验收。</p><ul>{links}</ul><p>bank SHA-256: <code>{bank_hash}</code></p></body></html>"""
    (output_dir / "index.html").write_text(root_page, encoding="utf-8")
    print(json.dumps({"status": output_manifest["status"], "samples": len(all_records), "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
