from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from canonical_reset_schema import sha256_file, write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GALLERY = PROJECT_ROOT / (
    "reports/m3_2_30_index_near_surface/canonical_reset_distribution_o0/visual_gallery"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit O0 visual-gallery artifacts.")
    parser.add_argument("--gallery", default=str(DEFAULT_GALLERY))
    args = parser.parse_args()
    gallery = Path(args.gallery).resolve()
    manifest_path = gallery / "visual_gallery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_path = Path(manifest["bank"])
    with np.load(bank_path, allow_pickle=False) as loaded:
        bank = {key: loaded[key] for key in loaded.files}

    failures: list[str] = []
    canonical_rows: dict[str, dict[str, object]] = {}
    seen_bank_rows: set[int] = set()
    total = 0
    canonical_ids = tuple(sorted(manifest["canonical_galleries"]))
    expected_per_canonical = int(manifest["samples_per_canonical"])
    expected_total = expected_per_canonical * len(canonical_ids)
    expected_size = (
        int(manifest.get("width", 960)),
        int(manifest.get("height", 720)),
    )
    for canonical_id in canonical_ids:
        records = manifest["canonical_galleries"].get(canonical_id, [])
        if len(records) != expected_per_canonical:
            failures.append(f"{canonical_id}: sample count {len(records)}")
        contact_changes = 0
        split_counts = {"train": 0, "validation": 0, "test": 0}
        for record in records:
            total += 1
            bank_row = int(record["bank_row"])
            if bank_row in seen_bank_rows:
                failures.append(f"duplicate bank row {bank_row}")
            seen_bank_rows.add(bank_row)
            if str(bank["metadata_canonical_id"][bank_row]) != canonical_id:
                failures.append(f"{canonical_id}: bank-row label mismatch at {bank_row}")
            if int(bank["metadata_state_id"][bank_row]) != int(record["state_id"]):
                failures.append(f"{canonical_id}: state-id mismatch at {bank_row}")
            split_counts[str(record["split"])] += 1
            if record["contact_state"] != record["reinstantiated_contact_state"]:
                contact_changes += 1

            screenshot = gallery / record["screenshot"]
            skeleton = gallery / record["skeleton"]
            video = gallery / record["video"]
            for path in (screenshot, skeleton, video):
                if not path.is_file() or path.stat().st_size == 0:
                    failures.append(f"missing or empty: {path}")
            if screenshot.is_file():
                with Image.open(screenshot) as image:
                    if image.size != expected_size:
                        failures.append(f"bad screenshot size: {screenshot} {image.size}")
                    if float(np.asarray(image).std()) <= 2.0:
                        failures.append(f"blank screenshot: {screenshot}")
            if skeleton.is_file():
                with Image.open(skeleton) as image:
                    if min(image.size) < 600:
                        failures.append(f"undersized skeleton: {skeleton} {image.size}")
            if video.is_file():
                capture = cv2.VideoCapture(str(video))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                readable, frame = capture.read()
                capture.release()
                if frame_count != int(record["video_frame_count"]):
                    failures.append(f"bad video frame count: {video} {frame_count}")
                if (width, height) != expected_size:
                    failures.append(f"bad video size: {video} {(width, height)}")
                if not readable or frame is None or float(frame.std()) <= 2.0:
                    failures.append(f"unreadable/blank video: {video}")
        canonical_rows[canonical_id] = {
            "samples": len(records),
            "contact_state_changed_after_reinstantiation": contact_changes,
            "split_counts": split_counts,
        }

    gates = {
        "manifest_status_pass": manifest["status"] == "pass",
        "bank_hash_matches": sha256_file(bank_path) == manifest["bank_sha256"],
        "expected_total_samples": total == expected_total,
        "unique_bank_rows": len(seen_bank_rows) == expected_total,
        "early_termination_zero": manifest["early_termination_count"] == 0,
        "blank_screenshot_zero": manifest["blank_screenshot_count"] == 0,
        "media_failures_zero": len(failures) == 0,
    }
    report = {
        "status": "pass" if all(gates.values()) else "fail",
        "render_backend": manifest["render_backend"],
        "rtx_mesh_rendered": manifest["rtx_mesh_rendered"],
        "gates": gates,
        "canonical_summary": canonical_rows,
        "failures": failures,
        "gallery": str(gallery),
        "manifest": str(manifest_path),
        "bank": str(bank_path),
        "bank_sha256": manifest["bank_sha256"],
    }
    write_json_atomic(gallery / "visual_gallery_audit.json", report)
    lines = [
        "# O0 Canonical Visual Gallery Audit",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Backend: `{report['render_backend']}`",
        f"- RTX mesh rendered: `{str(report['rtx_mesh_rendered']).lower()}`",
        f"- Samples: `{total}`",
        f"- Media failures: `{len(failures)}`",
        "",
        "| Canonical | Samples | Contact changed after reinstantiation | train/validation/test |",
        "|---|---:|---:|---:|",
    ]
    for canonical_id, row in canonical_rows.items():
        split = row["split_counts"]
        lines.append(
            f"| {canonical_id} | {row['samples']} | "
            f"{row['contact_state_changed_after_reinstantiation']} | "
            f"{split['train']}/{split['validation']}/{split['test']} |"
        )
    lines.extend(
        (
            "",
            "The software backend projects actual Isaac PhysX body states and pen poses.",
            "It is not an RTX mesh screenshot. Use the RTX wrapper on a compatible 5090 host",
            "to add mesh-level visual evidence without changing the sampled bank rows.",
            "",
        )
    )
    (gallery / "visual_gallery_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
