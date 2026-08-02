#!/usr/bin/env python3
"""Build the guarded 0-2.0 s prefix used for the 9.8 mm pen audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_SHA256 = "44e54a4915cc43faaace932c3c301e6e57e2c8028ec45885a8448e5a72305230"
PREFIX_ROWS = 201


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "trajectory1_slow_100hz.npz"
    source_manifest = json.loads(
        (root / "trajectory1_slow_manifest.json").read_text(encoding="utf-8")
    )
    if sha256(source) != SOURCE_SHA256:
        raise SystemExit("slow diagnostic source hash changed")
    with np.load(source, allow_pickle=False) as data:
        command = np.asarray(data["sdk_absolute_target_100hz"], dtype=np.float64)
        hz = float(data["command_hz"])
    prefix = command[:PREFIX_ROWS].copy()
    output = root / "thin_pen_9p8mm_prefix_2s_100hz.npz"
    np.savez_compressed(
        output,
        sdk_absolute_target_100hz=prefix,
        trajectory_offsets=np.asarray([0, len(prefix)], dtype=np.int64),
        command_hz=np.asarray(hz, dtype=np.float64),
    )
    delta = np.diff(prefix, axis=0)
    manifest = {
        "schema": "dex5-s0.9-slow-diagnostic-v1",
        "status": "OFFLINE_PREFIX_NOT_EXECUTED",
        "output": str(output),
        "output_sha256": sha256(output),
        "source": str(source),
        "source_sha256": SOURCE_SHA256,
        "source_trajectory_id": 1,
        "mapping_sha256": source_manifest["mapping_sha256"],
        "trajectory_count": 1,
        "command_rows": int(len(prefix)),
        "command_hz": hz,
        "duration_s": float((len(prefix) - 1) / hz),
        "maximum_delta_rad": float(np.max(np.abs(delta))),
        "maximum_speed_rad_s": float(np.max(np.abs(delta)) * hz),
        "physical_condition": {"pen_middle_diameter_mm": 9.8},
        "boundary": (
            "Stops at 2.0 s, before both prior Roll_31R failures. Completion "
            "does not authorize the remaining trajectory."
        ),
    }
    (root / "thin_pen_9p8mm_prefix_2s_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
