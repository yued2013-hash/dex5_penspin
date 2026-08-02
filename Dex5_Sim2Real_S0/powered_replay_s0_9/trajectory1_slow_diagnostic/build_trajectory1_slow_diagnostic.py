#!/usr/bin/env python3
"""Build a separately versioned, speed-limited diagnostic for S0.9 trajectory 1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_TRAJECTORY_ID = 1
COMMAND_HZ = 100.0
MAX_SPEED_RAD_S = 0.03
MAX_DELTA_RAD = MAX_SPEED_RAD_S / COMMAND_HZ
EXPECTED_SOURCE_SHA256 = (
    "dfcc88f356fe63815b527bde4b38db0a948e3a7a8444fbcc2942989565725108"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def densify(segment: np.ndarray) -> np.ndarray:
    rows = [segment[0].astype(np.float64)]
    for target in segment[1:]:
        start = rows[-1]
        steps = max(1, int(np.ceil(np.max(np.abs(target - start)) / MAX_DELTA_RAD)))
        for alpha in np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[1:]:
            rows.append(start + alpha * (target - start))
    output = np.asarray(rows, dtype=np.float64)
    output[0] = segment[0]
    output[-1] = segment[-1]
    return output


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    source = output_dir.parent / "mapped_replay_100hz.npz"
    source_manifest_path = output_dir.parent / "mapped_replay_manifest.json"
    output = output_dir / "trajectory1_slow_100hz.npz"
    manifest_path = output_dir / "trajectory1_slow_manifest.json"

    source_hash = sha256(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"frozen source hash changed: {source_hash}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as data:
        command = np.asarray(data["sdk_absolute_target_100hz"], dtype=np.float64)
        offsets = np.asarray(data["trajectory_offsets"], dtype=np.int64)
        source_hz = float(data["command_hz"])
    if source_hz != COMMAND_HZ:
        raise SystemExit(f"unexpected source command rate: {source_hz}")

    source_segment = command[
        offsets[SOURCE_TRAJECTORY_ID] : offsets[SOURCE_TRAJECTORY_ID + 1]
    ]
    slow = densify(source_segment)
    delta = np.diff(slow, axis=0)
    maximum_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    maximum_speed = maximum_delta * COMMAND_HZ
    if maximum_delta > MAX_DELTA_RAD + 1.0e-12:
        raise SystemExit(f"generated command exceeds diagnostic delta cap: {maximum_delta}")
    if not np.array_equal(slow[0], source_segment[0]) or not np.array_equal(
        slow[-1], source_segment[-1]
    ):
        raise SystemExit("diagnostic interpolation changed a trajectory endpoint")

    np.savez_compressed(
        output,
        sdk_absolute_target_100hz=slow,
        trajectory_offsets=np.asarray([0, len(slow)], dtype=np.int64),
        command_hz=np.asarray(COMMAND_HZ, dtype=np.float64),
    )
    output_hash = sha256(output)
    manifest = {
        "schema": "dex5-s0.9-slow-diagnostic-v1",
        "status": "OFFLINE_DIAGNOSTIC_NOT_EXECUTED",
        "output": str(output),
        "output_sha256": output_hash,
        "source": str(source),
        "source_sha256": source_hash,
        "source_trajectory_id": SOURCE_TRAJECTORY_ID,
        "mapping_sha256": source_manifest["mapping_sha256"],
        "trajectory_count": 1,
        "source_command_rows": int(len(source_segment)),
        "command_rows": int(len(slow)),
        "command_hz": COMMAND_HZ,
        "maximum_delta_rad": maximum_delta,
        "maximum_speed_rad_s": maximum_speed,
        "endpoint_preservation": True,
        "required_replay_control_mode": "low_gain_hybrid",
        "required_feedback_speed_abort_rad_s": 0.12,
        "required_feedback_speed_abort_windows": 1,
        "boundary": (
            "This is a separately versioned low-speed hardware diagnostic for source "
            "trajectory 1. It does not replace the frozen replay package and completion "
            "does not establish rotation-task success or dynamic equivalence."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
