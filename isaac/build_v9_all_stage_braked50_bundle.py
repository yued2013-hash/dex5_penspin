from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
OUTPUT = REPORT / "v9_all_stage_to_braked50_physx_bundle.npz"
SOURCES = (
    (0.0, "v9_p3_strict_to_braked_p4_physx_bundle.npz", 3),
    (7.0, "v9_entry7_to_approach20_physx_bundle.npz", 1),
    (20.0, "v9_approach20_to_entry25_physx_bundle.npz", 1),
    (25.0, "v9_entry25_to_mid30_paired109_physx_bundle.npz", 1),
    (30.0, "v9_mid30_to_mid35_physx_bundle.npz", 1),
    (35.0, "v9_mid35_to_approach40_physx_bundle.npz", 1),
    (40.0, "v9_approach40_to_entry45_lowpen136_physx_bundle.npz", 1),
    (45.0, "v9_entry45_to_braked50_relaxed50ms_lowpen197_physx_bundle.npz", 1),
)
KEYS = (
    "source_qpos",
    "source_qvel",
    "source_target_qpos",
    "source_policy_action",
    "target_qpos",
    "target_qvel",
    "phase_ids",
    "source_episode_step",
    "source_phase_offset_rad",
    "target_remaining_rad",
    "policy_target_pen_quat",
    "source_contact_force_n",
)


def main() -> None:
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in KEYS}
    stage_chunks: list[np.ndarray] = []
    metadata_sources: list[dict[str, object]] = []
    for stage, name, repeat in SOURCES:
        path = REPORT / name
        with np.load(path) as data:
            rows = len(data["source_qpos"])
            for key in KEYS:
                chunks[key].append(np.repeat(np.asarray(data[key]), repeat, axis=0))
        sampled_rows = rows * repeat
        stage_chunks.append(np.full(sampled_rows, stage, dtype=np.float32))
        metadata_sources.append(
            {"stage_deg": stage, "file": name, "rows": rows, "repeat": repeat,
             "sampled_rows": sampled_rows}
        )

    payload = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    payload["curriculum_stage_deg"] = np.concatenate(stage_chunks)
    count = len(payload["source_qpos"])
    payload["source_teacher_actions"] = np.zeros((count, 0, 20), dtype=np.float32)
    np.savez_compressed(OUTPUT, **payload)
    metadata = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "total_rows": count,
        "teacher_prefill": 0,
        "sources": metadata_sources,
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
