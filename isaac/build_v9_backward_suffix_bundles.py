from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
SOURCES = (
    (0, "v9_p3_strict_to_braked_p4_physx_bundle.npz"),
    (7, "v9_entry7_to_approach20_physx_bundle.npz"),
    (20, "v9_approach20_to_entry25_physx_bundle.npz"),
    (25, "v9_entry25_to_mid30_paired109_physx_bundle.npz"),
    (30, "v9_mid30_to_mid35_physx_bundle.npz"),
    (35, "v9_mid35_to_approach40_physx_bundle.npz"),
    (40, "v9_approach40_to_entry45_lowpen136_physx_bundle.npz"),
    (45, "v9_entry45_to_braked50_relaxed50ms_lowpen197_physx_bundle.npz"),
)
SUFFIX_STARTS = (40, 35, 30, 25, 20, 7, 0)
FINAL_SOURCE = REPORT / "v9_entry45_to_braked50_relaxed50ms_lowpen197_physx_bundle.npz"
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


def select_evenly(values: np.ndarray, count: int = 200) -> np.ndarray:
    if len(values) <= count:
        return values.copy()
    ids = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[ids]


def main() -> None:
    with np.load(FINAL_SOURCE) as data:
        final_qpos = np.asarray(data["target_qpos"]).copy()
        final_qvel = np.asarray(data["target_qvel"]).copy()

    loaded: dict[int, dict[str, np.ndarray]] = {}
    for stage, name in SOURCES:
        with np.load(REPORT / name) as data:
            loaded[stage] = {key: select_evenly(np.asarray(data[key])) for key in KEYS}

    outputs = []
    for start in SUFFIX_STARTS:
        stages = [stage for stage, _ in SOURCES if stage >= start]
        chunks: dict[str, list[np.ndarray]] = {key: [] for key in KEYS}
        stage_chunks = []
        source_rows = []
        final_cursor = 0
        for stage in stages:
            chunk = loaded[stage]
            rows = len(chunk["source_qpos"])
            ids = np.arange(final_cursor, final_cursor + rows) % len(final_qpos)
            final_cursor += rows
            for key in KEYS:
                if key == "target_qpos":
                    values = final_qpos[ids]
                elif key == "target_qvel":
                    values = final_qvel[ids]
                elif key == "policy_target_pen_quat":
                    values = final_qpos[ids, 23:27]
                else:
                    values = chunk[key]
                chunks[key].append(values)
            stage_chunks.append(np.full(rows, float(stage), dtype=np.float32))
            source_rows.append({"stage_deg": stage, "rows": rows})

        payload = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
        payload["curriculum_stage_deg"] = np.concatenate(stage_chunks)
        count = len(payload["source_qpos"])
        payload["source_teacher_actions"] = np.zeros((count, 0, 20), dtype=np.float32)
        output = REPORT / f"v9_backward_suffix{start}_to_braked50_physx_bundle.npz"
        np.savez_compressed(output, **payload)
        metadata = {
            "output": str(output.relative_to(ROOT)),
            "start_stage_deg": start,
            "stages": stages,
            "total_rows": count,
            "final_target_source": str(FINAL_SOURCE.relative_to(ROOT)),
            "sources": source_rows,
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        outputs.append(metadata)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
