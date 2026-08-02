from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
P3 = REPORT / "v9_p3_strict_to_braked_p4_physx_bundle.npz"
ENTRY45 = REPORT / "v9_entry45_to_braked50_relaxed50ms_lowpen197_physx_bundle.npz"
OUTPUT = REPORT / "v9_dual_entry0_45_to_braked50_physx_bundle.npz"
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
    "curriculum_stage_deg",
)


def main() -> None:
    with np.load(P3) as p3_data, np.load(ENTRY45) as entry45_data:
        p3 = {key: np.asarray(p3_data[key]).copy() for key in KEYS}
        entry45 = {key: np.asarray(entry45_data[key]).copy() for key in KEYS}

    # Seven repeats make 197 late-entry rows approximately balance 1315 P3 rows.
    late_repeat = 7
    payload = {
        key: np.concatenate((p3[key], np.repeat(entry45[key], late_repeat, axis=0)), axis=0)
        for key in KEYS
    }
    # Keep the physical phase offset untouched, but expose two exact reset labels
    # so evaluation reports one success rate for each supported entrance.
    payload["curriculum_stage_deg"] = np.concatenate(
        (
            np.zeros(len(p3["source_qpos"]), dtype=np.float32),
            np.full(
                len(entry45["source_qpos"]) * late_repeat, 45.0, dtype=np.float32
            ),
        )
    )
    count = len(payload["source_qpos"])
    payload["source_teacher_actions"] = np.zeros((count, 0, 20), dtype=np.float32)
    payload["dual_entry_label"] = np.concatenate(
        (
            np.zeros(len(p3["source_qpos"]), dtype=np.int64),
            np.full(len(entry45["source_qpos"]) * late_repeat, 45, dtype=np.int64),
        )
    )
    np.savez_compressed(OUTPUT, **payload)
    metadata = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "p3_source": str(P3.relative_to(ROOT)),
        "entry45_source": str(ENTRY45.relative_to(ROOT)),
        "p3_rows": len(p3["source_qpos"]),
        "entry45_unique_rows": len(entry45["source_qpos"]),
        "entry45_repeat": late_repeat,
        "entry45_sampled_rows": len(entry45["source_qpos"]) * late_repeat,
        "total_rows": count,
        "teacher_prefill": 0,
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
