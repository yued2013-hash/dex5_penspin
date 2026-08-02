from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = (
    "reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_p4_cascade1024_v20_supported45_report_braked_P4_entry_bundle.npz",
    "reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_braked_p4_teacher200_diverse_report_braked_P4_entry_bundle.npz",
    "reports/m3_2_30_index_near_surface/"
    "v9_continuous_p3_braked_p4_teacher200_diverse_part2_report_braked_P4_entry_bundle.npz",
)
DEFAULT_CAPTURE = (
    "reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "capture_collision_free_cache_seed1503.npz"
)
DEFAULT_OUTPUT = (
    "reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "p3_p4_real_terminal_to_p5_capture_bundle.npz"
)
DEFAULT_MIDDLE_OUTPUT = (
    "reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
    "p3_p4_real_terminal_to_middle_clear_bundle.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a P5 capture reset bundle from real P3-to-P4 terminal states."
    )
    parser.add_argument("--input", nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument("--capture-template", default=DEFAULT_CAPTURE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--middle-output", default=DEFAULT_MIDDLE_OUTPUT)
    args = parser.parse_args()

    sources: dict[str, list[np.ndarray]] = {
        "source_qpos": [],
        "source_qvel": [],
        "source_target_qpos": [],
        "source_policy_action": [],
        "source_contact_force_n": [],
    }
    source_file_ids: list[np.ndarray] = []
    source_row_ids: list[np.ndarray] = []
    for file_id, path_string in enumerate(args.input):
        path = Path(path_string)
        with np.load(path) as data:
            count = len(data["source_qpos"])
            for key in sources:
                if key in data:
                    values = np.asarray(data[key], dtype=np.float32)
                elif key == "source_policy_action":
                    values = np.zeros((count, 20), dtype=np.float32)
                else:
                    values = np.zeros((count, 5), dtype=np.float32)
                sources[key].append(values)
            source_file_ids.append(
                np.full(count, file_id, dtype=np.int64)
            )
            source_row_ids.append(np.arange(count, dtype=np.int64))

    payload = {key: np.concatenate(rows) for key, rows in sources.items()}
    count = len(payload["source_qpos"])
    with np.load(args.capture_template) as capture:
        template_target = np.asarray(capture["target_qpos"], dtype=np.float32)
        target_hand = template_target[:, :20].mean(axis=0)

    target_qpos = payload["source_qpos"].copy()
    target_qpos[:, :20] = target_hand
    target_qvel = np.zeros((count, 26), dtype=np.float32)
    payload.update(
        {
            "target_qpos": target_qpos,
            "target_qvel": target_qvel,
            # Capture tasks filter phase 0; the P5 stage is set by the task cfg.
            "phase_ids": np.zeros(count, dtype=np.int64),
            "source_teacher_actions": np.zeros(
                (count, 0, 20), dtype=np.float32
            ),
            "source_file_ids": np.concatenate(source_file_ids),
            "source_row_ids": np.concatenate(source_row_ids),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    manifest = {
        "output": str(output),
        "rows": count,
        "inputs": args.input,
        "rows_per_input": [len(rows) for rows in sources["source_qpos"]],
        "capture_template": args.capture_template,
        "target_pen_rule": "preserve each upstream terminal pen pose",
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    middle_payload = dict(payload)
    middle_target_qpos = payload["source_qpos"].copy()
    middle_target_qpos[:, :20] = payload["source_target_qpos"]
    middle_target_qpos[:, 4:8] = target_hand[4:8]
    middle_payload["target_qpos"] = middle_target_qpos
    middle_output = Path(args.middle_output)
    middle_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(middle_output, **middle_payload)
    middle_manifest = dict(manifest)
    middle_manifest.update(
        {
            "output": str(middle_output),
            "target_hand_rule": (
                "preserve upstream support targets; replace middle with P5 retract target"
            ),
        }
    )
    middle_output.with_suffix(".json").write_text(
        json.dumps(middle_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
