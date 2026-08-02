from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget upstream middle-clear success states to full P5."
    )
    parser.add_argument(
        "--input",
        default=(
            "reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
            "upstream_middle_learned_success_cache_1200.npz"
        ),
    )
    parser.add_argument(
        "--capture-template",
        default=(
            "reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
            "capture_collision_free_cache_seed1503.npz"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/m3_2_30_index_near_surface/p4_p5_curriculum/"
            "upstream_middle_success_to_p5_capture_bundle.npz"
        ),
    )
    args = parser.parse_args()

    with np.load(args.input) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    with np.load(args.capture_template) as template:
        p5_hand = np.asarray(template["target_qpos"], dtype=np.float32)[
            :, :20
        ].mean(axis=0)

    target_qpos = np.asarray(payload["source_qpos"], dtype=np.float32).copy()
    target_qpos[:, :20] = np.asarray(
        payload["source_target_qpos"], dtype=np.float32
    )
    target_qpos[:, :4] = p5_hand[:4]
    target_qpos[:, 16:20] = p5_hand[16:20]
    payload["target_qpos"] = target_qpos
    payload["target_qvel"] = np.zeros(
        (len(target_qpos), 26), dtype=np.float32
    )
    payload["phase_ids"] = np.zeros(len(target_qpos), dtype=np.int64)
    payload["source_teacher_actions"] = np.zeros(
        (len(target_qpos), 0, 20), dtype=np.float32
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    manifest = {
        "input": args.input,
        "capture_template": args.capture_template,
        "output": str(output),
        "rows": len(target_qpos),
        "target_rule": "full P5 hand; preserve each learned upstream pen pose",
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
