from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "reports/m3_2_30_index_near_surface/"
    "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
    "v9_p1_p2_initial_bundle.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v9 P1/P2 canonical states for PhysX projection.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-per-phase", type=int, default=16)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    phases = {row["phase"]: row for row in manifest["phases"]}
    qpos_rows = []
    qvel_rows = []
    target_rows = []
    phase_ids = []
    for phase_id in (1, 2):
        row = phases[phase_id]
        hand = np.asarray(row["hand_qpos"], dtype=np.float32)
        pen = np.concatenate(
            (np.asarray(row["pen_pos"], dtype=np.float32), np.asarray(row["pen_quat"], dtype=np.float32))
        )
        qpos = np.concatenate((hand, pen))
        qvel = np.concatenate((np.zeros(20, dtype=np.float32), np.asarray(row["pen_qvel"], dtype=np.float32)))
        for _ in range(args.rows_per_phase):
            qpos_rows.append(qpos)
            qvel_rows.append(qvel)
            target_rows.append(hand)
            phase_ids.append(phase_id)

    source_qpos = np.stack(qpos_rows)
    source_qvel = np.stack(qvel_rows)
    source_target = np.stack(target_rows)
    count = len(source_qpos)
    payload = {
        "source_qpos": source_qpos,
        "source_qvel": source_qvel,
        "source_target_qpos": source_target,
        "source_policy_action": np.zeros((count, 20), dtype=np.float32),
        "source_teacher_actions": np.zeros((count, 1, 20), dtype=np.float32),
        "target_qpos": source_qpos.copy(),
        "target_qvel": np.zeros_like(source_qvel),
        "phase_ids": np.asarray(phase_ids, dtype=np.int64),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    report = {
        "输出": str(output.relative_to(ROOT)),
        "来源": str(MANIFEST.relative_to(ROOT)),
        "每个阶段行数": args.rows_per_phase,
        "总行数": count,
        "P1要求接触": ["thumb", "index", "middle"],
        "P2要求接触": ["thumb", "index", "middle", "ring"],
        "P2主承力": ["thumb", "index", "middle"],
        "P2辅助": ["ring"],
        "禁止主承力": ["pinky"],
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
