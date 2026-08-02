from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one progressive triad-to-P1 strict reset bundle."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--blend", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--active_output",
        default=(
            "reports/m3_2_30_index_near_surface/p5_p1_curriculum/"
            "p5_p1_strict_active_bundle.npz"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "reports/m3_2_30_index_near_surface/"
            "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("--blend must be in [0, 1]")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    p1 = next(row for row in manifest["phases"] if int(row["phase"]) == 1)
    p1_hand = np.asarray(p1["hand_qpos"], dtype=np.float32)
    with np.load(args.source) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}

    source_qpos = np.asarray(payload["source_qpos"], dtype=np.float32)
    count = len(source_qpos)
    target_qpos = source_qpos.copy()
    target_qpos[:, :12] = (
        (1.0 - args.blend) * source_qpos[:, :12]
        + args.blend * p1_hand[:12]
    )
    payload["target_qpos"] = target_qpos
    payload["target_qvel"] = np.zeros((count, 26), dtype=np.float32)
    payload["phase_ids"] = np.full(count, 4, dtype=np.int64)
    payload["source_teacher_actions"] = np.zeros(
        (count, 0, 20), dtype=np.float32
    )

    output = Path(args.output)
    active_output = Path(args.active_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    active_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    np.savez_compressed(active_output, **payload)

    start_rmse = np.sqrt(
        np.mean(
            (source_qpos[:, :12] - p1_hand[None, :12]) ** 2,
            axis=1,
        )
    )
    target_rmse = np.sqrt(
        np.mean(
            (target_qpos[:, :12] - p1_hand[None, :12]) ** 2,
            axis=1,
        )
    )
    report = {
        "source": args.source,
        "blend": args.blend,
        "output": str(output),
        "active_output": str(active_output),
        "rows": count,
        "canonical_p1_rmse_before_rad": {
            "mean": float(start_rmse.mean()),
            "max": float(start_rmse.max()),
        },
        "canonical_p1_rmse_target_rad": {
            "mean": float(target_rmse.mean()),
            "max": float(target_rmse.max()),
        },
        "target_semantics": (
            "blend=0 表示与真实笔位姿兼容的 projected P1 三指局部目标；"
            "blend>0 时再渐进靠近旧 canonical P1 关节角。始终保留真实"
            "笔位姿及无名指、小指状态。"
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
