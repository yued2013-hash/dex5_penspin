from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = (
    ROOT
    / "reports/manus_mujoco_presets/canonical_p3_manus_constrained_m3_2_40/"
    "canonical_P3_candidate_seed.json"
)
DEFAULT_OUTPUT = ROOT / "reports/m3_2_30_index_near_surface/isaac_canonical_p3_bundle.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the accepted canonical P3 as a PhysX reset bundle.")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()

    record = json.loads(args.seed.read_text(encoding="utf-8"))
    hand_qpos = np.asarray(record["hand_qpos"], dtype=np.float32)
    servo_target = np.asarray(record["target_qpos"], dtype=np.float32)
    pen_pose = np.concatenate(
        (
            np.asarray(record["pen_pos"], dtype=np.float32),
            np.asarray(record["pen_quat"], dtype=np.float32),
        )
    )
    canonical_qpos = np.concatenate((hand_qpos, pen_pose))
    source_qpos = np.repeat(canonical_qpos[None, :], args.count, axis=0)
    source_qvel = np.zeros((args.count, 26), dtype=np.float32)
    source_target = np.repeat(servo_target[None, :], args.count, axis=0)
    target_qpos = source_qpos.copy()

    bundle = {
        "source_qpos": source_qpos,
        "source_qvel": source_qvel,
        "source_target_qpos": source_target,
        "source_policy_action": np.zeros((args.count, 20), dtype=np.float32),
        "source_teacher_actions": np.zeros((args.count, 1, 20), dtype=np.float32),
        "target_qpos": target_qpos,
        "target_qvel": source_qvel.copy(),
        "phase_ids": np.full(args.count, 3, dtype=np.int64),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **bundle)
    metadata = {
        "stage": "canonical P3 PhysX reset bundle",
        "source": str(args.seed.relative_to(ROOT)),
        "output": str(args.output.relative_to(ROOT)),
        "count": args.count,
        "required_contacts": ["thumb", "index", "middle"],
        "manus_rms_rad": record["validation"]["max_hand_rms_rad"],
        "mujoco_hold_ms": record["validation"]["triad_hold_ms"],
        "note": "Repeated deterministic seed; DirectRLEnv reset noise supplies the initial curriculum distribution.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
