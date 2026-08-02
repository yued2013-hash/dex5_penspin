from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
DEFAULT_SOURCE = REPORT / "v9_dagger6_oracle861_dual_formal4096.npz"
DEFAULT_MANIFEST = REPORT / "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
DEFAULT_OUTPUT = REPORT / "v9_p4_to_p5_physx_bundle.npz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a PhysX P4-to-P5 reset bank from successful v9 braked-P4 trajectories."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=2048)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    p5 = next(row for row in manifest["phases"] if int(row["phase"]) == 5)
    p5_hand = np.asarray(p5["hand_qpos"], dtype=np.float32)

    with np.load(args.source) as data:
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        terminal_ids = np.maximum(lengths - 1, 0)
        row_ids = np.arange(len(lengths))
        qpos = np.asarray(data["qpos"][row_ids, terminal_ids], dtype=np.float32)
        qvel = np.asarray(data["qvel"][row_ids, terminal_ids], dtype=np.float32)
        processed_target = np.asarray(
            data["processed_target"][row_ids, terminal_ids], dtype=np.float32
        )
        action = np.asarray(data["action"][row_ids, terminal_ids], dtype=np.float32)
        contact = np.asarray(
            data["contact_force_n"][row_ids, terminal_ids], dtype=np.float32
        )
        phase = np.asarray(data["phase_deg"][row_ids, terminal_ids], dtype=np.float32)
        speed = np.asarray(
            data["angular_speed_rad_s"][row_ids, terminal_ids], dtype=np.float32
        )
        penetration = np.asarray(
            data["max_penetration_m"][row_ids, terminal_ids], dtype=np.float32
        )
        entry_labels = np.asarray(data["entry_labels"], dtype=np.int64)

    valid = (
        (phase >= 50.0)
        & (phase <= 60.0)
        & (speed <= 12.0)
        & np.all(contact[:, :3] > 0.03, axis=1)
        & (penetration <= 0.0005)
        & np.isfinite(qpos).all(axis=1)
        & np.isfinite(qvel).all(axis=1)
    )
    selected = np.flatnonzero(valid)
    if len(selected) == 0:
        raise RuntimeError("No terminal braked-P4 states passed the P4-to-P5 reset gate")
    if args.limit > 0 and len(selected) > args.limit:
        # Deterministic coverage across the complete formal rollout instead of taking
        # one contiguous batch.
        positions = np.linspace(0, len(selected) - 1, args.limit, dtype=np.int64)
        selected = selected[positions]

    source_qpos = qpos[selected]
    source_qvel = qvel[selected]
    source_target = processed_target[selected]
    target_qpos = source_qpos.copy()
    # Thumb/index keep the physically reached P4 pinch. Middle/ring/pinky retreat
    # toward the canonical P5 rear-side posture while the pen target stays at the
    # actual PhysX terminal pose.
    target_qpos[:, 4:16] = p5_hand[4:16]
    target_qvel = np.zeros_like(source_qvel)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source_qpos=source_qpos,
        source_qvel=source_qvel,
        source_target_qpos=source_target,
        source_policy_action=action[selected],
        source_teacher_actions=np.zeros((len(selected), 0, 20), dtype=np.float32),
        target_qpos=target_qpos,
        target_qvel=target_qvel,
        phase_ids=np.zeros(len(selected), dtype=np.int64),
        source_contact_force_n=contact[selected],
        source_phase_deg=phase[selected],
        source_angular_speed_rad_s=speed[selected],
        source_max_penetration_m=penetration[selected],
        source_entry_labels=entry_labels[selected],
        source_trajectory_ids=selected.astype(np.int64),
    )
    report = {
        "source": str(args.source.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output": str(output),
        "source_trajectory_count": int(len(lengths)),
        "valid_terminal_count": int(valid.sum()),
        "exported_count": int(len(selected)),
        "phase_deg": {
            "min": float(phase[selected].min()),
            "mean": float(phase[selected].mean()),
            "max": float(phase[selected].max()),
        },
        "angular_speed_rad_s": {
            "min": float(speed[selected].min()),
            "mean": float(speed[selected].mean()),
            "max": float(speed[selected].max()),
        },
        "max_penetration_mm": {
            "min": float(penetration[selected].min() * 1000.0),
            "mean": float(penetration[selected].mean() * 1000.0),
            "max": float(penetration[selected].max() * 1000.0),
        },
        "target_semantics": (
            "Keep each PhysX P4 terminal pen pose and thumb/index joint target; "
            "move middle/ring/pinky toward canonical P5 rear-side posture."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
