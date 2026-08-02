from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Merge and validate reset-free v9 teacher shards.")
parser.add_argument("--inputs", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--limit", type=int, default=200)
args = parser.parse_args()


def main() -> None:
    shards = [np.load(path) for path in args.inputs]
    try:
        stage_names = shards[0]["stage_names"]
        control_dt_s = float(shards[0]["control_dt_s"])
        for shard in shards:
            if not bool(shard["continuous_without_intermediate_reset"]):
                raise ValueError("A shard is not marked reset-free")
            if not np.array_equal(shard["stage_names"], stage_names):
                raise ValueError("Stage definitions differ across shards")
            if not np.isclose(float(shard["control_dt_s"]), control_dt_s):
                raise ValueError("Control time steps differ across shards")

        counts = [len(shard["lengths"]) for shard in shards]
        available = sum(counts)

        # Select by initial-state identity before applying the limit. This prevents
        # repeated reset batches from inflating the teacher/cache count.
        selected_rows: list[tuple[int, int]] = []
        seen_initial_states: set[bytes] = set()
        for shard_id, shard in enumerate(shards):
            rounded = np.round(shard["initial_qpos"], decimals=6)
            for row_id, initial_state in enumerate(rounded):
                key = initial_state.tobytes()
                if key in seen_initial_states:
                    continue
                seen_initial_states.add(key)
                if len(selected_rows) < args.limit:
                    selected_rows.append((shard_id, row_id))
        if len(selected_rows) < args.limit:
            raise ValueError(
                f"Only {len(selected_rows)} unique trajectories are available "
                f"from {available} rows; need {args.limit}"
            )

        trajectory_keys = [
            key
            for key in shards[0].files
            if key
            not in {
                "lengths",
                "initial_qpos",
                "initial_qvel",
                "initial_bank_ids",
                "reset_seeds",
                "control_dt_s",
                "stage_names",
                "continuous_without_intermediate_reset",
            }
        ]
        lengths = np.asarray(
            [shards[shard_id]["lengths"][row_id] for shard_id, row_id in selected_rows],
            dtype=np.int64,
        )
        initial_qpos = np.stack(
            [shards[shard_id]["initial_qpos"][row_id] for shard_id, row_id in selected_rows]
        )
        initial_qvel = np.stack(
            [shards[shard_id]["initial_qvel"][row_id] for shard_id, row_id in selected_rows]
        )
        max_length = int(lengths.max())
        payload: dict[str, np.ndarray] = {
            "lengths": lengths,
            "initial_qpos": initial_qpos,
            "initial_qvel": initial_qvel,
            "control_dt_s": np.asarray(control_dt_s, dtype=np.float32),
            "stage_names": stage_names,
            "continuous_without_intermediate_reset": np.asarray(True),
        }
        for metadata_key in ("initial_bank_ids", "reset_seeds"):
            if all(metadata_key in shard.files for shard in shards):
                payload[metadata_key] = np.asarray(
                    [
                        shards[shard_id][metadata_key][row_id]
                        for shard_id, row_id in selected_rows
                    ],
                    dtype=np.int64,
                )
        for key in trajectory_keys:
            sample = shards[0][key]
            merged = np.zeros((args.limit, max_length, *sample.shape[2:]), dtype=sample.dtype)
            for output_row, (shard_id, shard_row) in enumerate(selected_rows):
                length = lengths[output_row]
                merged[output_row, :length] = shards[shard_id][key][shard_row, :length]
            payload[key] = merged

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **payload)

        final_stage = len(stage_names) - 1
        hold_steps = 18
        checks = {
            "stage_sequence_monotonic": [],
            "reaches_final_stage": [],
            "final_phase_50_60_deg": [],
            "final_omega_le_12_rad_s": [],
            "final_triad_contact": [],
            "max_penetration_le_0_5_mm": [],
        }
        terminal_phase = []
        terminal_omega = []
        max_penetration_mm = []
        for row, length in enumerate(lengths):
            stages = payload["stage_id"][row, :length]
            phase = payload["phase_deg"][row, :length]
            omega = payload["angular_speed_rad_s"][row, :length]
            force = payload["contact_force_n"][row, :length]
            penetration = payload["max_penetration_m"][row, :length]
            final_slice = slice(length - hold_steps, length)
            checks["stage_sequence_monotonic"].append(bool(np.all(np.diff(stages) >= 0)))
            checks["reaches_final_stage"].append(bool(stages[-1] == final_stage))
            checks["final_phase_50_60_deg"].append(
                bool(np.all((phase[final_slice] >= 50.0) & (phase[final_slice] <= 60.0)))
            )
            checks["final_omega_le_12_rad_s"].append(bool(np.all(omega[final_slice] <= 12.0)))
            checks["final_triad_contact"].append(
                bool(np.all(force[final_slice, :3] > 0.03))
            )
            checks["max_penetration_le_0_5_mm"].append(bool(np.max(penetration) <= 0.0005))
            terminal_phase.append(float(phase[-1]))
            terminal_omega.append(float(omega[-1]))
            max_penetration_mm.append(float(np.max(penetration) * 1000.0))

        rounded_initial = np.round(initial_qpos, decimals=6)
        unique_initial_states = len(np.unique(rounded_initial, axis=0))
        checks["unique_initial_state"] = [unique_initial_states == args.limit]
        report = {
            "output": str(args.output),
            "continuous_without_intermediate_reset": True,
            "input_shards": [str(path) for path in args.inputs],
            "input_counts": counts,
            "available_rows": int(available),
            "available_unique_initial_states_at_1e_6": int(len(seen_initial_states)),
            "trajectory_count": int(args.limit),
            "unique_initial_states_at_1e_6": int(unique_initial_states),
            "trajectory_length_steps": {
                "min": int(lengths.min()),
                "mean": float(lengths.mean()),
                "max": int(lengths.max()),
            },
            "terminal_phase_deg": {
                "min": float(np.min(terminal_phase)),
                "mean": float(np.mean(terminal_phase)),
                "max": float(np.max(terminal_phase)),
            },
            "terminal_omega_rad_s": {
                "min": float(np.min(terminal_omega)),
                "mean": float(np.mean(terminal_omega)),
                "max": float(np.max(terminal_omega)),
            },
            "max_penetration_mm": {
                "min": float(np.min(max_penetration_mm)),
                "mean": float(np.mean(max_penetration_mm)),
                "max": float(np.max(max_penetration_mm)),
            },
            "checks": {
                name: {"passed": int(sum(values)), "total": len(values)}
                for name, values in checks.items()
            },
            "all_checks_pass": bool(all(all(values) for values in checks.values())),
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        for shard in shards:
            shard.close()


if __name__ == "__main__":
    main()
