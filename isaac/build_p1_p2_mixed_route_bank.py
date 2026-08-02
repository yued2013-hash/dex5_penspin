from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Mix P1 route resets with local arrival resets.")
parser.add_argument("--p1-bank", required=True, type=Path)
parser.add_argument("--p1-ids", required=True, type=Path)
parser.add_argument("--arrival-bank", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--p1-copies", type=int, default=512)
args = parser.parse_args()

keys = (
    "source_qpos",
    "source_qvel",
    "source_target_qpos",
    "source_policy_action",
    "source_teacher_actions",
    "target_qpos",
    "target_qvel",
    "source_global_bank_ids",
)
with np.load(args.p1_bank, allow_pickle=False) as p1_data:
    p1_ids = np.asarray(np.load(args.p1_ids, allow_pickle=False), dtype=np.int64)
    p1_id = int(p1_ids[0])
    p1 = {}
    for key in keys:
        if key in p1_data:
            value = np.asarray(p1_data[key][p1_id : p1_id + 1])
        elif key == "source_policy_action":
            value = np.zeros((1, 20), dtype=np.float32)
        elif key == "source_teacher_actions":
            value = np.zeros((1, 1, 20), dtype=np.float32)
        elif key == "target_qpos":
            value = np.asarray(p1_data["source_qpos"][p1_id : p1_id + 1])
        elif key == "target_qvel":
            value = np.asarray(p1_data["source_qvel"][p1_id : p1_id + 1])
        else:
            value = np.asarray((617,), dtype=np.int64)
        p1[key] = np.repeat(value, args.p1_copies, axis=0)

with np.load(args.arrival_bank, allow_pickle=False) as arrival_data:
    arrival = {key: np.asarray(arrival_data[key]) for key in keys}

teacher_horizon = max(
    p1["source_teacher_actions"].shape[1], arrival["source_teacher_actions"].shape[1]
)
for source in (p1, arrival):
    actions = source["source_teacher_actions"]
    if actions.shape[1] < teacher_horizon:
        padding = np.zeros(
            (actions.shape[0], teacher_horizon - actions.shape[1], actions.shape[2]),
            dtype=actions.dtype,
        )
        source["source_teacher_actions"] = np.concatenate((actions, padding), axis=1)

payload = {key: np.concatenate((p1[key], arrival[key]), axis=0) for key in keys}
payload["phase_ids"] = np.concatenate(
    (
        np.ones(args.p1_copies, dtype=np.int64),
        np.full(arrival["source_qpos"].shape[0], 2, dtype=np.int64),
    )
)
payload["mixed_source_kind"] = payload["phase_ids"].copy()
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output, **payload)
summary = {
    "output": str(args.output.resolve()),
    "rows": int(payload["phase_ids"].size),
    "p1_rows": int(args.p1_copies),
    "arrival_rows": int(arrival["source_qpos"].shape[0]),
    "note": "P1 rows and normal-PhysX arrival rows are sampled equally by one route policy.",
}
args.output.with_suffix(".json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("P1_P2_MIXED_ROUTE_BANK=" + json.dumps(summary, ensure_ascii=False))
