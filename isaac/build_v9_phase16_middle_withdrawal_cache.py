from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Build phase-16 middle-withdrawal entries from phase-15 terminal states."
)
parser.add_argument("--inputs", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--limit", type=int, default=200)
parser.add_argument("--target-phase", type=int, default=16)
parser.add_argument(
    "--description",
    default="phase15 保持100ms后的 phase16 中指撤回入口",
)
parser.add_argument(
    "--action-semantics",
    default="拇指和食指保持接管，中指开始撤回，无名指和小指不承力",
)
args = parser.parse_args()


FIELD_MAP = {
    "source_qpos": "terminal_qpos",
    "source_qvel": "terminal_qvel",
    "source_target_qpos": "terminal_target_qpos",
    "source_policy_action": "terminal_policy_action",
    "source_target_hand_qpos": "terminal_target_hand_qpos",
    "source_closure_origin_hand": "terminal_closure_origin_hand",
    "source_closure_brace_hand": "terminal_closure_brace_hand",
    "source_closure_origin_pen_pos": "terminal_closure_origin_pen_pos",
    "source_closure_origin_pen_quat": "terminal_closure_origin_pen_quat",
    "source_init_pen_pos": "terminal_init_pen_pos",
    "source_init_pen_axis": "terminal_init_pen_axis",
    "source_max_penetration_m": "terminal_max_penetration_m",
    "source_contact_force_n": "terminal_contact_force_n",
}


def main() -> None:
    selected: list[dict[str, np.ndarray]] = []
    seen: set[bytes] = set()
    counts: dict[str, int] = {}
    for path in args.inputs:
        with np.load(path) as data:
            rows = np.flatnonzero(data["success"])
            counts[str(path)] = int(len(rows))
            for row_id in rows:
                qpos = np.asarray(data["terminal_qpos"][row_id])
                key = np.round(qpos, decimals=6).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                selected.append(
                    {
                        output_key: np.asarray(data[input_key][row_id])
                        for output_key, input_key in FIELD_MAP.items()
                    }
                )
    if len(selected) < args.limit:
        raise ValueError(
            f"Only {len(selected)} unique phase-16 entries; need {args.limit}"
        )
    selected = selected[: args.limit]
    payload = {
        key: np.stack([row[key] for row in selected])
        for key in FIELD_MAP
    }
    payload["source_chain_phase_id"] = np.full(
        args.limit, args.target_phase, dtype=np.int64
    )
    payload["target_qpos"] = payload["source_qpos"].copy()
    payload["target_qvel"] = payload["source_qvel"].copy()
    payload["phase_ids"] = np.zeros(args.limit, dtype=np.int64)
    payload["source_teacher_actions"] = np.zeros(
        (args.limit, 0, 20), dtype=np.float32
    )
    payload["source"] = np.asarray(
        f"same-episode verified terminal -> phase{args.target_phase} entry"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    report = {
        "说明": args.description,
        "输入成功终态数": counts,
        "去重后终态数": len(seen),
        "导出数": args.limit,
        "phase": args.target_phase,
        "动作语义": args.action_semantics,
        "输出": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("PHASE16_CACHE=" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
