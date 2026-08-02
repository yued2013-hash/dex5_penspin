from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Merge successful phase-15 hold trajectories for BC."
)
parser.add_argument("--inputs", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--limit", type=int, default=200)
parser.add_argument("--stage-id", type=int, default=15)
parser.add_argument(
    "--description",
    default="phase15 食指/中指稳夹、拇指下托接管 BC 轨迹",
)
parser.add_argument(
    "--action-semantics",
    default="index_middle_clamp_thumb_under_takeover",
)
args = parser.parse_args()


def main() -> None:
    rows: list[dict[str, np.ndarray]] = []
    seen: set[bytes] = set()
    input_successes: dict[str, int] = {}
    for path in args.inputs:
        with np.load(path) as data:
            successful = np.flatnonzero(data["success"])
            input_successes[str(path)] = int(len(successful))
            for row_id in successful:
                identity = np.concatenate(
                    (
                        data["entry_hand_qpos"][row_id],
                        data["entry_pen_root_state"][row_id],
                        data["candidate_action_first"][row_id],
                        data["candidate_action_second"][row_id],
                        np.atleast_1d(data["first_duration"][row_id]),
                    )
                )
                key = np.round(identity, decimals=6).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    key: np.asarray(data[key][row_id])
                    for key in (
                        "observation_trace",
                        "action_trace",
                        "entry_hand_qpos",
                        "entry_hand_qvel",
                        "entry_pen_root_state",
                        "candidate_action_first",
                        "candidate_action_second",
                        "first_duration",
                        "second_duration",
                        "source_bank_id",
                        "max_hold_steps",
                        "start_step",
                        "success_step",
                    )
                }
                if "armed_observation_trace" in data:
                    row["observation"] = np.asarray(
                        data["armed_observation_trace"][row_id]
                    )
                    row["action"] = np.asarray(
                        data["armed_action_trace"][row_id]
                    )
                    row["stage_id"] = np.asarray(
                        data["armed_phase_trace"][row_id]
                    )
                    row["lengths"] = np.asarray(
                        data["armed_trace_length"][row_id]
                    )
                else:
                    row["observation"] = row["observation_trace"]
                    row["action"] = row["action_trace"]
                    row["stage_id"] = np.full(
                        row["observation"].shape[0],
                        args.stage_id,
                        dtype=np.int64,
                    )
                    row["lengths"] = np.asarray(
                        row["observation"].shape[0] - 1,
                        dtype=np.int64,
                    )
                rows.append(row)
    if len(rows) < args.limit:
        raise ValueError(
            f"Only {len(rows)} unique successful trajectories; "
            f"need {args.limit}"
        )
    rows = rows[: args.limit]
    payload = {
        key: np.stack([row[key] for row in rows])
        for key in rows[0]
    }
    payload.update(
        {
            "success": np.ones(args.limit, dtype=bool),
            "selection_kind": np.full(args.limit, "success"),
            "control_dt_s": np.asarray(1.0 / 120.0, dtype=np.float32),
            "continuous_without_intermediate_reset": np.asarray(True),
            "action_semantics": np.asarray(
                args.action_semantics
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    ring_pinky_max = float(
        np.abs(payload["action"][:, :, 8:16]).max()
    )
    report = {
        "说明": args.description,
        "阶段": args.stage_id,
        "动作语义": args.action_semantics,
        "输入正式成功数": input_successes,
        "去重后正式成功数": len(seen),
        "导出轨迹数": args.limit,
        "每条入口合格帧数": 12,
        "每条有效控制帧数范围": [
            int(payload["lengths"].min()),
            int(payload["lengths"].max()),
        ],
        "控制周期毫秒": 1000.0 / 120.0,
        "无中间重置": True,
        "无名指和小指动作绝对值最大值": ring_pinky_max,
        "输出": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("PHASE15_BC_MERGE=" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
