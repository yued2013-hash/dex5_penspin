from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建拇指继续绕侧/中指开始卸载双入口训练缓存。"
    )
    parser.add_argument("--wrap-entry", type=Path, required=True)
    parser.add_argument("--unload-entry", type=Path, required=True)
    parser.add_argument("--rows-per-route", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3052)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sources = (args.wrap_entry.resolve(), args.unload_entry.resolve())
    loaded = [np.load(path, allow_pickle=True) for path in sources]
    try:
        required = (
            "source_qpos",
            "source_qvel",
            "source_target_qpos",
            "source_policy_action",
            "source_target_hand_qpos",
            "source_closure_origin_hand",
            "source_closure_brace_hand",
            "source_closure_origin_pen_pos",
            "source_closure_origin_pen_quat",
            "source_init_pen_pos",
            "source_init_pen_axis",
            "source_max_penetration_m",
            "source_contact_force_n",
            "source_chain_phase_id",
            "target_qpos",
            "target_qvel",
            "phase_ids",
        )
        for data, path in zip(loaded, sources, strict=True):
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{path} 缺少字段: {missing}")

        selections = []
        for data in loaded:
            count = len(data["source_qpos"])
            replace = count < args.rows_per_route
            selections.append(
                rng.choice(count, args.rows_per_route, replace=replace)
            )

        output = {
            key: np.concatenate(
                [data[key][rows] for data, rows in zip(loaded, selections, strict=True)]
            )
            for key in required
        }
        output["route_id"] = np.repeat(
            np.arange(2, dtype=np.int64), args.rows_per_route
        )
        output["route_name"] = np.asarray(
            ["拇指继续绕侧"] * args.rows_per_route
            + ["接管后中指开始卸载"] * args.rows_per_route
        )
        output["source"] = np.asarray("双入口拇食指对侧夹持课程")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **output)
        phase_ids = output["source_chain_phase_id"]
        report = {
            "说明": "两个入口共享严格拇指/食指对侧夹持终点，不强制唯一动作顺序。",
            "拇指继续绕侧入口": str(sources[0]),
            "接管后中指卸载入口": str(sources[1]),
            "每类入口条数": args.rows_per_route,
            "总条数": int(len(phase_ids)),
            "入口phase分布": {
                str(int(phase)): int(np.sum(phase_ids == phase))
                for phase in np.unique(phase_ids)
            },
            "输出": str(args.output.resolve()),
            "seed": args.seed,
        }
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        for data in loaded:
            data.close()


if __name__ == "__main__":
    main()
