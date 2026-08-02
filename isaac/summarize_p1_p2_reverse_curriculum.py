from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVERSE_DIR = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/reverse_curriculum"
)


@dataclass(frozen=True)
class Decision:
    label: str
    reason: str


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_policy_eval(
    record: dict[str, Any],
    min_episodes: int,
    task_contains: str,
    filename_contains: str,
    path: Path,
) -> bool:
    if task_contains and task_contains not in str(record.get("task", "")):
        return False
    if filename_contains and filename_contains not in path.name:
        return False
    if "checkpoint" not in record:
        return False
    if not record.get("evaluation_complete", False):
        return False
    if int(record.get("completed_episodes", 0)) < min_episodes:
        return False
    if int(record.get("leaked_id_count", 1)) != 0:
        return False
    if int(record.get("invalid_episodes", 1)) != 0:
        return False
    return True


def _curriculum_label(record: dict[str, Any], path: Path) -> str:
    text = f"{record.get('task', '')} {path.name}".lower()
    if "validated" in text and "r0" in text:
        return "R0_validated_P2_hold"
    for label in ("r0", "r1", "r2", "r3", "r4", "r5"):
        if label in text:
            return label.upper()
    return "unknown"


def _decision(success_rate: float) -> Decision:
    if success_rate > 0.80:
        return Decision("advance", "成功率 >80%，可向更远起点扩展")
    if success_rate >= 0.20:
        return Decision("frontier", "成功率在 20%-80%，保留为训练前沿")
    return Decision("too_hard", "成功率 <20%，暂不继续扩展")


def _record_summary(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    success_rate = float(record.get("success_rate", 0.0))
    decision = _decision(success_rate)
    return {
        "file": str(path),
        "curriculum": _curriculum_label(record, path),
        "task": record.get("task", ""),
        "split": record.get("split", ""),
        "seed": record.get("seed"),
        "checkpoint": record.get("checkpoint", ""),
        "checkpoint_sha256": record.get("checkpoint_sha256", ""),
        "completed_episodes": int(record.get("completed_episodes", 0)),
        "success_rate": success_rate,
        "drop_rate": float(record.get("drop_rate", 0.0)),
        "invalid_episodes": int(record.get("invalid_episodes", 0)),
        "leaked_id_count": int(record.get("leaked_id_count", 0)),
        "observed_max_penetration_mm": float(
            record.get("observed_max_penetration_mm", float("nan"))
        ),
        "ring_target_distance_mean_mm": float(
            record.get("ring_target_distance_mean_mm", float("nan"))
        ),
        "ring_target_distance_min_mm": float(
            record.get("ring_target_distance_min_mm", float("nan"))
        ),
        "gate_pass_ratios": record.get("gate_pass_ratios", {}),
        "contact_ratio_thumb_index_middle_ring_pinky": record.get(
            "step_contact_ratio_thumb_index_middle_ring_pinky", []
        ),
        "force_share_thumb_index_middle_ring_pinky": record.get(
            "force_share_thumb_index_middle_ring_pinky", []
        ),
        "decision": decision.label,
        "decision_reason": decision.reason,
    }


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _write_markdown(path: Path, records: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    lines = [
        "# P1->P2 反向课程严格评估汇总",
        "",
        "筛选规则：只统计完成评估、无测试泄漏、无 NaN/非法 episode、episode 数达标的 policy evaluation JSON。",
        "",
    ]
    if best is None:
        lines.append("当前没有符合筛选条件的评估结果。")
    else:
        lines.extend(
            [
                "## 当前前沿",
                "",
                f"- 课程：{best['curriculum']}",
                f"- 成功率：{_format_pct(best['success_rate'])}",
                f"- 掉笔率：{_format_pct(best['drop_rate'])}",
                f"- 最大穿透：{best['observed_max_penetration_mm']:.3f} mm",
                f"- 决策：{best['decision']}，{best['decision_reason']}",
                f"- checkpoint：`{best['checkpoint']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 候选结果",
            "",
            "| 课程 | 成功率 | 掉笔率 | 最大穿透 mm | episodes | 决策 | 文件 |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["curriculum"],
                    _format_pct(item["success_rate"]),
                    _format_pct(item["drop_rate"]),
                    f"{item['observed_max_penetration_mm']:.3f}",
                    str(item["completed_episodes"]),
                    item["decision"],
                    Path(item["file"]).name,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize strict P1->P2 reverse-curriculum evaluations."
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_REVERSE_DIR)
    parser.add_argument("--min_episodes", type=int, default=1024)
    parser.add_argument(
        "--task_contains",
        default="",
        help="Only include evaluations whose task string contains this text.",
    )
    parser.add_argument(
        "--filename_contains",
        default="",
        help="Only include evaluation JSON files whose filename contains this text.",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    args = parser.parse_args()

    evals: list[dict[str, Any]] = []
    for path in sorted(args.dir.glob("*.json")):
        record = _load_json(path)
        if record is None or not _is_policy_eval(
            record,
            args.min_episodes,
            args.task_contains,
            args.filename_contains,
            path,
        ):
            continue
        evals.append(_record_summary(path, record))
    evals.sort(
        key=lambda item: (
            item["success_rate"],
            -item["drop_rate"],
            -item["observed_max_penetration_mm"],
        ),
        reverse=True,
    )
    best = evals[0] if evals else None
    summary = {
        "source_dir": str(args.dir),
        "min_episodes": args.min_episodes,
        "task_contains": args.task_contains,
        "filename_contains": args.filename_contains,
        "num_valid_evaluations": len(evals),
        "best": best,
        "evaluations": evals,
    }

    output_json = args.output_json or (args.dir / "P1_P2_reverse_curriculum_eval_summary.json")
    output_md = args.output_md or (args.dir / "P1_P2反向课程严格评估汇总.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output_md, evals, best)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
