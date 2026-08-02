from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MAIN = REPORTS / "m3_2_30_index_near_surface"
RECORD = REPORTS / "清理记录_2026-07-22"


def files_under(path: Path) -> list[Path]:
    if path.is_file() or path.is_symlink():
        return [path]
    return [item for item in path.rglob("*") if item.is_file() or item.is_symlink()]


def main() -> None:
    parser = argparse.ArgumentParser(description="保守清理 Dex5 转笔报告生成物。")
    parser.add_argument("--apply", action="store_true", help="执行已生成的清理清单")
    args = parser.parse_args()
    RECORD.mkdir(parents=True, exist_ok=True)
    manifest_path = RECORD / "待删除文件.json"

    if args.apply:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        removed = 0
        removed_bytes = 0
        skipped = []
        for row in manifest["文件"]:
            path = ROOT / row["相对路径"]
            if not path.exists() and not path.is_symlink():
                skipped.append(row["相对路径"])
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
            removed_bytes += row["字节"]
        for directory in sorted(MAIN.rglob("*"), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    pass
        result = {
            "执行时间": datetime.now().astimezone().isoformat(timespec="seconds"),
            "删除文件数": removed,
            "按清单释放字节": removed_bytes,
            "跳过": skipped,
        }
        (RECORD / "执行结果.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    reasons: dict[Path, str] = {}

    def add(path: Path, reason: str) -> None:
        for file_path in files_under(path):
            reasons.setdefault(file_path.resolve(), reason)

    # Current v9 canonical is authoritative; previous projected variants are superseded.
    for path in MAIN.iterdir():
        if path.is_dir() and path.name.startswith("canonical_") and "_v9_" not in path.name:
            add(path, "已由 canonical v9 取代")
        if (
            path.is_file()
            and path.name.startswith("canonical_p1_p5_manifest")
            and "_v9_" not in path.name
        ):
            add(path, "已由 canonical v9 manifest 取代")

    # Root-level smoke/screen/scan/probe artifacts have formal or retained task reports.
    for path in MAIN.iterdir():
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(token in lower for token in ("smoke", "screen", "scan", "probe")):
            add(path, "正式/最佳评测已存在的中间 smoke 或筛查文件")

    # Old standalone exploratory report directories; reusable Manus caches are untouched.
    for path in REPORTS.iterdir():
        if not path.is_dir() or path in {MAIN, REPORTS / "manus_mujoco_presets", RECORD}:
            continue
        lower = path.name.lower()
        if path.name == "m3_1_hold_oracle_smoke":
            continue
        if any(token in lower for token in ("_smoke", "_probe", "_sweep")):
            add(path, "已被后续正式阶段取代的独立探索目录")

    # In retained current task folders, remove only smoke and checkpoint-scan reports.
    retained_task_dirs = [
        MAIN / "no_reset_chain",
        MAIN / "p4_p5_curriculum",
        MAIN / "p5_p1_curriculum",
    ]
    for task_dir in retained_task_dirs:
        if not task_dir.exists():
            continue
        for path in task_dir.rglob("*"):
            if path.is_file() and ("smoke" in path.name.lower() or "扫描" in path.name):
                add(path, "当前任务已有正式结果替代的 smoke/扫描文件")

    for path in REPORTS.rglob("__pycache__"):
        add(path, "Python 临时缓存")

    # Protect everything directly referenced by the two current visual models.
    protected: set[Path] = set()
    for model_path in [
        MAIN / "当前任务可靠数据模型总览_v1/当前任务可靠数据模型.json",
        MAIN / "P1到P5可靠数据模型可视化总汇_v2/P1到P5可靠数据模型.json",
    ]:
        if not model_path.exists():
            continue
        protected.add(model_path.resolve())
        data = load_json(model_path)
        for row in data.get("证据来源", []):
            value = row.get("路径") or row.get("相对路径")
            if value:
                protected.add((ROOT / value).resolve())
    for html_path in [
        MAIN / "当前任务可靠数据模型总览_v1/index.html",
        MAIN / "P1到P5可靠数据模型可视化总汇_v2/index.html",
    ]:
        if html_path.exists():
            protected.add(html_path.resolve())

    for path in list(reasons):
        if path in protected:
            del reasons[path]

    rows = []
    for path, reason in sorted(reasons.items(), key=lambda item: str(item[0])):
        if not path.exists() and not path.is_symlink():
            continue
        rows.append(
            {
                "相对路径": str(path.relative_to(ROOT)),
                "字节": path.lstat().st_size,
                "原因": reason,
            }
        )
    manifest = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "模式": "保守清理",
        "范围": "仅 reports 生成物；不处理源码、IsaacLab 日志和 checkpoint",
        "保留": [
            "当前 v9 canonical 与最终中文总览",
            "A/B 正式评测和多种子复核",
            "C 级 teacher/reset/cache 与 P1-P2 可复用历史数据",
            "当前待确认 X 任务、失败边界与正式大样本",
            "P4-P5、P5-P1 curriculum 和 no-reset 正式数据",
        ],
        "文件数": len(rows),
        "总字节": sum(row["字节"] for row in rows),
        "文件": rows,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (RECORD / "待删除路径.txt").write_text(
        "\n".join(row["相对路径"] for row in rows) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "文件数": manifest["文件数"],
                "总GB": round(manifest["总字节"] / 1024**3, 3),
                "清单": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
