from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/m3_2_30_index_near_surface"
CHAIN_ROOT = REPORT_ROOT / "no_reset_chain"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_record(path: Path, purpose: str) -> dict:
    return {
        "用途": purpose,
        "相对路径": str(path.relative_to(ROOT)),
        "SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "修改时间": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
            timespec="seconds"
        ),
    }


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def link(path: str, label: str) -> str:
    return f'<a href="{html.escape(path)}">{html.escape(label)}</a>'


def badge(level: str) -> str:
    labels = {
        "A": "A 严格复核",
        "B": "B 物理验证",
        "C": "C 训练入口/旧门",
        "D": "D 运动学参考",
        "X": "X 未完成/不通过",
    }
    return f'<span class="badge level-{level}">{labels[level]}</span>'


def main() -> None:
    parser = argparse.ArgumentParser(description="生成当前任务可靠数据模型总览。")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_path = REPORT_ROOT / "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
    p3_p4_path = REPORT_ROOT / "p3_braked_p4_hold_timing_fixed_seed2331.json"
    p3_p4_teacher_path = (
        REPORT_ROOT / "v9_current_continuous_p3_braked50_teacher200_validation.json"
    )
    alignment_path = REPORT_ROOT / "v9_braked50_mujoco_physx_alignment_100ms.json"
    phase18_path = CHAIN_ROOT / "phase18中指继续撤回_bank60局部搜索_seed2891.json"
    dual_bundle_path = CHAIN_ROOT / "拇指继续绕侧_中指卸载_双入口400条_报告.json"
    dual_smoke_path = CHAIN_ROOT / "拇指继续绕侧_中指卸载_双入口smoke_seed3054.json"
    strict_paths = [
        CHAIN_ROOT / "phase17_无teacher_margin11成功策略_严格10mm_seed3040.json",
        CHAIN_ROOT / "phase17_无teacher_严格10mm_多seed复核_seed3041.json",
        CHAIN_ROOT / "phase17_无teacher_严格10mm_多seed复核_seed3042.json",
    ]
    p1_p2_micro_hold_path = (
        ROOT
        / "reports/manus_mujoco_presets/p1_p2_micro_rebuild_m3_2_18/hold_summary.csv"
    )
    p1_p2_ring_under_path = (
        ROOT
        / "reports/manus_mujoco_presets/b12_ring_under_recapture_m3_2_32/summary.json"
    )
    p1_p2_narrow_path = (
        ROOT
        / "reports/manus_mujoco_presets/narrow_021_five_step_chain_m3_2_33/B12_narrow_validation.csv"
    )
    p1_p2_continuous_path = (
        ROOT
        / "reports/manus_mujoco_presets/continuous_p1_p2_p3_m3_2_34/summary.json"
    )
    canonical = load_json(canonical_path)
    p3_p4 = load_json(p3_p4_path)
    p3_p4_teacher = load_json(p3_p4_teacher_path)
    alignment = load_json(alignment_path)
    phase18 = load_json(phase18_path)
    dual_bundle = load_json(dual_bundle_path)
    dual_smoke = load_json(dual_smoke_path)
    strict_reports = [load_json(path) for path in strict_paths]
    p1_p2_micro_hold = load_csv(p1_p2_micro_hold_path)
    p1_p2_ring_under = load_json(p1_p2_ring_under_path)
    p1_p2_narrow = load_csv(p1_p2_narrow_path)
    p1_p2_continuous = load_json(p1_p2_continuous_path)
    p1_p2_narrow_passed = [row for row in p1_p2_narrow if row["passed"] == "True"]
    strict_results = [
        {
            "seed": row["seed"],
            "成功": row["handoff_hold_diagnostics"]["strict_success_envs"],
            "样本": row["num_envs"],
            "保持_ms": row["handoff_hold_diagnostics"]["best_hold_ms"],
        }
        for row in strict_reports
    ]

    canonical_rows = [
        {
            "id": f"P{row['phase']}",
            "名称": row["name"],
            "接触": row.get("contact_set", []),
            "证据": row.get("source_validation_status", "unknown"),
            "可靠性": "D",
            "解释": "关节与笔姿态参考；尚不能替代当前物理门动态验收。",
        }
        for row in canonical["phases"]
    ]

    stages = [
        {
            "范围": "P1–P5 canonical",
            "状态": "参考可用",
            "可靠性": "D",
            "指标": "5 个姿态、明确接触集合",
            "结论": "用于 reset、reward 和诊断，不代表静态或动态物理成功。",
        },
        {
            "范围": "旧 P1→P2 / B12 动态数据",
            "状态": "旧语义局部成立",
            "可靠性": "C",
            "指标": (
                f"窄扰动 {len(p1_p2_narrow_passed)}/{len(p1_p2_narrow)}；"
                f"ring-under arrival cache {p1_p2_ring_under['arrival_cache_states']} 条"
            ),
            "结论": "末端主接触偏向 thumb/ring/pinky，不满足当前 P2 的 thumb/index/middle 保持定义。",
        },
        {
            "范围": "P3→braked P4",
            "状态": "局部技能成立",
            "可靠性": "B",
            "指标": (
                f"{p3_p4['successes']}/{p3_p4['episodes']} = "
                f"{pct(p3_p4['success_rate'])}；平均穿透 "
                f"{p3_p4['mean_episode_max_penetration_mm']:.3f} mm"
            ),
            "结论": "当前 PhysX 单 seed 策略验证；另有 200 条无中间 reset teacher。",
        },
        {
            "范围": "phase6–11 下托/继续绕侧",
            "状态": "可作训练入口",
            "可靠性": "C",
            "指标": "存在 PhysX 局部成功轨迹",
            "结论": "旧门未检查最终对侧轴向对齐，不能称完整越侧夹持。",
        },
        {
            "范围": "旧 phase12–14 接管候选",
            "状态": "新门不通过",
            "可靠性": "X",
            "指标": "拇食指沿笔轴错位约 33–45 mm；新上限 20 mm",
            "结论": "历史 successful_local_primitive 标签已降级。",
        },
        {
            "范围": "phase14→15 接管后卸载",
            "状态": "未完成",
            "可靠性": "X",
            "指标": "0/1024；最佳正式窗口 8.3 ms",
            "结论": "入口可保留，但不能作为已完成 transition。",
        },
        {
            "范围": "phase16→17 P1 严格保持",
            "状态": "无 teacher 严格成立",
            "可靠性": "A",
            "指标": "3 seeds 均 11/1024；连续 100 ms",
            "结论": "当前最强晚期闭环证据，但从晚期 reset 开始。",
        },
        {
            "范围": "phase18 中指继续撤回",
            "状态": "未完成",
            "可靠性": "X",
            "指标": (
                f"0/{phase18['num_envs']}；最佳 "
                f"{phase18['best_hold_ms']:.1f} ms / 100 ms"
            ),
            "结论": "尚未获得严格 P1 最终保持。",
        },
        {
            "范围": "双入口拇食指夹持任务",
            "状态": "任务与 reset 已验证",
            "可靠性": "C",
            "指标": "phase11 200 + phase16 200；smoke 无非法数值",
            "结论": "可训练，不代表策略已经学会严格对侧夹持。",
        },
        {
            "范围": "完整 P1→P5→P1 当前语义链",
            "状态": "未完成",
            "可靠性": "X",
            "指标": "没有当前几何门下的一次无重置严格成功",
            "结论": "旧 4/554 完整链属于旧语义，只保留历史参考。",
        },
    ]

    entities = [
        ("PenSpec", "实测笔参数", "质量、三段直径、长度、质心、摩擦区间"),
        ("CanonicalGrasp", "P1–P5 姿态", "hand qpos、pen pose、接触集合"),
        ("TransitionPrimitive", "局部动态片段", "起终 phase、动作、接触、保持"),
        ("ResetState", "训练入口状态", "qpos/qvel、目标、初始笔轴、phase"),
        ("TeacherTrajectory", "teacher 时序", "observation/action/state trace"),
        ("RLTask", "Isaac Lab 任务", "reset 分布、reward、gate、终止条件"),
        ("Checkpoint", "策略参数", "BC/PPO 权重与 normalization"),
        ("Evaluation", "独立评测", "seed、样本数、严格成功、失败原因"),
        ("CrossSimReport", "MuJoCo/PhysX 对齐", "50/100 ms 状态误差与接触一致性"),
    ]

    data_model = {
        "数据模型版本": "1.2",
        "生成日期": "2026-07-22",
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "当前主线": "v9 / PhysX",
        "可靠性定义": {
            "A": "当前严格门，多 seed 独立复核",
            "B": "当前物理门下有明确成功，但复核范围有限",
            "C": "训练入口、smoke 或旧门成功，不等于目标完成",
            "D": "运动学/canonical 参考",
            "X": "当前门下未完成或明确不通过",
        },
        "笔模型": {
            "质量_g": 9.4,
            "长度_mm": 141.5,
            "直径_mm": {"前": 11.2, "中": 9.8, "后": 9.5},
            "质心距左端_mm": 80.1,
            "质心距后端_mm": 61.4,
            "摩擦角_deg": "约 5–10，低摩擦，尚未精确重复测量",
            "可靠性": "B/C",
        },
        "canonical": canonical_rows,
        "P1到P2历史数据": {
            "证据等级": "C",
            "当前语义兼容": False,
            "当前P1接触定义": canonical_rows[0]["接触"],
            "当前P2接触定义": canonical_rows[1]["接触"],
            "早期关键帧保持": {
                row["label"]: {
                    "保持_s": float(row["survival_s"]),
                    "末端接触": row["final_contacts"],
                    "最大穿透_mm": float(row["max_penetration_mm"]),
                }
                for row in p1_p2_micro_hold
                if row["label"] in {"hold_P1_micro", "hold_P2_micro"}
            },
            "0.21s窄扰动": {
                "通过": len(p1_p2_narrow_passed),
                "总数": len(p1_p2_narrow),
                "通过率": len(p1_p2_narrow_passed) / len(p1_p2_narrow),
                "通过样本最坏穿透_mm": max(
                    float(row["max_penetration_mm"]) for row in p1_p2_narrow_passed
                ),
            },
            "ring_under再捕获": {
                "入口时间_s": p1_p2_ring_under["selected_source_t"],
                "动态arrival_cache条数": p1_p2_ring_under["arrival_cache_states"],
                "投影P2保持_s": p1_p2_ring_under["selected_entry"][
                    "projected_p2_hold_s"
                ],
                "最大穿透_mm": p1_p2_ring_under["selected_entry"][
                    "max_penetration_mm"
                ],
                "末端接触": p1_p2_ring_under["selected_entry"]["final_contacts"],
            },
            "无重置P1到P2到P3入口": {
                "总生存_s": p1_p2_continuous["continuous_chain"]["survival_s"],
                "边界重置次数": p1_p2_continuous["continuous_chain"][
                    "boundary_reset_count"
                ],
                "整段最大穿透_mm": p1_p2_continuous["continuous_chain"][
                    "max_penetration_mm"
                ],
                "P2末端接触": p1_p2_continuous["running_to_p2"]["final_contacts"],
            },
            "结论": "控制器、动态速度和 reset cache 可复用；当前 P2 reward/gate 与末端接触必须重建。",
        },
        "阶段证据": stages,
        "双入口任务": {
            "task": dual_smoke["task"],
            "入口": dual_bundle["入口phase分布"],
            "终点几何门": {
                "拇食指径向余弦_max": -0.5,
                "轴向错位_max_mm": 20.0,
                "同时要求": "拇指/食指有效接触与物理安全门",
            },
            "当前状态": "仅完成 reset/task smoke，未训练出严格成功策略",
        },
        "严格评测": strict_results,
        "P3到P4": {
            "策略成功率": p3_p4["success_rate"],
            "策略成功数": p3_p4["successes"],
            "episodes": p3_p4["episodes"],
            "teacher轨迹": p3_p4_teacher["selected_trajectories"],
            "teacher无中间reset": p3_p4_teacher[
                "continuous_without_intermediate_reset"
            ],
        },
        "跨引擎风险": {
            "50ms位置误差_mm": alignment["summary_50ms"][
                "pen_position_error_mm"
            ],
            "100ms位置误差_mm": alignment["summary_100ms"][
                "pen_position_error_mm"
            ],
            "100ms姿态误差_deg": alignment["summary_100ms"][
                "pen_orientation_error_deg"
            ],
            "结论": "当前 MuJoCo/PhysX 只能用于定性互检，尚未达到轨迹级对齐。",
        },
        "实体关系": [
            "PenSpec -> CanonicalGrasp",
            "CanonicalGrasp -> TransitionPrimitive",
            "TransitionPrimitive -> ResetState",
            "ResetState -> RLTask",
            "TeacherTrajectory -> Checkpoint",
            "RLTask + Checkpoint -> Evaluation",
            "Evaluation -> 下一轮 ResetState/TeacherTrajectory",
            "CrossSimReport -> 物理可信度约束",
        ],
        "证据来源": [
            source_record(canonical_path, "P1-P5 canonical manifest"),
            source_record(p3_p4_path, "P3 到制动 P4 策略评测"),
            source_record(p3_p4_teacher_path, "P3 到制动 P4 teacher 连续性验证"),
            source_record(alignment_path, "MuJoCo/PhysX 100 ms 对齐报告"),
            source_record(phase18_path, "phase18 中指撤回局部搜索"),
            source_record(dual_bundle_path, "双入口 reset bundle"),
            source_record(dual_smoke_path, "双入口任务 smoke"),
            source_record(p1_p2_micro_hold_path, "旧 P1/P2 关键帧保持"),
            source_record(p1_p2_ring_under_path, "旧 B12 ring-under 再捕获"),
            source_record(p1_p2_narrow_path, "旧 B12 0.21 s 窄扰动验证"),
            source_record(p1_p2_continuous_path, "旧无重置 P1 到 P2 到 P3-entry"),
            *[
                source_record(path, f"phase16 到 17 严格复核 seed {report['seed']}")
                for path, report in zip(strict_paths, strict_reports)
            ],
        ],
    }
    (output_dir / "当前任务可靠数据模型.json").write_text(
        json.dumps(data_model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stage_rows = "".join(
        f"<tr><td>{html.escape(row['范围'])}</td>"
        f"<td>{badge(row['可靠性'])}</td>"
        f"<td>{html.escape(row['状态'])}</td>"
        f"<td>{html.escape(row['指标'])}</td>"
        f"<td>{html.escape(row['结论'])}</td></tr>"
        for row in stages
    )
    canonical_html = "".join(
        f"<article><header><strong>{row['id']}</strong>{badge('D')}</header>"
        f"<h3>{html.escape(row['名称'])}</h3>"
        f"<p>接触：{html.escape(' / '.join(row['接触']))}</p>"
        f"<small>{html.escape(row['证据'])}</small></article>"
        for row in canonical_rows
    )
    entity_html = "".join(
        f'<div class="entity"><b>{html.escape(code)}</b>'
        f"<span>{html.escape(name)}</span><small>{html.escape(desc)}</small></div>"
        for code, name, desc in entities
    )
    strict_html = "".join(
        f"<tr><td>{row['seed']}</td><td>{row['成功']}/{row['样本']}</td>"
        f"<td>{row['保持_ms']:.0f} ms</td><td>无在线 teacher</td></tr>"
        for row in strict_results
    )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dex5 转笔任务可靠数据模型总览</title>
<style>
:root{{--bg:#0e1113;--band:#15191c;--line:#343b40;--text:#edf1f2;--muted:#aeb7bd;
--green:#50c878;--cyan:#58b9d6;--yellow:#e7bd54;--red:#e87568;--gray:#8f9aa0}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:"Noto Sans CJK SC",sans-serif}}
main{{max-width:1440px;margin:auto;padding:28px}} h1{{font-size:30px;margin:0}} h2{{font-size:20px;margin:0 0 16px}}
h3{{font-size:15px;margin:10px 0 8px}} p{{color:var(--muted);line-height:1.65}} a{{color:#7ec9e0}}
.top{{display:flex;justify-content:space-between;align-items:end;border-bottom:1px solid var(--line);padding-bottom:20px}}
.stamp{{color:var(--muted);font-size:13px}} section{{padding:24px 0;border-bottom:1px solid var(--line)}}
.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .metric{{border-left:4px solid var(--gray);padding:10px 14px;background:var(--band)}}
.metric b{{display:block;font-size:24px}} .metric span{{color:var(--muted);font-size:13px}} .metric.good{{border-color:var(--green)}}
.metric.warn{{border-color:var(--yellow)}} .metric.bad{{border-color:var(--red)}}
.badge{{display:inline-block;padding:3px 7px;border:1px solid;border-radius:4px;font-size:12px;white-space:nowrap}}
.level-A{{color:var(--green)}} .level-B{{color:var(--cyan)}} .level-C{{color:var(--yellow)}} .level-D{{color:var(--gray)}} .level-X{{color:var(--red)}}
.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;align-items:stretch}} .node{{padding:14px;background:var(--band);border-top:3px solid var(--gray);min-height:112px}}
.node strong{{display:block;margin-bottom:8px}} .node span{{color:var(--muted);font-size:13px;line-height:1.5}} .node.a{{border-color:var(--green)}} .node.b{{border-color:var(--cyan)}} .node.c{{border-color:var(--yellow)}} .node.x{{border-color:var(--red)}}
.branch{{display:grid;grid-template-columns:1fr 90px 1fr;gap:12px;align-items:center}} .arrow{{text-align:center;color:var(--cyan);font-weight:700}}
.gate{{border:1px solid var(--cyan);padding:16px;background:var(--band)}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:10px;border-bottom:1px solid #2d3337;vertical-align:top}} th{{color:#d8dfe2;background:#171b1e;position:sticky;top:0}}
.canonical{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}} article{{background:var(--band);border:1px solid var(--line);padding:14px;border-radius:6px}}
article header{{display:flex;justify-content:space-between;align-items:center}} article p,article small{{font-size:12px}}
.media{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} figure{{margin:0;background:var(--band);border:1px solid var(--line);padding:12px}}
figure img{{display:block;width:100%;height:auto}} figcaption{{color:var(--muted);padding-top:10px;font-size:13px}}
.entities{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .entity{{padding:12px;background:var(--band);border-left:3px solid var(--cyan)}}
.entity b,.entity span,.entity small{{display:block}} .entity span{{margin:5px 0}} .entity small{{color:var(--muted)}}
.warning{{border-left:4px solid var(--red);padding:14px;background:#1a1717;color:#f0c6c1}}
.links{{display:flex;flex-wrap:wrap;gap:12px}} .links a{{border:1px solid var(--line);padding:8px 10px;text-decoration:none;border-radius:4px}}
@media(max-width:950px){{.summary,.flow,.canonical,.entities{{grid-template-columns:1fr 1fr}}.media,.branch{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg)}}}}
@media(max-width:600px){{main{{padding:16px}}.summary,.flow,.canonical,.entities{{grid-template-columns:1fr}}.top{{align-items:start;gap:10px;flex-direction:column}}}}
</style></head><body><main>
<div class="top"><div><h1>Dex5 转笔任务可靠数据模型</h1><p>当前事实、训练入口与未完成项分层总览</p></div><div class="stamp">生成日期 2026-07-22 · 当前 v9 / PhysX 主线</div></div>

<section><div class="summary">
<div class="metric good"><b>1</b><span>A 级局部结论：phase16→17 严格保持</span></div>
<div class="metric"><b>{pct(p3_p4['success_rate'])}</b><span>P3→braked P4，1024 episodes / 单 seed</span></div>
<div class="metric warn"><b>400</b><span>双入口 reset：phase11 + phase16</span></div>
<div class="metric bad"><b>未完成</b><span>当前语义完整 P1→P5→P1 无重置链</span></div>
</div></section>

<section><h2>当前任务图</h2><div class="flow">
<div class="node"><strong>P1–P5 canonical</strong><span>D 级运动学参考。定义接触与 reset，不等于物理成功。</span></div>
<div class="node c"><strong>P1→P2 当前语义</strong><span>旧 B12 有 C 级动态数据，但末端接触不兼容；当前 gate 下仍未完成。</span></div>
<div class="node b"><strong>P3→braked P4</strong><span>B 级 PhysX 局部策略；{p3_p4['successes']}/{p3_p4['episodes']}。</span></div>
<div class="node c"><strong>phase6→11</strong><span>食中稳夹，拇指下托并继续绕侧；可作为训练入口。</span></div>
<div class="node x"><strong>严格拇食指对侧夹持</strong><span>新几何门：余弦 ≤ -0.50、轴向错位 ≤ 20 mm。尚未训练成功。</span></div>
<div class="node a"><strong>phase16→17</strong><span>A 级：3 个独立 seed 均达到严格 100 ms 保持，但从晚期 reset 开始。</span></div>
<div class="node x"><strong>phase18→P1 最终保持</strong><span>X 级：0/2048；当前最佳 66.7 ms，尚未达到 100 ms。</span></div>
</div></section>

<section><h2>P1→P2 历史数据与当前差距</h2>
<div class="summary">
<div class="metric warn"><b>{len(p1_p2_narrow_passed)}/{len(p1_p2_narrow)}</b><span>0.21 s 附近旧 B12 窄扰动通过</span></div>
<div class="metric warn"><b>{p1_p2_ring_under['arrival_cache_states']}</b><span>ring-under 动态 arrival cache</span></div>
<div class="metric"><b>{p1_p2_ring_under['selected_entry']['projected_p2_hold_s']:.2f} s</b><span>旧 projected P2 短时保持</span></div>
<div class="metric bad"><b>{p1_p2_continuous['continuous_chain']['max_penetration_mm']:.2f} mm</b><span>旧无重置链最大穿透</span></div>
</div>
<p>旧数据确实证明过 P1→B12→ring-under→projected P2 的动态可行性，并有一次持续 {p1_p2_continuous['continuous_chain']['survival_s']:.2f} s、边界重置为 0 的 P1→P2→P3-entry rollout。但旧 P2 末端接触是 <strong>{html.escape('/'.join(p1_p2_continuous['running_to_p2']['final_contacts']))}</strong>；当前 P2 明确要求 <strong>{html.escape('/'.join(canonical_rows[1]['接触']))}</strong>，即 P1 的拇/食/中三指继续控笔、无名指只转移到底部辅助。因此旧控制器和 cache 可作为 teacher/prefill，不能作为当前 P1→P2 成功标签。</p>
</section>

<section><h2>双入口汇合模型</h2><div class="branch">
<div class="node c"><strong>入口 A · phase11</strong><span>食指/中指仍承力，拇指继续绕侧。200 条 reset。</span></div>
<div class="arrow">汇入 →</div>
<div class="gate"><strong>统一终点 · 拇指/食指对侧夹笔</strong><p>两指有效接触；径向余弦 ≤ -0.50；轴向错位 ≤ 20 mm；通过位置、笔轴、速度、穿透与峰值力门。</p></div>
</div><div class="branch" style="margin-top:12px">
<div class="node c"><strong>入口 B · phase16</strong><span>接管后中指开始卸载，拇指仍可继续调整。200 条 reset。</span></div>
<div class="arrow">汇入 →</div><div class="node x"><strong>当前策略状态</strong><span>task/reset smoke 已通过；严格终点成功仍为 0，等待并行训练。</span></div>
</div></section>

<section><h2>阶段证据表</h2><div style="overflow:auto"><table><thead><tr><th>范围</th><th>可靠性</th><th>状态</th><th>可量化证据</th><th>可作出的结论</th></tr></thead><tbody>{stage_rows}</tbody></table></div></section>

<section><h2>P1–P5 canonical 数据</h2><p>这些数据仍然有价值，但证据类型是姿态/接触先验，不应显示为动态技能完成。</p><div class="canonical">{canonical_html}</div></section>

<section><h2>严格 phase16→17 复核</h2><table><thead><tr><th>seed</th><th>严格成功</th><th>最佳连续保持</th><th>部署条件</th></tr></thead><tbody>{strict_html}</tbody></table></section>

<section><h2>数据实体关系</h2><div class="entities">{entity_html}</div><p>主链：PenSpec → CanonicalGrasp → TransitionPrimitive → ResetState → RLTask → Checkpoint → Evaluation；评测结果再生成下一轮 reset/teacher。CrossSimReport 作为物理可信度约束，不参与虚增成功等级。</p></section>

<section><h2>代表性可视证据</h2><div class="media">
<figure><img src="../canonical_p1_p5_v9_P4拇指下托笔端/canonical_p1_p5_realmesh_cycle.gif"><figcaption>P1–P5 canonical：D 级姿态参考。</figcaption></figure>
<figure><img src="../P3与P3到P4训练结果_v9正式评估/P3到制动P4代表成功轨迹.gif"><figcaption>P3→braked P4：当前 PhysX 局部成功代表。</figcaption></figure>
<figure><img src="../no_reset_chain/拇指彻底越侧与对侧夹持_几何门修正版_v6/5_5 拇指继续绕侧.gif"><figcaption>phase10→11：可作为双入口课程的继续绕侧状态。</figcaption></figure>
<figure><img src="../no_reset_chain/拇指彻底越侧与对侧夹持_几何门修正版_v6/7_7 旧phase13候选：尚未形成对侧拇食指夹持.gif"><figcaption>旧 phase13：新几何门下不合格，仅作反例。</figcaption></figure>
</div></section>

<section><h2>物理可信度与风险</h2><div class="warning"><strong>MuJoCo/PhysX 尚未轨迹级对齐。</strong><br>50 ms 笔位置误差 {alignment['summary_50ms']['pen_position_error_mm']:.2f} mm；100 ms 位置误差 {alignment['summary_100ms']['pen_position_error_mm']:.2f} mm、姿态误差 {alignment['summary_100ms']['pen_orientation_error_deg']:.2f}°。当前应以 PhysX 任务内独立评测为主，MuJoCo 用于姿态设计与定性诊断。</div>
<p>代表笔：9.4 g，长 141.5 mm，直径 11.2 / 9.8 / 9.5 mm，质心距左端约 80.1 mm。摩擦角仅粗估 5–10°，仍是 sim2real 不确定项。</p></section>

<section><h2>可靠入口</h2><div class="links">
{link('../canonical_p1_p5_v9_P4拇指下托笔端/index.html','P1–P5 canonical')}
{link('../P3与P3到P4训练结果_v9正式评估/index.html','P3→P4 训练结果')}
{link('../no_reset_chain/拇指彻底越侧与对侧夹持_几何门修正版_v6/index.html','拇指越侧几何复核')}
{link('../no_reset_chain/phase16到17_完整teacher_PPO_DAgger_完成报告.md','phase16→17 完成报告')}
{link('当前任务可靠数据模型.json','机器可读数据模型 JSON')}
</div></section>
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
