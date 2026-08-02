from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
CHAIN = REPORT / "no_reset_chain"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source(path: Path, purpose: str) -> dict:
    return {
        "用途": purpose,
        "路径": str(path.relative_to(ROOT)),
        "SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "修改时间": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
            timespec="seconds"
        ),
    }


def badge(level: str) -> str:
    labels = {
        "A": "A 严格复核",
        "B": "B 物理验证",
        "C": "C 训练入口/旧门",
        "D": "D 姿态参考",
        "X": "X 当前未完成",
    }
    return f'<span class="badge l{level}">{labels[level]}</span>'


def esc(value: object) -> str:
    return html.escape(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 P1-P5 可靠数据模型可视化总汇。")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = REPORT / "canonical_p1_p5_manifest_v9_P4拇指下托笔端.json"
    overview_path = REPORT / "当前任务可靠数据模型总览_v1/当前任务可靠数据模型.json"
    p3p4_path = REPORT / "p3_braked_p4_hold_timing_fixed_seed2331.json"
    p4p5_a_path = CHAIN / "current_p4_p5_no_reset_formal_seed2430.json"
    p4p5_b_path = CHAIN / "current_p4_p5_no_reset_export_seed2431.json"
    p4p5p1_path = CHAIN / "P4到P5到P1连续缓存_正式_seed2448.json"
    phase10_path = CHAIN / "phase10到phase11食指中指稳夹拇指继续绕侧_seed2721.json"
    phase11_path = CHAIN / "phase11到phase12食指中指稳夹拇指完成绕侧_seed2722.json"
    phase18_path = CHAIN / "phase18中指继续撤回_bank60局部搜索_seed2891.json"
    strict_paths = [
        CHAIN / "phase17_无teacher_margin11成功策略_严格10mm_seed3040.json",
        CHAIN / "phase17_无teacher_严格10mm_多seed复核_seed3041.json",
        CHAIN / "phase17_无teacher_严格10mm_多seed复核_seed3042.json",
    ]

    manifest = load(manifest_path)
    overview = load(overview_path)
    p3p4 = load(p3p4_path)
    p4p5_a = load(p4p5_a_path)
    p4p5_b = load(p4p5_b_path)
    p4p5p1 = load(p4p5p1_path)
    phase10 = load(phase10_path)
    phase11 = load(phase11_path)
    phase18 = load(phase18_path)
    strict = [load(path) for path in strict_paths]

    canonical = [
        {
            "id": f"P{row['phase']}",
            "名称": row["name"],
            "接触": row["contact_set"],
            "可靠性": "D",
            "图像": f"../canonical_p1_p5_v9_P4拇指下托笔端/P{row['phase']}_realmesh.png",
        }
        for row in manifest["phases"]
    ]

    transitions = [
        {
            "id": "P1→P2",
            "语义": "拇/食/中继续控笔，无名指移动到笔下辅助",
            "等级": "C",
            "当前结论": "旧 B12 动态可用作 teacher，但末端接触不符合当前 P2。",
            "指标": "旧窄扰动 15/17；arrival cache 30；projected hold 0.59 s",
            "视图": "../../manus_mujoco_presets/b12_ring_under_recapture_m3_2_32/b12_ring_under_recapture.gif",
            "视图说明": "旧 ring-under 再捕获，属于历史语义",
        },
        {
            "id": "P2→P3",
            "语义": "无名指/拇指/中指控笔，食指移至外侧并由食/中夹持驱动",
            "等级": "X",
            "当前结论": "当前 P2 入口改动后尚无 PhysX 连续严格验证。",
            "指标": "旧无重置链只到 4.99° P3-entry；完整目标要求 >7°",
            "视图": "../../manus_mujoco_presets/continuous_p1_p2_p3_m3_2_34/continuous_P1_to_P2_to_P3.gif",
            "视图说明": "旧 P1→P2→P3-entry，无重置但非当前接触语义",
        },
        {
            "id": "P3→P4",
            "语义": "食指/中指发力夹笔并驱动绕拇指，拇指下方辅助",
            "等级": "B",
            "当前结论": "当前 PhysX 局部技能成立。",
            "指标": f"{p3p4['successes']}/{p3p4['episodes']} = {100*p3p4['success_rate']:.2f}%；平均穿透 {p3p4['mean_episode_max_penetration_mm']:.3f} mm",
            "视图": "../P3与P3到P4训练结果_v9正式评估/P3到制动P4代表成功轨迹.gif",
            "视图说明": "当前 P3→制动 P4 代表成功轨迹",
        },
        {
            "id": "P4→P5",
            "语义": "食/中稳夹后拇指下托并绕侧，形成拇/食夹持",
            "等级": "B",
            "当前结论": "同一 PhysX episode、无手/笔 reset 的局部级联成立。",
            "指标": f"seed2430 {100*p4p5_a['cascade_success_rate']:.2f}%；seed2431 {100*p4p5_b['cascade_success_rate']:.2f}%；均无掉笔",
            "视图": "../no_reset_chain/P4到P5无重置动态视图/P4到P5无重置代表轨迹.gif",
            "视图说明": "当前 P4→P5 无重置代表轨迹",
        },
        {
            "id": "P5→P1",
            "语义": "拇/食接管后中指撤回，再形成最终 P1 三指保持",
            "等级": "X",
            "当前结论": "后缀局部严格保持成立，但完整入口到最终 P1 仍未闭合。",
            "指标": f"phase16→17：3 seeds 均 11/1024、100 ms；phase18：0/{phase18['num_envs']}，最佳 {phase18['best_hold_ms']:.1f} ms",
            "视图": "../no_reset_chain/食指中指稳夹_phase14到15首次推进_动态视图_v3/P4到P5到P1无重置代表轨迹.gif",
            "视图说明": "接管后中指开始卸载，当前仍未完成",
        },
        {
            "id": "整圈闭合",
            "语义": "当前几何门下 P1→P2→P3→P4→P5→P1 一次无重置",
            "等级": "X",
            "当前结论": "没有当前语义下的整圈严格成功。",
            "指标": f"旧语义 P4→P5→P1 为 {p4p5p1['successes']}/{p4p5p1['episodes']} = {100*p4p5p1['success_rate']:.2f}%，仅作历史参考",
            "视图": "../no_reset_chain/P4到P5到P1无重置动态视图/P4到P5到P1无重置代表轨迹.gif",
            "视图说明": "旧几何门下后半圈无重置代表，不是当前整圈成功",
        },
    ]

    phase_gallery = [
        ("食指中指稳夹，拇指下方接近", "1_1 食指中指稳夹，拇指从下方接近.gif", "C"),
        ("拇指下托稳定", "2_2 拇指下托稳定.gif", "C"),
        ("拇指开始绕侧", "3_3 拇指开始绕侧.gif", "C"),
        ("拇指缓慢绕侧", "4_4 拇指缓慢绕侧.gif", "C"),
        ("拇指继续绕侧", "5_5 拇指继续绕侧.gif", "C"),
        ("旧 phase12：越侧不彻底", "6_6 旧phase12候选：越侧不彻底，需轴向对齐.gif", "X"),
        ("旧 phase13：未形成对侧夹持", "7_7 旧phase13候选：尚未形成对侧拇食指夹持.gif", "X"),
        ("旧 phase14：接管几何不合格", "8_8 旧phase14候选：接管几何不合格.gif", "X"),
        ("中指卸载：等待上游几何修复", "9_9 接管后中指卸载（上游几何修复前暂不验收）.gif", "X"),
    ]

    view_links = [
        ("当前 P1–P5 canonical", "../canonical_p1_p5_v9_P4拇指下托笔端/index.html", "D"),
        ("P3 与 P3→P4 正式评估", "../P3与P3到P4训练结果_v9正式评估/index.html", "B"),
        ("P4→P5 无重置", "../no_reset_chain/P4到P5无重置动态视图/index.html", "B"),
        ("P4→P5→P1 旧语义无重置", "../no_reset_chain/P4到P5到P1无重置动态视图/index.html", "C"),
        ("当前拇指下托到中指卸载九段", "../no_reset_chain/当前拇指下托越侧到中指卸载_完整分段视图_v5/index.html", "C"),
        ("拇指越侧几何门修正版", "../no_reset_chain/拇指彻底越侧与对侧夹持_几何门修正版_v6/index.html", "X"),
        ("食中稳夹与拇指接管", "../no_reset_chain/食指中指稳夹_拇指下托接管_PhysX动态视图_v4/index.html", "X"),
        ("phase14 拇食指接管", "../no_reset_chain/食指中指稳夹_phase14拇食指接管_动态视图_v2/index.html", "X"),
        ("phase14→15 中指开始卸载", "../no_reset_chain/食指中指稳夹_phase14到15首次推进_动态视图_v3/index.html", "X"),
        ("P4 夹持到拇指下托接触", "../no_reset_chain/P4夹持到拇指下托接触_连续动态视图/index.html", "X"),
        ("当前定向笔轴最深未完成轨迹", "../no_reset_chain/当前定向笔轴动态视图/index.html", "X"),
        ("旧 P4 到 P1 拇指接管 v2", "../no_reset_chain/P4到P1先夹持后拇指接管_动态视图_v2/index.html", "X"),
    ]

    sources = [
        source(manifest_path, "canonical P1-P5"),
        source(overview_path, "可靠数据模型 v1.2"),
        source(p3p4_path, "P3 到制动 P4"),
        source(p4p5_a_path, "P4 到 P5 无重置 seed2430"),
        source(p4p5_b_path, "P4 到 P5 无重置 seed2431"),
        source(p4p5p1_path, "旧 P4 到 P5 到 P1 无重置"),
        source(phase10_path, "拇指继续绕侧 phase10 到 11"),
        source(phase11_path, "拇指完成绕侧 phase11 到 12"),
        source(phase18_path, "phase18 中指撤回"),
        *[source(path, "phase16 到 17 严格复核") for path in strict_paths],
    ]

    model = {
        "数据模型版本": "2.0",
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "目标": "P1→P2→P3→P4→P5→P1 当前语义无重置闭环",
        "当前结论": "局部 P3→P4、P4→P5 成立；P1→P2/P2→P3 需按新 P2 重建；P5→P1 仅晚期后缀成立；整圈未完成。",
        "canonical": canonical,
        "transition": transitions,
        "关键阶段": {
            "phase10到11成功率": phase10["success_rate"],
            "phase11到12成功率": phase11["success_rate"],
            "phase16到17严格多种子": [
                {
                    "seed": row["seed"],
                    "成功": row["handoff_hold_diagnostics"]["strict_success_envs"],
                    "样本": row["num_envs"],
                    "保持_ms": row["handoff_hold_diagnostics"]["best_hold_ms"],
                }
                for row in strict
            ],
        },
        "可视化入口": [
            {"名称": name, "路径": path, "等级": level}
            for name, path, level in view_links
        ],
        "证据来源": sources,
    }
    (out / "P1到P5可靠数据模型.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    canonical_html = "".join(
        f'<article class="canon"><img src="{row["图像"]}"><div><header><b>{row["id"]}</b>{badge("D")}</header>'
        f'<h3>{esc(row["名称"])}</h3><p>接触：{esc(" / ".join(row["接触"]))}</p></div></article>'
        for row in canonical
    )
    transition_html = "".join(
        f'<article class="transition level{row["等级"]}"><div class="copy"><header><h3>{row["id"]}</h3>{badge(row["等级"])}</header>'
        f'<b>{esc(row["语义"])}</b><p>{esc(row["当前结论"])}</p><small>{esc(row["指标"])}</small></div>'
        f'<figure><img src="{row["视图"]}"><figcaption>{esc(row["视图说明"])}</figcaption></figure></article>'
        for row in transitions
    )
    gallery_root = "../no_reset_chain/拇指彻底越侧与对侧夹持_几何门修正版_v6"
    phase_html = "".join(
        f'<figure class="phase"><div>{badge(level)}<b>{esc(title)}</b></div><img src="{gallery_root}/{esc(filename)}"></figure>'
        for title, filename, level in phase_gallery
    )
    links_html = "".join(
        f'<a class="view lbox{level}" href="{path}">{badge(level)}<b>{esc(name)}</b><span>打开原始视图</span></a>'
        for name, path, level in view_links
    )
    evidence_html = "".join(
        f'<tr><td>{esc(row["用途"])}</td><td><code>{esc(row["路径"])}</code></td><td><code>{row["SHA256"][:12]}…</code></td></tr>'
        for row in sources
    )

    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dex5 P1-P5 可靠数据模型可视化总汇</title><style>
:root{{--bg:#0d1012;--panel:#15191c;--line:#343b40;--text:#edf1f2;--muted:#aeb7bd;--a:#4dcc76;--b:#55bddc;--c:#e7bd54;--d:#929da3;--x:#ee766b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:"Noto Sans CJK SC",sans-serif}}main{{max-width:1480px;margin:auto;padding:28px}}
h1{{font-size:30px;margin:0}}h2{{font-size:21px;margin:0 0 16px}}h3{{font-size:16px;margin:0}}p{{color:var(--muted);line-height:1.65}}section{{padding:24px 0;border-bottom:1px solid var(--line)}}
.top{{display:flex;justify-content:space-between;gap:20px;align-items:end}}.stamp{{color:var(--muted);font-size:13px}}.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.metric{{background:var(--panel);padding:14px;border-top:3px solid var(--d)}}.metric b{{font-size:23px;display:block}}.metric span,small,figcaption{{color:var(--muted);font-size:12px}}
.badge{{display:inline-block;border:1px solid;border-radius:4px;padding:3px 7px;font-size:11px;white-space:nowrap}}.lA{{color:var(--a)}}.lB{{color:var(--b)}}.lC{{color:var(--c)}}.lD{{color:var(--d)}}.lX{{color:var(--x)}}
.cycle{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}}.cycle div{{background:var(--panel);padding:12px;border-top:3px solid var(--line);min-height:88px}}.cycle b{{display:block;margin-bottom:6px}}.cycle span{{color:var(--muted);font-size:12px;line-height:1.45}}
.canon-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}article.canon{{background:var(--panel);border:1px solid var(--line)}}.canon img{{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}}.canon div{{padding:12px}}header{{display:flex;justify-content:space-between;gap:10px;align-items:center}}.canon p{{font-size:12px;margin-bottom:0}}
.transitions{{display:grid;gap:14px}}article.transition{{display:grid;grid-template-columns:minmax(320px,.8fr) minmax(480px,1.2fr);background:var(--panel);border-left:4px solid var(--d)}}article.levelA{{border-color:var(--a)}}article.levelB{{border-color:var(--b)}}article.levelC{{border-color:var(--c)}}article.levelX{{border-color:var(--x)}}
.copy{{padding:18px}}.copy>b{{display:block;margin-top:14px}}.copy small{{display:block;border-top:1px solid var(--line);padding-top:10px}}figure{{margin:0}}.transition figure{{padding:12px}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding-top:8px}}
.phases{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure.phase{{background:var(--panel);border:1px solid var(--line);padding:10px}}.phase div{{display:flex;align-items:center;gap:8px;margin-bottom:9px}}.phase b{{font-size:13px}}
.views{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}a.view{{display:flex;align-items:center;gap:10px;background:var(--panel);border:1px solid var(--line);padding:12px;text-decoration:none;color:var(--text)}}.view b{{flex:1;font-size:13px}}.view span{{color:var(--b);font-size:12px}}
.warning{{background:#1b1717;border-left:4px solid var(--x);padding:14px;color:#f3c8c3}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:9px;text-align:left;border-bottom:1px solid var(--line)}}th{{background:var(--panel)}}code{{color:#b8c7cd;word-break:break-all}}
@media(max-width:980px){{.summary,.canon-grid,.views{{grid-template-columns:repeat(2,1fr)}}.cycle,.phases{{grid-template-columns:repeat(2,1fr)}}article.transition{{grid-template-columns:1fr}}}}
@media(max-width:600px){{main{{padding:16px}}.top{{align-items:start;flex-direction:column}}.summary,.canon-grid,.views,.cycle,.phases{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="top"><div><h1>Dex5 P1–P5 可靠数据模型可视化总汇</h1><p>canonical、transition、训练入口、动态视图和失败边界统一索引</p></div><div class="stamp">模型 v2.0 · 2026-07-22 · PhysX 为当前主线</div></div>
<section><div class="summary"><div class="metric"><b>5</b><span>canonical，均为 D 级姿态先验</span></div><div class="metric"><b>2</b><span>当前 B 级动态段：P3→P4、P4→P5</span></div><div class="metric"><b>1</b><span>A 级后缀：phase16→17 严格保持</span></div><div class="metric"><b>未闭环</b><span>当前 P1→P5→P1 无重置整圈</span></div></div></section>
<section><h2>闭环证据地图</h2><div class="cycle"><div><b>P1→P2</b><span>C：旧 teacher 可复用，当前 P2 接触门待重建</span></div><div><b>P2→P3</b><span>X：新 P2 入口后尚无 PhysX 严格连续验证</span></div><div><b>P3→P4</b><span>B：77.05% 当前 PhysX 局部技能</span></div><div><b>P4→P5</b><span>B：同 episode 无 reset 61.3%–64.2%</span></div><div><b>P5→P1</b><span>X：后缀 A，但 phase18 和完整入口未闭合</span></div><div><b>回到 P1</b><span>X：当前几何门下无整圈严格成功</span></div></div></section>
<section><h2>P1–P5 canonical</h2><p>canonical 只定义关节、笔姿态与预期接触，不用静态渲染冒充动态物理成功。</p><div class="canon-grid">{canonical_html}</div></section>
<section><h2>逐段动态证据</h2><div class="transitions">{transition_html}</div></section>
<section><h2>P4→P5→P1 细分 phase 视图</h2><div class="warning">前五段可作为训练入口；旧 phase12–14 在新的“拇食指对侧夹持”几何门下不合格，因此作为反例展示。图片存在不等于阶段通过。</div><div class="phases" style="margin-top:14px">{phase_html}</div></section>
<section><h2>全部可靠可视化入口</h2><div class="views">{links_html}</div></section>
<section><h2>跨引擎与物理风险</h2><div class="warning">MuJoCo/PhysX 100 ms 笔位置误差约 {overview['跨引擎风险']['100ms位置误差_mm']:.2f} mm、姿态误差约 {overview['跨引擎风险']['100ms姿态误差_deg']:.2f}°。MuJoCo 视图用于姿态设计和历史诊断；当前技能成功以 PhysX 独立评测为准。</div></section>
<section><h2>证据溯源</h2><div style="overflow:auto"><table><thead><tr><th>用途</th><th>原始文件</th><th>SHA-256</th></tr></thead><tbody>{evidence_html}</tbody></table></div><p><a href="P1到P5可靠数据模型.json" style="color:var(--b)">打开机器可读数据模型 JSON</a></p></section>
</main></body></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    print(out / "index.html")


if __name__ == "__main__":
    main()
