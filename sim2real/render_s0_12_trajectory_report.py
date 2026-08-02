from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


CONTROL_HZ = 60.0
CONTACT_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
CONTACT_COLORS = ("#c9413a", "#2574a9", "#2f855a", "#d28b16", "#7651a8")
SERIES_COLORS = (
    "#1769aa",
    "#c9413a",
    "#2f855a",
    "#d28b16",
    "#7651a8",
    "#16858c",
    "#bb4d88",
    "#68737d",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[:, 0] * right[:, 0] - np.sum(
        left[:, 1:] * right[:, 1:], axis=1
    )
    vector = (
        left[:, :1] * right[:, 1:]
        + right[:, :1] * left[:, 1:]
        + np.cross(left[:, 1:], right[:, 1:])
    )
    return np.concatenate((scalar[:, None], vector), axis=1)


def _quaternion_progress(
    quaternion: np.ndarray, initial_quaternion: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    conjugate = np.repeat(initial_quaternion[None, :], len(quaternion), axis=0)
    conjugate[:, 1:] *= -1.0
    relative = _quat_mul(quaternion, conjugate)
    relative /= np.linalg.norm(relative, axis=1, keepdims=True)
    relative = np.where(relative[:, :1] < 0.0, -relative, relative)
    vector_norm = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(
        vector_norm, np.clip(relative[:, 0], 1.0e-9, None)
    )
    axis_z = np.divide(
        relative[:, 3],
        vector_norm,
        out=np.zeros_like(vector_norm),
        where=vector_norm > 1.0e-9,
    )
    return np.rad2deg(angle * axis_z), axis_z


def _episode_arrays(data: dict[str, np.ndarray], row: dict) -> dict[str, np.ndarray]:
    start = int(row["row_start"])
    stop = int(row["row_stop"])
    result = {
        key: value[start:stop]
        for key, value in data.items()
        if value.ndim > 0 and value.shape[0] == data["pen_pose"].shape[0]
    }
    rotation_deg, alignment = _quaternion_progress(
        result["pen_pose_next"][:, 3:7], result["pen_pose"][0, 3:7]
    )
    result["rotation_deg"] = rotation_deg
    result["axis_alignment"] = alignment
    result["height_loss_mm"] = 1000.0 * (
        result["pen_pose"][0, 2] - result["pen_pose_next"][:, 2]
    )
    result["time_s"] = np.arange(len(rotation_deg), dtype=np.float64) / CONTROL_HZ
    result["mapped_delta"] = np.concatenate(
        (
            np.zeros((1, 20), dtype=np.float64),
            np.diff(result["adapter_projected_target"], axis=0),
        ),
        axis=0,
    )
    result["mapped_speed"] = np.max(np.abs(result["mapped_delta"]), axis=1) * CONTROL_HZ
    velocity = result["pen_angular_velocity"]
    contacts = result["contact_state_next"]
    result["strict_hold"] = (
        (rotation_deg >= 30.0)
        & (rotation_deg <= 40.0)
        & (alignment >= 0.70)
        & (np.abs(velocity[:, 2]) <= 2.5)
        & (np.linalg.norm(velocity[:, :2], axis=1) <= 2.0)
        & (result["height_loss_mm"] <= 5.0)
        & contacts[:, :3].all(axis=1)
        & (~contacts[:, 3:].any(axis=1))
    )
    return result


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(True, axis="y", color="#dfe4e8", linewidth=0.7)
    axis.tick_params(colors="#4a5660", labelsize=8)


def _save_overview(
    output: Path, selected: list[dict], episodes: list[dict[str, np.ndarray]]
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=170)
    tab20 = plt.get_cmap("tab20")
    for index, (row, episode) in enumerate(zip(selected, episodes, strict=True)):
        progress = np.linspace(0.0, 1.0, len(episode["rotation_deg"]))
        axes[0, 0].plot(
            progress,
            episode["rotation_deg"],
            color=tab20(index),
            linewidth=1.25,
            alpha=0.88,
        )
    axes[0, 0].axhspan(30.0, 40.0, color="#2f855a", alpha=0.10)
    axes[0, 0].axhline(35.0, color="#2f855a", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Quaternion target-axis rotation | selected 20")
    axes[0, 0].set_xlabel("Normalized trajectory progress")
    axes[0, 0].set_ylabel("Rotation (deg)")

    ranks = np.arange(1, len(selected) + 1)
    finals = [float(row["final_target_quaternion_rotation_deg"]) for row in selected]
    axes[0, 1].bar(ranks, finals, color=[tab20(i) for i in range(len(selected))])
    axes[0, 1].axhspan(30.0, 40.0, color="#2f855a", alpha=0.08)
    axes[0, 1].set_title("Final quaternion rotation by rank")
    axes[0, 1].set_xlabel("Candidate rank")
    axes[0, 1].set_ylabel("Final rotation (deg)")
    axes[0, 1].set_xticks(ranks)

    tim = np.asarray([float(row["tim_contact_ratio"]) * 100.0 for row in selected])
    delta = np.asarray(
        [float(row["maximum_mapped_target_delta_rad"]) for row in selected]
    )
    hold = np.asarray([float(row["strict_hold_ms"]) for row in selected])
    scatter = axes[1, 0].scatter(
        delta,
        tim,
        c=hold,
        cmap=ListedColormap(("#d28b16", "#1769aa", "#2f855a")),
        s=75,
        edgecolors="#ffffff",
        linewidths=0.8,
    )
    for rank, x_value, y_value in zip(ranks, delta, tim, strict=True):
        axes[1, 0].annotate(
            str(rank), (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=7
        )
    axes[1, 0].axhline(75.0, color="#c9413a", linestyle="--", linewidth=1.0)
    axes[1, 0].axvline(0.01, color="#c9413a", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Contact continuity vs command smoothness")
    axes[1, 0].set_xlabel("Maximum mapped target delta (rad)")
    axes[1, 0].set_ylabel("TIM contact ratio (%)")
    colorbar = figure.colorbar(scatter, ax=axes[1, 0], pad=0.02)
    colorbar.set_label("Strict hold (ms)")

    scores = np.asarray([float(row["score"]) for row in selected])
    colors = ["#1769aa" if value >= np.median(scores) else "#68737d" for value in scores]
    axes[1, 1].barh(ranks[::-1], scores[::-1], color=colors[::-1])
    axes[1, 1].set_title("Offline replay suitability score")
    axes[1, 1].set_xlabel("Score")
    axes[1, 1].set_ylabel("Candidate rank")
    axes[1, 1].set_yticks(ranks)

    for axis in axes.flat:
        _style_axis(axis)
    figure.suptitle("Dex5 S0.12 | Oracle replay candidate overview", fontsize=17, y=0.995)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor="#ffffff")
    plt.close(figure)


def _save_episode_chart(path: Path, rank: int, row: dict, episode: dict[str, np.ndarray]) -> None:
    time_s = episode["time_s"]
    figure, axes = plt.subplots(5, 1, figsize=(12, 11), dpi=150, sharex=True)

    axes[0].plot(time_s, episode["rotation_deg"], color=SERIES_COLORS[0], linewidth=2.2)
    axes[0].axhspan(30.0, 40.0, color=SERIES_COLORS[2], alpha=0.11)
    axes[0].axhline(35.0, color=SERIES_COLORS[2], linestyle="--", linewidth=1.0)
    axes[0].fill_between(
        time_s,
        0.0,
        1.0,
        where=episode["strict_hold"],
        transform=axes[0].get_xaxis_transform(),
        color=SERIES_COLORS[3],
        alpha=0.16,
        label="Strict hold",
    )
    axes[0].set_ylabel("Rotation (deg)")
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].plot(time_s, episode["axis_alignment"], color=SERIES_COLORS[4], linewidth=1.8, label="Axis alignment")
    axes[1].axhline(0.70, color=SERIES_COLORS[1], linestyle="--", linewidth=1.0, label="Minimum 0.70")
    height_axis = axes[1].twinx()
    height_axis.plot(time_s, episode["height_loss_mm"], color=SERIES_COLORS[3], linewidth=1.4, label="Height loss")
    height_axis.axhline(5.0, color=SERIES_COLORS[1], linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("Axis alignment")
    height_axis.set_ylabel("Height loss (mm)", color=SERIES_COLORS[3])
    axes[1].legend(loc="upper left", frameon=False)

    force = episode["contact_force_norm_next"]
    for name, color, values in zip(CONTACT_NAMES, CONTACT_COLORS, force.T, strict=True):
        axes[2].plot(time_s, values, color=color, linewidth=1.25, label=name)
    axes[2].axhline(0.03, color="#17202a", linestyle="--", linewidth=0.9)
    axes[2].set_ylabel("Contact force (N)")
    axes[2].legend(loc="upper right", ncol=5, frameon=False, fontsize=8)

    axes[3].imshow(
        episode["contact_state_next"].T.astype(float),
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(time_s[0], time_s[-1] + 1.0 / CONTROL_HZ, 4.5, -0.5),
        cmap=ListedColormap(("#edf0f2", "#2f855a")),
        vmin=0,
        vmax=1,
    )
    axes[3].set_yticks(range(5), CONTACT_NAMES)
    axes[3].set_ylabel("Contact")

    axes[4].plot(time_s, episode["mapped_speed"], color=SERIES_COLORS[1], linewidth=1.7, label="Peak joint target speed")
    axes[4].axhline(0.60, color=SERIES_COLORS[3], linestyle="--", linewidth=1.0, label="0.01 rad / 60 Hz gate")
    axes[4].set_ylabel("Mapped speed (rad/s)")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(loc="upper right", frameon=False)

    for axis in axes:
        _style_axis(axis)
    figure.suptitle(
        f"Rank {rank:02d} | episode {row['episode_index']} | source {row['source_trajectory_id']}",
        fontsize=15,
        y=0.995,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="#ffffff")
    plt.close(figure)


def _candidate_table(selected: list[dict]) -> str:
    rows = []
    for rank, row in enumerate(selected, 1):
        rows.append(
            "<tr>"
            f"<td>{rank:02d}</td>"
            f"<td><a href=\"#trajectory-{rank:02d}\">{row['episode_index']}</a></td>"
            f"<td><code>{row['source_trajectory_id']}</code></td>"
            f"<td>{row['peak_target_quaternion_rotation_deg']:.2f}°</td>"
            f"<td>{row['strict_hold_ms']:.1f} ms</td>"
            f"<td>{row['tim_contact_ratio'] * 100.0:.1f}%</td>"
            f"<td>{row['maximum_mapped_target_delta_rad']:.5f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _gallery(selected: list[dict], assets: list[str]) -> str:
    sections = []
    for rank, (row, asset) in enumerate(zip(selected, assets, strict=True), 1):
        sections.append(
            f"""
<article class="trajectory" id="trajectory-{rank:02d}">
  <div class="trajectory-head">
    <div><span class="rank">#{rank:02d}</span><h2>Episode {row['episode_index']}</h2></div>
    <dl>
      <div><dt>四元数旋转</dt><dd>{row['final_target_quaternion_rotation_deg']:.2f}°</dd></div>
      <div><dt>严格保持</dt><dd>{row['strict_hold_ms']:.1f} ms</dd></div>
      <div><dt>TIM 接触</dt><dd>{row['tim_contact_ratio'] * 100.0:.1f}%</dd></div>
      <div><dt>最大单步目标</dt><dd>{row['maximum_mapped_target_delta_rad']:.5f} rad</dd></div>
    </dl>
  </div>
  <img src="{html.escape(asset)}" loading="lazy" alt="episode {row['episode_index']} trajectory chart">
  <p class="provenance">source <code>{row['source_trajectory_id']}</code> · bank <code>{row['bank_id']}</code> · state <code>{row['state_id']}</code></p>
</article>"""
        )
    return "".join(sections)


def _write_html(output: Path, manifest: dict, assets: list[str]) -> None:
    selected = manifest["selected_trajectories"]
    peak = [float(row["peak_target_quaternion_rotation_deg"]) for row in selected]
    tim = [float(row["tim_contact_ratio"]) * 100.0 for row in selected]
    delta = [float(row["maximum_mapped_target_delta_rad"]) for row in selected]
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dex5 S0.12 Oracle Replay 候选轨迹</title>
<style>
:root{{--ink:#17202a;--muted:#59656f;--line:#d7dde2;--paper:#f4f6f7;--white:#fff;--blue:#1769aa;--green:#2f855a;--red:#c9413a;--amber:#d28b16;--violet:#7651a8}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;letter-spacing:0}}
header{{background:var(--white);border-bottom:1px solid var(--line);padding:34px max(24px,calc((100vw - 1180px)/2)) 28px}}header h1{{margin:0 0 8px;font-size:30px;letter-spacing:0}}header p{{max-width:900px;margin:5px 0;color:var(--muted)}}.status{{display:inline-flex;align-items:center;gap:8px;color:var(--green);font-weight:700;font-size:13px}}.status::before{{content:"";width:9px;height:9px;border-radius:50%;background:var(--green)}}
main{{max-width:1180px;margin:0 auto;padding:26px 24px 64px}}.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:28px}}.kpi{{background:var(--white);padding:18px}}.kpi strong{{display:block;font-size:24px}}.kpi span{{color:var(--muted);font-size:13px}}
.band{{padding:25px 0;border-top:1px solid var(--line)}}h2{{letter-spacing:0}}.band h2{{margin:0 0 13px;font-size:21px}}.overview{{display:block;width:100%;background:var(--white);border:1px solid var(--line)}}
.table-wrap{{overflow:auto;background:var(--white);border:1px solid var(--line)}}table{{width:100%;border-collapse:collapse;min-width:850px}}th,td{{padding:11px 13px;text-align:left;border-bottom:1px solid #e5e9ec;white-space:nowrap}}th{{font-size:12px;color:var(--muted);background:#f9fafb}}td a{{color:var(--blue);font-weight:700;text-decoration:none}}code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}}
.trajectory{{padding:30px 0 36px;border-top:1px solid var(--line);scroll-margin-top:12px}}.trajectory-head{{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;margin-bottom:14px}}.trajectory-head>div{{display:flex;align-items:center;gap:10px}}.trajectory h2{{font-size:20px;margin:0}}.rank{{display:inline-grid;place-items:center;width:42px;height:30px;background:var(--blue);color:#fff;font-weight:800;border-radius:4px}}dl{{display:grid;grid-template-columns:repeat(4,minmax(105px,1fr));gap:16px;margin:0}}dl div{{border-left:3px solid var(--line);padding-left:10px}}dt{{font-size:11px;color:var(--muted)}}dd{{margin:1px 0 0;font-weight:750}}.trajectory img{{display:block;width:100%;aspect-ratio:12/11;object-fit:contain;background:var(--white);border:1px solid var(--line)}}.provenance{{margin:8px 0 0;color:var(--muted)}}
.note{{border-left:4px solid var(--amber);padding:13px 16px;background:#fff9e9;margin-top:20px}}footer{{color:var(--muted);font-size:12px;padding-top:22px;border-top:1px solid var(--line)}}
@media(max-width:820px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.trajectory-head{{align-items:flex-start;flex-direction:column}}dl{{width:100%;grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:520px){{header h1{{font-size:25px}}main{{padding-left:14px;padding-right:14px}}.kpis{{grid-template-columns:1fr 1fr}}.kpi{{padding:13px}}.kpi strong{{font-size:20px}}dl{{grid-template-columns:1fr 1fr;gap:10px}}.trajectory img{{aspect-ratio:auto}}}}
</style>
</head>
<body>
<header>
  <div class="status">OFFLINE SELECTION PASS</div>
  <h1>Dex5 S0.12 Oracle Replay 候选轨迹</h1>
  <p>首批 20 条轨迹来自 20 个不同 train source。旋转进度完全由同步记录的笔四元数计算，不使用瞬时 PhysX 角速度积分替代姿态变化。</p>
  <p>任务范围：P3 起始抓持 → 约 35° 目标轴旋转 → 制动并严格保持至少 150 ms。不是完整 PenSpin，也不代表真机已经成功。</p>
</header>
<main>
  <section class="kpis">
    <div class="kpi"><strong>20 / 20</strong><span>不同 source</span></div>
    <div class="kpi"><strong>{min(peak):.2f}–{max(peak):.2f}°</strong><span>实际四元数峰值旋转</span></div>
    <div class="kpi"><strong>{min(tim):.1f}–{max(tim):.1f}%</strong><span>TIM 接触时间占比</span></div>
    <div class="kpi"><strong>{min(delta):.5f}–{max(delta):.5f}</strong><span>最大映射单步目标 rad</span></div>
  </section>
  <section class="band"><h2>候选集合总览</h2><img class="overview" src="assets/overview.png" alt="selected trajectory overview"></section>
  <section class="band"><h2>20 条候选索引</h2><div class="table-wrap"><table><thead><tr><th>排名</th><th>Episode</th><th>Source</th><th>旋转</th><th>保持</th><th>TIM</th><th>最大单步目标</th></tr></thead><tbody>{_candidate_table(selected)}</tbody></table></div></section>
  <div class="note">明日真机测试应从排名前 3 条、0.12 rad/s 档开始。页面中的成功是仿真成功；真实成功仍需外部视频或姿态追踪确认笔实际旋转 ≥30°。</div>
  {_gallery(selected, assets)}
  <footer>rollout SHA256: <code>{manifest['rollout_sha256']}</code><br>selection manifest SHA256: <code>{_sha256(Path(manifest['_manifest_path']))}</code></footer>
</main>
</body>
</html>"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_OFFLINE_SELECTION":
        raise RuntimeError("selection manifest is not accepted")
    if _sha256(args.rollout) != manifest["rollout_sha256"]:
        raise RuntimeError("rollout hash differs from selection manifest")
    selected = manifest["selected_trajectories"]
    if len(selected) != 20:
        raise RuntimeError("visual report expects the frozen 20-trajectory batch")

    with np.load(args.rollout, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    episodes = [_episode_arrays(data, row) for row in selected]
    output = args.output_dir.resolve()
    assets_dir = output / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    _save_overview(assets_dir / "overview.png", selected, episodes)
    assets: list[str] = []
    for rank, (row, episode) in enumerate(zip(selected, episodes, strict=True), 1):
        name = f"trajectory_{rank:02d}_episode_{row['episode_index']:04d}.png"
        _save_episode_chart(assets_dir / name, rank, row, episode)
        assets.append(f"assets/{name}")

    manifest_for_html = {**manifest, "_manifest_path": str(args.selection.resolve())}
    _write_html(output / "index.html", manifest_for_html, assets)
    visual_manifest = {
        "schema": "dex5-s0.12-trajectory-visual-report-v1",
        "status": "PASS",
        "rollout": str(args.rollout.resolve()),
        "rollout_sha256": _sha256(args.rollout),
        "selection": str(args.selection.resolve()),
        "selection_sha256": _sha256(args.selection),
        "trajectory_count": len(selected),
        "angle_source": "relative pen quaternion target-axis component",
        "assets": ["assets/overview.png", *assets],
        "index": str((output / "index.html").resolve()),
        "hardware_commands_issued": False,
    }
    (output / "visualization_manifest.json").write_text(
        json.dumps(visual_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(visual_manifest, indent=2))


if __name__ == "__main__":
    main()
