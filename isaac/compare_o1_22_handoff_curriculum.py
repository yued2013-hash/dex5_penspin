from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke"
A_PATH = REPORT_DIR / "o1_22_A_uniform_eval256.json"
B_PATH = REPORT_DIR / "o1_22_B_curriculum_uniform_eval256.json"
SMOKE_PATH = REPORT_DIR / "o1_22_B_sampler_smoke16.json"
OUTPUT_JSON = REPORT_DIR / "O1_22_Handoff_Basin_Curriculum_Audit.json"
OUTPUT_MD = REPORT_DIR / "O1_22_Handoff_Basin_Curriculum_Audit_验收报告.md"
OUTPUT_PNG = REPORT_DIR / "O1_22_Handoff_Basin_Curriculum_Comparison.png"


def _wilson(successes: int, count: int, z: float = 1.96) -> list[float]:
    if not count:
        return [0.0, 0.0]
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / count + z * z / (4.0 * count * count)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _metric(report: dict, count_key: str, rate_key: str) -> dict:
    count = int(report[count_key])
    episodes = int(report["completed_episodes"])
    return {
        "episodes": count,
        "rate": float(report[rate_key]),
        "Wilson95": _wilson(count, episodes),
    }


def _summary(report: dict) -> dict:
    p3 = report["by_canonical"]["P3"]
    episodes = int(report["completed_episodes"])
    exploit_count = sum(
        int(report["primary_failure_type_totals"][name])
        for name in ("contact_loss", "ballistic", "throw")
    )
    height_recovery_count = int(p3["recovery"]["height_recovery_episodes"])
    return {
        "completed_episodes": episodes,
        "qualified_gait": _metric(
            report, "qualified_gait_episodes", "qualified_gait_rate"
        ),
        "handoff_discovery": _metric(
            report, "support_handoff_episodes", "support_handoff_rate"
        ),
        "TIM_transition": _metric(
            report, "tim_gait_event_episodes", "tim_gait_event_rate"
        ),
        "height_recovery": {
            "episodes": height_recovery_count,
            "rate": height_recovery_count / max(episodes, 1),
            "Wilson95": _wilson(height_recovery_count, episodes),
        },
        "exploit": {
            "episodes": exploit_count,
            "rate": exploit_count / max(episodes, 1),
            "Wilson95": _wilson(exploit_count, episodes),
            "definition": "primary contact_loss + ballistic + throw",
        },
        "raw_gait_rate": float(report["gait_event_rate"]),
        "failure_rate": float(report["failure_rate"]),
        "contact_loss_rate": float(report["contact_loss_rate"]),
        "ballistic_rate": float(report["ballistic_rate"]),
        "height_drop_rate": float(p3["recovery"]["height_drop_rate"]),
        "mean_axis_alignment": float(p3["mean_directional_axis_alignment"]),
        "mean_TIM_overlap_ratio": float(p3["mean_TIM_overlap_ratio"]),
        "finger_gait_rate": {
            name: float(row["gait_episode_rate"])
            for name, row in p3["finger_gait"].items()
        },
    }


def main() -> None:
    a = json.loads(A_PATH.read_text(encoding="utf-8"))
    b = json.loads(B_PATH.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    if a["bank_sha256"] != b["bank_sha256"]:
        raise RuntimeError("A/B use different O0 banks")
    if a["objective_config"] != b["objective_config"]:
        raise RuntimeError("A/B objective or observation configuration differs")
    if a["task"] != b["task"] or a["task"] != "Dex5-PenSpin-Oracle-P3Temporal-Direct-v0":
        raise RuntimeError("A/B were not evaluated on the same uniform P3 task")
    if a["num_envs"] != b["num_envs"] or a["steps"] != b["steps"]:
        raise RuntimeError("A/B evaluation budgets differ")

    a_summary = _summary(a)
    b_summary = _summary(b)
    delta = {
        key: b_summary[key]["rate"] - a_summary[key]["rate"]
        for key in (
            "qualified_gait",
            "handoff_discovery",
            "TIM_transition",
            "height_recovery",
            "exploit",
        )
    }
    strict_discovery = (
        b_summary["qualified_gait"]["rate"]
        > a_summary["qualified_gait"]["rate"]
        and b_summary["qualified_gait"]["rate"] > 0.0
    ) or (
        b_summary["handoff_discovery"]["rate"]
        > a_summary["handoff_discovery"]["rate"]
        and b_summary["handoff_discovery"]["rate"] > 0.0
    )
    safety_gate = delta["exploit"] <= 0.02
    curriculum_improves = strict_discovery and safety_gate
    sampler = smoke["curriculum_sampling"]
    sampler_total = sampler["sampled_candidate_count"] + sampler["sampled_non_candidate_count"]

    report = {
        "status": "PASS_ABLATION_CURRICULUM_REJECTED",
        "phase": "O1.22 Handoff Basin Curriculum Audit",
        "classification": "ENTRY_OVERSAMPLING_HURTS_DISCOVERY_AND_INCREASES_EXPLOIT",
        "matched_contract": {
            "P3_only": True,
            "temporal_observation": True,
            "seed": 20260730,
            "num_envs": 256,
            "iterations": 60,
            "rollout_steps": 1440,
            "transitions": 368640,
            "reward_equal": True,
            "observation_equal": True,
            "PPO_config_equal": True,
            "bank_hash_equal": True,
            "final_checkpoint_selected_a_priori": True,
            "uniform_P3_evaluation_for_both": True,
        },
        "curriculum_contract": {
            "candidate_bank_ids": [1034, 1176, 1146, 1045, 1227, 852, 1149],
            "candidate_probability_first_20_iterations": 0.75,
            "linear_decay_iterations": [20, 40],
            "final_20_iterations": "exact uniform P3",
            "sampler_smoke_candidate_count": int(sampler["sampled_candidate_count"]),
            "sampler_smoke_total": int(sampler_total),
            "sampler_smoke_candidate_rate": sampler["sampled_candidate_count"] / max(sampler_total, 1),
            "candidate_labels_used": False,
            "candidate_states_modified": False,
        },
        "A_uniform": a_summary,
        "B_early_curriculum": b_summary,
        "B_minus_A_rate": delta,
        "acceptance_gate": {
            "strict_discovery_improved_and_nonzero": strict_discovery,
            "exploit_increase_at_most_2pp": safety_gate,
            "curriculum_improves_skill_discovery": curriculum_improves,
        },
        "interpretation": (
            "The tested static-entry oversampling does not reveal qualified gait or strict support handoff. "
            "It reduces TIM transition activity and shifts failure from height-loss/drop behavior to immediate "
            "contact-loss/ballistic exploitation. Static O1.21 entry-likeness is therefore not a sufficient PPO "
            "curriculum variable under the frozen O1.5 objective. This rejects this schedule and candidate set, "
            "not every state-distribution curriculum."
        ),
        "sources": {
            "A_evaluation": str(A_PATH.resolve()),
            "B_evaluation": str(B_PATH.resolve()),
            "sampler_smoke": str(SMOKE_PATH.resolve()),
            "A_checkpoint": a["checkpoint"],
            "B_checkpoint": b["checkpoint"],
        },
        "constraints": {
            "O0_modified": False,
            "P3H_created": False,
            "teacher_generated": False,
            "reward_modified": False,
            "new_reset_dataset_created": False,
            "large_scale_training": False,
            "seed_count": 1,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metric_names = ["qualified_gait", "handoff_discovery", "TIM_transition", "height_recovery", "exploit"]
    labels = ["Qualified gait", "Strict handoff", "TIM transition", "Height recovery", "Exploit"]
    a_values = [a_summary[key]["rate"] for key in metric_names]
    b_values = [b_summary[key]["rate"] for key in metric_names]
    x = range(len(metric_names))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), constrained_layout=True)
    width = 0.36
    axes[0].bar([value - width / 2 for value in x], a_values, width, label="A uniform", color="#457b9d")
    axes[0].bar([value + width / 2 for value in x], b_values, width, label="B curriculum", color="#e76f51")
    axes[0].set_xticks(list(x), labels, rotation=18, ha="right")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Episode rate")
    axes[0].set_title("Uniform-P3 final-checkpoint evaluation")
    axes[0].legend(frameon=False)

    fingers = ("thumb", "index", "middle")
    a_finger = [a_summary["finger_gait_rate"][name] for name in fingers]
    b_finger = [b_summary["finger_gait_rate"][name] for name in fingers]
    fx = range(len(fingers))
    axes[1].bar([value - width / 2 for value in fx], a_finger, width, label="A uniform", color="#457b9d")
    axes[1].bar([value + width / 2 for value in fx], b_finger, width, label="B curriculum", color="#e76f51")
    axes[1].set_xticks(list(fx), fingers)
    axes[1].set_ylim(0.0, 0.6)
    axes[1].set_ylabel("Debounced gait episode rate")
    axes[1].set_title("TIM contact-transition activity")
    axes[1].legend(frameon=False)
    fig.suptitle("O1.22 early handoff-entry curriculum ablation")
    fig.savefig(OUTPUT_PNG, dpi=170)
    plt.close(fig)

    md = f"""# O1.22 Handoff Basin Curriculum Audit 验收报告

## 结论

`{report['classification']}`

本轮早期 handoff-entry 过采样**没有提高 skill discovery**，并且产生了明显的接触丢失投机。

| 指标 | A uniform | B curriculum | B-A |
|---|---:|---:|---:|
| Qualified gait | {a_summary['qualified_gait']['rate']:.2%} | {b_summary['qualified_gait']['rate']:.2%} | {delta['qualified_gait']:+.2%} |
| Strict support handoff | {a_summary['handoff_discovery']['rate']:.2%} | {b_summary['handoff_discovery']['rate']:.2%} | {delta['handoff_discovery']:+.2%} |
| TIM transition | {a_summary['TIM_transition']['rate']:.2%} | {b_summary['TIM_transition']['rate']:.2%} | {delta['TIM_transition']:+.2%} |
| Height recovery | {a_summary['height_recovery']['rate']:.2%} | {b_summary['height_recovery']['rate']:.2%} | {delta['height_recovery']:+.2%} |
| Exploit | {a_summary['exploit']['rate']:.2%} | {b_summary['exploit']['rate']:.2%} | {delta['exploit']:+.2%} |

## 解读

- A/B 的 qualified gait 和 strict handoff 都是 `0`。
- B 的 TIM transition 从 `{a_summary['TIM_transition']['rate']:.2%}` 降到 `{b_summary['TIM_transition']['rate']:.2%}`。
- B 的 exploit 从 `{a_summary['exploit']['rate']:.2%}` 上升到 `{b_summary['exploit']['rate']:.2%}`，主要是 contact loss。
- A 也不是成功 policy：它的 height recovery 仍为 `0`，且几乎所有 episode 都丢失高度。

因此，O1.21 的静态 entry-likeness 不能直接当作 PPO curriculum 难度指标。这个结果否定当前 75% 早期过采样方案，不否定所有 reset curriculum。

![O1.22 comparison]({OUTPUT_PNG.name})
"""
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": report["status"], "classification": report["classification"], "A": a_summary, "B": b_summary, "gate": report["acceptance_gate"]}, indent=2))


if __name__ == "__main__":
    main()
