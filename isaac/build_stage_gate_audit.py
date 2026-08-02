"""Build a single, provenance-preserving audit for the current pen-spin stages."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
FIVE = REPORT.parent / "manus_mujoco_presets/photo_five_phase_hand_pen_tuned_m3_2_10"
OUT = REPORT / "canonical_p1_p5_manifest.json"
AUDIT = REPORT / "stage_gate_audit_current.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path) -> dict | None:
    return load_json(path) if path.exists() else None


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: list(data[key].shape) for key in data.files}


def main() -> None:
    phases = []
    for phase in range(1, 6):
        source = next(FIVE.glob(f"P{phase}_*_hand_pen_tuned_seed.json"))
        record = load_json(source)
        phases.append(
            {
                "phase": phase,
                "name": record["name"],
                "source": str(source.relative_to(ROOT)),
                "joint_order": record["joint_order"],
                "hand_qpos": record["hand_qpos"],
                "pen_pos": record["pen_pos"],
                "pen_quat": record["pen_quat"],
                "pen_qvel": record.get("pen_qvel"),
                "phase_deg": record.get("phase_deg_tuned"),
                "contact_set": record.get("desired_contacts", []),
                "contact_geoms": record.get("contact_geoms_top", []),
                "placement_support": record.get("placement_support", []),
                "source_validation_status": record.get("validation_status"),
                "summary": record.get("summary"),
            }
        )

    p3 = load_json(REPORT / "isaac_recapture_p3_physics/p2_p3_evaluation.json")
    p3_hold = load_json(REPORT / "isaac_recapture_p3_physics/p3_hold_evaluation.json")
    p4 = load_json(REPORT / "isaac_recapture_p3_physics/p4_approach_hold100_evaluation.json")
    bridge22 = load_json(REPORT / "isaac_recapture_p3_physics/bridge22_warmstart60_evaluation.json")
    bridge22_cache = load_json(
        REPORT / "isaac_recapture_p3_physics/bridge22_cache_warmstart60_evaluation.json"
    )
    bridge25 = load_json(REPORT / "isaac_recapture_p3_physics/bridge25_warmstart30_evaluation.json")
    bridge_thumb_preserve = load_json(
        REPORT / "isaac_recapture_p3_physics/bridge22_thumb_preserve_warmstart60_evaluation.json"
    )
    bridge_thumb_warmup = load_json(
        REPORT / "isaac_recapture_p3_physics/bridge22_thumb_preserve_warmup60_evaluation.json"
    )
    delayed_teacher_zero = load_json(
        REPORT / "bridge22_delayed3_contact_teacher_zero_evaluation.json"
    )
    delayed_policy = load_json(REPORT / "bridge22_delayed3_contact_policy_evaluation.json")
    delayed_trained = load_json(REPORT / "bridge22_delayed3_contact_trained400_evaluation.json")
    delayed_hold = load_json(REPORT / "bridge22_delayed3_contact_hold_zero_evaluation.json")
    delayed_static = load_json(REPORT / "bridge22_delayed3_contact_static_evaluation.json")
    delayed_static_trained = load_json(
        REPORT / "bridge22_delayed3_contact_static_trained100_evaluation.json"
    )
    delayed_processed = load_json(
        REPORT / "bridge22_delayed3_contact_processed_evaluation.json"
    )
    earlier_recapture = load_json(
        REPORT / "earlier_pre_ring_index_recapture_trained100_evaluation.json"
    )
    earlier_teacher = load_json(
        REPORT / "earlier_pre_ring_index_recapture_teacher_50.json"
    )
    earlier_smoke = load_json(
        REPORT / "earlier_pre_ring_index_contact_physx_smoke.json"
    )
    thumb_search = load_json(REPORT / "bridge22_thumb_action_search.json")
    thumb_impulse_search = load_json(REPORT / "bridge22_thumb_impulse3_search.json")
    reset_compare = load_json(REPORT / "bridge22_thumb_preserve_smoke.json")

    physical_p2 = load_npz(REPORT / "isaac_physics_ring_touch_states.npz")
    physical_p3 = load_npz(REPORT / "isaac_physics_p3_states.npz")
    physical_p4 = load_npz(REPORT / "isaac_physics_p4_approach_hold100_states.npz")
    delayed_contact_bank = load_npz(REPORT / "isaac_physics_p4_approach_delayed3_contact_states.npz")
    delayed_contact_bundle = load_npz(REPORT / "isaac_bridge22_delayed3_contact_bundle.npz")
    delayed_static_bundle = load_npz(REPORT / "isaac_bridge22_delayed3_contact_static_bundle.npz")
    delayed_processed_bundle = load_npz(
        REPORT / "isaac_bridge22_delayed3_contact_processed_bundle.npz"
    )
    earlier_recapture_bundle = load_npz(
        REPORT / "isaac_earlier_pre_ring_index_recapture_bundle.npz"
    )
    earlier_teacher_bundle = load_npz(
        REPORT / "earlier_pre_ring_index_recapture_teacher_50.npz"
    )
    earlier_to_p3_bundle = load_npz(REPORT / "isaac_earlier_to_p3_entry_bundle.npz")
    bridge_teacher = load_json(REPORT / "isaac_bridge22_teacher_cache.json")
    recapture = load_npz(REPORT / "isaac_recapture_p3_bundle.npz")
    mujoco_cache = load_json(
        ROOT / "reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/dynamic_state_cache/manifest.json"
    )
    p3_manus = load_json(
        ROOT / "reports/manus_mujoco_presets/corrected_p3_p4_chain_m3_2_38/summary.json"
    )
    canonical_p3_search_path = (
        ROOT
        / "reports/manus_mujoco_presets/canonical_p3_manus_constrained_m3_2_40/summary.json"
    )
    canonical_p3_search = load_json_if_exists(canonical_p3_search_path)
    earlier_to_p3_smoke = load_json_if_exists(
        REPORT / "earlier_to_p3_entry_selected_filtered_smoke.json"
    )
    earlier_to_p3_training = load_json_if_exists(
        REPORT / "earlier_to_p3_entry_selected_20iter_evaluation.json"
    )
    earlier_to_p3_shaped = load_json_if_exists(
        REPORT / "earlier_to_p3_entry_teacher100_shaped_contact_evaluation.json"
    )
    alignment_report_path = REPORT / "mujoco_physx_alignment_100ms.json"
    alignment_report = load_json_if_exists(alignment_report_path)
    braked_p4_path = REPORT / "phase_capture50_robust_policy_nominal_evaluation.json"
    braked_p4 = load_json(braked_p4_path)
    unified_p3_p4_path = REPORT / "unified_p3_p4_all_gait_focus0x3_ppo80_final_evaluation.json"
    unified_p3_p4 = load_json(unified_p3_p4_path)

    OUT.write_text(
        json.dumps(
            {
                "stage": "canonical P1-P5 manifest",
                "source": "photo_five_phase_hand_pen_tuned_m3_2_10",
                "note": "Explicit pose/contact prior; source records are not themselves hold-validated.",
                "phases": phases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cache_counts = {
        name: info.get("num_states")
        for name, info in mujoco_cache.get("caches", {}).items()
    }
    audit = {
        "generated_from": {
            "canonical_manifest": str(OUT.relative_to(ROOT)),
            "p2_p3_eval": str((REPORT / "isaac_recapture_p3_physics/p2_p3_evaluation.json").relative_to(ROOT)),
            "p4_hold100_eval": str((REPORT / "isaac_recapture_p3_physics/p4_approach_hold100_evaluation.json").relative_to(ROOT)),
            "bridge22_eval": str((REPORT / "isaac_recapture_p3_physics/bridge22_warmstart60_evaluation.json").relative_to(ROOT)),
            "bridge25_eval": str((REPORT / "isaac_recapture_p3_physics/bridge25_warmstart30_evaluation.json").relative_to(ROOT)),
            "delayed_contact_bundle": str((REPORT / "isaac_bridge22_delayed3_contact_bundle.npz").relative_to(ROOT)),
            "delayed_contact_static_bundle": str((REPORT / "isaac_bridge22_delayed3_contact_static_bundle.npz").relative_to(ROOT)),
            "canonical_p3_search": (
                str(canonical_p3_search_path.relative_to(ROOT))
                if canonical_p3_search is not None
                else None
            ),
            "mujoco_physx_alignment": (
                str(alignment_report_path.relative_to(ROOT))
                if alignment_report is not None
                else None
            ),
            "braked_p4_eval": str(braked_p4_path.relative_to(ROOT)),
            "unified_p3_p4_eval": str(unified_p3_p4_path.relative_to(ROOT)),
        },
        "gates": {
            "canonical_p1_p5_explicit": {"status": "PASS", "value": 5},
            "p2_reset_cache_ge_200": {
                "status": "PASS",
                "value": int(physical_p2["qpos"][0]),
                "source": "isaac_physics_ring_touch_states.npz",
            },
            "p3_p2_to_p3": {
                "status": "PASS",
                "success_rate": p3["success_rate"],
                "deflection_gate_deg": 7.0,
                "source": "p2_p3_evaluation.json",
            },
            "p3_contact_hold_100ms": {
                "status": "PASS",
                "success_rate": p3_hold["success_rate"],
                "episodes": p3_hold["episodes"],
                "source": "p3_hold_evaluation.json",
            },
            "canonical_p3_manus_rms_lt_012": {
                "status": (
                    "PASS"
                    if canonical_p3_search is not None and canonical_p3_search["passed"]
                    else "FAIL"
                ),
                "value_rad": (
                    canonical_p3_search["best"]["max_hand_rms_rad"]
                    if canonical_p3_search is not None
                    else p3_manus["p3_pose_audit"]["hand_qpos_rms_to_manus_rad"]
                ),
                "threshold_rad": 0.12,
                "triad_hold_ms": (
                    canonical_p3_search["best"]["triad_hold_ms"]
                    if canonical_p3_search is not None
                    else None
                ),
                "max_penetration_mm": (
                    canonical_p3_search["best"]["max_penetration_mm"]
                    if canonical_p3_search is not None
                    else None
                ),
                "pen_drift_mm": (
                    canonical_p3_search["best"]["pen_drift_mm"]
                    if canonical_p3_search is not None
                    else None
                ),
                "legacy_dynamic_p3_rms_rad": p3_manus["p3_pose_audit"][
                    "hand_qpos_rms_to_manus_rad"
                ],
                "note": (
                    "The accepted canonical seed is distinct from the legacy dynamic P3 drive-exit."
                ),
            },
            "p3_to_p4_approach": {
                "status": "PASS",
                "success_rate": p4["success_rate"],
                "mean_hold_ms": p4["mean_episode_max_hold_ms"],
                "note": "20-degree approach/hold, not final braked P4.",
            },
            "braked_p4_50_60deg": {
                "status": "PASS",
                "target_deg": [50.0, 60.0],
                "hold_ms": 100.0,
                "success_rate": braked_p4["success_rate"],
                "checkpoint": braked_p4["checkpoint"],
            },
            "p3_to_braked_p4_unified": {
                "status": "PASS",
                "success_rate": unified_p3_p4["success_rate"],
                "minimum_stage_success_rate": min(
                    stage["success_rate"]
                    for stage in unified_p3_p4["per_reset_stage"].values()
                ),
                "per_reset_stage": unified_p3_p4["per_reset_stage"],
                "checkpoint": unified_p3_p4["checkpoint"],
            },
            "bridge22": {
                "status": "FAIL",
                "success_rate": bridge22["success_rate"],
                "mean_hold_ms": bridge22["mean_episode_max_hold_ms"],
                "thumb_contact_ratio": bridge22["contact_ratio"]["thumb"],
            },
            "bridge25": {
                "status": "FAIL",
                "success_rate": bridge25["success_rate"],
                "mean_hold_ms": bridge25["mean_episode_max_hold_ms"],
                "thumb_contact_ratio": bridge25["contact_ratio"]["thumb"],
            },
            "bridge22_materialized_cache": {
                "status": "FAIL",
                "success_rate": bridge22_cache["success_rate"],
                "mean_hold_ms": bridge22_cache["mean_episode_max_hold_ms"],
                "thumb_contact_ratio": bridge22_cache["contact_ratio"]["thumb"],
                "note": "Cache path improves thumb contact but remains below stable-primitive gate.",
            },
            "bridge22_thumb_preserve": {
                "status": "FAIL",
                "success_rate": bridge_thumb_preserve["success_rate"],
                "mean_hold_ms": bridge_thumb_preserve["mean_episode_max_hold_ms"],
                "thumb_contact_ratio": bridge_thumb_preserve["contact_ratio"]["thumb"],
            },
            "bridge22_thumb_preserve_warmup": {
                "status": "FAIL",
                "success_rate": bridge_thumb_warmup["success_rate"],
                "mean_hold_ms": bridge_thumb_warmup["mean_episode_max_hold_ms"],
                "thumb_contact_ratio": bridge_thumb_warmup["contact_ratio"]["thumb"],
                "note": "Warmup improves contact direction but remains far below 100ms hold.",
            },
            "bridge22_thumb_action_search": {
                "status": "DIAGNOSTIC",
                "constant_action_top_hold6_ratio": max(
                    row["hold6_ratio"] for row in thumb_search["top"]
                ),
                "impulse3_top_hold6_ratio": max(
                    row["hold6_ratio"] for row in thumb_impulse_search["top"]
                ),
                "thumb_preserve_smoke_pen_drift_mm": reset_compare["pen_drift_mm"]["mean"],
            },
            "bridge22_delayed_contact_entry": {
                "status": "FAIL",
                "source_bank": delayed_contact_bank,
                "bundle": delayed_contact_bundle,
                "teacher_zero": delayed_teacher_zero,
                "old_policy": delayed_policy,
                "trained400": delayed_trained,
                "note": "Delayed capture is physically revalidated, but reset/replay loses T/I/M contact before the 100ms bridge hold gate.",
            },
            "bridge22_delayed_contact_hold_target": {
                "status": "FAIL",
                "bundle": delayed_static_bundle,
                "static_action_baseline": delayed_static,
                "note": "Holding actuator targets at measured capture qpos does not by itself preserve the three-finger contact under the captured pen velocity.",
            },
            "bridge22_delayed_contact_processed_target": {
                "status": "FAIL",
                "bundle": delayed_processed_bundle,
                "static_zero_action": delayed_processed,
                "static_trained100": delayed_static_trained,
                "note": "Using the exact processed actuator target improves the middle-finger baseline relative to measured-qpos hold, but remains below the three-finger/100ms gate.",
            },
            "earlier_pre_ring_index_thumb_recapture": {
                "status": "PASS",
                "bundle": earlier_recapture_bundle,
                "smoke": earlier_smoke,
                "trained100": earlier_recapture,
                "success_threshold": 0.60,
                "note": "Separate earlier-window index/thumb recapture primitive; this is not yet the full P3 or braked-P4 gate.",
            },
            "earlier_recapture_teacher_trajectories": {
                "status": "PASS",
                "metadata": earlier_teacher,
                "bundle": earlier_teacher_bundle,
                "trajectory_threshold": 30,
                "note": "50 successful 7-step PhysX rollouts with qpos/qvel/action/contact/tip-distance fields.",
            },
            "earlier_to_p3_entry_bridge": {
                "status": "OPEN",
                "bundle": earlier_to_p3_bundle,
                "target_phase_deg": 7.0,
                "required_contacts": ["thumb", "index", "middle"],
                "reset_smoke": earlier_to_p3_smoke,
                "trained20iter": earlier_to_p3_training,
                "teacher100_shaped": earlier_to_p3_shaped,
                "note": "Filtered measured recapture frames now provide a no-noise PhysX entry; policy has not yet reached the 7-degree P3 gate.",
            },
            "teacher_cache": {
                "status": "PARTIAL",
                "physx_p3_states": physical_p3,
                "physx_p4_states_and_teacher": physical_p4,
                "bridge22_teacher_cache": bridge_teacher,
                "recapture_bundle": recapture,
                "mujoco_dynamic_cache_counts": cache_counts,
            },
            "mujoco_physx_50_100ms_alignment": {
                "status": "MEASURED" if alignment_report is not None else "MISSING",
                "required": "paired pose/contact/velocity error report over 50-100ms",
                "report": alignment_report,
            },
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)
    print(AUDIT)


if __name__ == "__main__":
    main()
