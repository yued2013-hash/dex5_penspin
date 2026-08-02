import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / (
    "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/"
    "O1_23_Transition_Failure_Manifold_Audit.json"
)


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_o1_23_uses_matched_fixed_horizon_rows() -> None:
    matched = _report()["matched_contract"]
    assert matched["same_ordered_bank_rows"] is True
    assert matched["same_bank_hash"] is True
    assert matched["same_horizon_steps"] == 120


def test_o1_23_locates_failure_after_contact_transition() -> None:
    report = _report()
    summary = report["uniform"]["summary"]
    assert summary["phase_reach_count"]["release"] == 256
    assert summary["phase_reach_count"]["same_finger_recontact"] == 173
    assert summary["phase_reach_count"]["support_continuity"] == 116
    assert summary["phase_reach_count"]["height_recovery"] == 0
    assert report["decision_gate"]["uniform_policy_dominant_failure_phase"] == (
        "height_not_recovered"
    )


def test_o1_23_requires_capture_audit_before_phase_design() -> None:
    decision = _report()["decision_gate"]
    assert decision["state_curriculum_conclusion"] == "REJECTED_BY_O1_22"
    assert decision["transition_phase_data_exists"] is True
    assert decision["transition_curriculum_design_authorized"] is False
    assert decision["p4_p5_capture_window_audit_required"] is True
    assert decision["penetration_audit_required"] is True
    assert decision["skill_pretraining_or_hierarchy_authorized"] is False
    assert decision["curriculum_checkpoint_effect"]["recontact_rate_delta"] < -0.40


def test_o1_23_artifacts_are_diagnostic_only() -> None:
    assert _report()["artifact_role"] == "diagnostic_only_not_teacher_not_reset"
