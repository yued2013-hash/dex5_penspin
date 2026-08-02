import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/"
    "O1_20_Canonical_Entry_Audit.json"
)


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_o1_20_uses_corrected_strict_evidence() -> None:
    report = _report()
    evidence = report["evidence_reconciliation"]
    assert evidence["O1_8_corrected_strict_feasible"] == 0
    assert evidence["O1_15_conditional_population"] == 37
    assert evidence["O1_15_strict_positive_count"] == 3
    assert set(evidence["strict_positive_bank_ids"]) == {831, 894, 1155}
    assert evidence["positive_parent_count"] == 3
    assert evidence["positive_source_count"] == 3


def test_o1_20_does_not_change_training_assets() -> None:
    constraints = _report()["constraints"]
    assert constraints == {
        "PPO_trained": False,
        "reward_modified": False,
        "O0_modified": False,
        "teacher_generated": False,
        "reset_bank_modified": False,
        "CEM_action_serialized": False,
    }


def test_o1_20_candidate_layer_is_quarantined() -> None:
    report = _report()
    ranking = report["quarantined_candidate_ranking"]
    assert ranking["status"] == "analysis_only_do_not_use_as_reset_filter"
    assert ranking["parent_source_overlap_with_O1_15"] is False
    assert len(ranking["rows"]) == 20
    assert len({row["parent_state_id"] for row in ranking["rows"]}) == 20
    assert len({row["source_trajectory_id"] for row in ranking["rows"]}) == 20


def test_o1_20_classifier_claim_obeys_pre_registered_gate() -> None:
    classifier = _report()["skill_entry_classifier"]
    gate = classifier["promotion_gate"]
    assert gate["all_pass"] == all(
        (
            gate["combined_auc_at_least_0P80"],
            gate["combined_permutation_p_below_0P05"],
            gate["every_positive_above_failure_median"],
        )
    )
    if not gate["all_pass"]:
        assert classifier["status"] == "REJECTED_INSUFFICIENT_GENERALIZATION"
