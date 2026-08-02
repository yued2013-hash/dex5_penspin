import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/"
    "O1_21_Skill_Entry_Coverage_Audit.json"
)


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_o1_21_preserves_unknown_population_semantics() -> None:
    population = _report()["population"]
    assert population["P3_states"] == 396
    assert population["long_horizon_measured_states"] == 37
    assert population["long_horizon_unmeasured_states"] == 359
    assert population["known_strict_feasible"] == 3
    assert "unknown" in population["scope_warning"]


def test_o1_21_density_layers_are_ordered_and_not_labels() -> None:
    density = _report()["candidate_density"]
    assert density["zero_known_failure_core"]["states"] <= density["all_positive_sensitivity_envelope"]["states"]
    assert density["all_positive_sensitivity_envelope"]["states"] <= density["known_failure_5pct_candidate_band"]["states"]
    for layer in density.values():
        assert layer["role"] == "analysis_only_not_feasibility_labels_or_reset_filter"


def test_o1_21_decision_is_a_plus_c_with_b_deferred() -> None:
    decision = _report()["decision"]
    assert decision["A_existing_P3_coverage_insufficient"]["supported"] is True
    assert decision["B_resample_canonical_now"]["supported"] is False
    assert decision["C_policy_exploration_remains_primary"]["supported"] is True
    assert decision["overall"] == "A_AND_C_SUPPORTED_B_DEFERRED"


def test_o1_21_does_not_create_training_assets() -> None:
    assert _report()["constraints"] == {
        "O0_modified": False,
        "P3H_created": False,
        "PPO_trained": False,
        "teacher_generated": False,
        "reset_bank_modified": False,
        "new_reset_dataset_created": False,
    }
