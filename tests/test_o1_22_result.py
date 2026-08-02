import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/o1_privileged_oracle_smoke/"
    "O1_22_Handoff_Basin_Curriculum_Audit.json"
)


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_o1_22_matched_contract() -> None:
    matched = _report()["matched_contract"]
    assert matched["seed"] == 20260730
    assert matched["num_envs"] == 256
    assert matched["iterations"] == 60
    assert matched["transitions"] == 368640
    assert matched["reward_equal"] is True
    assert matched["observation_equal"] is True
    assert matched["bank_hash_equal"] is True
    assert matched["uniform_P3_evaluation_for_both"] is True


def test_o1_22_rejects_curriculum_on_strict_and_safety_gates() -> None:
    report = _report()
    gate = report["acceptance_gate"]
    assert gate["strict_discovery_improved_and_nonzero"] is False
    assert gate["exploit_increase_at_most_2pp"] is False
    assert gate["curriculum_improves_skill_discovery"] is False
    assert report["B_early_curriculum"]["exploit"]["rate"] > 0.99


def test_o1_22_preserves_project_assets() -> None:
    constraints = _report()["constraints"]
    assert constraints["O0_modified"] is False
    assert constraints["P3H_created"] is False
    assert constraints["teacher_generated"] is False
    assert constraints["reward_modified"] is False
    assert constraints["new_reset_dataset_created"] is False
