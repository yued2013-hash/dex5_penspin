import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = (
    ROOT
    / "isaac/dex5_penspin_isaaclab/tasks/index_near_surface/"
    "privileged_oracle_physics_env.py"
)


def _class_assignments(name: str) -> dict[str, object]:
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    target = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )
    values = {}
    for node in target.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def test_o1_22_changes_only_reset_curriculum_config() -> None:
    values = _class_assignments("Dex5PrivilegedOracleP3EntryCurriculumCfg")
    assert set(values) == {
        "curriculum_candidate_bank_ids",
        "curriculum_candidate_probability_start",
        "curriculum_hold_control_steps",
        "curriculum_decay_end_control_steps",
    }
    assert values["curriculum_candidate_probability_start"] == 0.75
    assert values["curriculum_hold_control_steps"] == 480
    assert values["curriculum_decay_end_control_steps"] == 960


def test_o1_22_candidates_are_o1_21_quarantined_rows() -> None:
    values = _class_assignments("Dex5PrivilegedOracleP3EntryCurriculumCfg")
    assert values["curriculum_candidate_bank_ids"] == (
        1034,
        1176,
        1146,
        1045,
        1227,
        852,
        1149,
    )


def test_o1_22_inherits_temporal_observation_and_frozen_reward() -> None:
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Dex5PrivilegedOracleP3EntryCurriculumCfg"
    )
    assert [base.id for base in target.bases] == ["Dex5PrivilegedOracleP3TemporalCfg"]
