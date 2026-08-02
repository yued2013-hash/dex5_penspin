from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/sim2real_s0_2"
O0 = ROOT / (
    "reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0/"
    "canonical_reset_bank/P1_P5_v9_physx_projected_R3_validated.npz"
)
EXPECTED_HASH = "73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65"


def load(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    runs = {
        "baseline_zero": load("baseline_zero.json"),
        "s0_2_zero": load("s0_2_zero.json"),
        "baseline_small": load("baseline_small.json"),
        "s0_2_small": load("s0_2_small.json"),
    }
    zero = runs["s0_2_zero"]
    small = runs["s0_2_small"]
    baseline_zero = runs["baseline_zero"]
    baseline_small = runs["baseline_small"]
    o0_hash = hashlib.sha256(O0.read_bytes()).hexdigest()

    execution_chain = all(
        run["finite"]
        and run["done_count"] == 0
        and run["max_target_coupling_error_rad"] <= 1.0e-7
        and run["max_target_limit_violation_rad"] <= 1.0e-7
        and run["command_updates_per_env"] == {"min": 100, "max": 100}
        for run in (zero, small)
    )
    safety = all(
        run["drop_ever_rate"] == 0.0
        and run["contact_loss_ever_rate"] == 0.0
        and run["ballistic_ever_rate"] == 0.0
        and run["max_penetration_mm"] <= 0.5
        and run["tim_at_least_two_contact_ratio"] >= 0.99
        for run in (zero, small)
    )
    strict_tim_drop = {
        "zero_percentage_points": 100.0
        * (baseline_zero["tim_all_contact_ratio"] - zero["tim_all_contact_ratio"]),
        "small_percentage_points": 100.0
        * (baseline_small["tim_all_contact_ratio"] - small["tim_all_contact_ratio"]),
    }
    strict_tim_retained = max(strict_tim_drop.values()) <= 10.0
    frozen_o0 = o0_hash == EXPECTED_HASH and all(
        run["o0_sha256"] == EXPECTED_HASH for run in runs.values()
    )
    status = (
        "PASS"
        if execution_chain and safety and strict_tim_retained and frozen_o0
        else "PARTIAL_CONTACT_SHIFT"
        if execution_chain and safety and frozen_o0
        else "FAIL"
    )
    summary = {
        "status": status,
        "phase": "S0.2 Hardware-Compatible Simulation",
        "scope": "OFFLINE_AND_PHYSX_AUDIT_ONLY",
        "training_started": False,
        "hardware_used": False,
        "gates": {
            "execution_chain": execution_chain,
            "physical_safety": safety,
            "strict_tim_retention_within_10pp": strict_tim_retained,
            "frozen_o0_unchanged": frozen_o0,
        },
        "strict_tim_drop": strict_tim_drop,
        "o0_sha256": o0_hash,
        "runs": runs,
    }
    (REPORT_DIR / "s0_2_regression_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
