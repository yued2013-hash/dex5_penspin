#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    expected = json.loads(args.expected.read_text(encoding="utf-8"))

    assert actual["status"] == "PASS_DIAGNOSTIC"
    assert actual["artifact_role"] == expected["artifact_role"]
    for arm in ("uniform", "curriculum"):
        actual_summary = actual[arm]["summary"]
        expected_summary = expected[arm]["summary"]
        assert actual_summary["phase_reach_count"] == expected_summary["phase_reach_count"]
        assert actual_summary["first_failed_gate_count"] == expected_summary["first_failed_gate_count"]
    assert actual["decision_gate"] == expected["decision_gate"]
    print("O1.23 deterministic offline reproduction: PASS")


if __name__ == "__main__":
    main()
