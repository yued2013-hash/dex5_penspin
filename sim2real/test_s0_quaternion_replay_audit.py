#!/usr/bin/env python3
"""Deterministic checks for the S0 quaternion replay gate."""

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("audit_and_build_s0_quaternion_replays.py")
SPEC = importlib.util.spec_from_file_location("s0_quaternion_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rotation_quaternion(degrees: float, axis: tuple[float, float, float]) -> np.ndarray:
    half_angle = np.deg2rad(degrees) / 2.0
    return np.asarray(
        [np.cos(half_angle), *(np.sin(half_angle) * np.asarray(axis))],
        dtype=np.float64,
    )


def check_rotation(degrees: float, axis: tuple[float, float, float]) -> dict[str, float]:
    quaternions = np.stack(
        [np.asarray([1.0, 0.0, 0.0, 0.0]), rotation_quaternion(degrees, axis)]
    )
    metrics = MODULE.quaternion_metrics(quaternions)
    assert np.isclose(metrics["quaternion_max_geodesic_deg"], degrees)
    assert np.isclose(metrics["quaternion_final_geodesic_deg"], degrees)
    return metrics


if __name__ == "__main__":
    z30 = check_rotation(30.0, (0.0, 0.0, 1.0))
    z45 = check_rotation(45.0, (0.0, 0.0, 1.0))
    x45 = check_rotation(45.0, (1.0, 0.0, 0.0))
    assert np.isclose(z30["quaternion_max_abs_world_z_component_deg"], 30.0)
    assert np.isclose(z45["quaternion_max_abs_world_z_component_deg"], 45.0)
    assert np.isclose(x45["quaternion_max_abs_world_z_component_deg"], 0.0)
    print("S0 quaternion replay audit checks: PASS")
