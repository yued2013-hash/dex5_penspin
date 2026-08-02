#!/usr/bin/env python3
"""Deterministic checks for S0.9.2 quaternion progress metrics."""

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("audit_s0_9_2_oracle_trajectory_validity.py")
SPEC = importlib.util.spec_from_file_location("s0_9_2_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def z_quaternion(degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) / 2.0
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)])


if __name__ == "__main__":
    angles = np.arange(0.0, 46.0, 5.0)
    quaternion = np.stack([z_quaternion(value) for value in angles])
    pose = np.zeros((len(angles), 7), dtype=np.float64)
    pose[:, 2] = 0.06
    pose[:, 3:7] = quaternion
    angular_velocity = np.zeros((len(angles), 3), dtype=np.float64)
    contact = np.zeros((len(angles), 5), dtype=bool)
    contact[:, :3] = True
    geodesic, target = MODULE.quaternion_progress(quaternion)
    derived = MODULE.quaternion_derived_angular_velocity(quaternion)
    assert np.allclose(geodesic, angles)
    assert np.allclose(target, angles)
    assert np.allclose(derived[:, 2], np.deg2rad(5.0) * MODULE.CONTROL_HZ)
    metrics = MODULE.episode_metrics(pose, angular_velocity, contact)
    assert np.isclose(metrics["quaternion_final_geodesic_deg"], 45.0)
    assert np.isclose(metrics["target_axis_final_alignment"], 1.0)
    assert np.isclose(metrics["monotonic_efficiency"], 1.0)
    assert metrics["tier"] == "C_FALSE_ROTATION"
    # Only four samples are at or above 30 degrees, so the 150 ms gate must fail.
    assert metrics["max_strict_hold_steps"] == 4
    print("S0.9.2 oracle trajectory validity checks: PASS")
