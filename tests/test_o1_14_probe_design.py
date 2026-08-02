from __future__ import annotations

import numpy as np

from o1_14_probe_design import (
    TIM_ACTION_INDICES,
    least_squares_gain_norms,
    make_symmetric_probes,
    principal_coordinates,
)


def test_symmetric_probes_are_deterministic_antithetic_and_tim_only() -> None:
    probes = make_symmetric_probes(20, 31, 0.06, 20260731)
    np.testing.assert_array_equal(probes, make_symmetric_probes(20, 31, 0.06, 20260731))
    np.testing.assert_allclose(probes[0], 0.0)
    np.testing.assert_allclose(probes[1:16], -probes[16:])
    inactive = sorted(set(range(20)) - set(TIM_ACTION_INDICES))
    np.testing.assert_allclose(probes[:, inactive], 0.0)
    assert np.max(np.abs(probes)) <= 0.06


def test_gain_norm_recovers_linear_action_response() -> None:
    probes = make_symmetric_probes(20, 31, 0.06, 7)
    weights = np.linspace(-1.0, 1.0, len(TIM_ACTION_INDICES))
    response = probes[:, TIM_ACTION_INDICES] @ weights
    gains = least_squares_gain_norms(probes, response[None, :])
    np.testing.assert_allclose(gains[0, 0], np.linalg.norm(weights), rtol=2.0e-3)


def test_principal_coordinates_shape_and_finiteness() -> None:
    values = np.arange(40, dtype=np.float64).reshape(10, 4)
    coordinates = principal_coordinates(values)
    assert coordinates.shape == (10, 2)
    assert np.isfinite(coordinates).all()
