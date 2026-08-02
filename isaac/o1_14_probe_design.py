from __future__ import annotations

import numpy as np


TIM_ACTION_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19)


def make_symmetric_probes(
    action_dim: int,
    probe_count: int,
    max_offset_rad: float,
    seed: int,
) -> np.ndarray:
    """Create one zero probe and deterministic antithetic TIM probes."""
    if probe_count < 3 or probe_count % 2 == 0:
        raise ValueError("probe_count must be odd and at least 3")
    if action_dim <= max(TIM_ACTION_INDICES):
        raise ValueError("action_dim is incompatible with the Dex5 joint order")
    if max_offset_rad <= 0.0:
        raise ValueError("max_offset_rad must be positive")

    generator = np.random.default_rng(seed)
    half = (probe_count - 1) // 2
    active = generator.normal(size=(half, len(TIM_ACTION_INDICES)))
    active /= np.max(np.abs(active), axis=1, keepdims=True).clip(min=1.0e-9)
    positive = np.zeros((half, action_dim), dtype=np.float32)
    positive[:, TIM_ACTION_INDICES] = active.astype(np.float32) * max_offset_rad
    return np.concatenate(
        (np.zeros((1, action_dim), dtype=np.float32), positive, -positive), axis=0
    )


def least_squares_gain_norms(
    probe_offsets: np.ndarray,
    responses: np.ndarray,
    ridge: float = 1.0e-6,
) -> np.ndarray:
    """Return per-state local response gain norms for one or more outputs."""
    x = np.asarray(probe_offsets, dtype=np.float64)[:, TIM_ACTION_INDICES]
    y = np.asarray(responses, dtype=np.float64)
    if y.ndim == 2:
        y = y[..., None]
    if y.ndim != 3 or y.shape[1] != x.shape[0]:
        raise ValueError("responses must have shape [states, probes, outputs]")
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    projector = np.linalg.solve(gram, x.T)
    coefficients = np.einsum("ap,spo->sao", projector, y)
    return np.linalg.norm(coefficients, axis=1)


def principal_coordinates(features: np.ndarray) -> np.ndarray:
    """Compute a stable two-dimensional PCA projection without sklearn."""
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    variable = std > 1.0e-10
    if variable.sum() < 2:
        return np.zeros((values.shape[0], 2), dtype=np.float64)
    standardized = np.clip(
        (values[:, variable] - mean[variable]) / std[variable], -5.0, 5.0
    )
    centered = standardized - standardized.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vh[: min(2, vh.shape[0])].T
    if coordinates.shape[1] == 1:
        coordinates = np.pad(coordinates, ((0, 0), (0, 1)))
    return coordinates
