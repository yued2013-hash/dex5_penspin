from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim2real.dex5_hardware_adapter import (
    COUPLED_PAIRS,
    JOINT_LOWER,
    JOINT_UPPER,
    Dex5AdapterConfig,
    Dex5HardwareCompatibilityAdapter,
)


def _midpoint() -> np.ndarray:
    return Dex5HardwareCompatibilityAdapter.project_virtual_target(
        0.5 * (JOINT_LOWER + JOINT_UPPER)
    )


def _load_torch_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "isaac/dex5_penspin_isaaclab/tasks/index_near_surface/hardware_compatible_action.py"
    )
    spec = importlib.util.spec_from_file_location("dex5_s0_torch_wrapper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_torch_wrapper():
    return _load_torch_module().Dex5IsaacHardwareCompatibilityWrapper


def test_projection_enforces_limits_and_shared_motor_pairs() -> None:
    rng = np.random.default_rng(20260731)
    targets = rng.uniform(-3.0, 3.0, size=(4096, 20))
    projected = Dex5HardwareCompatibilityAdapter.project_virtual_target(targets)
    assert np.all(projected >= JOINT_LOWER - 1.0e-12)
    assert np.all(projected <= JOINT_UPPER + 1.0e-12)
    for proximal, distal in COUPLED_PAIRS:
        np.testing.assert_allclose(projected[:, proximal], projected[:, distal])


def test_distal_command_wins_shared_pair_conflict() -> None:
    target = _midpoint()
    target[2] = 0.2
    target[3] = 0.9
    projected = Dex5HardwareCompatibilityAdapter.project_virtual_target(target)
    np.testing.assert_allclose(projected[[2, 3]], 0.9)


def test_logical_actuator_round_trip_is_exact_on_projected_manifold() -> None:
    rng = np.random.default_rng(4)
    projected = Dex5HardwareCompatibilityAdapter.project_virtual_target(
        rng.uniform(JOINT_LOWER, JOINT_UPPER, size=(256, 20))
    )
    logical = Dex5HardwareCompatibilityAdapter.virtual_to_logical_actuator(projected)
    reconstructed = Dex5HardwareCompatibilityAdapter.logical_actuator_to_virtual(logical)
    np.testing.assert_allclose(reconstructed, projected)


def test_delta_is_bounded_by_delta_and_velocity_limits() -> None:
    config = Dex5AdapterConfig(max_policy_delta_rad=0.025, max_velocity_rad_s=0.6)
    initial = _midpoint()
    adapter = Dex5HardwareCompatibilityAdapter(initial, config)
    destination = adapter.submit_policy_delta(np.full(20, 10.0), 1.0)
    # 0.6 / 60 = 0.01 rad is tighter than the 0.025-rad policy bound.
    assert np.max(np.abs(destination - initial)) <= 0.0100000001


def test_60_to_100_hz_interpolation_is_continuous_and_projected() -> None:
    config = Dex5AdapterConfig(max_policy_delta_rad=0.02, max_velocity_rad_s=0.5)
    initial = _midpoint()
    adapter = Dex5HardwareCompatibilityAdapter(initial, config)
    rng = np.random.default_rng(8)
    policy_times = np.arange(0.0, 1.0, 1.0 / config.policy_hz)
    command_times = np.arange(0.0, 1.0, 1.0 / config.command_hz)
    policy_index = 0
    outputs = []
    for command_time in command_times:
        while policy_index < len(policy_times) and policy_times[policy_index] <= command_time + 1e-12:
            adapter.submit_policy_delta(rng.uniform(-0.03, 0.03, size=20), policy_times[policy_index])
            policy_index += 1
        outputs.append(adapter.sample(command_time))
    outputs = np.asarray(outputs)
    assert np.max(np.abs(np.diff(outputs, axis=0))) <= 0.005000001
    for proximal, distal in COUPLED_PAIRS:
        np.testing.assert_allclose(outputs[:, proximal], outputs[:, distal])


def test_watchdog_holds_last_safe_command() -> None:
    adapter = Dex5HardwareCompatibilityAdapter(_midpoint())
    adapter.submit_policy_delta(np.full(20, 0.01), 0.0)
    safe = adapter.sample(0.09)
    stale = adapter.sample(0.101)
    assert adapter.watchdog_tripped
    np.testing.assert_allclose(stale, safe)


def test_invalid_shapes_and_non_finite_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        Dex5HardwareCompatibilityAdapter.project_virtual_target(np.zeros(19))
    target = np.zeros(20)
    target[4] = np.nan
    with pytest.raises(ValueError):
        Dex5HardwareCompatibilityAdapter.project_virtual_target(target)


def test_torch_wrapper_matches_numpy_projection_and_round_trip() -> None:
    wrapper_type = _load_torch_wrapper()
    lower = torch.as_tensor(JOINT_LOWER, dtype=torch.float64)
    upper = torch.as_tensor(JOINT_UPPER, dtype=torch.float64)
    wrapper = wrapper_type(lower, upper)
    rng = np.random.default_rng(12)
    values = rng.uniform(-2.0, 2.0, size=(64, 20))
    expected = Dex5HardwareCompatibilityAdapter.project_virtual_target(values)
    actual = wrapper.project_targets(torch.as_tensor(values)).numpy()
    np.testing.assert_allclose(actual, expected)
    logical = wrapper.virtual_to_logical_actuator(torch.as_tensor(actual))
    reconstructed = wrapper.logical_actuator_to_virtual(logical).numpy()
    np.testing.assert_allclose(reconstructed, expected)


def test_torch_wrapper_applies_projected_targets_to_fake_isaac_env() -> None:
    wrapper_type = _load_torch_wrapper()
    lower = torch.as_tensor(JOINT_LOWER, dtype=torch.float64)
    upper = torch.as_tensor(JOINT_UPPER, dtype=torch.float64)
    wrapper = wrapper_type(lower, upper)

    class FakeHand:
        def __init__(self) -> None:
            self.target = None
            self.joint_ids = None

        def set_joint_position_target(self, target, *, joint_ids) -> None:
            self.target = target.clone()
            self.joint_ids = tuple(joint_ids)

    class FakeEnv:
        def __init__(self) -> None:
            self.processed_targets = torch.as_tensor(_midpoint()).unsqueeze(0)
            self.hand = FakeHand()
            self.dex5_joint_ids = list(range(20))

    env = FakeEnv()
    actions = torch.ones((1, 20), dtype=torch.float64)
    target = wrapper.apply_to_env(env, actions, action_scale_rad=0.025)
    assert env.hand.joint_ids == tuple(range(20))
    torch.testing.assert_close(env.hand.target, target)
    for proximal, distal in COUPLED_PAIRS:
        torch.testing.assert_close(target[:, proximal], target[:, distal])


def test_s0_actuator_model_uses_100_hz_command_clock_on_120_hz_physics() -> None:
    model_type = _load_torch_module().Dex5S0ActuatorModel
    lower = torch.as_tensor(JOINT_LOWER, dtype=torch.float64)
    upper = torch.as_tensor(JOINT_UPPER, dtype=torch.float64)
    model = model_type(lower, upper, num_envs=2)
    initial = torch.as_tensor(_midpoint(), dtype=torch.float64).repeat(2, 1)
    model.reset(initial, time_s=0.0)
    model.submit_policy_action(
        torch.ones_like(initial), action_scale_rad=0.025, time_s=0.0
    )
    for physics_step in range(1, 121):
        model.physics_step(physics_step / 120.0)
    torch.testing.assert_close(
        model.command_update_count,
        torch.full((2,), 100, dtype=torch.long),
    )


def test_s0_actuator_model_enforces_delay_gain_velocity_and_coupling() -> None:
    model_type = _load_torch_module().Dex5S0ActuatorModel
    lower = torch.as_tensor(JOINT_LOWER, dtype=torch.float64)
    upper = torch.as_tensor(JOINT_UPPER, dtype=torch.float64)
    model = model_type(lower, upper, num_envs=2, max_delay_ticks=2)
    initial = torch.as_tensor(_midpoint(), dtype=torch.float64).repeat(2, 1)
    model.reset(
        initial,
        time_s=0.0,
        delay_ticks=torch.tensor([0, 2]),
        tracking_gain=torch.tensor([1.0, 0.5], dtype=torch.float64),
        velocity_limit_rad_s=torch.tensor([0.6, 0.3], dtype=torch.float64),
    )
    destination = model.submit_policy_action(
        torch.ones_like(initial), action_scale_rad=0.025, time_s=0.0
    )
    assert torch.max(torch.abs(destination[0] - initial[0])) <= 0.0100000001
    assert torch.max(torch.abs(destination[1] - initial[1])) <= 0.0050000001
    first = model.physics_step(0.01).clone()
    assert torch.max(torch.abs(first[0] - initial[0])) > 0.0
    torch.testing.assert_close(first[1], initial[1])
    model.physics_step(0.02)
    third = model.physics_step(0.03)
    assert torch.max(torch.abs(third[1] - initial[1])) > 0.0
    for proximal, distal in COUPLED_PAIRS:
        torch.testing.assert_close(third[:, proximal], third[:, distal])
    assert torch.all(third >= lower - 1.0e-12)
    assert torch.all(third <= upper + 1.0e-12)


def test_batched_torch_projection_supports_partial_environment_reset() -> None:
    wrapper_type = _load_torch_wrapper()
    lower = torch.as_tensor(JOINT_LOWER, dtype=torch.float64).repeat(4, 1)
    upper = torch.as_tensor(JOINT_UPPER, dtype=torch.float64).repeat(4, 1)
    wrapper = wrapper_type(lower, upper)
    ids = torch.tensor([1, 3], dtype=torch.long)
    targets = torch.as_tensor(_midpoint(), dtype=torch.float64).repeat(2, 1)
    targets[:, 3] += 0.1
    projected = wrapper.project_targets(targets, env_ids=ids)
    assert projected.shape == (2, 20)
    for proximal, distal in COUPLED_PAIRS:
        torch.testing.assert_close(projected[:, proximal], projected[:, distal])
    with pytest.raises(ValueError):
        wrapper.project_targets(targets)
