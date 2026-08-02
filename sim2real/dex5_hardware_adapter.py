"""Offline Dex5 20D virtual-joint to hardware-compatible target adapter.

The public Dex5 SDK still accepts 20 absolute URDF joint positions. This
module models the 16 physical actuators without bypassing that SDK: it
projects virtual targets onto the shared PIP/DIP manifold and returns a safe
20D absolute command. It deliberately contains no ROS or DDS publishing code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINT_ORDER = (
    "Roll_21R", "Pitch_22R", "Pitch_23R", "Pitch_24R",
    "Roll_31R", "Pitch_32R", "Pitch_33R", "Pitch_34R",
    "Roll_41R", "Pitch_42R", "Pitch_43R", "Pitch_44R",
    "Roll_51R", "Pitch_52R", "Pitch_53R", "Pitch_54R",
    "Yaw_11R", "Roll_12R", "Pitch_13R", "Pitch_14R",
)
JOINT_COUNT = len(JOINT_ORDER)
LOGICAL_ACTUATOR_COUNT = 16
JOINT_TO_ACTUATOR = np.asarray(
    (0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15),
    dtype=np.int64,
)
COUPLED_PAIRS = ((2, 3), (6, 7), (10, 11), (14, 15))
JOINT_LOWER = np.asarray(
    (
        -0.38397, 0.0, 0.0, 0.0,
        -0.38397, 0.0, 0.0, 0.0,
        -0.38397, 0.0, 0.0, 0.0,
        -0.38397, 0.0, 0.0, 0.0,
        -0.5867, 0.0, 0.0, 0.0,
    ),
    dtype=np.float64,
)
JOINT_UPPER = np.asarray(
    (
        0.38397, 1.5707963, 1.684242699, 1.396263378,
        0.38397, 1.5707963, 1.684242699, 1.396263378,
        0.38397, 1.5708, 1.6842, 1.3963,
        0.38397, 1.5708, 1.6842, 1.3963,
        0.6803, 1.8151, 1.765226, 1.6406,
    ),
    dtype=np.float64,
)


def _joint_vector(value: float | tuple[float, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(JOINT_COUNT, float(array), dtype=np.float64)
    if array.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must be scalar or shape ({JOINT_COUNT},), got {array.shape}")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _validate_last_dim(values: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have final dimension {size}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


@dataclass(frozen=True)
class Dex5AdapterConfig:
    """Frozen offline adapter parameters.

    The velocity and delta defaults are provisional software-audit bounds,
    not approved autonomous hardware limits. S0.1 hardware acceptance must
    replace them with measured values.
    """

    policy_hz: float = 60.0
    command_hz: float = 100.0
    max_policy_delta_rad: float | tuple[float, ...] = 0.025
    max_velocity_rad_s: float | tuple[float, ...] = 0.5
    watchdog_timeout_s: float = 0.100
    coupling_mode: str = "distal_wins"

    def __post_init__(self) -> None:
        if self.policy_hz <= 0.0 or self.command_hz <= 0.0:
            raise ValueError("policy_hz and command_hz must be positive")
        if self.watchdog_timeout_s <= 0.0:
            raise ValueError("watchdog_timeout_s must be positive")
        if self.coupling_mode != "distal_wins":
            raise ValueError("only the audited provisional distal_wins coupling is supported")
        _joint_vector(self.max_policy_delta_rad, "max_policy_delta_rad")
        _joint_vector(self.max_velocity_rad_s, "max_velocity_rad_s")


class Dex5HardwareCompatibilityAdapter:
    """Stateful delta-to-absolute adapter with interpolation and watchdog."""

    def __init__(self, initial_target: np.ndarray, config: Dex5AdapterConfig | None = None):
        self.config = config or Dex5AdapterConfig()
        self.max_policy_delta = _joint_vector(
            self.config.max_policy_delta_rad, "max_policy_delta_rad"
        )
        self.max_velocity = _joint_vector(self.config.max_velocity_rad_s, "max_velocity_rad_s")
        initial = self.project_virtual_target(initial_target)
        self._segment_start = initial.copy()
        self._segment_target = initial.copy()
        self._last_output = initial.copy()
        self._segment_start_time: float | None = None
        self._last_policy_time: float | None = None
        self._watchdog_tripped = False

    @property
    def current_target(self) -> np.ndarray:
        return self._segment_target.copy()

    @property
    def last_output(self) -> np.ndarray:
        return self._last_output.copy()

    @property
    def watchdog_tripped(self) -> bool:
        return self._watchdog_tripped

    @staticmethod
    def project_virtual_target(target: np.ndarray) -> np.ndarray:
        """Project 20D targets onto the provisional 16-actuator manifold.

        For each shared actuator, the distal command wins, matching the SDK
        source comment and current Manus derived-joint configuration. Both
        virtual joints are then set to the value allowed by their intersected
        limits. Hardware identification may replace this equality model.
        """

        projected = np.clip(
            _validate_last_dim(target, JOINT_COUNT, "target"), JOINT_LOWER, JOINT_UPPER
        ).copy()
        for proximal, distal in COUPLED_PAIRS:
            lower = max(JOINT_LOWER[proximal], JOINT_LOWER[distal])
            upper = min(JOINT_UPPER[proximal], JOINT_UPPER[distal])
            value = np.clip(projected[..., distal], lower, upper)
            projected[..., proximal] = value
            projected[..., distal] = value
        return projected

    @staticmethod
    def virtual_to_logical_actuator(target: np.ndarray) -> np.ndarray:
        """Compress to 16 logical actuator coordinates.

        These values are not motor-shaft angles. The vendor anemone process
        applies its own CasADi URDF-to-motor mappings, including nonlinear
        thumb conversion. Hardware commands must remain on the SDK 20D URDF
        interface.
        """

        projected = Dex5HardwareCompatibilityAdapter.project_virtual_target(target)
        output = np.empty(
            projected.shape[:-1] + (LOGICAL_ACTUATOR_COUNT,), dtype=np.float64
        )
        for joint_index, actuator_index in enumerate(JOINT_TO_ACTUATOR):
            output[..., actuator_index] = projected[..., joint_index]
        return output

    @staticmethod
    def logical_actuator_to_virtual(actuator_target: np.ndarray) -> np.ndarray:
        """Expand logical coordinates for offline manifold checks only."""

        actuator = _validate_last_dim(
            actuator_target, LOGICAL_ACTUATOR_COUNT, "actuator_target"
        )
        virtual = actuator[..., JOINT_TO_ACTUATOR]
        return Dex5HardwareCompatibilityAdapter.project_virtual_target(virtual)

    def submit_policy_delta(self, delta: np.ndarray, timestamp_s: float) -> np.ndarray:
        """Accept one 60 Hz delta action and return its safe absolute destination."""

        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if self._last_policy_time is not None and timestamp <= self._last_policy_time:
            raise ValueError("policy timestamps must be strictly increasing")

        if self._segment_start_time is None:
            current = self._last_output.copy()
        else:
            current = self.sample(timestamp, check_watchdog=False)

        requested = _validate_last_dim(delta, JOINT_COUNT, "delta")
        if requested.shape != (JOINT_COUNT,):
            raise ValueError("stateful adapter accepts one 20D delta at a time")
        policy_dt = 1.0 / self.config.policy_hz
        allowed = np.minimum(self.max_policy_delta, self.max_velocity * policy_dt)
        bounded_delta = np.clip(requested, -allowed, allowed)
        destination = self.project_virtual_target(current + bounded_delta)

        self._segment_start = current
        self._segment_target = destination
        self._segment_start_time = timestamp
        self._last_policy_time = timestamp
        self._watchdog_tripped = False
        return destination.copy()

    def sample(self, timestamp_s: float, *, check_watchdog: bool = True) -> np.ndarray:
        """Sample the piecewise-linear absolute command for the 100 Hz loop."""

        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if self._segment_start_time is None:
            self._last_output = self._segment_target.copy()
            return self.last_output
        if timestamp < self._segment_start_time:
            raise ValueError("command timestamp precedes the current interpolation segment")

        if (
            check_watchdog
            and self._last_policy_time is not None
            and timestamp - self._last_policy_time > self.config.watchdog_timeout_s
        ):
            self._watchdog_tripped = True
            return self.last_output

        alpha = np.clip(
            (timestamp - self._segment_start_time) * self.config.policy_hz, 0.0, 1.0
        )
        output = self._segment_start + alpha * (self._segment_target - self._segment_start)
        self._last_output = self.project_virtual_target(output)
        return self.last_output

    def reset(self, target: np.ndarray) -> None:
        safe = self.project_virtual_target(target)
        self._segment_start = safe.copy()
        self._segment_target = safe.copy()
        self._last_output = safe.copy()
        self._segment_start_time = None
        self._last_policy_time = None
        self._watchdog_tripped = False
