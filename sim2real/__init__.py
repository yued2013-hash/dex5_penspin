"""Dex5 simulation-to-real interface utilities."""

from .dex5_hardware_adapter import (
    COUPLED_PAIRS,
    JOINT_COUNT,
    JOINT_ORDER,
    LOGICAL_ACTUATOR_COUNT,
    Dex5AdapterConfig,
    Dex5HardwareCompatibilityAdapter,
)

__all__ = [
    "COUPLED_PAIRS",
    "JOINT_COUNT",
    "JOINT_ORDER",
    "LOGICAL_ACTUATOR_COUNT",
    "Dex5AdapterConfig",
    "Dex5HardwareCompatibilityAdapter",
]
