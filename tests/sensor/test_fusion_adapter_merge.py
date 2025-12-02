"""Tests for MergedObservationFusion wrapper.

This focuses on verifying that registry-based sensors are merged into the base
SensorFusion observation dict under custom.<name> keys without changing
SensorFusion internals.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.sensor.dummy_constant import DummyConstantSensor
from robot_sf.sensor.fusion_adapter import (
    MergedObservationFusion,
    create_sensors_from_config,
)
from robot_sf.sensor.registry import list_sensors, register_sensor, unregister_sensor


class _BaseFusionStub:
    """Minimal stand-in for SensorFusion with compatible API."""

    def __init__(self) -> None:
        """Init.

        Returns:
            None: Auto-generated placeholder description.
        """
        self._reset_called = False

    def next_obs(self) -> dict[str, np.ndarray]:
        """Next obs.

        Returns:
            dict[str, np.ndarray]: Auto-generated placeholder description.
        """
        return {"drive_state": np.array([1.0], dtype=np.float32)}

    def reset_cache(self) -> None:
        """Reset cache.

        Returns:
            None: Auto-generated placeholder description.
        """
        self._reset_called = True


@pytest.fixture
def _clean_registry():
    """Clean registry.

    Returns:
        Any: Auto-generated placeholder description.
    """
    original = set(list_sensors().keys())
    yield
    current = set(list_sensors().keys())
    for name in current - original:
        unregister_sensor(name)


def test_merged_observation_fusion_adds_custom_keys(_clean_registry):
    """Test merged observation fusion adds custom keys.

    Args:
        _clean_registry: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Arrange: register a constant sensor and build via adapter
    # (sensor may already be registered via __init__.py)
    register_sensor("dummy_constant", DummyConstantSensor, override=True)
    cfgs = [
        {
            "type": "dummy_constant",
            "name": "bias",
            "value": [0.5, 0.25],
            "shape": [2],
            "dtype": "float32",
        }
    ]
    sensors = create_sensors_from_config(cfgs)
    base = _BaseFusionStub()
    fusion = MergedObservationFusion(base, sensors, ["bias"], sim=None, robot_id=0)

    # Act
    obs = fusion.next_obs()

    # Assert
    assert "drive_state" in obs
    assert "custom.bias" in obs
    np.testing.assert_allclose(obs["custom.bias"], np.array([0.5, 0.25], dtype=np.float32))

    # And reset should propagate
    fusion.reset_cache()
    # Validate that reset propagated to base (introspective check in stub)
    assert getattr(base, "_reset_called", False) is True
