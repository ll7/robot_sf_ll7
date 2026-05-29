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
        """Track whether reset was propagated by the wrapper."""
        self._reset_called = False

    def next_obs(self) -> dict[str, np.ndarray]:
        """Return the base observation mapping that custom sensors augment."""
        return {"drive_state": np.array([1.0], dtype=np.float32)}

    def reset_cache(self) -> None:
        """Record that the wrapper called the base reset hook."""
        self._reset_called = True


class _RecordingSensor:
    """Sensor test double that records state and reset propagation."""

    def __init__(self) -> None:
        """Initialize the recording slots used by assertions."""
        self.last_state = None
        self.reset_called = False

    def step(self, state) -> None:
        """Capture the lightweight fusion state dict."""
        self.last_state = state

    def get_observation(self) -> np.ndarray:
        """Return a deterministic custom observation."""
        return np.array([2.0, 3.0], dtype=np.float32)

    def reset(self) -> None:
        """Record reset propagation."""
        self.reset_called = True


@pytest.fixture
def _clean_registry():
    """Remove sensors registered by a test after that test finishes."""
    original = set(list_sensors().keys())
    yield
    current = set(list_sensors().keys())
    for name in current - original:
        unregister_sensor(name)


def test_merged_observation_fusion_adds_custom_keys(_clean_registry):
    """Custom sensors are merged under stable ``custom.<name>`` keys."""
    # Arrange: register a constant sensor and build via adapter.
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


def test_merged_observation_fusion_passes_state_and_resets_sensor() -> None:
    """Custom sensors receive ``sim``/``robot_id`` and reset propagation."""
    sensor = _RecordingSensor()
    sim = object()
    base = _BaseFusionStub()
    fusion = MergedObservationFusion(base, [sensor], ["recorder"], sim=sim, robot_id=7)

    obs = fusion.next_obs()

    assert sensor.last_state == {"sim": sim, "robot_id": 7}
    np.testing.assert_allclose(obs["custom.recorder"], np.array([2.0, 3.0], dtype=np.float32))

    fusion.reset_cache()

    assert sensor.reset_called is True
    assert base._reset_called is True


def test_merged_observation_fusion_rejects_name_count_mismatch() -> None:
    """A configuration error should fail before the first observation step."""
    with pytest.raises(ValueError, match="one sensor name per sensor"):
        MergedObservationFusion(_BaseFusionStub(), [_RecordingSensor()], [])


def test_merged_observation_fusion_rejects_case_insensitive_duplicate_names() -> None:
    """Custom sensor names must not silently collide after normalization."""
    sensors = [_RecordingSensor(), _RecordingSensor()]

    with pytest.raises(ValueError, match="unique sensor names"):
        MergedObservationFusion(_BaseFusionStub(), sensors, ["Bias", "bias"])
