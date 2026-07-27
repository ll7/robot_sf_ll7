"""Direct contract tests for :mod:`robot_sf.sensor.dummy_constant`.

These tests lock the public surface of :class:`DummyConstantSensor` and the
``factory`` registration helper without starting an environment: every input is
a plain in-memory numpy value. Covered contracts:

- scalar and vector construction;
- dtype selection (default ``float32`` and configured ``int32``);
- shape broadcasting success and incompatible-shape failure;
- observation value/shape/dtype preservation and stability across reads;
- callers cannot mutate the sensor's stored constant through a returned array;
- ``reset``/``step`` are declared no-ops that preserve the constant;
- ``factory(config)`` returns a correctly configured ``DummyConstantSensor``.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.sensor.dummy_constant import DummyConstantSensor, factory


# --------------------------------------------------------------------------- #
# Construction and dtype selection
# --------------------------------------------------------------------------- #
def test_scalar_construction_defaults_to_float32():
    """A scalar value constructs a 0-d float32 observation preserving the value."""
    sensor = DummyConstantSensor({"name": "c", "value": 1.5})
    obs = sensor.get_observation()
    assert obs.dtype == np.float32
    assert obs.shape == ()
    assert obs == np.float32(1.5)


def test_scalar_construction_honors_configured_int32():
    """Configured ``dtype='int32'`` selects numpy int32 for a scalar value."""
    sensor = DummyConstantSensor({"name": "c", "value": 2, "dtype": "int32"})
    obs = sensor.get_observation()
    assert obs.dtype == np.int32
    assert obs.shape == ()
    assert obs == np.int32(2)


def test_default_dtype_is_float32_when_omitted_for_integer_value():
    """Omitting ``dtype`` resolves to float32 even for an integer-valued input."""
    sensor = DummyConstantSensor({"name": "c", "value": 7})
    assert sensor.get_observation().dtype == np.float32


def test_explicit_float32_dtype_is_respected():
    """An explicit ``dtype='float32'`` keeps the float32 dtype on a vector value."""
    sensor = DummyConstantSensor({"name": "c", "value": [1, 2, 3], "dtype": "float32"})
    obs = sensor.get_observation()
    assert obs.dtype == np.float32
    assert obs.shape == (3,)
    np.testing.assert_allclose(obs, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_vector_construction_preserves_int32_values_shape_dtype():
    """A list value with ``dtype='int32'`` constructs an int32 vector in order."""
    sensor = DummyConstantSensor({"name": "c", "value": [1, 2, 3], "dtype": "int32"})
    obs = sensor.get_observation()
    assert obs.dtype == np.int32
    assert obs.shape == (3,)
    np.testing.assert_array_equal(obs, np.array([1, 2, 3], dtype=np.int32))


# --------------------------------------------------------------------------- #
# Shape broadcasting
# --------------------------------------------------------------------------- #
def test_scalar_broadcast_to_shape_succeeds():
    """A scalar value broadcasts to the configured shape as float32."""
    sensor = DummyConstantSensor({"name": "c", "value": 1.5, "shape": [3]})
    obs = sensor.get_observation()
    assert obs.dtype == np.float32
    assert obs.shape == (3,)
    np.testing.assert_allclose(obs, np.array([1.5, 1.5, 1.5], dtype=np.float32))


def test_vector_broadcast_to_higher_dim_succeeds():
    """A length-1 vector broadcasts to a larger compatible shape."""
    sensor = DummyConstantSensor({"name": "c", "value": [1.0], "shape": [4]})
    obs = sensor.get_observation()
    assert obs.dtype == np.float32
    assert obs.shape == (4,)
    np.testing.assert_allclose(obs, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))


def test_int32_scalar_broadcast_preserves_dtype():
    """Broadcast preserves the configured int32 dtype across the requested shape."""
    sensor = DummyConstantSensor({"name": "c", "value": 2, "shape": [2], "dtype": "int32"})
    obs = sensor.get_observation()
    assert obs.dtype == np.int32
    assert obs.shape == (2,)
    np.testing.assert_array_equal(obs, np.array([2, 2], dtype=np.int32))


def test_incompatible_vector_shape_raises_value_error():
    """A value that cannot broadcast to the shape fails fast with ValueError."""
    with pytest.raises(ValueError):
        DummyConstantSensor({"name": "c", "value": [1.0, 2.0], "shape": [3]})


def test_incompatible_multidim_shape_raises_value_error():
    """An incompatible multi-dimensional reshape also raises ValueError."""
    with pytest.raises(ValueError):
        DummyConstantSensor({"name": "c", "value": [1.0, 2.0, 3.0], "shape": [2, 2]})


# --------------------------------------------------------------------------- #
# Observation preservation and stability
# --------------------------------------------------------------------------- #
def test_get_observation_is_stable_across_calls():
    """Repeated observation reads return the same constant value/shape/dtype."""
    sensor = DummyConstantSensor({"name": "c", "value": [1.0, 2.0], "shape": [2]})
    first = sensor.get_observation()
    second = sensor.get_observation()
    assert first.dtype == second.dtype == np.float32
    assert first.shape == second.shape == (2,)
    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# Non-mutation of the sensor's stored constant
# --------------------------------------------------------------------------- #
def _assert_constant_is_protected(sensor: DummyConstantSensor, expected: np.ndarray) -> None:
    """Callers must not be able to mutate the sensor's stored constant array.

    The stored constant is read-only, so direct mutation of the returned array
    is rejected and a fresh observation still equals the configured constant.
    """
    obs = sensor.get_observation()
    assert obs.flags.writeable is False
    with pytest.raises((ValueError, RuntimeError)):
        obs[...] = 999
    np.testing.assert_array_equal(sensor.get_observation(), expected)


def test_scalar_observation_cannot_be_mutated_by_caller():
    """The scalar branch exposes a read-only constant the caller cannot mutate."""
    sensor = DummyConstantSensor({"name": "c", "value": 1.5})
    _assert_constant_is_protected(sensor, np.array(1.5, dtype=np.float32))


def test_vector_observation_cannot_be_mutated_by_caller():
    """The vector branch exposes a read-only constant the caller cannot mutate."""
    sensor = DummyConstantSensor({"name": "c", "value": [1.0, 2.0, 3.0]})
    _assert_constant_is_protected(sensor, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_broadcast_observation_cannot_be_mutated_by_caller():
    """The shape-broadcast branch exposes a read-only constant the caller cannot mutate."""
    sensor = DummyConstantSensor({"name": "c", "value": 1.5, "shape": [3]})
    _assert_constant_is_protected(sensor, np.array([1.5, 1.5, 1.5], dtype=np.float32))


def test_int32_observation_cannot_be_mutated_by_caller():
    """Configured int32 observations are also read-only against caller mutation."""
    sensor = DummyConstantSensor({"name": "c", "value": [1, 2, 3], "dtype": "int32"})
    _assert_constant_is_protected(sensor, np.array([1, 2, 3], dtype=np.int32))


# --------------------------------------------------------------------------- #
# Lifecycle no-ops
# --------------------------------------------------------------------------- #
def test_reset_is_noop_and_preserves_constant():
    """``reset`` is a declared no-op and must not change the constant observation."""
    sensor = DummyConstantSensor({"name": "c", "value": 1.5})
    expected = np.array(1.5, dtype=np.float32)
    assert sensor.reset() is None
    np.testing.assert_array_equal(sensor.get_observation(), expected)


def test_step_is_noop_and_preserves_constant():
    """``step`` ignores the passed state and must not change the constant observation."""
    sensor = DummyConstantSensor({"name": "c", "value": [1.0, 2.0]})
    expected = np.array([1.0, 2.0], dtype=np.float32)
    assert sensor.step({"anything": object()}) is None
    np.testing.assert_array_equal(sensor.get_observation(), expected)


# --------------------------------------------------------------------------- #
# Factory delegation
# --------------------------------------------------------------------------- #
def test_factory_returns_dummy_constant_sensor_instance():
    """The public ``factory`` returns a fully constructed DummyConstantSensor."""
    sensor = factory({"name": "c", "value": 1.5})
    assert isinstance(sensor, DummyConstantSensor)


def test_factory_configures_float32_scalar_observation():
    """Factory wiring produces the configured scalar float32 observation."""
    sensor = factory({"name": "bias", "value": 0.5})
    obs = sensor.get_observation()
    assert isinstance(sensor, DummyConstantSensor)
    assert obs.dtype == np.float32
    assert obs.shape == ()
    assert obs == np.float32(0.5)


def test_factory_configures_int32_vector_with_shape():
    """Factory wiring honors value/shape/dtype for an int32 vector configuration."""
    config = {"name": "bias", "value": [1, 2, 3], "shape": [3], "dtype": "int32"}
    sensor = factory(config)
    obs = sensor.get_observation()
    assert isinstance(sensor, DummyConstantSensor)
    assert obs.dtype == np.int32
    assert obs.shape == (3,)
    np.testing.assert_array_equal(obs, np.array([1, 2, 3], dtype=np.int32))


def test_factory_observation_matches_direct_construction():
    """Factory output is equivalent to direct construction for the same config."""
    config = {"name": "bias", "value": 1.5, "shape": [4]}
    via_factory = factory(config).get_observation()
    via_direct = DummyConstantSensor(config).get_observation()
    assert via_factory.dtype == via_direct.dtype == np.float32
    assert via_factory.shape == via_direct.shape == (4,)
    np.testing.assert_array_equal(via_factory, via_direct)
