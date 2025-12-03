"""Tests for sensor registry and fusion adapter."""
# Pytest fixtures intentionally shadow fixture names

import pytest

from robot_sf.sensor.fusion_adapter import (
    create_sensors_from_config,
    validate_sensor_configs,
)
from robot_sf.sensor.registry import (
    get_sensor,
    list_sensors,
    register_sensor,
    unregister_sensor,
)


class DummySensor:
    """Test sensor implementation."""

    def __init__(self, config):
        self.config = config
        self.observation = None

    def reset(self) -> None:
        self.observation = None

    def step(self, state) -> None:
        self.observation = state

    def get_observation(self):
        return self.observation


@pytest.fixture
def clean_registry():
    """Clean registry before and after each test."""
    # Store original registry state
    original_sensors = list(list_sensors().keys())

    yield

    # Clean up any test sensors
    current_sensors = list(list_sensors().keys())
    for sensor_name in current_sensors:
        if sensor_name not in original_sensors:
            unregister_sensor(sensor_name)


def test_register_and_get_sensor(clean_registry):
    """Test registering and retrieving a sensor."""
    register_sensor("test_dummy", DummySensor)

    factory = get_sensor("test_dummy")
    sensor = factory({"param": "value"})

    assert isinstance(sensor, DummySensor)
    assert sensor.config == {"param": "value"}


def test_register_duplicate_raises_error(clean_registry):
    """Test that registering duplicate sensor name raises error."""

    def dummy_factory(config):
        return DummySensor(config)

    register_sensor("test_duplicate", dummy_factory)

    with pytest.raises(ValueError, match="already registered"):
        register_sensor("test_duplicate", dummy_factory)


def test_register_duplicate_with_override(clean_registry):
    """Test that override=True allows re-registration."""

    def factory1(config):
        return DummySensor(config)

    def factory2(config):
        return DummySensor({**config, "overridden": True})

    register_sensor("test_override", factory1)
    register_sensor("test_override", factory2, override=True)

    factory = get_sensor("test_override")
    sensor = factory({"test": "value"})

    assert "overridden" in sensor.config


def test_get_unknown_sensor_raises_error(clean_registry):
    """Test that getting unknown sensor raises KeyError with suggestions."""
    register_sensor("known1", DummySensor)
    register_sensor("known2", DummySensor)

    with pytest.raises(KeyError, match="Unknown sensor 'unknown'"):
        get_sensor("unknown")


def test_list_sensors(clean_registry):
    """Test listing all registered sensors."""
    register_sensor("sensor1", DummySensor)
    register_sensor("sensor2", DummySensor)

    sensors = list_sensors()

    assert "sensor1" in sensors
    assert "sensor2" in sensors


def test_unregister_sensor(clean_registry):
    """Test unregistering a sensor."""
    register_sensor("test_unregister", DummySensor)

    assert "test_unregister" in list_sensors()

    unregister_sensor("test_unregister")

    assert "test_unregister" not in list_sensors()


def test_unregister_unknown_raises_error(clean_registry):
    """Test that unregistering unknown sensor raises error."""
    with pytest.raises(KeyError, match="Cannot unregister unknown sensor"):
        unregister_sensor("nonexistent")


def test_create_sensors_from_config(clean_registry):
    """Test creating sensors from configuration list."""
    register_sensor("dummy", DummySensor)

    configs = [
        {"type": "dummy", "param1": "value1"},
        {"type": "dummy", "param2": "value2"},
    ]

    sensors = create_sensors_from_config(configs)

    assert len(sensors) == 2
    assert sensors[0].config["param1"] == "value1"
    assert sensors[1].config["param2"] == "value2"


def test_create_sensors_missing_type_raises_error(clean_registry):
    """Test that missing 'type' key raises ValueError."""
    configs = [
        {"param": "value"},  # Missing "type"
    ]

    with pytest.raises(ValueError, match="missing 'type' key"):
        create_sensors_from_config(configs)


def test_create_sensors_unknown_type_raises_error(clean_registry):
    """Test that unknown sensor type raises KeyError."""
    register_sensor("known", lambda c: DummySensor(c))

    configs = [
        {"type": "unknown"},
    ]

    with pytest.raises(KeyError, match="Unknown sensor 'unknown'"):
        create_sensors_from_config(configs)


def test_validate_sensor_configs_valid(clean_registry):
    """Test validation with valid configs."""
    register_sensor("valid", DummySensor)

    configs = [
        {"type": "valid", "param": "value"},
    ]

    errors = validate_sensor_configs(configs)

    assert len(errors) == 0


def test_validate_sensor_configs_missing_type(clean_registry):
    """Test validation catches missing 'type' key."""
    configs = [
        {"param": "value"},  # Missing type
    ]

    errors = validate_sensor_configs(configs)

    assert len(errors) == 1
    assert "missing 'type' key" in errors[0]


def test_validate_sensor_configs_unknown_type(clean_registry):
    """Test validation catches unknown sensor type."""
    register_sensor("known", DummySensor)

    configs = [
        {"type": "unknown"},
    ]

    errors = validate_sensor_configs(configs)

    assert len(errors) == 1
    assert "unknown sensor type 'unknown'" in errors[0]


def test_sensor_protocol_conformance():
    """Test that DummySensor conforms to Sensor protocol."""
    sensor = DummySensor({"test": "config"})

    # Should have all required methods
    assert hasattr(sensor, "reset")
    assert hasattr(sensor, "step")
    assert hasattr(sensor, "get_observation")

    # Test basic lifecycle
    sensor.reset()
    assert sensor.observation is None

    sensor.step("test_state")
    assert sensor.observation == "test_state"

    obs = sensor.get_observation()
    assert obs == "test_state"
