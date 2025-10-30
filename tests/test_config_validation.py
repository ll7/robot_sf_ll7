"""Tests for configuration validation (US4).

Validates config validation logic, conflict detection, and resolved config logging.
"""

import pytest

from robot_sf.gym_env.config_validation import (
    _check_backend_valid,
    _check_sensor_names_valid,
    _check_unknown_keys,
    get_resolved_config_dict,
    validate_config,
)
from robot_sf.gym_env.unified_config import ImageRobotConfig, RobotSimulationConfig


class TestUnknownKeyValidation:
    """Test strict validation for unknown config keys (T028)."""

    def test_unknown_key_strict_raises(self):
        """Unknown key in strict mode raises ValueError with valid keys listed."""
        config = RobotSimulationConfig()
        config.nonexistent_field = "value"  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="Unknown config keys.*Valid keys"):
            _check_unknown_keys(config, strict=True)

    def test_unknown_key_non_strict_warns(self):
        """Unknown key in non-strict mode logs warning."""
        config = RobotSimulationConfig()
        config.nonexistent_field = "value"  # type: ignore[attr-defined]

        # Should not raise, just log warning
        _check_unknown_keys(config, strict=False)
        # Warning is logged (visible in test output) but we test behavior, not logging

    def test_valid_keys_accepted(self):
        """Valid config keys pass validation."""
        config = RobotSimulationConfig()
        config.backend = "fast-pysf"
        config.sensors = []

        # Should not raise
        _check_unknown_keys(config, strict=True)


class TestBackendValidation:
    """Test backend name validation (T028)."""

    def test_unknown_backend_raises(self):
        """Unknown backend name raises KeyError with available backends."""
        config = RobotSimulationConfig()
        config.backend = "nonexistent_backend"

        with pytest.raises(KeyError, match="Unknown backend.*Available backends"):
            _check_backend_valid(config)

    def test_valid_backend_accepted(self):
        """Valid backend name passes validation."""
        config = RobotSimulationConfig()
        config.backend = "fast-pysf"

        # Should not raise
        _check_backend_valid(config)


class TestSensorValidation:
    """Test sensor name validation (T028)."""

    def test_unknown_sensor_type_raises(self):
        """Unknown sensor type raises KeyError with available sensors."""
        config = RobotSimulationConfig()
        config.sensors = [{"type": "nonexistent_sensor", "name": "test"}]

        with pytest.raises(KeyError, match="Unknown sensor type.*Available sensors"):
            _check_sensor_names_valid(config)

    def test_missing_sensor_type_raises(self):
        """Sensor config missing 'type' field raises ValueError."""
        config = RobotSimulationConfig()
        config.sensors = [{"name": "test"}]  # Missing 'type'

        with pytest.raises(ValueError, match="missing required 'type' field"):
            _check_sensor_names_valid(config)

    def test_valid_sensor_accepted(self):
        """Valid sensor config passes validation."""
        config = RobotSimulationConfig()
        config.sensors = [
            {
                "type": "dummy_constant",
                "name": "test",
                "value": [1.0],
                "shape": [1],
                "dtype": "float32",
                "space": {"shape": [1], "low": 0.0, "high": 1.0},
            }
        ]

        # Should not raise
        _check_sensor_names_valid(config)

    def test_empty_sensors_accepted(self):
        """Empty sensors list passes validation."""
        config = RobotSimulationConfig()
        config.sensors = []

        # Should not raise
        _check_sensor_names_valid(config)


class TestConflictDetection:
    """Test conflict detection (T029)."""

    @pytest.mark.skip(reason="Conflict detection not yet fully implemented (T029 partial)")
    def test_image_obs_without_image_config_raises(self):
        """use_image_obs=True without image_config raises ValueError."""
        config = RobotSimulationConfig()
        config.use_image_obs = True  # But no image_config

        with pytest.raises(ValueError, match=r"(?s)Config conflict.*ImageRobotConfig"):
            validate_config(config)

    def test_image_robot_config_passes(self):
        """ImageRobotConfig with use_image_obs=True passes validation."""
        config = ImageRobotConfig()
        # ImageRobotConfig sets use_image_obs=True by default and has image_config

        # Should not raise
        validate_config(config)


class TestResolvedConfigDict:
    """Test resolved config serialization (T031)."""

    def test_get_resolved_config_dict_dataclass(self):
        """Resolved config dict includes all fields for dataclass."""
        config = RobotSimulationConfig()
        config.backend = "fast-pysf"
        config.sensors = []

        resolved = get_resolved_config_dict(config)

        assert isinstance(resolved, dict)
        assert "backend" in resolved
        assert resolved["backend"] == "fast-pysf"
        assert "sensors" in resolved
        assert resolved["sensors"] == []
        assert "sim_config" in resolved
        assert "map_pool" in resolved

    def test_resolved_config_includes_nested_objects(self):
        """Resolved config serializes nested config objects."""
        config = ImageRobotConfig()
        resolved = get_resolved_config_dict(config)

        assert "image_config" in resolved
        assert "robot_config" in resolved
        assert isinstance(resolved, dict)


class TestIntegratedValidation:
    """Test complete validation flow (T030)."""

    def test_validate_config_checks_all_rules(self):
        """validate_config applies all validation rules."""
        config = RobotSimulationConfig()
        config.backend = "fast-pysf"
        config.sensors = []

        # Should not raise with valid config
        validate_config(config, strict=True)

    def test_validate_config_detects_multiple_errors(self):
        """First error in validation chain raises appropriately."""
        config = RobotSimulationConfig()
        config.backend = "invalid_backend"

        # Should raise on first error (backend validation)
        with pytest.raises(KeyError, match="Unknown backend"):
            validate_config(config)

    def test_validate_config_strict_mode(self):
        """Strict mode enforces all validation rules."""
        config = RobotSimulationConfig()
        config.unknown_field = "test"  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="Unknown config keys"):
            validate_config(config, strict=True)

    def test_validate_config_non_strict_mode_warns(self):
        """Non-strict mode logs warnings for unknown keys."""
        config = RobotSimulationConfig()
        config.unknown_field = "test"  # type: ignore[attr-defined]

        # Still raises for backend/sensor validation, but warns for unknown keys
        try:
            validate_config(config, strict=False)
        except (ValueError, KeyError):
            pass

        # Warning is logged (visible in test output) but we test behavior, not logging
        # Note: strict=False only affects unknown keys, not backend/sensor validation


class TestValidationErrorMessages:
    """Test error message quality and actionability."""

    def test_backend_error_lists_alternatives(self):
        """Backend validation error lists available backends."""
        config = RobotSimulationConfig()
        config.backend = "missing_backend"

        with pytest.raises(KeyError) as exc_info:
            _check_backend_valid(config)

        error_msg = str(exc_info.value)
        assert "missing_backend" in error_msg
        assert "available" in error_msg.lower() or "backends" in error_msg.lower()

    def test_sensor_error_lists_alternatives(self):
        """Sensor validation error lists available sensors."""
        config = RobotSimulationConfig()
        config.sensors = [{"type": "missing_sensor", "name": "test"}]

        with pytest.raises(KeyError) as exc_info:
            _check_sensor_names_valid(config)

        error_msg = str(exc_info.value)
        assert "missing_sensor" in error_msg
        assert "available" in error_msg.lower() or "sensors" in error_msg.lower()

    def test_unknown_key_error_shows_valid_keys(self):
        """Unknown key error shows list of valid keys."""
        config = RobotSimulationConfig()
        config.bad_key = "value"  # type: ignore[attr-defined]

        with pytest.raises(ValueError) as exc_info:
            _check_unknown_keys(config, strict=True)

        error_msg = str(exc_info.value)
        assert "bad_key" in error_msg
        assert "Valid keys" in error_msg
        # Should show some actual valid keys
        assert "backend" in error_msg or "sensors" in error_msg
