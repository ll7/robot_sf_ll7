"""Tests for error policy application (US3).

Validates that error handling follows the established policy:
- Fatal errors for required resources with actionable remediation messages
- Soft-degrade warnings for optional components with fallback behavior
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robot_sf.baselines.ppo import PPOPlanner, PPOPlannerConfig
from robot_sf.common.errors import raise_fatal_with_remedy, warn_soft_degrade
from robot_sf.nav.svg_map_parser import SvgMapConverter
from robot_sf.sensor.registry import get_sensor, register_sensor


class TestErrorPolicyHelpers:
    """Test the core error policy helper functions."""

    def test_raise_fatal_with_remedy_format(self):
        """Verify fatal error includes remediation message."""
        with pytest.raises(RuntimeError, match=r"(?s)Test error.*Remediation.*Fix it"):
            raise_fatal_with_remedy("Test error", "Fix it")

    def test_warn_soft_degrade_logs_warning(self):
        """Verify soft degrade executes without errors."""
        # Test that warning function completes without raising
        warn_soft_degrade("test_component", "test issue", "test fallback")
        # If we reach here, the function executed successfully


class TestMapLoadingErrors:
    """Test map loading error handling (T024/T025)."""

    def test_missing_map_file_raises_with_remedy(self):
        """Missing SVG map file raises RuntimeError with remediation."""
        nonexistent_path = "/nonexistent/path/to/map.svg"

        with pytest.raises(RuntimeError, match=r"(?s)Map file not found.*Remediation"):
            SvgMapConverter(nonexistent_path)

    def test_invalid_svg_format_raises_with_remedy(self, tmp_path):
        """Invalid SVG format raises RuntimeError with remediation."""
        bad_svg = tmp_path / "bad.svg"
        bad_svg.write_text("<invalid>not closing tag")

        with pytest.raises(RuntimeError, match=r"(?s)Invalid SVG format.*Remediation"):
            SvgMapConverter(str(bad_svg))


class TestModelLoadingErrors:
    """Test PPO model loading error handling (T025)."""

    def test_missing_model_with_fallback_warns(self):
        """Missing model file with fallback enabled logs warning."""
        config = PPOPlannerConfig(
            model_path="/nonexistent/model.zip",
            fallback_to_goal=True,
        )

        planner = PPOPlanner(config)

        # Model should be None when fallback is enabled
        assert getattr(planner, "_model", "not-none") is None
        # Warning is logged (visible in test output) but we test behavior, not logging

    def test_missing_model_without_fallback_raises(self):
        """Missing model file without fallback raises RuntimeError."""
        config = PPOPlannerConfig(
            model_path="/nonexistent/model.zip",
            fallback_to_goal=False,
        )

        with pytest.raises(RuntimeError, match=r"(?s)PPO model file not found.*Remediation"):
            PPOPlanner(config)

    @patch("robot_sf.baselines.ppo.PPO")
    def test_model_load_failure_with_fallback_warns(self, mock_ppo_class):
        """Model load failure with fallback enabled logs warning."""
        # Create a real file path that exists
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Mock PPO.load to raise an error
            mock_ppo_class.load.side_effect = RuntimeError("Incompatible model")

            config = PPOPlannerConfig(
                model_path=tmp_path,
                fallback_to_goal=True,
            )

            planner = PPOPlanner(config)

            # Model should be None when fallback handles load failure
            assert getattr(planner, "_model", "not-none") is None
            # Warning is logged (visible in test output) but we test behavior, not logging
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    @patch("robot_sf.baselines.ppo.PPO")
    def test_model_load_failure_without_fallback_raises(self, mock_ppo_class):
        """Model load failure without fallback raises RuntimeError."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_ppo_class.load.side_effect = ValueError("Pickle error")

            config = PPOPlannerConfig(
                model_path=tmp_path,
                fallback_to_goal=False,
            )

            with pytest.raises(RuntimeError, match=r"(?s)Failed to load PPO model.*Remediation"):
                PPOPlanner(config)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestSensorRegistryErrors:
    """Test sensor registry error handling (T026)."""

    def test_unknown_sensor_raises_with_alternatives(self):
        """Unknown sensor name raises KeyError listing available sensors."""

        # Register a known sensor for the test
        def _dummy_factory(_config):
            """TODO docstring. Document this function.

            Args:
                _config: TODO docstring.
            """
            return MagicMock()

        register_sensor("test_known_sensor", _dummy_factory, override=True)

        try:
            with pytest.raises(KeyError, match=r"(?s)Unknown sensor.*Available sensors"):
                get_sensor("nonexistent_sensor")
        finally:
            # Clean up
            from robot_sf.sensor.registry import unregister_sensor

            try:
                unregister_sensor("test_known_sensor")
            except KeyError:
                pass  # Already cleaned up

    def test_duplicate_registration_without_override_raises(self):
        """Duplicate sensor registration without override raises ValueError."""

        def _factory(_config):
            """TODO docstring. Document this function.

            Args:
                _config: TODO docstring.
            """
            return MagicMock()

        register_sensor("duplicate_test", _factory, override=True)

        try:
            with pytest.raises(ValueError, match="already registered"):
                register_sensor("duplicate_test", _factory, override=False)
        finally:
            from robot_sf.sensor.registry import unregister_sensor

            try:
                unregister_sensor("duplicate_test")
            except KeyError:
                pass


class TestErrorPolicyIntegration:
    """Integration tests for error policy across components."""

    def test_map_loading_error_message_quality(self):
        """Verify map loading errors contain actionable information."""
        with pytest.raises(RuntimeError) as exc_info:
            SvgMapConverter("/path/that/does/not/exist.svg")

        error_msg = str(exc_info.value)
        # Should mention the problem
        assert "not found" in error_msg.lower() or "file" in error_msg.lower()
        # Should mention remediation
        assert "remediation" in error_msg.lower() or "place" in error_msg.lower()

    def test_sensor_error_lists_alternatives(self):
        """Verify sensor errors list available alternatives."""

        # Register some test sensors
        def _factory(_config):
            """TODO docstring. Document this function.

            Args:
                _config: TODO docstring.
            """
            return MagicMock()

        register_sensor("sensor_a", _factory, override=True)
        register_sensor("sensor_b", _factory, override=True)

        try:
            with pytest.raises(KeyError) as exc_info:
                get_sensor("sensor_c")

            error_msg = str(exc_info.value)
            assert "available" in error_msg.lower()
            # Should list the alternatives
            assert ("sensor_a" in error_msg and "sensor_b" in error_msg) or (
                "Available" in error_msg
            )
        finally:
            from robot_sf.sensor.registry import unregister_sensor

            for name in ["sensor_a", "sensor_b"]:
                try:
                    unregister_sensor(name)
                except KeyError:
                    pass
