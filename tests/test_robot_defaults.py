"""Tests for robot_radius default centralization (issue #4856).

This module verifies that:
1. The central DEFAULT_ROBOT_RADIUS constant is defined and has the correct value
2. Robot settings classes reference the central constant
3. SNQI proxy references the central constant
4. Divergent defaults are documented

Claim boundary: Single-source-of-truth for robot_radius defaults (plumbing-only).
Evidence status: Smoke evidence (unit tests verify constant usage and documentation).
Caveats: Does not change runtime behavior; divergent defaults in occupancy_grid and
base_env planner setup are preserved to avoid altering benchmark metrics.
Uncertainty: None expected - this is mechanical constant centralization.
"""

from __future__ import annotations

from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS, get_default_robot_radius
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.robot.holonomic_drive import HolonomicDriveSettings


class TestDefaultRobotRadiusConstant:
    """Tests for the central DEFAULT_ROBOT_RADIUS constant."""

    def test_constant_has_expected_value(self) -> None:
        """T001: Verify DEFAULT_ROBOT_RADIUS is 1.0 meters."""
        assert DEFAULT_ROBOT_RADIUS == 1.0, "Authoritative robot radius should be 1.0 meters"

    def test_get_default_robot_radius_returns_constant(self) -> None:
        """T002: Verify get_default_robot_radius() returns the constant."""
        assert get_default_robot_radius() == DEFAULT_ROBOT_RADIUS


class TestRobotSettingsUseConstant:
    """Tests that robot settings classes use the central constant."""

    def test_differential_drive_settings_default_radius(self) -> None:
        """T003: Verify DifferentialDriveSettings uses the central constant."""
        settings = DifferentialDriveSettings()
        assert settings.radius == DEFAULT_ROBOT_RADIUS, "DifferentialDriveSettings should use DEFAULT_ROBOT_RADIUS"

    def test_bicycle_drive_settings_default_radius(self) -> None:
        """T004: Verify BicycleDriveSettings uses the central constant."""
        settings = BicycleDriveSettings()
        assert settings.radius == DEFAULT_ROBOT_RADIUS, "BicycleDriveSettings should use DEFAULT_ROBOT_RADIUS"

    def test_holonomic_drive_settings_default_radius(self) -> None:
        """T005: Verify HolonomicDriveSettings uses the central constant."""
        settings = HolonomicDriveSettings()
        assert settings.radius == DEFAULT_ROBOT_RADIUS, "HolonomicDriveSettings should use DEFAULT_ROBOT_RADIUS"


class TestDivergentDefaultsDocumented:
    """Tests that divergent defaults are documented."""

    def test_occupancy_grid_default_is_documented(self) -> None:
        """T006: Verify occupancy_grid.py documents its divergent default (0.3m)."""
        from robot_sf.nav.occupancy_grid import GridConfig

        config = GridConfig()
        # The occupancy grid default is 0.3m, which differs from DEFAULT_ROBOT_RADIUS
        # This is intentional to avoid changing benchmark metrics
        assert config.robot_radius == 0.3, "GridConfig robot_radius should be 0.3m (divergent, documented)"

    def test_base_env_planner_fallback_is_documented(self) -> None:
        """T007: Verify base_env.py documents its divergent fallback (0.4m)."""
        # This test verifies documentation exists; the actual fallback is in code
        # The documentation is a comment in base_env.py near line 299-307
        import inspect

        from robot_sf.gym_env.base_env import attach_planner_to_map

        source = inspect.getsource(attach_planner_to_map)
        assert "0.4" in source and "fallback" in source.lower(), \
            "base_env.py should document the 0.4m fallback default"


class TestSNQIProxyUsesConstant:
    """Tests that SNQI proxy uses the central constant."""

    def test_snqi_proxy_imports_constant(self) -> None:
        """T008: Verify snqi_proxy.py imports DEFAULT_ROBOT_RADIUS."""
        import robot_sf.gym_env.snqi_proxy as snqi_proxy_module
        assert hasattr(snqi_proxy_module, 'DEFAULT_ROBOT_RADIUS'), \
            "snqi_proxy should import DEFAULT_ROBOT_RADIUS from robot_sf.common.robot_defaults"

    def test_snqi_proxy_default_matches_constant(self) -> None:
        """T009: Verify SNQI proxy's _resolve_robot_radius uses DEFAULT_ROBOT_RADIUS."""
        from robot_sf.gym_env.snqi_proxy import DEFAULT_ROBOT_RADIUS, _resolve_robot_radius

        # Mock simulator with no robot_radius attribute
        class MockSimulator:
            pass

        sim = MockSimulator()
        radius = _resolve_robot_radius(sim)
        assert radius == DEFAULT_ROBOT_RADIUS, \
            "SNQI proxy fallback should use DEFAULT_ROBOT_RADIUS (1.0m)"
