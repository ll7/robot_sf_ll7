"""Unit tests for ``TelemetryConfigMixin._validate_telemetry`` validation and normalization."""

from __future__ import annotations

import pytest

from robot_sf.gym_env.telemetry_config import TelemetryConfigMixin
from robot_sf.telemetry import DEFAULT_TELEMETRY_METRICS


def test_validate_telemetry_accepts_default_config_without_mutation() -> None:
    """A fully-default config validates and keeps its good values unchanged."""
    config = TelemetryConfigMixin()
    original_metrics = list(config.telemetry_metrics)

    config._validate_telemetry()

    assert config.telemetry_refresh_hz == 1.0
    assert config.telemetry_decimation == 1
    assert config.telemetry_pane_layout == "vertical_split"
    assert config.telemetry_metrics == original_metrics


def test_validate_telemetry_accepts_horizontal_split_layout() -> None:
    """``horizontal_split`` is a valid alternative layout and is preserved."""
    config = TelemetryConfigMixin(telemetry_pane_layout="horizontal_split")

    config._validate_telemetry()  # should not raise

    assert config.telemetry_pane_layout == "horizontal_split"


def test_validate_telemetry_preserves_custom_valid_metrics() -> None:
    """A list of valid metric strings survives validation unchanged and in order."""
    config = TelemetryConfigMixin(telemetry_metrics=["reward", "collisions"])

    config._validate_telemetry()

    assert config.telemetry_metrics == ["reward", "collisions"]


@pytest.mark.parametrize("refresh_hz", [0, -1, -0.5])
def test_validate_telemetry_rejects_non_positive_refresh_hz(refresh_hz: float) -> None:
    config = TelemetryConfigMixin(telemetry_refresh_hz=refresh_hz)

    with pytest.raises(ValueError, match="telemetry_refresh_hz must be > 0"):
        config._validate_telemetry()


@pytest.mark.parametrize("decimation", [0, -1, -5])
def test_validate_telemetry_rejects_non_positive_decimation(decimation: int) -> None:
    config = TelemetryConfigMixin(telemetry_decimation=decimation)

    with pytest.raises(ValueError, match="telemetry_decimation must be >= 1"):
        config._validate_telemetry()


@pytest.mark.parametrize("layout", ["", "diagonal", "Vertical_Split", "grid"])
def test_validate_telemetry_rejects_unknown_pane_layout(layout: str) -> None:
    config = TelemetryConfigMixin(telemetry_pane_layout=layout)

    with pytest.raises(ValueError, match="telemetry_pane_layout must be"):
        config._validate_telemetry()


def test_validate_telemetry_filters_blank_and_non_string_metrics() -> None:
    """Blank and non-string entries are dropped while valid strings are kept in order."""
    config = TelemetryConfigMixin(
        telemetry_metrics=["reward", "", "   ", 123, None, "collisions", "min_ped_distance"],
    )

    config._validate_telemetry()

    assert config.telemetry_metrics == ["reward", "collisions", "min_ped_distance"]


def test_validate_telemetry_restores_defaults_when_metrics_become_empty() -> None:
    """When filtering removes every entry, defaults are restored."""
    config = TelemetryConfigMixin(telemetry_metrics=["", "   ", 42, None])

    config._validate_telemetry()

    assert config.telemetry_metrics == list(DEFAULT_TELEMETRY_METRICS)


def test_validate_telemetry_restores_defaults_when_metrics_is_none() -> None:
    """A ``None`` metrics list exercises the ``self.telemetry_metrics or []`` fallback."""
    config = TelemetryConfigMixin(telemetry_metrics=None)  # type: ignore[arg-type]

    config._validate_telemetry()

    assert config.telemetry_metrics == list(DEFAULT_TELEMETRY_METRICS)
