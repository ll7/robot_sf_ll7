"""Tests for the internal-proxy rollover stability margin (issue #3479)."""

from __future__ import annotations

import pytest

from robot_sf.robot.rollover_proxy import (
    PROXY_SCHEMA_VERSION,
    RolloverProxyParams,
    critical_lateral_acceleration,
    is_rollover_critical,
    lateral_acceleration,
    rollover_proxy_telemetry,
    stability_margin,
)

# a_y,crit = g * (t_w / (2 h_c)) * (a / L) for the documented default proxy geometry.
_DEFAULT_CRIT = 9.81 * (0.50 / (2 * 0.40)) * (0.30 / 0.60)


def test_critical_lateral_acceleration_matches_closed_form() -> None:
    """The critical lateral acceleration must match the issue's closed form."""
    assert critical_lateral_acceleration(RolloverProxyParams()) == pytest.approx(_DEFAULT_CRIT)


def test_lateral_acceleration_is_v_times_omega() -> None:
    """Proxy lateral acceleration must be ``v · ω``."""
    assert lateral_acceleration(1.5, 2.0) == pytest.approx(3.0)


def test_feasible_command_stays_stable() -> None:
    """A low-speed, low-yaw command must stay well within the proxy threshold."""
    margin = stability_margin(0.5, 0.5)  # a_y = 0.25 << a_y,crit ~ 3.07

    assert margin == pytest.approx(1.0 - 0.25 / _DEFAULT_CRIT)
    assert not is_rollover_critical(margin)


def test_over_yaw_command_trips_rollover_critical() -> None:
    """An over-yaw command exceeding the critical accel must trip ROLLOVER_CRITICAL."""
    margin = stability_margin(2.0, 2.0)  # a_y = 4.0 > a_y,crit ~ 3.07

    assert margin == 0.0
    assert is_rollover_critical(margin)


def test_margin_is_clamped_to_unit_interval() -> None:
    """Zero motion yields full margin; gross over-demand clamps to zero."""
    assert stability_margin(0.0, 0.0) == 1.0
    assert stability_margin(100.0, 100.0) == 0.0


def test_margin_at_exact_threshold_is_critical() -> None:
    """Demanding exactly the critical lateral acceleration yields a zero margin."""
    crit = critical_lateral_acceleration(RolloverProxyParams())
    # Pick v, omega whose product equals the critical lateral acceleration.
    margin = stability_margin(crit, 1.0)

    assert margin == pytest.approx(0.0, abs=1e-12)
    assert is_rollover_critical(margin)


def test_margin_decreases_monotonically_with_demand() -> None:
    """Higher |v · ω| must never increase the stability margin."""
    margins = [stability_margin(v, 1.0) for v in (0.0, 0.5, 1.0, 2.0, 3.0)]
    assert margins == sorted(margins, reverse=True)


def test_margin_depends_on_magnitude_not_sign() -> None:
    """Opposite yaw signs of equal magnitude must give the same margin."""
    assert stability_margin(1.0, 1.5) == pytest.approx(stability_margin(1.0, -1.5))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"track_width_m": 0.0},
        {"cog_height_m": -0.1},
        {"front_axle_to_cog_m": 0.0},
        {"wheelbase_m": -1.0},
        {"gravity_m_s2": 0.0},
        {"front_axle_to_cog_m": 0.7, "wheelbase_m": 0.6},  # a > L
    ],
)
def test_invalid_params_are_rejected(kwargs: dict[str, float]) -> None:
    """Non-physical proxy geometry must fail closed at construction."""
    with pytest.raises(ValueError):
        RolloverProxyParams(**kwargs)


def test_telemetry_is_schema_tagged_and_labels_non_hardware() -> None:
    """Telemetry must carry the schema version and the internal-proxy provenance label."""
    record = rollover_proxy_telemetry(2.0, 2.0)

    assert record["schema_version"] == PROXY_SCHEMA_VERSION
    assert record["proxy_kind"] == "internal_non_hardware"
    assert record["rollover_critical"] is True
    assert record["stability_margin"] == 0.0
    assert record["lateral_acceleration"] == pytest.approx(4.0)
    assert record["critical_lateral_acceleration"] == pytest.approx(_DEFAULT_CRIT)
