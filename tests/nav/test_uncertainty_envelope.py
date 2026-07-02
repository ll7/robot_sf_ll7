"""Tests for the pedestrian uncertainty-envelope abstraction (issue #4141)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from robot_sf.nav import (
    ENVELOPE_SCHEMA_VERSION,
    ConformalInflationPolicy,
    PedestrianUncertaintyEnvelope,
    effective_pedestrian_radius,
    envelope_diagnostics,
    envelope_from_position,
    linear_inflation_policy,
)

# --------------------------------------------------------------------------- #
# linear_inflation_policy geometry
# --------------------------------------------------------------------------- #


def test_linear_inflation_policy_zero_step_returns_zero() -> None:
    """r_eff(0) must equal base_radius: the linear policy adds nothing at step 0."""
    policy = linear_inflation_policy(alpha=0.1, dt=0.25)
    assert policy(0) == 0.0


def test_linear_inflation_policy_strictly_increases_for_alpha_positive() -> None:
    """With alpha > 0 the added radius grows strictly with the horizon step."""
    policy = linear_inflation_policy(alpha=0.1, dt=0.25)
    added = [policy(i) for i in range(5)]
    assert all(later > earlier for earlier, later in pairwise(added))


def test_linear_inflation_policy_matches_closed_form() -> None:
    """The inflation equals alpha * horizon_step * dt exactly."""
    alpha, dt = 0.2, 0.5
    policy = linear_inflation_policy(alpha=alpha, dt=dt)
    for i in range(4):
        assert policy(i) == pytest.approx(alpha * i * dt)


def test_linear_inflation_policy_alpha_zero_is_constant_zero() -> None:
    """alpha == 0.0 is the deterministic fallback: no inflation at any step."""
    policy = linear_inflation_policy(alpha=0.0, dt=0.25)
    assert all(policy(i) == 0.0 for i in range(10))


@pytest.mark.parametrize("bad_alpha", [-0.1, float("nan"), float("inf")])
def test_linear_inflation_policy_rejects_negative_alpha(bad_alpha: float) -> None:
    """Negative or non-finite alpha is invalid and must fail closed."""
    with pytest.raises(ValueError):
        linear_inflation_policy(alpha=bad_alpha, dt=0.25)


@pytest.mark.parametrize("bad_dt", [0.0, -0.25, float("nan"), float("inf")])
def test_linear_inflation_policy_rejects_nonpositive_dt(bad_dt: float) -> None:
    """Non-positive or non-finite dt is invalid and must fail closed."""
    with pytest.raises(ValueError):
        linear_inflation_policy(alpha=0.1, dt=bad_dt)


def test_linear_inflation_policy_rejects_negative_horizon_step() -> None:
    """Querying a negative horizon step is a programming error and must raise."""
    policy = linear_inflation_policy(alpha=0.1, dt=0.25)
    with pytest.raises(ValueError):
        policy(-1)


# --------------------------------------------------------------------------- #
# PedestrianUncertaintyEnvelope
# --------------------------------------------------------------------------- #


def test_pedestrian_uncertainty_envelope_is_importable_from_robot_sf_nav() -> None:
    """The public type must be importable from the ``robot_sf.nav`` package."""
    from robot_sf import nav

    assert "PedestrianUncertaintyEnvelope" in nav.__all__
    assert nav.PedestrianUncertaintyEnvelope is PedestrianUncertaintyEnvelope


def test_pedestrian_uncertainty_envelope_effective_radius_at_zero_equals_base_radius() -> None:
    """Step 0 returns the base radius under the standard linear policy."""
    env = PedestrianUncertaintyEnvelope(
        position=(1.0, 2.0),
        base_radius=0.4,
        spatial_inflation=linear_inflation_policy(alpha=0.1, dt=0.25),
    )
    assert env.effective_radius(0) == pytest.approx(0.4)
    assert env.effective_radius(1) == pytest.approx(0.425)
    assert env.effective_radius(2) > env.effective_radius(1)


def test_pedestrian_uncertainty_envelope_rejects_nonfinite_position() -> None:
    """A non-finite position component is invalid."""
    with pytest.raises(ValueError):
        PedestrianUncertaintyEnvelope(
            position=(float("nan"), 0.0),
            base_radius=0.4,
            spatial_inflation=linear_inflation_policy(alpha=0.1, dt=0.25),
        )


@pytest.mark.parametrize("bad_radius", [-0.1, float("nan"), float("inf")])
def test_pedestrian_uncertainty_envelope_rejects_invalid_base_radius(bad_radius: float) -> None:
    """A negative or non-finite base radius is invalid."""
    with pytest.raises(ValueError):
        PedestrianUncertaintyEnvelope(
            position=(0.0, 0.0),
            base_radius=bad_radius,
            spatial_inflation=linear_inflation_policy(alpha=0.1, dt=0.25),
        )


def test_pedestrian_uncertainty_envelope_rejects_negative_inflation_policy() -> None:
    """A policy that returns a negative inflation at step 0 must fail closed."""
    with pytest.raises(ValueError):
        PedestrianUncertaintyEnvelope(
            position=(0.0, 0.0),
            base_radius=0.4,
            spatial_inflation=lambda _step: -1.0,
        )


def test_envelope_from_position_uses_linear_policy() -> None:
    """The convenience factory builds a linear-policy envelope at ``position``."""
    env = envelope_from_position((0.0, 0.0), 0.25, alpha=0.1, dt=0.5)
    assert env.effective_radius(0) == pytest.approx(0.25)
    assert env.effective_radius(4) == pytest.approx(0.25 + 0.1 * 4 * 0.5)


# --------------------------------------------------------------------------- #
# effective_pedestrian_radius helper (planner substitution point)
# --------------------------------------------------------------------------- #


def test_effective_pedestrian_radius_disabled_returns_base_radius() -> None:
    """A disabled envelope is a bit-for-bit no-op regardless of horizon step."""
    for step in range(5):
        r = effective_pedestrian_radius(
            base_radius=0.4, horizon_step=step, alpha=0.1, dt=0.25, enabled=False
        )
        assert r == 0.4


def test_effective_pedestrian_radius_enabled_inflates_with_horizon() -> None:
    """An enabled positive-alpha envelope inflates with the horizon step."""
    r0 = effective_pedestrian_radius(
        base_radius=0.4, horizon_step=0, alpha=0.1, dt=0.25, enabled=True
    )
    r2 = effective_pedestrian_radius(
        base_radius=0.4, horizon_step=2, alpha=0.1, dt=0.25, enabled=True
    )
    assert r0 == pytest.approx(0.4)
    assert r2 == pytest.approx(0.4 + 0.1 * 2 * 0.25)
    assert r2 > r0


def test_effective_pedestrian_radius_enabled_alpha_zero_matches_base() -> None:
    """Enabled with alpha == 0.0 still equals the base radius at every step."""
    for step in range(5):
        r = effective_pedestrian_radius(
            base_radius=0.4, horizon_step=step, alpha=0.0, dt=0.25, enabled=True
        )
        assert r == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Diagnostics payload and stub interface
# --------------------------------------------------------------------------- #


def test_envelope_diagnostics_records_settings_and_claim_boundary() -> None:
    """The diagnostics payload records the schema version, settings, and boundary."""
    payload = envelope_diagnostics(enabled=True, alpha=0.1, dt=0.25)
    assert payload["schema_version"] == ENVELOPE_SCHEMA_VERSION
    assert payload["enabled"] is True
    assert payload["policy"] == "linear"
    assert payload["alpha_mps"] == pytest.approx(0.1)
    assert payload["dt"] == pytest.approx(0.25)
    assert "not conformal calibration" in payload["claim_boundary"]


def test_envelope_diagnostics_disabled_is_deterministic_policy() -> None:
    """A disabled or zero-alpha envelope is reported as a deterministic policy."""
    assert envelope_diagnostics(enabled=False, alpha=0.1, dt=0.25)["policy"] == "deterministic"
    assert envelope_diagnostics(enabled=True, alpha=0.0, dt=0.25)["policy"] == "deterministic"


def test_linear_policy_satisfies_conformal_stub_protocol() -> None:
    """Any inflation callable structurally satisfies the future conformal seam."""
    policy = linear_inflation_policy(alpha=0.1, dt=0.25)
    assert isinstance(policy, ConformalInflationPolicy)
