"""Tests for the tracking-precision drift mask + speed contract (issue #3480)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.tracking_precision_contract import (
    TRACKING_PRECISION_SCHEMA,
    TrackingPrecisionContract,
    apply_tracking_drift,
    drift_std_for_motp,
    is_contract_honored,
    minimum_separation,
    speed_cap_for_precision,
    tracking_precision_telemetry,
)


def test_motp_zero_is_exact_pass_through() -> None:
    """MOTP = 0 must return the ground-truth positions unchanged."""
    rng = np.random.default_rng(0)
    positions = np.array([[1.0, 2.0], [3.0, -4.0]])

    out = apply_tracking_drift(positions, 0.0, rng)

    assert np.array_equal(out, positions)
    assert out is not positions  # a copy, not the same buffer


def test_drift_std_matches_rayleigh_mapping() -> None:
    """The per-axis std must invert the Rayleigh mean so realized MOTP matches target."""
    assert drift_std_for_motp(2.5) == pytest.approx(2.5 / math.sqrt(math.pi / 2.0))
    assert drift_std_for_motp(0.0) == 0.0


def test_realized_mean_error_matches_target_motp() -> None:
    """A large drifted sample must have mean Euclidean error close to the target MOTP."""
    rng = np.random.default_rng(1234)
    target_motp = 2.5
    positions = np.zeros((20000, 2))

    drifted = apply_tracking_drift(positions, target_motp, rng)
    mean_error = float(np.mean(np.linalg.norm(drifted, axis=1)))

    assert mean_error == pytest.approx(target_motp, rel=0.03)


def test_drift_is_deterministic_for_a_seed() -> None:
    """The same seed must reproduce the same drift (reproducibility)."""
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    a = apply_tracking_drift(positions, 1.0, np.random.default_rng(7))
    b = apply_tracking_drift(positions, 1.0, np.random.default_rng(7))

    assert np.array_equal(a, b)


def test_minimum_separation_on_observed_vector() -> None:
    """Minimum separation must be computed against the provided (corrupted) actors."""
    actors = np.array([[3.0, 0.0], [0.0, 4.0], [10.0, 10.0]])

    assert minimum_separation(actors, [0.0, 0.0]) == pytest.approx(3.0)
    assert minimum_separation(np.empty((0, 2)), [0.0, 0.0]) == float("inf")


def test_speed_cap_drops_to_defensive_above_threshold() -> None:
    """The contract must drop the cap to the defensive ceiling at/above T_u."""
    contract = TrackingPrecisionContract()

    assert speed_cap_for_precision(1.0, contract) == contract.default_speed_cap
    assert speed_cap_for_precision(2.4999, contract) == contract.default_speed_cap
    assert speed_cap_for_precision(2.5, contract) == contract.defensive_speed_cap
    assert speed_cap_for_precision(5.0, contract) == contract.defensive_speed_cap


def test_contract_honored_and_violated() -> None:
    """A cap within the contracted ceiling is honored; exceeding it is a violation."""
    contract = TrackingPrecisionContract()

    # Degraded precision requires the defensive ceiling.
    assert is_contract_honored(contract.defensive_speed_cap, 3.0, contract)
    assert not is_contract_honored(contract.default_speed_cap, 3.0, contract)
    # Good precision permits the default cap.
    assert is_contract_honored(contract.default_speed_cap, 0.5, contract)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"precision_threshold_m": 0.0},
        {"default_speed_cap": 0.0},
        {"defensive_speed_cap": -1.0},
        {"default_speed_cap": 1.0, "defensive_speed_cap": 2.0},  # defensive > default
    ],
)
def test_invalid_contract_is_rejected(kwargs: dict[str, float]) -> None:
    """Non-physical contract parameters must fail closed at construction."""
    with pytest.raises(ValueError):
        TrackingPrecisionContract(**kwargs)


def test_negative_motp_is_rejected() -> None:
    """Negative MOTP is meaningless and must be rejected."""
    with pytest.raises(ValueError):
        drift_std_for_motp(-0.1)
    with pytest.raises(ValueError):
        speed_cap_for_precision(-0.1)


def test_telemetry_is_schema_tagged() -> None:
    """Telemetry must carry the schema tag, provenance label, and regime flags."""
    record = tracking_precision_telemetry(3.0, 2.0)

    assert record["schema_version"] == TRACKING_PRECISION_SCHEMA
    assert record["proxy_kind"] == "internal_non_hardware"
    assert record["defensive_regime"] is True
    assert record["contracted_speed_cap"] == TrackingPrecisionContract().defensive_speed_cap
    assert record["contract_honored"] is False  # applied 2.0 > defensive ceiling
