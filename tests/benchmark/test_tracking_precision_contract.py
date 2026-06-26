"""Tests for the tracking-precision drift mask + speed contract (issue #3480)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.tracking_precision_contract import (
    TRACKING_PRECISION_SCHEMA,
    TrackingPrecisionContract,
    apply_speed_contract,
    apply_tracking_drift,
    apply_tracking_precision_spec,
    drift_std_for_motp,
    is_contract_honored,
    make_tracking_precision_rng,
    minimum_separation,
    normalize_tracking_precision_spec,
    speed_cap_for_precision,
    tracking_precision_hash,
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


def test_normalized_spec_is_default_off_and_hash_stable() -> None:
    """Absent spec preserves current behavior and hashes canonically."""
    normalized = normalize_tracking_precision_spec(None)

    assert normalized["enabled"] is False
    assert normalized["target_motp_m"] == 0.0
    assert normalized["speed_contract"]["mode"] == "diagnostic"
    assert tracking_precision_hash(normalized) == tracking_precision_hash(
        normalize_tracking_precision_spec(dict(reversed(list(normalized.items()))))
    )


def test_normalized_spec_accepts_issue_aliases() -> None:
    """Issue #3480 speed-contract aliases normalize to one operational contract."""
    normalized = normalize_tracking_precision_spec(
        {
            "target_motp_m": 3.0,
            "speed_contract": {
                "precision_threshold_m": 2.0,
                "default_speed_cap": 1.4,
                "defensive_speed_cap": 0.4,
                "mode": "clamp",
            },
            "seed_salt": 17,
        }
    )

    assert normalized["enabled"] is True
    assert normalized["speed_contract"] == {
        "threshold_m": 2.0,
        "default_speed": 1.4,
        "defensive_speed": 0.4,
        "mode": "clamp",
    }
    assert normalized["seed_salt"] == 17


@pytest.mark.parametrize(
    "spec",
    [
        {"speed_contract": {"mode": "unknown"}},
        {"target_motp_m": -0.1},
        {"seed_salt": -1},
        {"unexpected": True},
        {"speed_contract": {"unexpected": True}},
    ],
)
def test_invalid_spec_is_rejected(spec: dict[str, object]) -> None:
    """Malformed run specs fail before they can enter benchmark provenance."""
    with pytest.raises((TypeError, ValueError)):
        normalize_tracking_precision_spec(spec)


def test_tracking_precision_rng_is_episode_deterministic() -> None:
    """Spec, scenario, and seed produce stable but distinct drift streams."""
    spec = normalize_tracking_precision_spec({"target_motp_m": 2.5, "seed_salt": 4})
    positions = np.zeros((3, 2))

    rng_a = make_tracking_precision_rng(spec, seed=12, scenario_id="map-a")
    rng_b = make_tracking_precision_rng(spec, seed=12, scenario_id="map-a")
    rng_c = make_tracking_precision_rng(spec, seed=12, scenario_id="map-b")

    drift_a = apply_tracking_precision_spec(positions, spec, rng_a)
    drift_b = apply_tracking_precision_spec(positions, spec, rng_b)
    drift_c = apply_tracking_precision_spec(positions, spec, rng_c)

    assert np.array_equal(drift_a, drift_b)
    assert not np.array_equal(drift_a, drift_c)


def test_tracking_precision_spec_zero_or_disabled_passes_through() -> None:
    """Default-off specs do not corrupt observations before explicit opt-in."""
    rng = np.random.default_rng(42)
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])

    disabled = apply_tracking_precision_spec(positions, None, rng)
    zero_motp = apply_tracking_precision_spec(
        positions,
        {"enabled": True, "target_motp_m": 0.0},
        rng,
    )

    assert np.array_equal(disabled, positions)
    assert np.array_equal(zero_motp, positions)
    assert disabled is not positions
    assert zero_motp is not positions


def test_speed_contract_diagnostic_mode_records_without_clamping() -> None:
    """Diagnostic mode preserves command speed and marks contract violation."""
    applied, record = apply_speed_contract(
        1.5,
        3.0,
        {
            "enabled": True,
            "target_motp_m": 3.0,
            "speed_contract": {"defensive_speed": 0.5, "mode": "diagnostic"},
        },
    )

    assert applied == 1.5
    assert record["contract_mode"] == "diagnostic"
    assert record["tracking_precision_enabled"] is True
    assert record["contract_honored"] is False


def test_speed_contract_clamp_mode_applies_defensive_ceiling() -> None:
    """Clamp mode applies the defensive cap once MOTP crosses the threshold."""
    applied, record = apply_speed_contract(
        1.5,
        3.0,
        {
            "enabled": True,
            "target_motp_m": 3.0,
            "speed_contract": {"defensive_speed": 0.5, "mode": "clamp"},
        },
    )

    assert applied == 0.5
    assert record["contract_mode"] == "clamp"
    assert record["contract_honored"] is True
