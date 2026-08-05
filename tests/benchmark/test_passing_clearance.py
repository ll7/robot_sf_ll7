"""Tests for the opt-in PassingClearanceContract.v1 interface."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics
from robot_sf.benchmark.passing_clearance import (
    PASSING_CLEARANCE_SCHEMA_VERSION,
    PassingClearanceContract,
    PassingClearanceContractError,
    neggers_source_transfer_prior,
)
from robot_sf.benchmark.social_compliance import build_social_compliance_episode_block
from robot_sf.benchmark.thresholds import build_metric_parameters
from robot_sf.nav.proxemic_costmap import ProxemicCostmapConfig, proxemic_cost_at_points


def _contract(**overrides: object) -> PassingClearanceContract:
    """Build an explicit test contract."""
    values: dict[str, object] = {
        "profile_id": "test-passing-v1",
        "robot_radius_m": 1.0,
        "pedestrian_radius_m": 0.4,
        "encounter_type": "passing",
        "desired_clearance_m": 0.56,
        "minimum_clearance_m": 0.36,
        "source_citation": "test-source",
        "source_platform_geometry": "test platform",
        "evidence_class": "author_defined",
        "limitation": "Test contract only; not a physical-robot requirement.",
    }
    values.update(overrides)
    return PassingClearanceContract(**values)


def _episode() -> EpisodeData:
    """Build one small episode for contract propagation tests."""
    peds = np.asarray([[[1.8, 0.0]], [[1.8, 0.0]]], dtype=float)
    return EpisodeData(
        robot_pos=np.zeros((2, 2), dtype=float),
        robot_vel=np.zeros((2, 2), dtype=float),
        robot_acc=np.zeros((2, 2), dtype=float),
        peds_pos=peds,
        ped_forces=np.ones_like(peds),
        goal=np.asarray([2.0, 0.0]),
        dt=0.5,
        robot_radius=0.1,
        ped_radius=0.1,
    )


def test_conversion_is_explicit_and_source_prior_is_not_a_default() -> None:
    """Surface clearance conversion uses the declared proxy radii only."""
    contract = _contract()
    assert contract.center_distance_from_surface_clearance(0.36) == pytest.approx(1.76)
    assert contract.center_distance_from_surface_clearance(0.56) == pytest.approx(1.96)
    assert contract.surface_clearance_from_center_distance(1.8) == pytest.approx(0.4)
    assert contract.to_dict()["schema_version"] == PASSING_CLEARANCE_SCHEMA_VERSION

    prior = neggers_source_transfer_prior()
    assert prior.evidence_class == "derived_transfer_prior"
    assert prior.center_distance_from_surface_clearance(0.36) == pytest.approx(0.8)
    assert prior.center_distance_from_surface_clearance(0.56) == pytest.approx(1.0)


def test_round_trip_serialization_and_hash_are_stable() -> None:
    """Serialized profile metadata round-trips with a stable profile hash."""
    contract = _contract(speed_range_mps=(0.2, 1.0))
    payload = contract.to_dict()
    restored = PassingClearanceContract.from_mapping(copy.deepcopy(payload))

    assert restored == contract
    assert restored.profile_hash == contract.profile_hash
    assert payload["profile_hash"] == contract.profile_hash


@pytest.mark.parametrize(
    "kwargs",
    [
        {"robot_radius_m": -0.1},
        {"pedestrian_radius_m": float("nan")},
        {"distance_basis": "center_distance_m"},
        {"evidence_class": "invented"},
        {"minimum_clearance_m": 0.7},
    ],
)
def test_invalid_contracts_fail_closed(kwargs: dict[str, object]) -> None:
    """Invalid radii, units, evidence, and threshold ordering are rejected."""
    with pytest.raises(PassingClearanceContractError):
        _contract(**kwargs)


def test_missing_schema_and_mixed_units_fail_closed() -> None:
    """A mapping without the explicit contract identity cannot be consumed."""
    payload = _contract().to_dict()
    payload.pop("schema_version")
    with pytest.raises(PassingClearanceContractError, match="schema_version"):
        PassingClearanceContract.from_mapping(payload)
    payload = _contract().to_dict()
    payload["units"]["distance"] = "cm"
    with pytest.raises(PassingClearanceContractError, match="units"):
        PassingClearanceContract.from_mapping(payload)


def test_threshold_metadata_carries_contract_and_hash_only_when_selected() -> None:
    """Legacy threshold metadata is unchanged until a caller selects a contract."""
    legacy = build_metric_parameters()
    selected = build_metric_parameters(passing_clearance_contract=_contract())

    assert "passing_clearance_contract" not in legacy["threshold_profile"]
    profile = selected["threshold_profile"]
    assert profile["passing_clearance_contract"]["profile_hash"] == _contract().profile_hash
    assert profile["passing_clearance_contract_hash"] == _contract().profile_hash


def test_social_compliance_uses_contract_surface_radius_and_proxy_radii() -> None:
    """The selected contract controls both the surface band and proxy radii."""
    block = build_social_compliance_episode_block(
        _episode(), passing_clearance_contract=_contract()
    )
    comfort = block["metrics"]["comfort_exposure_person_s"]

    assert comfort["status"] == "available"
    assert comfort["value"] == pytest.approx(1.0)
    assert block["parameters"]["passing_clearance_contract_hash"] == _contract().profile_hash


def test_compute_all_metrics_propagates_explicit_contract() -> None:
    """The public metric entry point can opt into the same contract."""
    values = compute_all_metrics(
        _episode(),
        horizon=2,
        passing_clearance_contract=_contract(),
    )
    assert values["social_compliance"]["parameters"]["passing_clearance_contract"][
        "profile_id"
    ] == ("test-passing-v1")


def test_proxemic_costmap_resolves_surface_clearance_to_center_distance() -> None:
    """An explicit contract changes only the opted-in center-distance field."""
    points = np.asarray([[1.9, 0.0]], dtype=float)
    pedestrian = np.asarray([[0.0, 0.0]], dtype=float)
    velocity = np.zeros((1, 2), dtype=float)
    legacy = ProxemicCostmapConfig(enabled=True, social_weight=1.0, personal_weight=0.0)
    selected = ProxemicCostmapConfig(
        enabled=True,
        social_weight=1.0,
        personal_weight=0.0,
        passing_clearance_contract=_contract(),
    )

    legacy_cost = proxemic_cost_at_points(points, pedestrian, velocity, legacy)
    selected_cost = proxemic_cost_at_points(points, pedestrian, velocity, selected)

    assert legacy_cost[0] == pytest.approx(0.0)
    assert selected_cost[0] > 0.0
