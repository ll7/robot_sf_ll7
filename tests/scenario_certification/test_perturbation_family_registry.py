"""Tests for perturbation family registry contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from robot_sf.scenario_certification import (
    perturbation_families,
    perturbation_family,
    supported_perturbation_families,
    validate_perturbation_family_parameters,
)
from robot_sf.scenario_certification.perturbation_family_registry import (
    PerturbationFamily,
)


def test_supported_families_covers_manifest_enum() -> None:
    """Every family the manifest schema accepts must be registered."""
    expected = {
        "noop",
        "robot_route_offset",
        "pedestrian_route_offset",
        "single_pedestrian_start_delay_offset",
        "single_pedestrian_speed_offset",
        "single_pedestrian_wait_duration_offset",
        "single_pedestrian_trajectory_waypoint_offset",
        "pedestrian_density_offset",
    }
    assert supported_perturbation_families() == expected


def test_perturbation_families_iterable_is_ordered() -> None:
    """The families tuple should return the same order each call."""
    families = perturbation_families()
    assert len(families) == 8
    names = [f.name for f in families]
    assert names[0] == "noop"
    assert names[-1] == "pedestrian_density_offset"


def test_perturbation_family_lookup() -> None:
    """Lookup by name should return an immutable PerturbationFamily."""
    family = perturbation_family("robot_route_offset")
    assert isinstance(family, PerturbationFamily)
    assert family.name == "robot_route_offset"
    assert family.target_surface == "robot_route_waypoints"
    assert "max_route_offset_m" in family.validity_constraints
    assert len(family.required_parameters) == 3
    assert "dx_m" in family.required_parameters
    assert "dy_m" in family.required_parameters
    assert "max_magnitude_m" in family.required_parameters


def test_perturbation_family_noop_has_no_parameters() -> None:
    """The noop family should accept no perturbation parameters."""
    family = perturbation_family("noop")
    assert family.required_parameters == ()
    assert family.optional_parameters == ()


def test_perturbation_family_unknown_raises() -> None:
    """Unknown families should fail for any caller with a clear message."""
    with pytest.raises(ValueError, match="unsupported perturbation family"):
        perturbation_family("not_a_family")


def test_perturbation_family_is_immutable() -> None:
    """PerturbationFamily must be frozen for registry safety."""
    family = perturbation_family("robot_route_offset")
    with pytest.raises(FrozenInstanceError):
        family.name = "altered"  # type: ignore[misc]


def test_validate_parameters_rejects_missing_required() -> None:
    """Validation must reject parameters missing required keys."""
    reasons, _family = validate_perturbation_family_parameters(
        "robot_route_offset",
        {"max_magnitude_m": 0.5},
    )
    assert len(reasons) == 1
    assert "missing required parameters" in reasons[0]


def test_validate_parameters_rejects_extra_keys() -> None:
    """Validation must reject parameters with keys not in required+optional."""
    reasons, _family = validate_perturbation_family_parameters(
        "robot_route_offset",
        {
            "dx_m": 0.25,
            "dy_m": 0.0,
            "max_magnitude_m": 0.5,
            "dt_s": 0.5,
        },
    )
    assert len(reasons) == 1
    assert "unrecognized parameters" in reasons[0]


def test_validate_parameters_accepts_valid() -> None:
    """Validation should accept parameters with all required keys."""
    reasons, family = validate_perturbation_family_parameters(
        "robot_route_offset",
        {"dx_m": 0.25, "dy_m": 0.0, "max_magnitude_m": 0.5},
    )
    assert reasons == []
    assert family.name == "robot_route_offset"


def test_validate_parameters_accepts_required_plus_optional() -> None:
    """Validation should accept parameters that include optional keys."""
    reasons, family = validate_perturbation_family_parameters(
        "single_pedestrian_speed_offset",
        {
            "speed_delta_m_s": 0.25,
            "max_abs_speed_delta_m_s": 0.5,
            "pedestrian_id": "h3",
        },
    )
    assert reasons == []
    assert family.name == "single_pedestrian_speed_offset"


def test_validate_parameters_for_noop_accepts_no_parameters() -> None:
    """noop validation should accept an empty parameters mapping."""
    reasons, family = validate_perturbation_family_parameters(
        "noop",
        {},
    )
    assert reasons == []
    assert family.name == "noop"


def test_all_families_have_descriptions_and_boundaries() -> None:
    """Every registered family must carry concrete metadata for docs and preflight."""
    for family in perturbation_families():
        assert family.description, f"{family.name} missing description"
        assert family.target_surface, f"{family.name} missing target_surface"
        assert family.semantic_boundary, f"{family.name} missing semantic_boundary"
        assert family.validity_constraints, f"{family.name} missing validity_constraints"
        assert family.fail_closed_rules, f"{family.name} missing fail_closed_rules"


def test_trajectory_family_requires_pedestrian_id() -> None:
    """Trajectory waypoint offset must require pedestrian_id."""
    family = perturbation_family("single_pedestrian_trajectory_waypoint_offset")
    assert "pedestrian_id" in family.required_parameters
    assert "waypoint_selector" in family.required_parameters


def test_density_family_optional_max_ped_density() -> None:
    """Density family must accept an optional max_ped_density cap."""
    family = perturbation_family("pedestrian_density_offset")
    assert "max_ped_density" in family.optional_parameters
    assert "density_delta" in family.required_parameters
    assert "max_abs_density_delta" in family.required_parameters
