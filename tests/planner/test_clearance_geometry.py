"""Hand-built boundary oracles for versioned planner clearance geometry."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.clearance_geometry import (
    CENTER_CLEARANCE_V1,
    SURFACE_CLEARANCE_V2,
    occupied_cell_clearance,
    pedestrian_clearance,
    time_to_circle_contact,
)


def test_center_v1_preserves_historical_center_distance() -> None:
    """Legacy clearance mode leaves center distances unchanged."""
    assert pedestrian_clearance(
        2.5,
        model=CENTER_CLEARANCE_V1,
        robot_radius=1.0,
        pedestrian_radius=0.4,
    ) == pytest.approx(2.5)


def test_surface_v2_reports_touching_and_overlapping_circles() -> None:
    """Two circles touch at the sum of radii and overlap below that boundary."""
    center_distances = np.asarray([1.4, 1.2])
    clearances = pedestrian_clearance(
        center_distances,
        model=SURFACE_CLEARANCE_V2,
        robot_radius=1.0,
        pedestrian_radius=0.4,
    )
    assert isinstance(clearances, np.ndarray)
    assert clearances.tolist() == pytest.approx([0.0, -0.2])


def test_surface_v2_measures_to_occupied_cell_square_then_robot_body() -> None:
    """The clearance uses the occupied square boundary, then subtracts the robot radius."""
    # At a three-cell horizontal offset with 0.2 m cells, the occupied square
    # begins 0.5 m away; subtracting the 0.4 m robot radius leaves 0.1 m.
    clearance = occupied_cell_clearance(
        np.asarray([0.0]),
        np.asarray([3.0]),
        resolution=0.2,
        model=SURFACE_CLEARANCE_V2,
        robot_radius=0.4,
    )
    assert clearance == pytest.approx(0.1)


def test_center_v1_occupancy_distance_and_empty_grid_are_unchanged() -> None:
    """The legacy grid metric remains center-to-center and empty cells remain infinite."""
    assert occupied_cell_clearance(
        np.asarray([3.0]),
        np.asarray([4.0]),
        resolution=0.2,
        model=CENTER_CLEARANCE_V1,
        robot_radius=1.0,
    ) == pytest.approx(1.0)
    assert occupied_cell_clearance(
        np.asarray([]),
        np.asarray([]),
        resolution=0.2,
        model=SURFACE_CLEARANCE_V2,
        robot_radius=1.0,
    ) == float("inf")


def test_surface_ttc_is_first_contact_not_center_closest_approach() -> None:
    """Surface TTC reaches the combined-radius boundary before center overlap."""
    assert time_to_circle_contact(
        np.asarray([3.0, 0.0]),
        np.asarray([-1.0, 0.0]),
        combined_radius=1.4,
    ) == pytest.approx(1.6)
    assert time_to_circle_contact(
        np.asarray([3.0, 0.0]),
        np.asarray([1.0, 0.0]),
        combined_radius=1.4,
    ) == float("inf")
