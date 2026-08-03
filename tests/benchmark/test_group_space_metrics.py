"""Tests for robot_sf.benchmark.group_space_metrics — group-space intrusion metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.group_space_metrics import (
    compute_group_space_metrics,
    group_specs_from_map,
)


def _group_spec(
    group_id: str = "grp-1",
    centroid: tuple[float, float] = (0.0, 0.0),
    radius: float = 2.0,
    polygon: list[list[float]] | None = None,
) -> dict:
    """Build a minimal group spec mapping."""
    spec: dict = {"group_id": group_id, "centroid": list(centroid), "radius": radius}
    if polygon is not None:
        spec["o_space_polygon"] = polygon
    return spec


class TestGroupSpecsFromMap:
    """Tests for group_specs_from_map extraction."""

    def test_no_social_groups_attribute(self) -> None:
        """An object without social_groups returns an empty list."""

        class NoGroups:
            pass

        assert group_specs_from_map(NoGroups()) == []

    def test_none_social_groups(self) -> None:
        """social_groups=None returns an empty list."""

        class NullGroups:
            social_groups = None

        assert group_specs_from_map(NullGroups()) == []

    def test_mapping_groups_extracted(self) -> None:
        """Mapping-style groups are extracted as dicts."""

        class MapDef:
            social_groups = [{"group_id": "g1", "centroid": [1.0, 2.0], "radius": 1.5}]

        specs = group_specs_from_map(MapDef())
        assert len(specs) == 1
        assert specs[0]["group_id"] == "g1"

    def test_as_spec_callable_groups(self) -> None:
        """Groups exposing as_spec() are converted via that method."""

        class GroupObj:
            def as_spec(self) -> dict:
                return {"group_id": "g2", "centroid": [3.0, 4.0], "radius": 2.0}

        class MapDef:
            social_groups = [GroupObj()]

        specs = group_specs_from_map(MapDef())
        assert len(specs) == 1
        assert specs[0]["group_id"] == "g2"


class TestComputeGroupSpaceMetrics:
    """Tests for compute_group_space_metrics behavior."""

    def test_no_groups_returns_empty_metrics(self) -> None:
        """With no groups, metrics must be zeroed with NaN distances."""
        pos = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = compute_group_space_metrics(pos, [])
        assert result["group_space_available"] == 0.0
        assert result["group_count"] == 0.0
        assert result["group_intrusion_episode_rate"] == 0.0
        assert math.isnan(result["min_distance_to_group_centroid"])

    def test_empty_trajectory_returns_empty_metrics(self) -> None:
        """An empty trajectory must return zeroed metrics."""
        result = compute_group_space_metrics(np.zeros((0, 2)), [_group_spec()])
        assert result["group_space_available"] == 0.0
        assert result["group_metric_timestep_count"] == 0.0

    def test_robot_outside_group_no_intrusion(self) -> None:
        """A robot far from the group must have zero intrusion."""
        pos = np.array([[10.0, 10.0], [11.0, 11.0]])
        result = compute_group_space_metrics(pos, [_group_spec(centroid=(0.0, 0.0), radius=2.0)])
        assert result["group_space_available"] == 1.0
        assert result["group_intrusion_episode_rate"] == 0.0
        assert result["group_intrusion_time_ratio"] == 0.0
        assert result["group_intrusion_step_count"] == 0.0

    def test_robot_inside_group_intrusion_detected(self) -> None:
        """A robot inside the group radius must register intrusion."""
        pos = np.array([[0.5, 0.0], [0.0, 0.5]])
        result = compute_group_space_metrics(pos, [_group_spec(centroid=(0.0, 0.0), radius=2.0)])
        assert result["group_intrusion_episode_rate"] == 1.0
        assert result["group_intrusion_time_ratio"] == 1.0
        assert result["group_intrusion_step_count"] == 2.0

    def test_partial_intrusion_time_ratio(self) -> None:
        """Intrusion at only some steps must give a fractional time ratio."""
        pos = np.array([[0.0, 0.0], [10.0, 10.0], [0.5, 0.0], [20.0, 20.0]])
        result = compute_group_space_metrics(pos, [_group_spec(centroid=(0.0, 0.0), radius=2.0)])
        assert result["group_intrusion_step_count"] == 2.0
        assert result["group_intrusion_time_ratio"] == pytest.approx(0.5)

    def test_min_distance_to_centroid(self) -> None:
        """min_distance_to_group_centroid must be the closest approach."""
        pos = np.array([[3.0, 4.0], [1.0, 0.0]])
        result = compute_group_space_metrics(pos, [_group_spec(centroid=(0.0, 0.0), radius=1.0)])
        assert result["min_distance_to_group_centroid"] == pytest.approx(1.0)

    def test_min_distance_to_boundary_positive_outside(self) -> None:
        """Boundary clearance must be positive when outside the group."""
        pos = np.array([[5.0, 0.0]])
        result = compute_group_space_metrics(pos, [_group_spec(centroid=(0.0, 0.0), radius=2.0)])
        assert result["min_distance_to_group_boundary"] == pytest.approx(3.0)

    def test_min_distance_to_boundary_negative_inside(self) -> None:
        """Boundary clearance must be negative when inside the group."""
        pos = np.array([[0.5, 0.0]])
        result = compute_group_space_metrics(pos, [_group_spec(centroid=(0.0, 0.0), radius=2.0)])
        assert result["min_distance_to_group_boundary"] == pytest.approx(-1.5)

    def test_nearest_group_id_tracked(self) -> None:
        """nearest_group_id must identify the closest group."""
        pos = np.array([[0.5, 0.0]])
        groups = [
            _group_spec(group_id="far", centroid=(100.0, 100.0), radius=1.0),
            _group_spec(group_id="near", centroid=(0.0, 0.0), radius=2.0),
        ]
        result = compute_group_space_metrics(pos, groups)
        assert result["nearest_group_id"] == "near"

    def test_multiple_groups_count(self) -> None:
        """group_count must reflect the number of declared groups."""
        pos = np.array([[0.0, 0.0]])
        groups = [_group_spec(group_id="g1"), _group_spec(group_id="g2")]
        result = compute_group_space_metrics(pos, groups)
        assert result["group_count"] == 2.0

    def test_polygon_group_intrusion(self) -> None:
        """A polygon o-space must detect intrusion via signed clearance."""
        square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        pos = np.array([[2.0, 2.0]])
        result = compute_group_space_metrics(
            pos, [_group_spec(centroid=(2.0, 2.0), radius=1.0, polygon=square)]
        )
        assert result["group_intrusion_episode_rate"] == 1.0
        assert result["min_distance_to_group_boundary"] < 0.0

    def test_polygon_group_no_intrusion_outside(self) -> None:
        """A robot outside a polygon o-space must have no intrusion."""
        square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        pos = np.array([[10.0, 10.0]])
        result = compute_group_space_metrics(
            pos, [_group_spec(centroid=(2.0, 2.0), radius=1.0, polygon=square)]
        )
        assert result["group_intrusion_episode_rate"] == 0.0
        assert result["min_distance_to_group_boundary"] > 0.0

    def test_nan_positions_filtered(self) -> None:
        """NaN positions must be filtered without crashing."""
        pos = np.array([[float("nan"), float("nan")], [1.0, 0.0]])
        result = compute_group_space_metrics(pos, [_group_spec(centroid=(0.0, 0.0), radius=2.0)])
        assert result["group_space_available"] == 1.0
        assert result["group_metric_timestep_count"] == 2.0

    def test_all_nan_positions_returns_empty(self) -> None:
        """All-NaN positions must return empty metrics."""
        pos = np.array([[float("nan"), float("nan")]])
        result = compute_group_space_metrics(pos, [_group_spec()])
        assert result["group_space_available"] == 0.0

    def test_single_mapping_group_input(self) -> None:
        """A single mapping (not a list) must be accepted as groups input."""
        pos = np.array([[0.5, 0.0]])
        result = compute_group_space_metrics(pos, _group_spec(centroid=(0.0, 0.0), radius=2.0))
        assert result["group_space_available"] == 1.0
        assert result["group_count"] == 1.0
