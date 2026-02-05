"""Additional coverage tests for occupancy_grid helpers."""

from __future__ import annotations

import numpy as np
import pytest

import robot_sf.nav.occupancy_grid as og
from robot_sf.nav.occupancy_grid import (
    GridChannel,
    GridConfig,
    OccupancyGrid,
    POIQuery,
    POIQueryType,
    POIResult,
    RobotPoseRecord,
)


def test_poi_query_line_requires_endpoints() -> None:
    """Ensure LINE queries validate that end coordinates are provided."""
    with pytest.raises(ValueError):
        POIQuery(x=0.0, y=0.0, query_type=POIQueryType.LINE)


def test_poi_result_properties_reflect_thresholds() -> None:
    """Validate occupancy thresholds and channel breakdown helpers."""
    result = POIResult(occupancy=0.04, query_type=POIQueryType.POINT)
    assert result.safe_to_spawn is True
    assert result.is_occupied is False
    assert result.occupancy_fraction == pytest.approx(0.04)
    assert result.per_channel_results == {}


def test_robot_pose_record_equality_and_hash() -> None:
    """Check tolerant equality and hashing for robot pose records."""
    record = RobotPoseRecord((1.0, 2.0), 0.5)
    assert record == ((1.0, 2.0), 0.5)
    assert record == ((1.0 + 1e-10, 2.0), 0.5)
    assert record != ((1.0, 2.0, 3.0), 0.5)
    assert isinstance(hash(record), int)


def test_bresenham_line_includes_endpoints() -> None:
    """Verify Bresenham helper returns a connected line path."""
    cells = OccupancyGrid._bresenham_line(0, 0, 2, 1)
    assert cells[0] == (0, 0)
    assert cells[-1] == (2, 1)
    assert len(cells) >= 3


def test_metadata_observation_converts_values() -> None:
    """Ensure metadata is exposed as numpy arrays with expected keys."""
    config = GridConfig(resolution=1.0, width=2.0, height=2.0, channels=[GridChannel.OBSTACLES])
    grid = OccupancyGrid(config)
    grid._grid_origin = (0.0, 0.0)
    grid._last_robot_pose = RobotPoseRecord((1.0, 1.0), 0.0)
    grid._last_use_ego_frame = False
    meta = grid.metadata_observation()
    assert meta["origin"].shape == (2,)
    assert meta["resolution"].shape == (1,)
    assert meta["robot_pose"].shape == (3,)


def test_render_pygame_requires_grid_and_pygame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render should fail gracefully when prerequisites are missing."""
    config = GridConfig(resolution=1.0, width=2.0, height=2.0, channels=[GridChannel.OBSTACLES])
    grid = OccupancyGrid(config)
    with pytest.raises(RuntimeError):
        grid.render_pygame(surface=None, robot_pose=((0.0, 0.0), 0.0))

    grid._grid_data = np.zeros((1, config.grid_height, config.grid_width), dtype=float)
    monkeypatch.setattr(og, "pygame", None)
    with pytest.raises(RuntimeError):
        grid.render_pygame(surface=None, robot_pose=((0.0, 0.0), 0.0))
