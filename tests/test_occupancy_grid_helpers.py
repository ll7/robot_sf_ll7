"""Additional coverage tests for occupancy_grid helpers."""

from __future__ import annotations

import numpy as np
import pytest
from loguru import logger

import robot_sf.nav.occupancy_grid as og
from robot_sf.nav import occupancy_grid_rasterization as rasterization
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


def test_rasterize_obstacles_aggregates_out_of_bounds_debug_logs() -> None:
    """Verify out-of-bounds obstacle logs are emitted once per batch, not once per segment."""
    config = GridConfig(
        resolution=1.0,
        width=4.0,
        height=4.0,
        channels=[GridChannel.OBSTACLES],
    )
    grid = np.zeros((config.grid_height, config.grid_width), dtype=config.dtype)
    obstacles = [
        ((-3.0, -3.0), (-2.0, -2.0)),
        ((5.0, 5.0), (6.0, 6.0)),
        ((1.0, 1.0), (2.0, 2.0)),
    ]
    captured: list[str] = []
    handler_id = logger.add(
        lambda message: captured.append(message.record["message"]),
        level="DEBUG",
    )
    try:
        count = rasterization.rasterize_obstacles(obstacles, grid, config)
    finally:
        logger.remove(handler_id)

    assert count == 1
    assert captured.count("Skipped 2/3 obstacle segments outside grid bounds") == 1
    assert not any("Line segment" in message for message in captured)


def test_generate_formats_polygon_fill_loguru_message() -> None:
    """Verify polygon rasterization diagnostics interpolate values with Loguru formatting."""
    config = GridConfig(
        resolution=1.0,
        width=4.0,
        height=4.0,
        channels=[GridChannel.OBSTACLES],
    )
    grid = OccupancyGrid(config)
    captured: list[str] = []
    handler_id = logger.add(
        lambda message: captured.append(message.record["message"]),
        level="DEBUG",
    )
    try:
        result = grid.generate(
            obstacles=[],
            pedestrians=[],
            robot_pose=((0.0, 0.0), 0.0),
            obstacle_polygons=[[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]],
        )
    finally:
        logger.remove(handler_id)

    assert np.any(result > 0)
    assert any(message.startswith("Filled ") for message in captured)
    assert not any("%s" in message for message in captured)


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
