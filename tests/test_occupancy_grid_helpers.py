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


def _classic_bresenham_cells(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Return the historical cell-order reference for all-octant line walks."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            return cells
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


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


def test_bresenham_line_matches_legacy_cell_order_exhaustively() -> None:
    """Characterize canonical rasterizer order across signed all-octant endpoints."""
    coordinates = range(-2, 3)
    for x0 in coordinates:
        for y0 in coordinates:
            for x1 in coordinates:
                for y1 in coordinates:
                    rows, cols = rasterization._bresenham_line(
                        row0=y0,
                        col0=x0,
                        row1=y1,
                        col1=x1,
                    )
                    expected = list(zip(cols.tolist(), rows.tolist(), strict=True))
                    assert OccupancyGrid._bresenham_line(x0, y0, x1, y1) == expected, (
                        f"mismatch for ({x0}, {y0}) -> ({x1}, {y1})"
                    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((0.0, 0.0), (5.0, 3.0)),
        ((5.0, 3.0), (0.0, 0.0)),
        ((1.0, 0.0), (3.0, 3.0)),
        ((0.0, 1.0), (5.0, 3.0)),
        ((2.0, 1.0), (2.0, 1.0)),
        ((-2.0, -1.0), (7.0, 5.0)),
    ],
)
def test_line_query_preserves_statistics_on_non_square_grid(
    start: tuple[float, float], end: tuple[float, float]
) -> None:
    """Preserve LINE cell filtering and statistics for boundary and octant variants."""
    config = GridConfig(
        resolution=1.0,
        width=6.0,
        height=4.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )
    grid = OccupancyGrid(config)
    grid.generate(obstacles=[], pedestrians=[], robot_pose=((0.0, 0.0), 0.0))

    obstacle_values = (
        np.arange(config.grid_height * config.grid_width, dtype=float).reshape(
            config.grid_height, config.grid_width
        )
        / 100.0
    )
    pedestrian_values = np.flip(obstacle_values, axis=(0, 1))
    grid._grid_data[0] = obstacle_values
    grid._grid_data[1] = pedestrian_values

    start_col = int(start[0] / config.resolution)
    start_row = int(start[1] / config.resolution)
    start_col = int(np.clip(start_col, 0, config.grid_width - 1))
    start_row = int(np.clip(start_row, 0, config.grid_height - 1))
    end_col = int(end[0] / config.resolution)
    end_row = int(end[1] / config.resolution)
    expected_cells = [
        (row, col)
        for col, row in _classic_bresenham_cells(start_col, start_row, end_col, end_row)
        if 0 <= col < config.grid_width and 0 <= row < config.grid_height
    ]
    expected_values = grid._grid_data[
        :, [row for row, _col in expected_cells], [col for _row, col in expected_cells]
    ].astype(float)
    expected_per_cell_max = expected_values.max(axis=0)

    result = grid.query(
        POIQuery(
            x=start[0],
            y=start[1],
            x2=end[0],
            y2=end[1],
            query_type=POIQueryType.LINE,
        )
    )

    assert result.num_cells == len(expected_cells)
    assert result.occupancy == pytest.approx(expected_per_cell_max.mean())
    assert result.min_occupancy == pytest.approx(expected_per_cell_max.min())
    assert result.max_occupancy == pytest.approx(expected_per_cell_max.max())
    assert result.mean_occupancy == pytest.approx(expected_per_cell_max.mean())
    assert result.channel_results[GridChannel.OBSTACLES] == pytest.approx(expected_values[0].mean())
    assert result.channel_results[GridChannel.PEDESTRIANS] == pytest.approx(
        expected_values[1].mean()
    )


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
