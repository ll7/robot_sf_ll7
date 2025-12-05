"""Test fixtures for occupancy grid tests.

Provides reusable pytest fixtures for:
1. Grid configurations (various sizes and channels)
2. Synthetic test obstacles (lines, rectangles, circles)
3. Synthetic test pedestrians
4. Robot poses (centered, corners, rotated)
5. Pre-generated grids with known layouts
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.common.types import Circle2D, Line2D
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig, OccupancyGrid


@pytest.fixture
def simple_grid_config():
    """Basic 10x10m grid with 0.1m resolution (100x100 cells)."""
    return GridConfig(
        resolution=0.1,
        width=10.0,
        height=10.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )


@pytest.fixture
def large_grid_config():
    """Larger 20x20m grid with 0.1m resolution (200x200 cells)."""
    return GridConfig(
        resolution=0.1,
        width=20.0,
        height=20.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.ROBOT],
    )


@pytest.fixture
def coarse_grid_config():
    """Coarse 10x10m grid with 0.5m resolution (20x20 cells)."""
    return GridConfig(
        resolution=0.5,
        width=10.0,
        height=10.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )


@pytest.fixture
def single_channel_config():
    """Grid with only obstacles channel."""
    return GridConfig(
        resolution=0.1,
        width=10.0,
        height=10.0,
        channels=[GridChannel.OBSTACLES],
    )


@pytest.fixture
def occupancy_grid(simple_grid_config):
    """Instantiated OccupancyGrid with simple config."""
    return OccupancyGrid(config=simple_grid_config)


@pytest.fixture
def robot_pose_center():
    """Robot at center of a 10x10m grid (world frame origin)."""
    return ((5.0, 5.0), 0.0)


@pytest.fixture
def robot_pose_corner():
    """Robot at corner of grid."""
    return ((1.0, 1.0), 0.0)


@pytest.fixture
def robot_pose_rotated():
    """Robot at center with 45° rotation."""
    import numpy as np

    return ((5.0, 5.0), float(np.pi / 4))


@pytest.fixture
def simple_obstacles():
    """Simple obstacle layout: two horizontal walls."""
    return [
        Line2D((1.0, 3.0), (9.0, 3.0)),  # Horizontal wall at Y=3
        Line2D((1.0, 7.0), (9.0, 7.0)),  # Horizontal wall at Y=7
    ]


@pytest.fixture
def complex_obstacles():
    """More complex obstacle layout: rectangular room with interior walls."""
    return [
        # Outer walls
        Line2D((0.5, 0.5), (9.5, 0.5)),  # Bottom
        Line2D((0.5, 9.5), (9.5, 9.5)),  # Top
        Line2D((0.5, 0.5), (0.5, 9.5)),  # Left
        Line2D((9.5, 0.5), (9.5, 9.5)),  # Right
        # Interior walls
        Line2D((3.0, 2.0), (3.0, 8.0)),  # Vertical divider
        Line2D((7.0, 2.0), (7.0, 8.0)),  # Vertical divider
    ]


@pytest.fixture
def simple_pedestrians():
    """Simple pedestrian layout: two pedestrians."""
    return [
        Circle2D((3.0, 5.0), 0.3),  # Pedestrian at (3, 5)
        Circle2D((7.0, 5.0), 0.3),  # Pedestrian at (7, 5)
    ]


@pytest.fixture
def crowded_pedestrians():
    """Crowded layout: 5 pedestrians in middle of grid."""
    return [
        Circle2D((4.5, 4.5), 0.3),
        Circle2D((5.5, 4.5), 0.3),
        Circle2D((5.0, 5.5), 0.3),
        Circle2D((4.5, 5.5), 0.3),
        Circle2D((5.5, 5.5), 0.3),
    ]


@pytest.fixture
def empty_pedestrians():
    """Empty pedestrian list."""
    return []


@pytest.fixture
def pre_generated_grid(occupancy_grid, simple_obstacles, simple_pedestrians, robot_pose_center):
    """Pre-generated grid with simple layout."""
    grid = occupancy_grid
    grid.generate(
        obstacles=simple_obstacles,
        pedestrians=simple_pedestrians,
        robot_pose=robot_pose_center,
        ego_frame=False,
    )
    return grid


# Performance benchmarking fixtures


@pytest.fixture
def perf_benchmark_grid_config():
    """Grid configuration for performance benchmarks."""
    return GridConfig(
        resolution=0.1,
        width=20.0,
        height=20.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )


@pytest.fixture
def perf_benchmark_obstacles():
    """Realistic obstacle layout for performance benchmarks."""
    obstacles = []

    # Perimeter walls
    obstacles.append(Line2D((0.5, 0.5), (19.5, 0.5)))  # Bottom
    obstacles.append(Line2D((0.5, 19.5), (19.5, 19.5)))  # Top
    obstacles.append(Line2D((0.5, 0.5), (0.5, 19.5)))  # Left
    obstacles.append(Line2D((19.5, 0.5), (19.5, 19.5)))  # Right

    # Interior walls (grid pattern)
    for i in range(2, 20, 5):
        obstacles.append(Line2D((i, 2.0), (i, 18.0)))  # Vertical
        obstacles.append(Line2D((2.0, i), (18.0, i)))  # Horizontal

    return obstacles


@pytest.fixture
def perf_benchmark_pedestrians():
    """Realistic pedestrian density for performance benchmarks."""
    pedestrians = []

    # Random-like grid of pedestrians
    for x in np.arange(2.0, 18.0, 2.5):
        for y in np.arange(2.0, 18.0, 2.5):
            pedestrians.append(Circle2D((x, y), 0.3))

    return pedestrians
