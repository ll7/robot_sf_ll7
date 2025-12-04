"""Occupancy Grid Extension for robot_sf Navigation Module.

This module extends the collision-checking capabilities of `occupancy.py`
with a rasterized occupancy grid representation. It provides:

1. Grid Generation: Convert continuous obstacles and pedestrians to discrete grid
2. Gymnasium Integration: Support Box observations with occupancy channels
3. POI Queries: Query specific points or regions for occupancy information
4. Visualization: Pygame rendering of occupancy grids
5. Multi-channel Support: Obstacle, pedestrian, robot occupancy channels

The implementation maintains backward compatibility with existing `occupancy.py`
collision checking while adding new grid-based functionality.

Architecture:
- OccupancyGrid: Main container managing multiple occupancy layers
- GridChannel: Individual occupancy layer (obstacles, peds, robot, etc.)
- GridConfig: Configuration specification for grid parameters
- POIQuery / POIResult: Query API for occupancy information

Performance Targets:
- Grid generation: <5ms per frame
- POI queries: <1ms for single-point queries
- Rendering: 30+ FPS on typical hardware

Constitution Compliance:
- Principle I (Deterministic): All grid generation seeded from global RNG
- Principle II (Factory): Factory-provided via environment_factory
- Principle IV (Config): Unified config integration
- Principle VII (Backward Compat): Extends occupancy.py without modifying existing API
- Principle XII (Logging): Structured logging via Loguru

Example Usage:
```python
from robot_sf.nav.occupancy_grid import OccupancyGrid, GridConfig
from robot_sf.common.types import RobotPose

# Create a grid
config = GridConfig(
    resolution=0.1,  # 10cm per cell
    width=10.0,  # 10m width
    height=10.0,  # 10m height
    channels=["obstacles", "pedestrians", "robot"],
)
grid = OccupancyGrid(config=config)

# Update with obstacles and pedestrians
obstacles = [...]  # List of line segments
pedestrians = [...]  # List of (x, y, r) circles
robot_pose = RobotPose(x=5.0, y=5.0, theta=0.0)

occupancy_array = grid.generate(
    obstacles=obstacles, pedestrians=pedestrians, robot_pose=robot_pose, ego_frame=False
)

# Query a point of interest
from robot_sf.nav.occupancy_grid import POIQuery, POIQueryType

query = POIQuery(x=5.5, y=5.5, query_type=POIQueryType.POINT)
result = grid.query(query)
print(f"Occupancy at (5.5, 5.5): {result.occupancy}")

# Render visualization
grid.render_pygame(surface=my_pygame_surface, robot_pose=robot_pose)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from robot_sf.common.types import Circle2D, Line2D, RobotPose

from robot_sf.nav import occupancy_grid_rasterization as rasterization


class GridChannel(Enum):
    """Occupancy grid channel identifiers."""

    OBSTACLES = "obstacles"
    PEDESTRIANS = "pedestrians"
    ROBOT = "robot"
    COMBINED = "combined"  # Aggregated occupancy


class POIQueryType(Enum):
    """Types of point-of-interest queries."""

    POINT = "point"  # Single point query
    CIRCLE = "circle"  # Circular region query
    RECT = "rect"  # Rectangular region query
    LINE = "line"  # Line segment query


@dataclass
class GridConfig:
    """Configuration for occupancy grid generation.

    Attributes:
        resolution: Grid cell size in world units (default: 0.1m)
        width: Grid width in world units (default: 20.0m)
        height: Grid height in world units (default: 20.0m)
        channels: List of channels to generate (default: obstacles, pedestrians)
        dtype: NumPy data type for grid values (default: float32)
        max_distance: Max distance for continuous occupancy (default: 0.5m)
        use_ego_frame: Whether to use robot's ego frame (default: False)

    Invariants:
        - resolution > 0
        - width > 0
        - height > 0
        - len(channels) > 0
        - dtype in (float16, float32, float64, uint8)
    """

    resolution: float = 0.1
    width: float = 20.0
    height: float = 20.0
    channels: list[GridChannel] = field(
        default_factory=lambda: [
            GridChannel.OBSTACLES,
            GridChannel.PEDESTRIANS,
        ]
    )
    dtype: type = np.float32
    max_distance: float = 0.5
    use_ego_frame: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {self.resolution}")
        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")
        if not self.channels:
            raise ValueError("channels must not be empty")

        # Validate dtype
        valid_dtypes = (np.float16, np.float32, np.float64, np.uint8)
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}, got {self.dtype}")

        logger.debug(
            "GridConfig initialized: "
            f"resolution={self.resolution}, "
            f"size=({self.width}x{self.height}), "
            f"channels={[c.value for c in self.channels]}"
        )

    @property
    def grid_width(self) -> int:
        """Number of cells in width direction (columns)."""
        return int(np.ceil(self.width / self.resolution))

    @property
    def grid_height(self) -> int:
        """Number of cells in height direction (rows)."""
        return int(np.ceil(self.height / self.resolution))

    @property
    def num_channels(self) -> int:
        """Number of occupancy channels."""
        return len(self.channels)


@dataclass
class POIQuery:
    """Point-of-Interest query specification.

    Attributes:
        x: X coordinate (world units or ego frame)
        y: Y coordinate (world units or ego frame)
        query_type: Type of query (POINT, CIRCLE, RECT, LINE)
        radius: Radius for CIRCLE query (default: 0.1m)
        width: Width for RECT query (default: 0.1m)
        height: Height for RECT query (default: 0.1m)
        x2: End X for LINE query
        y2: End Y for LINE query
        channels: Specific channels to query (default: all)

    Invariants:
        - x, y are finite numbers
        - radius, width, height > 0 for their respective query types
    """

    x: float
    y: float
    query_type: POIQueryType = POIQueryType.POINT
    radius: float = 0.1
    width: float = 0.1
    height: float = 0.1
    x2: float | None = None
    y2: float | None = None
    channels: list[GridChannel] | None = None

    def __post_init__(self):
        """Validate query parameters."""
        if not (np.isfinite(self.x) and np.isfinite(self.y)):
            raise ValueError(f"Invalid coordinates: ({self.x}, {self.y})")

        if self.query_type == POIQueryType.CIRCLE and self.radius <= 0:
            raise ValueError(f"radius must be > 0 for CIRCLE query, got {self.radius}")

        if self.query_type == POIQueryType.RECT and (self.width <= 0 or self.height <= 0):
            raise ValueError(
                f"width and height must be > 0 for RECT query, got ({self.width}, {self.height})"
            )

        if self.query_type == POIQueryType.LINE:
            if self.x2 is None or self.y2 is None:
                raise ValueError("x2 and y2 required for LINE query")
            if not (np.isfinite(self.x2) and np.isfinite(self.y2)):
                raise ValueError(f"Invalid end coordinates: ({self.x2}, {self.y2})")


@dataclass
class POIResult:
    """Result of a point-of-interest query.

    Attributes:
        occupancy: Occupancy value for the query (per channel or aggregated)
        query_type: Type of query that produced this result
        num_cells: Number of cells affected by the query
        min_occupancy: Minimum occupancy value in query region
        max_occupancy: Maximum occupancy value in query region
        mean_occupancy: Mean occupancy value in query region
        channel_results: Per-channel occupancy values if channels were specified

    Invariants:
        - 0 <= occupancy <= 1 (for typical binary/continuous grids)
        - 0 <= min_occupancy <= mean_occupancy <= max_occupancy <= 1
        - num_cells >= 0
    """

    occupancy: float
    query_type: POIQueryType
    num_cells: int = 0
    min_occupancy: float = 0.0
    max_occupancy: float = 0.0
    mean_occupancy: float = 0.0
    channel_results: dict[GridChannel, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result parameters."""
        if not np.isfinite(self.occupancy):
            raise ValueError(f"occupancy must be finite, got {self.occupancy}")
        if self.num_cells < 0:
            raise ValueError(f"num_cells must be >= 0, got {self.num_cells}")


class OccupancyGrid:
    """Main occupancy grid container and API.

    Manages generation, querying, and visualization of rasterized occupancy
    representation of the environment.

    Attributes:
        config: GridConfig specifying grid parameters
        _grid_data: NumPy array of shape [C, H, W] containing occupancy values
        _last_robot_pose: Last robot pose used for ego-frame transforms

    Invariants:
        - _grid_data has shape [C, H, W] matching config
        - All values in _grid_data are valid dtype and in [0, 1] range
    """

    def __init__(self, config: GridConfig):
        """Initialize occupancy grid.

        Args:
            config: Grid configuration parameters

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._grid_data: np.ndarray | None = None
        self._last_robot_pose: RobotPose | None = None

        logger.debug(
            f"OccupancyGrid initialized: "
            f"{self.config.grid_width}x{self.config.grid_height}x"
            f"{self.config.num_channels} cells"
        )

    def generate(  # noqa: C901
        self,
        obstacles: list[Line2D],
        pedestrians: list[Circle2D],
        robot_pose: RobotPose,
        ego_frame: bool = False,
    ) -> np.ndarray:
        """Generate occupancy grid from obstacles and pedestrians.

        Args:
            obstacles: List of line segment obstacles
            pedestrians: List of circular pedestrian objects
            robot_pose: Robot's current pose
            ego_frame: If True, generate grid in robot's ego frame;
                      if False, use world frame

        Returns:
            Occupancy array of shape [C, H, W] with values in [0, 1]

        Raises:
            ValueError: If inputs are invalid or incompatible

        Performance:
            - O(N*M) where N = grid cells, M = obstacles+pedestrians
            - Target: <5ms for typical 200x200 grid with 10-20 obstacles
        """
        if not isinstance(obstacles, list):
            raise TypeError(f"obstacles must be list, got {type(obstacles)}")
        if not isinstance(pedestrians, list):
            raise TypeError(f"pedestrians must be list, got {type(pedestrians)}")

        self._last_robot_pose = robot_pose

        # Determine grid bounds
        if ego_frame:
            # TODO (Phase 2): Implement ego-frame coordinate transformation
            grid_origin_x = robot_pose.x - self.config.width / 2
            grid_origin_y = robot_pose.y - self.config.height / 2
        else:
            grid_origin_x = 0.0
            grid_origin_y = 0.0

        # Initialize grid
        shape = (
            self.config.num_channels,
            self.config.grid_height,
            self.config.grid_width,
        )
        self._grid_data = np.zeros(shape, dtype=self.config.dtype)

        logger.debug(
            f"Generating grid: obstacles={len(obstacles)}, "
            f"pedestrians={len(pedestrians)}, "
            f"ego_frame={ego_frame}"
        )

        # Rasterize each channel
        for channel_idx, channel in enumerate(self.config.channels):
            if channel == GridChannel.OBSTACLES:
                # Rasterize obstacles into this channel
                num_rasterized = rasterization.rasterize_obstacles(
                    obstacles,
                    self._grid_data[channel_idx],
                    self.config,
                    grid_origin_x,
                    grid_origin_y,
                    value=1.0,
                )
                logger.debug(f"Rasterized {num_rasterized} obstacles")

            elif channel == GridChannel.PEDESTRIANS:
                # Rasterize pedestrians into this channel
                num_rasterized = rasterization.rasterize_pedestrians(
                    pedestrians,
                    self._grid_data[channel_idx],
                    self.config,
                    grid_origin_x,
                    grid_origin_y,
                    value=1.0,
                )
                logger.debug(f"Rasterized {num_rasterized} pedestrians")

            elif channel == GridChannel.ROBOT:
                # Rasterize robot as a circle (assume 0.3m radius)
                robot_radius = 0.3  # TODO: Make configurable
                success = rasterization.rasterize_robot(
                    robot_pose,
                    robot_radius,
                    self._grid_data[channel_idx],
                    self.config,
                    grid_origin_x,
                    grid_origin_y,
                    value=1.0,
                )
                logger.debug(f"Rasterized robot: {success}")

            elif channel == GridChannel.COMBINED:
                # Combine all other channels (max occupancy)
                combined = np.zeros_like(self._grid_data[channel_idx])
                for other_idx, other_channel in enumerate(self.config.channels):
                    if other_channel != GridChannel.COMBINED:
                        combined = np.maximum(combined, self._grid_data[other_idx])
                self._grid_data[channel_idx] = combined
                logger.debug("Generated combined channel")

        return self._grid_data

    def query(self, query: POIQuery) -> POIResult:
        """Query occupancy at point(s) of interest.

        Args:
            query: POI query specification

        Returns:
            POIResult with occupancy information for the query

        Raises:
            ValueError: If query is invalid or grid not generated
            RuntimeError: If grid has not been generated

        Performance:
            - Single point: O(1)
            - Circular region: O(π * r²)
            - Rectangular region: O(w * h)
            - Line segment: O(length / resolution)
        """
        if self._grid_data is None:
            raise RuntimeError("Grid has not been generated yet. Call generate() first.")

        # TODO (Phase 3): Implement query logic
        # - Validate query coordinates within grid bounds
        # - Convert world/ego coordinates to grid indices
        # - Extract occupancy values based on query type
        # - Compute statistics (min, max, mean)
        # - Return POIResult

        return POIResult(
            occupancy=0.0,
            query_type=query.query_type,
            num_cells=0,
        )

    def render_pygame(
        self,
        surface,  # pygame.Surface
        robot_pose: RobotPose,
        scale: float = 1.0,
        alpha: int = 128,
    ) -> None:
        """Render occupancy grid using Pygame.

        Args:
            surface: Pygame surface to render to
            robot_pose: Robot's pose for visualization reference
            scale: Scale factor for rendering (pixels per cell)
            alpha: Alpha transparency [0-255]

        Raises:
            ValueError: If surface is invalid or grid not generated
            RuntimeError: If grid has not been generated

        Performance:
            - O(C * H * W) where C = channels, H = height, W = width
            - Target: 30+ FPS on typical 200x200 grid
        """
        if self._grid_data is None:
            raise RuntimeError("Grid has not been generated yet. Call generate() first.")

        # TODO (Phase 4): Implement Pygame rendering
        # - Convert grid values to color values
        # - Create visualization surface
        # - Render channels with different colors
        # - Draw grid overlay if requested
        # - Draw robot pose indicator
        # - Blit to target surface

    def reset(self) -> None:
        """Reset the occupancy grid.

        Clears grid data and stored poses. Next call to generate() will create
        a new grid.
        """
        self._grid_data = None
        self._last_robot_pose = None
        logger.debug("Occupancy grid reset")

    @property
    def is_initialized(self) -> bool:
        """Check if grid has been generated."""
        return self._grid_data is not None

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get grid shape [C, H, W]."""
        if self._grid_data is None:
            return (self.config.num_channels, self.config.grid_height, self.config.grid_width)
        return self._grid_data.shape

    def get_channel(self, channel: GridChannel) -> np.ndarray:
        """Get a specific channel data.

        Args:
            channel: Channel to retrieve

        Returns:
            2D numpy array of shape [H, W] for the channel

        Raises:
            RuntimeError: If grid not generated
            ValueError: If channel not in grid
        """
        if self._grid_data is None:
            raise RuntimeError("Grid has not been generated yet.")

        if channel not in self.config.channels:
            raise ValueError(f"Channel {channel.value} not in grid")

        channel_idx = self.config.channels.index(channel)
        return self._grid_data[channel_idx]

    def __repr__(self) -> str:
        """String representation of occupancy grid.

        Returns:
            str: String representation including config, shape, and initialization status.
        """
        status = "initialized" if self.is_initialized else "not initialized"
        return f"OccupancyGrid(config={self.config}, shape={self.shape}, status={status})"
