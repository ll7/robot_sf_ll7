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

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from shapely.geometry import Point as _ShapelyPoint
from shapely.geometry import Polygon as _ShapelyPolygon
from shapely.prepared import PreparedGeometry, prep

try:
    import pygame
except ImportError:  # pragma: no cover - handled at runtime
    pygame = None

if TYPE_CHECKING:
    from robot_sf.common.types import Circle2D, Line2D, RobotPose

from robot_sf.nav import occupancy_grid_rasterization as rasterization
from robot_sf.nav import occupancy_grid_utils as grid_utils

# Threshold below which a cell/region is treated as free for spawning or visualization.
OCCUPANCY_FREE_THRESHOLD = 0.05


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
        robot_radius: Radius for rasterizing the robot channel (default: 0.3m)
        center_on_robot: When False (default), world-frame grids start at (0,0); when True,
            grids translate to keep the robot near center without rotating axes.

    Invariants:
        - resolution > 0
        - width > 0
        - height > 0
        - len(channels) > 0
        - dtype in (float16, float32, float64, uint8)
        - robot_radius > 0
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
    robot_radius: float = 0.3
    center_on_robot: bool = False

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
        if self.robot_radius <= 0:
            raise ValueError(f"robot_radius must be > 0, got {self.robot_radius}")
        if not isinstance(self.center_on_robot, bool):
            raise ValueError("center_on_robot must be a boolean")

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

    @property
    def is_occupied(self) -> bool:
        """Check if query region is occupied (occupancy > 0.1 threshold)."""
        return self.occupancy > 0.1

    @property
    def safe_to_spawn(self) -> bool:
        """Check if safe to spawn based on the free-space threshold."""
        return self.occupancy < OCCUPANCY_FREE_THRESHOLD

    @property
    def occupancy_fraction(self) -> float:
        """Return occupancy as a fraction [0, 1]."""
        return float(np.clip(self.occupancy, 0.0, 1.0))

    @property
    def per_channel_results(self) -> dict[GridChannel, float]:
        """Return per-channel occupancy breakdown."""
        return self.channel_results.copy()


@dataclass(frozen=True)
class RobotPoseRecord:
    """Lightweight wrapper to store the last robot pose with safe equality semantics."""

    position: tuple[float, float]
    theta: float

    def __iter__(self):
        """Iterate position and heading for tuple unpacking."""
        yield self.position
        yield self.theta

    def __eq__(self, other: object) -> bool:
        """Compare pose using tolerance to absorb floating-point noise.

        Returns:
            bool: True if positions and headings match within tolerance.
        """
        try:
            pos, theta = other  # type: ignore[misc]
            pos_tuple = tuple(float(v) for v in pos)
            if len(pos_tuple) != 2:
                return False
            return math.isclose(float(theta), self.theta) and all(
                math.isclose(a, b) for a, b in zip(pos_tuple, self.position, strict=False)
            )
        except (TypeError, ValueError):
            return False

    def __hash__(self) -> int:
        """Hash pose consistently with approximate equality semantics.

        Returns:
            int: Hash derived from rounded position and heading.
        """
        return hash((round(self.position[0], 9), round(self.position[1], 9), round(self.theta, 9)))


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
        self._last_robot_pose: RobotPoseRecord | None = None
        self._grid_origin: tuple[float, float] | None = None
        self._last_use_ego_frame: bool = False
        self._prepared_obstacles: list[PreparedGeometry] | None = None
        self._obstacle_polygons: list[list[tuple[float, float]]] = []

        logger.debug(
            f"OccupancyGrid initialized: "
            f"{self.config.grid_width}x{self.config.grid_height}x"
            f"{self.config.num_channels} cells"
        )

    @staticmethod
    def _parse_robot_pose(robot_pose: RobotPose) -> tuple[float, float, float]:
        """Return (x, y, theta) as floats from a RobotPose tuple-like value."""
        try:
            position, theta = robot_pose
            x, y = position
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "robot_pose must be a tuple like ((x, y), theta) or (array, theta)"
            ) from exc

        return float(x), float(y), float(theta)

    def generate(  # noqa: C901
        self,
        obstacles: list[Line2D],
        pedestrians: list[Circle2D],
        robot_pose: RobotPose,
        ego_frame: bool = False,
        obstacle_polygons: list[list[tuple[float, float]]] | None = None,
    ) -> np.ndarray:
        """Generate occupancy grid from obstacles and pedestrians.

        Args:
            obstacles: List of line segment obstacles
            obstacle_polygons: Optional list of polygon vertices for filled obstacles
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
            logger.error(f"Invalid obstacles type: {type(obstacles).__name__}, expected list")
            raise TypeError(f"obstacles must be list, got {type(obstacles)}")
        if not isinstance(pedestrians, list):
            logger.error(f"Invalid pedestrians type: {type(pedestrians).__name__}, expected list")
            raise TypeError(f"pedestrians must be list, got {type(pedestrians)}")

        use_ego_frame = ego_frame or self.config.use_ego_frame
        self._last_use_ego_frame = use_ego_frame
        robot_x, robot_y, robot_theta = self._parse_robot_pose(robot_pose)
        self._last_robot_pose = RobotPoseRecord((robot_x, robot_y), robot_theta)

        # Determine grid bounds
        if use_ego_frame:
            grid_origin_x = -self.config.width / 2
            grid_origin_y = -self.config.height / 2
            logger.debug(f"Ego-frame grid origin: ({grid_origin_x:.2f}, {grid_origin_y:.2f})")
        elif self.config.center_on_robot:
            grid_origin_x = robot_x - self.config.width / 2
            grid_origin_y = robot_y - self.config.height / 2
            logger.debug(
                "World-frame grid centered on robot: origin=(%.2f, %.2f)",
                grid_origin_x,
                grid_origin_y,
            )
        else:
            grid_origin_x = 0.0
            grid_origin_y = 0.0
        self._grid_origin = (grid_origin_x, grid_origin_y)

        # Initialize grid
        shape = (
            self.config.num_channels,
            self.config.grid_height,
            self.config.grid_width,
        )
        # Boundary assertion: ensure grid dimensions are positive
        assert all(dim > 0 for dim in shape), f"Invalid grid shape: {shape}"
        self._grid_data = np.zeros(shape, dtype=self.config.dtype)

        logger.debug(
            f"Generating grid: shape={shape}, obstacles={len(obstacles)}, "
            f"pedestrians={len(pedestrians)}, ego_frame={use_ego_frame}, "
            f"origin=({grid_origin_x:.2f}, {grid_origin_y:.2f})"
        )

        # Transform coordinates to ego frame when requested
        if use_ego_frame:

            def _to_ego_point(point: tuple[float, float]) -> tuple[float, float]:
                return grid_utils.world_to_ego(point[0], point[1], self._last_robot_pose)  # type: ignore[arg-type]

            transformed_obstacles: list[Line2D] = [
                (_to_ego_point(start), _to_ego_point(end)) for start, end in obstacles
            ]
            transformed_polygons: list[list[tuple[float, float]]] | None = None
            if obstacle_polygons is not None:
                transformed_polygons = [
                    [_to_ego_point(vertex) for vertex in polygon] for polygon in obstacle_polygons
                ]
            transformed_pedestrians: list[Circle2D] = [
                (_to_ego_point(center), radius) for center, radius in pedestrians
            ]
        else:
            transformed_obstacles = obstacles
            transformed_polygons = obstacle_polygons
            transformed_pedestrians = pedestrians

        # Rasterize each channel
        for channel_idx, channel in enumerate(self.config.channels):
            if channel == GridChannel.COMBINED:
                # Combine after base channels are processed
                continue

            if channel == GridChannel.OBSTACLES:
                # Rasterize obstacles into this channel
                num_rasterized = rasterization.rasterize_obstacles(
                    transformed_obstacles,
                    self._grid_data[channel_idx],
                    self.config,
                    grid_origin_x,
                    grid_origin_y,
                    value=1.0,
                )
                logger.debug(f"Rasterized {num_rasterized} obstacles")
                if transformed_polygons:
                    filled = 0
                    for polygon in transformed_polygons:
                        filled += rasterization.rasterize_polygon(
                            polygon,
                            self._grid_data[channel_idx],
                            self.config,
                            grid_origin_x,
                            grid_origin_y,
                            value=1.0,
                        )
                    logger.debug("Filled %s obstacle cells via polygon rasterization", filled)

            elif channel == GridChannel.PEDESTRIANS:
                # Rasterize pedestrians into this channel
                num_rasterized = rasterization.rasterize_pedestrians(
                    transformed_pedestrians,
                    self._grid_data[channel_idx],
                    self.config,
                    grid_origin_x,
                    grid_origin_y,
                    value=1.0,
                )
                logger.debug(f"Rasterized {num_rasterized} pedestrians")

            elif channel == GridChannel.ROBOT:
                # Rasterize robot as a circle
                robot_radius = self.config.robot_radius
                robot_pose_for_grid = robot_pose
                if use_ego_frame and self._last_robot_pose is not None:
                    ego_x, ego_y = grid_utils.world_to_ego(
                        robot_pose[0][0], robot_pose[0][1], self._last_robot_pose
                    )  # type: ignore[arg-type]
                    robot_pose_for_grid = ((ego_x, ego_y), robot_pose[1])
                success = rasterization.rasterize_robot(
                    robot_pose_for_grid,
                    robot_radius,
                    self._grid_data[channel_idx],
                    self.config,
                    grid_origin_x,
                    grid_origin_y,
                    value=1.0,
                )
                logger.debug(f"Rasterized robot: {success}")

        if GridChannel.COMBINED in self.config.channels:
            combined_idx = self.config.channels.index(GridChannel.COMBINED)
            source_indices = [
                idx for idx, ch in enumerate(self.config.channels) if ch != GridChannel.COMBINED
            ]
            if source_indices:
                combined = np.max(self._grid_data[source_indices], axis=0)
            else:
                combined = np.zeros_like(self._grid_data[combined_idx])
            self._grid_data[combined_idx] = combined.astype(self.config.dtype, copy=False)
            logger.debug("Generated combined channel from %s indices", source_indices)

        # Cache obstacle polygons for direct spatial queries
        self._obstacle_polygons = transformed_polygons or []
        self._prepared_obstacles = self._prepare_obstacles(self._obstacle_polygons)

        return self._grid_data

    def query(self, query: POIQuery) -> POIResult:  # noqa: C901
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
            logger.error("Query called before grid generation")
            raise RuntimeError("Grid has not been generated yet. Call generate() first.")

        origin_x, origin_y = self._grid_origin or (0.0, 0.0)

        # Convert query coordinates into grid frame (ego/world)
        query_x, query_y = query.x, query.y
        if self._last_use_ego_frame:
            if self._last_robot_pose is None:
                raise RuntimeError("Cannot query ego-frame grid without a robot pose.")
            query_x, query_y = grid_utils.world_to_ego(query.x, query.y, self._last_robot_pose)  # type: ignore[arg-type]

        local_x = query_x - origin_x
        local_y = query_y - origin_y
        grid_col = int(local_x / self.config.resolution)
        grid_row = int(local_y / self.config.resolution)

        grid_col = int(np.clip(grid_col, 0, self.config.grid_width - 1))
        grid_row = int(np.clip(grid_row, 0, self.config.grid_height - 1))

        # Track cells to query based on query type
        cells_to_check: list[tuple[int, int]] = []

        contains_obstacle = False
        if query.query_type == POIQueryType.POINT:
            self._ensure_prepared_obstacles()
            if self._prepared_obstacles:
                pt = _ShapelyPoint(query_x, query_y)
                contains_obstacle = any(poly.contains(pt) for poly in self._prepared_obstacles)

        if query.query_type == POIQueryType.POINT:
            # Single cell query
            cells_to_check = [(grid_row, grid_col)]

        elif query.query_type == POIQueryType.CIRCLE:
            # Circular AOI: find all cells within radius
            radius_cells = int(query.radius / self.config.resolution) + 1
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    dist = math.sqrt(dx**2 + dy**2) * self.config.resolution
                    if dist <= query.radius:
                        row = grid_row + dy
                        col = grid_col + dx
                        if 0 <= col < self.config.grid_width and 0 <= row < self.config.grid_height:
                            cells_to_check.append((row, col))

        elif query.query_type == POIQueryType.RECT:
            # Rectangular AOI: find all cells in rectangle
            half_width = query.width / (2 * self.config.resolution)
            half_height = query.height / (2 * self.config.resolution)
            for dx in range(-int(half_width) - 1, int(half_width) + 2):
                for dy in range(-int(half_height) - 1, int(half_height) + 2):
                    row = grid_row + dy
                    col = grid_col + dx
                    if 0 <= col < self.config.grid_width and 0 <= row < self.config.grid_height:
                        cells_to_check.append((row, col))

        elif query.query_type == POIQueryType.LINE:
            # Line segment query: use Bresenham's line
            if query.x2 is None or query.y2 is None:
                raise ValueError("x2 and y2 required for LINE query")
            x2, y2 = query.x2, query.y2
            if self._last_use_ego_frame and self._last_robot_pose is not None:
                x2, y2 = grid_utils.world_to_ego(query.x2, query.y2, self._last_robot_pose)  # type: ignore[arg-type]
            grid_x2 = int((x2 - origin_x) / self.config.resolution)
            grid_y2 = int((y2 - origin_y) / self.config.resolution)
            # Use Bresenham-like line traversal
            line_cells = self._bresenham_line(grid_col, grid_row, grid_x2, grid_y2)
            cells_to_check = [
                (row, col)
                for col, row in line_cells
                if 0 <= col < self.config.grid_width and 0 <= row < self.config.grid_height
            ]

        # Compute occupancy statistics
        if not cells_to_check:
            return POIResult(
                occupancy=0.0,
                query_type=query.query_type,
                num_cells=0,
                min_occupancy=0.0,
                max_occupancy=0.0,
                mean_occupancy=0.0,
            )

        # Respect optional channel filter
        selected_channels = query.channels or self.config.channels
        selected_channels = [ch for ch in selected_channels if ch in self.config.channels]
        channel_indices = [self.config.channels.index(ch) for ch in selected_channels]
        if not channel_indices:
            return POIResult(
                occupancy=0.0,
                query_type=query.query_type,
                num_cells=len(cells_to_check),
                min_occupancy=0.0,
                max_occupancy=0.0,
                mean_occupancy=0.0,
                channel_results={},
            )

        if contains_obstacle and GridChannel.OBSTACLES in selected_channels:
            channel_results = {
                ch: (1.0 if ch == GridChannel.OBSTACLES else 0.0) for ch in selected_channels
            }
            return POIResult(
                occupancy=1.0,
                query_type=query.query_type,
                num_cells=len(cells_to_check),
                min_occupancy=1.0,
                max_occupancy=1.0,
                mean_occupancy=1.0,
                channel_results=channel_results,
            )

        # Extract occupancy values from all channels (aggregate)
        rows = np.array([r for r, _c in cells_to_check], dtype=int)
        cols = np.array([c for _r, c in cells_to_check], dtype=int)

        channel_values = self._grid_data[channel_indices][:, rows, cols].astype(float, copy=False)
        per_cell_max = channel_values.max(axis=0)
        per_cell_max = np.clip(per_cell_max, 0.0, 1.0)

        channel_results = {
            ch: float(np.clip(channel_values[idx], 0.0, 1.0).mean())
            for idx, ch in enumerate(selected_channels)
        }

        # Compute statistics
        min_occ = float(per_cell_max.min()) if per_cell_max.size > 0 else 0.0
        max_occ = float(per_cell_max.max()) if per_cell_max.size > 0 else 0.0
        mean_occ = float(per_cell_max.mean()) if per_cell_max.size > 0 else 0.0

        logger.debug(
            f"Query {query.query_type.value} at ({query.x}, {query.y}): "
            f"occupancy={mean_occ:.3f}, cells={len(cells_to_check)}"
        )

        return POIResult(
            occupancy=mean_occ,
            query_type=query.query_type,
            num_cells=len(cells_to_check),
            min_occupancy=min_occ,
            max_occupancy=max_occ,
            mean_occupancy=mean_occ,
            channel_results=channel_results,
        )

    def _prepare_obstacles(
        self, polygons: list[list[tuple[float, float]]]
    ) -> list[PreparedGeometry] | None:
        if not polygons:
            return None
        shapely_polygons = [_ShapelyPolygon(poly) for poly in polygons]
        return [prep(poly) for poly in shapely_polygons if not poly.is_empty]

    def _ensure_prepared_obstacles(self) -> None:
        if self._prepared_obstacles is None and self._obstacle_polygons:
            self._prepared_obstacles = self._prepare_obstacles(self._obstacle_polygons)

    def __getstate__(self):
        """Customize pickling to drop non-serializable prepared geometries.

        Returns:
            dict: Serializable state without prepared geometries.
        """
        state = self.__dict__.copy()
        # Drop shapely prepared geometries to keep pickling safe
        state["_prepared_obstacles"] = None
        return state

    def __setstate__(self, state):
        """Restore pickled state."""
        self.__dict__.update(state)
        self._prepared_obstacles = None

    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        """Bresenham's line algorithm to get all cells on a line.

        Args:
            x0: Starting x cell coordinate
            y0: Starting y cell coordinate
            x1: Ending x cell coordinate
            y1: Ending y cell coordinate

        Returns:
            List of cells on the line from (x0, y0) to (x1, y1)
        """
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
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return cells

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

        if pygame is None:
            raise RuntimeError("Pygame is required for occupancy grid rendering")

        if not (0 <= alpha <= 255):
            raise ValueError("alpha must be between 0 and 255")

        cell_size: int = max(1, round(scale))
        overlay = pygame.Surface(
            (self.config.grid_width * cell_size, self.config.grid_height * cell_size),
            pygame.SRCALPHA,
        )

        color_map: dict[GridChannel, tuple[int, int, int, int]] = {
            GridChannel.OBSTACLES: (255, 235, 128, alpha),  # light yellow
            GridChannel.PEDESTRIANS: (220, 80, 80, alpha),  # red tint
            GridChannel.ROBOT: (80, 140, 255, alpha),  # blue tint
            GridChannel.COMBINED: (255, 170, 120, alpha),  # orange overlay
        }

        for channel_idx, channel in enumerate(self.config.channels):
            channel_data = self._grid_data[channel_idx]
            rows, cols = np.nonzero(channel_data > 0)
            if rows.size == 0:
                continue

            color = color_map.get(channel, (200, 200, 200, alpha))
            for row, col in zip(rows, cols, strict=True):
                rect = pygame.Rect(int(col * cell_size), int(row * cell_size), cell_size, cell_size)
                overlay.fill(color, rect)

        surface.blit(overlay, (0, 0))
        logger.debug(
            "Rendered occupancy grid overlay: shape=%s scale=%s alpha=%s",
            overlay.get_size(),
            scale,
            alpha,
        )

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
            logger.error(f"Attempted to get channel {channel.value} before grid generation")
            raise RuntimeError("Grid has not been generated yet.")

        if channel not in self.config.channels:
            logger.error(
                f"Invalid channel request: {channel.value}, available: "
                f"{[c.value for c in self.config.channels]}"
            )
            raise ValueError(f"Channel {channel.value} not in grid")

        channel_idx = self.config.channels.index(channel)
        return self._grid_data[channel_idx]

    def to_observation(self) -> np.ndarray:
        """Convert occupancy grid to gymnasium observation array.

        Returns grid data as numpy array suitable for gymnasium observation spaces.
        Array has shape [C, H, W] with dtype float32 and values in [0, 1].

        Returns:
            numpy array of shape [C, H, W] with dtype float32, values in [0, 1]

        Raises:
            RuntimeError: If grid has not been generated yet

        Example:
            >>> grid = OccupancyGrid(config)
            >>> grid.generate(obstacles=obs, pedestrians=peds, robot_pose=pose)
            >>> obs_array = grid.to_observation()
            >>> obs_array.shape  # (num_channels, height, width)
            (3, 200, 200)
            >>> obs_array.dtype
            dtype('float32')
            >>> (obs_array.min(), obs_array.max())
            (0.0, 1.0)

        Notes:
            - This method is optimized for gymnasium/StableBaselines3 compatibility
            - Output is always float32 regardless of internal grid dtype
            - Values are guaranteed to be in [0, 1] range (clipped if necessary)
            - Channel order matches config.channels (e.g., [obstacles, pedestrians, combined])
        """
        if self._grid_data is None:
            raise RuntimeError(
                "Grid has not been generated yet. Call generate() before to_observation()."
            )

        # Convert to float32 if needed
        if self._grid_data.dtype != np.float32:
            obs_array = self._grid_data.astype(np.float32)
        else:
            # Make a copy to avoid modifying internal state
            obs_array = self._grid_data.copy()

        # Ensure values are in [0, 1] range (clip if needed)
        np.clip(obs_array, 0.0, 1.0, out=obs_array)

        logger.debug(
            f"Grid to observation: shape={obs_array.shape}, "
            f"dtype={obs_array.dtype}, "
            f"range=[{obs_array.min():.3f}, {obs_array.max():.3f}]"
        )

        return obs_array

    def __repr__(self) -> str:
        """String representation of occupancy grid.

        Returns:
            str: String representation including config, shape, and initialization status.
        """
        status = "initialized" if self.is_initialized else "not initialized"
        return f"OccupancyGrid(config={self.config}, shape={self.shape}, status={status})"
