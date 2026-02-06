"""
define the map configuration
"""

import json
import os
import random
from dataclasses import dataclass, field
from math import sqrt

import matplotlib.axes
import numpy as np
from loguru import logger
from shapely.geometry import Point, Polygon

from robot_sf.common.types import Line2D, Rect, Vec2D
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.obstacle import Obstacle


@dataclass
class PedestrianWaitRule:
    """
    Represents a wait action at a specific trajectory waypoint.

    Attributes:
        waypoint_index (int): Index into the pedestrian's trajectory list.
        wait_s (float): Duration in seconds to wait at the waypoint.
        note (str | None): Optional human-readable note for documentation.
    """

    waypoint_index: int
    wait_s: float
    note: str | None = None

    def __post_init__(self):
        """Validate waypoint index and wait duration."""
        if self.waypoint_index < 0:
            raise ValueError("wait_at waypoint_index must be >= 0")
        if self.wait_s < 0:
            raise ValueError("wait_at wait_s must be >= 0")


@dataclass
class SinglePedestrianDefinition:
    """
    Represents an individually controlled pedestrian with a unique ID, start position,
    and either a goal position or a trajectory (mutually exclusive).

    Attributes:
        id (str): Unique identifier for the pedestrian
        start (Vec2D): Starting position as (x, y) coordinates
        goal (Vec2D | None): Goal position for goal-based navigation (mutually exclusive with trajectory)
        trajectory (list[Vec2D] | None): List of waypoints for trajectory-based movement (mutually exclusive with goal)
        speed_m_s (float | None): Optional per-pedestrian speed override (m/s).
        wait_at (list[PedestrianWaitRule] | None): Optional waits at trajectory waypoints.
        note (str | None): Optional human-readable note for scenario documentation.
        role (str | None): Optional runtime behavior role (wait, follow, lead, accompany, join, leave).
        role_target_id (str | None): Optional target identifier for role behaviors (e.g., "robot:0").
        role_offset (Vec2D | None): Optional (forward, lateral) offset for follow/lead/accompany roles.
    """

    id: str
    start: Vec2D
    goal: Vec2D | None = None
    trajectory: list[Vec2D] | None = None
    speed_m_s: float | None = None
    wait_at: list[PedestrianWaitRule] | None = None
    note: str | None = None
    role: str | None = None
    role_target_id: str | None = None
    role_offset: Vec2D | None = None

    def __post_init__(self):
        """
        Validates the pedestrian definition.
        Ensures goal and trajectory are mutually exclusive and that required fields are present.

        Raises:
            ValueError: If validation fails for critical errors (mutually exclusive goal/trajectory,
                       invalid ID, goal equals start, duplicate trajectory waypoints)
        """
        self._validate_id()
        self._validate_start_position()
        self._validate_goal_trajectory_exclusivity()
        self._validate_goal()
        self._validate_trajectory()
        self._validate_speed()
        self._validate_waits()
        self._validate_role()
        self._validate_role_target()
        self._validate_role_offset()
        self._warn_if_static()

    def _validate_id(self):
        """Validate pedestrian ID is a non-empty string."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Pedestrian ID must be a non-empty string, got: {self.id!r}")

    def _validate_start_position(self):
        """Validate start position is a 2-tuple."""
        if not isinstance(self.start, tuple) or len(self.start) != 2:
            raise ValueError(
                f"Pedestrian '{self.id}': start position must be a 2-tuple (x, y), got: {self.start!r}"
            )

    def _validate_goal_trajectory_exclusivity(self):
        """Ensure goal and trajectory are mutually exclusive."""
        if self.goal is not None and self.trajectory is not None:
            raise ValueError(
                f"Pedestrian '{self.id}': goal and trajectory are mutually exclusive. "
                "Provide either goal or trajectory, not both."
            )

    def _validate_goal(self):
        """Validate goal if present."""
        if self.goal is None:
            return

        if not isinstance(self.goal, tuple) or len(self.goal) != 2:
            raise ValueError(
                f"Pedestrian '{self.id}': goal must be a 2-tuple (x, y), got: {self.goal!r}"
            )

        if self.goal == self.start:
            logger.warning(
                f"Pedestrian '{self.id}': goal equals start position. "
                "Pedestrian will remain at the same location."
            )

    def _validate_trajectory(self):
        """Validate trajectory if present."""
        if self.trajectory is None:
            return

        if not isinstance(self.trajectory, list):
            raise ValueError(
                f"Pedestrian '{self.id}': trajectory must be a list of waypoints, "
                f"got: {type(self.trajectory)}"
            )

        if len(self.trajectory) == 0:
            logger.info(
                f"Pedestrian '{self.id}': trajectory is empty. "
                "Pedestrian will remain static or move directly to goal."
            )
            return

        # Validate each waypoint
        for idx, wp in enumerate(self.trajectory):
            if not isinstance(wp, tuple) or len(wp) != 2:
                raise ValueError(
                    f"Pedestrian '{self.id}': trajectory waypoint {idx} must be "
                    f"a 2-tuple (x, y), got: {wp!r}"
                )

        # Check for duplicate consecutive waypoints
        self._warn_duplicate_waypoints()

    def _validate_speed(self):
        """Validate optional speed override."""
        if self.speed_m_s is None:
            return
        try:
            if float(self.speed_m_s) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(
                f"Pedestrian '{self.id}': speed_m_s must be a positive number, got: {self.speed_m_s!r}"
            ) from None

    def _validate_waits(self):
        """Validate optional wait rules."""
        if self.wait_at is None:
            return
        if not isinstance(self.wait_at, list):
            raise ValueError(
                f"Pedestrian '{self.id}': wait_at must be a list, got: {type(self.wait_at)}"
            )
        for rule in self.wait_at:
            if not isinstance(rule, PedestrianWaitRule):
                raise ValueError(
                    f"Pedestrian '{self.id}': wait_at entries must be PedestrianWaitRule instances"
                )

    def _warn_duplicate_waypoints(self):
        """Warn about duplicate consecutive trajectory waypoints."""
        if not self.trajectory or len(self.trajectory) < 2:
            return

        for idx in range(len(self.trajectory) - 1):
            if self.trajectory[idx] == self.trajectory[idx + 1]:
                logger.warning(
                    f"Pedestrian '{self.id}': duplicate consecutive waypoints at index {idx} "
                    f"({self.trajectory[idx]}). This may cause navigation issues."
                )

    def _warn_if_static(self):
        """Warn if pedestrian has neither goal nor trajectory (static)."""
        if self.goal is None and self.trajectory is None and self.role is None:
            logger.warning(
                f"Pedestrian '{self.id}': neither goal nor trajectory specified. "
                "Pedestrian will remain static."
            )

    def _validate_role(self) -> None:
        """Validate and normalize the optional role field."""
        if self.role is None:
            return
        if not isinstance(self.role, str):
            raise ValueError(f"Pedestrian '{self.id}': role must be a string, got {self.role!r}")
        normalized = self.role.strip().lower()
        if not normalized or normalized == "none":
            self.role = None
            return
        allowed = {"wait", "follow", "lead", "accompany", "join", "leave"}
        if normalized not in allowed:
            raise ValueError(
                f"Pedestrian '{self.id}': role must be one of {sorted(allowed)}, got {self.role!r}"
            )
        self.role = normalized

    def _validate_role_target(self) -> None:
        """Validate optional role target identifiers."""
        if self.role_target_id is None:
            return
        if not isinstance(self.role_target_id, str) or not self.role_target_id.strip():
            raise ValueError(
                f"Pedestrian '{self.id}': role_target_id must be a non-empty string, "
                f"got {self.role_target_id!r}"
            )
        self.role_target_id = self.role_target_id.strip()

    def _validate_role_offset(self) -> None:
        """Validate optional role offsets for follow/lead/accompany behavior."""
        if self.role_offset is None:
            return
        if not isinstance(self.role_offset, (tuple, list)) or len(self.role_offset) != 2:
            raise ValueError(
                f"Pedestrian '{self.id}': role_offset must be a 2-item tuple/list, "
                f"got {self.role_offset!r}"
            )
        self.role_offset = (float(self.role_offset[0]), float(self.role_offset[1]))


@dataclass
class MapDefinition:
    """
    A class to represent a map definition.

    Methods
    -------
    __post_init__():
        Validates the map definition and initializes the obstacles_pysf and
            robot_routes_by_spawn_id attributes.
    num_start_pos():
        Returns the number of start positions.
    max_target_dist():
        Returns the maximum target distance.
    find_route(spawn_id: int, goal_id: int):
        Returns the route for the given spawn id and goal id.
    """

    width: float
    """The width of the map."""

    height: float

    obstacles: list[Obstacle]
    """The obstacles in the map."""

    robot_spawn_zones: list[Rect]
    """The robot spawn zones. Mustn't be empty."""

    ped_spawn_zones: list[Rect]
    robot_goal_zones: list[Rect]
    """The robot goal zones. Mustn't be empty."""

    bounds: list[Line2D]
    """Map boundary segments in either flat (x_start, x_end, y_start, y_end) or pair-of-points form.

    Runtime normalization converts pair-of-points bounds into the flat tuple format
    expected by the fast-pysf backend and legacy map utilities.
    """
    robot_routes: list[GlobalRoute]
    ped_goal_zones: list[Rect]
    ped_crowded_zones: list[Rect]
    ped_routes: list[GlobalRoute]
    single_pedestrians: list[SinglePedestrianDefinition] = field(default_factory=list)
    """List of individually controlled pedestrians with explicit start/goal/trajectory definitions."""

    poi_positions: list[Vec2D] = field(default_factory=list)
    """Point-of-interest waypoints parsed from SVG maps."""

    poi_labels: dict[str, str] = field(default_factory=dict)
    """Mapping from POI identifiers to human-readable labels."""

    allowed_areas: list[Polygon] | None = None
    """Explicit driveable areas from OSM. Optional, used for path planning bounds."""

    _poi_positions_by_label: dict[str, Vec2D] = field(init=False, default_factory=dict, repr=False)
    """Internal lookup table from POI label to position for faster access."""
    obstacles_pysf: list[Line2D] = field(init=False)
    """Transformed obstacles in pysf format. Are generated in __post_init__."""
    robot_routes_by_spawn_id: dict[int, list[GlobalRoute]] = field(init=False)

    def __post_init__(self):
        """
        Validates the map definition and initializes the obstacles_pysf and
        robot_routes_by_spawn_id attributes.
        Raises a ValueError if the width or height is less than 0,
        if the robot spawn zones or goal zones are empty,
        or if the bounds are not exactly 4.
        """
        self.bounds = _normalize_bounds(self.bounds)
        obstacle_lines = [line for obstacle in self.obstacles for line in obstacle.lines]
        self.obstacles_pysf = obstacle_lines + self.bounds

        self.robot_routes_by_spawn_id = {}
        for route in self.robot_routes:
            if route.spawn_id in self.robot_routes_by_spawn_id:
                self.robot_routes_by_spawn_id[route.spawn_id].append(route)
            else:
                self.robot_routes_by_spawn_id[route.spawn_id] = [route]
        if not self.robot_routes:
            logger.warning("MapDefinition has no robot routes; planners may synthesize defaults.")

        if self.width <= 0 or self.height <= 0:
            logger.critical(
                "Map width and height mustn't be zero or negative! "
                + f"Width: {self.width}, Height: {self.height}",
            )

        if not self.robot_spawn_zones:
            logger.error("Robot spawn zones mustn't be empty!")

        if not self.robot_goal_zones:
            logger.error("Robot goal zones mustn't be empty!")

        if len(self.bounds) != 4:
            logger.critical(
                "Invalid bounds! Expected exactly 4 bounds! " + f"Found {len(self.bounds)} bounds!",
            )

        # Validate single pedestrians
        self._validate_single_pedestrians()
        self._validate_pois()
        self._build_poi_lookup()

    def _validate_single_pedestrians(self):
        """
        Validates single pedestrian definitions for:
        - Unique IDs
        - Valid coordinates within map bounds
        - Mutually exclusive goal/trajectory (already validated in SinglePedestrianDefinition)
        - Warns about overlapping start positions
        """
        if not self.single_pedestrians:
            return

        self._check_duplicate_ids()
        self._validate_pedestrian_coordinates()
        self._warn_overlapping_positions()

    def _check_duplicate_ids(self):
        """Check for duplicate pedestrian IDs."""
        seen_ids = set()
        duplicates = []
        for ped in self.single_pedestrians:
            if ped.id in seen_ids:
                duplicates.append(ped.id)
            seen_ids.add(ped.id)

        if duplicates:
            raise ValueError(
                f"Duplicate single pedestrian IDs found: {', '.join(set(duplicates))}. "
                "Ensure all pedestrian IDs are unique."
            )

    def _validate_pedestrian_coordinates(self):
        """Validate that pedestrian coordinates are within map bounds."""
        for ped in self.single_pedestrians:
            self._check_start_position(ped)
            self._check_goal_position(ped)
            self._check_trajectory_waypoints(ped)

    def _check_start_position(self, ped: SinglePedestrianDefinition):
        """Check if start position is within bounds."""
        x, y = ped.start
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            logger.warning(
                f"Pedestrian '{ped.id}' start position ({x}, {y}) is outside map bounds "
                f"(0, 0) to ({self.width}, {self.height})."
            )

    def _check_goal_position(self, ped: SinglePedestrianDefinition):
        """Check if goal position is within bounds."""
        if ped.goal is None:
            return
        gx, gy = ped.goal
        if not (0 <= gx <= self.width and 0 <= gy <= self.height):
            logger.warning(
                f"Pedestrian '{ped.id}' goal position ({gx}, {gy}) is outside map bounds "
                f"(0, 0) to ({self.width}, {self.height})."
            )

    def _check_trajectory_waypoints(self, ped: SinglePedestrianDefinition):
        """Check if trajectory waypoints are within bounds."""
        if ped.trajectory is None:
            return
        for idx, waypoint in enumerate(ped.trajectory):
            wx, wy = waypoint
            if not (0 <= wx <= self.width and 0 <= wy <= self.height):
                logger.warning(
                    f"Pedestrian '{ped.id}' trajectory waypoint {idx} ({wx}, {wy}) "
                    f"is outside map bounds (0, 0) to ({self.width}, {self.height})."
                )

    def _warn_overlapping_positions(self):
        """Warn about pedestrians with overlapping start positions."""
        for i, ped1 in enumerate(self.single_pedestrians):
            for ped2 in self.single_pedestrians[i + 1 :]:
                dist = sqrt(
                    (ped1.start[0] - ped2.start[0]) ** 2 + (ped1.start[1] - ped2.start[1]) ** 2
                )
                if dist < 0.5:  # threshold for overlap warning
                    logger.warning(
                        f"Pedestrians '{ped1.id}' and '{ped2.id}' have overlapping start positions "
                        f"(distance: {dist:.2f}m < 0.5m threshold)."
                    )

    def _validate_pois(self) -> None:
        """Validate POI identifiers, counts, and bounds."""
        if not self.poi_positions and not self.poi_labels:
            return

        if len(self.poi_positions) != len(self.poi_labels):
            raise ValueError(
                "poi_positions and poi_labels must have the same length: "
                f"{len(self.poi_positions)} positions vs {len(self.poi_labels)} labels."
            )

        if len(set(self.poi_labels.keys())) != len(self.poi_labels):
            raise ValueError("POI identifiers must be unique.")

        for poi in self.poi_positions:
            if not (0 <= poi[0] <= self.width and 0 <= poi[1] <= self.height):
                raise ValueError(
                    f"POI position {poi} is outside map bounds (0, 0) to ({self.width}, {self.height})."
                )

    def _build_poi_lookup(self) -> None:
        """Construct a label-to-position mapping for efficient POI lookup."""
        self._poi_positions_by_label.clear()
        for idx, poi_label in enumerate(self.poi_labels.values()):
            # Preserve first occurrence if duplicate labels exist
            self._poi_positions_by_label.setdefault(poi_label, self.poi_positions[idx])

    @property
    def num_start_pos(self) -> int:
        """
        Returns the number of start positions as an integer.
        """
        return len({r.spawn_id for r in self.robot_routes})

    @property
    def max_target_dist(self) -> float:
        """
        Returns the maximum target distance as a float.
        """
        return sqrt(2) * (max(self.width, self.height) * 2)

    def find_route(self, spawn_id: int, goal_id: int) -> GlobalRoute | None:
        """
        Returns the route for the given spawn id and goal id. If no route is found, returns None.

        Returns:
            GlobalRoute | None: The matching route object, or None if no route matches the
                given spawn and goal IDs.
        """
        return next(
            filter(
                lambda r: r.goal_id == goal_id and r.spawn_id == spawn_id,
                self.robot_routes,
            ),
            None,
        )

    def get_poi_by_label(self, label: str) -> Vec2D:
        """Retrieve POI position by human-readable label.

        Args:
            label: POI label to search for.

        Returns:
            POI position as a Vec2D tuple.

        Raises:
            KeyError: If the requested label does not exist.
        """
        try:
            return self._poi_positions_by_label[label]
        except KeyError as exc:  # pragma: no cover - thin wrapper
            raise KeyError(f"No POI with label '{label}' found in map.") from exc

    def get_map_bounds(self):
        """Returns the min and max of x and y bounds.

        Returns:
            tuple: Contains minimum and maximum coordinates as (x_min, x_max, y_min, y_max).
                    - x_min (float): Minimum x coordinate
                    - x_max (float): Maximum x coordinate
                    - y_min (float): Minimum y coordinate
                    - y_max (float): Maximum y coordinate
        """

        # Flatten list of tuples into separate x and y coordinates
        x_coords = []
        y_coords = []

        for x_start, x_end, y_start, y_end in self.bounds:
            x_coords.extend([x_start, x_end])
            y_coords.extend([y_start, y_end])

        # Get min/max values
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return x_min, x_max, y_min, y_max

    def plot_map_obstacles(self, ax):
        """Plot map obstacles on the given matplotlib axis.

        Args:
            ax: matplotlib.axes.Axes
                The axis on which to plot the obstacles.

        Raises:
            TypeError: If ax is not a matplotlib.axes.Axes object.
        """

        if not isinstance(ax, matplotlib.axes.Axes):
            raise TypeError("ax must be a matplotlib.axes.Axes object")
        for obstacle in self.obstacles:
            vertices = np.array(obstacle.vertices)
            ax.fill(vertices[:, 0], vertices[:, 1], "black")

    def is_point_in_driveable_area(self, point: Vec2D) -> bool:
        """Check if a point is in a driveable area.

        If allowed_areas is set (from OSM), checks containment in those explicit areas.
        Otherwise, checks that the point is NOT in any obstacle.

        Args:
            point: (x, y) coordinate tuple

        Returns:
            True if point is in driveable area, False otherwise
        """
        p = Point(point)

        if self.allowed_areas is not None:
            # Use explicit driveable areas if present (from OSM)
            return any(poly.contains(p) for poly in self.allowed_areas)

        # Fallback: point is driveable if NOT in any obstacle
        return not any(obs.contains_point(point) for obs in self.obstacles)

    def __getstate__(self):
        """Customize pickling to drop non-serializable prepared geometries.

        Returns:
            dict: Serializable state without prepared geometries.
        """
        state = self.__dict__.copy()
        # Drop shapely prepared geometries to keep pickling safe
        state.pop("_prepared_obstacles", None)
        return state

    def __setstate__(self, state):
        """Restore state from unpickling, reinitializing prepared geometries as None."""
        self.__dict__.update(state)
        # Prepared obstacles will be lazily recreated when needed
        if not hasattr(self, "_prepared_obstacles"):
            self._prepared_obstacles = None
        # Legacy pickle compatibility: initialize missing POI-related fields
        if not hasattr(self, "_poi_positions_by_label"):
            self._poi_positions_by_label = {}
        if not hasattr(self, "poi_labels"):
            self.poi_labels = {}
        if not hasattr(self, "poi_positions"):
            self.poi_positions = []
        if not hasattr(self, "single_pedestrians"):
            self.single_pedestrians = []
        self._build_poi_lookup()


@dataclass
class MapDefinitionPool:
    """
    A class to represent a pool of map definitions.

    Attributes
    ----------
    maps_folder : str
        The directory where the map files are located.
    map_defs : Dict[str, MapDefinition]
        The dictionary of map definitions, with the map name as the key.

    Methods
    -------
    __post_init__():
        Validates and initializes the map_defs attribute.
    choose_random_map():
        Returns a random map definition from the pool.
    """

    maps_folder: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "maps")
    """The directory where the **default** map files are located."""
    map_defs: dict[str, MapDefinition] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validates and initializes the map_defs attribute.
        If map_defs is empty, it loads the map definitions from the files in the
        maps_folder directory.
        Raises a ValueError if the maps_folder directory does not exist or if
        map_defs is still empty after loading.
        """

        # If map_defs is empty, load the map definitions from the files
        if not self.map_defs:
            self.map_defs = self._load_json_map_definitions_from_folder(self.maps_folder)

        # If map_defs is still empty, raise an error
        if not self.map_defs:
            raise ValueError("Map pool is empty! Please specify some maps!")

    def _load_json_map_definitions_from_folder(self, maps_folder: str) -> dict[str, MapDefinition]:
        """
        Load json map definitions from a folder in the maps_folder directory.

        Returns:
            dict[str, MapDefinition]: Dictionary mapping filenames (without extension) to
                parsed MapDefinition objects for all .json files found in the folder.
        """

        # Check if the maps_folder directory exists
        if not os.path.exists(maps_folder) or not os.path.isdir(maps_folder):
            raise ValueError(f"Map directory '{maps_folder}' does not exist!")

        # Function to load a JSON file
        def load_json(path: str) -> dict:
            """Load and parse a JSON file from the given path.

            Returns:
                dict: Parsed JSON content as a Python dictionary.

            Args:
                path: TODO docstring.

            Returns:
                TODO docstring.
            """
            with open(path, encoding="utf-8") as file:
                return json.load(file)

        # Get only JSON files (skip directories and non-json entries)
        map_files = [
            os.path.join(maps_folder, f)
            for f in os.listdir(maps_folder)
            if f.lower().endswith(".json") and os.path.isfile(os.path.join(maps_folder, f))
        ]

        if not map_files:
            logger.debug(
                "No JSON map definition files found in '%s'; returning empty map_defs", maps_folder
            )
            return {}

        # Load the map definitions from the JSON files
        map_defs = {
            os.path.splitext(os.path.basename(f))[0]: serialize_map(load_json(f)) for f in map_files
        }

        return map_defs

    def choose_random_map(self) -> MapDefinition:
        """
        Returns a random map definition from the pool.

        Returns
        -------
        MapDefinition
            A random map definition.
        """

        return random.choice(list(self.map_defs.values()))

    def get_map(self, map_id: str) -> MapDefinition:
        """Return a map definition by id.

        Args:
            map_id: Key in the map_defs dictionary.

        Returns:
            MapDefinition: The requested map definition.

        Raises:
            KeyError: If map_id is not present in the pool.
        """
        key = map_id.strip()
        if not key:
            raise KeyError("map_id must be a non-empty string")
        try:
            return self.map_defs[key]
        except KeyError as exc:
            known = ", ".join(sorted(self.map_defs.keys())) or "<none>"
            raise KeyError(f"Unknown map_id '{key}'. Known: {known}") from exc


def _normalize_position(pos: Vec2D, *, min_x: float, min_y: float) -> Vec2D:
    """Normalize a position against map margins.

    Args:
        pos: Original position as (x, y).
        min_x: Minimum x margin.
        min_y: Minimum y margin.

    Returns:
        Vec2D: Normalized position.
    """
    return (pos[0] - min_x, pos[1] - min_y)


def _normalize_zone(rect: Rect, *, min_x: float, min_y: float) -> Rect:
    """Normalize a rectangular zone using map margins.

    Args:
        rect: Zone represented by three points.
        min_x: Minimum x margin.
        min_y: Minimum y margin.

    Returns:
        Rect: Normalized zone.
    """
    return (
        _normalize_position(rect[0], min_x=min_x, min_y=min_y),
        _normalize_position(rect[1], min_x=min_x, min_y=min_y),
        _normalize_position(rect[2], min_x=min_x, min_y=min_y),
    )


def _normalize_zones(zones: list[Rect], *, min_x: float, min_y: float) -> list[Rect]:
    """Normalize a list of zones using map margins.

    Args:
        zones: List of zone rectangles.
        min_x: Minimum x margin.
        min_y: Minimum y margin.

    Returns:
        list[Rect]: Normalized zones.
    """
    return [_normalize_zone(zone, min_x=min_x, min_y=min_y) for zone in zones]


def _normalize_obstacles(
    obstacle_vertices: list[list[Vec2D]],
    *,
    min_x: float,
    min_y: float,
) -> list[Obstacle]:
    """Normalize obstacle vertices using map margins.

    Args:
        obstacle_vertices: Raw obstacle vertex lists.
        min_x: Minimum x margin.
        min_y: Minimum y margin.

    Returns:
        list[Obstacle]: Normalized obstacles.
    """
    return [
        Obstacle([_normalize_position(p, min_x=min_x, min_y=min_y) for p in vertices])
        for vertices in obstacle_vertices
    ]


def _normalize_routes(
    route_defs: list[dict],
    *,
    spawn_zones: list[Rect],
    goal_zones: list[Rect],
    min_x: float,
    min_y: float,
) -> list[GlobalRoute]:
    """Normalize route definitions into GlobalRoute objects.

    Args:
        route_defs: Raw route dictionaries.
        spawn_zones: Normalized spawn zones for lookups.
        goal_zones: Normalized goal zones for lookups.
        min_x: Minimum x margin.
        min_y: Minimum y margin.

    Returns:
        list[GlobalRoute]: Normalized routes.
    """
    return [
        GlobalRoute(
            entry["spawn_id"],
            entry["goal_id"],
            [_normalize_position(point, min_x=min_x, min_y=min_y) for point in entry["waypoints"]],
            spawn_zones[entry["spawn_id"]],
            goal_zones[entry["goal_id"]],
        )
        for entry in route_defs
    ]


def _reverse_route(route: GlobalRoute) -> GlobalRoute:
    """Create a reversed route for a GlobalRoute.

    Args:
        route: Route to reverse.

    Returns:
        GlobalRoute: Reversed route.
    """
    return GlobalRoute(
        route.goal_id,
        route.spawn_id,
        list(reversed(route.waypoints)),
        route.goal_zone,
        route.spawn_zone,
    )


def _parse_wait_rules(wait_rules: list[dict] | None) -> list[PedestrianWaitRule] | None:
    """Parse wait rules from JSON map definitions.

    Returns:
        list[PedestrianWaitRule] | None: Parsed wait rules or ``None`` when unset.
    """
    if wait_rules is None:
        return None
    if not isinstance(wait_rules, list):
        raise ValueError("single_pedestrians.wait_at must be a list of wait rules")
    parsed: list[PedestrianWaitRule] = []
    for idx, rule in enumerate(wait_rules):
        if not isinstance(rule, dict):
            raise ValueError(
                f"single_pedestrians.wait_at[{idx}] must be a mapping, got {type(rule)}"
            )
        if "waypoint_index" not in rule:
            raise ValueError(f"single_pedestrians.wait_at[{idx}] missing required waypoint_index")
        if "wait_s" not in rule:
            raise ValueError(f"single_pedestrians.wait_at[{idx}] missing required wait_s")
        note = rule.get("note")
        parsed.append(
            PedestrianWaitRule(
                waypoint_index=int(rule["waypoint_index"]),
                wait_s=float(rule["wait_s"]),
                note=str(note) if note is not None else None,
            )
        )
    return parsed


def _parse_single_pedestrians(
    map_structure: dict,
    *,
    min_x: float,
    min_y: float,
) -> list[SinglePedestrianDefinition]:
    """Parse single pedestrian definitions from a JSON map definition.

    Args:
        map_structure: Raw map structure dictionary.
        min_x: Minimum x margin.
        min_y: Minimum y margin.

    Returns:
        list[SinglePedestrianDefinition]: Parsed single pedestrians.
    """
    if "single_pedestrians" not in map_structure:
        return []
    single_pedestrians: list[SinglePedestrianDefinition] = []
    for ped_def in map_structure["single_pedestrians"]:
        ped_id = ped_def["id"]
        start = _normalize_position(tuple(ped_def["start"]), min_x=min_x, min_y=min_y)
        goal = (
            _normalize_position(tuple(ped_def["goal"]), min_x=min_x, min_y=min_y)
            if ped_def.get("goal")
            else None
        )
        trajectory = (
            [
                _normalize_position(tuple(wp), min_x=min_x, min_y=min_y)
                for wp in ped_def["trajectory"]
            ]
            if ped_def.get("trajectory")
            else None
        )
        speed = ped_def.get("speed_m_s")
        wait_at = _parse_wait_rules(ped_def.get("wait_at"))
        note = ped_def.get("note")
        role = ped_def.get("role")
        role_target_id = ped_def.get("role_target_id")
        role_offset = ped_def.get("role_offset")
        role_target_value = str(role_target_id) if role_target_id is not None else None
        role_offset_tuple = None
        if role_offset is not None:
            if not isinstance(role_offset, (list, tuple)) or len(role_offset) != 2:
                raise ValueError(
                    "single_pedestrians.role_offset must be a 2-item list or tuple, "
                    f"got: {role_offset!r}"
                )
            role_offset_tuple = (float(role_offset[0]), float(role_offset[1]))
        single_pedestrians.append(
            SinglePedestrianDefinition(
                id=ped_id,
                start=start,
                goal=goal,
                trajectory=trajectory,
                speed_m_s=float(speed) if speed is not None else None,
                wait_at=wait_at,
                note=str(note) if note is not None else None,
                role=str(role) if role is not None else None,
                role_target_id=role_target_value,
                role_offset=role_offset_tuple,
            )
        )
    return single_pedestrians


def _build_map_bounds(width: float, height: float) -> list[Line2D]:
    """Build boundary lines for a rectangular map.

    Args:
        width: Map width.
        height: Map height.

    Returns:
        list[Line2D]: Boundary lines.
    """
    return [
        (0, width, 0, 0),  # bottom
        (0, width, height, height),  # top
        (0, 0, 0, height),  # left
        (width, width, 0, height),  # right
    ]


def _normalize_bounds(bounds: list[Line2D]) -> list[Line2D]:
    """Normalize bounds into the flat (x_start, x_end, y_start, y_end) format.

    Accepts either flat 4-tuples or pair-of-point Line2D entries and returns
    the flat representation used by the fast-pysf backend and map utilities.

    Returns:
        list[Line2D]: Bounds normalized into flat tuple format.
    """
    normalized: list[Line2D] = []
    for bound in bounds:
        if isinstance(bound, (tuple, list)) and len(bound) == 2:
            try:
                (x1, y1), (x2, y2) = bound  # type: ignore[misc]
                normalized.append((float(x1), float(x2), float(y1), float(y2)))
                continue
            except (TypeError, ValueError):
                # Fall back to raw bound when unpacking fails.
                pass
        normalized.append(bound)
    return normalized


def serialize_map(map_structure: dict) -> MapDefinition:
    """Convert a map structure dictionary into a MapDefinition object.

    Args:
        map_structure: The map structure dictionary.

    Returns:
        MapDefinition: The parsed map definition.
    """
    (min_x, max_x), (min_y, max_y) = map_structure["x_margin"], map_structure["y_margin"]
    width, height = max_x - min_x, max_y - min_y

    obstacles = _normalize_obstacles(map_structure["obstacles"], min_x=min_x, min_y=min_y)
    robot_goal_zones = _normalize_zones(map_structure["robot_goal_zones"], min_x=min_x, min_y=min_y)
    robot_spawn_zones = _normalize_zones(
        map_structure["robot_spawn_zones"], min_x=min_x, min_y=min_y
    )
    ped_goal_zones = _normalize_zones(map_structure["ped_goal_zones"], min_x=min_x, min_y=min_y)
    ped_spawn_zones = _normalize_zones(map_structure["ped_spawn_zones"], min_x=min_x, min_y=min_y)
    ped_crowded_zones = _normalize_zones(
        map_structure["ped_crowded_zones"], min_x=min_x, min_y=min_y
    )

    robot_routes = _normalize_routes(
        map_structure["robot_routes"],
        spawn_zones=robot_spawn_zones,
        goal_zones=robot_goal_zones,
        min_x=min_x,
        min_y=min_y,
    )
    ped_routes = _normalize_routes(
        map_structure["ped_routes"],
        spawn_zones=ped_spawn_zones,
        goal_zones=ped_goal_zones,
        min_x=min_x,
        min_y=min_y,
    )
    robot_routes = robot_routes + [_reverse_route(route) for route in robot_routes]

    single_pedestrians = _parse_single_pedestrians(map_structure, min_x=min_x, min_y=min_y)
    map_bounds = _build_map_bounds(width, height)

    return MapDefinition(
        width,
        height,
        obstacles,
        robot_spawn_zones,
        ped_spawn_zones,
        robot_goal_zones,
        map_bounds,
        robot_routes,
        ped_goal_zones,
        ped_crowded_zones,
        ped_routes,
        single_pedestrians,
    )
