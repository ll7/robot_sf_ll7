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

from robot_sf.common.types import Line2D, Rect, Vec2D
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.obstacle import Obstacle


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
    """

    id: str
    start: Vec2D
    goal: Vec2D | None = None
    trajectory: list[Vec2D] | None = None

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
        if self.goal is None and self.trajectory is None:
            logger.warning(
                f"Pedestrian '{self.id}': neither goal nor trajectory specified. "
                "Pedestrian will remain static."
            )


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
    robot_routes: list[GlobalRoute]
    ped_goal_zones: list[Rect]
    ped_crowded_zones: list[Rect]
    ped_routes: list[GlobalRoute]
    single_pedestrians: list[SinglePedestrianDefinition] = field(default_factory=list)
    """List of individually controlled pedestrians with explicit start/goal/trajectory definitions."""
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
        obstacle_lines = [line for obstacle in self.obstacles for line in obstacle.lines]
        self.obstacles_pysf = obstacle_lines + self.bounds

        self.robot_routes_by_spawn_id = {}
        for route in self.robot_routes:
            if route.spawn_id in self.robot_routes_by_spawn_id:
                self.robot_routes_by_spawn_id[route.spawn_id].append(route)
            else:
                self.robot_routes_by_spawn_id[route.spawn_id] = [route]

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
        """
        return next(
            filter(
                lambda r: r.goal_id == goal_id and r.spawn_id == spawn_id,
                self.robot_routes,
            ),
            None,
        )

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
        """

        # Check if the maps_folder directory exists
        if not os.path.exists(maps_folder) or not os.path.isdir(maps_folder):
            raise ValueError(f"Map directory '{maps_folder}' does not exist!")

        # Function to load a JSON file
        def load_json(path: str) -> dict:
            """Load json.

            Args:
                path: Filesystem path to the resource.

            Returns:
                dict: Dictionary value.
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


def serialize_map(map_structure: dict) -> MapDefinition:
    """
    Converts a map structure dictionary into a MapDefinition object.

    Parameters
    ----------
    map_structure : dict
        The map structure dictionary.

    Returns
    -------
    MapDefinition
        The MapDefinition object.
    """

    # Extract the x and y margins and calculate the width and height
    (min_x, max_x), (min_y, max_y) = (
        map_structure["x_margin"],
        map_structure["y_margin"],
    )
    width, height = max_x - min_x, max_y - min_y

    # Function to normalize a position
    def norm_pos(pos: Vec2D) -> Vec2D:
        """Norm pos.

        Args:
            pos: Position vector.

        Returns:
            Vec2D: 2D vector (x, y).
        """
        return (pos[0] - min_x, pos[1] - min_y)

    # Normalize the obstacles
    obstacles = [
        Obstacle([norm_pos(p) for p in vertices]) for vertices in map_structure["obstacles"]
    ]

    # Function to normalize a zone
    def norm_zone(rect: Rect) -> Rect:
        """Norm zone.

        Args:
            rect: Rectangle definition.

        Returns:
            Rect: Rectangle definition.
        """
        return (norm_pos(rect[0]), norm_pos(rect[1]), norm_pos(rect[2]))

    # Normalize the zones
    robot_goal_zones = [norm_zone(z) for z in map_structure["robot_goal_zones"]]
    robot_spawn_zones = [norm_zone(z) for z in map_structure["robot_spawn_zones"]]
    ped_goal_zones = [norm_zone(z) for z in map_structure["ped_goal_zones"]]
    ped_spawn_zones = [norm_zone(z) for z in map_structure["ped_spawn_zones"]]
    ped_crowded_zones = [norm_zone(z) for z in map_structure["ped_crowded_zones"]]

    # Normalize the routes
    robot_routes = [
        GlobalRoute(
            o["spawn_id"],
            o["goal_id"],
            [norm_pos(p) for p in o["waypoints"]],
            robot_spawn_zones[o["spawn_id"]],
            robot_goal_zones[o["goal_id"]],
        )
        for o in map_structure["robot_routes"]
    ]
    ped_routes = [
        GlobalRoute(
            o["spawn_id"],
            o["goal_id"],
            [norm_pos(p) for p in o["waypoints"]],
            ped_spawn_zones[o["spawn_id"]],
            ped_goal_zones[o["goal_id"]],
        )
        for o in map_structure["ped_routes"]
    ]

    # Function to reverse a route
    def reverse_route(route: GlobalRoute) -> GlobalRoute:
        """Reverse route.

        Args:
            route: Route definition.

        Returns:
            GlobalRoute: Global route definition used by navigation.
        """
        return GlobalRoute(
            route.goal_id,
            route.spawn_id,
            list(reversed(route.waypoints)),
            route.goal_zone,
            route.spawn_zone,
        )

    # Reverse the robot routes and add them to the list of routes
    rev_robot_routes = [reverse_route(r) for r in robot_routes]
    robot_routes = robot_routes + rev_robot_routes

    # Parse single pedestrians if present
    single_pedestrians = []
    if "single_pedestrians" in map_structure:
        for ped_def in map_structure["single_pedestrians"]:
            ped_id = ped_def["id"]
            start = norm_pos(tuple(ped_def["start"]))
            goal = norm_pos(tuple(ped_def["goal"])) if ped_def.get("goal") else None
            trajectory = (
                [norm_pos(tuple(wp)) for wp in ped_def["trajectory"]]
                if ped_def.get("trajectory")
                else None
            )
            single_pedestrians.append(
                SinglePedestrianDefinition(
                    id=ped_id,
                    start=start,
                    goal=goal,
                    trajectory=trajectory,
                )
            )

    # Define the map bounds
    map_bounds = [
        (0, width, 0, 0),  # bottom
        (0, width, height, height),  # top
        (0, 0, 0, height),  # left
        (width, width, 0, height),
    ]  # right

    # Return the MapDefinition object
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
