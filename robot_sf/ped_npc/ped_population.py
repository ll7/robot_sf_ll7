"""Pedestrian population generation and simulation setup.

This module provides utilities for spawning and managing pedestrian agents in simulation
environments. It supports multiple spawning strategies including:
  - Route-following pedestrians (along predefined paths)
  - Crowded zone pedestrians (in open areas)
  - Single pedestrians (individually controlled with explicit start/goal/trajectory)

Key Components:
    - PedSpawnConfig: Configuration for pedestrian density, group sizes, and movement.
    - sample_route(): Sample positions along a route respecting sidewalk constraints.
    - populate_ped_routes(): Create pedestrian groups following predefined routes.
    - populate_crowded_zones(): Create pedestrian groups in open areas.
    - populate_single_pedestrians(): Create individually controlled pedestrians.
    - populate_simulation(): Orchestrate all spawning strategies and return initialized state.

Obstacle Handling:
    All spawning functions support optional obstacle avoidance using Shapely geometry.
    Prepared geometries are cached for efficiency in containment checks.

Example:
    >>> config = PedSpawnConfig(peds_per_area_m2=0.05, max_group_members=3)
    >>> pysf_state, groups, behaviors = populate_simulation(
    ...     tau=0.5,
    ...     spawn_config=config,
    ...     ped_routes=routes,
    ...     ped_crowded_zones=zones,
    ...     obstacle_polygons=obstacles,
    ...     single_pedestrians=special_peds,
    ... )
"""

from dataclasses import dataclass, field
from math import atan2, ceil, cos, dist, sin

import numpy as np
from shapely.geometry import Point as _ShapelyPoint
from shapely.prepared import PreparedGeometry

from robot_sf.common.types import PedGrouping, PedState, Vec2D, Zone, ZoneAssignments
from robot_sf.nav.map_config import GlobalRoute
from robot_sf.ped_npc.ped_behavior import (
    CrowdedZoneBehavior,
    FollowRouteBehavior,
    PedestrianBehavior,
    SinglePedestrianBehavior,
)
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
from robot_sf.ped_npc.ped_zone import prepare_obstacle_polygons, sample_zone


def _route_point_at_offset(route: GlobalRoute, offset: float) -> tuple[Vec2D, int]:
    """Return a point and section index along a route for a given offset."""
    if not route.sections:
        raise ValueError("Route must contain at least two waypoints to sample along its length.")
    remaining = float(np.clip(offset, 0.0, route.total_length))
    for idx, (start, end) in enumerate(route.sections):
        segment_length = dist(start, end)
        if remaining <= segment_length or idx == len(route.sections) - 1:
            if segment_length > 0:
                ratio = remaining / segment_length
                point = (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                )
            else:
                point = start
            return point, idx
        remaining -= segment_length
    return route.sections[-1][1], len(route.sections) - 1


def _sample_points_near_anchor(
    anchor: Vec2D,
    num_samples: int,
    sidewalk_width: float,
    rng_local,
    prepared_obstacles: list[PreparedGeometry],
) -> list[Vec2D]:
    """Sample points near an anchor while avoiding obstacles.

    Returns:
        list[Vec2D]: Sampled points near the anchor.
    """
    samples: list[Vec2D] = []
    attempts = 0
    max_attempts = num_samples * 50

    def clip_spread(v):
        return np.clip(v, -sidewalk_width / 2, sidewalk_width / 2)

    while len(samples) < num_samples and attempts < max_attempts:
        progress = attempts / max_attempts if max_attempts else 0.0
        scale = max(0.2, 1.0 - progress)
        std_dev = (sidewalk_width / 4) * scale
        x_offset = float(clip_spread(rng_local.normal(0.0, std_dev, (1,))[0]))
        y_offset = float(clip_spread(rng_local.normal(0.0, std_dev, (1,))[0]))
        pt = (anchor[0] + x_offset, anchor[1] + y_offset)
        attempts += 1
        if prepared_obstacles and _point_in_any_obstacle(pt, prepared_obstacles):
            continue
        samples.append(pt)

    return samples


@dataclass
class PedSpawnConfig:
    """Configuration for pedestrian spawning in a simulation environment.

    Attributes:
        peds_per_area_m2: The density of pedestrians per square meter area.
        max_group_members: Maximum number of members allowed in a pedestrian group.
        group_member_probs: A list representing the probability distribution for the
                            number of members in a group. Each index corresponds to
                            group size - 1, with its value being the respective probability.
        initial_speed: The initial walking speed for pedestrians.
        group_size_decay: The rate at which the probability of larger group sizes decays.
        sidewalk_width: The width of the sidewalk where pedestrians can spawn.
        route_spawn_distribution: How route pedestrians are placed along a route.
            "cluster" keeps groups near a shared offset (default behavior).
            "spread" spaces groups along the route length.
        route_spawn_jitter_frac: Fraction of spacing used as random jitter when spreading.
        route_spawn_seed: Optional RNG seed for deterministic spawn placement/jitter.
    """

    peds_per_area_m2: float
    max_group_members: int
    group_member_probs: list[float] = field(default_factory=list)
    initial_speed: float = 0.5
    group_size_decay: float = 0.3
    sidewalk_width: float = 3.0
    route_spawn_distribution: str = "cluster"
    route_spawn_jitter_frac: float = 0.0
    route_spawn_seed: int | None = None

    def __post_init__(self):
        """
        Ensures that `group_member_probs` has exactly `max_group_members`
        elements by creating a power-law distributed list if needed.
        """
        if len(self.group_member_probs) != self.max_group_members:
            # initialize group size probabilities decaying by power law
            power_dist = [self.group_size_decay**i for i in range(self.max_group_members)]
            # Normalize the distribution so that the sum equals 1
            self.group_member_probs = [p / sum(power_dist) for p in power_dist]
        if self.route_spawn_distribution not in {"cluster", "spread"}:
            raise ValueError(
                "route_spawn_distribution must be 'cluster' or 'spread' "
                f"(got {self.route_spawn_distribution!r})"
            )
        if self.route_spawn_jitter_frac < 0:
            raise ValueError("route_spawn_jitter_frac must be >= 0")


def sample_route(
    route: GlobalRoute,
    num_samples: int,
    sidewalk_width: float,
    obstacle_polygons: list[list[Vec2D]] | list[PreparedGeometry] | None = None,
    *,
    offset: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[Vec2D], int]:
    """
    Samples points along a given route within the bounds of a sidewalk.
    Offsets anchor the sample along the route length before lateral spread is applied.

    Args:
        route: A `GlobalRoute` object representing the route to sample from.
        num_samples: The number of points to sample along the route.
        sidewalk_width: The width of the sidewalk for constraining sample spread.
        obstacle_polygons: Optional prepared or raw obstacle polygons to avoid spawning inside.
        offset: Optional fixed offset along the route length to anchor sampling.
        rng: Optional RNG for deterministic sampling; defaults to NumPy global RNG.

    Returns:
        A tuple containing:
            - A list of `Vec2D` objects representing the sampled points.
            - The section index of the route containing the sampled offset (sec_id).

    Raises:
        ValueError: If `num_samples` is not positive.
    """

    # Error handling
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive.")

    # Randomly choose a starting offset along the total length of the route
    rng_local = rng if rng is not None else np.random
    if offset is None:
        sampled_offset = float(rng_local.uniform(0, route.total_length))
    else:
        sampled_offset = float(np.clip(offset, 0.0, route.total_length))

    prepared_obstacles = prepare_obstacle_polygons(obstacle_polygons or [])
    anchor_attempts = 0
    max_anchor_attempts = 5
    base_point, sec_id = _route_point_at_offset(route, sampled_offset)

    while anchor_attempts < max_anchor_attempts:
        if prepared_obstacles and _point_in_any_obstacle(base_point, prepared_obstacles):
            sampled_offset = float(rng_local.uniform(0, route.total_length))
            base_point, sec_id = _route_point_at_offset(route, sampled_offset)
            anchor_attempts += 1
            continue

        samples = _sample_points_near_anchor(
            base_point,
            num_samples,
            sidewalk_width,
            rng_local,
            prepared_obstacles,
        )
        if len(samples) == num_samples:
            return samples, sec_id

        sampled_offset = float(rng_local.uniform(0, route.total_length))
        base_point, sec_id = _route_point_at_offset(route, sampled_offset)
        anchor_attempts += 1

    raise RuntimeError(
        f"Failed to sample {num_samples} route points after {max_anchor_attempts} anchor tries "
        f"(sampled_offset={sampled_offset:.3f}, sec_id={sec_id}) without violating obstacle "
        "constraints. Consider adjusting the route, sidewalk width, or obstacle geometry."
    )


def _point_in_any_obstacle(point: Vec2D, obstacles: list[PreparedGeometry]) -> bool:
    """Return True when point lies inside any prepared obstacle polygon."""
    pt = _ShapelyPoint(point)
    return any(poly.contains(pt) for poly in obstacles)


@dataclass
class ZonePointsGenerator:
    """
    A generator for creating points within specified zones.

    Attributes:
        zones: A list of `Zone` objects that represent different areas.

    Calculated attributes through __post_init__:
        zone_areas: Computed area of each zone based on vertices.
        _zone_probs: Normalized probabilities of choosing each zone.
    """

    zones: list[Zone]
    obstacle_polygons: list[list[Vec2D]] | None = None
    zone_areas: list[float] = field(init=False)
    _zone_probs: list[float] = field(init=False)

    def __post_init__(self):
        """Calculate zone areas and normalized selection probabilities.

        Computes the area of each zone (assuming rectangular geometry) by
        measuring distances between consecutive vertices. Normalizes areas to
        create probability weights for area-proportional zone selection.
        """
        self.zone_areas = [dist(p1, p2) * dist(p2, p3) for p1, p2, p3 in self.zones]

        # Sum the areas to use for normalizing probabilities
        total_area = sum(self.zone_areas)

        # Calculate the probability for each zone based on its area
        self._zone_probs = [area / total_area for area in self.zone_areas]
        # Proportional distribution by zone area is considered

    def generate(self, num_samples: int) -> tuple[list[Vec2D], int]:
        """
        Generates sample points within a randomly selected zone.

        Args:
            num_samples: The number of sample points to generate.

        Returns:
            A tuple containing:
                - A list of `Vec2D` objects representing the generated points.
                - The index of the zone where the points were generated (zone_id).
        """

        # Randomly select a zone based on the calculated probabilities
        zone_id = np.random.choice(len(self.zones), size=1, p=self._zone_probs)[0]

        # Generate sample points using a function `sample_zone`
        return (
            sample_zone(
                self.zones[zone_id],
                num_samples,
                obstacle_polygons=self.obstacle_polygons,
            ),
            zone_id,
        )


@dataclass
class RoutePointsGenerator:
    """
    A generator for creating points within specified routes with given sidewalk widths.

    Attributes:
        routes: A list of `GlobalRoute` objects representing different paths.
        sidewalk_width: The width of the sidewalks along the routes.

    Calculated attributes through __post_init__:
        _route_probs: Normalized probabilities of choosing each route based on length.
    """

    routes: list[GlobalRoute]
    sidewalk_width: float
    obstacle_polygons: list[list[Vec2D]] | list[PreparedGeometry] | None = None
    _route_probs: list[float] = field(init=False)

    def __post_init__(self):
        """
        Initialize calculated fields and compute route probabilities.
        """
        if not self.routes:
            raise ValueError("RoutePointsGenerator requires at least one route.")
        # Calculate the probability for each route based on its length.
        # It assumes that the area per route is approximated by multiplying
        # the total length of the route with the sidewalk width.
        # info: distribute proportionally by zone area; area ~ route length * sidewalk width
        lengths = [r.total_length for r in self.routes]
        total_len = sum(lengths)
        if total_len > 0:
            self._route_probs = [length / total_len for length in lengths]
        else:
            # Fallback to uniform sampling when all routes have zero length to avoid ZeroDivisionError.
            uniform = 1.0 / len(lengths)
            self._route_probs = [uniform for _ in lengths]

    @property
    def total_length(self) -> float:
        """
        Calculate the total length of all routes.

        Returns:
            The sum of lengths of all routes.
        """
        return sum(r.total_length for r in self.routes)

    @property
    def total_sidewalks_area(self) -> float:
        """
        Calculate the total sidewalk area for all routes.

        Returns:
            The total area covered by the sidewalks alongside the routes.
        """
        return self.total_length * self.sidewalk_width

    def generate(
        self,
        num_samples: int,
        *,
        rng: np.random.Generator | None = None,
        offset: float | None = None,
    ) -> tuple[list[Vec2D], int, int]:
        """
        Generates sample points within a randomly selected route.

        Args:
            num_samples: The number of sample points to generate.
            rng: Optional RNG for deterministic sampling; defaults to NumPy global RNG.
            offset: Optional fixed offset along the route length to anchor sampling.

        Returns:
            A tuple containing:
                - A list of `Vec2D` objects representing the generated points.
                - The index of the route where the points were generated (route_id).
                - The section id of the route where the points were generated (sec_id).
        """
        # Randomly select a route based on the calculated probabilities
        rng_local = rng if rng is not None else np.random
        route_id = rng_local.choice(len(self.routes), size=1, p=self._route_probs)[0]

        # Generate sample points using a function `sample_route`
        spawn_pos, sec_id = sample_route(
            self.routes[route_id],
            num_samples,
            self.sidewalk_width,
            obstacle_polygons=self.obstacle_polygons,
            offset=offset,
            rng=rng_local if isinstance(rng_local, np.random.Generator) else None,
        )
        return spawn_pos, route_id, sec_id


def populate_ped_routes(
    config: PedSpawnConfig,
    routes: list[GlobalRoute],
    obstacle_polygons: list[list[Vec2D]] | list[PreparedGeometry] | None = None,
) -> tuple[np.ndarray, list[PedGrouping], dict[int, GlobalRoute], list[int]]:
    """
    Populate routes with pedestrian groups according to the configuration.

    Args:
        config: A `PedSpawnConfig` object containing pedestrian spawn specifications.
        routes: A list of `GlobalRoute` objects representing various pathways.
        obstacle_polygons: Optional obstacle polygons used to avoid spawning inside obstacles.

    Returns:
        A tuple consisting of:
            - A numpy array representing the state information for all pedestrians.
            - A list of sets where each set contains the indices of pedestrians in a group.
            - A dictionary mapping group indices to their corresponding `GlobalRoute` objects.
            - A list of initial section indices for each pedestrian group.

    Notes:
        When ``config.route_spawn_distribution`` is "spread", groups are assigned to
        routes proportionally by length and spaced along each route with optional jitter.
    """
    if not routes:
        return np.zeros((0, 6)), [], {}, []
    # Initialize a route points generator with the provided routes and sidewalk width
    proportional_spawn_gen = RoutePointsGenerator(
        routes,
        config.sidewalk_width,
        obstacle_polygons=obstacle_polygons,
    )
    total_num_peds = ceil(proportional_spawn_gen.total_sidewalks_area * config.peds_per_area_m2)
    ped_states, groups = np.zeros((total_num_peds, 6)), []
    num_unassigned_peds = total_num_peds
    # Dictionary to hold assignments of groups to routes
    route_assignments = {}
    # List to track the initial sections for each group
    initial_sections = []

    if config.route_spawn_distribution == "spread" and total_num_peds > 0:
        rng = np.random.default_rng(config.route_spawn_seed)
        probs = config.group_member_probs
        group_sizes: list[int] = []
        while num_unassigned_peds > 0:
            num_peds_in_group = int(rng.choice(len(probs), p=probs) + 1)
            num_peds_in_group = min(num_peds_in_group, num_unassigned_peds)
            group_sizes.append(num_peds_in_group)
            num_unassigned_peds -= num_peds_in_group

        group_count = len(group_sizes)
        route_ids = rng.choice(
            len(routes),
            size=group_count,
            p=proportional_spawn_gen._route_probs,
        )
        route_to_groups: dict[int, list[int]] = {}
        for group_idx, route_id in enumerate(route_ids):
            route_to_groups.setdefault(int(route_id), []).append(group_idx)

        ped_offset = 0
        for route_id in sorted(route_to_groups):
            group_indices = route_to_groups[route_id]
            route = routes[route_id]
            ordered_indices = sorted(group_indices)
            spacing = route.total_length / max(len(ordered_indices), 1)
            jitter_range = spacing * float(config.route_spawn_jitter_frac)
            for order, group_idx in enumerate(ordered_indices):
                base_offset = (order + 0.5) * spacing
                if jitter_range > 0:
                    base_offset += float(rng.uniform(-jitter_range, jitter_range))
                base_offset = float(np.clip(base_offset, 0.0, route.total_length))
                num_peds_in_group = group_sizes[group_idx]
                ped_ids = list(range(ped_offset, ped_offset + num_peds_in_group))
                ped_offset += num_peds_in_group
                groups.append(set(ped_ids))

                spawn_points, sec_id = sample_route(
                    route,
                    num_peds_in_group,
                    config.sidewalk_width,
                    obstacle_polygons=obstacle_polygons,
                    offset=base_offset,
                    rng=rng,
                )
                group_goal = route.sections[sec_id][1]
                initial_sections.append(sec_id)
                route_assignments[len(groups) - 1] = route

                centroid = np.mean(spawn_points, axis=0)
                rot = atan2(group_goal[1] - centroid[1], group_goal[0] - centroid[0])
                velocity = np.array([cos(rot), sin(rot)]) * config.initial_speed
                ped_states[ped_ids, 0:2] = spawn_points
                ped_states[ped_ids, 2:4] = velocity
                ped_states[ped_ids, 4:6] = group_goal
    else:
        while num_unassigned_peds > 0:
            # Determine number of members in next group based on configured probabilities
            probs = config.group_member_probs
            num_peds_in_group = np.random.choice(len(probs), p=probs) + 1
            num_peds_in_group = min(num_peds_in_group, num_unassigned_peds)
            # Calculate range of IDs for newly assigned pedestrians
            num_assigned_peds = total_num_peds - num_unassigned_peds
            ped_ids = list(range(num_assigned_peds, total_num_peds))[:num_peds_in_group]
            # Add set of new pedestrian IDs to groups list
            groups.append(set(ped_ids))

            # spawn all group members along a uniformly sampled route with respect to the route's length
            # Generate spawn points for current group, route ID, and section ID
            spawn_points, route_id, sec_id = proportional_spawn_gen.generate(num_peds_in_group)
            # Determine group's goal point from the selected route and section
            group_goal = routes[route_id].sections[sec_id][1]
            # Record initial section ID for this group
            initial_sections.append(sec_id)
            # Assign current route to the latest group
            route_assignments[len(groups) - 1] = routes[route_id]

            centroid = np.mean(spawn_points, axis=0)
            rot = atan2(group_goal[1] - centroid[1], group_goal[0] - centroid[0])
            velocity = np.array([cos(rot), sin(rot)]) * config.initial_speed
            ped_states[ped_ids, 0:2] = spawn_points
            ped_states[ped_ids, 2:4] = velocity
            ped_states[ped_ids, 4:6] = group_goal

            num_unassigned_peds -= num_peds_in_group

    return ped_states, groups, route_assignments, initial_sections


def populate_crowded_zones(
    config: PedSpawnConfig,
    crowded_zones: list[Zone],
    obstacle_polygons: list[list[Vec2D]] | list[PreparedGeometry] | None = None,
) -> tuple[PedState, list[PedGrouping], ZoneAssignments]:
    """Spawn pedestrian groups within crowded zones (open areas).

    Creates pedestrian groups distributed across specified zones according to
    area-weighted probabilities. Each group spawns at a random location within
    a selected zone and receives a goal position also within that zone to keep
    groups localized. Respects obstacle boundaries during spawning.

    Args:
        config: Pedestrian spawn configuration (density, group sizes, speed).
        crowded_zones: List of Zone geometries where pedestrians spawn.
        obstacle_polygons: Optional obstacle geometries to avoid during spawning.
            Can be raw coordinate lists or Shapely PreparedGeometry objects.

    Returns:
        Tuple of (pedestrian_states, groups, zone_assignments):
            - pedestrian_states: NumPy array (N, 6) with positions, velocities, goals.
            - groups: List of sets containing pedestrian IDs in each group.
            - zone_assignments: Dict mapping pedestrian ID to zone index.
    """
    proportional_spawn_gen = ZonePointsGenerator(crowded_zones, obstacle_polygons=obstacle_polygons)
    total_num_peds = ceil(sum(proportional_spawn_gen.zone_areas) * config.peds_per_area_m2)
    ped_states, groups = np.zeros((total_num_peds, 6)), []
    num_unassigned_peds = total_num_peds
    zone_assignments = {}

    while num_unassigned_peds > 0:
        probs = config.group_member_probs
        num_peds_in_group = np.random.choice(len(probs), p=probs) + 1
        num_peds_in_group = min(num_peds_in_group, num_unassigned_peds)
        num_assigned_peds = total_num_peds - num_unassigned_peds
        ped_ids = list(range(num_assigned_peds, total_num_peds))[:num_peds_in_group]
        groups.append(set(ped_ids))

        # spawn all group members in the same randomly sampled zone and also
        # keep them within that zone by picking the group's goal accordingly
        spawn_points, zone_id = proportional_spawn_gen.generate(num_peds_in_group)
        group_goal = sample_zone(crowded_zones[zone_id], 1, obstacle_polygons=obstacle_polygons)[0]

        centroid = np.mean(spawn_points, axis=0)
        rot = atan2(group_goal[1] - centroid[1], group_goal[0] - centroid[0])
        velocity = np.array([cos(rot), sin(rot)]) * config.initial_speed
        ped_states[ped_ids, 0:2] = spawn_points
        ped_states[ped_ids, 2:4] = velocity
        ped_states[ped_ids, 4:6] = group_goal
        for pid in ped_ids:
            zone_assignments[pid] = zone_id

        num_unassigned_peds -= num_peds_in_group

    return ped_states, groups, zone_assignments


def populate_single_pedestrians(
    single_pedestrians: list,  # list[SinglePedestrianDefinition] - avoid import cycle
    initial_speed: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """
    Populate single pedestrians from SinglePedestrianDefinition objects.

    Args:
        single_pedestrians: List of SinglePedestrianDefinition objects
        initial_speed: Initial walking speed (default: 0.5 m/s)

    Returns:
        tuple[np.ndarray, list[dict]]:
            - NumPy array of pedestrian states (Nx7): [x, y, vx, vy, gx, gy, tau]
            - List of metadata dicts (one per pedestrian) containing id, goal, trajectory info,
              and optional per-ped speed, wait rules, role tags, and note fields
    """
    if not single_pedestrians:
        return np.empty((0, 7)), []

    num_peds = len(single_pedestrians)
    ped_states = np.zeros((num_peds, 7))
    metadata = []

    for i, ped in enumerate(single_pedestrians):
        ped_speed = ped.speed_m_s if ped.speed_m_s is not None else initial_speed
        role = ped.role
        # Position (x, y)
        ped_states[i, 0:2] = ped.start

        # Initial velocity pointing toward goal or first trajectory waypoint
        if ped.goal is not None:
            direction = atan2(ped.goal[1] - ped.start[1], ped.goal[0] - ped.start[0])
            ped_states[i, 2:4] = [ped_speed * cos(direction), ped_speed * sin(direction)]
            ped_states[i, 4:6] = ped.goal
        elif ped.trajectory:
            first_wp = ped.trajectory[0]
            direction = atan2(first_wp[1] - ped.start[1], first_wp[0] - ped.start[0])
            ped_states[i, 2:4] = [ped_speed * cos(direction), ped_speed * sin(direction)]
            # For trajectory-based, goal is first waypoint initially
            ped_states[i, 4:6] = first_wp
        else:
            # Static pedestrian (no goal, no trajectory)
            if role in {"follow", "lead", "accompany", "join", "leave"}:
                ped_states[i, 2:4] = [ped_speed, 0]
            else:
                ped_states[i, 2:4] = [0, 0]
            ped_states[i, 4:6] = ped.start  # Goal equals start for static peds

        # Tau (relaxation time) - use default 0.5 seconds
        ped_states[i, 6] = 0.5

        # Store metadata
        metadata.append(
            {
                "id": ped.id,
                "has_goal": ped.goal is not None,
                "has_trajectory": ped.trajectory is not None and len(ped.trajectory) > 0,
                "trajectory": ped.trajectory if ped.trajectory else [],
                "current_waypoint_index": 0,
                "speed_m_s": ped_speed,
                "note": ped.note,
                "wait_at": ped.wait_at or [],
                "role": ped.role,
                "role_target_id": ped.role_target_id,
                "role_offset": ped.role_offset,
            }
        )

    return ped_states, metadata


def populate_simulation(
    tau: float,
    spawn_config: PedSpawnConfig,
    ped_routes: list[GlobalRoute],
    ped_crowded_zones: list[Zone],
    obstacle_polygons: list[list[Vec2D]] | list[PreparedGeometry] | None = None,
    single_pedestrians: list | None = None,  # list[SinglePedestrianDefinition] - optional
    time_step_s: float = 0.1,
    single_ped_goal_threshold: float | None = None,
) -> tuple[PedestrianStates, PedestrianGroupings, list[PedestrianBehavior]]:
    """Orchestrate complete pedestrian population initialization for simulation.

    Combines three independent spawning strategies:
      1. Route followers: Groups traveling along predefined paths.
      2. Crowded zone pedestrians: Groups in open areas with local goals.
      3. Single pedestrians: Individually controlled agents with explicit behavior.

    All pedestrian states are merged into a unified array with consistent tau
    (relaxation time) values. Group memberships and behavioral controllers are
    created and returned for use in the physics simulation.

    Args:
        tau: Relaxation time (seconds) for all pedestrians in the simulation.
        spawn_config: Configuration for density, group sizes, and initial speed.
        ped_routes: GlobalRoute objects defining paths for route-following pedestrians.
        ped_crowded_zones: Zone geometries for crowded zone pedestrians.
        obstacle_polygons: Optional obstacle geometries to avoid during spawning.
            Can be raw coordinate lists or Shapely PreparedGeometry objects.
        single_pedestrians: Optional list of SinglePedestrianDefinition objects
            for individually controlled pedestrians with explicit goals/trajectories.
        time_step_s: Simulation step time in seconds (used for wait behavior).
        single_ped_goal_threshold: Optional distance threshold for single-ped waypoint arrival.

    Returns:
        Tuple of (pysf_state, groups, ped_behaviors):
            - pysf_state: PedestrianStates view of all pedestrians for physics engine.
            - groups: PedestrianGroupings tracking group memberships.
            - ped_behaviors: List of PedestrianBehavior controllers for crowd dynamics
              (includes CrowdedZoneBehavior, FollowRouteBehavior, and SinglePedestrianBehavior).

    Example:
        >>> pysf_state, groups, behaviors = populate_simulation(
        ...     tau=0.5,
        ...     spawn_config=config,
        ...     ped_routes=routes,
        ...     ped_crowded_zones=zones,
        ... )
        >>> # Use pysf_state and behaviors in physics simulation
    """
    prepared_obstacles = prepare_obstacle_polygons(obstacle_polygons or [])

    crowd_ped_states_np, crowd_groups, zone_assignments = populate_crowded_zones(
        spawn_config,
        ped_crowded_zones,
        obstacle_polygons=prepared_obstacles,
    )
    route_ped_states_np, route_groups, route_assignments, initial_sections = populate_ped_routes(
        spawn_config,
        ped_routes,
        obstacle_polygons=prepared_obstacles,
    )

    # Populate single pedestrians if provided
    if single_pedestrians:
        single_ped_states_np, _single_ped_metadata = populate_single_pedestrians(
            single_pedestrians,
            spawn_config.initial_speed,
        )
    else:
        single_ped_states_np = np.empty((0, 7))

    # Combine all pedestrian states: crowd + route + single
    combined_ped_states_np = np.concatenate(
        (crowd_ped_states_np, route_ped_states_np, single_ped_states_np[:, :6]),
    )
    taus = np.full((combined_ped_states_np.shape[0]), tau)
    ped_states = np.concatenate((combined_ped_states_np, np.expand_dims(taus, -1)), axis=-1)

    # Calculate ID offsets for each pedestrian category
    route_offset = crowd_ped_states_np.shape[0]
    single_offset = route_offset + route_ped_states_np.shape[0]

    # Adjust group IDs for routes
    combined_groups = crowd_groups + [{id + route_offset for id in peds} for peds in route_groups]

    # Single pedestrians start as single-member groups for optional join/leave behaviors.

    # Create pedestrian state views
    pysf_state = PedestrianStates(lambda: ped_states)
    crowd_pysf_state = PedestrianStates(lambda: ped_states[:route_offset])
    route_pysf_state = PedestrianStates(lambda: ped_states[route_offset:single_offset])

    groups = PedestrianGroupings(pysf_state)
    for ped_ids in combined_groups:
        groups.new_group(ped_ids)
    crowd_groupings = PedestrianGroupings(crowd_pysf_state)
    for ped_ids in crowd_groups:
        crowd_groupings.new_group(ped_ids)
    route_groupings = PedestrianGroupings(route_pysf_state)
    for ped_ids in route_groups:
        route_groupings.new_group(ped_ids)

    crowd_behavior = CrowdedZoneBehavior(crowd_groupings, zone_assignments, ped_crowded_zones)
    route_behavior = FollowRouteBehavior(
        route_groupings,
        route_assignments,
        initial_sections,
        obstacle_polygons=prepared_obstacles,
    )
    ped_behaviors: list[PedestrianBehavior] = [crowd_behavior, route_behavior]

    if single_pedestrians:
        for ped_id in range(single_offset, single_offset + len(single_pedestrians)):
            groups.new_group({ped_id})
        ped_behaviors.append(
            SinglePedestrianBehavior(
                pysf_state,
                groups,
                single_pedestrians,
                single_offset,
                time_step_s=time_step_s,
                goal_proximity_threshold=single_ped_goal_threshold,
            )
        )
    return pysf_state, groups, ped_behaviors
