from dataclasses import dataclass, field
from math import atan2, ceil, cos, dist, sin

import numpy as np

from robot_sf.common.types import PedGrouping, PedState, Vec2D, Zone, ZoneAssignments
from robot_sf.nav.map_config import GlobalRoute
from robot_sf.ped_npc.ped_behavior import (
    CrowdedZoneBehavior,
    FollowRouteBehavior,
    PedestrianBehavior,
)
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
from robot_sf.ped_npc.ped_zone import sample_zone


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
    """

    peds_per_area_m2: float
    max_group_members: int
    group_member_probs: list[float] = field(default_factory=list)
    initial_speed: float = 0.5
    group_size_decay: float = 0.3
    sidewalk_width: float = 3.0

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


def sample_route(
    route: GlobalRoute,
    num_samples: int,
    sidewalk_width: float,
) -> tuple[list[Vec2D], int]:
    """
    Samples points along a given route within the bounds of a sidewalk.

    Args:
        route: A `GlobalRoute` object representing the route to sample from.
        num_samples: The number of points to sample along the route.
        sidewalk_width: The width of the sidewalk for constraining sample spread.

    Returns:
        A tuple containing:
            - A list of `Vec2D` objects representing the sampled points.
            - The section index of the route where sampling starts (sec_id).

    Raises:
        ValueError: If `num_samples` is not positive.
    """

    # Error handling
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive.")

    # Randomly choose a starting offset along the total length of the route
    sampled_offset = np.random.uniform(0, route.total_length)

    # Find the section index that corresponds to the sampled offset
    sec_id = next(
        iter([i - 1 for i, o in enumerate(route.section_offsets) if o >= sampled_offset]),
        -1,
    )

    # Get start and end points of the chosen section
    start, end = route.sections[sec_id]

    # Define helper functions for vector operations
    def add_vecs(v1, v2):
        return (v1[0] + v2[0], v1[1] + v2[1])

    def sub_vecs(v1, v2):
        return (v1[0] - v2[0], v1[1] - v2[1])

    # Clip function to constrain random spread to the sidewalk width
    def clip_spread(v):
        return np.clip(v, -sidewalk_width / 2, sidewalk_width / 2)

    # Calculate the center point between the start and end of the section
    center = add_vecs(start, sub_vecs(end, start))

    # Define standard deviation for normal distribution based on sidewalk width
    std_dev = sidewalk_width / 4

    # Sample x and y offsets from a normal distribution centered at midpoint
    x_offsets = clip_spread(np.random.normal(center[0], std_dev, (num_samples, 1)))
    y_offsets = clip_spread(np.random.normal(center[1], std_dev, (num_samples, 1)))

    # Create the sampled points by combining offsets with the section's center
    points = np.concatenate((x_offsets, y_offsets), axis=1) + center

    # Return sampled points as a list of Vec2D tuples and the section index
    return [(x, y) for x, y in points], sec_id


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
    zone_areas: list[float] = field(init=False)
    _zone_probs: list[float] = field(init=False)

    def __post_init__(self):
        # Calculate the area for each zone assuming zones are rectangular
        # This uses an external `dist` function to measure distances
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
        return sample_zone(self.zones[zone_id], num_samples), zone_id


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

    def generate(self, num_samples: int) -> tuple[list[Vec2D], int, int]:
        """
        Generates sample points within a randomly selected route.

        Args:
            num_samples: The number of sample points to generate.

        Returns:
            A tuple containing:
                - A list of `Vec2D` objects representing the generated points.
                - The index of the route where the points were generated (route_id).
                - The section id of the route where the points were generated (sec_id).
        """
        # Randomly select a route based on the calculated probabilities
        route_id = np.random.choice(len(self.routes), size=1, p=self._route_probs)[0]

        # Generate sample points using a function `sample_route`
        spawn_pos, sec_id = sample_route(self.routes[route_id], num_samples, self.sidewalk_width)
        return spawn_pos, route_id, sec_id


def populate_ped_routes(
    config: PedSpawnConfig,
    routes: list[GlobalRoute],
) -> tuple[np.ndarray, list[PedGrouping], dict[int, GlobalRoute], list[int]]:
    """
    Populate routes with pedestrian groups according to the configuration.

    Args:
        config: A `PedSpawnConfig` object containing pedestrian spawn specifications.
        routes: A list of `GlobalRoute` objects representing various pathways.

    Returns:
        A tuple consisting of:
            - A numpy array representing the state information for all pedestrians.
            - A list of sets where each set contains the indices of pedestrians in a group.
            - A dictionary mapping group indices to their corresponding `GlobalRoute` objects.
            - A list of initial section indices for each pedestrian group.
    """
    if not routes:
        return np.zeros((0, 6)), [], {}, []
    # Initialize a route points generator with the provided routes and sidewalk width
    proportional_spawn_gen = RoutePointsGenerator(routes, config.sidewalk_width)
    total_num_peds = ceil(proportional_spawn_gen.total_sidewalks_area * config.peds_per_area_m2)
    ped_states, groups = np.zeros((total_num_peds, 6)), []
    num_unassigned_peds = total_num_peds
    # Dictionary to hold assignments of groups to routes
    route_assignments = {}
    # List to track the initial sections for each group
    initial_sections = []

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
) -> tuple[PedState, list[PedGrouping], ZoneAssignments]:
    proportional_spawn_gen = ZonePointsGenerator(crowded_zones)
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
        group_goal = sample_zone(crowded_zones[zone_id], 1)[0]

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
            - List of metadata dicts (one per pedestrian) containing id, goal, trajectory info
    """
    if not single_pedestrians:
        return np.empty((0, 7)), []

    num_peds = len(single_pedestrians)
    ped_states = np.zeros((num_peds, 7))
    metadata = []

    for i, ped in enumerate(single_pedestrians):
        # Position (x, y)
        ped_states[i, 0:2] = ped.start

        # Initial velocity pointing toward goal or first trajectory waypoint
        if ped.goal is not None:
            direction = atan2(ped.goal[1] - ped.start[1], ped.goal[0] - ped.start[0])
            ped_states[i, 2:4] = [initial_speed * cos(direction), initial_speed * sin(direction)]
            ped_states[i, 4:6] = ped.goal
        elif ped.trajectory:
            first_wp = ped.trajectory[0]
            direction = atan2(first_wp[1] - ped.start[1], first_wp[0] - ped.start[0])
            ped_states[i, 2:4] = [initial_speed * cos(direction), initial_speed * sin(direction)]
            # For trajectory-based, goal is first waypoint initially
            ped_states[i, 4:6] = first_wp
        else:
            # Static pedestrian (no goal, no trajectory)
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
            }
        )

    return ped_states, metadata


def populate_simulation(
    tau: float,
    spawn_config: PedSpawnConfig,
    ped_routes: list[GlobalRoute],
    ped_crowded_zones: list[Zone],
    single_pedestrians: list | None = None,  # list[SinglePedestrianDefinition] - optional
) -> tuple[PedestrianStates, PedestrianGroupings, list[PedestrianBehavior]]:
    crowd_ped_states_np, crowd_groups, zone_assignments = populate_crowded_zones(
        spawn_config,
        ped_crowded_zones,
    )
    route_ped_states_np, route_groups, route_assignments, initial_sections = populate_ped_routes(
        spawn_config,
        ped_routes,
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

    # Single pedestrians are individual (no groups), so no group entries needed

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
    route_behavior = FollowRouteBehavior(route_groupings, route_assignments, initial_sections)
    ped_behaviors: list[PedestrianBehavior] = [crowd_behavior, route_behavior]
    return pysf_state, groups, ped_behaviors
