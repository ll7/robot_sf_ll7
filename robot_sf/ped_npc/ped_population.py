from math import dist, atan2, sin, cos, ceil
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Dict
import numpy as np

from robot_sf.nav.map_config import GlobalRoute
from robot_sf.ped_npc.ped_grouping import PedestrianStates, PedestrianGroupings
from robot_sf.ped_npc.ped_behavior import \
    PedestrianBehavior, CrowdedZoneBehavior, FollowRouteBehavior
from robot_sf.ped_npc.ped_zone import sample_zone

PedState = np.ndarray
PedGrouping = Set[int]
Vec2D = Tuple[float, float]
Zone = Tuple[Vec2D, Vec2D, Vec2D] # rect ABC with sides |A B|, |B C| and diagonal |A C|
ZoneAssignments = Dict[int, int]


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
    group_member_probs: List[float] = field(default_factory=list)
    initial_speed: float = 0.5
    group_size_decay: float = 0.3
    sidewalk_width: float = 3.0

    def __post_init__(self):
        """
        Ensures that `group_member_probs` has exactly `max_group_members`
        elements by creating a power-law distributed list if needed.
        """
        if not len(self.group_member_probs) == self.max_group_members:
            # initialize group size probabilities decaying by power law
            power_dist = [self.group_size_decay**i for i in range(self.max_group_members)]
            # Normalize the distribution so that the sum equals 1
            self.group_member_probs = [p / sum(power_dist) for p in power_dist]



def sample_route(
        route: GlobalRoute,
        num_samples: int,
        sidewalk_width: float
    ) -> Tuple[List[Vec2D], int]:
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
        iter([i - 1 for i, o in enumerate(route.section_offsets) if o >= sampled_offset]), -1)

    # Get start and end points of the chosen section
    start, end = route.sections[sec_id]

    # Define helper functions for vector operations
    add_vecs = lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1])
    sub_vecs = lambda v1, v2: (v1[0] - v2[0], v1[1] - v2[1])

    # Clip function to constrain random spread to the sidewalk width
    clip_spread = lambda v: np.clip(v, -sidewalk_width / 2, sidewalk_width / 2)

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
    zones: List[Zone]
    zone_areas: List[float] = field(init=False)
    _zone_probs: List[float] = field(init=False)

    def __post_init__(self):
        self.zone_areas = [dist(p1, p2) * dist(p2, p3) for p1, p2, p3 in self.zones]
        total_area = sum(self.zone_areas)
        self._zone_probs = [area / total_area for area in self.zone_areas]
        # info: distribute proportionally by zone area

    def generate(self, num_samples: int) -> Tuple[List[Vec2D], int]:
        zone_id = np.random.choice(len(self.zones), size=1, p=self._zone_probs)[0]
        return sample_zone(self.zones[zone_id], num_samples), zone_id


@dataclass
class RoutePointsGenerator:
    routes: List[GlobalRoute]
    sidewalk_width: float
    _route_probs: List[float] = field(init=False)

    def __post_init__(self):
        # info: distribute proportionally by zone area; area ~ route length * sidewalk width
        self._zone_probs = [r.total_length / self.total_length for r in self.routes]

    @property
    def total_length(self) -> float:
        return sum([r.total_length for r in self.routes])

    @property
    def total_sidewalks_area(self) -> float:
        return self.total_length * self.sidewalk_width

    def generate(self, num_samples: int) -> Tuple[List[Vec2D], int, int]:
        route_id = np.random.choice(len(self.routes), size=1, p=self._zone_probs)[0]
        spawn_pos, sec_id = sample_route(self.routes[route_id], num_samples, self.sidewalk_width)
        return spawn_pos, route_id, sec_id


def populate_ped_routes(config: PedSpawnConfig, routes: List[GlobalRoute]) \
        -> Tuple[np.ndarray, List[PedGrouping], Dict[int, GlobalRoute], List[int]]:

    proportional_spawn_gen = RoutePointsGenerator(routes, config.sidewalk_width)
    total_num_peds = ceil(proportional_spawn_gen.total_sidewalks_area * config.peds_per_area_m2)
    ped_states, groups = np.zeros((total_num_peds, 6)), []
    num_unassigned_peds = total_num_peds
    route_assignments = dict()
    initial_sections = []

    while num_unassigned_peds > 0:
        probs = config.group_member_probs
        num_peds_in_group = np.random.choice(len(probs), p=probs) + 1
        num_peds_in_group = min(num_peds_in_group, num_unassigned_peds)
        num_assigned_peds = total_num_peds - num_unassigned_peds
        ped_ids = list(range(num_assigned_peds, total_num_peds))[:num_peds_in_group]
        groups.append(set(ped_ids))

        # spawn all group members along a uniformly sampled route with respect to the route's length
        spawn_points, route_id, sec_id = proportional_spawn_gen.generate(num_peds_in_group)
        group_goal = routes[route_id].sections[sec_id][1]
        initial_sections.append(sec_id)
        route_assignments[len(groups) - 1] = routes[route_id]

        centroid = np.mean(spawn_points, axis=0)
        rot = atan2(group_goal[1] - centroid[1], group_goal[0] - centroid[0])
        velocity = np.array([cos(rot), sin(rot)]) * config.initial_speed
        ped_states[ped_ids, 0:2] = spawn_points
        ped_states[ped_ids, 2:4] = velocity
        ped_states[ped_ids, 4:6] = group_goal

        num_unassigned_peds -= num_peds_in_group

    return ped_states, groups, route_assignments, initial_sections


def populate_crowded_zones(config: PedSpawnConfig, crowded_zones: List[Zone]) \
        -> Tuple[PedState, List[PedGrouping], ZoneAssignments]:

    proportional_spawn_gen = ZonePointsGenerator(crowded_zones)
    total_num_peds = ceil(sum(proportional_spawn_gen.zone_areas) * config.peds_per_area_m2)
    ped_states, groups = np.zeros((total_num_peds, 6)), []
    num_unassigned_peds = total_num_peds
    zone_assignments = dict()

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


def populate_simulation(
        tau: float, spawn_config: PedSpawnConfig,
        ped_routes: List[GlobalRoute], ped_crowded_zones: List[Zone]
    ) -> Tuple[PedestrianStates, PedestrianGroupings, List[PedestrianBehavior]]:

    crowd_ped_states_np, crowd_groups, zone_assignments = \
        populate_crowded_zones(spawn_config, ped_crowded_zones)
    route_ped_states_np, route_groups, route_assignments, initial_sections = \
        populate_ped_routes(spawn_config, ped_routes)

    combined_ped_states_np = np.concatenate((crowd_ped_states_np, route_ped_states_np))
    taus = np.full((combined_ped_states_np.shape[0]), tau)
    ped_states = np.concatenate((combined_ped_states_np, np.expand_dims(taus, -1)), axis=-1)
    id_offset = crowd_ped_states_np.shape[0]
    combined_groups = crowd_groups + [{id + id_offset for id in peds} for peds in route_groups]

    pysf_state = PedestrianStates(lambda: ped_states)
    crowd_pysf_state = PedestrianStates(lambda: ped_states[:id_offset])
    route_pysf_state = PedestrianStates(lambda: ped_states[id_offset:])

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
    ped_behaviors: List[PedestrianBehavior] = [crowd_behavior, route_behavior]
    return pysf_state, groups, ped_behaviors
