from math import dist, atan2, sin, cos, ceil
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Dict
import numpy as np

from pysocialforce.map_config import GlobalRoute, sample_zone
from pysocialforce.ped_grouping import PedestrianStates, PedestrianGroupings
from pysocialforce.ped_behavior import \
    PedestrianBehavior, CrowdedZoneBehavior, FollowRouteBehavior

PedState = np.ndarray
PedGrouping = Set[int]
Vec2D = Tuple[float, float]
Zone = Tuple[Vec2D, Vec2D, Vec2D]  # rect ABC with sides |A B|, |B C| and diagonal |A C|
ZoneAssignments = Dict[int, int]


@dataclass
class PedSpawnConfig:
    """
    Configuration class for pedestrian spawning.
    """

    peds_per_area_m2: float = 0.04
    max_group_members: int = 5
    group_member_probs: List[float] = field(default_factory=list)
    initial_speed: float = 0.5
    group_size_decay: float = 0.3
    sidewalk_width: float = 3.0

    def __post_init__(self):
        """
        Initialize the PedSpawnConfig object.
        If the length of group_member_probs is not equal to max_group_members,
        initialize group size probabilities decaying by power law.
        """
        if len(self.group_member_probs) != self.max_group_members:
            # initialize group size probabilities decaying by power law
            power_dist = [self.group_size_decay**i for i in range(self.max_group_members)]
            self.group_member_probs = [p / sum(power_dist) for p in power_dist]


def sample_group_spawn_on_route(
        route: GlobalRoute, num_samples: int,
        sidewalk_width: float) -> Tuple[List[Vec2D], int]:
    """
    Sample points along a given route.

    Args:
        route (GlobalRoute): The global route to sample points from.
        num_samples (int): The number of points to sample.
        sidewalk_width (float): The width of the sidewalk.

    Returns:
        Tuple[List[Vec2D], int]: A tuple containing a list of sampled points and the section ID of the sampled point.
    """

    sampled_offset = np.random.uniform(0, route.total_length)
    sec_id = next(
        iter([i - 1 for i, o in enumerate(route.section_offsets) if o >= sampled_offset]), -1)
    sec_offset = sampled_offset - route.section_offsets[sec_id]
    sec_len = route.section_lengths[sec_id]

    start, end = route.sections[sec_id]
    def add_vecs(v1, v2): return (v1[0] + v2[0], v1[1] + v2[1])
    def sub_vecs(v1, v2): return (v1[0] - v2[0], v1[1] - v2[1])
    def scale_vec(v, f): return (v[0] * f, v[1] * f)
    def clip_spread(v): return np.clip(v, -sidewalk_width / 2, sidewalk_width / 2)
    center = add_vecs(start, scale_vec(sub_vecs(end, start), sec_offset / sec_len))
    std_dev = sidewalk_width / 4

    x_offsets = clip_spread(np.random.normal(center[0], std_dev, (num_samples, 1)))
    y_offsets = clip_spread(np.random.normal(center[1], std_dev, (num_samples, 1)))
    points = np.concatenate((x_offsets, y_offsets), axis=1) + center
    return [(x, y) for x, y in points], sec_id


@dataclass
class ZonePointsGenerator:
    """
    Generates points within different zones based on their areas.

    Attributes:
        zones (List[Zone]): List of Zone objects representing different zones.
        zone_areas (List[float]): List of zone areas calculated based on the zone points.
        _zone_probs (List[float]): List of probabilities for each zone based on their areas.
    """

    zones: List[Zone]
    zone_areas: List[float] = field(init=False)
    _zone_probs: List[float] = field(init=False)

    def __post_init__(self):
        self.zone_areas = [dist(p1, p2) * dist(p2, p3) for p1, p2, p3 in self.zones]
        total_area = sum(self.zone_areas)
        self._zone_probs = [area / total_area for area in self.zone_areas]
        # info: distribute proportionally by zone area

    def generate(self, num_samples: int) -> Tuple[List[Vec2D], int]:
        """
        Generates a specified number of samples within a randomly selected zone.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            Tuple[List[Vec2D], int]: A tuple containing the generated samples and the ID of the selected zone.
        """
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
        spawn_pos, sec_id = sample_group_spawn_on_route(
            self.routes[route_id], num_samples, self.sidewalk_width)
        return spawn_pos, route_id, sec_id


def populate_ped_routes(config: PedSpawnConfig, routes: List[GlobalRoute]) \
        -> Tuple[np.ndarray, List[PedGrouping], Dict[int, GlobalRoute], List[int]]:
    """
    Populates pedestrian routes with initial states and assignments.

    Args:
        config (PedSpawnConfig): The configuration for pedestrian spawning.
        routes (List[GlobalRoute]): The list of global routes.

    Returns:
        Tuple[np.ndarray, List[PedGrouping], Dict[int, GlobalRoute], List[int]]: A tuple containing the following:
            - ped_states (np.ndarray): An array of pedestrian states.
            - groups (List[PedGrouping]): A list of pedestrian groupings.
            - route_assignments (Dict[int, GlobalRoute]): A dictionary mapping group indices to assigned routes.
            - initial_sections (List[int]): A list of initial section indices for each group.
    """
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
    """
    Populates crowded zones with pedestrians.

    Args:
        config (PedSpawnConfig): The configuration for pedestrian spawning.
        crowded_zones (List[Zone]): The list of crowded zones.

    Returns:
        Tuple[PedState, List[PedGrouping], ZoneAssignments]: A tuple containing the pedestrian states,
        the list of pedestrian groupings, and the zone assignments.
    """
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
    """
    Populates the simulation with pedestrians based on the given parameters.

    Args:
        tau (float): The time step for the simulation.
        spawn_config (PedSpawnConfig): The configuration for spawning pedestrians.
        ped_routes (List[GlobalRoute]): The global routes for the pedestrians.
        ped_crowded_zones (List[Zone]): The crowded zones for the pedestrians.

    Returns:
        Tuple[PedestrianStates, PedestrianGroupings, List[PedestrianBehavior]]: A tuple containing the pedestrian states,
        pedestrian groupings, and pedestrian behaviors for the simulation.
    """

    crowd_ped_states_np, crowd_groups, zone_assignments = \
        populate_crowded_zones(spawn_config, ped_crowded_zones)
    route_ped_states_np, route_groups, route_assignments, initial_sections = \
        populate_ped_routes(spawn_config, ped_routes)

    combined_ped_states_np = np.concatenate((crowd_ped_states_np, route_ped_states_np))
    taus = np.full((combined_ped_states_np.shape[0]), tau)
    ped_states = np.concatenate((combined_ped_states_np, np.expand_dims(taus, -1)), axis=-1)
    id_offset = crowd_ped_states_np.shape[0]
    combined_groups = crowd_groups + [{id + id_offset for id in peds} for peds in route_groups]

    pysf_state = PedestrianStates(ped_states)
    crowd_pysf_state = PedestrianStates(ped_states[:id_offset])
    route_pysf_state = PedestrianStates(ped_states[id_offset:])

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
