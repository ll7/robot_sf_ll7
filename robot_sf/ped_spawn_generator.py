from math import dist, atan2, sin, cos, ceil
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Dict
import numpy as np

from robot_sf.map_config import GlobalRoute


PedState = np.ndarray
PedGrouping = Set[int]
Vec2D = Tuple[float, float]
Zone = Tuple[Vec2D, Vec2D, Vec2D] # rect ABC with sides |A B|, |B C| and diagonal |A C|
ZoneAssignments = Dict[int, int]


@dataclass
class PedSpawnConfig:
    peds_per_area_m2: float
    max_group_members: int
    group_member_probs: List[float] = field(default_factory=list)
    initial_speed: float = 0.5
    group_size_decay: float = 0.3
    sidewalk_width: float = 3.0

    def __post_init__(self):
        if not len(self.group_member_probs) == self.max_group_members:
            # initialize group size probabilities decaying by power law
            power_dist = [self.group_size_decay**i for i in range(self.max_group_members)]
            self.group_member_probs = [p / sum(power_dist) for p in power_dist]


def rotate(point: Vec2D, angle: float) -> Vec2D:
    pos_x, pos_y = point
    new_x = pos_x * cos(angle) - pos_y * sin(angle)
    new_y = pos_x * sin(angle) + pos_y * cos(angle)
    return new_x, new_y


def sample_zone(zone: Zone, num_samples: int) -> List[Vec2D]:
    a, b, c = zone
    width, height = dist(b, c), dist(a, b)
    rot = atan2(c[1] - b[1], c[0] - b[0])

    x_pos = np.random.uniform(0, width, (num_samples, 1))
    y_pos = np.random.uniform(0, height, (num_samples, 1))
    norm_points = np.concatenate((x_pos, y_pos), axis=1)

    add_vec = lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1])
    return [add_vec(rotate((x, y), rot), b) for x, y in norm_points]


def sample_route(
        route: GlobalRoute, num_samples: int,
        sidewalk_width: float) -> Tuple[List[Vec2D], int]:

    sampled_offset = np.random.uniform(0, route.total_length)
    sec_id = next(iter([i - 1 for i, o in enumerate(route.section_offsets) if o >= sampled_offset]), -1)

    start, end = route.sections[sec_id]
    add_vecs = lambda v1, v2: (v1[0] + v2[0], v1[1] + v2[1])
    sub_vecs = lambda v1, v2: (v1[0] - v2[0], v1[1] - v2[1])
    clip_spread = lambda v: np.clip(v, -sidewalk_width / 2, sidewalk_width / 2)
    center = add_vecs(start, sub_vecs(end, start))
    std_dev = sidewalk_width / 4

    x_offsets = clip_spread(np.random.normal(center[0], std_dev, (num_samples, 1)))
    y_offsets = clip_spread(np.random.normal(center[1], std_dev, (num_samples, 1)))
    points = np.concatenate((x_offsets, y_offsets), axis=1) + center
    return [p for p in points], sec_id


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
