from math import dist, atan2, sin, cos, ceil
from dataclasses import dataclass, field
from typing import Tuple, List, Set, Dict
import numpy as np


PedState = np.ndarray
PedGrouping = Set[int]
Vec2D = Tuple[float, float]
SpawnZone = Tuple[Vec2D, Vec2D, Vec2D] # rect ABC with sides |A B|, |B C| and diagonal |A C|
ZoneAssignments = Dict[int, int]


@dataclass
class PedSpawnConfig:
    peds_per_area_m2: float
    max_group_members: int
    group_member_probs: List[float] = field(default_factory=list)
    initial_speed: float = 0.5
    group_size_decay: float = 0.3

    def __post_init__(self):
        if not len(self.group_member_probs) == self.max_group_members:
            # initialize group size probabilities decaying by power law
            power_dist = [self.group_size_decay**i for i in range(self.max_group_members)]
            self.group_member_probs = [p / sum(power_dist) for p in power_dist]


@dataclass
class SpawnGenerator:
    spawn_zones: List[SpawnZone]
    zone_areas: List[float] = field(init=False)
    _zone_probs: List[float] = field(init=False)

    def __post_init__(self):
        self.zone_areas = [dist(p1, p2) * dist(p2, p3) for p1, p2, p3 in self.spawn_zones]
        total_area = sum(self.zone_areas)
        self._zone_probs = [area / total_area for area in self.zone_areas]
        # info: distribute proportionally by zone area

    def generate(self, num_samples: int) -> Tuple[List[Vec2D], int]:
        zone_id = np.random.choice(len(self.spawn_zones), size=1, p=self._zone_probs)[0]
        p_1, p_2, p_3 = self.spawn_zones[zone_id]

        d_x, d_y = dist(p_2, p_3), dist(p_1, p_2)
        rot = atan2(p_3[1] - p_2[1], p_3[0] - p_2[0])
        x_pos = np.random.uniform(0, d_x, (num_samples, 1))
        y_pos = np.random.uniform(0, d_y, (num_samples, 1))
        norm_points = np.concatenate((x_pos, y_pos), axis=1)

        def rotate(point: Vec2D, angle: float) -> Vec2D:
            pos_x, pos_y = point
            new_x = pos_x * cos(angle) - pos_y * sin(angle)
            new_y = pos_x * sin(angle) + pos_y * cos(angle)
            return new_x, new_y

        rotated_points = [rotate((x, y), rot) for x, y in norm_points]
        shifted_points = [(x + p_2[0], y + p_2[1]) for x, y in rotated_points]
        return shifted_points, zone_id


def initialize_pedestrians(config: PedSpawnConfig, spawn_zones: List[SpawnZone]) \
        -> Tuple[PedState, List[PedGrouping], ZoneAssignments]:

    spawn_gen = SpawnGenerator(spawn_zones)
    goal_gens = [SpawnGenerator([z]) for z in spawn_zones]

    total_num_peds = ceil(sum(spawn_gen.zone_areas) * config.peds_per_area_m2)
    ped_states, groups = np.zeros((total_num_peds, 6)), []
    num_unassigned_peds = total_num_peds
    zone_assignments = dict()

    while num_unassigned_peds > 0:
        probs = config.group_member_probs
        num_peds_in_group = np.random.choice(len(probs), p=probs) + 1
        num_peds_in_group = min(num_peds_in_group, num_unassigned_peds)
        num_assigned_peds = total_num_peds - num_unassigned_peds
        ped_ids = list(range(num_assigned_peds, total_num_peds))[:num_peds_in_group]

        if len(ped_ids) > 1:
            groups.append(set(ped_ids))

        # spawn all group members in the same randomly sampled zone and also
        # keep them within that zone by picking the group's goal accordingly
        spawn_points, zone_id = spawn_gen.generate(num_peds_in_group)
        group_goal = goal_gens[zone_id].generate(1)[0][0]

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
