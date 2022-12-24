from math import dist, atan2, sin, cos
from dataclasses import dataclass, field
from typing import Tuple, List, Set
import numpy as np


PedState = np.ndarray
PedGrouping = Set[int]
Vec2D = Tuple[float, float]
SpawnZone = Tuple[Vec2D, Vec2D, Vec2D] # rect ABC with sides |A B|, |B C| and diagonal |A C|


@dataclass
class PedSpawnConfig:
    total_num_peds: int
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
    _zone_probs: List[float] = field(init=False)

    def __post_init__(self):
        zone_areas = [dist(p1, p2) * dist(p2, p3) for p1, p2, p3 in self.spawn_zones]
        total_area = sum(zone_areas)
        self._zone_probs = [area / total_area for area in zone_areas]
        # info: distribute proportionally by zone area

    def generate(self, num_samples: int, scale: float=1.0) -> List[Vec2D]:
        zone_id = np.random.choice(len(self.spawn_zones), size=1, p=self._zone_probs)[0]
        p_1, p_2, p_3 = self.spawn_zones[zone_id]

        s_1, s_2 = dist(p_1, p_2), dist(p_2, p_3)
        rot = atan2(p_3[1] - p_2[1], p_3[0] - p_2[0])
        center = np.random.uniform(0, s_1), np.random.uniform(0, s_2)
        norm_points = np.random.normal(center, scale, size=(num_samples, 2))
        # TODO: handle points outside of the spawn zone

        def rotate(point: Vec2D, angle: float) -> Vec2D:
            pos_x, pos_y = point
            new_x = pos_x * cos(angle) - pos_y * sin(angle)
            new_y = pos_x * sin(angle) + pos_y * cos(angle)
            return new_x, new_y

        rotated_points = [rotate((x, y), rot) for x, y in norm_points]
        shifted_points = [(x + p_2[0], y + p_2[1]) for x, y in rotated_points]
        return shifted_points


def initialize_pedestrians(
        config: PedSpawnConfig, spawn_gen: SpawnGenerator,
        goal_gen: SpawnGenerator) -> Tuple[PedState, List[PedGrouping]]:

    ped_states, groups = np.zeros((config.total_num_peds, 6)), []
    num_unassigned_peds = config.total_num_peds

    while num_unassigned_peds > 0:
        probs = config.group_member_probs
        num_peds_in_group = np.random.choice(len(probs), p=probs) + 1
        num_peds_in_group = min(num_peds_in_group, num_unassigned_peds)
        num_assigned_peds = config.total_num_peds - num_unassigned_peds
        ped_ids = list(range(num_assigned_peds, config.total_num_peds))[:num_peds_in_group]

        if len(ped_ids) > 1:
            groups.append(set(ped_ids))

        spawn_points = spawn_gen.generate(num_peds_in_group)
        group_goal = goal_gen.generate(1)[0]
        centroid = np.mean(spawn_points, axis=0)
        rot = atan2(group_goal[1] - centroid[1], group_goal[0] - centroid[0])
        velocity = np.array([cos(rot), sin(rot)]) * config.initial_speed
        ped_states[ped_ids, 0:2] = spawn_points
        ped_states[ped_ids, 2:4] = velocity
        ped_states[ped_ids, 4:6] = group_goal

        num_unassigned_peds -= num_peds_in_group

    return ped_states, groups
