from math import dist
from typing import List, Dict, Tuple, Protocol
from dataclasses import dataclass, field

from robot_sf.map_config import GlobalRoute
from robot_sf.navigation import RouteNavigator
from robot_sf.ped_spawn_generator import sample_zone
from robot_sf.ped_grouping import PedestrianGroupings

Vec2D = Tuple[float, float]
Zone = Tuple[Vec2D, Vec2D, Vec2D]


class PedestrianBehavior(Protocol):
    def step(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


@dataclass
class CrowdedZoneBehavior:
    groups: PedestrianGroupings
    zone_assignments: Dict[int, int]
    crowded_zones: List[Zone]
    goal_proximity_threshold: float = 1

    def step(self):
        for gid in self.groups.group_ids:
            centroid = self.groups.group_centroid(gid)
            goal = self.groups.goal_of_group(gid)
            dist_to_goal = dist(centroid, goal)
            if dist_to_goal < self.goal_proximity_threshold:
                any_pid = next(iter(self.groups.groups[gid]))
                zone = self.crowded_zones[self.zone_assignments[any_pid]]
                new_goal = sample_zone(zone, 1)[0]
                self.groups.redirect_group(gid, new_goal)

    def reset(self):
        for gid in self.groups.group_ids:
            any_pid = next(iter(self.groups.groups[gid]))
            zone = self.crowded_zones[self.zone_assignments[any_pid]]
            new_goal = sample_zone(zone, 1)[0]
            self.groups.redirect_group(gid, new_goal)


@dataclass
class FollowRouteBehavior:
    groups: PedestrianGroupings
    route_assignments: Dict[int, GlobalRoute]
    initial_sections: List[int]
    goal_proximity_threshold: float = 1
    navigators: Dict[int, RouteNavigator] = field(init=False)

    def __post_init__(self):
        self.navigators = {}
        for (gid, route), sec_id in zip(self.route_assignments.items(), self.initial_sections):
            group_pos = self.groups.group_centroid(gid)
            self.navigators[gid] = RouteNavigator(
                route.waypoints, sec_id + 1, self.goal_proximity_threshold, group_pos)

    def step(self):
        for gid, nav in self.navigators.items():
            group_pos = self.groups.group_centroid(gid)
            nav.update_position(group_pos)
            if nav.reached_destination:
                self.reset_group(gid)
            elif nav.reached_waypoint:
                self.groups.redirect_group(gid, nav.current_waypoint)

    def reset(self):
        for gid in self.navigators:
            self.reset_group(gid)

    def reset_group(self, gid: int):
        nav = self.navigators[gid]
        num_peds = self.groups.group_size(gid)
        spawn_zone = self.route_assignments[gid].spawn_zone
        spawn_positions = sample_zone(spawn_zone, num_peds)
        self.groups.reposition_group(gid, spawn_positions)
        self.groups.redirect_group(gid, nav.current_waypoint)
        nav.waypoint_id = 0
