"""
ped_behavior.py
- Defines the behavior of pedestrian groups
- Pedestrian groups can be assigned to follow a route or to move around a crowded zone
"""
from math import dist
from typing import List, Dict, Tuple, Protocol
from dataclasses import dataclass, field

from robot_sf.nav.map_config import GlobalRoute
from robot_sf.nav.navigation import RouteNavigator
from robot_sf.ped_npc.ped_zone import sample_zone
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings

Vec2D = Tuple[float, float]
Zone = Tuple[Vec2D, Vec2D, Vec2D]


class PedestrianBehavior(Protocol):
    """
    !!! Not Implemented !!!
    TODO: What is the purpose of this class?
    """
    def step(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


@dataclass
class CrowdedZoneBehavior:
    """
    A class that defines the behavior of pedestrians in crowded zones.

    Attributes
    ----------
    groups : PedestrianGroupings
        The groups of pedestrians.
    zone_assignments : Dict[int, int]
        The assignments of pedestrians to zones.
    crowded_zones : List[Zone]
        The crowded zones.
    goal_proximity_threshold : float
        The distance threshold for proximity to a goal. Default is 1.
        TODO: What is the unit of this distance?

    Methods
    -------
    step():
        Update the goals of groups that are close to their current goal.
    reset():
        Reset the goals of all groups.
    """

    groups: PedestrianGroupings
    zone_assignments: Dict[int, int]
    crowded_zones: List[Zone]
    goal_proximity_threshold: float = 1

    def step(self):
        """
        Update the goals of groups that are close to their current goal.

        For each group, if the distance from the group's centroid to its goal is less
        than the goal proximity threshold, a new goal is sampled from the group's assigned
        crowded zone and the group is redirected to this new goal.
        """
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
        """
        Reset the goals of all groups.

        For each group, a new goal is sampled from the group's assigned crowded zone and
        the group is redirected to this new goal.
        """
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
                self.respawn_group_at_start(gid)
            elif nav.reached_waypoint:
                self.groups.redirect_group(gid, nav.current_waypoint)

    def reset(self):
        pass

    def respawn_group_at_start(self, gid: int):
        nav = self.navigators[gid]
        num_peds = self.groups.group_size(gid)
        spawn_zone = self.route_assignments[gid].spawn_zone
        spawn_positions = sample_zone(spawn_zone, num_peds)
        self.groups.reposition_group(gid, spawn_positions)
        self.groups.redirect_group(gid, nav.waypoints[0])
        nav.waypoint_id = 0
