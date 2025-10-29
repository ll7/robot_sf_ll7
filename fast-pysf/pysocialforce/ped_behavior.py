from math import dist
from typing import List, Dict, Tuple, Protocol
from dataclasses import dataclass, field

from pysocialforce.map_config import GlobalRoute, sample_circle, sample_zone
from pysocialforce.navigation import RouteNavigator
from pysocialforce.ped_grouping import PedestrianGroupings

Vec2D = Tuple[float, float]
Zone = Tuple[Vec2D, Vec2D, Vec2D]


class PedestrianBehavior(Protocol):
    def step(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


@dataclass
class CrowdedZoneBehavior:
    """
    A class representing the behavior of pedestrians in crowded zones.

    Attributes:
        groups (PedestrianGroupings): The pedestrian groupings.
        zone_assignments (Dict[int, int]): The assignments of zones to pedestrians.
        crowded_zones (List[Zone]): The list of crowded zones.
        goal_proximity_threshold (float): The proximity threshold to the goal.

    Methods:
        step(): Perform a step of the behavior.
        reset(): Reset the behavior.
    """

    groups: PedestrianGroupings
    zone_assignments: Dict[int, int]
    crowded_zones: List[Zone]
    goal_proximity_threshold: float = 1

    def step(self):
        """
        Perform a step of the behavior.

        If a pedestrian group is close to its goal, redirect the group to a new goal within its assigned crowded zone.
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
        Reset the behavior.

        Redirect each pedestrian group to a new goal within its assigned crowded zone.
        """
        for gid in self.groups.group_ids:
            any_pid = next(iter(self.groups.groups[gid]))
            zone = self.crowded_zones[self.zone_assignments[any_pid]]
            new_goal = sample_zone(zone, 1)[0]
            self.groups.redirect_group(gid, new_goal)


@dataclass
class FollowRouteBehavior:
    """
    Represents the behavior of pedestrians following a predefined route.

    Attributes:
        groups (PedestrianGroupings): The pedestrian groupings.
        route_assignments (Dict[int, GlobalRoute]): The route assignments for each group.
        initial_sections (List[int]): The initial sections for each group.
        goal_proximity_threshold (float): The proximity threshold to consider a goal reached.
        navigators (Dict[int, RouteNavigator]): The route navigators for each group.
    """
    groups: PedestrianGroupings
    route_assignments: Dict[int, GlobalRoute]
    initial_sections: List[int]
    goal_proximity_threshold: float = 1
    navigators: Dict[int, RouteNavigator] = field(init=False)

    def __post_init__(self):
        """
        Initializes the route navigators for each group based on the route assignments and initial sections.
        """
        self.navigators = {}
        for (gid, route), sec_id in zip(self.route_assignments.items(), self.initial_sections):
            group_pos = self.groups.group_centroid(gid)
            self.navigators[gid] = RouteNavigator(
                route.waypoints, sec_id + 1, self.goal_proximity_threshold, group_pos)

    def step(self):
        """
        Performs a step of the behavior for each group.
        """
        for gid, nav in self.navigators.items():
            group_pos = self.groups.group_centroid(gid)
            nav.update_position(group_pos)
            if nav.reached_destination:
                self.respawn_group_at_start(gid)
            elif nav.reached_waypoint:
                self.groups.redirect_group(gid, nav.current_waypoint)

    def reset(self):
        """
        TODO: why does the reset method do nothing?
        """
        pass

    def respawn_group_at_start(self, gid: int):
        """
        Respawns a group at the start of the route.

        Args:
            gid (int): The group ID.
        """
        nav = self.navigators[gid]
        num_peds = self.groups.group_size(gid)
        circle = self.route_assignments[gid].spawn_circle
        spawn_positions = sample_circle(circle, num_peds)
        self.groups.reposition_group(gid, spawn_positions)
        self.groups.redirect_group(gid, nav.waypoints[0])
        nav.waypoint_id = 0
