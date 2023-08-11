from random import sample, randint
from math import dist
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from robot_sf.nav.map_config import MapDefinition
from robot_sf.ped_npc.ped_zone import sample_zone

Vec2D = Tuple[float, float]
Zone = Tuple[Vec2D, Vec2D, Vec2D]


@dataclass
class RouteNavigator:
    waypoints: List[Vec2D] = field(default_factory=list)
    waypoint_id: int = 0
    proximity_threshold: float = 1.0 # info: should be set to vehicle radius + goal radius
    pos: Vec2D = field(default=(0, 0))
    reached_waypoint: bool = False

    @property
    def reached_destination(self) -> bool:
        return len(self.waypoints) == 0 or \
            dist(self.waypoints[-1], self.pos) <= self.proximity_threshold

    @property
    def current_waypoint(self) -> Vec2D:
        return self.waypoints[self.waypoint_id]

    @property
    def next_waypoint(self) -> Optional[Vec2D]:
        return self.waypoints[self.waypoint_id + 1] \
            if self.waypoint_id + 1 < len(self.waypoints) else None

    def update_position(self, pos: Vec2D):
        reached_waypoint = dist(self.current_waypoint, pos) <= self.proximity_threshold
        if reached_waypoint:
            self.waypoint_id = min(len(self.waypoints) - 1, self.waypoint_id + 1)
        self.pos = pos
        self.reached_waypoint = reached_waypoint

    def new_route(self, route: List[Vec2D]):
        self.waypoints = route
        self.waypoint_id = 0


def sample_route(
        map_def: MapDefinition,
        spawn_id: Optional[int]=None) -> List[Vec2D]:

    spawn_id = spawn_id if spawn_id is not None else \
        randint(0, len(map_def.robot_spawn_zones) - 1)
    routes = map_def.robot_routes_by_spawn_id[spawn_id]
    route = sample(routes, k=1)[0]
    initial_spawn = sample_zone(route.spawn_zone, 1)[0]
    final_goal = sample_zone(route.goal_zone, 1)[0]
    route = [initial_spawn] + route.waypoints + [final_goal]
    # TODO: add noise to the exact waypoint positions to avoid learning routes by heart
    return route
