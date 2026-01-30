"""
ped_behavior.py
- Defines the behavior of pedestrian groups
- Pedestrian groups can be assigned to follow a route or to move around a crowded zone
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from math import cos, dist, sin
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from robot_sf.common.types import RobotPose, Vec2D, Zone
from robot_sf.nav.map_config import GlobalRoute, SinglePedestrianDefinition
from robot_sf.nav.navigation import RouteNavigator
from robot_sf.ped_npc.ped_grouping import PedestrianGroupings, PedestrianStates
from robot_sf.ped_npc.ped_zone import sample_zone

if TYPE_CHECKING:
    from shapely.prepared import PreparedGeometry


class PedestrianBehavior(Protocol):
    """
    !!! Not Implemented !!!
    TODO: What is the purpose of this class?
    """

    def step(self):
        """TODO docstring. Document this function."""
        raise NotImplementedError()

    def reset(self):
        """TODO docstring. Document this function."""
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
    zone_assignments: dict[int, int]
    crowded_zones: list[Zone]
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
    """
    A class that defines the behavior of pedestrian groups following a route.

    Attributes
    ----------
    groups : PedestrianGroupings
        The groups of pedestrians.
    route_assignments : Dict[int, GlobalRoute]
        The assignments of groups to routes.
    initial_sections : List[int]
        The initial sections of the routes.
    obstacle_polygons : List[PreparedGeometry] | None
        Obstacles to avoid when respawning at the route start.
    goal_proximity_threshold : float
        The distance threshold for proximity to a goal. Default is 1.
    navigators : Dict[int, RouteNavigator]
        The navigators for each group. Initialized in the post-init method.

    Methods
    -------
    step():
        Update the positions of groups and respawn any groups that have reached their
        destination.
    reset():
        No action is performed in this method.
    respawn_group_at_start(gid: int):
        Respawn a group at the start of its route.
    """

    groups: PedestrianGroupings
    route_assignments: dict[int, GlobalRoute]
    initial_sections: list[int]
    goal_proximity_threshold: float = 1
    obstacle_polygons: list["PreparedGeometry"] | None = None
    navigators: dict[int, RouteNavigator] = field(init=False)

    def __post_init__(self):
        """
        Initialize the navigators for each group.

        For each group, a RouteNavigator is created with the group's assigned route,
        initial section, goal proximity threshold, and current position.
        """
        self.navigators = {}
        for (gid, route), sec_id in zip(
            self.route_assignments.items(),
            self.initial_sections,
            strict=False,
        ):
            group_pos = self.groups.group_centroid(gid)
            self.navigators[gid] = RouteNavigator(
                route.waypoints,
                sec_id + 1,
                self.goal_proximity_threshold,
                group_pos,
            )

    def step(self):
        """
        Update the positions of groups and respawn any groups that have reached their
        destination.

        For each group, the group's position is updated in its navigator. If the group
        has reached its destination, it is respawned at the start of its route. If the
        group has reached a waypoint, it is redirected to the current waypoint.
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
        Reset the behavior state.

        For FollowRouteBehavior, groups maintain their current navigation state
        across episodes. When groups reach their destination, they are respawned
        via respawn_group_at_start() during step() execution.

        This method intentionally performs no action to preserve group continuity
        within ongoing route navigation sessions. The method is retained to satisfy
        the PedestrianBehavior protocol interface.
        """

    def respawn_group_at_start(self, gid: int):
        """
        Respawn a group at the start of its route.

        The group is repositioned to the spawn zone of its route (avoiding obstacles
        when provided), and it is redirected to the first waypoint of its route.
        The waypoint ID of its navigator is reset to 0.

        Parameters
        ----------
        gid : int
            The ID of the group to respawn.
        """
        nav = self.navigators[gid]
        num_peds = self.groups.group_size(gid)
        spawn_zone = self.route_assignments[gid].spawn_zone
        spawn_positions = sample_zone(
            spawn_zone,
            num_peds,
            obstacle_polygons=self.obstacle_polygons,
        )
        self.groups.reposition_group(gid, spawn_positions)
        self.groups.redirect_group(gid, nav.waypoints[0])
        nav.waypoint_id = 0


@dataclass
class SinglePedestrianRuntime:
    """Runtime state for a single pedestrian with optional trajectory/wait behavior."""

    ped_id: int
    definition: SinglePedestrianDefinition
    trajectory: list[Vec2D]
    waypoint_index: int = 0
    pending_waits: dict[int, float] = field(default_factory=dict)
    wait_remaining_s: float = 0.0
    waiting_for_advance: bool = False
    joined_group_id: int | None = None
    left_group: bool = False


@dataclass
class SinglePedestrianBehavior:
    """Behavior controller for trajectory-driven single pedestrians."""

    states: PedestrianStates
    groups: PedestrianGroupings
    single_pedestrians: list[SinglePedestrianDefinition]
    single_offset: int
    time_step_s: float = 0.1
    goal_proximity_threshold: float | None = None
    robot_pose_provider: Callable[[], list[RobotPose]] | None = None
    _runtimes: list[SinglePedestrianRuntime] = field(init=False, default_factory=list)
    _id_to_global: dict[str, int] = field(init=False, default_factory=dict)
    _warned_missing_targets: set[int] = field(init=False, default_factory=set)

    def __post_init__(self):
        """Initialize runtime state for each single pedestrian."""
        if self.goal_proximity_threshold is None:
            self.goal_proximity_threshold = 0.2
        for idx, ped in enumerate(self.single_pedestrians):
            global_id = self.single_offset + idx
            self._id_to_global[ped.id] = global_id
            waits = {rule.waypoint_index: rule.wait_s for rule in ped.wait_at or []}
            self._runtimes.append(
                SinglePedestrianRuntime(
                    ped_id=global_id,
                    definition=ped,
                    trajectory=ped.trajectory or [],
                    pending_waits=waits,
                )
            )

    def set_robot_pose_provider(self, provider: Callable[[], list[RobotPose]] | None) -> None:
        """Set or update the robot pose provider callback."""
        self.robot_pose_provider = provider

    def step(self):
        """Advance single-pedestrian behaviors for one timestep."""
        for runtime in self._runtimes:
            role = runtime.definition.role
            if role == "wait":
                self._hold_position(runtime)
                continue
            if role in {"follow", "lead", "accompany"}:
                if self._apply_robot_relative_role(runtime, role):
                    continue
            if role == "join":
                if self._apply_join_role(runtime):
                    continue
            if role == "leave":
                self._apply_leave_role(runtime)
            self._advance_trajectory(runtime)

    def reset(self):
        """Reset per-pedestrian runtime state for a new episode."""
        for runtime in self._runtimes:
            runtime.waypoint_index = 0
            runtime.wait_remaining_s = 0.0
            runtime.waiting_for_advance = False
            runtime.joined_group_id = None
            runtime.left_group = False

    def _advance_trajectory(self, runtime: SinglePedestrianRuntime) -> None:
        """Advance trajectory waypoints and honor wait rules."""
        if not runtime.trajectory:
            return

        if self._tick_wait(runtime):
            return

        current_target = runtime.trajectory[runtime.waypoint_index]
        pos = self.states.pos_of(runtime.ped_id)
        if dist(pos, current_target) > self.goal_proximity_threshold:
            return

        if runtime.waypoint_index in runtime.pending_waits:
            wait_s = runtime.pending_waits.pop(runtime.waypoint_index)
            runtime.wait_remaining_s = wait_s
            runtime.waiting_for_advance = True
            self._hold_position(runtime)
            return

        self._advance_waypoint(runtime)

    def _advance_waypoint(self, runtime: SinglePedestrianRuntime) -> None:
        """Advance to the next waypoint if available."""
        next_idx = runtime.waypoint_index + 1
        if next_idx >= len(runtime.trajectory):
            return
        runtime.waypoint_index = next_idx
        self.states.redirect(runtime.ped_id, runtime.trajectory[next_idx])

    def _tick_wait(self, runtime: SinglePedestrianRuntime) -> bool:
        """Update wait timers and hold position while waiting.

        Returns:
            bool: ``True`` if the pedestrian should continue waiting.
        """
        if runtime.wait_remaining_s <= 0:
            return False
        runtime.wait_remaining_s = max(0.0, runtime.wait_remaining_s - self.time_step_s)
        self._hold_position(runtime)
        if runtime.wait_remaining_s <= 0 and runtime.waiting_for_advance:
            runtime.waiting_for_advance = False
            self._advance_waypoint(runtime)
        return runtime.wait_remaining_s > 0

    def _hold_position(self, runtime: SinglePedestrianRuntime) -> None:
        """Hold the pedestrian in place by zeroing velocity and goal."""
        pos = self.states.pos_of(runtime.ped_id)
        self.states.redirect(runtime.ped_id, pos)
        self.states.set_velocity(runtime.ped_id, (0.0, 0.0))

    def _apply_robot_relative_role(
        self,
        runtime: SinglePedestrianRuntime,
        role: str,
    ) -> bool:
        """Apply follow/lead/accompany roles that track a robot pose.

        Returns:
            bool: ``True`` when the role handled goal updates.
        """
        if role == "lead" and (runtime.definition.goal or runtime.definition.trajectory):
            return False
        robot_pose = self._resolve_robot_pose(runtime)
        if robot_pose is None:
            return False
        offset = runtime.definition.role_offset or self._default_role_offset(role)
        target = self._pose_with_offset(robot_pose, offset)
        self.states.redirect(runtime.ped_id, target)
        return True

    def _apply_join_role(self, runtime: SinglePedestrianRuntime) -> bool:
        """Direct a pedestrian to join a target group.

        Returns:
            bool: ``True`` when the join role controls the pedestrian goal.
        """
        target_group = runtime.joined_group_id or self._resolve_target_group_id(runtime)
        if target_group is None or not self.groups.groups.get(target_group):
            return False
        target_pos = self.groups.group_centroid(target_group)
        self.states.redirect(runtime.ped_id, target_pos)
        pos = self.states.pos_of(runtime.ped_id)
        if dist(pos, target_pos) <= self.goal_proximity_threshold:
            self.groups.add_to_group(runtime.ped_id, target_group)
            runtime.joined_group_id = target_group
        return True

    def _apply_leave_role(self, runtime: SinglePedestrianRuntime) -> None:
        """Remove a pedestrian from its current group once per episode."""
        if runtime.left_group:
            return
        if runtime.ped_id in self.groups.group_by_ped_id:
            self.groups.new_group({runtime.ped_id})
        runtime.left_group = True

    def _resolve_target_group_id(self, runtime: SinglePedestrianRuntime) -> int | None:
        """Resolve the target group for join roles.

        Returns:
            int | None: Target group id when available.
        """
        target_id = runtime.definition.role_target_id
        if target_id:
            if target_id.startswith("group:"):
                try:
                    group_id = int(target_id.split(":", 1)[1])
                except ValueError:
                    self._warn_missing_target(runtime, target_id)
                    return None
                if group_id not in self.groups.groups:
                    self._warn_missing_target(runtime, target_id)
                    return None
                return group_id
            target_global = self._id_to_global.get(target_id)
            if target_global is None:
                self._warn_missing_target(runtime, target_id)
                return None
            return self.groups.ensure_group_for_ped(target_global)

        group_ids = list(self.groups.group_ids)
        if not group_ids:
            return None
        own_group = self.groups.group_by_ped_id.get(runtime.ped_id)
        candidates = [gid for gid in group_ids if gid != own_group]
        if not candidates:
            return None
        pos = self.states.pos_of(runtime.ped_id)
        return min(candidates, key=lambda gid: dist(pos, self.groups.group_centroid(gid)))

    def _resolve_robot_pose(self, runtime: SinglePedestrianRuntime) -> RobotPose | None:
        """Resolve the robot pose for robot-relative roles.

        Returns:
            RobotPose | None: Robot pose for role targeting, if available.
        """
        if self.robot_pose_provider is None:
            self._warn_missing_target(runtime, "robot")
            return None
        poses = self.robot_pose_provider()
        if not poses:
            self._warn_missing_target(runtime, "robot")
            return None
        target_id = runtime.definition.role_target_id
        if target_id and not (target_id == "robot" or target_id.startswith("robot:")):
            self._warn_missing_target(runtime, target_id)
            return None
        if target_id and target_id.startswith("robot:"):
            try:
                idx = int(target_id.split(":", 1)[1])
            except ValueError:
                self._warn_missing_target(runtime, target_id)
                return None
            if idx < 0 or idx >= len(poses):
                self._warn_missing_target(runtime, target_id)
                return None
            return poses[idx]
        return poses[0]

    def _pose_with_offset(self, pose: RobotPose, offset: Vec2D) -> Vec2D:
        """Convert a robot-relative offset into a world-space target point.

        Returns:
            Vec2D: Target point in world coordinates.
        """
        (x, y), heading = pose
        forward, lateral = offset
        dx = forward * cos(heading) - lateral * sin(heading)
        dy = forward * sin(heading) + lateral * cos(heading)
        return (x + dx, y + dy)

    @staticmethod
    def _default_role_offset(role: str) -> Vec2D:
        """Default offsets for robot-relative roles.

        Returns:
            Vec2D: Default (forward, lateral) offset for the role.
        """
        defaults = {
            "follow": (-1.0, 0.0),
            "lead": (1.5, 0.0),
            "accompany": (0.0, 1.0),
        }
        return defaults.get(role, (0.0, 0.0))

    def _warn_missing_target(self, runtime: SinglePedestrianRuntime, target: str) -> None:
        """Warn once per pedestrian when a role target cannot be resolved."""
        if runtime.ped_id in self._warned_missing_targets:
            return
        self._warned_missing_targets.add(runtime.ped_id)
        logger.warning(
            "Single pedestrian '{}' role target '{}' could not be resolved; using fallback behavior.",
            runtime.definition.id,
            target,
        )
