"""Pedestrian behavior controllers for crowd zones, routes, and scripted actors."""

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

#: Fail-open release time (seconds) for a proximity hold when no ``hold_timeout_s`` is configured.
#: The pedestrian releases after this dwell even if no robot ever approaches, so a stalled or
#: absent robot can never deadlock the scenario.
DEFAULT_HOLD_TIMEOUT_S = 6.0


class PedestrianBehavior(Protocol):
    """Protocol implemented by pedestrian behavior controllers."""

    def step(self):
        """Advance behavior state for one simulation timestep."""
        raise NotImplementedError()

    def reset(self):
        """Reset behavior state at an episode boundary."""
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
        Uses the same world-coordinate units as pedestrian positions and zones.

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
    reset_at_start : bool
        If True, groups will reset to the start of their routes on reset.

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
    reset_at_start: bool = False

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

        However, if reset_at_start is True, all groups will be immediately respawned at
        the start.
        """
        if self.reset_at_start:
            for gid in self.navigators.keys():
                self.respawn_group_at_start(gid)

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
    start_delay_remaining_s: float = 0.0
    wait_remaining_s: float = 0.0
    waiting_for_advance: bool = False
    joined_group_id: int | None = None
    left_group: bool = False
    hold_waypoint_index: int | None = None
    proximity_hold_engaged: bool = False
    proximity_hold_released: bool = False
    proximity_hold_elapsed_s: float = 0.0
    hold_released_by: str | None = None


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
                    start_delay_remaining_s=float(ped.start_delay_s),
                    hold_waypoint_index=self._resolve_hold_waypoint_index(ped),
                )
            )

    def set_robot_pose_provider(self, provider: Callable[[], list[RobotPose]] | None) -> None:
        """Set or update the robot pose provider callback."""
        self.robot_pose_provider = provider

    def step(self):
        """Advance single-pedestrian behaviors for one timestep."""
        for runtime in self._runtimes:
            if self._tick_start_delay(runtime):
                continue
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
            runtime.start_delay_remaining_s = float(runtime.definition.start_delay_s)
            runtime.wait_remaining_s = 0.0
            runtime.waiting_for_advance = False
            runtime.joined_group_id = None
            runtime.left_group = False
            runtime.proximity_hold_engaged = False
            runtime.proximity_hold_released = False
            runtime.proximity_hold_elapsed_s = 0.0
            runtime.hold_released_by = None

    def _advance_trajectory(self, runtime: SinglePedestrianRuntime) -> None:
        """Advance trajectory waypoints and honor wait rules."""
        if not runtime.trajectory:
            return

        if self._tick_wait(runtime):
            return

        if self._tick_proximity_hold(runtime):
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

    def _tick_start_delay(self, runtime: SinglePedestrianRuntime) -> bool:
        """Hold a pedestrian at its start until the configured release time.

        Returns:
            bool: True while the pedestrian should keep waiting at the start.
        """
        if runtime.start_delay_remaining_s <= 0:
            return False
        runtime.start_delay_remaining_s = max(
            0.0,
            runtime.start_delay_remaining_s - self.time_step_s,
        )
        self._hold_position(runtime)
        if runtime.start_delay_remaining_s <= 0:
            self._release_start_delay(runtime)
            return False
        return True

    def _release_start_delay(self, runtime: SinglePedestrianRuntime) -> None:
        """Restore the pedestrian's configured goal after a start-delay dwell."""
        if runtime.trajectory:
            self.states.redirect(runtime.ped_id, runtime.trajectory[runtime.waypoint_index])
        elif runtime.definition.goal is not None:
            self.states.redirect(runtime.ped_id, runtime.definition.goal)

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

    @staticmethod
    def _resolve_hold_waypoint_index(ped: SinglePedestrianDefinition) -> int | None:
        """Resolve the trajectory index where a proximity hold should engage.

        The pedestrian holds at the waypoint immediately preceding ``hold_ref_point`` (the curb
        before the conflict point), so a release triggers a genuine short crossing into the
        reference point.

        Returns:
            int | None: Trajectory index to hold at, or ``None`` when no proximity hold applies
            or the reference point is not on the trajectory (hold disabled, fail-open).
        """
        if (
            ped.hold_until_robot_within_m is None
            or not ped.trajectory
            or ped.hold_ref_point is None
        ):
            return None
        for k, waypoint in enumerate(ped.trajectory):
            if dist(waypoint, ped.hold_ref_point) <= 1e-6:
                return max(0, k - 1)
        logger.warning(
            "Single pedestrian '{}' hold_ref_point {} is not on its trajectory; "
            "proximity hold disabled (fail-open).",
            ped.id,
            ped.hold_ref_point,
        )
        return None

    def _tick_proximity_hold(self, runtime: SinglePedestrianRuntime) -> bool:
        """Hold the pedestrian at the curb until a robot approaches or the timeout elapses.

        The hold latches once the pedestrian first arrives at its designated hold waypoint and
        then keeps holding every step (independent of small physics drift away from the exact
        waypoint) until it releases. It releases when any robot is within
        ``hold_until_robot_within_m`` of ``hold_ref_point`` or, fail-open, once ``hold_timeout_s``
        (default :data:`DEFAULT_HOLD_TIMEOUT_S`) has elapsed.

        Returns:
            bool: ``True`` while the pedestrian should keep holding at the curb.
        """
        if runtime.hold_waypoint_index is None or runtime.proximity_hold_released:
            return False
        if runtime.waypoint_index != runtime.hold_waypoint_index:
            return False

        if not runtime.proximity_hold_engaged:
            pos = self.states.pos_of(runtime.ped_id)
            hold_waypoint = runtime.trajectory[runtime.hold_waypoint_index]
            if dist(pos, hold_waypoint) > self.goal_proximity_threshold:
                return False  # still walking to the curb; let normal trajectory logic run.
            runtime.proximity_hold_engaged = True

        timeout = runtime.definition.hold_timeout_s
        if timeout is None:
            timeout = DEFAULT_HOLD_TIMEOUT_S
        if runtime.proximity_hold_elapsed_s >= timeout:
            self._release_proximity_hold(runtime, "timeout")
            return False
        if self._robot_within_hold_radius(runtime):
            self._release_proximity_hold(runtime, "robot_proximity")
            return False

        runtime.proximity_hold_elapsed_s += self.time_step_s
        self._hold_position(runtime)
        return True

    def _release_proximity_hold(self, runtime: SinglePedestrianRuntime, reason: str) -> None:
        """Release the proximity hold and step the pedestrian off the curb into the crossing.

        Advancing the waypoint here (rather than waiting for the normal arrival check) makes the
        release the explicit "cross now" signal, so the pedestrian steps off even after physics
        drift has nudged it a little away from the exact curb waypoint during a long hold.
        """
        runtime.proximity_hold_released = True
        runtime.hold_released_by = reason
        self._advance_waypoint(runtime)
        logger.info(
            "Single pedestrian '{}' proximity hold released by {}.",
            runtime.definition.id,
            reason,
        )

    def _robot_within_hold_radius(self, runtime: SinglePedestrianRuntime) -> bool:
        """Return whether any robot is within the hold radius of the reference point.

        Returns:
            bool: ``True`` when a robot pose is within ``hold_until_robot_within_m`` of
            ``hold_ref_point``. Missing pose provider or empty poses read as "not yet"
            (the timeout still guarantees release).
        """
        ref_point = runtime.definition.hold_ref_point
        radius = runtime.definition.hold_until_robot_within_m
        if ref_point is None or radius is None:
            return False
        if self.robot_pose_provider is None:
            return False
        poses = self.robot_pose_provider()
        if not poses:
            return False
        return any(
            dist((position[0], position[1]), ref_point) <= radius for (position, _heading) in poses
        )

    def hold_release_reasons(self) -> dict[str, str | None]:
        """Report how each proximity-holding pedestrian was released, for episode metadata.

        Returns:
            dict[str, str | None]: Maps pedestrian id to ``robot_proximity``/``timeout`` once
            released, or ``None`` while still holding. Only includes pedestrians with a
            configured proximity hold.
        """
        return {
            runtime.definition.id: runtime.hold_released_by
            for runtime in self._runtimes
            if runtime.hold_waypoint_index is not None
        }

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
