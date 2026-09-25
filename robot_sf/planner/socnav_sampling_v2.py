"""Opt-in ``bounded_v2`` version of the in-repository sampling heuristic.

``algo=socnav_sampling`` runs :class:`~robot_sf.planner.socnav_base.SamplingPlannerAdapter`
with ``use_upstream=False``: a heuristic inspired by the SocNavBench sampling planner
(Biswas et al., vendored subset in ``third_party/socnavbench``).  The reference planner
rolls out dynamically feasible trajectories within the robot's velocity bounds and scores
each trajectory by its distance to obstacles along the whole path.  The ``legacy_v1``
heuristic deviates from that in three ways that drive the robot into walls
(issues #9727 and #9746):

1. The summed inverse-distance repulsion of a distant crowd can outvote the unit goal
   vector and reverse the heading.
2. The occupancy check samples a single centre line along the desired heading. It ignores
   the robot radius and the arc the robot actually drives while it turns.
3. Speed is ``clip(goal_distance * scale, 0, max_linear_speed)``. With a far goal the
   distance factor swamps the occupancy and heading scaling, and ``max_linear_speed``
   (3.0 m/s) exceeds the drive limit of the bound robot.

``bounded_v2`` corrects these while keeping the heuristic's structure (goal vector plus
pedestrian repulsion, a fan of candidate headings, occupancy-weighted choice, speed scaled
by heading error and occupancy):

* Pedestrians whose surface distance is within ``sampling_near_field_surface_distance``
  keep the full legacy repulsion. Only the far-field sum is capped at
  ``sampling_max_repulsion_ratio``.
* Like the reference planner, it samples (heading, speed) pairs and rolls each one out
  over ``sampling_horizon_s`` under the bound drive's limits: current speed, acceleration
  and braking, maximum speed and turn rate, with the heuristic's own heading controller.
  The robot footprint (its radius plus ``sampling_footprint_margin``) is swept along each
  rollout against the obstacle grid and against pedestrian discs.  The blocked fraction
  of the rollout is the occupancy penalty.
* Speed is ``min(v_cap, goal_distance)`` scaled by heading error, the sampled speed
  fraction and the free fraction, where ``v_cap`` is the smaller of ``max_linear_speed``
  and the bound drive's maximum speed.  Goal distance therefore only slows the robot on
  arrival, and a far goal can no longer swamp the occupancy scaling.

``sampling_braking_envelope`` (off in the release config) adds an enhancement that the
reference method does not have: a hard bound ``v**2 / (2 b) + v * dt + margin <= free``
so the robot can always brake to a stop before the nearest swept surface.
"""

from __future__ import annotations

from math import atan2, ceil, cos, inf, pi, sin, sqrt
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from robot_sf.common.math_utils import wrap_angle_pi_closed

_DEFAULT_DT = 0.1
_DEFAULT_DECEL = 1.0
_STOPPED_SPEED = 0.1
_ESCAPE_HEADINGS = 16


def _positive(value: Any) -> float | None:
    """Return ``value`` as a positive float, or ``None`` when absent or non-positive.

    Returns:
        float | None: Positive float value or ``None``.
    """
    try:
        arr = np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.isfinite(arr[0]) or arr[0] <= 0.0:
        return None
    return float(arr[0])


def stopping_distance(speed: float, decel: float, dt: float, margin: float) -> float:
    """Distance needed to stop from ``speed``: one control step of travel plus braking.

    Returns:
        float: ``v**2 / (2 b) + v * dt + margin``.
    """
    speed = max(0.0, float(speed))
    return speed * speed / (2.0 * decel) + speed * dt + margin


def braking_speed_limit(free_distance: float, decel: float, dt: float, margin: float) -> float:
    """Largest speed whose stopping distance fits within ``free_distance``.

    Returns:
        float: Non-negative speed bound in m/s.
    """
    room = float(free_distance) - margin
    if room <= 0.0:
        return 0.0
    return max(0.0, decel * (-dt + sqrt(dt * dt + 2.0 * room / decel)))


class _ObstacleClearance:
    """Vectorised clearance from world points to the nearest occupied grid cell."""

    def __init__(self, adapter: Any, observation: dict) -> None:
        """Build the clearance field from the observation's occupancy grid."""
        self.available = False
        payload = adapter._extract_grid_payload(observation)
        if payload is None:
            return
        grid, meta = payload
        if grid.ndim < 3:
            return
        channel = adapter._grid_channel_index(meta, "obstacles")
        if channel < 0:
            channel = adapter._grid_channel_index(meta, "combined")
        if channel < 0 or channel >= grid.shape[0]:
            return
        resolution = _positive(meta.get("resolution"))
        if resolution is None:
            return
        occupied = np.asarray(grid[channel], dtype=float) > 0.5
        self.resolution = resolution
        self.origin = adapter._as_1d_float(meta.get("origin", [0.0, 0.0]), pad=2)[:2]
        self.size = adapter._as_1d_float(meta.get("size", [0.0, 0.0]), pad=2)[:2]
        use_ego = adapter._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)[0] > 0.5
        pose = adapter._as_1d_float(meta.get("robot_pose", [0.0, 0.0, 0.0]), pad=3)
        self.use_ego = bool(use_ego)
        self.pose = pose
        self.shape = occupied.shape
        if occupied.any():
            # Distance from each cell centre to the nearest occupied cell centre,
            # minus half a cell so the value approximates distance to the cell edge.
            self.field = ndimage.distance_transform_edt(~occupied) * resolution
            self.field = self.field - 0.5 * resolution
        else:
            self.field = None
        self.available = True

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Return clearance for ``points`` (N, 2); off-grid points count as blocked.

        Returns:
            np.ndarray: Clearance in metres per point.
        """
        return self.evaluate(points)[0]

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the clearance lower bound, the cell-centre clearance and the cell id.

        Off-grid points get clearance 0 and cell id -1.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Per-point lower bound, cell value
            and flat cell index.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        count = points.shape[0]
        if not self.available:
            return np.full(count, inf), np.full(count, inf), np.full(count, -1)
        local = points
        if self.use_ego:
            c, s = cos(-self.pose[2]), sin(-self.pose[2])
            delta = points - self.pose[:2]
            local = np.column_stack(
                (delta[:, 0] * c - delta[:, 1] * s, delta[:, 0] * s + delta[:, 1] * c)
            )
        local = local - self.origin
        inside = (
            (local[:, 0] >= 0.0)
            & (local[:, 1] >= 0.0)
            & (local[:, 0] <= self.size[0])
            & (local[:, 1] <= self.size[1])
        )
        lower = np.zeros(count, dtype=float)
        value = np.zeros(count, dtype=float)
        cell = np.full(count, -1)
        inner = local[inside]
        cols = np.clip((inner[:, 0] / self.resolution).astype(int), 0, self.shape[1] - 1)
        rows = np.clip((inner[:, 1] / self.resolution).astype(int), 0, self.shape[0] - 1)
        cell[inside] = rows * self.shape[1] + cols
        if self.field is None:
            lower[inside] = inf
            value[inside] = inf
            return lower, value, cell
        centres = (np.column_stack((cols, rows)) + 0.5) * self.resolution
        value[inside] = self.field[rows, cols]
        # Lower bound on the true clearance: the cell-centre value minus the point's
        # offset from its cell centre (triangle inequality).
        lower[inside] = value[inside] - np.linalg.norm(inner - centres, axis=1)
        return lower, value, cell


def _rollout(
    start: np.ndarray,
    heading: float,
    speed0: float,
    target_heading: float,
    target_speed: float,
    horizon_s: float,
    dt: float,
    limits: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the unicycle forward under the drive's speed, acceleration and turn limits.

    The heading controller is the heuristic's own (``w = clip(gain * error)``); speed
    moves from the current speed toward ``target_speed`` at the drive's acceleration or
    braking limit.

    Returns:
        tuple[np.ndarray, np.ndarray]: (N, 2) positions and (N,) cumulative arc length.
    """
    gain, max_turn_rate, accel, decel = limits
    count = max(1, ceil(horizon_s / dt))
    points = np.empty((count, 2), dtype=float)
    travelled = np.empty(count, dtype=float)
    x, y, theta, v, s = float(start[0]), float(start[1]), float(heading), float(speed0), 0.0
    for idx in range(count):
        v += min(max(target_speed - v, -decel * dt), accel * dt)
        err = wrap_angle_pi_closed(target_heading - theta)
        theta += min(max(gain * err, -max_turn_rate), max_turn_rate) * dt
        x += v * dt * cos(theta)
        y += v * dt * sin(theta)
        s += v * dt
        points[idx] = (x, y)
        travelled[idx] = s
    return points, travelled


def _blocked(clear: np.ndarray, start: float, threshold: float) -> np.ndarray:
    """Return which samples put the footprint within ``threshold`` of a pedestrian.

    The threshold is absolute.  The only exception is a robot that already starts inside
    it: a sample that strictly increases the clearance over the start value is allowed,
    so the robot can back away, but it can never hold or lose clearance there.

    Returns:
        np.ndarray: Boolean mask over the samples.
    """
    return (clear < threshold) & (clear <= start)


def _blocked_grid(
    clearance: _ObstacleClearance, start: np.ndarray, points: np.ndarray, threshold: float
) -> np.ndarray:
    """Absolute-threshold blocking against the obstacle grid.

    A sample blocks when its clearance lower bound is below ``threshold``.  A robot that
    already starts inside the threshold may stay within its own grid cell or move to a
    cell with strictly more clearance; any other cell below the threshold blocks, so the
    robot cannot ratchet along or into a wall step by step.

    Returns:
        np.ndarray: Boolean mask over the samples.
    """
    lower, value, cell = clearance.evaluate(points)
    start_lower, start_value, start_cell = clearance.evaluate(start[np.newaxis, :])
    blocked = lower < threshold
    if start_lower[0] < threshold:
        same = (cell >= 0) & (cell == start_cell[0])
        blocked &= ~(same | ((cell >= 0) & (value > start_value[0])))
        if blocked.any():
            first = int(np.argmax(blocked))
            if same[:first].all():
                # Blocked as soon as the path leaves the start cell: no safe progress.
                blocked[:first] = True
    return blocked


def _first_blocked(
    start: np.ndarray,
    points: np.ndarray,
    clearance: _ObstacleClearance,
    peds: np.ndarray,
    ped_radius: float,
    threshold: float,
    ped_vel: np.ndarray,
    dt: float,
) -> int:
    """First sample at which the swept footprint comes within ``threshold`` of a surface.

    Returns:
        int: Index of the first blocked sample, or ``len(points)`` when none blocks.
    """
    blocked = np.zeros(points.shape[0], dtype=bool)
    if clearance.available:
        blocked |= _blocked_grid(clearance, start, points, threshold)
    if peds.size:
        # Pedestrians advance along their current velocity (``ped_vel``, world frame)
        # to each sample's time, so an approaching pedestrian blocks early.
        times = (np.arange(points.shape[0]) + 1.0) * dt
        predicted = peds[np.newaxis] + times[:, np.newaxis, np.newaxis] * ped_vel[np.newaxis]
        ped_clear = (
            np.min(np.linalg.norm(points[:, np.newaxis, :] - predicted, axis=2), axis=1)
            - ped_radius
        )
        ped_start = float(np.min(np.linalg.norm(peds - start, axis=1)) - ped_radius)
        blocked |= _blocked(ped_clear, ped_start, threshold)
    if not blocked.any():
        return points.shape[0]
    return int(np.argmax(blocked))


class GoalPathField:
    """Path distance to a goal around the static map, as in the reference planner.

    The SocNavBench sampling planner scores goal progress by fast-marching distance on the
    traversible map, not by straight-line distance.  This field does the same on a grid
    rasterised once from the bound map's obstacle segments: cells closer than the robot
    radius to an obstacle are not traversable, and a multi-source Dijkstra from the free
    cells around the goal gives the path distance of every other cell.
    """

    def __init__(
        self,
        segments: np.ndarray,
        goal: np.ndarray,
        robot_radius: float,
        resolution: float,
        goal_radius: float,
    ) -> None:
        """Rasterise ``segments`` and compute path distance to ``goal``."""
        self.resolution = float(resolution)
        segments = np.asarray(segments, dtype=float).reshape(-1, 4)
        points = np.vstack((segments[:, :2], segments[:, 2:], goal[np.newaxis]))
        self.origin = points.min(axis=0) - 2.0
        extent = points.max(axis=0) + 2.0 - self.origin
        cols, rows = (np.ceil(extent / self.resolution).astype(int) + 1).tolist()
        self.shape = (rows, cols)
        occupied = np.zeros(self.shape, dtype=bool)
        for x1, y1, x2, y2 in segments:
            count = max(2, ceil(np.hypot(x2 - x1, y2 - y1) / (0.5 * self.resolution)) + 1)
            xs = np.linspace(x1, x2, count)
            ys = np.linspace(y1, y2, count)
            c = np.clip(((xs - self.origin[0]) / self.resolution).astype(int), 0, cols - 1)
            r = np.clip(((ys - self.origin[1]) / self.resolution).astype(int), 0, rows - 1)
            occupied[r, c] = True
        clearance = (
            ndimage.distance_transform_edt(~occupied) * self.resolution
            if occupied.any()
            else np.full(self.shape, inf)
        )
        free = clearance >= robot_radius
        centres_x = self.origin[0] + (np.arange(cols) + 0.5) * self.resolution
        centres_y = self.origin[1] + (np.arange(rows) + 0.5) * self.resolution
        gx, gy = np.meshgrid(centres_x, centres_y)
        to_goal = np.hypot(gx - goal[0], gy - goal[1])
        sources = free & (to_goal <= goal_radius)
        self.distance = np.full(self.shape, inf)
        if sources.any():
            self.distance = self._dijkstra(free, sources, to_goal)

    def _dijkstra(self, free: np.ndarray, sources: np.ndarray, to_goal: np.ndarray) -> np.ndarray:
        """Multi-source shortest path over 8-connected free cells.

        Returns:
            np.ndarray: Path distance per cell (``inf`` when unreachable).
        """
        rows, cols = self.shape
        index = np.arange(rows * cols).reshape(self.shape)
        heads, tails, weights = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            r0, r1 = max(0, -dr), rows - max(0, dr)
            c0, c1 = max(0, -dc), cols - max(0, dc)
            a = free[r0:r1, c0:c1] & free[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            ia = index[r0:r1, c0:c1][a]
            ib = index[r0 + dr : r1 + dr, c0 + dc : c1 + dc][a]
            w = self.resolution * (sqrt(2.0) if dr and dc else 1.0)
            heads += [ia, ib]
            tails += [ib, ia]
            weights += [np.full(ia.size, w), np.full(ia.size, w)]
        virtual = rows * cols
        src = index[sources]
        heads.append(np.full(src.size, virtual))
        tails.append(src)
        weights.append(to_goal[sources])
        graph = csr_matrix(
            (np.concatenate(weights), (np.concatenate(heads), np.concatenate(tails))),
            shape=(virtual + 1, virtual + 1),
        )
        dist = dijkstra(graph, directed=True, indices=virtual)
        return np.asarray(dist[:virtual], dtype=float).reshape(self.shape)

    def _cell(self, point: np.ndarray) -> tuple[int, int] | None:
        """Return the (row, col) grid cell of ``point``, or ``None`` off the field.

        Returns:
            tuple[int, int] | None: Grid cell or ``None``.
        """
        col, row = ((np.asarray(point, dtype=float) - self.origin) / self.resolution).astype(int)
        if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
            return int(row), int(col)
        return None

    def waypoint(self, position: np.ndarray, lookahead: float) -> np.ndarray | None:
        """Follow steepest descent of the path distance for ``lookahead`` metres.

        Returns:
            np.ndarray | None: World point to steer toward, or ``None`` when the robot is
            off the field or no finite path distance is near it.
        """
        cell = self._cell(position)
        if cell is None:
            return None
        row, col = cell
        if not np.isfinite(self.distance[row, col]):
            # Near a wall the robot centre can sit inside the inflated band; start from
            # the best reachable cell within one robot step instead.
            reach = max(1, round(0.6 / self.resolution))
            r0, c0 = max(0, row - reach), max(0, col - reach)
            window = self.distance[r0 : row + reach + 1, c0 : col + reach + 1]
            if not np.isfinite(window).any():
                return None
            dr, dc = np.unravel_index(int(np.argmin(window)), window.shape)
            row, col = r0 + int(dr), c0 + int(dc)
        for _ in range(max(1, round(lookahead / self.resolution))):
            r0, c0 = max(0, row - 1), max(0, col - 1)
            window = self.distance[r0 : row + 2, c0 : col + 2]
            dr, dc = np.unravel_index(int(np.argmin(window)), window.shape)
            nxt = (r0 + int(dr), c0 + int(dc))
            if nxt == (row, col):
                break
            row, col = nxt
        return self.origin + (np.array([col, row], dtype=float) + 0.5) * self.resolution


def _goal_direction(adapter: Any, robot_pos: np.ndarray, goal: np.ndarray, radius: float):
    """Return the unit direction of goal progress, by path distance when a map is bound.

    Returns:
        tuple[np.ndarray, bool]: Direction vector and whether the path field was used.
    """
    to_goal = goal - robot_pos
    straight = to_goal / (np.linalg.norm(to_goal) + 1e-6)
    config = adapter.config
    segments = getattr(adapter, "_sampling_obstacle_segments", None)
    if not bool(config.sampling_path_distance) or segments is None or len(segments) == 0:
        return straight, False
    key = (round(float(goal[0]), 2), round(float(goal[1]), 2), round(float(radius), 3))
    cache = adapter.__dict__.setdefault("_sampling_path_fields", {})
    field = cache.get(key)
    if field is None:
        field = GoalPathField(
            segments,
            goal,
            radius,
            float(config.sampling_path_resolution),
            max(float(config.goal_tolerance), 1.5),
        )
        cache[key] = field
    waypoint = field.waypoint(robot_pos, float(config.sampling_path_lookahead))
    if waypoint is None:
        return straight, False
    delta = waypoint - robot_pos
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return straight, False
    return delta / norm, True


def _repulsion_direction(
    config: Any,
    robot_pos: np.ndarray,
    goal_dir: np.ndarray,
    peds: np.ndarray,
    robot_radius: float,
    ped_radius: float,
) -> np.ndarray:
    """Goal direction plus near-field (uncapped) and far-field (capped) repulsion.

    Returns:
        np.ndarray: Direction vector, normalised when non-degenerate.
    """
    base_vec = np.asarray(goal_dir, dtype=float)
    near = np.zeros(2, dtype=float)
    far = np.zeros(2, dtype=float)
    threshold = float(config.sampling_near_field_surface_distance)
    for ped in peds:
        delta = robot_pos - ped
        dist = np.linalg.norm(delta) + 1e-6
        term = delta / dist**2
        if dist - robot_radius - ped_radius <= threshold:
            near += term
        else:
            far += term
    weight = float(config.social_force_repulsion_weight)
    far = weight * far
    far_norm = float(np.linalg.norm(far))
    cap = float(config.sampling_max_repulsion_ratio)
    if far_norm > cap:
        far *= cap / far_norm
    repulse = weight * near + far
    if np.linalg.norm(repulse) > 1e-6:
        base_vec = base_vec + repulse
        if np.linalg.norm(base_vec) > 1e-6:
            base_vec = base_vec / np.linalg.norm(base_vec)
    return base_vec


def _pedestrian_world_velocities(
    config: Any, ped_state: dict, peds: np.ndarray, heading: float
) -> np.ndarray:
    """Return pedestrian velocities in the world frame (zeros when unavailable).

    Returns:
        np.ndarray: (M, 2) velocities aligned with ``peds``.
    """
    ped_vel = np.zeros_like(peds)
    raw_vel = ped_state.get("velocities") if ped_state else None
    if not bool(config.sampling_pedestrian_prediction) or raw_vel is None or not peds.size:
        return ped_vel
    vel = np.asarray(raw_vel, dtype=float).reshape(-1, 2)[: peds.shape[0]]
    if vel.shape != peds.shape or not np.isfinite(vel).all():
        return ped_vel
    # Observation velocities are in the robot frame; rotate them to world.
    c, s = cos(heading), sin(heading)
    return np.column_stack((c * vel[:, 0] - s * vel[:, 1], s * vel[:, 0] + c * vel[:, 1]))


def plan_bounded_v2(adapter: Any, observation: dict) -> tuple[float, float]:  # noqa: PLR0915
    """Compute the ``bounded_v2`` (v, w) command for ``SamplingPlannerAdapter``.

    Returns:
        tuple[float, float]: Linear and angular velocity command.
    """
    config = adapter.config
    robot_state, goal_state, ped_state = adapter._socnav_fields(observation)
    robot_pos = adapter._as_1d_float(robot_state["position"], pad=2)[:2]
    heading = float(adapter._as_1d_float(robot_state["heading"], pad=1)[0])
    goal = adapter._as_1d_float(goal_state["current"], pad=2)[:2]
    to_goal = goal - robot_pos
    distance = float(np.linalg.norm(to_goal))
    if distance < config.goal_tolerance:
        adapter._last_sampling_v2 = {"reason": "goal_reached"}
        return 0.0, 0.0

    limits = dict(getattr(adapter, "_sampling_drive_limits", {}) or {})
    robot_radius = _positive(robot_state.get("radius")) or limits.get("radius", 0.0)
    ped_radius = _positive(ped_state.get("radius") if ped_state else None) or 0.0
    sim = observation.get("sim", {}) or {}
    dt = _positive(sim.get("timestep") if isinstance(sim, dict) else None) or _DEFAULT_DT
    v_cap = min(float(config.max_linear_speed), limits.get("max_linear_speed", inf))
    decel = limits.get("max_linear_decel") or limits.get("max_linear_accel") or _DEFAULT_DECEL
    accel = limits.get("max_linear_accel") or _DEFAULT_DECEL

    peds = np.asarray(ped_state.get("positions", []) if ped_state else [], dtype=float)
    ped_count = int(adapter._as_1d_float(ped_state.get("count", [0]), pad=1)[0]) if ped_state else 0
    peds = peds.reshape(-1, 2)[:ped_count] if peds.size else np.zeros((0, 2), dtype=float)
    ped_vel = _pedestrian_world_velocities(config, ped_state, peds, heading)
    goal_dir, path_used = _goal_direction(adapter, robot_pos, goal, robot_radius)
    direction = _repulsion_direction(config, robot_pos, goal_dir, peds, robot_radius, ped_radius)
    base_angle = atan2(direction[1], direction[0])

    clearance = _ObstacleClearance(adapter, observation)
    threshold = robot_radius + float(config.sampling_footprint_margin)
    brake_margin = float(config.sampling_braking_margin)
    braking = bool(config.sampling_braking_envelope)
    sweep = float(config.occupancy_heading_sweep)
    count = max(1, int(config.sampling_heading_candidates))
    offsets = np.linspace(-sweep / 2, sweep / 2, count) if count > 1 else np.zeros(1)
    half_sweep = sweep / 2 if sweep > 0 else 1.0
    v_nominal = min(v_cap, distance)
    speed0 = float(adapter._as_1d_float(robot_state.get("speed", [0.0]), pad=1)[0])
    speed0 = min(max(speed0, 0.0), v_cap) if np.isfinite(speed0) else 0.0
    horizon_s = float(config.sampling_horizon_s)
    if braking:
        # Long enough to cover a stop from the drive maximum.
        horizon_s = max(horizon_s, v_cap / decel + 2.0 * dt)
    drive = (
        float(config.angular_gain),
        float(config.max_angular_speed),
        accel,
        decel,
    )

    best: dict[str, float] | None = None
    for offset in offsets:
        target = base_angle + float(offset)
        err = wrap_angle_pi_closed(target - heading)
        heading_speed = v_nominal * max(0.0, 1.0 - abs(err) / pi)
        for fraction in config.sampling_speed_fractions:
            speed = heading_speed * float(fraction)
            points, travelled = _rollout(
                robot_pos, heading, speed0, target, speed, horizon_s, dt, drive
            )
            first = _first_blocked(
                robot_pos, points, clearance, peds, ped_radius, threshold, ped_vel, dt
            )
            penalty = 1.0 - first / points.shape[0]
            free = inf if first >= points.shape[0] else (travelled[first - 1] if first else 0.0)
            cost = (
                float(config.occupancy_weight) * penalty
                + float(config.occupancy_angle_weight) * abs(float(offset)) / half_sweep
                + float(config.sampling_speed_weight) * (1.0 - float(fraction))
            )
            if best is None or cost < best["cost"]:
                best = {
                    "cost": cost,
                    "target": target,
                    "err": err,
                    "speed": speed,
                    "free": float(free),
                    "penalty": penalty,
                }

    if best is None:  # empty sampling_speed_fractions: nothing was sampled
        adapter._last_sampling_v2 = {"reason": "no_samples"}
        return 0.0, 0.0
    if best["penalty"] >= 1.0 or best["speed"] * (1.0 - best["penalty"]) < _STOPPED_SPEED * 0.1:
        return _all_blocked_command(
            adapter,
            best,
            base_angle,
            heading,
            speed0,
            robot_pos,
            clearance,
            (peds, ped_vel, ped_radius),
            threshold,
            (horizon_s, dt, drive, v_nominal),
        )
    linear = best["speed"] * (1.0 - best["penalty"])
    brake_limit = inf
    if braking:
        brake_limit = braking_speed_limit(best["free"], decel, dt, brake_margin)
        linear = min(linear, brake_limit)
    linear = float(np.clip(linear, 0.0, v_cap))
    angular = float(
        np.clip(
            float(config.angular_gain) * best["err"],
            -float(config.max_angular_speed),
            float(config.max_angular_speed),
        )
    )
    adapter._last_sampling_v2 = {
        "desired_heading": float(best["target"]),
        "base_heading": float(base_angle),
        "free_distance": float(best["free"]),
        "horizon_s": float(horizon_s),
        "penalty": float(best["penalty"]),
        "speed_cap": float(v_cap),
        "brake_limit": float(brake_limit),
        "robot_radius": float(robot_radius),
        "path_distance": bool(path_used),
    }
    return linear, angular


def _all_blocked_command(  # noqa: PLR0913
    adapter: Any,
    best: dict[str, float],
    base_angle: float,
    heading: float,
    speed0: float,
    robot_pos: np.ndarray,
    clearance: _ObstacleClearance,
    pedestrians: tuple[np.ndarray, np.ndarray, float],
    threshold: float,
    rollout: tuple[float, float, tuple[float, float, float, float], float],
) -> tuple[float, float]:
    """Command when every sample is blocked or the best sample does not move.

    A moving robot brakes straight (turning while it coasts would sweep the footprint
    sideways into the surface).  A stopped robot turns in place toward the heading
    around the full circle whose rollout stays clear longest, or holds still.

    Returns:
        tuple[float, float]: Linear and angular velocity command.
    """
    config = adapter.config
    decision = {
        "desired_heading": float(best["target"]),
        "base_heading": float(base_angle),
        "free_distance": 0.0,
        "penalty": 1.0,
        "all_blocked": True,
    }
    if speed0 > _STOPPED_SPEED:
        decision["reason"] = "brake_straight"
        adapter._last_sampling_v2 = decision
        return 0.0, 0.0
    horizon_s, dt, drive, v_nominal = rollout
    peds, ped_vel, ped_radius = pedestrians
    speed = max(v_nominal, _STOPPED_SPEED) * 0.5
    best_key, best_target = (0, 0.0), None
    # Offsets ordered by distance from the goal-and-repulsion heading, so ties keep the
    # heading closest to it.
    offsets = sorted(np.linspace(-pi, pi, _ESCAPE_HEADINGS, endpoint=False), key=abs)
    for offset in offsets:
        target = base_angle + float(offset)
        points, _ = _rollout(robot_pos, target, 0.0, target, speed, horizon_s, dt, drive)
        first = _first_blocked(
            robot_pos, points, clearance, peds, ped_radius, threshold, ped_vel, dt
        )
        if first > best_key[0]:
            best_key, best_target = (first, float(offset)), target
    if best_target is None:
        decision["reason"] = "hold"
        adapter._last_sampling_v2 = decision
        return 0.0, 0.0
    err = wrap_angle_pi_closed(best_target - heading)
    decision.update({"reason": "turn_in_place", "desired_heading": float(best_target)})
    adapter._last_sampling_v2 = decision
    rate = float(config.max_angular_speed)
    return 0.0, float(np.clip(float(config.angular_gain) * err, -rate, rate))


__all__ = ["GoalPathField", "braking_speed_limit", "plan_bounded_v2", "stopping_distance"]
