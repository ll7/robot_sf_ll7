"""Development-only velocity sampler with executable static-obstacle checks.

Both variants forecast at the simulator timestep, irrespective of the inherited
``forecast_dt`` setting used by the original velocity sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, pi
from typing import Any, TypeVar

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi_closed
from robot_sf.planner.clearance_geometry import swept_pedestrian_clearance
from robot_sf.planner.socnav_sampling_v2 import (
    _goal_direction,
    _ObstacleClearance,
    _pedestrian_world_velocities,
    _positive,
)
from robot_sf.planner.velocity_sampler import (
    SamplerWeightsT,
    VelocitySamplerAdapter,
    VelocitySamplerWeights,
    _forecast,
    _risk,
    reachable_commands,
)


def _point_segment_distance_sq(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate squared distances with broadcasting over points and segments.

    Returns:
        np.ndarray: Squared distance for each broadcast pair.
    """
    span = b - a
    fraction = np.sum((points - a) * span, axis=-1) / np.maximum(
        np.sum(span * span, axis=-1), 1e-20
    )
    delta = points - (a + np.clip(fraction, 0.0, 1.0)[..., None] * span)
    return np.sum(delta * delta, axis=-1)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def swept_wall_clearance(points: np.ndarray, walls: np.ndarray, radius: float) -> np.ndarray:
    """Distance between each motion segment and the physical wall segments, less radius.

    Every motion segment includes its endpoints.  Intersections are detected between
    samples; an empty map gives infinite clearance. The simulator uses the same wall
    segments and circular robot footprint for its static-contact test.

    Returns:
        np.ndarray: Minimum surface clearance for each candidate and motion segment.
    """
    points = np.asarray(points, dtype=float)
    walls = np.asarray(walls, dtype=float).reshape(-1, 4)
    if not len(walls):
        return np.full((points.shape[0], points.shape[1] - 1), np.inf)
    p = points[:, :-1, None, :]
    p_next = points[:, 1:, None, :]
    q = walls[None, None, :, :2]
    q_next = walls[None, None, :, 2:]
    # Drop walls whose bounding boxes cannot be reached by any point in this batch.
    # This only reduces work and cannot remove the nearest wall within the footprint.
    p_min = points.min(axis=(0, 1)) - radius - 3.0
    p_max = points.max(axis=(0, 1)) + radius + 3.0
    keep = np.all(np.maximum(walls[:, :2], walls[:, 2:]) >= p_min, axis=1) & np.all(
        np.minimum(walls[:, :2], walls[:, 2:]) <= p_max, axis=1
    )
    if not keep.any():
        return np.full((points.shape[0], points.shape[1] - 1), np.inf)
    q = q[:, :, keep, :]
    q_next = q_next[:, :, keep, :]
    d2 = np.minimum(
        np.minimum(
            _point_segment_distance_sq(p, q, q_next), _point_segment_distance_sq(p_next, q, q_next)
        ),
        np.minimum(
            _point_segment_distance_sq(q, p, p_next), _point_segment_distance_sq(q_next, p, p_next)
        ),
    )
    r, s = p_next - p, q_next - q
    denominator = _cross(r, s)
    valid = np.abs(denominator) > 1e-12
    divisor = np.where(valid, denominator, 1.0)
    t = _cross(q - p, s) / divisor
    u = _cross(q - p, r) / divisor
    crossing = valid & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    d2 = np.where(crossing, 0.0, d2)
    return np.sqrt(np.min(d2, axis=-1)) - radius


def _braking_forecast(
    position: np.ndarray,
    heading: float,
    speed0: float,
    turn0: float,
    speeds: np.ndarray,
    turns: np.ndarray,
    dt: float,
    drive: Any,
) -> np.ndarray:
    """Follow a first reachable target and then brake at the actual drive limits.

    Returns:
        np.ndarray: Time-ordered robot positions for each first target.
    """
    steps = 1 + ceil(float(np.max(speeds)) / float(drive.max_linear_decel) / dt) + 1
    points = np.empty((len(speeds), steps + 1, 2))
    points[:, 0] = position
    angle = np.full(len(speeds), heading)
    speed = np.full(len(speeds), speed0)
    turn = np.full(len(speeds), turn0)
    for i in range(steps):
        next_speed = speeds if i == 0 else np.maximum(0.0, speed - drive.max_linear_decel * dt)
        next_turn = (
            turns
            if i == 0
            else turn + np.clip(-turn, -drive.max_angular_accel * dt, drive.max_angular_accel * dt)
        )
        next_angle = angle + (turn + next_turn) * 0.5 * dt
        mid = (angle + next_angle) * 0.5
        distance = (speed + next_speed) * 0.5 * dt
        points[:, i + 1, 0] = points[:, i, 0] + distance * np.cos(mid)
        points[:, i + 1, 1] = points[:, i, 1] + distance * np.sin(mid)
        speed, turn, angle = next_speed, next_turn, next_angle
    return points


class StaticSafeVelocitySamplerAdapter(VelocitySamplerAdapter[SamplerWeightsT]):
    """Opt-in sampler rejecting wall-intersecting forecasts and braking paths."""

    _pedestrian_braking_wall_margin_m: float | None = None

    def reset(self, *, seed: int | None = None) -> None:
        """Clear episode state and the count of steps without a wall-clear candidate."""
        super().reset(seed=seed)
        self._no_safe_static = 0

    def bind_env(self, env: Any) -> None:
        """Bind the drive and the simulator's actual static obstacle segments."""
        super().bind_env(env)
        simulator = getattr(env, "simulator", None)
        getter = getattr(simulator, "iter_obstacle_segments", None)
        if not callable(getter):
            raise ValueError("static-safe sampler requires simulator obstacle segments")
        walls = np.asarray(list(getter()), dtype=float).reshape(-1, 4)
        if not np.isfinite(walls).all():
            raise ValueError("static-safe sampler requires finite obstacle segments")
        self._walls = walls

    def _candidate_forecasts(  # noqa: PLR0913
        self, position, heading, speed0, turn0, speeds, turns, dt, drive, direction, peds, ped_vel
    ):
        del drive, direction, peds, ped_vel
        points, headings, times = _forecast(
            position, heading, speed0, turn0, speeds, turns, dt, self.weights.horizon_s, dt
        )
        return speeds, turns, points, headings, times

    def _speed_rewards(self, speeds: np.ndarray) -> np.ndarray:
        return speeds

    def _select_candidate(  # noqa: PLR0913
        self, costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
    ) -> int:
        del ped_clear, points, times, peds, ped_vel, direction
        if safe.any():
            return int(np.argmin(np.where(safe, costs, np.inf)))
        self._no_safe_static += 1
        # Prefer braking if the current candidate bank contains no wall-clear path.
        least_speed = speeds.min()
        options = np.where(np.isclose(speeds, least_speed), brake_clear.min(axis=1), -np.inf)
        return int(np.argmax(options))

    def _record_candidate(self, chosen: int, sides: np.ndarray, nearby: bool) -> None:
        side = int(sides[chosen])
        if nearby and side:
            self._side_switches += int(self._last_side not in (0, side))
            self._last_side = side
        elif not nearby:
            self._last_side = 0

    def _clear_braking_paths(
        self, safe, braking, brake_clear, dt, peds, ped_vel, contact_radius
    ) -> np.ndarray:
        margin = self._pedestrian_braking_wall_margin_m
        if margin is None:
            return safe
        wall_near = safe & (brake_clear.min(axis=1) <= margin)
        if wall_near.any():
            ped_brake = swept_pedestrian_clearance(
                braking[wall_near], np.arange(braking.shape[1]) * dt, peds, ped_vel, contact_radius
            )
            safe[wall_near] = ped_brake.min(axis=1) > 0.0
        return safe

    def plan(self, observation: dict) -> tuple[float, float]:
        """Prefer reachable commands whose rollout and braking path clear walls.

        Returns:
            tuple[float, float]: Linear and angular target velocity.
        """
        drive = getattr(self, "_velocity_drive", None)
        if drive is None:
            raise RuntimeError("velocity sampler requires bound drive limits")
        robot, goal, pedestrians = self._socnav_fields(observation)
        position = self._as_1d_float(robot["position"], pad=2)[:2]
        heading = float(self._as_1d_float(robot["heading"], pad=1)[0])
        target = self._as_1d_float(goal["current"], pad=2)[:2]
        current = self._as_1d_float(robot.get("speed", [0.0]), pad=1)
        speed0 = float(current[0])
        turn0 = (
            float(current[1])
            if current.size > 1
            else float(self._as_1d_float(robot.get("angular_velocity", [0.0]), pad=1)[0])
        )
        dt = self._simulation_timestep(observation)
        if not np.isfinite([*position, heading, speed0, turn0, dt]).all() or dt <= 0:
            raise ValueError("velocity sampler needs finite drive state and timestep")
        weights = self.weights
        speeds, turns = reachable_commands(
            speed0,
            turn0,
            dt,
            drive,
            speed_cap=min(float(self.config.max_linear_speed), drive.max_linear_speed),
            turn_cap=min(float(self.config.max_angular_speed), drive.max_angular_speed),
            speed_samples=weights.speed_samples,
            turn_samples=weights.turn_samples,
        )
        radius = _positive(robot.get("radius")) or drive.radius
        ped_radius = _positive(pedestrians.get("radius") if pedestrians else None) or 0.0
        raw = pedestrians.get("positions", []) if pedestrians else []
        peds = np.asarray(raw if raw is not None else [], dtype=float).reshape(-1, 2)
        count = (
            int(self._as_1d_float(pedestrians.get("count", [0]), pad=1)[0]) if pedestrians else 0
        )
        if count < 0 or count > len(peds) or not np.isfinite(peds[:count]).all():
            raise ValueError("pedestrian position observations are incomplete")
        peds = peds[:count]
        if len(peds):
            raw_vel = pedestrians.get("velocities")
            if raw_vel is None:
                raise ValueError("pedestrian velocity observations are required")
            observed = np.asarray(raw_vel, dtype=float).reshape(-1, 2)
            if len(observed) < len(peds) or not np.isfinite(observed[: len(peds)]).all():
                raise ValueError("pedestrian velocity observations are incomplete")
        ped_vel = _pedestrian_world_velocities(self.config, pedestrians, peds, heading)
        direction, path_used = _goal_direction(self, position, target, radius)
        target_angle = atan2(direction[1], direction[0])
        speeds, turns, points, headings, times = self._candidate_forecasts(
            position, heading, speed0, turn0, speeds, turns, dt, drive, direction, peds, ped_vel
        )
        ped_clear = swept_pedestrian_clearance(points, times, peds, ped_vel, radius + ped_radius)
        wall_clear = swept_wall_clearance(points, self._walls, radius)
        braking = _braking_forecast(position, heading, speed0, turn0, speeds, turns, dt, drive)
        brake_clear = swept_wall_clearance(braking, self._walls, radius)
        safe = (wall_clear.min(axis=1) > 1e-5) & (brake_clear.min(axis=1) > 1e-5)
        safe = self._clear_braking_paths(
            safe, braking, brake_clear, dt, peds, ped_vel, radius + ped_radius
        )
        ped_cost = _risk(ped_clear, weights.clearance_scale)
        wall_cost = _risk(wall_clear, weights.clearance_scale)
        ped_cost = 0.5 * (ped_cost.max(axis=1) + ped_cost.mean(axis=1))
        wall_cost = 0.5 * (wall_cost.max(axis=1) + wall_cost.mean(axis=1))
        progress = (points[:, -1] - position) @ direction
        heading_error = np.arctan2(
            np.sin(headings[:, -1] - target_angle), np.cos(headings[:, -1] - target_angle)
        )
        desired_turn = np.clip(
            float(self.config.angular_gain) * wrap_angle_pi_closed(target_angle - heading),
            -drive.max_angular_speed,
            drive.max_angular_speed,
        )
        sides = np.sign(turns - desired_turn).astype(int)
        nearby = bool(len(peds) and np.min(np.linalg.norm(peds - position, axis=1)) < 5.0)
        switch = (
            ((sides != 0) & (self._last_side != 0) & (sides != self._last_side))
            if nearby
            else np.zeros(len(speeds), dtype=bool)
        )
        costs = (
            weights.pedestrian_weight * ped_cost
            + weights.obstacle_weight * wall_cost
            - weights.progress_weight * progress / weights.horizon_s
            + weights.heading_weight * np.abs(heading_error) / pi
            - weights.speed_weight * self._speed_rewards(speeds)
            + weights.switch_weight * switch
        )
        chosen = self._select_candidate(
            costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
        )
        self._record_candidate(chosen, sides, nearby)
        clearance = _ObstacleClearance(self, observation)
        self._steps += 1
        self._pedestrian_steps += int(bool(len(peds)))
        self._static_grid_steps += int(clearance.available)
        self._path_field_steps += int(path_used)
        return float(speeds[chosen]), float(turns[chosen])

    def diagnostics(self) -> dict[str, Any]:
        """Report runtime counters for this experimental variant.

        Returns:
            dict[str, Any]: Planner identity and counters.
        """
        result = super().diagnostics()
        result.update(
            planner_type="StaticSafeVelocitySamplerAdapter",
            implementation_mode="static_safe_velocity_sampling",
            no_safe_static=self._no_safe_static,
        )
        return result


@dataclass(frozen=True)
class MultiStageWeights(VelocitySamplerWeights):
    """Controls for the development multistage primitive bank."""

    stage_s: float = 0.6
    continuation_s: float = 0.3

    def __post_init__(self) -> None:
        """Validate stage durations alongside the inherited weights."""
        super().__post_init__()
        if (
            not np.isfinite([self.stage_s, self.continuation_s]).all()
            or self.stage_s <= 0
            or self.continuation_s < 0
        ):
            raise ValueError(
                "stage timing must be finite and nonnegative, with a positive first stage"
            )


MultiStageWeightsT = TypeVar("MultiStageWeightsT", bound=MultiStageWeights)


class MultiStageVelocitySamplerAdapter(StaticSafeVelocitySamplerAdapter[MultiStageWeightsT]):
    """Evaluate braking, turning and yielding with their later continuations."""

    def reset(self, *, seed: int | None = None) -> None:
        """Clear any committed manoeuvre and passing side at episode reset."""
        super().reset(seed=seed)
        self._active_profile = None
        self._locked_side = 0
        self._continued_steps = 0
        self._staged_choices = 0
        self._yield_choices = 0
        self._safety_side_changes = 0

    def _candidate_forecasts(  # noqa: PLR0913, C901
        self, position, heading, speed0, turn0, speeds, turns, dt, drive, direction, peds, ped_vel
    ):
        count_steps = max(1, ceil(self.weights.horizon_s / dt))
        stage_steps = max(1, round(self.weights.stage_s / dt))
        # A profile is v1, w1, v2, w2, steps in stage 1, stage-2 heading mode, kind.
        profiles = [(v, w, v, w, count_steps + 1, 0, 0) for v, w in zip(speeds, turns, strict=True)]
        vmax = min(float(self.config.max_linear_speed), drive.max_linear_speed)
        wmax = min(float(self.config.max_angular_speed), drive.max_angular_speed)
        goal_turn = np.clip(
            self.config.angular_gain
            * wrap_angle_pi_closed(atan2(direction[1], direction[0]) - heading),
            -wmax,
            wmax,
        )
        for first_turn in (-wmax, 0.0, wmax, goal_turn):
            for v1 in (0.0, 0.5 * vmax, vmax):
                for v2 in (0.65 * vmax, vmax):
                    for mode in (0, 1):
                        profiles.append((v1, first_turn, v2, 0.0, stage_steps, mode, 1))
        if len(peds):
            for first_turn in (-wmax, 0.0, wmax):
                for v2 in (0.65 * vmax, vmax):
                    profiles.append((0.0, first_turn, v2, 0.0, stage_steps, 2, 2))
        self._continuation_index = None
        if self._active_profile is not None:
            self._continuation_index = len(profiles)
            profiles.append(self._active_profile[0])
        self._profiles = profiles
        values = np.asarray(profiles, dtype=float)
        v1, w1, v2, w2 = (values[:, i] for i in range(4))
        stage = values[:, 4].astype(int)
        mode = values[:, 5].astype(int)
        n = len(profiles)
        points = np.empty((n, count_steps + 1, 2), dtype=float)
        headings = np.empty((n, count_steps + 1), dtype=float)
        points[:, 0] = position
        headings[:, 0] = heading
        speed = np.full(n, speed0)
        turn = np.full(n, turn0)
        direction_angle = atan2(direction[1], direction[0])
        near_index = int(np.argmin(np.linalg.norm(peds - position, axis=1))) if len(peds) else 0
        initial_speeds = initial_turns = None
        for i in range(count_steps):
            first = i < stage
            desired_v = np.where(first, v1, v2)
            desired_w = np.where(first, w1, w2)
            track_goal = ~first & (mode == 1)
            angle = headings[:, i]
            goal_error = np.arctan2(
                np.sin(direction_angle - angle), np.cos(direction_angle - angle)
            )
            desired_w = np.where(
                track_goal, np.clip(goal_error * self.config.angular_gain, -wmax, wmax), desired_w
            )
            if len(peds):
                velocity = ped_vel[near_index]
                norm = np.linalg.norm(velocity)
                movement = velocity / norm if norm > 0.1 else direction
                behind = peds[near_index] + velocity * (i * dt) - movement * (drive.radius + 0.7)
                delta = behind - points[:, i]
                bearing = np.arctan2(delta[:, 1], delta[:, 0])
                error = np.arctan2(np.sin(bearing - angle), np.cos(bearing - angle))
                track_behind = ~first & (mode == 2)
                desired_w = np.where(
                    track_behind, np.clip(error * self.config.angular_gain, -wmax, wmax), desired_w
                )
            next_speed = speed + np.clip(
                desired_v - speed, -drive.max_linear_decel * dt, drive.max_linear_accel * dt
            )
            next_turn = turn + np.clip(
                desired_w - turn, -drive.max_angular_accel * dt, drive.max_angular_accel * dt
            )
            next_angle = angle + (turn + next_turn) * 0.5 * dt
            midpoint = (angle + next_angle) * 0.5
            distance = (speed + next_speed) * 0.5 * dt
            points[:, i + 1, 0] = points[:, i, 0] + distance * np.cos(midpoint)
            points[:, i + 1, 1] = points[:, i, 1] + distance * np.sin(midpoint)
            headings[:, i + 1] = next_angle
            if i == 0:
                initial_speeds, initial_turns = next_speed.copy(), next_turn.copy()
            speed, turn = next_speed, next_turn
        self._terminal_speeds = speed
        return initial_speeds, initial_turns, points, headings, np.arange(count_steps + 1) * dt

    def _speed_rewards(self, speeds: np.ndarray) -> np.ndarray:
        del speeds
        return self._terminal_speeds

    def _select_candidate(  # noqa: PLR0913
        self, costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
    ) -> int:
        self._candidate_sides = np.zeros(len(costs), dtype=int)
        nearby = bool(len(peds) and np.min(np.linalg.norm(peds - points[0, 0], axis=1)) < 5.0)
        if nearby:
            near = int(np.argmin(np.linalg.norm(peds - points[0, 0], axis=1)))
            relative = points[:, -1] - (peds[near] + ped_vel[near] * times[-1])
            lateral = _cross(np.broadcast_to(direction, relative.shape), relative)
            self._candidate_sides = np.where(np.abs(lateral) > 0.1, np.sign(lateral), 0).astype(int)
        else:
            self._locked_side = 0
        if not safe.any():
            return super()._select_candidate(
                costs, safe, speeds, brake_clear, ped_clear, points, times, peds, ped_vel, direction
            )
        eligible = safe.copy()
        ped_safe = ped_clear.min(axis=1) > 0.0
        if self._locked_side and nearby:
            matching = safe & (
                (self._candidate_sides == self._locked_side) | (self._candidate_sides == 0)
            )
            if (matching & ped_safe).any() or (matching.any() and not (safe & ped_safe).any()):
                eligible = matching
            elif (safe & ped_safe).any():
                eligible = safe & ped_safe
        previous = self._continuation_index
        if (
            previous is not None
            and eligible[previous]
            and (ped_safe[previous] or not (eligible & ped_safe).any())
        ):
            self._continued_steps += 1
            return previous
        return int(np.argmin(np.where(eligible, costs, np.inf)))

    def _record_candidate(self, chosen: int, sides: np.ndarray, nearby: bool) -> None:
        del sides
        side = int(self._candidate_sides[chosen]) if nearby else 0
        if side:
            if self._locked_side and self._locked_side != side:
                self._side_switches += 1
                self._safety_side_changes += 1
            self._locked_side = side
            self._last_side = side
        elif not nearby:
            self._last_side = 0
        profile = self._profiles[chosen]
        kind = int(profile[6])
        if kind:
            remaining = max(0, int(profile[4]) - 1)
            if chosen == self._continuation_index and self._active_profile is not None:
                commitment = self._active_profile[1] - 1
            else:
                self._staged_choices += 1
                self._yield_choices += int(kind == 2)
                commitment = (
                    int(profile[4]) + round(self.weights.continuation_s / self._last_dt) - 1
                )
            continued = (*profile[:4], remaining, *profile[5:])
            self._active_profile = (continued, commitment) if commitment > 0 else None
        else:
            self._active_profile = None

    def plan(self, observation: dict) -> tuple[float, float]:
        """Plan using the current control interval for all forecast stages.

        Returns:
            tuple[float, float]: Linear and angular target velocity.
        """
        self._last_dt = self._simulation_timestep(observation)
        return super().plan(observation)

    def diagnostics(self) -> dict[str, Any]:
        """Report stage, continuation, and side-selection counters.

        Returns:
            dict[str, Any]: Planner identity and counters.
        """
        result = super().diagnostics()
        result.update(
            planner_type="MultiStageVelocitySamplerAdapter",
            implementation_mode="multistage_velocity_sampling",
            continued_steps=self._continued_steps,
            staged_choices=self._staged_choices,
            yield_choices=self._yield_choices,
            safety_side_changes=self._safety_side_changes,
        )
        return result
