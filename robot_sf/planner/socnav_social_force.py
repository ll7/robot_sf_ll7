"""Social-force planner-family implementation extracted from the SocNav facade."""

from math import atan2, pi
from typing import Any

import numpy as np
from pysocialforce.config import (
    LEGACY_SHIFTED_GRADIENT_V1,
    OBSTACLE_FORCE_DISTANCE_FLOOR,
    obstacle_force_law_metadata,
    resolve_obstacle_force_law,
)
from scipy import ndimage

from robot_sf.planner import socnav as _socnav
from robot_sf.planner.socnav_base import (
    SOCIAL_FORCE_GOAL_APPROACH_LEGACY_V1,
    SOCIAL_FORCE_GOAL_APPROACH_TERMINAL_V1,
    SOCIAL_FORCE_PED_LEGACY_KERNEL,
    SOCIAL_FORCE_PED_SURFACE_V3,
    SOCIAL_FORCE_PLANNER_LEGACY_V1,
    SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2,
    resolve_social_force_ped_version,
    resolve_social_force_planner_version,
)
from robot_sf.sim.pedestrian_model_variants import _pairwise_social_force_kernel

SamplingPlannerAdapter = _socnav.SamplingPlannerAdapter
SocNavPlannerConfig = _socnav.SocNavPlannerConfig
SocNavPlannerPolicy = _socnav.SocNavPlannerPolicy
sf_forces = _socnav.sf_forces

SOCIAL_FORCE_GOAL_APPROACH_METADATA_SCHEMA = "social_force_goal_approach_metadata.v1"

# Issue #9724: parameter derivation for the ``resolution_independent_v2``
# obstacle term (defaults live on ``SocNavPlannerConfig``).
#
# The goal ("driving") term is ``(v_des * e_goal - v) / tau``.  With
# v_des = 1.0 m/s and tau = 0.5 s it is at most v_des / tau = 2 m/s^2 (robot at
# rest) and about 0 at cruise.  Integrated through the relaxation, a constant
# interaction force F shifts the steady-state velocity by
# ``dv = tau * w * F`` with the repulsion weight w = 0.8, i.e. dv = 0.4 * F.
#
# The v2 obstacle term is one exponential repulsion per visible obstacle
# surface patch, ``F(d) = A * exp(-d / B)`` along the outward normal, where d is
# the SURFACE distance (centre distance to the nearest point of the occupied
# region minus the robot radius from the observation).  With A = 5 m/s^2 and
# B = 0.6 m:
#   d = 0.0 m -> w*F = 4.0 m/s^2 (2x the largest goal force: a robot driving
#                straight at a wall at v_des is stopped and turned, dv = 2 m/s)
#   d = 0.5 m -> w*F = 1.74 (dv 0.87 m/s, the same order as v_des)
#   d = 1.0 m -> w*F = 0.76 (dv 0.38 m/s, still a clear steering bias)
#   d = 2.0 m -> w*F = 0.14 (dv 0.07 m/s, 7 % of the maximum goal force)
#   d = 3.0 m -> w*F = 0.03 (negligible)
#   d = 5.0 m -> w*F = 0.001
# So the wall term is comparable to the goal term within about 1-2 m of the
# robot surface and negligible beyond a few metres, independent of how finely
# the occupancy grid resolves the wall.  (The v1 per-cell sum reached 40-550
# at the same distances because every occupied cell added its own term.)

# Issue #9758: parameter derivation for the opt-in ``surface_v3`` pedestrian
# term (defaults live on ``SocNavPlannerConfig``).
#
# The legacy ped term evaluates the ped-ped kernel at CENTRE distance, so at
# the release radii (robot 1.0 m, pedestrian 0.4 m) a pedestrian at 1.4 m
# centre distance contributes about 0.09 against a goal force of about 2: the
# robot drives through standing pedestrians.  The v3 term is one exponential
# repulsion per pedestrian, ``F(d_s) = A * exp(-d_s / B)`` along the
# velocity-anisotropic interaction direction, where d_s is the SURFACE
# distance (centre distance minus both radii, floored at zero).  With
# A = 6 m/s^2 and B = 0.5 m (and w = 0.8 as above):
#   d_s = 0.0 m -> w*F = 4.8 (2.4x the largest goal force: contact is refused)
#   d_s = 0.5 m -> w*F = 1.77 (goal-force order half a metre before contact)
#   d_s = 1.0 m -> w*F = 0.65 (clear steering bias)
#   d_s = 2.0 m -> w*F = 0.09 (negligible, like the wall term at range)
# So the ped term exceeds the goal term before contact at release speeds and
# fades within about a metre of the robot surface.  The direction keeps the
# kernel's velocity anisotropy (``lambda * v_rel + dir``) but does not touch
# the kernel's angle wrap (issue #9764 owns that seam).


class SocialForcePlannerAdapter(SamplingPlannerAdapter):
    """Social-force planner adapter using fast-pysf interaction forces."""

    _EPS = 1e-6
    _TURN_HYSTERESIS_RAD = 5.0 * pi / 6.0  # 150 deg
    _BLOCKED_RELEASE_RAD = pi / 3.0  # 60 deg
    _BALANCE_FRACTION = 0.1
    _BALANCE_CREEP_FRACTION = 0.1

    def __init__(self, config: SocNavPlannerConfig | None = None) -> None:
        """Initialize the social-force adapter with optional configuration."""
        self.config = config or SocNavPlannerConfig()
        if sf_forces is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pysocialforce is required for SocialForcePlannerAdapter. "
                "Install the fast-pysf dependency."
            )
        self._obstacle_force_applied = False
        self._obstacle_force_runtime_parameters: dict[str, Any] = {}
        self._goal_approach_applied = False
        self._goal_approach_runtime_parameters: dict[str, Any] = {}
        self._last_turn_sign = 0.0
        self._avoid_side = 0.0
        self._blocked_latched = False
        self._target_speed = float("inf")
        self._filtered_target = None

    def reset(self, *, seed: int | None = None) -> None:
        """Reset episode-local obstacle-force application diagnostics."""
        del seed
        self._obstacle_force_applied = False
        self._obstacle_force_runtime_parameters = {}
        self._goal_approach_applied = False
        self._goal_approach_runtime_parameters = {}
        self._last_turn_sign = 0.0
        self._avoid_side = 0.0
        self._blocked_latched = False
        self._target_speed = float("inf")
        self._filtered_target = None

    def plan_velocity_world(self, observation: dict) -> np.ndarray:
        """Compute a world-frame translational velocity using the social-force model.

        Returns:
            np.ndarray: World-frame ``[vx, vy]`` translational velocity.
        """
        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        robot_speed = self._as_1d_float(robot_state.get("speed", [0.0, 0.0]), pad=2)
        linear_speed = float(robot_speed[0])
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        robot_vel = np.array([linear_speed * cos_h, linear_speed * sin_h], dtype=float)

        goal = np.asarray(goal_state.get("current", [0.0, 0.0]), dtype=float)[:2]
        to_goal = goal - robot_pos
        goal_dist = float(np.linalg.norm(to_goal))
        self._goal_approach_applied = False
        self._goal_approach_runtime_parameters = {}
        self._target_velocity = None
        if goal_dist < self.config.goal_tolerance:
            return np.zeros(2, dtype=float)

        dt = self._resolve_dt(observation)
        goal_approach = self._goal_approach_context(
            observation,
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            robot_state=robot_state,
            goal_state=goal_state,
            goal=goal,
            goal_dist=goal_dist,
        )
        desired_speed = min(self.config.social_force_desired_speed, self.config.max_linear_speed)
        desired_speed = min(desired_speed, goal_dist / max(dt, self._EPS))
        goal_dir = to_goal / (goal_dist + self._EPS)
        desired_vel = goal_dir * desired_speed
        desired_force = (desired_vel - robot_vel) / max(self.config.social_force_tau, self._EPS)

        # Passed positionally (not by keyword) so existing ``*_args`` test
        # doubles keep working without churn.
        social_force = self._compute_social_force(
            robot_pos,
            robot_vel,
            ped_state,
            robot_heading,
            float(self._as_1d_float(robot_state.get("radius", [0.0]), pad=1)[0]),
        )
        if goal_approach is None:
            obstacle_force = self._compute_obstacle_force(
                observation, robot_pos, robot_heading, robot_vel, robot_state
            )
        else:
            # The segment-clear check in ``_goal_approach_context`` is the
            # opt-in terminal controller's wall-avoidance guard.  Keep the
            # pedestrian force active, but do not let the legacy static-wall
            # gradient reintroduce the known goal-limit cycle.
            obstacle_force = np.zeros(2, dtype=float)
            self._goal_approach_applied = True
            self._obstacle_force_applied = False
        interaction_force = self.config.social_force_repulsion_weight * (
            social_force + obstacle_force
        )

        total_force = self._clip_force(desired_force + interaction_force)
        # Steady-state target of the relaxation, v* = v + tau * F: the velocity
        # the model is relaxing toward.  It does not depend on dt or on the
        # current velocity, so v2 steers by its direction (issue #9724).
        target = robot_vel + float(self.config.social_force_tau) * total_force
        # Low-pass the steering target with time constant tau.  The occupancy
        # grid is re-rasterised in the ego frame every step, so while the robot
        # turns in place the obstacle patches (and the small net vector near a
        # force balance) jump from step to step; unfiltered, the turn direction
        # flipped every few steps in front of a too-narrow doorway.
        previous = getattr(self, "_filtered_target", None)
        if previous is None:
            self._filtered_target = target
        else:
            alpha = dt / (float(self.config.social_force_tau) + dt)
            self._filtered_target = (1.0 - alpha) * previous + alpha * target
        self._target_velocity = self._filtered_target
        velocity_world = robot_vel + total_force * dt
        speed = float(np.linalg.norm(velocity_world))
        if speed < self._EPS:
            return np.zeros(2, dtype=float)
        speed_limit = self._speed_limit()
        if speed > speed_limit:
            velocity_world = velocity_world / (speed + self._EPS) * speed_limit
        if goal_approach is not None:
            self._target_velocity = None  # steer by the blended approach velocity
            approach_speed = min(
                float(self.config.social_force_goal_approach_max_speed),
                float(self.config.max_linear_speed),
                max(
                    0.0,
                    (goal_dist - float(self.config.social_force_goal_approach_stop_distance))
                    / max(dt, self._EPS),
                ),
            )
            approach_velocity = goal_dir * approach_speed
            velocity_world = 0.5 * velocity_world + 0.5 * approach_velocity
            speed = float(np.linalg.norm(velocity_world))
            if speed > approach_speed:
                velocity_world = velocity_world / (speed + self._EPS) * approach_speed
        return np.asarray(velocity_world, dtype=float)

    def _goal_approach_context(
        self,
        observation: dict,
        *,
        robot_pos: np.ndarray,
        robot_heading: float,
        robot_state: dict,
        goal_state: dict,
        goal: np.ndarray,
        goal_dist: float,
    ) -> bool | None:
        """Return whether the explicit terminal goal controller may take over.

        The correction is intentionally narrow: it applies only to the final
        waypoint (``goal.next`` is the route sentinel), within a bounded radius,
        and when occupancy cells leave a swept robot-radius corridor clear. A
        blocked segment always falls through to the historical obstacle force.
        """
        version = getattr(
            self.config,
            "social_force_goal_approach_version",
            SOCIAL_FORCE_GOAL_APPROACH_LEGACY_V1,
        )
        if version != SOCIAL_FORCE_GOAL_APPROACH_TERMINAL_V1:
            return None

        next_goal = self._as_1d_float(goal_state.get("next", [0.0, 0.0]), pad=2)[:2]
        if float(np.linalg.norm(next_goal)) > max(float(self.config.goal_tolerance), self._EPS):
            return None
        approach_radius = max(float(self.config.social_force_goal_approach_radius), 0.0)
        if goal_dist > approach_radius:
            return None

        # The correction is an occupancy-backed exception. Without a grid we
        # cannot establish that the terminal segment is free, so retain the
        # historical wall force (and its degraded-input semantics).
        if self._obstacle_grid_payload(observation) is None:
            return None
        centers, radii = self._extract_obstacles_from_grid(
            observation,
            robot_pos,
            robot_heading,
        )
        robot_radius = float(self._as_1d_float(robot_state.get("radius", [0.0]), pad=1)[0])
        clearance = max(float(self.config.social_force_goal_approach_clearance), 0.0)
        segment = goal - robot_pos
        segment_sq = float(np.dot(segment, segment))
        segment_clear = segment_sq > self._EPS
        if segment_clear:
            for center, obstacle_radius in zip(centers, radii, strict=False):
                projection = float(
                    np.clip(np.dot(center - robot_pos, segment) / segment_sq, 0.0, 1.0)
                )
                nearest = robot_pos + projection * segment
                required_clearance = robot_radius + float(obstacle_radius) + clearance
                if float(np.linalg.norm(center - nearest)) <= required_clearance:
                    segment_clear = False
                    break

        self._goal_approach_runtime_parameters = {
            "final_goal": True,
            "approach_radius_m": approach_radius,
            "stop_distance_m": float(self.config.social_force_goal_approach_stop_distance),
            "max_speed_mps": float(self.config.social_force_goal_approach_max_speed),
            "corridor_clearance_m": clearance,
            "obstacle_points_considered": int(centers.shape[0]),
            "segment_clear": segment_clear,
        }
        return True if segment_clear else None

    def plan(self, observation: dict) -> tuple[float, float]:
        """Compute (v, w) using social-force goal + interaction forces.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, goal_state, _ped_state = self._socnav_fields(observation)
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        desired_vel = self.plan_velocity_world(observation)
        speed = float(np.linalg.norm(desired_vel))
        if speed < self._EPS:
            return 0.0, 0.0

        steer_vel = desired_vel
        if self._planner_version() == SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2:
            target = getattr(self, "_target_velocity", None)
            if target is not None and float(np.linalg.norm(target)) > self._EPS:
                steer_vel = target
            self._target_speed = (
                float(np.linalg.norm(target)) if target is not None else float("inf")
            )
        desired_heading = atan2(steer_vel[1], steer_vel[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
        if self._planner_version() == SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2:
            robot_pos = np.asarray(robot_state.get("position", [0.0, 0.0]), dtype=float)[:2]
            to_goal = np.asarray(goal_state.get("current", [0.0, 0.0]), dtype=float)[:2] - robot_pos
            goal_ahead = (
                float(to_goal @ np.array([np.cos(robot_heading), np.sin(robot_heading)])) > 0.0
            )
            return self._unicycle_command_v2(speed, heading_error, goal_ahead=goal_ahead)
        angular = float(
            np.clip(
                self.config.angular_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            ),
        )
        linear = float(
            np.clip(
                speed * max(0.0, 1.0 - abs(heading_error) / pi),
                0.0,
                self.config.max_linear_speed,
            ),
        )
        return linear, angular

    def _unicycle_command_v2(
        self, speed: float, heading_error: float, *, goal_ahead: bool
    ) -> tuple[float, float]:
        """Map a desired world velocity to (v, w) for ``resolution_independent_v2``.

        ``heading_error`` is measured to the low-passed steady-state target
        ``v* = v + tau * F`` (see ``plan``); ``speed`` is the magnitude of the
        integrated velocity ``v + F * dt``.

        * Drive only the component of the desired velocity along the current
          heading (``speed * max(0, cos(error))``), capped at the speed limit.
          A sideways or backward desired velocity therefore turns the robot in
          place at the limited turn rate instead of orbiting at speed.
        * Blocked ahead: when the target points backwards (``|error| > 90
          deg``) while the current goal lies in front of the robot, an
          obstacle or pedestrian in front pushes harder than the goal pulls.
          A unicycle cannot back up, so the robot turns toward the side of the
          lateral component of the target (sign of ``cross(heading, v*)``)
          and then drives around the obstacle once the target is in front.
          Both the state and the side have hysteresis: the state is entered
          at 90 deg and released when the target is within 60 deg of the
          heading (or the goal falls behind); the side is kept, also while
          the target stays behind the robot, until the robot drives at half
          its speed limit or more.
        * Force balance: when ``|v*|`` is below 10 % of v_des the goal and
          interaction forces cancel and the target's direction is noise.  The
          robot keeps turning in its current direction and creeps forward at
          10 % of its speed limit (turn radius 0.1 m at the defaults), which
          moves it off the balance point.
        * Hysteresis: otherwise, when ``|error| > 150 deg`` keep the previous
          turn direction, so a target that flips around the robot's back does
          not reverse the turn every step.

        There is no hold state: no static scene keeps the robot still.  In a
        true dead end (a doorway narrower than the robot) the robot circles
        slowly near the force balance without turn-sign flips; before these
        rules it spun in place with a sign flip nearly every step.

        Timing: the integrated velocity is ``robot_vel + F * dt``, so from rest
        its magnitude, and how soon a turn produces forward motion, scale with
        the timestep.  The turn decisions use the target direction and
        magnitude, which do not depend on dt; the filter has time constant
        ``tau``.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        max_turn = float(self.config.max_angular_speed)
        backwards = abs(heading_error) > 0.5 * pi
        if backwards and goal_ahead:
            self._blocked_latched = True
        elif not goal_ahead or abs(heading_error) < self._BLOCKED_RELEASE_RAD:
            self._blocked_latched = False
        along_heading = max(0.0, float(np.cos(heading_error)))
        linear = float(np.clip(speed * along_heading, 0.0, self._speed_limit()))
        balanced = self._target_speed < self._BALANCE_FRACTION * max(
            float(self.config.social_force_desired_speed), self._EPS
        )
        if balanced:
            # Goal and interaction forces (nearly) cancel, so the direction of
            # the tiny net vector is noise.  Keep turning the way the robot
            # already turns instead of following it: no sign flips, and no
            # state in which a static scene keeps the robot still.
            angular = (self._avoid_side or self._last_turn_sign or 1.0) * max_turn
            # Creep forward slowly (radius v/w = 0.1 m at the defaults) so the
            # robot leaves the balance point instead of spinning on it.
            linear = max(linear, self._BALANCE_CREEP_FRACTION * self._speed_limit())
        elif self._blocked_latched or (backwards and self._avoid_side != 0.0):
            if self._avoid_side == 0.0:
                lateral = float(np.sin(heading_error))
                self._avoid_side = 1.0 if lateral >= 0.0 else -1.0
            angular = self._avoid_side * max_turn
        else:
            angular = float(
                np.clip(float(self.config.angular_gain) * heading_error, -max_turn, max_turn)
            )
            if abs(heading_error) > self._TURN_HYSTERESIS_RAD and self._last_turn_sign != 0.0:
                angular = self._last_turn_sign * abs(angular)
        if abs(angular) > self._EPS:
            self._last_turn_sign = 1.0 if angular > 0.0 else -1.0
        if not self._blocked_latched and linear >= 0.5 * self._speed_limit():
            # The robot is driving again: a later blockage may pick a new side.
            self._avoid_side = 0.0
        return linear, angular

    def _planner_version(self) -> str:
        """Return the resolved social-force planner version (issue #9724)."""
        return resolve_social_force_planner_version(
            getattr(self.config, "social_force_planner_version", None)
        )

    def _ped_version(self) -> str:
        """Return the resolved pedestrian-term version (issue #9758)."""
        return resolve_social_force_ped_version(
            getattr(self.config, "social_force_ped_version", None)
        )

    def _speed_limit(self) -> float:
        """Return the translational speed cap for the configured planner version.

        v1 caps only at ``max_linear_speed`` (3 m/s by default), so commands could
        reach three times the social-force desired speed.  v2 also respects the
        desired speed ``v_des``, as in the social-force model itself.

        Returns:
            float: Maximum commanded translational speed in m/s.
        """
        max_speed = float(self.config.max_linear_speed)
        if self._planner_version() == SOCIAL_FORCE_PLANNER_LEGACY_V1:
            return max_speed
        return max(0.0, min(max_speed, float(self.config.social_force_desired_speed)))

    def _resolve_dt(self, observation: dict) -> float:
        """Return the simulation timestep (fallback to config defaults)."""
        sim = observation.get("sim", {})
        timestep = self._as_1d_float(sim.get("timestep", [0.0]), pad=1)[0]
        if timestep <= 0.0:
            return float(self.config.social_force_tau)
        return float(timestep)

    @staticmethod
    def _rotate_velocities_to_world(velocities: np.ndarray, heading: float) -> np.ndarray:
        """Rotate ego-frame velocities into world coordinates.

        Returns:
            np.ndarray: Rotated velocity vectors in world coordinates.
        """
        if velocities.size == 0:
            return velocities
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        vx = cos_h * velocities[:, 0] - sin_h * velocities[:, 1]
        vy = sin_h * velocities[:, 0] + cos_h * velocities[:, 1]
        return np.stack([vx, vy], axis=1)

    def _compute_social_force(
        self,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        ped_state: dict,
        robot_heading: float,
        robot_radius: float | None = None,
    ) -> np.ndarray:
        """Compute social-force repulsion from pedestrians.

        The legacy kernel path evaluates the ped-ped law at centre distance.
        ``surface_v3`` (issue #9758) instead repels per pedestrian from the
        surface distance; see the module derivation.  ``robot_radius`` selects
        the release default (1.0 m) when absent or non-positive.

        Returns:
            np.ndarray: Combined social-force vector.
        """
        ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
        if ped_positions.ndim == 1:
            ped_positions = ped_positions.reshape(-1, 2)
        ped_count = int(self._as_1d_float(ped_state.get("count", [0]), pad=1)[0])
        ped_positions = ped_positions[:ped_count]
        if ped_positions.size == 0:
            return np.zeros(2, dtype=float)

        ped_velocities = np.asarray(ped_state.get("velocities", []), dtype=float)
        if ped_velocities.size == 0:
            ped_velocities = np.zeros_like(ped_positions, dtype=float)
        elif ped_velocities.ndim == 1:
            ped_velocities = ped_velocities.reshape(-1, 2)
        ped_velocities = ped_velocities[:ped_count]
        ped_vel_world = self._rotate_velocities_to_world(ped_velocities, robot_heading)

        if self._ped_version() == SOCIAL_FORCE_PED_SURFACE_V3:
            return self._compute_ped_surface_v3(
                robot_pos, robot_vel, ped_positions, ped_vel_world, ped_state, robot_radius
            )

        # Vectorized social-force broadcast (issue #5412). Each pedestrian
        # contributed via the scalar ``sf_forces.social_force_ped_ped`` kernel in
        # a Python loop; this evaluates the identical closed-form force law across
        # all pedestrians at once through the shared NumPy port
        # (``_pairwise_social_force_kernel``). The degenerate zero-difference
        # handling matches the scalar kernel, so non-finite inputs (which the
        # scalar loop swallowed via try/except -> continue) are masked out here:
        # such pairs map to a zero force and are excluded from the reduction.
        pos_diff = (robot_pos[np.newaxis, :] - ped_positions).astype(float)  # (M, 2)
        if self._planner_version() == SOCIAL_FORCE_PLANNER_LEGACY_V1:
            # Historical sign: ``v_self - v_other``.  The kernel expects
            # ``v_other - v_self`` (fast-pysf ``social_force`` pairs ``p_i - p_j``
            # with ``v_j - v_i``), so for an approaching pedestrian the
            # interaction direction points away from the pedestrian and the
            # force all but vanishes.  Kept only for trace reproducibility.
            vel_diff = (robot_vel[np.newaxis, :] - ped_vel_world).astype(float)  # (M, 2)
        else:
            # v2 (issue #9724): the kernel's own convention, ``v_j - v_i``.
            vel_diff = (ped_vel_world - robot_vel[np.newaxis, :]).astype(float)  # (M, 2)
        forces = _pairwise_social_force_kernel(
            pos_diff,
            vel_diff,
            n=int(self.config.social_force_n),
            n_prime=int(self.config.social_force_n_prime),
            lambda_importance=float(self.config.social_force_lambda_importance),
            gamma=float(self.config.social_force_gamma),
        )
        finite_mask = np.isfinite(forces).all(axis=1)
        total = np.sum(forces[finite_mask], axis=0) if np.any(finite_mask) else np.zeros(2)
        return total * float(self.config.social_force_factor)

    def _compute_ped_surface_v3(
        self,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        ped_positions: np.ndarray,
        ped_vel_world: np.ndarray,
        ped_state: dict,
        robot_radius: float | None,
    ) -> np.ndarray:
        """Compute per-pedestrian surface-distance repulsion (issue #9758).

        One exponential term per pedestrian, ``F = A * exp(-d_s / B)`` along the
        velocity-anisotropic interaction direction, where ``d_s`` is the surface
        distance (centre distance minus both radii, floored at zero).  The
        direction mirrors the shared kernel (``lambda * v_rel + dir``,
        normalized; degenerate pairs contribute zero) but the kernel's angle
        wrap is untouched (issue #9764 owns that seam).

        Returns:
            np.ndarray: World-frame ped force, scaled so the historical
                ``social_force_factor`` default normalizes away (as the v2
                obstacle term normalizes ``obstacle_factor``).
        """
        radius = float(robot_radius) if robot_radius else 0.0
        if radius <= 0.0:
            radius = 1.0  # release robot radius; ped_state rarely carries it
        default_ped_radius = float(self.config.social_force_ped_v3_default_ped_radius)
        # Nested observations carry a scalar ``radius``; the flat form carries
        # ``pedestrians_radius`` (see ``_socnav_fields``).  Either may be absent.
        raw_radius = ped_state.get("radii", ped_state.get("radius", default_ped_radius))
        radius_arr = np.atleast_1d(np.asarray(raw_radius, dtype=float)).reshape(-1)
        count = ped_positions.shape[0]
        if radius_arr.size == count:
            ped_radii = radius_arr
        else:
            scalar = float(radius_arr.flat[0]) if radius_arr.size > 0 else default_ped_radius
            ped_radii = np.full(count, scalar)
        ped_radii = np.where(
            np.isfinite(ped_radii) & (ped_radii > 0.0), ped_radii, default_ped_radius
        )
        strength = float(self.config.social_force_ped_v3_strength)
        length = max(float(self.config.social_force_ped_v3_length), self._EPS)
        lambda_importance = float(self.config.social_force_lambda_importance)

        diff = (np.asarray(robot_pos, dtype=float)[np.newaxis, :] - ped_positions).astype(float)
        centre = np.linalg.norm(diff, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            diff_dir = np.where(
                centre[:, np.newaxis] > 0.0,
                diff / np.maximum(centre, self._EPS)[:, np.newaxis],
                0.0,
            )
        vel_diff = (ped_vel_world - np.asarray(robot_vel, dtype=float)[np.newaxis, :]).astype(float)
        interaction = lambda_importance * vel_diff + diff_dir
        inter_length = np.linalg.norm(interaction, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            interaction_dir = np.where(
                inter_length[:, np.newaxis] > 0.0,
                interaction / np.maximum(inter_length, self._EPS)[:, np.newaxis],
                0.0,
            )
        surface = np.maximum(centre - (radius + ped_radii), 0.0)
        magnitudes = strength * np.exp(-surface / length)
        forces = magnitudes[:, np.newaxis] * interaction_dir
        finite_mask = np.isfinite(forces).all(axis=1)
        total = np.sum(forces[finite_mask], axis=0) if np.any(finite_mask) else np.zeros(2)
        # ``social_force_factor`` keeps its role as an on/off and ablation
        # scale; the v3 magnitude lives in ``ped_v3_strength``, so the
        # historical default of 5.1 is normalised away here.
        return total * (float(self.config.social_force_factor) / 5.1)

    def _compute_obstacle_force(
        self,
        observation: dict,
        robot_pos: np.ndarray,
        robot_heading: float,
        robot_vel: np.ndarray,
        robot_state: dict,
    ) -> np.ndarray:
        """Compute obstacle repulsion using occupancy-grid obstacle points.

        Returns:
            np.ndarray: Combined obstacle repulsion vector.
        """
        if self._planner_version() == SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2:
            return self._compute_obstacle_force_v2(
                observation, robot_pos, robot_heading, robot_state
            )
        law_version = resolve_obstacle_force_law(
            getattr(self.config, "social_force_obstacle_law", None)
        )
        centers, radii = self._extract_obstacles_from_grid(observation, robot_pos, robot_heading)
        if centers.size == 0:
            return np.zeros(2, dtype=float)

        robot_radius = float(self._as_1d_float(robot_state.get("radius", [0.0]), pad=1)[0])
        obstacle_factor = float(self.config.social_force_obstacle_factor)
        self._obstacle_force_runtime_parameters["robot_radius"] = robot_radius
        cell_radius = self._obstacle_force_runtime_parameters.get("cell_radius")
        if cell_radius is not None:
            self._obstacle_force_runtime_parameters["effective_offset"] = robot_radius + float(
                cell_radius
            )
        # Vectorized point-obstacle force broadcast (issue #5412). The scalar loop
        # built a degenerate single-point line ``(cx, cy, cx, cy)`` per obstacle
        # and called ``sf_forces.obstacle_force``. That degenerate line exercises
        # only the point-obstacle branch of the reference kernel, whose closed
        # form is ``der_potential * grad(dist)`` with ``der_potential =
        # 1/obst_dist**3`` and ``obst_dist = max(raw_dist - ped_radius, 1e-5)``.
        # Evaluating it across every obstacle at once changes the float reduction
        # order (vectorized sum vs scalar accumulation) and ``pow`` vs ``**``; the
        # residual stays at machine-epsilon relative error (see the #5412 parity
        # gate). The scalar kernel remains the numeric-parity reference. ``ortho``
        # was only consumed by the segment-intersection branches of the reference
        # kernel, which the degenerate point line never reaches, so it is dropped.
        ped_radius = robot_radius + np.asarray(radii, dtype=float)  # (M,)
        diff = (robot_pos[np.newaxis, :] - centers).astype(float)  # (M, 2)
        if law_version != LEGACY_SHIFTED_GRADIENT_V1:
            force = sf_forces.surface_distance_unit_normal_force_vectors(diff, ped_radius)
            self._obstacle_force_applied = self._obstacle_force_enabled()
            return np.sum(force, axis=0) * obstacle_factor

        raw_dist = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        obst_dist = np.maximum(raw_dist - ped_radius, OBSTACLE_FORCE_DISTANCE_FLOOR)
        finite = np.isfinite(obst_dist)
        if not np.any(finite):
            return np.zeros(2, dtype=float)
        diff_f = diff[finite]
        obst_dist_f = obst_dist[finite]
        der_potential = 1.0 / obst_dist_f**3
        grad = diff_f / obst_dist_f[:, np.newaxis]
        force = der_potential[:, np.newaxis] * grad
        total = np.sum(force, axis=0)
        self._obstacle_force_applied = self._obstacle_force_enabled()
        return total * obstacle_factor

    def _compute_obstacle_force_v2(
        self,
        observation: dict,
        robot_pos: np.ndarray,
        robot_heading: float,
        robot_state: dict,
    ) -> np.ndarray:
        """Resolution-independent obstacle repulsion (``resolution_independent_v2``).

        One exponential term per visible obstacle surface patch (see
        ``_visible_obstacle_points``), evaluated at the surface distance between
        the robot disc and the occupied region.  Parameters and their derivation
        are documented at the top of this module.

        Returns:
            np.ndarray: World-frame obstacle force (before the repulsion weight).
        """
        points, normals, distances = self._visible_obstacle_points(
            observation, robot_pos, robot_heading
        )
        robot_radius = float(self._as_1d_float(robot_state.get("radius", [0.0]), pad=1)[0])
        strength = float(self.config.social_force_obstacle_v2_strength)
        length = max(float(self.config.social_force_obstacle_v2_length), self._EPS)
        obstacle_factor = float(self.config.social_force_obstacle_factor)
        self._obstacle_force_runtime_parameters.update(
            {
                "robot_radius": robot_radius,
                "visible_obstacle_terms": int(points.shape[0]),
            }
        )
        if points.shape[0] == 0:
            return np.zeros(2, dtype=float)
        surface = np.maximum(distances - robot_radius, 0.0)
        magnitudes = strength * np.exp(-surface / length)
        force = np.sum(magnitudes[:, np.newaxis] * normals, axis=0)
        self._obstacle_force_applied = self._obstacle_force_enabled()
        # ``social_force_obstacle_factor`` keeps its role as an on/off and
        # ablation scale; the v2 magnitude lives in ``..._v2_strength``, so the
        # historical default of 10 is normalised away here.
        return force * (obstacle_factor / 10.0)

    def _visible_obstacle_points(
        self, observation: dict, robot_pos: np.ndarray, robot_heading: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select one nearest point per visible obstacle surface patch.

        Every occupied cell is treated as a filled square, and its nearest point
        to the robot centre is computed exactly, so the occupied region (not the
        cell count) defines the geometry.  Points are then chosen greedily,
        nearest first.  After choosing point ``q`` with bearing
        ``u = (q - robot) / |q - robot|``, the remaining points of the same
        8-connected occupied region as ``q`` are discarded when they lie on or
        behind the tangent line through ``q`` (``(p - q) . u >= -(tol +
        sin(15 deg) * |p - q|)``) or within
        ``social_force_obstacle_v2_min_separation_deg`` (30 deg) of ``u``.
        Points of other occupied regions are never discarded.

        ``tol`` is one cell diagonal.  The distance-proportional 15 deg part is
        needed because on a rasterised oblique wall the nearest staircase
        corner tilts ``u`` by a few degrees; with a resolution-only slack,
        walls at 5-15 deg to the grid axes produced two or three terms at every
        tested resolution (0.05-0.2 m).  It only acts inside one connected
        region, so it cannot drop a separate nearby obstacle.

        Consequences: a straight wall yields one term (like one fast-pysf
        segment); a convex obstacle one term; the two walls of a corridor or
        the legs of an inside corner one term each, so their lateral forces
        cancel on the centre line.  Every separate obstacle (a post in front
        of a wall, each column of a row) keeps its own term, as with
        fast-pysf's per-obstacle sum, up to
        ``social_force_obstacle_v2_max_terms`` (8) nearest terms; an obstacle
        hidden behind a nearer one also keeps its (weaker) term.  Merging
        happens only inside one connected region: surfaces within 15 deg of
        coplanar, or within 30 deg of bearing, count as one patch.  Note the
        attached-pillar case: a pillar that touches a wall rasterises into the
        same connected region, so when it lies within 30 deg of the wall's
        nearest point (or behind the tangent line there) it merges into that
        wall term instead of getting its own.  Tested cell sizes: 0.05-0.2 m.  Because
        selection is discrete, a patch term can appear or disappear as the
        robot moves; the exponential decay keeps such steps small except at
        contact range.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: World-frame nearest points,
            unit normals pointing from the obstacle to the robot, and centre
            distances (one row per selected term).
        """
        empty = (
            np.zeros((0, 2), dtype=float),
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=float),
        )
        payload = self._obstacle_grid_payload(observation)
        if payload is None:
            return empty
        grid, meta, channel_idx, resolution = payload
        origin = self._as_1d_float(meta.get("origin", [0.0, 0.0]), pad=2)
        use_ego = bool(self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)[0] > 0.5)
        self._obstacle_force_runtime_parameters.update(
            {
                "grid_resolution": float(resolution),
                "grid_origin": [float(origin[0]), float(origin[1])],
                "grid_frame": "ego" if use_ego else "world",
                "obstacle_channel_index": int(channel_idx),
            }
        )
        mask = np.asarray(grid[channel_idx]) >= float(self.config.social_force_obstacle_threshold)
        if not np.any(mask):
            return empty
        # Interior cells can never be the nearest point of the occupied region.
        padded = np.pad(mask, 1, constant_values=False)
        interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        indices = np.argwhere(mask & ~interior)
        if indices.size == 0:
            return empty
        labels, _count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        components = labels[indices[:, 0], indices[:, 1]]

        half = 0.5 * float(resolution)
        centers = self._grid_cell_centers(indices, origin, resolution)
        robot_grid = np.zeros(2, dtype=float) if use_ego else np.asarray(robot_pos, dtype=float)
        nearest = np.clip(robot_grid[np.newaxis, :], centers - half, centers + half)
        offsets = nearest - robot_grid[np.newaxis, :]
        dist = np.sqrt(np.einsum("ij,ij->i", offsets, offsets))
        max_range = float(self.config.social_force_obstacle_range)
        keep = np.isfinite(dist) & (dist <= max_range)
        if not np.any(keep):
            return empty
        offsets = offsets[keep]
        dist = dist[keep]
        components = components[keep]
        # A robot centre inside an occupied cell has no defined surface normal
        # from the clamped point; fall back to the cell-centre direction.
        inside = dist < self._EPS
        if np.any(inside):
            center_offsets = centers[keep][inside] - robot_grid[np.newaxis, :]
            offsets[inside] = center_offsets
        norms = np.maximum(np.linalg.norm(offsets, axis=1), self._EPS)
        directions = offsets / norms[:, np.newaxis]

        tolerance = np.sqrt(2.0) * float(resolution)
        same_surface_slack = float(np.sin(np.deg2rad(15.0)))
        cos_window = float(
            np.cos(np.deg2rad(float(self.config.social_force_obstacle_v2_min_separation_deg)))
        )
        max_terms = max(int(self.config.social_force_obstacle_v2_max_terms), 0)
        alive = np.ones(dist.shape[0], dtype=bool)
        chosen: list[int] = []
        while np.any(alive) and (max_terms == 0 or len(chosen) < max_terms):
            candidates = np.flatnonzero(alive)
            best = int(candidates[np.argmin(dist[candidates])])
            chosen.append(best)
            u = directions[best]
            rel = offsets - offsets[best]
            rel_norm = np.sqrt(np.einsum("ij,ij->i", rel, rel))
            same_surface = (components == components[best]) & (
                (rel @ u >= -(tolerance + same_surface_slack * rel_norm))
                | (directions @ u >= cos_window)
            )
            alive &= ~same_surface
            alive[best] = False

        sel_offsets = offsets[chosen]
        sel_dist = dist[chosen]
        sel_normals = -directions[chosen]
        if use_ego:
            cos_h = float(np.cos(robot_heading))
            sin_h = float(np.sin(robot_heading))
            rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=float)
            sel_offsets = sel_offsets @ rotation.T
            sel_normals = sel_normals @ rotation.T
        points = sel_offsets + np.asarray(robot_pos, dtype=float)[np.newaxis, :]
        return points, sel_normals, sel_dist

    def _obstacle_force_enabled(self) -> bool:
        """Return whether the configured obstacle-force factor can contribute."""
        try:
            return float(self.config.social_force_obstacle_factor) != 0.0
        except (AttributeError, TypeError, ValueError):
            return True

    @staticmethod
    def _grid_cell_centers(
        indices: np.ndarray, origin: np.ndarray, resolution: float
    ) -> np.ndarray:
        """Convert grid indices to grid-frame centers.

        Returns:
            np.ndarray: Grid-frame centers for the provided indices.
        """
        rows = indices[:, 0].astype(float)
        cols = indices[:, 1].astype(float)
        x = origin[0] + (cols + 0.5) * resolution
        y = origin[1] + (rows + 0.5) * resolution
        return np.stack([x, y], axis=1)

    @staticmethod
    def _ego_centers_to_world(
        centers: np.ndarray, robot_pos: np.ndarray, robot_heading: float
    ) -> np.ndarray:
        """Rotate/translate ego-frame centers into world coordinates.

        Returns:
            np.ndarray: World-space centers.
        """
        cos_h = float(np.cos(robot_heading))
        sin_h = float(np.sin(robot_heading))
        x_world = cos_h * centers[:, 0] - sin_h * centers[:, 1]
        y_world = sin_h * centers[:, 0] + cos_h * centers[:, 1]
        return np.stack([x_world, y_world], axis=1) + np.asarray(robot_pos, dtype=float)

    @staticmethod
    def _select_nearby_points(
        centers: np.ndarray,
        robot_pos: np.ndarray,
        max_range: float,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter centers by range and cap to the closest points.

        Returns:
            tuple[np.ndarray, np.ndarray]: Filtered centers and squared distances.
        """
        offsets = centers - np.asarray(robot_pos, dtype=float)
        dist_sq = np.einsum("ij,ij->i", offsets, offsets)
        keep = dist_sq <= max_range**2
        if not np.any(keep):
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
        centers = centers[keep]
        dist_sq = dist_sq[keep]
        if max_points > 0 and centers.shape[0] > max_points:
            order = np.argsort(dist_sq)[:max_points]
            centers = centers[order]
            dist_sq = dist_sq[order]
        return centers, dist_sq

    @staticmethod
    def _forward_lateral_components(
        centers: np.ndarray,
        robot_pos: np.ndarray,
        robot_heading: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project world-space obstacle centers onto robot-forward and lateral axes.

        Returns:
            tuple[np.ndarray, np.ndarray]: Forward and lateral distances.
        """
        forward = np.array([np.cos(robot_heading), np.sin(robot_heading)], dtype=float)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        offsets = centers - robot_pos[None, :]
        return offsets @ forward, offsets @ lateral

    def _coalesce_static_obstacle_points(
        self,
        *,
        centers: np.ndarray,
        radii: np.ndarray,
        robot_pos: np.ndarray,
        robot_heading: float,
        resolution: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce dense occupied-cell clouds into a smaller static obstacle set.

        Returns:
            tuple[np.ndarray, np.ndarray]: Coalesced obstacle centers and radii.
        """
        if centers.shape[0] <= 1:
            return centers, radii

        forward_dist, lateral_dist = self._forward_lateral_components(
            centers,
            robot_pos,
            robot_heading,
        )
        ahead_mask = forward_dist >= -resolution
        if np.any(ahead_mask):
            centers = centers[ahead_mask]
            radii = radii[ahead_mask]
            forward_dist = forward_dist[ahead_mask]
            lateral_dist = lateral_dist[ahead_mask]
        if centers.shape[0] <= 1:
            return centers, radii

        forward_bin = max(resolution * 2.0, float(self.config.orca_forward_probe_distance) * 0.5)
        lateral_bin = max(resolution * 2.0, float(self.config.orca_side_probe_offset) * 1.5)
        clusters: dict[tuple[int, int], list[int]] = {}
        for index, (forward_value, lateral_value) in enumerate(
            zip(forward_dist, lateral_dist, strict=False)
        ):
            key = (
                int(np.floor(forward_value / max(forward_bin, self._EPS))),
                int(np.floor(lateral_value / max(lateral_bin, self._EPS))),
            )
            clusters.setdefault(key, []).append(index)

        coalesced_centers: list[np.ndarray] = []
        coalesced_radii: list[float] = []
        for member_indices in clusters.values():
            cluster_centers = centers[member_indices]
            cluster_radii = radii[member_indices]
            center = np.mean(cluster_centers, axis=0)
            spread = (
                float(np.max(np.linalg.norm(cluster_centers - center[None, :], axis=1)))
                if cluster_centers.shape[0] > 1
                else 0.0
            )
            radius = float(np.max(cluster_radii) + spread)
            coalesced_centers.append(center)
            coalesced_radii.append(radius)

        result_centers = np.asarray(coalesced_centers, dtype=float)
        result_radii = np.asarray(coalesced_radii, dtype=float)
        if result_centers.shape[0] <= 1:
            return result_centers, result_radii

        dist_sq = np.einsum(
            "ij,ij->i", result_centers - robot_pos[None, :], result_centers - robot_pos[None, :]
        )
        max_points = max(int(self.config.orca_obstacle_max_points), 0)
        if max_points > 0 and result_centers.shape[0] > max_points:
            order = np.argsort(dist_sq)[:max_points]
            result_centers = result_centers[order]
            result_radii = result_radii[order]
        return result_centers, result_radii

    def _extract_obstacles_from_grid(
        self, observation: dict, robot_pos: np.ndarray, robot_heading: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract nearby obstacle centers from the occupancy grid.

        Returns:
            tuple[np.ndarray, np.ndarray]: World-space obstacle centers and per-point radii.
        """
        payload = self._obstacle_grid_payload(observation)
        if payload is None:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
        grid, meta, channel_idx, resolution = payload
        origin = self._as_1d_float(meta.get("origin", [0.0, 0.0]), pad=2)
        use_ego = bool(self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)[0] > 0.5)
        self._obstacle_force_runtime_parameters.update(
            {
                "grid_resolution": float(resolution),
                "grid_origin": [float(origin[0]), float(origin[1])],
                "grid_frame": "ego" if use_ego else "world",
                "obstacle_channel_index": int(channel_idx),
                "cell_radius": 0.5
                * np.sqrt(2.0)
                * float(resolution)
                * float(self.config.social_force_obstacle_radius_scale),
            }
        )

        obstacle_mask = grid[channel_idx] >= float(self.config.social_force_obstacle_threshold)
        if not np.any(obstacle_mask):
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        indices = np.argwhere(obstacle_mask)
        if indices.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        centers = self._grid_cell_centers(indices, origin, resolution)
        if use_ego:
            centers = self._ego_centers_to_world(centers, robot_pos, robot_heading)

        centers, _dist_sq = self._select_nearby_points(
            centers,
            robot_pos,
            float(self.config.social_force_obstacle_range),
            max(int(self.config.social_force_obstacle_max_points), 0),
        )
        if centers.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        base_radius = (
            0.5 * np.sqrt(2.0) * resolution * float(self.config.social_force_obstacle_radius_scale)
        )
        radii = np.full((centers.shape[0],), base_radius, dtype=float)
        return centers, radii

    def _clip_force(self, force: np.ndarray) -> np.ndarray:
        """Clip total force magnitude to avoid numerical spikes.

        Returns:
            np.ndarray: Clipped force vector.
        """
        if not self.config.social_force_clip_force:
            return force
        norm = float(np.linalg.norm(force))
        if norm < self._EPS or norm <= self.config.social_force_max_force:
            return force
        return force / (norm + self._EPS) * float(self.config.social_force_max_force)

    def diagnostics(self) -> dict[str, Any]:
        """Return execution diagnostics."""
        return {
            "planner_type": "SocialForcePlannerAdapter",
            "planner_version": self._planner_version()
            if getattr(self, "config", None) is not None
            else SOCIAL_FORCE_PLANNER_LEGACY_V1,
            "ped_version": self._ped_version()
            if getattr(self, "config", None) is not None
            else SOCIAL_FORCE_PED_LEGACY_KERNEL,
            "obstacle_force_law": self.obstacle_force_law_metadata(),
            "goal_approach": self.goal_approach_metadata(),
        }

    def goal_approach_metadata(self) -> dict[str, Any]:
        """Return explicit goal-approach version and runtime parameters."""
        config = getattr(self, "config", None)
        version = getattr(
            config,
            "social_force_goal_approach_version",
            SOCIAL_FORCE_GOAL_APPROACH_LEGACY_V1,
        )
        parameters: dict[str, Any] = {
            "approach_radius_m": float(getattr(config, "social_force_goal_approach_radius", 4.0)),
            "stop_distance_m": float(
                getattr(config, "social_force_goal_approach_stop_distance", 1.75)
            ),
            "max_speed_mps": float(getattr(config, "social_force_goal_approach_max_speed", 0.75)),
            "corridor_clearance_m": float(
                getattr(config, "social_force_goal_approach_clearance", 0.25)
            ),
        }
        parameters.update(getattr(self, "_goal_approach_runtime_parameters", {}))
        return {
            "schema_version": SOCIAL_FORCE_GOAL_APPROACH_METADATA_SCHEMA,
            "version": str(version),
            "enabled": str(version) != SOCIAL_FORCE_GOAL_APPROACH_LEGACY_V1,
            "applied": bool(getattr(self, "_goal_approach_applied", False)),
            "resolution_mode": getattr(
                config,
                "social_force_goal_approach_resolution_mode",
                "historical_unversioned",
            ),
            "parameters": parameters,
        }

    def obstacle_force_law_metadata(self) -> dict[str, Any]:
        """Return planner obstacle-law metadata without making an evidence claim."""
        config = getattr(self, "config", None)
        parameters: dict[str, Any] | None = None
        if config is not None:
            parameters = {
                "force_factor": float(config.social_force_obstacle_factor),
                "obstacle_threshold": float(config.social_force_obstacle_threshold),
                "obstacle_range": float(config.social_force_obstacle_range),
                "obstacle_max_points": int(config.social_force_obstacle_max_points),
                "radius_scale": float(config.social_force_obstacle_radius_scale),
                "distance_floor": OBSTACLE_FORCE_DISTANCE_FLOOR,
            }
            planner_version = resolve_social_force_planner_version(
                getattr(config, "social_force_planner_version", None)
            )
            if planner_version != SOCIAL_FORCE_PLANNER_LEGACY_V1:
                parameters.update(
                    {
                        "planner_version": planner_version,
                        "v2_strength": float(config.social_force_obstacle_v2_strength),
                        "v2_length": float(config.social_force_obstacle_v2_length),
                        "v2_max_terms": int(config.social_force_obstacle_v2_max_terms),
                        "v2_min_separation_deg": float(
                            config.social_force_obstacle_v2_min_separation_deg
                        ),
                    }
                )
            parameters.update(getattr(self, "_obstacle_force_runtime_parameters", {}))
        is_v2 = (
            config is not None
            and resolve_social_force_planner_version(
                getattr(config, "social_force_planner_version", None)
            )
            != SOCIAL_FORCE_PLANNER_LEGACY_V1
        )
        return obstacle_force_law_metadata(
            getattr(config, "social_force_obstacle_law", None),
            site="socnav_social_force",
            geometry_convention=(
                "occupancy_visible_nearest_points" if is_v2 else "occupancy_cell_centers"
            ),
            radius_convention=(
                "robot_radius_surface_distance"
                if is_v2
                else "cell_derived_radius_plus_robot_radius"
            ),
            enabled=self._obstacle_force_enabled() if config is not None else True,
            applied=bool(getattr(self, "_obstacle_force_applied", False)),
            resolution_mode=getattr(
                config,
                "obstacle_force_law_resolution_mode",
                None,
            ),
            parameters=parameters,
        )


def make_social_force_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for social-force-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping SocialForcePlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=SocialForcePlannerAdapter(config=config))


__all__ = [
    "SocialForcePlannerAdapter",
    "make_social_force_policy",
]
