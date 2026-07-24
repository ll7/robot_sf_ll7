"""Social-force planner-family implementation extracted from the SocNav facade."""

from math import atan2, pi

import numpy as np

from robot_sf.planner import socnav as _socnav
from robot_sf.sim.pedestrian_model_variants import _pairwise_social_force_kernel

SamplingPlannerAdapter = _socnav.SamplingPlannerAdapter
SocNavPlannerConfig = _socnav.SocNavPlannerConfig
SocNavPlannerPolicy = _socnav.SocNavPlannerPolicy
sf_forces = _socnav.sf_forces


class SocialForcePlannerAdapter(SamplingPlannerAdapter):
    """Social-force planner adapter using fast-pysf interaction forces."""

    _EPS = 1e-6

    def __init__(self, config: SocNavPlannerConfig | None = None) -> None:
        """Initialize the social-force adapter with optional configuration."""
        self.config = config or SocNavPlannerConfig()
        if sf_forces is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pysocialforce is required for SocialForcePlannerAdapter. "
                "Install the fast-pysf dependency."
            )

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
        if goal_dist < self.config.goal_tolerance:
            return np.zeros(2, dtype=float)

        dt = self._resolve_dt(observation)
        desired_speed = min(self.config.social_force_desired_speed, self.config.max_linear_speed)
        desired_speed = min(desired_speed, goal_dist / max(dt, self._EPS))
        goal_dir = to_goal / (goal_dist + self._EPS)
        desired_vel = goal_dir * desired_speed
        desired_force = (desired_vel - robot_vel) / max(self.config.social_force_tau, self._EPS)

        social_force = self._compute_social_force(robot_pos, robot_vel, ped_state, robot_heading)
        obstacle_force = self._compute_obstacle_force(
            observation, robot_pos, robot_heading, robot_vel, robot_state
        )
        interaction_force = self.config.social_force_repulsion_weight * (
            social_force + obstacle_force
        )

        total_force = self._clip_force(desired_force + interaction_force)
        velocity_world = robot_vel + total_force * dt
        speed = float(np.linalg.norm(velocity_world))
        if speed < self._EPS:
            return np.zeros(2, dtype=float)
        if speed > self.config.max_linear_speed:
            velocity_world = (
                velocity_world / (speed + self._EPS) * float(self.config.max_linear_speed)
            )
        return np.asarray(velocity_world, dtype=float)

    def plan(self, observation: dict) -> tuple[float, float]:
        """Compute (v, w) using social-force goal + interaction forces.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        robot_state, _goal_state, _ped_state = self._socnav_fields(observation)
        robot_heading = float(self._as_1d_float(robot_state.get("heading", [0.0]), pad=1)[0])
        desired_vel = self.plan_velocity_world(observation)
        speed = float(np.linalg.norm(desired_vel))
        if speed < self._EPS:
            return 0.0, 0.0

        desired_heading = atan2(desired_vel[1], desired_vel[0])
        heading_error = self._wrap_angle(desired_heading - robot_heading)
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
    ) -> np.ndarray:
        """Compute social-force repulsion from pedestrians.

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

        # Vectorized social-force broadcast (issue #5412). Each pedestrian
        # contributed via the scalar ``sf_forces.social_force_ped_ped`` kernel in
        # a Python loop; this evaluates the identical closed-form force law across
        # all pedestrians at once through the shared NumPy port
        # (``_pairwise_social_force_kernel``). The degenerate zero-difference
        # handling matches the scalar kernel, so non-finite inputs (which the
        # scalar loop swallowed via try/except -> continue) are masked out here:
        # such pairs map to a zero force and are excluded from the reduction.
        pos_diff = (robot_pos[np.newaxis, :] - ped_positions).astype(float)  # (M, 2)
        vel_diff = (robot_vel[np.newaxis, :] - ped_vel_world).astype(float)  # (M, 2)
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
        centers, radii = self._extract_obstacles_from_grid(observation, robot_pos, robot_heading)
        if centers.size == 0:
            return np.zeros(2, dtype=float)

        robot_radius = float(self._as_1d_float(robot_state.get("radius", [0.0]), pad=1)[0])
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
        raw_dist = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        obst_dist = np.maximum(raw_dist - ped_radius, 1e-5)
        finite = np.isfinite(obst_dist)
        if not np.any(finite):
            return np.zeros(2, dtype=float)
        diff_f = diff[finite]
        obst_dist_f = obst_dist[finite]
        der_potential = 1.0 / obst_dist_f**3
        grad = diff_f / obst_dist_f[:, np.newaxis]
        force = der_potential[:, np.newaxis] * grad
        total = np.sum(force, axis=0)
        return total * float(self.config.social_force_obstacle_factor)

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

        obstacle_mask = grid[channel_idx] >= float(self.config.social_force_obstacle_threshold)
        if not np.any(obstacle_mask):
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        indices = np.argwhere(obstacle_mask)
        if indices.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        origin = self._as_1d_float(meta.get("origin", [0.0, 0.0]), pad=2)
        centers = self._grid_cell_centers(indices, origin, resolution)
        use_ego = bool(self._as_1d_float(meta.get("use_ego_frame", [0.0]), pad=1)[0] > 0.5)
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


def make_social_force_policy(config: SocNavPlannerConfig | None = None) -> SocNavPlannerPolicy:
    """
    Convenience constructor for social-force-like planner policy.

    Returns:
        SocNavPlannerPolicy: Policy wrapping SocialForcePlannerAdapter.
    """

    return SocNavPlannerPolicy(adapter=SocialForcePlannerAdapter(config=config))
