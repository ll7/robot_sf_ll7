"""Opt-in anisotropic Gaussian human-cost local planner (issue #7603, parent #7319).

Implements one experimental local-planner comparator: a motion-aligned anisotropic
Gaussian pedestrian cost field and repulsive force core with a bounded target-following
step, exposed through the canonical :class:`LocalPlannerProtocol`. It is opt-in only
and does not change any default planner or release roster.

Plain-language summary:
- Computes motion-aligned anisotropic Gaussian cost and repulsive force fields around pedestrians.
- Velocity-aligned longitudinal axis with speed-dependent expansion and front/rear asymmetry.
- Deterministic limiting rule for stationary or near-zero-velocity pedestrians (isotropic Gaussian).
- Integrates attractive goal-following and static obstacle clearance with hard speed and rate predicates.
- Emits structured, versioned diagnostics and fails closed on invalid/non-finite inputs.
- Experimental comparator only; makes no human-behavior, safety, or superiority claims.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.nav.occupancy_grid_utils import ego_to_world
from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin

PLANNER_TYPE = "anisotropic_gaussian_cost"
DIAGNOSTICS_SCHEMA = "anisotropic_gaussian_cost.v1"

_AGGREGATION_MODES = frozenset({"max", "sum"})


@dataclass(frozen=True)
class AnisotropicGaussianCostConfig:
    """Immutable experimental configuration for the anisotropic Gaussian human-cost planner.

    All parameters, scale factors, cutoffs, weights, and rate limits are declared
    here; invalid values are rejected in ``__post_init__``.
    """

    amplitude: float = 1.0
    sigma_long_base_m: float = 0.8
    sigma_lat_base_m: float = 0.5
    velocity_scale_long: float = 0.5
    velocity_scale_lat: float = 0.0
    asymmetry_front_ratio: float = 1.5
    min_velocity_threshold_mps: float = 0.05
    stationary_sigma_m: float = 0.6
    mahalanobis_cutoff: float = 3.0
    cutoff_distance_m: float = 4.0
    aggregation_mode: Literal["max", "sum"] = "max"
    attractive_weight: float = 1.0
    repulsive_weight: float = 2.0
    static_obstacle_weight: float = 1.5
    static_obstacle_radius_m: float = 2.0
    force_saturation: float = 5.0
    look_ahead_min_m: float = 0.5
    look_ahead_max_m: float = 2.0
    look_ahead_gain: float = 0.8
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.2
    max_linear_rate: float = 0.8
    max_angular_rate: float = 1.5
    control_dt: float = 0.2
    numerical_epsilon: float = 1e-6
    obstacle_grid_threshold: float = 0.5
    obstacle_grid_max_points: int = 256
    obstacle_input_mode: str = "observation_contract"
    pedestrian_input_mode: str = "observation_contract"

    def __post_init__(self) -> None:
        """Reject invalid configuration values fail-closed."""
        self._validate_positive_fields()
        self._validate_non_negative_fields()
        self._validate_enum_and_discrete_fields()

    def _validate_positive_fields(self) -> None:
        positives = (
            ("amplitude", self.amplitude),
            ("sigma_long_base_m", self.sigma_long_base_m),
            ("sigma_lat_base_m", self.sigma_lat_base_m),
            ("asymmetry_front_ratio", self.asymmetry_front_ratio),
            ("min_velocity_threshold_mps", self.min_velocity_threshold_mps),
            ("stationary_sigma_m", self.stationary_sigma_m),
            ("mahalanobis_cutoff", self.mahalanobis_cutoff),
            ("cutoff_distance_m", self.cutoff_distance_m),
            ("attractive_weight", self.attractive_weight),
            ("repulsive_weight", self.repulsive_weight),
            ("static_obstacle_weight", self.static_obstacle_weight),
            ("static_obstacle_radius_m", self.static_obstacle_radius_m),
            ("force_saturation", self.force_saturation),
            ("look_ahead_min_m", self.look_ahead_min_m),
            ("look_ahead_max_m", self.look_ahead_max_m),
            ("look_ahead_gain", self.look_ahead_gain),
            ("max_linear_speed", self.max_linear_speed),
            ("max_angular_speed", self.max_angular_speed),
            ("max_linear_rate", self.max_linear_rate),
            ("max_angular_rate", self.max_angular_rate),
            ("control_dt", self.control_dt),
            ("numerical_epsilon", self.numerical_epsilon),
        )
        for name, value in positives:
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                msg = f"{name} must be a positive finite number, got {value}"
                raise ValueError(msg)

    def _validate_non_negative_fields(self) -> None:
        non_negatives = (
            ("velocity_scale_long", self.velocity_scale_long),
            ("velocity_scale_lat", self.velocity_scale_lat),
        )
        for name, value in non_negatives:
            if not math.isfinite(float(value)) or float(value) < 0.0:
                msg = f"{name} must be a non-negative finite number, got {value}"
                raise ValueError(msg)

    def _validate_enum_and_discrete_fields(self) -> None:
        if self.look_ahead_min_m > self.look_ahead_max_m:
            msg = (
                f"look_ahead_min_m ({self.look_ahead_min_m}) must not exceed "
                f"look_ahead_max_m ({self.look_ahead_max_m})"
            )
            raise ValueError(msg)

        if self.aggregation_mode not in _AGGREGATION_MODES:
            msg = f"aggregation_mode must be one of {sorted(_AGGREGATION_MODES)}, got {self.aggregation_mode!r}"
            raise ValueError(msg)

        if not math.isfinite(float(self.obstacle_grid_threshold)) or not (
            0.0 <= float(self.obstacle_grid_threshold) <= 1.0
        ):
            msg = f"obstacle_grid_threshold must be in [0, 1], got {self.obstacle_grid_threshold}"
            raise ValueError(msg)

        if (
            isinstance(self.obstacle_grid_max_points, bool)
            or not isinstance(self.obstacle_grid_max_points, int)
            or self.obstacle_grid_max_points <= 0
        ):
            msg = f"obstacle_grid_max_points must be a positive int, got {self.obstacle_grid_max_points}"
            raise ValueError(msg)

        if self.obstacle_input_mode not in {"observation_contract", "oracle"}:
            msg = f"obstacle_input_mode must be observation_contract or oracle, got {self.obstacle_input_mode!r}"
            raise ValueError(msg)

        if self.pedestrian_input_mode not in {"observation_contract", "oracle"}:
            msg = f"pedestrian_input_mode must be observation_contract or oracle, got {self.pedestrian_input_mode!r}"
            raise ValueError(msg)

    def digest(self) -> str:
        """Return a stable configuration digest for diagnostics.

        Returns:
            A 64-character lowercase SHA-256 hex digest of the config.
        """
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_anisotropic_gaussian_cost_config(
    cfg: dict[str, Any] | None,
) -> AnisotropicGaussianCostConfig:
    """Build the immutable planner config from an algorithm mapping.

    Parameters:
        cfg: Raw configuration dictionary or None.

    Returns:
        The validated AnisotropicGaussianCostConfig instance.
    """
    payload = cfg if isinstance(cfg, dict) else {}
    allowed = {field.name for field in fields(AnisotropicGaussianCostConfig)}
    filtered = {k: v for k, v in payload.items() if k in allowed}
    return AnisotropicGaussianCostConfig(**filtered)


def evaluate_anisotropic_gaussian_cost(
    query_points: np.ndarray,
    ped_positions: np.ndarray,
    ped_velocities: np.ndarray,
    config: AnisotropicGaussianCostConfig,
) -> np.ndarray:
    """Evaluate motion-aligned anisotropic Gaussian human cost at query points.

    Parameters:
        query_points: Array of shape ``(N, 2)`` containing 2D query coordinates.
        ped_positions: Array of shape ``(M, 2)`` containing pedestrian positions.
        ped_velocities: Array of shape ``(M, 2)`` containing pedestrian velocities.
        config: Validated :class:`AnisotropicGaussianCostConfig`.

    Returns:
        Array of shape ``(N,)`` containing aggregated costs at query points.
    """
    pts = np.asarray(query_points, dtype=float).reshape(-1, 2)
    n_pts = pts.shape[0]
    if n_pts == 0:
        return np.zeros(0, dtype=float)

    if not np.all(np.isfinite(pts)):
        msg = "query_points must contain finite values"
        raise ValueError(msg)

    peds_pos = np.asarray(ped_positions, dtype=float).reshape(-1, 2)
    peds_vel = np.asarray(ped_velocities, dtype=float).reshape(-1, 2)
    n_peds = peds_pos.shape[0]

    if n_peds == 0:
        return np.zeros(n_pts, dtype=float)

    if peds_pos.shape != peds_vel.shape:
        msg = f"ped_positions shape {peds_pos.shape} and ped_velocities shape {peds_vel.shape} must match"
        raise ValueError(msg)

    if not np.all(np.isfinite(peds_pos)) or not np.all(np.isfinite(peds_vel)):
        msg = "pedestrian positions and velocities must contain finite values"
        raise ValueError(msg)

    per_ped_costs = np.zeros((n_pts, n_peds), dtype=float)

    amp = float(config.amplitude)
    v_thresh = float(config.min_velocity_threshold_mps)
    stat_sigma = float(config.stationary_sigma_m)
    d_m_cut = float(config.mahalanobis_cutoff)
    dist_cut = float(config.cutoff_distance_m)
    sigma_long_0 = float(config.sigma_long_base_m)
    sigma_lat_0 = float(config.sigma_lat_base_m)
    v_scale_long = float(config.velocity_scale_long)
    v_scale_lat = float(config.velocity_scale_lat)
    asym_ratio = float(config.asymmetry_front_ratio)

    for j in range(n_peds):
        px, py = peds_pos[j]
        vx, vy = peds_vel[j]
        speed = math.hypot(vx, vy)

        dx = pts[:, 0] - px
        dy = pts[:, 1] - py
        dist = np.hypot(dx, dy)

        if speed < v_thresh:
            # Stationary isotropic limiting rule
            d_m = dist / stat_sigma
            mask = (dist <= dist_cut) & (d_m <= d_m_cut)
            per_ped_costs[mask, j] = amp * np.exp(-0.5 * (d_m[mask] ** 2))
        else:
            # Moving anisotropic rule
            cos_th = vx / speed
            sin_th = vy / speed

            d_long = dx * cos_th + dy * sin_th
            d_lat = -dx * sin_th + dy * cos_th

            sig_long = (sigma_long_0 + v_scale_long * speed) * np.where(
                d_long > 0.0, asym_ratio, 1.0
            )
            sig_lat = sigma_lat_0 + v_scale_lat * speed

            d_m_sq = (d_long / sig_long) ** 2 + (d_lat / sig_lat) ** 2
            mask = (dist <= dist_cut) & (d_m_sq <= d_m_cut**2)
            per_ped_costs[mask, j] = amp * np.exp(-0.5 * d_m_sq[mask])

    if config.aggregation_mode == "sum":
        return np.sum(per_ped_costs, axis=1)
    return np.max(per_ped_costs, axis=1)


def evaluate_anisotropic_repulsive_force(
    robot_pos: tuple[float, float],
    ped_positions: np.ndarray,
    ped_velocities: np.ndarray,
    config: AnisotropicGaussianCostConfig,
) -> tuple[float, float]:
    """Evaluate total repulsive force vector from pedestrians on the robot.

    Computes the analytic negative gradient of the anisotropic Gaussian cost
    summed across visible pedestrians.

    Parameters:
        robot_pos: 2D coordinates of the robot (x, y).
        ped_positions: Array of shape ``(M, 2)`` of pedestrian positions.
        ped_velocities: Array of shape ``(M, 2)`` of pedestrian velocities.
        config: Validated :class:`AnisotropicGaussianCostConfig`.

    Returns:
        ``(fx, fy)`` repulsive force components.
    """
    rx, ry = float(robot_pos[0]), float(robot_pos[1])
    if not (math.isfinite(rx) and math.isfinite(ry)):
        return (0.0, 0.0)

    peds_pos = np.asarray(ped_positions, dtype=float).reshape(-1, 2)
    peds_vel = np.asarray(ped_velocities, dtype=float).reshape(-1, 2)
    n_peds = peds_pos.shape[0]

    if n_peds == 0 or not np.all(np.isfinite(peds_pos)) or not np.all(np.isfinite(peds_vel)):
        return (0.0, 0.0)

    fx_total = 0.0
    fy_total = 0.0

    amp = float(config.amplitude)
    v_thresh = float(config.min_velocity_threshold_mps)
    stat_sigma = float(config.stationary_sigma_m)
    d_m_cut = float(config.mahalanobis_cutoff)
    dist_cut = float(config.cutoff_distance_m)
    sigma_long_0 = float(config.sigma_long_base_m)
    sigma_lat_0 = float(config.sigma_lat_base_m)
    v_scale_long = float(config.velocity_scale_long)
    v_scale_lat = float(config.velocity_scale_lat)
    asym_ratio = float(config.asymmetry_front_ratio)
    eps = float(config.numerical_epsilon)

    for j in range(n_peds):
        px, py = peds_pos[j]
        vx, vy = peds_vel[j]
        speed = math.hypot(vx, vy)

        dx = rx - px
        dy = ry - py
        dist = math.hypot(dx, dy)

        if dist < eps or dist > dist_cut:
            continue

        if speed < v_thresh:
            d_m = dist / stat_sigma
            if d_m > d_m_cut:
                continue
            cost = amp * math.exp(-0.5 * (d_m**2))
            inv_sig_sq = 1.0 / (stat_sigma**2)
            fx_total += cost * dx * inv_sig_sq
            fy_total += cost * dy * inv_sig_sq
        else:
            cos_th = vx / speed
            sin_th = vy / speed

            d_long = dx * cos_th + dy * sin_th
            d_lat = -dx * sin_th + dy * cos_th

            sig_long = (sigma_long_0 + v_scale_long * speed) * (asym_ratio if d_long > 0.0 else 1.0)
            sig_lat = sigma_lat_0 + v_scale_lat * speed

            d_m_sq = (d_long / sig_long) ** 2 + (d_lat / sig_lat) ** 2
            if d_m_sq > d_m_cut**2:
                continue

            cost = amp * math.exp(-0.5 * d_m_sq)
            grad_long = d_long / (sig_long**2)
            grad_lat = d_lat / (sig_lat**2)

            fx_total += cost * (grad_long * cos_th - grad_lat * sin_th)
            fy_total += cost * (grad_long * sin_th + grad_lat * cos_th)

    return (fx_total * config.repulsive_weight, fy_total * config.repulsive_weight)


def _extract_robot_pose(observation: dict[str, Any]) -> tuple[float, float, float] | None:
    """Extract validated (x, y, theta) robot pose.

    Returns:
        Tuple of (x, y, theta) or None if missing or non-finite.
    """
    if "robot_state" in observation and isinstance(
        observation["robot_state"], (tuple, list, np.ndarray)
    ):
        rs = observation["robot_state"]
        if len(rs) >= 3:
            rx, ry, rth = float(rs[0]), float(rs[1]), float(rs[2])
            if math.isfinite(rx) and math.isfinite(ry) and math.isfinite(rth):
                return (rx, ry, rth)

    if "robot_position" in observation and "robot_heading" in observation:
        rp = observation["robot_position"]
        rth = observation["robot_heading"]
        if len(rp) >= 2:
            rx, ry = float(rp[0]), float(rp[1])
            rth_f = float(rth)
            if math.isfinite(rx) and math.isfinite(ry) and math.isfinite(rth_f):
                return (rx, ry, rth_f)
    return None


def _extract_goal_position(observation: dict[str, Any]) -> tuple[float, float] | None:
    """Extract validated (x, y) goal coordinates.

    Returns:
        Tuple of (gx, gy) or None if missing or non-finite.
    """
    raw = observation.get("goal_position", observation.get("goal"))
    if raw is not None and isinstance(raw, (tuple, list, np.ndarray)) and len(raw) >= 2:
        gx, gy = float(raw[0]), float(raw[1])
        if math.isfinite(gx) and math.isfinite(gy):
            return (gx, gy)
    return None


def _extract_pedestrian_states(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract pedestrian position and velocity arrays.

    Returns:
        Tuple of (positions, velocities) as (M, 2) float arrays.
    """
    peds_pos_list: list[tuple[float, float]] = []
    peds_vel_list: list[tuple[float, float]] = []

    if "pedestrians" in observation and isinstance(observation["pedestrians"], dict):
        p_dict = observation["pedestrians"]
        positions = p_dict.get("positions") or p_dict.get("pos") or []
        velocities = p_dict.get("velocities") or p_dict.get("vel") or []
        for p, v in zip(positions, velocities, strict=False):
            if len(p) >= 2 and len(v) >= 2:
                px, py = float(p[0]), float(p[1])
                vx, vy = float(v[0]), float(v[1])
                if (
                    math.isfinite(px)
                    and math.isfinite(py)
                    and math.isfinite(vx)
                    and math.isfinite(vy)
                ):
                    peds_pos_list.append((px, py))
                    peds_vel_list.append((vx, vy))
    elif "pedestrian_positions" in observation:
        positions = observation["pedestrian_positions"]
        velocities = observation.get("pedestrian_velocities", np.zeros_like(positions))
        for p, v in zip(positions, velocities, strict=False):
            if len(p) >= 2 and len(v) >= 2:
                px, py = float(p[0]), float(p[1])
                vx, vy = float(v[0]), float(v[1])
                if (
                    math.isfinite(px)
                    and math.isfinite(py)
                    and math.isfinite(vx)
                    and math.isfinite(vy)
                ):
                    peds_pos_list.append((px, py))
                    peds_vel_list.append((vx, vy))

    pos_arr = np.array(peds_pos_list, dtype=float).reshape(-1, 2)
    vel_arr = np.array(peds_vel_list, dtype=float).reshape(-1, 2)
    return pos_arr, vel_arr


def _extract_obstacle_points(
    observation: dict[str, Any],
    robot_pose: tuple[float, float, float] | None,
    threshold: float,
    max_points: int,
) -> np.ndarray:
    """Extract static obstacle coordinates from observation or occupancy grid.

    Returns:
        Array of shape (K, 2) containing static obstacle coordinates.
    """
    obs_list: list[tuple[float, float]] = []
    if "obstacles" in observation and isinstance(
        observation["obstacles"], (list, tuple, np.ndarray)
    ):
        for o in observation["obstacles"]:
            if len(o) >= 2:
                ox, oy = float(o[0]), float(o[1])
                if math.isfinite(ox) and math.isfinite(oy):
                    obs_list.append((ox, oy))
    elif "occupancy_grid" in observation and robot_pose is not None:
        grid = observation["occupancy_grid"]
        if isinstance(grid, np.ndarray) and grid.ndim in {2, 3}:
            occ = grid if grid.ndim == 2 else grid[0]
            indices = np.argwhere(occ > threshold)
            if len(indices) > max_points:
                step = len(indices) // max_points
                indices = indices[::step][:max_points]

            res = float(observation.get("occupancy_grid_resolution", 0.1))
            rx, ry, rth = robot_pose
            for row, col in indices:
                ego_x = (col - occ.shape[1] / 2.0) * res
                ego_y = (row - occ.shape[0] / 2.0) * res
                wx, wy = ego_to_world(ego_x, ego_y, ((rx, ry), rth))
                obs_list.append((wx, wy))

    return np.array(obs_list, dtype=float).reshape(-1, 2)


class AnisotropicGaussianCostPlanner(OccupancyAwarePlannerMixin):
    """Opt-in local planner implementing motion-aligned anisotropic Gaussian human cost.

    Implements the canonical :class:`LocalPlannerProtocol`:
    ``plan`` / ``reset(*, seed=...)`` / ``diagnostics`` / ``close``.
    """

    def __init__(
        self,
        config: AnisotropicGaussianCostConfig | None = None,
        planner_type: str = PLANNER_TYPE,
    ) -> None:
        """Initialize the planner with optional configuration.

        Parameters:
            config: Optional configuration instance.
            planner_type: Identifier string for diagnostics.
        """
        super().__init__()
        self.config = config or AnisotropicGaussianCostConfig()
        self.planner_type = planner_type
        self._current_linear_speed: float = 0.0
        self._current_angular_speed: float = 0.0
        self._last_diagnostics: dict[str, Any] = {}
        self._closed: bool = False
        self.reset()

    def reset(self, *, seed: int | None = None) -> None:
        """Reset planner execution state for a new episode.

        Parameters:
            seed: Optional RNG seed (ignored for deterministic planning).
        """
        self._current_linear_speed = 0.0
        self._current_angular_speed = 0.0
        self._last_diagnostics = {
            "planner_type": self.planner_type,
            "diagnostics_schema": DIAGNOSTICS_SCHEMA,
            "config_digest": self.config.digest(),
            "status": "ok",
            "active_rate_limits": [],
            "pedestrian_count": 0,
            "max_pedestrian_cost": 0.0,
            "stop_requested": False,
            "linear_speed": 0.0,
            "angular_speed": 0.0,
        }

    def close(self) -> None:
        """Release held resources idempotently."""
        self._closed = True

    def diagnostics(self) -> dict[str, Any]:
        """Return execution diagnostics matching the canonical schema.

        Returns:
            Dictionary carrying status, config digest, and active limits.
        """
        return dict(self._last_diagnostics)

    def _compute_total_force(
        self,
        robot_pos: tuple[float, float],
        robot_theta: float,
        goal_pos: tuple[float, float],
        peds_pos: np.ndarray,
        peds_vel: np.ndarray,
        obs_points: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute combined attractive and repulsive force vector.

        Returns:
            Tuple of (fx, fy, goal_distance).
        """
        rx, ry = robot_pos
        gx, gy = goal_pos
        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)

        if dist < self.config.numerical_epsilon:
            return 0.0, 0.0, 0.0

        f_att_x = (dx / dist) * self.config.attractive_weight
        f_att_y = (dy / dist) * self.config.attractive_weight

        f_ped_x, f_ped_y = evaluate_anisotropic_repulsive_force(
            (rx, ry), peds_pos, peds_vel, self.config
        )

        f_obs_x = 0.0
        f_obs_y = 0.0
        for ox, oy in obs_points:
            odx = rx - ox
            ody = ry - oy
            odist = math.hypot(odx, ody)
            if self.config.numerical_epsilon < odist <= self.config.static_obstacle_radius_m:
                rep_mag = (
                    self.config.static_obstacle_weight
                    * ((1.0 / odist) - (1.0 / self.config.static_obstacle_radius_m))
                    / (odist**2)
                )
                f_obs_x += rep_mag * odx
                f_obs_y += rep_mag * ody

        fx = f_att_x + f_ped_x + f_obs_x
        fy = f_att_y + f_ped_y + f_obs_y
        f_mag = math.hypot(fx, fy)

        if f_mag > self.config.force_saturation:
            fx = (fx / f_mag) * self.config.force_saturation
            fy = (fy / f_mag) * self.config.force_saturation

        return fx, fy, dist

    def _apply_rate_limits(
        self,
        desired_linear: float,
        desired_angular: float,
        dt: float,
    ) -> tuple[float, float, list[str]]:
        """Clip speed commands against linear and angular rate limits.

        Returns:
            Tuple of (actual_linear, actual_angular, active_rate_limits).
        """
        max_d_lin = self.config.max_linear_rate * dt
        max_d_ang = self.config.max_angular_rate * dt

        delta_lin = desired_linear - self._current_linear_speed
        delta_ang = desired_angular - self._current_angular_speed

        active_limits: list[str] = []
        if abs(delta_lin) > max_d_lin:
            active_limits.append("linear_rate_limit")
            actual_lin = self._current_linear_speed + math.copysign(max_d_lin, delta_lin)
        else:
            actual_lin = desired_linear

        if abs(delta_ang) > max_d_ang:
            active_limits.append("angular_rate_limit")
            actual_ang = self._current_angular_speed + math.copysign(max_d_ang, delta_ang)
        else:
            actual_ang = desired_angular

        actual_lin = max(0.0, min(self.config.max_linear_speed, actual_lin))
        actual_ang = max(
            -self.config.max_angular_speed, min(self.config.max_angular_speed, actual_ang)
        )

        return actual_lin, actual_ang, active_limits

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Compute the next (linear_speed, angular_rate) command.

        Parameters:
            observation: Structured environment observation.

        Returns:
            Tuple of (linear_speed_mps, angular_rate_radps).
        """
        if self._closed:
            return (0.0, 0.0)

        robot_pose = _extract_robot_pose(observation)
        goal_pos = _extract_goal_position(observation)

        if robot_pose is None or goal_pos is None:
            self._last_diagnostics = {
                "planner_type": self.planner_type,
                "diagnostics_schema": DIAGNOSTICS_SCHEMA,
                "config_digest": self.config.digest(),
                "status": "invalid",
                "active_rate_limits": [],
                "pedestrian_count": 0,
                "max_pedestrian_cost": 0.0,
                "stop_requested": True,
                "linear_speed": 0.0,
                "angular_speed": 0.0,
                "reason": "missing_required_robot_pose_or_goal",
            }
            return (0.0, 0.0)

        rx, ry, rth = robot_pose
        peds_pos, peds_vel = _extract_pedestrian_states(observation)
        obs_points = _extract_obstacle_points(
            observation,
            robot_pose,
            self.config.obstacle_grid_threshold,
            self.config.obstacle_grid_max_points,
        )

        fx, fy, dist_goal = self._compute_total_force(
            (rx, ry), rth, goal_pos, peds_pos, peds_vel, obs_points
        )

        if dist_goal < self.config.numerical_epsilon:
            self._current_linear_speed = 0.0
            self._current_angular_speed = 0.0
            self._last_diagnostics = {
                "planner_type": self.planner_type,
                "diagnostics_schema": DIAGNOSTICS_SCHEMA,
                "config_digest": self.config.digest(),
                "status": "goal_reached",
                "active_rate_limits": [],
                "pedestrian_count": len(peds_pos),
                "max_pedestrian_cost": 0.0,
                "stop_requested": False,
                "linear_speed": 0.0,
                "angular_speed": 0.0,
            }
            return (0.0, 0.0)

        f_mag = math.hypot(fx, fy)
        desired_heading = rth if f_mag < self.config.numerical_epsilon else math.atan2(fy, fx)
        heading_error = wrap_angle_pi(desired_heading - rth)

        desired_linear = min(self.config.max_linear_speed, self.config.look_ahead_gain * dist_goal)
        if abs(heading_error) > math.pi / 2.0:
            desired_linear = max(0.0, desired_linear * math.cos(heading_error))

        desired_angular = max(
            -self.config.max_angular_speed,
            min(self.config.max_angular_speed, heading_error / self.config.control_dt),
        )

        dt = float(observation.get("sim.timestep", self.config.control_dt))
        if not math.isfinite(dt) or dt <= 0.0:
            dt = self.config.control_dt

        actual_linear, actual_angular, active_limits = self._apply_rate_limits(
            desired_linear, desired_angular, dt
        )

        if len(peds_pos) > 0:
            ped_costs = evaluate_anisotropic_gaussian_cost(
                np.array([[rx, ry]]), peds_pos, peds_vel, self.config
            )
            max_ped_cost = float(np.max(ped_costs))
        else:
            max_ped_cost = 0.0

        self._current_linear_speed = actual_linear
        self._current_angular_speed = actual_angular

        self._last_diagnostics = {
            "planner_type": self.planner_type,
            "diagnostics_schema": DIAGNOSTICS_SCHEMA,
            "config_digest": self.config.digest(),
            "status": "ok",
            "active_rate_limits": active_limits,
            "pedestrian_count": len(peds_pos),
            "max_pedestrian_cost": max_ped_cost,
            "stop_requested": False,
            "linear_speed": actual_linear,
            "angular_speed": actual_angular,
        }

        return (actual_linear, actual_angular)
