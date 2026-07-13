"""Local risk/potential surface contract for learned planner prototypes.

This module provides the Robot SF-native boundary for issue #1675: a learned
or hand-authored local model may produce a bounded ego-frame risk surface, and
existing local planners may consume that surface through the repository's
occupancy-grid metadata contract. It intentionally does not load external code,
weights, or datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from robot_sf.common.math_utils import wrap_angle_pi
from robot_sf.errors import RobotSfError
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter
from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin


class RiskSurfaceUnavailable(RobotSfError, ValueError):
    """Raised when a local risk surface cannot satisfy the runtime contract."""


@dataclass(frozen=True)
class LocalRiskSurfaceSpec:
    """Reviewable spatial contract for local learned risk or potential fields.

    The default frame is robot ego-frame: +x points forward, +y points left, and
    the robot sits at `(0, 0, heading=0)`. `origin` is the lower-left grid corner
    in that frame and `resolution` is metres per cell.
    """

    resolution: float = 0.25
    width: float = 4.0
    height: float = 4.0
    origin: tuple[float, float] | None = None
    frame: str = "ego"
    risk_threshold: float = 0.5
    producer_id: str = "deterministic_fixture_v0"

    def __post_init__(self) -> None:
        """Validate risk-surface dimensions and conventions."""
        if self.frame != "ego":
            raise RiskSurfaceUnavailable("local risk surfaces currently require frame='ego'")
        if self.resolution <= 0.0:
            raise RiskSurfaceUnavailable("risk surface resolution must be > 0")
        if self.width <= 0.0 or self.height <= 0.0:
            raise RiskSurfaceUnavailable("risk surface width and height must be > 0")
        if not 0.0 <= self.risk_threshold <= 1.0:
            raise RiskSurfaceUnavailable("risk_threshold must be in [0, 1]")
        if self.origin is not None:
            origin = np.asarray(self.origin, dtype=float)
            if origin.shape != (2,) or not np.all(np.isfinite(origin)):
                raise RiskSurfaceUnavailable("origin must contain two finite values")

    @property
    def grid_width(self) -> int:
        """Number of columns in the surface."""
        return int(np.ceil(self.width / self.resolution))

    @property
    def grid_height(self) -> int:
        """Number of rows in the surface."""
        return int(np.ceil(self.height / self.resolution))

    @property
    def grid_origin(self) -> tuple[float, float]:
        """Lower-left origin in the local risk-surface frame."""
        if self.origin is not None:
            return (float(self.origin[0]), float(self.origin[1]))
        return (-float(self.width) / 2.0, -float(self.height) / 2.0)


@dataclass(frozen=True)
class LocalRiskSurface:
    """Bounded local risk field with occupancy-compatible metadata."""

    values: np.ndarray
    spec: LocalRiskSurfaceSpec
    status: str = "available"

    def __post_init__(self) -> None:
        """Validate shape, finite values, and status semantics."""
        values = np.asarray(self.values, dtype=np.float32)
        if values.shape != (self.spec.grid_height, self.spec.grid_width):
            raise RiskSurfaceUnavailable("risk surface shape must match spec grid height and width")
        if not np.all(np.isfinite(values)):
            raise RiskSurfaceUnavailable("risk surface values must be finite")
        if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
            raise RiskSurfaceUnavailable("risk surface values must be normalized to [0, 1]")
        if self.status != "available":
            raise RiskSurfaceUnavailable("risk surface status must be 'available'")
        object.__setattr__(self, "values", values)

    def occupancy_grid(self) -> np.ndarray:
        """Return a three-channel occupancy-compatible risk payload.

        The same normalized risk values are exposed as obstacle and combined
        channels so existing occupancy-aware planners can consume the surface
        without a special-case code path. The pedestrian channel remains empty
        because this payload is a model output, not raw pedestrian state.
        """
        grid = np.zeros((3, self.spec.grid_height, self.spec.grid_width), dtype=np.float32)
        grid[0] = self.values
        grid[2] = self.values
        return grid

    def occupancy_meta(self) -> dict[str, Any]:
        """Return metadata compatible with `OccupancyAwarePlannerMixin`."""
        return {
            "origin": list(self.spec.grid_origin),
            "resolution": [float(self.spec.resolution)],
            "size": [float(self.spec.width), float(self.spec.height)],
            "use_ego_frame": [1.0],
            "center_on_robot": [1.0],
            "channel_indices": [0, 1, -1, 2],
            "robot_pose": [0.0, 0.0, 0.0],
            "risk_surface": {
                "producer_id": self.spec.producer_id,
                "status": self.status,
                "frame": self.spec.frame,
                "risk_threshold": float(self.spec.risk_threshold),
            },
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-compatible contract diagnostics for smoke reports."""
        return {
            "status": self.status,
            "producer_id": self.spec.producer_id,
            "frame": self.spec.frame,
            "resolution": float(self.spec.resolution),
            "size": [float(self.spec.width), float(self.spec.height)],
            "risk_threshold": float(self.spec.risk_threshold),
            "max_risk": float(np.max(self.values)),
            "mean_risk": float(np.mean(self.values)),
            "risk_cells_at_or_above_threshold": int(
                np.count_nonzero(self.values >= float(self.spec.risk_threshold))
            ),
        }


def _as_2d_points(values: Any, *, name: str) -> np.ndarray:
    """Normalize point-like values into an `(N, 2)` finite float array.

    Returns:
        np.ndarray: Finite point array with shape `(N, 2)`.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape((-1, 2))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RiskSurfaceUnavailable(f"{name} must be an array of 2D points")
    if not np.all(np.isfinite(arr)):
        raise RiskSurfaceUnavailable(f"{name} must contain finite values")
    return arr.astype(float, copy=False)


def _extract_robot_pose(observation: dict[str, Any]) -> tuple[tuple[float, float], float]:
    """Extract finite robot pose from a structured Robot SF observation.

    Returns:
        tuple[tuple[float, float], float]: Robot position and heading.
    """
    robot = observation.get("robot")
    if not isinstance(robot, dict):
        raise RiskSurfaceUnavailable("observation must contain structured robot state")
    if "position" not in robot or "heading" not in robot:
        raise RiskSurfaceUnavailable("robot position and heading are required")
    position = np.asarray(robot["position"], dtype=float).reshape(-1)
    heading = np.asarray(robot["heading"], dtype=float).reshape(-1)
    if position.size < 2 or heading.size < 1:
        raise RiskSurfaceUnavailable("robot position and heading are required")
    if not np.all(np.isfinite(position[:2])) or not np.isfinite(heading[0]):
        raise RiskSurfaceUnavailable("robot pose must contain finite values")
    return (float(position[0]), float(position[1])), float(heading[0])


def _pedestrian_positions(observation: dict[str, Any]) -> np.ndarray:
    """Extract pedestrian positions from a structured Robot SF observation.

    Returns:
        np.ndarray: Pedestrian positions with shape `(N, 2)`.
    """
    pedestrians = observation.get("pedestrians", {})
    if not isinstance(pedestrians, dict):
        raise RiskSurfaceUnavailable("pedestrians field must be a mapping")
    positions = _as_2d_points(pedestrians.get("positions", []), name="pedestrian positions")
    count_raw = np.asarray(pedestrians.get("count", [positions.shape[0]]), dtype=float).reshape(-1)
    count = max(int(count_raw[0]), 0) if count_raw.size else positions.shape[0]
    return positions[:count]


def deterministic_pedestrian_risk_surface(
    observation: dict[str, Any],
    spec: LocalRiskSurfaceSpec | None = None,
    *,
    pedestrian_sigma: float = 0.45,
    floor_risk: float = 0.0,
) -> LocalRiskSurface:
    """Produce a deterministic local risk surface from pedestrian positions.

    This is the smoke fixture and reference producer for the contract. A future
    learned model may replace the value computation, but it must preserve the
    same shape, frame, normalization, and fail-closed behavior.

    Returns:
        LocalRiskSurface: Normalized local ego-frame risk surface.
    """
    if not isinstance(observation, dict):
        raise RiskSurfaceUnavailable("observation must be a mapping")
    cfg = spec or LocalRiskSurfaceSpec()
    if pedestrian_sigma <= 0.0:
        raise RiskSurfaceUnavailable("pedestrian_sigma must be > 0")
    if not 0.0 <= floor_risk <= 1.0:
        raise RiskSurfaceUnavailable("floor_risk must be in [0, 1]")

    robot_pose = _extract_robot_pose(observation)
    ped_positions = _pedestrian_positions(observation)
    origin = np.asarray(cfg.grid_origin, dtype=float)
    xs = origin[0] + (np.arange(cfg.grid_width, dtype=float) + 0.5) * float(cfg.resolution)
    ys = origin[1] + (np.arange(cfg.grid_height, dtype=float) + 0.5) * float(cfg.resolution)
    xx, yy = np.meshgrid(xs, ys)
    values = np.full((cfg.grid_height, cfg.grid_width), float(floor_risk), dtype=np.float32)

    if ped_positions.shape[0] > 0:
        # Vectorized world_to_ego: apply rotation + translation to all pedestrians.
        robot_x, robot_y, theta = robot_pose[0][0], robot_pose[0][1], robot_pose[1]
        ped_wx = ped_positions[:, 0]  # (N,)
        ped_wy = ped_positions[:, 1]  # (N,)
        dx = ped_wx - robot_x
        dy = ped_wy - robot_y
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        ego_x = dx * cos_t - dy * sin_t  # (N,)
        ego_y = dx * sin_t + dy * cos_t  # (N,)
        # dist_sq shape: (N, grid_height, grid_width) then reduce max over pedestrians
        dist_sq = (xx[None, :, :] - ego_x[:, None, None]) ** 2 + (
            yy[None, :, :] - ego_y[:, None, None]
        ) ** 2
        bumps = np.exp(-0.5 * dist_sq / float(pedestrian_sigma) ** 2)
        values = np.maximum(values, bumps.astype(np.float32).max(axis=0))

    return LocalRiskSurface(values=np.clip(values, 0.0, 1.0), spec=cfg)


def attach_risk_surface_to_observation(
    observation: dict[str, Any],
    surface: LocalRiskSurface,
) -> dict[str, Any]:
    """Attach a local risk surface as an occupancy-grid payload.

    Returns:
        dict[str, Any]: Observation copy enriched with risk-surface grid metadata.
    """
    if not isinstance(observation, dict):
        raise RiskSurfaceUnavailable("observation must be a mapping")
    robot_position, robot_heading = _extract_robot_pose(observation)
    robot_pose = [robot_position[0], robot_position[1], robot_heading]
    enriched = dict(observation)
    meta = surface.occupancy_meta()
    if surface.spec.frame == "ego":
        meta["robot_pose"] = robot_pose
    diagnostics = surface.diagnostics()
    diagnostics["robot_pose"] = robot_pose
    enriched["occupancy_grid"] = surface.occupancy_grid()
    enriched["occupancy_grid_meta"] = meta
    enriched["local_risk_surface_diagnostics"] = diagnostics
    return enriched


def build_local_risk_surface_spec(cfg: dict[str, Any] | None) -> LocalRiskSurfaceSpec:
    """Build a local risk-surface spec from a YAML-style mapping.

    Returns:
        LocalRiskSurfaceSpec: Validated surface geometry and metadata.
    """
    payload = cfg if isinstance(cfg, dict) else {}
    allowed = {field.name for field in fields(LocalRiskSurfaceSpec)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    origin = filtered.get("origin")
    if origin is not None:
        try:
            origin_values = np.asarray(origin, dtype=float).reshape(-1)
            if origin_values.size != 2:
                raise ValueError("origin must contain two finite values")
            filtered["origin"] = (float(origin_values[0]), float(origin_values[1]))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RiskSurfaceUnavailable("origin must contain two finite values") from exc
    return LocalRiskSurfaceSpec(**filtered)


class RiskSurfacePlannerAdapter(OccupancyAwarePlannerMixin):
    """Adapter that lets an existing local planner consume a risk surface."""

    def __init__(
        self,
        *,
        spec: LocalRiskSurfaceSpec | None = None,
        planner: Any | None = None,
    ) -> None:
        """Initialize with a risk-surface spec and occupancy-aware planner."""
        self.spec = spec or LocalRiskSurfaceSpec()
        self.planner = planner or RiskDWAPlannerAdapter()
        self._last_status = "not_run"
        self._last_error: str | None = None
        self._last_surface: dict[str, Any] | None = None

    def reset(self, seed: int | None = None) -> None:
        """Reset wrapped planner state when it exposes a reset hook."""
        reset = getattr(self.planner, "reset", None)
        if callable(reset):
            try:
                reset(seed=seed)
            except TypeError:
                reset()
        self._last_status = "not_run"
        self._last_error = None
        self._last_surface = None

    def _observation_for_surface(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return an observation shape accepted by the risk-surface producer.

        Returns:
            dict[str, Any]: Original observation enriched with structured SocNav fields when
            the caller provided the flattened benchmark observation contract.
        """
        if isinstance(observation.get("robot"), dict):
            return observation

        required_flat_keys = ("robot_position", "robot_heading", "goal_current")
        if not all(key in observation for key in required_flat_keys):
            return observation

        robot_state, goal_state, ped_state = self._socnav_fields(observation)
        normalized = dict(observation)
        normalized["robot"] = robot_state
        normalized["goal"] = goal_state
        normalized["pedestrians"] = ped_state
        return normalized

    def adapt_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Produce and attach a local risk surface for planner consumption.

        Returns:
            dict[str, Any]: Observation copy enriched with the surface payload.
        """
        normalized = self._observation_for_surface(observation)
        surface = deterministic_pedestrian_risk_surface(normalized, self.spec)
        enriched = attach_risk_surface_to_observation(normalized, surface)
        self._last_surface = enriched["local_risk_surface_diagnostics"]
        return enriched

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Plan with fail-closed behavior when the surface is unavailable.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        try:
            adapted = self.adapt_observation(observation)
            linear, angular = self.planner.plan(adapted)
        except Exception as exc:  # noqa: BLE001 - adapter boundary must fail closed.
            self._last_status = "not_available"
            self._last_error = str(exc)
            self._last_surface = None
            return 0.0, 0.0
        self._last_status = "ok"
        self._last_error = None
        return float(linear), float(wrap_angle_pi(float(angular)))

    def diagnostics(self) -> dict[str, Any]:
        """Expose adapter status and contract details.

        Returns:
            dict[str, Any]: JSON-compatible diagnostics for the last plan call.
        """
        return {
            "adapter": "RiskSurfacePlannerAdapter",
            "status": self._last_status,
            "error": self._last_error,
            "execution_mode": "adapter",
            "readiness_status": "adapter" if self._last_status == "ok" else "not_available",
            "availability_status": "available" if self._last_status == "ok" else self._last_status,
            "surface_contract": {
                "frame": self.spec.frame,
                "resolution": float(self.spec.resolution),
                "size": [float(self.spec.width), float(self.spec.height)],
                "risk_threshold": float(self.spec.risk_threshold),
                "producer_id": self.spec.producer_id,
            },
            "derived_payload": "local_risk_surface_as_occupancy_grid",
            "surface": self._last_surface,
            "benchmark_strength": False,
            "benchmark_strength_blockers": [
                "deterministic fixture only",
                "no trained model provenance",
                "no multi-scenario benchmark validation",
            ],
        }


__all__ = [
    "LocalRiskSurface",
    "LocalRiskSurfaceSpec",
    "RiskSurfacePlannerAdapter",
    "RiskSurfaceUnavailable",
    "attach_risk_surface_to_observation",
    "build_local_risk_surface_spec",
    "deterministic_pedestrian_risk_surface",
]
