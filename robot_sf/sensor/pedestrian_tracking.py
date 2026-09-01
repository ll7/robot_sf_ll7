"""Observation-derived pedestrian tracking and frame normalization.

This module is a default-off side channel for prediction experiments.  It accepts
only a narrow observation snapshot and assigns episode-local track IDs from the
observed geometry.  Simulator objects, route assignments, goals, PySocialForce
state, and simulator pedestrian IDs are deliberately absent from the actor API.

The canonical internal frame is ``global_xy``.  A robot pose uses the usual
right-handed convention: the robot's positive x axis points along heading
``theta`` in the global frame and positive heading is counter-clockwise.  Thus
``R(theta) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]`` maps a
robot-frame vector into global coordinates.

The tracker is an implementation-integrity baseline, not a tracking-quality or
benchmark result.  ``OracleTrackingEvaluator`` is separate from actor tracking
and may receive identity only after a tracking result has been produced.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from robot_sf.prediction._contract_utils import stable_config_hash
from robot_sf.sensor.history_stack import append_history_row

PEDESTRIAN_TRACKING_SCHEMA_VERSION = "pedestrian_tracking.v1"
HISTORY_ORDER = "oldest_to_newest"
CANONICAL_FRAME = "global_xy"


class PedestrianCoordinateFrame(StrEnum):
    """Frames accepted at the observation boundary."""

    GLOBAL_XY = "global_xy"
    ROBOT_EGO_XY = "robot_ego_xy"


class TrackStatus(StrEnum):
    """Lifecycle state of an observation-derived track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    RETIRED = "retired"


# These aliases make the contract discoverable from either the domain-specific
# or the shorter adapter vocabulary without changing the serialized values.
CoordinateFrame = PedestrianCoordinateFrame
TrackingStatus = TrackStatus


def _finite_scalar(value: Any, field_name: str) -> float:
    """Return a finite scalar float."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a real number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a real number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _non_negative(value: Any, field_name: str) -> float:
    """Return a finite non-negative scalar float."""
    normalized = _finite_scalar(value, field_name)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized


def _positive(value: Any, field_name: str) -> float:
    """Return a finite strictly positive scalar float."""
    normalized = _finite_scalar(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _readonly_array(value: Any, *, dtype: Any, field_name: str) -> np.ndarray:
    """Return an owned read-only NumPy array."""
    try:
        array = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an array-like value") from exc
    array.setflags(write=False)
    return array


def _parse_frame(value: Any, field_name: str = "coordinate_frame") -> PedestrianCoordinateFrame:
    """Parse a supported frame name and reject implicit frame aliases.

    Returns:
        The validated coordinate frame.
    """
    candidate = getattr(value, "value", value)
    if not isinstance(candidate, str):
        raise TypeError(f"{field_name} must be a supported frame name")
    try:
        return PedestrianCoordinateFrame(candidate)
    except ValueError as exc:
        allowed = ", ".join(frame.value for frame in PedestrianCoordinateFrame)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _as_pose(value: Any) -> RobotPoseGlobal:
    """Coerce a public pose value without accepting simulator state objects.

    Returns:
        A validated global robot pose.
    """
    if isinstance(value, RobotPoseGlobal):
        return value
    if isinstance(value, Mapping):
        position = value.get("position_global_xy", value.get("position"))
        heading = value.get("heading_rad", value.get("heading"))
        if position is None or heading is None:
            raise ValueError("robot_pose_global mapping needs position_global_xy and heading_rad")
        return RobotPoseGlobal(position_global_xy=position, heading_rad=heading)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("robot_pose_global must be RobotPoseGlobal or [x, y, heading_rad]") from exc
    if array.shape != (3,):
        raise ValueError("robot_pose_global must be RobotPoseGlobal or [x, y, heading_rad]")
    return RobotPoseGlobal(position_global_xy=array[:2], heading_rad=float(array[2]))


def _rotation_global_from_ego(heading_rad: float) -> np.ndarray:
    """Return the robot-ego-to-global rotation matrix."""
    heading = _finite_scalar(heading_rad, "heading_rad")
    cosine = math.cos(heading)
    sine = math.sin(heading)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def _require_last_axis_two(values: Any, field_name: str) -> np.ndarray:
    """Convert a point/vector tensor and require a final dimension of two.

    Returns:
        A finite floating-point tensor.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[-1] != 2:
        raise ValueError(f"{field_name} must have final shape dimension 2")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain finite values")
    return array


def _symmetrize_covariance(covariance: np.ndarray, field_name: str) -> np.ndarray:
    """Validate a covariance tensor and return a symmetric float array.

    Returns:
        A symmetric positive-semidefinite covariance tensor.
    """
    array = np.asarray(covariance, dtype=float)
    if array.ndim < 2 or array.shape[-2:] != (2, 2):
        raise ValueError(f"{field_name} must have final shape (2, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain finite values")
    symmetric = (array + np.swapaxes(array, -1, -2)) / 2.0
    if not np.allclose(array, np.swapaxes(array, -1, -2), rtol=0.0, atol=1e-8):
        raise ValueError(f"{field_name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.any(eigenvalues < -1e-7):
        raise ValueError(f"{field_name} must be positive semidefinite")
    return symmetric


@dataclass(frozen=True, slots=True)
class RobotPoseGlobal:
    """Finite global robot pose used for same-step frame transforms."""

    position_global_xy: np.ndarray
    heading_rad: float

    def __post_init__(self) -> None:
        """Validate and freeze the pose payload."""
        position = np.asarray(self.position_global_xy, dtype=float)
        if position.shape != (2,):
            raise ValueError("position_global_xy must have shape (2,)")
        if not np.all(np.isfinite(position)):
            raise ValueError("position_global_xy must contain finite values")
        object.__setattr__(
            self,
            "position_global_xy",
            _readonly_array(position, dtype=float, field_name="position_global_xy"),
        )
        object.__setattr__(self, "heading_rad", _finite_scalar(self.heading_rad, "heading_rad"))


def transform_position_to_global_xy(
    values: np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_pose_global: RobotPoseGlobal | Sequence[float],
) -> np.ndarray:
    """Transform points from ``global_xy`` or ``robot_ego_xy`` to global XY.

    Returns:
        A newly allocated global-frame point tensor.
    """
    points = _require_last_axis_two(values, "positions")
    frame = _parse_frame(coordinate_frame)
    pose = _as_pose(robot_pose_global)
    if frame is PedestrianCoordinateFrame.GLOBAL_XY:
        return np.array(points, dtype=float, copy=True)
    rotation = _rotation_global_from_ego(pose.heading_rad)
    return np.matmul(points, rotation.T) + pose.position_global_xy


def transform_velocity_to_global_xy(
    values: np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_pose_global: RobotPoseGlobal | Sequence[float],
) -> np.ndarray:
    """Transform planar velocity vectors to global XY without translation.

    Returns:
        A newly allocated global-frame velocity tensor.
    """
    vectors = _require_last_axis_two(values, "velocities")
    frame = _parse_frame(coordinate_frame)
    pose = _as_pose(robot_pose_global)
    if frame is PedestrianCoordinateFrame.GLOBAL_XY:
        return np.array(vectors, dtype=float, copy=True)
    rotation = _rotation_global_from_ego(pose.heading_rad)
    return np.matmul(vectors, rotation.T)


def transform_covariance_to_global_xy(
    values: np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_pose_global: RobotPoseGlobal | Sequence[float],
) -> np.ndarray:
    """Rotate 2x2 covariance matrices into global XY and preserve PSD.

    Returns:
        A symmetric global-frame covariance tensor.
    """
    covariance = _symmetrize_covariance(values, "covariances")
    frame = _parse_frame(coordinate_frame)
    pose = _as_pose(robot_pose_global)
    if frame is PedestrianCoordinateFrame.GLOBAL_XY:
        return np.array(covariance, dtype=float, copy=True)
    rotation = _rotation_global_from_ego(pose.heading_rad)
    transformed = np.einsum("ij,...jk,lk->...il", rotation, covariance, rotation)
    return (transformed + np.swapaxes(transformed, -1, -2)) / 2.0


def transform_heading_to_global_xy(
    values: float | np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_pose_global: RobotPoseGlobal | Sequence[float],
) -> np.ndarray | float:
    """Transform a heading or direction angle to global XY and wrap it.

    Returns:
        A wrapped scalar or array with the same scalar/array rank as ``values``.
    """
    angles = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("headings must contain finite values")
    frame = _parse_frame(coordinate_frame)
    pose = _as_pose(robot_pose_global)
    transformed = (
        angles if frame is PedestrianCoordinateFrame.GLOBAL_XY else angles + pose.heading_rad
    )
    wrapped = (transformed + math.pi) % (2.0 * math.pi) - math.pi
    if np.ndim(values) == 0:
        return float(wrapped)
    return np.array(wrapped, dtype=float, copy=True)


def transform_position_from_global_xy(
    values: np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_pose_global: RobotPoseGlobal | Sequence[float],
) -> np.ndarray:
    """Transform global XY points into a declared observation frame.

    Returns:
        A newly allocated point tensor in ``coordinate_frame``.
    """
    points = _require_last_axis_two(values, "positions")
    frame = _parse_frame(coordinate_frame)
    pose = _as_pose(robot_pose_global)
    if frame is PedestrianCoordinateFrame.GLOBAL_XY:
        return np.array(points, dtype=float, copy=True)
    rotation = _rotation_global_from_ego(pose.heading_rad)
    return np.matmul(points - pose.position_global_xy, rotation)


def transform_velocity_from_global_xy(
    values: np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_pose_global: RobotPoseGlobal | Sequence[float],
) -> np.ndarray:
    """Transform global XY velocity vectors into a declared observation frame.

    Returns:
        A newly allocated velocity tensor in ``coordinate_frame``.
    """
    vectors = _require_last_axis_two(values, "velocities")
    frame = _parse_frame(coordinate_frame)
    pose = _as_pose(robot_pose_global)
    if frame is PedestrianCoordinateFrame.GLOBAL_XY:
        return np.array(vectors, dtype=float, copy=True)
    rotation = _rotation_global_from_ego(pose.heading_rad)
    return np.matmul(vectors, rotation)


def transform_covariance_from_global_xy(
    values: np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_pose_global: RobotPoseGlobal | Sequence[float],
) -> np.ndarray:
    """Rotate global XY covariance matrices into a declared observation frame.

    Returns:
        A symmetric covariance tensor in ``coordinate_frame``.
    """
    covariance = _symmetrize_covariance(values, "covariances")
    frame = _parse_frame(coordinate_frame)
    pose = _as_pose(robot_pose_global)
    if frame is PedestrianCoordinateFrame.GLOBAL_XY:
        return np.array(covariance, dtype=float, copy=True)
    rotation = _rotation_global_from_ego(pose.heading_rad)
    transformed = np.einsum("ji,...jk,kl->...il", rotation, covariance, rotation)
    return (transformed + np.swapaxes(transformed, -1, -2)) / 2.0


def transform_heading_from_global_xy(
    values: float | np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_pose_global: RobotPoseGlobal | Sequence[float],
) -> np.ndarray | float:
    """Transform global heading angles into a declared observation frame.

    Returns:
        A wrapped scalar or array with the same scalar/array rank as ``values``.
    """
    angles = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("headings must contain finite values")
    frame = _parse_frame(coordinate_frame)
    pose = _as_pose(robot_pose_global)
    transformed = (
        angles if frame is PedestrianCoordinateFrame.GLOBAL_XY else angles - pose.heading_rad
    )
    wrapped = (transformed + math.pi) % (2.0 * math.pi) - math.pi
    if np.ndim(values) == 0:
        return float(wrapped)
    return np.array(wrapped, dtype=float, copy=True)


def transform_history_to_global_xy(
    values: np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_poses_global: Sequence[RobotPoseGlobal | Sequence[float]],
    *,
    value_kind: Literal["position", "velocity", "covariance"] = "position",
) -> np.ndarray:
    """Transform oldest-to-newest history rows with their same-step poses.

    The first dimension is never reordered.  A separate pose is required for
    each history row so a later robot heading cannot silently rewrite an older
    observation.

    Returns:
        A transformed tensor retaining the input's oldest-to-newest order.
    """
    array = np.asarray(values, dtype=float)
    poses = tuple(_as_pose(pose) for pose in robot_poses_global)
    if array.ndim == 0 or array.shape[0] != len(poses):
        raise ValueError("values first dimension must match robot_poses_global")
    if value_kind == "position":
        transformed = [
            transform_position_to_global_xy(row, coordinate_frame, pose)
            for row, pose in zip(array, poses, strict=True)
        ]
    elif value_kind == "velocity":
        transformed = [
            transform_velocity_to_global_xy(row, coordinate_frame, pose)
            for row, pose in zip(array, poses, strict=True)
        ]
    elif value_kind == "covariance":
        transformed = [
            transform_covariance_to_global_xy(row, coordinate_frame, pose)
            for row, pose in zip(array, poses, strict=True)
        ]
    else:
        raise ValueError("value_kind must be position, velocity, or covariance")
    if not transformed:
        return np.array(array, dtype=float, copy=True)
    return np.stack(transformed, axis=0)


def transform_history_from_global_xy(
    values: np.ndarray,
    coordinate_frame: str | PedestrianCoordinateFrame,
    robot_poses_global: Sequence[RobotPoseGlobal | Sequence[float]],
    *,
    value_kind: Literal["position", "velocity", "covariance"] = "position",
) -> np.ndarray:
    """Invert global normalization while retaining oldest-to-newest history order.

    Returns:
        A transformed tensor retaining the input's oldest-to-newest order.
    """
    array = np.asarray(values, dtype=float)
    poses = tuple(_as_pose(pose) for pose in robot_poses_global)
    if array.ndim == 0 or array.shape[0] != len(poses):
        raise ValueError("values first dimension must match robot_poses_global")
    if value_kind == "position":
        transformed = [
            transform_position_from_global_xy(row, coordinate_frame, pose)
            for row, pose in zip(array, poses, strict=True)
        ]
    elif value_kind == "velocity":
        transformed = [
            transform_velocity_from_global_xy(row, coordinate_frame, pose)
            for row, pose in zip(array, poses, strict=True)
        ]
    elif value_kind == "covariance":
        transformed = [
            transform_covariance_from_global_xy(row, coordinate_frame, pose)
            for row, pose in zip(array, poses, strict=True)
        ]
    else:
        raise ValueError("value_kind must be position, velocity, or covariance")
    if not transformed:
        return np.array(array, dtype=float, copy=True)
    return np.stack(transformed, axis=0)


# Short names keep adapter call sites readable while retaining explicit frame semantics.
position_to_global_xy = transform_position_to_global_xy
velocity_to_global_xy = transform_velocity_to_global_xy
covariance_to_global_xy = transform_covariance_to_global_xy
heading_to_global_xy = transform_heading_to_global_xy
history_to_global_xy = transform_history_to_global_xy
position_from_global_xy = transform_position_from_global_xy
velocity_from_global_xy = transform_velocity_from_global_xy
covariance_from_global_xy = transform_covariance_from_global_xy
heading_from_global_xy = transform_heading_from_global_xy
history_from_global_xy = transform_history_from_global_xy


@dataclass(frozen=True, slots=True)
class PedestrianTrackingConfig:
    """Immutable configuration for the transparent tracking baseline.

    Mahalanobis gate thresholds are squared normalized distances.  When
    ``gating_mode='euclidean'`` the same fields are interpreted as metres and
    metres/second, respectively; that mode is explicit and never selected as a
    silent fallback.
    """

    enabled: bool = False
    process_noise: float = 1.0
    initial_position_covariance: float = 1.0
    initial_velocity_covariance: float = 1.0
    measurement_position_covariance: float = 0.25
    measurement_velocity_covariance: float = 0.25
    position_gate_threshold: float = 9.21
    velocity_gate_threshold: float = 9.21
    gating_mode: str = "mahalanobis"
    confirmation_steps: int = 2
    max_missed_seconds: float = 1.0
    max_missed_steps: int | None = None
    use_velocity: bool = True
    tie_break_policy: str = "track_id_then_observation"
    max_tracks: int = 64
    history_capacity: int = 8
    tie_break_epsilon: float = 1e-9

    def __post_init__(self) -> None:  # noqa: C901
        """Validate all configuration values and normalize scalar types."""
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if not isinstance(self.use_velocity, bool):
            raise TypeError("use_velocity must be a bool")
        for field_name in (
            "process_noise",
            "initial_position_covariance",
            "initial_velocity_covariance",
            "measurement_position_covariance",
            "measurement_velocity_covariance",
        ):
            object.__setattr__(
                self, field_name, _non_negative(getattr(self, field_name), field_name)
            )
        for field_name in ("position_gate_threshold", "velocity_gate_threshold"):
            object.__setattr__(self, field_name, _positive(getattr(self, field_name), field_name))
        object.__setattr__(
            self, "max_missed_seconds", _positive(self.max_missed_seconds, "max_missed_seconds")
        )
        if type(self.confirmation_steps) is not int or self.confirmation_steps < 1:
            raise ValueError("confirmation_steps must be a positive integer")
        if self.max_missed_steps is not None and (
            type(self.max_missed_steps) is not int or self.max_missed_steps < 1
        ):
            raise ValueError("max_missed_steps must be None or a positive integer")
        if type(self.max_tracks) is not int or self.max_tracks < 1:
            raise ValueError("max_tracks must be a positive integer")
        if type(self.history_capacity) is not int or self.history_capacity < 1:
            raise ValueError("history_capacity must be a positive integer")
        gating_mode = str(self.gating_mode).strip().lower()
        if gating_mode not in {"mahalanobis", "euclidean"}:
            raise ValueError("gating_mode must be 'mahalanobis' or explicitly 'euclidean'")
        object.__setattr__(self, "gating_mode", gating_mode)
        if self.tie_break_policy != "track_id_then_observation":
            raise ValueError("tie_break_policy must be 'track_id_then_observation'")
        object.__setattr__(
            self, "tie_break_epsilon", _non_negative(self.tie_break_epsilon, "tie_break_epsilon")
        )
        if self.tie_break_epsilon >= 1e-3:
            raise ValueError("tie_break_epsilon must remain a tiny deterministic perturbation")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe configuration mapping used for hashing."""
        return {
            "schema_version": PEDESTRIAN_TRACKING_SCHEMA_VERSION,
            "enabled": self.enabled,
            "process_noise": self.process_noise,
            "initial_position_covariance": self.initial_position_covariance,
            "initial_velocity_covariance": self.initial_velocity_covariance,
            "measurement_position_covariance": self.measurement_position_covariance,
            "measurement_velocity_covariance": self.measurement_velocity_covariance,
            "position_gate_threshold": self.position_gate_threshold,
            "velocity_gate_threshold": self.velocity_gate_threshold,
            "gating_mode": self.gating_mode,
            "confirmation_steps": self.confirmation_steps,
            "max_missed_seconds": self.max_missed_seconds,
            "max_missed_steps": self.max_missed_steps,
            "use_velocity": self.use_velocity,
            "tie_break_policy": self.tie_break_policy,
            "max_tracks": self.max_tracks,
            "history_capacity": self.history_capacity,
            "tie_break_epsilon": self.tie_break_epsilon,
        }

    @property
    def config_hash(self) -> str:
        """Return the stable SHA-256 configuration hash."""
        return stable_config_hash(self.to_dict())

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any] | None) -> PedestrianTrackingConfig:
        """Parse a strict mapping with a few documented compatibility aliases.

        Returns:
            An immutable validated tracking configuration.
        """
        if spec is None:
            return cls()
        if not isinstance(spec, Mapping):
            raise TypeError("pedestrian tracking config must be a mapping or None")
        aliases = {
            "initial_covariance": "initial_position_covariance",
            "position_process_noise": "process_noise",
            "velocity_process_noise": "process_noise",
            "position_gate": "position_gate_threshold",
            "velocity_gate": "velocity_gate_threshold",
            "max_missed_duration_s": "max_missed_seconds",
        }
        fields = {
            "enabled",
            "process_noise",
            "initial_position_covariance",
            "initial_velocity_covariance",
            "measurement_position_covariance",
            "measurement_velocity_covariance",
            "position_gate_threshold",
            "velocity_gate_threshold",
            "gating_mode",
            "confirmation_steps",
            "max_missed_seconds",
            "max_missed_steps",
            "use_velocity",
            "tie_break_policy",
            "max_tracks",
            "history_capacity",
            "tie_break_epsilon",
        }
        normalized: dict[str, Any] = {}
        for key, value in spec.items():
            canonical = aliases.get(str(key), str(key))
            if canonical == "schema_version":
                if value != PEDESTRIAN_TRACKING_SCHEMA_VERSION:
                    raise ValueError("unsupported pedestrian tracking schema_version")
                continue
            if canonical not in fields:
                raise ValueError(f"unknown pedestrian tracking config key: {key}")
            normalized[canonical] = value
        return cls(**normalized)


def pedestrian_tracking_config_from_spec(
    spec: Mapping[str, Any] | None,
) -> PedestrianTrackingConfig:
    """Return an immutable tracking config from a config-first mapping."""
    return PedestrianTrackingConfig.from_mapping(spec)


@dataclass(frozen=True, slots=True)
class PedestrianObservationSnapshot:
    """Narrow actor input containing only current observation-derived fields."""

    timestamp_s: float
    step_index: int
    coordinate_frame: str | PedestrianCoordinateFrame
    robot_pose_global: RobotPoseGlobal | Sequence[float]
    positions: np.ndarray
    velocities: np.ndarray | None
    valid_mask: np.ndarray
    visible_mask: np.ndarray
    position_covariances: np.ndarray | None = None
    velocity_covariances: np.ndarray | None = None
    radius: np.ndarray | float | None = None
    velocity_valid_mask: np.ndarray | None = None
    velocity_coordinate_frame: str | PedestrianCoordinateFrame | None = None

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Validate shapes, masks, finite active rows, and same-step pose data."""
        timestamp = _finite_scalar(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        frame = _parse_frame(self.coordinate_frame)
        velocity_frame = _parse_frame(
            frame if self.velocity_coordinate_frame is None else self.velocity_coordinate_frame,
            "velocity_coordinate_frame",
        )
        pose = _as_pose(self.robot_pose_global)

        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        row_count = positions.shape[0]
        valid = np.asarray(self.valid_mask, dtype=bool)
        visible = np.asarray(self.visible_mask, dtype=bool)
        if valid.shape != (row_count,) or visible.shape != (row_count,):
            raise ValueError("valid_mask and visible_mask must have shape (N,)")
        if np.any(visible & ~valid):
            raise ValueError("visible_mask cannot be true for an invalid row")
        detection_mask = valid & visible
        if np.any(detection_mask[:, np.newaxis] & ~np.isfinite(positions)):
            raise ValueError("visible valid positions must be finite")
        positions = np.array(positions, dtype=float, copy=True)
        positions[~np.isfinite(positions)] = 0.0
        positions[~valid] = 0.0

        if self.velocities is None:
            if self.velocity_valid_mask is not None:
                supplied_velocity_valid = np.asarray(self.velocity_valid_mask, dtype=bool)
                if supplied_velocity_valid.shape != (row_count,):
                    raise ValueError("velocity_valid_mask must have shape (N,)")
                if np.any(supplied_velocity_valid):
                    raise ValueError(
                        "velocity_valid_mask cannot be true when velocities are absent"
                    )
            velocities = np.zeros((row_count, 2), dtype=float)
            velocity_valid = np.zeros((row_count,), dtype=bool)
        else:
            velocities = np.asarray(self.velocities, dtype=float)
            if velocities.shape != (row_count, 2):
                raise ValueError("velocities must have shape (N, 2)")
            finite_velocity = np.all(np.isfinite(velocities), axis=1)
            if self.velocity_valid_mask is None:
                velocity_valid = detection_mask & finite_velocity
            else:
                velocity_valid = np.asarray(self.velocity_valid_mask, dtype=bool)
                if velocity_valid.shape != (row_count,):
                    raise ValueError("velocity_valid_mask must have shape (N,)")
                if np.any(velocity_valid & ~detection_mask):
                    raise ValueError("velocity_valid_mask can only be true for visible valid rows")
                if np.any(velocity_valid & ~finite_velocity):
                    raise ValueError("velocity-valid rows must contain finite velocities")
            velocities = np.array(velocities, dtype=float, copy=True)
            velocities[~np.isfinite(velocities)] = 0.0
            velocities[~detection_mask] = 0.0

        position_covariances = _coerce_covariance_rows(
            self.position_covariances,
            row_count=row_count,
            validation_mask=detection_mask,
            field_name="position_covariances",
        )
        velocity_covariances = _coerce_covariance_rows(
            self.velocity_covariances,
            row_count=row_count,
            validation_mask=velocity_valid,
            field_name="velocity_covariances",
        )
        radius = _coerce_radius(self.radius, row_count, detection_mask)

        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "coordinate_frame", frame.value)
        object.__setattr__(self, "velocity_coordinate_frame", velocity_frame.value)
        object.__setattr__(self, "robot_pose_global", pose)
        object.__setattr__(
            self, "positions", _readonly_array(positions, dtype=float, field_name="positions")
        )
        object.__setattr__(
            self, "velocities", _readonly_array(velocities, dtype=float, field_name="velocities")
        )
        object.__setattr__(
            self, "valid_mask", _readonly_array(valid, dtype=bool, field_name="valid_mask")
        )
        object.__setattr__(
            self, "visible_mask", _readonly_array(visible, dtype=bool, field_name="visible_mask")
        )
        object.__setattr__(
            self,
            "velocity_valid_mask",
            _readonly_array(velocity_valid, dtype=bool, field_name="velocity_valid_mask"),
        )
        object.__setattr__(self, "position_covariances", position_covariances)
        object.__setattr__(self, "velocity_covariances", velocity_covariances)
        object.__setattr__(self, "radius", radius)

    @property
    def detection_mask(self) -> np.ndarray:
        """Return the current visible, valid row mask."""
        mask = np.asarray(self.valid_mask & self.visible_mask, dtype=bool)
        mask.setflags(write=False)
        return mask

    def to_global_xy(self) -> NormalizedPedestrianObservation:
        """Normalize all current fields into the canonical global XY frame.

        Returns:
            An immutable normalized observation side channel.
        """
        pose = _as_pose(self.robot_pose_global)
        velocity_frame = _parse_frame(
            self.coordinate_frame
            if self.velocity_coordinate_frame is None
            else self.velocity_coordinate_frame,
            "velocity_coordinate_frame",
        )
        velocities = (
            np.zeros_like(self.positions)
            if self.velocities is None
            else np.asarray(self.velocities, dtype=float)
        )
        velocity_valid = (
            np.zeros((self.positions.shape[0],), dtype=bool)
            if self.velocity_valid_mask is None
            else np.asarray(self.velocity_valid_mask, dtype=bool)
        )
        normalized_radius = None if self.radius is None else np.asarray(self.radius, dtype=float)
        positions = transform_position_to_global_xy(self.positions, self.coordinate_frame, pose)
        velocities = transform_velocity_to_global_xy(velocities, velocity_frame, pose)
        position_covariances = (
            None
            if self.position_covariances is None
            else transform_covariance_to_global_xy(
                self.position_covariances, self.coordinate_frame, pose
            )
        )
        velocity_covariances = (
            None
            if self.velocity_covariances is None
            else transform_covariance_to_global_xy(
                self.velocity_covariances,
                velocity_frame,
                pose,
            )
        )
        return NormalizedPedestrianObservation(
            timestamp_s=self.timestamp_s,
            step_index=self.step_index,
            robot_pose_global=pose,
            positions_global_xy=positions,
            velocities_global_xy=velocities,
            valid_mask=self.valid_mask,
            visible_mask=self.visible_mask,
            velocity_valid_mask=velocity_valid,
            position_covariances_global_xy=position_covariances,
            velocity_covariances_global_xy=velocity_covariances,
            radius=normalized_radius,
        )


def _coerce_covariance_rows(
    value: np.ndarray | None,
    *,
    row_count: int,
    validation_mask: np.ndarray,
    field_name: str,
) -> np.ndarray | None:
    """Normalize optional per-row covariance input to ``(N, 2, 2)``.

    Returns:
        A read-only covariance tensor, or ``None`` when no covariance was supplied.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape == (2, 2) and row_count == 1:
        array = array[np.newaxis, ...]
    elif array.shape == (row_count, 2):
        diagonal = np.zeros((row_count, 2, 2), dtype=float)
        diagonal[:, 0, 0] = array[:, 0]
        diagonal[:, 1, 1] = array[:, 1]
        array = diagonal
    elif array.shape != (row_count, 2, 2):
        raise ValueError(f"{field_name} must have shape (N, 2, 2), (N, 2), or (2, 2) for N=1")
    array = np.array(array, dtype=float, copy=True)
    if np.any(validation_mask[:, np.newaxis, np.newaxis] & ~np.isfinite(array)):
        raise ValueError(f"active {field_name} must contain finite values")
    for index in np.flatnonzero(validation_mask):
        array[index] = _symmetrize_covariance(array[index], f"{field_name}[{index}]")
    array[~validation_mask] = 0.0
    return _readonly_array(array, dtype=float, field_name=field_name)


def _coerce_radius(
    value: np.ndarray | float | None,
    row_count: int,
    validation_mask: np.ndarray,
) -> np.ndarray | None:
    """Normalize an optional scalar or per-row radius payload.

    Returns:
        A read-only per-row radius array, or ``None`` when omitted.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full((row_count,), float(array), dtype=float)
    if array.shape != (row_count,):
        raise ValueError("radius must be a scalar or have shape (N,)")
    if np.any(validation_mask & ~np.isfinite(array)):
        raise ValueError("active radius values must be finite")
    if np.any(validation_mask & (array < 0.0)):
        raise ValueError("active radius values must be non-negative")
    array = np.array(array, dtype=float, copy=True)
    array[~np.isfinite(array)] = 0.0
    array[~validation_mask] = 0.0
    return _readonly_array(array, dtype=float, field_name="radius")


@dataclass(frozen=True, slots=True)
class NormalizedPedestrianObservation:
    """Immutable global-frame observation side channel."""

    timestamp_s: float
    step_index: int
    robot_pose_global: RobotPoseGlobal
    positions_global_xy: np.ndarray
    velocities_global_xy: np.ndarray
    valid_mask: np.ndarray
    visible_mask: np.ndarray
    velocity_valid_mask: np.ndarray
    position_covariances_global_xy: np.ndarray | None
    velocity_covariances_global_xy: np.ndarray | None
    radius: np.ndarray | None

    @property
    def coordinate_frame(self) -> str:
        """Return the canonical frame name for every normalized field."""
        return CANONICAL_FRAME

    def __post_init__(self) -> None:  # noqa: C901
        """Defensively freeze normalized arrays."""
        timestamp = _finite_scalar(self.timestamp_s, "timestamp_s")
        if timestamp < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "robot_pose_global", _as_pose(self.robot_pose_global))
        position = np.asarray(self.positions_global_xy, dtype=float)
        velocity = np.asarray(self.velocities_global_xy, dtype=float)
        if position.ndim != 2 or position.shape[1] != 2 or velocity.shape != position.shape:
            raise ValueError("normalized positions and velocities must have shape (N, 2)")
        row_count = position.shape[0]
        valid = np.asarray(self.valid_mask, dtype=bool)
        visible = np.asarray(self.visible_mask, dtype=bool)
        velocity_valid = np.asarray(self.velocity_valid_mask, dtype=bool)
        if (
            valid.shape != (row_count,)
            or visible.shape != (row_count,)
            or velocity_valid.shape != (row_count,)
        ):
            raise ValueError("normalized masks must have shape (N,)")
        if np.any(visible & ~valid) or np.any(velocity_valid & ~(valid & visible)):
            raise ValueError("normalized masks contain an invalid visible or velocity-valid row")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("normalized positions and velocities must be finite")
        object.__setattr__(
            self,
            "positions_global_xy",
            _readonly_array(position, dtype=float, field_name="positions_global_xy"),
        )
        object.__setattr__(
            self,
            "velocities_global_xy",
            _readonly_array(velocity, dtype=float, field_name="velocities_global_xy"),
        )
        object.__setattr__(
            self, "valid_mask", _readonly_array(valid, dtype=bool, field_name="valid_mask")
        )
        object.__setattr__(
            self, "visible_mask", _readonly_array(visible, dtype=bool, field_name="visible_mask")
        )
        object.__setattr__(
            self,
            "velocity_valid_mask",
            _readonly_array(velocity_valid, dtype=bool, field_name="velocity_valid_mask"),
        )
        for field_name in ("position_covariances_global_xy", "velocity_covariances_global_xy"):
            covariance = getattr(self, field_name)
            if covariance is None:
                continue
            normalized_covariance = np.asarray(covariance, dtype=float)
            if normalized_covariance.shape != (row_count, 2, 2):
                raise ValueError(f"{field_name} must have shape (N, 2, 2)")
            normalized_covariance = np.array(normalized_covariance, dtype=float, copy=True)
            for index, row in enumerate(normalized_covariance):
                normalized_covariance[index] = _symmetrize_covariance(row, f"{field_name}[{index}]")
            object.__setattr__(
                self,
                field_name,
                _readonly_array(
                    normalized_covariance,
                    dtype=float,
                    field_name=field_name,
                ),
            )
        if self.radius is not None:
            radius = np.asarray(self.radius, dtype=float)
            if radius.shape != (row_count,):
                raise ValueError("radius must have shape (N,)")
            if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
                raise ValueError("radius must contain finite non-negative values")
            object.__setattr__(
                self, "radius", _readonly_array(radius, dtype=float, field_name="radius")
            )

    @property
    def detection_mask(self) -> np.ndarray:
        """Return the normalized visible detection mask."""
        mask = np.asarray(self.valid_mask & self.visible_mask, dtype=bool)
        mask.setflags(write=False)
        return mask


@dataclass(frozen=True, slots=True)
class TrackAssociation:
    """One accepted gated association, retaining the source slot diagnostically."""

    track_id: int
    observation_slot: int
    cost: float
    position_distance: float
    velocity_distance: float | None
    used_velocity: bool
    confidence: float

    def __post_init__(self) -> None:
        """Validate association values."""
        if type(self.track_id) is not int or self.track_id < 1:
            raise ValueError("track_id must be a positive integer")
        if type(self.observation_slot) is not int or self.observation_slot < 0:
            raise ValueError("observation_slot must be a non-negative integer")
        for field_name in ("cost", "position_distance"):
            value = _finite_scalar(getattr(self, field_name), field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        if self.velocity_distance is not None:
            value = _finite_scalar(self.velocity_distance, "velocity_distance")
            if value < 0.0:
                raise ValueError("velocity_distance must be non-negative")
            object.__setattr__(self, "velocity_distance", value)
        if not isinstance(self.used_velocity, bool):
            raise TypeError("used_velocity must be a bool")
        confidence = _finite_scalar(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class PedestrianTrack:
    """Immutable per-track output consumed by future prediction adapters."""

    track_id: int
    timestamp_s: float
    step_index: int
    position_global_xy: np.ndarray
    velocity_global_xy: np.ndarray
    position_covariance: np.ndarray
    velocity_covariance: np.ndarray
    age_steps: int
    visible_age_steps: int
    missed_steps: int
    status: TrackStatus | str
    association_confidence: float
    last_observation_slot: int | None
    history_valid_mask: np.ndarray
    position_history_global_xy: np.ndarray
    velocity_history_global_xy: np.ndarray
    timestamp_history_s: np.ndarray
    blockers: tuple[str, ...]
    config_hash: str

    def __post_init__(self) -> None:  # noqa: C901
        """Validate and defensively freeze track state."""
        if type(self.track_id) is not int or self.track_id < 1:
            raise ValueError("track_id must be a positive integer")
        object.__setattr__(self, "timestamp_s", _finite_scalar(self.timestamp_s, "timestamp_s"))
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        for field_name in ("age_steps", "visible_age_steps", "missed_steps"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        try:
            status = TrackStatus(getattr(self.status, "value", self.status))
        except ValueError as exc:
            raise ValueError("status must be tentative, confirmed, lost, or retired") from exc
        object.__setattr__(self, "status", status)
        confidence = _finite_scalar(self.association_confidence, "association_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("association_confidence must be between 0 and 1")
        object.__setattr__(self, "association_confidence", confidence)
        if self.last_observation_slot is not None and (
            type(self.last_observation_slot) is not int or self.last_observation_slot < 0
        ):
            raise ValueError("last_observation_slot must be None or a non-negative integer")
        position = np.asarray(self.position_global_xy, dtype=float)
        velocity = np.asarray(self.velocity_global_xy, dtype=float)
        if position.shape != (2,) or velocity.shape != (2,):
            raise ValueError("track position_global_xy and velocity_global_xy must have shape (2,)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("track position and velocity must be finite")
        pos_cov = _symmetrize_covariance(self.position_covariance, "position_covariance")
        vel_cov = _symmetrize_covariance(self.velocity_covariance, "velocity_covariance")
        history_mask = np.asarray(self.history_valid_mask, dtype=bool)
        pos_history = np.asarray(self.position_history_global_xy, dtype=float)
        vel_history = np.asarray(self.velocity_history_global_xy, dtype=float)
        timestamps = np.asarray(self.timestamp_history_s, dtype=float)
        if (
            history_mask.ndim != 1
            or pos_history.shape != (history_mask.shape[0], 2)
            or vel_history.shape != pos_history.shape
            or timestamps.shape != history_mask.shape
        ):
            raise ValueError("track history fields must use a common oldest-to-newest capacity")
        if (
            not np.all(np.isfinite(pos_history))
            or not np.all(np.isfinite(vel_history))
            or not np.all(np.isfinite(timestamps))
        ):
            raise ValueError("track history fields must be finite")
        blockers = tuple(str(blocker) for blocker in self.blockers)
        if len(self.config_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.config_hash
        ):
            raise ValueError("config_hash must be a lowercase SHA-256 digest")
        object.__setattr__(
            self,
            "position_global_xy",
            _readonly_array(position, dtype=float, field_name="position_global_xy"),
        )
        object.__setattr__(
            self,
            "velocity_global_xy",
            _readonly_array(velocity, dtype=float, field_name="velocity_global_xy"),
        )
        object.__setattr__(
            self,
            "position_covariance",
            _readonly_array(pos_cov, dtype=float, field_name="position_covariance"),
        )
        object.__setattr__(
            self,
            "velocity_covariance",
            _readonly_array(vel_cov, dtype=float, field_name="velocity_covariance"),
        )
        object.__setattr__(
            self,
            "history_valid_mask",
            _readonly_array(history_mask, dtype=bool, field_name="history_valid_mask"),
        )
        object.__setattr__(
            self,
            "position_history_global_xy",
            _readonly_array(pos_history, dtype=float, field_name="position_history_global_xy"),
        )
        object.__setattr__(
            self,
            "velocity_history_global_xy",
            _readonly_array(vel_history, dtype=float, field_name="velocity_history_global_xy"),
        )
        object.__setattr__(
            self,
            "timestamp_history_s",
            _readonly_array(timestamps, dtype=float, field_name="timestamp_history_s"),
        )
        object.__setattr__(self, "blockers", blockers)

    @property
    def history_order(self) -> str:
        """Return the shared sensor-history ordering contract."""
        return HISTORY_ORDER

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic representation."""
        return {
            "track_id": self.track_id,
            "timestamp_s": self.timestamp_s,
            "step_index": self.step_index,
            "position_global_xy": self.position_global_xy.tolist(),
            "velocity_global_xy": self.velocity_global_xy.tolist(),
            "position_covariance": self.position_covariance.tolist(),
            "velocity_covariance": self.velocity_covariance.tolist(),
            "age_steps": self.age_steps,
            "visible_age_steps": self.visible_age_steps,
            "missed_steps": self.missed_steps,
            "status": (self.status.value if isinstance(self.status, TrackStatus) else self.status),
            "association_confidence": self.association_confidence,
            "last_observation_slot": self.last_observation_slot,
            "history_valid_mask": self.history_valid_mask.tolist(),
            "position_history_global_xy": self.position_history_global_xy.tolist(),
            "velocity_history_global_xy": self.velocity_history_global_xy.tolist(),
            "timestamp_history_s": self.timestamp_history_s.tolist(),
            "blockers": list(self.blockers),
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True, slots=True)
class PedestrianTrackingDiagnostics:
    """Compact per-update diagnostic counters."""

    enabled: bool
    timestamp_gap_s: float
    input_row_count: int
    valid_row_count: int
    visible_row_count: int
    detection_count: int
    invalid_row_count: int
    velocity_unavailable_count: int
    association_count: int
    new_track_count: int
    lost_track_count: int
    retired_track_count: int
    active_track_count: int
    tentative_track_count: int
    confirmed_track_count: int
    cost_matrix_shape: tuple[int, int]
    estimated_cost_matrix_bytes: int
    update_latency_ms: float
    blockers: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate non-negative counters and finite diagnostics."""
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        object.__setattr__(
            self, "timestamp_gap_s", _non_negative(self.timestamp_gap_s, "timestamp_gap_s")
        )
        for field_name in (
            "input_row_count",
            "valid_row_count",
            "visible_row_count",
            "detection_count",
            "invalid_row_count",
            "velocity_unavailable_count",
            "association_count",
            "new_track_count",
            "lost_track_count",
            "retired_track_count",
            "active_track_count",
            "tentative_track_count",
            "confirmed_track_count",
            "estimated_cost_matrix_bytes",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if len(self.cost_matrix_shape) != 2 or any(
            type(value) is not int or value < 0 for value in self.cost_matrix_shape
        ):
            raise ValueError("cost_matrix_shape must contain two non-negative integers")
        object.__setattr__(
            self, "update_latency_ms", _non_negative(self.update_latency_ms, "update_latency_ms")
        )
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))


@dataclass(frozen=True, slots=True)
class PedestrianTrackingResult:
    """Immutable result of one tracker update."""

    timestamp_s: float
    step_index: int
    tracks: tuple[PedestrianTrack, ...]
    associations: tuple[TrackAssociation, ...]
    diagnostics: PedestrianTrackingDiagnostics
    history_order: str = HISTORY_ORDER

    def __post_init__(self) -> None:
        """Validate deterministic output ordering."""
        object.__setattr__(self, "timestamp_s", _finite_scalar(self.timestamp_s, "timestamp_s"))
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        tracks = tuple(self.tracks)
        associations = tuple(self.associations)
        if any(not isinstance(track, PedestrianTrack) for track in tracks):
            raise TypeError("tracks must contain PedestrianTrack values")
        if any(not isinstance(association, TrackAssociation) for association in associations):
            raise TypeError("associations must contain TrackAssociation values")
        if tuple(track.track_id for track in tracks) != tuple(
            sorted(track.track_id for track in tracks)
        ):
            raise ValueError("tracks must be sorted by track_id")
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "associations", associations)
        if self.history_order != HISTORY_ORDER:
            raise ValueError(f"history_order must be {HISTORY_ORDER}")

    def track(self, track_id: int) -> PedestrianTrack | None:
        """Return one track by persistent ID."""
        return next((track for track in self.tracks if track.track_id == track_id), None)


@dataclass(slots=True)
class _PredictedState:
    """Mutable prediction scratch space used only within one update."""

    position: np.ndarray
    velocity: np.ndarray
    position_covariance: np.ndarray
    velocity_covariance: np.ndarray


@dataclass(slots=True)
class _TrackState:
    """Mutable internal track; public results are defensive immutable copies."""

    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    position_covariance: np.ndarray
    velocity_covariance: np.ndarray
    timestamp_s: float
    step_index: int
    age_steps: int
    visible_age_steps: int
    missed_steps: int
    missed_seconds: float
    status: TrackStatus
    association_confidence: float
    last_observation_slot: int | None
    blockers: tuple[str, ...]
    position_history: np.ndarray
    velocity_history: np.ndarray
    timestamp_history: np.ndarray
    history_valid_mask: np.ndarray


def _isotropic_covariance(value: float) -> np.ndarray:
    """Return an isotropic 2x2 covariance."""
    return np.eye(2, dtype=float) * float(value)


def _predict_state(state: _TrackState, dt_s: float, process_noise: float) -> _PredictedState:
    """Perform a constant-velocity prediction with acceleration process noise.

    Returns:
        Predicted mean and covariance scratch state.
    """
    dt = _non_negative(dt_s, "dt_s")
    position = state.position + state.velocity * dt
    dt2 = dt * dt
    process_position = process_noise * (dt2 * dt2 / 4.0)
    process_velocity = process_noise * dt2
    position_covariance = state.position_covariance + state.velocity_covariance * dt2
    position_covariance = position_covariance + _isotropic_covariance(process_position)
    velocity_covariance = state.velocity_covariance + _isotropic_covariance(process_velocity)
    return _PredictedState(
        position=np.array(position, dtype=float, copy=True),
        velocity=np.array(state.velocity, dtype=float, copy=True),
        position_covariance=_symmetrize_covariance(
            position_covariance, "predicted position covariance"
        ),
        velocity_covariance=_symmetrize_covariance(
            velocity_covariance, "predicted velocity covariance"
        ),
    )


def _kalman_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a small independent linear Gaussian measurement update.

    Returns:
        Updated mean and symmetric covariance.
    """
    innovation = measurement - mean
    innovation_covariance = covariance + measurement_covariance
    inverse = np.linalg.pinv(innovation_covariance, rcond=1e-12)
    gain = covariance @ inverse
    updated_mean = mean + gain @ innovation
    identity = np.eye(2, dtype=float)
    updated_covariance = (identity - gain) @ covariance
    return updated_mean, _symmetrize_covariance(updated_covariance, "updated covariance")


def _squared_distance(residual: np.ndarray, covariance: np.ndarray) -> float:
    """Return a finite covariance-normalized squared distance."""
    inverse = np.linalg.pinv(covariance, rcond=1e-12)
    value = float(residual @ inverse @ residual)
    if not math.isfinite(value):
        return math.inf
    return max(0.0, value)


def _history_append(
    state: _TrackState, position: np.ndarray, velocity: np.ndarray, timestamp_s: float, valid: bool
) -> None:
    """Append one oldest-to-newest history row using shared sensor semantics."""
    append_history_row(state.position_history, position)
    append_history_row(state.velocity_history, velocity)
    append_history_row(state.timestamp_history, np.array(timestamp_s, dtype=float))
    append_history_row(state.history_valid_mask, np.array(valid, dtype=bool))


def _observation_sort_key(
    observation: NormalizedPedestrianObservation, index: int
) -> tuple[Any, ...]:
    """Return a content-derived key independent of nearest-first slot order."""
    position = observation.positions_global_xy[index]
    velocity = observation.velocities_global_xy[index]
    covariance = (
        observation.position_covariances_global_xy[index].reshape(-1)
        if observation.position_covariances_global_xy is not None
        else np.zeros((4,), dtype=float)
    )
    velocity_covariance = (
        observation.velocity_covariances_global_xy[index].reshape(-1)
        if observation.velocity_covariances_global_xy is not None
        else np.zeros((4,), dtype=float)
    )
    return (
        float(position[0]),
        float(position[1]),
        int(not observation.velocity_valid_mask[index]),
        float(velocity[0]),
        float(velocity[1]),
        *(float(value) for value in covariance),
        *(float(value) for value in velocity_covariance),
        float(observation.radius[index]) if observation.radius is not None else 0.0,
        index,
    )


class PedestrianTracker:
    """Default-off deterministic observation-derived tracker.

    The tracker stores only normalized observations and mutable estimator state.
    ``last_observation_slot`` and source slots are retained for diagnostics but
    never participate in identity generation or tie-breaking except as a final
    ordering key for observationally indistinguishable rows.
    """

    def __init__(self, config: PedestrianTrackingConfig | Mapping[str, Any] | None = None) -> None:
        """Create a tracker with an immutable configuration."""
        if config is None:
            self.config = PedestrianTrackingConfig()
        elif isinstance(config, PedestrianTrackingConfig):
            self.config = config
        else:
            self.config = PedestrianTrackingConfig.from_mapping(config)
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1
        self._last_timestamp_s: float | None = None
        self._last_step_index: int | None = None

    @property
    def tracks(self) -> tuple[PedestrianTrack, ...]:
        """Return current non-retired tracks in deterministic ID order."""
        return tuple(
            self._public_track(self._tracks[track_id]) for track_id in sorted(self._tracks)
        )

    def reset(self) -> None:
        """Reset all estimator memory and restart episode-local IDs at one."""
        self._tracks.clear()
        self._next_track_id = 1
        self._last_timestamp_s = None
        self._last_step_index = None

    def update(  # noqa: C901, PLR0912, PLR0915
        self, snapshot: PedestrianObservationSnapshot
    ) -> PedestrianTrackingResult:
        """Consume one snapshot and return deterministic tracks plus diagnostics.

        Returns:
            An immutable tracking result for the snapshot timestamp.
        """
        if not isinstance(snapshot, PedestrianObservationSnapshot):
            raise TypeError("snapshot must be PedestrianObservationSnapshot")
        started = time.perf_counter()
        normalized = snapshot.to_global_xy()
        if not self.config.enabled:
            diagnostics = self._diagnostics(
                normalized,
                timestamp_gap_s=0.0,
                association_count=0,
                new_track_count=0,
                retired_track_count=0,
                cost_matrix_shape=(0, 0),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                blockers=("tracking_disabled",),
            )
            return PedestrianTrackingResult(
                timestamp_s=snapshot.timestamp_s,
                step_index=snapshot.step_index,
                tracks=(),
                associations=(),
                diagnostics=diagnostics,
            )
        if self._last_timestamp_s is not None and snapshot.timestamp_s < self._last_timestamp_s:
            raise ValueError("timestamp_s must be monotonically non-decreasing")
        if self._last_step_index is not None and snapshot.step_index <= self._last_step_index:
            raise ValueError("step_index must be strictly increasing")
        timestamp_gap_s = (
            0.0 if self._last_timestamp_s is None else snapshot.timestamp_s - self._last_timestamp_s
        )
        self._last_timestamp_s = snapshot.timestamp_s
        self._last_step_index = snapshot.step_index

        retired_states: list[_TrackState] = []
        eligible_states: list[tuple[_TrackState, _PredictedState, int, float]] = []
        for state in sorted(self._tracks.values(), key=lambda item: item.track_id):
            dt_s = snapshot.timestamp_s - state.timestamp_s
            gap_steps = max(1, snapshot.step_index - state.step_index)
            predicted = _predict_state(state, dt_s, self.config.process_noise)
            potential_missed_steps = state.missed_steps + max(0, gap_steps - 1)
            potential_missed_seconds = state.missed_seconds + dt_s
            if self._exceeds_missed_limit(potential_missed_steps, potential_missed_seconds):
                self._apply_miss(
                    state,
                    predicted,
                    normalized,
                    gap_steps=gap_steps,
                    dt_s=dt_s,
                    occluded=self._has_occlusion(normalized),
                    retire=True,
                )
                retired_states.append(state)
                self._tracks.pop(state.track_id, None)
            else:
                eligible_states.append((state, predicted, gap_steps, dt_s))

        detection_indices = [int(index) for index in np.flatnonzero(normalized.detection_mask)]
        detection_indices.sort(key=lambda index: _observation_sort_key(normalized, index))
        cost_matrix, cost_details = self._build_cost_matrix(
            eligible_states, normalized, detection_indices
        )
        matched_states: dict[int, int] = {}
        matched_observations: set[int] = set()
        associations: list[TrackAssociation] = []
        if cost_matrix.size > 0:
            assignment_cost = np.array(cost_matrix, dtype=float, copy=True)
            row_count, column_count = assignment_cost.shape
            for row in range(row_count):
                for column in range(column_count):
                    if math.isfinite(assignment_cost[row, column]):
                        assignment_cost[row, column] += self.config.tie_break_epsilon * (
                            row * (column_count + 1) + column
                        )
                    else:
                        assignment_cost[row, column] = 1e30
            rows, columns = linear_sum_assignment(assignment_cost)
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                if not math.isfinite(cost_matrix[row, column]):
                    continue
                state, _, _, _ = eligible_states[row]
                observation_index = detection_indices[column]
                matched_states[state.track_id] = observation_index
                matched_observations.add(observation_index)
                position_distance, velocity_distance, used_velocity = cost_details[row][column]
                confidence = self._association_confidence(
                    position_distance, velocity_distance, used_velocity
                )
                associations.append(
                    TrackAssociation(
                        track_id=state.track_id,
                        observation_slot=observation_index,
                        cost=float(cost_matrix[row, column]),
                        position_distance=position_distance,
                        velocity_distance=velocity_distance,
                        used_velocity=used_velocity,
                        confidence=confidence,
                    )
                )

        for row, (state, predicted, gap_steps, dt_s) in enumerate(eligible_states):
            observation_index = matched_states.get(state.track_id)
            if observation_index is None:
                retired = self._apply_miss(
                    state,
                    predicted,
                    normalized,
                    gap_steps=gap_steps,
                    dt_s=dt_s,
                    occluded=self._has_occlusion(normalized),
                    retire=False,
                )
                if retired:
                    self._tracks.pop(state.track_id, None)
                    retired_states.append(state)
                continue
            self._apply_match(
                state,
                predicted,
                normalized,
                observation_index,
                gap_steps=gap_steps,
                association=next(
                    association
                    for association in associations
                    if association.track_id == state.track_id
                ),
            )

        unmatched = [index for index in detection_indices if index not in matched_observations]
        capacity_exceeded = False
        new_track_count = 0
        for observation_index in unmatched:
            if len(self._tracks) >= self.config.max_tracks:
                capacity_exceeded = True
                break
            new_state = self._new_track(normalized, observation_index)
            self._tracks[new_state.track_id] = new_state
            new_track_count += 1

        output_states = list(self._tracks.values()) + retired_states
        output_states.sort(key=lambda item: item.track_id)
        output_tracks = tuple(self._public_track(state) for state in output_states)
        status_blockers = ["track_capacity_exceeded"] if capacity_exceeded else []
        diagnostics = self._diagnostics(
            normalized,
            timestamp_gap_s=timestamp_gap_s,
            association_count=len(associations),
            new_track_count=new_track_count,
            retired_track_count=len(retired_states),
            cost_matrix_shape=(int(cost_matrix.shape[0]), int(cost_matrix.shape[1])),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            blockers=tuple(status_blockers),
        )
        return PedestrianTrackingResult(
            timestamp_s=snapshot.timestamp_s,
            step_index=snapshot.step_index,
            tracks=output_tracks,
            associations=tuple(sorted(associations, key=lambda association: association.track_id)),
            diagnostics=diagnostics,
        )

    def _build_cost_matrix(
        self,
        eligible_states: Sequence[tuple[_TrackState, _PredictedState, int, float]],
        observation: NormalizedPedestrianObservation,
        detection_indices: Sequence[int],
    ) -> tuple[np.ndarray, list[list[tuple[float, float | None, bool]]]]:
        """Build a gated position/velocity innovation cost matrix.

        Returns:
            The raw cost matrix and position/velocity distance details.
        """
        row_count = len(eligible_states)
        column_count = len(detection_indices)
        costs = np.full((row_count, column_count), math.inf, dtype=float)
        details: list[list[tuple[float, float | None, bool]]] = [
            [(math.inf, None, False) for _ in range(column_count)] for _ in range(row_count)
        ]
        for row, (_, predicted, _, _) in enumerate(eligible_states):
            for column, observation_index in enumerate(detection_indices):
                position_covariance = (
                    observation.position_covariances_global_xy[observation_index]
                    if observation.position_covariances_global_xy is not None
                    else _isotropic_covariance(self.config.measurement_position_covariance)
                )
                position_residual = (
                    observation.positions_global_xy[observation_index] - predicted.position
                )
                if self.config.gating_mode == "mahalanobis":
                    position_distance = _squared_distance(
                        position_residual, predicted.position_covariance + position_covariance
                    )
                    position_allowed = position_distance <= self.config.position_gate_threshold
                else:
                    position_distance = float(np.linalg.norm(position_residual))
                    position_allowed = position_distance <= self.config.position_gate_threshold
                if not position_allowed:
                    continue
                velocity_distance: float | None = None
                used_velocity = bool(
                    self.config.use_velocity and observation.velocity_valid_mask[observation_index]
                )
                if used_velocity:
                    velocity_covariance = (
                        observation.velocity_covariances_global_xy[observation_index]
                        if observation.velocity_covariances_global_xy is not None
                        else _isotropic_covariance(self.config.measurement_velocity_covariance)
                    )
                    velocity_residual = (
                        observation.velocities_global_xy[observation_index] - predicted.velocity
                    )
                    if self.config.gating_mode == "mahalanobis":
                        velocity_distance = _squared_distance(
                            velocity_residual, predicted.velocity_covariance + velocity_covariance
                        )
                        velocity_allowed = velocity_distance <= self.config.velocity_gate_threshold
                    else:
                        velocity_distance = float(np.linalg.norm(velocity_residual))
                        velocity_allowed = velocity_distance <= self.config.velocity_gate_threshold
                    if not velocity_allowed:
                        continue
                position_score = (
                    position_distance / self.config.position_gate_threshold
                    if self.config.gating_mode == "mahalanobis"
                    else (position_distance / self.config.position_gate_threshold) ** 2
                )
                velocity_score = (
                    0.0
                    if velocity_distance is None
                    else (
                        velocity_distance / self.config.velocity_gate_threshold
                        if self.config.gating_mode == "mahalanobis"
                        else (velocity_distance / self.config.velocity_gate_threshold) ** 2
                    )
                )
                costs[row, column] = position_score + velocity_score
                details[row][column] = (position_distance, velocity_distance, used_velocity)
        return costs, details

    def _association_confidence(
        self,
        position_distance: float,
        velocity_distance: float | None,
        used_velocity: bool,
    ) -> float:
        """Convert normalized innovation distances into a bounded confidence.

        Returns:
            A confidence value in the closed unit interval.
        """
        position_ratio = position_distance / self.config.position_gate_threshold
        velocity_ratio = (
            0.0
            if velocity_distance is None
            else velocity_distance / self.config.velocity_gate_threshold
        )
        confidence = math.exp(-0.5 * (position_ratio + velocity_ratio))
        if not used_velocity:
            confidence *= 0.5
        return min(1.0, max(0.0, confidence))

    def _new_track(
        self,
        observation: NormalizedPedestrianObservation,
        observation_index: int,
    ) -> _TrackState:
        """Create a new monotonic-ID track from one visible observation.

        Returns:
            The mutable internal state for the new track.
        """
        position = np.array(
            observation.positions_global_xy[observation_index], dtype=float, copy=True
        )
        if observation.velocity_valid_mask[observation_index]:
            velocity = np.array(
                observation.velocities_global_xy[observation_index], dtype=float, copy=True
            )
            confidence = 1.0
            blockers: tuple[str, ...] = ()
        else:
            velocity = np.zeros((2,), dtype=float)
            confidence = 0.5
            blockers = ("velocity_unavailable",)
        position_covariance = (
            np.array(
                observation.position_covariances_global_xy[observation_index],
                dtype=float,
                copy=True,
            )
            if observation.position_covariances_global_xy is not None
            else _isotropic_covariance(self.config.initial_position_covariance)
        )
        velocity_covariance = (
            np.array(
                observation.velocity_covariances_global_xy[observation_index],
                dtype=float,
                copy=True,
            )
            if observation.velocity_covariances_global_xy is not None
            and observation.velocity_valid_mask[observation_index]
            else _isotropic_covariance(self.config.initial_velocity_covariance)
        )
        capacity = self.config.history_capacity
        state = _TrackState(
            track_id=self._next_track_id,
            position=position,
            velocity=velocity,
            position_covariance=_symmetrize_covariance(position_covariance, "position covariance"),
            velocity_covariance=_symmetrize_covariance(velocity_covariance, "velocity covariance"),
            timestamp_s=observation.timestamp_s,
            step_index=observation.step_index,
            age_steps=1,
            visible_age_steps=1,
            missed_steps=0,
            missed_seconds=0.0,
            status=(
                TrackStatus.CONFIRMED
                if self.config.confirmation_steps <= 1
                else TrackStatus.TENTATIVE
            ),
            association_confidence=confidence,
            last_observation_slot=observation_index,
            blockers=blockers,
            position_history=np.zeros((capacity, 2), dtype=float),
            velocity_history=np.zeros((capacity, 2), dtype=float),
            timestamp_history=np.zeros((capacity,), dtype=float),
            history_valid_mask=np.zeros((capacity,), dtype=bool),
        )
        _history_append(state, position, velocity, observation.timestamp_s, True)
        self._next_track_id += 1
        return state

    def _apply_match(
        self,
        state: _TrackState,
        predicted: _PredictedState,
        observation: NormalizedPedestrianObservation,
        observation_index: int,
        *,
        gap_steps: int,
        association: TrackAssociation,
    ) -> None:
        """Update a track with one accepted visible observation."""
        position_measurement = observation.positions_global_xy[observation_index]
        position_measurement_covariance = (
            observation.position_covariances_global_xy[observation_index]
            if observation.position_covariances_global_xy is not None
            else _isotropic_covariance(self.config.measurement_position_covariance)
        )
        position, position_covariance = _kalman_update(
            predicted.position,
            predicted.position_covariance,
            position_measurement,
            position_measurement_covariance,
        )
        velocity = predicted.velocity
        velocity_covariance = predicted.velocity_covariance
        blockers: tuple[str, ...] = ()
        if observation.velocity_valid_mask[observation_index]:
            velocity_measurement_covariance = (
                observation.velocity_covariances_global_xy[observation_index]
                if observation.velocity_covariances_global_xy is not None
                else _isotropic_covariance(self.config.measurement_velocity_covariance)
            )
            velocity, velocity_covariance = _kalman_update(
                predicted.velocity,
                predicted.velocity_covariance,
                observation.velocities_global_xy[observation_index],
                velocity_measurement_covariance,
            )
        else:
            blockers = ("velocity_unavailable",)
        state.position = position
        state.velocity = velocity
        state.position_covariance = position_covariance
        state.velocity_covariance = velocity_covariance
        state.timestamp_s = observation.timestamp_s
        state.step_index = observation.step_index
        state.age_steps += gap_steps
        state.visible_age_steps += 1
        state.missed_steps = 0
        state.missed_seconds = 0.0
        if state.status is not TrackStatus.CONFIRMED:
            state.status = (
                TrackStatus.CONFIRMED
                if state.visible_age_steps >= self.config.confirmation_steps
                else TrackStatus.TENTATIVE
            )
        state.association_confidence = association.confidence
        state.last_observation_slot = observation_index
        state.blockers = blockers
        _history_append(state, position, velocity, observation.timestamp_s, True)

    def _apply_miss(
        self,
        state: _TrackState,
        predicted: _PredictedState,
        observation: NormalizedPedestrianObservation,
        *,
        gap_steps: int,
        dt_s: float,
        occluded: bool,
        retire: bool,
    ) -> bool:
        """Advance a lost track and optionally retire it after the configured limit.

        Returns:
            Whether the track reached the retirement limit during this update.
        """
        state.position = predicted.position
        state.velocity = predicted.velocity
        state.position_covariance = predicted.position_covariance
        state.velocity_covariance = predicted.velocity_covariance
        state.timestamp_s = observation.timestamp_s
        state.step_index = observation.step_index
        state.age_steps += gap_steps
        state.missed_steps += gap_steps
        state.missed_seconds += dt_s
        should_retire = retire or self._exceeds_missed_limit(
            state.missed_steps, state.missed_seconds
        )
        state.status = TrackStatus.RETIRED if should_retire else TrackStatus.LOST
        state.association_confidence = max(0.0, state.association_confidence * 0.75)
        blockers = ["prediction_only"]
        blockers.append("occluded" if occluded else "unobserved")
        if should_retire:
            blockers.append("retired_after_missed_limit")
        state.blockers = tuple(blockers)
        _history_append(state, state.position, state.velocity, observation.timestamp_s, False)
        return should_retire

    def _exceeds_missed_limit(self, missed_steps: int, missed_seconds: float) -> bool:
        """Return whether a track can no longer be reacquired."""
        if missed_seconds > self.config.max_missed_seconds:
            return True
        return (
            self.config.max_missed_steps is not None and missed_steps > self.config.max_missed_steps
        )

    @staticmethod
    def _has_occlusion(observation: NormalizedPedestrianObservation) -> bool:
        """Distinguish explicit invisible valid rows from padding/unavailable rows.

        Returns:
            Whether the observation explicitly carries an invisible valid row.
        """
        return bool(np.any(observation.valid_mask & ~observation.visible_mask))

    def _public_track(self, state: _TrackState) -> PedestrianTrack:
        """Copy mutable state into the immutable output contract.

        Returns:
            A defensive immutable track snapshot.
        """
        return PedestrianTrack(
            track_id=state.track_id,
            timestamp_s=state.timestamp_s,
            step_index=state.step_index,
            position_global_xy=state.position,
            velocity_global_xy=state.velocity,
            position_covariance=state.position_covariance,
            velocity_covariance=state.velocity_covariance,
            age_steps=state.age_steps,
            visible_age_steps=state.visible_age_steps,
            missed_steps=state.missed_steps,
            status=state.status,
            association_confidence=state.association_confidence,
            last_observation_slot=state.last_observation_slot,
            history_valid_mask=state.history_valid_mask,
            position_history_global_xy=state.position_history,
            velocity_history_global_xy=state.velocity_history,
            timestamp_history_s=state.timestamp_history,
            blockers=state.blockers,
            config_hash=self.config.config_hash,
        )

    def _diagnostics(
        self,
        observation: NormalizedPedestrianObservation,
        *,
        timestamp_gap_s: float,
        association_count: int,
        new_track_count: int,
        retired_track_count: int,
        cost_matrix_shape: tuple[int, int],
        latency_ms: float,
        blockers: tuple[str, ...],
    ) -> PedestrianTrackingDiagnostics:
        """Construct bounded per-update counters.

        Returns:
            Immutable counters for the current update.
        """
        active_tracks = tuple(self._tracks.values())
        return PedestrianTrackingDiagnostics(
            enabled=self.config.enabled,
            timestamp_gap_s=timestamp_gap_s,
            input_row_count=observation.positions_global_xy.shape[0],
            valid_row_count=int(np.count_nonzero(observation.valid_mask)),
            visible_row_count=int(np.count_nonzero(observation.visible_mask)),
            detection_count=int(np.count_nonzero(observation.detection_mask)),
            invalid_row_count=int(np.count_nonzero(~observation.valid_mask)),
            velocity_unavailable_count=int(
                np.count_nonzero(observation.detection_mask & ~observation.velocity_valid_mask)
            ),
            association_count=association_count,
            new_track_count=max(0, new_track_count),
            lost_track_count=sum(track.status is TrackStatus.LOST for track in active_tracks),
            retired_track_count=retired_track_count,
            active_track_count=len(active_tracks),
            tentative_track_count=sum(
                track.status is TrackStatus.TENTATIVE for track in active_tracks
            ),
            confirmed_track_count=sum(
                track.status is TrackStatus.CONFIRMED for track in active_tracks
            ),
            cost_matrix_shape=cost_matrix_shape,
            estimated_cost_matrix_bytes=cost_matrix_shape[0] * cost_matrix_shape[1] * 8,
            update_latency_ms=latency_ms,
            blockers=blockers,
        )


@dataclass(frozen=True, slots=True)
class OracleTrackingMetrics:
    """Diagnostic metrics from evaluator-only identity linkage."""

    assignment_accuracy: float | None
    identity_switches: int
    track_fragmentation: int
    false_matches: int
    missed_detections: int
    reacquisition_accuracy: float | None
    time_to_confirmation: Mapping[str, int]
    error_by_visibility: Mapping[str, float | None]
    goal_inference_history_coverage: float
    blockers: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Freeze metric mappings and validate bounded values."""
        for field_name in ("assignment_accuracy", "reacquisition_accuracy"):
            value = getattr(self, field_name)
            if value is not None:
                normalized = _finite_scalar(value, field_name)
                if not 0.0 <= normalized <= 1.0:
                    raise ValueError(f"{field_name} must be between 0 and 1")
                object.__setattr__(self, field_name, normalized)
        for field_name in (
            "identity_switches",
            "track_fragmentation",
            "false_matches",
            "missed_detections",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        coverage = _finite_scalar(
            self.goal_inference_history_coverage, "goal_inference_history_coverage"
        )
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("goal_inference_history_coverage must be between 0 and 1")
        object.__setattr__(self, "goal_inference_history_coverage", coverage)
        object.__setattr__(
            self, "time_to_confirmation", MappingProxyType(dict(self.time_to_confirmation))
        )
        object.__setattr__(
            self, "error_by_visibility", MappingProxyType(dict(self.error_by_visibility))
        )
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))


class OracleTrackingEvaluator:
    """Score continuity after tracking without feeding identity back to the actor.

    ``simulator_identity_by_observation_slot`` is accepted only by this evaluator
    method.  The actor-side ``PedestrianTracker`` has no parameter or state slot
    for identity, so changing the evaluator's permutation cannot alter tracking.
    Metrics are diagnostic continuity evidence, not a benchmark claim.
    """

    def __init__(self) -> None:
        """Initialize evaluator-only linkage memory."""
        self.reset()

    def reset(self) -> None:
        """Clear evaluator linkage for a new episode."""
        self._track_identity: dict[int, str] = {}
        self._identity_track_ids: dict[str, set[int]] = {}
        self._last_track_step: dict[int, int] = {}
        self._reacquisition_total = 0
        self._reacquisition_correct = 0
        self._identity_first_step: dict[str, int] = {}
        self._identity_confirmation_step: dict[str, int] = {}
        self._identity_errors: dict[str, list[float]] = {"visible": [], "occluded": []}
        self._identity_switches = 0
        self._assignment_total = 0
        self._assignment_correct = 0
        self._frames = 0

    def evaluate(  # noqa: C901
        self,
        result: PedestrianTrackingResult,
        simulator_identity_by_observation_slot: Mapping[int, str],
        *,
        simulator_position_global_xy_by_identity: Mapping[str, Sequence[float]] | None = None,
    ) -> OracleTrackingMetrics:
        """Evaluate one result using identity only in this post-tracking path.

        Returns:
            Evaluator-only continuity and identity-linkage metrics.
        """
        if not isinstance(result, PedestrianTrackingResult):
            raise TypeError("result must be PedestrianTrackingResult")
        identity_by_slot = {
            int(slot): str(identity)
            for slot, identity in simulator_identity_by_observation_slot.items()
        }
        positions_by_identity: dict[str, np.ndarray] = {}
        if simulator_position_global_xy_by_identity is not None:
            for identity, position in simulator_position_global_xy_by_identity.items():
                array = np.asarray(position, dtype=float)
                if array.shape != (2,) or not np.all(np.isfinite(array)):
                    raise ValueError("simulator identity positions must be finite XY values")
                positions_by_identity[str(identity)] = array
        matched_identity_by_track: dict[int, str] = {}
        correct = 0
        reacquisition_total = 0
        reacquisition_correct = 0
        duplicate_counts: dict[str, int] = {}
        for association in result.associations:
            observed_identity = identity_by_slot.get(association.observation_slot)
            if observed_identity is None:
                continue
            track_id = association.track_id
            matched_identity_by_track[track_id] = observed_identity
            duplicate_counts[observed_identity] = duplicate_counts.get(observed_identity, 0) + 1
            prior_identity = self._track_identity.get(track_id)
            if prior_identity is None:
                self._track_identity[track_id] = observed_identity
                self._identity_track_ids.setdefault(observed_identity, set()).add(track_id)
                correct += 1
            elif prior_identity == observed_identity:
                correct += 1
            else:
                self._identity_switches += 1
                self._track_identity[track_id] = observed_identity
                self._identity_track_ids.setdefault(observed_identity, set()).add(track_id)
            previous_step = self._last_track_step.get(track_id)
            if previous_step is not None and result.step_index > previous_step + 1:
                reacquisition_total += 1
                if prior_identity == observed_identity:
                    reacquisition_correct += 1
            self._last_track_step[track_id] = result.step_index
            self._identity_first_step.setdefault(observed_identity, result.step_index)
            track = result.track(track_id)
            if track is not None and track.status is TrackStatus.CONFIRMED:
                self._identity_confirmation_step.setdefault(observed_identity, result.step_index)
            if observed_identity in positions_by_identity and track is not None:
                error = float(
                    np.linalg.norm(
                        track.position_global_xy - positions_by_identity[observed_identity]
                    )
                )
                self._identity_errors["visible"].append(error)
        self._reacquisition_total += reacquisition_total
        self._reacquisition_correct += reacquisition_correct
        self._assignment_total += len(matched_identity_by_track)
        self._assignment_correct += correct
        self._frames += 1
        all_identities = set(identity_by_slot.values())
        missed = len(all_identities - set(matched_identity_by_track.values()))
        false_matches = sum(max(0, count - 1) for count in duplicate_counts.values())
        fragmentation = sum(
            max(0, len(track_ids) - 1) for track_ids in self._identity_track_ids.values()
        )
        history_coverage = (
            float(np.mean([np.mean(track.history_valid_mask) for track in result.tracks]))
            if result.tracks
            else 0.0
        )
        error_by_visibility = {
            "visible": (
                float(np.mean(self._identity_errors["visible"]))
                if self._identity_errors["visible"]
                else None
            ),
            "occluded": None,
        }
        blockers = ("occluded_error_requires_identity_positions_for_hidden_rows",)
        return OracleTrackingMetrics(
            assignment_accuracy=(
                self._assignment_correct / self._assignment_total
                if self._assignment_total
                else None
            ),
            identity_switches=self._identity_switches,
            track_fragmentation=fragmentation,
            false_matches=false_matches,
            missed_detections=missed,
            reacquisition_accuracy=(
                self._reacquisition_correct / self._reacquisition_total
                if self._reacquisition_total
                else None
            ),
            time_to_confirmation={
                identity: self._identity_confirmation_step[identity] - first_step
                for identity, first_step in self._identity_first_step.items()
                if identity in self._identity_confirmation_step
            },
            error_by_visibility=error_by_visibility,
            goal_inference_history_coverage=history_coverage,
            blockers=blockers,
        )


__all__ = [
    "CANONICAL_FRAME",
    "HISTORY_ORDER",
    "PEDESTRIAN_TRACKING_SCHEMA_VERSION",
    "CoordinateFrame",
    "NormalizedPedestrianObservation",
    "OracleTrackingEvaluator",
    "OracleTrackingMetrics",
    "PedestrianCoordinateFrame",
    "PedestrianObservationSnapshot",
    "PedestrianTrack",
    "PedestrianTracker",
    "PedestrianTrackingConfig",
    "PedestrianTrackingDiagnostics",
    "PedestrianTrackingResult",
    "RobotPoseGlobal",
    "TrackAssociation",
    "TrackStatus",
    "TrackingStatus",
    "canonical_config_hash",
    "covariance_from_global_xy",
    "covariance_to_global_xy",
    "heading_from_global_xy",
    "heading_to_global_xy",
    "history_from_global_xy",
    "history_to_global_xy",
    "pedestrian_tracking_config_from_spec",
    "position_from_global_xy",
    "position_to_global_xy",
    "transform_covariance_from_global_xy",
    "transform_covariance_to_global_xy",
    "transform_heading_from_global_xy",
    "transform_heading_to_global_xy",
    "transform_history_from_global_xy",
    "transform_history_to_global_xy",
    "transform_position_from_global_xy",
    "transform_position_to_global_xy",
    "transform_velocity_from_global_xy",
    "transform_velocity_to_global_xy",
    "velocity_from_global_xy",
    "velocity_to_global_xy",
]


def canonical_config_hash(config: PedestrianTrackingConfig) -> str:
    """Return the config hash through a named side-channel helper."""
    if not isinstance(config, PedestrianTrackingConfig):
        raise TypeError("config must be PedestrianTrackingConfig")
    return config.config_hash
