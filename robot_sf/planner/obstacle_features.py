"""Deterministic local obstacle features for predictive planner inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.common.types import Line2D, Vec2D
from robot_sf.errors import RobotSfError

PREDICTIVE_OBSTACLE_FEATURE_SCHEMA = "predictive_obstacle_features_v1"
PREDICTIVE_OBSTACLE_FEATURE_DIM = 6
PREDICTIVE_OBSTACLE_UNAVAILABLE_FEATURE_ROW = (50.0, 0.0, 0.0, 0.0, 0.0, 0.0)
PREDICTIVE_LEGACY_FEATURE_SCHEMA = "predictive_legacy_v1"
PREDICTIVE_LEGACY_FEATURE_DIM = 4
PREDICTIVE_EGO_FEATURE_SCHEMA = "predictive_ego_v1"
PREDICTIVE_EGO_FEATURE_DIM = 9
PREDICTIVE_EGO_MOTION_CHANNEL_SLOTS = (4, 5)
PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME = "same_seed_hardcase_runtime_robot_speed_v1"
PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE = "standalone_rollout_velocity_xy_preferred_v1"


class ObstacleFeatureSchemaError(RobotSfError, ValueError):
    """Raised when obstacle-feature metadata is incompatible with the expected schema."""


@dataclass(frozen=True)
class ObstacleFeatureSchema:
    """Stable metadata for predictive obstacle feature vectors."""

    name: str = PREDICTIVE_OBSTACLE_FEATURE_SCHEMA
    feature_dim: int = PREDICTIVE_OBSTACLE_FEATURE_DIM

    def to_metadata(self) -> dict[str, int | str]:
        """Return JSON-compatible schema metadata.

        Returns
        -------
        dict[str, int | str]
            Schema name and feature dimension.
        """
        return {"name": self.name, "feature_dim": self.feature_dim}

    def validate_metadata(self, metadata: dict[str, object]) -> None:
        """Fail closed if serialized metadata does not match this schema."""
        actual_name = metadata.get("name")
        actual_dim = metadata.get("feature_dim")
        if actual_name != self.name:
            raise ObstacleFeatureSchemaError(
                f"Obstacle feature schema mismatch: expected {self.name!r}, got {actual_name!r}"
            )
        if actual_dim != self.feature_dim:
            raise ObstacleFeatureSchemaError(
                "Obstacle feature dimension mismatch: "
                f"expected {self.feature_dim}, got {actual_dim!r}"
            )


@dataclass(frozen=True)
class LocalObstacleFeature:
    """Nearest-obstacle feature vector for one query point.

    Feature order:
    ``[distance, normal_x, normal_y, tangent_x, tangent_y, valid_mask]``.
    """

    distance: float
    normal: Vec2D
    tangent: Vec2D
    valid_mask: float

    def as_array(self) -> np.ndarray:
        """Return the feature as a float32 vector.

        Returns
        -------
        np.ndarray
            Six-element vector in the stable predictive-obstacle feature order.
        """
        return np.asarray(
            [
                self.distance,
                self.normal[0],
                self.normal[1],
                self.tangent[0],
                self.tangent[1],
                self.valid_mask,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class LocalObstacleFeatureExtractor:
    """Extract deterministic nearest-line obstacle features."""

    unavailable_distance: float = PREDICTIVE_OBSTACLE_UNAVAILABLE_FEATURE_ROW[0]

    def __post_init__(self) -> None:
        """Validate unavailable-feature sentinel configuration."""
        if not np.isfinite(self.unavailable_distance) or self.unavailable_distance < 0.0:
            raise ValueError("unavailable_distance must be a finite, non-negative float")

    @property
    def schema(self) -> str:
        """Return the stable feature schema identifier.

        Returns
        -------
        str
            Predictive obstacle feature schema name.
        """
        return PREDICTIVE_OBSTACLE_FEATURE_SCHEMA

    @property
    def feature_dim(self) -> int:
        """Return the feature vector width.

        Returns
        -------
        int
            Number of elements in each obstacle feature row.
        """
        return PREDICTIVE_OBSTACLE_FEATURE_DIM

    @property
    def schema_metadata(self) -> dict[str, int | str]:
        """Return JSON-compatible metadata for downstream datasets/checkpoints.

        Returns
        -------
        dict[str, int | str]
            Stable schema name and feature dimension.
        """
        return ObstacleFeatureSchema().to_metadata()

    def extract(self, query_point: Vec2D, obstacle_lines: Iterable[Line2D]) -> np.ndarray:
        """Return nearest-obstacle features for one query point.

        Returns
        -------
        np.ndarray
            One six-element predictive obstacle feature vector.
        """
        feature = self._nearest_feature(query_point, list(obstacle_lines))
        return feature.as_array()

    def extract_many(
        self,
        query_points: Iterable[Vec2D],
        obstacle_lines: Iterable[Line2D],
    ) -> np.ndarray:
        """Return nearest-obstacle features for multiple query points.

        Returns
        -------
        np.ndarray
            ``(num_query_points, 6)`` feature matrix.
        """
        lines = list(obstacle_lines)
        rows = [self._nearest_feature(point, lines).as_array() for point in query_points]
        if not rows:
            return np.empty((0, PREDICTIVE_OBSTACLE_FEATURE_DIM), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _nearest_feature(self, query_point: Vec2D, lines: list[Line2D]) -> LocalObstacleFeature:
        """Compute nearest-line feature with deterministic tie-breaking.

        Returns
        -------
        LocalObstacleFeature
            Nearest-line feature or unavailable sentinel when no valid line exists.
        """
        if not lines:
            return LocalObstacleFeature(
                distance=self.unavailable_distance,
                normal=(0.0, 0.0),
                tangent=(0.0, 0.0),
                valid_mask=0.0,
            )

        point = np.asarray(query_point, dtype=float)
        best: tuple[float, int, np.ndarray, np.ndarray] | None = None
        for index, line in enumerate(lines):
            start = np.asarray(line[0], dtype=float)
            segment = np.asarray(line[1], dtype=float) - start
            length_sq = float(np.dot(segment, segment))
            if length_sq <= 0.0:
                continue
            rel_point = point - start
            projection = float(np.dot(rel_point, segment) / length_sq)
            projection = float(np.clip(projection, 0.0, 1.0))
            offset = rel_point - projection * segment
            distance = float(np.linalg.norm(offset))
            candidate = (distance, index, offset, segment)
            if best is None or (distance, index) < (best[0], best[1]):
                best = candidate

        if best is None:
            return LocalObstacleFeature(
                distance=self.unavailable_distance,
                normal=(0.0, 0.0),
                tangent=(0.0, 0.0),
                valid_mask=0.0,
            )

        distance, _index, offset, segment = best
        normal = _unit_or_zero(offset)
        tangent = _unit_or_zero(segment)
        return LocalObstacleFeature(
            distance=distance,
            normal=(float(normal[0]), float(normal[1])),
            tangent=(float(tangent[0]), float(tangent[1])),
            valid_mask=1.0,
        )


def _coerce_point(value: Any) -> tuple[float, float] | None:
    """Return a finite 2D point tuple, or ``None`` for malformed input."""
    try:
        x, y = value
        x_f = float(x)
        y_f = float(y)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x_f) or not np.isfinite(y_f):
        return None
    return (x_f, y_f)


def _line_from_pair(value: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Coerce pair-of-point line geometry into the canonical ``Line2D`` shape.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]] | None
        Canonical line segment, or ``None`` for malformed or degenerate input.
    """
    try:
        start, end = value
    except (TypeError, ValueError):
        return None
    start_point = _coerce_point(start)
    end_point = _coerce_point(end)
    if start_point is None or end_point is None or start_point == end_point:
        return None
    return (start_point, end_point)


def _line_from_flat_xxyy(value: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Coerce legacy map flat ``(x1, x2, y1, y2)`` line geometry.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]] | None
        Canonical line segment, or ``None`` for malformed or degenerate input.
    """
    try:
        x1, x2, y1, y2 = value
    except (TypeError, ValueError):
        return None
    return _line_from_pair(((x1, y1), (x2, y2)))


def _line_from_flat_xyxy(value: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Coerce raw simulator-style flat ``(x1, y1, x2, y2)`` line geometry.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]] | None
        Canonical line segment, or ``None`` for malformed or degenerate input.
    """
    try:
        x1, y1, x2, y2 = value
    except (TypeError, ValueError):
        return None
    return _line_from_pair(((x1, y1), (x2, y2)))


def _append_map_line(lines: list[Line2D], raw_line: Any) -> None:
    """Append one legacy map line when it can be normalized."""
    candidate = _line_from_flat_xxyy(raw_line)
    if candidate is None:
        candidate = _line_from_pair(raw_line)
    if candidate is not None:
        lines.append(candidate)


def obstacle_lines_from_map(map_def: Any) -> list[Line2D]:
    """Return deterministic obstacle and bound lines from a map definition.

    MapDefinition stores obstacle and bound lines in the legacy flat
    ``(x1, x2, y1, y2)`` order, while other callers may already expose pair-of-point
    ``Line2D`` values. Malformed entries are skipped so unavailable geometry degrades
    to the explicit sentinel features rather than corrupting the schema.

    Returns
    -------
    list[Line2D]
        Canonical obstacle and bound line segments, preserving map order.
    """
    if map_def is None:
        return []
    lines: list[Line2D] = []

    for obstacle in getattr(map_def, "obstacles", []) or []:
        raw_lines = getattr(obstacle, "lines", None)
        if raw_lines is None:
            _append_map_line(lines, obstacle)
            continue
        for raw_line in raw_lines:
            _append_map_line(lines, raw_line)

    for raw_bound in getattr(map_def, "bounds", []) or []:
        _append_map_line(lines, raw_bound)

    return lines


def normalize_obstacle_lines(obstacle_lines: Any) -> list[Line2D]:
    """Normalize explicit raw obstacle-line payloads into canonical line tuples.

    Explicit payloads are interpreted as pair-of-point lines or flat simulator-style
    ``(x1, y1, x2, y2)`` rows. Use :func:`obstacle_lines_from_map` for
    ``MapDefinition`` objects because their flat lines use legacy ``(x1, x2, y1, y2)``
    order.

    Returns
    -------
    list[Line2D]
        Canonical line segments, skipping malformed or degenerate rows.
    """
    if obstacle_lines is None:
        return []
    arr = np.asarray(obstacle_lines, dtype=object)
    if arr.size == 0:
        return []
    values: list[Any]
    if arr.ndim == 1 and arr.size == 4:
        values = [arr.tolist()]
    else:
        values = list(obstacle_lines)
    lines: list[Line2D] = []
    for value in values:
        candidate = _line_from_pair(value)
        if candidate is None:
            candidate = _line_from_flat_xyxy(value)
        if candidate is not None:
            lines.append(candidate)
    return lines


def obstacle_lines_from_observation(observation: dict[str, Any]) -> list[Line2D]:
    """Return obstacle lines exposed by a structured runtime observation, if any."""
    if not isinstance(observation, dict):
        return []
    map_payload = observation.get("map")
    if isinstance(map_payload, dict):
        lines = normalize_obstacle_lines(map_payload.get("obstacle_lines"))
        if lines:
            return lines
    for key in ("obstacle_lines", "obstacles"):
        lines = normalize_obstacle_lines(observation.get(key))
        if lines:
            return lines
    return []


def _unit_or_zero(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector or zeros for degenerate vectors.

    Returns
    -------
    np.ndarray
        Unit-length vector, or a two-element zero vector for degenerate input.
    """
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return np.zeros(2, dtype=float)
    return vector / norm


def append_obstacle_features(
    agent_features: np.ndarray,
    obstacle_features: np.ndarray,
    *,
    schema_metadata: dict[str, object],
) -> np.ndarray:
    """Append validated obstacle feature rows to predictive agent features.

    Returns
    -------
    np.ndarray
        Concatenated feature matrix with obstacle features appended to each agent row.
    """
    ObstacleFeatureSchema().validate_metadata(schema_metadata)
    agents = np.asarray(agent_features, dtype=np.float32)
    obstacles = np.asarray(obstacle_features, dtype=np.float32)
    if agents.ndim != 2:
        raise ValueError("agent_features must have shape (agents, features)")
    if obstacles.ndim != 2:
        raise ValueError("obstacle_features must have shape (agents, obstacle_features)")
    if agents.shape[0] != obstacles.shape[0]:
        raise ValueError(
            "agent_features and obstacle_features must have the same number of rows "
            f"(got {agents.shape[0]} and {obstacles.shape[0]})"
        )
    if obstacles.shape[1] != PREDICTIVE_OBSTACLE_FEATURE_DIM:
        raise ObstacleFeatureSchemaError(
            "Obstacle feature dimension mismatch: "
            f"expected {PREDICTIVE_OBSTACLE_FEATURE_DIM}, got {obstacles.shape[1]}"
        )
    return np.concatenate([agents, obstacles], axis=1)


def predictive_feature_schema_metadata(
    *,
    model_family: str,
    ego_conditioning: bool = False,
    ego_motion_channel_producer: str | None = None,
) -> dict[str, object]:
    """Return predictive input-schema metadata for a dataset/model family.

    Returns
    -------
    dict[str, object]
        JSON-compatible metadata describing the predictive feature schema.
    """
    normalized = str(model_family).strip() or PREDICTIVE_LEGACY_FEATURE_SCHEMA
    if normalized == PREDICTIVE_OBSTACLE_FEATURE_SCHEMA:
        base_dim = (
            PREDICTIVE_EGO_FEATURE_DIM if bool(ego_conditioning) else PREDICTIVE_LEGACY_FEATURE_DIM
        )
        base_schema = (
            PREDICTIVE_EGO_FEATURE_SCHEMA
            if bool(ego_conditioning)
            else PREDICTIVE_LEGACY_FEATURE_SCHEMA
        )
        return {
            "name": PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
            "base_schema": base_schema,
            "base_feature_dim": base_dim,
            "obstacle_feature_schema": ObstacleFeatureSchema().to_metadata(),
            "input_dim": base_dim + PREDICTIVE_OBSTACLE_FEATURE_DIM,
            "ego_motion_channel_producer": _predictive_ego_motion_channel_producer_metadata(
                ego_motion_channel_producer,
                base_schema=base_schema,
            ),
        }
    if normalized == PREDICTIVE_LEGACY_FEATURE_SCHEMA:
        return {
            "name": PREDICTIVE_LEGACY_FEATURE_SCHEMA,
            "base_schema": PREDICTIVE_LEGACY_FEATURE_SCHEMA,
            "base_feature_dim": PREDICTIVE_LEGACY_FEATURE_DIM,
            "obstacle_feature_schema": None,
            "input_dim": PREDICTIVE_LEGACY_FEATURE_DIM,
            "ego_motion_channel_producer": None,
        }
    if normalized == PREDICTIVE_EGO_FEATURE_SCHEMA:
        return {
            "name": PREDICTIVE_EGO_FEATURE_SCHEMA,
            "base_schema": PREDICTIVE_EGO_FEATURE_SCHEMA,
            "base_feature_dim": PREDICTIVE_EGO_FEATURE_DIM,
            "obstacle_feature_schema": None,
            "input_dim": PREDICTIVE_EGO_FEATURE_DIM,
            "ego_motion_channel_producer": _predictive_ego_motion_channel_producer_metadata(
                ego_motion_channel_producer,
                base_schema=PREDICTIVE_EGO_FEATURE_SCHEMA,
            ),
        }
    raise ObstacleFeatureSchemaError(f"Unsupported predictive feature schema: {model_family!r}")


def infer_predictive_feature_schema(state_dim: int) -> dict[str, object]:
    """Infer legacy predictive feature schema metadata from an input dimension.

    Returns
    -------
    dict[str, object]
        JSON-compatible schema metadata.
    """
    dim = int(state_dim)
    if dim == PREDICTIVE_LEGACY_FEATURE_DIM:
        return predictive_feature_schema_metadata(model_family=PREDICTIVE_LEGACY_FEATURE_SCHEMA)
    if dim == PREDICTIVE_EGO_FEATURE_DIM:
        return predictive_feature_schema_metadata(model_family=PREDICTIVE_EGO_FEATURE_SCHEMA)
    if dim in {
        PREDICTIVE_LEGACY_FEATURE_DIM + PREDICTIVE_OBSTACLE_FEATURE_DIM,
        PREDICTIVE_EGO_FEATURE_DIM + PREDICTIVE_OBSTACLE_FEATURE_DIM,
    }:
        return {
            "name": PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
            "base_schema": PREDICTIVE_EGO_FEATURE_SCHEMA
            if dim == PREDICTIVE_EGO_FEATURE_DIM + PREDICTIVE_OBSTACLE_FEATURE_DIM
            else PREDICTIVE_LEGACY_FEATURE_SCHEMA,
            "base_feature_dim": dim - PREDICTIVE_OBSTACLE_FEATURE_DIM,
            "obstacle_feature_schema": ObstacleFeatureSchema().to_metadata(),
            "input_dim": dim,
            "ego_motion_channel_producer": None,
        }
    raise ObstacleFeatureSchemaError(f"Cannot infer predictive feature schema for input_dim={dim}")


def _base_schema_feature_dim(schema_name: str) -> int | None:
    """Return the exact input width for non-obstacle predictive schemas."""
    if schema_name == PREDICTIVE_LEGACY_FEATURE_SCHEMA:
        return PREDICTIVE_LEGACY_FEATURE_DIM
    if schema_name == PREDICTIVE_EGO_FEATURE_SCHEMA:
        return PREDICTIVE_EGO_FEATURE_DIM
    return None


def _predictive_ego_motion_channel_producer_metadata(
    producer: str | None,
    *,
    base_schema: str,
) -> dict[str, object] | None:
    """Return machine-readable producer metadata for ego motion slots when applicable."""
    if base_schema != PREDICTIVE_EGO_FEATURE_SCHEMA:
        return None
    normalized = str(producer or "").strip()
    if not normalized:
        return None
    comparability = {
        "group": "predictive_ego_motion_slots_v1",
        "same_producer_key_required": True,
        "mixed_producer_status": "not_comparable_without_caveat",
    }
    slot_range = list(PREDICTIVE_EGO_MOTION_CHANNEL_SLOTS)
    if normalized == PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME:
        return {
            "producer_key": PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
            "slot_range": slot_range,
            "source": {
                "observation_key": "robot.speed",
                "channel_labels": ["linear_speed", "angular_speed"],
            },
            "semantics": "benchmark_linear_angular_speed",
            "fallback": None,
            "comparability": comparability,
        }
    if normalized == PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE:
        return {
            "producer_key": PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE,
            "slot_range": slot_range,
            "source": {
                "observation_key": "robot.velocity_xy",
                "channel_labels": ["vx_world", "vy_world"],
                "selection": "preferred_when_available",
            },
            "semantics": "world_velocity_xy_with_robot_speed_fallback",
            "fallback": {
                "observation_key": "robot.speed",
                "channel_labels": ["linear_speed", "angular_speed"],
                "condition": "when_robot_velocity_xy_missing_or_short",
            },
            "comparability": comparability,
        }
    raise ObstacleFeatureSchemaError(
        f"Unsupported predictive ego motion producer metadata: {producer!r}"
    )


def predictive_ego_motion_channel_producer_key(metadata: dict[str, object]) -> str | None:
    """Return the stable ego-motion producer key from predictive schema metadata."""
    producer = metadata.get("ego_motion_channel_producer")
    if not isinstance(producer, dict):
        return None
    producer_key = producer.get("producer_key")
    if producer_key is None:
        return None
    return str(producer_key).strip() or None


def _validate_predictive_ego_motion_channel_producer(metadata: dict[str, object]) -> None:
    """Validate optional ego-motion producer metadata without requiring it on legacy artifacts."""
    if "ego_motion_channel_producer" not in metadata:
        return
    producer_meta = metadata.get("ego_motion_channel_producer")
    if producer_meta is None:
        return
    if not isinstance(producer_meta, dict):
        raise ObstacleFeatureSchemaError("ego_motion_channel_producer must be a mapping")
    producer_key = predictive_ego_motion_channel_producer_key(metadata)
    if producer_key is None:
        raise ObstacleFeatureSchemaError(
            "ego_motion_channel_producer.producer_key must be a non-empty string when provided"
        )
    base_schema = str(metadata.get("base_schema") or "").strip()
    if base_schema != PREDICTIVE_EGO_FEATURE_SCHEMA:
        raise ObstacleFeatureSchemaError(
            "Predictive ego motion producer metadata is only valid for "
            f"base_schema={PREDICTIVE_EGO_FEATURE_SCHEMA!r}; got {base_schema!r}"
        )
    _predictive_ego_motion_channel_producer_metadata(
        producer_key,
        base_schema=base_schema,
    )


def validate_predictive_feature_schema_metadata(
    metadata: dict[str, object],
    *,
    input_dim: int,
    expected_schema_name: str | None = None,
) -> None:
    """Fail closed when predictive feature metadata and input dimensions disagree."""
    if not isinstance(metadata, dict):
        raise ObstacleFeatureSchemaError("Predictive feature schema metadata must be a mapping")
    schema_name = str(metadata.get("name") or "").strip()
    if expected_schema_name is not None and schema_name != str(expected_schema_name):
        raise ObstacleFeatureSchemaError(
            "Predictive feature schema mismatch: "
            f"expected {expected_schema_name!r}, got {schema_name!r}"
        )
    metadata_dim = int(metadata.get("input_dim", -1))
    if metadata_dim != int(input_dim):
        raise ObstacleFeatureSchemaError(
            "Predictive input dimension mismatch: "
            f"metadata input_dim={metadata_dim}, model input_dim={int(input_dim)}"
        )
    _validate_predictive_ego_motion_channel_producer(metadata)
    if schema_name == PREDICTIVE_OBSTACLE_FEATURE_SCHEMA:
        obstacle_meta = metadata.get("obstacle_feature_schema")
        if not isinstance(obstacle_meta, dict):
            raise ObstacleFeatureSchemaError("Obstacle feature schema metadata is required")
        ObstacleFeatureSchema().validate_metadata(obstacle_meta)
        base_dim = int(metadata.get("base_feature_dim", -1))
        expected_dim = base_dim + PREDICTIVE_OBSTACLE_FEATURE_DIM
        if expected_dim != int(input_dim):
            raise ObstacleFeatureSchemaError(
                "Predictive obstacle feature dimension mismatch: "
                f"base_dim={base_dim}, obstacle_dim={PREDICTIVE_OBSTACLE_FEATURE_DIM}, "
                f"input_dim={int(input_dim)}"
            )
    else:
        expected_base_dim = _base_schema_feature_dim(schema_name)
        if expected_base_dim is None:
            raise ObstacleFeatureSchemaError(
                f"Unsupported predictive feature schema: {schema_name!r}"
            )
        if int(input_dim) != expected_base_dim:
            feature_label = "legacy" if schema_name == PREDICTIVE_LEGACY_FEATURE_SCHEMA else "ego"
            raise ObstacleFeatureSchemaError(
                f"Predictive {feature_label} feature dimension mismatch: "
                f"expected {expected_base_dim}, got {int(input_dim)}"
            )


def validate_predictive_runtime_feature_schema(metadata: dict[str, object]) -> None:
    """Reject ego-conditioned checkpoints whose motion-channel producer differs from runtime.

    The structural schema validator accepts either registered ego-motion producer so that
    artifacts remain self-describing. Runtime planning is stricter: its generated robot-speed
    channels must use the same producer as the checkpoint. Keep this comparison shared so
    CPU-only preflight audits and :class:`PredictionPlannerAdapter` fail on the same condition.
    """
    if not isinstance(metadata, dict):
        return
    if str(metadata.get("base_schema") or "").strip() != PREDICTIVE_EGO_FEATURE_SCHEMA:
        return
    actual_producer = predictive_ego_motion_channel_producer_key(metadata)
    if actual_producer is None:
        return
    expected_producer = predictive_ego_motion_channel_producer_key(
        predictive_feature_schema_metadata(
            model_family=str(metadata.get("name") or ""),
            ego_conditioning=True,
            ego_motion_channel_producer=PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
        )
    )
    if actual_producer != expected_producer:
        raise ObstacleFeatureSchemaError(
            "Predictive ego motion producer mismatch: "
            f"runtime expects {expected_producer!r}, got {actual_producer!r}"
        )
