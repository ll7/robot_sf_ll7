"""ForecastBatch.v1 artifact contract for benchmark forecast interchange.

The schema is an artifact boundary, not a prediction-quality claim. It records
enough provenance for benchmark consumers to decide whether a forecast was
produced from deployable observations, degraded/fallback execution, or an
explicit oracle-state source before using it in downstream evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

FORECAST_BATCH_SCHEMA_VERSION = "ForecastBatch.v1"


def _require_non_empty_str(name: str, value: object) -> str:
    """Return a stripped string or raise for missing required provenance."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _require_positive_float(name: str, value: object) -> float:
    """Return a finite positive float."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_float_list(name: str, value: object) -> list[float]:
    """Return a non-empty list of finite positive floats."""
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list or array")
    parsed = [_require_positive_float(f"{name}[]", item) for item in value]
    if any(later <= earlier for earlier, later in pairwise(parsed)):
        raise ValueError(f"{name} must be strictly increasing")
    return parsed


def _require_mapping(name: str, value: object) -> dict[str, Any]:
    """Return a mapping with string keys."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _normalize_actor_classes(value: object, actor_ids: list[str]) -> dict[str, str]:
    """Return optional actor-class labels keyed by actor id."""
    if value is None:
        return {}
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != len(actor_ids):
            raise ValueError("actor_classes list must align with actor_ids")
        value = {
            actor_id: actor_class
            for actor_id, actor_class in zip(actor_ids, value, strict=True)
            if actor_class is not None
        }
    classes = _require_mapping("actor_classes", value)
    unknown_actor_ids = sorted(set(classes) - set(actor_ids))
    if unknown_actor_ids:
        raise ValueError("actor_classes keys must be declared in actor_ids")
    return {
        actor_id: _require_non_empty_str("actor_classes[]", actor_class)
        for actor_id, actor_class in classes.items()
        if actor_class is not None
    }


def _contains_oracle_key(value: object) -> bool:
    """True when nested metadata contains oracle-looking keys or values.

    Returns:
        True when nested metadata would blur deployable and oracle state.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized != "oracle_state" and "oracle" in normalized:
                return True
            if _contains_oracle_key(item):
                return True
    elif isinstance(value, list | tuple):
        return any(_contains_oracle_key(item) for item in value)
    elif isinstance(value, str):
        return "oracle" in value.lower()
    return False


@dataclass
class CoordinateFrame:
    """Coordinate-frame semantics for forecast positions.

    Attributes:
        name: Stable frame label such as ``"world"`` or ``"robot"``.
        units: Position units. ForecastBatch.v1 expects meters.
        axes: Ordered axis names for each point coordinate.
        origin: Optional human-readable origin description.
    """

    name: str
    units: str = "m"
    axes: tuple[str, str] = ("x", "y")
    origin: str | None = None

    def __post_init__(self) -> None:
        """Validate coordinate-frame fields."""
        self.name = _require_non_empty_str("frame.name", self.name)
        self.units = _require_non_empty_str("frame.units", self.units)
        if self.units != "m":
            raise ValueError("frame.units must be 'm'")
        if len(self.axes) != 2:
            raise ValueError("frame.axes must contain exactly two axes")
        self.axes = tuple(_require_non_empty_str("frame.axes[]", axis) for axis in self.axes)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoordinateFrame:
        """Build a coordinate frame from JSON-compatible data.

        Returns:
            Validated coordinate-frame metadata.
        """
        return cls(
            name=data.get("name", ""),
            units=data.get("units", "m"),
            axes=tuple(data.get("axes", ("x", "y"))),
            origin=data.get("origin"),
        )


@dataclass
class ForecastBatchProvenance:
    """Required provenance for a ForecastBatch.v1 artifact."""

    predictor_id: str
    predictor_family: str
    observation_tier: str
    frame: CoordinateFrame
    dt_s: float
    horizons_s: list[float]
    scenario_id: str
    seed: int
    fallback_status: str
    degraded_status: str
    actor_ids: list[str]
    actor_mask: list[bool]
    actor_mask_metadata: dict[str, Any]
    feature_schema: dict[str, Any]
    oracle_state: bool = False
    actor_classes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fail closed on missing provenance or inconsistent actor masks."""
        self.predictor_id = _require_non_empty_str("predictor_id", self.predictor_id)
        self.predictor_family = _require_non_empty_str("predictor_family", self.predictor_family)
        self.observation_tier = _require_non_empty_str("observation_tier", self.observation_tier)
        if not isinstance(self.frame, CoordinateFrame):
            self.frame = CoordinateFrame.from_dict(_require_mapping("frame", self.frame))
        self.dt_s = _require_positive_float("dt_s", self.dt_s)
        self.horizons_s = _require_float_list("horizons_s", self.horizons_s)
        self.scenario_id = _require_non_empty_str("scenario_id", self.scenario_id)
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.seed = int(self.seed)
        self.fallback_status = _require_non_empty_str("fallback_status", self.fallback_status)
        self.degraded_status = _require_non_empty_str("degraded_status", self.degraded_status)
        if not isinstance(self.actor_ids, (list, tuple, np.ndarray)) or len(self.actor_ids) == 0:
            raise ValueError("actor_ids must be a non-empty list, tuple, or array")
        self.actor_ids = [
            _require_non_empty_str("actor_ids[]", actor_id) for actor_id in self.actor_ids
        ]
        if len(set(self.actor_ids)) != len(self.actor_ids):
            raise ValueError("actor_ids must be unique")
        self.actor_classes = _normalize_actor_classes(self.actor_classes, self.actor_ids)
        if not isinstance(self.actor_mask, (list, tuple, np.ndarray)) or len(
            self.actor_mask
        ) != len(self.actor_ids):
            raise ValueError("actor_mask must align with actor_ids")
        if not all(isinstance(value, (bool, np.bool_)) for value in self.actor_mask):
            raise ValueError("actor_mask values must be boolean")
        self.actor_mask = [bool(value) for value in self.actor_mask]
        self.actor_mask_metadata = _require_mapping(
            "actor_mask_metadata",
            self.actor_mask_metadata,
        )
        if not self.actor_mask_metadata:
            raise ValueError("actor_mask_metadata is required")
        self.feature_schema = _require_mapping("feature_schema", self.feature_schema)
        if not self.feature_schema:
            raise ValueError("feature_schema is required")
        self.oracle_state = bool(self.oracle_state)
        if (
            _contains_oracle_key(self.feature_schema)
            or _contains_oracle_key(self.actor_mask_metadata)
        ) and not self.oracle_state:
            raise ValueError("oracle fields require explicit oracle_state=True")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastBatchProvenance:
        """Build provenance from JSON-compatible data.

        Returns:
            Validated forecast-batch provenance.
        """
        frame = data.get("frame")
        if isinstance(frame, dict):
            frame = CoordinateFrame.from_dict(frame)
        return cls(
            predictor_id=data.get("predictor_id", ""),
            predictor_family=data.get("predictor_family", ""),
            observation_tier=data.get("observation_tier", ""),
            frame=frame,
            dt_s=data.get("dt_s"),
            horizons_s=data.get("horizons_s"),
            scenario_id=data.get("scenario_id", ""),
            seed=data.get("seed", 0),
            fallback_status=data.get("fallback_status", ""),
            degraded_status=data.get("degraded_status", ""),
            actor_ids=list(data.get("actor_ids", [])),
            actor_mask=list(data.get("actor_mask", [])),
            actor_mask_metadata=data.get("actor_mask_metadata", {}),
            feature_schema=data.get("feature_schema", {}),
            actor_classes=data.get("actor_classes", {}),
            oracle_state=bool(data.get("oracle_state", False)),
        )


def _array_or_none(name: str, value: object, *, ndim: int) -> np.ndarray | None:
    """Normalize an optional numeric array.

    Returns:
        Float array or None when the optional payload is absent.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape ending in (..., 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass
class ActorForecast:
    """Optional forecast payloads for one actor.

    Every predictor may emit any subset of deterministic trajectories, samples,
    modes, occupancy summaries, and uncertainty metadata as long as the shared
    provenance is present.
    """

    actor_id: str
    deterministic: np.ndarray | None = None
    samples: np.ndarray | None = None
    mode_probabilities: list[float] | None = None
    occupancy_summary: dict[str, Any] | None = None
    uncertainty_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate per-actor forecast payload shapes."""
        self.actor_id = _require_non_empty_str("actor_id", self.actor_id)
        self.deterministic = _array_or_none("deterministic", self.deterministic, ndim=2)
        self.samples = _array_or_none("samples", self.samples, ndim=3)
        if self.mode_probabilities is not None:
            probs = np.asarray(self.mode_probabilities, dtype=float)
            if probs.ndim != 1 or probs.size == 0 or not np.all(np.isfinite(probs)):
                raise ValueError("mode_probabilities must be a finite 1D list")
            if np.any(probs < 0.0) or not np.isclose(float(np.sum(probs)), 1.0):
                raise ValueError("mode_probabilities must be non-negative and sum to 1")
            self.mode_probabilities = [float(value) for value in probs]
            if self.samples is not None and len(self.mode_probabilities) != self.samples.shape[0]:
                raise ValueError("mode_probabilities must align with samples")
        if self.occupancy_summary is not None:
            self.occupancy_summary = _require_mapping("occupancy_summary", self.occupancy_summary)
        if self.uncertainty_metadata is not None:
            self.uncertainty_metadata = _require_mapping(
                "uncertainty_metadata",
                self.uncertainty_metadata,
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActorForecast:
        """Build an actor forecast from JSON-compatible data.

        Returns:
            Validated actor forecast payload.
        """
        return cls(
            actor_id=data.get("actor_id", ""),
            deterministic=data.get("deterministic"),
            samples=data.get("samples"),
            mode_probabilities=data.get("mode_probabilities"),
            occupancy_summary=data.get("occupancy_summary"),
            uncertainty_metadata=data.get("uncertainty_metadata"),
        )


@dataclass
class ForecastBatch:
    """Versioned, JSON-serializable forecast artifact."""

    provenance: ForecastBatchProvenance
    forecasts: list[ActorForecast]
    schema_version: str = FORECAST_BATCH_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate schema version, provenance, payload alignment, and oracle scope."""
        if self.schema_version != FORECAST_BATCH_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {FORECAST_BATCH_SCHEMA_VERSION}")
        if not isinstance(self.provenance, ForecastBatchProvenance):
            self.provenance = ForecastBatchProvenance.from_dict(
                _require_mapping("provenance", self.provenance),
            )
        self.forecasts = [
            forecast if isinstance(forecast, ActorForecast) else ActorForecast.from_dict(forecast)
            for forecast in self.forecasts
        ]
        forecast_ids = [forecast.actor_id for forecast in self.forecasts]
        if len(forecast_ids) != len(set(forecast_ids)):
            raise ValueError("forecasts must not contain duplicate actor_ids")
        forecast_id_set = set(forecast_ids)
        expected_ids = {
            actor_id
            for actor_id, included in zip(
                self.provenance.actor_ids,
                self.provenance.actor_mask,
                strict=True,
            )
            if included
        }
        if forecast_id_set != expected_ids:
            raise ValueError("forecast actor_ids must exactly match actor_mask=True actors")
        expected_steps = len(self.provenance.horizons_s)
        for forecast in self.forecasts:
            if (
                forecast.deterministic is not None
                and forecast.deterministic.shape[0] != expected_steps
            ):
                raise ValueError("deterministic trajectories must align with horizons_s")
            if forecast.samples is not None and forecast.samples.shape[1] != expected_steps:
                raise ValueError("sampled trajectories must align with horizons_s")
        self.metadata = _require_mapping("metadata", self.metadata)
        if (
            _contains_oracle_key(self.metadata)
            or any(
                _contains_oracle_key(forecast.occupancy_summary)
                or _contains_oracle_key(forecast.uncertainty_metadata)
                for forecast in self.forecasts
            )
        ) and not self.provenance.oracle_state:
            raise ValueError("oracle fields require explicit oracle_state=True")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastBatch:
        """Build and validate a ForecastBatch from JSON-compatible data.

        Returns:
            Validated forecast batch.
        """
        return cls(
            schema_version=data.get("schema_version", ""),
            provenance=ForecastBatchProvenance.from_dict(data.get("provenance", {})),
            forecasts=[ActorForecast.from_dict(item) for item in data.get("forecasts", [])],
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return as_dict(self)


def as_dict(value: Any) -> Any:
    """Convert dataclasses and numpy arrays to JSON-compatible values.

    Returns:
        JSON-compatible value.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {item.name: as_dict(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): as_dict(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [as_dict(item) for item in value]
    return value


def load_forecast_batch(path: str | Path) -> ForecastBatch:
    """Load a ForecastBatch.v1 JSON artifact from disk.

    Returns:
        Validated forecast batch loaded from JSON.
    """
    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("forecast batch JSON must be an object")
    return ForecastBatch.from_dict(data)


def save_forecast_batch(batch: ForecastBatch, path: str | Path) -> None:
    """Write a ForecastBatch.v1 JSON artifact to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(batch.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")


def validate_forecast_batch(data: ForecastBatch | dict[str, Any]) -> ForecastBatch:
    """Return a validated ForecastBatch, raising ValueError on contract failures."""
    if isinstance(data, ForecastBatch):
        data.__post_init__()
        return data
    return ForecastBatch.from_dict(data)


__all__ = [
    "FORECAST_BATCH_SCHEMA_VERSION",
    "ActorForecast",
    "CoordinateFrame",
    "ForecastBatch",
    "ForecastBatchProvenance",
    "load_forecast_batch",
    "save_forecast_batch",
    "validate_forecast_batch",
]
