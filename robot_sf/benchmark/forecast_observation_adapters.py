"""Observation-tier adapters for ForecastBatch.v1 inputs.

These adapters deliberately separate oracle/full-state trace inputs from
deployable tracked-observation inputs before forecast artifacts are built.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
)
from robot_sf.benchmark.pedestrian_forecast import (
    DEFAULT_FORECAST_HORIZONS_S,
    PedestrianState,
    constant_velocity_gaussian_baseline,
)


def _require_non_empty_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _require_feature_schema(feature_schema: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(feature_schema, dict) or not feature_schema:
        raise ValueError("feature_schema is required")
    return {str(key): value for key, value in feature_schema.items()}


def _frame_at(trace: dict[str, Any], step_index: int) -> dict[str, Any]:
    frames = trace.get("frames") or trace.get("steps")
    if not isinstance(frames, list):
        raise ValueError("trace must contain frames or steps")
    if step_index < 0 or step_index >= len(frames):
        raise ValueError("step_index is outside trace frames")
    frame = frames[step_index]
    if not isinstance(frame, dict):
        raise ValueError("trace frame must be a mapping")
    return frame


def _default_dt_s(trace: dict[str, Any]) -> float:
    frames = trace.get("frames") or trace.get("steps") or []
    if len(frames) < 2:
        return 0.1
    try:
        dt_s = float(frames[1].get("time_s", 0.1)) - float(frames[0].get("time_s", 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.1
    return dt_s if dt_s > 0.0 else 0.1


def _stable_state_id(actor_id: object) -> int:
    if isinstance(actor_id, bool):
        raise ValueError("actor id must not be bool")
    if isinstance(actor_id, (float, np.floating)):
        digest = hashlib.sha256(str(actor_id).encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**31)
    try:
        return int(actor_id)
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(actor_id).encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**31)


def _actor_id_label(payload: dict[str, Any]) -> str:
    raw_id = payload.get("actor_id", payload.get("id"))
    return _require_non_empty_str("actor id", str(raw_id) if raw_id is not None else "")


def _actor_available(payload: dict[str, Any]) -> bool:
    if payload.get("masked") is True or payload.get("occluded") is True:
        return False
    for key in ("forecast_available", "visible", "tracked"):
        if key in payload:
            return bool(payload[key])
    return True


def _missing_reason(payload: dict[str, Any], default: str) -> str:
    reason = payload.get("missing_reason") or payload.get("mask_reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = default
    return _require_non_empty_str("missing_actor_reasons[]", str(reason))


def _state_from_payload(payload: dict[str, Any]) -> PedestrianState:
    normalized = dict(payload)
    normalized["id"] = _stable_state_id(payload.get("id", payload.get("actor_id")))
    return PedestrianState.from_trace(normalized)


@dataclass(frozen=True)
class ForecastActorObservation:
    """One actor accepted by a forecast observation adapter."""

    actor_id: str
    state: PedestrianState


@dataclass(frozen=True)
class ForecastObservationBatch:
    """Typed adapter output before predictor-specific forecasts are generated."""

    provenance: ForecastBatchProvenance
    actors: tuple[ForecastActorObservation, ...]


@dataclass(frozen=True)
class ForecastObservationAdapter:
    """Convert one declared trace observation tier into forecast-ready states."""

    observation_tier: str
    source_key: str
    oracle_state: bool
    missing_reason: str
    predictor_id: str = "constant-velocity-gaussian-v1"
    predictor_family: str = "constant_velocity"

    def __post_init__(self) -> None:
        """Validate adapter declarations before any trace can be adapted."""
        _require_non_empty_str("observation_tier", self.observation_tier)
        _require_non_empty_str("source_key", self.source_key)
        _require_non_empty_str("missing_reason", self.missing_reason)
        _require_non_empty_str("predictor_id", self.predictor_id)
        _require_non_empty_str("predictor_family", self.predictor_family)

    def adapt_trace(
        self,
        trace: dict[str, Any],
        *,
        feature_schema: dict[str, Any],
        frame: CoordinateFrame | None = None,
        horizons_s: list[float] | tuple[float, ...] = DEFAULT_FORECAST_HORIZONS_S,
        dt_s: float | None = None,
        step_index: int = 0,
        expected_actor_ids: list[str] | tuple[str, ...] | None = None,
    ) -> ForecastObservationBatch:
        """Adapt one trace frame into typed actor states and provenance.

        Returns:
            Forecast-ready actor observations plus ForecastBatch.v1 provenance.
        """

        schema = _require_feature_schema(feature_schema)
        frame_payload = _frame_at(trace, step_index)
        raw_actors = frame_payload.get(self.source_key)
        if raw_actors is None:
            raise ValueError(
                f"observation tier '{self.observation_tier}' missing {self.source_key}"
            )
        if not isinstance(raw_actors, list):
            raise ValueError(f"{self.source_key} must be a list")

        (
            actor_ids,
            actor_mask,
            missing_actor_reasons,
            actor_classes,
            observations,
        ) = self._collect_actor_observations(raw_actors, expected_actor_ids)

        provenance = ForecastBatchProvenance(
            predictor_id=self.predictor_id,
            predictor_family=self.predictor_family,
            observation_tier=self.observation_tier,
            frame=frame or CoordinateFrame(name="world", units="m", axes=("x", "y")),
            dt_s=dt_s if dt_s is not None else _default_dt_s(trace),
            horizons_s=list(horizons_s),
            scenario_id=str(
                trace.get("scenario_id")
                or trace.get("metadata", {}).get("scenario_id")
                or "unknown"
            ),
            seed=int(trace.get("seed", trace.get("metadata", {}).get("seed", 0))),
            fallback_status="native",
            degraded_status="degraded_observation" if missing_actor_reasons else "none",
            actor_ids=actor_ids,
            actor_mask=actor_mask,
            actor_mask_metadata={
                "semantics": "true means the actor was available in the declared observation tier",
                "source_key": self.source_key,
                "missing_actor_reasons": missing_actor_reasons,
            },
            feature_schema=schema,
            oracle_state=self.oracle_state,
            actor_classes=actor_classes,
        )
        return ForecastObservationBatch(provenance=provenance, actors=tuple(observations))

    def _collect_actor_observations(
        self,
        raw_actors: list[object],
        expected_actor_ids: list[str] | tuple[str, ...] | None,
    ) -> tuple[
        list[str],
        list[bool],
        dict[str, str],
        dict[str, str],
        list[ForecastActorObservation],
    ]:
        """Normalize actor denominators, masks, classes, and included states.

        Returns:
            Actor ids, mask flags, missing reasons, classes, and included observations.
        """
        actors_by_id = self._actors_by_id(raw_actors)
        actor_ids = (
            list(expected_actor_ids) if expected_actor_ids is not None else list(actors_by_id)
        )
        actor_ids = [_require_non_empty_str("actor_ids[]", str(actor_id)) for actor_id in actor_ids]
        if not actor_ids:
            raise ValueError("actor_ids must be non-empty")
        if len(set(actor_ids)) != len(actor_ids):
            raise ValueError("actor_ids must be unique")

        observations: list[ForecastActorObservation] = []
        actor_mask: list[bool] = []
        missing_actor_reasons: dict[str, str] = {}
        actor_classes: dict[str, str] = {}
        for actor_id in actor_ids:
            payload = actors_by_id.get(actor_id)
            if payload is None:
                actor_mask.append(False)
                missing_actor_reasons[actor_id] = self.missing_reason
                continue
            actor_class = str(payload.get("actor_type") or "pedestrian").strip() or "pedestrian"
            actor_classes[actor_id] = actor_class
            if not _actor_available(payload):
                actor_mask.append(False)
                missing_actor_reasons[actor_id] = _missing_reason(payload, self.missing_reason)
                continue
            actor_mask.append(True)
            observations.append(
                ForecastActorObservation(actor_id=actor_id, state=_state_from_payload(payload)),
            )
        return actor_ids, actor_mask, missing_actor_reasons, actor_classes, observations

    def _actors_by_id(self, raw_actors: list[object]) -> dict[str, dict[str, Any]]:
        """Return source actors keyed by stable actor id."""
        actors_by_id: dict[str, dict[str, Any]] = {}
        for payload in raw_actors:
            if not isinstance(payload, dict):
                raise ValueError(f"{self.source_key} actors must be mappings")
            actor_id = _actor_id_label(payload)
            if actor_id in actors_by_id:
                raise ValueError("actor ids must be unique within an observation tier")
            actors_by_id[actor_id] = payload
        return actors_by_id


class OracleFullStateForecastAdapter(ForecastObservationAdapter):
    """Adapter for privileged simulator state."""

    def __init__(self) -> None:
        """Create the privileged oracle/full-state adapter."""
        super().__init__(
            observation_tier="oracle_full_state",
            source_key="pedestrians",
            oracle_state=True,
            missing_reason="missing from oracle pedestrian state",
        )


class TrackedAgentsForecastAdapter(ForecastObservationAdapter):
    """Adapter for deployable tracked-agent observations."""

    def __init__(self) -> None:
        """Create the deployable tracked-agent adapter."""
        super().__init__(
            observation_tier="tracked_agents",
            source_key="tracked_agents",
            oracle_state=False,
            missing_reason="not present in tracked-agent observation",
        )


def build_constant_velocity_forecast_batch(
    observation: ForecastObservationBatch,
) -> ForecastBatch:
    """Build a deterministic CV ForecastBatch from adapted observations.

    Returns:
        Validated ForecastBatch.v1 artifact for smoke evaluation.
    """

    forecasts = []
    for actor in observation.actors:
        forecast = constant_velocity_gaussian_baseline(
            actor.state,
            horizons_s=observation.provenance.horizons_s,
        )
        forecasts.append(
            ActorForecast(
                actor_id=actor.actor_id,
                deterministic=np.asarray(
                    [prediction.mean for prediction in forecast.predictions],
                    dtype=float,
                ),
            ),
        )
    return ForecastBatch(
        provenance=observation.provenance,
        forecasts=forecasts,
        metadata={
            "artifact_role": "forecast_observation_adapter_smoke",
            "observation_tier": observation.provenance.observation_tier,
        },
    )


__all__ = [
    "ForecastActorObservation",
    "ForecastObservationAdapter",
    "ForecastObservationBatch",
    "OracleFullStateForecastAdapter",
    "TrackedAgentsForecastAdapter",
    "build_constant_velocity_forecast_batch",
]
