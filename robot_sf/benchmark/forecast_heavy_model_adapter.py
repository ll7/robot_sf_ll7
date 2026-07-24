"""Adapter FROM heavy-model predictions TO ForecastBatch.v1 / ActorForecast.

This is the offline-prediction adapter scaffold for issue #2845. It does NOT
ship or invoke any transformer / CVAE / diffusion model. It defines the
conversion contract so a real heavy model's output can be scored through the
existing forecast metrics, calibration, and baseline-comparison surfaces.

Scope and safety:

- ``build_heavy_model_forecast_batch`` is the sole public entry point. It
  consumes per-actor raw predictions (numpy arrays + dict metadata) and produces
  a validated ``ForecastBatch``.
- The adapter is *stateless*: it does not hold model weights, inference state,
  or mutable configuration.
- Fail-closed: missing provenance fields or shape misalignment raises
  ``ValueError`` before any forecast artifact is emitted.
- This module adds **no** new third-party ML dependency (torch, diffusers,
  etc.). The heavy model itself is injected from outside the adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
)

__all__ = [
    "FORECAST_HEAVY_MODEL_ADAPTER_VERSION",
    "HeavyModelAdapterConfig",
    "HeavyModelForecastAdapter",
    "HeavyModelPredictionPayload",
    "build_heavy_model_forecast_batch",
]

FORECAST_HEAVY_MODEL_ADAPTER_VERSION = "forecast_heavy_model_adapter.v1"

DEFAULT_PREDICTOR_ID = "heavy_model_offline_study"
DEFAULT_PREDICTOR_FAMILY = "heavy_model_adapter_placeholder"


@dataclass(frozen=True)
class HeavyModelAdapterConfig:
    """Configuration for one heavy-model forecast batch conversion.

    Attributes:
        predictor_id:
            Stable identifier for the predictor that produced the raw output.
        predictor_family:
            Family label such as ``"transformer"``, ``"cvae"``, ``"diffusion"``.
        observation_tier:
            Observation tier (e.g. ``"deployable"``, ``"oracle_full_state"``).
        dt_s:
            Simulation timestep in seconds.
        horizons_s:
            Strictly increasing forecast horizons in seconds.
        scenario_id:
            Scenario identifier from the evaluation trace.
        seed:
            Random seed used by the predictor or scenario.
        frame:
            Coordinate frame metadata. Defaults to world frame in meters.
        fallback_status:
            Whether the predictor fell back to a simpler method.
        degraded_status:
            Whether the predictor operated under degraded inputs.
        oracle_state:
            Whether the predictions use oracle (non-deployable) state.
    """

    predictor_id: str = DEFAULT_PREDICTOR_ID
    predictor_family: str = DEFAULT_PREDICTOR_FAMILY
    observation_tier: str = "offline_study"
    dt_s: float = 0.1
    horizons_s: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    scenario_id: str = "heavy_model_offline_scenario"
    seed: int = 0
    frame: CoordinateFrame | None = None
    fallback_status: str = "none"
    degraded_status: str = "none"
    oracle_state: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible config dictionary."""
        return {
            "predictor_id": self.predictor_id,
            "predictor_family": self.predictor_family,
            "observation_tier": self.observation_tier,
            "dt_s": self.dt_s,
            "horizons_s": list(self.horizons_s),
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "fallback_status": self.fallback_status,
            "degraded_status": self.degraded_status,
            "oracle_state": self.oracle_state,
        }


@dataclass
class HeavyModelPredictionPayload:
    """Optional per-batch prediction payload bundle.

    Encapsulates all optional prediction outputs that ``build_batch`` converts
    into the ``ForecastBatch`` contract. Using a single bundle avoids excessive
    method-argument counts while keeping each field independently optional.
    """

    samples: np.ndarray | None = None
    mode_probabilities: list[float] | None = None
    gaussian_params: list[dict[str, Any]] | None = None
    reachable_set: list[dict[str, Any]] | None = None
    occupancy_summary: dict[str, Any] | None = None
    uncertainty_metadata: dict[str, Any] | None = None
    actor_classes: dict[str, str] | None = None
    feature_schema: dict[str, Any] | None = None
    actor_mask_metadata: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class HeavyModelForecastAdapter:
    """Stateless adapter from raw heavy-model predictions to ForecastBatch.v1.

    This adapter does NOT own a model. Callers inject raw predictions as numpy
    arrays and dict metadata; the adapter validates shapes, attaches provenance,
    and returns a ``ForecastBatch`` ready for the metric/calibration surfaces.

    Usage::

        adapter = HeavyModelForecastAdapter()
        payload = HeavyModelPredictionPayload(
            samples=samples_array,
            actor_classes={"ped_0": "pedestrian"},
        )
        batch = adapter.build_batch(
            config=cfg,
            actor_ids=["ped_0", "ped_1"],
            deterministic_trajectories=det_array,
            payload=payload,
        )
    """

    def __init__(self) -> None:
        """Initialize the stateless adapter."""

    def build_batch(
        self,
        config: HeavyModelAdapterConfig,
        actor_ids: list[str],
        deterministic_trajectories: np.ndarray,
        payload: HeavyModelPredictionPayload | None = None,
    ) -> ForecastBatch:
        """Convert raw per-actor predictions into a validated ForecastBatch.

        Args:
            config: Conversion configuration (predictor identity, domain params).
            actor_ids: Ordered list of actor identifiers.
            deterministic_trajectories:
                Shape ``(n_actors, n_horizons, 2)`` — one deterministic
                trajectory per actor.  Must be finite.
            payload:
                Optional bundle of additional prediction outputs (samples,
                mode probabilities, Gaussian params, reachable sets, etc).

        Returns:
            Validated ``ForecastBatch`` ready for metric evaluation.

        Raises:
            ValueError: On shape mismatch, missing required fields, or
                non-finite values.
        """
        p = payload or HeavyModelPredictionPayload()
        n_actors = len(actor_ids)
        n_horizons = len(config.horizons_s)

        det = np.asarray(deterministic_trajectories, dtype=float)
        if (
            det.ndim != 3
            or det.shape[0] != n_actors
            or det.shape[1] != n_horizons
            or det.shape[2] != 2
        ):
            raise ValueError(
                f"deterministic_trajectories must have shape ({n_actors}, {n_horizons}, 2), "
                f"got {det.shape}"
            )
        if not np.all(np.isfinite(det)):
            raise ValueError("deterministic_trajectories must contain only finite values")

        if p.samples is not None:
            sam = np.asarray(p.samples, dtype=float)
            if (
                sam.ndim != 4
                or sam.shape[0] != n_actors
                or sam.shape[2] != n_horizons
                or sam.shape[3] != 2
            ):
                raise ValueError(
                    f"samples must have shape ({n_actors}, n_samples, {n_horizons}, 2), "
                    f"got {sam.shape}"
                )
            if not np.all(np.isfinite(sam)):
                raise ValueError("samples must contain only finite values")
            if p.mode_probabilities is not None and len(p.mode_probabilities) != sam.shape[1]:
                raise ValueError(
                    f"mode_probabilities length {len(p.mode_probabilities)} does not "
                    f"match samples n_modes {sam.shape[1]}"
                )

        actor_mask = [True] * n_actors
        safe_feature_schema = dict(p.feature_schema) if p.feature_schema else {"position": "x_y"}
        safe_actor_mask_metadata = (
            dict(p.actor_mask_metadata) if p.actor_mask_metadata else {"mask": "all_included"}
        )
        safe_actor_classes = dict(p.actor_classes) if p.actor_classes else {}

        provenance = ForecastBatchProvenance(
            predictor_id=config.predictor_id,
            predictor_family=config.predictor_family,
            observation_tier=config.observation_tier,
            frame=config.frame or CoordinateFrame(name="world"),
            dt_s=config.dt_s,
            horizons_s=list(config.horizons_s),
            scenario_id=config.scenario_id,
            seed=config.seed,
            timestamp=datetime.now(tz=UTC).isoformat(),
            fallback_status=config.fallback_status,
            degraded_status=config.degraded_status,
            actor_ids=list(actor_ids),
            actor_mask=actor_mask,
            actor_mask_metadata=safe_actor_mask_metadata,
            feature_schema=safe_feature_schema,
            oracle_state=config.oracle_state,
            actor_classes=safe_actor_classes,
        )

        forecasts: list[ActorForecast] = []
        for i, actor_id in enumerate(actor_ids):
            actor_samples = sam[i] if p.samples is not None else None
            forecast = ActorForecast(
                actor_id=actor_id,
                deterministic=det[i],
                samples=actor_samples,
                mode_probabilities=p.mode_probabilities,
                gaussian=p.gaussian_params,
                reachable_set=p.reachable_set,
                occupancy_summary=p.occupancy_summary,
                uncertainty_metadata=p.uncertainty_metadata,
            )
            forecasts.append(forecast)

        safe_metadata = dict(p.metadata) if p.metadata else {}
        return ForecastBatch(
            provenance=provenance,
            forecasts=forecasts,
            metadata=safe_metadata,
        )


def build_heavy_model_forecast_batch(
    config: HeavyModelAdapterConfig,
    actor_ids: list[str],
    deterministic_trajectories: np.ndarray,
    payload: HeavyModelPredictionPayload | None = None,
) -> ForecastBatch:
    """Convenience entry point: build a heavy-model ForecastBatch in one call.

    Delegates to ``HeavyModelForecastAdapter().build_batch(...)``.

    Args:
        config: Conversion configuration.
        actor_ids: Ordered actor identifiers.
        deterministic_trajectories:
            Shape ``(n_actors, n_horizons, 2)``.
        payload:
            Optional bundle of additional prediction outputs.

    Returns:
        Validated ``ForecastBatch``.
    """
    adapter = HeavyModelForecastAdapter()
    return adapter.build_batch(
        config=config,
        actor_ids=actor_ids,
        deterministic_trajectories=deterministic_trajectories,
        payload=payload,
    )
