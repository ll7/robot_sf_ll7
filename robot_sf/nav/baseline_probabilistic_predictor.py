"""Baseline ProbabilisticPredictor implementations for benchmark forecast variants.

This module provides small, deterministic predictors that consume the
SocNav-structured observation used by the rest of the planner stack and emit
:class:`ProbabilisticPrediction` artifacts.  They are intended as the smallest
honest unblocking layer for the live forecast replay gate: the forecast actually
flows into a simple replay policy and produces per-variant closed-loop metrics,
but the predictors themselves make no claim about human realism, calibration, or
planning benefit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.common.forecast_variants import FORECAST_VARIANT_CHOICES
from robot_sf.nav.predictive_types import (
    NeighborContext,
    PedestrianState,
    ProbabilisticPrediction,
    TrajectoryDistribution,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from robot_sf.benchmark.pedestrian_forecast import PedestrianForecast

    ForecastBaselineFunction = Callable[
        [PedestrianState, list[float] | tuple[float, ...]],
        PedestrianForecast,
    ]

DEFAULT_BASELINE_HORIZONS_S: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_BASELINE_DT_S: float = 0.1
DEFAULT_RISK_DISTANCE_M: float = 3.0


def _baseline_for_variant(variant: str) -> ForecastBaselineFunction:
    """Return the benchmark baseline function for a named variant.

    Args:
        variant: One of the supported baseline variant names.

    Returns:
        The baseline forecast function.  The caller is responsible for passing
        additional keyword arguments required by interaction-aware and risk-filtered
        variants (``neighbors`` and ``robot_position`` respectively).
    """
    from robot_sf.benchmark.pedestrian_forecast import (  # noqa: PLC0415
        constant_velocity_gaussian_baseline,
        interaction_aware_cv_baseline,
        risk_filtered_cv_baseline,
        semantic_cv_baseline,
    )

    if variant == "cv":
        return constant_velocity_gaussian_baseline
    if variant == "semantic":
        return semantic_cv_baseline
    if variant == "interaction_aware":
        return interaction_aware_cv_baseline
    if variant == "risk_filtered":
        return risk_filtered_cv_baseline
    raise ValueError(f"unsupported baseline variant: {variant}")


def _prediction_timestamp(observation: dict[str, Any]) -> float:
    """Return observation simulation time when available."""

    sim = observation.get("sim") or {}
    time_value = sim.get("time_s")
    if time_value is None:
        return -1.0
    time_array = np.asarray(time_value, dtype=float)
    if time_array.size == 0:
        return -1.0
    return float(time_array.reshape(-1)[0])


@dataclass
class BaselineProbabilisticPredictor:
    """Deterministic baseline predictor satisfying :class:`ProbabilisticPredictor`.

    The predictor converts the SocNav-structured observation produced by
    :class:`robot_sf.sensor.socnav_observation.SocNavObservationFusion` into
    the pedestrian-state representation used by the benchmark baseline functions,
    then emits per-pedestrian trajectory distributions.

    Attributes:
        variant: Forecast variant to use.  ``none`` returns static predictions.
        horizons_s: Forecast horizons in seconds.  Used to determine the maximum
            prediction horizon; trajectories are emitted at ``dt_s`` resolution.
        dt_s: Timestep between consecutive predicted positions.
        risk_distance_m: Distance threshold for the ``risk_filtered`` variant.
    """

    variant: str = "none"
    horizons_s: tuple[float, ...] = field(default_factory=lambda: DEFAULT_BASELINE_HORIZONS_S)
    dt_s: float = DEFAULT_BASELINE_DT_S
    risk_distance_m: float = DEFAULT_RISK_DISTANCE_M

    def __post_init__(self) -> None:
        """Validate the variant and pre-compute per-step horizons."""

        if self.variant not in FORECAST_VARIANT_CHOICES:
            raise ValueError(
                f"variant must be one of {list(FORECAST_VARIANT_CHOICES)}; got {self.variant!r}"
            )
        if not self.horizons_s:
            raise ValueError("horizons_s must contain at least one horizon")
        if float(self.dt_s) <= 0.0:
            raise ValueError("dt_s must be positive")
        self.horizons_s = tuple(float(h) for h in self.horizons_s)
        self.dt_s = float(self.dt_s)
        self.risk_distance_m = float(self.risk_distance_m)
        max_horizon_s = max(self.horizons_s)
        self._steps = math.ceil(max_horizon_s / self.dt_s)
        if self._steps < 1:
            raise ValueError("max(horizons_s) must be at least dt_s")
        self._step_horizons_s = tuple(
            (step_index + 1) * self.dt_s for step_index in range(self._steps)
        )

    def predict(self, observation: dict[str, Any]) -> ProbabilisticPrediction:
        """Return probabilistic future trajectories for observed pedestrians.

        Args:
            observation: SocNav-structured dict with ``robot``, ``goal``,
                ``pedestrians``, ``map``, and ``sim`` keys.

        Returns:
            ProbabilisticPrediction with one TrajectoryDistribution per observed
            pedestrian.  Velocities are converted from the ego-frame representation
            in the observation back to world frame before forecasting.
        """

        robot = observation.get("robot") or {}
        robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float)
        heading = float(np.asarray(robot.get("heading", [0.0])).reshape(-1)[0])

        pedestrians = observation.get("pedestrians") or {}
        positions = np.asarray(pedestrians.get("positions", np.zeros((0, 2), dtype=np.float32)))
        velocities_ego = np.asarray(
            pedestrians.get("velocities", np.zeros((0, 2), dtype=np.float32))
        )
        count = round(float(np.asarray(pedestrians.get("count", [0.0])).reshape(-1)[0]))
        count = max(0, min(count, positions.shape[0], velocities_ego.shape[0]))

        # SocNav observations store pedestrian velocities in ego frame; rotate back
        # to world frame so the constant-velocity baselines operate in the same frame
        # as the pedestrian positions.
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        velocities_world = np.zeros_like(velocities_ego)
        velocities_world[:, 0] = cos_h * velocities_ego[:, 0] - sin_h * velocities_ego[:, 1]
        velocities_world[:, 1] = sin_h * velocities_ego[:, 0] + cos_h * velocities_ego[:, 1]

        states: list[PedestrianState] = []
        for index in range(count):
            states.append(
                PedestrianState(
                    id=index,
                    position=np.asarray(positions[index], dtype=float),
                    velocity=np.asarray(velocities_world[index], dtype=float),
                )
            )

        timestamp = _prediction_timestamp(observation)
        if self.variant == "none" or not states:
            return ProbabilisticPrediction(
                predictions=[
                    TrajectoryDistribution(
                        mean=np.tile(
                            np.asarray(state.position, dtype=np.float32), (self._steps, 1)
                        ),
                        pedestrian_id=state.id,
                    )
                    for state in states
                ],
                prediction_horizon=self._steps * self.dt_s,
                prediction_dt=self.dt_s,
                timestamp=timestamp,
                sample_count=1,
                metadata={"variant": self.variant, "model": "baseline_none"},
            )

        neighbor_contexts = [
            NeighborContext(position=state.position, velocity=state.velocity) for state in states
        ]

        baseline_fn = _baseline_for_variant(self.variant)
        predictions: list[TrajectoryDistribution] = []
        for state in states:
            if self.variant == "interaction_aware":
                other_neighbors = [
                    context
                    for context_state, context in zip(states, neighbor_contexts, strict=True)
                    if context_state.id != state.id
                ]
                forecast = baseline_fn(
                    state, list(self._step_horizons_s), neighbors=other_neighbors
                )
            elif self.variant == "risk_filtered":
                forecast = baseline_fn(
                    state,
                    list(self._step_horizons_s),
                    robot_position=robot_pos,
                    risk_distance_m=self.risk_distance_m,
                )
            else:
                forecast = baseline_fn(state, list(self._step_horizons_s))

            step_means = np.asarray(
                [prediction.mean for prediction in forecast.predictions], dtype=np.float32
            )
            std_by_step = []
            for prediction in forecast.predictions:
                metadata = prediction.metadata
                if not isinstance(metadata, dict):
                    metadata = {}
                std_by_step.append(float(metadata.get("std_m", 0.3)))
            std_m = np.asarray(std_by_step, dtype=np.float32)
            std = np.repeat(std_m[:, np.newaxis], 2, axis=1)
            covariance = np.asarray(
                [np.eye(2, dtype=np.float32) * float(step_std**2) for step_std in std_m],
                dtype=np.float32,
            )
            predictions.append(
                TrajectoryDistribution(
                    mean=step_means,
                    std=std,
                    covariance=covariance,
                    confidence=1.0,
                    pedestrian_id=state.id,
                )
            )

        return ProbabilisticPrediction(
            predictions=predictions,
            prediction_horizon=self._steps * self.dt_s,
            prediction_dt=self.dt_s,
            timestamp=timestamp,
            sample_count=1,
            metadata={"variant": self.variant, "model": f"baseline_{self.variant}"},
        )


__all__ = [
    "DEFAULT_BASELINE_DT_S",
    "DEFAULT_BASELINE_HORIZONS_S",
    "DEFAULT_RISK_DISTANCE_M",
    "BaselineProbabilisticPredictor",
]
