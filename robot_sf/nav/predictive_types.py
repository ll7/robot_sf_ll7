"""Probabilistic pedestrian prediction types and protocol.

This module defines the minimal interface contract for probabilistic pedestrian
trajectory prediction. Planners can consume these types to obtain future
trajectory distributions and per-pedestrian confidence without committing to any
specific predictor implementation, training regime, or prediction quality claim.

The interface is intentionally additive: existing deterministic predictors can
emit confidence=1.0 and identity covariance to signal "no uncertainty estimate."
Any claim about prediction accuracy or planning benefit from using these types
requires separate benchmark evidence per the project's maintainer values.

.. admonition:: Claim boundary
   :class: note

   Defining or implementing this interface does **not** constitute evidence of
   prediction quality, calibration, or planning improvement. Benchmark runs
   are required before any such claim may be made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _require_float_array(
    name: str,
    value: NDArray[np.float32],
    *,
    ndim: int,
) -> NDArray[np.float32]:
    """Validate and normalize a numeric prediction array for the public contract.

    Returns:
        Float32 array with the same shape as the input.
    """
    array = np.asarray(value)
    if array.ndim != ndim or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (T, 2)" if ndim == 2 else f"{name} invalid")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float32, copy=False)


def _require_covariance_array(value: NDArray[np.float32], *, steps: int) -> NDArray[np.float32]:
    """Validate and normalize full per-timestep covariance matrices.

    Returns:
        Float32 array with shape ``(T, 2, 2)``.
    """
    covariance = np.asarray(value)
    expected_shape = (steps, 2, 2)
    if covariance.shape != expected_shape:
        raise ValueError("covariance must have shape (T, 2, 2)")
    if not np.issubdtype(covariance.dtype, np.floating):
        raise ValueError("covariance must use a floating dtype")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2)):
        raise ValueError("covariance matrices must be symmetric")
    if np.any(np.linalg.eigvalsh(covariance) < -1e-6):
        raise ValueError("covariance matrices must be positive semidefinite")
    return covariance.astype(np.float32, copy=False)


@dataclass
class TrajectoryDistribution:
    """Probabilistic future trajectory for a single pedestrian.

    Attributes:
        mean: Mean future positions in robot frame, shape ``(T, 2)`` where
            ``T`` is the number of predicted timesteps and columns are
            ``(x, y)`` in world or robot-frame coordinates.
        std: Per-timestep per-axis standard deviation, shape ``(T, 2)``.
            ``None`` when the predictor only emits means (deterministic mode).
        covariance: Full per-timestep covariance matrices, shape ``(T, 2, 2)``.
            May be ``None`` when only diagonal uncertainty is available.
        confidence: Scalar confidence in ``[0, 1]`` reflecting the predictor's
            own assessment of this trajectory's reliability. A deterministic
            predictor may emit ``1.0`` (no uncertainty expressed).
        pedestrian_id: Index or identifier for this pedestrian within the
            observation's pedestrian array.
    """

    mean: NDArray[np.float32]
    std: NDArray[np.float32] | None = None
    covariance: NDArray[np.float32] | None = None
    confidence: float = 1.0
    pedestrian_id: int = -1

    def __post_init__(self) -> None:
        """Validate shape and confidence fields for one pedestrian trajectory."""
        self.mean = _require_float_array("mean", self.mean, ndim=2)
        if self.std is not None:
            self.std = _require_float_array("std", self.std, ndim=2)
            if self.std.shape != self.mean.shape:
                raise ValueError("std must match mean shape")
            if np.any(self.std < 0.0):
                raise ValueError("std must be non-negative")
        if self.covariance is not None:
            self.covariance = _require_covariance_array(
                self.covariance,
                steps=self.mean.shape[0],
            )
        self.confidence = float(self.confidence)
        self.pedestrian_id = int(self.pedestrian_id)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class ProbabilisticPrediction:
    """Container for multi-agent probabilistic pedestrian predictions.

    This is the top-level return type of :class:`ProbabilisticPredictor`.
    It bundles per-pedestrian trajectory distributions together with shared
    metadata so consumers do not need to track prediction horizon or
    timestamps separately.

    Attributes:
        predictions: One :class:`TrajectoryDistribution` per pedestrian.
        prediction_horizon: Forecast horizon in seconds.
        prediction_dt: Timestep between consecutive predicted positions
            in seconds.
        timestamp: Simulation timestamp (seconds) at which this prediction
            was produced. May be ``-1`` when the caller does not provide it.
        sample_count: Number of Monte-Carlo or scenario samples used to
            derive the uncertainty estimates. ``1`` for deterministic.
        metadata: Free-form key-value store for predictor-specific data
            (e.g. model version, feature schema, fallback mode).
    """

    predictions: list[TrajectoryDistribution] = field(default_factory=list)
    prediction_horizon: float = 0.0
    prediction_dt: float = 0.1
    timestamp: float = -1.0
    sample_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shared prediction metadata fields."""
        self.prediction_horizon = float(self.prediction_horizon)
        self.prediction_dt = float(self.prediction_dt)
        self.timestamp = float(self.timestamp)
        self.sample_count = int(self.sample_count)
        if self.prediction_horizon < 0.0:
            raise ValueError("prediction_horizon must be non-negative")
        if self.prediction_dt <= 0.0:
            raise ValueError("prediction_dt must be positive")
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        if self.predictions:
            expected_steps = self.prediction_horizon / self.prediction_dt
            for prediction in self.predictions:
                if not np.isclose(prediction.mean.shape[0], expected_steps):
                    raise ValueError(
                        "prediction_horizon must equal trajectory steps multiplied by prediction_dt"
                    )


@runtime_checkable
class ProbabilisticPredictor(Protocol):
    """Protocol for probabilistic pedestrian trajectory predictors.

    Any object that implements ``predict(observation) -> ProbabilisticPrediction``
    satisfies this protocol. The observation dict follows the SocNav-structured
    schema produced by :class:`robot_sf.sensor.socnav_observation.SocNavObservationFusion`.

    Implementing this protocol does **not** commit the predictor to any accuracy,
    calibration, or planning-benefit claim. See module-level docstring.

    Example::

        class MyPredictor:
            def predict(self, observation: dict[str, Any]) -> ProbabilisticPrediction: ...
    """

    def predict(self, observation: dict[str, Any]) -> ProbabilisticPrediction:
        """Return probabilistic future trajectories for all observed pedestrians.

        Args:
            observation: SocNav-structured dict with keys ``"robot"``,
                ``"goal"``, ``"pedestrians"``, ``"map"``, ``"sim"``.

        Returns:
            ProbabilisticPrediction: Per-pedestrian trajectory distributions
            with associated uncertainty and confidence.
        """


@dataclass(frozen=True)
class PedestrianState:
    """One pedestrian state at a single timestep.

    Canonical pedestrian state used by nav predictors and benchmark baselines.
    Intent and signal fields are optional semantic context; when absent the
    state reduces to position + velocity.
    """

    id: int
    position: np.ndarray
    velocity: np.ndarray
    intent: str | None = None
    signal: str | None = None
    signal_available: bool = False
    actor_type: str = "pedestrian"

    @classmethod
    def from_trace(cls, payload: dict[str, Any]) -> PedestrianState:
        """Build a state from ``simulation_step_trace.steps[].pedestrians[]``.

        Returns:
            Trace-compatible pedestrian state.
        """

        signal_state = payload.get("signal_state")
        signal_available = False
        signal: str | None = None
        if isinstance(signal_state, dict):
            signal_available = bool(
                signal_state.get("available")
                if "available" in signal_state
                else signal_state.get("label") is not None
            )
            if signal_available and signal_state.get("label") is not None:
                signal = str(signal_state["label"])
        elif payload.get("signal_label") is not None:
            signal_available = True
            signal = str(payload["signal_label"])

        return cls(
            id=int(payload["id"]),
            position=np.asarray(payload["position"], dtype=float),
            velocity=np.asarray(payload["velocity"], dtype=float),
            intent=str(payload["intent_label"])
            if payload.get("intent_label") is not None
            else None,
            signal=signal,
            signal_available=signal_available,
            actor_type=str(payload.get("actor_type") or "pedestrian"),
        )


@dataclass(frozen=True)
class NeighborContext:
    """Snapshot of a neighboring pedestrian's state for interaction-aware forecasts."""

    position: np.ndarray
    velocity: np.ndarray
    actor_type: str = "pedestrian"


__all__ = [
    "NeighborContext",
    "PedestrianState",
    "ProbabilisticPrediction",
    "ProbabilisticPredictor",
    "TrajectoryDistribution",
]
