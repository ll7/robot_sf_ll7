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

from collections.abc import Mapping, Sequence
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


def _validate_modes_sequence(
    modes: Sequence[TrajectoryMode],
    pedestrian_id: int,
) -> list[TrajectoryMode]:
    """Validate uniqueness, horizon consistency, and probability sum of modes.

    Returns:
        Validated list of TrajectoryMode instances.
    """
    if not modes:
        raise ValueError(f"pedestrian {pedestrian_id} must have at least one mode")

    mode_ids: set[str] = set()
    total_prob = 0.0
    expected_steps: int | None = None
    normalized_modes: list[TrajectoryMode] = []

    for mode in modes:
        if not isinstance(mode, TrajectoryMode):
            raise TypeError(f"expected TrajectoryMode, got {type(mode).__name__}")
        if mode.mode_id in mode_ids:
            raise ValueError(f"duplicate mode_id {mode.mode_id!r} for pedestrian {pedestrian_id}")
        mode_ids.add(mode.mode_id)
        total_prob += mode.probability
        steps = mode.mean.shape[0]
        if expected_steps is None:
            expected_steps = steps
        elif steps != expected_steps:
            raise ValueError(
                f"mode {mode.mode_id!r} step count {steps} does not match other modes ({expected_steps})"
            )
        normalized_modes.append(mode)

    if not np.isclose(total_prob, 1.0, atol=1e-3):
        raise ValueError(
            f"mode probabilities must sum to 1.0 (within tolerance), got sum={total_prob}"
        )
    return normalized_modes


def _normalize_forecast_mapping(
    forecasts: Mapping[int, PedestrianForecast] | Sequence[PedestrianForecast],
) -> dict[int, PedestrianForecast]:
    """Normalize input mapping or sequence into a pedestrian-keyed dictionary.

    Returns:
        Dictionary mapping integer pedestrian IDs to PedestrianForecast objects.
    """
    normalized: dict[int, PedestrianForecast] = {}
    if isinstance(forecasts, Mapping):
        for key, forecast in forecasts.items():
            if not isinstance(forecast, PedestrianForecast):
                raise TypeError(f"expected PedestrianForecast, got {type(forecast).__name__}")
            ped_id = int(key)
            if ped_id != forecast.pedestrian_id:
                raise ValueError(
                    f"key mismatch: dict key {ped_id} != forecast.pedestrian_id {forecast.pedestrian_id}"
                )
            normalized[ped_id] = forecast
    elif isinstance(forecasts, Sequence):
        for forecast in forecasts:
            if not isinstance(forecast, PedestrianForecast):
                raise TypeError(f"expected PedestrianForecast, got {type(forecast).__name__}")
            if forecast.pedestrian_id in normalized:
                raise ValueError(f"duplicate pedestrian_id {forecast.pedestrian_id}")
            normalized[forecast.pedestrian_id] = forecast
    else:
        raise TypeError("forecasts must be a dict or sequence of PedestrianForecast")
    return normalized


def _validate_prediction_steps(
    forecasts: dict[int, PedestrianForecast],
    prediction_horizon: float,
    prediction_dt: float,
) -> None:
    """Validate that forecast trajectory step counts match prediction_horizon and prediction_dt."""
    if not forecasts or prediction_horizon <= 0.0:
        return
    expected_steps = prediction_horizon / prediction_dt
    for ped_id, forecast in forecasts.items():
        for mode in forecast.modes:
            if not np.isclose(mode.mean.shape[0], expected_steps):
                raise ValueError(
                    f"pedestrian {ped_id} mode {mode.mode_id} steps ({mode.mean.shape[0]}) "
                    f"does not match expected steps from horizon/dt ({expected_steps})"
                )


@dataclass
class TrajectoryDistribution:
    """Probabilistic future trajectory for a single pedestrian.

    Attributes:
        mean: Mean future positions in robot/world frame, shape ``(T, 2)`` where
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
        self.mean = np.array(_require_float_array("mean", self.mean, ndim=2), copy=True)
        if self.std is not None:
            self.std = np.array(_require_float_array("std", self.std, ndim=2), copy=True)
            if self.std.shape != self.mean.shape:
                raise ValueError("std must match mean shape")
            if np.any(self.std < 0.0):
                raise ValueError("std must be non-negative")
        if self.covariance is not None:
            self.covariance = np.array(
                _require_covariance_array(
                    self.covariance,
                    steps=self.mean.shape[0],
                ),
                copy=True,
            )
        self.confidence = float(self.confidence)
        self.pedestrian_id = int(self.pedestrian_id)
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
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
        self.metadata = dict(self.metadata) if self.metadata else {}

        if not np.isfinite(self.prediction_horizon) or self.prediction_horizon < 0.0:
            raise ValueError("prediction_horizon must be non-negative and finite")
        if not np.isfinite(self.prediction_dt) or self.prediction_dt <= 0.0:
            raise ValueError("prediction_dt must be positive and finite")
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        if self.predictions:
            expected_steps = self.prediction_horizon / self.prediction_dt
            for prediction in self.predictions:
                if not np.isclose(prediction.mean.shape[0], expected_steps):
                    raise ValueError(
                        "prediction_horizon must equal trajectory steps multiplied by prediction_dt"
                    )


@dataclass
class TrajectoryMode:
    """One predicted future trajectory mode for a single pedestrian.

    Attributes:
        mode_id: Unique string identifier for this mode within the pedestrian's
            forecast (e.g. "primary", "turn_left", "cross_street", "mode_0").
        probability: Probability weight in [0, 1] for this mode.
        mean: Mean future positions in robot/world frame, shape ``(T, 2)``.
        std: Per-timestep per-axis standard deviation, shape ``(T, 2)``.
        covariance: Full per-timestep covariance matrices, shape ``(T, 2, 2)``.
        intent: Optional semantic intent label (e.g. "crossing", "waiting").
        metadata: Free-form key-value store for mode-specific information.
    """

    mode_id: str
    probability: float
    mean: NDArray[np.float32]
    std: NDArray[np.float32] | None = None
    covariance: NDArray[np.float32] | None = None
    intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and defensively normalize mode fields."""
        if not isinstance(self.mode_id, str):
            raise ValueError("mode_id must be a non-empty string")
        self.mode_id = self.mode_id.strip()
        if not self.mode_id:
            raise ValueError("mode_id must be a non-empty string")

        prob = float(self.probability)
        if not np.isfinite(prob):
            raise ValueError("mode probability must be finite")
        if not 0.0 <= prob <= 1.0:
            raise ValueError("mode probability must be in [0, 1]")
        self.probability = prob

        self.mean = np.array(_require_float_array("mean", self.mean, ndim=2), copy=True)
        if self.std is not None:
            self.std = np.array(_require_float_array("std", self.std, ndim=2), copy=True)
            if self.std.shape != self.mean.shape:
                raise ValueError("std must match mean shape")
            if np.any(self.std < 0.0):
                raise ValueError("std must be non-negative")
        if self.covariance is not None:
            self.covariance = np.array(
                _require_covariance_array(self.covariance, steps=self.mean.shape[0]),
                copy=True,
            )
        if self.intent is not None:
            self.intent = str(self.intent)
        self.metadata = dict(self.metadata) if self.metadata else {}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Dictionary containing mode parameters and arrays as lists.
        """
        payload: dict[str, Any] = {
            "mode_id": self.mode_id,
            "probability": float(self.probability),
            "mean": self.mean.tolist(),
        }
        if self.std is not None:
            payload["std"] = self.std.tolist()
        if self.covariance is not None:
            payload["covariance"] = self.covariance.tolist()
        if self.intent is not None:
            payload["intent"] = self.intent
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrajectoryMode:
        """Construct a TrajectoryMode from a dictionary.

        Returns:
            Instantiated TrajectoryMode instance.
        """
        return cls(
            mode_id=payload["mode_id"],
            probability=float(payload["probability"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32)
            if "std" in payload and payload["std"] is not None
            else None,
            covariance=np.asarray(payload["covariance"], dtype=np.float32)
            if "covariance" in payload and payload["covariance"] is not None
            else None,
            intent=payload.get("intent"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class PedestrianForecast:
    """Multimodal future trajectory forecast for a single pedestrian.

    Attributes:
        pedestrian_id: Track ID or index of this pedestrian.
        modes: List of distinct trajectory modes representing alternate hypotheses.
        existence_probability: Probability in [0, 1] that this pedestrian exists.
        confidence: Overall forecast confidence in [0, 1].
        age: Observation age / duration in seconds.
        metadata: Free-form key-value store for per-pedestrian forecast data.
    """

    pedestrian_id: int
    modes: list[TrajectoryMode] = field(default_factory=list)
    existence_probability: float = 1.0
    confidence: float = 1.0
    age: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and defensively normalize pedestrian forecast fields."""
        self.pedestrian_id = int(self.pedestrian_id)
        exist_prob = float(self.existence_probability)
        if not np.isfinite(exist_prob) or not 0.0 <= exist_prob <= 1.0:
            raise ValueError("existence_probability must be in [0, 1]")
        self.existence_probability = exist_prob

        conf = float(self.confidence)
        if not np.isfinite(conf) or not 0.0 <= conf <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.confidence = conf

        age = float(self.age)
        if not np.isfinite(age) or age < 0.0:
            raise ValueError("age must be non-negative and finite")
        self.age = age

        self.metadata = dict(self.metadata) if self.metadata else {}
        self.modes = _validate_modes_sequence(self.modes, self.pedestrian_id)

    def primary_mode(self) -> TrajectoryMode:
        """Return the highest-probability mode, with mode_id tie-breaking.

        Returns:
            Highest probability TrajectoryMode instance.
        """
        return max(self.modes, key=lambda m: (m.probability, m.mode_id))

    def sorted_modes(self) -> list[TrajectoryMode]:
        """Return modes sorted canonically by descending probability, then mode_id.

        Returns:
            List of modes ordered by descending probability.
        """
        return sorted(self.modes, key=lambda m: (-m.probability, m.mode_id))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Dictionary containing pedestrian forecast payload.
        """
        payload: dict[str, Any] = {
            "pedestrian_id": int(self.pedestrian_id),
            "existence_probability": float(self.existence_probability),
            "confidence": float(self.confidence),
            "age": float(self.age),
            "modes": [m.to_dict() for m in self.sorted_modes()],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PedestrianForecast:
        """Construct a PedestrianForecast from a dictionary.

        Returns:
            Instantiated PedestrianForecast instance.
        """
        modes = [TrajectoryMode.from_dict(m) for m in payload.get("modes", [])]
        return cls(
            pedestrian_id=int(payload["pedestrian_id"]),
            modes=modes,
            existence_probability=float(payload.get("existence_probability", 1.0)),
            confidence=float(payload.get("confidence", 1.0)),
            age=float(payload.get("age", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class MultimodalPrediction:
    """Multi-agent multimodal future trajectory prediction container.

    Attributes:
        forecasts: Mapping from pedestrian_id to PedestrianForecast.
        prediction_horizon: Forecast horizon in seconds.
        prediction_dt: Timestep between consecutive predicted positions in seconds.
        timestamp: Simulation timestamp (seconds) at which this prediction was produced.
        sample_count: Number of samples used to derive uncertainty.
        schema_version: Version identifier for contract reproducibility.
        metadata: Free-form key-value store for predictor-specific metadata.
    """

    forecasts: dict[int, PedestrianForecast] = field(default_factory=dict)
    prediction_horizon: float = 0.0
    prediction_dt: float = 0.1
    timestamp: float = -1.0
    sample_count: int = 1
    schema_version: str = "multimodal-prediction.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and defensively normalize multimodal prediction fields."""
        self.prediction_horizon = float(self.prediction_horizon)
        self.prediction_dt = float(self.prediction_dt)
        self.timestamp = float(self.timestamp)
        self.sample_count = int(self.sample_count)
        self.schema_version = str(self.schema_version)
        self.metadata = dict(self.metadata) if self.metadata else {}

        if not np.isfinite(self.prediction_horizon) or self.prediction_horizon < 0.0:
            raise ValueError("prediction_horizon must be non-negative and finite")
        if not np.isfinite(self.prediction_dt) or self.prediction_dt <= 0.0:
            raise ValueError("prediction_dt must be positive and finite")
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least 1")

        normalized = _normalize_forecast_mapping(self.forecasts)
        _validate_prediction_steps(normalized, self.prediction_horizon, self.prediction_dt)
        self.forecasts = normalized

    def ordered_pedestrian_ids(self) -> list[int]:
        """Return pedestrian IDs sorted in canonical ascending order.

        Returns:
            Sorted list of integer pedestrian IDs.
        """
        return sorted(self.forecasts.keys())

    def ordered_forecasts(self) -> list[PedestrianForecast]:
        """Return pedestrian forecasts sorted canonically by pedestrian ID.

        Returns:
            List of PedestrianForecast instances sorted by pedestrian ID.
        """
        return [self.forecasts[pid] for pid in self.ordered_pedestrian_ids()]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Dictionary containing multimodal prediction payload.
        """
        return {
            "schema_version": self.schema_version,
            "prediction_horizon": float(self.prediction_horizon),
            "prediction_dt": float(self.prediction_dt),
            "timestamp": float(self.timestamp),
            "sample_count": int(self.sample_count),
            "metadata": dict(self.metadata),
            "forecasts": {
                str(pid): self.forecasts[pid].to_dict() for pid in self.ordered_pedestrian_ids()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultimodalPrediction:
        """Construct a MultimodalPrediction from a dictionary.

        Returns:
            Instantiated MultimodalPrediction instance.
        """
        forecasts_payload = payload.get("forecasts", {})
        forecasts: dict[int, PedestrianForecast] = {}
        for key_str, f_dict in forecasts_payload.items():
            f = PedestrianForecast.from_dict(f_dict)
            forecasts[int(key_str)] = f

        return cls(
            forecasts=forecasts,
            prediction_horizon=float(payload.get("prediction_horizon", 0.0)),
            prediction_dt=float(payload.get("prediction_dt", 0.1)),
            timestamp=float(payload.get("timestamp", -1.0)),
            sample_count=int(payload.get("sample_count", 1)),
            schema_version=str(payload.get("schema_version", "multimodal-prediction.v1")),
            metadata=dict(payload.get("metadata", {})),
        )

    def as_probabilistic_prediction(self) -> ProbabilisticPrediction:
        """Convert multimodal prediction to legacy unimodal ProbabilisticPrediction.

        Returns:
            Unimodal ProbabilisticPrediction containing each pedestrian's primary mode.
        """
        distributions: list[TrajectoryDistribution] = []
        for pid in self.ordered_pedestrian_ids():
            forecast = self.forecasts[pid]
            primary = forecast.primary_mode()
            distributions.append(
                TrajectoryDistribution(
                    mean=primary.mean.copy(),
                    std=primary.std.copy() if primary.std is not None else None,
                    covariance=primary.covariance.copy()
                    if primary.covariance is not None
                    else None,
                    confidence=float(forecast.confidence * primary.probability),
                    pedestrian_id=pid,
                )
            )
        return ProbabilisticPrediction(
            predictions=distributions,
            prediction_horizon=self.prediction_horizon,
            prediction_dt=self.prediction_dt,
            timestamp=self.timestamp,
            sample_count=self.sample_count,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_probabilistic_prediction(
        cls,
        prediction: ProbabilisticPrediction,
        *,
        mode_id: str = "primary",
    ) -> MultimodalPrediction:
        """Convert legacy unimodal ProbabilisticPrediction into MultimodalPrediction.

        Returns:
            MultimodalPrediction containing single-mode forecasts.
        """
        forecasts: dict[int, PedestrianForecast] = {}
        for dist in prediction.predictions:
            mode = TrajectoryMode(
                mode_id=mode_id,
                probability=1.0,
                mean=dist.mean.copy(),
                std=dist.std.copy() if dist.std is not None else None,
                covariance=dist.covariance.copy() if dist.covariance is not None else None,
            )
            forecast = PedestrianForecast(
                pedestrian_id=dist.pedestrian_id,
                modes=[mode],
                existence_probability=1.0,
                confidence=dist.confidence,
                age=0.0,
            )
            forecasts[dist.pedestrian_id] = forecast

        return cls(
            forecasts=forecasts,
            prediction_horizon=prediction.prediction_horizon,
            prediction_dt=prediction.prediction_dt,
            timestamp=prediction.timestamp,
            sample_count=prediction.sample_count,
            metadata=dict(prediction.metadata),
        )


def as_multimodal_prediction(
    prediction: ProbabilisticPrediction | MultimodalPrediction,
) -> MultimodalPrediction:
    """Normalize a prediction into canonical MultimodalPrediction format.

    Args:
        prediction: Either a ProbabilisticPrediction or MultimodalPrediction instance.

    Returns:
        MultimodalPrediction: Canonical multimodal representation.

    Raises:
        TypeError: If prediction is neither ProbabilisticPrediction nor MultimodalPrediction.
    """
    if isinstance(prediction, MultimodalPrediction):
        return prediction
    if isinstance(prediction, ProbabilisticPrediction):
        return MultimodalPrediction.from_probabilistic_prediction(prediction)
    raise TypeError(
        f"expected ProbabilisticPrediction or MultimodalPrediction, got {type(prediction).__name__}"
    )


def build_normalized_modes(
    raw_modes: Sequence[dict[str, Any] | TrajectoryMode],
) -> list[TrajectoryMode]:
    """Build a list of TrajectoryMode instances with normalized probabilities.

    Given a collection of modes with non-negative raw weights, computes normalized
    probabilities summing to 1.0.

    Args:
        raw_modes: Sequence of TrajectoryMode instances or dict specs with weights/probabilities.

    Returns:
        List of TrajectoryMode with probabilities summing exactly to 1.0.

    Raises:
        ValueError: If weights are non-finite, negative, or sum to <= 0.
    """
    if not raw_modes:
        raise ValueError("raw_modes must not be empty")

    weights: list[float] = []
    items: list[
        tuple[
            str,
            NDArray[np.float32],
            NDArray[np.float32] | None,
            NDArray[np.float32] | None,
            str | None,
            dict[str, Any],
        ]
    ] = []

    for item in raw_modes:
        if isinstance(item, TrajectoryMode):
            w = float(item.probability)
            mode_id = item.mode_id
            mean = item.mean
            std = item.std
            cov = item.covariance
            intent = item.intent
            meta = item.metadata
        elif isinstance(item, dict):
            w = float(item.get("probability", item.get("weight", 0.0)))
            if not np.isfinite(w) or w < 0.0:
                raise ValueError(f"mode weight must be finite and non-negative, got {w}")
            if "mode_id" not in item:
                raise ValueError("mode dict must contain 'mode_id'")
            if "mean" not in item:
                raise ValueError("mode dict must contain 'mean'")
            mode_id = str(item["mode_id"])
            mean = np.asarray(item["mean"], dtype=np.float32)
            std = np.asarray(item["std"], dtype=np.float32) if item.get("std") is not None else None
            cov = (
                np.asarray(item["covariance"], dtype=np.float32)
                if item.get("covariance") is not None
                else None
            )
            intent = item.get("intent")
            meta = dict(item.get("metadata", {}))
        else:
            raise TypeError(f"expected dict or TrajectoryMode, got {type(item).__name__}")
        weights.append(w)
        items.append((mode_id, mean, std, cov, intent, meta))

    total_weight = sum(weights)
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        raise ValueError(f"sum of mode weights must be positive and finite, got {total_weight}")

    normalized: list[TrajectoryMode] = []
    for i, (mode_id, mean, std, cov, intent, meta) in enumerate(items):
        prob = weights[i] / total_weight
        normalized.append(
            TrajectoryMode(
                mode_id=mode_id,
                probability=prob,
                mean=mean,
                std=std,
                covariance=cov,
                intent=intent,
                metadata=meta,
            )
        )
    return normalized


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
    "MultimodalPrediction",
    "NeighborContext",
    "PedestrianForecast",
    "PedestrianState",
    "ProbabilisticPrediction",
    "ProbabilisticPredictor",
    "TrajectoryDistribution",
    "TrajectoryMode",
    "as_multimodal_prediction",
    "build_normalized_modes",
]
