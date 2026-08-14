"""Fail-closed Step 3 model-quality gate for issue #6318.

This module is the smallest compute-light continuation of the clean-room dynamics module in
``open_dreamer_dynamics``.  It consumes the existing ``RLTrajectoryDataset.v1`` contract through
the merged structured-observation adapter, keeps episode boundaries intact, fits the small
action-conditioned latent model on a training split, and evaluates one-step and short multi-step
prediction on a held-out split.

The gate compares the fitted model with two deliberately simple references:

* a persistence predictor that repeats the current structured observation; and
* a deterministic random-feature multilayer perceptron (MLP) with a ridge-fitted output head.

The MLP is a diagnostic reference, not a claim about the best available predictor.  Every target
dimension remains visible in the report; no aggregate score can hide a regression in reward,
continuation, or observation prediction.  The gate returns ``blocked_insufficient_data`` when the
dataset lacks explicit training and holdout episodes or the minimum transition counts.  A report
from that state is a readiness result only and must never be treated as research evidence.

No SAC, replay-buffer integration, benchmark admission, policy promotion, or paper-facing claim is
performed here.  The later matched SAC arms remain the separately gated Step 4 of #6318.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from robot_sf.benchmark.rl_trajectory_dataset import (
    load_rl_trajectory_dataset,
)
from robot_sf.research.open_dreamer_adapter import (
    ActionBounds,
    StructuredEpisode,
    adapt_episodes,
)
from robot_sf.research.open_dreamer_dynamics import (
    DynamicsConfig,
    DynamicsWeights,
    LatentDynamicsModel,
    OpenDreamerDynamicsError,
)

OPEN_DREAMER_MODEL_QUALITY_VERSION = "open_dreamer_model_quality.v1"
EVIDENCE_BOUNDARY = "diagnostic_only"

QualityStatus = Literal[
    "passed",
    "failed_model_quality",
    "blocked_insufficient_data",
    "blocked_contract",
]
BaselineName = Literal["persistence", "mlp"]


class OpenDreamerQualityError(ValueError):
    """Raised when the quality gate configuration cannot be evaluated safely."""


class _InsufficientQualityData(OpenDreamerQualityError):
    """Internal marker for a valid dataset that cannot support the configured quality claims."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenDreamerQualityError(f"{name} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenDreamerQualityError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise OpenDreamerQualityError(f"{name} must be finite, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OpenDreamerQualityError(f"{name} must be finite, got {value!r}") from exc
    if not np.isfinite(result):
        raise OpenDreamerQualityError(f"{name} must be finite, got {value!r}")
    return result


def _config_int(value: Any, name: str) -> int:
    """Coerce one config integer without allowing booleans or fractional truncation.

    Returns:
        The parsed integer.
    """
    if isinstance(value, bool):
        raise OpenDreamerQualityError(f"{name} must be an integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OpenDreamerQualityError(f"{name} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise OpenDreamerQualityError(f"{name} must be an integer, got {value!r}")
    return result


def _config_string(value: Any, name: str) -> str:
    """Return one non-empty string config value without stringifying malformed YAML."""
    if not isinstance(value, str) or not value:
        raise OpenDreamerQualityError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _validate_quality_paths(config: ModelQualityConfig) -> None:
    if not isinstance(config.dataset_path, Path):
        raise OpenDreamerQualityError("dataset_path must be a pathlib.Path")
    if not config.dataset_path.name:
        raise OpenDreamerQualityError("dataset_path must name a dataset file")


def _validate_quality_splits(config: ModelQualityConfig) -> None:
    for name, value in (
        ("train_split", config.train_split),
        ("holdout_split", config.holdout_split),
    ):
        if not isinstance(value, str) or not value:
            raise OpenDreamerQualityError(f"{name} must be a non-empty string")
    if config.train_split == config.holdout_split:
        raise OpenDreamerQualityError("train_split and holdout_split must differ")


def _validate_quality_actions(config: ModelQualityConfig) -> None:
    max_linear = _finite_float(config.max_linear_speed, "max_linear_speed")
    max_angular = _finite_float(config.max_angular_speed, "max_angular_speed")
    min_linear = _finite_float(config.min_linear_speed, "min_linear_speed")
    if max_linear <= 0.0 or max_angular <= 0.0:
        raise OpenDreamerQualityError("action speed maxima must be strictly positive")
    if not -max_linear <= min_linear < max_linear:
        raise OpenDreamerQualityError("min_linear_speed must lie below max_linear_speed")


def _validate_quality_dimensions(config: ModelQualityConfig) -> None:
    _positive_int(config.latent_dim, "latent_dim")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise OpenDreamerQualityError("seed must be a non-negative integer")
    for name in (
        "min_train_episodes",
        "min_holdout_episodes",
        "min_train_transitions",
        "min_holdout_transitions",
        "multi_step_horizon",
        "mlp_hidden_dim",
    ):
        _positive_int(getattr(config, name), name)
    alpha = _finite_float(config.ridge_alpha, "ridge_alpha")
    if alpha < 0.0:
        raise OpenDreamerQualityError("ridge_alpha must be non-negative")


def _validate_quality_baselines(config: ModelQualityConfig) -> None:
    if not config.required_baselines:
        raise OpenDreamerQualityError("required_baselines must not be empty")
    invalid = set(config.required_baselines).difference({"persistence", "mlp"})
    if invalid:
        raise OpenDreamerQualityError(f"unsupported required baselines: {sorted(invalid)}")


@dataclass(frozen=True, slots=True)
class ModelQualityConfig:
    """Config-first contract for the issue-6318 Step 3 quality gate."""

    dataset_path: Path
    train_split: str = "train"
    holdout_split: str = "test"
    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.0
    min_linear_speed: float = 0.0
    latent_dim: int = 5
    seed: int = 6318
    min_train_episodes: int = 2
    min_holdout_episodes: int = 2
    min_train_transitions: int = 8
    min_holdout_transitions: int = 4
    multi_step_horizon: int = 3
    mlp_hidden_dim: int = 16
    ridge_alpha: float = 1.0e-4
    required_baselines: tuple[BaselineName, ...] = ("persistence", "mlp")

    def __post_init__(self) -> None:
        """Validate the complete quality-gate configuration."""
        _validate_quality_paths(self)
        _validate_quality_splits(self)
        _validate_quality_actions(self)
        _validate_quality_dimensions(self)
        _validate_quality_baselines(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_dir: Path) -> ModelQualityConfig:
        """Build a validated config from a YAML/JSON mapping.

        Returns:
            A validated model-quality configuration.
        """
        if not isinstance(payload, Mapping):
            raise OpenDreamerQualityError("quality config must be a mapping")
        schema_version = payload.get("schema_version")
        if schema_version != OPEN_DREAMER_MODEL_QUALITY_VERSION:
            raise OpenDreamerQualityError(
                f"expected schema_version {OPEN_DREAMER_MODEL_QUALITY_VERSION!r}, "
                f"got {schema_version!r}"
            )
        raw_path = payload.get("dataset_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise OpenDreamerQualityError("dataset_path must be a non-empty string")
        dataset_path = Path(raw_path)
        if not dataset_path.is_absolute():
            dataset_path = (base_dir / dataset_path).resolve()
        actions = payload.get("action_bounds", {})
        if not isinstance(actions, Mapping):
            raise OpenDreamerQualityError("action_bounds must be a mapping")
        baselines = payload.get("required_baselines", ["persistence", "mlp"])
        if isinstance(baselines, str) or not isinstance(baselines, Sequence):
            raise OpenDreamerQualityError("required_baselines must be a sequence")
        return cls(
            dataset_path=dataset_path,
            train_split=_config_string(payload.get("train_split", "train"), "train_split"),
            holdout_split=_config_string(payload.get("holdout_split", "test"), "holdout_split"),
            max_linear_speed=_finite_float(
                actions.get("max_linear_speed", 1.0), "max_linear_speed"
            ),
            max_angular_speed=_finite_float(
                actions.get("max_angular_speed", 1.0), "max_angular_speed"
            ),
            min_linear_speed=_finite_float(
                actions.get("min_linear_speed", 0.0), "min_linear_speed"
            ),
            latent_dim=_config_int(payload.get("latent_dim", 5), "latent_dim"),
            seed=_config_int(payload.get("seed", 6318), "seed"),
            min_train_episodes=_config_int(
                payload.get("min_train_episodes", 2), "min_train_episodes"
            ),
            min_holdout_episodes=_config_int(
                payload.get("min_holdout_episodes", 2), "min_holdout_episodes"
            ),
            min_train_transitions=_config_int(
                payload.get("min_train_transitions", 8), "min_train_transitions"
            ),
            min_holdout_transitions=_config_int(
                payload.get("min_holdout_transitions", 4), "min_holdout_transitions"
            ),
            multi_step_horizon=_config_int(
                payload.get("multi_step_horizon", 3), "multi_step_horizon"
            ),
            mlp_hidden_dim=_config_int(payload.get("mlp_hidden_dim", 16), "mlp_hidden_dim"),
            ridge_alpha=_finite_float(payload.get("ridge_alpha", 1.0e-4), "ridge_alpha"),
            required_baselines=tuple(str(value) for value in baselines),  # type: ignore[arg-type]
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ModelQualityConfig:
        """Load a config from YAML, resolving the dataset relative to the config file.

        Returns:
            A validated model-quality configuration.
        """
        config_path = Path(path).resolve()
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise OpenDreamerQualityError(f"could not read quality config {config_path}") from exc
        return cls.from_mapping(payload, base_dir=config_path.parent)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe quality configuration."""
        return {
            "schema_version": OPEN_DREAMER_MODEL_QUALITY_VERSION,
            "dataset_path": str(self.dataset_path),
            "train_split": self.train_split,
            "holdout_split": self.holdout_split,
            "action_bounds": {
                "max_linear_speed": self.max_linear_speed,
                "max_angular_speed": self.max_angular_speed,
                "min_linear_speed": self.min_linear_speed,
            },
            "latent_dim": self.latent_dim,
            "seed": self.seed,
            "min_train_episodes": self.min_train_episodes,
            "min_holdout_episodes": self.min_holdout_episodes,
            "min_train_transitions": self.min_train_transitions,
            "min_holdout_transitions": self.min_holdout_transitions,
            "multi_step_horizon": self.multi_step_horizon,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "ridge_alpha": self.ridge_alpha,
            "required_baselines": list(self.required_baselines),
        }


@dataclass(frozen=True, slots=True)
class ModelQualityReport:
    """Serializable result of one quality-gate evaluation."""

    status: QualityStatus
    reason: str | None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the report with status and evidence boundary included."""
        result = dict(self.payload)
        result.update(
            {
                "schema_version": OPEN_DREAMER_MODEL_QUALITY_VERSION,
                "status": self.status,
                "reason": self.reason,
                "evidence_boundary": EVIDENCE_BOUNDARY,
            }
        )
        return _json_safe(result)


@dataclass(frozen=True, slots=True)
class _EpisodeFeatures:
    episode: StructuredEpisode
    observations: np.ndarray
    actions: np.ndarray
    features: np.ndarray


@dataclass(frozen=True, slots=True)
class _Transitions:
    current: np.ndarray
    actions: np.ndarray
    target: np.ndarray
    rewards: np.ndarray
    continuations: np.ndarray
    episode_index: tuple[int, ...]
    step_index: tuple[int, ...]

    @property
    def count(self) -> int:
        """Return the transition count."""
        return int(self.rewards.shape[0])


@dataclass(frozen=True, slots=True)
class _Predictions:
    observations: np.ndarray
    rewards: np.ndarray
    continuations: np.ndarray


def evaluate_model_quality(
    config: ModelQualityConfig,
    *,
    dataset_path: Path | str | None = None,
) -> ModelQualityReport:
    """Evaluate a dataset or return a fail-closed readiness report.

    Returns:
        A model-quality report. The report is ``blocked_contract`` for malformed source data and
    ``blocked_insufficient_data`` when the explicit train/holdout minima are not met.  Neither
    blocked state is a model result.
    """
    path = Path(dataset_path).resolve() if dataset_path is not None else config.dataset_path
    source = _source_metadata(path)
    try:
        episodes = _load_quality_dataset(path)
    except OpenDreamerQualityError as exc:
        return _blocked_report(
            "blocked_contract",
            str(exc),
            config=config,
            source=source,
        )

    bounds = ActionBounds(
        max_linear_speed=config.max_linear_speed,
        max_angular_speed=config.max_angular_speed,
        min_linear_speed=config.min_linear_speed,
    )
    try:
        structured = _adapt_quality_dataset(episodes, bounds)
    except OpenDreamerQualityError as exc:
        return _blocked_report(
            "blocked_contract",
            str(exc),
            config=config,
            source=source,
        )

    train, holdout, split_summary, insufficiencies = _partition_quality_episodes(structured, config)
    payload = {
        "source": source,
        "config": config.to_dict(),
        "split_summary": split_summary,
        "episode_count": len(structured),
    }
    if insufficiencies:
        return ModelQualityReport(
            status="blocked_insufficient_data",
            reason="; ".join(insufficiencies),
            payload=payload,
        )

    return _evaluate_or_block(
        config=config,
        source=source,
        train=train,
        holdout=holdout,
        split_summary=split_summary,
    )


def _partition_quality_episodes(
    structured: Sequence[StructuredEpisode], config: ModelQualityConfig
) -> tuple[list[StructuredEpisode], list[StructuredEpisode], dict[str, Any], list[str]]:
    """Partition adapted episodes and compute explicit split-data insufficiencies.

    Returns:
        Training episodes, holdout episodes, split summary, and insufficiency messages.
    """
    by_split: defaultdict[str, list[StructuredEpisode]] = defaultdict(list)
    for episode in structured:
        by_split[episode.split].append(episode)
    train = by_split[config.train_split]
    holdout = by_split[config.holdout_split]
    train_transitions = sum(max(episode.step_count - 1, 0) for episode in train)
    holdout_transitions = sum(max(episode.step_count - 1, 0) for episode in holdout)
    split_summary = _split_summary(by_split, train_transitions, holdout_transitions, config)
    insufficiencies = []
    if len(train) < config.min_train_episodes:
        insufficiencies.append(
            f"{config.train_split} episodes {len(train)} < {config.min_train_episodes}"
        )
    if len(holdout) < config.min_holdout_episodes:
        insufficiencies.append(
            f"{config.holdout_split} episodes {len(holdout)} < {config.min_holdout_episodes}"
        )
    if train_transitions < config.min_train_transitions:
        insufficiencies.append(
            f"{config.train_split} transitions {train_transitions} < {config.min_train_transitions}"
        )
    if holdout_transitions < config.min_holdout_transitions:
        insufficiencies.append(
            f"{config.holdout_split} transitions {holdout_transitions} < "
            f"{config.min_holdout_transitions}"
        )
    return train, holdout, split_summary, insufficiencies


def _evaluate_or_block(
    *,
    config: ModelQualityConfig,
    source: Mapping[str, Any],
    train: Sequence[StructuredEpisode],
    holdout: Sequence[StructuredEpisode],
    split_summary: Mapping[str, Any],
) -> ModelQualityReport:
    """Run the sufficient-data gate and classify every contract failure explicitly.

    Returns:
        A passed, failed, or fail-closed readiness report.
    """
    try:
        return _evaluate_sufficient_dataset(
            config=config,
            source=source,
            train=train,
            holdout=holdout,
            split_summary=split_summary,
        )
    except _InsufficientQualityData as exc:
        return _blocked_report(
            "blocked_insufficient_data",
            str(exc),
            config=config,
            source=source,
            split_summary=split_summary,
        )
    except (OpenDreamerDynamicsError, OpenDreamerQualityError, ValueError) as exc:
        return _blocked_report(
            "blocked_contract",
            f"quality-gate contract failed closed: {exc}",
            config=config,
            source=source,
            split_summary=split_summary,
        )


def _load_quality_dataset(path: Path) -> list[Any]:
    """Load the versioned trajectory dataset and preserve a contract-specific failure message.

    Returns:
        Validated episode rows from the trajectory dataset.
    """
    try:
        return load_rl_trajectory_dataset(path)
    except (OSError, ValueError) as exc:
        raise OpenDreamerQualityError(f"RLTrajectoryDataset.v1 could not be loaded: {exc}") from exc


def _adapt_quality_dataset(
    episodes: Sequence[Any], bounds: ActionBounds
) -> list[StructuredEpisode]:
    """Adapt source episodes into the structured model-quality view.

    Returns:
        Structured episodes with validated actions and observations.
    """
    try:
        return adapt_episodes(episodes, action_bounds=bounds)
    except (TypeError, ValueError) as exc:
        raise OpenDreamerQualityError(
            f"structured-observation adaptation failed closed: {exc}"
        ) from exc


def write_model_quality_report(report: ModelQualityReport, path: Path | str) -> None:
    """Write one deterministic JSON report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _evaluate_sufficient_dataset(
    *,
    config: ModelQualityConfig,
    source: Mapping[str, Any],
    train: Sequence[StructuredEpisode],
    holdout: Sequence[StructuredEpisode],
    split_summary: Mapping[str, Any],
) -> ModelQualityReport:
    train_observations = [_structured_observations(episode) for episode in train]
    holdout_observations = [_structured_observations(episode) for episode in holdout]
    observation_widths = {array.shape[1] for array in train_observations + holdout_observations}
    if len(observation_widths) != 1:
        raise OpenDreamerQualityError(
            f"structured observation widths differ across episodes: {sorted(observation_widths)}"
        )
    obs_dim = observation_widths.pop()
    if config.latent_dim != obs_dim:
        raise OpenDreamerQualityError(
            f"latent_dim must equal structured observation width {obs_dim} for the v1 gate, "
            f"got {config.latent_dim}"
        )
    train_mean = np.mean(np.concatenate(train_observations, axis=0), axis=0)
    train_scale = np.std(np.concatenate(train_observations, axis=0), axis=0)
    train_scale = np.where(train_scale < 1.0e-6, 1.0, train_scale)
    action_bounds = ActionBounds(
        max_linear_speed=config.max_linear_speed,
        max_angular_speed=config.max_angular_speed,
        min_linear_speed=config.min_linear_speed,
    )
    train_features = _build_episode_features(train, train_mean, train_scale, action_bounds)
    holdout_features = _build_episode_features(holdout, train_mean, train_scale, action_bounds)
    train_batch = _build_transitions(train_features)
    holdout_batch = _build_transitions(holdout_features)
    if train_batch.count == 0 or holdout_batch.count == 0:
        raise OpenDreamerQualityError("quality gate requires at least one transition per split")
    _require_continuation_diversity(train_batch, "train")
    _require_continuation_diversity(holdout_batch, "holdout")

    model = _fit_latent_model(config, train_batch, train_features, train_mean, train_scale)
    mlp = _RandomFeatureMLP(
        input_dim=obs_dim + 2,
        output_dim=obs_dim + 2,
        hidden_dim=config.mlp_hidden_dim,
        seed=config.seed + 1,
        ridge_alpha=config.ridge_alpha,
    )
    mlp.fit(
        np.column_stack([train_batch.current, train_batch.actions]),
        np.column_stack([train_batch.target, train_batch.rewards, train_batch.continuations]),
    )
    train_reward_mean = float(np.mean(train_batch.rewards))
    train_continuation_mean = float(np.mean(train_batch.continuations))

    one_step = {
        "model": _metrics(
            _predict_model(model, holdout_batch, train_mean, train_scale), holdout_batch
        ),
        "persistence": _metrics(
            _predict_persistence(holdout_batch, train_reward_mean, train_continuation_mean),
            holdout_batch,
        ),
        "mlp": _metrics(_predict_mlp(mlp, holdout_batch), holdout_batch),
    }
    multi_step = {
        "model": _multi_step_metrics(
            holdout_features,
            model,
            train_mean,
            train_scale,
            config.multi_step_horizon,
        ),
        "persistence": _multi_step_metrics(
            holdout_features,
            _PersistencePredictor(train_reward_mean, train_continuation_mean),
            train_mean,
            train_scale,
            config.multi_step_horizon,
        ),
        "mlp": _multi_step_metrics(
            holdout_features,
            mlp,
            train_mean,
            train_scale,
            config.multi_step_horizon,
        ),
    }
    gate = _gate_metrics(config, one_step, multi_step)
    scenario_metrics = _scenario_metrics(
        holdout_features,
        model,
        train_mean,
        train_scale,
        config.multi_step_horizon,
    )
    payload = {
        "source": source,
        "config": config.to_dict(),
        "split_summary": split_summary,
        "episode_count": len(train) + len(holdout),
        "feature_transform": {
            "standardization": "training-split mean and standard deviation",
            "latent_projection": "tanh",
            "observation_units": "preserved adapter recorder units before normalization",
            "action_units": "adapter physical velocity units inverted to bounded [-1, 1]",
            "action_bounds": action_bounds.to_dict(),
        },
        "model": {
            "architecture": "clean-room action-conditioned latent transition",
            "fitted": True,
            "fit_method": "ridge_closed_form",
            "route": "clean_room",
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "obs_dim": obs_dim,
            "latent_dim": config.latent_dim,
        },
        "one_step_metrics": one_step,
        "multi_step_metrics": multi_step,
        "scenario_metrics": scenario_metrics,
        "gate": gate,
    }
    status: QualityStatus = "passed" if gate["passed"] else "failed_model_quality"
    reason = None if status == "passed" else "fitted model did not beat every required baseline"
    return ModelQualityReport(status=status, reason=reason, payload=payload)


def _fit_latent_model(
    config: ModelQualityConfig,
    batch: _Transitions,
    features: Sequence[_EpisodeFeatures],
    mean: np.ndarray,
    scale: np.ndarray,
) -> LatentDynamicsModel:
    del features, mean, scale
    dynamics_config = DynamicsConfig(
        obs_dim=batch.current.shape[1],
        latent_dim=config.latent_dim,
        seed=config.seed,
    )
    transition_features = np.column_stack([batch.current, batch.actions])
    transition_targets = _safe_atanh(batch.target)
    transition_coefficients = _ridge_fit(
        transition_features,
        transition_targets,
        alpha=config.ridge_alpha,
        include_bias=True,
    )
    head_features = np.column_stack([batch.target, np.ones(batch.count)])
    reward_coefficients = _ridge_fit(
        head_features,
        batch.rewards[:, None],
        alpha=config.ridge_alpha,
        include_bias=False,
    )
    continuation_logits = _safe_logit(batch.continuations)[:, None]
    continuation_coefficients = _ridge_fit(
        head_features,
        continuation_logits,
        alpha=config.ridge_alpha,
        include_bias=False,
    )
    obs_dim = dynamics_config.obs_dim
    latent_dim = dynamics_config.latent_dim
    if obs_dim != latent_dim:
        raise OpenDreamerQualityError(
            "v1 fitted model requires equal observation and latent widths"
        )
    weights = DynamicsWeights(
        w_enc=np.eye(obs_dim, dtype=float),
        b_enc=np.zeros(latent_dim, dtype=float),
        w_latent=transition_coefficients[:latent_dim].T,
        w_action=transition_coefficients[latent_dim : latent_dim + 2].T,
        b_latent=transition_coefficients[-1],
        w_reward=reward_coefficients[:latent_dim, 0],
        b_reward=float(reward_coefficients[-1, 0]),
        w_cont=continuation_coefficients[:latent_dim, 0],
        b_cont=float(continuation_coefficients[-1, 0]),
    )
    return LatentDynamicsModel(dynamics_config, weights)


def _ridge_fit(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    include_bias: bool,
) -> np.ndarray:
    """Fit a finite ridge readout, returning coefficients with bias in the final row.

    Returns:
        Fitted coefficient matrix, with a final bias row when requested.
    """
    if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
        raise OpenDreamerQualityError("ridge inputs and targets must be aligned matrices")
    design = np.column_stack([inputs, np.ones(inputs.shape[0])]) if include_bias else inputs
    gram = design.T @ design
    gram += alpha * np.eye(gram.shape[0], dtype=float)
    if include_bias and alpha:
        gram[-1, -1] -= alpha
    rhs = design.T @ targets
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    if not np.all(np.isfinite(coefficients)):
        raise OpenDreamerQualityError("ridge fit produced non-finite coefficients")
    return coefficients


class _RandomFeatureMLP:
    """Small deterministic one-hidden-layer MLP reference with a fitted output head."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        seed: int,
        ridge_alpha: float,
    ) -> None:
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._ridge_alpha = ridge_alpha
        rng = np.random.default_rng(seed)
        self._w_hidden = rng.normal(0.0, 1.0 / np.sqrt(input_dim), (input_dim, hidden_dim))
        self._b_hidden = rng.normal(0.0, 0.1, hidden_dim)
        self._output: np.ndarray | None = None
        self._target_mean: np.ndarray | None = None
        self._target_scale: np.ndarray | None = None

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        if inputs.ndim != 2 or inputs.shape[1] != self._input_dim:
            raise OpenDreamerQualityError("MLP input width mismatch")
        if targets.ndim != 2 or targets.shape != (inputs.shape[0], self._output_dim):
            raise OpenDreamerQualityError("MLP target shape mismatch")
        self._target_mean = np.mean(targets, axis=0)
        target_std = np.std(targets, axis=0)
        self._target_scale = np.where(target_std < 1.0e-6, 1.0, target_std)
        scaled_targets = (targets - self._target_mean) / self._target_scale
        self._output = _ridge_fit(
            self._hidden_design(inputs),
            scaled_targets,
            alpha=self._ridge_alpha,
            include_bias=False,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self._output is None or self._target_mean is None or self._target_scale is None:
            raise OpenDreamerQualityError("MLP must be fitted before prediction")
        scaled = self._hidden_design(inputs) @ self._output
        result = scaled * self._target_scale + self._target_mean
        if not np.all(np.isfinite(result)):
            raise OpenDreamerQualityError("MLP produced non-finite predictions")
        return result

    def _hidden_design(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 2 or inputs.shape[1] != self._input_dim:
            raise OpenDreamerQualityError("MLP input width mismatch")
        hidden = np.tanh(inputs @ self._w_hidden + self._b_hidden)
        return np.column_stack([hidden, np.ones(inputs.shape[0])])


class _PersistencePredictor:
    """Adapter-shaped persistence predictor used for multi-step evaluation."""

    def __init__(self, reward: float, continuation: float) -> None:
        self._reward = reward
        self._continuation = continuation

    def predict_step(
        self, latent: np.ndarray, action: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        del action
        return latent.copy(), self._reward, self._continuation


def _structured_observations(episode: StructuredEpisode) -> np.ndarray:
    rows = []
    for step in episode.observations:
        row = (
            np.concatenate([step.drive_state, step.rays])
            if step.rays_available
            else step.drive_state
        )
        rows.append(np.asarray(row, dtype=float))
    result = np.stack(rows, axis=0)
    if not np.all(np.isfinite(result)):
        raise OpenDreamerQualityError("structured observations contain non-finite values")
    return result


def _structured_actions(episode: StructuredEpisode) -> np.ndarray:
    result = np.asarray([step.raw for step in episode.actions], dtype=float)
    if result.ndim != 2 or result.shape[1] != 2 or not np.all(np.isfinite(result)):
        raise OpenDreamerQualityError("structured actions must be finite two-dimensional values")
    return result


def _normalize_actions(actions: np.ndarray, bounds: ActionBounds) -> np.ndarray:
    """Invert the adapter's physical action map into the dynamics model's bounded action space.

    Returns:
        A finite action matrix with both components clipped to ``[-1, 1]``.
    """
    linear_span = bounds.max_linear_speed - bounds.min_linear_speed
    normalized = np.column_stack(
        [
            2.0 * (actions[:, 0] - bounds.min_linear_speed) / linear_span - 1.0,
            actions[:, 1] / bounds.max_angular_speed,
        ]
    )
    if not np.all(np.isfinite(normalized)) or np.any(np.abs(normalized) > 1.0 + 1.0e-9):
        raise OpenDreamerQualityError("physical actions could not be mapped into [-1, 1]")
    return np.clip(normalized, -1.0, 1.0)


def _build_episode_features(
    episodes: Sequence[StructuredEpisode],
    mean: np.ndarray,
    scale: np.ndarray,
    action_bounds: ActionBounds,
) -> list[_EpisodeFeatures]:
    result = []
    for episode in episodes:
        observations = _structured_observations(episode)
        actions = _normalize_actions(_structured_actions(episode), action_bounds)
        standardized = (observations - mean) / scale
        features = np.tanh(standardized)
        result.append(
            _EpisodeFeatures(
                episode=episode,
                observations=observations,
                actions=actions,
                features=features,
            )
        )
    return result


def _build_transitions(features: Sequence[_EpisodeFeatures]) -> _Transitions:
    current: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    target: list[np.ndarray] = []
    rewards: list[float] = []
    continuations: list[float] = []
    episode_indices: list[int] = []
    step_indices: list[int] = []
    for episode_index, item in enumerate(features):
        for step_index in range(max(item.episode.step_count - 1, 0)):
            current.append(item.features[step_index])
            actions.append(item.actions[step_index])
            target.append(item.features[step_index + 1])
            rewards.append(float(item.episode.rewards[step_index]))
            continuations.append(
                float(
                    not (
                        item.episode.terminated[step_index + 1]
                        or item.episode.truncated[step_index + 1]
                    )
                )
            )
            episode_indices.append(episode_index)
            step_indices.append(step_index)
    if not rewards:
        return _Transitions(
            current=np.empty((0, 0)),
            actions=np.empty((0, 2)),
            target=np.empty((0, 0)),
            rewards=np.empty(0),
            continuations=np.empty(0),
            episode_index=(),
            step_index=(),
        )
    return _Transitions(
        current=np.stack(current),
        actions=np.stack(actions),
        target=np.stack(target),
        rewards=np.asarray(rewards, dtype=float),
        continuations=np.asarray(continuations, dtype=float),
        episode_index=tuple(episode_indices),
        step_index=tuple(step_indices),
    )


def _require_continuation_diversity(batch: _Transitions, split: str) -> None:
    """Require both continuing and terminal transitions before judging the continuation head."""
    observed = {float(value) for value in batch.continuations}
    if observed != {0.0, 1.0}:
        raise _InsufficientQualityData(
            f"{split} continuation targets must contain both continuing and terminal transitions; "
            f"observed {sorted(observed)}"
        )


def _predict_model(
    model: LatentDynamicsModel,
    batch: _Transitions,
    mean: np.ndarray,
    scale: np.ndarray,
) -> _Predictions:
    del mean, scale
    observations = []
    rewards = []
    continuations = []
    for current, action in zip(batch.current, batch.actions, strict=True):
        transition = model.step(current, action)
        observations.append(transition.latent)
        rewards.append(transition.reward)
        continuations.append(transition.continuation)
    return _Predictions(
        observations=np.stack(observations),
        rewards=np.asarray(rewards, dtype=float),
        continuations=np.asarray(continuations, dtype=float),
    )


def _predict_persistence(batch: _Transitions, reward: float, continuation: float) -> _Predictions:
    return _Predictions(
        observations=np.array(batch.current, copy=True),
        rewards=np.full(batch.count, reward, dtype=float),
        continuations=np.full(batch.count, continuation, dtype=float),
    )


def _predict_mlp(mlp: _RandomFeatureMLP, batch: _Transitions) -> _Predictions:
    output = mlp.predict(np.column_stack([batch.current, batch.actions]))
    return _Predictions(
        observations=np.clip(output[:, :-2], -1.0, 1.0),
        rewards=output[:, -2],
        continuations=np.clip(output[:, -1], 0.0, 1.0),
    )


def _metrics(predictions: _Predictions, target: _Transitions) -> dict[str, float]:
    return {
        "next_observation_rmse": float(
            np.sqrt(np.mean((predictions.observations - target.target) ** 2))
        ),
        "reward_mae": float(np.mean(np.abs(predictions.rewards - target.rewards))),
        "continuation_brier": float(
            np.mean((predictions.continuations - target.continuations) ** 2)
        ),
    }


def _multi_step_metrics(
    features: Sequence[_EpisodeFeatures],
    predictor: Any,
    mean: np.ndarray,
    scale: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    del mean, scale
    observation_errors: list[float] = []
    reward_errors: list[float] = []
    continuation_errors: list[float] = []
    for item in features:
        for start in range(max(item.episode.step_count - 1, 0)):
            latent = item.features[start].copy()
            for offset in range(min(horizon, item.episode.step_count - start - 1)):
                step_index = start + offset
                latent, reward, continuation = _predict_step(
                    predictor, latent, item.actions[step_index]
                )
                observation_errors.append(
                    float(np.mean((latent - item.features[step_index + 1]) ** 2))
                )
                reward_errors.append(float(abs(reward - item.episode.rewards[step_index])))
                target_continuation = float(
                    not (
                        item.episode.terminated[step_index + 1]
                        or item.episode.truncated[step_index + 1]
                    )
                )
                continuation_errors.append(float((continuation - target_continuation) ** 2))
    if not observation_errors:
        raise OpenDreamerQualityError("multi-step evaluation has no usable transitions")
    return {
        "next_observation_rmse": float(np.sqrt(np.mean(observation_errors))),
        "reward_mae": float(np.mean(reward_errors)),
        "continuation_brier": float(np.mean(continuation_errors)),
    }


def _predict_step(
    predictor: Any, latent: np.ndarray, action: np.ndarray
) -> tuple[np.ndarray, float, float]:
    if isinstance(predictor, LatentDynamicsModel):
        transition = predictor.step(latent, action)
        return transition.latent, transition.reward, transition.continuation
    if isinstance(predictor, _RandomFeatureMLP):
        output = predictor.predict(np.concatenate([latent, action])[None, :])[0]
        return (
            np.clip(output[:-2], -1.0, 1.0),
            float(output[-2]),
            float(np.clip(output[-1], 0.0, 1.0)),
        )
    return predictor.predict_step(latent, action)


def _scenario_metrics(
    features: Sequence[_EpisodeFeatures],
    model: LatentDynamicsModel,
    mean: np.ndarray,
    scale: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    grouped: dict[str, list[_EpisodeFeatures]] = defaultdict(list)
    for item in features:
        grouped[item.episode.scenario_id].append(item)
    return {
        scenario: _multi_step_metrics(items, model, mean, scale, horizon)
        for scenario, items in sorted(grouped.items())
    }


def _gate_metrics(
    config: ModelQualityConfig,
    one_step: Mapping[str, Mapping[str, float]],
    multi_step: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    metric_names = tuple(one_step["model"])
    multi_step_metric_names = tuple(multi_step["model"])
    if set(metric_names) != set(multi_step_metric_names):
        raise OpenDreamerQualityError(
            "one-step and multi-step metric names must match before quality gating"
        )
    results: dict[str, Any] = {}
    all_passed = True
    for baseline in config.required_baselines:
        comparisons = {}
        baseline_passed = True
        for horizon, metrics in (("one_step", one_step), ("multi_step", multi_step)):
            for name in metric_names:
                model_value = metrics["model"][name]
                baseline_value = metrics[baseline][name]
                passed = _strictly_better(model_value, baseline_value)
                comparisons[f"{horizon}.{name}"] = {
                    "model": model_value,
                    "baseline": baseline_value,
                    "passed": passed,
                }
                baseline_passed = baseline_passed and passed
        results[baseline] = {"passed": baseline_passed, "comparisons": comparisons}
        all_passed = all_passed and baseline_passed
    return {
        "required_baselines": list(config.required_baselines),
        "comparison_rule": (
            "model must strictly improve every reported required metric at one-step and "
            "multi-step horizons"
        ),
        "per_baseline": results,
        "passed": all_passed,
    }


def _strictly_better(model_value: float, baseline_value: float) -> bool:
    tolerance = 1.0e-12
    return model_value < baseline_value - tolerance


def _safe_atanh(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    result = np.arctanh(clipped)
    if not np.all(np.isfinite(result)):
        raise OpenDreamerQualityError("latent target transform produced non-finite values")
    return result


def _safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1.0e-4, 1.0 - 1.0e-4)
    result = np.log(clipped / (1.0 - clipped))
    if not np.all(np.isfinite(result)):
        raise OpenDreamerQualityError("continuation target transform produced non-finite values")
    return result


def _source_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset_path": str(path),
        "durability": "worktree_local_until_promoted",
    }
    try:
        dataset_bytes = path.read_bytes()
    except OSError:
        metadata["dataset_sha256"] = None
        metadata["dataset_bytes"] = None
    else:
        metadata["dataset_sha256"] = hashlib.sha256(dataset_bytes).hexdigest()
        metadata["dataset_bytes"] = len(dataset_bytes)
    metadata["manifest"] = _manifest_metadata(path)
    metadata["collection"] = _collection_metadata(path)
    try:
        metadata["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["git_commit"] = "unknown"
    return metadata


def _manifest_metadata(path: Path) -> dict[str, Any]:
    """Return compact metadata from the adjacent trajectory manifest when available."""
    candidates = [path.with_suffix(".manifest.json")]
    if path.name.endswith(".preview.jsonl"):
        candidates.append(
            path.with_name(path.name.removesuffix(".preview.jsonl") + ".manifest.json")
        )
    manifest_path = next((candidate for candidate in candidates if candidate.exists()), None)
    result: dict[str, Any] = {
        "available": manifest_path is not None,
        "path": str(manifest_path) if manifest_path is not None else None,
    }
    if manifest_path is None:
        return result
    try:
        manifest_bytes = manifest_path.read_bytes()
        payload = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"could not read manifest: {exc}"
        return result
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    result.update(
        {
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "schema_version": payload.get("schema_version"),
            "dataset_sha256": payload.get("dataset_sha256"),
            "git_commit": provenance.get("git_commit"),
            "artifact_durability": provenance.get("artifact_durability"),
            "source_route": _manifest_source_route(provenance),
        }
    )
    return result


def _manifest_source_route(provenance: Mapping[str, Any]) -> str:
    source_jsonl = str(provenance.get("source_jsonl", ""))
    command = str(provenance.get("command", ""))
    if "simulation_step_trace" in command or "simulation_step_trace" in source_jsonl:
        return "native_map_runner_trace"
    if "synthetic" in source_jsonl or "synthetic" in command:
        return "synthetic_or_contract_fixture"
    return "unknown"


def _collection_metadata(path: Path) -> dict[str, Any]:
    """Return route metadata from a collector sidecar when a local diagnostic has one."""
    collection_path = path.with_suffix(".collection.json")
    result: dict[str, Any] = {
        "available": collection_path.exists(),
        "path": str(collection_path) if collection_path.exists() else None,
    }
    if not collection_path.exists():
        return result
    try:
        collection_bytes = collection_path.read_bytes()
        payload = json.loads(collection_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"could not read collection metadata: {exc}"
        return result
    if not isinstance(payload, Mapping):
        result["error"] = "collection metadata must be a JSON object"
        return result
    result.update(
        {
            "sha256": hashlib.sha256(collection_bytes).hexdigest(),
            "schema_version": payload.get("schema_version"),
            "source_route": payload.get("source_route"),
            "git_commit": payload.get("git_commit"),
            "artifact_durability": payload.get("artifact_durability"),
        }
    )
    return result


def _split_summary(
    by_split: Mapping[str, Sequence[StructuredEpisode]],
    train_transitions: int,
    holdout_transitions: int,
    config: ModelQualityConfig,
) -> dict[str, Any]:
    summary = {}
    for split, episodes in sorted(by_split.items()):
        summary[split] = {
            "episode_count": len(episodes),
            "transition_count": sum(max(episode.step_count - 1, 0) for episode in episodes),
            "scenario_ids": sorted({episode.scenario_id for episode in episodes}),
            "episode_ids": sorted(episode.episode_id for episode in episodes),
        }
    summary["requirements"] = {
        "train_split": config.train_split,
        "holdout_split": config.holdout_split,
        "train_transitions_observed": train_transitions,
        "holdout_transitions_observed": holdout_transitions,
    }
    return summary


def _blocked_report(
    status: Literal["blocked_contract", "blocked_insufficient_data"],
    reason: str,
    *,
    config: ModelQualityConfig,
    source: Mapping[str, Any],
    split_summary: Mapping[str, Any] | None = None,
) -> ModelQualityReport:
    payload: dict[str, Any] = {"source": source, "config": config.to_dict()}
    if split_summary is not None:
        payload["split_summary"] = split_summary
    return ModelQualityReport(status=status, reason=reason, payload=payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "EVIDENCE_BOUNDARY",
    "OPEN_DREAMER_MODEL_QUALITY_VERSION",
    "ModelQualityConfig",
    "ModelQualityReport",
    "OpenDreamerQualityError",
    "evaluate_model_quality",
    "write_model_quality_report",
]
