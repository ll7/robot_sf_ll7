"""Progress-weighted expert-action NLL objective for behavioral cloning.

Issue #6951 concrete slice: a named, configurable progress-weighted
expert-action negative-log-likelihood objective with two experiment arms:

* **Arm A (uniform control)**: uniform transition weights, equivalent to
  ordinary mean expert-action NLL.
* **Arm B (progress-weighted)**: signed per-step progress signal derived
  *only* from reduction in explicitly recorded remaining route length,
  normalized by a declared scale, with bounded positive weights.

Progress is never derived from positions, goal coordinates, observations,
displacement, reward, success, timeout, future outcome labels, or any hidden
proxy.  The module fails closed when aligned route-progress provenance is
missing, malformed, non-finite, or misaligned with action steps.

Claim boundary: this module owns the loss objective and config plumbing only.
It is NOT benchmark evidence, navigation-quality evidence, or a paper-facing
result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger

from robot_sf.errors import RobotSfError

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProgressWeightedBcError(RobotSfError, ValueError):
    """Raised when the progress-weighted BC objective fails a fail-closed invariant."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_OBJECTIVE_NAME_UNWEIGHTED: str = "mean_expert_action_nll"
_OBJECTIVE_NAME_PROGRESS_WEIGHTED: str = "progress_weighted_expert_action_nll"

# Default dataset-provenance metadata key for the remaining-route-length array.
_REMAINING_ROUTE_LENGTH_KEY: str = "remaining_route_length"
_REMAINING_ROUTE_LENGTH_METADATA_KEY: str = "remaining_route_length_metadata"
_REQUIRED_ROUTE_LENGTH_PROVENANCE = {
    "schema_version": "remaining_route_length.v1",
    "alignment": "one_value_per_observation",
    "derived_signal": "remaining_before_minus_after",
    "semantics": "remaining_route_length_meters",
    "units": "m",
}


@dataclass(frozen=True, slots=True)
class ProgressWeightedObjectiveConfig:
    """Configuration for the progress-weighted expert-action NLL objective.

    Attributes:
        objective_name: Declared objective name exposed in config and manifest.
            Arm A uses ``"mean_expert_action_nll"``; Arm B uses
            ``"progress_weighted_expert_action_nll"``.
        arm: Experiment arm label (``"A"`` for uniform, ``"B"`` for progress-weighted).
        progress_lambda: Sensitivity of the weight to per-step progress.  For
            Arm A this must be ``0.0``; for Arm B it must be a finite float.
        progress_normalization_scale: Declared scale used to normalize the raw
            per-step progress before the linear weight transform.  Must be > 0.
        weight_min: Lower bound for the clipped per-step weight.  Must be > 0.
        weight_max: Upper bound for the clipped per-step weight.  Must be >= weight_min.
        remaining_route_length_key: NPZ array key for the per-episode
            remaining-route-length signal.  Must be ``"remaining_route_length"``.
        dataset_digest: SHA-256 hex digest of the source NPZ dataset for
            provenance tracking.  Empty string when unknown.
        random_seed: Deterministic seed for any internal RNG operations.
    """

    objective_name: str
    arm: str
    progress_lambda: float = 0.0
    progress_normalization_scale: float = 1.0
    weight_min: float | None = None
    weight_max: float | None = None
    remaining_route_length_key: str = _REMAINING_ROUTE_LENGTH_KEY
    dataset_digest: str = ""
    random_seed: int = 0

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate config coherence after construction."""
        if self.arm not in ("A", "B"):
            raise ProgressWeightedBcError(f"arm must be 'A' or 'B', got {self.arm!r}")
        if self.weight_min is None:
            object.__setattr__(self, "weight_min", 1.0 if self.arm == "A" else 0.5)
        if self.weight_max is None:
            object.__setattr__(self, "weight_max", 1.0 if self.arm == "A" else 2.0)
        weight_min = cast("float", self.weight_min)
        weight_max = cast("float", self.weight_max)
        numeric_values = {
            "progress_lambda": self.progress_lambda,
            "progress_normalization_scale": self.progress_normalization_scale,
            "weight_min": weight_min,
            "weight_max": weight_max,
        }
        for name, value in numeric_values.items():
            if not np.isfinite(value):
                raise ProgressWeightedBcError(f"{name} must be finite, got {value!r}")
        if self.progress_lambda < 0.0:
            raise ProgressWeightedBcError(
                f"progress_lambda must be non-negative, got {self.progress_lambda}"
            )
        if self.arm == "A":
            expected_name = _OBJECTIVE_NAME_UNWEIGHTED
            if self.progress_lambda != 0.0:
                raise ProgressWeightedBcError(
                    f"Arm A requires progress_lambda=0.0, got {self.progress_lambda}"
                )
        else:
            expected_name = _OBJECTIVE_NAME_PROGRESS_WEIGHTED
        if self.objective_name != expected_name:
            raise ProgressWeightedBcError(
                f"objective_name for arm {self.arm!r} must be {expected_name!r}, "
                f"got {self.objective_name!r}"
            )
        if self.progress_normalization_scale <= 0.0:
            raise ProgressWeightedBcError(
                f"progress_normalization_scale must be > 0, got {self.progress_normalization_scale}"
            )
        if weight_min <= 0.0:
            raise ProgressWeightedBcError(f"weight_min must be > 0, got {weight_min}")
        if weight_max < weight_min:
            raise ProgressWeightedBcError(
                f"weight_max ({weight_max}) must be >= weight_min ({weight_min})"
            )
        if self.remaining_route_length_key != _REMAINING_ROUTE_LENGTH_KEY:
            raise ProgressWeightedBcError(
                f"remaining_route_length_key must be {_REMAINING_ROUTE_LENGTH_KEY!r}, "
                f"got {self.remaining_route_length_key!r}"
            )
        if self.arm == "A" and (weight_min != 1.0 or weight_max != 1.0):
            raise ProgressWeightedBcError("Arm A requires weight_min=weight_max=1.0")
        if self.dataset_digest and (
            len(self.dataset_digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in self.dataset_digest)
        ):
            raise ProgressWeightedBcError(
                "dataset_digest must be a 64-character SHA-256 hex digest"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ProgressWeightedObjectiveConfig:
        """Build a validated objective config from YAML/JSON-compatible values.

        Returns:
            A validated objective configuration.
        """

        try:
            return cls(
                objective_name=str(raw["objective_name"]),
                arm=str(raw["arm"]),
                progress_lambda=float(raw.get("progress_lambda", 0.0)),
                progress_normalization_scale=float(raw.get("progress_normalization_scale", 1.0)),
                weight_min=float(raw["weight_min"]) if "weight_min" in raw else None,
                weight_max=float(raw["weight_max"]) if "weight_max" in raw else None,
                remaining_route_length_key=str(
                    raw.get("remaining_route_length_key", _REMAINING_ROUTE_LENGTH_KEY)
                ),
                dataset_digest=str(raw.get("dataset_digest", "")),
                random_seed=int(raw.get("random_seed", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProgressWeightedBcError(
                f"Failed to parse progress-weighted objective config: {exc}"
            ) from exc

    @classmethod
    def arm_a(
        cls,
        *,
        dataset_digest: str = "",
        random_seed: int = 0,
        weight_min: float = 1.0,
        weight_max: float = 1.0,
    ) -> ProgressWeightedObjectiveConfig:
        """Create a uniform-control (Arm A) config with lambda=0.

        Returns:
            Arm A configuration instance.
        """
        return cls(
            objective_name=_OBJECTIVE_NAME_UNWEIGHTED,
            arm="A",
            progress_lambda=0.0,
            progress_normalization_scale=1.0,
            weight_min=weight_min,
            weight_max=weight_max,
            dataset_digest=dataset_digest,
            random_seed=random_seed,
        )

    @classmethod
    def arm_b(
        cls,
        *,
        progress_lambda: float,
        progress_normalization_scale: float,
        weight_min: float = 0.5,
        weight_max: float = 2.0,
        dataset_digest: str = "",
        random_seed: int = 0,
    ) -> ProgressWeightedObjectiveConfig:
        """Create a progress-weighted (Arm B) config.

        Returns:
            Arm B configuration instance.
        """
        return cls(
            objective_name=_OBJECTIVE_NAME_PROGRESS_WEIGHTED,
            arm="B",
            progress_lambda=progress_lambda,
            progress_normalization_scale=progress_normalization_scale,
            weight_min=weight_min,
            weight_max=weight_max,
            dataset_digest=dataset_digest,
            random_seed=random_seed,
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize config to a JSON-friendly manifest dictionary.

        Returns:
            Deterministic, key-sorted dictionary of config fields.
        """
        return dict(sorted(asdict(self).items()))

    def dataset_digest_or_sha256(self, npz_path: Path | None = None) -> str:
        """Return the declared digest or compute SHA-256 from the NPZ path.

        Returns:
            SHA-256 hex digest string, or empty string when unknown.
        """
        if self.dataset_digest:
            return self.dataset_digest
        if npz_path is None:
            return ""
        return _sha256_file(npz_path)


# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------


def load_remaining_route_length_from_npz(
    npz_path: Path,
    *,
    array_key: str = _REMAINING_ROUTE_LENGTH_KEY,
    require_action_alignment: bool = True,
) -> dict[str, Any]:
    """Load and validate the per-episode remaining-route-length array from an NPZ.

    The NPZ must carry:
    * ``<array_key>``: per-episode remaining-route-length with one value per
      observation (actions + 1) per episode.
    * ``<array_key>_metadata``: a metadata mapping declaring provenance and
      semantics. This metadata is required for the progress-weighted arm.

    The function fails closed when the array is absent, contains non-finite
    values, has wrong alignment with action steps, or is structurally malformed.
    The caller must provide a trusted local trajectory artifact because ragged
    per-episode arrays and provenance mappings use NumPy object-array encoding.

    Args:
        npz_path: Path to the trajectory dataset NPZ.
        array_key: Expected array key for remaining-route-length data.
        require_action_alignment: Require an ``actions`` array and validate
            one route-length value per action observation boundary.

    Returns:
        A mapping with keys ``"remaining_route_length"`` (list of 1-D arrays,
        one per episode), ``"provenance"`` (metadata dict), and
        ``"dataset_digest"`` (SHA-256 hex digest of the NPZ file).

    Raises:
        ProgressWeightedBcError: When the array is missing, malformed, or
            non-finite.
    """
    path = Path(npz_path)
    if not path.is_file():
        raise ProgressWeightedBcError(f"remaining-route-length dataset not found at {path}")

    dataset_sha256 = _sha256_file(path)

    raw_actions: np.ndarray | None = None
    # Ragged per-episode arrays are represented as object arrays in the local
    # trajectory artifact format; this loader validates their contents before
    # admitting them to the objective.
    with np.load(str(path), allow_pickle=True) as npz:
        if array_key not in npz.files:
            raise ProgressWeightedBcError(
                f"NPZ is missing required array {array_key!r}; found {sorted(npz.files)}"
            )
        raw = npz[array_key]
        provenance = _route_length_provenance_from_npz(npz, array_key=array_key)
        if "actions" in npz.files:
            raw_actions = npz["actions"]

    remaining = _normalize_remaining_route_length(raw, array_key=array_key)
    _validate_alignment_with_actions(
        remaining,
        raw_actions,
        array_key=array_key,
        require_actions=require_action_alignment,
    )
    _validate_route_length_provenance(provenance)

    return {
        "remaining_route_length": remaining,
        "provenance": provenance,
        "dataset_digest": dataset_sha256,
    }


def _route_length_provenance_from_npz(npz: Any, *, array_key: str) -> dict[str, Any]:
    """Read explicit route-length provenance from a dataset NPZ.

    Returns:
        The declared provenance mapping, or an empty mapping when absent.
    """

    candidates = [f"{array_key}_metadata", _REMAINING_ROUTE_LENGTH_METADATA_KEY]
    for key in candidates:
        if key not in npz.files:
            continue
        raw = npz[key]
        if getattr(raw, "ndim", None) != 0:
            return {}
        value = raw.item()
        return dict(value) if isinstance(value, Mapping) else {}

    metadata_raw = npz.get("metadata")
    if metadata_raw is not None and getattr(metadata_raw, "ndim", None) == 0:
        metadata = metadata_raw.item()
        if isinstance(metadata, Mapping):
            nested = metadata.get("route_progress_provenance")
            if isinstance(nested, Mapping):
                return dict(nested)
    return {}


def _validate_route_length_provenance(provenance: Mapping[str, object]) -> None:
    """Require the explicit non-proxy route-length provenance contract."""

    missing = [
        key
        for key, expected in _REQUIRED_ROUTE_LENGTH_PROVENANCE.items()
        if provenance.get(key) != expected
    ]
    source = str(provenance.get("source", "")).strip().lower()
    forbidden_source_terms = (
        "position",
        "goal",
        "displacement",
        "reward",
        "success",
        "timeout",
        "future",
        "outcome",
        "observation",
    )
    if not source or "route" not in source or "remaining" not in source:
        missing.append("source")
    elif any(term in source for term in forbidden_source_terms):
        missing.append("source_non_proxy")
    if missing:
        raise ProgressWeightedBcError(
            "remaining_route_length provenance is missing or invalid; required "
            "schema_version=remaining_route_length.v1, "
            "alignment=one_value_per_observation, "
            "derived_signal=remaining_before_minus_after, "
            "semantics=remaining_route_length_meters, units=m, and a non-empty source "
            f"(missing/invalid: {sorted(set(missing))})"
        )


def _normalize_remaining_route_length(raw: np.ndarray, *, array_key: str) -> list[np.ndarray]:
    """Normalize raw NPZ data into a list of 1-D float64 per-episode arrays.

    The array may be:
    * rectangular ``(episodes, steps)`` -> list of 1-D slices.
    * ragged ``(episodes,)`` object array -> list of per-episode 1-D arrays.

    Returns:
        List of 1-D float64 arrays, one per episode.

    Raises:
        ProgressWeightedBcError: When the data is scalar, empty, or contains
            non-finite values.
    """
    data = np.asarray(raw)
    if data.ndim == 0:
        raise ProgressWeightedBcError(
            f"NPZ array {array_key!r} is scalar; expected per-episode route length"
        )

    if data.ndim == 1 and data.dtype == object:
        episodes = [np.asarray(entry, dtype=np.float64).ravel() for entry in data]
    elif data.ndim == 1:
        episodes = [np.asarray(data, dtype=np.float64).ravel()]
    elif data.ndim == 2:
        episodes = [np.asarray(row, dtype=np.float64).ravel() for row in data]
    else:
        raise ProgressWeightedBcError(f"NPZ array {array_key!r} has unexpected ndim={data.ndim}")

    if not episodes:
        raise ProgressWeightedBcError(f"NPZ array {array_key!r} has zero episodes")

    for idx, ep in enumerate(episodes):
        if ep.size == 0:
            raise ProgressWeightedBcError(f"NPZ array {array_key!r} episode {idx} has zero steps")
        if not np.all(np.isfinite(ep)):
            raise ProgressWeightedBcError(
                f"NPZ array {array_key!r} episode {idx} contains non-finite values"
            )

    return episodes


def _validate_alignment_with_actions(
    remaining: list[np.ndarray],
    raw_actions: np.ndarray | None,
    *,
    array_key: str,
    require_actions: bool,
) -> None:
    """Validate that remaining-route-length has one value per observation (actions+1).

    This function also checks alignment against the ``actions`` array in the
    same NPZ.  Action alignment is required for Arm-B route-progress inputs.

    Raises:
        ProgressWeightedBcError: When alignment is wrong.
    """
    if raw_actions is None:
        if require_actions:
            raise ProgressWeightedBcError(
                f"NPZ array {array_key!r} requires an actions array for per-step alignment"
            )
        return
    actions_episodes = _count_action_episodes(raw_actions, episode_count=len(remaining))

    if len(remaining) != len(actions_episodes):
        raise ProgressWeightedBcError(
            f"remaining_route_length has {len(remaining)} episodes but actions has "
            f"{len(actions_episodes)} episodes"
        )

    for idx, (rl, act_steps) in enumerate(zip(remaining, actions_episodes, strict=True)):
        expected_obs = act_steps + 1
        if rl.size != expected_obs:
            raise ProgressWeightedBcError(
                f"episode {idx}: remaining_route_length has {rl.size} values but "
                f"expected {expected_obs} (actions={act_steps} + 1)"
            )


def _count_action_episodes(
    raw_actions: np.ndarray,
    *,
    episode_count: int | None = None,
) -> list[int]:
    """Return per-episode action step counts from the NPZ actions array.

    For rectangular arrays ``(episodes, steps, action_dim)`` all episodes share
    the same step count; the function returns one entry per episode.  For ragged
    object arrays ``(episodes,)`` each episode may have a different step count.
    """
    data = np.asarray(raw_actions)
    if data.ndim == 0:
        return []
    if data.ndim == 1 and data.dtype == object:
        return [int(np.asarray(ep).shape[0]) for ep in data]
    if data.ndim == 1:
        return [int(data.shape[0])]
    if data.ndim == 2 and episode_count == 1:
        return [int(data.shape[0])]
    if data.ndim >= 2:
        step_count = int(data.shape[1])
        episode_count = int(data.shape[0])
        return [step_count] * episode_count
    return []


# ---------------------------------------------------------------------------
# Progress weight computation
# ---------------------------------------------------------------------------


def compute_progress_weights(  # noqa: C901
    remaining_route_length: list[np.ndarray] | None,
    config: ProgressWeightedObjectiveConfig,
    *,
    action_step_counts: list[int] | None = None,
) -> list[np.ndarray]:
    """Compute per-step bounded positive weights from remaining route length.

    For each episode and each step t (where t ranges over action steps):
        progress_t = remaining[t] - remaining[t + 1]
        normalized_t = progress_t / progress_normalization_scale
        raw_weight_t = 1 + lambda * normalized_t
        weight_t = clip(raw_weight_t, weight_min, weight_max)

    Stalled samples (progress <= 0) produce weights <= 1.0 and are not
    silently removed.  Regressing samples (progress < 0) produce weights
    < 1.0 down to the clipped minimum.

    For Arm A (lambda=0), all weights are 1.0 regardless of progress.

    Args:
        remaining_route_length: Per-episode remaining-route-length arrays,
            each with actions+1 values. Arm A may pass ``None`` when
            ``action_step_counts`` is supplied because it does not consume
            route-progress data.
        config: Objective configuration.
        action_step_counts: Optional per-episode action counts used to build
            uniform Arm-A weights without requiring route-progress data.

    Returns:
        Per-episode list of 1-D weight arrays, one per action step.

    Raises:
        ProgressWeightedBcError: When the progress signal is non-finite or
            the arrays are misaligned.
    """
    if config.arm == "A":
        if remaining_route_length is None:
            if action_step_counts is None or not action_step_counts:
                raise ProgressWeightedBcError(
                    "Arm A requires action_step_counts when route-progress data is absent"
                )
            if any(int(count) <= 0 for count in action_step_counts):
                raise ProgressWeightedBcError("Arm A action_step_counts must all be positive")
            return [np.ones(int(count), dtype=np.float64) for count in action_step_counts]

        uniform_weights: list[np.ndarray] = []
        for ep_idx, rl in enumerate(remaining_route_length):
            n_steps = np.asarray(rl).size - 1
            if n_steps <= 0:
                raise ProgressWeightedBcError(
                    f"episode {ep_idx}: route-length data must contain at least one action"
                )
            uniform_weights.append(np.ones(n_steps, dtype=np.float64))
        return uniform_weights

    if remaining_route_length is None:
        raise ProgressWeightedBcError(
            "Arm B requires explicit remaining_route_length provenance; none was provided"
        )

    per_episode_weights: list[np.ndarray] = []

    for ep_idx, rl in enumerate(remaining_route_length):
        rl = np.asarray(rl, dtype=np.float64).ravel()
        if rl.size < 2:
            raise ProgressWeightedBcError(
                f"episode {ep_idx}: remaining_route_length must have >= 2 values "
                f"to compute progress, got {rl.size}"
            )

        if not np.all(np.isfinite(rl)):
            raise ProgressWeightedBcError(
                f"episode {ep_idx}: remaining_route_length contains non-finite values"
            )

        # progress[t] = remaining[t] - remaining[t+1] for t in 0..N-2
        progress = rl[:-1] - rl[1:]

        if not np.all(np.isfinite(progress)):
            raise ProgressWeightedBcError(
                f"episode {ep_idx}: computed progress contains non-finite values"
            )

        normalized = progress / config.progress_normalization_scale
        raw_weights = 1.0 + config.progress_lambda * normalized
        clipped = np.clip(
            raw_weights,
            cast("float", config.weight_min),
            cast("float", config.weight_max),
        )
        per_episode_weights.append(clipped.astype(np.float64))

    return per_episode_weights


# ---------------------------------------------------------------------------
# Weighted NLL loss
# ---------------------------------------------------------------------------


def weighted_expert_action_nll(
    predicted_log_probs: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Compute the weighted negative log-likelihood of expert actions.

    Args:
        predicted_log_probs: Per-step log-probabilities of the expert action,
            shape ``(n_steps,)``.
        weights: Per-step non-negative weights, shape ``(n_steps,)``.
            If ``None``, computes the unweighted mean NLL.

    Returns:
        The weighted mean NLL (scalar).

    Raises:
        ProgressWeightedBcError: When shapes mismatch or inputs are non-finite.
    """
    log_probs = np.asarray(predicted_log_probs, dtype=np.float64).ravel()
    if log_probs.size == 0:
        raise ProgressWeightedBcError("predicted_log_probs must not be empty")
    if not np.all(np.isfinite(log_probs)):
        raise ProgressWeightedBcError("predicted_log_probs contains non-finite values")

    if weights is None:
        return float(-np.mean(log_probs))

    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != log_probs.shape:
        raise ProgressWeightedBcError(
            f"weights shape {w.shape} does not match log_probs shape {log_probs.shape}"
        )
    if not np.all(np.isfinite(w)):
        raise ProgressWeightedBcError("weights contain non-finite values")
    if np.any(w < 0.0):
        raise ProgressWeightedBcError("weights must be non-negative")

    total_weight = np.sum(w)
    if total_weight <= 0.0:
        raise ProgressWeightedBcError(f"sum of weights is {total_weight}; must be > 0")

    return float(-np.sum(w * log_probs) / total_weight)


# ---------------------------------------------------------------------------
# Progress-weighted BC trainer adapter
# ---------------------------------------------------------------------------


class ProgressWeightedBCTrainer:
    """Standalone BC trainer that applies per-step weighted expert-action NLL.

    This adapter bypasses imitation's default loss and computes the
    progress-weighted (or uniform) NLL directly, so that the weighted
    objective is never silently replaced by ordinary imitation BC.

    It requires a PPO policy model (holding the neural-net policy) and
    a list of imitation-compatible trajectories.  The trainer exposes a
    ``train(n_epochs)`` interface compatible with the existing pipeline.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        observation_space: Any,
        action_space: Any,
        demonstrations: list[Any],
        policy: Any,
        config: ProgressWeightedObjectiveConfig,
        weights: list[np.ndarray] | None = None,
        batch_size: int = 32,
        rng: np.random.Generator | None = None,
        device: str = "cpu",
        learning_rate: float = 0.0003,
    ) -> None:
        """Initialize the progress-weighted BC trainer."""
        if batch_size <= 0:
            raise ProgressWeightedBcError(f"batch_size must be positive, got {batch_size}")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ProgressWeightedBcError(
                f"learning_rate must be finite and positive, got {learning_rate}"
            )
        if config.arm == "B" and weights is None:
            raise ProgressWeightedBcError("Arm B requires per-transition progress weights")
        self._obs_space = observation_space
        self._act_space = action_space
        self._demonstrations = demonstrations
        self._policy = policy
        self._config = config
        self._weights = weights
        self._batch_size = batch_size
        self._rng = rng if rng is not None else np.random.default_rng(config.random_seed)
        self._device = device
        self._learning_rate = float(learning_rate)
        self._n_updates = 0
        self._last_loss = float("nan")
        self._optimizer: Any = None

    def set_demonstrations(self, demonstrations: list[Any]) -> None:
        """Replace the demonstration dataset."""
        self._demonstrations = demonstrations

    def train(self, *, n_epochs: int = 10) -> None:
        """Run weighted BC training for the given number of epochs.

        Each epoch processes all demonstrations shuffled into batches.
        The loss is the per-step weighted mean NLL over the current batch.
        """
        if not self._demonstrations:
            raise ProgressWeightedBcError("No demonstrations provided for weighted BC training")

        all_obs, all_acts, all_weights = self._unpack_demonstrations()
        n_samples = all_obs.shape[0]

        for _epoch in range(n_epochs):
            perm = self._rng.permutation(n_samples)
            for start in range(0, n_samples, self._batch_size):
                idx = perm[start : start + self._batch_size]
                batch_obs = all_obs[idx]
                batch_acts = all_acts[idx]
                batch_w = all_weights[idx]

                self._update_step(batch_obs, batch_acts, batch_w)
                self._n_updates += 1
            logger.bind(
                arm=self._config.arm,
                objective=self._config.objective_name,
            ).info("weighted BC epoch complete epoch={} loss={}", _epoch + 1, self._last_loss)

    def _unpack_demonstrations(  # noqa: C901
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten all demonstrations into (obs, acts, weights) arrays.

        Returns:
            Tuple of (observations, actions, weights) arrays with all
            episodes concatenated along the first axis.
        """
        from gymnasium.spaces.utils import flatten as flatten_space  # noqa: PLC0415

        all_obs: list[np.ndarray] = []
        all_acts: list[np.ndarray] = []
        all_weights: list[np.ndarray] = []

        ep_idx = 0
        for traj in self._demonstrations:
            obs_ep = np.asarray(traj.obs, dtype=np.float32)
            acts_ep = np.asarray(traj.acts, dtype=np.float32)

            # Validate the transition-length contract before any observation
            # flattening.  In particular, an empty object/dict observation
            # array cannot be safely indexed by the flattening probe below.
            n_actions = acts_ep.shape[0]
            if obs_ep.shape[0] < n_actions:
                raise ProgressWeightedBcError(
                    f"Episode {ep_idx}: observations ({obs_ep.shape[0]}) are shorter than "
                    f"actions ({n_actions})"
                )

            # Flatten observations if they are dict-like
            if (obs_ep.ndim == 1 and obs_ep.dtype == object) or (
                obs_ep.ndim > 1 and hasattr(obs_ep[0], "keys")
            ):
                flat_obs = []
                for o in obs_ep:
                    try:
                        flat_obs.append(
                            np.asarray(flatten_space(self._obs_space, o), dtype=np.float32)
                        )
                    except (AssertionError, ValueError, TypeError):
                        flat_obs.append(np.asarray(o, dtype=np.float32).ravel())
                obs_ep = np.stack(flat_obs)
            elif obs_ep.ndim > 2:
                obs_ep = obs_ep.reshape(obs_ep.shape[0], -1).astype(np.float32)

            # observations[0..T-1] pair with actions[0..T-1]
            obs_slice = obs_ep[:n_actions]

            if self._weights is None:
                ep_weights = np.ones(n_actions, dtype=np.float64)
            else:
                if ep_idx >= len(self._weights):
                    raise ProgressWeightedBcError(
                        f"Missing weights for demonstration episode {ep_idx}"
                    )
                ep_weights = np.asarray(self._weights[ep_idx], dtype=np.float64)
            if ep_weights.shape[0] != n_actions:
                raise ProgressWeightedBcError(
                    f"Episode {ep_idx}: weights shape {ep_weights.shape} != "
                    f"actions shape {acts_ep.shape}"
                )
            if not np.all(np.isfinite(ep_weights)) or np.any(ep_weights <= 0.0):
                raise ProgressWeightedBcError(
                    f"Episode {ep_idx}: weights must be finite and strictly positive"
                )

            all_obs.append(obs_slice)
            all_acts.append(acts_ep)
            all_weights.append(ep_weights)
            ep_idx += 1

        if not all_obs:
            raise ProgressWeightedBcError("Demonstrations contain no action transitions")
        if self._weights is not None and ep_idx != len(self._weights):
            raise ProgressWeightedBcError(
                f"Weights contain {len(self._weights)} episodes but demonstrations contain {ep_idx}"
            )

        return (
            np.concatenate(all_obs, axis=0),
            np.concatenate(all_acts, axis=0),
            np.concatenate(all_weights, axis=0),
        )

    def _update_step(
        self,
        batch_obs: np.ndarray,
        batch_acts: np.ndarray,
        batch_weights: np.ndarray,
    ) -> None:
        """Single gradient step on a mini-batch with weighted NLL loss."""
        import torch  # noqa: PLC0415

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._learning_rate)

        policy_device = getattr(self._policy, "device", self._device)
        if self._device != "auto":
            policy_device = self._device
        obs_t = torch.as_tensor(batch_obs, dtype=torch.float32, device=policy_device)
        acts_t = torch.as_tensor(batch_acts, dtype=torch.float32, device=policy_device)
        w_t = torch.as_tensor(batch_weights, dtype=torch.float32, device=policy_device)

        self._policy.train()
        dist = self._policy.get_distribution(obs_t)
        log_probs = dist.log_prob(acts_t)

        if log_probs.ndim > 1:
            log_probs = log_probs.sum(dim=-1)

        loss_tensor = -torch.sum(w_t * log_probs) / torch.sum(w_t)
        if not bool(torch.isfinite(loss_tensor).item()):
            raise ProgressWeightedBcError("weighted BC loss is non-finite")

        self._optimizer.zero_grad()
        loss_tensor.backward()
        self._optimizer.step()
        self._last_loss = float(loss_tensor.detach().cpu().item())


def serialize_objective_config(
    config: ProgressWeightedObjectiveConfig,
    *,
    npz_path: Path | None = None,
) -> dict[str, Any]:
    """Serialize the objective config to a deterministic JSON-ready mapping.

    The output is key-sorted and deterministic so it can be embedded in a run
    manifest and compared across invocations.

    Args:
        config: The objective configuration.
        npz_path: Optional NPZ path for dataset digest computation.

    Returns:
        Deterministic, JSON-serializable mapping of the config fields.
    """
    digest = config.dataset_digest_or_sha256(npz_path=npz_path)
    d = config.to_manifest_dict()
    d["dataset_digest"] = digest
    return dict(sorted(d.items()))


def objective_config_json(
    config: ProgressWeightedObjectiveConfig,
    *,
    npz_path: Path | None = None,
) -> str:
    """Return a deterministic JSON string of the objective config."""
    payload = serialize_objective_config(config, npz_path=npz_path)
    return json.dumps(payload, indent=2, sort_keys=True)


def sha256_objective_config(
    config: ProgressWeightedObjectiveConfig,
    *,
    npz_path: Path | None = None,
) -> str:
    """Return SHA-256 hex digest of the deterministic JSON config string.

    Returns:
        64-character hex digest string.
    """
    return hashlib.sha256(
        objective_config_json(config, npz_path=npz_path).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Returns:
        64-character hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "ProgressWeightedBCTrainer",
    "ProgressWeightedBcError",
    "ProgressWeightedObjectiveConfig",
    "compute_progress_weights",
    "load_remaining_route_length_from_npz",
    "objective_config_json",
    "serialize_objective_config",
    "sha256_objective_config",
    "weighted_expert_action_nll",
]
