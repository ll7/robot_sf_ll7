"""Clean-room action-conditioned latent dynamics model (issue #6318 Step 3 module slice).

This module implements the **Step 3 (model module slice, compute-free)** piece of parent issue
#6318's maintainer-authorized sequenced plan. It defines a small, repo-owned, **clean-room** latent
dynamics model with reward and continuation prediction heads that consumes the merged Step 2
adapter (:mod:`robot_sf.research.open_dreamer_adapter`) **read-only**. The adapter module is never
edited by this slice; only its public :class:`~robot_sf.research.open_dreamer_adapter.StructuredEpisode`
view and a couple of contract constants are imported.

Provenance boundary (hard)
--------------------------

Gate 0 (:file:`docs/context/issue_6318_open_dreamer_license_architecture.md`) determined that the
upstream Open Dreamer ``LICENSE`` at the pinned commit reserves all rights, so the only permissible
route is **clean-room**. This module is derived from the public Dreamer 4 paper
(arXiv:2509.24527) and the public upstream *documentation* (README/roadmap) only. No upstream Open
Dreamer source code is copied, vendored, adapted line-by-line, paraphrased as code, or committed
here. The mechanisms below -- an action-conditioned latent transition plus scalar reward and
continuation readouts -- are the paper's publicly described ingredients, restated in Robot SF's own
code around Robot SF's own bounded ``(linear, angular)`` action contract.

What this module does
---------------------

It defines a deterministic, pure-NumPy latent dynamics model:

* an **encoder** that projects one structured observation vector (the adapter's ``drive_state``
  group, concatenated with the ``rays`` group when rays are available) into a bounded latent state
  via ``tanh``;
* an **action-conditioned transition** ``latent_{t+1} = tanh(W_z z_t + W_a a_t + b)`` that predicts
  the next latent state from the current latent state and one bounded 2D action;
* a **reward head** ``r = w_r . z + b_r`` producing a finite scalar reward from a latent state;
* a **continuation head** ``c = sigmoid(w_c . z + b_c)`` producing a finite continuation
  probability strictly inside ``(0, 1)`` from a latent state.

The public entry point :meth:`LatentDynamicsModel.imagine` consumes a
:class:`~robot_sf.research.open_dreamer_adapter.StructuredEpisode`, bootstraps the latent state from
the first encoded observation, and rolls the dynamics forward over the episode's action sequence
(**open-loop latent imagination**), returning a finite :class:`LatentRollout` of next-latent states
plus per-step reward and continuation predictions. The view stays **episode-major**: a single
episode is imagined at a time and episode boundaries are never crossed.

What this module does NOT do (out of scope -- the Step 3 quality gate and Step 4 on parent #6318)
-------------------------------------------------------------------------------------------------

It does **not** train the weights, run a holdout one-step/multi-step prediction quality gate,
compare against a persistence/MLP predictor, integrate with SAC or a replay buffer, or make any
benchmark, metric, or policy claim. The weights are a deterministic small-scale initialization from
a caller-supplied seed (or caller-supplied frozen weights); they are **untrained**. ``evidence_tier``
stays :data:`EVIDENCE_BOUNDARY` (``idea``): a successful dynamics contract smoke is
**diagnostic/contract evidence only**.

Compute-free contract
---------------------

The module is pure NumPy (a core dependency); it requires no GPU, no training loop, no JAX/PyTorch,
and no external data download. Every produced value is finite-validated and the model fails closed
(raises :class:`OpenDreamerDynamicsError`) on non-finite or mis-shaped inputs, on any non-finite
produced output, on a continuation outside ``(0, 1)``, on a config/weight shape mismatch, or on an
episode whose observation width does not match the model's configured width.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

# Read-only consumption of the merged Step 2 adapter contract. This slice must NOT edit
# robot_sf/research/open_dreamer_adapter.py; it imports the public structured-episode view and
# contract constants as an immutable source.
from robot_sf.research.open_dreamer_adapter import (
    DRIVE_STATE_LAYOUT,
    OPEN_DREAMER_OBSERVATION_CONTRACT,
    StructuredEpisode,
)

#: Dynamics contract version. Bumped only if the latent transition or head shapes change.
OPEN_DREAMER_DYNAMICS_VERSION = "open_dreamer_dynamics.v1"

#: Evidence tier for this slice. Diagnostic/contract evidence only -- never a benchmark claim.
EVIDENCE_BOUNDARY = "idea"

#: Dimensionality of the bounded ``(linear, angular)`` action consumed by the transition. This
#: matches the adapter's two-element physical action view, not any upstream action container.
ACTION_DIM = 2

#: Default latent-state width for the deterministic initialization.
DEFAULT_LATENT_DIM = 16

#: Provenance key under which a rollout records the dynamics metadata.
DYNAMICS_PROVENANCE_KEY = "open_dreamer_dynamics"


class OpenDreamerDynamicsError(ValueError):
    """Raised when the dynamics model cannot preserve its finite-output contract.

    Every raise is a fail-closed contract boundary: a non-finite or mis-shaped input, a non-finite
    produced latent/reward/continuation, a continuation outside ``(0, 1)``, a config/weight shape
    mismatch, or an episode whose observation width does not match the model. Callers must treat
    this exception as "blocked", not as a recoverable skip.
    """


def _require_positive_int(value: Any, name: str) -> int:
    """Return one strictly positive integer dimension, rejecting booleans and non-integers.

    Args:
        value: Candidate dimension value.
        name: Dimension name used in the error message.

    Returns:
        The validated positive integer.

    Raises:
        OpenDreamerDynamicsError: If the value is a boolean, not an integer, or not strictly
            positive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenDreamerDynamicsError(f"{name} must be a positive integer, got {value!r}")
    if value <= 0:
        raise OpenDreamerDynamicsError(f"{name} must be strictly positive, got {value!r}")
    return value


def _require_seed(value: Any) -> int:
    """Return one non-boolean integer seed suitable for ``np.random.default_rng``.

    Args:
        value: Candidate seed value.

    Returns:
        The validated integer seed.

    Raises:
        OpenDreamerDynamicsError: If the value is a boolean or not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenDreamerDynamicsError(f"seed must be a non-boolean integer, got {value!r}")
    return value


def _as_finite_float(value: Any, name: str) -> float:
    """Coerce one value to a finite float, rejecting booleans and non-finite values.

    Args:
        value: Candidate scalar value.
        name: Scalar name used in the error message.

    Returns:
        The finite float.

    Raises:
        OpenDreamerDynamicsError: If the value is non-numeric or non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OpenDreamerDynamicsError(f"{name} must be a finite real number, got {value!r}")
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OpenDreamerDynamicsError(f"{name} must be finite, got {value!r}") from exc
    if not np.isfinite(numeric_value):
        raise OpenDreamerDynamicsError(f"{name} must be finite, got {value!r}")
    return numeric_value


def _require_finite_array(
    value: Any,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Return one finite float ndarray with an exact expected shape, frozen read-only.

    Args:
        value: Candidate array value.
        name: Array name used in the error message.
        expected_shape: Exact shape the array must have.

    Returns:
        A read-only finite float ndarray with shape ``expected_shape``.

    Raises:
        OpenDreamerDynamicsError: If the value is not an ndarray, has the wrong shape, has a
            non-float dtype, or contains any non-finite value.
    """
    if not isinstance(value, np.ndarray):
        raise OpenDreamerDynamicsError(
            f"{name} must be a NumPy ndarray, got {type(value).__name__}"
        )
    if value.shape != expected_shape:
        raise OpenDreamerDynamicsError(
            f"{name} must have shape {expected_shape}, got {value.shape}"
        )
    if value.dtype.kind != "f":
        raise OpenDreamerDynamicsError(f"{name} must have a float dtype, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise OpenDreamerDynamicsError(f"{name} must contain only finite values")
    frozen = np.array(value, copy=True)
    frozen.setflags(write=False)
    return frozen


def _require_finite_float_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    """Return one finite float ndarray with an exact number of dimensions.

    Args:
        value: Candidate array value.
        name: Array name used in the error message.
        ndim: Exact number of dimensions the array must have.

    Returns:
        The validated finite float ndarray (not copied; callers freeze as needed).

    Raises:
        OpenDreamerDynamicsError: If the value is not an ndarray, has the wrong number of
            dimensions, has a non-float dtype, or contains any non-finite value.
    """
    if not isinstance(value, np.ndarray):
        raise OpenDreamerDynamicsError(
            f"{name} must be a NumPy ndarray, got {type(value).__name__}"
        )
    if value.ndim != ndim:
        raise OpenDreamerDynamicsError(f"{name} must be {ndim}D, got ndim {value.ndim}")
    if value.dtype.kind != "f" or not np.all(np.isfinite(value)):
        raise OpenDreamerDynamicsError(f"{name} must contain only finite floats")
    return value


def _require_continuations_in_unit_interval(continuations: np.ndarray) -> None:
    """Require every continuation value to lie strictly inside ``(0, 1)``.

    Args:
        continuations: A finite 1D continuation-probability array.

    Raises:
        OpenDreamerDynamicsError: If any value is not strictly inside ``(0, 1)``.
    """
    if continuations.size > 0 and not np.all((continuations > 0.0) & (continuations < 1.0)):
        raise OpenDreamerDynamicsError(
            "continuations must lie strictly inside (0, 1) at every step"
        )


def _stable_sigmoid(logit: float) -> float:
    """Return ``sigmoid(logit)`` computed without overflowing ``exp`` for extreme finite logits.

    Args:
        logit: A finite real-valued logit.

    Returns:
        The sigmoid probability, strictly inside ``(0, 1)`` for any finite logit.
    """
    if logit >= 0.0:
        return float(1.0 / (1.0 + np.exp(-logit)))
    exp_logit = float(np.exp(logit))
    return float(exp_logit / (1.0 + exp_logit))


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    """Shape configuration for the clean-room latent dynamics model.

    The config fixes the observation width the model consumes, the action width (the adapter's
    bounded ``(linear, angular)`` command), the latent-state width, and the deterministic
    initialization seed. It carries no weights; pair it with :class:`DynamicsWeights` built by
    :meth:`DynamicsWeights.from_config`.

    Attributes:
        obs_dim: Width of the structured observation vector consumed by the encoder. Equals
            ``len(DRIVE_STATE_LAYOUT)`` when rays are unavailable, or that plus the ray-vector width
            when rays are available. Must be a positive integer.
        action_dim: Width of the bounded action vector consumed by the transition. Defaults to
            :data:`ACTION_DIM`. Must be a positive integer.
        latent_dim: Width of the latent state. Defaults to :data:`DEFAULT_LATENT_DIM`. Must be a
            positive integer.
        seed: Deterministic seed for the weight initialization. Must be a non-boolean integer.
    """

    obs_dim: int
    action_dim: int = ACTION_DIM
    latent_dim: int = DEFAULT_LATENT_DIM
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate that every dimension is a positive integer and the seed is an integer."""
        _require_positive_int(self.obs_dim, "obs_dim")
        _require_positive_int(self.action_dim, "action_dim")
        _require_positive_int(self.latent_dim, "latent_dim")
        _require_seed(self.seed)

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe representation of the shape configuration.

        Returns:
            A mapping with ``obs_dim``, ``action_dim``, ``latent_dim``, and ``seed``.
        """
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "latent_dim": int(self.latent_dim),
            "seed": int(self.seed),
        }


@dataclass(frozen=True, slots=True)
class DynamicsWeights:
    """Frozen, finite weight bundle for the latent dynamics model and its two heads.

    All arrays are validated for exact shape and finiteness on construction and stored read-only.
    The deterministic initializer :meth:`from_config` draws small-scale Gaussian weights from
    ``np.random.default_rng(seed)`` so the untrained model is reproducible and its ``tanh``/sigmoid
    outputs stay well-conditioned. These weights are **untrained**; this slice performs no learning.

    Attributes:
        w_enc: Encoder projection of shape ``(latent_dim, obs_dim)``.
        b_enc: Encoder bias of shape ``(latent_dim,)``.
        w_latent: Latent self-transition of shape ``(latent_dim, latent_dim)``.
        w_action: Action conditioning of shape ``(latent_dim, action_dim)``.
        b_latent: Transition bias of shape ``(latent_dim,)``.
        w_reward: Reward readout of shape ``(latent_dim,)``.
        b_reward: Reward bias scalar.
        w_cont: Continuation readout of shape ``(latent_dim,)``.
        b_cont: Continuation bias scalar.
    """

    w_enc: np.ndarray
    b_enc: np.ndarray
    w_latent: np.ndarray
    w_action: np.ndarray
    b_latent: np.ndarray
    w_reward: np.ndarray
    b_reward: float
    w_cont: np.ndarray
    b_cont: float

    def __post_init__(self) -> None:
        """Validate every weight array's shape and finiteness, then freeze the bundle."""
        # Shape consistency is checked against the encoder/transition arrays; the config-level
        # dimensions are enforced by LatentDynamicsModel when it pairs a config with these weights.
        if self.w_enc.ndim != 2:
            raise OpenDreamerDynamicsError(
                f"w_enc must be 2D (latent_dim, obs_dim), got shape {self.w_enc.shape}"
            )
        latent_dim, obs_dim = self.w_enc.shape
        _require_positive_int(latent_dim, "latent_dim")
        _require_positive_int(obs_dim, "obs_dim")
        w_enc = _require_finite_array(self.w_enc, "w_enc", (latent_dim, obs_dim))
        b_enc = _require_finite_array(self.b_enc, "b_enc", (latent_dim,))
        w_latent = _require_finite_array(self.w_latent, "w_latent", (latent_dim, latent_dim))
        if self.w_action.ndim != 2 or self.w_action.shape[0] != latent_dim:
            raise OpenDreamerDynamicsError(
                "w_action must be 2D (latent_dim, action_dim) with action_dim >= 1, "
                f"got shape {self.w_action.shape}"
            )
        action_dim = self.w_action.shape[1]
        _require_positive_int(action_dim, "action_dim")
        w_action = _require_finite_array(self.w_action, "w_action", (latent_dim, action_dim))
        b_latent = _require_finite_array(self.b_latent, "b_latent", (latent_dim,))
        w_reward = _require_finite_array(self.w_reward, "w_reward", (latent_dim,))
        w_cont = _require_finite_array(self.w_cont, "w_cont", (latent_dim,))
        b_reward = _as_finite_float(self.b_reward, "b_reward")
        b_cont = _as_finite_float(self.b_cont, "b_cont")
        object.__setattr__(self, "w_enc", w_enc)
        object.__setattr__(self, "b_enc", b_enc)
        object.__setattr__(self, "w_latent", w_latent)
        object.__setattr__(self, "w_action", w_action)
        object.__setattr__(self, "b_latent", b_latent)
        object.__setattr__(self, "w_reward", w_reward)
        object.__setattr__(self, "w_cont", w_cont)
        object.__setattr__(self, "b_reward", b_reward)
        object.__setattr__(self, "b_cont", b_cont)

    @property
    def latent_dim(self) -> int:
        """Return the latent-state width implied by the encoder projection.

        Returns:
            The latent-state width.
        """
        return int(self.w_enc.shape[0])

    @property
    def obs_dim(self) -> int:
        """Return the observation width implied by the encoder projection.

        Returns:
            The observation width.
        """
        return int(self.w_enc.shape[1])

    @property
    def action_dim(self) -> int:
        """Return the action width implied by the action-conditioning matrix.

        Returns:
            The action width.
        """
        return int(self.w_action.shape[1])

    @classmethod
    def from_config(cls, config: DynamicsConfig) -> DynamicsWeights:
        """Build a deterministic, finite, untrained weight bundle from a shape configuration.

        Weights are drawn from ``np.random.default_rng(config.seed)`` with a ``1 / sqrt(fan_in)``
        scale so the untrained ``tanh``/sigmoid outputs stay well-conditioned and finite. Biases are
        zero. The result is fully reproducible for a given config.

        Args:
            config: Shape configuration fixing ``obs_dim``, ``action_dim``, ``latent_dim``, and
                ``seed``.

        Returns:
            A frozen :class:`DynamicsWeights` bundle consistent with ``config``.

        Raises:
            OpenDreamerDynamicsError: If the config is invalid or any drawn weight is non-finite.
        """
        if not isinstance(config, DynamicsConfig):
            raise OpenDreamerDynamicsError(
                f"config must be a DynamicsConfig, got {type(config).__name__}"
            )
        rng = np.random.default_rng(config.seed)
        latent_dim = config.latent_dim
        obs_dim = config.obs_dim
        action_dim = config.action_dim
        weights = cls(
            w_enc=rng.normal(0.0, 1.0 / np.sqrt(obs_dim), (latent_dim, obs_dim)),
            b_enc=np.zeros(latent_dim, dtype=float),
            w_latent=rng.normal(0.0, 1.0 / np.sqrt(latent_dim), (latent_dim, latent_dim)),
            w_action=rng.normal(0.0, 1.0 / np.sqrt(action_dim), (latent_dim, action_dim)),
            b_latent=np.zeros(latent_dim, dtype=float),
            w_reward=rng.normal(0.0, 1.0 / np.sqrt(latent_dim), (latent_dim,)),
            b_reward=0.0,
            w_cont=rng.normal(0.0, 1.0 / np.sqrt(latent_dim), (latent_dim,)),
            b_cont=0.0,
        )
        return weights


@dataclass(frozen=True, slots=True)
class DynamicsStep:
    """One finite action-conditioned dynamics transition result.

    Attributes:
        latent: The predicted next latent state, a finite vector of width ``latent_dim``. Always
            strictly inside ``(-1, 1)`` per component because the transition applies ``tanh``.
        reward: The finite scalar reward predicted from the next latent state.
        continuation: The finite continuation probability predicted from the next latent state,
            strictly inside ``(0, 1)``.
    """

    latent: np.ndarray
    reward: float
    continuation: float

    def __post_init__(self) -> None:
        """Validate the finite-output contract for the next latent state and both heads."""
        if not isinstance(self.latent, np.ndarray):
            raise OpenDreamerDynamicsError(
                f"latent must be a NumPy ndarray, got {type(self.latent).__name__}"
            )
        if self.latent.ndim != 1 or self.latent.dtype.kind != "f":
            raise OpenDreamerDynamicsError(
                f"latent must be a 1D float vector, got shape {self.latent.shape}"
            )
        if not np.all(np.isfinite(self.latent)):
            raise OpenDreamerDynamicsError("latent must contain only finite values")
        reward = _as_finite_float(self.reward, "reward")
        continuation = _as_finite_float(self.continuation, "continuation")
        if not 0.0 < continuation < 1.0:
            raise OpenDreamerDynamicsError(
                f"continuation must lie strictly inside (0, 1), got {continuation!r}"
            )
        latent = np.array(self.latent, copy=True)
        latent.setflags(write=False)
        object.__setattr__(self, "latent", latent)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "continuation", continuation)


@dataclass(frozen=True, slots=True)
class LatentRollout:
    """Episode-major finite result of an open-loop latent imagination rollout.

    The rollout bootstraps the latent state from the first encoded observation and rolls the
    dynamics forward over the action sequence. ``latents[0]`` is the bootstrap latent; ``latents[t]``
    for ``t >= 1`` is the predicted next latent after applying ``actions[t-1]``. ``rewards[t]`` and
    ``continuations[t]`` are the heads read from ``latents[t+1]``.

    Attributes:
        latents: Finite latent trajectory of shape ``(step_count + 1, latent_dim)``.
        rewards: Finite per-step reward predictions of shape ``(step_count,)``.
        continuations: Finite per-step continuation probabilities of shape ``(step_count,)``, each
            strictly inside ``(0, 1)``.
        provenance: Rollout provenance including the dynamics version, evidence boundary, clean-room
            route, consumed adapter observation contract, config, and step count.
    """

    latents: np.ndarray
    rewards: np.ndarray
    continuations: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the aligned, finite, episode-major rollout contract."""
        latents = _require_finite_float_array(self.latents, "latents", ndim=2)
        rewards = _require_finite_float_array(self.rewards, "rewards", ndim=1)
        continuations = _require_finite_float_array(self.continuations, "continuations", ndim=1)
        step_count = rewards.shape[0]
        if continuations.shape[0] != step_count:
            raise OpenDreamerDynamicsError(
                "rewards and continuations must have the same step count, "
                f"got {rewards.shape[0]} and {continuations.shape[0]}"
            )
        if latents.shape[0] != step_count + 1:
            raise OpenDreamerDynamicsError(
                "latents must have shape (step_count + 1, latent_dim); "
                f"expected {step_count + 1} rows, got {latents.shape[0]}"
            )
        _require_continuations_in_unit_interval(continuations)
        if not isinstance(self.provenance, Mapping):
            raise OpenDreamerDynamicsError(
                f"provenance must be a mapping, got {type(self.provenance).__name__}"
            )
        latents = np.array(latents, copy=True)
        rewards = np.array(rewards, copy=True)
        continuations = np.array(continuations, copy=True)
        latents.setflags(write=False)
        rewards.setflags(write=False)
        continuations.setflags(write=False)
        object.__setattr__(self, "latents", latents)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "continuations", continuations)

    @property
    def step_count(self) -> int:
        """Return the number of imagined transitions (action steps).

        Returns:
            The imagined transition count.
        """
        return int(self.rewards.shape[0])

    @property
    def latent_dim(self) -> int:
        """Return the latent-state width of the rollout.

        Returns:
            The latent-state width.
        """
        return int(self.latents.shape[1])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the rollout (arrays as nested lists).

        Returns:
            A dictionary with the latent trajectory, reward and continuation predictions, step
            count, and provenance.
        """
        return {
            "step_count": self.step_count,
            "latent_dim": self.latent_dim,
            "latents": self.latents.tolist(),
            "rewards": self.rewards.tolist(),
            "continuations": self.continuations.tolist(),
            "provenance": dict(self.provenance),
        }


class LatentDynamicsModel:
    """Clean-room, compute-free action-conditioned latent dynamics model with two heads.

    The model pairs a :class:`DynamicsConfig` with a consistent :class:`DynamicsWeights` bundle and
    exposes the encoder, the action-conditioned transition, the reward head, the continuation head,
    and an episode-major :meth:`imagine` entry point that consumes a merged-adapter
    :class:`~robot_sf.research.open_dreamer_adapter.StructuredEpisode` read-only. All outputs are
    finite-validated; any contract violation raises :class:`OpenDreamerDynamicsError`.
    """

    def __init__(self, config: DynamicsConfig, weights: DynamicsWeights) -> None:
        """Bind a config to a weight bundle after verifying their shapes agree.

        Args:
            config: Shape configuration for the model.
            weights: Frozen weight bundle whose implied dimensions must match ``config``.

        Raises:
            OpenDreamerDynamicsError: If either argument has the wrong type or the weight bundle's
                implied dimensions disagree with ``config``.
        """
        if not isinstance(config, DynamicsConfig):
            raise OpenDreamerDynamicsError(
                f"config must be a DynamicsConfig, got {type(config).__name__}"
            )
        if not isinstance(weights, DynamicsWeights):
            raise OpenDreamerDynamicsError(
                f"weights must be a DynamicsWeights, got {type(weights).__name__}"
            )
        if weights.obs_dim != config.obs_dim:
            raise OpenDreamerDynamicsError(
                f"weights obs_dim {weights.obs_dim} does not match config obs_dim {config.obs_dim}"
            )
        if weights.action_dim != config.action_dim:
            raise OpenDreamerDynamicsError(
                f"weights action_dim {weights.action_dim} does not match config "
                f"action_dim {config.action_dim}"
            )
        if weights.latent_dim != config.latent_dim:
            raise OpenDreamerDynamicsError(
                f"weights latent_dim {weights.latent_dim} does not match config "
                f"latent_dim {config.latent_dim}"
            )
        self._config = config
        self._weights = weights

    @property
    def config(self) -> DynamicsConfig:
        """Return the model's shape configuration.

        Returns:
            The bound :class:`DynamicsConfig`.
        """
        return self._config

    @property
    def weights(self) -> DynamicsWeights:
        """Return the model's frozen weight bundle.

        Returns:
            The bound :class:`DynamicsWeights`.
        """
        return self._weights

    @classmethod
    def from_config(cls, config: DynamicsConfig) -> LatentDynamicsModel:
        """Build a model with deterministic untrained weights from a shape configuration.

        Args:
            config: Shape configuration fixing the observation, action, and latent widths and the
                initialization seed.

        Returns:
            A :class:`LatentDynamicsModel` with weights from :meth:`DynamicsWeights.from_config`.

        Raises:
            OpenDreamerDynamicsError: If the config is invalid or the derived weights are
                non-finite.
        """
        if not isinstance(config, DynamicsConfig):
            raise OpenDreamerDynamicsError(
                f"config must be a DynamicsConfig, got {type(config).__name__}"
            )
        return cls(config, DynamicsWeights.from_config(config))

    @staticmethod
    def observation_width(episode: StructuredEpisode) -> int:
        """Return the structured observation vector width for an adapted episode.

        The width is ``len(DRIVE_STATE_LAYOUT)`` when rays are unavailable, or that plus the
        episode-wide ray-vector width when rays are available. The adapter guarantees ray
        availability is episode-wide with one fixed width, so the first step's ray width is
        authoritative.

        Args:
            episode: A merged-adapter structured episode (consumed read-only).

        Returns:
            The observation vector width the encoder must consume for this episode.

        Raises:
            OpenDreamerDynamicsError: If the episode is not a structured episode or its ray width is
                inconsistent with availability.
        """
        _require_structured_episode(episode)
        base_width = len(DRIVE_STATE_LAYOUT)
        if not episode.rays_available:
            return base_width
        if episode.step_count < 1:
            raise OpenDreamerDynamicsError(
                "a rays-available episode must contain at least one step to read the ray width"
            )
        ray_width = episode.observations[0].rays.size
        if ray_width <= 0:
            raise OpenDreamerDynamicsError(
                "a rays-available episode must expose a positive ray-vector width"
            )
        return base_width + int(ray_width)

    @classmethod
    def from_episode(
        cls,
        episode: StructuredEpisode,
        *,
        latent_dim: int = DEFAULT_LATENT_DIM,
        seed: int = 0,
    ) -> LatentDynamicsModel:
        """Build a model whose observation width matches an adapted episode's structured view.

        The observation width is derived from the episode via :meth:`observation_width`; the action
        width is :data:`ACTION_DIM`. Weights are the deterministic untrained initialization for the
        supplied ``latent_dim`` and ``seed``.

        Args:
            episode: A merged-adapter structured episode (consumed read-only).
            latent_dim: Latent-state width for the model. Must be a positive integer.
            seed: Deterministic initialization seed. Must be a non-boolean integer.

        Returns:
            A :class:`LatentDynamicsModel` configured to consume this episode's observation view.

        Raises:
            OpenDreamerDynamicsError: If the episode is not a structured episode or any dimension is
                invalid.
        """
        _require_structured_episode(episode)
        config = DynamicsConfig(
            obs_dim=cls.observation_width(episode),
            action_dim=ACTION_DIM,
            latent_dim=latent_dim,
            seed=seed,
        )
        return cls.from_config(config)

    def encode(self, observation: Sequence[float] | np.ndarray) -> np.ndarray:
        """Encode one structured observation vector into a bounded latent state.

        Args:
            observation: A finite float vector of width ``config.obs_dim`` (the adapter's
                ``drive_state`` group, concatenated with ``rays`` when rays are available).

        Returns:
            A finite latent vector of width ``config.latent_dim``, strictly inside ``(-1, 1)`` per
            component.

        Raises:
            OpenDreamerDynamicsError: If the observation is not a finite float vector of the
                configured width, or the produced latent is non-finite.
        """
        obs = _as_finite_vector(observation, "observation", self._config.obs_dim)
        latent = np.tanh(self._weights.w_enc @ obs + self._weights.b_enc)
        return _finish_latent(latent, "encoded latent")

    def reward_head(self, latent: Sequence[float] | np.ndarray) -> float:
        """Read the finite scalar reward head from a latent state.

        Args:
            latent: A finite float vector of width ``config.latent_dim``.

        Returns:
            The finite scalar reward prediction.

        Raises:
            OpenDreamerDynamicsError: If the latent is not a finite float vector of the configured
                width, or the produced reward is non-finite.
        """
        latent_vector = _as_finite_vector(latent, "latent", self._config.latent_dim)
        reward = float(self._weights.w_reward @ latent_vector + self._weights.b_reward)
        if not np.isfinite(reward):
            raise OpenDreamerDynamicsError("reward head produced a non-finite value")
        return reward

    def continuation_head(self, latent: Sequence[float] | np.ndarray) -> float:
        """Read the finite continuation probability head from a latent state.

        Args:
            latent: A finite float vector of width ``config.latent_dim``.

        Returns:
            The continuation probability, strictly inside ``(0, 1)``.

        Raises:
            OpenDreamerDynamicsError: If the latent is not a finite float vector of the configured
                width, or the produced continuation is non-finite or outside ``(0, 1)``.
        """
        latent_vector = _as_finite_vector(latent, "latent", self._config.latent_dim)
        logit = float(self._weights.w_cont @ latent_vector + self._weights.b_cont)
        if not np.isfinite(logit):
            raise OpenDreamerDynamicsError("continuation head produced a non-finite logit")
        continuation = _stable_sigmoid(logit)
        if not 0.0 < continuation < 1.0:
            raise OpenDreamerDynamicsError(
                f"continuation head must lie strictly inside (0, 1), got {continuation!r}"
            )
        return continuation

    def step(
        self,
        latent: Sequence[float] | np.ndarray,
        action: Sequence[float] | np.ndarray,
    ) -> DynamicsStep:
        """Apply one action-conditioned latent transition and read both heads.

        The transition is ``latent_{t+1} = tanh(W_z latent_t + W_a action_t + b)``; the reward and
        continuation heads are then read from the predicted next latent state.

        Args:
            latent: The current finite latent state, width ``config.latent_dim``.
            action: A finite bounded action vector, width ``config.action_dim``.

        Returns:
            A :class:`DynamicsStep` with the finite next latent state and both head predictions.

        Raises:
            OpenDreamerDynamicsError: If either input is mis-shaped or non-finite, or any produced
                output is non-finite or the continuation is outside ``(0, 1)``.
        """
        latent_vector = _as_finite_vector(latent, "latent", self._config.latent_dim)
        action_vector = _as_finite_vector(action, "action", self._config.action_dim)
        pre_activation = (
            self._weights.w_latent @ latent_vector
            + self._weights.w_action @ action_vector
            + self._weights.b_latent
        )
        next_latent = np.tanh(pre_activation)
        next_latent = _finish_latent(next_latent, "next latent")
        reward = self.reward_head(next_latent)
        continuation = self.continuation_head(next_latent)
        return DynamicsStep(latent=next_latent, reward=reward, continuation=continuation)

    def imagine(self, episode: StructuredEpisode) -> LatentRollout:
        """Imagine an open-loop latent rollout over one adapted episode's action sequence.

        The latent state is bootstrapped by encoding the episode's first structured observation,
        then the dynamics is rolled forward over the episode's bounded action sequence. The episode
        is consumed **read-only** and stays episode-major: a single episode is imagined and episode
        boundaries are never crossed. Subsequent observations are validated finite but are not
        consumed by the prior rollout (a trained posterior/encoder would use them; that is out of
        scope for this compute-free slice).

        Args:
            episode: A merged-adapter structured episode with at least one step.

        Returns:
            A finite :class:`LatentRollout` with ``step_count`` imagined transitions.

        Raises:
            OpenDreamerDynamicsError: If the episode is not a structured episode, has no steps, its
                observation width does not match the model, or any produced value is non-finite.
        """
        _require_structured_episode(episode)
        if episode.step_count < 1:
            raise OpenDreamerDynamicsError("episode must contain at least one step to imagine")
        episode_width = self.observation_width(episode)
        if episode_width != self._config.obs_dim:
            raise OpenDreamerDynamicsError(
                f"episode observation width {episode_width} does not match model "
                f"obs_dim {self._config.obs_dim}"
            )
        observations = _episode_observation_array(episode, self._config.obs_dim)
        actions = _episode_action_array(episode, self._config.action_dim)
        return self._imagine_aligned(observations, actions, episode=episode)

    def imagine_from_arrays(
        self,
        observations: Sequence[Sequence[float]] | np.ndarray,
        actions: Sequence[Sequence[float]] | np.ndarray,
    ) -> LatentRollout:
        """Imagine an open-loop rollout from aligned observation and action arrays.

        This is the array-level entry point underlying :meth:`imagine`. The latent state is
        bootstrapped from ``observations[0]`` and rolled forward over ``actions``. The observation
        and action sequences must be aligned (equal length) and contain at least one step, matching
        the episode-major invariant.

        Args:
            observations: Finite float array of shape ``(step_count, config.obs_dim)``.
            actions: Finite float array of shape ``(step_count, config.action_dim)``.

        Returns:
            A finite :class:`LatentRollout` with ``step_count`` imagined transitions.

        Raises:
            OpenDreamerDynamicsError: If the arrays are mis-shaped, mis-aligned, empty, non-finite,
                or any produced value is non-finite.
        """
        obs_array = _as_finite_matrix(observations, "observations", self._config.obs_dim)
        action_array = _as_finite_matrix(actions, "actions", self._config.action_dim)
        if obs_array.shape[0] != action_array.shape[0]:
            raise OpenDreamerDynamicsError(
                "observations and actions must be aligned (equal step count), "
                f"got {obs_array.shape[0]} and {action_array.shape[0]}"
            )
        if obs_array.shape[0] < 1:
            raise OpenDreamerDynamicsError(
                "observations and actions must contain at least one step"
            )
        return self._imagine_aligned(obs_array, action_array, episode=None)

    def _imagine_aligned(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        episode: StructuredEpisode | None,
    ) -> LatentRollout:
        """Roll the dynamics forward over aligned finite arrays and assemble a validated rollout.

        Args:
            observations: Finite float array of shape ``(step_count, obs_dim)``.
            actions: Finite float array of shape ``(step_count, action_dim)``.
            episode: The source structured episode when imagined via :meth:`imagine`, used only to
                enrich provenance; ``None`` for the array-level entry point.

        Returns:
            A finite :class:`LatentRollout`.

        Raises:
            OpenDreamerDynamicsError: If any produced latent, reward, or continuation is non-finite
                or a continuation is outside ``(0, 1)``.
        """
        step_count = actions.shape[0]
        latent_dim = self._config.latent_dim
        latent_trajectory = np.empty((step_count + 1, latent_dim), dtype=float)
        rewards = np.empty(step_count, dtype=float)
        continuations = np.empty(step_count, dtype=float)
        latent_trajectory[0] = self.encode(observations[0])
        for step_index in range(step_count):
            transition = self.step(latent_trajectory[step_index], actions[step_index])
            latent_trajectory[step_index + 1] = transition.latent
            rewards[step_index] = transition.reward
            continuations[step_index] = transition.continuation
        provenance = self._rollout_provenance(step_count, episode=episode)
        return LatentRollout(
            latents=latent_trajectory,
            rewards=rewards,
            continuations=continuations,
            provenance=provenance,
        )

    def _rollout_provenance(
        self,
        step_count: int,
        *,
        episode: StructuredEpisode | None,
    ) -> dict[str, Any]:
        """Build the JSON-safe provenance recorded on an imagined rollout.

        Args:
            step_count: Number of imagined transitions.
            episode: The source structured episode when available, used to record the consumed
                episode identity and adapter observation contract.

        Returns:
            A provenance mapping recording the dynamics version, evidence boundary, clean-room
            route, config, and (when available) the consumed episode identity.
        """
        provenance: dict[str, Any] = {
            "dynamics_version": OPEN_DREAMER_DYNAMICS_VERSION,
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "route": "clean_room",
            "trained": False,
            "compute_free": True,
            "consumed_observation_contract": OPEN_DREAMER_OBSERVATION_CONTRACT,
            "config": self._config.to_dict(),
            "step_count": int(step_count),
        }
        if episode is not None:
            provenance["episode_id"] = episode.episode_id
            provenance["scenario_id"] = episode.scenario_id
            provenance["seed"] = int(episode.seed)
            provenance["rays_available"] = bool(episode.rays_available)
        return provenance


def _require_structured_episode(episode: Any) -> StructuredEpisode:
    """Require a merged-adapter structured episode before reading its fields.

    Args:
        episode: Candidate episode value.

    Returns:
        The validated structured episode.

    Raises:
        OpenDreamerDynamicsError: If the value is not a structured episode.
    """
    if not isinstance(episode, StructuredEpisode):
        raise OpenDreamerDynamicsError(
            f"episode must be a StructuredEpisode from the merged adapter, "
            f"got {type(episode).__name__}"
        )
    return episode


def _as_finite_vector(value: Any, name: str, expected_width: int) -> np.ndarray:
    """Coerce a sequence to a finite 1D float vector of an exact width.

    Args:
        value: Candidate vector value.
        name: Vector name used in the error message.
        expected_width: Exact width the vector must have.

    Returns:
        A finite float ndarray of shape ``(expected_width,)``.

    Raises:
        OpenDreamerDynamicsError: If the value is not a numeric sequence of the expected width or
            contains any non-finite value.
    """
    if isinstance(value, str | bytes | Mapping):
        raise OpenDreamerDynamicsError(
            f"{name} must be a numeric vector, got {type(value).__name__}"
        )
    try:
        array = np.asarray(value, dtype=float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OpenDreamerDynamicsError(f"{name} must be a finite numeric vector") from exc
    if array.shape != (expected_width,):
        raise OpenDreamerDynamicsError(
            f"{name} must have shape ({expected_width},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise OpenDreamerDynamicsError(f"{name} must contain only finite values")
    return array


def _as_finite_matrix(value: Any, name: str, expected_width: int) -> np.ndarray:
    """Coerce a nested sequence to a finite 2D float matrix with an exact column width.

    Args:
        value: Candidate matrix value.
        name: Matrix name used in the error message.
        expected_width: Exact column width the matrix must have.

    Returns:
        A finite float ndarray of shape ``(step_count, expected_width)``.

    Raises:
        OpenDreamerDynamicsError: If the value is not a 2D numeric matrix with the expected column
            width or contains any non-finite value.
    """
    if isinstance(value, str | bytes | Mapping):
        raise OpenDreamerDynamicsError(
            f"{name} must be a numeric matrix, got {type(value).__name__}"
        )
    try:
        array = np.asarray(value, dtype=float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OpenDreamerDynamicsError(f"{name} must be a finite numeric matrix") from exc
    if array.ndim != 2 or array.shape[1] != expected_width:
        raise OpenDreamerDynamicsError(
            f"{name} must have shape (step_count, {expected_width}), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise OpenDreamerDynamicsError(f"{name} must contain only finite values")
    return array


def _finish_latent(latent: np.ndarray, name: str) -> np.ndarray:
    """Validate a produced latent vector is finite and return it frozen read-only.

    Args:
        latent: Candidate produced latent vector.
        name: Latent name used in the error message.

    Returns:
        A read-only finite float ndarray.

    Raises:
        OpenDreamerDynamicsError: If the produced latent is non-finite.
    """
    if latent.dtype.kind != "f" or not np.all(np.isfinite(latent)):
        raise OpenDreamerDynamicsError(f"{name} produced a non-finite value")
    frozen = np.array(latent, copy=True)
    frozen.setflags(write=False)
    return frozen


def _episode_observation_array(episode: StructuredEpisode, obs_dim: int) -> np.ndarray:
    """Stack an episode's structured observation groups into one finite matrix.

    Args:
        episode: A merged-adapter structured episode (consumed read-only).
        obs_dim: Expected observation vector width.

    Returns:
        A finite float ndarray of shape ``(step_count, obs_dim)``.

    Raises:
        OpenDreamerDynamicsError: If any step's concatenated observation is non-finite or has the
            wrong width.
    """
    rows: list[np.ndarray] = []
    for step_index, step in enumerate(episode.observations):
        if step.rays_available:
            row = np.concatenate([step.drive_state, step.rays])
        else:
            row = np.asarray(step.drive_state, dtype=float)
        if row.shape != (obs_dim,):
            raise OpenDreamerDynamicsError(
                f"observation at step {step_index} must have width {obs_dim}, got {row.shape}"
            )
        if not np.all(np.isfinite(row)):
            raise OpenDreamerDynamicsError(
                f"observation at step {step_index} must contain only finite values"
            )
        rows.append(row)
    return np.stack(rows, axis=0)


def _episode_action_array(episode: StructuredEpisode, action_dim: int) -> np.ndarray:
    """Stack an episode's bounded action view into one finite matrix.

    Args:
        episode: A merged-adapter structured episode (consumed read-only).
        action_dim: Expected action vector width.

    Returns:
        A finite float ndarray of shape ``(step_count, action_dim)``.

    Raises:
        OpenDreamerDynamicsError: If any step's action is non-finite or has the wrong width.
    """
    rows: list[np.ndarray] = []
    for step_index, step in enumerate(episode.actions):
        row = np.asarray(step.raw, dtype=float)
        if row.shape != (action_dim,):
            raise OpenDreamerDynamicsError(
                f"action at step {step_index} must have width {action_dim}, got {row.shape}"
            )
        if not np.all(np.isfinite(row)):
            raise OpenDreamerDynamicsError(
                f"action at step {step_index} must contain only finite values"
            )
        rows.append(row)
    return np.stack(rows, axis=0)


__all__ = [
    "ACTION_DIM",
    "DEFAULT_LATENT_DIM",
    "DYNAMICS_PROVENANCE_KEY",
    "EVIDENCE_BOUNDARY",
    "OPEN_DREAMER_DYNAMICS_VERSION",
    "DynamicsConfig",
    "DynamicsStep",
    "DynamicsWeights",
    "LatentDynamicsModel",
    "LatentRollout",
    "OpenDreamerDynamicsError",
]
