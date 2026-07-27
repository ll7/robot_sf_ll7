"""Clean-room episode-major adapter for Open Dreamer-style structured observations (issue #6318 Step 2).

This module implements **Step 2 (adapter contract smoke, compute-free)** of parent issue #6318's
maintainer-authorized sequenced plan. Step 1 (Gate 0, the license/architecture note merged in PR
#6322) established that the upstream Open Dreamer ``LICENSE`` at the pinned commit reserves all
rights, so the only permissible route is a **clean-room** adapter derived from the public paper and
the Robot SF trajectory contract -- no upstream source is copied, vendored, adapted line-by-line, or
committed anywhere here.

What this module does
---------------------

It *consumes* the existing, stable ``RLTrajectoryDataset.v1`` / ``RLTrajectoryEpisode.v1`` contract
by importing it **read-only** from :mod:`robot_sf.benchmark.rl_trajectory_dataset` (that benchmark
module is never edited by this slice) and produces a leakage-safe, episode-major
**structured-observation view** with two named groups plus a bounded action mapping:

* ``drive_state`` -- a finite float vector ``[x, y, heading, vx, vy]`` derived from each step's
  ``robot_states`` entry (the robot's own drive/kinematic state). This is the structured replacement
  for a flat scalar observation and is the group a future group-aware encoder would consume.
* ``rays`` -- a finite float vector of range/lidar readings derived from each step's ``observations``
  entry when it exposes a recognized ray-like key. When the source observation carries no ray-like
  field (the current ``RLTrajectoryDataset.v1`` recorder does not record lidar rays), the group is
  honestly reported as **unavailable** (empty vector, ``available=False``). Ray availability is an
  episode-wide contract: all steps must expose a ray-like field with one common vector width, or no
  step may expose one. The adapter rejects partial or width-heterogeneous ray records rather than
  padding or fabricating data. It never fabricates ray distances from pedestrian positions -- doing
  so would invent new contract semantics.
* ``[-1, 1] -> (linear velocity, angular velocity)`` action mapping -- the clean-room replacement
  for upstream's VPT-style action container. The mapping is parameterized by caller-supplied speed
  bounds (:class:`ActionBounds`) so the adapter never hardcodes or assumes a particular robot
  drivetrain; the bounds used are recorded in the structured episode's provenance.

Despite sharing group labels with the environment, this is an adapter-specific raw-recording
contract, not ``ObservationMode.DEFAULT_GYM``: its ``drive_state`` layout is
``[x, y, heading, vx, vy]`` rather than the native sensor-fusion layout, and neither group is
implicitly normalized. :data:`OPEN_DREAMER_OBSERVATION_CONTRACT` and the provenance fields make
that boundary explicit for later model consumers.

What this module does NOT do (out of scope -- Steps 3-4 on parent #6318)
-----------------------------------------------------------------------

It does **not** implement a dynamics model, reward/continuation heads, an imagined-replay bridge, or
any SAC integration. It does not flatten episodes into transition arrays -- the view stays
**episode-major** by construction so episode boundaries (and therefore terminal/truncated semantics)
cannot be silently crossed. ``evidence_tier`` stays :data:`EVIDENCE_BOUNDARY` (``idea``): a
successful adapter smoke is **diagnostic/contract evidence only**, not a benchmark, metric, or
paper-facing claim.

Fail-closed contract
--------------------

The adapter fails closed (raises :class:`OpenDreamerAdapterError`) when:

* a required field is missing (e.g. ``robot_states`` lacks ``position``/``heading``/``velocity``);
* any produced output is non-finite or physically invalid (NaN/Inf in ``drive_state``, ``rays``,
  ``rewards``, ``return_to_go``, or the mapped velocity; negative range readings);
* ray-like observations appear for only some episode steps, or their vector lengths differ across
  steps (the structured sequence view requires one fixed ray width when rays are available);
* a stored action is not a finite 2D continuous ``(linear, angular)`` command, does not use the
  recorder's canonical ``linear_velocity``/``angular_velocity`` mapping, or falls outside the
  caller-supplied physical action bounds (incompatible action space);
* metadata or terminal flags would require coercion, or source provenance already owns the
  adapter's reserved provenance key;
* a stored split does not equal the canonical deterministic assignment for its
  ``(scenario_id, seed)`` key, or the dataset leaks either a scenario or scenario/seed key across
  more than one split.

Split policy is the canonical :func:`assign_deterministic_split` from the benchmark contract;
:func:`adapt_episode` enforces it and :func:`validate_split_leakage` verifies the source manifest's
scenario-level and scenario-seed split ownership rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

# Read-only consumption of the stable benchmark contract. This slice must NOT edit
# robot_sf/benchmark/rl_trajectory_dataset.py; it imports the v1 symbols as an immutable source.
from robot_sf.benchmark.rl_trajectory_dataset import (
    RL_TRAJECTORY_EPISODE_SCHEMA_VERSION,
    SPLIT_NAMES,
    RLTrajectoryEpisode,
    assign_deterministic_split,
    validate_rl_trajectory_episode,
)

#: Adapter contract version. Bumped only if the structured-observation view changes shape.
OPEN_DREAMER_ADAPTER_VERSION = "open_dreamer_adapter.v1"

#: Evidence tier for this slice. Diagnostic/contract evidence only -- never a benchmark claim.
EVIDENCE_BOUNDARY = "idea"

#: Explicit version for the adapter's raw, structured-observation view. This prevents consumers
#: from confusing its ``drive_state`` group with the differently shaped and normalized native
#: ``ObservationMode.DEFAULT_GYM`` group.
OPEN_DREAMER_OBSERVATION_CONTRACT = "open_dreamer_adapter.raw_structured_observation.v1"

#: Values remain in the source recorder's units; later model consumers must declare normalization.
OBSERVATION_NORMALIZATION = "raw_recorder_values_unscaled"

#: Provenance key under which the adapter records its own metadata on each structured episode.
ADAPTER_PROVENANCE_KEY = "open_dreamer_adapter"

#: Fixed component order of the ``drive_state`` group, derived from ``robot_states``.
DRIVE_STATE_LAYOUT: tuple[str, ...] = ("x", "y", "heading", "vx", "vy")

#: Recognized observation keys that populate the ``rays`` group, in priority order.
#: The current ``RLTrajectoryDataset.v1`` recorder does not emit any of these; the adapter reports
#: ``rays_available=False`` for such episodes rather than fabricating ranges from pedestrians.
RAY_OBSERVATION_KEYS: tuple[str, ...] = (
    "rays",
    "lidar_rays",
    "lidar_ranges",
    "ray_distances",
    "ranges",
    "laser_scan",
    "laser",
    "lidar",
)

#: Expected dimensionality of the stored continuous ``(linear, angular)`` action command.
EXPECTED_ACTION_DIM = 2


class OpenDreamerAdapterError(ValueError):
    """Raised when the Open Dreamer adapter cannot preserve the v1 contract without weakening it.

    Every raise is a fail-closed contract boundary: a missing field, a non-finite produced value,
    an incompatible action space, or scenario/seed split leakage. The adapter never silently drops
    or relaxes a required semantic; callers must treat this exception as "blocked", not as a
    recoverable skip.
    """


def _finite_action_bound(value: Any, field_name: str) -> float:
    """Return one finite real-valued action bound, rejecting booleans and coercible strings."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OpenDreamerAdapterError(f"{field_name} must be a finite real number, got {value!r}")
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OpenDreamerAdapterError(f"{field_name} must be finite, got {value!r}") from exc
    if not np.isfinite(numeric_value):
        raise OpenDreamerAdapterError(f"{field_name} must be finite, got {value!r}")
    return numeric_value


@dataclass(frozen=True, slots=True)
class ActionBounds:
    """Speed bounds for the bounded ``[-1, 1] -> (linear, angular)`` velocity mapping.

    The mapping is parameterized by caller-supplied bounds so the adapter never assumes a particular
    robot drivetrain. Bounds are recorded in each structured episode's provenance under
    :data:`ADAPTER_PROVENANCE_KEY`.

    Attributes:
        max_linear_speed: Maximum forward linear velocity (m/s). Must be strictly positive.
        max_angular_speed: Maximum angular velocity magnitude (rad/s). Must be strictly positive.
        min_linear_speed: Minimum linear velocity (m/s). ``0.0`` means forward-only; a negative
            value enables backwards motion. Must satisfy
            ``-max_linear_speed <= min_linear_speed < max_linear_speed``.
    """

    max_linear_speed: float
    max_angular_speed: float
    min_linear_speed: float = 0.0

    def __post_init__(self) -> None:
        """Validate that the speed bounds define a non-degenerate, finite velocity envelope."""
        max_linear_speed = _finite_action_bound(self.max_linear_speed, "max_linear_speed")
        max_angular_speed = _finite_action_bound(self.max_angular_speed, "max_angular_speed")
        min_linear_speed = _finite_action_bound(self.min_linear_speed, "min_linear_speed")
        if max_linear_speed <= 0.0:
            raise OpenDreamerAdapterError(
                f"max_linear_speed must be positive and finite, got {self.max_linear_speed!r}"
            )
        if max_angular_speed <= 0.0:
            raise OpenDreamerAdapterError(
                f"max_angular_speed must be positive and finite, got {self.max_angular_speed!r}"
            )
        if min_linear_speed < -max_linear_speed:
            raise OpenDreamerAdapterError(
                f"min_linear_speed ({self.min_linear_speed}) must be >= "
                f"-max_linear_speed (-{self.max_linear_speed})"
            )
        if min_linear_speed >= max_linear_speed:
            raise OpenDreamerAdapterError(
                f"min_linear_speed ({self.min_linear_speed}) must be strictly less than "
                f"max_linear_speed ({self.max_linear_speed})"
            )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-safe representation of the speed bounds.

        Returns:
            A mapping with ``max_linear_speed``, ``min_linear_speed``, and ``max_angular_speed``.
        """
        return {
            "max_linear_speed": float(self.max_linear_speed),
            "min_linear_speed": float(self.min_linear_speed),
            "max_angular_speed": float(self.max_angular_speed),
        }


def _require_action_bounds(bounds: Any) -> ActionBounds:
    """Return validated action bounds without leaking an implementation-level attribute error."""
    if not isinstance(bounds, ActionBounds):
        raise OpenDreamerAdapterError(
            f"action_bounds must be an ActionBounds instance, got {type(bounds).__name__}"
        )
    return bounds


@dataclass(frozen=True, slots=True)
class StructuredObservationStep:
    """One step of the leakage-safe structured-observation view.

    Attributes:
        drive_state: Finite float vector in :data:`DRIVE_STATE_LAYOUT` order, derived from the
            step's ``robot_states`` entry. Always length ``len(DRIVE_STATE_LAYOUT)``.
        rays: Finite float vector of range readings when :attr:`rays_available` is true, else an
            empty vector. Never contains non-finite values.
        rays_available: Whether the source observation exposed a recognized ray-like field. False
            for the current ``RLTrajectoryDataset.v1`` recorder, which does not record lidar rays.
            :func:`adapt_episode` additionally enforces episode-wide availability and one shared
            ray-vector width before returning a structured episode.
    """

    drive_state: np.ndarray
    rays: np.ndarray
    rays_available: bool

    def __post_init__(self) -> None:
        """Validate the finite-output contract for both groups."""
        if not isinstance(self.rays_available, bool):
            raise OpenDreamerAdapterError(
                f"rays_available must be a boolean, got {self.rays_available!r}"
            )
        if not isinstance(self.drive_state, np.ndarray):
            raise OpenDreamerAdapterError("drive_state must be a 1D float ndarray")
        if not isinstance(self.rays, np.ndarray):
            raise OpenDreamerAdapterError("rays must be a 1D float ndarray")
        if self.drive_state.ndim != 1 or self.drive_state.shape[0] != len(DRIVE_STATE_LAYOUT):
            raise OpenDreamerAdapterError(
                "drive_state must be a 1D vector of length "
                f"{len(DRIVE_STATE_LAYOUT)}, got shape {self.drive_state.shape}"
            )
        if self.drive_state.dtype.kind != "f" or not np.all(np.isfinite(self.drive_state)):
            raise OpenDreamerAdapterError("drive_state must be finite floats")
        _validate_rays_group(self.rays, self.rays_available)

        drive_state = np.array(self.drive_state, copy=True)
        rays = np.array(self.rays, copy=True)
        drive_state.setflags(write=False)
        rays.setflags(write=False)
        object.__setattr__(self, "drive_state", drive_state)
        object.__setattr__(self, "rays", rays)


def _validate_rays_group(rays: np.ndarray, rays_available: bool) -> None:
    """Validate one structured ray group, including its physical availability contract."""
    if rays.ndim != 1 or rays.dtype.kind != "f":
        raise OpenDreamerAdapterError("rays must be a 1D float vector")
    if not np.all(np.isfinite(rays)):
        raise OpenDreamerAdapterError("rays must be finite floats")
    if np.any(rays < 0.0):
        raise OpenDreamerAdapterError("rays must contain non-negative ranges")
    if not rays_available and rays.size != 0:
        raise OpenDreamerAdapterError("rays_available=False requires an empty rays vector")
    if rays_available and rays.size == 0:
        raise OpenDreamerAdapterError("rays_available=True requires at least one ray value")


@dataclass(frozen=True, slots=True)
class StructuredActionStep:
    """One step of the preserved action view.

    The recorder's canonical mapping and legacy two-element sequences are normalized to the same
    physical ``(linear, angular)`` command, then validated against the caller-supplied bounds. The
    forward ``[-1, 1] -> velocity`` mapping is provided separately by :func:`map_action_to_velocity`;
    the bounds used are recorded in the episode provenance.

    Attributes:
        raw: The normalized physical ``(linear, angular)`` action, finite- and bounds-validated.
    """

    raw: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the stored action as a finite 2D continuous command."""
        if isinstance(self.raw, str | bytes) or not isinstance(self.raw, Sequence | np.ndarray):
            raise OpenDreamerAdapterError(
                "stored action must be a sequence of two finite numeric values"
            )
        try:
            raw_values = tuple(self.raw)
        except TypeError as exc:
            raise OpenDreamerAdapterError(
                "stored action must be a sequence of two finite numeric values"
            ) from exc
        if len(raw_values) != EXPECTED_ACTION_DIM:
            raise OpenDreamerAdapterError(
                f"stored action must be {EXPECTED_ACTION_DIM}D (linear, angular), "
                f"got {len(raw_values)}D"
            )
        normalized: list[float] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise OpenDreamerAdapterError(f"stored action must be numeric, got {value!r}")
            try:
                numeric_value = float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise OpenDreamerAdapterError(
                    f"stored action must be finite, got {value!r}"
                ) from exc
            if not np.isfinite(numeric_value):
                raise OpenDreamerAdapterError(f"stored action must be finite, got {value!r}")
            normalized.append(numeric_value)
        object.__setattr__(self, "raw", tuple(normalized))


@dataclass(frozen=True, slots=True)
class StructuredEpisode:
    """Episode-major structured-observation + action view with full v1 provenance preserved.

    The view is **episode-major**: per-step fields are aligned tuples whose lengths equal the
    episode step count. Episodes are never flattened into transitions here, so terminal/truncated
    semantics cannot be silently crossed. Every v1 field is preserved verbatim in addition to the
    structured groups; ``raw_observations`` and ``raw_actions`` retain the original per-step
    payloads verbatim; ``provenance`` carries the original episode provenance plus an
    :data:`ADAPTER_PROVENANCE_KEY` entry recording the adapter version, bounds, and rays
    availability.

    Attributes:
        dataset_id: Preserved v1 dataset id.
        episode_id: Preserved v1 episode id.
        scenario_id: Preserved v1 scenario id.
        seed: Preserved v1 seed.
        source_policy_id: Preserved v1 source policy id.
        split: Preserved v1 split name (one of :data:`SPLIT_NAMES`).
        observations: Per-step structured-observation view (drive_state + rays).
        raw_observations: Original v1 per-step observation mappings, retained verbatim alongside
            the derived groups so unrecognized source fields remain recoverable.
        actions: Per-step preserved action view.
        raw_actions: Original v1 per-step action payloads, retained verbatim alongside the
            normalized physical action view so source representation remains auditable.
        rewards: Preserved v1 per-step rewards, finite-validated.
        return_to_go: Preserved v1 per-step return-to-go, finite-validated.
        terminated: Preserved v1 per-step terminated flags.
        truncated: Preserved v1 per-step truncated flags.
        pedestrians: Preserved v1 per-step pedestrian state.
        robot_states: Preserved v1 per-step robot state.
        provenance: Original episode provenance plus the adapter provenance entry.
        rays_available: True only when every step exposed a recognized ray-like field with one
            shared ray-vector width. False when no step exposed one; partial availability or mixed
            widths fail closed during adaptation.
        drive_state_layout: The fixed component order of :attr:`observations[i].drive_state`.
        observation_contract: Versioned adapter-specific observation contract. It intentionally
            differs from the native environment's ``ObservationMode.DEFAULT_GYM`` contract.
    """

    dataset_id: str
    episode_id: str
    scenario_id: str
    seed: int
    source_policy_id: str
    split: str
    observations: tuple[StructuredObservationStep, ...]
    raw_observations: tuple[Any, ...]
    actions: tuple[StructuredActionStep, ...]
    raw_actions: tuple[Any, ...]
    rewards: tuple[float, ...]
    return_to_go: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    pedestrians: tuple[Any, ...]
    robot_states: tuple[Any, ...]
    provenance: Mapping[str, Any]
    rays_available: bool
    drive_state_layout: tuple[str, ...] = field(default=DRIVE_STATE_LAYOUT)
    observation_contract: str = OPEN_DREAMER_OBSERVATION_CONTRACT

    @property
    def step_count(self) -> int:
        """Return the aligned per-step count without flattening the episode."""
        return len(self.rewards)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the structured episode (groups as lists).

        Returns:
            A dictionary with all preserved v1 fields, including raw observation/action payloads,
            and the structured-observation/action groups.
        """
        return {
            "schema_version": OPEN_DREAMER_ADAPTER_VERSION,
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "source_policy_id": self.source_policy_id,
            "split": self.split,
            "observation_contract": self.observation_contract,
            "observation_normalization": OBSERVATION_NORMALIZATION,
            "drive_state_layout": list(self.drive_state_layout),
            "rays_available": self.rays_available,
            "observations": [
                {
                    "drive_state": step.drive_state.tolist(),
                    "rays": step.rays.tolist(),
                    "rays_available": step.rays_available,
                }
                for step in self.observations
            ],
            "raw_observations": _json_safe_value(self.raw_observations, "raw_observations"),
            "actions": [list(step.raw) for step in self.actions],
            "raw_actions": _json_safe_value(self.raw_actions, "raw_actions"),
            "rewards": list(self.rewards),
            "return_to_go": list(self.return_to_go),
            "terminated": list(self.terminated),
            "truncated": list(self.truncated),
            "pedestrians": _json_safe_value(self.pedestrians, "pedestrians"),
            "robot_states": _json_safe_value(self.robot_states, "robot_states"),
            "provenance": _json_safe_value(self.provenance, "provenance"),
        }


@dataclass(frozen=True, slots=True)
class SplitLeakageReport:
    """Result of checking that no scenario or ``(scenario_id, seed)`` key leaks across splits.

    Attributes:
        ok: True when every scenario and scenario-seed key maps to exactly one split.
        split_scenario_ids: Mapping of split name to the sorted scenario ids it owns.
        split_scenario_seed_keys: Mapping of split name to the sorted scenario-seed keys it owns.
        leaked_scenario_ids: Scenario ids that appear in more than one split (empty when ``ok``).
        leaked_keys: Scenario-seed keys that appear in more than one split (empty when ``ok``).
    """

    ok: bool
    split_scenario_seed_keys: Mapping[str, tuple[str, ...]]
    leaked_keys: tuple[str, ...]
    split_scenario_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    leaked_scenario_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the leakage check.

        Returns:
            A dictionary with ``ok``, per-split scenario and scenario-seed keys, and any leaks.
        """
        return {
            "ok": bool(self.ok),
            "split_scenario_ids": {
                name: list(ids) for name, ids in self.split_scenario_ids.items()
            },
            "split_scenario_seed_keys": {
                name: list(keys) for name, keys in self.split_scenario_seed_keys.items()
            },
            "leaked_scenario_ids": list(self.leaked_scenario_ids),
            "leaked_keys": list(self.leaked_keys),
        }


def map_action_to_velocity(
    normalized: Sequence[float] | np.ndarray,
    bounds: ActionBounds,
) -> tuple[float, float]:
    """Map a bounded ``[-1, 1]`` action to physical ``(linear_velocity, angular_velocity)``.

    This is the clean-room replacement for upstream's VPT-style action container. Linear velocity
    uses an affine map from ``[-1, 1]`` onto ``[min_linear_speed, max_linear_speed]`` so forward-only
    drivetrains (``min_linear_speed=0``) are represented faithfully; angular velocity uses a
    symmetric map onto ``[-max_angular_speed, max_angular_speed]``. The domain is strictly bounded:
    inputs must lie in ``[-1, 1]`` (a tiny float tolerance absorbs roundoff); accepted roundoff is
    clamped to that interval before mapping, while genuinely out-of-domain inputs are rejected. This
    prevents tolerated roundoff from escaping the declared envelope. Both outputs are checked finite
    and against the supplied bounds.

    Args:
        normalized: A length-2 action with both components in ``[-1, 1]``.
        bounds: Speed bounds defining the velocity envelope.

    Returns:
        The mapped ``(linear_velocity, angular_velocity)``.

    Raises:
        OpenDreamerAdapterError: If the action is not length-2, is out of ``[-1, 1]``, is
            non-finite, or maps to a non-finite velocity.
    """
    bounds = _require_action_bounds(bounds)
    try:
        arr = np.asarray(normalized, dtype=float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OpenDreamerAdapterError(
            "normalized action must be a numeric length-2 sequence"
        ) from exc
    if arr.shape != (EXPECTED_ACTION_DIM,):
        raise OpenDreamerAdapterError(
            f"normalized action must have shape ({EXPECTED_ACTION_DIM},), got {arr.shape}"
        )
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in normalized):
        raise OpenDreamerAdapterError("normalized action must be a numeric length-2 sequence")
    if not np.all(np.isfinite(arr)):
        raise OpenDreamerAdapterError("normalized action must be finite")
    if np.any(arr < -1.0 - 1e-9) or np.any(arr > 1.0 + 1e-9):
        raise OpenDreamerAdapterError("normalized action must lie in [-1, 1]")
    arr = np.clip(arr, -1.0, 1.0)
    linear_range = bounds.max_linear_speed - bounds.min_linear_speed
    linear_velocity = bounds.min_linear_speed + (arr[0] + 1.0) * 0.5 * linear_range
    angular_velocity = arr[1] * bounds.max_angular_speed
    linear_velocity = float(linear_velocity)
    angular_velocity = float(angular_velocity)
    if not (np.isfinite(linear_velocity) and np.isfinite(angular_velocity)):
        raise OpenDreamerAdapterError("mapped velocity must be finite")
    if not (
        bounds.min_linear_speed <= linear_velocity <= bounds.max_linear_speed
        and -bounds.max_angular_speed <= angular_velocity <= bounds.max_angular_speed
    ):
        raise OpenDreamerAdapterError("mapped velocity must lie within the supplied action bounds")
    return linear_velocity, angular_velocity


def canonical_split(scenario_id: str, seed: int) -> str:
    """Return the canonical deterministic split for a ``(scenario_id, seed)`` key.

    This is a thin delegation to the benchmark contract's
    :func:`assign_deterministic_split`, re-exposed so consumers of the adapter have a single
    import surface and the split policy enforced by :func:`adapt_episode` is explicit.
    The source manifest additionally requires every scenario id to stay in one split, which
    :func:`validate_split_leakage` checks across an episode batch because this per-key assignment can
    otherwise place different seeds for one scenario in different splits.

    Args:
        scenario_id: Scenario id.
        seed: Integer seed.

    Returns:
        One of :data:`SPLIT_NAMES`.
    """
    if not isinstance(scenario_id, str) or not scenario_id:
        raise OpenDreamerAdapterError(
            f"scenario_id must be a non-empty string, got {scenario_id!r}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OpenDreamerAdapterError(f"seed must be a non-boolean integer, got {seed!r}")
    return assign_deterministic_split(scenario_id, seed)


def validate_split_leakage(
    episodes: Sequence[RLTrajectoryEpisode | StructuredEpisode],
) -> SplitLeakageReport:
    """Verify no scenario or ``(scenario_id, seed)`` key appears in more than one stored split.

    :func:`adapt_episode` separately rejects each stored split that differs from the canonical
    :func:`assign_deterministic_split` result. This batch-level report additionally preserves the
    source ``RLTrajectoryDataset.v1`` manifest's stricter ownership rule: each scenario id and each
    scenario-seed key must belong to one split. The per-key hash can otherwise assign separate seeds
    for one scenario to different splits, which would leak scenario-specific information into a
    future model.

    Args:
        episodes: Episodes carrying ``scenario_id``, ``seed``, and ``split`` attributes.

    Returns:
        A :class:`SplitLeakageReport` with per-split scenario/scenario-seed keys and any leaks.
    """
    episodes = _require_episode_sequence(episodes, field_name="episodes")
    scenario_owners: dict[str, str] = {}
    owners: dict[str, str] = {}
    split_scenario_ids: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    split_keys: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    leaked_scenario_ids: list[str] = []
    leaked: list[str] = []
    for episode in episodes:
        _require_supported_episode(episode)
        split = _episode_split(episode)
        if split not in SPLIT_NAMES:
            raise OpenDreamerAdapterError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
        scenario_id = _episode_scenario_id(episode)
        key = f"{scenario_id}:{_episode_seed(episode)}"
        split_scenario_ids[split].append(scenario_id)
        split_keys[split].append(key)
        scenario_owner = scenario_owners.get(scenario_id)
        if scenario_owner is None:
            scenario_owners[scenario_id] = split
        elif scenario_owner != split and scenario_id not in leaked_scenario_ids:
            leaked_scenario_ids.append(scenario_id)
        existing = owners.get(key)
        if existing is None:
            owners[key] = split
        elif existing != split and key not in leaked:
            leaked.append(key)
    frozen_scenarios = {
        name: tuple(sorted(set(ids))) for name, ids in split_scenario_ids.items() if ids
    }
    frozen = {name: tuple(sorted(set(keys))) for name, keys in split_keys.items() if keys}
    return SplitLeakageReport(
        ok=not leaked_scenario_ids and not leaked,
        split_scenario_seed_keys=frozen,
        leaked_keys=tuple(leaked),
        split_scenario_ids=frozen_scenarios,
        leaked_scenario_ids=tuple(leaked_scenario_ids),
    )


def adapt_episode(
    episode: RLTrajectoryEpisode,
    *,
    action_bounds: ActionBounds,
) -> StructuredEpisode:
    """Adapt one ``RLTrajectoryEpisode.v1`` into a leakage-safe structured-observation episode.

    The episode is re-validated through the canonical v1 validator, every per-step field is
    preserved, including the original per-step observations, and the structured ``drive_state`` /
    ``rays`` groups plus the validated action view are produced. The adapter stays
    **episode-major**: no flattening to transitions occurs. The original provenance is preserved
    and augmented under :data:`ADAPTER_PROVENANCE_KEY`.

    Args:
        episode: A validated ``RLTrajectoryEpisode.v1`` from the benchmark contract.
        action_bounds: Speed bounds for the ``[-1, 1] -> velocity`` mapping, recorded in provenance.

    Returns:
        A :class:`StructuredEpisode` with all v1 fields preserved plus the structured groups.

    Raises:
        OpenDreamerAdapterError: If a required field is missing, any produced output is non-finite,
            ray availability is partial or ray widths are inconsistent across episode steps,
            the stored action is not a finite 2D continuous command within the supplied physical
            action bounds, or the stored split is invalid or differs from its canonical deterministic
            assignment.
    """
    action_bounds = _require_action_bounds(action_bounds)
    _require_source_episode(episode)
    try:
        validate_rl_trajectory_episode(episode)
    except (TypeError, ValueError) as exc:
        # The adapter presents a single fail-closed error type for any v1 contract violation so
        # callers do not have to catch both ValueError and OpenDreamerAdapterError.
        raise OpenDreamerAdapterError(
            f"upstream RLTrajectoryEpisode.v1 validation failed: {exc}"
        ) from exc
    _validate_episode_per_step_containers(episode)
    _validate_episode_metadata_and_flags(episode)
    if episode.split not in SPLIT_NAMES:
        raise OpenDreamerAdapterError(f"split must be one of {SPLIT_NAMES}, got {episode.split!r}")
    expected_split = canonical_split(episode.scenario_id, episode.seed)
    if episode.split != expected_split:
        raise OpenDreamerAdapterError(
            "stored split does not match canonical deterministic split for "
            f"{episode.scenario_id!r}:{episode.seed}: expected {expected_split!r}, "
            f"got {episode.split!r}"
        )
    rewards = _finite_floats(episode.rewards, "rewards")
    return_to_go = _finite_floats(episode.return_to_go, "return_to_go")

    structured_obs: list[StructuredObservationStep] = []
    structured_actions: list[StructuredActionStep] = []
    for step_index in range(episode.step_count):
        robot_state = _episode_step_value(episode.robot_states, "robot_states", step_index)
        observation = _episode_step_value(episode.observations, "observations", step_index)
        drive_state = _extract_drive_state(robot_state)
        rays, rays_available = _extract_rays(observation)
        structured_obs.append(
            StructuredObservationStep(
                drive_state=drive_state,
                rays=rays,
                rays_available=rays_available,
            )
        )
        raw_action = _episode_step_value(episode.actions, "actions", step_index)
        structured_actions.append(_coerce_action_step(raw_action, step_index, action_bounds))

    rays_available = _validate_episode_ray_contract(structured_obs)
    provenance = _augment_provenance(
        episode.provenance,
        action_bounds=action_bounds,
        rays_available=rays_available,
    )
    return StructuredEpisode(
        dataset_id=episode.dataset_id,
        episode_id=episode.episode_id,
        scenario_id=episode.scenario_id,
        seed=episode.seed,
        source_policy_id=episode.source_policy_id,
        split=episode.split,
        observations=tuple(structured_obs),
        raw_observations=tuple(episode.observations),
        actions=tuple(structured_actions),
        raw_actions=tuple(episode.actions),
        rewards=rewards,
        return_to_go=return_to_go,
        terminated=tuple(episode.terminated),
        truncated=tuple(episode.truncated),
        pedestrians=tuple(episode.pedestrians),
        robot_states=tuple(episode.robot_states),
        provenance=provenance,
        rays_available=rays_available,
    )


def adapt_episodes(
    episodes: Sequence[RLTrajectoryEpisode],
    *,
    action_bounds: ActionBounds,
) -> list[StructuredEpisode]:
    """Adapt a sequence of v1 episodes into episode-major structured episodes.

    The batch is first checked for cross-episode scenario and scenario/seed split leakage, then each
    episode is adapted independently (see :func:`adapt_episode`), including canonical split
    enforcement. The result is returned in input order and stays episode-major. Rejecting leakage
    here prevents a caller from passing a mixed-split batch to a future model while incorrectly
    treating the adapter as the fail-closed contract boundary.

    Args:
        episodes: Validated ``RLTrajectoryEpisode.v1`` rows from the benchmark contract.
        action_bounds: Speed bounds for the ``[-1, 1] -> velocity`` mapping.

    Returns:
        A list of :class:`StructuredEpisode` in input order.

    Raises:
        OpenDreamerAdapterError: If the batch leaks a scenario or ``(scenario_id, seed)`` key across
            splits or any episode fails the fail-closed contract in :func:`adapt_episode`.
    """
    action_bounds = _require_action_bounds(action_bounds)
    episodes = _require_episode_sequence(episodes, field_name="episodes")
    for episode in episodes:
        _require_source_episode(episode)
    leakage_report = validate_split_leakage(episodes)
    if not leakage_report.ok:
        leak_descriptions: list[str] = []
        if leakage_report.leaked_scenario_ids:
            leak_descriptions.append(
                "scenario split leakage: " + ", ".join(leakage_report.leaked_scenario_ids)
            )
        if leakage_report.leaked_keys:
            leak_descriptions.append(
                "scenario/seed split leakage: " + ", ".join(leakage_report.leaked_keys)
            )
        raise OpenDreamerAdapterError(
            "split leakage detected across batch: " + "; ".join(leak_descriptions)
        )
    return [adapt_episode(episode, action_bounds=action_bounds) for episode in episodes]


def _require_episode_sequence(
    episodes: Any,
    *,
    field_name: str,
) -> Sequence[RLTrajectoryEpisode | StructuredEpisode]:
    """Require an ordered episode sequence at public batch boundaries.

    Returns:
        The validated ordered episode sequence.
    """
    if isinstance(episodes, str | bytes | Mapping) or not isinstance(episodes, Sequence):
        raise OpenDreamerAdapterError(
            f"{field_name} must be an ordered sequence of episodes, got {type(episodes).__name__}"
        )
    return episodes


def _require_source_episode(episode: Any) -> RLTrajectoryEpisode:
    """Require a source v1 episode before accessing its fields.

    Returns:
        The validated source episode.
    """
    if not isinstance(episode, RLTrajectoryEpisode):
        raise OpenDreamerAdapterError(
            f"episode must be an RLTrajectoryEpisode, got {type(episode).__name__}"
        )
    return episode


def _require_supported_episode(episode: Any) -> RLTrajectoryEpisode | StructuredEpisode:
    """Require one supported episode type before split-leakage field access.

    Returns:
        The validated source or structured episode.
    """
    if not isinstance(episode, RLTrajectoryEpisode | StructuredEpisode):
        raise OpenDreamerAdapterError(
            "episodes must contain RLTrajectoryEpisode or StructuredEpisode values, "
            f"got {type(episode).__name__}"
        )
    return episode


def _episode_step_value(values: Any, field_name: str, step_index: int) -> Any:
    """Return one position from a v1 per-step field through the adapter error boundary."""
    try:
        return values[step_index]
    except (IndexError, KeyError, TypeError) as exc:
        raise OpenDreamerAdapterError(
            f"{field_name} must support positional access at step {step_index}"
        ) from exc


def _extract_drive_state(robot_state: Any) -> np.ndarray:
    """Derive the finite ``[x, y, heading, vx, vy]`` drive_state vector from a robot_states entry.

    Args:
        robot_state: One ``robot_states`` entry, expected to expose ``position`` (2 floats),
            ``heading`` (1 float), and ``velocity`` (2 floats).

    Returns:
        A finite float ndarray of length ``len(DRIVE_STATE_LAYOUT)``.

    Raises:
        OpenDreamerAdapterError: If the entry is not a mapping or any required component is missing
            or non-finite.
    """
    if not isinstance(robot_state, Mapping):
        raise OpenDreamerAdapterError(
            f"robot_states entry must be a mapping, got {type(robot_state).__name__}"
        )
    position = robot_state.get("position")
    heading = robot_state.get("heading")
    velocity = robot_state.get("velocity")
    if position is None or heading is None or velocity is None:
        raise OpenDreamerAdapterError(
            "robot_states entry must expose position, heading, and velocity; "
            f"got keys {sorted(str(key) for key in robot_state)}"
        )
    components = [
        *_as_finite_float_pair(position, "position"),
        _as_finite_float(heading, "heading"),
        *_as_finite_float_pair(velocity, "velocity"),
    ]
    return np.asarray(components, dtype=float)


def _extract_rays(observation: Any) -> tuple[np.ndarray, bool]:
    """Derive the finite ``rays`` vector from an observation step, or report it unavailable.

    The adapter looks up the first recognized key in :data:`RAY_OBSERVATION_KEYS`. When none is
    present (the current ``RLTrajectoryDataset.v1`` recorder records no lidar rays), an empty vector
    is returned with ``available=False``. Ray distances are never fabricated from pedestrian state.

    Args:
        observation: One ``observations`` entry, expected to be a mapping.

    Returns:
        A ``(rays, available)`` pair where ``rays`` is a finite float 1D ndarray and ``available``
        is False when no ray-like field was found.

    Raises:
        OpenDreamerAdapterError: If the observation exposes a ray-like key whose value is not a
            finite, non-negative numeric sequence.
    """
    if not isinstance(observation, Mapping):
        raise OpenDreamerAdapterError(
            f"observations entry must be a mapping, got {type(observation).__name__}"
        )
    for key in RAY_OBSERVATION_KEYS:
        if key not in observation:
            continue
        return _coerce_ray_ranges(observation[key], key), True
    return np.asarray([], dtype=float), False


def _coerce_ray_ranges(raw: Any, key: str) -> np.ndarray:
    """Return one finite, non-negative ray-range vector from a recognized observation field."""
    try:
        raw_array = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise OpenDreamerAdapterError(f"observation ray-like key {key!r} must be numeric") from exc
    if raw_array.ndim != 1:
        raise OpenDreamerAdapterError(f"observation ray-like key {key!r} must be a 1D sequence")
    if raw_array.size == 0:
        raise OpenDreamerAdapterError(
            f"observation ray-like key {key!r} must contain at least one range"
        )
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_array):
        raise OpenDreamerAdapterError(
            f"observation ray-like key {key!r} must contain finite real values"
        )
    try:
        arr = raw_array.astype(float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OpenDreamerAdapterError(
            f"observation ray-like key {key!r} must contain finite real values"
        ) from exc
    if not np.all(np.isfinite(arr)):
        raise OpenDreamerAdapterError(f"observation ray-like key {key!r} must be finite")
    if np.any(arr < 0.0):
        raise OpenDreamerAdapterError(
            f"observation ray-like key {key!r} must contain non-negative ranges"
        )
    return arr


def _validate_episode_ray_contract(steps: Sequence[StructuredObservationStep]) -> bool:
    """Require one stackable ray representation across an episode's structured steps.

    The adapter deliberately has no masking or padding policy. An episode is therefore either
    entirely ray-free (every step reports ``rays_available=False`` and has an empty vector), or
    entirely ray-bearing with one fixed ray-vector width. This keeps a future structured sequence
    encoder from receiving an object/ragged array while preserving the recorder's semantics rather
    than inventing missing ranges.

    Args:
        steps: Structured observation steps already validated for finite per-step values.

    Returns:
        ``True`` when all steps carry one common-width ray vector; ``False`` when no step carries
        rays.

    Raises:
        OpenDreamerAdapterError: If ray availability is partial or ray-vector widths differ.
    """
    availability = tuple(step.rays_available for step in steps)
    if not any(availability):
        return False

    available_steps = [index for index, available in enumerate(availability) if available]
    unavailable_steps = [index for index, available in enumerate(availability) if not available]
    if unavailable_steps:
        raise OpenDreamerAdapterError(
            "ray availability must be episode-wide: recognized ray-like observations are present "
            f"at steps {available_steps} but absent at steps {unavailable_steps}"
        )

    widths_by_step = [step.rays.size for step in steps]
    if len(set(widths_by_step)) != 1:
        width_summary = ", ".join(
            f"step {index}: {width}" for index, width in enumerate(widths_by_step)
        )
        raise OpenDreamerAdapterError(
            f"ray vectors must have a consistent length across an episode; got {width_summary}"
        )
    return True


def _coerce_action_step(
    raw_action: Any,
    step_index: int,
    action_bounds: ActionBounds,
) -> StructuredActionStep:
    """Normalize and validate one stored physical ``(linear, angular)`` command.

    Args:
        raw_action: The raw stored action for the given step. The current recorder emits a mapping
            with ``linear_velocity`` and ``angular_velocity`` keys; a two-element sequence remains
            supported for compatible legacy datasets.
        step_index: Step index used only for a clear error message.
        action_bounds: Physical linear and angular velocity envelope that the command must satisfy.

    Returns:
        A :class:`StructuredActionStep` containing the normalized ``(linear, angular)`` command.

    Raises:
        OpenDreamerAdapterError: If the action does not have a supported representation, is not a
            finite 2D numeric ``(linear, angular)`` command, or lies outside ``action_bounds``.
    """
    if isinstance(raw_action, Mapping):
        required_keys = ("linear_velocity", "angular_velocity")
        missing_keys = [key for key in required_keys if key not in raw_action]
        if missing_keys:
            raise OpenDreamerAdapterError(
                f"action at step {step_index} mapping must contain {required_keys}, "
                f"missing {missing_keys}"
            )
        unexpected_keys = sorted(str(key) for key in raw_action if key not in required_keys)
        if unexpected_keys:
            raise OpenDreamerAdapterError(
                f"action at step {step_index} mapping must contain only {required_keys}, "
                f"unexpected {unexpected_keys} -- incompatible action space"
            )
        values = [raw_action[key] for key in required_keys]
    elif isinstance(raw_action, Sequence | np.ndarray) and not isinstance(raw_action, str | bytes):
        values = (
            list(np.atleast_1d(raw_action))
            if isinstance(raw_action, np.ndarray)
            else list(raw_action)
        )
    else:
        raise OpenDreamerAdapterError(
            f"action at step {step_index} must be a recorder action mapping or sequence, "
            f"got {type(raw_action).__name__}"
        )
    if len(values) != EXPECTED_ACTION_DIM:
        raise OpenDreamerAdapterError(
            f"action at step {step_index} must be {EXPECTED_ACTION_DIM}D (linear, angular), "
            f"got {len(values)}D -- incompatible action space"
        )
    coerced = [_finite_action_component(value, step_index) for value in values]
    linear_velocity, angular_velocity = coerced
    if not (
        action_bounds.min_linear_speed <= linear_velocity <= action_bounds.max_linear_speed
        and -action_bounds.max_angular_speed <= angular_velocity <= action_bounds.max_angular_speed
    ):
        raise OpenDreamerAdapterError(
            f"action at step {step_index} must lie within supplied action bounds: "
            f"linear in [{action_bounds.min_linear_speed}, {action_bounds.max_linear_speed}], "
            f"angular in [-{action_bounds.max_angular_speed}, {action_bounds.max_angular_speed}]"
        )
    return StructuredActionStep(raw=tuple(coerced))


def _finite_action_component(value: Any, step_index: int) -> float:
    """Return one finite stored-action component through the public adapter error boundary."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OpenDreamerAdapterError(f"action at step {step_index} must be numeric, got {value!r}")
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OpenDreamerAdapterError(
            f"action at step {step_index} must be finite, got {value!r}"
        ) from exc
    if not np.isfinite(numeric_value):
        raise OpenDreamerAdapterError(f"action at step {step_index} must be finite")
    return numeric_value


def _finite_floats(values: Sequence[Any], field_name: str) -> tuple[float, ...]:
    """Coerce a sequence to finite floats, failing closed on non-finite values.

    Args:
        values: A sequence of numeric values (e.g. ``rewards`` or ``return_to_go``).
        field_name: Field name used in the error message.

    Returns:
        A tuple of finite floats.

    Raises:
        OpenDreamerAdapterError: If any value is non-numeric or non-finite.
    """
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise OpenDreamerAdapterError(f"{field_name} must be numeric, got {value!r}")
        try:
            out.append(float(value))
        except (OverflowError, TypeError, ValueError) as exc:
            raise OpenDreamerAdapterError(f"{field_name} must be finite, got {value!r}") from exc
    if not all(np.isfinite(out)):
        raise OpenDreamerAdapterError(f"{field_name} must be finite")
    return tuple(out)


def _validate_episode_metadata_and_flags(episode: RLTrajectoryEpisode) -> None:
    """Validate v1 metadata and terminal flags that the upstream validator leaves permissive.

    The stable v1 validator owns episode alignment and terminal-marker placement, but programmatic
    producers can still construct dataclass instances with coercible metadata or non-boolean flags.
    The adapter must reject those values instead of changing their meaning while adapting them.

    Args:
        episode: Episode already checked by :func:`validate_rl_trajectory_episode`.

    Raises:
        OpenDreamerAdapterError: If identifiers are not non-empty strings, the seed is not a
            non-boolean integer, provenance is not a mapping, or terminal flags are not booleans.
    """
    for field_name in ("dataset_id", "episode_id", "scenario_id", "source_policy_id", "split"):
        value = getattr(episode, field_name)
        if not isinstance(value, str) or not value:
            raise OpenDreamerAdapterError(f"{field_name} must be a non-empty string, got {value!r}")
    if isinstance(episode.seed, bool) or not isinstance(episode.seed, int):
        raise OpenDreamerAdapterError(f"seed must be a non-boolean integer, got {episode.seed!r}")
    if not isinstance(episode.provenance, Mapping):
        raise OpenDreamerAdapterError(
            f"provenance must be a mapping, got {type(episode.provenance).__name__}"
        )
    for field_name in ("terminated", "truncated"):
        for step_index, value in enumerate(getattr(episode, field_name)):
            if not isinstance(value, bool):
                raise OpenDreamerAdapterError(
                    f"{field_name}[{step_index}] must be a boolean, got {value!r}"
                )


def _validate_episode_per_step_containers(episode: RLTrajectoryEpisode) -> None:
    """Require every v1 per-step field to retain ordered sequence semantics.

    ``validate_rl_trajectory_episode`` validates aligned lengths, but a programmatic producer can
    still supply a mapping with numeric keys. Iterating such a mapping yields its keys, which could
    silently replace recorded rewards or raw payloads with unrelated integers. The adapter rejects
    those containers rather than treating mapping keys as trajectory values.
    """
    field_names = (
        "observations",
        "actions",
        "rewards",
        "return_to_go",
        "terminated",
        "truncated",
        "pedestrians",
        "robot_states",
    )
    for field_name in field_names:
        values = getattr(episode, field_name)
        if isinstance(values, str | bytes | Mapping) or not isinstance(
            values, Sequence | np.ndarray
        ):
            raise OpenDreamerAdapterError(
                f"{field_name} must be an ordered per-step sequence, got {type(values).__name__}"
            )


def _json_safe_value(value: Any, field_name: str) -> Any:
    """Return a recursively JSON-safe copy of an accepted raw v1 value.

    Args:
        value: Raw v1 value to serialize.
        field_name: Field path used in fail-closed error messages.

    Returns:
        A JSON-safe scalar, list, or string-keyed mapping.

    Raises:
        OpenDreamerAdapterError: If a mapping key is not a string or a value has no supported
            JSON representation.
    """
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise OpenDreamerAdapterError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_safe_numpy_scalar(value, field_name)
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist(), field_name)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise OpenDreamerAdapterError(
                    f"{field_name} mapping keys must be strings for JSON serialization, got {key!r}"
                )
            out[key] = _json_safe_value(item, f"{field_name}.{key}")
        return out
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [
            _json_safe_value(item, f"{field_name}[{index}]") for index, item in enumerate(value)
        ]
    raise OpenDreamerAdapterError(
        f"{field_name} contains unsupported JSON value {type(value).__name__}"
    )


def _json_safe_numpy_scalar(value: np.generic, field_name: str) -> Any:
    """Convert a NumPy scalar without recursing on extended-precision floating values.

    Returns:
        A recursively JSON-safe scalar value.
    """
    if isinstance(value, np.floating):
        # ``np.longdouble.item()`` can return another NumPy scalar rather than a Python float.
        # JSON has no extended-precision scalar type, so serialize through a checked float instead.
        return _json_safe_value(float(value), field_name)
    return _json_safe_value(value.item(), field_name)


def _as_finite_float(value: Any, name: str) -> float:
    """Coerce one value to a finite float.

    Args:
        value: A numeric scalar.
        name: Component name used in the error message.

    Returns:
        The finite float.

    Raises:
        OpenDreamerAdapterError: If the value is non-numeric or non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OpenDreamerAdapterError(f"drive_state {name} must be numeric, got {value!r}")
    try:
        out = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OpenDreamerAdapterError(f"drive_state {name} must be finite, got {value!r}") from exc
    if not np.isfinite(out):
        raise OpenDreamerAdapterError(f"drive_state {name} must be finite, got {value!r}")
    return out


def _as_finite_float_pair(value: Any, name: str) -> tuple[float, float]:
    """Coerce a length-2 sequence to a pair of finite floats.

    Args:
        value: A length-2 numeric sequence (e.g. ``position`` or ``velocity``).
        name: Component name used in the error message.

    Returns:
        A pair of finite floats.

    Raises:
        OpenDreamerAdapterError: If the value is not a length-2 finite numeric sequence.
    """
    if isinstance(value, str | bytes | Mapping) or not isinstance(value, Sequence | np.ndarray):
        raise OpenDreamerAdapterError(
            f"drive_state {name} must be a sequence, got {type(value).__name__}"
        )
    try:
        items = list(value)
    except TypeError as exc:
        raise OpenDreamerAdapterError(
            f"drive_state {name} must be a sequence, got {type(value).__name__}"
        ) from exc
    if len(items) != 2:
        raise OpenDreamerAdapterError(
            f"drive_state {name} must have length 2, got length {len(items)}"
        )
    return _as_finite_float(items[0], f"{name}[0]"), _as_finite_float(items[1], f"{name}[1]")


def _augment_provenance(
    provenance: Mapping[str, Any],
    *,
    action_bounds: ActionBounds,
    rays_available: bool,
) -> Mapping[str, Any]:
    """Preserve the original episode provenance and add the adapter metadata entry.

    The original mapping is never mutated; a shallow copy is returned with the
    :data:`ADAPTER_PROVENANCE_KEY` entry added. A source collision fails closed so existing
    provenance is never overwritten.

    Args:
        provenance: The original v1 episode provenance mapping.
        action_bounds: Speed bounds to record for the ``[-1, 1] -> velocity`` mapping.
        rays_available: Whether every episode step exposed one fixed-width ray-like observation.

    Returns:
        A new mapping with the adapter provenance entry nested under ADAPTER_PROVENANCE_KEY.

    Raises:
        OpenDreamerAdapterError: If the source provenance already owns the adapter key.
    """
    if ADAPTER_PROVENANCE_KEY in provenance:
        raise OpenDreamerAdapterError(
            f"provenance already contains reserved key {ADAPTER_PROVENANCE_KEY!r}; "
            "refusing to overwrite source provenance"
        )
    merged: dict[str, Any] = dict(provenance)
    merged[ADAPTER_PROVENANCE_KEY] = {
        "adapter_version": OPEN_DREAMER_ADAPTER_VERSION,
        "consumed_episode_schema": RL_TRAJECTORY_EPISODE_SCHEMA_VERSION,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "split_policy": "assign_deterministic_split",
        "observation_contract": OPEN_DREAMER_OBSERVATION_CONTRACT,
        "observation_normalization": OBSERVATION_NORMALIZATION,
        "drive_state_layout": list(DRIVE_STATE_LAYOUT),
        "action_mapping": {
            "kind": "affine_linear_symmetric_angular",
            "input_domain": [-1.0, 1.0],
            "output": "(linear_velocity, angular_velocity)",
        },
        "action_bounds": action_bounds.to_dict(),
        "rays_available": bool(rays_available),
        "ray_observation_keys": list(RAY_OBSERVATION_KEYS),
    }
    return merged


def _episode_split(episode: RLTrajectoryEpisode | StructuredEpisode) -> str:
    """Return the split name from either episode type.

    Args:
        episode: An episode with a ``split`` attribute.

    Returns:
        The split name string.
    """
    split = episode.split
    if not isinstance(split, str):
        raise OpenDreamerAdapterError(f"split must be a string, got {split!r}")
    return split


def _episode_scenario_id(episode: RLTrajectoryEpisode | StructuredEpisode) -> str:
    """Return the scenario id from either episode type.

    Args:
        episode: An episode with a ``scenario_id`` attribute.

    Returns:
        The scenario id string.
    """
    scenario_id = episode.scenario_id
    if not isinstance(scenario_id, str) or not scenario_id:
        raise OpenDreamerAdapterError(
            f"scenario_id must be a non-empty string, got {scenario_id!r}"
        )
    return scenario_id


def _episode_seed(episode: RLTrajectoryEpisode | StructuredEpisode) -> int:
    """Return the integer seed from either episode type.

    Args:
        episode: An episode with a ``seed`` attribute.

    Returns:
        The integer seed.
    """
    seed = episode.seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OpenDreamerAdapterError(f"seed must be a non-boolean integer, got {seed!r}")
    return seed


__all__ = [
    "ADAPTER_PROVENANCE_KEY",
    "DRIVE_STATE_LAYOUT",
    "EVIDENCE_BOUNDARY",
    "EXPECTED_ACTION_DIM",
    "OBSERVATION_NORMALIZATION",
    "OPEN_DREAMER_ADAPTER_VERSION",
    "OPEN_DREAMER_OBSERVATION_CONTRACT",
    "RAY_OBSERVATION_KEYS",
    "ActionBounds",
    "OpenDreamerAdapterError",
    "SplitLeakageReport",
    "StructuredActionStep",
    "StructuredEpisode",
    "StructuredObservationStep",
    "adapt_episode",
    "adapt_episodes",
    "canonical_split",
    "map_action_to_velocity",
    "validate_split_leakage",
]
