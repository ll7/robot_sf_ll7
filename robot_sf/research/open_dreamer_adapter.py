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
  honestly reported as **unavailable** (empty vector, ``available=False``). The adapter never
  fabricates ray distances from pedestrian positions -- doing so would invent new contract
  semantics.
* ``[-1, 1] -> (linear velocity, angular velocity)`` action mapping -- the clean-room replacement
  for upstream's VPT-style action container. The mapping is parameterized by caller-supplied speed
  bounds (:class:`ActionBounds`) so the adapter never hardcodes or assumes a particular robot
  drivetrain; the bounds used are recorded in the structured episode's provenance.

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
* any produced output is non-finite (NaN/Inf in ``drive_state``, ``rays``, ``rewards``,
  ``return_to_go``, or the mapped velocity);
* a stored action is not a finite 2D continuous ``(linear, angular)`` command (incompatible action
  space);
* the dataset leaks a ``(scenario_id, seed)`` key across more than one split.

Split policy is the canonical :func:`assign_deterministic_split` from the benchmark contract;
:func:`validate_split_leakage` verifies no scenario-seed key appears in two splits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
            ``-max_linear_speed <= min_linear_speed <= max_linear_speed``.
    """

    max_linear_speed: float
    max_angular_speed: float
    min_linear_speed: float = 0.0

    def __post_init__(self) -> None:
        """Validate that the speed bounds define a non-degenerate, finite velocity envelope."""
        if not np.isfinite(self.max_linear_speed) or self.max_linear_speed <= 0.0:
            raise OpenDreamerAdapterError(
                f"max_linear_speed must be positive and finite, got {self.max_linear_speed!r}"
            )
        if not np.isfinite(self.max_angular_speed) or self.max_angular_speed <= 0.0:
            raise OpenDreamerAdapterError(
                f"max_angular_speed must be positive and finite, got {self.max_angular_speed!r}"
            )
        if not np.isfinite(self.min_linear_speed):
            raise OpenDreamerAdapterError(
                f"min_linear_speed must be finite, got {self.min_linear_speed!r}"
            )
        if self.min_linear_speed < -self.max_linear_speed:
            raise OpenDreamerAdapterError(
                f"min_linear_speed ({self.min_linear_speed}) must be >= "
                f"-max_linear_speed (-{self.max_linear_speed})"
            )
        if self.min_linear_speed > self.max_linear_speed:
            raise OpenDreamerAdapterError(
                f"min_linear_speed ({self.min_linear_speed}) must be <= "
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
    """

    drive_state: np.ndarray
    rays: np.ndarray
    rays_available: bool

    def __post_init__(self) -> None:
        """Validate the finite-output contract for both groups."""
        if self.drive_state.ndim != 1 or self.drive_state.shape[0] != len(DRIVE_STATE_LAYOUT):
            raise OpenDreamerAdapterError(
                "drive_state must be a 1D vector of length "
                f"{len(DRIVE_STATE_LAYOUT)}, got shape {self.drive_state.shape}"
            )
        if self.drive_state.dtype.kind != "f" or not np.all(np.isfinite(self.drive_state)):
            raise OpenDreamerAdapterError("drive_state must be finite floats")
        if self.rays.ndim != 1 or self.rays.dtype.kind != "f":
            raise OpenDreamerAdapterError("rays must be a 1D float vector")
        if not np.all(np.isfinite(self.rays)):
            raise OpenDreamerAdapterError("rays must be finite floats")
        if not self.rays_available and self.rays.size != 0:
            raise OpenDreamerAdapterError("rays_available=False requires an empty rays vector")


@dataclass(frozen=True, slots=True)
class StructuredActionStep:
    """One step of the preserved action view.

    The raw stored ``(linear, angular)`` command is preserved verbatim (semantics untouched) and
    validated to be a finite 2D continuous command. The forward ``[-1, 1] -> velocity`` mapping is
    provided separately by :func:`map_action_to_velocity` so the adapter does not assert an
    unverified unit for the stored action; the bounds used are recorded in the episode provenance.

    Attributes:
        raw: The preserved raw ``(linear, angular)`` action exactly as recorded, finite-validated.
    """

    raw: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the stored action is a finite 2D continuous command."""
        if len(self.raw) != EXPECTED_ACTION_DIM:
            raise OpenDreamerAdapterError(
                f"stored action must be {EXPECTED_ACTION_DIM}D (linear, angular), "
                f"got {len(self.raw)}D"
            )
        for value in self.raw:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise OpenDreamerAdapterError(f"stored action must be numeric, got {value!r}")
            if not np.isfinite(float(value)):
                raise OpenDreamerAdapterError(f"stored action must be finite, got {value!r}")


@dataclass(frozen=True, slots=True)
class StructuredEpisode:
    """Episode-major structured-observation + action view with full v1 provenance preserved.

    The view is **episode-major**: per-step fields are aligned tuples whose lengths equal the
    episode step count. Episodes are never flattened into transitions here, so terminal/truncated
    semantics cannot be silently crossed. Every v1 field is preserved verbatim in addition to the
    structured groups; ``provenance`` carries the original episode provenance plus an
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
        actions: Per-step preserved action view.
        rewards: Preserved v1 per-step rewards, finite-validated.
        return_to_go: Preserved v1 per-step return-to-go, finite-validated.
        terminated: Preserved v1 per-step terminated flags.
        truncated: Preserved v1 per-step truncated flags.
        pedestrians: Preserved v1 per-step pedestrian state.
        robot_states: Preserved v1 per-step robot state.
        provenance: Original episode provenance plus the adapter provenance entry.
        rays_available: Whether any step exposed a recognized ray-like field.
        drive_state_layout: The fixed component order of :attr:`observations[i].drive_state`.
    """

    dataset_id: str
    episode_id: str
    scenario_id: str
    seed: int
    source_policy_id: str
    split: str
    observations: tuple[StructuredObservationStep, ...]
    actions: tuple[StructuredActionStep, ...]
    rewards: tuple[float, ...]
    return_to_go: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    pedestrians: tuple[Any, ...]
    robot_states: tuple[Any, ...]
    provenance: Mapping[str, Any]
    rays_available: bool
    drive_state_layout: tuple[str, ...] = field(default=DRIVE_STATE_LAYOUT)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the structured episode (groups as lists).

        Returns:
            A dictionary with all preserved v1 fields and the structured-observation/action groups.
        """
        return {
            "schema_version": OPEN_DREAMER_ADAPTER_VERSION,
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "source_policy_id": self.source_policy_id,
            "split": self.split,
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
            "actions": [list(step.raw) for step in self.actions],
            "rewards": list(self.rewards),
            "return_to_go": list(self.return_to_go),
            "terminated": list(self.terminated),
            "truncated": list(self.truncated),
            "pedestrians": list(self.pedestrians),
            "robot_states": list(self.robot_states),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class SplitLeakageReport:
    """Result of checking that no ``(scenario_id, seed)`` key leaks across splits.

    Attributes:
        ok: True when every scenario-seed key maps to exactly one split.
        split_scenario_seed_keys: Mapping of split name to the sorted scenario-seed keys it owns.
        leaked_keys: Scenario-seed keys that appear in more than one split (empty when ``ok``).
    """

    ok: bool
    split_scenario_seed_keys: Mapping[str, tuple[str, ...]]
    leaked_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the leakage check.

        Returns:
            A dictionary with ``ok``, per-split scenario-seed keys, and any leaked keys.
        """
        return {
            "ok": bool(self.ok),
            "split_scenario_seed_keys": {
                name: list(keys) for name, keys in self.split_scenario_seed_keys.items()
            },
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
    inputs must lie in ``[-1, 1]`` (a tiny float tolerance absorbs roundoff); out-of-domain inputs
    are rejected rather than silently clipped, so a future policy cannot emit an action that escapes
    the declared envelope unnoticed. Both outputs are checked finite.

    Args:
        normalized: A length-2 action with both components in ``[-1, 1]``.
        bounds: Speed bounds defining the velocity envelope.

    Returns:
        The mapped ``(linear_velocity, angular_velocity)``.

    Raises:
        OpenDreamerAdapterError: If the action is not length-2, is out of ``[-1, 1]``, is
            non-finite, or maps to a non-finite velocity.
    """
    arr = np.asarray(normalized, dtype=float)
    if arr.shape != (EXPECTED_ACTION_DIM,):
        raise OpenDreamerAdapterError(
            f"normalized action must have shape ({EXPECTED_ACTION_DIM},), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise OpenDreamerAdapterError("normalized action must be finite")
    if np.any(arr < -1.0 - 1e-9) or np.any(arr > 1.0 + 1e-9):
        raise OpenDreamerAdapterError("normalized action must lie in [-1, 1]")
    linear_range = bounds.max_linear_speed - bounds.min_linear_speed
    linear_velocity = bounds.min_linear_speed + (arr[0] + 1.0) * 0.5 * linear_range
    angular_velocity = arr[1] * bounds.max_angular_speed
    linear_velocity = float(linear_velocity)
    angular_velocity = float(angular_velocity)
    if not (np.isfinite(linear_velocity) and np.isfinite(angular_velocity)):
        raise OpenDreamerAdapterError("mapped velocity must be finite")
    return linear_velocity, angular_velocity


def canonical_split(scenario_id: str, seed: int) -> str:
    """Return the canonical deterministic split for a ``(scenario_id, seed)`` key.

    This is a thin delegation to the benchmark contract's
    :func:`assign_deterministic_split`, re-exposed so consumers of the adapter have a single
    import surface and the split policy respected by :func:`validate_split_leakage` is explicit.
    A dataset split entirely through this function cannot leak a scenario-seed key by
    construction.

    Args:
        scenario_id: Scenario id.
        seed: Integer seed.

    Returns:
        One of :data:`SPLIT_NAMES`.
    """
    return assign_deterministic_split(scenario_id, seed)


def validate_split_leakage(
    episodes: Sequence[RLTrajectoryEpisode | StructuredEpisode],
) -> SplitLeakageReport:
    """Verify no ``(scenario_id, seed)`` key appears in more than one split.

    The split policy respected here is the canonical :func:`assign_deterministic_split`: a
    deterministic scenario-seed hash. Because that hash is a pure function of
    ``(scenario_id, seed)``, a dataset assigned entirely through it cannot leak by construction;
    this check still fails closed for datasets whose stored ``split`` field was assigned by any
    other policy, so leakage is caught rather than silently propagated to a future model.

    Args:
        episodes: Episodes carrying ``scenario_id``, ``seed``, and ``split`` attributes.

    Returns:
        A :class:`SplitLeakageReport` with per-split scenario-seed keys and any leaked keys.
    """
    owners: dict[str, str] = {}
    split_keys: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    leaked: list[str] = []
    for episode in episodes:
        split = _episode_split(episode)
        if split not in SPLIT_NAMES:
            raise OpenDreamerAdapterError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
        key = f"{_episode_scenario_id(episode)}:{_episode_seed(episode)}"
        split_keys[split].append(key)
        existing = owners.get(key)
        if existing is None:
            owners[key] = split
        elif existing != split and key not in leaked:
            leaked.append(key)
    frozen = {name: tuple(sorted(set(keys))) for name, keys in split_keys.items() if keys}
    return SplitLeakageReport(
        ok=len(leaked) == 0,
        split_scenario_seed_keys=frozen,
        leaked_keys=tuple(leaked),
    )


def adapt_episode(
    episode: RLTrajectoryEpisode,
    *,
    action_bounds: ActionBounds,
) -> StructuredEpisode:
    """Adapt one ``RLTrajectoryEpisode.v1`` into a leakage-safe structured-observation episode.

    The episode is re-validated through the canonical v1 validator, every per-step field is
    preserved, and the structured ``drive_state`` / ``rays`` groups plus the validated action view
    are produced. The adapter stays **episode-major**: no flattening to transitions occurs. The
    original provenance is preserved and augmented under :data:`ADAPTER_PROVENANCE_KEY`.

    Args:
        episode: A validated ``RLTrajectoryEpisode.v1`` from the benchmark contract.
        action_bounds: Speed bounds for the ``[-1, 1] -> velocity`` mapping, recorded in provenance.

    Returns:
        A :class:`StructuredEpisode` with all v1 fields preserved plus the structured groups.

    Raises:
        OpenDreamerAdapterError: If a required field is missing, any produced output is non-finite,
            the stored action is not a finite 2D continuous command, or the episode split is invalid.
    """
    try:
        validate_rl_trajectory_episode(episode)
    except ValueError as exc:
        # The adapter presents a single fail-closed error type for any v1 contract violation so
        # callers do not have to catch both ValueError and OpenDreamerAdapterError.
        raise OpenDreamerAdapterError(
            f"upstream RLTrajectoryEpisode.v1 validation failed: {exc}"
        ) from exc
    if episode.split not in SPLIT_NAMES:
        raise OpenDreamerAdapterError(f"split must be one of {SPLIT_NAMES}, got {episode.split!r}")
    rewards = _finite_floats(episode.rewards, "rewards")
    return_to_go = _finite_floats(episode.return_to_go, "return_to_go")

    structured_obs: list[StructuredObservationStep] = []
    structured_actions: list[StructuredActionStep] = []
    any_rays = False
    for step_index in range(episode.step_count):
        robot_state = episode.robot_states[step_index]
        observation = episode.observations[step_index]
        drive_state = _extract_drive_state(robot_state)
        rays, rays_available = _extract_rays(observation)
        any_rays = any_rays or rays_available
        structured_obs.append(
            StructuredObservationStep(
                drive_state=drive_state,
                rays=rays,
                rays_available=rays_available,
            )
        )
        raw_action = episode.actions[step_index]
        structured_actions.append(_coerce_action_step(raw_action, step_index))

    provenance = _augment_provenance(
        episode.provenance,
        action_bounds=action_bounds,
        rays_available=any_rays,
    )
    return StructuredEpisode(
        dataset_id=episode.dataset_id,
        episode_id=episode.episode_id,
        scenario_id=episode.scenario_id,
        seed=episode.seed,
        source_policy_id=episode.source_policy_id,
        split=episode.split,
        observations=tuple(structured_obs),
        actions=tuple(structured_actions),
        rewards=rewards,
        return_to_go=return_to_go,
        terminated=tuple(bool(value) for value in episode.terminated),
        truncated=tuple(bool(value) for value in episode.truncated),
        pedestrians=tuple(episode.pedestrians),
        robot_states=tuple(episode.robot_states),
        provenance=provenance,
        rays_available=any_rays,
    )


def adapt_episodes(
    episodes: Sequence[RLTrajectoryEpisode],
    *,
    action_bounds: ActionBounds,
) -> list[StructuredEpisode]:
    """Adapt a sequence of v1 episodes into episode-major structured episodes.

    The batch is first checked for cross-episode scenario/seed split leakage, then each episode is
    adapted independently (see :func:`adapt_episode`). The result is returned in input order and
    stays episode-major. Rejecting leakage here prevents a caller from passing a mixed-split batch
    to a future model while incorrectly treating the adapter as the fail-closed contract boundary.

    Args:
        episodes: Validated ``RLTrajectoryEpisode.v1`` rows from the benchmark contract.
        action_bounds: Speed bounds for the ``[-1, 1] -> velocity`` mapping.

    Returns:
        A list of :class:`StructuredEpisode` in input order.

    Raises:
        OpenDreamerAdapterError: If the batch leaks a ``(scenario_id, seed)`` key across splits or
            any episode fails the fail-closed contract in :func:`adapt_episode`.
    """
    leakage_report = validate_split_leakage(episodes)
    if not leakage_report.ok:
        raise OpenDreamerAdapterError(
            "scenario/seed split leakage detected across batch: "
            f"{', '.join(leakage_report.leaked_keys)}"
        )
    return [adapt_episode(episode, action_bounds=action_bounds) for episode in episodes]


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
            f"got keys {sorted(robot_state.keys())}"
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
            finite numeric sequence.
    """
    if not isinstance(observation, Mapping):
        raise OpenDreamerAdapterError(
            f"observations entry must be a mapping, got {type(observation).__name__}"
        )
    for key in RAY_OBSERVATION_KEYS:
        if key not in observation:
            continue
        raw = observation[key]
        try:
            arr = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise OpenDreamerAdapterError(
                f"observation ray-like key {key!r} must be numeric"
            ) from exc
        if arr.ndim != 1:
            raise OpenDreamerAdapterError(f"observation ray-like key {key!r} must be a 1D sequence")
        if arr.size and not np.all(np.isfinite(arr)):
            raise OpenDreamerAdapterError(f"observation ray-like key {key!r} must be finite")
        return arr, True
    return np.asarray([], dtype=float), False


def _coerce_action_step(raw_action: Any, step_index: int) -> StructuredActionStep:
    """Validate and preserve one stored action as a finite 2D continuous command.

    Args:
        raw_action: The raw stored action for the given step.
        step_index: Step index used only for a clear error message.

    Returns:
        A :class:`StructuredActionStep` preserving the raw ``(linear, angular)`` command.

    Raises:
        OpenDreamerAdapterError: If the action is not a finite 2D numeric ``(linear, angular)``
            command (incompatible action space).
    """
    try:
        values = list(raw_action)
    except TypeError as exc:
        raise OpenDreamerAdapterError(
            f"action at step {step_index} must be a sequence, got {type(raw_action).__name__}"
        ) from exc
    if len(values) != EXPECTED_ACTION_DIM:
        raise OpenDreamerAdapterError(
            f"action at step {step_index} must be {EXPECTED_ACTION_DIM}D (linear, angular), "
            f"got {len(values)}D -- incompatible action space"
        )
    coerced: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise OpenDreamerAdapterError(
                f"action at step {step_index} must be numeric, got {value!r}"
            )
        coerced.append(float(value))
    if not all(np.isfinite(coerced)):
        raise OpenDreamerAdapterError(f"action at step {step_index} must be finite")
    return StructuredActionStep(raw=tuple(coerced))


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
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise OpenDreamerAdapterError(f"{field_name} must be numeric, got {value!r}")
        out.append(float(value))
    if not all(np.isfinite(out)):
        raise OpenDreamerAdapterError(f"{field_name} must be finite")
    return tuple(out)


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
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OpenDreamerAdapterError(f"drive_state {name} must be numeric, got {value!r}")
    out = float(value)
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
    :data:`ADAPTER_PROVENANCE_KEY` entry added or replaced.

    Args:
        provenance: The original v1 episode provenance mapping.
        action_bounds: Speed bounds to record for the ``[-1, 1] -> velocity`` mapping.
        rays_available: Whether the episode exposed any ray-like observation field.

    Returns:
        A new mapping with the adapter provenance entry nested under ADAPTER_PROVENANCE_KEY.
    """
    merged: dict[str, Any] = dict(provenance)
    merged[ADAPTER_PROVENANCE_KEY] = {
        "adapter_version": OPEN_DREAMER_ADAPTER_VERSION,
        "consumed_episode_schema": RL_TRAJECTORY_EPISODE_SCHEMA_VERSION,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "split_policy": "assign_deterministic_split",
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
    return str(episode.split)


def _episode_scenario_id(episode: RLTrajectoryEpisode | StructuredEpisode) -> str:
    """Return the scenario id from either episode type.

    Args:
        episode: An episode with a ``scenario_id`` attribute.

    Returns:
        The scenario id string.
    """
    return str(episode.scenario_id)


def _episode_seed(episode: RLTrajectoryEpisode | StructuredEpisode) -> int:
    """Return the integer seed from either episode type.

    Args:
        episode: An episode with a ``seed`` attribute.

    Returns:
        The integer seed.
    """
    return int(episode.seed)


__all__ = [
    "ADAPTER_PROVENANCE_KEY",
    "DRIVE_STATE_LAYOUT",
    "EVIDENCE_BOUNDARY",
    "EXPECTED_ACTION_DIM",
    "OPEN_DREAMER_ADAPTER_VERSION",
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
