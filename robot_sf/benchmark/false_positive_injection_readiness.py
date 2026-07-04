"""Fail-closed readiness check for false-positive actor-injection replay inputs.

Issue #3300 carries the false-positive actor-injection acceptance dimension that
PR #3271 closed out of issue #2927 as *unavailable*.  Before any replay campaign
can produce that evidence, the replay *condition spec* must be well-formed: it
needs the injected-actor inputs that the canonical
:class:`~robot_sf.benchmark.observation_perturbation.ObservationPerturbationSpec`
can consume, plus the provenance fields the report contract requires (scenario,
seed, planner mode, perturbation family, execution mode, issue link).

This module is a **bounded readiness contract only**.  It validates *inputs and
provenance*; it does not run a replay, change sensor semantics, or make any
benchmark or safety claim.  It exists so an unavailable or malformed
false-positive replay condition fails closed with an actionable blocker instead
of silently passing as success — the explicit fail-closed discipline named in
``docs/context/issue_691_benchmark_fallback_policy.md``.

Status vocabulary (aligned with ``robot_sf.benchmark.fallback_policy``):

- ``ready``: inputs and provenance are valid; the replay condition can be built.
- ``not_available``: no false-positive actors are requested (empty/omitted
  injection input).  This is the accepted-unavailable state that mirrors how
  #2927 recorded the missing dimension — not an error, but not evidence either.
- ``blocked``: inputs are malformed or required provenance is missing.  The
  blocker list names exactly what must be fixed before the condition is usable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from robot_sf.benchmark.observation_levels import observation_level_spec
from robot_sf.benchmark.observation_perturbation import (
    NOISE_PROFILE_FALSE_POSITIVE,
    ObservationPerturbationSpec,
)
from robot_sf.common.issue_provenance import (
    FALSE_POSITIVE_INJECTION_REPLAY_READINESS_ISSUE,
)

FALSE_POSITIVE_INJECTION_READINESS_SCHEMA_VERSION = "false_positive_injection_readiness.v1"

#: Canonical name for the perturbation family this readiness contract gates.
FALSE_POSITIVE_PERTURBATION_FAMILY = NOISE_PROFILE_FALSE_POSITIVE

STATUS_READY = "ready"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_BLOCKED = "blocked"

#: Provenance fields a false-positive replay condition must carry so its result
#: can be attributed (issue Definition of Done: scenario, seed, planner mode,
#: perturbation family, execution mode).  ``perturbation_family`` is validated
#: against :data:`FALSE_POSITIVE_PERTURBATION_FAMILY`.
REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "scenario_id",
    "seed",
    "planner_mode",
    "perturbation_family",
    "execution_mode",
)

#: Keys consumed as injected false-positive actor inputs.  These map directly to
#: ``ObservationPerturbationSpec`` constructor arguments so input validation is
#: not re-implemented here.
INJECTION_INPUT_KEYS: tuple[str, ...] = (
    "false_positive_positions",
    "false_positive_velocities",
    "false_positive_ids",
)

OBSERVATION_VIEW_KEYS: tuple[str, ...] = (
    "planner_observation_view",
    "observation_view",
    "observation_contract",
)
OBSERVATION_LEVEL_KEYS: tuple[str, ...] = (
    "observation_level",
    "benchmark_observation_level",
)
OBSERVATION_MODE_KEYS: tuple[str, ...] = (
    "active_observation_mode",
    "observation_mode",
    "planner_observation_mode",
)
REQUIRED_INPUT_KEYS: tuple[str, ...] = (
    "required_inputs",
    "planner_required_inputs",
)
PEDESTRIAN_INPUT_KEYS = frozenset({"pedestrians"})
INCOMPATIBLE_OBSERVATION_MODES = frozenset({"goal_state"})


@dataclass(frozen=True)
class FalsePositiveInjectionReadiness:
    """Structured readiness verdict for one false-positive replay condition.

    Attributes:
        status: One of :data:`STATUS_READY`, :data:`STATUS_NOT_AVAILABLE`, or
            :data:`STATUS_BLOCKED`.
        blockers: Actionable messages naming missing/malformed inputs or
            provenance.  Empty unless ``status == blocked``.
        injected_actor_count: Number of observed-only false-positive actors the
            spec requests (``0`` when not-available or blocked-before-parse).
        noise_profile: Canonical perturbation profile label, or ``None`` when the
            spec could not be constructed.
        provenance: The provenance subset echoed back for the report contract.
    """

    status: str
    blockers: list[str] = field(default_factory=list)
    injected_actor_count: int = 0
    noise_profile: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    observation_view: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Return True only when the replay condition is constructible."""
        return self.status == STATUS_READY

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable readiness payload."""
        return {
            "schema_version": FALSE_POSITIVE_INJECTION_READINESS_SCHEMA_VERSION,
            "issue": FALSE_POSITIVE_INJECTION_REPLAY_READINESS_ISSUE,
            "status": self.status,
            "blockers": list(self.blockers),
            "injected_actor_count": self.injected_actor_count,
            "noise_profile": self.noise_profile,
            "perturbation_family": FALSE_POSITIVE_PERTURBATION_FAMILY,
            "provenance": dict(self.provenance),
            "observation_view": dict(self.observation_view),
        }


def _check_provenance(spec: dict[str, Any]) -> list[str]:
    """Return blocker messages for missing or inconsistent provenance fields."""
    blockers: list[str] = []
    for key in REQUIRED_PROVENANCE_FIELDS:
        value = spec.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            blockers.append(f"missing required provenance field: {key!r}")
    family = spec.get("perturbation_family")
    if family is not None and str(family) != FALSE_POSITIVE_PERTURBATION_FAMILY:
        blockers.append(
            f"perturbation_family must be {FALSE_POSITIVE_PERTURBATION_FAMILY!r}, got {family!r}"
        )
    return blockers


def _provenance_echo(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the provenance subset present in the spec for the report contract."""
    return {key: spec[key] for key in REQUIRED_PROVENANCE_FIELDS if key in spec}


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    """Return first non-empty mapping value for keys."""
    for key in keys:
        value = mapping.get(key)
        if value is not None and not (isinstance(value, str) and not value.strip()):
            return value
    return None


def _list_strings(value: Any) -> list[str]:
    """Normalize scalar/list view metadata into strings.

    Returns:
        Non-empty string values extracted from ``value``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _observation_view(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Extract declared planner observation-view metadata from a readiness spec.

    Returns:
        Normalized observation-level, active-mode, and required-input metadata.
    """
    view: Mapping[str, Any] = {}
    for key in OBSERVATION_VIEW_KEYS:
        value = spec.get(key)
        if isinstance(value, Mapping):
            view = value
            break

    observation_level = _first_present(view, OBSERVATION_LEVEL_KEYS)
    if observation_level is None:
        observation_level = _first_present(spec, OBSERVATION_LEVEL_KEYS)

    observation_mode = _first_present(view, OBSERVATION_MODE_KEYS)
    if observation_mode is None:
        observation_mode = _first_present(spec, OBSERVATION_MODE_KEYS)

    required_inputs = _list_strings(_first_present(view, REQUIRED_INPUT_KEYS))
    if not required_inputs:
        required_inputs = _list_strings(_first_present(spec, REQUIRED_INPUT_KEYS))

    if not required_inputs and observation_level is not None:
        try:
            required_inputs = list(observation_level_spec(str(observation_level)).required_inputs)
        except ValueError:
            required_inputs = []

    return {
        "observation_level": str(observation_level).strip() if observation_level else None,
        "active_observation_mode": str(observation_mode).strip() if observation_mode else None,
        "required_inputs": required_inputs,
    }


def _check_observation_view_compatibility(view: Mapping[str, Any]) -> list[str]:
    """Return blockers when injected actors cannot reach planner observations."""
    required_inputs = set(_list_strings(view.get("required_inputs")))
    observation_mode = str(view.get("active_observation_mode") or "").strip()

    blockers: list[str] = []
    if not any(required in required_inputs for required in PEDESTRIAN_INPUT_KEYS):
        blockers.append(
            "planner observation view must expose structured pedestrians for "
            "false-positive actor injection; add planner_observation_view.required_inputs "
            "including 'pedestrians' or use a compatible planner/config"
        )
    if observation_mode in INCOMPATIBLE_OBSERVATION_MODES:
        blockers.append(
            f"planner observation mode {observation_mode!r} cannot carry structured "
            "injected pedestrians; use a pedestrian-observation mode such as "
            "'socnav_state', 'headed_socnav_state', or 'gst_human_state'"
        )
    return blockers


def check_false_positive_injection_readiness(
    spec: dict[str, Any],
) -> FalsePositiveInjectionReadiness:
    """Fail-closed readiness check for a false-positive actor-injection condition.

    The check is intentionally narrow: it proves a replay *condition spec* could be
    turned into a valid perturbation with attributable provenance.  It never
    executes a replay or asserts a safety outcome.

    Args:
        spec: Replay-condition mapping.  Recognized injection inputs are
            :data:`INJECTION_INPUT_KEYS`; required provenance is
            :data:`REQUIRED_PROVENANCE_FIELDS`.

    Returns:
        A :class:`FalsePositiveInjectionReadiness` verdict.  ``blocked`` whenever
        injection inputs are malformed or provenance is incomplete; ``not_available``
        when no false-positive actors are requested; ``ready`` otherwise.

    Raises:
        TypeError: When ``spec`` is not a mapping.
    """
    if not isinstance(spec, dict):
        raise TypeError("false-positive injection spec must be a mapping")

    blockers: list[str] = []
    provenance = _provenance_echo(spec)
    observation_view = _observation_view(spec)
    blockers.extend(_check_provenance(spec))

    # Reuse the canonical perturbation spec for input validation rather than
    # re-deriving shape rules here.  A ValueError means the injection inputs are
    # malformed and the condition must fail closed.
    injection_kwargs = {key: spec[key] for key in INJECTION_INPUT_KEYS if key in spec}
    perturbation_spec: ObservationPerturbationSpec | None = None
    try:
        perturbation_spec = ObservationPerturbationSpec(**injection_kwargs)
    except (ValueError, TypeError) as exc:
        blockers.append(f"malformed false-positive injection input: {exc}")

    if blockers:
        return FalsePositiveInjectionReadiness(
            status=STATUS_BLOCKED,
            blockers=blockers,
            provenance=provenance,
            observation_view=observation_view,
        )

    assert perturbation_spec is not None  # guaranteed when no blockers recorded
    injected_actor_count = perturbation_spec.false_positive_actor_count
    if injected_actor_count == 0:
        # No false-positive actors requested: explicitly unavailable, mirroring how
        # #2927 recorded the missing dimension.  Not an error, but not evidence.
        return FalsePositiveInjectionReadiness(
            status=STATUS_NOT_AVAILABLE,
            injected_actor_count=0,
            noise_profile=perturbation_spec.noise_profile,
            provenance=provenance,
            observation_view=observation_view,
        )

    blockers.extend(_check_observation_view_compatibility(observation_view))
    if blockers:
        return FalsePositiveInjectionReadiness(
            status=STATUS_BLOCKED,
            blockers=blockers,
            injected_actor_count=injected_actor_count,
            noise_profile=perturbation_spec.noise_profile,
            provenance=provenance,
            observation_view=observation_view,
        )

    return FalsePositiveInjectionReadiness(
        status=STATUS_READY,
        injected_actor_count=injected_actor_count,
        noise_profile=perturbation_spec.noise_profile,
        provenance=provenance,
        observation_view=observation_view,
    )
