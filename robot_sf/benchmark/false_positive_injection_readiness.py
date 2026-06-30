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

from dataclasses import dataclass, field
from typing import Any

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
        )

    return FalsePositiveInjectionReadiness(
        status=STATUS_READY,
        injected_actor_count=injected_actor_count,
        noise_profile=perturbation_spec.noise_profile,
        provenance=provenance,
    )
