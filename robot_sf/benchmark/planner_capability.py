"""Capability-aware planner routing and handoff ledger (schema/smoke only).

Evidence tier: schema/smoke. This module defines a versioned capability ledger
and pure routing/handoff eligibility functions. It performs NO live planner
switching, NO mutation of a running control loop, NO planner-performance
comparison, NO benchmark ranking, and makes NO general capability claim.

Canonical owners (read-only references):
- ``robot_sf/benchmark/algorithm_readiness.py``: planner identity and readiness tier.
- ``robot_sf/benchmark/algorithm_metadata.py``: execution semantics.
- ``robot_sf/planner/planner_selector_v2_diagnostic.py``: auditable selector precedent.

Issue: https://github.com/ll7/robot_sf_ll7/issues/6580
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

PLANNER_CAPABILITY_SCHEMA_VERSION = "planner_capability.v1"

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _validate_repository_relative_refs(
    field_name: str, evidence_refs: tuple[str, ...]
) -> tuple[str, ...]:
    """Return errors for evidence references that are not repository-relative paths."""
    errors: list[str] = []
    if not isinstance(evidence_refs, tuple):
        return (f"{field_name} must be a tuple of repository-relative paths",)
    for index, evidence_ref in enumerate(evidence_refs):
        if not isinstance(evidence_ref, str) or not evidence_ref.strip():
            errors.append(f"{field_name}[{index}] must be a non-empty string")
            continue
        normalized = evidence_ref.strip().replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            normalized.startswith("/")
            or _WINDOWS_ABSOLUTE_PATH.match(normalized)
            or "://" in normalized
            or ".." in path.parts
        ):
            errors.append(
                f"{field_name}[{index}] must be a repository-relative path: {evidence_ref!r}"
            )
    return tuple(errors)


def _merge_evidence_refs(*evidence_ref_groups: tuple[str, ...]) -> tuple[str, ...]:
    """Merge evidence references while preserving first-seen order.

    Returns:
        Tuple of unique references in first-seen order.
    """
    return tuple(dict.fromkeys(ref for group in evidence_ref_groups for ref in group))


@dataclass(frozen=True)
class MeasuredRange:
    """A measured or assumed numeric range with evidence provenance.

    When ``low`` and ``high`` are both ``None`` the range is unknown and must
    not be inferred. Every non-unknown range carries at least one
    repository-relative ``evidence_refs``.
    """

    low: float | None = None
    high: float | None = None
    assumption: bool = False
    evidence_refs: tuple[str, ...] = ()

    @property
    def is_unknown(self) -> bool:
        """Return True when the range has not been measured."""
        return self.low is None and self.high is None

    def validate(self, field_name: str = "range") -> tuple[str, ...]:
        """Return validation errors for the range and its provenance."""
        errors: list[str] = []
        if not isinstance(self.assumption, bool):
            errors.append(f"{field_name}.assumption must be a boolean")

        if self.low is None and self.high is None:
            errors.extend(
                _validate_repository_relative_refs(
                    f"{field_name}.evidence_refs", self.evidence_refs
                )
            )
            return tuple(errors)

        if self.low is None or self.high is None:
            errors.append(f"{field_name} must provide both low and high, or neither")
        else:
            for bound_name, bound in (("low", self.low), ("high", self.high)):
                if isinstance(bound, bool) or not isinstance(bound, int | float):
                    errors.append(f"{field_name}.{bound_name} must be a finite number")
                elif not math.isfinite(float(bound)):
                    errors.append(f"{field_name}.{bound_name} must be a finite number")
            if (
                isinstance(self.low, int | float)
                and not isinstance(self.low, bool)
                and isinstance(self.high, int | float)
                and not isinstance(self.high, bool)
                and math.isfinite(float(self.low))
                and math.isfinite(float(self.high))
                and self.low > self.high
            ):
                errors.append(f"{field_name}.low must not exceed {field_name}.high")

        if not self.evidence_refs:
            errors.append(f"{field_name} is measured but has no evidence_refs")
        else:
            errors.extend(
                _validate_repository_relative_refs(
                    f"{field_name}.evidence_refs", self.evidence_refs
                )
            )
        return tuple(errors)


@dataclass(frozen=True)
class PlannerCapabilityEntry:
    """Versioned capability ledger entry for one planner.

    Every measured field carries at least one repository-relative evidence_ref.
    Assumptions are explicitly marked via ``MeasuredRange.assumption``.
    Non-measured assumptions are named in ``assumption_fields``.
    Unknown ranges are preserved as ``None`` rather than inferred.
    """

    schema_version: str = PLANNER_CAPABILITY_SCHEMA_VERSION
    planner_id: str = ""
    supported_scenarios: tuple[str, ...] = ()
    vehicle_footprints: tuple[str, ...] = ()
    speed_range_mps: MeasuredRange = field(default_factory=MeasuredRange)
    pedestrian_density_range: MeasuredRange = field(default_factory=MeasuredRange)
    required_observations: tuple[str, ...] = ()
    known_failure_signatures: tuple[str, ...] = ()
    required_preconditions: tuple[str, ...] = ()
    handoff_targets: tuple[str, ...] = ()
    assumption_fields: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def validate(self) -> tuple[str, ...]:
        """Return validation errors; empty tuple means valid."""
        errors: list[str] = []
        if not self.planner_id:
            errors.append("planner_id must not be empty")
        if self.schema_version != PLANNER_CAPABILITY_SCHEMA_VERSION:
            errors.append(
                f"unsupported schema_version: {self.schema_version!r}; "
                f"expected {PLANNER_CAPABILITY_SCHEMA_VERSION!r}"
            )
        if not self.evidence_refs:
            errors.append("entry-level evidence_refs must not be empty")
        else:
            errors.extend(_validate_repository_relative_refs("evidence_refs", self.evidence_refs))
        errors.extend(self.speed_range_mps.validate("speed_range_mps"))
        errors.extend(self.pedestrian_density_range.validate("pedestrian_density_range"))
        known_fields = {
            "supported_scenarios",
            "vehicle_footprints",
            "speed_range_mps",
            "pedestrian_density_range",
            "required_observations",
            "known_failure_signatures",
            "required_preconditions",
            "handoff_targets",
        }
        unknown_assumption_fields = set(self.assumption_fields) - known_fields
        if unknown_assumption_fields:
            errors.append(
                "assumption_fields contains unknown fields: "
                f"{tuple(sorted(unknown_assumption_fields))}"
            )
        return tuple(errors)


@dataclass(frozen=True)
class EligibilityResult:
    """Result of an assignment or handoff eligibility check.

    Fail-closed: ``eligible`` is False unless all preconditions are met and
    evidence is present. Handoff results include source and target evidence.
    ``reasons`` explains every rejection.
    """

    eligible: bool
    planner_id: str
    reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    preconditions_checked: tuple[str, ...] = ()
    prior_planner_id: str | None = None


def _CAPABILITY_LEDGER() -> dict[str, PlannerCapabilityEntry]:
    """Return the conservative capability ledger populated from repository metadata.

    Sources:
    - robot_sf/benchmark/algorithm_readiness.py (identity, tier, opt-in)
    - robot_sf/benchmark/algorithm_metadata.py (observation specs, semantics)
    """
    return {
        "goal": PlannerCapabilityEntry(
            planner_id="goal",
            supported_scenarios=("open_space", "corridor", "crossing"),
            assumption_fields=("supported_scenarios",),
            speed_range_mps=MeasuredRange(
                low=0.0,
                high=2.0,
                assumption=True,
                evidence_refs=("robot_sf/benchmark/algorithm_readiness.py",),
            ),
            pedestrian_density_range=MeasuredRange(),
            required_observations=("robot_state", "goal"),
            required_preconditions=("goal_position_available",),
            handoff_targets=("planner_selector_v2_diagnostic",),
            evidence_refs=(
                "robot_sf/benchmark/algorithm_readiness.py",
                "robot_sf/benchmark/algorithm_metadata.py",
            ),
        ),
        "planner_selector_v2_diagnostic": PlannerCapabilityEntry(
            planner_id="planner_selector_v2_diagnostic",
            supported_scenarios=("open_space", "corridor", "bottleneck", "crossing"),
            assumption_fields=("supported_scenarios",),
            speed_range_mps=MeasuredRange(),
            pedestrian_density_range=MeasuredRange(),
            required_observations=("robot_state", "goal", "pedestrians"),
            required_preconditions=(
                "explicit_opt_in",
                "candidate_planners_available",
            ),
            handoff_targets=("goal",),
            evidence_refs=(
                "robot_sf/benchmark/algorithm_readiness.py",
                "robot_sf/benchmark/algorithm_metadata.py",
                "robot_sf/planner/planner_selector_v2_diagnostic.py",
            ),
        ),
    }


def get_capability_entry(planner_id: str) -> PlannerCapabilityEntry | None:
    """Return the capability entry for a planner, or None if not in the ledger."""
    return _CAPABILITY_LEDGER().get(planner_id)


def check_assignment_eligibility(
    *,
    planner_id: str,
    scenario: str,
    preconditions_met: dict[str, bool] | None = None,
) -> EligibilityResult:
    """Pure assignment eligibility check; fails closed.

    Returns ineligible with explicit reasons when:
    - The planner is not in the capability ledger.
    - The entry fails schema validation (missing evidence).
    - The scenario is not in supported_scenarios.
    - A required precondition is not met.

    Returns:
        EligibilityResult with eligible=True only when all checks pass.
    """
    preconditions_met = preconditions_met if preconditions_met is not None else {}
    entry = get_capability_entry(planner_id)

    if entry is None:
        return EligibilityResult(
            eligible=False,
            planner_id=planner_id,
            reasons=(f"planner '{planner_id}' not found in capability ledger",),
        )

    validation_errors = entry.validate()
    if validation_errors:
        return EligibilityResult(
            eligible=False,
            planner_id=planner_id,
            reasons=validation_errors,
            evidence_refs=entry.evidence_refs,
        )

    reasons: list[str] = []
    preconditions_checked: list[str] = []

    if not entry.supported_scenarios:
        reasons.append("supported_scenarios is unknown")
    elif scenario not in entry.supported_scenarios:
        reasons.append(
            f"scenario '{scenario}' not in supported_scenarios {entry.supported_scenarios}"
        )

    for precondition in entry.required_preconditions:
        preconditions_checked.append(precondition)
        if preconditions_met.get(precondition) is not True:
            reasons.append(f"precondition '{precondition}' not met")

    return EligibilityResult(
        eligible=len(reasons) == 0,
        planner_id=planner_id,
        reasons=tuple(reasons),
        evidence_refs=entry.evidence_refs,
        preconditions_checked=tuple(preconditions_checked),
    )


def check_handoff_eligibility(
    *,
    from_planner_id: str,
    to_planner_id: str,
    scenario: str,
    preconditions_met: dict[str, bool] | None = None,
) -> EligibilityResult:
    """Pure handoff eligibility check; fails closed.

    Returns ineligible with explicit reasons when:
    - The source planner is not in the ledger.
    - The target planner is not in the ledger.
    - The source entry fails schema validation.
    - The target is not a declared handoff_target of the source.
    - The target's assignment eligibility fails.

    Returns:
        EligibilityResult with eligible=True only when all checks pass and
        evidence_refs contains both source and target references.
    """
    preconditions_met = preconditions_met if preconditions_met is not None else {}

    source_entry = get_capability_entry(from_planner_id)
    if source_entry is None:
        return EligibilityResult(
            eligible=False,
            planner_id=to_planner_id,
            reasons=(f"source planner '{from_planner_id}' not found in capability ledger",),
            prior_planner_id=from_planner_id,
        )

    target_entry = get_capability_entry(to_planner_id)
    if target_entry is None:
        source_validation_errors = source_entry.validate()
        return EligibilityResult(
            eligible=False,
            planner_id=to_planner_id,
            reasons=source_validation_errors
            + (f"target planner '{to_planner_id}' not found in capability ledger",),
            evidence_refs=source_entry.evidence_refs,
            prior_planner_id=from_planner_id,
        )

    reasons: list[str] = list(source_entry.validate())
    preconditions_checked: list[str] = []

    if to_planner_id not in source_entry.handoff_targets:
        reasons.append(
            f"'{to_planner_id}' not in handoff_targets of '{from_planner_id}' "
            f"(declared: {source_entry.handoff_targets})"
        )

    target_result = check_assignment_eligibility(
        planner_id=to_planner_id,
        scenario=scenario,
        preconditions_met=preconditions_met,
    )
    preconditions_checked.extend(target_result.preconditions_checked)
    reasons.extend(target_result.reasons)

    return EligibilityResult(
        eligible=len(reasons) == 0,
        planner_id=to_planner_id,
        reasons=tuple(reasons),
        evidence_refs=_merge_evidence_refs(source_entry.evidence_refs, target_entry.evidence_refs),
        preconditions_checked=tuple(preconditions_checked),
        prior_planner_id=from_planner_id,
    )


__all__ = [
    "PLANNER_CAPABILITY_SCHEMA_VERSION",
    "EligibilityResult",
    "MeasuredRange",
    "PlannerCapabilityEntry",
    "check_assignment_eligibility",
    "check_handoff_eligibility",
    "get_capability_entry",
]
