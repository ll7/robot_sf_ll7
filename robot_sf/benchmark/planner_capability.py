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

from dataclasses import dataclass, field

PLANNER_CAPABILITY_SCHEMA_VERSION = "planner_capability.v1"


@dataclass(frozen=True)
class MeasuredRange:
    """A measured or assumed numeric range with evidence provenance.

    When ``low`` and ``high`` are both ``None`` the range is unknown and must
    not be inferred. Every non-unknown range carries at least one
    repository-relative ``evidence_ref``.
    """

    low: float | None = None
    high: float | None = None
    assumption: bool = False
    evidence_refs: tuple[str, ...] = ()

    @property
    def is_unknown(self) -> bool:
        """Return True when the range has not been measured."""
        return self.low is None and self.high is None


@dataclass(frozen=True)
class PlannerCapabilityEntry:
    """Versioned capability ledger entry for one planner.

    Every measured field carries at least one repository-relative evidence_ref.
    Assumptions are explicitly marked via ``MeasuredRange.assumption``.
    Unknown ranges are preserved as ``None`` rather than inferred.
    """

    schema_version: str = PLANNER_CAPABILITY_SCHEMA_VERSION
    planner_id: str = ""
    supported_scenarios: tuple[str, ...] = ()
    speed_range_mps: MeasuredRange = field(default_factory=MeasuredRange)
    pedestrian_density_range: MeasuredRange = field(default_factory=MeasuredRange)
    required_observations: tuple[str, ...] = ()
    required_preconditions: tuple[str, ...] = ()
    handoff_targets: tuple[str, ...] = ()
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
        if not self.speed_range_mps.is_unknown and not self.speed_range_mps.evidence_refs:
            errors.append("speed_range_mps is measured but has no evidence_refs")
        if (
            not self.pedestrian_density_range.is_unknown
            and not self.pedestrian_density_range.evidence_refs
        ):
            errors.append("pedestrian_density_range is measured but has no evidence_refs")
        if not self.evidence_refs:
            errors.append("entry-level evidence_refs must not be empty")
        return tuple(errors)


@dataclass(frozen=True)
class EligibilityResult:
    """Result of an assignment or handoff eligibility check.

    Fail-closed: ``eligible`` is False unless all preconditions are met and
    evidence is present. ``reasons`` explains every rejection.
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

    if entry.supported_scenarios and scenario not in entry.supported_scenarios:
        reasons.append(
            f"scenario '{scenario}' not in supported_scenarios {entry.supported_scenarios}"
        )

    for precondition in entry.required_preconditions:
        preconditions_checked.append(precondition)
        if not preconditions_met.get(precondition, False):
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
    - The target is not a declared handoff_target of the source.
    - The target's assignment eligibility fails.

    Returns:
        EligibilityResult with eligible=True only when all checks pass.
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
        return EligibilityResult(
            eligible=False,
            planner_id=to_planner_id,
            reasons=(f"target planner '{to_planner_id}' not found in capability ledger",),
            evidence_refs=source_entry.evidence_refs,
            prior_planner_id=from_planner_id,
        )

    reasons: list[str] = []
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
        evidence_refs=target_entry.evidence_refs,
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
