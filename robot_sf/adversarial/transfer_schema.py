"""Gate A v2 transfer row schema and provenance types.

This module holds the typed row-level contract for issue #6146 Gate A. It is
pure schema: no empirical file I/O, no planner execution, no benchmark claim.
The constraints-first outcome vocabulary mirrors
:mod:`robot_sf.adversarial.objectives` but does not read episode records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConstraintsFirstOutcome:
    """Ordered constraints-first outcome vector for one evaluation.

    Hard constraints are evaluated in fixed order:
    1. collision or severe intrusion;
    2. liveness / goal completion;
    3. comfort / efficiency only after the above pass.

    A comfort or efficiency advantage can never compensate for a failed
    safety or liveness condition.
    """

    collision_or_severe_intrusion: bool | None
    liveness_or_goal_completion: bool | None
    comfort_and_efficiency: dict[str, Any] | None
    status: str = "observed"

    def failed(self) -> bool:
        """Return whether any hard constraint failed.

        Unavailable outcomes are not silently treated as failures; callers
        must reject ``status != "observed"`` rows explicitly.
        """
        if self.status != "observed":
            return False
        return bool(
            self.collision_or_severe_intrusion is True or self.liveness_or_goal_completion is True
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "status": self.status,
            "collision_or_severe_intrusion": self.collision_or_severe_intrusion,
            "liveness_or_goal_completion": self.liveness_or_goal_completion,
            "comfort_and_efficiency": self.comfort_and_efficiency,
        }


@dataclass(frozen=True)
class CandidateProvenance:
    """Immutable candidate identity and certification lineage."""

    source_target_planner: str
    source_campaign_identity: str
    source_candidate_identity: str
    normalized_candidate_hash: str
    certification_hash: str
    recertification_hash: str | None
    scenario_family_hash: str
    scenario_config_hash: str
    execution_commit: str
    execution_context_path: str
    record_hash: str
    admission_status: str
    admission_reason: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "source_target_planner": self.source_target_planner,
            "source_campaign_identity": self.source_campaign_identity,
            "source_candidate_identity": self.source_candidate_identity,
            "normalized_candidate_hash": self.normalized_candidate_hash,
            "certification_hash": self.certification_hash,
            "recertification_hash": self.recertification_hash,
            "scenario_family_hash": self.scenario_family_hash,
            "scenario_config_hash": self.scenario_config_hash,
            "execution_commit": self.execution_commit,
            "execution_context_path": self.execution_context_path,
            "record_hash": self.record_hash,
            "admission_status": self.admission_status,
            "admission_reason": self.admission_reason,
        }


@dataclass(frozen=True)
class PlannerEvalProvenance:
    """Immutable evaluated-planner lineage."""

    evaluated_planner: str
    planner_config_hash: str
    scenario_config_hash: str
    execution_mode: str
    deterministic_replay_lineage: str
    independent_confirmation_lineage: str
    execution_commit: str
    execution_context_path: str
    record_hash: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "evaluated_planner": self.evaluated_planner,
            "planner_config_hash": self.planner_config_hash,
            "scenario_config_hash": self.scenario_config_hash,
            "execution_mode": self.execution_mode,
            "deterministic_replay_lineage": self.deterministic_replay_lineage,
            "independent_confirmation_lineage": self.independent_confirmation_lineage,
            "execution_commit": self.execution_commit,
            "execution_context_path": self.execution_context_path,
            "record_hash": self.record_hash,
        }


@dataclass(frozen=True)
class GateATransferRow:
    """One immutable candidate x evaluated-planner x fresh-seed Gate A v2 row.

    This is the atomic unit of the capability-only transfer contract. Every
    row pins source identity, normalized hashes, evaluated-planner lineage,
    execution mode, the ordered constraints-first outcome vector, mechanism
    retention, and deterministic replay / independent-confirmation lineage.
    """

    config_id: str
    target_planner: str
    evaluated_planner: str
    scenario_seed: int
    eval_seed: int
    candidate_provenance: CandidateProvenance
    planner_provenance: PlannerEvalProvenance
    outcome: ConstraintsFirstOutcome
    robustness_diagnostic: float
    transferred: bool
    mechanism_retained: bool
    primary_mechanism: str
    observed_mechanism: str
    attribution_review_status: str
    lineage_complete: bool
    immutable_record_hash: str

    def to_json(self) -> dict[str, Any]:
        """Return a deterministic JSON-serialisable payload."""
        return {
            "config_id": self.config_id,
            "target_planner": self.target_planner,
            "evaluated_planner": self.evaluated_planner,
            "scenario_seed": self.scenario_seed,
            "eval_seed": self.eval_seed,
            "candidate_provenance": self.candidate_provenance.to_json(),
            "planner_provenance": self.planner_provenance.to_json(),
            "outcome": self.outcome.to_json(),
            "robustness_diagnostic": self.robustness_diagnostic,
            "transferred": self.transferred,
            "mechanism_retained": self.mechanism_retained,
            "primary_mechanism": self.primary_mechanism,
            "observed_mechanism": self.observed_mechanism,
            "attribution_review_status": self.attribution_review_status,
            "lineage_complete": self.lineage_complete,
            "immutable_record_hash": self.immutable_record_hash,
        }


__all__ = [
    "CandidateProvenance",
    "ConstraintsFirstOutcome",
    "GateATransferRow",
    "PlannerEvalProvenance",
]
