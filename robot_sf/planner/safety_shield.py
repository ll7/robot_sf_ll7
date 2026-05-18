"""Prediction-aware safety shield contracts for benchmark instrumentation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

ActionCommand = tuple[float, float]


class SafetyShield(Protocol):
    """Protocol for action filters that expose structured shield decisions."""

    def choose_command_decision(
        self,
        observation: dict[str, Any],
        proposed_command: ActionCommand,
    ) -> ShieldDecision:
        """Return the filtered command plus benchmark-facing shield metadata."""


@dataclass(frozen=True)
class ShieldDecision:
    """One action-filter decision emitted by a safety shield."""

    proposed_action: ActionCommand
    filtered_action: ActionCommand
    decision_label: str
    intervention_reason: str
    violated_constraints: tuple[str, ...] = ()
    prediction_source: str = "unknown"
    prediction_horizon_steps: int | None = None
    prediction_dt: float | None = None
    uncertainty_metadata: dict[str, Any] = field(default_factory=dict)
    calibration_metadata: dict[str, Any] = field(default_factory=dict)
    fallback_controller_state: dict[str, Any] = field(default_factory=dict)
    proposed_evaluation: dict[str, Any] = field(default_factory=dict)
    selected_evaluation: dict[str, Any] = field(default_factory=dict)
    intervened: bool | None = None
    hard_constraint_violation: bool | None = None

    @property
    def override_applied(self) -> bool:
        """Return whether the filtered command differs from the proposed command."""
        return any(
            not math.isclose(float(proposed), float(filtered), rel_tol=1e-9, abs_tol=1e-9)
            for proposed, filtered in zip(self.proposed_action, self.filtered_action, strict=True)
        )

    @property
    def is_intervention(self) -> bool:
        """Return whether this decision represents shield intervention."""
        if self.intervened is not None:
            return bool(self.intervened)
        return self.decision_label not in {"ppo_clear", "ppo_safe", "goal_reached"}

    @property
    def final_constraint_violation(self) -> bool:
        """Return whether the selected action still violates hard shield constraints."""
        if self.hard_constraint_violation is not None:
            return bool(self.hard_constraint_violation)
        safe = self.selected_evaluation.get("safe")
        return bool(safe is False)

    def as_command_result(self) -> tuple[ActionCommand, str]:
        """Return the legacy ``(command, decision_label)`` pair."""
        return self.filtered_action, self.decision_label

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable shield decision payload."""
        return {
            "schema_version": "shield-decision.v1",
            "decision_label": self.decision_label,
            "proposed_action": [float(self.proposed_action[0]), float(self.proposed_action[1])],
            "filtered_action": [float(self.filtered_action[0]), float(self.filtered_action[1])],
            "intervened": self.is_intervention,
            "override_applied": self.override_applied,
            "hard_constraint_violation": self.final_constraint_violation,
            "violated_constraints": list(self.violated_constraints),
            "intervention_reason": self.intervention_reason,
            "prediction": {
                "source": self.prediction_source,
                "horizon_steps": self.prediction_horizon_steps,
                "dt": self.prediction_dt,
                "uncertainty": _sanitize_payload(self.uncertainty_metadata),
                "calibration": _sanitize_payload(self.calibration_metadata),
            },
            "fallback_controller_state": _sanitize_payload(self.fallback_controller_state),
            "proposed_evaluation": _sanitize_payload(self.proposed_evaluation),
            "selected_evaluation": _sanitize_payload(self.selected_evaluation),
        }


def shield_contract_metadata(
    *,
    shield_name: str,
    prediction_source: str,
    fallback_policy: str,
    calibration_status: str = "not_calibrated",
) -> dict[str, Any]:
    """Return stable benchmark metadata for a shield implementation."""
    return {
        "schema_version": "safety-shield-contract.v1",
        "shield_name": shield_name,
        "prediction_source": prediction_source,
        "fallback_policy": fallback_policy,
        "calibration_status": calibration_status,
        "metrics": [
            "shield_intervention_rate",
            "shield_override_rate",
            "shield_hard_constraint_violation_rate",
        ],
        "interpretation": (
            "Benchmark instrumentation for action filtering; not a formal safety certificate."
        ),
    }


def new_shield_stats() -> dict[str, Any]:
    """Return an empty shield statistics accumulator."""
    return {
        "schema_version": "safety-shield-stats.v1",
        "decision_count": 0,
        "pass_through_count": 0,
        "intervention_count": 0,
        "override_count": 0,
        "hard_constraint_violation_count": 0,
        "decision_counts": {},
        "violated_constraint_counts": {},
        "last_decision": None,
    }


def update_shield_stats(stats: dict[str, Any], decision: ShieldDecision) -> dict[str, Any]:
    """Update a shield statistics accumulator in place and return it.

    Returns:
        dict[str, Any]: The updated stats mapping.
    """
    if "schema_version" not in stats:
        stats.update(new_shield_stats())

    stats["decision_count"] = int(stats.get("decision_count", 0)) + 1
    if decision.is_intervention:
        stats["intervention_count"] = int(stats.get("intervention_count", 0)) + 1
    else:
        stats["pass_through_count"] = int(stats.get("pass_through_count", 0)) + 1
    if decision.override_applied:
        stats["override_count"] = int(stats.get("override_count", 0)) + 1
    if decision.final_constraint_violation:
        stats["hard_constraint_violation_count"] = (
            int(stats.get("hard_constraint_violation_count", 0)) + 1
        )

    decision_counts = stats.setdefault("decision_counts", {})
    if isinstance(decision_counts, dict):
        decision_counts[decision.decision_label] = (
            int(decision_counts.get(decision.decision_label, 0)) + 1
        )

    constraint_counts = stats.setdefault("violated_constraint_counts", {})
    if isinstance(constraint_counts, dict):
        for constraint in decision.violated_constraints:
            constraint_counts[constraint] = int(constraint_counts.get(constraint, 0)) + 1

    stats["last_decision"] = decision.to_metadata()
    return stats


def shield_metrics_from_stats(stats: dict[str, Any]) -> dict[str, float]:
    """Return scalar benchmark metrics derived from shield statistics."""
    decisions = int(stats.get("decision_count", 0) or 0)
    interventions = int(stats.get("intervention_count", 0) or 0)
    overrides = int(stats.get("override_count", 0) or 0)
    violations = int(stats.get("hard_constraint_violation_count", 0) or 0)
    denominator = float(decisions) if decisions > 0 else 1.0
    return {
        "shield_decision_count": float(decisions),
        "shield_intervention_count": float(interventions),
        "shield_override_count": float(overrides),
        "shield_hard_constraint_violation_count": float(violations),
        "shield_intervention_rate": float(interventions / denominator) if decisions else 0.0,
        "shield_override_rate": float(overrides / denominator) if decisions else 0.0,
        "shield_hard_constraint_violation_rate": (
            float(violations / denominator) if decisions else 0.0
        ),
    }


def _sanitize_payload(value: Any) -> Any:
    """Return a JSON-friendly payload without NaN or infinite numeric values."""
    if isinstance(value, dict):
        return {
            str(key): cleaned
            for key, item in value.items()
            if (cleaned := _sanitize_payload(item)) is not None
        }
    if isinstance(value, (list, tuple)):
        return [cleaned for item in value if (cleaned := _sanitize_payload(item)) is not None]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


__all__ = [
    "ActionCommand",
    "SafetyShield",
    "ShieldDecision",
    "new_shield_stats",
    "shield_contract_metadata",
    "shield_metrics_from_stats",
    "update_shield_stats",
]
