"""Gated preparation scaffolding for continuation and transfer experiments.

The live #7381 ruling permits preparation but not an exploratory campaign.  This
module therefore records the four comparison modes, authority prerequisites, and
cost equations without launching a simulator or treating a blocked plan as data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

PLAN_SCHEMA = "continuation_experiment_plan.v1"
BLOCKED_ADMISSION = "BLOCKED_ADMISSION"
READY_NOT_LAUNCHED = "READY_NOT_LAUNCHED"
CONTINUATION_MODES = (
    "full_parent",
    "exact_restart",
    "pose_velocity_only",
    "late_crop",
)


def _nonnegative_finite(value: Any, name: str) -> float:
    """Normalize one cost input before arithmetic and reject unsafe values.

    Returns:
        The finite, non-negative cost.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


@dataclass(frozen=True, slots=True)
class ContinuationExperimentPlan:
    """A frozen, non-executing plan for a gated continuation comparison."""

    plan_id: str
    authority_issue_ids: tuple[str, ...] = ("#7381", "#7383", "#7384", "#7393", "#7394")
    modes: tuple[str, ...] = CONTINUATION_MODES
    independent_parent_holdout: bool = True
    full_static_geometry: bool = True
    all_actors_retained: bool = True
    execution_allowed: bool = False
    status: str = "preparation_only"

    def __post_init__(self) -> None:
        """Validate that the plan cannot silently omit a negative-control mode."""
        if not self.plan_id.strip():
            raise ValueError("plan_id must be non-empty")
        if self.execution_allowed and (
            not self.authority_issue_ids
            or any(
                not isinstance(issue_id, str) or not issue_id.strip()
                for issue_id in self.authority_issue_ids
            )
        ):
            raise ValueError("execution-enabled plans require non-empty authority_issue_ids")
        if tuple(self.modes) != CONTINUATION_MODES:
            raise ValueError(f"modes must be exactly {CONTINUATION_MODES}")
        if not self.independent_parent_holdout:
            raise ValueError("independent parent/seed holdout is required")
        if not self.full_static_geometry or not self.all_actors_retained:
            raise ValueError("preparation plan must retain full geometry and all actors")
        if self.status != "preparation_only":
            raise ValueError("only preparation_only plans are supported in this lane")

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable gated plan."""
        return {
            "schema_version": PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "authority_issue_ids": list(self.authority_issue_ids),
            "modes": list(self.modes),
            "independent_parent_holdout": self.independent_parent_holdout,
            "full_static_geometry": self.full_static_geometry,
            "all_actors_retained": self.all_actors_retained,
            "execution_allowed": self.execution_allowed,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class AdmissionReceipt:
    """Fail-closed result of checking live execution authority."""

    status: str
    blockers: tuple[str, ...]
    required_authority: tuple[str, ...]
    granted_authority: tuple[str, ...]
    next_smallest_step: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe admission receipt."""
        return {
            "status": self.status,
            "blockers": list(self.blockers),
            "required_authority": list(self.required_authority),
            "granted_authority": list(self.granted_authority),
            "next_smallest_step": self.next_smallest_step,
        }


def evaluate_admission(
    plan: ContinuationExperimentPlan, granted_authority: tuple[str, ...] = ()
) -> AdmissionReceipt:
    """Check authority without launching execution.

    Returns:
        A blocked or not-launched admission receipt.
    """
    required = tuple(sorted(set(plan.authority_issue_ids)))
    granted = tuple(sorted(set(granted_authority)))
    missing = tuple(issue for issue in required if issue not in granted)
    blockers = [f"missing_live_authority:{issue}" for issue in missing]
    if plan.execution_allowed and (
        not plan.authority_issue_ids
        or any(
            not isinstance(issue_id, str) or not issue_id.strip()
            for issue_id in plan.authority_issue_ids
        )
    ):
        blockers.append("execution_requires_nonempty_authority_issue_ids")
    if not plan.execution_allowed:
        blockers.append("execution_not_enabled_in_preparation_plan")
    if blockers:
        return AdmissionReceipt(
            status=BLOCKED_ADMISSION,
            blockers=tuple(blockers),
            required_authority=required,
            granted_authority=granted,
            next_smallest_step="obtain applicable live ruling, then rerun admission only",
        )
    return AdmissionReceipt(
        status=READY_NOT_LAUNCHED,
        blockers=(),
        required_authority=required,
        granted_authority=granted,
        next_smallest_step="review the frozen plan before any execution",
    )


def estimate_continuation_cost(
    *,
    parent_steps: int,
    window_steps: int,
    branches: int,
    parent_cost_s: float,
    selection_cost_s: float,
    validation_cost_s: float,
    load_cost_s: float,
    step_cost_s: float,
) -> dict[str, Any]:
    """Compute transparent first-use and repeated-use cost estimates.

    Returns:
        A cost table including a crossover count or ``no_positive_break_even``.
    """
    if min(parent_steps, window_steps, branches) <= 0:
        raise ValueError("steps and branches must be positive")
    costs = {
        "parent_generation_s": _nonnegative_finite(parent_cost_s, "parent_cost_s"),
        "selection_s": _nonnegative_finite(selection_cost_s, "selection_cost_s"),
        "validation_s": _nonnegative_finite(validation_cost_s, "validation_cost_s"),
        "load_s": _nonnegative_finite(load_cost_s, "load_cost_s"),
    }
    step_cost = _nonnegative_finite(step_cost_s, "step_cost_s")
    costs.update(
        {
            "full_branch_s": float(parent_steps * step_cost),
            "window_branch_s": float(window_steps * step_cost),
        }
    )
    full_per_branch = costs["full_branch_s"]
    window_per_branch = costs["load_s"] + costs["window_branch_s"]
    fixed = costs["parent_generation_s"] + costs["selection_s"] + costs["validation_s"]
    savings = full_per_branch - window_per_branch
    crossover = math.ceil(fixed / savings) if savings > 0.0 else None
    costs.update(
        {
            "branches": branches,
            "plain_total_s": float(branches * full_per_branch),
            "window_first_use_total_s": float(fixed + branches * window_per_branch),
            "marginal_savings_per_branch_s": float(savings),
            "crossover_reuse_count": crossover,
            "break_even_status": "positive" if crossover is not None else "no_positive_break_even",
            "equation": (
                "C_plain = K*C(T); C_window = C_parent + C_selection + C_validation "
                "+ K*(C_load + C(W))"
            ),
        }
    )
    return costs


__all__ = [
    "BLOCKED_ADMISSION",
    "CONTINUATION_MODES",
    "PLAN_SCHEMA",
    "READY_NOT_LAUNCHED",
    "AdmissionReceipt",
    "ContinuationExperimentPlan",
    "estimate_continuation_cost",
    "evaluate_admission",
]
