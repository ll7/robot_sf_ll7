"""Downstream analysis-eligibility gate for search evaluations (issue #9304).

A successful search can end with an unanalysable result: analysis telemetry is
opt-in, and excluded attempts can vanish instead of being retained with
reasons. This module gives every :class:`CandidateEvaluation` an explicit
machine-readable verdict so the optimizer/archive update path applies the same
eligibility gate as downstream analysis, and every non-analysed attempt
carries a reason.

Eligibility requires all of: an allowed certificate, an executed episode with
a trace artifact, native (non-degraded) execution mode, a scored objective,
and a bound effective-scenario hash. Anything else is ineligible with stable
reason codes; nothing raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_sf.adversarial.config import CandidateEvaluation

ELIGIBLE_EXECUTION_MODES = frozenset({"native"})
SCHEMA_VERSION = "search_analysis_eligibility.v1"


@dataclass(frozen=True)
class EligibilityReceipt:
    """Machine-readable analysis-eligibility verdict for one evaluation."""

    eligible: bool
    reason_codes: list[str] = field(default_factory=list)
    effective_scenario_hash: str | None = None
    trace_present: bool = False
    certificate_ok: bool = False
    execution_mode: str | None = None
    objective_scored: bool = False


def _attribution_details(evaluation: CandidateEvaluation) -> dict[str, object]:
    """Return the attribution details mapping, tolerating missing attribution."""
    attribution = evaluation.failure_attribution
    details = getattr(attribution, "details", None)
    return dict(details) if isinstance(details, dict) else {}


def analysis_eligibility(evaluation: CandidateEvaluation) -> EligibilityReceipt:
    """Judge whether an evaluation is eligible for downstream analysis.

    Every ``False`` contributes a stable reason code so excluded attempts are
    retained with reasons instead of vanishing. Never raises: malformed inputs
    yield ``eligible=False`` with the narrowest applicable reason.
    """
    reasons: list[str] = []
    certificate_ok = bool(getattr(evaluation.certification_status, "passed", False))
    if not certificate_ok:
        reasons.append("certificate_not_allowed")
    trace_present = evaluation.episode_record_path is not None
    if not trace_present:
        reasons.append("trace_missing")
    details = _attribution_details(evaluation)
    execution_mode = details.get("execution_mode")
    execution_mode = str(execution_mode) if execution_mode is not None else None
    if execution_mode not in ELIGIBLE_EXECUTION_MODES:
        reasons.append("execution_mode_not_native" if execution_mode else "execution_mode_unknown")
    objective_scored = evaluation.objective_value is not None
    if not objective_scored:
        reasons.append("objective_unscored")
    if evaluation.effective_scenario_hash is None:
        reasons.append("effective_hash_unbound")
    if evaluation.error is not None:
        reasons.append("evaluation_errored")
    return EligibilityReceipt(
        eligible=not reasons,
        reason_codes=sorted(set(reasons)),
        effective_scenario_hash=evaluation.effective_scenario_hash,
        trace_present=trace_present,
        certificate_ok=certificate_ok,
        execution_mode=execution_mode,
        objective_scored=objective_scored,
    )
