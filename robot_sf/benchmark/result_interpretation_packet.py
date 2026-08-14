"""Bounded result-interpretation packet contract (issue #7029).

Provides a typed, schema-validated container for structured interpretation of
benchmark or diagnostic results.  The packet is a **contract-only** slice: it
describes what a set of evidence answers, how it was computed, and what it
explicitly does *not* claim.  It does not re-run experiments or infer values
from filenames or plots.

The packet preserves existing ``artifact_catalog`` and ``figure_qa`` contracts
by referencing artifacts via ``file_ref`` digests rather than re-registering
them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.benchmark.artifact_catalog import load_artifact_catalog
from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "result_interpretation_packet.v1"
RESULT_INTERPRETATION_PACKET_SCHEMA_VERSION = SCHEMA_VERSION
_SCHEMA_FILE = Path(__file__).with_name("schemas") / "result_interpretation_packet.v1.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_ZERO_SHA256 = "0" * 64
_CLAIM_ESCALATION_PHRASES = (
    "universally superior",
    "proves a causal",
    "establishes a causal",
    "supports a causal",
    "causal effect",
    "real-world superiority",
    "real-world validity",
    "paper-facing claim",
    "dissertation-ready",
)
_VALID_DECISION_OUTCOMES = frozenset(
    {"supported", "not_supported", "inconclusive", "invalid", "unavailable"}
)
_VALID_EVIDENCE_TIERS = frozenset(
    {"smoke_diagnostic", "visualization_fixture", "nominal_benchmark", "paper_grade"}
)
_VALID_ADMISSION_STATES = frozenset(
    {"diagnostic_only", "unavailable_causal_inference", "bounded_simulator_defined", "admitted"}
)
_VALID_TIER_ADMISSION_STATES = {
    "smoke_diagnostic": frozenset({"diagnostic_only"}),
    "visualization_fixture": frozenset({"unavailable_causal_inference"}),
    "nominal_benchmark": frozenset({"bounded_simulator_defined", "admitted"}),
    "paper_grade": frozenset({"admitted"}),
}
_REVIEW_REQUIRED_ADMISSION_STATES = frozenset({"admitted"})
_REVIEW_REQUIRED_EVIDENCE_TIERS = frozenset({"paper_grade"})
_VALID_DESIRABILITY = frozenset(
    {"higher_is_better", "lower_is_better", "target_range", "not_applicable"}
)
_VALID_MISSINGNESS = frozenset({"complete", "partial", "unavailable", "not_imputed"})
_VALID_UNAVAILABLE_HANDLING = frozenset({"fail_closed", "diagnostic_only", "excluded"})
_VALID_EXECUTION_MODES = frozenset(
    {"native", "adapter", "fallback", "degraded", "unavailable", "invalid", "rejected"}
)
_VALID_ACTOR_STATUSES = frozenset({"draft", "reviewed", "final"})
_VALID_CAPTION_STATUSES = frozenset({"observed", "inferred", "unavailable"})
_VALID_COMPARATOR_DIRECTIONS = frozenset(
    {"comparison_minus_reference", "reference_minus_comparison", "not_applicable"}
)
_VALID_FIGURE_ENCODINGS = frozenset({"png", "pdf", "svg", "unavailable"})
_VALID_CAPTION_TEMPLATES = frozenset(
    {"observed_visualization.v1", "unavailable_visualization.v1", "metric_decision.v1"}
)
_FIGURE_SUFFIXES = {"png": ".png", "pdf": ".pdf", "svg": ".svg"}
_VALID_CAPTION_FIELD_REFS = frozenset(
    {
        "packet_id",
        "question.text",
        "evidence.tier",
        "evidence.admission_state",
        "population.total",
        "population.included",
        "population.excluded",
        "execution_mode.counts",
        "estimand_id",
        "analysis_unit",
        "resampling_unit",
        "pairing_key",
        "comparison.direction",
        "claim_boundary.allowed",
        "claim_boundary.forbidden",
        "forbidden_claims",
        "findings",
        "limitations",
    }
)
_VALID_CAPTION_DYNAMIC_FIELDS = {
    "metric": frozenset(
        {"support", "denominator", "effect", "uncertainty", "null_value", "multiplicity"}
    ),
    "decision": frozenset({"outcome", "rationale", "comparator", "effect"}),
    "figure": frozenset({"artifact_id", "visual_contract", "encoding"}),
    "source": frozenset({"sha256", "commit", "tracked_commit", "kind"}),
}
_SCRIPT_PATH_RE = re.compile(r"^(?:\./)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py$")
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SHELL_COMMAND_SEPARATORS = frozenset({"&&", "||", ";", "|", "&"})
_SHELL_WRAPPERS = frozenset({"command", "env", "exec"})
_CAPTION_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ResultInterpretationPacketError(RobotSfError, ValueError):
    """Raised when a result interpretation packet fails validation."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Question:
    """Identity of the interpretive question being answered."""

    question_id: str
    text: str
    issue_refs: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Evidence:
    """Admission identity and tier for the evidence represented by a packet."""

    evidence_id: str
    tier: str
    admission_state: str
    rationale: str


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A tracked source artifact with generation provenance and digest."""

    source_id: str
    path: str
    sha256: str
    kind: str
    commit: str
    tracked_commit: str
    command: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class PopulationAttrition:
    """Execution-mode attrition counts within the population."""

    native: int
    adapter: int
    fallback: int
    degraded: int
    unavailable: int
    invalid: int
    rejected: int


@dataclass(frozen=True, slots=True)
class Population:
    """Population accounting for the interpretation."""

    total: int
    included: int
    excluded: int
    attrition: PopulationAttrition
    exclusion_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ExecutionMode:
    """Declared execution-mode counts and fallback/degraded policy."""

    counts: dict[str, int]
    fallback_permitted: bool = False
    degraded_permitted: bool = False


@dataclass(frozen=True, slots=True)
class Comparator:
    """Reference and comparison identifiers with direction."""

    reference: str
    comparison: str
    direction: str = "not_applicable"


@dataclass(frozen=True, slots=True)
class Estimand:
    """Estimand identity, analysis/resampling units, and pairing."""

    estimand_id: str
    analysis_unit: str
    resampling_unit: str
    description: str = ""
    pairing_key: str | None = None
    clustering_key: str | None = None
    comparator: Comparator | None = None
    contrast_direction: str = ""


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """Uncertainty quantification for a metric."""

    declared: bool
    method: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value_raw: float | None = None
    p_value_adjusted: float | None = None


@dataclass(frozen=True, slots=True)
class Multiplicity:
    """Multiplicity accounting for a metric."""

    declared: bool
    method: str | None = None
    n_comparisons: int | None = None


@dataclass(frozen=True, slots=True)
class MetricEntry:
    """One metric with support, effect, uncertainty, and multiplicity."""

    metric_id: str
    unit: str
    desirability: str
    support: int
    denominator: int
    missingness: str
    unavailable_handling: str = "fail_closed"
    effect: float | None = None
    uncertainty: Uncertainty | None = None
    null_value: float | None = None
    multiplicity: Multiplicity | None = None
    sensitivity: list[str] | None = None
    support_threshold: int | None = None


@dataclass(frozen=True, slots=True)
class DecisionEntry:
    """A controlled vocabulary decision for a metric."""

    decision_id: str
    metric_id: str
    outcome: str
    rationale: str
    comparator: Comparator | None = None
    contrast_result: ContrastResult | None = None
    effect: float | None = None
    refusal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ContrastResult:
    """Decision-specific statistical result bound to one comparator."""

    comparator: Comparator
    effect: float
    support: int
    denominator: int
    support_threshold: int
    null_value: float
    uncertainty: Uncertainty
    multiplicity: Multiplicity


@dataclass(frozen=True, slots=True)
class FileRef:
    """Checksum-bound file reference."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactCatalogRef:
    """Tracked artifact catalog that owns an available figure output."""

    catalog_id: str
    path: str
    sha256: str
    commit: str


@dataclass(frozen=True, slots=True)
class FigureVisualContract:
    """Explicit visual grammar for a figure or unavailable figure slot."""

    estimand_id: str
    plot_type: str
    rationale: str
    encodings: dict[str, str]
    transforms: list[str] = field(default_factory=list)
    limits: dict[str, str] = field(default_factory=dict)
    reference_lines: list[str] = field(default_factory=list)
    ordering: list[str] = field(default_factory=list)
    faceting: list[str] = field(default_factory=list)
    uncertainty_encoding: str | None = None
    sample_size_display: str | None = None
    legend_identities: list[str] = field(default_factory=list)
    accessibility_contract: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FigureLink:
    """Link to a figure artifact and its explicit visual contract."""

    figure_id: str
    artifact_id: str
    path: str
    sha256: str
    encoding: str
    visual_contract: FigureVisualContract
    caption_file: FileRef | None = None
    artifact_catalog: ArtifactCatalogRef | None = None


@dataclass(frozen=True, slots=True)
class CaptionAssertion:
    """A structured caption assertion bound to figure fields."""

    figure_id: str
    template_id: str
    assertion_text: str
    status: str
    bound_to_packet_fields: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ClaimBoundary:
    """Allowed and forbidden claim statements."""

    allowed: list[str]
    forbidden: list[str]


@dataclass(frozen=True, slots=True)
class ActorRef:
    """Producer or reviewer identity."""

    actor_id: str
    commit: str = ""
    command: str = ""
    status: str = "draft"


@dataclass(frozen=True, slots=True)
class ResultInterpretationPacket:
    """Typed ``result_interpretation_packet.v1`` payload."""

    schema_version: str
    packet_id: str
    question: Question
    evidence: Evidence
    sources: list[SourceRef]
    population: Population
    execution_mode: ExecutionMode
    estimand: Estimand
    metrics: list[MetricEntry]
    decisions: list[DecisionEntry]
    figure_links: list[FigureLink]
    caption_assertions: list[CaptionAssertion]
    claim_boundary: ClaimBoundary
    producer: ActorRef
    findings: list[str]
    limitations: list[str]
    fail_closed_changes: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    reviewer: ActorRef | None = None
    reviewed_packet_digest: str | None = None
    post_review_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe primitives.

        Returns:
            A JSON-compatible dictionary representation.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def load_schema() -> dict[str, Any]:
    """Load the result interpretation packet JSON Schema.

    Returns:
        The parsed JSON Schema dictionary.
    """
    return json.loads(_SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_schema_is_valid() -> None:
    """Assert the schema file is itself a valid Draft 2020-12 schema."""
    Draft202012Validator.check_schema(load_schema())


# ---------------------------------------------------------------------------
# Semantic validation helpers
# ---------------------------------------------------------------------------


def _check_id_uniqueness(
    items: list[Any],
    id_field: str,
    label: str,
    errors: list[str],
) -> None:
    seen: set[str] = set()
    for item in items:
        val = getattr(item, id_field, None)
        if val is None:
            continue
        if val in seen:
            errors.append(f"duplicate {label} id: {val!r}")
        seen.add(val)


def _validate_population(p: Population, errors: list[str]) -> None:
    if p.included + p.excluded != p.total:
        errors.append(
            f"population included ({p.included}) + excluded ({p.excluded}) != total ({p.total})"
        )
    a = p.attrition
    mode_sum = (
        a.native + a.adapter + a.fallback + a.degraded + a.unavailable + a.invalid + a.rejected
    )
    if mode_sum != p.excluded:
        errors.append(f"attrition sum ({mode_sum}) != excluded ({p.excluded})")


def _validate_evidence(evidence: Evidence, errors: list[str]) -> None:
    """Require controlled evidence identity and a compatible admission state."""

    for field_name in ("evidence_id", "tier", "admission_state", "rationale"):
        if not getattr(evidence, field_name).strip():
            errors.append(f"evidence.{field_name} is required")
    if evidence.tier not in _VALID_EVIDENCE_TIERS:
        errors.append(f"evidence.tier {evidence.tier!r} not in {sorted(_VALID_EVIDENCE_TIERS)}")
    if evidence.admission_state not in _VALID_ADMISSION_STATES:
        errors.append(
            "evidence.admission_state "
            f"{evidence.admission_state!r} not in {sorted(_VALID_ADMISSION_STATES)}"
        )
    allowed_states = _VALID_TIER_ADMISSION_STATES.get(evidence.tier)
    if allowed_states is not None and evidence.admission_state not in allowed_states:
        errors.append(
            f"evidence tier {evidence.tier!r} cannot use admission state "
            f"{evidence.admission_state!r}; expected one of {sorted(allowed_states)}"
        )


def _validate_execution_mode(mode: ExecutionMode, errors: list[str]) -> None:
    """Validate mode names and ensure execution counts reconcile to the population."""

    unknown = set(mode.counts).difference(_VALID_EXECUTION_MODES)
    if unknown:
        errors.append(f"execution_mode has unsupported modes: {sorted(unknown)}")
    if any(count < 0 for count in mode.counts.values()):
        errors.append("execution_mode counts must be non-negative")
    if sum(mode.counts.values()) == 0:
        errors.append("execution_mode counts must contain at least one positive count")
    if mode.counts.get("fallback", 0) > 0 and not mode.fallback_permitted:
        errors.append("execution_mode fallback rows require fallback_permitted=true")
    if mode.counts.get("degraded", 0) > 0 and not mode.degraded_permitted:
        errors.append("execution_mode degraded rows require degraded_permitted=true")
    for excluded_mode in ("unavailable", "invalid", "rejected"):
        if mode.counts.get(excluded_mode, 0) > 0:
            errors.append(f"execution_mode {excluded_mode} rows cannot be included")


def _validate_estimator(est: Estimand, errors: list[str]) -> None:
    if not est.analysis_unit:
        errors.append("estimand.analysis_unit is required")
    if not est.resampling_unit:
        errors.append("estimand.resampling_unit is required")
    if est.comparator and est.comparator.direction not in _VALID_COMPARATOR_DIRECTIONS:
        errors.append(
            f"comparator.direction {est.comparator.direction!r} not in "
            f"{sorted(_VALID_COMPARATOR_DIRECTIONS)}"
        )


def _validate_uncertainty(
    metric_id: str,
    uncertainty: Uncertainty | None,
    errors: list[str],
) -> None:
    if uncertainty is None or not uncertainty.declared:
        return
    if not uncertainty.method:
        errors.append(f"metric {metric_id!r}: uncertainty declared but method is null")
    _validate_uncertainty_values(metric_id, uncertainty, errors)
    if uncertainty.ci_low is not None and uncertainty.ci_high is not None:
        if uncertainty.ci_low > uncertainty.ci_high:
            errors.append(f"metric {metric_id!r}: uncertainty ci_low exceeds ci_high")


def _validate_uncertainty_values(
    metric_id: str,
    uncertainty: Uncertainty,
    errors: list[str],
) -> None:
    for value_name, value in (
        ("ci_low", uncertainty.ci_low),
        ("ci_high", uncertainty.ci_high),
        ("p_value_raw", uncertainty.p_value_raw),
        ("p_value_adjusted", uncertainty.p_value_adjusted),
    ):
        if value is not None and not math.isfinite(value):
            errors.append(f"metric {metric_id!r}: {value_name} must be finite")
        elif value is not None and value_name.startswith("p_") and not 0.0 <= value <= 1.0:
            errors.append(f"metric {metric_id!r}: {value_name} must be in [0, 1]")


def _validate_multiplicity(
    metric_id: str,
    multiplicity: Multiplicity | None,
    errors: list[str],
) -> None:
    if multiplicity is None or not multiplicity.declared:
        return
    if not multiplicity.method:
        errors.append(f"metric {metric_id!r}: multiplicity declared but method is null")
    if multiplicity.n_comparisons is None:
        errors.append(f"metric {metric_id!r}: multiplicity declared but n_comparisons is null")


def _validate_metric_accounting(m: MetricEntry, errors: list[str]) -> None:
    if m.denominator <= 0:
        errors.append(f"metric {m.metric_id!r}: denominator must be > 0")
    if m.support < 0:
        errors.append(f"metric {m.metric_id!r}: support must be >= 0")
    if m.support > m.denominator:
        errors.append(
            f"metric {m.metric_id!r}: support ({m.support}) > denominator ({m.denominator})"
        )
    if m.support_threshold is not None:
        if m.support_threshold < 1:
            errors.append(f"metric {m.metric_id!r}: support_threshold must be >= 1")
        if m.support_threshold > m.denominator:
            errors.append(f"metric {m.metric_id!r}: support_threshold exceeds denominator")


def _validate_metric_values(m: MetricEntry, errors: list[str]) -> None:
    for value_name, value in (("effect", m.effect), ("null_value", m.null_value)):
        if value is not None and not math.isfinite(value):
            errors.append(f"metric {m.metric_id!r}: {value_name} must be finite")


def _validate_metric_vocabulary(m: MetricEntry, errors: list[str]) -> None:
    if m.missingness not in _VALID_MISSINGNESS:
        errors.append(
            f"metric {m.metric_id!r}: missingness {m.missingness!r} not in "
            f"{sorted(_VALID_MISSINGNESS)}"
        )
    if m.unavailable_handling not in _VALID_UNAVAILABLE_HANDLING:
        errors.append(
            f"metric {m.metric_id!r}: unavailable_handling "
            f"{m.unavailable_handling!r} not in {sorted(_VALID_UNAVAILABLE_HANDLING)}"
        )
    if m.desirability not in _VALID_DESIRABILITY:
        errors.append(
            f"metric {m.metric_id!r}: desirability {m.desirability!r} not in "
            f"{sorted(_VALID_DESIRABILITY)}"
        )
    if m.missingness == "not_imputed":
        errors.append(
            f"metric {m.metric_id!r}: missingness 'not_imputed' is not allowed; "
            "use 'unavailable' or 'excluded'"
        )


def _validate_metric(m: MetricEntry, errors: list[str]) -> None:
    _validate_metric_accounting(m, errors)
    _validate_metric_values(m, errors)
    _validate_metric_vocabulary(m, errors)
    _validate_uncertainty(m.metric_id, m.uncertainty, errors)
    _validate_multiplicity(m.metric_id, m.multiplicity, errors)


def _validate_supported_decision(
    d: DecisionEntry,
    metric: MetricEntry,
    execution_mode: ExecutionMode,
    errors: list[str],
) -> None:
    contrast = _validate_supported_contrast(d, errors)
    requirements = (
        (
            metric.unavailable_handling != "fail_closed",
            f"decision {d.decision_id!r}: supported outcome requires fail_closed metric handling",
        ),
        (
            metric.uncertainty is None or not metric.uncertainty.declared,
            f"decision {d.decision_id!r}: supported outcome requires declared uncertainty",
        ),
        (
            metric.multiplicity is None or not metric.multiplicity.declared,
            f"decision {d.decision_id!r}: supported outcome requires declared multiplicity",
        ),
        (
            metric.missingness != "complete",
            f"decision {d.decision_id!r}: supported outcome requires complete metric data",
        ),
        (
            metric.null_value is None,
            f"decision {d.decision_id!r}: supported outcome requires a declared null_value",
        ),
        (
            d.comparator is None or d.comparator.direction == "not_applicable",
            f"decision {d.decision_id!r}: supported outcome requires a directed comparator",
        ),
        (
            d.effect is None,
            f"decision {d.decision_id!r}: supported outcome requires an effect",
        ),
    )
    for invalid, message in requirements:
        if invalid:
            errors.append(message)
    if metric.uncertainty is not None and metric.uncertainty.declared:
        if not any(
            value is not None
            for value in (
                metric.uncertainty.ci_low,
                metric.uncertainty.ci_high,
                metric.uncertainty.p_value_raw,
                metric.uncertainty.p_value_adjusted,
            )
        ):
            errors.append(
                f"decision {d.decision_id!r}: supported outcome requires observed uncertainty values"
            )
    if contrast is not None and contrast.support < contrast.support_threshold:
        errors.append(
            f"decision {d.decision_id!r}: contrast_result support is below the declared "
            "support_threshold"
        )
    if metric.support_threshold is None:
        errors.append(f"decision {d.decision_id!r}: supported outcome requires support_threshold")
    elif metric.support < metric.support_threshold:
        errors.append(
            f"decision {d.decision_id!r}: support is below the declared support_threshold"
        )
    if execution_mode.counts.get("fallback", 0) or execution_mode.counts.get("degraded", 0):
        errors.append(
            f"decision {d.decision_id!r}: fallback/degraded rows cannot support a success outcome"
        )
    if d.refusal_reason is not None:
        errors.append(f"decision {d.decision_id!r}: supported outcome cannot have refusal_reason")
    _validate_claim_escalation(d.decision_id, d.rationale, errors)


def _validate_supported_contrast(
    decision: DecisionEntry,
    errors: list[str],
) -> ContrastResult | None:
    """Validate and return the decision-specific contrast binding.

    Returns:
        The validated contrast result, or ``None`` when the binding is absent.
    """

    contrast = decision.contrast_result
    if contrast is None:
        errors.append(
            f"decision {decision.decision_id!r}: supported outcome requires "
            "decision-level contrast_result"
        )
        return None
    _validate_contrast_result(decision.decision_id, contrast, errors)
    if decision.comparator != contrast.comparator:
        errors.append(
            f"decision {decision.decision_id!r}: comparator must match contrast_result.comparator"
        )
    if decision.effect != contrast.effect:
        errors.append(
            f"decision {decision.decision_id!r}: effect must match contrast_result.effect"
        )
    if not contrast.uncertainty.declared or not any(
        value is not None
        for value in (
            contrast.uncertainty.ci_low,
            contrast.uncertainty.ci_high,
            contrast.uncertainty.p_value_raw,
            contrast.uncertainty.p_value_adjusted,
        )
    ):
        errors.append(
            f"decision {decision.decision_id!r}: contrast_result requires "
            "observed uncertainty values"
        )
    if not contrast.multiplicity.declared:
        errors.append(
            f"decision {decision.decision_id!r}: contrast_result requires declared multiplicity"
        )
    return contrast


def _validate_contrast_result(
    decision_id: str,
    contrast: ContrastResult,
    errors: list[str],
) -> None:
    """Validate decision-specific result accounting and inferential fields."""

    if contrast.comparator.direction not in _VALID_COMPARATOR_DIRECTIONS:
        errors.append(
            f"decision {decision_id!r}: contrast comparator direction "
            f"{contrast.comparator.direction!r} is unsupported"
        )
    if contrast.denominator <= 0:
        errors.append(f"decision {decision_id!r}: contrast denominator must be > 0")
    if contrast.support < 0:
        errors.append(f"decision {decision_id!r}: contrast support must be >= 0")
    if contrast.support > contrast.denominator:
        errors.append(
            f"decision {decision_id!r}: contrast support ({contrast.support}) > "
            f"denominator ({contrast.denominator})"
        )
    if contrast.support_threshold < 1:
        errors.append(f"decision {decision_id!r}: contrast support_threshold must be >= 1")
    if contrast.support_threshold > contrast.denominator:
        errors.append(f"decision {decision_id!r}: contrast support_threshold exceeds denominator")
    if not math.isfinite(contrast.effect):
        errors.append(f"decision {decision_id!r}: contrast effect must be finite")
    if not math.isfinite(contrast.null_value):
        errors.append(f"decision {decision_id!r}: contrast null_value must be finite")
    _validate_uncertainty(decision_id, contrast.uncertainty, errors)
    _validate_multiplicity(decision_id, contrast.multiplicity, errors)


def _validate_decision(
    d: DecisionEntry,
    metrics_by_id: dict[str, MetricEntry],
    execution_mode: ExecutionMode,
    errors: list[str],
) -> None:
    if d.outcome not in _VALID_DECISION_OUTCOMES:
        errors.append(
            f"decision {d.decision_id!r}: outcome {d.outcome!r} not in "
            f"{sorted(_VALID_DECISION_OUTCOMES)}"
        )
        return
    metric = metrics_by_id.get(d.metric_id)
    if metric is None:
        return
    if d.effect is not None and not math.isfinite(d.effect):
        errors.append(f"decision {d.decision_id!r}: effect must be finite")
    _validate_claim_escalation(f"decision {d.decision_id} rationale", d.rationale, errors)
    if d.outcome == "supported":
        _validate_supported_decision(d, metric, execution_mode, errors)
    elif not d.refusal_reason:
        errors.append(f"decision {d.decision_id!r}: non-supported outcome requires refusal_reason")


def _validate_claim_boundary(
    cb: ClaimBoundary,
    forbidden_claims: list[str],
    errors: list[str],
) -> None:
    if not cb.allowed:
        errors.append("claim_boundary.allowed must have at least one entry")
    if not cb.forbidden:
        errors.append("claim_boundary.forbidden must have at least one entry")
    if set(cb.allowed).intersection(cb.forbidden):
        errors.append("claim_boundary.allowed and forbidden must be disjoint")
    if forbidden_claims != cb.forbidden:
        errors.append("forbidden_claims must exactly match claim_boundary.forbidden")
    for index, claim in enumerate(cb.allowed):
        _validate_claim_escalation(f"claim_boundary.allowed[{index}]", claim, errors)


def _validate_claim_escalation(label: str, text: str, errors: list[str]) -> None:
    """Reject a small, explicit set of positive high-risk claim phrases."""

    lowered = text.casefold()
    matches = [phrase for phrase in _CLAIM_ESCALATION_PHRASES if phrase in lowered]
    if matches:
        errors.append(f"{label}: forbidden positive claim phrase(s): {sorted(matches)}")


def _validate_caption_assertions(  # noqa: C901
    captions: list[CaptionAssertion],
    packet: ResultInterpretationPacket,
    errors: list[str],
) -> None:
    figure_ids = {fl.figure_id for fl in packet.figure_links}
    figures_by_id = {fl.figure_id: fl for fl in packet.figure_links}
    for ca in captions:
        if ca.figure_id not in figure_ids:
            errors.append(f"caption assertion for {ca.figure_id!r} references an undeclared figure")
        if ca.template_id not in _VALID_CAPTION_TEMPLATES:
            errors.append(
                f"caption assertion for {ca.figure_id!r}: template_id {ca.template_id!r} "
                f"not in {sorted(_VALID_CAPTION_TEMPLATES)}"
            )
        if ca.status not in _VALID_CAPTION_STATUSES:
            errors.append(
                f"caption assertion for {ca.figure_id!r}: status {ca.status!r} "
                f"not in {sorted(_VALID_CAPTION_STATUSES)}"
            )
        if ca.status == "inferred":
            errors.append(
                f"caption assertion for {ca.figure_id!r}: 'inferred' status is "
                "forbidden; use 'observed' or 'unavailable'"
            )
        figure = figures_by_id.get(ca.figure_id)
        if figure is not None:
            if figure.encoding == "unavailable" and (
                ca.template_id != "unavailable_visualization.v1" or ca.status != "unavailable"
            ):
                errors.append(
                    f"caption assertion for unavailable figure {ca.figure_id!r} must use "
                    "unavailable_visualization.v1 with status 'unavailable'"
                )
            if (
                figure.encoding != "unavailable"
                and ca.template_id == "unavailable_visualization.v1"
            ):
                errors.append(
                    f"caption assertion for available figure {ca.figure_id!r} cannot use "
                    "unavailable_visualization.v1"
                )
        if not ca.bound_to_packet_fields:
            errors.append(
                f"caption assertion for {ca.figure_id!r}: bound_to_packet_fields is required"
            )
        for field_ref in ca.bound_to_packet_fields:
            if not _caption_field_ref_exists(field_ref, packet):
                errors.append(
                    f"caption assertion for {ca.figure_id!r}: unknown packet field {field_ref!r}"
                )
        if ca.status == "observed":
            _validate_claim_escalation(
                f"caption assertion for {ca.figure_id!r}", ca.assertion_text, errors
            )
        expected_text = _render_caption_assertion(ca, packet)
        if expected_text is None:
            errors.append(
                f"caption assertion for {ca.figure_id!r}: template {ca.template_id!r} "
                "does not have sufficient structured bindings"
            )
        elif ca.assertion_text != expected_text:
            errors.append(
                f"caption assertion for {ca.figure_id!r}: assertion_text must equal the "
                f"generated {ca.template_id} text"
            )


def _render_caption_assertion(
    assertion: CaptionAssertion,
    packet: ResultInterpretationPacket,
) -> str | None:
    """Render the controlled caption grammar for one structured assertion.

    Returns:
        Generated caption text, or ``None`` when the template lacks enough
        structured bindings to render safely.
    """

    if assertion.template_id == "observed_visualization.v1":
        figure = next(
            (item for item in packet.figure_links if item.figure_id == assertion.figure_id),
            None,
        )
        if figure is None or figure.encoding == "unavailable" or assertion.status != "observed":
            return None
        required_fields = {
            "estimand_id",
            "comparison.direction",
            "claim_boundary.allowed",
        }
        if not required_fields.issubset(assertion.bound_to_packet_fields):
            return None
        direction = (
            packet.estimand.comparator.direction
            if packet.estimand.comparator is not None
            else "not_applicable"
        )
        return (
            f"Observed figure '{assertion.figure_id}' for estimand "
            f"'{packet.estimand.estimand_id}' with direction '{direction}'."
        )
    if assertion.template_id == "unavailable_visualization.v1":
        figure = next(
            (item for item in packet.figure_links if item.figure_id == assertion.figure_id),
            None,
        )
        if figure is None or figure.encoding != "unavailable" or assertion.status != "unavailable":
            return None
        return (
            f"Figure '{assertion.figure_id}' is unavailable under admission state "
            f"'{packet.evidence.admission_state}'."
        )
    if assertion.template_id == "metric_decision.v1":
        decision_refs = [
            field_ref
            for field_ref in assertion.bound_to_packet_fields
            if field_ref.startswith("decision.") and field_ref.endswith(".outcome")
        ]
        if len(decision_refs) != 1:
            return None
        decision_id = decision_refs[0].split(".")[1]
        decision = next(
            (item for item in packet.decisions if item.decision_id == decision_id), None
        )
        if decision is None:
            return None
        return (
            f"Decision '{decision.decision_id}' for metric '{decision.metric_id}' is "
            f"'{decision.outcome}'."
        )
    return None


def _caption_field_ref_exists(field_ref: str, packet: ResultInterpretationPacket) -> bool:
    """Return whether a caption binding names a known packet field."""

    if field_ref in _VALID_CAPTION_FIELD_REFS:
        return True
    parts = field_ref.split(".")
    if len(parts) != 3 or parts[0] not in _VALID_CAPTION_DYNAMIC_FIELDS:
        return False
    if (
        not _CAPTION_ID_RE.match(parts[1])
        or parts[2] not in _VALID_CAPTION_DYNAMIC_FIELDS[parts[0]]
    ):
        return False
    identifiers = {
        "metric": {metric.metric_id for metric in packet.metrics},
        "decision": {decision.decision_id for decision in packet.decisions},
        "figure": {figure.figure_id for figure in packet.figure_links},
        "source": {source.source_id for source in packet.sources},
    }
    return parts[1] in identifiers[parts[0]]


def _validate_figure_links(
    figures: list[FigureLink],
    estimand_id: str,
    errors: list[str],
) -> None:
    for fl in figures:
        if fl.visual_contract.estimand_id != estimand_id:
            errors.append(
                f"figure {fl.figure_id!r}: visual contract estimand_id "
                f"{fl.visual_contract.estimand_id!r} does not match estimand {estimand_id!r}"
            )
        if fl.encoding not in _VALID_FIGURE_ENCODINGS:
            errors.append(
                f"figure {fl.figure_id!r}: encoding {fl.encoding!r} not in "
                f"{sorted(_VALID_FIGURE_ENCODINGS)}"
            )
        if fl.encoding != "unavailable" and not _SHA256_RE.match(fl.sha256):
            errors.append(f"figure {fl.figure_id!r}: sha256 must be 64-hex for an available figure")
        if fl.encoding != "unavailable" and fl.sha256 == _ZERO_SHA256:
            errors.append(f"figure {fl.figure_id!r}: available figures require a non-zero sha256")
        _validate_file_ref(
            fl.path,
            fl.sha256,
            f"figure {fl.figure_id!r}",
            errors,
            require_exists=fl.encoding != "unavailable",
            verify_digest=fl.encoding != "unavailable",
        )
        if fl.encoding != "unavailable":
            _validate_figure_file_type(fl, errors)
            if fl.artifact_catalog is None:
                errors.append(
                    f"figure {fl.figure_id!r}: available figures require an artifact_catalog ref"
                )
            else:
                _validate_artifact_catalog_binding(fl, errors)
        if fl.caption_file is not None:
            _validate_file_ref(
                fl.caption_file.path,
                fl.caption_file.sha256,
                f"figure {fl.figure_id!r} caption_file",
                errors,
                require_exists=True,
                verify_digest=True,
            )
            _validate_caption_file_type(fl, errors)


def _validate_figure_file_type(figure: FigureLink, errors: list[str]) -> None:
    """Require a rendered figure's suffix and magic bytes to match its encoding."""

    path = _REPO_ROOT / figure.path
    if not path.is_file():
        return
    expected_suffix = _FIGURE_SUFFIXES.get(figure.encoding)
    if expected_suffix is None:
        return
    if path.suffix.casefold() != expected_suffix:
        errors.append(
            f"figure {figure.figure_id!r}: path suffix {path.suffix!r} does not match "
            f"encoding {figure.encoding!r}"
        )
        return
    prefix = path.read_bytes()[:32]
    try:
        svg_text = path.read_text(encoding="utf-8")[:4096] if figure.encoding == "svg" else ""
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"figure {figure.figure_id!r}: cannot read SVG bytes: {exc}")
        return
    signature_matches = {
        "png": prefix.startswith(b"\x89PNG\r\n\x1a\n"),
        "pdf": prefix.startswith(b"%PDF-"),
        "svg": bool(re.search(r"<svg(?:\s|>)", svg_text, re.I)),
    }
    if not signature_matches[figure.encoding]:
        errors.append(
            f"figure {figure.figure_id!r}: bytes do not match declared encoding {figure.encoding!r}"
        )


def _validate_caption_file_type(figure: FigureLink, errors: list[str]) -> None:
    """Require caption references to be durable UTF-8 text files."""

    if figure.caption_file is None:
        return
    path = _REPO_ROOT / figure.caption_file.path
    if path.suffix.casefold() not in {".md", ".markdown", ".txt"}:
        errors.append(
            f"figure {figure.figure_id!r} caption_file must be Markdown or plain text: "
            f"{figure.caption_file.path}"
        )
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"figure {figure.figure_id!r} caption_file is not readable UTF-8 text: {exc}")
        return
    if not text.strip():
        errors.append(f"figure {figure.figure_id!r} caption_file must not be empty")


def _validate_artifact_catalog_binding(figure: FigureLink, errors: list[str]) -> None:  # noqa: C901, PLR0912
    """Bind an available figure and caption to an existing artifact catalog entry."""

    catalog_ref = figure.artifact_catalog
    if catalog_ref is None:
        return
    catalog_path = _REPO_ROOT / catalog_ref.path
    _validate_file_ref(
        catalog_ref.path,
        catalog_ref.sha256,
        f"figure {figure.figure_id!r} artifact_catalog",
        errors,
        require_exists=True,
        verify_digest=True,
    )
    if not _COMMIT_RE.match(catalog_ref.commit):
        errors.append(
            f"figure {figure.figure_id!r} artifact_catalog commit is not a hexadecimal git revision"
        )
    elif not _git_commit_exists(catalog_ref.commit):
        errors.append(
            f"figure {figure.figure_id!r} artifact_catalog commit is unavailable: "
            f"{catalog_ref.commit}"
        )
    else:
        tracked_digest = _git_file_sha256(catalog_ref.commit, catalog_ref.path)
        if tracked_digest != catalog_ref.sha256:
            errors.append(
                f"figure {figure.figure_id!r} artifact_catalog is not bound to commit "
                f"{catalog_ref.commit}"
            )
    if not catalog_path.is_file():
        return
    try:
        catalog = load_artifact_catalog(catalog_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"figure {figure.figure_id!r} artifact_catalog could not be loaded: {exc}")
        return
    if catalog.catalog_id != catalog_ref.catalog_id:
        errors.append(
            f"figure {figure.figure_id!r}: artifact_catalog id {catalog.catalog_id!r} does not "
            f"match declared {catalog_ref.catalog_id!r}"
        )
    entries = [entry for entry in catalog.artifacts if entry.artifact_id == figure.artifact_id]
    if len(entries) != 1:
        errors.append(
            f"figure {figure.figure_id!r}: artifact_id {figure.artifact_id!r} must identify "
            "exactly one catalog entry"
        )
        return
    entry = entries[0]
    if entry.artifact_kind != "figure":
        errors.append(
            f"figure {figure.figure_id!r}: catalog entry must have artifact_kind 'figure'"
        )
    output = entry.outputs.get(figure.encoding)
    if output is None:
        errors.append(
            f"figure {figure.figure_id!r}: catalog entry has no {figure.encoding!r} output"
        )
    else:
        expected_path = (catalog_path.parent / output.path).resolve()
        actual_path = (_REPO_ROOT / figure.path).resolve()
        if expected_path != actual_path:
            errors.append(
                f"figure {figure.figure_id!r}: path does not match the catalog output path"
            )
        if output.sha256 != figure.sha256:
            errors.append(f"figure {figure.figure_id!r}: sha256 does not match the catalog output")
        tracked_figure_digest = _git_file_sha256(catalog_ref.commit, figure.path)
        if tracked_figure_digest is None:
            errors.append(
                f"figure {figure.figure_id!r}: catalog commit does not contain the figure bytes"
            )
        elif tracked_figure_digest != figure.sha256:
            errors.append(
                f"figure {figure.figure_id!r}: catalog commit figure bytes do not match the digest"
            )
    if entry.caption_file is None:
        errors.append(f"figure {figure.figure_id!r}: catalog entry has no caption_file")
    elif figure.caption_file is None:
        errors.append(f"figure {figure.figure_id!r}: available figures require a caption_file")
    else:
        expected_caption = (catalog_path.parent / entry.caption_file.path).resolve()
        actual_caption = (_REPO_ROOT / figure.caption_file.path).resolve()
        if expected_caption != actual_caption:
            errors.append(
                f"figure {figure.figure_id!r}: caption_file path does not match the catalog"
            )
        if entry.caption_file.sha256 != figure.caption_file.sha256:
            errors.append(
                f"figure {figure.figure_id!r}: caption_file sha256 does not match the catalog"
            )
        tracked_caption_digest = _git_file_sha256(catalog_ref.commit, figure.caption_file.path)
        if tracked_caption_digest is None:
            errors.append(
                f"figure {figure.figure_id!r}: catalog commit does not contain the caption bytes"
            )
        elif tracked_caption_digest != figure.caption_file.sha256:
            errors.append(
                f"figure {figure.figure_id!r}: catalog commit caption bytes do not match the digest"
            )
    if not _COMMIT_RE.match(entry.generation_commit) or not _git_commit_exists(
        entry.generation_commit
    ):
        errors.append(
            f"figure {figure.figure_id!r}: catalog generation_commit is unavailable: "
            f"{entry.generation_commit}"
        )


def _validate_file_ref(
    path: str,
    sha256: str,
    label: str,
    errors: list[str],
    *,
    require_exists: bool,
    verify_digest: bool,
) -> None:
    """Validate durable path safety and, when available, its tracked bytes."""

    file_path = Path(path)
    if file_path.is_absolute() or ".." in file_path.parts:
        errors.append(f"{label}: path must be repository-relative")
        return
    if file_path.parts and (
        file_path.parts[0] in {"output", "results", ".git", ".venv"}
        or ".worktrees" in file_path.parts
    ):
        errors.append(f"{label}: path is local-only: {path}")
        return
    resolved = (_REPO_ROOT / file_path).resolve()
    try:
        resolved.relative_to(_REPO_ROOT.resolve())
    except ValueError:
        errors.append(f"{label}: path resolves outside the repository: {path}")
        return
    if not resolved.is_file():
        if require_exists:
            errors.append(f"{label}: file does not exist: {path}")
        return
    if verify_digest:
        actual = _sha256_file(resolved)
        if actual != sha256:
            errors.append(
                f"{label}: digest mismatch for {path} (declared {sha256}, actual {actual})"
            )


def _validate_source_refs(sources: list[SourceRef], errors: list[str]) -> None:
    """Verify that source refs are durable repository files with matching bytes."""

    for source in sources:
        _validate_file_ref(
            source.path,
            source.sha256,
            f"source {source.source_id!r}",
            errors,
            require_exists=True,
            verify_digest=True,
        )
        for commit_label, commit in (
            ("commit", source.commit),
            ("tracked_commit", source.tracked_commit),
        ):
            if not _COMMIT_RE.match(commit):
                errors.append(
                    f"source {source.source_id!r}: {commit_label} is not a hexadecimal git revision"
                )
                continue
            if not _git_commit_exists(commit):
                errors.append(
                    f"source {source.source_id!r}: {commit_label} is unavailable: {commit}"
                )
        _validate_generation_command(source, errors)
        tracked_digest = _git_file_sha256(source.tracked_commit, source.path)
        if tracked_digest is None:
            errors.append(
                f"source {source.source_id!r}: tracked_commit does not contain {source.path}"
            )
        elif tracked_digest != source.sha256:
            errors.append(
                f"source {source.source_id!r}: tracked_commit bytes do not match {source.path} "
                f"(declared {source.sha256}, tracked {tracked_digest})"
            )


def _validate_generation_command(source: SourceRef, errors: list[str]) -> None:
    """Bind recorded generation commands to scripts present at their commit."""

    if source.command == "evidence-review-marker.v1":
        if "review" not in source.kind.casefold():
            errors.append(
                f"source {source.source_id!r}: review marker command requires a review source kind"
            )
        return
    try:
        command_tokens = shlex.split(source.command)
    except ValueError as exc:
        errors.append(f"source {source.source_id!r}: command is not shell-parseable: {exc}")
        return
    script_paths = sorted(
        {
            normalized
            for token in command_tokens
            if (normalized := _normalise_script_path(token)) is not None
        }
    )
    if not script_paths:
        errors.append(
            f"source {source.source_id!r}: command must name a tracked Python script or review marker"
        )
        return
    invoked_script_paths = _invoked_script_paths(command_tokens)
    missing_invocations = sorted(set(script_paths).difference(invoked_script_paths))
    if missing_invocations:
        errors.append(
            f"source {source.source_id!r}: command must invoke each named Python script, "
            f"not merely mention it: {missing_invocations}"
        )
    for script_path in script_paths:
        if _git_file_sha256(source.commit, script_path) is None:
            errors.append(
                f"source {source.source_id!r}: generation commit {source.commit} does not contain "
                f"command script {script_path}"
            )


def _normalise_script_path(token: str) -> str | None:
    """Return a normalized script path when a shell token names a Python file."""
    normalized = token.removeprefix("./")
    return normalized if _SCRIPT_PATH_RE.fullmatch(token) else None


def _invoked_script_paths(tokens: list[str]) -> set[str]:
    """Return Python scripts that appear in executable shell command positions."""
    invoked: set[str] = set()
    segment: list[str] = []
    for token in tokens:
        if token in _SHELL_COMMAND_SEPARATORS:
            invoked.update(_invoked_script_paths_in_segment(segment))
            segment = []
        else:
            segment.append(token)
    invoked.update(_invoked_script_paths_in_segment(segment))
    return invoked


def _invoked_script_paths_in_segment(tokens: list[str]) -> set[str]:
    """Return scripts invoked by one shell command segment."""
    tokens = _strip_leading_assignments(tokens)
    if not tokens:
        return set()
    direct_script = _normalise_script_path(tokens[0])
    if direct_script is not None:
        return {direct_script}

    executable = Path(tokens[0]).name.casefold()
    if executable in {"python", "python3", "pypy", "pypy3"}:
        return _python_script_argument(tokens[1:])
    if executable == "uv":
        try:
            run_index = tokens.index("run", 1)
        except ValueError:
            return set()
        return _invoked_script_paths_in_segment(tokens[run_index + 1 :])
    if executable in _SHELL_WRAPPERS:
        wrapped = tokens[1:]
        if executable == "env":
            wrapped = _strip_env_options_and_assignments(wrapped)
        return _invoked_script_paths_in_segment(wrapped)
    return set()


def _python_script_argument(tokens: list[str]) -> set[str]:
    """Return a script passed as the executable Python positional argument."""
    for token in tokens:
        if token == "--":
            continue
        if token.startswith("-"):
            # ``-c`` and ``-m`` execute code/module names rather than a script path.
            if token in {"-c", "--command", "-m", "--module"}:
                return set()
            continue
        script = _normalise_script_path(token)
        return {script} if script is not None else set()
    return set()


def _strip_leading_assignments(tokens: list[str]) -> list[str]:
    """Remove shell variable assignments that precede a command.

    Returns:
        The command tokens after leading assignments.
    """
    index = 0
    while index < len(tokens) and _ASSIGNMENT_RE.fullmatch(tokens[index]):
        index += 1
    return tokens[index:]


def _strip_env_options_and_assignments(tokens: list[str]) -> list[str]:
    """Remove the bounded ``env`` options supported by command provenance.

    Returns:
        The wrapped command tokens after supported ``env`` prefixes.
    """
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _ASSIGNMENT_RE.fullmatch(token) or token in {"-i", "--ignore-environment"}:
            index += 1
            continue
        break
    return tokens[index:]


def _git_commit_exists(commit: str) -> bool:
    """Return whether a commit object is available in the repository."""

    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _git_file_sha256(commit: str, path: str) -> str | None:
    """Hash a repository file as stored at a tracked commit.

    Returns:
        The SHA-256 digest of the tracked bytes, or ``None`` when the commit
        does not contain the path.
    """

    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _validate_digest_field(value: str | None, field_name: str, errors: list[str]) -> None:
    if value is not None and not _SHA256_RE.match(value):
        errors.append(f"{field_name}: {value!r} is not a valid 64-hex SHA-256 digest")


def validate_packet(payload: dict[str, Any]) -> list[str]:
    """Validate a result interpretation packet dict semantically.

    Returns:
        A list of human-readable error strings. Empty means valid.
    """
    errors: list[str] = []

    schema_errors = _validate_against_schema(payload)
    errors.extend(schema_errors)
    if errors:
        return errors

    packet = _dict_to_packet(payload)

    _validate_evidence(packet.evidence, errors)
    _validate_population(packet.population, errors)
    _validate_execution_mode(packet.execution_mode, errors)
    _validate_estimator(packet.estimand, errors)

    for m in packet.metrics:
        _validate_metric(m, errors)

    metrics_by_id = {m.metric_id: m for m in packet.metrics}
    for d in packet.decisions:
        _validate_decision(d, metrics_by_id, packet.execution_mode, errors)

    _validate_claim_boundary(packet.claim_boundary, packet.forbidden_claims, errors)
    for index, finding in enumerate(packet.findings):
        _validate_claim_escalation(f"findings[{index}]", finding, errors)

    _validate_caption_assertions(packet.caption_assertions, packet, errors)
    _validate_figure_links(packet.figure_links, packet.estimand.estimand_id, errors)
    _validate_source_refs(packet.sources, errors)

    _check_id_uniqueness(packet.metrics, "metric_id", "metric", errors)
    _check_id_uniqueness(packet.decisions, "decision_id", "decision", errors)
    _check_id_uniqueness(packet.figure_links, "figure_id", "figure_link", errors)
    _check_id_uniqueness(packet.sources, "source_id", "source", errors)

    _validate_digest_field(packet.reviewed_packet_digest, "reviewed_packet_digest", errors)
    _validate_digest_field(packet.post_review_digest, "post_review_digest", errors)
    errors.extend(validate_review_binding(packet))

    if packet.decisions:
        declared_metrics = {m.metric_id for m in packet.metrics}
        for d in packet.decisions:
            if d.metric_id not in declared_metrics:
                errors.append(
                    f"decision {d.decision_id!r} references undeclared metric {d.metric_id!r}"
                )

    admitted_mode_count = sum(
        packet.execution_mode.counts.get(mode, 0)
        for mode in ("native", "adapter", "fallback", "degraded")
    )
    if admitted_mode_count != packet.population.included:
        errors.append(
            "execution_mode counts "
            f"({admitted_mode_count}) != population included "
            f"({packet.population.included})"
        )

    return errors


def _validate_against_schema(payload: dict[str, Any]) -> list[str]:
    """Run JSON Schema validation and return error strings.

    Returns:
        Schema-validation errors in deterministic path order.
    """
    schema = load_schema()
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path)):
        path = "/".join(str(p) for p in error.absolute_path)
        prefix = f"{path}: " if path else ""
        errors.append(f"{prefix}{error.message}")
    return errors


# ---------------------------------------------------------------------------
# Packet construction helpers
# ---------------------------------------------------------------------------


def _dict_to_packet(d: dict[str, Any]) -> ResultInterpretationPacket:
    """Convert a validated dict to a typed packet.

    Returns:
        The typed packet representation.
    """
    q = d["question"]
    question = Question(
        question_id=q["question_id"],
        text=q["text"],
        issue_refs=q.get("issue_refs", []),
    )
    ev = d["evidence"]
    evidence = Evidence(
        evidence_id=ev["evidence_id"],
        tier=ev["tier"],
        admission_state=ev["admission_state"],
        rationale=ev["rationale"],
    )
    sources = [
        SourceRef(
            source_id=s["source_id"],
            path=s["path"],
            sha256=s["sha256"],
            kind=s["kind"],
            commit=s["commit"],
            tracked_commit=s["tracked_commit"],
            command=s["command"],
            description=s.get("description", ""),
        )
        for s in d["sources"]
    ]
    pop = d["population"]
    attrition = PopulationAttrition(
        native=pop["attrition"]["native"],
        adapter=pop["attrition"]["adapter"],
        fallback=pop["attrition"]["fallback"],
        degraded=pop["attrition"]["degraded"],
        unavailable=pop["attrition"]["unavailable"],
        invalid=pop["attrition"]["invalid"],
        rejected=pop["attrition"]["rejected"],
    )
    population = Population(
        total=pop["total"],
        included=pop["included"],
        excluded=pop["excluded"],
        attrition=attrition,
        exclusion_reasons=pop.get("exclusion_reasons", []),
    )
    em = d["execution_mode"]
    execution_mode = ExecutionMode(
        counts={str(name): int(count) for name, count in em["counts"].items()},
        fallback_permitted=em.get("fallback_permitted", False),
        degraded_permitted=em.get("degraded_permitted", False),
    )
    est = d["estimand"]
    comparator = None
    if "comparator" in est and est["comparator"] is not None:
        c = est["comparator"]
        comparator = Comparator(
            reference=c["reference"],
            comparison=c["comparison"],
            direction=c.get("direction", "not_applicable"),
        )
    estimand = Estimand(
        estimand_id=est["estimand_id"],
        analysis_unit=est["analysis_unit"],
        resampling_unit=est["resampling_unit"],
        description=est.get("description", ""),
        pairing_key=est.get("pairing_key"),
        clustering_key=est.get("clustering_key"),
        comparator=comparator,
        contrast_direction=est.get("contrast_direction", ""),
    )

    metrics: list[MetricEntry] = []
    for m in d["metrics"]:
        unc = None
        if "uncertainty" in m and m["uncertainty"] is not None:
            u = m["uncertainty"]
            unc = Uncertainty(
                declared=u["declared"],
                method=u.get("method"),
                ci_low=u.get("ci_low"),
                ci_high=u.get("ci_high"),
                p_value_raw=u.get("p_value_raw"),
                p_value_adjusted=u.get("p_value_adjusted"),
            )
        mult = None
        if "multiplicity" in m and m["multiplicity"] is not None:
            mu = m["multiplicity"]
            mult = Multiplicity(
                declared=mu["declared"],
                method=mu.get("method"),
                n_comparisons=mu.get("n_comparisons"),
            )
        metrics.append(
            MetricEntry(
                metric_id=m["metric_id"],
                unit=m["unit"],
                desirability=m["desirability"],
                support=m["support"],
                denominator=m["denominator"],
                missingness=m["missingness"],
                unavailable_handling=m.get("unavailable_handling", "fail_closed"),
                effect=m.get("effect"),
                uncertainty=unc,
                null_value=m.get("null_value"),
                multiplicity=mult,
                sensitivity=m.get("sensitivity"),
                support_threshold=m.get("support_threshold"),
            )
        )

    decisions = [
        DecisionEntry(
            decision_id=dd["decision_id"],
            metric_id=dd["metric_id"],
            outcome=dd["outcome"],
            rationale=dd["rationale"],
            comparator=(
                Comparator(
                    reference=dd["comparator"]["reference"],
                    comparison=dd["comparator"]["comparison"],
                    direction=dd["comparator"].get("direction", "not_applicable"),
                )
                if dd.get("comparator") is not None
                else None
            ),
            contrast_result=(
                ContrastResult(
                    comparator=Comparator(
                        reference=dd["contrast_result"]["comparator"]["reference"],
                        comparison=dd["contrast_result"]["comparator"]["comparison"],
                        direction=dd["contrast_result"]["comparator"].get(
                            "direction", "not_applicable"
                        ),
                    ),
                    effect=dd["contrast_result"]["effect"],
                    support=dd["contrast_result"]["support"],
                    denominator=dd["contrast_result"]["denominator"],
                    support_threshold=dd["contrast_result"]["support_threshold"],
                    null_value=dd["contrast_result"]["null_value"],
                    uncertainty=Uncertainty(
                        declared=dd["contrast_result"]["uncertainty"]["declared"],
                        method=dd["contrast_result"]["uncertainty"].get("method"),
                        ci_low=dd["contrast_result"]["uncertainty"].get("ci_low"),
                        ci_high=dd["contrast_result"]["uncertainty"].get("ci_high"),
                        p_value_raw=dd["contrast_result"]["uncertainty"].get("p_value_raw"),
                        p_value_adjusted=dd["contrast_result"]["uncertainty"].get(
                            "p_value_adjusted"
                        ),
                    ),
                    multiplicity=Multiplicity(
                        declared=dd["contrast_result"]["multiplicity"]["declared"],
                        method=dd["contrast_result"]["multiplicity"].get("method"),
                        n_comparisons=dd["contrast_result"]["multiplicity"].get("n_comparisons"),
                    ),
                )
                if dd.get("contrast_result") is not None
                else None
            ),
            effect=dd.get("effect"),
            refusal_reason=dd.get("refusal_reason"),
        )
        for dd in d["decisions"]
    ]

    figure_links: list[FigureLink] = []
    for fl in d.get("figure_links", []):
        cap = None
        if fl.get("caption_file") is not None:
            cf = fl["caption_file"]
            cap = FileRef(path=cf["path"], sha256=cf["sha256"])
        artifact_catalog = None
        if fl.get("artifact_catalog") is not None:
            catalog = fl["artifact_catalog"]
            artifact_catalog = ArtifactCatalogRef(
                catalog_id=catalog["catalog_id"],
                path=catalog["path"],
                sha256=catalog["sha256"],
                commit=catalog["commit"],
            )
        vc = fl["visual_contract"]
        visual_contract = FigureVisualContract(
            estimand_id=vc["estimand_id"],
            plot_type=vc["plot_type"],
            rationale=vc["rationale"],
            encodings=vc["encodings"],
            transforms=vc.get("transforms", []),
            limits=vc.get("limits", {}),
            reference_lines=vc.get("reference_lines", []),
            ordering=vc.get("ordering", []),
            faceting=vc.get("faceting", []),
            uncertainty_encoding=vc.get("uncertainty_encoding"),
            sample_size_display=vc.get("sample_size_display"),
            legend_identities=vc.get("legend_identities", []),
            accessibility_contract=vc.get("accessibility_contract", []),
        )
        figure_links.append(
            FigureLink(
                figure_id=fl["figure_id"],
                artifact_id=fl["artifact_id"],
                path=fl["path"],
                sha256=fl["sha256"],
                encoding=fl["encoding"],
                visual_contract=visual_contract,
                caption_file=cap,
                artifact_catalog=artifact_catalog,
            )
        )

    caption_assertions = [
        CaptionAssertion(
            figure_id=ca["figure_id"],
            template_id=ca["template_id"],
            assertion_text=ca["assertion_text"],
            status=ca["status"],
            bound_to_packet_fields=ca.get("bound_to_packet_fields", []),
        )
        for ca in d.get("caption_assertions", [])
    ]

    cb = d["claim_boundary"]
    claim_boundary = ClaimBoundary(
        allowed=cb["allowed"],
        forbidden=cb["forbidden"],
    )

    producer = ActorRef(
        actor_id=d["producer"]["actor_id"],
        commit=d["producer"].get("commit", ""),
        command=d["producer"].get("command", ""),
        status=d["producer"].get("status", "draft"),
    )

    reviewer = None
    if d.get("reviewer") is not None:
        r = d["reviewer"]
        reviewer = ActorRef(
            actor_id=r["actor_id"],
            commit=r.get("commit", ""),
            command=r.get("command", ""),
            status=r.get("status", "draft"),
        )

    return ResultInterpretationPacket(
        schema_version=d["schema_version"],
        packet_id=d["packet_id"],
        question=question,
        evidence=evidence,
        sources=sources,
        population=population,
        execution_mode=execution_mode,
        estimand=estimand,
        metrics=metrics,
        decisions=decisions,
        figure_links=figure_links,
        caption_assertions=caption_assertions,
        claim_boundary=claim_boundary,
        forbidden_claims=d.get("forbidden_claims", []),
        producer=producer,
        reviewer=reviewer,
        reviewed_packet_digest=d.get("reviewed_packet_digest"),
        post_review_digest=d.get("post_review_digest"),
        findings=d["findings"],
        limitations=d["limitations"],
        fail_closed_changes=d.get("fail_closed_changes", []),
    )


def build_and_validate_packet(payload: dict[str, Any]) -> ResultInterpretationPacket:
    """Validate and convert a dict to a typed packet.

    Raises:
        ResultInterpretationPacketError: If validation fails.

    Returns:
        The validated typed packet.
    """
    errors = validate_packet(payload)
    if errors:
        raise ResultInterpretationPacketError(errors)
    return _dict_to_packet(payload)


def _canonical_digest_payload(
    packet: ResultInterpretationPacket,
    *,
    include_reviewer: bool,
    context: str,
) -> dict[str, Any]:
    """Build a stable, non-circular payload for packet digest calculation.

    Returns:
        A canonicalization-ready dictionary.
    """

    payload = packet.to_dict()
    if not include_reviewer:
        payload["reviewer"] = None
    payload["reviewed_packet_digest"] = None
    payload["post_review_digest"] = None
    payload["_digest_context"] = context
    return payload


def compute_packet_digest(packet: ResultInterpretationPacket) -> str:
    """Compute the pre-review content digest, excluding review metadata.

    Returns:
        A lowercase hexadecimal SHA-256 digest.
    """
    canonical = json.dumps(
        _canonical_digest_payload(packet, include_reviewer=False, context="packet"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_post_review_digest(packet: ResultInterpretationPacket) -> str:
    """Compute a post-review digest that includes reviewer binding.

    Returns:
        A lowercase hexadecimal SHA-256 digest.
    """
    canonical = json.dumps(
        _canonical_digest_payload(packet, include_reviewer=True, context="post_review"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_review_binding(packet: ResultInterpretationPacket) -> list[str]:  # noqa: C901
    """Validate the optional review binding against the packet content.

    Returns:
        Human-readable review-binding errors. Empty means no drift was found.
    """

    errors: list[str] = []
    if packet.reviewer is None:
        if _review_is_required(packet):
            errors.append(
                "evidence admission requires an independent reviewer with exact review digests"
            )
        if packet.reviewed_packet_digest is not None or packet.post_review_digest is not None:
            errors.append("review digests require a reviewer identity")
        return errors
    if packet.reviewer.actor_id == packet.producer.actor_id:
        errors.append("reviewer identity must be independent from producer identity")
    if packet.reviewer.status not in {"reviewed", "final"}:
        if _review_is_required(packet):
            errors.append("evidence admission requires reviewer status 'reviewed' or 'final'")
        if packet.reviewed_packet_digest is not None or packet.post_review_digest is not None:
            errors.append(
                "reviewer status must be 'reviewed' or 'final' when review digests are set"
            )
        return errors
    if not _COMMIT_RE.match(packet.reviewer.commit):
        errors.append("reviewer commit is not a hexadecimal git revision")
    elif not _git_commit_exists(packet.reviewer.commit):
        errors.append(f"reviewer commit is unavailable: {packet.reviewer.commit}")
    if packet.reviewed_packet_digest is None:
        errors.append("reviewed_packet_digest is required for a reviewed packet")
    elif packet.reviewed_packet_digest != compute_packet_digest(packet):
        errors.append("reviewed_packet_digest does not match the packet content")
    if packet.post_review_digest is None:
        errors.append("post_review_digest is required for a reviewed packet")
    elif packet.post_review_digest != compute_post_review_digest(packet):
        errors.append("post_review_digest does not match the reviewed packet content")
    return errors


def _review_is_required(packet: ResultInterpretationPacket) -> bool:
    """Return whether the evidence tier/state may only be admitted after review."""

    return (
        packet.evidence.admission_state in _REVIEW_REQUIRED_ADMISSION_STATES
        or packet.evidence.tier in _REVIEW_REQUIRED_EVIDENCE_TIERS
    )


def write_deterministic_json(payload: dict[str, Any], path: Path) -> None:
    """Write a packet dict as deterministic single-line JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Returns:
        A lowercase hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_caption(packet: ResultInterpretationPacket) -> str:
    """Render a deterministic caption from observed or unavailable assertions.

    Returns:
        Caption text with a trailing newline.
    """

    lines = [
        f"{packet.packet_id}.",
        f"Question: {packet.question.text}",
        f"Evidence: {packet.evidence.tier}; admission={packet.evidence.admission_state}",
    ]
    for assertion in packet.caption_assertions:
        lines.append(f"[{assertion.status}] {assertion.assertion_text}")
    lines.append("Claim boundary: " + " ".join(packet.claim_boundary.forbidden))
    return "\n".join(lines) + "\n"


def write_caption(packet: ResultInterpretationPacket, path: Path) -> None:
    """Write the deterministic caption text for a packet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_caption(packet), encoding="utf-8")


def write_review_report(packet: ResultInterpretationPacket, path: Path) -> None:
    """Write a deterministic review report without changing admission state."""

    reviewer = None if packet.reviewer is None else asdict(packet.reviewer)
    report = {
        "schema_version": "result_interpretation_review.v1",
        "packet_id": packet.packet_id,
        "packet_digest": compute_packet_digest(packet),
        "post_review_digest": packet.post_review_digest,
        "status": "pending" if packet.reviewer is None else packet.reviewer.status,
        "reviewer": reviewer,
        "findings": packet.findings,
        "limitations": packet.limitations,
        "fail_closed_changes": packet.fail_closed_changes,
    }
    write_deterministic_json(report, path)


def _packet_checksum_files(packet: ResultInterpretationPacket) -> dict[str, Path]:
    """Return durable source and rendered-evidence files for a packet manifest."""

    files: dict[str, Path] = {}
    for source in packet.sources:
        files[f"source/{source.source_id}/{source.path}"] = _REPO_ROOT / source.path
    for figure in packet.figure_links:
        if figure.artifact_catalog is not None:
            files[f"catalog/{figure.figure_id}/{figure.artifact_catalog.path}"] = (
                _REPO_ROOT / figure.artifact_catalog.path
            )
        if figure.encoding != "unavailable":
            files[f"figure/{figure.figure_id}/{figure.path}"] = _REPO_ROOT / figure.path
        if figure.caption_file is not None:
            files[f"caption/{figure.figure_id}/{figure.caption_file.path}"] = (
                _REPO_ROOT / figure.caption_file.path
            )
    return files


def write_checksum_manifest(
    files: dict[str, Path],
    path: Path,
    *,
    packet: ResultInterpretationPacket | None = None,
) -> None:
    """Write sorted SHA-256 entries for outputs and packet evidence files."""

    manifest_files = dict(files)
    if packet is not None:
        manifest_files.update(_packet_checksum_files(packet))
    lines = [f"{_sha256_file(manifest_files[name])}  {name}" for name in sorted(manifest_files)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_result_interpretation_packet(path: Path) -> ResultInterpretationPacket:
    """Load and validate a result interpretation packet from JSON.

    Returns:
        The validated typed packet.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return build_and_validate_packet(raw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="build_result_interpretation_packet",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input packet JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the validated output JSON. If omitted, validates only.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Validate the input and exit without writing output.",
    )
    parser.add_argument(
        "--show-digest",
        action="store_true",
        default=False,
        help="Print the packet digest and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for building result interpretation packets.

    Returns:
        Zero on success, one for packet validation errors, or two for I/O/JSON errors.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        packet = build_and_validate_packet(raw)
        if args.show_digest:
            digest = compute_packet_digest(packet)
            sys.stdout.write(f"packet_digest: {digest}\n")
            return 0
        if args.validate_only:
            sys.stdout.write(f"packet {packet.packet_id!r} is valid\n")
            return 0
        output = args.output or args.input.with_suffix(".validated.json")
        write_deterministic_json(packet.to_dict(), output)
        sys.stdout.write(f"written {output}\n")
        return 0
    except ResultInterpretationPacketError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
