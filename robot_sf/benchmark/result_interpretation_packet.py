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
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "result_interpretation_packet.v1"
RESULT_INTERPRETATION_PACKET_SCHEMA_VERSION = SCHEMA_VERSION
_SCHEMA_FILE = Path(__file__).with_name("schemas") / "result_interpretation_packet.v1.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_VALID_DECISION_OUTCOMES = frozenset(
    {"supported", "not_supported", "inconclusive", "invalid", "unavailable"}
)
_VALID_DESIRABILITY = frozenset(
    {"higher_is_better", "lower_is_better", "target_range", "not_applicable"}
)
_VALID_MISSINGNESS = frozenset({"complete", "partial", "unavailable", "not_imputed"})
_VALID_UNAVAILABLE_HANDLING = frozenset({"fail_closed", "diagnostic_only", "excluded"})
_VALID_EXECUTION_MODES = frozenset(
    {"native", "adapter", "fallback", "degraded", "unavailable", "invalid"}
)
_VALID_ACTOR_STATUSES = frozenset({"draft", "reviewed", "final"})
_VALID_CAPTION_STATUSES = frozenset({"observed", "inferred", "unavailable"})
_VALID_COMPARATOR_DIRECTIONS = frozenset(
    {"comparison_minus_reference", "reference_minus_comparison", "not_applicable"}
)
_VALID_FIGURE_ENCODINGS = frozenset({"png", "pdf", "svg", "unavailable"})


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


@dataclass(frozen=True, slots=True)
class DecisionEntry:
    """A controlled vocabulary decision for a metric."""

    decision_id: str
    metric_id: str
    outcome: str
    rationale: str
    refusal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FileRef:
    """Checksum-bound file reference."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FigureVisualContract:
    """Explicit visual grammar for a figure or unavailable figure slot."""

    plot_type: str
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


@dataclass(frozen=True, slots=True)
class CaptionAssertion:
    """A structured caption assertion bound to figure fields."""

    figure_id: str
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
    mode_sum = a.native + a.adapter + a.fallback + a.degraded + a.unavailable + a.invalid
    if mode_sum != p.excluded:
        errors.append(f"attrition sum ({mode_sum}) != excluded ({p.excluded})")


def _validate_evidence(evidence: Evidence, errors: list[str]) -> None:
    """Require explicit evidence identity and admission rationale."""

    for field_name in ("evidence_id", "tier", "admission_state", "rationale"):
        if not getattr(evidence, field_name).strip():
            errors.append(f"evidence.{field_name} is required")


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
    for excluded_mode in ("unavailable", "invalid"):
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
    if uncertainty.ci_low is not None and uncertainty.ci_high is not None:
        if uncertainty.ci_low > uncertainty.ci_high:
            errors.append(f"metric {metric_id!r}: uncertainty ci_low exceeds ci_high")
    for value_name, value in (
        ("p_value_raw", uncertainty.p_value_raw),
        ("p_value_adjusted", uncertainty.p_value_adjusted),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
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


def _validate_metric(m: MetricEntry, errors: list[str]) -> None:
    if m.denominator <= 0:
        errors.append(f"metric {m.metric_id!r}: denominator must be > 0")
    if m.support < 0:
        errors.append(f"metric {m.metric_id!r}: support must be >= 0")
    if m.support > m.denominator:
        errors.append(
            f"metric {m.metric_id!r}: support ({m.support}) > denominator ({m.denominator})"
        )
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
    _validate_uncertainty(m.metric_id, m.uncertainty, errors)
    _validate_multiplicity(m.metric_id, m.multiplicity, errors)
    if m.missingness == "not_imputed":
        errors.append(
            f"metric {m.metric_id!r}: missingness 'not_imputed' is not allowed; "
            "use 'unavailable' or 'excluded'"
        )


def _validate_decision(
    d: DecisionEntry,
    metrics_by_id: dict[str, MetricEntry],
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
    if d.outcome == "supported":
        if metric.unavailable_handling != "fail_closed":
            errors.append(
                f"decision {d.decision_id!r}: supported outcome requires fail_closed metric handling"
            )
        if metric.uncertainty is None or not metric.uncertainty.declared:
            errors.append(
                f"decision {d.decision_id!r}: supported outcome requires declared uncertainty"
            )
        if metric.multiplicity is None or not metric.multiplicity.declared:
            errors.append(
                f"decision {d.decision_id!r}: supported outcome requires declared multiplicity"
            )
        if d.refusal_reason is not None:
            errors.append(
                f"decision {d.decision_id!r}: supported outcome cannot have refusal_reason"
            )
    elif not d.refusal_reason:
        errors.append(f"decision {d.decision_id!r}: non-supported outcome requires refusal_reason")


def _validate_claim_boundary(cb: ClaimBoundary, errors: list[str]) -> None:
    if not cb.allowed:
        errors.append("claim_boundary.allowed must have at least one entry")
    if not cb.forbidden:
        errors.append("claim_boundary.forbidden must have at least one entry")


def _validate_caption_assertions(
    captions: list[CaptionAssertion],
    figure_ids: set[str],
    errors: list[str],
) -> None:
    for ca in captions:
        if ca.figure_id not in figure_ids:
            errors.append(f"caption assertion for {ca.figure_id!r} references an undeclared figure")
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


def _validate_figure_links(
    figures: list[FigureLink],
    errors: list[str],
) -> None:
    for fl in figures:
        if fl.encoding not in _VALID_FIGURE_ENCODINGS:
            errors.append(
                f"figure {fl.figure_id!r}: encoding {fl.encoding!r} not in "
                f"{sorted(_VALID_FIGURE_ENCODINGS)}"
            )
        if fl.encoding != "unavailable" and not _SHA256_RE.match(fl.sha256):
            errors.append(
                f"figure {fl.figure_id!r}: sha256 must be 64-hex when encoding is not 'unavailable'"
            )
        if fl.encoding != "unavailable" and fl.path.startswith(("output/", "/tmp/", "/home/")):
            errors.append(
                f"figure {fl.figure_id!r}: durable figure path cannot be local-only: {fl.path}"
            )


def _validate_source_refs(sources: list[SourceRef], errors: list[str]) -> None:
    """Verify that source refs are durable repository files with matching bytes."""

    for source in sources:
        source_path = Path(source.path)
        if source_path.is_absolute() or ".." in source_path.parts:
            errors.append(f"source {source.source_id!r}: path must be repository-relative")
            continue
        if source_path.parts and source_path.parts[0] in {"output", ".git", ".venv"}:
            errors.append(f"source {source.source_id!r}: path is local-only: {source.path}")
            continue
        resolved = _REPO_ROOT / source_path
        if not resolved.is_file():
            errors.append(f"source {source.source_id!r}: file does not exist: {source.path}")
            continue
        actual = _sha256_file(resolved)
        if actual != source.sha256:
            errors.append(
                f"source {source.source_id!r}: digest mismatch for {source.path} "
                f"(declared {source.sha256}, actual {actual})"
            )
        if not _COMMIT_RE.match(source.commit):
            errors.append(f"source {source.source_id!r}: commit is not a hexadecimal git revision")


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
        _validate_decision(d, metrics_by_id, errors)

    _validate_claim_boundary(packet.claim_boundary, errors)

    figure_ids = {fl.figure_id for fl in packet.figure_links}
    _validate_caption_assertions(packet.caption_assertions, figure_ids, errors)
    _validate_figure_links(packet.figure_links, errors)
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
            )
        )

    decisions = [
        DecisionEntry(
            decision_id=dd["decision_id"],
            metric_id=dd["metric_id"],
            outcome=dd["outcome"],
            rationale=dd["rationale"],
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
        vc = fl["visual_contract"]
        visual_contract = FigureVisualContract(
            plot_type=vc["plot_type"],
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
            )
        )

    caption_assertions = [
        CaptionAssertion(
            figure_id=ca["figure_id"],
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


def validate_review_binding(packet: ResultInterpretationPacket) -> list[str]:
    """Validate the optional review binding against the packet content.

    Returns:
        Human-readable review-binding errors. Empty means no drift was found.
    """

    errors: list[str] = []
    if packet.reviewer is None:
        if packet.reviewed_packet_digest is not None or packet.post_review_digest is not None:
            errors.append("review digests require a reviewer identity")
        return errors
    if packet.reviewer.status not in {"reviewed", "final"}:
        if packet.reviewed_packet_digest is not None or packet.post_review_digest is not None:
            errors.append(
                "reviewer status must be 'reviewed' or 'final' when review digests are set"
            )
        return errors
    if packet.reviewed_packet_digest is None:
        errors.append("reviewed_packet_digest is required for a reviewed packet")
    elif packet.reviewed_packet_digest != compute_packet_digest(packet):
        errors.append("reviewed_packet_digest does not match the packet content")
    if packet.post_review_digest is None:
        errors.append("post_review_digest is required for a reviewed packet")
    elif packet.post_review_digest != compute_post_review_digest(packet):
        errors.append("post_review_digest does not match the reviewed packet content")
    return errors


def write_deterministic_json(payload: dict[str, Any], path: Path) -> None:
    """Write a packet dict as deterministic single-line JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
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


def write_checksum_manifest(files: dict[str, Path], path: Path) -> None:
    """Write sorted SHA-256 entries for generated packet outputs."""

    lines = [f"{_sha256_file(files[name])}  {name}" for name in sorted(files)]
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
