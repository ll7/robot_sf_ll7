"""Deterministic evaluation fixtures for agent interpretation of figure packets.

This module scores ephemeral interpretation projections derived from one
canonical ``result_interpretation_packet.v1`` fixture. It does not call
external providers, read generated benchmark packets from other branches, or
promote fixture outputs as benchmark evidence. Optional workflow variants,
reviewer accounting, and correction rankings are still fixture metadata and
must not be read as benchmark or scientific results.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.result_interpretation_packet import (
    ResultInterpretationPacketError,
    load_result_interpretation_packet,
)

EVAL_SCHEMA_VERSION = "agent_figure_interpretation_eval.v1"
MANIFEST_SCHEMA_VERSION = "agent_figure_interpretation_eval_manifest.v1"
REPLAY_SCHEMA_VERSION = "agent_figure_interpretation_replay.v1"
CANDIDATE_SCHEMA_VERSION = "agent_figure_interpretation_candidate.v1"
EXPECTED_PACKET_SCHEMA = "result_interpretation_packet.v1"
EXPECTED_MANIFEST_STATUS = "evaluation_artifacts_only"
EXPECTED_MANIFEST_CLAIM_BOUNDARY = (
    "frozen fixture replay only; no external model calls, no benchmark claims, "
    "no generated evidence promotion"
)
EXPECTED_PACKET_CLAIM_BOUNDARY = "fixture replay only; not benchmark evidence"
EXPECTED_REPORT_CLAIM_BOUNDARY = (
    "fixture replay only; no external model calls, no benchmark claims, "
    "and no generated evidence promotion"
)
EXPECTED_CANDIDATE_ARTIFACT_KIND = "candidate_interpretation"
EXPECTED_CANDIDATE_PROVIDER = "none"
REQUIRED_CANDIDATE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "provider",
        "fixture_id",
        "mutation_id",
        "workflow",
        "figure",
        "limitations",
        "confidence",
        "unresolved_questions",
        "claim_boundary",
        "interpretation",
        "mutation",
        "findings",
        "unavailable",
        "not_applicable",
        "provenance",
        "replay_provenance",
        "verdict",
    }
)
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
LOCAL_ONLY_MANIFEST_PARTS = frozenset({".git", ".venv", ".worktrees", "output", "results"})
DIMENSIONS = (
    "source_denominator",
    "estimand_unit",
    "stats_multiplicity",
    "visual_semantics",
    "caption_accuracy",
    "evidence_tier_availability",
    "claim_boundary",
    "correction_usefulness",
)
CRITICAL_ERROR_KINDS = (
    "unavailable_to_zero",
    "denominator_loss",
    "analysis_unit_mismatch",
    "wrong_pairing_resampling",
    "fallback_degraded_promotion",
    "causal_overclaim",
    "unsupported_ranking",
    "null_overclaim",
    "effect_direction_desirability",
    "native_adapter_merge",
    "multiplicity_language",
)
REQUIRED_SCIENTIFIC_ERROR_MUTATIONS = CRITICAL_ERROR_KINDS
INTEGRITY_MUTATION_IDS = (
    "digest_omission",
    "stale_post_review_bytes",
)
SYNTHETIC_MUTATION_FIXTURE_ID = "ch7_visualization_causal_abstention_fixture"
SYNTHETIC_MUTATION_IDS = CRITICAL_ERROR_KINDS
INTERPRETATION_VARIANTS = ("baseline", "packet_constrained")
SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}
CRITICAL_ERROR_DIMENSIONS = {
    "unavailable_to_zero": "evidence_tier_availability",
    "denominator_loss": "source_denominator",
    "analysis_unit_mismatch": "estimand_unit",
    "wrong_pairing_resampling": "stats_multiplicity",
    "fallback_degraded_promotion": "evidence_tier_availability",
    "causal_overclaim": "claim_boundary",
    "unsupported_ranking": "visual_semantics",
    "null_overclaim": "claim_boundary",
    "effect_direction_desirability": "visual_semantics",
    "native_adapter_merge": "evidence_tier_availability",
    "multiplicity_language": "stats_multiplicity",
}
EXPECTED_MUTATION_IDS = ("clean", *CRITICAL_ERROR_KINDS)
CANONICAL_FIXTURE_ID = "ch7_visualization_causal_abstention_fixture"
_HIGHER_THAN_DIAGNOSTIC = {
    "smoke",
    "smoke evidence",
    "benchmark",
    "nominal benchmark evidence",
    "paper_facing",
    "paper-grade",
    "paper-grade evidence",
}
_FALLBACK_DEGRADED_MODES = frozenset({"fallback", "degraded"})


class AgentFigureEvalError(ValueError):
    """Raised when a frozen evaluation artifact fails closed before scoring."""


@dataclass(frozen=True, slots=True)
class DimensionScore:
    """Score for one interpretation dimension."""

    dimension: str
    score: float
    passed: bool
    expected: Any
    observed: Any


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """Deterministic evaluation result for one frozen fixture packet."""

    packet_id: str
    artifact_kind: str
    status: str
    scores: list[DimensionScore]
    critical_errors: dict[str, bool]
    aggregate_score: float
    claim_boundary: str
    interpretation_variant_comparison: dict[str, Any] | None = None
    reviewer_accounting: dict[str, Any] | None = None
    correction_priority_ranking: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["scores"] = [asdict(score) for score in self.scores]
        if self.interpretation_variant_comparison is None:
            payload.pop("interpretation_variant_comparison")
        if self.reviewer_accounting is None:
            payload.pop("reviewer_accounting")
        if self.correction_priority_ranking is None:
            payload.pop("correction_priority_ranking")
        return payload


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest for *path*."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(data: Any) -> str:
    """Serialize JSON deterministically for stable CLI output.

    Returns:
        Compact JSON with sorted keys.
    """

    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from *path*.

    Returns:
        Parsed JSON object.
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentFigureEvalError(f"{path}: unreadable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentFigureEvalError(f"{path}: expected a JSON object")
    return data


def load_verified_packets(manifest_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load canonical packets and deterministic mutation projections.

    The manifest is intentionally small and provider-independent:
    ``expected_packet_schema`` must exactly match
    :data:`EXPECTED_PACKET_SCHEMA`; the referenced packet is validated by the
    repository's canonical loader before mutation projections are created.

    Returns:
        Pairs of the canonical packet path and ephemeral evaluator projections.
    """

    manifest = _load_eval_manifest(manifest_path)
    packet_record = manifest["packet"]
    packet = _load_canonical_packet(
        manifest_path=manifest_path,
        packet_record=packet_record,
        expected_schema=manifest["expected_packet_schema"],
    )
    base = _canonical_evaluation_packet(packet)
    packets: list[tuple[Path, dict[str, Any]]] = []
    packet_path = _resolve_manifest_path(
        manifest_path=manifest_path,
        rel_path=packet_record["path"],
        index=0,
        path_key="path",
    )
    for mutation_id in manifest["mutations"]:
        projected = (
            json.loads(canonical_json(base))
            if mutation_id == "clean"
            else _apply_synthetic_mutation(base, mutation_id)
        )
        projected["packet_id"] = mutation_id
        packets.append((packet_path, projected))
    return packets


def _load_eval_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate the small evaluator manifest without duplicating packet schema.

    Returns:
        The validated manifest mapping.
    """

    manifest = load_json(manifest_path)
    _validate_eval_manifest_header(manifest)
    packet = manifest.get("packet")
    if not isinstance(packet, dict):
        raise AgentFigureEvalError("manifest packet must be an object")
    _validate_eval_manifest_packet(packet)
    _validate_eval_manifest_mutations(manifest.get("mutations"))
    return manifest


def _validate_eval_manifest_header(manifest: Mapping[str, Any]) -> None:
    """Validate fixed manifest identity and claim-boundary fields."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise AgentFigureEvalError("manifest schema_version mismatch")
    if manifest.get("status") != EXPECTED_MANIFEST_STATUS:
        raise AgentFigureEvalError(f"manifest status must be {EXPECTED_MANIFEST_STATUS!r}")
    expected_schema = manifest.get("expected_packet_schema")
    if expected_schema != EXPECTED_PACKET_SCHEMA:
        raise AgentFigureEvalError(
            "manifest expected_packet_schema must be "
            f"{EXPECTED_PACKET_SCHEMA!r}, got {expected_schema!r}"
        )
    if manifest.get("claim_boundary") != EXPECTED_MANIFEST_CLAIM_BOUNDARY:
        raise AgentFigureEvalError(
            "manifest claim_boundary must preserve the evaluation-artifacts-only boundary"
        )


def _validate_eval_manifest_packet(packet: Mapping[str, Any]) -> None:
    """Validate the manifest's one canonical packet record."""

    for key in ("id", "path", "sha256", "source_sha256", "reference_sha256"):
        if not isinstance(packet.get(key), str) or not packet[key]:
            raise AgentFigureEvalError(f"manifest packet {key} must be non-empty text")
    if not DIGEST_RE.fullmatch(packet["sha256"]):
        raise AgentFigureEvalError("manifest packet sha256 must be a SHA-256 digest")
    for key in ("source_sha256", "reference_sha256"):
        if not DIGEST_RE.fullmatch(packet[key]):
            raise AgentFigureEvalError(f"manifest packet {key} must be a SHA-256 digest")
    if packet["id"] != CANONICAL_FIXTURE_ID:
        raise AgentFigureEvalError(
            f"manifest packet id must be {CANONICAL_FIXTURE_ID!r}; canonical fixture ownership is fixed"
        )


def _validate_eval_manifest_mutations(mutations: Any) -> None:
    """Require the complete deterministic mutation matrix exactly once."""

    if not isinstance(mutations, list) or any(
        not isinstance(mutation_id, str) for mutation_id in mutations
    ):
        raise AgentFigureEvalError("manifest mutations must be a list of text identifiers")
    if len(mutations) != len(set(mutations)):
        raise AgentFigureEvalError("manifest mutations must not contain duplicates")
    if set(mutations) != set(EXPECTED_MUTATION_IDS):
        raise AgentFigureEvalError(
            "manifest mutations must cover exactly clean and every critical detector"
        )


def _load_canonical_packet(
    *, manifest_path: Path, packet_record: Mapping[str, Any], expected_schema: str
) -> dict[str, Any]:
    """Load one source-backed packet through the canonical typed loader.

    Returns:
        The canonical packet serialized through its typed representation.
    """

    packet_path = _verified_manifest_file(
        manifest_path=manifest_path,
        artifact=packet_record,
        index=0,
        path_key="path",
        sha_key="sha256",
    )
    try:
        typed_packet = load_result_interpretation_packet(packet_path)
    except (OSError, ValueError, ResultInterpretationPacketError) as exc:
        raise AgentFigureEvalError(
            f"{packet_record['path']}: canonical result packet validation failed: {exc}"
        ) from exc
    packet = typed_packet.to_dict()
    if packet.get("schema_version") != expected_schema:
        raise AgentFigureEvalError(
            f"{packet_record['path']}: canonical packet schema_version must be {expected_schema!r}"
        )
    if packet.get("packet_id") != packet_record["id"]:
        raise AgentFigureEvalError(
            f"{packet_record['path']}: canonical packet_id must match manifest packet id"
        )
    observed_source_digest = _canonical_digest(packet.get("sources"))
    if observed_source_digest != packet_record["source_sha256"]:
        raise AgentFigureEvalError(
            f"{packet_record['path']}: canonical source binding digest does not match manifest"
        )
    if packet_record["reference_sha256"] != packet_record["sha256"]:
        raise AgentFigureEvalError(
            "manifest reference_sha256 must equal the canonical packet digest; "
            "the evaluator does not carry a second reference packet"
        )
    return packet


def _canonical_evaluation_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Project canonical packet fields into an ephemeral scoring view.

    The projection is never written as a packet or registered as evidence. It
    exists only so deterministic mutation operators can compare candidate
    interpretation fields against the typed canonical packet.

    Returns:
        An ephemeral evaluator-scoring projection.
    """

    packet_id = _canonical_text(packet, "packet_id")
    evidence = _canonical_mapping(packet, "evidence")
    population = _canonical_mapping(packet, "population")
    execution_mode = _canonical_mapping(packet, "execution_mode")
    estimand = _canonical_mapping(packet, "estimand")
    sources = _canonical_list_of_mappings(packet, "sources")
    metrics = _canonical_list_of_mappings(packet, "metrics")
    decisions = _canonical_list_of_mappings(packet, "decisions")
    figure_links = _canonical_list_of_mappings(packet, "figure_links")
    caption_assertions = _canonical_list_of_mappings(packet, "caption_assertions")
    claim_boundary = _canonical_mapping(packet, "claim_boundary")
    forbidden_claims = packet.get("forbidden_claims")
    if not isinstance(forbidden_claims, list) or any(
        not isinstance(claim, str) for claim in forbidden_claims
    ):
        raise AgentFigureEvalError("canonical packet forbidden_claims must be a list of text")
    metric = metrics[0]
    denominator = metric.get("denominator")
    if not isinstance(denominator, int) or isinstance(denominator, bool) or denominator < 0:
        raise AgentFigureEvalError("canonical packet first metric denominator must be non-negative")
    counts = execution_mode.get("counts")
    if not isinstance(counts, dict) or any(
        not isinstance(mode, str)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for mode, count in counts.items()
    ):
        raise AgentFigureEvalError("canonical packet execution_mode.counts is invalid")
    active_modes = [mode for mode, count in counts.items() if count > 0]
    execution_name = active_modes[0] if len(active_modes) == 1 else "mixed"
    uncertainty = metric.get("uncertainty")
    multiplicity = metric.get("multiplicity")
    if not isinstance(uncertainty, dict) or not isinstance(multiplicity, dict):
        raise AgentFigureEvalError(
            "canonical packet metric uncertainty/multiplicity must be objects"
        )
    comparator = estimand.get("comparator")
    if comparator is not None and not isinstance(comparator, dict):
        raise AgentFigureEvalError("canonical packet estimand comparator must be an object or null")
    visual_contract = figure_links[0].get("visual_contract") if figure_links else None
    if visual_contract is not None and not isinstance(visual_contract, dict):
        raise AgentFigureEvalError("canonical packet visual_contract must be an object")
    admission_state = evidence.get("admission_state")
    if not isinstance(admission_state, str):
        raise AgentFigureEvalError("canonical packet evidence admission_state must be text")
    outcomes = [decision.get("outcome") for decision in decisions]
    unavailable = admission_state == "unavailable_causal_inference" or all(
        metric_entry.get("effect") is None for metric_entry in metrics
    )
    ranking_supported = not any("ranking" in claim.casefold() for claim in forbidden_claims)
    causal_allowed = not any("causal" in claim.casefold() for claim in forbidden_claims)
    null_result_claim = (
        "supported"
        if any(outcome in {"supported", "supported_equivalence"} for outcome in outcomes)
        else "not_supported"
    )
    caption = caption_assertions[0] if caption_assertions else {}
    reference = {
        "source_denominator": {
            "source_ids": [source.get("source_id") for source in sources],
            "denominator_n": denominator,
            "support": metric.get("support"),
            "population_total": population.get("total"),
        },
        "estimand_unit": {
            "estimand": estimand.get("estimand_id"),
            "analysis_unit": estimand.get("analysis_unit"),
            "resampling_unit": estimand.get("resampling_unit"),
            "pairing_key": estimand.get("pairing_key"),
        },
        "stats_multiplicity": {
            "statistic": uncertainty.get("method") or "not_declared",
            "paired": estimand.get("pairing_key") is not None,
            "resampling": uncertainty.get("method") or "not_declared",
            "multiplicity": multiplicity.get("method") or "not_declared",
            "multiplicity_language": multiplicity.get("method") or "not_declared",
        },
        "visual_semantics": {
            "chart_type": visual_contract.get("plot_type") if visual_contract else "not_available",
            "encoding": visual_contract.get("encodings") if visual_contract else "not_available",
            "ranking_supported": ranking_supported,
            "effect_direction": comparator.get("direction") if comparator else "not_declared",
            "metric_desirability": metric.get("desirability", "not_declared"),
        },
        "caption_accuracy": {
            "caption": caption.get("assertion_text", "not_available"),
            "status": caption.get("status", "not_available"),
        },
        "evidence_tier_availability": {
            "evidence_tier": evidence.get("tier"),
            "availability_status": "unavailable" if unavailable else "available",
            "execution_mode": execution_name,
            "reported_value": metric.get("effect"),
            "row_provenance": active_modes,
            "rows_disclosed": True,
        },
        "claim_boundary": {
            "causal_claim_allowed": causal_allowed,
            "null_result_claim": null_result_claim,
            "boundary": evidence.get("admission_state"),
            "allowed": claim_boundary.get("allowed"),
            "forbidden": claim_boundary.get("forbidden"),
        },
        "correction_usefulness": {"correction": "; ".join(packet.get("fail_closed_changes", []))},
    }
    return {
        "schema_version": EXPECTED_PACKET_SCHEMA,
        "artifact_kind": "evaluation_artifact",
        "packet_id": "clean",
        "claim_boundary": EXPECTED_PACKET_CLAIM_BOUNDARY,
        "reference": reference,
        "interpretation": json.loads(canonical_json(reference)),
        "source": {"source_id": packet_id, "packet_id": packet_id},
    }


def _canonical_mapping(packet: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = packet.get(key)
    if not isinstance(value, dict):
        raise AgentFigureEvalError(f"canonical packet {key} must be an object")
    return value


def _canonical_list_of_mappings(packet: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = packet.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        raise AgentFigureEvalError(f"canonical packet {key} must be a non-empty object list")
    return value


def _canonical_text(packet: Mapping[str, Any], key: str) -> str:
    value = packet.get(key)
    if not isinstance(value, str) or not value:
        raise AgentFigureEvalError(f"canonical packet {key} must be non-empty text")
    return value


def validate_candidate_envelope(envelope: Mapping[str, Any]) -> None:
    """Validate one provider-free candidate interpretation envelope.

    The envelope carries only a candidate interpretation. Reference answers
    are resolved from the digest-pinned manifest during replay and therefore
    cannot be supplied by a candidate.

    Raises:
        AgentFigureEvalError: If the envelope is not the exact supported
            candidate contract.
    """

    if not isinstance(envelope, Mapping):
        raise AgentFigureEvalError("candidate envelope must be an object")
    _validate_candidate_keys(envelope)
    _validate_candidate_identity(envelope)
    _validate_candidate_interpretation(envelope)
    _validate_candidate_context(envelope)
    _validate_candidate_findings(envelope)
    _validate_candidate_provenance(envelope)


def _validate_candidate_keys(envelope: Mapping[str, Any]) -> None:
    """Validate the exact top-level candidate envelope keys."""

    if set(envelope) != REQUIRED_CANDIDATE_KEYS:
        missing = sorted(REQUIRED_CANDIDATE_KEYS - set(envelope))
        extra = sorted(set(envelope) - REQUIRED_CANDIDATE_KEYS)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise AgentFigureEvalError(f"candidate envelope keys invalid: {'; '.join(details)}")


def _validate_candidate_identity(envelope: Mapping[str, Any]) -> None:
    """Validate provider, boundary, and fixture/mutation identity fields."""

    if envelope.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise AgentFigureEvalError(f"candidate schema_version must be {CANDIDATE_SCHEMA_VERSION!r}")
    if envelope.get("artifact_kind") != EXPECTED_CANDIDATE_ARTIFACT_KIND:
        raise AgentFigureEvalError(
            f"candidate artifact_kind must be {EXPECTED_CANDIDATE_ARTIFACT_KIND!r}"
        )
    if envelope.get("provider") != EXPECTED_CANDIDATE_PROVIDER:
        raise AgentFigureEvalError("candidate provider must be 'none'")
    if envelope.get("claim_boundary") != EXPECTED_PACKET_CLAIM_BOUNDARY:
        raise AgentFigureEvalError(
            "candidate claim_boundary must preserve the evaluation-artifacts-only boundary"
        )
    for key in ("fixture_id", "mutation_id"):
        value = envelope.get(key)
        if not isinstance(value, str) or not value:
            raise AgentFigureEvalError(f"candidate {key} must be a non-empty string")
    if envelope.get("verdict") != "pending":
        raise AgentFigureEvalError("candidate verdict must be 'pending' before replay")
    _validate_candidate_mutation(envelope)


def _validate_candidate_mutation(envelope: Mapping[str, Any]) -> None:
    """Validate the candidate mutation identity and expected detector list."""

    mutation = envelope.get("mutation")
    if not isinstance(mutation, Mapping) or set(mutation) != {"id", "expected_detectors"}:
        raise AgentFigureEvalError("candidate mutation must contain id and expected_detectors")
    if mutation.get("id") != envelope.get("mutation_id"):
        raise AgentFigureEvalError("candidate mutation.id must match mutation_id")
    detectors = mutation.get("expected_detectors")
    if not isinstance(detectors, list) or any(
        not isinstance(detector, str) or detector not in CRITICAL_ERROR_KINDS
        for detector in detectors
    ):
        raise AgentFigureEvalError(
            "candidate mutation.expected_detectors must name known detectors"
        )


def _validate_candidate_context(envelope: Mapping[str, Any]) -> None:
    """Validate workflow, figure, confidence, and explicit unavailable fields."""

    _validate_candidate_workflow_and_figure(envelope)
    _validate_candidate_text_lists(envelope)
    _validate_candidate_confidence(envelope)


def _validate_candidate_workflow_and_figure(envelope: Mapping[str, Any]) -> None:
    """Validate workflow identity and the figure specification/caption."""

    workflow = envelope.get("workflow")
    if not isinstance(workflow, Mapping) or set(workflow) != {"id", "revision"}:
        raise AgentFigureEvalError("candidate workflow must contain id and revision")
    for key in ("id", "revision"):
        if not isinstance(workflow.get(key), str) or not workflow[key]:
            raise AgentFigureEvalError(f"candidate workflow.{key} must be non-empty text")

    figure = envelope.get("figure")
    if not isinstance(figure, Mapping) or set(figure) != {"spec", "caption"}:
        raise AgentFigureEvalError("candidate figure must contain spec and caption")
    if not isinstance(figure.get("spec"), Mapping):
        raise AgentFigureEvalError("candidate figure.spec must be an object")
    if not isinstance(figure.get("caption"), str):
        raise AgentFigureEvalError("candidate figure.caption must be text")


def _validate_candidate_text_lists(envelope: Mapping[str, Any]) -> None:
    """Validate explicit list fields used for limitations and availability."""

    for key in ("limitations", "unresolved_questions", "unavailable", "not_applicable"):
        values = envelope.get(key)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise AgentFigureEvalError(f"candidate {key} must be a list of strings")


def _validate_candidate_confidence(envelope: Mapping[str, Any]) -> None:
    """Validate confidence status and its nullable numeric value."""

    confidence = envelope.get("confidence")
    if not isinstance(confidence, Mapping) or set(confidence) != {"status", "value"}:
        raise AgentFigureEvalError("candidate confidence must contain status and value")
    status = confidence.get("status")
    value = confidence.get("value")
    if status not in {"available", "not_available", "not_applicable"}:
        raise AgentFigureEvalError("candidate confidence.status is invalid")
    if status == "available":
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise AgentFigureEvalError("candidate confidence.value must be a number in [0, 1]")
    elif value is not None:
        raise AgentFigureEvalError("unavailable confidence must carry a null value")


def _validate_candidate_findings(envelope: Mapping[str, Any]) -> None:
    """Require per-dimension findings without allowing hidden aggregate results."""

    findings = envelope.get("findings")
    if not isinstance(findings, Mapping) or set(findings) != set(DIMENSIONS):
        raise AgentFigureEvalError("candidate findings must cover exactly all scoring dimensions")
    for dimension in DIMENSIONS:
        finding = findings[dimension]
        if not isinstance(finding, Mapping) or set(finding) != {"status", "critical"}:
            raise AgentFigureEvalError(
                f"candidate findings.{dimension} must contain status and critical"
            )
        if finding.get("status") not in {
            "available",
            "not_available",
            "not_applicable",
            "requires_semantic_review",
        }:
            raise AgentFigureEvalError(f"candidate findings.{dimension}.status is invalid")
        if not isinstance(finding.get("critical"), bool):
            raise AgentFigureEvalError(f"candidate findings.{dimension}.critical must be boolean")


def _validate_candidate_findings_against_case(
    envelope: Mapping[str, Any], case: Mapping[str, Any]
) -> None:
    """Require candidate critical flags to match deterministic detector output.

    Finding status remains candidate metadata because semantic language may require
    independent review, but deterministic critical flags are evaluator-owned facts.
    A candidate cannot make a replay appear clean by declaring all critical flags false.
    """

    critical_errors = case.get("critical_errors")
    if not isinstance(critical_errors, Mapping):
        raise AgentFigureEvalError("replay case critical_errors must be an object")
    expected_by_dimension = dict.fromkeys(DIMENSIONS, False)
    for kind, triggered in critical_errors.items():
        if not isinstance(triggered, bool) or kind not in CRITICAL_ERROR_DIMENSIONS:
            raise AgentFigureEvalError("replay case critical_errors has an invalid detector")
        if triggered:
            expected_by_dimension[CRITICAL_ERROR_DIMENSIONS[kind]] = True

    findings = envelope["findings"]
    for dimension, expected in expected_by_dimension.items():
        actual = findings[dimension]["critical"]
        if actual != expected:
            raise AgentFigureEvalError(
                f"candidate findings.{dimension}.critical must match deterministic detector "
                f"output ({expected})"
            )


def _validate_candidate_provenance(envelope: Mapping[str, Any]) -> None:
    """Validate explicit replay provenance and the no-provider boundary."""

    provenance = envelope.get("provenance")
    required = {
        "manifest_schema_version",
        "source_sha256",
        "packet_sha256",
        "reference_sha256",
        "candidate_sha256",
        "figure_sha256",
        "caption_sha256",
        "review_sha256",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != required:
        raise AgentFigureEvalError("candidate provenance must contain the complete digest contract")
    if provenance.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise AgentFigureEvalError("candidate provenance manifest schema mismatch")
    _validate_required_provenance_digests(provenance)
    _validate_optional_provenance_digests(provenance)
    _validate_candidate_replay_contract(envelope)


def _validate_required_provenance_digests(provenance: Mapping[str, Any]) -> None:
    """Validate digests that must always be bound to manifest or candidate bytes."""

    for key in ("source_sha256", "packet_sha256", "reference_sha256", "candidate_sha256"):
        if not isinstance(provenance.get(key), str) or not DIGEST_RE.fullmatch(provenance[key]):
            raise AgentFigureEvalError(f"candidate provenance.{key} must be a SHA-256 digest")


def _validate_optional_provenance_digests(provenance: Mapping[str, Any]) -> None:
    """Validate status-bearing figure, caption, and review digest records."""

    for key in ("figure_sha256", "caption_sha256", "review_sha256"):
        digest = provenance.get(key)
        if not isinstance(digest, Mapping) or set(digest) != {"status", "sha256"}:
            raise AgentFigureEvalError(f"candidate provenance.{key} must contain status and sha256")
        if digest.get("status") not in {"available", "not_available", "not_applicable"}:
            raise AgentFigureEvalError(f"candidate provenance.{key}.status is invalid")
        value = digest.get("sha256")
        if digest["status"] == "available":
            if not isinstance(value, str) or not DIGEST_RE.fullmatch(value):
                raise AgentFigureEvalError(f"candidate provenance.{key}.sha256 must be a digest")
        elif value is not None:
            raise AgentFigureEvalError(f"candidate provenance.{key} must use null when unavailable")


def _validate_candidate_replay_contract(envelope: Mapping[str, Any]) -> None:
    """Validate the provider-free deterministic replay declaration."""

    replay = envelope.get("replay_provenance")
    expected_replay = {
        "mode": "fixture",
        "deterministic": True,
        "external_provider_called": False,
        "network_access": "none",
    }
    if replay != expected_replay:
        raise AgentFigureEvalError(
            "candidate replay_provenance must declare provider-free fixture mode"
        )


def _validate_candidate_interpretation(envelope: Mapping[str, Any]) -> None:
    """Validate the candidate's complete dimension mapping."""

    interpretation = envelope.get("interpretation")
    if not isinstance(interpretation, Mapping):
        raise AgentFigureEvalError("candidate interpretation must be an object")
    if set(interpretation) != set(DIMENSIONS):
        raise AgentFigureEvalError(
            "candidate interpretation must contain exactly the declared scoring dimensions"
        )
    for dimension in DIMENSIONS:
        if not isinstance(interpretation[dimension], Mapping):
            raise AgentFigureEvalError(f"candidate interpretation.{dimension} must be an object")
    _validate_candidate_evidence_types(interpretation["evidence_tier_availability"])


def _validate_candidate_evidence_types(evidence: Mapping[str, Any]) -> None:
    """Validate typed fields consumed by deterministic evidence-tier detectors.

    The candidate schema intentionally leaves dimension payloads open for packet-specific
    fields, but the replay evaluator still consumes a small set of fields directly. Validate
    those fields here so malformed JSON fails with the evaluator's structured error instead of
    raising a native ``TypeError`` during scoring.
    """

    for key in ("evidence_tier", "availability_status", "execution_mode"):
        if key in evidence and not isinstance(evidence[key], str):
            raise AgentFigureEvalError(f"candidate evidence_tier_availability.{key} must be text")

    row_provenance = evidence.get("row_provenance")
    if row_provenance is not None and (
        not isinstance(row_provenance, list)
        or any(not isinstance(row, str) for row in row_provenance)
    ):
        raise AgentFigureEvalError(
            "candidate evidence_tier_availability.row_provenance must be a list of text"
        )

    rows_disclosed = evidence.get("rows_disclosed")
    if rows_disclosed is not None and not isinstance(rows_disclosed, bool):
        raise AgentFigureEvalError(
            "candidate evidence_tier_availability.rows_disclosed must be boolean"
        )

    reported_value = evidence.get("reported_value")
    if reported_value is not None and (
        not isinstance(reported_value, (int, float))
        or isinstance(reported_value, bool)
        or (isinstance(reported_value, float) and not math.isfinite(reported_value))
    ):
        raise AgentFigureEvalError(
            "candidate evidence_tier_availability.reported_value must be a finite number or null"
        )


def _apply_synthetic_mutation(packet: Mapping[str, Any], mutation_id: str) -> dict[str, Any]:
    """Apply one deterministic mutation to an ephemeral canonical projection.

    Returns:
        A canonicalized packet with the requested synthetic mutation applied.
    """

    mutator = _MUTATION_HANDLERS.get(mutation_id)
    if mutator is None:
        raise AgentFigureEvalError(f"unknown synthetic mutation {mutation_id!r}")
    mutated = json.loads(canonical_json(packet))
    mutator(mutated["interpretation"])
    return mutated


def _mutate_unavailable_to_zero(interpretation: dict[str, Any]) -> None:
    evidence = interpretation["evidence_tier_availability"]
    evidence["availability_status"] = "available"
    evidence["reported_value"] = 0


def _mutate_denominator_loss(interpretation: dict[str, Any]) -> None:
    source = interpretation["source_denominator"]
    source["denominator_n"] = max(0, source["denominator_n"] - 1)


def _mutate_analysis_unit(interpretation: dict[str, Any]) -> None:
    interpretation["estimand_unit"]["analysis_unit"] = "unpaired_episode"


def _mutate_pairing(interpretation: dict[str, Any]) -> None:
    stats = interpretation["stats_multiplicity"]
    stats["paired"] = False
    stats["resampling"] = "unpaired bootstrap"


def _mutate_fallback_promotion(interpretation: dict[str, Any]) -> None:
    evidence = interpretation["evidence_tier_availability"]
    evidence["execution_mode"] = "fallback"
    evidence["evidence_tier"] = "nominal benchmark evidence"


def _mutate_causal_overclaim(interpretation: dict[str, Any]) -> None:
    interpretation["claim_boundary"]["causal_claim_allowed"] = True


def _mutate_unsupported_ranking(interpretation: dict[str, Any]) -> None:
    interpretation["visual_semantics"]["ranking_supported"] = True


def _mutate_null_overclaim(interpretation: dict[str, Any]) -> None:
    interpretation["claim_boundary"]["null_result_claim"] = "supported_equivalence"


def _mutate_effect_direction(interpretation: dict[str, Any]) -> None:
    visual = interpretation["visual_semantics"]
    visual["effect_direction"] = "reversed"
    visual["metric_desirability"] = "reversed"


def _mutate_native_adapter_merge(interpretation: dict[str, Any]) -> None:
    evidence = interpretation["evidence_tier_availability"]
    evidence["row_provenance"] = ["native", "adapter"]
    evidence["rows_disclosed"] = False


def _mutate_multiplicity_language(interpretation: dict[str, Any]) -> None:
    interpretation["stats_multiplicity"]["multiplicity_language"] = "unadjusted comparisons"


_MUTATION_HANDLERS = {
    "unavailable_to_zero": _mutate_unavailable_to_zero,
    "denominator_loss": _mutate_denominator_loss,
    "analysis_unit_mismatch": _mutate_analysis_unit,
    "wrong_pairing_resampling": _mutate_pairing,
    "fallback_degraded_promotion": _mutate_fallback_promotion,
    "causal_overclaim": _mutate_causal_overclaim,
    "unsupported_ranking": _mutate_unsupported_ranking,
    "null_overclaim": _mutate_null_overclaim,
    "effect_direction_desirability": _mutate_effect_direction,
    "native_adapter_merge": _mutate_native_adapter_merge,
    "multiplicity_language": _mutate_multiplicity_language,
}


def list_fixture_mutations(manifest_path: Path) -> dict[str, Any]:
    """List verified source fixtures and their deterministic mutation detectors.

    The inventory is derived from committed packet/source bytes. It is a
    diagnostic inventory only; it does not select a preferred interpretation
    or promote any result to benchmark evidence.

    Returns:
        Deterministic fixture and mutation inventory.
    """

    verified_packets = load_verified_packets(manifest_path)
    fixture_mutations, mutation_records, seen_mutations = _inventory_packet_mutations(
        verified_packets
    )

    missing = sorted(set(REQUIRED_SCIENTIFIC_ERROR_MUTATIONS) - seen_mutations)
    if missing:
        raise AgentFigureEvalError(
            "manifest is missing required scientific-error mutations: " + ", ".join(missing)
        )
    return _mutation_inventory_result(fixture_mutations, mutation_records)


def _inventory_packet_mutations(
    verified_packets: list[tuple[Path, dict[str, Any]]],
) -> tuple[dict[str, set[str]], list[dict[str, Any]], set[str]]:
    """Validate and inventory canonical-packet mutation projections.

    Returns:
        Fixture-to-mutation mapping, public mutation records, and seen IDs.
    """

    fixture_mutations: dict[str, set[str]] = {}
    mutation_records: list[dict[str, Any]] = []
    seen_mutations: set[str] = set()
    for _, packet in verified_packets:
        source = _required_mapping(packet, "source")
        fixture_id = _required_text(source, "source_id")
        mutation_id = _required_text(packet, "packet_id")
        if mutation_id in seen_mutations:
            raise AgentFigureEvalError(f"duplicate mutation_id {mutation_id!r}")
        seen_mutations.add(mutation_id)
        fixture_mutations.setdefault(fixture_id, set()).add(mutation_id)

        canonical_case = evaluate_packet(packet)
        detectors = [kind for kind in CRITICAL_ERROR_KINDS if canonical_case.critical_errors[kind]]
        if mutation_id == "clean":
            if detectors:
                raise AgentFigureEvalError(
                    "clean mutation must not trigger a scientific-error detector"
                )
        elif mutation_id in REQUIRED_SCIENTIFIC_ERROR_MUTATIONS and detectors != [mutation_id]:
            raise AgentFigureEvalError(
                f"mutation {mutation_id!r} must trigger exactly its named detector"
            )
        elif not detectors:
            raise AgentFigureEvalError(
                f"mutation {mutation_id!r} has no deterministic scientific-error detector"
            )
        mutation_records.append(
            {
                "fixture_id": fixture_id,
                "mutation_id": mutation_id,
                "expected_detectors": detectors,
            }
        )
    return fixture_mutations, mutation_records, seen_mutations


def _mutation_inventory_result(
    fixture_mutations: Mapping[str, set[str]], mutation_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the public provider-free mutation inventory report.

    Returns:
        A deterministic diagnostic inventory report.
    """

    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": EXPECTED_MANIFEST_STATUS,
        "claim_boundary": EXPECTED_REPORT_CLAIM_BOUNDARY,
        "fixtures": [
            {"fixture_id": fixture_id, "mutation_ids": sorted(mutation_ids)}
            for fixture_id, mutation_ids in sorted(fixture_mutations.items())
        ],
        "mutations": sorted(mutation_records, key=lambda record: record["mutation_id"]),
        "integrity_mutations": [
            {
                "mutation_id": mutation_id,
                "expected_detectors": ["digest_drift"],
                "mode": "manifest_validation",
            }
            for mutation_id in INTEGRITY_MUTATION_IDS
        ],
    }


def replay_fixture_mutation(
    manifest_path: Path,
    candidate_envelope: Mapping[str, Any],
    *,
    fixture_id: str | None = None,
    mutation_id: str | None = None,
) -> dict[str, Any]:
    """Replay one candidate against one verified fixture/mutation pair.

    Returns:
        A deterministic diagnostic replay report.
    """

    validate_candidate_envelope(candidate_envelope)
    envelope_fixture_id = str(candidate_envelope["fixture_id"])
    envelope_mutation_id = str(candidate_envelope["mutation_id"])
    if fixture_id is not None and fixture_id != envelope_fixture_id:
        raise AgentFigureEvalError("requested fixture_id does not match candidate envelope")
    if mutation_id is not None and mutation_id != envelope_mutation_id:
        raise AgentFigureEvalError("requested mutation_id does not match candidate envelope")

    packets = _verified_packet_index(manifest_path)
    packet = packets.get((envelope_fixture_id, envelope_mutation_id))
    artifact_id = envelope_mutation_id
    if packet is None and (
        envelope_fixture_id == SYNTHETIC_MUTATION_FIXTURE_ID
        and envelope_mutation_id in SYNTHETIC_MUTATION_IDS
    ):
        packet = packets.get((envelope_fixture_id, "clean"))
        if packet is not None:
            packet = _apply_synthetic_mutation(packet, envelope_mutation_id)
            artifact_id = "clean"
    if packet is None:
        raise AgentFigureEvalError(
            f"unknown fixture/mutation pair: {envelope_fixture_id!r}/{envelope_mutation_id!r}"
        )

    artifact = _manifest_artifact(manifest_path, artifact_id)
    canonical_case = evaluate_packet(packet)
    expected_detectors = [
        kind for kind in CRITICAL_ERROR_KINDS if canonical_case.critical_errors[kind]
    ]
    candidate_mutation = candidate_envelope["mutation"]
    if candidate_mutation["expected_detectors"] != expected_detectors:
        raise AgentFigureEvalError(
            "candidate mutation.expected_detectors does not match the verified mutation"
        )
    _validate_replay_provenance(candidate_envelope, artifact)
    candidate_packet = dict(packet)
    candidate_packet["interpretation"] = dict(candidate_envelope["interpretation"])
    candidate_case = evaluate_packet(candidate_packet).to_dict()
    _validate_candidate_findings_against_case(candidate_envelope, candidate_case)
    detected_detectors = [
        kind for kind in CRITICAL_ERROR_KINDS if candidate_case["critical_errors"][kind]
    ]
    detector_match = detected_detectors == expected_detectors
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": EXPECTED_MANIFEST_STATUS,
        "claim_boundary": EXPECTED_REPORT_CLAIM_BOUNDARY,
        "mode": "single",
        "fixture_id": envelope_fixture_id,
        "mutation_id": envelope_mutation_id,
        "expected_detectors": expected_detectors,
        "detected_detectors": detected_detectors,
        "detector_status": "pass" if detector_match else "fail",
        "verdict": "pass" if detector_match else "fail",
        "replay_provenance": candidate_envelope["replay_provenance"],
        "provenance": _replay_report_provenance(manifest_path),
        "case": candidate_case,
    }


def _manifest_artifact(manifest_path: Path, artifact_id: str) -> dict[str, Any]:
    """Return the canonical packet record for replay provenance checks."""

    manifest = _load_eval_manifest(manifest_path)
    if artifact_id not in manifest["mutations"]:
        raise AgentFigureEvalError(f"manifest mutation {artifact_id!r} is missing")
    return manifest["packet"]


def _canonical_digest(value: Any) -> str:
    """Digest canonical JSON for in-envelope provenance values.

    Returns:
        A lowercase SHA-256 digest.
    """

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_replay_provenance(envelope: Mapping[str, Any], artifact: Mapping[str, Any]) -> None:
    """Bind candidate digests to the verified manifest and candidate bytes."""

    provenance = envelope["provenance"]
    expected = {
        "source_sha256": artifact.get("source_sha256"),
        "packet_sha256": artifact.get("sha256"),
        "reference_sha256": artifact.get("reference_sha256"),
        "candidate_sha256": _candidate_envelope_digest(envelope),
    }
    for key, value in expected.items():
        if provenance[key] != value:
            raise AgentFigureEvalError(f"candidate provenance.{key} does not match verified bytes")

    figure_digest = provenance["figure_sha256"]
    caption_digest = provenance["caption_sha256"]
    if figure_digest["status"] == "available" and figure_digest["sha256"] != _canonical_digest(
        envelope["figure"]["spec"]
    ):
        raise AgentFigureEvalError("candidate figure digest does not match figure.spec")
    if (
        caption_digest["status"] == "available"
        and caption_digest["sha256"]
        != hashlib.sha256(envelope["figure"]["caption"].encode("utf-8")).hexdigest()
    ):
        raise AgentFigureEvalError("candidate caption digest does not match figure.caption")
    review_digest = provenance["review_sha256"]
    if review_digest["status"] == "available" and review_digest["sha256"] != _canonical_digest(
        _review_digest_payload(envelope)
    ):
        raise AgentFigureEvalError("candidate review digest does not match post-review bytes")


def _candidate_envelope_digest(envelope: Mapping[str, Any]) -> str:
    """Digest candidate envelope bytes without a circular candidate digest field.

    Returns:
        A lowercase SHA-256 digest for the canonical candidate envelope.
    """

    payload = json.loads(canonical_json(envelope))
    payload["provenance"]["candidate_sha256"] = None
    return _canonical_digest(payload)


def _review_digest_payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic candidate fields covered by a post-review digest.

    Returns:
        The review-state payload whose canonical bytes are digest-bound.
    """

    return {
        "confidence": envelope["confidence"],
        "findings": envelope["findings"],
        "limitations": envelope["limitations"],
        "not_applicable": envelope["not_applicable"],
        "unavailable": envelope["unavailable"],
        "unresolved_questions": envelope["unresolved_questions"],
    }


def _replay_report_provenance(manifest_path: Path) -> dict[str, str]:
    """Bind replay output to evaluator code, manifest config, and fixture bytes.

    Returns:
        Stable SHA-256 provenance fields for a replay report.
    """

    manifest = _load_eval_manifest(manifest_path)
    packet = manifest["packet"]
    fixture_records = {
        "packet": {
            key: packet.get(key)
            for key in ("id", "path", "sha256", "source_sha256", "reference_sha256")
        },
        "mutations": manifest["mutations"],
    }
    return {
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(manifest_path),
        "fixture_sha256": _canonical_digest(fixture_records),
    }


def replay_all_fixture_mutations(
    manifest_path: Path,
    candidate_envelopes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay a complete candidate envelope set against the verified corpus.

    Returns:
        A deterministic diagnostic replay-all report.
    """

    if isinstance(candidate_envelopes, (str, bytes)) or not isinstance(
        candidate_envelopes, Sequence
    ):
        raise AgentFigureEvalError("replay-all candidates must be a JSON array")
    expected_pairs = {
        (record["fixture_id"], record["mutation_id"])
        for record in list_fixture_mutations(manifest_path)["mutations"]
    }
    validated: list[Mapping[str, Any]] = []
    observed_pairs: set[tuple[str, str]] = set()
    for index, envelope in enumerate(candidate_envelopes):
        try:
            validate_candidate_envelope(envelope)
        except AgentFigureEvalError as exc:
            raise AgentFigureEvalError(f"candidate {index}: {exc}") from exc
        pair = (str(envelope["fixture_id"]), str(envelope["mutation_id"]))
        if pair in observed_pairs:
            raise AgentFigureEvalError(f"replay-all contains duplicate pair {pair!r}")
        observed_pairs.add(pair)
        validated.append(envelope)
    missing = sorted(expected_pairs - observed_pairs)
    extra = sorted(observed_pairs - expected_pairs)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing pairs: {missing}")
        if extra:
            details.append(f"unexpected pairs: {extra}")
        raise AgentFigureEvalError("replay-all coverage mismatch: " + "; ".join(details))

    cases = [replay_fixture_mutation(manifest_path, envelope) for envelope in validated]
    cases.sort(key=lambda case: (case["fixture_id"], case["mutation_id"]))
    failed_count = sum(case["detector_status"] != "pass" for case in cases)
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": EXPECTED_MANIFEST_STATUS,
        "claim_boundary": EXPECTED_REPORT_CLAIM_BOUNDARY,
        "mode": "all",
        "case_count": len(cases),
        "passed_case_count": len(cases) - failed_count,
        "failed_case_count": failed_count,
        "detector_status": "pass" if failed_count == 0 else "fail",
        "provenance": _replay_report_provenance(manifest_path),
        "cases": cases,
    }


def _verified_packet_index(manifest_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Index verified packets by source fixture and mutation identifiers.

    Returns:
        Mapping from fixture/mutation pairs to verified packets.
    """

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for _, packet in load_verified_packets(manifest_path):
        source = _required_mapping(packet, "source")
        fixture_id = _required_text(source, "source_id")
        mutation_id = _required_text(packet, "packet_id")
        pair = (fixture_id, mutation_id)
        if pair in index:
            raise AgentFigureEvalError(f"duplicate fixture/mutation pair {pair!r}")
        index[pair] = packet
    return index


def evaluate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Evaluate every verified fixture packet in *manifest_path*.

    Returns:
        JSON-serializable evaluation report.
    """

    case_results = [
        evaluate_packet(packet).to_dict() for _, packet in load_verified_packets(manifest_path)
    ]
    critical_counts = {
        kind: sum(1 for case in case_results if case["critical_errors"][kind])
        for kind in CRITICAL_ERROR_KINDS
    }
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "status": "evaluation_artifacts_only",
        "claim_boundary": EXPECTED_REPORT_CLAIM_BOUNDARY,
        "case_count": len(case_results),
        "critical_error_counts": critical_counts,
        "aggregate_summary": _aggregate_summary(case_results, critical_counts),
        "cases": case_results,
    }


def _aggregate_summary(
    case_results: list[dict[str, Any]], critical_counts: dict[str, int]
) -> dict[str, Any]:
    """Summarize the fixture corpus without collapsing critical errors.

    The summary is deliberately descriptive. It reports per-dimension pass
    rates, exact critical-error examples, reviewer coverage, and the status of
    any paired workflow variants. It does not choose a preferred workflow or
    promote a fixture result to benchmark evidence.

    Returns:
        Deterministic corpus-level evaluation metadata.
    """

    case_status_counts = {"clean": 0, "failed": 0}
    dimension_pass_counts = dict.fromkeys(DIMENSIONS, 0)
    critical_failure_examples = {kind: [] for kind in CRITICAL_ERROR_KINDS}

    for case in case_results:
        status = case["status"]
        if status in case_status_counts:
            case_status_counts[status] += 1
        for score in case["scores"]:
            if score["passed"]:
                dimension_pass_counts[score["dimension"]] += 1
        for kind in CRITICAL_ERROR_KINDS:
            if case["critical_errors"][kind]:
                critical_failure_examples[kind].append(case["packet_id"])

    case_count = len(case_results)
    dimension_scores = {
        dimension: {
            "case_count": case_count,
            "passed_count": passed_count,
            "failed_count": case_count - passed_count,
            "pass_rate": passed_count / case_count if case_count else None,
        }
        for dimension, passed_count in dimension_pass_counts.items()
    }

    reviewer_records = [
        case["reviewer_accounting"]
        for case in case_results
        if case.get("reviewer_accounting") is not None
    ]
    reviewer_status = (
        "not_available"
        if not reviewer_records
        else "available"
        if len(reviewer_records) == case_count
        else "partial"
    )
    reviewer_summary = {
        "status": reviewer_status,
        "reviewed_case_count": len(reviewer_records),
        "adjudication_complete_case_count": sum(
            1 for record in reviewer_records if record["adjudication_complete"]
        ),
        "disagreement_count": sum(record["disagreement_count"] for record in reviewer_records),
        "mean_agreement_rate": (
            sum(record["agreement_rate"] for record in reviewer_records) / len(reviewer_records)
            if reviewer_records
            else None
        ),
    }

    variant_records = [
        case["interpretation_variant_comparison"]
        for case in case_results
        if case.get("interpretation_variant_comparison") is not None
    ]
    if not variant_records:
        workflow_variants = {
            "status": "not_available",
            "paired_case_count": 0,
            "baseline_case_count": 0,
            "packet_constrained_case_count": 0,
            "mean_aggregate_score_delta": None,
            "critical_error_count_delta": None,
            "packet_constrained_reduces_critical_errors": None,
            "packet_constrained_preserves_source_fidelity": None,
        }
    else:
        deltas = [record["delta"] for record in variant_records]
        workflow_variants = {
            "status": "available" if len(variant_records) == case_count else "partial",
            "paired_case_count": len(variant_records),
            "baseline_case_count": len(variant_records),
            "packet_constrained_case_count": len(variant_records),
            "mean_aggregate_score_delta": sum(delta["aggregate_score"] for delta in deltas)
            / len(deltas),
            "critical_error_count_delta": sum(delta["critical_error_count"] for delta in deltas),
            "packet_constrained_reduces_critical_errors": all(
                delta["packet_constrained_reduces_critical_errors"] for delta in deltas
            ),
            "packet_constrained_preserves_source_fidelity": all(
                delta["packet_constrained_preserves_source_fidelity"] for delta in deltas
            ),
        }

    return {
        "case_status_counts": case_status_counts,
        "dimension_scores": dimension_scores,
        "critical_error_counts": critical_counts,
        "critical_failure_examples": critical_failure_examples,
        "reviewer_accounting": reviewer_summary,
        "workflow_variants": workflow_variants,
    }


def evaluate_packet(packet: dict[str, Any]) -> CaseEvaluation:
    """Score one ephemeral interpretation projection.

    Returns:
        Per-case evaluation with dimension scores and critical flags.
    """

    if packet.get("schema_version") != EXPECTED_PACKET_SCHEMA:
        raise AgentFigureEvalError(f"packet schema_version must be {EXPECTED_PACKET_SCHEMA!r}")
    if packet.get("artifact_kind") != "evaluation_artifact":
        raise AgentFigureEvalError("packet artifact_kind must be evaluation_artifact")
    packet_id = _required_text(packet, "packet_id")
    claim_boundary = _required_text(packet, "claim_boundary")
    if claim_boundary != EXPECTED_PACKET_CLAIM_BOUNDARY:
        raise AgentFigureEvalError(
            "packet claim_boundary must preserve the evaluation-artifacts-only boundary"
        )
    reference = _required_mapping(packet, "reference")
    observed = _required_mapping(packet, "interpretation")
    scores = _dimension_scores(reference, observed)
    critical_errors = _critical_errors(reference, observed)
    aggregate_score = sum(score.score for score in scores) / len(scores)
    status = "clean" if aggregate_score == 1.0 and not any(critical_errors.values()) else "failed"
    variant_comparison = _interpretation_variant_comparison(packet, reference)
    reviewer_accounting = _reviewer_accounting(packet)
    correction_ranking = _correction_priority_ranking(packet, scores, critical_errors)
    return CaseEvaluation(
        packet_id=packet_id,
        artifact_kind=_required_text(packet, "artifact_kind"),
        status=status,
        scores=scores,
        critical_errors=critical_errors,
        aggregate_score=aggregate_score,
        claim_boundary=claim_boundary,
        interpretation_variant_comparison=variant_comparison,
        reviewer_accounting=reviewer_accounting,
        correction_priority_ranking=correction_ranking,
    )


def _dimension_scores(reference: dict[str, Any], observed: dict[str, Any]) -> list[DimensionScore]:
    reference_dimensions = _required_dimension_mappings(reference, "reference")
    observed_dimensions = _required_dimension_mappings(observed, "interpretation")
    return [
        DimensionScore(
            dimension=dimension,
            score=1.0 if reference_dimensions[dimension] == observed_dimensions[dimension] else 0.0,
            passed=reference_dimensions[dimension] == observed_dimensions[dimension],
            expected=reference_dimensions[dimension],
            observed=observed_dimensions[dimension],
        )
        for dimension in DIMENSIONS
    ]


def _has_fallback_or_degraded_evidence(evidence: Mapping[str, Any]) -> bool:
    """Return whether execution or any disclosed row is fallback/degraded."""

    row_provenance = evidence.get("row_provenance", [])
    return evidence.get("execution_mode") in _FALLBACK_DEGRADED_MODES or (
        isinstance(row_provenance, list)
        and any(row in _FALLBACK_DEGRADED_MODES for row in row_provenance)
    )


def _critical_errors(reference: dict[str, Any], observed: dict[str, Any]) -> dict[str, bool]:
    ref_evidence = _required_mapping(
        reference, "evidence_tier_availability", label="reference.evidence_tier_availability"
    )
    obs_evidence = _required_mapping(
        observed, "evidence_tier_availability", label="interpretation.evidence_tier_availability"
    )
    ref_stats = _required_mapping(
        reference, "stats_multiplicity", label="reference.stats_multiplicity"
    )
    obs_stats = _required_mapping(
        observed, "stats_multiplicity", label="interpretation.stats_multiplicity"
    )
    ref_source = _required_mapping(
        reference, "source_denominator", label="reference.source_denominator"
    )
    obs_source = _required_mapping(
        observed, "source_denominator", label="interpretation.source_denominator"
    )
    ref_estimand = _required_mapping(reference, "estimand_unit", label="reference.estimand_unit")
    obs_estimand = _required_mapping(
        observed, "estimand_unit", label="interpretation.estimand_unit"
    )
    ref_boundary = _required_mapping(reference, "claim_boundary", label="reference.claim_boundary")
    obs_boundary = _required_mapping(
        observed, "claim_boundary", label="interpretation.claim_boundary"
    )
    ref_visual = _required_mapping(
        reference, "visual_semantics", label="reference.visual_semantics"
    )
    obs_visual = _required_mapping(
        observed, "visual_semantics", label="interpretation.visual_semantics"
    )

    return {
        "unavailable_to_zero": (
            ref_evidence.get("availability_status") == "unavailable"
            and (
                obs_evidence.get("availability_status") != "unavailable"
                or obs_evidence.get("reported_value") == 0
            )
        ),
        "denominator_loss": ref_source.get("denominator_n") != obs_source.get("denominator_n"),
        "analysis_unit_mismatch": (
            ref_estimand.get("analysis_unit") != obs_estimand.get("analysis_unit")
        ),
        "wrong_pairing_resampling": (
            ref_stats.get("paired") != obs_stats.get("paired")
            or ref_stats.get("resampling") != obs_stats.get("resampling")
        ),
        "fallback_degraded_promotion": (
            (
                _has_fallback_or_degraded_evidence(obs_evidence)
                or _has_fallback_or_degraded_evidence(ref_evidence)
            )
            and obs_evidence.get("evidence_tier") in _HIGHER_THAN_DIAGNOSTIC
        ),
        "causal_overclaim": (
            ref_boundary.get("causal_claim_allowed") is False
            and obs_boundary.get("causal_claim_allowed") is True
        ),
        "unsupported_ranking": (
            ref_visual.get("ranking_supported") is False
            and obs_visual.get("ranking_supported") is True
        ),
        "null_overclaim": (
            ref_boundary.get("null_result_claim") == "not_supported"
            and obs_boundary.get("null_result_claim") != "not_supported"
        ),
        "effect_direction_desirability": (
            ref_visual.get("effect_direction", "not_declared")
            != obs_visual.get("effect_direction", "not_declared")
            or ref_visual.get("metric_desirability", "not_declared")
            != obs_visual.get("metric_desirability", "not_declared")
        ),
        "native_adapter_merge": (
            len(obs_evidence.get("row_provenance", [])) > 1
            and obs_evidence.get("rows_disclosed") is False
        ),
        "multiplicity_language": (
            ref_stats.get("multiplicity_language", "not_declared")
            != obs_stats.get("multiplicity_language", "not_declared")
        ),
    }


def _interpretation_variant_comparison(
    packet: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any] | None:
    variants = packet.get("interpretation_variants")
    if variants is None:
        return None
    if not isinstance(variants, dict):
        raise AgentFigureEvalError("interpretation_variants must be an object")
    if set(variants) != set(INTERPRETATION_VARIANTS):
        raise AgentFigureEvalError(
            "interpretation_variants must contain exactly baseline and packet_constrained"
        )

    summaries: dict[str, dict[str, Any]] = {}
    for variant in INTERPRETATION_VARIANTS:
        observed = variants[variant]
        if not isinstance(observed, dict):
            raise AgentFigureEvalError(f"interpretation_variants.{variant} must be an object")
        scores = _dimension_scores(reference, observed)
        critical_errors = _critical_errors(reference, observed)
        summaries[variant] = {
            "aggregate_score": sum(score.score for score in scores) / len(scores),
            "critical_error_count": sum(1 for triggered in critical_errors.values() if triggered),
            "dimension_failures": [score.dimension for score in scores if not score.passed],
            "critical_errors": critical_errors,
        }

    baseline = summaries["baseline"]
    packet_constrained = summaries["packet_constrained"]
    return {
        "baseline": baseline,
        "packet_constrained": packet_constrained,
        "delta": {
            "aggregate_score": (
                packet_constrained["aggregate_score"] - baseline["aggregate_score"]
            ),
            "critical_error_count": (
                packet_constrained["critical_error_count"] - baseline["critical_error_count"]
            ),
            "packet_constrained_reduces_critical_errors": (
                packet_constrained["critical_error_count"] < baseline["critical_error_count"]
            ),
            "packet_constrained_preserves_source_fidelity": (
                "source_denominator" not in packet_constrained["dimension_failures"]
            ),
        },
    }


def _reviewer_accounting(packet: dict[str, Any]) -> dict[str, Any] | None:
    metadata = packet.get("reference_metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise AgentFigureEvalError("reference_metadata must be an object")
    _validate_reviewer_metadata(metadata)
    reviewer_scores = _load_reviewer_scores(metadata["reviewers"])
    disagreements = _reviewer_disagreements(reviewer_scores)
    adjudicated_dimensions = _adjudicated_dimensions(metadata, disagreements)

    agreed_dimensions = len(DIMENSIONS) - len(disagreements)
    return {
        "reviewed": True,
        "blinded": True,
        "reviewer_count": len(reviewer_scores),
        "dimension_count": len(DIMENSIONS),
        "agreed_dimensions": agreed_dimensions,
        "disagreement_count": len(disagreements),
        "agreement_rate": agreed_dimensions / len(DIMENSIONS),
        "disagreements": disagreements,
        "adjudicated_dimensions": adjudicated_dimensions,
        "adjudication_complete": len(adjudicated_dimensions) == len(disagreements),
    }


def _validate_reviewer_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("reviewed") is not True:
        raise AgentFigureEvalError("reference_metadata.reviewed must be true")
    if metadata.get("blinded") is not True:
        raise AgentFigureEvalError("reference_metadata.blinded must be true")
    reviewers = metadata.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) < 2:
        raise AgentFigureEvalError(
            "reference_metadata.reviewers must contain at least two reviewers"
        )


def _load_reviewer_scores(reviewers: Any) -> dict[str, dict[str, float]]:
    reviewer_scores: dict[str, dict[str, float]] = {}
    for index, reviewer in enumerate(reviewers):
        if not isinstance(reviewer, dict):
            raise AgentFigureEvalError(f"reference_metadata.reviewers[{index}] must be an object")
        reviewer_id = reviewer.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id:
            raise AgentFigureEvalError(
                f"reference_metadata.reviewers[{index}].reviewer_id must be a string"
            )
        if reviewer_id in reviewer_scores:
            raise AgentFigureEvalError(f"duplicate reviewer_id {reviewer_id!r}")
        scores = reviewer.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
            raise AgentFigureEvalError(
                f"reference_metadata.reviewers[{index}].scores must cover all dimensions"
            )
        reviewer_scores[reviewer_id] = {
            dimension: _review_score(scores[dimension], reviewer_id, dimension)
            for dimension in DIMENSIONS
        }
    return reviewer_scores


def _reviewer_disagreements(
    reviewer_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    disagreements: dict[str, dict[str, float]] = {}
    for dimension in DIMENSIONS:
        dimension_scores = {
            reviewer_id: scores[dimension] for reviewer_id, scores in reviewer_scores.items()
        }
        if len(set(dimension_scores.values())) > 1:
            disagreements[dimension] = dimension_scores
    return disagreements


def _adjudicated_dimensions(
    metadata: dict[str, Any], disagreements: dict[str, dict[str, float]]
) -> list[str]:
    adjudication = metadata.get("adjudication")
    if not disagreements:
        if adjudication is not None and not isinstance(adjudication, dict):
            raise AgentFigureEvalError("reference_metadata.adjudication must be an object")
        return []
    if not isinstance(adjudication, dict):
        raise AgentFigureEvalError("reference_metadata.adjudication required for disagreements")
    adjudicator_id = adjudication.get("adjudicator_id")
    if not isinstance(adjudicator_id, str) or not adjudicator_id:
        raise AgentFigureEvalError("reference_metadata.adjudication.adjudicator_id required")
    resolved_scores = adjudication.get("resolved_scores")
    if not isinstance(resolved_scores, dict):
        raise AgentFigureEvalError("reference_metadata.adjudication.resolved_scores required")
    if set(resolved_scores) != set(disagreements):
        raise AgentFigureEvalError(
            "reference_metadata.adjudication.resolved_scores must exactly cover disagreements"
        )
    return [
        dimension
        for dimension in DIMENSIONS
        if dimension in disagreements
        and _review_score(resolved_scores[dimension], adjudicator_id, dimension) >= 0
    ]


def _review_score(value: Any, reviewer_id: str, dimension: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
        or value > 1
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise AgentFigureEvalError(
            f"review score for {reviewer_id}.{dimension} must be a number in [0, 1]"
        )
    return float(value)


def _correction_priority_ranking(
    packet: dict[str, Any],
    scores: list[DimensionScore],
    critical_errors: dict[str, bool],
) -> list[dict[str, Any]] | None:
    candidates = packet.get("correction_candidates")
    if candidates is None:
        return None
    if not isinstance(candidates, list) or not candidates:
        raise AgentFigureEvalError("correction_candidates must be a non-empty list")

    failed_dimensions = {score.dimension for score in scores if not score.passed}
    critical_dimensions = {
        CRITICAL_ERROR_DIMENSIONS[kind] for kind, triggered in critical_errors.items() if triggered
    }
    ranked: list[tuple[tuple[int, int, int, int, str], dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise AgentFigureEvalError(f"correction_candidates[{index}] must be an object")
        candidate_id = candidate.get("id")
        dimension = candidate.get("dimension")
        severity = candidate.get("severity")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise AgentFigureEvalError(f"correction_candidates[{index}].id must be a string")
        if candidate_id in seen_ids:
            raise AgentFigureEvalError(f"duplicate correction candidate id {candidate_id!r}")
        seen_ids.add(candidate_id)
        if dimension not in DIMENSIONS:
            raise AgentFigureEvalError(
                f"correction_candidates[{index}].dimension must be a scoring dimension"
            )
        if severity not in SEVERITY_ORDER:
            raise AgentFigureEvalError(
                f"correction_candidates[{index}].severity must be critical, major, or minor"
            )
        payload = {
            "id": candidate_id,
            "dimension": dimension,
            "severity": severity,
            "triggered_by_critical_error": dimension in critical_dimensions,
            "dimension_failed": dimension in failed_dimensions,
        }
        sort_key = (
            0 if dimension in critical_dimensions else 1,
            0 if dimension in failed_dimensions else 1,
            SEVERITY_ORDER[severity],
            DIMENSIONS.index(dimension),
            candidate_id,
        )
        ranked.append((sort_key, payload))

    return [
        {"rank": rank, **payload}
        for rank, (_, payload) in enumerate(sorted(ranked, key=lambda item: item[0]), start=1)
    ]


def _verified_manifest_file(
    *,
    manifest_path: Path,
    artifact: dict[str, Any],
    index: int,
    path_key: str,
    sha_key: str,
) -> Path:
    """Resolve one manifest file reference after SHA-256 verification.

    Returns:
        Resolved file path.
    """

    rel_path = artifact.get(path_key)
    expected_sha = artifact.get(sha_key)
    if not isinstance(rel_path, str) or not rel_path:
        raise AgentFigureEvalError(f"manifest artifact {index}: missing {path_key}")
    if not isinstance(expected_sha, str) or not DIGEST_RE.fullmatch(expected_sha):
        raise AgentFigureEvalError(f"manifest artifact {index}: missing {sha_key}")

    path = _resolve_manifest_path(
        manifest_path=manifest_path,
        rel_path=rel_path,
        index=index,
        path_key=path_key,
    )
    if not path.is_file():
        raise AgentFigureEvalError(f"manifest artifact {index}: missing file {rel_path}")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise AgentFigureEvalError(
            f"manifest artifact {index}: sha256 mismatch for {rel_path}: "
            f"expected {expected_sha}, observed {observed_sha}"
        )
    return path


def _resolve_manifest_path(
    *, manifest_path: Path, rel_path: str, index: int, path_key: str
) -> Path:
    """Resolve one manifest path within its fixture root without symlink hops.

    Returns:
        Resolved path within the manifest directory.
    """

    candidate = Path(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AgentFigureEvalError(
            f"manifest artifact {index}: {path_key} must be repository-relative without traversal"
        )
    if any(part in LOCAL_ONLY_MANIFEST_PARTS for part in candidate.parts):
        raise AgentFigureEvalError(
            f"manifest artifact {index}: {path_key} points to a local-only path"
        )
    root = manifest_path.parent.resolve()
    unresolved = root / candidate
    current = root
    try:
        for part in candidate.parts:
            current /= part
            if current.is_symlink():
                raise AgentFigureEvalError(
                    f"manifest artifact {index}: {path_key} must not traverse a symlink"
                )
        path = unresolved.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise AgentFigureEvalError(
            f"manifest artifact {index}: {path_key} cannot be resolved"
        ) from exc
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AgentFigureEvalError(
            f"manifest artifact {index}: {path_key} resolves outside the manifest root"
        ) from exc
    return path


def _required_mapping(
    data: dict[str, Any], key: str, *, label: str | None = None
) -> dict[str, Any]:
    value = data.get(key) if isinstance(data, dict) else None
    if not isinstance(value, dict):
        raise AgentFigureEvalError(f"packet missing object field {label or key!r}")
    return value


def _required_dimension_mappings(data: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    """Require every declared scoring dimension before comparing fixture fields.

    Returns:
        Mapping from each declared dimension to its validated object value.
    """

    return {
        dimension: _required_mapping(data, dimension, label=f"{label}.{dimension}")
        for dimension in DIMENSIONS
    }


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise AgentFigureEvalError(f"packet missing string field {key!r}")
    return value


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "CRITICAL_ERROR_KINDS",
    "DIMENSIONS",
    "EVAL_SCHEMA_VERSION",
    "EXPECTED_PACKET_SCHEMA",
    "INTEGRITY_MUTATION_IDS",
    "MANIFEST_SCHEMA_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "REQUIRED_SCIENTIFIC_ERROR_MUTATIONS",
    "SYNTHETIC_MUTATION_IDS",
    "AgentFigureEvalError",
    "CaseEvaluation",
    "DimensionScore",
    "canonical_json",
    "evaluate_manifest",
    "evaluate_packet",
    "list_fixture_mutations",
    "load_verified_packets",
    "replay_all_fixture_mutations",
    "replay_fixture_mutation",
    "sha256_file",
    "validate_candidate_envelope",
]
