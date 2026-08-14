"""Deterministic evaluation fixtures for agent interpretation of figure packets.

This module scores frozen, packet-shaped JSON fixtures only. It does not call
external providers, read generated benchmark packets from other branches, or
promote fixture outputs as benchmark evidence. Optional workflow variants,
reviewer accounting, and correction rankings are still fixture metadata and
must not be read as benchmark or scientific results.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EVAL_SCHEMA_VERSION = "agent_figure_interpretation_eval.v1"
MANIFEST_SCHEMA_VERSION = "agent_figure_interpretation_eval_manifest.v1"
EXPECTED_PACKET_SCHEMA = "result_interpretation_packet.v1"
EXPECTED_MANIFEST_STATUS = "evaluation_artifacts_only"
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
    "wrong_pairing_resampling",
    "fallback_degraded_promotion",
    "causal_overclaim",
    "unsupported_ranking",
    "null_overclaim",
)
INTERPRETATION_VARIANTS = ("baseline", "packet_constrained")
SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}
CRITICAL_ERROR_DIMENSIONS = {
    "unavailable_to_zero": "evidence_tier_availability",
    "denominator_loss": "source_denominator",
    "wrong_pairing_resampling": "stats_multiplicity",
    "fallback_degraded_promotion": "evidence_tier_availability",
    "causal_overclaim": "claim_boundary",
    "unsupported_ranking": "visual_semantics",
    "null_overclaim": "claim_boundary",
}
_HIGHER_THAN_DIAGNOSTIC = {
    "smoke",
    "smoke evidence",
    "benchmark",
    "nominal benchmark evidence",
    "paper_facing",
    "paper-grade",
    "paper-grade evidence",
}


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
    """Load all fixture packets from a digest-pinned manifest.

    The manifest is intentionally small and provider-independent:
    ``expected_packet_schema`` must exactly match
    :data:`EXPECTED_PACKET_SCHEMA`, and every packet, source, and reference
    digest must match the bytes currently on disk.

    Returns:
        Pairs of resolved packet path and parsed packet payload.
    """

    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise AgentFigureEvalError("manifest schema_version mismatch")
    if manifest.get("status") != EXPECTED_MANIFEST_STATUS:
        raise AgentFigureEvalError(f"manifest status must be {EXPECTED_MANIFEST_STATUS!r}")
    if not isinstance(manifest.get("claim_boundary"), str) or not manifest["claim_boundary"]:
        raise AgentFigureEvalError("manifest claim_boundary must be a non-empty string")
    expected_schema = manifest.get("expected_packet_schema")
    if expected_schema != EXPECTED_PACKET_SCHEMA:
        raise AgentFigureEvalError(
            "manifest expected_packet_schema must be "
            f"{EXPECTED_PACKET_SCHEMA!r}, got {expected_schema!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AgentFigureEvalError("manifest artifacts must be a non-empty list")

    packets: list[tuple[Path, dict[str, Any]]] = []
    artifact_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise AgentFigureEvalError(f"manifest artifact {index}: expected object")
        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise AgentFigureEvalError(f"manifest artifact {index}: id must be a non-empty string")
        if artifact_id in artifact_ids:
            raise AgentFigureEvalError(f"manifest artifact {index}: duplicate id {artifact_id!r}")
        artifact_ids.add(artifact_id)
        packets.append(
            _load_verified_packet(
                manifest_path=manifest_path,
                artifact=artifact,
                index=index,
                expected_schema=expected_schema,
            )
        )
    return packets


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
        "claim_boundary": (
            "fixture replay only; no external model calls, no benchmark claims, "
            "and no generated evidence promotion"
        ),
        "case_count": len(case_results),
        "critical_error_counts": critical_counts,
        "cases": case_results,
    }


def evaluate_packet(packet: dict[str, Any]) -> CaseEvaluation:
    """Score one frozen packet-shaped interpretation fixture.

    Returns:
        Per-case evaluation with dimension scores and critical flags.
    """

    if packet.get("schema_version") != EXPECTED_PACKET_SCHEMA:
        raise AgentFigureEvalError(f"packet schema_version must be {EXPECTED_PACKET_SCHEMA!r}")
    if packet.get("artifact_kind") != "evaluation_artifact":
        raise AgentFigureEvalError("packet artifact_kind must be evaluation_artifact")
    packet_id = _required_text(packet, "packet_id")
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
        claim_boundary=_required_text(packet, "claim_boundary"),
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
        "wrong_pairing_resampling": (
            ref_stats.get("paired") != obs_stats.get("paired")
            or ref_stats.get("resampling") != obs_stats.get("resampling")
        ),
        "fallback_degraded_promotion": (
            ref_evidence.get("execution_mode") in {"fallback", "degraded"}
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
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 or value > 1:
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


def _load_verified_packet(
    *,
    manifest_path: Path,
    artifact: Any,
    index: int,
    expected_schema: str,
) -> tuple[Path, dict[str, Any]]:
    """Load one digest-pinned fixture packet and its source/reference contracts.

    Returns:
        Resolved packet path and parsed packet payload.
    """

    if not isinstance(artifact, dict):
        raise AgentFigureEvalError(f"manifest artifact {index}: expected object")
    path = _verified_manifest_file(
        manifest_path=manifest_path,
        artifact=artifact,
        index=index,
        path_key="path",
        sha_key="sha256",
    )
    source_path = _verified_manifest_file(
        manifest_path=manifest_path,
        artifact=artifact,
        index=index,
        path_key="source_path",
        sha_key="source_sha256",
    )
    reference_path = _verified_manifest_file(
        manifest_path=manifest_path,
        artifact=artifact,
        index=index,
        path_key="reference_path",
        sha_key="reference_sha256",
    )
    packet = load_json(path)
    packet_schema = packet.get("schema_version")
    if packet_schema != expected_schema:
        raise AgentFigureEvalError(
            f"{artifact['path']}: packet schema_version must be {expected_schema!r}, "
            f"got {packet_schema!r}"
        )
    if packet.get("artifact_kind") != "evaluation_artifact":
        raise AgentFigureEvalError(
            f"{artifact['path']}: fixture must be labeled evaluation_artifact"
        )
    artifact_id = artifact.get("id")
    if packet.get("packet_id") != artifact_id:
        raise AgentFigureEvalError(
            f"{artifact['path']}: packet_id must match manifest artifact id {artifact_id!r}"
        )
    source = load_json(source_path)
    reference = load_json(reference_path)
    if packet.get("source") != source:
        raise AgentFigureEvalError(
            f"{artifact['path']}: source fixture does not match packet source"
        )
    if packet.get("reference") != reference:
        raise AgentFigureEvalError(
            f"{artifact['path']}: reference fixture does not match packet reference"
        )
    if source.get("packet_id") != artifact_id:
        raise AgentFigureEvalError(
            f"{artifact['path']}: source packet_id must match manifest artifact id {artifact_id!r}"
        )
    return path, packet


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
    "CRITICAL_ERROR_KINDS",
    "DIMENSIONS",
    "EVAL_SCHEMA_VERSION",
    "EXPECTED_PACKET_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "AgentFigureEvalError",
    "CaseEvaluation",
    "DimensionScore",
    "canonical_json",
    "evaluate_manifest",
    "evaluate_packet",
    "load_verified_packets",
    "sha256_file",
]
