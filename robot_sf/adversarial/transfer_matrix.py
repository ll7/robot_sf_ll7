"""Cross-planner adversarial transfer matrix (capability-only, Gate A).

Slice 1 of issue #5303 (cheap-lane, capability-only) plus the Gate A
transfer-contract repair from issue #6146. This module *measures* transfer
structure: it reuses the existing adversarial archive of certified worst-case
configs found against ONE target planner and builds the K x N transfer matrix —
does a discovered weak point transfer to the other planners, or is it
policy-specific?

It deliberately does NOT run the minimax search game (that is a later slice,
only if slice 1 shows meaningful transfer structure). It also does NOT run any
bench re-evaluations itself: those run on the ops queue (the issue pins
"compute via ops queue"). Instead it consumes a per-planner evaluation table
whose rows are produced by replaying each certified config against each planner
at the standard seed protocol and summarising the episode with
:func:`robot_sf.adversarial.robustness.compute_robustness_report` (so the
transfer metric uses the same signed-robustness semantics as the search
objectives).

Gate A repair (#6146): the matrix is now built from immutable
candidate x evaluated-planner x fresh-seed rows. It rejects stress_only,
pre-correction, fallback, degraded, unavailable, duplicate, malformed,
blind-corner, and lineage-incomplete inputs. It reports candidate-clustered
uncertainty with explicit candidate and seed denominators, a scalar robustness
diagnostic only, and no minimax or regret claims. A side-effect-free
:func:`check_issue_6145_activation` helper is provided so that downstream
activation can remain fail-closed on ``promote``, ``>= 5`` admitted candidates,
required hashes, and valid lineage; issue closure alone never activates
anything.

Version boundary
----------------

- ``build_gate_a_transfer_matrix`` emits ``adversarial_transfer_matrix.v2``
  with full candidate x planner x seed rows and candidate-clustered
  uncertainty. It requires complete Gate A provenance.
- ``build_transfer_matrix`` is the legacy v1 entry point. It emits
  ``adversarial_transfer_matrix.v1`` and only builds cells and the historical
  per-planner ranking. It does not emit the required Gate A rows, so it cannot
  be mistaken for a bounded Gate A evidence packet.
- ``PlannerRanking`` and ``minimax_regret`` are retained only for the legacy
  v1 compatibility boundary; Gate A v2 emits capability diagnostics without
  a conventional regret field.

Capability-not-evidence boundary: the matrix is built only from archive paths
and pinned configs/seeds. No benchmark or paper-facing claim is made here; the
report is explicitly labelled capability-only.

Status: research/exploratory. These artifacts are transfer measurements, not
reported benchmark metrics.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import random
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from robot_sf.adversarial.archive import (
    ARCHIVE_SCHEMA_VERSION,
    SEARCH_MANIFEST_SCHEMA_VERSION,
)
from robot_sf.adversarial.provenance import (
    ReceiptItem,
    gather_execution_context,
    sha256_of_file,
    write_execution_context,
    write_receipt_manifest,
)
from robot_sf.adversarial.transfer_schema import (
    CandidateProvenance,
    ConstraintsFirstOutcome,
    GateATransferRow,
    PlannerEvalProvenance,
)

_TRANSFER_MATRIX_SCHEMA_VERSION_V2 = "adversarial_transfer_matrix.v2"
_TRANSFER_MATRIX_SCHEMA_VERSION_V1 = "adversarial_transfer_matrix.v1"
_TRANSFER_MATRIX_SCHEMA_VERSION = _TRANSFER_MATRIX_SCHEMA_VERSION_V2

# Durable archive subpath for the K x N transfer run inside the adversarial
# archive. The issue pins "adversarial archive path — never the release
# evidence store", so results live here, not in the release evidence tree.
_TRANSFER_ARCHIVE_DIRNAME = "transfer_matrix"
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

# Benchmark eligibility tiers that count as "certified / repliable" for slice 1.
_CERTIFIED_ELIGIBILITY = frozenset({"eligible", "stress_only"})

# Gate A transfer-contract eligibility: only fully eligible rows are admitted.
_GATE_A_ELIGIBILITY = frozenset({"eligible"})

# Row classes that Gate A rejects as discoveries (they may remain in a primary
# intention-to-search denominator elsewhere, but never in the transfer matrix).
_GATE_A_EXCLUDED_ROW_CLASSES: tuple[str, ...] = (
    "fallback",
    "degraded",
    "unavailable",
    "geometry_artifact",
    "knife_edge",
    "stress_only",
    "duplicate",
    "pre_correction",
    "malformed",
    "lineage_incomplete",
    "blind_corner",
    "blind-corner",
)
_GATE_A_EXCLUDED_ROW_CLASS_ALIASES = frozenset(
    {
        *_GATE_A_EXCLUDED_ROW_CLASSES,
        *(row_class.replace("_", "-") for row_class in _GATE_A_EXCLUDED_ROW_CLASSES),
    }
)

# Frozen #6145 terminal result schema that Gate A activation checks.
_PROMOTION_RESULT_SCHEMA_VERSION = "issue_5303_search_promotion_result.v2"
_PROMOTION_RESULT_REQUIRED_FIELDS = (
    "schema_version",
    "decision",
    "contract_sha256",
    "execution_commit",
    "admitted_candidate_count",
    "candidate_manifest_sha256",
    "evidence_packet_sha256",
)
_PROMOTION_RESULT_DECISION_VALUES = ("promote", "stop", "inconclusive")
_PROMOTION_MIN_ADMITTED_CANDIDATES = 5
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# Gate A frozen pilot contract: exactly one 3-planner roster and five fresh
# seeds per candidate/planner unless #6147 preregisters a stricter design.
_GATE_A_REQUIRED_PLANNERS = 3
_GATE_A_SEEDS_PER_PLANNER = 5


def _validated_run_id(run_id: str) -> str:
    """Return one safe archive path component or fail closed."""
    if run_id in {".", ".."} or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must be a single 1-128 character path component containing only "
            "letters, digits, '.', '_', or '-'"
        )
    return run_id


_ELIGIBILITY_SEVERITY = {"eligible": 0, "stress_only": 1, "excluded": 2}

# Default 3-planner mechanism-stratified roster for slice 1: the issue asks for
# the target planner plus 2 other planners. The roster mirrors the engineered
# candidates called out in the issue's evidence-grade promotion plan.
DEFAULT_TRANSFER_ROSTER: tuple[str, ...] = (
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "ppo",
)


@dataclass(frozen=True)
class CertifiedConfig:
    """One certified worst-case config selected from a target-planner archive.

    Attributes
    ----------
    config_id : str
        Stable id for the config within the transfer matrix.
    target_planner : str
        Planner the config was optimized / certified against.
    candidate : dict[str, Any]
        The perturbable scenario candidate (start/goal/seed/speed/...).
    objective_value : float
        Worst-case composite objective (signed-robustness based) against the
        target planner. Larger = worse (more negative robustness).
    source_manifest : str
        Origin manifest path (provenance; archive path only).
    source_candidate_index : int
        Candidate index within the source manifest.
    certification_tier : str
        Benchmark eligibility tier from certification (eligible / stress_only).
    scenario_seed : int | None
        Pinned scenario seed for replay reproducibility.
    primary_mechanism : str
        Predeclared primary failure mechanism for mechanism-retention checks.
    row_class : str
        Normalized row class used for Gate A exclusion checks.
    candidate_provenance : CandidateProvenance | None
        Immutable Gate A candidate lineage. Required for
        :func:`build_gate_a_transfer_matrix`.
    """

    config_id: str
    target_planner: str
    candidate: dict[str, Any]
    objective_value: float
    source_manifest: str
    source_candidate_index: int
    certification_tier: str
    scenario_seed: int | None
    primary_mechanism: str = "unspecified"
    row_class: str = "eligible"
    candidate_provenance: CandidateProvenance | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        provenance: dict[str, Any] | None = None
        if self.candidate_provenance is not None:
            provenance = self.candidate_provenance.to_json()
        return {
            "config_id": self.config_id,
            "target_planner": self.target_planner,
            "candidate": self.candidate,
            "objective_value": self.objective_value,
            "source_manifest": self.source_manifest,
            "source_candidate_index": self.source_candidate_index,
            "certification_tier": self.certification_tier,
            "scenario_seed": self.scenario_seed,
            "primary_mechanism": self.primary_mechanism,
            "row_class": self.row_class,
            "candidate_provenance": provenance,
        }


@dataclass(frozen=True)
class PlannerEval:
    """One per-planner re-evaluation result for a certified config.

    Attributes
    ----------
    config_id : str
        Config this result belongs to (matches :class:`CertifiedConfig`).
    planner : str
        Evaluated planner.
    robustness : float
        Overall signed robustness against this planner (negative = violated).
    failed : bool
        Whether the planner reproduced a failure (robustness < 0).
    seed : int | None
        Pinned evaluation seed (standard seed protocol).
    mechanism : str
        Observed primary failure mechanism for mechanism-retention checks.
    eval_seed : int | None
        Alias for ``seed`` kept for explicit readability in Gate A rows.
    constraints_first_outcome : ConstraintsFirstOutcome | None
        Ordered safety/liveness/comfort outcome vector. Required for Gate A.
    planner_provenance : PlannerEvalProvenance | None
        Immutable evaluated-planner lineage. Required for Gate A.
    """

    config_id: str
    planner: str
    robustness: float
    failed: bool
    seed: int | None
    mechanism: str = "unspecified"
    eval_seed: int | None = None
    constraints_first_outcome: ConstraintsFirstOutcome | None = None
    planner_provenance: PlannerEvalProvenance | None = None
    attribution_review_status: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        outcome: dict[str, Any] | None = None
        if self.constraints_first_outcome is not None:
            outcome = self.constraints_first_outcome.to_json()
        provenance: dict[str, Any] | None = None
        if self.planner_provenance is not None:
            provenance = self.planner_provenance.to_json()
        return {
            "config_id": self.config_id,
            "planner": self.planner,
            "robustness": self.robustness,
            "failed": self.failed,
            "seed": self.seed,
            "mechanism": self.mechanism,
            "eval_seed": self.eval_seed,
            "constraints_first_outcome": outcome,
            "planner_provenance": provenance,
            "attribution_review_status": self.attribution_review_status,
        }


@dataclass(frozen=True)
class TransferCell:
    """One cell of the K x N transfer matrix (legacy v1 aggregation unit)."""

    config_id: str
    planner: str
    robustness: float
    failed: bool
    transferred: bool


# Gate A v2 row is the authoritative transfer-contract row.
TransferRow = GateATransferRow


@dataclass(frozen=True)
class CandidateCluster:
    """Candidate-clustered uncertainty summary for one certified config.

    The cluster aggregates every fresh-seed evaluation of one candidate across
    all evaluated planners. It reports explicit candidate and seed
    denominators, a scalar robustness diagnostic, and a fail-closed
    mechanism-retention flag.
    """

    config_id: str
    target_planner: str
    scenario_seed: int
    n_evaluated_seeds: int
    n_failed: int
    n_transferred: int
    n_non_target_seeds: int
    n_non_target_transferred: int
    primary_mechanism: str
    mechanism_retained: bool
    robustness_diagnostic: float


@dataclass(frozen=True)
class CapabilityRanking:
    """Capability-only ranking for one planner (no minimax/regret claim)."""

    planner: str
    worst_case_robustness: float
    transfer_failure_rate: float
    rank: int


@dataclass(frozen=True)
class PlannerRanking:
    """Legacy v1 ranking shape retained for compatibility only."""

    planner: str
    worst_case_robustness: float
    transfer_failure_rate: float
    minimax_regret: float
    rank: int


def minimax_regret(worst_case_robustness: float) -> float:
    """Return the historical v1 compatibility value for one diagnostic.

    This helper exists only for legacy v1 consumers. Gate A reports
    ``worst_case_robustness`` as a descriptive diagnostic and makes no
    conventional regret claim.
    """
    return -worst_case_robustness if math.isfinite(worst_case_robustness) else float("nan")


@dataclass(frozen=True)
class TransferMatrix:
    """The full K x N transfer measurement plus summary statistics."""

    schema_version: str = _TRANSFER_MATRIX_SCHEMA_VERSION
    target_planner: str = ""
    configs: tuple[CertifiedConfig, ...] = ()
    config_ids: tuple[str, ...] = ()
    planners: tuple[str, ...] = ()
    cells: tuple[TransferCell, ...] = ()
    rows: tuple[TransferRow, ...] = ()
    clusters: tuple[CandidateCluster, ...] = ()
    ranking: tuple[CapabilityRanking | PlannerRanking, ...] = ()
    overall_transfer_rate: float = 0.0
    transfer_rate_ci: tuple[float, float] = (0.0, 0.0)
    transfer_rate_bootstrap_n: int = 0
    n_candidates: int = 0
    n_seed_evals: int = 0
    capability_only: bool = True

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "schema_version": self.schema_version,
            "target_planner": self.target_planner,
            "configs": [config.to_json() for config in self.configs],
            "config_ids": list(self.config_ids),
            "planners": list(self.planners),
            "cells": [c.__dict__ for c in self.cells],
            "rows": [r.to_json() for r in self.rows],
            "clusters": [c.__dict__ for c in self.clusters],
            "ranking": [r.__dict__ for r in self.ranking],
            "overall_transfer_rate": self.overall_transfer_rate,
            "transfer_rate_ci": list(self.transfer_rate_ci),
            "transfer_rate_bootstrap_n": self.transfer_rate_bootstrap_n,
            "n_candidates": self.n_candidates,
            "n_seed_evals": self.n_seed_evals,
            "capability_only": self.capability_only,
        }


def _candidate_certification_tier(candidate: dict[str, Any]) -> str | None:
    """Extract the benchmark eligibility tier from a candidate payload."""
    cert = candidate.get("certification_status")
    if not isinstance(cert, dict):
        return None
    certificates = (
        cert.get("details", {}).get("certificates")
        if isinstance(cert.get("details"), dict)
        else None
    )
    if isinstance(certificates, list) and certificates:
        tiers: list[str] = []
        for certificate in certificates:
            if not isinstance(certificate, dict):
                return None
            tier = str(certificate.get("benchmark_eligibility", "")).strip().lower()
            if tier not in _ELIGIBILITY_SEVERITY:
                return None
            tiers.append(tier)
        return max(tiers, key=_ELIGIBILITY_SEVERITY.__getitem__)
    # Fall back only when a top-level status already uses the eligibility vocabulary.
    status = str(cert.get("status", "")).strip().lower()
    return status if status in _ELIGIBILITY_SEVERITY else None


def _candidate_scenario_seed(candidate: dict[str, Any]) -> int | None:
    """Extract the pinned scenario seed from a candidate payload."""
    seed = candidate.get("scenario_seed")
    if seed is None:
        return None
    try:
        parsed = float(seed)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not parsed.is_integer() or parsed < 0:
        return None
    return int(parsed)


def _candidate_primary_mechanism(candidate_payload: dict[str, Any]) -> str:
    """Extract the predeclared primary mechanism from a candidate payload.

    Falls back to the classification label or ``unspecified`` when no
    mechanism is declared. Gate A mechanism retention compares this value to
    the observed mechanism reported by each fresh-seed evaluation.
    """
    if not isinstance(candidate_payload, dict):
        return "unspecified"
    candidate = (
        candidate_payload.get("candidate")
        if isinstance(candidate_payload.get("candidate"), dict)
        else {}
    )
    mechanism = candidate.get("primary_mechanism")
    if isinstance(mechanism, str) and mechanism.strip():
        return mechanism.strip()
    cert = candidate_payload.get("certification_status")
    details = cert.get("details") if isinstance(cert, dict) else {}
    certificates = details.get("certificates") if isinstance(details, dict) else []
    if isinstance(certificates, list) and certificates:
        first = certificates[0]
        if isinstance(first, dict):
            classification = first.get("classification")
            if isinstance(classification, str) and classification.strip():
                return classification.strip()
    return "unspecified"


def _candidate_row_class(candidate_payload: dict[str, Any]) -> str:
    """Return the normalized row class from the first certificate classification."""
    if not isinstance(candidate_payload, dict):
        return "malformed"
    cert = candidate_payload.get("certification_status")
    details = cert.get("details") if isinstance(cert, dict) else {}
    certificates = details.get("certificates") if isinstance(details, dict) else []
    if isinstance(certificates, list) and certificates:
        first = certificates[0]
        if isinstance(first, dict):
            classification = str(first.get("classification", "")).strip().lower()
            if classification:
                return classification
    return "eligible"


def _extract_candidate_provenance(
    candidate_payload: dict[str, Any],
    *,
    config_id: str,
    target_planner: str,
) -> CandidateProvenance | None:
    """Build Gate A candidate provenance from an explicit payload block.

    Returns ``None`` when no provenance block is supplied. Gate A builders fail
    closed on missing provenance; legacy builders tolerate its absence.
    """
    if not isinstance(candidate_payload, dict):
        return None
    provenance = candidate_payload.get("candidate_provenance")
    if not isinstance(provenance, dict):
        return None
    return CandidateProvenance(
        source_target_planner=str(provenance.get("source_target_planner", "")),
        source_campaign_identity=str(provenance.get("source_campaign_identity", "")),
        source_candidate_identity=str(provenance.get("source_candidate_identity", "")),
        normalized_candidate_hash=str(provenance.get("normalized_candidate_hash", "")),
        certification_hash=str(provenance.get("certification_hash", "")),
        recertification_hash=(
            str(provenance.get("recertification_hash"))
            if provenance.get("recertification_hash") is not None
            else None
        ),
        scenario_family_hash=str(provenance.get("scenario_family_hash", "")),
        scenario_config_hash=str(provenance.get("scenario_config_hash", "")),
        execution_commit=str(provenance.get("execution_commit", "")),
        execution_context_path=str(provenance.get("execution_context_path", "")),
        record_hash=str(provenance.get("record_hash", "")),
        admission_status=str(provenance.get("admission_status", "")),
        admission_reason=str(provenance.get("admission_reason", "")),
    )


def _is_excluded_row_class(candidate_payload: dict[str, Any]) -> str | None:
    """Return the excluded row class reason, or None when the row is admissible.

    Gate A rejects any candidate whose certification status or classification
    matches one of the frozen excluded row classes. The returned string is the
    matched class and is meant for error messages.
    """
    if not isinstance(candidate_payload, dict):
        return "malformed"
    cert = candidate_payload.get("certification_status")
    details = cert.get("details") if isinstance(cert, dict) else {}
    certificates = details.get("certificates") if isinstance(details, dict) else []
    classes: set[str] = set()
    if isinstance(certificates, list):
        for certificate in certificates:
            if isinstance(certificate, dict):
                classification = str(certificate.get("classification", "")).strip().lower()
                if classification:
                    classes.add(classification)
                    classes.add(classification.replace("-", "_"))
                eligibility = str(certificate.get("benchmark_eligibility", "")).strip().lower()
                if eligibility:
                    classes.add(eligibility)
    # Reject the frozen excluded classes and any non-eligible tier.
    for excluded in _GATE_A_EXCLUDED_ROW_CLASS_ALIASES:
        if excluded in classes:
            return excluded
    tier = _candidate_certification_tier(candidate_payload)
    if tier is None or tier not in _GATE_A_ELIGIBILITY:
        return tier or "lineage_incomplete"
    return None


def _is_certified(candidate: dict[str, Any]) -> bool:
    """Return whether a candidate is certified / repliable for slice 1."""
    tier = _candidate_certification_tier(candidate)
    return tier in _CERTIFIED_ELIGIBILITY


def _objective_value(candidate: dict[str, Any]) -> float | None:
    """Return the worst-case objective value for a candidate."""
    value = candidate.get("objective_value")
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _load_target_manifest(
    manifest_path: Path, *, target_planner: str
) -> tuple[dict[str, Any], list[Any]]:
    """Load one search manifest and verify its schema and target-planner lineage."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Search manifest must be a JSON object: {manifest_path}")
    schema = payload.get("schema_version")
    if schema != SEARCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported search manifest schema for {manifest_path}: {schema!r}; "
            f"expected {SEARCH_MANIFEST_SCHEMA_VERSION!r}"
        )
    manifest_config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    manifest_policy = str(manifest_config.get("policy", "")).strip()
    if manifest_policy != target_planner:
        raise ValueError(
            f"Target planner mismatch for {manifest_path}: manifest policy "
            f"{manifest_policy!r} != {target_planner!r}"
        )
    candidates = payload.get("candidates") or []
    return manifest_config, candidates if isinstance(candidates, list) else []


def _certified_config_from_payload(
    candidate_payload: Any,
    *,
    manifest_path: Path,
    target_planner: str,
    index: int,
) -> CertifiedConfig | None:
    """Build one fail-closed selected config or return None when it is not repliable."""
    if not isinstance(candidate_payload, dict) or not _is_certified(candidate_payload):
        return None
    candidate = (
        candidate_payload.get("candidate")
        if isinstance(candidate_payload.get("candidate"), dict)
        else {}
    )
    objective_value = _objective_value(candidate_payload)
    scenario_seed = _candidate_scenario_seed(candidate)
    if not candidate or objective_value is None or scenario_seed is None:
        return None
    config_id = f"{manifest_path.as_posix()}#{index}"
    return CertifiedConfig(
        config_id=config_id,
        target_planner=target_planner,
        candidate=candidate,
        objective_value=objective_value,
        source_manifest=manifest_path.as_posix(),
        source_candidate_index=index,
        certification_tier=_candidate_certification_tier(candidate_payload) or "unknown",
        scenario_seed=scenario_seed,
        primary_mechanism=_candidate_primary_mechanism(candidate_payload),
        row_class=_candidate_row_class(candidate_payload),
        candidate_provenance=_extract_candidate_provenance(
            candidate_payload,
            config_id=config_id,
            target_planner=target_planner,
        ),
    )


def select_certified_configs(
    manifest_paths: list[str | Path],
    *,
    target_planner: str,
    K: int,
    scenario_template: str | None = None,
    eligible_only: bool = False,
) -> list[CertifiedConfig]:
    """Select the top-K certified worst-case configs against ONE planner.

    Reads adversarial search manifests (real ``adversarial-search-manifest.v1``
    schema), keeps only certified, repliable candidates, optionally filters by
    scenario template, and returns the K configs with the worst (largest)
    objective value, i.e. the strongest discovered weak points.

    Parameters
    ----------
    manifest_paths : list[str | Path]
        Search manifests to read configs from.
    target_planner : str
        Planner the configs were optimized / certified against.
    K : int
        Maximum number of configs to return (>= 5 required by the issue for the
        transfer measurement).
    scenario_template : str | None
        Optional scenario-template filter (exact match on manifest config).
    eligible_only : bool
        When True (Gate A), reject ``stress_only`` and any excluded row class.
        The legacy default False keeps the original ``eligible`` /
        ``stress_only`` slice-1 behavior for backward compatibility.

    Returns
    -------
    list[CertifiedConfig]
        Up to K certified worst-case configs, sorted worst-first.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if not target_planner.strip():
        raise ValueError("target_planner must be non-empty")

    eligibility = _GATE_A_ELIGIBILITY if eligible_only else _CERTIFIED_ELIGIBILITY
    configs: list[CertifiedConfig] = []
    for manifest_path in sorted(Path(p) for p in manifest_paths):
        manifest_config, candidates = _load_target_manifest(
            manifest_path, target_planner=target_planner
        )
        if (
            scenario_template is not None
            and manifest_config.get("scenario_template") != scenario_template
        ):
            continue
        for index, candidate_payload in enumerate(candidates):
            if eligible_only:
                excluded = _is_excluded_row_class(candidate_payload)
                if excluded is not None:
                    continue
            elif not isinstance(candidate_payload, dict) or not _is_certified(candidate_payload):
                continue
            config = _certified_config_from_payload(
                candidate_payload,
                manifest_path=manifest_path,
                target_planner=target_planner,
                index=index,
            )
            if config is not None and config.certification_tier in eligibility:
                configs.append(config)

    configs.sort(key=lambda c: (-c.objective_value, c.config_id))
    return configs[:K]


def _sha256_json(payload: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of a deterministic JSON encoding."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bootstrap_transfer_rate(
    failures: list[int],
    evaluations: list[int],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a normal-approximation CI for the transfer (failure) rate.

    The transfer rate is the fraction of (config, planner) pairs in which a
    config that failed against the target planner *also* failed against the
    evaluated planner. This measures whether discovered weak points are
    structural (transfer) or policy-specific.

    Returns
        (point_estimate, ci_low, ci_high).
    """
    total_fails = sum(failures)
    total_evals = sum(evaluations)
    if total_evals == 0:
        return 0.0, 0.0, 0.0
    point = total_fails / total_evals
    if n_resamples <= 0:
        return point, point, point
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        acc = 0
        for f, n in zip(failures, evaluations, strict=True):
            if n == 0:
                continue
            acc += sum(1 for _ in range(n) if rng.random() < (f / n))
        means.append(acc / total_evals)
    means.sort()
    low = means[max(0, int(0.025 * len(means)))]
    high = means[min(len(means) - 1, int(0.975 * len(means)))]
    return point, low, high


def _candidate_clustered_transfer_rate_ci(
    rows: list[TransferRow],
    target_planner: str,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, tuple[float, float], int, int, int]:
    """Candidate-clustered bootstrap CI for the transfer rate.

    Seeds are nested within candidate x planner. The bootstrap resamples
    candidates (clusters) with replacement and computes the mean cluster
    transfer rate. Both candidate-level and seed-level denominators are
    returned so reports can label small-K intervals as exploratory.
    """
    by_candidate: dict[str, list[TransferRow]] = {}
    for row in rows:
        if row.evaluated_planner == target_planner:
            continue
        by_candidate.setdefault(row.config_id, []).append(row)

    n_candidates = len(by_candidate)
    if n_candidates == 0:
        return 0.0, (0.0, 0.0), 0, 0, 0

    cluster_rates: list[float] = []
    n_seed_evals = 0
    n_transferred = 0
    for candidate_rows in by_candidate.values():
        failed = sum(1 for r in candidate_rows if r.transferred)
        total = len(candidate_rows)
        n_seed_evals += total
        n_transferred += failed
        cluster_rates.append(failed / total if total else 0.0)

    point = n_transferred / n_seed_evals if n_seed_evals else 0.0
    if n_resamples <= 0:
        return point, (point, point), n_candidates, n_seed_evals, n_transferred

    rng = random.Random(seed)
    resampled_means: list[float] = []
    for _ in range(n_resamples):
        sample_total = 0.0
        for _ in range(n_candidates):
            idx = rng.randrange(n_candidates)
            sample_total += cluster_rates[idx]
        resampled_means.append(sample_total / n_candidates)
    resampled_means.sort()
    low = resampled_means[max(0, int(0.025 * len(resampled_means)))]
    high = resampled_means[min(len(resampled_means) - 1, int(0.975 * len(resampled_means)))]
    return point, (low, high), n_candidates, n_seed_evals, n_transferred


def _build_cells(
    config_ids: tuple[str, ...],
    planners: tuple[str, ...],
    eval_by_key: dict[tuple[str, str], PlannerEval],
) -> list[TransferCell]:
    """Assemble K x N transfer cells from certified configs and eval results."""
    cells: list[TransferCell] = []
    for cfg_id in config_ids:
        for planner in planners:
            ev = eval_by_key.get((cfg_id, planner))
            if ev is None:
                raise ValueError(f"Missing evaluation for config={cfg_id!r}, planner={planner!r}")
            # A weak point transfers when the target config's failure also
            # reproduces against the evaluated planner.
            cells.append(
                TransferCell(
                    config_id=cfg_id,
                    planner=planner,
                    robustness=ev.robustness,
                    failed=ev.failed,
                    transferred=ev.failed,
                )
            )
    return cells


def _build_cells_from_rows(
    rows: list[TransferRow],
    config_ids: tuple[str, ...],
    planners: tuple[str, ...],
) -> list[TransferCell]:
    """Derive K x N legacy cells from Gate A rows (worst robustness per cell)."""
    by_key: dict[tuple[str, str], TransferRow] = {}
    for row in rows:
        key = (row.config_id, row.evaluated_planner)
        if key not in by_key or (
            math.isfinite(row.robustness_diagnostic)
            and row.robustness_diagnostic < by_key[key].robustness_diagnostic
        ):
            by_key[key] = row

    cells: list[TransferCell] = []
    for cfg_id in config_ids:
        for planner in planners:
            row = by_key.get((cfg_id, planner))
            if row is None:
                raise ValueError(f"Missing row for config={cfg_id!r}, planner={planner!r}")
            cells.append(
                TransferCell(
                    config_id=cfg_id,
                    planner=planner,
                    robustness=row.robustness_diagnostic,
                    failed=row.outcome.failed(),
                    transferred=row.transferred,
                )
            )
    return cells


def _build_ranking(
    cells: list[TransferCell],
    planners: tuple[str, ...],
) -> list[CapabilityRanking]:
    """Compute per-planner capability-only ranking rows (no minimax/regret)."""
    per_planner = _group_cells_by_planner(cells, planners)

    ranking_rows: list[CapabilityRanking] = []
    for planner in planners:
        planner_cells = per_planner[planner]
        finite = [c.robustness for c in planner_cells if math.isfinite(c.robustness)]
        worst = min(finite) if finite else float("nan")
        failures = [c for c in planner_cells if c.failed]
        transfer_rate = len(failures) / len(planner_cells) if planner_cells else 0.0
        ranking_rows.append(
            CapabilityRanking(
                planner=planner,
                worst_case_robustness=worst,
                transfer_failure_rate=transfer_rate,
                rank=0,
            )
        )
    return ranking_rows


def _group_cells_by_planner(
    cells: list[TransferCell],
    planners: tuple[str, ...],
) -> dict[str, list[TransferCell]]:
    """Group transfer cells by planner column, in planner order."""
    per_planner: dict[str, list[TransferCell]] = {p: [] for p in planners}
    for cell in cells:
        if cell.planner in per_planner:
            per_planner[cell.planner].append(cell)
    return per_planner


def _validate_matrix_configs(
    configs: list[CertifiedConfig], *, bootstrap_n: int
) -> tuple[tuple[str, ...], str]:
    """Validate selected config provenance and return ids plus the shared target planner."""
    if not configs:
        raise ValueError("Cannot build a transfer matrix from zero certified configs")
    if len(configs) < 5:
        raise ValueError(
            f"Issue #5303 slice 1 requires K >= 5 certified configs; got {len(configs)}"
        )
    config_ids = tuple(config.config_id for config in configs)
    target_planner = configs[0].target_planner
    if not target_planner.strip():
        raise ValueError("Certified configs must name a target planner")
    if len(set(config_ids)) != len(config_ids):
        raise ValueError("Certified config ids must be unique")
    if any(config.target_planner != target_planner for config in configs):
        raise ValueError("All certified configs must share one target planner")
    if any(config.certification_tier not in _CERTIFIED_ELIGIBILITY for config in configs):
        raise ValueError("All configs must have eligible or stress_only certification")
    if any(not math.isfinite(config.objective_value) for config in configs):
        raise ValueError("All certified configs must have a finite objective value")
    if any(config.scenario_seed is None for config in configs):
        raise ValueError("All certified configs must pin scenario_seed")
    if bootstrap_n < 0:
        raise ValueError("bootstrap_n must be >= 0")
    return config_ids, target_planner


def _resolve_matrix_planners(
    evaluations: list[PlannerEval],
    *,
    target_planner: str,
    planners: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Resolve and validate the target-plus-two planner roster."""
    if planners is None:
        planners = tuple(dict.fromkeys(evaluation.planner for evaluation in evaluations))
    if len(planners) < 3:
        raise ValueError("Issue #5303 slice 1 requires the target planner plus 2 others")
    if len(set(planners)) != len(planners):
        raise ValueError("Planner names must be unique")
    if target_planner not in planners:
        raise ValueError("The target planner must be present in the matrix roster")
    return planners


def _validate_evaluation(evaluation: PlannerEval) -> None:
    """Validate one signed-robustness evaluation before aggregation."""
    if not math.isfinite(evaluation.robustness):
        raise ValueError(
            f"Evaluation robustness must be finite for config={evaluation.config_id!r}, "
            f"planner={evaluation.planner!r}"
        )
    seed = evaluation.eval_seed if evaluation.eval_seed is not None else evaluation.seed
    if seed is None or seed < 0:
        raise ValueError(
            f"Evaluation seed must be pinned and non-negative for config={evaluation.config_id!r}, "
            f"planner={evaluation.planner!r}"
        )
    if evaluation.failed != (evaluation.robustness < 0.0):
        raise ValueError(
            "Evaluation failed flag disagrees with signed robustness for "
            f"config={evaluation.config_id!r}, planner={evaluation.planner!r}"
        )


def _index_complete_evaluations(
    evaluations: list[PlannerEval],
    *,
    config_ids: tuple[str, ...],
    planners: tuple[str, ...],
) -> dict[tuple[str, str], PlannerEval]:
    """Return a complete unique evaluation index or fail closed."""
    valid_config_ids = set(config_ids)
    valid_planners = set(planners)
    indexed: dict[tuple[str, str], PlannerEval] = {}
    for evaluation in evaluations:
        key = (evaluation.config_id, evaluation.planner)
        if evaluation.config_id not in valid_config_ids:
            raise ValueError(f"Evaluation references unknown config: {evaluation.config_id!r}")
        if evaluation.planner not in valid_planners:
            raise ValueError(f"Evaluation references unknown planner: {evaluation.planner!r}")
        if key in indexed:
            raise ValueError(
                f"Duplicate evaluation for config={evaluation.config_id!r}, "
                f"planner={evaluation.planner!r}"
            )
        _validate_evaluation(evaluation)
        indexed[key] = evaluation
    expected = {(config_id, planner) for config_id in config_ids for planner in planners}
    missing = sorted(expected - set(indexed))
    if missing:
        config_id, planner = missing[0]
        raise ValueError(
            f"Transfer matrix is incomplete; missing {len(missing)} evaluation(s), "
            f"including config={config_id!r}, planner={planner!r}"
        )
    return indexed


def _validate_gate_a_candidate_identity(
    provenance: CandidateProvenance,
    *,
    expected_target_planner: str,
) -> list[str]:
    """Validate candidate identity and non-hash scalar provenance fields."""
    errors: list[str] = []
    if not provenance.source_target_planner:
        errors.append("candidate provenance missing source_target_planner")
    elif provenance.source_target_planner != expected_target_planner:
        errors.append(
            "candidate provenance source_target_planner does not match the matrix target planner"
        )

    for field, label in (
        ("source_campaign_identity", "source_campaign_identity"),
        ("source_candidate_identity", "source_candidate_identity"),
        ("execution_context_path", "execution_context_path"),
        ("admission_reason", "admission_reason"),
    ):
        if not getattr(provenance, field):
            errors.append(f"candidate provenance missing {label}")

    if not _FULL_COMMIT_RE.fullmatch(provenance.execution_commit):
        errors.append("candidate provenance execution_commit must be a 40-hex git SHA")
    return errors


def _validate_gate_a_candidate_hashes(provenance: CandidateProvenance) -> list[str]:
    """Validate candidate provenance hashes."""
    errors: list[str] = []
    for field, label in (
        ("normalized_candidate_hash", "normalized_candidate_hash"),
        ("certification_hash", "certification_hash"),
        ("scenario_family_hash", "scenario_family_hash"),
        ("scenario_config_hash", "scenario_config_hash"),
        ("record_hash", "record_hash"),
    ):
        if not _SHA256_RE.fullmatch(getattr(provenance, field)):
            errors.append(f"candidate provenance {label} must be a 64-hex SHA-256")

    if not _SHA256_RE.fullmatch(provenance.recertification_hash or ""):
        errors.append("candidate provenance recertification_hash must be a 64-hex SHA-256")
    return errors


def _validate_gate_a_candidate_provenance(
    provenance: CandidateProvenance,
    *,
    expected_target_planner: str,
) -> list[str]:
    """Validate one candidate provenance block; return a list of error strings."""
    errors = _validate_gate_a_candidate_identity(
        provenance, expected_target_planner=expected_target_planner
    )
    errors.extend(_validate_gate_a_candidate_hashes(provenance))
    if provenance.admission_status not in {"admitted", "excluded"}:
        errors.append("candidate provenance admission_status must be 'admitted' or 'excluded'")
    if provenance.admission_status != "admitted":
        errors.append(f"candidate provenance admission_status is {provenance.admission_status!r}")
    return errors


def _validate_gate_a_planner_provenance(
    provenance: PlannerEvalProvenance,
    *,
    expected_planner: str,
) -> list[str]:
    """Validate one evaluated-planner lineage block; return a list of error strings."""
    errors: list[str] = []
    if provenance.evaluated_planner != expected_planner:
        errors.append("planner provenance evaluated_planner does not match the row planner")

    for field, label in (
        ("planner_config_hash", "planner_config_hash"),
        ("scenario_config_hash", "scenario_config_hash"),
        ("record_hash", "record_hash"),
    ):
        if not _SHA256_RE.fullmatch(getattr(provenance, field)):
            errors.append(f"planner provenance {label} must be a 64-hex SHA-256")

    if provenance.execution_mode not in {"native", "fallback", "degraded", "unavailable"}:
        errors.append(
            "planner provenance execution_mode must be one of native/fallback/degraded/unavailable"
        )
    if provenance.execution_mode in {"fallback", "degraded", "unavailable"}:
        errors.append(
            f"Gate A rejects execution_mode {provenance.execution_mode!r} "
            f"for planner {expected_planner!r}"
        )

    for field, label in (
        ("deterministic_replay_lineage", "deterministic_replay_lineage"),
        ("independent_confirmation_lineage", "independent_confirmation_lineage"),
        ("execution_context_path", "execution_context_path"),
    ):
        if not getattr(provenance, field):
            errors.append(f"planner provenance missing {label}")

    if not _FULL_COMMIT_RE.fullmatch(provenance.execution_commit):
        errors.append("planner provenance execution_commit must be a 40-hex git SHA")
    return errors


def _validate_gate_a_outcome(outcome: ConstraintsFirstOutcome) -> list[str]:
    """Validate one constraints-first outcome vector; return a list of error strings."""
    errors: list[str] = []
    if outcome.status != "observed":
        errors.append(f"Gate A rejects unavailable outcome status {outcome.status!r}")
        return errors
    # observed
    if not isinstance(outcome.collision_or_severe_intrusion, bool):
        errors.append("observed outcome collision_or_severe_intrusion must be a boolean")
    if not isinstance(outcome.liveness_or_goal_completion, bool):
        errors.append("observed outcome liveness_or_goal_completion must be a boolean")
    if not isinstance(outcome.comfort_and_efficiency, dict):
        errors.append("observed outcome comfort_and_efficiency must be a mapping")
    else:
        for field in ("snqi", "near_misses", "path_efficiency"):
            if field not in outcome.comfort_and_efficiency:
                errors.append(f"observed outcome comfort_and_efficiency missing {field!r}")
    return errors


def _resolve_gate_a_eval_context(
    evaluation: PlannerEval,
    *,
    config_by_id: dict[str, CertifiedConfig],
    planners: tuple[str, ...],
    seen: set[tuple[str, str, int]],
    expected_seeds: dict[tuple[str, str], set[int]],
) -> tuple[CertifiedConfig, int]:
    """Validate one evaluation's config/planner/seed context and return (config, eval_seed)."""
    config = config_by_id.get(evaluation.config_id)
    if config is None:
        raise ValueError(f"Gate A row references unknown config: {evaluation.config_id!r}")
    if evaluation.planner not in planners:
        raise ValueError(f"Gate A row references unknown planner: {evaluation.planner!r}")
    if (
        evaluation.eval_seed is not None
        and evaluation.seed is not None
        and evaluation.eval_seed != evaluation.seed
    ):
        raise ValueError(
            f"Gate A row has mismatched seed and eval_seed for config={evaluation.config_id!r}, "
            f"planner={evaluation.planner!r}"
        )
    eval_seed = evaluation.eval_seed if evaluation.eval_seed is not None else evaluation.seed
    if (
        eval_seed is None
        or isinstance(eval_seed, bool)
        or not isinstance(eval_seed, int)
        or eval_seed < 0
    ):
        raise ValueError(
            f"Gate A row missing fresh eval seed for config={evaluation.config_id!r}, "
            f"planner={evaluation.planner!r}"
        )
    if config.scenario_seed is None:
        raise ValueError(f"Gate A row missing scenario seed for config={evaluation.config_id!r}")
    if eval_seed == config.scenario_seed:
        raise ValueError(
            f"Gate A fresh eval seed must differ from scenario seed for config={evaluation.config_id!r}, "
            f"planner={evaluation.planner!r}"
        )

    key = (evaluation.config_id, evaluation.planner, eval_seed)
    if key in seen:
        raise ValueError(
            f"Gate A duplicate row for config={evaluation.config_id!r}, "
            f"planner={evaluation.planner!r}, eval_seed={eval_seed}"
        )
    seen.add(key)
    expected_seeds[(evaluation.config_id, evaluation.planner)].add(eval_seed)
    return config, eval_seed


def _validate_gate_a_row_lineage(
    config: CertifiedConfig,
    evaluation: PlannerEval,
) -> list[str]:
    """Return a list of lineage errors for one evaluation, or empty if valid."""
    if config.candidate_provenance is None:
        return [f"Gate A row missing candidate_provenance for config={config.config_id!r}"]
    if evaluation.constraints_first_outcome is None:
        return [
            f"Gate A row missing constraints_first_outcome for config={config.config_id!r}, "
            f"planner={evaluation.planner!r}"
        ]
    if evaluation.planner_provenance is None:
        return [
            f"Gate A row missing planner_provenance for config={config.config_id!r}, "
            f"planner={evaluation.planner!r}"
        ]

    errors = _validate_gate_a_candidate_provenance(
        config.candidate_provenance,
        expected_target_planner=config.target_planner,
    )
    errors.extend(
        _validate_gate_a_planner_provenance(
            evaluation.planner_provenance,
            expected_planner=evaluation.planner,
        )
    )
    errors.extend(_validate_gate_a_outcome(evaluation.constraints_first_outcome))
    if not math.isfinite(evaluation.robustness):
        errors.append("Gate A robustness_diagnostic must be finite")
    if (
        not isinstance(evaluation.attribution_review_status, str)
        or not evaluation.attribution_review_status.strip()
    ):
        errors.append("Gate A row missing attribution_review_status")
    if (
        config.candidate_provenance.scenario_config_hash
        != evaluation.planner_provenance.scenario_config_hash
    ):
        errors.append("candidate and planner scenario_config_hash values do not match")
    return errors


def _make_gate_a_transfer_row(
    config: CertifiedConfig,
    evaluation: PlannerEval,
    eval_seed: int,
) -> TransferRow:
    """Build one Gate A transfer row with a deterministic immutable record hash."""
    mechanism = evaluation.mechanism or "unspecified"
    primary = config.primary_mechanism or "unspecified"
    mechanism_retained = primary in {"unspecified", mechanism}
    # The authoritative failure is the ordered constraints-first outcome;
    # scalar robustness is descriptive only.
    failed = evaluation.constraints_first_outcome.failed()
    transferred = failed and evaluation.planner != config.target_planner and mechanism_retained

    row = TransferRow(
        config_id=evaluation.config_id,
        target_planner=config.target_planner,
        evaluated_planner=evaluation.planner,
        scenario_seed=config.scenario_seed,
        eval_seed=eval_seed,
        candidate_provenance=config.candidate_provenance,
        planner_provenance=evaluation.planner_provenance,
        outcome=evaluation.constraints_first_outcome,
        robustness_diagnostic=evaluation.robustness,
        transferred=transferred,
        mechanism_retained=mechanism_retained,
        primary_mechanism=primary,
        observed_mechanism=mechanism,
        attribution_review_status=evaluation.attribution_review_status.strip(),
        lineage_complete=True,
        immutable_record_hash="",
    )
    row_json = {k: v for k, v in row.to_json().items() if k != "immutable_record_hash"}
    return replace(row, immutable_record_hash=_sha256_json(row_json))


def _build_gate_a_rows(
    configs: list[CertifiedConfig],
    evaluations: list[PlannerEval],
    planners: tuple[str, ...],
) -> list[TransferRow]:
    """Build immutable candidate x planner x fresh-seed Gate A rows.

    Each config must contribute exactly one row per evaluated planner per
    fresh seed. Missing, repeated, extra, or mismatched seeds fail closed.
    """
    config_by_id = {config.config_id: config for config in configs}
    expected_seeds: dict[tuple[str, str], set[int]] = {
        (config.config_id, planner): set() for config in configs for planner in planners
    }
    rows: list[TransferRow] = []
    seen: set[tuple[str, str, int]] = set()

    for evaluation in evaluations:
        config, eval_seed = _resolve_gate_a_eval_context(
            evaluation,
            config_by_id=config_by_id,
            planners=planners,
            seen=seen,
            expected_seeds=expected_seeds,
        )
        errors = _validate_gate_a_row_lineage(config, evaluation)
        if errors:
            raise ValueError(
                f"Gate A lineage validation failed for config={evaluation.config_id!r}, "
                f"planner={evaluation.planner!r}, eval_seed={eval_seed}: " + "; ".join(errors)
            )
        rows.append(_make_gate_a_transfer_row(config, evaluation, eval_seed))

    for (config_id, planner), seeds in expected_seeds.items():
        if len(seeds) != _GATE_A_SEEDS_PER_PLANNER:
            raise ValueError(
                f"Gate A matrix expected {_GATE_A_SEEDS_PER_PLANNER} distinct seeds "
                f"for config={config_id!r}, planner={planner!r}; got {sorted(seeds)}"
            )
    planner_config_hashes: dict[str, str] = {}
    for row in rows:
        planner = row.evaluated_planner
        config_hash = row.planner_provenance.planner_config_hash
        prior = planner_config_hashes.setdefault(planner, config_hash)
        if prior != config_hash:
            raise ValueError(
                f"Gate A planner roster has conflicting planner_config_hash values for {planner!r}"
            )
    return rows


def _build_candidate_clusters(rows: list[TransferRow]) -> list[CandidateCluster]:
    """Aggregate Gate A rows into candidate clusters with explicit denominators."""
    by_config: dict[str, list[TransferRow]] = {}
    for row in rows:
        by_config.setdefault(row.config_id, []).append(row)

    clusters: list[CandidateCluster] = []
    for config_id in sorted(by_config):
        config_rows = by_config[config_id]
        first = config_rows[0]
        mechanisms = [row.observed_mechanism for row in config_rows]
        primary = first.primary_mechanism
        if primary == "unspecified" and mechanisms:
            primary = max(set(mechanisms), key=mechanisms.count)
        mechanism_retained = all(
            first.primary_mechanism in {"unspecified", row.observed_mechanism}
            for row in config_rows
        )
        finite_robustness = [
            row.robustness_diagnostic
            for row in config_rows
            if math.isfinite(row.robustness_diagnostic)
        ]
        robustness_diagnostic = min(finite_robustness) if finite_robustness else float("nan")
        non_target_rows = [r for r in config_rows if r.evaluated_planner != first.target_planner]
        clusters.append(
            CandidateCluster(
                config_id=config_id,
                target_planner=first.target_planner,
                scenario_seed=first.scenario_seed,
                n_evaluated_seeds=len(config_rows),
                n_failed=sum(1 for row in config_rows if row.outcome.failed()),
                n_transferred=sum(1 for row in config_rows if row.transferred),
                n_non_target_seeds=len(non_target_rows),
                n_non_target_transferred=sum(1 for r in non_target_rows if r.transferred),
                primary_mechanism=primary,
                mechanism_retained=mechanism_retained,
                robustness_diagnostic=robustness_diagnostic,
            )
        )
    return clusters


def _validate_gate_a_configs(configs: list[CertifiedConfig]) -> None:
    """Fail closed when configs carry excluded classes or non-admitted status."""
    for config in configs:
        if config.certification_tier != "eligible":
            raise ValueError(
                f"Gate A rejects certification_tier {config.certification_tier!r} "
                f"for config={config.config_id!r}"
            )
        row_class = config.row_class.strip().lower().replace("-", "_")
        if not row_class or row_class == "excluded" or row_class in _GATE_A_EXCLUDED_ROW_CLASSES:
            raise ValueError(
                f"Gate A rejects row class {config.row_class!r} for config={config.config_id!r}"
            )
        if not isinstance(config.scenario_seed, int) or isinstance(config.scenario_seed, bool):
            raise ValueError(f"Gate A requires an integer scenario_seed for {config.config_id!r}")
        if not config.primary_mechanism.strip() or config.primary_mechanism == "unspecified":
            raise ValueError(
                f"Gate A requires a predeclared primary mechanism for {config.config_id!r}"
            )
        if config.candidate_provenance is None:
            raise ValueError(
                f"Gate A requires candidate_provenance for config={config.config_id!r}"
            )
        if config.candidate_provenance.admission_status != "admitted":
            raise ValueError(
                f"Gate A rejects non-admitted config {config.config_id!r}: "
                f"{config.candidate_provenance.admission_status!r}"
            )


def build_gate_a_transfer_matrix(
    configs: list[CertifiedConfig],
    evaluations: list[PlannerEval],
    *,
    planners: tuple[str, ...] | None = None,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 0,
) -> TransferMatrix:
    """Build the Gate A capability-only transfer matrix (v2 schema).

    This builder enforces the full Gate A contract itself: it rejects
    ``stress_only``, excluded row classes, fallback/degraded/unavailable
    execution, malformed lineage, missing or repeated seeds, and any roster
    other than exactly three planners. It does not depend on callers having
    used ``eligible_only=True``.
    """
    config_ids, target_planner = _validate_matrix_configs(configs, bootstrap_n=bootstrap_n)
    requested_planners = DEFAULT_TRANSFER_ROSTER if planners is None else tuple(planners)
    if requested_planners != DEFAULT_TRANSFER_ROSTER:
        raise ValueError(
            "Gate A requires the frozen three-planner roster "
            f"{DEFAULT_TRANSFER_ROSTER!r}; got {requested_planners!r}"
        )
    planners = _resolve_matrix_planners(
        evaluations, target_planner=target_planner, planners=requested_planners
    )
    if len(planners) != _GATE_A_REQUIRED_PLANNERS:
        raise ValueError(
            f"Gate A requires exactly {_GATE_A_REQUIRED_PLANNERS} planners; got {len(planners)}"
        )

    # Gate A rejects excluded classes and non-admitted configs regardless of
    # how the caller selected them.
    _validate_gate_a_configs(configs)

    rows = _build_gate_a_rows(configs, evaluations, planners)
    clusters = _build_candidate_clusters(rows)

    cells = _build_cells_from_rows(rows, config_ids, planners)
    ranking_rows = _build_ranking(cells, planners)
    ranking_rows.sort(
        key=lambda row: (
            not math.isfinite(row.worst_case_robustness),
            -row.worst_case_robustness if math.isfinite(row.worst_case_robustness) else 0.0,
            row.planner,
        )
    )
    for rank, row in enumerate(ranking_rows, start=1):
        ranking_rows[rank - 1] = CapabilityRanking(
            planner=row.planner,
            worst_case_robustness=row.worst_case_robustness,
            transfer_failure_rate=row.transfer_failure_rate,
            rank=rank,
        )

    rate, ci, n_candidates, n_seed_evals, _ = _candidate_clustered_transfer_rate_ci(
        rows,
        target_planner,
        n_resamples=bootstrap_n,
        seed=bootstrap_seed,
    )

    return TransferMatrix(
        schema_version=_TRANSFER_MATRIX_SCHEMA_VERSION_V2,
        target_planner=target_planner,
        configs=tuple(configs),
        config_ids=config_ids,
        planners=planners,
        cells=tuple(cells),
        rows=tuple(rows),
        clusters=tuple(clusters),
        ranking=tuple(ranking_rows),
        overall_transfer_rate=rate,
        transfer_rate_ci=ci,
        transfer_rate_bootstrap_n=bootstrap_n,
        n_candidates=n_candidates,
        n_seed_evals=n_seed_evals,
        capability_only=True,
    )


def _validate_required_activation_fields(payload: dict[str, Any]) -> list[str]:
    """Check that every required #6145 terminal result field is present."""
    errors: list[str] = []
    for field_name in _PROMOTION_RESULT_REQUIRED_FIELDS:
        if field_name not in payload:
            errors.append(f"#6145 activation payload missing required field {field_name!r}")
    return errors


def _validate_activation_hashes(
    payload: dict[str, Any], *, expected_contract_sha256: str | None
) -> list[str]:
    """Check that all referenced hashes are well-formed and match when expected."""
    errors: list[str] = []
    for hash_field in ("contract_sha256", "candidate_manifest_sha256", "evidence_packet_sha256"):
        value = payload.get(hash_field)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            errors.append(f"#6145 activation {hash_field} must be a 64-hex SHA-256 string")
    contract_sha256 = payload.get("contract_sha256")
    if (
        expected_contract_sha256 is not None
        and isinstance(contract_sha256, str)
        and _SHA256_RE.fullmatch(contract_sha256)
        and contract_sha256 != expected_contract_sha256
    ):
        errors.append(
            "#6145 activation contract_sha256 does not match the frozen powered contract hash"
        )
    return errors


def _validate_activation_decision_and_count(payload: dict[str, Any]) -> list[str]:
    """Check schema version, decision, execution commit, and admitted count."""
    errors: list[str] = []
    if payload.get("schema_version") != _PROMOTION_RESULT_SCHEMA_VERSION:
        errors.append(
            f"#6145 activation requires schema_version {_PROMOTION_RESULT_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    decision = payload.get("decision")
    if decision not in _PROMOTION_RESULT_DECISION_VALUES:
        errors.append(
            f"#6145 activation decision must be one of {_PROMOTION_RESULT_DECISION_VALUES}, "
            f"got {decision!r}"
        )
    execution_commit = payload.get("execution_commit")
    if not isinstance(execution_commit, str) or not _FULL_COMMIT_RE.fullmatch(execution_commit):
        errors.append("#6145 activation execution_commit must be a 40-hex git SHA")
    admitted = payload.get("admitted_candidate_count")
    if isinstance(admitted, bool) or not isinstance(admitted, int) or admitted < 0:
        errors.append("#6145 activation admitted_candidate_count must be a non-negative integer")
    elif isinstance(admitted, int) and admitted < _PROMOTION_MIN_ADMITTED_CANDIDATES:
        errors.append(
            f"#6145 activation requires admitted_candidate_count >= "
            f"{_PROMOTION_MIN_ADMITTED_CANDIDATES}, got {admitted}"
        )
    if decision != "promote":
        errors.append(f"#6145 activation requires decision 'promote', got {decision!r}")
    return errors


def check_issue_6145_activation(
    payload: Any, *, expected_contract_sha256: str | None = None
) -> list[str]:
    """Side-effect-free fail-closed checker for #6145 semantic activation.

    Returns a list of error strings. An empty list means the structural
    activation gates pass. This checker does not read the referenced files;
    byte-level hash verification is the responsibility of Gate B.

    Rejects ``stop``, ``inconclusive``, missing hashes, malformed hashes,
    invalid execution commit, and fewer than five admitted candidates. Issue
    closure alone never activates downstream work.
    """
    if not isinstance(payload, dict):
        return ["#6145 activation payload must be a mapping"]
    errors = _validate_required_activation_fields(payload)
    if errors:
        return errors
    errors.extend(
        _validate_activation_hashes(payload, expected_contract_sha256=expected_contract_sha256)
    )
    errors.extend(_validate_activation_decision_and_count(payload))
    return errors


def build_transfer_matrix(
    configs: list[CertifiedConfig],
    evaluations: list[PlannerEval],
    *,
    planners: tuple[str, ...] | None = None,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 0,
) -> TransferMatrix:
    """Build the legacy v1 K x N transfer matrix from certified configs + eval results.

    This is the historical slice-1 entry point. It emits
    ``adversarial_transfer_matrix.v1`` and builds cells plus the per-planner
    capability ranking. It does not require Gate A provenance and does not
    emit the required v2 candidate x planner x seed rows.

    Parameters
    ----------
    configs : list[CertifiedConfig]
        Certified worst-case configs (typically against the target planner).
    evaluations : list[PlannerEval]
        Per-planner re-evaluation results, keyed by config_id + planner.
    planners : tuple[str, ...] | None
        Planner order for the matrix columns. Defaults to the union of planners
        seen in ``evaluations`` preserving first-seen order.
    bootstrap_n : int
        Number of bootstrap resamples for the transfer-rate CI.
    bootstrap_seed : int
        Deterministic seed for the bootstrap resampling.

    Returns
    -------
    TransferMatrix
        The v1 transfer measurement, per-planner ranking, and bootstrap CI.
    """
    config_ids, target_planner = _validate_matrix_configs(configs, bootstrap_n=bootstrap_n)
    planners = _resolve_matrix_planners(
        evaluations, target_planner=target_planner, planners=planners
    )
    eval_by_key = _index_complete_evaluations(evaluations, config_ids=config_ids, planners=planners)

    cells = _build_cells(config_ids, planners, eval_by_key)
    ranking_rows = _build_ranking(cells, planners)
    ranking_rows.sort(
        key=lambda row: (
            not math.isfinite(row.worst_case_robustness),
            -row.worst_case_robustness if math.isfinite(row.worst_case_robustness) else 0.0,
            row.planner,
        )
    )
    for rank, row in enumerate(ranking_rows, start=1):
        ranking_rows[rank - 1] = PlannerRanking(
            planner=row.planner,
            worst_case_robustness=row.worst_case_robustness,
            transfer_failure_rate=row.transfer_failure_rate,
            minimax_regret=minimax_regret(row.worst_case_robustness),
            rank=rank,
        )

    # Overall transfer rate + bootstrap CI across evaluated planners (excluding
    # the target planner column, which is by construction the source of failure).
    other_planners = [p for p in planners if p != target_planner]
    if other_planners:
        grouped = _group_cells_by_planner(cells, planners)
        failures_per_planner: list[int] = []
        evals_per_planner: list[int] = []
        for planner in other_planners:
            planner_cells = grouped[planner]
            failures_per_planner.append(sum(1 for c in planner_cells if c.failed))
            evals_per_planner.append(len(planner_cells))
        rate, ci_low, ci_high = _bootstrap_transfer_rate(
            failures_per_planner,
            evals_per_planner,
            n_resamples=bootstrap_n,
            seed=bootstrap_seed,
        )
        overall_rate = rate
        ci = (ci_low, ci_high)
    else:
        overall_rate = 0.0
        ci = (0.0, 0.0)

    return TransferMatrix(
        schema_version=_TRANSFER_MATRIX_SCHEMA_VERSION_V1,
        target_planner=target_planner,
        configs=tuple(configs),
        config_ids=config_ids,
        planners=planners,
        cells=tuple(cells),
        rows=(),
        clusters=(),
        ranking=tuple(ranking_rows),
        overall_transfer_rate=overall_rate,
        transfer_rate_ci=ci,
        transfer_rate_bootstrap_n=bootstrap_n if other_planners else 0,
        n_candidates=0,
        n_seed_evals=0,
        capability_only=True,
    )


def _constraints_first_report_tier(outcome: ConstraintsFirstOutcome) -> str:
    """Return the worst observed constraints-first tier for report rendering."""
    if outcome.status != "observed":
        return "not_available"
    if outcome.collision_or_severe_intrusion is True:
        return "collision_or_severe_intrusion"
    if outcome.liveness_or_goal_completion is True:
        return "liveness_or_goal_completion"
    return "comfort_and_efficiency"


def _append_gate_a_report_diagnostics(matrix: TransferMatrix, lines: list[str]) -> None:
    """Append row-derived Gate A denominators and mechanism diagnostics."""
    if not matrix.rows:
        return
    non_target_rows = [row for row in matrix.rows if row.evaluated_planner != matrix.target_planner]
    candidate_clusters = {cluster.config_id: cluster for cluster in matrix.clusters}
    candidate_transfers = sum(
        cluster.n_non_target_transferred > 0 for cluster in candidate_clusters.values()
    )
    lines.append("## Gate A row-derived diagnostics")
    lines.append("")
    lines.append(
        "- Candidate clusters: "
        f"{len(candidate_clusters)}; transferred candidates: {candidate_transfers}; "
        f"candidate denominator: {matrix.n_candidates}"
    )
    lines.append(
        "- Non-target seed evaluations: "
        f"{len(non_target_rows)}; seed denominator: {matrix.n_seed_evals}"
    )
    lines.append(
        "- Scalar robustness is a descriptive diagnostic only; it cannot compensate "
        "for a hard constraints-first failure."
    )
    lines.append("")
    lines.append(
        "| evaluated planner | transferred seed rows | seed denominator | transfer rate | worst observed tier |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for planner in matrix.planners:
        if planner == matrix.target_planner:
            continue
        planner_rows = [row for row in non_target_rows if row.evaluated_planner == planner]
        transferred = sum(row.transferred for row in planner_rows)
        tiers = {_constraints_first_report_tier(row.outcome) for row in planner_rows}
        ordered_tiers = (
            "collision_or_severe_intrusion",
            "liveness_or_goal_completion",
            "comfort_and_efficiency",
            "not_available",
        )
        worst_tier = next((tier for tier in ordered_tiers if tier in tiers), "not_available")
        denominator = len(planner_rows)
        rate = transferred / denominator if denominator else 0.0
        lines.append(f"| `{planner}` | {transferred} | {denominator} | {rate:.3f} | {worst_tier} |")
    lines.append("")
    lines.append("### Observed mechanism distribution (non-target rows)")
    lines.append("")
    lines.append("| mechanism | seed rows |")
    lines.append("|---|---:|")
    mechanism_counts: dict[str, int] = {}
    for row in non_target_rows:
        mechanism_counts[row.observed_mechanism] = (
            mechanism_counts.get(row.observed_mechanism, 0) + 1
        )
    for mechanism, count in sorted(mechanism_counts.items()):
        lines.append(f"| `{mechanism}` | {count} |")
    lines.append("")


def render_transfer_report(matrix: TransferMatrix, *, configs: list[CertifiedConfig]) -> str:
    """Render a one-page transfer-measurement report (capability-only)."""
    if tuple(config.config_id for config in configs) != matrix.config_ids:
        raise ValueError("Report configs must match the transfer matrix config order")
    lines: list[str] = []
    schema_label = "v2" if matrix.schema_version == _TRANSFER_MATRIX_SCHEMA_VERSION_V2 else "v1"
    lines.append(
        f"# Cross-planner adversarial transfer matrix (slice 1, capability-only, {schema_label})"
    )
    lines.append("")
    lines.append(
        "> Capability-not-evidence boundary: built only from archive paths and "
        "pinned configs/seeds. Not a benchmark or paper-facing claim."
    )
    lines.append("")
    lines.append(f"- Target planner (weak points discovered against): `{matrix.target_planner}`")
    lines.append(f"- Certified configs (K): {len(matrix.config_ids)}")
    lines.append(f"- Evaluated planners (N): {len(matrix.planners)}")
    lines.append(f"- Overall transfer rate (excl. target): {matrix.overall_transfer_rate:.3f}")
    ci_label = (
        "Transfer-rate 95% CI (exploratory, small K)"
        if matrix.n_candidates < 10
        else "Transfer-rate 95% CI"
    )
    lines.append(
        f"- {ci_label}: [{matrix.transfer_rate_ci[0]:.3f}, "
        f"{matrix.transfer_rate_ci[1]:.3f}] "
        f"(bootstrap n={matrix.transfer_rate_bootstrap_n}, "
        f"candidates={matrix.n_candidates}, seed-evaluations={matrix.n_seed_evals})"
    )
    lines.append("")
    lines.append("## Capability-only ranking (diagnostic order; not a general planner rank)")
    lines.append("")
    lines.append("| rank | planner | worst-case robustness | transfer-failure rate |")
    lines.append("|---|---|---|---|")
    for row in matrix.ranking:
        wc = (
            f"{row.worst_case_robustness:.3f}"
            if math.isfinite(row.worst_case_robustness)
            else "n/a"
        )
        lines.append(f"| {row.rank} | `{row.planner}` | {wc} | {row.transfer_failure_rate:.3f} |")
    lines.append("")
    _append_gate_a_report_diagnostics(matrix, lines)
    lines.append(
        "## Transfer matrix (rows=configs, cols=planners; X=transferred failure, .=ok, ?=untested)"
    )
    lines.append("")
    header = (
        "| config | "
        + " | ".join(p.replace("scenario_adaptive_hybrid_orca", "orca") for p in matrix.planners)
        + " |"
    )
    sep = "|---|" + "|".join(["---"] * len(matrix.planners)) + "|"
    lines.append(header)
    lines.append(sep)
    by_config: dict[str, dict[str, TransferCell]] = {}
    for cell in matrix.cells:
        by_config.setdefault(cell.config_id, {})[cell.planner] = cell
    for cfg_id in matrix.config_ids:
        mark = []
        for planner in matrix.planners:
            cell = by_config.get(cfg_id, {}).get(planner)
            if cell is None or not math.isfinite(cell.robustness):
                mark.append("?")
            elif cell.transferred:
                mark.append("X")
            else:
                mark.append(".")
        lines.append(f"| `{cfg_id}` | " + " | ".join(mark) + " |")
    lines.append("")
    lines.append("## Certified config provenance")
    lines.append("")
    lines.append("| config | scenario_seed | objective | tier | source manifest |")
    lines.append("|---|---|---|---|---|")
    for cfg in configs:
        lines.append(
            f"| `{cfg.config_id}` | {cfg.scenario_seed} | {cfg.objective_value:.3f} | "
            f"{cfg.certification_tier} | `{cfg.source_manifest}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_transfer_artifact(matrix: TransferMatrix, *, out_dir: str | Path) -> Path:
    """Write the transfer matrix JSON + one-page report to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "transfer_matrix.json"
    report_path = out_dir / "transfer_report.md"
    matrix_path.write_text(
        json.dumps(matrix.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        render_transfer_report(matrix, configs=list(matrix.configs)), encoding="utf-8"
    )
    return matrix_path


def archive_transfer_run(
    matrix: TransferMatrix,
    *,
    archive_root: str | Path,
    run_id: str | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """Write the durable, provenance-pinned K x N transfer run artifact.

    This is the archival stage that issue #5303 leaves open after PR #5845's
    capability-only plumbing: it persists the transfer matrix under the
    adversarial archive path (never the release evidence store) together with
    the per-job ``execution_context.txt`` and a ``receipt_manifest.json`` that
    records every archived artifact's path and SHA-256 digest, exactly per the
    evidence-grade promotion plan's provenance discipline.

    Capability-not-evidence boundary: the archived artifacts describe *what ran*
    and the measured transfer structure; they are not a benchmark or paper-facing
    claim and are not written to the release evidence tree.

    Parameters
    ----------
    matrix : TransferMatrix
        The built transfer matrix (from :func:`build_transfer_matrix` or
        :func:`build_gate_a_transfer_matrix`).
    archive_root : str | Path
        Root of the adversarial archive. The run is written under
        ``<archive_root>/transfer_matrix/<run_id>/``.
    run_id : str | None
        Stable run identifier. Defaults to a UTC timestamped UUIDv4 string.
    repo_root : str | Path | None
        Repository root for commit resolution in the execution context.

    Returns
    -------
    Path
        The run directory containing the durable artifacts.
    """
    if not matrix.config_ids:
        raise ValueError("Cannot archive a transfer matrix with zero configs")
    if len(matrix.planners) < 3:
        raise ValueError("Transfer matrix must cover the target planner plus 2 others")
    run_id = _validated_run_id(
        run_id or _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    context = gather_execution_context(repo_root=repo_root)
    if context.commit_sha is None:
        raise RuntimeError(
            "Cannot archive a provenance-pinned transfer run without a resolved git commit"
        )
    run_dir = Path(archive_root) / _TRANSFER_ARCHIVE_DIRNAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    matrix_path = write_transfer_artifact(matrix, out_dir=run_dir)

    context_path = write_execution_context(run_dir, repo_root=repo_root)
    context_digest = sha256_of_file(context_path)
    matrix_digest = sha256_of_file(matrix_path)
    report_digest = sha256_of_file(run_dir / "transfer_report.md")
    items = [
        ReceiptItem(
            artifact="transfer_matrix_json",
            path=matrix_path.name,
            digest=matrix_digest,
            note=f"K={len(matrix.config_ids)} x N={len(matrix.planners)} transfer measurement",
        ),
        ReceiptItem(
            artifact="transfer_report_md",
            path="transfer_report.md",
            digest=report_digest,
            note="one-page capability-only transfer report",
        ),
        ReceiptItem(
            artifact="execution_context",
            path=context_path.name,
            digest=context_digest,
            note="pinned hostname/CPU/threads/commit provenance",
        ),
    ]
    write_receipt_manifest(
        run_dir,
        run_id=run_id,
        items=items,
        execution_context_path=context_path.name,
    )
    return run_dir


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "DEFAULT_TRANSFER_ROSTER",
    "CandidateCluster",
    "CapabilityRanking",
    "CertifiedConfig",
    "PlannerEval",
    "PlannerRanking",
    "TransferCell",
    "TransferMatrix",
    "TransferRow",
    "archive_transfer_run",
    "build_gate_a_transfer_matrix",
    "build_transfer_matrix",
    "check_issue_6145_activation",
    "minimax_regret",
    "render_transfer_report",
    "select_certified_configs",
    "write_transfer_artifact",
]
