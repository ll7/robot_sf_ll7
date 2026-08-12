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
pre-correction, fallback, degraded, unavailable, duplicate, malformed, and
lineage-incomplete inputs. It reports candidate-clustered uncertainty with
explicit candidate and seed denominators, a scalar robustness diagnostic only,
and no minimax or regret claims. A side-effect-free
:func:`check_issue_6145_activation` helper is provided so that downstream
activation can remain fail-closed on ``promote``, ``>= 5`` admitted candidates,
required hashes, and valid lineage; issue closure alone never activates
anything.

Capability-not-evidence boundary: the matrix is built only from archive paths
and pinned configs/seeds. No benchmark or paper-facing claim is made here; the
report is explicitly labelled capability-only.

Status: research/exploratory. These artifacts are transfer measurements, not
reported benchmark metrics.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import random
import re
import uuid
from dataclasses import dataclass
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

_TRANSFER_MATRIX_SCHEMA_VERSION = "adversarial_transfer_matrix.v2"

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
    """

    config_id: str
    planner: str
    robustness: float
    failed: bool
    seed: int | None
    mechanism: str = "unspecified"
    eval_seed: int | None = None


@dataclass(frozen=True)
class TransferCell:
    """One cell of the K x N transfer matrix."""

    config_id: str
    planner: str
    robustness: float
    failed: bool
    transferred: bool


@dataclass(frozen=True)
class TransferRow:
    """One immutable candidate x evaluated-planner x fresh-seed Gate A row.

    This is the atomic unit of the capability-only transfer contract: every
    row pins the scenario seed, the independent fresh evaluation seed, the
    observed robustness, the predeclared and observed mechanisms, and whether
    the row's lineage is complete enough to count.
    """

    config_id: str
    target_planner: str
    evaluated_planner: str
    scenario_seed: int
    eval_seed: int
    robustness: float
    failed: bool
    transferred: bool
    primary_mechanism: str
    observed_mechanism: str
    mechanism_retained: bool
    lineage_complete: bool


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
    ranking: tuple[CapabilityRanking, ...] = ()
    overall_transfer_rate: float = 0.0
    transfer_rate_ci: tuple[float, float] = (0.0, 0.0)
    transfer_rate_bootstrap_n: int = 0
    capability_only: bool = True

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "schema_version": self.schema_version,
            "target_planner": self.target_planner,
            "configs": [config.__dict__ for config in self.configs],
            "config_ids": list(self.config_ids),
            "planners": list(self.planners),
            "cells": [c.__dict__ for c in self.cells],
            "rows": [r.__dict__ for r in self.rows],
            "clusters": [c.__dict__ for c in self.clusters],
            "ranking": [r.__dict__ for r in self.ranking],
            "overall_transfer_rate": self.overall_transfer_rate,
            "transfer_rate_ci": list(self.transfer_rate_ci),
            "transfer_rate_bootstrap_n": self.transfer_rate_bootstrap_n,
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
                eligibility = str(certificate.get("benchmark_eligibility", "")).strip().lower()
                if eligibility:
                    classes.add(eligibility)
    # Reject the frozen excluded classes and any non-eligible tier.
    for excluded in _GATE_A_EXCLUDED_ROW_CLASSES:
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
    return CertifiedConfig(
        config_id=f"{manifest_path.as_posix()}#{index}",
        target_planner=target_planner,
        candidate=candidate,
        objective_value=objective_value,
        source_manifest=manifest_path.as_posix(),
        source_candidate_index=index,
        certification_tier=_candidate_certification_tier(candidate_payload) or "unknown",
        scenario_seed=scenario_seed,
        primary_mechanism=_candidate_primary_mechanism(candidate_payload),
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


def _build_gate_a_rows(
    configs: list[CertifiedConfig],
    evaluations: list[PlannerEval],
    planners: tuple[str, ...],
) -> list[TransferRow]:
    """Build immutable candidate x planner x fresh-seed Gate A rows.

    Each config must contribute one row per evaluated planner per fresh seed.
    The current architecture supports a single fresh seed per config/planner
    pair (the standard seed protocol); the row still makes the seed explicit so
    future multi-seed confirmation can be added without a schema break.
    """
    config_by_id = {config.config_id: config for config in configs}
    expected_seeds: dict[tuple[str, str], set[int]] = {
        (config.config_id, planner): set() for config in configs for planner in planners
    }
    rows: list[TransferRow] = []
    seen: set[tuple[str, str, int]] = set()

    for evaluation in evaluations:
        config = config_by_id.get(evaluation.config_id)
        if config is None:
            raise ValueError(f"Gate A row references unknown config: {evaluation.config_id!r}")
        if evaluation.planner not in planners:
            raise ValueError(f"Gate A row references unknown planner: {evaluation.planner!r}")
        eval_seed = evaluation.eval_seed if evaluation.eval_seed is not None else evaluation.seed
        if eval_seed is None or eval_seed < 0:
            raise ValueError(
                f"Gate A row missing fresh eval seed for config={evaluation.config_id!r}, "
                f"planner={evaluation.planner!r}"
            )
        if config.scenario_seed is None:
            raise ValueError(
                f"Gate A row missing scenario seed for config={evaluation.config_id!r}"
            )
        key = (evaluation.config_id, evaluation.planner, eval_seed)
        if key in seen:
            raise ValueError(
                f"Gate A duplicate row for config={evaluation.config_id!r}, "
                f"planner={evaluation.planner!r}, eval_seed={eval_seed}"
            )
        seen.add(key)
        expected_seeds[(evaluation.config_id, evaluation.planner)].add(eval_seed)
        mechanism = evaluation.mechanism or "unspecified"
        rows.append(
            TransferRow(
                config_id=evaluation.config_id,
                target_planner=config.target_planner,
                evaluated_planner=evaluation.planner,
                scenario_seed=config.scenario_seed,
                eval_seed=eval_seed,
                robustness=evaluation.robustness,
                failed=evaluation.failed,
                transferred=evaluation.failed and evaluation.planner != config.target_planner,
                primary_mechanism=config.primary_mechanism,
                observed_mechanism=mechanism,
                mechanism_retained=(config.primary_mechanism in ("unspecified", mechanism)),
                lineage_complete=True,
            )
        )

    missing = [
        (config_id, planner) for (config_id, planner), seeds in expected_seeds.items() if not seeds
    ]
    if missing:
        config_id, planner = missing[0]
        raise ValueError(
            f"Gate A matrix missing fresh-seed rows for config={config_id!r}, planner={planner!r}"
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
        finite_robustness = [row.robustness for row in config_rows if math.isfinite(row.robustness)]
        robustness_diagnostic = min(finite_robustness) if finite_robustness else float("nan")
        clusters.append(
            CandidateCluster(
                config_id=config_id,
                target_planner=first.target_planner,
                scenario_seed=first.scenario_seed,
                n_evaluated_seeds=len(config_rows),
                n_failed=sum(1 for row in config_rows if row.failed),
                n_transferred=sum(1 for row in config_rows if row.transferred),
                primary_mechanism=primary,
                mechanism_retained=mechanism_retained,
                robustness_diagnostic=robustness_diagnostic,
            )
        )
    return clusters


def build_gate_a_transfer_matrix(
    configs: list[CertifiedConfig],
    evaluations: list[PlannerEval],
    *,
    planners: tuple[str, ...] | None = None,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 0,
) -> TransferMatrix:
    """Build the Gate A capability-only transfer matrix.

    The caller is responsible for selecting configs with ``eligible_only=True``
    so that stress_only and excluded row classes are already rejected. This
    builder adds the immutable row contract, candidate clustering, and
    capability-only ranking.
    """
    config_ids, target_planner = _validate_matrix_configs(configs, bootstrap_n=bootstrap_n)
    planners = _resolve_matrix_planners(
        evaluations, target_planner=target_planner, planners=planners
    )
    eval_by_key = _index_complete_evaluations(evaluations, config_ids=config_ids, planners=planners)

    rows = _build_gate_a_rows(configs, evaluations, planners)
    clusters = _build_candidate_clusters(rows)

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
        ranking_rows[rank - 1] = CapabilityRanking(
            planner=row.planner,
            worst_case_robustness=row.worst_case_robustness,
            transfer_failure_rate=row.transfer_failure_rate,
            rank=rank,
        )

    other_planners = [p for p in planners if p != target_planner]
    if other_planners:
        failures_per_planner: list[int] = []
        evals_per_planner: list[int] = []
        grouped = _group_cells_by_planner(cells, planners)
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
        target_planner=target_planner,
        configs=tuple(configs),
        config_ids=config_ids,
        planners=planners,
        cells=tuple(cells),
        rows=tuple(rows),
        clusters=tuple(clusters),
        ranking=tuple(ranking_rows),
        overall_transfer_rate=overall_rate,
        transfer_rate_ci=ci,
        transfer_rate_bootstrap_n=bootstrap_n if other_planners else 0,
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
    """Build the K x N transfer matrix from certified configs + eval results.

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
        The transfer measurement, per-planner minimax ranking, and bootstrap CI.
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
        ranking_rows[rank - 1] = CapabilityRanking(
            planner=row.planner,
            worst_case_robustness=row.worst_case_robustness,
            transfer_failure_rate=row.transfer_failure_rate,
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
        capability_only=True,
    )


def render_transfer_report(matrix: TransferMatrix, *, configs: list[CertifiedConfig]) -> str:
    """Render a one-page transfer-measurement report (capability-only)."""
    if tuple(config.config_id for config in configs) != matrix.config_ids:
        raise ValueError("Report configs must match the transfer matrix config order")
    lines: list[str] = []
    lines.append("# Cross-planner adversarial transfer matrix (slice 1, capability-only)")
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
    lines.append(
        f"- Transfer-rate 95% CI: [{matrix.transfer_rate_ci[0]:.3f}, "
        f"{matrix.transfer_rate_ci[1]:.3f}] (bootstrap n={matrix.transfer_rate_bootstrap_n})"
    )
    lines.append("")
    lines.append("## Capability-only ranking")
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
        The built transfer matrix (from :func:`build_transfer_matrix`).
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
    "TransferCell",
    "TransferMatrix",
    "TransferRow",
    "archive_transfer_run",
    "build_gate_a_transfer_matrix",
    "build_transfer_matrix",
    "check_issue_6145_activation",
    "render_transfer_report",
    "select_certified_configs",
    "write_transfer_artifact",
]
