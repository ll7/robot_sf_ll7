#!/usr/bin/env python3
"""Preregistered cross-planner social-compliance report builder for issue #6474.

This is the *preregistered analysis script* referenced by issue #6639.  It is
intentionally written and statistically self-validated *before* the nominal
campaign data exist, so the analysis plan is frozen under the issue #6474
Domain-Aware Approval ("protocol preparation is approved, while production
execution remains separately gated").

Claim boundary (identical to the campaign preregistration)
----------------------------------------------------------
Simulator-defined social-compliance metric-family paired effects only, for the
``goal`` / ``social_force`` / ``orca`` planners on the frozen issue #6102
scenario suite with paired seeds 111-140.  The report states:

* paired mean effects by metric family and scenario family,
* percentile-bootstrap CI95,
* paired-permutation p-values with Holm step-down multiplicity control across
  the declared planner-pair-by-metric-family decisions, and
* declared support counts and denominators.

No composite social-compliance ranking, fairness, deployment-ethics,
legibility, social-validity, safety, welfare, universal-ranking, or real-world
claim is produced or permitted.  Complete valid output may reach *nominal
benchmark evidence* only for these simulator-defined metrics.

Fail-closed contract
--------------------
Rows whose execution mode is not benchmark-capable (``native`` or declared
``adapter``) are rejected and cause an unexpected-failure exit, because the
nominal campaign contract requires zero fallback / degraded / unavailable
execution rows.  ``unavailable`` *metric* statuses are honoured honestly: a
metric family that is unavailable on a row simply does not contribute support
to that row, and families with insufficient paired support are reported as
diagnostic-only rather than zero-imputed.

Exit codes
----------
0  benchmark-success: a report with declared paired effects was produced and
   every row was benchmark-capable (zero fallback/degraded).
2  unexpected failure: malformed input, schema-version mismatch, any
   fallback/degraded/unknown execution-mode row, or an internal error.
3  accepted-unavailable-only: every row was benchmark-capable but no declared
   planner-pair-by-metric-family decision reached the minimum paired support,
   so the output is diagnostic-only and not benchmark evidence.

Usage
-----
Campaign-driven (preferred)::

    uv run python scripts/benchmark/build_social_compliance_cross_planner_report_issue_6474.py \\
        --campaign-manifest docs/context/evidence/issue_6474_social_compliance_nominal_campaign_manifest.json \\
        --episode-rows output/.../social_compliance_rows.json \\
        --config configs/benchmarks/issue_6474_social_compliance_nominal_campaign.yaml \\
        --output-dir docs/context/evidence

Protocol self-test (no campaign required)::

    uv run python scripts/benchmark/build_social_compliance_cross_planner_report_issue_6474.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.control_action_latency_snqi import NATIVE_EXECUTION_MODES
from robot_sf.benchmark.fallback_policy import resolve_execution_mode
from robot_sf.benchmark.social_compliance import (
    SOCIAL_COMPLIANCE_CLAIM_CLASS,
    SOCIAL_COMPLIANCE_SCHEMA_VERSION,
)
from robot_sf.evidence.writers import write_json, write_text

REPORT_SCHEMA_VERSION = "social-compliance-cross-planner-report.v1"

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
ARTIFACT_MANIFEST_SCHEMA_VERSION = "social-compliance-campaign-artifact-manifest.v1"
MIN_PAIRED_SUPPORT = 5
DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_PERMUTATION_SAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 20260802
DEFAULT_CONFIDENCE = 0.95
DEFAULT_ALPHA = 0.05
DEFAULT_REFERENCE_PLANNER = "goal"
DEFAULT_COMPARISON_PLANNERS = ("social_force", "orca")
FROZEN_SEEDS = frozenset(range(111, 141))
FROZEN_SCENARIO_COUNT = 6
VALID_METRIC_STATUSES = frozenset({"available", "unavailable", "not_applicable"})

# Canonical metric id -> (family, units, denominator) mirror of
# robot_sf.benchmark.social_compliance, read at import time as the frozen
# contract.  Importing the module (read-only) is permitted; this script never
# edits it.
METRIC_CONTRACT: dict[str, tuple[str, str, str]] = {
    "pedestrian_deviation_mean_m": (
        "pedestrian_deviation",
        "meters",
        "tracked_pedestrian_steps_with_baseline",
    ),
    "flow_disruption_delay_s": (
        "flow_disruption",
        "seconds",
        "pedestrians_with_reference_arrival",
    ),
    "comfort_exposure_person_s": (
        "comfort_exposure",
        "person_seconds",
        "pedestrian_steps",
    ),
    "legibility_progress_deficit_m": (
        "legibility_progress",
        "meters",
        "robot_steps_before_terminal",
    ),
    "distributional_inconvenience_p90_p50_gap": (
        "distributional_inconvenience",
        "seconds",
        "pedestrians_with_delay_samples",
    ),
}
METRIC_IDS: tuple[str, ...] = tuple(METRIC_CONTRACT.keys())


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--self-test", action="store_true", help="run the protocol self-test and exit"
    )
    mode.add_argument("--campaign-manifest", type=Path, help="path to the campaign manifest JSON")
    parser.add_argument(
        "--episode-rows",
        type=Path,
        help="per-episode rows (JSON list or CSV with JSON-encoded nested metric fields)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/issue_6474_social_compliance_nominal_campaign.yaml"),
        help="campaign config path used for provenance hashing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/context/evidence"),
        help="directory for report.md / summary.json / artifact_manifest.json",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--reference-planner", default=DEFAULT_REFERENCE_PLANNER)
    parser.add_argument(
        "--comparison-planners",
        nargs="+",
        default=list(DEFAULT_COMPARISON_PLANNERS),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutation-samples", type=int, default=DEFAULT_PERMUTATION_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--report-name",
        default="issue_6474_social_compliance_nominal_campaign_report.md",
        help="filename stem written under --output-dir",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _git_head_sha(repo_root: Path) -> str:
    """Return the current HEAD commit SHA, or 'unknown' if git is unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _file_sha256(path: Path) -> str | None:
    """Return the hex SHA-256 of a file, or None if the file is absent."""

    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite non-boolean number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _normalise_text(value: Any, *, default: str = "unknown") -> str:
    """Return a lower-case non-empty label or a fail-closed default."""

    if not isinstance(value, str):
        return default
    normalised = value.strip().lower()
    return normalised or default


# --------------------------------------------------------------------------------------
# Row loading and validation
# --------------------------------------------------------------------------------------


_CSV_NESTED_FIELDS = (
    "statuses",
    "values",
    "support_counts",
    "denominators",
    "unavailable_reasons",
)
_CSV_BOOL_FIELDS = (
    "schema_valid",
    "all_families_present",
    "benchmark_success",
    "execution_mode_consistent",
)


def _decode_csv_row(raw_row: dict[str, str], *, line_number: int) -> dict[str, Any]:
    """Decode one CSV row whose nested social-compliance fields are JSON objects."""

    row: dict[str, Any] = dict(raw_row)
    for field in _CSV_NESTED_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise AnalysisError(f"CSV row {line_number} field {field!r} is not JSON") from error
        if not isinstance(decoded, dict):
            raise AnalysisError(f"CSV row {line_number} field {field!r} must decode to an object")
        row[field] = decoded
    if isinstance(row.get("seed"), str):
        try:
            row["seed"] = int(row["seed"])
        except ValueError as error:
            raise AnalysisError(f"CSV row {line_number} seed is not an integer") from error
    for field in _CSV_BOOL_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            row[field] = value.strip().lower() == "true"
    return row


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Load CSV rows with JSON-encoded nested social-compliance fields."""

    with path.open(newline="", encoding="utf-8") as handle:
        return [
            _decode_csv_row(dict(raw_row), line_number=line_number)
            for line_number, raw_row in enumerate(csv.DictReader(handle), start=2)
        ]


def _load_rows_from_path(path: Path) -> list[dict[str, Any]]:
    """Load per-episode rows from JSON or CSV with JSON-encoded nested fields."""

    if not path.exists():
        raise AnalysisError(f"episode rows file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("rows", "episode_rows", "social_compliance_rows"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return [row for row in candidate if isinstance(row, dict)]
        raise AnalysisError(f"JSON episode-rows payload at {path} is not a list of rows")
    if suffix in {".csv", ".tsv"}:
        return _load_csv_rows(path)
    raise AnalysisError(f"unsupported episode-rows format: {path}")


def _extract_rows_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Best-effort extraction of per-episode rows from a campaign manifest.

    The nominal campaign runner's exact per-episode artifact layout is defined
    by the preregistration sibling; this loader accepts the common shapes (an
    embedded ``rows`` / ``episode_rows`` list, or a sibling
    ``<stem>_social_compliance_rows.json`` next to the manifest) so the analysis
    is robust to the sibling's final choice.
    """

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisError(f"campaign manifest is not a JSON object: {manifest_path}")
    for key in ("social_compliance_rows", "episode_rows", "rows"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return [row for row in candidate if isinstance(row, dict)]
    # Look for an explicit artifact pointer.
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("social_compliance_rows", "episode_rows", "seed_episode_rows_json"):
            pointer = artifacts.get(key)
            if isinstance(pointer, str) and pointer:
                pointer_path = Path(pointer)
                candidate_path = (
                    pointer_path
                    if pointer_path.is_absolute()
                    else manifest_path.parent / pointer_path
                )
                if candidate_path.exists():
                    return _load_rows_from_path(candidate_path)
    sibling = manifest_path.with_name(manifest_path.stem + "_social_compliance_rows.json")
    if sibling.exists():
        return _load_rows_from_path(sibling)
    raise AnalysisError(
        "campaign manifest does not expose per-episode social-compliance rows; "
        "pass them explicitly via --episode-rows",
    )


def load_episode_rows(
    campaign_manifest: Path | None,
    episode_rows: Path | None,
) -> list[dict[str, Any]]:
    """Load per-episode rows, preferring an explicit --episode-rows path."""

    if episode_rows is not None:
        return _load_rows_from_path(episode_rows)
    if campaign_manifest is not None:
        return _extract_rows_from_manifest(campaign_manifest)
    raise AnalysisError("no input: provide --episode-rows or --campaign-manifest")


class AnalysisError(Exception):
    """Raised for unexpected, fail-closed analysis failures (exit code 2)."""


class AcceptedUnavailableOnly(Exception):
    """Raised when all rows are benchmark-capable but no decision has support (exit 3)."""


def _is_non_negative_int(value: Any, *, require_positive: bool = False) -> bool:
    """Return whether a value is a non-boolean integer in the required range."""

    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return value > 0 if require_positive else value >= 0


def validate_protocol_parameters(args: argparse.Namespace) -> None:
    """Reject CLI settings that would silently change the frozen analysis plan."""

    if args.reference_planner != DEFAULT_REFERENCE_PLANNER or tuple(args.comparison_planners) != (
        DEFAULT_COMPARISON_PLANNERS
    ):
        raise AnalysisError(
            "the preregistered analysis is frozen to goal versus social_force and orca",
        )
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise AnalysisError("bootstrap and permutation sample counts must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise AnalysisError("confidence must be strictly between zero and one")
    if not 0.0 < args.alpha < 1.0:
        raise AnalysisError("alpha must be strictly between zero and one")


def _row_schema_version(row: dict[str, Any]) -> str:
    """Return the social-compliance schema version recorded on a row."""

    return _normalise_text(
        row.get("social_compliance_schema_version") or row.get("schema_version"),
    )


def _row_execution_mode(row: dict[str, Any]) -> str:
    """Resolve a row's canonical execution mode, preferring explicit metadata."""

    explicit = _normalise_text(row.get("execution_mode"))
    if explicit != "unknown":
        return explicit
    resolved = _normalise_text(resolve_execution_mode(row))
    if resolved != "unknown":
        return resolved
    metadata = row.get("algorithm_metadata") or row.get("algorithm_metadata_contract")
    if isinstance(metadata, dict):
        return _normalise_text(resolve_execution_mode(metadata))
    return "unknown"


def _validate_row_identity(
    row: dict[str, Any],
    *,
    index: int,
    expected_planners: set[str],
    identities: set[tuple[str, str, int]],
) -> None:
    """Validate and record one unique frozen planner-scenario-seed identity."""

    planner = _normalise_text(row.get("planner"))
    scenario = _normalise_text(row.get("scenario_id"))
    seed = row.get("seed")
    if planner not in expected_planners:
        raise AnalysisError(f"row {index} has unexpected planner {planner!r}")
    if scenario == "unknown":
        raise AnalysisError(f"row {index} has missing scenario_id")
    if not _is_non_negative_int(seed) or seed not in FROZEN_SEEDS:
        raise AnalysisError(f"row {index} seed {seed!r} is outside frozen seeds 111-140")
    identity = (planner, scenario, seed)
    if identity in identities:
        raise AnalysisError(f"duplicate campaign row identity: {identity!r}")
    identities.add(identity)


def _validate_campaign_status(row: dict[str, Any], *, index: int) -> str:
    """Return the benchmark-capable execution mode after checking campaign status fields."""

    if (
        _normalise_text(row.get("readiness_status")) not in NATIVE_EXECUTION_MODES
        or _normalise_text(row.get("availability_status")) != "available"
        or row.get("benchmark_success") is not True
        or row.get("execution_mode_consistent") is not True
    ):
        raise AnalysisError(f"row {index} does not carry a benchmark-success campaign status")
    execution_mode = _row_execution_mode(row)
    if execution_mode not in NATIVE_EXECUTION_MODES:
        raise AnalysisError(
            "nominal campaign contract requires native or adapter execution; "
            f"row {index} has {execution_mode!r}",
        )
    return execution_mode


def _social_compliance_payloads(
    row: dict[str, Any], *, index: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the required flattened social-compliance maps for one validated row."""

    if row.get("schema_valid") is not True or row.get("all_families_present") is not True:
        raise AnalysisError(f"row {index} has an invalid or incomplete social-compliance block")
    payloads = tuple(
        row.get(field)
        for field in (
            "statuses",
            "values",
            "support_counts",
            "denominators",
            "unavailable_reasons",
        )
    )
    if not all(isinstance(payload, dict) for payload in payloads):
        raise AnalysisError(f"row {index} is missing flattened social-compliance fields")
    statuses, values, support_counts, denominators, unavailable_reasons = payloads
    if set(statuses) != set(METRIC_IDS) or set(denominators) != set(METRIC_IDS):
        raise AnalysisError(f"row {index} does not declare every social-compliance metric")
    return statuses, values, support_counts, denominators, unavailable_reasons


def _validate_metric_payload(
    *,
    index: int,
    metric_id: str,
    denominator: str,
    statuses: dict[str, Any],
    values: dict[str, Any],
    support_counts: dict[str, Any],
    denominators: dict[str, Any],
    unavailable_reasons: dict[str, Any],
) -> None:
    """Validate one metric's status, support, denominator, and value/reason fields."""

    status = _normalise_text(statuses.get(metric_id))
    support_count = support_counts.get(metric_id)
    if status not in VALID_METRIC_STATUSES:
        raise AnalysisError(f"row {index} metric {metric_id} has invalid status {status!r}")
    if denominators.get(metric_id) != denominator:
        raise AnalysisError(f"row {index} metric {metric_id} has wrong denominator")
    if not _is_non_negative_int(support_count, require_positive=status == "available"):
        raise AnalysisError(f"row {index} metric {metric_id} has invalid support count")
    if status == "available" and not _is_finite_number(values.get(metric_id)):
        raise AnalysisError(f"row {index} metric {metric_id} lacks a finite value")
    if status != "available" and (
        support_count != 0 or not isinstance(unavailable_reasons.get(metric_id), str)
    ):
        raise AnalysisError(f"row {index} metric {metric_id} has invalid unavailable metadata")


def _validate_social_compliance_row(row: dict[str, Any], *, index: int) -> None:
    """Validate all metrics in one flattened social-compliance block."""

    schema_version = _row_schema_version(row)
    if schema_version != SOCIAL_COMPLIANCE_SCHEMA_VERSION:
        raise AnalysisError(
            f"row {index} schema_version {schema_version!r} != "
            f"{SOCIAL_COMPLIANCE_SCHEMA_VERSION!r}",
        )
    statuses, values, support_counts, denominators, unavailable_reasons = (
        _social_compliance_payloads(
            row,
            index=index,
        )
    )
    for metric_id, (_family, _units, denominator) in METRIC_CONTRACT.items():
        _validate_metric_payload(
            index=index,
            metric_id=metric_id,
            denominator=denominator,
            statuses=statuses,
            values=values,
            support_counts=support_counts,
            denominators=denominators,
            unavailable_reasons=unavailable_reasons,
        )


def _validate_campaign_matrix(
    identities: set[tuple[str, str, int]], expected_planners: set[str]
) -> int:
    """Require the frozen six-scenario, 30-seed matrix for every planner."""

    planner_cells = {
        planner: {
            (scenario, seed) for row_planner, scenario, seed in identities if row_planner == planner
        }
        for planner in expected_planners
    }
    reference_cells = planner_cells[DEFAULT_REFERENCE_PLANNER]
    expected_cell_count = FROZEN_SCENARIO_COUNT * len(FROZEN_SEEDS)
    if len(reference_cells) != expected_cell_count:
        raise AnalysisError(
            "goal rows do not cover the frozen six-scenario by 30-seed matrix "
            f"(got {len(reference_cells)}, expected {expected_cell_count})",
        )
    if len({scenario for scenario, _seed in reference_cells}) != FROZEN_SCENARIO_COUNT:
        raise AnalysisError(
            f"goal rows do not contain exactly {FROZEN_SCENARIO_COUNT} frozen scenarios",
        )
    for planner, cells in planner_cells.items():
        if cells != reference_cells:
            raise AnalysisError(f"{planner} rows do not match the goal scenario-seed matrix")
    return expected_cell_count


def validate_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate per-episode rows under the fail-closed campaign contract.

    Returns:
        A (valid_rows, report) pair where ``report`` records schema checks and
        any rejected rows with reasons.  ``AnalysisError`` is raised when any
        row is fallback/degraded/unknown execution (the nominal contract
        requires zero such rows) or when the schema version is wrong.
    """

    if not rows:
        raise AnalysisError("no per-episode rows were supplied")
    schema_versions: dict[str, int] = {}
    execution_mode_counts: dict[str, int] = {}
    expected_planners = {DEFAULT_REFERENCE_PLANNER, *DEFAULT_COMPARISON_PLANNERS}
    identities: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows):
        _validate_row_identity(
            row,
            index=index,
            expected_planners=expected_planners,
            identities=identities,
        )
        schema_version = _row_schema_version(row)
        schema_versions[schema_version] = schema_versions.get(schema_version, 0) + 1
        _validate_social_compliance_row(row, index=index)
        execution_mode = _validate_campaign_status(row, index=index)
        execution_mode_counts[execution_mode] = execution_mode_counts.get(execution_mode, 0) + 1
    expected_cell_count = _validate_campaign_matrix(identities, expected_planners)
    report = {
        "row_count": len(rows),
        "valid_row_count": len(rows),
        "expected_row_count": expected_cell_count * len(expected_planners),
        "scenario_count": FROZEN_SCENARIO_COUNT,
        "seed_count": len(FROZEN_SEEDS),
        "schema_versions": schema_versions,
        "execution_mode_counts": execution_mode_counts,
        "rejected": [],
    }
    return list(rows), report


# --------------------------------------------------------------------------------------
# Paired effects, bootstrap CI, permutation p-value, Holm correction
# --------------------------------------------------------------------------------------


def _metric_value(row: dict[str, Any], metric_id: str) -> float | None:
    """Return a finite available metric value, or None if unavailable/invalid."""

    statuses = row.get("statuses")
    values = row.get("values")
    if not isinstance(statuses, dict) or not isinstance(values, dict):
        return None
    if _normalise_text(statuses.get(metric_id)) != "available":
        return None
    raw = values.get(metric_id)
    if not _is_finite_number(raw):
        return None
    return float(raw)


def build_matched_cells(
    rows: Sequence[dict[str, Any]],
    reference: str,
    comparison: str,
    metric_id: str,
) -> list[dict[str, Any]]:
    """Build paired cells keyed by (scenario_id, seed) for one pair and metric.

    A cell contributes only when both arms have a benchmark-capable row whose
    ``metric_id`` status is ``available`` with a finite value.
    """

    by_key: dict[tuple[str, Any], dict[str, dict[str, Any]]] = {}
    for row in rows:
        planner = _normalise_text(row.get("planner"))
        if planner not in {reference, comparison}:
            continue
        scenario = _normalise_text(row.get("scenario_id"))
        seed = row.get("seed")
        value = _metric_value(row, metric_id)
        cell_key = (scenario, seed)
        by_key.setdefault(cell_key, {})[planner] = {
            "value": value,
            "row": row,
        }
    matched: list[dict[str, Any]] = []
    for (scenario, seed), arms in by_key.items():
        ref_arm = arms.get(reference)
        comp_arm = arms.get(comparison)
        if not ref_arm or not comp_arm:
            continue
        if ref_arm["value"] is None or comp_arm["value"] is None:
            continue
        matched.append(
            {
                "scenario_id": scenario,
                "seed": seed,
                "reference_value": ref_arm["value"],
                "comparison_value": comp_arm["value"],
                "difference": comp_arm["value"] - ref_arm["value"],
            },
        )
    return matched


def percentile_bootstrap_ci(
    diffs: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the mean of paired differences."""

    if diffs.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = diffs.size
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        draw = rng.integers(0, n, size=n)
        means[i] = float(diffs[draw].mean())
    alpha = 1.0 - confidence
    low = float(np.percentile(means, 100.0 * alpha / 2.0))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def paired_permutation_pvalue(
    diffs: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    """Two-sided Monte-Carlo sign-flip p-value for the mean paired difference."""

    if diffs.size == 0:
        return float("nan")
    observed = float(np.abs(diffs.mean()))
    if not math.isfinite(observed):
        return float("nan")
    # Exact enumeration when feasible, otherwise Monte-Carlo sign-flipping.
    n = diffs.size
    if n <= 12:
        rng = np.random.default_rng(seed)
        total = 0
        extreme = 0
        for bits in range(1 << n):
            signs = np.array([1.0 if (bits >> k) & 1 else -1.0 for k in range(n)])
            stat = abs(float((diffs * signs).mean()))
            total += 1
            if stat >= observed - 1e-15:
                extreme += 1
        return (extreme + 1.0) / (total + 1.0)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(samples, n))
    stats = np.abs((diffs[None, :] * signs).mean(axis=1))
    extreme = int(np.count_nonzero(stats >= observed - 1e-15))
    return (extreme + 1.0) / (samples + 1.0)


def holm_multiplicity(p_values: Sequence[float], *, alpha: float) -> list[float]:
    """Holm step-down adjusted p-values (monotonized), mirroring the repo helper."""

    entries = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(entries)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, original_index in enumerate(entries):
        raw = float(p_values[original_index])
        if not math.isfinite(raw):
            correction = float("nan")
        else:
            correction = min(1.0, raw * (m - rank))
        if math.isfinite(correction):
            running_max = max(running_max, correction)
            adjusted[original_index] = running_max
        else:
            adjusted[original_index] = float("nan")
    return adjusted


# --------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------


def _planner_pairs(reference: str, comparisons: Sequence[str]) -> list[tuple[str, str]]:
    """Return the ordered (reference, comparison) planner pairs."""

    return [(reference, comp) for comp in comparisons]


def _scenario_family(scenario_id: str) -> str:
    """Coarse scenario-family bucket used for stratified reporting.

    The nominal campaign freezes six issue #6102 scenarios; the exact family
    partition is defined by the preregistration sibling.  We group descriptively
    by the recorded ``scenario_id`` so the analysis is robust to the sibling's
    final naming while still reporting per-scenario support.
    """

    return scenario_id


def compute_decisions(
    rows: Sequence[dict[str, Any]],
    *,
    reference: str,
    comparisons: Sequence[str],
    bootstrap_samples: int,
    permutation_samples: int,
    bootstrap_seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    """Compute all planner-pair-by-metric-family paired-effect decisions."""

    decisions: list[dict[str, Any]] = []
    for reference_planner, comparison_planner in _planner_pairs(reference, comparisons):
        for metric_id in METRIC_IDS:
            family, units, denominator = METRIC_CONTRACT[metric_id]
            cells = build_matched_cells(rows, reference_planner, comparison_planner, metric_id)
            diffs = np.asarray([cell["difference"] for cell in cells], dtype=np.float64)
            if diffs.size >= MIN_PAIRED_SUPPORT:
                mean_diff = float(diffs.mean())
                ci_low, ci_high = percentile_bootstrap_ci(
                    diffs,
                    samples=bootstrap_samples,
                    confidence=confidence,
                    seed=bootstrap_seed,
                )
                raw_p = paired_permutation_pvalue(
                    diffs,
                    samples=permutation_samples,
                    seed=bootstrap_seed,
                )
            else:
                mean_diff = float("nan")
                ci_low, ci_high = float("nan"), float("nan")
                raw_p = float("nan")
            # Per-scenario support, declared honestly.
            per_scenario: dict[str, dict[str, Any]] = {}
            for cell in cells:
                bucket = _scenario_family(cell["scenario_id"])
                per_scenario.setdefault(bucket, {"n": 0, "differences": []})
                per_scenario[bucket]["n"] += 1
                per_scenario[bucket]["differences"].append(cell["difference"])
            scenario_support = [
                {
                    "scenario_id": scenario,
                    "n_paired": data["n"],
                    "mean_difference": float(np.mean(data["differences"]))
                    if data["differences"]
                    else float("nan"),
                }
                for scenario, data in sorted(per_scenario.items())
            ]
            decisions.append(
                {
                    "reference_planner": reference_planner,
                    "comparison_planner": comparison_planner,
                    "metric_id": metric_id,
                    "metric_family": family,
                    "units": units,
                    "denominator": denominator,
                    "n_paired": int(diffs.size),
                    "mean_difference": mean_diff,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "raw_p_value": raw_p,
                    "confidence": confidence,
                    "scenario_support": scenario_support,
                },
            )
    return decisions


def apply_multiplicity(
    decisions: Sequence[dict[str, Any]],
    *,
    alpha: float,
) -> None:
    """Attach Holm-adjusted p-values and rejection flags in place."""

    raw_p_values = [decision["raw_p_value"] for decision in decisions]
    adjusted = holm_multiplicity(raw_p_values, alpha=alpha)
    for decision, adj in zip(decisions, adjusted, strict=True):
        decision["holm_adjusted_p_value"] = adj
        decision["rejected_at_alpha"] = bool(
            math.isfinite(adj) and adj <= alpha and decision["n_paired"] >= MIN_PAIRED_SUPPORT
        )


def build_report_markdown(
    *,
    decisions: Sequence[dict[str, Any]],
    validation_report: dict[str, Any],
    provenance: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    """Render the human-readable paired-effects report."""

    alpha = policy["alpha"]
    confidence = policy["confidence"]
    lines: list[str] = []
    lines.append("# Nominal social-compliance cross-planner report (issue #6474)")
    lines.append("")
    lines.append("> AI-GENERATED NEEDS-REVIEW")
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append(
        "Simulator-defined social-compliance metric-family paired effects only, for the "
        f"{provenance['reference_planner']} / "
        f"{' / '.join(provenance['comparison_planners'])} planners. Effects are mean paired "
        f"differences (comparison - reference) with percentile-bootstrap "
        f"{int(confidence * 100)}% CI and two-sided paired-permutation p-values under Holm "
        f"step-down multiplicity control across the planner-pair-by-metric-family decisions "
        f"(family-wise alpha = {alpha}). Declared support counts and denominators are reported "
        "per decision; metric families with insufficient paired support are marked "
        "diagnostic-only and never zero-imputed."
    )
    lines.append("")
    lines.append(
        "No composite social-compliance ranking, fairness, deployment-ethics, legibility, "
        "social-validity, safety, welfare, universal-ranking, or real-world claim is made. "
        "Complete valid output may reach nominal benchmark evidence only for these "
        "simulator-defined metrics."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- campaign config: `{provenance.get('config_path', 'unknown')}`")
    lines.append(f"- config sha256: `{provenance.get('config_sha256') or 'absent'}`")
    lines.append(f"- commit sha: `{provenance.get('commit_sha', 'unknown')}`")
    lines.append(
        f"- campaign manifest: `{provenance.get('campaign_manifest', 'none (self-test)')}`"
    )
    lines.append(
        f"- campaign manifest sha256: `{provenance.get('campaign_manifest_sha256') or 'absent'}`",
    )
    lines.append(f"- rows validated: {validation_report['row_count']}")
    lines.append(f"- benchmark-capable rows: {validation_report['valid_row_count']}")
    lines.append(
        f"- execution modes: {json.dumps(validation_report['execution_mode_counts'], sort_keys=True)}",
    )
    lines.append(
        f"- rejected (fallback/degraded/unknown) rows: {len(validation_report['rejected'])}",
    )
    lines.append(f"- schema version: `{SOCIAL_COMPLIANCE_SCHEMA_VERSION}`")
    lines.append(f"- claim class: `{SOCIAL_COMPLIANCE_CLAIM_CLASS}`")
    lines.append("")
    lines.append("## Paired effects by planner pair and metric family")
    lines.append("")
    for pair_key, pair_decisions in _group_by_pair(decisions):
        reference_planner, comparison_planner = pair_key
        lines.append(
            f"### {comparison_planner} vs {reference_planner} "
            f"(mean difference = {comparison_planner} - {reference_planner})",
        )
        lines.append("")
        lines.append(
            "| metric family | units | n paired | mean diff | CI95 low | CI95 high | "
            "raw p | Holm adj. p | reject @alpha | denominator |",
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for decision in pair_decisions:
            lines.append(
                "| {family} | {units} | {n} | {mean:.6g} | {lo:.6g} | {hi:.6g} | "
                "{rawp:.6g} | {adjp:.6g} | {rej} | {denom} |".format(
                    family=decision["metric_family"],
                    units=decision["units"],
                    n=decision["n_paired"],
                    mean=decision["mean_difference"],
                    lo=decision["ci95_low"],
                    hi=decision["ci95_high"],
                    rawp=decision["raw_p_value"],
                    adjp=decision["holm_adjusted_p_value"],
                    rej="yes" if decision["rejected_at_alpha"] else "no",
                    denom=decision["denominator"],
                ),
            )
        lines.append("")
    lines.append("## Scenario-family support")
    lines.append("")
    for decision in decisions:
        if decision["n_paired"] < MIN_PAIRED_SUPPORT:
            lines.append(
                f"- {decision['comparison_planner']} vs {decision['reference_planner']} / "
                f"{decision['metric_family']}: diagnostic-only (n_paired="
                f"{decision['n_paired']} < {MIN_PAIRED_SUPPORT}).",
            )
    lines.append("")
    lines.append("## Evidence classification")
    lines.append("")
    any_supported = any(decision["n_paired"] >= MIN_PAIRED_SUPPORT for decision in decisions)
    if any_supported:
        lines.append(
            "Supported decisions constitute nominal benchmark evidence for the stated "
            "simulator-defined metric-family estimands only. Diagnostic-only families are "
            "reported with declared support and are excluded from inference.",
        )
    else:
        lines.append(
            "No planner-pair-by-metric-family decision reached the minimum paired support; "
            "this output is diagnostic-only, not benchmark evidence.",
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _group_by_pair(
    decisions: Sequence[dict[str, Any]],
) -> Iterable[tuple[tuple[str, str], list[dict[str, Any]]]]:
    """Yield (pair, decisions) preserving planner-pair order."""

    order: list[tuple[str, str]] = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for decision in decisions:
        key = (decision["reference_planner"], decision["comparison_planner"])
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(decision)
    for key in order:
        yield key, buckets[key]


def write_artifacts(
    *,
    output_dir: Path,
    report_name: str,
    report_markdown: str,
    decisions: Sequence[dict[str, Any]],
    validation_report: dict[str, Any],
    provenance: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Path]:
    """Write report.md, summary.json, and (when provenance is real) artifact manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / report_name
    write_text(report_path, report_markdown, issue_ref="robot_sf#6474")
    summary_path = output_dir / report_path.with_suffix(".summary.json").name
    write_json(
        summary_path,
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "social_compliance_schema_version": SOCIAL_COMPLIANCE_SCHEMA_VERSION,
            "claim_class": SOCIAL_COMPLIANCE_CLAIM_CLASS,
            "claim_boundary": _CLAIM_BOUNDARY_SUMMARY,
            "provenance": provenance,
            "policy": policy,
            "validation": validation_report,
            "decisions": list(decisions),
        },
    )
    paths = {"report": report_path, "summary": summary_path}
    if (
        provenance.get("campaign_manifest")
        and provenance.get("campaign_manifest") != "none (self-test)"
    ):
        manifest_path = (
            output_dir / "issue_6474_social_compliance_nominal_campaign_artifact_manifest.json"
        )
        write_json(
            manifest_path,
            {
                "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
                "campaign_manifest": provenance.get("campaign_manifest"),
                "campaign_manifest_sha256": provenance.get("campaign_manifest_sha256"),
                "config_path": provenance.get("config_path"),
                "config_sha256": provenance.get("config_sha256"),
                "commit_sha": provenance.get("commit_sha"),
                "report_path": str(report_path),
                "summary_path": str(summary_path),
                "row_count": validation_report["row_count"],
                "valid_row_count": validation_report["valid_row_count"],
                "execution_mode_counts": validation_report["execution_mode_counts"],
                "rejected_row_count": len(validation_report["rejected"]),
                "note": (
                    "Promoted only after the full 540-row campaign completes with zero "
                    "fallback/degraded rows (issue #6639 stop conditions)."
                ),
            },
        )
        paths["artifact_manifest"] = manifest_path
    return paths


_CLAIM_BOUNDARY_SUMMARY = (
    "Simulator-defined social-compliance metric-family paired effects only; Holm multiplicity "
    "control over planner-pair-by-metric-family decisions; no composite ranking, fairness, "
    "ethics, safety, social-validity, or real-world claim."
)


# --------------------------------------------------------------------------------------
# Self-test (known-answer synthetic campaign)
# --------------------------------------------------------------------------------------


def _synthetic_row(
    planner: str,
    scenario_id: str,
    seed: int,
    metric_values: dict[str, float],
    *,
    unavailable: Sequence[str] = (),
    execution_mode: str = "native",
) -> dict[str, Any]:
    """Build one schema-valid synthetic episode row for the self-test."""

    statuses: dict[str, str] = {}
    values: dict[str, float] = {}
    support_counts: dict[str, int] = {}
    unavailable_reasons: dict[str, str | None] = {}
    denominators: dict[str, str] = {}
    for metric_id, (family, _units, denominator) in METRIC_CONTRACT.items():
        denominators[metric_id] = denominator
        if metric_id in unavailable:
            statuses[metric_id] = "unavailable"
            unavailable_reasons[metric_id] = f"{family} reference unavailable"
            support_counts[metric_id] = 0
        else:
            statuses[metric_id] = "available"
            values[metric_id] = metric_values.get(metric_id, 0.0)
            support_counts[metric_id] = 240
    return {
        "planner": planner,
        "scenario_id": scenario_id,
        "seed": seed,
        "execution_mode": execution_mode,
        "readiness_status": execution_mode,
        "benchmark_success": True,
        "availability_status": "available",
        "execution_mode_consistent": True,
        "social_compliance_schema_version": SOCIAL_COMPLIANCE_SCHEMA_VERSION,
        "statuses": statuses,
        "values": values,
        "support_counts": support_counts,
        "unavailable_reasons": unavailable_reasons,
        "denominators": denominators,
        "schema_valid": True,
        "all_families_present": True,
    }


def run_self_test() -> int:
    """Build a known-answer synthetic campaign and assert the pipeline recovers it.

    Returns process exit code (0 on success).  The synthetic campaign has a
    fixed comfort-exposure paired effect of social_force - goal = +0.5
    person*seconds across 6 scenarios x 30 seeds, with the other four families
    unavailable on every row.  The self-test asserts the recovered mean paired
    difference, CI coverage, Holm-adjusted p-value ordering, and the
    fail-closed fallback/degraded guard.
    """

    rng = np.random.default_rng(20260802)
    scenarios = [f"medium_band_scenario_{i}" for i in range(6)]
    seeds = list(range(111, 141))
    rows: list[dict[str, Any]] = []
    true_effect = 0.5
    for scenario in scenarios:
        for seed in seeds:
            noise = float(rng.normal(0.0, 0.05))
            goal_value = 1.0 + noise
            sf_value = goal_value + true_effect + float(rng.normal(0.0, 0.05))
            orca_value = goal_value + float(rng.normal(0.0, 0.05))  # null effect vs goal
            other_unavailable = (
                "pedestrian_deviation_mean_m",
                "flow_disruption_delay_s",
                "legibility_progress_deficit_m",
                "distributional_inconvenience_p90_p50_gap",
            )
            rows.append(
                _synthetic_row(
                    "goal",
                    scenario,
                    seed,
                    {"comfort_exposure_person_s": goal_value},
                    unavailable=other_unavailable,
                ),
            )
            rows.append(
                _synthetic_row(
                    "social_force",
                    scenario,
                    seed,
                    {"comfort_exposure_person_s": sf_value},
                    unavailable=other_unavailable,
                    execution_mode="adapter",
                ),
            )
            rows.append(
                _synthetic_row(
                    "orca",
                    scenario,
                    seed,
                    {"comfort_exposure_person_s": orca_value},
                    unavailable=other_unavailable,
                    execution_mode="adapter",
                ),
            )
    valid_rows, validation_report = validate_rows(rows)
    assert validation_report["valid_row_count"] == len(rows), "all synthetic rows must be valid"
    assert validation_report["rejected"] == [], "no synthetic row should be rejected"
    decisions = compute_decisions(
        valid_rows,
        reference="goal",
        comparisons=["social_force", "orca"],
        bootstrap_samples=2000,
        permutation_samples=2000,
        bootstrap_seed=20260802,
        confidence=0.95,
    )
    apply_multiplicity(decisions, alpha=0.05)
    by_key = {
        (d["reference_planner"], d["comparison_planner"], d["metric_id"]): d for d in decisions
    }
    sf_comfort = by_key[("goal", "social_force", "comfort_exposure_person_s")]
    orca_comfort = by_key[("goal", "orca", "comfort_exposure_person_s")]
    assert sf_comfort["n_paired"] == 180, sf_comfort["n_paired"]
    assert abs(sf_comfort["mean_difference"] - true_effect) < 0.03, sf_comfort["mean_difference"]
    assert sf_comfort["ci95_low"] < true_effect < sf_comfort["ci95_high"], (
        sf_comfort["ci95_low"],
        sf_comfort["ci95_high"],
    )
    assert sf_comfort["raw_p_value"] < 0.001, sf_comfort["raw_p_value"]
    assert sf_comfort["rejected_at_alpha"] is True
    # Null-effect pair should not be rejected and should have a larger raw p-value.
    assert orca_comfort["raw_p_value"] > sf_comfort["raw_p_value"], (
        orca_comfort["raw_p_value"],
        sf_comfort["raw_p_value"],
    )
    assert orca_comfort["holm_adjusted_p_value"] >= sf_comfort["holm_adjusted_p_value"]
    # Unavailable families are diagnostic-only.
    for metric_id in (
        "pedestrian_deviation_mean_m",
        "flow_disruption_delay_s",
        "legibility_progress_deficit_m",
        "distributional_inconvenience_p90_p50_gap",
    ):
        decision = by_key[("goal", "social_force", metric_id)]
        assert decision["n_paired"] == 0, metric_id
        assert decision["mean_difference"] != decision["mean_difference"]  # NaN
        assert decision["rejected_at_alpha"] is False
    # Fail-closed guard: a fallback row must raise AnalysisError.
    bad_rows = list(rows)
    bad_rows.append(
        _synthetic_row(
            "goal",
            "fallback_guard_scenario",
            seeds[0],
            {"comfort_exposure_person_s": 1.0},
            unavailable=other_unavailable,
            execution_mode="fallback",
        ),
    )
    try:
        validate_rows(bad_rows)
    except AnalysisError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("fallback execution row was not rejected by validate_rows")
    # A complete, unique frozen matrix and valid social-compliance blocks are mandatory.
    for malformed_rows, reason in (
        (rows[:-1], "incomplete campaign matrix"),
        (rows + [rows[0]], "duplicate campaign identity"),
        (
            [{**rows[0], "schema_valid": False}, *rows[1:]],
            "schema-invalid social-compliance block",
        ),
    ):
        try:
            validate_rows(malformed_rows)
        except AnalysisError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError(f"{reason} was not rejected by validate_rows")
    print(
        "self-test OK: recovered comfort_exposure social_force-goal mean diff "
        f"{sf_comfort['mean_difference']:.4f} (true {true_effect}), "
        f"CI95=[{sf_comfort['ci95_low']:.4f}, {sf_comfort['ci95_high']:.4f}], "
        f"raw p={sf_comfort['raw_p_value']:.4g}, Holm adj p={sf_comfort['holm_adjusted_p_value']:.4g}",
    )
    return 0


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report builder or the protocol self-test."""

    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    validate_protocol_parameters(args)
    policy = {
        "reference_planner": args.reference_planner,
        "comparison_planners": list(args.comparison_planners),
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "confidence": args.confidence,
        "alpha": args.alpha,
    }
    rows = load_episode_rows(args.campaign_manifest, args.episode_rows)
    valid_rows, validation_report = validate_rows(rows)
    decisions = compute_decisions(
        valid_rows,
        reference=args.reference_planner,
        comparisons=args.comparison_planners,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
    )
    apply_multiplicity(decisions, alpha=args.alpha)
    any_supported = any(decision["n_paired"] >= MIN_PAIRED_SUPPORT for decision in decisions)
    provenance = {
        "config_path": str(args.config),
        "config_sha256": _file_sha256(args.config),
        "commit_sha": _git_head_sha(args.repo_root),
        "campaign_manifest": str(args.campaign_manifest) if args.campaign_manifest else None,
        "campaign_manifest_sha256": _file_sha256(args.campaign_manifest)
        if args.campaign_manifest
        else None,
        "reference_planner": args.reference_planner,
        "comparison_planners": list(args.comparison_planners),
    }
    report_markdown = build_report_markdown(
        decisions=decisions,
        validation_report=validation_report,
        provenance=provenance,
        policy=policy,
    )
    paths = write_artifacts(
        output_dir=args.output_dir,
        report_name=args.report_name,
        report_markdown=report_markdown,
        decisions=decisions,
        validation_report=validation_report,
        provenance=provenance,
        policy=policy,
    )
    primary = paths["report"]
    print(str(primary))
    if not any_supported:
        raise AcceptedUnavailableOnly(
            "all rows benchmark-capable but no decision reached minimum paired support",
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcceptedUnavailableOnly as error:
        print(
            f"accepted-unavailable-only (diagnostic, not benchmark evidence): {error}",
            file=sys.stderr,
        )
        raise SystemExit(3)
    except AnalysisError as error:
        print(f"unexpected failure: {error}", file=sys.stderr)
        raise SystemExit(2)
