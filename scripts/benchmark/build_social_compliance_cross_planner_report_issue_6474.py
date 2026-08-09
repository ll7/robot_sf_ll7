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
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_PERMUTATION_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 123
DEFAULT_CONFIDENCE = 0.95
DEFAULT_ALPHA = 0.05
FROZEN_SEEDS = frozenset(range(111, 141))
FROZEN_PLANNERS = ("goal", "social_force", "orca")
FROZEN_PLANNER_PAIRS = (
    ("goal", "social_force"),
    ("goal", "orca"),
    ("social_force", "orca"),
)
FROZEN_SCENARIOS = (
    "classic_head_on_corridor_medium",
    "classic_doorway_medium",
    "classic_group_crossing_medium",
    "classic_merging_medium",
    "classic_overtaking_medium",
    "classic_station_platform_medium",
)
FROZEN_SCENARIO_FAMILIES = {
    "classic_head_on_corridor_medium": "head_on_corridor",
    "classic_doorway_medium": "doorway",
    "classic_group_crossing_medium": "group_crossing",
    "classic_merging_medium": "merging",
    "classic_overtaking_medium": "overtaking",
    "classic_station_platform_medium": "station_platform",
}
FROZEN_CONFIG_SHA256 = "fed85cef7ac43817d0aa47a3ac10f9e7f4b50b4be6410e796fdf3d837e69811e"
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
        help=(
            "per-episode rows (JSON/CSV, a JSONL file, or a directory containing "
            "episodes.jsonl files)"
        ),
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


def _path_sha256(path: Path | None) -> str | None:
    """Return a deterministic SHA-256 for a file or a directory tree."""

    if path is None or not path.exists():
        return None
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        file_digest = _file_sha256(item)
        if file_digest is None:  # pragma: no cover - a file cannot disappear in normal use
            raise AnalysisError(f"input file disappeared while hashing: {item}")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest.encode("ascii"))
    return digest.hexdigest()


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


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load strict JSONL episode rows and fail closed on malformed lines."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise AnalysisError(
                    f"JSONL episode-rows file has a blank line: {path}:{line_number}"
                )
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise AnalysisError(f"JSONL row {path}:{line_number} is not valid JSON") from error
            if not isinstance(payload, dict):
                raise AnalysisError(f"JSONL row {path}:{line_number} is not an object")
            rows.append(payload)
    return rows


def _parse_bool(value: Any) -> bool | None:
    """Parse JSON and summary-file boolean spellings without guessing."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _campaign_summary_path(manifest_path: Path) -> Path | None:
    """Find the canonical campaign summary adjacent to a retrieved manifest."""

    candidate = manifest_path.parent / "reports" / "campaign_summary.json"
    return candidate if candidate.is_file() else None


def _load_campaign_status_context(
    campaign_manifest: Path | None,
) -> dict[str, dict[str, Any]] | None:
    """Load planner execution status from the retrieved campaign summary.

    Raw episode JSONL rows contain episode outcomes (for example ``failure`` or
    ``collision``), not the campaign-level benchmark-success state.  The
    summary is therefore the authoritative source for the status fields that
    the analysis contract requires.  Missing context is retained as ``None``
    so raw input fails closed instead of receiving invented status values.
    """

    if campaign_manifest is None:
        return None
    if not campaign_manifest.is_file():
        raise AnalysisError(f"campaign manifest not found: {campaign_manifest}")
    summary_path = _campaign_summary_path(campaign_manifest)
    if summary_path is None:
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AnalysisError(f"campaign summary is not valid JSON: {summary_path}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("planner_rows"), list):
        raise AnalysisError(f"campaign summary has no planner_rows list: {summary_path}")
    context: dict[str, dict[str, Any]] = {}
    for index, planner_row in enumerate(payload["planner_rows"]):
        if not isinstance(planner_row, dict):
            raise AnalysisError(f"campaign summary planner row {index} is not an object")
        planner = _normalise_text(planner_row.get("algo") or planner_row.get("planner_key"))
        if planner in context:
            raise AnalysisError(f"campaign summary has duplicate planner row: {planner!r}")
        context[planner] = {
            "execution_mode": planner_row.get("execution_mode"),
            "readiness_status": planner_row.get("readiness_status"),
            "availability_status": planner_row.get("availability_status"),
            "benchmark_success": _parse_bool(planner_row.get("benchmark_success")),
            "summary_path": str(summary_path),
        }
    return context


def _flatten_raw_episode_row(
    raw_row: dict[str, Any],
    *,
    index: int,
    campaign_status: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Convert one camera-ready episode record to the analysis row contract."""

    metrics = raw_row.get("metrics")
    social = metrics.get("social_compliance") if isinstance(metrics, dict) else None
    metadata = raw_row.get("algorithm_metadata")
    kinematics = metadata.get("planner_kinematics") if isinstance(metadata, dict) else None
    if (
        not isinstance(social, dict)
        or not isinstance(metadata, dict)
        or not isinstance(kinematics, dict)
    ):
        raise AnalysisError(f"raw episode row {index} lacks required social-compliance metadata")

    planner = _normalise_text(raw_row.get("algo"))
    if campaign_status is None or planner not in campaign_status:
        raise AnalysisError(
            "raw JSONL input requires campaign summary planner status context; "
            f"missing context for {planner!r} (pass --campaign-manifest beside reports/campaign_summary.json)",
        )
    status = campaign_status[planner]
    raw_execution_mode = _normalise_text(kinematics.get("execution_mode"))
    expected_execution_mode = _normalise_text(status.get("execution_mode"))
    adapter_active = kinematics.get("adapter_active")
    metadata_status = _normalise_text(metadata.get("status"))
    execution_mode_consistent = (
        raw_execution_mode in NATIVE_EXECUTION_MODES
        and expected_execution_mode == raw_execution_mode
        and isinstance(adapter_active, bool)
        and adapter_active is (raw_execution_mode == "adapter")
        and metadata_status == "ok"
    )

    schema_version = social.get("schema_version")
    claim_class = social.get("claim_class")
    metric_rows = social.get("metrics")
    if (
        schema_version != SOCIAL_COMPLIANCE_SCHEMA_VERSION
        or claim_class != SOCIAL_COMPLIANCE_CLAIM_CLASS
        or not isinstance(metric_rows, dict)
        or set(metric_rows) != set(METRIC_IDS)
    ):
        raise AnalysisError(f"raw episode row {index} has an invalid social-compliance schema")

    result_provenance = raw_row.get("result_provenance")
    source_commit = raw_row.get("git_hash")
    if not isinstance(source_commit, str) and isinstance(result_provenance, dict):
        source_commit = result_provenance.get("repo_commit")

    statuses: dict[str, Any] = {}
    values: dict[str, Any] = {}
    support_counts: dict[str, Any] = {}
    denominators: dict[str, Any] = {}
    unavailable_reasons: dict[str, Any] = {}
    for metric_id in METRIC_IDS:
        metric_row = metric_rows[metric_id]
        if not isinstance(metric_row, dict):
            raise AnalysisError(f"raw episode row {index} metric {metric_id} is not an object")
        if metric_row.get("id", metric_id) != metric_id:
            raise AnalysisError(f"raw episode row {index} metric id does not match {metric_id}")
        statuses[metric_id] = metric_row.get("status")
        values[metric_id] = metric_row.get("value")
        support_counts[metric_id] = metric_row.get("support_count")
        denominators[metric_id] = metric_row.get("denominator")
        unavailable_reasons[metric_id] = metric_row.get("unavailable_reason")

    return {
        "planner": planner,
        "scenario_id": raw_row.get("scenario_id"),
        "seed": raw_row.get("seed"),
        "execution_mode": raw_execution_mode,
        "readiness_status": status.get("readiness_status"),
        "benchmark_success": status.get("benchmark_success"),
        "availability_status": status.get("availability_status"),
        "execution_mode_consistent": execution_mode_consistent,
        "social_compliance_schema_version": schema_version,
        "statuses": statuses,
        "values": values,
        "support_counts": support_counts,
        "unavailable_reasons": unavailable_reasons,
        "denominators": denominators,
        "schema_valid": True,
        "all_families_present": True,
        "campaign_source_commit_sha": source_commit,
    }


def _normalise_loaded_rows(
    rows: Sequence[dict[str, Any]],
    *,
    campaign_status: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize raw camera-ready rows while preserving already-flat rows."""

    normalised: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("social_compliance"), dict):
            normalised.append(
                _flatten_raw_episode_row(
                    row,
                    index=index,
                    campaign_status=campaign_status,
                ),
            )
        else:
            normalised.append(row)
    return normalised


def _campaign_source_commit(
    campaign_manifest: Path | None,
    rows: Sequence[dict[str, Any]],
) -> str | None:
    """Resolve and cross-check the producing commit recorded by the campaign."""

    candidates: set[str] = set()
    if campaign_manifest is not None:
        try:
            payload = json.loads(campaign_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnalysisError(
                f"cannot read campaign manifest provenance: {campaign_manifest}"
            ) from error
        git_metadata = payload.get("git") if isinstance(payload, dict) else None
        manifest_commit = git_metadata.get("commit") if isinstance(git_metadata, dict) else None
        if isinstance(manifest_commit, str) and manifest_commit.strip():
            candidates.add(manifest_commit.strip())
    for row in rows:
        source_commit = row.get("campaign_source_commit_sha")
        if isinstance(source_commit, str) and source_commit.strip():
            candidates.add(source_commit.strip())
    if len(candidates) > 1:
        raise AnalysisError(f"campaign source commit provenance disagrees: {sorted(candidates)}")
    return next(iter(candidates), None)


def _load_rows_from_directory(
    path: Path,
    *,
    campaign_status: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Load and normalize all deterministic ``episodes.jsonl`` files below a directory."""

    files = sorted(item for item in path.rglob("episodes.jsonl") if item.is_file())
    if not files:
        raise AnalysisError(f"episode rows directory has no episodes.jsonl files: {path}")
    rows: list[dict[str, Any]] = []
    for item in files:
        rows.extend(_load_jsonl_rows(item))
    return _normalise_loaded_rows(rows, campaign_status=campaign_status)


def _load_json_rows(
    path: Path,
    *,
    campaign_status: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Load and normalize a JSON list or a known wrapper object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AnalysisError(f"JSON episode-rows payload is not valid JSON: {path}") from error
    candidates: list[Any]
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = next(
            (
                candidate
                for key in ("rows", "episode_rows", "social_compliance_rows")
                if isinstance(candidate := payload.get(key), list)
            ),
            [],
        )
    else:
        candidates = []
    if not candidates or not all(isinstance(row, dict) for row in candidates):
        raise AnalysisError(f"JSON episode-rows payload at {path} is not a list of rows")
    return _normalise_loaded_rows(candidates, campaign_status=campaign_status)


def _load_rows_from_path(
    path: Path,
    *,
    campaign_status: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load flat rows or normalize raw JSONL episode rows from a file/tree."""

    if not path.exists():
        raise AnalysisError(f"episode rows file not found: {path}")
    if path.is_dir():
        return _load_rows_from_directory(path, campaign_status=campaign_status)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _normalise_loaded_rows(
            _load_jsonl_rows(path),
            campaign_status=campaign_status,
        )
    if suffix == ".json":
        return _load_json_rows(path, campaign_status=campaign_status)
    if suffix in {".csv", ".tsv"}:
        return _load_csv_rows(path)
    raise AnalysisError(f"unsupported episode-rows format: {path}")


def _embedded_manifest_rows(
    payload: dict[str, Any],
    *,
    campaign_status: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Return normalized rows embedded directly in a campaign manifest."""

    for key in ("social_compliance_rows", "episode_rows", "rows"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            if not all(isinstance(row, dict) for row in candidate):
                raise AnalysisError(f"campaign manifest field {key!r} contains non-object rows")
            return _normalise_loaded_rows(candidate, campaign_status=campaign_status)
    return None


def _pointed_manifest_rows(
    payload: dict[str, Any],
    manifest_path: Path,
    *,
    campaign_status: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Load rows from a supported relative or absolute manifest artifact pointer."""

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    for key in ("social_compliance_rows", "episode_rows", "seed_episode_rows_json"):
        pointer = artifacts.get(key)
        if not isinstance(pointer, str) or not pointer:
            continue
        pointer_path = Path(pointer)
        candidate_path = (
            pointer_path if pointer_path.is_absolute() else manifest_path.parent / pointer_path
        )
        if candidate_path.exists():
            return _load_rows_from_path(candidate_path, campaign_status=campaign_status)
    return None


def _extract_rows_from_manifest(
    manifest_path: Path,
    *,
    campaign_status: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Best-effort extraction of per-episode rows from a campaign manifest.

    The nominal campaign runner's exact per-episode artifact layout is defined
    by the preregistration sibling; this loader accepts the common shapes (an
    embedded ``rows`` / ``episode_rows`` list, or a sibling
    ``<stem>_social_compliance_rows.json`` next to the manifest) so the analysis
    is robust to the sibling's final choice.
    """

    if not manifest_path.is_file():
        raise AnalysisError(f"campaign manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AnalysisError(f"campaign manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise AnalysisError(f"campaign manifest is not a JSON object: {manifest_path}")
    embedded = _embedded_manifest_rows(payload, campaign_status=campaign_status)
    if embedded is not None:
        return embedded
    pointed = _pointed_manifest_rows(
        payload,
        manifest_path,
        campaign_status=campaign_status,
    )
    if pointed is not None:
        return pointed
    sibling = manifest_path.with_name(manifest_path.stem + "_social_compliance_rows.json")
    if sibling.exists():
        return _load_rows_from_path(sibling, campaign_status=campaign_status)
    raise AnalysisError(
        "campaign manifest does not expose per-episode social-compliance rows; "
        "pass them explicitly via --episode-rows",
    )


def load_episode_rows(
    campaign_manifest: Path | None,
    episode_rows: Path | None,
) -> list[dict[str, Any]]:
    """Load per-episode rows, preferring an explicit --episode-rows path."""

    campaign_status = _load_campaign_status_context(campaign_manifest)
    if episode_rows is not None:
        return _load_rows_from_path(episode_rows, campaign_status=campaign_status)
    if campaign_manifest is not None:
        return _extract_rows_from_manifest(campaign_manifest, campaign_status=campaign_status)
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

    frozen_parameters = {
        "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
        "permutation_samples": DEFAULT_PERMUTATION_SAMPLES,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "confidence": DEFAULT_CONFIDENCE,
        "alpha": DEFAULT_ALPHA,
    }
    for parameter, expected in frozen_parameters.items():
        if getattr(args, parameter) != expected:
            raise AnalysisError(
                f"the preregistered analysis requires {parameter}={expected!r}",
            )
    config_sha256 = _file_sha256(args.config)
    if config_sha256 != FROZEN_CONFIG_SHA256:
        raise AnalysisError(
            "campaign config is absent or does not match the preregistered frozen SHA-256",
        )


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
        support_count != 0
        or not isinstance(unavailable_reasons.get(metric_id), str)
        or not unavailable_reasons[metric_id].strip()
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
    reference_cells = planner_cells["goal"]
    expected_cell_count = len(FROZEN_SCENARIOS) * len(FROZEN_SEEDS)
    if len(reference_cells) != expected_cell_count:
        raise AnalysisError(
            "goal rows do not cover the frozen six-scenario by 30-seed matrix "
            f"(got {len(reference_cells)}, expected {expected_cell_count})",
        )
    if {scenario for scenario, _seed in reference_cells} != set(FROZEN_SCENARIOS):
        raise AnalysisError(
            "goal rows do not contain exactly the preregistered frozen scenarios",
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
    expected_planners = set(FROZEN_PLANNERS)
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
        "scenario_count": len(FROZEN_SCENARIOS),
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


def _paired_seed_blocks(cells: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    """Return per-seed paired-difference blocks in frozen seed order."""

    by_seed: dict[int, list[float]] = {}
    for cell in cells:
        by_seed.setdefault(cell["seed"], []).append(cell["difference"])
    return [np.asarray(by_seed[seed], dtype=np.float64) for seed in sorted(by_seed)]


def percentile_bootstrap_ci(
    seed_blocks: Sequence[np.ndarray],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap the mean paired difference by resampling whole seed blocks."""

    if not seed_blocks:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(seed_blocks)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        draw = rng.integers(0, n, size=n)
        means[i] = float(np.concatenate([seed_blocks[index] for index in draw]).mean())
    alpha = 1.0 - confidence
    low = float(np.percentile(means, 100.0 * alpha / 2.0))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def paired_permutation_pvalue(
    seed_blocks: Sequence[np.ndarray],
    *,
    samples: int,
    seed: int,
) -> float:
    """Two-sided blockwise sign-flip p-value for the mean paired difference."""

    if not seed_blocks:
        return float("nan")
    diffs = np.concatenate(seed_blocks)
    observed = float(np.abs(diffs.mean()))
    if not math.isfinite(observed):
        return float("nan")
    # Exact enumeration when feasible, otherwise Monte-Carlo sign-flipping.
    n = len(seed_blocks)
    if n <= 12:
        total = 0
        extreme = 0
        for bits in range(1 << n):
            signed = [
                block if (bits >> index) & 1 else -block for index, block in enumerate(seed_blocks)
            ]
            stat = abs(float(np.concatenate(signed).mean()))
            total += 1
            if stat >= observed - 1e-15:
                extreme += 1
        return (extreme + 1.0) / (total + 1.0)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(samples, n))
    weighted_sums = np.asarray([block.sum() for block in seed_blocks], dtype=np.float64)
    stats = np.abs((signs * weighted_sums).sum(axis=1) / diffs.size)
    extreme = int(np.count_nonzero(stats >= observed - 1e-15))
    return (extreme + 1.0) / (samples + 1.0)


def holm_multiplicity(p_values: Sequence[float], *, alpha: float) -> list[float]:
    """Holm-adjust finite p-values across the fixed declared decision family.

    Unsupported estimands carry ``NaN`` rather than a test result.  They remain
    in the declared family size but must not perturb the ranks of tested
    estimands merely because a metric was unavailable.
    """

    entries = sorted(
        (index for index, value in enumerate(p_values) if math.isfinite(float(value))),
        key=lambda index: float(p_values[index]),
    )
    m = len(p_values)
    adjusted = [float("nan")] * m
    running_max = 0.0
    for rank, original_index in enumerate(entries):
        raw = float(p_values[original_index])
        correction = min(1.0, raw * (m - rank))
        running_max = max(running_max, correction)
        adjusted[original_index] = running_max
    return adjusted


# --------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------


def _scenario_family(scenario_id: str) -> str:
    """Return the preregistered scenario family for a frozen scenario id."""

    return FROZEN_SCENARIO_FAMILIES[scenario_id]


def compute_decisions(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    bootstrap_seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    """Compute all planner-pair-by-metric-family paired-effect decisions."""

    decisions: list[dict[str, Any]] = []
    for reference_planner, comparison_planner in FROZEN_PLANNER_PAIRS:
        for metric_id in METRIC_IDS:
            family, units, denominator = METRIC_CONTRACT[metric_id]
            cells = build_matched_cells(rows, reference_planner, comparison_planner, metric_id)
            diffs = np.asarray([cell["difference"] for cell in cells], dtype=np.float64)
            seed_blocks = _paired_seed_blocks(cells)
            if diffs.size >= MIN_PAIRED_SUPPORT:
                mean_diff = float(diffs.mean())
                ci_low, ci_high = percentile_bootstrap_ci(
                    seed_blocks,
                    samples=bootstrap_samples,
                    confidence=confidence,
                    seed=bootstrap_seed,
                )
                raw_p = paired_permutation_pvalue(
                    seed_blocks,
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
                    "scenario_family": scenario,
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
        f"{', '.join(FROZEN_PLANNERS)} planner pairs. Effects are mean paired "
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
    lines.append(f"- analysis commit sha: `{provenance.get('commit_sha', 'unknown')}`")
    lines.append(
        f"- campaign source commit sha: `{provenance.get('campaign_source_commit_sha') or 'absent'}`",
    )
    lines.append(
        f"- campaign manifest: `{provenance.get('campaign_manifest', 'none (self-test)')}`"
    )
    lines.append(
        f"- campaign manifest sha256: `{provenance.get('campaign_manifest_sha256') or 'absent'}`",
    )
    lines.append(
        f"- episode rows: `{provenance.get('episode_rows', 'embedded or absent')}`",
    )
    lines.append(
        f"- episode rows sha256: `{provenance.get('episode_rows_sha256') or 'absent'}`",
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
        scenario_support = ", ".join(
            f"{item['scenario_family']}: n={item['n_paired']}, "
            f"mean diff={item['mean_difference']:.6g}"
            for item in decision["scenario_support"]
        )
        lines.append(
            f"- {decision['comparison_planner']} vs {decision['reference_planner']} / "
            f"{decision['metric_family']}: {scenario_support or 'no paired support'}.",
        )
        if decision["n_paired"] < MIN_PAIRED_SUPPORT:
            lines.append(
                f"  Diagnostic-only (n_paired={decision['n_paired']} < {MIN_PAIRED_SUPPORT}).",
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
                "campaign_source_commit_sha": provenance.get("campaign_source_commit_sha"),
                "episode_rows": provenance.get("episode_rows"),
                "episode_rows_sha256": provenance.get("episode_rows_sha256"),
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
    scenarios = list(FROZEN_SCENARIOS)
    seeds = sorted(FROZEN_SEEDS)
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
        bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
        permutation_samples=DEFAULT_PERMUTATION_SAMPLES,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        confidence=DEFAULT_CONFIDENCE,
    )
    apply_multiplicity(decisions, alpha=0.05)
    by_key = {
        (d["reference_planner"], d["comparison_planner"], d["metric_id"]): d for d in decisions
    }
    sf_comfort = by_key[("goal", "social_force", "comfort_exposure_person_s")]
    orca_comfort = by_key[("goal", "orca", "comfort_exposure_person_s")]
    sf_orca_comfort = by_key[("social_force", "orca", "comfort_exposure_person_s")]
    assert len(decisions) == len(FROZEN_PLANNER_PAIRS) * len(METRIC_IDS)
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
    assert sf_orca_comfort["n_paired"] == 180, sf_orca_comfort["n_paired"]
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
        ([{**rows[0], "scenario_id": "unfrozen_scenario"}, *rows[1:]], "unfrozen scenario"),
        (
            [{**rows[0], "schema_valid": False}, *rows[1:]],
            "schema-invalid social-compliance block",
        ),
        (
            [
                {
                    **rows[0],
                    "unavailable_reasons": {
                        **rows[0]["unavailable_reasons"],
                        "pedestrian_deviation_mean_m": " ",
                    },
                },
                *rows[1:],
            ],
            "blank unavailable reason",
        ),
    ):
        try:
            validate_rows(malformed_rows)
        except AnalysisError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError(f"{reason} was not rejected by validate_rows")
    # Accepted-unavailable-only gate (exit 3): a campaign whose rows are all
    # benchmark-capable but whose every metric family is unavailable must reach
    # no supported decision, so the run is diagnostic-only rather than evidence.
    all_unavailable_rows = [
        _synthetic_row(
            planner,
            scenario,
            seed,
            {},
            unavailable=tuple(METRIC_CONTRACT),
            execution_mode="native" if planner == "goal" else "adapter",
        )
        for scenario in scenarios
        for seed in seeds
        for planner in FROZEN_PLANNERS
    ]
    unavailable_valid_rows, unavailable_report = validate_rows(all_unavailable_rows)
    assert unavailable_report["rejected"] == [], "all-unavailable rows must stay benchmark-capable"
    unavailable_decisions = compute_decisions(
        unavailable_valid_rows,
        bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
        permutation_samples=DEFAULT_PERMUTATION_SAMPLES,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        confidence=DEFAULT_CONFIDENCE,
    )
    apply_multiplicity(unavailable_decisions, alpha=DEFAULT_ALPHA)
    assert len(unavailable_decisions) == len(FROZEN_PLANNER_PAIRS) * len(METRIC_IDS)
    assert not any(
        decision["n_paired"] >= MIN_PAIRED_SUPPORT for decision in unavailable_decisions
    ), "all-unavailable campaign must not reach the exit-3 support gate"
    assert not any(decision["rejected_at_alpha"] for decision in unavailable_decisions)
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
        "planner_pairs": [list(pair) for pair in FROZEN_PLANNER_PAIRS],
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
        "campaign_source_commit_sha": _campaign_source_commit(args.campaign_manifest, rows),
        "campaign_manifest": str(args.campaign_manifest) if args.campaign_manifest else None,
        "campaign_manifest_sha256": _file_sha256(args.campaign_manifest)
        if args.campaign_manifest
        else None,
        "episode_rows": str(args.episode_rows) if args.episode_rows else None,
        "episode_rows_sha256": _path_sha256(args.episode_rows),
        "planner_pairs": [list(pair) for pair in FROZEN_PLANNER_PAIRS],
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
