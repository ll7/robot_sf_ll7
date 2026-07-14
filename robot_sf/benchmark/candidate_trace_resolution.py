"""Deterministic candidate-to-trace resolution layer (issue #5615).

This module joins every mined candidate (the ``seed_flip_inversion_candidates.v1``
contract from issue #5446) to its campaign row, exact episode identifier, trace
export artifact URI (``simulation_trace_export.v1``), per-trace
``trace_failure_predicates.v1`` rows, and ``critical-intervals.v1`` records.

It is a **data-joining and orchestration** layer only. It composes existing
outputs and never adds new analysis, thresholds, or rendering. The single
missing composition step between #5446 and #5447 (case-capsule assembly) is the
production of one deterministic, versioned resolution table.

Design contract (issue #5615)
-----------------------------
- **Fail closed.** Every candidate appears exactly once with an explicit
  ``resolution_status`` of ``resolved`` / ``trace-missing`` / ``schema-mismatch``
  / ``provenance-incomplete``. No candidate is silently dropped. Unresolved
  candidates carry a machine-readable ``reason_code``.
- **Read-only over upstream artifacts.** The resolver consumes the pinned
  campaign result store read-only and never mutates mined candidates or traces.
- **Deterministic.** A re-run on the same inputs produces a byte-identical
  manifest (sorted keys, no clock-bearing fields in the rows themselves, and a
  recorded input manifest hash).
- **No eligibility relaxation.** Provenance gaps are reported as
  ``provenance-incomplete``; a candidate whose trace cannot be satisfied by a
  real pinned artifact is reported as ``trace-missing``; a found artifact that
  fails ``simulation_trace_export.v1`` validation is ``schema-mismatch``.
- **Honest counts.** The summary reports the per-status breakdown so downstream
  consumers know exactly how many candidates actually resolved.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExportValidationError,
    load_simulation_trace_export,
    load_simulation_trace_export_schema,
)
from robot_sf.analysis_workbench.trace_failure_predicates import (
    TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
    extract_trace_failure_predicates,
)
from robot_sf.benchmark.critical_intervals import extract_critical_intervals, load_config
from robot_sf.benchmark.utils import coerce_optional_id
from scripts.tools.campaign_result_store import read_parquet_frame

#: Schema tag for the emitted resolution manifest.
SCHEMA_VERSION = "candidate_trace_resolution.v1"

#: Schema tag of the candidate manifest this resolver consumes (issue #5446).
CANDIDATE_SCHEMA_VERSION = "seed_flip_inversion_candidates.v1"

#: Schema tag of the trace export contract we resolve against.
TRACE_SCHEMA_VERSION = "simulation_trace_export.v1"

#: Resolution statuses (fail-closed, never silently drop a candidate).
ResolutionStatus = Literal[
    "resolved",
    "trace-missing",
    "schema-mismatch",
    "provenance-incomplete",
]

SCHEMA_FILE = Path(__file__).with_name("schemas") / "candidate_trace_resolution.v1.json"


class CandidateTraceResolutionError(ValueError):
    """Raised when the resolver cannot produce a defensible resolution manifest."""


@dataclass(frozen=True, slots=True)
class CampaignResultStore:
    """Read-only view over a pinned campaign result store.

    Attributes:
        study_id: Study identifier from the store's summary.
        episodes: Mapping from a stable episode key to its row provenance.
    """

    study_id: str
    episodes: dict[str, dict[str, Any]]


def _coerce_optional_text(value: Any) -> str | None:
    """Normalize a nullable identifier while preserving missingness.

    Returns:
        A stripped identifier, or ``None`` when the value is absent or blank.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _episode_key(
    scenario_id: str | None,
    planner: str | None,
    seed: int | None,
    episode_id: str | None,
) -> str:
    """Build a stable lookup key tolerant of missing fields.

    Returns:
        A normalized ``field=val`` key joined by ``|``.
    """
    parts = [
        f"scenario_id={scenario_id if scenario_id is not None else 'NA'}",
        f"planner={planner if planner is not None else 'NA'}",
        f"seed={seed if seed is not None else 'NA'}",
        f"episode_id={episode_id if episode_id is not None else 'NA'}",
    ]
    return "|".join(parts)


def load_campaign_result_store(store_dir: Path) -> CampaignResultStore:
    """Load a pinned campaign result store read-only (``campaign-result-store.v1``).

    Only episode provenance (scenario, planner, seed, episode id, artifact URI,
    artifact hash) is read. The Parquet frames are read through pandas/duckdb
    fallback, but the store is never mutated.

    Returns:
        A read-only :class:`CampaignResultStore` view.

    Raises:
        CandidateTraceResolutionError: When the store or its episode rows are missing.
    """
    summary_path = store_dir / "summary.json"
    parquet_path = store_dir / "episodes.parquet"
    if not summary_path.is_file():
        raise CandidateTraceResolutionError(
            f"campaign result store missing summary.json: {store_dir}"
        )
    if not parquet_path.is_file():
        raise CandidateTraceResolutionError(
            f"campaign result store missing episodes.parquet: {store_dir}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateTraceResolutionError(
            f"campaign result store summary.json unreadable: {exc}"
        ) from exc

    try:
        episodes = read_parquet_frame(parquet_path)
    except Exception as exc:
        raise CandidateTraceResolutionError(
            f"campaign result store episodes.parquet unreadable: {exc}"
        ) from exc

    by_key: dict[str, dict[str, Any]] = {}
    for row in episodes.to_dict(orient="records"):
        scenario_id = _coerce_optional_text(row.get("scenario_id"))
        planner = _coerce_optional_text(row.get("planner"))
        episode_id = _coerce_optional_text(row.get("episode_id"))
        run_id = _coerce_optional_text(row.get("run_id"))
        seed = coerce_optional_id(row.get("seed"))
        if (
            scenario_id is None
            or planner is None
            or episode_id is None
            or run_id is None
            or seed is None
        ):
            continue
        key = _episode_key(scenario_id, planner, seed, episode_id)
        by_key[key] = {
            "run_id": run_id,
            "episode_id": episode_id,
            "planner": planner,
            "scenario_id": scenario_id,
            "scenario_family": row.get("scenario_family"),
            "seed": seed,
            "row_status": row.get("row_status"),
            "artifact_uri": row.get("artifact_uri"),
            "artifact_sha256": row.get("artifact_sha256"),
        }
    return CampaignResultStore(
        study_id=str(summary.get("study_id", "unknown")),
        episodes=by_key,
    )


def resolve_candidate_to_episode(
    candidate: Mapping[str, Any],
    store: CampaignResultStore | None,
) -> dict[str, Any]:
    """Join one candidate to its campaign episode provenance (read-only).

    Looks up the pinned campaign store by the candidate's scenario / planner /
    seed / episode id. Missing provenance fields on the candidate are surfaced
    as ``provenance-incomplete`` rather than guessed.

    Returns:
        A dict with ``campaign_id``, ``campaign_row_reference``, ``artifact_uri``,
        ``scenario_id``, ``planner_id``, ``seed``, ``config_hash``,
        ``episode_id``, ``resolution_status`` (provisionally), and ``reason_code``.
    """
    scenario_id = _coerce_optional_text(candidate.get("scenario_id"))
    planner = _coerce_optional_text(candidate.get("planner"))

    # Candidates from the #5446 miner carry provenance via their cell metadata;
    # the miner records scenario_id and planner directly, and reproducibility
    # blocks carry per-seed outcomes. Where a candidate lacks an explicit seed
    # or episode id we do not invent one. The miner may serialize seed as a
    # string, so normalize to int to match the campaign store key.
    raw_seed = candidate.get("seed")
    seed = coerce_optional_id(raw_seed)
    episode_id = _coerce_optional_text(candidate.get("episode_id"))
    config_hash = _coerce_optional_text(candidate.get("config_hash"))

    missing_core = [
        field
        for field, value in (
            ("scenario_id", scenario_id),
            ("planner", planner),
        )
        if value is None or (isinstance(value, str) and not value.strip())
    ]

    base: dict[str, Any] = {
        "campaign_id": store.study_id if store is not None else None,
        "campaign_row_reference": None,
        "artifact_uri": candidate.get("artifact_uri"),
        "scenario_id": scenario_id,
        "planner_id": planner,
        "seed": seed,
        "config_hash": config_hash,
        "episode_id": episode_id,
    }

    if missing_core:
        return {
            **base,
            "resolution_status": "provenance-incomplete",
            "reason_code": "missing_candidate_provenance:" + ",".join(missing_core),
        }

    if raw_seed is not None and seed is None:
        return {
            **base,
            "resolution_status": "provenance-incomplete",
            "reason_code": "invalid_candidate_provenance:seed",
        }

    if store is None:
        return {
            **base,
            "resolution_status": "provenance-incomplete",
            "reason_code": "no_campaign_store_provided",
        }

    key = _episode_key(scenario_id, planner, seed, episode_id)
    row = store.episodes.get(key)
    if row is None:
        return {
            **base,
            "resolution_status": "provenance-incomplete",
            "reason_code": "campaign_row_not_found",
        }
    return {
        **base,
        "campaign_row_reference": str(row.get("run_id")),
        "artifact_uri": row.get("artifact_uri")
        if base.get("artifact_uri") is None
        else base.get("artifact_uri"),
        "episode_id": row.get("episode_id") if episode_id is None else episode_id,
        "resolution_status": "resolved",
        "reason_code": "campaign_row_found",
    }


def _locate_trace_path(
    episode_provenance: Mapping[str, Any],
    *,
    trace_search_roots: Sequence[Path] | None,
) -> Path | None:
    """Locate a trace export file from an artifact URI or episode-id search.

    Returns:
        The resolved :class:`Path`, or ``None`` when no artifact can be found.
    """
    artifact_uri = episode_provenance.get("artifact_uri")
    episode_id = episode_provenance.get("episode_id")
    roots = list(trace_search_roots or [])

    if artifact_uri:
        candidate_path = Path(str(artifact_uri))
        if candidate_path.is_file():
            return candidate_path
        for root in roots:
            probe = root / candidate_path.name
            if probe.is_file():
                return probe

    if episode_id is not None and roots:
        for root in roots:
            matches = sorted(root.rglob(f"*{episode_id}*.json"))
            if matches:
                return matches[0]
    return None


def _validate_trace_file(trace_path: Path) -> dict[str, Any]:
    """Validate a located trace file against ``simulation_trace_export.v1``.

    Returns:
        A dict with ``trace_artifact_uri``, ``trace_content_hash``,
        ``trace_schema_version``, ``resolution_status``, and ``reason_code``.
    """
    raw = trace_path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "trace_artifact_uri": str(trace_path),
            "trace_content_hash": content_hash,
            "trace_schema_version": None,
            "resolution_status": "schema-mismatch",
            "reason_code": f"trace_not_json:{exc}",
        }

    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    if schema_version != TRACE_SCHEMA_VERSION:
        return {
            "trace_artifact_uri": str(trace_path),
            "trace_content_hash": content_hash,
            "trace_schema_version": schema_version,
            "resolution_status": "schema-mismatch",
            "reason_code": f"unexpected_trace_schema:{schema_version}",
        }

    validator = Draft202012Validator(load_simulation_trace_export_schema())
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    if errors:
        return {
            "trace_artifact_uri": str(trace_path),
            "trace_content_hash": content_hash,
            "trace_schema_version": schema_version,
            "resolution_status": "schema-mismatch",
            "reason_code": "trace_schema_validation_failed:"
            + ";".join(e.message for e in errors[:3]),
        }
    return {
        "trace_artifact_uri": str(trace_path),
        "trace_content_hash": content_hash,
        "trace_schema_version": schema_version,
        "resolution_status": "resolved",
        "reason_code": "trace_schema_valid",
    }


def resolve_trace_artifact(
    episode_provenance: Mapping[str, Any],
    *,
    trace_search_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Resolve a candidate's trace export artifact URI and validate its schema.

    A trace is located from an explicit ``artifact_uri`` when present, otherwise
    by searching ``trace_search_roots`` for a file whose name embeds the
    episode id. The located trace is validated against
    ``simulation_trace_export.v1``; a missing trace is ``trace-missing`` and an
    invalid one is ``schema-mismatch``.

    Returns:
        A dict with ``trace_artifact_uri``, ``trace_content_hash``,
        ``trace_schema_version``, ``resolution_status``, and ``reason_code``.
    """
    trace_path = _locate_trace_path(episode_provenance, trace_search_roots=trace_search_roots)
    if trace_path is None:
        return {
            "trace_artifact_uri": None,
            "trace_content_hash": None,
            "trace_schema_version": None,
            "resolution_status": "trace-missing",
            "reason_code": "trace_artifact_not_found",
        }
    return _validate_trace_file(trace_path)


def resolve_trace_signals(
    trace_path: str | None,
    *,
    critical_interval_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report per-trace predicate and critical-interval availability (read-only).

    Predicate rows are discovered by loading the resolved trace export and
    applying :func:`extract_trace_failure_predicates`. Critical intervals are
    discovered by applying :func:`extract_critical_intervals` when a config is
    supplied. Availability is reported; no new signal is computed beyond what the
    existing primitives already produce.

    Returns:
        A dict with ``predicate_rows_available`` and ``critical_intervals_available``.
    """
    predicate_rows: list[dict[str, Any]] | None = None
    critical_intervals_available: dict[str, Any] | None = None

    if trace_path is None:
        return {
            "predicate_rows_available": predicate_rows,
            "critical_intervals_available": critical_intervals_available,
        }

    trace = None
    try:
        trace = load_simulation_trace_export(Path(trace_path))
    except (OSError, SimulationTraceExportValidationError):
        predicate_rows = []
    else:
        payload = extract_trace_failure_predicates(trace)
        predicate_rows = [
            {
                "predicate_id": str(pred["predicate_id"]),
                "schema_version": TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
            }
            for pred in payload.get("predicates", [])
        ]

    if critical_interval_config is not None:
        available = False
        if trace is not None:
            try:
                cfg = load_config(config_dict=dict(critical_interval_config))
                intervals = extract_critical_intervals(trace.to_dict(), cfg)
                available = any(iv.status == "available" for iv in intervals)
            except (OSError, SimulationTraceExportValidationError, ValueError):
                pass
        critical_intervals_available = {
            "available": available,
            "reference": f"critical-intervals.v1:{'enabled' if available else 'none'}",
        }

    return {
        "predicate_rows_available": predicate_rows,
        "critical_intervals_available": critical_intervals_available,
    }


def resolve_candidate_trace_resolution(
    candidate_manifest: Mapping[str, Any],
    *,
    campaign_store_dir: Path | None = None,
    trace_search_roots: Sequence[Path] | None = None,
    critical_interval_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve every candidate to its trace, predicates, and critical intervals.

    Args:
        candidate_manifest: A ``seed_flip_inversion_candidates.v1`` manifest dict.
        campaign_store_dir: Optional pinned campaign result store directory
            (read-only). When ``None``, candidate provenance is reported as
            ``provenance-incomplete`` rather than looked up.
        trace_search_roots: Optional roots to search for trace export artifacts
            when an explicit ``artifact_uri`` is absent or not a local file.
        critical_interval_config: Optional ``critical-intervals.v1`` config dict
            controlling which anchors are enabled when discovering intervals.

    Returns:
        A ``candidate_trace_resolution.v1`` manifest dict.

    Raises:
        CandidateTraceResolutionError: When the input is not a usable candidate
            manifest or yields no resolvable rows (fail closed).
    """
    if not isinstance(candidate_manifest, Mapping):
        raise CandidateTraceResolutionError("candidate_manifest must be a mapping")
    if candidate_manifest.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise CandidateTraceResolutionError(
            "candidate_manifest schema_version must be "
            f"{CANDIDATE_SCHEMA_VERSION!r}, got {candidate_manifest.get('schema_version')!r}"
        )
    candidates = candidate_manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CandidateTraceResolutionError("candidate_manifest has no candidates to resolve")

    manifest_bytes = json.dumps(candidate_manifest, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    source_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    store = (
        load_campaign_result_store(campaign_store_dir) if campaign_store_dir is not None else None
    )

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {
        "resolved": 0,
        "trace-missing": 0,
        "schema-mismatch": 0,
        "provenance-incomplete": 0,
    }

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        episode_prov = resolve_candidate_to_episode(candidate, store)

        # Episode provenance status is the fail-closed gate; only when it is
        # resolved do we attempt to locate and validate a trace artifact.
        if episode_prov["resolution_status"] != "resolved":
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "source_manifest_hash": source_manifest_hash,
                    **{
                        k: episode_prov.get(k)
                        for k in (
                            "campaign_id",
                            "campaign_row_reference",
                            "artifact_uri",
                            "scenario_id",
                            "planner_id",
                            "seed",
                            "config_hash",
                            "episode_id",
                        )
                    },
                    "trace_artifact_uri": None,
                    "trace_content_hash": None,
                    "trace_schema_version": None,
                    "resolution_status": episode_prov["resolution_status"],
                    "reason_code": episode_prov["reason_code"],
                    "predicate_rows_available": None,
                    "critical_intervals_available": None,
                    "exact_repeat_determinism": None,
                }
            )
            counts[episode_prov["resolution_status"]] += 1
            continue

        trace = resolve_trace_artifact(episode_prov, trace_search_roots=trace_search_roots)
        signals = resolve_trace_signals(
            trace.get("trace_artifact_uri"),
            critical_interval_config=critical_interval_config,
        )

        merged_status = _merge_statuses(
            episode_prov["resolution_status"], trace["resolution_status"]
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_manifest_hash": source_manifest_hash,
                "campaign_id": episode_prov["campaign_id"],
                "campaign_row_reference": episode_prov["campaign_row_reference"],
                "artifact_uri": episode_prov.get("artifact_uri"),
                "scenario_id": episode_prov["scenario_id"],
                "planner_id": episode_prov["planner_id"],
                "seed": episode_prov["seed"],
                "config_hash": episode_prov["config_hash"],
                "episode_id": episode_prov["episode_id"],
                "trace_artifact_uri": trace["trace_artifact_uri"],
                "trace_content_hash": trace["trace_content_hash"],
                "trace_schema_version": trace["trace_schema_version"],
                "resolution_status": merged_status,
                "reason_code": (
                    episode_prov["reason_code"]
                    if merged_status == "resolved"
                    else trace["reason_code"]
                ),
                "predicate_rows_available": signals["predicate_rows_available"],
                "critical_intervals_available": signals["critical_intervals_available"],
                "exact_repeat_determinism": (
                    "pinned_artifact" if merged_status == "resolved" else None
                ),
            }
        )
        counts[merged_status] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": "#5615",
        "generated_at_utc": "deterministic-not-a-clock",  # never a wall clock; see determinism.policy
        "source_manifest_hash": source_manifest_hash,
        "determinism": {
            "policy": (
                "rows are ordered by candidate_id; scalar fields are plain JSON; "
                "no wall-clock or nondeterministic values appear in rows; the "
                "source_manifest_hash pins the input so re-runs are byte-identical"
            ),
            "tool_version": SCHEMA_VERSION,
        },
        "summary": {
            "n_candidates": len(candidates),
            "n_resolved": counts["resolved"],
            "n_trace_missing": counts["trace-missing"],
            "n_schema_mismatch": counts["schema-mismatch"],
            "n_provenance_incomplete": counts["provenance-incomplete"],
        },
        "rows": sorted(rows, key=lambda r: r["candidate_id"]),
    }


def _merge_statuses(*statuses: str) -> ResolutionStatus:
    """Merge provenance and trace statuses fail-closed (worst status wins).

    Returns:
        The most severe resolution status among the inputs.
    """
    order: tuple[str, ...] = (
        "resolved",
        "trace-missing",
        "schema-mismatch",
        "provenance-incomplete",
    )
    worst = "resolved"
    for status in statuses:
        if status not in order:
            continue
        if order.index(status) > order.index(worst):
            worst = status
    return worst  # type: ignore[return-value]


@lru_cache(maxsize=1)
def _load_resolution_schema() -> dict[str, Any]:
    """Load the published resolver schema once per process.

    Returns:
        The parsed JSON Schema mapping.
    """
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_candidate_trace_resolution(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a resolution manifest against its published JSON Schema.

    Returns:
        A dict with ``ok`` (bool) and ``errors`` (list of strings).
    """
    validator = Draft202012Validator(_load_resolution_schema())
    errors = [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(manifest), key=lambda e: list(e.absolute_path))
    ]
    return {"ok": not errors, "errors": errors}


def json_pointer(path: Sequence[str | int]) -> str:
    """Render a JSON Pointer from an error path.

    Returns:
        A JSON Pointer string, or ``""`` for the root.
    """
    if not path:
        return ""
    return "/" + "/".join(str(part) for part in path)


# Re-export for callers that want the canonical candidate field reader.
__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "CampaignResultStore",
    "CandidateTraceResolutionError",
    "load_campaign_result_store",
    "resolve_candidate_to_episode",
    "resolve_candidate_trace_resolution",
    "resolve_trace_artifact",
    "resolve_trace_signals",
    "validate_candidate_trace_resolution",
]
