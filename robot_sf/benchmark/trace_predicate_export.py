"""Trace-level predicate export lane (issue #5593).

In-sim trace-level safety-predicate producers already exist and are fixture-tested
(``robot_sf/benchmark/safety_predicates.py``). What this module adds is a **dedicated
export lane**: a versioned, queryable artifact that lets downstream consumers pull the
full set of emitted predicate values per episode/scenario/planner without re-deriving
them from raw per-step traces or writing a one-off analysis script each time.

The lane reads a completed campaign's trace bundles (``episodes.jsonl`` records that
already carry a ``safety_predicates`` block and an ``event_ledger``) and emits:

* a structured **export** (JSON-lines, one row per episode) joining each predicate
  record with scenario/planner/seed/run metadata;
* a **manifest** listing which predicate types are present, their schema versions,
  and every episode with a missing or degraded predicate field;
* a **coverage report** enumerating, per release, exported-vs-motivated predicates so a
  reviewer can check in one place whether a given predicate was actually *measured* for
  the release they are citing.

The export **fails closed** on missing/degraded required fields: it never silently
substitutes defaults. A missing predicate record or a degraded ``status`` is recorded in
the manifest and surfaced in the export as an explicit gap marker, consistent with the
existing ``missing trace fields fail closed`` rule in the predeclared-matrix contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.identity.hash_utils import read_jsonl as _read_jsonl
from robot_sf.benchmark.safety_predicates import (
    LATE_EVASIVE_PREDICATE_SCHEMA,
    OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA,
    OSCILLATORY_PREDICATE_SCHEMA,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

TRACE_PREDICATE_EXPORT_SCHEMA_VERSION = "trace_predicate_export.v1"
TRACE_PREDICATE_MANIFEST_SCHEMA_VERSION = "trace_predicate_manifest.v1"
TRACE_PREDICATE_COVERAGE_SCHEMA_VERSION = "trace_predicate_coverage.v1"
# Envelope schema the downstream scenario-evidence crosswalk (#5602) consumes. The
# crosswalk expects a single object with this schema version and a ``rows`` list whose
# per-row ``predicates`` is a *list* of ``{predicate, schema_version, status}`` records;
# the raw export lane emits per-row ``predicates`` as a *dict* keyed by predicate name, so
# a real export cannot flow through the crosswalk without the adapter below.
CROSSWALK_PREDICATE_EXPORT_SCHEMA_VERSION = "trace_predicate_export.v1"

# Predicates motivated by the taxonomy and produced by the shipped producers in
# robot_sf/benchmark/safety_predicates.py. ``record_key`` is the key under which the
# producer record appears in an episode's ``safety_predicates`` block.
MOTIVATED_TRACE_PREDICATES: tuple[dict[str, str], ...] = (
    {
        "predicate": "oscillatory_control",
        "record_key": "oscillatory_control_predicate",
        "schema_version": OSCILLATORY_PREDICATE_SCHEMA,
    },
    {
        "predicate": "late_evasive",
        "record_key": "late_evasive_predicate",
        "schema_version": LATE_EVASIVE_PREDICATE_SCHEMA,
    },
    {
        "predicate": "occlusion_near_miss",
        "record_key": "occlusion_near_miss_predicate",
        "schema_version": OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA,
    },
)
_SUPPORTED_PREDICATE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "oscillatory_control": (OSCILLATORY_PREDICATE_SCHEMA,),
    # Existing durable campaign bundles contain the v1 late-evasive record; retain its
    # provenance while accepting the current producer schema for new campaigns.
    "late_evasive": ("safety_predicate.late_evasive.v1", LATE_EVASIVE_PREDICATE_SCHEMA),
    "occlusion_near_miss": (OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA,),
}

EXPORT_STATUS_EXPORTED = "exported"
EXPORT_STATUS_MISSING = "missing"
EXPORT_STATUS_DEGRADED = "degraded"

# Degraded statuses a present predicate record may legitimately carry (fail closed:
# surfaced in the manifest, never silently treated as a clean measurement).
DEGRADED_PREDICATE_STATUSES = frozenset({"not_applicable", "unavailable"})
_PREDICATE_STATUSES = frozenset({"true", "false", *DEGRADED_PREDICATE_STATUSES})
_PREDICATE_FLAGS = {
    "oscillatory_control": "oscillation",
    "late_evasive": "late_evasive",
    "occlusion_near_miss": "occlusion_near_miss",
}


class TracePredicateExportError(ValueError):
    """Raised when the export contract cannot be satisfied for an input bundle."""


def _episode_metadata(record: Mapping[str, Any], *, run_id: str | None) -> dict[str, Any]:
    """Return the scenario/planner/seed/run metadata for one episode record."""

    def required_text(keys: tuple[str, ...], label: str) -> str:
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TracePredicateExportError(
                    f"episode field {label} must be a string, got {type(value).__name__}: {value!r}"
                )
            value = value.strip()
            if value == "":
                raise TracePredicateExportError(f"episode field {label} must be a non-empty string")
            return value
        raise TracePredicateExportError(f"missing required episode field: {label}")

    seed = record.get("seed")
    if isinstance(seed, bool) or seed is None:
        raise TracePredicateExportError("missing required episode field: seed")
    try:
        normalized_seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise TracePredicateExportError(f"episode seed must be an integer, got {seed!r}") from exc
    if isinstance(seed, float) and not seed.is_integer():
        raise TracePredicateExportError(f"episode seed must be an integer, got {seed!r}")

    return {
        "scenario_id": required_text(("scenario_id",), "scenario_id"),
        "seed": normalized_seed,
        "planner_id": required_text(("algo", "planner", "planner_id"), "planner"),
        "run_id": run_id or "unknown_run",
        "episode_id": required_text(("episode_id",), "episode_id"),
        "software_commit": (
            str(record["git_hash"]) if record.get("git_hash") is not None else None
        ),
    }


def _status_violations(predicate_record: Mapping[str, Any], *, predicate: str) -> list[str]:
    """Return fail-closed status violations for one predicate record."""

    status = predicate_record.get("status")
    if predicate == "occlusion_near_miss" and status is None:
        return ["degraded predicate occlusion_near_miss must carry a status"]
    if status is None:
        return []
    if not isinstance(status, str) or status not in _PREDICATE_STATUSES:
        return [f"predicate {predicate} has invalid status: {status!r}"]
    if status in DEGRADED_PREDICATE_STATUSES:
        reason = predicate_record.get("status_reason")
        if not isinstance(reason, str) or not reason.strip():
            return [f"degraded predicate {predicate} must carry a non-empty status_reason"]
    return []


def _predicate_record_violations(
    predicate_record: Mapping[str, Any], *, predicate: str, schema_versions: Sequence[str]
) -> list[str]:
    """Return fail-closed violations for one producer predicate record."""

    violations: list[str] = []
    if predicate_record.get("predicate") != predicate:
        violations.append(
            f"predicate {predicate} must declare predicate={predicate!r}, "
            f"got {predicate_record.get('predicate')!r}"
        )
    if predicate_record.get("schema_version") not in schema_versions:
        violations.append(
            f"predicate {predicate} must declare one of schema_versions={tuple(schema_versions)!r}, "
            f"got {predicate_record.get('schema_version')!r}"
        )
    evidence_kind = predicate_record.get("evidence_kind")
    if evidence_kind is not None and (
        not isinstance(evidence_kind, str) or not evidence_kind.strip()
    ):
        violations.append(f"predicate {predicate} evidence_kind must be a non-empty string")
    for field_name in ("fields", "thresholds"):
        if not isinstance(predicate_record.get(field_name), Mapping):
            violations.append(f"predicate {predicate} must carry a mapping for {field_name}")
    flag_name = _PREDICATE_FLAGS[predicate]
    if not isinstance(predicate_record.get(flag_name), bool):
        violations.append(f"predicate {predicate} must carry boolean field {flag_name}")

    violations.extend(_status_violations(predicate_record, predicate=predicate))
    return violations


def _export_predicate_block(
    metadata: Mapping[str, Any],
    record_key: str,
    schema_version: str,
    safety_predicates: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one predicate block for an episode, failing closed on gaps."""

    predicate_record = safety_predicates.get(record_key)
    predicate = record_key.removesuffix("_predicate")
    if record_key not in safety_predicates:
        return {
            **metadata,
            "predicate": predicate,
            "schema_version": schema_version,
            "export_status": EXPORT_STATUS_MISSING,
            "reason": "predicate_record_absent",
        }
    if not isinstance(predicate_record, Mapping):
        raise TracePredicateExportError(
            f"predicate {predicate} record must be a mapping, got {type(predicate_record).__name__}"
        )
    violations = _predicate_record_violations(
        predicate_record,
        predicate=predicate,
        schema_versions=_SUPPORTED_PREDICATE_SCHEMAS[predicate],
    )
    if violations:
        raise TracePredicateExportError("; ".join(violations))

    status = predicate_record.get("status")
    export_status = (
        EXPORT_STATUS_DEGRADED if status in DEGRADED_PREDICATE_STATUSES else EXPORT_STATUS_EXPORTED
    )
    block: dict[str, Any] = {
        **metadata,
        "predicate": predicate_record["predicate"],
        "schema_version": predicate_record["schema_version"],
        "export_status": export_status,
        "fields": predicate_record["fields"],
        "thresholds": predicate_record["thresholds"],
    }
    if "evidence_kind" in predicate_record:
        block["evidence_kind"] = predicate_record["evidence_kind"]
    # Preserve the diagnostic boolean and any fail-closed status reason.
    for flag in ("oscillation", "late_evasive", "occlusion_near_miss"):
        if flag in predicate_record:
            block[flag] = predicate_record[flag]
    if status is not None:
        block["status"] = status
        block["status_reason"] = predicate_record.get("status_reason")
    return block


def build_trace_predicate_export(
    episodes: Sequence[Mapping[str, Any]],
    *,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Build the structured per-episode predicate export rows.

    Args:
        episodes: Campaign episode records carrying a ``safety_predicates`` block.
        run_id: Optional run identifier (e.g. planner__kinematics dir name).

    Returns:
        List of export rows, one per episode, deterministically ordered by
        (run_id, scenario_id, seed, episode_id). Each row carries the episode metadata
        plus one block per motivated predicate; missing or degraded predicates are
        emitted as explicit gap markers rather than dropped.
    """

    rows: list[dict[str, Any]] = []
    for record in episodes:
        if not isinstance(record, Mapping):
            raise TracePredicateExportError("each episode record must be a mapping")
        metadata = _episode_metadata(record, run_id=run_id)
        safety_predicates = record.get("safety_predicates")
        safety_predicates = safety_predicates if isinstance(safety_predicates, Mapping) else {}
        row: dict[str, Any] = {
            "scenario_id": metadata["scenario_id"],
            "seed": metadata["seed"],
            "planner_id": metadata["planner_id"],
            "run_id": metadata["run_id"],
            "episode_id": metadata["episode_id"],
            "software_commit": metadata["software_commit"],
            "predicates": {},
        }
        for spec in MOTIVATED_TRACE_PREDICATES:
            block = _export_predicate_block(
                metadata, spec["record_key"], spec["schema_version"], safety_predicates
            )
            row["predicates"][spec["predicate"]] = block
        rows.append(row)

    rows.sort(key=lambda r: (r["run_id"], r["scenario_id"], str(r["seed"]), r["episode_id"]))
    return rows


def build_trace_predicate_manifest(
    *,
    export_rows: Sequence[Mapping[str, Any]],
    sources: Sequence[str],
    release: str,
) -> dict[str, Any]:
    """Build the export manifest listing predicate presence and every gap.

    Args:
        export_rows: Rows produced by :func:`build_trace_predicate_export`.
        sources: Campaign bundle source paths that were read.
        release: Release/run label this export was produced for.

    Returns:
        Manifest with per-predicate presence counts, every missing/degraded gap, and an
        overall ``complete`` flag that is ``False`` whenever any gap exists.
    """

    predicate_types: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for spec in MOTIVATED_TRACE_PREDICATES:
        predicate = spec["predicate"]
        present = 0
        degraded = 0
        observed_schema_versions: set[str] = set()
        for row in export_rows:
            block = row["predicates"][predicate]
            if block["export_status"] == EXPORT_STATUS_EXPORTED:
                present += 1
                if isinstance(block.get("schema_version"), str):
                    observed_schema_versions.add(block["schema_version"])
            elif block["export_status"] == EXPORT_STATUS_DEGRADED:
                degraded += 1
                if isinstance(block.get("schema_version"), str):
                    observed_schema_versions.add(block["schema_version"])
                gaps.append(
                    {
                        "scenario_id": row["scenario_id"],
                        "seed": row["seed"],
                        "planner_id": row["planner_id"],
                        "run_id": row["run_id"],
                        "episode_id": row["episode_id"],
                        "predicate": predicate,
                        "export_status": EXPORT_STATUS_DEGRADED,
                        "status": block.get("status"),
                        "status_reason": block.get("status_reason"),
                    }
                )
            else:
                gaps.append(
                    {
                        "scenario_id": row["scenario_id"],
                        "seed": row["seed"],
                        "planner_id": row["planner_id"],
                        "run_id": row["run_id"],
                        "episode_id": row["episode_id"],
                        "predicate": predicate,
                        "export_status": EXPORT_STATUS_MISSING,
                        "reason": block.get("reason"),
                    }
                )
        schema_versions = sorted(observed_schema_versions)
        predicate_types.append(
            {
                "predicate": predicate,
                "record_key": spec["record_key"],
                "schema_version": schema_versions[0] if schema_versions else spec["schema_version"],
                "schema_versions": schema_versions,
                "episodes_present": present,
                "episodes_degraded": degraded,
                "export_status": EXPORT_STATUS_EXPORTED if present else EXPORT_STATUS_MISSING,
            }
        )

    return {
        "schema_version": TRACE_PREDICATE_MANIFEST_SCHEMA_VERSION,
        "export_schema_version": TRACE_PREDICATE_EXPORT_SCHEMA_VERSION,
        "release": release,
        "sources": list(sources),
        "episode_count": len(export_rows),
        "predicate_types": predicate_types,
        "gaps": gaps,
        "complete": len(gaps) == 0,
    }


def build_trace_predicate_coverage_report(
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the coverage report: exported-vs-motivated predicates for a release.

    Args:
        manifest: Manifest produced by :func:`build_trace_predicate_manifest`.

    Returns:
        Coverage report enumerating which motivated predicates were actually exported
        versus motivated-but-not-exported, so a reviewer can verify in one place whether
        a predicate was *measured* for the cited release.
    """

    motivated = [spec["predicate"] for spec in MOTIVATED_TRACE_PREDICATES]
    predicate_types = manifest["predicate_types"]
    by_predicate = {entry["predicate"]: entry for entry in predicate_types}

    exported: list[str] = []
    motivated_not_exported: list[str] = []
    per_predicate: dict[str, Any] = {}
    for predicate in motivated:
        entry = by_predicate.get(predicate)
        if entry is None or entry["episodes_present"] == 0:
            motivated_not_exported.append(predicate)
            per_predicate[predicate] = {
                "export_status": EXPORT_STATUS_MISSING,
                "episodes_present": 0,
                "episodes_degraded": 0,
            }
        else:
            exported.append(predicate)
            per_predicate[predicate] = {
                "export_status": EXPORT_STATUS_EXPORTED,
                "episodes_present": entry["episodes_present"],
                "episodes_degraded": entry["episodes_degraded"],
            }

    return {
        "schema_version": TRACE_PREDICATE_COVERAGE_SCHEMA_VERSION,
        "release": manifest["release"],
        "motivated_predicates": motivated,
        "exported_predicates": sorted(exported),
        "motivated_not_exported": sorted(motivated_not_exported),
        "per_predicate": per_predicate,
        "summary": {
            "motivated_count": len(motivated),
            "exported_count": len(exported),
            "motivated_not_exported_count": len(motivated_not_exported),
        },
    }


def export_trace_predicates_from_bundle(
    bundle_paths: Iterable[Path],
    *,
    release: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    """Read campaign trace bundles and produce export, manifest, and coverage.

    Args:
        bundle_paths: One or more ``episodes.jsonl`` paths (one per campaign run).
        release: Release/run label for the manifest and coverage report.

    Returns:
        Tuple of (export_rows, manifest, coverage_report, failed_sources). A source that
        cannot be read is recorded in ``failed_sources`` rather than aborting the whole
        export (the gap is surfaced, not hidden).
    """

    export_rows: list[dict[str, Any]] = []
    sources: list[str] = []
    failed_sources: list[str] = []
    for path in bundle_paths:
        path = Path(path)
        try:
            episodes = _read_jsonl(path)
        except (OSError, ValueError) as exc:
            failed_sources.append(f"{path}: {exc}")
            continue
        run_id = path.parent.name or path.stem
        export_rows.extend(build_trace_predicate_export(episodes, run_id=run_id))
        sources.append(str(path))

    if failed_sources:
        details = "; ".join(failed_sources)
        raise TracePredicateExportError(
            f"refusing partial trace predicate export; failed source(s): {details}"
        )
    if not export_rows and not failed_sources:
        raise TracePredicateExportError("no episode records found in any provided bundle")

    manifest = build_trace_predicate_manifest(
        export_rows=export_rows, sources=sources, release=release
    )
    coverage = build_trace_predicate_coverage_report(manifest=manifest)
    return export_rows, manifest, coverage, failed_sources


def _crosswalk_predicate_status(block: Mapping[str, Any]) -> str:
    """Map an export predicate block's status to a crosswalk-recognized status.

    The scenario-evidence crosswalk classifies a predicate as missing/degraded only when
    its ``status`` is in ``{degraded, missing, unavailable, fallback}``. Export blocks
    carry ``export_status`` (exported/degraded/missing); a degraded block additionally
    carries a producer ``status`` such as ``not_applicable`` that the crosswalk would
    otherwise treat as a clean measurement. Map explicitly so a degraded predicate is
    never misrepresented as exported evidence downstream.

    Returns:
        ``missing`` for a missing block, ``degraded`` for a degraded block, else ``ok``.
    """

    export_status = block.get("export_status")
    if export_status == EXPORT_STATUS_MISSING:
        return "missing"
    if export_status == EXPORT_STATUS_DEGRADED:
        return "degraded"
    return "ok"


def _roll_up_predicates_per_scenario(
    export_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Roll per-episode predicate blocks up to one entry per scenario/predicate.

    A predicate counts as present for a scenario when at least one episode measured it
    (exported) or carried it degraded; a predicate missing from every episode is omitted so
    the crosswalk lists it as ``motivated_not_exported``. Among present statuses prefer
    ``exported`` over ``degraded`` (a clean measurement in any episode means the predicate
    was measured for the scenario).

    Returns:
        Mapping ``scenario_id -> {predicate -> {export_status, schema_version}}``.
    """

    rank = {EXPORT_STATUS_EXPORTED: 0, EXPORT_STATUS_DEGRADED: 1}
    per_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for row in export_rows:
        scenario_id = row.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            continue
        blocks = row.get("predicates")
        if not isinstance(blocks, Mapping):
            continue
        bucket = per_scenario.setdefault(scenario_id, {})
        _absorb_scenario_predicate_blocks(blocks, bucket, rank=rank)
    return per_scenario


def _absorb_scenario_predicate_blocks(
    blocks: Mapping[str, Any],
    bucket: dict[str, dict[str, Any]],
    *,
    rank: Mapping[str, int],
) -> None:
    """Fold one row's per-predicate blocks into a scenario bucket, preferring clean status."""

    for predicate, block in blocks.items():
        if not isinstance(block, Mapping):
            continue
        export_status = block.get("export_status")
        # Missing records are not "present"; skip so the predicate can surface as
        # motivated-not-exported when no episode measured it.
        if export_status not in rank:
            continue
        current = bucket.get(predicate)
        if current is None or rank[export_status] < rank[current["export_status"]]:
            bucket[predicate] = {
                "export_status": export_status,
                "schema_version": block.get("schema_version"),
            }


def build_crosswalk_predicate_export(
    export_rows: Sequence[Mapping[str, Any]],
    *,
    release: str | None = None,
) -> dict[str, Any]:
    """Flatten real export-lane rows into the scenario-evidence crosswalk envelope.

    The export lane's canonical artifact (``build_trace_predicate_export``) emits one row
    per episode whose ``predicates`` field is a *dict* keyed by predicate name, while the
    downstream scenario-evidence crosswalk (``robot_sf/benchmark/scenario_evidence_crosswalk.py``,
    issue #5602) consumes a single object with a ``rows`` list whose per-row ``predicates``
    is a *list* of ``{predicate, schema_version, status}`` records. This adapter is the
    bridge: it lets a real export-lane output flow through the crosswalk without a consumer
    re-deriving the shape each time, so the export lane actually serves its documented
    downstream consumers (issue #5593 motivation).

    ``status`` is derived from each block's ``export_status`` so a degraded/missing
    predicate is reported as such (``degraded``/``missing``) rather than silently treated
    as a clean measurement by the crosswalk.

    Episodes are rolled up **per scenario**: the export lane is per-episode while the
    crosswalk is per-scenario, so a predicate that is cleanly exported in any episode of a
    scenario is reported ``ok`` for that scenario (it *was* measured); a predicate that is
    only ever degraded across a scenario's episodes is reported ``degraded``; a predicate
    absent from every episode of a scenario is omitted so the crosswalk lists it as
    ``motivated_not_exported`` for that scenario (never silently inferred as measured).

    Args:
        export_rows: Rows produced by :func:`build_trace_predicate_export`.
        release: Optional release label carried in the envelope for traceability.

    Returns:
        Envelope ``{schema_version, release, rows}`` with one row per scenario, each
        carrying ``scenario_id`` and ``predicates`` as a list of
        ``{predicate, schema_version, status}`` records, deterministically ordered by
        ``scenario_id`` and predicate name.
    """

    per_scenario = _roll_up_predicates_per_scenario(export_rows)

    flat_rows: list[dict[str, Any]] = []
    for scenario_id in sorted(per_scenario):
        bucket = per_scenario[scenario_id]
        predicates: list[dict[str, Any]] = []
        for predicate in sorted(bucket):
            block = bucket[predicate]
            predicates.append(
                {
                    "predicate": predicate,
                    "schema_version": block["schema_version"],
                    "status": _crosswalk_predicate_status(block),
                }
            )
        flat_rows.append({"scenario_id": scenario_id, "predicates": predicates})

    envelope: dict[str, Any] = {
        "schema_version": CROSSWALK_PREDICATE_EXPORT_SCHEMA_VERSION,
        "rows": flat_rows,
    }
    if release is not None:
        envelope["release"] = release
    return envelope


def validate_trace_predicate_export(payload: Mapping[str, Any]) -> list[str]:
    """Validate a single decoded export row against the contract.

    Returns:
        List of violation messages; empty when the row is contract-conformant.
    """

    violations: list[str] = []
    for required in ("scenario_id", "seed", "planner_id", "episode_id", "predicates"):
        value = payload.get(required)
        if required not in payload or value is None or value == "":
            violations.append(f"export row missing required field: {required}")
    predicates = payload.get("predicates")
    if not isinstance(predicates, Mapping):
        violations.append("export row 'predicates' must be a mapping")
        return violations
    for spec in MOTIVATED_TRACE_PREDICATES:
        predicate = spec["predicate"]
        block = predicates.get(predicate)
        if not isinstance(block, Mapping):
            violations.append(f"missing predicate block: {predicate}")
            continue
        status = block.get("export_status")
        if status not in (EXPORT_STATUS_EXPORTED, EXPORT_STATUS_DEGRADED, EXPORT_STATUS_MISSING):
            violations.append(f"predicate {predicate} has invalid export_status: {status!r}")
        if status == EXPORT_STATUS_MISSING and not isinstance(block.get("reason"), str):
            violations.append(f"missing predicate {predicate} must carry a reason")
        if status != EXPORT_STATUS_MISSING:
            violations.extend(
                _predicate_record_violations(
                    block,
                    predicate=predicate,
                    schema_versions=_SUPPORTED_PREDICATE_SCHEMAS[predicate],
                )
            )
    return violations


def write_trace_predicate_export(
    *,
    export_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    coverage: Mapping[str, Any],
    export_jsonl: Path,
    manifest_json: Path,
    coverage_json: Path,
) -> tuple[Path, Path, Path]:
    """Write the export (JSON-lines), manifest, and coverage report to disk.

    The export is written with sorted JSON per line and rows in deterministic order, so
    the same input bundle yields a byte-identical file.

    Returns:
        Tuple of (export_jsonl, manifest_json, coverage_json) paths actually written.
    """

    export_jsonl = Path(export_jsonl)
    manifest_json = Path(manifest_json)
    coverage_json = Path(coverage_json)
    export_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    coverage_json.parent.mkdir(parents=True, exist_ok=True)

    with export_jsonl.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    coverage_json.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return export_jsonl, manifest_json, coverage_json


__all__ = [
    "CROSSWALK_PREDICATE_EXPORT_SCHEMA_VERSION",
    "DEGRADED_PREDICATE_STATUSES",
    "EXPORT_STATUS_DEGRADED",
    "EXPORT_STATUS_EXPORTED",
    "EXPORT_STATUS_MISSING",
    "MOTIVATED_TRACE_PREDICATES",
    "TRACE_PREDICATE_COVERAGE_SCHEMA_VERSION",
    "TRACE_PREDICATE_EXPORT_SCHEMA_VERSION",
    "TRACE_PREDICATE_MANIFEST_SCHEMA_VERSION",
    "TracePredicateExportError",
    "build_crosswalk_predicate_export",
    "build_trace_predicate_coverage_report",
    "build_trace_predicate_export",
    "build_trace_predicate_manifest",
    "export_trace_predicates_from_bundle",
    "validate_trace_predicate_export",
    "write_trace_predicate_export",
]
