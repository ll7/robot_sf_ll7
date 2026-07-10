#!/usr/bin/env python3
"""Build issue #4206 mechanism-level policy-structure cross-cut evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    GEOMETRY_ONLY_FIELDS,
    MECHANISM_SCHEMA_VERSION,
    REQUIRED_MECHANISM_FIELDS,
    TRACE_VERIFIED_EVIDENCE_MODES,
    FailureMechanismTaxonomyError,
    validate_failure_mechanism_record,
)
from robot_sf.benchmark.identity.hash_utils import read_jsonl as _read_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

CONFIG_SCHEMA_VERSION = "issue_4206_policy_structure_mechanism_crosscut_config.v1"
READY_STATUS = "analysis_ready_trace_verified"
BLOCKED_STATUS = "blocked_missing_trace_verified_mechanism_labels"
# The h600 run artifacts were never retrieved to this host, so no episode rows exist to inspect.
# This is a distinct, earlier blocker than "rows present but unlabeled" and needs a different next
# action (retrieve/hydrate the run outputs, not add mechanism instrumentation to the exporter).
BLOCKED_INPUT_STATUS = "blocked_missing_input_artifacts"
# Rows (or their declared sidecar) DO carry failure_mechanism_taxonomy.v1 fields, but every label is
# an explicit `not_derivable` unknown because the episodes predate trace capture (#4301). Adding
# another sidecar cannot recover these labels; only a trace-instrumented re-run can. Naming the
# generic "add a sidecar" action here would send a reviewer to redo work that already produced
# all-unknown labels, so this precise blocker gets its own status and follow-up.
BLOCKED_NOT_DERIVABLE_STATUS = "blocked_trace_labels_not_derivable_predates_trace_capture"
DIAGNOSTIC_STATUS = "diagnostic_only_supported_hypothesis"

# Tags attached to per-run missing entries so the packet can name the correct next empirical action.
BLOCKER_KIND_INPUT = "missing_input_artifact"
BLOCKER_KIND_LABEL = "missing_mechanism_labels"
BLOCKER_KIND_NOT_DERIVABLE = "mechanism_labels_not_derivable_predates_trace_capture"
REPORT_SCHEMA_VERSION = "issue_4206_policy_structure_mechanism_crosscut_report.v1"

LEGACY_MECHANISM_FIELD_ALIASES = (
    ("failure_mechanism_taxonomy_schema", "mechanism_schema_version"),
    ("failure_mechanism_label", "mechanism_label"),
    ("failure_mechanism_confidence", "mechanism_confidence"),
    ("failure_mechanism_trace_status", "mechanism_evidence_mode"),
    ("failure_mechanism_source", "mechanism_evidence_uri"),
)
# Marker fields the taxonomy backfill (#4278/#4242) writes to explain *why* a label is unknown.
# They are not part of the required-label contract, but they let this builder tell "the taxonomy was
# applied and no trace was derivable" apart from "no mechanism fields at all". Carry them across the
# sidecar merge so downstream classification can read them.
MECHANISM_MARKER_FIELDS = ("mechanism_backfill_status", "mechanism_caveat")
# Only "no trace exists at all because the episode predates capture" routes to the predates-trace
# status; it is fixed by a trace-instrumented re-run. A geometry-only assertion
# (`not_derivable_non_trace_verified_mechanism`) is a different gap (a label was asserted without a
# trace) and stays a plain missing-trace-verified-label block, so match the marker exactly.
NOT_DERIVABLE_MISSING_TRACE_MARKER = "not_derivable_missing_trace"
SIDECAR_NAMES = (
    "mechanism_labels.csv",
    "failure_mechanism_labels.csv",
    "trace_failure_mechanisms.csv",
    "mechanism_labels.jsonl",
    "failure_mechanism_labels.jsonl",
)
FALLBACK_STATUSES = {"fallback", "degraded", "failed", "partial-failure", "not_available"}


class BuildError(ValueError):
    """Raised when issue #4206 inputs or config are malformed."""


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BuildError(f"{path} must contain a YAML mapping")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _public_path(path: Path) -> str:
    """Return a repo-public path without local home/worktree prefixes."""
    resolved = path.resolve()
    for anchor in ("docs", "configs", "scripts", "tests", "output"):
        if anchor in resolved.parts:
            index = resolved.parts.index(anchor)
            return str(Path(*resolved.parts[index:]))
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _boolish_rate(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = _to_float(row.get(name))
        if value is not None:
            return 1.0 if value > 0 else 0.0
    return 0.0


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _stringify_key(value: Any) -> str:
    """Stringify key parts while preserving legitimate zero values."""
    return "" if value is None else str(value)


def _first_float(row: Mapping[str, Any], *names: str) -> float | None:
    """Return the first present float field without treating 0.0 as missing."""
    for name in names:
        value = _to_float(row.get(name))
        if value is not None:
            return value
    return None


def _row_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        _stringify_key(row.get("episode_id")),
        _stringify_key(row.get("scenario_id")),
        _stringify_key(
            row.get("planner_key") if row.get("planner_key") is not None else row.get("planner_id")
        ),
        _stringify_key(row.get("seed")),
        _stringify_key(
            row.get("repeat_index")
            if row.get("repeat_index") is not None
            else row.get("episode_index")
        ),
    )


def _sidecar_keys(row: Mapping[str, Any]) -> list[tuple[str, ...]]:
    full = _row_key(row)
    return [
        full,
        ("", full[1], full[2], full[3], full[4]),
    ]


def _discover_sidecar_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reports = root / "reports"
    for name in SIDECAR_NAMES:
        path = reports / name
        if not path.exists():
            continue
        if path.suffix == ".csv":
            rows.extend(_read_csv(path))
        elif path.suffix == ".jsonl":
            rows.extend(_read_jsonl(path))
    return rows


def _index_sidecars(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], Mapping[str, Any]]:
    index: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        for key in _sidecar_keys(row):
            if any(key):
                index.setdefault(key, row)
    return index


def _mechanism_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in REQUIRED_MECHANISM_FIELDS}


def _mechanism_validation_errors(row: Mapping[str, Any]) -> list[str]:
    missing_fields = [field for field in REQUIRED_MECHANISM_FIELDS if field not in row]
    if missing_fields:
        return missing_fields
    try:
        validate_failure_mechanism_record(_mechanism_payload(row))
    except FailureMechanismTaxonomyError as exc:
        return [f"failure_mechanism_taxonomy: {exc}"]
    return []


def _normalize_mechanism_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize v1 episode-row taxonomy fields to the report's stable names."""

    normalized = dict(row)
    for alias, canonical in LEGACY_MECHANISM_FIELD_ALIASES:
        if not normalized.get(canonical) and normalized.get(alias):
            normalized[canonical] = normalized[alias]
    return normalized


def _merge_mechanism_fields(
    row: Mapping[str, Any], sidecar_index: Mapping[tuple[str, ...], Mapping[str, Any]]
) -> dict[str, Any]:
    merged = _normalize_mechanism_fields(row)
    if not _mechanism_validation_errors(merged):
        return merged
    for key in _sidecar_keys(row):
        sidecar = sidecar_index.get(key)
        if sidecar is None:
            continue
        sidecar = _normalize_mechanism_fields(sidecar)
        for field in REQUIRED_MECHANISM_FIELDS:
            if (field not in merged or not merged.get(field)) and field in sidecar:
                merged[field] = sidecar[field]
        for field in MECHANISM_MARKER_FIELDS:
            if not merged.get(field) and sidecar.get(field):
                merged[field] = sidecar[field]
        merged = _normalize_mechanism_fields(merged)
        break
    return merged


def _is_not_derivable_row(row: Mapping[str, Any]) -> bool:
    """Return True when a row carries the v1 taxonomy but is an explicit ``not_derivable`` unknown.

    Such a row proves the taxonomy backfill ran and could not derive a real label (the episode
    predates trace capture), which is a different blocker than "no mechanism fields at all": a sidecar
    backfill cannot fix it, only a trace-instrumented re-run can.
    """

    if str(row.get("mechanism_schema_version", "")) != MECHANISM_SCHEMA_VERSION:
        return False
    label = str(row.get("mechanism_label", "")).strip().lower()
    confidence = str(row.get("mechanism_confidence", "")).strip().lower()
    # A real label alone rules out "every label is unknown": a row with a usable mechanism_label
    # must not be routed to the predates-trace status just because its confidence is blank while a
    # stale not_derivable marker lingers. Require the label itself to be unknown/absent.
    if label not in {"", "unknown"} or confidence not in {"", "unknown"}:
        return False
    markers = " ".join(str(row.get(field, "")).lower() for field in MECHANISM_MARKER_FIELDS)
    return NOT_DERIVABLE_MISSING_TRACE_MARKER in markers


def _resolve_declared_path(raw: str, base_dir: Path) -> Path:
    """Resolve a config-declared sidecar path against the repo base dir.

    Declared sidecar paths are documented as repo-root relative, so resolve them deterministically
    against ``base_dir`` (the config's repo root). Resolving against the process cwd first would let
    an unrelated same-named file under the caller's cwd silently win and corrupt the provenance.
    """
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def _load_declared_sidecars(
    config: Mapping[str, Any], base_dir: Path
) -> tuple[dict[tuple[str, ...], Mapping[str, Any]], list[dict[str, Any]]]:
    """Load config-declared external mechanism-label sidecars (the #4305 declared-sidecar path).

    The h600 confirm/extended episode rows predate trace capture (#4301), so their trace-verified
    mechanism labels live in a declared sidecar under docs/context/evidence/ rather than in the run
    output tree. Consuming the declared sidecar here is what lets the builder observe those labels
    (or, today, observe that every label is ``not_derivable``) instead of reporting a generic
    missing-label block.
    """
    declared = config.get("declared_mechanism_sidecars") or []
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for raw in declared:
        path = _resolve_declared_path(str(raw), base_dir)
        exists = path.exists()
        provenance.append({"path": _public_path(path), "exists": exists})
        if not exists:
            continue
        if path.suffix == ".jsonl":
            rows.extend(_read_jsonl(path))
        else:
            rows.extend(_read_csv(path))
    return _index_sidecars(rows), provenance


def _planner_to_class(config: Mapping[str, Any]) -> dict[str, str]:
    classes = config.get("structural_classes")
    if not isinstance(classes, dict):
        raise BuildError("config.structural_classes must be a mapping")
    planner_map: dict[str, str] = {}
    for class_name, payload in classes.items():
        if not isinstance(payload, dict):
            raise BuildError(f"structural class {class_name!r} must be a mapping")
        for planner in payload.get("planner_keys") or []:
            planner_key = str(planner)
            if planner_key in planner_map:
                raise BuildError(f"planner key {planner_key!r} appears in multiple classes")
            planner_map[planner_key] = str(class_name)
    return planner_map


def _load_run_rows(
    root: Path,
    run_name: str,
    declared_sidecar_index: Mapping[tuple[str, ...], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_path = root / "reports" / "seed_episode_rows.csv"
    if not rows_path.exists():
        return [], [
            {
                "run": run_name,
                "missing_path": _public_path(rows_path),
                "missing_fields": ["reports/seed_episode_rows.csv"],
                "blocker_kind": BLOCKER_KIND_INPUT,
            }
        ]
    # In-root sidecars take precedence, then config-declared external sidecars (the #4305 path) fill
    # any keys the run tree does not already carry.
    sidecar_index: dict[tuple[str, ...], Mapping[str, Any]] = dict(declared_sidecar_index or {})
    sidecar_index.update(_index_sidecars(_discover_sidecar_rows(root)))
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for raw in _read_csv(rows_path):
        row = _merge_mechanism_fields(raw, sidecar_index)
        row["source_run"] = run_name
        row["source_root"] = str(root)
        missing_fields = _mechanism_validation_errors(row)
        if row.get("mechanism_evidence_mode") not in TRACE_VERIFIED_EVIDENCE_MODES:
            missing_fields.append("mechanism_evidence_mode=trace_verified_source")
        if missing_fields:
            not_derivable = _is_not_derivable_row(row)
            missing.append(
                {
                    "run": run_name,
                    "episode_id": row.get("episode_id", ""),
                    "scenario_id": row.get("scenario_id", ""),
                    "planner_key": row.get("planner_key", row.get("planner_id", "")),
                    "missing_fields": missing_fields,
                    "blocker_kind": (
                        BLOCKER_KIND_NOT_DERIVABLE if not_derivable else BLOCKER_KIND_LABEL
                    ),
                    "mechanism_backfill_status": row.get("mechanism_backfill_status", ""),
                    "mechanism_caveat": row.get("mechanism_caveat", ""),
                    "geometry_only_fields_present": [
                        field for field in GEOMETRY_ONLY_FIELDS if str(row.get(field, ""))
                    ],
                }
            )
        rows.append(row)
    return rows, missing


def _is_fallback_or_degraded(row: Mapping[str, Any]) -> bool:
    fields = (
        row.get("status"),
        row.get("run_status"),
        row.get("planner_status"),
        row.get("availability_status"),
        row.get("execution_mode"),
    )
    return any(str(value).strip().lower() in FALLBACK_STATUSES for value in fields if value)


def _score(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    snqi = row.get("snqi_mean")
    snqi_score = -9999.0 if snqi is None else float(snqi)
    return (
        -float(row.get("success_rate", 0.0)),
        float(row.get("collision_event_rate", 0.0)),
        float(row.get("near_miss_event_rate", 0.0)),
        float(row.get("timeout_rate", 0.0)),
        -snqi_score,
    )


def _summarize_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows = [_normalize_mechanism_fields(row) for row in rows]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        grouped[(str(row["mechanism_label"]), str(row["structural_class"]))].append(row)

    summaries: list[dict[str, Any]] = []
    by_mechanism: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (mechanism, structural_class), group in sorted(grouped.items()):
        count = len(group)
        fallback_count = sum(1 for row in group if _is_fallback_or_degraded(row))
        summary = {
            "mechanism_label": mechanism,
            "structural_class": structural_class,
            "episode_count": count,
            "planner_keys": ";".join(sorted({str(row.get("planner_key", "")) for row in group})),
            "success_rate": _mean(_to_float(row.get("success")) for row in group) or 0.0,
            "collision_event_rate": _mean(
                _boolish_rate(row, "collision", "collisions") for row in group
            )
            or 0.0,
            "near_miss_event_rate": _mean(
                _boolish_rate(row, "near_miss", "near_misses") for row in group
            )
            or 0.0,
            "timeout_rate": _mean(_boolish_rate(row, "timeout", "timed_out") for row in group)
            or 0.0,
            "progress_mean": _mean(
                _first_float(row, "progress", "progress_ratio") for row in group
            ),
            "snqi_mean": _mean(_to_float(row.get("snqi")) for row in group),
            "fallback_or_degraded_count": fallback_count,
            "mechanism_confidence_mix": json.dumps(
                Counter(str(row.get("mechanism_confidence", "")) for row in group),
                sort_keys=True,
            ),
            "eligible_f_c4ii": fallback_count < count,
        }
        summaries.append(summary)
        by_mechanism[mechanism].append(summary)

    for mechanism_rows in by_mechanism.values():
        mechanism_rows.sort(key=_score)
        for rank, row in enumerate(mechanism_rows, start=1):
            row["mechanism_rank"] = rank
    return sorted(summaries, key=lambda row: (row["mechanism_label"], int(row["mechanism_rank"])))


def _predictive_probe(rank_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_mechanism: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        by_mechanism[str(row["mechanism_label"])].append(row)
    probes: list[dict[str, Any]] = []
    for mechanism, rows in sorted(by_mechanism.items()):
        by_class = {str(row["structural_class"]): row for row in rows}
        predictive = by_class.get("predictive")
        constraint = by_class.get("constraint_first_hybrid")
        learned = by_class.get("learned_policy")
        predictive_rank = int(predictive["mechanism_rank"]) if predictive else None
        constraint_rank = int(constraint["mechanism_rank"]) if constraint else None
        learned_rank = int(learned["mechanism_rank"]) if learned else None
        beats_constraint = (
            predictive_rank is not None
            and constraint_rank is not None
            and predictive_rank < constraint_rank
        )
        beats_learned = (
            predictive_rank is not None
            and learned_rank is not None
            and predictive_rank < learned_rank
        )
        probes.append(
            {
                "mechanism_label": mechanism,
                "predictive_rank": predictive_rank or "",
                "constraint_first_hybrid_rank": constraint_rank or "",
                "learned_policy_rank": learned_rank or "",
                "predictive_beats_constraint_first_hybrid": beats_constraint,
                "predictive_beats_learned_policy": beats_learned,
                "predictive_beats_both": beats_constraint and beats_learned,
                "eligible_f_c4ii": predictive is not None
                and constraint is not None
                and learned is not None,
            }
        )
    return probes


def _local_minimum_probe(
    rank_rows: Sequence[Mapping[str, Any]], local_minimum_mechanisms: set[str]
) -> list[dict[str, Any]]:
    return [
        {
            "mechanism_label": row["mechanism_label"],
            "structural_class": row["structural_class"],
            "mechanism_rank": row["mechanism_rank"],
            "success_rate": row["success_rate"],
            "timeout_rate": row["timeout_rate"],
            "progress_mean": row["progress_mean"] if row["progress_mean"] is not None else "",
            "fallback_or_degraded_count": row["fallback_or_degraded_count"],
            "eligible_f_c4ii": row["eligible_f_c4ii"],
        }
        for row in rank_rows
        if str(row["mechanism_label"]) in local_minimum_mechanisms
    ]


def _agreement_table(
    rows: Sequence[Mapping[str, Any]], rank_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rank_lookup = {
        (str(row["mechanism_label"]), str(row["structural_class"])): row.get("mechanism_rank", "")
        for row in rank_rows
    }
    seen: set[tuple[str, str, str, str]] = set()
    table: list[dict[str, Any]] = []
    for row in rows:
        geometry_bucket = str(
            row.get("geometry_bucket")
            or row.get("scenario_geometry_bucket")
            or row.get("geometry_label")
            or ""
        )
        mechanism = str(row.get("mechanism_label", ""))
        planner = str(row.get("planner_key") or row.get("planner_id") or "")
        structural_class = str(row.get("structural_class", ""))
        key = (geometry_bucket, mechanism, planner, structural_class)
        if key in seen:
            continue
        seen.add(key)
        status = (
            "geometry_present_not_rank_compared"
            if geometry_bucket
            else "not_comparable_scope_or_roster_change"
        )
        table.append(
            {
                "geometry_bucket": geometry_bucket,
                "mechanism_label": mechanism,
                "planner_key": planner,
                "structural_class": structural_class,
                "old_geometry_rank": "",
                "new_mechanism_rank": rank_lookup.get((mechanism, structural_class), ""),
                "agreement_status": status,
                "conclusion_survives": "",
                "caveat": "old geometry-bucket ranks are not recomputed by this mechanism-level builder",
            }
        )
    return sorted(
        table,
        key=lambda row: (
            row["geometry_bucket"],
            row["mechanism_label"],
            row["structural_class"],
            row["planner_key"],
        ),
    )


def _markdown_report(status: str, rank_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Issue #4206 mechanism-level policy-structure cross-cut",
        "",
        f"status: {status}",
        "",
        "This packet is diagnostic analysis support only. It is not a benchmark campaign,",
        "Slurm/GPU submission, or paper/dissertation claim edit.",
        "",
    ]
    if rank_rows:
        lines.extend(
            [
                "## Mechanism x Structural Class Ranks",
                "",
                "| mechanism_label | rank | structural_class | success_rate | collision_rate | near_miss_rate |",
                "| --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in rank_rows:
            lines.append(
                "| {mechanism_label} | {mechanism_rank} | {structural_class} | "
                "{success_rate:.3f} | {collision_event_rate:.3f} | "
                "{near_miss_event_rate:.3f} |".format(**row)
            )
    return "\n".join(lines).rstrip() + "\n"


def _blocked_reason(
    status: str,
    config: Mapping[str, Any],
    missing_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve the follow-up, next action, and claim boundary for a blocked status.

    The two blocked statuses need different guidance: an un-retrieved-artifact block is fixed by
    hydrating the run outputs to this host, while a missing-label block is fixed by adding
    trace-verified mechanism instrumentation to the exporter. Naming the wrong action sends a
    reviewer down the wrong path, so keep the mapping explicit.
    """

    if status == BLOCKED_INPUT_STATUS:
        missing_inputs = [
            row for row in missing_rows if row.get("blocker_kind") == BLOCKER_KIND_INPUT
        ]
        runs = sorted({str(row.get("run", "")) for row in missing_inputs if row.get("run")})
        followup = config.get("blocked_retrieval_followup") or {
            "title": "Retrieve h600 mechanism-crosscut episode artifacts to the analysis host",
            "body": (
                "Issue #4206 cannot inspect mechanism labels until the declared h600 run "
                "artifacts are present on the analysis host. Missing runs: "
                f"{', '.join(runs) or 'declared h600 runs'}. Hydrate reports/seed_episode_rows.csv "
                "(and any mechanism sidecars) from the canonical run store, then re-run the "
                "builder. This is an artifact-retrieval gap, not a mechanism-instrumentation gap."
            ),
        }
        next_action = (
            "Retrieve/hydrate the missing h600 run artifacts (reports/seed_episode_rows.csv and any "
            "mechanism sidecars) to this host, then re-run the builder. No exporter "
            "mechanism-label change is required until the rows are present."
        )
        claim_boundary = (
            "Blocked before any mechanism-label inspection because the declared h600 run artifacts "
            "are not present on this host."
        )
        return {"followup": followup, "next_action": next_action, "claim_boundary": claim_boundary}

    if status == BLOCKED_NOT_DERIVABLE_STATUS:
        followup = config.get("blocked_not_derivable_followup") or {
            "title": (
                "Re-run the h600 campaign with trace capture so failure-mechanism labels "
                "are derivable for F-C4(ii)"
            ),
            "body": (
                "Issue #4206 consumed the declared failure-mechanism sidecar, but every h600 row is "
                "an explicit `not_derivable` unknown: the confirm (13268) and extended-roster "
                "(13273) episodes predate trace capture (#4301), so no trace exists to derive a "
                "trace-verified mechanism label. A sidecar backfill cannot recover these labels. "
                "Re-run the h600 campaign with the #4301 trace-capable exporter (or a deterministic "
                "replay that regenerates traces), then re-run this builder."
            ),
        }
        next_action = (
            "The declared mechanism sidecar was consumed but all labels are `not_derivable` (rows "
            "predate trace capture #4301). Re-run the h600 campaign with trace capture (or a "
            "deterministic replay), then re-run the builder. Adding another sidecar cannot recover "
            "these labels."
        )
        claim_boundary = (
            "Blocked before F-C4(ii) rank conclusions: the mechanism taxonomy was applied but every "
            "label is a `not_derivable` unknown because the h600 episodes predate trace capture."
        )
        return {"followup": followup, "next_action": next_action, "claim_boundary": claim_boundary}

    followup = config.get("blocked_followup_issue", {})
    next_action = (
        "Add trace-verified failure-mechanism labels to the h600 episode exports (or a declared "
        "sidecar), then re-run the builder."
    )
    claim_boundary = (
        "Blocked before F-C4(ii) rank conclusions because trace-verified mechanism labels are "
        "unavailable."
    )
    return {"followup": followup, "next_action": next_action, "claim_boundary": claim_boundary}


def _blocked_outputs(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    generated_at: str,
    missing_rows: Sequence[Mapping[str, Any]],
    loaded_row_count: int,
    input_provenance: Mapping[str, Any],
    status: str = BLOCKED_STATUS,
    all_loaded_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reason = _blocked_reason(status, config, missing_rows)
    blocker_kinds = sorted(
        {str(row.get("blocker_kind", BLOCKER_KIND_LABEL)) for row in missing_rows}
    )
    missing_payload = {
        "status": status,
        "issue": 4206,
        "generated_at": generated_at,
        "blocker_kinds": blocker_kinds,
        "next_action": reason["next_action"],
        "required_fields": list(REQUIRED_MECHANISM_FIELDS),
        "loaded_row_count": loaded_row_count,
        "input_provenance": dict(input_provenance),
        "missing_rows_sample": list(missing_rows[:50]),
        "missing_row_count": len(missing_rows),
        "geometry_bucket_substitution_rejected": (
            config.get("forbidden_fallback", {}).get(
                "geometry_buckets_may_substitute_mechanism_labels"
            )
            is False
        )
        or any(row.get("geometry_only_fields_present") for row in missing_rows),
        "followup_issue_skeleton": reason["followup"],
    }
    _write_json(output_dir / "missing_instrumentation.json", missing_payload)
    metadata = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "issue": 4206,
        "generated_at": generated_at,
        "blocker_kinds": blocker_kinds,
        "next_action": reason["next_action"],
        "input_provenance": dict(input_provenance),
        "taxonomy_source": config.get("taxonomy_source"),
        "claim_boundary": reason["claim_boundary"],
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(
        output_dir / "mechanism_crosscut_report.json",
        {**metadata, "rank_rows": [], "probes": {}, "agreement_rows": []},
    )
    for filename, fieldnames in {
        "mechanism_by_structural_class.csv": [
            "mechanism_label",
            "mechanism_rank",
            "structural_class",
            "episode_count",
            "success_rate",
            "collision_event_rate",
            "near_miss_event_rate",
        ],
        "f_c4ii_probe_predictive_dominance.csv": [
            "mechanism_label",
            "predictive_rank",
            "constraint_first_hybrid_rank",
            "learned_policy_rank",
            "predictive_beats_both",
        ],
        "f_c4ii_probe_local_minimum_failures.csv": [
            "mechanism_label",
            "structural_class",
            "mechanism_rank",
            "success_rate",
            "timeout_rate",
        ],
        "geometry_vs_mechanism_agreement.csv": [
            "geometry_bucket",
            "mechanism_label",
            "planner_key",
            "structural_class",
            "old_geometry_rank",
            "new_mechanism_rank",
            "agreement_status",
            "conclusion_survives",
            "caveat",
        ],
    }.items():
        _write_csv(output_dir / filename, [], fieldnames)
    # Per-scenario geometry mapping: extracted independently of mechanism labels so the
    # scenario → bucket structure is re-derivable even while the builder is blocked. The
    # mapping is populated when episode rows carry geometry_bucket / geometry_label fields,
    # and left header-only when no geometry data is present in the loaded rows.
    geometry_mapping = _extract_scenario_geometry_mapping(all_loaded_rows)
    _write_csv(
        output_dir / "scenario_geometry_bucket_mapping.csv",
        geometry_mapping,
        list(_SCENARIO_GEOMETRY_MAPPING_FIELDS),
    )
    (output_dir / "mechanism_crosscut_report.md").write_text(
        _markdown_report(status, []), encoding="utf-8"
    )
    _write_readmes(output_dir, status, next_action=reason["next_action"])
    _write_sha256sums(output_dir)
    return metadata


def _write_readmes(output_dir: Path, status: str, *, next_action: str | None = None) -> None:
    next_action_line = f"\nCurrent blocker next action: {next_action}\n" if next_action else ""
    readme = f"""# Issue #4206 policy-structure mechanism cross-cut

status: {status}
{next_action_line}
This evidence packet is bounded to CPU-only diagnostic analysis for issue #4206. It does not run a
benchmark campaign, submit Slurm/GPU work, edit paper/dissertation claims, or promote generalized
causal claims.

Mechanism-level F-C4(ii) conclusions are allowed only when episode rows carry
`failure_mechanism_taxonomy.v1` fields with accepted confidence labels. Geometry buckets are used
only for the agreement/disagreement comparison table and never as substitute mechanism labels.

Blocked statuses distinguish three different next actions:

- `blocked_missing_input_artifacts`: the declared h600 run outputs are not present on this host;
  retrieve/hydrate them, then re-run. This is not a mechanism-instrumentation gap.
- `blocked_missing_trace_verified_mechanism_labels`: rows exist but lack trace-verified mechanism
  labels (no mechanism fields, or only geometry/non-trace assertions); add the labels to the exporter
  or a declared sidecar, then re-run.
- `blocked_trace_labels_not_derivable_predates_trace_capture`: the declared sidecar was consumed and
  every label is an explicit `not_derivable` unknown because the h600 episodes predate trace capture
  (#4301). A sidecar backfill cannot recover these labels; re-run the h600 campaign with trace
  capture (or a deterministic replay), then re-run.
"""
    claim_boundary = """# Claim Boundary

This packet can support a mechanism-level policy-structure cross-cut only for rows with
trace-verified failure-mechanism labels. Missing labels produce
`blocked_missing_trace_verified_mechanism_labels`, and retained rows whose labels are all
`not_derivable` because the episodes predate trace capture produce
`blocked_trace_labels_not_derivable_predates_trace_capture`; both stop F-C4(ii) rank conclusions.

Out of scope: no full benchmark campaign run, no Slurm/GPU submission, no paper/dissertation claim
edits, and no causal-mechanism claim from geometry buckets alone.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "claim_boundary.md").write_text(claim_boundary, encoding="utf-8")


def _extract_scenario_geometry_mapping(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract per-scenario geometry bucket mapping without requiring mechanism labels.

    Captures (scenario_id, geometry_bucket, scenario_family) from episode rows so the
    geometry structure is re-derivable even when the mechanism-level cross-cut is blocked.
    Each (scenario_id, geometry_bucket, scenario_family) triple is de-duplicated.
    Returns an empty list when no rows carry geometry data.
    """
    seen: set[tuple[str, str, str]] = set()
    mapping: list[dict[str, Any]] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "")
        geometry_bucket = str(
            row.get("geometry_bucket")
            or row.get("scenario_geometry_bucket")
            or row.get("geometry_label")
            or ""
        )
        scenario_family = str(row.get("scenario_family") or "")
        if not geometry_bucket:
            continue
        key = (scenario_id, geometry_bucket, scenario_family)
        if key in seen:
            continue
        seen.add(key)
        mapping.append(
            {
                "scenario_id": scenario_id,
                "geometry_bucket": geometry_bucket,
                "scenario_family": scenario_family,
            }
        )
    return sorted(mapping, key=lambda r: (r["geometry_bucket"], r["scenario_id"]))


def check_geometry_export_nonempty(path: Path) -> None:
    """Fail-closed guard: raise BuildError when the geometry CSV has no data rows.

    The geometry_vs_mechanism_agreement.csv starts header-only while the builder is
    blocked on mechanism labels. Once inputs are available and the builder produces a
    populated export, this guard prevents a silent regression to a header-only state.
    """
    if not path.exists():
        raise BuildError(f"geometry export not found: {path}")
    rows = _read_csv(path)
    if not rows:
        raise BuildError(
            f"geometry export is header-only (no data rows): {path} — "
            "re-run the builder with trace-verified mechanism labels to populate it"
        )


_SCENARIO_GEOMETRY_MAPPING_FIELDS = ("scenario_id", "geometry_bucket", "scenario_family")


def _write_sha256sums(output_dir: Path) -> None:
    lines: list[str] = []
    for path in sorted(item for item in output_dir.iterdir() if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {_public_path(path)}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _input_provenance(
    *,
    config_path: Path,
    confirm_root: Path,
    extended_root: Path,
    job13175_packet: Path,
    declared_sidecars: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Record the declared inputs checked by the issue #4206 builder."""

    return {
        "declared_mechanism_sidecars": [dict(item) for item in declared_sidecars],
        "config": {
            "path": _public_path(config_path),
            "exists": config_path.exists(),
        },
        "confirm_h600_13268": {
            "root": _public_path(confirm_root),
            "root_exists": confirm_root.exists(),
            "seed_episode_rows": _public_path(confirm_root / "reports" / "seed_episode_rows.csv"),
            "seed_episode_rows_exists": (
                confirm_root / "reports" / "seed_episode_rows.csv"
            ).exists(),
        },
        "extended_h600_13273": {
            "root": _public_path(extended_root),
            "root_exists": extended_root.exists(),
            "seed_episode_rows": _public_path(extended_root / "reports" / "seed_episode_rows.csv"),
            "seed_episode_rows_exists": (
                extended_root / "reports" / "seed_episode_rows.csv"
            ).exists(),
        },
        "continuity_h500_job13175": {
            "packet": _public_path(job13175_packet),
            "packet_exists": job13175_packet.exists(),
        },
    }


def _resolve_block_status(missing_rows: Sequence[Mapping[str, Any]]) -> str:
    """Pick the blocked status by precedence so the packet names the right next action.

    Most-fundamental first:
    1. un-retrieved input artifact -> hydrate the run outputs (nothing to inspect yet);
    2. taxonomy present but every remaining block is ``not_derivable`` -> trace-instrumented re-run
       (a sidecar backfill already ran and produced only unknowns);
    3. otherwise rows lack mechanism fields -> add taxonomy instrumentation.
    """

    if any(row.get("blocker_kind") == BLOCKER_KIND_INPUT for row in missing_rows):
        return BLOCKED_INPUT_STATUS
    non_input = [row for row in missing_rows if row.get("blocker_kind") != BLOCKER_KIND_INPUT]
    if non_input and all(
        row.get("blocker_kind") == BLOCKER_KIND_NOT_DERIVABLE for row in non_input
    ):
        return BLOCKED_NOT_DERIVABLE_STATUS
    return BLOCKED_STATUS


def build_packet(  # noqa: C901
    *,
    config_path: Path,
    confirm_root: Path,
    extended_root: Path,
    job13175_packet: Path,
    output_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Build the bounded issue #4206 evidence packet."""
    config = _load_yaml(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise BuildError("config schema_version mismatch")
    if int(config.get("issue", 0)) != 4206:
        raise BuildError("config.issue must be 4206")
    if config.get("mechanism_schema_version") != MECHANISM_SCHEMA_VERSION:
        raise BuildError("config mechanism_schema_version mismatch")

    # Declared external sidecars (the #4305 path) carry the trace-verified labels for h600 episode
    # rows that predate trace capture; resolve them relative to the config's repo root.
    repo_base = (
        config_path.resolve().parents[2] if len(config_path.resolve().parents) >= 3 else Path.cwd()
    )
    declared_index, declared_provenance = _load_declared_sidecars(config, repo_base)

    input_provenance = _input_provenance(
        config_path=config_path,
        confirm_root=confirm_root,
        extended_root=extended_root,
        job13175_packet=job13175_packet,
        declared_sidecars=declared_provenance,
    )

    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for run_name, root in (
        ("confirm_h600_13268", confirm_root),
        ("extended_h600_13273", extended_root),
    ):
        run_rows, run_missing = _load_run_rows(root, run_name, declared_index)
        rows.extend(run_rows)
        missing_rows.extend(run_missing)

    if missing_rows:
        return _blocked_outputs(
            output_dir=output_dir,
            config=config,
            generated_at=generated_at,
            missing_rows=missing_rows,
            loaded_row_count=len(rows),
            input_provenance=input_provenance,
            status=_resolve_block_status(missing_rows),
            all_loaded_rows=rows,
        )

    accepted_confidences = {
        str(item) for item in config.get("accepted_mechanism_confidences") or []
    }
    planner_map = _planner_to_class(config)
    enriched: list[dict[str, Any]] = []
    unclassified_count = 0
    for row in rows:
        if row.get("mechanism_schema_version") != MECHANISM_SCHEMA_VERSION:
            missing_rows.append({**_mechanism_payload(row), "reason": "schema_version_mismatch"})
            continue
        if row.get("mechanism_confidence") not in accepted_confidences:
            continue
        planner_key = str(row.get("planner_key") or row.get("planner_id") or "")
        structural_class = planner_map.get(planner_key, "unclassified")
        if structural_class == "unclassified":
            unclassified_count += 1
        enriched.append({**row, "planner_key": planner_key, "structural_class": structural_class})

    if missing_rows or not enriched:
        return _blocked_outputs(
            output_dir=output_dir,
            config=config,
            generated_at=generated_at,
            missing_rows=missing_rows
            or [
                {"missing_fields": list(REQUIRED_MECHANISM_FIELDS), "reason": "no accepted labels"}
            ],
            loaded_row_count=len(rows),
            input_provenance=input_provenance,
            all_loaded_rows=rows,
        )

    observed_count = sum(
        1 for row in enriched if row.get("mechanism_confidence") == "observed_mechanism"
    )
    status = READY_STATUS if observed_count else DIAGNOSTIC_STATUS
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_rows = _summarize_groups(
        [row for row in enriched if row.get("structural_class") != "unclassified"]
    )
    predictive_probe = _predictive_probe(rank_rows)
    local_probe = _local_minimum_probe(
        rank_rows, {str(item) for item in config.get("local_minimum_mechanisms") or []}
    )
    agreement = _agreement_table(enriched, rank_rows)

    metadata = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "issue": 4206,
        "generated_at": generated_at,
        "config": _public_path(config_path),
        "taxonomy_source": config.get("taxonomy_source"),
        "confirm_root": _public_path(confirm_root),
        "extended_root": _public_path(extended_root),
        "job13175_packet": _public_path(job13175_packet),
        "input_rows": len(rows),
        "accepted_rows": len(enriched),
        "observed_mechanism_rows": observed_count,
        "unclassified_planner_rows_excluded_from_f_c4ii": unclassified_count,
        "claim_boundary": "Mechanism-level conclusions only for accepted trace-labeled rows; "
        "geometry buckets are comparison-only.",
    }
    report = {
        **metadata,
        "rank_rows": rank_rows,
        "probes": {
            "predictive_dominance": predictive_probe,
            "local_minimum_failures": local_probe,
        },
        "agreement_rows": agreement,
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "mechanism_crosscut_report.json", report)
    (output_dir / "mechanism_crosscut_report.md").write_text(
        _markdown_report(status, rank_rows), encoding="utf-8"
    )
    _write_csv(
        output_dir / "mechanism_by_structural_class.csv",
        rank_rows,
        [
            "mechanism_label",
            "mechanism_rank",
            "structural_class",
            "episode_count",
            "planner_keys",
            "success_rate",
            "collision_event_rate",
            "near_miss_event_rate",
            "timeout_rate",
            "progress_mean",
            "snqi_mean",
            "fallback_or_degraded_count",
            "mechanism_confidence_mix",
            "eligible_f_c4ii",
        ],
    )
    _write_csv(
        output_dir / "f_c4ii_probe_predictive_dominance.csv",
        predictive_probe,
        [
            "mechanism_label",
            "predictive_rank",
            "constraint_first_hybrid_rank",
            "learned_policy_rank",
            "predictive_beats_constraint_first_hybrid",
            "predictive_beats_learned_policy",
            "predictive_beats_both",
            "eligible_f_c4ii",
        ],
    )
    _write_csv(
        output_dir / "f_c4ii_probe_local_minimum_failures.csv",
        local_probe,
        [
            "mechanism_label",
            "structural_class",
            "mechanism_rank",
            "success_rate",
            "timeout_rate",
            "progress_mean",
            "fallback_or_degraded_count",
            "eligible_f_c4ii",
        ],
    )
    _write_csv(
        output_dir / "geometry_vs_mechanism_agreement.csv",
        agreement,
        [
            "geometry_bucket",
            "mechanism_label",
            "planner_key",
            "structural_class",
            "old_geometry_rank",
            "new_mechanism_rank",
            "agreement_status",
            "conclusion_survives",
            "caveat",
        ],
    )
    # Fail-closed guard: a successful build must not silently produce a header-only agreement table.
    check_geometry_export_nonempty(output_dir / "geometry_vs_mechanism_agreement.csv")
    # Per-scenario geometry mapping: always written from all loaded rows (with or without mechanism
    # labels) so the scenario → bucket structure is independently re-derivable.
    _write_csv(
        output_dir / "scenario_geometry_bucket_mapping.csv",
        _extract_scenario_geometry_mapping(enriched),
        list(_SCENARIO_GEOMETRY_MAPPING_FIELDS),
    )
    if not (output_dir / "missing_instrumentation.json").exists():
        _write_json(
            output_dir / "missing_instrumentation.json",
            {"status": "not_applicable", "missing_row_count": 0},
        )
    _write_readmes(output_dir, status)
    _write_sha256sums(output_dir)
    return metadata


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm-root", type=Path, required=True)
    parser.add_argument("--extended-root", type=Path, required=True)
    parser.add_argument("--job13175-packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-at", default="now")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the issue #4206 evidence builder CLI."""
    args = _parse_args(argv or sys.argv[1:])
    generated_at = (
        datetime.now(UTC).isoformat(timespec="seconds")
        if args.generated_at == "now"
        else str(args.generated_at)
    )
    try:
        summary = build_packet(
            config_path=args.config,
            confirm_root=args.confirm_root,
            extended_root=args.extended_root,
            job13175_packet=args.job13175_packet,
            output_dir=args.output_dir,
            generated_at=generated_at,
        )
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"status: {summary['status']}")
        print(f"output_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
