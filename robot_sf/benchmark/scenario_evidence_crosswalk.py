"""Versioned scenario-evidence crosswalk (issue #5602).

A single deterministic join that downstream benchmark reports and exemplar
selectors can query instead of reconstructing taxonomy, predicate availability,
and trace/example availability manually.

The crosswalk stops at **benchmark-evidence metadata**. It must not encode
private manuscript claims and must not imply a causal failure mechanism from
geometry alone. Two contract rules from the issue are load-bearing:

1. Geometry groups and *validated* mechanisms live in **separate fields** with
   **separate provenance**. Geometry is descriptive only; a trace-verified
   mechanism is never derived from geometry (see
   :mod:`robot_sf.benchmark.failure_mechanism_taxonomy`).
2. Predicate availability is consumed from the #5593 predicate-export lane when
   present; until that lane lands, predicate fields are explicit ``unavailable``
   and taxonomy-motivated predicates are listed as ``motivated_not_exported``
   rather than re-derived from motivation text (issue stop rule).

The builder is fail-closed: duplicate scenario ids, empty/stale config hashes,
unknown predicate schema versions, broken artifact references, and
provenance-incomplete eligibility are all rejected rather than silently patched.

**Legacy schema compatibility (issue #5935).** The crosswalk accepts the legacy
``safety_predicate.late_evasive.v1`` schema carried by existing durable campaign
bundles (e.g. the issue4206 trace-capable rerun) and retains the exact source
schema version as provenance rather than normalizing it away. The v1->v2 bump
(PR #5063) is documented as an *additive* telemetry-instrumentation change
(``latency_unavailable_reason`` plus a seconds-valued latency) with ``no
benchmark metric semantics change``; the crosswalk only records predicate
identity + schema provenance + status, so v1 and v2 are compatible for this
public contract. The current/motivated version referenced by
``motivated_not_exported`` records remains ``v2`` (see
:data:`KNOWN_PREDICATE_SCHEMAS`), and the full accepted set is
:data:`SUPPORTED_PREDICATE_SCHEMAS`. This mirrors the producer
(:mod:`robot_sf.benchmark.trace_predicate_export`), which already accepts both
versions and preserves exact source provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.benchmark.safety_predicates import (
    LATE_EVASIVE_PREDICATE_SCHEMA,
    OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA,
    OSCILLATORY_PREDICATE_SCHEMA,
)
from robot_sf.benchmark.utils import _config_hash
from robot_sf.common.json_pointer import json_pointer as _render_json_pointer

SCHEMA_VERSION = "scenario_evidence_crosswalk.v1"

#: The legacy late-evasive schema carried by existing durable campaign bundles
#: (e.g. the issue4206 trace-capable rerun). The producer (`trace_predicate_export`)
#: accepts both v1 and v2; the v1->v2 bump (#5063) is documented as an *additive*
#: telemetry-instrumentation change (``latency_unavailable_reason`` and a
#: seconds-valued latency) with ``no benchmark metric semantics change``. The
#: crosswalk only records predicate identity + schema provenance + status, so v1
#: and v2 are compatible for this public contract; we retain the exact source
#: version rather than normalizing, to preserve provenance (issue #5935).
LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA = "safety_predicate.late_evasive.v1"

#: Predicate schema tags motivated by the failure taxonomy (#5593 lane). Each
#: entry maps a predicate name to its *current/motivated* schema version, which
#: is the version referenced by ``motivated_not_exported`` records.
KNOWN_PREDICATE_SCHEMAS: dict[str, str] = {
    "oscillatory_control": OSCILLATORY_PREDICATE_SCHEMA,
    "late_evasive": LATE_EVASIVE_PREDICATE_SCHEMA,
    "occlusion_near_miss": OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA,
}

#: Every predicate schema version the crosswalk accepts as valid provenance.
#: This is a superset of :data:`KNOWN_PREDICATE_SCHEMAS` values: it also retains
#: legacy versions that durable campaign bundles carry (issue #5935). Any other
#: version in a supplied export is rejected as an unknown predicate version.
SUPPORTED_PREDICATE_SCHEMAS: dict[str, frozenset[str]] = {
    "oscillatory_control": frozenset({OSCILLATORY_PREDICATE_SCHEMA}),
    "late_evasive": frozenset(
        {LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA, LATE_EVASIVE_PREDICATE_SCHEMA}
    ),
    "occlusion_near_miss": frozenset({OCCLUSION_NEAR_MISS_PREDICATE_SCHEMA}),
}

PREDICATE_EXPORT_SCHEMA_VERSION = "trace_predicate_export.v1"

EXCLUSION_REASON_PREDICATE_UNAVAILABLE = "predicate_export_unavailable"
EXCLUSION_REASON_NO_EVIDENCE_BUNDLE = "no_evidence_bundle_provided"
EXCLUSION_REASON_UNREADABLE_EPISODE_ARTIFACT = "unreadable_episode_artifact"

SCHEMA_FILE = Path(__file__).with_name("schemas") / "scenario_evidence_crosswalk.v1.json"


class ScenarioEvidenceCrosswalkError(ValueError):
    """Raised when a defensible crosswalk cannot be built or validated."""


def _git_sha_short(length: int = 7) -> str:
    """Return short git SHA for the current HEAD, or ``unknown``."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", f"--short={length}", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode("utf-8")
            .strip()
        )
        return sha or "unknown"
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ):
        return "unknown"


def _scenario_id(scenario: Mapping[str, Any], index: int) -> str:
    """Return the stable scenario identifier used across benchmark views.

    Returns:
        The scenario id string, or a deterministic fallback like ``scenario_NNN``.
    """
    for field in ("name", "id", "scenario_id"):
        value = scenario.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"scenario_{index:03d}"


def _metadata(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow metadata mapping for a scenario."""
    value = scenario.get("metadata")
    return dict(value) if isinstance(value, Mapping) else {}


def _first_nonempty_str(*values: Any) -> str | None:
    """Return the first non-empty string among ``values`` or ``None``."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _scenario_family(scenario: Mapping[str, Any]) -> str:
    """Return the scenario family, preferring explicit family fields."""
    meta = _metadata(scenario)
    return (
        _first_nonempty_str(
            meta.get("scenario_family"),
            scenario.get("scenario_family"),
            meta.get("family"),
            meta.get("archetype"),
        )
        or "unknown"
    )


def _interaction_class(scenario: Mapping[str, Any]) -> str:
    """Return the interaction class when explicitly declared, else ``unknown``."""
    meta = _metadata(scenario)
    return (
        _first_nonempty_str(meta.get("interaction_class"), scenario.get("interaction_class"))
        or "unknown"
    )


def _geometry_group(scenario: Mapping[str, Any]) -> str:
    """Return a *descriptive* geometry/topology group label.

    Geometry groups are descriptive topology labels only; they never substitute
    for a trace-verified mechanism. Prefer an explicit ``geometry_group`` field,
    otherwise fall back to the archetype as a topology hint.
    """
    meta = _metadata(scenario)
    return (
        _first_nonempty_str(
            meta.get("geometry_group"), scenario.get("geometry_group"), meta.get("archetype")
        )
        or "unknown"
    )


_HAZARD_SOURCE_KEYS = (
    "expected_failure_modes",
    "target_failure_mode",
    "primary_capability",
    "capability_tags",
    "hazard_classes",
)


def _hazard_tags(scenario: Mapping[str, Any]) -> list[str]:
    """Collect declared hazard/capability/stress tags from scenario metadata.

    Returns:
        Sorted unique tag strings.
    """
    meta = _metadata(scenario)
    tags: list[str] = []
    for key in _HAZARD_SOURCE_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            tags.append(value.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    tags.append(item.strip())
    # Platform semantics hazard regions (descriptive only).
    regions = scenario.get("platform_semantics", {})
    if isinstance(regions, Mapping):
        for region in regions.get("regions", []) or []:
            if isinstance(region, Mapping) and region.get("kind") == "hazard":
                rid = region.get("id")
                if isinstance(rid, str) and rid.strip():
                    tags.append(f"platform:{rid.strip()}")
    return sorted(set(tags))


def _source_config_hash(scenario: Mapping[str, Any]) -> str:
    """Return a stable deterministic hash of the scenario config source."""
    return _config_hash(dict(scenario))


def _strict_int(value: Any) -> int | None:
    """Return an integer seed only when the input is exactly an integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _seeds(scenario: Mapping[str, Any]) -> list[int]:
    """Return the deterministic seed list for a scenario."""
    raw = scenario.get("seeds")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        seeds = [seed for value in raw if (seed := _strict_int(value)) is not None]
        if seeds:
            return seeds
    return []


def _normalize_predicate_export(export: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Normalize a #5593 predicate-export manifest into a per-scenario map.

    Returns:
        Mapping from scenario id to a record with ``exported`` (name -> schema
        version) and ``predicate_status`` (name -> status string).
    """
    if export is None:
        return {}
    if not isinstance(export, Mapping):
        raise ScenarioEvidenceCrosswalkError("predicate export must be a mapping")
    schema = export.get("schema_version")
    if schema is not None and schema != PREDICATE_EXPORT_SCHEMA_VERSION:
        raise ScenarioEvidenceCrosswalkError(
            f"predicate export schema mismatch: expected {PREDICATE_EXPORT_SCHEMA_VERSION!r}, "
            f"got {schema!r}"
        )
    per_scenario: dict[str, dict[str, Any]] = {}
    rows = export.get("rows")
    if not isinstance(rows, list):
        return per_scenario
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        sid = _first_nonempty_str(row.get("scenario_id"), row.get("scenario"))
        if sid is None:
            continue
        entry = per_scenario.setdefault(sid, {"exported": {}, "predicate_status": {}})
        _absorb_predicate_row(row, entry)
    return per_scenario


def _absorb_predicate_row(row: Mapping[str, Any], entry: dict[str, Any]) -> None:
    """Fold one predicate-export row's predicates into ``entry`` dicts."""
    for pred in row.get("predicates", []) or []:
        if not isinstance(pred, Mapping):
            continue
        name = pred.get("predicate")
        if not isinstance(name, str) or not name:
            continue
        entry["exported"][name] = pred.get("schema_version")
        entry["predicate_status"][name] = pred.get("status", "ok")


def _is_supported_predicate_schema(name: object, version: object) -> bool:
    """Return whether a predicate name and schema version form a supported pair."""
    if version is None:
        return True
    return (
        isinstance(name, str)
        and isinstance(version, str)
        and version in SUPPORTED_PREDICATE_SCHEMAS.get(name, frozenset())
    )


def _predicate_section(
    scenario_id: str,
    export: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the predicate availability section for one scenario.

    Consumes a #5593 export when supplied; otherwise marks everything explicit
    ``unavailable`` and lists taxonomy-motivated predicates as
    ``motivated_not_exported`` (never inferred from motivation text).

    Returns:
        Predicate availability section dict.
    """
    exported: list[dict[str, Any]] = []
    missing_degraded: list[dict[str, Any]] = []
    motivated_not_exported: list[dict[str, Any]] = []

    entry = export.get(scenario_id)
    if entry is None:
        for name, version in KNOWN_PREDICATE_SCHEMAS.items():
            motivated_not_exported.append(
                {
                    "predicate": name,
                    "motivated_by": "failure_taxonomy",
                    "schema_version": version,
                    "status": "motivated_not_exported",
                }
            )
        return {
            "export_status": "unavailable",
            "export_status_reason": EXCLUSION_REASON_PREDICATE_UNAVAILABLE,
            "exported_predicates": exported,
            "missing_or_degraded_predicates": missing_degraded,
            "motivated_not_exported_predicates": motivated_not_exported,
        }

    for name, version in entry["exported"].items():
        if not _is_supported_predicate_schema(name, version):
            raise ScenarioEvidenceCrosswalkError(
                f"scenario {scenario_id}: unknown predicate schema version {version!r} for "
                f"predicate {name!r}"
            )
        status = entry["predicate_status"].get(name, "ok")
        record = {"predicate": name, "schema_version": version}
        if status in {"degraded", "missing", "unavailable", "fallback"}:
            record["status"] = status
            missing_degraded.append(record)
        else:
            exported.append(record)

    exported_names = {record["predicate"] for record in exported}
    for name, version in KNOWN_PREDICATE_SCHEMAS.items():
        if name not in exported_names and name not in {m["predicate"] for m in missing_degraded}:
            motivated_not_exported.append(
                {
                    "predicate": name,
                    "motivated_by": "failure_taxonomy",
                    "schema_version": version,
                    "status": "motivated_not_exported",
                }
            )

    return {
        "export_status": "available",
        "export_status_reason": None,
        "exported_predicates": exported,
        "missing_or_degraded_predicates": missing_degraded,
        "motivated_not_exported_predicates": motivated_not_exported,
    }


def _evidence_section(
    scenario_id: str,
    evidence_catalog: Mapping[str, Any] | None,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Build the evidence/replay eligibility section for one scenario.

    Fails closed: without a supplied evidence catalog, eligibility is
    ``excluded`` with a concrete reason rather than fabricated ids.

    Returns:
        Evidence section dict (eligibility, exclusion reason, and artifact ids).
    """
    if evidence_catalog is None:
        return {
            "eligibility": "excluded",
            "exclusion_reason": EXCLUSION_REASON_NO_EVIDENCE_BUNDLE,
            "validated_mechanism": None,
            "validated_mechanism_provenance": None,
            "trace_ids": [],
            "replay_ids": [],
            "generated_candidate_ids": [],
            "case_capsule_ids": [],
        }

    entry = evidence_catalog.get(scenario_id)
    if not isinstance(entry, Mapping):
        return {
            "eligibility": "excluded",
            "exclusion_reason": "scenario_absent_from_evidence_catalog",
            "validated_mechanism": None,
            "validated_mechanism_provenance": None,
            "trace_ids": [],
            "replay_ids": [],
            "generated_candidate_ids": [],
            "case_capsule_ids": [],
        }

    eligibility = entry.get("eligibility", "excluded")
    exclusion_reason = entry.get("exclusion_reason")
    if eligibility != "eligible":
        if not exclusion_reason:
            raise ScenarioEvidenceCrosswalkError(
                f"scenario {scenario_id}: non-eligible evidence must carry an exclusion_reason"
            )

    validated = entry.get("validated_mechanism")
    if validated is not None:
        raise ScenarioEvidenceCrosswalkError(
            f"scenario {scenario_id}: validated_mechanism must be supplied by a validated causal "
            f"report, not the scenario crosswalk (geometry-only rejection)"
        )

    ids_fields = ("trace_ids", "replay_ids", "generated_candidate_ids", "case_capsule_ids")
    resolved = _resolve_evidence_ids(scenario_id, entry, ids_fields, artifact_root)

    return {
        "eligibility": eligibility,
        "exclusion_reason": exclusion_reason if eligibility != "eligible" else None,
        "validated_mechanism": None,
        "validated_mechanism_provenance": None,
        "trace_ids": resolved["trace_ids"],
        "replay_ids": resolved["replay_ids"],
        "generated_candidate_ids": resolved["generated_candidate_ids"],
        "case_capsule_ids": resolved["case_capsule_ids"],
    }


def _resolve_evidence_ids(
    scenario_id: str,
    entry: Mapping[str, Any],
    ids_fields: Sequence[str],
    artifact_root: Path | None,
) -> dict[str, list[str]]:
    """Normalize evidence artifact id lists and fail-closed on broken refs.

    Returns:
        Mapping from each id field to its list of string references.
    """
    resolved: dict[str, list[str]] = {}
    for field in ids_fields:
        values = entry.get(field, []) or []
        if not isinstance(values, list):
            raise ScenarioEvidenceCrosswalkError(
                f"scenario {scenario_id}: {field} must be a list when present"
            )
        resolved[field] = [str(v) for v in values]

    if artifact_root is not None:
        trusted_root = _trusted_artifact_root(scenario_id, artifact_root)
        for field, ids in resolved.items():
            for ref in ids:
                _validate_artifact_reference(
                    scenario_id,
                    field,
                    ref,
                    artifact_root=artifact_root,
                    trusted_root=trusted_root,
                )
    return resolved


def _trusted_artifact_root(scenario_id: str, artifact_root: Path) -> Path:
    """Resolve the trusted artifact root or raise a structured blocker.

    Returns:
        The resolved trusted artifact root.
    """
    try:
        return artifact_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ScenarioEvidenceCrosswalkError(
            f"scenario {scenario_id}: {EXCLUSION_REASON_UNREADABLE_EPISODE_ARTIFACT} "
            f"(broken artifact reference): trusted root {str(artifact_root)!r} is unreadable"
        ) from exc


def _validate_artifact_reference(
    scenario_id: str,
    field: str,
    ref: str,
    *,
    artifact_root: Path,
    trusted_root: Path,
) -> None:
    """Reject unsafe artifact references before they can be consumed."""
    ref_path = Path(ref)
    reason: str | None = None
    if ref_path.is_absolute() or ".." in ref_path.parts:
        reason = "absolute or parent traversal"
    else:
        candidate = artifact_root / ref_path
        current = artifact_root
        try:
            for part in ref_path.parts:
                current /= part
                if current.is_symlink():
                    reason = "symlink component"
                    break
            resolved_candidate = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            reason = reason or "path resolution failed"
        else:
            if reason is None and not resolved_candidate.is_relative_to(trusted_root):
                reason = "outside trusted root"
            if reason is None and not resolved_candidate.is_file():
                reason = "not a regular file"
    if reason is not None:
        raise ScenarioEvidenceCrosswalkError(
            f"scenario {scenario_id}: {EXCLUSION_REASON_UNREADABLE_EPISODE_ARTIFACT} "
            f"(broken artifact reference) for {field} {ref!r}: {reason}"
        )


def build_scenario_evidence_crosswalk(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    source: str,
    predicate_export: Mapping[str, Any] | None = None,
    evidence_catalog: Mapping[str, Any] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic scenario-evidence crosswalk from canonical scenarios.

    Args:
        scenarios: Canonical scenario dicts (one per release cell).
        source: Human-readable source identifier (matrix path).
        predicate_export: Optional #5593 predicate-export manifest.
        evidence_catalog: Optional mapping from scenario id to evidence/eligibility.
        artifact_root: Optional root used to fail-closed on broken artifact refs.

    Returns:
        A schema-versioned crosswalk dict.

    Raises:
        ScenarioEvidenceCrosswalkError: On duplicate ids, empty config hashes,
            unknown predicate versions, or broken artifact references.
    """
    if not scenarios:
        raise ScenarioEvidenceCrosswalkError("crosswalk requires at least one scenario")

    export = _normalize_predicate_export(predicate_export)
    scenario_entries = [dict(scenario) for scenario in scenarios]

    seen: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenario_entries):
        scenario_id = _scenario_id(scenario, index)
        if scenario_id in seen:
            raise ScenarioEvidenceCrosswalkError(
                f"duplicate scenario id {scenario_id!r} at indices {seen[scenario_id]} and {index}"
            )
        seen[scenario_id] = index

        source_hash = _source_config_hash(scenario)
        if not source_hash or source_hash == "0" * len(source_hash):
            raise ScenarioEvidenceCrosswalkError(
                f"scenario {scenario_id}: empty/stale source config hash"
            )

        family = _scenario_family(scenario)
        geometry = _geometry_group(scenario)
        if geometry == family:
            # Defensive guard: geometry group must not silently equal the family
            # in a way that implies a mechanism; keep them labelled separately.
            geometry = f"{geometry} (geometry_only)"

        row = {
            "scenario_id": scenario_id,
            "source_config_hash": source_hash,
            "scenario_family": family,
            "interaction_class": _interaction_class(scenario),
            "map_id": _first_nonempty_str(
                scenario.get("map_id"), scenario.get("map_file"), scenario.get("map")
            )
            or "unknown",
            "geometry_group": geometry,
            "geometry_group_provenance": "config_metadata",
            "geometry_group_note": "descriptive topology label; not a causal mechanism",
            "hazard_tags": _hazard_tags(scenario),
            "hazard_tag_source": "config_metadata",
            "seeds": _seeds(scenario),
            "predicate_availability": _predicate_section(scenario_id, export),
            "evidence": _evidence_section(
                scenario_id, evidence_catalog, artifact_root=artifact_root
            ),
        }
        rows.append(row)

    rows.sort(key=lambda r: r["scenario_id"])

    json_text = json.dumps(
        {"schema_version": SCHEMA_VERSION, "rows": rows},
        sort_keys=True,
        separators=(",", ":"),
    )
    content_sha256 = hashlib.sha256(json_text.encode("utf-8")).hexdigest()

    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repo_commit": _git_sha_short(),
        "claim_boundary": (
            "Benchmark-evidence metadata join only. Geometry groups are descriptive topology "
            "labels and never imply a causal failure mechanism; predicate availability is "
            "consumed from the #5593 export lane when present and is explicit 'unavailable' "
            "otherwise; validated mechanisms require a validated causal report and are not "
            "derived here. No private manuscript claims are encoded."
        ),
        "predicate_export_consumed": predicate_export is not None,
        "evidence_catalog_consumed": evidence_catalog is not None,
        "summary": {
            "scenario_count": len(rows),
            "eligible_scenarios": sum(
                1 for r in rows if r["evidence"]["eligibility"] == "eligible"
            ),
            "excluded_scenarios": sum(
                1 for r in rows if r["evidence"]["eligibility"] != "eligible"
            ),
            "predicate_export_available": sum(
                1 for r in rows if r["predicate_availability"]["export_status"] == "available"
            ),
            "predicate_unavailable": sum(
                1 for r in rows if r["predicate_availability"]["export_status"] == "unavailable"
            ),
        },
        "content_sha256": content_sha256,
        "rows": rows,
    }


def _load_schema() -> dict[str, Any]:
    """Load the crosswalk JSON Schema.

    Returns:
        Parsed JSON Schema dictionary.
    """
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_scenario_evidence_crosswalk(crosswalk: Mapping[str, Any]) -> list[str]:
    """Validate a crosswalk against its JSON Schema and semantic gates.

    Returns:
        List of validation error strings; empty when valid.
    """
    errors: list[str] = []
    validator = Draft202012Validator(_load_schema())
    errors.extend(
        f"{json_pointer(list(err.absolute_path))}: {err.message}"
        for err in sorted(
            validator.iter_errors(dict(crosswalk)), key=lambda e: list(e.absolute_path)
        )
    )
    if crosswalk.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
            f"got {crosswalk.get('schema_version')!r}"
        )

    rows = crosswalk.get("rows")
    if not isinstance(rows, list):
        errors.append("crosswalk.rows must be a list")
        return errors

    seen_ids: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"rows[{index}]: not a mapping")
            continue
        sid = row.get("scenario_id")
        if sid in seen_ids:
            errors.append(f"rows[{index}]: duplicate scenario_id {sid!r}")
        seen_ids[sid] = index
        _validate_crosswalk_row(row, index, sid, errors)

    return errors


def _validate_crosswalk_row(
    row: Mapping[str, Any],
    index: int,
    sid: object,
    errors: list[str],
) -> None:
    """Append semantic-validation errors for a single crosswalk row."""
    if not row.get("source_config_hash"):
        errors.append(f"rows[{index}] ({sid}): empty source_config_hash")

    pred = row.get("predicate_availability")
    if isinstance(pred, Mapping):
        for rec in pred.get("exported_predicates", []) or []:
            name = rec.get("predicate")
            version = rec.get("schema_version")
            if not _is_supported_predicate_schema(name, version):
                errors.append(
                    f"rows[{index}] ({sid}): unknown predicate schema_version {version!r} for "
                    f"predicate {name!r}"
                )

    evidence = row.get("evidence")
    if isinstance(evidence, Mapping):
        if evidence.get("eligibility") != "eligible" and not evidence.get("exclusion_reason"):
            errors.append(f"rows[{index}] ({sid}): non-eligible evidence missing exclusion_reason")
        if evidence.get("validated_mechanism") is not None:
            errors.append(
                f"rows[{index}] ({sid}): validated_mechanism must not be set in crosswalk"
            )


def json_pointer(parts: Sequence[Any]) -> str:
    """Render a JSON-path list as a slash-prefixed pointer (or '' for root).

    Returns:
        The RFC6901 JSON pointer string, or ``""`` for the root path.
    """
    return _render_json_pointer(parts)


def crosswalk_markdown(crosswalk: Mapping[str, Any]) -> str:
    """Render a compact Markdown coverage report for the crosswalk.

    Returns:
        Markdown report body.
    """
    summary = crosswalk.get("summary", {})
    lines = [
        "# Scenario-Evidence Crosswalk",
        "",
        f"- Source: `{crosswalk.get('source', '')}`",
        f"- Schema: `{crosswalk.get('schema_version', SCHEMA_VERSION)}`",
        f"- Repo commit: `{crosswalk.get('repo_commit', 'unknown')}`",
        f"- Content SHA-256: `{crosswalk.get('content_sha256', '')}`",
        "",
        "## Summary",
        "",
        f"- Scenarios: {int(summary.get('scenario_count', 0))}",
        f"- Eligible (evidence): {int(summary.get('eligible_scenarios', 0))}",
        f"- Excluded (evidence): {int(summary.get('excluded_scenarios', 0))}",
        f"- Predicate export available: {int(summary.get('predicate_export_available', 0))}",
        f"- Predicate unavailable: {int(summary.get('predicate_unavailable', 0))}",
        "",
        "## Rows",
        "",
        "| Scenario | Family | Geometry (descriptive) | Interaction | Predicate export | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in crosswalk.get("rows", []) or []:
        pred = row.get("predicate_availability", {})
        pred_status = pred.get("export_status", "unknown")
        evidence = row.get("evidence", {})
        eligibility = evidence.get("eligibility", "unknown")
        lines.append(
            "| "
            f"{row.get('scenario_id', '')} | "
            f"{row.get('scenario_family', '')} | "
            f"{row.get('geometry_group', '')} | "
            f"{row.get('interaction_class', '')} | "
            f"{pred_status} | "
            f"{eligibility} |"
        )
    return "\n".join(lines) + "\n"


def write_scenario_evidence_crosswalk(
    crosswalk: Mapping[str, Any],
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    csv_path: Path | None = None,
) -> None:
    """Write crosswalk artifacts (JSON, Markdown, CSV) when paths are given."""
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(crosswalk, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(crosswalk_markdown(crosswalk), encoding="utf-8")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "scenario_id",
                    "source_config_hash",
                    "scenario_family",
                    "interaction_class",
                    "geometry_group",
                    "geometry_group_provenance",
                    "hazard_tags",
                    "predicate_export_status",
                    "evidence_eligibility",
                    "exclusion_reason",
                    "validated_mechanism",
                    "trace_ids",
                    "replay_ids",
                    "generated_candidate_ids",
                    "case_capsule_ids",
                ]
            )
            for row in crosswalk.get("rows", []) or []:
                pred = row.get("predicate_availability", {})
                evidence = row.get("evidence", {})
                writer.writerow(
                    [
                        row.get("scenario_id", ""),
                        row.get("source_config_hash", ""),
                        row.get("scenario_family", ""),
                        row.get("interaction_class", ""),
                        row.get("geometry_group", ""),
                        row.get("geometry_group_provenance", ""),
                        ";".join(row.get("hazard_tags", []) or []),
                        pred.get("export_status", ""),
                        evidence.get("eligibility", ""),
                        evidence.get("exclusion_reason") or "",
                        evidence.get("validated_mechanism") or "",
                        ";".join(evidence.get("trace_ids", []) or []),
                        ";".join(evidence.get("replay_ids", []) or []),
                        ";".join(evidence.get("generated_candidate_ids", []) or []),
                        ";".join(evidence.get("case_capsule_ids", []) or []),
                    ]
                )


__all__ = [
    "KNOWN_PREDICATE_SCHEMAS",
    "LEGACY_LATE_EVASIVE_PREDICATE_SCHEMA",
    "PREDICATE_EXPORT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SUPPORTED_PREDICATE_SCHEMAS",
    "ScenarioEvidenceCrosswalkError",
    "build_scenario_evidence_crosswalk",
    "crosswalk_markdown",
    "json_pointer",
    "validate_scenario_evidence_crosswalk",
    "write_scenario_evidence_crosswalk",
]
