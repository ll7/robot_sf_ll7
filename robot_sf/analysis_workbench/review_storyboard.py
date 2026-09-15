"""SREV-07 review-storyboard component: ranked candidates plus storyboard spec.

This module owns the SREV-07 leaf surface only: a ``run(request)`` adapter plus
a standalone CLI that consumes the SREV-01 shared contracts, ranks bundle
episodes with stable tie-breaking, and emits a deterministic
``visualization-spec.v1`` document with explicit source and override
provenance.  The component is diagnostic tooling: its fixture outputs are not
scientific, benchmark, safety, or paper-facing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
    review_bundle_from_dict,
    visualization_spec_from_dict,
)

COMPONENT_ID = "srev07-review-storyboard"
COMPONENT_VERSION = "1.0.0"

REQUIRED_CAPABILITIES = ("review-bundle",)
OPTIONAL_CAPABILITIES = ("exemplar-scores", "event-index")

OUTPUT_SPEC_FILENAME = "storyboard-spec.json"
OUTPUT_OVERRIDES_FILENAME = "override-provenance.json"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

TIE_BREAK_METHOD = "score-descending-then-episode-id-ascending"
EVIDENCE_BOUNDARY = "diagnostic_only"
MISSING_CAPABILITY_SCHEMA_VERSION = "missing-capability-report.v1"
EXEMPLAR_SCORES_SCHEMA_VERSION = "srev07-exemplar-scores.v1"
CANONICAL_SCORE_ADAPTER_VERSION = "trace-exemplar-interest.report-adapter.v1"

_DESCRIPTOR_DOCUMENT: dict[str, Any] = {
    "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": list(REQUIRED_CAPABILITIES),
    "optional_capabilities": list(OPTIONAL_CAPABILITIES),
    "output_types": [
        "visualization-spec.v1",
        "override-provenance.v1",
        "missing-capability-report.v1",
    ],
}

# Validate the public descriptor at import time so descriptor drift fails
# before a caller can invoke the component.
_DESCRIPTOR = component_descriptor_from_dict(_DESCRIPTOR_DOCUMENT)


@dataclass(frozen=True, slots=True)
class _StoryboardComponentResult(ComponentResult):
    """Local result envelope carrying the shared schema version field."""

    artifacts: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = COMPONENT_RESULT_SCHEMA_VERSION


@dataclass
class _Candidate:
    """One ranked storyboard candidate."""

    episode_id: str
    score: float | None = None
    score_source: str = "unavailable"
    pinned: bool = False
    excluded: bool = False


def _result(**kwargs: Any) -> ComponentResult:
    """Construct a result with the versioned local API envelope.

    Returns:
        A result carrying ``component-result.v1`` as ``schema_version``.
    """
    if "artifacts" in kwargs:
        kwargs["artifacts"] = list(kwargs["artifacts"])
    if "diagnostics" in kwargs:
        kwargs["diagnostics"] = list(kwargs["diagnostics"])
    return _StoryboardComponentResult(**kwargs)


def descriptor() -> dict[str, Any]:
    """Return this component's versioned, self-contained descriptor."""
    return json.loads(json.dumps(_DESCRIPTOR_DOCUMENT, allow_nan=False))


def result_document(result: ComponentResult) -> dict[str, Any]:
    """Serialize and validate the versioned result envelope for API/CLI use.

    Returns:
        A JSON-serializable, schema-validated result mapping.
    """
    payload: dict[str, Any] = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "request_id": result.request_id,
        "component_id": result.component_id,
        "status": result.status,
        "artifacts": [dict(item) for item in result.artifacts],
        "diagnostics": [dict(item) for item in result.diagnostics],
        "provenance": dict(result.provenance),
        "reason": result.reason,
    }
    component_result_from_dict(payload)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Atomically write strict JSON and return the written-byte digest.

    Returns:
        SHA-256 digest of the exact UTF-8 bytes written to ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return _sha256_bytes(text.encode("utf-8"))


def _stage_outputs(output_dir: Path, payloads: dict[str, Any]) -> dict[str, str]:
    """Write all sidecars into one new output directory transactionally.

    Returns:
        Mapping from sidecar filename to its exact-byte SHA-256 digest.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    committed = False
    try:
        digests = {
            filename: _write_json(staging / filename, payloads[filename])
            for filename in sorted(payloads)
        }
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"output collision: already exists: {output_dir}")
        staging.replace(output_dir)
        committed = True
        return digests
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)


def _canonical_digest(payload: Any) -> str:
    """Return the stable logical digest of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a failure reason when the request demands a newer component."""
    minimum = config.get("min_component_version")
    if minimum is None:
        return None
    try:
        wanted = int(str(minimum).split(".", maxsplit=1)[0])
        ours = int(COMPONENT_VERSION.split(".", maxsplit=1)[0])
    except ValueError:
        return f"incompatible_component_version: malformed min_component_version: {minimum!r}"
    if wanted > ours:
        return f"incompatible_component_version: request needs v{wanted}, component is v{ours}"
    return None


def _reject_not_applicable(request: ComponentRequest) -> ComponentResult | None:
    """Reject requests this component cannot serve.

    Returns:
        An unavailable or failed result, or ``None`` when the request applies.
    """
    if request.component_id != COMPONENT_ID:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"unsupported_component: {request.component_id}",
        )
    supported = set(_DESCRIPTOR.required_capabilities) | set(_DESCRIPTOR.optional_capabilities)
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(sorted(missing))}",
        )
    version_error = _check_version_compatible(request.config)
    if version_error is not None:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=version_error,
        )
    return None


def _resolve_under_root(path_value: str, root: Path) -> Path | None:
    """Return a path only when lexical and resolved containment both hold."""
    candidate = Path(path_value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    root_resolved = root.resolve()
    candidate_path = root / candidate
    try:
        candidate_path.resolve(strict=False).relative_to(root_resolved)
    except ValueError:
        return None
    return candidate_path


def _resolve_output_directory(root: Path, path_value: str) -> tuple[Path | None, str | None]:
    """Resolve an output directory without allowing traversal or symlink escape.

    Returns:
        The contained path and no error, or ``None`` and a stable rejection reason.
    """
    path = _resolve_under_root(path_value, root)
    if path is None:
        return None, f"unsafe_output_path: {path_value}"
    return path, None


def _resolve_source(path_value: str, base: Path) -> Path | None:
    """Resolve a source URI under ``base`` with symlink-aware containment.

    Returns:
        The lexical path when its resolved target remains contained, otherwise ``None``.
    """
    return _resolve_under_root(path_value, base)


def _ref_value(ref: Any, key: str, default: Any = "") -> Any:
    """Read a field from either a SourceRef or a validated bundle mapping.

    Returns:
        The requested field value or ``default`` when it is not present.
    """
    if hasattr(ref, key):
        return getattr(ref, key)
    if isinstance(ref, dict):
        return ref.get(key, default)
    return default


def _source_record(
    ref: Any,
    *,
    integrity_status: str,
    observed_sha256: str = "",
    episode_id: str | None = None,
    scope: str = "request",
) -> dict[str, Any]:
    """Build a deterministic, provenance-complete source record.

    Returns:
        A JSON-serializable source identity and integrity record.
    """
    record: dict[str, Any] = {
        "artifact_id": str(_ref_value(ref, "artifact_id")),
        "uri": str(_ref_value(ref, "uri")),
        "format": str(_ref_value(ref, "format")),
        "schema": str(_ref_value(ref, "schema")),
        "sha256": str(_ref_value(ref, "sha256")),
        "source_commit": str(_ref_value(ref, "source_commit")),
        "config_identity": str(_ref_value(ref, "config_identity")),
        "units": str(_ref_value(ref, "units")),
        "coordinate_frame": str(_ref_value(ref, "coordinate_frame")),
        "integrity_status": integrity_status,
        "scope": scope,
    }
    if observed_sha256:
        record["observed_sha256"] = observed_sha256
    if episode_id is not None:
        record["episode_id"] = episode_id
    return record


def _verify_reference(
    ref: Any,
    root: Path,
    *,
    episode_id: str | None = None,
    scope: str = "request",
) -> tuple[bytes | None, dict[str, Any], str | None]:
    """Read bytes, verify the declared digest, and report safe path status.

    Returns:
        Raw bytes when readable, the source record, and an optional diagnostic code.
    """
    artifact_id = str(_ref_value(ref, "artifact_id"))
    uri = str(_ref_value(ref, "uri"))
    path = _resolve_source(uri, root)
    if path is None:
        return (
            None,
            _source_record(ref, integrity_status="unsafe_path", episode_id=episode_id, scope=scope),
            f"{artifact_id}: source_uri_unsafe",
        )
    try:
        raw = path.read_bytes()
    except OSError:
        return (
            None,
            _source_record(ref, integrity_status="unavailable", episode_id=episode_id, scope=scope),
            f"{artifact_id}: source_unreadable",
        )
    observed = _sha256_bytes(raw)
    declared = str(_ref_value(ref, "sha256")).lower()
    if not declared:
        record = _source_record(
            ref,
            integrity_status="observed_unbound",
            observed_sha256=observed,
            episode_id=episode_id,
            scope=scope,
        )
        return raw, record, f"{artifact_id}: source_digest_missing"
    if declared != observed:
        record = _source_record(
            ref,
            integrity_status="digest_mismatch",
            observed_sha256=observed,
            episode_id=episode_id,
            scope=scope,
        )
        return raw, record, f"{artifact_id}: source_digest_mismatch"
    return (
        raw,
        _source_record(
            ref,
            integrity_status="verified",
            observed_sha256=observed,
            episode_id=episode_id,
            scope=scope,
        ),
        None,
    )


def _read_json(
    ref: Any, root: Path, *, scope: str = "request", episode_id: str | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    """Read one JSON object and return its provenance record and diagnostic.

    Returns:
        Parsed object when valid, its source record, and an optional diagnostic code.
    """
    raw, record, problem = _verify_reference(ref, root, episode_id=episode_id, scope=scope)
    if raw is None:
        return None, record, problem
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        record = {**record, "content_status": "invalid_json"}
        return None, record, f"{record['artifact_id']}: source_unreadable"
    if not isinstance(payload, dict):
        record = {**record, "content_status": "not_json_object"}
        return None, record, f"{record['artifact_id']}: source_not_json_object"
    return payload, {**record, "content_status": "json_object"}, problem


def _finite_number(value: Any) -> float | None:
    """Return a finite float, or None for malformed input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _rank_candidates(
    episode_ids: list[str],
    scores: dict[str, float],
    pins: list[str],
    excludes: list[str],
    score_format: str = "unavailable",
) -> tuple[list[_Candidate], list[str]]:
    """Rank candidates with stable tie-breaking and explicit overrides.

    Returns:
        Kept candidates in deterministic order and any override diagnostics.
    """
    diagnostics: list[str] = []
    known = set(episode_ids)
    pin_set = set(pins)
    exclude_set = set(excludes)
    for pinned in pins:
        if pinned not in known:
            diagnostics.append(f"override_pin_unknown:{pinned}")
    for excluded in excludes:
        if excluded not in known:
            diagnostics.append(f"override_exclude_unknown:{excluded}")
    candidates = [
        _Candidate(
            episode_id=episode_id,
            score=scores.get(episode_id),
            score_source=score_format if episode_id in scores else "unavailable",
            pinned=episode_id in pin_set,
            excluded=episode_id in exclude_set,
        )
        for episode_id in episode_ids
    ]
    kept = [candidate for candidate in candidates if not candidate.excluded]
    if excludes:
        diagnostics.append(f"excluded_episodes:{','.join(sorted(exclude_set & known))}")
    kept.sort(
        key=lambda candidate: (
            not candidate.pinned,
            -(candidate.score if candidate.score is not None else float("-inf")),
            candidate.episode_id,
        )
    )
    return kept, diagnostics


def _parse_overrides(
    config: dict[str, Any],
    bundle_doc: dict[str, Any],
    bundle_source_id: str,
    diagnostics: list[str],
) -> dict[str, Any]:
    """Parse overrides only when they are bound to the current bundle digest.

    Returns:
        Pin/exclude state, freshness, and declared/observed source digests.
    """
    raw = config.get("overrides", {})
    if not isinstance(raw, dict):
        diagnostics.append("overrides_malformed:ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": "",
            "observed_source_digest": _canonical_bundle_digest(bundle_doc),
            "source_artifact_id": bundle_source_id,
        }
    pins_raw = raw.get("pin", [])
    excludes_raw = raw.get("exclude", [])
    if not isinstance(pins_raw, list) or not isinstance(excludes_raw, list):
        diagnostics.append("overrides_malformed:ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": str(raw.get("source_digest", "")),
            "observed_source_digest": _canonical_bundle_digest(bundle_doc),
            "source_artifact_id": bundle_source_id,
        }
    pins = [item for item in pins_raw if isinstance(item, str)]
    excludes = [item for item in excludes_raw if isinstance(item, str)]
    observed = _canonical_bundle_digest(bundle_doc)
    declared = str(raw.get("source_digest", ""))
    has_overrides = bool(pins or excludes)
    if has_overrides and not declared:
        diagnostics.append("override_source_digest_missing:overrides_ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": "",
            "observed_source_digest": observed,
            "source_artifact_id": bundle_source_id,
        }
    if declared and declared.lower() != observed:
        diagnostics.append("stale_override_source:overrides_ignored")
        return {
            "pin": [],
            "exclude": [],
            "fresh": False,
            "declared_source_digest": declared,
            "observed_source_digest": observed,
            "source_artifact_id": bundle_source_id,
        }
    return {
        "pin": pins,
        "exclude": excludes,
        "fresh": has_overrides,
        "declared_source_digest": declared,
        "observed_source_digest": observed,
        "source_artifact_id": bundle_source_id,
    }


def _canonical_bundle_digest(bundle_doc: dict[str, Any]) -> str:
    """Return the stable logical digest of a bundle document."""
    return _canonical_digest(bundle_doc)


def _clip_intervals(
    intervals: list[Any], duration_s: float, diagnostics: list[str]
) -> list[dict[str, Any]]:
    """Clip event rows while preserving explicit interval identity.

    Returns:
        Deterministically clipped interval annotations with source identity.
    """
    clipped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(intervals):
        if not isinstance(item, dict):
            diagnostics.append(f"event_row_{position}_malformed")
            continue
        start = _finite_number(item.get("start_s"))
        end = _finite_number(item.get("end_s"))
        if start is None or end is None or not end > start:
            diagnostics.append(f"event_row_{position}_malformed")
            continue
        if start >= duration_s or end <= 0.0:
            diagnostics.append("interval_outside_duration")
            continue
        interval_id = item.get("interval_id")
        if interval_id is not None and not isinstance(interval_id, str):
            diagnostics.append(f"event_row_{position}_malformed")
            continue
        if isinstance(interval_id, str):
            if interval_id in seen_ids:
                diagnostics.append(f"duplicate_interval_id:{interval_id}")
                continue
            seen_ids.add(interval_id)
        entry: dict[str, Any] = {
            "start_s": max(start, 0.0),
            "end_s": min(end, duration_s),
            "source_row": position,
        }
        if isinstance(interval_id, str):
            entry["interval_id"] = interval_id
        event_id = item.get("event_id")
        if isinstance(event_id, str):
            entry["event_id"] = event_id
        clipped.append(entry)
    return clipped


def _spec_intervals(
    index_doc: dict[str, Any] | None, duration: float | None, diagnostics: list[str]
) -> list[dict[str, Any]] | None:
    """Derive clipped event rows, retaining identity for annotation consumers.

    Returns:
        Clipped interval annotations, or ``None`` when the optional index is unavailable.
    """
    if index_doc is None or duration is None:
        return None
    raw = index_doc.get("intervals")
    if not isinstance(raw, list):
        diagnostics.append("event_index_intervals_missing")
        return None
    return _clip_intervals(raw, duration, diagnostics)


def _read_family_docs(
    by_format: dict[str, list[Any]], root: Path, diagnostics: list[str]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Read each source family once and retain every source provenance record.

    Returns:
        Parsed family documents and all request-source provenance records.
    """
    docs: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for family in sorted(by_format):
        refs = by_format[family]
        if len(refs) != 1:
            diagnostics.append(f"{family}: source_family_collision")
            records.extend(_source_record(ref, integrity_status="family_collision") for ref in refs)
            continue
        payload, record, problem = _read_json(refs[0], root)
        records.append(record)
        if problem is not None:
            diagnostics.append(problem)
        if payload is not None:
            docs[family] = payload
    return docs, records


def _load_inputs(
    request: ComponentRequest, root: Path, diagnostics: list[str]
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
    dict[str, list[Any]],
]:
    """Load and validate bundle, scores, and event-index source families.

    Returns:
        Bundle, optional documents, source records, and grouped source references.
    """
    by_format: dict[str, list[Any]] = {}
    records: list[dict[str, Any]] = []
    known_formats = set(REQUIRED_CAPABILITIES) | set(OPTIONAL_CAPABILITIES)
    for ref in request.sources:
        if ref.format not in known_formats:
            records.append(_source_record(ref, integrity_status="unsupported"))
            diagnostics.append(f"{ref.artifact_id}: unknown_source_format:{ref.format}")
            continue
        by_format.setdefault(ref.format, []).append(ref)
    for family in REQUIRED_CAPABILITIES:
        if family not in by_format:
            diagnostics.append(f"required_source_family_missing:{family}")
    if any(code.startswith("required_source_family_missing") for code in diagnostics):
        return None, None, None, records, by_format
    docs, loaded_records = _read_family_docs(by_format, root, diagnostics)
    records.extend(loaded_records)
    bundle_doc = docs.get("review-bundle")
    if bundle_doc is not None:
        try:
            review_bundle_from_dict(bundle_doc)
        except ReviewContractsValidationError as error:
            diagnostics.append("bundle_invalid:" + error.errors[0][:120])
            bundle_doc = None
    return bundle_doc, docs.get("exemplar-scores"), docs.get("event-index"), records, by_format


def _bundle_episode_index(
    bundle_doc: dict[str, Any], root: Path, diagnostics: list[str]
) -> tuple[dict[str, tuple[dict[str, Any], ...]] | None, list[dict[str, Any]], bool]:
    """Validate episode identity and verify every referenced bundle artifact.

    Returns:
        Episode-to-reference mapping, bundle-reference records, and hard-failure state.
    """
    episode_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    records: list[dict[str, Any]] = []
    hard_failure = False
    for episode in bundle_doc["episodes"]:
        episode_id = str(episode["episode_id"])
        if episode_id in episode_by_id:
            diagnostics.append(f"duplicate_episode_id:{episode_id}")
            hard_failure = True
            continue
        refs = tuple(dict(ref) for ref in episode["references"])
        episode_by_id[episode_id] = refs
        verified_refs: list[dict[str, Any]] = []
        for ref in refs:
            raw, record, problem = _verify_reference(
                ref, root, episode_id=episode_id, scope="bundle-reference"
            )
            records.append(record)
            if problem is not None:
                diagnostics.append(
                    problem.replace("source_uri_unsafe", "source_reference_uri_unsafe")
                )
                if record["integrity_status"] != "verified":
                    hard_failure = True
            if raw is not None and record["integrity_status"] == "verified":
                verified_refs.append(ref)
        if len(verified_refs) != len(refs):
            hard_failure = True
    if hard_failure:
        return None, records, True
    return episode_by_id, records, False


def _family_records(
    records: list[dict[str, Any]], family: str, *, scope: str = "request"
) -> list[dict[str, Any]]:
    """Select records belonging to one source family and scope.

    Returns:
        Matching source provenance records.
    """
    return [
        record for record in records if record["format"] == family and record.get("scope") == scope
    ]


def _family_integrity_verified(records: list[dict[str, Any]], family: str) -> bool:
    """Return whether exactly one family source has verified bytes.

    Returns:
        Whether one and only one request source for the family is byte-verified.
    """
    family_records = _family_records(records, family)
    return len(family_records) == 1 and family_records[0]["integrity_status"] == "verified"


def _score_entries(entries: list[Any], known: set[str], diagnostics: list[str]) -> dict[str, float]:
    """Parse canonical trace-exemplar-interest episode entries.

    Returns:
        Finite scores keyed by known episode ID.
    """
    scores: dict[str, float] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            diagnostics.append(f"exemplar_scores_row_malformed:{position}")
            continue
        episode_id = entry.get("episode_id")
        number = _finite_number(entry.get("composite_score"))
        if not isinstance(episode_id, str) or number is None:
            diagnostics.append(f"exemplar_scores_row_malformed:{position}")
            continue
        if episode_id not in known:
            diagnostics.append(f"score_unknown_episode:{episode_id}")
            continue
        if episode_id in scores and scores[episode_id] != number:
            diagnostics.append(f"score_conflicting_duplicate:{episode_id}")
            continue
        scores[episode_id] = number
    return scores


def _score_map(
    raw_map: dict[str, Any], known: set[str], diagnostics: list[str]
) -> dict[str, float]:
    """Parse an explicitly versioned SREV-07 score map.

    Returns:
        Finite scores keyed by known episode ID.
    """
    scores: dict[str, float] = {}
    for episode_id, value in raw_map.items():
        number = _finite_number(value)
        if episode_id not in known:
            diagnostics.append(f"score_unknown_episode:{episode_id}")
            continue
        if number is None:
            diagnostics.append(f"score_malformed:{episode_id}")
            continue
        scores[episode_id] = number
    return scores


def _record_missing_scores(
    scores: dict[str, float], known: set[str], diagnostics: list[str]
) -> None:
    """Record known bundle episodes omitted by a score document."""
    for episode_id in sorted(known - set(scores)):
        diagnostics.append(f"score_missing:{episode_id}")


def _score_table(
    scores_doc: dict[str, Any] | None, episode_ids: list[str], diagnostics: list[str]
) -> tuple[dict[str, float], str]:
    """Extract only versioned or canonical-owner exemplar scores.

    Missing scores remain unavailable (``None`` in the output ranking); they
    never become an implicit numeric default.

    Returns:
        Parsed scores and the explicit input/adaptor format identifier.
    """
    if scores_doc is None:
        diagnostics.append("exemplar_scores_unavailable")
        return {}, "unavailable"
    known = set(episode_ids)
    raw_entries = scores_doc.get("episodes")
    if isinstance(raw_entries, list):
        scores = _score_entries(raw_entries, known, diagnostics)
        _record_missing_scores(scores, known, diagnostics)
        return scores, CANONICAL_SCORE_ADAPTER_VERSION
    raw_map = scores_doc.get("scores")
    if scores_doc.get("schema_version") != EXEMPLAR_SCORES_SCHEMA_VERSION or not isinstance(
        raw_map, dict
    ):
        diagnostics.append("exemplar_scores_unversioned_or_malformed")
        return {}, "unavailable"
    scores = _score_map(raw_map, known, diagnostics)
    _record_missing_scores(scores, known, diagnostics)
    return scores, EXEMPLAR_SCORES_SCHEMA_VERSION


def _blocking_diagnostics(
    diagnostics: list[str],
    *,
    requested_capabilities: tuple[str, ...],
    score_source_present: bool,
) -> list[str]:
    """Classify diagnostics without treating an absent optional stream as failure.

    Returns:
        Diagnostics that make the result partial rather than complete.
    """
    blocking: list[str] = []
    for item in diagnostics:
        if item.startswith("excluded_episodes:"):
            continue
        if item == "exemplar_scores_unavailable" and not score_source_present:
            if "exemplar-scores" not in requested_capabilities:
                continue
        blocking.append(item)
    return blocking


def run(  # noqa: C901, PLR0912, PLR0915
    request: ComponentRequest, *, base: Path | None = None
) -> ComponentResult:
    """Rank storyboard candidates and emit deterministic, provenance-bound sidecars.

    Returns:
        A versioned component result describing status, sidecars, and provenance.
    """
    root = (base if base is not None else Path.cwd()).resolve()
    rejected = _reject_not_applicable(request)
    if rejected is not None:
        return rejected
    output_dir, output_error = _resolve_output_directory(root, request.output_directory)
    if output_error is not None or output_dir is None:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=output_error or "unsafe_output_path",
        )
    if output_dir.exists() or output_dir.is_symlink():
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=f"output_collision: already exists: {request.output_directory}",
        )
    diagnostics: list[str] = []
    try:
        bundle_doc, scores_doc, index_doc, source_records, refs_by_format = _load_inputs(
            request, root, diagnostics
        )
        if bundle_doc is None:
            return _result(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                reason="required_source_family_unusable: "
                + "; ".join(sorted(set(diagnostics))[:5]),
            )
        bundle_records = _family_records(source_records, "review-bundle")
        if not _family_integrity_verified(source_records, "review-bundle"):
            diagnostics.append("required_source_integrity_unverified:review-bundle")
            return _result(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                reason="required_source_integrity_unverified: review-bundle; "
                + "; ".join(sorted(set(diagnostics))[:5]),
            )
        episode_by_id, bundle_reference_records, bundle_hard_failure = _bundle_episode_index(
            bundle_doc, root, diagnostics
        )
        source_records.extend(bundle_reference_records)
        if bundle_hard_failure or episode_by_id is None:
            return _result(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
                provenance={
                    "source_provenance": source_records,
                    "evidence_boundary": EVIDENCE_BOUNDARY,
                },
                reason="; ".join(sorted(set(diagnostics))[:8]),
            )

        episode_ids = list(episode_by_id)
        score_source_present = bool(refs_by_format.get("exemplar-scores"))
        event_source_present = bool(refs_by_format.get("event-index"))
        available_capabilities = ["review-bundle"]
        score_usable = scores_doc is not None and _family_integrity_verified(
            source_records, "exemplar-scores"
        )
        event_usable = index_doc is not None and _family_integrity_verified(
            source_records, "event-index"
        )
        if score_usable:
            available_capabilities.append("exemplar-scores")
        elif score_source_present:
            diagnostics.append("source_integrity_unverified:exemplar-scores")
        if event_usable:
            available_capabilities.append("event-index")
        elif event_source_present:
            diagnostics.append("source_integrity_unverified:event-index")
        missing_capabilities = [
            capability
            for capability in request.required_capabilities
            if capability not in available_capabilities
        ]
        for capability in missing_capabilities:
            if f"required_capability_missing:{capability}" not in diagnostics:
                diagnostics.append(f"required_capability_missing:{capability}")

        scores, score_format = _score_table(
            scores_doc if score_usable else None, episode_ids, diagnostics
        )
        bundle_source_id = str(bundle_records[0]["artifact_id"])
        overrides = _parse_overrides(request.config, bundle_doc, bundle_source_id, diagnostics)
        kept, rank_diagnostics = _rank_candidates(
            episode_ids,
            scores,
            overrides["pin"],
            overrides["exclude"],
            score_format,
        )
        diagnostics.extend(rank_diagnostics)
        duration = _finite_number(request.config.get("source_duration_s"))
        if duration is None or duration <= 0:
            diagnostics.append("source_duration_missing_or_invalid")
            duration = None
        event_intervals = _spec_intervals(
            index_doc if event_usable else None, duration, diagnostics
        )
        selected_episode_ids = [candidate.episode_id for candidate in kept]
        selected_source_artifact_ids = [
            str(ref["artifact_id"])
            for episode_id in selected_episode_ids
            for ref in episode_by_id[episode_id]
        ]
        spec_payload: dict[str, Any] = {
            "schema_version": "visualization-spec.v1",
            "spec_id": f"{request.request_id}-storyboard",
            "sources": [
                {"artifact_id": artifact_id} for artifact_id in selected_source_artifact_ids
            ],
            "annotations": [
                {
                    "ranking": [
                        {
                            "episode_id": candidate.episode_id,
                            "source_artifact_ids": [
                                str(ref["artifact_id"])
                                for ref in episode_by_id[candidate.episode_id]
                            ],
                            "score": candidate.score,
                            "score_source": candidate.score_source,
                            "pinned": candidate.pinned,
                        }
                        for candidate in kept
                    ],
                    "tie_break_method": TIE_BREAK_METHOD,
                    "source_binding": {
                        "selected_episode_ids": selected_episode_ids,
                        "selected_source_artifact_ids": selected_source_artifact_ids,
                    },
                }
            ],
        }
        if event_intervals is not None:
            spec_payload["source_intervals"] = [
                {
                    "start_s": interval["start_s"],
                    "end_s": interval["end_s"],
                    "include_terminal_frame": False,
                }
                for interval in event_intervals
            ]
            spec_payload["annotations"][0]["event_intervals"] = event_intervals
        visualization_spec_from_dict(spec_payload)

        blocking = _blocking_diagnostics(
            diagnostics,
            requested_capabilities=request.required_capabilities,
            score_source_present=score_source_present,
        )
        status = STATUS_PARTIAL if blocking else STATUS_COMPLETE
        diagnostics_payload = sorted(set(diagnostics))
        override_payload = {
            "schema_version": "override-provenance.v1",
            "result_status": status,
            "source_artifact_id": overrides["source_artifact_id"],
            "declared_source_digest": overrides["declared_source_digest"],
            "observed_source_digest": overrides["observed_source_digest"],
            "pins": sorted(set(overrides["pin"])),
            "excludes": sorted(set(overrides["exclude"])),
            "overrides_applied": overrides["fresh"],
            "diagnostics": diagnostics_payload,
        }
        capability_payload = {
            "schema_version": MISSING_CAPABILITY_SCHEMA_VERSION,
            "result_status": status,
            "requested_capabilities": list(request.required_capabilities),
            "available_capabilities": sorted(available_capabilities),
            "missing_capabilities": sorted(set(missing_capabilities)),
            "optional_capabilities_unavailable": sorted(
                set(OPTIONAL_CAPABILITIES) - set(available_capabilities)
            ),
            "diagnostics": diagnostics_payload,
        }
        payloads = {
            OUTPUT_CAPABILITY_FILENAME: capability_payload,
            OUTPUT_OVERRIDES_FILENAME: override_payload,
            OUTPUT_SPEC_FILENAME: spec_payload,
        }
        file_digests = _stage_outputs(output_dir, payloads)
        output_artifacts = tuple(
            {
                "artifact_id": filename,
                "uri": (Path(request.output_directory) / filename).as_posix(),
                "sha256": file_digests[filename],
            }
            for filename in sorted(payloads)
        )
        provenance = {
            "output_directory": request.output_directory,
            "component_version": COMPONENT_VERSION,
            "bundle_artifact_id": bundle_source_id,
            "bundle_logical_digest": _canonical_bundle_digest(bundle_doc),
            "selected_episode_ids": selected_episode_ids,
            "selected_source_artifact_ids": selected_source_artifact_ids,
            "source_provenance": source_records,
            "requested_capabilities": list(request.required_capabilities),
            "available_capabilities": sorted(available_capabilities),
            "missing_capabilities": sorted(set(missing_capabilities)),
            "score_input_format": score_format,
            "score_adapter_version": (
                CANONICAL_SCORE_ADAPTER_VERSION
                if score_format == CANONICAL_SCORE_ADAPTER_VERSION
                else ""
            ),
            "emitted_artifacts": [dict(item) for item in output_artifacts],
            "evidence_boundary": EVIDENCE_BOUNDARY,
        }
        result_artifacts = output_artifacts if status == STATUS_COMPLETE else ()
        reason = "" if status == STATUS_COMPLETE else "; ".join(blocking[:8])
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=status,
            artifacts=result_artifacts,
            diagnostics=tuple({"code": item} for item in diagnostics_payload),
            provenance=provenance,
            reason=reason,
        )
    except ReviewContractsValidationError as error:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            reason="; ".join(error.errors),
        )
    except (OSError, TypeError, ValueError) as error:
        return _result(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            reason=f"internal_failure: {error}",
        )


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the review-storyboard component."""
    parser = argparse.ArgumentParser(description="Rank storyboard candidates.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and print a schema-valid result envelope.

    Returns:
        Zero for a complete result, otherwise one.
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read request: {error}"]) from error
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
        if isinstance(config, dict):
            payload = {**payload, "config": {**payload.get("config", {}), **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(result_document(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
