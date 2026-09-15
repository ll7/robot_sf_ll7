"""Draft evidence-linked explanations over review contracts (SREV-26, issue #9297).

This module is the review-ai contract consumer: it turns explicitly selected
evidence artifacts (trace annotation sets, failure-diagnosis records) plus a
validated request config into a structured draft explanation with validated
row/number citations. It reuses the SREV-01 shared-contract surface
(:mod:`robot_sf.analysis_workbench.review_contracts`) for envelopes,
validation, and digests, and the canonical evidence owners
(:mod:`robot_sf.analysis_workbench.trace_annotation` and
:mod:`robot_sf.benchmark.failure_diagnosis`) to validate every cited record.
It never trains models, contacts providers, or executes experiments: the only
provider is a deterministic offline template renderer, packet text is quoted
as data (never instructions), and every interpretation is marked draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_request_from_dict,
)
from robot_sf.analysis_workbench.trace_annotation import (
    TraceAnnotationSetValidationError,
    trace_annotation_set_from_dict,
)
from robot_sf.benchmark.failure_diagnosis import (
    FailureDiagnosisError,
    validate_failure_diagnosis_record,
)

COMPONENT_ID = "srev26-review-ai"
COMPONENT_VERSION = "1.0.0"

SUPPORTED_PROVIDER = "fake-local"

FORMAT_TRACE_ANNOTATION_SET = "trace_annotation_set.v1"
FORMAT_FAILURE_DIAGNOSIS = "failure_diagnosis.v1"
CITABLE_FORMATS = (FORMAT_TRACE_ANNOTATION_SET, FORMAT_FAILURE_DIAGNOSIS)

EXPLANATION_SCHEMA_VERSION = "review-explanation.v1"
EXPLANATION_FILENAME = "explanation.json"
DESCRIPTOR_FILENAME = "component-descriptor.json"

_DESCRIBE_OUTPUT_TYPES = ("review-explanation.v1",)

_DESCRIPTOR_DOC: dict[str, Any] = {
    "schema_version": "component-descriptor.v1",
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": [],
    "optional_capabilities": [],
    "output_types": list(_DESCRIBE_OUTPUT_TYPES),
}

# Fail fast on descriptor drift: the shipped descriptor must stay schema-valid.
DESCRIPTOR = component_descriptor_from_dict(_DESCRIPTOR_DOC)

_SECRET_KEY_HINTS = ("token", "secret", "password", "api_key", "apikey", "credential")
REDACTED = "[redacted]"


def descriptor() -> dict[str, Any]:
    """Return the versioned capability descriptor for this component.

    Returns:
        Descriptor document declaring exact required/optional capabilities
        (none) and versioned result artifact types.
    """
    return json.loads(json.dumps(_DESCRIPTOR_DOC))


def _canonical_sha256(value: Any) -> str:
    """Hash logical content deterministically (location-independent).

    Returns:
        Hex SHA-256 digest of the canonical encoding.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Write JSON atomically and return the hex SHA-256 of the file bytes.

    Returns:
        Hex SHA-256 digest of the written file bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unsupported_capabilities(request: ComponentRequest) -> list[str]:
    """List requested capabilities this component does not provide.

    Returns:
        Requested capability names absent from the descriptor.
    """
    supported = set(DESCRIPTOR.required_capabilities) | set(DESCRIPTOR.optional_capabilities)
    return [name for name in request.required_capabilities if name not in supported]


def _major(version: str) -> str | None:
    """Return the ``v<major>`` marker of a dotted version string, if parseable."""
    head = version.strip().split(".", 1)[0].lstrip("vV")
    return f"v{head}" if head.isdigit() else None


def _redact_value(key: str, value: Any, redacted: list[str], trail: str) -> Any:
    """Recursively replace credential-looking config values with a marker.

    Returns:
        The value with secrets replaced by the redaction marker.
    """
    lowered = key.lower()
    if any(hint in lowered for hint in _SECRET_KEY_HINTS):
        redacted.append(trail or key)
        return REDACTED
    if isinstance(value, dict):
        return {
            item_key: _redact_value(str(item_key), item_value, redacted, f"{trail}.{item_key}")
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _redact_value(key, item, redacted, f"{trail}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _redact_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Redact credential-looking fields before config enters outputs.

    Returns:
        Tuple of (redacted config copy, sorted redacted field trails).
    """
    redacted: list[str] = []
    cleaned = _redact_value("config", config, redacted, "config")
    assert isinstance(cleaned, dict)
    return cleaned, sorted(redacted)


def _provider_name(config: dict[str, Any]) -> str:
    """Return the requested provider name, defaulting to the offline fake."""
    provider = config.get("provider", {"name": SUPPORTED_PROVIDER})
    if not isinstance(provider, dict):
        return ""
    name = provider.get("name", SUPPORTED_PROVIDER)
    return name if isinstance(name, str) else ""


def _provider_allows_remote(config: dict[str, Any]) -> bool:
    """Return whether the request explicitly enables a remote provider."""
    provider = config.get("provider")
    return isinstance(provider, dict) and provider.get("allow_remote") is True


def _resolve_source_file(source_base: Path, uri: str) -> Path | None:
    """Resolve a source URI under the base without allowing escapes.

    Returns:
        The resolved path, or ``None`` when it escapes the base.
    """
    try:
        resolved_base = source_base.resolve(strict=False)
        resolved = (resolved_base / uri).resolve(strict=False)
        resolved.relative_to(resolved_base)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _load_source_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read one source JSON document.

    Returns:
        Tuple of (payload, error); exactly one side is meaningful.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        return None, f"unreadable-source: {error}"
    except json.JSONDecodeError as error:
        return None, f"corrupt-source: invalid JSON: {error}"
    if not isinstance(payload, dict):
        return None, "corrupt-source: top-level JSON value must be an object"
    return payload, None


def _annotation_points(
    artifact_id: str, payload: dict[str, Any], path: Path
) -> tuple[
    list[dict[str, Any]] | None,
    str | None,
]:
    """Validate an annotation set through its canonical owner and cite rows.

    Returns:
        Tuple of (citation points, error); exactly one side is meaningful.
    """
    try:
        annotation_set = trace_annotation_set_from_dict(payload, source=path)
    except TraceAnnotationSetValidationError as error:
        return None, f"corrupt-source: {artifact_id}: " + "; ".join(error.errors)
    points = [
        {
            "kind": "annotation",
            "artifact_id": artifact_id,
            "annotation_id": annotation.annotation_id,
            "category": annotation.category,
            "evidence_type": annotation.evidence_type,
            "frame_start": annotation.anchor.frame_start,
            "frame_end": annotation.anchor.frame_end,
            "event_ids": list(annotation.anchor.event_ids),
            "entity_ids": [f"{entity.type}:{entity.id}" for entity in annotation.anchor.entities],
            "summary": annotation.summary,
            "timeline_trace_id": annotation_set.timeline.trace_id,
        }
        for annotation in annotation_set.annotations
    ]
    return points, None


def _diagnosis_points(
    artifact_id: str, payload: dict[str, Any]
) -> tuple[
    list[dict[str, Any]] | None,
    str | None,
]:
    """Validate a diagnosis record through its canonical owner and cite rows.

    Returns:
        Tuple of (citation points, error); exactly one side is meaningful.
    """
    try:
        record = validate_failure_diagnosis_record(payload)
    except FailureDiagnosisError as error:
        return None, f"corrupt-source: {artifact_id}: {error}"
    onset = record["onset_time_s"]
    points = [
        {
            "kind": "diagnosis",
            "artifact_id": artifact_id,
            "failure_level": record["failure_level"],
            "failure_type": record["failure_type"],
            "severity": record["severity"],
            "onset_time_s": onset,
            "onset_time_units": "seconds" if onset is not None else None,
            "onset_interval_s": list(record["onset_interval"]),
            "confidence": record["confidence"],
            "validity_status": record["validity_status"],
        }
    ]
    return points, None


def _validate_highlight(
    highlight: Any, index: int, usable_ids: set[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one config number citation against usable source identities.

    Returns:
        Tuple of (citation item, errors); exactly one side is meaningful.
    """
    where = f"corrupt-highlights[{index}]"
    if not isinstance(highlight, dict):
        return None, [f"{where}: highlight must be a mapping"]
    errors: list[str] = []
    metric = highlight.get("metric")
    if not isinstance(metric, str) or not metric:
        errors.append(f"{where}: 'metric' must be a non-empty string")
    value = highlight.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{where}: 'value' must be a finite number")
    elif not math.isfinite(float(value)):
        errors.append(f"{where}: 'value' must be a finite number")
    units = highlight.get("units")
    if not isinstance(units, str) or not units:
        errors.append(f"{where}: 'units' must be a non-empty string")
    source_id = highlight.get("source_artifact_id")
    if not isinstance(source_id, str) or not source_id:
        errors.append(f"{where}: 'source_artifact_id' must be a non-empty string")
    elif source_id not in usable_ids:
        errors.append(
            f"{where}: 'source_artifact_id' {source_id!r} does not name a usable source; "
            "fabricated citations are rejected"
        )
    if errors:
        return None, errors
    assert isinstance(metric, str) and isinstance(units, str) and isinstance(source_id, str)
    assert isinstance(value, (int, float))
    return {
        "kind": "highlight",
        "artifact_id": source_id,
        "metric": metric,
        "value": float(value),
        "units": units,
        "source_artifact_id": source_id,
    }, []


def _validate_highlights(
    highlights: Any, usable_ids: set[str]
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Validate config number citations against usable source identities.

    Returns:
        Tuple of (citation items, errors); exactly one side is meaningful.
    """
    if highlights is None:
        return [], []
    if not isinstance(highlights, list):
        return None, ["corrupt-highlights: 'highlights' must be a list"]
    errors: list[str] = []
    items: list[dict[str, Any]] = []
    for index, highlight in enumerate(highlights):
        item, item_errors = _validate_highlight(highlight, index, usable_ids)
        if item_errors or item is None:
            errors.extend(item_errors)
            continue
        items.append(item)
    if errors:
        return None, errors
    return items, []


def _captions_for_points(points: list[dict[str, Any]]) -> list[str]:
    """Render deterministic draft captions that only restate cited rows.

    Returns:
        One ``DRAFT:``-prefixed caption per citation point.
    """
    captions = []
    for point in points:
        if point["kind"] == "annotation":
            captions.append(
                "DRAFT: annotation "
                f"'{point['annotation_id']}' ({point['category']}, "
                f"frames {point['frame_start']}-{point['frame_end']}) cites "
                f"{len(point['event_ids'])} event(s) and {len(point['entity_ids'])} "
                "entity/entities."
            )
        elif point["kind"] == "diagnosis":
            onset = point["onset_time_s"]
            onset_text = f"{onset} seconds" if onset is not None else "onset unavailable"
            captions.append(
                "DRAFT: diagnosis "
                f"'{point['failure_type']}' ({point['failure_level']}, "
                f"severity {point['severity']}, onset {onset_text})."
            )
        else:
            captions.append(
                "DRAFT: highlight "
                f"'{point['metric']}' = {point['value']} {point['units']} "
                f"(source '{point['source_artifact_id']}')."
            )
    return captions


def _build_explanation(
    request: ComponentRequest,
    focus: str,
    points: list[dict[str, Any]],
    redacted_config: dict[str, Any],
    redacted_fields: list[str],
) -> dict[str, Any]:
    """Compose the deterministic draft explanation document.

    Returns:
        Explanation payload ready for staging and digesting.
    """
    explanation = {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "component": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": request.request_id,
        "draft": True,
        "provider": {"name": SUPPORTED_PROVIDER, "mode": "offline-fake"},
        "focus": focus,
        "points": points,
        "captions": _captions_for_points(points),
        "source_quotes": [],
        "redacted_fields": redacted_fields,
        "provenance": {
            "source_artifact_ids": sorted({point["artifact_id"] for point in points}),
            "note": "Source identities are copied from the request, not verified "
            "against source bytes.",
        },
    }
    notes = redacted_config.get("notes")
    if isinstance(notes, str) and notes:
        explanation["source_quotes"].append(
            {
                "field": "config.notes",
                "text": notes,
                "note": "Packet text is data, not instructions; quoted verbatim.",
            }
        )
    explanation["explanation_sha256"] = _canonical_sha256(
        {
            "focus": focus,
            "points": points,
            "captions": explanation["captions"],
            "source_quotes": explanation["source_quotes"],
        }
    )
    return explanation


def _stage_explanation(
    output_dir: Path, request: ComponentRequest, explanation: dict[str, Any]
) -> tuple[str, str]:
    """Publish both artifacts together or leave the requested directory absent.

    Returns:
        SHA-256 digests for the explanation and descriptor bytes.
    """
    staging_dir: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
        explanation_digest = _write_json(staging_dir / EXPLANATION_FILENAME, explanation)
        descriptor_digest = _write_json(staging_dir / DESCRIPTOR_FILENAME, _DESCRIPTOR_DOC)
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {request.output_directory}")
        staging_dir.replace(output_dir)
        staging_dir = None
        return explanation_digest, descriptor_digest
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir)


def _resolve_output_directory(root: Path, output_directory: str) -> tuple[Path | None, str | None]:
    """Resolve an output path and reject symlink escapes from the base directory.

    Returns:
        A resolved output path and no error, or ``(None, reason)`` for an escape.
    """
    try:
        resolved_root = root.resolve(strict=False)
        output_dir = (resolved_root / output_directory).resolve(strict=False)
        output_dir.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None, "unsafe-output-path: output directory must resolve within base"
    return output_dir, None


def _unavailable(request: ComponentRequest, reason: str) -> ComponentResult:
    """Build an unavailable result without artifacts.

    Returns:
        Unavailable component result carrying only the reason.
    """
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="unavailable",
        reason=reason,
    )


def _failed(
    request: ComponentRequest, reason: str, diagnostics: tuple[dict[str, Any], ...] = ()
) -> ComponentResult:
    """Build a failed result without artifacts.

    Returns:
        Failed component result carrying the reason and diagnostics.
    """
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="failed",
        diagnostics=diagnostics,
        reason=reason,
    )


def _availability_gate(request: ComponentRequest) -> ComponentResult | None:
    """Reject unsupported components, capabilities, versions, and providers.

    Returns:
        An unavailable result, or ``None`` when invocation may proceed.
    """
    if request.component_id != COMPONENT_ID:
        return _unavailable(request, f"unsupported component: {request.component_id}")
    missing = _unsupported_capabilities(request)
    if missing:
        return _unavailable(request, f"missing capabilities: {', '.join(sorted(missing))}")
    required_version = request.config.get("required_component_version")
    if required_version is not None and _major(str(required_version)) != _major(COMPONENT_VERSION):
        return _unavailable(
            request,
            f"incompatible-required-version: {required_version!r}; "
            f"this component implements {COMPONENT_VERSION}",
        )
    provider = _provider_name(request.config)
    if provider != SUPPORTED_PROVIDER:
        if _provider_allows_remote(request.config):
            return _unavailable(
                request,
                f"remote-provider-not-implemented: {provider or 'missing-name'}; "
                "only the offline fake provider is shipped",
            )
        return _unavailable(
            request,
            f"remote-provider-disabled: {provider or 'missing-name'}; "
            "remote providers require explicit enablement",
        )
    return None


def _cite_source(
    ref: Any, payload: dict[str, Any], source_file: Path
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Cite one source through its canonical evidence owner.

    Returns:
        Tuple of (citation points, error); exactly one side is meaningful.
    """
    if ref.format == FORMAT_TRACE_ANNOTATION_SET:
        return _annotation_points(ref.artifact_id, payload, source_file)
    if ref.format == FORMAT_FAILURE_DIAGNOSIS:
        return _diagnosis_points(ref.artifact_id, payload)
    return None, f"unsupported-evidence-format: {ref.format}"


def _collect_evidence(
    request: ComponentRequest, sources_root: Path
) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]], ComponentResult | None]:
    """Load and cite every source, diagnosing skipped ones without blocking.

    Returns:
        Tuple of (points, usable ids, diagnostics, terminal result). The
        terminal result is set for missing or wholly unusable evidence.
    """
    points: list[dict[str, Any]] = []
    usable_ids: set[str] = set()
    diagnostics: list[dict[str, Any]] = []
    for ref in request.sources:
        source_file = _resolve_source_file(sources_root, ref.uri)
        if source_file is None:
            diagnostics.append({"artifact_id": ref.artifact_id, "reason": "unsafe-source-uri"})
            continue
        payload, load_error = _load_source_payload(source_file)
        if load_error is not None or payload is None:
            diagnostics.append({"artifact_id": ref.artifact_id, "reason": load_error})
            continue
        cited, error = _cite_source(ref, payload, source_file)
        if error is not None or cited is None:
            diagnostics.append({"artifact_id": ref.artifact_id, "reason": error})
            continue
        points.extend(cited)
        usable_ids.add(ref.artifact_id)
    if not request.sources:
        return (
            points,
            usable_ids,
            diagnostics,
            _failed(request, "missing-evidence: request carries no sources"),
        )
    if not usable_ids:
        return (
            points,
            usable_ids,
            diagnostics,
            ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status="unavailable",
                diagnostics=tuple(diagnostics),
                reason="unsupported-evidence: no source could be cited",
            ),
        )
    return points, usable_ids, diagnostics, None


def run(
    request: ComponentRequest, *, base: Path | None = None, source_base: Path | None = None
) -> ComponentResult:
    """Draft one evidence-linked explanation from a validated component request.

    Args:
        request: Validated component request.
        base: Base directory the request output directory resolves under.
        source_base: Base directory source URIs resolve under (CLI: the request
            file's directory). Defaults to the current working directory.

    Returns:
        Component result: ``complete`` with explanation/descriptor artifacts,
        ``partial`` when some sources were skipped, ``unavailable`` for
        unsupported components, capabilities, versions, evidence, or providers,
        or ``failed`` for corrupt inputs and output collisions.
    """
    root = base if base is not None else Path.cwd()
    sources_root = source_base if source_base is not None else Path.cwd()
    gate = _availability_gate(request)
    if gate is not None:
        return gate
    if not isinstance(request.config, dict):
        return _failed(request, "corrupt-config: request config must be a mapping")
    raw_explanation = request.config.get("explanation")
    if not isinstance(raw_explanation, dict):
        return _failed(request, "corrupt-explanation: config must carry an 'explanation' mapping")
    focus = raw_explanation.get("focus")
    if not isinstance(focus, str) or not focus:
        return _failed(request, "corrupt-explanation: 'focus' must be a non-empty string")
    redacted_config, redacted_fields = _redact_config(request.config)

    points, usable_ids, diagnostics, terminal = _collect_evidence(request, sources_root)
    if terminal is not None:
        return terminal
    highlights, highlight_errors = _validate_highlights(
        raw_explanation.get("highlights"), usable_ids
    )
    if highlight_errors or highlights is None:
        return _failed(request, "; ".join(highlight_errors), tuple(diagnostics))
    points.extend(highlights)

    output_dir, path_error = _resolve_output_directory(root, request.output_directory)
    if path_error is not None or output_dir is None:
        return _failed(
            request, path_error or "unsafe-output-path: output directory must resolve within base"
        )
    if output_dir.exists():
        return _failed(
            request,
            f"output-collision: output directory already exists: {request.output_directory}",
        )
    try:
        explanation = _build_explanation(request, focus, points, redacted_config, redacted_fields)
        explanation_digest, descriptor_digest = _stage_explanation(output_dir, request, explanation)
    except (OSError, ValueError) as error:
        return _failed(request, f"output-write-failed: {error}")
    status = "complete" if not diagnostics else "partial"
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status=status,
        artifacts=(
            {
                "artifact_id": EXPLANATION_FILENAME,
                "uri": str(Path(request.output_directory) / EXPLANATION_FILENAME),
                "sha256": explanation_digest,
            },
            {
                "artifact_id": DESCRIPTOR_FILENAME,
                "uri": str(Path(request.output_directory) / DESCRIPTOR_FILENAME),
                "sha256": descriptor_digest,
            },
        ),
        diagnostics=tuple(diagnostics),
        provenance={
            "output_directory": request.output_directory,
            "explanation_sha256": explanation["explanation_sha256"],
            "component_version": COMPONENT_VERSION,
            "provider": SUPPORTED_PROVIDER,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the review-ai component.

    Returns:
        Argument parser with input/config/output/base options.
    """
    parser = argparse.ArgumentParser(description="Draft SREV-26 evidence-linked explanations.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-ai component.

    Returns:
        Process exit code (0 when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
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
    result = run(
        request,
        base=Path(args.base) if args.base is not None else None,
        source_base=input_path.parent,
    )
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
