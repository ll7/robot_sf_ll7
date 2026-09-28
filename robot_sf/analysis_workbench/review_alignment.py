"""Scenario-comparison alignment over review contracts (SREV-08, issue #9277).

This module is the review-alignment contract consumer: it turns two
explicitly selected simulation trace exports plus a declared anchor policy
into an alignment artifact with a standalone comparison report. It reuses the
SREV-01 shared-contract surface
(:mod:`robot_sf.analysis_workbench.review_contracts`) for envelopes,
validation, and digests, and the canonical compatibility owner
(:mod:`robot_sf.analysis_workbench.event_alignment`) for every equivalence
computation. It never normalizes durations, never invents causal pivots, and
never fills missing measurements: incompatible initial states, absent shared
prefixes, missing anchors, and unequal durations stay visible, and per-side
unavailable tails are reported, not interpolated.
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

from robot_sf.analysis_workbench.event_alignment import (
    PAIR_COMPARISON_GRAINS,
    build_pair_compatibility_record,
)
from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_request_from_dict,
)
from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    SimulationTraceExportValidationError,
    load_simulation_trace_export,
)

COMPONENT_ID = "srev08-review-alignment"
COMPONENT_VERSION = "1.0.0"

FORMAT_TRACE_EXPORT = "simulation_trace_export.v1"

ALIGNMENT_SCHEMA_VERSION = "review-alignment.v1"
ALIGNMENT_FILENAME = "alignment.json"
DESCRIPTOR_FILENAME = "component-descriptor.json"

_DESCRIBE_OUTPUT_TYPES = ("review-alignment.v1",)

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

ANCHOR_TYPES = ("event", "absolute-time")
DEFAULT_POSITION_TOLERANCE_M = 1e-6
DEFAULT_HEADING_TOLERANCE_RAD = 1e-6
DEFAULT_SHARED_PREFIX_STEPS = 3


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


def _positive_number(value: Any) -> float | None:
    """Return a positive finite float for tolerance-like inputs."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    probe = float(value)
    if not (probe > 0.0) or not math.isfinite(probe):
        return None
    return probe


def _validate_alignment(
    alignment: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the alignment mapping inside a request config.

    Returns:
        Tuple of (normalized alignment, errors); exactly one side is meaningful.
    """
    if not isinstance(alignment, dict):
        return None, ["corrupt-alignment: config must carry an 'alignment' mapping"]
    errors: list[str] = []
    left_id = alignment.get("left_artifact_id")
    if not isinstance(left_id, str) or not left_id:
        errors.append("corrupt-alignment: 'left_artifact_id' must be a non-empty string")
    right_id = alignment.get("right_artifact_id")
    if not isinstance(right_id, str) or not right_id:
        errors.append("corrupt-alignment: 'right_artifact_id' must be a non-empty string")
    if isinstance(left_id, str) and isinstance(right_id, str) and left_id and left_id == right_id:
        errors.append("corrupt-alignment: left and right artifacts must differ")
    grain = alignment.get("comparison_grain")
    if grain not in PAIR_COMPARISON_GRAINS:
        errors.append(
            "unsupported-comparison-grain: "
            f"{grain!r}; supported grains are {', '.join(sorted(PAIR_COMPARISON_GRAINS))}"
        )
    anchor, anchor_errors = _validate_anchor(alignment.get("anchor"))
    errors.extend(anchor_errors)
    tolerances, tolerance_errors = _validate_tolerances(alignment)
    errors.extend(tolerance_errors)
    if errors:
        return None, errors
    assert anchor is not None and tolerances is not None
    return {
        "left_artifact_id": left_id,
        "right_artifact_id": right_id,
        "comparison_grain": grain,
        "anchor": anchor,
        "tolerances": tolerances,
    }, []


def _validate_anchor(anchor: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the declared anchor policy without resolving it against traces.

    Returns:
        Tuple of (normalized anchor, errors); exactly one side is meaningful.
    """
    if not isinstance(anchor, dict):
        return None, ["corrupt-anchor: alignment must carry an 'anchor' mapping"]
    anchor_type = anchor.get("type")
    if anchor_type not in ANCHOR_TYPES:
        return None, [
            f"corrupt-anchor: 'type' must be one of {ANCHOR_TYPES}",
        ]
    if anchor_type == "event":
        event_id = anchor.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            return None, ["corrupt-anchor: event anchors need a non-empty 'event_id'"]
        return {"type": "event", "event_id": event_id}, []
    time_s = anchor.get("time_s")
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
        return None, ["corrupt-anchor: absolute-time anchors need a numeric 'time_s'"]
    if not (float(time_s) >= 0.0) or float(time_s) != float(time_s):
        return None, ["corrupt-anchor: absolute-time anchors need a finite 'time_s' >= 0"]
    return {"type": "absolute-time", "time_s": float(time_s)}, []


def _validate_tolerances(alignment: dict[str, Any]) -> tuple[dict[str, float] | None, list[str]]:
    """Validate optional tolerance overrides, falling back to owner defaults.

    Returns:
        Tuple of (tolerances, errors); exactly one side is meaningful.
    """
    tolerances = {
        "position_tolerance_m": DEFAULT_POSITION_TOLERANCE_M,
        "heading_tolerance_rad": DEFAULT_HEADING_TOLERANCE_RAD,
        "shared_prefix_steps": float(DEFAULT_SHARED_PREFIX_STEPS),
    }
    errors: list[str] = []
    for key in tolerances:
        if key not in alignment:
            continue
        value = _positive_number(alignment[key])
        if value is None:
            errors.append(f"corrupt-alignment: '{key}' must be a positive finite number")
            continue
        tolerances[key] = value
    if errors:
        return None, errors
    return tolerances, []


def _trace_event_ids(trace: SimulationTraceExport) -> list[str]:
    """Collect planner-reported event IDs in frame order, deduplicated.

    Returns:
        Event IDs in first-seen frame order.
    """
    seen: list[str] = []
    for frame in trace.frames:
        planner = frame.planner if isinstance(frame.planner, dict) else {}
        event_id = planner.get("event_id")
        if isinstance(event_id, str) and event_id and event_id not in seen:
            seen.append(event_id)
    return seen


def _trace_duration_s(trace: SimulationTraceExport) -> float | None:
    """Return the trace duration without normalizing anything."""
    if not trace.frames:
        return None
    return float(trace.frames[-1].time_s) - float(trace.frames[0].time_s)


def _event_records(trace: SimulationTraceExport) -> list[dict[str, Any]]:
    """Derive planner-reported event records for the compatibility owner.

    The records are explicitly marked as planner-reported (not independently
    detected): shared anchors only match identical planner reports on both
    sides, and anything else stays visible as unmatched.

    Returns:
        Event records ready for the compatibility owner.
    """
    records = []
    for frame in trace.frames:
        planner = frame.planner if isinstance(frame.planner, dict) else {}
        event_id = planner.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            continue
        records.append(
            {
                "event_id": event_id,
                "event_type": "planner_reported_event",
                "detector_profile_version": "trace-planner-reported.v1",
                "time_s": float(frame.time_s),
                "step": int(frame.step),
                "confidence": "reported",
                "actor_id": "robot",
                "zone_id": None,
                "source_fields": ["planner.event_id"],
                "event_relative_time": {},
                "visual_anchor_eligibility": {},
                "status": "available",
            }
        )
    return records


def _assess_anchor(
    anchor: dict[str, Any],
    left: SimulationTraceExport,
    right: SimulationTraceExport,
) -> dict[str, Any]:
    """Resolve the declared anchor against both traces without shifting them.

    Returns:
        Anchor assessment with an explicit availability status and reason.
    """
    if anchor["type"] == "event":
        event_id = anchor["event_id"]
        left_has = event_id in _trace_event_ids(left)
        right_has = event_id in _trace_event_ids(right)
        if left_has and right_has:
            return {"type": "event", "event_id": event_id, "status": "available"}
        missing = sorted(
            side for side, present in (("left", left_has), ("right", right_has)) if not present
        )
        return {
            "type": "event",
            "event_id": event_id,
            "status": "unavailable",
            "reason": f"anchor event missing on {', '.join(missing)} side(s)",
        }
    time_s = anchor["time_s"]
    spans = []
    for side, trace in (("left", left), ("right", right)):
        duration = _trace_duration_s(trace)
        start = float(trace.frames[0].time_s)
        end = float(trace.frames[-1].time_s)
        spans.append((side, start, end, duration))
    outside = sorted(side for side, start, end, _ in spans if not start <= time_s <= end)
    if not outside:
        return {"type": "absolute-time", "time_s": time_s, "status": "available"}
    return {
        "type": "absolute-time",
        "time_s": time_s,
        "status": "unavailable",
        "reason": f"anchor time outside {', '.join(outside)} trace span(s)",
        "spans_s": {side: [start, end] for side, start, end, _duration in spans},
    }


def _build_alignment(
    request: ComponentRequest,
    alignment: dict[str, Any],
    left: SimulationTraceExport,
    right: SimulationTraceExport,
    compatibility: dict[str, Any],
    anchor: dict[str, Any],
) -> dict[str, Any]:
    """Compose the alignment artifact with an admissible interpretation.

    Returns:
        Alignment payload ready for staging and digesting.
    """
    left_duration = _trace_duration_s(left)
    right_duration = _trace_duration_s(right)
    reasons: list[str] = []
    if compatibility.get("status") != "available":
        reasons.append(
            "comparison unavailable: "
            f"{compatibility.get('status')}: {compatibility.get('reason', 'see record')}"
        )
    if anchor.get("status") != "available":
        reasons.append(f"anchor unavailable: {anchor.get('reason', 'see assessment')}")
    if left_duration is not None and right_duration is not None and left_duration != right_duration:
        reasons.append(
            "unequal durations reported without normalization: "
            f"left {left_duration}s vs right {right_duration}s"
        )
    admissible = not reasons
    document = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "component": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": request.request_id,
        "comparison_grain": alignment["comparison_grain"],
        "tolerances": alignment["tolerances"],
        "compatibility": compatibility,
        "anchor": anchor,
        "durations_s": {"left": left_duration, "right": right_duration},
        "interpretation": {
            "admissible": admissible,
            "reasons": reasons
            or ["matched comparison with a resolved anchor; divergence output allowed"],
        },
        "provenance": {
            "left_artifact_id": alignment["left_artifact_id"],
            "right_artifact_id": alignment["right_artifact_id"],
            "left_trace_id": left.trace_id,
            "right_trace_id": right.trace_id,
            "note": "Source identities are copied from the request, not verified "
            "against source bytes.",
        },
    }
    document["alignment_sha256"] = _canonical_sha256(
        {
            "comparison_grain": document["comparison_grain"],
            "tolerances": document["tolerances"],
            "compatibility": compatibility,
            "anchor": anchor,
            "durations_s": document["durations_s"],
            "interpretation": document["interpretation"],
        }
    )
    return document


def _stage_alignment(
    output_dir: Path, request: ComponentRequest, alignment_doc: dict[str, Any]
) -> tuple[str, str]:
    """Publish both artifacts together or leave the requested directory absent.

    Returns:
        SHA-256 digests for the alignment and descriptor bytes.
    """
    staging_dir: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
        alignment_digest = _write_json(staging_dir / ALIGNMENT_FILENAME, alignment_doc)
        descriptor_digest = _write_json(staging_dir / DESCRIPTOR_FILENAME, _DESCRIPTOR_DOC)
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {request.output_directory}")
        staging_dir.replace(output_dir)
        staging_dir = None
        return alignment_digest, descriptor_digest
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
    """Reject unsupported components, capabilities, and versions.

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
    return None


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


def _load_trace_source(
    artifact_id: str, path: Path
) -> tuple[SimulationTraceExport | None, str | None]:
    """Validate one trace export through its canonical owner.

    Returns:
        Tuple of (trace, error); exactly one side is meaningful.
    """
    try:
        return load_simulation_trace_export(path), None
    except (SimulationTraceExportValidationError, OSError, ValueError) as error:
        return None, f"corrupt-source: {artifact_id}: {error}"


def _read_source_document(
    ref: Any, sources_root: Path
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read and parse one source document without validating its semantics.

    Returns:
        Tuple of (payload, diagnostic); exactly one side is meaningful.
    """
    source_file = _resolve_source_file(sources_root, ref.uri)
    if source_file is None:
        return None, {"artifact_id": ref.artifact_id, "reason": "unsafe-source-uri"}
    if ref.format != FORMAT_TRACE_EXPORT:
        return None, {
            "artifact_id": ref.artifact_id,
            "reason": f"unsupported-evidence-format: {ref.format}",
        }
    try:
        payload = json.loads(source_file.read_text(encoding="utf-8"))
    except OSError as error:
        return None, {"artifact_id": ref.artifact_id, "reason": f"unreadable: {error}"}
    except json.JSONDecodeError as error:
        return None, {"artifact_id": ref.artifact_id, "reason": f"corrupt JSON: {error}"}
    if not isinstance(payload, dict):
        return None, {
            "artifact_id": ref.artifact_id,
            "reason": "top-level JSON must be an object",
        }
    return {"payload": payload, "file": source_file}, None


def _collect_traces(
    request: ComponentRequest,
    alignment: dict[str, Any],
    sources_root: Path,
) -> tuple[
    SimulationTraceExport | None,
    SimulationTraceExport | None,
    list[dict[str, Any]],
    ComponentResult | None,
]:
    """Load and validate the two named trace sources, diagnosing the rest.

    Returns:
        Tuple of (left, right, diagnostics, terminal result). The terminal
        result is set when a named side is missing or unusable.
    """
    by_id: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for ref in request.sources:
        document, diagnostic = _read_source_document(ref, sources_root)
        if diagnostic is not None or document is None:
            diagnostics.append(diagnostic or {"artifact_id": ref.artifact_id, "reason": "unknown"})
            continue
        by_id[ref.artifact_id] = document
    left = right = None
    for side, key in (("left", "left_artifact_id"), ("right", "right_artifact_id")):
        wanted = alignment[key]
        entry = by_id.get(wanted)
        if entry is None:
            return (
                None,
                None,
                diagnostics,
                _failed(
                    request,
                    f"missing-evidence: {side} artifact {wanted!r} is not a usable trace source",
                    tuple(diagnostics),
                ),
            )
        trace, error = _load_trace_source(wanted, entry["file"])
        if error is not None or trace is None:
            return (
                None,
                None,
                diagnostics,
                _failed(request, error or "corrupt-source", tuple(diagnostics)),
            )
        if side == "left":
            left = trace
        else:
            right = trace
    assert left is not None and right is not None
    return left, right, diagnostics, None


def run(
    request: ComponentRequest, *, base: Path | None = None, source_base: Path | None = None
) -> ComponentResult:
    """Align one scenario comparison over two validated trace exports.

    Args:
        request: Validated component request.
        base: Base directory the request output directory resolves under.
        source_base: Base directory source URIs resolve under (CLI: the request
            file's directory). Defaults to the current working directory.

    Returns:
        Component result: ``complete`` with alignment/descriptor artifacts
        (inadmissible interpretations stay visible, never silent),
        ``unavailable`` for unsupported components, capabilities, versions,
        grains, or evidence, or ``failed`` for corrupt inputs and output
        collisions.
    """
    root = base if base is not None else Path.cwd()
    sources_root = source_base if source_base is not None else Path.cwd()
    gate = _availability_gate(request)
    if gate is not None:
        return gate
    if not isinstance(request.config, dict):
        return _failed(request, "corrupt-config: request config must be a mapping")
    alignment, alignment_errors = _validate_alignment(request.config.get("alignment"))
    if alignment_errors or alignment is None:
        raw_grain = (
            request.config.get("alignment", {}).get("comparison_grain")
            if isinstance(request.config.get("alignment"), dict)
            else None
        )
        if raw_grain is not None and raw_grain not in PAIR_COMPARISON_GRAINS:
            return _unavailable(
                request,
                f"unsupported-comparison-grain: {raw_grain!r}; "
                f"supported grains are {', '.join(sorted(PAIR_COMPARISON_GRAINS))}",
            )
        return _failed(request, "; ".join(alignment_errors))
    left, right, diagnostics, terminal = _collect_traces(request, alignment, sources_root)
    if terminal is not None or left is None or right is None:
        if terminal is not None:
            return terminal
        return _failed(request, "missing-evidence: comparison sides unresolved")
    tolerances = alignment["tolerances"]
    compatibility = build_pair_compatibility_record(
        left,
        right,
        left_events=_event_records(left),
        right_events=_event_records(right),
        comparison_grain=str(alignment["comparison_grain"]),
        position_tolerance_m=tolerances["position_tolerance_m"],
        heading_tolerance_rad=tolerances["heading_tolerance_rad"],
        shared_prefix_steps=int(tolerances["shared_prefix_steps"]),
    )
    anchor = _assess_anchor(alignment["anchor"], left, right)
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
        alignment_doc = _build_alignment(request, alignment, left, right, compatibility, anchor)
        alignment_digest, descriptor_digest = _stage_alignment(output_dir, request, alignment_doc)
    except (OSError, ValueError) as error:
        return _failed(request, f"output-write-failed: {error}")
    status = "complete" if not diagnostics else "partial"
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status=status,
        artifacts=(
            {
                "artifact_id": ALIGNMENT_FILENAME,
                "uri": str(Path(request.output_directory) / ALIGNMENT_FILENAME),
                "sha256": alignment_digest,
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
            "alignment_sha256": alignment_doc["alignment_sha256"],
            "component_version": COMPONENT_VERSION,
            "comparison_grain": alignment["comparison_grain"],
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the review-alignment component.

    Returns:
        Argument parser with input/config/output/base options.
    """
    parser = argparse.ArgumentParser(description="Align SREV-08 scenario comparisons.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-alignment component.

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
