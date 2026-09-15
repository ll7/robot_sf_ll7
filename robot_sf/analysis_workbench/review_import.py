"""SREV-02 review-import component: index benchmark artifacts into ``review-bundle.v1``.

This module owns the SREV-02 leaf surface only: a ``run(request)`` adapter plus a
standalone CLI that consumes the SREV-01 shared contracts, indexes existing
benchmark artifacts by reference (never rewriting source payloads), and reports
missing capabilities. Computation stays with the canonical owners
(``analysis_trace``, ``simulation_trace_export``, ``trace_dossier_package``,
``prepare_presentation_video_pack``); this module only reads, hashes, and indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
    review_bundle_canonical_digest,
    review_bundle_from_dict,
)

COMPONENT_ID = "srev02-review-import"
COMPONENT_VERSION = "1.0.0"

SOURCE_FAMILIES = (
    "episode-jsonl",
    "analysis-trace",
    "simulation-trace",
    "trace-dossier",
    "video-metadata",
)
OPTIONAL_FAMILIES = ("presentation-manifest",)

OUTPUT_BUNDLE_FILENAME = "review-bundle.json"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

_ZERO_COMMIT = "0" * 40

_DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("review-bundle.v1", "missing-capability-report.v1"),
    required_capabilities=SOURCE_FAMILIES,
    optional_capabilities=OPTIONAL_FAMILIES,
)


@dataclass
class _ImportOutcome:
    """Per-source import result before bundle assembly."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    file_sha256: str = ""
    diagnostics: list[str] = field(default_factory=list)
    skipped: bool = False


def descriptor() -> dict[str, Any]:
    """Return this component's self-contained capability descriptor."""
    return asdict(_DESCRIPTOR)


def _sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Atomically write strict-JSON.

    Returns:
        Hex digest of the written bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return _sha256_bytes(text.encode("utf-8"))


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a failure reason when the request demands a newer component.

    Returns:
        Failure reason string, or None when the component version satisfies the request.
    """
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
        An unavailable/failed result, or None when the request applies.
    """
    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"unsupported_component: {request.component_id}",
        )
    supported = set(_DESCRIPTOR.required_capabilities) | set(_DESCRIPTOR.optional_capabilities)
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(sorted(missing))}",
        )
    version_error = _check_version_compatible(request.config)
    if version_error is not None:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=version_error,
        )
    return None


def _resolve_source(path_value: str, base: Path) -> Path:
    """Resolve a source URI under the base directory.

    Returns:
        Resolved path; absolute URIs and traversal are rejected.
    """
    candidate = Path(path_value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReviewContractsValidationError(
            [f"source uri rejected (absolute or traversal): {path_value}"]
        )
    return base / candidate


def _probe_episode_id(record: Any, fallback: str) -> str:
    """Return the probed episode id, falling back to a synthetic scoped id."""
    if isinstance(record, dict):
        for key in ("episode_id", "id"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    return fallback


def _probe_text(record: Any, keys: tuple[str, ...], default: str = "unknown") -> str:
    """Return the first non-empty probed string field.

    Returns:
        Probed value, or the default when no field applies.
    """
    if isinstance(record, dict):
        for key in keys:
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    return default


def _index_episode_jsonl(
    raw: bytes, *, artifact_id: str, uri: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Index one JSONL source without rewriting its payload.

    Returns:
        Indexed entries plus non-fatal diagnostic codes.
    """
    entries: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [], [f"{artifact_id}: source_not_utf8"]
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], [f"{artifact_id}: source_empty"]
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            diagnostics.append(f"{artifact_id}: corrupt_json_line:{index}")
            continue
        if not isinstance(record, dict):
            diagnostics.append(f"{artifact_id}: corrupt_json_line:{index}")
            continue
        episode_id = _probe_episode_id(record, f"{artifact_id}#line{index}")
        entries.append(
            {
                "episode_id": episode_id,
                "artifact_id": f"{artifact_id}#line{index}",
                "uri": uri,
                "format": "episode-jsonl",
                "units": _probe_text(record, ("units",)),
                "actor_ids": (
                    [str(item) for item in record["actor_ids"]]
                    if isinstance(record.get("actor_ids"), list)
                    else []
                ),
                "evidence_status": _probe_text(record, ("evidence_status", "status")),
            }
        )
    return entries, diagnostics


def _index_single_document(
    raw: bytes, *, artifact_id: str, uri: str, family: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Index one whole-file source (trace, dossier, or video metadata).

    Returns:
        Single indexed entry plus non-fatal diagnostic codes.
    """
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [], [f"{artifact_id}: source_not_json"]
    if not isinstance(record, dict):
        return [], [f"{artifact_id}: source_not_json_object"]
    entry: dict[str, Any] = {
        "episode_id": _probe_episode_id(record, artifact_id),
        "artifact_id": artifact_id,
        "uri": uri,
        "format": family,
        "units": _probe_text(record, ("units",)),
        "actor_ids": [],
        "evidence_status": _probe_text(record, ("evidence_status", "status")),
    }
    diagnostics: list[str] = []
    if family == "video-metadata" and "telemetry" not in record:
        diagnostics.append(f"{artifact_id}: video_source_without_telemetry")
    return [entry], diagnostics


def _reference_for(
    entry: dict[str, Any], *, file_sha256: str, declared: dict[str, Any], family: str
) -> dict[str, Any]:
    """Project one indexed entry onto a valid ``review-bundle.v1`` reference.

    Returns:
        Bundle reference dict; undeclared commit hashes become an explicit
        zero placeholder so the reference stays schema-valid.
    """
    commit = declared.get("source_commit", "")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(c not in "0123456789abcdefABCDEF" for c in commit)
    ):
        commit = _ZERO_COMMIT
    units = declared.get("units") or entry.get("units") or "unknown"
    frame = declared.get("coordinate_frame") or "unknown"
    return {
        "artifact_id": entry["artifact_id"],
        "uri": entry["uri"],
        "format": entry["format"],
        "schema": declared.get("schema") or family,
        "sha256": file_sha256,
        "source_commit": commit,
        "config_identity": declared.get("config_identity", ""),
        "units": units,
        "coordinate_frame": frame,
    }


def _import_source(
    artifact_id: str, uri: str, family: str, declared: dict[str, Any], root: Path
) -> _ImportOutcome:
    """Read, hash, and index one source without touching its bytes.

    Returns:
        Import outcome with entries, file digest, and diagnostic codes.
    """
    try:
        raw = _resolve_source(uri, root).read_bytes()
    except OSError:
        return _ImportOutcome(diagnostics=[f"{artifact_id}: source_unreadable"])
    file_sha = _sha256_bytes(raw)
    expected_sha = declared.get("sha256", "")
    diagnostics: list[str] = []
    if expected_sha and expected_sha != file_sha:
        diagnostics.append(f"{artifact_id}: stale_digest")
    if family == "episode-jsonl":
        entries, problems = _index_episode_jsonl(raw, artifact_id=artifact_id, uri=uri)
    else:
        entries, problems = _index_single_document(
            raw, artifact_id=artifact_id, uri=uri, family=family
        )
    diagnostics.extend(problems)
    if not isinstance(declared.get("source_commit"), str) or len(declared["source_commit"]) != 40:
        diagnostics.append(f"{artifact_id}: source_commit_placeholder")
    return _ImportOutcome(entries=entries, file_sha256=file_sha, diagnostics=diagnostics)


def _finalize(
    request: ComponentRequest,
    outcomes: dict[str, _ImportOutcome],
    skipped_optional: list[str],
    root: Path,
) -> ComponentResult:
    """Assemble the bundle and capability report from per-source outcomes.

    Returns:
        Complete result only when every source indexed cleanly, else partial.
    """
    diagnostics: list[str] = list(skipped_optional)
    episodes: dict[str, dict[str, Any]] = {}
    for artifact_id, outcome in outcomes.items():
        diagnostics.extend(outcome.diagnostics)
        for entry in outcome.entries:
            if entry["episode_id"] in episodes:
                diagnostics.append(f"{artifact_id}: duplicate_episode_id:{entry['episode_id']}")
                continue
            declared = (request.config.get("source_metadata") or {}).get(
                entry["artifact_id"].split("#")[0], {}
            )
            family = entry["format"]
            episodes[entry["episode_id"]] = _reference_for(
                entry, file_sha256=outcome.file_sha256, declared=declared, family=family
            )
    if not episodes:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="no_episodes_indexed: " + "; ".join(sorted(set(diagnostics))[:5]),
        )
    bundle_payload = {
        "schema_version": "review-bundle.v1",
        "bundle_id": f"{request.request_id}-bundle",
        "episodes": [
            {"episode_id": episode_id, "references": [reference]}
            for episode_id, reference in sorted(episodes.items())
        ],
    }
    bundle = review_bundle_from_dict(bundle_payload)
    bundle_digest = review_bundle_canonical_digest(bundle)
    capability_payload = {
        "missing_capabilities": [],
        "skipped_optional_streams": sorted(skipped_optional),
        "diagnostics": sorted(set(diagnostics)),
        "families": sorted(SOURCE_FAMILIES),
    }
    output_dir = root / request.output_directory
    _write_json(output_dir / OUTPUT_BUNDLE_FILENAME, bundle_payload)
    _write_json(output_dir / OUTPUT_CAPABILITY_FILENAME, capability_payload)
    informational = {
        "source_commit_placeholder",
        "video_source_without_telemetry",
        "optional_stream_skipped",
    }
    blocking = [item for item in diagnostics if not any(tag in item for tag in informational)]
    partial = bool(blocking) or bool(skipped_optional)
    status = STATUS_PARTIAL if partial else STATUS_COMPLETE
    reason = "" if status == STATUS_COMPLETE else "; ".join(sorted(set(diagnostics))[:8])
    # The shared result contract forbids artifacts on non-complete results;
    # index files stay on disk as the useful-artifact example either way.
    artifacts: tuple[dict[str, Any], ...] = (
        (
            {
                "artifact_id": OUTPUT_BUNDLE_FILENAME,
                "uri": str(Path(request.output_directory) / OUTPUT_BUNDLE_FILENAME),
                "sha256": bundle_digest,
            },
        )
        if status == STATUS_COMPLETE
        else ()
    )
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status=status,
        artifacts=artifacts,
        diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
        provenance={
            "output_directory": request.output_directory,
            "bundle_digest": bundle_digest,
            "episodes": len(episodes),
        },
        reason=reason,
    )


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Import referenced sources into a ``review-bundle.v1`` index.

    Args:
        request: Validated component request.
        base: Base directory source URIs and the output directory resolve under.

    Returns:
        Component result: ``complete`` only when every source indexed cleanly,
        ``partial`` when any source was skipped or flagged, ``unavailable`` when
        the component or a required capability does not apply, ``failed`` on
        validation, version, collision, or internal errors.
    """
    root = base if base is not None else Path.cwd()
    rejected = _reject_not_applicable(request)
    if rejected is not None:
        return rejected
    output_dir = root / request.output_directory
    if output_dir.exists():
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=f"output_collision: already exists: {request.output_directory}",
        )
    declared_sources = request.config.get("source_metadata")
    if declared_sources is not None and not isinstance(declared_sources, dict):
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="invalid_config: source_metadata must be an object",
        )
    declared_sources = declared_sources or {}
    outcomes: dict[str, _ImportOutcome] = {}
    skipped_optional: list[str] = []
    try:
        for ref in request.sources:
            family = ref.format
            if family not in SOURCE_FAMILIES + OPTIONAL_FAMILIES:
                outcomes[ref.artifact_id] = _ImportOutcome(
                    diagnostics=[f"{ref.artifact_id}: unknown_source_format:{family}"]
                )
                continue
            if family in OPTIONAL_FAMILIES and family not in request.required_capabilities:
                skipped_optional.append(f"{ref.artifact_id}: optional_stream_skipped:{family}")
                continue
            declared = declared_sources.get(ref.artifact_id, {})
            if not isinstance(declared, dict):
                declared = {}
            outcomes[ref.artifact_id] = _import_source(
                ref.artifact_id, ref.uri, family, declared, root
            )
        return _finalize(request, outcomes, skipped_optional, root)
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="; ".join(error.errors),
        )


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the review-import component."""
    parser = argparse.ArgumentParser(description="Import sources into review-bundle.v1.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def _merge_config(payload: dict[str, Any], config_path: str) -> dict[str, Any]:
    """Merge an optional config file into the request payload.

    Returns:
        Payload with merged config mapping.
    """
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
    if isinstance(config, dict):
        payload = {**payload, "config": {**payload.get("config", {}), **config}}
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-import component.

    Returns:
        Process exit code (0 only when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read request: {error}"]) from error
    if args.config is not None:
        payload = _merge_config(payload, args.config)
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
