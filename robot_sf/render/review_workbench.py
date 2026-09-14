"""Offline local review workbench: SREV-15 browser artifact/provenance view (issue #9284).

Consumes the v1 scenario-review contracts fixed by #9270 (`review-bundle.v1`,
`visualization-spec.v1`, `component-request.v1`) and produces a self-contained,
offline browser view with a shared time/actor state plus extension slots. The
module is a thin contract adapter: it validates and indexes what the canonical
owners already produce, reuses :mod:`robot_sf.render.threejs_viewer` for scene
playback when a recording is declared, and never re-derives simulation semantics.

Boundaries this module enforces (issue #9284):

- simulation time is telemetry authority; a synchronized video overlay requires an
  explicit ``presentation_timestamp_map`` — frame/fps alignment is never guessed;
- actor and event ids stay episode-scoped;
- a missing measurement or artifact is reported ``unavailable`` with a reason, never
  silently dropped and never fabricated;
- source integrity (recorded ``sha256``) is reported separately from evidence
  admission, which this component does not grant;
- sources are read-only; only the requested output directory is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    RESULT_STATUSES,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
    review_bundle_from_dict,
    visualization_spec_from_dict,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

COMPONENT_ID = "srev15-review-workbench"
COMPONENT_VERSION = "1.0.0"
WORKBENCH_SCHEMA_VERSION = "review-workbench.v1"
PRESENTATION_PLAN_SCHEMA_VERSION = "presentation-plan.v1"
DESCRIPTOR_SCHEMA_VERSION = "component-descriptor.v1"

BUNDLE_FORMAT = "review-bundle.v1"
SPEC_FORMAT = "visualization-spec.v1"
RECORDING_FORMATS = ("recording-jsonl.v1", "jsonl-recording.v1")
VIDEO_FORMATS = ("video-mp4.v1", "video-frames.v1")

DEFAULT_PRESENTATION = {"width": 1920, "height": 1080, "fps": 30.0, "speed": 1.0}
TEST_PRESENTATION = {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}

REQUIRED_CAPABILITIES = ("scene-time-sync", "actor-identity", "artifact-provenance")
OPTIONAL_CAPABILITIES = ("extension-slots", "matplotlib-figures", "threejs-scene", "video-frames")
OUTPUT_TYPES = (
    "review-workbench.v1",
    "review-bundle-index.v1",
    "presentation-plan.v1",
    "threejs-viewer.v1",
)

DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=OUTPUT_TYPES,
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)


def descriptor_document() -> dict[str, Any]:
    """Return the schema-shaped capability descriptor for this component.

    Returns:
        ``component-descriptor.v1`` document for the review-workbench component.
    """
    payload = asdict(DESCRIPTOR)
    for key in (
        "supported_input_versions",
        "output_types",
        "required_capabilities",
        "optional_capabilities",
    ):
        payload[key] = list(payload[key])
    return {"schema_version": DESCRIPTOR_SCHEMA_VERSION, **payload}


def _resolve_output_dir(request: ComponentRequest, base: Path) -> Path:
    """Fail closed on an existing output directory so sources and prior runs survive.

    Returns:
        Resolved output directory path that does not exist yet.
    """
    output_dir = base / request.output_directory
    if output_dir.exists():
        raise ReviewContractsValidationError(
            [f"/output_directory: output collision, already exists: {request.output_directory}"]
        )
    return output_dir


def _write_text(path: Path, text: str) -> str:
    """Write one artifact atomically inside the requested output directory.

    Returns:
        SHA-256 digest of the written text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Write one JSON artifact with sorted keys and strict (NaN-free) encoding.

    Returns:
        SHA-256 digest of the written document.
    """
    return _write_text(path, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _read_source(base: Path, uri: str) -> tuple[bytes | None, str]:
    """Read one declared source inside the base directory.

    Returns:
        Tuple of (payload bytes or ``None``, stable status token).
    """
    candidate = (base / uri).resolve()
    root = base.resolve()
    if root not in candidate.parents and candidate != root:
        return None, "source_outside_base"
    if not candidate.is_file():
        return None, "source_missing"
    try:
        return candidate.read_bytes(), "ok"
    except OSError:
        return None, "source_unreadable"


def _verify_bundle_references(
    bundle: Any, base: Path, diagnostics: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Verify each bundle-declared artifact against its recorded digest.

    Integrity is reported separately from evidence admission: a mismatch is recorded and
    the artifact keeps ``admission: not_evaluated``. A reference whose file is absent is
    reported ``unavailable`` with a reason instead of being dropped.

    Returns:
        Verified artifact records in bundle order.
    """
    verified: list[dict[str, Any]] = []
    for episode in bundle.episodes:
        for ref in episode["references"]:
            record = {
                "episode_id": str(episode["episode_id"]),
                "artifact_id": str(ref["artifact_id"]),
                "uri": str(ref["uri"]),
                "format": str(ref["format"]),
                "declared_sha256": str(ref["sha256"]),
                "admission": "not_evaluated",
            }
            payload, status = _read_source(base, str(ref["uri"]))
            if payload is None:
                record["availability"] = status
                record["integrity"] = "unverifiable"
                diagnostics.append(
                    {
                        "artifact_id": record["artifact_id"],
                        "reason_code": status,
                        "detail": f"bundle-declared artifact is {status.replace('source_', '')}",
                    }
                )
            else:
                digest = hashlib.sha256(payload).hexdigest()
                record["availability"] = "ok"
                record["computed_sha256"] = digest
                record["integrity"] = "match" if digest == record["declared_sha256"] else "mismatch"
                if record["integrity"] == "mismatch":
                    diagnostics.append(
                        {
                            "artifact_id": record["artifact_id"],
                            "reason_code": "source_integrity_mismatch",
                            "detail": (
                                "bundle-declared sha256 does not match the artifact on disk; "
                                "integrity is reported without granting admission"
                            ),
                        }
                    )
            verified.append(record)
    return verified


def _load_declared_sources(
    request: ComponentRequest, base: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Load every declared source, keeping integrity separate from admission.

    Returns:
        Tuple of (source index, diagnostics, presentation config, verified bundle artifacts).
    """
    entries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    presentation: dict[str, Any] = {}
    verified: list[dict[str, Any]] = []
    for ref in request.sources:
        payload, status = _read_source(base, ref.uri)
        digest = hashlib.sha256(payload).hexdigest() if payload is not None else None
        entry: dict[str, Any] = {
            "artifact_id": ref.artifact_id,
            "uri": ref.uri,
            "format": ref.format,
            "computed_sha256": digest,
            "availability": status,
            "integrity": (
                "unverifiable"
                if not ref.sha256 or digest is None
                else ("match" if digest == ref.sha256 else "mismatch")
            ),
            "admission": "not_evaluated",
        }
        if payload is None:
            entry["reason_code"] = status
            diagnostics.append(
                {
                    "artifact_id": ref.artifact_id,
                    "reason_code": status,
                    "detail": f"declared source is {status.replace('source_', '')}",
                }
            )
            entries.append(entry)
            continue
        if ref.format == BUNDLE_FORMAT:
            try:
                bundle = review_bundle_from_dict(json.loads(payload), source=ref.uri)
            except (ReviewContractsValidationError, json.JSONDecodeError) as error:
                entry["availability"] = "invalid"
                entry["reason_code"] = "source_contract_invalid"
                diagnostics.append(
                    {
                        "artifact_id": ref.artifact_id,
                        "reason_code": "source_contract_invalid",
                        "detail": str(error)[:400],
                    }
                )
                entries.append(entry)
                continue
            entry["episode_ids"] = [str(episode["episode_id"]) for episode in bundle.episodes]
            verified.extend(_verify_bundle_references(bundle, base, diagnostics))
        elif ref.format == SPEC_FORMAT:
            try:
                spec = visualization_spec_from_dict(json.loads(payload), source=ref.uri)
            except (ReviewContractsValidationError, json.JSONDecodeError) as error:
                entry["availability"] = "invalid"
                entry["reason_code"] = "source_contract_invalid"
                diagnostics.append(
                    {
                        "artifact_id": ref.artifact_id,
                        "reason_code": "source_contract_invalid",
                        "detail": str(error)[:400],
                    }
                )
                entries.append(entry)
                continue
            declared = spec.document.get("presentation")
            if isinstance(declared, dict):
                presentation.update(declared)
        elif ref.format in VIDEO_FORMATS:
            # A declared video stream is a presentation source: it never blocks the
            # workbench, and it only ever syncs through an explicit timestamp map.
            entry["stream"] = "video"
        elif ref.format in RECORDING_FORMATS:
            # A recording is a scene source, not a bundle/spec: it never blocks the
            # workbench, and the canonical viewer owner is only invoked when asked.
            entry["scene_export"] = "available" if _threejs_requested(request) else "not_requested"
        else:
            entry["availability"] = "unsupported_format"
            entry["reason_code"] = "unsupported_source_format"
            diagnostics.append(
                {
                    "artifact_id": ref.artifact_id,
                    "reason_code": "unsupported_source_format",
                    "detail": f"format '{ref.format}' is not consumed by this component",
                }
            )
        entries.append(entry)
    return entries, diagnostics, presentation, verified


def _unsupported_capabilities(request: ComponentRequest) -> list[str]:
    """List requested capabilities the descriptor does not declare.

    Returns:
        Requested capability names the descriptor does not support.
    """
    supported = set(DESCRIPTOR.required_capabilities) | set(DESCRIPTOR.optional_capabilities)
    return [name for name in request.required_capabilities if name not in supported]


def _threejs_requested(request: ComponentRequest) -> bool:
    """Report whether the request explicitly requires the scene-export capability.

    Returns:
        True when ``threejs-scene`` is a required capability.
    """
    return "threejs-scene" in request.required_capabilities


def export_declared_recordings(
    request: ComponentRequest,
    entries: Sequence[Mapping[str, Any]],
    output_dir: Path,
    root: Path,
) -> list[dict[str, Any]]:
    """Reuse the canonical Three.js viewer owner for explicitly required recordings.

    Returns:
        Artifact records for the exported viewer directories.
    """
    from robot_sf.render.threejs_viewer import export_threejs_viewer  # noqa: PLC0415

    artifacts: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("scene_export") != "available":
            continue
        recording_path = root / str(entry["uri"])
        target = output_dir / "threejs" / str(entry["artifact_id"])
        result = export_threejs_viewer(recording_path, target)
        for path in (result.html_path, result.scene_path):
            artifacts.append(
                {
                    "artifact_id": str(path.relative_to(output_dir)),
                    "uri": str(Path(request.output_directory) / path.relative_to(output_dir)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return artifacts


def _presentation_block(
    presentation: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve the effective presentation settings from the spec and request config.

    Returns:
        Mapping of effective export settings plus the recorded edits.
    """
    preset = str(config.get("preset", presentation.get("preset", "default")))
    defaults = TEST_PRESENTATION if preset == "test" else DEFAULT_PRESENTATION
    effective = {
        "width": int(presentation.get("width", defaults["width"])),
        "height": int(presentation.get("height", defaults["height"])),
        "fps": float(presentation.get("fps", defaults["fps"])),
        "speed": float(presentation.get("speed", defaults["speed"])),
        "preset": preset,
    }
    recorded = {
        key: list(presentation.get(key, []))
        for key in ("cuts", "pauses", "crops")
        if isinstance(presentation.get(key), list)
    }
    return {"effective": effective, "recorded": recorded}


def _has_video_source(entries: Iterable[Mapping[str, Any]]) -> bool:
    """Report whether any declared source is a video stream.

    Returns:
        True when at least one declared source carries a video format.
    """
    return any(str(entry.get("format", "")) in VIDEO_FORMATS for entry in entries)


def _episode_index(bundle_entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build the episode-scoped actor/event index from the loaded bundles.

    Returns:
        One index entry per episode, scoped to its declaring bundle.
    """
    index: list[dict[str, Any]] = []
    for entry in bundle_entries:
        for episode_id in entry.get("episode_ids", []):
            index.append({"bundle": entry["artifact_id"], "episode_id": episode_id})
    return index


def _build_workbench_document(
    request: ComponentRequest,
    entries: Sequence[Mapping[str, Any]],
    verified: Sequence[Mapping[str, Any]],
    presentation: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the machine-readable workbench document rendered by the HTML view.

    Returns:
        JSON-safe workbench document consumed by the HTML view.
    """
    return {
        "schema_version": WORKBENCH_SCHEMA_VERSION,
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": request.request_id,
        "output_directory": request.output_directory,
        "episodes": _episode_index(entry for entry in entries if entry.get("episode_ids")),
        "artifacts": [dict(entry) for entry in entries],
        "verified_artifacts": [dict(entry) for entry in verified],
        "time_base": {
            "authority": "simulation_telemetry",
            "video_alignment": (
                "declared_presentation_timestamp_map"
                if presentation.get("presentation_timestamp_map")
                else "unavailable"
            ),
        },
        "presentation": dict(presentation),
        "extension_slots": ["panels", "annotations", "storyboard", "comparison"],
        "diagnostics": [dict(diagnostic) for diagnostic in diagnostics],
        "provenance": {
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "source_directory": "read_only",
            "admission": "not_evaluated",
        },
    }


def _render_html(document: Mapping[str, Any]) -> str:
    """Render the offline browser view with the workbench document embedded.

    Returns:
        Self-contained HTML document with no network references.
    """
    payload = json.dumps(document, sort_keys=True, allow_nan=False)
    safe_payload = payload.replace("</", "<\\/")
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>Robot SF review workbench</title>\n"
        "<style>\n"
        "body{font-family:system-ui,sans-serif;margin:1.5rem;color:#111}\n"
        "section{margin-bottom:1.5rem}\n"
        "table{border-collapse:collapse;width:100%}\n"
        "th,td{border:1px solid #ccc;padding:.3rem .5rem;text-align:left;font-size:.9rem}\n"
        ".slot{border:1px dashed #999;padding:.5rem;margin:.25rem 0;font-size:.9rem}\n"
        ".unavailable{color:#8a1f11}\n"
        "</style>\n</head>\n<body>\n"
        "<h1>Robot SF review workbench</h1>\n"
        '<p id="time-base"></p>\n'
        '<section id="episodes"><h2>Episodes</h2><ul id="episode-list"></ul></section>\n'
        '<section id="artifacts"><h2>Artifact provenance</h2>'
        "<table><thead><tr><th>Artifact</th><th>Format</th><th>Availability</th>"
        '<th>Integrity</th><th>Admission</th></tr></thead><tbody id="artifact-rows"></tbody>'
        "</table></section>\n"
        '<section id="presentation"><h2>Presentation</h2><pre id="presentation-block"></pre></section>\n'
        '<section id="slots"><h2>Extension slots</h2><div id="slot-list"></div></section>\n'
        '<script type="application/json" id="workbench-data">' + safe_payload + "</script>\n"
        "<script>\n"
        "const doc = JSON.parse(document.getElementById('workbench-data').textContent);\n"
        "document.getElementById('time-base').textContent =\n"
        "  'time base: ' + doc.time_base.authority + ' | video alignment: '\n"
        "  + doc.time_base.video_alignment;\n"
        "const episodes = document.getElementById('episode-list');\n"
        "for (const episode of doc.episodes) {\n"
        "  const item = document.createElement('li');\n"
        "  item.textContent = episode.bundle + ' :: ' + episode.episode_id;\n"
        "  episodes.appendChild(item);\n"
        "}\n"
        "const rows = document.getElementById('artifact-rows');\n"
        "for (const artifact of doc.artifacts) {\n"
        "  const row = document.createElement('tr');\n"
        "  for (const value of [artifact.artifact_id, artifact.format, artifact.availability,\n"
        "                       artifact.integrity, artifact.admission]) {\n"
        "    const cell = document.createElement('td');\n"
        "    cell.textContent = String(value);\n"
        "    if (String(value) !== 'ok' && String(value) !== 'match') {\n"
        "      cell.className = 'unavailable';\n"
        "    }\n"
        "    row.appendChild(cell);\n"
        "  }\n"
        "  rows.appendChild(row);\n"
        "}\n"
        "document.getElementById('presentation-block').textContent =\n"
        "  JSON.stringify(doc.presentation, null, 2);\n"
        "const slots = document.getElementById('slot-list');\n"
        "for (const slot of doc.extension_slots) {\n"
        "  const element = document.createElement('div');\n"
        "  element.className = 'slot';\n"
        "  element.textContent = slot;\n"
        "  slots.appendChild(element);\n"
        "}\n"
        "</script>\n</body>\n</html>\n"
    )


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Execute one review-workbench request against the declared sources.

    Args:
        request: Validated component request naming a review bundle and, optionally,
            a visualization spec and a recording.
        base: Base directory the request paths resolve under.

    Returns:
        Component result carrying the written artifacts, diagnostics, provenance and a
        status of ``complete``, ``partial``, ``unavailable`` or ``failed``.
    """
    root = base if base is not None else Path.cwd()
    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"unsupported component: {request.component_id}",
        )
    missing = _unsupported_capabilities(request)
    if missing:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"missing capabilities: {', '.join(sorted(missing))}",
        )
    try:
        output_dir = _resolve_output_dir(request, root)
        entries, diagnostics, presentation, verified = _load_declared_sources(request, root)
        bundle_entries = [entry for entry in entries if entry.get("episode_ids")]
        if not bundle_entries:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status="unavailable",
                reason="no review bundle available in the declared sources",
                diagnostics=tuple(diagnostics),
            )
        plan = _presentation_block(presentation, request.config)
        artifacts: list[dict[str, Any]] = []
        document = _build_workbench_document(request, entries, verified, plan, diagnostics)
        html_name = f"{WORKBENCH_SCHEMA_VERSION}.html"
        html_digest = _write_text(output_dir / html_name, _render_html(document))
        artifacts.append(
            {
                "artifact_id": html_name,
                "uri": str(Path(request.output_directory) / html_name),
                "sha256": html_digest,
            }
        )
        json_name = f"{WORKBENCH_SCHEMA_VERSION}.json"
        artifacts.append(
            {
                "artifact_id": json_name,
                "uri": str(Path(request.output_directory) / json_name),
                "sha256": _write_json(output_dir / json_name, document),
            }
        )
        plan_name = f"{PRESENTATION_PLAN_SCHEMA_VERSION}.json"
        if _has_video_source(entries) and not presentation.get("presentation_timestamp_map"):
            diagnostics.append(
                {
                    "artifact_id": plan_name,
                    "reason_code": "presentation_timestamp_map_missing",
                    "detail": (
                        "video sources are declared without an explicit "
                        "presentation_timestamp_map; frame/fps alignment is not guessed"
                    ),
                }
            )
        else:
            plan_document = {
                "schema_version": PRESENTATION_PLAN_SCHEMA_VERSION,
                "request_id": request.request_id,
                "presentation": plan["effective"],
                "recorded_edits": plan["recorded"],
                "timestamp_map": presentation.get("presentation_timestamp_map"),
            }
            artifacts.append(
                {
                    "artifact_id": plan_name,
                    "uri": str(Path(request.output_directory) / plan_name),
                    "sha256": _write_json(output_dir / plan_name, plan_document),
                }
            )
        artifacts.extend(export_declared_recordings(request, entries, output_dir, root))
        status = "partial" if diagnostics else "complete"
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=status,
            artifacts=tuple(artifacts),
            diagnostics=tuple(diagnostics),
            provenance={
                "output_directory": request.output_directory,
                "component_version": COMPONENT_VERSION,
                "source_integrity": {entry["artifact_id"]: entry["integrity"] for entry in entries},
                "verified_artifact_integrity": {
                    record["artifact_id"]: record["integrity"] for record in verified
                },
                "admission": "not_evaluated",
            },
        )
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason="; ".join(error.errors),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the offline review workbench view for a review bundle."
    )
    parser.add_argument("--input", required=False, default=None, help="Component request JSON.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=False, default=None, help="Output directory.")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    parser.add_argument(
        "--descriptor",
        action="store_true",
        help="Print the component descriptor instead of running a request.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-workbench component.

    Returns:
        Process exit code: 0 for a complete run, 2 for a partial or unavailable run,
        and 1 for a failed or invalid request.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.descriptor:
        print(json.dumps(descriptor_document(), sort_keys=True, indent=2))  # noqa: T201 - CLI output
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --descriptor is used")
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
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    if result.status == "complete":
        return 0
    if result.status in ("partial", "unavailable"):
        return 2
    return 1


__all__ = [
    "COMPONENT_ID",
    "COMPONENT_VERSION",
    "DESCRIPTOR",
    "PRESENTATION_PLAN_SCHEMA_VERSION",
    "RESULT_STATUSES",
    "WORKBENCH_SCHEMA_VERSION",
    "descriptor_document",
    "main",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
