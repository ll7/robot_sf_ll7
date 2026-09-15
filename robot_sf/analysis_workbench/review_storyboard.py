"""SREV-07 review-storyboard component: ranked candidates plus storyboard spec.

This module owns the SREV-07 leaf surface only: a ``run(request)`` adapter plus a
standalone CLI that consumes the SREV-01 shared contracts, ranks bundle episodes
into storyboard candidates with stable tie-breaking, and emits a deterministic
``visualization-spec.v1`` document with override provenance. User overrides
survive regeneration; stale override sources are detected, never applied blindly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
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

_DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("visualization-spec.v1", "override-provenance.v1"),
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)


@dataclass
class _Candidate:
    """One ranked storyboard candidate."""

    episode_id: str
    score: float = 0.0
    score_source: str = "default-zero"
    pinned: bool = False
    excluded: bool = False


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


def _canonical_digest(payload: Any) -> str:
    """Return the stable logical digest of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a failure reason when the request demands a newer component.

    Returns:
        Failure reason string, or None when the component version satisfies it.
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


def _read_json(artifact_id: str, uri: str, root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read and parse one JSON source document.

    Returns:
        Tuple of (payload or None, diagnostic code or None).
    """
    try:
        payload = json.loads(_resolve_source(uri, root).read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, f"{artifact_id}: source_unreadable"
    if not isinstance(payload, dict):
        return None, f"{artifact_id}: source_not_json_object"
    return payload, None


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
) -> tuple[list[_Candidate], list[str]]:
    """Rank candidates with stable tie-breaking and explicit overrides.

    Returns:
        Tuple of (ranked kept candidates, diagnostic codes).
    """
    diagnostics: list[str] = []
    known = set(episode_ids)
    for pinned in pins:
        if pinned not in known:
            diagnostics.append(f"override_pin_unknown:{pinned}")
    for excluded in excludes:
        if excluded not in known:
            diagnostics.append(f"override_exclude_unknown:{excluded}")
    candidates = [
        _Candidate(
            episode_id=episode_id,
            score=scores.get(episode_id, 0.0),
            score_source="exemplar" if episode_id in scores else "default-zero",
            pinned=episode_id in set(pins),
            excluded=episode_id in set(excludes),
        )
        for episode_id in episode_ids
    ]
    kept = [c for c in candidates if not c.excluded]
    if excludes:
        diagnostics.append(f"excluded_episodes:{','.join(sorted(set(excludes) & known))}")
    kept.sort(key=lambda c: (not c.pinned, -c.score, c.episode_id))
    return kept, diagnostics


def _clip_intervals(
    intervals: list[dict[str, Any]], duration_s: float, diagnostics: list[str]
) -> list[dict[str, Any]]:
    """Clip event intervals into the source duration, preserving identity.

    Returns:
        Clipped interval entries for the storyboard spec.
    """
    clipped: list[dict[str, Any]] = []
    for item in intervals:
        if not isinstance(item, dict):
            continue
        start = _finite_number(item.get("start_s"))
        end = _finite_number(item.get("end_s"))
        if start is None or end is None or not end > start:
            continue
        if start >= duration_s or end <= 0.0:
            diagnostics.append("interval_outside_duration")
            continue
        clipped.append({"start_s": max(start, 0.0), "end_s": min(end, duration_s)})
    return clipped


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Rank storyboard candidates and emit a deterministic storyboard spec.

    Args:
        request: Validated component request referencing a review bundle.
        base: Base directory source URIs and the output directory resolve under.

    Returns:
        Component result: ``complete`` only when ranking resolved cleanly,
        ``partial`` on skipped/unresolvable inputs, ``unavailable`` when the
        component or a required capability does not apply, ``failed`` on
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
    diagnostics: list[str] = []
    try:
        bundle_doc, scores_doc, index_doc = _load_inputs(request, root, diagnostics)
        if bundle_doc is None:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="required_source_family_missing: " + "; ".join(sorted(set(diagnostics))[:5]),
            )
        for family in request.required_capabilities:
            if family in OPTIONAL_CAPABILITIES and (
                (family == "exemplar-scores" and scores_doc is None)
                or (family == "event-index" and index_doc is None)
            ):
                diagnostics.append(f"required_optional_source_missing:{family}")
        episode_ids = [e["episode_id"] for e in bundle_doc["episodes"]]
        scores = _score_table(scores_doc, episode_ids, diagnostics)
        overrides = _parse_overrides(request.config, bundle_doc, diagnostics)
        kept, rank_diagnostics = _rank_candidates(
            episode_ids, scores, overrides["pin"], overrides["exclude"]
        )
        diagnostics.extend(rank_diagnostics)
        duration = _finite_number(request.config.get("source_duration_s"))
        if duration is None or duration <= 0:
            diagnostics.append("source_duration_missing_or_invalid")
            duration = None
        intervals = _spec_intervals(index_doc, duration, diagnostics)
        spec_payload = {
            "schema_version": "visualization-spec.v1",
            "spec_id": f"{request.request_id}-storyboard",
            "sources": [{"artifact_id": c.episode_id} for c in kept],
            "annotations": [
                {
                    "ranking": [
                        {
                            "episode_id": c.episode_id,
                            "score": c.score,
                            "score_source": c.score_source,
                            "pinned": c.pinned,
                        }
                        for c in kept
                    ],
                    "tie_break_method": TIE_BREAK_METHOD,
                }
            ],
        }
        if intervals is not None:
            spec_payload["source_intervals"] = intervals
        spec = visualization_spec_from_dict(spec_payload)
        override_payload = {
            "schema_version": "override-provenance.v1",
            "pins": sorted(set(overrides["pin"])),
            "excludes": sorted(set(overrides["exclude"])),
            "overrides_applied": overrides["fresh"],
            "diagnostics": sorted(set(diagnostics)),
        }
        spec_digest = _write_json(output_dir / "storyboard-spec.json", spec_payload)
        _write_json(output_dir / "override-provenance.json", override_payload)
        _write_json(
            output_dir / OUTPUT_CAPABILITY_FILENAME,
            {"missing_capabilities": [], "diagnostics": sorted(set(diagnostics))},
        )
        _ = spec
        informational = {"optional_stream_skipped", "exemplar_scores_absent", "excluded_episodes"}
        blocking = [item for item in diagnostics if not any(tag in item for tag in informational)]
        partial = bool(blocking)
        status = STATUS_PARTIAL if partial else STATUS_COMPLETE
        reason = "" if status == STATUS_COMPLETE else "; ".join(sorted(set(diagnostics))[:8])
        artifacts: tuple[dict[str, Any], ...] = (
            (
                {
                    "artifact_id": "storyboard-spec.json",
                    "uri": str(Path(request.output_directory) / "storyboard-spec.json"),
                    "sha256": spec_digest,
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
                "candidates": len(kept),
            },
            reason=reason,
        )
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="; ".join(error.errors),
        )


def _read_family_docs(
    by_format: dict[str, list[Any]], root: Path, diagnostics: list[str]
) -> dict[str, dict[str, Any]]:
    """Read one JSON document per present family.

    Returns:
        Mapping of family name to parsed document.
    """
    docs: dict[str, dict[str, Any]] = {}
    for family, refs in by_format.items():
        payload, problem = _read_json(refs[0].artifact_id, refs[0].uri, root)
        if problem is not None or payload is None:
            diagnostics.append(problem or f"{refs[0].artifact_id}: source_unreadable")
            continue
        docs.setdefault(family, payload)
    return docs


def _load_inputs(
    request: ComponentRequest, root: Path, diagnostics: list[str]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Load bundle, scores, and event-index sources by family.

    Returns:
        Tuple of (bundle document or None, scores document or None,
        event-index document or None).
    """
    by_format: dict[str, list[Any]] = {}
    for ref in request.sources:
        if ref.format not in REQUIRED_CAPABILITIES + OPTIONAL_CAPABILITIES:
            diagnostics.append(f"{ref.artifact_id}: unknown_source_format:{ref.format}")
            continue
        if ref.format in OPTIONAL_CAPABILITIES and ref.format not in request.required_capabilities:
            diagnostics.append(f"{ref.artifact_id}: optional_stream_skipped:{ref.format}")
            continue
        by_format.setdefault(ref.format, []).append(ref)
    for family in REQUIRED_CAPABILITIES:
        if family not in by_format:
            diagnostics.append(f"required_source_family_missing:{family}")
    if any(code.startswith("required_source_family_missing") for code in diagnostics):
        return None, None, None
    docs = _read_family_docs(by_format, root, diagnostics)
    return _validated_bundle_docs(docs, diagnostics)


def _validated_bundle_docs(
    docs: dict[str, dict[str, Any]], diagnostics: list[str]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Validate the bundle document and split optional inputs.

    Returns:
        Tuple of (bundle document or None, scores document or None,
        event-index document or None).
    """
    bundle_doc = docs.get("review-bundle")
    if bundle_doc is not None:
        try:
            review_bundle_from_dict(bundle_doc)
        except ReviewContractsValidationError as error:
            diagnostics.append("bundle_invalid:" + error.errors[0][:80])
            bundle_doc = None
    return bundle_doc, docs.get("exemplar-scores"), docs.get("event-index")


def _score_table(
    scores_doc: dict[str, Any] | None, episode_ids: list[str], diagnostics: list[str]
) -> dict[str, float]:
    """Extract exemplar scores for known episodes.

    Returns:
        Mapping of episode id to score (episodes without scores default to zero).
    """
    if scores_doc is None:
        diagnostics.append("exemplar_scores_absent:default_zero")
        return {}
    raw = scores_doc.get("scores")
    if not isinstance(raw, dict):
        diagnostics.append("exemplar_scores_malformed:default_zero")
        return {}
    known = set(episode_ids)
    scores: dict[str, float] = {}
    for episode_id, value in raw.items():
        number = _finite_number(value)
        if episode_id not in known:
            diagnostics.append(f"score_unknown_episode:{episode_id}")
            continue
        if number is None:
            diagnostics.append(f"score_malformed:{episode_id}")
            continue
        scores[episode_id] = number
    return scores


def _parse_overrides(
    config: dict[str, Any], bundle_doc: dict[str, Any], diagnostics: list[str]
) -> dict[str, Any]:
    """Parse user overrides with stale-source detection.

    Returns:
        Mapping with pin/exclude lists and whether they were applied fresh.
    """
    raw = config.get("overrides", {})
    if not isinstance(raw, dict):
        diagnostics.append("overrides_malformed:ignored")
        return {"pin": [], "exclude": [], "fresh": False}
    pins = [str(i) for i in raw.get("pin", []) if isinstance(i, str)]
    excludes = [str(i) for i in raw.get("exclude", []) if isinstance(i, str)]
    declared = raw.get("source_digest", "")
    if declared:
        observed = _canonical_bundle_digest(bundle_doc)
        if declared != observed:
            diagnostics.append("stale_override_source:overrides_ignored")
            return {"pin": [], "exclude": [], "fresh": False}
    return {"pin": pins, "exclude": excludes, "fresh": True}


def _canonical_bundle_digest(bundle_doc: dict[str, Any]) -> str:
    """Return the stable logical digest of a bundle document."""
    encoded = json.dumps(bundle_doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _spec_intervals(
    index_doc: dict[str, Any] | None, duration: float | None, diagnostics: list[str]
) -> list[dict[str, Any]] | None:
    """Derive clipped spec intervals from an event index, if provided.

    Returns:
        Interval list, or None when no event index applies.
    """
    if index_doc is None or duration is None:
        return None
    raw = index_doc.get("intervals")
    if not isinstance(raw, list):
        diagnostics.append("event_index_intervals_missing")
        return None
    clipped = _clip_intervals(raw, duration, diagnostics)
    return [{"start_s": c["start_s"], "end_s": c["end_s"]} for c in clipped]


def _clip_intervals(
    intervals: list[Any], duration_s: float, diagnostics: list[str]
) -> list[dict[str, Any]]:
    """Clip raw event rows into the source duration, preserving identity.

    Returns:
        Clipped interval dicts with start_s/end_s.
    """
    clipped: list[dict[str, Any]] = []
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
        clipped.append({"start_s": max(start, 0.0), "end_s": min(end, duration_s)})
    return clipped


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the review-storyboard component."""
    parser = argparse.ArgumentParser(description="Rank storyboard candidates.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-storyboard component.

    Returns:
        Process exit code (0 only when the result status is complete).
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
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
