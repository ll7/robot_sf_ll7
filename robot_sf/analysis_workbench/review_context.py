"""SREV-06 review-context component: cohort context reports from campaign results.

This module owns the SREV-06 leaf surface only: a ``run(request)`` adapter plus a
standalone CLI that consumes the SREV-01 shared contracts and builds a cohort
context report (denominators, outcome frequencies, metric positions, selection
coverage) from explicit campaign-result inputs. Repeated excerpts never inflate
counts; missing metrics retain denominator and missing count; tie percentiles
are deterministic and documented; absent campaign references yield unavailable
context, never population inference.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
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
)

COMPONENT_ID = "srev06-review-context"
COMPONENT_VERSION = "1.0.0"

REQUIRED_CAPABILITIES = ("campaign-result",)
OPTIONAL_CAPABILITIES = ("episode-selection",)

OUTPUT_REPORT_FILENAME = "context-report.json"
OUTPUT_HTML_FILENAME = "context-report.html"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

PERCENTILE_METHOD = "linear-interpolation-on-sorted-values"

_DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("review-context.v1", "missing-capability-report.v1"),
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)


@dataclass
class _Episode:
    """One deduplicated episode with metric values."""

    episode_id: str
    seed: Any = None
    config_id: str = "unknown"
    outcome: str = "unknown"
    metrics: dict[str, Any] = field(default_factory=dict)


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


def _finite_number(value: Any) -> float | None:
    """Return a finite float, or None for malformed input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolate one percentile over sorted values (deterministic).

    Returns:
        Interpolated percentile value.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = fraction * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _parse_episodes(raw: Any, diagnostics: list[str]) -> list[_Episode]:
    """Parse and deduplicate campaign episodes, preserving grain fields.

    Returns:
        Deduplicated episodes in first-seen order.
    """
    episodes: list[_Episode] = []
    seen: set[str] = set()
    entries = raw.get("episodes") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        diagnostics.append("episodes_missing_or_empty")
        return []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            diagnostics.append(f"episode_row_{index}_malformed")
            continue
        episode_id = item.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            diagnostics.append(f"episode_row_{index}_missing_id")
            continue
        if episode_id in seen:
            diagnostics.append(f"duplicate_episode_excerpt:{episode_id}")
            continue
        seen.add(episode_id)
        config = item.get("config", {})
        metrics = item.get("metrics", {})
        episodes.append(
            _Episode(
                episode_id=episode_id,
                seed=item.get("seed"),
                config_id=str(config.get("config_id", "unknown"))
                if isinstance(config, dict)
                else "unknown",
                outcome=str(item.get("outcome", "unknown")),
                metrics=dict(metrics) if isinstance(metrics, dict) else {},
            )
        )
    return episodes


def _summarize_metric(values: list[float]) -> dict[str, Any]:
    """Summarize one metric column with deterministic tie percentiles.

    Returns:
        Summary mapping with count, missing-aware stats, and method note.
    """
    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": count,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / count,
        "p25": _percentile(ordered, 0.25),
        "p50": _percentile(ordered, 0.50),
        "p75": _percentile(ordered, 0.75),
        "percentile_method": PERCENTILE_METHOD,
    }


def _build_report(
    episodes: list[_Episode],
    selection: list[str],
    unknown_selection: list[str],
    campaign_id: str | None,
    campaign_present: bool,
    diagnostics: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Build the cohort context report document.

    Returns:
        Tuple of (report document, diagnostic codes).
    """
    seeds = sorted({e.seed for e in episodes if isinstance(e.seed, int)})
    configs = sorted({e.config_id for e in episodes})
    outcomes: dict[str, int] = {}
    for episode in episodes:
        outcomes[episode.outcome] = outcomes.get(episode.outcome, 0) + 1
    metric_names = sorted({name for e in episodes for name in e.metrics})
    metrics: dict[str, Any] = {}
    for name in metric_names:
        observed = [
            float(e.metrics[name])
            for e in episodes
            if _finite_number(e.metrics.get(name)) is not None
        ]
        missing = len(episodes) - len(observed)
        summary: dict[str, Any] = {"missing": missing, "denominator": len(episodes)}
        if observed:
            summary.update(_summarize_metric(observed))
        else:
            summary.update({"count": 0})
        metrics[name] = summary
    known_ids = {e.episode_id for e in episodes}
    covered = sorted(set(selection) & known_ids)
    coverage = {
        "selected": len(covered),
        "denominator": len(episodes),
        "unknown_selected_ids": sorted(unknown_selection),
    }
    if unknown_selection:
        diagnostics.append(f"unknown_selected_ids:{','.join(sorted(unknown_selection))}")
    availability: dict[str, Any] = {"status": "complete", "reason": ""}
    if campaign_id is not None and not campaign_present:
        availability = {
            "status": "unavailable",
            "reason": f"campaign reference {campaign_id!r} has no source payload",
        }
        diagnostics.append("campaign_context_unavailable")
    document = {
        "schema_version": "review-context.v1",
        "grain": {"seeds": seeds, "config_ids": configs, "episodes": len(episodes)},
        "denominator": len(episodes),
        "outcomes": dict(sorted(outcomes.items())),
        "metrics": metrics,
        "selection_coverage": coverage,
        "campaign": {"campaign_id": campaign_id, "availability": availability},
    }
    return document, diagnostics


def _render_html(document: dict[str, Any]) -> str:
    """Render the standalone HTML report table.

    Returns:
        Self-contained HTML document string.
    """
    rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td>"
        f"<td>{m.get('count', 0)}</td><td>{m.get('missing', 0)}</td>"
        f"<td>{m.get('min', '')}</td><td>{m.get('p50', '')}</td>"
        f"<td>{m.get('max', '')}</td></tr>"
        for name, m in sorted(document["metrics"].items())
    )
    outcomes = "\n".join(
        f"<tr><td>{html.escape(str(outcome))}</td><td>{count}</td></tr>"
        for outcome, count in sorted(document["outcomes"].items())
    )
    grain = document["grain"]
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        "<title>Review context report</title></head><body>"
        f"<h1>Review context report</h1><p>Denominator: {document['denominator']} episodes; "
        f"seeds: {html.escape(str(grain['seeds']))}; configs: "
        f"{html.escape(str(grain['config_ids']))}.</p>"
        "<h2>Outcomes</h2><table><tr><th>Outcome</th><th>Count</th></tr>"
        f"{outcomes}</table>"
        "<h2>Metrics</h2><table><tr><th>Metric</th><th>n</th><th>missing</th>"
        f"<th>min</th><th>p50</th><th>max</th></tr>{rows}</table>"
        "</body></html>\n"
    )


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Build a cohort context report from campaign-result inputs.

    Args:
        request: Validated component request with campaign-result sources.
        base: Base directory source URIs and the output directory resolve under.

    Returns:
        Component result: ``complete`` only when every input verified and the
        campaign context (when referenced) resolved, ``partial`` on missing
        metrics-unavailable rows or unresolved references, ``unavailable`` when
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
    diagnostics: list[str] = []
    try:
        payloads = _load_sources(request, root, diagnostics)
        if payloads is None:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="required_source_family_missing: " + "; ".join(sorted(set(diagnostics))[:5]),
            )
        episodes = _parse_episodes(payloads["campaign"], diagnostics)
        episodes = _parse_episodes(payloads["campaign"], diagnostics)
        selection, unknown = _parse_selection(payloads.get("selection"), episodes)
        campaign_id = request.config.get("campaign_id")
        if campaign_id is not None and not isinstance(campaign_id, str):
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="invalid_config: campaign_id must be a string",
            )
        campaign_present = any(
            isinstance(doc, dict) and doc.get("campaign_id") == campaign_id
            for doc in payloads["docs"]
        )
        if not episodes:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="no_episodes_indexed: " + "; ".join(sorted(set(diagnostics))[:5]),
            )
        document, diagnostics = _build_report(
            episodes, selection, unknown, campaign_id, campaign_present, diagnostics
        )
        capability_payload = {
            "missing_capabilities": [],
            "diagnostics": sorted(set(diagnostics)),
        }
        report_digest = _write_json(output_dir / "context-report.json", document)
        (output_dir / "context-report.html").write_text(_render_html(document), encoding="utf-8")
        _write_json(output_dir / OUTPUT_CAPABILITY_FILENAME, capability_payload)
        informational = {"optional_stream_skipped", "duplicate_episode_excerpt"}
        blocking = [item for item in diagnostics if not any(tag in item for tag in informational)]
        partial = bool(blocking)
        status = STATUS_PARTIAL if partial else STATUS_COMPLETE
        reason = "" if status == STATUS_COMPLETE else "; ".join(sorted(set(diagnostics))[:8])
        artifacts: tuple[dict[str, Any], ...] = (
            (
                {
                    "artifact_id": "context-report.json",
                    "uri": str(Path(request.output_directory) / "context-report.json"),
                    "sha256": report_digest,
                },
                {
                    "artifact_id": "context-report.html",
                    "uri": str(Path(request.output_directory) / "context-report.html"),
                    "sha256": _sha256_bytes((output_dir / "context-report.html").read_bytes()),
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
                "episodes": len(episodes),
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


def _read_json_doc(
    artifact_id: str, uri: str, root: Path
) -> tuple[dict[str, Any] | None, str | None]:
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


def _group_sources(
    request: ComponentRequest, diagnostics: list[str]
) -> dict[str, list[Any]] | None:
    """Group sources by family, recording format problems.

    Returns:
        Family-to-refs mapping, or None when a required family is absent.
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
        return None
    return by_format


def _load_sources(
    request: ComponentRequest, root: Path, diagnostics: list[str]
) -> dict[str, Any] | None:
    """Load and validate source documents by family.

    Returns:
        Mapping with the merged campaign document, selection list, and raw docs,
        or None when a required family is absent.
    """
    by_format = _group_sources(request, diagnostics)
    if by_format is None:
        return None
    docs: list[dict[str, Any]] = []
    for ref in by_format.get("campaign-result", []):
        payload, problem = _read_json_doc(ref.artifact_id, ref.uri, root)
        if problem is not None or payload is None:
            diagnostics.append(problem or f"{ref.artifact_id}: source_unreadable")
            continue
        docs.append(payload)
    if not docs:
        diagnostics.append("campaign_payload_missing")
        return None
    merged: dict[str, Any] = {"episodes": []}
    for doc in docs:
        entries = doc.get("episodes")
        if isinstance(entries, list):
            merged["episodes"].extend(entries)
        if isinstance(doc.get("campaign_id"), str):
            merged["campaign_id"] = doc["campaign_id"]
    selection_doc: dict[str, Any] | None = None
    for ref in by_format.get("episode-selection", []):
        payload, problem = _read_json_doc(ref.artifact_id, ref.uri, root)
        if problem is not None:
            diagnostics.append(problem)
            continue
        selection_doc = payload
    return {"campaign": merged, "selection": selection_doc, "docs": docs}


def _parse_selection(
    selection_doc: dict[str, Any] | None, episodes: list[_Episode]
) -> tuple[list[str], list[str]]:
    """Split a selection list into known and unknown episode ids.

    Returns:
        Tuple of (known selected ids, unknown selected ids).
    """
    if selection_doc is None:
        return [], []
    raw = selection_doc.get("selected_episode_ids", [])
    if not isinstance(raw, list):
        return [], []
    known = {episode.episode_id for episode in episodes}
    selected = [str(item) for item in raw if isinstance(item, str)]
    return [i for i in selected if i in known], [i for i in selected if i not in known]


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the review-context component."""
    parser = argparse.ArgumentParser(description="Build cohort context reports.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-context component.

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
