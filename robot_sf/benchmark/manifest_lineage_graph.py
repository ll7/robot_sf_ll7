# ruff: noqa: DOC201
"""Build a compact lineage graph report for research manifest inputs.

This module converts per-manifest lineage backfill analysis into a JSON graph
and Markdown adjacency report.  It is intentionally read-only: it never
rewrites manifests and marks inferred or ambiguous lineage explicitly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json
from robot_sf.benchmark.manifest_lineage import MANDATORY_LINEAGE_FIELDS
from robot_sf.benchmark.manifest_lineage_backfill import (
    FieldBackfillEntry,
    FieldStatus,
    ManifestBackfillPlan,
    analyze_manifest,
)
from robot_sf.common.artifact_paths import get_repository_root

SCHEMA_VERSION = "manifest_lineage_graph.v1"
ID_HASH_HEX_LENGTH = 12


class LineageStatus:
    """Canonical statuses for lineage graph edges and traces."""

    CONNECTED = "connected"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    BLOCKED = "blocked"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class LineageNode:
    """One node in the manifest lineage graph."""

    node_id: str
    kind: str
    label: str
    path: str = ""
    status: str = LineageStatus.CONNECTED
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping."""
        return {
            "id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "path": self.path,
            "status": self.status,
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class LineageEdge:
    """One directed edge in the manifest lineage graph."""

    source: str
    target: str
    kind: str
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping."""
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "reason": self.reason,
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class ArtifactTrace:
    """Trace of one artifact candidate back to its source manifest and contract."""

    artifact_id: str
    artifact_kind: str
    source_manifest_path: str
    claim_boundary: str
    lineage_status: str
    reason: str
    field_statuses: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "source_manifest_path": self.source_manifest_path,
            "claim_boundary": self.claim_boundary,
            "lineage_status": self.lineage_status,
            "reason": self.reason,
            "field_statuses": self.field_statuses,
        }


@dataclass(frozen=True, slots=True)
class ManifestLineageGraph:
    """Full lineage graph report over a set of manifest inputs."""

    schema_version: str
    generated_at_utc: str
    nodes: list[LineageNode]
    edges: list[LineageEdge]
    traces: list[ArtifactTrace]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping."""
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "traces": [t.to_dict() for t in self.traces],
            "summary": self.summary,
        }


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path string when possible."""
    resolved = path.resolve()
    repo_root = get_repository_root().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _sanitize_id(value: str) -> str:
    """Return a graph-node-safe identifier from an arbitrary string."""
    return (
        value.replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "_")
    )


def _collision_safe_id(value: str) -> str:
    """Return a sanitized identifier with a deterministic disambiguation suffix.

    Distinct inputs that sanitize to the same base string (for example
    ``a/b`` and ``a__b``) still receive distinct node IDs because the suffix
    is computed from the original, unsanitized value.
    """
    sanitized = _sanitize_id(value)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:ID_HASH_HEX_LENGTH]
    return f"{sanitized}__{digest}"


def _manifest_node_id(path: str) -> str:
    """Stable node id for a manifest path."""
    return f"manifest__{_collision_safe_id(path)}"


def _field_node_id(manifest_path: str, field_name: str) -> str:
    """Stable node id for a lineage field inside a manifest."""
    return f"field__{_collision_safe_id(manifest_path)}__{field_name}"


def _proxy_node_id(manifest_path: str, proxy_path: str) -> str:
    """Stable node id for a proxy source path inside a manifest."""
    return f"proxy__{_collision_safe_id(manifest_path)}__{_collision_safe_id(proxy_path)}"


def _contract_node_id() -> str:
    """Stable node id for the shared lineage contract."""
    return "lineage_contract"


def _artifact_node_id(artifact_id: str) -> str:
    """Stable node id for an artifact candidate."""
    return f"artifact__{_collision_safe_id(artifact_id)}"


def _now_utc() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _parse_generated_at_utc(value: str) -> str:
    """Validate a user-supplied ISO-8601 UTC timestamp.

    Returns the original string on success so the report keeps the caller's
    exact formatting. Raises ValueError with an actionable message on failure.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid --generated-at-utc timestamp '{value}'; "
            "expected an ISO-8601 datetime such as 2026-06-15T00:00:00+00:00"
        ) from exc
    if parsed.tzinfo != UTC:
        raise ValueError(
            f"invalid --generated-at-utc timestamp '{value}'; "
            "expected a timezone-aware UTC ISO-8601 datetime such as "
            "2026-06-15T00:00:00+00:00"
        )
    return value


def _resolve_manifest_path(
    manifest_value: str,
    *,
    candidate_path: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Resolve a manifest path that may be relative to the candidate file or repo root."""
    candidate = Path(manifest_value)
    if candidate.is_absolute():
        return candidate
    if candidate_path is not None and candidate_path.is_file():
        candidate_relative = candidate_path.parent / candidate
        if candidate_relative.is_file():
            return candidate_relative.resolve()
    repo_root = repo_root or get_repository_root()
    repo_relative = repo_root / candidate
    if repo_relative.is_file():
        return repo_relative.resolve()
    return candidate.resolve()


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    """Return paths with duplicate resolved files removed while preserving order."""
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return deduped


def _lineage_status_from_entry(entry: FieldBackfillEntry) -> str:
    """Map a backfill field status to a lineage graph status."""
    mapping = {
        FieldStatus.PRESENT: LineageStatus.CONNECTED,
        FieldStatus.MISSING: LineageStatus.MISSING,
        FieldStatus.INFERRED: LineageStatus.CONNECTED,
        FieldStatus.AMBIGUOUS: LineageStatus.AMBIGUOUS,
        FieldStatus.BLOCKED: LineageStatus.BLOCKED,
    }
    return mapping.get(entry.status, LineageStatus.INCONCLUSIVE)


def _trace_lineage_status(field_entries: Sequence[FieldBackfillEntry]) -> tuple[str, str]:
    """Classify an entire manifest trace from its field entries.

    Returns:
        Tuple of (lineage_status, reason).
    """
    statuses = {_lineage_status_from_entry(e) for e in field_entries}
    if LineageStatus.BLOCKED in statuses:
        return (
            LineageStatus.BLOCKED,
            "one or more lineage fields are blocked by incompatible nearby values",
        )
    if LineageStatus.AMBIGUOUS in statuses:
        return (
            LineageStatus.AMBIGUOUS,
            "one or more lineage fields have ambiguous inference candidates",
        )
    if LineageStatus.MISSING in statuses:
        return (
            LineageStatus.MISSING,
            "one or more mandatory lineage fields are missing",
        )
    if all(s == LineageStatus.CONNECTED for s in statuses):
        return (
            LineageStatus.CONNECTED,
            "all mandatory lineage fields trace to source manifest and validation contract",
        )
    return (
        LineageStatus.INCONCLUSIVE,
        "lineage state could not be determined conclusively",
    )


def _field_statuses(entries: Sequence[FieldBackfillEntry]) -> dict[str, str]:
    """Return a mapping from field name to lineage status."""
    return {e.field_name: _lineage_status_from_entry(e) for e in entries}


def _analyze_manifest_at_path(path: Path) -> ManifestBackfillPlan:
    """Load and analyze a single manifest file."""
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Manifest payload at {path} must be a JSON object")
    rel_path = _repo_relative(path)
    return analyze_manifest(dict(payload), path=rel_path)


def _build_contract_node() -> tuple[str, LineageNode]:
    """Build the shared lineage contract node and return its id plus node."""
    contract_node_id = _contract_node_id()
    return contract_node_id, LineageNode(
        node_id=contract_node_id,
        kind="validation_contract",
        label="Manifest Lineage Contract",
        status=LineageStatus.CONNECTED,
        payload={"mandatory_fields": list(MANDATORY_LINEAGE_FIELDS)},
    )


def _build_manifest_subgraph(
    plan: ManifestBackfillPlan,
    contract_node_id: str,
    nodes: dict[str, LineageNode],
    edges: list[LineageEdge],
) -> None:
    """Add a manifest node, its field nodes, and proxy edges to the graph."""
    rel_path = plan.path
    manifest_node_id = _manifest_node_id(rel_path)
    nodes[manifest_node_id] = LineageNode(
        node_id=manifest_node_id,
        kind="manifest",
        label=Path(rel_path).name,
        path=rel_path,
        status=LineageStatus.CONNECTED,
        payload={"validation_errors": plan.validation_errors},
    )
    edges.append(
        LineageEdge(
            source=manifest_node_id,
            target=contract_node_id,
            kind="validates_with",
            reason="manifest is checked against the shared lineage contract",
        )
    )

    for entry in plan.fields:
        field_id = _field_node_id(rel_path, entry.field_name)
        field_status = _lineage_status_from_entry(entry)
        nodes[field_id] = LineageNode(
            node_id=field_id,
            kind="lineage_field",
            label=entry.field_name,
            path=rel_path,
            status=field_status,
            payload={"status": entry.status.value},
        )
        edges.append(
            LineageEdge(
                source=manifest_node_id,
                target=field_id,
                kind="has_field",
                reason=f"manifest contains lineage field {entry.field_name}",
            )
        )

        if entry.status == FieldStatus.PRESENT:
            continue

        if entry.status == FieldStatus.INFERRED and entry.inferred_from is not None:
            _add_inferred_proxy_edge(nodes, edges, rel_path, field_id, entry)
            continue

        # Derive proxy-related edges from structured metadata.  Older plans
        # that only stored a reason string fall back to parsing the reason.
        edge_kind, proxy_labels = _proxy_edge_kind_and_labels(entry)
        if not edge_kind:
            continue

        for label in proxy_labels:
            proxy_id = _proxy_node_id(rel_path, label)
            if proxy_id not in nodes:
                nodes[proxy_id] = LineageNode(
                    node_id=proxy_id,
                    kind="proxy_source",
                    label=label,
                    path=rel_path,
                    status=field_status,
                )
            edges.append(
                LineageEdge(
                    source=field_id,
                    target=proxy_id,
                    kind=edge_kind,
                    reason=entry.reason,
                )
            )


def _candidate_text(
    candidate: Mapping[str, Any],
    key: str,
    *,
    default: str = "",
    fallback_on_blank: bool = False,
) -> str:
    """Return stripped candidate text while treating explicit None as missing."""
    value = candidate.get(key)
    if value is None:
        return default
    text = str(value).strip()
    if fallback_on_blank and not text:
        return default
    return text


def _build_artifact_trace(
    candidate: Mapping[str, Any],
    path_to_plan: dict[str, ManifestBackfillPlan],
    repo_root: Path,
    candidate_source_path: Path | None,
    nodes: dict[str, LineageNode],
    edges: list[LineageEdge],
) -> ArtifactTrace:
    """Add an artifact candidate node/edge and return its trace record."""
    artifact_id = _candidate_text(
        candidate,
        "artifact_id",
        default="unnamed_artifact",
        fallback_on_blank=True,
    )
    artifact_kind = _candidate_text(candidate, "artifact_kind", default="artifact")
    claim_boundary = _candidate_text(candidate, "claim_boundary")
    manifest_value = _candidate_text(candidate, "source_manifest_path")

    if not manifest_value:
        trace_status = LineageStatus.INCONCLUSIVE
        reason = "candidate has no source_manifest_path"
        field_statuses: dict[str, str] = {}
        source_manifest_path = ""
    else:
        resolved = _resolve_manifest_path(
            manifest_value, candidate_path=candidate_source_path, repo_root=repo_root
        )
        source_manifest_path = _repo_relative(resolved)
        matching_plan = path_to_plan.get(source_manifest_path)
        if matching_plan is None:
            trace_status = LineageStatus.INCONCLUSIVE
            reason = f"source manifest not found in graph inputs: {source_manifest_path}"
            field_statuses = {}
        else:
            trace_status, reason = _trace_lineage_status(matching_plan.fields)
            field_statuses = _field_statuses(matching_plan.fields)

    artifact_node_id = _artifact_node_id(artifact_id)
    if artifact_node_id not in nodes:
        nodes[artifact_node_id] = LineageNode(
            node_id=artifact_node_id,
            kind="artifact_candidate",
            label=artifact_id,
            path=source_manifest_path,
            status=trace_status,
            payload={
                "artifact_kind": artifact_kind,
                "claim_boundary": claim_boundary,
            },
        )
    if source_manifest_path:
        manifest_node_id = _manifest_node_id(source_manifest_path)
        if manifest_node_id in nodes:
            edges.append(
                LineageEdge(
                    source=artifact_node_id,
                    target=manifest_node_id,
                    kind="traces_to",
                    reason=f"{artifact_kind} traces to source manifest",
                )
            )

    return ArtifactTrace(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        source_manifest_path=source_manifest_path,
        claim_boundary=claim_boundary,
        lineage_status=trace_status,
        reason=reason,
        field_statuses=field_statuses,
    )


def build_manifest_lineage_graph(
    manifest_paths: Sequence[Path],
    *,
    artifact_candidates: Sequence[Mapping[str, Any]] = (),
    candidate_source_path: Path | None = None,
    generated_at_utc: str | None = None,
) -> ManifestLineageGraph:
    """Build a lineage graph from manifest paths and optional artifact candidates.

    Args:
        manifest_paths: Manifest JSON files to analyze.
        artifact_candidates: Optional artifact/table/figure candidates that trace
            back to one of the supplied manifest paths.
        candidate_source_path: Optional path to the candidate file, used to
            resolve relative manifest references.
        generated_at_utc: Optional ISO-8601 UTC timestamp for the report. When
            omitted, the current wall-clock UTC time is used.

    Returns:
        ManifestLineageGraph with nodes, edges, traces, and summary.
    """
    nodes: dict[str, LineageNode] = {}
    edges: list[LineageEdge] = []
    traces: list[ArtifactTrace] = []

    contract_node_id, contract_node = _build_contract_node()
    nodes[contract_node_id] = contract_node

    plans = [_analyze_manifest_at_path(path) for path in _dedupe_paths(manifest_paths)]
    for plan in plans:
        _build_manifest_subgraph(plan, contract_node_id, nodes, edges)

    repo_root = get_repository_root()
    path_to_plan = {plan.path: plan for plan in plans}
    for candidate in artifact_candidates:
        traces.append(
            _build_artifact_trace(
                candidate,
                path_to_plan,
                repo_root,
                candidate_source_path,
                nodes,
                edges,
            )
        )

    summary = _build_summary(nodes.values(), edges, traces)
    timestamp = (
        _parse_generated_at_utc(generated_at_utc) if generated_at_utc is not None else _now_utc()
    )
    return ManifestLineageGraph(
        schema_version=SCHEMA_VERSION,
        generated_at_utc=timestamp,
        nodes=list(nodes.values()),
        edges=edges,
        traces=traces,
        summary=summary,
    )


def _add_inferred_proxy_edge(
    nodes: dict[str, LineageNode],
    edges: list[LineageEdge],
    rel_path: str,
    field_id: str,
    entry: FieldBackfillEntry,
) -> None:
    """Add an inferred_from edge between a field and its proxy source."""
    assert entry.inferred_from is not None
    proxy_id = _proxy_node_id(rel_path, entry.inferred_from)
    if proxy_id not in nodes:
        nodes[proxy_id] = LineageNode(
            node_id=proxy_id,
            kind="proxy_source",
            label=entry.inferred_from,
            path=rel_path,
            status=LineageStatus.CONNECTED,
        )
    edges.append(
        LineageEdge(
            source=field_id,
            target=proxy_id,
            kind="inferred_from",
            reason=entry.reason or f"inferred from {entry.inferred_from}",
            payload={"inferred_value": entry.inferred_value},
        )
    )


def _proxy_edge_kind_and_labels(entry: FieldBackfillEntry) -> tuple[str, list[str]]:
    """Return the edge kind and proxy labels for a non-present field entry.

    Uses structured metadata first, and falls back to parsing the reason
    string only for legacy plans that did not store structured metadata.
    """
    if entry.status == FieldStatus.AMBIGUOUS:
        labels = list(entry.candidate_sources + entry.conflicting_sources + entry.blocked_by)
        edge_kind = "ambiguous_between"
    elif entry.status == FieldStatus.BLOCKED:
        labels = list(entry.blocked_by)
        edge_kind = "blocked_by"
    else:
        return "", []

    if not labels and entry.reason:
        labels = _parse_proxy_labels(entry.reason)
    return edge_kind, labels


def _parse_proxy_labels(reason: str) -> list[str]:
    """Extract dotted proxy labels from a backfill reason string.

    This is a best-effort parser kept only for backward compatibility with
    older serialized plans that did not store structured proxy metadata.
    Newly produced ``FieldBackfillEntry`` objects should rely on
    ``candidate_sources``, ``conflicting_sources``, and ``blocked_by`` instead.
    It looks for labels like ``metadata.generator_id`` or
    ``config.validator_version``.
    """
    labels: list[str] = []
    # Split on common separators and punctuation.
    for raw in reason.replace(":", " ").replace(",", " ").split():
        label = raw.strip("().")
        if "." in label and all(part.isidentifier() for part in label.split(".")):
            labels.append(label)
    return labels


def _build_summary(
    nodes: Sequence[LineageNode],
    edges: Sequence[LineageEdge],
    traces: Sequence[ArtifactTrace],
) -> dict[str, Any]:
    """Build aggregate counts for the graph report."""
    node_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for node in nodes:
        node_counts[node.kind] = node_counts.get(node.kind, 0) + 1
        status_counts[node.status] = status_counts.get(node.status, 0) + 1

    edge_counts: dict[str, int] = {}
    for edge in edges:
        edge_counts[edge.kind] = edge_counts.get(edge.kind, 0) + 1

    trace_status_counts: dict[str, int] = {}
    for trace in traces:
        trace_status_counts[trace.lineage_status] = (
            trace_status_counts.get(trace.lineage_status, 0) + 1
        )

    return {
        "manifest_count": node_counts.get("manifest", 0),
        "artifact_candidate_count": node_counts.get("artifact_candidate", 0),
        "lineage_field_count": node_counts.get("lineage_field", 0),
        "proxy_source_count": node_counts.get("proxy_source", 0),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "trace_count": len(traces),
        "node_status_counts": status_counts,
        "edge_kind_counts": edge_counts,
        "trace_status_counts": trace_status_counts,
    }


def format_lineage_graph_markdown(graph: ManifestLineageGraph) -> str:
    """Format a lineage graph as a Markdown adjacency report.

    Returns:
        Markdown string with a summary, adjacency list, and trace table.
    """
    lines: list[str] = []
    lines.append("# Manifest Lineage Graph Report")
    lines.append("")
    lines.append(f"- **Schema**: `{graph.schema_version}`")
    lines.append(f"- **Generated at (UTC)**: {graph.generated_at_utc}")
    lines.append("")

    summary = graph.summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Manifests: {summary.get('manifest_count', 0)}")
    lines.append(f"- Artifact candidates: {summary.get('artifact_candidate_count', 0)}")
    lines.append(f"- Lineage fields: {summary.get('lineage_field_count', 0)}")
    lines.append(f"- Proxy sources: {summary.get('proxy_source_count', 0)}")
    lines.append(f"- Total nodes: {summary.get('node_count', 0)}")
    lines.append(f"- Total edges: {summary.get('edge_count', 0)}")
    lines.append(f"- Traces: {summary.get('trace_count', 0)}")
    lines.append("")

    status_counts = summary.get("node_status_counts", {})
    if status_counts:
        lines.append("### Node status counts")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("| --- | --- |")
        for status in sorted(status_counts):
            lines.append(f"| {status} | {status_counts[status]} |")
        lines.append("")

    edge_counts = summary.get("edge_kind_counts", {})
    if edge_counts:
        lines.append("### Edge kind counts")
        lines.append("")
        lines.append("| Kind | Count |")
        lines.append("| --- | --- |")
        for kind in sorted(edge_counts):
            lines.append(f"| {kind} | {edge_counts[kind]} |")
        lines.append("")

    lines.append("## Artifact traces")
    lines.append("")
    if graph.traces:
        lines.append("| Artifact | Kind | Source manifest | Claim boundary | Status | Reason |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for trace in graph.traces:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(trace.artifact_id),
                        _md_escape(trace.artifact_kind),
                        _md_escape(trace.source_manifest_path) or "NA",
                        _md_escape(trace.claim_boundary) or "NA",
                        _md_escape(trace.lineage_status),
                        _md_escape(trace.reason),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No artifact candidates supplied.")
    lines.append("")

    lines.append("## Adjacency list")
    lines.append("")
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for edge in graph.edges:
        adjacency.setdefault(edge.source, []).append((edge.target, edge.kind))

    node_by_id = {node.node_id: node for node in graph.nodes}
    for node_id in sorted(adjacency):
        node = node_by_id.get(node_id)
        label = node.label if node else node_id
        status = f" ({node.status})" if node else ""
        lines.append(f"- **{label}**{status} ->")
        for target_id, kind in sorted(adjacency[node_id]):
            target = node_by_id.get(target_id)
            target_label = target.label if target else target_id
            lines.append(f"  - `{kind}` -> **{target_label}**")
        lines.append("")

    return "\n".join(lines)


def _md_escape(text: str) -> str:
    """Escape Markdown pipe characters in table cells."""
    return text.replace("|", r"\|").replace("\n", " ")


def write_manifest_lineage_graph_report(
    graph: ManifestLineageGraph,
    *,
    json_path: Path,
    markdown_path: Path | None = None,
) -> dict[str, Path]:
    """Write JSON graph and optional Markdown adjacency report to disk.

    Args:
        graph: The lineage graph to write.
        json_path: Destination path for the JSON graph.
        markdown_path: Optional destination path for the Markdown report.

    Returns:
        Mapping of output kind to written path.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(graph.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written: dict[str, Path] = {"json": json_path}
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            format_lineage_graph_markdown(graph),
            encoding="utf-8",
        )
        written["markdown"] = markdown_path
    return written


__all__ = [
    "SCHEMA_VERSION",
    "ArtifactTrace",
    "LineageEdge",
    "LineageNode",
    "LineageStatus",
    "ManifestLineageGraph",
    "build_manifest_lineage_graph",
    "format_lineage_graph_markdown",
    "write_manifest_lineage_graph_report",
]
