#!/usr/bin/env python3
"""Build a conservative, public-safe consumer graph for compute-window artifacts.

The input is a compact JSON manifest with ``artifacts`` and optional consumer
collections (``configs``, ``registries``, ``manifests``, ``scripts``,
``reports``, ``releases``, ``papers``, and ``active_tasks``).  References are
explicit manifest identities or whole-line ``robot_sf-artifact-ref:
<logical-id>`` markers (optionally ``#``-prefixed); ``--root`` parses only
those markers from tracked text.  Private projections are sanitized and
missing static evidence is ``consumer_unknown``, never proof of orphanhood.

CLI: ``compute_window_artifact_consumers.py --input PATH [--root PATH]
[--check] [--format json|dot|markdown]``. Rendering always exits 0 after
producing a report; ``--check`` exits 2 when the report is fail-closed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "robot_sf.compute_window_artifact_consumers.v1"
VERIFICATION_CONTRACT = (
    "replacement_verified requires replacement_verified=true and a present replacement_for target; "
    "regenerable_verified requires regeneration_verified=true, regenerable=true, and a requires_for_resume edge. "
    "Flags and relations never imply verification."
)
EDGE_TYPES = tuple(
    "loads validates reports_from releases cites replays restores "
    "requires_for_resume supersedes".split()
)
CONSUMER_GROUPS = tuple(
    "consumers configs registries manifests scripts reports releases papers active_tasks".split()
)
PUBLIC_REF_RE = re.compile(
    r"(?m)^[ \t]*(?:#[ \t]*)?robot_sf-artifact-ref:[ \t]*([A-Za-z0-9][A-Za-z0-9._:/-]{0,127})[ \t]*$"
)
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
KNOWN_CONSUMER_STATES = frozenset("active running open queued historical published closed".split())
EVIDENCE_FAILURE_CODES = frozenset("invalid_records invalid_private_projection invalid_refs invalid_ref invalid_reference_id invalid_edge_type invalid_content_identity invalid_path".split())  # fmt: skip
PRIVATE_RE = re.compile(
    r"(?i)(?:://|api[_-]?key|secret|password|token|credential|bearer|^[/~\\]|@[A-Za-z]|(?:[a-z0-9][a-z0-9-]{0,61}\.){2,}[a-z]{2,24})"
)
PRIVATE_KEYS = frozenset("command environment host hostname locator password secret token url".split())  # fmt: skip


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and ID_RE.fullmatch(value) is not None


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _private(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(k).lower() in PRIVATE_KEYS or _private(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_private(v) for v in value)
    return isinstance(value, str) and PRIVATE_RE.search(value) is not None


def _finding(code: str, source: str, target: str | None, detail: str) -> dict[str, Any]:
    return {"code": code, "source": source, "target": target, "detail": detail}


def _safe_id(value: Any, fallback: str) -> str:
    return value if _valid_id(value) and not _private(value) else fallback


def _safe_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or _private(value):
        return False
    path = value.replace("\\", "/")
    return not (
        path != value or ":" in value or path.startswith(("/", "~")) or ".." in path.split("/")
    )


def _refs(  # noqa: C901
    item: Mapping[str, Any], source: str, findings: list[dict[str, Any]]
) -> list[tuple[str, str, str | None, str | None]]:
    raw = item.get("refs", item.get("references", []))
    if isinstance(raw, Mapping):
        raw = [
            dict(value, logical_id=key)
            if isinstance(value, Mapping)
            else {"logical_id": key, "edge": value}
            for key, value in raw.items()
        ]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        findings.append(
            _finding("invalid_refs", source, None, "references must be a list, mapping, or string")
        )
        return []
    result = []
    for index, ref in enumerate(raw):
        if isinstance(ref, str):
            if _valid_id(ref) and not _private(ref):
                result.append((ref, "loads", None, None))
            else:
                findings.append(
                    _finding(
                        "invalid_reference_id",
                        source,
                        None,
                        f"reference {index} has an invalid logical ID",
                    )
                )
            continue
        if not isinstance(ref, Mapping):
            findings.append(
                _finding(
                    "invalid_ref", source, None, f"reference {index} must be a mapping or string"
                )
            )
            continue
        logical_id = ref.get("logical_id", ref.get("artifact_id", ref.get("id")))
        edge = ref.get("edge", ref.get("relation", "loads"))
        if not _valid_id(logical_id) or _private(logical_id):
            findings.append(
                _finding(
                    "invalid_reference_id",
                    source,
                    None,
                    f"reference {index} has an invalid logical ID",
                )
            )
            continue
        if not isinstance(edge, str) or edge not in EDGE_TYPES:
            findings.append(
                _finding(
                    "invalid_edge_type",
                    source,
                    logical_id if _valid_id(logical_id) else None,
                    f"reference {index} has an invalid edge type",
                )
            )
            continue
        digest = ref.get("sha256", ref.get("content_identity"))
        if isinstance(digest, Mapping):
            digest = digest.get("sha256")
        if digest is not None and not _valid_sha(digest):
            findings.append(
                _finding(
                    "invalid_content_identity",
                    source,
                    logical_id,
                    f"reference {index} has an invalid content identity",
                )
            )
            digest = None
        path = ref.get("path")
        if path is not None and not _safe_path(path):
            findings.append(
                _finding(
                    "invalid_path", source, logical_id, f"reference {index} has an unsafe path"
                )
            )
            path = None
        result.append((logical_id, edge, digest, path))
    return result


def _record_entries(
    value: Any, group: str, findings: list[dict[str, Any]]
) -> list[Mapping[str, Any]]:
    private = group == "private_projection"
    invalid_code = "invalid_private_projection" if private else "invalid_records"
    details = (
        ("private_projection must be a list", "private projection entry must be a mapping")
        if private
        else ("consumer collection must be a list or mapping", "consumer entry must be a mapping")
    )
    value = list(value.values()) if isinstance(value, Mapping) and not private else value
    if not isinstance(value, list):
        findings.append(_finding(invalid_code, f"/{group}", None, details[0]))
        return []
    if any(not isinstance(item, Mapping) for item in value):
        findings.append(_finding(invalid_code, f"/{group}", None, details[1]))
    return [item for item in value if isinstance(item, Mapping)]


def _tracked_refs(root: Path, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    except UnicodeError:
        findings.append(_finding("tracked_decode_failed", "/tracked", None, "tracked scan failed"))
        return []
    except (OSError, subprocess.SubprocessError):
        findings.append(_finding("tracked_scan_failed", "/tracked", None, "tracked scan failed"))
        return []
    found = []
    for name in sorted(n for n in names if n and not n.startswith("output/")):
        try:
            text = (root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            code = (
                "tracked_decode_failed" if isinstance(exc, UnicodeError) else "tracked_read_failed"
            )
            findings.append(_finding(code, "/tracked", None, "tracked read failed"))
            continue
        hits = sorted(set(PUBLIC_REF_RE.findall(text)))
        hits = [hit for hit in hits if not _private(hit)]
        if hits:
            found.append(
                {
                    "id": f"tracked:{name}",
                    "kind": "tracked_file",
                    "state": "historical",
                    "refs": hits,
                }
            )
    return found


def build_graph(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Build the graph from explicit manifest references; ``root`` is legacy-only."""
    findings: list[dict[str, Any]] = []
    if payload.get("schema") not in {SCHEMA, "compute-window-artifact-consumers.v1"}:
        findings.append(_finding("invalid_schema", "/schema", None, "unexpected schema"))
    artifacts: dict[str, dict[str, Any]] = {}
    invalid_artifact_ids: set[str] = set()
    identities: dict[str, set[str]] = {}
    paths: dict[str, set[str]] = {}
    raw_artifacts = payload.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        findings.append(_finding("invalid_artifacts", "/artifacts", None, "list required"))
        raw_artifacts = []
    for index, raw in enumerate(raw_artifacts):
        if not isinstance(raw, Mapping):
            findings.append(
                _finding(
                    "invalid_artifact", f"/artifacts[{index}]", None, "artifact must be a mapping"
                )
            )
            continue
        raw_ident = raw.get("logical_id", raw.get("id"))
        ident = _safe_id(raw_ident, f"redacted-artifact-{index}")
        if ident != raw_ident:
            invalid_artifact_ids.add(ident)
            findings.append(
                _finding(
                    "invalid_artifact_id",
                    f"/artifacts[{index}]",
                    None,
                    "logical ID is invalid or private",
                )
            )
        digest = raw.get("sha256", raw.get("content_identity"))
        if isinstance(digest, Mapping):
            digest = digest.get("sha256")
        if digest is not None and not _valid_sha(digest):
            findings.append(
                _finding("invalid_identity", ident, None, "content identity must be sha256")
            )
            digest = None
        identities.setdefault(ident, set()).add(str(digest).strip() if digest else "unknown")
        path = raw.get("path")
        if path is not None and not _safe_path(path):
            findings.append(
                _finding("invalid_path", ident, None, "artifact path is unsafe or private")
            )
            path = None
        if path:
            paths.setdefault(ident, set()).add(str(path).strip())
        for field in PRIVATE_KEYS:
            if field in raw:
                findings.append(
                    _finding(
                        "redacted_private_field", ident, None, f"artifact field {field} was omitted"
                    )
                )
        kind = raw.get("kind", "artifact")
        if not isinstance(kind, str) or _private(kind):
            findings.append(
                _finding("invalid_artifact_metadata", ident, None, "artifact kind was omitted")
            )
            kind = "artifact"
        replacement = raw.get("replacement_for")
        if replacement is not None and (not _valid_id(replacement) or _private(replacement)):
            findings.append(
                _finding(
                    "invalid_replacement_for", ident, None, "replacement_for must be a logical ID"
                )
            )
            replacement = None
        booleans = {}
        for field in (
            "regenerable",
            "regeneration_verified",
            "orphan_candidate",
            "replacement_verified",
        ):
            value = raw.get(field, False)
            if not isinstance(value, bool):
                findings.append(
                    _finding("invalid_boolean", ident, None, f"{field} must be boolean")
                )
                value = False
            booleans[field] = value
        candidate = {
            "logical_id": ident,
            "kind": kind,
            "sha256": digest,
            "path": path,
            **booleans,
            "replacement_for": replacement,
        }
        if ident in artifacts:
            findings.append(
                _finding("duplicate_logical_id", ident, None, "artifact declaration is duplicated")
            )
        artifacts[ident] = min(
            (artifacts.get(ident, candidate), candidate),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    for ident, values in identities.items():
        if len(values - {"unknown"}) > 1:
            findings.append(
                _finding(
                    "conflicting_identity",
                    ident,
                    None,
                    "logical artifact has multiple content identities",
                )
            )
    records: list[tuple[str, Mapping[str, Any]]] = []
    consumer_identity_failure = False
    for group in (*CONSUMER_GROUPS, "private_projection"):
        if group not in payload:
            continue
        for item in _record_entries(payload[group], group, findings):
            raw_consumer_id = item.get("consumer_id", item.get("id", item.get("name")))
            if not _valid_id(raw_consumer_id) or _private(raw_consumer_id):
                code = "missing_consumer_id" if raw_consumer_id is None else "invalid_consumer_id"
                findings.append(
                    _finding(code, f"/{group}", None, "consumer ID is required and public")
                )
                consumer_identity_failure = True
                continue
            records.append((raw_consumer_id, item))
    if root is not None:
        tracked_records = _tracked_refs(root, findings)
        records.extend(
            (
                _safe_id(
                    item.get("consumer_id", item.get("id", item.get("name"))),
                    "redacted-consumer-tracked",
                ),
                item,
            )
            for item in tracked_records
        )
    nodes = [
        {"id": ident, "kind": item["kind"], "sha256": item["sha256"], "path": item["path"]}
        for ident, item in sorted(artifacts.items())
    ]
    edges: list[dict[str, Any]] = []
    consumers: dict[str, dict[str, Any]] = {}
    known: dict[str, list[dict[str, Any]]] = {ident: [] for ident in artifacts}
    for consumer_id, item in sorted(
        records, key=lambda row: (row[0], json.dumps(row[1], sort_keys=True, default=str))
    ):
        raw_state = item.get("state", item.get("status", "unknown"))
        raw_kind = item.get("kind", "consumer")
        state = str(raw_state).strip()
        kind = str(raw_kind).strip()
        if not isinstance(raw_kind, str) or not isinstance(raw_state, str):
            findings.append(
                _finding(
                    "invalid_record_metadata",
                    consumer_id,
                    None,
                    "consumer kind and state must be strings",
                )
            )
            kind, state = "consumer", "unknown"
        elif _private(raw_kind) or _private(raw_state):
            findings.append(
                _finding(
                    "redacted_record_metadata",
                    consumer_id,
                    None,
                    "private consumer metadata was omitted",
                )
            )
            kind, state = "consumer", "unknown"
        elif state not in KNOWN_CONSUMER_STATES:
            findings.append(_finding("unknown_consumer_state", consumer_id, None, "unknown"))
        if _private(item) and kind != "tracked_file":
            findings.append(
                _finding(
                    "unsanitized_private_reference",
                    consumer_id,
                    None,
                    "private locator or topology was supplied",
                )
            )
            continue
        if (
            state in {"active", "running", "queued"}
            and item.get("runtime") is True
            and str(item.get("issue_state", item.get("issue_status", ""))).strip().lower()
            == "closed"
        ):
            findings.append(
                _finding(
                    "closed_issue_active_runtime_consumer",
                    consumer_id,
                    None,
                    "closed issue still has an active runtime consumer",
                )
            )
        consumer = consumers.setdefault(
            consumer_id, {"id": consumer_id, "kind": kind, "state": state, "refs": []}
        )
        for logical_id, edge, digest, path in _refs(item, consumer_id, findings):
            target = artifacts.get(logical_id)
            if target is None:
                unknown_code = {
                    "paper": "unknown_dissertation_reference",
                    "dissertation": "unknown_dissertation_reference",
                    "release": "unknown_release_reference",
                }.get(kind, "dangling_logical_id")
                findings.append(
                    _finding(
                        unknown_code,
                        consumer_id,
                        logical_id,
                        "referenced artifact is absent",
                    )
                )
                continue
            if digest and target.get("sha256") and digest != target["sha256"]:
                findings.append(
                    _finding(
                        "conflicting_identity",
                        consumer_id,
                        logical_id,
                        "reference identity differs from artifact",
                    )
                )
            if path and target.get("path") and path != target["path"]:
                findings.append(
                    _finding(
                        "stale_path",
                        consumer_id,
                        logical_id,
                        "reference path differs from manifest path",
                    )
                )
            edge_row = {"source": consumer_id, "target": logical_id, "type": edge}
            edges.append(edge_row)
            consumer["refs"] = sorted((*consumer["refs"], logical_id))
            known[logical_id].append(
                {"consumer": consumer_id, "kind": kind, "state": state, "edge": edge}
            )
    for ident, artifact in artifacts.items():
        replacement = artifact.get("replacement_for")
        if replacement:
            if replacement not in artifacts:
                findings.append(
                    _finding(
                        "dangling_logical_id", ident, replacement, "replacement target is absent"
                    )
                )
            else:
                edges.append({"source": ident, "target": replacement, "type": "supersedes"})
        for path in paths.get(ident, set()):
            if root is not None and not (root / path).is_file():
                findings.append(_finding("stale_path", ident, path, "manifest path does not exist"))
    superseders: dict[str, set[str]] = {}
    for edge in edges:
        if edge["type"] == "supersedes":
            superseders.setdefault(edge["target"], set()).add(edge["source"])
    ambiguous_targets = {target for target, sources in superseders.items() if len(sources) > 1}
    for target in sorted(ambiguous_targets):
        findings.append(
            _finding(
                "ambiguous_superseders",
                target,
                None,
                "multiple superseding sources exist",
            )
        )
    supersedes = {
        source: target
        for target, sources in superseders.items()
        for source in sources
        if source in artifacts
    }
    cycle_nodes: set[str] = set()
    for start in sorted(artifacts):
        seen: set[str] = set()
        current = start
        while current in supersedes:
            if current in seen:
                cycle_nodes.update(seen)
                findings.append(
                    _finding(
                        "supersession_cycle", start, current, "supersession graph contains a cycle"
                    )
                )
                break
            seen.add(current)
            current = supersedes[current]
    classifications = []
    for ident, artifact in sorted(artifacts.items()):
        refs = known[ident]
        active = any(r["state"] in {"active", "running", "open", "queued"} for r in refs)
        historical = any(
            r["state"] in {"historical", "published", "closed"}
            or r["kind"] in {"paper", "dissertation", "release", "report", "tracked_file"}
            for r in refs
        )
        unknown_state = any(r["state"] not in KNOWN_CONSUMER_STATES for r in refs)
        conflict = any(
            f["target"] == ident or f["source"] == ident or f["code"].startswith("tracked_")
            for f in findings
        )
        conflict = (
            conflict
            or ident in cycle_nodes
            or ident in ambiguous_targets
            or ident in invalid_artifact_ids
            or any(
                f["code"] in EVIDENCE_FAILURE_CODES
                and not (
                    f["code"] == "invalid_path" and f["target"] is None and f["source"] in artifacts
                )
                and (f["target"] is None or f["target"] == ident)
                for f in findings
            )
        )
        if conflict:
            classification = "unresolved_conflict"
        elif active:
            classification = "active_required"
        elif unknown_state:
            classification = "consumer_unknown"
        elif (
            artifact["regeneration_verified"]
            and artifact["regenerable"]
            and any(r["edge"] == "requires_for_resume" for r in refs)
        ):
            classification = "regenerable_verified"
        elif historical:
            classification = "historical_required"
        elif (
            artifact["replacement_verified"]
            and artifact.get("replacement_for")
            and artifact["replacement_for"] in artifacts
        ):
            classification = "replacement_verified"
        elif artifact["orphan_candidate"] and not unknown_state and not consumer_identity_failure:
            classification = "orphan_candidate"
        else:
            classification = "consumer_unknown"
        classifications.append(
            {"logical_id": ident, "classification": classification, "consumer_count": len(refs)}
        )
    findings.sort(key=lambda x: (x["code"], x["source"], x["target"] or "", x["detail"]))
    edges.sort(key=lambda x: (x["source"], x["target"], x["type"]))
    return {
        "schema": SCHEMA,
        "claim_boundary": "Conservative static consumer projection for retention review; absence of a tracked reference never proves absence of a consumer, and no semantic equivalence is inferred.",
        "verification_contract": VERIFICATION_CONTRACT,
        "ok": not findings,
        "status": "ok" if not findings else "failed",
        "summary": {
            "artifact_count": len(nodes),
            "consumer_count": len(consumers),
            "edge_count": len(edges),
            "finding_count": len(findings),
        },
        "artifacts": nodes,
        "classifications": classifications,
        "consumers": [consumers[k] for k in sorted(consumers)],
        "edges": edges,
        "findings": findings,
    }


def render_json(report: Mapping[str, Any]) -> str:
    """Return deterministic JSON for a graph report."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_dot(report: Mapping[str, Any]) -> str:
    """Return a deterministic Graphviz projection of a graph report."""
    lines = ["digraph consumer_graph {", '  rankdir="LR";']
    for artifact in report["artifacts"]:
        lines.append(f'  "artifact:{artifact["id"]}" [shape=box];')
    for consumer in report["consumers"]:
        lines.append(f'  "consumer:{consumer["id"]}" [shape=ellipse];')
    for edge in report["edges"]:
        namespace = "artifact" if edge["type"] == "supersedes" else "consumer"
        source = f"{namespace}:{edge['source']}"
        target = f"artifact:{edge['target']}"
        lines.append(f'  "{source}" -> "{target}" [label="{edge["type"]}"];')
    return "\n".join(lines) + "\n} \n"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Return a concise Markdown projection of a graph report."""
    lines = [
        "# Compute-window artifact consumer graph",
        "",
        f"- Status: `{report['status']}`",
        f"- Artifacts: {report['summary']['artifact_count']} | Edges: {report['summary']['edge_count']} | Findings: {report['summary']['finding_count']}",
        "",
        report["claim_boundary"],
        "",
        "## Classifications",
        "",
        "| Logical artifact | Classification | Consumers |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| `{row['logical_id']}` | `{row['classification']}` | {row['consumer_count']} |"
        for row in report["classifications"]
    )
    findings = [
        f"- `{item['code']}`: `{item['source']}` → `{item['target'] or '-'}`"
        for item in report["findings"]
    ] or ["- none"]
    lines.extend(["", "## Findings", "", *findings])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the consumer graph CLI and return a shell-friendly status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--format", choices=("json", "dot", "markdown"), default="json")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {"schema": SCHEMA, "ok": False, "status": "failed", "error": type(exc).__name__},
                sort_keys=True,
            )
        )
        return 2
    if not isinstance(payload, Mapping):
        return 2
    report = build_graph(payload, root=args.root)
    rendered = {"json": render_json, "dot": render_dot, "markdown": render_markdown}[args.format](
        report
    )
    sys.stdout.write(rendered)
    return 0 if report["ok"] or not args.check else 2


if __name__ == "__main__":
    raise SystemExit(main())
