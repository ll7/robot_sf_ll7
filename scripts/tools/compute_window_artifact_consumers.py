#!/usr/bin/env python3
"""Build a conservative, public-safe consumer graph for compute-window artifacts.

The input is a compact JSON manifest with ``artifacts`` and optional consumer
collections (``configs``, ``registries``, ``manifests``, ``scripts``,
``reports``, ``releases``, ``papers``, and ``active_tasks``).  References are
explicit semantic identities, not filename guesses.  A private operational
system may provide only a sanitized ``private_projection`` containing the same
identity/reference shape.  The tool never moves, deletes, or reads private
content.  Missing static evidence is ``consumer_unknown`` rather than proof of
orphanhood.

CLI: ``compute_window_artifact_consumers.py --input PATH [--root PATH]
[--check] [--format json|dot|markdown]``. Exit 2 means the graph is
fail-closed because an integrity conflict was found.
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
EDGE_TYPES = (
    "loads",
    "validates",
    "reports_from",
    "releases",
    "cites",
    "replays",
    "restores",
    "requires_for_resume",
    "supersedes",
)
CLASSES = (
    "active_required",
    "historical_required",
    "replacement_verified",
    "regenerable_verified",
    "consumer_unknown",
    "orphan_candidate",
    "unresolved_conflict",
)
CONSUMER_GROUPS = (
    "consumers",
    "configs",
    "registries",
    "manifests",
    "scripts",
    "reports",
    "releases",
    "papers",
    "active_tasks",
)
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PRIVATE_RE = re.compile(
    r"(?i)(?:://|api[_-]?key|secret|password|token|credential|bearer|^[/~\\]|@[A-Za-z])"
)


def _key(value: Any) -> str:
    return str(value).strip()


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and ID_RE.fullmatch(value) is not None


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _private(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(k).lower() in {"locator", "url", "host", "command", "environment"} or _private(v)
            for k, v in value.items()
        )
    if isinstance(value, list):
        return any(_private(v) for v in value)
    return isinstance(value, str) and PRIVATE_RE.search(value) is not None


def _finding(code: str, source: str, target: str | None, detail: str) -> dict[str, Any]:
    return {"code": code, "source": source, "target": target, "detail": detail}


def _refs(item: Mapping[str, Any]) -> list[tuple[str, str, str | None, str | None]]:
    """Return (logical id, edge type, digest, path) references from one record."""
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
    result = []
    for ref in raw if isinstance(raw, list) else []:
        if isinstance(ref, str):
            result.append((ref, "loads", None, None))
            continue
        if not isinstance(ref, Mapping):
            continue
        logical_id = ref.get("logical_id", ref.get("artifact_id", ref.get("id")))
        edge = ref.get("edge", ref.get("relation", "loads"))
        if _valid_id(logical_id) and edge in EDGE_TYPES:
            digest = ref.get("sha256", ref.get("content_identity"))
            if isinstance(digest, Mapping):
                digest = digest.get("sha256")
            result.append(
                (
                    logical_id,
                    edge,
                    digest if isinstance(digest, str) else None,
                    ref.get("path") if isinstance(ref.get("path"), str) else None,
                )
            )
    return result


def _record_id(item: Mapping[str, Any], fallback: str) -> str:
    return _key(item.get("consumer_id", item.get("id", item.get("name", fallback))))


def _tracked_refs(root: Path, artifacts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive only public, tracked-file references; never emit file contents."""
    try:
        names = (
            subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
            .stdout.decode()
            .split("\0")
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return []
    found = []
    for name in sorted(n for n in names if n and not n.startswith("output/")):
        path = root / name
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        hits = sorted(ident for ident in artifacts if ident in text)
        if hits:
            found.append(
                {
                    "consumer_id": f"tracked:{name}",
                    "kind": "tracked_file",
                    "state": "historical",
                    "refs": hits,
                }
            )
    return found


def build_graph(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Build the graph from a sanitized manifest and optional repository root."""
    findings: list[dict[str, Any]] = []
    if payload.get("schema") not in {SCHEMA, "compute-window-artifact-consumers.v1"}:
        findings.append(_finding("invalid_schema", "/schema", None, "unexpected schema"))
    artifacts: dict[str, dict[str, Any]] = {}
    identities: dict[str, set[str]] = {}
    paths: dict[str, set[str]] = {}
    raw_artifacts = payload.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        findings.append(_finding("invalid_artifacts", "/artifacts", None, "list required"))
        raw_artifacts = []
    for index, raw in enumerate(raw_artifacts):
        if not isinstance(raw, Mapping) or not _valid_id(raw.get("logical_id", raw.get("id"))):
            findings.append(
                _finding("invalid_artifact", f"/artifacts[{index}]", None, "logical_id required")
            )
            continue
        ident = _key(raw.get("logical_id", raw.get("id")))
        digest = raw.get("sha256", raw.get("content_identity"))
        if isinstance(digest, Mapping):
            digest = digest.get("sha256")
        if digest is not None and not _valid_sha(digest):
            findings.append(
                _finding("invalid_identity", ident, None, "content identity must be sha256")
            )
        identities.setdefault(ident, set()).add(_key(digest) if digest else "unknown")
        path = raw.get("path")
        if path:
            paths.setdefault(ident, set()).add(_key(path))
        artifacts.setdefault(
            ident,
            {
                "logical_id": ident,
                "kind": raw.get("kind", "artifact"),
                "sha256": digest,
                "path": path,
                "regenerable": bool(raw.get("regenerable")),
                "orphan_candidate": bool(raw.get("orphan_candidate")),
                "replacement_for": raw.get("replacement_for"),
            },
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
    for group in CONSUMER_GROUPS:
        value = payload.get(group, [])
        if isinstance(value, Mapping):
            value = list(value.values())
        if isinstance(value, list):
            records.extend(
                (_record_id(item, f"{group}:{i}"), item)
                for i, item in enumerate(value)
                if isinstance(item, Mapping)
            )
    if root is not None:
        records.extend(
            (_record_id(item, "tracked"), item) for item in _tracked_refs(root, artifacts)
        )
    private_projection = payload.get("private_projection", [])
    if isinstance(private_projection, list):
        records.extend(
            (_record_id(item, f"private:{i}"), item)
            for i, item in enumerate(private_projection)
            if isinstance(item, Mapping)
        )
    nodes = [
        {"id": ident, "kind": item["kind"], "sha256": item["sha256"], "path": item["path"]}
        for ident, item in sorted(artifacts.items())
    ]
    edges: list[dict[str, Any]] = []
    consumers: dict[str, dict[str, Any]] = {}
    known: dict[str, list[dict[str, Any]]] = {ident: [] for ident in artifacts}
    for consumer_id, item in records:
        state = _key(item.get("state", item.get("status", "unknown")))
        kind = _key(item.get("kind", "consumer"))
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
            and _key(item.get("issue_state", item.get("issue_status", ""))).lower() == "closed"
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
        for logical_id, edge, digest, path in _refs(item):
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
            consumer["refs"].append(logical_id)
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
    supersedes = {edge["target"]: edge["source"] for edge in edges if edge["type"] == "supersedes"}
    for start in sorted(artifacts):
        seen: set[str] = set()
        current = start
        while current in supersedes:
            if current in seen:
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
        conflict = any(
            f["target"] == ident or f["source"] == ident
            for f in findings
            if f["code"] in {"conflicting_identity", "stale_path"}
        )
        if conflict:
            classification = "unresolved_conflict"
        elif active:
            classification = "active_required"
        elif artifact["regenerable"] and any(r["edge"] == "requires_for_resume" for r in refs):
            classification = "regenerable_verified"
        elif historical:
            classification = "historical_required"
        elif artifact.get("replacement_for") and artifact["replacement_for"] in artifacts:
            classification = "replacement_verified"
        elif any(edge["type"] == "supersedes" and edge["target"] == ident for edge in edges):
            classification = "replacement_verified"
        elif artifact["orphan_candidate"]:
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
        source = edge["source"] if edge["type"] == "supersedes" else f"consumer:{edge['source']}"
        target = f"artifact:{edge['target']}"
        source = source if source.startswith(("consumer:", "artifact:")) else f"artifact:{source}"
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
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
