#!/usr/bin/env python3
"""Sanitized job-to-issue-to-artifact lineage index (issue #8897).

Joins explicit compact records (schema ``sanitized_lineage_input.v1``) into one deterministic
``robot_sf.lineage_index.v1`` JSON plus Markdown projection keyed by stable semantic identity
(``kind:id``), never filename proximity or timestamps. Each row is rooted at one job attempt, or
at a record unreachable from any job, so retries and resumed shards stay separate rows linked by
``predecessor_job_id``, ``attempt_index``, and ``relation``. The compact sanitized-input contract,
its withheld ``private_projection``, and the missing-link classes and finding codes are documented
in ``docs/context/artifact_evidence_vocabulary.md``.

The tool reads only the given files, never private infrastructure, and mutates no state; output
never contains a private path, hostname, credential, or signed URL, and repeated generation is
byte-stable. CLI: ``lineage_index.py --input <path> [--input ...] [--check] [--format
json|markdown]``; ``lineage_index.py query --input ...`` selects rows containing
every supplied ``--issue``/``--job``/``--campaign``/``--artifact-digest``/``--commit``/``--config``
identity. Exit codes: 0 ok/match; 2 fail-closed or no query match.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "sanitized_lineage_input.v1"
INDEX_SCHEMA = "robot_sf.lineage_index.v1"
CLAIM_BOUNDARY = (
    "Sanitized cross-record lineage projection only: rows join declared semantic identities from "
    "compact public records and an optional sanitized private projection. A missing link is a "
    "custody observation, not a scientific, benchmark, or evidence verdict."
)
KINDS = "issue pull_request commit campaign config manifest job checkpoint model environment artifact analysis claim".split()
REF_KIND = {f"{kind}_ids": kind for kind in KINDS}
FIELD_BY_KIND = {kind: f"{kind}_ids" for kind in KINDS}
CATEGORIES = (*KINDS, *"submission_receipt raw_artifact compact_artifact artifact_locator".split())
STATES = {
    "relation": ("initial", "retry", "resume"),
    "submission_receipt": ("recorded", "not_recorded", "not_applicable"),
    "claim_state": ("diagnostic", "bounded", "not_applicable"),
    "artifact_kind": ("raw", "compact"),
    "locator_class": "public_release cloud_durable personal_durable tracked_path private_overlay not_recorded not_applicable".split(),
}
RECORD_KEYS = frozenset(
    "kind id refs digest owner attempt_index relation submission_receipt predecessor_job_id "
    "artifact_kind locator_class claim_state not_applicable".split()
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_KEY_RE = re.compile(r"^[a-z_]+:[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_FIELDS = frozenset(
    "account command command_line cmdline env environment host hostname node node_list nodelist "
    "nodes partition password private_path qos scheduler_account secret secrets slurm_account "
    "token url user user_name username".split()
)
_PRIVATE_PATTERNS = (
    (r"://", "URL or private path"),
    (r"(?i)[?&](?:sig|signature|token|expires|x-amz-|x-goog-)", "signed URL"),
    (
        r"(?i)(?:api[_-]?key|secret|password|passwd|token|credential|bearer)",
        "credential-like content",
    ),
    (r"(?i)(?:[a-z0-9][a-z0-9-]{0,61}\.){2,}[a-z]{2,24}", "hostname-like content"),
)


def _f(code: str, source: str, target: str | None, detail: str) -> dict[str, Any]:
    return {"code": code, "source": source, "target": target, "detail": detail}


def _n(code: str, source: str, detail: str) -> dict[str, Any]:
    return _f(code, source, None, detail)


def _key(kind: str, ident: str) -> str:
    return f"{kind}:{ident}"


def _ok_id(value: Any) -> bool:
    return isinstance(value, str) and _ID_RE.fullmatch(value) is not None and ".." not in value


def _ok_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _ok_na(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(set(value))
        and all(item in CATEGORIES for item in value)
    )


def _in(options: Sequence[str]):
    return lambda value: value in options


_FIELD_CHECKS = {
    "digest": _ok_sha,
    "owner": lambda value: isinstance(value, str) and bool(value.strip()),
    "attempt_index": lambda value: (
        isinstance(value, int) and not isinstance(value, bool) and value >= 1
    ),
    "predecessor_job_id": _ok_id,
    "not_applicable": _ok_na,
    **{field: _in(options) for field, options in STATES.items()},
}


def _scan_private(value: Any, location: str, findings: list[dict[str, Any]]) -> bool:
    """Append sanitized findings for private-looking values; never echo a value."""
    if isinstance(value, Mapping):
        children = [(f"{location}/{key}", key, item) for key, item in value.items()]
    elif isinstance(value, list):
        children = [(f"{location}[{index}]", None, item) for index, item in enumerate(value)]
    elif isinstance(value, str):
        details = [
            detail for pattern, detail in _PRIVATE_PATTERNS if re.search(pattern, value) is not None
        ]
        if (
            value.startswith(("/", "~", "\\"))
            or "@" in value
            or ".." in value.split("/")
            or any(ord(char) < 32 for char in value)
        ):
            details.append("private identity or path")
        findings.extend(_n("forbidden_value", location, detail) for detail in details)
        return bool(details)
    else:
        children = []
    leaked = False
    for child, key, item in children:
        if key is not None and str(key).lower() in _FORBIDDEN_FIELDS:
            findings.append(_n("forbidden_value", child, "private topology field"))
            leaked = True
        leaked |= _scan_private(item, child, findings)
    return leaked


def _record(  # noqa: C901
    raw: Any, location: str, findings: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Validate one sanitized record; any shape or value error rejects the record."""
    if not isinstance(raw, Mapping):
        findings.append(_n("invalid_record", location, "record must be a mapping"))
        return None
    before = len(findings)
    for key in sorted(set(raw) - RECORD_KEYS):
        findings.append(_n("invalid_record", f"{location}/{key}", "unknown record key"))
    kind, ident, refs = raw.get("kind"), raw.get("id"), raw.get("refs", {})
    if kind not in KINDS:
        findings.append(_n("invalid_record", f"{location}/kind", "unknown kind"))
    if not _ok_id(ident):
        findings.append(_n("invalid_record", f"{location}/id", "invalid id"))
    if not isinstance(refs, Mapping):
        findings.append(_n("invalid_record", f"{location}/refs", "mapping required"))
        refs = {}
    checked: dict[str, tuple[str, ...]] = {}
    for field in sorted(refs):
        targets = refs[field]
        if (
            field in REF_KIND
            and isinstance(targets, list)
            and all(map(_ok_id, targets))
            and len(targets) == len(set(targets))
        ):
            checked[field] = tuple(targets)
        else:
            findings.append(_n("invalid_record", f"{location}/refs/{field}", "invalid ref"))
    scalars: dict[str, Any] = {}
    for field in sorted(set(raw) - {"kind", "id", "refs"}):
        check = _FIELD_CHECKS.get(field)
        if check is None:
            continue
        if check(raw[field]):
            scalars[field] = raw[field]
        else:
            findings.append(_n("invalid_record", f"{location}/{field}", "invalid value"))
    if len(findings) != before:
        return None
    return {"kind": kind, "id": ident, "refs": checked, **scalars}


def _document(
    payload: Any, location: str, findings: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate one document; return its records and withheld projection entries."""
    if not isinstance(payload, Mapping) or payload.get("schema") != INPUT_SCHEMA:
        findings.append(_n("invalid_schema", f"{location}/schema", f"must be {INPUT_SCHEMA}"))
        return [], []
    for key in sorted(set(payload) - {"schema", "records", "private_projection"}):
        findings.append(_n("unknown_field", f"{location}/{key}", "not in schema"))
    records = payload.get("records")
    if not isinstance(records, list):
        findings.append(_n("invalid_input", f"{location}/records", "list required"))
        records = []
    nodes = [
        node
        for index, record in enumerate(records)
        if (node := _record(record, f"{location}/records[{index}]", findings))
    ]
    raw_projection = payload.get("private_projection", [])
    projection: list[dict[str, Any]] = []
    if not isinstance(raw_projection, list):
        findings.append(_n("invalid_input", f"{location}/private_projection", "list required"))
    else:
        for index, entry in enumerate(raw_projection):
            target = entry.get("target") if isinstance(entry, Mapping) else None
            if (
                isinstance(entry, Mapping)
                and set(entry) == {"target", "digest", "withheld"}
                and isinstance(target, str)
                and _KEY_RE.fullmatch(target) is not None
                and target.split(":", 1)[0] in KINDS
                and _ok_sha(entry.get("digest"))
                and entry.get("withheld") is True
            ):
                projection.append(entry)
            else:
                findings.append(
                    _f(
                        "invalid_record",
                        f"{location}/private_projection[{index}]",
                        None,
                        "withheld locator required",
                    )
                )
    return nodes, projection


def _load(paths: Sequence[Path], findings: list[dict[str, Any]]):
    """Return merged nodes, projection, and conflicted semantic IDs."""
    collected: list[dict[str, Any]] = []
    projection: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        location = f"/inputs[{index}]"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            findings.append(_n("invalid_input", location, "unreadable JSON input"))
            continue
        if _scan_private(payload, location, findings):
            continue
        nodes, entries = _document(payload, location, findings)
        collected.extend(nodes)
        projection.extend(entries)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for node in collected:
        grouped.setdefault(_key(node["kind"], node["id"]), []).append(node)
    merged: dict[str, dict[str, Any]] = {}
    conflicted: set[str] = set()
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) > 1:
            findings.append(_f("duplicate_semantic_id", f"/records/{key}", key, "duplicate id"))
        if len({json.dumps(node, sort_keys=True) for node in group}) > 1:
            conflicted.add(key)
            findings.append(
                _f("source_identity_conflict", f"/records/{key}", key, "identity conflict")
            )
        else:
            merged[key] = group[0]
    return merged, projection, frozenset(conflicted)


def _walk(
    root: Mapping[str, Any], nodes: Mapping[str, dict[str, Any]], conflicted: frozenset[str]
) -> set[str]:
    """Return the forward provenance closure of one root without entering another job."""
    visited = {_key(root["kind"], root["id"])}
    queue = deque(visited)
    if root["kind"] == "job":
        for key, node in sorted(nodes.items()):
            if root["id"] in node["refs"].get("job_ids", ()):
                visited.add(key)
                queue.append(key)
    while queue:
        node = nodes[queue.popleft()]
        for field, targets in sorted(node["refs"].items()):
            if REF_KIND[field] == "job":
                continue
            for target in targets:
                target_key = _key(REF_KIND[field], target)
                if target_key in visited or target_key in conflicted or target_key not in nodes:
                    continue
                visited.add(target_key)
                queue.append(target_key)
    return visited


def _link(category: str, missing_class: str, targets: Sequence[str] = ()) -> dict[str, Any]:
    return {"category": category, "class": missing_class, "targets": sorted(targets)}


def _missing_links(  # noqa: C901
    root: Mapping[str, Any],
    nodes: Mapping[str, dict[str, Any]],
    visited: set[str],
    records: Mapping[str, list[str]],
    conflicted: frozenset[str],
) -> list[dict[str, Any]]:
    """Return the classified missing-link inventory for one lineage row."""
    declared = {item for key in visited for item in nodes[key].get("not_applicable", ())}
    artifacts = [
        node for ident in records["artifact_ids"] if (node := nodes.get(_key("artifact", ident)))
    ]
    locators: dict[str, list[str]] = {}
    for node in artifacts:
        locators.setdefault(str(node.get("locator_class")), []).append(_key("artifact", node["id"]))

    def classify(category: str, field: str) -> list[dict[str, Any]]:
        attempted = sorted(
            {
                _key(REF_KIND[field], target)
                for key in visited
                for target in nodes[key]["refs"].get(field, ())
                if _key(REF_KIND[field], target) not in nodes
            }
        )
        if category in declared:
            return [_link(category, "not_applicable")]
        if any(target in conflicted for target in attempted):
            return [_link(category, "conflict", attempted)]
        if attempted:
            return [_link(category, "dangling", attempted)]
        return [_link(category, "not_recorded")]

    out: list[dict[str, Any]] = []
    for category in CATEGORIES:
        if category == "submission_receipt":
            state = root.get("submission_receipt", "not_recorded")
            if root["kind"] == "job" and state != "recorded":
                out.append(_link(category, state))
            continue
        if category == "artifact_locator":
            if locators.get("private_overlay"):
                out.append(_link(category, "private_unavailable", locators["private_overlay"]))
            elif locators.get("not_recorded"):
                out.append(_link(category, "not_recorded", locators["not_recorded"]))
            continue
        if category in ("raw_artifact", "compact_artifact"):
            if any(
                node.get("artifact_kind") == category.removesuffix("_artifact")
                for node in artifacts
            ):
                continue
            field = "artifact_ids"
        else:
            field = FIELD_BY_KIND[category]
            if records[field]:
                continue
        out.extend(classify(category, field))
    return sorted(out, key=lambda entry: entry["category"])


def _row(
    root: Mapping[str, Any], nodes: Mapping[str, dict[str, Any]], conflicted: frozenset[str]
) -> dict[str, Any]:
    """Build one deterministic lineage row from a job attempt or unreachable record."""
    visited = _walk(root, nodes, conflicted)
    collected: dict[str, set[str]] = {field: set() for field in REF_KIND}
    for key in sorted(visited):
        node = nodes[key]
        collected[FIELD_BY_KIND[node["kind"]]].add(node["id"])
        for field, targets in node["refs"].items():
            collected[field].update(
                target for target in targets if _key(REF_KIND[field], target) in nodes
            )
    records = {field: sorted(values) for field, values in collected.items()}
    digests = sorted(
        str(nodes[key]["digest"])
        for ident in records["artifact_ids"]
        if (key := _key("artifact", ident)) in nodes and "digest" in nodes[key]
    )
    owners = {
        kind: sorted(
            {
                str(nodes[key]["owner"])
                for ident in records[FIELD_BY_KIND[kind]]
                if (key := _key(kind, ident)) in nodes and "owner" in nodes[key]
            }
        )
        for kind in KINDS
    }
    missing = _missing_links(root, nodes, visited, records, conflicted)
    is_job = root["kind"] == "job"
    return {
        "lineage_key": _key(root["kind"], root["id"]),
        "root_kind": root["kind"],
        "root_id": root["id"],
        "attempt_index": root.get("attempt_index", 1) if is_job else None,
        "relation": root.get("relation", "initial") if is_job else "not_applicable",
        "predecessor_job_id": root.get("predecessor_job_id") if is_job else None,
        "records": records,
        "artifact_digests": digests,
        "owners": {kind: values for kind, values in owners.items() if values},
        "missing_links": missing,
        "reason_codes": sorted({entry["class"] for entry in missing}),
    }


def _reverse_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    lookup: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        lineage_key = str(row["lineage_key"])
        for field, values in row["records"].items():
            for value in values:
                lookup.setdefault(REF_KIND[field], {}).setdefault(value, set()).add(lineage_key)
        for digest in row["artifact_digests"]:
            lookup.setdefault("artifact_digest", {}).setdefault(digest, set()).add(lineage_key)
    return {
        kind: {value: sorted(keys) for value, keys in sorted(values.items())}
        for kind, values in sorted(lookup.items())
    }


def _integrity_findings(
    nodes: Mapping[str, dict[str, Any]],
    conflicted: frozenset[str],
    projection: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return dangling-pointer and public/private projection-drift findings."""
    findings: list[dict[str, Any]] = []
    for key in sorted(nodes):
        node = nodes[key]
        for field, targets in sorted(node["refs"].items()):
            for target in targets:
                target_key = _key(REF_KIND[field], target)
                if target_key in nodes or target_key in conflicted:
                    continue
                orphan = REF_KIND[field] == "artifact" and node["kind"] in ("analysis", "claim")
                findings.append(
                    _f(
                        "orphaned_artifact_pointer" if orphan else "dangling_reference",
                        key,
                        target_key,
                        "referenced record absent",
                    )
                )
    declared = {str(entry["target"]) for entry in projection}
    for entry in sorted(projection, key=lambda item: str(item["target"])):
        target = str(entry["target"])
        node = nodes.get(target)
        if target in conflicted or (node is not None and node.get("digest") == entry["digest"]):
            continue
        findings.append(
            _f(
                "projection_drift",
                "private_projection",
                target,
                "unknown target" if node is None else "public/private drift",
            )
        )
    findings.extend(
        _f("projection_drift", key, key, "withheld locator missing from projection")
        for key, node in sorted(nodes.items())
        if node["kind"] == "artifact"
        and node.get("locator_class") == "private_overlay"
        and key not in declared
    )
    return findings


def _payload(
    nodes: Sequence[Mapping[str, Any]],
    projection: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        findings, key=lambda item: (item["code"], item["source"], item["target"] or "")
    )
    missing = Counter(entry["class"] for row in rows for entry in row["missing_links"])
    return {
        "schema": INDEX_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "ok": not ordered,
        "status": "ok" if not ordered else "failed",
        "summary": {
            "record_count": len(nodes),
            "row_count": len(rows),
            "finding_count": len(ordered),
            "missing_link_counts": dict(sorted(missing.items())),
        },
        "findings": list(ordered),
        "reverse_lookup": _reverse_lookup(rows),
        "rows": list(rows),
    }


def build_index(inputs: Sequence[Path]) -> dict[str, Any]:
    """Load sanitized lineage inputs and return the deterministic index payload."""
    paths = sorted(
        {
            candidate
            for raw in inputs
            for candidate in (
                sorted(Path(raw).glob("*.json")) if Path(raw).is_dir() else [Path(raw)]
            )
        },
        key=str,
    )
    findings: list[dict[str, Any]] = []
    if not paths:
        findings.append(_n("invalid_input", "/inputs", "no inputs"))
        return _payload([], [], [], findings)
    nodes, projection, conflicted = _load(paths, findings)
    findings.extend(_integrity_findings(nodes, conflicted, projection))
    job_keys = sorted(key for key, node in nodes.items() if node["kind"] == "job")
    covered: set[str] = set()
    for key in job_keys:
        covered |= _walk(nodes[key], nodes, conflicted)
    roots = [nodes[key] for key in job_keys]
    roots.extend(nodes[key] for key in sorted(nodes) if key not in covered)
    roots.sort(key=lambda node: (KINDS.index(node["kind"]), node["id"]))
    return _payload(
        nodes.values(), projection, [_row(root, nodes, conflicted) for root in roots], findings
    )


def render_json(payload: Mapping[str, Any]) -> str:
    """Return byte-stable index JSON with sorted keys and a trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Return the deterministic human-readable Markdown index projection."""
    summary = payload["summary"]
    lines = [
        "# Lineage Index",
        "",
        f"- Schema: `{payload['schema']}` | Status: `{payload['status']}`",
        f"- Rows: {summary['row_count']} | Records: {summary['record_count']}"
        f" | Findings: {summary['finding_count']}",
        "",
        payload["claim_boundary"],
        "",
        "## Findings",
        "",
    ]
    lines.append(
        " | ".join(f"`{i['code']}`" for i in payload["findings"])
        if payload["findings"]
        else "- none"
    )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Lineage key | Attempt | Relation | Predecessor | Missing | Joined records |",
            "| --- | --- | --- | --- | --- | --- |",
            *(_row_line(row) for row in payload["rows"]),
        ]
    )
    return "\n".join(lines) + "\n"


def _row_line(row: Mapping[str, Any]) -> str:
    records = "; ".join(
        f"{field}=({','.join(values)})"
        for field, values in sorted(row["records"].items())
        if values
    )
    missing = "; ".join(f"{e['category']}={e['class']}" for e in row["missing_links"]) or "-"
    return (
        f"| `{row['lineage_key']}` | {row['attempt_index']} | `{row['relation']}` |"
        f" `{row['predecessor_job_id'] or '-'}` | {missing} | {records} |"
    )


def _query_matches(row: Mapping[str, Any], selectors: Mapping[str, str]) -> bool:
    checks = {
        "issue": row["records"]["issue_ids"],
        "job": row["records"]["job_ids"],
        "campaign": row["records"]["campaign_ids"],
        "commit": row["records"]["commit_ids"],
        "config": row["records"]["config_ids"],
        "artifact_digest": row["artifact_digests"],
    }
    return all(value in checks[key] for key, value in selectors.items())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lineage_index", description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", type=Path, help="Sanitized input file or dir.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Read-only check invocation (the tool never writes state).",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    query = parser.add_subparsers(dest="command").add_parser(
        "query", help="Select rows by stable semantic identity."
    )
    query.add_argument("--input", action="append", type=Path, required=True)
    for selector in ("issue", "job", "campaign", "artifact-digest", "commit", "config"):
        query.add_argument(f"--{selector}")
    query.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser


def _run_query(args: argparse.Namespace) -> int:
    selectors = {
        key.replace("-", "_"): value
        for key in ("issue", "job", "campaign", "artifact_digest", "commit", "config")
        if (value := getattr(args, key))
    }
    payload = build_index(args.input)
    rows = [row for row in payload["rows"] if _query_matches(row, selectors)]
    if args.format == "markdown":
        header = (
            f"# Lineage Query\n\n- Query: `{json.dumps(selectors, sort_keys=True)}`"
            f"\n- Matches: {len(rows)}\n\n"
        )
        sys.stdout.write(
            header + "\n".join(_row_line(row) for row in rows) + ("\n" if rows else "")
        )
    else:
        result = {
            "schema": INDEX_SCHEMA,
            "query": selectors,
            "match_count": len(rows),
            "rows": rows,
        }
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if rows else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the lineage index generator or query and return a shell-friendly exit code."""
    args = _build_parser().parse_args(argv)
    if getattr(args, "command", None) == "query":
        return _run_query(args)
    if not args.input:
        sys.stderr.write("FAIL invalid_input: --input is required\n")
        return 2
    payload = build_index(args.input)
    rendered = render_markdown(payload) if args.format == "markdown" else render_json(payload)
    sys.stdout.write(rendered)
    return 0 if payload["ok"] else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
