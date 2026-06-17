#!/usr/bin/env python3
"""Generate a deterministic release-readiness dashboard from local context sources."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import yaml

DASHBOARD_SCHEMA_VERSION = "release-readiness-dashboard.v1"
DEFAULT_CLAIM_MAP_PATH = Path("docs/context/issue_2943_fast_results_claim_map_v0.md")
DEFAULT_HANDOFF_PATH = Path("docs/context/issue_2689_release_evidence_handoff_2026_06_15.md")
DEFAULT_CATALOG_PATH = Path("docs/context/catalog.yaml")

READY_TIERS = {"schema", "candidate", "paper_ready", "smoke"}
DIAGNOSTIC_TIERS = {"diagnostic", "do_not_claim", "do-not-claim"}
BLOCKED_TIERS = {"blocked", "do_not_claim", "do-not-claim"}
COMPLETE_QUEUE_STATUS = {"ready", "closed", "done", "completed"}
FILE_EXTENSIONS = {
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".txt",
    ".md5",
    ".sha256",
    ".toml",
    ".png",
}


def get_repository_root() -> Path:
    """Return repository root for CLI-style relative path resolution."""
    return Path(__file__).resolve().parents[2]


class SourceIssueStatus:
    """Small helper for optional issue snapshot lookups."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        """Store normalized issue states."""
        self._mapping = {
            str(key): self._normalize_status(value) for key, value in (mapping or {}).items()
        }

    @staticmethod
    def _normalize_status(value: str | None) -> str:
        """Normalize issue states into canonical forms."""
        if value is None:
            return "unknown"
        text = str(value).strip().lower()
        if text in {"closed", "done", "merged", "completed", "resolved"}:
            return "closed"
        if text in {
            "open",
            "ready",
            "todo",
            "in_progress",
            "in progress",
            "blocked",
            "assigned",
            "reopened",
            "open+assigned",
        }:
            return "open"
        if text in {"not ready", "not_ready"}:
            return "open"
        return text

    def is_open(self, issue: str | int) -> bool:
        """Return `True` unless the issue is explicitly closed in snapshot."""
        return self._mapping.get(str(issue), "unknown") not in {
            "closed",
            "done",
            "merged",
            "completed",
            "resolved",
        }


def _read_text(path: Path) -> str:
    """Read UTF-8 file content."""
    return path.read_text(encoding="utf-8")


def _safe_relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    root = get_repository_root()
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    """Return an RFC3339-like UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_header_name(name: str) -> str:
    """Normalize a markdown table header cell to a canonical key."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _is_table_separator(line: str) -> bool:
    """Match a markdown table separator row."""
    return bool(re.match(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", line))


def _split_row(line: str) -> list[str]:
    """Split a markdown table row into trimmed cells."""
    clean = line.strip().strip("|")
    if not clean:
        return []
    return [cell.strip() for cell in clean.split("|")]


def _extract_section(lines: list[str], heading_prefix: str) -> list[str]:
    """Return text lines under a heading prefix."""
    start = None
    level = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(heading_prefix):
            start = idx
            level = len(re.match(r"^\s*(#+)", stripped).group(1))  # type: ignore[union-attr]
            break
    if start is None or level is None:
        return []
    section: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if re.match(r"^\s*#{1,6}\s+\S", stripped):
            if len(re.match(r"^\s*(#+)", stripped).group(1)) <= level:  # type: ignore[union-attr]
                break
        section.append(line)
    return section


def _extract_tables(section_lines: list[str]) -> list[tuple[list[str], list[dict[str, str]]]]:
    """Extract markdown tables as `(headers, rows)` from a line list."""
    tables: list[tuple[list[str], list[dict[str, str]]]] = []
    idx = 0
    while idx < len(section_lines):
        line = section_lines[idx].strip()
        if not line.startswith("|"):
            idx += 1
            continue
        if idx + 1 >= len(section_lines) or not _is_table_separator(section_lines[idx + 1]):
            idx += 1
            continue
        headers = [_normalize_header_name(cell) for cell in _split_row(line)]
        idx += 2
        rows: list[dict[str, str]] = []
        while idx < len(section_lines):
            current = section_lines[idx].strip()
            if not current.startswith("|"):
                break
            cells = _split_row(current)
            if cells:
                rows.append(
                    {headers[i]: (cells[i] if i < len(cells) else "") for i in range(len(headers))}
                )
            idx += 1
        if rows:
            tables.append((headers, rows))
    return tables


def _normalize_tier(raw: str) -> str:
    """Normalize tier tags while preserving inline-markdown wrapped values."""
    values = re.findall(r"`([^`]+)`", str(raw))
    if values:
        return values[-1].strip().lower()
    return str(raw).strip().lower()


def _strip_inline(text: str) -> str:
    """Remove simple markdown inline emphasis for normalized output."""
    return re.sub(r"[`*]", "", str(text)).strip()


def _extract_issue_refs(text: str) -> list[str]:
    """Return issue numbers found as `#<number>` tokens."""
    return [match.group(1) for match in re.finditer(r"#(\d+)", str(text))]


def _extract_issue_hints(text: str) -> dict[str, str]:
    """Extract parenthesized issue state hints like `#2910 (open)`."""
    hints: dict[str, str] = {}
    for match in re.finditer(r"#(\d+)\s*\(([^)]+)\)", str(text)):
        issue = match.group(1)
        state = match.group(2).lower()
        if "closed" in state or "implemented" in state:
            hints[issue] = "closed"
        elif (
            "open" in state
            or "ready" in state
            or "todo" in state
            or "in progress" in state
            or "blocked" in state
        ):
            hints[issue] = "open"
        elif state:
            hints[issue] = state.strip()
    return hints


def _parse_issue_state(
    issue: str, source_hints: dict[str, str], snapshot: SourceIssueStatus | None = None
) -> str:
    """Resolve issue state from inline hint or snapshot."""
    if issue in source_hints:
        return source_hints[issue]
    if snapshot is None:
        return "unknown"
    return "closed" if not snapshot.is_open(issue) else "open"


def parse_claim_map(
    text: str,
    issue_snapshot: SourceIssueStatus | None = None,
    claim_map_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse the claim map table and classify claims."""
    table_section = _extract_section(text.splitlines(), "### Claim Map Table")
    tables = _extract_tables(table_section)
    ready: list[dict[str, Any]] = []
    diagnostic_only: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    required_headers = {
        "id",
        "target_table_surface",
        "required_issue_s",
        "evidence_tier",
        "blocked_dependency",
        "do_not_claim_boundary",
    }

    for headers, rows in tables:
        if not required_headers.issubset(set(headers)):
            continue
        for row in rows:
            raw_required = _strip_inline(row.get("required_issue_s", ""))
            raw_blocked_dep = _strip_inline(row.get("blocked_dependency", ""))
            issue_hints = _extract_issue_hints(f"{raw_required} {raw_blocked_dep}")
            requirement = _strip_inline(
                row.get(
                    "target_table_surface",
                    row.get("target_table_or_surface", row.get("requirement", "")),
                )
            )
            required = sorted(_extract_issue_refs(raw_required), key=int)
            blockers = sorted(set(required + _extract_issue_refs(raw_blocked_dep)), key=int)
            unresolved_blockers = [
                issue
                for issue in blockers
                if _parse_issue_state(issue, issue_hints, issue_snapshot)
                not in {"closed", "done", "merged", "resolved"}
            ]
            dependency_blocked = bool(
                re.search(
                    r"\b(not\s+defined|not\s+available|missing\s+until|missing|must\s+close|blocked)\b",
                    raw_blocked_dep.lower(),
                )
            )
            record: dict[str, Any] = {
                "id": _strip_inline(row.get("id", "")),
                "requirement": requirement,
                "required_issues": required,
                "required_issue_hints": issue_hints,
                "evidence_tier": _normalize_tier(row.get("evidence_tier", "")),
                "raw_evidence_tier": _strip_inline(row.get("evidence_tier", "")),
                "blocked_dependency": raw_blocked_dep,
                "blocked_by_issues": blockers,
                "dependency_gate_blocked": dependency_blocked,
                "do_not_claim_boundary": _strip_inline(row.get("do_not_claim_boundary", "")),
                "blocked_dependency_issues": unresolved_blockers,
                "status": "blocked",
                "provenance": [str(_safe_relative(claim_map_path or DEFAULT_CLAIM_MAP_PATH))],
            }
            tier = record["evidence_tier"]
            if tier in DIAGNOSTIC_TIERS:
                record["status"] = "diagnostic_only"
                diagnostic_only.append(record)
            elif tier in BLOCKED_TIERS or unresolved_blockers or dependency_blocked:
                record["status"] = "blocked"
                blocked.append(record)
            elif tier in READY_TIERS:
                record["status"] = "ready"
                ready.append(record)
            else:
                record["status"] = "blocked"
                blocked.append(record)

    ready.sort(key=lambda item: item["id"])
    diagnostic_only.sort(key=lambda item: item["id"])
    blocked.sort(key=lambda item: item["id"])
    return ready, diagnostic_only, blocked


def parse_queue_rows(text: str) -> list[dict[str, Any]]:
    """Parse `p0_now`, `p1_after_gate`, and `parked_blocked` queue tables."""
    lines = text.splitlines()
    section_headers = [
        "### p0_now --",
        "### p1_after_gate --",
        "### parked_blocked --",
    ]
    rows: list[dict[str, Any]] = []
    for heading in section_headers:
        table_section = _extract_section(lines, heading)
        for headers, table_rows in _extract_tables(table_section):
            if not {
                "item",
                "owner_issue",
                "status",
                "next_command_or_artifact",
                "evidence_gate",
                "durable_evidence",
            }.issubset(set(headers)):
                continue
            section_name = heading.split("--")[0].replace("###", "").strip()
            for row in table_rows:
                owner_issue = _extract_issue_refs(row.get("owner_issue", ""))
                owner = owner_issue[0] if owner_issue else ""
                evidence_gate = _strip_inline(row.get("evidence_gate", ""))
                next_artifact = _strip_inline(row.get("next_command_or_artifact", ""))
                status = _normalize_tier(row.get("status", ""))
                blocked_by = _extract_issue_refs(
                    f"{row.get('evidence_gate', '')} {row.get('durable_evidence', '')} {row.get('item', '')}"
                )
                rows.append(
                    {
                        "requirement": _strip_inline(row.get("item", "")),
                        "owner_issue": owner,
                        "status": status,
                        "next_command_or_artifact": next_artifact,
                        "evidence_gate": evidence_gate,
                        "blocked_by_issues": sorted(set(blocked_by), key=int),
                        "section": section_name,
                    }
                )
    rows.sort(
        key=lambda item: (
            item["section"],
            item["status"],
            item["owner_issue"] or item["requirement"],
        )
    )
    return rows


def build_next_executable_requirements(
    queue_rows: list[dict[str, Any]],
    issue_snapshot: SourceIssueStatus | None = None,
) -> list[dict[str, Any]]:
    """Compute the next executable issue for each blocked or incomplete queue item."""
    next_items: list[dict[str, Any]] = []
    for row in queue_rows:
        status = row["status"]
        if status in COMPLETE_QUEUE_STATUS:
            continue

        blocker_candidates = [
            issue
            for issue in row.get("blocked_by_issues", [])
            if (issue_snapshot.is_open(issue) if issue_snapshot else True)
        ]
        owner_issue = row["owner_issue"] or None
        if (
            issue_snapshot is not None
            and owner_issue is not None
            and not issue_snapshot.is_open(owner_issue)
            and not blocker_candidates
        ):
            continue
        next_issue = blocker_candidates[0] if blocker_candidates else owner_issue
        next_items.append(
            {
                "requirement": row["requirement"],
                "owner_issue": owner_issue,
                "status": status,
                "section": row["section"],
                "next_executable_issue": next_issue,
                "blocked_by_issues": blocker_candidates,
                "next_command_or_artifact": row["next_command_or_artifact"],
            }
        )
    next_items.sort(
        key=lambda item: (
            item["section"],
            item["status"],
            item["owner_issue"] or item["requirement"],
        )
    )
    return next_items


def collect_missing_hazard_coverage(claim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect likely ODD/HAZARD coverage gaps from claim rows."""
    missing: list[dict[str, Any]] = []
    for row in claim_rows:
        text = f"{row.get('id', '')} {row.get('requirement', '')} {row.get('blocked_dependency', '')}".lower()
        if "odd" not in text and "hazard" not in text:
            continue
        if row.get("status") == "ready":
            continue
        missing.append(
            {
                "claim_id": row.get("id", ""),
                "requirement": row.get("requirement", ""),
                "status": row.get("status", ""),
                "missing": row.get("blocked_dependency", ""),
                "provenance": [str(DEFAULT_CLAIM_MAP_PATH)],
            }
        )
    return missing


def _load_catalog_paths(path: Path) -> list[str]:
    """Load catalog paths."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    return [
        str(item.get("path"))
        for item in entries
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ]


def _collect_artifact_refs(text: str) -> set[str]:
    """Collect repo-like local artifact references from markdown fragments."""
    refs: set[str] = set()

    for match in re.finditer(r"\]\(([^)]+)\)", text):
        token = match.group(1).strip()
        if token and "http" not in token and token.endswith(tuple(FILE_EXTENSIONS)):
            refs.add(token.split("#")[0].split("?")[0])

    for match in re.finditer(r"`([^`]+)`", text):
        token = match.group(1).strip()
        if not token or token.startswith("http") or " " in token:
            continue
        if "/" not in token:
            continue
        if token.endswith(tuple(FILE_EXTENSIONS)):
            refs.add(token)

    return refs


def collect_missing_artifacts(
    handoff_text: str,
    catalog_path: Path,
    repo_root: Path,
    *,
    include_catalog: bool = True,
) -> list[dict[str, Any]]:
    """Find catalog or handoff pointers that do not resolve in the repo."""
    missing: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for path_text in sorted(_collect_artifact_refs(handoff_text)):
        candidate = repo_root / path_text
        if not candidate.exists():
            key = (path_text, "issue_2689_release_evidence_handoff_2026_06_15.md")
            if key not in seen:
                seen.add(key)
                missing.append(
                    {
                        "path": path_text,
                        "source": "docs/context/issue_2689_release_evidence_handoff_2026_06_15.md",
                        "reason": "missing local artifact path",
                    }
                )

    if include_catalog:
        for path_text in _load_catalog_paths(catalog_path):
            candidate = repo_root / path_text
            if not candidate.exists():
                key = (path_text, "docs/context/catalog.yaml")
                if key not in seen:
                    seen.add(key)
                    missing.append(
                        {
                            "path": path_text,
                            "source": "docs/context/catalog.yaml",
                            "reason": "catalog entry path missing from repository",
                        }
                    )

    return missing


def _load_issue_snapshot(snapshot_path: Path | None) -> SourceIssueStatus:  # noqa: C901, PLR0912
    """Load issue snapshot structure with permissive shape handling."""
    if snapshot_path is None:
        return SourceIssueStatus({})

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    entries: list[tuple[str, str]] = []

    if isinstance(payload, dict):
        if "issues" in payload and isinstance(payload["issues"], list):
            for issue in payload["issues"]:
                if not isinstance(issue, dict):
                    continue
                number = issue.get("number") or issue.get("id") or issue.get("issue")
                state = issue.get("state") or issue.get("status") or issue.get("state_reason")
                if number is None or state is None:
                    continue
                entries.append((str(number), str(state)))
        elif "nodes" in payload:
            nodes = payload["nodes"]
            if isinstance(nodes, list):
                for issue in nodes:
                    if isinstance(issue, dict):
                        number = issue.get("number") or issue.get("id") or issue.get("issue")
                        state = (
                            issue.get("state") or issue.get("status") or issue.get("state_reason")
                        )
                        if number is None or state is None:
                            continue
                        entries.append((str(number), str(state)))
        else:
            for key, value in payload.items():
                if isinstance(key, (str, int)) and isinstance(value, (str, int)):
                    entries.append((str(key), str(value)))
    elif isinstance(payload, list):
        for issue in payload:
            if isinstance(issue, dict):
                number = issue.get("number") or issue.get("id") or issue.get("issue")
                state = issue.get("state") or issue.get("status") or issue.get("state_reason")
                if number is None or state is None:
                    continue
                entries.append((str(number), str(state)))

    return SourceIssueStatus(dict(entries))


def build_markdown(payload: dict[str, Any]) -> str:  # noqa: C901, PLR0912
    """Build a compact markdown dashboard with provenance and caveat text."""
    lines = [
        "# Release Readiness Dashboard",
        "",
        f"Generated (UTC): {payload['generated_at_utc']}",
        f"Schema: {payload['schema_version']}",
        "",
        "## Claim Boundaries",
        "- Source-local evidence only; no benchmark/paper claim is promoted beyond cited source notes.",
        "- Rows classified as `diagnostic_only` are caveated and should not be promoted to benchmark/paper claims.",
        "",
        "## Sources",
    ]
    for label, source in payload["sources"].items():
        lines.append(f"- **{label}**: {source}")

    lines.extend(["", "## Ready Claims"])
    if payload["ready_claims"]:
        for item in payload["ready_claims"]:
            lines.append(f"- {item['id']}: {item['requirement']} ({item['evidence_tier']})")
    else:
        lines.append("- None")

    lines.extend(["", "## Diagnostic-Only Claims"])
    if payload["diagnostic_only_claims"]:
        for item in payload["diagnostic_only_claims"]:
            lines.append(f"- {item['id']}: {item['requirement']} ({item['evidence_tier']})")
    else:
        lines.append("- None")

    lines.extend(["", "## Blocked Claims"])
    if payload["blocked_claims"]:
        for item in payload["blocked_claims"]:
            blockers = (
                ", ".join(item.get("blocked_dependency_issues", [])) or "blocked by source gate"
            )
            lines.append(f"- {item['id']}: {item['requirement']} (blockers: {blockers})")
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Hazard Coverage"])
    if payload["missing_hazard_coverage"]:
        for item in payload["missing_hazard_coverage"]:
            lines.append(f"- {item['claim_id']}: {item['missing'] or 'gap remains'}")
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Durable Artifact Pointers"])
    if payload["missing_durable_artifact_pointers"]:
        for item in payload["missing_durable_artifact_pointers"]:
            lines.append(f"- `{item['path']}` ({item['source']})")
    else:
        lines.append("- None")

    lines.extend(["", "## Next Executable Issue per Blocked/Incomplete Requirement"])
    if payload["next_executable_requirements"]:
        for item in payload["next_executable_requirements"]:
            lines.append(
                f"- {item['requirement']} ({item['section']}): {item['next_executable_issue'] or 'no open local issue identified'}"
            )
    else:
        lines.append("- None")

    lines.append("")
    lines.append(payload["diagnostic_only_caveat"])
    return "\n".join(lines) + "\n"


def generate_release_readiness_dashboard(
    *,
    claim_map_path: Path,
    handoff_path: Path,
    catalog_path: Path,
    issue_snapshot: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Generate structured dashboard payload and markdown from local sources."""
    if not claim_map_path.is_file():
        raise FileNotFoundError(f"Missing claim-map source: {claim_map_path}")
    if not handoff_path.is_file():
        raise FileNotFoundError(f"Missing handoff source: {handoff_path}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing catalog source: {catalog_path}")

    claim_map_text = _read_text(claim_map_path)
    handoff_text = _read_text(handoff_path)
    issue_status = _load_issue_snapshot(issue_snapshot)
    catalog_paths = _load_catalog_paths(catalog_path)

    ready_claims, diagnostic_only_claims, blocked_claims = parse_claim_map(
        claim_map_text,
        issue_snapshot=issue_status,
        claim_map_path=claim_map_path,
    )
    queue_rows = parse_queue_rows(claim_map_text)
    next_executable = build_next_executable_requirements(queue_rows, issue_status)

    all_claim_rows = ready_claims + diagnostic_only_claims + blocked_claims
    missing_hazard = collect_missing_hazard_coverage(all_claim_rows)
    missing_artifacts = collect_missing_artifacts(
        handoff_text=handoff_text,
        catalog_path=catalog_path,
        repo_root=get_repository_root(),
    )

    payload = {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "sources": {
            "claim_map": _safe_relative(claim_map_path),
            "handoff": _safe_relative(handoff_path),
            "catalog": _safe_relative(catalog_path),
        },
        "provenance": {
            "claim_map": _safe_relative(claim_map_path),
            "handoff": _safe_relative(handoff_path),
            "catalog": _safe_relative(catalog_path),
            "issue_snapshot": issue_snapshot.name if issue_snapshot else None,
            "script": _safe_relative(Path(__file__)),
        },
        "source_status": {
            "catalog_entry_count": len(catalog_paths),
            "claim_map_contains_in_catalog": _safe_relative(claim_map_path) in catalog_paths,
            "handoff_contains_in_catalog": _safe_relative(handoff_path) in catalog_paths,
        },
        "diagnostic_only_caveat": (
            "Diagnostic-only rows are allowed for local execution diagnostics and "
            "must not be promoted as benchmark/paper evidence."
        ),
        "ready_claims": ready_claims,
        "diagnostic_only_claims": diagnostic_only_claims,
        "blocked_claims": blocked_claims,
        "missing_hazard_coverage": missing_hazard,
        "missing_durable_artifact_pointers": missing_artifacts,
        "next_executable_requirements": next_executable,
    }
    return payload, build_markdown(payload)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-map", type=Path, default=DEFAULT_CLAIM_MAP_PATH)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--issue-snapshot", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    payload, markdown = generate_release_readiness_dashboard(
        claim_map_path=args.claim_map,
        handoff_path=args.handoff,
        catalog_path=args.catalog,
        issue_snapshot=args.issue_snapshot,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
