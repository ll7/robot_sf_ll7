"""Tests for the release-readiness dashboard generator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.tools import generate_release_readiness_dashboard as cli

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    """Create a UTF-8 text fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _write_yaml(path: Path, text: str) -> None:
    """Write YAML fixture content."""
    _write(path, text)


def test_parse_claim_map_classifies_rows_by_tier_and_open_dependency() -> None:
    """Claim rows are split into ready, diagnostic-only, and blocked deterministically."""
    claim_map = """
### Claim Map Table
| ID | Target table / surface | Required issue(s) | Evidence tier | Blocked dependency | Do-not-claim boundary |
|---|---|---|---|---|---|
| `cm-001` | Ready claim | #3010 (closed) | `schema` | None | - |
| `cm-002` | Diagnostic row | #3011 (closed) | `diagnostic` | None | Diagnostic only |
| `cm-003` | ODD/hazard coverage | #3012 (open) | `schema` | Odd coverage matrix undefined | Not ready until #3012 is closed |
| `cm-004` | Blocked dependency claim | #3013 (closed) | `blocked` | Do not claim before release freeze | Explicitly blocked |
"""

    ready, diagnostic, blocked = cli.parse_claim_map(
        claim_map, cli.SourceIssueStatus({"3010": "closed", "3013": "closed"})
    )

    assert [row["id"] for row in ready] == ["cm-001"]
    assert [row["id"] for row in diagnostic] == ["cm-002"]
    assert [row["id"] for row in blocked] == ["cm-003", "cm-004"]
    blocked_ids = {row["id"] for row in blocked}
    assert "cm-004" in blocked_ids


def test_issue_snapshot_controls_readiness(tmp_path: Path) -> None:
    """Known issue-snapshot states gate previously dependent claims."""
    claim_map = """
### Claim Map Table
| ID | Target table / surface | Required issue(s) | Evidence tier | Blocked dependency | Do-not-claim boundary |
|---|---|---|---|---|---|
| `cm-005` | Snapshot-driven requirement | #3020 | `schema` | Do not claim before #3020 closure | - |
"""

    open_ready, _, open_blocked = cli.parse_claim_map(
        claim_map,
        cli.SourceIssueStatus({"3020": "open"}),
    )
    closed_ready, _, _ = cli.parse_claim_map(claim_map, cli.SourceIssueStatus({"3020": "closed"}))

    assert open_blocked
    assert open_ready == []
    assert [row["id"] for row in closed_ready] == ["cm-005"]


def test_cli_generates_json_and_markdown_outputs_and_includes_hazard_gap(
    tmp_path: Path, monkeypatch
) -> None:
    """CLI writes both output formats and includes required dashboard fields."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(cli, "get_repository_root", lambda: repo_root)

    claim_map_path = repo_root / "docs" / "context" / "claim_map.md"
    _write(
        claim_map_path,
        """
# Claim map fixture

### Claim Map Table
| ID | Target table / surface | Required issue(s) | Evidence tier | Blocked dependency | Do-not-claim boundary |
|---|---|---|---|---|---|
| `cm-006` | Hazard coverage baseline | #4000 (closed) | `schema` | ODD/hazard matrix not defined | Do not claim hazard as closed |
| `cm-007` | Diagnostic-only behavior | #4001 (closed) | `diagnostic` | None | Diagnostic caveat |
| `cm-008` | Smoke execution | #4002 (closed) | `smoke` | None | - |

### p0_now -- No blocking gates; work can start or land immediately
| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Resolve #4000 | #4000 | ready | Artifact ready | p0 condition met | Missing until #4000 closes |
""",
    )

    handoff_path = repo_root / "docs" / "context" / "handoff.md"
    _write(
        handoff_path,
        """
# Handoff fixture

[one](docs/context/issue_existing.md)
`report.md`
`missing/path/to/artifact.json`
""",
    )

    existing = repo_root / "docs" / "context" / "issue_existing.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("ok", encoding="utf-8")

    catalog_path = repo_root / "docs" / "context" / "catalog.yaml"
    _write_yaml(
        catalog_path,
        """
version: 1
entries:
  - path: docs/context/issue_existing.md
  - path: docs/context/missing_catalog.json
""",
    )

    snapshot = tmp_path / "issue_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {"issues": [{"number": 4001, "state": "open"}, {"number": 4002, "state": "closed"}]}
        ),
        encoding="utf-8",
    )

    json_out = tmp_path / "release_readiness.json"
    md_out = tmp_path / "release_readiness.md"

    assert (
        cli.main(
            [
                "--claim-map",
                str(claim_map_path),
                "--handoff",
                str(handoff_path),
                "--catalog",
                str(catalog_path),
                "--issue-snapshot",
                str(snapshot),
                "--json-output",
                str(json_out),
                "--markdown-output",
                str(md_out),
            ]
        )
        == 0
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "release-readiness-dashboard.v1"
    assert str(tmp_path) not in json_out.read_text(encoding="utf-8")
    assert payload["ready_claims"][0]["id"] == "cm-008"
    assert payload["diagnostic_only_claims"][0]["id"] == "cm-007"
    assert payload["blocked_claims"][0]["id"] == "cm-006"
    assert payload["missing_hazard_coverage"]
    assert payload["missing_durable_artifact_pointers"]
    assert any(
        item["path"] == "missing/path/to/artifact.json"
        for item in payload["missing_durable_artifact_pointers"]
    )
    assert not any(
        item["path"] == "report.md" for item in payload["missing_durable_artifact_pointers"]
    )
    assert any(
        item["path"] == "docs/context/missing_catalog.json"
        for item in payload["missing_durable_artifact_pointers"]
    )

    markdown = md_out.read_text(encoding="utf-8")
    assert "Ready Claims" in markdown
    assert "Diagnostic-Only Claims" in markdown
    assert "Missing Durable Artifact Pointers" in markdown
    assert "must not be promoted as benchmark/paper evidence" in markdown


def test_next_executable_uses_open_blockers(tmp_path: Path, monkeypatch) -> None:
    """Queue parsing picks the next executable issue from open blocked dependencies."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(cli, "get_repository_root", lambda: repo_root)

    claim_map_path = repo_root / "docs" / "context" / "claim_map.md"
    _write(
        claim_map_path,
        """
### Claim Map Table
| ID | Target table / surface | Required issue(s) | Evidence tier | Blocked dependency | Do-not-claim boundary |
|---|---|---|---|---|---|
| `cm-001` | Smoke path | #5001 (closed) | `smoke` | None | - |

### p1_after_gate -- Gated on a named p0 precondition
| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| ODD/AMV suite freeze | #5000 | blocked | Freeze artifact | p0: #5002 | Missing until #5002 closes |
| Forecast benchmark | #5003 | blocked | Artifact: benchmark report | p0: #5001 | Missing until #5001 closes |
""",
    )

    handoff_path = repo_root / "docs" / "context" / "handoff.md"
    catalog_path = repo_root / "docs" / "context" / "catalog.yaml"
    _write(handoff_path, "[]")
    _write(catalog_path, "version: 1\nentries: []\n")
    snapshot_path = repo_root / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "issues": [
                    {"number": 5001, "state": "closed"},
                    {"number": 5002, "state": "open"},
                    {"number": 5003, "state": "open"},
                ]
            }
        ),
        encoding="utf-8",
    )

    payload, _ = cli.generate_release_readiness_dashboard(
        claim_map_path=claim_map_path,
        handoff_path=handoff_path,
        catalog_path=catalog_path,
        issue_snapshot=snapshot_path,
    )

    requirements = payload["next_executable_requirements"]
    assert requirements[0]["requirement"] == "ODD/AMV suite freeze"
    assert requirements[0]["next_executable_issue"] == "5002"
    assert requirements[1]["requirement"] == "Forecast benchmark"
    assert requirements[1]["next_executable_issue"] == "5003"


def test_inline_cleanup_preserves_identifier_underscores() -> None:
    """Markdown cleanup must not corrupt schema and claim identifiers."""

    ready, _, _ = cli.parse_claim_map(
        """
### Claim Map Table
| ID | Target table / surface | Required issue(s) | Evidence tier | Blocked dependency | Do-not-claim boundary |
|---|---|---|---|---|---|
| `cm-v0.mechanism.trace_schema` | `mechanism_trace.v1` source contract | #2923 (closed) | `schema` | None | - |
"""
    )

    assert ready[0]["id"] == "cm-v0.mechanism.trace_schema"
    assert ready[0]["requirement"] == "mechanism_trace.v1 source contract"
