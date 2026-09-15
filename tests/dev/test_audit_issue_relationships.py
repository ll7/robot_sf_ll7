"""Contract tests for the issue-relationship audit helper (issue #9353)."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.dev.audit_issue_relationships import (
    AuditResult,
    apply_migration,
    audit_issue,
    main,
    parse_relationship_block,
)

MIRROR_BODY = """## Goal

Something useful.

## Relationships

<!-- Native GitHub relationships are canonical; this block mirrors intentional links. -->
- Parent issue: #9293
- Blocked by: #100, #101
- Blocking: none
- Relates to: #55

## Scope

Narrow.
"""


def test_parse_relationship_block_reads_canonical_rows() -> None:
    """The canonical mirror block parses into typed parent/blocked lists."""
    mirror = parse_relationship_block(MIRROR_BODY)

    assert mirror is not None
    assert mirror.parent == 9293
    assert mirror.blocked_by == (100, 101)
    assert mirror.blocking == ()
    assert mirror.relates_to == (55,)
    assert mirror.ambiguous == ()
    assert mirror.cross_repo == ()


def test_parse_relationship_block_absent_returns_none() -> None:
    """Bodies without the block report absence instead of an empty mirror."""
    assert parse_relationship_block("## Goal\n\nNo block here.\n") is None


def test_parse_relationship_block_rejects_multiple_parents() -> None:
    """Two parent refs are ambiguous and must never resolve to one."""
    mirror = parse_relationship_block(
        "## Relationships\n- Parent issue: #1, #2\n- Blocked by: none\n"
    )

    assert mirror is not None
    assert mirror.parent is None
    assert "multiple_parent_refs" in mirror.ambiguous


def test_parse_relationship_block_flags_cross_repo_urls() -> None:
    """Full URLs to other repositories are preserved for explicit refusal."""
    mirror = parse_relationship_block(
        "## Relationships\n- Parent issue: none\n"
        "- Blocked by: https://github.com/other/repo/issues/7\n"
    )

    assert mirror is not None
    assert mirror.cross_repo == ("other/repo#7",)


def test_parse_relationship_block_flags_placeholders() -> None:
    """TBD-style tokens are ambiguous, never silently treated as none."""
    mirror = parse_relationship_block("## Relationships\n- Parent issue: TBD\n- Blocked by: none\n")

    assert mirror is not None
    assert "ambiguous_placeholder_token" in mirror.ambiguous


def test_apply_requires_exact_confirmation_token(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--apply without the exact token is a usage error, never a migration."""
    assert main(["123", "--apply"]) == 2
    assert main(["123", "--apply", "--confirm", "WRONG"]) == 2
    assert "RELATIONSHIP_MIGRATION" in capsys.readouterr().err


def test_apply_refuses_without_clean_drift() -> None:
    """Apply on a refused or clean audit never touches the network."""
    refused = AuditResult(
        issue=1,
        repo="ll7/robot_sf_ll7",
        mirror_present=True,
        refused=["ambiguous_mirror:multiple_parent_refs"],
        verdict="refused",
    )
    out = apply_migration(refused)

    assert out.error is not None
    assert out.applied == []

    clean = AuditResult(issue=1, repo="ll7/robot_sf_ll7", verdict="in_sync")
    assert apply_migration(clean).error is not None


def test_audit_reports_unreadable_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transport failures surface as error verdicts, never empty audits."""
    import scripts.dev.audit_issue_relationships as module

    def _boom(number: int, repo: str) -> tuple[None, str]:
        return None, "boom"

    monkeypatch.setattr(module, "_rest_issue_body", _boom)
    audit = audit_issue(4242, repo="ll7/robot_sf_ll7")

    assert audit.verdict == "error"
    assert audit.error == "boom"


def test_audit_computes_drift_from_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror/native comparison emits stable drift codes without network."""
    import scripts.dev.audit_issue_relationships as module

    monkeypatch.setattr(module, "_rest_issue_body", lambda number, repo: (MIRROR_BODY, None))
    monkeypatch.setattr(
        module,
        "_graphql_parent_children",
        lambda number, repo: ({"parent": None, "children": []}, None),
    )
    monkeypatch.setattr(module, "_rest_dependency_numbers", lambda number, repo, kind: ([], None))
    audit = audit_issue(4242, repo="ll7/robot_sf_ll7")

    assert audit.verdict == "drift"
    assert "parent_missing_native:9293" in audit.drift
    assert "blocked_by_missing_native:100" in audit.drift
    assert "blocked_by_missing_native:101" in audit.drift


def test_main_json_schema_shape(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The JSON payload carries the versioned schema and stable keys."""
    import scripts.dev.audit_issue_relationships as module

    monkeypatch.setattr(module, "_rest_issue_body", lambda number, repo: ("## Goal\n", None))
    assert main(["4242", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "issue_relationship_audit.v1"
    assert payload["verdict"] == "no_mirror_block"


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    return subprocess.run(
        ["uv", "run", "python", "scripts/dev/audit_issue_relationships.py", *argv],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_rejects_invalid_issue_number() -> None:
    """Non-positive issue numbers are usage errors before any network read."""
    proc = _run("0")
    assert proc.returncode == 2
