"""Tests for the blocker-staleness issue audit helper."""

from __future__ import annotations

import json

from scripts.tools import check_blocker_staleness as staleness


def test_extract_blocker_numbers_from_dependency_sections() -> None:
    """The parser should only read numbered blockers from recognized sections."""

    body = """## Goal / Problem

Mention #10 outside the dependency section.

## Blocked-By

- #101
- ll7/robot_sf_ll7#102
- ~~#103~~
- duplicate #101

### Nested detail

- #107

## Depends On:

Waiting for #104 and #105.

## Blocked by #108

Inline heading reference.

## Validation / Testing

This section mentions #106 but is not a blocker.
"""

    assert staleness.extract_blocker_numbers(body) == (101, 102, 107, 104, 105, 108)


def test_extract_blocker_numbers_ignores_other_repository_references() -> None:
    """Cross-repository refs should not be resolved against the selected repository."""

    body = """## Depends On #201

- ll7/robot_sf_ll7#202
- other/repo#203
- #204
"""

    assert staleness.extract_blocker_numbers(body, repo="ll7/robot_sf_ll7") == (201, 202, 204)


def test_classify_blockers_reports_fully_partially_and_still_blocked() -> None:
    """Resolved issue states should map to the expected stale-blocker buckets."""

    issue = staleness.IssueRef(number=200, title="candidate")

    fully = staleness.classify_blockers(issue, (101, 102), {101: "CLOSED", 102: "CLOSED"})
    partial = staleness.classify_blockers(issue, (101, 102), {101: "CLOSED", 102: "OPEN"})
    still = staleness.classify_blockers(issue, (101, 102), {101: "OPEN", 102: "OPEN"})

    assert fully.classification == staleness.BlockerClassification.FULLY_UNBLOCKED
    assert fully.closed_blockers == (101, 102)
    assert partial.classification == staleness.BlockerClassification.PARTIALLY_UNBLOCKED
    assert partial.closed_blockers == (101,)
    assert partial.open_blockers == (102,)
    assert still.classification == staleness.BlockerClassification.STILL_BLOCKED


def test_classify_blockers_keeps_unknown_states_separate() -> None:
    """Unknown blocker state should not be reported as safely unblocked."""

    result = staleness.classify_blockers(
        staleness.IssueRef(number=200),
        (101, 102),
        {101: "CLOSED"},
    )

    assert result.classification == staleness.BlockerClassification.UNKNOWN
    assert result.closed_blockers == (101,)
    assert result.unknown_blockers == (102,)


def test_audit_issue_bodies_omits_issues_without_numbered_blockers() -> None:
    """Issues without parsed numbered blockers are not stale-blocker candidates."""

    issues = [
        staleness.IssueRef(number=200, title="has blockers"),
        staleness.IssueRef(number=201, title="no blockers"),
    ]
    bodies = {
        200: "## Blocked by\n\n- #101\n- #102\n",
        201: "## Blocked by\n\n- external data upload\n",
    }

    results = staleness.audit_issue_bodies(issues, bodies, {101: "CLOSED", 102: "OPEN"})

    assert [result.issue.number for result in results] == [200]
    assert results[0].classification == staleness.BlockerClassification.PARTIALLY_UNBLOCKED


def test_fetch_issue_states_keeps_missing_blockers_unknown(monkeypatch) -> None:
    """A missing/private blocker should not abort the whole stale-blocker report."""

    def fake_gh_json(args: list[str]):
        number = int(args[2])
        if number == 101:
            return {"number": 101, "state": "CLOSED"}
        raise RuntimeError("not found")

    monkeypatch.setattr(staleness, "_gh_json", fake_gh_json)

    assert staleness.fetch_issue_states({101, 102}, repo="owner/repo") == {
        101: "CLOSED",
        102: "UNKNOWN",
    }


def test_main_exits_nonzero_only_for_enforced_fully_unblocked(
    monkeypatch,
    capsys,
) -> None:
    """The default CLI enforces fully-unblocked findings, while report-only stays zero."""

    def fake_fetch_open_issues(*, repo: str, label: str | None, limit: int):
        assert repo == "owner/repo"
        assert label == "state:blocked"
        assert limit == 10
        return [staleness.IssueRef(number=200, title="ready")], {200: "## Blocked by\n\n- #101\n"}

    monkeypatch.setattr(staleness, "fetch_open_issues", fake_fetch_open_issues)
    monkeypatch.setattr(staleness, "fetch_issue_states", lambda numbers, *, repo: {101: "CLOSED"})

    exit_code = staleness.main(
        ["--repo", "owner/repo", "--label", "state:blocked", "--limit", "10"]
    )
    assert exit_code == 1
    assert "fully_unblocked (1)" in capsys.readouterr().out

    exit_code = staleness.main(
        [
            "--repo",
            "owner/repo",
            "--label",
            "state:blocked",
            "--limit",
            "10",
            "--report-only",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "blocker_staleness_report.v1"
    assert set(payload["results"]) == {
        "fully_unblocked",
        "partially_unblocked",
        "still_blocked",
        "unknown",
    }
    assert payload["counts"]["fully_unblocked"] == 1
