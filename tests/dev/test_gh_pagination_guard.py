"""Tests for the gh list truncation guard (issue #4991).

Covers the shared helper plus the two highest-risk evidence/reporting call sites:
``snapshot_pr_queue.snapshot_active_prs`` and
``autopilot_state_snapshot.issue_queue_snapshot``. A mocked ``gh ... list`` that
returns exactly ``limit`` rows must record a ``truncated: true`` marker (or raise);
returning fewer rows must pass cleanly.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev import autopilot_state_snapshot as snapshot
from scripts.dev._gh_pagination import (
    GhListTruncated,
    assert_not_truncated,
    is_likely_truncated,
)
from scripts.dev.snapshot_pr_queue import snapshot_active_prs

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def test_is_likely_truncated_true_when_row_count_equals_limit() -> None:
    """A result of exactly `limit` rows is ambiguous and treated as truncated."""
    assert is_likely_truncated(20, limit=20) is True


def test_is_likely_truncated_true_when_row_count_exceeds_limit() -> None:
    """More than `limit` rows (possible with dedup-free raw counts) is truncated."""
    assert is_likely_truncated(21, limit=20) is True


def test_is_likely_truncated_false_when_below_limit() -> None:
    """Fewer rows than the cap cannot have been truncated."""
    assert is_likely_truncated(19, limit=20) is False
    assert is_likely_truncated(0, limit=20) is False


def test_is_likely_truncated_false_for_non_positive_or_none_limit() -> None:
    """A non-positive or None limit means no cap was applied."""
    assert is_likely_truncated(0, limit=0) is False
    assert is_likely_truncated(100, limit=0) is False
    assert is_likely_truncated(100, limit=-1) is False
    assert is_likely_truncated(100, limit=None) is False


def test_assert_not_truncated_raises_when_rows_equal_limit() -> None:
    """Exactly `limit` rows must raise so callers fail closed."""
    rows = list(range(20))
    with pytest.raises(GhListTruncated) as exc_info:
        assert_not_truncated(rows, limit=20, context="gh pr list")
    message = str(exc_info.value)
    assert "20" in message
    assert "--limit 20" in message
    assert "gh pr list" in message


def test_assert_not_truncated_passes_when_below_limit() -> None:
    """Fewer rows than the cap pass cleanly with no raise."""
    assert_not_truncated(list(range(19)), limit=20)
    assert_not_truncated([], limit=20)


def test_assert_not_truncated_message_without_context() -> None:
    """A missing context still produces a clear, actionable message."""
    with pytest.raises(GhListTruncated) as exc_info:
        assert_not_truncated([1, 2, 3], limit=3)
    assert "raise --limit or paginate" in str(exc_info.value)


def test_assert_not_truncated_unbounded_limit_never_raises() -> None:
    """A non-positive or None limit is unbounded and never reports truncation."""
    assert_not_truncated(list(range(100)), limit=0)
    assert_not_truncated(list(range(100)), limit=-5)
    assert_not_truncated(list(range(100)), limit=None)


# ---------------------------------------------------------------------------
# snapshot_active_prs (gh pr list)
# ---------------------------------------------------------------------------


def _pr_payloads(count: int) -> list[dict]:
    """Return `count` minimal open-PR payloads."""
    return [
        {
            "number": 5000 + index,
            "title": f"PR {index}",
            "state": "OPEN",
            "isDraft": False,
            "url": f"https://github.test/pull/{5000 + index}",
            "labels": [],
            "headRefName": "feature",
            "headRefOid": f"abc{index:03d}",
            "mergeable": "MERGEABLE",
            "statusCheckRollup": [{"name": "ci", "status": "completed", "conclusion": "success"}],
            "reviews": [],
            "comments": [],
        }
        for index in range(count)
    ]


def test_snapshot_active_prs_records_truncated_when_at_limit() -> None:
    """Exactly `limit` rows from `gh pr list` record a `truncated: true` marker."""
    limit = 3
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(
            returncode=0, stdout=json.dumps(_pr_payloads(limit)), stderr=""
        )
        payload = snapshot_active_prs(repo="ll7/robot_sf_ll7", limit=limit)

    assert payload["truncated"] is True
    assert payload["truncation_note"]
    assert "--limit 3" in payload["truncation_note"]
    assert len(payload["prs"]) == limit


def test_snapshot_active_prs_clean_when_below_limit() -> None:
    """Fewer rows than the cap pass cleanly with a false truncated marker."""
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(
            returncode=0, stdout=json.dumps(_pr_payloads(2)), stderr=""
        )
        payload = snapshot_active_prs(repo="ll7/robot_sf_ll7", limit=5)

    assert payload["truncated"] is False
    assert payload["truncation_note"] == ""
    assert len(payload["prs"]) == 2


def test_snapshot_active_prs_clean_when_zero_rows() -> None:
    """An empty `gh pr list` result is not truncated."""
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps([]), stderr="")
        payload = snapshot_active_prs(repo="ll7/robot_sf_ll7", limit=20)

    assert payload["truncated"] is False
    assert payload["truncation_note"] == ""
    assert payload["prs"] == []


def test_snapshot_active_prs_error_path_reports_not_truncated() -> None:
    """A gh failure must not report truncation; it reports the error instead."""
    with patch("scripts.dev.snapshot_pr_queue._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=1, stdout="", stderr="rate limited")
        payload = snapshot_active_prs(repo="ll7/robot_sf_ll7", limit=20)

    assert payload["truncated"] is False
    assert payload["truncation_note"] == ""
    assert payload["prs"][0]["status"] == "error"


# ---------------------------------------------------------------------------
# issue_queue_snapshot (gh issue list)
# ---------------------------------------------------------------------------


def _issue_payloads(count: int) -> list[dict]:
    """Return `count` minimal issue payloads."""
    return [
        {
            "number": 4900 + index,
            "title": f"Issue {index}",
            "state": "OPEN",
            "labels": [{"name": "agent"}],
            "updatedAt": "2026-07-10T00:00:00Z",
            "url": f"https://github.test/issues/{4900 + index}",
        }
        for index in range(count)
    ]


def test_issue_queue_snapshot_records_truncation_when_at_limit() -> None:
    """Exactly `limit` rows per search record a `truncated: true` marker."""
    limit = 4
    with patch("scripts.dev.autopilot_state_snapshot._run") as mock_run:
        mock_run.return_value = snapshot.CommandResult(
            command=("gh", "issue", "list"),
            returncode=0,
            stdout=json.dumps(_issue_payloads(limit)),
            stderr="",
        )
        rows, _sources, errors, truncations = snapshot.issue_queue_snapshot(
            ["is:issue is:open"], limit=limit
        )

    assert errors == []
    assert len(rows) == limit
    assert truncations
    marker = truncations[0]
    assert marker["truncated"] is True
    assert marker["row_count"] == limit
    assert marker["limit"] == limit
    assert "--limit 4" in marker["note"]


def test_issue_queue_snapshot_clean_when_below_limit() -> None:
    """Fewer rows than the cap pass cleanly with truncated markers set False."""
    with patch("scripts.dev.autopilot_state_snapshot._run") as mock_run:
        mock_run.return_value = snapshot.CommandResult(
            command=("gh", "issue", "list"),
            returncode=0,
            stdout=json.dumps(_issue_payloads(2)),
            stderr="",
        )
        rows, _sources, errors, truncations = snapshot.issue_queue_snapshot(
            ["is:issue is:open"], limit=20
        )

    assert errors == []
    assert len(rows) == 2
    assert truncations[0]["truncated"] is False
    assert truncations[0]["row_count"] == 2
    assert truncations[0]["note"] == ""


def test_issue_queue_snapshot_per_search_truncation_is_independent() -> None:
    """A truncated search and a clean search produce independent markers."""
    with patch("scripts.dev.autopilot_state_snapshot._run") as mock_run:
        mock_run.side_effect = [
            snapshot.CommandResult(
                command=("gh", "issue", "list"),
                returncode=0,
                stdout=json.dumps(_issue_payloads(3)),
                stderr="",
            ),
            snapshot.CommandResult(
                command=("gh", "issue", "list"),
                returncode=0,
                stdout=json.dumps(_issue_payloads(1)),
                stderr="",
            ),
        ]
        _rows, _sources, _errors, truncations = snapshot.issue_queue_snapshot(
            ["full:truncated", "sparse:clean"], limit=3
        )

    assert len(truncations) == 2
    assert truncations[0] == {
        "search": "full:truncated",
        "truncated": True,
        "row_count": 3,
        "limit": 3,
        "note": ("gh issue list may be capped: got 3 rows at --limit 3; raise --limit or paginate"),
    }
    assert truncations[1] == {
        "search": "sparse:clean",
        "truncated": False,
        "row_count": 1,
        "limit": 3,
        "note": "",
    }


def test_issue_queue_snapshot_no_truncation_marker_for_failed_search() -> None:
    """A failed gh call must not emit a truncation marker; it records an error."""
    with patch("scripts.dev.autopilot_state_snapshot._run") as mock_run:
        mock_run.return_value = snapshot.CommandResult(
            command=("gh", "issue", "list"),
            returncode=1,
            stdout="",
            stderr="not found",
        )
        _rows, _sources, errors, truncations = snapshot.issue_queue_snapshot(
            ["is:issue is:open"], limit=20
        )

    assert truncations == []
    assert errors and "not found" in errors[0]


def test_issue_queue_snapshot_rejects_non_list_json() -> None:
    """A successful but malformed gh JSON response must not silently omit issues."""
    with patch("scripts.dev.autopilot_state_snapshot._run") as mock_run:
        mock_run.return_value = snapshot.CommandResult(
            command=("gh", "issue", "list"),
            returncode=0,
            stdout=json.dumps({"number": 1}),
            stderr="",
        )
        rows, _sources, errors, truncations = snapshot.issue_queue_snapshot(
            ["is:issue is:open"], limit=20
        )

    assert rows == []
    assert truncations == []
    assert errors == ["issue search 'is:issue is:open': expected JSON array, got dict"]


def test_build_snapshot_surfaces_issues_truncated_markers(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The full snapshot payload must expose issue-search truncation markers."""
    captured: dict[str, object] = {}

    def fake_issue_queue_snapshot(searches, *, limit):  # type: ignore[no-untyped-def]
        captured["searches"] = searches
        captured["limit"] = limit
        marker = {
            "search": searches[0],
            "truncated": True,
            "row_count": limit,
            "limit": limit,
            "note": "gh issue list may be capped",
        }
        return [], [], [], [marker]

    monkeypatch.setattr(snapshot, "issue_queue_snapshot", fake_issue_queue_snapshot)
    # Short-circuit the git/claim/pr collection so the test stays focused.
    monkeypatch.setattr(
        snapshot,
        "git_snapshot",
        lambda **_kw: ({}, [], []),
    )
    monkeypatch.setattr(
        snapshot,
        "claim_snapshot",
        lambda *_a, **_kw: ([], [], []),
    )
    monkeypatch.setattr(
        snapshot,
        "pr_snapshot",
        lambda *_a, **_kw: ([], [], []),
    )

    args = snapshot._build_parser().parse_args(["--issue-search", "is:issue is:open"])
    payload = snapshot.build_snapshot(args)

    assert payload["issues_truncated_any"] is True
    assert payload["issues_truncated"][0]["truncated"] is True
    assert captured["limit"] == 20
