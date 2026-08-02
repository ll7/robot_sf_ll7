"""Tests read-only REST-first open-issue closure audit helpers (issue #6610).

The audit no longer touches GitHub's separately-rate-limited search endpoint. It
reads open issues and merged pull requests through bounded paginated REST passes,
builds the title-to-issue index locally, and reports deterministic
truncation/partial-inventory metadata. These tests lock that contract.
"""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from scripts.dev import open_issue_closure_audit
from scripts.dev.open_issue_closure_audit import PaginationMeta

# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _rest_issue(number: int, title: str, *, state: str = "open") -> dict[str, object]:
    """Raw REST issue payload shape (html_url + lowercase state)."""
    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "state": state,
    }


def _rest_pr(
    number: int,
    title: str,
    *,
    merged: bool = True,
    state: str = "closed",
) -> dict[str, object]:
    """Raw REST pull payload shape as returned by the list pulls endpoint.

    The list endpoint does NOT populate a ``merged`` boolean (it returns
    ``null``); it only exposes ``merged_at`` (null for closed-unmerged PRs). The
    audit must therefore infer merged status from ``merged_at``.
    """
    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/ll7/robot_sf_ll7/pull/{number}",
        "state": state,
        "merged": None,
        "merged_at": "2026-07-04T11:00:00Z" if merged else None,
        "closed_at": "2026-07-04T11:00:00Z",
    }


def _meta(
    *,
    pages_read: int,
    per_page: int,
    page_budget: int,
    row_count: int,
    truncated: bool,
) -> PaginationMeta:
    """Build a PaginationMeta fixture."""
    return PaginationMeta(
        pages_read=pages_read,
        per_page=per_page,
        page_budget=page_budget,
        row_count=row_count,
        truncated=truncated,
    )


def _completed(
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Build a CompletedProcess matching the shape _gh_api_get returns."""
    return subprocess.CompletedProcess(
        args=("gh", "api"), returncode=returncode, stdout=stdout, stderr=stderr
    )


def _page_factory(pages: list[list[dict[str, object]]]) -> list[list[str]]:
    """Capture the raw args of every subprocess.run call for later inspection."""
    captured: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(list(args))
        # ``args == ["gh", "api", <path>]``; match ``&page=N`` (anchored on
        # ``&``) so ``per_page=`` is not misread as the page index.
        match = re.search(r"&page=(\d+)", args[2])
        page = int(match.group(1)) if match else 1
        rows = pages[page - 1] if 0 < page <= len(pages) else []
        return _completed(json.dumps(rows))

    fake_run.captured = captured  # type: ignore[attr-defined]
    return fake_run  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Classification + collect semantics (behavior-preserving)
# ---------------------------------------------------------------------------


def test_collect_candidates_reports_open_issues_with_merged_title_linked_prs() -> None:
    """Open issues with merged title-linked PRs become review candidates."""
    index = open_issue_closure_audit.build_title_linked_index(
        [open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(99, "fix #12 checker"))],
        [12, 13],
    )
    candidates = open_issue_closure_audit.collect_candidates(
        [
            open_issue_closure_audit._normalize_rest_issue_row(_rest_issue(12, "add checker")),
            open_issue_closure_audit._normalize_rest_issue_row(_rest_issue(13, "no coverage yet")),
        ],
        index,
    )

    assert [candidate.number for candidate in candidates] == [12]
    assert candidates[0].classification == "closure_review_required"
    assert candidates[0].title_linked_prs[0].number == 99
    assert "acceptance_criteria" in candidates[0].recommended_action


def test_collect_candidates_treats_null_fields_as_empty_not_none_string() -> None:
    """Explicit JSON nulls must not coerce to the literal string ``"None"``."""
    issue_row: dict[str, object] = {
        "number": 20,
        "title": None,
        "url": "https://github.com/ll7/robot_sf_ll7/issues/20",
        "state": "open",
    }
    pr_row: dict[str, object] = {
        "number": 200,
        "title": "fix #20 handler",
        "url": "https://github.com/ll7/robot_sf_ll7/pull/200",
        "state": "merged",
        "closedAt": "2026-07-04T11:00:00Z",
        "mergedAt": None,
    }

    candidates = open_issue_closure_audit.collect_candidates([issue_row], {20: [pr_row]})

    assert len(candidates) == 1
    assert candidates[0].title == ""
    assert candidates[0].title_linked_prs[0].merged_at == "2026-07-04T11:00:00Z"


def test_collect_candidates_classifies_parent_roadmap_without_closure() -> None:
    """Parent or roadmap issues get ledger-update guidance, not closure guidance."""
    index = open_issue_closure_audit.build_title_linked_index(
        [open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(4000, "issue #3481 slice one"))],
        [3481],
    )
    candidates = open_issue_closure_audit.collect_candidates(
        [
            open_issue_closure_audit._normalize_rest_issue_row(
                _rest_issue(3481, "roadmap: multi-slice parent")
            )
        ],
        index,
    )

    assert candidates[0].classification == "parent_or_roadmap"
    assert candidates[0].recommended_action == (
        "update_status_ledger_with_merged_slices_and_remaining_work"
    )


def test_collect_candidates_ignores_closed_issues_pr_rows_and_unlinked_titles() -> None:
    """The audit stays issue-specific and requires explicit title linkage."""
    rows = [
        open_issue_closure_audit._normalize_rest_issue_row(
            _rest_issue(12, "closed issue", state="closed")
        ),
        {
            "number": 14,
            "title": "pull request shaped row",
            "url": "https://github.com/ll7/robot_sf_ll7/pull/14",
            "state": "open",
        },
        open_issue_closure_audit._normalize_rest_issue_row(_rest_issue(15, "open issue")),
    ]
    pr_rows_by_issue = {
        12: [open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(91, "fix #12"))],
        14: [open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(92, "fix #14"))],
        15: [open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(93, "fix #150 not fifteen"))],
    }

    assert open_issue_closure_audit.collect_candidates(rows, pr_rows_by_issue) == []


def test_build_report_emits_read_only_failure_summary() -> None:
    """Candidate reports expose stable counts and no-write guarantees."""
    index = open_issue_closure_audit.build_title_linked_index(
        [
            open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(91, "fix #12")),
            open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(92, "fix #13")),
        ],
        [12, 13],
    )
    candidates = open_issue_closure_audit.collect_candidates(
        [
            open_issue_closure_audit._normalize_rest_issue_row(
                _rest_issue(12, "tracking parent issue")
            ),
            open_issue_closure_audit._normalize_rest_issue_row(_rest_issue(13, "simple issue")),
        ],
        index,
    )
    report = open_issue_closure_audit.build_report(repo="ll7/robot_sf_ll7", candidates=candidates)

    assert report["schema"] == "open_issue_closure_audit.v1"
    assert report["ok"] is False
    assert report["read_only"] is True
    assert report["issue_writes"] is False
    assert report["project_writes"] is False
    assert report["candidate_count"] == 2
    assert report["parent_or_roadmap_count"] == 1
    assert report["closure_review_count"] == 1
    assert report["failure_summary"]["reason"] == "open_issues_with_merged_title_linked_prs"


# ---------------------------------------------------------------------------
# REST endpoint targeting (no search quota dependency)
# ---------------------------------------------------------------------------


def test_rest_reads_target_repos_endpoint_not_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open-issue and closed-PR reads go through ``gh api repos/...`` only."""
    fake = _page_factory([[_rest_issue(12, "x")]])

    def fake_pr_pages(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        # PR inventory is a single bounded pass; return one partial page so it stops.
        return _completed(json.dumps([_rest_pr(99, "fix #12")]))

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake)
    open_issue_closure_audit.fetch_open_issue_rows(
        repo="ll7/robot_sf_ll7", max_pages=1, per_page=100
    )
    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake_pr_pages)
    pr_rows, _ = open_issue_closure_audit.fetch_closed_pr_rows(
        repo="ll7/robot_sf_ll7", max_pages=1, per_page=100
    )

    issue_args = fake.captured[-1]  # type: ignore[attr-defined]
    assert issue_args[:2] == ["gh", "api"]
    issue_path = issue_args[2]
    assert "search" not in issue_path
    assert issue_path.startswith("repos/ll7/robot_sf_ll7/issues?state=open")
    assert "per_page=100" in issue_path and "&page=1" in issue_path

    assert pr_rows[0]["number"] == 99
    # No search subcommand is ever emitted by the REST helpers.
    assert all("search" not in arg for arg in [" ".join(issue_args[:2])])


def test_no_search_subcommand_anywhere_in_command_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every gh invocation the audit produces is a REST ``gh api`` read."""
    seen: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(list(args))
        return _completed("[]")

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake_run)
    open_issue_closure_audit.fetch_open_issue_rows(repo="ll7/robot_sf_ll7", max_pages=1, per_page=5)
    open_issue_closure_audit.fetch_closed_pr_rows(repo="ll7/robot_sf_ll7", max_pages=1, per_page=5)

    for args in seen:
        assert args[:2] == ["gh", "api"]
        assert args[2].startswith("repos/ll7/robot_sf_ll7/")
        assert "search" not in args[2]


# ---------------------------------------------------------------------------
# REST pagination: complete vs truncated/partial inventory
# ---------------------------------------------------------------------------


def test_fetch_open_issue_rows_paginates_until_partial_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short final page stops pagination and marks the inventory complete."""
    pages = [
        [_rest_issue(1, "a"), _rest_issue(2, "b")],
        [_rest_issue(3, "c"), _rest_issue(4, "d")],
        [_rest_issue(5, "e")],  # partial page -> definitive end
    ]
    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", _page_factory(pages))
    rows, meta = open_issue_closure_audit.fetch_open_issue_rows(
        repo="ll7/robot_sf_ll7", max_pages=5, per_page=2
    )

    assert [row["number"] for row in rows] == [1, 2, 3, 4, 5]
    assert meta.pages_read == 3
    assert meta.page_budget == 5
    assert meta.row_count == 5
    assert meta.truncated is False


def test_fetch_open_issue_rows_flags_truncation_at_page_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consuming the full page budget with every page full is a partial inventory."""
    pages = [[_rest_issue(1, "a"), _rest_issue(2, "b")], [_rest_issue(3, "c"), _rest_issue(4, "d")]]
    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", _page_factory(pages))
    rows, meta = open_issue_closure_audit.fetch_open_issue_rows(
        repo="ll7/robot_sf_ll7", max_pages=2, per_page=2
    )

    assert meta.pages_read == 2
    assert meta.row_count == 4
    assert meta.truncated is True
    assert [row["number"] for row in rows] == [1, 2, 3, 4]


def test_fetch_closed_pr_rows_uses_state_closed_single_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The merged-PR inventory is one bounded ``pulls?state=closed`` pass."""
    # A partial first page (2 rows < per_page=3) is a definitive end signal, so
    # pagination stops after a single read.
    fake = _page_factory([[_rest_pr(7, "fix #1"), _rest_pr(8, "fix #2")]])
    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake)
    rows, meta = open_issue_closure_audit.fetch_closed_pr_rows(
        repo="ll7/robot_sf_ll7", max_pages=3, per_page=3
    )

    assert meta.pages_read == 1
    assert meta.truncated is False
    assert [row["number"] for row in rows] == [7, 8]
    pr_path = fake.captured[0][2]  # type: ignore[attr-defined]
    assert pr_path.startswith("repos/ll7/robot_sf_ll7/pulls?state=closed")


# ---------------------------------------------------------------------------
# Merged-only filtering + local title-to-issue index
# ---------------------------------------------------------------------------


def test_merged_only_filtering_drops_unmerged_closed_prs() -> None:
    """Closed-unmerged PRs are excluded; merged PRs keep their merge timestamp."""
    # The list pulls endpoint returns ``merged: null`` for every row, so merged
    # status is inferred from a non-null ``merged_at`` (issue #6610 regression).
    merged = open_issue_closure_audit._normalize_rest_pr_row(
        _rest_pr(10, "fix #12", merged=True, state="closed")
    )
    unmerged = open_issue_closure_audit._normalize_rest_pr_row(
        _rest_pr(11, "fix #12", merged=False, state="closed")
    )

    assert open_issue_closure_audit._is_merged_pr_row(_rest_pr(10, "x", merged=True)) is True
    assert open_issue_closure_audit._is_merged_pr_row(_rest_pr(11, "x", merged=False)) is False
    assert merged["state"] == "merged"
    assert merged["mergedAt"] == "2026-07-04T11:00:00Z"
    assert unmerged["state"] == "closed"


def test_is_merged_pr_row_handles_list_endpoint_null_merged_boolean() -> None:
    """The list pulls endpoint returns ``merged: null``; ``merged_at`` decides."""
    list_merged = {"merged": None, "merged_at": "2026-08-02T10:40:15Z"}
    list_unmerged = {"merged": None, "merged_at": None}
    single_pr_merged = {"merged": True, "merged_at": None}
    single_pr_unmerged = {"merged": False, "merged_at": "2026-08-02T10:40:15Z"}

    assert open_issue_closure_audit._is_merged_pr_row(list_merged) is True
    assert open_issue_closure_audit._is_merged_pr_row(list_unmerged) is False
    # An explicit ``merged`` boolean (single-PR endpoint) wins over ``merged_at``.
    assert open_issue_closure_audit._is_merged_pr_row(single_pr_merged) is True
    assert open_issue_closure_audit._is_merged_pr_row(single_pr_unmerged) is False


def test_build_title_linked_index_matches_titles_locally() -> None:
    """One merged-PR pass is matched against every open issue number in-process."""
    merged_rows = [
        open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(91, "fix #12 contract")),
        open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(92, "cover #13 and #14 slices")),
        open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(93, "no issue reference")),
    ]
    index = open_issue_closure_audit.build_title_linked_index(merged_rows, [12, 13, 14])

    assert [pr["number"] for pr in index[12]] == [91]
    assert [pr["number"] for pr in index[13]] == [92]
    assert [pr["number"] for pr in index[14]] == [92]
    # Numbers not mentioned produce no index rows; an issue with no PR is absent.
    assert open_issue_closure_audit.build_title_linked_index([], [12, 13]) == {12: [], 13: []}


def test_build_title_linked_index_respects_digit_boundaries() -> None:
    """``#12`` must not match issue 1, 2, or 120; only standalone 12."""
    merged_rows = [
        open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(1, "fix #120")),
        open_issue_closure_audit._normalize_rest_pr_row(_rest_pr(2, "fix #12")),
    ]
    index = open_issue_closure_audit.build_title_linked_index(merged_rows, [1, 12, 120])

    assert [pr["number"] for pr in index[12]] == [2]
    assert [pr["number"] for pr in index[120]] == [1]
    assert index[1] == []


# ---------------------------------------------------------------------------
# CLI: complete vs partial inventory exit codes
# ---------------------------------------------------------------------------


def test_main_complete_inventory_exits_nonzero_only_for_candidates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A complete inventory with no candidates exits 0; with candidates exits 1."""
    complete_meta = _meta(pages_read=1, per_page=100, page_budget=10, row_count=1, truncated=False)

    def fake_issues(*, repo: str, max_pages: int) -> tuple[list[dict[str, object]], PaginationMeta]:
        assert (repo, max_pages) == ("ll7/robot_sf_ll7", 10)
        return ([_rest_issue(12, "simple issue")], complete_meta)

    def fake_prs(*, repo: str, max_pages: int) -> tuple[list[dict[str, object]], PaginationMeta]:
        assert (repo, max_pages) == ("ll7/robot_sf_ll7", 20)
        return ([_rest_pr(91, "issue #12 done")], complete_meta)

    monkeypatch.setattr(open_issue_closure_audit, "fetch_open_issue_rows", fake_issues)
    monkeypatch.setattr(open_issue_closure_audit, "fetch_closed_pr_rows", fake_prs)

    exit_code = open_issue_closure_audit.main(
        ["--repo", "ll7/robot_sf_ll7", "--max-issue-pages", "10", "--max-pr-pages", "20"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["number"] == 12
    assert payload["truncated_any"] is False
    assert payload["truncations"]["merged_prs"]["truncated"] is False


def test_main_complete_inventory_no_candidates_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A complete inventory with zero candidates is the only success exit."""
    complete_meta = _meta(pages_read=1, per_page=100, page_budget=10, row_count=1, truncated=False)

    monkeypatch.setattr(
        open_issue_closure_audit,
        "fetch_open_issue_rows",
        lambda *, repo, max_pages: ([_rest_issue(12, "no coverage")], complete_meta),
    )
    monkeypatch.setattr(
        open_issue_closure_audit,
        "fetch_closed_pr_rows",
        lambda *, repo, max_pages: ([_rest_pr(91, "unrelated title")], complete_meta),
    )

    exit_code = open_issue_closure_audit.main(["--repo", "ll7/robot_sf_ll7"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["candidate_count"] == 0
    assert payload["truncated_any"] is False


def test_main_partial_inventory_surfaces_truncation_and_nonsuccess_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partial REST inventory exits non-zero even with no candidates."""
    truncated_meta = _meta(pages_read=2, per_page=100, page_budget=2, row_count=200, truncated=True)

    monkeypatch.setattr(
        open_issue_closure_audit,
        "fetch_open_issue_rows",
        lambda *, repo, max_pages: ([_rest_issue(12, "open issue")], truncated_meta),
    )
    monkeypatch.setattr(
        open_issue_closure_audit,
        "fetch_closed_pr_rows",
        lambda *, repo, max_pages: ([_rest_pr(91, "no link")], truncated_meta),
    )

    exit_code = open_issue_closure_audit.main(["--repo", "ll7/robot_sf_ll7"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1  # non-success: inventory partial
    assert payload["ok"] is True  # no candidates, but inventory incomplete
    assert payload["candidate_count"] == 0
    assert payload["truncated_any"] is True
    assert payload["truncations"]["open_issues"]["truncated"] is True
    assert payload["truncations"]["open_issues"]["page_budget"] == 2
    assert payload["truncations"]["open_issues"]["pages_read"] == 2
    assert payload["truncations"]["open_issues"]["row_count"] == 200
    assert "raise --max-issue-pages" in payload["truncations"]["open_issues"]["note"]
    assert payload["truncations"]["merged_prs"]["truncated"] is True
    assert payload["truncations"]["merged_prs"]["page_budget"] == 2


def test_main_filters_unmerged_prs_and_builds_index_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Closed-unmerged PRs are dropped; the PR inventory is fetched exactly once."""
    complete_meta = _meta(pages_read=1, per_page=100, page_budget=20, row_count=3, truncated=False)
    pr_calls = {"count": 0}

    def fake_prs(*, repo: str, max_pages: int) -> tuple[list[dict[str, object]], PaginationMeta]:
        pr_calls["count"] += 1
        # A merged PR linking issue 12, a closed-unmerged PR also mentioning 12,
        # and a merged PR with no issue reference.
        return (
            [
                _rest_pr(91, "fix #12", merged=True),
                _rest_pr(92, "fix #12", merged=False),
                _rest_pr(93, "no ref", merged=True),
            ],
            complete_meta,
        )

    monkeypatch.setattr(
        open_issue_closure_audit,
        "fetch_open_issue_rows",
        lambda *, repo, max_pages: ([_rest_issue(12, "add checker")], complete_meta),
    )
    monkeypatch.setattr(open_issue_closure_audit, "fetch_closed_pr_rows", fake_prs)

    exit_code = open_issue_closure_audit.main(["--repo", "ll7/robot_sf_ll7"])
    payload = json.loads(capsys.readouterr().out)

    assert pr_calls["count"] == 1  # single bounded PR pass, not one per issue
    assert exit_code == 1
    assert payload["candidate_count"] == 1
    linked = payload["candidates"][0]["title_linked_prs"]
    assert [pr["number"] for pr in linked] == [91]  # unmerged #92 dropped


# ---------------------------------------------------------------------------
# REST error handling (fail closed)
# ---------------------------------------------------------------------------


def test_gh_api_get_reports_missing_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing GitHub CLI raises an actionable runtime error."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gh")

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GitHub CLI 'gh' was not found"):
        open_issue_closure_audit._gh_api_get("repos/ll7/robot_sf_ll7/issues")


def test_main_reports_rest_timeout_as_schema_valid_packet_exit_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bounded REST timeout fails closed with the standard error packet."""

    def timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=30)

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", timeout)

    exit_code = open_issue_closure_audit.main(["--repo", "ll7/robot_sf_ll7"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["schema"] == "open_issue_closure_audit.v1"
    assert payload["read_only"] is True
    assert payload["issue_writes"] is False
    assert payload["project_writes"] is False
    assert "GitHub REST read timed out after 30s" in payload["error"]


def test_paginate_rest_reports_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed REST read fails closed instead of silently returning partial data."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed("", returncode=1, stderr="HTTP 403: rate limit exceeded")

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GitHub REST read failed"):
        open_issue_closure_audit._paginate_rest(
            "repos/x/issues?state=open", max_pages=2, per_page=10
        )


def test_paginate_rest_reports_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed REST output is diagnosed before aggregation."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed("{not-json")

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="Invalid JSON"):
        open_issue_closure_audit._paginate_rest(
            "repos/x/issues?state=open", max_pages=1, per_page=10
        )


def test_paginate_rest_rejects_non_object_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A list containing a malformed row fails closed instead of looking complete."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(json.dumps([{"number": 1}, "malformed-row"]))

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="Expected JSON list of objects"):
        open_issue_closure_audit._paginate_rest(
            "repos/x/issues?state=open", max_pages=1, per_page=10
        )


def test_main_reports_rest_error_as_schema_valid_packet_exit_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A REST failure still emits the read-only schema packet and exits 2."""

    def boom(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gh")

    monkeypatch.setattr(open_issue_closure_audit.subprocess, "run", boom)

    exit_code = open_issue_closure_audit.main(["--repo", "ll7/robot_sf_ll7"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["schema"] == "open_issue_closure_audit.v1"
    assert payload["read_only"] is True
    assert payload["issue_writes"] is False
    assert payload["project_writes"] is False
    assert "GitHub CLI 'gh' was not found" in payload["error"]


def test_main_rejects_nonpositive_page_budget(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A zero/negative page budget fails closed with a schema-valid packet."""
    exit_code = open_issue_closure_audit.main(
        ["--repo", "ll7/robot_sf_ll7", "--max-issue-pages", "0"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["read_only"] is True
    assert "page budgets must be >= 1" in payload["error"]
