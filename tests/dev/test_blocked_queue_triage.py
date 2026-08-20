"""Contract tests for the fail-closed blocked-queue triage helper."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from subprocess import CompletedProcess
from typing import Any

import pytest

from scripts.dev import blocked_queue_triage as triage


def _issue(
    number: int = 42,
    *,
    title: str = "blocked issue",
    body: str = "Waiting for an external dataset.",
    labels: tuple[str, ...] = ("state:blocked", "resource:external-data"),
    updated_at: str = "2026-08-12T12:00:00Z",
) -> dict[str, Any]:
    """Build a minimal valid GitHub REST issue row."""

    return {
        "number": number,
        "title": title,
        "body": body,
        "html_url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "labels": [{"name": label} for label in labels],
        "created_at": "2026-08-01T12:00:00Z",
        "updated_at": updated_at,
    }


def test_classify_prefers_explicit_compute_label() -> None:
    """Scheduler labels must produce a high-confidence compute blocker."""

    result = triage._classify(
        _issue(body="A GPU allocation is unavailable.", labels=("state:blocked", "resource:slurm")),
        [],
    )

    assert result == ("compute", "high", ["label: resource:slurm"], ())


def test_classify_explicit_upstream_reference_precedes_licence_text() -> None:
    """A dependency reference must not be masked by incidental licence wording."""

    result = triage._classify(
        _issue(
            body="## Blocked by\n- #123 requires licence approval first.",
            labels=("state:blocked",),
        ),
        [],
    )

    assert result[0:2] == ("upstream_issue", "high")
    assert result[3] == (123,)


def test_classify_text_only_licence_signal_is_medium_confidence() -> None:
    """Text-only permission signals remain reviewable rather than high confidence."""

    result = triage._classify(
        _issue(body="Waiting for licence approval.", labels=("state:blocked",)),
        [],
    )

    assert result[0:2] == ("licence", "medium")


def test_classify_preserves_upstream_references() -> None:
    """Explicit upstream references must remain machine-checkable."""

    result = triage._classify(
        _issue(body="## Blocked by\n- #123 and #124 must land first.", labels=("state:blocked",)),
        [],
    )

    assert result[0:2] == ("upstream_issue", "high")
    assert result[3] == (123, 124)


def test_classify_uses_maintainer_gate_before_generic_external_signal() -> None:
    """A maintainer decision is not silently classified as an external fact."""

    result = triage._classify(
        _issue(
            body="A maintainer decision is required before proceeding.",
            labels=("state:blocked", "decision-required"),
        ),
        [],
    )

    assert result[0:2] == ("maintainer", "high")


def test_last_progress_ignores_automation_and_uses_human_comment() -> None:
    """Bot status comments must not masquerade as meaningful progress."""

    timestamp, source = triage._last_progress(
        _issue(updated_at="2026-08-12T12:00:00Z"),
        [
            {"user": {"login": "github-actions[bot]"}, "created_at": "2026-08-13T12:00:00Z"},
            {"user": {"login": "maintainer"}, "created_at": "2026-08-10T12:00:00Z"},
        ],
    )

    assert timestamp == "2026-08-10T12:00:00Z"
    assert source == "latest_human_comment"


def test_generated_triage_comment_is_not_signal_or_progress() -> None:
    """A prior bookkeeping comment must not alter later classification or age."""

    issue = _issue(number=42, body="Pending review.", labels=("state:blocked",))
    row = triage.build_report(
        [issue],
        {42: []},
        repo="owner/repo",
        label="state:blocked",
        generated_at="2026-08-13T12:00:00Z",
        next_check_at="weekly",
    )["issues"][0]
    comment = {"body": triage.render_comment(row)}

    result = triage._classify(issue, [comment])
    timestamp, source = triage._last_progress(issue, [comment])

    assert result[0:2] == ("external_fact", "low")
    assert timestamp == "2026-08-01T12:00:00Z"
    assert source == "issue_creation_fallback"


def test_build_report_counts_classes_modes_and_age_buckets() -> None:
    """The report exposes all required distribution dimensions."""

    issues = [
        _issue(number=1, labels=("state:blocked", "resource:slurm")),
        _issue(
            number=2,
            body="## Blocked by\n- #99",
            labels=("state:blocked",),
            updated_at="2026-05-01T12:00:00Z",
        ),
    ]
    report = triage.build_report(
        issues,
        {1: [], 2: []},
        repo="owner/repo",
        label="state:blocked",
        generated_at="2026-08-13T12:00:00Z",
        next_check_at="next workflow run",
    )

    assert report["schema"] == triage.SCHEMA
    assert report["source"]["pagination_complete"] is True
    assert report["counts"]["by_blocker_class"]["compute"] == 1
    assert report["counts"]["by_blocker_class"]["upstream_issue"] == 1
    assert report["counts"]["by_transition_class"]["compute_required"] == 1
    assert report["counts"]["by_transition_class"]["implementation_defect"] == 1
    assert report["counts"]["by_condition_mode"]["machine_testable"] == 1
    assert report["counts"]["by_progress_age"]["90_days_or_more"] == 1
    assert report["closure_candidates"] == []
    assert all(row["closure_recommendation"] == "keep_open" for row in report["issues"])


def test_render_comment_has_stable_marker_and_required_fields() -> None:
    """Published comments must be identifiable and contain the five fields."""

    row = triage.build_report(
        [_issue()],
        {42: []},
        repo="owner/repo",
        label="state:blocked",
        generated_at="2026-08-13T12:00:00Z",
        next_check_at="next workflow run",
    )["issues"][0]
    body = triage.render_comment(row)

    assert "<!-- blocked-queue-triage.v1 issue=42 digest=" in body
    assert "Blocker class:" in body
    assert "Unblock condition:" in body
    assert "Watcher:" in body
    assert "Next check:" in body
    assert "Last meaningful progress:" in body
    assert "Transition class: `external_input`" in body
    assert "Transition owner:" in body
    assert "keep open" in body


def test_fetch_blocked_issues_excludes_pull_requests_and_flattens_pages() -> None:
    """The inventory must not count pull-request rows as blocked issues."""

    calls: list[list[str]] = []

    def runner(args: list[str], _: str | None) -> CompletedProcess[str]:
        calls.append(args)
        return CompletedProcess(
            args,
            0,
            json.dumps([[_issue(number=1)], [{**_issue(number=2), "pull_request": {"url": "x"}}]]),
            "",
        )

    rows = triage._fetch_blocked_issues(
        repo="owner/repo", label="state:blocked", limit=10, runner=runner
    )

    assert [row["number"] for row in rows] == [1]
    assert calls[0][0:2] == ["api", "--paginate"]
    assert calls[0][2] == ("repos/owner/repo/issues?state=open&labels=state%3Ablocked&per_page=100")


def test_fetch_blocked_issues_rejects_invalid_issue_number() -> None:
    """The inventory boundary must reject rows before comment fetching."""

    def runner(args: list[str], _: str | None) -> CompletedProcess[str]:
        return CompletedProcess(args, 0, json.dumps([[{"title": "missing number"}]]), "")

    with pytest.raises(triage.TriageError, match="invalid number"):
        triage._fetch_blocked_issues(
            repo="owner/repo", label="state:blocked", limit=10, runner=runner
        )


def test_apply_comments_is_idempotent_and_verifies_writeback() -> None:
    """Repeated application must avoid duplicate comments and verify writes."""

    issue = _issue()
    report = triage.build_report(
        [issue],
        {42: []},
        repo="owner/repo",
        label="state:blocked",
        generated_at="2026-08-13T12:00:00Z",
        next_check_at="next workflow run",
    )
    calls: list[tuple[list[str], str | None]] = []

    def runner(args: list[str], input_text: str | None) -> CompletedProcess[str]:
        calls.append((args, input_text))
        payload = json.loads(input_text or "{}")
        return CompletedProcess(args, 0, json.dumps({"id": 9001, **payload}), "")

    first = triage.apply_comments(
        report["issues"],
        {42: []},
        repo="owner/repo",
        max_mutations=1,
        runner=runner,
    )
    assert first[0]["action"] == "created"
    assert calls[0][0][0:4] == ["api", "--method", "POST", "repos/owner/repo/issues/42/comments"]

    body = json.loads(calls[0][1] or "{}")["body"]
    unchanged = triage.apply_comments(
        report["issues"],
        {
            42: [
                {
                    "id": 9001,
                    "created_at": "2026-08-13T12:00:00Z",
                    "body": body + "\nGenerated at: `2026-08-13T12:00:00Z`",
                }
            ]
        },
        repo="owner/repo",
        max_mutations=0,
        runner=runner,
    )
    assert unchanged == [{"issue": 42, "action": "unchanged"}]


def test_apply_comments_retains_partial_operations_on_failure() -> None:
    """A failed write must preserve the partial mutation ledger for retry/audit."""

    rows = triage.build_report(
        [_issue(number=42), _issue(number=43)],
        {42: [], 43: []},
        repo="owner/repo",
        label="state:blocked",
        generated_at="2026-08-13T12:00:00Z",
        next_check_at="next workflow run",
    )["issues"]
    call_count = 0

    def runner(args: list[str], input_text: str | None) -> CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            payload = json.loads(input_text or "{}")
            return CompletedProcess(args, 0, json.dumps({"id": 9001, **payload}), "")
        return CompletedProcess(args, 1, "", "transient API error")

    with pytest.raises(triage.MutationError) as error:
        triage.apply_comments(
            rows,
            {42: [], 43: []},
            repo="owner/repo",
            max_mutations=2,
            runner=runner,
        )

    assert [item["action"] for item in error.value.operations] == ["created", "failed"]


def test_malformed_inventory_fails_closed() -> None:
    """A non-list REST payload cannot be treated as a complete inventory."""

    def runner(args: list[str], _: str | None) -> CompletedProcess[str]:
        return CompletedProcess(args, 0, json.dumps({"not": "a list"}), "")

    with pytest.raises(triage.TriageError, match="non-list"):
        triage._fetch_blocked_issues(
            repo="owner/repo", label="state:blocked", limit=10, runner=runner
        )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (CompletedProcess(["api"], 1, "", "permission denied"), "permission denied"),
        (CompletedProcess(["api"], 0, "not json", ""), "returned invalid JSON"),
    ],
)
def test_shared_transport_failures_translate_to_triage_errors(
    result: CompletedProcess[str], expected: str
) -> None:
    """The shared parser remains behind the established TriageError boundary."""
    with pytest.raises(triage.TriageError, match=expected):
        triage._json_result(result, operation="issue inventory")


def test_timestamp_parser_normalizes_utc() -> None:
    """Timestamp normalization must remain explicit and timezone-aware."""

    parsed = triage._parse_timestamp("2026-08-13T12:00:00+02:00")

    assert parsed == datetime(2026, 8, 13, 10, 0, tzinfo=UTC)


def test_report_only_mode_is_explicit_and_mutually_exclusive() -> None:
    """The watcher command must make its no-write mode executable and clear."""

    args = triage._build_parser().parse_args(["--report-only"])

    assert args.report_only is True
    assert args.apply_comments is False

    with pytest.raises(SystemExit):
        triage._build_parser().parse_args(["--report-only", "--apply-comments"])
