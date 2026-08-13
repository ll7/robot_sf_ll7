"""Contract tests for the fail-closed blocked queue watcher."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from scripts.dev.blocked_queue_watcher import (
    Evaluation,
    IssueCandidate,
    apply_fired,
    build_report,
    evaluate_candidates,
    parse_triage_record,
)


def _comment(
    condition: str, *, url: str = "https://github.com/ll7/robot_sf_ll7/issues/999#comment"
) -> dict[str, str]:
    return {
        "url": url,
        "body": f"""<!-- blocked-triage-v1 tracking=7067 -->
```yaml
blocker_class: upstream_issue
unblock_condition: >-
  {condition}
watcher: >-
  gh issue view 6710 --json state
next_check_at: next queue pass
last_meaningful_progress_at: "2026-08-13"
```
""",
    }


def _candidate(
    number: int = 999,
    *,
    condition: str = "#6710 is closed",
    labels: tuple[str, ...] = ("state:blocked",),
    comments: tuple[dict[str, str], ...] | None = None,
) -> IssueCandidate:
    return IssueCandidate(
        number=number,
        title="blocked fixture",
        body="",
        labels=labels,
        url=f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        comments=comments if comments is not None else (_comment(condition),),
    )


def _graphql_result(*, item: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "data": {
            "repository": {
                "item_1": item,
            }
        }
    }


def test_parse_triage_record_accepts_latest_complete_comment() -> None:
    outcome = parse_triage_record(_candidate())

    assert outcome.status == "ok"
    assert outcome.record is not None
    assert outcome.record.blocker_class == "upstream_issue"
    assert outcome.record.unblock_condition == "#6710 is closed"


@pytest.mark.parametrize(
    ("comments", "reason"),
    [
        ((), "marker is absent"),
        (({"body": "<!-- blocked-triage-v1 tracking=7067 -->"},), "no YAML fence"),
        (
            (
                {
                    "body": "<!-- blocked-triage-v1 tracking=7067 -->\n```yaml\nblocker_class: dependency\n```"
                },
            ),
            "missing",
        ),
    ],
)
def test_parse_triage_record_is_fail_closed(
    comments: tuple[dict[str, str], ...], reason: str
) -> None:
    outcome = parse_triage_record(_candidate(comments=comments))

    assert outcome.status in {"missing", "malformed"}
    assert reason in outcome.reason


def test_evaluate_candidates_uses_one_batched_graphql_request() -> None:
    calls: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            __import__("json").dumps(
                _graphql_result(
                    item={
                        "__typename": "Issue",
                        "number": 6710,
                        "state": "CLOSED",
                        "title": "done",
                        "url": "url",
                    },
                )
            ),
            "",
        )

    result = evaluate_candidates([_candidate()], runner=runner)

    assert len(calls) == 1
    assert calls[0][0:2] == ["api", "graphql"]
    assert "issueOrPullRequest" in calls[0][3]
    assert result[0].status == "fired"
    assert result[0].resolved_references == (6710,)


def test_evaluate_candidates_reports_unresolvable_reference() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args,
            0,
            '{"data":{"repository":{"item_1":null}}}',
            "",
        )

    result = evaluate_candidates([_candidate()], runner=runner)

    assert result[0].status == "unevaluatable"
    assert "unresolvable" in result[0].reason


def test_evaluate_candidates_recognizes_merged_pull_request() -> None:
    candidate = _candidate(condition="PR #6710 is merged")

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args,
            0,
            __import__("json").dumps(
                _graphql_result(
                    item={
                        "__typename": "PullRequest",
                        "number": 6710,
                        "state": "CLOSED",
                        "mergedAt": "2026-08-13T00:00:00Z",
                        "title": "merged",
                        "url": "url",
                    }
                )
            ),
            "",
        )

    result = evaluate_candidates([candidate], runner=runner)

    assert result[0].status == "fired"
    assert result[0].resolved_references == (6710,)


def test_evaluate_candidates_reports_graphql_failure_as_unevaluatable() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "GraphQL: API rate limit exceeded")

    result = evaluate_candidates([_candidate()], runner=runner)

    assert result[0].status == "unevaluatable"
    assert "dependency API error" in result[0].reason
    assert result[0].status != "not-fired"


def test_build_report_surfaces_graphql_failure_as_top_level_error() -> None:
    candidate = _candidate()

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "GraphQL: API rate limit exceeded")

    evaluations = evaluate_candidates([candidate], runner=runner)
    report = build_report([candidate], evaluations, repo="ll7/robot_sf_ll7")

    assert report["status"] == "error"
    assert report["summary"]["errors"] == 1
    assert "dependency API error" in report["errors"][0]


def test_apply_fired_never_writes_state_ready() -> None:
    candidate = _candidate(labels=("state:blocked", "state:ready"))
    evaluation = Evaluation(
        candidate.number,
        candidate.title,
        "fired",
        "tier-1-issue-graph",
        "fixture",
        (6710,),
        (6710,),
    )
    calls: list[tuple[int, str]] = []

    def writer(number: int, label: str) -> dict[str, str]:
        calls.append((number, label))
        return {"status": "ok"}

    applied, errors = apply_fired([evaluation], [candidate], writer=writer)

    assert applied == []
    assert calls == []
    assert errors == ["issue #999: refused triage write due to state:ready"]


def test_apply_fired_writes_only_needs_triage() -> None:
    candidate = _candidate()
    evaluation = Evaluation(
        candidate.number,
        candidate.title,
        "fired",
        "tier-1-issue-graph",
        "fixture",
        (6710,),
        (6710,),
    )
    calls: list[tuple[int, str]] = []

    def writer(number: int, label: str) -> dict[str, str]:
        calls.append((number, label))
        return {"status": "ok"}

    applied, errors = apply_fired([evaluation], [candidate], writer=writer)

    assert applied == [999]
    assert errors == []
    assert calls == [(999, "needs-triage")]
