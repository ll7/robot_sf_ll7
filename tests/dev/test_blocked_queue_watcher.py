"""Contract tests for the fail-closed blocked queue watcher."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from scripts.dev import blocked_queue_watcher as watcher
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


def _adapter_candidate(
    adapter: dict[str, Any],
    *,
    number: int = 998,
    condition: str = "the typed adapter condition is satisfied",
) -> IssueCandidate:
    payload = {
        "blocker_class": "dependency",
        "unblock_condition": condition,
        "watcher": "inspect the adapter result",
        "next_check_at": "next queue pass",
        "last_meaningful_progress_at": "2026-08-13",
        "adapter": adapter,
    }
    body = (
        "<!-- blocked-triage-v1 tracking=7067 -->\n"
        "```yaml\n"
        f"{yaml.safe_dump(payload, sort_keys=False)}"
        "```\n"
    )
    return _candidate(
        number=number,
        comments=({"url": f"https://example.test/issues/{number}#comment", "body": body},),
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


def test_evaluate_candidates_reports_graphql_failure_as_error() -> None:
    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "GraphQL: API rate limit exceeded")

    result = evaluate_candidates([_candidate()], runner=runner)

    assert result[0].status == "error"
    assert "dependency API error" in result[0].reason
    assert result[0].status != "not-fired"


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [("inventory detail", "", "inventory detail"), ("", "", "code 1")],
)
def test_inventory_translates_shared_transport_diagnostics(
    stdout: str, stderr: str, expected: str
) -> None:
    """Inventory failures remain watcher-specific RuntimeErrors with detail."""

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, stdout, stderr)

    with pytest.raises(RuntimeError, match=expected):
        watcher._inventory("owner/repo", runner=runner)


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


def test_path_presence_adapter_distinguishes_present_and_missing(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "frozen.yaml").write_text("version: 1\n", encoding="utf-8")
    present = _adapter_candidate(
        {
            "version": 1,
            "kind": "path_presence",
            "name": "path_exists",
            "path": "configs/frozen.yaml",
            "path_type": "file",
        }
    )
    missing = _adapter_candidate(
        {
            "version": 1,
            "kind": "path_presence",
            "name": "path_exists",
            "path": "configs/missing.yaml",
            "path_type": "file",
        },
        number=997,
    )

    result = evaluate_candidates([present, missing], repo_root=tmp_path)

    assert [row.status for row in result] == ["fired", "not-fired"]
    assert result[0].provenance["path"] == "configs/frozen.yaml"
    assert result[0].provenance["observed_type_match"] is True


def test_repo_predicate_adapter_is_bounded_and_literal(tmp_path: Path) -> None:
    source = tmp_path / "robot_sf"
    source.mkdir()
    (source / "marker.py").write_text("adversarial_independent_outcomes\n", encoding="utf-8")
    candidate = _adapter_candidate(
        {
            "version": 1,
            "kind": "repo_predicate",
            "name": "text_present",
            "path": "robot_sf",
            "text": "adversarial_independent_outcomes",
        }
    )

    result = evaluate_candidates([candidate], repo_root=tmp_path)

    assert result[0].status == "fired"
    assert result[0].tier == "adapter-repo_predicate"
    assert result[0].provenance["matched_path"] == "robot_sf/marker.py"


@pytest.mark.parametrize(
    "adapter",
    [
        {
            "version": 1,
            "kind": "path_presence",
            "name": "path_exists",
            "path": "../outside.txt",
        },
        {
            "version": 1,
            "kind": "external_probe",
            "name": "run_shell",
            "command": "echo unsafe",
            "minimum_remaining": 1,
        },
        {
            "version": 1,
            "kind": "repo_predicate",
            "name": "text_present",
            "path": "output",
            "text": "not allow-listed",
        },
    ],
)
def test_adapter_contract_rejects_unsafe_or_malformed_conditions(
    adapter: dict[str, Any],
) -> None:
    outcome = parse_triage_record(_adapter_candidate(adapter))

    assert outcome.status == "malformed"
    assert outcome.record is None


def test_external_probe_is_allow_listed_cached_and_provenanced() -> None:
    adapter = {
        "version": 1,
        "kind": "external_probe",
        "name": "github_graphql_quota",
        "minimum_remaining": 100,
    }
    candidates = [_adapter_candidate(adapter), _adapter_candidate(adapter, number=997)]
    calls: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "250\n", "")

    result = evaluate_candidates(candidates, runner=runner)

    assert [row.status for row in result] == ["fired", "fired"]
    assert calls == [["api", "rate_limit", "--jq", ".resources.graphql.remaining"]]
    assert result[0].provenance["observed_remaining"] == 250
    assert result[0].provenance["command"][0] == "gh"
    assert result[1].provenance["triage_source_url"].endswith("/997#comment")


def test_external_probe_failure_is_error_and_blocks_apply() -> None:
    candidate = _adapter_candidate(
        {
            "version": 1,
            "kind": "external_probe",
            "name": "github_graphql_quota",
            "minimum_remaining": 100,
        }
    )

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "rate limit unavailable")

    result = evaluate_candidates([candidate], runner=runner)
    report = build_report([candidate], result, repo="ll7/robot_sf_ll7")

    assert result[0].status == "error"
    assert report["summary"]["error"] == 1
    assert report["status"] == "error"
