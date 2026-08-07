"""Contract tests for the delta-only actionable-change monitor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import scripts.tools.check_actionable_change_monitor as monitor

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


def _labels(*names: str) -> list[dict[str, str]]:
    return [{"name": name} for name in names]


def _pull_request(number: int = 1, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "number": number,
        "title": "Benchmark research PR",
        "body": "planner benchmark update",
        "html_url": f"https://github.com/ll7/robot_sf_ll7/pull/{number}",
        "labels": _labels("research"),
        "draft": False,
        "updated_at": "2026-08-07T11:00:00Z",
        "head": {"sha": f"sha-{number}"},
    }
    payload.update(overrides)
    return payload


def _issue(number: int = 10, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "number": number,
        "title": "Research issue",
        "body": "navigation planner benchmark",
        "html_url": f"https://github.com/ll7/robot_sf_ll7/issues/{number}",
        "labels": _labels("research"),
        "updated_at": "2026-08-07T11:00:00Z",
    }
    payload.update(overrides)
    return payload


def test_build_findings_covers_research_pr_checks_and_stale_drafts() -> None:
    pull_request = _pull_request(
        draft=True,
        updated_at=(NOW - timedelta(hours=73)).isoformat(),
    )
    findings = monitor.build_findings(
        [pull_request],
        [],
        {
            1: [
                {"name": "pending-check", "status": "in_progress", "conclusion": None},
                {"name": "failed-check", "status": "completed", "conclusion": "failure"},
            ]
        },
        now=NOW,
    )

    kinds = {finding["kind"] for finding in findings}
    assert kinds == {
        "research_pr_activity",
        "stale_research_draft",
        "check_pending",
        "check_failed",
    }


def test_build_findings_covers_readiness_launch_and_evidence_contracts() -> None:
    contradictory_launch = _issue(
        10,
        title="SLURM launch packet",
        body="This blocked campaign is ready to proceed.",
        labels=_labels("research", "blocked", "state:ready", "evidence:launch-packet", "slurm"),
    )
    admitted_evidence = _issue(
        11,
        title="Admitted benchmark evidence",
        body="Job 13985 completed and the result was admitted.",
        labels=_labels("research", "evidence:nominal"),
    )

    findings = monitor.build_findings(
        [],
        [contradictory_launch, admitted_evidence],
        {},
        now=NOW,
    )

    by_number = {
        number: {finding["kind"] for finding in findings if finding["number"] == number}
        for number in (10, 11)
    }
    assert by_number[10] >= {
        "contradictory_readiness",
        "blocked_surface_with_ready_text",
        "launch_packet_without_job_id",
    }
    assert by_number[11] >= {
        "terminal_evidence_without_parent_propagation",
        "evidence_admission_without_dissertation_handoff",
    }


def test_successful_checks_and_propagated_evidence_are_not_findings() -> None:
    pull_request = _pull_request(2)
    issue = _issue(
        12,
        body="Job 13985 completed; propagated to parent #6474 with dissertation handoff.",
        labels=_labels("research", "evidence:nominal"),
    )
    findings = monitor.build_findings(
        [pull_request],
        [issue],
        {2: [{"name": "ci", "status": "completed", "conclusion": "success"}]},
        now=NOW,
    )

    assert {finding["kind"] for finding in findings} == {
        "research_pr_activity",
        "watched_research_surface",
    }


def test_target_issue_is_excluded_and_fingerprint_is_order_independent() -> None:
    target = _issue(6819, title="Actionable-change monitor", body="benchmark alert")
    other = _issue(13)
    findings = monitor.build_findings([], [target, other], {}, now=NOW, target_issue=6819)
    assert {finding["number"] for finding in findings} == {13}

    reversed_findings = list(reversed(findings))
    assert monitor.compute_fingerprint(findings) == monitor.compute_fingerprint(reversed_findings)
    assert monitor.compute_fingerprint(findings) != monitor.compute_fingerprint(
        [{**findings[0], "detail": "changed"}]
    )


def test_render_and_extract_fingerprint_marker() -> None:
    findings = [
        {
            "id": "issue:13:watched-surface",
            "kind": "watched_research_surface",
            "number": 13,
            "title": "A | title",
            "url": "https://example.test/13",
            "detail": "a | detail",
        }
    ]
    fingerprint = monitor.compute_fingerprint(findings)
    body = monitor.render_issue_body(findings, fingerprint=fingerprint, scanned_at=NOW)

    assert monitor.extract_previous_fingerprint(body) == fingerprint
    assert "A \\| title" in body
    assert "a \\| detail" in body


def test_run_monitor_writes_only_when_fingerprint_changes(monkeypatch) -> None:
    pull_request = _pull_request(20)
    snapshot = ([pull_request], [], {20: []})
    writes: list[tuple[str, int, str]] = []
    target = {"number": 6819, "state": "open", "body": ""}

    monkeypatch.setattr(monitor, "_fetch_snapshot", lambda repo, limit: snapshot)
    monkeypatch.setattr(monitor, "_read_target_issue", lambda repo, issue: target)
    monkeypatch.setattr(
        monitor,
        "_update_target_issue",
        lambda repo, issue, body: writes.append((repo, issue, body)),
    )

    first = monitor.run_monitor(repo="ll7/robot_sf_ll7", issue_number=6819, now=NOW)
    assert first["write_performed"] is True
    assert len(writes) == 1

    target["body"] = writes[0][2]
    second = monitor.run_monitor(repo="ll7/robot_sf_ll7", issue_number=6819, now=NOW)
    assert second["changed"] is False
    assert second["write_performed"] is False
    assert len(writes) == 1
