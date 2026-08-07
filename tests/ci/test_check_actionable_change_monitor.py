"""Contract tests for the delta-only actionable-change monitor (issue #6819).

These tests are pure: the finding and fingerprint computation never touches the
network, and the ``run_monitor`` orchestration is exercised through injected
seams so the no-op / write / fail-closed contract is verified offline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts.ci.check_actionable_change_monitor import (
    DEFAULT_ISSUE,
    DEFAULT_REPO,
    MonitorError,
    build_findings,
    compute_fingerprint,
    extract_previous_fingerprint,
    render_issue_body,
    run_monitor,
)

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


def _result_body(fingerprint: str) -> dict[str, object]:
    return {
        "number": DEFAULT_ISSUE,
        "state": "open",
        "body": f"<!-- actionable-change-monitor:fingerprint:{fingerprint} -->",
    }


# ── empty input scan ─────────────────────────────────────────────────────────


def test_empty_input_scan_yields_no_findings() -> None:
    findings = build_findings([], [], {}, now=NOW)
    assert findings == []


def test_empty_scan_is_a_no_op_success() -> None:
    result = run_monitor(
        repo=DEFAULT_REPO,
        issue_number=DEFAULT_ISSUE,
        now=NOW,
        fetch_snapshot=lambda repo, limit: ([], [], {}),
        read_target_issue=lambda repo, issue: _result_body("d" * 64),
        update_target_issue=lambda repo, issue, body: pytest.fail("empty scan must never write"),
    )
    assert result["changed"] is False
    assert result["write_performed"] is False
    assert result["reason"] == "empty_scan"
    assert result["finding_count"] == 0


# ── unchanged fingerprint yields the same digest ─────────────────────────────


def test_unchanged_input_yields_same_fingerprint() -> None:
    findings = build_findings([_pull_request(1)], [_issue(10)], {}, now=NOW)
    assert compute_fingerprint(findings) == compute_fingerprint(findings)


def test_unchanged_fingerprint_is_a_no_op_success() -> None:
    findings = build_findings([_pull_request(1)], [_issue(10)], {}, now=NOW)
    fingerprint = compute_fingerprint(findings)

    result = run_monitor(
        repo=DEFAULT_REPO,
        issue_number=DEFAULT_ISSUE,
        now=NOW,
        fetch_snapshot=lambda repo, limit: ([_pull_request(1)], [_issue(10)], {}),
        read_target_issue=lambda repo, issue: _result_body(fingerprint),
        update_target_issue=lambda repo, issue, body: pytest.fail(
            "unchanged fingerprint must never write"
        ),
    )
    assert result["changed"] is False
    assert result["write_performed"] is False
    assert result["reason"] == "unchanged"
    assert result["fingerprint"] == fingerprint


# ── changed input yields a different digest ──────────────────────────────────


def test_changed_input_yields_different_fingerprint() -> None:
    empty = compute_fingerprint([])
    with_findings = compute_fingerprint(build_findings([_pull_request(1)], [], {}, now=NOW))
    assert empty != with_findings


def test_changed_fingerprint_writes_rendered_body() -> None:
    findings = build_findings([_pull_request(1)], [_issue(10)], {}, now=NOW)
    fingerprint = compute_fingerprint(findings)
    written: dict[str, object] = {}

    def update_target_issue(repo: str, issue: int, body: str, **kwargs: object) -> None:
        written["repo"] = repo
        written["issue"] = issue
        written["body"] = body
        written["expected_fingerprint"] = kwargs["expected_fingerprint"]

    result = run_monitor(
        repo=DEFAULT_REPO,
        issue_number=DEFAULT_ISSUE,
        now=NOW,
        fetch_snapshot=lambda repo, limit: ([_pull_request(1)], [_issue(10)], {}),
        read_target_issue=lambda repo, issue: _result_body("d" * 64),
        update_target_issue=update_target_issue,
    )
    assert result["changed"] is True
    assert result["write_performed"] is True
    assert written["repo"] == DEFAULT_REPO
    assert written["issue"] == DEFAULT_ISSUE
    body = str(written["body"])
    assert f"actionable-change-monitor:fingerprint:{fingerprint}" in body
    assert extract_previous_fingerprint(body) == fingerprint
    assert written["expected_fingerprint"] == fingerprint


# ── target issue manifest round-trip ─────────────────────────────────────────


def test_render_extract_round_trip_preserves_fingerprint() -> None:
    findings = build_findings([_pull_request(1)], [_issue(10)], {}, now=NOW)
    fingerprint = compute_fingerprint(findings)
    body = render_issue_body(findings, fingerprint=fingerprint, scanned_at=NOW)
    assert extract_previous_fingerprint(body) == fingerprint


# ── fail-closed API error handling ───────────────────────────────────────────


def test_api_error_in_scan_aborts_with_no_write() -> None:
    def failing_fetch(repo: str, limit: int):
        raise MonitorError("GitHub API call failed for repos/ll7/robot_sf_ll7/pulls")

    with pytest.raises(MonitorError):
        run_monitor(
            repo=DEFAULT_REPO,
            issue_number=DEFAULT_ISSUE,
            fetch_snapshot=failing_fetch,
            read_target_issue=lambda repo, issue: _result_body("d" * 64),
            update_target_issue=lambda repo, issue, body: pytest.fail("API error must never write"),
        )


def test_update_target_issue_rejects_non_pinned_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.ci.check_actionable_change_monitor as mod

    monkeypatch.setattr(
        mod,
        "_read_target_issue",
        lambda repo, issue: _result_body("d" * 64),  # type: ignore[assignment]
    )
    monkeypatch.setattr(mod, "_gh_json", lambda *a, **k: pytest.fail("must not call GitHub"))

    with pytest.raises(MonitorError, match="refusing to write outside the pinned target"):
        mod._update_target_issue(
            "somewhere/else", DEFAULT_ISSUE, "body", expected_fingerprint="d" * 64
        )
    with pytest.raises(MonitorError, match="refusing to write outside the pinned target"):
        mod._update_target_issue(DEFAULT_REPO, 9999, "body", expected_fingerprint="d" * 64)


def test_update_target_issue_writes_after_identity_and_re_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.ci.check_actionable_change_monitor as mod

    re_reads: list[tuple[str, int]] = []
    patched: dict[str, object] = {}

    monkeypatch.setattr(
        mod,
        "_read_target_issue",
        lambda repo, issue: re_reads.append((repo, issue)) or _result_body("e" * 64),  # type: ignore[assignment]
    )
    monkeypatch.setattr(
        mod,
        "_gh_json",
        lambda path, repo, method, input_json: patched.update(  # type: ignore[arg-type]
            path=path, repo=repo, method=method, input_json=input_json
        ),
    )

    mod._update_target_issue(DEFAULT_REPO, DEFAULT_ISSUE, "new body", expected_fingerprint="d" * 64)

    assert re_reads == [(DEFAULT_REPO, DEFAULT_ISSUE)]
    assert patched["path"] == f"repos/{DEFAULT_REPO}/issues/{DEFAULT_ISSUE}"
    assert patched["method"] == "PATCH"
    assert patched["input_json"] == '{"body": "new body"}'


def test_update_target_issue_skips_write_when_rerun_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.ci.check_actionable_change_monitor as mod

    fingerprint = "ab" * 32
    monkeypatch.setattr(
        mod,
        "_read_target_issue",
        lambda repo, issue: _result_body(fingerprint),  # type: ignore[assignment]
    )
    monkeypatch.setattr(
        mod,
        "_gh_json",
        lambda *a, **k: pytest.fail("must not write when the re-read fingerprint matches"),  # type: ignore[misc]
    )

    mod._update_target_issue(
        DEFAULT_REPO, DEFAULT_ISSUE, "new body", expected_fingerprint=fingerprint
    )


# ── research surface coverage ────────────────────────────────────────────────


def test_build_findings_covers_research_pr_checks_and_stale_drafts() -> None:
    pull_request = _pull_request(
        draft=True,
        updated_at=(NOW - timedelta(hours=73)).isoformat(),
    )
    findings = build_findings(
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

    findings = build_findings([], [contradictory_launch, admitted_evidence], {}, now=NOW)

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
    findings = build_findings(
        [pull_request],
        [issue],
        {2: [{"name": "ci", "status": "completed", "conclusion": "success"}]},
        now=NOW,
    )

    kinds = {finding["kind"] for finding in findings}
    assert "check_pending" not in kinds
    assert "check_failed" not in kinds
    assert "terminal_evidence_without_parent_propagation" not in kinds
    assert "evidence_admission_without_dissertation_handoff" not in kinds
