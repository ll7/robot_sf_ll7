"""Contract tests for the delta-only actionable-change monitor."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

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
        "stale_research_draft",
        "check_pending",
        "check_failed",
    }

    inventory = monitor.build_inventory([pull_request], [], target_issue=6819)
    assert {finding["kind"] for finding in inventory} == {"research_pr_activity"}


def test_duplicate_check_run_names_have_unique_ids_and_stable_fingerprint() -> None:
    pull_request = _pull_request(4)
    check_runs = [
        {"id": 401, "name": "ci", "status": "completed", "conclusion": "failure"},
        {"id": 402, "name": "ci", "status": "in_progress", "conclusion": None},
    ]

    findings = monitor._check_findings(pull_request, check_runs)
    reversed_findings = monitor._check_findings(pull_request, list(reversed(check_runs)))

    assert len({finding["id"] for finding in findings}) == 2
    assert monitor.compute_fingerprint(findings) == monitor.compute_fingerprint(reversed_findings)


def test_build_findings_covers_legacy_commit_statuses() -> None:
    findings = monitor.build_findings(
        [_pull_request(3)],
        [],
        {3: []},
        now=NOW,
        statuses_by_pr={
            3: [
                {"context": "legacy-pending", "state": "pending"},
                {"context": "legacy-failure", "state": "failure"},
            ]
        },
    )

    assert {finding["kind"] for finding in findings} == {
        "check_pending",
        "check_failed",
    }
    inventory = monitor.build_inventory([_pull_request(3)], [])
    assert {finding["kind"] for finding in inventory} == {"research_pr_activity"}


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


@pytest.mark.parametrize(
    "label",
    ["blocked:needs-campaign", "evidence:provisional", "resource:external-data", "state:pending"],
)
def test_prefix_only_watch_labels_are_selected_as_research_surfaces(label: str) -> None:
    issue = _issue(14, body="ordinary maintenance", labels=_labels(label))

    findings = monitor.build_inventory([], [issue])

    assert {finding["kind"] for finding in findings} == {"watched_research_surface"}


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
        statuses_by_pr={2: [{"context": "legacy-ci", "state": "success"}]},
    )

    assert findings == []
    inventory = monitor.build_inventory([pull_request], [issue])
    assert {finding["kind"] for finding in inventory} == {
        "research_pr_activity",
        "watched_research_surface",
    }


def test_target_issue_is_excluded_and_fingerprint_is_order_independent() -> None:
    target = _issue(6819, title="Actionable-change monitor", body="benchmark alert")
    other = _issue(13)
    inventory = monitor.build_inventory([], [target, other], target_issue=6819)
    assert {finding["number"] for finding in inventory} == {13}

    reversed_inventory = list(reversed(inventory))
    assert monitor.compute_fingerprint(inventory) == monitor.compute_fingerprint(reversed_inventory)
    assert monitor.compute_fingerprint(inventory) != monitor.compute_fingerprint(
        [{**inventory[0], "detail": "changed"}]
    )


def test_report_fingerprint_uses_compact_inventory_counts() -> None:
    findings = [{"id": "issue:10:blocked", "kind": "blocked", "number": 10}]
    inventory = {
        "total": 103,
        "by_kind": {"research_pr_activity": 5, "watched_research_surface": 98},
    }

    assert monitor.compute_report_fingerprint(
        findings, inventory
    ) == monitor.compute_report_fingerprint(list(reversed(findings)), dict(inventory))
    changed = {"total": 104, "by_kind": {"research_pr_activity": 5, "watched_research_surface": 99}}
    assert monitor.compute_report_fingerprint(
        findings, inventory
    ) != monitor.compute_report_fingerprint(findings, changed)


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


def test_render_issue_body_summarizes_inventory_without_rows() -> None:
    body = monitor.render_issue_body(
        [],
        fingerprint="a" * 64,
        scanned_at=NOW,
        inventory={
            "total": 103,
            "by_kind": {"research_pr_activity": 5, "watched_research_surface": 98},
        },
    )

    assert "Finding count: `0`" in body
    assert "Stable inventory count: `103`" in body
    assert "| `watched_research_surface` | 98 |" in body
    assert "issue:10:watched-surface" not in body


def test_workflow_declares_read_permissions_for_both_check_state_apis() -> None:
    workflow_path = Path(".github/workflows/actionable-change-monitor.yml")
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert workflow["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
        "checks": "read",
        "statuses": "read",
        "issues": "write",
    }


def test_workflow_serializes_every_writer_under_one_target_wide_group() -> None:
    workflow_path = Path(".github/workflows/actionable-change-monitor.yml")
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    concurrency = workflow["concurrency"]
    group = str(concurrency["group"])

    # A ref-scoped group lets a scheduled main run and a workflow_dispatch run
    # from another ref write issue #6819 at the same time.
    assert "github.ref" not in group
    assert "${{" not in group
    assert str(monitor.DEFAULT_ISSUE) in group
    # Cancelling an in-flight writer between its recheck and its PATCH is the
    # race the single-writer group exists to prevent.
    assert concurrency["cancel-in-progress"] is False


def test_fetch_snapshot_paginates_check_runs_and_statuses(monkeypatch) -> None:
    calls: list[str] = []
    pull_request = _pull_request(20)

    def fake_gh_json(path: str, **kwargs: object) -> object:
        del kwargs
        calls.append(path)
        if "/pulls?" in path:
            return [pull_request]
        if "/issues?" in path:
            return []
        if "/check-runs?" in path:
            page = int(path.rsplit("page=", 1)[1])
            if page == 1:
                return {
                    "total_count": 101,
                    "check_runs": [
                        {"id": index + 1, "name": f"check-{index}", "status": "completed"}
                        for index in range(100)
                    ],
                }
            return {
                "total_count": 101,
                "check_runs": [{"id": 101, "name": "check-100", "status": "completed"}],
            }
        if "/status?" in path:
            return {
                "total_count": 1,
                "statuses": [{"context": "legacy-ci", "state": "success"}],
            }
        raise AssertionError(f"unexpected GitHub path: {path}")

    monkeypatch.setattr(monitor, "_gh_json", fake_gh_json)
    _, _, check_runs_by_pr, statuses_by_pr = monitor._fetch_snapshot(monitor.DEFAULT_REPO, 150)

    assert len(check_runs_by_pr[20]) == 101
    assert statuses_by_pr[20] == [{"context": "legacy-ci", "state": "success"}]
    assert any("/check-runs?per_page=100&page=2" in call for call in calls)
    assert any("/status?per_page=100&page=1" in call for call in calls)


@pytest.mark.parametrize(
    ("resource", "collection_key"),
    [("check-runs", "check_runs"), ("status", "statuses")],
)
def test_fetch_commit_collection_rejects_inconsistent_short_pages(
    monkeypatch, resource: str, collection_key: str
) -> None:
    calls: list[str] = []

    def fake_gh_json(path: str, **kwargs: object) -> object:
        del kwargs
        calls.append(path)
        row = (
            {"id": 1, "name": "only-row", "status": "completed"}
            if collection_key == "check_runs"
            else {"context": "only-row", "state": "success"}
        )
        return {"total_count": 101, collection_key: [row]}

    monkeypatch.setattr(monitor, "_gh_json", fake_gh_json)

    with pytest.raises(monitor.MonitorError, match="short page"):
        monitor._fetch_commit_collection(
            monitor.DEFAULT_REPO,
            "sha-inconsistent",
            resource=resource,
            collection_key=collection_key,
            limit=150,
        )

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("resource", "collection_key", "metadata"),
    [
        ("check-runs", "check_runs", {}),
        ("check-runs", "check_runs", {"total_count": None}),
        ("status", "statuses", {}),
        ("status", "statuses", {"total_count": None}),
    ],
)
def test_fetch_commit_collection_requires_total_count_metadata(
    monkeypatch, resource: str, collection_key: str, metadata: dict[str, object]
) -> None:
    calls: list[str] = []

    def fake_gh_json(path: str, **kwargs: object) -> object:
        del kwargs
        calls.append(path)
        return {**metadata, collection_key: []}

    monkeypatch.setattr(monitor, "_gh_json", fake_gh_json)

    with pytest.raises(monitor.MonitorError, match="total_count"):
        monitor._fetch_commit_collection(
            monitor.DEFAULT_REPO,
            "sha-missing-total-count",
            resource=resource,
            collection_key=collection_key,
            limit=150,
        )

    assert len(calls) == 1


def test_fetch_snapshot_rejects_malformed_check_run_collection(monkeypatch) -> None:
    pull_request = _pull_request(21)

    def fake_gh_json(path: str, **kwargs: object) -> object:
        del kwargs
        if "/pulls?" in path:
            return [pull_request]
        if "/issues?" in path:
            return []
        if "/check-runs?" in path:
            return {"total_count": 1}
        raise AssertionError(f"unexpected GitHub path: {path}")

    monkeypatch.setattr(monitor, "_gh_json", fake_gh_json)

    with pytest.raises(monitor.MonitorError, match="check_runs"):
        monitor._fetch_snapshot(monitor.DEFAULT_REPO, 100)


@pytest.mark.parametrize(
    ("resource", "malformed_pull_request"),
    [
        ("pulls", {"labels": [{"id": 1}]}),
        ("pulls", {"head": {"sha": None}}),
        ("issues", {"labels": [{"id": 1}]}),
        ("issues", {"number": "22"}),
    ],
)
def test_fetch_snapshot_rejects_malformed_surface_fields(
    monkeypatch, resource: str, malformed_pull_request: dict[str, object]
) -> None:
    pull_request = _pull_request(21)
    issue = _issue(22)
    (pull_request if resource == "pulls" else issue).update(malformed_pull_request)

    def fake_gh_json(path: str, **kwargs: object) -> object:
        del kwargs
        if "/pulls?" in path:
            return [pull_request] if resource == "pulls" else []
        if "/issues?" in path:
            return [issue] if resource == "issues" else []
        raise AssertionError(f"unexpected GitHub path: {path}")

    monkeypatch.setattr(monitor, "_gh_json", fake_gh_json)

    with pytest.raises(monitor.MonitorError, match="invalid|malformed"):
        monitor._fetch_snapshot(monitor.DEFAULT_REPO, 100)


@pytest.mark.parametrize(
    ("resource", "collection_key", "row"),
    [
        ("check-runs", "check_runs", {"name": "ci", "status": "completed"}),
        ("status", "statuses", {"context": "ci"}),
    ],
)
def test_fetch_commit_collection_rejects_malformed_rows(
    monkeypatch, resource: str, collection_key: str, row: dict[str, object]
) -> None:
    def fake_gh_json(path: str, **kwargs: object) -> object:
        del path, kwargs
        return {"total_count": 1, collection_key: [row]}

    monkeypatch.setattr(monitor, "_gh_json", fake_gh_json)

    with pytest.raises(monitor.MonitorError, match="invalid|malformed"):
        monitor._fetch_commit_collection(
            monitor.DEFAULT_REPO,
            "sha-malformed",
            resource=resource,
            collection_key=collection_key,
            limit=100,
        )


@pytest.mark.parametrize("resource", ["pulls", "issues"])
def test_fetch_snapshot_rejects_malformed_top_level_rows(monkeypatch, resource: str) -> None:
    pull_request = _pull_request(21)

    def fake_gh_json(path: str, **kwargs: object) -> object:
        del kwargs
        if "/pulls?" in path:
            return [pull_request, None] if resource == "pulls" else [pull_request]
        if "/issues?" in path:
            return [_issue(22), "malformed"] if resource == "issues" else []
        raise AssertionError(f"unexpected GitHub path: {path}")

    monkeypatch.setattr(monitor, "_gh_json", fake_gh_json)

    expected_resource = "pull requests" if resource == "pulls" else resource
    with pytest.raises(monitor.MonitorError, match=expected_resource):
        monitor._fetch_snapshot(monitor.DEFAULT_REPO, 100)


def test_run_monitor_rejects_noncanonical_target_before_reading(monkeypatch) -> None:
    def fail_snapshot(*args: object, **kwargs: object) -> object:
        raise AssertionError("non-canonical target must be rejected first")

    monkeypatch.setattr(monitor, "_fetch_snapshot", fail_snapshot)

    with pytest.raises(monitor.MonitorError, match="restricted"):
        monitor.run_monitor(repo="other/repository", issue_number=42, now=NOW)


def test_empty_scan_is_a_noop_even_without_a_previous_marker(monkeypatch) -> None:
    writes: list[object] = []
    target = {"number": 6819, "state": "open", "body": ""}

    monkeypatch.setattr(monitor, "_fetch_snapshot", lambda repo, limit: ([], [], {}, {}))
    monkeypatch.setattr(monitor, "_read_target_issue", lambda repo, issue: target)
    monkeypatch.setattr(
        monitor,
        "_update_target_issue",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    result = monitor.run_monitor(repo=monitor.DEFAULT_REPO, issue_number=6819, now=NOW)

    assert result["finding_count"] == 0
    assert result["changed"] is False
    assert result["write_performed"] is False
    assert writes == []


def test_run_monitor_writes_only_when_fingerprint_changes(monkeypatch) -> None:
    pull_request = _pull_request(20)
    snapshot = ([pull_request], [], {20: []}, {})
    writes: list[tuple[str, int, str]] = []
    target = {"number": 6819, "state": "open", "body": ""}

    monkeypatch.setattr(monitor, "_fetch_snapshot", lambda repo, limit: snapshot)
    monkeypatch.setattr(monitor, "_read_target_issue", lambda repo, issue: target)
    monkeypatch.setattr(
        monitor,
        "_update_target_issue",
        lambda repo, issue, body, **kwargs: writes.append((repo, issue, body)),
    )

    first = monitor.run_monitor(repo="ll7/robot_sf_ll7", issue_number=6819, now=NOW)
    assert first["write_performed"] is True
    assert len(writes) == 1

    target["body"] = writes[0][2]
    second = monitor.run_monitor(repo="ll7/robot_sf_ll7", issue_number=6819, now=NOW)
    assert second["changed"] is False
    assert second["write_performed"] is False
    assert len(writes) == 1


def test_run_monitor_reports_v2_actionable_and_inventory_counts(monkeypatch) -> None:
    pull_request = _pull_request(21)
    snapshot = ([pull_request], [], {21: []}, {})
    writes: list[tuple[object, ...]] = []
    target = {"number": 6819, "state": "open", "body": ""}

    monkeypatch.setattr(monitor, "_fetch_snapshot", lambda repo, limit: snapshot)
    monkeypatch.setattr(monitor, "_read_target_issue", lambda repo, issue: target)
    monkeypatch.setattr(
        monitor,
        "_update_target_issue",
        lambda *args, **kwargs: writes.append(args),
    )

    result = monitor.run_monitor(repo="ll7/robot_sf_ll7", issue_number=6819, now=NOW)

    assert result["schema_version"] == "robot_sf.actionable_change_monitor.v2"
    assert result["finding_count"] == 0
    assert result["inventory_count"] == 1
    assert result["inventory_by_kind"] == {"research_pr_activity": 1}


def test_run_monitor_refuses_stale_target_before_patching(monkeypatch) -> None:
    pull_request = _pull_request(22)
    target_reads = iter(
        [
            {"number": 6819, "state": "open", "body": ""},
            {
                "number": 6819,
                "state": "open",
                "body": "<!-- actionable-change-monitor:fingerprint:" + "a" * 64 + " -->",
            },
        ]
    )

    monkeypatch.setattr(
        monitor,
        "_fetch_snapshot",
        lambda repo, limit: ([pull_request], [], {22: []}, {}),
    )
    monkeypatch.setattr(monitor, "_read_target_issue", lambda repo, issue: next(target_reads))

    def fail_patch(*args: object, **kwargs: object) -> object:
        raise AssertionError("stale target must be rejected before PATCH")

    monkeypatch.setattr(monitor.subprocess, "run", fail_patch)

    with pytest.raises(monitor.MonitorError, match="stale write"):
        monitor.run_monitor(repo=monitor.DEFAULT_REPO, issue_number=6819, now=NOW)


def test_run_monitor_refuses_target_body_change_without_marker_change(monkeypatch) -> None:
    pull_request = _pull_request(23)
    target_reads = iter(
        [
            {
                "number": 6819,
                "state": "open",
                "body": "operator note",
            },
            {
                "number": 6819,
                "state": "open",
                "body": "operator note changed",
            },
        ]
    )

    monkeypatch.setattr(
        monitor,
        "_fetch_snapshot",
        lambda repo, limit: ([pull_request], [], {23: []}, {}),
    )
    monkeypatch.setattr(monitor, "_read_target_issue", lambda repo, issue: next(target_reads))

    with pytest.raises(monitor.MonitorError, match="stale write"):
        monitor.run_monitor(repo=monitor.DEFAULT_REPO, issue_number=6819, now=NOW)


@pytest.mark.parametrize(
    "malformed",
    ["not-an-ISO-timestamp", "", "   ", "2026-08-07T11:00:00"],
)
def test_malformed_updated_at_fails_closed_instead_of_classifying_as_stale(
    monkeypatch, malformed: str
) -> None:
    """A malformed or naive timestamp must stop the scan, not publish a finding."""
    pull_request = _pull_request(31, draft=True, updated_at=malformed)

    def fake_gh_json(path: str, **kwargs: object) -> object:
        del kwargs
        if "/pulls?" in path:
            return [pull_request]
        return []

    monkeypatch.setattr(monitor, "_gh_json", fake_gh_json)

    with pytest.raises(monitor.MonitorError, match="updated_at"):
        monitor._fetch_snapshot(monitor.DEFAULT_REPO, 5)


def test_parse_timestamp_never_silently_returns_the_epoch() -> None:
    with pytest.raises(monitor.MonitorError):
        monitor._parse_timestamp("not-an-ISO-timestamp")
    with pytest.raises(monitor.MonitorError):
        monitor._parse_timestamp(None)
    # Naive timestamps cannot be compared against the timezone-aware scan clock.
    with pytest.raises(monitor.MonitorError):
        monitor._parse_timestamp("2026-08-07T11:00:00")

    parsed = monitor._parse_timestamp("2026-08-07T11:00:00Z")
    assert parsed == datetime(2026, 8, 7, 11, 0, tzinfo=UTC)


def test_concurrent_overwrite_of_the_published_report_fails_closed(monkeypatch) -> None:
    """The losing writer in a PATCH race must not report a successful publish."""
    pull_request = _pull_request(32)
    stored = {"body": ""}
    patch_calls: list[str] = []

    def read_target(repo: str, issue: int) -> dict[str, object]:
        del repo, issue
        return {"number": 6819, "state": "open", "body": stored["body"]}

    def fake_patch(command, **kwargs):
        del kwargs
        field = next(arg for arg in command if str(arg).startswith("body="))
        patch_calls.append(str(field)[len("body=") :])
        stored["body"] = str(field)[len("body=") :]
        # A concurrent writer commits its own report between our PATCH and our
        # read-back, exactly as the exact-head concurrency probe demonstrated.
        stored["body"] = "body-from-the-other-writer"
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(
        monitor,
        "_fetch_snapshot",
        lambda repo, limit: ([pull_request], [], {32: []}, {}),
    )
    monkeypatch.setattr(monitor, "_read_target_issue", read_target)
    monkeypatch.setattr(monitor.subprocess, "run", fake_patch)

    with pytest.raises(monitor.MonitorError, match="overwritten by a concurrent writer"):
        monitor.run_monitor(repo=monitor.DEFAULT_REPO, issue_number=6819, now=NOW)

    assert len(patch_calls) == 1


def test_durable_write_passes_the_read_back_verification(monkeypatch) -> None:
    pull_request = _pull_request(33)
    stored = {"body": ""}

    def read_target(repo: str, issue: int) -> dict[str, object]:
        del repo, issue
        return {"number": 6819, "state": "open", "body": stored["body"]}

    def fake_patch(command, **kwargs):
        del kwargs
        field = next(arg for arg in command if str(arg).startswith("body="))
        stored["body"] = str(field)[len("body=") :]
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(
        monitor,
        "_fetch_snapshot",
        lambda repo, limit: ([pull_request], [], {33: []}, {}),
    )
    monkeypatch.setattr(monitor, "_read_target_issue", read_target)
    monkeypatch.setattr(monitor.subprocess, "run", fake_patch)

    result = monitor.run_monitor(repo=monitor.DEFAULT_REPO, issue_number=6819, now=NOW)

    assert result["write_performed"] is True
    assert monitor.extract_previous_fingerprint(stored["body"]) == result["fingerprint"]
