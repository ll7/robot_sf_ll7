"""Truncation-guard regression coverage for the remaining bounded gh callers.

Issue #5048 extends the shared ``gh ... list --limit N`` truncation guard from
#4991 / PR #5040 to the callers it left out. Each bounded list call must either
fail closed or record an explicit structured truncation marker once its returned
row count reaches the cap; a result below the cap must stay clean.

Call sites covered here:

- ``snapshot_issue_batch.snapshot_claimable_issues`` (``gh issue list``)
- ``closed_state_label_hygiene`` (``gh search issues`` per label)
- ``open_issue_closure_audit`` (``gh search issues`` + per-issue ``gh search prs``)
- ``watch_pr_ci_status.fetch_recent_successful_ci_durations`` (``gh run list``)
- ``compact_ci_snapshot._fetch_drift_sample`` (``gh run list``)
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

from scripts.dev import closed_state_label_hygiene, compact_ci_snapshot, open_issue_closure_audit
from scripts.dev import snapshot_issue_batch as batch
from scripts.dev import watch_pr_ci_status as watch

# ---------------------------------------------------------------------------
# snapshot_issue_batch.snapshot_claimable_issues (gh issue list)
# ---------------------------------------------------------------------------


def _issue_rows(count: int) -> list[dict]:
    """Return ``count`` minimal open-issue payloads for gh issue list."""
    return [
        {
            "number": 4000 + index,
            "title": f"Issue {index}",
            "state": "open",
            "labels": [],
            "url": f"https://github.test/issues/{4000 + index}",
            "assignees": [],
        }
        for index in range(count)
    ]


def test_snapshot_claimable_issues_records_truncated_at_limit() -> None:
    """Exactly ``limit`` issue rows record a ``truncated: true`` marker."""
    limit = 3
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(
            returncode=0, stdout=json.dumps(_issue_rows(limit)), stderr=""
        )
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses", return_value={}):
            payload = batch.snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7", remote="origin", body_limit=0, limit=limit
            )

    assert payload["truncated"] is True
    assert "--limit 3" in payload["truncation_note"]
    assert len(payload["issues"]) == limit


def test_snapshot_claimable_issues_clean_below_limit() -> None:
    """Fewer issue rows than the cap keep the marker false and note empty."""
    with patch("scripts.dev.snapshot_issue_batch._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(_issue_rows(2)), stderr="")
        with patch("scripts.dev.snapshot_issue_batch._batch_claim_statuses", return_value={}):
            payload = batch.snapshot_claimable_issues(
                repo="ll7/robot_sf_ll7", remote="origin", body_limit=0, limit=20
            )

    assert payload["truncated"] is False
    assert payload["truncation_note"] == ""


# ---------------------------------------------------------------------------
# closed_state_label_hygiene (gh search issues, per label)
# ---------------------------------------------------------------------------


def test_closed_state_label_truncations_are_per_label() -> None:
    """A capped label search is flagged; a sparse one stays clean."""
    rows_by_label = {
        "state:running": [{"number": n} for n in range(5)],
        "state:hold": [{"number": 99}],
    }
    markers = closed_state_label_hygiene.build_label_truncations(rows_by_label, limit=5)

    by_label = {marker["label"]: marker for marker in markers}
    assert by_label["state:running"]["truncated"] is True
    assert "--limit 5" in by_label["state:running"]["note"]
    assert by_label["state:hold"]["truncated"] is False
    assert by_label["state:hold"]["note"] == ""


def test_closed_state_build_report_surfaces_truncation_markers() -> None:
    """The audit report exposes ``truncated_any`` and the per-label markers."""
    markers = [
        {"label": "state:running", "truncated": True, "row_count": 5, "limit": 5, "note": "capped"}
    ]
    report = closed_state_label_hygiene.build_report(
        repo="ll7/robot_sf_ll7",
        checked_labels=("state:running",),
        stale_issues=[],
        truncations=markers,
    )

    assert report["truncated_any"] is True
    assert report["truncations"] == markers


def test_closed_state_build_report_defaults_to_no_truncation() -> None:
    """Omitting truncations keeps the report backward-compatible and clean."""
    report = closed_state_label_hygiene.build_report(
        repo="ll7/robot_sf_ll7", checked_labels=("state:running",), stale_issues=[]
    )

    assert report["truncated_any"] is False
    assert report["truncations"] == []


# ---------------------------------------------------------------------------
# open_issue_closure_audit (gh search issues + per-issue gh search prs)
# ---------------------------------------------------------------------------


def test_open_issue_truncations_flag_both_bounded_calls() -> None:
    """A capped open-issue scan and a capped per-issue PR search are both flagged."""
    truncations = open_issue_closure_audit.build_truncations(
        open_issue_rows=[{"number": n} for n in range(10)],
        issue_limit=10,
        merged_pr_rows_by_issue={101: [{"number": n} for n in range(4)], 102: [{"number": 1}]},
        pr_limit_per_issue=4,
    )

    assert truncations["open_issues"]["truncated"] is True
    assert "--limit 10" in truncations["open_issues"]["note"]
    assert "--issue-limit" in truncations["open_issues"]["note"]
    assert truncations["merged_prs_per_issue"]["truncated_issues"] == [
        {"issue_number": 101, "row_count": 4, "truncated": True}
    ]
    assert truncations["truncated_any"] is True


def test_open_issue_truncations_clean_below_limits() -> None:
    """Below both caps, nothing is flagged as truncated."""
    truncations = open_issue_closure_audit.build_truncations(
        open_issue_rows=[{"number": 1}],
        issue_limit=100,
        merged_pr_rows_by_issue={101: [{"number": 1}]},
        pr_limit_per_issue=20,
    )

    assert truncations["open_issues"]["truncated"] is False
    assert truncations["merged_prs_per_issue"]["truncated_issues"] == []
    assert truncations["truncated_any"] is False


def test_open_issue_build_report_surfaces_truncation_block() -> None:
    """The report embeds the truncation block and top-level ``truncated_any``."""
    truncations = {"truncated_any": True, "open_issues": {"truncated": True}}
    report = open_issue_closure_audit.build_report(
        repo="ll7/robot_sf_ll7", candidates=[], truncations=truncations
    )

    assert report["truncated_any"] is True
    assert report["truncations"] == truncations


# ---------------------------------------------------------------------------
# watch_pr_ci_status.fetch_recent_successful_ci_durations (gh run list)
# ---------------------------------------------------------------------------


def _run_rows(count: int) -> list[dict]:
    """Return ``count`` successful gh run rows with a parseable 60s duration."""
    return [
        {
            "databaseId": index,
            "displayTitle": f"run {index}",
            "status": "completed",
            "conclusion": "success",
            "createdAt": "2026-07-10T00:00:00Z",
            "updatedAt": "2026-07-10T00:01:00Z",
        }
        for index in range(count)
    ]


def test_watch_fetch_durations_warns_when_run_list_truncated(caplog) -> None:  # type: ignore[no-untyped-def]
    """A raw run list at the cap logs a structured truncation warning."""
    with patch("scripts.dev.watch_pr_ci_status._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(_run_rows(2)), stderr="")
        with caplog.at_level(logging.WARNING, logger="scripts.dev.watch_pr_ci_status"):
            durations = watch.fetch_recent_successful_ci_durations(workflow="CI", limit=2)

    assert durations == [60, 60]
    assert any("gh run list truncated" in record.message for record in caplog.records)


def test_watch_fetch_durations_quiet_below_limit(caplog) -> None:  # type: ignore[no-untyped-def]
    """A run list below the cap emits no truncation warning."""
    with patch("scripts.dev.watch_pr_ci_status._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(_run_rows(2)), stderr="")
        with caplog.at_level(logging.WARNING, logger="scripts.dev.watch_pr_ci_status"):
            durations = watch.fetch_recent_successful_ci_durations(workflow="CI", limit=10)

    assert durations == [60, 60]
    assert not any("gh run list truncated" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# compact_ci_snapshot._fetch_drift_sample (gh run list)
# ---------------------------------------------------------------------------


def test_compact_drift_sample_marks_truncated_at_limit() -> None:
    """A capped drift run list surfaces ``truncated: true`` on the DriftSample."""
    with patch("scripts.dev.compact_ci_snapshot._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(_run_rows(3)), stderr="")
        sample = compact_ci_snapshot._fetch_drift_sample("CI", 3, "ll7/robot_sf_ll7")

    assert sample.truncated is True
    assert sample.sample_count == 3


def test_compact_drift_sample_clean_below_limit() -> None:
    """A drift run list below the cap leaves ``truncated`` false."""
    with patch("scripts.dev.compact_ci_snapshot._gh") as mock_gh:
        mock_gh.return_value = MagicMock(returncode=0, stdout=json.dumps(_run_rows(2)), stderr="")
        sample = compact_ci_snapshot._fetch_drift_sample("CI", 10, "ll7/robot_sf_ll7")

    assert sample.truncated is False
    assert sample.sample_count == 2
