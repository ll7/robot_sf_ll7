"""Offline tests for the scheduled main-CI incident reconciler (#8114)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.dev import reconcile_main_ci_incidents as reconciler

REPO = "owner/repo"
INCIDENT_LABEL = reconciler.INCIDENT_LABEL


def _proc(payload: Any, *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Build one fake ``gh api`` response."""
    return subprocess.CompletedProcess(["gh", "api"], returncode, json.dumps(payload), stderr="")


def _run(run_id: int, conclusion: str, created_at: str) -> dict[str, Any]:
    """Build one completed CI run row."""
    return {
        "databaseId": run_id,
        "status": "completed",
        "conclusion": conclusion,
        "headSha": f"{run_id:040x}",
        "createdAt": created_at,
    }


def _body(run_id: int) -> str:
    """Build the canonical incident body used by the existing creator."""
    return "\n".join(
        [
            "<!-- ll7-main-red-incident:v1 -->",
            "Automated escalation.",
            f"Deciding failing run: https://github.com/{REPO}/actions/runs/{run_id}",
            "Close when two consecutive main CI runs pass.",
        ]
    )


def _issue(
    *, number: int = 1, run_id: int = 300, updated_at: str = "2026-09-01T00:00:00Z"
) -> dict[str, Any]:
    """Build one REST issue row with the canonical incident label."""
    return {
        "number": number,
        "title": "main CI is red",
        "body": _body(run_id),
        "state": "open",
        "updated_at": updated_at,
        "html_url": f"https://github.com/{REPO}/issues/{number}",
        "labels": [{"name": INCIDENT_LABEL}],
    }


class FakeREST:
    """Small stateful REST fake that exposes mutation order and payloads."""

    def __init__(self, issue: dict[str, Any], comments: list[dict[str, Any]] | None = None) -> None:
        """Initialize the fake issue and optional existing comments."""
        self.issue = issue
        self.comments = list(comments or [])
        self.calls: list[tuple[str, object | None, str | None]] = []
        self.issue_reads = 0
        self.drift_on_first_issue_read = False

    def __call__(
        self,
        path: str,
        payload: object | None = None,
        *,
        method: str | None = None,
        extra_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Answer the endpoint subset used by the reconciler."""
        assert extra_args is None
        self.calls.append((path, payload, method))
        if "/issues?" in path:
            rows = [self.issue] if self.issue["state"] == "open" else []
            return _proc(rows)
        if path.split("?", 1)[0].endswith("/comments"):
            if method == "POST":
                assert isinstance(payload, dict)
                body = payload["body"]
                self.comments.append({"id": 99, "body": body})
                self.issue["updated_at"] = "2026-09-01T00:01:00Z"
                return _proc({"id": 99, "body": body})
            return _proc(self.comments)
        if path.endswith(f"/issues/{self.issue['number']}"):
            if method == "PATCH":
                assert isinstance(payload, dict)
                assert payload == {"state": "closed", "state_reason": "completed"}
                self.issue["state"] = "closed"
                self.issue["state_reason"] = "completed"
                return _proc(self.issue)
            self.issue_reads += 1
            if self.drift_on_first_issue_read and self.issue_reads == 1:
                self.issue["updated_at"] = "2026-09-01T00:02:00Z"
            return _proc(self.issue)
        raise AssertionError(f"unexpected REST path: {path}")


def _fetcher(runs: list[dict[str, Any]]):
    """Return an injectable run-window fetcher."""
    return lambda _repo, _workflow, limit: runs[:limit]


def test_parse_deciding_run_requires_one_canonical_field() -> None:
    """Malformed or duplicated incident fields are pending, never eligible."""
    run_id, error = reconciler.parse_deciding_run_id(_body(300), repo=REPO)
    assert run_id == 300
    assert error is None

    missing, missing_error = reconciler.parse_deciding_run_id("", repo=REPO)
    duplicate, duplicate_error = reconciler.parse_deciding_run_id(
        _body(300) + "\n" + _body(301), repo=REPO
    )
    assert missing is None and missing_error == "incident body is missing"
    assert duplicate is None and duplicate_error is not None


def test_report_only_marks_stale_only_after_two_newer_green_runs() -> None:
    """Report-only mode identifies a stale candidate without writing hosted state."""
    fake = FakeREST(_issue(run_id=300))
    runs = [
        _run(500, "success", "2026-09-01T02:00:00Z"),
        _run(400, "success", "2026-09-01T01:00:00Z"),
        _run(300, "failure", "2026-09-01T00:00:00Z"),
    ]

    report = reconciler.reconcile_batch(
        repo=REPO,
        apply=False,
        runner=fake,
        run_fetcher=_fetcher(runs),
    )

    result = report["results"][0]
    assert report["status"] == "ok"
    assert result["classifier_status"] == "stale"
    assert result["status"] == "stale"
    assert result["action"] == "would_close"
    assert [run["id"] for run in result["green_runs"]] == [500, 400]
    assert not [call for call in fake.calls if call[2] in {"POST", "PATCH"}]


def test_active_incident_stays_open_without_comment_reads() -> None:
    """A newer decisive failure is active and must not be mutated."""
    fake = FakeREST(_issue(run_id=300))
    report = reconciler.reconcile_batch(
        repo=REPO,
        apply=True,
        runner=fake,
        run_fetcher=_fetcher(
            [
                _run(500, "failure", "2026-09-01T02:00:00Z"),
                _run(400, "success", "2026-09-01T01:00:00Z"),
            ]
        ),
    )

    result = report["results"][0]
    assert result["status"] == "active"
    assert result["action"] == "none"
    assert fake.issue["state"] == "open"
    assert not [call for call in fake.calls if call[2] in {"POST", "PATCH"}]


@pytest.mark.parametrize(
    "runs",
    [
        [_run(300, "success", "2026-09-01T00:00:00Z")],
        [_run(500, "in_progress", "2026-09-01T02:00:00Z")],
    ],
)
def test_pending_or_insufficient_green_evidence_stays_open(
    runs: list[dict[str, Any]],
) -> None:
    """Equal, incomplete, or non-decisive windows fail closed without writes."""
    fake = FakeREST(_issue(run_id=300))
    report = reconciler.reconcile_batch(
        repo=REPO,
        apply=True,
        runner=fake,
        run_fetcher=_fetcher(runs),
    )

    result = report["results"][0]
    assert result["action"] == "none"
    assert result["status"] == "pending"
    assert fake.issue["state"] == "open"
    assert not [call for call in fake.calls if call[2] in {"POST", "PATCH"}]


def test_malformed_deciding_run_is_pending_without_mutation() -> None:
    """An incident with no canonical deciding field is visible but untouched."""
    fake = FakeREST(_issue(run_id=300))
    fake.issue["body"] = fake.issue["body"].replace("Deciding failing run:", "Deciding run:")
    report = reconciler.reconcile_batch(
        repo=REPO,
        apply=True,
        runner=fake,
        run_fetcher=_fetcher([_run(500, "success", "2026-09-01T02:00:00Z")]),
    )

    result = report["results"][0]
    assert result["status"] == "pending"
    assert "canonical deciding run" in result["reason"]
    assert not [call for call in fake.calls if call[2] in {"POST", "PATCH"}]


def test_apply_posts_exact_evidence_then_closes_and_reads_back() -> None:
    """Apply mode preserves evidence-before-close ordering and payload exactness."""
    fake = FakeREST(_issue(run_id=300))
    runs = [
        _run(500, "success", "2026-09-01T02:00:00Z"),
        _run(400, "success", "2026-09-01T01:00:00Z"),
        _run(300, "failure", "2026-09-01T00:00:00Z"),
    ]
    report = reconciler.reconcile_batch(
        repo=REPO,
        apply=True,
        runner=fake,
        run_fetcher=_fetcher(runs),
    )

    result = report["results"][0]
    assert result["action"] == "closed"
    assert result["comment_action"] == "comment_created"
    assert fake.issue["state"] == "closed"
    post_index = next(index for index, call in enumerate(fake.calls) if call[2] == "POST")
    patch_index = next(index for index, call in enumerate(fake.calls) if call[2] == "PATCH")
    assert post_index < patch_index
    post_payload = fake.calls[post_index][1]
    assert isinstance(post_payload, dict)
    comment = post_payload["body"]
    assert comment.startswith("<!-- main-ci-incident-reconciled:v1 issue=1 deciding-run=300 -->")
    assert "actions/runs/500" in comment
    assert "actions/runs/400" in comment


def test_existing_exact_evidence_comment_makes_retry_idempotent() -> None:
    """A retry reuses the exact marker-bound comment and emits no duplicate POST."""
    green_runs = [
        {"id": 500, "created_at": "2026-09-01T02:00:00Z"},
        {"id": 400, "created_at": "2026-09-01T01:00:00Z"},
    ]
    existing_body = reconciler._comment_body(
        issue=1,
        repo=REPO,
        deciding_run_id=300,
        green_runs=green_runs,
    )
    fake = FakeREST(_issue(run_id=300), comments=[{"id": 7, "body": existing_body}])
    report = reconciler.reconcile_batch(
        repo=REPO,
        apply=True,
        runner=fake,
        run_fetcher=_fetcher(
            [
                _run(500, "success", "2026-09-01T02:00:00Z"),
                _run(400, "success", "2026-09-01T01:00:00Z"),
                _run(300, "failure", "2026-09-01T00:00:00Z"),
            ]
        ),
    )

    assert report["results"][0]["action"] == "closed"
    assert report["results"][0]["comment_action"] == "comment_existing"
    assert not [call for call in fake.calls if call[2] == "POST"]
    assert fake.issue["state"] == "closed"


def test_changed_issue_precondition_skips_all_writes() -> None:
    """A timestamp drift between inventory and mutation aborts safely."""
    fake = FakeREST(_issue(run_id=300))
    fake.drift_on_first_issue_read = True
    report = reconciler.reconcile_batch(
        repo=REPO,
        apply=True,
        runner=fake,
        run_fetcher=_fetcher(
            [
                _run(500, "success", "2026-09-01T02:00:00Z"),
                _run(400, "success", "2026-09-01T01:00:00Z"),
                _run(300, "failure", "2026-09-01T00:00:00Z"),
            ]
        ),
    )

    result = report["results"][0]
    assert result["action"] == "precondition_changed"
    assert fake.issue["state"] == "open"
    assert not [call for call in fake.calls if call[2] in {"POST", "PATCH"}]


def test_empty_inventory_does_not_require_a_run_fetch() -> None:
    """An empty open incident set is a successful no-op."""
    fake = FakeREST(_issue(run_id=300))
    fake.issue["state"] = "closed"

    def unexpected_fetch(*_args: object) -> list[dict[str, Any]]:
        raise AssertionError("empty inventory should not fetch Actions runs")

    report = reconciler.reconcile_batch(
        repo=REPO,
        apply=True,
        runner=fake,
        run_fetcher=unexpected_fetch,
    )

    assert report["status"] == "ok"
    assert report["results"] == []
    assert report["source"]["open_incident_count"] == 0


def test_malformed_run_window_fails_closed_before_any_issue_write() -> None:
    """A malformed fetched run cannot become evidence or trigger a close."""
    fake = FakeREST(_issue(run_id=300))
    with pytest.raises(reconciler.ReconciliationError, match="databaseId"):
        reconciler.reconcile_batch(
            repo=REPO,
            apply=True,
            runner=fake,
            run_fetcher=_fetcher(
                [
                    {
                        "status": "completed",
                        "conclusion": "success",
                        "createdAt": "2026-09-01T02:00:00Z",
                    }
                ]
            ),
        )
    assert not [call for call in fake.calls if call[2] in {"POST", "PATCH"}]


def test_scheduled_workflow_uses_explicit_apply_lane_and_narrow_permissions() -> None:
    """The hosted schedule invokes the report helper with only required rights."""
    root = Path(__file__).parents[2]
    workflow = (root / ".github/workflows/main-ci-incident-reconcile.yml").read_text()
    assert 'cron: "17 */6 * * *"' in workflow
    assert "actions: read" in workflow
    assert "issues: write" in workflow
    assert "scripts/dev/reconcile_main_ci_incidents.py" in workflow
    assert "--apply" in workflow
    assert "--output output/main-ci-incidents/report.json" in workflow
