"""Issue #8677 same-account static-report projection and custody tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from scripts.dev.same_account_review_report import (
    APPROVED_APP_ID,
    APPROVED_APP_OWNER,
    APPROVED_APP_SLUG,
    APPROVED_PRODUCER_IDENTITY,
    bind_report_evidence,
    fetch_same_account_static_reports,
    project_static_reports_from_comments,
    report_evidence_digest,
)
from scripts.dev.single_account_merge_receipt import classify_implementation_review

REPO = "ll7/robot_sf_ll7"
PR = 8769
HEAD = "a" * 40
OTHER_HEAD = "f" * 40
METADATA = "b" * 64
OTHER_METADATA = "e" * 64


def _report_body(
    *,
    head: str = HEAD,
    metadata: str = METADATA,
    verdict: str = "accepted",
    findings: int = 0,
) -> str:
    body = f"""## Independent implementation review

### Scope

Reviewed the exact diff and implementation contract at the declared head without modifying it.

### Findings

No unresolved correctness defect remains in the reviewed implementation and its focused tests.

### Validation

Inspected the changed source and exercised the focused deterministic verification path.

single-account-review: {verdict} @ {head}
metadata: {metadata}
evidence: {'0' * 64}
unresolved-correctness-findings: {findings}
"""
    return bind_report_evidence(body=body, repository=REPO, pr_number=PR)


def _app_comment(
    body: str,
    *,
    comment_id: int = 100,
    created_at: str = "2026-09-10T10:00:00Z",
    updated_at: str | None = None,
    publisher: str = "ll7",
    app_id: int = APPROVED_APP_ID,
    app_slug: str = APPROVED_APP_SLUG,
    app_owner: str = APPROVED_APP_OWNER,
) -> dict[str, Any]:
    updated = created_at if updated_at is None else updated_at
    return {
        "url": f"https://api.github.com/repos/{REPO}/issues/comments/{comment_id}",
        "html_url": f"https://github.com/{REPO}/pull/{PR}#issuecomment-{comment_id}",
        "issue_url": f"https://api.github.com/repos/{REPO}/issues/{PR}",
        "id": comment_id,
        "node_id": f"node-{comment_id}",
        "user": {"login": publisher},
        "created_at": created_at,
        "updated_at": updated,
        "body": body,
        "performed_via_github_app": {
            "id": app_id,
            "slug": app_slug,
            "owner": {"login": app_owner},
        },
        "minimized": None,
    }


def _classify(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return classify_implementation_review(
        {
            "head_sha": HEAD,
            "metadata_digest": METADATA,
            "waiver_actor": "ll7",
            "static_reports": reports,
        }
    )


def test_valid_live_shaped_report_passes_real_receipt_classifier() -> None:
    body = _report_body()
    reports, provenance = project_static_reports_from_comments(
        [_app_comment(body)], repository=REPO, pr_number=PR
    )

    result = _classify(reports)
    assert result["status"] == "accepted"
    assert result["carrier"]["kind"] == "static_report"
    assert result["carrier"]["identity"] == APPROVED_PRODUCER_IDENTITY
    assert reports[0]["publisher_identity"] == "ll7"
    assert reports[0]["evidence_digest"] == report_evidence_digest(
        body=body, repository=REPO, pr_number=PR
    )
    assert provenance["status"] == "accepted"
    assert provenance["selected_comment_id"] == 100


@pytest.mark.parametrize(
    "comment",
    [
        _app_comment(
            _report_body(),
            updated_at="2026-09-10T10:00:01Z",
        ),
        _app_comment(_report_body(), publisher="someone-else"),
    ],
)
def test_edited_or_wrong_publisher_custody_refuses(comment: dict[str, Any]) -> None:
    reports, _provenance = project_static_reports_from_comments(
        [comment], repository=REPO, pr_number=PR
    )
    result = _classify(reports)
    assert result["status"] == "unavailable"
    assert "review_carrier_source_not_approved" in result["reason_codes"]


def test_direct_owner_or_copied_app_identity_does_not_create_a_report() -> None:
    direct_owner = _app_comment(_report_body())
    direct_owner.pop("performed_via_github_app")
    copied_app = _app_comment(_report_body(), app_id=42)

    reports, provenance = project_static_reports_from_comments(
        [direct_owner, copied_app], repository=REPO, pr_number=PR
    )
    assert reports == []
    assert provenance["status"] == "missing"
    assert _classify(reports)["status"] == "missing"


def test_tampered_report_digest_refuses() -> None:
    comment = _app_comment(_report_body().replace("focused tests", "altered tests"))
    reports, provenance = project_static_reports_from_comments(
        [comment], repository=REPO, pr_number=PR
    )
    result = _classify(reports)
    assert result["status"] == "conflicting"
    assert "static_review_not_accepted" in result["reason_codes"]
    assert reports[0]["verdict"] == "invalid_evidence"
    assert provenance["status"] == "malformed"


@pytest.mark.parametrize(
    "body",
    [
        _report_body(head=OTHER_HEAD),
        _report_body(metadata=OTHER_METADATA),
    ],
)
def test_stale_original_binding_refuses(body: str) -> None:
    reports, _provenance = project_static_reports_from_comments(
        [_app_comment(body)], repository=REPO, pr_number=PR
    )
    result = _classify(reports)
    assert result["status"] == "stale"


def test_findings_bearing_report_refuses() -> None:
    reports, _provenance = project_static_reports_from_comments(
        [_app_comment(_report_body(verdict="changes_requested", findings=2))],
        repository=REPO,
        pr_number=PR,
    )
    result = _classify(reports)
    assert result["status"] == "conflicting"
    assert "static_review_not_accepted" in result["reason_codes"]


def test_latest_invalid_report_supersedes_older_acceptance() -> None:
    accepted = _app_comment(
        _report_body(), comment_id=100, created_at="2026-09-10T10:00:00Z"
    )
    tampered = _app_comment(
        _report_body().replace("focused tests", "tampered tests"),
        comment_id=101,
        created_at="2026-09-10T10:01:00Z",
    )
    reports, provenance = project_static_reports_from_comments(
        [accepted, tampered], repository=REPO, pr_number=PR
    )

    assert reports[0]["superseded"] is True
    assert reports[1].get("superseded") is not True
    assert _classify(reports)["status"] == "conflicting"
    assert provenance["selected_comment_id"] == 101


@dataclass
class _Result:
    stdout: str
    stderr: str = ""
    returncode: int = 0


def test_fetch_uses_complete_rest_comment_route() -> None:
    calls: list[list[str]] = []

    def gh(args: list[str], timeout: int = 30) -> _Result:
        assert timeout == 45
        calls.append(args)
        return _Result(stdout=json.dumps([_app_comment(_report_body())]))

    reports, provenance = fetch_same_account_static_reports(
        gh, repository=REPO, pr_number=PR
    )
    assert _classify(reports)["status"] == "accepted"
    assert provenance["status"] == "accepted"
    assert calls == [
        ["api", f"repos/{REPO}/issues/{PR}/comments?per_page=100&page=1"]
    ]


def test_rest_unavailability_does_not_create_approval() -> None:
    def gh(_args: list[str], timeout: int = 30) -> _Result:
        assert timeout == 45
        return _Result(stdout="", stderr="network unavailable", returncode=1)

    reports, provenance = fetch_same_account_static_reports(
        gh, repository=REPO, pr_number=PR
    )
    assert reports == []
    assert provenance["status"] == "unavailable"
    assert provenance["reason_codes"] == ["network unavailable"]
    assert _classify(reports)["status"] == "missing"
