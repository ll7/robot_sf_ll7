"""Offline tests for the post-merge audit helper (issue #7610)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from scripts.dev.pr_audit_merged import audit_merged_pr, main

HEAD_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
BASE_SHA = "b1c2d3e4f5061728394a5b6c7d8e9f0011121314"
MERGE_SHA = "c1d2e3f405162738495a6b7c8d9e0f1122334455"


def _proc(
    *, stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess:
    """Build a fake ``gh api`` response."""
    return subprocess.CompletedProcess(["gh", "api"], returncode, stdout=stdout, stderr=stderr)


def _merged_payload(*, state: str = "merged", merged_at: str = "2026-08-20T10:00:00Z") -> str:
    return json.dumps(
        {
            "state": state,
            "merged_at": merged_at,
            "head": {"sha": HEAD_SHA},
            "base": {"sha": BASE_SHA},
            "merge_commit_sha": MERGE_SHA,
        }
    )


def test_audit_records_compact_payload_for_merged_pr() -> None:
    """A merged PR yields a machine-readable post-merge audit disposition."""
    with patch(
        "scripts.dev.pr_audit_merged._gh_api_get",
        return_value=_proc(stdout=_merged_payload()),
    ):
        result = audit_merged_pr(7547)

    assert result["status"] == "ok"
    audit = result["audit"]
    assert audit["schema"] == "post-merge-audit.v1"
    assert audit["number"] == 7547
    assert audit["state"] == "merged"
    assert audit["head_sha"] == HEAD_SHA
    assert audit["base_sha"] == BASE_SHA
    assert audit["merge_commit_sha"] == MERGE_SHA
    assert audit["merged_at"] == "2026-08-20T10:00:00Z"
    assert "comment_url" not in result


def test_audit_with_comment_posts_disposition() -> None:
    """--comment posts a compact audit comment naming the exact merge state."""
    with (
        patch(
            "scripts.dev.pr_audit_merged._gh_api_get",
            return_value=_proc(stdout=_merged_payload()),
        ),
        patch(
            "scripts.dev.pr_audit_merged._gh_api_post",
            return_value=_proc(stdout=json.dumps({"html_url": "https://github.com/x#c1"})),
        ) as mock_post,
    ):
        result = audit_merged_pr(7547, comment=True)

    assert result["status"] == "ok"
    assert result["comment_url"] == "https://github.com/x#c1"
    body = mock_post.call_args.args[1]["body"]
    assert "Post-merge audit" in body
    assert HEAD_SHA in body
    assert MERGE_SHA in body


def test_audit_fails_closed_for_open_pr() -> None:
    """An open PR cannot receive a bogus post-merge audit disposition."""
    with patch(
        "scripts.dev.pr_audit_merged._gh_api_get",
        return_value=_proc(stdout=_merged_payload(state="open", merged_at=None)),
    ):
        result = audit_merged_pr(7547)

    assert result["status"] == "error"
    assert "is not merged" in result["error"]


def test_audit_fails_closed_on_transport_error() -> None:
    """An unreadable PR object is an error, never an audit disposition."""
    with patch(
        "scripts.dev.pr_audit_merged._gh_api_get",
        return_value=_proc(returncode=1, stderr="HTTP 403: forbidden"),
    ):
        result = audit_merged_pr(7547)

    assert result["status"] == "error"
    assert "HTTP 403" in result["error"]


def test_cli_prints_compact_success_json() -> None:
    """The command-line contract is a single machine-readable success result."""
    with patch(
        "scripts.dev.pr_audit_merged._gh_api_get",
        return_value=_proc(stdout=_merged_payload()),
    ):
        rc = main(["7547", "--repo", "ll7/robot_sf_ll7"])

    assert rc == 0


def test_cli_prints_error_json_to_stderr_on_failure(capsys) -> None:
    """A failed audit must print JSON to stderr and exit 1."""
    with patch(
        "scripts.dev.pr_audit_merged._gh_api_get",
        return_value=_proc(stdout=_merged_payload(state="open", merged_at=None)),
    ):
        rc = main(["7547"])

    captured = capsys.readouterr()
    assert rc == 1
    payload = json.loads(captured.err)
    assert payload["status"] == "error"
    assert "is not merged" in payload["error"]
