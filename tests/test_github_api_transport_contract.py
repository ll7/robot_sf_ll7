"""Transport-contract tests for the GraphQL-resistant GitHub wrappers (issue #8115).

These tests lock in the repository's GitHub transport policy so a small API
drift cannot silently re-break agent workflows:

1. ``gh_issue_rest.py`` retries only GraphQL-path failures via REST, while
   authentication, authorization, and repository-resolution failures stay
   fail-closed.
2. ``gh_pr_body_rest.py`` and ``gh_pr_label_rest.py`` write through REST only
   and verify the write before reporting success; malformed payloads are
   rejected instead of crashing.
3. ``gh_comment.sh`` publishes through the REST issue-comments endpoint and
   never depends on the GraphQL-backed ``gh issue comment`` path.
4. ``gh_pr_merge.sh`` retries GraphQL quota exhaustion only after a REST guard
   snapshot re-verifies exact head, PR state, clean mergeability, and the
   ``merge-ready`` label; auth, repository, and other non-quota errors stay
   fail-closed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.dev import gh_pr_body_rest, gh_pr_label_rest
from scripts.dev.gh_issue_rest import (
    FAIL_CLOSED_ERROR_MARKERS,
    FALLBACK_ELIGIBLE_MARKERS,
    PROJECT_CARDS_ERROR_MARKER,
    _is_fallback_eligible,
)

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "dev"


def _source(name: str) -> str:
    return (_SCRIPTS / name).read_text(encoding="utf-8")


# --- gh_issue_rest: REST fallback eligibility -------------------------------


@pytest.mark.parametrize("marker", FALLBACK_ELIGIBLE_MARKERS)
def test_fallback_eligible_markers_trigger_fallback(marker: str) -> None:
    assert _is_fallback_eligible(f"gh: {marker} something else") is True


def test_project_cards_marker_is_fallback_eligible() -> None:
    error = (
        "GraphQL: Projects (classic) is being deprecated in favor of the new Projects "
        "experience. (repository.issue.projectCards)"
    )
    assert PROJECT_CARDS_ERROR_MARKER in error
    assert _is_fallback_eligible(error) is True


@pytest.mark.parametrize("marker", FAIL_CLOSED_ERROR_MARKERS)
def test_fail_closed_markers_never_fall_back(marker: str) -> None:
    assert _is_fallback_eligible(f"GraphQL: {marker} (repository.issue.projectCards)") is False


def test_fail_closed_markers_are_not_fallback_eligible() -> None:
    assert not (set(FAIL_CLOSED_ERROR_MARKERS) & set(FALLBACK_ELIGIBLE_MARKERS))


def test_non_matching_native_failure_stays_fail_closed() -> None:
    assert _is_fallback_eligible("connection reset by peer") is False


# --- gh_pr_body_rest: REST-only writes with verification ---------------------


def test_body_rest_uses_the_rest_pulls_endpoint() -> None:
    source = _source("gh_pr_body_rest.py")
    assert "repos/{repo}/pulls/{number}" in source
    # Writes go through the shared REST PATCH helper, never a GraphQL route.
    assert "gh_api_patch" in source
    assert "gh api graphql" not in source.casefold()


def test_body_rest_rejects_non_object_payload() -> None:
    response, error = gh_pr_body_rest._decode_object(
        subprocess.CompletedProcess(args=[], returncode=0, stdout='["a", "b"]', stderr=""),
        operation="test",
    )
    assert response is None
    assert error is not None and "not an object" in error


def test_body_rest_rejects_invalid_json() -> None:
    response, error = gh_pr_body_rest._decode_object(
        subprocess.CompletedProcess(args=[], returncode=0, stdout="{not json", stderr=""),
        operation="test",
    )
    assert response is None
    assert error is not None and "invalid JSON" in error


def test_body_rest_reports_command_failure_without_exception() -> None:
    response, error = gh_pr_body_rest._decode_object(
        subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom"),
        operation="test",
    )
    assert response is None
    assert error is not None and "boom" in error


def test_body_rest_head_sha_requires_sha_string() -> None:
    assert gh_pr_body_rest._head_sha({"head": {"sha": "abc"}}) == "abc"
    assert gh_pr_body_rest._head_sha({"head": {}}) is None
    assert gh_pr_body_rest._head_sha({}) is None


def test_body_rest_verify_read_back_is_present() -> None:
    source = _source("gh_pr_body_rest.py")
    # The reconcile path re-reads the PR after writing before reporting success.
    assert "_gh_api_get" in source
    assert "verify" in source.casefold()


# --- gh_pr_label_rest: verified label writes ---------------------------------


def test_label_rest_uses_the_rest_issues_endpoint() -> None:
    source = _source("gh_pr_label_rest.py")
    assert "repos/{repo}/issues/" in source


def test_label_rest_absent_label_delete_is_idempotent_only_for_404() -> None:
    absent = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="gh: Label does not exist (HTTP 404)"
    )
    other = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="HTTP 403")
    assert gh_pr_label_rest._is_absent_label_delete(absent) is True
    assert gh_pr_label_rest._is_absent_label_delete(other) is False
    assert (
        gh_pr_label_rest._is_absent_label_delete(
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        )
        is False
    )


def test_label_rest_verify_after_write_is_present() -> None:
    add_source = _source("gh_pr_label_rest.py")
    assert "verify" in add_source.casefold()
    assert "_get_label_names" in add_source


# --- gh_comment.sh: REST-only publication ------------------------------------


def test_comment_wrapper_publishes_through_rest_issue_comments() -> None:
    source = _source("gh_comment.sh")
    assert "issues/$target_id/comments" in source
    assert "--method POST" in source or "-X POST" in source


def test_comment_wrapper_does_not_use_graphql_comment_commands() -> None:
    source = _source("gh_comment.sh")
    # The wrapper documents why it avoids the GraphQL-backed commands; strip
    # comment lines so documentation mentions do not count as invocations.
    code_lines = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    assert "gh issue comment" not in code
    assert "gh pr comment" not in code


def test_comment_wrapper_documents_the_transport_rule() -> None:
    source = _source("gh_comment.sh")
    assert "REST" in source
    assert "GraphQL" in source


# --- gh_pr_merge.sh: quota-only guarded REST fallback ------------------------


def test_merge_wrapper_quota_fallback_rechecks_rest_guard_snapshot() -> None:
    source = _source("gh_pr_merge.sh")
    assert 'gh api "repos/${repo}/pulls/${pr_number}"' in source
    assert ".head.sha" in source
    assert ".draft" in source
    assert ".mergeable_state" in source
    assert 'index("merge-ready")' in source
    assert 'git config --get remote.origin.url' in source
    assert '-f sha="$expected_head_sha"' in source


def test_merge_wrapper_quota_trigger_keeps_fail_closed_precedence() -> None:
    source = _source("gh_pr_merge.sh").casefold()
    assert 'elif is_graphql_quota_failure "$merge_error"; then' in source
    assert '"graphql:"' in source
    assert '"rate limit"' in source
    assert '"quota"' in source
    for marker in (
        "bad credentials",
        "http 401",
        "requires authentication",
        "resource not accessible",
        "could not resolve to a repository",
        "repository not found",
    ):
        assert marker in source
