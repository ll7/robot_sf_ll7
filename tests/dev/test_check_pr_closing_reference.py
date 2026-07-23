"""Tests for the PR closing-reference verification script (no live GitHub)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.check_pr_closing_reference import (
    _extract_issues_from_graphql,
    _fetch_closing_issues,
    _parse_json,
    _resolve_repo,
    _split_owner_repo,
    check_closing_reference,
    main,
)

REPO = "ll7/robot_sf_ll7"


def _gh_response(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Create a mock subprocess.CompletedProcess for gh calls."""
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def _graphql_response(
    pr_number: int, closing_issues: list[int], *, has_next_page: bool = False
) -> str:
    """Build a GraphQL response payload with closing issue references."""
    nodes = [
        {"number": num, "url": f"https://github.com/{REPO}/issues/{num}"} for num in closing_issues
    ]
    return json.dumps(
        {
            "data": {
                "repository": {
                    "pullRequest": {
                        "closingIssuesReferences": {
                            "nodes": nodes,
                            "pageInfo": {"hasNextPage": has_next_page},
                        }
                    }
                }
            }
        }
    )


# ── unit tests: _fetch_closing_issues ────────────────────────────────────────


def test_fetch_closing_issues_returns_numbers() -> None:
    """Parses closing issue numbers from a valid GraphQL response."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=_graphql_response(42, [1, 2, 3]))
        issues, error = _fetch_closing_issues("42", repo=REPO)

    assert error is None
    assert issues == [1, 2, 3]
    gh_args = mock_gh.call_args.args[0]
    assert "first: 100" in gh_args[gh_args.index("-f") + 1]


def test_fetch_closing_issues_returns_empty_list_when_none() -> None:
    """Returns an empty list when the PR has no closing references."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=_graphql_response(42, []))
        issues, error = _fetch_closing_issues("42", repo=REPO)

    assert error is None
    assert issues == []


def test_fetch_closing_issues_returns_error_on_gh_failure() -> None:
    """Returns error when gh api graphql fails."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="rate limit exceeded")
        issues, error = _fetch_closing_issues("42", repo=REPO)

    assert issues is None
    assert error is not None
    assert "rate limit" in error.lower()


def test_fetch_closing_issues_returns_error_on_invalid_json() -> None:
    """Returns error when gh output is not valid JSON."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout="not json")
        issues, error = _fetch_closing_issues("42", repo=REPO)

    assert issues is None
    assert error is not None
    assert "parse" in error.lower()


def test_fetch_closing_issues_returns_error_on_missing_pr() -> None:
    """Returns error when the PR does not exist."""
    payload = json.dumps({"data": {"repository": {"pullRequest": None}}})
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=payload)
        issues, error = _fetch_closing_issues("99999", repo=REPO)

    assert issues is None
    assert error is not None
    assert "not found" in error.lower()


def test_fetch_closing_issues_returns_error_on_graphql_errors() -> None:
    """Returns error when GraphQL returns top-level errors."""
    payload = json.dumps(
        {"errors": [{"message": "Field 'closingIssuesReferences' is unsupported"}]}
    )
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=payload)
        issues, error = _fetch_closing_issues("42", repo=REPO)

    assert issues is None
    assert error is not None
    assert "unsupported" in error.lower()


def test_fetch_closing_issues_handles_missing_repository() -> None:
    """Returns error when the repository field is missing."""
    payload = json.dumps({"data": {"repository": None}})
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=payload)
        issues, error = _fetch_closing_issues("42", repo=REPO)

    assert issues is None
    assert error is not None


def test_fetch_closing_issues_rejects_invalid_pr_number_without_calling_gh() -> None:
    """Returns an input error instead of raising or invoking gh for an invalid PR number."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        issues, error = _fetch_closing_issues("not-a-number", repo=REPO)

    assert issues is None
    assert error is not None
    assert "positive integer" in error
    mock_gh.assert_not_called()


# ── unit tests: _split_owner_repo ────────────────────────────────────────────


def test_split_owner_repo_valid() -> None:
    """Splits owner/repo correctly."""
    assert _split_owner_repo("ll7/robot_sf_ll7") == ("ll7", "robot_sf_ll7")


@pytest.mark.parametrize("value", ["invalid", "", "/missing_owner", "owner/repo/extra"])
def test_split_owner_repo_invalid_format(value: str) -> None:
    """Raises ValueError for invalid formats."""
    with pytest.raises(ValueError, match="Invalid repository format"):
        _split_owner_repo(value)


# ── unit tests: _parse_json ──────────────────────────────────────────────────


def test_parse_json_valid() -> None:
    """Parses valid JSON object."""
    data, error = _parse_json('{"key": "value"}')
    assert error is None
    assert data == {"key": "value"}


def test_parse_json_invalid() -> None:
    """Returns error for invalid JSON."""
    data, error = _parse_json("not json")
    assert data is None
    assert error is not None


def test_parse_json_non_object() -> None:
    """Returns error when JSON is not an object."""
    data, error = _parse_json("[1, 2, 3]")
    assert data is None
    assert error is not None


# ── unit tests: _resolve_repo ────────────────────────────────────────────────


def test_resolve_repo_uses_explicit() -> None:
    """Returns the explicit repository string when provided."""
    assert _resolve_repo("custom/example") == "custom/example"


def test_resolve_repo_auto_detects() -> None:
    """Auto-detects repository via gh repo view when no explicit repo given."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout="owner/auto_repo\n")
        repo = _resolve_repo("")

    assert repo == "owner/auto_repo"


def test_resolve_repo_returns_none_on_failure() -> None:
    """Returns None when auto-detection fails."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="not authenticated")
        repo = _resolve_repo("")

    assert repo is None


# ── unit tests: _extract_issues_from_graphql ─────────────────────────────────


def test_extract_issues_from_nodes() -> None:
    """Extracts issue numbers from valid GraphQL nodes."""
    payload = json.loads(_graphql_response(42, [1, 2, 3]))
    issues, error = _extract_issues_from_graphql(payload, "42")
    assert error is None
    assert issues == [1, 2, 3]


def test_extract_issues_rejects_null_nodes() -> None:
    """Fails closed when nodes is null instead of treating malformed data as no references."""
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "closingIssuesReferences": {
                        "nodes": None,
                        "pageInfo": {"hasNextPage": False},
                    }
                }
            }
        }
    }
    issues, error = _extract_issues_from_graphql(payload, "1")
    assert issues is None
    assert error is not None


def test_extract_issues_returns_error_on_missing_closing_refs() -> None:
    """Returns error when closingIssuesReferences field is missing."""
    payload = {"data": {"repository": {"pullRequest": {}}}}
    _, error = _extract_issues_from_graphql(payload, "1")
    assert error is not None


def test_extract_issues_handles_top_level_errors() -> None:
    """Returns error message from GraphQL top-level errors."""
    payload = {"errors": [{"message": "Something went wrong"}]}
    _, error = _extract_issues_from_graphql(payload, "1")
    assert error is not None
    assert "Something went wrong" in error


def test_extract_issues_rejects_partial_data_with_errors() -> None:
    """Fails closed when GraphQL reports errors alongside otherwise usable partial data."""
    payload = json.loads(_graphql_response(42, [6024]))
    payload["errors"] = [{"message": "closing references may be incomplete"}]

    issues, error = _extract_issues_from_graphql(payload, "42")

    assert issues is None
    assert error is not None
    assert "incomplete" in error


def test_extract_issues_rejects_malformed_empty_errors_object() -> None:
    """Fails closed when the GraphQL errors field has the wrong container type."""
    payload = json.loads(_graphql_response(42, [6024]))
    payload["errors"] = {}

    issues, error = _extract_issues_from_graphql(payload, "42")

    assert issues is None
    assert error == "GraphQL errors field is malformed"


def test_extract_issues_rejects_malformed_node() -> None:
    """Fails closed rather than silently dropping a malformed closing-reference node."""
    payload = json.loads(_graphql_response(42, [6024]))
    payload["data"]["repository"]["pullRequest"]["closingIssuesReferences"]["nodes"].append(
        {"number": "6025"}
    )

    issues, error = _extract_issues_from_graphql(payload, "42")

    assert issues is None
    assert error is not None
    assert "malformed" in error


def test_extract_issues_rejects_truncated_connection() -> None:
    """Fails closed when GitHub reports another page beyond the bounded query."""
    payload = json.loads(_graphql_response(42, [6024], has_next_page=True))

    issues, error = _extract_issues_from_graphql(payload, "42")

    assert issues is None
    assert error is not None
    assert "incomplete" in error


# ── unit tests: check_closing_reference ──────────────────────────────────────


def test_check_ok_when_expected_issue_matches() -> None:
    """Returns ok when the expected issue is among closing references."""
    with patch("scripts.dev.check_pr_closing_reference._fetch_closing_issues") as mock_fetch:
        mock_fetch.return_value = ([42, 43], None)
        result = check_closing_reference("1", 42, repo=REPO)

    assert result.status == "ok"
    assert result.expected_issue == 42
    assert result.actual_closing_issues == (42, 43)


def test_check_mismatch_when_expected_issue_not_found() -> None:
    """Returns mismatch when the expected issue is not among closing references."""
    with patch("scripts.dev.check_pr_closing_reference._fetch_closing_issues") as mock_fetch:
        mock_fetch.return_value = ([1, 2], None)
        result = check_closing_reference("1", 42, repo=REPO)

    assert result.status == "mismatch"
    assert result.expected_issue == 42
    assert result.actual_closing_issues == (1, 2)
    assert "does not close" in result.message


def test_check_mismatch_when_no_closing_references() -> None:
    """Returns mismatch when the PR has no closing references."""
    with patch("scripts.dev.check_pr_closing_reference._fetch_closing_issues") as mock_fetch:
        mock_fetch.return_value = ([], None)
        result = check_closing_reference("1", 42, repo=REPO)

    assert result.status == "mismatch"
    assert result.actual_closing_issues == ()
    assert "(none)" in result.message


def test_check_error_when_fetch_fails() -> None:
    """Returns error when fetching closing issues fails."""
    with patch("scripts.dev.check_pr_closing_reference._fetch_closing_issues") as mock_fetch:
        mock_fetch.return_value = (None, "network error")
        result = check_closing_reference("1", 42, repo=REPO)

    assert result.status == "error"
    assert "network error" in result.message


def test_check_error_for_non_positive_expected_issue_without_fetch() -> None:
    """Rejects an invalid expected issue before querying GitHub."""
    with patch("scripts.dev.check_pr_closing_reference._fetch_closing_issues") as mock_fetch:
        result = check_closing_reference("1", 0, repo=REPO)

    assert result.status == "error"
    assert "positive integer" in result.message
    mock_fetch.assert_not_called()


def test_check_multiple_closing_references_includes_expected() -> None:
    """Returns ok when expected issue is among several closing references."""
    with patch("scripts.dev.check_pr_closing_reference._fetch_closing_issues") as mock_fetch:
        mock_fetch.return_value = ([10, 20, 30, 40], None)
        result = check_closing_reference("5", 20, repo=REPO)

    assert result.status == "ok"
    assert result.actual_closing_issues == (10, 20, 30, 40)


def test_check_actual_issues_sorted() -> None:
    """Actual closing issues are always sorted."""
    with patch("scripts.dev.check_pr_closing_reference._fetch_closing_issues") as mock_fetch:
        mock_fetch.return_value = ([3, 1, 2], None)
        result = check_closing_reference("1", 1, repo=REPO)

    assert result.actual_closing_issues == (1, 2, 3)


# ── main() integration tests (mocked _gh) ────────────────────────────────────


def test_main_returns_0_on_match() -> None:
    """main() exits 0 when expected issue is a closing reference."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=f"{REPO}\n"),
            _gh_response(stdout=_graphql_response(42, [6024])),
        ]
        exit_code = main(["42", "6024"])

    assert exit_code == 0


def test_main_returns_1_on_mismatch() -> None:
    """main() exits 1 when expected issue is not a closing reference."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=f"{REPO}\n"),
            _gh_response(stdout=_graphql_response(42, [9999])),
        ]
        exit_code = main(["42", "6024"])

    assert exit_code == 1


def test_main_returns_2_on_api_failure() -> None:
    """main() exits 2 when the GraphQL API call fails."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=f"{REPO}\n"),
            _gh_response(returncode=1, stderr="rate limit exceeded"),
        ]
        exit_code = main(["42", "6024"])

    assert exit_code == 2


def test_main_returns_2_on_repo_detection_failure() -> None:
    """main() exits 2 when repo auto-detection fails."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(returncode=1, stderr="not authenticated")
        exit_code = main(["42", "6024"])

    assert exit_code == 2


def test_main_returns_2_when_repo_detection_times_out() -> None:
    """Maps a gh timeout during auto-detection to the documented indeterminate exit code."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.side_effect = subprocess.TimeoutExpired(cmd=["gh", "repo", "view"], timeout=30)
        exit_code = main(["42", "6024"])

    assert exit_code == 2


def test_main_explicit_repo() -> None:
    """main() accepts --repo flag instead of auto-detecting."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.return_value = _gh_response(stdout=_graphql_response(42, [6024]))
        exit_code = main(["42", "6024", "--repo", REPO])

    assert exit_code == 0


def test_main_json_output() -> None:
    """main() emits machine-readable JSON with --json."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=f"{REPO}\n"),
            _gh_response(stdout=_graphql_response(1, [6024])),
        ]
        exit_code = main(["1", "6024", "--json"])

    assert exit_code == 0


def test_main_json_mismatch_output() -> None:
    """main() JSON output includes mismatch details."""
    with patch("scripts.dev.check_pr_closing_reference._gh") as mock_gh:
        mock_gh.side_effect = [
            _gh_response(stdout=f"{REPO}\n"),
            _gh_response(stdout=_graphql_response(1, [])),
        ]
        exit_code = main(["1", "6024", "--json"])

    assert exit_code == 1
