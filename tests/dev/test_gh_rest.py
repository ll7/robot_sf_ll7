"""Tests for the shared ``gh api`` transport primitives (issue #7284)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.dev._gh_rest import (
    as_str,
    parse_json,
    run_gh_api,
    run_gh_api_or_raise,
    run_gh_command,
)


def _completed(*, stdout: str = "{}") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["gh", "api"], 0, stdout=stdout, stderr="")


def test_run_gh_api_preserves_method_argv_and_json_stdin() -> None:
    """HTTP method/endpoint flags stay in argv while JSON remains on stdin."""
    payload = {"labels": ["bug"]}
    with patch("scripts.dev._gh_rest.subprocess.run", return_value=_completed()) as mock_run:
        result = run_gh_api(
            "repos/example/repo/issues/7/labels",
            payload,
            method="POST",
            extra_args=["--jq", ".name"],
            timeout=9,
        )

    assert result.returncode == 0
    assert mock_run.call_args.args[0] == [
        "gh",
        "api",
        "--method",
        "POST",
        "repos/example/repo/issues/7/labels",
        "--jq",
        ".name",
        "--input",
        "-",
    ]
    assert mock_run.call_args.kwargs["input"] == '{"labels": ["bug"]}'
    assert mock_run.call_args.kwargs["timeout"] == 9
    assert mock_run.call_args.kwargs["check"] is False
    assert mock_run.call_args.kwargs.get("shell", False) is False
    assert "bug" not in " ".join(mock_run.call_args.args[0])


def test_run_gh_api_get_does_not_invent_stdin() -> None:
    """Read-only requests keep the exact no-payload argv contract."""
    with patch("scripts.dev._gh_rest.subprocess.run", return_value=_completed()) as mock_run:
        run_gh_api("repos/example/repo/issues/7", extra_args=["--field", "x=y"])

    assert mock_run.call_args.args[0] == [
        "gh",
        "api",
        "repos/example/repo/issues/7",
        "--field",
        "x=y",
    ]
    assert "input" not in mock_run.call_args.kwargs


def test_run_gh_command_preserves_argv_stdin_and_timeout() -> None:
    """Generic GitHub CLI commands use the same bounded no-shell transport."""
    with patch("scripts.dev._gh_rest.subprocess.run", return_value=_completed()) as mock_run:
        result = run_gh_command(
            ["api", "graphql", "-f", "query=query"],
            '{"variables": {}}',
            timeout=7,
        )

    assert result.returncode == 0
    assert mock_run.call_args.args[0] == ["gh", "api", "graphql", "-f", "query=query"]
    assert mock_run.call_args.kwargs["input"] == '{"variables": {}}'
    assert mock_run.call_args.kwargs["timeout"] == 7
    assert mock_run.call_args.kwargs["check"] is False
    assert mock_run.call_args.kwargs.get("shell", False) is False


@pytest.mark.parametrize(
    ("side_effect", "expected_code", "expected_text"),
    [
        (FileNotFoundError("gh"), 127, "gh CLI not found"),
        (subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=4), 124, "timed out after 4"),
    ],
)
def test_run_gh_command_normalizes_process_failures(
    side_effect: BaseException, expected_code: int, expected_text: str
) -> None:
    """Generic command callers receive deterministic missing-CLI/timeout results."""
    with patch("scripts.dev._gh_rest.subprocess.run", side_effect=side_effect):
        result = run_gh_command(["api", "graphql"], timeout=4)

    assert result.returncode == expected_code
    assert expected_text in result.stderr


@pytest.mark.parametrize(
    ("side_effect", "expected_code", "expected_text"),
    [
        (FileNotFoundError("gh"), 127, "gh CLI not found"),
        (subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=3), 124, "timed out after 3"),
    ],
)
def test_run_gh_api_converts_process_failures_to_completed_results(
    side_effect: BaseException, expected_code: int, expected_text: str
) -> None:
    """Missing CLI and timeout state never escape the shared transport."""
    with patch("scripts.dev._gh_rest.subprocess.run", side_effect=side_effect):
        result = run_gh_api("repos/example/repo", timeout=3, timeout_context="read not verified")

    assert result.returncode == expected_code
    assert expected_text in result.stderr


def test_run_gh_api_or_raise_preserves_read_audit_errors() -> None:
    """The closure audit can retain its exception-based fail-closed boundary."""
    with patch(
        "scripts.dev._gh_rest.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=30),
    ):
        with pytest.raises(RuntimeError, match="GitHub REST read timed out after 30s"):
            run_gh_api_or_raise("repos/example/repo/issues")


def test_shared_json_and_string_normalization_keep_existing_contract() -> None:
    """The identical issue/PR parser and nullable-string behavior are centralized."""
    data, error = parse_json(_completed(stdout='{"number": 7}'), what="issue 7")
    assert data == {"number": 7}
    assert error == ""
    assert as_str(None) == ""
    assert as_str(0) == "0"


def test_parse_json_normalizes_concatenated_paginated_pages_when_requested() -> None:
    """Older ``gh`` versions emit one JSON page per document without slurp."""
    result = _completed(stdout='[{"number": 1}]\n[{"number": 2}]\n')

    data, error = parse_json(result, what="paged issues", allow_concatenated=True)

    assert data == [[{"number": 1}], [{"number": 2}]]
    assert error == ""


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "expected"),
    [(1, "API detail", "", "API detail"), (0, "not json", "", "returned invalid JSON")],
)
def test_parse_json_preserves_nonzero_diagnostics_and_malformed_json(
    returncode: int, stdout: str, stderr: str, expected: str
) -> None:
    """Non-zero and empty/malformed results remain explicit and fail closed."""
    result = subprocess.CompletedProcess(["gh", "api"], returncode, stdout=stdout, stderr=stderr)

    data, error = parse_json(result, what="issue inventory")

    assert data is None
    assert expected in error


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/dev/gh_issue_rest.py",
        "scripts/dev/gh_pr_comments_rest.py",
        "scripts/dev/gh_pr_label_rest.py",
        "scripts/dev/gh_pr_review_rest.py",
        "scripts/dev/gh_pr_body_rest.py",
        "scripts/dev/open_issue_closure_audit.py",
        "scripts/dev/blocked_queue_triage.py",
        "scripts/dev/blocked_queue_watcher.py",
    ],
)
def test_rest_tools_import_shared_transport_without_local_duplicates(relative_path: str) -> None:
    """The REST tools must share transport ownership rather than fork it."""
    source = (Path(__file__).parents[2] / relative_path).read_text(encoding="utf-8")
    assert "scripts.dev._gh_rest" in source
    assert "def _gh_api" not in source
    if relative_path in {
        "scripts/dev/blocked_queue_triage.py",
        "scripts/dev/blocked_queue_watcher.py",
    }:
        assert "subprocess.run" not in source
