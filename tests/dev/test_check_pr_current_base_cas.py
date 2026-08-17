"""Tests for the fail-closed current-main/head compare-and-swap preflight."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.dev.check_pr_current_base_cas import (
    CAS_SCHEMA,
    CurrentBaseCASSnapshot,
    check_current_base_cas,
    evaluate_current_base_cas,
)

HEAD = "a" * 40
MAIN = "b" * 40
OLD_MAIN = "c" * 40


def test_ordinary_stale_base_passes_immediate_cas() -> None:
    """An ordinary PR may use CAS when only an unrelated main commit is newer."""
    result = evaluate_current_base_cas(
        CurrentBaseCASSnapshot(
            observed_head_sha=HEAD,
            observed_main_sha=MAIN,
            base_sha=OLD_MAIN,
            base_ref="main",
            state="open",
            is_draft=False,
        ),
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
    )

    assert result["schema"] == CAS_SCHEMA
    assert result["status"] == "passed"
    assert result["base_relation"] == "stale_allowed"
    assert result["reasons"] == []


def test_changed_head_fails_cas() -> None:
    """The exact reviewed head remains mandatory even on the ordinary path."""
    result = evaluate_current_base_cas(
        CurrentBaseCASSnapshot(
            observed_head_sha="d" * 40,
            observed_main_sha=MAIN,
            base_sha=OLD_MAIN,
            base_ref="main",
            state="OPEN",
            is_draft=False,
        ),
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
    )

    assert result["status"] == "blocked"
    assert result["reasons"] == ["head_sha_changed"]


def test_main_movement_during_preflight_fails_cas() -> None:
    """A main movement between the expected and observed snapshots fails closed."""
    result = evaluate_current_base_cas(
        CurrentBaseCASSnapshot(
            observed_head_sha=HEAD,
            observed_main_sha=OLD_MAIN,
            base_sha=OLD_MAIN,
            base_ref="main",
            state="OPEN",
            is_draft=False,
        ),
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
    )

    assert result["status"] == "blocked"
    assert result["reasons"] == ["main_sha_changed_during_preflight"]


def test_base_sensitive_pr_requires_fresh_base() -> None:
    """The marker-selected path still requires current-base validation."""
    result = evaluate_current_base_cas(
        CurrentBaseCASSnapshot(
            observed_head_sha=HEAD,
            observed_main_sha=MAIN,
            base_sha=OLD_MAIN,
            base_ref="main",
            state="OPEN",
            is_draft=False,
        ),
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
        require_fresh_base=True,
    )

    assert result["status"] == "blocked"
    assert result["reasons"] == ["base_sensitive_pr_base_is_stale"]


def test_unknown_or_unsafe_pull_request_state_fails_closed() -> None:
    """Missing/unsafe PR provenance cannot become a successful CAS."""
    result = evaluate_current_base_cas(
        CurrentBaseCASSnapshot(
            observed_head_sha=HEAD,
            observed_main_sha=MAIN,
            base_sha=MAIN,
            base_ref="feature",
            state="OPEN",
            is_draft=False,
        ),
        expected_head_sha=HEAD,
        expected_main_sha=MAIN,
    )

    assert result["status"] == "blocked"
    assert result["reasons"] == ["pull_request_base_is_not_main"]


def test_live_check_binds_pull_and_main_shas() -> None:
    """The live helper reads the pull request and main through REST-backed gh calls."""
    pull = {
        "state": "open",
        "draft": False,
        "head": {"sha": HEAD},
        "base": {"sha": OLD_MAIN, "ref": "main"},
    }
    with patch("scripts.dev.check_pr_current_base_cas._gh") as mock_gh:
        mock_gh.side_effect = [
            MagicMock(returncode=0, stdout=json.dumps(pull), stderr=""),
            MagicMock(returncode=0, stdout=f"{MAIN}\n", stderr=""),
        ]
        result = check_current_base_cas(
            "6272",
            repo="owner/repo",
            expected_head_sha=HEAD,
            expected_main_sha=MAIN,
        )

    assert result["status"] == "passed"
    assert result["base_relation"] == "stale_allowed"
    assert mock_gh.call_args_list[0].args[0] == ["api", "repos/owner/repo/pulls/6272"]
    assert mock_gh.call_args_list[1].args[0] == [
        "api",
        "repos/owner/repo/branches/main",
        "--jq",
        ".commit.sha",
    ]
