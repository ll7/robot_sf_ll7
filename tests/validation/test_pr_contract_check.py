"""Unit tests for scripts/ci/pr_contract_check.py."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path  # noqa: TC003
from unittest.mock import MagicMock, patch

import pytest

from scripts.ci import pr_contract_check


def test_find_closed_issues() -> None:
    """Test find_closed_issues matches closing keywords."""
    body = (
        "This fixes #123 and closes #456. Resolves https://github.com/ll7/robot_sf_ll7/issues/789."
    )
    closed = pr_contract_check.find_closed_issues(body)
    assert closed == ["123", "456", "789"]


def test_find_title_issues() -> None:
    """Test find_title_issues parses issue numbers from title."""
    assert pr_contract_check.find_title_issues("Issue #4735: some title") == ["4735"]
    assert pr_contract_check.find_title_issues("Refs #123, issue 456") == ["123", "456"]


def test_has_declaration_for_issue() -> None:
    """Test has_declaration_for_issue checks body matches."""
    body = "We reference Refs #123 here."
    assert pr_contract_check.has_declaration_for_issue("123", body) is True
    assert pr_contract_check.has_declaration_for_issue("456", body) is False


@patch("subprocess.run")
def test_check_closes_discipline(mock_run: MagicMock) -> None:
    """Test check_closes_discipline blocks closing epic issues."""
    # Test case 1: Issue has no epic label
    mock_run.return_value = MagicMock(returncode=0, stdout='{"labels": [{"name": "bug"}]}')
    blockers = pr_contract_check.check_closes_discipline("Closes #123", "ll7/robot_sf_ll7")
    assert not blockers

    # Test case 2: Issue has epic label
    mock_run.return_value = MagicMock(returncode=0, stdout='{"labels": [{"name": "epic"}]}')
    blockers = pr_contract_check.check_closes_discipline("Closes #123", "ll7/robot_sf_ll7")
    assert len(blockers) == 1
    assert "epic" in blockers[0]


def test_check_closure_declaration() -> None:
    """Test check_closure_declaration warns on missing declarations."""
    title = "Issue #123: fix bug"
    body_ok = "Closes #123"
    body_bad = "some description without refs"

    assert not pr_contract_check.check_closure_declaration(title, body_ok)
    warnings = pr_contract_check.check_closure_declaration(title, body_bad)
    assert len(warnings) == 1
    assert "closure declaration" in warnings[0]


def test_check_state_refresh_only() -> None:
    """Test check_state_refresh_only blocks state-only updates."""
    title = "State Update"
    body = "closure-audit refresh"
    changed_state_only = ["docs/context/issue_123_state.yaml"]
    changed_code = ["docs/context/issue_123_state.yaml", "robot_sf/sim/core.py"]

    # Blocked: only state files and matching patterns
    blockers = pr_contract_check.check_state_refresh_only(changed_state_only, title, body)
    assert len(blockers) == 1
    assert "touches ONLY docs/context/**" in blockers[0]

    # OK: touches code as well
    blockers = pr_contract_check.check_state_refresh_only(changed_code, title, body)
    assert not blockers


@patch("scripts.ci.pr_contract_check.is_file_new")
@patch("scripts.ci.pr_contract_check.get_new_files")
def test_check_evidence_tree_hygiene(
    mock_new_files: MagicMock, mock_is_new: MagicMock, tmp_path: Path
) -> None:
    """Test check_evidence_tree_hygiene checks new file markers and README claims."""
    mock_new_files.return_value = set()
    mock_is_new.return_value = True

    # Case 1: New file without marker
    f1 = tmp_path / "docs/context/evidence/test_report.md"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_text("Some random contents", encoding="utf-8")

    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f1)], "origin/main")
    assert len(blockers) == 1
    assert "marker convention" in blockers[0]

    # Case 2: New file with marker
    f2 = tmp_path / "docs/context/evidence/test_report2.md"
    f2.write_text("<!-- AI-GENERATED — NEEDS-REVIEW -->\nSome contents", encoding="utf-8")
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f2)], "origin/main")
    assert not blockers

    # Case 3: README claim without provenance
    f3 = tmp_path / "docs/context/evidence/README.md"
    f3.write_text(
        "<!-- AI-GENERATED — NEEDS-REVIEW -->\nThis proves that the model is stable.",
        encoding="utf-8",
    )
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f3)], "origin/main")
    assert len(blockers) == 1
    assert "provenance fields" in blockers[0]

    # Case 4: README claim with provenance
    f4 = tmp_path / "docs/context/evidence/README2.md"
    f4.write_text(
        "<!-- AI-GENERATED — NEEDS-REVIEW -->\nThis proves stability. seeds: 1, config: ppo, hash: abc",
        encoding="utf-8",
    )
    blockers = pr_contract_check.check_evidence_tree_hygiene([str(f4)], "origin/main")
    assert not blockers


@patch("subprocess.run")
def test_check_successor_discipline(mock_run: MagicMock) -> None:
    """Test check_successor_discipline warns on lack of successor statement."""
    # Issue in title has merged PRs, but body lacks successor statement
    title = "Issue #123: title"
    body_no_stmt = "some description"
    body_ok = "This is a successor slice; does not duplicate PR #12"

    # Merge exists
    mock_run.return_value = MagicMock(returncode=0, stdout='[{"number": 12}]')

    warnings = pr_contract_check.check_successor_discipline(title, body_no_stmt, "ll7/robot_sf_ll7")
    assert len(warnings) == 1
    assert "successor statement" in warnings[0]

    warnings = pr_contract_check.check_successor_discipline(title, body_ok, "ll7/robot_sf_ll7")
    assert not warnings


@patch("subprocess.run")
def test_check_worker_lane_provenance(mock_run: MagicMock) -> None:
    """Test check_worker_lane_provenance detects cheap lane and labels PR."""
    body_lane = "This PR was produced by the agy/Gemini-3.5-Flash cheap implementation lane"
    body_normal = "Some normal PR"

    # Lane provenance with PR number
    mock_run.return_value = MagicMock(returncode=0)
    info, labeled = pr_contract_check.check_worker_lane_provenance(
        body_lane, "123", "ll7/robot_sf_ll7"
    )
    assert labeled is True
    assert "Automatically added" in info

    info, labeled = pr_contract_check.check_worker_lane_provenance(
        body_normal, "123", "ll7/robot_sf_ll7"
    )
    assert labeled is False


def test_regression_last_20_merged_prs() -> None:
    """Run regression test on the last 20 merged PRs to ensure zero false blockers."""
    try:
        res = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "merged",
                "--limit",
                "20",
                "--json",
                "number,title,body",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        prs = json.loads(res.stdout)
    except Exception as e:
        pytest.skip(f"Skipping regression test because gh CLI query failed: {e}")
        return

    for pr in prs:
        title = pr.get("title", "")
        body = pr.get("body", "") or ""
        number = pr.get("number")

        try:
            res_files = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/ll7/robot_sf_ll7/pulls/{number}/files?per_page=100",
                    "--jq",
                    ".[].filename",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = [line.strip() for line in res_files.stdout.splitlines() if line.strip()]
        except Exception:
            changed_files = []

        blockers, _, _ = pr_contract_check.run_all_checks(
            title, body, changed_files, "ll7/robot_sf_ll7", "origin/main", str(number)
        )
        assert not blockers, f"PR #{number} ('{title}') triggered false blockers: {blockers}"
