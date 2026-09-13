"""Tests for the issue line-budget check (issue #9094)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dev.check_issue_line_budget import (
    STATUS_INVALID,
    STATUS_NO_CAP,
    STATUS_OVER,
    STATUS_OVERRIDE,
    STATUS_WITHIN,
    evaluate_budget,
    find_override_reason,
    main,
    measure_diffstat,
    parse_declared_caps,
)

if TYPE_CHECKING:
    from pathlib import Path

ISSUE_WITH_CAP = "Reviewability budget: Maximum 10 files and 800 net new lines for this child.\n"
ISSUE_WITH_COMMA_CAP = "Budget of 1,200 net new lines and 5 files.\n"
ISSUE_WITHOUT_CAP = "No budget is declared here; implement as needed.\n"

WITHIN_NUMSTAT = "100\t0\tscripts/dev/a.py\n20\t5\tscripts/dev/b.py\n"
OVER_NUMSTAT = (
    "900\t0\tscripts/dev/a.py\n"
    "300\t0\tscripts/dev/b.py\n"
    "150\t36\tscripts/dev/c.py\n"
    "40\t0\tscripts/dev/d.py\n"
    "24\t0\tscripts/dev/e.py\n"
)


def test_parse_declared_caps_reads_files_and_lines() -> None:
    """The declared caps are parsed from the issue body wording."""
    caps = parse_declared_caps(ISSUE_WITH_CAP)

    assert caps == {"files": 10, "lines": 800}


def test_parse_declared_caps_handles_commas_and_absence() -> None:
    """Comma-grouped numbers parse and uncapped bodies stay inert."""
    assert parse_declared_caps(ISSUE_WITH_COMMA_CAP) == {"files": 5, "lines": 1200}
    assert parse_declared_caps(ISSUE_WITHOUT_CAP) == {"files": None, "lines": None}


def test_parse_declared_caps_ignores_outcome_numbers() -> None:
    """Outcome counts never override the declared cap (issue #9090 phrasing)."""
    body = "Budget note: 1414 net new lines vs the issue's 800-line cap (+614, 77%), 5/10 files.\n"

    assert parse_declared_caps(body) == {"files": 10, "lines": 800}


def test_measure_diffstat_counts_files_additions_and_binary_rows() -> None:
    """Numstat rows produce file/add/delete counts; binary rows only count files."""
    measured = measure_diffstat(WITHIN_NUMSTAT + "-\t-\tassets/logo.png\n")

    assert measured == {"files": 3, "added": 120, "deleted": 5, "net": 115}


def test_evaluate_budget_enforces_net_new_lines() -> None:
    """Deletions reduce the measured line budget rather than being ignored."""
    result = evaluate_budget(
        issue_body=ISSUE_WITH_CAP,
        pr_body="Refs #1\n",
        numstat_text="1000\t500\tscripts/dev/rewrite.py\n",
    )

    assert result["status"] == STATUS_WITHIN
    assert result["ok"] is True


def test_evaluate_budget_within_cap_passes() -> None:
    """A diff inside both caps passes."""
    result = evaluate_budget(
        issue_body=ISSUE_WITH_CAP, pr_body="Refs #1\n", numstat_text=WITHIN_NUMSTAT
    )

    assert result["ok"] is True
    assert result["status"] == STATUS_WITHIN


def test_evaluate_budget_over_cap_fails_with_breach_detail() -> None:
    """An over-cap diff fails and names the breach."""
    result = evaluate_budget(
        issue_body=ISSUE_WITH_CAP, pr_body="Refs #1\n", numstat_text=OVER_NUMSTAT
    )

    assert result["ok"] is False
    assert result["status"] == STATUS_OVER
    assert "1378 net new lines > 800-line cap" in result["breaches"]
    assert "5 files > 10-file cap" not in result["breaches"]


def test_evaluate_budget_override_requires_a_reason() -> None:
    """A reasoned override passes; an empty reason does not."""
    override_body = "Refs #1\nbudget-override: split requested by review; follow-up #2\n"
    applied = evaluate_budget(
        issue_body=ISSUE_WITH_CAP, pr_body=override_body, numstat_text=OVER_NUMSTAT
    )

    assert applied["ok"] is True
    assert applied["status"] == STATUS_OVERRIDE
    assert "follow-up #2" in str(applied["message"])

    empty = evaluate_budget(
        issue_body=ISSUE_WITH_CAP,
        pr_body="Refs #1\nbudget-override:   \n",
        numstat_text=OVER_NUMSTAT,
    )

    assert empty["ok"] is False
    assert empty["status"] == STATUS_OVER
    assert empty["override_reason"] is None


def test_evaluate_budget_without_a_declared_cap_is_inert() -> None:
    """Issues without a declared cap never block."""
    result = evaluate_budget(
        issue_body=ISSUE_WITHOUT_CAP, pr_body="Refs #1\n", numstat_text=OVER_NUMSTAT
    )

    assert result["ok"] is True
    assert result["status"] == STATUS_NO_CAP


def test_evaluate_budget_rejects_unrepresentable_cap() -> None:
    """A malformed numeric cap is a stable fail-closed result, not a crash or no-cap pass."""
    result = evaluate_budget(
        issue_body=f"Maximum {'9' * 5000} lines\n",
        pr_body="Refs #1\n",
        numstat_text=WITHIN_NUMSTAT,
    )

    assert result["status"] == STATUS_INVALID
    assert result["ok"] is False
    assert result["breaches"] == ["declared budget cap is not a valid integer"]


def test_find_override_reason_reads_only_non_empty_lines() -> None:
    """Override parsing requires a non-empty reason on its own line."""
    assert (
        find_override_reason("narrative\nbudget-override: because reasons\n") == "because reasons"
    )
    assert find_override_reason("prose about budget-override: inline\n") is None
    assert find_override_reason("budget-override:\n") is None
    assert find_override_reason("budget-override: <reason>\n") is None


def test_main_exit_codes_and_json_output(tmp_path: Path, capsys) -> None:
    """The CLI returns 0/1 with a JSON payload."""
    issue_file = tmp_path / "issue.md"
    issue_file.write_text(ISSUE_WITH_CAP, encoding="utf-8")
    pr_file = tmp_path / "pr.md"
    numstat_file = tmp_path / "numstat.txt"

    pr_file.write_text("Refs #1\n", encoding="utf-8")
    numstat_file.write_text(WITHIN_NUMSTAT, encoding="utf-8")
    assert (
        main(
            [
                "--issue-body-file",
                str(issue_file),
                "--pr-body-file",
                str(pr_file),
                "--numstat-file",
                str(numstat_file),
            ]
        )
        == 0
    )

    numstat_file.write_text(OVER_NUMSTAT, encoding="utf-8")
    assert (
        main(
            [
                "--issue-body-file",
                str(issue_file),
                "--pr-body-file",
                str(pr_file),
                "--numstat-file",
                str(numstat_file),
            ]
        )
        == 1
    )
    assert '"status": "over_budget"' in capsys.readouterr().out
