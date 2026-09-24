"""Tests for the issue line-budget check (issue #9094)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.dev.check_issue_line_budget import (
    STATUS_INVALID,
    STATUS_NO_CAP,
    STATUS_OVER,
    STATUS_OVERRIDE,
    STATUS_WITHIN,
    DiffstatParseError,
    evaluate_budget,
    find_override_reason,
    has_declared_cap,
    main,
    measure_diffstat,
    parse_budget_dimensions,
    parse_context_budget,
    parse_declared_caps,
    split_budget_text,
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

    numstat_file.write_text("malformed row without tabs\n", encoding="utf-8")
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
    assert '"status": "invalid_cap"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "bad_line",
    [
        "malformed row without tabs",
        "10\t2",  # missing path
        "10\t2\t",  # empty path
        "invalid\t0\tscripts/dev/a.py",  # non-numeric additions
        "0\tinvalid\tscripts/dev/a.py",  # non-numeric deletions
    ],
)
def test_measure_diffstat_rejects_malformed_lines(bad_line: str) -> None:
    """Non-empty lines that fail git numstat structure raise DiffstatParseError."""
    with pytest.raises(DiffstatParseError, match="malformed numstat line"):
        measure_diffstat(bad_line + "\n")


def test_evaluate_budget_rejects_malformed_numstat() -> None:
    """A corrupt numstat text produces a fail-closed STATUS_INVALID result."""
    result = evaluate_budget(
        issue_body=ISSUE_WITH_CAP,
        pr_body="Refs #1\n",
        numstat_text="not a valid numstat line\n",
    )

    assert result["status"] == STATUS_INVALID
    assert result["ok"] is False
    assert any("malformed numstat line" in breach for breach in result["breaches"])


ISSUE_9279_CONTEXT_BUDGET = (
    "## Bounded agent task packet\n"
    "• Context budget: initially at most 12 files / 16,000 input tokens: this issue, "
    "contract owner, listed canonical owners and focused tests; expand only for a "
    "named unresolved symbol, not broad repository discovery.\n"
)

ISSUE_WITH_BOTH_BUDGETS = (
    "## Reviewability budget\n"
    "Maximum 10 files and 800 net new lines for this child.\n"
    "\n"
    "## Bounded agent task packet\n"
    "• Context budget: initially at most 12 files / 16,000 input tokens: this issue, "
    "contract owner, listed canonical owners and focused tests...\n"
)

NUMSTAT_38_FILES = "".join(f"5\t0\tscripts/dev/frame_{i:02d}.py\n" for i in range(38))
NUMSTAT_11_FILES = "".join(f"10\t0\tscripts/dev/module_{i:02d}.py\n" for i in range(11))


def test_context_budget_is_not_treated_as_pr_diff_cap() -> None:
    """A reading-context budget in files/tokens is not treated as a PR diff cap (issue #9641)."""
    caps = parse_declared_caps(ISSUE_9279_CONTEXT_BUDGET)
    assert caps == {"files": None, "lines": None}
    assert has_declared_cap(ISSUE_9279_CONTEXT_BUDGET) is False

    context = parse_context_budget(ISSUE_9279_CONTEXT_BUDGET)
    assert context == {"files": 12, "tokens": 16000}

    dimensions = parse_budget_dimensions(ISSUE_9279_CONTEXT_BUDGET)
    assert dimensions == {
        "diff": {"files": None, "lines": None},
        "context": {"files": 12, "tokens": 16000},
    }

    # A PR touching 38 files is within budget because no PR diff cap was declared.
    result = evaluate_budget(
        issue_body=ISSUE_9279_CONTEXT_BUDGET,
        pr_body="Refs #9279\n",
        numstat_text=NUMSTAT_38_FILES,
    )
    assert result["status"] == STATUS_NO_CAP
    assert result["ok"] is True
    assert result["breaches"] == []
    assert result["context_budget"] == {"files": 12, "tokens": 16000}


def test_issue_with_both_budgets_enforces_pr_diff_cap() -> None:
    """An issue with both context budget and PR diff cap enforces the real diff cap (issue #9641)."""
    caps = parse_declared_caps(ISSUE_WITH_BOTH_BUDGETS)
    assert caps == {"files": 10, "lines": 800}
    assert has_declared_cap(ISSUE_WITH_BOTH_BUDGETS) is True

    context = parse_context_budget(ISSUE_WITH_BOTH_BUDGETS)
    assert context == {"files": 12, "tokens": 16000}

    dimensions = parse_budget_dimensions(ISSUE_WITH_BOTH_BUDGETS)
    assert dimensions == {
        "diff": {"files": 10, "lines": 800},
        "context": {"files": 12, "tokens": 16000},
    }

    # 11 files exceeds the 10-file diff cap (even though it is within the 12-file context budget).
    result = evaluate_budget(
        issue_body=ISSUE_WITH_BOTH_BUDGETS,
        pr_body="Refs #1\n",
        numstat_text=NUMSTAT_11_FILES,
    )
    assert result["status"] == STATUS_OVER
    assert result["ok"] is False
    assert "11 files > 10-file cap" in result["breaches"]

    # Reasoned override permits merge when real cap is exceeded.
    overridden = evaluate_budget(
        issue_body=ISSUE_WITH_BOTH_BUDGETS,
        pr_body="Refs #1\nbudget-override: synthetic test frames required for coverage\n",
        numstat_text=NUMSTAT_11_FILES,
    )
    assert overridden["status"] == STATUS_OVERRIDE
    assert overridden["ok"] is True


def test_context_budget_phrasings_and_markdown_sections() -> None:
    """Various reading-context phrasings and section structures are recognized."""
    reversed_wording = "Context budget: 16,000 input tokens / 12 files\n"
    assert parse_declared_caps(reversed_wording) == {"files": None, "lines": None}
    assert parse_context_budget(reversed_wording) == {"files": 12, "tokens": 16000}

    review_context = "Review-context budget: 8 files and 10,000 tokens.\n"
    assert parse_declared_caps(review_context) == {"files": None, "lines": None}
    assert parse_context_budget(review_context) == {"files": 8, "tokens": 10000}

    section_body = (
        "## Context budget\n"
        "Initially at most 6 files and 20,000 input tokens.\n"
        "\n"
        "## Reviewability budget\n"
        "Maximum 4 files and 300 net new lines.\n"
    )
    assert parse_declared_caps(section_body) == {"files": 4, "lines": 300}
    assert parse_context_budget(section_body) == {"files": 6, "tokens": 20000}

    block_body = "• Context budget:\n  - at most 15 files\n  - 32,000 tokens\n"
    assert parse_declared_caps(block_body) == {"files": None, "lines": None}
    assert parse_context_budget(block_body) == {"files": 15, "tokens": 32000}


def test_split_budget_text_partitions_correctly() -> None:
    """split_budget_text isolates context budget from general diff cap text."""
    body = (
        "Intro text\n"
        "• Context budget: initially at most 12 files / 16,000 input tokens\n"
        "Reviewability budget: Maximum 10 files and 800 net new lines\n"
    )
    diff_text, context_text = split_budget_text(body)
    assert "Context budget" not in diff_text
    assert "Reviewability budget" in diff_text
    assert "Context budget" in context_text
    assert "Reviewability budget" not in context_text


def test_evaluate_budget_rejects_unrepresentable_context_cap() -> None:
    """An unrepresentable numeric cap in context budget raises a fail-closed STATUS_INVALID."""
    result = evaluate_budget(
        issue_body=f"Context budget: at most {'9' * 5000} files\n",
        pr_body="Refs #1\n",
        numstat_text=WITHIN_NUMSTAT,
    )
    assert result["status"] == STATUS_INVALID
    assert result["ok"] is False
    assert result["breaches"] == ["declared budget cap is not a valid integer"]
