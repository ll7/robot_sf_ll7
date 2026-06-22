"""Tests for changed-files coverage gate filtering."""

from __future__ import annotations

from scripts.coverage.check_changed_files_coverage import (
    _coverage_for_changed_lines,
    _is_doc_or_comment_only_python_change,
)


def test_doc_or_comment_only_python_change_detection() -> None:
    """Docstrings and comments should not require behavior coverage."""
    before = '''"""Old module docs."""

def answer() -> int:
    """Old helper docs."""
    return 1
'''
    after = '''"""New module docs."""

# Expanded context for maintainers.
def answer() -> int:
    """New helper docs."""
    # The behavior is unchanged.
    return 1
'''

    assert _is_doc_or_comment_only_python_change(before, after)
    assert not _is_doc_or_comment_only_python_change(before, after.replace("return 1", "return 2"))


def test_changed_line_coverage_uses_only_executable_changed_statements() -> None:
    """Large legacy files should not fail when the edited executable lines are covered."""
    file_data = {
        "executed_lines": [2, 5, 20],
        "missing_lines": [6, 21, 22],
    }

    coverage, scope = _coverage_for_changed_lines(
        file_data=file_data,
        changed_lines={1, 2, 5, 6},
    )

    assert coverage == 100.0 * 2 / 3
    assert scope == "changed executable lines 2/3"


def test_changed_line_coverage_treats_non_executable_edits_as_covered() -> None:
    """Comment-only diff lines that survive AST filtering should not create coverage debt."""
    coverage, scope = _coverage_for_changed_lines(
        file_data={"executed_lines": [10], "missing_lines": [20]},
        changed_lines={1, 2},
    )

    assert coverage == 100.0
    assert scope == "changed executable lines 0/0"
