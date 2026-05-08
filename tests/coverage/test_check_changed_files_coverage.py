"""Tests for changed-files coverage gate filtering."""

from __future__ import annotations

from scripts.coverage.check_changed_files_coverage import _is_doc_or_comment_only_python_change


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
