"""Tests for the diff-only TODO docstring checker."""

from __future__ import annotations

import textwrap

from scripts.validation import check_docstring_todos


def test_read_defs_handles_syntax_error(tmp_path):
    """Invalid Python input should be skipped without raising errors."""
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("def broken(:\n", encoding="utf-8")

    defs = check_docstring_todos._read_defs(bad_file)

    assert defs == []


def test_read_defs_collects_docstrings(tmp_path):
    """Definitions with docstrings are collected from valid source."""
    good_file = tmp_path / "good.py"
    good_file.write_text(
        textwrap.dedent(
            '''
            class Demo:
                """Demo class."""

                def method(self):
                    """Demo method."""
                    return 1

            def func():
                """Demo function."""
                return 2
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    defs = check_docstring_todos._read_defs(good_file)
    names = {d.name for d in defs}

    assert "Demo" in names
    assert "Demo.method" in names
    assert "func" in names
