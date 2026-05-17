"""Tests for placeholder-docstring validation."""

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


def test_backlog_report_counts_todo_docstrings_by_area_and_file(tmp_path):
    """Backlog reporting should aggregate placeholder debt without using git diff state."""
    robot_file = tmp_path / "robot_sf" / "module.py"
    script_file = tmp_path / "scripts" / "tool.py"
    robot_file.parent.mkdir()
    script_file.parent.mkdir()
    robot_file.write_text(
        textwrap.dedent(
            '''
            def needs_doc():
                """TODO docstring."""
                return 1

            def has_doc():
                """Real docstring."""
                return 2
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    script_file.write_text(
        textwrap.dedent(
            '''
            class Demo:
                """TODO docstring.

                TODO docstring.
                """
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    report = check_docstring_todos.build_backlog_report(
        tmp_path,
        roots=("robot_sf", "scripts"),
    )

    assert report["totals"]["total_occurrences"] == 3
    assert report["areas"]["robot_sf"]["files"] == 1
    assert report["areas"]["robot_sf"]["occurrences"] == 1
    assert report["areas"]["scripts"]["files"] == 1
    assert report["areas"]["scripts"]["occurrences"] == 2
    assert report["files"]["robot_sf/module.py"] == 1
    assert report["files"]["scripts/tool.py"] == 2


def test_backlog_ratchet_detects_increased_and_new_files():
    """Ratchet comparison should fail only when file-level debt increases."""
    baseline = {
        "files": {
            "robot_sf/module.py": 1,
            "scripts/old.py": 3,
        }
    }
    current = {
        "files": {
            "robot_sf/module.py": 2,
            "scripts/new.py": 1,
            "scripts/old.py": 2,
        }
    }

    increases = check_docstring_todos.compare_backlog_to_baseline(current, baseline)

    assert increases == [
        "robot_sf/module.py: 2 TODO docstring occurrences (baseline 1, +1)",
        "scripts/new.py: 1 TODO docstring occurrences (baseline 0, +1)",
    ]
