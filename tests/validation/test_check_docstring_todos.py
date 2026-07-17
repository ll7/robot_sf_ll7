"""Tests for placeholder-docstring validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

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


def test_read_backlog_baseline_fails_closed_for_missing_or_invalid_files(tmp_path, capsys):
    """Ratchet mode should explain unusable baseline files instead of crashing."""
    missing = tmp_path / "missing.json"
    baseline_dir = tmp_path / "baseline-dir"
    baseline_dir.mkdir()
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad json}\n", encoding="utf-8")
    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]\n", encoding="utf-8")

    assert check_docstring_todos._read_backlog_baseline(missing, tmp_path) is None
    assert "Run with --mode write-baseline to create it." in capsys.readouterr().err

    assert check_docstring_todos._read_backlog_baseline(baseline_dir, tmp_path) is None
    assert "not found or is a directory" in capsys.readouterr().err

    assert check_docstring_todos._read_backlog_baseline(malformed, tmp_path) is None
    assert "Failed to read baseline file" in capsys.readouterr().err

    assert check_docstring_todos._read_backlog_baseline(non_object, tmp_path) is None
    assert "must contain a JSON object" in capsys.readouterr().err


def test_count_from_source_matches_file_count(tmp_path):
    """Ref-based source counting must match the file-based counter (issue #5858)."""
    src = (
        textwrap.dedent(
            '''
        def needs_doc():
            """TODO docstring."""
            return 1

        class Demo:
            """TODO docstring.

            TODO docstring.
            """
    '''
        ).strip()
        + "\n"
    )
    source_file = tmp_path / "module.py"
    source_file.write_text(src, encoding="utf-8")

    from_file = check_docstring_todos._count_todo_docstrings(source_file)
    from_source = check_docstring_todos._count_todo_docstrings_in_source(src, "module.py")

    assert from_file == from_source == 3


def test_verify_baseline_detects_drift():
    """A stale baseline must be reported as drift, not passed silently (issue #5858)."""
    baseline = {"files": {"scripts/removed.py": 2, "scripts/tool.py": 1}}

    stale_report = {
        "totals": {"files": 1, "total_occurrences": 3},
        "files": {"scripts/tool.py": 3},
    }
    drift, reverse = check_docstring_todos.compare_baseline_drift(stale_report, baseline)

    assert drift == ["scripts/tool.py: base has 3, baseline has 1 (stale by +2)"]
    assert reverse == ["scripts/removed.py: base has 0, baseline has 2 (exceeds by 2)"]


def test_verify_baseline_docs_only_branch_does_not_fail():
    """A docs-only branch identical to base for Python must not fail readiness.

    Reproduces issue #5858: when the base ref and committed baseline agree, a
    branch that changes only non-Python files (docs) must report no drift.
    """
    baseline = {
        "totals": {"files": 1, "total_occurrences": 2},
        "files": {"scripts/tool.py": 2},
    }
    ref_report = {
        "totals": {"files": 1, "total_occurrences": 2},
        "files": {"scripts/tool.py": 2},
    }
    drift, reverse = check_docstring_todos.compare_baseline_drift(ref_report, baseline)

    assert drift == []
    assert reverse == []


def test_committed_baseline_matches_base_ref_no_phantom_keys():
    """The tracked baseline must not over-count files the base ref has already cleaned.

    Regression for issue #5908: the baseline recorded a stale phantom key for
    ``tests/validation/test_pr_contract_check.py`` (baseline 1, base 0) after the
    placeholder cleanup in #5897, which made ``verify-baseline`` fail unrelated
    docs-only PRs. The committed baseline must agree with the base ref so no
    reverse-drift (baseline exceeds base) lines remain.
    """
    repo_root = check_docstring_todos._repo_root()
    baseline = check_docstring_todos._read_backlog_baseline(
        repo_root / check_docstring_todos.DEFAULT_BASELINE_PATH, repo_root
    )
    ref_report = check_docstring_todos.build_backlog_report_for_ref(repo_root, "origin/main")

    drift, reverse = check_docstring_todos.compare_baseline_drift(ref_report, baseline)

    assert drift == [], f"baseline is stale vs base ref: {drift}"
    assert reverse == [], f"baseline exceeds base ref (phantom keys): {reverse}"


def test_committed_baseline_matches_working_tree_backlog(monkeypatch):
    """The tracked baseline must equal a fresh working-tree backlog.

    Reproduces issue #5894: after a cleanup PR removes placeholder docstrings,
    the committed ``docstring_todo_baseline.json`` must be regenerated so the
    ``verify-baseline`` freshness gate stays green. This guards against the
    baseline drifting from the actual file tree when such removals merge.
    """
    repo_root = Path(__file__).resolve().parents[2]
    baseline_path = repo_root / "scripts" / "validation" / "docstring_todo_baseline.json"
    monkeypatch.chdir(repo_root)

    baseline = check_docstring_todos._read_backlog_baseline(baseline_path, repo_root)
    assert baseline is not None

    report = check_docstring_todos.build_backlog_report(
        repo_root,
        roots=check_docstring_todos.DEFAULT_BACKLOG_ROOTS,
    )
    drift, reverse = check_docstring_todos.compare_baseline_drift(report, baseline)

    assert drift == []
    assert reverse == []
