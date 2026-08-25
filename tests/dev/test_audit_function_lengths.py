"""Tests for the AST-based function-length audit tool (issue #7899)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dev.audit_function_lengths import (
    DEFAULT_THRESHOLD,
    SCHEMA,
    _is_excluded,
    _markdown_summary,
    run_audit,
    run_check,
    scan_file,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "audit_function_lengths.py"

LONG_FUNCTION = "\n".join(f"    x = {i}" for i in range(250))


def _write_fixture(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_inclusive_line_count_pins_semantics(tmp_path: Path) -> None:
    """Inclusive count covers def line through last body line (comments/blanks inside)."""
    path = _write_fixture(
        tmp_path,
        "mod.py",
        '"""Docstring."""\n\n\ndef f() -> None:\n    # comment\n\n    x = 1\n    return x\n',
    )
    findings = scan_file(path, threshold=10, root=tmp_path)
    assert len(findings) == 0  # 5 inclusive lines (3-7) is under 10
    assert findings == []

    path2 = _write_fixture(
        tmp_path,
        "mod2.py",
        "def g() -> None:\n" + "".join(f"    y = {i}\n" for i in range(12)) + "    return y\n",
    )
    findings2 = scan_file(path2, threshold=10, root=tmp_path)
    assert len(findings2) == 1
    finding = findings2[0]
    assert finding.qualified_name == "g"
    assert finding.inclusive_lines == 14
    assert finding.kind == "function"
    assert finding.file_digest


def test_method_and_nested_function_naming(tmp_path: Path) -> None:
    """Methods and nested functions get qualified names with class/outer context."""
    content = """
class C:
    def method(self) -> None:
        x = 1

    async def amethod(self) -> None:
        y = 2

def outer() -> None:
    def inner() -> None:
        z = 3
"""
    path = _write_fixture(tmp_path, "mod.py", content)
    findings = scan_file(path, threshold=0, root=tmp_path)
    names = {finding.qualified_name: finding.kind for finding in findings}
    assert names["C.method"] == "method"
    assert names["C.amethod"] == "async_method"
    assert names["outer"] == "function"
    assert names["outer.inner"] == "function"


def test_decorators_count_toward_function_lines(tmp_path: Path) -> None:
    """Decorated functions are counted from their def line (decorator lines above)."""
    content = "import functools\n\n\n@functools.lru_cache\ndef f() -> None:\n    x = 1\n"
    path = _write_fixture(tmp_path, "mod.py", content)
    findings = scan_file(path, threshold=0, root=tmp_path)
    assert findings[0].start_line == 5  # def line; decorator on line 4 is not counted


def test_syntax_error_fails_closed(tmp_path: Path) -> None:
    """An unparseable file raises SyntaxError."""
    path = _write_fixture(tmp_path, "bad.py", "def broken(:\n")
    with pytest.raises(SyntaxError):
        scan_file(path, threshold=DEFAULT_THRESHOLD, root=tmp_path)


def test_run_audit_reports_over_threshold_sorted(tmp_path: Path) -> None:
    """The audit report lists findings sorted by descending length."""
    _write_fixture(
        tmp_path,
        "a.py",
        "def short():\n    pass\n",
    )
    _write_fixture(
        tmp_path,
        "b.py",
        "def long() -> None:\n" + "".join(f"    v = {i}\n" for i in range(250)),
    )
    report = run_audit(tmp_path, threshold=100)
    assert report["schema"] == SCHEMA
    assert report["findings_count"] == 1
    assert report["findings"][0]["qualified_name"] == "long"
    assert report["scan"]["file_count"] == 2


def test_run_audit_is_byte_stable(tmp_path: Path) -> None:
    """Repeated scans of an unchanged tree are byte-stable."""
    _write_fixture(
        tmp_path,
        "m.py",
        "def f() -> None:\n" + "".join(f"    a = {i}\n" for i in range(220)),
    )
    first = json.dumps(run_audit(tmp_path, threshold=100), sort_keys=True)
    second = json.dumps(run_audit(tmp_path, threshold=100), sort_keys=True)
    assert first == second


def test_exclusion_list_with_rationale(tmp_path: Path) -> None:
    """Versioned exclusions are named with rationale; vendored fragments skipped."""
    _write_fixture(tmp_path, "robot_sf/__init__.py", "from .mod import f\n")
    _write_fixture(
        tmp_path,
        "robot_sf/third_party/gen.py",
        "def big():\n" + "".join(f"    x = {i}\n" for i in range(300)),
    )
    report = run_audit(tmp_path / "robot_sf", threshold=100)
    assert report["scan"]["excluded_count"] == 2
    reasons = {entry["path"]: entry["rationale"] for entry in report["excluded_paths"]}
    assert "package namespace re-exports only" in reasons.get("__init__.py", "")
    assert "vendored/generated fragment" in reasons.get("third_party/gen.py", "")


def test_include_all_scans_excluded(tmp_path: Path) -> None:
    """--include-all (fixture mode) scans even excluded paths."""
    _write_fixture(
        tmp_path,
        "gen.py",
        "def big():\n" + "".join(f"    x = {i}\n" for i in range(300)),
    )
    report = run_audit(tmp_path, threshold=100, include_all=True)
    assert report["findings_count"] == 1


def test_check_allowlist_missing_entry_fails(tmp_path: Path) -> None:
    """A function over threshold without an allowlist entry fails --check."""
    _write_fixture(
        tmp_path,
        "m.py",
        "def big() -> None:\n" + "".join(f"    a = {i}\n" for i in range(220)),
    )
    report = run_audit(tmp_path, threshold=100)
    allowlist: dict[str, int] = {}
    exit_code, problems = run_check(report, allowlist)
    assert exit_code == 1
    assert any("without allowlist entry" in problem for problem in problems)


def test_check_allowlist_growth_fails(tmp_path: Path) -> None:
    """A function that grew past its allowlist count fails --check."""
    _write_fixture(
        tmp_path,
        "m.py",
        "def big() -> None:\n" + "".join(f"    a = {i}\n" for i in range(220)),
    )
    report = run_audit(tmp_path, threshold=100)
    allowlist = {"big": 200}
    exit_code, problems = run_check(report, allowlist)
    assert exit_code == 1
    assert any("grew from 200 to" in problem for problem in problems)


def test_check_allowlist_passes(tmp_path: Path) -> None:
    """Matching allowlist entries pass --check."""
    _write_fixture(
        tmp_path,
        "m.py",
        "def big() -> None:\n" + "".join(f"    a = {i}\n" for i in range(220)),
    )
    report = run_audit(tmp_path, threshold=100)
    allowlist = {"big": 221}
    exit_code, problems = run_check(report, allowlist)
    assert exit_code == 0
    assert problems == []


def test_cli_emits_deterministic_json(tmp_path: Path) -> None:
    """The CLI produces deterministic JSON for a fixture root."""
    _write_fixture(
        tmp_path,
        "m.py",
        "def f() -> None:\n" + "".join(f"    a = {i}\n" for i in range(220)),
    )
    first = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--threshold", "100"],
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--threshold", "100"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["findings_count"] == 1


def test_cli_markdown_summary(tmp_path: Path) -> None:
    """The Markdown summary lists findings in a table."""
    _write_fixture(
        tmp_path,
        "m.py",
        "def f() -> None:\n" + "".join(f"    a = {i}\n" for i in range(220)),
    )
    report = run_audit(tmp_path, threshold=100)
    summary = _markdown_summary(report)
    assert "Function-length audit" in summary
    assert "| `m` | `f` | 221 | function |" in summary


def test_cli_fails_closed_on_syntax_error(tmp_path: Path) -> None:
    """A syntax error in the tree fails the CLI closed."""
    _write_fixture(tmp_path, "bad.py", "def broken(:\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "syntax error" in proc.stdout


def test_is_excluded_matches_exclusions_and_vendored() -> None:
    """Exclusion helper matches exact paths and vendored fragments."""
    assert _is_excluded("robot_sf/__init__.py", {"robot_sf/__init__.py": "why"}) == "why"
    assert _is_excluded("x/third_party/y.py", {}) is not None
    assert _is_excluded("robot_sf/mod.py", {}) is None
