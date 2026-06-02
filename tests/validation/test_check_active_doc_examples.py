"""Tests for active documentation example validation."""

from __future__ import annotations

from pathlib import Path

from scripts.validation import check_active_doc_examples as checker


def test_scan_file_reports_known_bad_active_doc_patterns(tmp_path: Path) -> None:
    """Known stale command and artifact examples should be reported."""
    doc = tmp_path / "docs" / "quickstart.md"
    doc.parent.mkdir()
    doc.write_text(
        "\n".join(
            [
                "Run `python scripts/missing_tool.py --demo`.",
                "Old output is under `output/results/demo.json`.",
                "Use `robot_sf_bench run --suite core --episodes 10`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    diagnostics = checker.scan_file(Path("docs/quickstart.md"), tmp_path)

    rules = {diagnostic.rule for diagnostic in diagnostics}
    assert "bare-python-scripts-command" in rules
    assert "nested-output-results-path" in rules
    assert "unsupported-robot-sf-bench-run-flag" in rules
    assert "phantom-script-path" in rules


def test_scan_file_allows_marked_and_historical_lines(tmp_path: Path) -> None:
    """Historical mentions and explicit allows should not fail active docs."""
    doc = tmp_path / "docs" / "README.md"
    doc.parent.mkdir()
    doc.write_text(
        "Legacy `results/` paths have been migrated.\n"
        "`python scripts/old_example.py` <!-- active-docs-check: allow -->\n",
        encoding="utf-8",
    )

    diagnostics = checker.scan_file(Path("docs/README.md"), tmp_path)

    assert diagnostics == []


def test_default_paths_exclude_context_notes_and_include_spec_quickstarts(tmp_path: Path) -> None:
    """Default path selection should separate active docs from historical context notes."""
    (tmp_path / "docs" / "context").mkdir(parents=True)
    (tmp_path / "docs" / "context" / "issue_1.md").write_text("results/run.json\n")
    (tmp_path / "docs" / "README.md").write_text("hello\n")
    (tmp_path / "specs" / "001-demo").mkdir(parents=True)
    (tmp_path / "specs" / "001-demo" / "quickstart.md").write_text("hello\n")

    without_specs = checker._default_paths(
        tmp_path,
        include_specs=False,
        include_cli_sources=False,
    )
    with_specs = checker._default_paths(
        tmp_path,
        include_specs=True,
        include_cli_sources=False,
    )

    assert Path("docs/README.md") in without_specs
    assert Path("docs/context/issue_1.md") not in without_specs
    assert Path("specs/001-demo/quickstart.md") in with_specs


def test_main_fail_on_diagnostic_controls_exit_code(tmp_path: Path, monkeypatch) -> None:
    """The checker can report by default and fail when CI asks it to."""
    doc = tmp_path / "docs" / "quickstart.md"
    doc.parent.mkdir()
    doc.write_text("Run `python scripts/missing_tool.py`.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(checker, "_repo_root", lambda: tmp_path)

    assert checker.main(["docs/quickstart.md"]) == 0
    assert checker.main(["--fail-on-diagnostic", "docs/quickstart.md"]) == 1
