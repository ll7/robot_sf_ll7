"""Tests for the docs/proof consistency checker."""

from __future__ import annotations

from pathlib import Path

from scripts.validation.check_docs_proof_consistency import (
    ChangedFile,
    _collect_diagnostics,
    _context_readme_link_diagnostics,
)


def test_missing_context_note_index_link_is_reported() -> None:
    """New top-level context notes should be linked from the context README."""
    diagnostics = _context_readme_link_diagnostics(
        [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))],
        context_readme_text="# Context Notes Workflow\n",
    )

    assert diagnostics == [
        type(diagnostics[0])(
            path=Path("docs/context/issue_999_example.md"),
            message="added context note is not linked from docs/context/README.md",
        )
    ]


def test_absolute_local_path_in_tracked_evidence_is_reported(tmp_path: Path) -> None:
    """Tracked evidence should not preserve machine-local absolute paths."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n", encoding="utf-8"
    )
    evidence = repo_root / "docs/context/evidence/report.md"
    evidence.write_text("Artifact came from /home/user/output/report.json.\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/evidence/report.md"))],
        repo_root=repo_root,
    )

    assert any(
        "absolute local filesystem paths" in diagnostic.message for diagnostic in diagnostics
    )


def test_output_pointer_in_tracked_evidence_is_reported(tmp_path: Path) -> None:
    """Tracked evidence should not link to ignored output artifacts."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n", encoding="utf-8"
    )
    evidence = repo_root / "docs/context/evidence/report.md"
    evidence.write_text("[artifact](output/reports/run.json)\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/evidence/report.md"))],
        repo_root=repo_root,
    )

    assert any("ignored output/ artifacts" in diagnostic.message for diagnostic in diagnostics)


def test_context_note_output_command_is_allowed(tmp_path: Path) -> None:
    """Regular context notes may still mention output paths in reproducible commands."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text(
        "Run `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` and inspect `output/coverage/coverage.json` locally.\n",
        encoding="utf-8",
    )

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_stale_no_validation_phrase_with_commands_is_reported(tmp_path: Path) -> None:
    """Notes should not claim no validation ran while listing executed commands."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text(
        "No validation commands were run during this pass.\n\n"
        "- `uv run pytest tests/test_cli.py -q`\n",
        encoding="utf-8",
    )

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert any(
        "no validation commands were run" in diagnostic.message for diagnostic in diagnostics
    )
