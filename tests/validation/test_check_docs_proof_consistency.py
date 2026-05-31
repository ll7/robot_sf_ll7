"""Tests for the docs/proof consistency checker."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from scripts.validation.check_docs_proof_consistency import (
    ChangedFile,
    _collect_diagnostics,
    _context_catalog_diagnostics,
    _context_readme_link_diagnostics,
    _parse_name_status,
    _selected_files,
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


def test_context_note_anchor_link_counts_as_index_link() -> None:
    """Anchored links should satisfy the context-note discoverability check."""
    diagnostics = _context_readme_link_diagnostics(
        [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))],
        context_readme_text="- [Issue 999 Example](issue_999_example.md#validation)\n",
    )

    assert diagnostics == []


def test_context_note_same_basename_in_subdir_does_not_count_as_index_link() -> None:
    """A link to another note with the same basename should not index a top-level note."""
    diagnostics = _context_readme_link_diagnostics(
        [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))],
        context_readme_text="- [Archived Example](archive/issue_999_example.md)\n",
    )

    assert any(
        "not linked from docs/context/README.md" in diagnostic.message for diagnostic in diagnostics
    )


def test_copied_context_note_requires_index_link() -> None:
    """Copied top-level context notes should be handled like added notes."""
    diagnostics = _context_readme_link_diagnostics(
        [ChangedFile(status="C100", path=Path("docs/context/issue_999_example.md"))],
        context_readme_text="# Context Notes Workflow\n",
    )

    assert any(
        "not linked from docs/context/README.md" in diagnostic.message for diagnostic in diagnostics
    )


def test_parse_name_status_uses_new_path_for_rename() -> None:
    """Rename rows should track the current worktree path, not the old path pair."""
    changed = _parse_name_status("R100\tdocs/context/old.md\tdocs/context/new.md\n")

    assert changed == [ChangedFile(status="R100", path=Path("docs/context/new.md"))]


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


def test_inline_output_pointer_in_markdown_evidence_is_reported(tmp_path: Path) -> None:
    """Markdown evidence should flag inline ignored-output pointers, not only links."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n", encoding="utf-8"
    )
    evidence = repo_root / "docs/context/evidence/report.md"
    evidence.write_text("Manifest: `output/reports/run.json`\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/evidence/report.md"))],
        repo_root=repo_root,
    )

    assert any("ignored output/ artifacts" in diagnostic.message for diagnostic in diagnostics)


def test_prefix_sibling_of_evidence_dir_is_not_treated_as_tracked_evidence(tmp_path: Path) -> None:
    """Only the real evidence directory should trigger durable-evidence checks."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence_backup").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n", encoding="utf-8"
    )
    note = repo_root / "docs/context/evidence_backup/report.md"
    note.write_text("[artifact](output/reports/run.json)\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/evidence_backup/report.md"))],
        repo_root=repo_root,
    )

    assert diagnostics == []


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


def test_line_start_issue_reference_is_reported(tmp_path: Path) -> None:
    """Markdown issue references should not start lines as bare #123 tokens."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text("#1108 needs a clearer reference style.\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert any("Issue #1108" in diagnostic.message for diagnostic in diagnostics)


def test_preferred_issue_reference_form_is_allowed(tmp_path: Path) -> None:
    """The preferred prose form should satisfy the issue-reference style guard."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text("Issue #1108 keeps this link readable in Markdown.\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_issue_reference_guard_handles_bullets_tables_and_headings(tmp_path: Path) -> None:
    """Flag bare issue references in lists/tables without banning numeric headings."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text(
        "# 2026 plan\n\n"
        "- #1108 should use prose.\n"
        "| #1262 | missing prefix |\n"
        "```\n"
        "#999 is only an example inside a code block\n"
        "```\n",
        encoding="utf-8",
    )

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    messages = [diagnostic.message for diagnostic in diagnostics]
    assert any("Issue #1108" in message for message in messages)
    assert any("Issue #1262" in message for message in messages)
    assert not any("Issue #999" in message for message in messages)


def test_issue_reference_guard_ignores_indented_code_blocks(tmp_path: Path) -> None:
    """Four-space indented Markdown code blocks should not be prose diagnostics."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text("    #1108 is literal code, not prose.\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_issue_reference_guard_handles_indented_long_fences(tmp_path: Path) -> None:
    """Markdown fences can be indented up to three spaces and longer than three ticks."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text(
        "   ````python\n"
        "#1108 is only sample code.\n"
        "   ````\n"
        "#1262 is prose and should be flagged.\n",
        encoding="utf-8",
    )

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    messages = [diagnostic.message for diagnostic in diagnostics]
    assert not any("Issue #1108" in message for message in messages)
    assert any("Issue #1262" in message for message in messages)


def test_explicit_path_new_context_note_is_treated_as_added(tmp_path: Path) -> None:
    """Explicit path checks should still enforce added-note index-link validation."""
    repo_root = tmp_path
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "docs/context/README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "test fixture"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text("Fresh note body.\n", encoding="utf-8")

    selected = _selected_files(
        argparse.Namespace(path=[str(note)], base="HEAD"),
        repo_root=repo_root,
    )
    diagnostics = _collect_diagnostics(selected, repo_root=repo_root)

    assert selected == [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))]
    assert any(
        "not linked from docs/context/README.md" in diagnostic.message for diagnostic in diagnostics
    )


def test_explicit_path_outside_repo_is_rejected(tmp_path: Path) -> None:
    """Explicit path checks should not read files outside the repository root."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside_path = tmp_path / "outside.md"

    with pytest.raises(ValueError, match="repository root"):
        _selected_files(
            argparse.Namespace(path=[str(outside_path)], base="HEAD"),
            repo_root=repo_root,
        )


def test_context_catalog_reports_missing_status_and_path(tmp_path: Path) -> None:
    """Catalog entries need existing paths plus status and freshness metadata."""
    repo_root = tmp_path
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        """
version: 1
entries:
  - path: docs/context/missing.md
    freshness: maintained
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )
    messages = [diagnostic.message for diagnostic in diagnostics]

    assert any("path does not exist" in message for message in messages)
    assert any("status must be one of" in message for message in messages)


def test_context_catalog_accepts_valid_entry(tmp_path: Path) -> None:
    """A complete catalog row with declared metadata should pass."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/current.md").write_text("current\n", encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
status_values:
  current: Active source of truth.
  historical: Background.
  superseded: Replaced.
  evidence: Proof.
  proposal: Planned.
freshness_values:
  maintained: Update when changed.
  dated: Date-scoped.
  policy: Durable boundary.
  evidence: Evidence pointer.
entries:
  - path: docs/context/current.md
    status: current
    freshness: maintained
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_context_catalog_rejects_unknown_freshness(tmp_path: Path) -> None:
    """Freshness metadata should use the catalog vocabulary."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/current.md").write_text("current\n", encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
freshness_values:
  maintained: Update when changed.
entries:
  - path: docs/context/current.md
    status: current
    freshness: stale
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert any("freshness must be one of" in diagnostic.message for diagnostic in diagnostics)


def test_context_catalog_requires_superseded_replacement(tmp_path: Path) -> None:
    """Superseded catalog entries must name an existing replacement."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/old.md").write_text("old\n", encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
entries:
  - path: docs/context/old.md
    status: superseded
    freshness: dated
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert any("replacement is required" in diagnostic.message for diagnostic in diagnostics)


def test_context_catalog_evidence_rejects_output_pointer(tmp_path: Path) -> None:
    """Evidence catalog entries should not depend on ignored output paths."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    evidence = repo_root / "docs/context/evidence/report.md"
    evidence.write_text("Durable evidence: output/local/report.json\n", encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
entries:
  - path: docs/context/evidence/report.md
    status: evidence
    freshness: evidence
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert any("ignored output/ artifacts" in diagnostic.message for diagnostic in diagnostics)


def test_context_catalog_skips_binary_evidence_scan(tmp_path: Path) -> None:
    """Binary evidence entries should not crash text-only provenance checks."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    evidence = repo_root / "docs/context/evidence/frame.png"
    evidence.write_bytes(b"\x89PNG\r\n\x1a\n\x80\x81")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
entries:
  - path: docs/context/evidence/frame.png
    status: evidence
    freshness: evidence
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_context_catalog_reports_malformed_yaml(tmp_path: Path) -> None:
    """Malformed catalog YAML should return a diagnostic instead of a traceback."""
    repo_root = tmp_path
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("version: [\n", encoding="utf-8")

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert len(diagnostics) == 1
    assert "context catalog is not a valid YAML file" in diagnostics[0].message
