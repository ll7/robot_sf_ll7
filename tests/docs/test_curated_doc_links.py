"""Regression tests for the curated documentation link checker (issue #8725)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "check_curated_doc_links.py"

from scripts.dev.check_curated_doc_links import (  # noqa: E402
    check_curated,
    check_target,
    github_slug,
)


def test_github_slug_matches_heading_rules() -> None:
    assert github_slug("Issue State Labels And Dispatch") == "issue-state-labels-and-dispatch"
    assert github_slug("Adding A New Planner!") == "adding-a-new-planner"


def test_missing_file_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.dev.check_curated_doc_links as checker

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    source = tmp_path / "page.md"
    source.write_text("[gone](docs/missing.md)\n", encoding="utf-8")
    finding = check_target(source, 1, "docs/missing.md")
    assert finding is not None and finding.reason == "missing_file"


def test_missing_fragment_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.dev.check_curated_doc_links as checker

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    target = tmp_path / "target.md"
    target.write_text("## Real Heading\n", encoding="utf-8")
    source = tmp_path / "page.md"
    source.write_text("[frag](target.md#no-such-section)\n", encoding="utf-8")
    finding = check_target(source, 1, "target.md#no-such-section")
    assert finding is not None and finding.reason == "missing_fragment"


def test_valid_fragment_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.dev.check_curated_doc_links as checker

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    target = tmp_path / "target.md"
    target.write_text("## Real Heading\n", encoding="utf-8")
    source = tmp_path / "page.md"
    source.write_text("[frag](target.md#real-heading)\n", encoding="utf-8")
    assert check_target(source, 1, "target.md#real-heading") is None


def test_case_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.dev.check_curated_doc_links as checker

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    (tmp_path / "Readme.md").write_text("hi\n", encoding="utf-8")
    source = tmp_path / "page.md"
    source.write_text("[case](readme.md)\n", encoding="utf-8")
    finding = check_target(source, 1, "readme.md")
    assert finding is not None and finding.reason == "case_mismatch"


def test_path_escape_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.dev.check_curated_doc_links as checker

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    nested = tmp_path / "docs"
    nested.mkdir()
    source = nested / "page.md"
    source.write_text("[escape](../../outside.md)\n", encoding="utf-8")
    finding = check_target(source, 1, "../../outside.md")
    assert finding is not None and finding.reason == "path_escape"


def test_generated_alias_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.dev.check_curated_doc_links as checker

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "README.md").write_text("generated\n", encoding="utf-8")
    source = tmp_path / "page.md"
    source.write_text("[gen](examples/README.md)\n", encoding="utf-8")
    finding = check_target(source, 1, "examples/README.md")
    assert finding is not None and finding.reason == "alias_drift"


def test_curated_surface_has_zero_findings() -> None:
    assert check_curated() == []


def test_checker_cli_reports_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--format", "json"],
        capture_output=True,
        text=True,
        cwd=CHECKER.parents[2],
        check=False,
    )
    assert proc.returncode == 0
    payload = json.loads("\n".join(proc.stdout.splitlines()[:-1]))
    assert payload == []
