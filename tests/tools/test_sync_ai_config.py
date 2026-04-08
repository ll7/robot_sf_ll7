"""Tests for AI assistant configuration mirror checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tools import sync_ai_config


def test_check_link_accepts_expected_symlink(tmp_path: Path, monkeypatch) -> None:
    """An exact compatibility symlink should be accepted without errors."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    target = tmp_path / ".agents" / "skills"
    target.mkdir(parents=True)
    link = tmp_path / ".codex" / "skills"
    link.parent.mkdir()
    link.symlink_to(Path("../.agents/skills"))

    errors = sync_ai_config._check_link(
        sync_ai_config.LinkSpec(".codex/skills", "../.agents/skills"),
        fix=False,
    )

    assert errors == []


def test_check_link_repairs_missing_symlink(tmp_path: Path, monkeypatch) -> None:
    """The fix path should recreate supported mirrors from canonical `.agents` content."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    (tmp_path / ".agents" / "prompts" / "codex").mkdir(parents=True)

    errors = sync_ai_config._check_link(
        sync_ai_config.LinkSpec(".codex/prompts", "../.agents/prompts/codex"),
        fix=True,
    )

    assert errors == []
    assert (tmp_path / ".codex" / "prompts").is_symlink()
    assert (tmp_path / ".codex" / "prompts").readlink() == Path("../.agents/prompts/codex")


def test_check_link_rejects_paths_outside_repo(tmp_path: Path, monkeypatch) -> None:
    """Repo mirror specs should not be able to write outside the checkout."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="escapes repository root"):
        sync_ai_config._check_link(
            sync_ai_config.LinkSpec("../outside", "target"),
            fix=True,
        )


def test_pointer_file_must_reference_canonical_agents_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Pointer files should direct agents back to the canonical `AGENTS.md` surface."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    cursor_rules = tmp_path / ".cursorrules"
    cursor_rules.write_text("See AGENTS.md.\n", encoding="utf-8")

    assert sync_ai_config._check_pointer_file(".cursorrules", "AGENTS.md") == []

    cursor_rules.write_text("See .github/copilot-instructions.md.\n", encoding="utf-8")
    assert sync_ai_config._check_pointer_file(".cursorrules", "AGENTS.md") == [
        ".cursorrules: expected to reference 'AGENTS.md'"
    ]
