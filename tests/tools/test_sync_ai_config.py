"""Tests for AI assistant configuration mirror checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tools import sync_ai_config


def test_supported_ai_config_mirror_set_comes_from_manifest() -> None:
    """Supported tool mirrors come from the manifest consumed by the sync tool."""
    manifest = sync_ai_config.load_manifest()

    assert sync_ai_config.LINK_SPECS == manifest.symlink_mirrors
    assert (
        sync_ai_config.LinkSpec(".claude/skills", "../.agents/skills") in manifest.symlink_mirrors
    )
    assert sync_ai_config.PointerSpec(".cursorrules", "AGENTS.md") in manifest.pointer_files


def test_load_manifest_rejects_malformed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest shape errors should fail before drift checks run."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    manifest = tmp_path / ".agents" / "mirror_manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text("symlink_mirrors:\n  - path: .codex/skills\n", encoding="utf-8")

    with pytest.raises(ValueError, match="symlink_mirrors\\[0\\].target"):
        sync_ai_config.load_manifest()


def test_check_link_accepts_expected_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_check_link_repairs_missing_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix path should recreate mirrors from canonical `.agents` content."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    target = tmp_path / ".agents" / "skills"
    target.mkdir(parents=True)

    errors = sync_ai_config._check_link(
        sync_ai_config.LinkSpec(".codex/skills", "../.agents/skills"),
        fix=True,
    )

    assert errors == []
    assert (tmp_path / ".codex" / "skills").is_symlink()
    assert (tmp_path / ".codex" / "skills").readlink() == Path("../.agents/skills")


def test_check_link_reports_missing_symlink_without_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check mode reports missing mirrors instead of mutating the tree."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    (tmp_path / ".agents" / "skills").mkdir(parents=True)

    errors = sync_ai_config._check_link(
        sync_ai_config.LinkSpec(".codex/skills", "../.agents/skills"),
        fix=False,
    )

    assert errors == [".codex/skills: missing symlink to ../.agents/skills"]
    assert not (tmp_path / ".codex" / "skills").exists()


def test_check_link_reports_non_symlink_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real directories should never be silently replaced."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    (tmp_path / ".agents" / "skills").mkdir(parents=True)
    (tmp_path / ".codex" / "skills").mkdir(parents=True)

    errors = sync_ai_config._check_link(
        sync_ai_config.LinkSpec(".codex/skills", "../.agents/skills"),
        fix=True,
    )

    assert errors == [".codex/skills: exists but is not a symlink"]


def test_check_link_rejects_paths_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirror paths should not be able to write outside the checkout."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Path escapes repository root"):
        sync_ai_config._check_link(
            sync_ai_config.LinkSpec("../outside", ".agents/skills"),
            fix=True,
        )


def test_check_link_rejects_targets_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirror targets should stay inside the checkout."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Symlink target escapes repository root"):
        sync_ai_config._check_link(
            sync_ai_config.LinkSpec(".codex/skills", "../../outside"),
            fix=True,
        )


def test_check_link_repairs_stale_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix mode should replace stale symlink targets."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    (tmp_path / ".agents" / "skills").mkdir(parents=True)
    link = tmp_path / ".codex" / "skills"
    link.parent.mkdir()
    link.symlink_to(Path("../wrong"))

    errors = sync_ai_config._check_link(
        sync_ai_config.LinkSpec(".codex/skills", "../.agents/skills"),
        fix=True,
    )

    assert errors == []
    assert link.readlink() == Path("../.agents/skills")


def test_check_pointer_file_accepts_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointer files should mention their canonical instruction source."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    pointer = tmp_path / ".github" / "copilot-instructions.md"
    pointer.parent.mkdir()
    pointer.write_text("Follow AGENTS.md.\n", encoding="utf-8")

    assert sync_ai_config._check_pointer_file(".github/copilot-instructions.md", "AGENTS.md") == []


def test_check_pointer_file_reports_missing_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointer files should fail when they drift away from the canonical source."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    pointer = tmp_path / ".github" / "copilot-instructions.md"
    pointer.parent.mkdir()
    pointer.write_text("Only local instructions.\n", encoding="utf-8")

    assert sync_ai_config._check_pointer_file(".github/copilot-instructions.md", "AGENTS.md") == [
        ".github/copilot-instructions.md: expected to reference 'AGENTS.md'"
    ]


def test_check_pointer_file_reports_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory at a pointer path should report a drift error instead of raising."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    (tmp_path / ".cursorrules").mkdir()

    assert sync_ai_config._check_pointer_file(".cursorrules", "AGENTS.md") == [
        ".cursorrules: pointer path exists but is not a file"
    ]
