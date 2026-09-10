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

    spec = sync_ai_config.PointerSpec(".github/copilot-instructions.md", "AGENTS.md")
    assert sync_ai_config._check_pointer_file(spec) == []


def test_check_pointer_file_reports_missing_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointer files should fail when they drift away from the canonical source."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    pointer = tmp_path / ".github" / "copilot-instructions.md"
    pointer.parent.mkdir()
    pointer.write_text("Only local instructions.\n", encoding="utf-8")

    spec = sync_ai_config.PointerSpec(".github/copilot-instructions.md", "AGENTS.md")
    assert sync_ai_config._check_pointer_file(spec) == [
        ".github/copilot-instructions.md: expected to reference 'AGENTS.md'"
    ]


def test_check_pointer_file_reports_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory at a pointer path should report a drift error instead of raising."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    (tmp_path / ".cursorrules").mkdir()

    spec = sync_ai_config.PointerSpec(".cursorrules", "AGENTS.md")
    assert sync_ai_config._check_pointer_file(spec) == [
        ".cursorrules: pointer path exists but is not a file"
    ]


def test_check_pointer_file_reports_oversized_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pointer over the configured line budget should fail with a policy-home hint."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    pointer = tmp_path / ".github" / "copilot-instructions.md"
    pointer.parent.mkdir()
    pointer.write_text(
        "Follow AGENTS.md.\n" + "\n".join(f"policy line {i}" for i in range(12)),
        encoding="utf-8",
    )

    spec = sync_ai_config.PointerSpec(
        ".github/copilot-instructions.md", "AGENTS.md", max_nonblank_lines=10
    )
    errors = sync_ai_config._check_pointer_file(spec)

    assert len(errors) == 1
    assert "exceeds budget 10" in errors[0]
    assert "canonical source" in errors[0]


def test_check_pointer_file_accepts_within_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pointer within the configured line budget should pass."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    pointer = tmp_path / ".github" / "copilot-instructions.md"
    pointer.parent.mkdir()
    pointer.write_text(
        "Follow AGENTS.md.\n" + "\n".join(f"note {i}" for i in range(5)),
        encoding="utf-8",
    )

    spec = sync_ai_config.PointerSpec(
        ".github/copilot-instructions.md", "AGENTS.md", max_nonblank_lines=10
    )
    assert sync_ai_config._check_pointer_file(spec) == []


def test_check_pointer_file_reports_forbidden_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copied canonical policy section in a pointer should fail."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    pointer = tmp_path / ".github" / "copilot-instructions.md"
    pointer.parent.mkdir()
    pointer.write_text(
        "Follow AGENTS.md.\n\n## Test Failure Evaluation\n\nClassify failures first.\n",
        encoding="utf-8",
    )

    spec = sync_ai_config.PointerSpec(
        ".github/copilot-instructions.md",
        "AGENTS.md",
        forbidden_sections=("test failure evaluation",),
    )
    errors = sync_ai_config._check_pointer_file(spec)

    assert len(errors) == 1
    assert "forbidden duplicated section 'test failure evaluation'" in errors[0]


def test_check_pointer_file_allows_provider_specific_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An allowed provider-specific section should not fail the pointer."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    pointer = tmp_path / ".claude" / "CLAUDE.md"
    pointer.parent.mkdir()
    pointer.write_text(
        "Follow AGENTS.md.\n\n## Claude Code Model and Mode Selection\n\nDefault Opus.\n",
        encoding="utf-8",
    )

    spec = sync_ai_config.PointerSpec(
        ".claude/CLAUDE.md",
        "AGENTS.md",
        allowed_sections=("claude code model and mode selection",),
        forbidden_sections=("test failure evaluation",),
    )
    assert sync_ai_config._check_pointer_file(spec) == []


def _write_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    """Create a temporary claude adapter file for scope-boundary fixtures."""
    monkeypatch.setattr(sync_ai_config, "REPO_ROOT", tmp_path)
    pointer = tmp_path / ".claude" / "CLAUDE.md"
    pointer.parent.mkdir()
    pointer.write_text(body, encoding="utf-8")


def test_check_pointer_file_rejects_unknown_renamed_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renamed policy section cannot bypass the adapter boundary."""
    _write_pointer(
        tmp_path,
        monkeypatch,
        "Follow AGENTS.md.\n\n## Repository Rules\n\nNever push directly.\n",
    )
    spec = sync_ai_config.PointerSpec(
        ".claude/CLAUDE.md", "AGENTS.md", allowed_sections=("provider mechanics",)
    )

    errors = sync_ai_config._check_pointer_file(spec)

    assert any("not an allowed provider-mechanics section" in error for error in errors)


def test_check_pointer_file_rejects_validation_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generic validation section is not provider mechanics."""
    _write_pointer(
        tmp_path,
        monkeypatch,
        "Follow AGENTS.md.\n\n## Validation\n\nRun the full suite.\n",
    )
    spec = sync_ai_config.PointerSpec(
        ".claude/CLAUDE.md", "AGENTS.md", allowed_sections=("provider mechanics",)
    )

    errors = sync_ai_config._check_pointer_file(spec)

    assert any("'validation'" in error for error in errors)


def test_check_pointer_file_rejects_arbitrary_renamed_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invented heading receives the same treatment as a known policy heading."""
    _write_pointer(
        tmp_path,
        monkeypatch,
        "Follow AGENTS.md.\n\n## Project Directives\n\nNo force pushes.\n",
    )
    spec = sync_ai_config.PointerSpec(
        ".claude/CLAUDE.md", "AGENTS.md", allowed_sections=("provider mechanics",)
    )

    errors = sync_ai_config._check_pointer_file(spec)

    assert any("'project directives'" in error for error in errors)


def test_check_pointer_file_allows_headings_inside_code_fences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fenced examples may show headings without becoming adapter policy."""
    _write_pointer(
        tmp_path,
        monkeypatch,
        "Follow AGENTS.md.\n\n## Provider Mechanics\n\n```md\n## Validation\n```\n",
    )
    spec = sync_ai_config.PointerSpec(
        ".claude/CLAUDE.md", "AGENTS.md", allowed_sections=("provider mechanics",)
    )

    assert sync_ai_config._check_pointer_file(spec) == []


def test_live_adapters_pass_provider_scope() -> None:
    """The shipped provider adapters satisfy the enforced allowed-section contract."""
    manifest = sync_ai_config.load_manifest()
    errors: list[str] = []
    for spec in manifest.pointer_files:
        errors.extend(sync_ai_config._check_pointer_file(spec))

    assert errors == []


def test_live_adapters_contain_no_volatile_model_ids() -> None:
    """Adapter prose does not pin provider model versions."""
    claude = (sync_ai_config.REPO_ROOT / ".claude" / "CLAUDE.md").read_text(encoding="utf-8")
    lowered = claude.lower()
    for token in ("opus 4", "sonnet 4", "gpt-4", "gpt-5", "gemini 2"):
        assert token not in lowered, token
