"""Tests for repo-local AI skill surface discoverability."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / ".codex" / "skills"
DOCS_README = ROOT / "docs" / "README.md"
DEV_GUIDE = ROOT / "docs" / "dev_guide.md"
REPO_OVERVIEW = ROOT / "docs" / "ai" / "repo_overview.md"
COPILOT_INSTRUCTIONS = ROOT / ".github" / "copilot-instructions.md"
ADAPTATION_NOTE = ROOT / "docs" / "ai" / "awesome_copilot_adaptation.md"


def test_ai_skill_surfaces_are_documented_and_repo_scoped() -> None:
    """Skill discovery should remain explicit because issue 746 is about packaging repo-local AI workflow assets, not inventing a second guidance stack."""

    autoresearch_skill = SKILL_DIR / "autoresearch" / "SKILL.md"
    auto_improvement_skill = SKILL_DIR / "auto-improvement" / "SKILL.md"

    for path in [autoresearch_skill, auto_improvement_skill, ADAPTATION_NOTE]:
        assert path.exists(), f"missing expected AI workflow surface: {path}"

    autoresearch_text = autoresearch_skill.read_text(encoding="utf-8")
    assert "autonomous iterative experimentation loop" in autoresearch_text.lower()
    assert "results.tsv" in autoresearch_text
    assert "scripts/dev/pr_ready_check.sh" in autoresearch_text
    assert "fallback or degraded benchmark outcomes" in autoresearch_text

    auto_improvement_text = auto_improvement_skill.read_text(encoding="utf-8")
    assert "focused measurement-aware refinement loop" in auto_improvement_text.lower()
    assert "clean-up" in auto_improvement_text

    docs_readme_text = DOCS_README.read_text(encoding="utf-8")
    assert "Awesome Copilot Adaptation" in docs_readme_text
    assert "autoresearch" in docs_readme_text
    assert "auto-improvement" in docs_readme_text

    dev_guide_text = DEV_GUIDE.read_text(encoding="utf-8")
    assert "Codex skills can be found in `.codex/skills/`" in dev_guide_text

    repo_overview_text = REPO_OVERVIEW.read_text(encoding="utf-8")
    assert ".codex/skills/autoresearch/" in repo_overview_text
    assert ".codex/skills/auto-improvement/" in repo_overview_text
    assert "auto-improvement" in repo_overview_text

    copilot_instructions_text = COPILOT_INSTRUCTIONS.read_text(encoding="utf-8")
    assert "autoresearch/SKILL.md" in copilot_instructions_text
    assert "auto-improvement/SKILL.md" in copilot_instructions_text
    assert "awesome_copilot_adaptation.md" in copilot_instructions_text
