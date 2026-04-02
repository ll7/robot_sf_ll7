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


def test_ai_skill_files_exist() -> None:
    """The repo should expose the adopted AI workflow skills as real Codex skill files."""

    for path in [
        SKILL_DIR / "autoresearch" / "SKILL.md",
        SKILL_DIR / "auto-improvement" / "SKILL.md",
        SKILL_DIR / "context-map" / "SKILL.md",
        SKILL_DIR / "what-context-needed" / "SKILL.md",
        SKILL_DIR / "quality-playbook" / "SKILL.md",
        SKILL_DIR / "agentic-eval" / "SKILL.md",
        SKILL_DIR / "review-and-refactor" / "SKILL.md",
        SKILL_DIR / "update-docs-on-code-change" / "SKILL.md",
        ADAPTATION_NOTE,
    ]:
        assert path.exists(), f"missing expected AI workflow surface: {path}"


def test_ai_skill_files_contain_expected_language() -> None:
    """The individual skill files should keep the repo-specific workflow language intact."""

    skill_expectations = {
        "autoresearch": [
            "autonomous iterative experimentation loop",
            "results.tsv",
            "scripts/dev/pr_ready_check.sh",
            "fallback or degraded benchmark outcomes",
        ],
        "auto-improvement": [
            "focused measurement-aware refinement loop",
            "clean-up",
        ],
        "context-map": [
            "focused repository context map",
            "validation",
        ],
        "what-context-needed": [
            "minimum repository context",
            "up to three short questions",
        ],
        "quality-playbook": [
            "proof-first workflow",
            "non-trivial changes",
        ],
        "agentic-eval": [
            "small goldens",
            "rubrics",
        ],
        "review-and-refactor": [
            "review-then-refactor",
            "surgically",
        ],
        "update-docs-on-code-change": [
            "docs stale",
            "workflow docs",
        ],
    }

    for skill_name, expected_fragments in skill_expectations.items():
        text = (SKILL_DIR / skill_name / "SKILL.md").read_text(encoding="utf-8").lower()
        for fragment in expected_fragments:
            assert fragment.lower() in text, f"missing {fragment!r} in {skill_name}"


def test_ai_docs_reference_the_new_skill_surfaces() -> None:
    """The docs and instruction surfaces should point at the repo-local skills rather than prompts."""

    docs_readme_text = DOCS_README.read_text(encoding="utf-8")
    dev_guide_text = DEV_GUIDE.read_text(encoding="utf-8")
    repo_overview_text = REPO_OVERVIEW.read_text(encoding="utf-8")
    copilot_instructions_text = COPILOT_INSTRUCTIONS.read_text(encoding="utf-8")
    adaptation_text = ADAPTATION_NOTE.read_text(encoding="utf-8")

    assert "Awesome Copilot Adaptation" in docs_readme_text
    assert "autoresearch" in docs_readme_text
    assert "auto-improvement" in docs_readme_text
    assert "context-discovery" in docs_readme_text
    assert "quality" in docs_readme_text
    assert "doc-sync" in docs_readme_text

    assert "Codex skills can be found in `.codex/skills/`" in dev_guide_text

    for fragment in [
        ".codex/skills/autoresearch/",
        ".codex/skills/auto-improvement/",
        ".codex/skills/context-map/",
        ".codex/skills/what-context-needed/",
        ".codex/skills/quality-playbook/",
        ".codex/skills/agentic-eval/",
        ".codex/skills/review-and-refactor/",
        ".codex/skills/update-docs-on-code-change/",
        "doc-sync",
    ]:
        assert fragment in repo_overview_text

    for fragment in [
        "autoresearch/SKILL.md",
        "auto-improvement/SKILL.md",
        "context-map/SKILL.md",
        "what-context-needed/SKILL.md",
        "quality-playbook/SKILL.md",
        "agentic-eval/SKILL.md",
        "review-and-refactor/SKILL.md",
        "update-docs-on-code-change/SKILL.md",
        "awesome_copilot_adaptation.md",
    ]:
        assert fragment in copilot_instructions_text

    assert "context-map" in adaptation_text
    assert "what-context-needed" in adaptation_text
    assert "quality-playbook" in adaptation_text
    assert "agentic-eval" in adaptation_text
    assert "review-and-refactor" in adaptation_text
    assert "update-docs-on-code-change" in adaptation_text
