"""Tests for repo-local AI skill surface discoverability."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / ".agents" / "skills"
LEGACY_SKILL_DIR = ROOT / ".codex" / "skills"
OPENCODE_SKILL_DIR = ROOT / ".opencode" / "skills"
DOCS_README = ROOT / "docs" / "README.md"
DEV_GUIDE = ROOT / "docs" / "dev_guide.md"
REPO_OVERVIEW = ROOT / "docs" / "ai" / "repo_overview.md"
COPILOT_INSTRUCTIONS = ROOT / ".github" / "copilot-instructions.md"
ADAPTATION_NOTE = ROOT / "docs" / "ai" / "awesome_copilot_adaptation.md"
CODING_AGENTS_NOTE = ROOT / "docs" / "context" / "issue_728_coding_agents_compatibility.md"
AGENTS_MD = ROOT / "AGENTS.md"


def test_ai_skill_files_exist() -> None:
    """The repo should expose the adopted AI workflow skills as real Codex skill files."""

    assert LEGACY_SKILL_DIR.exists(), "expected legacy .codex/skills mirror to exist"

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


def test_legacy_skill_paths_mirror_agents_directory() -> None:
    """Legacy skill paths should remain working mirrors of the canonical tree."""

    assert SKILL_DIR.exists(), "expected canonical .agents/skills directory to exist"
    for legacy_dir, label in [
        (LEGACY_SKILL_DIR, ".codex/skills"),
        (OPENCODE_SKILL_DIR, ".opencode/skills"),
    ]:
        assert legacy_dir.exists(), f"expected legacy {label} mirror to exist"
        assert legacy_dir.is_symlink(), f"expected {label} to remain a symlink mirror"
        assert legacy_dir.resolve() == SKILL_DIR.resolve()


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

    assert "Canonical skills live in `.agents/skills/`" in dev_guide_text

    for fragment in [
        ".agents/skills/autoresearch/",
        ".agents/skills/auto-improvement/",
        ".agents/skills/context-map/",
        ".agents/skills/what-context-needed/",
        ".agents/skills/quality-playbook/",
        ".agents/skills/agentic-eval/",
        ".agents/skills/review-and-refactor/",
        ".agents/skills/update-docs-on-code-change/",
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


def test_coding_agents_compatibility_note_is_discoverable() -> None:
    """The coding-agents compatibility note must exist and be linked from the three agent entry points.

    Verifies that a contributor or AI assistant can find the cross-agent compatibility stance
    and the retrieval -> planning -> execution -> verification discipline by reading AGENTS.md,
    .github/copilot-instructions.md, or docs/dev_guide.md.
    """

    assert CODING_AGENTS_NOTE.exists(), (
        "canonical coding-agents compatibility note missing: "
        "docs/context/issue_728_coding_agents_compatibility.md"
    )

    note_ref = "issue_728_coding_agents_compatibility.md"

    agents_text = AGENTS_MD.read_text(encoding="utf-8")
    assert note_ref in agents_text, (
        f"{note_ref!r} not linked from AGENTS.md; add it to the Cross-Agent Compatibility section"
    )

    copilot_text = COPILOT_INSTRUCTIONS.read_text(encoding="utf-8")
    assert note_ref in copilot_text, f"{note_ref!r} not linked from .github/copilot-instructions.md"

    dev_guide_text = DEV_GUIDE.read_text(encoding="utf-8")
    assert note_ref in dev_guide_text, f"{note_ref!r} not linked from docs/dev_guide.md"
