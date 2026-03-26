"""Tests for the repo's agent-ready issue templates and issue workflow skills."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from scripts.tools.issue_template_audit import SECTION_ORDER

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"
DOCS_GUIDE = ROOT / "docs" / "dev_guide.md"
SKILL_FILES = [
    ROOT / ".codex" / "skills" / "gh-issue-creator" / "SKILL.md",
    ROOT / ".codex" / "skills" / "gh-issue-template-auditor" / "SKILL.md",
]
KNOWN_LABELS = {
    "bug",
    "documentation",
    "enhancement",
    "refactor",
    "benchmark",
    "validation",
    "local navigation",
}


def _load_template(path: Path) -> tuple[dict[str, object], str]:
    """Parse a GitHub issue template into frontmatter and markdown body."""

    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)\Z", text, re.S)
    assert match is not None, f"{path} does not have YAML frontmatter"
    frontmatter = yaml.safe_load(match.group(1))
    assert isinstance(frontmatter, dict), f"{path} frontmatter must be a YAML mapping"
    body = match.group(2)
    return frontmatter, body


def test_issue_templates_parse_and_include_required_sections() -> None:
    """Verify the templates parse cleanly and include the agent-ready sections.

    This matters because issue 693 is specifically about making issue bodies
    machine-executable for agents, so missing headings would defeat the change.
    """

    template_paths = sorted(TEMPLATE_DIR.glob("*.md"))
    assert template_paths, "expected issue templates to exist"

    for path in template_paths:
        frontmatter, body = _load_template(path)
        assert frontmatter["name"]
        assert frontmatter["about"]
        assert frontmatter["title"]
        labels = frontmatter.get("labels", [])
        assert isinstance(labels, list)
        assert set(labels).issubset(KNOWN_LABELS), f"unknown label(s) in {path.name}: {labels}"
        for heading in SECTION_ORDER:
            assert f"## {heading}" in body, f"missing section {heading!r} in {path.name}"


def test_specialized_issue_templates_include_domain_specific_sections() -> None:
    """Verify the new specialized templates expose the expected task-specific prompts.

    This matters because the new templates are only useful if they encode planner,
    benchmark, and general-task cues that agents can fill without inventing structure.
    """

    template_markers = {
        "issue_default.md": ["## Goal / Problem", "Task type", "Entry points"],
        "planner_integration.md": ["## Goal / Problem", "Planner", "Integration goal"],
        "benchmark_experiment.md": ["## Goal / Problem", "Scenario description", "Hypothesis"],
        "refactor.md": ["## Goal / Problem", "Objective", "Motivation"],
        "research.md": ["## Goal / Problem", "Research question", "Scientific motivation"],
    }

    for name, markers in template_markers.items():
        _, body = _load_template(TEMPLATE_DIR / name)
        for marker in markers:
            assert marker in body, f"missing marker {marker!r} in {name}"


def test_issue_template_docs_and_skills_reference_real_paths() -> None:
    """Verify the docs and skills point at real repo files and commands.

    This matters because the issue-template workflow must remain discoverable and
    the new skills should not refer to dead paths or invented gh commands.
    """

    docs_text = DOCS_GUIDE.read_text(encoding="utf-8")
    assert "../.github/ISSUE_TEMPLATE/issue_default.md" in docs_text
    assert "../.codex/skills/gh-issue-creator/SKILL.md" in docs_text
    assert "../.codex/skills/gh-issue-template-auditor/SKILL.md" in docs_text

    creator_text = SKILL_FILES[0].read_text(encoding="utf-8")
    assert "gh issue create" in creator_text
    assert "gh project item-add" in creator_text
    assert "gh project item-edit" in creator_text
    assert "issue_default.md" in creator_text
    assert "planner_integration.md" in creator_text
    assert "benchmark_experiment.md" in creator_text
    assert "Priority" in creator_text
    assert "Expected Duration in Hours" in creator_text
    assert "Reviewed" in creator_text

    auditor_text = SKILL_FILES[1].read_text(encoding="utf-8")
    assert "uv run python scripts/tools/issue_template_audit.py" in auditor_text
    assert "gh issue view" in auditor_text
    assert "gh issue edit" in auditor_text
    assert "decision-required" in auditor_text

    documentation_text = (TEMPLATE_DIR / "documentation.md").read_text(encoding="utf-8")
    assert "docs/README.md" in documentation_text
