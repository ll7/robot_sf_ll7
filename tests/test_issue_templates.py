"""Tests for the repo's agent-ready issue templates and issue workflow skills."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from scripts.tools.issue_template_audit import SECTION_ORDER

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"
DOCS_GUIDE = ROOT / "docs" / "dev_guide.md"
def _skill_path(name: str) -> Path:
    """Resolve an agent skill file by its directory name."""

    skill_file = ROOT / ".agents" / "skills" / name / "SKILL.md"
    assert skill_file.exists(), f"missing skill file: {skill_file}"
    return skill_file
KNOWN_LABELS = {
    "agent",
    "benchmark",
    "blocked",
    "bug",
    "documentation",
    "enhancement",
    "epic",
    "local planner",
    "refactor",
    "research",
    "technical-debt",
    "test",
    "validation",
    "workflow",
}

EXPECTED_ISSUE_FORMS = {
    "blocked-external-artifact.yml": [
        "Unavailable asset or runtime",
        "Unblock condition",
        "Fail-closed policy",
        "Artifact policy",
    ],
    "epic.yml": [
        "Child issues or child-creation task",
        "Blocked / ready state",
        "Acceptance criteria",
        "Estimate metadata",
    ],
    "execution-run.yml": [
        "Runtime and execution location",
        "Current phase",
        "Owner / agent handoff",
        "Artifact root",
        "Last log timestamp",
        "Next decision point",
    ],
    "research-validation.yml": [
        "Hypothesis",
        "Evidence grade",
        "Artifact policy",
        "Validation command",
    ],
    "test-debt.yml": [
        "Skipped or failing test evidence",
        "Intended behavior",
        "Targeted validation",
        "Acceptance criteria",
    ],
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


def _load_form(path: Path) -> dict[str, object]:
    """Parse a GitHub issue form YAML file."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} must be a YAML mapping"
    return payload


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
        assert "title" in frontmatter
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
        "issue_default.md": [
            "## Goal / Problem",
            "## Archetype Metadata",
            "archetype:",
            "evidence_tier:",
            "linked_policy:",
            "Task type",
            "Entry points",
        ],
        "planner_integration.md": ["## Goal / Problem", "Planner", "Integration goal"],
        "benchmark_experiment.md": ["## Goal / Problem", "Scenario description", "Hypothesis"],
        "refactor.md": ["## Goal / Problem", "Objective", "Motivation"],
        "research.md": ["## Goal / Problem", "Research question", "Scientific motivation"],
    }

    for name, markers in template_markers.items():
        _, body = _load_template(TEMPLATE_DIR / name)
        for marker in markers:
            assert marker in body, f"missing marker {marker!r} in {name}"


def test_issue_forms_cover_common_backlog_lanes() -> None:
    """Verify YAML issue forms guide the common backlog lanes added for issue 1256."""

    for form_name, markers in EXPECTED_ISSUE_FORMS.items():
        form_path = TEMPLATE_DIR / form_name
        assert form_path.exists(), f"missing issue form {form_name}"
        form = _load_form(form_path)
        assert form["name"]
        assert form["description"]
        labels = form.get("labels", [])
        assert isinstance(labels, list)
        assert set(labels).issubset(KNOWN_LABELS), f"unknown label(s) in {form_name}: {labels}"
        body = form.get("body")
        assert isinstance(body, list) and body, f"{form_name} body must be a non-empty list"
        serialized = yaml.safe_dump(form, sort_keys=True)
        for marker in markers:
            assert marker in serialized, f"missing marker {marker!r} in {form_name}"


def test_issue_form_config_keeps_existing_templates_available() -> None:
    """Verify adding issue forms does not disable existing Markdown templates."""

    config = _load_form(TEMPLATE_DIR / "config.yml")
    assert config["blank_issues_enabled"] is True


def test_issue_template_docs_and_skills_reference_real_paths() -> None:
    """Verify the docs and skills point at real repo files and commands.

    This matters because the issue-template workflow must remain discoverable and
    the new skills should not refer to dead paths or invented gh commands.
    """

    docs_text = DOCS_GUIDE.read_text(encoding="utf-8")
    assert "../.github/ISSUE_TEMPLATE/issue_default.md" in docs_text
    assert "../.github/ISSUE_TEMPLATE/research-validation.yml" in docs_text
    assert "../.github/ISSUE_TEMPLATE/test-debt.yml" in docs_text
    assert "../.github/ISSUE_TEMPLATE/blocked-external-artifact.yml" in docs_text
    assert "../.github/ISSUE_TEMPLATE/execution-run.yml" in docs_text
    assert "../.github/ISSUE_TEMPLATE/epic.yml" in docs_text
    assert "../.agents/skills/gh-issue-creator/SKILL.md" in docs_text
    assert "../.agents/skills/gh-issue-template-auditor/SKILL.md" in docs_text
    assert "../.agents/skills/gh-issue-priority-assessor/SKILL.md" in docs_text

    prioritization_text = (ROOT / "docs" / "project_prioritization.md").read_text(encoding="utf-8")
    assert "Plausibility Checks" in prioritization_text
    assert "Assessment Workflow" in prioritization_text
    assert "Expected Duration in Hours" in prioritization_text
    assert "High `Improvement`" in prioritization_text
    assert "High `Success Probability`" in prioritization_text

    creator_text = _skill_path("gh-issue-creator").read_text(encoding="utf-8")
    assert "GitHub MCP / GitHub app tools" in creator_text
    assert "gh issue create" in creator_text
    assert "gh project item-add" in creator_text
    assert "gh project item-edit" in creator_text
    assert "issue_default.md" in creator_text
    assert "planner_integration.md" in creator_text
    assert "benchmark_experiment.md" in creator_text
    assert "Priority" in creator_text
    assert "Expected Duration in Hours" in creator_text
    assert "Reviewed" in creator_text

    auditor_text = _skill_path("gh-issue-template-auditor").read_text(encoding="utf-8")
    assert "GitHub MCP / GitHub app tools" in auditor_text
    assert "uv run python scripts/tools/issue_template_audit.py" in auditor_text
    assert "gh issue view" in auditor_text
    assert "gh issue edit" in auditor_text
    assert "decision-required" in auditor_text
    assert "Archetype Metadata" in auditor_text
    assert "docs/context/issue_1512_issue_archetypes.md" in auditor_text

    assessor_text = _skill_path("gh-issue-priority-assessor").read_text(encoding="utf-8")
    assert "GitHub MCP / GitHub app tools" in assessor_text
    assert "docs/project_prioritization.md" in assessor_text
    assert "gh issue view" in assessor_text
    assert "gh project item-list" in assessor_text
    assert "gh project item-edit" in assessor_text
    assert "Priority Score" in assessor_text
    assert "Estimate Discussion" in assessor_text
    assert "plausibility" in assessor_text.lower()

    maintainer_text = _skill_path("issue-contract-maintainer").read_text(encoding="utf-8")
    assert "audit-template-compliance" in maintainer_text
    assert "Archetype Metadata" in maintainer_text
    assert "docs/context/issue_1512_issue_archetypes.md" in maintainer_text

    documentation_text = (TEMPLATE_DIR / "documentation.md").read_text(encoding="utf-8")
    assert "docs/README.md" in documentation_text


def test_issue_splitter_skill_defines_parent_child_contract() -> None:
    """Verify the parent-to-child issue-splitting mode stays conservative and auditable."""

    splitter_text = _skill_path("issue-splitter").read_text(encoding="utf-8")
    expected_markers = [
        "smallest independently implementable child",
        "duplicate check",
        "Next Implementable Child",
        "Parent issue",
        "Non-goals",
        "Validation / Testing",
        "Blocked by",
        "Project #5",
        "draft-only",
    ]
    for marker in expected_markers:
        assert marker in splitter_text, f"missing issue-splitter marker {marker!r}"

    maintainer_text = _skill_path("issue-contract-maintainer").read_text(encoding="utf-8")
    assert "split-parent-to-child" in maintainer_text
    assert "issue-splitter" in maintainer_text

    creator_text = _skill_path("gh-issue-creator").read_text(encoding="utf-8")
    assert "Parent issue" in creator_text
    assert "Blocked by" in creator_text

    goal_text = (ROOT / ".agents" / "skills" / "goal-issue-implementation" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "issue-splitter" in goal_text
    assert "Next Implementable Child" in goal_text
