"""Verify the batch-first GitHub workflow is discoverable across repo guidance."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_NOTE = ROOT / "docs" / "context" / "issue_713_batch_first_issue_workflow.md"
SURFACE_PATHS = {
    ROOT / "docs" / "README.md": "./context/issue_713_batch_first_issue_workflow.md",
    ROOT / "docs" / "dev_guide.md": "docs/context/issue_713_batch_first_issue_workflow.md",
    ROOT / "AGENTS.md": "docs/context/issue_713_batch_first_issue_workflow.md",
    ROOT
    / ".github"
    / "copilot-instructions.md": "docs/context/issue_713_batch_first_issue_workflow.md",
    ROOT
    / "docs"
    / "project_prioritization.md": "docs/context/issue_713_batch_first_issue_workflow.md",
}
SKILL_PATHS = [
    ROOT / ".codex" / "skills" / "gh-issue-autopilot" / "SKILL.md",
    ROOT / ".codex" / "skills" / "gh-issue-sequencer" / "SKILL.md",
    ROOT / ".codex" / "skills" / "gh-issue-creator" / "SKILL.md",
    ROOT / ".codex" / "skills" / "gh-issue-priority-assessor" / "SKILL.md",
    ROOT / ".codex" / "skills" / "gh-issue-clarifier" / "SKILL.md",
    ROOT / ".codex" / "skills" / "gh-issue-template-auditor" / "SKILL.md",
]


def test_batch_first_workflow_note_is_linked_from_repo_guidance() -> None:
    """Verify the workflow note is linked from the main guidance surfaces.

    This matters because issue- and Project #5-related work should point at one
    canonical batching rule rather than relying on repeated ad hoc instructions.
    """

    note_text = WORKFLOW_NOTE.read_text(encoding="utf-8")
    assert "batch-first" in note_text.lower()
    assert "Project #5" in note_text
    assert "score sync" in note_text.lower()

    for path, needle in SURFACE_PATHS.items():
        text = path.read_text(encoding="utf-8")
        assert needle in text, f"missing workflow note link in {path.name}"

    for path in SKILL_PATHS:
        text = path.read_text(encoding="utf-8")
        assert "issue_713_batch_first_issue_workflow.md" in text, (
            f"missing note link in {path.name}"
        )
        assert "batch" in text.lower(), f"missing batch guidance in {path.name}"
