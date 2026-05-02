"""Verify repo guidance consistently prefers MCP-first GitHub interaction."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_guidance_surfaces_are_mcp_first_with_gh_fallback() -> None:
    """Verify guidance consistently frames MCP as primary and gh as fallback.

    This matters because issue 756 is about migrating repo-local GitHub
    workflow guidance without overclaiming that scripted batch automation should
    stop using gh entirely.
    """

    agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Prefer GitHub MCP / GitHub app tools" in agents_text
    assert "Keep the GitHub CLI (`gh`) for scripted batch" in agents_text

    dev_guide_text = (ROOT / "docs" / "dev_guide.md").read_text(encoding="utf-8")
    assert "Prefer GitHub MCP / GitHub app tools for interactive issue, PR, and project work" in (
        dev_guide_text
    )
    assert "Keep `gh` for scripted batch operations" in dev_guide_text

    prioritization_text = (ROOT / "docs" / "project_prioritization.md").read_text(encoding="utf-8")
    assert "Prefer GitHub MCP / GitHub app tools" in prioritization_text
    assert "deterministic `gh` fallback" in prioritization_text

    workflow_text = (
        ROOT / "docs" / "context" / "issue_713_batch_first_issue_workflow.md"
    ).read_text(encoding="utf-8")
    assert "Prefer GitHub MCP / GitHub app tools" in workflow_text
    assert "Keep `gh` as the deterministic fallback" in workflow_text
