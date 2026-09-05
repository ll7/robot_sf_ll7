"""Tests for task-scoped context entrypoints and canonical task route contracts."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_MD = REPO_ROOT / "AGENTS.md"
AGENTS_README = REPO_ROOT / ".agents" / "README.md"
ENTRYPOINTS_DOC = REPO_ROOT / "docs" / "ai" / "agent_workflow_entrypoints.md"
RELOCATED_GUIDANCE = REPO_ROOT / "docs" / "dev" / "agents" / "relocated-agents-guidance.md"

CANONICAL_ROUTES = (
    "Read-only observation",
    "Documentation-only edit",
    "Implementation / runtime change",
    "Scientific / benchmark interpretation",
    "Environment / worktree repair",
)


def test_agent_workflow_entrypoints_defines_five_canonical_routes() -> None:
    """The entrypoints document must define the 5 canonical routes with required contract columns."""
    assert ENTRYPOINTS_DOC.is_file(), f"Missing {ENTRYPOINTS_DOC}"
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")

    assert "## Task Routes And Preflight Discipline" in text

    # Verify each canonical route is present in the route table
    for route in CANONICAL_ROUTES:
        assert route in text, f"Missing canonical route in table: {route}"

    # Extract the route table rows
    table_match = re.search(
        r"\| Route \| Purpose \| Required context / evidence \| First deterministic command \| "
        r"Permitted mutations \| Authoritative acceptance command \|\n"
        r"\| (?:--- \| ){5}---\ \|\n"
        r"((?:\| \*\*.*?\*\* \| .*? \|\n)+)",
        text,
    )
    assert table_match is not None, "Failed to locate standard task route table format"
    table_body = table_match.group(1)

    for route in CANONICAL_ROUTES:
        assert f"**{route}**" in table_body, f"Route row not found: {route}"


def test_agent_workflow_entrypoints_commands_exist_or_valid() -> None:
    """Preflight and acceptance commands in the route table must point to real repository scripts."""
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")

    referenced_scripts = (
        "scripts/dev/watch_pr_ci_status.py",
        "scripts/tools/sync_ai_config.py",
        "scripts/dev/pr_ready_check.sh",
        "scripts/dev/check_worktree_capacity.py",
        "scripts/dev/check_worktree_optional_deps.py",
    )
    for script_rel in referenced_scripts:
        assert script_rel in text, f"Script not mentioned in entrypoints doc: {script_rel}"
        script_path = REPO_ROOT / script_rel
        assert script_path.is_file(), f"Referenced route script not found: {script_rel}"


def test_agent_workflow_entrypoints_documents_route_boundaries_and_negative_rules() -> None:
    """The entrypoints document must explicitly define the negative boundaries and linked owners."""
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")

    assert "### Route Boundaries and Negative Rules" in text

    # Negative rule 1: Read-only review never mutates branches (ref #8321)
    assert "review_worktree_guard.py" in text
    assert "#8321" in text
    assert "never merge `origin/main` into the implementation branch" in text

    # Negative rule 2: Validation proportional to change risk
    assert "Validation proportional to change risk" in text

    # Negative rule 3: Environment blockers fail closed
    assert "Environment blockers are not relaxation licenses" in text
    assert "never authorizes lowering scientific gates" in text

    # Negative rule 4: Freshness before expensive proof (#7649)
    assert "Freshness before expensive proof" in text
    assert "#7649" in text

    # Negative rule 5: Observer / audit separation (#8304, #8307)
    assert "Separation of observer/audit collection from mutations" in text
    assert "#8304" in text
    assert "#8307" in text

    # Negative rule 6: Scientific indicator integrity
    assert "Integrity of scientific indicators" in text

    # Negative rule 7: Privacy and provenance boundaries
    assert "Privacy and provenance boundaries" in text


def test_compact_final_handoff_contract_fields() -> None:
    """The handoff contract must specify all standard acceptance elements."""
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")

    assert "## Compact Final Handoff Contract" in text
    required_handoff_fields = (
        "Result",
        "Revisions",
        "Changed paths",
        "Validation evidence",
        "Unrun or unavailable checks",
        "Scientific scope & limitations",
        "Next disposition",
    )
    for field in required_handoff_fields:
        assert f"**{field}**" in text, f"Missing handoff contract field: {field}"


def test_agents_md_task_scoped_context_and_mode_specific_sync() -> None:
    """AGENTS.md must define task-scoped context entrypoints and mode-specific branch sync."""
    text = AGENTS_MD.read_text(encoding="utf-8")

    assert "## Task-Scoped Context Entrypoints" in text
    assert "Always-required core context:" in text
    assert "docs/maintainer_values.md" in text
    assert "AGENTS.md" in text
    assert "docs/ai/agent_workflow_entrypoints.md" in text

    # Verify mode-specific branch sync distinction
    assert "branch synchronization is mode-specific:" in text
    assert "For implementation worktrees, fetch latest `origin/main` and merge it early" in text
    assert "never merge `origin/main` into the implementation branch" in text
    assert "review_worktree_guard.py" in text
    assert "#8321" in text


def test_relocated_guidance_mode_specific_sync() -> None:
    """relocated-agents-guidance.md must also reflect mode-specific branch sync."""
    text = RELOCATED_GUIDANCE.read_text(encoding="utf-8")

    assert "branch synchronization is mode-specific:" in text
    assert "never merge `origin/main` into the implementation branch" in text


def test_agents_readme_references_task_routes_and_mode_specific_sync() -> None:
    """.agents/README.md must reference task routes and mode-specific sync."""
    text = AGENTS_README.read_text(encoding="utf-8")

    assert "task-scoped context" in text
    assert "task route selection" in text
    assert "Branch synchronization is mode-specific" in text
