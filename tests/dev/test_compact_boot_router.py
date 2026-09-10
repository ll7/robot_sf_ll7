"""Tests for the compact root boot router and moved-rule backlinks (issues #8932, #8942).

The root contract is a router plus invariants. Procedure that moved out must have a canonical
owner, and mandatory boot content must stay at least 40% below the audit baseline recorded in
issue #8930 (228 non-blank lines, ~20402 characters).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_MD = REPO_ROOT / "AGENTS.md"
CODE_REVIEW = REPO_ROOT / "docs" / "code_review.md"
DEPENDABOT_POLICY = REPO_ROOT / "docs" / "dev" / "dependabot_update_policy.md"
MAINTAINER_VALUES = REPO_ROOT / "docs" / "maintainer_values.md"
ROUTE_ENTRYPOINTS = REPO_ROOT / "docs" / "ai" / "agent_workflow_entrypoints.md"
RELOCATED_INDEX = REPO_ROOT / "docs" / "dev" / "agents" / "relocated-agents-guidance.md"

AUDIT_BASELINE_NONBLANK = 228
AUDIT_BASELINE_CHARS = 20402
MAX_NONBLANK = int(AUDIT_BASELINE_NONBLANK * 0.6)
MAX_CHARS = int(AUDIT_BASELINE_CHARS * 0.6)


def _nonblank(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def test_root_boot_content_is_at_least_40_percent_smaller() -> None:
    """The root router must not regrow past the 40% reduction target."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    assert len(_nonblank(text)) <= MAX_NONBLANK
    assert len(text) <= MAX_CHARS


def test_root_keeps_required_router_sections() -> None:
    """Precedence, routing, mutation triggers, friction, and invariants stay in root."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    for heading in (
        "## Instruction Precedence",
        "## Task-Scoped Context Entrypoints",
        "## Mutation And Delivery Triggers",
        "## Friction And Follow-Up",
        "## Donts",
    ):
        assert heading in text, heading


def test_moved_procedure_has_canonical_backlinks() -> None:
    """Moved procedures and index entries point to current canonical owners."""
    code_review = CODE_REVIEW.read_text(encoding="utf-8")
    assert "## Validation Depth And Delivery Contract" in code_review
    assert "gh_pr_body_rest.py <pr-number>" in code_review
    assert "--body-file <final-body.md>" in code_review
    dependabot = DEPENDABOT_POLICY.read_text(encoding="utf-8")
    assert "## Workflow action-pin coupling" in dependabot
    assert "## Pinned Action References" not in dependabot
    assert "check_dependabot_update_policy.py" in dependabot

    maintainer_values = MAINTAINER_VALUES.read_text(encoding="utf-8")
    assert "docs/code_review.md" in maintainer_values
    assert "validation matrix in `AGENTS.md`" not in maintainer_values

    route_entrypoints = ROUTE_ENTRYPOINTS.read_text(encoding="utf-8")
    assert "docs/maintainer_values.md" in route_entrypoints
    assert "maintainer value hierarchy in `AGENTS.md`" not in route_entrypoints

    relocated_index = RELOCATED_INDEX.read_text(encoding="utf-8")
    assert "docs/ai/repo_overview.md" in relocated_index
    assert "docs/maintainer_values.md" in relocated_index
    assert "`AGENTS.md` Project Structure And Ownership" not in relocated_index
    assert "`AGENTS.md` Planning And Communication" not in relocated_index
    assert "`AGENTS.md` Delivery And Communication" in relocated_index


def test_root_contains_no_environment_specific_bootstrap() -> None:
    """Root carries no commands that belong to the worktree or build owners."""
    text = AGENTS_MD.read_text(encoding="utf-8")
    for fragment in (
        "uv venv .venv",
        "uv sync --all-extras",
        "git worktree add",
        "python3",
    ):
        assert fragment not in text, fragment
