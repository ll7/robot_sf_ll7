"""Tests for the decomposed workflow compendia (issue #8939).

The entrypoints document is an index plus command entrypoints, and the former relocated compendium
is a topic-to-owner index with no embedded procedure. Index targets are checked through the
canonical instruction graph.
"""

from __future__ import annotations

from scripts.dev.check_instruction_references import (
    CANONICAL_INSTRUCTION_GRAPH,
    REPO_ROOT,
    run_checks,
)

RELOCATED = REPO_ROOT / "docs" / "dev" / "agents" / "relocated-agents-guidance.md"
ENTRYPOINTS = REPO_ROOT / "docs" / "ai" / "agent_workflow_entrypoints.md"
HANDOFF_EXAMPLE = REPO_ROOT / "docs" / "templates" / "handoff.v2.example.yaml"


def _nonblank(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def test_relocated_guidance_is_a_topic_index() -> None:
    """The former compendium is a compact index, not a mixed policy document."""
    text = RELOCATED.read_text(encoding="utf-8")
    assert len(_nonblank(text)) <= 60, "relocated guidance drifted back into a compendium"
    assert "| Topic | Canonical owner | Trigger |" in text
    assert "```" not in text
    assert "adds no policy" in text
    table_rows = [
        line
        for line in text.splitlines()
        if line.startswith("| ") and "---" not in line and "Canonical owner" not in line
    ]
    assert len(table_rows) >= 15


def test_relocated_index_is_in_the_canonical_graph() -> None:
    """Index targets resolve because the index participates in the reference check."""
    assert "docs/dev/agents/relocated-agents-guidance.md" in CANONICAL_INSTRUCTION_GRAPH
    report = run_checks(REPO_ROOT)
    assert report["errors"] == []
    assert "docs/dev/agents/relocated-agents-guidance.md" in report["files"]


def test_entrypoints_has_no_large_schema_example() -> None:
    """The handoff schema example moved next to its template instead of the runtime index."""
    text = ENTRYPOINTS.read_text(encoding="utf-8")
    assert "handoff.v2-example:start" not in text
    assert "schema_version: handoff.v2" not in text
    assert "docs/templates/handoff.v2.example.yaml" in text
    assert len(_nonblank(text)) <= 200, "entrypoints document drifted back into a handbook"


def test_handoff_example_exists_with_contract_version() -> None:
    """The moved example stays a flat handoff.v2 contract."""
    text = HANDOFF_EXAMPLE.read_text(encoding="utf-8")
    assert "schema_version: handoff.v2" in text
    assert len(_nonblank(text)) <= 60
