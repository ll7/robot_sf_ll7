"""Contract tests for advisory ty/typecheck documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_GUIDE = ROOT / "docs" / "dev_guide.md"
VSCODE_TASKS = ROOT / ".vscode" / "tasks.json"


def _load_vscode_tasks() -> dict[str, str]:
    """Load VS Code tasks as structured data despite JSON-with-trailing-comma syntax."""
    text = VSCODE_TASKS.read_text(encoding="utf-8")
    normalized = re.sub(r",(\s*[}\]])", r"\1", text)
    return {task["label"]: task["command"] for task in json.loads(normalized)["tasks"]}


def test_dev_guide_documents_typecheck_as_advisory() -> None:
    """Merge guidance should match the current exit-zero typecheck behavior."""
    text = DEV_GUIDE.read_text(encoding="utf-8")

    assert "not a PR-readiness merge blocker by itself" in text
    assert "PRs are not blocked solely because the advisory `ty` phase reports findings" in text
    assert (
        "A fail-closed typecheck gate, changed-files ratchet, or baseline-reduction workflow"
        in text
    )
    assert "**All type errors must be addressed before merging PRs**" not in text
    assert "Type check clean (no type errors; warnings documented if present)" not in text


def test_vscode_typecheck_tasks_are_labeled_advisory() -> None:
    """VS Code task labels should not imply fail-closed typecheck behavior."""
    text = VSCODE_TASKS.read_text(encoding="utf-8")
    tasks = _load_vscode_tasks()

    assert tasks["Type Check (advisory)"] == "uvx ty check . --exit-zero"
    assert (
        tasks["Check Code Quality (Ruff + advisory ty)"]
        == "uv run ruff check . && uvx ty check . --exit-zero"
    )
    assert '"label": "Type Check",' not in text
    assert '"label": "Check Code Quality",' not in text
