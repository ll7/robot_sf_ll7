"""Tests for the principle-only maintainer values contract (issue #8940).

Values state stable trade-off principles; procedure lives with its canonical task owners. The drift
check fails when a procedure section or command block is copied back into the values file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dev.check_instruction_references import (
    MAINTAINER_VALUES,
    REPO_ROOT,
    check_maintainer_values,
    run_checks,
)

if TYPE_CHECKING:
    from pathlib import Path

PRINCIPLE_NAMES = (
    "Honest evidence",
    "Correctness and reproducibility before speed",
    "Proportional process",
    "Reuse before abstraction",
    "Preserve user scope",
    "Recoverability",
    "Discoverable public behavior",
    "Research progress is the point",
    "Calibrated uncertainty",
)


def _write_values(root: Path, body: str) -> None:
    """Create a values file under a temporary repository root."""
    path = root / MAINTAINER_VALUES
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_live_values_contract_is_valid_and_compact() -> None:
    """The live values file passes the drift check and stays compact."""
    assert check_maintainer_values(REPO_ROOT) == []
    assert run_checks(REPO_ROOT)["values_errors"] == []
    text = (REPO_ROOT / MAINTAINER_VALUES).read_text(encoding="utf-8")
    nonblank = [line for line in text.splitlines() if line.strip()]
    assert len(nonblank) <= 80, "maintainer values drifted back into a procedure manual"


def test_procedure_section_is_rejected(tmp_path: Path) -> None:
    """A re-added procedure section fails the drift check."""
    _write_values(
        tmp_path,
        "# Maintainer Values\n\n## Principles\n\n1. Honest evidence.\n\n## Validation\n\nRun ruff.\n",
    )

    errors = check_maintainer_values(tmp_path)

    assert any("Validation" in error for error in errors)


def test_code_block_is_rejected(tmp_path: Path) -> None:
    """Embedded command blocks fail the drift check."""
    _write_values(
        tmp_path,
        "# Maintainer Values\n\n## Principles\n\n```bash\nuv run pytest\n```\n",
    )

    errors = check_maintainer_values(tmp_path)

    assert any("code block" in error for error in errors)


def test_principles_cover_tradeoffs() -> None:
    """Each retained principle states the trade-off it resolves."""
    text = (REPO_ROOT / MAINTAINER_VALUES).read_text(encoding="utf-8")
    for name in PRINCIPLE_NAMES:
        assert name in text, f"missing principle: {name}"
    assert "Resolves:" in text


def test_agents_links_values_without_restating_principles() -> None:
    """Root guidance links values instead of repeating the principle text."""
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "docs/maintainer_values.md" in agents
    assert "Honest evidence." not in agents


def test_plans_are_profile_triggered() -> None:
    """Planning is triggered by the coordinated and evidence-critical profiles."""
    plans = (REPO_ROOT / ".agents" / "PLANS.md").read_text(encoding="utf-8")
    assert "coordinated" in plans
    assert "evidence_critical" in plans
    assert "Instruction Precedence" in plans
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Coordinated and Evidence-critical" in agents
