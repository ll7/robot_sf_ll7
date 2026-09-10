"""Tests for the single normative instruction-precedence contract.

Issue #8933 requires exactly one repository-internal precedence block, owned by ``AGENTS.md``, with
documented resolutions for the representative conflict scenarios. Provider adapters and other
surfaces link to the contract instead of restating an authority order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dev.check_instruction_references import (
    PRECEDENCE_END_MARKER,
    PRECEDENCE_START_MARKER,
    REPO_ROOT,
    check_precedence_contract,
    run_checks,
)

if TYPE_CHECKING:
    from pathlib import Path


def _block(body: str) -> str:
    """Wrap body text in the canonical precedence markers."""
    return f"{PRECEDENCE_START_MARKER}\n{body}\n{PRECEDENCE_END_MARKER}\n"


def _write_surface(root: Path, name: str, body: str) -> None:
    """Create one instruction surface under a temporary repository root."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_repo_has_single_precedence_owner() -> None:
    """The live canonical graph has exactly one precedence block, owned by AGENTS.md."""
    assert check_precedence_contract(REPO_ROOT) == []
    assert run_checks(REPO_ROOT)["precedence_errors"] == []


def test_duplicate_precedence_block_is_rejected(tmp_path: Path) -> None:
    """A second marked precedence block in the owner fails the consistency check."""
    _write_surface(
        tmp_path,
        "AGENTS.md",
        _block("First order.") + _block("Second order."),
    )

    errors = check_precedence_contract(tmp_path, graph=("AGENTS.md",))

    assert errors
    assert "exactly once" in errors[0]


def test_precedence_block_in_adapter_is_rejected(tmp_path: Path) -> None:
    """A provider adapter cannot introduce a competing precedence block."""
    _write_surface(tmp_path, ".claude/CLAUDE.md", _block("Adapter order."))

    errors = check_precedence_contract(tmp_path, graph=(".claude/CLAUDE.md",))

    assert errors
    assert "must be owned by AGENTS.md" in errors[0]


def test_unordered_markers_are_rejected(tmp_path: Path) -> None:
    """Reversed markers are a malformed block, not a valid second order."""
    _write_surface(
        tmp_path,
        "AGENTS.md",
        f"{PRECEDENCE_END_MARKER}\norder\n{PRECEDENCE_START_MARKER}\n",
    )

    errors = check_precedence_contract(tmp_path, graph=("AGENTS.md",))

    assert errors
    assert "in order" in errors[0]


def test_contract_documents_conflict_resolutions() -> None:
    """The canonical block states the outcome for each required conflict fixture."""
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    start = text.index(PRECEDENCE_START_MARKER)
    end = text.index(PRECEDENCE_END_MARKER)
    block = " ".join(text[start:end].split())

    for clause in (
        "cannot waive an invariant",
        "cannot silently weaken root invariants",
        "never outranks current code and evidence",
        "never create repository policy",
        "does not define a competing runtime precedence",
    ):
        assert clause in block, f"precedence contract lacks conflict resolution: {clause}"
