"""Tests for the canonical instruction-reference checker.

The checker owns the unresolved-reference contract for agent instruction surfaces (issue #8931).
Fixtures cover a missing required file, an optional/non-normative link, a symlink target, and a
generated target so the routing repair cannot regress silently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dev.check_instruction_references import (
    REPO_ROOT,
    check_instruction_graph,
    check_routing_ownership,
    run_checks,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_surface(root: Path, name: str, body: str) -> None:
    """Create one instruction surface under a temporary repository root."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_missing_required_reference_fails(tmp_path: Path) -> None:
    """A path presented as required must exist or the check fails closed."""
    _write_surface(tmp_path, "AGENTS.md", "Read `docs/missing_guide.md` before acting.\n")

    result = check_instruction_graph(tmp_path, graph=("AGENTS.md",))

    assert result.errors
    assert "docs/missing_guide.md" in result.errors[0]


def test_optional_reference_allows_missing_target(tmp_path: Path) -> None:
    """Optional or illustrative references do not fail when the target is absent."""
    _write_surface(
        tmp_path,
        "AGENTS.md",
        "Optional background reading: `docs/missing_guide.md`.\n",
    )

    result = check_instruction_graph(tmp_path, graph=("AGENTS.md",))

    assert result.errors == []
    assert result.optional_skipped == 1


def test_symlink_target_is_resolved(tmp_path: Path) -> None:
    """Symlinks resolve to their target; a broken symlink is reported."""
    _write_surface(tmp_path, "AGENTS.md", "Read `docs/current.md` for the live contract.\n")
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "real.md").write_text("# real\n", encoding="utf-8")
    (tmp_path / "docs" / "current.md").symlink_to("real.md")
    assert check_instruction_graph(tmp_path, graph=("AGENTS.md",)).errors == []

    (tmp_path / "docs" / "current.md").unlink()
    (tmp_path / "docs" / "current.md").symlink_to("absent.md")
    result = check_instruction_graph(tmp_path, graph=("AGENTS.md",))
    assert result.errors
    assert "symlink to a missing target" in result.errors[0]


def test_generated_target_allowlist(tmp_path: Path) -> None:
    """A missing generated target is allowed only when the manifest declares it generated."""
    _write_surface(tmp_path, "AGENTS.md", "Read `docs/index.md` for the published build output.\n")

    unlisted = check_instruction_graph(tmp_path, graph=("AGENTS.md",))
    assert unlisted.errors

    listed = check_instruction_graph(
        tmp_path,
        graph=("AGENTS.md",),
        generated_targets=frozenset({"docs/index.md"}),
    )
    assert listed.errors == []


def test_file_relative_reference_is_resolved(tmp_path: Path) -> None:
    """Repository links are resolved against the referencing file's directory."""
    _write_surface(tmp_path, "docs/README.md", "# Index\n")
    _write_surface(tmp_path, "docs/other.md", "Back to [the index](./README.md).\n")

    result = check_instruction_graph(tmp_path, graph=("docs/other.md",))

    assert result.errors == []


def test_canonical_instruction_graph_resolves() -> None:
    """Every required reference in the live canonical graph must resolve at the repository root."""
    report = run_checks(REPO_ROOT)
    errors = report["errors"]

    assert isinstance(errors, list)
    assert errors == [], "\n".join(str(item) for item in errors)
    assert report["references_checked"] > 0


def test_routing_owner_is_single_surface() -> None:
    """Exactly one canonical surface owns the task route table."""
    assert check_routing_ownership(REPO_ROOT) == []
