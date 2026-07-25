"""Tests for the mutmut-safe pytest temproot opt-out in the mutation ratchet (#6122)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dev import mutation_ratchet


def test_ensure_mutmut_safe_temproot_sets_stable_shared_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no override set, a stable, existing, shared root is published to the env."""
    monkeypatch.delenv("PYTEST_DEBUG_TEMPROOT", raising=False)

    first = mutation_ratchet.ensure_mutmut_safe_temproot(tmp_path)
    second = mutation_ratchet.ensure_mutmut_safe_temproot(tmp_path)

    # Stable + shared: the same path is returned and published both times.
    assert first == second
    assert os.environ["PYTEST_DEBUG_TEMPROOT"] == str(first)
    # The shared root is created and lives under the canonical worktree root.
    assert first.exists()
    assert first.name == "mutmut-shared"


def test_ensure_mutmut_safe_temproot_respects_caller_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller-provided PYTEST_DEBUG_TEMPROOT is honored and created if missing."""
    override = tmp_path / "manual-override"
    monkeypatch.setenv("PYTEST_DEBUG_TEMPROOT", str(override))

    result = mutation_ratchet.ensure_mutmut_safe_temproot(tmp_path)

    assert result == override
    assert override.exists()
    # The override path is retained verbatim, not rewritten to the shared root.
    assert "mutmut-shared" not in str(result)


def test_mutmut_shared_temproot_is_worktree_scoped(tmp_path: Path) -> None:
    """Different repo roots resolve to different worktree-scoped shared roots."""
    left = mutation_ratchet._mutmut_shared_temproot(tmp_path / "repo-a")
    right = mutation_ratchet._mutmut_shared_temproot(tmp_path / "repo-b")

    assert left != right
    # Both embed the canonical worktree-root lineage (pytest-of-<user>/wt-<hash>).
    assert left.parent.name.startswith("wt-")
    assert right.parent.name.startswith("wt-")


def test_run_mutmut_results_sets_temproot_before_mutmut_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shared temproot is published before the `mutmut run` subprocess starts.

    Monkeypatches subprocess so no real mutmut is executed; this proves the wiring
    that restores the bare `mutation_ratchet.py --check` command (#6122) without
    relying on an expensive full mutation pass.
    """
    monkeypatch.delenv("PYTEST_DEBUG_TEMPROOT", raising=False)
    captured: dict[str, str | None] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        if "results" in cmd:
            return SimpleNamespace(returncode=0, stdout="1. x: survived\n", stderr="")
        # `mutmut run` invocation: capture the env state at this instant.
        captured["temproot"] = os.environ.get("PYTEST_DEBUG_TEMPROOT")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mutation_ratchet.subprocess, "run", fake_run)

    mutation_ratchet._run_mutmut_results(tmp_path)

    temproot = captured.get("temproot")
    assert temproot, "PYTEST_DEBUG_TEMPROOT was not set before `mutmut run`"
    temproot_path = Path(temproot)
    assert temproot_path.exists(), "shared temproot must exist before workers start"
    assert "mutmut-shared" in temproot
