"""Tests for the gate-worktree health check wired into the base-sensitive gate.

Issue #5967: the base-sensitive staleness gate must verify the registered gate
worktree still exists before running git/gh operations against it, failing closed
with the lease cleanup owner reported instead of operating on a vanished path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.dev import check_base_sensitive_gates as gate
from scripts.dev.gate_worktree_guard import GateWorktreeHealth

_GUARD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "gate_worktree_guard.py"


class TestVerifyGateWorktree:
    """Tests for check_base_sensitive_gates._verify_gate_worktree."""

    def test_present_worktree_reports_exists(self, tmp_path) -> None:
        """An existing gate worktree is reported as present."""
        wt = tmp_path / "gate-wt"
        wt.mkdir()
        with patch.object(gate, "_GUARD_HELPER", _GUARD_PATH):
            with patch("scripts.dev.gate_worktree_guard.verify_gate_worktree") as mock_verify:
                mock_verify.return_value = GateWorktreeHealth(
                    schema="gate_worktree_guard.v1",
                    path=str(wt),
                    exists=True,
                    classification="healthy",
                )
                health = gate._verify_gate_worktree(str(wt))

        assert health is not None
        assert health["exists"] is True

    def test_missing_worktree_reports_owner(self, tmp_path) -> None:
        """A missing worktree with a live lease reports the cleanup owner."""
        wt = tmp_path / "gone-wt"
        with patch.object(gate, "_GUARD_HELPER", _GUARD_PATH):
            with patch("scripts.dev.gate_worktree_guard.verify_gate_worktree") as mock_verify:
                mock_verify.return_value = GateWorktreeHealth(
                    schema="gate_worktree_guard.v1",
                    path=str(wt),
                    exists=False,
                    classification="missing",
                    cleanup_owner="owner=auto-smart-routing; pr=#5819; gate=gate-5819",
                )
                health = gate._verify_gate_worktree(str(wt))

        assert health is not None
        assert health["exists"] is False
        assert "owner=auto-smart-routing" in health["cleanup_owner"]

    def test_missing_guard_helper_degrades_to_none(self, tmp_path) -> None:
        """When the guard helper is missing, the check degrades to None safely."""
        wt = tmp_path / "gate-wt"
        with patch.object(gate, "_GUARD_HELPER", tmp_path / "absent.py"):
            health = gate._verify_gate_worktree(str(wt))
        assert health is None


class TestCheckPrGateStalenessWorktreeGuard:
    """The staleness check must fail closed on a vanished gate worktree."""

    def test_vanished_gate_worktree_fails_closed(self, tmp_path) -> None:
        """A vanished registered gate worktree fails the gate before staleness ops."""
        wt = tmp_path / "gone-wt"
        overall: dict = {}
        rc = gate._check_pr_gate_staleness(
            "5819",
            {"needs_gate": True, "changed_sensitive_files": []},
            overall,
            as_json=True,
            gate_worktree_path=str(wt),
        )
        # _report_gate_error returns 2 (indeterminate/fail-closed).
        assert rc == 2
        assert "gate_worktree_health" in overall
        assert "vanished" in (overall.get("error") or "")

    def test_present_gate_worktree_proceeds(self, tmp_path) -> None:
        """A present registered gate worktree lets the staleness check continue."""
        wt = tmp_path / "gate-wt"
        wt.mkdir()
        overall: dict = {}
        with patch(
            "scripts.dev.check_base_sensitive_gates._branch_is_current_with_main", return_value=True
        ):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = gate.subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="sha123\n", stderr=""
                )
                rc = gate._check_pr_gate_staleness(
                    "5819",
                    {"needs_gate": True, "changed_sensitive_files": []},
                    overall,
                    as_json=True,
                    gate_worktree_path=str(wt),
                )
        assert rc is None
