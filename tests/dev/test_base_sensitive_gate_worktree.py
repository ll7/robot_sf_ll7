"""Tests for the gate-worktree health check wired into the base-sensitive gate.

Issue #5967: the base-sensitive staleness gate must verify the registered gate
worktree still exists before running git/gh operations against it, failing closed
with the lease cleanup owner reported instead of operating on a vanished path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
                    branch="feature",
                    head_sha="c" * 40,
                    recovery={
                        "status": "partial",
                        "local_state_restored": False,
                        "loss_boundary": "dirty_untracked_ignored_state_not_recoverable",
                    },
                    cleanup_owner="owner=auto-smart-routing; pr=#5819; gate=gate-5819",
                )
                health = gate._verify_gate_worktree(str(wt))

        assert health is not None
        assert health["exists"] is False
        assert "owner=auto-smart-routing" in health["cleanup_owner"]
        assert health["branch"] == "feature"
        assert health["head_sha"] == "c" * 40
        assert health["recovery"]["local_state_restored"] is False
        assert health["recovery"]["loss_boundary"] == (
            "dirty_untracked_ignored_state_not_recoverable"
        )

    def test_missing_guard_helper_degrades_to_none(self, tmp_path) -> None:
        """When the guard helper is missing, the check degrades to None safely."""
        wt = tmp_path / "gate-wt"
        with patch.object(gate, "_GUARD_HELPER", tmp_path / "absent.py"):
            health = gate._verify_gate_worktree(str(wt))
        assert health is None

    def test_direct_execution_can_import_gate_guard(self, tmp_path) -> None:
        """Direct CLI execution imports the gate guard without relying on PYTHONPATH."""
        gh = tmp_path / "gh"
        gh.write_text("#!/bin/sh\nprintf '%s\\n' tests/dev/test_base_sensitive_gate_worktree.py\n")
        gh.chmod(0o755)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"

        result = subprocess.run(
            [
                sys.executable,
                str(gate.REPO_ROOT / "scripts/dev/check_base_sensitive_gates.py"),
                "--pr",
                "5979",
                "--gate-worktree-path",
                str(tmp_path / "missing-wt"),
                "--json",
            ],
            cwd=gate.REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        assert result.returncode == 2
        payload = json.loads(result.stdout)
        assert payload["gate_worktree_health"]["classification"] == "missing"
        assert "No module named 'scripts'" not in result.stdout + result.stderr

    def test_direct_entrypoint_is_import_safe_without_site_or_implicit_paths(
        self, tmp_path: Path
    ) -> None:
        """Direct selection works from both cwd locations without ambient imports."""
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONSAFEPATH"}
        }
        environment["PYTHONSAFEPATH"] = "1"
        selection_args = ("--list-files", "--json")

        direct_root = subprocess.run(
            [
                sys.executable,
                "-S",
                str(gate.REPO_ROOT / "scripts/dev/check_base_sensitive_gates.py"),
                *selection_args,
            ],
            cwd=gate.REPO_ROOT,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
            timeout=30,
        )
        direct_unrelated = subprocess.run(
            [
                sys.executable,
                "-S",
                str(gate.REPO_ROOT / "scripts/dev/check_base_sensitive_gates.py"),
                *selection_args,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
            timeout=30,
        )

        module_environment = {
            key: value for key, value in environment.items() if key != "PYTHONSAFEPATH"
        }
        module = subprocess.run(
            [sys.executable, "-S", "-m", "scripts.dev.check_base_sensitive_gates", *selection_args],
            cwd=gate.REPO_ROOT,
            capture_output=True,
            text=True,
            env=module_environment,
            check=False,
            timeout=30,
        )

        for label, result in (
            ("direct from repository root", direct_root),
            ("direct from unrelated directory", direct_unrelated),
            ("module invocation", module),
        ):
            assert result.returncode == 0, f"{label} failed:\n{result.stdout}\n{result.stderr}"

        root_selection = json.loads(direct_root.stdout)
        assert root_selection == json.loads(direct_unrelated.stdout)
        assert root_selection == json.loads(module.stdout)

        shadow_root = tmp_path / "shadow-checkout"
        (shadow_root / "scripts" / "dev").mkdir(parents=True)
        (shadow_root / "scripts" / "__init__.py").write_text("", encoding="utf-8")
        (shadow_root / "scripts" / "dev" / "base_sensitive_selector.py").write_text(
            "raise RuntimeError('shadow checkout imported')\n", encoding="utf-8"
        )
        shadow_environment = {
            **module_environment,
            "PYTHONPATH": os.pathsep.join((str(shadow_root), str(gate.REPO_ROOT))),
            "PYTHONSAFEPATH": "1",
        }
        shadow_result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(gate.REPO_ROOT / "scripts/dev/check_base_sensitive_gates.py"),
                *selection_args,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env=shadow_environment,
            check=False,
            timeout=30,
        )

        assert shadow_result.returncode == 0, (
            f"direct invocation was shadowed by another checkout:\n"
            f"{shadow_result.stdout}\n{shadow_result.stderr}"
        )
        assert json.loads(shadow_result.stdout) == root_selection


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
