"""Tests for the issue-claim admission wrapper."""

from __future__ import annotations

import ast
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import yaml

from scripts.dev import goal_issue_admission
from scripts.dev.goal_issue_admission import admit_issue, compact_admission

if TYPE_CHECKING:
    from collections.abc import Iterator


def _preflight(*, ready: bool) -> dict[str, object]:
    return {
        "schema": "issue_implementability.v1",
        "issue": {
            "number": 7611,
            "title": "bounded workflow repair",
            "state": "OPEN",
            "labels": ["state:ready"],
            "assignees": [],
        },
        "classification": "ready" if ready else "needs_spec",
        "contract": {"body_sha256": "body-a"},
        "claim": {"ok": True, "claimed": False, "claim_ref": None, "sha": None},
        "ready": ready,
        "write_allowed": ready,
    }


def test_non_ready_issue_never_attempts_claim() -> None:
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            return_value=_preflight(ready=False),
        ),
        patch("scripts.dev.goal_issue_admission.issue_claim.acquire_issue") as acquire,
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=False,
        )

    assert payload["outcome"] == "not_admitted"
    assert payload["write_attempted"] is False
    acquire.assert_not_called()


def test_check_only_ready_issue_performs_no_write() -> None:
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            return_value=_preflight(ready=True),
        ),
        patch("scripts.dev.goal_issue_admission.issue_claim.acquire_issue") as acquire,
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=True,
        )

    assert payload["ok"] is True
    assert payload["outcome"] == "ready_check_only"
    assert payload["write_attempted"] is False
    acquire.assert_not_called()


def test_check_only_forwards_prospective_ready_to_live_preflight() -> None:
    """The readiness gate can opt into a private prospective label evaluation."""
    with patch(
        "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
        return_value=_preflight(ready=True),
    ) as live_report:
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=True,
            prospective_ready=True,
        )

    assert payload["ok"] is True
    live_report.assert_called_once_with(
        7611,
        repo="ll7/robot_sf_ll7",
        remote="origin",
        prospective_ready=True,
    )


def test_ready_issue_calls_atomic_claim_once() -> None:
    claim = {"ok": True, "claimed": True, "sha": "abc"}
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            return_value=_preflight(ready=True),
        ),
        patch(
            "scripts.dev.goal_issue_admission.issue_claim.acquire_issue",
            return_value=claim,
        ) as acquire,
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=False,
        )

    assert payload["ok"] is True
    assert payload["outcome"] == "claim_acquired"
    assert payload["write_attempted"] is True
    assert payload["revalidation"] == {"performed": True, "inputs_match": True}
    assert payload["claim"] == claim
    acquire.assert_called_once_with(
        7611,
        repo="ll7/robot_sf_ll7",
        remote="origin",
        source_ref="origin/main",
    )


def test_atomic_claim_failure_remains_explicit() -> None:
    claim = {"ok": False, "claimed": False, "error": "claim exists"}
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            return_value=_preflight(ready=True),
        ),
        patch(
            "scripts.dev.goal_issue_admission.issue_claim.acquire_issue",
            return_value=claim,
        ),
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=False,
        )

    assert payload["ok"] is False
    assert payload["outcome"] == "claim_failed"
    assert payload["write_attempted"] is True


def test_changed_issue_inputs_fail_closed_before_claim_write() -> None:
    initial = _preflight(ready=True)
    changed = _preflight(ready=True)
    changed["contract"] = {"body_sha256": "body-b"}
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            side_effect=[initial, changed],
        ) as live_report,
        patch("scripts.dev.goal_issue_admission.issue_claim.acquire_issue") as acquire,
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=False,
        )

    assert payload["outcome"] == "not_admitted"
    assert payload["write_attempted"] is False
    assert payload["revalidation"] == {"performed": True, "inputs_match": False}
    assert live_report.call_count == 2
    acquire.assert_not_called()


def test_changed_label_that_makes_issue_non_ready_fails_closed() -> None:
    initial = _preflight(ready=True)
    changed = _preflight(ready=False)
    changed["issue"] = {
        **changed["issue"],
        "labels": ["state:ready", "state:blocked"],
    }
    changed["classification"] = "blocked"
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            side_effect=[initial, changed],
        ),
        patch("scripts.dev.goal_issue_admission.issue_claim.acquire_issue") as acquire,
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=False,
        )

    assert payload["outcome"] == "not_admitted"
    assert payload["preflight"]["classification"] == "blocked"
    assert payload["claim_outcome"] == "unclaimed"
    acquire.assert_not_called()


def test_changed_label_with_state_qualifier_uses_blocker_semantics() -> None:
    initial = _preflight(ready=True)
    changed = _preflight(ready=False)
    changed["issue"] = {
        **changed["issue"],
        "labels": ["state:ready", "state:parked"],
    }
    changed["classification"] = "blocked"
    changed["admission_reason"] = "blocked"
    with (
        patch(
            "scripts.dev.goal_issue_admission.issue_implementability.live_issue_report",
            side_effect=[initial, changed],
        ),
        patch("scripts.dev.goal_issue_admission.issue_claim.acquire_issue") as acquire,
    ):
        payload = admit_issue(
            7611,
            repo="ll7/robot_sf_ll7",
            remote="origin",
            source_ref="origin/main",
            check_only=False,
        )

    assert payload["outcome"] == "not_admitted"
    assert payload["preflight"]["classification"] == "blocked"
    assert payload["claim_outcome"] == "unclaimed"
    acquire.assert_not_called()


def test_compact_admission_preserves_claim_and_write_boundary() -> None:
    """Queue projections must retain the canonical admission and claim outcomes."""
    payload = compact_admission(
        {
            "ok": True,
            "outcome": "ready_check_only",
            "write_attempted": False,
            "source_ref": "origin/main",
            "preflight": {
                "classification": "ready",
                "reasons": ["contract is complete"],
                "ready": True,
                "write_allowed": True,
                "claim": {
                    "ok": True,
                    "claimed": False,
                    "claim_ref": "agent-claims/issue-7611",
                    "sha": None,
                },
            },
        }
    )

    assert payload["outcome"] == "ready_check_only"
    assert payload["write_attempted"] is False
    assert payload["ready"] is True
    assert payload["write_allowed"] is True
    assert payload["claim_outcome"] == "unclaimed"


def test_only_goal_admission_calls_the_atomic_issue_claim_owner() -> None:
    """Repository call sites must not bypass the canonical admission wrapper."""
    root = Path(__file__).resolve().parents[2]
    call_sites: set[str] = set()
    for base in (root / "scripts" / "dev", root / "tests" / "dev"):
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                is_issue_claim_owner = (
                    isinstance(owner, ast.Name) and owner.id == "issue_claim"
                ) or (isinstance(owner, ast.Attribute) and owner.attr == "issue_claim")
                if node.func.attr == "acquire_issue" and is_issue_claim_owner:
                    call_sites.add(path.relative_to(root).as_posix())

    assert call_sites == {"scripts/dev/goal_issue_admission.py"}


@pytest.fixture
def isolated_cli_env(tmp_path: Path) -> Iterator[dict[str, str]]:
    """Stage only declared dependencies and fake commands for isolated CLI probes.

    ``PYTHONPATH`` deliberately names the copied PyYAML dependency, not this
    repository.  The direct-entrypoint bootstrap is responsible for finding the
    repository package; the subprocess still has the declared third-party
    dependency it needs.
    """
    dependencies = tmp_path / "dependencies"
    shutil.copytree(
        Path(yaml.__file__).parent,
        dependencies / "yaml",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    commands = tmp_path / "commands"
    commands.mkdir()
    audit = tmp_path / "external-calls"
    for name in ("gh", "git", "uv", "pip"):
        command = commands / name
        command.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$0" >> "$CALL_AUDIT"\nexit 97\n', encoding="utf-8"
        )
        command.chmod(0o755)
    env = {"PATH": str(commands), "PYTHONPATH": str(dependencies), "CALL_AUDIT": str(audit)}
    dependency = subprocess.run(
        [sys.executable, "-S", "-B", "-c", "import yaml; assert yaml.safe_load('ok: true')['ok']"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert dependency.returncode == 0, dependency.stderr
    yield env
    assert not audit.exists(), "Offline CLI invoked an external command"
    assert not list(tmp_path.rglob("*.pyc"))


def _run_admission_cli(
    root: Path,
    *,
    env: dict[str, str],
    cwd: Path,
    args: list[str],
    direct: bool,
    script: Path | None = None,
    flags: tuple[str, ...] = ("-S", "-B"),
) -> subprocess.CompletedProcess[str]:
    """Run one direct or module CLI with an explicitly controlled environment."""
    command = [sys.executable, *flags]
    if direct:
        command.append(str(script or root / "scripts/dev/goal_issue_admission.py"))
    else:
        command.extend(["-m", "scripts.dev.goal_issue_admission"])
    command.extend(args)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def _assert_repository_is_not_an_ambient_dependency(root: Path, env: dict[str, str]) -> None:
    """Keep the repository source path distinct from the explicitly staged dependency."""
    assert str(root) not in env["PYTHONPATH"].split(os.pathsep)


@pytest.mark.parametrize("foreign_cwd", [False, True], ids=["checkout", "foreign"])
def test_direct_help_without_ambient_source_path(
    tmp_path: Path,
    isolated_cli_env: dict[str, str],
    foreign_cwd: bool,
) -> None:
    """The direct entrypoint works without an ambient source path."""
    root = Path(__file__).resolve().parents[2]
    _assert_repository_is_not_an_ambient_dependency(root, isolated_cli_env)
    result = _run_admission_cli(
        root,
        env=isolated_cli_env,
        cwd=tmp_path if foreign_cwd else root,
        args=["--help"],
        direct=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Gate an atomic issue claim" in result.stdout
    assert result.stderr == ""


def test_direct_entrypoint_prefers_worktree_over_hostile_pythonpath(
    tmp_path: Path,
    isolated_cli_env: dict[str, str],
) -> None:
    """Direct execution must put the resolved worktree before an untrusted source path."""
    root = Path(__file__).resolve().parents[2]
    hostile = tmp_path / "hostile-source"
    (hostile / "scripts/dev").mkdir(parents=True)
    (hostile / "scripts/__init__.py").write_text(
        "raise RuntimeError('hostile scripts package imported')\n", encoding="utf-8"
    )
    (hostile / "scripts/dev/__init__.py").write_text(
        "raise RuntimeError('hostile scripts.dev package imported')\n", encoding="utf-8"
    )
    env = dict(isolated_cli_env)
    env["PYTHONPATH"] = os.pathsep.join([str(hostile), env["PYTHONPATH"]])
    _assert_repository_is_not_an_ambient_dependency(root, env)

    result = _run_admission_cli(
        root,
        env=env,
        cwd=tmp_path,
        args=["0", "--check-only"],
        direct=True,
    )

    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["outcome"] == "error"
    assert result.stderr == ""


def test_direct_entrypoint_resolves_symlinked_worktree_root(
    tmp_path: Path,
    isolated_cli_env: dict[str, str],
) -> None:
    """A symlinked worktree path still resolves imports from the real worktree root."""
    root = Path(__file__).resolve().parents[2]
    alias_root = tmp_path / "worktree-alias"
    alias_root.symlink_to(root, target_is_directory=True)
    _assert_repository_is_not_an_ambient_dependency(root, isolated_cli_env)

    result = _run_admission_cli(
        root,
        env=isolated_cli_env,
        cwd=tmp_path,
        args=["0", "--check-only"],
        direct=True,
        script=alias_root / "scripts/dev/goal_issue_admission.py",
        flags=("-P", "-S", "-B"),
    )

    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout) == {
        "error": "issue number must be positive",
        "issue": 0,
        "ok": False,
        "outcome": "error",
        "schema": "goal_issue_admission.v1",
        "write_attempted": False,
    }
    assert result.stderr == ""


def test_direct_and_module_invalid_issue_outputs_and_exit_codes_match(
    tmp_path: Path,
    isolated_cli_env: dict[str, str],
) -> None:
    """Direct and module entrypoints share invalid-input output and exit contracts."""
    root = Path(__file__).resolve().parents[2]
    _assert_repository_is_not_an_ambient_dependency(root, isolated_cli_env)
    args = ["0", "--check-only"]
    direct = _run_admission_cli(
        root,
        env=isolated_cli_env,
        cwd=tmp_path,
        args=args,
        direct=True,
    )
    module = _run_admission_cli(
        root,
        env=isolated_cli_env,
        cwd=root,
        args=args,
        direct=False,
    )

    assert direct.returncode == module.returncode == 1
    assert direct.stderr == module.stderr == ""
    assert json.loads(direct.stdout) == json.loads(module.stdout)


def test_module_import_preserves_sys_path() -> None:
    """Package imports do not apply the direct-entrypoint path bootstrap."""
    before = list(sys.path)
    importlib.reload(goal_issue_admission)
    assert sys.path == before
