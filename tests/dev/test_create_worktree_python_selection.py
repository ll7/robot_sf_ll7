"""Supported-Python selection must precede any worktree mutation (issue #9763).

``create_worktree.sh`` used bare ``python3`` for its capacity, lock, lease and
receipt helpers. On a nonlogin remote shell that can resolve to an interpreter
older than the repository's ``requires-python = ">=3.11"``, and
``pr_gate_lease.py`` imports ``datetime.UTC`` (3.11+). The failure then surfaced
only after the script had entered creation, so the caller had to roll back.

These tests pin both halves of the contract: an unsupported interpreter is
refused *before* any mutation, and a supported one is used for every helper call.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CREATE_WORKTREE = REPO_ROOT / "scripts" / "dev" / "create_worktree.sh"


def _write_py310_shim(directory: Path) -> Path:
    """Create a stand-in interpreter that behaves like a nonlogin 3.10 default.

    It refuses the version probe, mirroring an interpreter too old for the
    repository, and refuses any helper invocation so an accidental fall-through
    to it is visible rather than silent.
    """
    shim_dir = directory / "bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "python3"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-c" ]]; then\n'
        '  echo "3.10.0 (shim)" >&2\n'
        "  exit 1\n"
        "fi\n"
        'echo "py310-shim refused helper: $*" >&2\n'
        "exit 1\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def test_unsupported_interpreter_refused_before_any_mutation(tmp_path: Path) -> None:
    """A too-old interpreter fails closed before capacity, lock, or add."""
    shim = _write_py310_shim(tmp_path)
    target = tmp_path / "never-created"
    env = dict(os.environ)
    env["ROBOT_SF_PYTHON"] = str(shim)
    env["PATH"] = f"{shim.parent}{os.pathsep}{env.get('PATH', '')}"

    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(target),
            "--branch",
            "probe/issue-9763-refusal",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    assert "no supported Python interpreter found" in result.stderr
    assert "ROBOT_SF_PYTHON" in result.stderr
    # The refusal must precede the capacity preflight, so no target and no
    # worktree registry entry may exist.
    assert not target.exists(), "refusal must happen before any worktree mutation"
    assert "verdict: PASS" not in result.stdout + result.stderr, "capacity preflight must not run"


def test_supported_interpreter_passes_dry_run(tmp_path: Path) -> None:
    """A supported interpreter still reaches the capacity preflight."""
    target = tmp_path / "dry-run-target"
    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(target),
            "--branch",
            "probe/issue-9763-success",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "verdict: PASS" in output
    assert "git worktree add was not invoked" in output
    assert not target.exists(), "--dry-run must not create the worktree"


def test_too_old_default_python3_falls_through_to_a_supported_interpreter(
    tmp_path: Path,
) -> None:
    """The reported case: a 3.10 default must not break creation.

    With no ``ROBOT_SF_PYTHON`` override, a too-old ``python3`` first on PATH must
    fall through to a supported interpreter instead of failing the run or, worse,
    invoking a helper with the unsupported one.
    """
    if not any(shutil.which(name) for name in ("python3.13", "python3.12", "python3.11")):
        pytest.skip("no versioned supported interpreter available to fall through to")

    shim = _write_py310_shim(tmp_path)
    target = tmp_path / "fallthrough-target"
    env = dict(os.environ)
    env.pop("ROBOT_SF_PYTHON", None)
    env["PATH"] = f"{shim.parent}{os.pathsep}{env.get('PATH', '')}"

    result = subprocess.run(
        [
            str(CREATE_WORKTREE),
            "--path",
            str(target),
            "--branch",
            "probe/issue-9763-fallthrough",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "verdict: PASS" in output
    assert "py310-shim refused helper" not in output, (
        "the unsupported interpreter must never be used for a helper"
    )
    assert not target.exists(), "--dry-run must not create the worktree"


def test_every_python_helper_call_uses_the_selected_interpreter() -> None:
    """No bare ``python3`` helper invocation may remain in the script."""
    text = CREATE_WORKTREE.read_text(encoding="utf-8")
    # Strip comments, then remove the two blocks that legitimately name
    # python3.x: the resolver's candidate list and the operator-facing refusal
    # message. Everything left must invoke the validated interpreter.
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    resolver = re.search(r"resolve_supported_python\(\) \{.*?\n\}", code, re.S)
    assert resolver is not None, "supported-python resolver not found"
    code = code.replace(resolver.group(0), "")
    refusal = re.search(
        r'\{\n\s*echo "Error: no supported Python interpreter found.*?\n  \} >&2', code, re.S
    )
    assert refusal is not None, "interpreter refusal message not found"
    code = code.replace(refusal.group(0), "")

    bare = [
        line.strip()
        for line in code.splitlines()
        if re.search(r"(?<![\"$/\w])python3(?![\w.])", line)
    ]
    assert not bare, f"bare python3 helper invocation remains: {bare}"
    assert 'ROBOT_SF_PYTHON="$pr_ready_python"' in code, (
        "the validated interpreter must be exported so re-exec keeps using it"
    )


@pytest.mark.skipif(sys.version_info[:2] < (3, 11), reason="repository requires Python >= 3.11")
def test_resolver_prefers_supported_interpreters(tmp_path: Path) -> None:
    """The resolver rejects an old shim and accepts a supported interpreter."""
    shim = _write_py310_shim(tmp_path)
    env = dict(os.environ)
    env["ROBOT_SF_PYTHON"] = str(shim)
    result = subprocess.run(
        [str(CREATE_WORKTREE), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    # --help exits 0, but the interpreter gate runs before argument parsing, so
    # the refusal must win and the help text must not be printed.
    assert result.returncode == 2
    assert "no supported Python interpreter found" in result.stderr

    if shutil.which("python3.13") or shutil.which("python3.12") or shutil.which("python3.11"):
        good = (
            shutil.which("python3.13") or shutil.which("python3.12") or shutil.which("python3.11")
        )
        assert good is not None
        env["ROBOT_SF_PYTHON"] = good
        ok = subprocess.run(
            [str(CREATE_WORKTREE), "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert ok.returncode == 0, ok.stderr
        assert "Usage: scripts/dev/create_worktree.sh" in ok.stdout
