"""Tests for scripts/dev/check_context_notes.sh shared-venv worktree behavior."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tests.support.environment_guards import configure_git_identity

ROOT = Path(__file__).resolve().parents[2]
CHECK_CONTEXT_NOTES = ROOT / "scripts" / "dev" / "check_context_notes.sh"
RUN_WORKTREE_SHARED_VENV = ROOT / "scripts" / "dev" / "run_worktree_shared_venv.sh"


def test_check_context_notes_uses_shared_venv_wrapper() -> None:
    """The script must delegate commands through run_worktree_shared_venv.sh."""
    script_text = CHECK_CONTEXT_NOTES.read_text(encoding="utf-8")
    assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in script_text
    assert '"$SCRIPT_DIR/run_worktree_shared_venv.sh" -- python' in script_text
    assert "uv run python scripts/validation/check_docs_proof_consistency.py" not in script_text
    assert "uv run python scripts/dev/check_docs_evidence_integrity.py" not in script_text
    assert "uv run python scripts/tools/check_context_note_freshness.py" not in script_text


def test_check_context_notes_in_fresh_worktree_does_not_create_venv(tmp_path: Path) -> None:
    """Running check_context_notes.sh in a fresh linked worktree must not create a local .venv."""
    repo = tmp_path / "repo"
    script_dir = repo / "scripts" / "dev"
    validation_dir = repo / "scripts" / "validation"
    tools_dir = repo / "scripts" / "tools"
    docs_dir = repo / "docs" / "context"
    fake_bin = repo / "fake-bin"

    script_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    tools_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    fake_bin.mkdir(parents=True)

    # Initialize main repo venv
    main_venv = repo / ".venv"
    (main_venv / "bin").mkdir(parents=True)
    main_python = main_venv / "bin" / "python"
    main_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    main_python.chmod(0o755)

    (repo / ".gitignore").write_text(".venv/\n", encoding="utf-8")

    # Copy necessary scripts
    for script_name in (
        "check_context_notes.sh",
        "run_worktree_shared_venv.sh",
        "common_setup.sh",
        "check_worktree_optional_deps.py",
    ):
        source = ROOT / "scripts" / "dev" / script_name
        target = script_dir / script_name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)

    # Create dummy target scripts
    (validation_dir / "check_docs_proof_consistency.py").write_text(
        "print('proof ok')\n", encoding="utf-8"
    )
    (script_dir / "check_docs_evidence_integrity.py").write_text(
        "print('evidence ok')\n", encoding="utf-8"
    )
    (tools_dir / "check_context_note_freshness.py").write_text(
        "print('freshness ok')\n", encoding="utf-8"
    )
    (docs_dir / "INDEX.md").write_text("# Index\n", encoding="utf-8")
    (docs_dir / "catalog.yaml").write_text("catalog: {}\n", encoding="utf-8")
    (repo / "fast-pysf" / "pysocialforce").mkdir(parents=True)
    (repo / "fast-pysf" / "pysocialforce" / "__init__.py").write_text("", encoding="utf-8")

    # Set up fake uv to verify commands and enforce no local venv creation
    invocations_file = tmp_path / "uv_invocations.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf "%s\\n" "$*" >> "{invocations_file}"
# Verify UV_PROJECT_ENVIRONMENT is pointing to main venv
if [[ "${{UV_PROJECT_ENVIRONMENT:-}}" != "{main_venv}" ]]; then
  echo "ERROR: UV_PROJECT_ENVIRONMENT not set to main venv: ${{UV_PROJECT_ENVIRONMENT:-}}" >&2
  exit 1
fi
if [[ "${{UV_NO_SYNC:-}}" != "1" ]]; then
  echo "ERROR: UV_NO_SYNC not set to 1" >&2
  exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    configure_git_identity(repo, name="Agent", email="agent@example.invalid")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "base fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    # Create a fresh linked worktree without .venv
    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }

    # Execute check_context_notes.sh inside the fresh worktree
    result = subprocess.run(
        ["bash", str(worktree / "scripts" / "dev" / "check_context_notes.sh")],
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "1/3 changed-doc proof consistency" in result.stdout
    assert "2/3 evidence integrity" in result.stdout
    assert "3/3 context-note freshness" in result.stdout
    assert "OK check_context_notes.sh done" in result.stdout

    # Verify no .venv was created in the worktree
    assert not (worktree / ".venv").exists(), ".venv was created in fresh worktree!"

    # Verify all 3 steps were invoked through uv with proper shared venv exports
    invocations = invocations_file.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 3
    assert any("check_docs_proof_consistency.py" in line for line in invocations)
    assert any("check_docs_evidence_integrity.py" in line for line in invocations)
    assert any("check_context_note_freshness.py" in line for line in invocations)
