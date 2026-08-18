"""Fail-closed venv guards for the docs-proof diff entry point (issue #7478)."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_PROOF_DIFF = REPO_ROOT / "scripts" / "dev" / "check_docs_proof_consistency_diff.sh"
COMMON_SETUP = REPO_ROOT / "scripts" / "dev" / "common_setup.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _docs_proof_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a tiny git checkout with the script, common_setup, and a fake uv."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    scripts = repo / "scripts" / "dev"
    scripts.mkdir(parents=True)
    shutil.copy2(DOCS_PROOF_DIFF, scripts / "check_docs_proof_consistency_diff.sh")
    shutil.copy2(COMMON_SETUP, scripts / "common_setup.sh")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "uv-capture.txt"
    _write_executable(
        fake_bin / "uv",
        "#!/usr/bin/env bash\nprintf 'uv-invoked\\n' >> \"$CI_CAPTURE\"\n",
    )
    return repo, fake_bin, capture


def _run_docs_proof(
    repo: Path, fake_bin: Path, capture: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo / "scripts" / "dev" / "check_docs_proof_consistency_diff.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_docs_proof_diff_fails_without_local_venv(tmp_path: Path) -> None:
    """A fresh checkout without .venv must fail closed instead of creating one."""
    repo, fake_bin, capture = _docs_proof_fixture(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CI_CAPTURE": str(capture),
    }

    result = _run_docs_proof(repo, fake_bin, capture, env)

    assert result.returncode == 2
    diagnostic = result.stdout + result.stderr
    assert "no usable local virtual environment" in diagnostic
    assert "scripts/dev/bootstrap_worktree.sh" in diagnostic
    assert not capture.exists(), "uv must not run without a usable venv"


def test_docs_proof_diff_fails_when_local_venv_is_incomplete(tmp_path: Path) -> None:
    """A .venv missing required modules must fail closed with the bootstrap command."""
    repo, fake_bin, capture = _docs_proof_fixture(tmp_path)
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _write_executable(
        venv_bin / "python",
        "#!/usr/bin/env bash\nexit 1\n",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CI_CAPTURE": str(capture),
    }

    result = _run_docs_proof(repo, fake_bin, capture, env)

    assert result.returncode == 2
    diagnostic = result.stdout + result.stderr
    assert "incomplete: 'yaml' is not importable" in diagnostic
    assert "scripts/dev/bootstrap_worktree.sh" in diagnostic
    assert not capture.exists(), "uv must not run against an incomplete venv"
