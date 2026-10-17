"""Regression coverage for common_setup's selected dependency profile."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_SETUP = REPO_ROOT / "scripts" / "dev" / "common_setup.sh"


def _fake_python(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$0" "$@" > "$PREFLIGHT_MARKER"\n'
        'exit "${FAKE_PYTHON_STATUS:-0}"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _fake_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    checker = repo / "scripts" / "dev" / "check_worktree_optional_deps.py"
    checker.parent.mkdir(parents=True)
    checker.write_text("# fake checker path for the interpreter marker\n", encoding="utf-8")

    local_python = repo / ".venv" / "bin" / "python"
    shared_python = tmp_path / "shared-venv" / "bin" / "python"
    _fake_python(local_python)
    _fake_python(shared_python)
    return repo, local_python, shared_python


def _run_preflight(
    repo: Path,
    marker: Path,
    *,
    explicit_venv: str | None = None,
    virtual_env: str | None = None,
    uv_project_environment: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "ROBOT_SF_EXPLICIT_VENV_OVERRIDE",
    ):
        env.pop(name, None)
    env["PREFLIGHT_MARKER"] = str(marker)
    if explicit_venv is not None:
        env["ROBOT_SF_EXPLICIT_VENV_OVERRIDE"] = explicit_venv
    if virtual_env is not None:
        env["VIRTUAL_ENV"] = virtual_env
    if uv_project_environment is not None:
        env["UV_PROJECT_ENVIRONMENT"] = uv_project_environment

    return subprocess.run(
        [
            "bash",
            "-c",
            f"source {COMMON_SETUP}; preflight_check_worktree_dependency_profile all-extras",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_explicit_shared_venv_wins_over_partial_local_venv(tmp_path: Path) -> None:
    """Matching wrapper markers keep a local partial environment from shadowing."""
    repo, local_python, shared_python = _fake_repo(tmp_path)
    local_python.unlink()
    marker = tmp_path / "preflight.marker"
    selected = str(shared_python.parent.parent)

    result = _run_preflight(
        repo,
        marker,
        explicit_venv=selected,
        virtual_env=selected,
        uv_project_environment=selected,
    )

    assert result.returncode == 0, result.stderr
    observed = marker.read_text(encoding="utf-8").splitlines()
    assert observed == [
        str(shared_python),
        str(repo / "scripts" / "dev" / "check_worktree_optional_deps.py"),
        "--profile",
        "all-extras",
    ]


def test_no_explicit_selection_keeps_local_venv_preference(tmp_path: Path) -> None:
    """Without matching selection markers, the existing local choice is retained."""
    repo, local_python, _shared_python = _fake_repo(tmp_path)
    marker = tmp_path / "preflight.marker"

    result = _run_preflight(repo, marker)

    assert result.returncode == 0, result.stderr
    observed = marker.read_text(encoding="utf-8").splitlines()
    assert observed[0] == str(local_python)
    assert observed[-1] == "all-extras"


def test_mismatched_selection_markers_do_not_override_local_venv(tmp_path: Path) -> None:
    """An explicit marker is trusted only while all three paths still agree."""
    repo, local_python, shared_python = _fake_repo(tmp_path)
    marker = tmp_path / "preflight.marker"

    result = _run_preflight(
        repo,
        marker,
        explicit_venv=str(shared_python.parent.parent),
        virtual_env=str(shared_python.parent.parent),
        uv_project_environment=str(local_python.parent.parent),
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8").splitlines()[0] == str(local_python)


def test_incomplete_explicit_venv_fails_closed_even_with_local_venv(
    tmp_path: Path,
) -> None:
    """A complete local environment cannot mask an incomplete explicit choice."""
    repo, _local_python, shared_python = _fake_repo(tmp_path)
    shared_python.unlink()
    marker = tmp_path / "preflight.marker"
    selected = str(shared_python.parent.parent)

    result = _run_preflight(
        repo,
        marker,
        explicit_venv=selected,
        virtual_env=selected,
        uv_project_environment=selected,
    )

    assert result.returncode == 2
    assert "cannot use incomplete virtualenv" in result.stderr
    assert "bootstrap_worktree.sh" in result.stderr
    assert not marker.exists()
