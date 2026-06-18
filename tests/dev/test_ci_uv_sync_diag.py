"""Sanity checks for the uv sync diagnostic helper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _script_path() -> Path:
    """Return the repository-local uv sync diagnostic script path."""
    return Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_uv_sync_diag.sh"


def _repo_root() -> Path:
    """Return the repository root for workflow contract checks."""
    return Path(__file__).resolve().parents[2]


def test_ci_uv_sync_diag_shell_syntax() -> None:
    """Validate that the diagnostic helper passes bash syntax checks."""
    script = _script_path()
    assert script.exists(), "ci_uv_sync_diag.sh helper is missing"
    assert subprocess.run(["bash", "-n", str(script)], check=False, timeout=30).returncode == 0


def test_ci_uv_sync_diag_runs_without_uv(tmp_path: Path) -> None:
    """The probe must remain advisory and exit 0 even when uv is not on PATH."""
    script = _script_path()
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required for this test"

    fake_bin = tmp_path / "fake-empty-bin"
    fake_bin.mkdir(exist_ok=True)
    try:
        env = os.environ.copy()
        env["PATH"] = str(fake_bin)
        env.pop("UV_CACHE_DIR", None)
        result = subprocess.run(
            [bash_path, str(script), "no-uv-test"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "::group::no-uv-test" in result.stdout
        assert "uv_version=not_installed" in result.stdout
        assert "::endgroup::" in result.stdout
    finally:
        shutil.rmtree(fake_bin, ignore_errors=True)


def test_ci_uv_sync_diag_reports_runner_and_uv_state() -> None:
    """When uv is available, the probe reports runner, cache, and venv state."""
    script = _script_path()
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required for this test"

    result = subprocess.run(
        [bash_path, str(script), "uv-available-test"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = result.stdout
    assert "::group::uv-available-test" in output
    assert "uv_sync_diag runner_info" in output
    assert "uv_sync_diag uv_info" in output
    assert "uv_sync_diag cache_size" in output
    assert "uv_sync_diag venv_info" in output
    assert "::endgroup::" in output
    if shutil.which("uv"):
        assert "uv_version=uv " in output


def test_workflow_uv_cache_paths_match_setup_uv_cache_dir() -> None:
    """Workflow uv caching must match the UV_CACHE_DIR configured by setup-uv."""
    inline_cache_workflows = [
        ".github/workflows/perf-nightly.yml",
        ".github/workflows/pr-promoted-planner-smoke.yml",
    ]
    expected_archive = "${{ runner.temp }}/setup-uv-cache/archive-v0"
    expected_wheels = "${{ runner.temp }}/setup-uv-cache/wheels-v6"

    ci_workflow_text = (_repo_root() / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    ci_action_text = (_repo_root() / ".github/actions/setup-ci-python/action.yml").read_text(
        encoding="utf-8"
    )
    assert "uses: ./.github/actions/setup-ci-python" in ci_workflow_text
    assert "astral-sh/setup-uv@" in ci_action_text
    assert "~/.cache/uv" not in ci_workflow_text
    assert "~/.cache/uv" not in ci_action_text

    for workflow_path in inline_cache_workflows:
        workflow_text = (_repo_root() / workflow_path).read_text(encoding="utf-8")
        assert expected_archive in workflow_text, workflow_path
        assert expected_wheels in workflow_text, workflow_path
        assert "~/.cache/uv" not in workflow_text, workflow_path
