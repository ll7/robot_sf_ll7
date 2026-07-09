"""Sanity checks for the uv sync diagnostic helper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _script_path() -> Path:
    """Return the repository-local uv sync diagnostic script path."""
    return Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_uv_sync_diag.sh"


def _repo_root() -> Path:
    """Return the repository root for workflow contract checks."""
    return Path(__file__).resolve().parents[2]


def _clean_diag_env(tmp_path: Path) -> dict[str, str]:
    """Return deterministic diagnostic env independent of host runner metadata."""
    env = os.environ.copy()
    for key in (
        "GITHUB_ACTIONS",
        "ROBOT_SF_TEST_ENV",
        "RUNNER_ARCH",
        "RUNNER_NAME",
        "RUNNER_OS",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_ID",
        "UV_CACHE_DIR",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)
    home_dir = tmp_path / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home_dir)
    env["RUNNER_OS"] = "Linux"
    env["RUNNER_ARCH"] = "X64"
    env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    return env


@pytest.fixture(autouse=True)
def _clear_host_runner_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep diagnostic assertions independent of LiCCA/GitHub ambient metadata."""
    for key in (
        "GITHUB_ACTIONS",
        "ROBOT_SF_TEST_ENV",
        "RUNNER_ARCH",
        "RUNNER_NAME",
        "RUNNER_OS",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_ID",
        "VIRTUAL_ENV",
    ):
        monkeypatch.delenv(key, raising=False)


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
        env = _clean_diag_env(tmp_path)
        env["PATH"] = str(fake_bin)
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


def test_ci_uv_sync_diag_cache_sizing_is_single_pass(tmp_path: Path) -> None:
    """Cache sizing must traverse the cache exactly once (issue #3703).

    The probe previously ran ``du`` once for the whole cache and then again per
    curated subdirectory, re-walking the tree up to a dozen times. This test
    pins the optimized contract: a single ``du`` invocation over the cache tree
    while the curated per-subdirectory keys and the total are still emitted.
    """
    script = _script_path()
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required for this test"

    # Build a fake uv cache populated with a few curated subdirectories.
    cache_dir = tmp_path / "uv-cache"
    for sub in ("archive-v0", "wheels-v6", "git-v0"):
        (cache_dir / sub).mkdir(parents=True)
        (cache_dir / sub / "blob").write_bytes(b"x" * 1024)

    # Fake ``du`` that records every invocation and emits ``du -h -d 1`` style
    # tab-separated output (one line per immediate subdir plus the cache root).
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    du_calls = tmp_path / "du_calls.log"
    du_shim = fake_bin / "du"
    du_shim.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{du_calls}"\n'
        'target="${@: -1}"\n'
        'for d in "$target"/*/; do\n'
        '  [ -d "$d" ] && printf "1.0K\\t%s\\n" "${d%/}"\n'
        "done\n"
        'printf "16K\\t%s\\n" "$target"\n'
    )
    du_shim.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["UV_CACHE_DIR"] = str(cache_dir)

    result = subprocess.run(
        [bash_path, str(script), "single-pass-test"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,  # no .venv here, so du is only invoked for the cache
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # The cache tree must be walked by exactly one du invocation (single pass).
    invocations = du_calls.read_text().splitlines() if du_calls.exists() else []
    cache_invocations = [line for line in invocations if str(cache_dir) in line]
    assert len(cache_invocations) == 1, f"expected single du pass, got: {cache_invocations}"

    # Curated per-subdirectory keys and the total are preserved.
    output = result.stdout
    assert "cache_total_size=16K" in output
    assert "cache_archive-v0_size=1.0K" in output
    assert "cache_wheels-v6_size=1.0K" in output
    assert "cache_git-v0_size=1.0K" in output


def test_workflow_uv_cache_paths_match_setup_uv_cache_dir() -> None:
    """uv payload caching must live at the setup-uv cache dir, centralized in setup-ci-python.

    perf-nightly and pr-promoted-planner-smoke delegate to the shared
    ``setup-ci-python`` composite action, which owns the uv payload cache. The
    cache paths must match the UV_CACHE_DIR configured by setup-uv (never the
    user-local ``~/.cache/uv`` default) so a restored cache is actually reused
    by the subsequent ``uv sync``.
    """
    delegated_workflows = [
        ".github/workflows/ci.yml",
        ".github/workflows/perf-nightly.yml",
        ".github/workflows/pr-promoted-planner-smoke.yml",
    ]
    expected_archive = "${{ runner.temp }}/setup-uv-cache/archive-v0"
    expected_wheels = "${{ runner.temp }}/setup-uv-cache/wheels-v6"

    action_text = (_repo_root() / ".github/actions/setup-ci-python/action.yml").read_text(
        encoding="utf-8"
    )
    # The shared action owns the cache and must pin it to setup-uv's location.
    assert "astral-sh/setup-uv@" in action_text
    assert expected_archive in action_text
    assert expected_wheels in action_text
    assert (
        "uv-sync-payloads-${{ runner.os }}-${{ hashFiles('pyproject.toml', 'uv.lock') }}"
        in action_text
    )
    assert "~/.cache/uv" not in action_text

    # Every consumer delegates to the shared action and must not re-declare a
    # local uv cache path.
    for workflow_path in delegated_workflows:
        workflow_text = (_repo_root() / workflow_path).read_text(encoding="utf-8")
        assert "uses: ./.github/actions/setup-ci-python" in workflow_text, workflow_path
        assert "~/.cache/uv" not in workflow_text, workflow_path


def test_perf_nightly_runs_xdist_race_validation_route() -> None:
    """Nightly performance workflow should include the scheduled xdist stress lane."""
    workflow_text = (_repo_root() / ".github/workflows/perf-nightly.yml").read_text(
        encoding="utf-8"
    )

    assert "Run xdist race validation" in workflow_text
    # PR #4948 mitigation (issue #4942): the nightly uses "auto" (4 workers on
    # the 4-vCPU runner) instead of the old hardcoded "32", which saturated the
    # ~16 GiB runner and tripped GitHub's SIGTERM/143 eviction watchdog.
    assert 'XDIST_RACE_WORKERS: "auto"' in workflow_text
    assert 'XDIST_RACE_TIMEOUT_SECONDS: "7200"' in workflow_text
    assert "PYTEST_XDIST_DIST: worksteal" in workflow_text
    assert "scripts/dev/run_xdist_race_validation.sh" in workflow_text
    assert "Upload xdist race validation artifacts" in workflow_text
    assert "path: output/validation/xdist-race/" in workflow_text


def test_perf_nightly_includes_xdist_memory_diagnostic_step() -> None:
    """Nightly should produce the worker-memory evidence behind issue #4942.

    The diagnostic measures how xdist peak memory scales with worker count and
    classifies the projection against the ~16 GiB runner ceiling, turning the
    one-shot local reproduction of the SIGTERM/143 eviction into a tracked,
    re-runnable nightly artifact. The step is non-gating so it can never break
    the nightly, and the sweep is capped well under the ceiling the old 32-worker
    setting crossed.
    """
    workflow_text = (_repo_root() / ".github/workflows/perf-nightly.yml").read_text(
        encoding="utf-8"
    )
    assert "Measure xdist worker-memory scaling" in workflow_text
    assert "scripts/dev/measure_xdist_worker_memory.py" in workflow_text
    assert "continue-on-error: true" in workflow_text
    assert "--ceiling-gb 16" in workflow_text
    assert "output/validation/xdist-memory/memory_diagnostic.json" in workflow_text
    assert "Upload xdist worker-memory diagnostic" in workflow_text
    assert "path: output/validation/xdist-memory/" in workflow_text
