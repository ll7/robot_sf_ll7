"""Behavior and contract tests for the uv sync retry helper (issue #4889).

The helper wraps ``uv sync`` with bounded retry + exponential backoff so a
transient PyPI / index connection error (the ``nvidia-cufft-cu12`` download
failure that killed ``wheel-smoke-install``) no longer fails a CI run on its
first occurrence. ``uv sync --frozen`` resolves against a fixed lock, so the
only realistic non-zero exits are network/download failures, which are
transient; a genuine failure reproduces on every attempt and still fails the
step once the retry budget is exhausted.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


def _script_path() -> Path:
    """Return the repository-local uv sync retry helper path."""
    return Path(__file__).resolve().parents[2] / "scripts" / "dev" / "uv_sync_retry.sh"


def _repo_root() -> Path:
    """Return the repository root for workflow contract checks."""
    return Path(__file__).resolve().parents[2]


def _action_path() -> Path:
    """Return the shared CI Python setup composite action path."""
    return _repo_root() / ".github" / "actions" / "setup-ci-python" / "action.yml"


def _write_stub_uv(
    bin_dir: Path,
    *,
    counter: Path,
    log: Path,
    succeed_on: int,
) -> None:
    """Write a fake ``uv`` that fails until the ``succeed_on``-th call.

    Each invocation increments a counter file, appends its args to a log, and
    exits non-zero (simulating a transient connection error) until the call
    count reaches ``succeed_on``.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "uv"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            n="$(cat '{counter}' 2>/dev/null || echo 0)"
            n=$((n + 1))
            printf '%s\\n' "$n" > '{counter}'
            printf 'call %s: %s\\n' "$n" "$*" >> '{log}'
            if [[ "$n" -ge {succeed_on} ]]; then
              exit 0
            fi
            echo "stub uv: simulated transient connection error (call $n)" >&2
            exit 1
            """
        )
    )
    stub.chmod(0o755)


def _write_stub_sleep(bin_dir: Path, log: Path) -> None:
    """Write a fake ``sleep`` that records its argument instead of sleeping."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "sleep"
    stub.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$1\" >> '{log}'\n")
    stub.chmod(0o755)


def _run_wrapper(
    tmp_path: Path,
    *,
    args: list[str],
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run the retry helper under bash with the given args and environment."""
    script = _script_path()
    bash_path = shutil.which("bash")
    assert bash_path, "bash is required for these tests"
    return subprocess.run(
        [bash_path, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
        timeout=60,
    )


def _fast_retry_env(tmp_path: Path, *, with_uv: bool = True) -> tuple[Path, dict[str, str]]:
    """Return a fake-bin dir and an env with instant backoff and a stub uv on PATH."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Instant backoff so retry tests do not wait on real sleeps.
    env["UV_SYNC_BACKOFF_BASE"] = "0"
    env["UV_SYNC_BACKOFF_CAP"] = "0"
    env["UV_SYNC_MAX_ATTEMPTS"] = "3"
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    if with_uv:
        _write_stub_uv(
            fake_bin,
            counter=tmp_path / "counter",
            log=tmp_path / "uv.log",
            succeed_on=1,
        )
    return fake_bin, env


def test_uv_sync_retry_shell_syntax() -> None:
    """The helper must pass bash syntax checks and be executable."""
    script = _script_path()
    assert script.exists(), "uv_sync_retry.sh helper is missing"
    assert subprocess.run(["bash", "-n", str(script)], check=False, timeout=30).returncode == 0
    assert os.access(script, os.X_OK), "uv_sync_retry.sh must be executable"


def test_uv_sync_retry_succeeds_first_try(tmp_path: Path) -> None:
    """A sync that succeeds immediately must run once and exit 0."""
    fake_bin, env = _fast_retry_env(tmp_path)
    _write_stub_uv(fake_bin, counter=tmp_path / "counter", log=tmp_path / "uv.log", succeed_on=1)

    result = _run_wrapper(tmp_path, args=["--", "--all-extras", "--frozen"], env=env)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "uv sync (attempt 1/3)" in result.stdout
    assert "uv_sync_retry success attempt=1/3" in result.stdout
    calls = (tmp_path / "uv.log").read_text().splitlines()
    assert len(calls) == 1, f"expected one uv call, got: {calls}"


def test_uv_sync_retry_retries_then_succeeds(tmp_path: Path) -> None:
    """A transient failure on the first two calls must be absorbed by retry."""
    fake_bin, env = _fast_retry_env(tmp_path)
    _write_stub_uv(fake_bin, counter=tmp_path / "counter", log=tmp_path / "uv.log", succeed_on=3)

    result = _run_wrapper(tmp_path, args=["--", "--all-extras", "--frozen"], env=env)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "uv sync (attempt 1/3)" in result.stdout
    assert "uv sync (attempt 2/3)" in result.stdout
    assert "uv sync (attempt 3/3)" in result.stdout
    assert "uv_sync_retry transient status=1" in result.stdout
    assert "uv_sync_retry success attempt=3/3" in result.stdout
    calls = (tmp_path / "uv.log").read_text().splitlines()
    assert len(calls) == 3, f"expected three uv calls, got: {calls}"


def test_uv_sync_retry_fails_after_max_attempts(tmp_path: Path) -> None:
    """Persistent failure must exhaust the retry budget and exit non-zero."""
    fake_bin, env = _fast_retry_env(tmp_path)
    env["UV_SYNC_MAX_ATTEMPTS"] = "2"
    # succeed_on higher than any attempt count -> always fails.
    _write_stub_uv(fake_bin, counter=tmp_path / "counter", log=tmp_path / "uv.log", succeed_on=999)

    result = _run_wrapper(tmp_path, args=["--", "--frozen"], env=env)

    assert result.returncode == 1, f"expected non-zero exit, stdout: {result.stdout}"
    assert "uv sync (attempt 1/2)" in result.stdout
    assert "uv sync (attempt 2/2)" in result.stdout
    # The terminal failure banner is written to stderr (>&2).
    assert "failed after 2 attempt(s)" in result.stdout + result.stderr
    calls = (tmp_path / "uv.log").read_text().splitlines()
    assert len(calls) == 2, f"expected two uv calls, got: {calls}"


def test_uv_sync_retry_status_127_not_retried(tmp_path: Path) -> None:
    """A missing uv binary (status 127) is non-transient and must not retry."""
    # No stub uv on PATH at all.
    _, env = _fast_retry_env(tmp_path, with_uv=False)
    env["PATH"] = "/usr/bin:/bin"  # coreutils present, no uv.

    result = _run_wrapper(tmp_path, args=["--", "--frozen"], env=env)

    assert result.returncode == 127, (
        f"expected 127, stdout: {result.stdout}, stderr: {result.stderr}"
    )
    assert "not retryable" in result.stdout + result.stderr
    # Only one attempt group; no transient-retry banner.
    assert "uv_sync_retry transient" not in result.stdout


def test_uv_sync_retry_forwards_args_to_uv(tmp_path: Path) -> None:
    """Args after the ``--`` separator must be forwarded verbatim to uv sync."""
    fake_bin, env = _fast_retry_env(tmp_path)
    _write_stub_uv(fake_bin, counter=tmp_path / "counter", log=tmp_path / "uv.log", succeed_on=1)

    result = _run_wrapper(tmp_path, args=["--", "--all-extras", "--frozen"], env=env)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    log_text = (tmp_path / "uv.log").read_text()
    assert "call 1: sync --all-extras --frozen" in log_text, log_text


def test_uv_sync_retry_defaults_to_locked_sync_args(tmp_path: Path) -> None:
    """With no args, the helper defaults to ``--all-extras --frozen``."""
    fake_bin, env = _fast_retry_env(tmp_path)
    _write_stub_uv(fake_bin, counter=tmp_path / "counter", log=tmp_path / "uv.log", succeed_on=1)

    result = _run_wrapper(tmp_path, args=[], env=env)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    log_text = (tmp_path / "uv.log").read_text()
    assert "call 1: sync --all-extras --frozen" in log_text, log_text


def test_uv_sync_retry_invalid_max_attempts_defaults_to_three(tmp_path: Path) -> None:
    """A non-positive ``UV_SYNC_MAX_ATTEMPTS`` falls back to 3 with a warning."""
    fake_bin, env = _fast_retry_env(tmp_path)
    env["UV_SYNC_MAX_ATTEMPTS"] = "not-a-number"
    _write_stub_uv(fake_bin, counter=tmp_path / "counter", log=tmp_path / "uv.log", succeed_on=999)

    result = _run_wrapper(tmp_path, args=["--", "--frozen"], env=env)

    assert result.returncode == 1
    assert "defaulting to 3" in result.stdout + result.stderr
    calls = (tmp_path / "uv.log").read_text().splitlines()
    assert len(calls) == 3, f"expected default of 3 attempts, got: {calls}"


def test_uv_sync_retry_backoff_is_exponential_and_capped(tmp_path: Path) -> None:
    """Backoff doubles per attempt (base, base*2, ...) up to the cap."""
    tmp = tmp_path / "case"
    fake_bin = tmp / "bin"
    fake_bin.mkdir(parents=True)
    sleep_log = tmp / "sleep.log"
    _write_stub_sleep(fake_bin, sleep_log)
    _write_stub_uv(fake_bin, counter=tmp / "counter", log=tmp / "uv.log", succeed_on=3)

    env = os.environ.copy()
    env["UV_SYNC_MAX_ATTEMPTS"] = "3"
    env["UV_SYNC_BACKOFF_BASE"] = "2"
    env["UV_SYNC_BACKOFF_CAP"] = "10"
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    result = _run_wrapper(tmp, args=["--", "--frozen"], env=env)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    # attempt 1 -> backoff base*2**0 = 2; attempt 2 -> base*2**1 = 4; attempt 3 is last (no sleep).
    sleeps = sleep_log.read_text().splitlines() if sleep_log.exists() else []
    assert sleeps == ["2", "4"], f"expected exponential backoff [2, 4], got: {sleeps}"


def test_uv_sync_retry_backoff_caps_at_maximum(tmp_path: Path) -> None:
    """Backoff must not exceed ``UV_SYNC_BACKOFF_CAP``."""
    tmp = tmp_path / "case"
    fake_bin = tmp / "bin"
    fake_bin.mkdir(parents=True)
    sleep_log = tmp / "sleep.log"
    _write_stub_sleep(fake_bin, sleep_log)
    # Always fail so we exercise several backoff steps within a small cap.
    _write_stub_uv(fake_bin, counter=tmp / "counter", log=tmp / "uv.log", succeed_on=999)

    env = os.environ.copy()
    env["UV_SYNC_MAX_ATTEMPTS"] = "4"
    env["UV_SYNC_BACKOFF_BASE"] = "5"
    env["UV_SYNC_BACKOFF_CAP"] = "8"
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    result = _run_wrapper(tmp, args=["--", "--frozen"], env=env)

    assert result.returncode == 1
    # base=5: 5, 10->capped to 8, 20->capped to 8; attempt 4 is last (no sleep).
    sleeps = sleep_log.read_text().splitlines() if sleep_log.exists() else []
    assert sleeps == ["5", "8", "8"], f"expected capped backoff [5, 8, 8], got: {sleeps}"


# --- Workflow / composite-action contract checks ---


def test_setup_ci_python_uses_uv_sync_retry_wrapper() -> None:
    """The shared setup action must wrap uv sync in the retry helper."""
    action_text = _action_path().read_text(encoding="utf-8")
    assert "scripts/dev/uv_sync_retry.sh" in action_text
    # The original bare `uv sync --all-extras --frozen` invocation is gone.
    assert "uv sync --all-extras --frozen\n" not in action_text


def test_setup_ci_python_caches_setup_uv_payloads() -> None:
    """The setup action must cache uv payloads at the setup-uv cache location."""
    action_text = _action_path().read_text(encoding="utf-8")
    assert "${{ runner.temp }}/setup-uv-cache/archive-v0" in action_text
    assert "${{ runner.temp }}/setup-uv-cache/wheels-v6" in action_text
    assert (
        "uv-sync-payloads-${{ runner.os }}-${{ hashFiles('pyproject.toml', 'uv.lock') }}"
        in action_text
    )


def test_setup_ci_python_reports_cache_restore_outputs() -> None:
    """The pre-sync diagnostic must surface the actions/cache restore outputs."""
    action_text = _action_path().read_text(encoding="utf-8")
    assert "steps.cache-uv.outputs.cache-hit" in action_text
    assert "steps.cache-uv.outputs.cache-matched-key" in action_text


def test_setup_ci_python_has_no_local_uv_cache_path() -> None:
    """The setup action must not use the local ``~/.cache/uv`` cache path.

    Reinforces the convention pinned by ``test_ci_uv_sync_diag``: uv payloads
    live under ``runner.temp/setup-uv-cache`` (managed by setup-uv), never at
    the user-local default cache dir.
    """
    action_text = _action_path().read_text(encoding="utf-8")
    assert "~/.cache/uv" not in action_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
