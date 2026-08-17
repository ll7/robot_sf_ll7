"""Contract and behavior tests for the pinned uv setup fallback (issue #7368)."""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path


def _repo_root() -> Path:
    """Return the repository root for workflow contract checks."""
    return Path(__file__).resolve().parents[2]


def _script_path() -> Path:
    """Return the bounded uv installation helper path."""
    return _repo_root() / "scripts" / "dev" / "ci_install_uv_retry.sh"


def _action_path() -> Path:
    """Return the shared CI setup composite action path."""
    return _repo_root() / ".github" / "actions" / "setup-ci-python" / "action.yml"


def _write_stub_python(
    bin_dir: Path,
    *,
    counter: Path,
    log: Path,
    install_bin: Path,
    succeed_on: int,
) -> None:
    """Write a fake Python that simulates a transient pip installation."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "python"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            if [[ "$1" == "-m" && "${{2:-}}" == "pip" ]]; then
              n="$(cat '{counter}' 2>/dev/null || echo 0)"
              n=$((n + 1))
              printf '%s\\n' "$n" > '{counter}'
              printf '%s\\n' "$*" >> '{log}'
              if [[ "$n" -lt {succeed_on} ]]; then
                echo "stub pip: simulated transient index failure (call $n)" >&2
                exit 1
              fi
              mkdir -p '{install_bin}'
              cat > '{install_bin}/uv' <<'UV_STUB'
            #!/usr/bin/env bash
            printf 'uv %s\\n' "${{STUB_UV_VERSION}}"
            UV_STUB
              chmod 755 '{install_bin}/uv'
              exit 0
            fi
            echo "unexpected fake python invocation: $*" >&2
            exit 2
            """
        )
    )
    stub.chmod(0o755)


def _write_stub_uv(install_bin: Path) -> None:
    """Write an already-installed exact uv stub."""
    install_bin.mkdir(parents=True, exist_ok=True)
    stub = install_bin / "uv"
    stub.write_text("#!/usr/bin/env bash\nprintf 'uv %s\\n' \"${STUB_UV_VERSION}\"\n")
    stub.chmod(0o755)


def _write_stub_sleep(bin_dir: Path, log: Path) -> None:
    """Write a sleep shim that records backoff without waiting."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "sleep"
    stub.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$1\" >> '{log}'\n")
    stub.chmod(0o755)


def _run_helper(tmp_path: Path, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the helper with a deterministic temporary environment."""
    bash = shutil.which("bash")
    assert bash, "bash is required by ci_install_uv_retry.sh"
    return subprocess.run(
        [bash, str(_script_path()), "0.11.21"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )


def _retry_env(tmp_path: Path, *, succeed_on: int) -> tuple[dict[str, str], Path, Path]:
    """Build fake Python/sleep tools and the helper environment."""
    fake_bin = tmp_path / "bin"
    install_bin = tmp_path / "user-bin"
    counter = tmp_path / "counter"
    pip_log = tmp_path / "pip.log"
    _write_stub_python(
        fake_bin,
        counter=counter,
        log=pip_log,
        install_bin=install_bin,
        succeed_on=succeed_on,
    )
    _write_stub_sleep(fake_bin, tmp_path / "sleep.log")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "STUB_UV_VERSION": "0.11.21",
            "UV_INSTALL_BIN_DIR": str(install_bin),
            "UV_INSTALL_MAX_ATTEMPTS": "3",
            "UV_INSTALL_BACKOFF_BASE": "0",
            "UV_INSTALL_BACKOFF_CAP": "0",
            "GITHUB_PATH": str(tmp_path / "github_path"),
        }
    )
    return env, counter, pip_log


def test_ci_install_uv_retry_shell_syntax_and_executable() -> None:
    """The helper must be executable and pass Bash syntax validation."""
    script = _script_path()
    assert script.exists(), "ci_install_uv_retry.sh helper is missing"
    assert script.stat().st_mode & 0o111, "ci_install_uv_retry.sh must be executable"
    assert subprocess.run(["bash", "-n", str(script)], check=False, timeout=30).returncode == 0


def test_ci_install_uv_retry_retries_transient_pip_failure(tmp_path: Path) -> None:
    """Transient package-index failures are retried and then version-verified."""
    env, counter, pip_log = _retry_env(tmp_path, succeed_on=3)

    result = _run_helper(tmp_path, env=env)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Install pinned uv (attempt 1/3)" in result.stdout
    assert "Install pinned uv (attempt 3/3)" in result.stdout
    assert "ci_install_uv_retry success source=pypi version=0.11.21 attempt=3/3" in result.stdout
    assert counter.read_text().strip() == "3"
    assert "uv==0.11.21" in pip_log.read_text()
    assert (tmp_path / "github_path").read_text().strip() == str(tmp_path / "user-bin")


def test_ci_install_uv_retry_fails_closed_after_budget(tmp_path: Path) -> None:
    """Persistent pip failure must remain a failed setup, not a skipped test."""
    env, counter, _ = _retry_env(tmp_path, succeed_on=99)
    env["UV_INSTALL_MAX_ATTEMPTS"] = "2"

    result = _run_helper(tmp_path, env=env)

    assert result.returncode == 1, f"stdout: {result.stdout}, stderr: {result.stderr}"
    assert "failed after 2 attempt(s)" in result.stdout + result.stderr
    assert counter.read_text().strip() == "2"


def test_ci_install_uv_retry_reuses_exact_existing_binary(tmp_path: Path) -> None:
    """An exact preinstalled version avoids an unnecessary network request."""
    env, counter, pip_log = _retry_env(tmp_path, succeed_on=99)
    _write_stub_uv(tmp_path / "user-bin")

    result = _run_helper(tmp_path, env=env)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "reuse source=path version=0.11.21" in result.stdout
    assert not counter.exists()
    assert not pip_log.exists()


def test_ci_install_uv_retry_rejects_wrong_installed_version(tmp_path: Path) -> None:
    """A successful install that yields the wrong version must fail closed."""
    env, _, _ = _retry_env(tmp_path, succeed_on=1)
    env["STUB_UV_VERSION"] = "0.11.20"

    result = _run_helper(tmp_path, env=env)

    assert result.returncode == 1
    assert "expected 'uv 0.11.21'" in result.stdout + result.stderr


def test_shared_setup_keeps_pinned_action_and_adds_fail_closed_fallback() -> None:
    """The shared action preserves setup-uv cache behavior with a bounded fallback."""
    action_text = _action_path().read_text(encoding="utf-8")
    assert "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9" in action_text
    assert "continue-on-error: true" in action_text
    assert "steps.setup-uv.outcome == 'failure'" in action_text
    assert "scripts/dev/ci_install_uv_retry.sh" in action_text
    assert "uv --version" in action_text
    assert 'enable-cache: "true"' in action_text
    assert 'prune-cache: "true"' in action_text


def test_direct_uv_workflows_use_the_retry_helper() -> None:
    """Non-composite uv consumers must not retain an unwrapped action download."""
    for relative_path in (".github/workflows/ci.yml", ".github/workflows/packaging-extras.yml"):
        workflow_text = (_repo_root() / relative_path).read_text(encoding="utf-8")
        assert "scripts/dev/ci_install_uv_retry.sh" in workflow_text, relative_path
        assert "astral-sh/setup-uv@" not in workflow_text, relative_path
        assert 'UV_INSTALL_MAX_ATTEMPTS: "3"' in workflow_text, relative_path
