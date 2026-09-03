"""Tests for the cheap shell preflight in scripts/dev/pr_ready_check.sh."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DEV = REPO_ROOT / "scripts" / "dev"
OPTIONAL_ALLOWLIST = REPO_ROOT / "tests" / "support" / "optional_test_allowlist.txt"

_FOCUSED_SNQI_SELECTOR_TARGETS = (
    "tests/unit/benchmark/test_snqi_campaign_contract.py",
    "tests/benchmark/test_camera_ready_campaign.py",
    "tests/tools/test_run_camera_ready_benchmark.py",
)
_UNRELATED_OPTIONAL_IMPORTS = (
    "torch",
    "stable_baselines3",
    "duckdb",
    "optuna",
    "pyarrow",
    "sqlalchemy",
)

_POST_PREFLIGHT_SCRIPTS = [
    "check_pr_followups.py",
    "check_perf_evidence.py",
    "check_fast_results_claim_map.py",
    "ruff_fix_format.sh",
    "run_tests_parallel.sh",
    "check_changed_coverage.sh",
    "check_docstring_todos_diff.sh",
    "check_docstring_todos_ratchet.sh",
    "check_docstring_todos_baseline_freshness.sh",
    "check_optional_import_pr_freshness.py",
    # check_base_drift.py is invoked by pr_ready_check.sh as a final base-drift
    # recheck before recording the freshness stamp (issue #5782).  Stub it here so
    # the fake repo exercises the wiring cleanly instead of emitting a missing-file
    # error that the rc-2 fallback silently swallows.
    "check_base_drift.py",
    "pr_ready_freshness.py",
]


def _git(repo: Path, *args: str) -> None:
    if args and args[0] == "commit":
        # Tolerate no-op commits: test scaffolding such as the fake bin/ tools and
        # .home are gitignored (so they never trip the issue #5533 untracked guard),
        # which can leave nothing staged to commit on subsequent add -A calls.
        proc = subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@test", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        merged = proc.stdout + proc.stderr
        if proc.returncode != 0 and "nothing to commit" not in merged:
            raise subprocess.CalledProcessError(
                proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr
            )
        return
    if args and args[0] == "add":
        # Tolerate no-op add -A: when bin/ is excluded by .git/info/exclude the
        # fake-python tests have nothing else to stage. Use --allow-empty for the
        # subsequent commit to cover this case.
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@test", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        return
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_fake_bin(repo: Path, *, fail: bool = True) -> None:
    """Create fake ``python`` and ``uv`` in *repo*/bin that simulate missing modules."""
    bin_dir = repo / "bin"
    bin_dir.mkdir(exist_ok=True)

    fake = bin_dir / "python"
    if fail:
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "has_stdin=0\n"
            'for arg in "$@"; do\n'
            '  if [[ "$arg" == "-" ]]; then\n'
            "    has_stdin=1\n"
            "    break\n"
            "  fi\n"
            "done\n"
            'if [[ "$has_stdin" -eq 1 ]]; then\n'
            '  payload="$* $(cat)"\n'
            "else\n"
            '  payload="$*"\n'
            "fi\n"
            "if [[ \"$payload\" == *'import importlib'* ]]; then\n"
            "  echo 'duckdb, pyarrow, pandas' >&2\n"
            "  exit 1\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
    else:
        fake.write_text(
            "#!/usr/bin/env bash\n"
            'payload=" $* "\n'
            'if [[ "$payload" == *"check_cuda_runtime.py"* && "$payload" == *" --json "* ]]; then\n'
            '  printf \'%s\\n\' \'{"schema":"cuda_runtime_readiness.v1","status":"unavailable","reason":"fixture"}\'\n'
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
    fake.chmod(0o755)

    real_uv = shutil.which("uv") or "uv"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "run" ]]; then\n'
        "  shift\n"
        '  exec "$@"\n'
        "fi\n"
        f'exec "{real_uv}" "$@"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)


def _make_real_python_bin(repo: Path) -> None:
    """Create a fake ``uv`` that execs the real Python interpreter.

    The base-drift end-to-end tests need ``check_base_drift.py`` to run with a real
    interpreter (so its git logic executes), while still avoiding ``uv sync`` and
    stubbing the expensive readiness lanes. Unlike ``_make_fake_bin``, this does NOT
    replace ``python``: ``uv run python <script>`` execs ``python <script>`` which PATH
    resolves to the real interpreter, so the real ``check_base_drift.py`` logic runs in
    interim mode (where the final-mode analytics preflight is skipped).
    """
    bin_dir = repo / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "run" ]]; then\n'
        "  shift\n"
        '  exec "$@"\n'
        "fi\n"
        f'exec "{shutil.which("uv") or "uv"}" "$@"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)


def _make_fake_scripts(repo: Path) -> None:
    """Create no-op stubs for every script ``pr_ready_check.sh`` calls after the preflight."""
    scripts_dir = repo / "scripts" / "dev"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPTS_DEV / "pr_ready_termination.py", scripts_dir / "pr_ready_termination.py")
    for name in _POST_PREFLIGHT_SCRIPTS:
        stub = scripts_dir / name
        if name.endswith(".py"):
            stub.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
        else:
            stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            stub.chmod(0o755)
    # pr_ready_check.sh also invokes scripts/validation/check_broad_exceptions.py,
    # which lives outside scripts/dev/; stub it so the preflight lane reaches the
    # post-preflight scripts without a missing-file error.
    validation_dir = repo / "scripts" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "check_broad_exceptions.py").write_text(
        "import sys\nsys.exit(0)\n", encoding="utf-8"
    )
    # PR #4865 made pr_ready_check.sh hard-require tests/support/optional_test_
    # allowlist.txt: is_optional_readiness_path reads it to classify each
    # changed test path (e.g. tests/planner/ -> optional, tests/unit/ -> core)
    # and exits 1 when it is absent. Copy the real file so the fake repo matches
    # production; the lane-split assertions below depend on its contents.
    support_dir = repo / "tests" / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OPTIONAL_ALLOWLIST, support_dir / "optional_test_allowlist.txt")

    # The optional-lane dependency probe runs before the optional test wrapper.
    # Keep the fake repository self-contained so lane-routing tests do not invoke
    # the host project's uv environment.
    bin_dir = repo / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "run" ]]; then\n'
        "  shift\n"
        '  exec "$@"\n'
        "fi\n"
        f'exec "{shutil.which("uv") or "uv"}" "$@"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    optional_dep_check = scripts_dir / "check_worktree_optional_deps.py"
    optional_dep_check.write_text(
        "import json\n"
        "print(json.dumps({'schema': 'robot_sf.worktree_optional_deps.v1', "
        "'profile': 'all-extras', 'status': 'ready', 'exit_code': 0, "
        "'missing_optional': [], 'check_failures': [], "
        "'project_imports_performed': False}))\n",
        encoding="utf-8",
    )
    shutil.copy2(
        SCRIPTS_DEV / "validate_worktree_optional_deps.py",
        scripts_dir / "validate_worktree_optional_deps.py",
    )


def _write_lane_logging_stub(repo: Path) -> Path:
    """Replace the test wrapper with a logger that records each lane invocation."""
    stub = repo / "scripts" / "dev" / "run_tests_parallel.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s %s\\n" "${ROBOT_SF_TEST_LANE:-unset}" "$*" >> "$PWD/lane.log"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return repo / "lane.log"


@pytest.fixture()
def preflight_repo(tmp_path: Path) -> Path:
    """Return a committed git repo with the preflight and stub scripts."""
    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts" / "dev"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(SCRIPTS_DEV / "common_setup.sh", scripts_dir / "common_setup.sh")
    shutil.copy2(SCRIPTS_DEV / "pr_ready_check.sh", scripts_dir / "pr_ready_check.sh")
    _make_fake_scripts(repo)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    # Keep test scaffolding (fake bin/ tools, .home, lane.log) out of the
    # untracked-file set that pr_ready_check.sh treats as real changed-file-proof
    # gaps, so only the files a test deliberately adds exercise the issue #5533 guard.
    (repo / ".git" / "info" / "exclude").write_text(
        "bin/\n.home/\nlane.log\noutput/\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _pr_ready_environment(
    repo: Path, env_overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Build an isolated environment for a fake-repository readiness process."""
    env = {**os.environ, "PATH": f"{repo / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    home = repo / ".home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    for key in (
        "BASE_REF",
        "GITHUB_ACTIONS",
        "GITHUB_BASE_REF",
        "GITHUB_HEAD_REF",
        "GITHUB_REF",
        "GITHUB_SHA",
        "GITHUB_WORKSPACE",
        "PR_READY_FINAL",
        "PR_READY_MODE",
        "PR_READY_SKIP_PREFLIGHT",
        "ROBOT_SF_TEST_ENV",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_ID",
    ):
        env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)
    return env


def _run_pr_ready(
    repo: Path,
    *,
    env_overrides: dict[str, str] | None = None,
    help_flag: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``pr_ready_check.sh`` and return the result."""
    cmd = ["scripts/dev/pr_ready_check.sh"]
    if help_flag:
        cmd.append("--help")
    return subprocess.run(
        cmd,
        cwd=repo,
        env=_pr_ready_environment(repo, env_overrides),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _start_pr_ready(repo: Path, *, env_overrides: dict[str, str]) -> subprocess.Popen[str]:
    """Start readiness in its own process group so signal cleanup is testable."""
    return subprocess.Popen(
        ["scripts/dev/pr_ready_check.sh"],
        cwd=repo,
        env=_pr_ready_environment(repo, env_overrides),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _write_blocking_lane_stub(repo: Path) -> None:
    """Make the core lane wait on external markers without running expensive tests."""
    stub = repo / "scripts" / "dev" / "run_tests_parallel.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "invoked\\n" >> "$PR_READY_LOCK_TEST_LOG"\n'
        ': > "$PR_READY_LOCK_TEST_READY"\n'
        'while [[ ! -e "$PR_READY_LOCK_TEST_RELEASE" ]]; do sleep 0.05; done\n'
        'exit "${PR_READY_LOCK_TEST_EXIT_CODE:-0}"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)


def _write_signal_lane_stub(repo: Path) -> None:
    """Make each readiness lane wait on its own marker for signal-path tests."""
    stub = repo / "scripts" / "dev" / "run_tests_parallel.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'case "${ROBOT_SF_TEST_LANE:-unknown}" in\n'
        '  core) ready="$PR_READY_SIGNAL_CORE_READY"; release="$PR_READY_SIGNAL_CORE_RELEASE" ;;\n'
        '  optional) ready="$PR_READY_SIGNAL_OPTIONAL_READY"; release="$PR_READY_SIGNAL_OPTIONAL_RELEASE" ;;\n'
        '  *) echo "unexpected readiness lane" >&2; exit 44 ;;\n'
        "esac\n"
        ': > "$ready"\n'
        'while [[ ! -e "$release" ]]; do sleep 0.05; done\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)


def _wait_for_marker(
    marker: Path, process: subprocess.Popen[str], *, timeout: float = 10.0
) -> None:
    """Wait for a controlled lane marker, reporting an early process failure."""
    deadline = time.monotonic() + timeout
    while not marker.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"readiness exited before marker: rc={process.returncode}\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        if time.monotonic() >= deadline:
            _stop_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"readiness did not reach marker within {timeout}s\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        time.sleep(0.02)


def _lock_test_environment(
    tmp_path: Path, *, ready: Path, release: Path, log: Path
) -> dict[str, str]:
    """Return shared lock-root and marker settings for lock-process tests."""
    temp_root = tmp_path / "lock-tmp"
    temp_root.mkdir(exist_ok=True)
    return {
        "TMPDIR": str(temp_root),
        "PR_READY_LOCK_DIR": str(temp_root / "robot-sf-pr-ready-locks"),
        "PR_READY_MODE": "interim",
        "PR_READY_LOCK_TEST_READY": str(ready),
        "PR_READY_LOCK_TEST_RELEASE": str(release),
        "PR_READY_LOCK_TEST_LOG": str(log),
    }


def _lock_anchor(tmp_path: Path, repo: Path) -> Path:
    """Return the expected temporary lock anchor for a canonical worktree path."""
    canonical = str(repo.resolve())
    key = hashlib.sha256(os.fsencode(canonical)).hexdigest()
    return tmp_path / "lock-tmp" / "robot-sf-pr-ready-locks" / f"{key}.lock"


def _stop_process_group(process: subprocess.Popen[str], signum: signal.Signals) -> None:
    """Terminate a controlled readiness process and any lane child it owns."""
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signum)
            else:
                process.send_signal(signum)
        except ProcessLookupError:
            pass


def _collect_process(process: subprocess.Popen[str], *, timeout: float = 10.0) -> tuple[str, str]:
    """Collect a controlled readiness process, failing without leaving a child behind."""
    try:
        return process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _stop_process_group(process, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise AssertionError(
            f"readiness did not exit within {timeout}s\nstdout={stdout}\nstderr={stderr}"
        ) from exc


def test_pr_ready_lock_acquires_and_rejects_same_worktree_contention(
    preflight_repo: Path, tmp_path: Path
) -> None:
    """A first run acquires the lock and a same-worktree retry fails without entering a lane."""
    ready = tmp_path / "first-ready"
    release = tmp_path / "first-release"
    log = tmp_path / "lane.log"
    _write_blocking_lane_stub(preflight_repo)
    env = _lock_test_environment(tmp_path, ready=ready, release=release, log=log)

    first = _start_pr_ready(preflight_repo, env_overrides=env)
    try:
        _wait_for_marker(ready, first)
        assert _lock_anchor(tmp_path, preflight_repo).is_file()

        started_at = time.monotonic()
        other_tmp = tmp_path / "different-tmp-root"
        other_tmp.mkdir()
        second = _start_pr_ready(
            preflight_repo,
            env_overrides={**env, "TMPDIR": str(other_tmp)},
        )
        second_stdout, second_stderr = _collect_process(second, timeout=3.0)
        elapsed = time.monotonic() - started_at

        assert elapsed < 3.0
        assert second.returncode == 2, second_stdout + second_stderr
        assert f"PR readiness is already running for worktree {preflight_repo.resolve()}" in (
            second_stderr
        )
        assert "Wait for that run to finish, then retry: scripts/dev/pr_ready_check.sh" in (
            second_stderr
        )
        assert log.read_text(encoding="utf-8").splitlines() == ["invoked"]

        release.touch()
        first_stdout, first_stderr = _collect_process(first)
        assert first.returncode == 0, first_stdout + first_stderr
    finally:
        _stop_process_group(first, signal.SIGKILL)
        _collect_process(first)


def test_pr_ready_lock_releases_after_readiness_failure(
    preflight_repo: Path, tmp_path: Path
) -> None:
    """A failing readiness lane releases its lock so the next run can acquire it."""
    failure_marker = tmp_path / "failed-once"
    stub = preflight_repo / "scripts" / "dev" / "run_tests_parallel.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ ! -e "$PR_READY_LOCK_TEST_FAILURE" ]]; then\n'
        '  : > "$PR_READY_LOCK_TEST_FAILURE"\n'
        "  exit 37\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    temp_root = tmp_path / "lock-tmp"
    temp_root.mkdir()
    env = {
        "TMPDIR": str(temp_root),
        "PR_READY_LOCK_DIR": str(temp_root / "robot-sf-pr-ready-locks"),
        "PR_READY_MODE": "interim",
        "PR_READY_LOCK_TEST_FAILURE": str(failure_marker),
    }

    first = _run_pr_ready(preflight_repo, env_overrides=env)
    assert first.returncode == 37, first.stderr
    second = _run_pr_ready(preflight_repo, env_overrides=env)
    assert second.returncode == 0, second.stderr
    assert "already running" not in second.stderr


@pytest.mark.skipif(
    os.name != "posix", reason="signal and process-group semantics are POSIX-specific"
)
def test_pr_ready_lock_releases_after_signal(preflight_repo: Path, tmp_path: Path) -> None:
    """A terminated readiness process does not leave a held lock behind."""
    ready = tmp_path / "signal-ready"
    release = tmp_path / "signal-release"
    log = tmp_path / "signal-lane.log"
    _write_blocking_lane_stub(preflight_repo)
    env = _lock_test_environment(tmp_path, ready=ready, release=release, log=log)

    first = _start_pr_ready(preflight_repo, env_overrides=env)
    try:
        _wait_for_marker(ready, first)
        _stop_process_group(first, signal.SIGTERM)
        _collect_process(first)
        assert first.returncode != 0

        ready.unlink()
        second_release = tmp_path / "signal-second-release"
        second_env = {**env, "PR_READY_LOCK_TEST_RELEASE": str(second_release)}
        second = _start_pr_ready(preflight_repo, env_overrides=second_env)
        try:
            _wait_for_marker(ready, second)
            second_release.touch()
            stdout, stderr = _collect_process(second)
            assert second.returncode == 0, stdout + stderr
        finally:
            _stop_process_group(second, signal.SIGKILL)
            _collect_process(second)
    finally:
        _stop_process_group(first, signal.SIGKILL)
        _collect_process(first)


@pytest.mark.skipif(
    os.name != "posix", reason="signal and process-group semantics are POSIX-specific"
)
def test_pr_ready_sigterm_writes_core_receipt_and_cleans_lane(
    preflight_repo: Path, tmp_path: Path
) -> None:
    """A direct wrapper SIGTERM records the core phase and verifies group cleanup."""
    _write_signal_lane_stub(preflight_repo)
    ready = tmp_path / "core-ready"
    release = tmp_path / "core-release"
    receipt = tmp_path / "core-termination.json"
    env = {
        "PR_READY_MODE": "interim",
        "PR_READY_TERMINATION_RECEIPT": str(receipt),
        "PR_READY_SIGNAL_CORE_READY": str(ready),
        "PR_READY_SIGNAL_CORE_RELEASE": str(release),
    }

    process = _start_pr_ready(preflight_repo, env_overrides=env)
    try:
        _wait_for_marker(ready, process)
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = _collect_process(process)

        assert process.returncode == 143, stdout + stderr
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["phase"] == "core_lane"
        assert payload["lane"] == "core"
        assert payload["signal"]["name"] == "SIGTERM"
        assert payload["signal"]["exit_code"] == 143
        assert payload["last_progress"]["message"] in {
            "starting core readiness lane",
            "core readiness lane running",
        }
        assert "received SIGTERM" not in payload["last_progress"]["message"]
        assert payload["cleanup"]["verified"] is True
        assert payload["process"]["child_process_group_exists"] is False, payload["process"]
        assert "environment" not in payload
        assert "command" not in payload
        assert "termination receipt:" in stderr
    finally:
        _stop_process_group(process, signal.SIGKILL)
        _collect_process(process)


@pytest.mark.skipif(
    os.name != "posix", reason="signal and process-group semantics are POSIX-specific"
)
def test_pr_ready_sigterm_writes_optional_receipt_and_cleans_lane(
    preflight_repo: Path, tmp_path: Path
) -> None:
    """The optional lane has distinct phase context and the same cleanup contract."""
    _write_signal_lane_stub(preflight_repo)
    changed_file = preflight_repo / "tests" / "planner" / "test_signal_optional.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("def test_signal_optional(): pass\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "optional signal fixture")

    core_ready = tmp_path / "core-ready"
    core_release = tmp_path / "core-release"
    optional_ready = tmp_path / "optional-ready"
    optional_release = tmp_path / "optional-release"
    receipt = tmp_path / "optional-termination.json"
    env = {
        "BASE_REF": "HEAD~1",
        "PR_READY_MODE": "interim",
        "PR_READY_SKIP_PREFLIGHT": "1",
        "PR_READY_TERMINATION_RECEIPT": str(receipt),
        "PR_READY_SIGNAL_CORE_READY": str(core_ready),
        "PR_READY_SIGNAL_CORE_RELEASE": str(core_release),
        "PR_READY_SIGNAL_OPTIONAL_READY": str(optional_ready),
        "PR_READY_SIGNAL_OPTIONAL_RELEASE": str(optional_release),
    }

    process = _start_pr_ready(preflight_repo, env_overrides=env)
    try:
        _wait_for_marker(core_ready, process)
        core_release.touch()
        _wait_for_marker(optional_ready, process)
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = _collect_process(process)

        assert process.returncode == 143, stdout + stderr
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["phase"] == "optional_lane"
        assert payload["lane"] == "optional"
        assert payload["signal"]["name"] == "SIGTERM"
        assert payload["cleanup"]["verified"] is True
        assert payload["process"]["child_process_group_exists"] is False
        assert "termination receipt:" in stderr
    finally:
        _stop_process_group(process, signal.SIGKILL)
        _collect_process(process)


@pytest.mark.skipif(
    os.name != "posix", reason="linked-worktree process semantics are POSIX-specific"
)
def test_pr_ready_lock_allows_distinct_linked_worktrees(
    preflight_repo: Path, tmp_path: Path
) -> None:
    """Linked worktrees sharing Git metadata can hold independent readiness locks."""
    worktrees = [tmp_path / "worktree-one", tmp_path / "worktree-two"]
    for worktree in worktrees:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
            cwd=preflight_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        _make_fake_bin(worktree, fail=True)
        _write_blocking_lane_stub(worktree)

    common_dirs = []
    for worktree in worktrees:
        raw_common_dir = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        common_dir = Path(raw_common_dir)
        if not common_dir.is_absolute():
            common_dir = worktree / common_dir
        common_dirs.append(common_dir.resolve())
    assert len(set(common_dirs)) == 1
    assert worktrees[0].resolve() != worktrees[1].resolve()

    ready_markers = [tmp_path / "worktree-one-ready", tmp_path / "worktree-two-ready"]
    release_markers = [tmp_path / "worktree-one-release", tmp_path / "worktree-two-release"]
    logs = [tmp_path / "worktree-one.log", tmp_path / "worktree-two.log"]
    processes = [
        _start_pr_ready(
            worktree,
            env_overrides=_lock_test_environment(tmp_path, ready=ready, release=release, log=log),
        )
        for worktree, ready, release, log in zip(
            worktrees, ready_markers, release_markers, logs, strict=True
        )
    ]
    try:
        for process, ready in zip(processes, ready_markers, strict=True):
            _wait_for_marker(ready, process)
        assert all(process.poll() is None for process in processes)
        assert _lock_anchor(tmp_path, worktrees[0]).is_file()
        assert _lock_anchor(tmp_path, worktrees[1]).is_file()
        assert _lock_anchor(tmp_path, worktrees[0]) != _lock_anchor(tmp_path, worktrees[1])

        for release in release_markers:
            release.touch()
        for process in processes:
            stdout, stderr = _collect_process(process)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            _stop_process_group(process, signal.SIGKILL)
            _collect_process(process)


def test_help_bypasses_preflight(preflight_repo: Path) -> None:
    """--help exits 0 before preflight runs, even with missing modules."""
    _make_fake_bin(preflight_repo, fail=True)
    result = _run_pr_ready(preflight_repo, help_flag=True)
    assert result.returncode == 0
    assert "Final PR readiness requires analytics dependencies" not in result.stderr


def test_preflight_helper_exits_nonzero_when_modules_are_missing() -> None:
    """The embedded Python must fail, not only print missing modules."""
    common_setup = (SCRIPTS_DEV / "common_setup.sh").read_text(encoding="utf-8")
    assert "raise SystemExit(1)" in common_setup
    assert 'for module_name in ("duckdb", "pyarrow", "pandas")' in common_setup


def test_preflight_fails_when_modules_missing(preflight_repo: Path) -> None:
    """Preflight should exit 2 with a concise error when modules are unavailable."""
    _make_fake_bin(preflight_repo, fail=True)
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "fake tools")
    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={"PR_READY_MODE": "final"},
    )
    assert result.returncode == 2, f"Expected exit 2, got {result.returncode}"
    assert "Final PR readiness requires analytics dependencies" in result.stderr
    assert "uv sync --all-extras" in result.stderr


def test_interim_mode_keeps_existing_non_preflight_path(preflight_repo: Path) -> None:
    """Default interim readiness should not run the final-only dependency preflight."""
    _make_fake_bin(preflight_repo, fail=True)
    result = _run_pr_ready(preflight_repo, help_flag=False)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Final PR readiness requires analytics dependencies" not in result.stderr


def test_pr_ready_check_escalates_optional_changed_files_to_the_optional_lane(
    preflight_repo: Path,
) -> None:
    """Predictive or optional-path changes should trigger the optional lane."""
    lane_log = _write_lane_logging_stub(preflight_repo)

    changed_file = preflight_repo / "tests" / "planner" / "test_sonic_crowdnav.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("print('optional lane')\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "optional lane change")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 0, result.stderr
    lane_lines = lane_log.read_text(encoding="utf-8").splitlines()
    assert lane_lines == [
        "core --lane core",
        "optional --lane optional",
    ]
    assert "Optional-extra changed files requiring the predictive lane" in result.stderr


def test_pr_ready_scopes_formatting_to_committed_python_changes(preflight_repo: Path) -> None:
    """A base-only unformatted file must not be mutated by readiness formatting."""
    base_only = preflight_repo / "base_only.py"
    base_only.write_text("value= 1\n", encoding="utf-8")
    _git(preflight_repo, "add", "base_only.py")
    _git(preflight_repo, "commit", "-q", "-m", "unformatted base-only fixture")

    changed = preflight_repo / "changed.py"
    changed.write_text("value = 2\n", encoding="utf-8")
    _git(preflight_repo, "add", "changed.py")
    _git(preflight_repo, "commit", "-q", "-m", "changed python file")

    formatter = preflight_repo / "scripts" / "dev" / "ruff_fix_format.sh"
    formatter.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$PWD/format-targets.log"\nexit 0\n',
        encoding="utf-8",
    )
    formatter.chmod(0o755)
    exclude_file = preflight_repo / ".git" / "info" / "exclude"
    exclude_file.write_text(
        exclude_file.read_text(encoding="utf-8") + "format-targets.log\n",
        encoding="utf-8",
    )

    result = _run_pr_ready(
        preflight_repo,
        env_overrides={"BASE_REF": "HEAD~1", "PR_READY_MODE": "interim"},
    )

    assert result.returncode == 0, result.stderr
    assert (preflight_repo / "format-targets.log").read_text(encoding="utf-8").splitlines() == [
        "changed.py"
    ]
    assert base_only.read_text(encoding="utf-8") == "value= 1\n"


def test_optional_lane_preflight_reports_missing_extras_as_setup_evidence(
    preflight_repo: Path,
) -> None:
    """Missing optional imports block that lane with structured, actionable evidence."""
    lane_log = _write_lane_logging_stub(preflight_repo)
    _make_real_python_bin(preflight_repo)

    optional_dep_check = preflight_repo / "scripts" / "dev" / "check_worktree_optional_deps.py"
    optional_dep_check.write_text(
        "import json\n"
        "print(json.dumps({\n"
        "    'schema': 'robot_sf.worktree_optional_deps.v1',\n"
        "    'profile': 'all-extras',\n"
        "    'status': 'missing_optional',\n"
        "    'exit_code': 2,\n"
        "    'missing_optional': ['pandas'],\n"
        "    'check_failures': [],\n"
        "    'project_imports_performed': False,\n"
        "}))\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    changed_file = preflight_repo / "tests" / "planner" / "test_missing_extra.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("def test_missing_extra(): pass\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "optional dependency setup evidence")

    result = _run_pr_ready(
        preflight_repo,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 2, result.stderr
    assert '"status": "missing_optional"' in result.stderr
    assert "pandas" in result.stderr
    assert "This is setup evidence, not a changed-code failure." in result.stderr
    assert "uv sync --all-extras" in result.stderr
    assert lane_log.read_text(encoding="utf-8").splitlines() == ["core --lane core"]


@pytest.mark.parametrize(
    ("probe_output", "probe_exit_code"),
    [
        (
            '{"schema":"robot_sf.worktree_optional_deps.v1","profile":"all-extras",'
            '"status":"check_failed","exit_code":1,"missing_optional":[],'
            '"check_failures":["pandas"],"project_imports_performed":false}',
            1,
        ),
        (
            '{"schema":"robot_sf.worktree_optional_deps.v1","profile":"all-extras",'
            '"status":"ready","exit_code":0,"missing_optional":[],'
            '"check_failures":[],"project_imports_performed":false}',
            1,
        ),
        ("not-json", 1),
        (
            '{"schema":"robot_sf.worktree_optional_deps.v1","profile":"all-extras",'
            '"status":"ready","exit_code":7,"missing_optional":[],'
            '"check_failures":[],"project_imports_performed":false}',
            7,
        ),
    ],
    ids=["probe-failure", "status-exit-disagreement", "malformed-json", "unknown-exit"],
)
def test_optional_lane_preflight_rejects_invalid_probe_contract(
    preflight_repo: Path,
    probe_output: str,
    probe_exit_code: int,
) -> None:
    """Probe failures and contract disagreements never receive install guidance."""
    lane_log = _write_lane_logging_stub(preflight_repo)
    _make_real_python_bin(preflight_repo)
    optional_dep_check = preflight_repo / "scripts" / "dev" / "check_worktree_optional_deps.py"
    optional_dep_check.write_text(
        f"print({probe_output!r})\nraise SystemExit({probe_exit_code})\n",
        encoding="utf-8",
    )
    changed_file = preflight_repo / "tests" / "planner" / "test_invalid_preflight.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("def test_invalid_preflight(): pass\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "invalid optional dependency evidence")

    result = _run_pr_ready(
        preflight_repo,
        env_overrides={"BASE_REF": "HEAD~1", "PR_READY_MODE": "interim"},
    )

    assert result.returncode == 1, result.stderr
    assert "Optional dependency preflight tool failure" in result.stderr
    assert "uv sync --all-extras" not in result.stderr
    assert lane_log.read_text(encoding="utf-8").splitlines() == ["core --lane core"]


def test_pr_ready_check_keeps_core_only_changes_on_the_core_lane(preflight_repo: Path) -> None:
    """Core-only changes should not schedule the optional lane."""
    lane_log = _write_lane_logging_stub(preflight_repo)

    changed_file = preflight_repo / "tests" / "unit" / "test_core_lane.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("print('core lane')\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "core lane change")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 0, result.stderr
    lane_lines = lane_log.read_text(encoding="utf-8").splitlines()
    assert lane_lines == ["core --lane core"]
    assert "No committed changed files require the optional-extra lane." in result.stderr


def test_pr_ready_coverage_database_parent_survives_lanes_and_reporting(
    preflight_repo: Path,
    tmp_path: Path,
) -> None:
    """Readiness owns one absolute coverage DB until every lane and report completes."""
    lifetime_log = preflight_repo / ".home" / "coverage-lifetime.log"
    coverage_tmp = tmp_path / "coverage-tmp"
    coverage_tmp.mkdir()

    lane_stub = preflight_repo / "scripts" / "dev" / "run_tests_parallel.sh"
    lane_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        '[[ "$COVERAGE_FILE" == /* ]] || { echo "coverage path is not absolute" >&2; exit 41; }\n'
        'coverage_parent="$(dirname "$COVERAGE_FILE")"\n'
        '[[ -d "$coverage_parent" ]] || { echo "coverage parent missing in lane" >&2; exit 42; }\n'
        'if [[ -s "$COVERAGE_LIFETIME_LOG" ]]; then\n'
        '  first_path="$(sed -n \'1s/^[^:]*://p\' "$COVERAGE_LIFETIME_LOG")"\n'
        '  [[ "$first_path" == "$COVERAGE_FILE" ]] || { echo "coverage path changed" >&2; exit 43; }\n'
        "fi\n"
        'touch "$COVERAGE_FILE"\n'
        'printf "lane:%s\\n" "$COVERAGE_FILE" >> "$COVERAGE_LIFETIME_LOG"\n',
        encoding="utf-8",
    )
    lane_stub.chmod(0o755)

    report_stub = preflight_repo / "scripts" / "dev" / "check_changed_coverage.sh"
    report_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'coverage_parent="$(dirname "$COVERAGE_FILE")"\n'
        '[[ -d "$coverage_parent" ]] || { echo "coverage parent missing in report" >&2; exit 44; }\n'
        '[[ -f "$COVERAGE_FILE" ]] || { echo "coverage database missing in report" >&2; exit 45; }\n'
        'printf "report:%s\\n" "$COVERAGE_FILE" >> "$COVERAGE_LIFETIME_LOG"\n',
        encoding="utf-8",
    )
    report_stub.chmod(0o755)

    changed_file = preflight_repo / "tests" / "planner" / "test_coverage_lifetime.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("def test_optional(): pass\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "optional coverage lifetime fixture")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "COVERAGE_FILE": "relative/unstable/.coverage",
            "COVERAGE_LIFETIME_LOG": str(lifetime_log),
            "PR_READY_MODE": "interim",
            "TMPDIR": str(coverage_tmp),
        },
    )

    assert result.returncode == 0, result.stderr
    records = lifetime_log.read_text(encoding="utf-8").splitlines()
    assert [record.split(":", maxsplit=1)[0] for record in records] == [
        "lane",
        "lane",
        "report",
    ]
    coverage_paths = [Path(record.split(":", maxsplit=1)[1]) for record in records]
    assert len(set(coverage_paths)) == 1
    coverage_file = coverage_paths[0]
    assert coverage_file.is_absolute()
    assert coverage_file.parent.parent == coverage_tmp
    assert not coverage_file.parent.exists()


def test_interim_mode_reports_dirty_paths_excluded_from_changed_file_gates(
    preflight_repo: Path,
) -> None:
    """Dirty paths are explicit when interim diff-scoped gates only inspect HEAD.

    Uses tracked-but-modified paths only: the issue #5533 untracked-new-file guard
    has its own regression test and must not be tripped here.
    """
    lane_log = _write_lane_logging_stub(preflight_repo)
    tracked_file = preflight_repo / "tests" / "unit" / "test_dirty_tracked.py"
    tracked_file.parent.mkdir(parents=True, exist_ok=True)
    tracked_file.write_text("print('baseline')\n", encoding="utf-8")
    dirty_optional = preflight_repo / "tests" / "planner" / "test_dirty_optional.py"
    dirty_optional.parent.mkdir(parents=True, exist_ok=True)
    dirty_optional.write_text("print('baseline dirty optional')\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "tracked dirty fixture")
    tracked_file.write_text("print('dirty tracked path')\n", encoding="utf-8")
    dirty_optional.write_text("print('dirty optional lane')\n", encoding="utf-8")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 0, result.stderr
    assert lane_log.read_text(encoding="utf-8").splitlines() == ["core --lane core"]
    assert "Interim changed-file scope is committed HEAD vs HEAD." in result.stderr
    assert "Dirty paths excluded from diff-scoped gates:" in result.stderr
    assert "tests/unit/test_dirty_tracked.py" in result.stderr
    assert "tests/planner/test_dirty_optional.py" in result.stderr


def test_pr_ready_check_fails_clearly_when_untracked_new_files_exist(
    preflight_repo: Path,
) -> None:
    """Regression for issue #5533: a new-file-only worktree must not silently report changed-file proof.

    Untracked new files are invisible to the committed-HEAD diff gates (changed-file
    coverage, docstring TODO diff), which previously printed a misleading
    "No changed files" while omitting the only changed code. The wrapper must now
    fail clearly (exit 2) and name the untracked files instead of claiming proof.
    """
    # New-file-only worktree: nothing committed, only an untracked implementation file.
    new_file = preflight_repo / "robot_sf" / "new_module.py"
    new_file.parent.mkdir(parents=True, exist_ok=True)
    new_file.write_text("print('new implementation')\n", encoding="utf-8")

    # Interim mode previously proceeded with a misleading "No changed files" message.
    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={"PR_READY_MODE": "interim"},
    )
    assert result.returncode == 2, f"Expected exit 2, got {result.returncode}: {result.stderr}"
    assert "Changed-file proof cannot see the following untracked new files:" in result.stderr
    assert "robot_sf/new_module.py" in result.stderr
    assert "readiness cannot prove them" in result.stderr

    # Final/proof mode must also fail clearly rather than fabricate changed-file proof.
    result_final = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={"PR_READY_MODE": "final"},
    )
    assert result_final.returncode == 2, result_final.stderr
    assert "robot_sf/new_module.py" in result_final.stderr


def test_core_lane_collection_hook_skips_optional_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pytest collection hook should keep optional files out of the core lane."""
    import tests.conftest as test_conftest

    forecast_packet_test = Path("tests/prediction/test_forecast_heavy_model_decision_packet.py")
    monkeypatch.setenv("ROBOT_SF_TEST_LANE", "core")
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/planner/test_sonic_crowdnav.py"), None)
        is True
    )
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/unit/test_config_validation.py"), None)
        is False
    )
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/dev/test_pr_ready_preflight.py"), None)
        is False
    )
    assert test_conftest.pytest_ignore_collect(forecast_packet_test, None) is True
    assert test_conftest._is_optional_readiness_test_path(forecast_packet_test.as_posix()) is True
    assert (
        test_conftest._is_optional_readiness_test_path(
            "/tmp/tests-parent/repo/tests/planner/test_sonic_crowdnav.py::test_case"
        )
        is True
    )


def test_optional_lane_collection_hook_skips_core_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional lane should collect only optional-extra test paths."""
    import tests.conftest as test_conftest

    monkeypatch.setenv("ROBOT_SF_TEST_LANE", "optional")
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/planner/test_sonic_crowdnav.py"), None)
        is False
    )
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/dev/test_pr_ready_preflight.py"), None)
        is True
    )
    assert (
        test_conftest.pytest_ignore_collect(
            Path("tests/prediction/test_forecast_heavy_model_decision_packet.py"), None
        )
        is False
    )


def test_ped_npc_collection_stays_in_core_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pedestrian population tests are core-only, not optional-extra tests (#5753)."""
    import tests.conftest as test_conftest

    ped_npc_test = Path("tests/ped_npc/test_force_population_size_split.py")
    assert test_conftest._is_optional_readiness_test_path(ped_npc_test.as_posix()) is False

    monkeypatch.setenv("ROBOT_SF_TEST_LANE", "core")
    assert test_conftest.pytest_ignore_collect(ped_npc_test, None) is False

    monkeypatch.setenv("ROBOT_SF_TEST_LANE", "optional")
    assert test_conftest.pytest_ignore_collect(ped_npc_test, None) is True


def test_focused_snqi_selector_collects_without_unrelated_optional_stacks(tmp_path: Path) -> None:
    """Explicit SNQI targets must collect when unrelated optional imports are unavailable."""
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import builtins\n"
        f"BLOCKED = {set(_UNRELATED_OPTIONAL_IMPORTS)!r}\n"
        "original_import = builtins.__import__\n"
        "def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name.split('.', maxsplit=1)[0] in BLOCKED:\n"
        "        raise ModuleNotFoundError(f'blocked optional import: {name}')\n"
        "    return original_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = guarded_import\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *_FOCUSED_SNQI_SELECTOR_TARGETS,
            "-k",
            "snqi_contract or exit or camera_ready_summary",
            "--collect-only",
            "-q",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_base_ref_falls_back_to_head_without_crashing(preflight_repo: Path) -> None:
    """A BASE_REF that does not resolve must fall back to HEAD, not leak a git fatal.

    Regression for issue #3702: on fresh checkouts the default origin/main is not
    always present. The unresolved ref previously leaked a raw
    ``fatal: ambiguous argument`` from the ``git diff "$BASE_REF...HEAD"`` call
    (silently swallowed by ``set -e`` inside the process substitution) and left an
    invalid base ref flowing to the downstream coverage/perf/freshness checks.
    """
    _make_fake_bin(preflight_repo, fail=True)
    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "non_existent_branch",
            "PR_READY_MODE": "interim",
        },
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Falling back to BASE_REF=HEAD" in result.stderr
    # The raw git error must not leak; the missing ref is handled gracefully.
    assert "fatal" not in result.stderr.lower()
    assert "unknown revision" not in result.stderr.lower()


def test_valid_base_ref_is_used_unchanged(preflight_repo: Path) -> None:
    """A BASE_REF that resolves must be used as-is, with no fallback or fetch.

    Guards against the issue #3702 fix degrading the normal path: a valid ref
    (here HEAD~1) should never trigger the HEAD fallback or a fetch attempt.
    """
    _make_fake_bin(preflight_repo, fail=True)
    extra = preflight_repo / "tests" / "unit" / "test_valid_base_ref.py"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("print('valid base ref')\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "valid base ref change")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Falling back to BASE_REF=HEAD" not in result.stderr
    assert "does not resolve to a local commit" not in result.stderr
    assert "Attempting git fetch" not in result.stderr


def test_preflight_passes_when_modules_available(preflight_repo: Path) -> None:
    """Preflight should pass silently when python reports no missing modules."""
    _make_fake_bin(preflight_repo, fail=False)
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "fake tools")
    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={"PR_READY_MODE": "final"},
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Final PR readiness requires analytics dependencies" not in result.stderr


def test_preflight_skip_env_var_bypasses_check(preflight_repo: Path) -> None:
    """PR_READY_SKIP_PREFLIGHT=1 should skip the preflight entirely."""
    _make_fake_bin(preflight_repo, fail=True)
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "fake tools")
    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "PR_READY_SKIP_PREFLIGHT": "1",
            "PR_READY_MODE": "final",
        },
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Final PR readiness requires analytics dependencies" not in result.stderr


# Issue #5782: end-to-end wiring of the base-drift recheck in pr_ready_check.sh.
# These build a fake repo whose post-preflight lanes are stubbed (so the expensive
# pytest/coverage gates are skipped) but whose check_base_drift.py is the REAL
# script. The ``run_tests_parallel.sh`` stub advances origin/main mid-run to
# deterministically model main advancing during the long lanes, then the gate
# fails closed on related drift and surfaces the reuse path on unrelated drift.


def _wire_real_base_drift(repo: Path) -> None:
    """Replace the stubbed check_base_drift.py with the real script from this checkout."""
    shutil.copy2(
        SCRIPTS_DEV / "check_base_drift.py",
        repo / "scripts" / "dev" / "check_base_drift.py",
    )


def _commit(repo: Path, message: str) -> None:
    """Stage and commit all changes in *repo* with a test identity."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def _build_drift_pr_repo(tmp_path: Path, *, main_touches_pr_file: bool) -> tuple[Path, str, str]:
    """Build a fake repo with a PR branch on a base commit and an advanced main commit.

    Returns (repo, base_sha, moved_sha). The PR branch sits on top of ``base_sha``
    and changes ``shared.py``; a separate ``main`` branch advances one commit past
    ``base_sha``. When *main_touches_pr_file* is True that advance edits
    ``shared.py`` (related drift); otherwise it edits ``unrelated.py`` (unrelated
    drift). ``origin/main`` starts at ``base_sha`` so a run captures the validated
    base SHA; the lane stub advances it to ``moved_sha`` mid-run.
    """
    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts" / "dev"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(SCRIPTS_DEV / "common_setup.sh", scripts_dir / "common_setup.sh")
    shutil.copy2(SCRIPTS_DEV / "pr_ready_check.sh", scripts_dir / "pr_ready_check.sh")
    _make_fake_scripts(repo)
    _wire_real_base_drift(repo)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".git" / "info" / "exclude").write_text(
        "bin/\n.home/\nlane.log\noutput/\n", encoding="utf-8"
    )

    (repo / "shared.py").write_text("print('base')\n", encoding="utf-8")
    (repo / "unrelated.py").write_text("print('base')\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "init")
    (repo / "shared.py").write_text("print('base commit')\n", encoding="utf-8")
    _commit(repo, "base")
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    # PR commit on top of base changes shared.py.
    (repo / "shared.py").write_text("print('pr change')\n", encoding="utf-8")
    _commit(repo, "pr")
    pr_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    # Advance a main branch one commit past the base (simulating main moving).
    # Git may initialize the repository with ``main`` already configured as the
    # default branch.  Detach the PR commit first, then move that branch to the
    # base commit instead of assuming the name is available for creation.
    subprocess.run(["git", "checkout", "-q", "--detach"], cwd=repo, check=True)
    subprocess.run(["git", "branch", "-f", "main", base_sha], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    if main_touches_pr_file:
        (repo / "shared.py").write_text("print('main moved shared')\n", encoding="utf-8")
    else:
        (repo / "unrelated.py").write_text("print('main moved unrelated')\n", encoding="utf-8")
    _commit(repo, "main advance")
    moved_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    # origin/main starts AT THE BASE so the run captures the validated base SHA; the
    # lane stub advances it to moved_sha mid-run (see _wire_lane_stub_advancing_base).
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", base_sha], cwd=repo, check=True
    )
    # Return HEAD to the PR commit.
    subprocess.run(["git", "checkout", "-q", pr_commit], cwd=repo, check=True)
    return repo, base_sha, moved_sha


def _wire_lane_stub_advancing_base(repo: Path, moved_sha: str) -> None:
    """Make the stubbed ``run_tests_parallel.sh`` advance origin/main mid-run.

    pr_ready_check.sh captures the validated base SHA BEFORE the expensive lanes and
    rechecks it AFTER. In production, origin/main advances during those long lanes.
    The stubbed lanes complete instantly, so this stub advances origin/main to
    *moved_sha* while the lane runs, deterministically modelling that passage of
    time so the final drift recheck observes the moved base.
    """
    stub = repo / "scripts" / "dev" / "run_tests_parallel.sh"
    stub.write_text(
        f"#!/usr/bin/env bash\ngit update-ref refs/remotes/origin/main {moved_sha}\nexit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)


def test_pr_ready_check_fails_closed_on_related_base_drift(tmp_path: Path) -> None:
    """Issue #5782: drift touching a PR-changed file must fail the gate before stamping.

    The expensive pytest/coverage lanes are stubbed. The ``run_tests_parallel.sh``
    stub models the passage of time during the long lanes by advancing origin/main
    (as an unrelated PR merging would). pr_ready_check.sh captures the base SHA at
    the start, so when the final drift recheck sees the advanced base editing the
    PR's own file, it must exit nonzero and name the base to revalidate instead of
    recording a misleading stamp.
    """
    repo, _base_sha, moved = _build_drift_pr_repo(tmp_path, main_touches_pr_file=True)
    _make_real_python_bin(repo)
    _wire_lane_stub_advancing_base(repo, moved)

    result = _run_pr_ready(
        repo,
        help_flag=False,
        env_overrides={"BASE_REF": "origin/main", "PR_READY_MODE": "interim"},
    )
    assert result.returncode != 0, (
        f"expected base-drift failure, got rc={result.returncode}: {result.stderr}"
    )
    assert "revalidate against origin/main" in result.stderr
    assert "shared.py" in result.stderr
    assert "Validated base SHA for this run" in result.stderr


def test_pr_ready_check_surfaces_reuse_path_on_unrelated_base_drift(tmp_path: Path) -> None:
    """Issue #5782: drift unrelated to the PR's changed files recommends reuse.

    origin/main advances during the (stubbed) lanes but only edits a file the PR
    does not touch. The gate must still succeed and surface the reviewable reuse
    message rather than needlessly failing or silently ignoring the drift.
    """
    repo, _base_sha, moved = _build_drift_pr_repo(tmp_path, main_touches_pr_file=False)
    _make_real_python_bin(repo)
    _wire_lane_stub_advancing_base(repo, moved)

    result = _run_pr_ready(
        repo,
        help_flag=False,
        env_overrides={"BASE_REF": "origin/main", "PR_READY_MODE": "interim"},
    )
    assert result.returncode == 0, f"expected reuse success, got: {result.stderr}"
    # The reuse decision must be visible (reviewable), not silently swallowed.
    assert "reuse" in result.stderr.lower()


def test_publication_preflight_lane_coverage_routing(preflight_repo: Path) -> None:
    """Issue #5937: Verify that publication-preflight test and module routing works correctly."""
    # 1. Assert that the Python conftest side classifies the test file as optional.
    import tests.conftest as test_conftest

    assert (
        test_conftest._is_optional_readiness_test_path(
            "tests/validation/test_publication_preflight.py"
        )
        is True
    )

    # 2. Assert that the bash script classifies robot_sf/benchmark/artifact_publication.py as optional.
    lane_log = _write_lane_logging_stub(preflight_repo)

    changed_file = preflight_repo / "robot_sf" / "benchmark" / "artifact_publication.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("# publication module changes\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "benchmark publication change")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 0, result.stderr
    lane_lines = lane_log.read_text(encoding="utf-8").splitlines()
    # When an optional file changes, both core and optional lanes must be run
    assert lane_lines == [
        "core --lane core",
        "optional --lane optional",
    ]
    assert "Optional-extra changed files requiring the predictive lane" in result.stderr
    assert "robot_sf/benchmark/artifact_publication.py" in result.stderr


def test_pr_ready_check_fails_closed_when_changed_optional_test_omitted_from_allowlist(
    preflight_repo: Path,
) -> None:
    """Issue #6208: PR readiness must fail closed when a changed optional test is omitted from allowlist."""
    # Omit tests/planner/ from allowlist file
    allowlist = preflight_repo / "tests" / "support" / "optional_test_allowlist.txt"
    allowlist.write_text("tests/benchmark/\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "update allowlist fixture")

    changed_file = preflight_repo / "tests" / "planner" / "test_unlisted_opt.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("def test_unlisted(): pass\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "unlisted optional test change")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 2
    assert "omitted from optional test allowlist" in result.stderr
    assert "Expected lane: optional" in result.stderr
    assert "Actual collection decision: omitted" in result.stderr
    assert "Remediation:" in result.stderr


def test_pr_ready_check_regression_shapes_classification(preflight_repo: Path) -> None:
    """Issue #6208: Verify top-level, nested core, and optional regression shapes are properly classified."""
    lane_log = _write_lane_logging_stub(preflight_repo)

    top_level = preflight_repo / "tests" / "test_toplevel_shape.py"
    top_level.write_text("def test_top(): pass\n", encoding="utf-8")

    nested_core = preflight_repo / "tests" / "ped_npc" / "test_nested_shape.py"
    nested_core.parent.mkdir(parents=True, exist_ok=True)
    nested_core.write_text("def test_nested(): pass\n", encoding="utf-8")

    opt_test = preflight_repo / "tests" / "planner" / "test_optional_shape.py"
    opt_test.parent.mkdir(parents=True, exist_ok=True)
    opt_test.write_text("def test_opt(): pass\n", encoding="utf-8")

    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "add 3 regression shapes")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 0, result.stderr
    lane_lines = lane_log.read_text(encoding="utf-8").splitlines()
    assert lane_lines == [
        "core --lane core",
        "optional --lane optional",
    ]
