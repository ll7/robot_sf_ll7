"""Deterministic capacity and scratch contracts for local CI wrappers."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

from tests.support.environment_guards import configure_git_identity

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CI_LOCAL = REPO_ROOT / "scripts" / "dev" / "run_ci_local.sh"
RUN_SHARED_VENV = REPO_ROOT / "scripts" / "dev" / "run_worktree_shared_venv.sh"
RECOVER_FAST_PYSF = REPO_ROOT / "scripts" / "dev" / "recover_fast_pysf_worktree.sh"
RUN_DOCS_PROOF = REPO_ROOT / "scripts" / "dev" / "check_docs_proof_consistency_diff.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_df(fake_bin: Path, available_kib: int) -> None:
    _write_executable(
        fake_bin / "df",
        "#!/usr/bin/env bash\n"
        "cat <<'EOF'\n"
        "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
        f"fixture 1000000 999000 {available_kib} 99% /\n"
        "EOF\n",
    )


def _env_with_fake_bin(fake_bin: Path) -> dict[str, str]:
    return {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}


def _local_runner_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the local runner beside no-op dependencies so phases stay deterministic."""
    scripts = tmp_path / "scripts" / "dev"
    scripts.mkdir(parents=True)
    runner = scripts / "run_ci_local.sh"
    shutil.copy2(RUN_CI_LOCAL, runner)
    _write_executable(scripts / "common_setup.sh", "#!/usr/bin/env bash\nset -euo pipefail\n")
    _write_executable(
        scripts / "ci_driver.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == "--list-phases" ]]; then\n'
        "  printf '%s\\n' smoke\n"
        "  exit 0\n"
        "fi\n"
        "printf 'phase=%s TMPDIR=%s UV_CACHE_DIR=%s XDG_CACHE_HOME=%s MPLCONFIGDIR=%s\\n' "
        '  "$*" "$TMPDIR" "${UV_CACHE_DIR:-}" "${XDG_CACHE_HOME:-}" '
        '  "${MPLCONFIGDIR:-}" >> "$CI_CAPTURE"\n',
    )
    return runner, tmp_path / "ci-capture.txt"


def test_run_ci_local_fails_before_a_phase_when_scratch_is_low(tmp_path: Path) -> None:
    """Low capacity must stop local CI before the phase driver starts."""
    runner, capture = _local_runner_fixture(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_df(fake_bin, available_kib=64)
    ambient_tmp = tmp_path / "ambient-tmp"
    ambient_tmp.mkdir()

    env = _env_with_fake_bin(fake_bin)
    env.update(
        {
            "CI_CAPTURE": str(capture),
            "TMPDIR": str(ambient_tmp),
            "ROBOT_SF_CI_MIN_FREE_BYTES": str(1024 * 1024),
        }
    )
    result = subprocess.run(
        [str(runner), "--no-setup", "smoke"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    diagnostic = result.stdout + result.stderr
    assert "local CI scratch preflight failed" in diagnostic
    assert "No CI phase was started" in diagnostic
    assert not capture.exists()


def test_run_ci_local_scratch_dir_redirects_default_temp_and_caches(tmp_path: Path) -> None:
    """The opt-in scratch directory must reach the phase driver through the environment."""
    runner, capture = _local_runner_fixture(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_df(fake_bin, available_kib=10_000_000)
    scratch = tmp_path / "disk-backed-scratch"

    env = _env_with_fake_bin(fake_bin)
    env.update(
        {
            "CI_CAPTURE": str(capture),
            "ROBOT_SF_CI_MIN_FREE_BYTES": "0",
        }
    )
    result = subprocess.run(
        [str(runner), "--no-setup", "--scratch-dir", str(scratch), "smoke"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    expected_root = scratch.resolve()
    captured = capture.read_text(encoding="utf-8")
    assert f"TMPDIR={expected_root / 'tmp'}" in captured
    assert f"UV_CACHE_DIR={expected_root / 'uv-cache'}" in captured
    assert f"XDG_CACHE_HOME={expected_root / 'xdg-cache'}" in captured
    assert f"MPLCONFIGDIR={expected_root / 'mplconfig'}" in captured
    for child in ("tmp", "uv-cache", "xdg-cache", "mplconfig"):
        assert (expected_root / child).is_dir()


def _shared_venv_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a tiny git checkout, usable venv marker, and fake uv command."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _write_executable(venv_bin / "python", "#!/usr/bin/env bash\nexit 0\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "shared-capture.txt"
    _write_executable(
        fake_bin / "uv",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'TMPDIR=%s UV_CACHE_DIR=%s XDG_CACHE_HOME=%s MPLCONFIGDIR=%s UV_NO_SYNC=%s\\n' "
        '  "$TMPDIR" "${UV_CACHE_DIR:-}" "${XDG_CACHE_HOME:-}" '
        '  "${MPLCONFIGDIR:-}" "${UV_NO_SYNC:-}" >> "$CI_CAPTURE"\n',
    )
    return repo, fake_bin, capture


def _linked_recovery_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, dict[str, str]]:
    """Build a linked worktree and fake uv for explicit fast-pysf recovery tests."""
    repo = tmp_path / "main"
    repo.mkdir()
    script_dir = repo / "scripts" / "dev"
    script_dir.mkdir(parents=True)
    for source in (
        RUN_SHARED_VENV,
        RECOVER_FAST_PYSF,
        REPO_ROOT / "scripts" / "dev" / "check_fast_pysf_runtime.py",
        REPO_ROOT / "scripts" / "dev" / "check_worktree_capacity.py",
        REPO_ROOT / "scripts" / "dev" / "check_worktree_optional_deps.py",
    ):
        target = script_dir / source.name
        shutil.copy2(source, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)

    source_package = repo / "fast-pysf" / "pysocialforce"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text("\n", encoding="utf-8")
    (source_package / "forces.py").write_text(
        "def social_force_gil_releasing_context():\n    return None\n", encoding="utf-8"
    )
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "recovery-fixture"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    (repo / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    configure_git_identity(repo, name="Recovery Fixture", email="recovery@example.invalid")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "recovery fixture"], cwd=repo, check=True, capture_output=True
    )

    worktree = tmp_path / "linked-worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "uv-calls.txt"
    sync_started = tmp_path / "sync-started"
    _write_executable(
        fake_bin / "uv",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$UV_CAPTURE"\n'
        'case "${1:-}" in\n'
        "  venv)\n"
        '    target="${2:?missing venv path}"\n'
        '    mkdir -p "$target/bin"\n'
        "    cat > \"$target/bin/python\" <<'PY'\n"
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == *check_fast_pysf_runtime.py ]]; then\n'
        '  printf "fast-pysf runtime preflight passed\\n"\n'
        "fi\n"
        "exit 0\n"
        "PY\n"
        '    chmod +x "$target/bin/python"\n'
        "    ;;\n"
        "  sync)\n"
        '    if [[ -n "${UV_SYNC_STARTED:-}" ]]; then : > "$UV_SYNC_STARTED"; fi\n'
        '    if [[ "${UV_SYNC_SLEEP:-0}" != "0" ]]; then sleep "$UV_SYNC_SLEEP"; fi\n'
        "    ;;\n"
        "  run)\n"
        "    ;;\n"
        '  *) printf "unexpected uv invocation: %s\\n" "$*" >&2; exit 9 ;;\n'
        "esac\n",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "ROBOT_SF_CI_MIN_FREE_BYTES": "0",
        "ROBOT_SF_WORKTREE_MIN_FREE_BYTES": "0",
        "UV_CAPTURE": str(capture),
    }
    env.pop("PYTHONPATH", None)
    return repo, worktree, fake_bin, capture, sync_started, env


def _remove_linked_recovery_fixture(repo: Path, worktree: Path) -> None:
    """Remove only the temporary linked worktree owned by a recovery fixture."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=repo,
        capture_output=True,
        check=False,
    )


def test_recover_fast_pysf_helper_has_explicit_usage() -> None:
    """The recovery helper documents its ownership and no-implicit-main contract."""
    result = subprocess.run(
        [str(RECOVER_FAST_PYSF), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "current linked worktree's .venv" in result.stdout
    assert "refuses the main checkout" in result.stdout
    assert "ROBOT_SF_WORKTREE_MIN_FREE_BYTES" in result.stdout
    assert "--frozen" in result.stdout


def test_shared_venv_recovery_refreshes_stale_package_in_worktree(tmp_path: Path) -> None:
    """Explicit recovery creates only a local env and reaches the wrapped command."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    try:
        result = subprocess.run(
            [
                str(worktree / "scripts" / "dev" / RUN_SHARED_VENV.name),
                "--recover-stale-fast-pysf",
                "--",
                "python",
                "-c",
                "print('reached')",
            ],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert any(call.startswith("venv ") for call in calls)
        assert "sync --all-extras --reinstall-package robot-sf --frozen" in calls
        assert "run python -c print('reached')" in calls
        assert (worktree / ".venv" / "bin" / "python").is_file()
        assert not (repo / ".venv").exists()
        assert "verified worktree-owned fast-pysf environment" in result.stderr
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_reuses_fresh_local_environment(tmp_path: Path) -> None:
    """A coherent local environment avoids an unnecessary reinstall."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    local_python = worktree / ".venv" / "bin" / "python"
    local_python.parent.mkdir(parents=True)
    _write_executable(
        local_python,
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == *check_fast_pysf_runtime.py ]]; then\n'
        '  printf "fast-pysf runtime preflight passed\\n"\n'
        "fi\n"
        "exit 0\n",
    )
    try:
        result = subprocess.run(
            [
                str(worktree / "scripts" / "dev" / RUN_SHARED_VENV.name),
                "--recover-stale-fast-pysf",
                "--",
                "python",
                "-V",
            ],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert calls == ["run python -V"]
        assert "sync skipped" in result.stderr
        assert not (repo / ".venv").exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_blocks_insufficient_capacity(tmp_path: Path) -> None:
    """The worktree capacity gate blocks recovery before creating or syncing an env."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    env = {**env, "ROBOT_SF_WORKTREE_MIN_FREE_BYTES": str(2**63 - 1)}
    try:
        result = subprocess.run(
            [
                str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name),
            ],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        diagnostic = result.stdout + result.stderr
        assert "capacity gate blocked recovery before uv started" in diagnostic
        assert not capture.exists()
        assert not (worktree / ".venv").exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_rejects_freshness_bypass(tmp_path: Path) -> None:
    """Recovery cannot be combined with the wrapper's freshness bypass."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    try:
        result = subprocess.run(
            [
                str(worktree / "scripts" / "dev" / RUN_SHARED_VENV.name),
                "--recover-stale-fast-pysf",
                "--no-freshness-check",
                "--",
                "python",
                "-V",
            ],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "cannot be combined with a freshness bypass" in result.stderr
        assert not capture.exists()
        assert not (worktree / ".venv").exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_serializes_same_repository(tmp_path: Path) -> None:
    """A second recovery fails boundedly while the repository lock is held."""
    repo, worktree, _, capture, sync_started, env = _linked_recovery_fixture(tmp_path)
    first_env = {**env, "UV_SYNC_STARTED": str(sync_started), "UV_SYNC_SLEEP": "1.5"}
    first = subprocess.Popen(
        [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
        cwd=worktree,
        env=first_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not sync_started.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert sync_started.exists(), "first recovery did not reach the bounded sync window"

        second = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert second.returncode == 75
        assert "another fast-pysf recovery is active" in second.stderr

        first_stdout, first_stderr = first.communicate(timeout=30)
        assert first.returncode == 0, first_stdout + first_stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert calls.count("sync --all-extras --reinstall-package robot-sf --frozen") == 1
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=30)
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_refuses_main_checkout_without_editing_it(tmp_path: Path) -> None:
    """Recovery cannot turn a dirty main checkout into an implicit package owner."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    readme = repo / "README.md"
    original = readme.read_text(encoding="utf-8")
    readme.write_text("dirty main\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [str(repo / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "refusing to mutate the main checkout" in result.stderr
        assert not capture.exists()
        assert readme.read_text(encoding="utf-8") == "dirty main\n"
        assert original != readme.read_text(encoding="utf-8")
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def _docs_proof_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a minimal docs-proof checkout with an intentionally incomplete environment."""
    repo = tmp_path / "docs-proof-repo"
    script_dir = repo / "scripts" / "dev"
    validation_dir = repo / "scripts" / "validation"
    script_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    for source, target in (
        (RUN_DOCS_PROOF, script_dir / RUN_DOCS_PROOF.name),
        (REPO_ROOT / "scripts" / "dev" / "common_setup.sh", script_dir / "common_setup.sh"),
        (
            REPO_ROOT / "scripts" / "dev" / "check_worktree_optional_deps.py",
            script_dir / "check_worktree_optional_deps.py",
        ),
    ):
        shutil.copy2(source, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    configure_git_identity(repo, name="Fixture", email="fixture@example.invalid")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)

    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _write_executable(
        venv_bin / "python",
        "#!/usr/bin/env bash\n"
        "printf 'Worktree optional dependency preflight: missing_optional (core)\\n'\n"
        "printf 'Missing optional imports: yaml\\n'\n"
        "exit 2\n",
    )
    fake_bin = tmp_path / "docs-proof-bin"
    fake_bin.mkdir()
    capture = tmp_path / "docs-proof-uv-called"
    _write_executable(
        fake_bin / "uv",
        f"#!/usr/bin/env bash\nprintf 'uv-called\\n' > {capture}\nexit 0\n",
    )
    return repo, fake_bin, capture


def test_run_worktree_shared_venv_fails_before_uv_when_scratch_is_low(tmp_path: Path) -> None:
    """The shared-venv wrapper must not launch uv after a failed capacity check."""
    repo, fake_bin, capture = _shared_venv_fixture(tmp_path)
    _fake_df(fake_bin, available_kib=64)
    ambient_tmp = tmp_path / "ambient-tmp"
    ambient_tmp.mkdir()

    env = _env_with_fake_bin(fake_bin)
    env.update(
        {
            "CI_CAPTURE": str(capture),
            "TMPDIR": str(ambient_tmp),
            "ROBOT_SF_CI_MIN_FREE_BYTES": str(1024 * 1024),
        }
    )
    result = subprocess.run(
        [
            str(RUN_SHARED_VENV),
            "--standalone",
            "--venv",
            str(repo / ".venv"),
            "--",
            "python",
            "-c",
            "pass",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    diagnostic = result.stdout + result.stderr
    assert "shared-venv scratch preflight failed" in diagnostic
    assert "The uv command was not started" in diagnostic
    assert not capture.exists()


def test_run_worktree_shared_venv_scratch_dir_reaches_uv(tmp_path: Path) -> None:
    """The shared-venv scratch option must configure temp/cache paths before uv runs."""
    repo, fake_bin, capture = _shared_venv_fixture(tmp_path)
    _fake_df(fake_bin, available_kib=10_000_000)
    scratch = tmp_path / "disk-backed-scratch"

    env = _env_with_fake_bin(fake_bin)
    env.update(
        {
            "CI_CAPTURE": str(capture),
            "ROBOT_SF_CI_MIN_FREE_BYTES": "0",
        }
    )
    result = subprocess.run(
        [
            str(RUN_SHARED_VENV),
            "--standalone",
            "--venv",
            str(repo / ".venv"),
            "--scratch-dir",
            str(scratch),
            "--",
            "python",
            "-c",
            "pass",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    expected_root = scratch.resolve()
    captured = capture.read_text(encoding="utf-8")
    assert f"TMPDIR={expected_root / 'tmp'}" in captured
    assert f"UV_CACHE_DIR={expected_root / 'uv-cache'}" in captured
    assert f"XDG_CACHE_HOME={expected_root / 'xdg-cache'}" in captured
    assert f"MPLCONFIGDIR={expected_root / 'mplconfig'}" in captured
    assert "UV_NO_SYNC=1" in captured


def test_run_worktree_shared_venv_fails_before_uv_on_incomplete_dependency_profile(
    tmp_path: Path,
) -> None:
    """An incomplete current-worktree environment fails with the bootstrap remedy."""
    repo, fake_bin, capture = _shared_venv_fixture(tmp_path)
    _fake_df(fake_bin, available_kib=10_000_000)
    _write_executable(
        repo / ".venv" / "bin" / "python",
        "#!/usr/bin/env bash\n"
        "printf 'Worktree optional dependency preflight: missing_optional (core)\\n'\n"
        "printf 'Missing optional imports: yaml\\n'\n"
        "exit 2\n",
    )

    env = _env_with_fake_bin(fake_bin)
    env.update({"CI_CAPTURE": str(capture), "ROBOT_SF_CI_MIN_FREE_BYTES": "0"})
    result = subprocess.run(
        [
            str(RUN_SHARED_VENV),
            "--venv",
            str(repo / ".venv"),
            "--",
            "python",
            "-V",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    diagnostic = result.stdout + result.stderr
    assert "shared-venv dependency profile 'core' is incomplete" in diagnostic
    assert "bootstrap_worktree.sh" in diagnostic
    assert not capture.exists()


def test_docs_proof_fails_before_uv_on_incomplete_dependency_profile(tmp_path: Path) -> None:
    """Docs proof must reject a partial current-worktree environment before invoking uv."""
    repo, fake_bin, capture = _docs_proof_fixture(tmp_path)
    env = _env_with_fake_bin(fake_bin)
    env["BASE_REF"] = "HEAD"

    result = subprocess.run(
        [str(repo / "scripts" / "dev" / RUN_DOCS_PROOF.name)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    diagnostic = result.stdout + result.stderr
    assert "worktree dependency profile 'core' is incomplete" in diagnostic
    assert "bootstrap_worktree.sh" in diagnostic
    assert not capture.exists()
