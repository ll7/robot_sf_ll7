"""Deterministic capacity and scratch contracts for local CI wrappers."""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

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
        REPO_ROOT / "scripts" / "dev" / "worktree_creation_lock.py",
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
    rvo2_source = repo / "third_party" / "python-rvo2"
    rvo2_source.mkdir(parents=True)
    (rvo2_source / "UPSTREAM.md").write_text("fixture\n", encoding="utf-8")
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
        'if [[ -n "${UV_ENV_CAPTURE:-}" ]]; then printf "UV_PROJECT=%s\\n" "${UV_PROJECT-<unset>}" >> "$UV_ENV_CAPTURE"; fi\n'
        'if [[ "${1:-}" == "cache" && "${2:-}" == "dir" ]]; then\n'
        '  if [[ -n "${UV_CACHE_CAPTURE:-}" ]]; then printf "%s\\n" "$*" >> "$UV_CACHE_CAPTURE"; fi\n'
        '  if [[ -f "uv.toml" ]]; then sed -n \'s/^cache-dir[[:space:]]*=[[:space:]]*"\\(.*\\)"$/\\1/p\' uv.toml; exit 0; fi\n'
        '  printf "%s\\n" "${UV_CACHE_DIR_OUTPUT:-${UV_CACHE_DIR:-$PWD/.uv-cache}}"\n'
        "  exit 0\n"
        "fi\n"
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
    assert "ROBOT_SF_VENV_SEED_CACHE" in result.stdout
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
        assert "sync --reinstall-package robot-sf --frozen" in calls
        assert "run python -c print('reached')" in calls
        assert (worktree / ".venv" / "bin" / "python").is_file()
        assert not (repo / ".venv").exists()
        assert "verified worktree-owned fast-pysf environment" in result.stderr
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_auto_recovers_stale_default_environment_in_worktree(
    tmp_path: Path,
) -> None:
    """Default linked-worktree runs recover stale fast-pysf without touching main."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    main_python = repo / ".venv" / "bin" / "python"
    main_python.parent.mkdir(parents=True)
    _write_executable(
        main_python,
        "#!/usr/bin/env bash\n"
        'case "${1:-}" in\n'
        "  *check_worktree_optional_deps.py) exit 0 ;;\n"
        '  *check_fast_pysf_runtime.py) printf "installed pysocialforce package is stale relative to this checkout\\n" >&2; exit 1 ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n",
    )
    main_python_before = main_python.read_bytes()
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RUN_SHARED_VENV.name), "--", "python", "-V"],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert "sync --reinstall-package robot-sf --frozen" in calls
        assert "run python -V" in calls
        assert (worktree / ".venv" / "bin" / "python").is_file()
        assert main_python.read_bytes() == main_python_before
        assert "Recovering stale fast-pysf in the linked worktree" in result.stderr
        assert "Automatic fast-pysf recovery selected worktree environment" in result.stderr
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


@pytest.mark.parametrize(
    "checker_output",
    [
        "could not import pysocialforce.forces (No module named 'pysocialforce')",
        "pysocialforce.forces.social_force_gil_releasing_context is missing or not callable",
    ],
)
def test_shared_venv_does_not_auto_recover_other_checker_failures(
    tmp_path: Path,
    checker_output: str,
) -> None:
    """Only the exact stale-package diagnostic can trigger automatic recovery."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    main_python = repo / ".venv" / "bin" / "python"
    main_python.parent.mkdir(parents=True)
    _write_executable(
        main_python,
        "#!/usr/bin/env bash\n"
        'case "${1:-}" in\n'
        "  *check_worktree_optional_deps.py) exit 0 ;;\n"
        f'  *check_fast_pysf_runtime.py) printf "{checker_output}\\n" >&2; exit 1 ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n",
    )
    main_python_before = main_python.read_bytes()
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RUN_SHARED_VENV.name), "--", "python", "-V"],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert checker_output in result.stderr
        assert "refusing automatic recovery" in result.stderr
        assert "Recovering stale fast-pysf" not in result.stderr
        assert not (worktree / ".venv").exists()
        assert not capture.exists()
        assert main_python.read_bytes() == main_python_before
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_explicit_environment_does_not_auto_recover(
    tmp_path: Path,
) -> None:
    """An explicit environment remains fail-closed even for the stale diagnostic."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    main_python = repo / ".venv" / "bin" / "python"
    main_python.parent.mkdir(parents=True)
    _write_executable(
        main_python,
        "#!/usr/bin/env bash\n"
        'case "${1:-}" in\n'
        "  *check_worktree_optional_deps.py) exit 0 ;;\n"
        '  *check_fast_pysf_runtime.py) printf "installed pysocialforce package is stale relative to this checkout\\n" >&2; exit 1 ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n",
    )
    try:
        result = subprocess.run(
            [
                str(worktree / "scripts" / "dev" / RUN_SHARED_VENV.name),
                "--venv",
                str(repo / ".venv"),
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
        assert "Recovering stale fast-pysf" not in result.stderr
        assert not (worktree / ".venv").exists()
        assert not capture.exists()
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


@pytest.mark.parametrize(
    ("profile_args", "expected_profile", "expected_sync"),
    [
        ([], "core", "sync --reinstall-package robot-sf --frozen"),
        (
            ["--profile", "all-extras"],
            "all-extras",
            "sync --all-extras --reinstall-package robot-sf --frozen",
        ),
        (
            ["--profile", "training"],
            "training",
            "sync --extra training --reinstall-package robot-sf --frozen",
        ),
        (
            ["--profile", "orca"],
            "orca",
            "sync --reinstall-package robot-sf --frozen",
        ),
    ],
)
def test_recover_fast_pysf_postcondition_fails_on_incomplete_profile(
    tmp_path: Path,
    profile_args: list[str],
    expected_profile: str,
    expected_sync: str,
) -> None:
    """Issue #8811: recovery must certify the requested dependency profile before success.

    The fast-pysf checker passes while the profile probe reports a missing import,
    so a coherent-but-incomplete environment must still be refreshed and then
    reported as an incomplete recovery result instead of a certified one.
    """
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    local_python = worktree / ".venv" / "bin" / "python"
    local_python.parent.mkdir(parents=True)
    _write_executable(
        local_python,
        "#!/usr/bin/env bash\n"
        'case "${1:-}" in\n'
        '  *check_fast_pysf_runtime.py) printf "fast-pysf runtime preflight passed\\n"; exit 0 ;;\n'
        "  *check_worktree_optional_deps.py)\n"
        '    printf "Worktree optional dependency preflight: missing_optional\\n"\n'
        '    printf "Missing optional imports: yaml\\n"\n'
        "    exit 2 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
    )
    try:
        result = subprocess.run(
            [
                str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name),
                *profile_args,
            ],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2, result.stderr
        assert f"post-sync dependency profile '{expected_profile}' is incomplete" in result.stderr
        assert "Missing optional imports: yaml" in result.stderr
        assert "bootstrap_worktree.sh" in result.stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert calls.count(expected_sync) == 1
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_rejects_nested_main_venv_symlink(tmp_path: Path) -> None:
    """A nested bin symlink must not redirect recovery into the owning checkout."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    main_python = repo / ".venv" / "bin" / "python"
    main_python.parent.mkdir(parents=True)
    _write_executable(main_python, "#!/usr/bin/env bash\nexit 0\n")
    (worktree / ".venv").mkdir()
    (worktree / ".venv" / "bin").symlink_to(main_python.parent, target_is_directory=True)
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "environment component outside the worktree" in result.stderr
        assert not capture.exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_rejects_nested_external_site_packages_symlink(
    tmp_path: Path,
) -> None:
    """Package directories must not redirect recovery writes outside the worktree."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    outside = tmp_path / "outside-site-packages"
    outside.mkdir()
    local_site_packages = worktree / ".venv" / "lib" / "python3.13" / "site-packages"
    local_site_packages.parent.mkdir(parents=True)
    local_site_packages.symlink_to(outside, target_is_directory=True)
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "environment symlink outside the worktree" in result.stderr
        assert not capture.exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_rejects_unreadable_nested_environment(
    tmp_path: Path,
) -> None:
    """An unreadable environment subtree must not bypass recursive ownership checks."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    outside = tmp_path / "outside-site-packages"
    outside.mkdir()
    unreadable = worktree / ".venv" / "lib" / "python3.13"
    unreadable.mkdir(parents=True)
    (unreadable / "site-packages").symlink_to(outside, target_is_directory=True)
    unreadable.chmod(0o111)
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "could not inspect worktree environment symlinks" in result.stderr
        assert not capture.exists()
    finally:
        unreadable.chmod(0o755)
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_allows_external_standard_python_symlink(tmp_path: Path) -> None:
    """A standard interpreter link may target an existing host Python."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    local_python3 = worktree / ".venv" / "bin" / "python3"
    local_python3.parent.mkdir(parents=True)
    local_python3.symlink_to(Path("/usr/bin/python3"))
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "sync --reinstall-package robot-sf --frozen" in capture.read_text(encoding="utf-8")
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_rejects_host_python_link_into_owning_checkout(
    tmp_path: Path,
) -> None:
    """A standard interpreter alias must not redirect into the owning checkout."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    owner_script = repo / "scripts" / "dev" / "owner-python-target"
    _write_executable(owner_script, "#!/usr/bin/env bash\nexit 0\n")
    worktree_python = worktree / ".venv" / "bin" / "python3"
    worktree_python.parent.mkdir(parents=True)
    worktree_python.symlink_to(owner_script)
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "host-interpreter link into the owning checkout" in result.stderr
        assert not capture.exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_rejects_broken_host_python_link(tmp_path: Path) -> None:
    """A broken standard interpreter alias must fail before recovery starts."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    worktree_python = worktree / ".venv" / "bin" / "python3"
    worktree_python.parent.mkdir(parents=True)
    worktree_python.symlink_to(tmp_path / "missing-python")
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "broken host-interpreter link" in result.stderr
        assert not capture.exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_refuses_symlinked_repository_lock(tmp_path: Path) -> None:
    """The lock guard must not truncate or follow an arbitrary symlink target."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    sentinel = tmp_path / "lock-sentinel.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")
    lock_path = repo / ".git" / "robot-sf-fast-pysf-recovery.lock"
    lock_path.symlink_to(sentinel)
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "refusing a symlinked repository recovery lock" in result.stderr
        assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
        assert not capture.exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_rejects_dirty_rvo2_dependency_input(tmp_path: Path) -> None:
    """Frozen recovery must stop when the local path dependency is dirty."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    rvo2_readme = worktree / "third_party" / "python-rvo2" / "UPSTREAM.md"
    rvo2_readme.write_text("dirty\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "dirty dependency inputs" in result.stderr
        assert not capture.exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_clears_ambient_uv_project(tmp_path: Path) -> None:
    """Recovery and the wrapped command must resolve the current worktree project."""
    repo, worktree, _, _, _, env = _linked_recovery_fixture(tmp_path)
    env_capture = tmp_path / "uv-env.txt"
    env = {**env, "UV_PROJECT": str(repo), "UV_ENV_CAPTURE": str(env_capture)}
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
        assert env_capture.read_text(encoding="utf-8").splitlines()
        assert set(env_capture.read_text(encoding="utf-8").splitlines()) == {"UV_PROJECT=<unset>"}
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_blocks_insufficient_capacity(tmp_path: Path) -> None:
    """The worktree capacity gate blocks recovery before creating or syncing an env."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    env = {**env, "ROBOT_SF_RECOVERY_MIN_FREE_BYTES": str(2**63 - 1)}
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
        assert "capacity gate blocked recovery before materialization" in diagnostic
        assert not capture.exists()
        assert not (worktree / ".venv").exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_uses_profile_capacity_default(tmp_path: Path) -> None:
    """Large dependency profiles use a conservative gate when no override is set."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    capacity_capture = tmp_path / "capacity-args.txt"
    capacity_checker = worktree / "scripts" / "dev" / "check_worktree_capacity.py"
    capacity_checker.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPACITY_CAPTURE']).write_text(' '.join(__import__('sys').argv[1:]))\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    env = {
        **env,
        "CAPACITY_CAPTURE": str(capacity_capture),
        "ROBOT_SF_WORKTREE_MIN_FREE_BYTES": "",
    }
    try:
        result = subprocess.run(
            [
                str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name),
                "--profile",
                "all-extras",
            ],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "capacity gate blocked recovery before materialization" in result.stderr
        assert "--minimum-free-bytes 8589934592" in capacity_capture.read_text(encoding="utf-8")
        assert not capture.exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_blocks_distinct_low_capacity_uv_cache(tmp_path: Path) -> None:
    """A low-capacity cache filesystem blocks materialization before downloads."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    cache_root = tmp_path / "distinct-uv-cache"
    capacity_log = tmp_path / "capacity-paths.txt"
    capacity_checker = worktree / "scripts" / "dev" / "check_worktree_capacity.py"
    capacity_checker.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "path = Path(args[args.index('--path') + 1])\n"
        "with Path(os.environ['CAPACITY_LOG']).open('a', encoding='utf-8') as stream:\n"
        "    stream.write(f'{path}\\n')\n"
        "if str(path) == os.environ['LOW_CACHE_PATH']:\n"
        "    print('simulated low cache capacity')\n"
        "    raise SystemExit(2)\n",
        encoding="utf-8",
    )
    env = {
        **env,
        "CAPACITY_LOG": str(capacity_log),
        "LOW_CACHE_PATH": str(cache_root),
        "ROBOT_SF_RECOVERY_MIN_FREE_BYTES": "0",
        "UV_CACHE_DIR": str(cache_root),
    }
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "simulated low cache capacity" in result.stderr
        assert (
            f"capacity gate blocked recovery before materialization at {cache_root}"
            in result.stderr
        )
        assert capacity_log.read_text(encoding="utf-8").splitlines() == [
            str(worktree / ".venv"),
            str(cache_root),
        ]
        assert not capture.exists()
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_uses_uv_configured_cache_path(tmp_path: Path) -> None:
    """Capacity follows uv.toml's effective cache path instead of guessing XDG/HOME."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    configured_cache = worktree / "configured-cache"
    uv_cache_capture = tmp_path / "uv-cache-calls.txt"
    capacity_log = tmp_path / "capacity-paths.txt"
    (worktree / "uv.toml").write_text('cache-dir = "configured-cache"\n', encoding="utf-8")
    capacity_checker = worktree / "scripts" / "dev" / "check_worktree_capacity.py"
    capacity_checker.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "path = Path(args[args.index('--path') + 1])\n"
        "with Path(os.environ['CAPACITY_LOG']).open('a', encoding='utf-8') as stream:\n"
        "    stream.write(f'{path}\\n')\n",
        encoding="utf-8",
    )
    env = {
        **env,
        "CAPACITY_LOG": str(capacity_log),
        "UV_CACHE_CAPTURE": str(uv_cache_capture),
        "ROBOT_SF_RECOVERY_MIN_FREE_BYTES": "0",
    }
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert uv_cache_capture.read_text(encoding="utf-8").splitlines() == [
            f"cache dir --directory {worktree}"
        ]
        assert capacity_log.read_text(encoding="utf-8").splitlines() == [
            str(worktree / ".venv"),
            str(configured_cache),
        ]
        assert configured_cache.as_posix() in result.stderr
        assert "sync --reinstall-package robot-sf --frozen" in capture.read_text(encoding="utf-8")
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_recreates_environment_after_prior_state_marker(
    tmp_path: Path,
) -> None:
    """A retry can recreate an env after a marker-only interrupted attempt."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    state_path = worktree / ".venv" / ".robot-sf-recovery-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("prior partial state\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert not state_path.exists()
        assert "sync --reinstall-package robot-sf --frozen" in capture.read_text(encoding="utf-8")
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_failure_records_partial_environment_state(tmp_path: Path) -> None:
    """A failed profile sync leaves an inspectable, non-certified worktree state."""
    repo, worktree, fake_bin, capture, _, env = _linked_recovery_fixture(tmp_path)
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        fake_uv.read_text(encoding="utf-8").replace(
            "  sync)\n",
            '  sync)\n    if [[ "${UV_SYNC_FAIL:-0}" == "1" ]]; then exit 23; fi\n',
            1,
        ),
        encoding="utf-8",
    )
    env = {**env, "UV_SYNC_FAIL": "1"}
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        state_path = worktree / ".venv" / ".robot-sf-recovery-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["schema"] == "robot_sf.recovery_state.v1"
        assert state["status"] == "failed"
        assert state["dependency_profile"] == "core"
        assert "partial environment state preserved" in result.stderr
        assert "verified worktree-owned fast-pysf environment" not in result.stderr
        assert capture.read_text(encoding="utf-8").splitlines()[-1] == (
            "sync --reinstall-package robot-sf --frozen"
        )
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_signal_preserves_partial_environment_state(
    tmp_path: Path,
) -> None:
    """Interrupting a blocked sync stops its child and records actionable state."""
    repo, worktree, fake_bin, _capture, sync_started, env = _linked_recovery_fixture(tmp_path)
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        fake_uv.read_text(encoding="utf-8").replace(
            "  sync)\n",
            '  sync)\n    if [[ "${UV_SYNC_IGNORE_TERM:-0}" == "1" ]]; then trap "" TERM; fi\n',
            1,
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
        cwd=worktree,
        env={
            **env,
            "UV_SYNC_IGNORE_TERM": "1",
            "UV_SYNC_STARTED": str(sync_started),
            "UV_SYNC_SLEEP": "30",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not sync_started.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert sync_started.exists(), "recovery did not reach the bounded sync window"

        process.send_signal(signal.SIGTERM)
        _stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, stderr
        state_path = worktree / ".venv" / ".robot-sf-recovery-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["status"] == "interrupted"
        assert "recovery child ignored SIGTERM; sending SIGKILL" in stderr
        assert "partial environment state preserved" in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_rejects_unknown_profile_before_uv(tmp_path: Path) -> None:
    """An unsupported profile cannot fall through to an unscoped all-extras sync."""
    repo, worktree, _, capture, _, env = _linked_recovery_fixture(tmp_path)
    try:
        result = subprocess.run(
            [
                str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name),
                "--profile",
                "not-a-profile",
            ],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "unsupported dependency profile: not-a-profile" in result.stderr
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
        assert calls.count("sync --reinstall-package robot-sf --frozen") == 1
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=30)
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_waits_for_lock_and_succeeds_concurrently(tmp_path: Path) -> None:
    """A concurrent recovery waits boundedly for the lock and succeeds once free."""
    repo, worktree1, _, capture, sync_started, env = _linked_recovery_fixture(tmp_path)
    worktree2 = tmp_path / "linked-worktree-2"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree2)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    first_env = {**env, "UV_SYNC_STARTED": str(sync_started), "UV_SYNC_SLEEP": "1.0"}
    first = subprocess.Popen(
        [str(worktree1 / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
        cwd=worktree1,
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
            [
                str(worktree2 / "scripts" / "dev" / RECOVER_FAST_PYSF.name),
                "--wait-timeout",
                "10",
            ],
            cwd=worktree2,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert second.returncode == 0, second.stderr
        assert "waiting up to 10s for lock" in second.stderr
        assert "Lock owner metadata:" in second.stderr
        assert "PID:" in second.stderr
        assert "Status: alive" in second.stderr

        first_stdout, first_stderr = first.communicate(timeout=30)
        assert first.returncode == 0, first_stdout + first_stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert calls.count("sync --reinstall-package robot-sf --frozen") == 2
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=30)
        _remove_linked_recovery_fixture(repo, worktree2)
        _remove_linked_recovery_fixture(repo, worktree1)


def test_shared_venv_recovery_wait_times_out_and_fails_closed(tmp_path: Path) -> None:
    """Lock wait timeout fails closed with exit code 75 and owner diagnostics."""
    repo, worktree, _, capture, sync_started, env = _linked_recovery_fixture(tmp_path)
    first_env = {**env, "UV_SYNC_STARTED": str(sync_started), "UV_SYNC_SLEEP": "5.0"}
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
            [
                str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name),
                "--wait-timeout",
                "1",
            ],
            cwd=worktree,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert second.returncode == 75, second.stderr
        assert "timed out waiting for repository fast-pysf recovery lock after 1s" in second.stderr
        assert "another fast-pysf recovery is active" in second.stderr
        assert "Lock owner metadata:" in second.stderr
        assert "PID:" in second.stderr
        assert "Started:" in second.stderr
        assert "Worktree:" in second.stderr
        assert "Status: alive" in second.stderr

        first_stdout, first_stderr = first.communicate(timeout=30)
        assert first.returncode == 0, first_stdout + first_stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert calls.count("sync --reinstall-package robot-sf --frozen") == 1
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=30)
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_portable_lock_wait_timeout(tmp_path: Path) -> None:
    """Portable flock-less lock contender times out and exits 75 with diagnostics."""
    repo, worktree, fake_bin, _capture, sync_started, env = _linked_recovery_fixture(tmp_path)
    first_env = {**env, "UV_SYNC_STARTED": str(sync_started), "UV_SYNC_SLEEP": "5.0"}
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

        stub_bin = tmp_path / "stub-bin"
        contender_env = {**env, "PATH": _flockless_path(stub_bin, fake_bin)}
        second = subprocess.run(
            [
                str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name),
                "--wait-timeout",
                "1",
            ],
            cwd=worktree,
            env=contender_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert second.returncode == 75, second.stderr
        assert "timed out waiting for repository fast-pysf recovery lock after 1s" in second.stderr
        assert "another fast-pysf recovery is active" in second.stderr
        assert "Lock owner metadata:" in second.stderr

        first_stdout, first_stderr = first.communicate(timeout=30)
        assert first.returncode == 0, first_stdout + first_stderr
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=30)
        _remove_linked_recovery_fixture(repo, worktree)


def test_run_worktree_shared_venv_concurrent_recovery_waits_safely(tmp_path: Path) -> None:
    """Concurrent run_worktree_shared_venv invocations wait for recovery lock safely."""
    repo, worktree1, _, capture, sync_started, env = _linked_recovery_fixture(tmp_path)
    worktree2 = tmp_path / "linked-worktree-2"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree2)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    first_env = {**env, "UV_SYNC_STARTED": str(sync_started), "UV_SYNC_SLEEP": "1.0"}
    first = subprocess.Popen(
        [
            str(worktree1 / "scripts" / "dev" / RUN_SHARED_VENV.name),
            "--recover-stale-fast-pysf",
            "--recovery-timeout",
            "10",
            "--",
            "python",
            "-V",
        ],
        cwd=worktree1,
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
            [
                str(worktree2 / "scripts" / "dev" / RUN_SHARED_VENV.name),
                "--recover-stale-fast-pysf",
                "--recovery-timeout",
                "10",
                "--",
                "python",
                "-V",
            ],
            cwd=worktree2,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert second.returncode == 0, second.stderr

        first_stdout, first_stderr = first.communicate(timeout=30)
        assert first.returncode == 0, first_stdout + first_stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert calls.count("sync --reinstall-package robot-sf --frozen") == 2
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=30)
        _remove_linked_recovery_fixture(repo, worktree2)
        _remove_linked_recovery_fixture(repo, worktree1)


def _flockless_path(stub_bin: Path, fake_bin: Path) -> str:
    """Build a PATH without the flock CLI (e.g. macOS) keeping all other tools."""
    stub_bin.mkdir(exist_ok=True)
    for directory in ("/usr/bin", "/bin", "/usr/local/bin"):
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            if entry == "flock" or (stub_bin / entry).exists():
                continue
            try:
                (stub_bin / entry).symlink_to(os.path.join(directory, entry))
            except OSError:
                continue
    assert shutil.which("flock", path=str(stub_bin)) is None, "stub PATH must hide flock"
    return f"{fake_bin}{os.pathsep}{stub_bin}"


def test_shared_venv_recovery_portable_lock_preserves_contention_contract(
    tmp_path: Path,
) -> None:
    """A flock-less contender still exits 75 against a flock-CLI lock holder."""
    repo, worktree, fake_bin, _capture, sync_started, env = _linked_recovery_fixture(tmp_path)
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

        stub_bin = tmp_path / "stub-bin"
        contender_env = {**env, "PATH": _flockless_path(stub_bin, fake_bin)}
        second = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=contender_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert second.returncode == 75, second.stderr
        assert "another fast-pysf recovery is active" in second.stderr
        assert "flock is required" not in second.stderr

        first_stdout, first_stderr = first.communicate(timeout=30)
        assert first.returncode == 0, first_stdout + first_stderr
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=30)
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_portable_lock_succeeds_without_flock_cli(
    tmp_path: Path,
) -> None:
    """Uncontended recovery proceeds on the portable path without the flock CLI."""
    repo, worktree, fake_bin, capture, _, env = _linked_recovery_fixture(tmp_path)
    stub_bin = tmp_path / "stub-bin"
    fallback_env = {**env, "PATH": _flockless_path(stub_bin, fake_bin)}
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=fallback_env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "portable lock" in result.stderr
        assert "flock is required" not in result.stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert calls.count("sync --reinstall-package robot-sf --frozen") == 1
    finally:
        _remove_linked_recovery_fixture(repo, worktree)


def test_shared_venv_recovery_rejects_non_regular_lock_without_flock_cli(
    tmp_path: Path,
) -> None:
    """The portable path must refuse a directory lock before invoking Python locking."""
    repo, worktree, fake_bin, capture, _, env = _linked_recovery_fixture(tmp_path)
    lock_path = repo / ".git" / "robot-sf-fast-pysf-recovery.lock"
    lock_path.mkdir()
    stub_bin = tmp_path / "stub-bin"
    fallback_env = {**env, "PATH": _flockless_path(stub_bin, fake_bin)}
    try:
        result = subprocess.run(
            [str(worktree / "scripts" / "dev" / RECOVER_FAST_PYSF.name)],
            cwd=worktree,
            env=fallback_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 2
        assert "repository recovery lock is not a regular file" in result.stderr
        assert not capture.exists()
    finally:
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


def _seed_recovery_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    """Build a lockfile-complete fixture repo with two linked worktrees.

    Both worktrees check out the same commit, so they share one venv-seed
    identity. Returns (repo, worktree_a, worktree_b, seed_cache, env).
    """
    repo = tmp_path / "seed-main"
    repo.mkdir()
    script_dir = repo / "scripts" / "dev"
    script_dir.mkdir(parents=True)
    for source in (
        RUN_SHARED_VENV,
        RECOVER_FAST_PYSF,
        REPO_ROOT / "scripts" / "dev" / "worktree_creation_lock.py",
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
    (repo / "fast-pysf" / "pyproject.toml").write_text(
        '[project]\nname = "seed-fixture-pysf"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    (repo / "fast-pysf" / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    rvo2_source = repo / "third_party" / "python-rvo2"
    rvo2_source.mkdir(parents=True)
    (rvo2_source / "UPSTREAM.md").write_text("fixture\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "seed-fixture"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    configure_git_identity(repo, name="Seed Fixture", email="seed@example.invalid")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "seed fixture"], cwd=repo, check=True, capture_output=True
    )

    worktree_a = tmp_path / "seed-worktree-a"
    worktree_b = tmp_path / "seed-worktree-b"
    for worktree in (worktree_a, worktree_b):
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree)],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )

    fake_bin = tmp_path / "seed-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "uv",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == "cache" && "${2:-}" == "dir" ]]; then\n'
        '  printf "%s\\n" "${UV_CACHE_DIR_OUTPUT:-${UV_CACHE_DIR:-$PWD/.uv-cache}}"\n'
        "  exit 0\n"
        "fi\n"
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
        "    ;;\n"
        "  run)\n"
        "    ;;\n"
        '  *) printf "unexpected uv invocation: %s\\n" "$*" >&2; exit 9 ;;\n'
        "esac\n",
    )
    seed_cache = tmp_path / "seed-cache"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "ROBOT_SF_CI_MIN_FREE_BYTES": "0",
        "ROBOT_SF_WORKTREE_MIN_FREE_BYTES": "0",
        "ROBOT_SF_VENV_SEED_CACHE": str(seed_cache),
    }
    env.pop("PYTHONPATH", None)
    return repo, worktree_a, worktree_b, seed_cache, env


def _run_seed_recovery(
    worktree: Path, env: dict[str, str], capture: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    """Run the fixture recovery helper with a per-run uv capture file."""
    run_env = {**env, "UV_CAPTURE": str(capture)}
    return subprocess.run(
        [str(worktree / "scripts" / "dev" / "recover_fast_pysf_worktree.sh"), *args],
        cwd=worktree,
        env=run_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _teardown_seed_fixture(repo: Path, *worktrees: Path) -> None:
    for worktree in worktrees:
        _remove_linked_recovery_fixture(repo, worktree)


def test_recovery_publishes_checksum_keyed_seed(tmp_path: Path) -> None:
    """A verified recovery must publish a receipted seed for its identity."""
    repo, worktree_a, worktree_b, seed_cache, env = _seed_recovery_fixture(tmp_path)
    try:
        result = _run_seed_recovery(worktree_a, env, tmp_path / "uv-a.txt", "--profile", "core")

        assert result.returncode == 0, result.stderr
        seeds = [
            entry
            for entry in seed_cache.iterdir()
            if entry.is_dir() and (entry / "seed-receipt.json").is_file()
        ]
        assert len(seeds) == 1
        receipt = json.loads((seeds[0] / "seed-receipt.json").read_text(encoding="utf-8"))
        assert len(receipt["identity"]) == 64
        int(receipt["identity"], 16)
        assert receipt["identity"] == seeds[0].name
        assert receipt["dependency_profile"] == "core"
        assert receipt["source_venv"] == str(worktree_a / ".venv")
        calls = (tmp_path / "uv-a.txt").read_text(encoding="utf-8").splitlines()
        assert "sync --reinstall-package robot-sf --frozen" in calls
        assert "published checksum-keyed seed environment" in result.stderr
    finally:
        _teardown_seed_fixture(repo, worktree_a, worktree_b)


def test_recovery_restores_matching_seed_without_materialization(
    tmp_path: Path,
) -> None:
    """An identical worktree must clone the seed, rebind it, and verify it."""
    repo, worktree_a, worktree_b, seed_cache, env = _seed_recovery_fixture(tmp_path)
    try:
        first = _run_seed_recovery(worktree_a, env, tmp_path / "uv-a1.txt", "--profile", "core")
        assert first.returncode == 0, first.stderr
        # Plant a seed-path entry point, then republish so the seed carries it.
        marker = worktree_a / ".venv" / "bin" / "seed-tool"
        marker.write_text(
            f"#!{worktree_a / '.venv' / 'bin' / 'python'}\nprint('seed-tool')\n",
            encoding="utf-8",
        )
        marker.chmod(marker.stat().st_mode | stat.S_IXUSR)
        shutil.rmtree(seed_cache)
        republish = _run_seed_recovery(worktree_a, env, tmp_path / "uv-a2.txt", "--profile", "core")
        assert republish.returncode == 0, republish.stderr

        result = _run_seed_recovery(worktree_b, env, tmp_path / "uv-b.txt", "--profile", "core")

        assert result.returncode == 0, result.stderr
        calls = (tmp_path / "uv-b.txt").read_text(encoding="utf-8").splitlines()
        assert not any(call.startswith("venv ") for call in calls)
        assert "sync --reinstall-package robot-sf --frozen" in calls
        assert "restored checksum-keyed seed environment" in result.stderr
        rebound = (worktree_b / ".venv" / "bin" / "seed-tool").read_text(encoding="utf-8")
        assert str(worktree_a) not in rebound
        assert str(worktree_b / ".venv" / "bin" / "python") in rebound
        # The seed itself must be unmutated by the restore rebind.
        (seed_dir,) = [
            entry for entry in seed_cache.iterdir() if entry.is_dir() and entry.name[0] != "."
        ]
        seeded = (seed_dir / "bin" / "seed-tool").read_text(encoding="utf-8")
        assert str(worktree_a / ".venv" / "bin" / "python") in seeded
        assert "verified worktree-owned fast-pysf environment" in result.stderr
    finally:
        _teardown_seed_fixture(repo, worktree_a, worktree_b)


def test_recovery_falls_back_to_full_sync_on_seed_mismatch(tmp_path: Path) -> None:
    """A corrupt seed receipt must not poison recovery; full sync runs instead."""
    repo, worktree_a, worktree_b, seed_cache, env = _seed_recovery_fixture(tmp_path)
    try:
        first = _run_seed_recovery(worktree_a, env, tmp_path / "uv-a.txt", "--profile", "core")
        assert first.returncode == 0, first.stderr
        (seed_dir,) = [
            entry for entry in seed_cache.iterdir() if entry.is_dir() and entry.name[0] != "."
        ]
        receipt_path = seed_dir / "seed-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["identity"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        result = _run_seed_recovery(worktree_b, env, tmp_path / "uv-b.txt", "--profile", "core")

        assert result.returncode == 0, result.stderr
        calls = (tmp_path / "uv-b.txt").read_text(encoding="utf-8").splitlines()
        assert any(call.startswith("venv ") for call in calls)
        assert "sync --reinstall-package robot-sf --frozen" in calls
    finally:
        _teardown_seed_fixture(repo, worktree_a, worktree_b)


def test_recovery_seed_cache_off_disables_publish_and_restore(tmp_path: Path) -> None:
    """ROBOT_SF_VENV_SEED_CACHE=off must leave the seed store untouched."""
    repo, worktree_a, worktree_b, seed_cache, env = _seed_recovery_fixture(tmp_path)
    try:
        off_env = {**env, "ROBOT_SF_VENV_SEED_CACHE": "off"}
        result = _run_seed_recovery(worktree_a, off_env, tmp_path / "uv-a.txt", "--profile", "core")

        assert result.returncode == 0, result.stderr
        assert not seed_cache.exists()

        seeded_env = {**env}
        seeded = _run_seed_recovery(
            worktree_a, seeded_env, tmp_path / "uv-a2.txt", "--profile", "core"
        )
        assert seeded.returncode == 0, seeded.stderr
        assert seed_cache.is_dir()
        restore_off = _run_seed_recovery(
            worktree_b, off_env, tmp_path / "uv-b.txt", "--profile", "core"
        )

        assert restore_off.returncode == 0, restore_off.stderr
        calls = (tmp_path / "uv-b.txt").read_text(encoding="utf-8").splitlines()
        assert any(call.startswith("venv ") for call in calls)
    finally:
        _teardown_seed_fixture(repo, worktree_a, worktree_b)
