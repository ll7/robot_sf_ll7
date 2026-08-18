"""Deterministic capacity and scratch contracts for local CI wrappers."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CI_LOCAL = REPO_ROOT / "scripts" / "dev" / "run_ci_local.sh"
RUN_SHARED_VENV = REPO_ROOT / "scripts" / "dev" / "run_worktree_shared_venv.sh"


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
