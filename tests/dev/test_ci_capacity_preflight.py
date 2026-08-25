"""Deterministic capacity and scratch contracts for local CI wrappers."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

from tests.support.environment_guards import configure_git_identity

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CI_LOCAL = REPO_ROOT / "scripts" / "dev" / "run_ci_local.sh"
RUN_SHARED_VENV = REPO_ROOT / "scripts" / "dev" / "run_worktree_shared_venv.sh"
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
