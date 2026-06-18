"""Tests for the CI headless package installer helper."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path


def _script_path() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "scripts" / "dev" / "ci_install_headless_packages.sh"
    )


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _shell_quote(path: Path) -> str:
    return shlex.quote(str(path))


def test_ci_install_headless_packages_shell_syntax() -> None:
    """Validate that the package helper passes bash syntax checks."""
    script = _script_path()
    assert script.exists(), "ci_install_headless_packages.sh helper is missing"
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_ci_install_headless_packages_help_flag() -> None:
    """--help prints usage without attempting package inspection."""
    result = subprocess.run(
        ["bash", str(_script_path()), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_ci_install_headless_packages_requires_package() -> None:
    """The helper requires at least one package name."""
    result = subprocess.run(
        ["bash", str(_script_path())],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_ci_install_headless_packages_skips_apt_when_all_packages_present(tmp_path: Path) -> None:
    """Already-installed packages should not trigger apt update or install."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "commands.log"

    _write_executable(
        fake_bin / "dpkg-query",
        "#!/usr/bin/env bash\n"
        f"printf 'dpkg-query %s\\n' \"$*\" >> {_shell_quote(log_path)}\n"
        "echo 'install ok installed'\n",
    )
    _write_executable(
        fake_bin / "sudo",
        f"#!/usr/bin/env bash\nprintf 'sudo %s\\n' \"$*\" >> {_shell_quote(log_path)}\nexit 99\n",
    )

    grep_path = shutil.which("grep")
    bash_path = shutil.which("bash")
    assert grep_path, "grep is required for this test"
    assert bash_path, "bash is required for this test"
    os.symlink(bash_path, fake_bin / "bash")
    os.symlink(grep_path, fake_bin / "grep")

    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    result = subprocess.run(
        ["bash", str(_script_path()), "libgl1", "jq"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    assert "all requested packages already installed" in result.stdout
    assert "sudo" not in log_path.read_text(encoding="utf-8")


def test_ci_install_headless_packages_installs_only_missing_packages(tmp_path: Path) -> None:
    """Missing packages are installed with bounded apt network options."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "commands.log"

    _write_executable(
        fake_bin / "dpkg-query",
        f"""#!/usr/bin/env bash
printf 'dpkg-query %s\\n' "$*" >> {_shell_quote(log_path)}
case "$*" in
  *libgl1*) echo 'install ok installed' ;;
  *) exit 1 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "sudo",
        f'#!/usr/bin/env bash\nprintf \'sudo %s\\n\' "$*" >> {_shell_quote(log_path)}\n"$@"\n',
    )
    _write_executable(
        fake_bin / "apt-get",
        f"#!/usr/bin/env bash\nprintf 'apt-get %s\\n' \"$*\" >> {_shell_quote(log_path)}\n",
    )

    grep_path = shutil.which("grep")
    bash_path = shutil.which("bash")
    assert grep_path, "grep is required for this test"
    assert bash_path, "bash is required for this test"
    os.symlink(bash_path, fake_bin / "bash")
    os.symlink(grep_path, fake_bin / "grep")

    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    result = subprocess.run(
        ["bash", str(_script_path()), "libgl1", "jq"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    log_text = log_path.read_text(encoding="utf-8")
    assert "apt-get -o Acquire::Retries=2" in log_text
    assert "-o Acquire::http::Timeout=20" in log_text
    assert "-o Acquire::https::Timeout=20" in log_text
    assert "install -y --no-install-recommends jq" in log_text
    assert "install -y --no-install-recommends libgl1" not in log_text
