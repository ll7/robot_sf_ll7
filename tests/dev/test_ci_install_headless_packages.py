"""Tests for the CI headless package installer helper."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import sys
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


def _timer_test_environment(fake_bin: Path) -> dict[str, str]:
    """Provide only the runtime tools needed by the shell helper and its timer."""
    for command_name in ("bash", "date", "dirname", "grep", "mktemp", "rm", "sleep"):
        command_path = shutil.which(command_name)
        assert command_path, f"{command_name} is required for this test"
        os.symlink(command_path, fake_bin / command_name)
    os.symlink(sys.executable, fake_bin / "python3")

    env = os.environ.copy()
    env["PATH"] = str(fake_bin)
    return env


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

    env = _timer_test_environment(fake_bin)
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

    env = _timer_test_environment(fake_bin)
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


def test_ci_install_headless_packages_probe_failure_is_advisory(tmp_path: Path) -> None:
    """An unexpected package-probe failure still reaches the requested apt install."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "commands.log"

    _write_executable(fake_bin / "dpkg-query", "#!/usr/bin/env bash\nexit 42\n")
    _write_executable(
        fake_bin / "sudo",
        f'#!/usr/bin/env bash\nprintf \'sudo %s\\n\' "$*" >> {_shell_quote(log_path)}\n"$@"\n',
    )
    _write_executable(
        fake_bin / "apt-get",
        f"#!/usr/bin/env bash\nprintf 'apt-get %s\\n' \"$*\" >> {_shell_quote(log_path)}\n",
    )

    result = subprocess.run(
        ["bash", str(_script_path()), "poppler-utils"],
        capture_output=True,
        text=True,
        check=False,
        env=_timer_test_environment(fake_bin),
    )

    assert result.returncode == 0, result.stderr
    assert "warning=package_probe_failed package=poppler-utils probe_rc=42" in result.stdout
    assert "install -y --no-install-recommends poppler-utils" in log_path.read_text(
        encoding="utf-8"
    )


def test_ci_install_headless_packages_falls_back_to_official_mirror_after_timeout(
    tmp_path: Path,
) -> None:
    """An unavailable hosted-runner mirror should not block official Ubuntu packages."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "commands.log"

    _write_executable(
        fake_bin / "dpkg-query",
        "#!/usr/bin/env bash\nexit 1\n",
    )
    _write_executable(
        fake_bin / "sudo",
        f'#!/usr/bin/env bash\nprintf \'sudo %s\\n\' "$*" >> {_shell_quote(log_path)}\n"$@"\n',
    )
    _write_executable(
        fake_bin / "apt-get",
        f"""#!/usr/bin/env bash
printf 'apt-get %s\\n' "$*" >> {_shell_quote(log_path)}
if [[ "$*" == *' update' ]]; then
  if [[ "$*" == *'Dir::Etc::sourcelist='* ]]; then
    exit 0
  fi
  exit 124
fi
""",
    )

    env = _timer_test_environment(fake_bin)
    env["CI_HEADLESS_APT_MIRROR_FALLBACK_TIMEOUT_SECONDS"] = "1"
    result = subprocess.run(
        ["bash", str(_script_path()), "poppler-utils"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "warning=apt_update_official_mirror_fallback" in result.stdout
    log_text = log_path.read_text(encoding="utf-8")
    assert "Dir::Etc::sourcelist=" in log_text
    assert "Dir::Etc::sourceparts=-" in log_text
    assert "install -y --no-install-recommends poppler-utils" in log_text


def test_ci_install_headless_packages_reports_update_timeout_with_context(tmp_path: Path) -> None:
    """A slow apt update fails before the outer CI step timeout with actionable context."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    _write_executable(
        fake_bin / "dpkg-query",
        "#!/usr/bin/env bash\nexit 1\n",
    )
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\n"$@"\n',
    )
    _write_executable(
        fake_bin / "apt-get",
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *' update' ]]; then\n"
        "  if [[ \"$*\" == *'Dir::Etc::sourcelist='* ]]; then\n"
        "    exit 100\n"
        "  fi\n"
        "  echo 'Get:1 https://archive.ubuntu.com/ubuntu noble InRelease'\n"
        "  sleep 2\n"
        "fi\n",
    )
    env = _timer_test_environment(fake_bin)
    env["CI_HEADLESS_APT_PHASE_TIMEOUT_SECONDS"] = "1"

    result = subprocess.run(
        ["bash", str(_script_path()), "poppler-utils"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=10,
    )

    assert result.returncode == 124
    diagnostic = result.stderr
    assert "error=apt_update_timeout" in diagnostic
    assert "phase=update" in diagnostic
    assert "packages=poppler-utils" in diagnostic
    assert "sources=archive.ubuntu.com" in diagnostic
    assert "timeout_seconds=1" in diagnostic
    assert "elapsed_seconds=" in diagnostic
    assert "apt install" not in diagnostic


def test_ci_install_headless_packages_reports_install_timeout_with_context(tmp_path: Path) -> None:
    """A slow apt install is bounded independently after a successful update."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    _write_executable(
        fake_bin / "dpkg-query",
        "#!/usr/bin/env bash\nexit 1\n",
    )
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\n"$@"\n',
    )
    _write_executable(
        fake_bin / "apt-get",
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *' update' ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "echo 'Get:1 https://archive.ubuntu.com/ubuntu noble/main amd64 jq amd64 1.0'\n"
        "sleep 2\n",
    )
    env = _timer_test_environment(fake_bin)
    env["CI_HEADLESS_APT_PHASE_TIMEOUT_SECONDS"] = "1"

    result = subprocess.run(
        ["bash", str(_script_path()), "jq"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=10,
    )

    assert result.returncode == 124
    diagnostic = result.stderr
    assert "error=apt_install_timeout" in diagnostic
    assert "phase=install" in diagnostic
    assert "packages=jq" in diagnostic
    assert "sources=archive.ubuntu.com" in diagnostic
    assert "timeout_seconds=1" in diagnostic
    assert "elapsed_seconds=" in diagnostic


def test_ci_install_headless_packages_classifies_update_failure_by_source(tmp_path: Path) -> None:
    """A non-403 apt update error fails closed and reports its source and package set."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    _write_executable(
        fake_bin / "dpkg-query",
        "#!/usr/bin/env bash\nexit 1\n",
    )
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\n"$@"\n',
    )
    _write_executable(
        fake_bin / "apt-get",
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *' update' ]]; then\n"
        "  echo 'Err:1 https://archive.ubuntu.com/ubuntu noble InRelease'\n"
        "  echo '  500  Internal Server Error'\n"
        "  exit 100\n"
        "fi\n",
    )
    env = _timer_test_environment(fake_bin)

    result = subprocess.run(
        ["bash", str(_script_path()), "jq"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 100
    diagnostic = result.stderr
    assert "error=apt_update_failed" in diagnostic
    assert "phase=update" in diagnostic
    assert "packages=jq" in diagnostic
    assert "sources=archive.ubuntu.com" in diagnostic
    assert "apt_update_classification=failed_sources" in diagnostic


def test_ci_install_headless_packages_preserves_third_party_403_exception(tmp_path: Path) -> None:
    """A third-party 403 remains a warning while required packages still install."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "commands.log"

    _write_executable(
        fake_bin / "dpkg-query",
        "#!/usr/bin/env bash\nexit 1\n",
    )
    _write_executable(
        fake_bin / "sudo",
        f'#!/usr/bin/env bash\nprintf \'sudo %s\\n\' "$*" >> {_shell_quote(log_path)}\n"$@"\n',
    )
    _write_executable(
        fake_bin / "apt-get",
        "#!/usr/bin/env bash\n"
        f"printf 'apt-get %s\\n' \"$*\" >> {_shell_quote(log_path)}\n"
        "if [[ \"$*\" == *' update' ]]; then\n"
        "  echo 'Err:1 https://packages.example.invalid/repo stable InRelease'\n"
        "  echo '  403 Forbidden'\n"
        "  exit 100\n"
        "fi\n",
    )
    env = _timer_test_environment(fake_bin)

    result = subprocess.run(
        ["bash", str(_script_path()), "jq"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "warning=ignored_third_party_apt_403" in result.stdout
    assert "packages.example.invalid" in result.stdout
    assert "install -y --no-install-recommends jq" in log_path.read_text(encoding="utf-8")
