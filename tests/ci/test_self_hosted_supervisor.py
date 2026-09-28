"""Exercise runner replacement ordering without Docker or GitHub access."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[2] / "scripts/ci/self_hosted/setup.sh"


def _mock_command(directory: Path, name: str, body: str) -> None:
    command = directory / name
    command.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8")
    command.chmod(0o755)


@pytest.mark.parametrize("failure", ["none", "network", "probe"])
def test_supervisor_checks_isolation_before_each_registration(tmp_path: Path, failure: str) -> None:
    """A failed check skips token requests and waits 60 seconds before retrying."""
    commands = tmp_path / "commands"
    commands.mkdir()
    _mock_command(commands, "hostname", "printf 'imech039\\n'\n")
    _mock_command(
        commands,
        "docker",
        """
case "$1:$2" in
  network:inspect) printf '{}\\n' ;;
  run:*)
    case " $* " in
      *' --entrypoint '*)
        printf 'probe\\n' >>"$TRACE"
        [[ "$FAILURE" != probe ]]
        ;;
      *)
        cat >/dev/null
        printf 'runner\\n' >>"$TRACE"
        ;;
    esac
    ;;
  *) exit 2 ;;
esac
""",
    )
    _mock_command(
        commands,
        "jq",
        'printf \'network-verify\\n\' >>"$TRACE"\n[[ "$FAILURE" != network ]]\n',
    )
    _mock_command(
        commands,
        "gh",
        "printf 'gh\\n' >>\"$TRACE\"\nprintf 'fixture-token\\n'\n",
    )
    _mock_command(
        commands,
        "sleep",
        """
printf 'sleep:%s\\n' "$1" >>"$TRACE"
count=0
[[ ! -f "$SLEEP_COUNT" ]] || count="$(<"$SLEEP_COUNT")"
count=$((count + 1))
printf '%s\\n' "$count" >"$SLEEP_COUNT"
[[ "$count" -lt 2 ]] || exit 99
""",
    )
    environment = os.environ.copy()
    environment.update(
        PATH=f"{commands}:{environment['PATH']}",
        TRACE=str(tmp_path / "trace"),
        SLEEP_COUNT=str(tmp_path / "sleep-count"),
        FAILURE=failure,
    )

    result = subprocess.run(
        ["bash", str(SETUP), "supervise", "1"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 99, result.stderr
    events = (tmp_path / "trace").read_text(encoding="utf-8").splitlines()
    if failure == "none":
        assert events == ["network-verify", "probe", "gh", "runner", "sleep:15"] * 2
    elif failure == "probe":
        assert events == ["network-verify", "probe", "sleep:60"] * 2
        assert "Network isolation probe failed" in result.stderr
    else:
        assert events == ["network-verify", "sleep:60"] * 2
    if failure != "none":
        assert "network isolation check failed" in result.stderr
