"""Exercise runner replacement ordering without Docker or GitHub access."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[2] / "scripts/ci/self_hosted/setup.sh"


def _mock_command(directory: Path, name: str, body: str) -> None:
    command = directory / name
    command.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8")
    command.chmod(0o755)


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "network",
        "probe",
        "disk",
        "info",
        "ps",
        "df",
        "start",
        "delivery",
        "token",
        "job",
        "wait",
    ],
)
def test_supervisor_checks_isolation_before_each_registration(tmp_path: Path, failure: str) -> None:
    """Gate replacements, deliver stdin, and retry fetch/delivery/job/wait failures."""
    commands = tmp_path / "commands"
    commands.mkdir()
    _mock_command(commands, "hostname", "printf 'imech039\\n'\n")
    _mock_command(
        commands,
        "docker",
        """
case "$1:$2" in
  network:inspect) printf '{}\\n' ;;
  info:-f)
    printf 'docker-info\\n' >>"$TRACE"
    [[ "$FAILURE" != info ]] || exit 1
    printf '/var/lib/docker\\n'
    ;;
  ps:--format)
    printf 'docker-ps\\n' >>"$TRACE"
    [[ "$FAILURE" != ps ]]
    ;;
  run:*)
    case " $* " in
      *' --entrypoint '*)
        printf 'probe\\n' >>"$TRACE"
        [[ "$FAILURE" != probe ]]
        ;;
      *)
        printf 'runner-start\\n' >>"$TRACE"
        [[ "$FAILURE" != start ]]
        ;;
    esac
    ;;
  attach:*)
    # Empty upstream leaves the old attach waiting for a tokenless container.
    value="$(cat)"
    if [[ -z "$value" ]]; then /bin/sleep 30; fi
    # Nonempty stdin can end attach successfully before the actual job exits.
    printf 'attach-ended\\n' >>"$TRACE"
    ;;
  exec:*)
    [[ "$2" == --interactive ]]
    [[ "$4:$5:$6" == 'bash:-c:cat > /proc/1/fd/0' ]]
    value="$(cat)"
    [[ "$FAILURE" == token || "$value" == fixture-token ]]
    printf 'delivery\\n' >>"$TRACE"
    [[ "$FAILURE" != delivery ]]
    ;;
  wait:*)
    printf 'runner-wait\\n' >>"$TRACE"
    [[ "$FAILURE" != wait ]] || exit 1
    if [[ "$FAILURE" == job ]]; then printf '1\\n'; else printf '0\\n'; fi
    ;;
  stop:*) printf 'runner-stop\\n' >>"$TRACE" ;;
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
        "printf 'gh\\n' >>\"$TRACE\"\n"
        '[[ "$FAILURE" != token ]] || exit 1\n'
        "printf 'fixture-token\\n'\n",
    )
    _mock_command(
        commands,
        "df",
        """
printf 'df\\n' >>"$TRACE"
[[ "$FAILURE" != df ]] || exit 1
available=20971520
[[ "$FAILURE" != disk ]] || available=20971519
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'
printf 'fixture 100000000 0 %s 0%% /var/lib/docker\\n' "$available"
""",
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
        XDG_STATE_HOME=str(tmp_path / "state"),
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
    assert "fixture-token" not in result.stdout + result.stderr
    events = (tmp_path / "trace").read_text(encoding="utf-8").splitlines()
    checked = ["network-verify", "probe", "docker-info", "docker-ps", "df"]
    expected = {
        "none": checked + ["runner-start", "gh", "delivery", "runner-wait", "sleep:15"],
        "network": ["network-verify", "sleep:60"],
        "probe": ["network-verify", "probe", "sleep:60"],
        "info": checked[:3] + ["sleep:60"],
        "ps": checked[:4] + ["sleep:60"],
        "df": checked + ["sleep:60"],
        "disk": checked + ["sleep:60"],
        "start": checked + ["runner-start", "sleep:15"],
        "delivery": checked + ["runner-start", "gh", "delivery", "runner-stop", "sleep:15"],
        "token": checked + ["runner-start", "gh", "delivery", "runner-stop", "sleep:15"],
        "job": checked
        + ["runner-start", "gh", "delivery", "runner-wait", "runner-stop", "sleep:15"],
        "wait": checked
        + ["runner-start", "gh", "delivery", "runner-wait", "runner-stop", "sleep:15"],
    }
    assert events == expected[failure] * 2
    expected_error = {
        "network": "network isolation check failed",
        "probe": "network isolation check failed",
        "info": "Docker disk capacity check failed",
        "ps": "Docker disk capacity check failed",
        "df": "Docker disk capacity check failed",
        "disk": "Docker disk capacity check failed",
        "start": "container failed to start",
        "delivery": "failed to register",
        "token": "failed to register",
        "job": "failed to register",
        "wait": "failed to register",
    }
    if failure != "none":
        assert expected_error[failure] in result.stderr
    if failure == "probe":
        assert "Network isolation probe failed" in result.stderr
    if failure == "disk":
        assert "20 GiB required" in result.stderr


@pytest.mark.parametrize("running_slots", range(4))
@pytest.mark.parametrize("below_threshold", [False, True])
def test_disk_admission_counts_running_slots(
    tmp_path: Path, running_slots: int, below_threshold: bool
) -> None:
    """Each concurrent slot needs another 20 GiB of available Docker-root space."""
    commands = tmp_path / "commands"
    commands.mkdir()
    _mock_command(commands, "hostname", "printf 'imech039\\n'\n")
    _mock_command(
        commands,
        "docker",
        """
case "$1:$2" in
  network:inspect) printf '{}\\n' ;;
  info:-f) printf '/var/lib/docker\\n' ;;
  ps:--format)
    for ((slot=1; slot<=RUNNING_SLOTS; slot++)); do
      printf 'robot-sf-ci-imech156-u-%s\\n' "$slot"
    done
    ;;
  run:*)
    if [[ " $* " != *' --entrypoint '* ]]; then
      printf 'runner-start\\n' >>"$TRACE"
    fi
    ;;
  exec:*) cat >/dev/null ;;
  wait:*) printf '0\\n' ;;
  *) exit 2 ;;
esac
""",
    )
    _mock_command(commands, "jq", "exit 0\n")
    _mock_command(commands, "gh", "printf 'fixture-token\\n'\n")
    _mock_command(
        commands,
        "df",
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf 'fixture 100000000 0 %s 0%% /var/lib/docker\\n' \"$AVAILABLE_KIB\"\n",
    )
    _mock_command(commands, "sleep", "exit 99\n")
    required_gib = 20 * (running_slots + 1)
    environment = os.environ.copy()
    environment.update(
        PATH=f"{commands}:{environment['PATH']}",
        TRACE=str(tmp_path / "trace"),
        RUNNING_SLOTS=str(running_slots),
        AVAILABLE_KIB=str(required_gib * 1024 * 1024 - int(below_threshold)),
        XDG_STATE_HOME=str(tmp_path / "state"),
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
    events = (
        (tmp_path / "trace").read_text(encoding="utf-8").splitlines()
        if (tmp_path / "trace").exists()
        else []
    )
    assert ("runner-start" in events) is not below_threshold
    assert (tmp_path / "state/robot-sf-ci-runners/disk-admission.lock").exists()
    if below_threshold:
        assert (
            f"{required_gib} GiB required for {running_slots} running slots plus one"
            in result.stderr
        )


def test_disk_admission_serializes_simultaneous_supervisors(tmp_path: Path) -> None:
    """A second slot cannot check space until the first is visible to Docker ps."""
    commands = tmp_path / "commands"
    commands.mkdir()
    _mock_command(commands, "hostname", "printf 'imech039\\n'\n")
    _mock_command(
        commands,
        "docker",
        """
case "$1:$2" in
  network:inspect) printf '{}\\n' ;;
  info:-f) printf '/var/lib/docker\\n' ;;
  ps:--format)
    [[ ! -f "$ACTIVE_SLOT" ]] || cat "$ACTIVE_SLOT"
    if [[ ! -f "$FIRST_PS" ]]; then
      : >"$FIRST_PS"
      /bin/sleep 0.5
    fi
    ;;
  run:*)
    if [[ " $* " != *' --entrypoint '* ]]; then
      printf 'runner-start\\n' >>"$TRACE"
      printf 'robot-sf-ci-imech039-1\\n' >"$ACTIVE_SLOT"
    fi
    ;;
  exec:*) cat >/dev/null ;;
  wait:*) printf '0\\n' ;;
  *) exit 2 ;;
esac
""",
    )
    _mock_command(commands, "jq", "exit 0\n")
    _mock_command(commands, "gh", "printf 'fixture-token\\n'\n")
    _mock_command(
        commands,
        "df",
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf 'fixture 100000000 0 40894464 0%% /var/lib/docker\\n'\n",
    )
    _mock_command(commands, "sleep", 'printf \'sleep:%s\\n\' "$1" >>"$TRACE"\nexit 99\n')
    environment = os.environ.copy()
    environment.update(
        PATH=f"{commands}:{environment['PATH']}",
        TRACE=str(tmp_path / "trace"),
        ACTIVE_SLOT=str(tmp_path / "active-slot"),
        FIRST_PS=str(tmp_path / "first-ps"),
        XDG_STATE_HOME=str(tmp_path / "state"),
    )
    processes: list[subprocess.Popen[str]] = []
    try:
        processes.append(
            subprocess.Popen(
                ["bash", str(SETUP), "supervise", "1"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
        deadline = time.monotonic() + 5
        while not (tmp_path / "first-ps").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "first-ps").exists()
        processes.append(
            subprocess.Popen(
                ["bash", str(SETUP), "supervise", "2"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
        results = [process.communicate(timeout=10) for process in processes]
        assert all(process.returncode == 99 for process in processes), results
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()

    events = (tmp_path / "trace").read_text(encoding="utf-8").splitlines()
    assert events.count("runner-start") == 1
    assert events.count("sleep:15") == 1
    assert events.count("sleep:60") == 1
    assert "40 GiB required for 1 running slots plus one" in results[1][1]


@pytest.mark.parametrize("reported", ["imech039", "auxme-imech039"])
def test_setup_accepts_auxme_hostname_prefix(tmp_path: Path, reported: str) -> None:
    """The lab hosts report ``auxme-imech0xx``; the slot limit must still apply."""
    commands = tmp_path / "commands"
    commands.mkdir()
    _mock_command(commands, "hostname", f"printf '{reported}\\n'\n")
    environment = os.environ.copy()
    environment["PATH"] = f"{commands}:{environment['PATH']}"

    result = subprocess.run(
        ["bash", str(SETUP), "status", "3"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "SLOT must be in 1..2 on imech039" in result.stderr


@pytest.mark.parametrize(
    ("reported", "cpus", "memory", "workers"),
    [
        ("auxme-imech036", "4", "8g", "2"),
        ("auxme-imech039", "4", "8g", "2"),
        ("imech156-u", "8", "16g", "4"),
    ],
)
def test_runner_container_uses_host_slot_size(
    tmp_path: Path, reported: str, cpus: str, memory: str, workers: str
) -> None:
    """Each host starts runner containers with its own CPU, memory and pytest-worker size."""
    commands = tmp_path / "commands"
    commands.mkdir()
    _mock_command(commands, "hostname", f"printf '{reported}\\n'\n")
    _mock_command(
        commands,
        "docker",
        """
case "$1:$2" in
  network:inspect) printf '{}\\n' ;;
  info:-f) printf '/var/lib/docker\\n' ;;
  ps:--format) ;;
  run:*)
    case " $* " in
      *' --entrypoint '*) ;;
      *) printf '%s\\n' "$@" >"$RUN_ARGS"; exit 1 ;;
    esac
    ;;
  *) exit 2 ;;
esac
""",
    )
    _mock_command(commands, "jq", "true\n")
    _mock_command(
        commands,
        "df",
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf 'fixture 100000000 0 209715200 0%% /var/lib/docker\\n'\n",
    )
    _mock_command(commands, "sleep", "exit 99\n")
    environment = os.environ.copy()
    environment.update(
        PATH=f"{commands}:{environment['PATH']}",
        RUN_ARGS=str(tmp_path / "run-args"),
        XDG_STATE_HOME=str(tmp_path / "state"),
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
    args = (tmp_path / "run-args").read_text(encoding="utf-8").splitlines()
    assert args[args.index("--cpus") + 1] == cpus
    assert args[args.index("--memory") + 1] == memory
    assert args[args.index("--memory-swap") + 1] == memory
    assert f"PYTEST_NUM_WORKERS={workers}" in args
    assert "OMP_NUM_THREADS=1" in args
