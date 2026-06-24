"""CARLA-free tests for the pinned Docker runtime interface."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest


def _pretend_linux_x86_64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make preflight tests exercise Docker checks past the CARLA host-platform gate."""
    from robot_sf_carla_bridge import docker_runtime

    monkeypatch.setattr(docker_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(docker_runtime.platform, "machine", lambda: "x86_64")


def test_carla_docker_runtime_uses_pinned_0_9_16_image() -> None:
    """The runtime contract should never default to CARLA latest."""
    from robot_sf_carla_bridge.docker_runtime import CARLA_DOCKER_IMAGE, validate_carla_image

    assert CARLA_DOCKER_IMAGE == "carlasim/carla:0.9.16"

    with pytest.raises(ValueError, match="must be pinned"):
        validate_carla_image("carlasim/carla:latest")
    with pytest.raises(ValueError, match="carlasim/carla:0.9.16"):
        validate_carla_image("carlasim/carla:0.9.15")
    with pytest.raises(ValueError, match="carlasim/carla:0.9.16"):
        validate_carla_image("localhost:5000/carlasim/carla")


def test_build_carla_server_container_command_maps_ports_and_runs_headless() -> None:
    """The server command should expose CARLA ports explicitly and run offscreen."""
    from robot_sf_carla_bridge.docker_runtime import build_carla_server_container_command

    command = build_carla_server_container_command(container_name="robot-sf-carla-test")

    assert command[:4] == ["docker", "run", "--rm", "--detach"]
    assert "--gpus" in command
    assert command[command.index("--gpus") + 1] == "all"
    assert "--name" in command
    assert command[command.index("--name") + 1] == "robot-sf-carla-test"
    assert "-p" in command
    assert "2000:2000" in command
    assert "2001:2001" in command
    assert "2002:2002" in command
    assert "carlasim/carla:0.9.16" in command
    assert "-RenderOffScreen" in " ".join(command)
    assert "-carla-rpc-port=2000" in " ".join(command)


def test_build_carla_server_container_command_derives_ports_from_rpc_port() -> None:
    """Custom RPC ports should keep Docker mappings and CARLA startup args aligned."""
    from robot_sf_carla_bridge.docker_runtime import build_carla_server_container_command

    command = build_carla_server_container_command(rpc_port=2100)

    assert "2100:2100" in command
    assert "2101:2101" in command
    assert "2102:2102" in command
    assert "2000:2000" not in command
    assert "-carla-rpc-port=2100" in " ".join(command)


def test_run_command_reports_missing_executable_without_traceback() -> None:
    """Host prerequisite probes should fail closed when a binary is absent."""
    from robot_sf_carla_bridge.docker_runtime import run_command

    result = run_command(["robot-sf-missing-cmd-for-test"], timeout_s=0.1)

    assert result.returncode == 127
    assert result.stdout == ""
    assert result.stderr == "Command not found: robot-sf-missing-cmd-for-test"


def test_run_command_reports_timeout_without_traceback(monkeypatch) -> None:
    """Hung host commands should become machine-readable command results."""
    from robot_sf_carla_bridge import docker_runtime
    from robot_sf_carla_bridge.docker_runtime import run_command

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["docker", "version"], timeout=3.5)

    monkeypatch.setattr(docker_runtime.subprocess, "run", fake_run)

    result = run_command(["docker", "version"], timeout_s=3.5)

    assert result.returncode == 124
    assert result.stdout == ""
    assert result.stderr == "Command timed out after 3.5s"


def test_check_carla_runtime_ports_defaults_to_loopback(monkeypatch) -> None:
    """The preflight check should avoid binding all network interfaces by default."""
    from robot_sf_carla_bridge import docker_runtime

    bind_calls: list[tuple[str, int]] = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            bind_calls.append(address)

    monkeypatch.setattr(docker_runtime.socket, "socket", lambda *args, **kwargs: FakeSocket())

    assert docker_runtime.check_carla_runtime_ports((2000,)) == []
    assert bind_calls == [(docker_runtime.CARLA_DEFAULT_HOST, 2000)]


def test_preflight_reports_not_available_when_docker_daemon_is_unreachable() -> None:
    """No-sudo Docker daemon failure should stop before GPU/image/runtime work."""
    from robot_sf_carla_bridge.docker_runtime import CommandResult, run_carla_docker_preflight

    def fake_runner(command: list[str], *, timeout_s: float) -> CommandResult:
        assert command[:2] == ["docker", "version"]
        assert timeout_s > 0
        return CommandResult(
            args=command,
            returncode=1,
            stdout='{"Client": {"Version": "29.4.2"}, "Server": null}',
            stderr="permission denied while trying to connect to the docker API",
        )

    status = run_carla_docker_preflight(runner=fake_runner)

    assert status["status"] == "not-available"
    assert status["reason"] == "Docker daemon is not reachable without sudo"
    assert status["missing_capability"] == "docker-daemon"
    assert status["checks"][0]["name"] == "docker_daemon"
    assert status["checks"][0]["status"] == "not-available"


def test_preflight_requires_50_gib_free_before_pull_when_image_absent(monkeypatch) -> None:
    """Absent images should fail before pull when Docker storage has too little free space."""
    from robot_sf_carla_bridge import docker_runtime
    from robot_sf_carla_bridge.docker_runtime import CommandResult, run_carla_docker_preflight

    _pretend_linux_x86_64(monkeypatch)
    responses = {
        ("docker", "version"): CommandResult(["docker", "version"], 0, '{"Server": {}}', ""),
        ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"): (
            CommandResult(
                ["nvidia-smi"],
                0,
                "NVIDIA GeForce RTX 3080, 580.142, 10240 MiB\n",
                "",
            )
        ),
        ("docker", "run", "--rm", "--pull=never", "--gpus", "all"): CommandResult(
            ["docker", "run"],
            0,
            "NVIDIA-SMI 580.142\n",
            "",
        ),
        ("docker", "image", "inspect", "carlasim/carla:0.9.16"): CommandResult(
            ["docker", "image", "inspect"],
            1,
            "",
            "No such image",
        ),
        ("docker", "info", "--format", "{{json .DockerRootDir}}"): CommandResult(
            ["docker", "info"],
            0,
            '"/var/lib/docker"\n',
            "",
        ),
    }

    def fake_runner(command: list[str], *, timeout_s: float) -> CommandResult:
        for prefix, result in responses.items():
            if tuple(command[: len(prefix)]) == prefix:
                return result
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(docker_runtime.shutil, "disk_usage", lambda path: (100, 100, 49 * 1024**3))
    monkeypatch.setattr(docker_runtime, "check_carla_runtime_ports", lambda ports: [])

    status = run_carla_docker_preflight(runner=fake_runner, require_carla_api=False)

    assert status["status"] == "not-available"
    assert status["missing_capability"] == "docker-storage"
    assert "50 GiB" in status["reason"]
    assert status["docker_storage"]["free_gib"] == pytest.approx(49.0)


def test_preflight_nvidia_container_check_does_not_pull_test_image(monkeypatch) -> None:
    """NVIDIA runtime probing should not download CUDA images unless explicitly prepared."""
    from robot_sf_carla_bridge.docker_runtime import CommandResult, run_carla_docker_preflight

    _pretend_linux_x86_64(monkeypatch)
    observed_commands: list[list[str]] = []

    def fake_runner(command: list[str], *, timeout_s: float) -> CommandResult:
        observed_commands.append(command)
        if command[:2] == ["docker", "version"]:
            return CommandResult(command, 0, '{"Server": {}}', "")
        if command[:1] == ["nvidia-smi"]:
            return CommandResult(command, 0, "NVIDIA GeForce RTX 3080, 580.142, 10240 MiB\n", "")
        if command[:2] == ["docker", "run"]:
            return CommandResult(command, 1, "", "pull access denied")
        raise AssertionError(f"unexpected command: {command}")

    status = run_carla_docker_preflight(runner=fake_runner, require_carla_api=False)

    docker_run = next(command for command in observed_commands if command[:2] == ["docker", "run"])
    assert "--pull=never" in docker_run
    assert status["status"] == "not-available"
    assert status["missing_capability"] == "nvidia-container-toolkit"


def test_preflight_fails_when_post_pull_image_inspect_fails(monkeypatch) -> None:
    """Pulled images should not be marked available until metadata inspection succeeds."""
    from robot_sf_carla_bridge import docker_runtime
    from robot_sf_carla_bridge.docker_runtime import CommandResult, run_carla_docker_preflight

    _pretend_linux_x86_64(monkeypatch)
    image_inspect_count = 0

    def fake_runner(command: list[str], *, timeout_s: float) -> CommandResult:
        nonlocal image_inspect_count
        if command[:2] == ["docker", "version"]:
            return CommandResult(command, 0, '{"Server": {}}', "")
        if command[:1] == ["nvidia-smi"]:
            return CommandResult(command, 0, "NVIDIA GeForce RTX 3080, 580.142, 10240 MiB\n", "")
        if command[:2] == ["docker", "run"]:
            return CommandResult(command, 0, "NVIDIA-SMI 580.142\n", "")
        if command[:3] == ["docker", "image", "inspect"]:
            image_inspect_count += 1
            return CommandResult(command, 1, "", "No such image metadata")
        if command[:2] == ["docker", "info"]:
            return CommandResult(command, 0, '"/var/lib/docker"\n', "")
        if command[:2] == ["docker", "pull"]:
            return CommandResult(command, 0, "pulled\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        docker_runtime.shutil, "disk_usage", lambda path: (100, 80 * 1024**3, 80 * 1024**3)
    )
    monkeypatch.setattr(docker_runtime, "check_carla_runtime_ports", lambda ports: [])

    status = run_carla_docker_preflight(runner=fake_runner, pull=True, require_carla_api=False)

    assert image_inspect_count == 2
    assert status["status"] == "not-available"
    assert status["missing_capability"] == "carla-image"
    assert status["checks"][-1]["name"] == "carla_image_inspect"


def test_preflight_requires_image_digest_and_size(monkeypatch) -> None:
    """Image metadata without digest or size should fail before runtime evidence is trusted."""
    from robot_sf_carla_bridge import docker_runtime
    from robot_sf_carla_bridge.docker_runtime import CommandResult, run_carla_docker_preflight

    _pretend_linux_x86_64(monkeypatch)

    def fake_runner(command: list[str], *, timeout_s: float) -> CommandResult:
        if command[:2] == ["docker", "version"]:
            return CommandResult(command, 0, '{"Server": {}}', "")
        if command[:1] == ["nvidia-smi"]:
            return CommandResult(command, 0, "NVIDIA GeForce RTX 3080, 580.142, 10240 MiB\n", "")
        if command[:2] == ["docker", "run"]:
            return CommandResult(command, 0, "NVIDIA-SMI 580.142\n", "")
        if command[:3] == ["docker", "image", "inspect"]:
            return CommandResult(command, 0, json.dumps([{"RepoDigests": [], "Size": None}]), "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(docker_runtime, "check_carla_runtime_ports", lambda ports: [])

    status = run_carla_docker_preflight(runner=fake_runner, require_carla_api=False)

    assert status["status"] == "not-available"
    assert status["missing_capability"] == "carla-image"
    assert "missing digest or size" in status["reason"]


def test_build_carla_client_health_summary_records_versions_and_map(monkeypatch) -> None:
    """Health checks should summarize live client/server/map state without replay claims."""
    from robot_sf_carla_bridge import docker_runtime

    class FakeClient:
        """Small fake for the subset of the CARLA client API used by the health check."""

        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port
            self.timeout = None

        def set_timeout(self, timeout_s: float) -> None:
            self.timeout = timeout_s

        def get_client_version(self) -> str:
            return "0.9.16"

        def get_server_version(self) -> str:
            return "0.9.16"

        def get_world(self):
            return SimpleNamespace(get_map=lambda: SimpleNamespace(name="Town01"))

    fake_carla = SimpleNamespace(Client=FakeClient)
    monkeypatch.setattr(docker_runtime, "require_carla", lambda: fake_carla)

    summary = docker_runtime.build_carla_client_health_summary(host="127.0.0.1", port=2000)

    assert summary == {
        "status": "connected",
        "host": "127.0.0.1",
        "port": 2000,
        "client_version": "0.9.16",
        "server_version": "0.9.16",
        "map": "Town01",
        "boundary": "client-connectivity-only",
    }


def test_carla_docker_runtime_cli_prints_preflight_json(monkeypatch, capsys) -> None:
    """The packaged CLI should expose a machine-readable preflight status."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import carla_docker_runtime_main

    monkeypatch.setattr(
        cli_module,
        "run_carla_docker_preflight",
        lambda **kwargs: {
            "schema_version": "carla-docker-runtime.v1",
            "status": "not-available",
            "reason": "Docker daemon is not reachable without sudo",
            "image": "carlasim/carla:0.9.16",
            "checks": [],
        },
    )

    exit_code = carla_docker_runtime_main(["preflight", "--json"])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["image"] == "carlasim/carla:0.9.16"


def test_carla_docker_runtime_cli_prints_live_replay_json(monkeypatch, capsys) -> None:
    """The packaged Docker runtime CLI should expose the live replay workflow."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import carla_docker_runtime_main

    observed: dict[str, object] = {}

    def fake_live_replay(**kwargs):
        observed.update(kwargs)
        return {
            "schema_version": "carla-docker-runtime.v1",
            "stage": "live-replay",
            "status": "oracle-replay",
            "mode": "oracle-replay",
            "reason": "Oracle transforms were replayed against a live CARLA world",
        }

    monkeypatch.setattr(cli_module, "run_carla_docker_live_replay", fake_live_replay)

    exit_code = carla_docker_runtime_main(
        [
            "live-replay",
            "--manifest",
            "output/carla/manifest.json",
            "--scenario-id",
            "unit_crossing",
            "--max-steps",
            "3",
            "--json",
        ]
    )

    assert exit_code == 0
    status = json.loads(capsys.readouterr().out)
    assert status["stage"] == "live-replay"
    assert status["status"] == "oracle-replay"
    assert observed["manifest_path"] == "output/carla/manifest.json"
    assert observed["scenario_id"] == "unit_crossing"
    assert observed["max_steps"] == 3


def test_runtime_smoke_stops_container_when_health_check_fails(monkeypatch) -> None:
    """Lifecycle failures should still collect logs and remove the CARLA container."""
    from robot_sf_carla_bridge import docker_runtime
    from robot_sf_carla_bridge.availability import CarlaUnavailableError
    from robot_sf_carla_bridge.docker_runtime import CommandResult, run_carla_docker_runtime_smoke

    commands: list[list[str]] = []

    def fake_runner(command: list[str], *, timeout_s: float) -> CommandResult:
        commands.append(command)
        if command[:2] == ["docker", "run"]:
            return CommandResult(command, 0, "container-123\n", "")
        if command[:2] == ["docker", "logs"]:
            return CommandResult(command, 0, "log tail\n", "")
        if command[:2] == ["docker", "stop"]:
            return CommandResult(command, 0, "container-123\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        docker_runtime,
        "run_carla_docker_preflight",
        lambda **kwargs: {
            "schema_version": "carla-docker-runtime.v1",
            "status": "available",
            "reason": "CARLA Docker runtime prerequisites are available",
            "image": "carlasim/carla:0.9.16",
            "image_digest": "sha256:abc",
            "checks": [],
        },
    )
    monkeypatch.setattr(
        docker_runtime,
        "build_carla_client_health_summary",
        lambda **kwargs: (_ for _ in ()).throw(CarlaUnavailableError("client failed")),
    )

    summary = run_carla_docker_runtime_smoke(
        runner=fake_runner,
        startup_timeout_s=0,
        retry_interval_s=0,
    )

    assert summary["status"] == "failed"
    assert summary["reason"] == "client failed"
    assert summary["container"]["id"] == "container-123"
    assert summary["logs_tail"] == "log tail\n"
    assert any(command[:2] == ["docker", "stop"] for command in commands)


def test_runtime_smoke_attempts_cleanup_when_container_start_fails(monkeypatch) -> None:
    """Startup failures should still produce log-tail and cleanup evidence."""
    from robot_sf_carla_bridge import docker_runtime
    from robot_sf_carla_bridge.docker_runtime import CommandResult, run_carla_docker_runtime_smoke

    commands: list[list[str]] = []

    def fake_runner(command: list[str], *, timeout_s: float) -> CommandResult:
        commands.append(command)
        if command[:2] == ["docker", "run"]:
            return CommandResult(command, 1, "", "startup failed")
        if command[:2] == ["docker", "logs"]:
            return CommandResult(command, 1, "", "no logs")
        if command[:2] == ["docker", "stop"]:
            return CommandResult(command, 1, "", "no such container")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        docker_runtime,
        "run_carla_docker_preflight",
        lambda **kwargs: {
            "schema_version": "carla-docker-runtime.v1",
            "status": "available",
            "reason": "CARLA Docker runtime prerequisites are available",
            "image": "carlasim/carla:0.9.16",
            "image_digest": "sha256:abc",
            "checks": [],
        },
    )

    summary = run_carla_docker_runtime_smoke(runner=fake_runner)

    assert summary["status"] == "failed"
    assert summary["reason"] == "Failed to start pinned CARLA server container"
    assert summary["stderr"] == "startup failed"
    assert summary["logs_tail"] == "no logs"
    assert summary["cleanup"]["stderr"] == "no such container"
    assert any(command[:2] == ["docker", "stop"] for command in commands)


def test_docker_live_replay_runs_replay_before_cleanup(monkeypatch) -> None:
    """The live replay wrapper should keep CARLA up for replay, then stop the container."""
    from robot_sf_carla_bridge import docker_runtime
    from robot_sf_carla_bridge.docker_runtime import CommandResult, run_carla_docker_live_replay

    commands: list[list[str]] = []
    health_calls = 0

    def fake_runner(command: list[str], *, timeout_s: float) -> CommandResult:
        commands.append(command)
        if command[:2] == ["docker", "run"]:
            return CommandResult(command, 0, "container-123\n", "")
        if command[:2] == ["docker", "logs"]:
            return CommandResult(command, 0, "live log tail\n", "")
        if command[:2] == ["docker", "stop"]:
            return CommandResult(command, 0, "container-123\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        docker_runtime,
        "run_carla_docker_preflight",
        lambda **kwargs: {
            "schema_version": "carla-docker-runtime.v1",
            "status": "available",
            "reason": "CARLA Docker runtime prerequisites are available",
            "image": "carlasim/carla:0.9.16",
            "image_digest": "sha256:abc",
            "checks": [],
        },
    )

    def fake_health_summary(**kwargs):
        nonlocal health_calls
        health_calls += 1
        if health_calls == 1:
            raise RuntimeError("simulator not ready")
        return {"status": "connected", "map": "Town01"}

    monkeypatch.setattr(docker_runtime, "build_carla_client_health_summary", fake_health_summary)
    monkeypatch.setattr(
        docker_runtime,
        "run_t1_oracle_live_replay_against_server",
        lambda *args, **kwargs: {
            "schema_version": "carla-t1-oracle-live-replay.v1",
            "status": "oracle-replay",
            "mode": "oracle-replay",
            "stage": "live-replay",
            "reason": "Oracle transforms were replayed against a live CARLA world",
        },
    )

    summary = run_carla_docker_live_replay(
        "output/carla/manifest.json",
        runner=fake_runner,
        startup_timeout_s=1,
        retry_interval_s=0,
    )

    assert summary["status"] == "oracle-replay"
    assert summary["stage"] == "live-replay"
    assert summary["reason"] == "Oracle transforms were replayed against a live CARLA world"
    assert summary["container"]["id"] == "container-123"
    assert summary["health"] == {"status": "connected", "map": "Town01"}
    assert summary["replay"]["status"] == "oracle-replay"
    assert summary["logs_tail"] == "live log tail\n"
    assert health_calls == 2
    assert any(command[:2] == ["docker", "stop"] for command in commands)
