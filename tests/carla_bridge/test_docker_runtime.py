"""CARLA-free tests for the pinned Docker runtime interface."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_carla_docker_runtime_uses_pinned_0_9_16_image() -> None:
    """The runtime contract should never default to CARLA latest."""
    from robot_sf_carla_bridge.docker_runtime import CARLA_DOCKER_IMAGE, validate_carla_image

    assert CARLA_DOCKER_IMAGE == "carlasim/carla:0.9.16"

    with pytest.raises(ValueError, match="must be pinned"):
        validate_carla_image("carlasim/carla:latest")


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
        ("docker", "run", "--rm", "--gpus", "all"): CommandResult(
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
