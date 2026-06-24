"""Pinned CARLA Docker runtime preflight and lifecycle helpers."""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from robot_sf_carla_bridge.availability import check_carla_availability, require_carla
from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

if TYPE_CHECKING:
    from pathlib import Path

CARLA_DOCKER_RUNTIME_SCHEMA_VERSION = "carla-docker-runtime.v1"
CARLA_DOCKER_IMAGE = "carlasim/carla:0.9.16"
CARLA_PYTHON_API_REQUIREMENT = "carla==0.9.16"
CARLA_DEFAULT_CONTAINER_NAME = "robot-sf-carla-0-9-16"
CARLA_DEFAULT_HOST = "127.0.0.1"
CARLA_DEFAULT_RPC_PORT = 2000
CARLA_PORTS = (2000, 2001, 2002)
CARLA_PULL_MIN_FREE_GIB = 50
CARLA_PULL_WARN_FREE_GIB = 80
NVIDIA_DOCKER_TEST_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"


@dataclass(frozen=True)
class CommandResult:
    """Captured command result used by injectable CARLA Docker runners."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Callable command runner protocol used for Docker/CARLA runtime tests."""

    def __call__(self, command: list[str], *, timeout_s: float) -> CommandResult:
        """Run one command with a timeout."""


def run_command(command: list[str], *, timeout_s: float = 30.0) -> CommandResult:
    """Run one host command and capture text output for runtime evidence.

    Returns:
        Captured command result with args, return code, stdout, and stderr.
    """

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandResult(
            args=list(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except FileNotFoundError:
        return CommandResult(
            args=list(command),
            returncode=127,
            stdout="",
            stderr=f"Command not found: {command[0]}",
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            args=list(command),
            returncode=124,
            stdout="",
            stderr=f"Command timed out after {timeout_s}s",
        )


def validate_carla_image(image: str) -> str:
    """Return a CARLA image reference only when it is explicitly pinned."""

    if image != CARLA_DOCKER_IMAGE:
        raise ValueError(
            f"CARLA Docker image must be pinned to {CARLA_DOCKER_IMAGE}; got {image!r}"
        )
    return image


def build_carla_server_container_command(
    *,
    image: str = CARLA_DOCKER_IMAGE,
    container_name: str = CARLA_DEFAULT_CONTAINER_NAME,
    rpc_port: int = CARLA_DEFAULT_RPC_PORT,
) -> list[str]:
    """Build the pinned, headless CARLA server container command.

    Returns:
        Docker command argv with explicit CARLA port mappings and offscreen flags.
    """

    image = validate_carla_image(image)
    command = [
        "docker",
        "run",
        "--rm",
        "--detach",
        "--gpus",
        "all",
        "--name",
        container_name,
    ]
    for port in (rpc_port, rpc_port + 1, rpc_port + 2):
        command.extend(["-p", f"{port}:{port}"])
    command.extend(
        [
            image,
            "/bin/bash",
            "-lc",
            f"./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port={rpc_port}",
        ]
    )
    return command


def check_carla_runtime_ports(
    ports: tuple[int, ...] = CARLA_PORTS,
    *,
    host: str | None = None,
) -> list[int]:
    """Return CARLA host ports that cannot be bound before container startup."""

    bind_host = CARLA_DEFAULT_HOST if host is None else host
    unavailable: list[int] = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError:
                unavailable.append(port)
    return unavailable


def run_carla_docker_preflight(  # noqa: C901
    *,
    image: str = CARLA_DOCKER_IMAGE,
    runner: CommandRunner = run_command,
    pull: bool = False,
    require_carla_api: bool = True,
) -> dict[str, Any]:
    """Check pinned CARLA Docker prerequisites without starting replay semantics.

    Returns:
        Machine-readable availability status and evidence for each preflight check.
    """

    image = validate_carla_image(image)
    status: dict[str, Any] = {
        "schema_version": CARLA_DOCKER_RUNTIME_SCHEMA_VERSION,
        "status": "available",
        "reason": "CARLA Docker runtime prerequisites are available",
        "image": image,
        "python_api_requirement": CARLA_PYTHON_API_REQUIREMENT,
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "ports": list(CARLA_PORTS),
        "checks": [],
    }

    docker_version = runner(["docker", "version", "--format", "{{json .}}"], timeout_s=10.0)
    docker_payload = _parse_json(docker_version.stdout)
    if (
        docker_version.returncode != 0
        or not isinstance(docker_payload, dict)
        or "Server" not in docker_payload
        or docker_payload["Server"] is None
    ):
        return _not_available(
            status,
            check={
                "name": "docker_daemon",
                "status": "not-available",
                "command": docker_version.args,
                "stdout": docker_version.stdout,
                "stderr": docker_version.stderr,
            },
            missing_capability="docker-daemon",
            reason="Docker daemon is not reachable without sudo",
        )
    status["checks"].append({"name": "docker_daemon", "status": "available"})
    status["docker"] = docker_payload

    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        return _not_available(
            status,
            check={"name": "host_platform", "status": "not-available"},
            missing_capability="linux-x86_64-host",
            reason="CARLA Docker runtime success path requires Linux x86_64",
        )
    status["checks"].append({"name": "host_platform", "status": "available"})

    nvidia_smi = runner(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        timeout_s=10.0,
    )
    if nvidia_smi.returncode != 0:
        return _not_available(
            status,
            check={
                "name": "nvidia_gpu",
                "status": "not-available",
                "command": nvidia_smi.args,
                "stderr": nvidia_smi.stderr,
            },
            missing_capability="nvidia-gpu",
            reason="NVIDIA GPU is not visible on the host",
        )
    status["checks"].append(
        {"name": "nvidia_gpu", "status": "available", "summary": nvidia_smi.stdout.strip()}
    )

    nvidia_docker = runner(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--gpus",
            "all",
            NVIDIA_DOCKER_TEST_IMAGE,
            "nvidia-smi",
        ],
        timeout_s=120.0,
    )
    if nvidia_docker.returncode != 0:
        return _not_available(
            status,
            check={
                "name": "nvidia_container_toolkit",
                "status": "not-available",
                "command": nvidia_docker.args,
                "stderr": nvidia_docker.stderr,
            },
            missing_capability="nvidia-container-toolkit",
            reason="NVIDIA Container Toolkit is not available through Docker",
        )
    status["checks"].append({"name": "nvidia_container_toolkit", "status": "available"})

    blocked_ports = check_carla_runtime_ports(CARLA_PORTS)
    if blocked_ports:
        return _not_available(
            status,
            check={"name": "ports", "status": "not-available", "blocked": blocked_ports},
            missing_capability="carla-ports",
            reason=f"CARLA ports are not free: {blocked_ports}",
        )
    status["checks"].append({"name": "ports", "status": "available"})

    image_inspect = runner(["docker", "image", "inspect", image], timeout_s=30.0)
    if image_inspect.returncode != 0:
        storage = _docker_storage_status(runner)
        status["docker_storage"] = storage
        if storage.get("free_gib", 0.0) < CARLA_PULL_MIN_FREE_GIB:
            return _not_available(
                status,
                check={"name": "docker_storage", "status": "not-available", **storage},
                missing_capability="docker-storage",
                reason=f"At least {CARLA_PULL_MIN_FREE_GIB} GiB free is required before pulling {image}",
            )
        if storage.get("free_gib", 0.0) < CARLA_PULL_WARN_FREE_GIB:
            status.setdefault("warnings", []).append(
                f"Docker storage has less than {CARLA_PULL_WARN_FREE_GIB} GiB free; CARLA "
                "image layers, extraction, logs, and runtime artifacts can consume substantial space."
            )
        if not pull:
            return _not_available(
                status,
                check={"name": "carla_image", "status": "not-available", "image": image},
                missing_capability="carla-image",
                reason=f"{image} is not present locally; rerun with --pull to fetch the pinned image",
            )
        pull_result = runner(["docker", "pull", image], timeout_s=3600.0)
        if pull_result.returncode != 0:
            return _not_available(
                status,
                check={
                    "name": "carla_image_pull",
                    "status": "failed",
                    "command": pull_result.args,
                    "stderr": pull_result.stderr,
                },
                missing_capability="carla-image",
                reason=f"Failed to pull pinned CARLA image {image}",
            )
        image_inspect = runner(["docker", "image", "inspect", image], timeout_s=30.0)
    if image_inspect.returncode != 0:
        return _not_available(
            status,
            check={
                "name": "carla_image_inspect",
                "status": "failed",
                "command": image_inspect.args,
                "stderr": image_inspect.stderr,
            },
            missing_capability="carla-image",
            reason=f"Failed to inspect image metadata for {image}",
        )
    image_metadata = _image_metadata(image_inspect.stdout)
    if not image_metadata.get("digest") or image_metadata.get("size_bytes") is None:
        return _not_available(
            status,
            check={"name": "carla_image", "status": "failed", "image": image},
            missing_capability="carla-image",
            reason=f"Image metadata for {image} is missing digest or size information",
        )
    status["image_digest"] = image_metadata.get("digest")
    status["image_size_bytes"] = image_metadata.get("size_bytes")
    status["checks"].append({"name": "carla_image", "status": "available", **image_metadata})

    if require_carla_api:
        carla_status = check_carla_availability()
        if carla_status["status"] != "available":
            return _not_available(
                status,
                check={"name": "carla_python_api", **carla_status},
                missing_capability="carla-python-api",
                reason=f"{CARLA_PYTHON_API_REQUIREMENT} is not importable from the repo command path",
            )
        status["checks"].append({"name": "carla_python_api", "status": "available"})

    return status


def build_carla_client_health_summary(
    *,
    host: str = CARLA_DEFAULT_HOST,
    port: int = CARLA_DEFAULT_RPC_PORT,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Connect to a live CARLA server and return client/server version and map metadata.

    Returns:
        Client connectivity summary with version and map metadata.
    """

    carla_module = require_carla()
    client = carla_module.Client(host, port)
    client.set_timeout(timeout_s)
    world = client.get_world()
    carla_map = world.get_map()
    return {
        "status": "connected",
        "host": host,
        "port": port,
        "client_version": client.get_client_version(),
        "server_version": client.get_server_version(),
        "map": getattr(carla_map, "name", None),
        "boundary": "client-connectivity-only",
    }


def run_carla_docker_runtime_smoke(
    *,
    image: str = CARLA_DOCKER_IMAGE,
    container_name: str = CARLA_DEFAULT_CONTAINER_NAME,
    runner: CommandRunner = run_command,
    pull: bool = False,
    startup_timeout_s: float = 120.0,
    retry_interval_s: float = 2.0,
    log_tail_lines: int = 80,
) -> dict[str, Any]:
    """Start the pinned CARLA server container, prove client connectivity, and clean up.

    Returns:
        Runtime status with preflight evidence, Docker command, health check, logs, and cleanup.
    """

    preflight = run_carla_docker_preflight(image=image, runner=runner, pull=pull)
    if preflight["status"] != "available":
        return preflight

    command = build_carla_server_container_command(image=image, container_name=container_name)
    start = runner(command, timeout_s=60.0)
    container_id = start.stdout.strip() or container_name
    started = start.returncode == 0
    summary: dict[str, Any] = {
        **preflight,
        "docker_command": command,
        "container": {"name": container_name, "id": container_id},
    }

    deadline = time.monotonic() + startup_timeout_s
    last_error: Exception | None = None
    try:
        if not started:
            summary.update(
                {
                    "status": "failed",
                    "reason": "Failed to start pinned CARLA server container",
                    "stderr": start.stderr,
                }
            )
            return summary
        while True:
            try:
                summary["health"] = build_carla_client_health_summary()
                summary["status"] = "connected"
                summary["reason"] = "Python client connected to the pinned CARLA Docker server"
                return summary
            except Exception as exc:  # noqa: BLE001 - CARLA client may raise custom API errors.
                last_error = exc
                if time.monotonic() >= deadline:
                    summary["status"] = "failed"
                    summary["reason"] = str(exc)
                    return summary
                time.sleep(retry_interval_s)
    finally:
        logs = runner(
            ["docker", "logs", "--tail", str(log_tail_lines), container_id], timeout_s=30.0
        )
        summary["logs_tail"] = logs.stdout if logs.returncode == 0 else logs.stderr
        stop = runner(["docker", "stop", container_id], timeout_s=30.0)
        summary["cleanup"] = {
            "command": stop.args,
            "returncode": stop.returncode,
            "stdout": stop.stdout,
            "stderr": stop.stderr,
        }
        if last_error is not None and summary.get("status") != "connected":
            summary["reason"] = str(last_error)


def run_carla_docker_live_replay(  # noqa: PLR0913
    manifest_path: str | Path,
    *,
    scenario_id: str | None = None,
    image: str = CARLA_DOCKER_IMAGE,
    container_name: str = CARLA_DEFAULT_CONTAINER_NAME,
    runner: CommandRunner = run_command,
    pull: bool = False,
    startup_timeout_s: float = 120.0,
    retry_interval_s: float = 2.0,
    log_tail_lines: int = 80,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Start pinned CARLA Docker runtime, run one T1 live replay, and clean up.

    Returns:
        Replay status with preflight, Docker lifecycle, client health, replay, logs, and cleanup.
    """

    preflight = run_carla_docker_preflight(image=image, runner=runner, pull=pull)
    if preflight["status"] != "available":
        return {**preflight, "stage": "live-replay-preflight"}

    command = build_carla_server_container_command(image=image, container_name=container_name)
    start = runner(command, timeout_s=60.0)
    container_id = start.stdout.strip() or container_name
    summary: dict[str, Any] = {
        **preflight,
        "stage": "live-replay",
        "docker_command": command,
        "container": {"name": container_name, "id": container_id},
    }

    last_error: Exception | None = None
    try:
        if start.returncode != 0:
            summary.update(
                {
                    "status": "failed",
                    "mode": "failed",
                    "reason": "Failed to start pinned CARLA server container",
                    "stderr": start.stderr,
                }
            )
            return summary

        deadline = time.monotonic() + startup_timeout_s
        while True:
            try:
                summary["health"] = build_carla_client_health_summary()
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - CARLA client may raise custom API errors.
                last_error = exc
                if time.monotonic() >= deadline:
                    summary.update(
                        {
                            "status": "failed",
                            "mode": "failed",
                            "reason": str(exc),
                        }
                    )
                    return summary
                time.sleep(retry_interval_s)

        replay = run_t1_oracle_live_replay_against_server(
            manifest_path,
            scenario_id=scenario_id,
            max_steps=max_steps,
        )
        summary["replay"] = replay
        summary["status"] = replay["status"]
        summary["mode"] = replay.get("mode", replay["status"])
        summary["reason"] = replay["reason"]
        return summary
    finally:
        logs = runner(
            ["docker", "logs", "--tail", str(log_tail_lines), container_id], timeout_s=30.0
        )
        summary["logs_tail"] = logs.stdout if logs.returncode == 0 else logs.stderr
        stop = runner(["docker", "stop", container_id], timeout_s=30.0)
        summary["cleanup"] = {
            "command": stop.args,
            "returncode": stop.returncode,
            "stdout": stop.stdout,
            "stderr": stop.stderr,
        }
        if last_error is not None and "replay" not in summary:
            summary["reason"] = str(last_error)


def _not_available(
    status: dict[str, Any],
    *,
    check: dict[str, Any],
    missing_capability: str,
    reason: str,
) -> dict[str, Any]:
    status["status"] = "not-available"
    status["reason"] = reason
    status["missing_capability"] = missing_capability
    status["checks"].append(check)
    return status


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _docker_storage_status(runner: CommandRunner) -> dict[str, Any]:
    result = runner(["docker", "info", "--format", "{{json .DockerRootDir}}"], timeout_s=10.0)
    docker_root = _parse_json(result.stdout) if result.returncode == 0 else None
    if not isinstance(docker_root, str) or not docker_root:
        docker_root = "/"
    usage = shutil.disk_usage(docker_root)
    total = usage.total if hasattr(usage, "total") else usage[0]
    free = usage.free if hasattr(usage, "free") else usage[2]
    return {
        "path": docker_root,
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
    }


def _image_metadata(inspect_stdout: str) -> dict[str, Any]:
    payload = _parse_json(inspect_stdout)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return {}
    repo_digests = payload.get("RepoDigests") or []
    digest = repo_digests[0].split("@", maxsplit=1)[-1] if repo_digests else payload.get("Id")
    return {
        "digest": digest,
        "size_bytes": payload.get("Size"),
    }
