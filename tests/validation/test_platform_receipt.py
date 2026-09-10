"""Focused contract tests for the sanitized platform receipt tool (#8913)."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import pytest


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
tool = _load("_platform_receipt", ROOT / "scripts/validation/platform_receipt.py")
comparator = _load(
    "_compare_execution_environments",
    ROOT / "scripts/validation/compare_execution_environments.py",
)
CLEARED = (
    "CARLA_VERSION CARLA_BUILD CARLA_COMMIT ROS_DISTRO ROS_VERSION RMW_IMPLEMENTATION "
    "CARLA_SERVER_BIN CARLA_SERVER_COMMAND CARLA_ROOT UE_ROOT UNREAL_ENGINE_ROOT "
    "CARLA_UNREAL_PROJECT CARLA_UNREAL_PLUGINS DISPLAY WAYLAND_DISPLAY OMP_NUM_THREADS "
    "ROS_MASTER_URI UNREAL_ENGINE_VERSION CUDA_VERSION"
).split()
GOOD_FIELDS = {
    "carla_server": {"version": "0.9.16"},
    "carla_python_api": {"version": "0.9.16"},
    "ros_environment": {"packages": {"rclpy": "present"}},
    "bridge": {"message_packages": {"carla_msgs": "present"}},
    "maps_assets": {"catalog_sha256": "a" * 64},
    "graphics_runtime": {"display_present": True, "egl_library_present": True},
}
EXPECTATIONS = {
    "schema_version": tool.EXPECTATIONS_SCHEMA,
    "required_ros_packages": ["rclpy"],
    "required_bridge_packages": ["carla_msgs"],
    "require_display": True,
    "headless_required": True,
    "expected_map_catalog_sha256": "a" * 64,
}


def _capture(monkeypatch, **env: str) -> dict[str, Any]:
    for name in CLEARED:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return tool.capture_receipt("host-test")


def _good_receipt(monkeypatch) -> dict[str, Any]:
    receipt = _capture(monkeypatch)
    for name, fields in GOOD_FIELDS.items():
        receipt[name] |= {"status": "observed", "unavailable_reasons": [], **fields}
    return receipt


def test_capture_absent_deterministic_sanitized_and_check_only(monkeypatch, tmp_path, capsys):
    receipt = _capture(monkeypatch)
    assert receipt["schema_version"] == tool.RECEIPT_SCHEMA
    assert receipt["carla_python_api"]["unavailable_reasons"] == ["carla_python_api_missing"]
    assert receipt["carla_server"]["unavailable_reasons"] == ["carla_server_not_configured"]
    assert "maps_catalog_not_configured" in receipt["maps_assets"]["unavailable_reasons"]
    assert receipt["startup_smoke"]["unavailable_reasons"] == ["smoke_not_requested"]
    assert tool.validate_receipt(receipt) == ()
    env = {"CARLA_UNREAL_PROJECT": "CarlaUE4", "UNREAL_ENGINE_VERSION": "/home/private-user/e"}
    first, second = (_capture(monkeypatch, **env) for _ in range(2))
    assert tool.render_json(first) == tool.render_json(second)
    assert "private-user" not in tool.render_json(first)
    ros = _capture(monkeypatch, ROS_DISTRO="humble", ROS_MASTER_URI="http://10.0.0.7:11311")
    assert ros["ros_environment"]["master_address"] == "<url>"
    assert "10.0.0.7" not in tool.render_json(ros)
    first["unreal_project"]["project_identity"] = "/home/private-user/CarlaUE4"
    assert "forbidden_value" in {issue.code for issue in tool.validate_receipt(first)}
    out = tmp_path / "receipt.json"
    code = tool.main(["capture", "--check-only", "--json", "--output", str(out)])
    assert code == 0 and json.loads(capsys.readouterr().out)["ok"] is True
    assert "carla_python_api_missing" in out.read_text(encoding="utf-8")


def test_check_good_bad_and_malformed_receipts(tmp_path, monkeypatch, capsys):
    receipt = _good_receipt(monkeypatch)
    assert tool.check_receipt(receipt, EXPECTATIONS) == ()
    path = tmp_path / "receipt.json"
    path.write_text(tool.render_json(receipt), encoding="utf-8")
    assert tool.main(["check", "--receipt", str(path), "--json"]) == 0
    receipt["carla_python_api"]["version"] = "0.9.15"
    path.write_text(tool.render_json(receipt), encoding="utf-8")
    assert tool.main(["check", "--receipt", str(path), "--json"]) == 2
    capsys.readouterr()
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert tool.main(["check", "--receipt", str(bad), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["issue_count"] > 0
    receipt["startup_smoke"]["status"] = "failed"
    assert "startup_smoke_failed" in {i.code for i in tool.check_receipt(receipt, EXPECTATIONS)}


BAD_CASES = [
    (lambda r: r["carla_python_api"].update(version="0.8.0"), "server_client_version_mismatch"),
    (lambda r: r["ros_environment"]["packages"].update(rclpy="missing"), "ros_package_missing"),
    (
        lambda r: r["bridge"]["message_packages"].update(carla_msgs="missing"),
        "bridge_package_missing",
    ),
    (lambda r: r["maps_assets"].update(catalog_sha256="b" * 64), "map_catalog_digest_mismatch"),
    (lambda r: r["graphics_runtime"].update(display_present=False), "display_unavailable"),
    (lambda r: r["graphics_runtime"].update(egl_library_present=False), "headless_egl_unavailable"),
]


@pytest.mark.parametrize(("mutate", "expected"), BAD_CASES)
def test_check_detects_incompatibilities(monkeypatch, mutate, expected):
    receipt = _good_receipt(monkeypatch)
    mutate(receipt)
    assert expected in {issue.code for issue in tool.check_receipt(receipt, EXPECTATIONS)}


def _server(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake_server.py"
    path.write_text(body, encoding="utf-8")
    return path


def _smoke(monkeypatch, script: Path, timeout: float = 2.0, *extra: str) -> int:
    monkeypatch.delenv("CARLA_SERVER_COMMAND", raising=False)
    monkeypatch.delenv("CARLA_SERVER_BIN", raising=False)
    command = shlex.join([sys.executable, str(script), *extra])
    args = ["startup-smoke", "--server-command", command, "--timeout-sec", str(timeout), "--json"]
    return tool.main(args)


def test_startup_smoke_success_reports_metadata_and_clean_teardown(tmp_path, monkeypatch, capsys):
    ready = {"carla_version": "0.9.16", "map_name": "Town01", "world_frame": 7}
    body = (
        "import json, time\n"
        f"print('PLATFORM_RECEIPT_READY ' + json.dumps({ready!r}), flush=True)\n"
        "time.sleep(30)\n"
    )
    assert _smoke(monkeypatch, _server(tmp_path, body)) == 0
    section = json.loads(capsys.readouterr().out)["startup_smoke"]
    assert (section["status"], section["map_name"], section["world_frame"]) == (
        "passed",
        "Town01",
        7,
    )
    assert section["terminated_cleanly"] is True and section["orphan_process_count"] == 0


@pytest.mark.parametrize(
    ("body", "timeout", "expected_reason"),
    [
        ("import time\ntime.sleep(60)\n", 1.0, "startup_timeout"),
        ("import sys\nsys.exit(3)\n", 2.0, "server_exit_nonzero"),
    ],
)
def test_startup_smoke_failures_are_bounded_and_clean(
    tmp_path, monkeypatch, capsys, body, timeout, expected_reason
):
    start = time.monotonic()
    assert _smoke(monkeypatch, _server(tmp_path, body), timeout) == 2
    section = json.loads(capsys.readouterr().out)["startup_smoke"]
    assert section["unavailable_reasons"] == [expected_reason]
    assert section["terminated_cleanly"] is True
    assert time.monotonic() - start < 10


def test_startup_smoke_missing_command_fails_with_carla_unavailable(monkeypatch, capsys):
    for name in ("CARLA_SERVER_COMMAND", "CARLA_SERVER_BIN"):
        monkeypatch.delenv(name, raising=False)
    assert tool.main(["startup-smoke", "--json"]) == 2
    section = json.loads(capsys.readouterr().out)["startup_smoke"]
    assert section["status"] == "unavailable"
    assert section["unavailable_reasons"] == ["carla_unavailable"]


def test_startup_smoke_cleans_up_spawned_grandchild(tmp_path, monkeypatch):
    pid_file = tmp_path / "child.pid"
    body = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    assert _smoke(monkeypatch, _server(tmp_path, body), 1.0) == 2
    pid = int(pid_file.read_text(encoding="utf-8"))
    for _ in range(60):
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(0.05)
    assert not Path(f"/proc/{pid}").exists()


def test_receipt_projects_into_comparator_manifest(monkeypatch):
    manifest = tool.to_comparator_manifest(_good_receipt(monkeypatch))
    assert comparator.parse_host_manifest(manifest)["host_label"] == "host-test"
    requirements = comparator.parse_requirements(
        {
            "schema_version": comparator.REQUIREMENTS_SCHEMA_VERSION,
            "workload_id": "platform-smoke",
            "not_applicable_fields": [],
            "material_fields": [],
            "compatible_variations": [],
        }
    )
    twin = tool.to_comparator_manifest(_good_receipt(monkeypatch))
    report = comparator.compare_environments(manifest, twin, requirements)
    assert report["comparison_status"] == "equivalent_for_workload"
