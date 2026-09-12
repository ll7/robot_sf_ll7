#!/usr/bin/env python3
"""Capture, validate, and smoke-test the CARLA/Unreal/ROS/bridge platform contract.

A receipt (``platform_receipt.v1``) is a deterministic, sanitized record of the host-side
execution environment for CARLA, Unreal Engine, ROS, bridge packages, maps, graphics/headless
capability, and one bounded startup smoke: ``capture`` observes, ``check`` validates against
optional expectations and exits nonzero on incompatibility, and ``startup-smoke`` launches only
an explicitly provided local server command, waits for one ``PLATFORM_RECEIPT_READY <json>``
stdout line inside a bounded timeout, records version/world/map metadata, and terminates the
whole process tree. Absent components stay explicit as ``unavailable`` with reason codes; no
value is fabricated. Receipts never contain private paths, identities, IPs, URLs, or credentials.
``to_comparator_manifest`` projects one receipt into the ``cross_host_environment.v1`` input of
``scripts/validation/compare_execution_environments.py``. Exit codes: 0 success, 2
compatibility/startup failure, 3 malformed input.
"""

from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import importlib
import importlib.util
import json
import os
import queue
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "platform_receipt.v1"
SMOKE_SCHEMA = "platform_receipt.startup_smoke.v1"
CHECK_SCHEMA = "platform_receipt.check.v1"
EXPECTATIONS_SCHEMA = "platform_receipt.expectations.v1"
COMPARATOR_SCHEMA = "cross_host_environment.v1"
READY_PREFIX = "PLATFORM_RECEIPT_READY "
SECTION_NAMES = (
    "carla_server carla_python_api unreal_project ros_environment bridge maps_assets "
    "graphics_runtime startup_smoke".split()
)
SECTION_STATUSES = ("observed", "unavailable", "not_run", "passed", "failed")
CLAIM_BOUNDARY = (
    "Platform capability observation only; observed availability and a bounded startup smoke "
    "do not establish simulator fidelity, benchmark eligibility, or cross-hardware "
    "reproducibility."
)
MIN_SMOKE_TIMEOUT_S, MAX_SMOKE_TIMEOUT_S = 1.0, 300.0
PROCESS_GRACE_S = 0.5
SMOKE_POLL_S = 0.1
HOST_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAP_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
CARLA_API_SYMBOLS = ("Client", "World", "Map", "Actor", "VehicleControl", "Location")
ROS_PACKAGE_PROBES = ("rclpy",)
ROS_MESSAGE_PROBES = ("std_msgs", "sensor_msgs", "nav_msgs", "geometry_msgs")
BRIDGE_ROLE_MODULES = ("availability", "live_replay", "replay_smoke", "parity", "export")
BRIDGE_MESSAGE_PROBES = ("carla_msgs", "rosgraph_msgs")
ENV_VALUE_ALLOWLIST = (
    "ROS_DISTRO ROS_VERSION RMW_IMPLEMENTATION CARLA_VERSION".split()
    + "CARLA_BUILD CARLA_COMMIT CUDA_VERSION OMP_NUM_THREADS".split()
)
ENV_PATH_ALLOWLIST = (
    "CARLA_SERVER_BIN CARLA_SERVER_COMMAND CARLA_ROOT".split()
    + "UE_ROOT UNREAL_ENGINE_ROOT CARLA_UNREAL_PROJECT".split()
)
ENV_FLAG_ALLOWLIST = ("DISPLAY", "WAYLAND_DISPLAY")
ENVIRONMENT_ALLOWLIST = (*ENV_VALUE_ALLOWLIST, *ENV_PATH_ALLOWLIST, *ENV_FLAG_ALLOWLIST)
_SECRET_RE = re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|token|credential|bearer)")
_HOME_RE = re.compile(r"/(?:home|Users)/[^/\s\"']+")
_WIN_HOME_RE = re.compile(r"(?i)[a-z]:\\users\\[^\\\s\"']+")
_USER_AT_HOST_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b")
_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_ABS_PATH_RE = re.compile(r"(?<![\w<>])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){3,}[0-9a-fA-F]{0,4}\b|\b[0-9a-fA-F:]*::[0-9a-fA-F:]*\b"
)


@dataclass(frozen=True, slots=True)
class Issue:
    """One sanitized fail-closed issue, free of private values."""

    code: str
    location: str
    message: str


class ReceiptError(ValueError):
    """Fail-closed capture, validation, or smoke failure carrying sanitized issues."""

    def __init__(self, issues: Sequence[Issue]) -> None:
        """Initialize the error with the sanitized issue sequence."""

        super().__init__("platform receipt failed closed")
        self.issues: tuple[Issue, ...] = tuple(issues)


def _issue(code: str, location: str, message: str) -> Issue:
    return Issue(code=code, location=location, message=message)


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_documentation_commands as _SUPERVISOR  # noqa: E402 - sibling supervisor helpers


def sanitize_text(text: str) -> str:
    """Replace credentials, private paths, identities, URLs, and IPs with markers."""
    if _SECRET_RE.search(text):
        return "<redacted>"
    text = _HOME_RE.sub("<home>", _WIN_HOME_RE.sub("<home>", text))
    text = _USER_AT_HOST_RE.sub("<user>@<host>", text)
    text = _URL_RE.sub("<url>", text)
    return _IPV6_RE.sub("<ip>", _IPV4_RE.sub("<ip>", _ABS_PATH_RE.sub("<path>", text)))


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    return [_sanitize_value(item) for item in value] if isinstance(value, list) else value


def _scan_forbidden(value: Any, location: str, issues: list[Issue]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _scan_forbidden(item, f"{location}/{key}", issues)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_forbidden(item, f"{location}[{index}]", issues)
    elif isinstance(value, str) and sanitize_text(value) != value:
        issues.append(_issue("forbidden_value", location, "private or credential-like content"))


def _normalize_section(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError([_issue("invalid_section", "", "section mapping required")])
    section = _sanitize_value(dict(value))
    reasons = section.get("unavailable_reasons", [])
    if section.get("status") not in SECTION_STATUSES:
        raise ReceiptError([_issue("invalid_section_status", "", "unknown section status")])
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise ReceiptError([_issue("invalid_reasons", "", "reason list required")])
    section["unavailable_reasons"] = sorted(set(reasons))
    return section


def validate_receipt(receipt: Any) -> tuple[Issue, ...]:
    """Validate receipt shape and reject any residual private value."""
    if not isinstance(receipt, Mapping):
        return (_issue("malformed_receipt", "/", "JSON object required"),)
    issues: list[Issue] = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        issues.append(_issue("schema_mismatch", "/schema_version", RECEIPT_SCHEMA))
    label = receipt.get("host_label")
    if not (isinstance(label, str) and HOST_LABEL_RE.fullmatch(label)):
        issues.append(_issue("invalid_host_label", "/host_label", "pseudonym required"))
    for name in SECTION_NAMES:
        section = receipt.get(name)
        if not isinstance(section, Mapping):
            issues.append(_issue("missing_section", f"/{name}", "mapping required"))
            continue
        if section.get("status") not in SECTION_STATUSES:
            issues.append(_issue("invalid_section_status", f"/{name}/status", "unknown status"))
        reasons = section.get("unavailable_reasons", [])
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            issues.append(_issue("invalid_reasons", f"/{name}/unavailable_reasons", "list"))
    _scan_forbidden(receipt, "", issues)
    return tuple(issues)


def _unavailable(*reasons: str) -> dict[str, Any]:
    return {"status": "unavailable", "unavailable_reasons": list(reasons)}


def _observed(reasons: Sequence[str] = (), **fields: Any) -> dict[str, Any]:
    return {"status": "observed", "unavailable_reasons": list(reasons), **fields}


def _module_presence(name: str) -> str:
    try:
        return "present" if importlib.util.find_spec(name) is not None else "missing"
    except (ImportError, ValueError):
        return "missing"


def _probe_carla_python_api() -> dict[str, Any]:
    if _module_presence("carla") == "missing":
        return _unavailable("carla_python_api_missing")
    try:
        module = importlib.import_module("carla")
    except (ImportError, OSError):
        return _unavailable("carla_python_api_import_error")
    version = getattr(module, "__version__", None)
    recorded = isinstance(version, str) and bool(version)
    symbols = [name for name in CARLA_API_SYMBOLS if hasattr(module, name)]
    return _observed(
        [] if recorded else ["carla_python_api_version_unrecorded"],
        version=version if recorded else "unavailable",
        api_symbols=symbols,
    )


def _probe_carla_server() -> dict[str, Any]:
    fields = {
        "version": os.environ.get("CARLA_VERSION"),
        "build": os.environ.get("CARLA_BUILD"),
        "commit": os.environ.get("CARLA_COMMIT"),
    }
    configured = any(os.environ.get(name) for name in ("CARLA_SERVER_BIN", "CARLA_SERVER_COMMAND"))
    if not configured and not any(fields.values()):
        return _unavailable("carla_server_not_configured")
    reasons = [f"carla_server_{key}_unrecorded" for key, value in fields.items() if not value]
    recorded = {key: value or "unavailable" for key, value in fields.items()}
    return _observed(reasons, command_configured=configured, **recorded)


def _probe_unreal_project() -> dict[str, Any]:
    project = os.environ.get("CARLA_UNREAL_PROJECT")
    engine = os.environ.get("UE_ROOT") or os.environ.get("UNREAL_ENGINE_ROOT")
    if not project and not engine:
        return _unavailable("unreal_project_not_configured")
    plugins = (os.environ.get("CARLA_UNREAL_PLUGINS") or "").split(",")
    return _observed(
        [] if engine else ["unreal_engine_root_not_configured"],
        engine_root_configured=bool(engine),
        engine_version=os.environ.get("UNREAL_ENGINE_VERSION") or "unavailable",
        project_identity=Path(project).name if project else "unavailable",
        plugins=sorted(item.strip() for item in plugins if item.strip()) or "unavailable",
    )


def _probe_ros_environment() -> dict[str, Any]:
    if not os.environ.get("ROS_DISTRO"):
        return _unavailable("ros_environment_not_detected")
    return _observed(
        distro=os.environ["ROS_DISTRO"],
        ros_version=os.environ.get("ROS_VERSION") or "unavailable",
        rmw_implementation=os.environ.get("RMW_IMPLEMENTATION") or "unavailable",
        master_address="<url>" if os.environ.get("ROS_MASTER_URI") else "unavailable",
        packages={name: _module_presence(name) for name in ROS_PACKAGE_PROBES},
        message_packages={name: _module_presence(name) for name in ROS_MESSAGE_PROBES},
    )


def _probe_bridge() -> dict[str, Any]:
    if _module_presence("robot_sf_carla_bridge") == "missing":
        return _unavailable("bridge_package_missing")
    ready = _module_presence("carla") == "present"
    roles = {
        name: _module_presence(f"robot_sf_carla_bridge.{name}") for name in BRIDGE_ROLE_MODULES
    }
    return _observed(
        [] if ready else ["carla_python_api_missing"],
        runtime_ready=ready,
        modules=roles,
        message_packages={name: _module_presence(name) for name in BRIDGE_MESSAGE_PROBES},
    )


def _probe_maps_assets(maps_dir: Path | None) -> dict[str, Any]:
    if maps_dir is None:
        return _unavailable("maps_catalog_not_configured")
    if not maps_dir.is_dir():
        return _unavailable("maps_catalog_unreadable")
    digest, count = hashlib.sha256(), 0
    for path in sorted(maps_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(maps_dir).as_posix()
        try:
            content = path.read_bytes()
        except OSError:
            return _unavailable("maps_catalog_unreadable")
        digest.update(f"{relative}\t{hashlib.sha256(content).hexdigest()}\n".encode())
        count += 1
    return _observed(catalog_sha256=digest.hexdigest(), asset_count=count)


def _gpu_fields() -> dict[str, Any]:
    empty = dict.fromkeys(("gpu_query", "gpu_name"), "unavailable")
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return empty
    query = [executable, "--query-gpu=name", "--format=csv,noheader"]
    try:
        result = subprocess.run(query, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return empty
    fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    if result.returncode != 0 or len(fields) < 1:
        return empty
    return {"gpu_query": "observed", "gpu_name": fields[0]}


def _probe_graphics_runtime() -> dict[str, Any]:
    display = bool(os.environ.get("DISPLAY"))
    wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    egl = ctypes.util.find_library("EGL") is not None
    fields = _gpu_fields()
    reasons = [] if display or wayland else ["display_unavailable"]
    reasons += [] if egl else ["egl_unavailable"]
    reasons += ["gpu_query_unavailable"] if fields["gpu_query"] == "unavailable" else []
    return _observed(
        reasons,
        display_present=display,
        wayland_present=wayland,
        egl_library_present=egl,
        headless_ready=egl,
        **fields,
    )


def _default_probes(maps_dir: Path | None = None) -> dict[str, Callable[[], dict[str, Any]]]:
    return {
        "carla_server": _probe_carla_server,
        "carla_python_api": _probe_carla_python_api,
        "unreal_project": _probe_unreal_project,
        "ros_environment": _probe_ros_environment,
        "bridge": _probe_bridge,
        "maps_assets": lambda: _probe_maps_assets(maps_dir),
        "graphics_runtime": _probe_graphics_runtime,
    }


def _smoke_base(command: Sequence[str] | None, timeout_sec: float) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "unavailable_reasons": ["carla_unavailable"],
        "command_identity": Path(command[0]).name if command else "unavailable",
        "timeout_sec": timeout_sec if command else "unavailable",
        "terminated_cleanly": "not_observed",
        "orphan_process_count": "not_observed",
        "server_exit_code": "not_observed",
        "carla_version": "unavailable",
        "map_name": "unavailable",
        "world_frame": "unavailable",
    }


def _not_run_smoke() -> dict[str, Any]:
    return _smoke_base(None, 0.0) | {
        "status": "not_run",
        "unavailable_reasons": ["smoke_not_requested"],
    }


def capture_receipt(
    host_label: str, probes: Mapping[str, Callable[[], dict[str, Any]]] | None = None
) -> dict[str, Any]:
    """Observe the platform through ``probes`` (default: host probes) and return a receipt."""
    probe_map = probes if probes is not None else _default_probes()
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "host_label": host_label,
        "claim_boundary": CLAIM_BOUNDARY,
        "environment_allowlist": list(ENVIRONMENT_ALLOWLIST),
    }
    for name in SECTION_NAMES:
        section = _not_run_smoke() if name == "startup_smoke" else probe_map[name]()
        receipt[name] = _normalize_section(section)
    issues = validate_receipt(receipt)
    if issues:
        raise ReceiptError(issues)
    return receipt


def to_comparator_manifest(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Project one receipt into the comparator's ``cross_host_environment.v1`` input."""
    manifest = {"schema_version": COMPARATOR_SCHEMA, "host_label": receipt["host_label"]}
    for name in (*SECTION_NAMES, "environment_allowlist"):
        if isinstance(receipt.get(name), Mapping):
            manifest[name] = dict(receipt[name])
    return manifest


def _check_expectations(  # noqa: C901 - one expectation pass over all sections
    receipt: Mapping[str, Any], expectations: Any, issues: list[Issue]
) -> None:
    if expectations is None:
        return
    if (
        not isinstance(expectations, Mapping)
        or expectations.get("schema_version") != EXPECTATIONS_SCHEMA
    ):
        issues.append(_issue("invalid_expectations", "/expectations", EXPECTATIONS_SCHEMA))
        return
    pairs = (
        ("required_ros_packages", "ros_environment", "packages", "ros_package_missing"),
        ("required_bridge_packages", "bridge", "message_packages", "bridge_package_missing"),
    )
    for attr, section, leaf, code in pairs:
        required = expectations.get(attr)
        if required is None:
            continue
        if not isinstance(required, list):
            issues.append(_issue("invalid_expectations", f"/{section}/{leaf}", "package list"))
            continue
        packages = receipt[section].get(leaf) or {}
        for name in required:
            if packages.get(name) != "present":
                issues.append(_issue(code, f"/{section}/{leaf}/{name}", "package missing"))
    graphics = receipt["graphics_runtime"]
    if expectations.get("require_display") and graphics.get("display_present") is not True:
        issues.append(_issue("display_unavailable", "/graphics_runtime", "display required"))
    if expectations.get("headless_required") and graphics.get("egl_library_present") is not True:
        issues.append(_issue("headless_egl_unavailable", "/graphics_runtime", "EGL required"))
    expected = expectations.get("expected_map_catalog_sha256")
    if expected is not None:
        digest = receipt["maps_assets"].get("catalog_sha256")
        if not isinstance(expected, str):
            issues.append(_issue("invalid_expectations", "/expectations", "digest string required"))
        elif digest != expected:
            issues.append(_issue("map_catalog_digest_mismatch", "/maps_assets", "digest differs"))


def check_receipt(receipt: Any, expectations: Any = None) -> tuple[Issue, ...]:
    """Validate one receipt against internal compatibility and optional expectations."""
    issues = list(validate_receipt(receipt))
    if issues:
        return tuple(issues)
    server_version = receipt["carla_server"].get("version")
    api_version = receipt["carla_python_api"].get("version")
    if (
        isinstance(server_version, str)
        and re.match(r"^\d+\.\d+", server_version)
        and isinstance(api_version, str)
        and server_version != api_version
    ):
        issues.append(_issue("server_client_version_mismatch", "/carla_server/version", "differs"))
    if receipt["startup_smoke"].get("status") == "failed":
        issues.append(_issue("startup_smoke_failed", "/startup_smoke", "smoke failed"))
    _check_expectations(receipt, expectations, issues)
    return tuple(sorted(issues, key=lambda item: (item.code, item.location)))


def _parse_ready(line: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(line[len(READY_PREFIX) :].strip())
    except ValueError:
        return None, "ready_metadata_invalid"
    if not isinstance(payload, Mapping):
        return None, "ready_metadata_invalid"
    version, map_name = payload.get("carla_version"), payload.get("map_name")
    version_ok = isinstance(version, str) and re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version)
    map_ok = isinstance(map_name, str) and MAP_NAME_RE.fullmatch(map_name)
    if not (version_ok and map_ok):
        return None, "ready_metadata_invalid"
    if sanitize_text(line) != line:
        return None, "ready_metadata_untrusted"
    frame = payload.get("world_frame")
    world_frame = frame if isinstance(frame, int) and not isinstance(frame, bool) else "unavailable"
    return {"carla_version": version, "map_name": map_name, "world_frame": world_frame}, None


def _await_ready(
    process: subprocess.Popen[bytes], deadline: float, descendants: set[Any]
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    lines: queue.Queue[bytes | None] = queue.Queue()

    def _reader() -> None:
        for line in process.stdout or ():
            lines.put(line)
        lines.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    while time.monotonic() < deadline:
        descendants.update(_SUPERVISOR._linux_process_descendants(process.pid))
        try:
            line = lines.get(timeout=SMOKE_POLL_S)
        except queue.Empty:
            continue
        if line is None:
            return None, "server_exit_without_ready", process.poll()
        if line.startswith(READY_PREFIX.encode()):
            metadata, reason = _parse_ready(line.decode("utf-8", "replace"))
            return metadata, reason, process.returncode
    return None, "startup_timeout", process.returncode


def run_startup_smoke(server_command: Sequence[str] | None, timeout_sec: float) -> dict[str, Any]:
    """Launch one bounded local server command and return the ``startup_smoke`` section."""
    if not (MIN_SMOKE_TIMEOUT_S <= timeout_sec <= MAX_SMOKE_TIMEOUT_S):
        raise ReceiptError([_issue("invalid_timeout", "/timeout_sec", "1..300 seconds")])
    command = (
        list(server_command)
        if server_command
        else shlex.split(os.environ.get("CARLA_SERVER_COMMAND", ""))
    )
    if not command and os.environ.get("CARLA_SERVER_BIN"):
        command = [os.environ["CARLA_SERVER_BIN"]]
    section = _smoke_base(command, timeout_sec)
    if not command:
        return section
    executable = shutil.which(command[0])
    if executable is None:
        return section
    process = subprocess.Popen(
        [executable, *command[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    descendants: set[Any] = set()
    metadata, reason, exit_code = _await_ready(process, time.monotonic() + timeout_sec, descendants)
    _SUPERVISOR._terminate_process_group(process, descendants)
    orphans = sum(1 for item in descendants if _SUPERVISOR._linux_process_identity_exists(item))
    orphans += 1 if process.poll() is None else 0
    section["terminated_cleanly"] = orphans == 0 and not _SUPERVISOR._process_group_exists(
        process.pid
    )
    section["orphan_process_count"] = orphans
    section["server_exit_code"] = exit_code if exit_code is not None else "unavailable"
    if metadata is not None:
        section |= {"status": "passed", "unavailable_reasons": [], **metadata}
        section["server_exit_code"] = "not_applicable"
        return section
    if reason == "server_exit_without_ready" and isinstance(exit_code, int) and exit_code != 0:
        reason = "server_exit_nonzero"
    section.update({"status": "failed", "unavailable_reasons": [reason or "startup_failed"]})
    return section


def render_json(payload: Mapping[str, Any]) -> str:
    """Return byte-stable sorted-key JSON with a trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_markdown(receipt: Mapping[str, Any]) -> str:
    """Return a concise deterministic Markdown summary of one receipt."""
    lines = [f"- `{name}`: `{receipt[name].get('status')}`" for name in SECTION_NAMES]
    return "\n".join(["# Platform Receipt", *lines, "", CLAIM_BOUNDARY, ""])


def _load_object(path: Path, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReceiptError([_issue(code, path.name, type(exc).__name__)]) from exc
    if not isinstance(payload, Mapping):
        raise ReceiptError([_issue(code, path.name, "object required")])
    return dict(payload)


def _report_check(issues: Sequence[Issue], as_json: bool) -> None:
    report = {
        "schema_version": CHECK_SCHEMA,
        "ok": not issues,
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }
    if as_json:
        sys.stdout.write(render_json(report))
    elif not issues:
        sys.stdout.write("OK: platform receipt is compatible\n")
    for issue in issues:
        sys.stderr.write(f"FAIL {issue.code} at {issue.location}: {issue.message}\n")


def _run_capture(args: argparse.Namespace) -> int:
    probes = _default_probes(args.maps_dir) if args.maps_dir else None
    label = "host-" + hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12]
    receipt = capture_receipt(label, probes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_json(receipt), encoding="utf-8")
    if args.comparator_host_out:
        args.comparator_host_out.parent.mkdir(parents=True, exist_ok=True)
        manifest = render_json(to_comparator_manifest(receipt))
        args.comparator_host_out.write_text(manifest, encoding="utf-8")
    if args.check_only:
        expectations = (
            _load_object(args.expectations, "unreadable_expectations")
            if args.expectations
            else None
        )
        issues = check_receipt(receipt, expectations)
        _report_check(issues, args.json)
        return 0 if not issues else 2
    sys.stdout.write(render_json(receipt) if args.json else render_markdown(receipt))
    return 0


def _run_check(args: argparse.Namespace) -> int:
    expectations = (
        _load_object(args.expectations, "unreadable_expectations") if args.expectations else None
    )
    issues = check_receipt(_load_object(args.receipt, "unreadable_receipt"), expectations)
    _report_check(issues, args.json)
    return 0 if not issues else 2


def _run_smoke(args: argparse.Namespace) -> int:
    command = shlex.split(args.server_command) if args.server_command else None
    section = run_startup_smoke(command, args.timeout_sec)
    report = {
        "schema_version": SMOKE_SCHEMA,
        "startup_smoke": section,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if args.json:
        sys.stdout.write(render_json(report))
    else:
        sys.stdout.write(f"startup-smoke: {section['status']} {section['unavailable_reasons']}\n")
    return 0 if section["status"] == "passed" else 2


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the platform receipt argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_subparsers(dest="command", required=True)
    capture = actions.add_parser("capture", help="Observe this host and write a receipt.")
    capture.add_argument("--output", type=Path)
    capture.add_argument("--comparator-host-out", type=Path)
    capture.add_argument("--expectations", type=Path)
    capture.add_argument("--maps-dir", type=Path)
    capture.add_argument("--check-only", action="store_true")
    capture.add_argument("--json", action="store_true")
    check = actions.add_parser("check", help="Validate a receipt and expectations.")
    check.add_argument("--receipt", required=True, type=Path)
    check.add_argument("--expectations", type=Path)
    check.add_argument("--json", action="store_true")
    smoke = actions.add_parser("startup-smoke", help="Run one bounded local startup smoke.")
    smoke.add_argument("--server-command")
    smoke.add_argument("--timeout-sec", type=float, default=60.0)
    smoke.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested subcommand and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "capture":
            return _run_capture(args)
        if args.command == "check":
            return _run_check(args)
        return _run_smoke(args)
    except ReceiptError as error:
        for issue in error.issues:
            sys.stderr.write(f"FAIL {issue.code} at {issue.location}: {issue.message}\n")
        return 3


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
