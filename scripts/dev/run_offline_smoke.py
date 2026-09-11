#!/usr/bin/env python3
"""Run a bounded, no-cluster smoke of the core Robot SF local workflow.

This is smoke evidence only, not a benchmark.  The harness uses a fresh
task-owned root, an allowlisted environment, a controlled ``PATH``, and an
in-process plus child-process network guard.  Optional institutional features
are reported as unavailable instead of being inferred from the host.

The receipt contains no paths, timestamps, host metadata, exception text, or
real-episode metrics.  Stable unavailable reason codes are exposed through
``REASONS`` and in every optional capability row.

Example::

    uv run python scripts/dev/run_offline_smoke.py --offline --isolated --json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence


SCHEMA = "robot_sf.offline_smoke_receipt.v1"
MODE = "offline_isolated"
RECEIPT_NAME = "capability_receipt.json"
MARKER_NAME = ".robot-sf-offline-smoke-root"
MARKER_TEXT = '{"owner":"run_offline_smoke","schema":"robot_sf.offline_smoke_root.v1"}\n'
SOURCE_COMMAND = "uv run python scripts/dev/run_offline_smoke.py --offline --isolated --json"
CHILD_TIMEOUT_SECONDS = 45
STATUS_PASSED, STATUS_UNAVAILABLE, STATUS_FAILED = "passed", "unavailable", "failed"

REASONS = dict(
    zip(
        "ok package_import cli_help cli_discovery doctor config episode artifact lineage report examples models isolation path path_leak output_escape network_guard root_invalid root_not_owned root_not_fresh scheduler scheduler_present gpu gpu_not_disabled carla private_data private_exposed institutional institutional_exposed model model_empty model_not_empty network".split(),
        "OK PACKAGE_IMPORT_FAILED CLI_HELP_FAILED CLI_DISCOVERY_FAILED DOCTOR_FAILED CONFIG_RESOLUTION_FAILED HEADLESS_EPISODE_FAILED ARTIFACT_VERIFICATION_FAILED LINEAGE_VERIFICATION_FAILED REPORT_METRIC_FAILED EXAMPLE_DISCOVERY_FAILED MODEL_REGISTRY_FAILED ISOLATION_POLICY_FAILED PATH_CONTAINMENT_FAILED PATH_LEAKAGE_DETECTED OUTPUT_ESCAPE_DETECTED NETWORK_GUARD_FAILED OUTPUT_ROOT_INVALID OUTPUT_ROOT_NOT_OWNED OUTPUT_ROOT_NOT_FRESH SCHEDULER_UNAVAILABLE SCHEDULER_PRESENT_IN_ISOLATION GPU_UNAVAILABLE GPU_NOT_DISABLED CARLA_UNAVAILABLE PRIVATE_DATA_UNAVAILABLE PRIVATE_DATA_EXPOSED INSTITUTIONAL_CONTEXT_UNAVAILABLE INSTITUTIONAL_CONTEXT_EXPOSED MODEL_UNAVAILABLE MODEL_CACHE_EMPTY MODEL_CACHE_NOT_EMPTY NETWORK_DISABLED".split(),
        strict=True,
    )
)
CORE_CAPABILITIES = "package_import cli_help cli_discovery environment_doctor config_resolution headless_episode artifact_verification lineage_verification report_metric_fixture example_discovery model_registry isolation_policy path_containment network_isolation".split()
OPTIONAL_CAPABILITIES = (
    "scheduler gpu carla private_data institutional_context model_cache network".split()
)
ALL_CAPABILITIES = CORE_CAPABILITIES + OPTIONAL_CAPABILITIES
SCHEDULER_COMMANDS = "sbatch squeue sacct".split()
PRIVATE_ENV_KEYS = frozenset(
    "ROBOT_SF_PRIVATE_OPS ROBOT_SF_PRIVATE_DATA PRIVATE_OPS_ROOT INSTITUTIONAL_MOUNT WANDB_API_KEY HF_TOKEN AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY".split()
)
CLUSTER_ENV_PREFIXES = tuple("SLURM_ PBS_ LSB_ SGE_ CONDOR_ KUBERNETES_".split())
ROOT_RELATIVE = dict(
    zip(
        "HOME XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME TMPDIR UV_CACHE_DIR PIP_CACHE_DIR HF_HOME HF_DATASETS_CACHE TORCH_HOME WANDB_DIR MPLCONFIGDIR NUMBA_CACHE_DIR ROBOT_SF_ARTIFACT_ROOT".split(),
        "home cache config data tmp cache/uv cache/pip cache/huggingface cache/huggingface/datasets cache/torch cache/wandb cache/matplotlib cache/numba artifacts".split(),
        strict=True,
    )
)
PATH_MARKERS = ("private-ops", "private_ops", "local.machine.md", "wandb_api_key", "hf_token")
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|[A-Za-z]:[\\/]", re.IGNORECASE
)
_SITE_CUSTOMIZE = '''"""Deny network access for offline smoke children."""
import socket as _socket
_REAL_SOCKET = _socket.socket
class _OfflineSocket(_REAL_SOCKET):
    def _deny(self):
        raise OSError("offline smoke network disabled")
    def connect(self, address):
        self._deny()
    def connect_ex(self, address):
        self._deny()
    def sendto(self, data, *args):
        self._deny()
def _deny(*args, **kwargs):
    raise OSError("offline smoke network disabled")
_socket.socket = _OfflineSocket
_socket.create_connection = _deny
_socket.getaddrinfo = _deny
'''


@dataclass(frozen=True)
class Capability:
    """One deterministic capability receipt row."""

    capability_id: str
    required: bool
    status: str
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON row."""
        return {
            "id": self.capability_id,
            "required": self.required,
            "status": self.status,
            "reason_code": self.reason_code,
        }


class SmokeFailure(RuntimeError):
    """Internal failure carrying only a stable public reason code."""

    def __init__(self, reason_code: str) -> None:
        """Store the reason without exposing runtime details."""
        super().__init__(reason_code)
        self.reason_code = reason_code


def _compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _with_digest(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("receipt_digest", None)
    return {
        **unsigned,
        "receipt_digest": hashlib.sha256(_compact_json(unsigned).encode()).hexdigest(),
    }


def _fresh_root(repo_root: Path, requested: Path) -> Path:
    try:
        root = requested.expanduser().resolve()
        if root == repo_root or repo_root.is_relative_to(root):
            raise SmokeFailure(REASONS["root_invalid"])
        if root.exists() and not root.is_dir():
            raise SmokeFailure(REASONS["root_invalid"])
        root.mkdir(parents=True, exist_ok=True)
        entries = list(root.iterdir())
    except SmokeFailure:
        raise
    except (OSError, RuntimeError) as exc:
        raise SmokeFailure(REASONS["root_invalid"]) from exc
    marker = root / MARKER_NAME
    if entries and not marker.is_file():
        raise SmokeFailure(REASONS["root_not_owned"])
    if marker.is_file():
        try:
            valid = marker.read_text(encoding="utf-8") == MARKER_TEXT
        except OSError as exc:
            raise SmokeFailure(REASONS["root_not_owned"]) from exc
        if not valid:
            raise SmokeFailure(REASONS["root_not_owned"])
        if any(entry != marker for entry in entries):
            raise SmokeFailure(REASONS["root_not_fresh"])
    marker.write_text(MARKER_TEXT, encoding="utf-8")
    return root


def _prepare_root(repo_root: Path, requested: Path | None) -> tuple[Path, Any]:
    manager = None
    if requested is None:
        manager = tempfile.TemporaryDirectory(prefix="robot-sf-offline-smoke-")
        root = Path(manager.name).resolve()
        (root / MARKER_NAME).write_text(MARKER_TEXT, encoding="utf-8")
    else:
        root = _fresh_root(repo_root, requested)
    for relative in set(ROOT_RELATIVE.values()) | set(
        "cache/model python bin reports config data".split()
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root, manager


def _controlled_environment(root: Path, repo_root: Path) -> tuple[dict[str, str], dict[str, Path]]:
    bin_root = root / "bin"
    targets: dict[str, Path] = {}
    for name in ("git", "uv"):
        target_text = shutil.which(name)
        if target_text:
            targets[name] = Path(target_text).resolve()
            (bin_root / name).symlink_to(targets[name])
    python_root = root / "python"
    (python_root / "sitecustomize.py").write_text(_SITE_CUSTOMIZE, encoding="utf-8")
    paths = {key: root / relative for key, relative in ROOT_RELATIVE.items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    environment = {key: str(path) for key, path in paths.items()}
    environment.update(
        {
            "PATH": str(bin_root),
            "PYTHONPATH": os.pathsep.join((str(python_root), str(repo_root))),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "UV_OFFLINE": "1",
            "UV_NO_SYNC": "1",
            "UV_NO_CONFIG": "1",
            "MPLBACKEND": "Agg",
            "SDL_VIDEODRIVER": "dummy",
            "DISPLAY": "",
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "OMP_NUM_THREADS": "1",
        }
    )
    return environment, targets


@contextlib.contextmanager
def _temporary_environment(environment: Mapping[str, str]) -> Iterator[None]:
    original = os.environ.copy()
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _deny_socket(*args: Any, **kwargs: Any) -> None:
    raise OSError("offline smoke network disabled")


@contextlib.contextmanager
def _network_denied() -> Iterator[None]:
    original_socket, original_create, original_info = (
        socket.socket,
        socket.create_connection,
        socket.getaddrinfo,
    )

    class OfflineSocket(original_socket):  # type: ignore[misc, valid-type]
        connect = connect_ex = sendto = _deny_socket

    socket.socket, socket.create_connection, socket.getaddrinfo = (
        OfflineSocket,
        _deny_socket,
        _deny_socket,
    )  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket, socket.create_connection, socket.getaddrinfo = (
            original_socket,
            original_create,
            original_info,
        )  # type: ignore[assignment]


def _run_child(
    repo_root: Path,
    environment: Mapping[str, str],
    code: str,
    args: Sequence[str] = (),
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, "-c", code, *args],
            cwd=cwd or repo_root,
            env=dict(environment),
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmokeFailure(REASONS["isolation"]) from exc


def _run_cli(
    repo_root: Path, environment: Mapping[str, str], args: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    return _run_child(
        repo_root, environment, "from robot_sf.cli import main; raise SystemExit(main())", args
    )


def _require(condition: bool, reason_code: str) -> None:
    if not condition:
        raise SmokeFailure(reason_code)


def _stage(capability_id: str, callback: Callable[[], Any], reason_code: str) -> Capability:
    try:
        _require(callback() is not False, reason_code)
    except SmokeFailure as exc:
        return Capability(capability_id, True, STATUS_FAILED, exc.reason_code)
    except (OSError, RuntimeError, TypeError, ValueError, ImportError, AttributeError, LookupError):
        return Capability(capability_id, True, STATUS_FAILED, reason_code)
    return Capability(capability_id, True, STATUS_PASSED, REASONS["ok"])


def _probe_scheduler(path: str | None = None) -> Capability:
    path = os.environ.get("PATH", "") if path is None else path
    if any(shutil.which(command, path=path) for command in SCHEDULER_COMMANDS):
        return Capability("scheduler", False, STATUS_FAILED, REASONS["scheduler_present"])
    return Capability("scheduler", False, STATUS_UNAVAILABLE, REASONS["scheduler"])


def _probe_gpu(environment: Mapping[str, str] | None = None) -> Capability:
    values = os.environ if environment is None else environment
    if values.get("CUDA_VISIBLE_DEVICES") == "" and values.get("NVIDIA_VISIBLE_DEVICES") == "void":
        return Capability("gpu", False, STATUS_UNAVAILABLE, REASONS["gpu"])
    return Capability("gpu", False, STATUS_FAILED, REASONS["gpu_not_disabled"])


def _probe_model_cache(cache_root: Path, model_path: Path | None = None) -> Capability:
    if model_path is not None and not model_path.is_file():
        return Capability("model_cache", False, STATUS_UNAVAILABLE, REASONS["model"])
    if not any(cache_root.iterdir()):
        return Capability("model_cache", False, STATUS_UNAVAILABLE, REASONS["model_empty"])
    return Capability("model_cache", False, STATUS_FAILED, REASONS["model_not_empty"])


def _probe_private_data(environment: Mapping[str, str]) -> Capability:
    if any(key in PRIVATE_ENV_KEYS for key in environment):
        return Capability("private_data", False, STATUS_FAILED, REASONS["private_exposed"])
    return Capability("private_data", False, STATUS_UNAVAILABLE, REASONS["private_data"])


def _probe_institutional_context(environment: Mapping[str, str]) -> Capability:
    if any(key.startswith(CLUSTER_ENV_PREFIXES) for key in environment):
        return Capability(
            "institutional_context", False, STATUS_FAILED, REASONS["institutional_exposed"]
        )
    return Capability("institutional_context", False, STATUS_UNAVAILABLE, REASONS["institutional"])


def _package_import() -> bool:
    return bool(getattr(importlib.import_module("robot_sf"), "__file__", None))


def _cli_help(repo_root: Path, environment: Mapping[str, str]) -> bool:
    result = _run_cli(repo_root, environment, ("--help",))
    return result.returncode == 0 and "usage: robot-sf" in result.stdout


def _cli_discovery(repo_root: Path, environment: Mapping[str, str]) -> bool:
    result = _run_cli(repo_root, environment, ("envs", "list", "--format", "json"))
    _require(result.returncode == 0 and bool(json.loads(result.stdout)), REASONS["cli_discovery"])
    result = _run_cli(
        repo_root,
        environment,
        (
            "scenarios",
            "validate",
            "configs/scenarios/single/planner_sanity_simple.yaml",
            "--format",
            "json",
        ),
    )
    return result.returncode == 0 and json.loads(result.stdout).get("status") == "valid"


def _environment_doctor(repo_root: Path, root: Path) -> bool:
    doctor = importlib.import_module("robot_sf.benchmark.doctor")
    artifact_guard = importlib.import_module("scripts.tools.check_artifact_root")
    report = doctor.collect_doctor_report(
        artifact_root=root / "artifacts",
        run_env_smoke=False,
        run_quickstart_smoke=False,
        workspace_root=repo_root,
    )
    guard = artifact_guard.check_artifact_root(
        source_root=repo_root, artifact_root=root / "artifacts"
    )
    return doctor.doctor_exit_code(report) == 0 and guard.exit_code == 0


def _config_resolution(repo_root: Path) -> bool:
    path = repo_root / "configs/scenarios/single/planner_sanity_simple.yaml"
    runner = importlib.import_module("robot_sf.benchmark.runner")
    scenarios = importlib.import_module("robot_sf.cli_scenarios")
    return (
        len(runner.load_scenario_matrix(path)) == 1
        and scenarios.validate_scenario_payload(str(path)).get("status") == "valid"
    )


_EPISODE_CODE = """
from robot_sf.gym_env.environment_factory import make_robot_env
env = make_robot_env(debug=False, seed=0, recording_enabled=False, record_video=False, use_jsonl_recording=False)
try:
    if not isinstance(env.reset(seed=0), tuple): raise RuntimeError('unexpected reset result')
    if hasattr(env.action_space, 'seed'): env.action_space.seed(0)
    for _ in range(3):
        result = env.step(env.action_space.sample())
        if not isinstance(result, tuple) or len(result) != 5: raise RuntimeError('unexpected step result')
        if bool(result[2]) or bool(result[3]): break
finally:
    env.close()
"""


def _artifact_verification(repo_root: Path, environment: Mapping[str, str]) -> bool:
    result = _run_child(
        repo_root,
        environment,
        "from scripts.tools.validate_ecosystem_handoff_fixture import main; raise SystemExit(main())",
        ("--packet-dir", "tests/fixtures/ecosystem_handoff/v1"),
    )
    return result.returncode == 0 and result.stdout.startswith("PASS:")


def _lineage_verification(repo_root: Path, environment: Mapping[str, str]) -> bool:
    result = _run_child(
        repo_root,
        environment,
        "from scripts.tools.lineage_index import main; raise SystemExit(main())",
        (
            "--input",
            "tests/tools/fixtures/lineage_index/complete.json",
            "--check",
            "--format",
            "json",
        ),
    )
    return result.returncode == 0 and json.loads(result.stdout).get("ok") is True


def _report_metric_fixture(root: Path, repo_root: Path, environment: Mapping[str, str]) -> bool:
    aggregate = importlib.import_module("robot_sf.benchmark.aggregate")
    record = {
        "episode_id": "offline-smoke-episode",
        "scenario_id": "offline-smoke-scenario",
        "seed": 0,
        "scenario_params": {"algo": "simple_policy"},
        "metrics": {"success": True, "collisions": 0, "time_to_goal": 1.0, "path_length": 1.0},
    }
    report_dir = root / "reports"
    episode_path = report_dir / "episode_fixture.jsonl"
    episode_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    aggregates = aggregate.compute_aggregates(aggregate.read_jsonl(episode_path))
    _require(aggregates["simple_policy"]["success"]["mean"] == 1.0, REASONS["report"])
    claim = "Synthetic local smoke fixture only; not benchmark or research evidence."
    summary = {
        "status": "passed",
        "title": "Offline core smoke fixture",
        "metric_fixture_count": 1,
        "aggregation_group_count": len(aggregates),
        "source_command": SOURCE_COMMAND,
        "artifacts": ["episode_fixture.jsonl"],
        "claim_boundary": claim,
        "caveats": ["Synthetic record only; no real episode or performance conclusion is made."],
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    result = _run_child(
        repo_root,
        environment,
        "from scripts.reporting.generate_result_card import main; raise SystemExit(main())",
        (
            "summary.json",
            "--output-dir",
            "result_card",
            "--evidence-tier",
            "smoke",
            "--decision",
            "diagnostic",
            "--comparator",
            "synthetic fixture only",
            "--claim-boundary",
            claim,
        ),
        cwd=report_dir,
    )
    return result.returncode == 0 and all(
        (report_dir / relative).is_file()
        for relative in (
            "summary.json",
            "result_card/result_card.json",
            "result_card/result_card.md",
        )
    )


def _example_discovery(repo_root: Path, environment: Mapping[str, str]) -> bool:
    examples = importlib.import_module("robot_sf.examples.manifest_loader")
    manifest = examples.load_manifest(
        repo_root / "examples/examples_manifest.yaml", validate_paths=True
    )
    _require(bool(tuple(manifest.iter_ci_enabled_examples())), REASONS["examples"])
    result = _run_child(
        repo_root,
        environment,
        "from scripts.validation.validate_examples_manifest import main; raise SystemExit(main())",
        ("--manifest", "examples/examples_manifest.yaml"),
    )
    return result.returncode == 0 and "Examples manifest validation passed." in result.stdout


def _model_registry(repo_root: Path, environment: Mapping[str, str]) -> bool:
    result = _run_cli(repo_root, environment, ("models", "list", "--format", "json"))
    rows = json.loads(result.stdout)
    return (
        result.returncode == 0
        and isinstance(rows, list)
        and bool(rows)
        and all(isinstance(row, dict) and isinstance(row.get("model_id"), str) for row in rows)
    )


def _isolation_policy(root: Path, repo_root: Path, environment: Mapping[str, str]) -> bool:
    _require(
        not any(
            key in PRIVATE_ENV_KEYS or key.startswith(CLUSTER_ENV_PREFIXES) for key in environment
        ),
        REASONS["isolation"],
    )
    _require(
        all(
            environment.get(key) and Path(environment[key]).resolve().is_relative_to(root)
            for key in ROOT_RELATIVE
        ),
        REASONS["isolation"],
    )
    return (
        environment.get("PYTHONNOUSERSITE") == "1"
        and environment.get("UV_OFFLINE") == "1"
        and environment.get("UV_NO_SYNC") == "1"
        and environment.get("PATH") == str(root / "bin")
        and environment.get("PYTHONPATH", "").split(os.pathsep)
        == [str(root / "python"), str(repo_root)]
    )


def _check_file_text(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    except OSError as exc:
        raise SmokeFailure(REASONS["path"]) from exc
    if ABSOLUTE_PATH_PATTERN.search(text) or any(marker in text.lower() for marker in PATH_MARKERS):
        raise SmokeFailure(REASONS["path_leak"])


def _check_output_tree(root: Path, targets: Mapping[str, Path]) -> None:
    allowed = {root / "bin" / name: target for name, target in targets.items()}
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            _require(
                candidate in allowed and candidate.resolve() == allowed[candidate],
                REASONS["output_escape"],
            )
        elif candidate.is_file():
            _require(candidate.resolve().is_relative_to(root), REASONS["output_escape"])
            if not candidate.resolve().is_relative_to(root / "cache"):
                _check_file_text(candidate)


def _network_isolation(repo_root: Path, environment: Mapping[str, str]) -> bool:
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)
    except OSError:
        pass
    else:
        raise SmokeFailure(REASONS["network_guard"])
    result = _run_child(
        repo_root,
        environment,
        "import socket; socket.create_connection(('127.0.0.1', 9), timeout=0.1)",
    )
    return result.returncode != 0 and "network disabled" in result.stderr.lower()


def _build_receipt(capabilities: Sequence[Capability]) -> dict[str, Any]:
    rows = [capability.to_dict() for capability in capabilities]
    _require([row["id"] for row in rows] == list(ALL_CAPABILITIES), REASONS["isolation"])
    failed = any(row["status"] == STATUS_FAILED for row in rows)
    receipt = {
        "schema": SCHEMA,
        "status": STATUS_FAILED if failed else STATUS_PASSED,
        "mode": MODE,
        "capabilities": rows,
        "policy": {
            "network": "disabled",
            "scheduler": "unavailable",
            "gpu": "unavailable",
            "carla": "unavailable",
            "private_data": "unavailable",
            "institutional_context": "unavailable",
            "model_cache": "empty",
            "writes": "task_owned_root_only",
            "evidence": "smoke_only",
        },
        "outputs": [
            RECEIPT_NAME,
            "reports/episode_fixture.jsonl",
            "reports/summary.json",
            "reports/result_card/result_card.json",
            "reports/result_card/result_card.md",
        ],
    }
    _require(not ABSOLUTE_PATH_PATTERN.search(_compact_json(receipt)), REASONS["path_leak"])
    return _with_digest(receipt)


def _write_receipt(root: Path, receipt: Mapping[str, Any]) -> None:
    (root / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _core_capabilities(
    repo_root: Path, root: Path, environment: Mapping[str, str], targets: Mapping[str, Path]
) -> list[Capability]:
    callbacks = (
        _package_import,
        lambda: _cli_help(repo_root, environment),
        lambda: _cli_discovery(repo_root, environment),
        lambda: _environment_doctor(repo_root, root),
        lambda: _config_resolution(repo_root),
        lambda: _run_child(repo_root, environment, _EPISODE_CODE).returncode == 0,
        lambda: _artifact_verification(repo_root, environment),
        lambda: _lineage_verification(repo_root, environment),
        lambda: _report_metric_fixture(root, repo_root, environment),
        lambda: _example_discovery(repo_root, environment),
        lambda: _model_registry(repo_root, environment),
        lambda: _isolation_policy(root, repo_root, environment),
        lambda: _check_output_tree(root, targets),
        lambda: _network_isolation(repo_root, environment),
    )
    return [
        _stage(capability_id, callback, reason)
        for capability_id, callback, reason in zip(
            CORE_CAPABILITIES,
            callbacks,
            (
                REASONS[key]
                for key in "package_import cli_help cli_discovery doctor config episode artifact lineage report examples models isolation path network_guard".split()
            ),
            strict=True,
        )
    ]


def _optional_capabilities(environment: Mapping[str, str], root: Path) -> list[Capability]:
    rows = [
        _probe_scheduler(environment["PATH"]),
        _probe_gpu(environment),
        Capability("carla", False, STATUS_UNAVAILABLE, REASONS["carla"]),
        _probe_private_data(environment),
        _probe_institutional_context(environment),
        _probe_model_cache(root / "cache" / "model"),
        Capability("network", False, STATUS_UNAVAILABLE, REASONS["network"]),
    ]
    return rows


def run_smoke(*, repo_root: Path | None = None, output_root: Path | None = None) -> dict[str, Any]:
    """Run the bounded smoke and return its deterministic receipt."""
    resolved_repo = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    root, manager = _prepare_root(resolved_repo, output_root)
    environment, targets = _controlled_environment(root, resolved_repo)
    original_sys_path = list(sys.path)
    if str(resolved_repo) not in sys.path:
        sys.path.insert(0, str(resolved_repo))
    try:
        with (
            _temporary_environment(environment),
            _network_denied(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            receipt = _build_receipt(
                _core_capabilities(resolved_repo, root, environment, targets)
                + _optional_capabilities(environment, root)
            )
            _write_receipt(root, receipt)
    finally:
        sys.path[:] = original_sys_path
        if manager is not None:
            manager.cleanup()
    return receipt


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", required=True)
    parser.add_argument("--isolated", action="store_true", required=True)
    parser.add_argument("--json", action="store_true", help="Emit only the JSON receipt.")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line smoke harness."""
    args = build_parser().parse_args(argv)
    try:
        receipt = run_smoke(output_root=args.output_root)
    except SmokeFailure as exc:
        receipt = _with_digest(
            {
                "schema": SCHEMA,
                "status": STATUS_FAILED,
                "mode": MODE,
                "capabilities": [],
                "error_code": exc.reason_code,
            }
        )
    except (OSError, RuntimeError, TypeError, ValueError, ImportError, AttributeError, LookupError):
        receipt = _with_digest(
            {
                "schema": SCHEMA,
                "status": STATUS_FAILED,
                "mode": MODE,
                "capabilities": [],
                "error_code": REASONS["isolation"],
            }
        )
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        for capability in receipt.get("capabilities", []):
            print(f"{capability['status']}: {capability['id']} [{capability['reason_code']}]")
        print(f"status: {receipt['status']}")
    return 0 if receipt.get("status") == STATUS_PASSED else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
