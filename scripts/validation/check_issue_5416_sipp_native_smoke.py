#!/usr/bin/env python3
"""Fail-closed native #5416 smoke validator.

This validator first checks the frozen four-geometry packet, then runs exactly
one ``classic_head_on_corridor_low``/111 episode through a tracked native
planner command.  Its five-planner helper combines the frozen SIPP and
comparator one-cell rows.  It proves transport and analyzer eligibility only;
it does not run a campaign or interpret benchmark outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios
from scripts.analysis.analyze_issue_5416_sipp_four_geometry import build_analysis
from scripts.validation.check_issue_5416_sipp_four_geometry_packet import (
    load_packet,
    validate_packet,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"
_FROZEN_NATIVE_CONFIGS = (
    ("sipp_lattice", REPO_ROOT / "configs/algos/sipp_lattice_native_command.yaml"),
    ("hybrid_rule_v0_minimal", REPO_ROOT / "configs/algos/hybrid_rule_v0_minimal.yaml"),
    ("teb", REPO_ROOT / "configs/algos/teb_commitment_camera_ready.yaml"),
    ("nmpc_social", REPO_ROOT / "configs/algos/nmpc_social_exploratory.yaml"),
    ("dwa", REPO_ROOT / "configs/algos/dwa_classic.yaml"),
)
_NATIVE_ROW_TIMEOUT_SECONDS = 60.0
_NATIVE_ROW_PROGRESS_TIMEOUT_SECONDS = 15.0
_WATCHDOG_POLL_SECONDS = 0.1
_WATCHDOG_TERMINATE_GRACE_SECONDS = 0.2

# The outer watchdog launches this small wrapper as the process-group leader.
# It deliberately remains alive after the native command returns, giving the
# watchdog a live, non-reusable identity for the whole group until cleanup.
# Relying on ``ps`` session values is not safe here: macOS can report ``sess``
# as 0, which makes a recycled numeric process-group id indistinguishable from
# an unrelated process group.
_WATCHDOG_SENTINEL_CODE = """
import json
import os
from pathlib import Path
import subprocess
import sys
import time

status_path = Path(sys.argv[1])
child = subprocess.Popen(sys.argv[2:])
returncode = child.wait()
temporary_path = status_path.with_name(status_path.name + ".tmp")
temporary_path.write_text(json.dumps({"returncode": returncode}), encoding="utf-8")
os.replace(temporary_path, status_path)
while True:
    time.sleep(60)
"""


class SmokeError(ValueError):
    """Raised when the one-row native SIPP contract is not established."""


class SmokeWatchdogError(SmokeError):
    """Raised when a native row exceeds the bounded watchdog contract."""

    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        """Keep the machine-readable timeout evidence with the error message."""
        super().__init__(message)
        self.details = details


def _selected_scenario(packet: dict[str, Any], scenario_id: str, seed: int) -> dict[str, Any]:
    """Load one frozen scenario and override only its local smoke seed."""
    scenario_matrix = packet.get("scenario_contract", {}).get("scenario_matrix")
    if not isinstance(scenario_matrix, str):
        raise SmokeError("packet scenario matrix is missing")
    scenarios = load_scenarios(REPO_ROOT / scenario_matrix)
    for scenario in scenarios:
        identifier = str(scenario.get("name") or scenario.get("scenario_id") or "")
        if identifier == scenario_id:
            selected = deepcopy(scenario)
            selected["seeds"] = [seed]
            return selected
    raise SmokeError(f"scenario {scenario_id!r} is absent from the frozen scenario matrix")


def _run_native_row(
    *,
    packet_path: Path,
    native_config_path: Path,
    scenario_id: str,
    seed: int,
    horizon: int,
    dt: float,
    workers: int,
    episodes_path: Path,
) -> None:
    """Run the frozen native row in the watchdog-owned child process."""
    packet = load_packet(packet_path)
    scenario = _selected_scenario(packet, scenario_id, seed)
    run_map_batch(
        [scenario],
        episodes_path,
        SCHEMA_PATH,
        scenario_path=REPO_ROOT / packet["scenario_contract"]["scenario_matrix"],
        horizon=horizon,
        dt=dt,
        algo="native_command",
        algo_config_path=str(native_config_path),
        workers=workers,
        resume=False,
        record_forces=False,
    )


def _native_worker_command(
    *,
    packet_path: Path,
    native_config_path: Path,
    scenario_id: str,
    seed: int,
    horizon: int,
    dt: float,
    workers: int,
    episodes_path: Path,
) -> list[str]:
    """Build the private child command without changing native planner inputs."""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_run-native-row",
        "--packet",
        str(packet_path),
        "--native-config",
        str(native_config_path),
        "--scenario-id",
        scenario_id,
        "--seed",
        str(seed),
        "--horizon",
        str(horizon),
        "--dt",
        str(dt),
        "--workers",
        str(workers),
        "--output-dir",
        str(episodes_path.parent),
    ]


def _episode_progress_token(episodes_path: Path) -> tuple[int, int] | None:
    """Return a token that changes only when the native row writes output."""
    try:
        stat = episodes_path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _watchdog_sentinel_owns_process_group(
    process: subprocess.Popen[bytes], *, process_group_id: int
) -> bool:
    """Return whether our still-live sentinel is the exact group leader.

    A live process object prevents PID reuse.  We therefore never identify a
    group from ``ps``' numeric PGID/session fields: a macOS ``sess=0`` value can
    otherwise match an unrelated group after the original leader exited.
    """
    if process.poll() is not None or process.pid != process_group_id:
        return False
    try:
        return os.getpgid(process.pid) == process_group_id
    except (OSError, ProcessLookupError):
        return False


def _terminate_process_group(
    process: subprocess.Popen[bytes], *, process_group_id: int | None
) -> None:
    """Terminate the sentinel-owned native group, including exited-parent children."""
    if os.name == "posix":
        if process_group_id is not None and _watchdog_sentinel_owns_process_group(
            process,
            process_group_id=process_group_id,
        ):
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except (PermissionError, ProcessLookupError):
                pass
            else:
                time.sleep(_WATCHDOG_TERMINATE_GRACE_SECONDS)
                if _watchdog_sentinel_owns_process_group(
                    process,
                    process_group_id=process_group_id,
                ):
                    try:
                        os.killpg(process_group_id, signal.SIGKILL)
                    except (PermissionError, ProcessLookupError):
                        # macOS can report EPERM after SIGTERM has already removed the
                        # session leader and every remaining group member.
                        pass
    elif process.poll() is None:  # pragma: no cover - the frozen native comparator is POSIX-only.
        process.terminate()
    if process.poll() is None:
        try:
            process.wait(timeout=_WATCHDOG_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _watchdog_sentinel_command(*, command: list[str], status_path: Path) -> list[str]:
    """Wrap one native command in a cleanup sentinel with an atomic exit receipt."""
    return [sys.executable, "-c", _WATCHDOG_SENTINEL_CODE, str(status_path), *command]


def _read_native_returncode(status_path: Path) -> int | None:
    """Read the wrapped native command's atomic exit receipt when available."""
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    returncode = payload.get("returncode") if isinstance(payload, dict) else None
    return returncode if isinstance(returncode, int) and not isinstance(returncode, bool) else None


def _raise_for_watchdog_deadline(
    *,
    now: float,
    started_at: float,
    last_progress_at: float,
    timeout_seconds: float,
    progress_timeout_seconds: float,
    planner_id: str,
    episodes_path: Path,
) -> None:
    """Raise the bounded, structured failure that reached first."""
    details = {
        "planner_id": planner_id,
        "timeout_seconds": timeout_seconds,
        "progress_timeout_seconds": progress_timeout_seconds,
        "episode_path": str(episodes_path),
        "child_process_terminated": True,
    }
    if now - started_at >= timeout_seconds:
        raise SmokeWatchdogError(
            "native smoke exceeded its end-to-end watchdog timeout",
            details={"failure_kind": "native_end_to_end_timeout", **details},
        )
    if now - last_progress_at >= progress_timeout_seconds:
        raise SmokeWatchdogError(
            "native smoke made no episode-output progress before its watchdog deadline",
            details={"failure_kind": "native_progress_timeout", **details},
        )


def _run_native_row_with_watchdog(
    *,
    command: list[str],
    episodes_path: Path,
    timeout_seconds: float,
    progress_timeout_seconds: float,
    planner_id: str,
) -> None:
    """Bound one native row and fail closed when its episode output stops moving."""
    if timeout_seconds <= 0 or progress_timeout_seconds <= 0:
        raise SmokeError("native watchdog timeouts must be positive")
    if progress_timeout_seconds > timeout_seconds:
        raise SmokeError("native progress timeout must not exceed the end-to-end timeout")
    status_path = episodes_path.with_name(f".{episodes_path.name}.watchdog-{uuid.uuid4().hex}.json")
    process = subprocess.Popen(
        _watchdog_sentinel_command(command=command, status_path=status_path),
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix",
    )
    process_group_id = process.pid if os.name == "posix" else None
    try:
        started_at = time.monotonic()
        last_progress_at = started_at
        progress_token = _episode_progress_token(episodes_path)
        native_returncode: int | None = None
        while process.poll() is None:
            native_returncode = _read_native_returncode(status_path)
            if native_returncode is not None:
                break
            now = time.monotonic()
            current_token = _episode_progress_token(episodes_path)
            if current_token != progress_token:
                progress_token = current_token
                last_progress_at = now
            _raise_for_watchdog_deadline(
                now=now,
                started_at=started_at,
                last_progress_at=last_progress_at,
                timeout_seconds=timeout_seconds,
                progress_timeout_seconds=progress_timeout_seconds,
                planner_id=planner_id,
                episodes_path=episodes_path,
            )
            time.sleep(_WATCHDOG_POLL_SECONDS)
        if native_returncode is None:
            raise SmokeError("native smoke watchdog sentinel exited before its native exit receipt")
        if native_returncode != 0:
            raise SmokeError(f"native smoke child exited with status {native_returncode}")
    finally:
        _terminate_process_group(
            process,
            process_group_id=process_group_id,
        )
        try:
            status_path.unlink()
        except FileNotFoundError:
            pass


def validate_smoke(  # noqa: PLR0913
    *,
    packet_path: Path,
    native_config_path: Path,
    scenario_id: str,
    seed: int,
    horizon: int,
    dt: float,
    workers: int,
    output_dir: Path,
    expected_planner_id: str = "sipp_lattice",
    timeout_seconds: float = _NATIVE_ROW_TIMEOUT_SECONDS,
    progress_timeout_seconds: float = _NATIVE_ROW_PROGRESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Execute and inspect one exact frozen native smoke row."""
    if (scenario_id, seed, horizon, dt, workers) != (
        "classic_head_on_corridor_low",
        111,
        500,
        0.1,
        1,
    ):
        raise SmokeError("smoke arguments must stay pinned to corridor_low/111/500/0.1/1")
    packet = load_packet(packet_path)
    gate = validate_packet(packet, repo_root=REPO_ROOT)
    if gate.get("status") != "ready":
        raise SmokeError(f"frozen packet geometry gate is not ready: {gate.get('blocked_rows')}")
    if not native_config_path.is_file():
        raise SmokeError(f"native config is missing: {native_config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    _run_native_row_with_watchdog(
        command=_native_worker_command(
            packet_path=packet_path,
            native_config_path=native_config_path,
            scenario_id=scenario_id,
            seed=seed,
            horizon=horizon,
            dt=dt,
            workers=workers,
            episodes_path=episodes_path,
        ),
        episodes_path=episodes_path,
        timeout_seconds=timeout_seconds,
        progress_timeout_seconds=progress_timeout_seconds,
        planner_id=expected_planner_id,
    )
    rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1:
        raise SmokeError(f"native smoke must emit exactly one row, got {len(rows)}")
    row = rows[0]
    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, dict):
        raise SmokeError("native smoke row is missing algorithm metadata")
    native = metadata.get("native_command")
    if not isinstance(native, dict) or not native.get("geometry_input_verified"):
        raise SmokeError("native smoke row does not prove static geometry reached the command")
    geometry_consumption = native.get("geometry_consumption")
    if (
        not isinstance(geometry_consumption, dict)
        or geometry_consumption.get("obstacle_occupied_cells", 0) <= 0
        or geometry_consumption.get("combined_occupied_cells", 0)
        < max(
            geometry_consumption.get("obstacle_occupied_cells", 0),
            geometry_consumption.get("pedestrian_occupied_cells", 0),
        )
        or geometry_consumption.get("combined_matches_union") is not True
    ):
        raise SmokeError(
            "native smoke row does not prove canonical planner channels consumed geometry"
        )
    report = build_analysis(
        episode_paths=[episodes_path], output_dir=output_dir / "analyzer", packet_path=packet_path
    )
    matrix = report.get("matrix", {})
    planner_kinematics = metadata.get("planner_kinematics", {})
    diagnostics = metadata.get("planner_diagnostics")
    checks = {
        "planner_id": row.get("planner_id") or metadata.get("config", {}).get("planner_variant"),
        "execution_mode": planner_kinematics.get("execution_mode"),
        "fallback_or_degraded": metadata.get("fallback_or_degraded"),
        "eligible_rows": matrix.get("eligible_rows"),
        "excluded_rows": matrix.get("excluded_rows"),
        "deadlock_metric_present": isinstance(row.get("metrics", {}).get("deadlock"), bool),
        "planner_diagnostics_present": isinstance(diagnostics, dict)
        and bool(diagnostics.get("planner_step_runtime_seconds")),
        "geometry_input_verified": bool(native.get("geometry_input_verified")),
        "geometry_consumption": geometry_consumption,
        "episode_path": str(episodes_path),
    }
    expected = {
        "planner_id": expected_planner_id,
        "execution_mode": "native",
        "fallback_or_degraded": False,
        "eligible_rows": 1,
        "excluded_rows": 0,
        "deadlock_metric_present": True,
        "planner_diagnostics_present": True,
        "geometry_input_verified": True,
    }
    failures = {key: checks[key] for key, value in expected.items() if checks[key] != value}
    if failures:
        raise SmokeError(f"native smoke contract failed: {failures}")
    return {"status": "ready", **checks}


def validate_five_planner_smoke(*, packet_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the exact one-cell native rows for the frozen five-planner roster."""
    rows = {}
    episode_paths = []
    for planner_id, config_path in _FROZEN_NATIVE_CONFIGS:
        result = validate_smoke(
            packet_path=packet_path,
            native_config_path=config_path,
            scenario_id="classic_head_on_corridor_low",
            seed=111,
            horizon=500,
            dt=0.1,
            workers=1,
            output_dir=output_dir / planner_id,
            expected_planner_id=planner_id,
        )
        rows[planner_id] = result
        episode_paths.append(Path(result["episode_path"]))
    report = build_analysis(
        episode_paths=episode_paths,
        output_dir=output_dir / "five_planner_analyzer",
        packet_path=packet_path,
    )
    matrix = report.get("matrix", {})
    if matrix.get("eligible_rows") != 5 or matrix.get("excluded_rows") != 0:
        raise SmokeError(
            "five-planner native smoke contract failed: "
            f"eligible_rows={matrix.get('eligible_rows')}, excluded_rows={matrix.get('excluded_rows')}"
        )
    return {
        "status": "ready",
        "planners": rows,
        "eligible_rows": matrix["eligible_rows"],
        "excluded_rows": matrix["excluded_rows"],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the smoke validator and emit a compact JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=REPO_ROOT / "configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml",
    )
    parser.add_argument(
        "--native-config",
        type=Path,
        default=REPO_ROOT / "configs/algos/sipp_lattice_native_command.yaml",
    )
    parser.add_argument("--scenario-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-planner-id", default="sipp_lattice")
    parser.add_argument("--five-planner-smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--_run-native-row", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._run_native_row:
            _run_native_row(
                packet_path=args.packet,
                native_config_path=args.native_config,
                scenario_id=args.scenario_id,
                seed=args.seed,
                horizon=args.horizon,
                dt=args.dt,
                workers=args.workers,
                episodes_path=args.output_dir / "episodes.jsonl",
            )
            return 0
        if args.five_planner_smoke:
            result = validate_five_planner_smoke(
                packet_path=args.packet,
                output_dir=args.output_dir,
            )
        else:
            required_standard_args = (
                args.scenario_id,
                args.seed,
                args.horizon,
                args.dt,
                args.workers,
            )
            if any(value is None for value in required_standard_args):
                parser.error(
                    "--scenario-id, --seed, --horizon, --dt, and --workers are required "
                    "unless --five-planner-smoke is set"
                )
            result = validate_smoke(
                packet_path=args.packet,
                native_config_path=args.native_config,
                scenario_id=args.scenario_id,
                seed=args.seed,
                horizon=args.horizon,
                dt=args.dt,
                workers=args.workers,
                output_dir=args.output_dir,
                expected_planner_id=args.expected_planner_id,
            )
    # Convert expected validation, I/O, and assertion failures into auditable
    # blocked output. Fatal process conditions deliberately remain unhandled.
    except SmokeWatchdogError as exc:
        result = {"status": "blocked", "error": str(exc), **exc.details}
    except (
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        result = {"status": "blocked", "error": str(exc)}
    print(json.dumps(result, sort_keys=True) if args.json else result)
    return 0 if result.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
