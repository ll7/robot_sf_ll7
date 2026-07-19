#!/usr/bin/env python3
"""Fail-closed one-row native SIPP smoke validator for issue #5416.

This validator first checks the frozen four-geometry packet, then runs exactly
one ``classic_head_on_corridor_low``/111 episode through the tracked native SIPP
command. It proves transport and analyzer eligibility only; it does not run a
campaign or interpret benchmark outcomes.
"""

from __future__ import annotations

import argparse
import json
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


class SmokeError(ValueError):
    """Raised when the one-row native SIPP contract is not established."""


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


def validate_smoke(
    *,
    packet_path: Path,
    native_config_path: Path,
    scenario_id: str,
    seed: int,
    horizon: int,
    dt: float,
    workers: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute and inspect the exact one-row native SIPP smoke contract."""
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
        "episode_path": str(episodes_path),
    }
    expected = {
        "planner_id": "sipp_lattice",
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
        raise SmokeError(f"native SIPP smoke contract failed: {failures}")
    return {"status": "ready", **checks}


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
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--dt", required=True, type=float)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate_smoke(
            packet_path=args.packet,
            native_config_path=args.native_config,
            scenario_id=args.scenario_id,
            seed=args.seed,
            horizon=args.horizon,
            dt=args.dt,
            workers=args.workers,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        result = {"status": "blocked", "error": str(exc)}
    print(json.dumps(result, sort_keys=True) if args.json else result)
    return 0 if result.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
