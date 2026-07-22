#!/usr/bin/env python3
"""Fail fast when a campaign manifest's pedestrian population contract drifts.

For every row in a generated heterogeneous-population manifest, this CPU-only
canary runs a short episode and checks that the declared population equals the
population instantiated by the simulator, the runtime trace labels, and the
per-pedestrian rows in the emitted control trace.  It is a pre-campaign
diagnostic, not benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf.benchmark.heterogeneous_population_ablation import (
    PopulationMixNotRealizableError,
    build_runtime_population_control_trace_labels,
)
from robot_sf.benchmark.heterogeneous_population_ablation_runner import build_episode_scenario
from robot_sf.benchmark.map_runner import build_map_policy
from robot_sf.benchmark.map_runner_episode import run_map_episode

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTROL_TRACE_PATH = ("algorithm_metadata", "pedestrian_control_trace", "pedestrians")
_SPAWN_SYNTHESIS_ERROR = "force_population_size requires a pedestrian route or crowded zone"


def parse_args() -> argparse.Namespace:
    """Parse the bounded manifest canary command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Generated JSON manifest with a manifest_rows sequence.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum simulation steps per manifest row (default: 20).",
    )
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    """Load the existing generated-manifest row contract without inventing cells."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be a JSON mapping")
    rows = payload.get("manifest_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, str) or not rows:
        raise ValueError("manifest.manifest_rows must be a non-empty sequence")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("manifest.manifest_rows entries must be mappings")
    return [dict(row) for row in rows]


def _declared_population(row: Mapping[str, Any]) -> int:
    """Read one row's declared population from its arm counts."""

    arm_population = row.get("arm_population")
    if not isinstance(arm_population, Mapping):
        raise ValueError("arm_population must be a mapping")
    counts = arm_population.get("counts")
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("arm_population.counts must be a non-empty mapping")
    try:
        values = [int(value) for value in counts.values()]
    except (TypeError, ValueError) as exc:
        raise ValueError("arm_population.counts values must be integers") from exc
    if any(value < 0 for value in values):
        raise ValueError("arm_population.counts values must be non-negative")
    return sum(values)


def _cell_identity(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    """Return stable, human-readable coordinates for a manifest row."""

    return {
        "row_index": row_index,
        "scenario": str(row.get("scenario_id", f"manifest_rows[{row_index}]")),
        "population_arm": str(row.get("population_arm", "unknown")),
        "planner": str(row.get("planner", "unknown")),
        "seed": row.get("seed"),
    }


def _trace_row_count(record: Mapping[str, Any]) -> int:
    """Count the emitted per-pedestrian trace rows in one episode record."""

    value: Any = record
    for key in _CONTROL_TRACE_PATH:
        if not isinstance(value, Mapping):
            raise ValueError(f"emitted record missing {'.'.join(_CONTROL_TRACE_PATH)}")
        value = value.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"emitted record {'.'.join(_CONTROL_TRACE_PATH)} must be a sequence")
    return len(value)


def run_manifest_cell(
    row: Mapping[str, Any], *, manifest_path: Path, max_steps: int
) -> dict[str, int]:
    """Run one bounded manifest row and return independently observed counts."""

    declared = _declared_population(row)
    instantiated: int | None = None
    emitted_labels: int | None = None
    scenario = build_episode_scenario(dict(row), max_episode_steps=max_steps)

    def capture_runtime_labels(instantiated_count: int) -> list[dict[str, Any]]:
        nonlocal instantiated, emitted_labels
        instantiated = int(instantiated_count)
        labels = build_runtime_population_control_trace_labels(
            row["arm_population"],
            instantiated,
        )
        emitted_labels = len(labels)
        return labels

    record = run_map_episode(
        scenario=scenario,
        seed=int(row["seed"]),
        horizon=max_steps,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo=str(row["planner"]),
        scenario_path=manifest_path,
        record_planner_decision_trace=False,
        record_simulation_step_trace=True,
        pedestrian_control_trace_label_builder=capture_runtime_labels,
        policy_builder=build_map_policy,
    )
    if instantiated is None or emitted_labels is None:
        raise ValueError("runtime did not emit the instantiated population and trace labels")
    return {
        "declared_population": declared,
        "instantiated_pedestrians": instantiated,
        "emitted_labels": emitted_labels,
        "trace_rows": _trace_row_count(record),
    }


def _failure_disposition(exc: Exception) -> str:
    """Classify known no-spawn failures without treating them as a green canary."""

    if _SPAWN_SYNTHESIS_ERROR in str(exc):
        return "unrealizable_without_spawn_synthesis"
    if isinstance(exc, PopulationMixNotRealizableError):
        return "unrealizable_population_mix"
    return "runtime_error"


def run_canary(
    rows: Sequence[Mapping[str, Any]], *, manifest_path: Path, max_steps: int
) -> dict[str, Any]:
    """Check every manifest row and retain every failure for actionable triage."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        cell = _cell_identity(row, row_index)
        try:
            counts = run_manifest_cell(row, manifest_path=manifest_path, max_steps=max_steps)
            cell.update(counts)
            count_values = set(counts.values())
            cell["status"] = "passed" if len(count_values) == 1 else "failed"
            if cell["status"] == "failed":
                cell["reason"] = "declared!=instantiated!=labels!=traces"
        except (AssertionError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            cell["declared_population"] = _safe_declared_population(row)
            cell["instantiated_pedestrians"] = None
            cell["emitted_labels"] = None
            cell["trace_rows"] = None
            cell["status"] = _failure_disposition(exc)
            cell["reason"] = str(exc)
        cells.append(cell)

    failed_cells = [cell for cell in cells if cell["status"] != "passed"]
    return {
        "schema_version": "s30_campaign_canary.v1",
        "evidence_status": "diagnostic-only",
        "passed": not failed_cells,
        "cell_count": len(cells),
        "failed_cell_count": len(failed_cells),
        "cells": cells,
    }


def _safe_declared_population(row: Mapping[str, Any]) -> int | None:
    """Preserve a declared count in malformed-cell diagnostics when possible."""

    try:
        return _declared_population(row)
    except ValueError:
        return None


def main() -> int:
    """Run the pre-campaign diagnostic and return a fail-closed process status."""

    args = parse_args()
    if args.max_steps <= 0:
        print("error: --max-steps must be positive", file=sys.stderr)
        return 2
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (_REPO_ROOT / manifest_path).resolve()
    try:
        rows = load_manifest_rows(manifest_path)
        report = run_canary(rows, manifest_path=manifest_path, max_steps=args.max_steps)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
