#!/usr/bin/env python3
"""Run the bounded CPU-only DWA configuration sensitivity diagnostic for issue #5262.

This runner evaluates three explicit configuration points across three fixed-seed classic
archetypes. It writes raw JSONL only under ``output/`` and emits compact per-cell and
per-episode CSV tables that can be promoted into the issue evidence packet. The result is
diagnostic-only: it cannot promote DWA to the comparative roster or change frozen-suite policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
DEFAULT_MANIFEST = REPO_ROOT / "configs/benchmarks/issue_5262_dwa_config_sensitivity.yaml"
EXPECTED_SCENARIO_COUNT = 3
EXPECTED_CONFIG_POINT_COUNT = 3


def _repo_path(value: str) -> Path:
    """Resolve a repository-relative configuration path."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_scenario_ids(manifest: dict[str, Any]) -> list[str]:
    """Return the bounded unique scenario selection or reject an invalid manifest."""
    scenario_ids = manifest.get("scenario_ids")
    if not isinstance(scenario_ids, list) or len(scenario_ids) != EXPECTED_SCENARIO_COUNT:
        raise ValueError(f"manifest must select exactly {EXPECTED_SCENARIO_COUNT} scenarios")
    if len(set(scenario_ids)) != len(scenario_ids) or not all(
        isinstance(value, str) and value for value in scenario_ids
    ):
        raise ValueError("manifest scenario_ids must be unique non-empty strings")
    return scenario_ids


def _validate_config_points(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return explicit reversible config points or reject an invalid declaration."""
    config_points = manifest.get("config_points")
    if not isinstance(config_points, dict) or len(config_points) != EXPECTED_CONFIG_POINT_COUNT:
        raise ValueError(
            f"manifest must declare exactly {EXPECTED_CONFIG_POINT_COUNT} config points"
        )
    for name, point in config_points.items():
        if not isinstance(name, str) or not name:
            raise ValueError("config point names must be non-empty strings")
        if not isinstance(point, dict) or not isinstance(point.get("overrides"), dict):
            raise ValueError(f"config point {name!r} must declare mapping overrides")
    return config_points


def _validate_runner(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the DWA runner declaration or reject an invalid one."""
    runner = manifest.get("runner")
    if not isinstance(runner, dict):
        raise ValueError("manifest runner must be a mapping")
    if runner.get("algo") != "dwa":
        raise ValueError("manifest runner algo must be 'dwa'")
    return runner


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the committed sensitivity manifest."""
    with path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    if manifest.get("schema_version") != "dwa-config-sensitivity.v1":
        raise ValueError("manifest schema_version must be 'dwa-config-sensitivity.v1'")
    scenario_ids = _validate_scenario_ids(manifest)
    config_points = _validate_config_points(manifest)
    _validate_runner(manifest)
    if int(manifest.get("seeds_per_scenario", 0)) != 3:
        raise ValueError("manifest must pin three seeds per selected scenario")
    if len(scenario_ids) * len(config_points) * int(manifest["seeds_per_scenario"]) > 30:
        raise ValueError("manifest exceeds the issue's 30-episode bound")
    return manifest


def select_scenarios(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the source matrix and retain the manifest's exact scenario subset."""
    source = _repo_path(str(manifest["source_matrix"]))
    requested = list(manifest["scenario_ids"])
    by_name = {str(row.get("name")): dict(row) for row in load_scenarios(source, base_dir=source)}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"selected scenarios are absent from source matrix: {missing}")
    selected = [deepcopy(by_name[name]) for name in requested]
    expected_seeds = int(manifest["seeds_per_scenario"])
    for scenario in selected:
        seeds = scenario.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != expected_seeds:
            raise ValueError(
                f"scenario {scenario['name']!r} must provide exactly {expected_seeds} fixed seeds"
            )
    return selected


def effective_config(manifest: dict[str, Any], config_id: str) -> dict[str, Any]:
    """Merge one reversible point override onto the canonical DWA config."""
    runner = manifest["runner"]
    base_path = _repo_path(str(runner["base_algo_config"]))
    with base_path.open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    if not isinstance(base, dict):
        raise ValueError(f"base DWA config must be a mapping: {base_path}")
    point = manifest["config_points"][config_id]
    config = dict(base)
    config.update(point["overrides"])
    return config


def _config_hash(config: dict[str, Any]) -> str:
    """Return a compact stable hash of an effective algorithm config."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Load a runner JSONL file, failing clearly on malformed rows."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL row {line_number} in {path}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"JSONL row {line_number} in {path} must be an object")
        records.append(record)
    return records


def _outcome(record: dict[str, Any], key: str) -> bool:
    """Read a boolean outcome field from an episode record."""
    outcome = record.get("outcome")
    return bool(outcome.get(key, False)) if isinstance(outcome, dict) else False


def episode_rows(
    records: list[dict[str, Any]], *, config_id: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Normalize raw runner records into reviewable per-episode rows."""
    config_hash = _config_hash(config)
    rows = []
    for record in records:
        rows.append(
            {
                "config_id": config_id,
                "config_sha256_16": config_hash,
                "scenario_id": str(record.get("scenario_id", "unknown")),
                "seed": int(record.get("seed", -1)),
                "route_complete": int(_outcome(record, "route_complete")),
                "termination_reason": str(record.get("termination_reason", "unknown")),
                "collision_event": int(_outcome(record, "collision_event")),
                "timeout_event": int(_outcome(record, "timeout_event")),
                "steps": int(record.get("steps", 0)),
            }
        )
    return rows


def cell_rows(
    rows: list[dict[str, Any]], manifest: dict[str, Any], configs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate the fixed three-seed results for each archetype x config-point cell."""
    scenarios = {row["name"]: row for row in select_scenarios(manifest)}
    expected = int(manifest["seeds_per_scenario"])
    result = []
    for config_id in manifest["config_points"]:
        for scenario_id in manifest["scenario_ids"]:
            matching = [
                row
                for row in rows
                if row["config_id"] == config_id and row["scenario_id"] == scenario_id
            ]
            if len(matching) != expected:
                raise ValueError(
                    f"cell {config_id}/{scenario_id} wrote {len(matching)} rows, expected {expected}"
                )
            config = configs[config_id]
            result.append(
                {
                    "config_id": config_id,
                    "scenario_id": scenario_id,
                    "archetype": scenarios[scenario_id]
                    .get("metadata", {})
                    .get("archetype", "unknown"),
                    "seeds": ";".join(
                        str(row["seed"]) for row in sorted(matching, key=lambda row: row["seed"])
                    ),
                    "episodes": len(matching),
                    "route_complete": sum(row["route_complete"] for row in matching),
                    "timeouts": sum(row["timeout_event"] for row in matching),
                    "collisions": sum(row["collision_event"] for row in matching),
                    "median_steps": sorted(row["steps"] for row in matching)[len(matching) // 2],
                    "config_sha256_16": _config_hash(config),
                    "max_linear_speed": config["max_linear_speed"],
                    "max_angular_speed": config["max_angular_speed"],
                    "max_linear_acceleration": config["max_linear_acceleration"],
                    "max_angular_acceleration": config["max_angular_acceleration"],
                    "goal_tolerance": config["goal_tolerance"],
                    "heading_weight": config["heading_weight"],
                    "clearance_weight": config["clearance_weight"],
                    "velocity_weight": config["velocity_weight"],
                    "progress_weight": config["progress_weight"],
                }
            )
    return result


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    distance_convention: str | None = None,
) -> None:
    """Write homogeneous rows as a deterministic CSV artifact."""
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# AI-GENERATED NEEDS-REVIEW\n")
        if distance_convention is not None:
            handle.write(f"# distance_convention: {distance_convention}\n")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_campaign(manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Run every declared sensitivity point and write raw plus compact result tables."""
    scenarios = select_scenarios(manifest)
    runner = manifest["runner"]
    source_matrix = _repo_path(str(manifest["source_matrix"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir = out_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}
    expected_per_point = len(scenarios) * int(manifest["seeds_per_scenario"])
    for config_id in manifest["config_points"]:
        config = effective_config(manifest, config_id)
        configs[config_id] = config
        config_path = config_dir / f"{config_id}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        episodes_path = out_dir / f"episodes_{config_id}.jsonl"
        if episodes_path.exists():
            episodes_path.unlink()
        summary = run_map_batch(
            scenarios,
            episodes_path,
            schema_path=SCHEMA_PATH,
            scenario_path=source_matrix,
            algo=str(runner["algo"]),
            algo_config_path=str(config_path),
            horizon=int(runner["horizon"]),
            dt=float(runner["dt"]),
            record_forces=False,
            workers=int(runner["workers"]),
            resume=False,
            benchmark_profile=str(runner["benchmark_profile"]),
        )
        records = _read_records(episodes_path)
        if int(summary.get("failed_jobs", 0)) != 0 or len(records) != expected_per_point:
            raise RuntimeError(
                f"{config_id} did not complete cleanly: summary={summary}, records={len(records)}"
            )
        all_rows.extend(episode_rows(records, config_id=config_id, config=config))
    all_rows.sort(key=lambda row: (row["config_id"], row["scenario_id"], row["seed"]))
    cells = cell_rows(all_rows, manifest, configs)
    _write_csv(out_dir / "dwa_config_sensitivity_episode_rows.csv", all_rows)
    _write_csv(
        out_dir / "dwa_config_sensitivity_per_cell_rows.csv",
        cells,
        distance_convention="center_center",
    )
    report = {
        "schema_version": manifest["schema_version"],
        "issue": manifest["issue"],
        "claim_boundary": manifest["claim_boundary"],
        "episodes": len(all_rows),
        "cells": len(cells),
        "route_complete": sum(row["route_complete"] for row in all_rows),
        "timeouts": sum(row["timeout_event"] for row in all_rows),
        "collisions": sum(row["collision_event"] for row in all_rows),
        "config_hashes": {name: _config_hash(config) for name, config in configs.items()},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run static manifest validation or the complete bounded diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "output/benchmarks/issue_5262")
    parser.add_argument(
        "--check", action="store_true", help="Validate the manifest without running episodes"
    )
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    select_scenarios(manifest)
    if args.check:
        print("issue-5262 DWA sensitivity manifest is valid")
        return 0
    report = run_campaign(manifest, args.out_dir)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
