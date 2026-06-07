#!/usr/bin/env python3
"""Generate adversarial manifests, materialize valid cases, and run a tiny planner smoke."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from robot_sf.adversarial.config import SearchSpaceConfig
from robot_sf.adversarial.materialize import (
    materialize_manifest_route_overrides,
    materialize_manifest_scenario_payload,
)
from robot_sf.adversarial.scenario_manifest import (
    AdversarialScenarioManifest,
    ManifestCategory,
    SourceLineage,
    generate_manifests,
    write_manifest_yaml,
)
from robot_sf.benchmark.fallback_policy import availability_payload, benchmark_run_exit_code
from robot_sf.benchmark.runner import run_batch

DEFAULT_SCHEMA = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
DEFAULT_PLANNERS = ("goal", "social_force")
SUMMARY_SCHEMA_VERSION = "adversarial_manifest_smoke_summary.v1"
GENERATION_SUMMARY_SCHEMA_VERSION = "adversarial_scenario_manifest_generation_summary.v1"
EVIDENCE_BOUNDARY = (
    "smoke-only: generated cases are uncategorized development stress tests; this run does not "
    "establish adversarial coverage, planner weakness, leaderboard standing, or paper-facing claims."
)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML payload must be a mapping: {path}")
    return payload


def _load_template_info(template_path: Path) -> SourceLineage:
    """Extract compact source lineage from a scenario template YAML."""

    raw = _load_yaml_mapping(template_path)
    scenarios = raw.get("scenarios", [])
    first: Any = scenarios[0] if isinstance(scenarios, list) and scenarios else None
    map_id: str | None = None
    scenario_name: str | None = None
    if isinstance(first, dict):
        map_id = str(first.get("map_id", "")) or None
        scenario_name = str(first.get("name", "")) or None
    return SourceLineage(
        scenario_template=template_path.name,
        config_path=str(template_path),
        map_id=map_id,
        scenario_name=scenario_name,
    )


def _generation_summary(
    source: SourceLineage,
    *,
    generator_family: str,
    seed: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Return the generator summary payload written beside raw manifests."""

    return {
        "schema_version": GENERATION_SUMMARY_SCHEMA_VERSION,
        "source": source.to_dict(),
        "generator": {
            "family": generator_family,
            "generator_id": "RandomCandidateSampler",
            "seed": int(seed),
        },
        **summary,
    }


def _write_manifest_batch(
    manifests: list[AdversarialScenarioManifest],
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Write generated manifests and return compact manifest metadata."""

    manifest_rows: list[dict[str, Any]] = []
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        index = manifest.generator.candidate_index if manifest.generator is not None else 0
        path = manifests_dir / f"candidate_{index:04d}.yaml"
        write_manifest_yaml(manifest, path)
        validation = manifest.validation
        manifest_rows.append(
            {
                "candidate_index": int(index),
                "path": path.as_posix(),
                "validation_status": validation.status.value
                if validation is not None
                else "unknown",
                "normalized_control_hash": validation.normalized_control_hash
                if validation is not None
                else None,
                "errors": list(validation.errors) if validation is not None else [],
                "warnings": list(validation.warnings) if validation is not None else [],
            }
        )
    return manifest_rows


def _valid_manifests(
    manifests: list[AdversarialScenarioManifest],
) -> list[AdversarialScenarioManifest]:
    """Return manifests that passed validation."""

    return [
        manifest
        for manifest in manifests
        if manifest.validation is not None and manifest.validation.status is ManifestCategory.VALID
    ]


def _materialize_matrix(
    manifests: list[AdversarialScenarioManifest],
    *,
    scenario_template: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Materialize selected manifests into one benchmark scenario matrix."""

    scenarios: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        candidate_index = (
            manifest.generator.candidate_index if manifest.generator is not None else 0
        )
        route_path = output_dir / "routes" / f"candidate_{candidate_index:04d}_route_overrides.yaml"
        route_path.parent.mkdir(parents=True, exist_ok=True)
        route_path.write_text(
            yaml.safe_dump(
                materialize_manifest_route_overrides(manifest),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        route_rel = route_path.relative_to(output_dir).as_posix()
        payload = materialize_manifest_scenario_payload(
            manifest,
            scenario_template,
            route_file_name=route_rel,
        )
        scenario = payload["scenarios"][0]
        scenarios.append(scenario)
        assert manifest.validation is not None
        rows.append(
            {
                "candidate_index": candidate_index,
                "scenario_name": scenario.get("name"),
                "scenario_seed": scenario.get("seeds", [None])[0],
                "normalized_control_hash": manifest.validation.normalized_control_hash,
                "route_overrides_path": route_path.as_posix(),
            }
        )

    matrix_path = output_dir / "materialized_matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump({"scenarios": scenarios}, sort_keys=False), encoding="utf-8"
    )
    return matrix_path, rows


def _read_episode_rows(path: Path) -> list[dict[str, Any]]:
    """Read benchmark episode JSONL rows if the planner wrote them."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _numeric_metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    """Collect finite numeric values for one metric from episode rows."""

    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or metric not in metrics:
            continue
        try:
            value = float(metrics[metric])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize stable smoke-relevant metrics from episode rows."""

    metrics: dict[str, Any] = {"episodes": len(rows)}
    for metric in (
        "success",
        "collisions",
        "total_collision_count",
        "pedestrian_collision_count",
        "wall_collisions",
        "time_to_goal_norm",
        "path_efficiency",
    ):
        values = _numeric_metric_values(rows, metric)
        if not values:
            continue
        metrics[metric] = {
            "mean": mean(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
        }
    statuses = sorted({str(row.get("status", "unknown")) for row in rows})
    if statuses:
        metrics["statuses"] = statuses
    termination_reasons = sorted({str(row.get("termination_reason", "unknown")) for row in rows})
    if termination_reasons:
        metrics["termination_reasons"] = termination_reasons
    return metrics


def _planner_command(
    *,
    matrix_path: Path,
    out_path: Path,
    schema_path: Path,
    planner: str,
    horizon: int,
    dt: float,
    workers: int,
) -> list[str]:
    """Return a replayable command-equivalent argv array for a planner smoke."""

    return [
        "uv",
        "run",
        "robot_sf_bench",
        "run",
        "--matrix",
        matrix_path.as_posix(),
        "--out",
        out_path.as_posix(),
        "--schema",
        schema_path.as_posix(),
        "--horizon",
        str(horizon),
        "--dt",
        str(dt),
        "--algo",
        planner,
        "--workers",
        str(workers),
        "--no-resume",
        "--no-video",
        "--video-renderer",
        "none",
        "--structured-output",
        "json",
    ]


def _run_planner_smoke(
    *,
    matrix_path: Path,
    schema_path: Path,
    output_dir: Path,
    planner: str,
    horizon: int,
    dt: float,
    workers: int,
) -> dict[str, Any]:
    """Run one planner against the materialized matrix and summarize the result."""

    planner_dir = output_dir / "planner_runs" / planner
    planner_dir.mkdir(parents=True, exist_ok=True)
    out_path = planner_dir / "episodes.jsonl"
    command = _planner_command(
        matrix_path=matrix_path,
        out_path=out_path,
        schema_path=schema_path,
        planner=planner,
        horizon=horizon,
        dt=dt,
        workers=workers,
    )
    try:
        summary = run_batch(
            scenarios_or_path=matrix_path,
            out_path=out_path,
            schema_path=schema_path,
            horizon=horizon,
            dt=dt,
            record_forces=False,
            video_enabled=False,
            video_renderer="none",
            append=False,
            fail_fast=False,
            algo=planner,
            benchmark_profile="baseline-safe",
            workers=workers,
            resume=False,
        )
        if "benchmark_availability" not in summary:
            summary["benchmark_availability"] = availability_payload(summary)
        exit_code = benchmark_run_exit_code(summary)
    except Exception as exc:  # pragma: no cover - defensive CLI path
        summary = {
            "status": "failed",
            "total_jobs": 0,
            "written": 0,
            "failed_jobs": 1,
            "failures": [{"planner": planner, "error": repr(exc)}],
        }
        summary["benchmark_availability"] = availability_payload(summary)
        exit_code = 2

    rows = _read_episode_rows(out_path)
    return {
        "planner": planner,
        "exit_code": int(exit_code),
        "command": command,
        "out_path": out_path.as_posix(),
        "total_jobs": int(summary.get("total_jobs", 0) or 0),
        "written": int(summary.get("written", 0) or 0),
        "failed_jobs": int(summary.get("failed_jobs", 0) or 0),
        "failures": summary.get("failures", []),
        "benchmark_availability": summary.get("benchmark_availability"),
        "metrics": _metric_summary(rows),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Generate manifests, materialize valid cases, run planners, and write summary JSON."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_space = SearchSpaceConfig.from_file(args.search_space)
    scenario_template = _load_yaml_mapping(args.scenario_template)
    source = replace(
        _load_template_info(args.scenario_template),
        search_space=args.search_space.name,
        search_space_path=str(args.search_space),
    )

    manifests, generation_counts = generate_manifests(
        search_space,
        seed=args.seed,
        count=args.count,
        source=source,
        generator_family=args.generator_family,
    )
    generation = _generation_summary(
        source,
        generator_family=args.generator_family,
        seed=args.seed,
        summary=generation_counts,
    )
    (output_dir / "generation_summary.json").write_text(
        json.dumps(generation, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_rows = _write_manifest_batch(manifests, output_dir=output_dir)
    selected = _valid_manifests(manifests)[: args.max_valid]

    planner_runs: list[dict[str, Any]] = []
    matrix_path: Path | None = None
    materialized_rows: list[dict[str, Any]] = []
    if selected:
        matrix_path, materialized_rows = _materialize_matrix(
            selected,
            scenario_template=scenario_template,
            output_dir=output_dir,
        )
        for planner in args.planner:
            planner_runs.append(
                _run_planner_smoke(
                    matrix_path=matrix_path,
                    schema_path=args.schema,
                    output_dir=output_dir,
                    planner=planner,
                    horizon=args.horizon,
                    dt=args.dt,
                    workers=args.workers,
                )
            )

    result_classification = _classify_smoke_result(generation, selected, planner_runs)
    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "result_classification": result_classification,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "source": source.to_dict(),
        "generation": generation,
        "manifests": manifest_rows,
        "materialized": {
            "matrix_path": matrix_path.as_posix() if matrix_path is not None else None,
            "selected_valid_candidates": materialized_rows,
        },
        "planner_runs": planner_runs,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _classify_smoke_result(
    generation: dict[str, Any],
    selected: list[AdversarialScenarioManifest],
    planner_runs: list[dict[str, Any]],
) -> str:
    """Classify the smoke result without making benchmark-strength claims."""

    if int(generation.get("valid", 0) or 0) <= 0 or not selected:
        return "no_valid_manifests"
    if not planner_runs:
        return "not_run"
    if all(int(run.get("exit_code", 2)) == 0 for run in planner_runs):
        return "smoke_passed"
    if any(int(run.get("written", 0) or 0) > 0 for run in planner_runs):
        return "partial_smoke"
    return "smoke_failed"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--scenario-template", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-valid", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--generator-family", default="random")
    parser.add_argument("--planner", action="append", default=None)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.max_valid < 1:
        parser.error("--max-valid must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.planner is None:
        args.planner = list(DEFAULT_PLANNERS)
    payload = run_smoke(args)
    print(json.dumps({"summary_json": args.summary_json.as_posix(), **payload}, sort_keys=True))
    return 0 if payload["result_classification"] in {"smoke_passed", "partial_smoke"} else 2


if __name__ == "__main__":
    sys.exit(main())
