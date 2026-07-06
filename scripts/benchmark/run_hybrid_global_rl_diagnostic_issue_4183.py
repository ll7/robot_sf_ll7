#!/usr/bin/env python3
"""Run and rebuild the issue #4183 hybrid_global_rl diagnostic packet."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.hybrid_global_rl_diagnostic import (
    BASELINE_ARM,
    ROUTE_ARM,
    build_diagnostic_report,
    load_diagnostic_config,
    load_jsonl_records,
    preflight_configs,
)
from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.models.registry import resolve_model_path

DEFAULT_ROUTE_CONFIG = Path("configs/benchmarks/issue_4183_hybrid_global_rl_route_conditioned.yaml")
DEFAULT_BASELINE_CONFIG = Path("configs/benchmarks/issue_4183_learned_local_unconditioned.yaml")
# Use the canonical benchmark episode schema (as every other map-runner entry point does). The
# previous per-issue pointer to the stale strict schema rejected native PPO episode records with
# additionalProperties errors and left the baseline arm without valid rows.
DEFAULT_SCHEMA = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_4183_hybrid_global_rl_diagnostic")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE_CONFIG)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("output/issue_4183_hybrid_global_rl_run")
    )
    parser.add_argument("--horizon", type=int, help="Optional smoke horizon override.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--no-hydrate",
        action="store_true",
        help=(
            "Skip public-release hydration of the promoted learned checkpoint. Use in offline/CI "
            "contexts where the model cache is already populated."
        ),
    )
    return parser.parse_args(argv)


def _hydrate_checkpoints(*config_paths: Path) -> list[str]:
    """Hydrate promoted learned checkpoints referenced by the paired configs.

    Downloads the public benchmark-promoted checkpoint into the canonical model cache when a
    ``learned_policy_model_id`` is declared, so the download-free preflight can find it.

    Returns:
        list[str]: Human-readable notes describing each hydration attempt.
    """

    notes: list[str] = []
    seen: set[str] = set()
    for config_path in config_paths:
        model_id = load_diagnostic_config(config_path).learned_policy_model_id
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        resolved = resolve_model_path(model_id, allow_download=True)
        notes.append(f"hydrated {model_id} -> {resolved}")
    return notes


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping.")
    return payload


def _fixed_list_seeds(config_payload: dict[str, Any], *, path: Path) -> list[int]:
    seed_policy = config_payload.get("seed_policy") or {}
    if not isinstance(seed_policy, dict) or seed_policy.get("mode") != "fixed-list":
        raise ValueError(f"{path} must declare seed_policy.mode=fixed-list.")
    seeds = seed_policy.get("seeds") or []
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{path} must declare a non-empty fixed seed list.")
    return [int(seed) for seed in seeds]


def _scenario_payload_with_config_seeds(
    config_payload: dict[str, Any], *, path: Path
) -> list[dict[str, Any]]:
    scenario_matrix = Path(str(config_payload["scenario_matrix"]))
    matrix_payload = _load_yaml(scenario_matrix)
    scenarios = matrix_payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"{scenario_matrix} must declare a non-empty scenarios list.")
    seeds = _fixed_list_seeds(config_payload, path=path)
    resolved: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError(f"{scenario_matrix} contains a non-mapping scenario entry.")
        scenario_copy = dict(scenario)
        scenario_copy["seeds"] = list(seeds)
        resolved.append(scenario_copy)
    return resolved


def _planner_spec(config_payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    planners = config_payload.get("planners")
    if not isinstance(planners, list) or len(planners) != 1 or not isinstance(planners[0], dict):
        raise ValueError(f"{path} must declare exactly one planner mapping.")
    return planners[0]


def _write_algo_config(planner: dict[str, Any], *, directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}_algo.yaml"
    path.write_text(yaml.safe_dump(planner.get("config") or {}, sort_keys=True), encoding="utf-8")
    return path


def _failure_rows(summary: dict[str, Any], *, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for failure in summary.get("failures", []):
        if not isinstance(failure, dict):
            continue
        rows.append(
            {
                "arm": arm,
                "scenario_id": failure.get("scenario_id"),
                "seed": failure.get("seed"),
                "reason": failure.get("error"),
                "row_classification": "fail_closed_no_episode_row",
                "source": "issue_4183_paired_runner",
            }
        )
    return rows


def _run_arm(
    *,
    config_path: Path,
    arm: str,
    work_dir: Path,
    schema_path: Path,
    horizon_override: int | None,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    config_payload = _load_yaml(config_path)
    planner = _planner_spec(config_payload, path=config_path)
    scenarios = _scenario_payload_with_config_seeds(config_payload, path=config_path)
    arm_work_dir = work_dir / arm
    algo_config_path = _write_algo_config(planner, directory=arm_work_dir, stem=config_path.stem)
    episodes_path = arm_work_dir / "episodes.jsonl"
    episodes_path.unlink(missing_ok=True)
    horizon = horizon_override if horizon_override is not None else int(config_payload["horizon"])
    summary = run_map_batch(
        scenarios,
        episodes_path,
        schema_path,
        scenario_path=config_payload["scenario_matrix"],
        horizon=horizon,
        dt=float(config_payload["dt"]),
        record_forces=bool(config_payload.get("record_forces", True)),
        algo=str(planner["algo"]),
        algo_config_path=str(algo_config_path),
        benchmark_profile=str(planner.get("benchmark_profile", "diagnostic-only")),
        workers=int(config_payload.get("workers", 1)),
        resume=False,
    )
    return episodes_path, summary, _failure_rows(summary, arm=arm)


def main(argv: list[str] | None = None) -> int:
    """Run both issue #4183 arms and rebuild the diagnostic packet."""
    args = parse_args(argv)
    hydration_notes: list[str] = []
    if not args.no_hydrate:
        hydration_notes = _hydrate_checkpoints(args.route_config, args.baseline_config)
    preflight = preflight_configs(args.route_config, args.baseline_config, repo_root=args.repo_root)
    if preflight["status"] != "valid":
        print(
            json.dumps(
                {"preflight": preflight, "hydration": hydration_notes}, indent=2, sort_keys=True
            )
        )
        return 2

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="issue_4183_", dir=args.work_dir) as tmp:
        run_dir = Path(tmp)
        route_episodes, route_summary, route_failures = _run_arm(
            config_path=args.route_config,
            arm=ROUTE_ARM,
            work_dir=run_dir,
            schema_path=args.schema,
            horizon_override=args.horizon,
        )
        baseline_episodes, baseline_summary, baseline_failures = _run_arm(
            config_path=args.baseline_config,
            arm=BASELINE_ARM,
            work_dir=run_dir,
            schema_path=args.schema,
            horizon_override=args.horizon,
        )
        summary = build_diagnostic_report(
            route_records=load_jsonl_records(route_episodes),
            baseline_records=load_jsonl_records(baseline_episodes),
            route_config_path=args.route_config,
            baseline_config_path=args.baseline_config,
            output_dir=args.output_dir,
            run_failures=[*route_failures, *baseline_failures],
        )
    print(
        json.dumps(
            {
                "route_run": route_summary,
                "baseline_run": baseline_summary,
                "diagnostic_summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["included_diagnostic_rows"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
