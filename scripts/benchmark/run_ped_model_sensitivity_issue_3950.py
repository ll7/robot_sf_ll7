"""Run the issue #3950 CPU pedestrian-model sensitivity smoke harness."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.ped_model_sensitivity import (
    DEFAULT_MODELS,
    build_sensitivity_summary,
    load_jsonl_records,
    write_sensitivity_report,
)
from robot_sf.benchmark.runner import load_scenario_matrix
from robot_sf.sim.pedestrian_model_variants import normalize_pedestrian_model

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Smoke harness YAML config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Report output directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run configured sensitivity cells and write compact report artifacts."""

    args = parse_args(argv)
    config = _load_config(args.config)
    output_dir = _resolve_path(args.output_dir)
    run_dir = _resolve_path(
        config.get(
            "run_artifact_dir",
            "output/benchmarks/issue_3950_ped_model_sensitivity_runs",
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    development_models = _models_from_config(config, "development_models")
    evaluation_models = _models_from_config(config, "evaluation_models")
    planner = dict(config.get("planner") or {})
    planner_key = str(planner.get("key", "goal"))
    algo = str(planner.get("algo", "goal"))
    scenario_matrix = _resolve_path(config["scenario_matrix"])
    base_scenarios = _resolve_loaded_scenario_paths(
        load_scenario_matrix(scenario_matrix),
        base_dir=scenario_matrix.parent,
    )

    records: list[dict[str, Any]] = []
    for development_model in development_models:
        for evaluation_model in evaluation_models:
            cell_scenarios = _scenario_set_for_evaluation_model(base_scenarios, evaluation_model)
            algo_config_path = _write_cell_algo_config(
                run_dir=run_dir,
                development_model=development_model,
                planner=planner,
            )
            result_path = run_dir / f"{development_model}__{evaluation_model}.jsonl"
            run_map_batch(
                cell_scenarios,
                result_path,
                "robot_sf/benchmark/schemas/episode.schema.v1.json",
                horizon=int(config.get("horizon", 30)),
                dt=float(config.get("dt", 0.1)),
                record_forces=bool(config.get("record_forces", False)),
                algo=algo,
                algo_config_path=str(algo_config_path),
                benchmark_profile=str(planner.get("benchmark_profile", "baseline-safe")),
                workers=int(config.get("workers", 1)),
                resume=bool(config.get("resume", False)),
            )
            records.extend(load_jsonl_records(result_path))

    summary = build_sensitivity_summary(
        records,
        development_models=development_models,
        evaluation_models=evaluation_models,
        planner_key=planner_key,
        algo=algo,
    )
    write_sensitivity_report(summary, output_dir)
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    config_path = _resolve_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping config in {config_path}")
    if "scenario_matrix" not in payload:
        raise ValueError("Pedestrian-model sensitivity config requires 'scenario_matrix'")
    return payload


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _models_from_config(config: dict[str, Any], key: str) -> list[str]:
    raw = config.get(key, list(DEFAULT_MODELS))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{key} must be a non-empty list")
    return [normalize_pedestrian_model(str(model)) for model in raw]


def _scenario_set_for_evaluation_model(
    base_scenarios: list[dict[str, Any]],
    evaluation_model: str,
) -> list[dict[str, Any]]:
    scenarios = deepcopy(base_scenarios)
    for scenario in scenarios:
        sim_config = scenario.setdefault("simulation_config", {})
        if not isinstance(sim_config, dict):
            raise TypeError("scenario simulation_config must be a mapping when present")
        sim_config["pedestrian_model"] = evaluation_model
    return scenarios


def _resolve_loaded_scenario_paths(
    scenarios: list[dict[str, Any]],
    *,
    base_dir: Path,
) -> list[dict[str, Any]]:
    """Resolve scenario-local paths before passing mutated dicts to ``run_map_batch``."""

    resolved = deepcopy(scenarios)
    for scenario in resolved:
        raw_map_file = scenario.get("map_file")
        if isinstance(raw_map_file, str) and raw_map_file.strip():
            map_path = Path(raw_map_file)
            if not map_path.is_absolute():
                candidate = (base_dir / map_path).resolve()
                if candidate.exists():
                    scenario["map_file"] = str(candidate)
    return resolved


def _write_cell_algo_config(
    *,
    run_dir: Path,
    development_model: str,
    planner: dict[str, Any],
) -> Path:
    payload = dict(planner.get("algo_config") or {})
    payload["development_pedestrian_model"] = development_model
    path = run_dir / f"algo_{development_model}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
