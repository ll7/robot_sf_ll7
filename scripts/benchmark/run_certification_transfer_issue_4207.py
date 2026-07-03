"""Run the issue #4207 CPU certification-transfer probe."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.certification_transfer import (
    build_certification_transfer_report,
    load_yaml_mapping,
    validate_gate_spec,
    validate_probe_config,
    write_certification_transfer_evidence,
)
from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.runner import load_scenario_matrix
from robot_sf.sim.pedestrian_model_variants import normalize_pedestrian_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Probe YAML config.")
    parser.add_argument("--gate-spec", type=Path, required=True, help="Release-gate YAML spec.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Compact evidence directory."
    )
    parser.add_argument(
        "--generated-at",
        default="now",
        help="UTC timestamp for evidence metadata, or 'now'.",
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        help="Existing episode JSONL input for report-only mode; skips CPU episode execution.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run or summarize the CPU certification-transfer probe."""

    args = parse_args(argv)
    config_path = _resolve(args.config)
    gate_spec_path = _resolve(args.gate_spec)
    output_dir = _resolve(args.output_dir, must_exist=False)
    config = load_yaml_mapping(config_path)
    probe_config = validate_probe_config(config, base_dir=config_path.parent)
    gate_spec = load_yaml_mapping(gate_spec_path)
    validate_gate_spec(gate_spec, scenario_family=probe_config["scenario_family"])

    generated_at = _generated_at(args.generated_at)
    if args.episodes_jsonl is None:
        records = _run_probe(config_path=config_path, probe_config=probe_config)
    else:
        records = _load_jsonl_records(_resolve(args.episodes_jsonl))

    report = build_certification_transfer_report(
        records,
        probe_config=config,
        gate_spec=gate_spec,
        config_path=config_path,
        gate_spec_path=gate_spec_path,
        generated_at_utc=generated_at,
    )
    paths = write_certification_transfer_evidence(report, output_dir)
    print(
        json.dumps(
            {"status": "ok", "output_dir": str(output_dir), "artifacts": paths}, sort_keys=True
        )
    )
    return 0


def _run_probe(*, config_path: Path, probe_config: dict[str, Any]) -> list[dict[str, Any]]:
    run_root = _resolve(probe_config["run_artifact_dir"], must_exist=False)
    run_root.mkdir(parents=True, exist_ok=True)
    scenario_matrix_path = Path(probe_config["scenario_matrix"])
    scenarios = load_scenario_matrix(scenario_matrix_path)
    records: list[dict[str, Any]] = []
    for arm in probe_config["arms"]:
        for evaluation_model in probe_config["pedestrian_models"]:
            model = normalize_pedestrian_model(evaluation_model)
            cell_scenarios = _prepare_scenarios(
                scenarios,
                base_dir=scenario_matrix_path.parent,
                evaluation_model=model,
                scenario_family=probe_config["scenario_family"],
                seeds=probe_config["seed_policy"]["seeds"],
            )
            cell_dir = run_root / arm["key"] / model
            cell_dir.mkdir(parents=True, exist_ok=True)
            episodes_path = cell_dir / "episodes.jsonl"
            run_map_batch(
                cell_scenarios,
                episodes_path,
                DEFAULT_SCHEMA_PATH,
                horizon=probe_config["horizon"],
                dt=probe_config["dt"],
                record_forces=probe_config["record_forces"],
                algo=arm["algo"],
                algo_config_path=arm.get("algo_config"),
                benchmark_profile=arm["benchmark_profile"],
                observation_mode=arm.get("observation_mode"),
                observation_level=arm.get("observation_level"),
                workers=probe_config["workers"],
                resume=probe_config["resume"],
            )
            for record in _load_jsonl_records(episodes_path):
                enriched = dict(record)
                enriched["planner_key"] = arm["key"]
                enriched["structural_class"] = arm["structural_class"]
                enriched["scenario_family"] = probe_config["scenario_family"]
                enriched["certification_pedestrian_model"] = model
                enriched["evaluation_pedestrian_model"] = model
                enriched["development_pedestrian_model"] = arm["development_pedestrian_model"]
                records.append(enriched)
    return records


def _prepare_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    base_dir: Path,
    evaluation_model: str,
    scenario_family: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    prepared = deepcopy(scenarios)
    for scenario in prepared:
        scenario["scenario_family"] = scenario_family
        scenario["seeds"] = list(seeds)
        raw_map_file = scenario.get("map_file")
        if isinstance(raw_map_file, str) and raw_map_file.strip():
            map_path = Path(raw_map_file)
            if not map_path.is_absolute():
                candidate = (base_dir / map_path).resolve()
                if candidate.exists():
                    scenario["map_file"] = str(candidate)
        sim_config = dict(scenario.get("simulation_config") or {})
        sim_config["pedestrian_model"] = evaluation_model
        scenario["simulation_config"] = sim_config
    return prepared


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _generated_at(raw: str) -> str:
    if raw == "now":
        return datetime.now(UTC).isoformat()
    return raw


def _resolve(path: str | Path, *, must_exist: bool = True) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
