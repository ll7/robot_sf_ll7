#!/usr/bin/env python
"""Run the issue #4163 rare-event importance-sampling smoke harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.full_classic.planning import load_scenario_matrix
from robot_sf.benchmark.rare_event_sampling import (
    RareEventSamplingSpec,
    apply_sampled_scenario_mutation,
    build_sampling_summary,
    sample_scenario_rows,
)

DEFAULT_OUTPUT_DIR = Path("output/benchmarks/issue_4163/crossing_smoke")
DEFAULT_EVIDENCE_DIR = Path("docs/context/evidence/issue_4163_rare_event_smoke")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/rare_event/issue_4163_crossing_smoke.yaml"),
        help="Rare-event sampling config.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the sampling smoke and write JSONL plus compact evidence summary."""

    args = build_parser().parse_args(argv)
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    spec = RareEventSamplingSpec.from_payload(payload)
    rows = sample_scenario_rows(spec)
    scenario_payload = _load_smoke_scenario(payload)
    mutated_scenario = apply_sampled_scenario_mutation(scenario_payload, rows[0])
    events = _evaluate_synthetic_toy_model(payload, rows)
    summary = build_sampling_summary(
        spec=spec,
        rows=rows,
        events=events,
        scenario_payload=mutated_scenario,
    )
    summary["config_path"] = str(args.config)
    summary["scenario_matrix"] = payload.get("scenario_matrix")
    summary["smoke_runner"] = "synthetic_toy_model"
    summary["out_of_scope"] = [
        "no full benchmark campaign run",
        "no Slurm/GPU submission",
        "no paper/dissertation claim edits",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for row, event in zip(rows, events, strict=True):
            handle.write(
                json.dumps(
                    {
                        "sample": row.to_payload(),
                        "objective_event_observed": bool(event),
                        "scenario": {
                            "name": mutated_scenario.get("name"),
                            "parameter_vector_hash": row.parameter_vector_hash,
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.evidence_dir / "summary.json"
    summary["episodes_jsonl"] = str(episodes_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "episodes_jsonl": str(episodes_path)}))
    return 0


def _load_smoke_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    scenario_matrix = payload.get("scenario_matrix")
    if not scenario_matrix:
        return {"name": "synthetic_smoke", "simulation_config": {}, "metadata": {}}
    try:
        scenarios = load_scenario_matrix(str(scenario_matrix))
    except ValueError:
        scenarios = _load_raw_smoke_scenarios(Path(str(scenario_matrix)))
        if not isinstance(scenarios, list) or not scenarios:
            raise
    scenario_index = int(payload.get("smoke_scenario_index", 0))
    return dict(scenarios[scenario_index])


def _load_raw_smoke_scenarios(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return []
    scenarios = raw.get("scenarios")
    if isinstance(scenarios, list):
        return list(scenarios)
    includes = raw.get("include") or raw.get("includes") or []
    loaded: list[dict[str, Any]] = []
    for include in includes:
        loaded.extend(_load_raw_smoke_scenarios((path.parent / str(include)).resolve()))
    return loaded


def _evaluate_synthetic_toy_model(payload: dict[str, Any], rows: list[Any]) -> list[bool]:
    toy = payload.get("synthetic_toy_model") or {}
    parameter = str(toy.get("parameter", "ped_density"))
    threshold = float(toy.get("threshold", 0.065))
    return [float(row.parameters[parameter]) >= threshold for row in rows]


if __name__ == "__main__":
    raise SystemExit(main())
