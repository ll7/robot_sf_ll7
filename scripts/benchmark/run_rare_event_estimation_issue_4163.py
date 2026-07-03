#!/usr/bin/env python
"""Run issue #4163 rare-event importance-sampling smoke harness."""

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
    estimate_failure_probability,
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
    """Run smoke JSONL plus compact evidence summary."""

    args = build_parser().parse_args(argv)
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    spec = RareEventSamplingSpec.from_payload(payload)
    rows = sample_scenario_rows(spec)
    scenario_payload = _load_smoke_scenario(payload)
    mutated_scenario = apply_sampled_scenario_mutation(scenario_payload, rows[0])
    planner_arms = _planner_arms(payload)
    events_by_arm = {
        arm_key: _evaluate_synthetic_toy_model(payload, rows, arm_key=arm_key)
        for arm_key in planner_arms
    }

    primary_arm = planner_arms[0]
    summary = build_sampling_summary(
        spec=spec,
        rows=rows,
        events=events_by_arm[primary_arm],
        scenario_payload=mutated_scenario,
    )
    summary["config_path"] = str(args.config)
    summary["scenario_matrix"] = payload.get("scenario_matrix")
    summary["smoke_runner"] = "synthetic_toy_model"
    summary["planner_arms"] = planner_arms
    summary["episode_budget"] = len(rows) * len(planner_arms)
    summary["arm_estimates"] = {
        arm_key: estimate_failure_probability(
            rows,
            events,
            objective_event=spec.objective_event,
        ).to_payload()
        for arm_key, events in events_by_arm.items()
    }
    static_family = _static_constriction_family(payload)
    if static_family:
        summary["static_constriction_family"] = static_family
    summary["out_of_scope"] = [
        "no full benchmark campaign run",
        "no Slurm/GPU submission",
        "no paper/dissertation claim edits",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            for arm_key in planner_arms:
                handle.write(
                    json.dumps(
                        {
                            "planner_arm": arm_key,
                            "sample": row.to_payload(),
                            "objective_event_observed": bool(events_by_arm[arm_key][idx]),
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


def _planner_arms(payload: dict[str, Any]) -> list[str]:
    arms = payload.get("planner_arms")
    if arms is None:
        arms = ["synthetic_planner"]
    if not isinstance(arms, list) or not arms:
        raise ValueError("planner_arms must be a non-empty list when provided")
    return [str(arm) for arm in arms]


def _static_constriction_family(payload: dict[str, Any]) -> dict[str, Any] | None:
    family = payload.get("static_constriction_family")
    if family is None:
        return None
    if not isinstance(family, dict):
        raise ValueError("static_constriction_family must be a mapping")
    return {
        "family_id": family.get("family_id", "static_constriction"),
        "source_contract": family.get(
            "source_contract",
            "configs/benchmarks/issue_4205_static_constriction_codesign_loop_v1.yaml",
        ),
        "scenario_ids": list(family.get("scenario_ids") or []),
        "claim_boundary": family.get(
            "claim_boundary",
            "diagnostic rare-event application only; not a failure-rate claim",
        ),
    }


def _load_smoke_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    scenario_matrix = payload.get("scenario_matrix")
    if not scenario_matrix:
        return {"name": "synthetic_smoke", "simulation_config": {}, "metadata": {}}
    try:
        scenarios = load_scenario_matrix(str(scenario_matrix))
    except ValueError:
        scenarios = _load_raw_smoke_scenarios(Path(str(scenario_matrix)))
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"scenario_matrix produced no scenarios: {scenario_matrix}")
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


def _evaluate_synthetic_toy_model(
    payload: dict[str, Any],
    rows: list[Any],
    *,
    arm_key: str,
) -> list[bool]:
    toy = payload.get("synthetic_toy_model") or {}
    parameter = str(toy.get("parameter", "ped_density"))
    threshold = float(toy.get("threshold", 0.065))
    arm_adjustments = toy.get("arm_threshold_adjustments") or {}
    adjusted_threshold = threshold + float(arm_adjustments.get(arm_key, 0.0))
    return [float(row.parameters[parameter]) >= adjusted_threshold for row in rows]


if __name__ == "__main__":
    raise SystemExit(main())
