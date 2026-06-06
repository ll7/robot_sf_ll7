#!/usr/bin/env python3
"""Trace static-recenter activation on the Issue #2221 held-out smoke rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.map_runner import _run_map_episode
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_policy_search_candidate import (
    _effective_candidate_runtime_for_scenario,
    _filter_scenarios,
    _load_yaml,
    load_candidate_definition,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FUNNEL = Path(
    "configs/policy_search/transfer/issue_2221_static_recenter_heldout_smoke.yaml"
)
DEFAULT_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--funnel-config", type=Path, default=DEFAULT_FUNNEL)
    parser.add_argument("--candidate-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--baseline-candidate", default="hybrid_rule_v3_fast_progress")
    parser.add_argument("--mechanism-candidate", default="issue_2170_static_recenter_only")
    parser.add_argument("--stage", default="full_matrix")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _scenario_id(scenario: dict[str, Any]) -> str:
    """Return the stable scenario identifier."""
    return str(scenario.get("name") or scenario.get("scenario_id") or scenario.get("id"))


def _load_stage(funnel_config: Path, stage_name: str) -> dict[str, Any]:
    """Load one stage config from a policy-search funnel."""
    payload = _load_yaml(funnel_config)
    stages = payload.get("stages")
    if not isinstance(stages, dict) or stage_name not in stages:
        raise KeyError(f"Unknown stage '{stage_name}' in {funnel_config}")
    stage = stages[stage_name]
    if not isinstance(stage, dict):
        raise TypeError(f"Stage config must be a mapping: {stage_name}")
    return stage


def _trace_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the planner decision trace steps from a map-runner record."""
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    trace = metadata.get("planner_decision_trace")
    trace = trace if isinstance(trace, dict) else {}
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def _xy(step: dict[str, Any]) -> np.ndarray:
    """Return a step robot xy vector."""
    return np.array([float(step.get("robot_x_m", 0.0)), float(step.get("robot_y_m", 0.0))])


def _episode_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Return compact terminal fields for one record."""
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return {
        "status": record.get("status"),
        "termination_reason": record.get("termination_reason"),
        "success": bool(metrics.get("success", False)),
        "steps": int(record.get("steps", 0)),
        "near_misses": int(metrics.get("near_misses", 0) or 0),
        "collisions": int(metrics.get("collisions", 0) or 0),
    }


def _activation_summary(
    *,
    baseline_record: dict[str, Any],
    mechanism_record: dict[str, Any],
) -> dict[str, Any]:
    """Compare one baseline/mechanism record pair."""
    baseline_trace = _trace_steps(baseline_record)
    mechanism_trace = _trace_steps(mechanism_record)
    activated = [step for step in mechanism_trace if float(step.get("static_recenter", 0.0)) > 0.0]
    first_activation_step = int(activated[0]["step"]) if activated else None
    selected_sources = sorted({str(step.get("selected_source", "unknown")) for step in activated})
    source_changed = None
    if baseline_trace and mechanism_trace:
        limit = min(len(baseline_trace), len(mechanism_trace))
        source_changed = any(
            baseline_trace[idx].get("selected_source")
            != mechanism_trace[idx].get("selected_source")
            for idx in range(limit)
        )
    progress_delta_after_activation = None
    if first_activation_step is not None and first_activation_step < len(mechanism_trace):
        baseline_at = (
            baseline_trace[first_activation_step]
            if first_activation_step < len(baseline_trace)
            else {}
        )
        mechanism_at = mechanism_trace[first_activation_step]
        baseline_after = (
            float(baseline_trace[-1].get("route_progress_from_start_m", 0.0))
            - float(baseline_at.get("route_progress_from_start_m", 0.0))
            if baseline_trace
            else None
        )
        mechanism_after = float(
            mechanism_trace[-1].get("route_progress_from_start_m", 0.0)
        ) - float(mechanism_at.get("route_progress_from_start_m", 0.0))
        progress_delta_after_activation = {
            "baseline_m": baseline_after,
            "mechanism_m": mechanism_after,
            "mechanism_minus_baseline_m": (
                mechanism_after - baseline_after if baseline_after is not None else None
            ),
        }
    trajectory_delta = None
    if baseline_trace and mechanism_trace:
        trajectory_delta = float(np.linalg.norm(_xy(mechanism_trace[-1]) - _xy(baseline_trace[-1])))

    baseline_terminal = _episode_summary(baseline_record)
    mechanism_terminal = _episode_summary(mechanism_record)
    terminal_changed = baseline_terminal != mechanism_terminal
    if baseline_terminal["success"] and mechanism_terminal["success"]:
        classification = "comparator_already_solved_case"
    elif not activated:
        classification = "mechanism_inactive"
    elif not terminal_changed:
        classification = "mechanism_active_but_irrelevant"
    else:
        classification = "mechanism_active_but_irrelevant"

    return {
        "scenario_id": baseline_record.get("scenario_id"),
        "seed": baseline_record.get("seed"),
        "baseline": baseline_terminal,
        "mechanism": mechanism_terminal,
        "activation_count": len(activated),
        "first_activation_step": first_activation_step,
        "selected_command_source": selected_sources,
        "command_source_changed": source_changed,
        "progress_delta_after_activation": progress_delta_after_activation,
        "trajectory_delta_m": trajectory_delta,
        "terminal_outcome_changed": terminal_changed,
        "classification": classification,
    }


def _run_candidate(
    *,
    candidate_name: str,
    registry_path: Path,
    scenarios: list[dict[str, Any]],
    stage_matrix: Path,
    seed_list: list[int],
    horizon: int,
    dt: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Run one candidate with planner-decision tracing enabled."""
    _entry, candidate_payload, candidate_config, config_path = load_candidate_definition(
        registry_path,
        candidate_name,
    )
    default_algo = str(candidate_payload.get("algo", candidate_name)).strip().lower()
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for scenario in scenarios:
        algo, runtime_config = _effective_candidate_runtime_for_scenario(
            candidate_payload,
            candidate_config,
            scenario,
            default_algo=default_algo,
            config_anchor=config_path.parent,
        )
        for seed in seed_list:
            record = _run_map_episode(
                scenario,
                int(seed),
                horizon=horizon,
                dt=dt,
                record_forces=True,
                snqi_weights=None,
                snqi_baseline=None,
                algo=algo,
                algo_config=runtime_config,
                scenario_path=stage_matrix,
                record_planner_decision_trace=True,
            )
            records[(_scenario_id(scenario), int(seed))] = record
    return records


def main() -> None:
    """Run the static-recenter activation trace and write compact JSON."""
    args = parse_args()
    stage = _load_stage(args.funnel_config, args.stage)
    stage_matrix = Path(stage["scenario_matrix"])
    scenarios = [dict(item) for item in load_scenarios(stage_matrix)]
    scenario_filter = stage.get("scenario_filter")
    scenarios = _filter_scenarios(
        scenarios,
        [str(item) for item in scenario_filter] if isinstance(scenario_filter, list) else None,
    )
    seed_list = [int(seed) for seed in stage.get("seed_list", [111])]
    horizon = int(stage.get("horizon", 500))
    dt = float(stage.get("dt", 0.1))

    baseline_records = _run_candidate(
        candidate_name=args.baseline_candidate,
        registry_path=args.candidate_registry,
        scenarios=scenarios,
        stage_matrix=stage_matrix,
        seed_list=seed_list,
        horizon=horizon,
        dt=dt,
    )
    mechanism_records = _run_candidate(
        candidate_name=args.mechanism_candidate,
        registry_path=args.candidate_registry,
        scenarios=scenarios,
        stage_matrix=stage_matrix,
        seed_list=seed_list,
        horizon=horizon,
        dt=dt,
    )
    rows = [
        _activation_summary(
            baseline_record=baseline_records[key],
            mechanism_record=mechanism_records[key],
        )
        for key in sorted(baseline_records)
    ]
    payload = {
        "schema_version": "static-recenter-activation-trace.v1",
        "stage": args.stage,
        "scenario_matrix": stage_matrix.as_posix(),
        "seed_list": seed_list,
        "horizon": horizon,
        "dt": dt,
        "baseline_candidate": args.baseline_candidate,
        "mechanism_candidate": args.mechanism_candidate,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"output_json": args.output_json.as_posix(), "rows": len(rows)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
