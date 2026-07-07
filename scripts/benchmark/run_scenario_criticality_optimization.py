#!/usr/bin/env python3
"""Run classical optimization for scenario criticality search (Issue #4362).

This script implements a bounded classical-optimization lane for scenario
criticality search as a baseline for learned/adaptive stress-testing approaches.

The first version uses random search over a small parameter space with explicit
invalid-run handling and reproducible artifact generation.

Claim boundary: exploratory/diagnostic-only; not a validated benchmark method.

Example:
    uv run python scripts/benchmark/run_scenario_criticality_optimization.py \\
      --config configs/benchmarks/issue_4362_scenario_criticality_smoke.yaml \\
      --output-dir output/issue_4362_scenario_criticality_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.scenario_criticality_objective import (
    CriticalityObjectiveConfig,
    apply_criticality_parameters,
    compute_criticality_score,
)
from robot_sf.training.scenario_loader import load_scenarios


@dataclass
class ParameterDefinition:
    """Definition of a single optimization parameter.

    Attributes:
        param_type: "continuous" or "discrete"
        min: Minimum value (for continuous)
        max: Maximum value (for continuous)
        values: List of discrete values (for discrete)
    """

    param_type: str
    min: float | None = None
    max: float | None = None
    values: list[float] | None = None


@dataclass
class OptimizationConfig:
    """Configuration for criticality optimization run.

    Attributes:
        parameter_space: Dict of parameter name to ParameterDefinition
        optimizer_type: "random_search" (currently only supported)
        sample_budget: Number of candidates to evaluate
        optimizer_seed: Random seed for reproducibility
        scenario_family: Path to scenario YAML or list of scenario dicts
        planner_name: Planner to use for evaluation
        seeds: List of evaluation seeds per candidate
        objective_weights: Weights for criticality objective
        invalid_run_policy: "fail_closed" or "log_continue"
        diagnostic_only: Must be True to acknowledge exploratory status
        horizon: Simulation horizon in steps
        dt: Simulation timestep in seconds
        max_workers: Max parallel candidates (1 = sequential, 0 = auto)
    """

    parameter_space: dict[str, ParameterDefinition]
    optimizer_type: str = "random_search"
    sample_budget: int = 20
    optimizer_seed: int = 4362
    scenario_family: str | list[dict[str, Any]] | None = None
    planner_name: str = "safe_baseline"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    objective_weights: dict[str, float] = field(default_factory=dict)
    invalid_run_policy: str = "fail_closed"
    diagnostic_only: bool = True
    horizon: int = 80
    dt: float = 0.1
    max_workers: int = 1


@dataclass
class CandidateResult:
    """Result for a single optimization candidate.

    Attributes:
        candidate_id: Unique identifier
        parameters: Parameter values
        criticality_score: Mean score across seeds
        score_decomposition: Dict of term contributions
        status: "evaluated", "invalid_candidate", "not_evaluable"
        reason: Optional failure reason
        per_seed_scores: List of (seed, score, status) tuples
        runtime_s: Optional runtime in seconds
    """

    candidate_id: str
    parameters: dict[str, float]
    criticality_score: float
    score_decomposition: dict[str, float]
    status: str
    reason: str | None = None
    per_seed_scores: list[tuple[int, float, str]] = field(default_factory=list)
    runtime_s: float | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file with validation."""
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Compute SHA-256 hash of config for provenance."""
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _parse_parameter_space(raw: dict[str, Any]) -> dict[str, ParameterDefinition]:
    """Parse parameter space from YAML config."""
    result = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"parameter {name!r} must have a dict spec")

        param_type = spec.get("type", "continuous")
        if param_type == "continuous":
            result[name] = ParameterDefinition(
                param_type="continuous",
                min=float(spec["min"]),
                max=float(spec["max"]),
            )
        elif param_type == "discrete":
            values = [float(v) for v in spec.get("values", [])]
            if not values:
                raise ValueError(f"discrete parameter {name!r} must have values")
            result[name] = ParameterDefinition(
                param_type="discrete",
                values=values,
            )
        else:
            raise ValueError(f"unsupported param_type {param_type!r} for {name!r}")

    return result


def _parse_optimization_config(payload: dict[str, Any]) -> OptimizationConfig:
    """Parse full optimization config from YAML payload."""
    param_space_raw = payload.get("parameter_space", {})
    parameter_space = _parse_parameter_space(param_space_raw)

    return OptimizationConfig(
        parameter_space=parameter_space,
        optimizer_type=payload.get("optimizer_type", "random_search"),
        sample_budget=int(payload.get("sample_budget", 20)),
        optimizer_seed=int(payload.get("optimizer_seed", 4362)),
        scenario_family=payload.get("scenario_family", None),
        planner_name=payload.get("planner_name", "safe_baseline"),
        seeds=payload.get("seeds", [0, 1, 2]),
        objective_weights=payload.get("objective_weights", {}),
        invalid_run_policy=payload.get("invalid_run_policy", "fail_closed"),
        diagnostic_only=bool(payload.get("diagnostic_only_not_benchmark_gate", True)),
        horizon=int(payload.get("horizon", 80)),
        dt=float(payload.get("dt", 0.1)),
        max_workers=int(payload.get("max_workers", 1)),
    )


def _sample_parameters(
    parameter_space: dict[str, ParameterDefinition],
    rng: random.Random,
) -> dict[str, float]:
    """Sample a single candidate from parameter space."""
    params = {}
    for name, param_def in parameter_space.items():
        if param_def.param_type == "continuous":
            value = rng.uniform(param_def.min, param_def.max)
        elif param_def.param_type == "discrete":
            value = rng.choice(param_def.values)
        else:
            raise ValueError(f"unknown param_type {param_def.param_type!r}")
        params[name] = value
    return params


def _build_baseline_parameters(
    parameter_space: dict[str, ParameterDefinition],
) -> dict[str, float]:
    """Build baseline (unperturbed) parameter values."""
    params = {}
    for name, param_def in parameter_space.items():
        if param_def.param_type == "continuous":
            params[name] = (param_def.min + param_def.max) / 2.0
        elif param_def.param_type == "discrete":
            params[name] = param_def.values[0]
    return params


def _evaluate_candidate(  # noqa: C901
    candidate_id: str,
    parameters: dict[str, float],
    scenario: dict[str, Any],
    config: OptimizationConfig,
    objective_config: CriticalityObjectiveConfig,
    output_dir: Path,
    scenario_path: Path,
) -> CandidateResult:
    """Evaluate a single candidate by running the simulator."""
    import time
    from types import SimpleNamespace

    from robot_sf.benchmark.map_runner import run_map_batch
    from scripts.validation.run_scenario_perturbation_criticality_pilot import (
        resolve_planner_run_spec,
    )

    start_time = time.perf_counter()
    try:
        # Apply parameters to scenario without mutation
        patched_scenario = apply_criticality_parameters(scenario, parameters)
        patched_scenario["seeds"] = config.seeds

        # Temporary JSONL for this candidate's run
        import uuid

        temp_jsonl = output_dir / f"{candidate_id}_{uuid.uuid4().hex}_eval.episodes.jsonl"
        if temp_jsonl.exists():
            temp_jsonl.unlink()

        # Resolve the planner
        planner_name = config.planner_name
        if planner_name == "safe_baseline":
            planner_name = "goal"
        spec = resolve_planner_run_spec(planner_name)

        # Run the batch
        run_map_batch(
            scenarios_or_path=[patched_scenario],
            out_path=temp_jsonl,
            schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
            scenario_path=scenario_path,
            algo=spec.algo,
            algo_config_path=spec.algo_config_path.as_posix()
            if spec.algo_config_path is not None
            else None,
            horizon=config.horizon,
            dt=config.dt,
            resume=False,
            benchmark_profile="experimental",
        )

        # Load the generated JSONL records
        records = []
        if temp_jsonl.exists():
            for line in temp_jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))

        # Cleanup the temp JSONL
        if temp_jsonl.exists():
            temp_jsonl.unlink()

        # Group records by seed
        records_by_seed = {int(r.get("seed", 0)): r for r in records}

        per_seed_scores = []
        all_scores = []
        all_decompositions = []

        for seed in config.seeds:
            row = records_by_seed.get(seed)
            if row is None:
                per_seed_scores.append((seed, float("nan"), "missing"))
                continue

            episode_data = SimpleNamespace(metrics=row.get("metrics", {}))
            result = compute_criticality_score(episode_data, objective_config)

            if result.status == "not_evaluable":
                per_seed_scores.append((seed, float("nan"), result.status))
            else:
                per_seed_scores.append((seed, result.criticality_score, result.status))
                all_scores.append(result.criticality_score)
                all_decompositions.append(
                    {
                        "collision": result.collision_term,
                        "near_miss": result.near_miss_term,
                        "clearance": result.clearance_term,
                        "progress_failure": result.progress_failure_term,
                        "stalled_time": result.stalled_time_term,
                    }
                )

        runtime_s = time.perf_counter() - start_time

        if not all_scores:
            return CandidateResult(
                candidate_id=candidate_id,
                parameters=parameters,
                criticality_score=float("nan"),
                score_decomposition={},
                status="not_evaluable",
                reason="All seeds failed evaluation",
                per_seed_scores=per_seed_scores,
                runtime_s=runtime_s,
            )

        mean_score = sum(all_scores) / len(all_scores)
        mean_decomposition = {
            k: sum(d[k] for d in all_decompositions) / len(all_decompositions)
            for k in all_decompositions[0]
        }

        return CandidateResult(
            candidate_id=candidate_id,
            parameters=parameters,
            criticality_score=mean_score,
            score_decomposition=mean_decomposition,
            status="evaluated",
            per_seed_scores=per_seed_scores,
            runtime_s=runtime_s,
        )

    except Exception as e:  # noqa: BLE001
        runtime_s = time.perf_counter() - start_time
        return CandidateResult(
            candidate_id=candidate_id,
            parameters=parameters,
            criticality_score=float("nan"),
            score_decomposition={},
            status="invalid_candidate",
            reason=str(e),
            runtime_s=runtime_s,
        )


def run_criticality_optimization(  # noqa: C901
    config: OptimizationConfig,
    output_dir: Path | None = None,
) -> tuple[list[CandidateResult], dict[str, Any]]:
    """Run the criticality optimization search.

    Args:
        config: Optimization configuration
        output_dir: Optional output directory for artifacts

    Returns:
        Tuple of (candidate_results, manifest)
    """
    if output_dir is None:
        output_dir = Path("output/tmp_criticality")
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_path = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenarios = []
    if isinstance(config.scenario_family, (str, Path)):
        scenario_path = Path(config.scenario_family)
        scenarios = load_scenarios(scenario_path)
    elif isinstance(config.scenario_family, list):
        scenarios = list(config.scenario_family)

    if not scenarios:
        default_path = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
        if default_path.exists():
            scenarios = load_scenarios(default_path)
            scenario_path = default_path
        else:
            raise ValueError("No scenario_family configured, and default fallback not found.")

    scenario = dict(scenarios[0])

    rng = random.Random(config.optimizer_seed)
    baseline_params = _build_baseline_parameters(config.parameter_space)

    objective_config = CriticalityObjectiveConfig(
        collision_weight=config.objective_weights.get("collision", 10.0),
        near_miss_weight=config.objective_weights.get("near_miss", 2.0),
        clearance_margin=config.objective_weights.get("clearance_margin", 0.5),
        clearance_weight=config.objective_weights.get("clearance", 1.0),
        progress_failure_weight=config.objective_weights.get("progress_failure", 5.0),
        stalled_time_weight=config.objective_weights.get("stalled_time", 0.5),
    )

    # Pre-sample all candidate parameters for deterministic parallel execution
    candidate_params: list[tuple[str, dict[str, float]]] = [
        ("baseline_unperturbed", baseline_params),
    ]
    for i in range(config.sample_budget):
        params = _sample_parameters(config.parameter_space, rng)
        candidate_params.append((f"candidate_{i:04d}", params))

    # Resolve effective worker count: 0 means auto (cpu_count), 1 means sequential
    effective_workers = config.max_workers
    if effective_workers == 0:
        import os

        effective_workers = os.cpu_count() or 1

    candidates: list[CandidateResult] = []

    if effective_workers <= 1:
        # Sequential evaluation (original behavior)
        for cand_id, params in candidate_params:
            result = _evaluate_candidate(
                candidate_id=cand_id,
                parameters=params,
                scenario=scenario,
                config=config,
                objective_config=objective_config,
                output_dir=output_dir,
                scenario_path=scenario_path,
            )
            candidates.append(result)
    else:
        # Parallel evaluation with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_candidate,
                    candidate_id=cand_id,
                    parameters=params,
                    scenario=scenario,
                    config=config,
                    objective_config=objective_config,
                    output_dir=output_dir,
                    scenario_path=scenario_path,
                ): cand_id
                for cand_id, params in candidate_params
            }
            for future in as_completed(futures):
                candidates.append(future.result())

    evaluated = [c for c in candidates if c.status == "evaluated"]
    best_candidates = sorted(evaluated, key=lambda c: c.criticality_score, reverse=True)[:5]

    baseline_candidate = next(
        (c for c in candidates if c.candidate_id == "baseline_unperturbed"), None
    )
    baseline_score = baseline_candidate.criticality_score if baseline_candidate else float("nan")

    manifest = {
        "issue": 4362,
        "claim_boundary": "exploratory/diagnostic-only; not a validated benchmark method",
        "metrics_source": "simulator_run_map_batch",
        "optimizer_type": config.optimizer_type,
        "optimizer_seed": config.optimizer_seed,
        "sample_budget": config.sample_budget,
        "seeds_per_candidate": config.seeds,
        "parameter_space": {
            name: {
                "type": p.param_type,
                "min": p.min,
                "max": p.max,
                "values": p.values,
            }
            for name, p in config.parameter_space.items()
        },
        "objective_weights": config.objective_weights,
        "invalid_run_policy": config.invalid_run_policy,
        "max_workers": effective_workers,
        "total_candidates": len(candidates),
        "evaluated_count": len(evaluated),
        "invalid_count": len([c for c in candidates if c.status == "invalid_candidate"]),
        "not_evaluable_count": len([c for c in candidates if c.status == "not_evaluable"]),
        "best_candidate_id": best_candidates[0].candidate_id if best_candidates else None,
        "best_criticality_score": best_candidates[0].criticality_score if best_candidates else None,
        "baseline_score": baseline_score,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    return candidates, manifest


def write_optimization_report(
    candidates: list[CandidateResult],
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write optimization artifacts to output directory.

    Args:
        candidates: Candidate results
        manifest: Run manifest
        output_dir: Output directory

    Returns:
        Dict of artifact name to path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_jsonl = output_dir / "candidate_results.jsonl"
    with candidates_jsonl.open("w", encoding="utf-8") as f:
        for c in candidates:
            record = {
                "candidate_id": c.candidate_id,
                "parameters": c.parameters,
                "criticality_score": c.criticality_score,
                "score_decomposition": c.score_decomposition,
                "status": c.status,
                "reason": c.reason,
                "per_seed_scores": [
                    {"seed": s, "score": sc, "status": st} for s, sc, st in c.per_seed_scores
                ],
                "runtime_s": c.runtime_s,
            }
            f.write(json.dumps(record) + "\n")

    import csv

    candidate_summary_csv = output_dir / "candidate_summary.csv"
    with candidate_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "candidate_id",
                "criticality_score",
                "status",
                "collision_term",
                "near_miss_term",
                "clearance_term",
                "progress_failure_term",
                "stalled_time_term",
            ]
        )
        for c in candidates:
            writer.writerow(
                [
                    c.candidate_id,
                    c.criticality_score,
                    c.status,
                    c.score_decomposition.get("collision", ""),
                    c.score_decomposition.get("near_miss", ""),
                    c.score_decomposition.get("clearance", ""),
                    c.score_decomposition.get("progress_failure", ""),
                    c.score_decomposition.get("stalled_time", ""),
                ]
            )

    best_candidates_json = output_dir / "best_candidates.json"
    evaluated = [c for c in candidates if c.status == "evaluated"]
    best = sorted(evaluated, key=lambda c: c.criticality_score, reverse=True)[:5]
    with best_candidates_json.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "candidate_id": c.candidate_id,
                    "parameters": c.parameters,
                    "criticality_score": c.criticality_score,
                    "score_decomposition": c.score_decomposition,
                }
                for c in best
            ],
            f,
            indent=2,
        )

    optimization_manifest_json = output_dir / "optimization_manifest.json"
    with optimization_manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme_md = output_dir / "README.md"
    with readme_md.open("w", encoding="utf-8") as f:
        f.write("# Issue #4362: Classical Criticality Optimization\n\n")
        f.write(
            "**Claim boundary**: exploratory/diagnostic-only; not a validated benchmark method.\n\n"
        )
        if manifest.get("metrics_source") == "simulator_run_map_batch":
            f.write(
                "> **Metrics are generated using the simulator runner.** Evaluated candidates are "
                "run on the configured planner across the specified seeds.\n\n"
            )
        else:
            f.write(
                "> **Metrics are PLACEHOLDER, not simulator output.** In this v0 stub the candidate "
                "metrics come from a fixed RNG that does not depend on the candidate parameters, so "
                "every candidate (including the baseline) has identical metrics and the "
                "best/baseline scores below are **not** a meaningful optimization result — they only "
                "exercise the objective + artifact interface. Real simulator integration is a "
                "follow-up to #4362.\n\n"
            )
        f.write(f"- Optimizer: {manifest['optimizer_type']}\n")
        f.write(f"- Sample budget: {manifest['sample_budget']}\n")
        f.write(
            f"- Evaluated candidates: {manifest['evaluated_count']}/{manifest['total_candidates']}\n"
        )
        f.write(f"- Best score: {manifest['best_criticality_score']}\n")
        f.write(f"- Baseline score: {manifest['baseline_score']}\n")
        f.write(f"- Generated: {manifest['generated_at']}\n")

    return {
        "candidate_results_jsonl": candidates_jsonl,
        "candidate_summary_csv": candidate_summary_csv,
        "best_candidates_json": best_candidates_json,
        "optimization_manifest_json": optimization_manifest_json,
        "readme_md": readme_md,
    }


def main() -> int:
    """Parse CLI, run optimization, write artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/issue_4362_scenario_criticality_smoke.yaml"),
        help="YAML run configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for artifacts",
    )
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    args = parser.parse_args()

    payload = _load_yaml(args.config)
    if not payload.get("diagnostic_only_not_benchmark_gate"):
        raise ValueError("config must set diagnostic_only_not_benchmark_gate: true")

    config = _parse_optimization_config(payload)
    output_dir = args.output_dir or Path(
        payload.get("output_dir", "output/issue_4362_scenario_criticality_smoke")
    )

    candidates, manifest = run_criticality_optimization(config, output_dir)
    written = write_optimization_report(candidates, manifest, output_dir)

    print(f"candidate_results={written['candidate_results_jsonl']}")
    print(f"candidate_summary={written['candidate_summary_csv']}")
    print(f"best_candidates={written['best_candidates_json']}")
    print(f"optimization_manifest={written['optimization_manifest_json']}")
    print(f"readme={written['readme_md']}")
    print(f"best_score={manifest['best_criticality_score']}")
    print(f"baseline_score={manifest['baseline_score']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
