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
from scripts.validation.run_scenario_perturbation_criticality_pilot import (
    resolve_planner_run_spec,
)


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
        optimizer_type: "random_search", "differential_evolution", or "cma_es"
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
    # Differential-evolution-specific settings (ignored for random_search)
    de_maxiter: int = 50
    de_popsize: int = 15
    de_seed: int | None = None
    # CMA-ES-specific settings (ignored for other optimizer types)
    cma_es_maxiter: int = 30
    cma_es_sigma0: float = 0.5
    cma_es_seed: int | None = None


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
        patch_s: Time to apply parameters to scenario
        simulation_s: Time to run simulator episodes
        score_s: Time to compute criticality score from metrics
    """

    candidate_id: str
    parameters: dict[str, float]
    criticality_score: float
    score_decomposition: dict[str, float]
    status: str
    reason: str | None = None
    per_seed_scores: list[tuple[int, float, str]] = field(default_factory=list)
    runtime_s: float | None = None
    patch_s: float | None = None
    simulation_s: float | None = None
    score_s: float | None = None


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
        de_maxiter=int(payload.get("de_maxiter", 50)),
        de_popsize=int(payload.get("de_popsize", 15)),
        de_seed=(int(payload["de_seed"]) if payload.get("de_seed") is not None else None),
        cma_es_maxiter=int(payload.get("cma_es_maxiter", 30)),
        cma_es_sigma0=float(payload.get("cma_es_sigma0", 0.5)),
        cma_es_seed=(
            int(payload["cma_es_seed"]) if payload.get("cma_es_seed") is not None else None
        ),
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


def _evaluate_candidate(  # noqa: C901, PLR0912, PLR0913, PLR0915
    candidate_id: str,
    parameters: dict[str, float],
    scenario: dict[str, Any],
    config: OptimizationConfig,
    objective_config: CriticalityObjectiveConfig,
    output_dir: Path,
    scenario_path: Path,
    *,
    resolved_algo: str | None = None,
    resolved_algo_config_path: str | None = None,
) -> CandidateResult:
    """Evaluate a single candidate by running the simulator.

    Args:
        resolved_algo: Pre-resolved planner algo name (avoids redundant resolution
            per worker in parallel mode).
        resolved_algo_config_path: Pre-resolved algo config path string.
    """
    import time
    import uuid
    from types import SimpleNamespace

    from robot_sf.benchmark.map_runner import run_map_batch

    start_time = time.perf_counter()
    patch_s: float | None = None
    simulation_s: float | None = None
    score_s: float | None = None
    temp_jsonl = output_dir / f"{candidate_id}_{uuid.uuid4().hex}_eval.episodes.jsonl"

    try:
        # Stage 1: Apply parameters to scenario without mutation
        patch_start = time.perf_counter()
        patched_scenario = apply_criticality_parameters(scenario, parameters)
        patched_scenario["seeds"] = config.seeds
        patch_s = time.perf_counter() - patch_start

        # Temporary JSONL for this candidate's run
        if temp_jsonl.exists():
            temp_jsonl.unlink()

        # Use pre-resolved planner spec if provided, otherwise resolve locally
        if resolved_algo is not None:
            algo = resolved_algo
            algo_config_path = resolved_algo_config_path
        else:
            planner_name = config.planner_name
            if planner_name == "safe_baseline":
                planner_name = "goal"
            spec = resolve_planner_run_spec(planner_name)
            algo = spec.algo
            algo_config_path = (
                spec.algo_config_path.as_posix() if spec.algo_config_path is not None else None
            )

        # Stage 2: Run the batch (simulator episodes)
        sim_start = time.perf_counter()
        run_map_batch(
            scenarios_or_path=[patched_scenario],
            out_path=temp_jsonl,
            schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
            scenario_path=scenario_path,
            algo=algo,
            algo_config_path=algo_config_path,
            horizon=config.horizon,
            dt=config.dt,
            resume=False,
            benchmark_profile="experimental",
        )
        simulation_s = time.perf_counter() - sim_start

        # Stage 3: Load JSONL records and compute scores
        score_start = time.perf_counter()
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

        score_s = time.perf_counter() - score_start
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
                patch_s=patch_s,
                simulation_s=simulation_s,
                score_s=score_s,
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
            patch_s=patch_s,
            simulation_s=simulation_s,
            score_s=score_s,
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
            patch_s=patch_s,
            simulation_s=simulation_s,
            score_s=score_s,
        )
    finally:
        if temp_jsonl.exists():
            try:
                temp_jsonl.unlink()
            except OSError:
                pass


def _run_differential_evolution(
    config: OptimizationConfig,
    scenario: dict[str, Any],
    scenario_path: Path,
    objective_config: CriticalityObjectiveConfig,
    output_dir: Path,
    resolved_algo: str,
    resolved_algo_config_path: str | None,
) -> list[CandidateResult]:
    """Run differential evolution optimization.

    DE minimizes, so we negate the criticality score (which should be maximized).
    Only continuous parameters are supported; discrete params default to baseline.
    """
    from scipy.optimize import differential_evolution

    # DE requires all-continuous params; extract bounds and names
    continuous_names = [
        name for name, p in config.parameter_space.items() if p.param_type == "continuous"
    ]
    if not continuous_names:
        raise ValueError("differential_evolution requires at least one continuous parameter")

    bounds = [
        (config.parameter_space[name].min, config.parameter_space[name].max)
        for name in continuous_names
    ]

    # Container for evaluated candidates (DE calls this callback per iteration)
    de_candidates: list[CandidateResult] = []
    trial_counter: int = 0

    def _de_objective(x: list[float]) -> float:
        """DE minimizes; return negative criticality score."""
        nonlocal trial_counter
        params = dict(zip(continuous_names, x, strict=True))
        # Fill discrete params with baseline (first value)
        for name, pdef in config.parameter_space.items():
            if pdef.param_type == "discrete" and name not in params:
                params[name] = pdef.values[0] if pdef.values else 0.0

        trial_counter += 1
        result = _evaluate_candidate(
            candidate_id=f"de_trial_{trial_counter:04d}",
            parameters=params,
            scenario=scenario,
            config=config,
            objective_config=objective_config,
            output_dir=output_dir,
            scenario_path=scenario_path,
            resolved_algo=resolved_algo,
            resolved_algo_config_path=resolved_algo_config_path,
        )
        de_candidates.append(result)
        # DE minimizes; negate criticality (higher is better -> lower negative)
        if result.status == "evaluated":
            return -result.criticality_score
        # Unsuccessful evaluations: return large positive so DE avoids them
        return 1e12

    seed = config.de_seed if config.de_seed is not None else config.optimizer_seed
    de_result = differential_evolution(
        _de_objective,
        bounds=bounds,
        maxiter=config.de_maxiter,
        popsize=config.de_popsize,
        seed=seed,
        # Disable polish: the L-BFGS-B refinement pass adds finite-difference
        # gradient evaluations that are wasteful (and near-meaningless) for this
        # noisy, simulator-backed objective. SciPy only keeps the polished point
        # when it improves, so skipping it cannot worsen the DE result.
        polish=False,
        updating="deferred",
    )

    # Add the DE optimum as a named candidate
    optimum_params = dict(zip(continuous_names, de_result.x, strict=True))
    for name, pdef in config.parameter_space.items():
        if pdef.param_type == "discrete" and name not in optimum_params:
            optimum_params[name] = pdef.values[0] if pdef.values else 0.0

    de_best = _evaluate_candidate(
        candidate_id="de_optimum",
        parameters=optimum_params,
        scenario=scenario,
        config=config,
        objective_config=objective_config,
        output_dir=output_dir,
        scenario_path=scenario_path,
        resolved_algo=resolved_algo,
        resolved_algo_config_path=resolved_algo_config_path,
    )
    de_candidates.append(de_best)

    return de_candidates


def _run_cma_es(
    config: OptimizationConfig,
    scenario: dict[str, Any],
    scenario_path: Path,
    objective_config: CriticalityObjectiveConfig,
    output_dir: Path,
    resolved_algo: str,
    resolved_algo_config_path: str | None,
) -> list[CandidateResult]:
    """Run CMA-ES optimization.

    CMA-ES minimizes, so we negate the criticality score (which should be maximized).
    Only continuous parameters are supported; discrete params default to baseline.
    Requires the ``cma`` package (install via ``uv pip install -e ".[criticality]"``).
    """
    try:
        import cma
    except ImportError as exc:
        raise ImportError(
            "cma_es optimizer requires the 'cma' package. "
            'Install with: uv pip install -e ".[criticality]"'
        ) from exc

    # CMA-ES requires all-continuous params; extract bounds and dimensionality
    continuous_names = [
        name for name, p in config.parameter_space.items() if p.param_type == "continuous"
    ]
    if not continuous_names:
        raise ValueError("cma_es requires at least one continuous parameter")

    # Compute the search range midpoint and initial sigma
    midpoints = [
        (config.parameter_space[name].min + config.parameter_space[name].max) / 2.0
        for name in continuous_names
    ]
    half_ranges = [
        (config.parameter_space[name].max - config.parameter_space[name].min) / 2.0
        for name in continuous_names
    ]
    sigma0 = config.cma_es_sigma0 * max(half_ranges)

    # Container for evaluated candidates
    cma_candidates: list[CandidateResult] = []
    trial_counter: int = 0

    def _cma_objective(x: list[float]) -> float:
        """CMA-ES minimizes; return negative criticality score."""
        nonlocal trial_counter
        params = dict(zip(continuous_names, x, strict=True))
        # Fill discrete params with baseline (first value)
        for name, pdef in config.parameter_space.items():
            if pdef.param_type == "discrete" and name not in params:
                params[name] = pdef.values[0] if pdef.values else 0.0

        trial_counter += 1
        result = _evaluate_candidate(
            candidate_id=f"cma_trial_{trial_counter:04d}",
            parameters=params,
            scenario=scenario,
            config=config,
            objective_config=objective_config,
            output_dir=output_dir,
            scenario_path=scenario_path,
            resolved_algo=resolved_algo,
            resolved_algo_config_path=resolved_algo_config_path,
        )
        cma_candidates.append(result)
        if result.status == "evaluated":
            return -result.criticality_score
        return 1e12

    seed = config.cma_es_seed if config.cma_es_seed is not None else config.optimizer_seed

    lower_bounds = [config.parameter_space[name].min for name in continuous_names]
    upper_bounds = [config.parameter_space[name].max for name in continuous_names]

    options = {
        "maxiter": config.cma_es_maxiter,
        "seed": seed,
        "bounds": [lower_bounds, upper_bounds],
        "verbose": -9,
    }

    result = cma.fmin2(
        _cma_objective,
        x0=midpoints,
        sigma0=sigma0,
        options=options,
    )

    optimum_params = dict(zip(continuous_names, result[0], strict=True))
    for name, pdef in config.parameter_space.items():
        if pdef.param_type == "discrete" and name not in optimum_params:
            optimum_params[name] = pdef.values[0] if pdef.values else 0.0

    cma_best = _evaluate_candidate(
        candidate_id="cma_optimum",
        parameters=optimum_params,
        scenario=scenario,
        config=config,
        objective_config=objective_config,
        output_dir=output_dir,
        scenario_path=scenario_path,
        resolved_algo=resolved_algo,
        resolved_algo_config_path=resolved_algo_config_path,
    )
    cma_candidates.append(cma_best)

    return cma_candidates


def run_criticality_optimization(  # noqa: C901, PLR0912, PLR0915
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

    # Pre-sample all candidate parameters for random search
    random_search_candidates: list[tuple[str, dict[str, float]]] = [
        ("baseline_unperturbed", baseline_params),
    ]
    for i in range(config.sample_budget):
        params = _sample_parameters(config.parameter_space, rng)
        random_search_candidates.append((f"candidate_{i:04d}", params))

    # Resolve effective worker count: 0 means auto (cpu_count), 1 means sequential
    effective_workers = config.max_workers
    if effective_workers == 0:
        import os

        effective_workers = os.cpu_count() or 1
    # Never spawn more workers than there are candidates to evaluate: on a
    # high-core host, max_workers=0 (auto) would otherwise start dozens of idle
    # processes for a handful of candidates.
    if effective_workers > 1:
        effective_workers = min(effective_workers, max(1, len(random_search_candidates)))

    # Pre-resolve planner spec once in the main process so parallel workers
    # avoid redundant YAML registry / candidate-config file I/O.
    planner_name = config.planner_name
    if planner_name == "safe_baseline":
        planner_name = "goal"
    _planner_spec = resolve_planner_run_spec(planner_name)
    _resolved_algo = _planner_spec.algo
    _resolved_algo_config_path = (
        _planner_spec.algo_config_path.as_posix()
        if _planner_spec.algo_config_path is not None
        else None
    )

    candidates: list[CandidateResult] = []

    if config.optimizer_type == "random_search":
        if effective_workers <= 1:
            # Sequential evaluation
            for cand_id, params in random_search_candidates:
                result = _evaluate_candidate(
                    candidate_id=cand_id,
                    parameters=params,
                    scenario=scenario,
                    config=config,
                    objective_config=objective_config,
                    output_dir=output_dir,
                    scenario_path=scenario_path,
                    resolved_algo=_resolved_algo,
                    resolved_algo_config_path=_resolved_algo_config_path,
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
                        resolved_algo=_resolved_algo,
                        resolved_algo_config_path=_resolved_algo_config_path,
                    ): cand_id
                    for cand_id, params in random_search_candidates
                }
                for future in as_completed(futures):
                    candidates.append(future.result())
            # ProcessPoolExecutor completes futures in nondeterministic order.
            # Restore a stable, reproducible ordering by candidate_id so the whole
            # downstream (report listing, tie-breaking, manifest) matches the
            # sequential path regardless of which worker finished first.
            candidates.sort(key=lambda c: c.candidate_id)
    elif config.optimizer_type == "differential_evolution":
        de_results = _run_differential_evolution(
            config=config,
            scenario=scenario,
            scenario_path=scenario_path,
            objective_config=objective_config,
            output_dir=output_dir,
            resolved_algo=_resolved_algo,
            resolved_algo_config_path=_resolved_algo_config_path,
        )
        # DE results include the optimum; prepend baseline
        baseline_params = _build_baseline_parameters(config.parameter_space)
        baseline_result = _evaluate_candidate(
            candidate_id="baseline_unperturbed",
            parameters=baseline_params,
            scenario=scenario,
            config=config,
            objective_config=objective_config,
            output_dir=output_dir,
            scenario_path=scenario_path,
            resolved_algo=_resolved_algo,
            resolved_algo_config_path=_resolved_algo_config_path,
        )
        candidates = [baseline_result] + de_results
    elif config.optimizer_type == "cma_es":
        cma_results = _run_cma_es(
            config=config,
            scenario=scenario,
            scenario_path=scenario_path,
            objective_config=objective_config,
            output_dir=output_dir,
            resolved_algo=_resolved_algo,
            resolved_algo_config_path=_resolved_algo_config_path,
        )
        # CMA-ES results include the optimum; prepend baseline
        baseline_params = _build_baseline_parameters(config.parameter_space)
        baseline_result = _evaluate_candidate(
            candidate_id="baseline_unperturbed",
            parameters=baseline_params,
            scenario=scenario,
            config=config,
            objective_config=objective_config,
            output_dir=output_dir,
            scenario_path=scenario_path,
            resolved_algo=_resolved_algo,
            resolved_algo_config_path=_resolved_algo_config_path,
        )
        candidates = [baseline_result] + cma_results
    else:
        raise ValueError(
            f"unsupported optimizer_type {config.optimizer_type!r}; "
            "supported: random_search, differential_evolution, cma_es"
        )

    evaluated = [c for c in candidates if c.status == "evaluated"]
    # Secondary key on candidate_id makes "best" selection reproducible: parallel
    # workers complete in nondeterministic order, so ties on criticality_score must
    # break deterministically rather than on completion order.
    best_candidates = sorted(evaluated, key=lambda c: (-c.criticality_score, c.candidate_id))[:5]

    baseline_candidate = next(
        (c for c in candidates if c.candidate_id == "baseline_unperturbed"), None
    )
    baseline_score = baseline_candidate.criticality_score if baseline_candidate else float("nan")

    # DE-specific manifest fields
    de_manifest = {}
    if config.optimizer_type == "differential_evolution":
        de_manifest = {
            "de_maxiter": config.de_maxiter,
            "de_popsize": config.de_popsize,
            # Record the effective seed actually used by the DE run (the runner
            # falls back to optimizer_seed when de_seed is None), so the manifest
            # reflects a reproducible seed instead of null.
            "de_seed": config.de_seed if config.de_seed is not None else config.optimizer_seed,
        }
    # CMA-ES-specific manifest fields
    cma_manifest = {}
    if config.optimizer_type == "cma_es":
        cma_manifest = {
            "cma_es_maxiter": config.cma_es_maxiter,
            "cma_es_sigma0": config.cma_es_sigma0,
            "cma_es_seed": config.cma_es_seed
            if config.cma_es_seed is not None
            else config.optimizer_seed,
        }
    manifest = {
        "issue": 4362,
        "claim_boundary": "exploratory/diagnostic-only; not a validated benchmark method",
        "metrics_source": "simulator_run_map_batch",
        "optimizer_type": config.optimizer_type,
        "optimizer_seed": config.optimizer_seed,
        "sample_budget": config.sample_budget,
        "seeds_per_candidate": config.seeds,
        "planner_name": config.planner_name,
        "planner_algo": _resolved_algo,
        "planner_algo_config_path": _resolved_algo_config_path,
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
    manifest.update(de_manifest)
    manifest.update(cma_manifest)

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
                "patch_s": c.patch_s,
                "simulation_s": c.simulation_s,
                "score_s": c.score_s,
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
                "runtime_s",
                "patch_s",
                "simulation_s",
                "score_s",
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
                    c.runtime_s if c.runtime_s is not None else "",
                    c.patch_s if c.patch_s is not None else "",
                    c.simulation_s if c.simulation_s is not None else "",
                    c.score_s if c.score_s is not None else "",
                ]
            )

    best_candidates_json = output_dir / "best_candidates.json"
    evaluated = [c for c in candidates if c.status == "evaluated"]
    # Tie-break on candidate_id so the written "best" list is reproducible when
    # multiple candidates share a criticality_score (matches run_* selection).
    best = sorted(evaluated, key=lambda c: (-c.criticality_score, c.candidate_id))[:5]
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
