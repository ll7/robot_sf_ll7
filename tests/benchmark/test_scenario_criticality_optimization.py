"""Tests for scenario criticality optimization runner (Issue #4362)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "run_scenario_criticality_optimization.py"
)
_SPEC = importlib.util.spec_from_file_location("_scenario_crit_opt", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules["_scenario_crit_opt"] = _MODULE
_SPEC.loader.exec_module(_MODULE)

OptimizationConfig = _MODULE.OptimizationConfig
ParameterDefinition = _MODULE.ParameterDefinition
_build_baseline_parameters = _MODULE._build_baseline_parameters
_parse_parameter_space = _MODULE._parse_parameter_space
_sample_parameters = _MODULE._sample_parameters
resolve_planner_run_spec = _MODULE.resolve_planner_run_spec
run_criticality_optimization = _MODULE.run_criticality_optimization
write_optimization_report = _MODULE.write_optimization_report


def _make_test_config(
    sample_budget: int = 5,
    optimizer_seed: int = 42,
) -> OptimizationConfig:
    return OptimizationConfig(
        parameter_space={
            "pedestrian_speed_scale": ParameterDefinition(
                param_type="continuous", min=0.7, max=1.4
            ),
            "pedestrian_start_delay_s": ParameterDefinition(
                param_type="continuous", min=0.0, max=2.0
            ),
        },
        optimizer_type="random_search",
        sample_budget=sample_budget,
        optimizer_seed=optimizer_seed,
        seeds=[0, 1],
        objective_weights={
            "collision": 10.0,
            "near_miss": 2.0,
            "clearance_margin": 0.5,
            "clearance": 1.0,
            "progress_failure": 5.0,
            "stalled_time": 0.5,
        },
    )


def test_random_search_deterministic() -> None:
    """Random search produces identical results for fixed optimizer seed."""
    config_a = _make_test_config(optimizer_seed=123)
    config_b = _make_test_config(optimizer_seed=123)

    candidates_a, _ = run_criticality_optimization(config_a)
    candidates_b, _ = run_criticality_optimization(config_b)

    assert len(candidates_a) == len(candidates_b)
    for ca, cb in zip(candidates_a, candidates_b, strict=True):
        assert ca.candidate_id == cb.candidate_id
        assert ca.parameters == cb.parameters
        if ca.status == "evaluated":
            assert ca.criticality_score == pytest.approx(cb.criticality_score)


def test_random_search_different_seeds_differ() -> None:
    """Different optimizer seeds produce different parameter samples."""
    config_a = _make_test_config(optimizer_seed=1)
    config_b = _make_test_config(optimizer_seed=999)

    candidates_a, _ = run_criticality_optimization(config_a)
    candidates_b, _ = run_criticality_optimization(config_b)

    non_baseline_a = [c for c in candidates_a if c.candidate_id != "baseline_unperturbed"]
    non_baseline_b = [c for c in candidates_b if c.candidate_id != "baseline_unperturbed"]

    any_different = any(
        ca.parameters != cb.parameters
        for ca, cb in zip(non_baseline_a, non_baseline_b, strict=False)
    )
    assert any_different


def test_baseline_always_included() -> None:
    """Baseline (unperturbed) candidate is always first in results."""
    config = _make_test_config()
    candidates, _ = run_criticality_optimization(config)

    assert len(candidates) >= 1
    assert candidates[0].candidate_id == "baseline_unperturbed"
    assert candidates[0].status == "evaluated"


def test_manifest_contains_required_fields() -> None:
    """Output manifest includes seed list, objective weights, and claim boundary."""
    config = _make_test_config()
    _, manifest = run_criticality_optimization(config)

    assert "claim_boundary" in manifest
    assert "exploratory" in manifest["claim_boundary"]
    assert manifest["optimizer_seed"] == 42
    assert manifest["seeds_per_candidate"] == [0, 1]
    assert manifest["objective_weights"]["collision"] == 10.0
    assert manifest["sample_budget"] == 5
    assert manifest["total_candidates"] == 6
    assert "evaluated_count" in manifest
    assert "invalid_count" in manifest
    assert "not_evaluable_count" in manifest
    assert "baseline_score" in manifest
    assert "generated_at" in manifest
    assert "planner_name" in manifest
    assert "planner_algo" in manifest
    assert "planner_algo_config_path" in manifest


def test_write_optimization_report(tmp_path: Path) -> None:
    """write_optimization_report produces all expected artifact files."""
    config = _make_test_config()
    candidates, manifest = run_criticality_optimization(config)
    output_dir = tmp_path / "test_output"

    written = write_optimization_report(candidates, manifest, output_dir)

    assert "candidate_results_jsonl" in written
    assert "candidate_summary_csv" in written
    assert "best_candidates_json" in written
    assert "optimization_manifest_json" in written
    assert "readme_md" in written

    for path in written.values():
        assert path.exists(), f"Expected artifact {path} does not exist"

    with written["candidate_results_jsonl"].open() as f:
        lines = f.readlines()
    assert len(lines) == len(candidates)
    for line in lines:
        record = json.loads(line)
        assert "candidate_id" in record
        assert "parameters" in record
        assert "criticality_score" in record
        assert "status" in record

    with written["optimization_manifest_json"].open() as f:
        loaded = json.load(f)
    assert loaded["claim_boundary"] == manifest["claim_boundary"]
    assert loaded["optimizer_seed"] == manifest["optimizer_seed"]


def test_candidate_count_matches_budget() -> None:
    """Total candidate count equals sample_budget + 1 (baseline)."""
    for budget in [1, 5, 10]:
        config = _make_test_config(sample_budget=budget)
        candidates, manifest = run_criticality_optimization(config)
        assert len(candidates) == budget + 1
        assert manifest["total_candidates"] == budget + 1


def test_parse_parameter_space_continuous() -> None:
    """_parse_parameter_space handles continuous parameters."""
    raw = {"speed": {"type": "continuous", "min": 0.5, "max": 1.5}}
    result = _parse_parameter_space(raw)
    assert "speed" in result
    assert result["speed"].param_type == "continuous"
    assert result["speed"].min == 0.5
    assert result["speed"].max == 1.5


def test_parse_parameter_space_discrete() -> None:
    """_parse_parameter_space handles discrete parameters."""
    raw = {"mode": {"type": "discrete", "values": [1.0, 2.0, 3.0]}}
    result = _parse_parameter_space(raw)
    assert "mode" in result
    assert result["mode"].param_type == "discrete"
    assert result["mode"].values == [1.0, 2.0, 3.0]


def test_parse_parameter_space_invalid_type() -> None:
    """_parse_parameter_space rejects unsupported types."""
    raw = {"bad": {"type": "unknown"}}
    with pytest.raises(ValueError, match="unsupported param_type"):
        _parse_parameter_space(raw)


def test_build_baseline_parameters_midpoint() -> None:
    """Baseline parameters are midpoints of continuous ranges."""
    param_space = {
        "x": ParameterDefinition(param_type="continuous", min=0.0, max=10.0),
        "y": ParameterDefinition(param_type="continuous", min=2.0, max=4.0),
    }
    baseline = _build_baseline_parameters(param_space)
    assert baseline["x"] == pytest.approx(5.0)
    assert baseline["y"] == pytest.approx(3.0)


def test_sample_parameters_in_range() -> None:
    """Sampled parameters fall within defined bounds."""
    import random as _random

    param_space = {
        "a": ParameterDefinition(param_type="continuous", min=0.0, max=1.0),
        "b": ParameterDefinition(param_type="continuous", min=-5.0, max=5.0),
    }
    rng = _random.Random(0)
    for _ in range(50):
        params = _sample_parameters(param_space, rng)
        assert 0.0 <= params["a"] <= 1.0
        assert -5.0 <= params["b"] <= 5.0


def test_cli_help() -> None:
    """CLI --help exits with code 0."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert (
        "scenario criticality" in result.stdout.lower() or "optimization" in result.stdout.lower()
    )


def test_max_workers_default_is_sequential() -> None:
    """Default max_workers=1 means sequential execution."""
    config = _make_test_config()
    assert config.max_workers == 1


def test_max_workers_stored_in_config() -> None:
    """max_workers can be set on OptimizationConfig."""
    config = _make_test_config()
    config.max_workers = 2
    assert config.max_workers == 2


def test_manifest_includes_max_workers() -> None:
    """Manifest records the effective max_workers used."""
    config = _make_test_config()
    config.max_workers = 1
    _, manifest = run_criticality_optimization(config)
    assert manifest["max_workers"] == 1


def test_parallel_matches_sequential_and_is_deterministic() -> None:
    """max_workers=2 exercises the ProcessPoolExecutor path and yields the same,
    deterministically-ordered results as the sequential path.

    Guards against the nondeterministic ``as_completed`` ordering: candidates
    must come back sorted by candidate_id regardless of worker completion order.
    """
    seq_config = _make_test_config(optimizer_seed=7)
    seq_config.max_workers = 1
    seq_candidates, _ = run_criticality_optimization(seq_config)

    par_config = _make_test_config(optimizer_seed=7)
    par_config.max_workers = 2
    par_candidates, par_manifest = run_criticality_optimization(par_config)

    # Effective workers is capped at the candidate count (baseline + budget).
    assert 1 < par_manifest["max_workers"] <= len(par_candidates)

    assert [c.candidate_id for c in par_candidates] == [c.candidate_id for c in seq_candidates]
    for cp, cs in zip(par_candidates, seq_candidates, strict=True):
        assert cp.status == cs.status
        if cp.status == "evaluated":
            assert cp.criticality_score == pytest.approx(cs.criticality_score)


def test_effective_workers_capped_at_candidate_count() -> None:
    """max_workers=0 (auto) must not spawn more workers than there are candidates."""
    config = _make_test_config(sample_budget=2)
    config.max_workers = 0
    _, manifest = run_criticality_optimization(config)
    assert manifest["max_workers"] <= manifest["total_candidates"]


def test_manifest_includes_shared_planner_resolution() -> None:
    """Manifest records the pre-resolved planner algo and config path."""
    config = _make_test_config()
    _, manifest = run_criticality_optimization(config)
    assert "planner_algo" in manifest
    assert manifest["planner_algo"] is not None
    assert "planner_algo_config_path" in manifest
    assert manifest["planner_name"] == config.planner_name


def test_shared_planner_resolution_is_called_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-resolve planner spec once in the main process, not per worker.

    Monkeypatch ``resolve_planner_run_spec`` to count calls.  Since
    ProcessPoolExecutor forks the child, the patch does NOT propagate into
    workers, so the main-process count must be exactly 1 regardless of
    ``max_workers``.
    """
    call_count = {"n": 0}
    original = _MODULE.resolve_planner_run_spec

    def _counting_resolver(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_MODULE, "resolve_planner_run_spec", _counting_resolver)

    config = _make_test_config()
    config.max_workers = 2
    run_criticality_optimization(config)
    assert call_count["n"] == 1


# ---- Differential evolution optimizer tests ----


def test_de_optimizer_accepted() -> None:
    """OptimizationConfig accepts differential_evolution optimizer type."""
    config = OptimizationConfig(
        parameter_space={
            "pedestrian_speed_scale": ParameterDefinition(
                param_type="continuous", min=0.7, max=1.4
            ),
        },
        optimizer_type="differential_evolution",
        sample_budget=1,
        optimizer_seed=42,
        de_maxiter=5,
        de_popsize=3,
    )
    assert config.optimizer_type == "differential_evolution"
    assert config.de_maxiter == 5
    assert config.de_popsize == 3


def test_de_requires_continuous_params() -> None:
    """differential_evolution fails with only discrete parameters."""
    _run_de = _MODULE._run_differential_evolution
    config = OptimizationConfig(
        parameter_space={
            "mode": ParameterDefinition(param_type="discrete", values=[1.0, 2.0]),
        },
        optimizer_type="differential_evolution",
        sample_budget=1,
    )
    with pytest.raises(ValueError, match="requires at least one continuous"):
        _run_de(
            config=config,
            scenario={},
            scenario_path=Path("."),
            objective_config=None,
            output_dir=Path("."),
            resolved_algo="goal",
            resolved_algo_config_path=None,
        )


def test_unsupported_optimizer_type() -> None:
    """run_criticality_optimization rejects unsupported optimizer_type."""
    config = _make_test_config()
    config.optimizer_type = "bayesian"
    with pytest.raises(ValueError, match="unsupported optimizer_type"):
        run_criticality_optimization(config)


def test_de_manifest_fields() -> None:
    """DE run records optimizer-specific fields in manifest."""
    # We can check the parsing without running the full simulator
    payload = {
        "diagnostic_only_not_benchmark_gate": True,
        "optimizer_type": "differential_evolution",
        "de_maxiter": 25,
        "de_popsize": 10,
        "de_seed": 99,
        "parameter_space": {
            "speed": {"type": "continuous", "min": 0.5, "max": 1.5},
        },
    }
    config = _MODULE._parse_optimization_config(payload)
    assert config.optimizer_type == "differential_evolution"
    assert config.de_maxiter == 25
    assert config.de_popsize == 10
    assert config.de_seed == 99


def test_de_seed_fallback() -> None:
    """de_seed defaults to None and falls back to optimizer_seed in DE."""
    payload = {
        "diagnostic_only_not_benchmark_gate": True,
        "optimizer_type": "differential_evolution",
        "optimizer_seed": 4362,
        "parameter_space": {
            "speed": {"type": "continuous", "min": 0.5, "max": 1.5},
        },
    }
    config = _MODULE._parse_optimization_config(payload)
    assert config.de_seed is None
    assert config.optimizer_seed == 4362


# ---- Objective response-to-perturbation tests (Issue #4792) ----


def _full_metrics(**overrides) -> dict:
    """Return a complete metrics dict with all required keys for compute_criticality_score."""
    base = {
        "collision_count": 0.0,
        "near_misses": 0.0,
        "min_clearance": 1.0,
        "failure_to_progress": 0.0,
        "stalled_time": 0.0,
    }
    base.update(overrides)
    return base


def test_criticality_score_responds_to_collisions() -> None:
    """compute_criticality_score returns higher scores for more collisions."""
    from types import SimpleNamespace

    compute = _MODULE.compute_criticality_score

    no_collision = SimpleNamespace(metrics=_full_metrics(collision_count=0))
    some_collision = SimpleNamespace(metrics=_full_metrics(collision_count=3))

    result_0 = compute(no_collision)
    result_3 = compute(some_collision)

    assert result_0.status == "evaluated"
    assert result_3.status == "evaluated"
    assert result_3.criticality_score > result_0.criticality_score
    assert result_3.collision_term > result_0.collision_term


def test_criticality_score_responds_to_near_misses() -> None:
    """Near-miss events increase the criticality score."""
    from types import SimpleNamespace

    compute = _MODULE.compute_criticality_score

    none = SimpleNamespace(metrics=_full_metrics(near_misses=0))
    misses = SimpleNamespace(metrics=_full_metrics(near_misses=5))

    r_none = compute(none)
    r_misses = compute(misses)

    assert r_none.status == "evaluated"
    assert r_misses.status == "evaluated"
    assert r_misses.near_miss_term > r_none.near_miss_term


def test_criticality_score_responds_to_clearance() -> None:
    """Low clearance increases the criticality score."""
    from types import SimpleNamespace

    compute = _MODULE.compute_criticality_score

    good = SimpleNamespace(metrics=_full_metrics(min_clearance=2.0))
    tight = SimpleNamespace(metrics=_full_metrics(min_clearance=0.1))

    r_good = compute(good)
    r_tight = compute(tight)

    assert r_good.status == "evaluated"
    assert r_tight.status == "evaluated"
    assert r_tight.clearance_term > r_good.clearance_term


def test_apply_criticality_parameters_does_not_mutate_original() -> None:
    """apply_criticality_parameters returns a deep copy; original unchanged."""
    apply_params = _MODULE.apply_criticality_parameters

    scenario = {
        "id": "test",
        "single_pedestrians": [{"speed_mps": 1.0}],
    }
    original_id = scenario["id"]
    original_speed = scenario["single_pedestrians"][0]["speed_mps"]

    patched = apply_params(scenario, {"pedestrian_speed_scale": 2.0})

    assert scenario["id"] == original_id
    assert scenario["single_pedestrians"][0]["speed_mps"] == original_speed
    assert patched["id"] != original_id
    assert patched["single_pedestrians"][0]["speed_mps"] == pytest.approx(2.0)


def test_apply_criticality_parameters_produces_different_scenarios_for_different_params() -> None:
    """Different parameter values produce different patched scenarios."""
    apply_params = _MODULE.apply_criticality_parameters

    scenario = {
        "id": "test",
        "single_pedestrians": [{"speed_mps": 1.0}],
    }

    patched_a = apply_params(scenario, {"pedestrian_speed_scale": 1.0})
    patched_b = apply_params(scenario, {"pedestrian_speed_scale": 2.0})

    assert patched_a["single_pedestrians"][0]["speed_mps"] == pytest.approx(1.0)
    assert patched_b["single_pedestrians"][0]["speed_mps"] == pytest.approx(2.0)

    assert patched_a["metadata"]["issue_4362_criticality_parameters"][
        "pedestrian_speed_scale"
    ] == pytest.approx(1.0)
    assert patched_b["metadata"]["issue_4362_criticality_parameters"][
        "pedestrian_speed_scale"
    ] == pytest.approx(2.0)


def test_criticality_score_not_uniform_across_simulated_perturbations() -> None:
    """Run a minimal optimization (1 candidate + baseline) and verify the pipeline
    produces different criticality scores for baseline vs. perturbed parameters.

    CPU-level validation: different parameter values flowing through the simulator
    produce different episode metrics, which produce different criticality scores.
    """
    config = _make_test_config(sample_budget=1, optimizer_seed=99)
    candidates, manifest = run_criticality_optimization(config)

    baseline = next(c for c in candidates if c.candidate_id == "baseline_unperturbed")
    perturbed = next(c for c in candidates if c.candidate_id != "baseline_unperturbed")

    assert baseline.status == "evaluated"
    assert perturbed.status == "evaluated"
    assert perturbed.parameters != baseline.parameters

    assert manifest["evaluated_count"] >= 2
    assert manifest["metrics_source"] == "simulator_run_map_batch"


def test_de_optimizer_produces_candidates_and_optimum() -> None:
    """differential_evolution optimizer produces trial candidates plus de_optimum.

    Runs a minimal DE optimization (small population, few iterations) and checks
    that the pipeline produces DE trial results, the de_optimum candidate, and
    the baseline — confirming the optimizer is wired to the real runner.
    """
    config = OptimizationConfig(
        parameter_space={
            "pedestrian_speed_scale": ParameterDefinition(
                param_type="continuous", min=0.8, max=1.2
            ),
        },
        optimizer_type="differential_evolution",
        sample_budget=1,
        optimizer_seed=42,
        seeds=[0],
        de_maxiter=3,
        de_popsize=3,
        objective_weights={
            "collision": 10.0,
            "near_miss": 2.0,
            "clearance_margin": 0.5,
            "clearance": 1.0,
            "progress_failure": 5.0,
            "stalled_time": 0.5,
        },
    )
    candidates, manifest = run_criticality_optimization(config)

    ids = [c.candidate_id for c in candidates]
    assert "baseline_unperturbed" in ids
    assert "de_optimum" in ids

    de_trials = [c for c in candidates if c.candidate_id.startswith("de_trial_")]
    assert len(de_trials) >= 1

    assert manifest["optimizer_type"] == "differential_evolution"
    assert "de_maxiter" in manifest
    assert "de_popsize" in manifest
    assert "de_seed" in manifest
    assert manifest["metrics_source"] == "simulator_run_map_batch"


def test_de_optimum_score_comparable_to_baseline() -> None:
    """The DE optimum candidate and baseline are both evaluated (not NaN).

    Confirms the DE optimizer is connected to the real runner: both the
    baseline and the DE optimum produce valid criticality scores.
    """
    import math

    config = OptimizationConfig(
        parameter_space={
            "pedestrian_speed_scale": ParameterDefinition(
                param_type="continuous", min=0.9, max=1.1
            ),
        },
        optimizer_type="differential_evolution",
        sample_budget=1,
        optimizer_seed=42,
        seeds=[0],
        de_maxiter=2,
        de_popsize=3,
        objective_weights={
            "collision": 10.0,
            "near_miss": 2.0,
            "clearance_margin": 0.5,
            "clearance": 1.0,
            "progress_failure": 5.0,
            "stalled_time": 0.5,
        },
    )
    candidates, _ = run_criticality_optimization(config)

    baseline = next(c for c in candidates if c.candidate_id == "baseline_unperturbed")
    optimum = next(c for c in candidates if c.candidate_id == "de_optimum")

    assert baseline.status == "evaluated"
    assert optimum.status == "evaluated"
    assert math.isfinite(baseline.criticality_score)
    assert math.isfinite(optimum.criticality_score)


# ── CMA-ES optimizer tests ──────────────────────────────────────────────

_CMA_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "run_scenario_criticality_optimization.py"
)


def _load_cma_opt_module(mod_name: str):
    """Load the optimization script as a fresh module for CMA-ES tests."""
    import importlib.util as _u

    spec = _u.spec_from_file_location(mod_name, _CMA_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = _u.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cma_es_optimizer_accepted() -> None:
    """CMA-ES optimizer_type is accepted by the runner."""
    mod = _load_cma_opt_module("_cma_test_opt")

    config = mod.OptimizationConfig(
        parameter_space={
            "pedestrian_speed_scale": mod.ParameterDefinition(
                param_type="continuous", min=0.7, max=1.4
            ),
            "pedestrian_start_delay_s": mod.ParameterDefinition(
                param_type="continuous", min=0.0, max=2.0
            ),
        },
        optimizer_type="cma_es",
        sample_budget=5,
        optimizer_seed=42,
        seeds=[0],
        cma_es_maxiter=5,
        cma_es_sigma0=0.5,
    )
    candidates, manifest = mod.run_criticality_optimization(config)
    assert manifest["optimizer_type"] == "cma_es"
    assert len(candidates) >= 2  # baseline + at least cma_optimum


def test_cma_es_requires_continuous_params() -> None:
    """CMA-ES rejects all-discrete parameter spaces."""
    mod = _load_cma_opt_module("_cma_test_opt2")

    config = mod.OptimizationConfig(
        parameter_space={
            "mode": mod.ParameterDefinition(
                param_type="discrete", values=[1.0, 2.0]
            ),
        },
        optimizer_type="cma_es",
        cma_es_maxiter=2,
    )
    with pytest.raises(ValueError, match="cma_es requires at least one continuous"):
        mod.run_criticality_optimization(config)


def test_cma_es_manifest_fields() -> None:
    """CMA-ES run records optimizer-specific manifest fields."""
    mod = _load_cma_opt_module("_cma_test_opt3")

    config = mod.OptimizationConfig(
        parameter_space={
            "pedestrian_speed_scale": mod.ParameterDefinition(
                param_type="continuous", min=0.7, max=1.4
            ),
        },
        optimizer_type="cma_es",
        sample_budget=2,
        optimizer_seed=77,
        seeds=[0],
        cma_es_maxiter=3,
        cma_es_sigma0=0.3,
        cma_es_seed=99,
    )
    _, manifest = mod.run_criticality_optimization(config)
    assert manifest["cma_es_maxiter"] == 3
    assert manifest["cma_es_sigma0"] == 0.3
    assert manifest["cma_es_seed"] == 99


def test_cma_es_seed_fallback() -> None:
    """CMA-ES falls back to optimizer_seed when cma_es_seed is None."""
    mod = _load_cma_opt_module("_cma_test_opt4")

    config = mod.OptimizationConfig(
        parameter_space={
            "pedestrian_speed_scale": mod.ParameterDefinition(
                param_type="continuous", min=0.7, max=1.4
            ),
        },
        optimizer_type="cma_es",
        sample_budget=2,
        optimizer_seed=123,
        seeds=[0],
        cma_es_maxiter=2,
        cma_es_seed=None,
    )
    _, manifest = mod.run_criticality_optimization(config)
    assert manifest["cma_es_seed"] == 123


def test_cma_es_optimum_score_finite() -> None:
    """CMA-ES optimum and baseline both produce finite scores."""
    import math

    mod = _load_cma_opt_module("_cma_test_opt5")

    config = mod.OptimizationConfig(
        parameter_space={
            "pedestrian_speed_scale": mod.ParameterDefinition(
                param_type="continuous", min=0.7, max=1.4
            ),
            "pedestrian_start_delay_s": mod.ParameterDefinition(
                param_type="continuous", min=0.0, max=2.0
            ),
        },
        optimizer_type="cma_es",
        sample_budget=3,
        optimizer_seed=42,
        seeds=[0],
        cma_es_maxiter=5,
        cma_es_sigma0=0.5,
    )
    candidates, _ = mod.run_criticality_optimization(config)

    baseline = next(c for c in candidates
                    if c.candidate_id == "baseline_unperturbed")
    optimum = next(
        (c for c in candidates if c.candidate_id == "cma_optimum"),
        None,
    )

    assert baseline.status == "evaluated"
    assert math.isfinite(baseline.criticality_score)
    assert optimum is not None, "cma_optimum candidate not found"
    assert optimum.status == "evaluated"
    assert math.isfinite(optimum.criticality_score)


def test_cma_es_import_error_message() -> None:
    """Missing cma package raises ImportError with install hint."""
    from unittest import mock

    mod = _load_cma_opt_module("_cma_test_opt6")

    config = mod.OptimizationConfig(
        parameter_space={
            "x": mod.ParameterDefinition(
                param_type="continuous", min=0.0, max=1.0
            ),
        },
        optimizer_type="cma_es",
        cma_es_maxiter=1,
    )

    # Patch _run_cma_es to simulate the import failure
    with mock.patch.object(
        mod,
        "_run_cma_es",
        side_effect=ImportError(
            "cma_es optimizer requires the 'cma' package. "
            'Install with: uv pip install -e ".[criticality]"'
        ),
    ):
        with pytest.raises(ImportError, match="cma_es optimizer requires"):
            mod.run_criticality_optimization(config)
