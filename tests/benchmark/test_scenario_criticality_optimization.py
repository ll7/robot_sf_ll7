"""Tests for scenario criticality optimization runner (Issue #4362)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "run_scenario_criticality_optimization.py"
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
    assert "scenario criticality" in result.stdout.lower() or "optimization" in result.stdout.lower()
