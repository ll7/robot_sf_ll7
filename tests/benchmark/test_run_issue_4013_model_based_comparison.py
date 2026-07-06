"""Tests for the issue #4013 paired diagnostic model-based comparison runner.

These are lightweight contract tests for the runner's config wiring and checkpoint
handling. The full end-to-end run (three ``map_runner`` arms + report) is validated
manually and recorded in the PR; it is too heavy for a unit test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    script_path = _REPO_ROOT / "scripts/benchmark/run_issue_4013_model_based_comparison.py"
    spec = importlib.util.spec_from_file_location("issue_4013_comparison_runner", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_role_mapping_covers_analysis_config_required_roles() -> None:
    """Every role declared in the analysis config maps to an existing benchmark config."""
    runner = _load_runner()
    analysis_config = yaml.safe_load(
        (_REPO_ROOT / runner.DEFAULT_ANALYSIS_CONFIG).read_text(encoding="utf-8")
    )
    roles = {str(run["role"]) for run in analysis_config["runs"]}
    assert roles == set(runner.ROLE_TO_BENCHMARK_CONFIG), (
        "runner role map must exactly cover the analysis config roles"
    )
    for config_path in runner.ROLE_TO_BENCHMARK_CONFIG.values():
        assert (_REPO_ROOT / config_path).is_file(), f"missing benchmark config {config_path}"


def test_benchmark_configs_declare_single_planner_and_seeds() -> None:
    """Each mapped benchmark config declares exactly one planner and a fixed seed list.

    The two configs added for this run (model-based checkpoint arm, model-free baseline
    arm) also label their ``issue_4013_role`` to match the analysis role. The pre-existing
    ``cv_prediction_mpc`` config keeps its historical ``model_free_comparator`` label
    (consumed by the #4474 comparator preflight), so its role label is not asserted here.
    """
    runner = _load_runner()
    new_arm_roles = {"learned_prediction_mpc", "model_free_baseline"}
    for role, config_path in runner.ROLE_TO_BENCHMARK_CONFIG.items():
        payload = yaml.safe_load((_REPO_ROOT / config_path).read_text(encoding="utf-8"))
        planner = runner._planner_spec(payload, path=config_path)
        assert planner.get("algo"), "planner must declare an algo"
        seeds = runner._fixed_list_seeds(payload, path=config_path)
        assert seeds, "benchmark config must declare a non-empty fixed seed list"
        if role in new_arm_roles:
            assert planner["issue_4013_role"] == role


def test_scenario_payload_overrides_matrix_seeds_with_config_seeds(tmp_path: Path) -> None:
    """The runner injects the benchmark-config fixed seeds into each scenario entry."""
    runner = _load_runner()
    matrix = tmp_path / "scenarios.yaml"
    matrix.write_text(
        yaml.safe_dump({"scenarios": [{"name": "s1", "seeds": [999]}]}), encoding="utf-8"
    )
    payload = {
        "scenario_matrix": str(matrix),
        "seed_policy": {"mode": "fixed-list", "seeds": [4013]},
    }
    scenarios = runner._scenario_payload_with_config_seeds(payload, path=tmp_path / "cfg.yaml")
    assert scenarios == [{"name": "s1", "seeds": [4013]}]


def test_checkpoint_path_resolved_from_algo_config() -> None:
    """The model-based checkpoint path is read from the checkpoint algo config."""
    runner = _load_runner()
    checkpoint = runner._checkpoint_path_from_algo_config(
        _REPO_ROOT / runner.CHECKPOINT_ALGO_CONFIG
    )
    assert checkpoint.name.endswith(".pt")


def test_ensure_checkpoint_reuses_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the checkpoint already exists, ensure_checkpoint reuses it without training."""
    runner = _load_runner()
    checkpoint = tmp_path / "short_horizon_predictor.pt"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setattr(
        runner, "_checkpoint_path_from_algo_config", lambda _algo_config: checkpoint
    )
    note = runner.ensure_checkpoint(trainer_config=Path("unused.yaml"), skip_train=True)
    assert note["action"] == "reused_existing"
    assert note["checkpoint"] == str(checkpoint)


def test_ensure_checkpoint_fails_closed_when_missing_and_skip_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing checkpoint with --skip-train fails closed instead of silently degrading."""
    runner = _load_runner()
    missing = tmp_path / "absent.pt"
    monkeypatch.setattr(runner, "_checkpoint_path_from_algo_config", lambda _algo_config: missing)
    with pytest.raises(FileNotFoundError):
        runner.ensure_checkpoint(trainer_config=Path("unused.yaml"), skip_train=True)


def test_report_builder_is_callable() -> None:
    """The runner can load the canonical comparison report builder by file path."""
    runner = _load_runner()
    builder = runner._load_report_builder()
    assert callable(builder)
