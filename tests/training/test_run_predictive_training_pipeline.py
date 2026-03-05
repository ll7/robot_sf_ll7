"""Unit tests for predictive training pipeline helper utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from scripts.training import run_predictive_training_pipeline as pipeline

if TYPE_CHECKING:
    from pathlib import Path


def test_paths_from_config_resolves_root_relative_to_output_base(tmp_path: Path) -> None:
    """Resolve output root relative to output base directory and run id."""
    cfg = {"output": {"root": "output/tmp/predictive_planner/pipeline"}}
    run_id = "predictive_test_run"

    paths = pipeline._paths_from_config(
        cfg,
        run_id=run_id,
        base_dir=tmp_path,
        output_base_dir=tmp_path,
    )
    assert paths.root == (tmp_path / "output/tmp/predictive_planner/pipeline" / run_id).resolve()
    assert paths.base_dataset.name == "predictive_rollouts_base.npz"
    assert paths.checkpoint.name == "predictive_model.pt"
    assert paths.final_summary.name == "final_performance_summary.json"


def test_build_random_seed_manifest_generates_all_scenarios(monkeypatch, tmp_path: Path) -> None:
    """Generate deterministic per-scenario random seed lists for all scenarios."""
    scenario_matrix = tmp_path / "scenarios.yaml"
    scenario_matrix.write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(
        pipeline,
        "load_scenarios",
        lambda _path: [
            {"name": "classic_crossing_low"},
            {"name": "classic_doorway_high"},
        ],
    )

    out_path = tmp_path / "manifest.yaml"
    manifest_path = pipeline._build_random_seed_manifest(
        scenario_matrix=scenario_matrix,
        seeds_per_scenario=3,
        random_seed_base=100,
        output_path=out_path,
    )

    assert manifest_path == out_path
    payload = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert payload["classic_crossing_low"] == [100, 101, 102]
    assert payload["classic_doorway_high"] == [100100, 100101, 100102]
