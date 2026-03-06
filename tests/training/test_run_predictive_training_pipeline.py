"""Unit tests for predictive training pipeline helper utilities."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
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


def test_run_capture_json_anchors_subprocess_to_repo_root(monkeypatch) -> None:
    """Subprocess helpers should execute repo-relative commands from the repository root."""
    called: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        called["cmd"] = cmd
        called["cwd"] = kwargs.get("cwd")
        called["env"] = kwargs.get("env")
        return type("Result", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()

    monkeypatch.setattr(pipeline.subprocess, "run", _fake_run)
    payload = pipeline._run_capture_json(["python", "scripts/example.py"])
    assert payload["status"] == "ok"
    assert called["cwd"] == pipeline._REPO_ROOT


def test_write_dataset_manifest_records_contract_and_hashes(tmp_path: Path) -> None:
    """Dataset manifests should capture reset-v2 provenance and dataset digest."""
    dataset_path = tmp_path / "predictive_rollouts_base.npz"
    np.savez(
        dataset_path,
        state=np.zeros((2, 3, 4), dtype=np.float32),
        target=np.zeros((2, 3, 5, 2), dtype=np.float32),
        mask=np.ones((2, 3), dtype=np.float32),
        target_mask=np.ones((2, 3, 5), dtype=np.float32),
    )
    summary_path = dataset_path.with_suffix(".json")
    summary_path.write_text(json.dumps({"num_samples": 2}), encoding="utf-8")

    manifest_path = pipeline._write_dataset_manifest(
        dataset_path=dataset_path,
        summary_path=summary_path,
        role="predictive_base_dataset",
        run_id="run_123",
        config_path=tmp_path / "config.yaml",
        config_hash="abc123",
        git_commit="deadbeef",
        extra={"seed_manifest": "manifest.yaml"},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["contract_version"] == "benchmark-reset-v2"
    assert payload["training_family"] == "prediction_planner"
    assert payload["dataset_sha1"]
    assert payload["diagnostics"]["num_samples"] == 2


def test_dataset_npz_diagnostics_counts_empty_rows(tmp_path: Path) -> None:
    """NPZ diagnostics should surface empty agent and target rows."""
    dataset_path = tmp_path / "predictive_rollouts_mixed.npz"
    mask = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    target_mask = np.zeros((2, 2, 3), dtype=np.float32)
    target_mask[0, 0, :] = 1.0
    np.savez(
        dataset_path,
        state=np.zeros((2, 2, 4), dtype=np.float32),
        target=np.zeros((2, 2, 3, 2), dtype=np.float32),
        mask=mask,
        target_mask=target_mask,
    )

    payload = pipeline._dataset_npz_diagnostics(dataset_path)
    assert payload["empty_agent_rows"] == 1
    assert payload["empty_target_rows"] == 1
