"""Tests for multi-seed orchestration (User Story 2)."""

from __future__ import annotations

import pytest

from robot_sf.research.orchestrator import AblationOrchestrator, ReportOrchestrator
from tests.fixtures.minimal_manifests.generator import create_seed_set


def test_multi_seed_execution(tmp_path):
    """All seeds completed should yield full completeness."""

    baseline_manifests = create_seed_set("baseline", [101, 102, 103], tmp_path / "baseline")
    pretrained_manifests = create_seed_set("pretrained", [101, 102, 103], tmp_path / "pretrained")

    orchestrator = ReportOrchestrator(output_dir=tmp_path / "reports")
    records, completeness, seed_status = orchestrator.orchestrate_multi_seed(
        baseline_manifests,
        pretrained_manifests,
        expected_seeds=[101, 102, 103],
    )

    assert len(records) == 6  # 3 seeds × 2 conditions
    assert completeness["score"] == 100.0
    assert completeness["missing_seeds"] == []
    assert all(entry["baseline_status"] == "completed" for entry in seed_status)
    assert all(entry["pretrained_status"] == "completed" for entry in seed_status)


def test_partial_seed_failure(tmp_path):
    """Missing baseline manifest should reduce completeness score."""

    baseline_manifests = create_seed_set("baseline", [1, 2], tmp_path / "baseline")
    pretrained_manifests = create_seed_set("pretrained", [1, 2, 3], tmp_path / "pretrained")

    # Seed 3 baseline missing (no manifest)
    orchestrator = ReportOrchestrator(output_dir=tmp_path / "reports_partial")
    records, completeness, seed_status = orchestrator.orchestrate_multi_seed(
        baseline_manifests,
        pretrained_manifests,
        expected_seeds=[1, 2, 3],
    )

    assert len(records) == 5  # baseline (2) + pretrained (3)
    assert completeness["score"] == pytest.approx(66.7, rel=1e-2)
    assert "3" in completeness["missing_seeds"]

    missing_entry = next(entry for entry in seed_status if entry["seed"] == 3)
    assert missing_entry["baseline_status"] == "missing"
    assert missing_entry["pretrained_status"] == "completed"


def test_ablation_matrix_generation(tmp_path):
    """Test ablation matrix generation count (T069)."""
    params = {"bc_epochs": [5, 10], "dataset_size": [100, 200, 300]}
    orch = AblationOrchestrator(
        experiment_name="AblationTest",
        seeds=[1, 2],
        ablation_params=params,
        threshold=40.0,
        output_dir=tmp_path / "ablation",
    )
    variants = orch.run_ablation_matrix()
    assert len(variants) == 2 * 3  # Cartesian product
    # Each variant should have improvement_pct and decision
    assert all("improvement_pct" in v for v in variants)
    assert all("decision" in v for v in variants)


def test_incomplete_ablation(tmp_path):
    """Test marking incomplete variants (T073)."""
    orch = AblationOrchestrator(
        experiment_name="AblationIncomplete",
        seeds=[1],
        ablation_params={"bc_epochs": [5], "dataset_size": [100]},
        threshold=40.0,
        output_dir=tmp_path / "ablation_incomplete",
    )
    variants = orch.generate_matrix()
    # Simulate missing improvement by not running matrix evaluation
    marked = orch.handle_incomplete_variants(variants)
    assert marked[0]["decision"] == "INCOMPLETE"


def test_ablation_report_generation(tmp_path):
    """Integration style test for ablation report (T072)."""
    params = {"bc_epochs": [5, 10], "dataset_size": [100]}  # small set
    orch = AblationOrchestrator(
        experiment_name="AblationReport",
        seeds=[1, 2],
        ablation_params=params,
        threshold=40.0,
        output_dir=tmp_path / "ablation_report",
    )
    variants = orch.run_ablation_matrix()
    report_path = orch.generate_ablation_report(variants)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Ablation Matrix" in content
    assert "bc5_ds100" in content or "bc10_ds100" in content
