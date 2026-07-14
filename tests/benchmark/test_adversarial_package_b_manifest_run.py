"""Tests for the issue #3079 Package B manifest-driven comparison pipeline."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.adversarial_package_b_confirmation import (
    build_package_b_confirmation_sidecar,
    validate_package_b_confirmation,
)
from robot_sf.benchmark.adversarial_package_b_preflight import preflight_package_b_manifest
from robot_sf.benchmark.adversarial_package_b_report import validate_package_b_report
from scripts.tools.compare_adversarial_samplers import (
    build_comparison_payload,
    load_package_b_manifest,
    run_sampler_comparison,
)
from scripts.tools.run_adversarial_package_b import _run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_MANIFEST = REPO_ROOT / "configs/adversarial/issue_3079_package_b_budget_matched.yaml"


def test_manifest_drives_27_cell_budget_matched_run(tmp_path: Path) -> None:
    """The committed manifest produces exactly the 27-cell Package-B matrix."""
    config, samplers, budgets, seeds = load_package_b_manifest(SHIPPED_MANIFEST)
    report_json = tmp_path / "report.json"
    rows = run_sampler_comparison(
        config=config,
        sampler_names=samplers,
        objective_names=(config.objective,),
        synthetic=True,
        budgets=budgets,
        seeds=seeds,
    )

    payload = build_comparison_payload(
        rows=rows,
        objectives=(config.objective,),
        budgets=budgets,
        seeds=seeds,
    )
    report_json.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")

    gate = validate_package_b_report(report_json)
    assert gate.ready is True
    assert gate.matrix["observed_row_count"] == 27
    assert gate.status == "ready_for_empirical_review"


def test_package_b_pipeline_produces_artifacts_and_passes_gates(tmp_path: Path) -> None:
    """The orchestrator emits the durable report + confirmation and both gates pass."""
    summary = _run_pipeline(SHIPPED_MANIFEST, repo_root=REPO_ROOT)
    assert summary["stage"] == "complete"
    assert summary["preflight_ready"] is True
    assert summary["report_gate"]["ready"] is True
    assert summary["confirmation_gate"]["ready"] is True

    report_path = REPO_ROOT / summary["report_json"]
    confirmation_path = REPO_ROOT / summary["confirmation_json"]
    assert report_path.exists()
    assert confirmation_path.exists()


def test_confirmation_sidecar_is_censored_and_artifact_bound(tmp_path: Path) -> None:
    """The confirmation sidecar marks every cell censored and binds to the report sha."""
    config, samplers, budgets, seeds = load_package_b_manifest(SHIPPED_MANIFEST)
    report_json = tmp_path / "report.json"
    rows = run_sampler_comparison(
        config=config,
        sampler_names=samplers,
        objective_names=(config.objective,),
        synthetic=True,
        budgets=budgets,
        seeds=seeds,
    )
    report_json.write_text(
        __import__("json").dumps(
            build_comparison_payload(
                rows=rows,
                objectives=(config.objective,),
                budgets=budgets,
                seeds=seeds,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    confirmation_path = tmp_path / "confirmation.json"
    sidecar = build_package_b_confirmation_sidecar(report_json, confirmation_path=confirmation_path)
    sidecar.write()
    assert confirmation_path.exists()

    gate = validate_package_b_confirmation(report_json, confirmation_path)
    assert gate.ready is True
    assert gate.status == "ready_for_confirmed_failure_review"
    assert gate.matrix["confirmed_failure_count"] == 0
    assert gate.matrix["censored_row_count"] == 27
    assert gate.matrix["observed_row_count"] == 27


def test_committed_manifest_preflights_before_run() -> None:
    """The shipped manifest passes the fail-closed preflight used by the pipeline."""
    result = preflight_package_b_manifest(SHIPPED_MANIFEST, repo_root=REPO_ROOT)
    assert result.ready is True
