"""Tests for the issue #3079 Package B manifest-driven comparison pipeline."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import yaml

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
from scripts.tools.run_adversarial_package_b import main as run_package_b_main

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_MANIFEST = REPO_ROOT / "configs/adversarial/issue_3079_package_b_budget_matched.yaml"


def _copy_pipeline_fixture(tmp_path: Path) -> Path:
    """Build a minimal repository-shaped fixture for the end-to-end pipeline test."""
    relative_paths = (
        "configs/adversarial/issue_3079_package_b_budget_matched.yaml",
        "configs/adversarial/crossing_ttc_space.yaml",
        "configs/research/research_package_registry_issue_3057.yaml",
        "configs/scenarios/templates/crossing_ttc.yaml",
        "scripts/tools/compare_adversarial_samplers.py",
    )
    for relative_path in relative_paths:
        source = REPO_ROOT / relative_path
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path / relative_paths[0]


def test_manifest_drives_27_cell_budget_matched_run(tmp_path: Path) -> None:
    """The committed manifest produces exactly the 27-cell Package-B matrix."""
    config, samplers, budgets, seeds = load_package_b_manifest(SHIPPED_MANIFEST)
    config = replace(config, output_dir=tmp_path / "comparison")
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
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    gate = validate_package_b_report(report_json)
    assert gate.ready is True
    assert gate.matrix["observed_row_count"] == 27
    assert gate.status == "ready_for_empirical_review"


def test_package_b_pipeline_produces_artifacts_and_passes_gates(tmp_path: Path) -> None:
    """The orchestrator emits the durable report + confirmation and both gates pass."""
    manifest = _copy_pipeline_fixture(tmp_path)
    summary_path = tmp_path / "pipeline-summary.json"
    assert (
        run_package_b_main(
            [
                "--manifest",
                str(manifest),
                "--repo-root",
                str(tmp_path),
                "--output",
                str(summary_path),
                "--fail-closed",
            ]
        )
        == 0
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["stage"] == "complete"
    assert summary["preflight_ready"] is True
    assert summary["report_gate"]["ready"] is True
    assert summary["confirmation_gate"]["ready"] is True

    report_path = tmp_path / summary["report_json"]
    confirmation_path = tmp_path / summary["confirmation_json"]
    assert report_path.exists()
    assert confirmation_path.exists()


def test_confirmation_sidecar_is_censored_and_artifact_bound(tmp_path: Path) -> None:
    """The confirmation sidecar marks every cell censored and binds to the report sha."""
    config, samplers, budgets, seeds = load_package_b_manifest(SHIPPED_MANIFEST)
    config = replace(config, output_dir=tmp_path / "comparison")
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
        json.dumps(
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
    sidecar_payload = json.loads(confirmation_path.read_text(encoding="utf-8"))
    assert all(not key.startswith("_") for row in sidecar_payload["rows"] for key in row)


def test_manifest_paths_resolve_against_explicit_repo_root(tmp_path: Path, monkeypatch) -> None:
    """Manifest paths stay valid when the caller's working directory differs."""
    monkeypatch.chdir(tmp_path)
    config, _samplers, _budgets, _seeds = load_package_b_manifest(
        SHIPPED_MANIFEST,
        repo_root=REPO_ROOT,
    )

    assert config.scenario_template == REPO_ROOT / "configs/scenarios/templates/crossing_ttc.yaml"
    assert config.search_space_path == REPO_ROOT / "configs/adversarial/crossing_ttc_space.yaml"
    assert config.output_dir == REPO_ROOT / "output/adversarial/issue_3079_package_b"


def test_synthetic_run_preserves_adversarial_candidate_state_fixture(tmp_path: Path) -> None:
    """Synthetic artifacts retain the candidate state used by the adversarial search."""
    config, _samplers, _budgets, _seeds = load_package_b_manifest(SHIPPED_MANIFEST)
    config = replace(config, output_dir=tmp_path / "comparison")
    rows = run_sampler_comparison(
        config=config,
        sampler_names=("random",),
        objective_names=(config.objective,),
        synthetic=True,
        budgets=(16,),
        seeds=(1101,),
    )

    manifest = json.loads(Path(rows[0].manifest_path).read_text(encoding="utf-8"))
    candidate = manifest["candidates"][0]
    scenario_path = Path(candidate["scenario_yaml_path"])
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    stored_candidate = scenario_payload["scenarios"][0]["metadata"]["adversarial_candidate"]
    assert stored_candidate["start"] == candidate["candidate"]["start"]
    assert stored_candidate["goal"] == candidate["candidate"]["goal"]
    episode_path = Path(candidate["episode_record_path"])
    episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[0])
    assert episode["seed"] == candidate["candidate"]["scenario_seed"]


def test_committed_manifest_preflights_before_run() -> None:
    """The shipped manifest passes the fail-closed preflight used by the pipeline."""
    result = preflight_package_b_manifest(SHIPPED_MANIFEST, repo_root=REPO_ROOT)
    assert result.ready is True
