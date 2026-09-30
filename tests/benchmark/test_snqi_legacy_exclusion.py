"""Dev-only regression for exclusion across acquisition and publication bytes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.artifact_publication import (
    export_publication_bundle,
    verify_publication_bundle_preflight,
)
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._config_types import SeedPolicy
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config, run_campaign
from robot_sf.evidence.writers import write_json

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_5.yaml"
TEMPLATE = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)
RELEASE = ROOT / "configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml"


@pytest.mark.parametrize("source", [TEMPLATE, SMOKE], ids=["release-path", "smoke"])
def test_excluded_legacy_snqi_dev_campaign_writes_episodes_and_publication(tmp_path, source):
    """Real candidate bytes run a dev episode and cold-verify without legacy assets."""
    cfg = load_campaign_config(source)
    smoke = load_campaign_config(SMOKE)
    cfg = replace(
        cfg,
        scenario_matrix_path=smoke.scenario_matrix_path,
        planners=(next(planner for planner in cfg.planners if planner.key == "goal"),),
        seed_policy=SeedPolicy(
            mode="fixed-list", seeds=(1001,), seed_sets_path=cfg.seed_policy.seed_sets_path
        ),
        workers=1,
        arm_isolation="in_process",
        bootstrap_samples=10,
        export_publication_bundle=False,
    )
    assert {seed for scenario in _load_campaign_scenarios(cfg) for seed in scenario["seeds"]} == {
        1001
    }
    result = run_campaign(
        cfg, output_root=tmp_path, campaign_id="dev-probe", skip_publication_bundle=True
    )
    root = Path(result["campaign_root"])
    rows = [
        json.loads(line)
        for path in root.glob("runs/*/episodes.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 1 and rows[0]["seed"] == 1001
    summary = json.loads((root / "reports/campaign_summary.json").read_text())
    manifest = yaml.safe_load(RELEASE.read_text())
    for key in ("snqi_weights_path", "snqi_baseline_path"):
        if manifest.get("metrics", {}).get(key):
            manifest["metrics"][key] = str(
                (RELEASE.parent / manifest["metrics"][key]).resolve().relative_to(ROOT)
            )
    manifest["provenance"] = {"citation_path": "CITATION.cff"}
    release = root / "release"
    release.mkdir()
    write_json(release / "release_manifest.resolved.json", manifest)
    write_json(
        release / "release_result.json",
        {
            key: summary["campaign"][key]
            for key in ("status", "evidence_status", "total_episodes", "successful_runs")
        },
    )
    bundle = export_publication_bundle(
        root, tmp_path / "bundles", include_videos=False, release_tag="snqifix2-dev-probe", doi=""
    )
    report = verify_publication_bundle_preflight(bundle.bundle_dir)
    assert report["violation_count"] == 0
    assert report["evidence"]["snqi_field_consistency"]["checked"] is False
    assert not (root / "reports/snqi_diagnostics.json").exists()
    assert not (bundle.bundle_dir / "payload/release_metadata/snqi").exists()
    assert summary["campaign"]["snqi_v2"] == "pending_calibration"
    assert "snqi_contract_status" not in summary["campaign"]
    assert "mean_snqi" not in json.dumps(summary)
    for path in (root / "reports").glob("*.csv"):
        assert not any(
            column.startswith("snqi")
            for column in (path.read_text().splitlines() or [""])[0].split(",")
        )
