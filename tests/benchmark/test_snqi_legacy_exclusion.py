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


@pytest.mark.slow
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
    manifest["release_kind"] = "development_rehearsal"
    release = root / "release"
    release.mkdir()
    write_json(release / "release_manifest.resolved.json", manifest)
    write_json(
        release / "release_result.json",
        {
            key: summary["campaign"][key]
            for key in ("status", "evidence_status", "total_episodes", "successful_runs")
        }
        | {
            "benchmark_release": {"release_kind": "development_rehearsal"},
            "release_eligible": False,
            "release_benchmark_success": False,
        },
    )
    bundle = export_publication_bundle(
        root, tmp_path / "bundles", include_videos=False, release_tag="snqifix2-dev-probe", doi=""
    )
    report = verify_publication_bundle_preflight(bundle.bundle_dir)
    assert report["violation_count"] == 0
    assert report["release_eligible"] is False
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


def test_release_bundle_rejects_weights_without_baseline(tmp_path):
    """A release with only one legacy asset fails on the paired-asset contract."""
    run_root = tmp_path / "run"
    (run_root / "release").mkdir(parents=True)
    write_json(
        run_root / "release/release_manifest.resolved.json",
        {
            "metrics": {
                "snqi_weights_path": "configs/benchmarks/snqi_weights_camera_ready_v3.json"
            },
            "provenance": {"citation_path": "CITATION.cff"},
        },
    )
    write_json(run_root / "release/release_result.json", {"status": "ok"})
    with pytest.raises(
        ValueError, match="^Release legacy SNQI requires both weights and baseline$"
    ):
        export_publication_bundle(run_root, tmp_path / "bundles", include_videos=False)


def _excluded_bundle_payload(tmp_path):
    """Create a real excluded payload for cold-verification guard tests.

    Returns:
        The bundle payload directory.
    """
    payload = tmp_path / "bundle/payload"
    payload.mkdir(parents=True)
    write_json(payload / "campaign_manifest.json", {"legacy_snqi": "excluded"})
    return payload


def test_excluded_bundle_rejects_legacy_diagnostics(tmp_path):
    """Cold verification must detect diagnostics contradicting explicit exclusion."""
    from robot_sf.benchmark.artifact_publication import _publication_snqi_evidence

    payload = _excluded_bundle_payload(tmp_path)
    (payload / "reports").mkdir()
    write_json(payload / "reports/snqi_diagnostics.json", {})
    assert _publication_snqi_evidence(payload) == {
        "checked": False,
        "reason": "legacy_snqi_excluded",
        "violations": ["Legacy SNQI is excluded but its diagnostics are present"],
    }


def test_excluded_bundle_rejects_declared_legacy_assets(tmp_path):
    """Cold verification must reject paired legacy declarations in an excluded bundle."""
    from robot_sf.benchmark.artifact_publication import _publication_snqi_evidence

    payload = _excluded_bundle_payload(tmp_path)
    (payload / "release").mkdir()
    write_json(
        payload / "release/release_manifest.resolved.json",
        {"metrics": {"snqi_weights_path": "weights.json", "snqi_baseline_path": "baseline.json"}},
    )
    assert _publication_snqi_evidence(payload) == {
        "checked": False,
        "reason": "legacy_snqi_excluded",
        "violations": ["Legacy SNQI is excluded but the release declares legacy assets"],
    }


def test_excluded_bundle_rejects_staged_legacy_asset_directory(tmp_path):
    """Cold verification must detect the reserved legacy-asset directory itself."""
    from robot_sf.benchmark.artifact_publication import _publication_snqi_evidence

    payload = _excluded_bundle_payload(tmp_path)
    (payload / "release_metadata/snqi").mkdir(parents=True)
    assert _publication_snqi_evidence(payload) == {
        "checked": False,
        "reason": "legacy_snqi_excluded",
        "violations": ["Legacy SNQI is excluded but its bundle assets are present"],
    }
