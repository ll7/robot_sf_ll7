"""Tests for camera-ready benchmark campaign orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.artifact_publication import PublicationBundleResult
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config, run_campaign


def test_load_campaign_config_resolves_relative_paths(tmp_path: Path):
    """Config loader should resolve scenario and algo-config paths relative to config file."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True)

    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    algo_cfg_rel = Path("configs/algos/social_force_example.yaml")
    scenario_abs = (config_dir / scenario_rel).resolve()
    algo_cfg_abs = (config_dir / algo_cfg_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    algo_cfg_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    algo_cfg_abs.write_text("v_max: 1.0\n", encoding="utf-8")

    config_path = config_dir / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [101]",
                "planners:",
                "  - key: sf",
                "    algo: social_force",
                f"    algo_config: {algo_cfg_rel.as_posix()}",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)

    assert cfg.name == "test_campaign"
    assert cfg.scenario_matrix_path == scenario_abs
    assert cfg.scenario_matrix_path.exists()
    assert cfg.planners[0].algo_config_path is not None
    assert cfg.planners[0].algo_config_path == algo_cfg_abs
    assert cfg.planners[0].algo_config_path.exists()
    assert list(cfg.seed_policy.seeds) == [101]


def test_run_campaign_writes_core_artifacts(tmp_path: Path, monkeypatch):
    """Campaign runner should emit summary artifacts and publication metadata."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign_runner",
                "scenario_matrix: configs/scenarios/single/francis2023_blind_corner.yaml",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    benchmark_profile: baseline-safe",
                "  - key: ppo",
                "    algo: ppo",
                "    benchmark_profile: experimental",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)

    def _fake_run_batch(
        scenarios_or_path,
        out_path,
        schema_path,
        *,
        algo,
        benchmark_profile,
        **kwargs,
    ):
        _ = scenarios_or_path
        _ = schema_path
        _ = kwargs
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "episode_id": f"e-{algo}-0",
                "scenario_id": "mock",
                "seed": 111,
                "scenario_params": {"algo": algo},
                "metrics": {"success": 1.0, "collisions": 0.0, "near_misses": 0.0},
                "algorithm_metadata": {"algorithm": algo, "status": "ok"},
            },
        ]
        with out_file.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return {
            "total_jobs": len(records),
            "written": len(records),
            "failed_jobs": 0,
            "failures": [],
            "out_path": str(out_file),
            "algorithm_readiness": {
                "name": algo,
                "tier": "experimental" if benchmark_profile == "experimental" else "baseline-ready",
                "profile": benchmark_profile,
            },
            "preflight": {"status": "ok"},
        }

    def _fake_compute_aggregates_with_ci(
        records,
        *,
        group_by,
        bootstrap_samples,
        bootstrap_confidence,
        bootstrap_seed,
    ):
        _ = records
        _ = group_by
        _ = bootstrap_samples
        _ = bootstrap_confidence
        _ = bootstrap_seed
        return {
            "mock_group": {
                "success": {"mean": 1.0, "mean_ci": [1.0, 1.0]},
                "collisions": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
                "near_misses": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
                "time_to_goal_norm": {"mean": 0.5, "mean_ci": [0.4, 0.6]},
                "path_efficiency": {"mean": 0.9, "mean_ci": [0.8, 0.95]},
                "comfort_exposure": {"mean": 0.2, "mean_ci": [0.1, 0.3]},
                "jerk_mean": {"mean": 0.1, "mean_ci": [0.08, 0.12]},
                "snqi": {"mean": 0.7, "mean_ci": [0.65, 0.75]},
            },
            "_meta": {"warnings": [], "missing_algorithms": []},
        }

    def _fake_export_publication_bundle(
        run_dir,
        out_dir,
        *,
        bundle_name,
        include_videos,
        repository_url,
        release_tag,
        doi,
        overwrite,
    ):
        _ = run_dir
        _ = out_dir
        _ = bundle_name
        _ = include_videos
        _ = repository_url
        _ = release_tag
        _ = doi
        _ = overwrite
        bundle_dir = tmp_path / "publication" / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        archive_path = tmp_path / "publication" / "bundle.tar.gz"
        archive_path.write_text("archive", encoding="utf-8")
        manifest_path = bundle_dir / "publication_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        checksums_path = bundle_dir / "checksums.sha256"
        checksums_path.write_text("", encoding="utf-8")
        return PublicationBundleResult(
            bundle_dir=bundle_dir,
            archive_path=archive_path,
            manifest_path=manifest_path,
            checksums_path=checksums_path,
            file_count=3,
            total_bytes=7,
        )

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.compute_aggregates_with_ci",
        _fake_compute_aggregates_with_ci,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.export_publication_bundle",
        _fake_export_publication_bundle,
    )

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="test")

    campaign_root = Path(result["campaign_root"])
    assert campaign_root.exists()
    assert (campaign_root / "campaign_manifest.json").exists()
    assert (campaign_root / "reports" / "campaign_summary.json").exists()
    assert (campaign_root / "reports" / "campaign_table.csv").exists()
    assert (campaign_root / "reports" / "campaign_table.md").exists()
    assert (campaign_root / "reports" / "campaign_report.md").exists()
    assert result["publication_bundle"] is not None
