"""Tests for camera-ready benchmark campaign orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.artifact_publication import PublicationBundleResult
from robot_sf.benchmark.camera_ready_campaign import (
    DEFAULT_SEED_SETS_PATH,
    _load_campaign_scenarios,
    load_campaign_config,
    run_campaign,
)
from robot_sf.common.artifact_paths import get_repository_root


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


def test_run_campaign_writes_core_artifacts(tmp_path: Path, monkeypatch):  # noqa: PLR0915
    """Campaign runner should emit summary artifacts and publication metadata."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign_runner",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "paper_interpretation_profile: baseline-ready-core",
                "preview_scenario_limit: 0",
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
    assert cfg.scenario_matrix_path == scenario_abs

    def _fake_run_batch(
        scenarios_or_path,
        out_path,
        schema_path,
        *,
        algo,
        benchmark_profile,
        **kwargs,
    ):
        scenarios = list(scenarios_or_path) if isinstance(scenarios_or_path, list) else []
        _ = schema_path
        _ = kwargs
        if scenarios:
            map_file = scenarios[0].get("map_file")
            if isinstance(map_file, str):
                assert not Path(map_file).is_absolute()
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "episode_id": f"e-{algo}-0",
                "scenario_id": "mock",
                "seed": 111,
                "scenario_params": {"algo": algo, "metadata": {"archetype": "crossing"}},
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
            "preflight": {
                "status": "fallback" if algo == "ppo" else "ok",
                "learned_policy_contract": (
                    {
                        "status": "warn",
                        "critical_mismatches": ["obs_mode=image mismatch"],
                        "warnings": [],
                    }
                    if algo == "ppo"
                    else {"status": "not_applicable"}
                ),
            },
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
    assert (campaign_root / "reports" / "campaign_table_core.csv").exists()
    assert (campaign_root / "reports" / "campaign_table_core.md").exists()
    assert (campaign_root / "reports" / "campaign_table_experimental.csv").exists()
    assert (campaign_root / "reports" / "campaign_table_experimental.md").exists()
    assert (campaign_root / "reports" / "scenario_breakdown.csv").exists()
    assert (campaign_root / "reports" / "scenario_breakdown.md").exists()
    assert (campaign_root / "reports" / "scenario_family_breakdown.csv").exists()
    assert (campaign_root / "reports" / "scenario_family_breakdown.md").exists()
    assert (campaign_root / "reports" / "campaign_report.md").exists()
    assert (campaign_root / "preflight" / "validate_config.json").exists()
    assert (campaign_root / "preflight" / "preview_scenarios.json").exists()
    preview_payload = json.loads(
        (campaign_root / "preflight" / "preview_scenarios.json").read_text(encoding="utf-8")
    )
    assert preview_payload["truncated"] is True
    assert preview_payload["total_scenarios"] == 1
    assert preview_payload["preview_limit"] == 0
    assert preview_payload["scenarios"] == []
    report_text = (campaign_root / "reports" / "campaign_report.md").read_text(encoding="utf-8")
    assert "Readiness & Degraded/Fallback Status" in report_text
    assert "SocNav Strict-vs-Fallback Disclosure" in report_text
    assert "fallback" in report_text
    assert "learned contract" in report_text
    table_md = (campaign_root / "reports" / "campaign_table.md").read_text(encoding="utf-8")
    assert "readiness_status" in table_md
    assert "learned_policy_contract_status" in table_md
    assert "socnav_prereq_policy" in table_md
    run_meta = json.loads((campaign_root / "run_meta.json").read_text(encoding="utf-8"))
    assert "seed_policy" in run_meta
    assert "resolved_seeds" in run_meta["seed_policy"]
    assert run_meta["preflight_artifacts"]["validate_config"].endswith(
        "preflight/validate_config.json"
    )
    summary_payload = json.loads(
        (campaign_root / "reports" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["campaign"]["paper_interpretation_profile"] == "baseline-ready-core"
    assert result["publication_bundle"] is not None


def test_load_campaign_config_uses_repo_default_seed_sets_path(tmp_path: Path):
    """Seed-set mode without explicit seed_sets_path should use repository default path."""
    config_dir = tmp_path / "cfg" / "nested"
    config_dir.mkdir(parents=True)

    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (config_dir / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = config_dir / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_seed_set_default_path",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: seed-set",
                "  seed_set: canonical",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    assert (
        cfg.seed_policy.seed_sets_path == (get_repository_root() / DEFAULT_SEED_SETS_PATH).resolve()
    )


def test_load_campaign_scenarios_converts_absolute_repo_map_path_to_relative(tmp_path: Path):
    """Scenario map paths under repository root should normalize to repo-relative form."""
    repo_root = get_repository_root().resolve()
    abs_map = (repo_root / "maps" / "svg_maps" / "classic_crossing.svg").resolve()
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        f"- name: smoke\n  map_file: {abs_map.as_posix()}\n  seeds: [1]\n",
        encoding="utf-8",
    )
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        "\n".join(
            [
                "name: map_path_norm",
                f"scenario_matrix: {matrix_path.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(campaign_path)
    scenarios = _load_campaign_scenarios(cfg)
    assert scenarios
    map_file = scenarios[0].get("map_file")
    assert isinstance(map_file, str)
    assert map_file == "maps/svg_maps/classic_crossing.svg"
