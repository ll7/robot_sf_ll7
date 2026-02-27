"""Tests for camera-ready benchmark campaign orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.artifact_publication import PublicationBundleResult
from robot_sf.benchmark.camera_ready_campaign import (
    DEFAULT_SEED_SETS_PATH,
    CampaignConfig,
    PlannerSpec,
    SeedPolicy,
    _jsonable_repo_relative,
    _load_campaign_scenarios,
    _planner_report_row,
    _sanitize_csv_cell,
    _sanitize_git_remote,
    _sanitize_name,
    _write_campaign_report,
    load_campaign_config,
    prepare_campaign_preflight,
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
    assert (campaign_root / "reports" / "matrix_summary.csv").exists()
    assert (campaign_root / "reports" / "matrix_summary.json").exists()
    assert (campaign_root / "reports" / "amv_coverage_summary.json").exists()
    assert (campaign_root / "reports" / "amv_coverage_summary.md").exists()
    assert (campaign_root / "reports" / "comparability_matrix.json").exists()
    assert (campaign_root / "reports" / "comparability_matrix.md").exists()
    assert (campaign_root / "reports" / "scenario_breakdown.csv").exists()
    assert (campaign_root / "reports" / "scenario_breakdown.md").exists()
    assert (campaign_root / "reports" / "scenario_family_breakdown.csv").exists()
    assert (campaign_root / "reports" / "scenario_family_breakdown.md").exists()
    assert (campaign_root / "reports" / "kinematics_parity_table.csv").exists()
    assert (campaign_root / "reports" / "kinematics_parity_table.md").exists()
    assert (campaign_root / "reports" / "kinematics_skipped_combinations.csv").exists()
    assert (campaign_root / "reports" / "kinematics_skipped_combinations.md").exists()
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
    assert "planner_group" in table_md
    assert "kinematics" in table_md
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
    assert summary_payload["artifacts"]["matrix_summary_json"].endswith(
        "reports/matrix_summary.json"
    )
    assert summary_payload["artifacts"]["matrix_summary_csv"].endswith("reports/matrix_summary.csv")
    assert summary_payload["artifacts"]["amv_coverage_json"].endswith(
        "reports/amv_coverage_summary.json"
    )
    assert summary_payload["artifacts"]["comparability_json"].endswith(
        "reports/comparability_matrix.json"
    )
    assert "release_url_template" in summary_payload["campaign"]
    assert "release_asset_url_template" in summary_payload["campaign"]
    assert "doi_url_template" in summary_payload["campaign"]
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
    assert cfg.kinematics_matrix == ("differential_drive",)


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


def test_run_campaign_stops_on_partial_failure_when_configured(tmp_path: Path, monkeypatch) -> None:
    """Campaign should stop after first partial-failure when stop_on_failure is enabled."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "campaign_stop_on_partial.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_campaign_stop_on_partial",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "stop_on_failure: true",
                "planners:",
                "  - key: prediction_planner",
                "    algo: prediction_planner",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    call_order: list[str] = []

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
        _ = out_path
        _ = schema_path
        _ = benchmark_profile
        _ = kwargs
        call_order.append(algo)
        if algo == "prediction_planner":
            return {
                "total_jobs": 1,
                "written": 0,
                "failed_jobs": 1,
                "failures": [{"scenario_id": "mock", "seed": 111, "error": "mock"}],
                "preflight": {
                    "status": "ok",
                    "learned_policy_contract": {"status": "not_applicable"},
                },
            }
        raise AssertionError("run_batch must not be called for planners after partial-failure")

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="stop_partial")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    planner_rows = summary_payload["planner_rows"]

    assert call_order == ["prediction_planner"]
    assert len(planner_rows) == 1
    assert planner_rows[0]["planner_key"] == "prediction_planner"
    assert planner_rows[0]["status"] == "partial-failure"


def test_write_campaign_report_escapes_markdown_cells(tmp_path: Path) -> None:
    """Markdown report tables should escape raw cell separators from planner metadata."""
    report_path = tmp_path / "campaign_report.md"
    payload = {
        "campaign": {"campaign_id": "c1"},
        "warnings": [],
        "planner_rows": [
            {
                "planner_key": "planner|unsafe",
                "algo": "goal",
                "kinematics": "holonomic|vx_vy",
                "status": "ok",
                "started_at_utc": "now",
                "runtime_sec": 1.0,
                "episodes": 1,
                "episodes_per_second": 1.0,
                "success_mean": "1.0",
                "collisions_mean": "0.0",
                "snqi_mean": "0.5",
                "projection_rate": "0.0",
                "infeasible_rate": "0.0",
                "execution_mode": "native",
                "readiness_status": "ok",
                "readiness_tier": "baseline-ready",
                "preflight_status": "ok",
                "learned_policy_contract_status": "not_applicable",
                "socnav_prereq_policy": "fail-fast",
            }
        ],
    }
    _write_campaign_report(report_path, payload)
    report_text = report_path.read_text(encoding="utf-8")
    assert "planner\\|unsafe" in report_text
    assert "holonomic\\|vx_vy" in report_text


def test_planner_report_row_uses_nested_planner_kinematics_execution_mode() -> None:
    """Row builder should read execution_mode from nested planner_kinematics payload."""
    planner = PlannerSpec(key="prediction_planner", algo="prediction_planner")
    summary = {
        "status": "ok",
        "written": 1,
        "runtime_sec": 1.0,
        "episodes_per_second": 1.0,
        "algorithm_readiness": {"tier": "baseline-ready"},
        "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
        "algorithm_metadata_contract": {
            "planner_kinematics": {"execution_mode": "adapter"},
            "kinematics_feasibility": {
                "commands_evaluated": 4,
                "projection_rate": 0.25,
                "infeasible_rate": 0.25,
            },
        },
    }
    row = _planner_report_row(
        planner,
        summary,
        aggregates=None,
        kinematics="differential_drive",
    )
    assert row["execution_mode"] == "adapter"
    assert row["readiness_status"] == "adapter"


def test_jsonable_repo_relative_normalizes_paths_for_stable_hashing(tmp_path: Path) -> None:
    """Hash-prep helper should normalize Path values to stable repo-relative strings."""
    repo_root = get_repository_root().resolve()
    payload = {
        "scenario_matrix_path": repo_root / "configs/scenarios/classic_interactions.yaml",
        "other_path": tmp_path / "external.yaml",
    }
    normalized = _jsonable_repo_relative(payload)
    assert normalized["scenario_matrix_path"] == "configs/scenarios/classic_interactions.yaml"
    assert str(normalized["other_path"]).endswith("/external.yaml")


def test_sanitize_git_remote_strips_credentials() -> None:
    """Git remote helper should remove embedded credentials from URL-form remotes."""
    remote = "https://user:token@example.com/org/repo.git"
    assert _sanitize_git_remote(remote) == "https://example.com/org/repo.git"
    assert _sanitize_git_remote("git@github.com:ll7/robot_sf_ll7.git") == (
        "git@github.com:ll7/robot_sf_ll7.git"
    )


def test_sanitize_csv_cell_prefixes_formula_like_values() -> None:
    """CSV sanitizer should neutralize spreadsheet formula prefixes."""
    assert _sanitize_csv_cell("=1+1") == "'=1+1"
    assert _sanitize_csv_cell("@SUM(A1:A2)") == "'@SUM(A1:A2)"
    assert _sanitize_csv_cell("safe") == "safe"
    assert _sanitize_csv_cell(42) == 42


def test_prepare_campaign_preflight_validates_campaign_config(tmp_path: Path) -> None:
    """Programmatic preflight entrypoint should enforce campaign invariants."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    cfg = CampaignConfig(
        name="invalid_preflight_cfg",
        scenario_matrix_path=scenario_abs,
        planners=(PlannerSpec(key="goal", algo="goal", planner_group_explicit=True),),
        seed_policy=SeedPolicy(mode="fixed-list", seeds=(111,)),
        paper_facing=True,
        paper_profile_version=None,
    )
    with pytest.raises(ValueError, match="paper_profile_version"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="invalid")


def test_run_campaign_sanitizes_run_directory_keys(tmp_path: Path, monkeypatch) -> None:
    """Planner run directories should use sanitized planner/kinematics identifiers."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_sanitize.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: sanitize_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                'kinematics_matrix: ["holonomic/../unsafe"]',
                "planners:",
                '  - key: "../../planner|unsafe"',
                "    algo: goal",
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
        del scenarios_or_path, schema_path, benchmark_profile, kwargs
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(
                {
                    "episode_id": f"e-{algo}-0",
                    "scenario_id": "mock",
                    "seed": 111,
                    "scenario_params": {"algo": algo, "metadata": {"archetype": "crossing"}},
                    "metrics": {"success": 1.0, "collisions": 0.0, "near_misses": 0.0},
                    "algorithm_metadata": {"algorithm": algo, "status": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "out_path": str(out_file),
            "algorithm_readiness": {
                "name": algo,
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
            "preflight": {"status": "ok", "learned_policy_contract": {"status": "not_applicable"}},
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="sanitize")
    campaign_root = Path(result["campaign_root"])
    runs_dir = campaign_root / "runs"
    run_dirs = [path.name for path in runs_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    expected = f"{_sanitize_name('../../planner|unsafe')}__{_sanitize_name('holonomic/../unsafe')}"
    assert run_dirs[0] == expected


def test_run_campaign_marks_skipped_preflight_as_skipped(tmp_path: Path, monkeypatch) -> None:
    """Skipped planner/kinematics combinations should not be marked as successful."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_skipped.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: skipped_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(*args, **kwargs):
        del args, kwargs
        return {
            "total_jobs": 0,
            "written": 0,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {"status": "skipped", "compatibility_reason": "unsupported"},
            "algorithm_readiness": {
                "name": "goal",
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="skipped")
    summary_payload = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    assert summary_payload["planner_rows"][0]["status"] == "skipped"
    assert summary_payload["campaign"]["successful_runs"] == 0


def test_run_campaign_parity_table_includes_ci_columns(tmp_path: Path, monkeypatch) -> None:
    """Parity artifacts should preserve available CI values."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_ci.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: ci_campaign",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)

    def _fake_run_batch(*args, **kwargs):
        del args
        out_path = Path(kwargs["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "episode_id": "e-goal-0",
                    "scenario_id": "mock",
                    "seed": 111,
                    "scenario_params": {"algo": "goal", "metadata": {"archetype": "crossing"}},
                    "metrics": {"success": 1.0, "collisions": 0.0, "near_misses": 0.0},
                    "algorithm_metadata": {"algorithm": "goal", "status": "ok"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "total_jobs": 1,
            "written": 1,
            "failed_jobs": 0,
            "failures": [],
            "preflight": {"status": "ok"},
            "algorithm_readiness": {
                "name": "goal",
                "tier": "baseline-ready",
                "profile": "baseline-safe",
            },
        }

    def _fake_compute_aggregates_with_ci(*args, **kwargs):
        del args, kwargs
        return {
            "mock_group": {
                "success": {"mean": 1.0, "mean_ci": [0.8, 1.0]},
                "collisions": {"mean": 0.0, "mean_ci": [0.0, 0.2]},
                "snqi": {"mean": 0.7, "mean_ci": [0.6, 0.8]},
            },
            "_meta": {"warnings": [], "missing_algorithms": []},
        }

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", _fake_run_batch)
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.compute_aggregates_with_ci",
        _fake_compute_aggregates_with_ci,
    )

    result = run_campaign(cfg, output_root=tmp_path / "campaign_out", label="ci")
    parity_csv = (
        Path(result["campaign_root"]) / "reports" / "kinematics_parity_table.csv"
    ).read_text(encoding="utf-8")
    assert "success_ci_low" in parity_csv
    assert "success_ci_high" in parity_csv
    assert "collision_ci_low" in parity_csv
    assert "collision_ci_high" in parity_csv
    assert "snqi_ci_low" in parity_csv
    assert "snqi_ci_high" in parity_csv


def test_load_campaign_config_accepts_planner_group_and_paper_profile(tmp_path: Path) -> None:
    """Paper-facing config should parse planner groups and profile fields."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "paper_campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: paper_cfg",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    assert cfg.paper_facing is True
    assert cfg.paper_profile_version == "paper-matrix-v1"
    assert cfg.planners[0].planner_group == "core"


def test_load_campaign_config_rejects_invalid_planner_group(tmp_path: Path) -> None:
    """Planner group must be either core or experimental."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "invalid_group.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: invalid_group",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: invalid",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="planner_group"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_missing_paper_version(tmp_path: Path) -> None:
    """Paper-facing config requires explicit profile version."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "missing_version.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: missing_version",
                "paper_facing: true",
                "kinematics_matrix: [differential_drive]",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="paper_profile_version"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_non_differential_paper_kinematics(tmp_path: Path) -> None:
    """Paper-facing profile v1 should lock differential-drive-only matrix."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "bad_kinematics.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: bad_kinematics",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive, bicycle_drive]",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kinematics_matrix"):
        load_campaign_config(config_path)


def test_load_campaign_config_rejects_implicit_planner_group_for_paper(tmp_path: Path) -> None:
    """Paper-facing configs should require explicit planner_group fields."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "implicit_group.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: implicit_group",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit planner_group"):
        load_campaign_config(config_path)


def test_prepare_campaign_preflight_writes_matrix_summary(tmp_path: Path) -> None:
    """Preflight preparation should emit validate/preview and matrix-summary artifacts."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "paper_campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: paper_cfg",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="preflight")
    assert Path(prepared["validate_config_path"]).exists()
    assert Path(prepared["preview_scenarios_path"]).exists()
    assert Path(prepared["matrix_summary_json_path"]).exists()
    assert Path(prepared["matrix_summary_csv_path"]).exists()
    matrix_payload = json.loads(
        Path(prepared["matrix_summary_json_path"]).read_text(encoding="utf-8")
    )
    assert matrix_payload["rows"]
    first = matrix_payload["rows"][0]
    assert first["planner_group"] == "core"
    assert first["kinematics"] == "differential_drive"


def test_prepare_campaign_preflight_matrix_summary_is_deterministic(tmp_path: Path) -> None:
    """Matrix summary row ordering should be deterministic by group/key/kinematics."""
    scenario_rel = Path("configs/scenarios/single/francis2023_blind_corner.yaml")
    scenario_abs = (tmp_path / scenario_rel).resolve()
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "paper_order.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: paper_order",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [111]",
                f"scenario_matrix: {scenario_rel.as_posix()}",
                "planners:",
                "  - key: z_exp",
                "    algo: goal",
                "    planner_group: experimental",
                "  - key: a_core",
                "    algo: goal",
                "    planner_group: core",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="order")
    matrix_payload = json.loads(
        Path(prepared["matrix_summary_json_path"]).read_text(encoding="utf-8")
    )
    planner_keys = [row["planner_key"] for row in matrix_payload["rows"]]
    assert planner_keys == ["a_core", "z_exp"]


def test_prepare_campaign_preflight_writes_amv_and_comparability_artifacts(tmp_path: Path) -> None:
    """Preflight should emit AMV coverage and comparability artifacts."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "- name: amv_ok",
                "  map_file: maps/svg_maps/classic_crossing.svg",
                "  seeds: [111]",
                "  metadata:",
                "    archetype: classic_crossing",
                "  amv:",
                "    use_case: delivery_robot",
                "    context: sidewalk",
                "    speed_regime: walking_speed",
                "    maneuver_type: crossing",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: amv_contract",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "comparability_mapping: configs/benchmarks/alyassi_comparability_map_v1.yaml",
                "amv_profile:",
                "  coverage_enforcement: warn",
                "  required_dimensions:",
                "    use_case: [delivery_robot]",
                "    context: [sidewalk]",
                "    speed_regime: [walking_speed]",
                "    maneuver_type: [crossing]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)
    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="amv")
    assert Path(prepared["amv_coverage_json_path"]).exists()
    assert Path(prepared["amv_coverage_md_path"]).exists()
    assert Path(prepared["comparability_json_path"]).exists()
    assert Path(prepared["comparability_md_path"]).exists()

    manifest = json.loads((Path(prepared["campaign_root"]) / "campaign_manifest.json").read_text())
    assert manifest["amv_coverage_status"] == "pass"
    assert manifest["comparability_mapping_version"] == "alyassi-comparability-v1"
    assert manifest["artifacts"]["amv_coverage_json"].endswith("reports/amv_coverage_summary.json")
    assert manifest["artifacts"]["comparability_json"].endswith("reports/comparability_matrix.json")


def test_prepare_campaign_preflight_enforces_amv_coverage_error_mode(tmp_path: Path) -> None:
    """Missing AMV dimensions should fail preflight when enforcement is error."""
    scenario_path = tmp_path / "scenarios_missing_amv.yaml"
    scenario_path.write_text(
        "- name: amv_missing\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_error.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: amv_error_contract",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "amv_profile:",
                "  coverage_enforcement: error",
                "  required_dimensions:",
                "    use_case: [delivery_robot]",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    with pytest.raises(ValueError, match="AMV coverage contract validation failed"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="amv_error")


def test_load_campaign_config_rejects_invalid_amv_coverage_enforcement(tmp_path: Path) -> None:
    """AMV profile should reject unsupported enforcement values."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign_bad_amv.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: bad_amv_cfg",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "amv_profile:",
                "  coverage_enforcement: maybe",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coverage_enforcement"):
        load_campaign_config(config_path)


def test_prepare_campaign_preflight_rejects_invalid_comparability_mapping_for_paper(
    tmp_path: Path,
) -> None:
    """Paper-facing preflight should fail fast on invalid comparability mapping schema."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    bad_mapping = tmp_path / "bad_mapping.yaml"
    bad_mapping.write_text("mapping_version: x\n", encoding="utf-8")
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: invalid_mapping",
                "paper_facing: true",
                "paper_profile_version: paper-matrix-v1",
                "kinematics_matrix: [differential_drive]",
                f"scenario_matrix: {scenario_path.as_posix()}",
                f"comparability_mapping: {bad_mapping.as_posix()}",
                "planners:",
                "  - key: goal",
                "    algo: goal",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config_path)
    with pytest.raises(ValueError, match="scenario_family_mapping"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="bad_map")
