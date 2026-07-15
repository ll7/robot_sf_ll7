"""Contract tests for the issue #5579 MPC tuning-budget sensitivity packet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.mpc_tuning_sensitivity import (
    TARGET_ARM_KEYS,
    analyze_results,
    build_candidate_plan,
    load_sensitivity_config,
    normalize_episode_record,
    selected_scenarios,
    write_report,
)
from scripts.benchmark.run_mpc_tuning_sensitivity_issue_5579 import _display_path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/analysis/issue_5579_mpc_tuning_sensitivity.yaml"


def test_packet_freezes_two_target_arms_three_parameters_and_twenty_points() -> None:
    """The packet has a paired three-scenario scope and stays within the N<=20 bound."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    assert tuple(arm["key"] for arm in config["target_arms"]) == TARGET_ARM_KEYS
    assert config["search"]["candidate_count"] == 20
    assert config["search"]["top_parameters"] == [
        "max_linear_speed",
        "horizon_steps",
        "pedestrian_safety_margin",
    ]
    assert len(selected_scenarios(config, repo_root=ROOT)) == 3
    assert config["scenario_scope"]["seeds"] == [111, 112, 113]


def test_candidate_plan_preserves_arm_specific_base_and_only_varies_declared_axes() -> None:
    """Each target candidate applies only the three declared overrides to its own base config."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    plan = build_candidate_plan(config, repo_root=ROOT)
    targets = [entry for entry in plan if entry["target"]]
    incumbents = [entry for entry in plan if not entry["target"]]
    assert len(targets) == 40
    assert len(incumbents) == 4
    assert {entry["candidate_id"] for entry in targets} == {
        point["id"] for point in config["search"]["candidate_points"]
    }
    cbf_candidate = next(
        entry
        for entry in targets
        if entry["arm_key"] == "prediction_mpc_cbf" and entry["candidate_id"] == "speed_high"
    )
    assert cbf_candidate["effective_config"]["cbf_safety_filter"] == {"enabled": True}
    assert all(
        set(entry["overrides"]) <= set(config["search"]["top_parameters"]) for entry in targets
    )


def test_normalization_requires_explicit_outcome_and_availability_provenance() -> None:
    """Missing typed outcome or availability fields fail closed before aggregation."""
    record = _raw_record(route_complete=True, collision_event=False)
    row = normalize_episode_record(record, arm_key="prediction_mpc", candidate_id="incumbent")
    assert row["success"] is True
    assert row["execution_mode"] == "adapter"
    with pytest.raises(ValueError, match="missing explicit fields"):
        normalize_episode_record(
            {**record, "outcome": {"route_complete": True}},
            arm_key="prediction_mpc",
            candidate_id="incumbent",
        )
    with pytest.raises(ValueError, match="missing sensitivity_availability"):
        normalize_episode_record(
            {key: value for key, value in record.items() if key != "sensitivity_availability"},
            arm_key="prediction_mpc",
            candidate_id="incumbent",
        )


def test_report_applies_preregistered_read_to_best_found_configs(tmp_path: Path) -> None:
    """A fully eligible synthetic result produces the structural-vs-budget read only on the fixed slice."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    plan = build_candidate_plan(config, repo_root=ROOT)
    rows = []
    for entry in plan:
        target_success = bool(entry["target"] is False)
        for scenario_id in config["scenario_scope"]["scenario_ids"]:
            for seed in config["scenario_scope"]["seeds"]:
                rows.append(
                    {
                        "arm_key": entry["arm_key"],
                        "candidate_id": entry["candidate_id"],
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "success": target_success,
                        "execution_mode": "adapter",
                        "readiness_status": "adapter",
                        "availability_status": "available",
                        "benchmark_success": True,
                        "planner_runtime_status": "eligible",
                    }
                )
    report = analyze_results(
        config,
        rows,
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit="fixture",
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
    )
    assert report["status"] == "complete_diagnostic"
    assert report["read"]["decision"] == "structural_reading_strengthens_on_tested_slice"
    assert all(summary["best_candidate"] is not None for summary in report["target_summary"])
    paths = write_report(report, tmp_path)
    assert json.loads(Path(paths["json"]).read_text(encoding="utf-8"))["issue"] == 5579
    assert "Claim boundary" in Path(paths["markdown"]).read_text(encoding="utf-8")


def test_fallback_row_blocks_read_and_is_not_counted(tmp_path: Path) -> None:
    """Fallback/degraded provenance remains visible but cannot enter the success comparison."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    plan = build_candidate_plan(config, repo_root=ROOT)
    rows = _fixture_rows(config, plan)
    rows[0]["availability_status"] = "not_available"
    rows[0]["readiness_status"] = "fallback"
    report = analyze_results(
        config,
        rows,
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit="fixture",
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
    )
    assert report["status"] == "blocked"
    assert report["read"]["decision"] == "blocked"
    assert report["excluded_episode_rows"] == 1
    assert report["candidate_rows"][0]["status"] == "excluded"
    write_report(report, tmp_path)


def test_solver_fallback_runtime_blocks_read_and_is_not_counted() -> None:
    """Planner solver/fallback diagnostics are a fail-closed exclusion axis."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    rows = _fixture_rows(config, build_candidate_plan(config, repo_root=ROOT))
    rows[0]["planner_runtime_status"] = "fallback"
    report = analyze_results(
        config,
        rows,
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit="fixture",
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
    )
    assert report["status"] == "blocked"
    assert report["read"]["decision"] == "blocked"
    assert report["excluded_episode_rows"] == 1
    assert report["candidate_rows"][0]["exclusion_reasons"] == ["fallback"]


def test_missing_planner_runtime_blocks_read() -> None:
    """Missing per-episode planner runtime provenance cannot become evidence."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    rows = _fixture_rows(config, build_candidate_plan(config, repo_root=ROOT))
    rows[0].pop("planner_runtime_status")
    report = analyze_results(
        config,
        rows,
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit="fixture",
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
    )
    assert report["status"] == "blocked"
    assert report["candidate_rows"][0]["exclusion_reasons"] == ["missing"]


def test_external_output_path_has_stable_display() -> None:
    """Runner artifact paths remain valid when --out-dir is outside the repository."""
    external = Path("/tmp") / "issue-5579-output"
    assert _display_path(external) == str(external)
    assert _display_path(ROOT / "output") == "output"


def test_report_rejects_missing_paired_rows() -> None:
    """The paired fixed-scope denominator cannot silently shrink."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    rows = _fixture_rows(config, build_candidate_plan(config, repo_root=ROOT))
    with pytest.raises(ValueError, match="missing"):
        analyze_results(
            config,
            rows[:-1],
            repo_root=ROOT,
            config_path=str(CONFIG),
            run_commit="fixture",
            reproduction_command="fixture",
            raw_artifact_root="output/fixture",
        )


def _fixture_rows(config: dict, plan: list[dict]) -> list[dict]:
    """Build complete eligible rows for report-contract tests."""
    rows = []
    for entry in plan:
        for scenario_id in config["scenario_scope"]["scenario_ids"]:
            for seed in config["scenario_scope"]["seeds"]:
                rows.append(
                    {
                        "arm_key": entry["arm_key"],
                        "candidate_id": entry["candidate_id"],
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "success": False,
                        "execution_mode": "adapter",
                        "readiness_status": "adapter",
                        "availability_status": "available",
                        "benchmark_success": True,
                        "planner_runtime_status": "eligible",
                    }
                )
    return rows


def _raw_record(*, route_complete: bool, collision_event: bool) -> dict[str, object]:
    """Build the smallest raw runner row accepted by the normalizer."""
    return {
        "scenario_id": "classic_bottleneck_medium",
        "seed": 111,
        "status": "success" if route_complete else "failure",
        "outcome": {
            "route_complete": route_complete,
            "collision_event": collision_event,
        },
        "sensitivity_availability": {
            "execution_mode": "adapter",
            "readiness_status": "adapter",
            "availability_status": "available",
            "benchmark_success": True,
        },
        "algorithm_metadata": {
            "planner_runtime": {
                "solver_failures": 0,
                "fallback_stop_count": 0,
            }
        },
    }
