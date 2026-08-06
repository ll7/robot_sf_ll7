"""Contract tests for the issue #5579 MPC tuning-budget sensitivity packet."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import scripts.benchmark.run_mpc_tuning_sensitivity_issue_5579 as sensitivity_runner
from robot_sf.benchmark.mpc_tuning_sensitivity import (
    TARGET_ARM_KEYS,
    analyze_results,
    build_candidate_plan,
    compute_scenario_list_hash,
    config_hash,
    load_sensitivity_config,
    normalize_episode_record,
    selected_scenarios,
    validate_canary_rows,
    validate_sensitivity_config,
    write_report,
)
from scripts.benchmark.run_mpc_tuning_sensitivity_issue_5579 import _display_path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/analysis/issue_5579_mpc_tuning_sensitivity_v2.yaml"


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
    assert config["scenario_scope"]["seeds"] == [101, 102, 103]

    # Two-phase split contract assertions
    tuning = config["tuning_scope"]
    assert len(tuning["scenario_ids"]) == 3
    assert tuning["seeds"] == [101, 102, 103]
    assert tuning["scenario_list_hash"] == compute_scenario_list_hash(tuning["scenario_ids"])

    held_out = config["held_out_scope"]
    assert held_out["seeds"] == list(range(111, 121))
    assert len(held_out["scenario_ids"]) == 45
    assert held_out["scenario_ids"] == sorted(held_out["scenario_ids"])
    assert held_out["scenario_list_hash"] == compute_scenario_list_hash(held_out["scenario_ids"])
    assert held_out["excluded_scenarios"] == tuning["scenario_ids"]
    assert len(selected_scenarios(config, repo_root=ROOT, scope_name="held_out_scope")) == 45

    canary = config["canary"]
    assert canary["seed"] == 101
    assert canary["required_eligible_episodes"] == 6
    assert canary["stop_on_ineligible"] is True

    inference = config["inference"]
    assert inference["resampling_unit"] == "paired_seed_block"
    assert inference["bootstrap"]["confidence_level"] == 0.95
    assert inference["multiplicity"]["method"] == "holm_bonferroni"
    assert inference["multiplicity"]["contrast_count"] == 8


def test_packet_identity_is_versioned_and_rejects_legacy_study_id() -> None:
    """The active two-phase packet cannot reuse the historical v1 study identity."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    assert config["schema_version"] == "issue_5579_mpc_tuning_sensitivity.v2"
    assert config["study_id"] == "issue_5579_mpc_tuning_budget_sensitivity_v2"

    legacy_identity = deepcopy(config)
    legacy_identity["study_id"] = "issue_5579_mpc_tuning_budget_sensitivity_v1"
    with pytest.raises(ValueError, match="study_id must be"):
        validate_sensitivity_config(legacy_identity, repo_root=ROOT)


def test_phase_contract_is_required_and_held_out_drift_fails_closed() -> None:
    """The validator cannot silently fall back to the old tuning-only packet."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    missing_inference = deepcopy(config)
    missing_inference.pop("inference")
    with pytest.raises(ValueError, match="inference section is required"):
        validate_sensitivity_config(missing_inference, repo_root=ROOT)

    drifted_split = deepcopy(config)
    drifted_split["held_out_scope"]["scenario_ids"] = drifted_split["held_out_scope"][
        "scenario_ids"
    ][1:]
    with pytest.raises(ValueError, match="must exactly match the source matrix"):
        validate_sensitivity_config(drifted_split, repo_root=ROOT)

    drifted_inference = deepcopy(config)
    drifted_inference["inference"]["multiplicity"]["contrast_count"] = 7
    with pytest.raises(ValueError, match="contrast_count must be 8"):
        validate_sensitivity_config(drifted_inference, repo_root=ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [("horizon", 99), ("dt", 0.2), ("workers", 2)],
)
def test_tuning_scope_execution_settings_cannot_drift(field: str, value: object) -> None:
    """The duplicated tuning scope cannot change the frozen execution settings."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    drifted = deepcopy(config)
    drifted["tuning_scope"][field] = value
    with pytest.raises(ValueError, match=rf"tuning_scope\.{field} must match"):
        validate_sensitivity_config(drifted, repo_root=ROOT)


def test_frozen_tuning_scenario_ids_cannot_be_replaced() -> None:
    """The packet validator rejects a different three-scenario tuning slice."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    drifted = deepcopy(config)
    drifted["scenario_scope"]["scenario_ids"][0] = "classic_bottleneck_high"
    with pytest.raises(ValueError, match="frozen 2026-08-03 tuning scenarios"):
        validate_sensitivity_config(drifted, repo_root=ROOT)


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


def test_normalization_extracts_native_solver_provenance_and_update_evidence() -> None:
    """Raw target metadata is converted into the strict canary predicates."""
    record = _raw_record(route_complete=True, collision_event=False)
    record["sensitivity_availability"] = {
        "execution_mode": "native",
        "readiness_status": "native",
        "availability_status": "available",
        "benchmark_success": True,
    }
    effective_config = {"predictor_backend": "constant_velocity", "fallback_to_stop": False}
    metadata = record["algorithm_metadata"]
    assert isinstance(metadata, dict)
    metadata["config"] = effective_config
    metadata["config_hash"] = config_hash(effective_config)
    metadata["planner_kinematics"] = {
        "execution_mode": "native",
        "adapter_name": "PredictionMPCPlannerAdapter",
        "supports_native_commands": True,
    }
    runtime = metadata["planner_runtime"]
    assert isinstance(runtime, dict)
    runtime.update(
        {
            "solver_successes": 1,
            "nonzero_command_count": 1,
            "mean_abs_linear": 0.2,
            "mean_abs_angular": 0.1,
        }
    )
    row = normalize_episode_record(
        record,
        arm_key="prediction_mpc",
        candidate_id="incumbent",
        expected_config_hash=config_hash(effective_config),
    )
    assert row["solver_execution_mode"] == "native"
    assert row["valid_solver_provenance"] is True
    assert row["finite_commands"] is True
    assert row["native_solver_eligible"] is True

    metadata["planner_kinematics"]["execution_mode"] = "adapter"
    adapter_row = normalize_episode_record(
        record,
        arm_key="prediction_mpc",
        candidate_id="incumbent",
        expected_config_hash=config_hash(effective_config),
    )
    assert adapter_row["native_solver_eligible"] is False
    assert "planner_execution_mode_not_native" in adapter_row["native_solver_exclusion_reasons"]


def test_canary_requires_exact_native_solver_evidence_and_key_coverage() -> None:
    """Adapter availability alone cannot authorize the six-row production gate."""
    scenarios = [
        "classic_bottleneck_medium",
        "classic_cross_trap_high",
        "francis2023_intersection_wait",
    ]
    rows = [
        _strict_canary_row(arm_key=arm_key, scenario_id=scenario_id)
        for arm_key in TARGET_ARM_KEYS
        for scenario_id in scenarios
    ]
    result = validate_canary_rows(
        rows,
        scenario_ids=scenarios,
        seed=101,
        required_eligible=6,
    )
    assert result["status"] == "ok"
    assert result["eligible_episodes"] == 6

    adapter_only = deepcopy(rows)
    adapter_only[0]["execution_mode"] = "adapter"
    failed = validate_canary_rows(
        adapter_only,
        scenario_ids=scenarios,
        seed=101,
        required_eligible=6,
    )
    assert failed["status"] == "failed"
    assert "execution_mode_not_native" in failed["invalid_rows"][0]["reasons"]

    for field, reason in (
        ("solver_failures", "solver_failures_invalid"),
        ("fallback_stop_count", "fallback_stop_count_invalid"),
    ):
        negative_counter = deepcopy(rows)
        negative_counter[0][field] = -1
        failed_counter = validate_canary_rows(
            negative_counter,
            scenario_ids=scenarios,
            seed=101,
            required_eligible=6,
        )
        assert failed_counter["status"] == "failed"
        assert reason in failed_counter["invalid_rows"][0]["reasons"]

    duplicate = validate_canary_rows(
        rows + [rows[0]],
        scenario_ids=scenarios,
        seed=101,
        required_eligible=6,
    )
    assert duplicate["status"] == "failed"
    assert duplicate["duplicate_keys"]


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
                        "solver_execution_mode": "native",
                        "valid_solver_provenance": True,
                        "finite_commands": True,
                        "solver_successes": 1,
                        "solver_failures": 0,
                        "fallback_stop_count": 0,
                        "control_updates": 1,
                        "native_solver_eligible": True,
                        "native_solver_exclusion_reasons": [],
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


def test_analysis_excludes_target_rows_with_invalid_solver_provenance() -> None:
    """Normal analysis cannot count target rows that fail native solver provenance."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    plan = build_candidate_plan(config, repo_root=ROOT)
    rows = _fixture_rows(config, plan)
    for row in rows:
        if row["arm_key"] in TARGET_ARM_KEYS:
            row.update(
                {
                    "solver_execution_mode": "adapter",
                    "valid_solver_provenance": False,
                    "finite_commands": False,
                    "solver_successes": 0,
                    "solver_failures": 1,
                    "fallback_stop_count": 1,
                    "control_updates": 0,
                    "native_solver_eligible": False,
                    "native_solver_exclusion_reasons": ["solver_provenance_invalid"],
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

    target_rows = [row for row in report["candidate_rows"] if row["target"]]
    assert report["status"] == "blocked"
    assert report["read"]["decision"] == "blocked"
    assert sum(row["eligible_episodes"] for row in target_rows) == 0
    assert sum(row["excluded_episodes"] for row in target_rows) == 360
    exclusion_reasons = {reason for row in target_rows for reason in row["exclusion_reasons"]}
    assert "solver_provenance_invalid" in exclusion_reasons
    assert "eligible" not in exclusion_reasons


def test_incumbent_adapter_rows_remain_eligible_without_mpc_solver_evidence() -> None:
    """Frozen hybrid incumbents run via their declared adapter and must stay eligible.

    The strict native-solver evidence gate targets the prediction-aware MPC arms only.
    Incumbent ``hybrid_rule_local_planner`` arms cannot carry ``PredictionMPCPlannerAdapter``
    solver metadata, so excluding them on solver evidence would strand every target versus
    incumbent comparison and force the held-out production read to block forever.
    """
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    plan = build_candidate_plan(config, repo_root=ROOT)
    rows = _fixture_rows(config, plan)
    # Strip the MPC solver evidence from every incumbent row so they look like real
    # hybrid-rule adapter rows; keep the strict native-solver evidence on target arms.
    incumbent_solver_profile = {
        "solver_execution_mode": "unknown",
        "valid_solver_provenance": False,
        "finite_commands": False,
        "solver_successes": None,
        "solver_failures": None,
        "fallback_stop_count": None,
        "control_updates": None,
        "native_solver_eligible": False,
        "native_solver_exclusion_reasons": ["unexpected_solver_planner"],
    }
    for row in rows:
        if row["arm_key"] not in TARGET_ARM_KEYS:
            row["execution_mode"] = "adapter"
            row["readiness_status"] = "adapter"
            row.update(incumbent_solver_profile)

    report = analyze_results(
        config,
        rows,
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit="fixture",
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
    )

    incumbent_rows = [row for row in report["candidate_rows"] if not row["target"]]
    assert len(incumbent_rows) == len(config["incumbent_arms"])
    assert all(row["excluded_episodes"] == 0 for row in incumbent_rows)
    assert {row["arm_key"] for row in incumbent_rows} == {
        arm["key"] for arm in config["incumbent_arms"]
    }
    assert report["read"]["incumbent_rates"], "incumbent rates must be available for the read"


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


def test_runner_returns_nonzero_for_blocked_study(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A blocked report must fail the production wrapper instead of printing success."""

    def blocked_study(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "status": "blocked",
            "issue": 5579,
            "read": {"decision": "blocked"},
            "eligible_episode_rows": 0,
            "excluded_episode_rows": 1,
        }

    monkeypatch.setattr(sensitivity_runner, "run_study", blocked_study)
    assert (
        sensitivity_runner.main(
            ["--config", str(CONFIG), "--phase", "held_out", "--out-dir", str(tmp_path)]
        )
        == 1
    )


def test_canary_stops_on_missing_arm_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A skipped batch is reported as a missing arm, without launching the other arm."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    calls = 0

    def skipped_batch(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "benchmark_availability": {
                "execution_mode": "native",
                "readiness_status": "degraded",
                "availability_status": "not_available",
                "benchmark_success": False,
            }
        }

    monkeypatch.setattr(sensitivity_runner, "run_map_batch", skipped_batch)
    result = sensitivity_runner.run_canary_check(
        config,
        out_dir=tmp_path,
        config_path=CONFIG,
    )
    assert result["status"] == "failed"
    assert result["stop_reason"] == "missing_or_incomplete_arm"
    assert result["validated_arm_key"] == "prediction_mpc"
    assert calls == 1


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
                        "solver_execution_mode": "native",
                        "valid_solver_provenance": True,
                        "finite_commands": True,
                        "solver_successes": 1,
                        "solver_failures": 0,
                        "fallback_stop_count": 0,
                        "control_updates": 1,
                        "native_solver_eligible": True,
                        "native_solver_exclusion_reasons": [],
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


def _strict_canary_row(*, arm_key: str, scenario_id: str) -> dict[str, object]:
    """Build a normalized row with every strict native-solver canary predicate satisfied."""
    return {
        "arm_key": arm_key,
        "candidate_id": "incumbent",
        "scenario_id": scenario_id,
        "seed": 101,
        "execution_mode": "native",
        "readiness_status": "native",
        "availability_status": "available",
        "benchmark_success": True,
        "planner_runtime_status": "eligible",
        "solver_execution_mode": "native",
        "valid_solver_provenance": True,
        "finite_commands": True,
        "solver_successes": 1,
        "solver_failures": 0,
        "fallback_stop_count": 0,
        "control_updates": 1,
        "native_solver_eligible": True,
        "native_solver_exclusion_reasons": [],
    }
