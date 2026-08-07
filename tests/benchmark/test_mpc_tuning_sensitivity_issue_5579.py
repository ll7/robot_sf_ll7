"""Contract tests for the issue #5579 MPC tuning-budget sensitivity packet."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import scripts.benchmark.run_mpc_tuning_sensitivity_issue_5579 as sensitivity_runner
from robot_sf.benchmark.mpc_tuning_sensitivity import (
    TARGET_ARM_KEYS,
    analyze_results,
    build_candidate_plan,
    compute_scenario_list_hash,
    config_hash,
    load_sensitivity_config,
    load_tuning_selection,
    normalize_episode_record,
    selected_scenarios,
    solver_execution_contract,
    validate_canary_rows,
    validate_sensitivity_config,
    write_report,
    write_tuning_selection,
)
from scripts.benchmark.run_mpc_tuning_sensitivity_issue_5579 import _display_path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/analysis/issue_5579_mpc_tuning_sensitivity_v2.yaml"
FIXTURE_RUN_COMMIT = "0" * 40


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


def test_held_out_plan_contains_only_tuning_selected_targets() -> None:
    """Held-out execution cannot run or select any unselected target candidate."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    selected = {
        "prediction_mpc": "speed_low",
        "prediction_mpc_cbf": "horizon_high",
    }
    plan = build_candidate_plan(config, repo_root=ROOT, target_candidate_ids=selected)
    targets = [entry for entry in plan if entry["target"]]
    incumbents = [entry for entry in plan if not entry["target"]]
    assert len(targets) == len(TARGET_ARM_KEYS)
    assert {(entry["arm_key"], entry["candidate_id"]) for entry in targets} == set(selected.items())
    assert len(incumbents) == 4
    assert all(entry["candidate_id"] == "incumbent" for entry in incumbents)


def test_held_out_analysis_rejects_missing_selection() -> None:
    """The analyzer cannot re-select a candidate from held-out outcomes."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    with pytest.raises(ValueError, match="do not re-select candidates"):
        analyze_results(
            config,
            [],
            repo_root=ROOT,
            config_path=str(CONFIG),
            run_commit=FIXTURE_RUN_COMMIT,
            reproduction_command="fixture",
            raw_artifact_root="output/fixture",
            scope_name="held_out_scope",
        )


def test_tuning_selection_round_trip_and_fixed_held_out_report(tmp_path: Path) -> None:
    """Selection is frozen from tuning rows and held-out reporting uses one target per arm."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    tuning_plan = build_candidate_plan(config, repo_root=ROOT)
    tuning_report = analyze_results(
        config,
        _fixture_rows(config, tuning_plan),
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit=FIXTURE_RUN_COMMIT,
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
        scope_name="tuning_scope",
    )
    selection_path = tmp_path / "tuning_selection.json"
    source_report = _persisted_tuning_report(tuning_report, tmp_path / "tuning")
    payload = write_tuning_selection(
        tuning_report,
        config,
        output_path=selection_path,
        config_path=CONFIG,
        repo_root=ROOT,
        source_report=source_report,
    )
    selected = load_tuning_selection(
        selection_path,
        config,
        config_path=CONFIG,
        repo_root=ROOT,
    )
    assert payload["selection_scope"] == "tuning_scope"
    assert set(selected) == set(TARGET_ARM_KEYS)
    assert selected == payload["selected_target_candidates"]

    held_out_plan = build_candidate_plan(
        config,
        repo_root=ROOT,
        target_candidate_ids=selected,
    )
    held_out_report = analyze_results(
        config,
        _fixture_rows(config, held_out_plan, scope_name="held_out_scope"),
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit=FIXTURE_RUN_COMMIT,
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
        scope_name="held_out_scope",
        target_candidate_ids=selected,
        selection_artifact=str(selection_path),
    )
    assert held_out_report["selection_mode"] == "fixed_from_tuning"
    assert held_out_report["executed_candidate_count"] == 6
    assert held_out_report["total_episode_rows"] == 6 * 45 * 10
    assert all(summary["candidate_count"] == 1 for summary in held_out_report["target_summary"])
    assert held_out_report["read"]["selection_mode"] == "fixed_from_tuning"
    assert "tuning-selected" in held_out_report["read"]["detail"]

    tampered = json.loads(selection_path.read_text(encoding="utf-8"))
    tampered["config_sha256"] = "tampered"
    selection_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="config provenance"):
        load_tuning_selection(
            selection_path,
            config,
            config_path=CONFIG,
            repo_root=ROOT,
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
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    contract = solver_execution_contract(config)
    record = _raw_record(route_complete=True, collision_event=False)
    effective_config = {"predictor_backend": "constant_velocity", "fallback_to_stop": False}
    metadata = record["algorithm_metadata"]
    assert isinstance(metadata, dict)
    metadata["config"] = effective_config
    metadata["config_hash"] = config_hash(effective_config)
    metadata["planner_kinematics"] = {
        "execution_mode": "adapter",
        "adapter_name": "PredictionMPCPlannerAdapter",
        "supports_native_commands": False,
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
        solver_contract=contract,
    )
    assert row["solver_execution_mode"] == "prediction_mpc_native_solver"
    assert row["valid_solver_provenance"] is True
    assert row["finite_commands"] is True
    assert row["native_solver_eligible"] is True

    metadata["planner_kinematics"]["adapter_name"] = "SocialForcePlannerAdapter"
    foreign_planner_row = normalize_episode_record(
        record,
        arm_key="prediction_mpc",
        candidate_id="incumbent",
        expected_config_hash=config_hash(effective_config),
        solver_contract=contract,
    )
    assert foreign_planner_row["native_solver_eligible"] is False
    assert "unexpected_solver_planner" in foreign_planner_row["native_solver_exclusion_reasons"]


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

    mixed_execution = deepcopy(rows)
    mixed_execution[0]["execution_mode"] = "mixed"
    failed = validate_canary_rows(
        mixed_execution,
        scenario_ids=scenarios,
        seed=101,
        required_eligible=6,
    )
    assert failed["status"] == "failed"
    assert "execution_mode_mismatch" in failed["invalid_rows"][0]["reasons"]

    relabelled_solver = deepcopy(rows)
    relabelled_solver[0]["solver_execution_mode"] = "adapter"
    failed_solver = validate_canary_rows(
        relabelled_solver,
        scenario_ids=scenarios,
        seed=101,
        required_eligible=6,
    )
    assert failed_solver["status"] == "failed"
    assert "native_solver_execution_missing" in failed_solver["invalid_rows"][0]["reasons"]

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
                        "solver_execution_mode": "prediction_mpc_native_solver",
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
        run_commit=FIXTURE_RUN_COMMIT,
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
        run_commit=FIXTURE_RUN_COMMIT,
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
        run_commit=FIXTURE_RUN_COMMIT,
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
        run_commit=FIXTURE_RUN_COMMIT,
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
        run_commit=FIXTURE_RUN_COMMIT,
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
        run_commit=FIXTURE_RUN_COMMIT,
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


def test_runner_rejects_held_out_execution_without_selection(tmp_path: Path) -> None:
    """The production runner fails before any episode work without tuning selection."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    with pytest.raises(ValueError, match="requires a completed tuning selection artifact"):
        sensitivity_runner.run_study(
            config,
            out_dir=tmp_path,
            config_path=CONFIG,
            phase="held_out",
        )


def test_runner_rejects_unbound_held_out_candidates(tmp_path: Path) -> None:
    """A held-out run cannot proceed from candidate IDs without a validated artifact."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    selected = {"prediction_mpc": "speed_low", "prediction_mpc_cbf": "horizon_high"}
    with pytest.raises(ValueError, match="completed tuning selection artifact"):
        sensitivity_runner.run_study(
            config,
            out_dir=tmp_path,
            config_path=CONFIG,
            phase="held_out",
            target_candidate_ids=selected,
        )


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
    monkeypatch.setattr(
        sensitivity_runner,
        "load_tuning_selection",
        lambda *_args, **_kwargs: {
            "prediction_mpc": "speed_low",
            "prediction_mpc_cbf": "horizon_high",
        },
    )
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
            run_commit=FIXTURE_RUN_COMMIT,
            reproduction_command="fixture",
            raw_artifact_root="output/fixture",
        )


def test_canary_solver_contract_is_reachable_from_the_real_policy_builder() -> None:
    """The declared canary contract must be satisfiable by the actual runner metadata.

    The 2026-08-03 #5579 freeze demands a 6/6 native-solver canary before production
    compute. Binding that gate to ``planner_kinematics.execution_mode == "native"`` made
    it unsatisfiable: the canonical ``prediction_mpc`` planner is registry-declared as an
    adapter-projected unicycle_vw planner, so every real row was rejected before tuning and
    the campaign stop rule silently became a permanent block. This test walks the real
    policy builder for both declared target configs and proves the gate can pass.
    """
    from robot_sf.benchmark.map_runner import build_map_policy

    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    contract = solver_execution_contract(config)
    for arm in config["target_arms"]:
        algo_config = yaml.safe_load(
            (ROOT / str(arm["algo_config_path"])).read_text(encoding="utf-8")
        )
        _policy, metadata = build_map_policy(str(arm["algo"]), algo_config)
        kinematics = metadata["planner_kinematics"]
        assert kinematics["adapter_name"] == contract["solver_planner_adapter"]
        assert kinematics["execution_mode"] == contract["planner_execution_mode"]
        assert kinematics["supports_native_commands"] == contract["supports_native_commands"]

        record = _raw_record(route_complete=True, collision_event=False)
        record["algorithm_metadata"] = {
            **metadata,
            "config": algo_config,
            "config_hash": config_hash(algo_config),
            "planner_runtime": {
                "solver_successes": 1,
                "solver_failures": 0,
                "fallback_stop_count": 0,
                "successful_control_updates": 1,
                "mean_abs_linear": 0.3,
                "mean_abs_angular": 0.1,
            },
        }
        row = normalize_episode_record(
            record,
            arm_key=str(arm["key"]),
            candidate_id="incumbent",
            expected_config_hash=config_hash(algo_config),
            solver_contract=contract,
        )
        assert row["native_solver_exclusion_reasons"] == []
        assert row["native_solver_eligible"] is True

        gate = validate_canary_rows(
            [{**row, "scenario_id": scenario_id, "seed": 101} for scenario_id in ("a", "b", "c")],
            scenario_ids=("a", "b", "c"),
            seed=101,
            required_eligible=3,
            target_arm_keys=(str(arm["key"]),),
            solver_contract=contract,
        )
        assert gate["status"] == "ok", gate["invalid_rows"]


def test_packet_rejects_an_unreachable_canary_solver_contract() -> None:
    """A canary the runtime can never satisfy is a permanent block, not a stop rule."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    unreachable = deepcopy(config)
    unreachable["canary"]["solver_execution"]["planner_execution_mode"] = "native"
    unreachable["canary"]["solver_execution"]["supports_native_commands"] = True
    with pytest.raises(ValueError, match="unreachable for target arm"):
        validate_sensitivity_config(unreachable, repo_root=ROOT)

    relaxed = deepcopy(config)
    relaxed["canary"]["solver_execution"]["forbid_fallback"] = False
    with pytest.raises(ValueError, match="forbid_fallback must be true"):
        validate_sensitivity_config(relaxed, repo_root=ROOT)


def test_held_out_report_produces_paired_bootstrap_and_holm_inference() -> None:
    """A complete held-out report must carry all eight preregistered paired contrasts."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    selected = {"prediction_mpc": "speed_low", "prediction_mpc_cbf": "horizon_high"}
    plan = build_candidate_plan(config, repo_root=ROOT, target_candidate_ids=selected)
    rows = _fixture_rows(config, plan, scope_name="held_out_scope")
    # Give the incumbents a strictly higher paired success rate on every seed block.
    for row in rows:
        row["success"] = row["arm_key"] not in TARGET_ARM_KEYS

    report = analyze_results(
        config,
        rows,
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit=FIXTURE_RUN_COMMIT,
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
        scope_name="held_out_scope",
        target_candidate_ids=selected,
    )

    inference = report["inference"]
    assert inference["status"] == "complete"
    assert inference["bootstrap"]["replicates"] == 2000
    assert inference["bootstrap"]["confidence_level"] == 0.95
    assert inference["multiplicity"]["method"] == "holm_bonferroni"
    assert len(inference["contrasts"]) == 8
    assert {
        (contrast["target_arm"], contrast["incumbent_arm"]) for contrast in inference["contrasts"]
    } == {
        (target_arm, incumbent_arm)
        for target_arm in TARGET_ARM_KEYS
        for incumbent_arm in (arm["key"] for arm in config["incumbent_arms"])
    }
    for contrast in inference["contrasts"]:
        assert contrast["paired_units"] == 450
        assert contrast["paired_delta"] == pytest.approx(-1.0)
        assert contrast["ci_lower"] == pytest.approx(-1.0)
        assert contrast["ci_upper"] == pytest.approx(-1.0)
        assert 0.0 < contrast["p_value"] <= 1.0
        assert contrast["holm_rank"] in range(1, 9)
        assert contrast["holm_family_size"] == 8
        assert contrast["holm_significant"] is True
        assert contrast["target_candidate_id"] == selected[contrast["target_arm"]]

    read = report["read"]
    assert read["inference_status"] == "complete"
    assert read["inference_decision"] == "incumbent_advantage_supported_on_declared_contrasts"
    assert len(read["inference_paired_deltas"]) == 8

    # The frozen bootstrap seed keeps the interval and p-value reproducible.
    repeat = analyze_results(
        config,
        rows,
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit=FIXTURE_RUN_COMMIT,
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
        scope_name="held_out_scope",
        target_candidate_ids=selected,
    )
    assert repeat["inference"] == inference


def test_tuning_scope_report_marks_inference_not_applicable() -> None:
    """The frozen inference contract is defined on the held-out suite only."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    plan = build_candidate_plan(config, repo_root=ROOT)
    report = analyze_results(
        config,
        _fixture_rows(config, plan, scope_name="tuning_scope"),
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit=FIXTURE_RUN_COMMIT,
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
        scope_name="tuning_scope",
    )
    assert report["inference"]["status"] == "not_applicable"
    assert report["read"]["inference_decision"] == "not_established"


def test_tuning_selection_rejects_a_candidate_swapped_after_tuning(tmp_path: Path) -> None:
    """A post-tuning edit of the selected candidate cannot pass the config digest alone."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    plan = build_candidate_plan(config, repo_root=ROOT)
    tuning_report = analyze_results(
        config,
        _fixture_rows(config, plan, scope_name="tuning_scope"),
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit=FIXTURE_RUN_COMMIT,
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
        scope_name="tuning_scope",
    )
    selection_path = tmp_path / "tuning_selection.json"
    source_report = _persisted_tuning_report(tuning_report, tmp_path / "tuning")
    payload = write_tuning_selection(
        tuning_report,
        config,
        output_path=selection_path,
        config_path=CONFIG,
        repo_root=ROOT,
        source_report=source_report,
    )
    assert payload["source_report"] == source_report
    assert payload["source_report_sha256"]
    assert payload["selection_input_digest"]

    assert (
        load_tuning_selection(selection_path, config, config_path=CONFIG, repo_root=ROOT)
        == payload["selected_target_candidates"]
    )

    tampered_commit = deepcopy(payload)
    tampered_commit["source_report_run_commit"] = "1" * 40
    selection_path.write_text(json.dumps(tampered_commit), encoding="utf-8")
    with pytest.raises(ValueError, match="run commit does not match"):
        load_tuning_selection(selection_path, config, config_path=CONFIG, repo_root=ROOT)

    # A held-out-informed candidate swap keeps the config digest intact and must still fail.
    declared = [point["id"] for point in config["search"]["candidate_points"]]
    original = str(payload["selected_target_candidates"]["prediction_mpc"])
    swapped = deepcopy(payload)
    swapped["selected_target_candidates"]["prediction_mpc"] = next(
        candidate_id for candidate_id in declared if candidate_id != original
    )
    selection_path.write_text(json.dumps(swapped), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the source report selection rule"):
        load_tuning_selection(selection_path, config, config_path=CONFIG, repo_root=ROOT)

    # An unreferenced selection cannot be verified at all.
    unbound = deepcopy(payload)
    unbound.pop("source_report")
    selection_path.write_text(json.dumps(unbound), encoding="utf-8")
    with pytest.raises(ValueError, match="must record the source tuning report"):
        load_tuning_selection(selection_path, config, config_path=CONFIG, repo_root=ROOT)

    # Rewriting the tuning report after the fact breaks the recorded digest.
    selection_path.write_text(json.dumps(payload), encoding="utf-8")
    edited_report_path = Path(source_report)
    edited = json.loads(edited_report_path.read_text(encoding="utf-8"))
    edited["run_commit"] = "rewritten-after-selection"
    edited_report_path.write_text(json.dumps(edited, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="source report digest does not match"):
        load_tuning_selection(selection_path, config, config_path=CONFIG, repo_root=ROOT)


def test_tuning_selection_requires_a_40_character_git_commit_sha(tmp_path: Path) -> None:
    """Selection provenance cannot be satisfied by an arbitrary placeholder token."""
    config = load_sensitivity_config(CONFIG, repo_root=ROOT)
    plan = build_candidate_plan(config, repo_root=ROOT)
    tuning_report = analyze_results(
        config,
        _fixture_rows(config, plan, scope_name="tuning_scope"),
        repo_root=ROOT,
        config_path=str(CONFIG),
        run_commit=FIXTURE_RUN_COMMIT,
        reproduction_command="fixture",
        raw_artifact_root="output/fixture",
        scope_name="tuning_scope",
    )
    malformed_report = deepcopy(tuning_report)
    malformed_report["run_commit"] = "fixture"
    source_report = _persisted_tuning_report(malformed_report, tmp_path / "tuning")
    with pytest.raises(ValueError, match="40-character Git commit SHA"):
        write_tuning_selection(
            malformed_report,
            config,
            output_path=tmp_path / "tuning_selection.json",
            config_path=CONFIG,
            repo_root=ROOT,
            source_report=source_report,
        )


def _persisted_tuning_report(report: dict, out_dir: Path) -> str:
    """Write a tuning report to disk and return the path recorded by the selection artifact."""
    paths = write_report(report, out_dir)
    return str(Path(paths["json"]).resolve())


def _fixture_rows(
    config: dict, plan: list[dict], *, scope_name: str = "scenario_scope"
) -> list[dict]:
    """Build complete eligible rows for report-contract tests."""
    rows = []
    for entry in plan:
        for scenario_id in config[scope_name]["scenario_ids"]:
            for seed in config[scope_name]["seeds"]:
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
                        "solver_execution_mode": "prediction_mpc_native_solver",
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
        "execution_mode": "adapter",
        "readiness_status": "adapter",
        "availability_status": "available",
        "benchmark_success": True,
        "planner_runtime_status": "eligible",
        "solver_execution_mode": "prediction_mpc_native_solver",
        "valid_solver_provenance": True,
        "finite_commands": True,
        "solver_successes": 1,
        "solver_failures": 0,
        "fallback_stop_count": 0,
        "control_updates": 1,
        "native_solver_eligible": True,
        "native_solver_exclusion_reasons": [],
    }
