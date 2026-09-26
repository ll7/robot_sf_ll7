"""Tests for robot_sf.benchmark.map_runner.map_runner_batch_summary — batch-summary metadata."""

from __future__ import annotations

from robot_sf.benchmark.fallback_policy import summarize_benchmark_availability
from robot_sf.benchmark.map_runner.map_runner_batch_runner import _initial_feasibility_totals
from robot_sf.benchmark.map_runner.map_runner_batch_summary import (
    WorkerMetadataBridgeUpdate,
    _float_metadata_value,
    accumulate_batch_metadata,
    apply_worker_metadata_bridge,
    build_ammv_feasibility_summary,
    merge_runtime_algorithm_contract,
)


def _guarded_episode_metadata(
    *, decision_label: str, stop_count: object, safe_count: object
) -> dict[str, object]:
    """Build one guarded producer metadata payload with shield telemetry."""
    return {
        "algorithm": "ppo",
        "canonical_algorithm": "guarded_ppo",
        "planner_contract": {"planner_id": "guarded_ppo"},
        "planner_kinematics": {"execution_mode": "mixed"},
        "planner_runtime": {"last_decision": {"decision_label": decision_label}},
        "guard_stats": {"stop_best_effort": stop_count, "fallback_safe": safe_count},
        "shield_stats": {
            "decision_counts": {
                "stop_best_effort": stop_count,
                "fallback_safe": safe_count,
            },
            "last_decision": {"decision_label": decision_label},
        },
    }


def _guarded_contract_base() -> dict[str, object]:
    """Return the static identity that the real batch contract supplies."""
    return {
        "algorithm": "ppo",
        "canonical_algorithm": "guarded_ppo",
        "planner_contract": {"planner_id": "guarded_ppo"},
        "planner_kinematics": {"execution_mode": "mixed"},
    }


def test_stop_best_effort_from_earlier_episode_blocks_later_safe_summary() -> None:
    """Aggregate availability must retain an earlier stop decision."""
    earlier_stop = _guarded_episode_metadata(
        decision_label="stop_best_effort", stop_count=1, safe_count=0
    )
    later_safe = _guarded_episode_metadata(
        decision_label="fallback_safe", stop_count=0, safe_count=1
    )
    contract = merge_runtime_algorithm_contract(_guarded_contract_base(), earlier_stop)
    merge_runtime_algorithm_contract(contract, later_safe)
    summary = {
        "status": "ok",
        "total_jobs": 2,
        "written": 2,
        "failed_jobs": 0,
        "algorithm_readiness": {"name": "guarded_ppo"},
        "algorithm_metadata_contract": contract,
    }

    availability = summarize_benchmark_availability(summary)

    assert availability.benchmark_success is False
    assert contract["guard_stats"] == {"stop_best_effort": 1, "fallback_safe": 1}
    assert contract["shield_stats"]["decision_counts"] == {
        "stop_best_effort": 1,
        "fallback_safe": 1,
    }
    assert contract["shield_stats"]["last_decision"] == {"decision_label": "stop_best_effort"}


def test_safe_guard_telemetry_aggregates_numeric_and_remains_available() -> None:
    """Multiple identity-bound safe episodes retain numeric counters and availability."""
    first = _guarded_episode_metadata(decision_label="fallback_safe", stop_count=0, safe_count=1)
    second = _guarded_episode_metadata(decision_label="fallback_safe", stop_count=0, safe_count=2)
    contract = merge_runtime_algorithm_contract(_guarded_contract_base(), first)
    merge_runtime_algorithm_contract(contract, second)
    summary = {
        "status": "ok",
        "total_jobs": 2,
        "written": 2,
        "failed_jobs": 0,
        "algorithm_readiness": {"name": "guarded_ppo"},
        "algorithm_metadata_contract": contract,
    }

    availability = summarize_benchmark_availability(summary)

    assert contract["guard_stats"]["fallback_safe"] == 3
    assert contract["shield_stats"]["decision_counts"]["fallback_safe"] == 3
    assert "last_decision" not in contract["shield_stats"]
    assert availability.benchmark_success is True


def test_malformed_stop_counter_in_aggregate_fails_closed() -> None:
    """Malformed producer counters must survive aggregation as rejection evidence."""
    malformed = _guarded_episode_metadata(
        decision_label="fallback_safe", stop_count="1", safe_count=1
    )
    contract = merge_runtime_algorithm_contract(_guarded_contract_base(), malformed)
    summary = {
        "status": "ok",
        "total_jobs": 1,
        "written": 1,
        "failed_jobs": 0,
        "algorithm_readiness": {"name": "guarded_ppo"},
        "algorithm_metadata_contract": contract,
    }

    availability = summarize_benchmark_availability(summary)

    assert availability.benchmark_success is False


def test_malformed_telemetry_container_survives_later_safe_episode() -> None:
    """A later valid episode cannot erase malformed aggregate telemetry."""
    malformed = {
        **_guarded_contract_base(),
        "guard_stats": None,
        "shield_stats": {"decision_counts": None},
    }
    later_safe = _guarded_episode_metadata(
        decision_label="fallback_safe", stop_count=0, safe_count=1
    )
    contract = merge_runtime_algorithm_contract(_guarded_contract_base(), malformed)
    merge_runtime_algorithm_contract(contract, later_safe)
    summary = {
        "status": "ok",
        "total_jobs": 2,
        "written": 2,
        "failed_jobs": 0,
        "algorithm_readiness": {"name": "guarded_ppo"},
        "algorithm_metadata_contract": contract,
    }

    availability = summarize_benchmark_availability(summary)

    assert contract["guard_stats"] is None
    assert contract["shield_stats"]["decision_counts"] is None
    assert availability.benchmark_success is False


def test_shield_counter_merge_sums_valid_values_and_preserves_malformed_evidence() -> None:
    """Shield counters aggregate while malformed values remain visible to policy checks."""
    first = _guarded_episode_metadata(decision_label="fallback_safe", stop_count=0, safe_count=1)
    first_shield = first["shield_stats"]
    assert isinstance(first_shield, dict)
    first_shield.update(
        {
            "decision_count": 2,
            "override_count": 0,
            "intervention_count": "malformed-existing",
            "decision_counts": {
                "stop_best_effort": 0,
                "fallback_safe": 1,
                "preserve_invalid": "malformed-existing",
                "becomes_invalid": 0,
            },
            "violated_constraint_counts": {"wall": 1},
        }
    )
    second = _guarded_episode_metadata(
        decision_label="fallback_safe", stop_count="malformed-new", safe_count=2
    )
    second_shield = second["shield_stats"]
    assert isinstance(second_shield, dict)
    second_shield.update(
        {
            "decision_count": 3,
            "pass_through_count": 4,
            "override_count": "malformed-new",
            "intervention_count": 9,
            "decision_counts": {
                "stop_best_effort": "malformed-new",
                "fallback_safe": 2,
                "preserve_invalid": 5,
                "becomes_invalid": "malformed-new",
                "new_event": 4,
            },
            "violated_constraint_counts": "malformed-container",
            "last_decision": {},
            "diagnostic_payload": "new-value",
        }
    )

    contract = merge_runtime_algorithm_contract(_guarded_contract_base(), first)
    merge_runtime_algorithm_contract(contract, second)
    shield_stats = contract["shield_stats"]
    assert isinstance(shield_stats, dict)
    decision_counts = shield_stats["decision_counts"]
    assert isinstance(decision_counts, dict)
    assert contract["guard_stats"]["stop_best_effort"] == "malformed-new"
    assert contract["guard_stats"]["fallback_safe"] == 3
    assert shield_stats["decision_count"] == 5
    assert shield_stats["pass_through_count"] == 4
    assert shield_stats["override_count"] == "malformed-new"
    assert shield_stats["intervention_count"] == "malformed-existing"
    assert decision_counts == {
        "stop_best_effort": "malformed-new",
        "fallback_safe": 3,
        "preserve_invalid": "malformed-existing",
        "becomes_invalid": "malformed-new",
        "new_event": 4,
    }
    assert shield_stats["violated_constraint_counts"] == "malformed-container"
    assert "last_decision" not in shield_stats
    assert shield_stats["diagnostic_payload"] == "new-value"

    availability = summarize_benchmark_availability(
        {
            "status": "ok",
            "total_jobs": 2,
            "written": 2,
            "failed_jobs": 0,
            "algorithm_readiness": {"name": "guarded_ppo"},
            "algorithm_metadata_contract": contract,
        }
    )
    assert availability.benchmark_success is False


def test_malformed_guard_container_replaces_valid_batch_container() -> None:
    """A malformed later telemetry container remains available for fail-closed rejection."""
    base = _guarded_contract_base()
    base["guard_stats"] = {"fallback_safe": 2}

    contract = merge_runtime_algorithm_contract(base, {"guard_stats": None})

    assert contract["guard_stats"] is None


class TestFloatMetadataValue:
    """Tests for _float_metadata_value conversion."""

    def test_none_returns_default(self) -> None:
        """None must return the default value."""
        assert _float_metadata_value(None) == 0.0
        assert _float_metadata_value(None, default=5.0) == 5.0

    def test_numeric_value_converted(self) -> None:
        """Numeric values must be converted to float."""
        assert _float_metadata_value(3) == 3.0
        assert _float_metadata_value(2.5) == 2.5

    def test_zero_preserved(self) -> None:
        """Zero must not be treated as missing."""
        assert _float_metadata_value(0) == 0.0
        assert _float_metadata_value(0.0) == 0.0

    def test_falsy_but_valid(self) -> None:
        """Falsy but valid numeric values must be preserved."""
        assert _float_metadata_value(False) == 0.0


class TestAccumulateBatchMetadata:
    """Tests for accumulate_batch_metadata adapter and feasibility folding."""

    def test_empty_record(self) -> None:
        """A record without algorithm_metadata must return zero deltas."""
        totals = _initial_feasibility_totals()
        seen, native, adapted = accumulate_batch_metadata({}, feasibility_totals=totals)
        assert seen is False
        assert native == 0
        assert adapted == 0

    def test_adapter_impact_extracted(self) -> None:
        """Adapter impact fields must be extracted from algorithm_metadata."""
        rec = {
            "algorithm_metadata": {
                "adapter_impact": {
                    "requested": True,
                    "native_steps": 100,
                    "adapted_steps": 20,
                }
            }
        }
        totals = _initial_feasibility_totals()
        seen, native, adapted = accumulate_batch_metadata(rec, feasibility_totals=totals)
        assert seen is True
        assert native == 100
        assert adapted == 20

    def test_feasibility_totals_accumulated(self) -> None:
        """Feasibility counters must be accumulated into totals."""
        rec = {
            "algorithm_metadata": {
                "kinematics_feasibility": {
                    "commands_evaluated": 50,
                    "infeasible_native_count": 3,
                    "projected_count": 2,
                    "mean_abs_delta_linear": 0.1,
                    "mean_abs_delta_angular": 0.05,
                    "max_abs_delta_linear": 0.5,
                    "max_abs_delta_angular": 0.3,
                }
            }
        }
        totals = _initial_feasibility_totals()
        accumulate_batch_metadata(rec, feasibility_totals=totals)
        assert totals["commands_evaluated"] == 50
        assert totals["infeasible_native_count"] == 3
        assert totals["projected_count"] == 2
        assert totals["sum_abs_delta_linear"] == 5.0  # 0.1 * 50
        assert totals["max_abs_delta_linear"] == 0.5

    def test_ammv_feasibility_accumulated(self) -> None:
        """AMMV feasibility fields must be accumulated into totals."""
        rec = {
            "algorithm_metadata": {
                "ammv_feasibility": {
                    "n_commands": 10,
                    "n_curvature_violations": 1,
                    "feasible": True,
                    "tip_over_violation": False,
                    "min_stability_margin": 0.5,
                }
            }
        }
        totals = _initial_feasibility_totals()
        accumulate_batch_metadata(rec, feasibility_totals=totals)
        assert totals["ammv_episode_count"] == 1
        assert totals["ammv_commands_evaluated"] == 10
        assert totals["ammv_curvature_violation_count"] == 1
        assert totals["ammv_feasible_episode_count"] == 1
        assert totals["ammv_tip_over_episode_count"] == 0
        assert totals["ammv_min_stability_margin"] == 0.5


class TestMergeRuntimeAlgorithmContract:
    """Tests for merge_runtime_algorithm_contract."""

    def test_non_dict_inputs_return_base(self) -> None:
        """Non-dict inputs must return the base contract unchanged."""
        assert merge_runtime_algorithm_contract("not_a_dict", {}) == "not_a_dict"  # type: ignore[arg-type]
        assert merge_runtime_algorithm_contract({}, "not_a_dict") == {}

    def test_planner_kinematics_merged(self) -> None:
        """planner_kinematics from runtime must be merged into the base."""
        base: dict = {}
        runtime = {"planner_kinematics": {"robot_kinematics": "differential_drive"}}
        result = merge_runtime_algorithm_contract(base, runtime)
        assert result["planner_kinematics"]["robot_kinematics"] == "differential_drive"

    def test_placeholder_replaced_by_runtime(self) -> None:
        """Placeholder values (unknown, empty) must be replaced by runtime data."""
        base: dict = {"planner_kinematics": {"execution_mode": "unknown"}}
        runtime = {"planner_kinematics": {"execution_mode": "native"}}
        result = merge_runtime_algorithm_contract(base, runtime)
        assert result["planner_kinematics"]["execution_mode"] == "native"

    def test_conflicting_non_authoritative_becomes_mixed(self) -> None:
        """Conflicting non-authoritative values must become 'mixed'."""
        base: dict = {"planner_kinematics": {"custom_field": "value_a"}}
        runtime = {"planner_kinematics": {"custom_field": "value_b"}}
        result = merge_runtime_algorithm_contract(base, runtime)
        assert result["planner_kinematics"]["custom_field"] == "mixed"

    def test_authoritative_key_overrides(self) -> None:
        """Authoritative keys must be overridden by runtime values."""
        base: dict = {"planner_kinematics": {"adapter_name": "old_adapter"}}
        runtime = {"planner_kinematics": {"adapter_name": "new_adapter"}}
        result = merge_runtime_algorithm_contract(base, runtime)
        assert result["planner_kinematics"]["adapter_name"] == "new_adapter"

    def test_upstream_reference_merged(self) -> None:
        """upstream_reference from runtime must be merged."""
        base: dict = {}
        runtime = {"upstream_reference": {"source": "repo"}}
        result = merge_runtime_algorithm_contract(base, runtime)
        assert result["upstream_reference"]["source"] == "repo"

    def test_checkpoint_provenance_merged(self) -> None:
        """checkpoint_provenance from planner_runtime must be merged."""
        base: dict = {}
        runtime = {"planner_runtime": {"checkpoint_provenance": {"hash": "abc123"}}}
        result = merge_runtime_algorithm_contract(base, runtime)
        assert result["checkpoint_provenance"]["hash"] == "abc123"

    def test_runtime_fallback_markers_and_nested_metadata_are_merged(self) -> None:
        """Runtime fallback markers must be monotonic while nested metadata is preserved."""
        base: dict = {
            "planner_runtime": {
                "fallback": False,
                "fallback_count": 1,
                "readiness_status": "native",
                "provenance": {"source": "base"},
            }
        }
        runtime = {
            "planner_runtime": {
                "fallback": True,
                "fallback_count": 3,
                "readiness_status": "degraded",
                "provenance": {"checkpoint": "runtime"},
            }
        }

        result = merge_runtime_algorithm_contract(base, runtime)

        planner_runtime = result["planner_runtime"]
        assert planner_runtime["fallback"] is True
        assert planner_runtime["fallback_count"] == 3
        assert planner_runtime["readiness_status"] == "degraded"
        assert planner_runtime["provenance"] == {"source": "base", "checkpoint": "runtime"}

    def test_nested_runtime_merge_does_not_mutate_episode_metadata(self) -> None:
        """Batch aggregation must not alias and rewrite a previously emitted episode row."""
        first_episode = {
            "planner_runtime": {
                "selected_source_counts": {"dynamic_window": 4},
                "fallback_count": 0,
            }
        }
        contract = merge_runtime_algorithm_contract({}, first_episode)

        merge_runtime_algorithm_contract(
            contract,
            {
                "planner_runtime": {
                    "selected_source_counts": {"all_candidates_rejected": 3},
                    "fallback_count": 3,
                }
            },
        )

        assert first_episode["planner_runtime"]["selected_source_counts"] == {"dynamic_window": 4}

    def test_native_protective_stop_telemetry_aggregates_numerically(self) -> None:
        contract: dict = {}
        merge_runtime_algorithm_contract(
            contract,
            {
                "planner_runtime": {
                    "protective_stop_count": 2,
                    "selected_source_counts": {
                        "dynamic_window": 4,
                        "dynamic_protective_stop": 2,
                    },
                }
            },
        )
        merge_runtime_algorithm_contract(
            contract,
            {
                "planner_runtime": {
                    "protective_stop_count": 3,
                    "selected_source_counts": {
                        "dynamic_window": 5,
                        "static_protective_stop": 3,
                    },
                }
            },
        )

        assert contract["planner_runtime"]["protective_stop_count"] == 5
        assert contract["planner_runtime"]["selected_source_counts"] == {
            "dynamic_protective_stop": 2,
            "dynamic_window": 9,
            "static_protective_stop": 3,
        }


class TestBuildAmmvFeasibilitySummary:
    """Tests for build_ammv_feasibility_summary."""

    def test_no_episodes(self) -> None:
        """Without AMMV episodes, status must be 'no_ammv_episodes'."""
        result = build_ammv_feasibility_summary({})
        assert result["status"] == "no_ammv_episodes"
        assert result["episode_count"] == 0
        assert result["feasible"] is False
        assert result["min_stability_margin"] is None

    def test_all_feasible(self) -> None:
        """All-feasible episodes must produce feasible=True."""
        totals = {
            "ammv_episode_count": 3,
            "ammv_feasible_episode_count": 3,
            "ammv_commands_evaluated": 30,
            "ammv_tip_over_episode_count": 0,
            "ammv_curvature_violation_count": 0,
            "ammv_min_stability_margin": 0.8,
        }
        result = build_ammv_feasibility_summary(totals)
        assert result["status"] == "available"
        assert result["feasible"] is True
        assert result["tip_over_violation"] is False
        assert result["min_stability_margin"] == 0.8

    def test_tip_over_violation(self) -> None:
        """Any tip-over episode must set tip_over_violation=True."""
        totals = {
            "ammv_episode_count": 2,
            "ammv_feasible_episode_count": 1,
            "ammv_commands_evaluated": 20,
            "ammv_tip_over_episode_count": 1,
            "ammv_curvature_violation_count": 0,
            "ammv_min_stability_margin": 0.3,
        }
        result = build_ammv_feasibility_summary(totals)
        assert result["tip_over_violation"] is True
        assert result["feasible"] is False

    def test_schema_version_present(self) -> None:
        """The summary must include the schema version."""
        result = build_ammv_feasibility_summary({})
        assert result["schema_version"] == "ammv_feasibility.v1"

    def test_evidence_kind_markers(self) -> None:
        """The summary must carry diagnostic proxy claim-boundary markers."""
        result = build_ammv_feasibility_summary({})
        assert result["evidence_kind"] == "diagnostic_proxy"
        assert result["proxy_kind"] == "internal_non_hardware"


class TestApplyWorkerMetadataBridge:
    """Tests for apply_worker_metadata_bridge."""

    def test_returns_named_tuple(self) -> None:
        """The bridge must return a WorkerMetadataBridgeUpdate."""
        totals = _initial_feasibility_totals()
        update = apply_worker_metadata_bridge(
            {}, feasibility_totals=totals, runtime_algorithm_contract=None
        )
        assert isinstance(update, WorkerMetadataBridgeUpdate)
        assert update.adapter_requested_seen is False
        assert update.adapter_native_steps == 0

    def test_merges_runtime_contract(self) -> None:
        """The bridge must merge runtime algorithm metadata into the contract."""
        totals = _initial_feasibility_totals()
        rec = {"algorithm_metadata": {"planner_kinematics": {"robot_kinematics": "holonomic"}}}
        update = apply_worker_metadata_bridge(
            rec, feasibility_totals=totals, runtime_algorithm_contract={}
        )
        assert (
            update.runtime_algorithm_contract["planner_kinematics"]["robot_kinematics"]
            == "holonomic"
        )
