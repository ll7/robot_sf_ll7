"""Tests for robot_sf.benchmark.map_runner_batch_summary — batch-summary metadata."""

from __future__ import annotations

from robot_sf.benchmark.map_runner_batch_runner import _initial_feasibility_totals
from robot_sf.benchmark.map_runner_batch_summary import (
    WorkerMetadataBridgeUpdate,
    _float_metadata_value,
    accumulate_batch_metadata,
    apply_worker_metadata_bridge,
    build_ammv_feasibility_summary,
    merge_runtime_algorithm_contract,
)


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
