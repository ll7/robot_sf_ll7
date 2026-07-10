"""Compatibility tests for the remaining RobotSfError migration (#4993).

Covers every subpackage family migrated in this PR:
- analysis_workbench, benchmark (individual files), common, data_ingestion,
  examples, nav, planner, research (individual files), training.

For each representative class the contract is:
  1. is_a(RobotSfError)  -- new catch target works
  2. is_a(OriginalBase)  -- legacy catch still works
  3. is_a(ConcreteClass) -- specific catch still works

For hierarchy members (ProvenanceRequiredFieldError ← ProvenanceValidationError)
we also assert transitive RobotSfError membership.
"""

from __future__ import annotations

import pytest

from robot_sf.errors import RobotSfError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_triple_catch(cls: type, base: type, msg: str = "test") -> None:
    """Assert cls is a subclass of RobotSfError, base, and itself."""
    assert issubclass(cls, RobotSfError), f"{cls.__name__} not a subclass of RobotSfError"
    assert issubclass(cls, base), f"{cls.__name__} not a subclass of {base.__name__}"
    assert issubclass(cls, cls)
    obj = cls(msg)
    assert isinstance(obj, RobotSfError)
    assert isinstance(obj, base)


# ---------------------------------------------------------------------------
# analysis_workbench
# ---------------------------------------------------------------------------


class TestAnalysisWorkbenchExceptions:
    def test_real_trace_source_discovery_error(self) -> None:
        from robot_sf.analysis_workbench.real_trace_source_discovery import (
            RealTraceSourceDiscoveryError,
        )
        _assert_triple_catch(RealTraceSourceDiscoveryError, ValueError)

    def test_real_trace_validation_contract_error(self) -> None:
        from robot_sf.analysis_workbench.real_trace_validation_contract import (
            RealTraceValidationContractError,
        )
        _assert_triple_catch(RealTraceValidationContractError, ValueError)

    def test_simulation_timeline_validation_error(self) -> None:
        from robot_sf.analysis_workbench.simulation_timeline import (
            SimulationTimelineValidationError,
        )
        _assert_triple_catch(SimulationTimelineValidationError, ValueError)

    def test_simulation_trace_export_validation_error(self) -> None:
        from robot_sf.analysis_workbench.simulation_trace_export import (
            SimulationTraceExportValidationError,
        )
        _assert_triple_catch(SimulationTraceExportValidationError, ValueError)

    def test_trace_annotation_set_validation_error(self) -> None:
        from robot_sf.analysis_workbench.trace_annotation import (
            TraceAnnotationSetValidationError,
        )
        _assert_triple_catch(TraceAnnotationSetValidationError, ValueError)

    def test_analysis_workbench_errors_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.analysis_workbench.real_trace_source_discovery import (
            RealTraceSourceDiscoveryError,
        )
        with pytest.raises(RobotSfError):
            raise RealTraceSourceDiscoveryError("no sources found")

    def test_analysis_workbench_errors_catchable_by_value_error(self) -> None:
        from robot_sf.analysis_workbench.simulation_timeline import (
            SimulationTimelineValidationError,
        )
        with pytest.raises(ValueError):
            raise SimulationTimelineValidationError("bad timeline")


# ---------------------------------------------------------------------------
# benchmark individual files
# ---------------------------------------------------------------------------


class TestBenchmarkIndividualFileExceptions:
    def test_artifact_catalog_validation_error(self) -> None:
        from robot_sf.benchmark.artifact_catalog import ArtifactCatalogValidationError
        # Custom constructor requires ArtifactCatalogIssue objects; test subclass only.
        assert issubclass(ArtifactCatalogValidationError, RobotSfError)
        assert issubclass(ArtifactCatalogValidationError, ValueError)

    def test_benchmark_claim_error(self) -> None:
        from robot_sf.benchmark.benchmark_claim import BenchmarkClaimError
        _assert_triple_catch(BenchmarkClaimError, ValueError)

    def test_benchmark_protocol_error(self) -> None:
        from robot_sf.benchmark.benchmark_protocol import BenchmarkProtocolError
        _assert_triple_catch(BenchmarkProtocolError, ValueError)

    def test_campaign_checkpoint_preflight_error(self) -> None:
        from robot_sf.benchmark.campaign_checkpoint_preflight import (
            CampaignCheckpointPreflightError,
        )
        _assert_triple_catch(CampaignCheckpointPreflightError, RuntimeError)

    def test_latency_evidence_error(self) -> None:
        from robot_sf.benchmark.control_action_latency_evidence import LatencyEvidenceError
        _assert_triple_catch(LatencyEvidenceError, RuntimeError)

    def test_exemplar_selection_error_is_robot_sf_error(self) -> None:
        from robot_sf.benchmark.exemplar_selection import ExemplarSelectionError
        # Was Exception; now inherits directly from RobotSfError
        assert issubclass(ExemplarSelectionError, RobotSfError)
        assert issubclass(ExemplarSelectionError, Exception)

    def test_failure_mechanism_classification_error(self) -> None:
        from robot_sf.benchmark.failure_mechanism_classifier import (
            FailureMechanismClassificationError,
        )
        _assert_triple_catch(FailureMechanismClassificationError, ValueError)

    def test_near_miss_ttc_input_error(self) -> None:
        from robot_sf.benchmark.near_miss_ttc import NearMissTtcInputError
        _assert_triple_catch(NearMissTtcInputError, RuntimeError)

    def test_odd_contract_validation_error(self) -> None:
        from robot_sf.benchmark.odd_contract import OddContractValidationError
        _assert_triple_catch(OddContractValidationError, ValueError)

    def test_orca_rvo2_preflight_error(self) -> None:
        from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError
        _assert_triple_catch(OrcaRvo2PreflightError, RuntimeError)

    def test_release_gates_spec_error(self) -> None:
        from robot_sf.benchmark.release_gates import ReleaseGateSpecError
        _assert_triple_catch(ReleaseGateSpecError, ValueError)

    def test_release_preflight_error(self) -> None:
        from robot_sf.benchmark.release_preflight import ReleasePreflightError
        _assert_triple_catch(ReleasePreflightError, ValueError)

    def test_visualization_error_is_robot_sf_error(self) -> None:
        from robot_sf.benchmark.visualization import VisualizationError
        assert issubclass(VisualizationError, RobotSfError)
        assert issubclass(VisualizationError, Exception)

    def test_multi_planner_overlay_error_is_robot_sf_error(self) -> None:
        from robot_sf.benchmark.multi_planner_overlay import MultiPlannerOverlayError
        assert issubclass(MultiPlannerOverlayError, RobotSfError)
        assert issubclass(MultiPlannerOverlayError, Exception)

    def test_snqi_weight_provenance_error(self) -> None:
        from robot_sf.benchmark.snqi.weights_inventory import SNQIWeightProvenanceError
        _assert_triple_catch(SNQIWeightProvenanceError, RuntimeError)

    def test_provenance_validation_error_hierarchy(self) -> None:
        """ProvenanceValidationError and its children are all RobotSfError."""
        from robot_sf.benchmark.result_provenance import (
            ProvenanceArtifactError,
            ProvenanceRequiredFieldError,
            ProvenanceRowLinkError,
            ProvenanceValidationError,
        )
        for cls in (
            ProvenanceValidationError,
            ProvenanceRequiredFieldError,
            ProvenanceArtifactError,
            ProvenanceRowLinkError,
        ):
            assert issubclass(cls, RobotSfError), f"{cls.__name__} not RobotSfError"
            assert issubclass(cls, ValueError), f"{cls.__name__} not ValueError"

    def test_provenance_subclasses_still_caught_by_parent(self) -> None:
        from robot_sf.benchmark.result_provenance import (
            ProvenanceRequiredFieldError,
            ProvenanceValidationError,
        )
        with pytest.raises(ProvenanceValidationError):
            raise ProvenanceRequiredFieldError("required field absent")

    def test_benchmark_errors_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.benchmark.benchmark_claim import BenchmarkClaimError
        with pytest.raises(RobotSfError):
            raise BenchmarkClaimError("invalid claim")

    def test_benchmark_runtime_errors_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.benchmark.near_miss_ttc import NearMissTtcInputError
        with pytest.raises(RobotSfError):
            raise NearMissTtcInputError("bad input")

    def test_benchmark_runtime_errors_legacy_catch_preserved(self) -> None:
        from robot_sf.benchmark.campaign_checkpoint_preflight import (
            CampaignCheckpointPreflightError,
        )
        with pytest.raises(RuntimeError):
            raise CampaignCheckpointPreflightError("preflight failed")


# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------


class TestCommonExceptions:
    def test_unsafe_pickle_error(self) -> None:
        from robot_sf.common.safe_pickle import UnsafePickleError
        _assert_triple_catch(UnsafePickleError, RuntimeError)

    def test_unsafe_pickle_error_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.common.safe_pickle import UnsafePickleError
        with pytest.raises(RobotSfError):
            raise UnsafePickleError("unsafe pickle detected")

    def test_unsafe_pickle_error_legacy_catch_preserved(self) -> None:
        from robot_sf.common.safe_pickle import UnsafePickleError
        with pytest.raises(RuntimeError):
            raise UnsafePickleError("unsafe pickle detected")


# ---------------------------------------------------------------------------
# data_ingestion
# ---------------------------------------------------------------------------


class TestDataIngestionExceptions:
    def test_contract_error(self) -> None:
        from robot_sf.data_ingestion.real_trajectory_contract import ContractError
        _assert_triple_catch(ContractError, RuntimeError)

    def test_contract_error_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.data_ingestion.real_trajectory_contract import ContractError
        with pytest.raises(RobotSfError):
            raise ContractError("contract violated")

    def test_contract_error_legacy_catch_preserved(self) -> None:
        from robot_sf.data_ingestion.real_trajectory_contract import ContractError
        with pytest.raises(RuntimeError):
            raise ContractError("contract violated")


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------


class TestExamplesExceptions:
    def test_manifest_validation_error(self) -> None:
        from robot_sf.examples.manifest_loader import ManifestValidationError
        _assert_triple_catch(ManifestValidationError, ValueError)

    def test_manifest_validation_error_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.examples.manifest_loader import ManifestValidationError
        with pytest.raises(RobotSfError):
            raise ManifestValidationError("manifest invalid")


# ---------------------------------------------------------------------------
# nav
# ---------------------------------------------------------------------------


class TestNavExceptions:
    def test_footprint_orientation_config_error(self) -> None:
        from robot_sf.nav.footprint_diagnostic import FootprintOrientationConfigError
        _assert_triple_catch(FootprintOrientationConfigError, ValueError)

    def test_footprint_error_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.nav.footprint_diagnostic import FootprintOrientationConfigError
        with pytest.raises(RobotSfError):
            raise FootprintOrientationConfigError("bad config")


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


class TestPlannerExceptions:
    def test_planning_error_is_robot_sf_error(self) -> None:
        from robot_sf.planner.classic_global_planner import PlanningError
        assert issubclass(PlanningError, RobotSfError)
        assert issubclass(PlanningError, Exception)

    def test_planning_failed_error_is_robot_sf_error(self) -> None:
        from robot_sf.planner.visibility_planner import PlanningFailedError
        assert issubclass(PlanningFailedError, RobotSfError)
        assert issubclass(PlanningFailedError, Exception)

    def test_learned_policy_adapter_contract_error(self) -> None:
        from robot_sf.planner.learned_policy_adapter import LearnedPolicyAdapterContractError
        _assert_triple_catch(LearnedPolicyAdapterContractError, ValueError)

    def test_risk_surface_unavailable(self) -> None:
        from robot_sf.planner.learned_risk_surface import RiskSurfaceUnavailable
        _assert_triple_catch(RiskSurfaceUnavailable, ValueError)

    def test_lidar_occupancy_adapter_error(self) -> None:
        from robot_sf.planner.lidar_occupancy import LidarOccupancyAdapterError
        _assert_triple_catch(LidarOccupancyAdapterError, ValueError)

    def test_obstacle_feature_schema_error(self) -> None:
        from robot_sf.planner.obstacle_features import ObstacleFeatureSchemaError
        _assert_triple_catch(ObstacleFeatureSchemaError, ValueError)

    def test_planning_error_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.planner.classic_global_planner import PlanningError
        with pytest.raises(RobotSfError):
            raise PlanningError("no path found")

    def test_risk_surface_unavailable_legacy_catch(self) -> None:
        from robot_sf.planner.learned_risk_surface import RiskSurfaceUnavailable
        with pytest.raises(ValueError):
            raise RiskSurfaceUnavailable("surface missing")


# ---------------------------------------------------------------------------
# research (individual files)
# ---------------------------------------------------------------------------


class TestResearchIndividualFileExceptions:
    def test_amv_trace_manifest_error(self) -> None:
        from robot_sf.research.amv_command_response_trace_manifest import AmvTraceManifestError
        _assert_triple_catch(AmvTraceManifestError, ValueError)

    def test_scenario_prior_staging_contract_error(self) -> None:
        from robot_sf.research.scenario_prior_staging_contract import (
            ScenarioPriorStagingContractError,
        )
        _assert_triple_catch(ScenarioPriorStagingContractError, ValueError)

    def test_research_individual_errors_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.research.amv_command_response_trace_manifest import AmvTraceManifestError
        with pytest.raises(RobotSfError):
            raise AmvTraceManifestError("trace manifest invalid")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


class TestTrainingExceptions:
    def test_learned_risk_campaign_readiness_error(self) -> None:
        from robot_sf.training.learned_risk_campaign_readiness import (
            LearnedRiskCampaignReadinessError,
        )
        _assert_triple_catch(LearnedRiskCampaignReadinessError, ValueError)

    def test_learned_risk_launch_packet_error(self) -> None:
        from robot_sf.training.learned_risk_launch_packet import LearnedRiskLaunchPacketError
        _assert_triple_catch(LearnedRiskLaunchPacketError, ValueError)

    def test_learned_risk_trace_manifest_error(self) -> None:
        from robot_sf.training.learned_risk_trace_manifest import LearnedRiskTraceManifestError
        _assert_triple_catch(LearnedRiskTraceManifestError, ValueError)

    def test_launch_packet_error(self) -> None:
        from robot_sf.training.oracle_imitation_launch_packet import LaunchPacketError
        _assert_triple_catch(LaunchPacketError, ValueError)

    def test_warm_start_readiness_error_hierarchy(self) -> None:
        from robot_sf.training.oracle_imitation_warm_start_readiness import (
            PrerequisitesNotReadyError,
            WarmStartReadinessError,
        )
        assert issubclass(WarmStartReadinessError, RobotSfError)
        assert issubclass(WarmStartReadinessError, ValueError)
        # Subclass is covered transitively
        assert issubclass(PrerequisitesNotReadyError, RobotSfError)
        assert issubclass(PrerequisitesNotReadyError, ValueError)
        assert issubclass(PrerequisitesNotReadyError, WarmStartReadinessError)

    def test_prerequisites_not_ready_caught_by_warm_start_readiness(self) -> None:
        from robot_sf.training.oracle_imitation_warm_start_readiness import (
            PrerequisitesNotReadyError,
            WarmStartReadinessError,
        )
        with pytest.raises(WarmStartReadinessError):
            raise PrerequisitesNotReadyError("prerequisites missing")

    def test_oracle_trace_uri_registry_error(self) -> None:
        from robot_sf.training.oracle_trace_uri_registry import OracleTraceUriRegistryError
        _assert_triple_catch(OracleTraceUriRegistryError, ValueError)

    def test_predictive_retraining_readiness_error(self) -> None:
        from robot_sf.training.predictive_retraining_readiness import (
            PredictiveRetrainingReadinessError,
        )
        _assert_triple_catch(PredictiveRetrainingReadinessError, ValueError)

    def test_shielded_ppo_launch_packet_error(self) -> None:
        from robot_sf.training.shielded_ppo_launch_packet import ShieldedPPOLaunchPacketError
        _assert_triple_catch(ShieldedPPOLaunchPacketError, ValueError)

    def test_training_errors_catchable_by_robot_sf_error(self) -> None:
        from robot_sf.training.learned_risk_trainer import LearnedRiskTrainerError
        with pytest.raises(RobotSfError):
            raise LearnedRiskTrainerError("training failed")

    def test_training_errors_legacy_catch_preserved(self) -> None:
        from robot_sf.training.orca_residual_lineage_packet import OrcaResidualLineagePacketError
        with pytest.raises(ValueError):
            raise OrcaResidualLineagePacketError("lineage error")
