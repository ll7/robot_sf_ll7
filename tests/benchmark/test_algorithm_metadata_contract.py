"""Tests for benchmark algorithm metadata enrichment contracts."""

from __future__ import annotations

import pytest

from robot_sf.benchmark import algorithm_metadata
from robot_sf.benchmark.algorithm_metadata import (
    canonical_algorithm_name,
    enrich_algorithm_metadata,
    infer_execution_mode_from_counts,
    observation_spec_for_algorithm,
    planner_contract_for_algorithm,
    resolve_observation_mode,
)


def test_canonical_algorithm_name_resolves_aliases() -> None:
    """Alias names should resolve to canonical benchmark algorithm names."""
    assert canonical_algorithm_name("simple_policy") == "goal"
    assert canonical_algorithm_name("sf") == "social_force"
    assert canonical_algorithm_name("prediction_mpc_cbf") == "prediction_mpc"
    assert canonical_algorithm_name("unknown_algo") == "unknown_algo"


def test_observation_spec_declares_supported_modes_and_rejects_invalid_override() -> None:
    """Planner observation contracts should be inspectable and fail closed."""
    goal_spec = observation_spec_for_algorithm("goal")
    assert goal_spec["default_mode"] == "goal_state"
    assert goal_spec["supported_modes"] == ["goal_state", "socnav_state"]
    assert resolve_observation_mode("goal", "socnav_state") == "socnav_state"

    meta = enrich_algorithm_metadata(algo="goal", observation_mode="socnav_state")
    assert meta["observation_spec"]["active_mode"] == "socnav_state"
    assert meta["observation_spec"]["override_applied"] is True
    assert resolve_observation_mode("guarded_ppo", "socnav_state") == "socnav_state"

    with pytest.raises(ValueError) as excinfo:
        resolve_observation_mode("orca", "goal_state")
    assert "Observation mode 'goal_state' is not supported by algorithm 'orca'" in str(
        excinfo.value
    )
    assert "socnav_state" in str(excinfo.value)


@pytest.mark.parametrize("obs_mode", ["dict", "native_dict", "multi_input"])
def test_learned_checkpoint_contract_derives_ppo_dict_family_from_registry_metadata(
    obs_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dict-family PPO checkpoints should derive the active producer from checkpoint metadata."""

    def _registry_entry(_model_id: str) -> dict[str, object]:
        return {
            "benchmark_promotion": {
                "observation_level": "tracked_agents_no_noise",
                "observation_mode": "dict",
            }
        }

    monkeypatch.setattr(algorithm_metadata, "get_registry_entry", _registry_entry, raising=False)

    contract = algorithm_metadata.resolve_learned_checkpoint_observation_contract(
        "ppo",
        {"model_id": "ppo-grid", "obs_mode": obs_mode},
    )

    assert contract["active_observation_mode"] == "socnav_state"
    assert contract["status"] == "metadata_resolved"
    assert contract["metadata_source"] == "model_registry.benchmark_promotion"
    assert contract["planner_observation_mode"] == obs_mode


def test_learned_checkpoint_contract_explicit_observation_mode_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit observation-mode overrides should not require checkpoint metadata."""

    def _unexpected_registry_lookup(_model_id: str) -> dict[str, object]:
        raise AssertionError("explicit observation overrides must bypass metadata lookup")

    monkeypatch.setattr(
        algorithm_metadata,
        "get_registry_entry",
        _unexpected_registry_lookup,
        raising=False,
    )

    contract = algorithm_metadata.resolve_learned_checkpoint_observation_contract(
        "ppo",
        {"model_id": "metadata-not-needed", "obs_mode": "dict"},
        observation_mode="sensor_fusion_state",
    )

    assert contract["active_observation_mode"] == "sensor_fusion_state"
    assert contract["status"] == "explicit_override"
    assert contract["metadata_source"] == "explicit_observation_mode"


def test_learned_checkpoint_contract_explicit_observation_level_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit observation-level overrides should select their compatible producer."""

    def _unexpected_registry_lookup(_model_id: str) -> dict[str, object]:
        raise AssertionError("explicit observation overrides must bypass metadata lookup")

    monkeypatch.setattr(
        algorithm_metadata,
        "get_registry_entry",
        _unexpected_registry_lookup,
        raising=False,
    )

    contract = algorithm_metadata.resolve_learned_checkpoint_observation_contract(
        "ppo",
        {"model_id": "metadata-not-needed", "obs_mode": "dict"},
        observation_level="lidar_2d",
    )

    assert contract["active_observation_mode"] == "sensor_fusion_state"
    assert contract["status"] == "explicit_override"
    assert contract["metadata_source"] == "explicit_observation_level"


def test_learned_checkpoint_contract_rejects_missing_dict_family_metadata() -> None:
    """Dict-family learned checkpoints should fail closed when metadata is absent."""
    with pytest.raises(ValueError, match="requires learned checkpoint observation metadata"):
        algorithm_metadata.resolve_learned_checkpoint_observation_contract(
            "ppo",
            {"obs_mode": "dict"},
        )


def test_learned_checkpoint_contract_rejects_malformed_metadata() -> None:
    """Malformed checkpoint observation metadata should fail with a classified error."""
    with pytest.raises(ValueError, match="malformed learned checkpoint observation metadata"):
        algorithm_metadata.resolve_learned_checkpoint_observation_contract(
            "ppo",
            {"obs_mode": "dict", "observation_contract": "not-a-mapping"},
        )


def test_learned_checkpoint_contract_rejects_incompatible_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata that resolves to an unsupported producer should fail before execution."""

    def _registry_entry(_model_id: str) -> dict[str, object]:
        return {
            "benchmark_promotion": {
                "observation_level": "tracked_agents_no_noise",
                "active_observation_mode": "goal_state",
                "observation_mode": "dict",
            }
        }

    monkeypatch.setattr(algorithm_metadata, "get_registry_entry", _registry_entry, raising=False)

    with pytest.raises(ValueError, match="incompatible learned checkpoint observation metadata"):
        algorithm_metadata.resolve_learned_checkpoint_observation_contract(
            "ppo",
            {"model_id": "ppo-bad", "obs_mode": "dict"},
        )


def test_enriched_metadata_embeds_typed_planner_contract_payload() -> None:
    """Algorithm metadata should expose normalized observation/action contract payloads."""
    meta = enrich_algorithm_metadata(
        algo="orca",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
        observation_mode="socnav_state",
    )

    contract = meta["planner_contract"]
    assert contract["planner_id"] == "orca"
    assert contract["observation_contract"]["active_mode"] == "socnav_state"
    assert contract["action_contract"]["command_space"] == "unicycle_vw"
    assert contract["action_contract"]["output_keys"] == ["v", "omega"]
    assert contract["action_contract"]["active_robot_kinematics"] == "differential_drive"
    assert contract["action_contract"]["compatible_robot_kinematics"] == ["differential_drive"]
    assert contract["compatibility_scope"] == "metadata_only"


def test_planner_contract_helper_rejects_unsupported_observation_mode() -> None:
    """Typed contract resolution should fail closed on invalid observation overrides."""
    with pytest.raises(ValueError, match="Observation mode 'goal_state' is not supported"):
        planner_contract_for_algorithm("orca", observation_mode="goal_state")


def test_planner_contract_helper_rejects_unknown_command_space() -> None:
    """Typed contract resolution should fail closed on unknown action contracts."""
    with pytest.raises(ValueError, match="Unsupported planner command space"):
        planner_contract_for_algorithm("unknown_algo")


def test_random_baseline_metadata_marks_stochastic_reference() -> None:
    """Random baseline metadata should expose diagnostic stochastic semantics."""
    meta = enrich_algorithm_metadata(algo="random", metadata={"status": "ok"})
    assert meta["baseline_category"] == "diagnostic"
    assert meta["policy_semantics"] == "stochastic_uniform_action_reference"
    assert meta["stochastic_reference"] is True
    assert meta["distinct_from_goal_baseline"] is True


def test_planner_kinematics_and_adapter_impact_fields() -> None:
    """PPO metadata should include mixed command compatibility and impact scaffold."""
    meta = enrich_algorithm_metadata(
        algo="ppo",
        metadata={"status": "ok"},
        robot_kinematics="differential_drive",
        adapter_impact_requested=True,
    )
    planner = meta["planner_kinematics"]
    impact = meta["adapter_impact"]
    assert planner["planner_command_space"] == "mixed_vw_or_vxy"
    assert planner["supports_native_commands"] is True
    assert planner["supports_adapter_commands"] is True
    assert planner["robot_kinematics"] == "differential_drive"
    assert impact["requested"] is True
    assert impact["native_steps"] == 0
    assert impact["adapted_steps"] == 0


def test_sac_metadata_declares_native_unicycle_contract() -> None:
    """SAC map-runner policy metadata should not fall back to unknown actions."""
    meta = enrich_algorithm_metadata(algo="sac", metadata={"status": "ok"})
    planner = meta["planner_kinematics"]
    action = meta["planner_contract"]["action_contract"]

    assert planner["planner_command_space"] == "unicycle_vw"
    assert planner["supports_native_commands"] is True
    assert planner["supports_adapter_commands"] is False
    assert action["command_space"] == "unicycle_vw"
    assert action["output_keys"] == ["v", "omega"]


def test_safety_barrier_metadata_marks_testing_only_native_spike() -> None:
    """Safety-barrier metadata should expose the testing-only adapter contract."""
    meta = enrich_algorithm_metadata(
        algo="safety_barrier",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    assert meta["baseline_category"] == "classical"
    assert meta["policy_semantics"] == "native_barrier_style_safety_filter"
    assert planner["planner_command_space"] == "unicycle_vw"
    assert planner["adapter_name"] == "SafetyBarrierPlannerAdapter"


def test_grid_route_metadata_marks_testing_only_route_spike() -> None:
    """Grid-route metadata should expose the testing-only adapter contract."""
    meta = enrich_algorithm_metadata(
        algo="grid_route",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    assert meta["baseline_category"] == "classical"
    assert meta["policy_semantics"] == "occupancy_grid_route_tracking"
    assert planner["planner_command_space"] == "unicycle_vw"
    assert planner["adapter_name"] == "GridRoutePlannerAdapter"


def test_risk_surface_dwa_metadata_marks_prototype_surface_adapter() -> None:
    """Risk-surface DWA metadata should prevent learned-risk overclaiming."""
    assert canonical_algorithm_name("risk_surface_dwa_v0") == "risk_surface_dwa"
    meta = enrich_algorithm_metadata(
        algo="risk_surface_dwa_v0",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    assert meta["canonical_algorithm"] == "risk_surface_dwa"
    assert meta["baseline_category"] == "classical"
    assert meta["policy_semantics"] == "deterministic_local_risk_surface_dwa"
    assert meta["observation_spec"]["inputs"] == [
        "robot_state",
        "goal",
        "pedestrians",
        "local_risk_surface",
    ]
    assert planner["adapter_name"] == "RiskSurfacePlannerAdapter"
    assert planner["testing_only_adapter"] is True
    assert planner["prototype_only"] is True


def test_safety_barrier_accepts_lidar_level_through_sensor_fusion_contract() -> None:
    """Safety-barrier LiDAR compatibility should be explicit adapter metadata, not fallback."""
    meta = enrich_algorithm_metadata(
        algo="safety_barrier",
        metadata={"status": "ok"},
        execution_mode="adapter",
        adapter_name="LidarOccupancySafetyBarrierAdapter",
        observation_level="lidar_2d",
        robot_kinematics="differential_drive",
    )

    assert meta["observation_level"]["key"] == "lidar_2d"
    assert meta["observation_spec"]["active_mode"] == "sensor_fusion_state"
    assert meta["planner_kinematics"]["adapter_name"] == "LidarOccupancySafetyBarrierAdapter"
    assert meta["planner_contract"]["observation_contract"]["observation_level"] == "lidar_2d"


def test_trivial_reference_metadata_marks_diagnostic_template() -> None:
    """Reference adapter metadata should prevent benchmark-evidence overclaiming."""
    meta = enrich_algorithm_metadata(
        algo="reference_adapter",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    assert meta["canonical_algorithm"] == "trivial_reference"
    assert meta["baseline_category"] == "diagnostic"
    assert meta["policy_semantics"] == "diagnostic_adapter_template"
    assert planner["planner_command_space"] == "unicycle_vw"
    assert planner["adapter_name"] == "TrivialReferencePlannerAdapter"
    assert planner["diagnostic_reference_only"] is True


def test_orca_metadata_exposes_upstream_reference_and_projection_contract() -> None:
    """ORCA metadata should make the upstream source and projection policy explicit."""
    meta = enrich_algorithm_metadata(
        algo="orca",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert planner["upstream_command_space"] == "velocity_vector_xy"
    assert planner["benchmark_command_space"] == "unicycle_vw"
    assert planner["projection_policy"] == "heading_safe_velocity_to_unicycle_vw"
    assert upstream["repo_url"] == "https://github.com/mit-acl/Python-RVO2"
    assert upstream["vendored_path"] == "third_party/python-rvo2"


def test_hrvo_metadata_exposes_local_provenance_boundary() -> None:
    """HRVO metadata should describe the local implementation and its references honestly."""
    meta = enrich_algorithm_metadata(
        algo="hrvo",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert meta["baseline_category"] == "classical"
    assert meta["policy_semantics"] == "hybrid_reciprocal_velocity_obstacle"
    assert planner["upstream_command_space"] == "velocity_vector_xy"
    assert planner["projection_policy"] == "heading_safe_velocity_to_unicycle_vw"
    assert upstream["repo_url"] == "https://github.com/snape/HRVO"
    assert upstream["provenance_note"] == (
        "Local implementation informed by upstream references; not a wrapped upstream runtime."
    )


def test_socnav_orca_variant_metadata_is_registered_and_experimental() -> None:
    """SocNav ORCA variants should expose classical benchmark semantics and explicit opt-in readiness."""
    for algo in (
        "socnav_orca_nonholonomic",
        "socnav_orca_dd",
        "socnav_orca_relaxed",
        "socnav_hrvo",
    ):
        meta = enrich_algorithm_metadata(
            algo=algo,
            metadata={"status": "ok"},
            execution_mode="adapter",
            robot_kinematics="differential_drive",
        )
        assert meta["baseline_category"] == "classical"
        assert meta["planner_kinematics"]["planner_command_space"] == "unicycle_vw"
        assert meta["planner_kinematics"]["supports_adapter_commands"] is True


def test_drl_vo_metadata_exposes_reference_contract() -> None:
    """DRL-VO metadata should expose hybrid planner reference contract."""
    meta = enrich_algorithm_metadata(
        algo="drl_vo",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    assert meta["baseline_category"] == "learning"
    assert meta["policy_semantics"] == "hybrid_deep_reinforcement_velocity_obstacle"
    assert meta["planner_kinematics"]["upstream_command_space"] == "velocity_vector_xy"
    assert meta["planner_kinematics"]["projection_policy"] == "heading_safe_velocity_to_unicycle_vw"
    upstream = meta["upstream_reference"]
    assert upstream["repo_url"] == "https://github.com/TempleRAIL/drl_vo_nav"
    assert upstream["commit"] == "6d734b6e0df77fd4c4faa4649ca0fcb3e69cf835"


def test_sicnav_metadata_exposes_pinned_external_repo_contract() -> None:
    """SICNav metadata should point at the pinned external-repo staging contract."""
    meta = enrich_algorithm_metadata(
        algo="sicnav",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )

    upstream = meta["upstream_reference"]
    assert upstream["repo_url"] == "https://github.com/sepsamavi/safe-interactive-crowdnav"
    assert upstream["commit"] == "c702fb8ac9ba6439ca61da7dde68b8524bbc6a1f"
    assert upstream["checkout_path"] == "third_party/external_repos/sicnav"


def test_social_navigation_pyenvs_orca_metadata_exposes_upstream_wrapper_contract() -> None:
    """Prototype external ORCA metadata should expose upstream repo and projection boundary."""
    meta = enrich_algorithm_metadata(
        algo="social_navigation_pyenvs_orca",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert meta["baseline_category"] == "classical"
    assert planner["upstream_command_space"] == "velocity_vector_xy"
    assert planner["benchmark_command_space"] == "unicycle_vw"
    assert planner["projection_policy"] == "heading_safe_velocity_to_unicycle_vw"
    assert upstream["repo_url"] == "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs"
    assert upstream["upstream_policy"] == "crowd_nav.policy_no_train.orca.ORCA"


def test_social_navigation_pyenvs_force_model_metadata_exposes_upstream_wrapper_contract() -> None:
    """Prototype external force-model metadata should expose upstream repo and policy path."""
    socialforce = enrich_algorithm_metadata(
        algo="social_navigation_pyenvs_socialforce",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    sfm = enrich_algorithm_metadata(
        algo="social_navigation_pyenvs_sfm_helbing",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    assert (
        socialforce["planner_kinematics"]["projection_policy"]
        == "heading_safe_velocity_to_unicycle_vw"
    )
    assert socialforce["planner_kinematics"]["runtime_strategy"] == (
        "crowdnav_socialforce_compat_shim"
    )
    assert socialforce["planner_kinematics"]["runtime_dependency"] == "socialforce==0.2.3"
    assert socialforce["upstream_reference"]["upstream_policy"] == (
        "crowd_nav.policy_no_train.socialforce.SocialForce"
    )
    assert socialforce["upstream_reference"]["runtime_dependency"] == "socialforce==0.2.3"
    assert socialforce["upstream_reference"]["runtime_strategy"] == (
        "crowdnav_socialforce_compat_shim"
    )
    assert sfm["upstream_reference"]["upstream_policy"] == (
        "crowd_nav.policy_no_train.sfm_helbing.SFMHelbing"
    )


def test_social_navigation_pyenvs_hsfm_metadata_exposes_headed_wrapper_contract() -> None:
    """Prototype external HSFM metadata should expose headed upstream command semantics."""
    meta = enrich_algorithm_metadata(
        algo="social_navigation_pyenvs_hsfm_new_guo",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert planner["upstream_command_space"] == "body_velocity_xy_plus_omega"
    assert planner["projection_policy"] == "body_velocity_heading_safe_to_unicycle_vw"
    assert upstream["upstream_policy"] == "crowd_nav.policy_no_train.hsfm_new_guo.HSFMNewGuo"


def test_crowdnav_height_metadata_exposes_checkpoint_wrapper_contract() -> None:
    """CrowdNav_HEIGHT metadata should expose the upstream repo and checkpoint boundary."""
    meta = enrich_algorithm_metadata(
        algo="crowdnav_height",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert meta["baseline_category"] == "learning"
    assert meta["policy_semantics"] == "upstream_crowdnav_height_checkpoint_wrapper"
    assert planner["upstream_command_space"] == "discrete_delta_v_and_delta_theta"
    assert planner["benchmark_command_space"] == "unicycle_vw"
    assert planner["projection_policy"] == "upstream_discrete_delta_vw_to_unicycle_vw_stateful"
    assert upstream["repo_url"] == "https://github.com/Shuijing725/CrowdNav_HEIGHT"
    assert upstream["default_checkpoint"] == "HEIGHT/checkpoints/237800.pt"


def test_sonic_crowdnav_metadata_exposes_checkpoint_wrapper_contract() -> None:
    """SoNIC metadata should expose model-only wrapper boundaries and projection policy."""
    meta = enrich_algorithm_metadata(
        algo="sonic_gst",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert meta["baseline_category"] == "learning"
    assert meta["policy_semantics"] == "upstream_sonic_checkpoint_wrapper"
    assert planner["upstream_command_space"] == "holonomic_velocity_xy"
    assert planner["benchmark_command_space"] == "unicycle_vw"
    assert planner["projection_policy"] == "heading_safe_velocity_to_unicycle_vw"
    assert upstream["repo_url"] == "https://github.com/tasl-lab/SoNIC-Social-Nav"
    assert upstream["default_model_name"] == "SoNIC_GST"


def test_gensafenav_ours_metadata_exposes_checkpoint_wrapper_contract() -> None:
    """GenSafeNav Ours_GST metadata should expose the upstream checkpoint boundary."""
    meta = enrich_algorithm_metadata(
        algo="ours_gst",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert meta["baseline_category"] == "learning"
    assert meta["policy_semantics"] == "upstream_gensafenav_checkpoint_wrapper"
    assert planner["upstream_command_space"] == "holonomic_velocity_xy"
    assert planner["benchmark_command_space"] == "unicycle_vw"
    assert planner["projection_policy"] == "heading_safe_velocity_to_unicycle_vw"
    assert upstream["repo_url"] == "https://github.com/tasl-lab/GenSafeNav"
    assert upstream["default_model_name"] == "Ours_GST"


def test_gensafenav_gst_predictor_rand_metadata_exposes_checkpoint_wrapper_contract() -> None:
    """GenSafeNav CrowdNav++-style metadata should expose the upstream checkpoint boundary."""
    meta = enrich_algorithm_metadata(
        algo="gst_predictor_rand",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert meta["baseline_category"] == "learning"
    assert meta["policy_semantics"] == "upstream_gensafenav_checkpoint_wrapper"
    assert planner["upstream_command_space"] == "holonomic_velocity_xy"
    assert planner["benchmark_command_space"] == "unicycle_vw"
    assert planner["projection_policy"] == "heading_safe_velocity_to_unicycle_vw"
    assert upstream["repo_url"] == "https://github.com/tasl-lab/GenSafeNav"
    assert upstream["default_model_name"] == "GST_predictor_rand"


def test_gensafenav_ours_guarded_metadata_exposes_mixed_guarded_contract() -> None:
    """Guarded Ours_GST metadata should expose mixed execution and fallback boundary."""
    meta = enrich_algorithm_metadata(
        algo="ours_gst_guarded",
        metadata={"status": "ok"},
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert meta["baseline_category"] == "learning"
    assert meta["policy_semantics"] == "guarded_upstream_gensafenav_checkpoint_wrapper"
    assert planner["supports_native_commands"] is True
    assert planner["supports_adapter_commands"] is True
    assert planner["execution_mode"] == "mixed"
    assert upstream["default_model_name"] == "Ours_GST"


def test_gensafenav_gst_predictor_rand_guarded_metadata_exposes_mixed_guarded_contract() -> None:
    """Guarded GST_predictor_rand metadata should expose mixed execution and fallback boundary."""
    meta = enrich_algorithm_metadata(
        algo="gst_predictor_rand_guarded",
        metadata={"status": "ok"},
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    upstream = meta["upstream_reference"]
    assert meta["baseline_category"] == "learning"
    assert meta["policy_semantics"] == "guarded_upstream_gensafenav_checkpoint_wrapper"
    assert planner["supports_native_commands"] is True
    assert planner["supports_adapter_commands"] is True
    assert planner["execution_mode"] == "mixed"
    assert upstream["default_model_name"] == "GST_predictor_rand"


def test_guarded_gensafenav_metadata_execution_mode_override_still_applies() -> None:
    """Guarded wrappers should still honor an explicit execution-mode override."""
    meta = enrich_algorithm_metadata(
        algo="ours_gst_guarded",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )

    assert meta["planner_kinematics"]["execution_mode"] == "adapter"


def test_nmpc_social_metadata_exposes_native_optimizer_contract() -> None:
    """NMPC metadata should classify the planner as a native optimizer-style adapter."""
    meta = enrich_algorithm_metadata(
        algo="nmpc_social",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    planner = meta["planner_kinematics"]
    assert meta["baseline_category"] == "classical"
    assert meta["policy_semantics"] == "nonlinear_model_predictive_local_planner"
    assert planner["planner_command_space"] == "unicycle_vw"
    assert planner["adapter_name"] == "NMPCSocialPlannerAdapter"


def test_prediction_mpc_metadata_exposes_constraint_mpc_contract() -> None:
    """Prediction-aware MPC metadata should name its time-varying constraint boundary."""
    meta = enrich_algorithm_metadata(
        algo="prediction_mpc",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )

    planner = meta["planner_kinematics"]
    assert meta["baseline_category"] == "classical"
    assert meta["policy_semantics"] == "prediction_aware_model_predictive_local_planner"
    assert planner["planner_command_space"] == "unicycle_vw"
    assert planner["adapter_name"] == "PredictionMPCPlannerAdapter"
    assert planner["projection_policy"] == "constant_velocity_time_varying_pedestrian_constraints"
    assert meta["observation_spec"]["inputs"] == [
        "robot_state",
        "goal",
        "pedestrians",
        "predicted_pedestrian_futures",
    ]


def test_prediction_mpc_cbf_alias_reuses_supported_mpc_command_contract() -> None:
    """Trace rosters may name the CBF arm without losing MPC command metadata."""
    meta = enrich_algorithm_metadata(
        algo="prediction_mpc_cbf",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )

    planner = meta["planner_kinematics"]
    action = meta["planner_contract"]["action_contract"]
    assert meta["algorithm"] == "prediction_mpc_cbf"
    assert meta["canonical_algorithm"] == "prediction_mpc"
    assert planner["planner_command_space"] == "unicycle_vw"
    assert planner["adapter_name"] == "PredictionMPCPlannerAdapter"
    assert action["command_space"] == "unicycle_vw"
    assert action["output_keys"] == ["v", "omega"]


def test_infer_execution_mode_from_counts() -> None:
    """Execution mode inference should reflect observed native/adapted step counts."""
    assert infer_execution_mode_from_counts(native_steps=3, adapted_steps=0) == "native"
    assert infer_execution_mode_from_counts(native_steps=0, adapted_steps=3) == "adapter"
    assert infer_execution_mode_from_counts(native_steps=3, adapted_steps=2) == "mixed"
    assert infer_execution_mode_from_counts(native_steps=0, adapted_steps=0) == "unknown"
