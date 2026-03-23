"""Tests for benchmark algorithm metadata enrichment contracts."""

from __future__ import annotations

from robot_sf.benchmark.algorithm_metadata import (
    canonical_algorithm_name,
    enrich_algorithm_metadata,
    infer_execution_mode_from_counts,
)


def test_canonical_algorithm_name_resolves_aliases() -> None:
    """Alias names should resolve to canonical benchmark algorithm names."""
    assert canonical_algorithm_name("simple_policy") == "goal"
    assert canonical_algorithm_name("sf") == "social_force"
    assert canonical_algorithm_name("unknown_algo") == "unknown_algo"


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


def test_infer_execution_mode_from_counts() -> None:
    """Execution mode inference should reflect observed native/adapted step counts."""
    assert infer_execution_mode_from_counts(native_steps=3, adapted_steps=0) == "native"
    assert infer_execution_mode_from_counts(native_steps=0, adapted_steps=3) == "adapter"
    assert infer_execution_mode_from_counts(native_steps=3, adapted_steps=2) == "mixed"
    assert infer_execution_mode_from_counts(native_steps=0, adapted_steps=0) == "unknown"
