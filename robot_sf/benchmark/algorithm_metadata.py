"""Shared algorithm metadata helpers for benchmark episode outputs.

This module centralizes baseline category labels, policy semantics, and
planner-kinematics metadata so benchmark writers emit a consistent contract.
"""

from __future__ import annotations

from typing import Any

from robot_sf.baselines.interface import ActionContract, ObservationContract, PlannerMetadata
from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.observation_levels import (
    OBSERVATION_LEVEL_KEYS,
    observation_level_for_mode,
    resolve_observation_level_contract,
)
from robot_sf.models.registry import get_registry_entry

_BASELINE_CATEGORY_BY_CANONICAL: dict[str, str] = {
    "goal": "classical",
    "social_force": "classical",
    "orca": "classical",
    "hrvo": "classical",
    "social_navigation_pyenvs_orca": "classical",
    "social_navigation_pyenvs_socialforce": "classical",
    "social_navigation_pyenvs_sfm_helbing": "classical",
    "social_navigation_pyenvs_hsfm_new_guo": "classical",
    "socnav_orca_nonholonomic": "classical",
    "socnav_orca_dd": "classical",
    "socnav_orca_relaxed": "classical",
    "socnav_hrvo": "classical",
    "crowdnav_height": "learning",
    "sonic_crowdnav": "learning",
    "gensafenav_ours_gst": "learning",
    "gensafenav_ours_gst_guarded": "learning",
    "gensafenav_gst_predictor_rand": "learning",
    "gensafenav_gst_predictor_rand_guarded": "learning",
    "ppo": "learning",
    "sac": "learning",
    "hybrid_global_rl": "diagnostic",
    "guarded_ppo": "learning",
    "socnav_sampling": "classical",
    "sacadrl": "learning",
    "sicnav": "classical",
    "dr_mpc": "learning",
    "prediction_planner": "learning",
    "predictive_mppi": "learning",
    "risk_dwa": "classical",
    "risk_surface_dwa": "classical",
    "hybrid_rule_local_planner": "classical",
    "adaptive_proxemic_selector_v0": "diagnostic",
    "adaptive_proxemic_selector_v1": "diagnostic",
    "safety_barrier": "classical",
    "grid_route": "classical",
    "topology_guided_hybrid_rule_v0": "diagnostic",
    "lidar_social_force": "classical",
    "lidar_grid_route": "classical",
    "trivial_reference": "diagnostic",
    "mppi_social": "classical",
    "hybrid_portfolio": "classical",
    "planner_selector_v2_diagnostic": "diagnostic",
    "policy_stack_v1": "classical",
    "hybrid_orca_sampler": "classical",
    "stream_gap": "classical",
    "gap_prediction": "classical",
    "socnav_bench": "classical",
    "drl_vo": "learning",
    "random": "diagnostic",
    "fast_pysf_planner": "diagnostic",
    "rvo": "classical",
    "dwa": "classical",
    "teb": "classical",
    "nmpc_social": "classical",
    "prediction_mpc": "classical",
}

_POLICY_SEMANTICS_BY_CANONICAL: dict[str, str] = {
    "goal": "deterministic_goal_seeking",
    "social_force": "social_force_adapter",
    "orca": "orca_adapter",
    "socnav_orca_nonholonomic": "orca_adapter",
    "socnav_orca_dd": "orca_adapter",
    "socnav_orca_relaxed": "orca_adapter",
    "hrvo": "hybrid_reciprocal_velocity_obstacle",
    "socnav_hrvo": "hybrid_reciprocal_velocity_obstacle",
    "social_navigation_pyenvs_orca": "upstream_social_navigation_pyenvs_orca_wrapper",
    "social_navigation_pyenvs_socialforce": "upstream_social_navigation_pyenvs_socialforce_wrapper",
    "social_navigation_pyenvs_sfm_helbing": "upstream_social_navigation_pyenvs_sfm_helbing_wrapper",
    "social_navigation_pyenvs_hsfm_new_guo": "upstream_social_navigation_pyenvs_hsfm_wrapper",
    "crowdnav_height": "upstream_crowdnav_height_checkpoint_wrapper",
    "sonic_crowdnav": "upstream_sonic_checkpoint_wrapper",
    "gensafenav_ours_gst": "upstream_gensafenav_checkpoint_wrapper",
    "gensafenav_ours_gst_guarded": "guarded_upstream_gensafenav_checkpoint_wrapper",
    "gensafenav_gst_predictor_rand": "upstream_gensafenav_checkpoint_wrapper",
    "gensafenav_gst_predictor_rand_guarded": "guarded_upstream_gensafenav_checkpoint_wrapper",
    "sicnav": "upstream_sicnav_checkpoint_or_policy_wrapper",
    "dr_mpc": "upstream_dr_mpc_residual_mpc_wrapper",
    "ppo": "policy_network_inference",
    "sac": "policy_network_inference",
    "hybrid_global_rl": "route_conditioned_policy_network_inference",
    "guarded_ppo": "guarded_policy_network_inference",
    "socnav_sampling": "heuristic_sampling_adapter",
    "sacadrl": "learned_value_adapter",
    "prediction_planner": "predictive_model_based_adapter",
    "predictive_mppi": "predictive_sequence_optimizer",
    "risk_dwa": "risk_aware_dynamic_window",
    "risk_surface_dwa": "deterministic_local_risk_surface_dwa",
    "hybrid_rule_local_planner": "hybrid_rule_deterministic_local_planner",
    "adaptive_proxemic_selector_v0": "diagnostic_fixed_proxemic_profile_selector",
    "adaptive_proxemic_selector_v1": "diagnostic_neutral_default_proxemic_profile_selector",
    "safety_barrier": "native_barrier_style_safety_filter",
    "grid_route": "occupancy_grid_route_tracking",
    "topology_guided_hybrid_rule_v0": "diagnostic_topology_hypothesis_guided_hybrid_rule",
    "lidar_social_force": "lidar_endpoint_tracked_social_force_adapter",
    "lidar_grid_route": "lidar_ego_occupancy_grid_route_tracking",
    "trivial_reference": "diagnostic_adapter_template",
    "mppi_social": "sampled_sequence_optimizer",
    "hybrid_portfolio": "risk_regime_hybrid_switcher",
    "planner_selector_v2_diagnostic": "deterministic_diagnostic_planner_selector",
    "policy_stack_v1": "policy_stack_v1_portfolio",
    "hybrid_orca_sampler": "orca_primary_with_sampled_progress_repair",
    "stream_gap": "gap_acceptance_local_planner",
    "gap_prediction": "predictive_planner_with_gap_veto",
    "socnav_bench": "socnav_adapter",
    "drl_vo": "hybrid_deep_reinforcement_velocity_obstacle",
    "random": "stochastic_uniform_action_reference",
    "fast_pysf_planner": "social_force_reference",
    "rvo": "placeholder_adapter",
    "dwa": "placeholder_adapter",
    "teb": "corridor_commitment_local_planner",
    "nmpc_social": "nonlinear_model_predictive_local_planner",
    "prediction_mpc": "prediction_aware_model_predictive_local_planner",
}

_DEFAULT_OBSERVATION_SPEC: dict[str, Any] = {
    "default_mode": "socnav_state",
    "supported_modes": ("socnav_state",),
    "inputs": ("robot_state", "goal", "pedestrians"),
    "notes": (
        "Structured Robot SF social-navigation state: robot pose/velocity, route goal, "
        "and pedestrian state when present."
    ),
}

_DICT_STYLE_PLANNER_OBSERVATION_MODES = {"dict", "native_dict", "multi_input"}

_OBSERVATION_SPEC_BY_CANONICAL: dict[str, dict[str, Any]] = {
    "goal": {
        "default_mode": "goal_state",
        "supported_modes": ("goal_state", "socnav_state"),
        "inputs": ("robot_state", "goal"),
        "notes": (
            "Goal baseline consumes robot and goal state only; it can run under socnav_state "
            "as a parity control because extra pedestrian channels are ignored."
        ),
    },
    "social_force": _DEFAULT_OBSERVATION_SPEC,
    "orca": _DEFAULT_OBSERVATION_SPEC,
    "hrvo": _DEFAULT_OBSERVATION_SPEC,
    "socnav_orca_nonholonomic": _DEFAULT_OBSERVATION_SPEC,
    "socnav_orca_dd": _DEFAULT_OBSERVATION_SPEC,
    "socnav_orca_relaxed": _DEFAULT_OBSERVATION_SPEC,
    "socnav_hrvo": _DEFAULT_OBSERVATION_SPEC,
    "social_navigation_pyenvs_orca": _DEFAULT_OBSERVATION_SPEC,
    "social_navigation_pyenvs_socialforce": _DEFAULT_OBSERVATION_SPEC,
    "social_navigation_pyenvs_sfm_helbing": _DEFAULT_OBSERVATION_SPEC,
    "social_navigation_pyenvs_hsfm_new_guo": {
        "default_mode": "headed_socnav_state",
        "supported_modes": ("headed_socnav_state",),
        "inputs": ("robot_state", "robot_heading", "goal", "pedestrians"),
        "notes": "Structured headed social-navigation state for HSFM-style adapters.",
    },
    "stream_gap": _DEFAULT_OBSERVATION_SPEC,
    "gap_prediction": _DEFAULT_OBSERVATION_SPEC,
    "risk_dwa": _DEFAULT_OBSERVATION_SPEC,
    "risk_surface_dwa": {
        "default_mode": "socnav_state",
        "supported_modes": ("socnav_state",),
        "inputs": ("robot_state", "goal", "pedestrians", "local_risk_surface"),
        "notes": (
            "Exploratory deterministic risk-surface producer consumes structured SocNav state, "
            "derives an ego-frame occupancy-compatible risk grid, and wraps risk_dwa."
        ),
    },
    "mppi_social": _DEFAULT_OBSERVATION_SPEC,
    "hybrid_portfolio": _DEFAULT_OBSERVATION_SPEC,
    "planner_selector_v2_diagnostic": _DEFAULT_OBSERVATION_SPEC,
    "hybrid_orca_sampler": _DEFAULT_OBSERVATION_SPEC,
    "policy_stack_v1": _DEFAULT_OBSERVATION_SPEC,
    "safety_barrier": {
        "default_mode": "socnav_state",
        "supported_modes": ("socnav_state", "sensor_fusion_state"),
        "inputs": ("robot_state", "goal", "occupancy_grid", "lidar_rays"),
        "notes": (
            "Safety-barrier planner consumes SocNav occupancy by default; sensor_fusion_state "
            "is valid only through the explicit LiDAR-to-ego-occupancy adapter."
        ),
    },
    "prediction_planner": {
        "default_mode": "socnav_state",
        "supported_modes": ("socnav_state",),
        "inputs": ("robot_state", "goal", "pedestrians", "history"),
        "notes": "Predictive planner consumes structured SocNav state plus short history features.",
    },
    "predictive_mppi": {
        "default_mode": "socnav_state",
        "supported_modes": ("socnav_state",),
        "inputs": ("robot_state", "goal", "pedestrians", "prediction_model"),
        "notes": "Predictive MPPI consumes SocNav state and learned prediction features.",
    },
    "prediction_mpc": {
        "default_mode": "socnav_state",
        "supported_modes": ("socnav_state",),
        "inputs": ("robot_state", "goal", "pedestrians", "predicted_pedestrian_futures"),
        "notes": (
            "Consumes structured SocNav state. The constant-velocity backend derives "
            "time-varying pedestrian futures from tracked pedestrian positions and "
            "ego-frame pedestrian velocities rotated into world coordinates."
        ),
    },
    "ppo": {
        "default_mode": "sensor_fusion_state",
        "supported_modes": ("sensor_fusion_state", "socnav_state"),
        "inputs": ("robot_state", "goal", "lidar_rays", "history"),
        "notes": (
            "PPO checkpoint inference uses sensor-fusion by default; BC/materialized "
            "checkpoints may opt into the SocNav structured observation stack they were "
            "trained with."
        ),
    },
    "hybrid_global_rl": {
        "default_mode": "sensor_fusion_state",
        "supported_modes": ("sensor_fusion_state", "socnav_state"),
        "inputs": ("robot_state", "goal", "route_waypoint", "pedestrians"),
        "notes": (
            "Route-conditioned learned local planner rewrites goal.current/goal_current "
            "to a short-horizon waypoint before invoking an existing PPO or SAC policy."
        ),
    },
    "guarded_ppo": {
        "default_mode": "sensor_fusion_state",
        "supported_modes": ("sensor_fusion_state", "socnav_state"),
        "inputs": (
            "robot_state",
            "goal",
            "lidar_rays",
            "history",
            "safety_guard",
        ),
        "notes": (
            "Guarded PPO uses sensor-fusion policy inputs plus a local safety guard by "
            "default; materialized BC checkpoints may request SocNav structured inputs."
        ),
    },
    "crowdnav_height": {
        "default_mode": "lidar_human_state",
        "supported_modes": ("lidar_human_state",),
        "inputs": ("robot_state", "goal", "lidar_rays", "humans"),
        "notes": "CrowdNav HEIGHT wrapper reconstructs the upstream lidar-plus-human dict input.",
    },
    "lidar_social_force": {
        "default_mode": "sensor_fusion_state",
        "supported_modes": ("sensor_fusion_state",),
        "inputs": ("robot_state", "goal", "lidar_rays"),
        "notes": (
            "Testing-only adapter derives visible endpoint-cluster tracks from LiDAR rays, "
            "assigns zero velocity without identity persistence, and does not consume "
            "privileged pedestrian or map state."
        ),
    },
    "lidar_grid_route": {
        "default_mode": "sensor_fusion_state",
        "supported_modes": ("sensor_fusion_state",),
        "inputs": ("robot_state", "goal", "lidar_rays"),
        "notes": (
            "Testing-only adapter derives an ego-frame occupancy grid from LiDAR ray endpoints "
            "and does not consume privileged map, pedestrian, or simulator state."
        ),
    },
    "sonic_crowdnav": {
        "default_mode": "gst_human_state",
        "supported_modes": ("gst_human_state",),
        "inputs": ("robot_state", "goal", "humans"),
        "notes": "SoNIC/GenSafeNav-style GST checkpoint input contract.",
    },
    "gensafenav_ours_gst": {
        "default_mode": "gst_human_state",
        "supported_modes": ("gst_human_state",),
        "inputs": ("robot_state", "goal", "humans"),
        "notes": "GenSafeNav Ours_GST checkpoint input contract.",
    },
    "gensafenav_ours_gst_guarded": {
        "default_mode": "gst_human_state",
        "supported_modes": ("gst_human_state",),
        "inputs": ("robot_state", "goal", "humans", "safety_guard"),
        "notes": "Guarded GenSafeNav Ours_GST checkpoint input contract.",
    },
    "gensafenav_gst_predictor_rand": {
        "default_mode": "gst_human_state",
        "supported_modes": ("gst_human_state",),
        "inputs": ("robot_state", "goal", "humans"),
        "notes": "GenSafeNav GST_predictor_rand checkpoint input contract.",
    },
    "gensafenav_gst_predictor_rand_guarded": {
        "default_mode": "gst_human_state",
        "supported_modes": ("gst_human_state",),
        "inputs": ("robot_state", "goal", "humans", "safety_guard"),
        "notes": "Guarded GenSafeNav GST_predictor_rand checkpoint input contract.",
    },
}

_UPSTREAM_REFERENCE_BY_CANONICAL: dict[str, dict[str, Any]] = {
    "orca": {
        "repo_url": "https://github.com/mit-acl/Python-RVO2",
        "commit": "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
        "vendored_path": "third_party/python-rvo2",
        "adapter_boundary": (
            "Use upstream Python-RVO2 to solve reciprocal-avoidance velocity in world coordinates, "
            "then project the selected velocity into Robot SF unicycle_vw commands."
        ),
    },
    "socnav_orca_nonholonomic": {
        "repo_url": "https://github.com/mit-acl/Python-RVO2",
        "commit": "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
        "vendored_path": "third_party/python-rvo2",
        "adapter_boundary": (
            "Use upstream Python-RVO2 to solve reciprocal-avoidance velocity in world coordinates, "
            "apply nonholonomic commitment heuristics, and project the selected velocity into "
            "Robot SF unicycle_vw commands."
        ),
    },
    "socnav_orca_dd": {
        "repo_url": "https://github.com/mit-acl/Python-RVO2",
        "commit": "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
        "vendored_path": "third_party/python-rvo2",
        "adapter_boundary": (
            "Use upstream Python-RVO2 to solve reciprocal-avoidance velocity in world coordinates, "
            "tune the result for differential-drive compatibility, and project it into Robot SF "
            "unicycle_vw commands."
        ),
    },
    "socnav_orca_relaxed": {
        "repo_url": "https://github.com/mit-acl/Python-RVO2",
        "commit": "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
        "vendored_path": "third_party/python-rvo2",
        "adapter_boundary": (
            "Use upstream Python-RVO2 to solve reciprocal-avoidance velocity in world coordinates, "
            "apply relaxed safety tuning, and project the selected velocity into Robot SF "
            "unicycle_vw commands."
        ),
    },
    "socnav_hrvo": {
        "repo_url": "https://github.com/snape/HRVO",
        "license": "Apache-2.0",
        "reference_repo_url": (
            "https://github.com/atb033/multi_agent_path_planning/blob/master/"
            "decentralized/velocity_obstacle/velocity_obstacle.py"
        ),
        "adapter_boundary": (
            "Run the local Robot SF HRVO geometry solver inspired by the upstream HRVO library, "
            "then project the selected world-frame velocity into Robot SF unicycle_vw commands."
        ),
        "provenance_note": (
            "Local implementation informed by upstream references; not a wrapped upstream runtime."
        ),
    },
    "hrvo": {
        "repo_url": "https://github.com/snape/HRVO",
        "license": "Apache-2.0",
        "reference_repo_url": (
            "https://github.com/atb033/multi_agent_path_planning/blob/master/"
            "decentralized/velocity_obstacle/velocity_obstacle.py"
        ),
        "adapter_boundary": (
            "Run the local Robot SF HRVO geometry solver inspired by the upstream HRVO library "
            "and VO reference, then project the selected world-frame velocity into "
            "Robot SF unicycle_vw commands."
        ),
        "provenance_note": (
            "Local implementation informed by upstream references; not a wrapped upstream runtime."
        ),
    },
    "drl_vo": {
        "repo_url": "https://github.com/TempleRAIL/drl_vo_nav",
        "commit": "6d734b6e0df77fd4c4faa4649ca0fcb3e69cf835",
        "adapter_boundary": (
            "Hybrid DRL-VO planner: learned policy coupled with velocity obstacle safety fallback. "
            "For benchmark contract, map policy output to Robot SF unicycle_vw commands."
        ),
    },
    "social_navigation_pyenvs_orca": {
        "repo_url": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
        "commit": "checked_out_local_probe_2026_03_20",
        "checkout_path": "output/repos/Social-Navigation-PyEnvs",
        "upstream_policy": "crowd_nav.policy_no_train.orca.ORCA",
        "adapter_boundary": (
            "Map Robot SF SocNav observations into the upstream Social-Navigation-PyEnvs "
            "JointState contract, run upstream ORCA predict(), then project ActionXY into "
            "Robot SF unicycle_vw commands."
        ),
    },
    "social_navigation_pyenvs_socialforce": {
        "repo_url": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
        "commit": "f9cd244d3e529247ca1031364de22954717b9493",
        "checkout_path": "output/repos/Social-Navigation-PyEnvs",
        "upstream_policy": "crowd_nav.policy_no_train.socialforce.SocialForce",
        "adapter_boundary": (
            "Map Robot SF SocNav observations into the upstream Social-Navigation-PyEnvs "
            "JointState contract, run upstream SocialForce predict() through an explicit "
            "CrowdNav-style compatibility runtime for socialforce==0.2.3, then project "
            "ActionXY into Robot SF unicycle_vw commands."
        ),
        "runtime_dependency": "socialforce==0.2.3",
        "runtime_strategy": "crowdnav_socialforce_compat_shim",
    },
    "social_navigation_pyenvs_sfm_helbing": {
        "repo_url": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
        "commit": "f9cd244d3e529247ca1031364de22954717b9493",
        "checkout_path": "output/repos/Social-Navigation-PyEnvs",
        "upstream_policy": "crowd_nav.policy_no_train.sfm_helbing.SFMHelbing",
        "adapter_boundary": (
            "Map Robot SF SocNav observations into the upstream Social-Navigation-PyEnvs "
            "JointState contract, run upstream SFM-Helbing predict(), then project ActionXY "
            "into Robot SF unicycle_vw commands."
        ),
    },
    "social_navigation_pyenvs_hsfm_new_guo": {
        "repo_url": "https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
        "commit": "f9cd244d3e529247ca1031364de22954717b9493",
        "checkout_path": "output/repos/Social-Navigation-PyEnvs",
        "upstream_policy": "crowd_nav.policy_no_train.hsfm_new_guo.HSFMNewGuo",
        "adapter_boundary": (
            "Map Robot SF SocNav observations into the upstream Social-Navigation-PyEnvs "
            "headed JointState contract, run upstream HSFM-New-Guo predict(), then project "
            "body-frame ActionXYW or NewHeadedState outputs into Robot SF unicycle_vw commands."
        ),
    },
    "crowdnav_height": {
        "repo_url": "https://github.com/Shuijing725/CrowdNav_HEIGHT",
        "commit": "65451bcdd1f3fbebaf6e96a0de73aaa56d74ca05",
        "checkout_path": "output/repos/CrowdNav_HEIGHT",
        "checkpoint_bundle": (
            "https://drive.google.com/drive/folders/1B1EA_gTMKg3hFQ_PXpQYjA8JBRHgmEQR?usp=drive_link"
        ),
        "default_checkpoint": "HEIGHT/checkpoints/237800.pt",
        "upstream_policy": "training.networks.model.Policy[selfAttn_merge_srnn_lidar]",
        "adapter_boundary": (
            "Rebuild the upstream lidar-plus-human dict observation from Robot SF SocNav state, "
            "run checkpoint inference, and translate the upstream discrete delta-v/delta-theta "
            "action table into stateful Robot SF unicycle_vw commands."
        ),
    },
    "sonic_crowdnav": {
        "repo_url": "https://github.com/tasl-lab/SoNIC-Social-Nav",
        "reference_repo_url": "https://github.com/tasl-lab/GenSafeNav",
        "commit": "24d4a64",
        "checkout_path": "output/repos/SoNIC-Social-Nav",
        "default_model_name": "SoNIC_GST",
        "default_checkpoint": "trained_models/SoNIC_GST/checkpoints/05207.pt",
        "upstream_policy": "rl.networks.model.Policy[selfAttn_merge_srnn]",
        "adapter_boundary": (
            "Map Robot SF SocNav observations into the SoNIC model-only dict contract, run "
            "upstream checkpoint inference with explicit import/runtime shims, and project "
            "upstream ActionXY velocities into Robot SF unicycle_vw commands."
        ),
    },
    "gensafenav_ours_gst": {
        "repo_url": "https://github.com/tasl-lab/GenSafeNav",
        "reference_repo_url": "https://github.com/tasl-lab/SoNIC-Social-Nav",
        "commit": "01baf92",
        "checkout_path": "output/repos/GenSafeNav",
        "default_model_name": "Ours_GST",
        "default_checkpoint": "trained_models/Ours_GST/checkpoints/05207.pt",
        "upstream_policy": "rl.networks.model.Policy[selfAttn_merge_srnn]",
        "adapter_boundary": (
            "Map Robot SF SocNav observations into the GenSafeNav model-only dict contract, run "
            "the upstream constrained selfAttn_merge_srnn checkpoint with explicit import/runtime "
            "shims, and project upstream ActionXY velocities into Robot SF unicycle_vw commands."
        ),
    },
    "gensafenav_ours_gst_guarded": {
        "repo_url": "https://github.com/tasl-lab/GenSafeNav",
        "reference_repo_url": "https://github.com/tasl-lab/SoNIC-Social-Nav",
        "commit": "01baf92",
        "checkout_path": "output/repos/GenSafeNav",
        "default_model_name": "Ours_GST",
        "default_checkpoint": "trained_models/Ours_GST/checkpoints/05207.pt",
        "upstream_policy": "rl.networks.model.Policy[selfAttn_merge_srnn]",
        "adapter_boundary": (
            "Run the GenSafeNav model-only Ours_GST checkpoint through the SoNIC-compatible "
            "adapter contract, then apply an explicit short-horizon safety guard with goal-policy "
            "fallback before emitting Robot SF unicycle_vw commands."
        ),
    },
    "gensafenav_gst_predictor_rand": {
        "repo_url": "https://github.com/tasl-lab/GenSafeNav",
        "reference_repo_url": "https://github.com/tasl-lab/SoNIC-Social-Nav",
        "commit": "01baf92",
        "checkout_path": "output/repos/GenSafeNav",
        "default_model_name": "GST_predictor_rand",
        "default_checkpoint": "trained_models/GST_predictor_rand/checkpoints/05207.pt",
        "upstream_policy": "rl.networks.model.Policy[selfAttn_merge_srnn]",
        "adapter_boundary": (
            "Map Robot SF SocNav observations into the GenSafeNav CrowdNav++-style model-only "
            "dict contract, run the upstream selfAttn_merge_srnn checkpoint with explicit "
            "import/runtime shims, and project upstream ActionXY velocities into Robot SF "
            "unicycle_vw commands."
        ),
    },
    "gensafenav_gst_predictor_rand_guarded": {
        "repo_url": "https://github.com/tasl-lab/GenSafeNav",
        "reference_repo_url": "https://github.com/tasl-lab/SoNIC-Social-Nav",
        "commit": "01baf92",
        "checkout_path": "output/repos/GenSafeNav",
        "default_model_name": "GST_predictor_rand",
        "default_checkpoint": "trained_models/GST_predictor_rand/checkpoints/05207.pt",
        "upstream_policy": "rl.networks.model.Policy[selfAttn_merge_srnn]",
        "adapter_boundary": (
            "Run the GenSafeNav model-only GST_predictor_rand checkpoint through the "
            "SoNIC-compatible adapter contract, then apply an explicit short-horizon safety "
            "guard with goal-policy fallback before emitting Robot SF unicycle_vw commands."
        ),
    },
    "sicnav": {
        "repo_url": "https://github.com/sepsamavi/safe-interactive-crowdnav",
        "commit": "c702fb8ac9ba6439ca61da7dde68b8524bbc6a1f",
        "checkout_path": "third_party/external_repos/sicnav",
        "upstream_policy": "sicnav_diffusion.policy.sicnav_acados.SICNavAcados",
        "default_checkpoint": (
            "sicnav_diffusion/JMID/MID/checkpoints/jrdb_bev_0_25_multi_class_epoch16.pt"
        ),
        "adapter_boundary": (
            "Map Robot SF structured robot/human state into the upstream SICNav checkpoint "
            "contract, run the external MPC policy, and project the selected velocity or "
            "unicycle command into Robot SF unicycle_vw semantics."
        ),
    },
    "dr_mpc": {
        "repo_url": "https://github.com/James-R-Han/DR-MPC",
        "commit": "local_vendor_reference",
        "checkout_path": "third_party/external_mpc_repos/dr_mpc",
        "upstream_policy": "scripts.models.model.Policy",
        "adapter_boundary": (
            "Map Robot SF structured robot/human state into the upstream DR-MPC residual "
            "control contract, run the external residual-MPC policy, and project the selected "
            "action into Robot SF unicycle_vw semantics."
        ),
    },
}

_KINEMATICS_PROFILE_BY_CANONICAL: dict[str, dict[str, Any]] = {
    "goal": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": True,
        "supports_adapter_commands": False,
        "default_execution_mode": "native",
    },
    "social_force": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SocialForcePlannerAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "orca": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "ORCAPlannerAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "socnav_orca_nonholonomic": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "ORCAPlannerAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "socnav_orca_dd": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "ORCAPlannerAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "socnav_orca_relaxed": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "ORCAPlannerAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "socnav_hrvo": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "HRVOPlannerAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "hrvo": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "HRVOPlannerAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "drl_vo": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "DRLVOPlannerAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "social_navigation_pyenvs_orca": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SocialNavigationPyEnvsORCAAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "social_navigation_pyenvs_socialforce": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SocialNavigationPyEnvsForceModelAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
        "runtime_dependency": "socialforce==0.2.3",
        "runtime_strategy": "crowdnav_socialforce_compat_shim",
    },
    "social_navigation_pyenvs_sfm_helbing": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SocialNavigationPyEnvsForceModelAdapter",
        "upstream_command_space": "velocity_vector_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "social_navigation_pyenvs_hsfm_new_guo": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SocialNavigationPyEnvsHSFMAdapter",
        "upstream_command_space": "body_velocity_xy_plus_omega",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "body_velocity_heading_safe_to_unicycle_vw",
        "projection_documented": True,
    },
    "crowdnav_height": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "CrowdNavHeightAdapter",
        "upstream_command_space": "discrete_delta_v_and_delta_theta",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "upstream_discrete_delta_vw_to_unicycle_vw_stateful",
        "projection_documented": True,
    },
    "sonic_crowdnav": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SonicCrowdNavAdapter",
        "upstream_command_space": "holonomic_velocity_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "gensafenav_ours_gst": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SonicCrowdNavAdapter",
        "upstream_command_space": "holonomic_velocity_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "gensafenav_ours_gst_guarded": {
        "planner_command_space": "mixed_vw_or_unicycle",
        "supports_native_commands": True,
        "supports_adapter_commands": True,
        "default_execution_mode": "mixed",
        "default_adapter_name": "sonic_guarded_goal_fallback",
        "upstream_command_space": "holonomic_velocity_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "gensafenav_gst_predictor_rand": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SonicCrowdNavAdapter",
        "upstream_command_space": "holonomic_velocity_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "gensafenav_gst_predictor_rand_guarded": {
        "planner_command_space": "mixed_vw_or_unicycle",
        "supports_native_commands": True,
        "supports_adapter_commands": True,
        "default_execution_mode": "mixed",
        "default_adapter_name": "sonic_guarded_goal_fallback",
        "upstream_command_space": "holonomic_velocity_xy",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "sicnav": {
        "planner_command_space": "mixed_vw_or_unicycle",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SICNavPlanner",
        "upstream_command_space": "velocity_vector_xy_or_unicycle",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "dr_mpc": {
        "planner_command_space": "mixed_vw_or_unicycle",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "DRMPCPlanner",
        "upstream_command_space": "velocity_vector_xy_or_unicycle",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "heading_safe_velocity_to_unicycle_vw",
        "projection_documented": True,
    },
    "ppo": {
        "planner_command_space": "mixed_vw_or_vxy",
        "supports_native_commands": True,
        "supports_adapter_commands": True,
        "default_execution_mode": "mixed",
        "default_adapter_name": "ppo_action_to_unicycle",
    },
    "sac": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": True,
        "supports_adapter_commands": False,
        "default_execution_mode": "native",
    },
    "hybrid_global_rl": {
        "planner_command_space": "mixed_vw_or_vxy",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "HybridGlobalRLLocalAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "route_waypoint_conditioned_rl_action_to_unicycle_vw",
        "projection_documented": True,
        "diagnostic_reference_only": True,
    },
    "guarded_ppo": {
        "planner_command_space": "mixed_vw_or_vxy",
        "supports_native_commands": True,
        "supports_adapter_commands": True,
        "default_execution_mode": "mixed",
        "default_adapter_name": "guarded_ppo_action_to_unicycle",
    },
    "socnav_sampling": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SamplingPlannerAdapter",
    },
    "sacadrl": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SACADRLPlannerAdapter",
    },
    "prediction_planner": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "PredictionPlannerAdapter",
    },
    "predictive_mppi": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "PredictiveMPPIAdapter",
    },
    "risk_dwa": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "RiskDWAPlannerAdapter",
    },
    "risk_surface_dwa": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "RiskSurfacePlannerAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "deterministic_local_risk_surface_to_risk_dwa_unicycle_vw",
        "projection_documented": True,
        "testing_only_adapter": True,
        "prototype_only": True,
    },
    "hybrid_rule_local_planner": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "HybridRuleLocalPlannerAdapter",
    },
    "adaptive_proxemic_selector_v0": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "AdaptiveProxemicSelectorAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "local_context_selector_over_fixed_hybrid_rule_profiles",
        "projection_documented": True,
        "diagnostic_reference_only": True,
    },
    "adaptive_proxemic_selector_v1": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "AdaptiveProxemicSelectorAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "neutral_default_local_context_selector_over_fixed_hybrid_rule_profiles",
        "projection_documented": True,
        "diagnostic_reference_only": True,
    },
    "safety_barrier": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SafetyBarrierPlannerAdapter",
    },
    "grid_route": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "GridRoutePlannerAdapter",
    },
    "topology_guided_hybrid_rule_v0": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "TopologyGuidedHybridRulePlannerAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "masked_route_hypotheses_to_hybrid_rule_corridor_subgoal",
        "projection_documented": True,
        "testing_only_adapter": True,
        "diagnostic_reference_only": True,
    },
    "lidar_social_force": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "LidarTrackedSocialForceAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "lidar_endpoint_clusters_to_social_force_unicycle_vw",
        "projection_documented": True,
        "testing_only_adapter": True,
        "perception_tracking_mode": "single_frame_endpoint_clusters_zero_velocity",
    },
    "lidar_grid_route": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "LidarOccupancyGridRouteAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "lidar_ray_endpoint_ego_grid_to_grid_route_unicycle_vw",
        "projection_documented": True,
        "testing_only_adapter": True,
    },
    "trivial_reference": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "TrivialReferencePlannerAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "bounded_goal_facing_reference_command",
        "projection_documented": True,
        "diagnostic_reference_only": True,
    },
    "mppi_social": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "MPPISocialPlannerAdapter",
    },
    "hybrid_portfolio": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "HybridPortfolioAdapter",
    },
    "planner_selector_v2_diagnostic": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "PlannerSelectorV2DiagnosticAdapter",
        "diagnostic_only": True,
        "benchmark_strength": False,
    },
    "policy_stack_v1": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": True,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "PolicyStackV1Adapter",
    },
    "hybrid_orca_sampler": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "HybridORCASamplerAdapter",
    },
    "stream_gap": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "StreamGapPlannerAdapter",
    },
    "gap_prediction": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "GapAwarePredictionAdapter",
    },
    "socnav_bench": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SocNavBenchSamplingAdapter",
    },
    "random": {
        "planner_command_space": "randomized_vxy_or_vw",
        "supports_native_commands": True,
        "supports_adapter_commands": False,
        "default_execution_mode": "native",
    },
    "fast_pysf_planner": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "PlannerActionAdapter",
    },
    "rvo": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SamplingPlannerAdapter",
    },
    "dwa": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "SamplingPlannerAdapter",
    },
    "teb": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "TEBCommitmentPlannerAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "native_corridor_commitment_unicycle_vw",
        "projection_documented": True,
    },
    "nmpc_social": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "NMPCSocialPlannerAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "native_nmpc_unicycle_vw",
        "projection_documented": True,
    },
    "prediction_mpc": {
        "planner_command_space": "unicycle_vw",
        "supports_native_commands": False,
        "supports_adapter_commands": True,
        "default_execution_mode": "adapter",
        "default_adapter_name": "PredictionMPCPlannerAdapter",
        "benchmark_command_space": "unicycle_vw",
        "projection_policy": "constant_velocity_time_varying_pedestrian_constraints",
        "projection_documented": True,
    },
}


def canonical_algorithm_name(algo: str) -> str:
    """Normalize algorithm aliases to canonical benchmark names when possible.

    Returns:
        Canonical algorithm name when known, otherwise normalized input alias.
    """
    alias = str(algo).strip().lower()
    readiness = get_algorithm_readiness(alias)
    return readiness.canonical_name if readiness is not None else alias


def observation_spec_for_algorithm(algo: str) -> dict[str, Any]:
    """Return the declared observation contract for an algorithm.

    Returns:
        dict[str, Any]: Stable observation-spec payload with default and supported modes.
    """
    canonical = canonical_algorithm_name(algo)
    spec = _OBSERVATION_SPEC_BY_CANONICAL.get(canonical, _DEFAULT_OBSERVATION_SPEC)
    return {
        "default_mode": str(spec["default_mode"]),
        "supported_modes": [str(mode) for mode in spec["supported_modes"]],
        "inputs": [str(value) for value in spec.get("inputs", ())],
        "notes": str(spec.get("notes", "")),
    }


def resolve_observation_mode(
    algo: str,
    requested_mode: str | None = None,
    *,
    observation_level: str | None = None,
) -> str:
    """Resolve and validate the active observation mode for an algorithm.

    Args:
        algo: Algorithm label or alias.
        requested_mode: Optional explicit observation mode override.
        observation_level: Optional graded observation-level override.

    Returns:
        The active observation mode.

    Raises:
        ValueError: If ``requested_mode`` is unsupported for ``algo``.
    """
    spec = observation_spec_for_algorithm(algo)
    default_mode = str(spec["default_mode"])
    supported = tuple(str(mode) for mode in spec["supported_modes"])
    if observation_level is not None:
        contract = resolve_observation_level_contract(
            canonical_algorithm_name(algo),
            observation_level=observation_level,
            requested_observation_mode=requested_mode,
            algorithm_default_mode=default_mode,
            algorithm_supported_modes=supported,
        )
        return str(contract["active_observation_mode"])

    requested = str(requested_mode).strip() if requested_mode is not None else ""
    active_mode = requested or default_mode
    if active_mode not in supported:
        supported_text = ", ".join(supported)
        raise ValueError(
            f"Observation mode '{active_mode}' is not supported by algorithm "
            f"'{canonical_algorithm_name(algo)}'. Supported modes: {supported_text}."
        )
    return active_mode


def _normalize_optional_string(value: Any) -> str | None:
    """Return a stripped string or ``None`` for blank/missing values."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _planner_observation_modes_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two planner-side observation mode labels describe the same family."""
    left_mode = _normalize_optional_string(left)
    right_mode = _normalize_optional_string(right)
    if left_mode is None or right_mode is None:
        return True
    left_key = left_mode.lower()
    right_key = right_mode.lower()
    if left_key == right_key:
        return True
    return (
        left_key in _DICT_STYLE_PLANNER_OBSERVATION_MODES
        and right_key in _DICT_STYLE_PLANNER_OBSERVATION_MODES
    )


def _dict_style_checkpoint_metadata_required(canonical: str, planner_mode: str | None) -> bool:
    """Return whether a dict-style checkpoint would otherwise use the wrong default producer."""
    normalized = _normalize_optional_string(planner_mode)
    if normalized is None or normalized.lower() not in _DICT_STYLE_PLANNER_OBSERVATION_MODES:
        return False
    return resolve_observation_mode(canonical) != "socnav_state"


def _observation_contract_from_transfer_metadata(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return transfer-metadata observation contract when present."""
    transfer = payload.get("transfer_benchmark")
    if not isinstance(transfer, dict):
        algorithm_metadata = payload.get("algorithm_metadata")
        if isinstance(algorithm_metadata, dict):
            transfer = algorithm_metadata.get("transfer_benchmark")
    if not isinstance(transfer, dict):
        return None, None
    contract = transfer.get("observation_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            "malformed learned checkpoint observation metadata: "
            "transfer_benchmark.observation_contract must be a mapping"
        )
    return contract, "algo_config.transfer_benchmark.observation_contract"


def _observation_contract_from_registry(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return model-registry benchmark-promotion metadata when a model_id is configured."""
    model_id = _normalize_optional_string(payload.get("model_id"))
    if model_id is None:
        return None, None
    try:
        registry_entry = get_registry_entry(model_id)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "requires learned checkpoint observation metadata but model registry lookup failed "
            f"for model_id={model_id!r}: {exc}"
        ) from exc
    promotion = registry_entry.get("benchmark_promotion")
    if not isinstance(promotion, dict):
        return None, None
    return promotion, "model_registry.benchmark_promotion"


def _learned_checkpoint_observation_metadata(
    algo_config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the first supported learned-checkpoint observation metadata block."""
    transfer_contract, transfer_source = _observation_contract_from_transfer_metadata(algo_config)
    if transfer_contract is not None:
        return transfer_contract, transfer_source

    direct_contract = algo_config.get("observation_contract")
    if isinstance(direct_contract, dict):
        return direct_contract, "algo_config.observation_contract"
    if direct_contract is not None:
        raise ValueError(
            "malformed learned checkpoint observation metadata: "
            "observation_contract must be a mapping"
        )

    direct_promotion = algo_config.get("benchmark_promotion")
    if isinstance(direct_promotion, dict):
        return direct_promotion, "algo_config.benchmark_promotion"
    if direct_promotion is not None:
        raise ValueError(
            "malformed learned checkpoint observation metadata: "
            "benchmark_promotion must be a mapping"
        )

    return _observation_contract_from_registry(algo_config)


def _planner_observation_mode_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Return the metadata-declared planner-side observation mode, if any."""
    for key in ("planner_observation_mode", "observation_mode", "obs_mode"):
        value = _normalize_optional_string(metadata.get(key))
        if value is not None:
            return value
    return None


def _metadata_declares_observation_contract(metadata: dict[str, Any]) -> bool:
    """Return whether metadata contains fields that can resolve an observation producer."""
    for key in (
        "observation_level",
        "active_observation_mode",
        "runtime_observation_mode",
        "runner_observation_mode",
        "observation_producer_mode",
        "planner_observation_mode",
        "observation_mode",
        "obs_mode",
    ):
        if _normalize_optional_string(metadata.get(key)) is not None:
            return True
    return False


def _active_observation_mode_from_checkpoint_metadata(
    canonical: str,
    metadata: dict[str, Any],
) -> tuple[str, str | None]:
    """Resolve the runtime observation producer declared by checkpoint metadata.

    Returns:
        Active observation mode and the raw metadata observation-level label, when present.
    """
    runtime_mode = None
    for key in (
        "active_observation_mode",
        "runtime_observation_mode",
        "runner_observation_mode",
        "observation_producer_mode",
    ):
        runtime_mode = _normalize_optional_string(metadata.get(key))
        if runtime_mode is not None:
            break
    observation_level = _normalize_optional_string(metadata.get("observation_level"))

    try:
        if runtime_mode is not None:
            level = (
                observation_level
                if observation_level is not None and observation_level in OBSERVATION_LEVEL_KEYS
                else None
            )
            return resolve_observation_mode(
                canonical,
                runtime_mode,
                observation_level=level,
            ), observation_level
        if observation_level is not None and observation_level in OBSERVATION_LEVEL_KEYS:
            return (
                resolve_observation_mode(canonical, observation_level=observation_level),
                observation_level,
            )
        if observation_level is not None:
            return resolve_observation_mode(canonical, observation_level), observation_level
    except ValueError as exc:
        raise ValueError(
            "incompatible learned checkpoint observation metadata for "
            f"algorithm '{canonical}': {exc}"
        ) from exc

    raise ValueError(
        "malformed learned checkpoint observation metadata: expected observation_level "
        "or active_observation_mode"
    )


def resolve_learned_checkpoint_observation_contract(
    algo: str,
    algo_config: dict[str, Any] | None = None,
    *,
    observation_mode: str | None = None,
    observation_level: str | None = None,
) -> dict[str, Any]:
    """Resolve the active observation producer for learned-checkpoint benchmark execution.

    Explicit runner overrides win. Without overrides, dict-family checkpoint configs whose
    algorithm default is not SocNav state must provide metadata that declares the required
    observation contract; this prevents running those checkpoints with the wrong producer.

    Returns:
        A classification payload containing the active observation mode and metadata source.
    """
    canonical = canonical_algorithm_name(algo)
    config = dict(algo_config or {})
    config_planner_mode = _normalize_optional_string(config.get("obs_mode"))

    explicit_mode = _normalize_optional_string(observation_mode)
    explicit_level = _normalize_optional_string(observation_level)
    if explicit_mode is not None or explicit_level is not None:
        active_mode = resolve_observation_mode(
            canonical,
            explicit_mode,
            observation_level=explicit_level,
        )
        if explicit_mode is not None and explicit_level is not None:
            source = "explicit_observation_mode_and_level"
        elif explicit_mode is not None:
            source = "explicit_observation_mode"
        else:
            source = "explicit_observation_level"
        return {
            "status": "explicit_override",
            "metadata_source": source,
            "active_observation_mode": active_mode,
            "observation_level": explicit_level,
            "observation_level_key": (
                explicit_level if explicit_level in OBSERVATION_LEVEL_KEYS else None
            ),
            "planner_observation_mode": config_planner_mode,
        }

    metadata_required = _dict_style_checkpoint_metadata_required(canonical, config_planner_mode)
    metadata, metadata_source = _learned_checkpoint_observation_metadata(config)

    if metadata is None:
        if metadata_required:
            raise ValueError(
                f"Algorithm '{canonical}' with obs_mode={config_planner_mode!r} requires "
                "learned checkpoint observation metadata; provide model registry "
                "benchmark_promotion, transfer_benchmark.observation_contract, or an explicit "
                "observation_mode/observation_level override."
            )
        active_mode = resolve_observation_mode(canonical)
        return {
            "status": "not_applicable",
            "metadata_source": "algorithm_default",
            "active_observation_mode": active_mode,
            "observation_level": observation_level_for_mode(active_mode).key,
            "observation_level_key": observation_level_for_mode(active_mode).key,
            "planner_observation_mode": config_planner_mode,
        }
    if not _metadata_declares_observation_contract(metadata):
        if metadata_required:
            raise ValueError(
                f"Algorithm '{canonical}' with obs_mode={config_planner_mode!r} requires "
                "learned checkpoint observation metadata, but the resolved metadata source "
                f"{metadata_source!r} does not declare observation_level or active_observation_mode."
            )
        active_mode = resolve_observation_mode(canonical)
        return {
            "status": "not_applicable",
            "metadata_source": "algorithm_default",
            "active_observation_mode": active_mode,
            "observation_level": observation_level_for_mode(active_mode).key,
            "observation_level_key": observation_level_for_mode(active_mode).key,
            "planner_observation_mode": config_planner_mode,
        }

    metadata_planner_mode = _planner_observation_mode_from_metadata(metadata)
    planner_mode = config_planner_mode or metadata_planner_mode
    if not _planner_observation_modes_compatible(config_planner_mode, metadata_planner_mode):
        raise ValueError(
            "incompatible learned checkpoint observation metadata for "
            f"algorithm '{canonical}': config obs_mode={config_planner_mode!r} conflicts with "
            f"metadata planner_observation_mode={metadata_planner_mode!r}"
        )

    active_mode, metadata_level = _active_observation_mode_from_checkpoint_metadata(
        canonical,
        metadata,
    )
    return {
        "status": "metadata_resolved",
        "metadata_source": metadata_source,
        "active_observation_mode": active_mode,
        "observation_level": metadata_level or observation_level_for_mode(active_mode).key,
        "observation_level_key": (
            metadata_level if metadata_level in OBSERVATION_LEVEL_KEYS else None
        )
        or observation_level_for_mode(active_mode).key,
        "planner_observation_mode": planner_mode,
    }


def _resolve_observation_metadata(
    canonical: str,
    *,
    observation_mode: str | None = None,
    observation_level: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve active observation mode and level metadata for an algorithm.

    Returns:
        tuple[str, dict[str, Any]]: Active observation mode and level metadata.
    """
    observation_spec = observation_spec_for_algorithm(canonical)
    if observation_level is None:
        active_mode = resolve_observation_mode(canonical, observation_mode)
        level_metadata = observation_level_for_mode(active_mode).to_metadata(
            active_observation_mode=active_mode
        )
    else:
        level_contract = resolve_observation_level_contract(
            canonical,
            observation_level=observation_level,
            requested_observation_mode=observation_mode,
            algorithm_default_mode=str(observation_spec["default_mode"]),
            algorithm_supported_modes=tuple(
                str(mode) for mode in observation_spec["supported_modes"]
            ),
        )
        active_mode = str(level_contract["active_observation_mode"])
        level_metadata = dict(level_contract["observation_level"])
    return active_mode, level_metadata


def _action_output_keys(command_space: str) -> tuple[str, ...]:
    """Return canonical action payload keys for a command-space label."""
    command = command_space.strip().lower()
    if command == "unicycle_vw":
        return ("v", "omega")
    if command == "body_velocity_xy_plus_omega":
        return ("vx", "vy", "omega")
    if command == "holonomic_vxy_world" or "velocity_xy" in command:
        return ("vx", "vy")
    if command == "discrete_delta_v_and_delta_theta":
        return ("delta_v", "delta_theta")
    if command in {"mixed_vw_or_vxy", "randomized_vxy_or_vw"}:
        return ("v", "omega", "vx", "vy")
    if command == "mixed_vw_or_unicycle":
        return ("v", "omega")
    raise ValueError(f"Unsupported planner command space: {command_space!r}")


def _action_frame(command_space: str) -> str:
    """Return the frame label associated with a command-space label."""
    command = command_space.strip().lower()
    if command == "body_velocity_xy_plus_omega":
        return "body"
    if command == "holonomic_vxy_world" or "velocity_xy" in command:
        return "world"
    return "robot"


def _compatible_robot_kinematics(
    profile: dict[str, Any],
    robot_kinematics: str | None,
) -> tuple[str, ...]:
    """Return the kinematics labels the action contract actually covers."""
    explicit = profile.get("compatible_robot_kinematics")
    if isinstance(explicit, (list, tuple)):
        values = tuple(str(value).strip().lower() for value in explicit if str(value).strip())
        if values:
            return values
    active = str(robot_kinematics or "").strip().lower()
    return (active or "unknown",)


def _observation_normalization(active_mode: str) -> str:
    """Return the normalization label for a planner observation mode."""
    if active_mode == "sensor_fusion_state":
        return "normalized_by_space_high"
    return "raw"


def planner_contract_for_algorithm(
    algo: str,
    *,
    observation_mode: str | None = None,
    observation_level: str | None = None,
    planner_kinematics: dict[str, Any] | None = None,
    robot_kinematics: str | None = None,
) -> PlannerMetadata:
    """Return typed planner compatibility metadata for a benchmark algorithm.

    The contract is declarative compatibility metadata. It does not certify
    benchmark quality or performance.
    """
    canonical = canonical_algorithm_name(algo)
    observation_spec = observation_spec_for_algorithm(canonical)
    active_mode, level_metadata = _resolve_observation_metadata(
        canonical,
        observation_mode=observation_mode,
        observation_level=observation_level,
    )
    profile = dict(_KINEMATICS_PROFILE_BY_CANONICAL.get(canonical, {}))
    if planner_kinematics is not None:
        profile.update(planner_kinematics)
    command_space = str(profile.get("planner_command_space", "unknown"))
    observation = ObservationContract(
        mode=str(observation_spec["default_mode"]),
        active_mode=active_mode,
        observation_level=str(level_metadata["key"]),
        perception_assumption=str(level_metadata["perception_assumption"]),
        supported_modes=tuple(str(mode) for mode in observation_spec["supported_modes"]),
        required_inputs=tuple(str(value) for value in observation_spec.get("inputs", ())),
        frame="world",
        normalization=_observation_normalization(active_mode),
        pedestrian_ordering="distance_ascending",
        notes=str(observation_spec.get("notes", "")),
    )
    action = ActionContract(
        command_space=command_space,
        output_keys=_action_output_keys(command_space),
        frame=_action_frame(command_space),
        normalization="raw",
        units="mps_radps",
        compatible_robot_kinematics=_compatible_robot_kinematics(profile, robot_kinematics),
        active_robot_kinematics=(
            str(robot_kinematics)
            if robot_kinematics not in {None, "", "mixed", "unknown"}
            else None
        ),
        notes=str(profile.get("projection_policy") or profile.get("execution_detail") or ""),
    )
    return PlannerMetadata(
        planner_id=canonical,
        observation_contract=observation,
        action_contract=action,
        reset_contract="seeded_reset",
        compatibility_scope="metadata_only",
        notes="Compatibility metadata only; not a benchmark quality claim.",
    )


def _base_kinematics_metadata(
    canonical_algo: str,
    *,
    execution_mode: str | None = None,
    adapter_name: str | None = None,
    robot_kinematics: str | None = None,
) -> dict[str, Any]:
    """Construct planner/robot kinematics metadata for a canonical algorithm.

    Args:
        canonical_algo: Canonical algorithm key used to resolve profile defaults.
        execution_mode: Optional runtime override for execution mode.
        adapter_name: Optional runtime override for adapter label.
        robot_kinematics: Optional runtime override for robot kinematics label.

    Returns:
        dict[str, Any]: Kinematics metadata containing robot kinematics, planner command space,
        native/adapter support flags, execution mode, adapter name, and adapter_active.
    """
    profile = _KINEMATICS_PROFILE_BY_CANONICAL.get(canonical_algo, {})
    metadata = {
        "robot_kinematics": robot_kinematics or "unknown",
        "planner_command_space": profile.get("planner_command_space", "unknown"),
        "supports_native_commands": bool(profile.get("supports_native_commands", False)),
        "supports_adapter_commands": bool(profile.get("supports_adapter_commands", False)),
        "execution_mode": execution_mode or profile.get("default_execution_mode", "unknown"),
        "adapter_name": adapter_name or profile.get("default_adapter_name", "none"),
    }
    for key, value in profile.items():
        if key not in metadata and not key.startswith("default_"):
            metadata[key] = value
    metadata["adapter_active"] = metadata["execution_mode"] in {"adapter", "mixed"}
    return metadata


def enrich_algorithm_metadata(
    *,
    algo: str,
    metadata: dict[str, Any] | None = None,
    execution_mode: str | None = None,
    adapter_name: str | None = None,
    robot_kinematics: str | None = None,
    adapter_impact_requested: bool | None = None,
    observation_mode: str | None = None,
    observation_level: str | None = None,
) -> dict[str, Any]:
    """Return metadata enriched with baseline category and compatibility fields.

    Args:
        algo: Algorithm label as selected by the caller.
        metadata: Existing metadata payload to preserve and augment.
        execution_mode: Optional runtime override (`native`/`adapter`/`mixed`).
        adapter_name: Optional runtime adapter override.
        robot_kinematics: Optional robot kinematics tag for this episode/run.
        adapter_impact_requested: Optional marker that adapter-impact probing was requested.
        observation_mode: Optional runtime observation-mode override.
        observation_level: Optional benchmark observation-level override.

    Returns:
        A metadata dictionary with stable benchmark contract keys.
    """
    enriched: dict[str, Any] = dict(metadata or {})
    requested = str(algo).strip().lower()
    canonical = canonical_algorithm_name(requested)

    enriched.setdefault("algorithm", requested)
    enriched.setdefault("status", "ok")
    enriched["canonical_algorithm"] = canonical
    enriched.setdefault(
        "baseline_category",
        _BASELINE_CATEGORY_BY_CANONICAL.get(canonical, "unknown"),
    )
    enriched.setdefault(
        "policy_semantics",
        _POLICY_SEMANTICS_BY_CANONICAL.get(canonical, "unspecified"),
    )
    upstream_reference = _UPSTREAM_REFERENCE_BY_CANONICAL.get(canonical)
    if upstream_reference is not None:
        enriched.setdefault("upstream_reference", dict(upstream_reference))

    observation_spec = observation_spec_for_algorithm(canonical)
    active_observation_mode, observation_level_metadata = _resolve_observation_metadata(
        canonical,
        observation_mode=observation_mode,
        observation_level=observation_level,
    )
    observation_spec["active_mode"] = active_observation_mode
    observation_spec["observation_level"] = observation_level_metadata["key"]
    observation_spec["override_applied"] = bool(
        observation_mode is not None
        and str(observation_mode).strip()
        and active_observation_mode != observation_spec["default_mode"]
    )
    enriched["observation_spec"] = observation_spec
    enriched["observation_level"] = observation_level_metadata

    current_kinematics = enriched.get("planner_kinematics")
    base_kinematics = _base_kinematics_metadata(
        canonical,
        execution_mode=execution_mode,
        adapter_name=adapter_name,
        robot_kinematics=robot_kinematics,
    )
    merged = dict(base_kinematics)
    if isinstance(current_kinematics, dict):
        merged.update(current_kinematics)
    if execution_mode is not None:
        merged["execution_mode"] = execution_mode
    if adapter_name is not None:
        merged["adapter_name"] = adapter_name
    if robot_kinematics is not None:
        merged["robot_kinematics"] = robot_kinematics
    merged["adapter_active"] = merged.get("execution_mode") in {"adapter", "mixed"}
    enriched["planner_kinematics"] = merged
    enriched["planner_contract"] = planner_contract_for_algorithm(
        canonical,
        observation_mode=active_observation_mode,
        observation_level=observation_level_metadata["key"],
        planner_kinematics=merged,
        robot_kinematics=str(merged.get("robot_kinematics", "unknown")),
    ).to_metadata()

    if canonical == "random":
        enriched.setdefault("stochastic_reference", True)
        enriched.setdefault("distinct_from_goal_baseline", True)

    if adapter_impact_requested is not None:
        impact = enriched.get("adapter_impact")
        if not isinstance(impact, dict):
            impact = {}
        impact.setdefault("requested", bool(adapter_impact_requested))
        impact.setdefault("native_steps", 0)
        impact.setdefault("adapted_steps", 0)
        impact.setdefault("status", "pending" if adapter_impact_requested else "disabled")
        enriched["adapter_impact"] = impact

    return enriched


def infer_execution_mode_from_counts(native_steps: int, adapted_steps: int) -> str:
    """Infer execution mode from runtime native/adapted step counters.

    Returns:
        One of ``native``, ``adapter``, ``mixed``, or ``unknown``.
    """
    if native_steps > 0 and adapted_steps > 0:
        return "mixed"
    if adapted_steps > 0:
        return "adapter"
    if native_steps > 0:
        return "native"
    return "unknown"


__all__ = [
    "canonical_algorithm_name",
    "enrich_algorithm_metadata",
    "infer_execution_mode_from_counts",
    "observation_spec_for_algorithm",
    "planner_contract_for_algorithm",
    "resolve_learned_checkpoint_observation_contract",
    "resolve_observation_mode",
]
