"""Export contract for the reviewed ``robot_sf.gym_env`` modules.

Guards the reviewed ``__all__`` surface for issue #6799: every declared export
resolves to the pre-change object on its pre-change import path, no declared
name is missing, and missing, misspelled, or stale names never leak into the
public surface. Re-exported names (e.g. ``EnvSettings``, ``VisualizableSimState``)
must keep the identity of their defining module rather than the exporter.
"""

from __future__ import annotations

import importlib

import pytest

EXPECTED_ALL: dict[str, list[str]] = {
    "_stub_robot_model": ["StubRobotModel"],
    "abstract_envs": ["BaseSimulationEnv", "MultiAgentEnv", "SingleAgentEnv"],
    "base_env": ["BaseEnv", "attach_planner_to_map"],
    "config_validation": [
        "_check_backend_valid",
        "_check_sensor_names_valid",
        "_check_unknown_keys",
        "get_resolved_config_dict",
        "validate_config",
    ],
    "crowd_sim_env": ["CrowdSimEnv", "CrowdSimulationConfig"],
    "env_config": [
        "BaseEnvSettings",
        "BaseSimulationConfig",
        "BicycleDriveRobot",
        "BicycleDriveSettings",
        "DifferentialDriveRobot",
        "DifferentialDriveSettings",
        "EnvSettings",
        "EnvSettingsNew",
        "LidarScannerSettings",
        "MapDefinitionPool",
        "PedEnvSettings",
        "PedEnvSettingsNew",
        "RobotEnvSettings",
        "RobotEnvSettingsNew",
        "SimulationSettings",
    ],
    "env_registry": [
        "BETA",
        "EXPERIMENTAL",
        "STABLE",
        "EnvEntry",
        "describe_env",
        "env_ids",
        "get_env",
        "list_envs",
        "register_env",
    ],
    "env_util": [
        "AgentType",
        "_pedestrian_coords_with_ego",
        "create_spaces",
        "create_spaces_with_image",
        "global_reset_seed",
        "init_collision_and_sensors",
        "init_collision_and_sensors_with_image",
        "init_ped_collision_and_sensors",
        "init_ped_spaces",
        "init_spaces",
        "make_grid_observation_spaces",
        "prepare_pedestrian_actions",
        "reset_episode_counter_for_seed",
    ],
    "environment_factory": [
        "EnvironmentFactory",
        "JsonlRecordingOptions",
        "RecordingOptions",
        "RenderOptions",
        "TelemetryOptions",
        "make_crowd_sim_env",
        "make_image_robot_env",
        "make_multi_robot_env",
        "make_pedestrian_env",
        "make_robot_env",
    ],
    "multi_robot_env": ["MultiRobotEnv"],
    "observation_config": [
        "ObservationStackSettings",
        "get_observation_stack_steps",
        "set_observation_stack_steps",
        "sync_observation_stack_settings",
    ],
    "observation_mode": ["ObservationMode"],
    "options": ["JsonlRecordingOptions", "RecordingOptions", "RenderOptions", "TelemetryOptions"],
    "pedestrian_env": ["PedestrianEnv", "_reward_function_name"],
    "reset_metadata": ["build_reset_metadata", "resolve_map_id"],
    "reward": [
        "_ROUTE_COMPLETION_V2_WEIGHTS",
        "_ROUTE_COMPLETION_V3_WEIGHTS",
        "_SOCIAL_QUALITY_V1_WEIGHTS",
        "build_reward_curriculum_function",
        "build_reward_function",
        "punish_action_reward",
        "route_completion_v2_reward",
        "route_completion_v3_reward",
        "simple_ped_reward",
        "simple_reward",
        "snqi_step_reward",
        "social_quality_v1_reward",
        "stationary_collision_ped_reward",
    ],
    "reward_alyassi": [
        "AlyassiRewardWeights",
        "alyassi_component_citations",
        "alyassi_component_scores",
        "alyassi_reward",
    ],
    "robot_env": [
        "EnvSettings",
        "RobotEnv",
        "VisualizableSimState",
        "_FlatteningObservationWrapper",
        "_attach_goal_posterior_planner_input",
        "_build_goal_posterior_planner_input",
        "_build_step_info",
        "_flatten_nested_dict_obs",
        "_flatten_nested_dict_spaces",
        "_flatten_occupancy_grid_metadata",
        "_make_telemetry_run_id",
        "_stable_config_hash",
    ],
    "robot_env_with_image": ["RobotEnvWithImage"],
    "robot_env_with_pedestrian_obstacle_forces": ["RobotEnvWithPedestrianObstacleForces"],
    "snqi_proxy": [
        "DEFAULT_ROBOT_RADIUS",
        "StepSNQIProxy",
        "StepSNQIProxyState",
        "_resolve_robot_radius",
        "coerce_xy_rows",
        "compute_snqi_step_proxies",
        "extract_robot_xy",
        "resolve_snqi_thresholds",
    ],
    "telemetry_config": ["TelemetryConfigMixin"],
    "unified_config": [
        "BaseSimulationConfig",
        "EnvSettings",
        "GridConfig",
        "ImageRobotConfig",
        "MultiRobotConfig",
        "ObservationVisibilitySettings",
        "PedestrianSimulationConfig",
        "RobotSimulationConfig",
        "sync_pedestrian_obstacle_force_alias",
    ],
}

# Names whose defining module differs from the exporting module, or that are data
# values without ``__module__``/``__qualname__``. Everything else must define the
# name in its own module with ``__qualname__ == name``.
_EXPECTED_IDENTITIES: dict[tuple[str, str], tuple[str, str] | None] = {
    ("env_config", "BaseSimulationConfig"): (
        "robot_sf.gym_env.unified_config",
        "BaseSimulationConfig",
    ),
    ("env_config", "BicycleDriveRobot"): ("robot_sf.robot.bicycle_drive", "BicycleDriveRobot"),
    ("env_config", "BicycleDriveSettings"): (
        "robot_sf.robot.bicycle_drive",
        "BicycleDriveSettings",
    ),
    ("env_config", "DifferentialDriveRobot"): (
        "robot_sf.robot.differential_drive",
        "DifferentialDriveRobot",
    ),
    ("env_config", "DifferentialDriveSettings"): (
        "robot_sf.robot.differential_drive",
        "DifferentialDriveSettings",
    ),
    ("env_config", "EnvSettingsNew"): ("robot_sf.gym_env.unified_config", "RobotSimulationConfig"),
    ("env_config", "LidarScannerSettings"): (
        "robot_sf.sensor.range_sensor",
        "LidarScannerSettings",
    ),
    ("env_config", "MapDefinitionPool"): ("robot_sf.nav.map_config", "MapDefinitionPool"),
    ("env_config", "PedEnvSettingsNew"): (
        "robot_sf.gym_env.unified_config",
        "PedestrianSimulationConfig",
    ),
    ("env_config", "RobotEnvSettingsNew"): ("robot_sf.gym_env.unified_config", "ImageRobotConfig"),
    ("env_config", "SimulationSettings"): ("robot_sf.sim.sim_config", "SimulationSettings"),
    ("env_registry", "BETA"): None,
    ("env_registry", "EXPERIMENTAL"): None,
    ("env_registry", "STABLE"): None,
    ("environment_factory", "JsonlRecordingOptions"): (
        "robot_sf.gym_env.options",
        "JsonlRecordingOptions",
    ),
    ("environment_factory", "RecordingOptions"): ("robot_sf.gym_env.options", "RecordingOptions"),
    ("environment_factory", "RenderOptions"): ("robot_sf.gym_env.options", "RenderOptions"),
    ("environment_factory", "TelemetryOptions"): ("robot_sf.gym_env.options", "TelemetryOptions"),
    ("robot_env", "EnvSettings"): ("robot_sf.gym_env.env_config", "EnvSettings"),
    ("robot_env", "VisualizableSimState"): ("robot_sf.render.sim_state", "VisualizableSimState"),
    ("reward", "_ROUTE_COMPLETION_V2_WEIGHTS"): None,
    ("reward", "_ROUTE_COMPLETION_V3_WEIGHTS"): None,
    ("reward", "_SOCIAL_QUALITY_V1_WEIGHTS"): None,
    ("snqi_proxy", "DEFAULT_ROBOT_RADIUS"): None,
    ("unified_config", "GridConfig"): ("robot_sf.nav.occupancy_grid", "GridConfig"),
}

_ALL_EXPORT_PAIRS = [(module, name) for module, names in EXPECTED_ALL.items() for name in names]


def _import_gym_env_module(module: str) -> object:
    return importlib.import_module(f"robot_sf.gym_env.{module}")


@pytest.mark.parametrize("module", sorted(EXPECTED_ALL))
def test_gym_env_module_declares_reviewed_export_surface(module: str) -> None:
    """Each gym_env module exports exactly its reviewed ``__all__``."""
    mod = _import_gym_env_module(module)
    assert mod.__all__ == EXPECTED_ALL[module]
    assert set(mod.__all__) <= set(dir(mod))


@pytest.mark.parametrize("module,name", _ALL_EXPORT_PAIRS)
def test_gym_env_export_resolves_on_pre_change_path(module: str, name: str) -> None:
    """Every declared export resolves with its pre-change identity."""
    mod = _import_gym_env_module(module)
    export = getattr(mod, name)
    expected = _EXPECTED_IDENTITIES.get((module, name), (f"robot_sf.gym_env.{module}", name))
    if expected is None:
        assert export is not None
        return
    expected_module, expected_qualname = expected
    assert export.__module__ == expected_module
    assert export.__qualname__ == expected_qualname


def test_gym_env_reviewed_names_all_reported_in_the_contract() -> None:
    """The contract table and the live modules stay in sync."""
    live = {module: _import_gym_env_module(module).__all__ for module in EXPECTED_ALL}
    assert live == EXPECTED_ALL
