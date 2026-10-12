"""Real producer-shaped fixtures for complete campaign identity checks."""

from pathlib import Path
from typing import Any

from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    resolve_learned_checkpoint_observation_contract,
)
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec
from robot_sf.benchmark.map_runner.map_runner_identity import (
    _scenario_identity_payload,
    _scenario_with_episode_seed_defaults,
)


def fixture_campaign(planners, *, horizon=None, dt=None, kinematics=("differential_drive",)):
    """Return explicitly trusted settings for a deterministic fixture campaign.

    Returns:
        Fixture campaign configuration.
    """
    return CampaignConfig(
        name="fixture",
        scenario_matrix_path=Path("scenarios.yaml"),
        planners=tuple(
            PlannerSpec(
                key=p["key"],
                algo=p.get("algo", "test"),
                algo_config_path=Path(p["algo_config_path"]) if p.get("algo_config_path") else None,
            )
            for p in planners
        ),
        horizon=horizon,
        dt=dt,
        record_forces=False,
        kinematics_matrix=kinematics,
    )


def producer_params(scenario, seed, *, algo, algo_config=None, **options: Any):
    """Build a producer payload before a test mutates it; never consult validation.

    Returns:
        Signed scenario parameters from the real producer.
    """
    config = algo_config or {}
    observation = resolve_learned_checkpoint_observation_contract(
        algo,
        config,
        observation_mode=options.get("observation_mode"),
        observation_level=options.get("observation_level"),
    )
    mode = str(observation["active_observation_mode"])
    contract = enrich_algorithm_metadata(
        algo=algo,
        metadata={},
        observation_mode=mode,
        observation_level=observation.get("observation_level_key"),
    )
    return _scenario_identity_payload(
        _scenario_with_episode_seed_defaults(scenario, seed=seed),
        algo=algo,
        algo_config=config,
        horizon=options.get("horizon"),
        dt=options.get("dt"),
        record_forces=options.get("record_forces", False),
        observation_mode=mode,
        observation_level=str(contract["observation_level"]["key"]),
        observation_noise=options.get("observation_noise"),
        safety_wrapper=options.get("safety_wrapper"),
        record_planner_decision_trace=options.get("record_planner_decision_trace", False),
        record_simulation_step_trace=options.get("record_simulation_step_trace", False),
    )
