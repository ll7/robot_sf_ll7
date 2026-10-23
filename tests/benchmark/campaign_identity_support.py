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


def producer_record_identity(record, *, telemetry_profile=None, observation_noise=None):
    """Complete a fresh fixture with the real producer's declared identity copies.

    Returns:
        Producer-shaped row, before the test introduces any defect.
    """
    from robot_sf.benchmark.map_runner.map_runner_identity import _compute_map_episode_id
    from robot_sf.benchmark.observation_noise import (
        normalize_observation_noise_spec,
        observation_noise_hash,
    )
    from robot_sf.benchmark.provenance_identity import publish_episode_identity
    from robot_sf.benchmark.result_provenance import build_simulator_settings_provenance
    from robot_sf.benchmark.tracking_precision_contract import (
        normalize_tracking_precision_spec,
        tracking_precision_hash,
    )
    from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback

    params = record["scenario_params"]
    record["algo"] = params["algo"]
    record["config_hash"] = _config_hash(params)
    record["episode_id"] = _compute_map_episode_id(params, record["seed"])
    record.setdefault("git_hash", _git_hash_fallback())
    record.setdefault("observation_noise", normalize_observation_noise_spec(observation_noise))
    record["observation_noise_hash"] = observation_noise_hash(record["observation_noise"])
    record.setdefault("tracking_precision", normalize_tracking_precision_spec(None))
    record["tracking_precision_hash"] = tracking_precision_hash(record["tracking_precision"])
    if telemetry_profile is not None:
        record.setdefault("algorithm_metadata", {})["telemetry"] = telemetry_profile
    settings = build_simulator_settings_provenance(
        horizon=params.get("run_horizon")
        or params.get("simulation_config", {}).get("max_episode_steps")
        or 200,
        dt=params.get("run_dt")
        or params.get("simulation_config", {}).get("time_per_step_in_secs")
        or 0.1,
        record_forces=params.get("record_forces", False),
        active_observation_mode=params["observation_mode"],
        active_observation_level=params["observation_level"],
        noise_hash=record["observation_noise_hash"],
        tracking_precision_hash=record["tracking_precision_hash"],
    )
    publish_episode_identity(record, simulator_settings=settings)
    return record
