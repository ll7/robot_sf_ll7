"""Bind campaign rows to the runner's per-scenario planner configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.algorithm_metadata import (
    enrich_algorithm_metadata,
    resolve_learned_checkpoint_observation_contract,
)
from robot_sf.benchmark.analysis_trace import normalize_telemetry_profile
from robot_sf.benchmark.camera_ready._config import (
    _apply_fixed_campaign_horizon,
    _scenario_with_kinematics,
)
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec
from robot_sf.benchmark.map_runner.map_runner_identity import (
    _scenario_identity_payload,
    _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    resolve_episode_policy_runtime,
)
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.result_provenance import build_simulator_settings_provenance
from robot_sf.benchmark.tracking_precision_contract import (
    normalize_tracking_precision_spec,
    tracking_precision_hash,
)
from robot_sf.benchmark.utils import _config_hash
from robot_sf.common.artifact_paths import get_repository_root


def expected_arm_scenarios(
    scenarios: list[dict[str, Any]], *, cfg: CampaignConfig, planner: PlannerSpec, kinematics: str
) -> list[dict[str, Any]]:
    """Build the same scenario inputs for dispatch, resume and final admission.

    Returns:
        Kinematics, telemetry and horizon-bound scenario inputs for this arm.
    """
    horizon = planner.horizon_override if planner.horizon_override is not None else cfg.horizon
    scoped = []
    for scenario in scenarios:
        selected = _scenario_with_kinematics(
            scenario, kinematics=kinematics, holonomic_command_mode=cfg.holonomic_command_mode
        )
        if cfg.telemetry is not None:
            selected["telemetry"] = dict(cfg.telemetry)
        scoped.append(selected)
    if cfg.telemetry is not None:
        for selected in scoped:
            selected.setdefault("metadata", {})["telemetry"] = normalize_telemetry_profile(
                cfg.telemetry
            ).to_mapping()
    return _apply_fixed_campaign_horizon(
        scoped,
        horizon=horizon,
        horizon_policy=cfg.horizon_policy,
        protocol_version=cfg.protocol_version,
    )


def _resolved_policy(
    planner: Mapping[str, Any],
    scenario: Mapping[str, Any],
    seed: int,
    *,
    config_root: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve the episode identity using repo-relative or explicit bundle config paths.

    ``config_root`` anchors both the planner path and repository-relative candidate
    references. It defaults to the checkout containing this module, never the cwd.

    Returns:
        Effective algorithm name and resolved planner configuration.
    """
    algo = planner.get("algo")
    if not isinstance(algo, str) or not algo.strip():
        raise ValueError("planner is missing its expected algorithm")
    root = (config_root or get_repository_root()).resolve()
    path = planner.get("algo_config_path") or planner.get("algo_config")
    if path:
        path = Path(str(path))
        if not path.is_absolute():
            path = root / path
    algo, config = resolve_episode_policy_runtime(
        default_algo=algo,
        algo_config_path=str(path) if path else None,
        scenario=_scenario_with_episode_seed_defaults(dict(scenario), seed=seed),
        seed=seed,
        config_root=root,
    )
    return algo, config


# Closed runner-envelope inventory. Authored scenario keys (including all nested
# robot, pedestrian, goal and simulator settings) are allowed only when present
# in the selected producer payload, and are compared without a projection.
RUNTIME_IDENTITY_FIELDS = frozenset(
    {
        "algo",
        "algo_config_hash",
        "run_horizon",
        "run_dt",
        "record_forces",
        "observation_mode",
        "observation_level",
        "benchmark_track",
        "track_schema_version",
        "observation_noise_profile",
        "observation_noise_hash",
        "tracking_precision",
        "tracking_precision_hash",
        "synthetic_actuation_profile",
        "latency_stress_profile",
        "safety_wrapper",
        "cbf_safety_filter",
        "record_planner_decision_trace",
        "record_simulation_step_trace",
    }
)
IDENTITY_EXCLUSIONS = {
    "scenario.seed/seeds": "Producer removes inventories; exact integer row seed is checked separately.",
    "algorithm_metadata": "Adapter diagnostics may expand defaults or describe a child planner; signed algo_config_hash binds the selected policy.",
    "git_hash/result_provenance.repo_commit": "Software provenance is checked separately against the campaign manifest by final integrity.",
    "result_provenance.schema_version/episode_id/raw_artifact/jsonl_line/postprocessing": "Envelope and artifact lineage, not runtime settings; provenance validation checks artifact evidence separately.",
    "metrics/outcome/trajectories/timing": "Measured outputs, not runtime configuration; behaviour receipts provide execution evidence.",
}


@dataclass(frozen=True)
class ExpectedEpisodeIdentity:
    """Trusted complete producer identity; never derived from a saved row."""

    scenario_id: str
    seed: int
    params: dict[str, Any]
    optional_copies: dict[str, Any]
    simulator_settings: dict[str, Any]


def expected_runtime_identity(
    planner: Mapping[str, Any],
    scenario: Mapping[str, Any],
    seed: int,
    *,
    config_root: Path | None = None,
) -> tuple[str, str]:
    """Return the resolved algorithm/hash pair for configuration inspection."""
    algo, config = _resolved_policy(planner, scenario, seed, config_root=config_root)
    return algo, _config_hash(config)


def selected_campaign_planner(
    cfg: CampaignConfig | None, planner: Mapping[str, Any]
) -> PlannerSpec:
    """Resolve a selected arm using trusted campaign values, never saved overrides.

    Returns:
        The canonical selected planner specification.
    """
    if not isinstance(cfg, CampaignConfig):
        raise ValueError("complete campaign configuration is required for runtime identity")
    spec = next((p for p in cfg.planners if p.key == planner.get("key")), None)
    if spec is None:
        raise ValueError("planner is not selected by the campaign")
    if planner.get("kinematics") not in cfg.kinematics_matrix:
        raise ValueError("kinematics is not selected by the campaign")
    return spec


def expected_episode_identity(
    planner: Mapping[str, Any],
    scenario: Mapping[str, Any],
    seed: int,
    *,
    cfg: CampaignConfig | None,
    config_root: Path | None = None,
) -> ExpectedEpisodeIdentity:
    """Build every signed runtime field from the selected campaign and planner.

    Missing campaign context is an admission error. Optional fields must have the
    exact presence chosen by the producer; absence never means ignore a field.

    Returns:
        Complete selected runtime identity.
    """
    spec = selected_campaign_planner(cfg, planner)
    assert cfg is not None
    selected = _scenario_with_episode_seed_defaults(dict(scenario), seed=seed)
    algo, config = _resolved_policy(
        {"algo": spec.algo, "algo_config_path": spec.algo_config_path},
        selected,
        seed,
        config_root=config_root,
    )
    observation = resolve_learned_checkpoint_observation_contract(
        algo,
        config,
        observation_mode=spec.observation_mode or cfg.observation_mode,
    )
    mode = str(observation["active_observation_mode"])
    contract = enrich_algorithm_metadata(
        algo=algo,
        metadata={},
        observation_mode=mode,
        observation_level=observation.get("observation_level_key"),
    )
    horizon = spec.horizon_override if spec.horizon_override is not None else cfg.horizon
    dt = spec.dt_override if spec.dt_override is not None else cfg.dt
    effective_dt = (
        dt
        if dt is not None and dt > 0
        else selected.get("simulation_config", {}).get("time_per_step_in_secs", 0.1)
    )
    wrapper = spec.safety_wrapper if spec.safety_wrapper is not None else cfg.safety_wrapper
    params = _scenario_identity_payload(
        selected,
        algo=algo,
        algo_config=config,
        horizon=horizon,
        dt=dt,
        record_forces=cfg.record_forces,
        observation_mode=mode,
        observation_level=str(contract["observation_level"]["key"]),
        observation_noise=cfg.observation_noise,
        synthetic_actuation_profile=cfg.synthetic_actuation_profile.to_metadata()
        if cfg.synthetic_actuation_profile is not None
        else None,
        latency_stress_profile=cfg.latency_stress_profile.to_metadata(dt=effective_dt)
        if cfg.latency_stress_profile is not None
        else None,
        safety_wrapper=wrapper,
        record_planner_decision_trace=cfg.record_planner_decision_trace,
        record_simulation_step_trace=cfg.record_simulation_step_trace,
    )
    unknown = set(params) - (set(selected) | {"id"} | RUNTIME_IDENTITY_FIELDS)
    if unknown:
        raise ValueError("producer runtime fields lack an explicit admission decision")
    effective_horizon = (
        int(horizon)
        if horizon is not None and horizon > 0
        else int(selected.get("simulation_config", {}).get("max_episode_steps") or 200)
    )
    noise_hash = observation_noise_hash(normalize_observation_noise_spec(cfg.observation_noise))
    precision_hash = tracking_precision_hash(normalize_tracking_precision_spec(None))
    return ExpectedEpisodeIdentity(
        str(selected.get("name") or selected.get("scenario_id") or selected.get("id") or "unknown"),
        seed,
        params,
        {
            **params,
            "horizon": effective_horizon,
            "dt": float(effective_dt),
            "observation_noise": normalize_observation_noise_spec(cfg.observation_noise),
            "observation_noise_hash": noise_hash,
            "tracking_precision": normalize_tracking_precision_spec(None),
            "tracking_precision_hash": precision_hash,
        },
        build_simulator_settings_provenance(
            horizon=effective_horizon,
            dt=float(effective_dt),
            record_forces=cfg.record_forces,
            active_observation_mode=mode,
            active_observation_level=str(contract["observation_level"]["key"]),
            noise_hash=noise_hash,
            tracking_precision_hash=precision_hash,
        ),
    )


def _payload_errors(
    params: Mapping[str, Any], expected: Mapping[str, Any], *, prefix: str = "scenario_params"
) -> dict[str, Any]:
    """Return exact presence/value differences across the closed producer payload.

    Returns:
        Field-specific mismatches, without observed seed/config disclosure.
    """
    errors: dict[str, Any] = {}
    for field in sorted(set(params) | set(expected)):
        if field not in expected:
            errors[f"{prefix}.{field}"] = "unexpected field"
        elif field not in params:
            errors[f"{prefix}.{field}"] = "missing field"
        elif json.dumps(params[field], sort_keys=True) != json.dumps(
            expected[field], sort_keys=True
        ):
            errors[f"{prefix}.{field}"] = "different from selected campaign"
    return errors


def _runtime_copy_errors(record: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Check optional schema mirrors against the authoritative signed configuration.

    Returns:
        Conflicting runtime-label fields, including unsigned row-level copies.
    """
    errors: dict[str, Any] = {}
    for field in RUNTIME_IDENTITY_FIELDS | {"horizon", "dt", "observation_noise"}:
        if field in record and json.dumps(record[field], sort_keys=True) != json.dumps(
            expected.get(field), sort_keys=True
        ):
            errors[field] = "different from selected runtime copy"
    return errors


def row_runtime_identity_errors(
    record: Mapping[str, Any],
    expected: ExpectedEpisodeIdentity,
) -> dict[str, Any]:
    """Compare the closed complete identity, including missing and extra fields.

    No signed scenario field is excluded. Unknown runner-envelope additions are
    refused just like unknown authored keys. Saved metadata is never normalized.

    Returns:
        Structured field mismatches; an empty mapping means the identity matches.
    """
    params = record.get("scenario_params")
    if not isinstance(params, Mapping):
        return {"scenario_params": "missing"}
    errors: dict[str, Any] = {}
    if "metadata" in params and not isinstance(params["metadata"], Mapping):
        errors["scenario_params.metadata"] = "must be a mapping"
    errors.update(_payload_errors(params, expected.params))
    if type(record.get("seed")) is not int or record["seed"] != expected.seed:
        errors["seed"] = "different from selected integer seed"
    for field, value in (
        ("scenario_id", expected.scenario_id),
        ("algo", expected.params["algo"]),
        ("config_hash", _config_hash(expected.params)),
    ):
        if record.get(field) != value:
            errors[field] = "missing or different from selected campaign"
    errors.update(_runtime_copy_errors(record, expected.optional_copies))
    if "result_provenance" in record:
        errors.update(_provenance_runtime_identity_errors(record["result_provenance"], expected))
    return errors


def _provenance_runtime_identity_errors(
    provenance: Any, expected: ExpectedEpisodeIdentity
) -> dict[str, Any]:
    """Refuse any conflicting saved provenance copy of the trusted runtime identity.

    Returns:
        Exact copy mismatches, or a malformed-block error.
    """
    if not isinstance(provenance, Mapping):
        return {"result_provenance": "must be a mapping"}
    errors: dict[str, Any] = {}
    # Provenance is another saved copy, never an authority for resolving runtime.
    errors.update(
        {
            f"result_provenance.{field}": error
            for field, error in _runtime_copy_errors(provenance, expected.optional_copies).items()
        }
    )
    if "simulator_settings" in provenance:
        settings = provenance["simulator_settings"]
        if not isinstance(settings, Mapping):
            errors["result_provenance.simulator_settings"] = "must be a mapping"
        else:
            errors.update(
                _payload_errors(
                    settings,
                    expected.simulator_settings,
                    prefix="result_provenance.simulator_settings",
                )
            )
    for field, value in (
        ("scenario_id", expected.scenario_id),
        ("seed", expected.seed),
        ("config_hash", _config_hash(expected.params)),
    ):
        if (
            field not in provenance
            or type(provenance[field]) is not type(value)
            or provenance[field] != value
        ):
            errors[f"result_provenance.{field}"] = "missing or different from selected campaign"
    return errors
