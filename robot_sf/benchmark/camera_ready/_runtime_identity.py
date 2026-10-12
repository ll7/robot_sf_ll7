"""Bind campaign rows to the runner's per-scenario planner configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.camera_ready._config import (
    _apply_fixed_campaign_horizon,
    _scenario_with_kinematics,
)

if TYPE_CHECKING:
    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

from robot_sf.benchmark.map_runner.map_runner_identity import (
    _historical_authored_identity,
    _scenario_with_episode_seed_defaults,
    planner_independent_scenario_case_payload,
)
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    resolve_episode_policy_runtime,
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
    return _apply_fixed_campaign_horizon(
        scoped,
        horizon=horizon,
        horizon_policy=cfg.horizon_policy,
        protocol_version=cfg.protocol_version,
    )


def expected_runtime_identity(
    planner: Mapping[str, Any],
    scenario: Mapping[str, Any],
    seed: int,
    *,
    config_root: Path | None = None,
) -> tuple[str, str]:
    """Resolve the episode identity using repo-relative or explicit bundle config paths.

    ``config_root`` anchors both the planner path and repository-relative candidate
    references. It defaults to the checkout containing this module, never the cwd.

    Returns:
        Effective algorithm name and planner config hash.
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
    return algo, _config_hash(config)


def row_runtime_identity_errors(
    record: Mapping[str, Any],
    expected: tuple[str, str],
    *,
    scenario: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Report mismatched recorded algorithms, planner hashes and scenario hashes.

    The episode config hash signs scenario_params, whose algo_config_hash binds
    the effective planner config. Bind the signed scenario to the selected input,
    then check both hash layers and every recorded copy.
    algorithm_metadata describes the constructed adapter (which may expand config
    defaults or report its PPO child), so it is not the episode identity contract.

    Returns:
        Mismatched field names with expected and recorded values.
    """
    algo, config_hash = expected
    params = record.get("scenario_params")
    if not isinstance(params, Mapping):
        return {"scenario_params": "missing"}
    provenance = record.get("result_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    expected_fields = {
        "scenario_params.algo": (params.get("algo"), algo),
        "scenario_params.algo_config_hash": (params.get("algo_config_hash"), config_hash),
        "config_hash": (record.get("config_hash"), _config_hash(params)),
    }
    if "map_file" in scenario or "simulation_config" in scenario:
        selected = planner_independent_scenario_case_payload(
            _historical_authored_identity(dict(scenario)), seed=seed
        )
        saved_params = _historical_authored_identity(dict(params))
        saved = planner_independent_scenario_case_payload(saved_params, seed=seed)
        # Saved rows already contain runner-seeded inputs. Do not synthesize missing
        # simulator defaults when validating the producer; absence must stay visible.
        saved["simulation_config"] = saved_params.get("simulation_config")
        expected_fields["scenario_params.scenario_identity"] = (
            _config_hash(saved),
            _config_hash(selected),
        )
    for label, mapping, field, value in (
        ("algo", record, "algo", algo),
        ("result_provenance.config_hash", provenance, "config_hash", _config_hash(params)),
    ):
        if field in mapping:
            expected_fields[label] = (mapping[field], value)
    return {
        label: {"observed": observed, "expected": value}
        for label, (observed, value) in expected_fields.items()
        if observed != value
    }
