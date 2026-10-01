"""Bind campaign rows to the runner's per-scenario planner configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner.map_runner_identity import _scenario_with_episode_seed_defaults
from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
    resolve_episode_policy_runtime,
)
from robot_sf.benchmark.utils import _config_hash
from robot_sf.common.artifact_paths import get_repository_root


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
    record: Mapping[str, Any], expected: tuple[str, str]
) -> dict[str, Any]:
    """Report mismatched recorded algorithms, planner hashes and scenario hashes.

    The episode config hash signs scenario_params, whose algo_config_hash binds
    the effective planner config. Check both layers and every recorded copy.
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
