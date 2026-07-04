"""Readiness checks for issue #3300 stronger false-positive replay matrices.

This module validates the predeclared matrix/config contract only. It does not run a
benchmark campaign or promote robustness claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.camera_ready import load_campaign_config
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.observation_noise import (
    apply_observation_noise,
    make_observation_noise_rng,
)

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig

ISSUE = 3300
SCHEMA_VERSION = "false_positive_actor_injection_matrix_readiness.v1"
STATUS_READY = "ready"
STATUS_BLOCKED = "blocked"
REQUIRED_OBSERVATION_MODE = "socnav_state"
REQUIRED_OBSERVATION_NOISE_PROFILE = "issue_3300_false_positive_actor_injection_v1"
DEFAULT_MIN_SCENARIOS = 2
DEFAULT_MIN_SEEDS = 2


@dataclass(frozen=True)
class FalsePositiveMatrixReadiness:
    """Structured verdict for a paired #3300 false-positive replay matrix."""

    status: str
    blockers: list[str] = field(default_factory=list)
    scenario_ids: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    planner_observation_modes: list[str] = field(default_factory=list)
    pedestrian_scenario_ids: list[str] = field(default_factory=list)
    injection_probe: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Return True only when the paired matrix satisfies the #3300 contract."""

        return self.status == STATUS_READY

    def to_dict(self) -> dict[str, Any]:
        """Serialize the readiness verdict for CLI and PR evidence.

        Returns:
            JSON-serializable readiness payload.
        """

        return {
            "schema_version": SCHEMA_VERSION,
            "issue": ISSUE,
            "status": self.status,
            "blockers": list(self.blockers),
            "scenario_ids": list(self.scenario_ids),
            "seeds": list(self.seeds),
            "planner_observation_modes": list(self.planner_observation_modes),
            "pedestrian_scenario_ids": list(self.pedestrian_scenario_ids),
            "injection_probe": dict(self.injection_probe),
        }


@dataclass(frozen=True)
class _ReadinessInputs:
    nominal_cfg: CampaignConfig
    perturbed_cfg: CampaignConfig
    nominal_ids: Sequence[str]
    perturbed_ids: Sequence[str]
    scenario_ids: Sequence[str]
    seeds: Sequence[int]
    planner_modes: Sequence[str]
    pedestrian_scenario_ids: Sequence[str]
    injection_probe: Mapping[str, Any]


def check_false_positive_matrix_readiness(
    nominal_config_path: Path,
    perturbed_config_path: Path,
    *,
    min_scenarios: int = DEFAULT_MIN_SCENARIOS,
    min_seeds: int = DEFAULT_MIN_SEEDS,
) -> FalsePositiveMatrixReadiness:
    """Validate paired #3300 stronger-matrix configs fail closed before replay.

    Args:
        nominal_config_path: Camera-ready nominal replay config.
        perturbed_config_path: Camera-ready false-positive replay config.
        min_scenarios: Minimum predeclared scenario count.
        min_seeds: Minimum fixed seed count.

    Returns:
        Readiness verdict with actionable blockers when the matrix is too weak.
    """

    nominal_cfg = load_campaign_config(nominal_config_path)
    perturbed_cfg = load_campaign_config(perturbed_config_path)

    nominal_scenarios = _load_campaign_scenarios(nominal_cfg)
    perturbed_scenarios = _load_campaign_scenarios(perturbed_cfg)
    nominal_ids = _scenario_ids(nominal_scenarios)
    perturbed_ids = _scenario_ids(perturbed_scenarios)
    scenario_ids = nominal_ids

    seeds = _fixed_seeds(perturbed_cfg)
    planner_modes = sorted(
        {mode for cfg in (nominal_cfg, perturbed_cfg) for mode in _planner_observation_modes(cfg)}
    )
    pedestrian_scenario_ids = [
        scenario_id
        for scenario_id, scenario in zip(scenario_ids, nominal_scenarios, strict=True)
        if _scenario_has_pedestrians(scenario)
    ]
    injection_probe = _structured_injection_probe(
        perturbed_cfg.observation_noise
        if isinstance(perturbed_cfg.observation_noise, Mapping)
        else None
    )
    blockers = _readiness_blockers(
        _ReadinessInputs(
            nominal_cfg=nominal_cfg,
            perturbed_cfg=perturbed_cfg,
            nominal_ids=nominal_ids,
            perturbed_ids=perturbed_ids,
            scenario_ids=scenario_ids,
            seeds=seeds,
            planner_modes=planner_modes,
            pedestrian_scenario_ids=pedestrian_scenario_ids,
            injection_probe=injection_probe,
        ),
        min_scenarios=min_scenarios,
        min_seeds=min_seeds,
    )

    return FalsePositiveMatrixReadiness(
        status=STATUS_BLOCKED if blockers else STATUS_READY,
        blockers=blockers,
        scenario_ids=scenario_ids,
        seeds=seeds,
        planner_observation_modes=planner_modes,
        pedestrian_scenario_ids=pedestrian_scenario_ids,
        injection_probe=injection_probe,
    )


def _readiness_blockers(
    inputs: _ReadinessInputs,
    *,
    min_scenarios: int,
    min_seeds: int,
) -> list[str]:
    """Return fail-closed blockers for the paired #3300 matrix."""

    blockers: list[str] = []
    if inputs.nominal_ids != inputs.perturbed_ids:
        blockers.append(
            "nominal and perturbed configs must resolve the same ordered scenario matrix"
        )
    if len(set(inputs.scenario_ids)) < min_scenarios:
        blockers.append(
            f"stronger matrix requires at least {min_scenarios} distinct scenarios; "
            f"found {len(set(inputs.scenario_ids))}"
        )
    if _fixed_seeds(inputs.nominal_cfg) != list(inputs.seeds):
        blockers.append("nominal and perturbed configs must use the same fixed seed list")
    if len(set(inputs.seeds)) < min_seeds:
        blockers.append(
            f"stronger matrix requires at least {min_seeds} distinct fixed seeds; "
            f"found {len(set(inputs.seeds))}"
        )
    blockers.extend(_planner_mode_blockers(inputs.planner_modes))
    blockers.extend(_observation_noise_blockers(inputs.nominal_cfg, inputs.perturbed_cfg))
    if not inputs.pedestrian_scenario_ids:
        blockers.append("matrix must include at least one pedestrian-bearing scenario")
    if int(inputs.injection_probe.get("pedestrians_added", 0)) <= 0:
        blockers.append(
            "false-positive profile must add at least one actor to a structured "
            "pedestrian observation probe"
        )
    return blockers


def _planner_mode_blockers(planner_modes: Sequence[str]) -> list[str]:
    non_structured = [mode for mode in planner_modes if mode != REQUIRED_OBSERVATION_MODE]
    if not non_structured:
        return []
    return [
        "all #3300 false-positive matrix planners must use "
        f"{REQUIRED_OBSERVATION_MODE}; found {non_structured}"
    ]


def _observation_noise_blockers(
    nominal_cfg: CampaignConfig,
    perturbed_cfg: CampaignConfig,
) -> list[str]:
    blockers: list[str] = []
    profile = None
    if isinstance(perturbed_cfg.observation_noise, Mapping):
        profile = perturbed_cfg.observation_noise.get("profile")
    if profile != REQUIRED_OBSERVATION_NOISE_PROFILE:
        blockers.append(
            "perturbed config must reference the issue #3300 false-positive "
            f"profile {REQUIRED_OBSERVATION_NOISE_PROFILE}"
        )
    if nominal_cfg.observation_noise is not None:
        blockers.append("nominal config must leave observation_noise disabled")
    return blockers


def _scenario_ids(scenarios: Sequence[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for scenario in scenarios:
        value = scenario.get("name")
        if value is None:
            value = scenario.get("scenario_id")
        if value is None:
            value = scenario.get("id")
        ids.append(str(value) if value is not None else "unknown")
    return ids


def _fixed_seeds(cfg: CampaignConfig) -> list[int]:
    if cfg.seed_policy.mode != "fixed-list":
        return []
    return [int(seed) for seed in cfg.seed_policy.seeds]


def _planner_observation_modes(cfg: CampaignConfig) -> list[str]:
    modes: list[str] = []
    for planner in cfg.planners:
        mode = planner.observation_mode or cfg.observation_mode
        if mode:
            modes.append(str(mode))
    return modes


def _scenario_has_pedestrians(scenario: Mapping[str, Any]) -> bool:
    if scenario.get("single_pedestrians"):
        return True
    sim_config = scenario.get("simulation_config")
    if isinstance(sim_config, Mapping):
        try:
            return float(sim_config.get("ped_density", 0.0)) > 0.0
        except (TypeError, ValueError):
            return False
    return False


def _structured_injection_probe(noise_spec: Mapping[str, Any] | None) -> dict[str, Any]:
    if noise_spec is None:
        return {"pedestrians_added": 0, "steps_with_noise": 0}
    noise_dict = dict(noise_spec)
    observation = {
        "robot": {"position": [1.0, 2.0]},
        "pedestrians_positions": [[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        "pedestrians_velocities": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "pedestrians_count": [1.0],
    }
    rng = make_observation_noise_rng(
        noise_dict,
        seed=0,
        scenario_id="issue_3300_matrix_readiness_probe",
    )
    noisy, stats = apply_observation_noise(observation, noise_dict, rng)
    return {
        "pedestrians_added": int(stats.get("pedestrians_added", 0)),
        "steps_with_noise": int(stats.get("steps_with_noise", 0)),
        "pedestrians_count": _first_int(noisy.get("pedestrians_count")),
    }


def _first_int(value: Any) -> int:
    if isinstance(value, list | tuple) and value:
        return int(value[0])
    return int(value or 0)
