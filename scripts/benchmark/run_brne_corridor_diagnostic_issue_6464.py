#!/usr/bin/env python3
"""Run the approved corridor-only BRNE diagnostic preflight (#6464).

The harness executes BRNE, ORCA, and social-force on the same declared
scenario/seed cells through the map runner. It records native/degraded
eligibility, goal reaching, trace-backed non-degenerate motion, and corridor
violations. It never ranks planners and never treats fallback/degraded rows as
BRNE evidence.

Example::

    uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \\
        --config configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml

The pinned BRNE source is local-only and must be staged before execution with
``scripts/tools/manage_external_repos.py stage brne``. Missing dependencies are
reported as unavailable rather than substituted.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from robot_sf.baselines.brne import BRNE_PINNED_SHA
from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml"
EXPECTED_PLANNERS = ("brne", "orca", "social_force")
EXPECTED_SCENARIO = "classic_head_on_corridor_low"
EXPECTED_SEEDS = (111, 112, 113)
EXPECTED_HORIZON = 500
EXPECTED_DT = 0.1
EXPECTED_SCENARIO_MATRIX = (
    REPO_ROOT / "configs/scenarios/issue_6464_brne_corridor_diagnostic.yaml"
).resolve()
EXPECTED_PLANNER_CONFIGS = {
    "brne": (REPO_ROOT / "configs/baselines/issue_6464_brne_corridor_diagnostic.yaml").resolve(),
    "orca": (REPO_ROOT / "configs/baselines/issue_6464_orca_corridor_diagnostic.yaml").resolve(),
    "social_force": (
        REPO_ROOT / "configs/baselines/issue_6464_social_force_corridor_diagnostic.yaml"
    ).resolve(),
}
EXPECTED_PLANNER_FIELDS = {
    "brne": {
        "num_samples": 49,
        "expected_effective_num_samples": 42,
        "plan_steps": 25,
        "dt": 0.1,
        "maximum_agents": 8,
        "corridor_y_min": 2.5,
        "corridor_y_max": 37.5,
        "step_budget_s": 0.1,
        "fallback_on_error": False,
        "allow_testing_algorithms": True,
        "include_in_paper": False,
    },
    "orca": {"allow_fallback": False},
    "social_force": {"action_space": "unicycle", "allow_fallback": False},
}
ZERO_MOTION_EPSILON_M = 1.0e-6


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML mapping and fail closed on malformed configuration."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    return payload


def _resolve_repo_path(value: Any, *, field: str) -> Path:
    """Resolve a repository-relative path from campaign configuration."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{field} must stay inside the repository: {value}") from exc
    return resolved


def _finite_float(value: Any, *, field: str) -> float:
    """Parse a finite float from configuration."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite: {value!r}")
    return parsed


def _validate_campaign_header(config: dict[str, Any]) -> list[int]:
    """Validate the immutable campaign identity and return its seeds."""
    if config.get("schema_version") != "brne-corridor-diagnostic.v1":
        raise ValueError("unsupported BRNE diagnostic schema_version")
    try:
        issue = int(config.get("issue", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("the diagnostic config must target issue 6464") from exc
    if issue != 6464:
        raise ValueError("the diagnostic config must target issue 6464")
    if not str(config.get("claim_boundary", "")).strip():
        raise ValueError("claim_boundary must be explicit")
    scenario_ids = config.get("scenario_ids")
    if scenario_ids != [EXPECTED_SCENARIO]:
        raise ValueError("issue #6464 diagnostic must select exactly classic_head_on_corridor_low")
    raw_seeds = config.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("seeds must be a non-empty list")
    seeds = [int(seed) for seed in raw_seeds]
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be distinct non-negative integers")
    if tuple(seeds) != EXPECTED_SEEDS:
        raise ValueError(f"issue #6464 diagnostic seeds are frozen to {list(EXPECTED_SEEDS)}")
    return seeds


def _validate_campaign_horizon(config: dict[str, Any]) -> tuple[int, float]:
    """Validate the fixed horizon and timestep."""
    horizon = int(config.get("horizon", 0))
    dt = _finite_float(config.get("dt"), field="dt")
    if horizon <= 0 or dt <= 0.0:
        raise ValueError("horizon and dt must be positive")
    if horizon != EXPECTED_HORIZON or not math.isclose(
        dt, EXPECTED_DT, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("issue #6464 diagnostic horizon and dt are frozen to 500 and 0.1")
    return horizon, dt


def _validate_corridor(config: dict[str, Any]) -> dict[str, float]:
    """Validate and normalize corridor thresholds."""
    corridor = config.get("corridor")
    if not isinstance(corridor, dict):
        raise ValueError("corridor must be a mapping")
    y_min = _finite_float(corridor.get("y_min"), field="corridor.y_min")
    y_max = _finite_float(corridor.get("y_max"), field="corridor.y_max")
    radius = _finite_float(corridor.get("robot_radius_m"), field="corridor.robot_radius_m")
    min_displacement = _finite_float(
        corridor.get("min_displacement_m"), field="corridor.min_displacement_m"
    )
    max_zero_fraction = _finite_float(
        corridor.get("max_zero_motion_fraction"), field="corridor.max_zero_motion_fraction"
    )
    if not y_min < y_max or radius < 0.0 or min_displacement < 0.0:
        raise ValueError("corridor bounds and thresholds are inconsistent")
    if not 0.0 <= max_zero_fraction <= 1.0:
        raise ValueError("corridor.max_zero_motion_fraction must be in [0, 1]")
    expected_corridor = {
        "y_min": 2.5,
        "y_max": 37.5,
        "robot_radius_m": 1.0,
        "min_displacement_m": 0.5,
        "max_zero_motion_fraction": 0.95,
    }
    if any(
        not math.isclose(value, expected_corridor[field], rel_tol=0.0, abs_tol=1.0e-12)
        for field, value in {
            "y_min": y_min,
            "y_max": y_max,
            "robot_radius_m": radius,
            "min_displacement_m": min_displacement,
            "max_zero_motion_fraction": max_zero_fraction,
        }.items()
    ):
        raise ValueError("issue #6464 diagnostic corridor thresholds are frozen")
    return {
        "y_min": y_min,
        "y_max": y_max,
        "robot_radius_m": radius,
        "min_displacement_m": min_displacement,
        "max_zero_motion_fraction": max_zero_fraction,
    }


def _validate_planner_entry(  # noqa: C901
    raw_planner: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one planner entry and load its config."""
    if not isinstance(raw_planner, dict):
        raise ValueError("each planner entry must be a mapping")
    key = str(raw_planner.get("key", "")).strip()
    algo = str(raw_planner.get("algo", "")).strip()
    if key not in EXPECTED_PLANNERS or algo != key:
        raise ValueError(f"unsupported planner entry: key={key!r}, algo={algo!r}")
    config_path = _resolve_repo_path(raw_planner.get("config_path"), field=f"{key}.config_path")
    if not config_path.is_file():
        raise FileNotFoundError(f"missing planner config: {config_path}")
    if config_path != EXPECTED_PLANNER_CONFIGS[key]:
        raise ValueError(f"{key} must use its frozen issue #6464 planner config")
    planner_config = _load_mapping(config_path)
    for field, expected in EXPECTED_PLANNER_FIELDS[key].items():
        observed = planner_config.get(field)
        matches = (
            math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-12)
            if isinstance(expected, float)
            else observed == expected
        )
        if not matches:
            raise ValueError(f"{key}.{field} does not match the frozen issue #6464 value")
    if key == "brne":
        if bool(planner_config.get("fallback_on_error", False)):
            raise ValueError("BRNE fallback_on_error must be false")
        if bool(planner_config.get("include_in_paper", False)):
            raise ValueError("BRNE include_in_paper must be false")
    if key in {"orca", "social_force"} and bool(planner_config.get("allow_fallback", False)):
        raise ValueError(f"{key} fallback must be disabled for this diagnostic")
    return {"key": key, "algo": algo, "config_path": str(config_path)}, planner_config


def _validate_planners(config: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    """Validate planner configs and return entries plus frozen BRNE limits."""
    raw_planners = config.get("planners")
    if not isinstance(raw_planners, list):
        raise ValueError("planners must be a list")
    planners: list[dict[str, Any]] = []
    keys: list[str] = []
    brne_config: dict[str, Any] | None = None
    for raw_planner in raw_planners:
        planner, planner_config = _validate_planner_entry(raw_planner)
        keys.append(str(planner["key"]))
        planners.append(planner)
        if planner["key"] == "brne":
            brne_config = planner_config
    if tuple(keys) != EXPECTED_PLANNERS:
        raise ValueError(f"planners must be exactly {EXPECTED_PLANNERS}")
    if brne_config is None:
        raise ValueError("BRNE planner config is required")
    try:
        maximum_agents = int(brne_config.get("maximum_agents", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("BRNE maximum_agents must be a positive integer") from exc
    if maximum_agents < 1:
        raise ValueError("BRNE maximum_agents must be a positive integer")
    try:
        expected_effective_num_samples = int(brne_config["expected_effective_num_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BRNE expected_effective_num_samples must be a positive integer") from exc
    if expected_effective_num_samples < 1:
        raise ValueError("BRNE expected_effective_num_samples must be a positive integer")
    return planners, maximum_agents - 1, expected_effective_num_samples


def validate_campaign_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the frozen diagnostic contract.

    Returns:
        A shallow normalized copy with resolved numeric fields and planner paths.
    """
    seeds = _validate_campaign_header(config)
    horizon, dt = _validate_campaign_horizon(config)
    corridor = _validate_corridor(config)
    planners, max_pedestrians, expected_effective_num_samples = _validate_planners(config)

    scenario_matrix = _resolve_repo_path(config.get("scenario_matrix"), field="scenario_matrix")
    if not scenario_matrix.is_file():
        raise FileNotFoundError(f"missing scenario matrix: {scenario_matrix}")
    if scenario_matrix != EXPECTED_SCENARIO_MATRIX:
        raise ValueError("issue #6464 diagnostic must use its frozen scenario matrix")

    normalized = dict(config)
    normalized.update(
        {
            "scenario_matrix": str(scenario_matrix),
            "seeds": seeds,
            "horizon": horizon,
            "dt": dt,
            "corridor": corridor,
            "planners": planners,
            "max_pedestrians": max_pedestrians,
            "expected_effective_num_samples": expected_effective_num_samples,
        }
    )
    return normalized


def select_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load and validate the exact corridor scenario/seed matrix."""
    scenarios = [dict(scenario) for scenario in load_scenarios(config["scenario_matrix"])]
    wanted = set(config["scenario_ids"])
    selected = [scenario for scenario in scenarios if scenario.get("name") in wanted]
    if len(selected) != len(wanted):
        found = sorted(str(scenario.get("name")) for scenario in selected)
        raise ValueError(f"scenario matrix did not provide the requested cells: {found}")
    for scenario in selected:
        if scenario.get("name") != EXPECTED_SCENARIO:
            raise ValueError(f"unsupported scenario in diagnostic: {scenario.get('name')!r}")
        if "classic_head_on_corridor.svg" not in str(scenario.get("map_file", "")):
            raise ValueError("BRNE diagnostic accepts only the classic head-on corridor map")
        metadata = scenario.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("archetype") != "head_on_corridor":
            raise ValueError("scenario is missing the approved head_on_corridor archetype")
        scenario_seeds = [int(seed) for seed in scenario.get("seeds", [])]
        if scenario_seeds != config["seeds"]:
            raise ValueError(
                f"scenario seeds {scenario_seeds} do not match the frozen seeds {config['seeds']}"
            )
        scenario_horizon = scenario.get("run_horizon")
        if scenario_horizon is not None and int(scenario_horizon) != int(config["horizon"]):
            raise ValueError("scenario horizon does not match the frozen diagnostic horizon")
        scenario_dt = scenario.get("run_dt")
        if scenario_dt is not None and not math.isclose(
            _finite_float(scenario_dt, field="scenario.run_dt"),
            float(config["dt"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("scenario timestep does not match the frozen diagnostic timestep")
        if any(key in scenario for key in ("single_pedestrians", "map_semantics")):
            raise ValueError("unsupported static/marker geometry in BRNE corridor diagnostic")
    return selected


def _trace_summary(
    record: dict[str, Any],
) -> tuple[list[tuple[float, float]] | None, int | None]:
    """Extract finite robot positions and the maximum traced pedestrian count."""
    metadata = record.get("algorithm_metadata")
    trace = metadata.get("simulation_step_trace") if isinstance(metadata, dict) else None
    steps = trace.get("steps") if isinstance(trace, dict) else None
    if not isinstance(steps, list) or not steps:
        return None, None
    positions: list[tuple[float, float]] = []
    max_pedestrians = 0
    for step in steps:
        robot = step.get("robot") if isinstance(step, dict) else None
        position = robot.get("position") if isinstance(robot, dict) else None
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            return None, None
        x, y = float(position[0]), float(position[1])
        if not math.isfinite(x) or not math.isfinite(y):
            return None, None
        pedestrians = step.get("pedestrians") if isinstance(step, dict) else None
        if not isinstance(pedestrians, list):
            return None, None
        max_pedestrians = max(max_pedestrians, len(pedestrians))
        positions.append((x, y))
    return positions, max_pedestrians


def _finite_pair(value: Any, *, field: str) -> list[float]:
    """Validate one finite two-dimensional world-frame vector."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-element vector")
    parsed = [float(component) for component in value]
    if not all(math.isfinite(component) for component in parsed):
        raise ValueError(f"{field} must contain finite values")
    return parsed


def _finite_optional(value: Any, *, field: str) -> float | None:
    """Validate an optional finite scalar."""
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite when present")
    return parsed


def _finite_required(value: Any, *, field: str) -> float:
    """Validate a required finite scalar."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} is required")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite when present")
    return parsed


def _validate_nominal_command(value: Any, *, step_index: int) -> None:
    """Validate the adapter nominal command and its construction mode."""
    if not isinstance(value, dict):
        raise ValueError(f"step {step_index} is missing nominal command")
    _finite_required(value.get("v_m_s"), field="nominal v")
    _finite_required(value.get("omega_rad_s"), field="nominal omega")
    mode = value.get("construction_mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"step {step_index} has an invalid nominal command construction mode")


def _validate_pedestrian_selection(value: Any, *, step_index: int, selected_count: int) -> None:
    """Validate observed, activation-radius, and passed-agent counts."""
    if not isinstance(value, dict):
        raise ValueError(f"step {step_index} is missing pedestrian selection metadata")
    counts: dict[str, int] = {}
    for field in (
        "observed_count",
        "within_upstream_activation_radius_count",
        "passed_to_brne_count",
    ):
        count = value.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"step {step_index} has an invalid pedestrian {field}")
        counts[field] = count
    if counts["within_upstream_activation_radius_count"] > counts["observed_count"]:
        raise ValueError(f"step {step_index} has more activated than observed pedestrians")
    if counts["passed_to_brne_count"] > counts["observed_count"]:
        raise ValueError(f"step {step_index} passes more pedestrians than observed")
    if counts["passed_to_brne_count"] != selected_count:
        raise ValueError(f"step {step_index} has inconsistent passed pedestrian count")
    radius = _finite_required(
        value.get("upstream_activation_radius_m"),
        field="upstream activation radius",
    )
    if radius < 0.0:
        raise ValueError(f"step {step_index} has a negative upstream activation radius")
    if not isinstance(value.get("activation_gate_applied"), bool):
        raise ValueError(f"step {step_index} has an invalid activation gate flag")
    mode = value.get("selection_mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"step {step_index} has an invalid pedestrian selection mode")


def _validate_applied_environment_action(value: Any, *, step_index: int) -> None:
    """Validate the action payload that was passed to the environment step."""
    if not isinstance(value, dict):
        raise ValueError(f"step {step_index} is missing applied environment action")
    linear = value.get("linear_velocity", value.get("v"))
    angular = value.get("angular_velocity", value.get("omega"))
    _finite_required(linear, field="applied environment linear action")
    _finite_required(angular, field="applied environment angular action")


def _validate_candidate_distribution(  # noqa: C901
    value: Any, *, step_index: int
) -> None:
    """Validate bounded candidate/weight summaries without accepting raw tensors."""
    if not isinstance(value, dict):
        raise ValueError(f"step {step_index} has malformed candidate distribution")
    status = value.get("status")
    if status == "unavailable":
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise ValueError(
                f"step {step_index} has an unavailable candidate distribution without reason"
            )
        return
    if status != "available" or value.get("schema_version") != "brne-candidate-distribution.v1":
        raise ValueError(f"step {step_index} has an invalid candidate distribution status")
    sample_count = value.get("sample_count")
    plan_step_count = value.get("plan_step_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or isinstance(plan_step_count, bool)
        or not isinstance(plan_step_count, int)
        or plan_step_count < 1
    ):
        raise ValueError(f"step {step_index} has invalid candidate distribution dimensions")

    def validate_stats(summary: Any, *, field: str) -> None:
        if not isinstance(summary, dict):
            raise ValueError(f"step {step_index} has malformed {field} distribution")
        for statistic in ("min", "q25", "median", "mean", "q75", "max", "std"):
            _finite_required(summary.get(statistic), field=f"{field}.{statistic}")

    def validate_step_summary(summary: Any, *, field: str) -> None:
        if not isinstance(summary, dict):
            raise ValueError(f"step {step_index} has malformed {field} candidate summary")
        controls = summary.get("candidate_controls")
        if not isinstance(controls, dict):
            raise ValueError(f"step {step_index} is missing {field} candidate controls")
        validate_stats(controls.get("v_m_s"), field=f"{field}.v_m_s")
        validate_stats(controls.get("omega_rad_s"), field=f"{field}.omega_rad_s")
        validate_stats(summary.get("weights"), field=f"{field}.weights")
        weighted_mean = summary.get("weighted_mean")
        if not isinstance(weighted_mean, dict):
            raise ValueError(f"step {step_index} is missing {field} weighted mean")
        _finite_required(weighted_mean.get("v_m_s"), field=f"{field}.weighted_mean.v_m_s")
        _finite_required(
            weighted_mean.get("omega_rad_s"), field=f"{field}.weighted_mean.omega_rad_s"
        )

    validate_step_summary(value.get("first"), field="first")
    second = value.get("second")
    if plan_step_count > 1:
        validate_step_summary(second, field="second")
    elif second is not None:
        validate_step_summary(second, field="second")
    first_to_second = value.get("first_to_second")
    if plan_step_count > 1:
        if not isinstance(first_to_second, dict):
            raise ValueError(f"step {step_index} is missing first-to-second candidate deltas")
        for field in (
            "candidate_mean_delta_v_m_s",
            "weighted_mean_delta_v_m_s",
            "candidate_mean_delta_omega_rad_s",
            "weighted_mean_delta_omega_rad_s",
        ):
            _finite_required(first_to_second.get(field), field=f"first_to_second.{field}")


def _validate_brne_mechanism_trace(  # noqa: C901, PLR0912, PLR0915
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Validate the compact BRNE trace required for mechanism diagnosis."""
    metadata = record.get("algorithm_metadata")
    runtime = metadata.get("planner_runtime") if isinstance(metadata, dict) else None
    planner_meta = runtime.get("planner_metadata") if isinstance(runtime, dict) else None
    trace = planner_meta.get("mechanism_trace") if isinstance(planner_meta, dict) else None
    if not isinstance(trace, dict):
        return None, "missing"
    if trace.get("schema_version") != "brne-mechanism-trace.v1":
        return None, "invalid_schema_version"
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, "missing_steps"
    simulation_trace = metadata.get("simulation_step_trace") if isinstance(metadata, dict) else None
    simulation_steps = simulation_trace.get("steps") if isinstance(simulation_trace, dict) else None
    if not isinstance(simulation_steps, list) or len(steps) != len(simulation_steps):
        return None, "step_count_mismatch"
    try:
        for index, simulation_step in enumerate(simulation_steps):
            planner = simulation_step.get("planner") if isinstance(simulation_step, dict) else None
            _validate_applied_environment_action(
                planner.get("applied_environment_action") if isinstance(planner, dict) else None,
                step_index=index,
            )
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or int(step.get("step", -1)) != index:
                raise ValueError(f"step {index} has an invalid index")
            observation = step.get("observation")
            if not isinstance(observation, dict):
                raise ValueError(f"step {index} is missing observation")
            _finite_pair(observation.get("robot_position_world_m"), field="robot position")
            _finite_pair(observation.get("robot_velocity_world_m_s"), field="robot velocity")
            _finite_pair(observation.get("goal_position_world_m"), field="goal position")
            for field in (
                "declared_heading_rad",
                "velocity_derived_heading_rad",
                "goal_bearing_rad",
                "heading_goal_angular_difference_rad",
            ):
                _finite_optional(observation.get(field), field=field)
            selected_pedestrians = step.get("selected_pedestrians")
            if not isinstance(selected_pedestrians, list):
                raise ValueError(f"step {index} is missing selected pedestrians")
            for pedestrian in selected_pedestrians:
                if not isinstance(pedestrian, dict):
                    raise ValueError(f"step {index} has a malformed selected pedestrian")
                _finite_pair(pedestrian.get("position_world_m"), field="pedestrian position")
                _finite_pair(pedestrian.get("velocity_world_m_s"), field="pedestrian velocity")
                _finite_optional(pedestrian.get("distance_m"), field="pedestrian distance")
            _validate_nominal_command(step.get("nominal_command"), step_index=index)
            _validate_pedestrian_selection(
                step.get("pedestrian_selection"),
                step_index=index,
                selected_count=len(selected_pedestrians),
            )
            selected_action = step.get("selected_action")
            if not isinstance(selected_action, dict):
                raise ValueError(f"step {index} is missing selected action")
            _finite_optional(selected_action.get("v_m_s"), field="selected v")
            _finite_optional(selected_action.get("omega_rad_s"), field="selected omega")
            if selected_action.get("v_m_s") is None or selected_action.get("omega_rad_s") is None:
                raise ValueError(f"step {index} has incomplete selected action")
            pre_clamp_action = step.get("pre_clamp_action")
            if pre_clamp_action is not None:
                if not isinstance(pre_clamp_action, dict):
                    raise ValueError(f"step {index} has malformed pre-clamp action")
                _finite_optional(pre_clamp_action.get("v_m_s"), field="pre-clamp v")
                _finite_optional(pre_clamp_action.get("omega_rad_s"), field="pre-clamp omega")
                if (
                    pre_clamp_action.get("v_m_s") is None
                    or pre_clamp_action.get("omega_rad_s") is None
                ):
                    raise ValueError(f"step {index} has incomplete pre-clamp action")
            action_clipping = step.get("action_clipping")
            if action_clipping is not None:
                if not isinstance(action_clipping, dict) or any(
                    not isinstance(action_clipping.get(field), bool)
                    for field in ("v_clipped", "omega_clipped", "any_clipped")
                ):
                    raise ValueError(f"step {index} has malformed action clipping")
            action_delta = step.get("action_delta")
            if action_delta is not None:
                if not isinstance(action_delta, dict) or not isinstance(
                    action_delta.get("changed"), bool
                ):
                    raise ValueError(f"step {index} has malformed action delta")
                _finite_optional(action_delta.get("v"), field="action delta v")
                _finite_optional(action_delta.get("omega"), field="action delta omega")
            ensemble = step.get("ensemble")
            if not isinstance(ensemble, dict):
                raise ValueError(f"step {index} is missing ensemble metadata")
            effective = ensemble.get("effective_num_samples")
            if isinstance(effective, bool) or not isinstance(effective, int) or effective < 1:
                raise ValueError(f"step {index} has invalid effective sample count")
            for field in ("control_ensemble_shape", "weight_shape"):
                shape = ensemble.get(field)
                if shape is not None and (
                    not isinstance(shape, list)
                    or not shape
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
                ):
                    raise ValueError(f"step {index} has malformed {field}")
            if ensemble.get("aggregation_mode") not in {
                "not_applied",
                "plan_step_first",
                "samples_first",
            }:
                raise ValueError(f"step {index} has unknown aggregation mode")
            expected_formula = {
                "not_applied": "not_applied",
                "plan_step_first": "mean_plan_step_first_over_samples",
                "samples_first": "mean_samples_first_over_samples",
            }[ensemble["aggregation_mode"]]
            if ensemble.get("aggregation_formula") != expected_formula:
                raise ValueError(f"step {index} has an inconsistent aggregation formula")
            candidate_distribution = ensemble.get("candidate_distribution")
            _validate_candidate_distribution(candidate_distribution, step_index=index)
            runtime_step = step.get("runtime")
            if not isinstance(runtime_step, dict):
                raise ValueError(f"step {index} is missing runtime metadata")
            if runtime_step.get("status") not in {"ok", "failed"}:
                raise ValueError(f"step {index} has unknown runtime status")
            failure_count = runtime_step.get("failure_count")
            if (
                isinstance(failure_count, bool)
                or not isinstance(failure_count, int)
                or failure_count < 0
            ):
                raise ValueError(f"step {index} has invalid runtime failure count")
            if not isinstance(runtime_step.get("failure_reasons"), list) or any(
                not isinstance(reason, str) for reason in runtime_step["failure_reasons"]
            ):
                raise ValueError(f"step {index} has malformed runtime failure reasons")
            if (
                runtime_step.get("status") == "ok"
                and ensemble.get("aggregation_mode") != "not_applied"
            ):
                if pre_clamp_action is None:
                    raise ValueError(f"step {index} is missing pre-clamp action")
                if action_clipping is None:
                    raise ValueError(f"step {index} is missing action clipping metadata")
                if candidate_distribution.get("status") != "available":
                    raise ValueError(f"step {index} is missing candidate distribution")
            _finite_optional(runtime_step.get("elapsed_s"), field="solver elapsed time")
            _finite_optional(runtime_step.get("budget_s"), field="solver budget")
    except (TypeError, ValueError) as exc:
        return None, f"malformed:{exc}"
    return trace, "available"


def _phase_progress(goal_distances: list[float]) -> list[dict[str, Any]] | None:
    """Return signed goal-distance progress over three trace phases."""
    if len(goal_distances) < 3:
        return None
    boundaries = [
        0,
        len(goal_distances) // 3,
        (2 * len(goal_distances)) // 3,
        len(goal_distances) - 1,
    ]
    phases: list[dict[str, Any]] = []
    for name, start, end in zip(
        ("early", "middle", "late"), boundaries[:-1], boundaries[1:], strict=True
    ):
        phases.append(
            {
                "phase": name,
                "start_step": start,
                "end_step": end,
                "signed_progress_m": float(goal_distances[start] - goal_distances[end]),
            }
        )
    return phases


def _simulation_action_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actions passed to ``env.step`` from the common simulation trace."""
    actions: list[dict[str, float]] = []
    applied_count = 0
    for step in steps:
        planner = step.get("planner") if isinstance(step, dict) else None
        payload = planner.get("applied_environment_action") if isinstance(planner, dict) else None
        if isinstance(payload, dict):
            applied_count += 1
        elif isinstance(planner, dict):
            # Preserve context-only summaries for historical comparator traces. Native BRNE
            # rows are fail-closed above when the actual environment action is absent.
            payload = planner.get("selected_action")
        if not isinstance(payload, dict):
            continue
        linear = payload.get("linear_velocity", payload.get("v"))
        angular = payload.get("angular_velocity", payload.get("omega"))
        try:
            linear_value = float(linear)
            angular_value = float(angular)
        except (TypeError, ValueError):
            continue
        if math.isfinite(linear_value) and math.isfinite(angular_value):
            actions.append({"v_m_s": linear_value, "omega_rad_s": angular_value})
    changes = sum(
        not math.isclose(previous["v_m_s"], current["v_m_s"])
        or not math.isclose(previous["omega_rad_s"], current["omega_rad_s"])
        for previous, current in pairwise(actions)
    )
    has_legacy_fallback = bool(actions) and applied_count != len(actions)
    return {
        "source": (
            "algorithm_metadata.simulation_step_trace.steps[].planner.applied_environment_action"
            if applied_count and not has_legacy_fallback
            else "algorithm_metadata.simulation_step_trace.steps[].planner.selected_action"
        ),
        "command_space": (
            "environment_step_action"
            if applied_count and not has_legacy_fallback
            else "planner_command"
        ),
        "semantics": (
            "action payload passed to env.step after planner-command conversion"
            if applied_count and not has_legacy_fallback
            else "legacy planner selected-action payload; environment conversion not recorded"
        ),
        "available": bool(actions),
        "first": actions[0] if actions else None,
        "last": actions[-1] if actions else None,
        "action_change_count": changes,
    }


def _mechanism_action_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize finite pre-clamp actions and observed safety clipping."""
    actions: list[dict[str, float]] = []
    clipped_steps = 0
    for step in steps:
        payload = step.get("pre_clamp_action") if isinstance(step, dict) else None
        if not isinstance(payload, dict):
            continue
        try:
            linear_value = float(payload.get("v_m_s"))
            angular_value = float(payload.get("omega_rad_s"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(linear_value) and math.isfinite(angular_value):
            actions.append({"v_m_s": linear_value, "omega_rad_s": angular_value})
        clipping = step.get("action_clipping") if isinstance(step, dict) else None
        if isinstance(clipping, dict) and clipping.get("any_clipped") is True:
            clipped_steps += 1
    changes = sum(
        not math.isclose(previous["v_m_s"], current["v_m_s"])
        or not math.isclose(previous["omega_rad_s"], current["omega_rad_s"])
        for previous, current in pairwise(actions)
    )
    return {
        "source": "algorithm_metadata.planner_runtime.planner_metadata.mechanism_trace.steps[].pre_clamp_action",
        "available": bool(actions),
        "first": actions[0] if actions else None,
        "last": actions[-1] if actions else None,
        "v_range_m_s": (
            [min(action["v_m_s"] for action in actions), max(action["v_m_s"] for action in actions)]
            if actions
            else None
        ),
        "omega_range_rad_s": (
            [
                min(action["omega_rad_s"] for action in actions),
                max(action["omega_rad_s"] for action in actions),
            ]
            if actions
            else None
        ),
        "action_change_count": changes,
        "clipped_steps": clipped_steps,
    }


def _selected_post_clamp_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize selected BRNE commands after the planner safety clamp."""
    actions: list[dict[str, float]] = []
    for step in steps:
        payload = step.get("selected_action") if isinstance(step, dict) else None
        if not isinstance(payload, dict):
            continue
        try:
            linear_value = float(payload.get("v_m_s"))
            angular_value = float(payload.get("omega_rad_s"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(linear_value) and math.isfinite(angular_value):
            actions.append({"v_m_s": linear_value, "omega_rad_s": angular_value})
    changes = sum(
        not math.isclose(previous["v_m_s"], current["v_m_s"])
        or not math.isclose(previous["omega_rad_s"], current["omega_rad_s"])
        for previous, current in pairwise(actions)
    )
    return {
        "source": "algorithm_metadata.planner_runtime.planner_metadata.mechanism_trace.steps[].selected_action",
        "available": bool(actions),
        "first": actions[0] if actions else None,
        "last": actions[-1] if actions else None,
        "action_change_count": changes,
    }


def _candidate_distribution_summary(ensemble_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain only first-to-second candidate/weight summaries for the compact report."""
    distributions = [item.get("candidate_distribution") for item in ensemble_steps]
    if not distributions:
        return {
            "status": "unavailable",
            "reason": "candidate_distribution_not_recorded",
        }
    first_distribution = distributions[0]
    if not isinstance(first_distribution, dict) or first_distribution.get("status") != "available":
        reason = (
            first_distribution.get("reason", "candidate_distribution_unavailable")
            if isinstance(first_distribution, dict)
            else "candidate_distribution_malformed"
        )
        return {"status": "unavailable", "reason": str(reason)}
    second_distribution = distributions[1] if len(distributions) > 1 else None
    first_observation = first_distribution.get("first")
    second_observation = (
        second_distribution.get("first") if isinstance(second_distribution, dict) else None
    )
    observation_step_transition = {"status": "unavailable", "reason": "second_trace_step_missing"}
    if isinstance(first_observation, dict) and isinstance(second_observation, dict):
        first_controls = first_observation.get("candidate_controls", {})
        second_controls = second_observation.get("candidate_controls", {})
        first_v = first_controls.get("v_m_s", {})
        second_v = second_controls.get("v_m_s", {})
        first_omega = first_controls.get("omega_rad_s", {})
        second_omega = second_controls.get("omega_rad_s", {})
        first_weights = first_observation.get("weights", {})
        second_weights = second_observation.get("weights", {})
        first_mean = first_observation.get("weighted_mean", {})
        second_mean = second_observation.get("weighted_mean", {})
        observation_step_transition = {
            "status": "available",
            "from_trace_step": 0,
            "to_trace_step": 1,
            "from": first_observation,
            "to": second_observation,
            "delta": {
                "candidate_mean_delta_v_m_s": float(second_v["mean"] - first_v["mean"]),
                "weighted_mean_delta_v_m_s": float(second_mean["v_m_s"] - first_mean["v_m_s"]),
                "candidate_mean_delta_omega_rad_s": float(
                    second_omega["mean"] - first_omega["mean"]
                ),
                "weighted_mean_delta_omega_rad_s": float(
                    second_mean["omega_rad_s"] - first_mean["omega_rad_s"]
                ),
                "weight_mean_delta": float(second_weights["mean"] - first_weights["mean"]),
                "weight_std_delta": float(second_weights["std"] - first_weights["std"]),
            },
        }
    return {
        "status": "available",
        "schema_version": "brne-candidate-distribution.v1",
        "sample_count": first_distribution.get("sample_count"),
        "plan_step_count": first_distribution.get("plan_step_count"),
        "first": first_distribution.get("first"),
        "second": first_distribution.get("second"),
        "first_to_second": (
            first_distribution.get("first_to_second")
            if isinstance(first_distribution, dict)
            else None
        ),
        "observation_step_transition": observation_step_transition,
        "source": (
            "algorithm_metadata.planner_runtime.planner_metadata.mechanism_trace.steps[]"
            ".ensemble.candidate_distribution"
        ),
    }


def _nominal_command_summary(mechanism_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize nominal adapter commands without retaining the full command sequence."""
    commands = [step.get("nominal_command") for step in mechanism_steps]
    commands = [
        item for item in commands if isinstance(item, dict) and item.get("status") != "unavailable"
    ]
    if not commands:
        return {
            "available": False,
            "source": "algorithm_metadata.planner_runtime.planner_metadata.mechanism_trace.steps[].nominal_command",
        }
    return {
        "available": True,
        "source": "algorithm_metadata.planner_runtime.planner_metadata.mechanism_trace.steps[].nominal_command",
        "first": commands[0],
        "last": commands[-1],
        "construction_modes": sorted({str(item.get("construction_mode")) for item in commands}),
        "v_range_m_s": [
            min(float(item["v_m_s"]) for item in commands),
            max(float(item["v_m_s"]) for item in commands),
        ],
        "omega_range_rad_s": [
            min(float(item["omega_rad_s"]) for item in commands),
            max(float(item["omega_rad_s"]) for item in commands),
        ],
    }


def _pedestrian_selection_summary(mechanism_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-step observed and passed pedestrian counts."""
    selections = [step.get("pedestrian_selection") for step in mechanism_steps]
    selections = [item for item in selections if isinstance(item, dict)]
    if not selections:
        return {
            "available": False,
            "source": "algorithm_metadata.planner_runtime.planner_metadata.mechanism_trace.steps[].pedestrian_selection",
        }
    return {
        "available": True,
        "source": "algorithm_metadata.planner_runtime.planner_metadata.mechanism_trace.steps[].pedestrian_selection",
        "first": selections[0],
        "last": selections[-1],
        "observed_count_range": [
            min(int(item["observed_count"]) for item in selections),
            max(int(item["observed_count"]) for item in selections),
        ],
        "within_upstream_activation_radius_count_range": [
            min(int(item["within_upstream_activation_radius_count"]) for item in selections),
            max(int(item["within_upstream_activation_radius_count"]) for item in selections),
        ],
        "passed_to_brne_count_range": [
            min(int(item["passed_to_brne_count"]) for item in selections),
            max(int(item["passed_to_brne_count"]) for item in selections),
        ],
    }


def _event_step(record: dict[str, Any], *, event_type: str, dt: float) -> int | None:
    """Extract the first collision or terminal goal step when the record supports it."""
    ledger = record.get("event_ledger")
    if event_type == "collision" and isinstance(ledger, dict):
        events = ledger.get("collision_events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                if isinstance(event.get("step"), int):
                    return int(event["step"])
                try:
                    collision_time = float(event.get("collision_time"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(collision_time) and dt > 0.0:
                    return max(0, round(collision_time / dt) - 1)
    outcome = record.get("outcome")
    if event_type == "goal" and isinstance(outcome, dict) and outcome.get("route_complete"):
        metadata = record.get("algorithm_metadata")
        trace = metadata.get("simulation_step_trace") if isinstance(metadata, dict) else None
        steps = trace.get("steps") if isinstance(trace, dict) else None
        if isinstance(steps, list) and steps:
            final_step = steps[-1].get("step") if isinstance(steps[-1], dict) else None
            return int(final_step) if isinstance(final_step, int) else None
    return None


def _mechanism_table_row(
    record: dict[str, Any], classified: dict[str, Any], *, planner_key: str
) -> dict[str, Any]:
    """Build one compact trace-backed mechanism row for the issue report."""
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    simulation_trace = metadata.get("simulation_step_trace")
    simulation_steps = simulation_trace.get("steps") if isinstance(simulation_trace, dict) else None
    if not isinstance(simulation_steps, list) or not simulation_steps:
        return {
            "schema_version": "brne-mechanism-table-row.v1",
            "planner": planner_key,
            "scenario_id": record.get("scenario_id"),
            "seed": record.get("seed"),
            "status": "unavailable",
            "unavailable_reason": "simulation_trace_missing",
        }
    mechanism_trace, mechanism_status = _validate_brne_mechanism_trace(record)
    if planner_key == "brne" and mechanism_trace is None:
        return {
            "schema_version": "brne-mechanism-table-row.v1",
            "planner": planner_key,
            "scenario_id": record.get("scenario_id"),
            "seed": record.get("seed"),
            "status": "unavailable",
            "unavailable_reason": f"mechanism_trace_{mechanism_status}",
        }
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    interaction = record.get("interaction_exposure")
    interaction = interaction if isinstance(interaction, dict) else None
    mechanism_steps = mechanism_trace.get("steps", []) if isinstance(mechanism_trace, dict) else []
    observations = [step.get("observation", {}) for step in mechanism_steps]
    observations = [item for item in observations if isinstance(item, dict)]
    goal_distances = []
    for observation in observations:
        try:
            value = float(
                math.dist(
                    _finite_pair(observation.get("robot_position_world_m"), field="robot"),
                    _finite_pair(observation.get("goal_position_world_m"), field="goal"),
                )
            )
        except (TypeError, ValueError):
            goal_distances = []
            break
        if not math.isfinite(value):
            goal_distances = []
            break
        goal_distances.append(value)
    goal_bearings = [observation.get("goal_bearing_rad") for observation in observations]
    angular_differences = [
        observation.get("heading_goal_angular_difference_rad") for observation in observations
    ]
    declared_headings = [observation.get("declared_heading_rad") for observation in observations]
    velocity_headings = [
        observation.get("velocity_derived_heading_rad") for observation in observations
    ]
    valid_goal_bearings = [value for value in goal_bearings if value is not None]
    valid_angular_differences = [value for value in angular_differences if value is not None]
    runtime_steps = [step.get("runtime", {}) for step in mechanism_steps]
    runtime_steps = [item for item in runtime_steps if isinstance(item, dict)]
    ensemble_steps = [step.get("ensemble", {}) for step in mechanism_steps]
    ensemble_steps = [item for item in ensemble_steps if isinstance(item, dict)]
    pre_clamp_action = _mechanism_action_summary(mechanism_steps)
    selected_post_clamp_command = _selected_post_clamp_summary(mechanism_steps)
    selected_pedestrians = [step.get("selected_pedestrians", []) for step in mechanism_steps]
    selected_pedestrians = [item for item in selected_pedestrians if isinstance(item, list)]
    nominal_command = _nominal_command_summary(mechanism_steps)
    pedestrian_selection = _pedestrian_selection_summary(mechanism_steps)
    runtime_meta = metadata.get("planner_runtime")
    runtime_planner_meta = (
        runtime_meta.get("planner_metadata") if isinstance(runtime_meta, dict) else None
    )
    runtime_planner_meta = runtime_planner_meta if isinstance(runtime_planner_meta, dict) else {}
    termination = record.get("outcome")
    termination = termination if isinstance(termination, dict) else {}
    row_status = classified.get("status", "unavailable")
    applied_environment_command = _simulation_action_summary(simulation_steps)
    return {
        "schema_version": "brne-mechanism-table-row.v1",
        "planner": planner_key,
        "scenario_id": record.get("scenario_id"),
        "seed": record.get("seed"),
        "status": row_status,
        "native_core_via_adapter": bool(classified.get("native_core_via_adapter", False)),
        "trace_status": mechanism_status if planner_key == "brne" else "simulation_trace_only",
        "heading": {
            "declared_first_rad": declared_headings[0] if declared_headings else None,
            "declared_last_rad": declared_headings[-1] if declared_headings else None,
            "velocity_derived_first_rad": velocity_headings[0] if velocity_headings else None,
            "velocity_derived_last_rad": velocity_headings[-1] if velocity_headings else None,
            "goal_bearing_first_rad": valid_goal_bearings[0] if valid_goal_bearings else None,
            "goal_bearing_last_rad": valid_goal_bearings[-1] if valid_goal_bearings else None,
            "angular_difference_first_rad": (
                valid_angular_differences[0] if valid_angular_differences else None
            ),
            "angular_difference_last_rad": (
                valid_angular_differences[-1] if valid_angular_differences else None
            ),
            "unavailable_reason": None if observations else "goal_geometry_not_recorded",
        },
        "pre_clamp_action": pre_clamp_action,
        "selected_action": applied_environment_command,
        "selected_post_clamp_command": selected_post_clamp_command,
        "applied_environment_command": applied_environment_command,
        "nominal_command": nominal_command,
        "action_clipping": {
            "available": pre_clamp_action["available"],
            "clipped_steps": pre_clamp_action["clipped_steps"],
        },
        "runtime": {
            "status": runtime_planner_meta.get("runtime_status", "not_applicable"),
            "failure_count": runtime_planner_meta.get("failure_count", 0),
            "failure_reasons": runtime_planner_meta.get("failure_reasons", []),
            "effective_num_samples": runtime_planner_meta.get("effective_num_samples"),
            "source_commit": runtime_planner_meta.get("source_commit"),
            "per_step_statuses": sorted({item.get("status") for item in runtime_steps}),
            "per_step_budget_exceeded_count": sum(
                bool(item.get("budget_exceeded")) for item in runtime_steps
            ),
        },
        "goal": {
            "initial_distance_m": goal_distances[0]
            if goal_distances
            else simulation_trace.get("initial_goal_distance_m"),
            "final_distance_m": goal_distances[-1] if goal_distances else None,
            "signed_progress_by_phase": _phase_progress(goal_distances),
            "unavailable_reason": None
            if goal_distances
            else "goal_geometry_not_recorded_in_common_simulation_trace",
        },
        "motion": {
            "displacement_m": classified.get("displacement_m"),
            "zero_motion_fraction": classified.get("zero_motion_fraction"),
            "nondegenerate": classified.get("nondegenerate"),
        },
        "interaction_zone": {
            "exposure_share": interaction.get("interaction_exposure_share")
            if interaction
            else None,
            "exposure_steps": interaction.get("interaction_exposure_steps")
            if interaction
            else None,
            "radius_m": interaction.get("interaction_exposure_radius_m") if interaction else None,
            "status": interaction.get("interaction_exposure_status")
            if interaction
            else "unavailable",
        },
        "clearance": {
            "min_clearance_m": metrics.get("min_clearance"),
            "mean_clearance_m": metrics.get("mean_clearance"),
            "interpretation": "radius-aware surface-clearance proxy",
        },
        "events": {
            "termination_reason": record.get("termination_reason"),
            "collision_step": _event_step(record, event_type="collision", dt=0.1),
            "goal_step": _event_step(record, event_type="goal", dt=0.1),
            "collision_event": termination.get("collision_event"),
            "goal_reached": bool(classified.get("goal_reached")),
        },
        "pedestrian_world_frame": {
            "source": (
                "planner adapter selected agents"
                if planner_key == "brne"
                else "algorithm_metadata.simulation_step_trace"
            ),
            "selected_first": selected_pedestrians[0] if selected_pedestrians else None,
            "selected_last": selected_pedestrians[-1] if selected_pedestrians else None,
            "simulation_first": simulation_steps[0].get("pedestrians"),
            "simulation_last": simulation_steps[-1].get("pedestrians"),
            "max_observed_count": classified.get("max_pedestrians"),
        },
        "pedestrian_selection": pedestrian_selection,
        "aggregation": {
            "requested_num_samples": sorted(
                {item.get("requested_num_samples") for item in ensemble_steps}
            ),
            "effective_num_samples": sorted(
                {item.get("effective_num_samples") for item in ensemble_steps}
            ),
            "control_ensemble_shapes": sorted(
                {
                    json.dumps(item.get("control_ensemble_shape"), sort_keys=True)
                    for item in ensemble_steps
                }
            ),
            "weight_shapes": sorted(
                {json.dumps(item.get("weight_shape"), sort_keys=True) for item in ensemble_steps}
            ),
            "modes": sorted({item.get("aggregation_mode") for item in ensemble_steps}),
            "candidate_distribution": _candidate_distribution_summary(ensemble_steps),
            "unavailable_reason": None
            if planner_key == "brne"
            else "upstream_aggregation_not_exposed_by_comparator",
        },
    }


def _runtime_source_fields(runtime_planner_meta: Any) -> tuple[Any, Any, Any]:
    """Extract source-integrity fields from runtime planner metadata."""
    if not isinstance(runtime_planner_meta, dict):
        return None, None, None
    return (
        runtime_planner_meta.get("source_commit"),
        runtime_planner_meta.get("source_pin"),
        runtime_planner_meta.get("source_integrity"),
    )


def classify_record(
    record: dict[str, Any], config: dict[str, Any], *, planner_key: str
) -> dict[str, Any]:
    """Classify one episode without promoting it to benchmark evidence."""
    metadata = record.get("algorithm_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    status = str(metadata.get("status", "unknown")).strip().lower()
    planner_meta = metadata.get("planner_metadata")
    planner_status = (
        str(planner_meta.get("status", "unknown")).strip().lower()
        if isinstance(planner_meta, dict)
        else "unknown"
    )
    diagnostic_meta = metadata.get("brne_diagnostic")
    planner_kinematics = metadata.get("planner_kinematics")
    planner_runtime = metadata.get("planner_runtime")
    runtime_planner_meta = (
        planner_runtime.get("planner_metadata") if isinstance(planner_runtime, dict) else None
    )
    runtime_status = (
        str(runtime_planner_meta.get("runtime_status", "unknown")).strip().lower()
        if isinstance(runtime_planner_meta, dict)
        else "not_applicable"
    )
    runtime_dependency_status = (
        str(runtime_planner_meta.get("status", "unknown")).strip().lower()
        if isinstance(runtime_planner_meta, dict)
        else "not_applicable"
    )
    runtime_source_commit, runtime_source_pin, runtime_source_integrity = _runtime_source_fields(
        runtime_planner_meta
    )
    try:
        runtime_failure_count = (
            int(runtime_planner_meta.get("failure_count", 0))
            if isinstance(runtime_planner_meta, dict)
            else 0
        )
    except (TypeError, ValueError):
        runtime_failure_count = -1
    effective_num_samples = (
        runtime_planner_meta.get("effective_num_samples")
        if isinstance(runtime_planner_meta, dict)
        else None
    )
    record_status = str(record.get("status", "")).strip().lower()
    record_failed = record_status in {"failed", "error"}
    fallback = bool(
        metadata.get("fallback_reason")
        or metadata.get("fallback_triggered")
        or status in {"fallback", "degraded", "unknown"}
        or planner_status in {"fallback", "degraded"}
    )
    runtime_invalid = planner_key == "brne" and (
        runtime_status != "ok" or runtime_failure_count != 0
    )
    diagnostic_metadata_valid = planner_key != "brne" or (
        isinstance(diagnostic_meta, dict)
        and diagnostic_meta.get("status") == "native_core_via_adapter"
        and diagnostic_meta.get("execution_semantics")
        == "native_upstream_core_through_robot_sf_adapter"
    )
    runtime_provenance_valid = planner_key != "brne" or (
        runtime_status == "ok"
        and runtime_dependency_status == "ok"
        and runtime_failure_count == 0
        and runtime_source_commit == BRNE_PINNED_SHA
        and runtime_source_pin == BRNE_PINNED_SHA
        and runtime_source_integrity == "clean_pinned_worktree"
        and isinstance(effective_num_samples, int)
        and not isinstance(effective_num_samples, bool)
        and effective_num_samples == int(config["expected_effective_num_samples"])
    )
    canonical_metadata_valid = planner_key != "brne" or (
        isinstance(planner_kinematics, dict)
        and planner_kinematics.get("execution_mode") == "adapter"
        and planner_kinematics.get("adapter_active") is True
        and planner_kinematics.get("adapter_name") == "BRNEPlanner"
        and planner_kinematics.get("supports_native_commands") is True
        and planner_kinematics.get("supports_adapter_commands") is True
        and planner_kinematics.get("planner_command_space") == "unicycle_vw"
    )
    _, mechanism_trace_status = _validate_brne_mechanism_trace(record)
    mechanism_trace_valid = planner_key != "brne" or mechanism_trace_status == "available"
    positions, max_pedestrians = _trace_summary(record)
    corridor = config["corridor"]
    trace_status = "available" if positions is not None else "unavailable"
    violation_count = 0
    displacement = 0.0
    zero_motion_fraction: float | None = None
    if positions:
        displacement = math.dist(positions[0], positions[-1])
        deltas = [math.dist(a, b) for a, b in pairwise(positions)]
        zero_motion_fraction = (
            sum(delta <= ZERO_MOTION_EPSILON_M for delta in deltas) / len(deltas) if deltas else 1.0
        )
        lower = float(corridor["y_min"])
        upper = float(corridor["y_max"])
        violation_count = sum(y < lower or y > upper for _, y in positions)

    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    success_value = metrics.get("success", metrics.get("success_rate", 0.0))
    try:
        goal_reached = float(success_value) > 0.0
    except (TypeError, ValueError):
        goal_reached = False
    execution_ok = (
        status == "ok"
        and not record_failed
        and not fallback
        and not runtime_invalid
        and diagnostic_metadata_valid
        and runtime_provenance_valid
        and canonical_metadata_valid
        and mechanism_trace_valid
    )
    native = (
        planner_key == "brne"
        and execution_ok
        and planner_status == "ok"
        and diagnostic_metadata_valid
        and runtime_provenance_valid
        and canonical_metadata_valid
        and mechanism_trace_valid
    )
    crowd_within_budget = max_pedestrians is not None and max_pedestrians <= int(
        config["max_pedestrians"]
    )
    nondegenerate = (
        positions is not None
        and displacement >= float(corridor["min_displacement_m"])
        and zero_motion_fraction is not None
        and zero_motion_fraction <= float(corridor["max_zero_motion_fraction"])
    )
    corridor_valid = positions is not None and violation_count == 0
    eligible = (
        execution_ok
        and (native if planner_key == "brne" else True)
        and trace_status == "available"
        and corridor_valid
        and nondegenerate
        and crowd_within_budget
    )
    if planner_key == "brne" and eligible:
        evidence_status = "available_native"
    elif planner_key != "brne" and eligible:
        evidence_status = "available_comparator"
    else:
        evidence_status = "unavailable"
    return {
        "episode_id": record.get("episode_id"),
        "scenario_id": record.get("scenario_id"),
        "seed": record.get("seed"),
        "status": evidence_status,
        "native": native,
        "execution_ok": execution_ok,
        "fallback_or_degraded": fallback,
        "record_status": record_status,
        "planner_status": status,
        "planner_dependency_status": planner_status,
        "planner_runtime_status": runtime_status,
        "planner_runtime_dependency_status": runtime_dependency_status,
        "planner_runtime_source_commit": runtime_source_commit,
        "planner_runtime_source_pin": runtime_source_pin,
        "planner_runtime_source_integrity": runtime_source_integrity,
        "planner_runtime_failure_count": runtime_failure_count,
        "effective_num_samples": effective_num_samples,
        "goal_reached": goal_reached,
        "trace_status": trace_status,
        "max_pedestrians": max_pedestrians,
        "crowd_within_budget": crowd_within_budget,
        "displacement_m": displacement,
        "zero_motion_fraction": zero_motion_fraction,
        "nondegenerate": nondegenerate,
        "corridor_violation_count": violation_count,
        "corridor_valid": corridor_valid,
        "diagnostic_metadata_present": isinstance(diagnostic_meta, dict),
        "diagnostic_metadata_valid": diagnostic_metadata_valid,
        "runtime_provenance_valid": runtime_provenance_valid,
        "canonical_metadata_valid": canonical_metadata_valid,
        "mechanism_trace_status": mechanism_trace_status,
        "mechanism_trace_valid": mechanism_trace_valid,
        "native_core_via_adapter": native,
        "claim_boundary": config["claim_boundary"],
    }


def summarize_records(
    *,
    planner_key: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    execution_summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build an arm summary with explicit unavailable-row accounting."""
    classified = [classify_record(record, config, planner_key=planner_key) for record in records]
    mechanism_rows = [
        _mechanism_table_row(record, row, planner_key=planner_key)
        for record, row in zip(records, classified, strict=True)
    ]
    expected = {
        (scenario_id, seed) for scenario_id in config["scenario_ids"] for seed in config["seeds"]
    }
    observed_sequence: list[tuple[str, int]] = []
    invalid_pair_rows = 0
    for row in classified:
        scenario_id = row.get("scenario_id")
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            invalid_pair_rows += 1
            continue
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            invalid_pair_rows += 1
            continue
        observed_sequence.append((scenario_id, seed))
    observed = set(observed_sequence)
    duplicate_pairs = sorted(pair for pair in observed if observed_sequence.count(pair) > 1)
    unexpected_pairs = sorted(observed - expected)
    missing_pairs = sorted(expected - observed)
    pair_coverage_exact = (
        not invalid_pair_rows
        and not duplicate_pairs
        and not unexpected_pairs
        and not missing_pairs
        and len(observed_sequence) == len(expected)
    )
    arm_status = "unavailable" if error or not classified else "partial"
    if arm_status == "partial" and pair_coverage_exact:
        arm_status = "available"
    eligible_statuses = {"available_native", "available_comparator"}
    return {
        "planner": planner_key,
        "status": arm_status,
        "error": error,
        "expected_rows": len(expected),
        "observed_rows": len(classified),
        "unique_observed_rows": len(observed),
        "pair_coverage_exact": pair_coverage_exact,
        "missing_pairs": [list(pair) for pair in missing_pairs],
        "duplicate_pairs": [list(pair) for pair in duplicate_pairs],
        "unexpected_pairs": [list(pair) for pair in unexpected_pairs],
        "invalid_pair_rows": invalid_pair_rows,
        "native_rows": sum(bool(row["native"]) for row in classified),
        "execution_ok_rows": sum(bool(row["execution_ok"]) for row in classified),
        "unavailable_rows": sum(row["status"] == "unavailable" for row in classified),
        "goal_reached_rows": sum(
            bool(row["goal_reached"]) and row["status"] in eligible_statuses for row in classified
        ),
        "goal_reached_unavailable_rows": sum(
            bool(row["goal_reached"]) and row["status"] not in eligible_statuses
            for row in classified
        ),
        "nondegenerate_rows": sum(bool(row["nondegenerate"]) for row in classified),
        "corridor_violation_rows": sum(not row["corridor_valid"] for row in classified),
        "crowd_over_budget_rows": sum(not row["crowd_within_budget"] for row in classified),
        "diagnostic_eligible_rows": sum(row["status"] in eligible_statuses for row in classified),
        "execution_summary": execution_summary,
        "rows": classified,
        "mechanism_table": mechanism_rows,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records, ignoring no malformed rows."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL at {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        records.append(payload)
    return records


def _markdown_cell(value: Any) -> str:
    """Render a report cell with an explicit fail-closed missing value."""
    return "not available" if value is None else str(value)


def _markdown_action_summary(summary: Any, *, include_clipped: bool = False) -> str:
    """Render a compact first-to-last action summary for the mechanism table."""
    if not isinstance(summary, dict) or summary.get("available") is not True:
        return "not available"
    first = summary.get("first")
    last = summary.get("last")
    if not isinstance(first, dict) or not isinstance(last, dict):
        return "not available"
    try:
        text = (
            f"{float(first['v_m_s']):.3f}/{float(first['omega_rad_s']):.3f}"
            f" -> {float(last['v_m_s']):.3f}/{float(last['omega_rad_s']):.3f}"
        )
    except (KeyError, TypeError, ValueError):
        return "not available"
    if include_clipped:
        text += f"; clipped={summary.get('clipped_steps', 'not available')}"
    return text


def _markdown_candidate_transition(summary: Any) -> str:
    """Render the first-to-second planner-observation candidate transition."""
    if not isinstance(summary, dict) or summary.get("status") != "available":
        return "not available"
    transition = summary.get("observation_step_transition")
    if not isinstance(transition, dict) or transition.get("status") != "available":
        return "not available"
    source = transition.get("from")
    target = transition.get("to")
    if not isinstance(source, dict) or not isinstance(target, dict):
        return "not available"
    source_controls = source.get("candidate_controls")
    target_controls = target.get("candidate_controls")
    if not isinstance(source_controls, dict) or not isinstance(target_controls, dict):
        return "not available"
    source_v = source_controls.get("v_m_s")
    target_v = target_controls.get("v_m_s")
    source_weights = source.get("weights")
    target_weights = target.get("weights")
    if not isinstance(source_v, dict) or not isinstance(target_v, dict):
        return "not available"
    if not isinstance(source_weights, dict) or not isinstance(target_weights, dict):
        return "not available"
    try:
        return (
            f"v mean {float(source_v['mean']):.3f}->{float(target_v['mean']):.3f}; "
            f"weighted {float(source['weighted_mean']['v_m_s']):.3f}"
            f"->{float(target['weighted_mean']['v_m_s']):.3f}; "
            f"w mean {float(source_weights['mean']):.3f}"
            f"->{float(target_weights['mean']):.3f}"
        )
    except (KeyError, TypeError, ValueError):
        return "not available"


def _markdown_nominal_command(summary: Any) -> str:
    """Render the nominal adapter command and construction mode."""
    if not isinstance(summary, dict) or summary.get("available") is not True:
        return "not available"
    first = summary.get("first")
    last = summary.get("last")
    modes = summary.get("construction_modes")
    if not isinstance(first, dict) or not isinstance(last, dict) or not isinstance(modes, list):
        return "not available"
    try:
        return (
            f"{float(first['v_m_s']):.3f}/{float(first['omega_rad_s']):.3f}"
            f" -> {float(last['v_m_s']):.3f}/{float(last['omega_rad_s']):.3f}; "
            f"mode={','.join(str(mode) for mode in modes)}"
        )
    except (KeyError, TypeError, ValueError):
        return "not available"


def _markdown_pedestrian_selection(summary: Any) -> str:
    """Render observed, activation-radius, and passed-agent counts."""
    if not isinstance(summary, dict) or summary.get("available") is not True:
        return "not available"
    first = summary.get("first")
    last = summary.get("last")
    if not isinstance(first, dict) or not isinstance(last, dict):
        return "not available"
    try:
        first_counts = (
            int(first["observed_count"]),
            int(first["within_upstream_activation_radius_count"]),
            int(first["passed_to_brne_count"]),
        )
        last_counts = (
            int(last["observed_count"]),
            int(last["within_upstream_activation_radius_count"]),
            int(last["passed_to_brne_count"]),
        )
    except (KeyError, TypeError, ValueError):
        return "not available"
    return (
        f"{first_counts[0]}/{first_counts[1]}/{first_counts[2]}"
        f" -> {last_counts[0]}/{last_counts[1]}/{last_counts[2]}"
    )


def _write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable and human-readable diagnostic reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "diagnostic_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# BRNE corridor diagnostic preflight (#6464)",
        "",
        f"- Status: **{report['status']}**",
        f"- Scenario matrix: `{report['config']['scenario_matrix']}`",
        f"- Scenario/seed cells: `{report['expected_pairs']}`",
        "- Evidence tier: smoke/diagnostic only",
        "- Fallback/degraded rows: unavailable and excluded",
        "- Goal-reaching counts: eligible rows only; unavailable rows are reported separately",
        "",
        "This report does not rank planners and is not benchmark, safety, realism, matched-compute, or paper evidence.",
        "",
        "## Arm accounting",
        "",
        "| planner | status | observed | exact pairs | native | eligible | goal reached | non-degenerate | corridor violations |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in report["arms"]:
        lines.append(
            f"| {arm['planner']} | {arm['status']} | {arm['observed_rows']} | "
            f"{'yes' if arm['pair_coverage_exact'] else 'no'} | {arm['native_rows']} | "
            f"{arm['diagnostic_eligible_rows']} | "
            f"{arm['goal_reached_rows']} | {arm['nondegenerate_rows']} | "
            f"{arm['corridor_violation_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism trace table",
            "",
            "Rows retain diagnostic telemetry even when native eligibility is unavailable. "
            "`not available` is a fail-closed state, not a zero or a success.",
            "",
            "| planner | seed | status | runtime | failures | nominal command (v/omega/mode) | pedestrian selection (observed/within/passed) | pre-clamp action (v/omega) | selected post-clamp command (v/omega) | applied environment action (linear/angular) | candidate transition (v mean/weighted; weight mean) | heading/goal first (decl/vel/goal rad) | goal start -> end (m) | phase progress (m) | displacement (m) | interaction share | min clearance (m) | terminal | collision step | goal step |",
            "| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in report["mechanism_table"]:
        goal = row.get("goal", {})
        runtime = row.get("runtime", {})
        heading = row.get("heading", {})
        motion = row.get("motion", {})
        interaction = row.get("interaction_zone", {})
        clearance = row.get("clearance", {})
        nominal_text = _markdown_nominal_command(row.get("nominal_command"))
        pedestrian_selection_text = _markdown_pedestrian_selection(row.get("pedestrian_selection"))
        candidate_transition = _markdown_candidate_transition(
            row.get("aggregation", {}).get("candidate_distribution")
            if isinstance(row.get("aggregation"), dict)
            else None
        )
        pre_clamp_text = _markdown_action_summary(row.get("pre_clamp_action"), include_clipped=True)
        selected_text = _markdown_action_summary(row.get("selected_post_clamp_command"))
        applied_text = _markdown_action_summary(row.get("applied_environment_command"))
        heading_text = (
            f"{_markdown_cell(heading.get('declared_first_rad'))}/"
            f"{_markdown_cell(heading.get('velocity_derived_first_rad'))}/"
            f"{_markdown_cell(heading.get('goal_bearing_first_rad'))}"
        )
        phase_progress = goal.get("signed_progress_by_phase")
        phase_text = (
            ", ".join(
                f"{phase['phase']}={phase['signed_progress_m']:.3f}" for phase in phase_progress
            )
            if isinstance(phase_progress, list)
            else "not available"
        )
        start = goal.get("initial_distance_m")
        end = goal.get("final_distance_m")
        goal_text = (
            f"{start:.3f} -> {end:.3f}"
            if isinstance(start, (float, int)) and isinstance(end, (float, int))
            else "not available"
        )
        lines.append(
            f"| {row.get('planner', 'unknown')} | {row.get('seed', 'unknown')} | "
            f"{row.get('status', 'unavailable')} | {runtime.get('status', 'not available')} | "
            f"{runtime.get('failure_count', 'not available')} | {nominal_text} | "
            f"{pedestrian_selection_text} | {pre_clamp_text} | {selected_text} | {applied_text} | "
            f"{candidate_transition} | {heading_text} | {goal_text} | {phase_text} | "
            f"{_markdown_cell(motion.get('displacement_m'))} | "
            f"{_markdown_cell(interaction.get('exposure_share'))} | "
            f"{_markdown_cell(clearance.get('min_clearance_m'))} | "
            f"{_markdown_cell(row.get('events', {}).get('termination_reason'))} | "
            f"{_markdown_cell(row.get('events', {}).get('collision_step'))} | "
            f"{_markdown_cell(row.get('events', {}).get('goal_step'))} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(report["config"]["claim_boundary"]),
            "",
            "A later benchmark-arm proposal requires a separately approved preregistration and a broader evidence contract.",
        ]
    )
    markdown_path = output_dir / "diagnostic_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def run_campaign(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Execute all predeclared arms and write the diagnostic report."""
    selected_scenarios = select_scenarios(config)
    arms: list[dict[str, Any]] = []
    for planner in config["planners"]:
        key = str(planner["key"])
        arm_dir = output_dir / key
        episodes_path = arm_dir / "episodes.jsonl"
        try:
            execution_summary = run_map_batch(
                selected_scenarios,
                episodes_path,
                REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
                scenario_path=config["scenario_matrix"],
                horizon=int(config["horizon"]),
                dt=float(config["dt"]),
                record_forces=True,
                algo=key,
                algo_config_path=planner["config_path"],
                benchmark_profile="experimental",
                socnav_missing_prereq_policy="fail-fast",
                record_simulation_step_trace=True,
                workers=1,
                resume=False,
            )
            records = _read_jsonl(episodes_path)
            arms.append(
                summarize_records(
                    planner_key=key,
                    records=records,
                    config=config,
                    execution_summary=execution_summary,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed arm must be reported, not promoted.
            arms.append(
                summarize_records(
                    planner_key=key,
                    records=_read_jsonl(episodes_path),
                    config=config,
                    execution_summary=None,
                    error=str(exc),
                )
            )
    expected_pairs = len(config["scenario_ids"]) * len(config["seeds"])
    complete = all(
        arm["status"] == "available"
        and arm["pair_coverage_exact"]
        and arm["observed_rows"] == expected_pairs
        and arm["unavailable_rows"] == 0
        for arm in arms
    )
    report: dict[str, Any] = {
        "schema_version": "brne-corridor-diagnostic-report.v1",
        "status": "diagnostic_complete" if complete else "diagnostic_incomplete",
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "expected_pairs": expected_pairs,
        "paired_coverage_exact": all(
            arm["pair_coverage_exact"] and arm["observed_rows"] == expected_pairs for arm in arms
        ),
        "arms": arms,
        "mechanism_table": [row for arm in arms for row in arm["mechanism_table"]],
        "claim_boundary": config["claim_boundary"],
    }
    json_path, markdown_path = _write_report(report, output_dir)
    report["report_paths"] = {"json": str(json_path), "markdown": str(markdown_path)}
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    """Run or preflight the bounded BRNE diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config = validate_campaign_config(_load_mapping(args.config.resolve()))
    selected = select_scenarios(config)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_ok",
                    "scenario_ids": [str(scenario["name"]) for scenario in selected],
                    "seeds": config["seeds"],
                    "planners": [planner["key"] for planner in config["planners"]],
                    "claim_boundary": config["claim_boundary"],
                },
                sort_keys=True,
            )
        )
        return 0

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        REPO_ROOT / "output/benchmarks" / f"issue_6464_brne_{timestamp}"
    )
    report = run_campaign(config, output_dir=output_dir.resolve())
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir.resolve())}))
    return 0 if report["status"] == "diagnostic_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
