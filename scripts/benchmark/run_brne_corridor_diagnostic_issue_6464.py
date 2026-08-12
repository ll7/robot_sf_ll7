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
from robot_sf.benchmark.brne_trace_diagnostic import build_trace_table
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
        "effective_num_samples": 42,
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
        expected_effective_num_samples = int(brne_config["effective_num_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BRNE effective_num_samples must be a positive integer") from exc
    if expected_effective_num_samples < 1:
        raise ValueError("BRNE effective_num_samples must be a positive integer")
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
        radius = float(corridor["robot_radius_m"])
        center_lower = lower + radius
        center_upper = upper - radius
        violation_count = sum(y < center_lower or y > center_upper for _, y in positions)

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
    )
    native = (
        planner_key == "brne"
        and execution_ok
        and planner_status == "ok"
        and diagnostic_metadata_valid
        and runtime_provenance_valid
        and canonical_metadata_valid
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


def _build_trace_tables(
    records: list[dict[str, Any]],
    *,
    planner_key: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build trace tables with exact eligible pair coverage.

    Returns:
        List of trace table dicts, one for every declared eligible pair.

    Raises:
        ValueError: If any declared row is missing, duplicated, unavailable, or malformed.
    """
    expected = {
        (str(scenario_id), int(seed))
        for scenario_id in config["scenario_ids"]
        for seed in config["seeds"]
    }
    observed: list[tuple[str, int]] = []
    for record in records:
        try:
            pair = (str(record["scenario_id"]), int(record["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("trace table row has an invalid scenario/seed pair") from exc
        observed.append(pair)
    duplicates = sorted(pair for pair in set(observed) if observed.count(pair) > 1)
    observed_set = set(observed)
    if len(observed) != len(expected) or duplicates or observed_set != expected:
        raise ValueError(
            "trace table pair coverage is not exact: "
            f"missing={sorted(expected - observed_set)}, "
            f"unexpected={sorted(observed_set - expected)}, duplicates={duplicates}"
        )
    tables: list[dict[str, Any]] = []
    for record in records:
        classified = classify_record(record, config, planner_key=planner_key)
        if not classified["execution_ok"]:
            raise ValueError(
                f"{planner_key} trace row {record.get('episode_id')} is not execution-admissible: "
                f"{classified['status']}"
            )
        if not classified["crowd_within_budget"]:
            raise ValueError(
                f"{planner_key} trace row {record.get('episode_id')} exceeds the crowd cap"
            )
        tables.append(
            build_trace_table(
                record,
                planner_key=planner_key,
                expected_effective_num_samples=(
                    int(config["expected_effective_num_samples"]) if planner_key == "brne" else None
                ),
            )
        )
    return tables


def _write_trace_tables(
    tables: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write trace tables as compact JSONL to the output directory.

    Returns:
        Path to the written trace tables file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace_tables.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for table in tables:
            f.write(json.dumps(table, sort_keys=True) + "\n")
    return trace_path


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
        "| planner | status | observed | exact pairs | trace tables | native | eligible | unavailable | goal reached | non-degenerate | corridor violations |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in report["arms"]:
        trace_table_status = arm.get("trace_table_status", "unavailable")
        trace_table_count = arm.get("trace_table_count")
        trace_table_cell = (
            f"{trace_table_status} ({trace_table_count}/{report['expected_pairs']})"
            if trace_table_count is not None
            else trace_table_status
        )
        lines.append(
            f"| {arm['planner']} | {arm['status']} | {arm['observed_rows']} | "
            f"{'yes' if arm['pair_coverage_exact'] else 'no'} | {trace_table_cell} | "
            f"{arm['native_rows']} | {arm['diagnostic_eligible_rows']} | "
            f"{arm['unavailable_rows']} | "
            f"{arm['goal_reached_rows']} | {arm['nondegenerate_rows']} | "
            f"{arm['corridor_violation_rows']} |"
        )
    trace_errors = [
        f"- `{arm['planner']}`: {arm['trace_table_error']}"
        for arm in report["arms"]
        if arm.get("trace_table_error")
    ]
    if trace_errors:
        lines.extend(["", "Trace-table exclusions:", "", *trace_errors])
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
    trace_table_paths: list[dict[str, Any]] = []
    for planner in config["planners"]:
        key = str(planner["key"])
        arm_dir = output_dir / key
        episodes_path = arm_dir / "episodes.jsonl"
        execution_summary: dict[str, Any] | None = None
        execution_error: str | None = None
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
        except Exception as exc:  # noqa: BLE001 - a failed arm must be reported, not promoted.
            execution_error = str(exc)
        records = _read_jsonl(episodes_path)
        arm = summarize_records(
            planner_key=key,
            records=records,
            config=config,
            execution_summary=execution_summary,
            error=execution_error,
        )
        try:
            trace_tables = _build_trace_tables(records, planner_key=key, config=config)
        except (ValueError, KeyError) as exc:
            arm["trace_table_status"] = "unavailable"
            arm["trace_table_error"] = str(exc)
        else:
            trace_path = _write_trace_tables(trace_tables, arm_dir)
            arm["trace_table_status"] = "available"
            arm["trace_table_count"] = len(trace_tables)
            trace_table_paths.append(
                {
                    "planner": key,
                    "trace_tables_path": str(trace_path),
                    "trace_table_count": len(trace_tables),
                }
            )
        arms.append(arm)
    expected_pairs = len(config["scenario_ids"]) * len(config["seeds"])
    complete = all(
        arm["status"] == "available"
        and arm["pair_coverage_exact"]
        and arm["observed_rows"] == expected_pairs
        and arm["unavailable_rows"] == 0
        and arm.get("trace_table_status") == "available"
        and arm.get("trace_table_count") == expected_pairs
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
        "trace_tables": trace_table_paths,
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
