#!/usr/bin/env python3
"""Run or fail-close the issue #2777 live observation-perturbation replay batch."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.training.scenario_loader import load_scenarios

SCHEMA_VERSION = "issue_2777_observation_noise_live_replay.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_2777_live_observation_noise_replay")
DEFAULT_SCENARIO_MATRIX = Path(
    "configs/scenarios/sets/issue_3320_occluded_emergence_live_replay.yaml"
)
DEFAULT_CANDIDATE = "risk_surface_dwa_v0"
DEFAULT_STAGE = "smoke"
TRACE_RUNNER = Path("scripts/validation/run_policy_search_step_diagnostics.py")
ISSUE_2756_FIXTURE_SCENARIO = "issue_2756_occluded_emergence"
ISSUE_2756_FIXTURE_FAMILY = "occluded_emergence"
ISSUE_2756_FIXTURE_LABEL = "deterministic_occluded_emergence"
ISSUE_2756_FIXTURE_SEED = 111
REQUIRED_CONDITIONS = (
    "noop",
    "low_noise",
    "medium_noise",
    "missed_detection_only",
    "false_positive_only",
    "occlusion_only",
    "delay_only",
    "combined",
)
DEFAULT_CONDITION_SET = "issue_2755_default"
ISSUE_3328_CONDITION_SET = "issue_3328_behavior_probe"
ISSUE_3330_CONDITION_SET = "issue_3330_seed_amplitude_grid"
ISSUE_3328_REQUIRED_CONDITIONS = (
    "noop",
    "medium_noise",
    "delay_only",
    "high_noise_3328",
)
ISSUE_3330_MEDIUM_CONDITIONS = (
    "medium_noise_2755",
    "medium_noise_3328",
    "medium_noise_3330",
)
ISSUE_3330_HIGH_CONDITIONS = (
    "high_noise_2755",
    "high_noise_3328",
    "high_noise_3330",
)
ISSUE_3330_REQUIRED_CONDITIONS = (
    "noop",
    "delay_only",
    *ISSUE_3330_MEDIUM_CONDITIONS,
    *ISSUE_3330_HIGH_CONDITIONS,
)
PROGRESS_FIELDS = (
    "net_goal_progress",
    "best_goal_progress",
    "closest_robot_ped_distance",
    "closest_robot_ped_step",
    "collision_flag_counts",
    "progress_step_count",
    "regression_step_count",
    "stagnant_step_count",
    "longest_stagnant_run",
)


@dataclass(frozen=True)
class Condition:
    """One #2755 perturbation family expressed as diagnostics-runner CLI flags."""

    name: str
    description: str
    flags: tuple[str, ...] = ()


CONDITIONS = (
    Condition("noop", "Unperturbed live planner/environment replay."),
    Condition(
        "low_noise",
        "Bounded Gaussian pedestrian-position perturbation, std=0.10 m, bound=0.20 m.",
        (
            "--observation-noise-std-m",
            "0.10",
            "--observation-noise-bound-m",
            "0.20",
            "--observation-perturbation-seed",
            "2755",
        ),
    ),
    Condition(
        "medium_noise",
        "Bounded Gaussian pedestrian-position perturbation, std=0.30 m, bound=0.60 m.",
        (
            "--observation-noise-std-m",
            "0.30",
            "--observation-noise-bound-m",
            "0.60",
            "--observation-perturbation-seed",
            "2755",
        ),
    ),
    Condition(
        "missed_detection_only",
        "All live pedestrians removed from planner input by missed-detection probability 1.0.",
        ("--missed-detection-probability", "1.0", "--observation-perturbation-seed", "2755"),
    ),
    Condition(
        "false_positive_only",
        "One deterministic observed-only pedestrian injected into planner input.",
        (
            "--false-positive-actor-count",
            "1",
            "--false-positive-offset-x-m",
            "1.0",
            "--false-positive-offset-y-m",
            "0.0",
            "--observation-perturbation-seed",
            "2755",
        ),
    ),
    Condition(
        "occlusion_only",
        "All live pedestrians occluded from planner input with a zero-distance occlusion gate.",
        ("--occlusion-distance-m", "0.0"),
    ),
    Condition(
        "delay_only",
        "Two-step delayed pedestrian observation, preserving the #2755 expected lag.",
        ("--observation-delay-steps", "2"),
    ),
    Condition(
        "combined",
        "Medium Gaussian perturbation plus full live-pedestrian occlusion.",
        (
            "--observation-noise-std-m",
            "0.30",
            "--observation-noise-bound-m",
            "0.60",
            "--occlusion-distance-m",
            "0.0",
            "--observation-perturbation-seed",
            "2755",
        ),
    ),
)
_CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}
HIGH_NOISE_3328_CONDITION = Condition(
    "high_noise_3328",
    "Issue #3328 high-noise behavior probe, std=1.0 m, bound=2.0 m, seed=3328.",
    (
        "--observation-noise-std-m",
        "1.0",
        "--observation-noise-bound-m",
        "2.0",
        "--observation-perturbation-seed",
        "3328",
    ),
)


def _noise_condition(name: str, *, std_m: str, bound_m: str, seed: int) -> Condition:
    """Return a bounded-Gaussian condition with explicit seed/amplitude metadata."""
    amplitude = "medium" if std_m == "0.30" else "high"
    return Condition(
        name,
        (
            f"Issue #3330 {amplitude}-noise seed/amplitude grid condition, "
            f"std={std_m} m, bound={bound_m} m, seed={seed}."
        ),
        (
            "--observation-noise-std-m",
            std_m,
            "--observation-noise-bound-m",
            bound_m,
            "--observation-perturbation-seed",
            str(seed),
        ),
    )


ISSUE_3330_GRID_CONDITIONS = (
    _CONDITION_BY_NAME["noop"],
    _CONDITION_BY_NAME["delay_only"],
    _noise_condition("medium_noise_2755", std_m="0.30", bound_m="0.60", seed=2755),
    _noise_condition("medium_noise_3328", std_m="0.30", bound_m="0.60", seed=3328),
    _noise_condition("medium_noise_3330", std_m="0.30", bound_m="0.60", seed=3330),
    _noise_condition("high_noise_2755", std_m="1.00", bound_m="2.00", seed=2755),
    _noise_condition("high_noise_3328", std_m="1.00", bound_m="2.00", seed=3328),
    _noise_condition("high_noise_3330", std_m="1.00", bound_m="2.00", seed=3330),
)
CONDITION_SETS = {
    DEFAULT_CONDITION_SET: CONDITIONS,
    ISSUE_3328_CONDITION_SET: (
        _CONDITION_BY_NAME["noop"],
        _CONDITION_BY_NAME["medium_noise"],
        _CONDITION_BY_NAME["delay_only"],
        HIGH_NOISE_3328_CONDITION,
    ),
    ISSUE_3330_CONDITION_SET: ISSUE_3330_GRID_CONDITIONS,
}


def _repo_path(path: Path) -> Path:
    """Resolve a repository-relative path."""
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_matrix_yaml(matrix_path: Path) -> None:
    """Raise a stable error when a selected scenario matrix is not valid YAML."""
    try:
        yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{_durable_ref(matrix_path)} is not valid YAML: {exc}") from exc


def _scenario_name(scenario: Mapping[str, Any]) -> str:
    """Return a scenario identifier from common scenario fields."""
    return str(scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "")


def _metadata_mapping(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return scenario metadata as a plain mapping."""
    metadata = scenario.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _fixture_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the nested fixture contract metadata as a plain mapping."""
    fixture = metadata.get("fixture_contract")
    return dict(fixture) if isinstance(fixture, Mapping) else {}


def _metadata_field(metadata: Mapping[str, Any], fixture: Mapping[str, Any], key: str) -> Any:
    """Read fixture metadata, preferring the nested fixture-contract block."""
    return fixture.get(key, metadata.get(key))


def _source_issue_matches(value: Any) -> bool:
    """Return whether a source-issue metadata value points at issue #2756."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value == 2756
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized == "2756" or normalized.endswith("/issues/2756")
    return False


def _optional_int(value: Any) -> int | None:
    """Return a non-boolean integer when coercion is exact enough for metadata checks."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int]:
    """Parse a list of integer-like values while ignoring invalid entries."""
    if not isinstance(value, list):
        return []
    parsed: list[int] = []
    for item in value:
        parsed_item = _optional_int(item)
        if parsed_item is not None:
            parsed.append(parsed_item)
    return parsed


def _fixture_candidate(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the #2756 fixture-facing metadata from one loaded scenario."""
    metadata = _metadata_mapping(scenario)
    fixture = _fixture_mapping(metadata)
    return {
        "name": _scenario_name(scenario),
        "source_issue": metadata.get("source_issue", fixture.get("source_issue")),
        "family": str(metadata.get("family") or scenario.get("scenario_family") or ""),
        "label": str(metadata.get("label") or fixture.get("label") or ""),
        "seeds": _int_list(scenario.get("seeds")),
        "first_visible_step": _optional_int(
            _metadata_field(metadata, fixture, "first_visible_step")
        ),
        "delay_steps": _optional_int(_metadata_field(metadata, fixture, "delay_steps")),
        "delay_only_expected_first_observed_step": _optional_int(
            _metadata_field(
                metadata,
                fixture,
                "delay_only_expected_first_observed_step",
            )
        ),
    }


def _fixture_candidate_reasons(candidate: Mapping[str, Any]) -> list[str]:
    """Return fail-closed reasons for a candidate #2756 fixture scenario."""
    reasons: list[str] = []
    if candidate["name"] != ISSUE_2756_FIXTURE_SCENARIO:
        reasons.append(
            f"scenario name is {candidate['name']!r}, expected {ISSUE_2756_FIXTURE_SCENARIO!r}"
        )
    if not _source_issue_matches(candidate["source_issue"]):
        reasons.append("metadata.source_issue does not point at issue #2756")
    if candidate["family"] != ISSUE_2756_FIXTURE_FAMILY:
        reasons.append(
            f"metadata family is {candidate['family']!r}, expected {ISSUE_2756_FIXTURE_FAMILY!r}"
        )
    if candidate["label"] != ISSUE_2756_FIXTURE_LABEL:
        reasons.append(
            f"metadata label is {candidate['label']!r}, expected {ISSUE_2756_FIXTURE_LABEL!r}"
        )
    if ISSUE_2756_FIXTURE_SEED not in candidate["seeds"]:
        reasons.append(f"scenario seeds do not include {ISSUE_2756_FIXTURE_SEED}")
    for key, expected in (
        ("first_visible_step", 5),
        ("delay_steps", 2),
        ("delay_only_expected_first_observed_step", 7),
    ):
        if candidate.get(key) != expected:
            reasons.append(f"{key} is {candidate.get(key)!r}, expected {expected!r}")
    return reasons


def _fixture_contract(matrix_path: Path) -> dict[str, Any]:
    """Check whether a live matrix preserves the #2755/#2756 fixture boundary."""
    fixture_contract: dict[str, Any] = {
        "required_source_issue": 2756,
        "required_scenario": ISSUE_2756_FIXTURE_SCENARIO,
        "required_family": f"{ISSUE_2756_FIXTURE_FAMILY}/{ISSUE_2756_FIXTURE_LABEL}",
        "required_seed": ISSUE_2756_FIXTURE_SEED,
        "first_visible_step": 5,
        "delay_steps": 2,
        "delay_only_expected_first_observed_step": 7,
        "scenario_matrix": _durable_ref(matrix_path),
        "satisfied": False,
        "blocker": None,
        "matched_scenario": None,
    }
    try:
        _validate_matrix_yaml(matrix_path)
        scenarios = load_scenarios(matrix_path)
    except ValueError as exc:
        fixture_contract["blocker"] = str(exc)
        return fixture_contract
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        fixture_contract["blocker"] = (
            f"{_durable_ref(matrix_path)} could not be loaded as a scenario matrix: {exc}"
        )
        return fixture_contract

    candidates = [_fixture_candidate(dict(scenario)) for scenario in scenarios]
    named_candidates = [
        candidate for candidate in candidates if candidate["name"] == ISSUE_2756_FIXTURE_SCENARIO
    ]
    if not named_candidates:
        fixture_contract["blocker"] = (
            "No checked-in live scenario matrix preserving the #2755/#2756 "
            "occluded-emergence fixture boundary was found in the selected matrix."
        )
        return fixture_contract

    candidate = named_candidates[0]
    reasons = _fixture_candidate_reasons(candidate)
    fixture_contract["matched_scenario"] = candidate
    if reasons:
        fixture_contract["blocker"] = (
            "Selected matrix contains issue_2756_occluded_emergence, but it does not "
            f"preserve the required fixture contract: {'; '.join(reasons)}."
        )
        return fixture_contract

    fixture_contract["satisfied"] = True
    return fixture_contract


def _write_generated_funnel(
    *,
    output_dir: Path,
    scenario_matrix: Path,
    stage: str,
    horizon: int,
) -> Path:
    """Write a tiny funnel config that points diagnostics at the selected matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    funnel = {
        "stage_order": [stage],
        "stages": {
            stage: {
                "scenario_matrix": _repo_path(scenario_matrix).as_posix(),
                "seed_list": [ISSUE_2756_FIXTURE_SEED],
                "benchmark_profile": "experimental",
                "horizon": int(horizon),
                "dt": 0.1,
                "workers": 1,
                "requires_slurm": False,
            }
        },
    }
    path = output_dir / "generated_policy_search_funnel.yaml"
    path.write_text(yaml.safe_dump(funnel, sort_keys=False), encoding="utf-8")
    return path


def _condition_command(
    *,
    condition: Condition,
    output_dir: Path,
    funnel_config: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build one live diagnostics subprocess command."""
    condition_dir = output_dir / "traces" / condition.name
    command = [
        sys.executable,
        str(_repo_path(TRACE_RUNNER)),
        "--candidate",
        args.candidate,
        "--stage",
        args.stage,
        "--candidate-registry",
        str(_repo_path(args.candidate_registry)),
        "--funnel-config",
        str(funnel_config),
        "--scenario-index",
        str(args.scenario_index),
        "--seed-index",
        str(args.seed_index),
        "--horizon",
        str(args.horizon),
        "--output-dir",
        str(condition_dir),
    ]
    if args.scenario_name:
        command.extend(["--scenario-name", args.scenario_name])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    command.extend(condition.flags)
    return command


def _selected_conditions(args: argparse.Namespace) -> tuple[Condition, ...]:
    """Return the conditions selected by the requested condition set."""
    return CONDITION_SETS[str(args.condition_set)]


def _durable_ref(path: Path) -> str:
    """Return a report-safe path reference."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_trace(path: Path) -> dict[str, Any]:
    """Load a diagnostics trace."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("steps"), list):
        raise ValueError(f"{path} is not a diagnostics trace")
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    """Return mapping values and coerce JSON null to empty."""
    return value if isinstance(value, dict) else {}


def _finite_number(value: Any) -> float | None:
    """Return a finite non-boolean number, otherwise ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _values_differ(left: Any, right: Any) -> bool:
    """Compare report values while treating paired NaN values as unchanged."""
    if isinstance(left, int | float) and isinstance(right, int | float):
        left_float = float(left)
        right_float = float(right)
        if math.isnan(left_float) and math.isnan(right_float):
            return False
    return left != right


def _commands(trace: dict[str, Any]) -> list[Any]:
    """Return selected policy commands from a diagnostics trace."""
    return [_mapping(row).get("policy_command") for row in trace.get("steps", [])]


def _observation_totals(trace: dict[str, Any]) -> dict[str, Any]:
    """Summarize perturbation metadata across trace rows."""
    totals = {
        "missed_actor_observations_total": 0,
        "occluded_actor_observations_total": 0,
        "visibility_hidden_actor_observations_total": 0,
        "false_positive_actor_observations_total": 0,
        "min_observed_actor_count": None,
        "max_observed_actor_count": None,
        "noise_profiles": set(),
    }
    observed_counts: list[int] = []
    for row in trace.get("steps", []):
        meta = _mapping(row.get("observation_perturbation"))
        totals["missed_actor_observations_total"] += int(meta.get("missed_actor_count", 0) or 0)
        totals["occluded_actor_observations_total"] += int(meta.get("occluded_actor_count", 0) or 0)
        totals["visibility_hidden_actor_observations_total"] += int(
            meta.get("visibility_hidden_actor_count", 0) or 0
        )
        totals["false_positive_actor_observations_total"] += int(
            meta.get("false_positive_actor_count", 0) or 0
        )
        observed_counts.append(int(meta.get("observed_actor_count", 0) or 0))
        profile = meta.get("noise_profile")
        if profile:
            totals["noise_profiles"].add(str(profile))
    totals["min_observed_actor_count"] = min(observed_counts) if observed_counts else None
    totals["max_observed_actor_count"] = max(observed_counts) if observed_counts else None
    totals["noise_profiles"] = sorted(totals["noise_profiles"])
    return totals


def _progress_delta(noop: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    """Compare selected progress/risk summary fields."""
    noop_summary = _mapping(noop.get("progress_summary"))
    condition_summary = _mapping(condition.get("progress_summary"))
    return {
        field: {
            "noop": noop_summary.get(field),
            "condition": condition_summary.get(field),
            "changed": _values_differ(noop_summary.get(field), condition_summary.get(field)),
        }
        for field in PROGRESS_FIELDS
    }


def _near_miss_total(trace: dict[str, Any]) -> tuple[float | None, bool]:
    """Return the summed near-miss metadata and whether the field was reported."""
    total = 0.0
    available = False
    for row in trace.get("steps", []):
        meta = _mapping(_mapping(row).get("meta"))
        if "near_misses" not in meta:
            continue
        available = True
        try:
            value = float(meta.get("near_misses", 0) or 0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            total += value
    return (total if available else None), available


def _near_miss_summary(
    noop_trace: dict[str, Any], condition_trace: dict[str, Any]
) -> dict[str, Any]:
    """Compare near-miss counts when diagnostics traces expose that field."""
    noop_total, noop_available = _near_miss_total(noop_trace)
    condition_total, condition_available = _near_miss_total(condition_trace)
    if not noop_available and not condition_available:
        return {
            "status": "unavailable",
            "noop": None,
            "condition": None,
            "changed": False,
            "limitation": "Diagnostics trace rows did not expose meta.near_misses.",
        }
    status = "available" if noop_available and condition_available else "partial"
    return {
        "status": status,
        "noop": noop_total,
        "condition": condition_total,
        "changed": (
            noop_total != condition_total if noop_available and condition_available else False
        ),
    }


def _collision_summary(
    noop_trace: dict[str, Any], condition_trace: dict[str, Any]
) -> dict[str, Any]:
    """Compare progress-summary collision counts."""
    noop_counts = _mapping(noop_trace.get("progress_summary")).get("collision_flag_counts")
    condition_counts = _mapping(condition_trace.get("progress_summary")).get(
        "collision_flag_counts"
    )
    if noop_counts is None and condition_counts is None:
        return {
            "status": "unavailable",
            "noop": None,
            "condition": None,
            "changed": False,
            "limitation": "Diagnostics progress summaries did not expose collision_flag_counts.",
        }
    status = "available" if noop_counts is not None and condition_counts is not None else "partial"
    return {
        "status": status,
        "noop": noop_counts,
        "condition": condition_counts,
        "changed": noop_counts != condition_counts if status == "available" else False,
    }


def _command_speed(command: Any) -> float | None:
    """Return the linear-speed component from a policy command when present."""
    if not isinstance(command, list | tuple) or not command:
        return None
    value = command[0]
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_command_stop_step(trace: dict[str, Any]) -> int | None:
    """Return the first step whose selected command has zero linear speed."""
    for fallback_step, row in enumerate(trace.get("steps", [])):
        row = _mapping(row)
        speed = _command_speed(row.get("policy_command"))
        if speed is not None and abs(speed) <= 1e-9:
            return _optional_int(row.get("step")) or fallback_step
    return None


def _has_command_speed(trace: dict[str, Any]) -> bool:
    """Return whether any trace row has a parseable selected linear-speed command."""
    return any(
        _command_speed(_mapping(row).get("policy_command")) is not None
        for row in trace.get("steps", [])
    )


def _changed_command_steps(
    noop_trace: dict[str, Any], condition_trace: dict[str, Any]
) -> list[int]:
    """Return trace steps where the selected policy command differs from no-op."""
    noop_steps = [_mapping(row) for row in noop_trace.get("steps", [])]
    condition_steps = [_mapping(row) for row in condition_trace.get("steps", [])]
    changed: list[int] = []
    for index in range(max(len(noop_steps), len(condition_steps))):
        noop_row = noop_steps[index] if index < len(noop_steps) else {}
        condition_row = condition_steps[index] if index < len(condition_steps) else {}
        if noop_row.get("policy_command") == condition_row.get("policy_command"):
            continue
        step = _optional_int(condition_row.get("step"))
        if step is None:
            step = _optional_int(noop_row.get("step"))
        changed.append(index if step is None else step)
    return changed


def _stop_yield_timing_proxy(
    noop_trace: dict[str, Any],
    condition_trace: dict[str, Any],
) -> dict[str, Any]:
    """Report the available stop/yield proxy without inventing direct event timing."""
    noop_available = _has_command_speed(noop_trace)
    condition_available = _has_command_speed(condition_trace)
    noop_stop = _first_command_stop_step(noop_trace)
    condition_stop = _first_command_stop_step(condition_trace)
    status = "available" if noop_available and condition_available else "unavailable"
    return {
        "status": status,
        "definition": (
            "First diagnostics trace step where the selected policy_command linear "
            "speed component is present and abs(speed_mps) <= 1e-9. Direct stop/yield "
            "event timing is not available in these traces."
        ),
        "direct_event_available": False,
        "noop_first_stop_step": noop_stop if noop_available else None,
        "condition_first_stop_step": condition_stop if condition_available else None,
        "changed": noop_stop != condition_stop if status == "available" else False,
    }


def _behavior_change_summary(
    noop_trace: dict[str, Any],
    condition_trace: dict[str, Any],
) -> dict[str, Any]:
    """Name behavior-change dimensions used for #2777/#3328 interpretation."""
    progress_delta = _progress_delta(noop_trace, condition_trace)
    changed_steps = _changed_command_steps(noop_trace, condition_trace)
    near_miss = _near_miss_summary(noop_trace, condition_trace)
    collision = _collision_summary(noop_trace, condition_trace)
    stop_proxy = _stop_yield_timing_proxy(noop_trace, condition_trace)
    changed_progress_fields = [
        field for field, payload in progress_delta.items() if payload["changed"]
    ]
    return {
        "command_sequence_changed": bool(changed_steps),
        "changed_command_steps": changed_steps,
        "changed_command_step_count": len(changed_steps),
        "progress_or_risk_changed": bool(changed_progress_fields),
        "progress_delta_changed_fields": changed_progress_fields,
        "closest_robot_ped": {
            "distance": progress_delta["closest_robot_ped_distance"],
            "step": progress_delta["closest_robot_ped_step"],
        },
        "min_distance_changed": progress_delta["closest_robot_ped_distance"]["changed"],
        "collision_or_near_miss_changed": (
            bool(collision["changed"]) or bool(near_miss["changed"])
        ),
        "collision_summary": collision,
        "near_miss_summary": near_miss,
        "stop_yield_timing_proxy": stop_proxy,
        "stop_yield_timing": {
            "direct_event_available": False,
            "limitation": (
                "Direct stop/yield event timing is not available in diagnostics traces; "
                "only a zero-linear-speed command proxy is reported."
            ),
            "command_stop_proxy": {
                "definition": stop_proxy["definition"],
                "noop_first_stop_step": stop_proxy["noop_first_stop_step"],
                "condition_first_stop_step": stop_proxy["condition_first_stop_step"],
                "changed": stop_proxy["changed"],
            },
        },
    }


def _first_observed_step(trace: dict[str, Any]) -> int | None:
    """Return the first trace step with at least one planner-visible actor."""
    for row in trace.get("steps", []):
        meta = _mapping(_mapping(row).get("observation_perturbation"))
        if int(meta.get("observed_actor_count", 0) or 0) > 0:
            return _optional_int(row.get("step"))
    return None


def _classification(
    *,
    noop_trace: dict[str, Any],
    condition_trace: dict[str, Any],
    fixture_contract_satisfied: bool,
) -> dict[str, str]:
    """Classify one live condition against the no-op trace."""
    behavior = _behavior_change_summary(noop_trace, condition_trace)
    observation_changed = _observation_totals(noop_trace) != _observation_totals(condition_trace)
    closest = _mapping(noop_trace.get("progress_summary")).get("closest_robot_ped_distance")
    finite_closest = _finite_number(closest)
    near_field = finite_closest is not None and finite_closest <= 2.0
    if not fixture_contract_satisfied:
        return {
            "label": "diagnostic_only",
            "rationale": (
                "Live replay ran on a proxy scenario, not the #2755/#2756 "
                "occluded-emergence fixture boundary."
            ),
        }
    if (
        behavior["command_sequence_changed"]
        or behavior["progress_or_risk_changed"]
        or behavior["collision_or_near_miss_changed"]
        or behavior["stop_yield_timing_proxy"]["changed"]
    ):
        return {
            "label": "behavior_sensitive_diagnostic_only",
            "rationale": (
                "Perturbation changed selected commands or progress/risk fields. "
                "This is live behavior evidence, but one seed is not enough for "
                "a benchmark-strength robustness claim."
            ),
        }
    if observation_changed and near_field:
        return {
            "label": "policy_insensitive",
            "rationale": (
                "Perturbation changed planner-input observations in a near-field trace, "
                "but selected commands and progress/risk summaries were identical."
            ),
        }
    return {
        "label": "scenario_too_weak",
        "rationale": (
            "The live condition did not expose a near-field behavior difference "
            "against the no-op trace."
        ),
    }


def _compare_condition(
    *,
    noop_trace_path: Path,
    condition_trace_path: Path,
    fixture_contract_satisfied: bool,
) -> dict[str, Any]:
    """Compare one condition trace against the no-op trace."""
    noop_trace = _load_trace(noop_trace_path)
    condition_trace = _load_trace(condition_trace_path)
    behavior = _behavior_change_summary(noop_trace, condition_trace)
    return {
        "trace": _durable_ref(condition_trace_path),
        "scenario": {
            "noop": noop_trace.get("scenario_id"),
            "condition": condition_trace.get("scenario_id"),
            "same": noop_trace.get("scenario_id") == condition_trace.get("scenario_id"),
        },
        "seed": {
            "noop": noop_trace.get("seed"),
            "condition": condition_trace.get("seed"),
            "same": noop_trace.get("seed") == condition_trace.get("seed"),
        },
        "planner_mode": {
            "candidate": condition_trace.get("candidate"),
            "algo": condition_trace.get("algo"),
            "stage": condition_trace.get("stage"),
        },
        "observation_summary": {
            "noop": _observation_totals(noop_trace),
            "condition": _observation_totals(condition_trace),
        },
        "command_summary": {
            "sequence_changed": behavior["command_sequence_changed"],
            "changed_steps": behavior["changed_command_steps"],
            "changed_step_count": behavior["changed_command_step_count"],
            "noop_first": _commands(noop_trace)[0] if _commands(noop_trace) else None,
            "condition_first": _commands(condition_trace)[0]
            if _commands(condition_trace)
            else None,
            "noop_last": _commands(noop_trace)[-1] if _commands(noop_trace) else None,
            "condition_last": _commands(condition_trace)[-1]
            if _commands(condition_trace)
            else None,
        },
        "progress_delta": _progress_delta(noop_trace, condition_trace),
        "behavior_change_summary": behavior,
        "fixture_visibility": {
            "noop_first_observed_step": _first_observed_step(noop_trace),
            "condition_first_observed_step": _first_observed_step(condition_trace),
        },
        "classification": _classification(
            noop_trace=noop_trace,
            condition_trace=condition_trace,
            fixture_contract_satisfied=fixture_contract_satisfied,
        ),
    }


def _fail_closed_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    fixture_contract: dict[str, Any],
    blocker: str,
) -> dict[str, Any]:
    """Return a fail-closed report without running live diagnostics."""
    report = {
        "schema_version": SCHEMA_VERSION,
        "issue": 2777,
        "status": "fail_closed",
        "classification": {
            "label": "blocked",
            "rationale": blocker,
        },
        "claim_boundary": (
            "No benchmark-facing robustness claim. The command failed closed before "
            "live replay because the #2755/#2756 occluded-emergence fixture boundary "
            "could not be preserved."
        ),
        "fixture_contract": fixture_contract,
        "run_config": _run_config(args=args, output_dir=output_dir),
        "conditions": [
            {
                "name": condition.name,
                "description": condition.description,
                "status": "blocked",
                "blocker": blocker,
            }
            for condition in _selected_conditions(args)
        ],
        "blockers": [blocker],
    }
    if args.condition_set == ISSUE_3330_CONDITION_SET:
        report["grid_interpretation"] = _issue_3330_grid_interpretation(
            conditions=report["conditions"],
            blockers=report["blockers"],
        )
    return report


def _run_config(*, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    """Return the stable command configuration payload."""
    return {
        "candidate": args.candidate,
        "stage": args.stage,
        "scenario_matrix": _durable_ref(_repo_path(args.scenario_matrix)),
        "scenario_name": args.scenario_name,
        "scenario_index": args.scenario_index,
        "seed": args.seed,
        "seed_index": args.seed_index,
        "horizon": args.horizon,
        "output_dir": _durable_ref(output_dir),
        "condition_set": args.condition_set,
        "condition_names": [condition.name for condition in _selected_conditions(args)],
        "allow_non_occluded_live_fixture": bool(args.allow_non_occluded_live_fixture),
        "dry_run": bool(args.dry_run),
    }


def _write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    """Write JSON and Markdown report artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(_markdown(report), encoding="utf-8")


def _markdown_value(value: Any) -> str:
    """Return a compact table value for optional report fields."""
    return "n/a" if value is None else str(value)


def _markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report."""
    lines = [
        "# Issue #2777 Live Observation-Noise Replay",
        "",
        "## Status",
        "",
        f"- Status: `{report['status']}`",
        f"- Classification: `{report['classification']['label']}`",
        f"- Rationale: {report['classification']['rationale']}",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Fixture Contract",
        "",
    ]
    contract = report["fixture_contract"]
    for key in (
        "required_scenario",
        "required_family",
        "first_visible_step",
        "delay_steps",
        "delay_only_expected_first_observed_step",
        "scenario_matrix",
        "satisfied",
        "blocker",
    ):
        lines.append(f"- `{key}`: `{contract.get(key)}`")
    if report.get("grid_interpretation"):
        interpretation = _mapping(report.get("grid_interpretation"))
        lines.extend(
            [
                "",
                "## Grid Interpretation",
                "",
                f"- Evidence status: `{interpretation.get('evidence_status')}`",
                f"- Label: `{interpretation.get('label')}`",
                f"- Summary: {interpretation.get('summary')}",
                "- Sensitive conditions: "
                f"`{', '.join(interpretation.get('sensitive_conditions') or []) or 'none'}`",
                "- Medium-amplitude sensitive conditions: "
                f"`{', '.join(interpretation.get('medium_sensitive_conditions') or []) or 'none'}`",
                "- High-noise sensitive conditions: "
                f"`{', '.join(interpretation.get('high_sensitive_conditions') or []) or 'none'}`",
                f"- Limitation: {interpretation.get('limitation')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "| Condition | Status | Classification | Command changed | Progress/risk changed | "
            "Min distance changed | Collision/near-miss changed | Stop/yield proxy changed | Caveat |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for condition in report["conditions"]:
        classification = _mapping(condition.get("classification"))
        label = classification.get("label") or (
            "baseline" if condition["name"] == "noop" else condition.get("status", "")
        )
        behavior = _mapping(condition.get("behavior_change_summary"))
        stop_proxy = _mapping(behavior.get("stop_yield_timing_proxy"))
        lines.append(
            f"| `{condition['name']}` | `{condition['status']}` | "
            f"`{label}` | "
            f"`{_markdown_value(behavior.get('command_sequence_changed'))}` | "
            f"`{_markdown_value(behavior.get('progress_or_risk_changed'))}` | "
            f"`{_markdown_value(behavior.get('min_distance_changed'))}` | "
            f"`{_markdown_value(behavior.get('collision_or_near_miss_changed'))}` | "
            f"`{_markdown_value(stop_proxy.get('changed'))}` | "
            f"{condition.get('blocker') or classification.get('rationale', '')} |"
        )
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    lines.append("")
    return "\n".join(lines)


def _planned_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    fixture_contract: dict[str, Any],
    funnel_config: Path,
) -> dict[str, Any]:
    """Return a dry-run report with planned live commands."""
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 2777,
        "status": "diagnostic_only",
        "classification": {
            "label": "diagnostic_only",
            "rationale": "Dry run only; no live planner/environment replay was executed.",
        },
        "claim_boundary": (
            "Dry-run command plan only. This makes no benchmark-facing robustness claim."
        ),
        "fixture_contract": fixture_contract,
        "run_config": _run_config(args=args, output_dir=output_dir),
        "conditions": [
            {
                "name": condition.name,
                "description": condition.description,
                "status": "planned",
                "command": _condition_command(
                    condition=condition,
                    output_dir=output_dir,
                    funnel_config=funnel_config,
                    args=args,
                ),
            }
            for condition in _selected_conditions(args)
        ],
        "blockers": [] if fixture_contract["satisfied"] else [fixture_contract["blocker"]],
    }


def _execute_conditions(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    funnel_config: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run all condition subprocesses and collect raw condition statuses."""
    conditions: list[dict[str, Any]] = []
    blockers: list[str] = []
    for condition in _selected_conditions(args):
        command = _condition_command(
            condition=condition,
            output_dir=output_dir,
            funnel_config=funnel_config,
            args=args,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            blocker = (
                f"{condition.name} live replay timed out after "
                f"{args.timeout_seconds:.1f}s; stderr: {str(exc.stderr or '')[:500]}"
            )
            blockers.append(blocker)
            conditions.append(
                {
                    "name": condition.name,
                    "description": condition.description,
                    "status": "blocked",
                    "command": command,
                    "blocker": blocker,
                }
            )
            continue
        trace_path = output_dir / "traces" / condition.name / "trace.json"
        report_path = output_dir / "traces" / condition.name / "report.md"
        if completed.returncode != 0 or not trace_path.exists():
            blocker = (
                f"{condition.name} live replay failed with exit {completed.returncode}; "
                f"stderr: {completed.stderr.strip()[:500]}"
            )
            blockers.append(blocker)
            conditions.append(
                {
                    "name": condition.name,
                    "description": condition.description,
                    "status": "blocked",
                    "command": command,
                    "blocker": blocker,
                }
            )
            continue
        conditions.append(
            {
                "name": condition.name,
                "description": condition.description,
                "status": "live_replay",
                "command": command,
                "trace": _durable_ref(trace_path),
                "report": _durable_ref(report_path),
            }
        )
    return conditions, blockers


def _attach_condition_comparisons(
    *,
    output_dir: Path,
    conditions: list[dict[str, Any]],
    fixture_contract_satisfied: bool,
) -> list[str]:
    """Attach no-op comparisons to completed live conditions."""
    blockers: list[str] = []
    noop = next((item for item in conditions if item["name"] == "noop"), None)
    if noop is None or noop.get("status") != "live_replay":
        blockers.append("No no-op live replay trace was available for condition comparison.")
    else:
        noop_trace = output_dir / "traces" / "noop" / "trace.json"
        for item in conditions:
            if item["name"] == "noop" or item.get("status") != "live_replay":
                continue
            comparison = _compare_condition(
                noop_trace_path=noop_trace,
                condition_trace_path=output_dir / "traces" / item["name"] / "trace.json",
                fixture_contract_satisfied=fixture_contract_satisfied,
            )
            item.update(comparison)
    return blockers


def _verify_near_field_probe_contract_guardrails(
    *,
    fixture_contract: Mapping[str, Any],
    issue_label: str,
) -> list[str]:
    """Verify the static fixture-contract side of an opt-in near-field probe."""
    blockers: list[str] = []
    if not fixture_contract.get("satisfied"):
        blockers.append(f"{issue_label} requires fixture_contract.satisfied=true.")

    matched = _mapping(fixture_contract.get("matched_scenario"))
    if matched.get("name") != ISSUE_2756_FIXTURE_SCENARIO:
        blockers.append(
            f"{issue_label} requires scenario "
            f"{ISSUE_2756_FIXTURE_SCENARIO!r}; got {matched.get('name')!r}."
        )
    if ISSUE_2756_FIXTURE_SEED not in _int_list(matched.get("seeds")):
        blockers.append(
            f"{issue_label} requires seed {ISSUE_2756_FIXTURE_SEED} in matrix seeds; "
            f"got {matched.get('seeds')!r}."
        )
    return blockers


def _verify_near_field_probe_trace_identity(
    label: str,
    trace: Mapping[str, Any],
) -> list[str]:
    """Verify one opt-in probe trace still names the intended scenario and seed."""
    blockers: list[str] = []
    if trace.get("scenario_id") != ISSUE_2756_FIXTURE_SCENARIO:
        blockers.append(
            f"{label} trace scenario is {trace.get('scenario_id')!r}, "
            f"expected {ISSUE_2756_FIXTURE_SCENARIO!r}."
        )
    if _optional_int(trace.get("seed")) != ISSUE_2756_FIXTURE_SEED:
        blockers.append(
            f"{label} trace seed is {trace.get('seed')!r}, expected {ISSUE_2756_FIXTURE_SEED!r}."
        )
    return blockers


def _verify_near_field_probe_trace(
    noop_trace: Mapping[str, Any],
    *,
    issue_label: str,
) -> list[str]:
    """Verify an opt-in probe no-op trace has near-field interaction geometry."""
    closest = _mapping(noop_trace.get("progress_summary")).get("closest_robot_ped_distance")
    finite_closest = _finite_number(closest)
    if finite_closest is None:
        return [f"{issue_label} requires a finite no-op closest_robot_ped_distance value."]
    if finite_closest > 2.0:
        return [
            f"{issue_label} requires no-op closest_robot_ped_distance <= 2.0 m; observed {closest}."
        ]
    return []


def _verify_near_field_behavior_probe_guardrails(
    *,
    output_dir: Path,
    fixture_contract: Mapping[str, Any],
    issue_label: str,
) -> list[str]:
    """Fail closed unless an opt-in probe remains a near-field #2756 replay."""
    blockers = _verify_near_field_probe_contract_guardrails(
        fixture_contract=fixture_contract,
        issue_label=issue_label,
    )
    try:
        noop_trace = _load_trace(output_dir / "traces" / "noop" / "trace.json")
        delay_trace = _load_trace(output_dir / "traces" / "delay_only" / "trace.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [*blockers, f"Could not verify {issue_label} guardrails: {exc}"]

    for label, trace in (("noop", noop_trace), ("delay_only", delay_trace)):
        blockers.extend(_verify_near_field_probe_trace_identity(label, trace))

    noop_first_observed = _first_observed_step(noop_trace)
    delay_first_observed = _first_observed_step(delay_trace)
    if noop_first_observed != 5:
        blockers.append(
            f"{issue_label} requires no-op first observed step 5; observed {noop_first_observed}."
        )
    if delay_first_observed != 7:
        blockers.append(
            f"{issue_label} requires delay-only first observed step 7; observed "
            f"{delay_first_observed}."
        )
    blockers.extend(_verify_near_field_probe_trace(noop_trace, issue_label=issue_label))
    return blockers


def _verify_issue_3328_behavior_probe_guardrails(
    *,
    output_dir: Path,
    fixture_contract: Mapping[str, Any],
) -> list[str]:
    """Fail closed unless the opt-in #3328 probe remains a near-field #2756 replay."""
    return _verify_near_field_behavior_probe_guardrails(
        output_dir=output_dir,
        fixture_contract=fixture_contract,
        issue_label="Issue #3328 behavior probe",
    )


def _verify_issue_3330_seed_amplitude_grid_guardrails(
    *,
    output_dir: Path,
    fixture_contract: Mapping[str, Any],
) -> list[str]:
    """Fail closed unless the opt-in #3330 grid remains a near-field #2756 replay."""
    return _verify_near_field_behavior_probe_guardrails(
        output_dir=output_dir,
        fixture_contract=fixture_contract,
        issue_label="Issue #3330 seed/amplitude grid",
    )


def _verify_fixture_observation_boundary(
    *,
    output_dir: Path,
    fixture_contract: Mapping[str, Any],
) -> list[str]:
    """Verify live traces preserve the configured first-visible/delay boundary."""
    if not fixture_contract.get("satisfied"):
        return []
    try:
        noop_trace = _load_trace(output_dir / "traces" / "noop" / "trace.json")
        delay_trace = _load_trace(output_dir / "traces" / "delay_only" / "trace.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"Could not verify #2756 live fixture observation boundary: {exc}"]

    expected_first_visible = int(fixture_contract["first_visible_step"])
    expected_delay_first_observed = int(fixture_contract["delay_only_expected_first_observed_step"])
    noop_first_observed = _first_observed_step(noop_trace)
    delay_first_observed = _first_observed_step(delay_trace)
    blockers: list[str] = []
    if noop_first_observed != expected_first_visible:
        blockers.append(
            "No-op live replay did not preserve the #2756 first-visible boundary: "
            f"observed {noop_first_observed}, expected {expected_first_visible}."
        )
    if delay_first_observed != expected_delay_first_observed:
        blockers.append(
            "Delay-only live replay did not preserve the #2756 delayed first-observed boundary: "
            f"observed {delay_first_observed}, expected {expected_delay_first_observed}."
        )
    return blockers


def _fixture_observation_boundary_summary(output_dir: Path) -> dict[str, int | None]:
    """Return the observed no-op and delay-only fixture timing summary."""
    summary = {
        "noop_first_observed_step": None,
        "delay_only_first_observed_step": None,
        "expected_noop_first_observed_step": 5,
        "expected_delay_only_first_observed_step": 7,
    }
    try:
        noop_trace = _load_trace(output_dir / "traces" / "noop" / "trace.json")
        delay_trace = _load_trace(output_dir / "traces" / "delay_only" / "trace.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return summary
    summary["noop_first_observed_step"] = _first_observed_step(noop_trace)
    summary["delay_only_first_observed_step"] = _first_observed_step(delay_trace)
    return summary


def _compact_live_conditions(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop worktree-local command and raw trace paths from durable report rows."""
    compact_conditions: list[dict[str, Any]] = []
    for condition in conditions:
        compact = dict(condition)
        for key in ("command", "trace", "report"):
            compact.pop(key, None)
        compact_conditions.append(compact)
    return compact_conditions


def _condition_classification_label(condition: Mapping[str, Any]) -> str:
    """Return the compact classification label for a condition row."""
    return str(_mapping(condition.get("classification")).get("label") or "")


def _issue_3330_grid_interpretation(
    *,
    conditions: list[dict[str, Any]],
    blockers: list[str],
) -> dict[str, Any]:
    """Classify the #3330 seed/amplitude grid without promoting benchmark claims."""
    by_name = {str(condition.get("name")): condition for condition in conditions}
    missing = [name for name in ISSUE_3330_REQUIRED_CONDITIONS if name not in by_name]
    unavailable = bool(blockers or missing) or any(
        by_name[name].get("status") != "live_replay"
        for name in ISSUE_3330_REQUIRED_CONDITIONS
        if name in by_name
    )
    if unavailable:
        reasons = list(blockers)
        if missing:
            reasons.append(f"Missing #3330 condition rows: {', '.join(missing)}.")
        return {
            "label": "unavailable_fail_closed",
            "evidence_status": "diagnostic-only",
            "summary": (
                "Grid interpretation is unavailable/fail-closed because at least one "
                "required live replay condition did not complete under the fixture guardrails."
            ),
            "sensitive_conditions": [],
            "medium_sensitive_conditions": [],
            "high_sensitive_conditions": [],
            "unavailable_conditions": [
                name
                for name in ISSUE_3330_REQUIRED_CONDITIONS
                if name not in by_name or by_name[name].get("status") != "live_replay"
            ],
            "limitation": "Diagnostic-only; no robustness claim is available.",
            "reasons": reasons,
        }

    medium_sensitive = [
        name
        for name in ISSUE_3330_MEDIUM_CONDITIONS
        if _condition_classification_label(by_name[name]) == "behavior_sensitive_diagnostic_only"
    ]
    high_sensitive = [
        name
        for name in ISSUE_3330_HIGH_CONDITIONS
        if _condition_classification_label(by_name[name]) == "behavior_sensitive_diagnostic_only"
    ]
    sensitive_conditions = [*medium_sensitive, *high_sensitive]
    if medium_sensitive:
        label = "medium_amplitude_sensitive"
        summary = (
            "Diagnostic-only grid classification: medium-amplitude-sensitive because at "
            "least one std=0.30 m, bound=0.60 m condition changed live behavior."
        )
    elif len(high_sensitive) == len(ISSUE_3330_HIGH_CONDITIONS):
        label = "high_noise_persistent"
        summary = (
            "Diagnostic-only grid classification: high-noise persistent because all "
            "std=1.00 m, bound=2.00 m seed conditions changed live behavior."
        )
    elif high_sensitive:
        label = "mixed_seed_specific"
        summary = (
            "Diagnostic-only grid classification: mixed/seed-specific because behavior "
            "changed for only a subset of high-noise seeds."
        )
    else:
        label = "not_reproduced"
        summary = (
            "Diagnostic-only grid classification: not reproduced because the completed "
            "seed/amplitude grid did not expose behavior-sensitive condition rows."
        )
    return {
        "label": label,
        "evidence_status": "diagnostic-only",
        "summary": summary,
        "sensitive_conditions": sensitive_conditions,
        "medium_sensitive_conditions": medium_sensitive,
        "high_sensitive_conditions": high_sensitive,
        "unavailable_conditions": [],
        "limitation": (
            "One scenario and one fixture seed only; this is diagnostic behavior-probe "
            "evidence, not a robustness, sensor-realism, or planner-superiority claim."
        ),
        "reasons": [],
    }


def _final_classification(
    *,
    blockers: list[str],
    fixture_contract_satisfied: bool,
    conditions: list[dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    """Resolve the top-level status/classification for the issue report."""
    if blockers:
        return "fail_closed", {"label": "blocked", "rationale": blockers[0]}
    if not fixture_contract_satisfied:
        return (
            "diagnostic_only",
            {
                "label": "diagnostic_only",
                "rationale": (
                    "Live replay completed on a proxy scenario, not the #2755/#2756 "
                    "occluded-emergence fixture."
                ),
            },
        )
    labels = {
        _mapping(item.get("classification")).get("label")
        for item in conditions
        if item["name"] != "noop"
    }
    if "behavior_sensitive_diagnostic_only" in labels:
        label = "behavior_sensitive_diagnostic_only"
    elif "diagnostic_only" in labels:
        label = "diagnostic_only"
    elif "policy_insensitive" in labels:
        label = "policy_insensitive"
    elif "scenario_too_weak" in labels:
        label = "scenario_too_weak"
    else:
        label = "diagnostic_only"
    return (
        "live_replay",
        {
            "label": label,
            "rationale": (
                "All selected live replays completed. One scenario fixture seed is diagnostic "
                "only and does not support a robustness, sensor-realism, or "
                "planner-superiority claim."
            ),
        },
    )


def run_live_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Run the seven live diagnostics conditions and return the summary report."""
    output_dir = args.output_dir
    output_dir = _repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_matrix = _repo_path(args.scenario_matrix)
    fixture_contract = _fixture_contract(scenario_matrix)
    if not fixture_contract["satisfied"] and not args.allow_non_occluded_live_fixture:
        return _fail_closed_report(
            output_dir=output_dir,
            args=args,
            fixture_contract=fixture_contract,
            blocker=str(fixture_contract["blocker"]),
        )

    funnel_config = _write_generated_funnel(
        output_dir=output_dir,
        scenario_matrix=args.scenario_matrix,
        stage=args.stage,
        horizon=args.horizon,
    )
    if args.dry_run:
        return _planned_report(
            args=args,
            output_dir=output_dir,
            fixture_contract=fixture_contract,
            funnel_config=funnel_config,
        )

    conditions, blockers = _execute_conditions(
        args=args,
        output_dir=output_dir,
        funnel_config=funnel_config,
    )
    blockers.extend(
        _attach_condition_comparisons(
            output_dir=output_dir,
            conditions=conditions,
            fixture_contract_satisfied=bool(fixture_contract["satisfied"]),
        )
    )
    blockers.extend(
        _verify_fixture_observation_boundary(
            output_dir=output_dir,
            fixture_contract=fixture_contract,
        )
    )
    if args.condition_set == ISSUE_3328_CONDITION_SET:
        blockers.extend(
            _verify_issue_3328_behavior_probe_guardrails(
                output_dir=output_dir,
                fixture_contract=fixture_contract,
            )
        )
    if args.condition_set == ISSUE_3330_CONDITION_SET:
        blockers.extend(
            _verify_issue_3330_seed_amplitude_grid_guardrails(
                output_dir=output_dir,
                fixture_contract=fixture_contract,
            )
        )
    status, classification = _final_classification(
        blockers=blockers,
        fixture_contract_satisfied=bool(fixture_contract["satisfied"]),
        conditions=conditions,
    )

    compact_conditions = _compact_live_conditions(conditions)
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_shape": "compact_summary_without_raw_traces",
        "issue": 2777,
        "status": status,
        "classification": classification,
        "claim_boundary": (
            "Stress-slice live planner/environment replay for one scenario fixture seed "
            "and the selected perturbation condition set. Treat behavior-sensitive "
            "differences as diagnostic-only from this fixture; do not infer robustness, "
            "sensor realism, planner superiority, or scenario-general behavior."
        ),
        "inputs": {
            "scenario_matrix": _durable_ref(scenario_matrix),
            "raw_trace_json": (
                "worktree-local generated traces summarized here; raw trace.json files "
                "are intentionally not committed"
            ),
            "generated_funnel": (
                "worktree-local generated_policy_search_funnel.yaml was intentionally not committed"
            ),
        },
        "fixture_contract": fixture_contract,
        "fixture_observation_boundary": _fixture_observation_boundary_summary(output_dir),
        "run_config": _run_config(args=args, output_dir=output_dir),
        "conditions": compact_conditions,
        "blockers": blockers,
    }
    if args.condition_set == ISSUE_3330_CONDITION_SET:
        report["grid_interpretation"] = _issue_3330_grid_interpretation(
            conditions=compact_conditions,
            blockers=blockers,
        )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario-matrix", type=Path, default=DEFAULT_SCENARIO_MATRIX)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=Path("docs/context/policy_search/candidate_registry.yaml"),
    )
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--condition-set",
        choices=sorted(CONDITION_SETS),
        default=DEFAULT_CONDITION_SET,
        help=(
            "Perturbation condition set to run. The default preserves the seven #2755 "
            "families; issue_3328_behavior_probe and issue_3330_seed_amplitude_grid "
            "are opt-in behavior probes."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--allow-non-occluded-live-fixture",
        action="store_true",
        help="Run a diagnostic proxy live replay even when the #2755 fixture boundary is absent.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    args.output_dir = _repo_path(args.output_dir)
    if tuple(condition.name for condition in CONDITIONS) != REQUIRED_CONDITIONS:
        raise RuntimeError("Issue #2777 condition set drifted from the required seven families")
    if (
        tuple(condition.name for condition in CONDITION_SETS[ISSUE_3328_CONDITION_SET])
        != ISSUE_3328_REQUIRED_CONDITIONS
    ):
        raise RuntimeError("Issue #3328 behavior probe condition set drifted")
    if (
        tuple(condition.name for condition in CONDITION_SETS[ISSUE_3330_CONDITION_SET])
        != ISSUE_3330_REQUIRED_CONDITIONS
    ):
        raise RuntimeError("Issue #3330 seed/amplitude grid condition set drifted")
    report = run_live_batch(args)
    _write_outputs(report, args.output_dir)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": report["status"],
                "classification": report["classification"],
                "summary": _durable_ref(args.output_dir / "summary.json"),
                "readme": _durable_ref(args.output_dir / "README.md"),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] != "fail_closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
