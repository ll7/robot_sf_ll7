"""Aggregate release-row anomaly checks for published benchmark bundles.

This is a read-only companion to the Benchmark Auditor's episode detectors.
It consumes recorded episode summaries, never simulator steps, and emits BA-03
``Signal`` records for every candidate finding. A signal is diagnostic; it is
not an attribution of a collision to a planner or a confirmed auditor finding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.analysis_workbench.audit_contracts import (
    AuditContractError,
    Signal,
    canonical_json,
    record_from_dict,
    record_to_dict,
)
from robot_sf.analysis_workbench.audit_detectors import (
    DetectorRegistry,
    DetectorSpec,
    execution_admission_failure,
)
from robot_sf.analysis_workbench.audit_store import AuditStore, BatchCommitResult, CommitResult
from robot_sf.analysis_workbench.release_row_bundle import load_release_rows
from robot_sf.benchmark.event_ledger import EPISODE_EVENT_LEDGER_SCHEMA_VERSION

SCHEMA_VERSION = "release-row-anomalies.v1"
DETECTOR_VERSION = "1.0.1"
COLLISION_DETECTOR_VERSION = "1.1.0"
ORBIT_DETECTOR_VERSION = "1.1.0"
DEFAULT_CONFIG: dict[str, Any] = {
    "short_collision_max_steps": 20,
    "same_step_max_steps": 20,
    "max_contact_speed_m_s": 10.0,
    "min_orbit_curvature": 1.0,
    "min_orbit_path_length_m": 5.0,
    "max_zero_progress_m": 0.5,
    "max_progress_ratio": 0.1,
    "min_planners_per_cell": 2,
    "min_paired_cells": 5,
    "min_success_rate_gap": 0.2,
    "baseline_planner": "goal",
    # Empty means discover pedestrian-free scenarios from paired release rows.
    "pedestrian_free_scenarios": [],
    "pedestrian_aware_planners": [
        "guarded_ppo",
        "hybrid_rule_v3_fast_progress_static_escape",
        "hybrid_rule_v3_fast_progress_static_escape_continuous",
        "orca",
        "ppo",
        "prediction_planner",
        "predictive_mppi",
        "risk_dwa",
        "sacadrl",
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
        "scenario_adaptive_hybrid_orca_v2_collision_guard",
        "social_force",
        "socnav_sampling",
    ],
    "max_unannotated_findings": 0,
    "require_preflight": True,
}
DETECTOR_IDS = (
    "same_step_all_planners",
    "short_collision",
    "impossible_contact_speed",
    "orbit_zero_progress",
    "pedestrian_free_baseline_regression",
    "universal_failure_unannotated",
    "invalid_run_preflight_mismatch",
)
COLLISION_DETECTOR_ID = "collision_metric_inconsistent"
RELEASE_TERMINAL_STATUSES = frozenset({"success", "collision", "failure"})
COLLISION_METRIC_FIELDS = (
    "ped_collision_count",
    "obstacle_collision_count",
    "agent_collision_count",
    "total_collision_count",
    "collisions",
)
COLLISION_COUNT_TOLERANCE = 0.0
# Floats above this bound cannot represent every adjacent integer. Keep large
# JSON integer counts exact, but reject large float counts in the strict gate.
MAX_EXACT_FLOAT_COLLISION_COUNT = 2**53 - 1
COLLISION_CONFIG_KEYS = frozenset(
    {"collision_metric_contract", "collision_roster_status", "collision_expected_arm_count"}
)
RELEASE_0_0_8_ROSTER_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)


class ReleaseRowError(ValueError):
    """Malformed release rows, configuration, or accounting inputs."""


def _release_0_0_8_roster() -> set[str] | None:
    """Read the #9751 arm identities from the committed campaign template.

    Returns:
        The 14-arm roster, or None when the source is unavailable or invalid.
    """

    try:
        campaign = yaml.safe_load(RELEASE_0_0_8_ROSTER_SOURCE.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(campaign, dict) or not isinstance(campaign.get("planners"), list):
        return None
    planners = campaign["planners"]
    if len(planners) != 14 or any(
        not isinstance(planner, dict)
        or planner.get("enabled", True) is not True
        or not isinstance(planner.get("key"), str)
        or not planner["key"].strip()
        for planner in planners
    ):
        return None
    roster = {planner["key"] for planner in planners}
    return roster if len(roster) == 14 else None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReleaseRowError(f"{name} must be an integer >= {minimum}")
    return value


def _unique_string_ids(value: object, name: str) -> list[str]:
    """Validate one unique list of non-empty scenario or planner identifiers.

    Returns:
        A copy of the validated ID list.
    """

    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ReleaseRowError(f"{name} must be a list of non-empty IDs")
    if len(value) != len(set(value)):
        raise ReleaseRowError(f"{name} contains duplicates")
    return list(value)


def _validate_collision_contract_settings(settings: Mapping[str, Any]) -> None:
    """Validate the explicit legacy or 0.0.8 collision-roster contract."""

    contract = settings.get("collision_metric_contract", "legacy_diagnostic")
    roster_status = settings.get("collision_roster_status", "legacy_diagnostic")
    expected_arm_count = settings.get("collision_expected_arm_count")
    if contract not in {"legacy_diagnostic", "release_0_0_8"}:
        raise ReleaseRowError(
            "collision_metric_contract must be legacy_diagnostic or release_0_0_8"
        )
    if contract == "legacy_diagnostic":
        if roster_status != "legacy_diagnostic" or expected_arm_count is not None:
            raise ReleaseRowError(
                "legacy collision metric contract cannot declare a candidate roster"
            )
    elif roster_status not in {"unfrozen_template", "frozen"}:
        raise ReleaseRowError("0.0.8 collision roster must be unfrozen_template or frozen")
    else:
        _positive_integer(expected_arm_count, "collision_expected_arm_count", minimum=1)


def _release_row_admission(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """Apply shared execution admission while preserving valid terminal outcomes.

    Returns:
        The shared admission failure, if one exists.
    """

    admission_row = dict(row)
    has_terminal_status = "status" in row
    terminal_status = row.get("status")
    valid_terminal_status = (
        isinstance(terminal_status, str)
        and terminal_status.strip().lower() in RELEASE_TERMINAL_STATUSES
    )
    if valid_terminal_status:
        admission_row.pop("status", None)
    admission = execution_admission_failure(admission_row, check_nonfinite=False)
    if has_terminal_status and not valid_terminal_status and admission is None:
        return "unavailable", "unknown_release_terminal_outcome_status"
    return admission


def _configured(config: Mapping[str, Any] | None) -> dict[str, Any]:  # noqa: C901
    if config is None:
        supplied: Mapping[str, Any] = {}
    elif isinstance(config, Mapping):
        supplied = config
    else:
        raise ReleaseRowError("config must be an object")
    unknown = set(supplied) - (set(DEFAULT_CONFIG) | COLLISION_CONFIG_KEYS)
    if unknown:
        raise ReleaseRowError(f"unknown config keys: {', '.join(sorted(unknown))}")
    result = {**DEFAULT_CONFIG, **supplied}
    for key in (
        "short_collision_max_steps",
        "same_step_max_steps",
        "min_planners_per_cell",
        "min_paired_cells",
    ):
        _positive_integer(result[key], key, minimum=1)
    _positive_integer(result["max_unannotated_findings"], "max_unannotated_findings")
    for key in (
        "max_contact_speed_m_s",
        "min_orbit_curvature",
        "min_orbit_path_length_m",
        "max_zero_progress_m",
        "max_progress_ratio",
        "min_success_rate_gap",
    ):
        value = _finite(result[key])
        if value is None or value < 0:
            raise ReleaseRowError(f"{key} must be a finite number >= 0")
        result[key] = value
    if result["max_contact_speed_m_s"] == 0:
        raise ReleaseRowError("max_contact_speed_m_s must be > 0")
    if result["min_success_rate_gap"] > 1:
        raise ReleaseRowError("min_success_rate_gap must be <= 1")
    if result["max_progress_ratio"] > 1:
        raise ReleaseRowError("max_progress_ratio must be <= 1")
    if not isinstance(result["baseline_planner"], str) or not result["baseline_planner"].strip():
        raise ReleaseRowError("baseline_planner must be a nonempty string")
    result["pedestrian_free_scenarios"] = _unique_string_ids(
        result["pedestrian_free_scenarios"], "pedestrian_free_scenarios"
    )
    aware_planners = _unique_string_ids(
        result["pedestrian_aware_planners"], "pedestrian_aware_planners"
    )
    result["pedestrian_aware_planners"] = aware_planners
    if result["baseline_planner"] in aware_planners:
        raise ReleaseRowError("pedestrian_aware_planners must not include baseline_planner")
    if type(result["require_preflight"]) is not bool:
        raise ReleaseRowError("require_preflight must be a boolean")
    _validate_collision_contract_settings(result)
    return result


def _rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ReleaseRowError("release rows must be a nonempty sequence")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ReleaseRowError(f"row {index} must be an object")
        row = dict(raw)
        scenario = row.get("scenario_id")
        planner = row.get("_release_arm", row.get("planner_id", row.get("algo")))
        seed = row.get("seed")
        if not isinstance(scenario, str) or not scenario.strip():
            raise ReleaseRowError(f"row {index} lacks scenario_id")
        if not isinstance(planner, str) or not planner.strip():
            raise ReleaseRowError(f"row {index} lacks planner identity")
        _positive_integer(seed, f"row {index} seed")
        _positive_integer(row.get("steps"), f"row {index} steps")
        key = (planner, scenario, seed)
        if key in seen:
            raise ReleaseRowError(f"duplicate planner/scenario/seed: {key}")
        seen.add(key)
        outcome = row.get("outcome")
        if not isinstance(outcome, Mapping) or any(
            type(outcome.get(name)) is not bool
            for name in ("route_complete", "collision_event", "timeout_event")
        ):
            raise ReleaseRowError(f"row {index} lacks explicit outcome booleans")
        if not isinstance(row.get("metrics"), Mapping):
            raise ReleaseRowError(f"row {index} lacks metrics object")
        # In this published-row contract, top-level `status` is the episode
        # outcome (`success`, `collision`, or `failure`). Execution admission
        # remains bound to the Auditor's explicit execution-status fields and
        # nested provenance surfaces.
        admission = _release_row_admission(row)
        row["_release_execution_status"] = "eligible" if admission is None else admission[0]
        row["_release_execution_reason"] = "" if admission is None else admission[1]
        row["_release_arm"] = planner
        row.setdefault("episode_id", f"{planner}:{scenario}:{seed}")
        if not isinstance(row["episode_id"], str) or not row["episode_id"].strip():
            raise ReleaseRowError(f"row {index} has invalid episode_id")
        normalized.append(row)
    normalized.sort(key=lambda row: (row["scenario_id"], row["seed"], row["_release_arm"]))
    return normalized, sorted({row["_release_arm"] for row in normalized})


def _annotation_entries(annotations: object) -> list[dict[str, Any]]:  # noqa: C901
    if annotations is None:
        return []
    if isinstance(annotations, Mapping):
        if annotations.get("schema_version") not in (None, "release-row-annotations.v1"):
            raise ReleaseRowError("unsupported annotation schema")
        annotations = annotations.get("annotations")
    if not isinstance(annotations, list):
        raise ReleaseRowError("annotations must be an array")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(annotations):
        if not isinstance(entry, Mapping):
            raise ReleaseRowError(f"annotation {index} must be an object")
        root_cause = entry.get("root_cause")
        source_ref = entry.get("source_ref")
        if not isinstance(root_cause, str) or not root_cause.strip():
            raise ReleaseRowError(f"annotation {index} lacks root_cause")
        if not isinstance(source_ref, str) or not source_ref.strip():
            raise ReleaseRowError(f"annotation {index} lacks source_ref")
        selectors = ("finding_id", "scenario_id", "seed", "planner_id", "detector_id")
        if not any(key in entry for key in ("finding_id", "scenario_id")):
            raise ReleaseRowError(f"annotation {index} needs finding_id or scenario_id")
        if "seed" in entry:
            _positive_integer(entry["seed"], f"annotation {index} seed")
        for key in selectors:
            if (
                key != "seed"
                and key in entry
                and (not isinstance(entry[key], str) or not entry[key].strip())
            ):
                raise ReleaseRowError(f"annotation {index} has invalid {key}")
        manifest_digest = entry.get("manifest_sha256")
        if manifest_digest is not None and (
            not isinstance(manifest_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest)
        ):
            raise ReleaseRowError(f"annotation {index} has invalid manifest_sha256")
        if "finding_id" not in entry and manifest_digest is None:
            raise ReleaseRowError(
                f"annotation {index} needs manifest_sha256 for scenario-scoped matching"
            )
        entries.append(dict(entry))
    return entries


def _matching_annotation(
    finding: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for entry in entries:
        if "finding_id" in entry:
            if entry["finding_id"] == finding["finding_id"]:
                return entry
            continue
        if entry.get("manifest_sha256") != source.get("manifest_sha256"):
            continue
        if all(
            key not in entry or entry[key] == finding.get(key)
            for key in ("scenario_id", "seed", "planner_id", "detector_id")
        ):
            return entry
    return None


def _preflight_cells(preflight: object) -> dict[tuple[str, int], bool] | None:
    if preflight is None:
        return None
    if isinstance(preflight, Mapping):
        if preflight.get("schema_version") not in (None, "release-row-preflight.v1"):
            raise ReleaseRowError("unsupported preflight schema")
        preflight = preflight.get("cells")
    if not isinstance(preflight, list):
        raise ReleaseRowError("preflight cells must be an array")
    cells: dict[tuple[str, int], bool] = {}
    for index, cell in enumerate(preflight):
        if not isinstance(cell, Mapping):
            raise ReleaseRowError(f"preflight cell {index} must be an object")
        scenario = cell.get("scenario_id")
        seed = cell.get("seed")
        invalid = cell.get("invalid_run")
        if not isinstance(scenario, str) or not scenario.strip():
            raise ReleaseRowError(f"preflight cell {index} lacks scenario_id")
        _positive_integer(seed, f"preflight cell {index} seed")
        if type(invalid) is not bool:
            raise ReleaseRowError(f"preflight cell {index} lacks invalid_run boolean")
        key = (scenario, seed)
        if key in cells:
            raise ReleaseRowError(f"duplicate preflight cell: {key}")
        cells[key] = invalid
    return cells


def _invalid_run(row: Mapping[str, Any]) -> bool | None:
    ledger = row.get("event_ledger")
    if not isinstance(ledger, Mapping):
        return None
    exact = ledger.get("exact_events")
    if not isinstance(exact, Mapping):
        return None
    value = exact.get("invalid_run")
    return value if type(value) is bool else None


def _collision_integer_count(value: object) -> tuple[int | None, str | None]:
    """Validate a sampled count without rounding an integer through float.

    Returns:
        The exact nonnegative integer count or a named domain problem.
    """

    if type(value) is int:
        return (value, None) if value >= 0 else (None, "invalid_count_domain")
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, "missing_or_nonfinite"
        if value < 0 or not value.is_integer() or value > MAX_EXACT_FLOAT_COLLISION_COUNT:
            return None, "invalid_count_domain"
        return int(value), None
    return None, "missing_or_nonfinite"


def _typed_ledger_collision_problems(row: Mapping[str, Any], total: int | None) -> list[str]:
    """Check equivalent typed-ledger fields without counting exact events.

    Returns:
        Named reconciliation and schema problems, if a typed ledger is present.
    """

    ledger = row.get("event_ledger")
    if not isinstance(ledger, Mapping) or "schema_version" not in ledger:
        return []
    if ledger["schema_version"] != EPISODE_EVENT_LEDGER_SCHEMA_VERSION:
        return ["unsupported_event_ledger_schema"]
    problems: list[str] = []
    reconciliation = ledger.get("reconciliation")
    exact = ledger.get("exact_events")
    if not isinstance(reconciliation, Mapping):
        problems.append("missing_collision_reconciliation")
    else:
        ledger_value, _ = _collision_integer_count(reconciliation.get("collision_metric_value"))
        if ledger_value is None:
            problems.append("missing_ledger_collision_metric_value")
        elif total is not None and ledger_value != total:
            problems.append("ledger_collision_metric_mismatch")
        if reconciliation.get("collision_metric_source") != "metrics.total_collision_count":
            problems.append("ledger_collision_metric_source_mismatch")
    if not isinstance(exact, Mapping) or type(exact.get("collision")) is not bool:
        problems.append("missing_exact_collision_event")
    elif exact["collision"] is not row["outcome"]["collision_event"]:
        problems.append("exact_collision_event_mismatch")
    return problems


def _collision_metric_problems(row: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Check sampled collision arithmetic and typed-ledger identity.

    Returns:
        Named contract problems and measured values for an episode row.
    """

    metrics = row["metrics"]
    values: dict[str, int | None] = {}
    problems: list[str] = []
    for field in COLLISION_METRIC_FIELDS:
        values[field], problem = _collision_integer_count(metrics.get(field))
        if problem is not None:
            problems.append(f"{problem}_{field}")
    total = values["total_collision_count"]
    alias = values["collisions"]
    components = [values[field] for field in COLLISION_METRIC_FIELDS[:3]]
    component_sum = sum(components) if all(value is not None for value in components) else None
    if total is not None and alias is not None and total != alias:
        problems.append("collision_alias_mismatch")
    if total is not None and component_sum is not None and total != component_sum:
        problems.append("collision_component_sum_mismatch")

    problems.extend(_typed_ledger_collision_problems(row, total))
    ledger = row.get("event_ledger")
    return problems, {
        "metrics": values,
        "component_sum": component_sum,
        "event_ledger_schema": ledger.get("schema_version")
        if isinstance(ledger, Mapping)
        else None,
    }


def _displacement(row: Mapping[str, Any]) -> tuple[float | None, str | None]:
    metrics = row["metrics"]
    for key in ("robot_displacement_m", "net_displacement_m", "displacement_m"):
        value = _finite(metrics.get(key))
        if value is not None and value >= 0:
            return value, f"metrics.{key}"
    return None, None


def _progress_ratio(row: Mapping[str, Any]) -> float | None:
    predicates = row.get("safety_predicates")
    oscillation = (
        predicates.get("oscillatory_control_predicate") if isinstance(predicates, Mapping) else None
    )
    fields = oscillation.get("fields") if isinstance(oscillation, Mapping) else None
    value = _finite(fields.get("progress_ratio")) if isinstance(fields, Mapping) else None
    return value if value is not None and 0 <= value <= 1 else None


def _pedestrian_free(row: Mapping[str, Any]) -> bool | None:
    integrity = row.get("integrity")
    effective = integrity.get("effective_view") if isinstance(integrity, Mapping) else None
    count = effective.get("observation_ped_count") if isinstance(effective, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count == 0


def _contact_speeds(row: Mapping[str, Any]) -> tuple[list[float] | None, str | None]:
    ledger = row.get("event_ledger")
    events = ledger.get("collision_events") if isinstance(ledger, Mapping) else None
    if events is not None:
        if not isinstance(events, list) or any(not isinstance(item, Mapping) for item in events):
            raise ReleaseRowError("collision_events must be an array of objects")
        values = [_finite(item.get("relative_speed_at_contact")) for item in events]
        if any(value is not None and value < 0 for value in values):
            raise ReleaseRowError("relative_speed_at_contact must be >= 0")
        available = [value for value in values if value is not None]
        if available:
            return available, "event_ledger.collision_events[].relative_speed_at_contact"
    value = _finite(row["metrics"].get("max_relative_contact_speed_m_s"))
    if value is not None:
        if value < 0:
            raise ReleaseRowError("max_relative_contact_speed_m_s must be >= 0")
        return [value], "metrics.max_relative_contact_speed_m_s"
    return None, None


def _reported_failure_mode(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize shared row-level outcome and collision metadata without inferring cause.

    Returns:
        A planner-indexed summary and a consistency classification.
    """

    by_planner: dict[str, dict[str, Any]] = {}
    partially_observed = False
    for row in rows:
        planner = row["_release_arm"]
        outcome = row["outcome"]
        ledger = row.get("event_ledger")
        events = ledger.get("collision_events") if isinstance(ledger, Mapping) else None
        collision = outcome["collision_event"]
        valid_events = isinstance(events, list) and all(
            isinstance(event, Mapping) for event in events
        )
        partner_types = (
            sorted(
                {
                    value
                    for event in events
                    if isinstance((value := event.get("collision_partner_type")), str) and value
                }
            )
            if valid_events
            else None
        )
        event_sources = (
            sorted(
                {
                    value
                    for event in events
                    if isinstance((value := event.get("exact_event_source")), str) and value
                }
            )
            if valid_events
            else None
        )
        invalid_run = _invalid_run(row)
        missing_event_detail = collision and (
            not valid_events
            or not events
            or any(
                not isinstance(event.get("collision_partner_type"), str)
                or not event.get("collision_partner_type")
                or not isinstance(event.get("exact_event_source"), str)
                or not event.get("exact_event_source")
                for event in events
            )
        )
        partially_observed |= invalid_run is None or missing_event_detail
        signature = {
            "outcome": {
                "route_complete": outcome["route_complete"],
                "collision_event": collision,
                "timeout_event": outcome["timeout_event"],
            },
            "reported_status": row.get("status") if isinstance(row.get("status"), str) else None,
            "invalid_run": invalid_run,
            "collision_event_count": len(events) if valid_events else None,
            "collision_partner_types": partner_types,
            "exact_event_sources": event_sources,
        }
        by_planner[planner] = signature

    encoded = {
        planner: json.dumps(signature, sort_keys=True, separators=(",", ":"))
        for planner, signature in by_planner.items()
    }
    consistent = len(set(encoded.values())) == 1
    consistency = (
        "mixed" if not consistent else "partially_observed" if partially_observed else "consistent"
    )
    return {
        "consistency": consistency,
        "common_signature": next(iter(by_planner.values())) if consistent else None,
        "by_planner": dict(sorted(by_planner.items())),
        "root_cause_attribution": "unavailable_from_release_rows",
    }


def _finding_id(detector_id: str, scope: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    identity = {
        "detector_id": detector_id,
        "scope": scope,
        "bundle_sha256": source.get("bundle_sha256"),
        "manifest_sha256": source.get("manifest_sha256"),
        "detector_registry_digest": source.get("detector_registry_digest"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"rr-{digest[:20]}"


def _new_finding(  # noqa: PLR0913
    detector_id: str,
    *,
    scenario_id: str,
    seed: int | None = None,
    planner_id: str | None = None,
    rows: Sequence[Mapping[str, Any]] = (),
    measured: Mapping[str, Any] | None = None,
    threshold: Mapping[str, Any] | None = None,
    reason: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    scope = {"scenario_id": scenario_id, "seed": seed, "planner_id": planner_id}
    return {
        "finding_id": _finding_id(detector_id, scope, source),
        "detector_id": detector_id,
        "detector_version": (
            COLLISION_DETECTOR_VERSION
            if detector_id == COLLISION_DETECTOR_ID
            else ORBIT_DETECTOR_VERSION
            if detector_id == "orbit_zero_progress"
            else DETECTOR_VERSION
        ),
        "reason_code": reason,
        "scenario_id": scenario_id,
        "seed": seed,
        "planner_id": planner_id,
        "episode_ids": sorted(str(row["episode_id"]) for row in rows),
        "source_members": sorted(
            {str(row.get("_source_member")) for row in rows if row.get("_source_member")}
        ),
        "measured": dict(measured or {}),
        "threshold": dict(threshold or {}),
    }


def _signal(finding: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    signal = Signal(
        signal_id=finding["finding_id"],
        detector_id=finding["detector_id"],
        detector_version=finding["detector_version"],
        status="flagged",
        reason_code=finding["reason_code"],
        episode_id=finding["episode_ids"][0] if len(finding["episode_ids"]) == 1 else "",
        evidence=(
            {
                "kind": "release_row_scope",
                "scenario_id": finding["scenario_id"],
                "seed": finding["seed"],
                "planner_id": finding["planner_id"],
                "episode_ids": finding["episode_ids"],
            },
            {
                "kind": "source_identity",
                "bundle_sha256": source.get("bundle_sha256"),
                "manifest_sha256": source.get("manifest_sha256"),
                "source_members": finding["source_members"],
            },
        ),
        measured=finding["measured"],
        threshold=finding["threshold"],
    )
    return record_to_dict(signal)


def release_row_registry(config: Mapping[str, Any] | None = None) -> DetectorRegistry:
    """Declare release-row methods using the Benchmark Auditor registry contract.

    Returns:
        A versioned registry with disclosed thresholds and row-only provenance.
    """

    settings = _configured(config)
    descriptions = {
        "same_step_all_planners": "All complete planner arms fail one cell at the same early step.",
        "short_collision": "A collision terminates within the configured step bound.",
        "impossible_contact_speed": "Recorded relative contact speed exceeds a physical limit.",
        "orbit_zero_progress": "Recorded curvature, displacement, progress, or deadlock_stall windows indicate a stall.",
        "pedestrian_free_baseline_regression": "Paired pedestrian-free success is worse than blind goal.",
        "universal_failure_unannotated": "Every planner fails a cell without root-cause annotation.",
        "invalid_run_preflight_mismatch": "Episode invalid_run differs from scenario-seed preflight.",
        "collision_metric_inconsistent": (
            "Canonical collision counts, alias, or typed-ledger reconciliation disagree."
        ),
    }
    parameters = {
        "same_step_all_planners": ("same_step_max_steps", "min_planners_per_cell"),
        "short_collision": ("short_collision_max_steps",),
        "impossible_contact_speed": ("max_contact_speed_m_s",),
        "orbit_zero_progress": (
            "min_orbit_curvature",
            "min_orbit_path_length_m",
            "max_zero_progress_m",
            "max_progress_ratio",
        ),
        "pedestrian_free_baseline_regression": (
            "baseline_planner",
            "pedestrian_free_scenarios",
            "pedestrian_aware_planners",
            "min_paired_cells",
            "min_success_rate_gap",
        ),
        "universal_failure_unannotated": ("min_planners_per_cell",),
        "invalid_run_preflight_mismatch": (),
        "collision_metric_inconsistent": (),
    }
    detector_ids = (
        (*DETECTOR_IDS, COLLISION_DETECTOR_ID)
        if settings.get("collision_metric_contract", "legacy_diagnostic") == "release_0_0_8"
        else DETECTOR_IDS
    )
    specs = tuple(
        DetectorSpec(
            detector_id=detector_id,
            family="release_row_anomaly",
            description=descriptions[detector_id],
            version=(
                COLLISION_DETECTOR_VERSION
                if detector_id == COLLISION_DETECTOR_ID
                else ORBIT_DETECTOR_VERSION
                if detector_id == "orbit_zero_progress"
                else DETECTOR_VERSION
            ),
            required_capabilities=("episode",),
            cohort_definition={"kind": "release_cell", "key": ["scenario_id", "seed"]},
            parameters={key: settings[key] for key in parameters[detector_id]},
            units={"steps": "count", "contact_speed": "m/s", "displacement": "m"},
            provenance={
                "owner": "robot_sf.analysis_workbench.release_row_anomalies",
                "source": "published_episode_rows",
                "evidence_boundary": "diagnostic_only",
            },
        )
        for detector_id in sorted(detector_ids)
    )
    return DetectorRegistry(version="release-row-detector-registry.v1", detectors=specs)


def analyze_release_rows(  # noqa: C901, PLR0912, PLR0915
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
    annotations: object = None,
    preflight: object = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate recorded release rows.

    Returns:
        A deterministic report with diagnostic findings and BA-03 signals.
    """

    settings = _configured(config)
    records, observed_planners = _rows(rows)
    # Source manifests declare the denominator; legacy defaults describe only
    # the historical pedestrian comparison cohort.
    if (
        source
        and "planner_ids" in source
        and (config is None or "pedestrian_aware_planners" not in config)
    ):
        roster = _unique_string_ids(source["planner_ids"], "source.planner_ids")
        settings["pedestrian_aware_planners"] = [
            planner for planner in roster if planner != settings["baseline_planner"]
        ]
    registry = release_row_registry(settings)
    try:
        source_info = json.loads(canonical_json(dict(source or {})))
    except (AuditContractError, TypeError, ValueError, RecursionError) as error:
        raise ReleaseRowError("source identity must be strict JSON") from error
    source_info["detector_registry_digest"] = registry.digest
    expected_planners = source_info.get("planner_ids", observed_planners)
    if (
        not isinstance(expected_planners, list)
        or any(not isinstance(item, str) or not item for item in expected_planners)
        or len(expected_planners) != len(set(expected_planners))
    ):
        raise ReleaseRowError("source.planner_ids must be a unique list of planner IDs")
    expected_planners = sorted(expected_planners)
    if set(observed_planners) - set(expected_planners):
        raise ReleaseRowError("rows include planner IDs outside source.planner_ids")
    source_info["planner_ids"] = expected_planners
    configured_roster = set(expected_planners)
    if settings["pedestrian_aware_planners"]:
        configured_roster.add(settings["baseline_planner"])
        configured_roster.update(settings["pedestrian_aware_planners"])
    coverage_planners = set(configured_roster)
    collision_roster_reasons: list[str] = []
    if settings.get("collision_metric_contract", "legacy_diagnostic") == "release_0_0_8":
        if settings.get("collision_roster_status", "legacy_diagnostic") != "frozen":
            collision_roster_reasons.append("collision_roster_unfrozen_template")
        if len(expected_planners) != settings.get("collision_expected_arm_count") or set(
            expected_planners
        ) != {settings["baseline_planner"], *settings["pedestrian_aware_planners"]}:
            collision_roster_reasons.append("collision_roster_manifest_mismatch")
        frozen_roster = _release_0_0_8_roster()
        if frozen_roster is None:
            collision_roster_reasons.append("collision_roster_source_unavailable")
        elif (
            settings.get("collision_expected_arm_count") != len(frozen_roster)
            or set(expected_planners) != frozen_roster
            or {settings["baseline_planner"], *settings["pedestrian_aware_planners"]}
            != frozen_roster
        ):
            collision_roster_reasons.append("collision_roster_source_mismatch")
    source_info.setdefault("row_count", len(records))
    annotation_entries = _annotation_entries(annotations)
    preflight_map = _preflight_cells(preflight)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_scenario_planner: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    admission_counts: Counter[str] = Counter()
    admission_reasons: Counter[str] = Counter()
    for row in records:
        grouped[(row["scenario_id"], row["seed"])].append(row)
        status = row["_release_execution_status"]
        admission_counts[status] += 1
        if status == "eligible":
            by_scenario_planner[(row["scenario_id"], row["_release_arm"])][row["seed"]] = row
        else:
            admission_reasons[row["_release_execution_reason"]] += 1

    findings: list[dict[str, Any]] = []
    missingness: Counter[str] = Counter()
    if preflight_map is None:
        missingness["preflight_unavailable"] = 1
    coverage = {"complete_cells": 0, "incomplete_cells": 0, "incomplete_cell_ids": []}
    annotated_universal_cells = 0

    def append(finding: dict[str, Any]) -> None:
        entry = _matching_annotation(finding, annotation_entries, source_info)
        finding["annotated"] = entry is not None
        finding["annotation"] = (
            {"root_cause": entry["root_cause"], "source_ref": entry["source_ref"]}
            if entry
            else None
        )
        findings.append(finding)

    for (scenario, seed), all_cell_rows in sorted(grouped.items()):
        cell = [row for row in all_cell_rows if row["_release_execution_status"] == "eligible"]
        planners = {row["_release_arm"] for row in cell}
        complete = planners == coverage_planners
        coverage["complete_cells" if complete else "incomplete_cells"] += 1
        if not complete:
            coverage["incomplete_cell_ids"].append(
                {
                    "scenario_id": scenario,
                    "seed": seed,
                    "missing_planners": sorted(coverage_planners - planners),
                }
            )
        enough = len(planners) >= settings["min_planners_per_cell"]
        failures = [not row["outcome"]["route_complete"] for row in cell]
        all_failed = complete and enough and all(failures)
        steps = {row["steps"] for row in cell}
        if all_failed and len(steps) == 1 and next(iter(steps)) <= settings["same_step_max_steps"]:
            append(
                _new_finding(
                    "same_step_all_planners",
                    scenario_id=scenario,
                    seed=seed,
                    rows=cell,
                    measured={
                        "planner_count": len(planners),
                        "failure_step": next(iter(steps)),
                        "reported_failure_mode": _reported_failure_mode(cell),
                    },
                    threshold={"max_step": settings["same_step_max_steps"]},
                    reason="common_early_failure_step",
                    source=source_info,
                )
            )
        if all_failed:
            candidate = _new_finding(
                "universal_failure_unannotated",
                scenario_id=scenario,
                seed=seed,
                rows=cell,
                measured={"planner_count": len(planners), "failed_planners": sorted(planners)},
                threshold={"minimum_planners": settings["min_planners_per_cell"]},
                reason="all_planners_failed_without_root_cause",
                source=source_info,
            )
            if _matching_annotation(candidate, annotation_entries, source_info):
                annotated_universal_cells += 1
            else:
                append(candidate)
        if cell and preflight_map is not None:
            expected_invalid = preflight_map.get((scenario, seed))
            if expected_invalid is None:
                missingness["preflight_cell_missing"] += 1
            else:
                mismatched = [row for row in cell if _invalid_run(row) != expected_invalid]
                if mismatched:
                    append(
                        _new_finding(
                            "invalid_run_preflight_mismatch",
                            scenario_id=scenario,
                            seed=seed,
                            rows=mismatched,
                            measured={
                                "preflight_invalid_run": expected_invalid,
                                "row_invalid_run": {
                                    row["_release_arm"]: _invalid_run(row) for row in mismatched
                                },
                            },
                            threshold={"mismatches_allowed": 0},
                            reason="invalid_run_does_not_match_preflight",
                            source=source_info,
                        )
                    )
        for row in cell:
            planner = row["_release_arm"]
            if settings.get(
                "collision_metric_contract", "legacy_diagnostic"
            ) == "release_0_0_8" and not isinstance(row.get("event_ledger"), Mapping):
                missingness["typed_collision_ledger_unavailable"] += 1
            elif settings.get(
                "collision_metric_contract", "legacy_diagnostic"
            ) == "release_0_0_8" and not row["event_ledger"].get("schema_version"):
                missingness["typed_collision_ledger_unavailable"] += 1
            collision_problems, collision_measured = (
                _collision_metric_problems(row)
                if settings.get("collision_metric_contract", "legacy_diagnostic") == "release_0_0_8"
                else ([], {})
            )
            if collision_problems:
                append(
                    _new_finding(
                        "collision_metric_inconsistent",
                        scenario_id=scenario,
                        seed=seed,
                        planner_id=planner,
                        rows=[row],
                        measured={**collision_measured, "problems": collision_problems},
                        threshold={
                            "required_fields": list(COLLISION_METRIC_FIELDS),
                            "absolute_tolerance": COLLISION_COUNT_TOLERANCE,
                        },
                        reason="collision_metric_contract_violated",
                        source=source_info,
                    )
                )
            if (
                row["outcome"]["collision_event"]
                and row["steps"] <= settings["short_collision_max_steps"]
            ):
                append(
                    _new_finding(
                        "short_collision",
                        scenario_id=scenario,
                        seed=seed,
                        planner_id=planner,
                        rows=[row],
                        measured={"steps": row["steps"]},
                        threshold={"max_steps": settings["short_collision_max_steps"]},
                        reason="early_collision",
                        source=source_info,
                    )
                )
            if row["outcome"]["collision_event"]:
                speeds, speed_source = _contact_speeds(row)
                if speeds is None:
                    missingness["contact_speed_unavailable"] += 1
                elif max(speeds) > settings["max_contact_speed_m_s"]:
                    append(
                        _new_finding(
                            "impossible_contact_speed",
                            scenario_id=scenario,
                            seed=seed,
                            planner_id=planner,
                            rows=[row],
                            measured={
                                "max_relative_speed_at_contact_m_s": max(speeds),
                                "speed_source": speed_source,
                            },
                            threshold={"max_contact_speed_m_s": settings["max_contact_speed_m_s"]},
                            reason="contact_speed_exceeds_physical_limit",
                            source=source_info,
                        )
                    )
            metrics = row["metrics"]
            curvature = _finite(metrics.get("curvature_mean"))
            path_length = _finite(metrics.get("socnavbench_path_length"))
            displacement, displacement_source = _displacement(row)
            progress_ratio = _progress_ratio(row)
            stall = metrics.get("deadlock_stall")
            stall_count = None
            if isinstance(stall, Mapping) and stall.get("status") == "ok":
                stall_count = _finite(stall.get("stall_window_count"))
                if stall_count is None or stall_count < 0 or not stall_count.is_integer():
                    raise ReleaseRowError(
                        "deadlock_stall.stall_window_count must be a nonnegative integer"
                    )
            if curvature is None:
                missingness["curvature_unavailable"] += 1
            if path_length is None:
                missingness["path_length_unavailable"] += 1
            if displacement is None:
                missingness["displacement_unavailable"] += 1
            if stall_count is None:
                missingness["deadlock_stall_unavailable"] += 1
            if progress_ratio is None:
                missingness["progress_ratio_unavailable"] += 1
            orbit = (
                curvature is not None
                and path_length is not None
                and curvature >= settings["min_orbit_curvature"]
                and path_length >= settings["min_orbit_path_length_m"]
            )
            zero_progress = (
                displacement is not None
                and path_length is not None
                and displacement <= settings["max_zero_progress_m"]
                and path_length >= settings["min_orbit_path_length_m"]
            )
            low_progress_ratio = (
                progress_ratio is not None
                and path_length is not None
                and progress_ratio <= settings["max_progress_ratio"]
                and path_length >= settings["min_orbit_path_length_m"]
            )
            deadlock_stall = (
                stall_count is not None and stall_count > 0 and not row["outcome"]["route_complete"]
            )
            if orbit or zero_progress or low_progress_ratio or deadlock_stall:
                append(
                    _new_finding(
                        "orbit_zero_progress",
                        scenario_id=scenario,
                        seed=seed,
                        planner_id=planner,
                        rows=[row],
                        measured={
                            "curvature_mean": curvature,
                            "path_length_m": path_length,
                            "displacement_m": displacement,
                            "displacement_source": displacement_source,
                            "progress_ratio": progress_ratio,
                            "deadlock_stall_window_count": stall_count,
                            "signatures": [
                                name
                                for name, active in (
                                    ("high_curvature", orbit),
                                    ("low_displacement", zero_progress),
                                    ("low_progress_ratio", low_progress_ratio),
                                    ("deadlock_stall", deadlock_stall),
                                )
                                if active
                            ],
                        },
                        threshold={
                            "min_orbit_curvature": settings["min_orbit_curvature"],
                            "min_orbit_path_length_m": settings["min_orbit_path_length_m"],
                            "max_zero_progress_m": settings["max_zero_progress_m"],
                            "max_progress_ratio": settings["max_progress_ratio"],
                        },
                        reason="orbit_or_zero_progress_signature",
                        source=source_info,
                    )
                )

    automatic_pedestrian_free_discovery = bool(
        settings["pedestrian_aware_planners"] and not settings["pedestrian_free_scenarios"]
    )
    pedestrian_free_scenarios = (
        settings["pedestrian_free_scenarios"]
        or sorted(
            {
                scenario
                for (scenario, planner), seed_rows in by_scenario_planner.items()
                if planner
                in {
                    settings["baseline_planner"],
                    *settings["pedestrian_aware_planners"],
                }
                and any(_pedestrian_free(row) is True for row in seed_rows.values())
            }
        )
        if settings["pedestrian_aware_planners"]
        else []
    )
    missing_configured_aware_planners = sorted(
        set(settings["pedestrian_aware_planners"]) - set(expected_planners)
    )
    if missing_configured_aware_planners:
        missingness["pedestrian_aware_planner_missing"] += len(missing_configured_aware_planners)
    if automatic_pedestrian_free_discovery:
        discovery_planners = {
            settings["baseline_planner"],
            *settings["pedestrian_aware_planners"],
        }
        unavailable_discovery_rows = sum(
            _pedestrian_free(row) is None
            for row in records
            if row["_release_arm"] in discovery_planners
        )
        if unavailable_discovery_rows:
            missingness["pedestrian_free_status_unavailable"] += unavailable_discovery_rows
    if settings["pedestrian_aware_planners"] and (
        settings["baseline_planner"] not in expected_planners
        or not any(planner == settings["baseline_planner"] for _, planner in by_scenario_planner)
    ):
        missingness["pedestrian_free_baseline_planner_unavailable"] += 1
    for scenario in pedestrian_free_scenarios:
        baseline_rows = by_scenario_planner.get((scenario, settings["baseline_planner"]), {})
        if not baseline_rows:
            missingness["pedestrian_free_baseline_missing"] += 1
            continue
        for planner in settings["pedestrian_aware_planners"]:
            if planner not in expected_planners:
                continue
            candidate_rows = by_scenario_planner.get((scenario, planner), {})
            shared_seeds = set(baseline_rows) & set(candidate_rows)
            paired_seeds = sorted(
                seed
                for seed in shared_seeds
                if _pedestrian_free(baseline_rows[seed]) is True
                and _pedestrian_free(candidate_rows[seed]) is True
            )
            unavailable_pairs = 0
            mismatched_pairs = 0
            for seed in shared_seeds:
                baseline_free = _pedestrian_free(baseline_rows[seed])
                candidate_free = _pedestrian_free(candidate_rows[seed])
                if baseline_free is None or candidate_free is None:
                    unavailable_pairs += 1
                elif baseline_free != candidate_free:
                    mismatched_pairs += 1
            if unavailable_pairs and not automatic_pedestrian_free_discovery:
                missingness["pedestrian_free_status_unavailable"] += unavailable_pairs
            if mismatched_pairs:
                missingness["pedestrian_free_status_mismatch"] += mismatched_pairs
            if len(paired_seeds) < settings["min_paired_cells"]:
                missingness["pedestrian_free_pair_too_small"] += 1
                continue
            baseline_success = sum(
                baseline_rows[seed]["outcome"]["route_complete"] for seed in paired_seeds
            )
            candidate_success = sum(
                candidate_rows[seed]["outcome"]["route_complete"] for seed in paired_seeds
            )
            gap = (baseline_success - candidate_success) / len(paired_seeds)
            if gap >= settings["min_success_rate_gap"]:
                regressed_seeds = [
                    seed
                    for seed in paired_seeds
                    if baseline_rows[seed]["outcome"]["route_complete"]
                    and not candidate_rows[seed]["outcome"]["route_complete"]
                ]
                append(
                    _new_finding(
                        "pedestrian_free_baseline_regression",
                        scenario_id=scenario,
                        planner_id=planner,
                        rows=[candidate_rows[seed] for seed in regressed_seeds],
                        measured={
                            "paired_cells": len(paired_seeds),
                            "baseline_successes": baseline_success,
                            "planner_successes": candidate_success,
                            "success_rate_gap": gap,
                            "regressed_seeds": regressed_seeds,
                        },
                        threshold={
                            "min_success_rate_gap": settings["min_success_rate_gap"],
                            "min_paired_cells": settings["min_paired_cells"],
                        },
                        reason="worse_than_blind_baseline_without_pedestrians",
                        source=source_info,
                    )
                )

    findings.sort(
        key=lambda finding: (
            finding["scenario_id"],
            finding["seed"] if finding["seed"] is not None else -1,
            finding["planner_id"] or "",
            finding["detector_id"],
        )
    )
    seen_cells = set(grouped)
    extra_preflight = sorted(set(preflight_map) - seen_cells) if preflight_map is not None else []
    if extra_preflight:
        missingness["preflight_cell_extra"] = len(extra_preflight)
    preflight_status = (
        "unavailable"
        if preflight_map is None
        else (
            "incomplete" if missingness["preflight_cell_missing"] or extra_preflight else "complete"
        )
    )
    unannotated = sum(not item["annotated"] for item in findings)
    reasons = []
    if unannotated > settings["max_unannotated_findings"]:
        reasons.append("unannotated_findings_above_threshold")
    if coverage["incomplete_cells"]:
        reasons.append("incomplete_planner_cells")
    if settings["require_preflight"] and preflight_status != "complete":
        reasons.append("preflight_accounting_unavailable_or_incomplete")
    counts = dict(sorted(Counter(item["detector_id"] for item in findings).items()))
    if counts.get("invalid_run_preflight_mismatch", 0):
        reasons.append("invalid_run_preflight_mismatch")
    if counts.get("collision_metric_inconsistent", 0):
        reasons.append("collision_metric_inconsistent")
    reasons.extend(collision_roster_reasons)
    if admission_counts.get("unavailable", 0) or admission_counts.get("error", 0):
        reasons.append("execution_admission_incomplete")
    if missingness.get("pedestrian_aware_planner_missing", 0):
        reasons.append("pedestrian_aware_planner_missing")
    if missingness.get("pedestrian_free_baseline_planner_unavailable", 0):
        reasons.append("pedestrian_baseline_planner_unavailable")
    incomplete_pedestrian_cohort_fields = (
        "pedestrian_free_baseline_missing",
        "pedestrian_free_baseline_planner_unavailable",
        "pedestrian_free_status_unavailable",
        "pedestrian_free_status_mismatch",
        "pedestrian_free_pair_too_small",
    )
    if any(missingness.get(key, 0) for key in incomplete_pedestrian_cohort_fields):
        reasons.append("pedestrian_comparison_cohort_incomplete")
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": "Diagnostic release-row signals; no per-step reconstruction or causal attribution.",
        "source": source_info,
        "config": settings,
        "detector_registry": registry.to_dict(),
        "detector_registry_digest": registry.digest,
        "coverage": coverage,
        "execution_admission": {
            "eligible_rows": admission_counts.get("eligible", 0),
            "unavailable_rows": admission_counts.get("unavailable", 0),
            "error_rows": admission_counts.get("error", 0),
            "by_reason": dict(sorted(admission_reasons.items())),
        },
        "preflight_accounting": {
            "status": preflight_status,
            "expected_cells": len(preflight_map) if preflight_map is not None else None,
            "extra_cells": [
                {"scenario_id": scenario, "seed": seed} for scenario, seed in extra_preflight
            ],
        },
        "counts": {
            "rows": len(records),
            "cells": len(grouped),
            "planners": len(expected_planners),
            "admissible_rows": admission_counts.get("eligible", 0),
            "findings": len(findings),
            "by_detector": counts,
            "annotated_universal_cells": annotated_universal_cells,
        },
        "missingness": dict(sorted(missingness.items())),
        "findings": findings,
        "signals": [_signal(item, source_info) for item in findings],
        "gate": {
            "blocked": bool(reasons),
            "reasons": reasons,
            "unannotated_findings": unannotated,
            "max_unannotated_findings": settings["max_unannotated_findings"],
        },
    }


def _validated_release_registry(report: Mapping[str, Any]) -> tuple[str, frozenset[str]]:
    """Validate the aggregate detector registry.

    Returns:
        The verified detector-registry digest and IDs allowed by its config.
    """
    registry_payload = report.get("detector_registry")
    registry_digest = report.get("detector_registry_digest")
    if not isinstance(registry_payload, Mapping) or not isinstance(registry_digest, str):
        raise ReleaseRowError("release-row report lacks detector registry identity")
    try:
        observed_registry_digest = hashlib.sha256(
            canonical_json(registry_payload).encode()
        ).hexdigest()
    except (AuditContractError, TypeError, ValueError, RecursionError) as error:
        raise ReleaseRowError("release-row detector registry is malformed") from error
    if observed_registry_digest != registry_digest:
        raise ReleaseRowError("release-row detector registry digest does not match")
    try:
        expected_registry = release_row_registry(report.get("config"))
    except (ReleaseRowError, TypeError, ValueError) as error:
        raise ReleaseRowError("release-row report config is invalid") from error
    if (
        registry_payload != expected_registry.to_dict()
        or registry_digest != expected_registry.digest
    ):
        raise ReleaseRowError("release-row detector registry does not match report config")
    registry_detectors = registry_payload.get("detectors")
    detector_ids = (
        [item.get("detector_id") for item in registry_detectors if isinstance(item, Mapping)]
        if isinstance(registry_detectors, list)
        else []
    )
    if (
        len(detector_ids) != len(expected_registry.ids)
        or any(not isinstance(item, str) for item in detector_ids)
        or set(detector_ids) != set(expected_registry.ids)
    ):
        raise ReleaseRowError("release-row detector registry has an invalid detector set")
    return registry_digest, frozenset(expected_registry.ids)


def _validated_release_source(report: Mapping[str, Any], registry_digest: str) -> Mapping[str, Any]:
    """Require loader-produced publication provenance for the BA store handoff.

    Returns:
        The verified source identity mapping.
    """
    source = report.get("source")
    if not isinstance(source, Mapping):
        raise ReleaseRowError("release-row report lacks source identity")
    if source.get("detector_registry_digest") != registry_digest:
        raise ReleaseRowError("release-row source does not match the detector registry")
    if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("manifest_sha256", ""))):
        raise ReleaseRowError("release-row Auditor handoff requires a verified manifest digest")
    bundle_digest = source.get("bundle_sha256")
    if bundle_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", str(bundle_digest)):
        raise ReleaseRowError("release-row Auditor handoff has a malformed bundle digest")
    if not source.get("bundle_sha256") and not source.get("manifest_sha256"):
        raise ReleaseRowError("release-row Auditor handoff requires verified bundle provenance")
    members = source.get("episode_members")
    if (
        not isinstance(members, list)
        or not members
        or any(not isinstance(item, str) or not item for item in members)
        or len(members) != len(set(members))
    ):
        raise ReleaseRowError("release-row Auditor handoff requires verified episode members")
    try:
        canonical_json(source)
    except (AuditContractError, TypeError, ValueError, RecursionError) as error:
        raise ReleaseRowError("release-row source identity is not strict JSON") from error
    return source


def _typed_release_signal(
    index: int, payload: Any, *, allowed_detector_ids: Collection[str] = DETECTOR_IDS
) -> Signal:
    """Deserialize and validate one canonical BA-03 release candidate signal.

    Returns:
        The validated typed BA-03 signal.
    """
    if not isinstance(payload, Mapping):
        raise ReleaseRowError(f"release-row signal {index} must be an object")
    try:
        signal = record_from_dict(payload)
    except (AuditContractError, KeyError, TypeError, ValueError) as error:
        raise ReleaseRowError(f"release-row signal {index} is malformed") from error
    if not isinstance(signal, Signal):
        raise ReleaseRowError(f"release-row signal {index} is not a BA-03 Signal")
    if signal.detector_id not in allowed_detector_ids or signal.status != "flagged":
        raise ReleaseRowError(f"release-row signal {index} is outside the candidate contract")
    if record_to_dict(signal) != dict(payload):
        raise ReleaseRowError(f"release-row signal {index} is not canonically serialized")
    return signal


def _expected_release_signal(
    index: int,
    finding: Any,
    source: Mapping[str, Any],
    *,
    allowed_detector_ids: Collection[str] = DETECTOR_IDS,
) -> Signal:
    """Construct a typed BA-03 signal from one report finding.

    Returns:
        The expected typed signal.
    """

    if not isinstance(finding, Mapping):
        raise ReleaseRowError(f"release-row finding {index} must be an object")
    try:
        finding_members = finding.get("source_members")
        source_members = source.get("episode_members")
        if (
            not isinstance(finding_members, list)
            or not finding_members
            or any(not isinstance(item, str) or not item for item in finding_members)
            or finding_members != sorted(set(finding_members))
            or not isinstance(source_members, list)
            or not set(finding_members).issubset(source_members)
        ):
            raise ReleaseRowError(
                f"release-row finding {index} source members do not match source identity"
            )
        scope = {
            "scenario_id": finding["scenario_id"],
            "seed": finding["seed"],
            "planner_id": finding["planner_id"],
        }
        if finding.get("finding_id") != _finding_id(finding["detector_id"], scope, source):
            raise ReleaseRowError(f"release-row finding {index} does not match its source identity")
        return _typed_release_signal(
            index, _signal(finding, source), allowed_detector_ids=allowed_detector_ids
        )
    except ReleaseRowError:
        raise
    except (AuditContractError, KeyError, TypeError, ValueError, RecursionError) as error:
        raise ReleaseRowError(f"release-row finding {index} is malformed") from error


def _expected_release_signals(
    finding_payloads: list[Any],
    source: Mapping[str, Any],
    *,
    allowed_detector_ids: Collection[str] = DETECTOR_IDS,
) -> dict[str, dict[str, Any]]:
    """Index canonical BA-03 projections by source-bound finding IDs.

    Returns:
        The expected canonical payloads keyed by signal ID.
    """

    expected: dict[str, dict[str, Any]] = {}
    for index, finding in enumerate(finding_payloads):
        expected_signal = _expected_release_signal(
            index, finding, source, allowed_detector_ids=allowed_detector_ids
        )
        if expected_signal.signal_id in expected:
            raise ReleaseRowError(f"release-row finding {index} duplicates a finding ID")
        expected[expected_signal.signal_id] = record_to_dict(expected_signal)
    return expected


def _release_signal_records(
    report: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    allowed_detector_ids: Collection[str] = DETECTOR_IDS,
) -> list[Signal]:
    """Validate that typed signals exactly project the report findings.

    Returns:
        The canonical signal records sorted by ID.
    """

    finding_payloads = report.get("findings")
    if not isinstance(finding_payloads, list):
        raise ReleaseRowError("release-row report findings must be an array")
    expected = _expected_release_signals(
        finding_payloads, source, allowed_detector_ids=allowed_detector_ids
    )

    signal_payloads = report.get("signals")
    if not isinstance(signal_payloads, list):
        raise ReleaseRowError("release-row report signals must be an array")
    signals: list[Signal] = []
    seen_signal_ids: set[str] = set()
    for index, payload in enumerate(signal_payloads):
        signal = _typed_release_signal(index, payload, allowed_detector_ids=allowed_detector_ids)
        if signal.signal_id in seen_signal_ids:
            raise ReleaseRowError(f"release-row signal {index} duplicates a signal ID")
        seen_signal_ids.add(signal.signal_id)
        signals.append(signal)
    observed = {signal.signal_id: record_to_dict(signal) for signal in signals}
    if observed != expected:
        raise ReleaseRowError("release-row signals do not match report findings and source")
    return sorted(signals, key=lambda signal: signal.signal_id)


def handoff_release_row_signals(
    report: Mapping[str, Any], store: AuditStore
) -> CommitResult | BatchCommitResult | None:
    """Validate and persist release-row findings as typed BA-03 signals.

    This is a signal-store handoff only. It deliberately does not construct a
    BA-01 campaign scan, BA-02 queue summary, or human finding identity from
    aggregate release cells.

    Returns:
        The idempotent BA-03 store commit receipt, or ``None`` when no flagged
        signals were produced.
    """

    if not isinstance(report, Mapping) or report.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseRowError("release-row report has an unsupported schema")
    registry_digest, allowed_detector_ids = _validated_release_registry(report)
    source = _validated_release_source(report, registry_digest)
    signals = _release_signal_records(report, source, allowed_detector_ids=allowed_detector_ids)
    if not signals:
        return None
    try:
        identity = canonical_json(
            {
                "source": source,
                "detector_registry_digest": registry_digest,
                "signals": [record_to_dict(item) for item in signals],
            }
        )
    except (AuditContractError, TypeError, ValueError, RecursionError) as error:
        raise ReleaseRowError("release-row signal source identity is not strict JSON") from error
    operation_id = "release-row-anomalies:" + hashlib.sha256(identity.encode()).hexdigest()
    return store.commit(
        signals,
        operation_id=operation_id,
        actor="detector",
        actor_id="release-row-anomaly-gate",
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a complete, deterministic review table from a machine report.

    Returns:
        The Markdown report.
    """

    def cell(value: object) -> str:
        return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")

    gate = report["gate"]
    handoff = report.get("auditor_handoff")
    lines = [
        "# Release-row anomaly report",
        "",
        "Diagnostic row signals only; they provide neither per-step reconstruction nor proof of planner causation.",
        "",
        f"- Detector gate status: **{'BLOCKED' if gate['blocked'] else 'PASS'}** ({', '.join(gate['reasons']) or 'no blocking findings'})",
        f"- Input: {report['counts']['rows']} rows, {report['counts']['cells']} cells, {report['counts']['planners']} planners",
        f"- Findings: {report['counts']['findings']} ({gate['unannotated_findings']} unannotated)",
        f"- Preflight accounting: {report['preflight_accounting']['status']}",
        f"- Bundle SHA-256: `{report['source'].get('bundle_sha256') or 'unavailable'}`",
        f"- Execution rows: {report['execution_admission']['eligible_rows']} eligible, "
        f"{report['execution_admission']['unavailable_rows']} unavailable, "
        f"{report['execution_admission']['error_rows']} malformed",
        "Missing summary fields remain unavailable.",
        "",
        "## Detector counts",
        "",
        "| Detector | Findings |",
        "| --- | ---: |",
    ]
    if isinstance(handoff, Mapping):
        lines.insert(
            7,
            "- Benchmark Auditor BA-03 store handoff: "
            f"**{handoff['status']}** ({handoff['signal_count']} signals)",
        )
    lines += [
        f"| {cell(name)} | {count} |" for name, count in report["counts"]["by_detector"].items()
    ]
    lines += ["", "## Missing evidence", "", "| Field | Count |", "| --- | ---: |"]
    lines += [f"| {cell(name)} | {count} |" for name, count in report["missingness"].items()]
    lines += [
        "",
        "## Findings",
        "",
        "| Detector | Scenario | Seed | Planner | Evidence | Annotated |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for finding in report["findings"]:
        measured = json.dumps(finding["measured"], sort_keys=True, separators=(",", ":"))
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    finding["detector_id"],
                    finding["scenario_id"],
                    finding["seed"],
                    finding["planner_id"],
                    measured,
                    "yes" if finding["annotated"] else "no",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline report and optional release gate.

    Returns:
        Exit status: 0 for report success, 1 for a blocked gate, 2 for invalid input.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, required=True, help="Publication .tar.gz or extracted bundle root"
    )
    parser.add_argument("--config", type=Path, help="JSON threshold configuration")
    parser.add_argument("--annotations", type=Path, help="JSON root-cause annotations")
    parser.add_argument("--preflight", type=Path, help="JSON scenario/seed invalid_run preflight")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--audit-store",
        type=Path,
        help="optionally commit typed release-row BA-03 signals to this AuditStore directory",
    )
    parser.add_argument(
        "--expected-bundle-sha256",
        help="Require the archive bytes to match a separately recorded SHA-256 digest",
    )
    parser.add_argument(
        "--release-gate", action="store_true", help="Exit 1 when the report blocks release"
    )
    args = parser.parse_args(argv)
    try:
        rows, source = load_release_rows(args.bundle)
        if args.expected_bundle_sha256 is not None:
            expected_digest = args.expected_bundle_sha256.lower()
            if (
                len(expected_digest) != 64
                or any(character not in "0123456789abcdef" for character in expected_digest)
                or source.get("bundle_sha256") != expected_digest
            ):
                raise ReleaseRowError("publication archive SHA-256 does not match expected digest")
        report = analyze_release_rows(
            rows,
            config=_json_file(args.config) if args.config else None,
            annotations=_json_file(args.annotations) if args.annotations else None,
            preflight=_json_file(args.preflight) if args.preflight else None,
            source=source,
        )
        if args.audit_store is not None:
            with AuditStore(args.audit_store) as store:
                receipt = handoff_release_row_signals(report, store)
            report["auditor_handoff"] = {
                "status": "submitted" if receipt is not None else "no_findings",
                "operation_id": receipt.operation_id if receipt is not None else None,
                "signal_count": len(report["signals"]),
            }
        for path, content in (
            (
                args.output_json,
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            ),
            (args.output_markdown, render_markdown(report)),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as error:
        sys.stderr.write(f"release-row audit: {error}\n")
        return 2
    return 1 if args.release_gate and report["gate"]["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
