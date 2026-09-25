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
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.audit_contracts import Signal, record_to_dict
from robot_sf.analysis_workbench.audit_detectors import DetectorRegistry, DetectorSpec
from robot_sf.analysis_workbench.release_row_bundle import load_release_rows

SCHEMA_VERSION = "release-row-anomalies.v1"
DETECTOR_VERSION = "1.0.0"
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
    "pedestrian_free_scenarios": ["classic_bottleneck_low"],
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


class ReleaseRowError(ValueError):
    """Malformed release rows, configuration, or accounting inputs."""


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReleaseRowError(f"{name} must be an integer >= {minimum}")
    return value


def _configured(config: Mapping[str, Any] | None) -> dict[str, Any]:  # noqa: C901
    if config is None:
        supplied: Mapping[str, Any] = {}
    elif isinstance(config, Mapping):
        supplied = config
    else:
        raise ReleaseRowError("config must be an object")
    unknown = set(supplied) - set(DEFAULT_CONFIG)
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
    scenarios = result["pedestrian_free_scenarios"]
    if not isinstance(scenarios, list) or any(
        not isinstance(item, str) or not item.strip() for item in scenarios
    ):
        raise ReleaseRowError("pedestrian_free_scenarios must be a list of scenario IDs")
    if len(scenarios) != len(set(scenarios)):
        raise ReleaseRowError("pedestrian_free_scenarios contains duplicates")
    if type(result["require_preflight"]) is not bool:
        raise ReleaseRowError("require_preflight must be a boolean")
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
        entries.append(dict(entry))
    return entries


def _matching_annotation(
    finding: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    for entry in entries:
        if "finding_id" in entry:
            if entry["finding_id"] == finding["finding_id"]:
                return entry
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


def _contact_speeds(row: Mapping[str, Any]) -> list[float] | None:
    ledger = row.get("event_ledger")
    events = ledger.get("collision_events") if isinstance(ledger, Mapping) else None
    if events is not None:
        if not isinstance(events, list) or any(not isinstance(item, Mapping) for item in events):
            raise ReleaseRowError("collision_events must be an array of objects")
        values = [_finite(item.get("relative_speed_at_contact")) for item in events]
        if any(value is not None and value < 0 for value in values):
            raise ReleaseRowError("relative_speed_at_contact must be >= 0")
        return [value for value in values if value is not None] or None
    value = _finite(row["metrics"].get("max_relative_contact_speed_m_s"))
    if value is not None:
        if value < 0:
            raise ReleaseRowError("max_relative_contact_speed_m_s must be >= 0")
        return [value]
    return None


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
        "detector_version": DETECTOR_VERSION,
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
        detector_version=DETECTOR_VERSION,
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
        "orbit_zero_progress": "Recorded curvature, displacement, progress, or deadlock indicates a stall.",
        "pedestrian_free_baseline_regression": "Paired pedestrian-free success is worse than blind goal.",
        "universal_failure_unannotated": "Every planner fails a cell without root-cause annotation.",
        "invalid_run_preflight_mismatch": "Episode invalid_run differs from scenario-seed preflight.",
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
            "min_paired_cells",
            "min_success_rate_gap",
        ),
        "universal_failure_unannotated": ("min_planners_per_cell",),
        "invalid_run_preflight_mismatch": (),
    }
    specs = tuple(
        DetectorSpec(
            detector_id=detector_id,
            family="release_row_anomaly",
            description=descriptions[detector_id],
            version=DETECTOR_VERSION,
            required_capabilities=("episode",),
            cohort_definition={"kind": "release_cell", "key": ["scenario_id", "seed"]},
            parameters={key: settings[key] for key in parameters[detector_id]},
            units={"steps": "count", "contact_speed": "m/s", "displacement": "m"},
            provenance={
                "owner": __name__,
                "source": "published_episode_rows",
                "evidence_boundary": "diagnostic_only",
            },
        )
        for detector_id in sorted(DETECTOR_IDS)
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
    registry = release_row_registry(settings)
    records, observed_planners = _rows(rows)
    source_info = dict(source or {})
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
    source_info.setdefault("row_count", len(records))
    annotation_entries = _annotation_entries(annotations)
    preflight_map = _preflight_cells(preflight)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_scenario_planner: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        grouped[(row["scenario_id"], row["seed"])].append(row)
        by_scenario_planner[(row["scenario_id"], row["_release_arm"])][row["seed"]] = row

    findings: list[dict[str, Any]] = []
    missingness: Counter[str] = Counter()
    if preflight_map is None:
        missingness["preflight_unavailable"] = 1
    coverage = {"complete_cells": 0, "incomplete_cells": 0, "incomplete_cell_ids": []}
    annotated_universal_cells = 0

    def append(finding: dict[str, Any]) -> None:
        entry = _matching_annotation(finding, annotation_entries)
        finding["annotated"] = entry is not None
        finding["annotation"] = (
            {"root_cause": entry["root_cause"], "source_ref": entry["source_ref"]}
            if entry
            else None
        )
        findings.append(finding)

    for (scenario, seed), cell in sorted(grouped.items()):
        planners = {row["_release_arm"] for row in cell}
        complete = planners == set(expected_planners)
        coverage["complete_cells" if complete else "incomplete_cells"] += 1
        if not complete:
            coverage["incomplete_cell_ids"].append(
                {
                    "scenario_id": scenario,
                    "seed": seed,
                    "missing_planners": sorted(set(expected_planners) - planners),
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
                    measured={"planner_count": len(planners), "failure_step": next(iter(steps))},
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
            if _matching_annotation(candidate, annotation_entries):
                annotated_universal_cells += 1
            else:
                append(candidate)
        if preflight_map is not None:
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
                speeds = _contact_speeds(row)
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
                            measured={"max_relative_speed_at_contact_m_s": max(speeds)},
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
            deadlock = metrics.get("deadlock")
            if deadlock is not None and type(deadlock) is not bool:
                raise ReleaseRowError("metrics.deadlock must be a boolean when present")
            if curvature is None:
                missingness["curvature_unavailable"] += 1
            if path_length is None:
                missingness["path_length_unavailable"] += 1
            if displacement is None:
                missingness["displacement_unavailable"] += 1
            if deadlock is None:
                missingness["deadlock_unavailable"] += 1
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
            if orbit or zero_progress or low_progress_ratio or deadlock is True:
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
                            "deadlock": deadlock,
                            "signatures": [
                                name
                                for name, active in (
                                    ("high_curvature", orbit),
                                    ("low_displacement", zero_progress),
                                    ("low_progress_ratio", low_progress_ratio),
                                    ("deadlock", deadlock is True),
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

    for scenario in settings["pedestrian_free_scenarios"]:
        baseline_rows = by_scenario_planner.get((scenario, settings["baseline_planner"]), {})
        if not baseline_rows:
            missingness["pedestrian_free_baseline_missing"] += 1
            continue
        for planner in expected_planners:
            if planner == settings["baseline_planner"]:
                continue
            candidate_rows = by_scenario_planner.get((scenario, planner), {})
            shared_seeds = set(baseline_rows) & set(candidate_rows)
            paired_seeds = sorted(
                seed
                for seed in shared_seeds
                if _pedestrian_free(baseline_rows[seed]) is True
                and _pedestrian_free(candidate_rows[seed]) is True
            )
            unavailable_pairs = sum(
                _pedestrian_free(baseline_rows[seed]) is None
                or _pedestrian_free(candidate_rows[seed]) is None
                for seed in shared_seeds
            )
            if unavailable_pairs:
                missingness["pedestrian_free_status_unavailable"] += unavailable_pairs
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
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": "Diagnostic release-row signals; no per-step reconstruction or causal attribution.",
        "source": source_info,
        "config": settings,
        "detector_registry": registry.to_dict(),
        "detector_registry_digest": registry.digest,
        "coverage": coverage,
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


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a complete, deterministic review table from a machine report.

    Returns:
        The Markdown report.
    """

    def cell(value: object) -> str:
        return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")

    gate = report["gate"]
    lines = [
        "# Release-row anomaly report",
        "",
        f"- Gate: **{'BLOCKED' if gate['blocked'] else 'PASS'}** ({', '.join(gate['reasons']) or 'no blocking findings'})",
        f"- Input: {report['counts']['rows']} rows, {report['counts']['cells']} cells, {report['counts']['planners']} planners",
        f"- Findings: {report['counts']['findings']} ({gate['unannotated_findings']} unannotated)",
        f"- Preflight accounting: {report['preflight_accounting']['status']}",
        f"- Bundle SHA-256: `{report['source'].get('bundle_sha256') or 'unavailable'}`",
        "",
        "These are diagnostic row signals, not proof of planner causation. Missing summary fields remain unavailable.",
        "",
        "## Detector counts",
        "",
        "| Detector | Findings |",
        "| --- | ---: |",
    ]
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
        for path, content in (
            (
                args.output_json,
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            ),
            (args.output_markdown, render_markdown(report)),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        sys.stderr.write(f"release-row audit: {error}\n")
        return 2
    return 1 if args.release_gate and report["gate"]["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
