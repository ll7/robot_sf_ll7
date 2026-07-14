#!/usr/bin/env python3
"""Build a fail-closed paired-outcome report for issue #5416.

This CPU-only utility consumes retained episode JSONL and explicit execution
provenance manifests. It does not execute episodes, submit compute, or change
metric definitions. Invalid, non-native, incomplete, duplicated, and unproven
rows remain visible and block interpretation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.algorithm_metadata import canonical_algorithm_name
from scripts.validation import check_issue_5416_sipp_four_geometry_packet as packet_checker

ISSUE = 5416
EXECUTION_PROVENANCE_SCHEMA = "issue_5416_execution_provenance.v1"
RUNTIME_FIELD = "planner_step_runtime_seconds"
DIAGNOSTICS = "expansion_limit_hits runtime_bound_exits fallback_count commitment_invalidations".split()  # fmt: skip
MANIFEST_FIELDS = "exact_command environment_manifest cpu_route job_id resource_request wall_time_seconds".split()  # fmt: skip
CLAIM_BOUNDARY = (
    "diagnostic paired-outcome aggregation only; no planner superiority, safety, liveness, "
    "benchmark, paper, or dissertation claim is promoted"
)
CRITERION_REMAINING = [
    tuple(item.split("|", 1))
    for item in (
        "Native 5-planner × 4-geometry × 5-seed matrix is complete.|Provide one integrity-clean native row for every frozen planner/scenario/seed key.",
        "Primary outcomes use paired success-only progress/time denominators.|Retain the required outcome fields in every eligible row and rerun the report.",
        "Planner expansion, bound, fallback, invalidation, and step-runtime diagnostics are present.|Emit explicit planner_diagnostics; episode wall time is not a step-runtime substitute.",
        "Execution provenance is complete for all five planner inputs.|Provide issue_5416_execution_provenance.v1 manifests with command, environment, CPU/resource, wall time, and artifact hash.",
        "Paired comparison and the predeclared advancement rule are auditable.|Complete the native bundle and resolve exclusions before interpreting the decision.",
    )
]


class AnalysisError(ValueError):
    """Raised when an input cannot be interpreted under the frozen contract."""


def _number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise AnalysisError(f"{label} must be finite and >= {minimum}")
    return result


def _metric(metrics: Mapping[str, Any], key: str, reasons: list[str]) -> float | None:
    try:
        return _number(metrics[key], f"metrics.{key}", minimum=0.0)
    except (AnalysisError, KeyError):
        reasons.append(f"metrics.{key} is missing or invalid")
        return None


def _nonnegative_int(value: object, label: str) -> int:
    try:
        result = int(value)
        if isinstance(value, bool) or float(value) != result or result < 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{label} must be a non-negative integer") from exc
    return result


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _load_jsonl(paths: Sequence[Path]) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            errors.append(f"missing episode JSONL: {path}")
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            errors.append(f"cannot read episode JSONL {path}: {exc}")
            continue
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{number} invalid JSON: {exc.msg}")
                continue
            if isinstance(payload, dict):
                rows.append((path, payload))
            else:
                errors.append(f"{path}:{number} must be a JSON object")
    return rows, errors


def _packet(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        packet = packet_checker.load_packet(path)
        gate = packet_checker.validate_packet(packet)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as exc:
        return {}, [f"preregistration packet is not ready: {exc}"]
    if gate.get("status") != "ready":
        return packet, [
            "preregistration geometry gate is blocked: "
            + ", ".join(str(row) for row in gate.get("blocked_rows", []))
        ]
    return packet, []


def _dimensions(
    packet: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    contract = packet["scenario_contract"]
    selected = contract.get("selected_rows")
    roster = packet["planner_roster"]["required"]
    scenarios = tuple(str(row["scenario_id"]) for row in selected)
    seeds = tuple(int(seed) for seed in contract.get("result_producing_seeds", []))
    planners = tuple(str(row["planner_id"]) for row in roster)
    return scenarios, seeds, planners


def _planner_id(row: Mapping[str, Any], planners: Sequence[str]) -> str | None:
    direct = row.get("planner_id")
    metadata = row.get("algorithm_metadata")
    config = metadata.get("config") if isinstance(metadata, Mapping) else None
    variant = config.get("planner_variant") if isinstance(config, Mapping) else None
    if isinstance(direct, str) and direct in planners:
        return direct
    if isinstance(variant, str) and variant in planners:
        return variant
    raw = metadata.get("algorithm") if isinstance(metadata, Mapping) else row.get("algo")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        canonical = canonical_algorithm_name(raw)
    except (TypeError, ValueError):
        return None
    return canonical if canonical in planners else None


def _row_provenance(row: Mapping[str, Any], scenario: str, seed: int) -> list[str]:
    provenance = row.get("result_provenance")
    if not isinstance(provenance, Mapping):
        return ["missing result_provenance"]
    reasons = []
    if provenance.get("schema_version") != "benchmark_row_provenance.v1":
        reasons.append("row provenance schema mismatch")
    if provenance.get("scenario_id") != scenario or provenance.get("seed") != seed:
        reasons.append("row provenance scenario/seed mismatch")
    for key in ("config_hash", "repo_commit"):
        value = provenance.get(key)
        if not isinstance(value, str) or not value.strip() or value.lower() == "unknown":
            reasons.append(f"row provenance {key} missing")
    simulator = provenance.get("simulator_settings")
    if not isinstance(simulator, Mapping) or simulator.get("horizon") != 500:
        reasons.append("row provenance horizon missing or mismatched")
    try:
        if not math.isclose(float(simulator.get("dt")), 0.1, rel_tol=0.0, abs_tol=1e-9):
            reasons.append("row provenance dt mismatch")
    except (AttributeError, TypeError, ValueError):
        reasons.append("row provenance dt missing")
    return reasons


def _measurement(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    metrics, outcome = row.get("metrics"), row.get("outcome")
    if not isinstance(metrics, Mapping) or not isinstance(outcome, Mapping):
        return None, ["metrics and outcome mappings are required"]
    if not all(
        isinstance(outcome.get(key), bool)
        for key in ("route_complete", "collision_event", "timeout_event")
    ):
        return None, ["outcome booleans are incomplete"]
    collision = bool(outcome["collision_event"])
    success = bool(outcome["route_complete"]) and not collision
    reasons: list[str] = []
    ped_count = _metric(metrics, "ped_collision_count", reasons)
    static_count = _metric(metrics, "obstacle_collision_count", reasons)
    deadlock = metrics.get("deadlock")
    if not isinstance(deadlock, bool):
        reasons.append("deadlock signal is missing")
        deadlock = None
    values = {
        key: _metric(metrics, key, reasons) for key in ("time_to_goal_norm", "path_efficiency")
    }
    return {
        "success": success,
        "pedestrian_and_static_collision": (
            ped_count + static_count > 0
            if ped_count is not None and static_count is not None
            else None
        ),
        "deadlock": deadlock,
        "timeout_event": bool(outcome["timeout_event"])
        or str(row.get("termination_reason", "")).lower() in {"max_steps", "truncated"},
        **values,
    }, reasons


def _diagnostics(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    metadata = row.get("algorithm_metadata")
    source = metadata.get("planner_diagnostics") if isinstance(metadata, Mapping) else None
    if not isinstance(source, Mapping):
        return None, ["algorithm_metadata.planner_diagnostics is missing"]
    result: dict[str, Any] = {}
    errors = []
    for key in DIAGNOSTICS:
        try:
            result[key] = _nonnegative_int(source.get(key), f"planner_diagnostics.{key}")
        except AnalysisError as exc:
            errors.append(str(exc))
    runtimes = source.get(RUNTIME_FIELD)
    if isinstance(runtimes, (str, bytes)) or not isinstance(runtimes, Sequence) or not runtimes:
        errors.append(f"planner_diagnostics.{RUNTIME_FIELD} must be a non-empty sequence")
    else:
        try:
            result[RUNTIME_FIELD] = [
                _number(value, RUNTIME_FIELD, minimum=0.0) for value in runtimes
            ]
        except AnalysisError as exc:
            errors.append(str(exc))
    return (result if not errors else None), errors


def _parse_row(  # noqa: C901
    row: Mapping[str, Any], planner: str | None, scenario: str | None, seed: int | None
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    reasons = []
    reasons.extend(
        message
        for value, message in (
            (planner, "planner is not in the frozen roster"),
            (scenario, "scenario is not in the frozen matrix"),
            (seed, "seed is not in the frozen result-producing set"),
        )
        if value is None
    )
    if row.get("version") != "v1":
        reasons.append("episode schema version is not v1")
    if row.get("horizon") != 500:
        reasons.append("episode horizon is not 500")
    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        reasons.append("algorithm_metadata is missing")
    else:
        if metadata.get("status") != "ok":
            reasons.append(f"planner metadata status is {metadata.get('status')!r}")
        if metadata.get("fallback_or_degraded") is True:
            reasons.append("planner metadata is fallback/degraded")
        kinematics = metadata.get("planner_kinematics")
        if not isinstance(kinematics, Mapping) or kinematics.get("execution_mode") != "native":
            reasons.append("execution_mode is not native")
    integrity = row.get("integrity")
    if not isinstance(integrity, Mapping):
        reasons.append("integrity is missing")
    elif integrity.get("contradictions") != []:
        reasons.append("integrity contradictions are present")
    if planner and scenario and seed is not None:
        reasons.extend(_row_provenance(row, scenario, seed))
    measurement, measurement_errors = _measurement(row)
    reasons.extend(measurement_errors)
    diagnostics, diagnostic_errors = _diagnostics(row)
    if reasons or measurement is None:
        return None, reasons or ["row cannot be keyed"], diagnostic_errors
    return (
        {
            **measurement,
            "diagnostics": diagnostics,
            RUNTIME_FIELD: diagnostics.get(RUNTIME_FIELD, []) if diagnostics else [],
        },
        reasons,
        diagnostic_errors,
    )


def _manifest_errors(
    path: Path,
    payload: Mapping[str, Any],
    episodes: Sequence[Path],
    planners: Sequence[str],
    seen: Mapping[str, Any],
) -> tuple[str | None, list[str]]:
    planner = payload.get("planner_id")
    errors = []
    if not isinstance(planner, str) or planner not in planners:
        return None, [f"execution manifest {path} has unknown planner_id"]
    if planner in seen:
        errors.append(f"duplicate execution manifest for planner {planner}")
    if payload.get("schema_version") != EXECUTION_PROVENANCE_SCHEMA:
        errors.append(f"execution manifest {path} schema mismatch")
    source = payload.get("episodes_path")
    source_path = Path(source) if isinstance(source, str) else Path()
    if not source_path.is_absolute():
        source_path = path.parent / source_path
    if not isinstance(source, str) or not any(
        source_path.resolve() == item.resolve() for item in episodes
    ):
        errors.append(f"execution manifest {path} does not point to an input JSONL")
    provenance = payload.get("execution_provenance")
    if not isinstance(provenance, Mapping):
        errors.append(f"execution manifest {path} is missing execution_provenance")
    else:
        errors.extend(
            f"execution manifest {path} is missing {key}"
            for key in MANIFEST_FIELDS
            if provenance.get(key) in (None, "", {})
        )
        try:
            _number(provenance.get("wall_time_seconds"), "wall_time_seconds", minimum=0.0)
        except AnalysisError as exc:
            errors.append(f"execution manifest {path}: {exc}")
    digest = payload.get("episodes_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append(f"execution manifest {path} is missing episodes_sha256")
    elif source_path.is_file() and hashlib.sha256(source_path.read_bytes()).hexdigest() != digest:
        errors.append(f"execution manifest {path} episodes_sha256 mismatch")
    return planner, errors


def _load_manifests(
    paths: Sequence[Path], episodes: Sequence[Path], planners: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    manifests: dict[str, Any] = {}
    errors = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot load execution manifest {path}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"execution manifest {path} must be an object")
            continue
        planner, manifest_errors = _manifest_errors(path, payload, episodes, planners, manifests)
        errors.extend(manifest_errors)
        if planner is not None and planner not in manifests:
            manifests[planner] = payload
    errors.extend(
        f"execution manifest missing for planner {planner}"
        for planner in planners
        if planner not in manifests
    )
    return manifests, errors


def _summary(planner: str, scenario: str, values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [item for item in values if item["success"]]
    complete_diagnostics = all(
        isinstance(item.get("diagnostics"), Mapping) and item.get(RUNTIME_FIELD) for item in values
    )
    runtimes = (
        [runtime for item in values for runtime in item.get(RUNTIME_FIELD, [])]
        if complete_diagnostics
        else []
    )
    counts = {
        key: sum(int(item["diagnostics"][key]) for item in values) if complete_diagnostics else None
        for key in DIAGNOSTICS
    }
    return {
        "planner_id": planner,
        "scenario_id": scenario,
        "success_rate": _mean([float(item["success"]) for item in values]),
        "pedestrian_and_static_collision_count_rate": _mean(
            [float(item["pedestrian_and_static_collision"]) for item in values]
        ),
        "deadlock_timeout_max_steps_rate": _mean(
            [float(item["timeout_event"] or item["deadlock"]) for item in values]
        ),
        "paired_path_efficiency_mean_success_only": _mean(
            [item["path_efficiency"] for item in successful if item["path_efficiency"] is not None]
        ),
        "paired_time_to_goal_norm_mean_success_only": _mean(
            [
                item["time_to_goal_norm"]
                for item in successful
                if item["time_to_goal_norm"] is not None
            ]
        ),
        "success_only_denominator": len(successful),
        "planner_step_runtime_median_seconds": _percentile(runtimes, 0.5),
        "planner_step_runtime_p95_seconds": _percentile(runtimes, 0.95),
        **counts,
    }


def _paired(
    values: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    planners: Sequence[str],
) -> list[dict[str, Any]]:
    by_key = {(item["scenario_id"], item["seed"], item["planner_id"]): item for item in values}
    candidate, result = planners[0], []
    for scenario, seed, comparator in product(scenarios, seeds, planners[1:]):
        left, right = (
            by_key.get((scenario, seed, candidate)),
            by_key.get((scenario, seed, comparator)),
        )
        if left is None or right is None:
            continue
        result.append(
            {
                "scenario_id": scenario,
                "seed": seed,
                "candidate": candidate,
                "comparator": comparator,
                "success_delta": int(left["success"]) - int(right["success"]),
                "pedestrian_and_static_collision_delta": int(
                    left["pedestrian_and_static_collision"]
                )
                - int(right["pedestrian_and_static_collision"]),
                "candidate_deadlock_timeout_max_steps": left["deadlock"] or left["timeout_event"],
                "comparator_deadlock_timeout_max_steps": right["deadlock"]
                or right["timeout_event"],
                "path_efficiency_delta_success_only": left["path_efficiency"]
                - right["path_efficiency"]
                if left["success"] and right["success"]
                else None,
                "time_to_goal_norm_delta_success_only": left["time_to_goal_norm"]
                - right["time_to_goal_norm"]
                if left["success"] and right["success"]
                else None,
            }
        )
    return result


def _decision(
    summaries: Sequence[Mapping[str, Any]],
    planners: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_planner = {row["planner_id"]: row for row in summaries}
    if any(planner not in by_planner for planner in planners):
        return {"status": "blocked", "reason": "complete planner summaries are required"}
    candidate = planners[0]
    left = by_planner[candidate]
    no_collision_regression = all(
        left["pedestrian_and_static_collision_count_rate"]
        <= by_planner[planner]["pedestrian_and_static_collision_count_rate"]
        for planner in planners[1:]
    )
    improvement = any(
        left["success_rate"] > by_planner[planner]["success_rate"]
        or left["deadlock_timeout_max_steps_rate"]
        < by_planner[planner]["deadlock_timeout_max_steps_rate"]
        for planner in planners[1:]
    )
    return {
        "status": "pass" if no_collision_regression and improvement else "stop_or_revise",
        "candidate": candidate,
        "comparators": list(planners[1:]),
        "no_collision_regression": no_collision_regression,
        "improvement_observed": improvement,
    }


def _criterion(name: str, status: str, evidence: str, remaining: str) -> dict[str, str]:
    return {"criterion": name, "status": status, "evidence": evidence, "remaining_work": remaining}


def build_analysis(
    *,
    episode_paths: Sequence[str | Path],
    output_dir: str | Path,
    packet_path: str
    | Path = "configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml",
    provenance_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build and write the paired-analysis artifacts; return the report mapping."""
    episode_paths = [Path(path) for path in episode_paths]
    provenance_paths = [Path(path) for path in provenance_paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet, packet_errors = _packet(Path(packet_path))
    dimensions: tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]] = ((), (), ())
    try:
        if packet:
            dimensions = _dimensions(packet)
    except AnalysisError as exc:
        packet_errors.append(str(exc))
    scenarios, seeds, planners = dimensions
    rows, load_errors = (
        _load_jsonl(episode_paths) if planners else ([], ["packet dimensions unavailable"])
    )
    expected = set(product(scenarios, seeds, planners))
    seen: defaultdict[tuple[str, int, str], int] = defaultdict(int)
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    diagnostic_errors = []
    for source, row in rows:
        planner = _planner_id(row, planners)
        raw_scenario, raw_seed = row.get("scenario_id"), row.get("seed")
        scenario = (
            raw_scenario if isinstance(raw_scenario, str) and raw_scenario in scenarios else None
        )
        seed = raw_seed if isinstance(raw_seed, int) and raw_seed in seeds else None
        key = (scenario, seed, planner) if planner and scenario and seed is not None else None
        if key:
            seen[key] += 1
        measurement, reasons, row_diagnostic_errors = _parse_row(row, planner, scenario, seed)
        diagnostic_errors.extend(
            f"{source}: {scenario}/{seed}/{planner}: {error}" for error in row_diagnostic_errors
        )
        if reasons or key is None or measurement is None:
            exclusions.append(
                {
                    "source": str(source),
                    "scenario_id": scenario,
                    "seed": seed,
                    "planner_id": planner,
                    "key": list(key) if key else None,
                    "reasons": sorted(set(reasons or ["row cannot be keyed"])),
                }
            )
            continue
        eligible.append(
            {**measurement, "scenario_id": scenario, "seed": seed, "planner_id": planner}
        )
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        exclusions.extend(
            {
                "source": "matrix",
                "scenario_id": key[0],
                "seed": key[1],
                "planner_id": key[2],
                "key": list(key),
                "reasons": ["duplicate matrix key"],
            }
            for key in duplicates
        )
        eligible = [
            item
            for item in eligible
            if (item["scenario_id"], item["seed"], item["planner_id"]) not in duplicates
        ]
    missing = sorted(
        expected - {(item["scenario_id"], item["seed"], item["planner_id"]) for item in eligible}
    )
    planned: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        planned[item["planner_id"]].append(item)
    summaries = [
        _summary(planner, "__all__", planned[planner]) for planner in planners if planned[planner]
    ]
    pairs = _paired(eligible, scenarios, seeds, planners) if planners else []
    manifests, provenance_errors = _load_manifests(provenance_paths, episode_paths, planners)
    matrix_complete = (
        bool(expected)
        and not packet_errors
        and not load_errors
        and not duplicates
        and not missing
        and len(eligible) == len(expected)
    )
    outcomes_complete = matrix_complete and all(
        item["pedestrian_and_static_collision"] is not None
        and item["deadlock"] is not None
        and (
            not item["success"]
            or (item["path_efficiency"] is not None and item["time_to_goal_norm"] is not None)
        )
        for item in eligible
    )
    diagnostics_complete = (
        matrix_complete
        and not diagnostic_errors
        and all(
            isinstance(item.get("diagnostics"), Mapping) and item.get(RUNTIME_FIELD)
            for item in eligible
        )
    )
    provenance_complete = (
        bool(planners) and not provenance_errors and len(manifests) == len(planners)
    )
    paired_complete = outcomes_complete and len(pairs) == len(scenarios) * len(seeds) * (
        len(planners) - 1
    )
    all_summaries = [row for row in summaries if row["scenario_id"] == "__all__"]
    decision = (
        _decision(all_summaries, planners, pairs)
        if paired_complete and diagnostics_complete and provenance_complete
        else {
            "status": "blocked",
            "reason": "native outcomes, diagnostics, and execution provenance must pass before applying the advancement rule",
        }
    )
    statuses = [
        matrix_complete,
        outcomes_complete,
        diagnostics_complete,
        provenance_complete,
        paired_complete and decision["status"] in {"pass", "stop_or_revise"},
    ]
    evidence = [
        f"eligible rows={len(eligible)}, expected rows={len(expected)}; exclusions={len(exclusions)}",
        f"summary rows={len(summaries)}; collision components, deadlock, progress, and time fields are required",
        f"diagnostic errors={len(diagnostic_errors)}; runtime field={RUNTIME_FIELD}",
        f"manifests={len(manifests)}/{len(planners)}; provenance errors={len(provenance_errors)}",
        f"paired comparison rows={len(pairs)}; decision={decision['status']}",
    ]
    criteria = [
        _criterion(name, "met" if status else "blocked", detail, remaining)
        for (name, remaining), status, detail in zip(
            CRITERION_REMAINING, statuses, evidence, strict=True
        )
    ]
    report = {
        "issue": ISSUE,
        "status": "complete" if all(item["status"] == "met" for item in criteria) else "partial",
        "claim_boundary": CLAIM_BOUNDARY,
        "matrix": {
            "expected_rows": len(expected),
            "eligible_rows": len(eligible),
            "excluded_rows": len(exclusions),
            "missing_keys": [list(key) for key in missing],
            "duplicate_keys": [list(key) for key in duplicates],
        },
        "input_errors": load_errors + packet_errors,
        "denominator_exclusions": exclusions,
        "planner_summaries": summaries,
        "paired_comparisons": pairs,
        "diagnostic_errors": diagnostic_errors,
        "provenance": {
            "complete": provenance_complete,
            "manifests": sorted(manifests),
            "errors": provenance_errors,
        },
        "advancement_rule": decision,
        "criteria": criteria,
        "blockers": [item["remaining_work"] for item in criteria if item["status"] != "met"],
    }
    _write_artifacts(report, output_dir)
    return report


def _write_artifacts(report: Mapping[str, Any], output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    exclusions = []
    for item in report["denominator_exclusions"]:
        exclusions.append(
            {
                "source": item.get("source"),
                "scenario_id": item.get("scenario_id"),
                "seed": item.get("seed"),
                "planner_id": item.get("planner_id"),
                "key": json.dumps(item.get("key"), separators=(",", ":")),
                "reasons": "; ".join(item.get("reasons", [])),
            }
        )
    with (output_dir / "denominator_exclusions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("source", "scenario_id", "seed", "planner_id", "key", "reasons")
        )
        writer.writeheader()
        writer.writerows(exclusions)
    comparisons = report["paired_comparisons"]
    with (output_dir / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for item in comparisons for key in item}) or ["scenario_id"],
        )
        writer.writeheader()
        writer.writerows(comparisons)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analyzer and return non-zero while the native bundle is blocked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=Path, required=True)
    parser.add_argument("--provenance-manifest", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path("configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/context/evidence/issue_5416_sipp_four_geometry"),
    )
    args = parser.parse_args(argv)
    report = build_analysis(
        episode_paths=args.episodes,
        output_dir=args.output_dir,
        packet_path=args.packet,
        provenance_paths=args.provenance_manifest,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": str(args.output_dir / "summary.json"),
                "eligible_rows": report["matrix"]["eligible_rows"],
                "excluded_rows": report["matrix"]["excluded_rows"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
