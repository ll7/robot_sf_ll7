"""Fail-closed evaluator and Markdown renderer for reference-planner oracle runs.

The report describes episode-level results for a pedestrian-free goal oracle, a
stationary robot with the source pedestrian population, and configured paired
pedestrian-aware planner arms. A passing gate is a row-contract result for the
declared input matrix; it does not by itself establish release or paper evidence.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

REPORT_SCHEMA_VERSION = "reference_oracle_report.v1"
_THRESHOLD_KEYS = frozenset(
    {
        "max_goal_failure_count",
        "max_stationary_contact_rate",
        "max_dominance_regressions",
    }
)
_ALLOWED_STATUSES = frozenset({"success", "collision", "failure"})
_NATIVE_RUN_MODES = frozenset({"native", "adapter"})
_DEGRADED_MARKERS = frozenset(
    {"fallback", "degraded", "failed", "unavailable", "not_available", "not-available"}
)


def evaluate_oracles(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    release_id: str,
    scenario_ids: list[str],
    seeds: list[int],
    rows_by_arm: dict[str, list[dict]],
    goal_key: str,
    stationary_key: str,
    aware_keys: list[str],
    probe_scenario_ids: list[str],
    thresholds: dict[str, int | float],
    source_sha: str | None = None,
    stationary_no_pedestrian_scenario_ids: list[str] | None = None,
    expected_algorithms_by_arm: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate completeness, episode outcomes, contacts, and paired regressions.

    Malformed configuration and evidence are represented as gate reasons so the
    caller can still persist a useful failed report. Only a complete, unique,
    native-run row is used in the three threshold calculations. Diagnostic
    failure listings include every goal row whose available outcome fields do
    not jointly establish success, including duplicate or otherwise invalid rows.

    Returns:
        A JSON-serializable ``reference_oracle_report.v1`` payload.
    """
    violations: list[dict[str, Any]] = []

    def add_violation(code: str, message: str, **details: Any) -> None:
        item: dict[str, Any] = {"code": code, "message": message}
        if details:
            item["details"] = details
        violations.append(item)

    normalized_release_id = release_id.strip() if isinstance(release_id, str) else ""
    if not normalized_release_id:
        add_violation("release_id_invalid", "release_id must be a non-empty string")

    scenarios = _unique_strings(scenario_ids)
    if scenarios is None:
        scenarios = []
        add_violation("scenario_ids_invalid", "scenario_ids must be a list of non-empty strings")
    elif not scenarios:
        add_violation("scenario_ids_empty", "scenario_ids must contain at least one scenario")
    if isinstance(scenario_ids, list) and len(scenarios) != len(scenario_ids):
        add_violation("scenario_ids_duplicate", "scenario_ids must not contain duplicates")

    normalized_seeds = _unique_seeds(seeds)
    if normalized_seeds is None:
        normalized_seeds = []
        add_violation("seeds_invalid", "seeds must be a list of unique integers")
    elif not normalized_seeds:
        add_violation("seeds_empty", "seeds must contain at least one seed")
    if isinstance(seeds, list) and len(normalized_seeds) != len(seeds):
        add_violation("seeds_duplicate", "seeds must not contain duplicates")

    probes = _unique_strings(probe_scenario_ids)
    if probes is None:
        probes = []
        add_violation(
            "probe_scenario_ids_invalid",
            "probe_scenario_ids must be a list of non-empty strings",
        )
    elif isinstance(probe_scenario_ids, list) and len(probes) != len(probe_scenario_ids):
        add_violation(
            "probe_scenario_ids_duplicate",
            "probe_scenario_ids must not contain duplicates",
        )
    for probe in probes:
        if probe not in scenarios:
            add_violation(
                "unknown_probe_scenario",
                f"Probe scenario {probe!r} is not in the declared scenario matrix",
                scenario_id=probe,
            )

    no_pedestrian_scenarios = _unique_strings(
        []
        if stationary_no_pedestrian_scenario_ids is None
        else stationary_no_pedestrian_scenario_ids
    )
    if no_pedestrian_scenarios is None:
        no_pedestrian_scenarios = []
        add_violation(
            "stationary_no_pedestrian_scenario_ids_invalid",
            "stationary_no_pedestrian_scenario_ids must be a list of non-empty strings",
        )
    elif isinstance(stationary_no_pedestrian_scenario_ids, list) and len(
        no_pedestrian_scenarios
    ) != len(stationary_no_pedestrian_scenario_ids):
        add_violation(
            "stationary_no_pedestrian_scenario_ids_duplicate",
            "stationary_no_pedestrian_scenario_ids must not contain duplicates",
        )
    for scenario_id in no_pedestrian_scenarios:
        if scenario_id not in scenarios:
            add_violation(
                "unknown_stationary_no_pedestrian_scenario",
                f"Stationary no-pedestrian scenario {scenario_id!r} is not in the declared matrix",
                scenario_id=scenario_id,
            )

    normalized_goal_key = _nonempty_string(goal_key)
    normalized_stationary_key = _nonempty_string(stationary_key)
    normalized_aware_keys = _unique_strings(aware_keys)
    if normalized_goal_key is None:
        add_violation("goal_key_invalid", "goal_key must be a non-empty string")
        normalized_goal_key = ""
    if normalized_stationary_key is None:
        add_violation("stationary_key_invalid", "stationary_key must be a non-empty string")
        normalized_stationary_key = ""
    if normalized_aware_keys is None:
        normalized_aware_keys = []
        add_violation("aware_keys_invalid", "aware_keys must be a list of non-empty strings")
    elif isinstance(aware_keys, list) and len(normalized_aware_keys) != len(aware_keys):
        add_violation("aware_keys_duplicate", "aware_keys must not contain duplicates")

    configured_keys = [normalized_goal_key, normalized_stationary_key, *normalized_aware_keys]
    nonempty_configured = [key for key in configured_keys if key]
    if len(set(nonempty_configured)) != len(nonempty_configured):
        add_violation("arm_keys_duplicate", "goal, stationary, and aware arm keys must be distinct")

    normalized_thresholds = _validate_thresholds(thresholds, add_violation)
    expected_source_sha = _nonempty_string(source_sha)
    if expected_source_sha is None or expected_source_sha.lower() == "unknown":
        add_violation("source_sha_missing", "source_sha is required to evaluate oracle evidence")
        expected_source_sha = None
    elif not re.fullmatch(r"[0-9a-fA-F]{40}", expected_source_sha):
        add_violation(
            "source_sha_invalid", "source_sha must be a full 40-character hexadecimal SHA"
        )
        expected_source_sha = None

    if not isinstance(rows_by_arm, Mapping):
        add_violation("rows_by_arm_invalid", "rows_by_arm must be a mapping from arm keys to rows")
        row_mapping: Mapping[str, Any] = {}
    else:
        row_mapping = rows_by_arm

    configured_set = set(nonempty_configured)
    expected_algorithms: dict[str, str] = {}
    if not isinstance(expected_algorithms_by_arm, Mapping):
        add_violation(
            "expected_algorithms_missing",
            "expected_algorithms_by_arm is required to bind rows to configured arms",
        )
    else:
        for arm_key in nonempty_configured:
            algorithm = _nonempty_string(expected_algorithms_by_arm.get(arm_key))
            if algorithm is None:
                add_violation(
                    "expected_algorithm_missing",
                    f"Expected algorithm is missing for arm {arm_key!r}",
                    arm=arm_key,
                )
            else:
                expected_algorithms[arm_key] = algorithm
        for unexpected_key in sorted(
            str(key) for key in expected_algorithms_by_arm if str(key) not in configured_set
        ):
            add_violation(
                "unexpected_expected_algorithm",
                f"expected_algorithms_by_arm contains unconfigured arm {unexpected_key!r}",
                arm=unexpected_key,
            )

    for unexpected_key in sorted(str(key) for key in row_mapping if str(key) not in configured_set):
        add_violation(
            "unconfigured_arm_rows",
            f"rows_by_arm contains unconfigured arm {unexpected_key!r}",
            arm=unexpected_key,
        )

    expected_cells = [(scenario_id, seed) for scenario_id in scenarios for seed in normalized_seeds]
    expected_scenario_set = set(scenarios)
    expected_seed_set = set(normalized_seeds)
    role_by_arm: dict[str, str] = {}
    if normalized_goal_key:
        role_by_arm[normalized_goal_key] = "goal"
    if normalized_stationary_key:
        role_by_arm[normalized_stationary_key] = "stationary"
    for arm_key in normalized_aware_keys:
        role_by_arm.setdefault(arm_key, "aware")

    coverage: dict[str, dict[str, Any]] = {}
    valid_cells_by_arm: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    single_rows_by_arm: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    all_goal_failures: list[dict[str, Any]] = []
    execution_modes_by_arm: dict[str, Counter[str]] = {}

    for arm_key in dict.fromkeys(nonempty_configured):
        role = role_by_arm[arm_key]
        raw_rows = row_mapping.get(arm_key, [])
        if not isinstance(raw_rows, list):
            add_violation(
                "arm_rows_invalid",
                f"rows for arm {arm_key!r} must be a list",
                arm=arm_key,
            )
            rows: list[Any] = []
        else:
            rows = raw_rows

        cell_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        malformed_rows: list[dict[str, Any]] = []
        unexpected_rows: list[dict[str, Any]] = []
        mode_counts: Counter[str] = Counter()

        for row_index, raw_row in enumerate(rows):
            row_errors: list[str] = []
            if not isinstance(raw_row, Mapping):
                row_errors.append("row must be an object")
                malformed_rows.append({"row_index": row_index, "errors": row_errors})
                add_violation(
                    "row_invalid",
                    f"{arm_key} row {row_index} is invalid: row must be an object",
                    arm=arm_key,
                    row_index=row_index,
                )
                continue

            scenario_value = raw_row.get("scenario_id")
            seed_value = raw_row.get("seed")
            scenario_id = _nonempty_string(scenario_value)
            seed = (
                seed_value
                if isinstance(seed_value, int) and not isinstance(seed_value, bool)
                else None
            )
            algorithm = _nonempty_string(raw_row.get("algo"))
            expected_algorithm = expected_algorithms.get(arm_key)
            if expected_algorithm is not None:
                if algorithm is None:
                    row_errors.append("algo is missing or invalid")
                elif algorithm.lower() != expected_algorithm.lower():
                    row_errors.append(
                        f"algo {algorithm!r} does not match configured {expected_algorithm!r}"
                    )
            if scenario_id is None:
                row_errors.append("scenario_id is missing or invalid")
            if seed is None:
                row_errors.append("seed is missing or invalid")

            if scenario_id is not None and scenario_id not in expected_scenario_set:
                row_errors.append(f"scenario_id {scenario_id!r} is outside the declared matrix")
            if seed is not None and seed not in expected_seed_set:
                row_errors.append(f"seed {seed} is outside the declared seed set")

            outcome = _outcome_facts(raw_row, row_errors)
            mode = _execution_mode(raw_row, row_errors)
            mode_counts[mode or "missing"] += 1
            _validate_degraded_markers(raw_row, row_errors)
            _validate_source_provenance(
                raw_row,
                expected_source_sha,
                row_errors,
            )

            if role in {"goal", "aware"}:
                _validate_pedestrian_free_population(raw_row, row_errors)
                _validate_pedestrian_free_exposure(raw_row, row_errors)
            else:
                _validate_stationary_population(
                    raw_row,
                    row_errors,
                    no_pedestrians=scenario_id in no_pedestrian_scenarios,
                )

            metrics = raw_row.get("metrics")
            metrics = metrics if isinstance(metrics, Mapping) else None
            ped_collision_count = _finite_nonnegative_metric(
                metrics,
                "ped_collision_count",
            )
            if role == "stationary" and ped_collision_count is None:
                row_errors.append("metrics.ped_collision_count must be finite and non-negative")

            has_valid_identity = scenario_id is not None and seed is not None
            expected_identity = (
                has_valid_identity
                and scenario_id in expected_scenario_set
                and seed in expected_seed_set
            )
            facts = {
                "row": raw_row,
                "row_index": row_index,
                "scenario_id": scenario_id,
                "seed": seed,
                "errors": row_errors,
                "success": outcome["success"],
                "outcome_status": outcome["status"],
                "route_complete": outcome["route_complete"],
                "execution_mode": mode,
                "ped_collision_count": ped_collision_count,
            }

            if role == "goal" and outcome["is_failure"]:
                all_goal_failures.append(
                    {
                        "arm": arm_key,
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "status": outcome["status"],
                        "route_complete": outcome["route_complete"],
                        "execution_mode": mode,
                        "probe_exempt": scenario_id in probes,
                        "counted_for_threshold": False,
                        "row_index": row_index,
                        "errors": list(row_errors),
                    }
                )

            if expected_identity:
                cell_rows[(scenario_id, seed)].append(facts)
            else:
                entry = {
                    "row_index": row_index,
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "errors": list(row_errors),
                }
                if has_valid_identity:
                    unexpected_rows.append(entry)
                    add_violation(
                        "unexpected_row",
                        f"{arm_key} row {row_index} is outside the declared matrix",
                        arm=arm_key,
                        row_index=row_index,
                        scenario_id=scenario_id,
                        seed=seed,
                    )
                else:
                    malformed_rows.append(entry)
                if row_errors:
                    add_violation(
                        "row_invalid",
                        f"{arm_key} row {row_index} is invalid: {'; '.join(row_errors)}",
                        arm=arm_key,
                        row_index=row_index,
                        scenario_id=scenario_id,
                        seed=seed,
                    )
                    entry["errors"] = list(row_errors)

            if row_errors and expected_identity:
                add_violation(
                    "row_invalid",
                    f"{arm_key} row {row_index} is invalid: {'; '.join(row_errors)}",
                    arm=arm_key,
                    row_index=row_index,
                    scenario_id=scenario_id,
                    seed=seed,
                )

        missing_cells: list[dict[str, Any]] = []
        duplicate_cells: list[dict[str, Any]] = []
        invalid_cells: list[dict[str, Any]] = []
        valid_cells: dict[tuple[str, int], dict[str, Any]] = {}
        single_rows: dict[tuple[str, int], dict[str, Any]] = {}

        for cell in expected_cells:
            found = cell_rows.get(cell, [])
            cell_payload = {"scenario_id": cell[0], "seed": cell[1]}
            if not found:
                missing_cells.append(cell_payload)
                add_violation(
                    "missing_cell",
                    f"{arm_key} is missing scenario {cell[0]!r}, seed {cell[1]}",
                    arm=arm_key,
                    scenario_id=cell[0],
                    seed=cell[1],
                )
            elif len(found) > 1:
                duplicate_cells.append({**cell_payload, "row_count": len(found)})
                add_violation(
                    "duplicate_cell",
                    f"{arm_key} has {len(found)} rows for scenario {cell[0]!r}, seed {cell[1]}",
                    arm=arm_key,
                    scenario_id=cell[0],
                    seed=cell[1],
                    row_count=len(found),
                )
            else:
                fact = found[0]
                single_rows[cell] = fact
                if fact["errors"]:
                    invalid_cells.append(
                        {**cell_payload, "row_index": fact["row_index"], "errors": fact["errors"]}
                    )
                else:
                    valid_cells[cell] = fact

        coverage[arm_key] = {
            "role": role,
            "expected_cell_count": len(expected_cells),
            "received_row_count": len(rows),
            "valid_cell_count": len(valid_cells),
            "missing_cells": missing_cells,
            "duplicate_cells": duplicate_cells,
            "invalid_cells": invalid_cells,
            "malformed_rows": malformed_rows,
            "unexpected_rows": unexpected_rows,
            "execution_mode_counts": dict(sorted(mode_counts.items())),
        }
        valid_cells_by_arm[arm_key] = valid_cells
        single_rows_by_arm[arm_key] = single_rows
        execution_modes_by_arm[arm_key] = mode_counts

    # Count the complete, unique goal cells used by the threshold. The diagnostic
    # list above deliberately keeps failures from probes and invalid rows visible.
    goal_valid_cells = valid_cells_by_arm.get(normalized_goal_key, {})
    goal_failure_count = 0
    threshold_goal_failure_count = 0
    for (scenario_id, seed), fact in goal_valid_cells.items():
        if fact["success"]:
            continue
        goal_failure_count += 1
        is_probe = scenario_id in probes
        if not is_probe:
            threshold_goal_failure_count += 1
        for failure in all_goal_failures:
            if (
                failure["arm"] == normalized_goal_key
                and failure["scenario_id"] == scenario_id
                and failure["seed"] == seed
                and failure["row_index"] == fact["row_index"]
            ):
                failure["counted_for_threshold"] = not is_probe
                break

    # Recompute a complete failure count over valid cells so probes remain part
    # of the finding count but are only exempted in the threshold value.
    all_valid_goal_failure_count = sum(
        1 for fact in goal_valid_cells.values() if not fact["success"]
    )
    if normalized_thresholds["max_goal_failure_count"] is not None:
        if threshold_goal_failure_count > normalized_thresholds["max_goal_failure_count"]:
            add_violation(
                "goal_failure_threshold_exceeded",
                "Non-probe goal failures exceed max_goal_failure_count",
                observed=threshold_goal_failure_count,
                allowed=normalized_thresholds["max_goal_failure_count"],
            )

    stationary_unique_rows = valid_cells_by_arm.get(normalized_stationary_key, {})
    contact_values: list[float] = []
    stationary_expected_count = sum(
        1 for scenario_id, _seed in expected_cells if scenario_id not in no_pedestrian_scenarios
    )
    for (scenario_id, _seed), fact in stationary_unique_rows.items():
        if scenario_id in no_pedestrian_scenarios:
            continue
        if fact["ped_collision_count"] is not None:
            contact_values.append(fact["ped_collision_count"])
    contact_episode_count = sum(value > 0 for value in contact_values)
    contact_denominator = len(contact_values)
    stationary_contact_rate = (
        contact_episode_count / contact_denominator if contact_denominator else None
    )
    if contact_denominator < stationary_expected_count:
        add_violation(
            "stationary_contact_data_incomplete",
            "Stationary contact rate lacks typed metrics for every expected scenario/seed cell",
            observed=contact_denominator,
            expected=stationary_expected_count,
        )
    if stationary_expected_count == 0:
        add_violation(
            "stationary_contact_denominator_empty",
            "No pedestrian-present stationary cells remain for a contact-rate estimate",
        )
    max_contact_rate = normalized_thresholds["max_stationary_contact_rate"]
    if max_contact_rate is not None and stationary_contact_rate is not None:
        if stationary_contact_rate > max_contact_rate:
            add_violation(
                "stationary_contact_threshold_exceeded",
                "Stationary pedestrian contact rate exceeds max_stationary_contact_rate",
                observed=stationary_contact_rate,
                allowed=max_contact_rate,
                contact_episodes=contact_episode_count,
                denominator=contact_denominator,
            )

    dominance_regressions: list[dict[str, Any]] = []
    dominance_coverage: dict[str, dict[str, Any]] = {}
    goal_rows = valid_cells_by_arm.get(normalized_goal_key, {})
    for aware_key in normalized_aware_keys:
        aware_rows = valid_cells_by_arm.get(aware_key, {})
        missing_pairs: list[dict[str, Any]] = []
        compared_pairs = 0
        for scenario_id, seed in expected_cells:
            identity = (scenario_id, seed)
            goal_fact = goal_rows.get(identity)
            aware_fact = aware_rows.get(identity)
            if goal_fact is None or aware_fact is None:
                missing_pairs.append(
                    {
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "missing": [
                            arm
                            for arm, fact in (
                                (normalized_goal_key, goal_fact),
                                (aware_key, aware_fact),
                            )
                            if fact is None
                        ],
                    }
                )
                continue
            compared_pairs += 1
            if goal_fact["success"] and not aware_fact["success"]:
                dominance_regressions.append(
                    {
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "goal_arm": normalized_goal_key,
                        "aware_arm": aware_key,
                        "goal_status": goal_fact["outcome_status"],
                        "aware_status": aware_fact["outcome_status"],
                        "goal_route_complete": goal_fact["route_complete"],
                        "aware_route_complete": aware_fact["route_complete"],
                    }
                )
        dominance_coverage[aware_key] = {
            "expected_pair_count": len(expected_cells),
            "compared_pair_count": compared_pairs,
            "missing_pairs": missing_pairs,
        }
        if missing_pairs:
            add_violation(
                "dominance_pairs_incomplete",
                f"Dominance comparison for {aware_key!r} lacks valid paired rows",
                arm=aware_key,
                missing_pair_count=len(missing_pairs),
            )

    max_regressions = normalized_thresholds["max_dominance_regressions"]
    if max_regressions is not None and len(dominance_regressions) > max_regressions:
        add_violation(
            "dominance_threshold_exceeded",
            "Paired goal-to-aware regressions exceed max_dominance_regressions",
            observed=len(dominance_regressions),
            allowed=max_regressions,
        )

    failure_list = sorted(
        all_goal_failures,
        key=lambda row: (
            str(row.get("scenario_id") or ""),
            row.get("seed") if isinstance(row.get("seed"), int) else -1,
            row.get("row_index", -1),
        ),
    )
    dominance_regressions.sort(key=lambda row: (row["scenario_id"], row["seed"], row["aware_arm"]))

    reason_messages = list(dict.fromkeys(str(item["message"]) for item in violations))
    gate_status = "pass" if not violations else "fail"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "release_id": normalized_release_id or None,
        "source_sha": expected_source_sha,
        "matrix": {
            "scenario_ids": scenarios,
            "seeds": normalized_seeds,
            "probe_scenario_ids": probes,
            "stationary_no_pedestrian_scenario_ids": no_pedestrian_scenarios,
            "expected_cell_count_per_arm": len(expected_cells),
        },
        "arms": {
            "goal": normalized_goal_key or None,
            "stationary": normalized_stationary_key or None,
            "aware": normalized_aware_keys,
            "expected_algorithms": expected_algorithms,
        },
        "thresholds": normalized_thresholds,
        "coverage": coverage,
        "dominance_coverage": dominance_coverage,
        "goal_failures": failure_list,
        "dominance_regressions": dominance_regressions,
        "stationary_contact": {
            "arm": normalized_stationary_key or None,
            "contact_episode_count": contact_episode_count,
            "denominator": contact_denominator,
            "expected_denominator": stationary_expected_count,
            "excluded_no_pedestrian_scenario_ids": no_pedestrian_scenarios,
            "rate": stationary_contact_rate,
        },
        "stationary_no_pedestrian_scenarios": {
            "scenario_ids": no_pedestrian_scenarios,
            "excluded_cell_count": len(no_pedestrian_scenarios) * len(normalized_seeds),
        },
        "counts": {
            "goal_failure_count": len(failure_list),
            "validated_goal_failure_count": all_valid_goal_failure_count,
            "non_probe_goal_failure_count": threshold_goal_failure_count,
            "probe_goal_failure_count": sum(
                1 for failure in failure_list if failure["probe_exempt"]
            ),
            "reported_goal_failure_row_count": len(failure_list),
            "stationary_contact_episode_count": contact_episode_count,
            "stationary_contact_denominator": contact_denominator,
            "dominance_regression_count": len(dominance_regressions),
        },
        "violations": violations,
        "gate": {"status": gate_status, "reasons": reason_messages},
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise Markdown summary, retaining every failure and regression.

    Returns:
        Markdown report text.
    """
    report = report if isinstance(report, dict) else {}
    gate = report.get("gate") if isinstance(report.get("gate"), Mapping) else {}
    coverage = report.get("coverage") if isinstance(report.get("coverage"), Mapping) else {}
    contact = (
        report.get("stationary_contact")
        if isinstance(report.get("stationary_contact"), Mapping)
        else {}
    )
    lines = [
        "# Reference oracle report",
        "",
        f"- Schema: `{_md(report.get('schema_version', 'unknown'))}`",
        f"- Release: `{_md(report.get('release_id', 'unknown'))}`",
        f"- Source SHA: `{_md(report.get('source_sha', 'missing'))}`",
        f"- Gate: **{_md(str(gate.get('status', 'fail')).upper())}**",
        "",
        "## Coverage",
        "",
        "| Arm | Role | Valid cells | Expected cells | Missing | Duplicate | Invalid | Execution modes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm_key, arm_coverage in coverage.items():
        if not isinstance(arm_coverage, Mapping):
            continue
        mode_counts = arm_coverage.get("execution_mode_counts", {})
        mode_summary = (
            ", ".join(
                f"{_md(mode)}: {count}"
                for mode, count in sorted(mode_counts.items())
                if isinstance(count, int)
            )
            if isinstance(mode_counts, Mapping)
            else ""
        )
        lines.append(
            "| {arm} | {role} | {valid} | {expected} | {missing} | {duplicate} | {invalid} | {modes} |".format(
                arm=_md(arm_key),
                role=_md(arm_coverage.get("role", "unknown")),
                valid=arm_coverage.get("valid_cell_count", 0),
                expected=arm_coverage.get("expected_cell_count", 0),
                missing=len(arm_coverage.get("missing_cells", [])),
                duplicate=len(arm_coverage.get("duplicate_cells", [])),
                invalid=len(arm_coverage.get("invalid_cells", []))
                + len(arm_coverage.get("malformed_rows", []))
                + len(arm_coverage.get("unexpected_rows", [])),
                modes=mode_summary or "—",
            )
        )

    rate = contact.get("rate")
    rate_text = "unavailable" if rate is None else f"{rate:.6f}"
    thresholds = report.get("thresholds", {})
    thresholds = thresholds if isinstance(thresholds, Mapping) else {}
    lines.extend(
        [
            "",
            "## Threshold results",
            "",
            f"- Goal failures: {report.get('counts', {}).get('goal_failure_count', 0)} listed; "
            f"{report.get('counts', {}).get('non_probe_goal_failure_count', 0)} validated non-probe failures "
            f"against a limit of {thresholds.get('max_goal_failure_count', 'invalid')}.",
            f"- Stationary contact rate: {rate_text} "
            f"({contact.get('contact_episode_count', 0)}/{contact.get('denominator', 0)} episodes; "
            f"expected denominator {contact.get('expected_denominator', 0)}; "
            f"limit {thresholds.get('max_stationary_contact_rate', 'invalid')}).",
            f"- Paired dominance regressions: {report.get('counts', {}).get('dominance_regression_count', 0)} "
            f"against a limit of {thresholds.get('max_dominance_regressions', 'invalid')}.",
            f"- Stationary scenarios excluded from the contact denominator: "
            f"{', '.join(str(scenario) for scenario in contact.get('excluded_no_pedestrian_scenario_ids', [])) or 'none'}.",
            "",
            "## Goal failures",
            "",
        ]
    )
    failures = report.get("goal_failures", [])
    if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)) and failures:
        lines.append(
            "| Arm | Scenario | Seed | Status | Route complete | Probe | Counted | Details |"
        )
        lines.append("| --- | --- | ---: | --- | --- | --- | --- | --- |")
        for failure in failures:
            if not isinstance(failure, Mapping):
                continue
            errors = failure.get("errors", [])
            detail = "; ".join(str(error) for error in errors) if isinstance(errors, list) else ""
            lines.append(
                "| {arm} | {scenario} | {seed} | {status} | {route} | {probe} | {counted} | {details} |".format(
                    arm=_md(failure.get("arm", "")),
                    scenario=_md(failure.get("scenario_id", "")),
                    seed=_md(failure.get("seed", "")),
                    status=_md(failure.get("status", "unknown")),
                    route=_md(failure.get("route_complete", "unknown")),
                    probe="yes" if failure.get("probe_exempt") else "no",
                    counted="yes" if failure.get("counted_for_threshold") else "no",
                    details=_md(detail),
                )
            )
    else:
        lines.append("No goal failures were reported.")

    lines.extend(["", "## Dominance regressions", ""])
    regressions = report.get("dominance_regressions", [])
    if (
        isinstance(regressions, Sequence)
        and not isinstance(regressions, (str, bytes))
        and regressions
    ):
        lines.append("| Aware arm | Scenario | Seed | Goal status | Aware status |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for regression in regressions:
            if not isinstance(regression, Mapping):
                continue
            lines.append(
                "| {arm} | {scenario} | {seed} | {goal} | {aware} |".format(
                    arm=_md(regression.get("aware_arm", "")),
                    scenario=_md(regression.get("scenario_id", "")),
                    seed=_md(regression.get("seed", "")),
                    goal=_md(regression.get("goal_status", "unknown")),
                    aware=_md(regression.get("aware_status", "unknown")),
                )
            )
    else:
        lines.append("No paired dominance regressions were reported.")

    reasons = gate.get("reasons", [])
    if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) and reasons:
        lines.extend(["", "## Gate reasons", ""])
        lines.extend(f"- {_md(reason)}" for reason in reasons)

    return "\n".join(lines).rstrip() + "\n"


def _validate_thresholds(
    raw: Any,
    add_violation: Any,
) -> dict[str, int | float | None]:
    """Validate the fixed threshold vocabulary and normalize accepted numbers.

    Returns:
        Normalized threshold values, with ``None`` for absent or malformed values.
    """
    normalized: dict[str, int | float | None] = dict.fromkeys(sorted(_THRESHOLD_KEYS), None)
    if not isinstance(raw, Mapping):
        add_violation("thresholds_invalid", "thresholds must be a mapping")
        return normalized

    missing = _THRESHOLD_KEYS - set(raw)
    extra = set(raw) - _THRESHOLD_KEYS
    if missing:
        add_violation(
            "thresholds_missing",
            "Required thresholds are missing: " + ", ".join(sorted(str(key) for key in missing)),
            keys=sorted(str(key) for key in missing),
        )
    if extra:
        add_violation(
            "thresholds_unknown",
            "Unknown threshold keys: " + ", ".join(sorted(str(key) for key in extra)),
            keys=sorted(str(key) for key in extra),
        )

    for key in _THRESHOLD_KEYS & set(raw):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            add_violation(
                "threshold_invalid",
                f"Threshold {key!r} must be a finite number",
                key=key,
            )
            continue
        number = float(value)
        if not math.isfinite(number):
            add_violation(
                "threshold_invalid",
                f"Threshold {key!r} must be finite",
                key=key,
            )
            continue
        if key == "max_stationary_contact_rate":
            if number < 0 or number > 1:
                add_violation(
                    "threshold_invalid",
                    "max_stationary_contact_rate must be between 0 and 1 inclusive",
                    key=key,
                    value=value,
                )
                continue
            normalized[key] = number
        else:
            if number < 0 or not number.is_integer():
                add_violation(
                    "threshold_invalid",
                    f"{key} must be a non-negative integer",
                    key=key,
                    value=value,
                )
                continue
            normalized[key] = int(number)
    return normalized


def _unique_strings(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        normalized = _nonempty_string(item)
        if normalized is None:
            return None
        if normalized in result:
            continue
        result.append(normalized)
    return result


def _unique_seeds(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        if item in result:
            continue
        result.append(item)
    return result


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _outcome_facts(row: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping):
        errors.append("outcome must be an object")
        return {
            "status": _nonempty_string(row.get("status")),
            "route_complete": None,
            "success": False,
            "is_failure": False,
        }

    route_complete = outcome.get("route_complete")
    if not isinstance(route_complete, bool):
        errors.append("outcome.route_complete must be a boolean")
        route_complete = None
    status = _nonempty_string(row.get("status"))
    if status is None:
        errors.append("status is missing or invalid")
    else:
        status = status.lower()
        if status not in _ALLOWED_STATUSES:
            errors.append(f"status {status!r} is not a supported episode status")
    success = status == "success" and route_complete is True
    if route_complete is not None and status in _ALLOWED_STATUSES:
        if (status == "success") != route_complete:
            errors.append("status and outcome.route_complete disagree")
    is_failure = route_complete is not None and not success
    return {
        "status": status,
        "route_complete": route_complete,
        "success": success,
        "is_failure": is_failure,
    }


def _execution_mode(row: Mapping[str, Any], errors: list[str]) -> str | None:
    candidates: list[tuple[str, Any]] = []
    if "execution_mode" in row:
        candidates.append(("execution_mode", row.get("execution_mode")))
    metadata = row.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        if "execution_mode" in metadata:
            candidates.append(("algorithm_metadata.execution_mode", metadata.get("execution_mode")))
        kinematics = metadata.get("planner_kinematics")
        if isinstance(kinematics, Mapping) and "execution_mode" in kinematics:
            candidates.append(
                (
                    "algorithm_metadata.planner_kinematics.execution_mode",
                    kinematics.get("execution_mode"),
                )
            )
    normalized: list[tuple[str, str]] = []
    for path, value in candidates:
        mode = _nonempty_string(value)
        if mode is None:
            errors.append(f"{path} is missing or invalid")
            continue
        normalized.append((path, mode.lower()))
    if not normalized:
        errors.append("an explicit native or adapter execution_mode is required")
        return None
    values = {mode for _path, mode in normalized}
    if len(values) > 1:
        errors.append("execution_mode provenance fields disagree")
        return sorted(values)[0]
    mode = normalized[0][1]
    if mode not in _NATIVE_RUN_MODES:
        errors.append(f"execution_mode {mode!r} is not native-run evidence")
    return mode


def _validate_degraded_markers(row: Mapping[str, Any], errors: list[str]) -> None:  # noqa: C901
    markers: list[tuple[str, Any]] = []
    metadata = row.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        for key in ("status", "readiness_status", "availability_status"):
            if key in metadata:
                markers.append((f"algorithm_metadata.{key}", metadata.get(key)))
        runtime = metadata.get("planner_runtime")
        if isinstance(runtime, Mapping):
            for key in ("status", "execution_mode", "availability_status"):
                if key in runtime:
                    markers.append((f"algorithm_metadata.planner_runtime.{key}", runtime.get(key)))
    for key in ("readiness_status", "availability_status"):
        if key in row:
            markers.append((key, row.get(key)))
    for path, value in markers:
        marker = _nonempty_string(value)
        if marker is not None and marker.lower() in _DEGRADED_MARKERS:
            errors.append(f"{path}={marker!r} marks a fallback, degraded, or failed row")


def _validate_source_provenance(
    row: Mapping[str, Any],
    expected_source_sha: str | None,
    errors: list[str],
) -> None:
    values: list[tuple[str, str]] = []
    if "git_hash" in row:
        value = _nonempty_string(row.get("git_hash"))
        if value is None:
            errors.append("git_hash is present but invalid")
        else:
            values.append(("git_hash", value))
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping) and "git_hash" in provenance:
        value = _nonempty_string(provenance.get("git_hash"))
        if value is None:
            errors.append("provenance.git_hash is present but invalid")
        else:
            values.append(("provenance.git_hash", value))
    if not values or all(value.lower() == "unknown" for _path, value in values):
        errors.append("row source SHA is missing or unknown")
        return
    distinct = {value for _path, value in values}
    if len(distinct) != 1:
        errors.append("git_hash and provenance.git_hash disagree")
    if expected_source_sha is not None:
        for path, value in values:
            if value != expected_source_sha:
                errors.append(f"{path} does not match source_sha")


def _validate_pedestrian_free_population(row: Mapping[str, Any], errors: list[str]) -> None:
    scenario_params = row.get("scenario_params")
    if not isinstance(scenario_params, Mapping):
        errors.append("scenario_params must be an object for pedestrian-free rows")
        return
    if scenario_params.get("reference_population_mode") != "pedestrian_free_v1":
        errors.append("scenario_params.reference_population_mode must equal pedestrian_free_v1")
    simulation_config = scenario_params.get("simulation_config")
    if "simulation_config" in scenario_params and not isinstance(simulation_config, Mapping):
        errors.append("scenario_params.simulation_config must be an object when present")
        return
    if not isinstance(simulation_config, Mapping):
        errors.append("scenario_params.simulation_config is required for pedestrian-free rows")
        return
    if "instantiated_population_size" not in simulation_config:
        errors.append("scenario_params.simulation_config.instantiated_population_size is required")
    for field in ("population_size", "instantiated_population_size"):
        if field not in simulation_config:
            continue
        value = simulation_config.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"scenario_params.simulation_config.{field} must be numeric zero")
        elif not math.isfinite(float(value)) or value != 0:
            errors.append(f"scenario_params.simulation_config.{field} must equal zero")


def _validate_stationary_population(
    row: Mapping[str, Any],
    errors: list[str],
    *,
    no_pedestrians: bool,
) -> None:
    scenario_params = row.get("scenario_params")
    if not isinstance(scenario_params, Mapping):
        errors.append("scenario_params must be an object for stationary rows")
        return
    if scenario_params.get("reference_population_mode") == "pedestrian_free_v1":
        errors.append("stationary rows must retain the original pedestrian population")
    interaction_exposure = row.get("interaction_exposure")
    if not isinstance(interaction_exposure, Mapping):
        errors.append("interaction_exposure must be an object for stationary rows")
    else:
        expected_status = "not_derivable_no_pedestrians" if no_pedestrians else "computed"
        if interaction_exposure.get("interaction_exposure_status") != expected_status:
            errors.append(
                "stationary rows require interaction_exposure_status="
                f"{expected_status} for this scenario"
            )
    return


def _validate_pedestrian_free_exposure(row: Mapping[str, Any], errors: list[str]) -> None:
    """Require the native no-pedestrian trace status for a pedestrian-free arm."""
    interaction_exposure = row.get("interaction_exposure")
    if not isinstance(interaction_exposure, Mapping):
        errors.append("interaction_exposure must be an object for pedestrian-free rows")
    elif interaction_exposure.get("interaction_exposure_status") != "not_derivable_no_pedestrians":
        errors.append(
            "pedestrian-free rows require interaction_exposure_status=not_derivable_no_pedestrians"
        )


def _finite_nonnegative_metric(metrics: Mapping[str, Any] | None, key: str) -> float | None:
    if metrics is None or key not in metrics:
        return None
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        return None
    return normalized


def _md(value: Any) -> str:
    """Escape untrusted values for compact Markdown table cells.

    Returns:
        A single-line Markdown-safe representation.
    """
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|")
