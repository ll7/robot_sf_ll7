"""Conservative failure-mechanism classification for paired benchmark episodes."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "failure_mechanism_classification.v1"
CLASSIFICATION_SOURCE = "failure_mechanism_classifier.v1"

FAILURE_MECHANISM_LABELS = (
    "time_budget_clean_relief",
    "exposure_enabled_completion",
    "safety_regressed_long_horizon",
    "persistent_low_progress_timeout",
    "scenario_contract_blocker",
    "unsupported_wait_then_go_hypothesis",
    "collision",
    "near_miss",
    "timeout_without_progress",
    "unavailable",
)
_REQUIRED_COVERAGE_LABELS = ("collision", "near_miss", "timeout_without_progress")
_BLOCKING_CERT_CLASSES = {
    "invalid",
    "geometrically_infeasible",
    "kinodynamically_infeasible",
    "dynamically_overconstrained",
}
_BLOCKING_ELIGIBILITY = {"excluded"}
_UNAVAILABLE_STATUSES = {
    "fallback",
    "degraded",
    "failed",
    "partial-failure",
    "partial_failure",
    "not_available",
    "unavailable",
}
_REQUIRED_PAIRED_METRICS = (
    "success",
    "collisions",
    "near_misses",
    "comfort_exposure",
    "min_distance",
    "time_to_goal_norm",
)


class FailureMechanismClassificationError(RobotSfError, ValueError):
    """Raised when failure-mechanism classification inputs are malformed."""


def classify_failure_mechanisms_from_jsonl(
    episodes_jsonl: str | Path | list[str | Path],
    *,
    scenario_certificates: str | Path | None = None,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    fixed_horizon: int = 100,
    long_horizon: int = 500,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Classify paired fixed/long-horizon episode records from JSONL input.

    Returns:
        ``failure_mechanism_classification.v1`` payload.
    """
    records = read_jsonl(episodes_jsonl)
    certs = load_scenario_certificates(scenario_certificates) if scenario_certificates else {}
    payload = classify_failure_mechanisms(
        records,
        scenario_certificates=certs,
        fixed_horizon=fixed_horizon,
        long_horizon=long_horizon,
        generated_at_utc=generated_at_utc,
        inputs={
            "episodes_jsonl": [str(path) for path in _as_paths(episodes_jsonl)],
            "scenario_certificates": str(scenario_certificates) if scenario_certificates else None,
            "fixed_horizon": int(fixed_horizon),
            "long_horizon": int(long_horizon),
        },
    )
    if output_json is not None:
        write_failure_mechanism_json(payload, output_json)
    if output_csv is not None:
        write_failure_mechanism_csv(payload, output_csv)
    return payload


def classify_failure_mechanisms(
    records: list[dict[str, Any]],
    *,
    scenario_certificates: dict[str, dict[str, Any]] | None = None,
    fixed_horizon: int = 100,
    long_horizon: int = 500,
    generated_at_utc: str | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify paired fixed/long-horizon episode records.

    The classifier is intentionally rule-based and fail-closed: labels that require pair evidence
    are emitted only when the matching scenario/planner/seed pair and required metrics are present.

    Returns:
        ``failure_mechanism_classification.v1`` payload.
    """
    if not records:
        raise FailureMechanismClassificationError("Cannot classify zero episode records")
    certs = scenario_certificates or {}
    rows: list[dict[str, Any]] = []
    for key, group in sorted(_group_records(records).items()):
        scenario_id, planner_id, seed = key
        row = _classify_group(
            scenario_id=scenario_id,
            planner_id=planner_id,
            seed=seed,
            records=group,
            scenario_certificate=certs.get(scenario_id),
            fixed_horizon=fixed_horizon,
            long_horizon=long_horizon,
        )
        rows.append(row)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification_source": CLASSIFICATION_SOURCE,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "inputs": inputs
        or {
            "fixed_horizon": int(fixed_horizon),
            "long_horizon": int(long_horizon),
        },
        "rows": rows,
        "coverage_axes": {"failure_modes": _coverage_axes(rows)},
        "missing_fields": _missing_fields(rows),
        "caveats": _payload_caveats(rows),
    }
    return validate_failure_mechanism_payload(payload)


def validate_failure_mechanism_payload(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
    """Validate the local classifier payload shape.

    Returns:
        Normalized payload.
    """
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise FailureMechanismClassificationError(
            f"Unsupported failure mechanism schema_version: {payload.get('schema_version')!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise FailureMechanismClassificationError("failure mechanism payload rows must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise FailureMechanismClassificationError(f"row {index} must be an object")
        label = row.get("label")
        if label not in FAILURE_MECHANISM_LABELS:
            raise FailureMechanismClassificationError(
                f"row {index} has unsupported label {label!r}"
            )
        confidence = _number_or_none(row.get("confidence"))
        if confidence is None or not 0.0 <= confidence <= 1.0:
            raise FailureMechanismClassificationError(f"row {index} has invalid confidence")
    failure_modes = _nested_value(payload, "coverage_axes", "failure_modes")
    if not isinstance(failure_modes, dict):
        raise FailureMechanismClassificationError("coverage_axes.failure_modes is required")
    for label in _REQUIRED_COVERAGE_LABELS:
        if not isinstance(failure_modes.get(label), dict):
            raise FailureMechanismClassificationError(
                f"coverage_axes.failure_modes.{label} is required"
            )
        if "classification_source" not in failure_modes[label]:
            raise FailureMechanismClassificationError(
                f"coverage_axes.failure_modes.{label}.classification_source is required"
            )
    return payload


def load_scenario_certificates(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load scenario certificates from JSON or JSONL.

    Returns:
        Mapping from scenario id to certificate payload.
    """
    cert_path = Path(path)
    text = cert_path.read_text(encoding="utf-8")
    payloads: list[dict[str, Any]]
    if cert_path.suffix == ".jsonl":
        payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("certificates"), list):
            payloads = payload["certificates"]
        elif isinstance(payload, list):
            payloads = payload
        elif isinstance(payload, dict):
            payloads = [payload]
        else:
            raise FailureMechanismClassificationError(f"Unsupported certificate payload: {path}")
    certs: dict[str, dict[str, Any]] = {}
    for item in payloads:
        if not isinstance(item, dict):
            continue
        scenario_id = item.get("scenario_id")
        if scenario_id is not None:
            certs[str(scenario_id)] = item
    return certs


def write_failure_mechanism_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Validate and write a JSON classifier payload.

    Returns:
        Output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = validate_failure_mechanism_payload(payload)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_failure_mechanism_csv(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write classifier rows as CSV.

    Returns:
        Output path.
    """
    report = validate_failure_mechanism_payload(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scenario_id",
        "planner_id",
        "seed",
        "fixed_episode_id",
        "long_episode_id",
        "label",
        "confidence",
        "required_inputs_present",
        "unavailable_reason",
        "caveats",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow(
                {
                    field: (
                        "; ".join(str(item) for item in row.get(field, []))
                        if field == "caveats"
                        else row.get(field)
                    )
                    for field in fields
                }
            )
    return path


def _classify_group(  # noqa: C901
    *,
    scenario_id: str,
    planner_id: str,
    seed: str,
    records: list[dict[str, Any]],
    scenario_certificate: dict[str, Any] | None,
    fixed_horizon: int,
    long_horizon: int,
) -> dict[str, Any]:
    """Classify one scenario/planner/seed group.

    Returns:
        Normalized classifier row.
    """
    fixed = _select_horizon_record(records, fixed_horizon)
    long = _select_horizon_record(records, long_horizon)
    base = {
        "episode_id": _episode_id(long or fixed),
        "scenario_id": scenario_id,
        "planner_id": planner_id,
        "seed": seed,
        "fixed_episode_id": _episode_id(fixed),
        "long_episode_id": _episode_id(long),
        "classification_source": CLASSIFICATION_SOURCE,
    }
    unavailable_reason = _unavailable_status_reason(records)
    if unavailable_reason is not None:
        return _row(
            base,
            "unavailable",
            confidence=1.0,
            required_inputs_present=False,
            unavailable_reason=unavailable_reason,
            caveats=["Unavailable or degraded execution is not mechanism evidence."],
        )
    cert_reason = _scenario_certificate_blocker(scenario_certificate)
    if cert_reason is not None:
        return _row(
            base,
            "scenario_contract_blocker",
            confidence=0.95,
            required_inputs_present=True,
            evidence_inputs={"scenario_certificate": scenario_certificate},
            caveats=[cert_reason],
        )
    if fixed is None or long is None:
        return _row(
            base,
            "unavailable",
            confidence=1.0,
            required_inputs_present=False,
            unavailable_reason="missing_fixed_or_long_horizon_pair",
            caveats=["Paired labels require matching fixed and long horizon records."],
        )
    missing_metrics = _missing_required_paired_metrics(fixed, long)
    if missing_metrics:
        return _row(
            base,
            "unavailable",
            confidence=1.0,
            required_inputs_present=False,
            unavailable_reason="missing_required_metrics",
            evidence_inputs={"missing_metrics": missing_metrics},
            caveats=["Required paired metrics were absent; no mechanism label inferred."],
        )
    fixed_facts = _episode_facts(fixed)
    long_facts = _episode_facts(long)
    evidence_inputs = {
        "fixed": fixed_facts,
        "long": long_facts,
    }
    if not fixed_facts["collision"] and long_facts["collision"]:
        return _row(
            base,
            "safety_regressed_long_horizon",
            confidence=0.9,
            evidence_inputs=evidence_inputs,
        )
    if (
        fixed_facts["unfinished"]
        and long_facts["success"]
        and _long_run_cleaner_or_equal(fixed_facts, long_facts)
    ):
        return _row(
            base,
            "time_budget_clean_relief",
            confidence=0.9,
            evidence_inputs=evidence_inputs,
        )
    if (
        fixed_facts["unfinished"]
        and long_facts["success"]
        and _long_run_increases_exposure(fixed_facts, long_facts)
    ):
        return _row(
            base,
            "exposure_enabled_completion",
            confidence=0.85,
            evidence_inputs=evidence_inputs,
            caveats=["Completion carries additional exposure or clearance pressure."],
        )
    if (
        fixed_facts["timeout"]
        and long_facts["timeout"]
        and not fixed_facts["collision"]
        and not long_facts["collision"]
        and not fixed_facts["near_miss"]
        and not long_facts["near_miss"]
        and fixed_facts["low_progress"]
        and long_facts["low_progress"]
    ):
        return _row(
            base,
            "persistent_low_progress_timeout",
            confidence=0.85,
            evidence_inputs=evidence_inputs,
        )
    if fixed_facts["collision"] or long_facts["collision"]:
        return _row(base, "collision", confidence=0.8, evidence_inputs=evidence_inputs)
    if fixed_facts["near_miss"] or long_facts["near_miss"]:
        return _row(base, "near_miss", confidence=0.75, evidence_inputs=evidence_inputs)
    if fixed_facts["timeout"] or long_facts["timeout"]:
        return _row(
            base,
            "timeout_without_progress",
            confidence=0.7,
            evidence_inputs=evidence_inputs,
        )
    return _row(
        base,
        "unavailable",
        confidence=1.0,
        required_inputs_present=False,
        unavailable_reason="no_failure_mechanism_observed",
        evidence_inputs=evidence_inputs,
        caveats=["No failure mechanism was observed in the paired records."],
    )


def _row(
    base: dict[str, Any],
    label: str,
    *,
    confidence: float,
    required_inputs_present: bool = True,
    unavailable_reason: str | None = None,
    evidence_inputs: dict[str, Any] | None = None,
    caveats: list[str] | None = None,
) -> dict[str, Any]:
    """Build one normalized classifier row.

    Returns:
        Classifier row payload.
    """
    return {
        **base,
        "label": label,
        "confidence": float(confidence),
        "evidence_inputs": evidence_inputs or {},
        "required_inputs_present": bool(required_inputs_present),
        "unavailable_reason": unavailable_reason,
        "caveats": caveats or [],
    }


def _coverage_axes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build stress-coverage-compatible failure mode counts.

    Returns:
        Failure-mode coverage mapping.
    """
    total = len(rows)
    counts = Counter(str(row["label"]) for row in rows if row.get("label") != "unavailable")
    scenario_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = str(row.get("label"))
        if label == "unavailable":
            continue
        scenario_ids[label].add(str(row.get("scenario_id", "unknown")))
    labels = sorted(
        set(FAILURE_MECHANISM_LABELS) - {"unavailable"} | set(_REQUIRED_COVERAGE_LABELS)
    )
    return {
        label: {
            "observed_episodes": int(counts.get(label, 0)),
            "observed_fraction": float(counts.get(label, 0) / total) if total else 0.0,
            "scenario_ids": sorted(scenario_ids.get(label, set())),
            "classification_source": CLASSIFICATION_SOURCE,
        }
        for label in labels
    }


def _missing_fields(rows: list[dict[str, Any]]) -> list[str]:
    """Summarize row-level unavailable causes.

    Returns:
        Sorted unavailable reason list.
    """
    missing = set()
    for row in rows:
        reason = row.get("unavailable_reason")
        if reason:
            missing.add(str(reason))
    return sorted(missing)


def _payload_caveats(rows: list[dict[str, Any]]) -> list[str]:
    """Return payload-level interpretation caveats."""
    caveats = [
        "Classifier labels are rule-based diagnostics, not causal proof without trace review.",
        "Unavailable rows are excluded from observed failure-mode coverage counts.",
    ]
    if any(row.get("label") == "unsupported_wait_then_go_hypothesis" for row in rows):
        caveats.append("Wait-then-go remains unsupported without trace or video evidence.")
    return caveats


def _group_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Group records by scenario, planner, and seed.

    Returns:
        Mapping from grouping key to records.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scenario_id = str(record.get("scenario_id", "unknown"))
        planner_id = str(
            record.get("planner_id")
            or record.get("planner")
            or record.get("algo")
            or record.get("algorithm")
            or "unknown"
        )
        seed = str(record.get("seed", record.get("episode_seed", "unknown")))
        groups[(scenario_id, planner_id, seed)].append(record)
    return groups


def _select_horizon_record(records: list[dict[str, Any]], horizon: int) -> dict[str, Any] | None:
    """Select the first record matching a horizon.

    Returns:
        Matching record or None.
    """
    for record in records:
        value = (
            record.get("horizon")
            or record.get("horizon_steps")
            or _nested_value(record, "scenario_params", "horizon_steps")
        )
        number = _number_or_none(value)
        if number is not None and int(number) == int(horizon):
            return record
    return None


def _episode_facts(record: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized classification facts from one episode record.

    Returns:
        Normalized episode facts.
    """
    success = _success(record)
    timeout = _timeout(record, success=success)
    collisions = _metric(record, "collisions")
    near_misses = _metric(record, "near_misses", "near_miss", "near_miss_count")
    comfort_exposure = _metric(record, "comfort_exposure")
    min_distance = _metric(record, "min_distance", "min_distance_m")
    time_to_goal_norm = _metric(record, "time_to_goal_norm")
    progress = _metric(record, "progress", "route_progress", "progress_ratio")
    collision = bool(
        (collisions is not None and collisions > 0.0)
        or _truthy(_nested_value(record, "outcome", "collision_event"))
        or _truthy(_nested_value(record, "outcome", "collision"))
    )
    low_progress = bool(
        (progress is not None and progress <= 0.25)
        or (time_to_goal_norm is not None and time_to_goal_norm >= 0.95)
    )
    return {
        "success": success,
        "unfinished": not success,
        "timeout": timeout,
        "collision": collision,
        "near_miss": bool(near_misses is not None and near_misses > 0.0),
        "collisions": collisions,
        "near_misses": near_misses,
        "comfort_exposure": comfort_exposure,
        "min_distance": min_distance,
        "time_to_goal_norm": time_to_goal_norm,
        "progress": progress,
        "low_progress": low_progress,
        "termination_reason": record.get("termination_reason"),
    }


def _success(record: dict[str, Any]) -> bool:
    """Return true when explicit success/route completion evidence is present."""
    success_metric = _success_metric(record)
    if success_metric is not None:
        return success_metric > 0.0
    return bool(
        _truthy(_nested_value(record, "outcome", "route_complete"))
        or _truthy(_nested_value(record, "outcome", "success"))
    )


def _timeout(record: dict[str, Any], *, success: bool) -> bool:
    """Return true when the episode timed out without completion."""
    reason = str(record.get("termination_reason", "")).strip().lower()
    timed_out = bool(
        _truthy(_nested_value(record, "outcome", "timeout"))
        or _truthy(_nested_value(record, "outcome", "timeout_event"))
        or reason in {"timeout", "max_steps", "time_limit", "truncated"}
    )
    return timed_out and not success


def _long_run_cleaner_or_equal(fixed: dict[str, Any], long: dict[str, Any]) -> bool:
    """Return true when long-horizon completion does not add safety/comfort pressure."""
    return bool(
        not long["collision"]
        and not long["near_miss"]
        and float(long["comfort_exposure"]) <= float(fixed["comfort_exposure"]) + 1e-9
        and float(long["min_distance"]) >= float(fixed["min_distance"]) - 1e-9
    )


def _long_run_increases_exposure(fixed: dict[str, Any], long: dict[str, Any]) -> bool:
    """Return true when long-horizon completion carries additional exposure pressure."""
    return bool(
        long["near_miss"]
        or float(long["comfort_exposure"]) > float(fixed["comfort_exposure"]) + 1e-9
        or float(long["min_distance"]) < float(fixed["min_distance"]) - 1e-9
    )


def _missing_required_paired_metrics(
    fixed: dict[str, Any],
    long: dict[str, Any],
) -> list[str]:
    """Return required metric names missing from either paired record."""
    missing: list[str] = []
    for prefix, record in (("fixed", fixed), ("long", long)):
        for metric_name in _REQUIRED_PAIRED_METRICS:
            if metric_name == "success":
                if _success_metric(record) is None:
                    missing.append(f"{prefix}.metrics.success")
                continue
            aliases = (
                (metric_name, "min_distance_m") if metric_name == "min_distance" else (metric_name,)
            )
            if _metric(record, *aliases) is None:
                missing.append(f"{prefix}.metrics.{metric_name}")
    return missing


def _scenario_certificate_blocker(certificate: dict[str, Any] | None) -> str | None:
    """Return a certification blocker caveat when one is present."""
    if not certificate:
        return None
    classification = str(certificate.get("classification", "")).strip().lower()
    eligibility = str(certificate.get("benchmark_eligibility", "")).strip().lower()
    if classification in _BLOCKING_CERT_CLASSES or eligibility in _BLOCKING_ELIGIBILITY:
        return (
            "Scenario certificate blocks planner-mechanism attribution: "
            f"classification={classification or 'unknown'}, "
            f"benchmark_eligibility={eligibility or 'unknown'}."
        )
    return None


def _unavailable_status_reason(records: list[dict[str, Any]]) -> str | None:
    """Return unavailable reason for fallback, degraded, failed, or unavailable rows."""
    for record in records:
        nested_status = _nested_value(record, "algorithm_metadata", "status")
        if nested_status is not None:
            text = str(nested_status).strip().lower()
            if text in _UNAVAILABLE_STATUSES:
                return f"unavailable_execution_status:algorithm_metadata.status={text}"
        for key in (
            "execution_mode",
            "availability_status",
            "status",
            "planner_mode",
            "benchmark_status",
        ):
            value = record.get(key)
            if value is None:
                continue
            text = str(value).strip().lower()
            if text in _UNAVAILABLE_STATUSES:
                return f"unavailable_execution_status:{key}={text}"
    return None


def _success_metric(record: dict[str, Any]) -> float | None:
    """Read canonical success metrics, including boolean schema values.

    Returns:
        Success value as 0.0/1.0 or a finite numeric value.
    """
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or "success" not in metrics:
        return None
    value = metrics.get("success")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _number_or_none(value)


def _metric(record: dict[str, Any], *names: str) -> float | None:
    """Read the first finite numeric metric for the provided names.

    Returns:
        Finite metric value or None.
    """
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    for name in names:
        value = _number_or_none(metrics.get(name))
        if value is not None:
            return value
    return None


def _number_or_none(value: Any) -> float | None:
    """Normalize finite numeric values.

    Returns:
        Finite float value or None.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    """Normalize common truthy scalar values.

    Returns:
        Boolean interpretation of a scalar value.
    """
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    """Read a nested mapping value.

    Returns:
        Nested value or None.
    """
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _episode_id(record: dict[str, Any] | None) -> str | None:
    """Return a stable episode id if present."""
    if not record:
        return None
    value = record.get("episode_id") or record.get("id")
    return str(value) if value is not None else None


def _as_paths(paths: str | Path | list[str | Path]) -> list[Path]:
    """Normalize one or many paths.

    Returns:
        List of normalized paths.
    """
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


__all__ = [
    "CLASSIFICATION_SOURCE",
    "FAILURE_MECHANISM_LABELS",
    "SCHEMA_VERSION",
    "FailureMechanismClassificationError",
    "classify_failure_mechanisms",
    "classify_failure_mechanisms_from_jsonl",
    "load_scenario_certificates",
    "validate_failure_mechanism_payload",
    "write_failure_mechanism_csv",
    "write_failure_mechanism_json",
]
