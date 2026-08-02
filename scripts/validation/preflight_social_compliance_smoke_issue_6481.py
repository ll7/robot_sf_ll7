#!/usr/bin/env python3
"""Issue #6481 social-compliance preflight smoke receipt.

Run the frozen nine-row native/adapter smoke config and produce a compact
preflight receipt recording config/source SHA, expected/observed row
identities, execution modes, and metric-block schema version.

Evidence tier: smoke / preflight only.  No planner ranking, fairness,
ethics, safety, or real-world validity claim.

Execution-mode contract:
    Per the issue #691 benchmark fallback policy and
    ``control_action_latency_snqi.NATIVE_EXECUTION_MODES``, a *declared*
    adapter runs the planner through a benchmark-capable compatibility adapter
    and is grouped with native execution.  ``social_force`` and ``orca`` are
    inherently adapter planners (``supports_native_commands: False``) and
    cannot run native, so the preflight passes when every row is
    benchmark-capable (native or declared adapter) and no row is
    fallback/degraded/unavailable.

Usage:
    DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
      uv run python scripts/validation/preflight_social_compliance_smoke_issue_6481.py \
        --output-root output/benchmarks/issue_6481_preflight

Exit codes:
    0  preflight passed (all nine rows benchmark-capable [native or declared
       adapter] with schema-valid social block and no fallback/degraded rows)
    1  preflight failed (row-count, schema, or execution-mode mismatch)
    2  campaign execution error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.control_action_latency_snqi import NATIVE_EXECUTION_MODES
from robot_sf.benchmark.fallback_policy import resolve_execution_mode
from robot_sf.benchmark.social_compliance import (
    SOCIAL_COMPLIANCE_CLAIM_CLASS,
    SOCIAL_COMPLIANCE_SCHEMA_VERSION,
)
from robot_sf.evidence.writers import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_6481_social_compliance_preflight_smoke.yaml"
SCENARIO_MATRIX_PATH = ROOT / "configs/scenarios/issue_6481_social_compliance_preflight.yaml"
METRIC_CONTRACT_PATH = ROOT / "configs/benchmarks/social_compliance_metric_contract_v1.yaml"
EXPECTED_PLANNERS = ("goal", "social_force", "orca")
EXPECTED_SEEDS = (111, 112, 113)
EXPECTED_SCENARIO = "single_ped_crossing_orthogonal"
EXPECTED_ROW_COUNT = len(EXPECTED_PLANNERS) * len(EXPECTED_SEEDS)  # 9


def _load_metric_contract() -> dict[str, dict[str, str]]:
    """Load the canonical metric fields used to validate each emitted row."""
    payload = yaml.safe_load(METRIC_CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("social-compliance metric contract is not a mapping")
    if payload.get("schema_version") != SOCIAL_COMPLIANCE_SCHEMA_VERSION:
        raise ValueError("social-compliance metric contract schema version mismatch")
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, list):
        raise ValueError("social-compliance metric contract metrics are not a list")

    contract: dict[str, dict[str, str]] = {}
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, dict):
            raise ValueError("social-compliance metric contract contains a non-mapping metric")
        metric_id = raw_metric.get("id")
        if not isinstance(metric_id, str) or not metric_id.strip() or metric_id in contract:
            raise ValueError("social-compliance metric contract contains an invalid metric id")
        fields = {}
        for field in ("family", "units", "denominator", "claim_class"):
            value = raw_metric.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"social-compliance metric contract field is invalid: {field}")
            fields[field] = value
        contract[metric_id] = fields

    if not contract:
        raise ValueError("social-compliance metric contract has no metrics")
    return contract


EXPECTED_METRIC_CONTRACT = _load_metric_contract()
EXPECTED_METRIC_FAMILIES = {
    metric_id: fields["family"] for metric_id, fields in EXPECTED_METRIC_CONTRACT.items()
}
REQUIRED_FAMILIES = set(EXPECTED_METRIC_FAMILIES.values())
VALID_STATUSES = {"available", "unavailable", "not_applicable"}
VALID_EXECUTION_MODES = {
    "native",
    "adapter",
    "mixed",
    "fallback",
    "degraded",
    "unavailable",
    "unknown",
}
VALID_READINESS_STATUSES = {"native", "adapter", "fallback", "degraded", "unknown"}
VALID_AVAILABILITY_STATUSES = {
    "available",
    "not_available",
    "partial-failure",
    "failed",
    "unknown",
}


def _file_sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head_sha() -> str:
    """Return the current HEAD commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return result.stdout.strip()


def _find_last_json_object(stdout: str) -> Any | None:
    """Return the last complete JSON object embedded in noisy process output."""
    depth = 0
    end = len(stdout)
    for i in range(end - 1, -1, -1):
        if stdout[i] == "}":
            if depth == 0:
                end = i + 1
            depth += 1
        elif stdout[i] == "{":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stdout[i:end])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_campaign_stdout(stdout: str, returncode: int) -> dict[str, Any]:
    """Parse the camera-ready runner's final JSON object and retain its process status."""
    if not stdout:
        return {"_runner_returncode": returncode}
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = _find_last_json_object(stdout)
    if isinstance(payload, dict):
        return {**payload, "_runner_returncode": returncode}
    return {"_runner_returncode": returncode, "raw_stdout": stdout[-2000:]}


def _is_zero_exit_code(value: Any) -> bool:
    """Return whether a JSON exit-code field is the canonical integer zero."""
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite non-boolean number without coercion errors."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _is_valid_support_count(value: Any, *, require_positive: bool = False) -> bool:
    """Return whether a support count has the contract type and range."""
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if value <= 0 if require_positive else value < 0:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _normalized_text(value: Any, *, default: str = "unknown") -> str:
    """Return a lower-case non-empty status label or a fail-closed default."""
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    return normalized or default


def _resolve_row_execution_mode(record: dict[str, Any]) -> str:
    """Resolve execution mode from canonical episode metadata without native defaults."""
    for key in ("algorithm_metadata", "algorithm_metadata_contract"):
        payload = record.get(key)
        mode = _normalized_text(resolve_execution_mode(payload))
        if mode != "unknown":
            return mode if mode in VALID_EXECUTION_MODES else "unknown"

    mode = _normalized_text(resolve_execution_mode(record))
    return mode if mode in VALID_EXECUTION_MODES else "unknown"


def _as_bool(value: Any) -> bool:
    """Parse the campaign summary's boolean fields without truthy-string coercion."""
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() == "true"


def _load_campaign_planner_statuses(
    campaign_result: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Load canonical planner readiness fields from the campaign summary."""
    raw_rows = campaign_result.get("planner_rows")
    if not isinstance(raw_rows, list):
        summary_path = campaign_result.get("summary_json")
        if isinstance(summary_path, str) and summary_path:
            try:
                summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}, False
            raw_rows = summary.get("planner_rows")
    if not isinstance(raw_rows, list):
        return {}, False

    statuses: dict[str, dict[str, Any]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            return {}, False
        planner = _normalized_text(raw_row.get("planner_key") or raw_row.get("algo"))
        if planner in statuses or planner == "unknown":
            return {}, False
        statuses[planner] = {
            "execution_mode": _normalized_text(raw_row.get("execution_mode")),
            "readiness_status": _normalized_text(raw_row.get("readiness_status")),
            "availability_status": _normalized_text(raw_row.get("availability_status")),
            "benchmark_success": _as_bool(raw_row.get("benchmark_success")),
        }
    return statuses, True


def _load_campaign_summary_runs(campaign_result: dict[str, Any]) -> list[Any] | None:
    """Load campaign run entries from the result envelope or its summary JSON."""
    raw_runs = campaign_result.get("runs")
    if not isinstance(raw_runs, list):
        summary_path = campaign_result.get("summary_json")
        if isinstance(summary_path, str) and summary_path:
            try:
                summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            raw_runs = summary.get("runs")
    return raw_runs if isinstance(raw_runs, list) else None


def _run_planner_identifiers(raw_run: Any) -> tuple[str, str] | None:
    """Return the planner key and algorithm name from one campaign run entry."""
    if not isinstance(raw_run, dict):
        return None
    planner_payload = raw_run.get("planner")
    if isinstance(planner_payload, dict):
        planner = _normalized_text(planner_payload.get("key") or planner_payload.get("algo"))
        algo = _normalized_text(planner_payload.get("algo"))
    else:
        planner = _normalized_text(raw_run.get("planner_key") or raw_run.get("algo"))
        algo = _normalized_text(raw_run.get("algo"))
    if planner == "unknown":
        return None
    return planner, algo


def _load_campaign_social_aggregates(
    campaign_result: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Load social-compliance aggregate blocks from the canonical campaign summary."""
    raw_runs = _load_campaign_summary_runs(campaign_result)
    if raw_runs is None:
        return {}, False

    blocks: dict[str, dict[str, Any]] = {}
    for raw_run in raw_runs:
        identifiers = _run_planner_identifiers(raw_run)
        if identifiers is None:
            return {}, False
        planner, algo = identifiers
        if planner in blocks:
            return {}, False

        if not isinstance(raw_run, dict):
            return {}, False
        aggregates = raw_run.get("aggregates")
        if not isinstance(aggregates, dict):
            return {}, False
        aggregate_group = aggregates.get(planner)
        if not isinstance(aggregate_group, dict) and algo != "unknown":
            aggregate_group = aggregates.get(algo)
        if not isinstance(aggregate_group, dict):
            return {}, False
        social = aggregate_group.get("social_compliance")
        if not isinstance(social, dict):
            return {}, False
        blocks[planner] = social
    return blocks, True


def _aggregate_label(value: Any) -> str:
    """Normalize aggregate metadata labels using the aggregator's fail-visible default."""
    return value.strip() if isinstance(value, str) and value.strip() else "unknown"


def _aggregate_contract_is_ok(
    campaign_result: dict[str, Any],
    row_classifications: list[dict[str, Any]],
) -> bool:
    """Verify that canonical campaign aggregates preserve social-compliance metadata."""
    aggregate_blocks, aggregates_loaded = _load_campaign_social_aggregates(campaign_result)
    if not aggregates_loaded:
        return False

    rows_by_planner: dict[str, list[dict[str, Any]]] = {}
    for row in row_classifications:
        rows_by_planner.setdefault(row["planner"], []).append(row)
    if set(rows_by_planner) != set(aggregate_blocks):
        return False

    for planner, rows in rows_by_planner.items():
        block = aggregate_blocks[planner]
        if block.get("schema_version") != SOCIAL_COMPLIANCE_SCHEMA_VERSION:
            return False
        aggregate_metrics = block.get("metrics")
        if not isinstance(aggregate_metrics, dict):
            return False
        if set(aggregate_metrics) != set(EXPECTED_METRIC_CONTRACT):
            return False
        if not all(
            _aggregate_metric_contract_is_ok(metric_id, aggregate_metrics[metric_id], rows)
            for metric_id in EXPECTED_METRIC_CONTRACT
        ):
            return False
    return True


def _aggregate_metric_contract_is_ok(
    metric_id: str,
    aggregate_metric: Any,
    rows: list[dict[str, Any]],
) -> bool:
    """Verify status, support, denominator, reason, and reducer metadata for one metric."""
    if not isinstance(aggregate_metric, dict):
        return False
    statuses = [row["statuses"].get(metric_id, "unavailable") for row in rows]
    expected_status_counts = dict(Counter(str(status) for status in statuses))
    expected_support_count = sum(
        support
        for row, status in zip(rows, statuses, strict=True)
        for support in (row["support_counts"].get(metric_id, 0),)
        if status == "available" and _is_valid_support_count(support)
    )
    expected_denominators = dict(
        sorted(
            Counter(_aggregate_label(row["denominators"].get(metric_id)) for row in rows).items()
        )
    )
    expected_reasons = dict(
        sorted(
            Counter(
                _aggregate_label(row["unavailable_reasons"].get(metric_id))
                for row, status in zip(rows, statuses, strict=True)
                if status != "available"
            ).items()
        )
    )
    if aggregate_metric.get("status_counts") != expected_status_counts:
        return False
    if aggregate_metric.get("support_count") != int(expected_support_count):
        return False
    if aggregate_metric.get("denominators") != expected_denominators:
        return False
    if aggregate_metric.get("unavailable_reasons") != expected_reasons:
        return False

    reducers = {"mean", "median", "p95"}
    available_values = [
        values[metric_id]
        for row, status in zip(rows, statuses, strict=True)
        for values in (row.get("values", {}),)
        if status == "available"
        and _is_valid_support_count(row["support_counts"].get(metric_id), require_positive=True)
        and isinstance(values, dict)
        and metric_id in values
        and _is_finite_number(values[metric_id])
    ]
    if available_values:
        if not reducers.issubset(aggregate_metric):
            return False
        expected_reducers = {
            "mean": float(np.mean(available_values)),
            "median": float(np.median(available_values)),
            "p95": float(np.percentile(available_values, 95)),
        }
        return all(
            _is_finite_number(aggregate_metric.get(key))
            and math.isclose(
                float(aggregate_metric[key]),
                expected,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for key, expected in expected_reducers.items()
        )
    return not reducers & aggregate_metric.keys()


def _normalize_identity_text(value: Any) -> tuple[str, bool]:
    """Normalize a required planner or scenario identity field fail-closed."""
    if isinstance(value, str) and value.strip():
        return value.strip(), True
    return "unknown", False


def _normalize_identity_seed(value: Any) -> tuple[int | None, bool]:
    """Normalize a required seed without accepting bools or float-equal integers."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value, True
    return None, False


def _identity_sort_key(identity: tuple[Any, Any, Any]) -> tuple[str, str, str]:
    """Return a total-order key for possibly malformed identity tuples."""
    return tuple(str(value) for value in identity)


def _run_campaign(output_root: Path) -> dict[str, Any]:
    """Execute the camera-ready campaign and return the JSON result."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts/tools/run_camera_ready_benchmark.py"),
        "--config",
        str(CONFIG_PATH),
        "--output-root",
        str(output_root),
        "--skip-publication-bundle",
    ]
    env = {
        k: v for k, v in os.environ.items() if k not in ("DISPLAY", "MPLBACKEND", "SDL_VIDEODRIVER")
    }
    env["DISPLAY"] = ""
    env["MPLBACKEND"] = "Agg"
    env["SDL_VIDEODRIVER"] = "dummy"
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"_runner_returncode": 124, "error": "campaign timed out after 600 seconds"}
    return _parse_campaign_stdout(result.stdout.strip(), result.returncode)


def _read_episodes(campaign_root: Path) -> list[dict[str, Any]]:
    """Read all episode JSONL files from the campaign runs directory."""
    episodes: list[dict[str, Any]] = []
    runs_dir = campaign_root / "runs"
    if not runs_dir.is_dir():
        return episodes
    for jsonl_path in sorted(runs_dir.glob("*/episodes.jsonl")):
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"episode row is not a JSON object: {jsonl_path}")
                episodes.append(record)
    return episodes


def _classify_row(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one episode row for the preflight receipt."""
    scenario_params = record.get("scenario_params", {})
    if not isinstance(scenario_params, dict):
        scenario_params = {}
    raw_algo = scenario_params.get("algo") if "algo" in scenario_params else record.get("algo")
    planner, planner_identity_valid = _normalize_identity_text(raw_algo)
    scenario_id, scenario_identity_valid = _normalize_identity_text(record.get("scenario_id"))
    seed, seed_identity_valid = _normalize_identity_seed(record.get("seed"))
    metrics = record.get("metrics", {})
    social = metrics.get("social_compliance", {}) if isinstance(metrics, dict) else {}
    raw_social_metrics = social.get("metrics", {}) if isinstance(social, dict) else {}
    social_metrics = raw_social_metrics if isinstance(raw_social_metrics, dict) else {}

    schema_version = social.get("schema_version") if isinstance(social, dict) else None
    families_present = set()
    statuses: dict[str, str] = {}
    support_counts: dict[str, int] = {}
    denominators: dict[str, Any] = {}
    unavailable_reasons: dict[str, str | None] = {}
    values: dict[str, float] = {}
    schema_valid = (
        isinstance(social, dict)
        and social.get("claim_class") == SOCIAL_COMPLIANCE_CLAIM_CLASS
        and isinstance(social_metrics, dict)
        and set(social_metrics) == set(EXPECTED_METRIC_CONTRACT)
        and schema_version == SOCIAL_COMPLIANCE_SCHEMA_VERSION
    )
    for metric_id, row in social_metrics.items():
        if isinstance(row, dict):
            expected_contract = EXPECTED_METRIC_CONTRACT.get(metric_id)
            family = row.get("family", metric_id)
            if isinstance(family, str):
                families_present.add(family)
            status = row.get("status", "unknown")
            support_count = row.get("support_count", 0)
            denominator = row.get("denominator")
            reason = row.get("unavailable_reason")
            statuses[metric_id] = status
            support_counts[metric_id] = support_count
            denominators[metric_id] = denominator
            unavailable_reasons[metric_id] = reason

            row_valid = (
                expected_contract is not None
                and metric_id in EXPECTED_METRIC_FAMILIES
                and row.get("id") == metric_id
                and family == expected_contract["family"]
                and row.get("claim_class") == expected_contract["claim_class"]
                and row.get("units") == expected_contract["units"]
                and denominator == expected_contract["denominator"]
                and isinstance(status, str)
                and status in VALID_STATUSES
                and _is_valid_support_count(support_count)
            )
            if status == "available":
                raw_value = row.get("value")
                row_valid = (
                    row_valid
                    and _is_valid_support_count(support_count, require_positive=True)
                    and _is_finite_number(raw_value)
                )
                if row_valid:
                    values[metric_id] = float(raw_value)
            else:
                row_valid = (
                    row_valid
                    and support_count == 0
                    and isinstance(reason, str)
                    and bool(reason.strip())
                )
            schema_valid = schema_valid and row_valid
        else:
            schema_valid = False

    schema_valid = schema_valid and set(statuses) == set(EXPECTED_METRIC_FAMILIES)

    execution_mode = _resolve_row_execution_mode(record)

    return {
        "planner": planner,
        "scenario_id": scenario_id,
        "seed": seed,
        "identity_valid": (
            planner_identity_valid and scenario_identity_valid and seed_identity_valid
        ),
        "execution_mode": execution_mode,
        "social_compliance_schema_version": schema_version,
        "families_present": sorted(families_present),
        "statuses": statuses,
        "support_counts": support_counts,
        "denominators": denominators,
        "unavailable_reasons": unavailable_reasons,
        "values": values,
        "schema_valid": schema_valid,
        "all_families_present": REQUIRED_FAMILIES <= families_present,
    }


def _attach_campaign_status(
    row: dict[str, Any],
    planner_statuses: dict[str, dict[str, Any]],
) -> None:
    """Attach canonical planner status and detect episode/summary mode drift."""
    status = planner_statuses.get(row["planner"])
    if status is None:
        row.update(
            {
                "readiness_status": "unknown",
                "availability_status": "unknown",
                "benchmark_success": False,
                "execution_mode_consistent": False,
            }
        )
        return

    summary_mode = status["execution_mode"]
    row.update(
        {
            "readiness_status": (
                status["readiness_status"]
                if status["readiness_status"] in VALID_READINESS_STATUSES
                else "unknown"
            ),
            "availability_status": (
                status["availability_status"]
                if status["availability_status"] in VALID_AVAILABILITY_STATUSES
                else "unknown"
            ),
            "benchmark_success": status["benchmark_success"],
            "execution_mode_consistent": (
                row["execution_mode"] != "unknown"
                and summary_mode in VALID_EXECUTION_MODES
                and row["execution_mode"] == summary_mode
            ),
        }
    )


def _campaign_result_is_ok(campaign_result: dict[str, Any]) -> bool:
    """Return whether the child process and canonical campaign both succeeded."""
    return (
        bool(campaign_result.get("campaign_root"))
        and _is_zero_exit_code(campaign_result.get("_runner_returncode"))
        and campaign_result.get("campaign_execution_status") == "completed"
        and _is_zero_exit_code(campaign_result.get("exit_code"))
    )


def _receipt_aggregation_failure_messages(receipt: dict[str, Any]) -> list[str]:
    """Return aggregate-contract failure text for the CLI receipt."""
    if receipt["aggregation_contract_ok"]:
        return []
    return ["canonical aggregate lost social-compliance contract metadata"]


def _receipt_failure_messages(receipt: dict[str, Any]) -> list[str]:
    """Return concise fail-closed reasons for the CLI result."""
    failures: list[str] = []
    if not receipt["row_count_ok"]:
        failures.append(f"row count {receipt['observed_row_count']} != {EXPECTED_ROW_COUNT}")
    if not receipt["identities_ok"]:
        failures.append("row identity mismatch")
    if not receipt["all_benchmark_capable_execution"]:
        failures.append(
            "benchmark-capable execution contract not met (rows must be native or a declared "
            "adapter per issue #691; fallback/degraded/unavailable are excluded): "
            + ", ".join(
                f"{planner}={modes}" for planner, modes in receipt["execution_modes"].items()
            )
        )
    if not receipt["planner_summary_ok"]:
        failures.append("canonical planner status summary missing or malformed")
    if not receipt["all_execution_modes_recorded"]:
        failures.append("execution mode missing or unknown")
    if not receipt["execution_modes_consistent"]:
        failures.append("episode and campaign-summary execution modes differ")
    if not receipt["no_fallback_or_degraded"]:
        failures.append("fallback/degraded/unavailable planner status detected")
    if not receipt["all_schema_valid"]:
        failures.append("schema-invalid social_compliance block")
    if not receipt["all_families_present"]:
        failures.append("missing metric families")
    failures.extend(_receipt_aggregation_failure_messages(receipt))
    return failures


def build_receipt(
    campaign_result: dict[str, Any],
    episodes: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    """Build the compact preflight receipt."""
    row_classifications = [_classify_row(ep) for ep in episodes]
    planner_statuses, planner_summary_ok = _load_campaign_planner_statuses(campaign_result)
    planner_summary_ok = planner_summary_ok and set(planner_statuses) == set(EXPECTED_PLANNERS)
    for row in row_classifications:
        _attach_campaign_status(row, planner_statuses)
    observed_identities = sorted(
        {(r["planner"], r["scenario_id"], r["seed"]) for r in row_classifications},
        key=_identity_sort_key,
    )
    expected_identities = sorted(
        ((p, EXPECTED_SCENARIO, s) for p in EXPECTED_PLANNERS for s in EXPECTED_SEEDS),
        key=_identity_sort_key,
    )

    all_native = bool(row_classifications) and all(
        r["execution_mode"] == "native" for r in row_classifications
    )
    all_benchmark_capable_execution = bool(row_classifications) and all(
        r["execution_mode"] in NATIVE_EXECUTION_MODES for r in row_classifications
    )
    all_execution_modes_recorded = bool(row_classifications) and all(
        r["execution_mode"] != "unknown" for r in row_classifications
    )
    execution_modes_consistent = bool(row_classifications) and all(
        r["execution_mode_consistent"] for r in row_classifications
    )
    no_fallback_or_degraded = bool(row_classifications) and all(
        r["readiness_status"] in {"native", "adapter"}
        and r["availability_status"] == "available"
        and r["benchmark_success"]
        for r in row_classifications
    )
    all_schema_valid = all(r["schema_valid"] for r in row_classifications)
    all_families = all(r["all_families_present"] for r in row_classifications)
    aggregation_contract_ok = _aggregate_contract_is_ok(campaign_result, row_classifications)
    row_count_ok = len(episodes) == EXPECTED_ROW_COUNT
    identities_ok = (
        all(row["identity_valid"] for row in row_classifications)
        and observed_identities == expected_identities
    )
    campaign_returncode = campaign_result.get("_runner_returncode")
    campaign_exit_code = campaign_result.get("exit_code")
    campaign_ok = _campaign_result_is_ok(campaign_result)

    passed = (
        campaign_ok
        and row_count_ok
        and identities_ok
        and planner_summary_ok
        and all_execution_modes_recorded
        and execution_modes_consistent
        and all_benchmark_capable_execution
        and no_fallback_or_degraded
        and all_schema_valid
        and all_families
        and aggregation_contract_ok
    )

    return {
        "receipt_schema": "issue_6481_preflight_receipt.v1",
        "evidence_tier": "smoke_preflight",
        "claim_class": "diagnostic_proxy",
        "passed": passed,
        "config_sha256": _file_sha256(CONFIG_PATH),
        "scenario_matrix_sha256": _file_sha256(SCENARIO_MATRIX_PATH),
        "source_sha": _git_head_sha(),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "observed_row_count": len(episodes),
        "expected_identities": [list(t) for t in expected_identities],
        "observed_identities": [list(t) for t in observed_identities],
        "row_count_ok": row_count_ok,
        "identities_ok": identities_ok,
        "all_native": all_native,
        "all_benchmark_capable_execution": all_benchmark_capable_execution,
        "all_execution_modes_recorded": all_execution_modes_recorded,
        "execution_modes_consistent": execution_modes_consistent,
        "no_fallback_or_degraded": no_fallback_or_degraded,
        "planner_summary_ok": planner_summary_ok,
        "all_schema_valid": all_schema_valid,
        "all_families_present": all_families,
        "aggregation_contract_ok": aggregation_contract_ok,
        "campaign_ok": campaign_ok,
        "campaign_returncode": campaign_returncode,
        "campaign_exit_code": campaign_exit_code,
        "execution_modes": {
            planner: sorted(
                {r["execution_mode"] for r in row_classifications if r["planner"] == planner}
            )
            for planner in EXPECTED_PLANNERS
        },
        "social_compliance_schema_version": SOCIAL_COMPLIANCE_SCHEMA_VERSION,
        "rows": row_classifications,
        "campaign_result_status": campaign_result.get("campaign_execution_status", "unknown"),
        "local_output_root": {
            "status": "ignored_runtime_artifact",
            "note": (
                "Campaign rows were written under the --output-root location during "
                "generation. That path is intentionally omitted from the durable "
                "receipt because repository evidence must not point at ignored "
                "runtime output artifacts."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the preflight smoke and emit the receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output/benchmarks/issue_6481_preflight",
        help="Campaign output root directory.",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help="Optional path to write the receipt JSON (defaults to <output-root>/preflight_receipt.json).",
    )
    args = parser.parse_args(argv)

    output_root = args.output_root
    receipt_path = args.receipt_output or (output_root / "preflight_receipt.json")

    print(f"[issue_6481] Running social-compliance preflight smoke ({EXPECTED_ROW_COUNT} rows)...")
    campaign_result = _run_campaign(output_root)

    campaign_root_str = campaign_result.get("campaign_root", "")
    episodes: list[dict[str, Any]] = []
    episode_read_error: str | None = None
    if campaign_root_str:
        campaign_root = Path(campaign_root_str)
        try:
            episodes = _read_episodes(campaign_root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            episode_read_error = str(exc)

    campaign_ok = _campaign_result_is_ok(campaign_result)
    if not campaign_ok:
        print("[issue_6481] ERROR: campaign did not complete successfully", file=sys.stderr)
        print(json.dumps(campaign_result, indent=2), file=sys.stderr)
    elif episode_read_error is not None:
        print(
            f"[issue_6481] ERROR: could not read campaign episodes: {episode_read_error}",
            file=sys.stderr,
        )

    if not campaign_ok or episode_read_error is not None:
        receipt = build_receipt(campaign_result, episodes, output_root)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt_path, receipt)
        return 2

    receipt = build_receipt(campaign_result, episodes, output_root)

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))

    if receipt["passed"]:
        print(
            f"\n[issue_6481] PREFLIGHT PASSED: {EXPECTED_ROW_COUNT} benchmark-capable rows "
            "(native or declared adapter; zero fallback/degraded), schema-valid social blocks."
        )
        return 0

    failures = _receipt_failure_messages(receipt)
    print(f"\n[issue_6481] PREFLIGHT FAILED: {'; '.join(failures)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
