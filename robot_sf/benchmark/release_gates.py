"""Release-gate reporting over existing benchmark metric rows.

This module evaluates predeclared safety and comfort gates against already
aggregated benchmark rows. It does not collect new metrics or certify that a
provisional threshold is release-approved.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file
from robot_sf.errors import RobotSfError

RELEASE_GATE_REPORT_SCHEMA_VERSION = "benchmark_release_gate_report.v1"
RELEASE_GATE_SPEC_SCHEMA_VERSION = "benchmark_release_gate_spec.v1"
GATE_STATUSES = {"pass", "fail", "not_evaluable"}
_VALID_DIRECTIONS = {"max", "min"}
_VALID_CATEGORIES = {"safety", "comfort"}
_DEFAULT_PLANNER_KEYS = (
    "planner_key",
    "planner",
    "algorithm",
    "algo",
    "policy_id",
    "model_id",
)
_DEFAULT_FAMILY_KEYS = (
    "scenario_family",
    "family",
    "scenario_id",
    "scenario",
    "scenario_params.scenario_family",
    "scenario_params.family",
)


class ReleaseGateSpecError(RobotSfError, ValueError):
    """Raised when release-gate YAML cannot be evaluated safely."""


@dataclass(frozen=True)
class GateSpec:
    """One configured release gate for an existing metric key."""

    gate_id: str
    metric: str
    threshold: float
    direction: str
    category: str
    provenance: str
    required: bool = True
    scope: Mapping[str, Any] | None = None
    description: str | None = None


def load_release_gate_spec(path: str | Path) -> list[GateSpec]:
    """Load and validate release gates from YAML.

    Returns:
        Validated release gate specifications.
    """

    spec_path = Path(path)
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ReleaseGateSpecError("release gate spec must be a YAML mapping")
    schema_version = payload.get("schema_version")
    if schema_version not in (None, RELEASE_GATE_SPEC_SCHEMA_VERSION):
        raise ReleaseGateSpecError(
            f"unsupported release gate spec schema_version: {schema_version!r}"
        )
    gates_raw = payload.get("gates")
    if not isinstance(gates_raw, Sequence) or isinstance(gates_raw, str) or not gates_raw:
        raise ReleaseGateSpecError("release gate spec must contain non-empty gates list")

    seen: set[str] = set()
    gates: list[GateSpec] = []
    for index, item in enumerate(gates_raw):
        if not isinstance(item, Mapping):
            raise ReleaseGateSpecError(f"gates[{index}] must be a mapping")
        gate = _parse_gate(item, index=index)
        if gate.gate_id in seen:
            raise ReleaseGateSpecError(f"duplicate release gate id: {gate.gate_id}")
        seen.add(gate.gate_id)
        gates.append(gate)
    return gates


def evaluate_release_gates(
    rows: Sequence[Mapping[str, Any]],
    gates: Sequence[GateSpec],
) -> dict[str, Any]:
    """Evaluate safety and comfort gates for each planner x scenario family group.

    Unknown or missing metric keys produce ``not_evaluable`` gate rows instead of a
    pass. A group passes only when both safety and comfort categories pass.

    Returns:
        Report payload containing matrix rows and per-gate detail.
    """

    if not gates:
        raise ReleaseGateSpecError("at least one gate required")
    normalized_rows = [_normalize_input_row(row) for row in rows]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        groups[(str(row["planner_key"]), str(row["scenario_family"]))].append(row)

    results = []
    matrix_rows = []
    for planner_key, scenario_family in sorted(groups):
        group_rows = groups[(planner_key, scenario_family)]
        gate_results = [
            _evaluate_gate_for_group(gate, group_rows, scenario_family=scenario_family)
            for gate in gates
            if _gate_applies_to_family(gate, scenario_family)
        ]
        category_statuses = {
            category: _category_status(gate_results, category=category)
            for category in sorted(_VALID_CATEGORIES)
        }
        overall_status = _overall_status(category_statuses)
        failed_gate_ids = [
            result["gate_id"] for result in gate_results if result["status"] == "fail"
        ]
        not_evaluable_gate_ids = [
            result["gate_id"] for result in gate_results if result["status"] == "not_evaluable"
        ]
        result_row = {
            "planner_key": planner_key,
            "scenario_family": scenario_family,
            "safety_status": category_statuses["safety"],
            "comfort_status": category_statuses["comfort"],
            "overall_status": overall_status,
            "failed_gate_ids": failed_gate_ids,
            "not_evaluable_gate_ids": not_evaluable_gate_ids,
            "gate_results": gate_results,
        }
        results.append(result_row)
        matrix_rows.append(
            {
                "planner_key": planner_key,
                "scenario_family": scenario_family,
                "safety_status": category_statuses["safety"],
                "comfort_status": category_statuses["comfort"],
                "overall_status": overall_status,
                "failed_gate_ids": ";".join(failed_gate_ids),
                "not_evaluable_gate_ids": ";".join(not_evaluable_gate_ids),
            }
        )

    gate_payload = [
        {
            "id": gate.gate_id,
            "metric": gate.metric,
            "threshold": gate.threshold,
            "direction": gate.direction,
            "category": gate.category,
            "required": gate.required,
            "provenance": gate.provenance,
            "scope": dict(gate.scope or {}),
        }
        for gate in gates
    ]
    return {
        "schema_version": RELEASE_GATE_REPORT_SCHEMA_VERSION,
        "claim_boundary": (
            "Release gates are evaluated over existing benchmark rows only. "
            "Provisional thresholds are not certification or paper-grade approval."
        ),
        "status_counts": _count_statuses(matrix_rows),
        "gate_spec": {
            "schema_version": RELEASE_GATE_SPEC_SCHEMA_VERSION,
            "gates": gate_payload,
        },
        "matrix_rows": matrix_rows,
        "results": results,
    }


def build_release_gate_report(
    rows: Sequence[Mapping[str, Any]],
    gates: Sequence[GateSpec],
    *,
    input_path: Path | None = None,
    gate_spec_path: Path | None = None,
    command: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a provenance-bearing release-gate report payload.

    Returns:
        Release-gate report with source and gate-spec provenance.
    """

    report = evaluate_release_gates(rows, gates)
    report["generated_at_utc"] = generated_at_utc or datetime.now(UTC).isoformat()
    report["provenance"] = {
        "input": _path_provenance(input_path),
        "gate_spec": _path_provenance(gate_spec_path),
        "command": command,
    }
    return report


def load_metric_rows(path: str | Path) -> list[Mapping[str, Any]]:
    """Load existing benchmark summary rows from a JSON artifact.

    Returns:
        Existing metric rows from a supported JSON container.
    """

    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = _rows_from_mapping(payload)
    else:
        raise ReleaseGateSpecError("input JSON must be a list or mapping containing rows")
    if not isinstance(rows, list):
        raise ReleaseGateSpecError("input JSON row container must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ReleaseGateSpecError(f"input row {index} must be a mapping")
    return rows


def write_release_gate_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write release-gate JSON plus optional matrix CSV and Markdown.

    Returns:
        Mapping from artifact kind to written path.
    """

    paths: dict[str, Path] = {}
    out_json = Path(json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["json"] = out_json
    if csv_path is not None:
        out_csv = Path(csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text(format_release_gate_csv(report), encoding="utf-8")
        paths["csv"] = out_csv
    if markdown_path is not None:
        out_md = Path(markdown_path)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(format_release_gate_markdown(report), encoding="utf-8")
        paths["markdown"] = out_md
    return paths


def format_release_gate_csv(report: Mapping[str, Any]) -> str:
    """Format planner x family release-gate matrix as CSV.

    Returns:
        CSV text for the report matrix rows.
    """

    rows = report.get("matrix_rows", [])
    columns = (
        "planner_key",
        "scenario_family",
        "safety_status",
        "comfort_status",
        "overall_status",
        "failed_gate_ids",
        "not_evaluable_gate_ids",
    )
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def format_release_gate_markdown(report: Mapping[str, Any]) -> str:
    """Format release-gate report as a compact Markdown review artifact.

    Returns:
        Markdown report text.
    """

    lines = [
        "# Paired Safety And Comfort Release-Gate Matrix",
        "",
        str(report.get("claim_boundary", "")),
        "",
        "## Pass/Fail Matrix",
        "",
        "| Planner | Scenario family | Safety | Comfort | Overall | Failed gates | Not evaluable |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("matrix_rows", []):
        lines.append(
            "| {planner_key} | {scenario_family} | {safety_status} | {comfort_status} | "
            "{overall_status} | {failed_gate_ids} | {not_evaluable_gate_ids} |".format(
                **{key: _markdown_cell(row.get(key, "")) for key in row}
            )
        )
    lines.extend(["", "## Failed Or Not-Evaluable Gates", ""])
    any_detail = False
    for result in report.get("results", []):
        detail_rows = [
            gate
            for gate in result.get("gate_results", [])
            if gate.get("status") in {"fail", "not_evaluable"}
        ]
        if not detail_rows:
            continue
        any_detail = True
        lines.append(f"### {result['planner_key']} / {result['scenario_family']}")
        for gate in detail_rows:
            observed = gate.get("observed_values")
            lines.append(
                "- `{gate_id}`: {status}; metric `{metric}` observed={observed}, "
                "threshold={direction} {threshold}; provenance={provenance}".format(
                    gate_id=gate.get("gate_id"),
                    status=gate.get("status"),
                    metric=gate.get("metric"),
                    observed=observed,
                    direction=gate.get("direction"),
                    threshold=gate.get("threshold"),
                    provenance=gate.get("provenance"),
                )
            )
    if not any_detail:
        lines.append("No failed or not-evaluable gates.")
    lines.extend(["", "## Threshold Provenance", ""])
    for gate in report.get("gate_spec", {}).get("gates", []):
        lines.append(
            "- `{id}` ({category}, `{metric}`): {direction} {threshold}; {provenance}".format(
                **gate
            )
        )
    return "\n".join(lines) + "\n"


def _parse_gate(item: Mapping[str, Any], *, index: int) -> GateSpec:
    required = ("id", "metric", "threshold", "direction", "category", "provenance")
    missing = [field for field in required if field not in item]
    if missing:
        raise ReleaseGateSpecError(f"gates[{index}] missing required fields: {missing}")
    gate_id = _required_string(item["id"], f"gates[{index}].id")
    metric = _required_string(item["metric"], f"gates[{index}].metric")
    direction = _required_string(item["direction"], f"gates[{index}].direction")
    if direction not in _VALID_DIRECTIONS:
        raise ReleaseGateSpecError(f"gates[{index}].direction must be one of max, min")
    category = _required_string(item["category"], f"gates[{index}].category")
    if category not in _VALID_CATEGORIES:
        raise ReleaseGateSpecError(f"gates[{index}].category must be safety or comfort")
    threshold = _finite_float(item["threshold"], f"gates[{index}].threshold")
    provenance = _required_string(item["provenance"], f"gates[{index}].provenance")
    scope = item.get("scope")
    if scope is not None and not isinstance(scope, Mapping):
        raise ReleaseGateSpecError(f"gates[{index}].scope must be a mapping")
    return GateSpec(
        gate_id=gate_id,
        metric=metric,
        threshold=threshold,
        direction=direction,
        category=category,
        provenance=provenance,
        required=bool(item.get("required", True)),
        scope=scope,
        description=str(item["description"]) if item.get("description") is not None else None,
    )


def _evaluate_gate_for_group(
    gate: GateSpec,
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario_family: str,
) -> dict[str, Any]:
    observed_values = [_finite_float_or_none(_get_nested(row, gate.metric)) for row in rows]
    observed_values = [value for value in observed_values if value is not None]
    if not observed_values:
        status = "not_evaluable"
    elif gate.direction == "max":
        status = "pass" if all(value <= gate.threshold for value in observed_values) else "fail"
    else:
        status = "pass" if all(value >= gate.threshold for value in observed_values) else "fail"
    return {
        "gate_id": gate.gate_id,
        "metric": gate.metric,
        "category": gate.category,
        "scenario_family": scenario_family,
        "status": status,
        "observed_values": observed_values,
        "threshold": gate.threshold,
        "direction": gate.direction,
        "required": gate.required,
        "provenance": gate.provenance,
        "scope": dict(gate.scope or {}),
    }


def _normalize_input_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["planner_key"] = _first_present(
        row, _DEFAULT_PLANNER_KEYS, default="unknown_planner"
    )
    normalized["scenario_family"] = _first_present(row, _DEFAULT_FAMILY_KEYS, default="all")
    return normalized


def _rows_from_mapping(payload: Mapping[str, Any]) -> Any:
    # ``planner_rows`` is the canonical container emitted by the retained
    # camera-ready campaign summaries (``reports/campaign_summary.json``), so the
    # merged evaluator can consume a real retained campaign summary directly.
    for key in (
        "rows",
        "matrix_rows",
        "summary_rows",
        "planner_rows",
        "aggregates",
        "summaries",
        "results",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ReleaseGateSpecError(
        "input JSON mapping must contain one of rows, matrix_rows, summary_rows, "
        "planner_rows, aggregates, summaries, or results"
    )


def _category_status(gate_results: Sequence[Mapping[str, Any]], *, category: str) -> str:
    category_results = [
        result
        for result in gate_results
        if result.get("category") == category and result.get("required", True)
    ]
    if not category_results:
        return "not_evaluable"
    statuses = {str(result.get("status")) for result in category_results}
    if "fail" in statuses:
        return "fail"
    if "not_evaluable" in statuses:
        return "not_evaluable"
    return "pass"


def _overall_status(category_statuses: Mapping[str, str]) -> str:
    statuses = set(category_statuses.values())
    if "fail" in statuses:
        return "fail"
    if "not_evaluable" in statuses:
        return "not_evaluable"
    return "pass"


def _gate_applies_to_family(gate: GateSpec, scenario_family: str) -> bool:
    if not gate.scope:
        return True
    expected = gate.scope.get("scenario_family")
    if expected is None:
        return True
    if isinstance(expected, str):
        return expected == scenario_family
    if isinstance(expected, Sequence):
        return scenario_family in {str(item) for item in expected}
    return False


def _count_statuses(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(sorted(GATE_STATUSES), 0)
    for row in rows:
        status = str(row.get("overall_status", "not_evaluable"))
        counts[status if status in counts else "not_evaluable"] += 1
    return counts


def _first_present(row: Mapping[str, Any], keys: Sequence[str], *, default: str) -> str:
    for key in keys:
        value = _get_nested(row, key)
        if value not in (None, ""):
            return str(value)
    return default


def _get_nested(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    current: Any = row
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseGateSpecError(f"{field_name} must be a non-empty string")
    return value.strip()


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ReleaseGateSpecError(f"{field_name} must be numeric")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ReleaseGateSpecError(f"{field_name} must be numeric") from exc
    if not math.isfinite(threshold):
        raise ReleaseGateSpecError(f"{field_name} must be finite")
    return threshold


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return None
    return observed if math.isfinite(observed) else None


def _path_provenance(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {"path": str(path), "sha256": _sha256_file(resolved)}


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")
