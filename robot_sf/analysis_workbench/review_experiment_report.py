"""Offline comparison of recorded experiment outcomes for scenario review.

The component deliberately consumes recorded results instead of importing an
executor.  It makes effects conditional on a valid control and keeps
falsified, contradictory, and inconclusive observations visible alongside
survived observations.  A shared parent identifies one dependent family; the
report never treats its branches as independent samples.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
)
from robot_sf.errors import RobotSfError

EXPERIMENT_RESULTS_SCHEMA_VERSION = "experiment-results.v1"
EXPERIMENT_COMPARISON_SCHEMA_VERSION = "experiment-comparison.v1"
EXPERIMENT_COMPARISON_HTML_TYPE = "experiment-comparison-html.v1"
COMPONENT_ID = "srev25-experiment-report"
COMPONENT_VERSION = "1.0.0"
RECORDED_RESULTS_CAPABILITY = "recorded-experiment-results"
ACTIVATION_CAPABILITY = "activation-status"
JSON_ARTIFACT_NAME = "experiment-comparison.json"
HTML_ARTIFACT_NAME = "experiment-comparison.html"
EVIDENCE_BOUNDARY = "diagnostic-only recorded-results comparison; not benchmark or paper evidence"
OUTCOMES = ("survived", "falsified", "inconclusive", "contradictory")
VALID_GATE_STATUSES = ("pass", "fail", "missing", "unknown", "not_applicable")
_ALLOWED_CONFIG_KEYS = {"metric_order", "report_title"}


class ExperimentReportError(RobotSfError, ValueError):
    """Raised when a recorded-results report cannot be built safely."""

    def __init__(self, code: str, message: str):
        """Store a stable machine-readable code and human-readable detail."""

        self.code = code
        super().__init__(f"{code}: {message}")


class _UnavailableReportError(ExperimentReportError):
    """Raised when the component cannot support a requested input."""


DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=(EXPERIMENT_COMPARISON_SCHEMA_VERSION, EXPERIMENT_COMPARISON_HTML_TYPE),
    required_capabilities=(RECORDED_RESULTS_CAPABILITY,),
    optional_capabilities=(ACTIVATION_CAPABILITY,),
)


def component_descriptor() -> dict[str, Any]:
    """Return the versioned descriptor for this standalone component."""

    payload = asdict(DESCRIPTOR)
    for field_name in (
        "supported_input_versions",
        "output_types",
        "required_capabilities",
        "optional_capabilities",
    ):
        payload[field_name] = list(payload[field_name])
    return {"schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION, **payload}


def descriptor() -> dict[str, Any]:
    """Return the component descriptor for callers using the short alias.

    Returns:
        JSON-safe component descriptor.
    """

    return component_descriptor()


def _result(
    request: ComponentRequest,
    status: str,
    *,
    reason: str = "",
    artifacts: tuple[dict[str, Any], ...] = (),
    provenance: Mapping[str, Any] | None = None,
) -> ComponentResult:
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status=status,
        artifacts=artifacts,
        reason=reason,
        provenance=dict(provenance or {}),
    )


def _finite_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentReportError("invalid_input", f"{path} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ExperimentReportError("invalid_input", f"{path} must be a finite number") from None
    if not math.isfinite(number):
        raise ExperimentReportError("invalid_input", f"{path} must be a finite number")
    return number


def _non_empty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentReportError("invalid_input", f"{path} must be a non-empty string")
    return value


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentReportError("invalid_input", f"{path} must be an object")
    return value


def _validate_strict_json(value: Any, *, path: str = "/source") -> None:
    """Reject JSON parser extensions that would make artifacts non-canonical."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ExperimentReportError("invalid_input", f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ExperimentReportError("invalid_input", f"{path} has a non-string object key")
            _validate_strict_json(child, path=f"{path}/{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _validate_strict_json(child, path=f"{path}/{index}")


def _list(value: Any, *, path: str) -> list[Any]:
    if type(value) is not list:
        raise ExperimentReportError("invalid_input", f"{path} must be an array")
    return value


def _validate_config(config: Mapping[str, Any]) -> tuple[list[str], str | None]:
    unknown = sorted(set(config) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ExperimentReportError(
            "invalid_config", f"unsupported config keys: {', '.join(unknown)}"
        )
    metric_order_value = config.get("metric_order", [])
    metric_order = _list(metric_order_value, path="/config/metric_order")
    if any(not isinstance(item, str) or not item.strip() for item in metric_order):
        raise ExperimentReportError(
            "invalid_config", "/config/metric_order must contain non-empty strings"
        )
    if len(set(metric_order)) != len(metric_order):
        raise ExperimentReportError("invalid_config", "/config/metric_order must be unique")
    report_title = config.get("report_title")
    if report_title is not None:
        report_title = _non_empty_string(report_title, path="/config/report_title")
    return list(metric_order), report_title


def _validate_gate(gate: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(gate, path=path)
    status = _non_empty_string(payload.get("status"), path=f"{path}/status")
    if status not in VALID_GATE_STATUSES:
        raise ExperimentReportError(
            "invalid_input",
            f"{path}/status must be one of {', '.join(VALID_GATE_STATUSES)}",
        )
    result = {"status": status}
    for key in sorted(payload):
        if key != "status":
            result[key] = payload[key]
    return result


def _validate_measurements(measurements: Any, *, path: str) -> dict[str, dict[str, Any]]:
    payload = _mapping(measurements, path=path)
    result: dict[str, dict[str, Any]] = {}
    for metric_name in sorted(payload):
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ExperimentReportError("invalid_input", f"{path} has an invalid metric name")
        measurement = _mapping(payload[metric_name], path=f"{path}/{metric_name}")
        units = _non_empty_string(measurement.get("units"), path=f"{path}/{metric_name}/units")
        if "value" not in measurement:
            raise ExperimentReportError(
                "invalid_input", f"{path}/{metric_name}/value is required; use null when missing"
            )
        value = measurement["value"]
        if value is not None:
            value = _finite_number(value, path=f"{path}/{metric_name}/value")
        item: dict[str, Any] = {"units": units, "value": value}
        if "expected_direction" in measurement:
            expected = measurement["expected_direction"]
            if not isinstance(expected, str):
                raise ExperimentReportError(
                    "invalid_input",
                    f"{path}/{metric_name}/expected_direction must be a string",
                )
            item["expected_direction"] = expected
        result[metric_name] = item
    return result


def _validate_source(payload: Any) -> dict[str, Any]:  # noqa: C901
    _validate_strict_json(payload)
    source = _mapping(payload, path="/source")
    schema_version = _non_empty_string(source.get("schema_version"), path="/source/schema_version")
    if schema_version != EXPERIMENT_RESULTS_SCHEMA_VERSION:
        raise _UnavailableReportError(
            "incompatible_version",
            f"source schema {schema_version!r} is not supported; expected "
            f"{EXPERIMENT_RESULTS_SCHEMA_VERSION}",
        )
    experiment_id = _non_empty_string(source.get("experiment_id"), path="/source/experiment_id")
    hypothesis = _non_empty_string(source.get("hypothesis"), path="/source/hypothesis")
    source_identity = dict(_mapping(source.get("source_identity"), path="/source/source_identity"))
    families = _list(source.get("families"), path="/source/families")
    if not families:
        raise ExperimentReportError("invalid_input", "/source/families must not be empty")

    normalized_families: list[dict[str, Any]] = []
    seen_family_ids: set[str] = set()
    seen_shared_parent_ids: set[str] = set()
    for family_index, family_value in enumerate(families):
        family = _mapping(family_value, path=f"/source/families/{family_index}")
        family_id = _non_empty_string(
            family.get("family_id"), path=f"/source/families/{family_index}/family_id"
        )
        if family_id in seen_family_ids:
            raise ExperimentReportError("invalid_input", f"duplicate family_id: {family_id}")
        seen_family_ids.add(family_id)
        shared_parent_id = _non_empty_string(
            family.get("shared_parent_id"),
            path=f"/source/families/{family_index}/shared_parent_id",
        )
        if shared_parent_id in seen_shared_parent_ids:
            raise ExperimentReportError(
                "invalid_input",
                f"shared_parent_id must identify one family: {shared_parent_id}",
            )
        seen_shared_parent_ids.add(shared_parent_id)
        control_id = _non_empty_string(
            family.get("control_condition_id"),
            path=f"/source/families/{family_index}/control_condition_id",
        )
        conditions = _list(
            family.get("conditions"), path=f"/source/families/{family_index}/conditions"
        )
        if not conditions:
            raise ExperimentReportError(
                "invalid_input", f"/source/families/{family_index}/conditions must not be empty"
            )
        normalized_conditions: list[dict[str, Any]] = []
        seen_condition_ids: set[str] = set()
        control_count = 0
        for condition_index, condition_value in enumerate(conditions):
            condition = _mapping(
                condition_value,
                path=f"/source/families/{family_index}/conditions/{condition_index}",
            )
            condition_path = f"/source/families/{family_index}/conditions/{condition_index}"
            condition_id = _non_empty_string(
                condition.get("condition_id"), path=f"{condition_path}/condition_id"
            )
            if condition_id in seen_condition_ids:
                raise ExperimentReportError(
                    "invalid_input", f"duplicate condition_id in {family_id}: {condition_id}"
                )
            seen_condition_ids.add(condition_id)
            label = _non_empty_string(condition.get("label"), path=f"{condition_path}/label")
            role = _non_empty_string(condition.get("role"), path=f"{condition_path}/role")
            if role not in {"control", "treatment"}:
                raise ExperimentReportError(
                    "invalid_input", f"{condition_path}/role must be control or treatment"
                )
            if role == "control":
                control_count += 1
            outcome = _non_empty_string(condition.get("outcome"), path=f"{condition_path}/outcome")
            if outcome not in OUTCOMES:
                raise ExperimentReportError(
                    "invalid_input",
                    f"{condition_path}/outcome must be one of {', '.join(OUTCOMES)}",
                )
            normalized_conditions.append(
                {
                    "condition_id": condition_id,
                    "label": label,
                    "role": role,
                    "outcome": outcome,
                    "fidelity": _validate_gate(
                        condition.get("fidelity"), path=f"{condition_path}/fidelity"
                    ),
                    "activation": _validate_gate(
                        condition.get("activation"), path=f"{condition_path}/activation"
                    ),
                    "measurements": _validate_measurements(
                        condition.get("measurements"), path=f"{condition_path}/measurements"
                    ),
                }
            )
        if control_count != 1:
            raise ExperimentReportError(
                "invalid_input",
                f"family {family_id} must contain exactly one control condition",
            )
        if control_id not in seen_condition_ids:
            raise ExperimentReportError(
                "invalid_input", f"control condition {control_id} is missing in {family_id}"
            )
        control_condition = next(
            condition
            for condition in normalized_conditions
            if condition["condition_id"] == control_id
        )
        if control_condition["role"] != "control":
            raise ExperimentReportError(
                "invalid_input",
                f"control_condition_id {control_id} must identify the role=control condition "
                f"in {family_id}",
            )
        normalized_families.append(
            {
                "family_id": family_id,
                "shared_parent_id": shared_parent_id,
                "control_condition_id": control_id,
                "conditions": normalized_conditions,
            }
        )
    return {
        "schema_version": schema_version,
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "source_identity": source_identity,
        "families": normalized_families,
    }


def _ordered_metrics(
    control: Mapping[str, Any], treatment: Mapping[str, Any], configured: list[str]
) -> list[str]:
    names = set(control) | set(treatment)
    return [name for name in configured if name in names] + sorted(names - set(configured))


def _gate_ready(gate: Mapping[str, Any]) -> bool:
    return gate.get("status") == "pass"


def _effect_reason(
    control: Mapping[str, Any], treatment: Mapping[str, Any], *, metric: str
) -> str | None:
    if not _gate_ready(control["fidelity"]):
        return "control_fidelity_not_verified"
    if not _gate_ready(control["activation"]):
        return "control_activation_not_verified"
    if not _gate_ready(treatment["fidelity"]):
        return "treatment_fidelity_not_verified"
    if not _gate_ready(treatment["activation"]):
        return "treatment_activation_not_verified"
    if metric not in control["measurements"] or metric not in treatment["measurements"]:
        return "measurement_missing_in_one_condition"
    control_measurement = control["measurements"][metric]
    treatment_measurement = treatment["measurements"][metric]
    if control_measurement["units"] != treatment_measurement["units"]:
        return "measurement_units_mismatch"
    if control_measurement["value"] is None or treatment_measurement["value"] is None:
        return "measurement_value_missing"
    return None


def _build_effects(
    control: Mapping[str, Any], treatment: Mapping[str, Any], configured: list[str]
) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    for metric in _ordered_metrics(control["measurements"], treatment["measurements"], configured):
        control_measurement = control["measurements"].get(metric)
        treatment_measurement = treatment["measurements"].get(metric)
        units = None
        if control_measurement is not None:
            units = control_measurement["units"]
        elif treatment_measurement is not None:
            units = treatment_measurement["units"]
        effect: dict[str, Any] = {
            "metric": metric,
            "units": units,
            "control_value": control_measurement["value"] if control_measurement else None,
            "treatment_value": treatment_measurement["value"] if treatment_measurement else None,
            "status": "blocked",
            "reason": None,
        }
        reason = _effect_reason(control, treatment, metric=metric)
        if reason is None:
            control_value = float(control_measurement["value"])
            treatment_value = float(treatment_measurement["value"])
            delta = treatment_value - control_value
            if not math.isfinite(delta):
                raise ExperimentReportError(
                    "invalid_input", f"difference for {metric} must be finite"
                )
            effect.update(
                {
                    "status": "interpretable",
                    "delta": delta,
                    "direction": "increase"
                    if delta > 0
                    else "decrease"
                    if delta < 0
                    else "no_change",
                }
            )
        else:
            effect["reason"] = reason
        effects.append(effect)
    return effects


def _condition_summary(condition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": condition["condition_id"],
        "label": condition["label"],
        "role": condition["role"],
        "outcome": condition["outcome"],
        "fidelity": dict(condition["fidelity"]),
        "activation": dict(condition["activation"]),
        "measurements": {
            metric: dict(measurement)
            for metric, measurement in sorted(condition["measurements"].items())
        },
    }


def _build_report(
    source: Mapping[str, Any],
    *,
    request: ComponentRequest,
    source_ref: Mapping[str, Any],
    source_sha256: str,
    metric_order: list[str],
    report_title: str | None,
) -> dict[str, Any]:
    outcome_counts = dict.fromkeys(OUTCOMES, 0)
    effect_counts = {"interpretable": 0, "blocked": 0}
    negative_findings: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    condition_count = 0
    for family in source["families"]:
        conditions = family["conditions"]
        condition_by_id = {condition["condition_id"]: condition for condition in conditions}
        control = condition_by_id[family["control_condition_id"]]
        treatment_reports: list[dict[str, Any]] = []
        family_effects: list[dict[str, Any]] = []
        for condition in conditions:
            outcome_counts[condition["outcome"]] += 1
            condition_count += 1
            if condition["outcome"] != "survived":
                negative_findings.append(
                    {
                        "family_id": family["family_id"],
                        "shared_parent_id": family["shared_parent_id"],
                        "condition_id": condition["condition_id"],
                        "outcome": condition["outcome"],
                        "finding": "outcome_retained_without_positive_claim",
                    }
                )
            if condition["role"] == "treatment":
                effects = _build_effects(control, condition, metric_order)
                family_effects.extend(effects)
                treatment_reports.append(
                    {
                        "condition": _condition_summary(condition),
                        "effects": effects,
                    }
                )
        for effect in family_effects:
            effect_counts[effect["status"]] += 1
        control_ready = _gate_ready(control["fidelity"]) and _gate_ready(control["activation"])
        families.append(
            {
                "family_id": family["family_id"],
                "shared_parent_id": family["shared_parent_id"],
                "sample_unit": "dependent_family",
                "control": _condition_summary(control),
                "treatments": treatment_reports,
                "all_conditions": [_condition_summary(condition) for condition in conditions],
                "control_prerequisites": {
                    "fidelity_status": control["fidelity"]["status"],
                    "activation_status": control["activation"]["status"],
                    "effect_interpretation_allowed": control_ready,
                },
                "effect_interpretation": "allowed" if control_ready else "blocked",
                "effects": family_effects,
            }
        )
    report: dict[str, Any] = {
        "schema_version": EXPERIMENT_COMPARISON_SCHEMA_VERSION,
        "report_id": f"{source['experiment_id']}-comparison",
        "component_id": request.component_id,
        "request_id": request.request_id,
        "title": report_title or f"Experiment comparison: {source['experiment_id']}",
        "hypothesis": source["hypothesis"],
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "source": {
            "artifact_id": source_ref["artifact_id"],
            "uri": source_ref["uri"],
            "format": source_ref["format"],
            "sha256": source_sha256,
            "schema_version": source["schema_version"],
            "identity": dict(source["source_identity"]),
        },
        "families": families,
        "negative_findings": negative_findings,
        "summary": {
            "family_count": len(families),
            "condition_count": condition_count,
            "outcome_counts": outcome_counts,
            "effect_counts": effect_counts,
            "negative_finding_count": len(negative_findings),
            "sample_unit": "shared_parent_family",
            "independent_family_count": len(families),
            "recorded_results_only": True,
            "executor_required": False,
        },
        "provenance": {
            "component_version": COMPONENT_VERSION,
            "input_schema": EXPERIMENT_RESULTS_SCHEMA_VERSION,
            "execution_mode": "recorded_results_only",
            "evidence_tier": "diagnostic",
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "metric_order": metric_order,
            "source_sha256": source_sha256,
            "source_identity": dict(source["source_identity"]),
        },
    }
    return report


def _render_html(report: Mapping[str, Any]) -> str:
    """Render a dependency-free, deterministic HTML report.

    Returns:
        Complete UTF-8 HTML document.
    """

    def safe(value: Any) -> str:
        return html.escape(str(value))

    family_sections: list[str] = []
    for family in report["families"]:
        rows: list[str] = []
        for condition in family["all_conditions"]:
            rows.append(
                "<tr>"
                f"<td>{safe(condition['label'])}</td>"
                f"<td>{safe(condition['role'])}</td>"
                f"<td>{safe(condition['outcome'])}</td>"
                f"<td>{safe(condition['fidelity']['status'])}</td>"
                f"<td>{safe(condition['activation']['status'])}</td>"
                "</tr>"
            )
        effect_rows = []
        for effect in family["effects"]:
            effect_rows.append(
                "<tr>"
                f"<td>{safe(effect['metric'])}</td>"
                f"<td>{safe(effect['units'])}</td>"
                f"<td>{safe(effect['status'])}</td>"
                f"<td>{safe(effect.get('delta', '—'))}</td>"
                f"<td>{safe(effect.get('reason', ''))}</td>"
                "</tr>"
            )
        family_sections.append(
            f"<section><h2>Family {safe(family['family_id'])}</h2>"
            f"<p>Shared parent: <code>{safe(family['shared_parent_id'])}</code>; "
            f"sample unit: <code>{safe(family['sample_unit'])}</code>; "
            f"effect interpretation: <strong>{safe(family['effect_interpretation'])}</strong>.</p>"
            "<h3>Recorded outcomes and prerequisites</h3>"
            "<table><thead><tr><th>Condition</th><th>Role</th><th>Outcome</th>"
            "<th>Fidelity</th><th>Activation</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            "<h3>Effects</h3>"
            "<table><thead><tr><th>Metric</th><th>Units</th><th>Status</th>"
            "<th>Delta</th><th>Reason</th></tr></thead><tbody>"
            + "".join(effect_rows)
            + "</tbody></table></section>"
        )
    outcome_items = "".join(
        f"<li>{safe(outcome)}: {safe(count)}</li>"
        for outcome, count in report["summary"]["outcome_counts"].items()
    )
    finding_items = "".join(
        "<li>"
        f"<code>{safe(finding['family_id'])}</code> / "
        f"<code>{safe(finding['condition_id'])}</code>: "
        f"{safe(finding['outcome'])}</li>"
        for finding in report["negative_findings"]
    )
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>{safe(report['title'])}</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2rem auto;line-height:1.4}"
        "table{border-collapse:collapse;margin:1rem 0;width:100%}"
        "th,td{border:1px solid #bbb;padding:.35rem;text-align:left}"
        "th{background:#eee}code{white-space:nowrap}"
        ".boundary{border-left:4px solid #c33;padding:.5rem 1rem;background:#fff4f4}</style>"
        "</head><body>"
        f"<h1>{safe(report['title'])}</h1>"
        f"<p>{safe(report['hypothesis'])}</p>"
        f'<p class="boundary"><strong>Evidence boundary:</strong> '
        f"{safe(report['evidence_boundary'])}</p>"
        '<p class="authority"><strong>Authoritative detail:</strong> The JSON artifact is '
        "authoritative for complete measurement values, units, expected directions, and source "
        f"provenance; see <code>{safe(JSON_ARTIFACT_NAME)}</code>. This HTML is a human-readable "
        "summary.</p>"
        f"<p>Families: {safe(report['summary']['family_count'])}; "
        f"conditions: {safe(report['summary']['condition_count'])}; "
        "comparison uses recorded results only and requires no executor.</p>"
        f"<h2>Outcome inventory</h2><ul>{outcome_items}</ul>"
        f"<h2>Negative findings</h2><ul>{finding_items or '<li>None recorded</li>'}</ul>"
        + "".join(family_sections)
        + "</body></html>\n"
    )


def _resolve_inside(root: Path, value: str, *, path: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ExperimentReportError(
            "invalid_input", f"{path} resolves outside the component base"
        ) from error
    return candidate


def _read_source(
    root: Path, request: ComponentRequest
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not request.sources:
        raise ExperimentReportError(
            "invalid_input", "at least one recorded-results source is required"
        )
    if len(request.sources) != 1:
        raise ExperimentReportError(
            "invalid_input", "exactly one recorded-results source is supported"
        )
    source_ref = asdict(request.sources[0])
    if source_ref["format"] != EXPERIMENT_RESULTS_SCHEMA_VERSION:
        raise _UnavailableReportError(
            "incompatible_version",
            f"source format {source_ref['format']!r} is not supported; expected "
            f"{EXPERIMENT_RESULTS_SCHEMA_VERSION}",
        )
    source_path = _resolve_inside(root, source_ref["uri"], path="/sources/0/uri")
    try:
        raw = source_path.read_bytes()
    except OSError as error:
        raise _UnavailableReportError(
            "source_unavailable", f"cannot read source {source_ref['uri']!r}: {error}"
        ) from error
    source_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExperimentReportError(
            "invalid_input", f"source is not valid UTF-8 JSON: {error}"
        ) from error
    try:
        payload = json.loads(decoded)
    except ValueError as error:
        raise ExperimentReportError(
            "invalid_input", f"source is not valid UTF-8 JSON: {error}"
        ) from error
    return _validate_source(payload), source_ref, source_sha256


def _write_artifact(path: Path, content: str) -> str:
    try:
        raw = content.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ExperimentReportError(
            "invalid_input", "report contains invalid Unicode text"
        ) from error
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _write_artifacts(output_dir: Path, artifacts: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """Write all report artifacts before publishing their output directory.

    Returns:
        SHA-256 digests in the same order as ``artifacts``.
    """

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging-"
    ) as staging_name:
        staging_dir = Path(staging_name)
        digests = tuple(
            _write_artifact(staging_dir / artifact_name, content)
            for artifact_name, content in artifacts
        )
        staging_dir.replace(output_dir)
    return digests


def _resolve_output_directory(root: Path, request: ComponentRequest) -> Path:
    output_dir = _resolve_inside(root, request.output_directory, path="/output_directory")
    if output_dir.exists():
        raise ExperimentReportError(
            "output_collision", f"output directory already exists: {request.output_directory}"
        )
    return output_dir


def _unsupported_capabilities(request: ComponentRequest) -> list[str]:
    supported = set(DESCRIPTOR.required_capabilities) | set(DESCRIPTOR.optional_capabilities)
    return sorted(name for name in request.required_capabilities if name not in supported)


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Compare one recorded-results source without starting an executor.

    Args:
        request: Validated shared component request.
        base: Directory against which source and output paths are resolved.

    Returns:
        A complete result with JSON/HTML artifacts, or a status-bearing result
        with no artifacts when the request cannot be supported safely.
    """

    if request.component_id != COMPONENT_ID:
        return _result(
            request,
            "unavailable",
            reason=f"unsupported_component: {request.component_id}",
        )
    missing = _unsupported_capabilities(request)
    if missing:
        return _result(
            request,
            "unavailable",
            reason=f"missing_capability: {', '.join(missing)}",
        )
    root = (base if base is not None else Path.cwd()).resolve()
    try:
        output_dir = _resolve_output_directory(root, request)
        metric_order, report_title = _validate_config(request.config)
        source, source_ref, source_sha256 = _read_source(root, request)
        report = _build_report(
            source,
            request=request,
            source_ref=source_ref,
            source_sha256=source_sha256,
            metric_order=metric_order,
            report_title=report_title,
        )
        json_text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        html_text = _render_html(report)
        json_digest, html_digest = _write_artifacts(
            output_dir,
            (
                (JSON_ARTIFACT_NAME, json_text),
                (HTML_ARTIFACT_NAME, html_text),
            ),
        )
        artifacts = (
            {
                "artifact_id": JSON_ARTIFACT_NAME,
                "uri": str(Path(request.output_directory) / JSON_ARTIFACT_NAME),
                "sha256": json_digest,
            },
            {
                "artifact_id": HTML_ARTIFACT_NAME,
                "uri": str(Path(request.output_directory) / HTML_ARTIFACT_NAME),
                "sha256": html_digest,
            },
        )
        return _result(
            request,
            "complete",
            artifacts=artifacts,
            provenance=report["provenance"],
        )
    except _UnavailableReportError as error:
        return _result(request, "unavailable", reason=str(error))
    except ExperimentReportError as error:
        return _result(request, "failed", reason=str(error))
    except OSError as error:
        return _result(request, "failed", reason=f"output_write_error: {error}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare recorded Robot SF experiment results offline."
    )
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", default=None, help="Optional validated config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory; it must not exist.")
    parser.add_argument("--base", default=None, help="Base directory for source/output paths.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the component CLI and print its shared result envelope.

    Returns:
        Zero when the report is complete, otherwise one.
    """

    args = _build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read request: {error}"]) from error
    if not isinstance(payload, dict):
        raise ReviewContractsValidationError(["request must be a JSON object"])
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
        if not isinstance(config, dict):
            raise ReviewContractsValidationError(["config must be a JSON object"])
        existing_config = payload.get("config", {})
        if not isinstance(existing_config, dict):
            raise ReviewContractsValidationError(["request config must be a JSON object"])
        payload = {**payload, "config": {**existing_config, **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))  # noqa: T201 - CLI output
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
