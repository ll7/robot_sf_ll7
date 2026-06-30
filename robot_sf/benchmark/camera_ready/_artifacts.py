"""Artifact writer helpers for camera-ready benchmark campaigns."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.camera_ready._util import _sanitize_csv_cell
from robot_sf.benchmark.camera_ready_campaign_config import _AMV_DIMENSIONS
from robot_sf.benchmark.seed_variance import build_seed_variability_csv_rows
from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES

if TYPE_CHECKING:
    from pathlib import Path


def _escape_markdown_cell(value: Any) -> str:
    """Escape markdown table cell content to prevent row/column injection.

    Returns:
        Escaped single-line markdown-safe cell value.
    """
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ").replace("\r", " ")
    return text


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting and trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], headers: tuple[str, ...]) -> None:
    """Write a table in Markdown format using explicit header order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        values = [_escape_markdown_cell(row.get(col, "")) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write campaign summary table in CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _sanitize_csv_cell(value) for key, value in row.items()})


def _write_table_artifacts(
    reports_dir: Path,
    base_name: str,
    rows: list[dict[str, Any]],
    *,
    headers: tuple[str, ...],
) -> tuple[Path, Path]:
    """Write CSV and Markdown table artifacts for one table dataset.

    Returns:
        Tuple of generated ``(csv_path, markdown_path)``.
    """
    csv_path = reports_dir / f"{base_name}.csv"
    md_path = reports_dir / f"{base_name}.md"
    csv_rows = [{key: row.get(key, "") for key in headers} for row in rows]
    _write_csv(csv_path, csv_rows)
    _write_markdown_table(md_path, rows, headers=headers)
    return csv_path, md_path


def _write_matrix_summary_artifacts(
    reports_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write matrix-definition summary artifacts.

    Returns:
        Tuple of ``(matrix_summary_json_path, matrix_summary_csv_path)``.
    """
    json_path = reports_dir / "matrix_summary.json"
    csv_path = reports_dir / "matrix_summary.csv"
    payload = {
        "schema_version": "benchmark-matrix-summary.v1",
        "rows": rows,
    }
    _write_json(json_path, payload)
    _write_csv(csv_path, rows)
    return json_path, csv_path


def _write_seed_variability_artifacts(
    reports_dir: Path,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Write paper-facing seed-variability JSON and CSV artifacts.

    Returns:
        Tuple of ``(json_path, csv_path)`` for the generated artifacts.
    """
    json_path = reports_dir / "seed_variability_by_scenario.json"
    csv_path = reports_dir / "seed_variability_by_scenario.csv"
    _write_json(json_path, payload)
    csv_rows = build_seed_variability_csv_rows(
        payload.get("rows") or [],
        metrics=payload.get("metrics") or [],
    )
    _write_csv(csv_path, csv_rows)
    return json_path, csv_path


def _write_seed_episode_rows_artifact(
    reports_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write flat planner-aware per-episode seed traceability CSV.

    Returns:
        Path to the generated CSV artifact.
    """
    csv_path = reports_dir / "seed_episode_rows.csv"
    _write_csv(csv_path, rows)
    return csv_path


def _write_statistical_sufficiency_artifact(
    reports_dir: Path,
    payload: dict[str, Any],
) -> Path:
    """Write statistical sufficiency JSON artifact.

    Returns:
        Path to the generated JSON artifact.
    """
    json_path = reports_dir / "statistical_sufficiency.json"
    _write_json(json_path, payload)
    return json_path


def _write_amv_coverage_artifacts(
    reports_dir: Path,
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Write AMV coverage summary JSON + Markdown artifacts.

    Returns:
        tuple[Path, Path]: Output paths ``(json_path, markdown_path)``.
    """
    json_path = reports_dir / "amv_coverage_summary.json"
    md_path = reports_dir / "amv_coverage_summary.md"
    _write_json(json_path, summary)

    lines = [
        "# AMV Coverage Summary",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Profile: `{summary.get('profile_name', 'unknown')}`",
        f"- Contract version: `{summary.get('contract_version', 'unknown')}`",
        f"- Enforcement: `{summary.get('coverage_enforcement', 'warn')}`",
        f"- Scenario count: `{summary.get('scenario_count', 0)}`",
        "",
        "| Dimension | Required | Observed | Missing |",
        "|---|---|---|---|",
    ]
    required = summary.get("required_dimensions", {})
    observed = summary.get("observed_dimensions", {})
    missing = summary.get("missing_dimensions", {})
    for dimension in _AMV_DIMENSIONS:
        required_values = ", ".join(required.get(dimension, [])) or "-"
        observed_values = ", ".join(observed.get(dimension, [])) or "-"
        missing_values = ", ".join(missing.get(dimension, [])) or "-"
        lines.append(
            "| "
            f"{_escape_markdown_cell(dimension)} | "
            f"{_escape_markdown_cell(required_values)} | "
            f"{_escape_markdown_cell(observed_values)} | "
            f"{_escape_markdown_cell(missing_values)} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _write_actuation_envelope_artifacts(
    reports_dir: Path,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Write synthetic actuation-envelope JSON and Markdown artifacts.

    Returns:
        Paths to the JSON and Markdown artifacts.
    """
    json_path = reports_dir / "actuation_envelope_summary.json"
    md_path = reports_dir / "actuation_envelope_summary.md"
    _write_json(json_path, payload)
    lines = [
        "# Synthetic Actuation Envelope Summary",
        "",
        f"- Campaign ID: `{payload.get('campaign_id', 'unknown')}`",
        f"- Claim boundary: {payload.get('claim_boundary', '')}",
        "",
        "## Profile",
        "",
    ]
    profile = payload.get("synthetic_actuation_profile")
    if isinstance(profile, dict):
        for key, value in profile.items():
            lines.append(f"- {key}: `{value}`")
    lines.append(f"- AMV coverage status: `{payload.get('amv_coverage_status', 'unknown')}`")
    scenario_rows = payload.get("scenario_amv_rows")
    if isinstance(scenario_rows, list) and scenario_rows:
        lines.extend(["", "## Scenario AMV Rows", ""])
        for row in scenario_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- "
                f"{row.get('name', 'unknown')} ({row.get('scenario_family', 'unknown')}): "
                f"{json.dumps(row.get('amv', {}), sort_keys=True)}"
            )
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        lines.extend(["", "## Planner Rows", ""])
        for row in rows:
            if not isinstance(row, dict):
                continue
            saturation = row.get("saturation_metrics")
            if not isinstance(saturation, dict):
                saturation = {}
            lines.append(
                "- "
                f"{row.get('planner_key', 'unknown')} ({row.get('algo', 'unknown')}, "
                f"{row.get('kinematics', 'unknown')}): status={row.get('status', 'unknown')}, "
                f"projection={row.get('projection_policy', 'unknown')}, "
                f"planner_cmd={row.get('planner_command_space', 'unknown')}, "
                f"benchmark_cmd={row.get('benchmark_command_space', 'unknown')}, "
                f"clip={saturation.get('command_clip_fraction', 'not_available')}, "
                f"yaw_sat={saturation.get('yaw_rate_saturation_fraction', 'not_available')}, "
                f"braking_peak={saturation.get('signed_braking_peak_m_s2', 'not_available')}"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _markdown_rows_from_mapping_rows(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> list[str]:
    """Render Markdown table rows from mapping rows and column order.

    Returns:
        list[str]: Markdown table rows in ``| a | b |`` form.
    """
    output: list[str] = []
    for row in rows:
        cells = [_escape_markdown_cell(row.get(column)) for column in columns]
        output.append("| " + " | ".join(cells) + " |")
    return output


def _write_comparability_artifacts(
    reports_dir: Path,
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Write comparability summary JSON + Markdown artifacts.

    Returns:
        tuple[Path, Path]: Output paths ``(json_path, markdown_path)``.
    """
    json_path = reports_dir / "comparability_matrix.json"
    md_path = reports_dir / "comparability_matrix.md"
    _write_json(json_path, summary)
    lines = [
        "# Alyassi Comparability Summary",
        "",
        f"- Mapping path: `{summary.get('mapping_path', 'unknown')}`",
        f"- Mapping version: `{summary.get('mapping_version', 'unknown')}`",
        f"- Mapping hash: `{summary.get('mapping_hash', 'unknown')}`",
        "",
        "## Coverage Overlap Matrix",
        "",
        "| Robot SF Family | Scenario Count | Alyassi Category | Overlap |",
        "|---|---:|---|---|",
    ]
    lines.extend(
        _markdown_rows_from_mapping_rows(
            list(summary.get("coverage_overlap_rows", [])),
            ("robot_sf_family", "scenario_count", "alyassi_category", "overlap"),
        )
    )
    lines.extend(
        [
            "",
            "## Metric Comparability",
            "",
            "| Metric | Classification | Alyassi Metric | Rationale |",
            "|---|---|---|---|",
        ]
    )
    lines.extend(
        _markdown_rows_from_mapping_rows(
            list(summary.get("metric_comparability_rows", [])),
            ("metric", "classification", "alyassi_metric", "rationale"),
        )
    )
    lines.extend(["", "## AMV-Specific Extensions", ""])
    extensions = summary.get("amv_specific_extensions", [])
    if extensions:
        for extension in extensions:
            lines.append(f"- {_escape_markdown_cell(extension)}")
    else:
        lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _write_snqi_diagnostics_artifacts(  # noqa: C901
    reports_dir: Path,
    payload: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Write SNQI diagnostics JSON/Markdown and sensitivity CSV artifacts.

    Returns:
        Tuple of ``(json_path, markdown_path, csv_path)``.
    """
    json_path = reports_dir / "snqi_diagnostics.json"
    md_path = reports_dir / "snqi_diagnostics.md"
    csv_path = reports_dir / "snqi_sensitivity.csv"
    _write_json(json_path, payload)

    lines = [
        "# SNQI Diagnostics",
        "",
        f"- Contract status: `{payload.get('contract_status', 'unknown')}`",
        f"- Rank alignment (Spearman): `{payload.get('rank_alignment_spearman', 0.0):.4f}`",
        f"- Outcome separation: `{payload.get('outcome_separation', 0.0):.4f}`",
        f"- Objective score: `{payload.get('objective_score', 0.0):.4f}`",
        f"- Dominant component: `{payload.get('dominant_component', 'unknown')}`",
        f"- Dominant component mean |contribution|: `{payload.get('dominant_component_mean_abs', 0.0):.4f}`",
        "",
        "## SNQI Assets",
        "",
        f"- Weights path: `{payload.get('weights_path', 'derived')}`",
        f"- Weights version: `{payload.get('weights_version', 'unknown')}`",
        f"- Weights SHA-256: `{payload.get('weights_sha256', 'unknown')}`",
        f"- Baseline path: `{payload.get('baseline_path', 'derived')}`",
        f"- Baseline version: `{payload.get('baseline_version', 'unknown')}`",
        f"- Baseline SHA-256: `{payload.get('baseline_sha256', 'unknown')}`",
        "",
        "## Baseline Normalization",
        "",
        f"- Source: `{payload.get('baseline_source', 'unknown')}`",
        f"- Degeneracy adjustments: `{payload.get('baseline_adjustments', 0)}`",
        "",
        "## Positioning",
        "",
        f"- Recommendation: `{payload.get('positioning', {}).get('recommendation', 'unknown')}`",
        f"- Claim scope: `{payload.get('positioning', {}).get('claim_scope', 'unknown')}`",
        f"- Aligned variable metrics: `{payload.get('positioning', {}).get('aligned_metric_count', 0)}` / `{payload.get('positioning', {}).get('variable_metric_count', 0)}`",
        "",
        "## Planner Ordering",
        "",
        "| Rank | Planner | Kinematics | Mean SNQI | Episodes |",
        "|---:|---|---|---:|---:|",
    ]
    ordering = payload.get("planner_ordering")
    if isinstance(ordering, list):
        for row in ordering:
            lines.append(
                "| {rank} | {planner_key} | {kinematics} | {mean_snqi:.6f} | {episode_count} |".format(
                    rank=int(row.get("rank", 0) or 0),
                    planner_key=str(row.get("planner_key", "unknown")),
                    kinematics=str(row.get("kinematics", "unknown")),
                    mean_snqi=float(row.get("mean_snqi", 0.0) or 0.0),
                    episode_count=int(row.get("episode_count", 0) or 0),
                )
            )
    lines.extend(
        [
            "",
            "## Component Correlations",
            "",
            "| Metric | Direction | Spearman | Variable | Aligned |",
            "|---|---|---:|---|---|",
        ]
    )
    correlations = payload.get("component_correlations")
    if isinstance(correlations, dict):
        for metric_name, row in sorted(correlations.items()):
            spearman = row.get("spearman")
            spearman_text = "n/a" if spearman is None else f"{float(spearman):.6f}"
            aligned = row.get("aligned_with_expected_direction")
            aligned_text = "n/a" if aligned is None else ("yes" if aligned else "no")
            lines.append(
                "| {metric} | {direction} | {spearman} | {variable} | {aligned} |".format(
                    metric=_escape_markdown_cell(str(metric_name)),
                    direction=_escape_markdown_cell(str(row.get("direction", "unknown"))),
                    spearman=spearman_text,
                    variable="yes" if bool(row.get("variable")) else "no",
                    aligned=aligned_text,
                )
            )
    lines.extend(
        [
            "",
            "## Component Dominance (mean absolute contribution)",
            "",
            "| Component | Mean |",
            "|---|---:|",
        ]
    )
    dominance = payload.get("component_dominance")
    if isinstance(dominance, dict):
        for key, value in sorted(dominance.items()):
            lines.append(f"| {_escape_markdown_cell(key)} | {float(value):.6f} |")
    caveats = payload.get("positioning", {}).get("caveats")
    if isinstance(caveats, list) and caveats:
        lines.extend(["", "## Caveats", ""])
        for caveat in caveats:
            lines.append(f"- {_escape_markdown_cell(str(caveat))}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    calibrated = payload.get("calibrated_weights")
    headers = (
        "component",
        "metric_name",
        "configured_weight",
        "configured_weight_share",
        "calibrated_weight",
        "delta",
        "mean_abs_contribution",
        "mean_abs_score_delta_if_ablated",
        "episode_rank_correlation_if_ablated",
        "planner_rank_correlation_if_ablated",
        "planner_order_changed_if_ablated",
        "sensitivity_rank",
    )
    rows: list[dict[str, Any]] = []
    sensitivity_rows = payload.get("weight_sensitivity")
    sensitivity_by_component = (
        {str(row.get("weight_name")): row for row in sensitivity_rows if isinstance(row, dict)}
        if isinstance(sensitivity_rows, list)
        else {}
    )
    if isinstance(calibrated, dict):
        configured = payload.get("configured_weights", {})
        for name in WEIGHT_NAMES:
            configured_value = float((configured or {}).get(name, 1.0))
            calibrated_value = float(calibrated.get(name, configured_value))
            sensitivity = sensitivity_by_component.get(name, {})
            rows.append(
                {
                    "component": name,
                    "metric_name": sensitivity.get("metric_name", ""),
                    "configured_weight": configured_value,
                    "configured_weight_share": float(
                        sensitivity.get("configured_weight_share", 0.0)
                    ),
                    "calibrated_weight": calibrated_value,
                    "delta": calibrated_value - configured_value,
                    "mean_abs_contribution": float(sensitivity.get("mean_abs_contribution", 0.0)),
                    "mean_abs_score_delta_if_ablated": float(
                        sensitivity.get("mean_abs_score_delta_if_ablated", 0.0)
                    ),
                    "episode_rank_correlation_if_ablated": float(
                        sensitivity.get("episode_rank_correlation_if_ablated", 1.0)
                    ),
                    "planner_rank_correlation_if_ablated": float(
                        sensitivity.get("planner_rank_correlation_if_ablated", 1.0)
                    ),
                    "planner_order_changed_if_ablated": bool(
                        sensitivity.get("planner_order_changed_if_ablated")
                    ),
                    "sensitivity_rank": int(sensitivity.get("sensitivity_rank", 0) or 0),
                }
            )
    _write_csv(csv_path, [{key: row.get(key) for key in headers} for row in rows])
    return json_path, md_path, csv_path
