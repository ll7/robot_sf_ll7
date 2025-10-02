"""Summary writers for multi-extractor training runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.training.multi_extractor_models import ExtractorRunRecord, TrainingRunSummary

SUMMARY_JSON_FILENAME = "summary.json"
SUMMARY_MARKDOWN_FILENAME = "summary.md"


def write_summary_artifacts(*, summary: TrainingRunSummary, destination: Path) -> dict[str, Path]:
    """Persist JSON and Markdown summaries and return their paths."""

    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / SUMMARY_JSON_FILENAME
    markdown_path = destination / SUMMARY_MARKDOWN_FILENAME

    json_payload = summary.to_dict()
    json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=False), encoding="utf-8")

    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")

    return {"json": json_path, "markdown": markdown_path}


def _render_markdown(summary: TrainingRunSummary) -> str:
    lines: list[str] = []
    lines.append("# Multi-Extractor Training Summary")
    lines.append(f"Run ID: {summary.run_id}")
    lines.append(f"Created At: {summary.created_at}")
    lines.append(f"Output Directory: {summary.output_root}")
    lines.append("")

    lines.extend(_render_hardware_overview(summary))
    lines.append("")
    lines.extend(_render_extractor_table(summary.extractor_results))
    lines.append("")
    lines.extend(_render_failures(summary.extractor_results))
    lines.append("")
    lines.extend(_render_aggregate_metrics(summary.aggregate_metrics))
    lines.append("")
    lines.extend(_render_reproducibility(summary))
    lines.append("")

    return "\n".join(lines)


def _render_hardware_overview(summary: TrainingRunSummary) -> list[str]:
    lines = ["## Hardware Overview"]
    lines.append("| Platform | Arch | GPU Model | CUDA Version | Python | Workers |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for profile in summary.hardware_overview:
        gpu_model = profile.gpu_model or "N/A"
        cuda_version = profile.cuda_version or "N/A"
        lines.append(
            f"| {profile.platform} | {profile.arch} | {gpu_model} | {cuda_version} | "
            f"{profile.python_version} | {profile.workers} |"
        )
    return lines


def _render_extractor_table(records: list[ExtractorRunRecord]) -> list[str]:
    lines = ["## Extractor Results"]
    lines.append("| Extractor Name | Status | Worker Mode | Duration (s) | Best Metric | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    for record in records:
        duration = (
            f"{record.duration_seconds:.1f}" if record.duration_seconds is not None else "N/A"
        )
        metric_name, metric_value = _best_metric(record)
        notes = record.reason or ""
        lines.append(
            f"| {record.config_name} | {record.status} | {record.worker_mode} | {duration} | "
            f"{metric_name}: {metric_value} | {notes} |"
        )
    return lines


def _best_metric(record: ExtractorRunRecord) -> tuple[str, str]:
    if record.metrics:
        key, value = next(iter(record.metrics.items()))
        if isinstance(value, int | float):
            formatted = f"{float(value):.3f}"
        else:
            formatted = str(value)
        return key, formatted
    return "N/A", "N/A"


def _render_failures(records: list[ExtractorRunRecord]) -> list[str]:
    lines = ["## Failures & Skips"]
    issues = [record for record in records if record.status != "success"]
    if not issues:
        lines.append("- None")
    else:
        for record in issues:
            reason = record.reason or "No reason provided"
            lines.append(f"- {record.config_name}: {reason}")
    return lines


def _render_aggregate_metrics(metrics: dict[str, float]) -> list[str]:
    lines = ["## Aggregated Metrics"]
    if not metrics:
        lines.append("- None recorded")
        return lines

    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for key, value in metrics.items():
        if isinstance(value, int | float):
            formatted = f"{float(value):.3f}"
        else:
            formatted = str(value)
        lines.append(f"| {key} | {formatted} |")
    return lines


def _render_reproducibility(summary: TrainingRunSummary) -> list[str]:
    lines = ["## Reproducibility"]
    lines.append(f"- JSON summary: `{SUMMARY_JSON_FILENAME}`")
    lines.append(f"- Markdown summary: `{SUMMARY_MARKDOWN_FILENAME}`")

    artifact_paths = set()
    for record in summary.extractor_results:
        for value in record.artifacts.values():
            artifact_paths.add(value)

    if artifact_paths:
        lines.append("- Extractor artifacts:")
        for path in sorted(artifact_paths):
            lines.append(f"  - `{path}`")
    else:
        lines.append("- Extractor artifacts: none recorded")
    return lines
