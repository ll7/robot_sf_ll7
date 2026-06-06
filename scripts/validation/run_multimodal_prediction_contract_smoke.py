#!/usr/bin/env python3
"""Run the issue #2496 multimodal prediction contract smoke."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import yaml

from robot_sf.nav.predictive_types import ProbabilisticPrediction, TrajectoryDistribution

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/benchmarks/multimodal_prediction_smoke_issue_2496.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "output/benchmarks/multimodal_prediction_smoke_issue_2496"
FAIL_CLOSED_STATUSES = {"not_available", "failed"}
PASSING_STATUS = "native"


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_config(path: Path) -> dict[str, Any]:
    """Load and validate the smoke YAML as a mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _trajectory(
    *,
    pedestrian_id: int,
    hypothesis_index: int,
    confidence: float,
    horizon_steps: int,
    prediction_dt: float,
) -> TrajectoryDistribution:
    """Build one deterministic prediction hypothesis for the smoke fixture."""
    x_values = np.arange(1, horizon_steps + 1, dtype=np.float32) * np.float32(prediction_dt)
    lateral_offset = np.float32(0.1 * hypothesis_index)
    mean = np.column_stack((x_values, np.full(horizon_steps, lateral_offset, dtype=np.float32)))
    std = np.full((horizon_steps, 2), 0.05 + (0.01 * hypothesis_index), dtype=np.float32)
    return TrajectoryDistribution(
        mean=mean,
        std=std,
        confidence=confidence,
        pedestrian_id=pedestrian_id,
    )


def _prediction(
    *,
    pedestrian_id: int,
    confidences: list[float],
    horizon_steps: int,
    prediction_dt: float,
) -> ProbabilisticPrediction:
    """Build a prediction container used to validate the trace fixture fields."""
    predictions = [
        _trajectory(
            pedestrian_id=pedestrian_id,
            hypothesis_index=index,
            confidence=confidence,
            horizon_steps=horizon_steps,
            prediction_dt=prediction_dt,
        )
        for index, confidence in enumerate(confidences)
    ]
    return ProbabilisticPrediction(
        predictions=predictions,
        prediction_horizon=horizon_steps * prediction_dt,
        prediction_dt=prediction_dt,
        sample_count=len(predictions) or 1,
        metadata={"contract_smoke": "issue_2496"},
    )


def _group_confidences(
    prediction: ProbabilisticPrediction | None,
    pedestrian_ids: list[int],
) -> dict[str, list[float]]:
    """Return confidence vectors grouped by pedestrian id."""
    grouped: dict[str, list[float]] = {str(pedestrian_id): [] for pedestrian_id in pedestrian_ids}
    if prediction is None:
        return grouped
    for trajectory in prediction.predictions:
        grouped.setdefault(str(trajectory.pedestrian_id), []).append(float(trajectory.confidence))
    return grouped


def _hypothesis_counts(confidences_by_pedestrian: dict[str, list[float]]) -> dict[str, int]:
    """Return hypothesis counts for each pedestrian."""
    return {
        pedestrian_id: len(confidences)
        for pedestrian_id, confidences in confidences_by_pedestrian.items()
    }


def _base_metrics(*, success: bool) -> dict[str, Any]:
    """Return deterministic smoke metrics for a contract row."""
    return {
        "collision": False,
        "success": success,
        "timeout": False,
        "min_pedestrian_distance": 1.25 if success else None,
        "time_to_goal": 8.0 if success else None,
    }


def _comparator_row(
    comparator: dict[str, Any],
    *,
    pedestrian_ids: list[int],
    horizon_steps: int,
    prediction_dt: float,
) -> dict[str, Any]:
    """Build one passing comparator row from the smoke config."""
    confidences = [float(value) for value in comparator.get("hypothesis_confidences", [])]
    prediction = (
        None
        if not confidences
        else _prediction(
            pedestrian_id=pedestrian_ids[0],
            confidences=confidences,
            horizon_steps=horizon_steps,
            prediction_dt=prediction_dt,
        )
    )
    confidence_vectors = _group_confidences(prediction, pedestrian_ids)
    return {
        "planner_key": comparator["key"],
        "prediction_mode": comparator["prediction_mode"],
        "prediction_source": comparator["prediction_source"],
        "prediction_horizon_steps": 0 if prediction is None else horizon_steps,
        "prediction_dt": prediction_dt,
        "prediction_sample_count": 0 if prediction is None else prediction.sample_count,
        "hypothesis_count_per_pedestrian": _hypothesis_counts(confidence_vectors),
        "hypothesis_confidence_vector": confidence_vectors,
        "selected_or_weighted_hypothesis_id": comparator.get("selected_or_weighted_hypothesis_id"),
        "fallback_or_degraded_reason": None,
        "readiness_status": comparator.get("required_status", PASSING_STATUS),
        "expected_fail_closed": False,
        **_base_metrics(success=True),
    }


def _fail_closed_row(
    case: dict[str, Any],
    *,
    pedestrian_ids: list[int],
    prediction_dt: float,
) -> dict[str, Any]:
    """Build one expected fail-closed row from the smoke config."""
    confidence_vectors = {str(pedestrian_id): [] for pedestrian_id in pedestrian_ids}
    return {
        "planner_key": case["key"],
        "prediction_mode": case["prediction_mode"],
        "prediction_source": case["prediction_source"],
        "prediction_horizon_steps": 0,
        "prediction_dt": prediction_dt,
        "prediction_sample_count": 0,
        "hypothesis_count_per_pedestrian": _hypothesis_counts(confidence_vectors),
        "hypothesis_confidence_vector": confidence_vectors,
        "selected_or_weighted_hypothesis_id": None,
        "fallback_or_degraded_reason": case["fallback_or_degraded_reason"],
        "readiness_status": case["expected_status"],
        "expected_fail_closed": True,
        **_base_metrics(success=False),
    }


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build all smoke rows from the loaded config."""
    defaults = config.get("prediction_defaults", {})
    pedestrian_ids = [int(value) for value in defaults.get("pedestrian_ids", [0])]
    horizon_steps = int(defaults.get("prediction_horizon_steps", 1))
    prediction_dt = float(defaults.get("prediction_dt", 0.1))
    if horizon_steps < 1:
        raise ValueError("prediction_horizon_steps must be at least 1")
    rows = [
        _comparator_row(
            comparator,
            pedestrian_ids=pedestrian_ids,
            horizon_steps=horizon_steps,
            prediction_dt=prediction_dt,
        )
        for comparator in config.get("comparators", [])
    ]
    rows.extend(
        _fail_closed_row(case, pedestrian_ids=pedestrian_ids, prediction_dt=prediction_dt)
        for case in config.get("fail_closed_cases", [])
    )
    return rows


def _confidence_sum_failures(row: dict[str, Any]) -> list[str]:
    """Return confidence-normalization failures for one row."""
    failures = []
    vectors = row.get("hypothesis_confidence_vector", {})
    counts = row.get("hypothesis_count_per_pedestrian", {})
    if not isinstance(vectors, dict) or not isinstance(counts, dict):
        return ["hypothesis confidence/count fields must be mappings"]
    for pedestrian_id, confidences in vectors.items():
        count = counts.get(pedestrian_id)
        if count != len(confidences):
            failures.append(f"{row['planner_key']}: count mismatch for pedestrian {pedestrian_id}")
        if count:
            confidence_sum = sum(float(value) for value in confidences)
            if not np.isclose(confidence_sum, 1.0):
                failures.append(
                    f"{row['planner_key']}: confidence sum {confidence_sum:.6f} "
                    f"for pedestrian {pedestrian_id}"
                )
    return failures


def _sample_count_failures(row: dict[str, Any]) -> list[str]:
    """Return sample-count consistency failures for one native prediction row."""
    counts = row.get("hypothesis_count_per_pedestrian", {})
    if not isinstance(counts, dict):
        return ["hypothesis_count_per_pedestrian must be a mapping"]
    positive_counts = {int(count) for count in counts.values() if int(count) > 0}
    sample_count = int(row.get("prediction_sample_count", 0))
    if not positive_counts:
        return [] if sample_count == 0 else [f"{row['planner_key']}: unexpected sample count"]
    if len(positive_counts) != 1:
        return [f"{row['planner_key']}: inconsistent hypothesis counts across pedestrians"]
    expected = positive_counts.pop()
    if sample_count != expected:
        return [
            f"{row['planner_key']}: prediction_sample_count={sample_count} "
            f"does not match hypothesis count {expected}"
        ]
    return []


def validate_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    """Validate row fields, native rows, and fail-closed rows."""
    required_fields = set(config.get("required_trace_fields", []))
    required_status_values = set(config.get("required_status_values", []))
    failures: list[str] = []
    for row in rows:
        planner_key = row.get("planner_key", "<missing>")
        missing = sorted(required_fields.difference(row))
        if missing:
            failures.append(f"{planner_key}: missing required fields {missing}")
        status = row.get("readiness_status")
        if status not in required_status_values:
            failures.append(f"{planner_key}: unsupported readiness_status={status!r}")
        if row.get("expected_fail_closed"):
            if status not in FAIL_CLOSED_STATUSES:
                failures.append(f"{planner_key}: expected fail-closed status, got {status!r}")
            if not row.get("fallback_or_degraded_reason"):
                failures.append(f"{planner_key}: fail-closed row needs a reason")
            continue
        if status != PASSING_STATUS:
            failures.append(f"{planner_key}: native comparator used status {status!r}")
        if row.get("fallback_or_degraded_reason") is not None:
            failures.append(f"{planner_key}: native comparator used fallback/degraded reason")
        failures.extend(_confidence_sum_failures(row))
        failures.extend(_sample_count_failures(row))
    return failures


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write smoke rows as JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def build_summary(
    *,
    config_path: Path,
    output_root: Path,
    rows_path: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    failures: list[str],
) -> dict[str, Any]:
    """Build the machine-readable smoke summary."""
    statuses: dict[str, int] = defaultdict(int)
    for row in rows:
        statuses[str(row.get("readiness_status"))] += 1
    return {
        "schema_version": "multimodal-prediction-contract-smoke.v1",
        "issue": 2496,
        "parent_issue": 2476,
        "benchmark_evidence": False,
        "passed": not failures,
        "config": _repo_relative(config_path),
        "parent_contract": config.get("parent_contract"),
        "output_root": _repo_relative(output_root),
        "rows_path": _repo_relative(rows_path),
        "row_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "comparator_keys": [row["planner_key"] for row in rows if not row["expected_fail_closed"]],
        "fail_closed_keys": [row["planner_key"] for row in rows if row["expected_fail_closed"]],
        "required_trace_fields": config.get("required_trace_fields", []),
        "failures": failures,
    }


def render_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Render a concise Markdown smoke report."""
    lines = [
        "# Multimodal prediction contract smoke",
        "",
        f"- Result: {'PASS' if summary['passed'] else 'FAIL'}",
        f"- Rows: `{summary['rows_path']}`",
        f"- Benchmark evidence: `{str(summary['benchmark_evidence']).lower()}`",
        "",
        "| Row | Status | Mode | Hypotheses | Reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        counts = row["hypothesis_count_per_pedestrian"]
        max_count = max(counts.values()) if counts else 0
        reason = row["fallback_or_degraded_reason"] or ""
        lines.append(
            f"| {row['planner_key']} | {row['readiness_status']} | "
            f"{row['prediction_mode']} | {max_count} | {reason} |"
        )
    if summary["failures"]:
        lines.append("")
        lines.append("## Failures")
        lines.extend(f"- {failure}" for failure in summary["failures"])
    lines.append("")
    lines.append(
        "This smoke is contract evidence only. It does not measure planner quality or prediction "
        "benefit."
    )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--github-step-summary",
        type=Path,
        default=Path(os.environ["GITHUB_STEP_SUMMARY"])
        if os.environ.get("GITHUB_STEP_SUMMARY")
        else None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the contract smoke and write JSON/Markdown artifacts."""
    args = _build_parser().parse_args(argv)
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path)
    rows = build_rows(config)
    failures = validate_rows(config, rows)

    rows_path = output_root / "rows.jsonl"
    summary_path = output_root / "summary.json"
    markdown_path = output_root / "summary.md"
    _write_jsonl(rows_path, rows)
    summary = build_summary(
        config_path=config_path,
        output_root=output_root,
        rows_path=rows_path,
        config=config,
        rows=rows,
        failures=failures,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown = render_markdown(summary, rows)
    markdown_path.write_text(markdown, encoding="utf-8")
    if args.github_step_summary is not None:
        args.github_step_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.github_step_summary.open("a", encoding="utf-8") as handle:
            handle.write(markdown)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
