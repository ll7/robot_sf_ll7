"""Pedestrian-model sensitivity provenance and CPU smoke report helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf.benchmark.utils import _config_hash
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_TOTAL_FORCE_V1,
    SOCIAL_FORCE_DEFAULT,
    normalize_pedestrian_model,
)

SCHEMA_VERSION = "ped_model_sensitivity.v1"
PEDESTRIAN_MODEL_SCHEMA_VERSION = "pedestrian-model.v1"
CLAIM_BOUNDARY = (
    "CPU diagnostic sensitivity harness. No new training. Development-model axis is "
    "declared policy provenance unless backed by a training artifact."
)
DEFAULT_MODELS = (SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1)

CSV_COLUMNS = (
    "development_model",
    "evaluation_model",
    "planner_key",
    "algo",
    "episodes",
    "success_incidence",
    "collision_incidence",
    "fallback_degraded_rows",
    "status",
)


def resolve_development_pedestrian_model(
    policy_cfg: Mapping[str, Any] | None,
    *,
    algorithm_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Return declared policy development pedestrian model or ``unknown``."""

    for source in (policy_cfg, algorithm_metadata):
        if not isinstance(source, Mapping):
            continue
        for key in (
            "development_pedestrian_model",
            "development_model",
            "human_model_variant",
        ):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return "unknown"


def build_pedestrian_model_provenance(
    *,
    sim_config: Any,
    policy_cfg: Mapping[str, Any] | None,
    algorithm_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build additive pedestrian-model provenance for one benchmark episode row.

    Returns:
        Nested provenance payload suitable for the benchmark episode record.
    """

    evaluation_model = normalize_pedestrian_model(getattr(sim_config, "pedestrian_model", None))
    development_model = resolve_development_pedestrian_model(
        policy_cfg, algorithm_metadata=algorithm_metadata
    )
    selector_payload = {
        "development_model": development_model,
        "evaluation_model": evaluation_model,
        "source": "simulation_config.pedestrian_model",
    }
    return {
        "schema_version": PEDESTRIAN_MODEL_SCHEMA_VERSION,
        **selector_payload,
        "selector_config_hash": _config_hash(selector_payload),
        "fallback_degraded_status": _fallback_degraded_status(algorithm_metadata),
        "claim_boundary": "diagnostic_cpu_sensitivity_no_training",
    }


def attach_pedestrian_model_fields(record: dict[str, Any], provenance: Mapping[str, Any]) -> None:
    """Attach nested and flattened pedestrian-model fields to an episode row."""

    record["pedestrian_model"] = dict(provenance)
    record["development_pedestrian_model"] = str(provenance["development_model"])
    record["evaluation_pedestrian_model"] = str(provenance["evaluation_model"])


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read benchmark JSONL episode records from ``path``.

    Returns:
        Parsed episode records in file order.
    """

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object record in {path}")
            records.append(payload)
    return records


def build_sensitivity_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    development_models: Sequence[str],
    evaluation_models: Sequence[str],
    planner_key: str,
    algo: str,
) -> dict[str, Any]:
    """Aggregate episode rows into a deterministic development x evaluation matrix.

    Returns:
        Summary payload containing the issue metadata and one cell for each requested model pair.
    """

    normalized_development = [_normalize_matrix_model(model) for model in development_models]
    normalized_evaluation = [_normalize_matrix_model(model) for model in evaluation_models]
    record_list = [dict(record) for record in records]
    cells: list[dict[str, Any]] = []
    for development_model in normalized_development:
        for evaluation_model in normalized_evaluation:
            matching = [
                record
                for record in record_list
                if str(record.get("development_pedestrian_model", "unknown")) == development_model
                and str(record.get("evaluation_pedestrian_model", "unknown")) == evaluation_model
            ]
            cells.append(
                _summarize_cell(
                    matching,
                    development_model=development_model,
                    evaluation_model=evaluation_model,
                    planner_key=planner_key,
                    algo=algo,
                )
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 3950,
        "claim_boundary": CLAIM_BOUNDARY,
        "models": sorted(set(normalized_development) | set(normalized_evaluation)),
        "cells": cells,
    }


def write_sensitivity_report(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write summary JSON, matrix CSV, and Markdown README for a sensitivity report.

    Returns:
        Mapping of artifact labels to written filesystem paths.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "sensitivity_matrix.csv"
    readme_path = out_dir / "README.md"

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for cell in summary.get("cells", []):
            writer.writerow({column: cell.get(column) for column in CSV_COLUMNS})
    readme_path.write_text(_render_readme(summary), encoding="utf-8")
    return {
        "summary_json": str(summary_path),
        "sensitivity_matrix_csv": str(csv_path),
        "readme": str(readme_path),
    }


def _normalize_matrix_model(model: str) -> str:
    if str(model).strip() == "unknown":
        return "unknown"
    return normalize_pedestrian_model(str(model))


def _summarize_cell(
    records: Sequence[Mapping[str, Any]],
    *,
    development_model: str,
    evaluation_model: str,
    planner_key: str,
    algo: str,
) -> dict[str, Any]:
    episodes = len(records)
    fallback_degraded_rows = sum(1 for record in records if _record_is_fallback_or_degraded(record))
    if episodes == 0:
        return {
            "development_model": development_model,
            "evaluation_model": evaluation_model,
            "planner_key": planner_key,
            "algo": algo,
            "episodes": 0,
            "success_incidence": None,
            "collision_incidence": None,
            "fallback_degraded_rows": 0,
            "status": "unavailable",
        }
    successes = sum(1 for record in records if _record_success(record))
    collisions = sum(1 for record in records if _record_collision(record))
    return {
        "development_model": development_model,
        "evaluation_model": evaluation_model,
        "planner_key": planner_key,
        "algo": algo,
        "episodes": episodes,
        "success_incidence": successes / episodes,
        "collision_incidence": collisions / episodes,
        "fallback_degraded_rows": fallback_degraded_rows,
        "status": "ok" if fallback_degraded_rows == 0 else "degraded",
    }


def _record_success(record: Mapping[str, Any]) -> bool:
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping) and "success" in outcome:
        return bool(outcome["success"])
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping) and "success" in metrics:
        return float(metrics["success"]) > 0.0
    return str(record.get("status")) == "success"


def _record_collision(record: Mapping[str, Any]) -> bool:
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping) and "collision" in outcome:
        return bool(outcome["collision"])
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping) and "collisions" in metrics:
        return float(metrics["collisions"]) > 0.0
    return str(record.get("termination_reason")) == "collision"


def _record_is_fallback_or_degraded(record: Mapping[str, Any]) -> bool:
    pedestrian_model = record.get("pedestrian_model")
    if isinstance(pedestrian_model, Mapping):
        status = pedestrian_model.get("fallback_degraded_status")
        if status not in {None, "native"}:
            return True
    return _fallback_degraded_status(record.get("algorithm_metadata")) != "native"


def _fallback_degraded_status(metadata: Mapping[str, Any] | None) -> str:
    if not isinstance(metadata, Mapping):
        return "native"
    candidates = [
        metadata.get("status"),
        metadata.get("execution_mode"),
        metadata.get("availability_status"),
    ]
    runtime = metadata.get("planner_runtime")
    if isinstance(runtime, Mapping):
        candidates.extend(
            [
                runtime.get("status"),
                runtime.get("execution_mode"),
                runtime.get("availability_status"),
            ]
        )
    for candidate in candidates:
        if str(candidate).strip().lower() in {"fallback", "degraded", "not_available", "failed"}:
            return str(candidate).strip().lower()
    return "native"


def _render_readme(summary: Mapping[str, Any]) -> str:
    rows = [
        "| Development model | Evaluation model | Episodes | Success incidence | "
        "Collision incidence | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for cell in summary.get("cells", []):
        rows.append(
            "| {development_model} | {evaluation_model} | {episodes} | {success} | "
            "{collision} | {status} |".format(
                development_model=cell.get("development_model"),
                evaluation_model=cell.get("evaluation_model"),
                episodes=cell.get("episodes"),
                success=_format_rate(cell.get("success_incidence")),
                collision=_format_rate(cell.get("collision_incidence")),
                status=cell.get("status"),
            )
        )
    return (
        "# Issue #3950 Pedestrian-Model Sensitivity Smoke\n\n"
        "This is a CPU-only diagnostic smoke report. It does not train a policy and does not "
        "make paper-facing claims.\n\n"
        f"Claim boundary: {summary.get('claim_boundary', CLAIM_BOUNDARY)}\n\n"
        "The development-model axis is declared policy provenance unless a training artifact "
        "explicitly supports it. The evaluation-model axis is the active "
        "`simulation_config.pedestrian_model` selector.\n\n" + "\n".join(rows) + "\n"
    )


def _format_rate(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6f}"
