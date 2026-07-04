"""Certification-transfer probe helpers for issue #4207."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.sim.pedestrian_model_variants import (
    HSFM_TOTAL_FORCE_V1,
    SOCIAL_FORCE_DEFAULT,
    normalize_pedestrian_model,
)

# Repository root, used to emit portable repo-relative provenance paths in the
# committed evidence packet (issue #4324; same defect class as #4302). This module
# lives at ``robot_sf/benchmark/certification_transfer.py``, so the repo root is two
# parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE_SCHEMA_VERSION = "certification-transfer-probe.v1"
GATE_SPEC_SCHEMA_VERSION = "benchmark_release_gate_spec.v1"
REPORT_SCHEMA_VERSION = "certification-transfer-report.v1"
ISSUE = 4207
CLAIM_BOUNDARY = "diagnostic_certification_transfer_probe_no_deployment_claim"
DEFAULT_PEDESTRIAN_MODELS = (SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1)

# Near-field proxemic band (metres). ``social_force_default`` and ``hsfm_total_force_v1``
# differ only in pedestrian force dynamics, so their pass/fail decision can only diverge
# when the robot actually enters the pedestrian near field. A cell whose episodes never
# bring the robot within this band exercises no model-sensitive dynamics; a stable transfer
# status over such a cell is therefore vacuous, not evidence of certification robustness.
# The 5 m band mirrors the ``robot_ped_within_5m_frac`` comfort metric already used by the
# #4166-style gate spec.
INTERACTION_NEAR_FIELD_M = 5.0
INTERACTION_STATUSES = ("interacting", "non_interacting", "unknown")
# Metric keys the runner can aggregate per gate cell (see ``_aggregate_metrics``). A release gate
# whose ``metric`` is not in this set can never resolve to ``pass``/``fail``: every cell would be
# ``not_evaluable``, the vacuously-inconclusive trap the issue #4207 preflight forbids.
# ``preflight_gate_evaluability`` uses this set to fail closed with
# ``blocked_no_evaluable_gate_family`` before any simulation runs.
AGGREGATABLE_METRICS = frozenset(
    {
        "collision_rate",
        "success_rate",
        "near_miss_rate",
        "min_clearance_m",
        "proxemic_intrusion_rate",
        "robot_ped_within_5m_frac",
        "jerk_mean",
    }
)
PREFLIGHT_OK = "ok"
PREFLIGHT_BLOCKED_NO_EVALUABLE_GATE_FAMILY = "blocked_no_evaluable_gate_family"
TRAINED_PLANNER_STRUCTURAL_CLASSES = frozenset({"learned_policy", "predictive"})
TRAINED_PLANNER_ALGOS = frozenset({"ppo", "guarded_ppo", "prediction_planner"})
TRAINED_PLANNER_ELIGIBLE = "eligible"
TRAINED_PLANNER_NOT_A_TRAINED_PLANNER = "not_a_trained_planner"
TRAINED_PLANNER_EXCLUDED_MISSING_CHECKPOINT = "excluded_missing_checkpoint_or_config"
TRAINED_PLANNER_EXCLUDED_FALLBACK = "excluded_fallback_execution"
TRAINED_PLANNER_REQUIRED_FIELDS = ("algo_config", "checkpoint", "training_manifest")

GATE_CELL_COLUMNS = (
    "planner_key",
    "structural_class",
    "scenario_family",
    "evaluation_model",
    "gate_status",
    "interaction_status",
    "trained_planner_claim_status",
    "trained_planner_claim_exclusion",
    "failed_gate_ids",
    "not_evaluable_gate_ids",
    "episodes",
)
TRANSFER_MATRIX_COLUMNS = (
    "planner_key",
    "structural_class",
    "scenario_family",
    "certification_model",
    "evaluation_model",
    "certification_gate_status",
    "evaluation_gate_status",
    "transfer_status",
    "interaction_status",
    "interaction_exercised",
    "trained_planner_claim_status",
    "trained_planner_claim_exclusion",
    "flip_type",
    "failed_gate_ids",
    "not_evaluable_gate_ids",
)
METRIC_DELTA_COLUMNS = (
    "planner_key",
    "structural_class",
    "scenario_family",
    "metric",
    "certification_model",
    "evaluation_model",
    "certification_value",
    "evaluation_value",
    "delta",
)


def validate_probe_config(config: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Validate and normalize the pre-registered certification-transfer config.

    Returns:
        Normalized probe config with resolved scenario and algorithm config paths.
    """

    if config.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {PROBE_SCHEMA_VERSION!r}")
    if int(config.get("issue", 0)) != ISSUE:
        raise ValueError(f"issue must be {ISSUE}")
    if config.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError(f"claim_boundary must be {CLAIM_BOUNDARY!r}")
    if bool(config.get("paper_facing")):
        raise ValueError("issue #4207 first slice must be paper_facing: false")

    models = [
        normalize_pedestrian_model(str(model)) for model in config.get("pedestrian_models", [])
    ]
    if tuple(models) != DEFAULT_PEDESTRIAN_MODELS:
        raise ValueError(
            "pedestrian_models must predeclare "
            f"{list(DEFAULT_PEDESTRIAN_MODELS)!r} in certification-transfer order"
        )

    scenario_family = str(config.get("scenario_family", "")).strip()
    if not scenario_family:
        raise ValueError("scenario_family is required")
    scenario_matrix = _resolve_existing_path(config.get("scenario_matrix"), base_dir=base_dir)

    seed_policy = config.get("seed_policy")
    if not isinstance(seed_policy, Mapping) or seed_policy.get("mode") != "fixed-list":
        raise ValueError("seed_policy.mode must be fixed-list")
    seeds = seed_policy.get("seeds")
    if not isinstance(seeds, Sequence) or isinstance(seeds, str) or not seeds:
        raise ValueError("seed_policy.seeds must be a non-empty list")
    normalized_seeds = [int(seed) for seed in seeds]

    return {
        "name": str(config.get("name", "issue_4207_certification_transfer_probe")),
        "schema_version": PROBE_SCHEMA_VERSION,
        "issue": ISSUE,
        "paper_facing": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "pedestrian_models": models,
        "scenario_family": scenario_family,
        "scenario_matrix": str(scenario_matrix),
        "seed_policy": {"mode": "fixed-list", "seeds": normalized_seeds},
        "arms": _validate_arms(config.get("arms"), base_dir=base_dir),
        "horizon": int(config.get("horizon", 60)),
        "dt": float(config.get("dt", 0.1)),
        "workers": int(config.get("workers", 1)),
        "record_forces": bool(config.get("record_forces", False)),
        "resume": bool(config.get("resume", True)),
        "run_artifact_dir": str(config.get("run_artifact_dir", "output/benchmarks/issue_4207")),
        "trained_planner_claim_policy": {
            "schema_version": "trained-planner-claim-policy.v1",
            "claim": (
                "certification-transfer packet rows with fallback or missing checkpoints are "
                "excluded from trained-planner comparison claims."
            ),
            "eligible_status": TRAINED_PLANNER_ELIGIBLE,
            "excluded_statuses": [
                TRAINED_PLANNER_EXCLUDED_MISSING_CHECKPOINT,
                TRAINED_PLANNER_EXCLUDED_FALLBACK,
            ],
            "required_fields": list(TRAINED_PLANNER_REQUIRED_FIELDS),
        },
    }


def validate_gate_spec(gate_spec: Mapping[str, Any], *, scenario_family: str) -> dict[str, Any]:
    """Validate and normalize the provisional #4166-style release-gate spec.

    Returns:
        Normalized gate specification with scoped gate mappings.
    """

    if gate_spec.get("schema_version") != GATE_SPEC_SCHEMA_VERSION:
        raise ValueError(f"gate spec schema_version must be {GATE_SPEC_SCHEMA_VERSION!r}")
    gates = gate_spec.get("gates")
    if not isinstance(gates, Sequence) or isinstance(gates, str) or not gates:
        raise ValueError("gate spec must contain at least one gate")

    normalized_gates: list[dict[str, Any]] = []
    for gate in gates:
        if not isinstance(gate, Mapping):
            raise ValueError("each gate must be a mapping")
        scope = gate.get("scope") or {}
        scoped_family = scope.get("scenario_family") if isinstance(scope, Mapping) else None
        if scoped_family != scenario_family:
            raise ValueError(
                f"gate {gate.get('id')!r} scope.scenario_family must be {scenario_family!r}"
            )
        direction = str(gate.get("direction", "")).strip()
        if direction not in {"max", "min"}:
            raise ValueError(f"gate {gate.get('id')!r} direction must be 'max' or 'min'")
        normalized_gates.append(
            {
                "id": str(gate["id"]),
                "metric": str(gate["metric"]),
                "threshold": float(gate["threshold"]),
                "direction": direction,
                "category": str(gate.get("category", "unspecified")),
                "provenance": str(gate.get("provenance", "unspecified")),
                "required": bool(gate.get("required", True)),
                "scope": {"scenario_family": scenario_family},
            }
        )
    return {
        "schema_version": GATE_SPEC_SCHEMA_VERSION,
        "description": str(gate_spec.get("description", "")),
        "gates": normalized_gates,
    }


def preflight_gate_evaluability(
    probe_config: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Preflight-check that the declared scenario family has evaluable required gate metrics.

    The certification-transfer probe declares a single, hard-coded scenario family (it is not
    discovered dynamically at run time). A required release gate whose ``metric`` is not one the
    runner can aggregate (:data:`AGGREGATABLE_METRICS`) can never produce a ``pass``/``fail``
    decision: every cell would be ``not_evaluable``, the vacuously-inconclusive trap the issue
    #4207 plan forbids. Such a family is not evaluable; preflight fails closed with
    :data:`PREFLIGHT_BLOCKED_NO_EVALUABLE_GATE_FAMILY` instead of running a moot probe.

    Optional gates (``required: false``) with non-aggregatable metrics do not block: they surface
    as ``not_evaluable`` at evaluation time but do not drive the pass/fail decision.

    Args:
        probe_config: Normalized probe config (from :func:`validate_probe_config`).
        gate_spec: Normalized gate spec (from :func:`validate_gate_spec`).

    Returns:
        Preflight payload with ``status`` (``ok`` or ``blocked_no_evaluable_gate_family``), the
        declared ``scenario_family``, the runner's ``aggregatable_metrics``, the required gate
        metrics, and the blocking ``not_evaluable_gate_ids`` / ``not_evaluable_gate_metrics``
        (empty when the family is evaluable).
    """

    scenario_family = str(probe_config["scenario_family"])
    gates = list(gate_spec.get("gates") or [])
    required_gates = [gate for gate in gates if bool(gate.get("required", True))]
    required_metrics = sorted({str(gate["metric"]) for gate in required_gates})
    blocking = [
        {"gate_id": str(gate["id"]), "metric": str(gate["metric"])}
        for gate in required_gates
        if str(gate["metric"]) not in AGGREGATABLE_METRICS
    ]
    not_evaluable_gate_ids = sorted({entry["gate_id"] for entry in blocking})
    not_evaluable_gate_metrics = sorted({entry["metric"] for entry in blocking})
    blocked = bool(blocking)
    return {
        "status": PREFLIGHT_BLOCKED_NO_EVALUABLE_GATE_FAMILY if blocked else PREFLIGHT_OK,
        "scenario_family": scenario_family,
        "aggregatable_metrics": sorted(AGGREGATABLE_METRICS),
        "required_gate_metrics": required_metrics,
        "not_evaluable_gate_ids": not_evaluable_gate_ids,
        "not_evaluable_gate_metrics": not_evaluable_gate_metrics,
        "blocking_gates": blocking,
    }


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``.

    Returns:
        Parsed YAML mapping.
    """

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return payload


def build_certification_transfer_report(
    records: Iterable[Mapping[str, Any]],
    *,
    probe_config: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
    config_path: str | Path,
    gate_spec_path: str | Path,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a certification-transfer report from episode records.

    Returns:
        Summary payload with gate cells, transfer matrix, deltas, flips, and provenance.
    """

    config_path = Path(config_path)
    gate_spec_path = Path(gate_spec_path)
    normalized_config = validate_probe_config(probe_config, base_dir=config_path.parent)
    normalized_gate_spec = validate_gate_spec(
        gate_spec,
        scenario_family=normalized_config["scenario_family"],
    )
    record_list = [dict(record) for record in records]
    gate_cells = _build_gate_cells(
        record_list,
        arms=normalized_config["arms"],
        models=normalized_config["pedestrian_models"],
        scenario_family=normalized_config["scenario_family"],
        gates=normalized_gate_spec["gates"],
    )
    transfer_matrix = _build_transfer_matrix(
        gate_cells,
        arms=normalized_config["arms"],
        models=normalized_config["pedestrian_models"],
        scenario_family=normalized_config["scenario_family"],
    )
    metric_deltas = _build_metric_deltas(
        gate_cells,
        arms=normalized_config["arms"],
        models=normalized_config["pedestrian_models"],
        scenario_family=normalized_config["scenario_family"],
    )
    interaction_metric_summary = _interaction_metric_summary(gate_cells)
    flip_cases = [
        row
        for row in transfer_matrix
        if row["transfer_status"] in {"fragile_pass_to_fail", "conservative_fail_to_pass"}
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "issue": ISSUE,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "claim_boundary": CLAIM_BOUNDARY,
        "paper_facing": False,
        "config": {
            "path": _repo_relative_path(config_path),
            "sha256": _sha256_file(config_path),
            "schema_version": normalized_config["schema_version"],
        },
        "gate_spec": {
            "path": _repo_relative_path(gate_spec_path),
            "sha256": _sha256_file(gate_spec_path),
            "schema_version": normalized_gate_spec["schema_version"],
        },
        "pedestrian_models": normalized_config["pedestrian_models"],
        "scenario_family": normalized_config["scenario_family"],
        "seed_policy": normalized_config["seed_policy"],
        "arms": [
            {
                "key": arm["key"],
                "structural_class": arm["structural_class"],
                "algo": arm["algo"],
                "algo_config": arm.get("algo_config"),
                "development_pedestrian_model": arm["development_pedestrian_model"],
                "trained_planner_claim_status": arm["trained_planner_claim_status"],
                "trained_planner_claim_exclusion": arm["trained_planner_claim_exclusion"],
            }
            for arm in normalized_config["arms"]
        ],
        "trained_planner_claim_policy": normalized_config["trained_planner_claim_policy"],
        "trained_planner_readiness": _trained_planner_readiness(normalized_config["arms"]),
        "trained_planner_claim_status_counts": dict(
            Counter(row["trained_planner_claim_status"] for row in transfer_matrix)
        ),
        "gate_cells": gate_cells,
        "certification_transfer_matrix": transfer_matrix,
        "metric_deltas_by_model": metric_deltas,
        "flip_cases": flip_cases,
        "row_status_counts": dict(Counter(row["gate_status"] for row in gate_cells)),
        "transfer_status_counts": dict(Counter(row["transfer_status"] for row in transfer_matrix)),
        "interaction_status_counts": dict(
            Counter(row["interaction_status"] for row in transfer_matrix)
        ),
        "interaction_metric_summary": interaction_metric_summary,
        "model_sensitivity_exercised": any(row["interaction_exercised"] for row in transfer_matrix),
        "claim_boundary_notes": [
            "Diagnostic transfer probe only.",
            "Provisional gate thresholds are not certification approval.",
            "not_evaluable is never treated as pass.",
            "Model-transfer flips indicate model-assumption fragility in the gate decision.",
            "A stable transfer status over non_interacting cells is vacuous: the robot never "
            "entered the pedestrian near field, so the social-force model (SFM) / headed "
            "social-force model (HSFM) swap was not exercised and no certification-robustness "
            "conclusion follows.",
            "Learned or predictive arms that run through fallback execution or without a "
            "resolved trained checkpoint/config are excluded from trained-planner comparison "
            "claims.",
            "No deployment, real-world safety, or general planner-superiority claim follows.",
        ],
    }


def write_certification_transfer_evidence(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write compact issue #4207 evidence artifacts and checksums.

    Returns:
        Mapping from artifact label to written path.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": out_dir / "summary.json",
        "metadata_json": out_dir / "metadata.json",
        "certification_gate_cells_csv": out_dir / "certification_gate_cells.csv",
        "certification_transfer_matrix_csv": out_dir / "certification_transfer_matrix.csv",
        "metric_deltas_by_model_csv": out_dir / "metric_deltas_by_model.csv",
        "flip_cases_csv": out_dir / "flip_cases.csv",
        "claim_boundary_md": out_dir / "claim_boundary.md",
        "readme": out_dir / "README.md",
    }
    paths["summary_json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["metadata_json"].write_text(
        json.dumps(_metadata_payload(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(paths["certification_gate_cells_csv"], GATE_CELL_COLUMNS, report["gate_cells"])
    _write_csv(
        paths["certification_transfer_matrix_csv"],
        TRANSFER_MATRIX_COLUMNS,
        report["certification_transfer_matrix"],
    )
    _write_csv(
        paths["metric_deltas_by_model_csv"], METRIC_DELTA_COLUMNS, report["metric_deltas_by_model"]
    )
    _write_csv(paths["flip_cases_csv"], TRANSFER_MATRIX_COLUMNS, report["flip_cases"])
    paths["claim_boundary_md"].write_text(_claim_boundary_markdown(report), encoding="utf-8")
    paths["readme"].write_text(_readme_markdown(report), encoding="utf-8")
    checksum_path = out_dir / "SHA256SUMS"
    checksum_path.write_text(_checksums(paths.values()), encoding="utf-8")
    paths["sha256sums"] = checksum_path
    return {key: str(value) for key, value in paths.items()}


def _build_gate_cells(
    records: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    scenario_family: str,
    gates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for arm in arms:
        for evaluation_model in models:
            matching = [
                record
                for record in records
                if str(record.get("planner_key")) == arm["key"]
                and str(record.get("evaluation_pedestrian_model")) == evaluation_model
                and str(record.get("scenario_family", scenario_family)) == scenario_family
            ]
            metrics = _aggregate_metrics(matching)
            gate_results = [_evaluate_gate(metrics, gate) for gate in gates]
            failed = [result["id"] for result in gate_results if result["status"] == "fail"]
            not_evaluable = [
                result["id"] for result in gate_results if result["status"] == "not_evaluable"
            ]
            if not_evaluable:
                gate_status = "not_evaluable"
            elif failed:
                gate_status = "fail"
            else:
                gate_status = "pass"
            cells.append(
                {
                    "planner_key": arm["key"],
                    "structural_class": arm["structural_class"],
                    "scenario_family": scenario_family,
                    "evaluation_model": evaluation_model,
                    "gate_status": gate_status,
                    "interaction_status": classify_interaction_status(metrics),
                    "trained_planner_claim_status": arm["trained_planner_claim_status"],
                    "trained_planner_claim_exclusion": arm["trained_planner_claim_exclusion"],
                    "failed_gate_ids": ";".join(failed),
                    "not_evaluable_gate_ids": ";".join(not_evaluable),
                    "episodes": len(matching),
                    "metrics": metrics,
                    "gate_results": gate_results,
                    "certification_pedestrian_model": evaluation_model,
                    "development_pedestrian_model": arm.get(
                        "development_pedestrian_model", "unknown"
                    ),
                }
            )
    return cells


def _build_transfer_matrix(
    gate_cells: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    scenario_family: str,
) -> list[dict[str, Any]]:
    cells_by_key = {(cell["planner_key"], cell["evaluation_model"]): cell for cell in gate_cells}
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for certification_model in models:
            certification_cell = cells_by_key[(arm["key"], certification_model)]
            for evaluation_model in models:
                evaluation_cell = cells_by_key[(arm["key"], evaluation_model)]
                transfer_status, flip_type = _transfer_status(
                    str(certification_cell["gate_status"]),
                    str(evaluation_cell["gate_status"]),
                )
                interaction_status, interaction_exercised = _transfer_interaction(
                    str(certification_cell["interaction_status"]),
                    str(evaluation_cell["interaction_status"]),
                )
                rows.append(
                    {
                        "planner_key": arm["key"],
                        "structural_class": arm["structural_class"],
                        "scenario_family": scenario_family,
                        "certification_model": certification_model,
                        "evaluation_model": evaluation_model,
                        "certification_gate_status": certification_cell["gate_status"],
                        "evaluation_gate_status": evaluation_cell["gate_status"],
                        "transfer_status": transfer_status,
                        "interaction_status": interaction_status,
                        "interaction_exercised": interaction_exercised,
                        "trained_planner_claim_status": arm["trained_planner_claim_status"],
                        "trained_planner_claim_exclusion": arm["trained_planner_claim_exclusion"],
                        "flip_type": flip_type,
                        "failed_gate_ids": evaluation_cell["failed_gate_ids"],
                        "not_evaluable_gate_ids": evaluation_cell["not_evaluable_gate_ids"],
                    }
                )
    return rows


def _build_metric_deltas(
    gate_cells: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[Mapping[str, Any]],
    models: Sequence[str],
    scenario_family: str,
) -> list[dict[str, Any]]:
    if len(models) != 2:
        return []
    cells_by_key = {(cell["planner_key"], cell["evaluation_model"]): cell for cell in gate_cells}
    rows: list[dict[str, Any]] = []
    left_model, right_model = models
    for arm in arms:
        left_metrics = cells_by_key[(arm["key"], left_model)]["metrics"]
        right_metrics = cells_by_key[(arm["key"], right_model)]["metrics"]
        for metric in sorted(set(left_metrics) | set(right_metrics)):
            left_value = left_metrics.get(metric)
            right_value = right_metrics.get(metric)
            rows.append(
                {
                    "planner_key": arm["key"],
                    "structural_class": arm["structural_class"],
                    "scenario_family": scenario_family,
                    "metric": metric,
                    "certification_model": left_model,
                    "evaluation_model": right_model,
                    "certification_value": _csv_value(left_value),
                    "evaluation_value": _csv_value(right_value),
                    "delta": _csv_value(
                        None
                        if left_value is None or right_value is None
                        else right_value - left_value
                    ),
                }
            )
    return rows


def _aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    if not records:
        return {}
    metrics: dict[str, float | None] = {
        "collision_rate": _mean_optional([_collision_value(record) for record in records]),
        "success_rate": _mean_optional([_success_value(record) for record in records]),
        "near_miss_rate": _mean_named_metric(records, ("near_miss_rate", "near_misses")),
        "min_clearance_m": _min_named_metric(records, ("min_clearance_m", "min_clearance")),
        "proxemic_intrusion_rate": _mean_named_metric(
            records,
            ("proxemic_intrusion_rate", "proxemic_intrusions"),
        ),
        "robot_ped_within_5m_frac": _mean_named_metric(records, ("robot_ped_within_5m_frac",)),
        "jerk_mean": _mean_named_metric(records, ("jerk_mean", "mean_jerk")),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def _interaction_metric_summary(gate_cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize near-field interaction metrics across evaluated gate cells.

    Returns:
        Packet-level near-field proof fields for compact evidence artifacts.
    """

    within_values: list[float] = []
    clearance_values: list[float] = []
    interacting_cells: list[Mapping[str, Any]] = []

    for cell in gate_cells:
        metrics = cell.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        within = metrics.get("robot_ped_within_5m_frac")
        if within is not None:
            within_values.append(float(within))
        clearance = metrics.get("min_clearance_m")
        if clearance is not None:
            clearance_values.append(float(clearance))
        if cell.get("interaction_status") == "interacting":
            interacting_cells.append(cell)

    max_within = max(within_values) if within_values else None
    return {
        "cell_count": len(gate_cells),
        "interacting_cell_count": len(interacting_cells),
        "interacting_planner_keys": sorted(
            {str(cell.get("planner_key")) for cell in interacting_cells}
        ),
        "interacting_evaluation_models": sorted(
            {str(cell.get("evaluation_model")) for cell in interacting_cells}
        ),
        "max_robot_ped_within_5m_frac": max_within,
        "min_robot_ped_within_5m_frac": min(within_values) if within_values else None,
        "min_clearance_m": min(clearance_values) if clearance_values else None,
        "physics_near_field_confirmed": bool(max_within is not None and max_within > 0.0),
    }


def _evaluate_gate(metrics: Mapping[str, float | None], gate: Mapping[str, Any]) -> dict[str, Any]:
    metric = str(gate["metric"])
    value = metrics.get(metric)
    if value is None:
        return {"id": gate["id"], "metric": metric, "status": "not_evaluable", "value": None}
    threshold = float(gate["threshold"])
    direction = str(gate["direction"])
    passes = value <= threshold if direction == "max" else value >= threshold
    return {
        "id": gate["id"],
        "metric": metric,
        "status": "pass" if passes else "fail",
        "value": value,
        "threshold": threshold,
        "direction": direction,
    }


def classify_interaction_status(metrics: Mapping[str, float | None]) -> str:
    """Classify whether a cell's episodes exercised the robot-pedestrian near field.

    ``social_force_default`` and ``hsfm_total_force_v1`` only diverge when the robot enters
    the pedestrian near field, so a cell that never does cannot demonstrate certification
    fragility (its stable transfer status is vacuous).

    Returns:
        ``"interacting"`` when a proximity metric shows near-field contact,
        ``"non_interacting"`` when proximity metrics are present but show no contact, and
        ``"unknown"`` when no proximity metric is available (e.g. an empty/not_evaluable cell).
    """

    within_frac = metrics.get("robot_ped_within_5m_frac")
    min_clearance = metrics.get("min_clearance_m")
    if within_frac is None and min_clearance is None:
        return "unknown"
    entered_near_field = (within_frac is not None and within_frac > 0.0) or (
        min_clearance is not None and min_clearance < INTERACTION_NEAR_FIELD_M
    )
    return "interacting" if entered_near_field else "non_interacting"


def _transfer_interaction(certification_status: str, evaluation_status: str) -> tuple[str, bool]:
    """Combine two cells' interaction statuses into a transfer-row interaction verdict.

    Returns:
        The combined interaction status (``unknown`` dominates, then ``non_interacting``) and
        an ``interaction_exercised`` flag that is true only when both cells are interacting.
    """

    statuses = {certification_status, evaluation_status}
    if "unknown" in statuses:
        combined = "unknown"
    elif statuses == {"interacting"}:
        combined = "interacting"
    else:
        combined = "non_interacting"
    return combined, combined == "interacting"


def _transfer_status(certification_status: str, evaluation_status: str) -> tuple[str, str]:
    if "not_evaluable" in {certification_status, evaluation_status}:
        return "not_evaluable", ""
    if certification_status == "pass" and evaluation_status == "pass":
        return "stable_pass", ""
    if certification_status == "pass" and evaluation_status == "fail":
        return "fragile_pass_to_fail", "pass_to_fail"
    if certification_status == "fail" and evaluation_status == "pass":
        return "conservative_fail_to_pass", "fail_to_pass"
    return "stable_fail", ""


def _validate_arms(raw_arms: object, *, base_dir: Path) -> list[dict[str, Any]]:
    if (
        not isinstance(raw_arms, Sequence)
        or isinstance(raw_arms, str)
        or not (3 <= len(raw_arms) <= 4)
    ):
        raise ValueError("arms must predeclare three or four planner arms")

    normalized_arms: list[dict[str, Any]] = []
    for arm in raw_arms:
        if not isinstance(arm, Mapping):
            raise ValueError("each arm must be a mapping")
        key = str(arm.get("key", "")).strip()
        structural_class = str(arm.get("structural_class", "")).strip()
        algo = str(arm.get("algo", "")).strip()
        if not key or not structural_class or not algo:
            raise ValueError("each arm requires key, structural_class, and algo")
        normalized = _normalized_arm_base(
            arm, key=key, structural_class=structural_class, algo=algo
        )
        _copy_optional_arm_fields(normalized, arm, base_dir=base_dir)
        status, exclusion = _trained_planner_claim_status(normalized)
        normalized["trained_planner_claim_status"] = status
        normalized["trained_planner_claim_exclusion"] = exclusion
        normalized_arms.append(normalized)
    return normalized_arms


def _copy_optional_arm_fields(
    normalized: dict[str, Any],
    arm: Mapping[str, Any],
    *,
    base_dir: Path,
) -> None:
    for field in ("observation_mode", "observation_level"):
        if arm.get(field) is not None:
            normalized[field] = str(arm[field])
    for field in ("fallback_execution", "fallback_to_goal"):
        if arm.get(field) is not None:
            normalized[field] = bool(arm[field])
    _copy_optional_existing_path(normalized, arm, "algo_config", base_dir=base_dir)
    for artifact_field in ("checkpoint", "training_manifest"):
        _copy_optional_existing_path(
            normalized,
            arm,
            artifact_field,
            base_dir=base_dir,
        )


def _normalized_arm_base(
    arm: Mapping[str, Any],
    *,
    key: str,
    structural_class: str,
    algo: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "structural_class": structural_class,
        "algo": algo,
        "benchmark_profile": str(arm.get("benchmark_profile", "experimental")),
        "development_pedestrian_model": str(arm.get("development_pedestrian_model", "unknown")),
    }


def _copy_optional_existing_path(
    target: dict[str, Any],
    source: Mapping[str, Any],
    field: str,
    *,
    base_dir: Path,
) -> None:
    if source.get(field) is not None:
        target[field] = str(_resolve_existing_path(source.get(field), base_dir=base_dir))


def _trained_planner_claim_status(arm: Mapping[str, Any]) -> tuple[str, str]:
    """Classify whether an arm may support trained-planner comparison claims.

    Returns:
        Pair of claim status and optional exclusion reason.
    """

    structural_class = str(arm.get("structural_class", ""))
    algo = str(arm.get("algo", ""))
    if (
        structural_class not in TRAINED_PLANNER_STRUCTURAL_CLASSES
        and algo not in TRAINED_PLANNER_ALGOS
    ):
        return TRAINED_PLANNER_NOT_A_TRAINED_PLANNER, ""
    if bool(arm.get("fallback_execution")) or bool(arm.get("fallback_to_goal")):
        return TRAINED_PLANNER_EXCLUDED_FALLBACK, "fallback_execution"
    if not str(arm.get("algo_config", "")).strip():
        return TRAINED_PLANNER_EXCLUDED_MISSING_CHECKPOINT, "missing_checkpoint_or_config"
    return TRAINED_PLANNER_ELIGIBLE, ""


def _trained_planner_readiness(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        status = str(arm["trained_planner_claim_status"])
        missing_fields: list[str] = []
        if _is_trained_planner_arm(arm):
            if not str(arm.get("algo_config", "")).strip():
                missing_fields.append("algo_config")
            if not str(arm.get("checkpoint", "")).strip():
                missing_fields.append("checkpoint")
            if not str(arm.get("training_manifest", "")).strip():
                missing_fields.append("training_manifest")
        rows.append(
            {
                "planner_key": arm["key"],
                "structural_class": arm["structural_class"],
                "algo": arm["algo"],
                "eligible_for_trained_planner_claim": status == TRAINED_PLANNER_ELIGIBLE
                and not missing_fields,
                "readiness_status": _readiness_status(status, missing_fields),
                "missing_fields": missing_fields,
                "trained_planner_claim_status": status,
                "trained_planner_claim_exclusion": arm["trained_planner_claim_exclusion"],
            }
        )

    blockers = [
        row
        for row in rows
        if row["readiness_status"]
        in {"blocked_missing_artifact_provenance", "blocked_fallback_execution"}
    ]
    eligible_rows = [row for row in rows if row["eligible_for_trained_planner_claim"]]
    return {
        "schema_version": "trained-planner-readiness.v1",
        "claim_boundary": (
            "eligible rows may support trained-planner comparison claims only after a fresh "
            "certification-transfer run; excluded rows remain diagnostic-only"
        ),
        "required_fields": list(TRAINED_PLANNER_REQUIRED_FIELDS),
        "all_trained_planner_arms_ready": not blockers,
        "eligible_trained_planner_arm_count": len(eligible_rows),
        "blocker_count": len(blockers),
        "rows": rows,
    }


def _readiness_status(status: str, missing_fields: Sequence[str]) -> str:
    if status == TRAINED_PLANNER_NOT_A_TRAINED_PLANNER:
        return "not_required_baseline"
    if status == TRAINED_PLANNER_EXCLUDED_FALLBACK:
        return "blocked_fallback_execution"
    if missing_fields:
        return "blocked_missing_artifact_provenance"
    return "ready_for_fresh_probe"


def _is_trained_planner_arm(arm: Mapping[str, Any]) -> bool:
    return (
        str(arm.get("structural_class", "")) in TRAINED_PLANNER_STRUCTURAL_CLASSES
        or str(arm.get("algo", "")) in TRAINED_PLANNER_ALGOS
    )


def _collision_value(record: Mapping[str, Any]) -> float | None:
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping) and outcome.get("collision") is not None:
        return 1.0 if bool(outcome["collision"]) else 0.0
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        for key in ("collision_rate", "collisions", "collision"):
            if metrics.get(key) is not None:
                return float(metrics[key])
    return None


def _success_value(record: Mapping[str, Any]) -> float | None:
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping) and outcome.get("success") is not None:
        return 1.0 if bool(outcome["success"]) else 0.0
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        for key in ("success_rate", "success"):
            if metrics.get(key) is not None:
                return float(metrics[key])
    return None


def _mean_named_metric(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> float | None:
    return _mean_optional(_metric_values(records, names))


def _min_named_metric(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> float | None:
    values = [value for value in _metric_values(records, names) if value is not None]
    return min(values) if values else None


def _metric_values(
    records: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> list[float | None]:
    values: list[float | None] = []
    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            values.append(None)
            continue
        value = next((metrics[name] for name in names if metrics.get(name) is not None), None)
        values.append(None if value is None else float(value))
    return values


def _mean_optional(values: Sequence[float | None]) -> float | None:
    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return sum(concrete) / len(concrete)


def _resolve_existing_path(raw_path: object, *, base_dir: Path) -> Path:
    if raw_path is None or not str(raw_path).strip():
        raise ValueError("path value is required")
    path = Path(str(raw_path))
    candidates = [path] if path.is_absolute() else [base_dir / path, Path.cwd() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"required path does not exist: {raw_path}")


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _metadata_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report["schema_version"],
        "issue": report["issue"],
        "generated_at_utc": report["generated_at_utc"],
        "claim_boundary": report["claim_boundary"],
        "config": report["config"],
        "gate_spec": report["gate_spec"],
        "pedestrian_models": report["pedestrian_models"],
        "scenario_family": report["scenario_family"],
        "seed_policy": report["seed_policy"],
        "arms": report["arms"],
        "trained_planner_claim_policy": report["trained_planner_claim_policy"],
        "trained_planner_readiness": report["trained_planner_readiness"],
        "trained_planner_claim_status_counts": report["trained_planner_claim_status_counts"],
        "row_status_counts": report["row_status_counts"],
        "transfer_status_counts": report["transfer_status_counts"],
        "interaction_status_counts": report["interaction_status_counts"],
        "interaction_metric_summary": report["interaction_metric_summary"],
        "model_sensitivity_exercised": report["model_sensitivity_exercised"],
    }


def _claim_boundary_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Claim Boundary",
        "",
        "- Diagnostic certification-transfer probe only.",
        "- Provisional gates are reporting thresholds, not certification approval.",
        "- `not_evaluable` cells are fail-closed and never count as `pass`.",
        "- `non_interacting` cells (robot never inside the 5 m pedestrian near field) cannot "
        "demonstrate certification robustness; a stable status over them is vacuous, because the "
        "social-force model (SFM) / headed social-force model (HSFM) swap was never exercised.",
        "- Learned or predictive arms that run through fallback execution or without a resolved "
        "trained checkpoint/config are excluded from trained-planner comparison claims.",
        "- Pass/fail flips are a result: model-assumption fragility in the certification decision.",
        "- No full benchmark campaign, Slurm or GPU submission, retraining, deployment claim, "
        "real-world safety claim, or paper/dissertation claim promotion is included.",
        "",
        f"Claim boundary token: `{report['claim_boundary']}`",
    ]
    return "\n".join(lines) + "\n"


def _readme_markdown(report: Mapping[str, Any]) -> str:
    interaction_metrics = report["interaction_metric_summary"]
    lines = [
        "# Issue #4207 Certification-Transfer Probe Evidence",
        "",
        "This compact packet records a CPU diagnostic probe for certification-transfer "
        "between `social_force_default` and `hsfm_total_force_v1` pedestrian models.",
        "",
        "The provisional release gates are not deployment approval. Missing gate metrics are "
        "`not_evaluable`, never `pass`. Transfer flips are reported as model-assumption "
        "fragility in the gate decision, not as a failed experiment.",
        "",
        "## Counts",
        "",
        f"- Gate status counts: `{dict(report['row_status_counts'])}`",
        f"- Transfer status counts: `{dict(report['transfer_status_counts'])}`",
        f"- Interaction status counts: `{dict(report['interaction_status_counts'])}`",
        f"- Trained-planner claim status counts: "
        f"`{dict(report['trained_planner_claim_status_counts'])}`",
        f"- Trained-planner readiness: "
        f"`{report['trained_planner_readiness']['eligible_trained_planner_arm_count']}` "
        f"eligible, `{report['trained_planner_readiness']['blocker_count']}` blocked",
        f"- Physics near-field confirmed: `{interaction_metrics['physics_near_field_confirmed']}`",
        f"- Max robot-pedestrian within-5m fraction: "
        f"`{interaction_metrics['max_robot_ped_within_5m_frac']}`",
        f"- Minimum clearance: `{interaction_metrics['min_clearance_m']}`",
        f"- Interacting gate cells: `{interaction_metrics['interacting_cell_count']}` "
        f"of `{interaction_metrics['cell_count']}`",
        f"- Model sensitivity exercised: `{report['model_sensitivity_exercised']}`",
        f"- Flip cases: `{len(report['flip_cases'])}`",
        "",
        "If `Model sensitivity exercised` is `false`, every transfer cell is `non_interacting` "
        "or `unknown`: the robot never entered the pedestrian near field, so the stable statuses "
        "above are vacuous and do not demonstrate certification robustness.",
        "Learned or predictive arms with `excluded_missing_checkpoint_or_config` or "
        "`excluded_fallback_execution` trained-planner claim status are diagnostic "
        "certification-transfer rows only, not trained-planner comparison evidence.",
        "",
        "## Files",
        "",
        "- `summary.json`",
        "- `metadata.json`",
        "- `certification_gate_cells.csv`",
        "- `certification_transfer_matrix.csv`",
        "- `metric_deltas_by_model.csv`",
        "- `flip_cases.csv`",
        "- `claim_boundary.md`",
        "- `SHA256SUMS`",
    ]
    return "\n".join(lines) + "\n"


def _checksums(paths: Iterable[Path]) -> str:
    rows = [
        f"{_sha256_file(path)}  {path.name}" for path in sorted(paths, key=lambda item: item.name)
    ]
    return "\n".join(rows) + "\n"


def _repo_relative_path(path: str | Path) -> str:
    """Return ``path`` as a portable repo-relative POSIX string when possible.

    Evidence packets are committed and shared across machines and CI, so their
    provenance fields must never bake in absolute, username-bearing local paths
    such as ``/home/<user>/git/robot_sf_ll7.worktrees/.../config.yaml`` (issue
    #4324; same non-reproducibility defect as #4302). When ``path`` resolves
    inside this checkout it is rendered relative to :data:`REPO_ROOT`; otherwise
    it falls back to the bare file name so no absolute home-dir path can leak
    into the durable artifact. The pre-commit guard
    (``hooks/check_config_abs_paths.py``) enforces this fail-closed at commit
    time for both ``configs/**`` and ``docs/context/evidence/**``.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
